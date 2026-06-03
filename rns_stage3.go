// rns_stage3.go — Stage 3: CRNS matrix-vector product via AVX512IFMA
//
// The O(t²) CRNS inner loop is vectorized: broadcastMulAcc52 processes
// 8 target lanes per AVX512 instruction pair (VPMADD52LUQ + VPMADD52HUQ).
// The O(t) parts (k estimation, correction, elementwise ops) remain scalar.

package main

import (
	"math/big"
	"math/bits"
)

// --------------------------------------------------------------------------
// AVX512-friendly CRNS matrix
// --------------------------------------------------------------------------

func padTo8(n int) int {
	return ((n + 7) / 8) * 8
}

// CRNSMatrixAVX512 stores the CRNS constants in a layout optimized for
// processing 8 target lanes at a time with broadcastMulAcc52.
type CRNSMatrixAVX512 struct {
	APadded  [][]uint64 // [tFrom][paddedTTo] — rows padded to multiple of 8
	F        []uint64   // [tFrom] — fixed-point k estimation vector
	CPadded  []uint64   // [paddedTTo] — correction vector, padded
	Pow52Mod []uint64   // [paddedTTo] — (1<<52) % n_j for combining hi+lo
	Prec     uint
	ToMod    []uint64 // actual target moduli (unpadded)
	TTo      int      // actual target count
	PadTTo   int      // padded to multiple of 8
}

// NewCRNSMatrixAVX512 builds an AVX512-friendly CRNS matrix from the base params.
func NewCRNSMatrixAVX512(from, to *RNSBaseU64, y, z *big.Int) *CRNSMatrixAVX512 {
	// Build regular matrix first (reuse Stage 2 precomputation)
	reg := PrecomputeCRNS(from, to, y, z)

	tFrom := len(from.Moduli)
	tTo := len(to.Moduli)
	padded := padTo8(tTo)

	mat := &CRNSMatrixAVX512{
		APadded:  make([][]uint64, tFrom),
		F:        reg.F,
		CPadded:  make([]uint64, padded),
		Pow52Mod: make([]uint64, padded),
		Prec:     reg.Prec,
		ToMod:    reg.ToMod,
		TTo:      tTo,
		PadTTo:   padded,
	}

	// Pad A rows to multiple of 8
	for i := 0; i < tFrom; i++ {
		mat.APadded[i] = make([]uint64, padded)
		copy(mat.APadded[i], reg.A[i])
	}

	// Pad C
	copy(mat.CPadded, reg.C)

	// Precompute (1<<52) % n_j for each target modulus
	pow52 := new(big.Int).Lsh(bigOne, 52)
	for j, nj := range to.Moduli {
		mat.Pow52Mod[j] = new(big.Int).Mod(pow52, new(big.Int).SetUint64(nj)).Uint64()
	}

	return mat
}

// ApplyAVX512 performs the CRNS base change using AVX512IFMA.
//   Step 1: matrix-vector product via broadcastMulAcc52 (8 lanes at a time)
//   Step 2: combine hi*2^52 + lo, reduce mod n_j
//   Step 3: k estimation (scalar)
//   Step 4: correction
func (m *CRNSMatrixAVX512) ApplyAVX512(r []uint64, out, accLo, accHi []uint64) {
	tFrom := len(r)

	// Step 1: AVX512 matrix-vector product
	// Zero accumulators
	for i := 0; i < m.PadTTo; i++ {
		accLo[i] = 0
		accHi[i] = 0
	}
	// For each source residue, accumulate into all target groups
	for i := 0; i < tFrom; i++ {
		for g := 0; g < m.PadTTo; g += 8 {
			broadcastMulAcc52(&accLo[g], &accHi[g], r[i], &m.APadded[i][g])
		}
	}

	// Step 2: combine hi*2^52 + lo → reduce mod n_j
	for j := 0; j < m.TTo; j++ {
		nj := m.ToMod[j]
		hiPart := mulmod(accHi[j], m.Pow52Mod[j], nj)
		out[j] = (hiPart + accLo[j]) % nj
	}

	// Step 3: k estimation via fixed-point dot product (scalar)
	var sumHi, sumLo uint64
	for i := 0; i < tFrom; i++ {
		hi, lo := bits.Mul64(r[i], m.F[i])
		var carry uint64
		sumLo, carry = bits.Add64(sumLo, lo, 0)
		sumHi, _ = bits.Add64(sumHi, hi, carry)
	}
	var k uint64
	if m.Prec < 64 {
		k = (sumHi << (64 - m.Prec)) | (sumLo >> m.Prec)
	} else {
		k = sumHi >> (m.Prec - 64)
	}

	// Step 4: apply correction
	for j := 0; j < m.TTo; j++ {
		nj := m.ToMod[j]
		kc := mulmod(k, m.CPadded[j], nj)
		out[j] = addmod(out[j], kc, nj)
	}
}

// --------------------------------------------------------------------------
// Stage 3 parameters and workspace
// --------------------------------------------------------------------------

type MontParamsStage3 struct {
	P     *big.Int
	BaseM *RNSBaseU64
	BaseN *RNSBaseU64
	T     *big.Int
	TInv  *big.Int
	CRNS1 *CRNSMatrixAVX512 // M→N
	CRNS2 *CRNSMatrixAVX512 // N→M
	W     *WorkspaceStage3
}

type WorkspaceStage3 struct {
	// M-base (padded)
	qM         []uint64
	rM         []uint64
	crns2AccLo []uint64
	crns2AccHi []uint64
	// N-base (padded)
	abN        []uint64
	crns1AccLo []uint64
	crns1AccHi []uint64
	crns1R     []uint64
	rN         []uint64
}

func SetupRNSParamsStage3(p *big.Int) *MontParamsStage3 {
	// Generate moduli (same as previous stages)
	pBits := p.BitLen()
	modBits := 32
	numModM := (pBits+4+modBits-1)/modBits + 2
	numModN := (pBits+3+modBits-1)/modBits + 2
	if numModM < 2 {
		numModM = 2
	}
	if numModN < 2 {
		numModN = 2
	}

	allBig := generateDistinctPrimes(numModM+numModN, modBits, p)
	mModuli := make([]uint64, numModM)
	nModuli := make([]uint64, numModN)
	for i := 0; i < numModM; i++ {
		mModuli[i] = allBig[i].Uint64()
	}
	for i := 0; i < numModN; i++ {
		nModuli[i] = allBig[numModM+i].Uint64()
	}

	baseM := NewRNSBaseU64(mModuli)
	baseN := NewRNSBaseU64(nModuli)
	M := baseM.Product
	N := baseN.Product
	MN := new(big.Int).Mul(M, N)

	// Verify
	nineP := new(big.Int).Mul(big.NewInt(9), p)
	sixP := new(big.Int).Mul(big.NewInt(6), p)
	if M.Cmp(nineP) <= 0 {
		panic("M must be > 9p")
	}
	if N.Cmp(sixP) <= 0 {
		panic("N must be > 6p")
	}

	// T
	NInvM := new(big.Int).ModInverse(N, M)
	MInvN := new(big.Int).ModInverse(M, N)
	MInvNSq := new(big.Int).Mul(MInvN, MInvN)
	T := new(big.Int).Add(
		new(big.Int).Mul(N, NInvM),
		new(big.Int).Mul(M, MInvNSq),
	)
	T.Mod(T, MN)
	TInv := new(big.Int).ModInverse(T, MN)

	// CRNS1: from=M, to=N, y=(-p^{-1} mod M), z=(p·M^{-2} mod N)
	negPInvM := new(big.Int).ModInverse(p, M)
	negPInvM.Sub(M, negPInvM)
	MInv2N := new(big.Int).Mul(MInvN, MInvN)
	MInv2N.Mod(MInv2N, N)
	pMInv2N := new(big.Int).Mul(p, MInv2N)
	pMInv2N.Mod(pMInv2N, N)
	crns1 := NewCRNSMatrixAVX512(baseM, baseN, negPInvM, pMInv2N)

	// CRNS2: from=N, to=M, y=(M mod N), z=1
	MmodN := new(big.Int).Mod(M, N)
	crns2 := NewCRNSMatrixAVX512(baseN, baseM, MmodN, bigOne)

	// Workspace with padded buffers
	padM := crns2.PadTTo // padded M-base size (for CRNS2 output)
	padN := crns1.PadTTo // padded N-base size (for CRNS1 output)
	tM := len(mModuli)
	tN := len(nModuli)
	// Use the larger of actual or padded for safety
	if padM < tM {
		padM = tM
	}
	if padN < tN {
		padN = tN
	}

	w := &WorkspaceStage3{
		qM:         make([]uint64, padM),
		rM:         make([]uint64, padM),
		crns2AccLo: make([]uint64, padM),
		crns2AccHi: make([]uint64, padM),
		abN:        make([]uint64, padN),
		crns1AccLo: make([]uint64, padN),
		crns1AccHi: make([]uint64, padN),
		crns1R:     make([]uint64, padN),
		rN:         make([]uint64, padN),
	}

	return &MontParamsStage3{
		P: new(big.Int).Set(p), BaseM: baseM, BaseN: baseN,
		T: T, TInv: TInv,
		CRNS1: crns1, CRNS2: crns2,
		W: w,
	}
}

// --------------------------------------------------------------------------
// VROOMStage3 — full pipeline with AVX512 CRNS
// --------------------------------------------------------------------------

func VROOMStage3(aM, aN, bM, bN []uint64, p *MontParamsStage3) ([]uint64, []uint64) {
	w := p.W
	tM := len(p.BaseM.Moduli)
	tN := len(p.BaseN.Moduli)

	// Step 2: q_M = a_M · b_M % M  [elementwise, scalar]
	for i := 0; i < tM; i++ {
		w.qM[i] = mulmod(aM[i], bM[i], p.BaseM.Moduli[i])
	}

	// Step 3a: a_N · b_N  [elementwise, scalar]
	for j := 0; j < tN; j++ {
		w.abN[j] = mulmod(aN[j], bN[j], p.BaseN.Moduli[j])
	}

	// Step 3b: CRNS1(q_M) via AVX512
	p.CRNS1.ApplyAVX512(w.qM[:tM], w.crns1R, w.crns1AccLo, w.crns1AccHi)

	// Step 3c: r_N = a_N·b_N + CRNS1(q_M)  [elementwise add]
	for j := 0; j < tN; j++ {
		w.rN[j] = addmod(w.abN[j], w.crns1R[j], p.BaseN.Moduli[j])
	}

	// Step 4: r_M = CRNS2(r_N) via AVX512
	p.CRNS2.ApplyAVX512(w.rN[:tN], w.rM, w.crns2AccLo, w.crns2AccHi)

	return w.rM[:tM], w.rN[:tN]
}

// --------------------------------------------------------------------------
// Encoding / decoding
// --------------------------------------------------------------------------

func ToVROOMEncodingStage3(a *big.Int, p *MontParamsStage3) ([]uint64, []uint64) {
	v := new(big.Int).Mul(a, p.BaseM.Product)
	v.Mod(v, p.P)
	MN := new(big.Int).Mul(p.BaseM.Product, p.BaseN.Product)
	aMN := new(big.Int).Mul(v, p.T)
	aMN.Mod(aMN, MN)
	return p.BaseM.Encode(aMN), p.BaseN.Encode(aMN)
}

func FromVROOMEncodingStage3(rM []uint64, p *MontParamsStage3) *big.Int {
	rPrime := p.BaseM.Decode(rM)
	MInvP := new(big.Int).ModInverse(p.BaseM.Product, p.P)
	result := new(big.Int).Mul(rPrime, MInvP)
	return result.Mod(result, p.P)
}
