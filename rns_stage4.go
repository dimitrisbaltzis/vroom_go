// rns_stage4.go — Stage 4: Register-resident matvec + division-free reduction
//
// Combines two optimizations:
//
// #4 — Instruction interleaving (matvec kernel):
//   All accumulators stay in ZMM registers across the source loop.
//   6 independent VPMADD52 per iteration → fully pipelined on ICL.
//   Eliminates Go function-call overhead (63 calls → 1 kernel call).
//   Expected: step1 351ns → ~45ns (7-8x)
//
// #2 — Division-free modular reduction:
//   Replace mulmod (= bits.Div64 = DIVQ ~35 cycles) with:
//   • Shoup's method: 2× bits.Mul64 + 1 mul + 1 sub + 1 cond = ~10 cycles
//   • Barrett reduction: 1× bits.Mul64 + 1 mul + 1 sub + 1 cond = ~8 cycles
//   42 DIVQ → 21 Shoup + 21 Barrett + 21 Barrett
//   Expected: step2+4 322ns → ~170ns (~2x)
//
// Combined expected: Apply 719ns → ~260ns (2.7x per Apply)
//                    VROOM 1750ns → ~700ns (2.5x)

package vroom

import (
	"math/big"
	"math/bits"
)

// ============================================================================
// Division-free modular arithmetic for 52-bit moduli
// ============================================================================

// mulmodShoup52 computes (a * w) mod n using Shoup's precomputed quotient.
//
// Precompute: wPrime = floor(w * 2^64 / n)
// Requires:   a < 2^64, w < n < 2^52
// Cost:       2× MULQ + 1 MUL + 1 SUB + 1 CMOV ≈ 10 cycles (vs DIVQ ≈ 35)
//
//go:nosplit
func mulmodShoup52(a, w, wPrime, n uint64) uint64 {
	q, _ := bits.Mul64(a, wPrime) // q = ⌊a·w'/2^64⌋ (quotient estimate)
	_, awLo := bits.Mul64(a, w)   // awLo = (a·w) mod 2^64
	r := awLo - q*n               // r ∈ [0, 2n) — uint64 wrapping is correct
	if r >= n {
		r -= n
	}
	return r
}

// barrettReduce52 computes x mod n using Barrett reduction.
//
// Precompute: mu = floor(2^64 / n)
// Requires:   x < 2^64, n < 2^52 (error is at most 1)
// Cost:       1× MULQ + 1 MUL + 1 SUB + 2 CMOV ≈ 8 cycles
//
//go:nosplit
func barrettReduce52(x, n, mu uint64) uint64 {
	q, _ := bits.Mul64(x, mu)
	r := x - q*n
	if r >= n {
		r -= n
	}
	if r >= n { // second correction for edge cases
		r -= n
	}
	return r
}

// ============================================================================
// CRNSMatrixStage4 — CRNS matrix with division-free constants
// ============================================================================

// CRNSMatrixStage4 stores CRNS constants optimized for:
//   • AVX512IFMA matvec kernel (AFlat contiguous layout)
//   • Shoup modular multiply (Pow52Shoup, CShoup precomputed quotients)
//   • Barrett modular reduce (BarrettMu precomputed reciprocals)
type CRNSMatrixStage4 struct {
	// Matrix data (from Stage 3)
	APadded  [][]uint64 // [tFrom][padTTo] — for reference / fallback
	AFlat    []uint64   // [tFrom * padTTo] — contiguous row-major for asm kernel

	// k estimation (unchanged)
	F    []uint64 // [tFrom] low 64 bits of fixed-point F
	FHi  []uint64 // [tFrom] high bits of F
	Prec uint

	// Target moduli
	ToMod []uint64 // actual target moduli (unpadded)
	TTo   int      // actual target count
	PadTTo int     // padded to multiple of 8

	// Combine constants (step 2): reduce accHi*2^52 + accLo mod nj
	Pow52Mod   []uint64 // [padTTo] — (1<<52) % nj
	Pow52Shoup []uint64 // [padTTo] — ⌊Pow52Mod[j] * 2^64 / nj⌋ (Shoup quotient)
	BarrettMu  []uint64 // [padTTo] — ⌊2^64 / nj⌋ (Barrett reciprocal)

	// Correction constants (step 4): k * C[j] mod nj
	CPadded []uint64 // [padTTo] — correction vector
	CShoup  []uint64 // [padTTo] — ⌊C[j] * 2^64 / nj⌋ (Shoup quotient)
}

// NewCRNSMatrixStage4 builds the Stage 4 CRNS matrix from RNS bases.
func NewCRNSMatrixStage4(from, to *RNSBaseU64, y, z *big.Int) *CRNSMatrixStage4 {
	// Reuse Stage 2 precomputation for the core constants
	reg := PrecomputeCRNS(from, to, y, z)

	tFrom := len(from.Moduli)
	tTo := len(to.Moduli)
	padded := padTo8(tTo)

	twoPow64 := new(big.Int).Lsh(bigOne, 64)
	pow52 := new(big.Int).Lsh(bigOne, 52)

	mat := &CRNSMatrixStage4{
		APadded:    make([][]uint64, tFrom),
		AFlat:      make([]uint64, tFrom*padded),
		F:          reg.F,
		FHi:        reg.FHi,
		Prec:       reg.Prec,
		ToMod:      reg.ToMod,
		TTo:        tTo,
		PadTTo:     padded,
		Pow52Mod:   make([]uint64, padded),
		Pow52Shoup: make([]uint64, padded),
		BarrettMu:  make([]uint64, padded),
		CPadded:    make([]uint64, padded),
		CShoup:     make([]uint64, padded),
	}

	// Pad and flatten A matrix
	for i := 0; i < tFrom; i++ {
		mat.APadded[i] = make([]uint64, padded)
		copy(mat.APadded[i], reg.A[i])
		copy(mat.AFlat[i*padded:], mat.APadded[i])
	}

	// Copy and pad C
	copy(mat.CPadded, reg.C)

	// Precompute division-free constants for each target modulus
	for j, nj := range to.Moduli {
		njBig := new(big.Int).SetUint64(nj)

		// Pow52Mod[j] = 2^52 mod nj
		p52 := new(big.Int).Mod(pow52, njBig).Uint64()
		mat.Pow52Mod[j] = p52

		// Pow52Shoup[j] = floor(Pow52Mod[j] * 2^64 / nj)
		if p52 > 0 {
			shoup := new(big.Int).SetUint64(p52)
			shoup.Mul(shoup, twoPow64)
			shoup.Div(shoup, njBig)
			mat.Pow52Shoup[j] = shoup.Uint64()
		}

		// BarrettMu[j] = floor(2^64 / nj)
		mu := new(big.Int).Div(twoPow64, njBig)
		mat.BarrettMu[j] = mu.Uint64()

		// CShoup[j] = floor(C[j] * 2^64 / nj)
		cj := mat.CPadded[j]
		if cj > 0 {
			cs := new(big.Int).SetUint64(cj)
			cs.Mul(cs, twoPow64)
			cs.Div(cs, njBig)
			mat.CShoup[j] = cs.Uint64()
		}
	}

	return mat
}

// ApplyStage4 performs the CRNS base change using:
//   Step 1: AVX512 matvec kernel (register-resident accumulators)
//   Step 2: Shoup combine + Barrett reduce (division-free)
//   Step 3: 192-bit k estimation (scalar, unchanged)
//   Step 4: Shoup correction + conditional add (division-free)
func (m *CRNSMatrixStage4) ApplyStage4(r []uint64, out, accLo, accHi []uint64) {
	tFrom := len(r)
	nGroups := m.PadTTo / 8

	// ── Step 1: matrix-vector product via AVX512 kernel ──────────────
	switch nGroups {
	case 3:
		matvecAVX512_3g(&accLo[0], &accHi[0], &r[0], &m.AFlat[0], tFrom, m.PadTTo)
	case 6:
		matvecAVX512_6g(&accLo[0], &accHi[0], &r[0], &m.AFlat[0], tFrom, m.PadTTo)
	default:
		// Generic kernel requires pre-zeroed accumulators
		for i := 0; i < m.PadTTo; i++ {
			accLo[i] = 0
			accHi[i] = 0
		}
		matvecAVX512Gen(&accLo[0], &accHi[0], &r[0], &m.AFlat[0], tFrom, m.PadTTo)
	}

	// ── Step 2: combine hi*2^52 + lo mod nj ─────────────────────────
	// Replaces: mulmod(accHi[j], Pow52Mod[j], nj) + accLo[j] % nj
	// Old cost: 1 DIVQ per lane = ~35 cycles
	// New cost: 1 Shoup + 1 Barrett per lane = ~18 cycles
	for j := 0; j < m.TTo; j++ {
		nj := m.ToMod[j]
		// (accHi[j] * Pow52Mod[j]) mod nj via Shoup
		hiPart := mulmodShoup52(accHi[j], m.Pow52Mod[j], m.Pow52Shoup[j], nj)
		// (hiPart + accLo[j]) mod nj via Barrett
		sum := hiPart + accLo[j]
		out[j] = barrettReduce52(sum, nj, m.BarrettMu[j])
	}

	// ── Step 3: k estimation via 192-bit accumulator ─────────────────
	k := computeK192(r[:tFrom], m.F, m.FHi, m.Prec, tFrom)

	// ── Step 4: correction ───────────────────────────────────────────
	// Replaces: mulmod(k, C[j], nj) per lane
	// Old cost: 1 DIVQ per lane = ~35 cycles
	// New cost: 1 Shoup + 1 conditional subtract = ~12 cycles
	for j := 0; j < m.TTo; j++ {
		nj := m.ToMod[j]
		// (k * C[j]) mod nj via Shoup
		kc := mulmodShoup52(k, m.CPadded[j], m.CShoup[j], nj)
		// out[j] + kc mod nj — both < nj, sum < 2*nj
		sum := out[j] + kc
		if sum >= nj {
			sum -= nj
		}
		out[j] = sum
	}
}

// ============================================================================
// Stage 4 parameters and workspace
// ============================================================================

// MontParamsStage4 holds everything needed for VROOM with Stage 4 optimizations.
type MontParamsStage4 struct {
	P     *big.Int
	BaseM *RNSBaseU64
	BaseN *RNSBaseU64
	T     *big.Int
	TInv  *big.Int
	CRNS1 *CRNSMatrixStage4 // M→N
	CRNS2 *CRNSMatrixStage4 // N→M
	W     *WorkspaceStage4
}

// WorkspaceStage4 holds pre-allocated buffers for zero-allocation runtime.
type WorkspaceStage4 struct {
	// M-base scratch
	qM         []uint64
	rM         []uint64
	crns2AccLo []uint64
	crns2AccHi []uint64
	crns2Out   []uint64
	// N-base scratch
	abN        []uint64
	crns1AccLo []uint64
	crns1AccHi []uint64
	crns1Out   []uint64
	rN         []uint64
}

// SetupRNSParamsStage4 builds Stage 4 params with 52-bit moduli.
// Uses generate52BitPrimes for primes in (2^51, 2^52) matching VPMADD52.
func SetupRNSParamsStage4(p *big.Int) *MontParamsStage4 {
	pBits := p.BitLen()
	numModM := computeModuliCount(pBits, 52, 4) // M > 9p
	numModN := computeModuliCount(pBits, 52, 3) // N > 6p

	all := generate52BitPrimes(numModM+numModN, p)
	return NewMontParamsStage4(p, all[:numModM], all[numModM:])
}

// NewMontParamsStage4 builds Stage 4 params from explicit moduli.
func NewMontParamsStage4(p *big.Int, mModuli, nModuli []uint64) *MontParamsStage4 {
	baseM := NewRNSBaseU64(mModuli)
	baseN := NewRNSBaseU64(nModuli)
	M := baseM.Product
	N := baseN.Product
	MN := new(big.Int).Mul(M, N)

	// Verify bounds
	nineP := new(big.Int).Mul(big.NewInt(9), p)
	sixP := new(big.Int).Mul(big.NewInt(6), p)
	if M.Cmp(nineP) <= 0 {
		panic("M must be > 9p")
	}
	if N.Cmp(sixP) <= 0 {
		panic("N must be > 6p")
	}

	// T ≡ 1 (mod M), T ≡ M^{-1} (mod N) — same construction as Stage 2/3
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
	crns1 := NewCRNSMatrixStage4(baseM, baseN, negPInvM, pMInv2N)

	// CRNS2: from=N, to=M, y=(M mod N), z=1
	MmodN := new(big.Int).Mod(M, N)
	crns2 := NewCRNSMatrixStage4(baseN, baseM, MmodN, bigOne)

	// Workspace: all buffers padded to max of padded sizes
	padM := crns2.PadTTo
	padN := crns1.PadTTo
	tM := len(mModuli)
	tN := len(nModuli)
	if padM < tM {
		padM = tM
	}
	if padN < tN {
		padN = tN
	}

	w := &WorkspaceStage4{
		qM:         make([]uint64, padM),
		rM:         make([]uint64, padM),
		crns2AccLo: make([]uint64, padM),
		crns2AccHi: make([]uint64, padM),
		crns2Out:   make([]uint64, padM),
		abN:        make([]uint64, padN),
		crns1AccLo: make([]uint64, padN),
		crns1AccHi: make([]uint64, padN),
		crns1Out:   make([]uint64, padN),
		rN:         make([]uint64, padN),
	}

	return &MontParamsStage4{
		P: new(big.Int).Set(p), BaseM: baseM, BaseN: baseN,
		T: T, TInv: TInv,
		CRNS1: crns1, CRNS2: crns2,
		W: w,
	}
}

// ============================================================================
// VROOMStage4 — Algorithm 2 with both optimizations
// ============================================================================

// VROOMStage4 performs VROOM modular multiplication using:
//   • AVX512IFMA matvec kernel (register-resident accumulators)
//   • Division-free modular reduction (Shoup + Barrett)
//
// Zero big.Int allocations, zero DIVQ instructions in the hot path.
func VROOMStage4(aM, aN, bM, bN []uint64, p *MontParamsStage4) ([]uint64, []uint64) {
	w := p.W
	tM := len(p.BaseM.Moduli)
	tN := len(p.BaseN.Moduli)

	// Step 2: q_M = a_M · b_M mod m_i  [elementwise, Shoup]
	for i := 0; i < tM; i++ {
		w.qM[i] = mulmod(aM[i], bM[i], p.BaseM.Moduli[i])
	}

	// Step 3a: a_N · b_N  [elementwise, Shoup]
	for j := 0; j < tN; j++ {
		w.abN[j] = mulmod(aN[j], bN[j], p.BaseN.Moduli[j])
	}

	// Step 3b: CRNS1(q_M) via Stage 4 Apply
	p.CRNS1.ApplyStage4(w.qM[:tM], w.crns1Out, w.crns1AccLo, w.crns1AccHi)

	// Step 3c: r_N = a_N·b_N + CRNS1(q_M)  [elementwise add]
	for j := 0; j < tN; j++ {
		w.rN[j] = addmod(w.abN[j], w.crns1Out[j], p.BaseN.Moduli[j])
	}

	// Step 4: r_M = CRNS2(r_N) via Stage 4 Apply
	p.CRNS2.ApplyStage4(w.rN[:tN], w.rM, w.crns2AccLo, w.crns2AccHi)

	return w.rM[:tM], w.rN[:tN]
}

// ============================================================================
// Encoding / decoding (same interface)
// ============================================================================

// ToVROOMEncodingStage4 encodes a value for VROOM multiplication.
func ToVROOMEncodingStage4(a *big.Int, p *MontParamsStage4) ([]uint64, []uint64) {
	v := new(big.Int).Mul(a, p.BaseM.Product)
	v.Mod(v, p.P)
	MN := new(big.Int).Mul(p.BaseM.Product, p.BaseN.Product)
	aMN := new(big.Int).Mul(v, p.T)
	aMN.Mod(aMN, MN)
	return p.BaseM.Encode(aMN), p.BaseN.Encode(aMN)
}

// FromVROOMEncodingStage4 decodes a VROOM result back to a big.Int.
func FromVROOMEncodingStage4(rM []uint64, p *MontParamsStage4) *big.Int {
	rPrime := p.BaseM.Decode(rM)
	MInvP := new(big.Int).ModInverse(p.BaseM.Product, p.P)
	result := new(big.Int).Mul(rPrime, MInvP)
	return result.Mod(result, p.P)
}
