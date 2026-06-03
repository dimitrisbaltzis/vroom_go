// rns_stage2.go — Stage 2: CRNS via matrix-vector product
//
// Replaces the CRT reconstruction in CRNS with the paper's Appendix A approach:
//   1. Precompute matrix A[i][j], fixed-point vector f[i], correction vector c[j]
//   2. At runtime: matrix-vector multiply + dot product + correction
//   3. Zero big.Int allocations in the hot path
//
// This is where most of the speedup comes from: CRNS was ~75% of runtime.

package main

import (
	"math/big"
	"math/bits"
)

// --------------------------------------------------------------------------
// CRNSMatrixU64 — precomputed constants for fast base change
// --------------------------------------------------------------------------

// CRNSMatrixU64 holds precomputed constants for a CRNS^{from*y}_{to*z} operation.
// All runtime operations use only uint64 arithmetic.
type CRNSMatrixU64 struct {
	A     [][]uint64 // [fromSize][toSize] matrix
	F     []uint64   // [fromSize] fixed-point approximation vector
	C     []uint64   // [toSize] correction vector
	Prec  uint       // precision bits for fixed-point k estimation
	ToMod []uint64   // target moduli (for runtime reduction)
}

// PrecomputeCRNS builds the CRNS matrix for CRNS^{from*y}_{to*z}.
// All heavy math is done here with big.Int; only uint64 values are stored.
func PrecomputeCRNS(from, to *RNSBaseU64, y, z *big.Int) *CRNSMatrixU64 {
	tFrom := len(from.Moduli)
	tTo := len(to.Moduli)
	fromProd := from.Product

	// Precision for fixed-point k estimation.
	// Paper requires u > w + log2(t) + 1. We use generous margin
	// to avoid edge-case rounding errors with 32-bit moduli.
	logT := uint(bits.Len(uint(tFrom)))
	prec := uint(52 + logT)
	// Verify F values will fit in uint64 (F < 2^prec + 1)
	if prec > 62 {
		prec = 62
	}

	mat := &CRNSMatrixU64{
		A:     make([][]uint64, tFrom),
		F:     make([]uint64, tFrom),
		C:     make([]uint64, tTo),
		Prec:  prec,
		ToMod: to.Moduli,
	}

	// For each source modulus m_i, compute:
	//   ICRT_i = (M/m_i) * ((M/m_i)^{-1} mod m_i)
	//   ICRT_i_y = (ICRT_i * y) mod M
	//   A[i][j] = (ICRT_i_y * z) mod n_j
	//   f[i]    = ceil(2^prec * ICRT_i_y / M)
	twoPrec := new(big.Int).Lsh(bigOne, prec)

	for i, mi := range from.Moduli {
		miBig := new(big.Int).SetUint64(mi)

		// ICRT_i = (M/m_i) * ((M/m_i)^{-1} mod m_i)
		Mi := new(big.Int).Div(fromProd, miBig)
		MiInv := new(big.Int).ModInverse(Mi, miBig)
		ICRTi := new(big.Int).Mul(Mi, MiInv)

		// ICRT_i_y = (ICRT_i * y) mod M
		ICRTiY := new(big.Int).Mul(ICRTi, y)
		ICRTiY.Mod(ICRTiY, fromProd)

		// A[i][j] = (ICRT_i_y * z) mod n_j
		ICRTiYZ := new(big.Int).Mul(ICRTiY, z)
		mat.A[i] = make([]uint64, tTo)
		for j, nj := range to.Moduli {
			njBig := new(big.Int).SetUint64(nj)
			mat.A[i][j] = new(big.Int).Mod(ICRTiYZ, njBig).Uint64()
		}

		// f[i] = ceil(2^prec * ICRT_i_y / M)
		fi := new(big.Int).Mul(twoPrec, ICRTiY)
		fi.Add(fi, new(big.Int).Sub(fromProd, bigOne)) // add (M-1) for ceiling
		fi.Div(fi, fromProd)
		mat.F[i] = fi.Uint64()
	}

	// c[j] = (-M * z) mod n_j
	negMZ := new(big.Int).Neg(fromProd)
	negMZ.Mul(negMZ, z)
	for j, nj := range to.Moduli {
		njBig := new(big.Int).SetUint64(nj)
		cj := new(big.Int).Mod(negMZ, njBig)
		if cj.Sign() < 0 {
			cj.Add(cj, njBig)
		}
		mat.C[j] = cj.Uint64()
	}

	return mat
}

// Apply performs the CRNS base change at runtime using only uint64 arithmetic.
//
// Algorithm (Appendix A of the paper):
//   Step 1: a[j] = Σ_i r[i] * A[i][j]  mod n_j    (matrix-vector product)
//   Step 2: k    = ⌊ Σ_i r[i] * f[i] / 2^prec ⌋   (overflow estimation)
//   Step 3: result[j] = a[j] + k * c[j]  mod n_j   (correction)
func (m *CRNSMatrixU64) Apply(r []uint64) []uint64 {
	tFrom := len(r)
	tTo := len(m.ToMod)

	// Step 1: matrix-vector product (the O(t²) core)
	a := make([]uint64, tTo)
	for j := 0; j < tTo; j++ {
		nj := m.ToMod[j]
		var acc uint64
		for i := 0; i < tFrom; i++ {
			prod := mulmod(r[i], m.A[i][j], nj)
			acc = addmod(acc, prod, nj)
		}
		a[j] = acc
	}

	// Step 2: compute k via fixed-point dot product with 128-bit accumulator
	var sumHi, sumLo uint64
	for i := 0; i < tFrom; i++ {
		hi, lo := bits.Mul64(r[i], m.F[i])
		var carry uint64
		sumLo, carry = bits.Add64(sumLo, lo, 0)
		sumHi, _ = bits.Add64(sumHi, hi, carry)
	}
	// k = (sumHi:sumLo) >> prec
	var k uint64
	if m.Prec < 64 {
		k = (sumHi << (64 - m.Prec)) | (sumLo >> m.Prec)
	} else {
		k = sumHi >> (m.Prec - 64)
	}

	// Step 3: apply correction
	result := make([]uint64, tTo)
	for j := 0; j < tTo; j++ {
		nj := m.ToMod[j]
		kc := mulmod(k, m.C[j], nj)
		result[j] = addmod(a[j], kc, nj)
	}

	return result
}

// --------------------------------------------------------------------------
// MontParamsStage2 — params with precomputed CRNS matrices
// --------------------------------------------------------------------------

// MontParamsStage2 holds everything needed for VROOM with fast CRNS.
type MontParamsStage2 struct {
	P     *big.Int
	BaseM *RNSBaseU64
	BaseN *RNSBaseU64
	T     *big.Int
	TInv  *big.Int
	CRNS1 *CRNSMatrixU64 // CRNS^{M*(-p^{-1})}_{N*(p·M^{-2})} for step 3
	CRNS2 *CRNSMatrixU64 // CRNS^{N*M}_{M*1}                   for step 4
}

// NewMontParamsStage2 builds Stage 2 params.
func NewMontParamsStage2(p *big.Int, mModuli, nModuli []uint64) *MontParamsStage2 {
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

	// T ≡ 1 (mod M), T ≡ M^{-1} (mod N)
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

	crns1 := PrecomputeCRNS(baseM, baseN, negPInvM, pMInv2N)

	// CRNS2: from=N, to=M, y=(M mod N), z=1
	MmodN := new(big.Int).Mod(M, N)
	crns2 := PrecomputeCRNS(baseN, baseM, MmodN, bigOne)

	return &MontParamsStage2{
		P: new(big.Int).Set(p), BaseM: baseM, BaseN: baseN,
		T: T, TInv: TInv,
		CRNS1: crns1, CRNS2: crns2,
	}
}

// --------------------------------------------------------------------------
// VROOMStage2 — Algorithm 2 with fast CRNS (zero big.Int at runtime)
// --------------------------------------------------------------------------

// VROOMStage2 performs VROOM modular multiplication.
// The entire hot path uses only uint64 and bits.Mul64/bits.Div64.
func VROOMStage2(aM, aN, bM, bN []uint64, p *MontParamsStage2) ([]uint64, []uint64) {
	// Step 2: q_M = a_M · b_M % M  [elementwise]
	qM := rnsMulU64(aM, bM, p.BaseM.Moduli)

	// Step 3: r_N = (a_N · b_N + CRNS1(q_M)) % N  [elementwise]
	abN := rnsMulU64(aN, bN, p.BaseN.Moduli)
	crnsResult := p.CRNS1.Apply(qM)
	rN := rnsAddU64(abN, crnsResult, p.BaseN.Moduli)

	// Step 4: r_M = CRNS2(r_N)
	rM := p.CRNS2.Apply(rN)

	return rM, rN
}

// --------------------------------------------------------------------------
// Encoding / decoding (same interface, using Stage 2 params)
// --------------------------------------------------------------------------

func ToVROOMEncodingStage2(a *big.Int, p *MontParamsStage2) ([]uint64, []uint64) {
	v := new(big.Int).Mul(a, p.BaseM.Product)
	v.Mod(v, p.P)
	MN := new(big.Int).Mul(p.BaseM.Product, p.BaseN.Product)
	aMN := new(big.Int).Mul(v, p.T)
	aMN.Mod(aMN, MN)
	return p.BaseM.Encode(aMN), p.BaseN.Encode(aMN)
}

func FromVROOMEncodingStage2(rM []uint64, p *MontParamsStage2) *big.Int {
	rPrime := p.BaseM.Decode(rM)
	MInvP := new(big.Int).ModInverse(p.BaseM.Product, p.P)
	result := new(big.Int).Mul(rPrime, MInvP)
	return result.Mod(result, p.P)
}

// --------------------------------------------------------------------------
// Setup helper
// --------------------------------------------------------------------------

func SetupRNSParamsStage2(p *big.Int) *MontParamsStage2 {
	pBits := p.BitLen()
	modBits := 32
	numModM := (pBits+4+modBits-1)/modBits + 1
	numModN := (pBits+3+modBits-1)/modBits + 1
	if numModM < 2 {
		numModM = 2
	}
	if numModN < 2 {
		numModN = 2
	}
	numModM++
	numModN++

	allBig := generateDistinctPrimes(numModM+numModN, modBits, p)
	mModuli := make([]uint64, numModM)
	nModuli := make([]uint64, numModN)
	for i := 0; i < numModM; i++ {
		mModuli[i] = allBig[i].Uint64()
	}
	for i := 0; i < numModN; i++ {
		nModuli[i] = allBig[numModM+i].Uint64()
	}

	return NewMontParamsStage2(p, mModuli, nModuli)
}
