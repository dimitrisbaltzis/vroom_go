// rns_fast.go — Stage 1: native uint64 residue arithmetic
//
// Key difference from rns.go: residues and precomputed constants are []uint64,
// not []*big.Int. All elementwise operations use native Go integer arithmetic.
// math/big is used only during setup (precomputation) and for CRNS base-change
// (that is Stage 2).

package vroom

import (
	"math/big"
	"math/bits"
)

// --------------------------------------------------------------------------
// Fast RNS base — moduli and residues as uint64
// --------------------------------------------------------------------------

// RNSBaseU64 is like RNSBase but with uint64 moduli.
// math/big is kept only for the product (used in setup and CRNS).
type RNSBaseU64 struct {
	Moduli  []uint64
	Product *big.Int // still needed for CRNS and precomputation
}

// NewRNSBaseU64 builds an RNS base from uint64 moduli.
func NewRNSBaseU64(moduli []uint64) *RNSBaseU64 {
	prod := new(big.Int).SetUint64(1)
	for _, m := range moduli {
		prod.Mul(prod, new(big.Int).SetUint64(m))
	}
	return &RNSBaseU64{Moduli: moduli, Product: prod}
}

// Encode converts a big.Int to []uint64 residues.
func (b *RNSBaseU64) Encode(x *big.Int) []uint64 {
	res := make([]uint64, len(b.Moduli))
	for i, m := range b.Moduli {
		res[i] = new(big.Int).Mod(x, new(big.Int).SetUint64(m)).Uint64()
	}
	return res
}

// Decode reconstructs x from uint64 residues using CRT (returns *big.Int).
func (b *RNSBaseU64) Decode(residues []uint64) *big.Int {
	result := new(big.Int)
	for i, m := range b.Moduli {
		mBig := new(big.Int).SetUint64(m)
		Mi := new(big.Int).Div(b.Product, mBig)
		MiInv := new(big.Int).ModInverse(Mi, mBig)
		term := new(big.Int).SetUint64(residues[i])
		term.Mul(term, Mi)
		term.Mul(term, MiInv)
		result.Add(result, term)
	}
	return result.Mod(result, b.Product)
}

// --------------------------------------------------------------------------
// Precomputed parameters (all constants as uint64)
// --------------------------------------------------------------------------

// MontParamsU64 mirrors MontParams but stores all constants as uint64.
type MontParamsU64 struct {
	P     *big.Int
	BaseM *RNSBaseU64
	BaseN *RNSBaseU64

	// Algorithm 1 constants
	NegPInvM []uint64 // (-p^{-1}) mod m_i
	PModN    []uint64 // p mod n_j
	MInvN    []uint64 // M^{-1} mod n_j

	// VROOM constants
	T    *big.Int
	TInv *big.Int

	// CRNS1: CRNS^{M*(-p^{-1})}_{N*(p*M^{-2})}
	CRNS1Pre  []uint64 // (-p^{-1}) mod m_i  (same as NegPInvM)
	CRNS1Post []uint64 // (p * M^{-2}) mod n_j

	// CRNS2: CRNS^{N*M}_{M*1}
	CRNS2Pre  []uint64 // M mod n_j
	CRNS2Post []uint64 // 1 for each m_i
}

// NewMontParamsU64 builds fast params from uint64 moduli slices.
func NewMontParamsU64(p *big.Int, mModuli, nModuli []uint64) *MontParamsU64 {
	baseM := NewRNSBaseU64(mModuli)
	baseN := NewRNSBaseU64(nModuli)
	M := baseM.Product
	N := baseN.Product

	nineP := new(big.Int).Mul(big.NewInt(9), p)
	sixP := new(big.Int).Mul(big.NewInt(6), p)
	if M.Cmp(nineP) <= 0 {
		panic("M must be > 9p")
	}
	if N.Cmp(sixP) <= 0 {
		panic("N must be > 6p")
	}

	params := &MontParamsU64{P: new(big.Int).Set(p), BaseM: baseM, BaseN: baseN}

	toU64 := func(x *big.Int) uint64 { return x.Uint64() }

	// (-p^{-1}) mod m_i
	params.NegPInvM = make([]uint64, len(mModuli))
	for i, m := range mModuli {
		mBig := new(big.Int).SetUint64(m)
		pInv := new(big.Int).ModInverse(p, mBig)
		params.NegPInvM[i] = toU64(new(big.Int).Sub(mBig, pInv))
	}

	// p mod n_j
	params.PModN = make([]uint64, len(nModuli))
	for j, n := range nModuli {
		params.PModN[j] = toU64(new(big.Int).Mod(p, new(big.Int).SetUint64(n)))
	}

	// M^{-1} mod n_j
	params.MInvN = make([]uint64, len(nModuli))
	for j, n := range nModuli {
		params.MInvN[j] = toU64(new(big.Int).ModInverse(M, new(big.Int).SetUint64(n)))
	}

	// T: T ≡ 1 (mod M), T ≡ M^{-1} (mod N)
	MN := new(big.Int).Mul(M, N)
	NInvM := new(big.Int).ModInverse(N, M)
	MInvN := new(big.Int).ModInverse(M, N)
	MInvNSq := new(big.Int).Mul(MInvN, MInvN)
	params.T = new(big.Int).Add(
		new(big.Int).Mul(N, NInvM),
		new(big.Int).Mul(M, MInvNSq),
	)
	params.T.Mod(params.T, MN)
	params.TInv = new(big.Int).ModInverse(params.T, MN)

	// CRNS1 post: (p * M^{-2}) mod n_j
	MInv2 := new(big.Int).Mul(MInvN, MInvN)
	MInv2.Mod(MInv2, N)
	pMInv2 := new(big.Int).Mul(p, MInv2)
	pMInv2.Mod(pMInv2, N)
	params.CRNS1Pre = params.NegPInvM
	params.CRNS1Post = make([]uint64, len(nModuli))
	for j, n := range nModuli {
		params.CRNS1Post[j] = toU64(new(big.Int).Mod(pMInv2, new(big.Int).SetUint64(n)))
	}

	// CRNS2 pre: M mod n_j
	params.CRNS2Pre = make([]uint64, len(nModuli))
	for j, n := range nModuli {
		params.CRNS2Pre[j] = toU64(new(big.Int).Mod(M, new(big.Int).SetUint64(n)))
	}
	params.CRNS2Post = make([]uint64, len(mModuli))
	for i := range mModuli {
		params.CRNS2Post[i] = 1
	}

	return params
}

// --------------------------------------------------------------------------
// Elementwise uint64 operations — zero allocations
// --------------------------------------------------------------------------

// mulmod computes a * b mod m using 128-bit intermediate.
// Works for any a, b < m ≤ 2^32 (fits in uint64 product).
// For moduli up to 32 bits: a*b < 2^64, so hi=0 always.
// We keep the full 128-bit path for correctness with larger moduli.
func mulmod(a, b, m uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	if hi == 0 {
		return lo % m
	}
	// (hi:lo) % m via bits.Div64 — requires hi%m < m (always true)
	_, rem := bits.Div64(hi%m, lo, m)
	return rem
}

// addmod computes (a + b) mod m with no overflow.
func addmod(a, b, m uint64) uint64 {
	s := a + b
	if s >= m {
		s -= m
	}
	return s
}

// rnsMulU64 multiplies two residue vectors elementwise.
func rnsMulU64(a, b []uint64, moduli []uint64) []uint64 {
	r := make([]uint64, len(a))
	for i, m := range moduli {
		r[i] = mulmod(a[i], b[i], m)
	}
	return r
}

// rnsAddU64 adds two residue vectors elementwise.
func rnsAddU64(a, b []uint64, moduli []uint64) []uint64 {
	r := make([]uint64, len(a))
	for i, m := range moduli {
		r[i] = addmod(a[i], b[i], m)
	}
	return r
}

// rnsMulScalarU64 multiplies a residue vector by a scalar vector elementwise.
func rnsMulScalarU64(a, scalar []uint64, moduli []uint64) []uint64 {
	r := make([]uint64, len(a))
	for i, m := range moduli {
		r[i] = mulmod(a[i], scalar[i], m)
	}
	return r
}

// --------------------------------------------------------------------------
// Fast CRNS — uint64 residues in, uint64 residues out
// Stage 1: still uses big.Int CRT internally (Stage 2 will remove this)
// --------------------------------------------------------------------------

func crnsU64(residues []uint64, from, to *RNSBaseU64) []uint64 {
	x := from.Decode(residues)
	return to.Encode(x)
}

func crnsWithPrePostU64(residues []uint64, from, to *RNSBaseU64,
	yPre, zPost []uint64) []uint64 {

	// Premultiply
	premul := make([]uint64, len(residues))
	for i, m := range from.Moduli {
		premul[i] = mulmod(residues[i], yPre[i], m)
	}
	// Base change
	changed := crnsU64(premul, from, to)
	// Postmultiply
	result := make([]uint64, len(changed))
	for j, n := range to.Moduli {
		result[j] = mulmod(changed[j], zPost[j], n)
	}
	return result
}

// --------------------------------------------------------------------------
// Algorithm 1 (fast) — Posch & Posch with uint64 residues
// --------------------------------------------------------------------------

func PoschandPoschFast(aM, aN, bM, bN []uint64, p *MontParamsU64) ([]uint64, []uint64) {
	// Step 2: q_M = a_M · b_M · (-p^{-1}) % M
	qM := rnsMulU64(aM, bM, p.BaseM.Moduli)
	qM = rnsMulScalarU64(qM, p.NegPInvM, p.BaseM.Moduli)

	// Step 3: q_N = CRNS_{M→N}(q_M)
	qN := crnsU64(qM, p.BaseM, p.BaseN)

	// Step 4: r_N = (a_N · b_N + q_N · p) · M^{-1} % N
	abN := rnsMulU64(aN, bN, p.BaseN.Moduli)
	qpN := rnsMulScalarU64(qN, p.PModN, p.BaseN.Moduli)
	sumN := rnsAddU64(abN, qpN, p.BaseN.Moduli)
	rN := rnsMulScalarU64(sumN, p.MInvN, p.BaseN.Moduli)

	// Step 5: r_M = CRNS_{N→M}(r_N)
	rM := crnsU64(rN, p.BaseN, p.BaseM)

	return rM, rN
}

// --------------------------------------------------------------------------
// Algorithm 2 (fast) — VROOM with uint64 residues
// --------------------------------------------------------------------------

func VROOMFast(aM, aN, bM, bN []uint64, p *MontParamsU64) ([]uint64, []uint64) {
	// Step 2: q_M = a_M · b_M % M
	qM := rnsMulU64(aM, bM, p.BaseM.Moduli)

	// Step 3: r_N = (a_N · b_N + CRNS^{M*(-p^{-1})}_{N*(p·M^{-2})}(q_M)) % N
	abN := rnsMulU64(aN, bN, p.BaseN.Moduli)
	crnsResult := crnsWithPrePostU64(qM, p.BaseM, p.BaseN, p.CRNS1Pre, p.CRNS1Post)
	rN := rnsAddU64(abN, crnsResult, p.BaseN.Moduli)

	// Step 4: r_M = CRNS^{N*M}_{M*1}(r_N)
	rM := crnsWithPrePostU64(rN, p.BaseN, p.BaseM, p.CRNS2Pre, p.CRNS2Post)

	return rM, rN
}

// --------------------------------------------------------------------------
// Encoding / decoding helpers for the fast path
// --------------------------------------------------------------------------

// ToVROOMEncodingFast encodes a into VROOM's T-rotated form as uint64 residues.
func ToVROOMEncodingFast(a *big.Int, p *MontParamsU64) ([]uint64, []uint64) {
	v := new(big.Int).Mul(a, p.BaseM.Product)
	v.Mod(v, p.P)
	MN := new(big.Int).Mul(p.BaseM.Product, p.BaseN.Product)
	aMN := new(big.Int).Mul(v, p.T)
	aMN.Mod(aMN, MN)
	return p.BaseM.Encode(aMN), p.BaseN.Encode(aMN)
}

// FromVROOMEncodingFast decodes a VROOM result to a·b mod p.
func FromVROOMEncodingFast(rM []uint64, p *MontParamsU64) *big.Int {
	rPrime := p.BaseM.Decode(rM)
	MInvP := new(big.Int).ModInverse(p.BaseM.Product, p.P)
	result := new(big.Int).Mul(rPrime, MInvP)
	return result.Mod(result, p.P)
}

// ToMontgomeryRNSFast encodes a into standard Montgomery form as uint64 residues.
func ToMontgomeryRNSFast(a *big.Int, p *MontParamsU64) ([]uint64, []uint64) {
	aPrime := new(big.Int).Mul(a, p.BaseM.Product)
	aPrime.Mod(aPrime, p.P)
	return p.BaseM.Encode(aPrime), p.BaseN.Encode(aPrime)
}

// FromMontgomeryRNSFast decodes a standard Montgomery RNS result.
func FromMontgomeryRNSFast(rM []uint64, p *MontParamsU64) *big.Int {
	rPrime := p.BaseM.Decode(rM)
	MInvP := new(big.Int).ModInverse(p.BaseM.Product, p.P)
	result := new(big.Int).Mul(rPrime, MInvP)
	return result.Mod(result, p.P)
}

// --------------------------------------------------------------------------
// Setup helper
// --------------------------------------------------------------------------

// SetupRNSParamsU64 generates uint64 moduli and builds fast params for p.
func SetupRNSParamsU64(p *big.Int) *MontParamsU64 {
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

	// Generate distinct primes as big.Int, convert to uint64
	allBig := generateDistinctPrimes(numModM+numModN, modBits, p)
	mModuli := make([]uint64, numModM)
	nModuli := make([]uint64, numModN)
	for i := 0; i < numModM; i++ {
		mModuli[i] = allBig[i].Uint64()
	}
	for i := 0; i < numModN; i++ {
		nModuli[i] = allBig[numModM+i].Uint64()
	}

	return NewMontParamsU64(p, mModuli, nModuli)
}