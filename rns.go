// Package vroom implements the VROOM RNS Montgomery multiplication algorithm
// from "VROOM: Accelerating (Almost All) Number-Theoretic Cryptography
// Using Vectorization and the Residue Number System" (Langowski, He, Devadas)
//
// This is a pure-Go reference implementation using math/big for clarity.
// The real performance gains come from AVX512IFMA vectorization (see the paper).
package vroom

import (
	"crypto/rand"
	"fmt"
	"math/big"
)

var bigZero = big.NewInt(0)
var bigOne = big.NewInt(1)
var bigTwo = big.NewInt(2)

// --------------------------------------------------------------------------
// RNSBase: a set of coprime moduli and helper operations
// --------------------------------------------------------------------------

// RNSBase represents a set of pairwise coprime moduli.
type RNSBase struct {
	Moduli  []*big.Int // m_1, ..., m_t (or n_1, ..., n_t)
	Product *big.Int   // M = ∏ m_i  (or N = ∏ n_j)
}

// NewRNSBase creates an RNS base from the given coprime moduli.
func NewRNSBase(moduli []*big.Int) *RNSBase {
	prod := new(big.Int).Set(bigOne)
	for _, m := range moduli {
		prod.Mul(prod, m)
	}
	return &RNSBase{
		Moduli:  moduli,
		Product: prod,
	}
}

// Encode converts a big integer x into its RNS residues: (x%m_1, ..., x%m_t).
func (b *RNSBase) Encode(x *big.Int) []*big.Int {
	res := make([]*big.Int, len(b.Moduli))
	for i, m := range b.Moduli {
		res[i] = new(big.Int).Mod(x, m)
	}
	return res
}

// Decode reconstructs x from RNS residues using the Chinese Remainder Theorem.
// The result is in [0, Product-1].
func (b *RNSBase) Decode(residues []*big.Int) *big.Int {
	// Standard CRT: x = Σ r_i * (M/m_i) * ((M/m_i)^{-1} mod m_i) mod M
	result := new(big.Int)
	for i, m := range b.Moduli {
		Mi := new(big.Int).Div(b.Product, m)        // M / m_i
		MiInv := new(big.Int).ModInverse(Mi, m)      // (M/m_i)^{-1} mod m_i
		term := new(big.Int).Mul(residues[i], Mi)     // r_i * (M/m_i)
		term.Mul(term, MiInv)                         // r_i * (M/m_i) * (M/m_i)^{-1}
		result.Add(result, term)
	}
	return result.Mod(result, b.Product)
}

// --------------------------------------------------------------------------
// CRNS: Change of RNS base
// --------------------------------------------------------------------------

// CRNS converts residues from one RNS base to another using CRT reconstruction.
// Given x in RNS-from, compute x in RNS-to.
// This is exact (no ±M error) since we use full CRT.
func CRNS(residues []*big.Int, from, to *RNSBase) []*big.Int {
	x := from.Decode(residues)
	return to.Encode(x)
}

// CRNSWithPrePost performs CRNS^{from*y}_{to*z}(x):
//   1. Premultiply each residue by the corresponding y_i (mod from moduli)
//   2. Base-change from → to
//   3. Postmultiply each residue by the corresponding z_j (mod to moduli)
func CRNSWithPrePost(residues []*big.Int, from, to *RNSBase,
	yPre []*big.Int, zPost []*big.Int) []*big.Int {

	// Step 1: premultiply
	premul := make([]*big.Int, len(residues))
	for i, m := range from.Moduli {
		premul[i] = new(big.Int).Mul(residues[i], yPre[i])
		premul[i].Mod(premul[i], m)
	}

	// Step 2: base change
	changed := CRNS(premul, from, to)

	// Step 3: postmultiply
	result := make([]*big.Int, len(changed))
	for j, n := range to.Moduli {
		result[j] = new(big.Int).Mul(changed[j], zPost[j])
		result[j].Mod(result[j], n)
	}

	return result
}

// --------------------------------------------------------------------------
// Elementwise RNS operations
// --------------------------------------------------------------------------

// rnsAdd adds two RNS vectors elementwise modulo each modulus.
func rnsAdd(a, b []*big.Int, base *RNSBase) []*big.Int {
	r := make([]*big.Int, len(a))
	for i, m := range base.Moduli {
		r[i] = new(big.Int).Add(a[i], b[i])
		r[i].Mod(r[i], m)
	}
	return r
}

// rnsMul multiplies two RNS vectors elementwise modulo each modulus.
func rnsMul(a, b []*big.Int, base *RNSBase) []*big.Int {
	r := make([]*big.Int, len(a))
	for i, m := range base.Moduli {
		r[i] = new(big.Int).Mul(a[i], b[i])
		r[i].Mod(r[i], m)
	}
	return r
}

// rnsMulScalar multiplies an RNS vector by a scalar (given per-modulus) elementwise.
func rnsMulScalar(a []*big.Int, scalar []*big.Int, base *RNSBase) []*big.Int {
	r := make([]*big.Int, len(a))
	for i, m := range base.Moduli {
		r[i] = new(big.Int).Mul(a[i], scalar[i])
		r[i].Mod(r[i], m)
	}
	return r
}

// --------------------------------------------------------------------------
// Precomputed parameters for RNS Montgomery multiplication
// --------------------------------------------------------------------------

// MontParams holds all precomputed constants for both Algorithm 1 and VROOM.
type MontParams struct {
	P     *big.Int // The prime modulus p
	BaseM *RNSBase // M-base
	BaseN *RNSBase // N-base

	// -- Algorithm 1 (Posch & Posch) constants --
	// For each m_i: (-p^{-1}) mod m_i
	NegPInvM []*big.Int
	// For each n_j: p mod n_j
	PModN []*big.Int
	// For each n_j: M^{-1} mod n_j
	MInvN []*big.Int

	// -- VROOM (Algorithm 2) additional constants --
	T    *big.Int // T ≡ 1 (mod M), T ≡ M^{-1} (mod N)
	TInv *big.Int // T^{-1} mod (MN)

	// CRNS^{M*(-p^{-1})}_{N*(p*M^{-2})} constants per modulus
	VroomCRNS1Pre  []*big.Int // (-p^{-1}) mod m_i (same as NegPInvM)
	VroomCRNS1Post []*big.Int // (p * M^{-2}) mod n_j

	// CRNS^{N*M}_{M*1} constants per modulus
	VroomCRNS2Pre  []*big.Int // M mod n_j
	VroomCRNS2Post []*big.Int // 1 mod m_i (just 1s)
}

// NewMontParams creates and precomputes all Montgomery parameters.
func NewMontParams(p *big.Int, mModuli, nModuli []*big.Int) *MontParams {
	baseM := NewRNSBase(mModuli)
	baseN := NewRNSBase(nModuli)
	M := baseM.Product
	N := baseN.Product

	// Verify constraints
	nineP := new(big.Int).Mul(big.NewInt(9), p)
	sixP := new(big.Int).Mul(big.NewInt(6), p)
	if M.Cmp(nineP) <= 0 {
		panic(fmt.Sprintf("M (%d bits) must be > 9p (%d bits)", M.BitLen(), nineP.BitLen()))
	}
	if N.Cmp(sixP) <= 0 {
		panic(fmt.Sprintf("N (%d bits) must be > 6p (%d bits)", N.BitLen(), sixP.BitLen()))
	}

	params := &MontParams{
		P:     new(big.Int).Set(p),
		BaseM: baseM,
		BaseN: baseN,
	}

	// ----- Algorithm 1 constants -----

	// (-p^{-1}) mod m_i
	params.NegPInvM = make([]*big.Int, len(mModuli))
	for i, m := range mModuli {
		pInv := new(big.Int).ModInverse(p, m)
		params.NegPInvM[i] = new(big.Int).Sub(m, pInv) // -p^{-1} mod m
	}

	// p mod n_j
	params.PModN = make([]*big.Int, len(nModuli))
	for j, n := range nModuli {
		params.PModN[j] = new(big.Int).Mod(p, n)
	}

	// M^{-1} mod n_j
	params.MInvN = make([]*big.Int, len(nModuli))
	for j, n := range nModuli {
		params.MInvN[j] = new(big.Int).ModInverse(M, n)
	}

	// ----- VROOM constants -----

	// T: unique value in [0, MN-1] with T ≡ 1 (mod M) and T ≡ M^{-1} (mod N)
	// By CRT: T = N*(N^{-1} mod M) + M*(M^{-1} mod N)^2 mod (MN)
	MN := new(big.Int).Mul(M, N)
	NInvM := new(big.Int).ModInverse(N, M)     // N^{-1} mod M
	MInvN := new(big.Int).ModInverse(M, N)     // M^{-1} mod N
	term1 := new(big.Int).Mul(N, NInvM)         // N * (N^{-1} mod M)
	MInvNSq := new(big.Int).Mul(MInvN, MInvN)   // (M^{-1} mod N)^2
	term2 := new(big.Int).Mul(M, MInvNSq)       // M * (M^{-1} mod N)^2
	params.T = new(big.Int).Add(term1, term2)
	params.T.Mod(params.T, MN)

	// Verify T properties
	tModM := new(big.Int).Mod(params.T, M)
	tModN := new(big.Int).Mod(params.T, N)
	if tModM.Cmp(bigOne) != 0 {
		panic("T ≢ 1 (mod M)")
	}
	expectedMInvN := new(big.Int).ModInverse(M, N)
	if tModN.Cmp(expectedMInvN) != 0 {
		panic("T ≢ M^{-1} (mod N)")
	}

	// T^{-1} mod MN
	params.TInv = new(big.Int).ModInverse(params.T, MN)

	// CRNS1: premul = (-p^{-1}) mod m_i, postmul = (p * M^{-2}) mod n_j
	params.VroomCRNS1Pre = params.NegPInvM // reuse

	MInvNVal := new(big.Int).ModInverse(M, N)
	MInv2N := new(big.Int).Mul(MInvNVal, MInvNVal)
	MInv2N.Mod(MInv2N, N) // M^{-2} mod N
	pMInv2 := new(big.Int).Mul(p, MInv2N)
	pMInv2.Mod(pMInv2, N) // p * M^{-2} mod N

	params.VroomCRNS1Post = make([]*big.Int, len(nModuli))
	for j, n := range nModuli {
		params.VroomCRNS1Post[j] = new(big.Int).Mod(pMInv2, n)
	}

	// CRNS2: premul = M mod n_j, postmul = 1 mod m_i
	params.VroomCRNS2Pre = make([]*big.Int, len(nModuli))
	for j, n := range nModuli {
		params.VroomCRNS2Pre[j] = new(big.Int).Mod(M, n)
	}
	params.VroomCRNS2Post = make([]*big.Int, len(mModuli))
	for i := range mModuli {
		params.VroomCRNS2Post[i] = new(big.Int).Set(bigOne)
	}

	return params
}

// --------------------------------------------------------------------------
// Algorithm 1: Posch & Posch RNS Montgomery multiplication
// --------------------------------------------------------------------------

// PoschandPosch implements Algorithm 1 from the paper.
// Inputs:  a_MN, b_MN in Montgomery form, stored as (M-residues, N-residues)
// Output:  r_MN ≡ a * b * M^{-1} (mod p) in Montgomery form
func PoschandPosch(aM, aN, bM, bN []*big.Int, params *MontParams) ([]*big.Int, []*big.Int) {
	// Step 2: q_M = a_M · b_M · (-p^{-1}) % M  [elementwise]
	qM := rnsMul(aM, bM, params.BaseM)
	qM = rnsMulScalar(qM, params.NegPInvM, params.BaseM)

	// Step 3: q_N = CRNS_{M→N}(q_M)
	qN := CRNS(qM, params.BaseM, params.BaseN)

	// Step 4: r_N = (a_N · b_N + q_N · p) · M^{-1} % N  [elementwise]
	abN := rnsMul(aN, bN, params.BaseN)
	qpN := rnsMulScalar(qN, params.PModN, params.BaseN)
	sumN := rnsAdd(abN, qpN, params.BaseN)
	rN := rnsMulScalar(sumN, params.MInvN, params.BaseN)

	// Step 5: r_M = CRNS_{N→M}(r_N)
	rM := CRNS(rN, params.BaseN, params.BaseM)

	return rM, rN
}

// --------------------------------------------------------------------------
// Algorithm 2: VROOM — optimized RNS Montgomery multiplication
// --------------------------------------------------------------------------

// VROOM implements Algorithm 2 from the paper.
// Inputs must be in VROOM encoding: a_MN = (a·M%p)·T % (MN)
//   so a_M = (a·M%p) % M  (since T ≡ 1 mod M)
//   and a_N = (a·M%p)·M^{-1} % N  (since T ≡ M^{-1} mod N)
// Output: r_MN ≡ (a·b·M%p)·T (mod MN)
func VROOM(aM, aN, bM, bN []*big.Int, params *MontParams) ([]*big.Int, []*big.Int) {
	// Step 2: q_M = a_M · b_M % M  [elementwise — no -p^{-1} factor!]
	qM := rnsMul(aM, bM, params.BaseM)

	// Step 3: r_N = (a_N · b_N + CRNS^{M*(-p^{-1})}_{N*(p·M^{-2})}(q_M)) % N
	abN := rnsMul(aN, bN, params.BaseN)
	crnsResult := CRNSWithPrePost(qM, params.BaseM, params.BaseN,
		params.VroomCRNS1Pre, params.VroomCRNS1Post)
	rN := rnsAdd(abN, crnsResult, params.BaseN)

	// Step 4: r_M = CRNS^{N*M}_{M*1}(r_N)
	rM := CRNSWithPrePost(rN, params.BaseN, params.BaseM,
		params.VroomCRNS2Pre, params.VroomCRNS2Post)

	return rM, rN
}

// --------------------------------------------------------------------------
// Encoding / decoding helpers
// --------------------------------------------------------------------------

// ToMontgomeryRNS encodes a value a into standard Montgomery RNS form
// for Algorithm 1: a' = a*M mod p, then split into (a'%M residues, a'%N residues).
func ToMontgomeryRNS(a *big.Int, params *MontParams) ([]*big.Int, []*big.Int) {
	// a' = a * M mod p
	aPrime := new(big.Int).Mul(a, params.BaseM.Product)
	aPrime.Mod(aPrime, params.P)
	return params.BaseM.Encode(aPrime), params.BaseN.Encode(aPrime)
}

// FromMontgomeryRNS decodes a standard Montgomery RNS result back to an integer.
// r = decoded_value * M^{-1} mod p
func FromMontgomeryRNS(rM []*big.Int, params *MontParams) *big.Int {
	// Reconstruct r' from M-base (r' < 3p < M, so CRT gives exact value)
	rPrime := params.BaseM.Decode(rM)
	// r = r' * M^{-1} mod p
	MInvP := new(big.Int).ModInverse(params.BaseM.Product, params.P)
	result := new(big.Int).Mul(rPrime, MInvP)
	result.Mod(result, params.P)
	return result
}

// ToVROOMEncoding encodes a value a into VROOM's T-rotated Montgomery form:
// a_MN = (a*M%p)*T % (MN)
func ToVROOMEncoding(a *big.Int, params *MontParams) ([]*big.Int, []*big.Int) {
	// v = a * M mod p
	v := new(big.Int).Mul(a, params.BaseM.Product)
	v.Mod(v, params.P)
	// a_MN = v * T mod MN
	MN := new(big.Int).Mul(params.BaseM.Product, params.BaseN.Product)
	aMN := new(big.Int).Mul(v, params.T)
	aMN.Mod(aMN, MN)
	return params.BaseM.Encode(aMN), params.BaseN.Encode(aMN)
}

// FromVROOMEncoding decodes a VROOM result back to an integer.
// The encoded value is (a*b*M%p)*T mod MN.
// To get a*b mod p: decode, multiply by T^{-1}, then by M^{-1} mod p.
func FromVROOMEncoding(rM []*big.Int, params *MontParams) *big.Int {
	// r_M encodes the value at mod M. Since T ≡ 1 mod M,
	// r_M = (a*b*M % p) mod M. And since (a*b*M%p) < p < M/9,
	// CRT on M-base gives the exact value (a*b*M % p).
	rPrime := params.BaseM.Decode(rM)
	// result = rPrime * M^{-1} mod p = a*b mod p
	MInvP := new(big.Int).ModInverse(params.BaseM.Product, params.P)
	result := new(big.Int).Mul(rPrime, MInvP)
	result.Mod(result, params.P)
	return result
}

// --------------------------------------------------------------------------
// Prime generation for RNS moduli
// --------------------------------------------------------------------------

// generateDistinctPrimes generates n distinct primes of the given bit size,
// ensuring none equals p.
func generateDistinctPrimes(n, bitSize int, p *big.Int) []*big.Int {
	primes := make([]*big.Int, 0, n)
	seen := make(map[string]bool)
	for len(primes) < n {
		q, err := rand.Prime(rand.Reader, bitSize)
		if err != nil {
			panic(err)
		}
		key := q.String()
		if seen[key] || q.Cmp(p) == 0 {
			continue
		}
		// Ensure coprime with p (always true since both are prime and distinct)
		seen[key] = true
		primes = append(primes, q)
	}
	return primes
}

// SetupRNSParams generates RNS moduli and creates MontParams for a given prime p.
func SetupRNSParams(p *big.Int) *MontParams {
	pBits := p.BitLen()
	// Moduli bit size: use 32-bit primes so products of residues fit in 64 bits
	modBits := 32

	// Need M > 9p ⟹ log2(M) > log2(9) + log2(p) ≈ pBits + 4
	// Need N > 6p ⟹ log2(N) > log2(6) + log2(p) ≈ pBits + 3
	numModM := (pBits + 4 + modBits - 1) / modBits
	if numModM < 2 {
		numModM = 2
	}
	numModN := (pBits + 3 + modBits - 1) / modBits
	if numModN < 2 {
		numModN = 2
	}
	// Add one extra modulus to each for safety margin
	numModM++
	numModN++

	allModuli := generateDistinctPrimes(numModM+numModN, modBits, p)
	mModuli := allModuli[:numModM]
	nModuli := allModuli[numModM:]

	return NewMontParams(p, mModuli, nModuli)
}

// randomInRange generates a random big.Int in [0, max-1].
func randomInRange(max *big.Int) *big.Int {
	n, err := rand.Int(rand.Reader, max)
	if err != nil {
		panic(err)
	}
	return n
}
