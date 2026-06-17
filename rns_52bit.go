// rns_52bit.go — 52-bit moduli for VROOM
//
// Switches from 32-bit to 52-bit RNS moduli. This is the single largest
// optimization: fewer moduli means smaller CRNS matrices and fewer iterations.
//
// Example for 256-bit prime p:
//   32-bit moduli: tM ≈ 10, tN ≈ 10 → CRNS is 10×10 = 100 mul-adds
//   52-bit moduli: tM ≈  6, tN ≈  6 → CRNS is  6×6  =  36 mul-adds
//   ~2.8× fewer CRNS operations, ~1.5-2× total speedup
//
// 52-bit moduli are specifically chosen to match AVX512IFMA's VPMADD52
// instruction which operates on 52-bit unsigned integers. The scalar
// mulmod/addmod functions already handle 52-bit operands correctly.

package vroom

import (
	"crypto/rand"
	"math/big"
)

// --------------------------------------------------------------------------
// 52-bit prime generation
// --------------------------------------------------------------------------

// generate52BitPrimes generates n distinct primes in (2^51, 2^52-1),
// ensuring none equals p. This range guarantees:
//   - All primes fit in 52 bits (required by VPMADD52)
//   - Products of two residues fit in 104 bits (handled by bits.Mul64)
//   - a + b < 2^53 (fits in uint64 for addmod)
func generate52BitPrimes(n int, p *big.Int) []uint64 {
	// Range: [2^51, 2^52 - 1] — all 52-bit odd numbers
	lo := new(big.Int).Lsh(bigOne, 51)
	hi := new(big.Int).Lsh(bigOne, 52)
	rangeSize := new(big.Int).Sub(hi, lo)

	primes := make([]uint64, 0, n)
	seen := make(map[uint64]bool)

	for len(primes) < n {
		// Generate random prime in the full bit range
		q, err := rand.Prime(rand.Reader, 52)
		if err != nil {
			panic(err)
		}
		// Ensure it's actually 52 bits (at least 2^51)
		if q.Cmp(lo) < 0 {
			// Shift into range
			offset, _ := rand.Int(rand.Reader, rangeSize)
			q.Add(lo, offset)
			// Find next prime
			if !q.ProbablyPrime(20) {
				q.Add(q, bigOne)
				for !q.ProbablyPrime(20) {
					q.Add(q, bigTwo)
				}
			}
		}
		// Must be < 2^52
		if q.BitLen() > 52 {
			continue
		}
		val := q.Uint64()
		if seen[val] || (p.BitLen() <= 52 && val == p.Uint64()) {
			continue
		}
		seen[val] = true
		primes = append(primes, val)
	}
	return primes
}

// computeModuliCount returns the number of moduli needed for M > factor*p
// with the given modulus bit size.
func computeModuliCount(pBits, modBits int, factorBits int) int {
	// Need log2(M) > factorBits + pBits
	totalBits := pBits + factorBits
	n := (totalBits + modBits - 1) / modBits
	if n < 2 {
		n = 2
	}
	n++ // safety margin
	return n
}

// --------------------------------------------------------------------------
// Stage 1 (uint64 residues) — 52-bit moduli
// --------------------------------------------------------------------------

func SetupRNSParamsU64_52(p *big.Int) *MontParamsU64 {
	pBits := p.BitLen()
	numModM := computeModuliCount(pBits, 52, 4) // M > 9p ≈ 2^(pBits+3.2), use 4 for margin
	numModN := computeModuliCount(pBits, 52, 3) // N > 6p ≈ 2^(pBits+2.6), use 3 for margin

	all := generate52BitPrimes(numModM+numModN, p)
	return NewMontParamsU64(p, all[:numModM], all[numModM:])
}

// --------------------------------------------------------------------------
// Stage 2 (matrix CRNS) — 52-bit moduli
// --------------------------------------------------------------------------

func SetupRNSParamsStage2_52(p *big.Int) *MontParamsStage2 {
	pBits := p.BitLen()
	numModM := computeModuliCount(pBits, 52, 4)
	numModN := computeModuliCount(pBits, 52, 3)

	all := generate52BitPrimes(numModM+numModN, p)
	return NewMontParamsStage2(p, all[:numModM], all[numModM:])
}

// --------------------------------------------------------------------------
// Stage 2.5 (no-alloc) — 52-bit moduli
// --------------------------------------------------------------------------

func SetupRNSParamsNoAlloc_52(p *big.Int) *MontParamsNoAlloc {
	s2 := SetupRNSParamsStage2_52(p)
	tM := len(s2.BaseM.Moduli)
	tN := len(s2.BaseN.Moduli)
	return &MontParamsNoAlloc{
		P:     s2.P,
		BaseM: s2.BaseM,
		BaseN: s2.BaseN,
		T:     s2.T,
		TInv:  s2.TInv,
		CRNS1: s2.CRNS1,
		CRNS2: s2.CRNS2,
		W:     NewWorkspace(tM, tN),
	}
}

// --------------------------------------------------------------------------
// Stage 3 (AVX512) — 52-bit moduli
// --------------------------------------------------------------------------

func SetupRNSParamsStage3_52(p *big.Int) *MontParamsStage3 {
	pBits := p.BitLen()
	numModM := computeModuliCount(pBits, 52, 4)
	numModN := computeModuliCount(pBits, 52, 3)

	all := generate52BitPrimes(numModM+numModN, p)
	mModuli := all[:numModM]
	nModuli := all[numModM:]

	baseM := NewRNSBaseU64(mModuli)
	baseN := NewRNSBaseU64(nModuli)
	M := baseM.Product
	N := baseN.Product
	MN := new(big.Int).Mul(M, N)

	// Verify constraints
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
	crns1 := NewCRNSMatrixAVX512(baseM, baseN, negPInvM, pMInv2N)

	// CRNS2: from=N, to=M, y=(M mod N), z=1
	MmodN := new(big.Int).Mod(M, N)
	crns2 := NewCRNSMatrixAVX512(baseN, baseM, MmodN, bigOne)

	// Workspace with padded buffers
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
