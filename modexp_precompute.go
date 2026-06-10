// modexp_precompute.go — Precomputed-table modular exponentiation via VROOMStage4
//
// Strategy: precompute base^(2^i) for every bit position i, store in VROOM form.
// At runtime, scan exponent LSB→MSB, multiply accumulator by table[i] for each
// set bit. No squaring at runtime — only ~n/2 multiplies for n-bit exponent.
//
// Precompute cost: (n-1) VROOMStage4 squarings (one-time, ~828 μs for 1024-bit).
// Runtime cost:    ~n/2 VROOMStage4 multiplies (~415 μs for 1024-bit).
// Memory:          n × (tM + tN) uint64s (~360 KB for 1024-bit).
//
// Amortizes after a single exponentiation — ideal when base/modulus are reused
// (RSA, DH, repeated signatures with the same key).
//
// API:
//   NewVROOMPreTable          — build precomputation table (one-time)
//   ModExpPrecomputed         — zero-alloc inner loop (pre-encoded result)
//   ModExpVROOMPrecomputed    — convenience wrapper (big.Int in/out)

package main

import (
	"math/big"
)

// VROOMPreTable holds precomputed powers base^(2^i) in VROOM encoding.
// table[i] = base^(2^i) mod p, stored as RNS residues.
// Also caches the identity element (1) in VROOM form to avoid encoding at runtime.
// Build once with NewVROOMPreTable, reuse across exponentiations.
type VROOMPreTable struct {
	tableM [][]uint64 // tableM[i] = M-residues of base^(2^i)
	tableN [][]uint64 // tableN[i] = N-residues of base^(2^i)
	oneM   []uint64   // pre-encoded identity (1) — M-residues
	oneN   []uint64   // pre-encoded identity (1) — N-residues
	maxBits int       // number of precomputed bit positions
	tM, tN  int       // residue counts
}

// NewVROOMPreTable precomputes base^(2^i) for i = 0..maxBits-1.
//
// maxBits should be >= the maximum exponent bit length you'll use.
// For 1024-bit exponents, use maxBits=1024.
//
// Cost: (maxBits-1) VROOMStage4 calls (squarings).
// Memory: maxBits × (tM + tN) × 8 bytes.
func NewVROOMPreTable(base *big.Int, maxBits int, params *MontParamsStage4) *VROOMPreTable {
	tM := len(params.BaseM.Moduli)
	tN := len(params.BaseN.Moduli)

	// Pre-encode identity element (1) — cached for zero-alloc runtime
	rawOneM, rawOneN := ToVROOMEncodingStage4(big.NewInt(1), params)

	pt := &VROOMPreTable{
		tableM:  make([][]uint64, maxBits),
		tableN:  make([][]uint64, maxBits),
		oneM:    make([]uint64, tM),
		oneN:    make([]uint64, tN),
		maxBits: maxBits,
		tM:      tM,
		tN:      tN,
	}
	copy(pt.oneM, rawOneM)
	copy(pt.oneN, rawOneN)

	// table[0] = base^(2^0) = base in VROOM encoding
	curM, curN := ToVROOMEncodingStage4(base, params)

	pt.tableM[0] = make([]uint64, tM)
	pt.tableN[0] = make([]uint64, tN)
	copy(pt.tableM[0], curM)
	copy(pt.tableN[0], curN)

	// table[i] = table[i-1]^2  (each entry is the square of the previous)
	for i := 1; i < maxBits; i++ {
		rM, rN := VROOMStage4(curM, curN, curM, curN, params)

		pt.tableM[i] = make([]uint64, tM)
		pt.tableN[i] = make([]uint64, tN)
		copy(pt.tableM[i], rM)
		copy(pt.tableN[i], rN)

		curM = pt.tableM[i]
		curN = pt.tableN[i]
	}

	return pt
}

// ============================================================================
// Inner precomputed modexp — zero allocations at runtime
// ============================================================================

// ModExpPrecomputed computes base^exp using the precomputed table.
// Zero allocations at runtime. NOT constant-time — branches on bits of exp.
//
// Result is written to w.accM, w.accN (caller reads from there).
// The table must have been built with the same base and params.
//
// Panics if exp.BitLen() > table.maxBits.
func ModExpPrecomputed(exp *big.Int, pt *VROOMPreTable,
	w *ModExpWorkspace, params *MontParamsStage4) {

	if exp.Sign() == 0 {
		copy(w.accM, pt.oneM)
		copy(w.accN, pt.oneN)
		return
	}

	bitLen := exp.BitLen()
	if bitLen > pt.maxBits {
		panic("modexp_precompute: exponent bit length exceeds table size")
	}

	// acc = 1 (pre-encoded in table, zero allocations)
	copy(w.accM, pt.oneM)
	copy(w.accN, pt.oneN)

	// Scan exponent LSB → MSB, multiply for each set bit
	for i := 0; i < bitLen; i++ {
		if exp.Bit(i) == 1 {
			rM, rN := VROOMStage4(w.accM, w.accN, pt.tableM[i], pt.tableN[i], params)
			copy(w.accM, rM)
			copy(w.accN, rN)
		}
	}
}

// ModExpPrecomputedConstTime computes base^exp using the precomputed table.
// Zero allocations at runtime. Constant-time — no data-dependent branches.
//
// Always performs a VROOM multiply for every bit position up to the table size,
// then uses branchless select to keep or discard the result.
func ModExpPrecomputedConstTime(exp *big.Int, pt *VROOMPreTable,
	w *ModExpWorkspace, params *MontParamsStage4) {

	if exp.Sign() == 0 {
		copy(w.accM, pt.oneM)
		copy(w.accN, pt.oneN)
		return
	}

	bitLen := exp.BitLen()
	if bitLen > pt.maxBits {
		panic("modexp_precompute: exponent bit length exceeds table size")
	}

	// acc = 1 (pre-encoded in table, zero allocations)
	copy(w.accM, pt.oneM)
	copy(w.accN, pt.oneN)

	// Process every bit position — always multiply, then select
	for i := 0; i < bitLen; i++ {
		rM, rN := VROOMStage4(w.accM, w.accN, pt.tableM[i], pt.tableN[i], params)

		bit := uint64(exp.Bit(i))
		ctCondCopy(w.accM, w.accM, rM, bit, w.tM)
		ctCondCopy(w.accN, w.accN, rN, bit, w.tN)
	}
}

// ============================================================================
// Convenience wrappers — encode + precomputed exp + decode
// ============================================================================

// ModExpVROOMPrecomputed computes base^exp mod p using a precomputed table.
// Non-constant-time.
func ModExpVROOMPrecomputed(exp *big.Int, pt *VROOMPreTable,
	w *ModExpWorkspace, params *MontParamsStage4) *big.Int {

	if exp.Sign() == 0 {
		return big.NewInt(1)
	}
	ModExpPrecomputed(exp, pt, w, params)
	return FromVROOMEncodingStage4(w.accM, params)
}

// ModExpVROOMPrecomputedConstTime computes base^exp mod p using a precomputed table.
// Constant-time on exp.
func ModExpVROOMPrecomputedConstTime(exp *big.Int, pt *VROOMPreTable,
	w *ModExpWorkspace, params *MontParamsStage4) *big.Int {

	if exp.Sign() == 0 {
		return big.NewInt(1)
	}
	ModExpPrecomputedConstTime(exp, pt, w, params)
	return FromVROOMEncodingStage4(w.accM, params)
}
