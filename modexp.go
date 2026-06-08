// modexp.go — Modular exponentiation via VROOMStage4
//
// a^e mod p using square-and-multiply, where each multiplication
// is a single VROOMStage4 call (~800 ns for 1024-bit).
//
// Two variants:
//   ModExpVROOM          — left-to-right binary, non-constant-time (public exponents)
//   ModExpVROOMConstTime — always-multiply with conditional copy (secret exponents)

package main

import (
	"math/big"
)

// ModExpWorkspace holds pre-allocated buffers for zero-allocation modexp.
type ModExpWorkspace struct {
	accM, accN   []uint64 // current accumulator
	sqM, sqN     []uint64 // saved square result (constant-time only)
	tM, tN       int
}

// NewModExpWorkspace allocates buffers for modular exponentiation.
func NewModExpWorkspace(params *MontParamsStage4) *ModExpWorkspace {
	tM := len(params.BaseM.Moduli)
	tN := len(params.BaseN.Moduli)
	return &ModExpWorkspace{
		accM: make([]uint64, tM),
		accN: make([]uint64, tN),
		sqM:  make([]uint64, tM),
		sqN:  make([]uint64, tN),
		tM:   tM,
		tN:   tN,
	}
}

// ============================================================================
// Public-exponent modular exponentiation (non-constant-time)
// ============================================================================

// ModExpVROOM computes base^exp mod p using left-to-right binary method.
//
// NOT constant-time — branches on bits of exp. Safe only for public exponents
// (e.g. RSA verify with e = 65537).
//
// For 1024-bit p, e = 65537: 17 squares + 1 multiply = 18 VROOM calls ≈ 14 μs.
// For 1024-bit p, 1024-bit e: ~1536 VROOM calls ≈ 1.2 ms.
func ModExpVROOM(base, exp *big.Int, params *MontParamsStage4) *big.Int {
	if exp.Sign() == 0 {
		return big.NewInt(1)
	}

	w := NewModExpWorkspace(params)

	// Encode base into VROOM form (T-rotated Montgomery RNS)
	baseM, baseN := ToVROOMEncodingStage4(base, params)

	// Initialize accumulator = base (MSB is always 1)
	copy(w.accM, baseM)
	copy(w.accN, baseN)

	// Process remaining bits from second-highest to lowest
	for i := exp.BitLen() - 2; i >= 0; i-- {
		// Square: acc = acc * acc
		rM, rN := VROOMStage4(w.accM, w.accN, w.accM, w.accN, params)
		copy(w.accM, rM)
		copy(w.accN, rN)

		// Multiply if bit is 1: acc = acc * base
		if exp.Bit(i) == 1 {
			rM, rN = VROOMStage4(w.accM, w.accN, baseM, baseN, params)
			copy(w.accM, rM)
			copy(w.accN, rN)
		}
	}

	return FromVROOMEncodingStage4(w.accM, params)
}

// ============================================================================
// Secret-exponent modular exponentiation (constant-time)
// ============================================================================

// ModExpVROOMConstTime computes base^exp mod p in constant time.
//
// Always executes both square and multiply for every bit of the exponent,
// then uses a constant-time conditional copy to select the correct result.
// No data-dependent branches, no timing side-channel on exp.
//
// Cost: 2 VROOM calls per bit of exp (vs 1.5 average for non-constant-time).
// For 1024-bit e: 2048 VROOM calls ≈ 1.6 ms.
func ModExpVROOMConstTime(base, exp *big.Int, params *MontParamsStage4) *big.Int {
	if exp.Sign() == 0 {
		return big.NewInt(1)
	}

	w := NewModExpWorkspace(params)

	baseM, baseN := ToVROOMEncodingStage4(base, params)

	// Initialize accumulator = base (MSB is always 1)
	copy(w.accM, baseM)
	copy(w.accN, baseN)

	for i := exp.BitLen() - 2; i >= 0; i-- {
		// Square: acc = acc * acc → save to sq
		rM, rN := VROOMStage4(w.accM, w.accN, w.accM, w.accN, params)
		copy(w.sqM, rM)
		copy(w.sqN, rN)

		// Multiply: sq * base → result in workspace
		rM, rN = VROOMStage4(w.sqM, w.sqN, baseM, baseN, params)

		// Constant-time conditional copy:
		//   bit=1 → acc = multiply result (rM, rN)
		//   bit=0 → acc = square result (sqM, sqN)
		bit := uint64(exp.Bit(i))
		ctCondCopy(w.accM, w.sqM, rM, bit, w.tM)
		ctCondCopy(w.accN, w.sqN, rN, bit, w.tN)
	}

	return FromVROOMEncodingStage4(w.accM, params)
}

// ctCondCopy performs a constant-time conditional copy:
//
//	if sel == 1: dst[i] = ifOne[i]
//	if sel == 0: dst[i] = ifZero[i]
//
// No branches, no data-dependent memory access. sel must be 0 or 1.
//
//go:nosplit
func ctCondCopy(dst, ifZero, ifOne []uint64, sel uint64, n int) {
	mask := -sel // 0xFFFFFFFFFFFFFFFF if sel=1, 0x0 if sel=0
	for i := 0; i < n; i++ {
		dst[i] = (ifOne[i] & mask) | (ifZero[i] & ^mask)
	}
}
