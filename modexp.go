// modexp.go — Modular exponentiation via VROOMStage4
//
// a^e mod p using square-and-multiply, where each multiplication
// is a single VROOMStage4 call (~800 ns for 1024-bit).
//
// Two API levels:
//   ModExpVROOM / ModExpVROOMConstTime — convenience (encode + exp + decode)
//   ModExpInner / ModExpInnerConstTime — zero-alloc inner loop (pre-encoded)

package vroom

import (
	"math/big"
)

// ModExpWorkspace holds pre-allocated buffers for zero-allocation modexp.
// Allocate once, reuse across calls.
type ModExpWorkspace struct {
	accM, accN []uint64 // current accumulator
	sqM, sqN   []uint64 // saved square result (constant-time only)
	selM, selN []uint64 // selected table entry (windowed constant-time only)
	tM, tN     int
}

// NewModExpWorkspace allocates buffers once.
func NewModExpWorkspace(params *MontParamsStage4) *ModExpWorkspace {
	tM := len(params.BaseM.Moduli)
	tN := len(params.BaseN.Moduli)
	return &ModExpWorkspace{
		accM: make([]uint64, tM),
		accN: make([]uint64, tN),
		sqM:  make([]uint64, tM),
		sqN:  make([]uint64, tN),
		selM: make([]uint64, tM),
		selN: make([]uint64, tN),
		tM:   tM,
		tN:   tN,
	}
}

// ============================================================================
// Inner modexp — zero allocations, works on pre-encoded VROOM values
// ============================================================================

// ModExpInner computes base^exp in VROOM form. Zero allocations.
// NOT constant-time — branches on bits of exp.
//
// baseM, baseN: pre-encoded base via ToVROOMEncodingStage4
// Result is written to w.accM, w.accN (caller reads from there).
func ModExpInner(baseM, baseN []uint64, exp *big.Int,
	w *ModExpWorkspace, params *MontParamsStage4) {

	if exp.Sign() == 0 {
		// a^0 = 1 — encode 1 into accumulator
		oneM, oneN := ToVROOMEncodingStage4(big.NewInt(1), params)
		copy(w.accM, oneM)
		copy(w.accN, oneN)
		return
	}

	// acc = base (MSB is always 1)
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
}

// ModExpInnerConstTime computes base^exp in VROOM form. Zero allocations.
// Constant-time — no data-dependent branches.
func ModExpInnerConstTime(baseM, baseN []uint64, exp *big.Int,
	w *ModExpWorkspace, params *MontParamsStage4) {

	if exp.Sign() == 0 {
		oneM, oneN := ToVROOMEncodingStage4(big.NewInt(1), params)
		copy(w.accM, oneM)
		copy(w.accN, oneN)
		return
	}

	copy(w.accM, baseM)
	copy(w.accN, baseN)

	for i := exp.BitLen() - 2; i >= 0; i-- {
		// Square → save to sq
		rM, rN := VROOMStage4(w.accM, w.accN, w.accM, w.accN, params)
		copy(w.sqM, rM)
		copy(w.sqN, rN)

		// Multiply: sq * base → workspace
		rM, rN = VROOMStage4(w.sqM, w.sqN, baseM, baseN, params)

		// Constant-time select: bit=1 → mul result, bit=0 → sq result
		bit := uint64(exp.Bit(i))
		ctCondCopy(w.accM, w.sqM, rM, bit, w.tM)
		ctCondCopy(w.accN, w.sqN, rN, bit, w.tN)
	}
}

// ctCondCopy: dst = sel ? ifOne : ifZero. No branches.
//
//go:nosplit
func ctCondCopy(dst, ifZero, ifOne []uint64, sel uint64, n int) {
	mask := -sel
	for i := 0; i < n; i++ {
		dst[i] = (ifOne[i] & mask) | (ifZero[i] & ^mask)
	}
}

// ============================================================================
// Convenience wrappers — encode + inner + decode
// ============================================================================

// ModExpVROOM computes base^exp mod p. Non-constant-time.
func ModExpVROOM(base, exp *big.Int, w *ModExpWorkspace, params *MontParamsStage4) *big.Int {
	if exp.Sign() == 0 {
		return big.NewInt(1)
	}
	baseM, baseN := ToVROOMEncodingStage4(base, params)
	ModExpInner(baseM, baseN, exp, w, params)
	return FromVROOMEncodingStage4(w.accM, params)
}

// ModExpVROOMConstTime computes base^exp mod p. Constant-time on exp.
func ModExpVROOMConstTime(base, exp *big.Int, w *ModExpWorkspace, params *MontParamsStage4) *big.Int {
	if exp.Sign() == 0 {
		return big.NewInt(1)
	}
	baseM, baseN := ToVROOMEncodingStage4(base, params)
	ModExpInnerConstTime(baseM, baseN, exp, w, params)
	return FromVROOMEncodingStage4(w.accM, params)
}