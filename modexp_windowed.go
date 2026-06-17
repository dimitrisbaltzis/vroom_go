// modexp_windowed.go — Windowed precomputed-table modular exponentiation via VROOMStage4
//
// Extension of modexp_precompute.go: instead of scanning 1 bit at a time,
// scan k=4 bits at a time. Precompute base^(j · 2^(4w)) for each window
// position w and digit j=1..15. At runtime, one table lookup + multiply
// per window instead of ~2 multiplies per window.
//
// Precompute cost: 1023 squarings + 14×256 multiplies ≈ 4607 VROOM calls (~3.7 ms).
// Runtime cost:    ~240 multiplies for 1024-bit exponent (vs ~512 for 1-bit).
// Memory:          256 windows × 15 digits × (tM+tN) uint64s ≈ 1.35 MB (1024-bit).
//
// Trade-off: 4.5× higher precompute cost, ~2.1× faster runtime.
// Best when the same base is used for many exponentiations (RSA, DH).
//
// API:
//   NewVROOMWindowTable           — build windowed table (one-time)
//   ModExpWindowed                — zero-alloc runtime, non-constant-time
//   ModExpVROOMWindowed           — convenience wrapper (big.Int in/out)

package vroom

import (
	"math/big"
)

const windowK = 4
const windowSize = 1 << windowK // 16
const windowMask = windowSize - 1 // 15

// VROOMWindowTable holds precomputed powers for k=4 windowed exponentiation.
//
// windowM[w][j] = M-residues of base^(j · 2^(4w))   for w=0..numWindows-1, j=1..15
// windowN[w][j] = N-residues of base^(j · 2^(4w))
//
// Index j=0 is unused (multiplying by base^0 = 1 is a no-op).
type VROOMWindowTable struct {
	windowM    [][][]uint64 // [numWindows][windowSize][tM]
	windowN    [][][]uint64 // [numWindows][windowSize][tN]
	oneM       []uint64     // pre-encoded identity (1) — M-residues
	oneN       []uint64     // pre-encoded identity (1) — N-residues
	numWindows int          // ceil(maxBits / k)
	maxBits    int
	tM, tN     int
}

// NewVROOMWindowTable builds the windowed precomputation table.
//
// maxBits should be >= the maximum exponent bit length you'll use.
// For 1024-bit exponents, use maxBits=1024.
//
// Precompute cost: (maxBits-1) squarings + 14*(maxBits/4) multiplies.
// For 1024-bit: 1023 + 3584 = 4607 VROOM calls (~3.7 ms one-time).
func NewVROOMWindowTable(base *big.Int, maxBits int, params *MontParamsStage4) *VROOMWindowTable {
	tM := len(params.BaseM.Moduli)
	tN := len(params.BaseN.Moduli)
	numWindows := (maxBits + windowK - 1) / windowK

	// Pre-encode identity
	rawOneM, rawOneN := ToVROOMEncodingStage4(big.NewInt(1), params)

	wt := &VROOMWindowTable{
		windowM:    make([][][]uint64, numWindows),
		windowN:    make([][][]uint64, numWindows),
		oneM:       make([]uint64, tM),
		oneN:       make([]uint64, tN),
		numWindows: numWindows,
		maxBits:    maxBits,
		tM:         tM,
		tN:         tN,
	}
	copy(wt.oneM, rawOneM)
	copy(wt.oneN, rawOneN)

	// First build the 1-bit table: base^(2^i) for i=0..maxBits-1
	// (same as VROOMPreTable — we reuse these as the j=1 entries)
	bitTableM := make([][]uint64, maxBits)
	bitTableN := make([][]uint64, maxBits)

	curM, curN := ToVROOMEncodingStage4(base, params)
	bitTableM[0] = make([]uint64, tM)
	bitTableN[0] = make([]uint64, tN)
	copy(bitTableM[0], curM)
	copy(bitTableN[0], curN)

	for i := 1; i < maxBits; i++ {
		rM, rN := VROOMStage4(curM, curN, curM, curN, params)
		bitTableM[i] = make([]uint64, tM)
		bitTableN[i] = make([]uint64, tN)
		copy(bitTableM[i], rM)
		copy(bitTableN[i], rN)
		curM = bitTableM[i]
		curN = bitTableN[i]
	}

	// For each window w, build entries j=1..15:
	//   windowTable[w][1] = base^(1 · 2^(4w)) = bitTable[4w]
	//   windowTable[w][j] = windowTable[w][j-1] · windowTable[w][1]
	for w := 0; w < numWindows; w++ {
		wt.windowM[w] = make([][]uint64, windowSize)
		wt.windowN[w] = make([][]uint64, windowSize)

		// j=0 unused (identity), allocate nil
		wt.windowM[w][0] = nil
		wt.windowN[w][0] = nil

		// j=1: base^(2^(4w))
		baseIdx := w * windowK
		wt.windowM[w][1] = make([]uint64, tM)
		wt.windowN[w][1] = make([]uint64, tN)
		if baseIdx < maxBits {
			copy(wt.windowM[w][1], bitTableM[baseIdx])
			copy(wt.windowN[w][1], bitTableN[baseIdx])
		} else {
			// beyond maxBits — entry is 1 (identity, won't be used)
			copy(wt.windowM[w][1], rawOneM)
			copy(wt.windowN[w][1], rawOneN)
		}

		// j=2..15: multiply previous by j=1 entry
		for j := 2; j < windowSize; j++ {
			rM, rN := VROOMStage4(
				wt.windowM[w][j-1], wt.windowN[w][j-1],
				wt.windowM[w][1], wt.windowN[w][1],
				params,
			)
			wt.windowM[w][j] = make([]uint64, tM)
			wt.windowN[w][j] = make([]uint64, tN)
			copy(wt.windowM[w][j], rM)
			copy(wt.windowN[w][j], rN)
		}
	}

	return wt
}

// extractWindow extracts k=4 bits from exp starting at bit position start.
// Returns a value in [0, 15].
func extractWindow(exp *big.Int, start int) uint {
	var v uint
	for bit := 0; bit < windowK; bit++ {
		if exp.Bit(start+bit) == 1 {
			v |= 1 << uint(bit)
		}
	}
	return v
}

// ============================================================================
// Runtime — windowed exponentiation
// ============================================================================

// ModExpWindowed computes base^exp using the windowed precomputed table.
// Zero allocations at runtime. NOT constant-time.
//
// Result is written to w.accM, w.accN.
// Panics if exp.BitLen() > wt.maxBits.
func ModExpWindowed(exp *big.Int, wt *VROOMWindowTable,
	w *ModExpWorkspace, params *MontParamsStage4) {

	if exp.Sign() == 0 {
		copy(w.accM, wt.oneM)
		copy(w.accN, wt.oneN)
		return
	}

	if exp.BitLen() > wt.maxBits {
		panic("modexp_windowed: exponent bit length exceeds table size")
	}

	// acc = 1
	copy(w.accM, wt.oneM)
	copy(w.accN, wt.oneN)

	// Scan windows LSB → MSB
	// For each window w, extract 4-bit digit, multiply if non-zero
	for win := 0; win < wt.numWindows; win++ {
		digit := extractWindow(exp, win*windowK)
		if digit == 0 {
			continue
		}
		rM, rN := VROOMStage4(w.accM, w.accN, wt.windowM[win][digit], wt.windowN[win][digit], params)
		copy(w.accM, rM)
		copy(w.accN, rN)
	}
}

// ModExpWindowedConstTime computes base^exp using the windowed precomputed table.
// Constant-time — no data-dependent branches.
//
// Uses constant-time table lookup: reads all 15 entries per window and
// selects the correct one without branching.
func ModExpWindowedConstTime(exp *big.Int, wt *VROOMWindowTable,
	w *ModExpWorkspace, params *MontParamsStage4) {

	if exp.Sign() == 0 {
		copy(w.accM, wt.oneM)
		copy(w.accN, wt.oneN)
		return
	}

	if exp.BitLen() > wt.maxBits {
		panic("modexp_windowed: exponent bit length exceeds table size")
	}

	// acc = 1
	copy(w.accM, wt.oneM)
	copy(w.accN, wt.oneN)

	// Temp buffer for the selected table entry — pre-allocated in workspace
	selM := w.selM
	selN := w.selN

	for win := 0; win < wt.numWindows; win++ {
		digit := extractWindow(exp, win*windowK)

		// Constant-time select: build selected entry without branching
		// sel = windowTable[win][digit], or identity if digit=0
		copy(selM, wt.oneM)
		copy(selN, wt.oneN)
		for j := 1; j < windowSize; j++ {
			// mask = all-ones if digit == j, else all-zeros
			mask := -uint64(1 - subtle_neq(uint64(digit), uint64(j)))
			for i := 0; i < wt.tM; i++ {
				selM[i] = (selM[i] &^ mask) | (wt.windowM[win][j][i] & mask)
			}
			for i := 0; i < wt.tN; i++ {
				selN[i] = (selN[i] &^ mask) | (wt.windowN[win][j][i] & mask)
			}
		}

		// Always multiply — acc * sel (sel = identity if digit=0)
		rM, rN := VROOMStage4(w.accM, w.accN, selM, selN, params)
		copy(w.accM, rM)
		copy(w.accN, rN)
	}
}

// subtle_neq returns 1 if a != b, 0 if a == b. Constant-time.
//
//go:nosplit
func subtle_neq(a, b uint64) uint64 {
	x := a ^ b
	return (x | -x) >> 63
}

// ============================================================================
// Convenience wrappers
// ============================================================================

// ModExpVROOMWindowed computes base^exp mod p using windowed precomputed table.
// Non-constant-time.
func ModExpVROOMWindowed(exp *big.Int, wt *VROOMWindowTable,
	w *ModExpWorkspace, params *MontParamsStage4) *big.Int {

	if exp.Sign() == 0 {
		return big.NewInt(1)
	}
	ModExpWindowed(exp, wt, w, params)
	return FromVROOMEncodingStage4(w.accM, params)
}

// ModExpVROOMWindowedConstTime computes base^exp mod p. Constant-time on exp.
func ModExpVROOMWindowedConstTime(exp *big.Int, wt *VROOMWindowTable,
	w *ModExpWorkspace, params *MontParamsStage4) *big.Int {

	if exp.Sign() == 0 {
		return big.NewInt(1)
	}
	ModExpWindowedConstTime(exp, wt, w, params)
	return FromVROOMEncodingStage4(w.accM, params)
}