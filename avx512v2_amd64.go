package main

// avx512v2_amd64.go — Go stubs for Stage 4 AVX512IFMA assembly kernels.
// These are implemented in avx512v2_amd64.s

// matvecAVX512_3g performs a full matrix-vector product for 3 groups (24 lanes).
//
// All 6 ZMM accumulators (3 lo + 3 hi) stay in registers across the entire
// source loop, eliminating per-group load/store traffic and Go call overhead.
//
// accLo, accHi: output arrays (≥24 elements, zeroed by kernel)
// r:            source residue vector [tFrom]
// aFlat:        row-major A matrix [tFrom × rowStride]
// tFrom:        number of source residues
// rowStride:    elements per row in aFlat (= padTTo ≥ 24)
//
// On Ice Lake: ~7-8 cycles per source residue (vs ~50+ with broadcastMulAcc52 calls)
//
//go:noescape
func matvecAVX512_3g(accLo, accHi, r, aFlat *uint64, tFrom, rowStride int)

// matvecAVX512_6g performs a full matrix-vector product for 6 groups (48 lanes).
//
// All 12 ZMM accumulators stay in registers. For larger primes (e.g. 2048-bit).
//
//go:noescape
func matvecAVX512_6g(accLo, accHi, r, aFlat *uint64, tFrom, rowStride int)

// matvecAVX512Gen performs a matrix-vector product for arbitrary padTTo.
//
// Processes one group (8 lanes) at a time with load/store, but eliminates
// Go function call overhead and shares the broadcast scalar across groups.
// OOO execution overlaps independent groups.
//
// Falls back to this for nGroups > 6.
// accLo and accHi must be pre-zeroed by the caller.
//
//go:noescape
func matvecAVX512Gen(accLo, accHi, r, aFlat *uint64, tFrom, padTTo int)
