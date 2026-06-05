#include "textflag.h"

// ============================================================================
// Stage 4 AVX512IFMA assembly kernels
//
// These kernels address optimization #4: instruction interleaving.
// By keeping all accumulators in ZMM registers across the entire source loop,
// we eliminate:
//   1. Go function call overhead (was 63 calls to broadcastMulAcc52)
//   2. Load/store traffic for accumulators (was 126 loads + 126 stores)
//   3. Latency stalls (6 independent VPMADD52 per iteration hide 4-cycle latency)
//
// Register convention (safe for Go asm — only Z0-Z15 and X0-X15 used):
//   Z9:     broadcast scalar (via MOVQ AX,X9 + VPBROADCASTQ X9,Z9)
//   Z6-Z8:  transient A matrix loads (reused across group batches)
// ============================================================================

// func matvecAVX512_3g(accLo, accHi, r, aFlat *uint64, tFrom, rowStride int)
//
// Specialized 3-group (24 lanes) matrix-vector product kernel.
// All 6 accumulators (Z0-Z2 lo, Z3-Z5 hi) stay in ZMM registers.
//
// Register allocation:
//   Z0-Z2:  accLo for groups 0, 1, 2  (persistent across loop)
//   Z3-Z5:  accHi for groups 0, 1, 2  (persistent across loop)
//   Z6-Z8:  A matrix row loads         (transient per iteration)
//   Z9:     broadcast r[i]             (transient per iteration)
//
// 6 independent VPMADD52 per iteration → 3 cycle throughput on ICL (2/cycle).
// Total: ~7-8 cycles/source residue (broadcast + 3 loads + 6 VPMADD52 + loop).
TEXT ·matvecAVX512_3g(SB), NOSPLIT, $0-48
	MOVQ accLo+0(FP), R8
	MOVQ accHi+8(FP), R9
	MOVQ r+16(FP), R10
	MOVQ aFlat+24(FP), R11
	MOVQ tFrom+32(FP), R12
	MOVQ rowStride+40(FP), R13

	// Row stride in bytes
	SHLQ $3, R13

	// Zero all 6 accumulators
	VPXORQ Z0, Z0, Z0
	VPXORQ Z1, Z1, Z1
	VPXORQ Z2, Z2, Z2
	VPXORQ Z3, Z3, Z3
	VPXORQ Z4, Z4, Z4
	VPXORQ Z5, Z5, Z5

	TESTQ R12, R12
	JZ    done_3g

loop_3g:
	// Broadcast r[i] → Z9 (via X9, safe: Z9 not used as accumulator)
	MOVQ    (R10), AX
	MOVQ    AX, X9
	VPBROADCASTQ X9, Z9

	// Load 3 groups of A[i]
	VMOVDQU64 0(R11), Z6
	VMOVDQU64 64(R11), Z7
	VMOVDQU64 128(R11), Z8

	// 6 independent VPMADD52: all different destination registers
	// → zero data dependencies → OOO dispatches 2/cycle → 3 cycles
	VPMADD52LUQ Z6, Z9, Z0
	VPMADD52HUQ Z6, Z9, Z3
	VPMADD52LUQ Z7, Z9, Z1
	VPMADD52HUQ Z7, Z9, Z4
	VPMADD52LUQ Z8, Z9, Z2
	VPMADD52HUQ Z8, Z9, Z5

	ADDQ $8, R10
	ADDQ R13, R11
	DECQ R12
	JNZ  loop_3g

done_3g:
	// Store accumulators — one-time cost at the end
	VMOVDQU64 Z0, 0(R8)
	VMOVDQU64 Z1, 64(R8)
	VMOVDQU64 Z2, 128(R8)
	VMOVDQU64 Z3, 0(R9)
	VMOVDQU64 Z4, 64(R9)
	VMOVDQU64 Z5, 128(R9)

	VZEROUPPER
	RET

// ============================================================================
// func matvecAVX512_6g(accLo, accHi, r, aFlat *uint64, tFrom, rowStride int)
//
// Specialized 6-group (48 lanes) kernel for larger primes (2048-bit).
// 12 ZMM accumulators stay in registers. Processes groups in 2 batches
// of 3 to keep all registers in the Z0-Z15 range (Go asm safe).
//
// Register allocation:
//   Z0-Z5:   accLo for groups 0-5  (persistent)
//   Z10-Z15: accHi for groups 0-5  (persistent)
//   Z6-Z8:   A matrix loads         (transient, loaded twice per source)
//   Z9:      broadcast scalar        (transient)
//
// 12 VPMADD52 per iteration in two batches of 6. OOO overlaps batches.
TEXT ·matvecAVX512_6g(SB), NOSPLIT, $0-48
	MOVQ accLo+0(FP), R8
	MOVQ accHi+8(FP), R9
	MOVQ r+16(FP), R10
	MOVQ aFlat+24(FP), R11
	MOVQ tFrom+32(FP), R12
	MOVQ rowStride+40(FP), R13
	SHLQ $3, R13

	// Zero 12 accumulators
	VPXORQ Z0, Z0, Z0
	VPXORQ Z1, Z1, Z1
	VPXORQ Z2, Z2, Z2
	VPXORQ Z3, Z3, Z3
	VPXORQ Z4, Z4, Z4
	VPXORQ Z5, Z5, Z5
	VPXORQ Z10, Z10, Z10
	VPXORQ Z11, Z11, Z11
	VPXORQ Z12, Z12, Z12
	VPXORQ Z13, Z13, Z13
	VPXORQ Z14, Z14, Z14
	VPXORQ Z15, Z15, Z15

	TESTQ R12, R12
	JZ    done_6g

loop_6g:
	MOVQ    (R10), AX
	MOVQ    AX, X9
	VPBROADCASTQ X9, Z9

	// ── Batch 1: groups 0-2 ──
	VMOVDQU64 0(R11), Z6
	VMOVDQU64 64(R11), Z7
	VMOVDQU64 128(R11), Z8

	VPMADD52LUQ Z6, Z9, Z0
	VPMADD52HUQ Z6, Z9, Z10
	VPMADD52LUQ Z7, Z9, Z1
	VPMADD52HUQ Z7, Z9, Z11
	VPMADD52LUQ Z8, Z9, Z2
	VPMADD52HUQ Z8, Z9, Z12

	// ── Batch 2: groups 3-5 (reuse Z6-Z8) ──
	VMOVDQU64 192(R11), Z6
	VMOVDQU64 256(R11), Z7
	VMOVDQU64 320(R11), Z8

	VPMADD52LUQ Z6, Z9, Z3
	VPMADD52HUQ Z6, Z9, Z13
	VPMADD52LUQ Z7, Z9, Z4
	VPMADD52HUQ Z7, Z9, Z14
	VPMADD52LUQ Z8, Z9, Z5
	VPMADD52HUQ Z8, Z9, Z15

	ADDQ $8, R10
	ADDQ R13, R11
	DECQ R12
	JNZ  loop_6g

done_6g:
	VMOVDQU64 Z0, 0(R8)
	VMOVDQU64 Z1, 64(R8)
	VMOVDQU64 Z2, 128(R8)
	VMOVDQU64 Z3, 192(R8)
	VMOVDQU64 Z4, 256(R8)
	VMOVDQU64 Z5, 320(R8)
	VMOVDQU64 Z10, 0(R9)
	VMOVDQU64 Z11, 64(R9)
	VMOVDQU64 Z12, 128(R9)
	VMOVDQU64 Z13, 192(R9)
	VMOVDQU64 Z14, 256(R9)
	VMOVDQU64 Z15, 320(R9)

	VZEROUPPER
	RET

// ============================================================================
// func matvecAVX512Gen(accLo, accHi, r, aFlat *uint64, tFrom, padTTo int)
//
// Generic matvec kernel for arbitrary padTTo (any number of groups).
// Processes one group at a time with memory-resident accumulators,
// but eliminates Go function-call overhead and shares broadcast across groups.
//
// accLo and accHi MUST be pre-zeroed by the caller.
//
// Register allocation:
//   Z0, Z1: accLo, accHi for current group (transient)
//   Z2:     A matrix load                    (transient)
//   Z9:     broadcast scalar                 (persistent within outer iter)
TEXT ·matvecAVX512Gen(SB), NOSPLIT, $0-48
	MOVQ accLo+0(FP), R8
	MOVQ accHi+8(FP), R9
	MOVQ r+16(FP), R10
	MOVQ aFlat+24(FP), R11
	MOVQ tFrom+32(FP), R12
	MOVQ padTTo+40(FP), R13

	// Row stride and inner limit in bytes
	MOVQ R13, R14
	SHLQ $3, R14
	MOVQ R14, R15

	TESTQ R12, R12
	JZ    done_gen

outer_gen:
	MOVQ    (R10), AX
	MOVQ    AX, X9
	VPBROADCASTQ X9, Z9

	XORQ CX, CX

inner_gen:
	CMPQ CX, R15
	JGE  next_src_gen

	VMOVDQU64 (R8)(CX*1), Z0
	VMOVDQU64 (R9)(CX*1), Z1
	VMOVDQU64 (R11)(CX*1), Z2

	VPMADD52LUQ Z2, Z9, Z0
	VPMADD52HUQ Z2, Z9, Z1

	VMOVDQU64 Z0, (R8)(CX*1)
	VMOVDQU64 Z1, (R9)(CX*1)

	ADDQ $64, CX
	JMP  inner_gen

next_src_gen:
	ADDQ $8, R10
	ADDQ R14, R11
	DECQ R12
	JNZ  outer_gen

done_gen:
	VZEROUPPER
	RET
