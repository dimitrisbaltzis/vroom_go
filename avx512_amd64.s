#include "textflag.h"

// func vpmadd52luq(dst, a, b *uint64)
// dst[i] += low52(a[i] * b[i]) for i = 0..7
// Inputs are treated as 52-bit unsigned integers.
// The 64-bit accumulator (dst) can overflow beyond 52 bits.
TEXT ·vpmadd52luq(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), DX
	VMOVDQU64 (DI), Z0        // Z0 = dst[0..7]
	VMOVDQU64 (SI), Z1        // Z1 = a[0..7]
	VMOVDQU64 (DX), Z2        // Z2 = b[0..7]
	VPMADD52LUQ Z2, Z1, Z0    // Z0 += low52(Z1[i] * Z2[i])
	VMOVDQU64 Z0, (DI)        // store result
	VZEROUPPER
	RET

// func vpmadd52huq(dst, a, b *uint64)
// dst[i] += high52(a[i] * b[i]) for i = 0..7
// high52 = bits [103:52] of the 104-bit product.
TEXT ·vpmadd52huq(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), DX
	VMOVDQU64 (DI), Z0        // Z0 = dst[0..7]
	VMOVDQU64 (SI), Z1        // Z1 = a[0..7]
	VMOVDQU64 (DX), Z2        // Z2 = b[0..7]
	VPMADD52HUQ Z2, Z1, Z0    // Z0 += high52(Z1[i] * Z2[i])
	VMOVDQU64 Z0, (DI)        // store result
	VZEROUPPER
	RET

// func broadcastMulAccLo52(dst *uint64, scalar uint64, b *uint64)
// dst[i] += low52(scalar * b[i]) for i = 0..7
// scalar is broadcast to all 8 lanes before multiplication.
// This is the key operation for CRNS matrix-vector product.
TEXT ·broadcastMulAccLo52(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ scalar+8(FP), AX
	MOVQ b+16(FP), DX
	VMOVDQU64 (DI), Z0        // Z0 = dst[0..7] (accumulator)
	MOVQ AX, X1               // X1 = scalar (low 64 bits of XMM1)
	VPBROADCASTQ X1, Z1       // Z1 = [scalar, scalar, ..., scalar]
	VMOVDQU64 (DX), Z2        // Z2 = b[0..7]
	VPMADD52LUQ Z2, Z1, Z0    // Z0 += low52(scalar * b[i])
	VMOVDQU64 Z0, (DI)        // store result
	VZEROUPPER
	RET

// func broadcastMulAcc52(lo, hi *uint64, scalar uint64, b *uint64)
// lo[i] += low52(scalar * b[i])  for i = 0..7
// hi[i] += high52(scalar * b[i]) for i = 0..7
// Combined lo+hi accumulation in one call: 2 VPMADD52 instructions,
// shared broadcast and load. This is the core CRNS kernel for Stage 3.
TEXT ·broadcastMulAcc52(SB), NOSPLIT, $0-32
	MOVQ lo+0(FP), DI
	MOVQ hi+8(FP), SI
	MOVQ scalar+16(FP), AX
	MOVQ b+24(FP), DX
	VMOVDQU64 (DI), Z0        // Z0 = lo[0..7]
	VMOVDQU64 (SI), Z3        // Z3 = hi[0..7]
	MOVQ AX, X1               // X1 = scalar
	VPBROADCASTQ X1, Z1       // Z1 = [scalar, ..., scalar]
	VMOVDQU64 (DX), Z2        // Z2 = b[0..7]
	VPMADD52LUQ Z2, Z1, Z0    // Z0 += low52(scalar * b[i])
	VPMADD52HUQ Z2, Z1, Z3    // Z3 += high52(scalar * b[i])
	VMOVDQU64 Z0, (DI)        // store lo
	VMOVDQU64 Z3, (SI)        // store hi
	VZEROUPPER
	RET
