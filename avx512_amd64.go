package main

// avx512_amd64.go — Go stubs for AVX512IFMA assembly functions.
// These are implemented in avx512_amd64.s

// vpmadd52luq performs: dst[i] += low52(a[i] * b[i]) for i = 0..7
// All three pointers must reference at least 8 contiguous uint64 values (64 bytes, 512 bits).
// Requires AVX512IFMA (Ice Lake+, Sapphire Rapids, or SDE emulation).
//
//go:noescape
func vpmadd52luq(dst, a, b *uint64)

// vpmadd52huq performs: dst[i] += high52(a[i] * b[i]) for i = 0..7
// high52 means bits [103:52] of the 104-bit product.
//
//go:noescape
func vpmadd52huq(dst, a, b *uint64)

// broadcastMulAccLo52 performs: dst[i] += low52(scalar * b[i]) for i = 0..7
// scalar is broadcast to all 8 lanes, then multiplied with b[i].
//
//go:noescape
func broadcastMulAccLo52(dst *uint64, scalar uint64, b *uint64)

// broadcastMulAcc52 performs both low and high accumulation in one call:
//   lo[i] += low52(scalar * b[i])  for i = 0..7
//   hi[i] += high52(scalar * b[i]) for i = 0..7
// This is the core CRNS kernel: 2 VPMADD52 instructions with shared broadcast.
//
//go:noescape
func broadcastMulAcc52(lo, hi *uint64, scalar uint64, b *uint64)
