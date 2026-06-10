package main

import (
	"crypto/rand"
	"math/big"
	"testing"
)

// ============================================================================
// 2048-bit correctness tests
// ============================================================================

func TestModExpVROOM_2048bit(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 2048)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)

	for i := 0; i < 10; i++ {
		base := randomInRange(p)
		exp := randomInRange(p)

		got := ModExpVROOM(base, exp, w, params)
		want := new(big.Int).Exp(base, exp, p)

		if got.Cmp(want) != 0 {
			t.Fatalf("test %d: got %s, want %s", i, got, want)
		}
	}
	t.Log("2048-bit: 10 random exponentiations correct")
}

func TestModExpPrecomputed_2048bit(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 2048)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)

	for i := 0; i < 10; i++ {
		base := randomInRange(p)
		exp := randomInRange(p)

		pt := NewVROOMPreTable(base, 2048, params)

		got := ModExpVROOMPrecomputed(exp, pt, w, params)
		want := new(big.Int).Exp(base, exp, p)

		if got.Cmp(want) != 0 {
			t.Fatalf("test %d: got %s, want %s", i, got, want)
		}
	}
	t.Log("2048-bit precomputed: 10 random exponentiations correct")
}

// ============================================================================
// 2048-bit benchmarks — multiplication baseline
// ============================================================================

func BenchmarkVROOMStage4_2048bit(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 2048)
	params := SetupRNSParamsStage4(p)
	a := randomInRange(p)
	c := randomInRange(p)
	aM, aN := ToVROOMEncodingStage4(a, params)
	cM, cN := ToVROOMEncodingStage4(c, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		VROOMStage4(aM, aN, cM, cN, params)
	}
}

func BenchmarkBigIntMul_2048bit(b *testing.B) {
	p, _ := rand.Prime(rand.Reader, 2048)
	a := randomInRange(p)
	c := randomInRange(p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := new(big.Int).Mul(a, c)
		r.Mod(r, p)
	}
}

// ============================================================================
// 2048-bit benchmarks — exponentiation
// ============================================================================

func BenchmarkModExpInner_2048bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 2048)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	baseM, baseN := ToVROOMEncodingStage4(base, params)
	exp, _ := rand.Int(rand.Reader, p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpInner(baseM, baseN, exp, w, params)
	}
}

func BenchmarkModExpPrecomputed_2048bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 2048)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	pt := NewVROOMPreTable(base, 2048, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpPrecomputed(exp, pt, w, params)
	}
}

func BenchmarkModExpPrecomputedConstTime_2048bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 2048)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	pt := NewVROOMPreTable(base, 2048, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpPrecomputedConstTime(exp, pt, w, params)
	}
}

func BenchmarkModExpBigInt_2048bit_exp(b *testing.B) {
	p, _ := rand.Prime(rand.Reader, 2048)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		new(big.Int).Exp(base, exp, p)
	}
}

func BenchmarkNewVROOMPreTable_2048(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 2048)
	params := SetupRNSParamsStage4(p)
	base := randomInRange(p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewVROOMPreTable(base, 2048, params)
	}
}
