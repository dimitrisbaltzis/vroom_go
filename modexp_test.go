package main

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"os"
	"testing"
)

func skipWithoutAVX512ME(t *testing.T) {
	if os.Getenv("AVX512_TEST") == "" {
		t.Skip("Set AVX512_TEST=1 (needs AVX512IFMA or SDE)")
	}
}

func skipWithoutAVX512MEB(b *testing.B) {
	if os.Getenv("AVX512_TEST") == "" {
		b.Skip("Set AVX512_TEST=1 (needs AVX512IFMA or SDE)")
	}
}

// ============================================================================
// Correctness tests
// ============================================================================

func TestModExpVROOM_Small(t *testing.T) {
	skipWithoutAVX512ME(t)

	p := big.NewInt(7)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)

	result := ModExpVROOM(big.NewInt(3), big.NewInt(13), w, params)
	expected := big.NewInt(3)
	if result.Cmp(expected) != 0 {
		t.Fatalf("3^13 mod 7: got %s, want %s", result, expected)
	}

	p2, _ := rand.Prime(rand.Reader, 16)
	params2 := SetupRNSParamsStage4(p2)
	w2 := NewModExpWorkspace(params2)
	base := big.NewInt(2)
	exp := big.NewInt(10)
	got := ModExpVROOM(base, exp, w2, params2)
	want := new(big.Int).Exp(base, exp, p2)
	if got.Cmp(want) != 0 {
		t.Fatalf("2^10 mod %s: got %s, want %s", p2, got, want)
	}
	t.Log("Small values: correct")
}

func TestModExpVROOM_EdgeCases(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 256)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	a := randomInRange(p)

	got := ModExpVROOM(a, big.NewInt(0), w, params)
	if got.Cmp(big.NewInt(1)) != 0 {
		t.Fatalf("a^0: got %s, want 1", got)
	}

	got = ModExpVROOM(a, big.NewInt(1), w, params)
	if got.Cmp(a) != 0 {
		t.Fatalf("a^1: got %s, want %s", got, a)
	}

	got = ModExpVROOM(a, big.NewInt(2), w, params)
	want := new(big.Int).Mul(a, a)
	want.Mod(want, p)
	if got.Cmp(want) != 0 {
		t.Fatalf("a^2: got %s, want %s", got, want)
	}

	pMinus1 := new(big.Int).Sub(p, bigOne)
	got = ModExpVROOM(a, pMinus1, w, params)
	if got.Cmp(big.NewInt(1)) != 0 {
		t.Fatalf("a^(p-1) mod p: got %s, want 1 (Fermat)", got)
	}
	t.Log("Edge cases: correct")
}

func TestModExpVROOM_Random(t *testing.T) {
	skipWithoutAVX512ME(t)

	for _, bits := range []int{64, 128, 256, 512, 1024, 2048} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)
			w := NewModExpWorkspace(params)

			for i := 0; i < 20; i++ {
				base := randomInRange(p)
				exp := randomInRange(p)

				got := ModExpVROOM(base, exp, w, params)
				want := new(big.Int).Exp(base, exp, p)

				if got.Cmp(want) != 0 {
					t.Fatalf("test %d: got %s, want %s", i, got, want)
				}
			}
			t.Logf("%d-bit: 20 random exponentiations correct", bits)
		})
	}
}

func TestModExpVROOM_RSAPublicExponent(t *testing.T) {
	skipWithoutAVX512ME(t)

	e := big.NewInt(65537)

	for _, bits := range []int{256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)
			w := NewModExpWorkspace(params)

			for i := 0; i < 10; i++ {
				base := randomInRange(p)
				got := ModExpVROOM(base, e, w, params)
				want := new(big.Int).Exp(base, e, p)

				if got.Cmp(want) != 0 {
					t.Fatalf("test %d: got %s, want %s", i, got, want)
				}
			}
			t.Logf("%d-bit: 10 RSA-style exponentiations correct", bits)
		})
	}
}

func TestModExpVROOMConstTime_Random(t *testing.T) {
	skipWithoutAVX512ME(t)

	for _, bits := range []int{64, 128, 256, 512, 1024, 2048} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)
			w := NewModExpWorkspace(params)

			for i := 0; i < 20; i++ {
				base := randomInRange(p)
				exp := randomInRange(p)

				got := ModExpVROOMConstTime(base, exp, w, params)
				want := new(big.Int).Exp(base, exp, p)

				if got.Cmp(want) != 0 {
					t.Fatalf("test %d: got %s, want %s", i, got, want)
				}
			}
			t.Logf("%d-bit: 20 constant-time exponentiations correct", bits)
		})
	}
}

func TestModExpVROOMConstTime_MatchesNonConstTime(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 512)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)

	for i := 0; i < 30; i++ {
		base := randomInRange(p)
		exp := randomInRange(p)

		got1 := ModExpVROOM(base, exp, w, params)
		got2 := ModExpVROOMConstTime(base, exp, w, params)

		if got1.Cmp(got2) != 0 {
			t.Fatalf("test %d: non-CT=%s, CT=%s", i, got1, got2)
		}
	}
	t.Log("30 tests: constant-time matches non-constant-time")
}

func TestModExpVROOMConstTime_Fermat(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 256)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	pMinus1 := new(big.Int).Sub(p, bigOne)

	for i := 0; i < 10; i++ {
		a := randomInRange(p)
		if a.Sign() == 0 {
			continue
		}
		got := ModExpVROOMConstTime(a, pMinus1, w, params)
		if got.Cmp(big.NewInt(1)) != 0 {
			t.Fatalf("test %d: a^(p-1) mod p = %s, want 1", i, got)
		}
	}
	t.Log("Fermat's little theorem: 10 tests correct")
}

// ============================================================================
// Benchmarks — Full (encode + exp + decode)
// ============================================================================

func BenchmarkModExpVROOM_RSAVerify_1024(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	e := big.NewInt(65537)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpVROOM(base, e, w, params)
	}
}

func BenchmarkModExpVROOM_1024bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpVROOM(base, exp, w, params)
	}
}

func BenchmarkModExpVROOMConstTime_1024bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpVROOMConstTime(base, exp, w, params)
	}
}

// ============================================================================
// Benchmarks — Inner only (zero-alloc, pre-encoded)
// ============================================================================

func BenchmarkModExpInner_RSAVerify_1024(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	baseM, baseN := ToVROOMEncodingStage4(base, params)
	e := big.NewInt(65537)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpInner(baseM, baseN, e, w, params)
	}
}

func BenchmarkModExpInner_1024bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
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

func BenchmarkModExpInnerConstTime_1024bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	baseM, baseN := ToVROOMEncodingStage4(base, params)
	exp, _ := rand.Int(rand.Reader, p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpInnerConstTime(baseM, baseN, exp, w, params)
	}
}

// ============================================================================
// Baseline
// ============================================================================

func BenchmarkModExpBigInt_1024bit_exp(b *testing.B) {
	p, _ := rand.Prime(rand.Reader, 1024)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		new(big.Int).Exp(base, exp, p)
	}
}
