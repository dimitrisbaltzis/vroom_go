package vroom

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"testing"
)

// ============================================================================
// Correctness tests — Precomputed table
// ============================================================================

func TestModExpPrecomputed_Small(t *testing.T) {
	skipWithoutAVX512ME(t)

	p := big.NewInt(7)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)

	base := big.NewInt(3)
	pt := NewVROOMPreTable(base, 16, params)

	// 3^13 mod 7 = 3
	got := ModExpVROOMPrecomputed(big.NewInt(13), pt, w, params)
	want := big.NewInt(3)
	if got.Cmp(want) != 0 {
		t.Fatalf("3^13 mod 7: got %s, want %s", got, want)
	}

	// 3^0 mod 7 = 1
	got = ModExpVROOMPrecomputed(big.NewInt(0), pt, w, params)
	if got.Cmp(big.NewInt(1)) != 0 {
		t.Fatalf("3^0 mod 7: got %s, want 1", got)
	}

	// 3^1 mod 7 = 3
	got = ModExpVROOMPrecomputed(big.NewInt(1), pt, w, params)
	if got.Cmp(big.NewInt(3)) != 0 {
		t.Fatalf("3^1 mod 7: got %s, want 3", got)
	}

	t.Log("Small values: correct")
}

func TestModExpPrecomputed_EdgeCases(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 256)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	a := randomInRange(p)
	pt := NewVROOMPreTable(a, 256, params)

	// a^0 = 1
	got := ModExpVROOMPrecomputed(big.NewInt(0), pt, w, params)
	if got.Cmp(big.NewInt(1)) != 0 {
		t.Fatalf("a^0: got %s, want 1", got)
	}

	// a^1 = a
	got = ModExpVROOMPrecomputed(big.NewInt(1), pt, w, params)
	if got.Cmp(a) != 0 {
		t.Fatalf("a^1: got %s, want %s", got, a)
	}

	// a^2 = a*a mod p
	got = ModExpVROOMPrecomputed(big.NewInt(2), pt, w, params)
	want := new(big.Int).Mul(a, a)
	want.Mod(want, p)
	if got.Cmp(want) != 0 {
		t.Fatalf("a^2: got %s, want %s", got, want)
	}

	// Fermat: a^(p-1) = 1 mod p
	pMinus1 := new(big.Int).Sub(p, bigOne)
	got = ModExpVROOMPrecomputed(pMinus1, pt, w, params)
	if got.Cmp(big.NewInt(1)) != 0 {
		t.Fatalf("a^(p-1) mod p: got %s, want 1 (Fermat)", got)
	}

	t.Log("Edge cases: correct")
}

func TestModExpPrecomputed_Random(t *testing.T) {
	skipWithoutAVX512ME(t)

	for _, bits := range []int{64, 128, 256, 512, 1024, 2048} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)
			w := NewModExpWorkspace(params)

			for i := 0; i < 20; i++ {
				base := randomInRange(p)
				exp := randomInRange(p)

				pt := NewVROOMPreTable(base, bits, params)

				got := ModExpVROOMPrecomputed(exp, pt, w, params)
				want := new(big.Int).Exp(base, exp, p)

				if got.Cmp(want) != 0 {
					t.Fatalf("test %d: got %s, want %s", i, got, want)
				}
			}
			t.Logf("%d-bit: 20 random precomputed exponentiations correct", bits)
		})
	}
}

func TestModExpPrecomputed_MatchesNaive(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 512)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)

	for i := 0; i < 30; i++ {
		base := randomInRange(p)
		exp := randomInRange(p)

		pt := NewVROOMPreTable(base, 512, params)

		got1 := ModExpVROOM(base, exp, w, params)
		got2 := ModExpVROOMPrecomputed(exp, pt, w, params)

		if got1.Cmp(got2) != 0 {
			t.Fatalf("test %d: naive=%s, precomputed=%s", i, got1, got2)
		}
	}
	t.Log("30 tests: precomputed matches naive")
}

func TestModExpPrecomputed_RSAPublicExponent(t *testing.T) {
	skipWithoutAVX512ME(t)

	e := big.NewInt(65537)

	for _, bits := range []int{256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)
			w := NewModExpWorkspace(params)

			for i := 0; i < 10; i++ {
				base := randomInRange(p)
				pt := NewVROOMPreTable(base, bits, params)

				got := ModExpVROOMPrecomputed(e, pt, w, params)
				want := new(big.Int).Exp(base, e, p)

				if got.Cmp(want) != 0 {
					t.Fatalf("test %d: got %s, want %s", i, got, want)
				}
			}
			t.Logf("%d-bit: 10 RSA-style precomputed exponentiations correct", bits)
		})
	}
}

func TestModExpPrecomputed_ReuseTable(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 512)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)

	// Build table once, use for many different exponents
	pt := NewVROOMPreTable(base, 512, params)

	for i := 0; i < 50; i++ {
		exp := randomInRange(p)

		got := ModExpVROOMPrecomputed(exp, pt, w, params)
		want := new(big.Int).Exp(base, exp, p)

		if got.Cmp(want) != 0 {
			t.Fatalf("test %d: got %s, want %s", i, got, want)
		}
	}
	t.Log("50 tests: single table, many exponents, all correct")
}

// ============================================================================
// Correctness tests — Precomputed constant-time
// ============================================================================

func TestModExpPrecomputedConstTime_Random(t *testing.T) {
	skipWithoutAVX512ME(t)

	for _, bits := range []int{64, 128, 256, 512, 1024, 2048} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)
			w := NewModExpWorkspace(params)

			for i := 0; i < 20; i++ {
				base := randomInRange(p)
				exp := randomInRange(p)

				pt := NewVROOMPreTable(base, bits, params)

				got := ModExpVROOMPrecomputedConstTime(exp, pt, w, params)
				want := new(big.Int).Exp(base, exp, p)

				if got.Cmp(want) != 0 {
					t.Fatalf("test %d: got %s, want %s", i, got, want)
				}
			}
			t.Logf("%d-bit: 20 CT precomputed exponentiations correct", bits)
		})
	}
}

func TestModExpPrecomputedConstTime_MatchesNonConstTime(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 512)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)

	for i := 0; i < 30; i++ {
		base := randomInRange(p)
		exp := randomInRange(p)

		pt := NewVROOMPreTable(base, 512, params)

		got1 := ModExpVROOMPrecomputed(exp, pt, w, params)
		got2 := ModExpVROOMPrecomputedConstTime(exp, pt, w, params)

		if got1.Cmp(got2) != 0 {
			t.Fatalf("test %d: non-CT=%s, CT=%s", i, got1, got2)
		}
	}
	t.Log("30 tests: CT precomputed matches non-CT precomputed")
}

func TestModExpPrecomputedConstTime_Fermat(t *testing.T) {
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
		pt := NewVROOMPreTable(a, 256, params)

		got := ModExpVROOMPrecomputedConstTime(pMinus1, pt, w, params)
		if got.Cmp(big.NewInt(1)) != 0 {
			t.Fatalf("test %d: a^(p-1) mod p = %s, want 1", i, got)
		}
	}
	t.Log("Fermat's little theorem (CT precomputed): 10 tests correct")
}

// ============================================================================
// Benchmarks — Precomputed table build
// ============================================================================

func BenchmarkNewVROOMPreTable_1024(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	base := randomInRange(p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewVROOMPreTable(base, 1024, params)
	}
}

// ============================================================================
// Benchmarks — Precomputed runtime (the fast part)
// ============================================================================

func BenchmarkModExpPrecomputed_1024bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	pt := NewVROOMPreTable(base, 1024, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpPrecomputed(exp, pt, w, params)
	}
}

func BenchmarkModExpPrecomputedConstTime_1024bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	pt := NewVROOMPreTable(base, 1024, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpPrecomputedConstTime(exp, pt, w, params)
	}
}

func BenchmarkModExpPrecomputed_RSAVerify_1024(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	e := big.NewInt(65537)

	pt := NewVROOMPreTable(base, 1024, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpPrecomputed(e, pt, w, params)
	}
}

// ============================================================================
// Benchmarks — Full (with decode) for comparison
// ============================================================================

func BenchmarkModExpVROOMPrecomputed_1024bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	pt := NewVROOMPreTable(base, 1024, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpVROOMPrecomputed(exp, pt, w, params)
	}
}
