package main

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"testing"
)

// ============================================================================
// Correctness tests
// ============================================================================

func TestModExpWindowed_Small(t *testing.T) {
	skipWithoutAVX512ME(t)

	p := big.NewInt(7)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)

	base := big.NewInt(3)
	wt := NewVROOMWindowTable(base, 16, params)

	// 3^13 mod 7 = 3
	got := ModExpVROOMWindowed(big.NewInt(13), wt, w, params)
	if got.Cmp(big.NewInt(3)) != 0 {
		t.Fatalf("3^13 mod 7: got %s, want 3", got)
	}

	// 3^0 = 1
	got = ModExpVROOMWindowed(big.NewInt(0), wt, w, params)
	if got.Cmp(big.NewInt(1)) != 0 {
		t.Fatalf("3^0: got %s, want 1", got)
	}

	// 3^1 = 3
	got = ModExpVROOMWindowed(big.NewInt(1), wt, w, params)
	if got.Cmp(big.NewInt(3)) != 0 {
		t.Fatalf("3^1: got %s, want 3", got)
	}

	t.Log("Small values: correct")
}

func TestModExpWindowed_EdgeCases(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 256)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	a := randomInRange(p)
	wt := NewVROOMWindowTable(a, 256, params)

	// a^0 = 1
	got := ModExpVROOMWindowed(big.NewInt(0), wt, w, params)
	if got.Cmp(big.NewInt(1)) != 0 {
		t.Fatalf("a^0: got %s, want 1", got)
	}

	// a^1 = a
	got = ModExpVROOMWindowed(big.NewInt(1), wt, w, params)
	if got.Cmp(a) != 0 {
		t.Fatalf("a^1: got %s, want %s", got, a)
	}

	// a^2 = a*a mod p
	got = ModExpVROOMWindowed(big.NewInt(2), wt, w, params)
	want := new(big.Int).Mul(a, a)
	want.Mod(want, p)
	if got.Cmp(want) != 0 {
		t.Fatalf("a^2: got %s, want %s", got, want)
	}

	// Fermat: a^(p-1) = 1 mod p
	pMinus1 := new(big.Int).Sub(p, bigOne)
	got = ModExpVROOMWindowed(pMinus1, wt, w, params)
	if got.Cmp(big.NewInt(1)) != 0 {
		t.Fatalf("a^(p-1): got %s, want 1", got)
	}

	t.Log("Edge cases: correct")
}

func TestModExpWindowed_Random(t *testing.T) {
	skipWithoutAVX512ME(t)

	for _, bits := range []int{64, 128, 256, 512, 1024, 2048} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)
			w := NewModExpWorkspace(params)

			for i := 0; i < 20; i++ {
				base := randomInRange(p)
				exp := randomInRange(p)

				wt := NewVROOMWindowTable(base, bits, params)
				got := ModExpVROOMWindowed(exp, wt, w, params)
				want := new(big.Int).Exp(base, exp, p)

				if got.Cmp(want) != 0 {
					t.Fatalf("test %d: got %s, want %s", i, got, want)
				}
			}
			t.Logf("%d-bit: 20 windowed exponentiations correct", bits)
		})
	}
}

func TestModExpWindowed_MatchesPrecomputed(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 512)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)

	for i := 0; i < 30; i++ {
		base := randomInRange(p)
		exp := randomInRange(p)

		pt := NewVROOMPreTable(base, 512, params)
		wt := NewVROOMWindowTable(base, 512, params)

		got1 := ModExpVROOMPrecomputed(exp, pt, w, params)
		got2 := ModExpVROOMWindowed(exp, wt, w, params)

		if got1.Cmp(got2) != 0 {
			t.Fatalf("test %d: precomputed=%s, windowed=%s", i, got1, got2)
		}
	}
	t.Log("30 tests: windowed matches precomputed")
}

func TestModExpWindowed_ReuseTable(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 512)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	wt := NewVROOMWindowTable(base, 512, params)

	for i := 0; i < 50; i++ {
		exp := randomInRange(p)
		got := ModExpVROOMWindowed(exp, wt, w, params)
		want := new(big.Int).Exp(base, exp, p)
		if got.Cmp(want) != 0 {
			t.Fatalf("test %d: got %s, want %s", i, got, want)
		}
	}
	t.Log("50 tests: single windowed table, many exponents, all correct")
}

func TestModExpWindowed_Fermat(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 256)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	pMinus1 := new(big.Int).Sub(p, bigOne)

	for i := 0; i < 10; i++ {
		a := randomInRange(p)
		wt := NewVROOMWindowTable(a, 256, params)
		got := ModExpVROOMWindowed(pMinus1, wt, w, params)
		if got.Cmp(big.NewInt(1)) != 0 {
			t.Fatalf("test %d: a^(p-1) mod p = %s, want 1", i, got)
		}
	}
	t.Log("Fermat's little theorem: 10 tests correct")
}

// ============================================================================
// Constant-time tests
// ============================================================================

func TestModExpWindowedConstTime_Random(t *testing.T) {
	skipWithoutAVX512ME(t)

	for _, bits := range []int{64, 128, 256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)
			w := NewModExpWorkspace(params)

			for i := 0; i < 20; i++ {
				base := randomInRange(p)
				exp := randomInRange(p)

				wt := NewVROOMWindowTable(base, bits, params)
				got := ModExpVROOMWindowedConstTime(exp, wt, w, params)
				want := new(big.Int).Exp(base, exp, p)

				if got.Cmp(want) != 0 {
					t.Fatalf("test %d: got %s, want %s", i, got, want)
				}
			}
			t.Logf("%d-bit: 20 CT windowed exponentiations correct", bits)
		})
	}
}

func TestModExpWindowedConstTime_MatchesNonCT(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 512)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)

	for i := 0; i < 30; i++ {
		base := randomInRange(p)
		exp := randomInRange(p)
		wt := NewVROOMWindowTable(base, 512, params)

		got1 := ModExpVROOMWindowed(exp, wt, w, params)
		got2 := ModExpVROOMWindowedConstTime(exp, wt, w, params)

		if got1.Cmp(got2) != 0 {
			t.Fatalf("test %d: non-CT=%s, CT=%s", i, got1, got2)
		}
	}
	t.Log("30 tests: CT windowed matches non-CT windowed")
}

// ============================================================================
// Benchmarks
// ============================================================================

func BenchmarkNewVROOMWindowTable_1024(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	base := randomInRange(p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewVROOMWindowTable(base, 1024, params)
	}
}

func BenchmarkModExpWindowed_1024bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)
	wt := NewVROOMWindowTable(base, 1024, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpWindowed(exp, wt, w, params)
	}
}

func BenchmarkModExpWindowedConstTime_1024bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)
	wt := NewVROOMWindowTable(base, 1024, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpWindowedConstTime(exp, wt, w, params)
	}
}

func BenchmarkModExpWindowed_RSAVerify_1024(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	e := big.NewInt(65537)
	wt := NewVROOMWindowTable(base, 1024, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpWindowed(e, wt, w, params)
	}
}

func BenchmarkNewVROOMWindowTable_2048(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 2048)
	params := SetupRNSParamsStage4(p)
	base := randomInRange(p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewVROOMWindowTable(base, 2048, params)
	}
}

func BenchmarkModExpWindowed_2048bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 2048)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)
	wt := NewVROOMWindowTable(base, 2048, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpWindowed(exp, wt, w, params)
	}
}
