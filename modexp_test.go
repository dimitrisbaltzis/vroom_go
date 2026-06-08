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

	// 3^13 mod 7 = 3
	p := big.NewInt(7)
	params := SetupRNSParamsStage4(p)

	result := ModExpVROOM(big.NewInt(3), big.NewInt(13), params)
	expected := big.NewInt(3)
	if result.Cmp(expected) != 0 {
		t.Fatalf("3^13 mod 7: got %s, want %s", result, expected)
	}

	// 2^10 mod 1000 = 1024 mod 1000 = 24
	p2, _ := rand.Prime(rand.Reader, 16)
	params2 := SetupRNSParamsStage4(p2)
	base := big.NewInt(2)
	exp := big.NewInt(10)
	got := ModExpVROOM(base, exp, params2)
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
	a := randomInRange(p)

	// a^0 mod p = 1
	got := ModExpVROOM(a, big.NewInt(0), params)
	if got.Cmp(big.NewInt(1)) != 0 {
		t.Fatalf("a^0: got %s, want 1", got)
	}

	// a^1 mod p = a
	got = ModExpVROOM(a, big.NewInt(1), params)
	if got.Cmp(a) != 0 {
		t.Fatalf("a^1: got %s, want %s", got, a)
	}

	// a^2 mod p = a*a mod p
	got = ModExpVROOM(a, big.NewInt(2), params)
	want := new(big.Int).Mul(a, a)
	want.Mod(want, p)
	if got.Cmp(want) != 0 {
		t.Fatalf("a^2: got %s, want %s", got, want)
	}

	// a^(p-1) mod p = 1 (Fermat's little theorem)
	pMinus1 := new(big.Int).Sub(p, bigOne)
	got = ModExpVROOM(a, pMinus1, params)
	if got.Cmp(big.NewInt(1)) != 0 {
		t.Fatalf("a^(p-1) mod p: got %s, want 1 (Fermat)", got)
	}

	t.Log("Edge cases: correct")
}

func TestModExpVROOM_Random(t *testing.T) {
	skipWithoutAVX512ME(t)

	for _, bits := range []int{64, 128, 256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)

			for i := 0; i < 20; i++ {
				base := randomInRange(p)
				exp := randomInRange(p)

				got := ModExpVROOM(base, exp, params)
				want := new(big.Int).Exp(base, exp, p)

				if got.Cmp(want) != 0 {
					t.Fatalf("test %d: %s^%s mod %s: got %s, want %s",
						i, base, exp, p, got, want)
				}
			}
			t.Logf("%d-bit: 20 random exponentiations correct", bits)
		})
	}
}

func TestModExpVROOM_RSAPublicExponent(t *testing.T) {
	skipWithoutAVX512ME(t)

	// RSA verify: e = 65537 = 2^16 + 1 → only 17 squares + 1 multiply
	e := big.NewInt(65537)

	for _, bits := range []int{256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)

			for i := 0; i < 10; i++ {
				base := randomInRange(p)
				got := ModExpVROOM(base, e, params)
				want := new(big.Int).Exp(base, e, p)

				if got.Cmp(want) != 0 {
					t.Fatalf("test %d: base^65537 mod p: got %s, want %s", i, got, want)
				}
			}
			t.Logf("%d-bit: 10 RSA-style exponentiations correct", bits)
		})
	}
}

// ============================================================================
// Constant-time variant tests
// ============================================================================

func TestModExpVROOMConstTime_Random(t *testing.T) {
	skipWithoutAVX512ME(t)

	for _, bits := range []int{64, 128, 256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)

			for i := 0; i < 20; i++ {
				base := randomInRange(p)
				exp := randomInRange(p)

				got := ModExpVROOMConstTime(base, exp, params)
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

	for i := 0; i < 30; i++ {
		base := randomInRange(p)
		exp := randomInRange(p)

		got1 := ModExpVROOM(base, exp, params)
		got2 := ModExpVROOMConstTime(base, exp, params)

		if got1.Cmp(got2) != 0 {
			t.Fatalf("test %d: non-CT=%s, CT=%s", i, got1, got2)
		}
	}
	t.Log("30 tests: constant-time matches non-constant-time")
}

func TestModExpVROOMConstTime_Fermat(t *testing.T) {
	skipWithoutAVX512ME(t)

	// Fermat's little theorem: a^(p-1) ≡ 1 (mod p)
	p, _ := rand.Prime(rand.Reader, 256)
	params := SetupRNSParamsStage4(p)
	pMinus1 := new(big.Int).Sub(p, bigOne)

	for i := 0; i < 10; i++ {
		a := randomInRange(p)
		if a.Sign() == 0 {
			continue
		}
		got := ModExpVROOMConstTime(a, pMinus1, params)
		if got.Cmp(big.NewInt(1)) != 0 {
			t.Fatalf("test %d: a^(p-1) mod p = %s, want 1", i, got)
		}
	}
	t.Log("Fermat's little theorem: 10 tests correct")
}

// ============================================================================
// Benchmarks
// ============================================================================

func BenchmarkModExpVROOM_RSAVerify_1024(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	base := randomInRange(p)
	e := big.NewInt(65537) // 17 squares + 1 multiply = 18 VROOM calls

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpVROOM(base, e, params)
	}
}

func BenchmarkModExpVROOM_1024bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpVROOM(base, exp, params)
	}
}

func BenchmarkModExpVROOMConstTime_1024bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpVROOMConstTime(base, exp, params)
	}
}

func BenchmarkModExpBigInt_1024bit_exp(b *testing.B) {
	// Baseline: math/big.Int.Exp for comparison
	p, _ := rand.Prime(rand.Reader, 1024)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		new(big.Int).Exp(base, exp, p)
	}
}
