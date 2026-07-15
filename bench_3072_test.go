package vroom

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"testing"
)

// ============================================================================
// 3072-bit correctness tests
// ============================================================================

func TestModExpVROOM_3072bit(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 3072)
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
	t.Log("3072-bit: 10 random exponentiations correct")
}

func TestModExpPrecomputed_3072bit(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 3072)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)

	for i := 0; i < 10; i++ {
		base := randomInRange(p)
		exp := randomInRange(p)

		pt := NewVROOMPreTable(base, 3072, params)

		got := ModExpVROOMPrecomputed(exp, pt, w, params)
		want := new(big.Int).Exp(base, exp, p)

		if got.Cmp(want) != 0 {
			t.Fatalf("test %d: got %s, want %s", i, got, want)
		}
	}
	t.Log("3072-bit precomputed: 10 random exponentiations correct")
}

func TestModExpWindowed_3072bit(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 3072)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)

	for i := 0; i < 10; i++ {
		base := randomInRange(p)
		exp := randomInRange(p)

		wt := NewVROOMWindowTable(base, 3072, params)
		got := ModExpVROOMWindowed(exp, wt, w, params)
		want := new(big.Int).Exp(base, exp, p)

		if got.Cmp(want) != 0 {
			t.Fatalf("test %d: got %s, want %s", i, got, want)
		}
	}
	t.Log("3072-bit windowed: 10 random exponentiations correct")
}

// ============================================================================
// 3072-bit benchmarks — multiplication
// ============================================================================

func BenchmarkVROOMStage4_3072bit(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 3072)
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

func BenchmarkBigIntMul_3072bit(b *testing.B) {
	p, _ := rand.Prime(rand.Reader, 3072)
	a := randomInRange(p)
	c := randomInRange(p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := new(big.Int).Mul(a, c)
		r.Mod(r, p)
	}
}

// ============================================================================
// 3072-bit benchmarks — exponentiation
// ============================================================================

func BenchmarkModExpWindowed_3072bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 3072)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)
	wt := NewVROOMWindowTable(base, 3072, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpWindowed(exp, wt, w, params)
	}
}

func BenchmarkModExpPrecomputed_3072bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 3072)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)
	pt := NewVROOMPreTable(base, 3072, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpPrecomputed(exp, pt, w, params)
	}
}

func BenchmarkModExpInner_3072bit_exp(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 3072)
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

func BenchmarkModExpBigInt_3072bit_exp(b *testing.B) {
	p, _ := rand.Prime(rand.Reader, 3072)
	base := randomInRange(p)
	exp, _ := rand.Int(rand.Reader, p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		new(big.Int).Exp(base, exp, p)
	}
}

func BenchmarkModExpWindowed_RSAVerify_3072(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 3072)
	params := SetupRNSParamsStage4(p)
	w := NewModExpWorkspace(params)
	base := randomInRange(p)
	e := big.NewInt(65537)
	wt := NewVROOMWindowTable(base, 3072, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpWindowed(e, wt, w, params)
	}
}

func BenchmarkNewVROOMWindowTable_3072(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 3072)
	params := SetupRNSParamsStage4(p)
	base := randomInRange(p)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NewVROOMWindowTable(base, 3072, params)
	}
}

// ============================================================================
// 3072-bit Random tests (extend existing suites)
// ============================================================================

func TestModExpVROOM_Random_3072(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 3072)
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
	t.Logf("3072-bit: 20 random exponentiations correct")
}

func TestModExpWindowed_Random_3072(t *testing.T) {
	skipWithoutAVX512ME(t)

	for _, bits := range []int{3072} {
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
