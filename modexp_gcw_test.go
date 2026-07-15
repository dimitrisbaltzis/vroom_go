package vroom

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"testing"
)

// ============================================================================
// Correctness tests — GCW Double
// ============================================================================

func TestModExpDoubleWindowed_Random(t *testing.T) {
	skipWithoutAVX512ME(t)

	for _, bits := range []int{64, 128, 256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)
			gw := NewGCWWorkspace(params)

			for i := 0; i < 20; i++ {
				base := randomInRange(p)
				exp1 := randomInRange(p)
				exp2 := randomInRange(p)

				wt := NewVROOMWindowTable(base, bits, params)

				got1, got2 := ModExpDoubleWindowed(exp1, exp2, wt, gw, params)
				want1 := new(big.Int).Exp(base, exp1, p)
				want2 := new(big.Int).Exp(base, exp2, p)

				if got1.Cmp(want1) != 0 {
					t.Fatalf("test %d exp1: got %s, want %s", i, got1, want1)
				}
				if got2.Cmp(want2) != 0 {
					t.Fatalf("test %d exp2: got %s, want %s", i, got2, want2)
				}
			}
			t.Logf("%d-bit: 20 double GCW exponentiations correct", bits)
		})
	}
}

func TestModExpDoubleWindowed_MatchesWindowed(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 512)
	params := SetupRNSParamsStage4(p)
	gw := NewGCWWorkspace(params)
	w := NewModExpWorkspace(params)

	for i := 0; i < 30; i++ {
		base := randomInRange(p)
		exp1 := randomInRange(p)
		exp2 := randomInRange(p)

		wt := NewVROOMWindowTable(base, 512, params)

		got1, got2 := ModExpDoubleWindowed(exp1, exp2, wt, gw, params)
		want1 := ModExpVROOMWindowed(exp1, wt, w, params)
		want2 := ModExpVROOMWindowed(exp2, wt, w, params)

		if got1.Cmp(want1) != 0 {
			t.Fatalf("test %d exp1: GCW=%s, windowed=%s", i, got1, want1)
		}
		if got2.Cmp(want2) != 0 {
			t.Fatalf("test %d exp2: GCW=%s, windowed=%s", i, got2, want2)
		}
	}
	t.Log("30 tests: double GCW matches independent windowed")
}

func TestModExpDoubleWindowed_EdgeCases(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 256)
	params := SetupRNSParamsStage4(p)
	gw := NewGCWWorkspace(params)
	a := randomInRange(p)
	wt := NewVROOMWindowTable(a, 256, params)

	// exp1=0
	got1, got2 := ModExpDoubleWindowed(big.NewInt(0), big.NewInt(3), wt, gw, params)
	if got1.Cmp(big.NewInt(1)) != 0 {
		t.Fatalf("exp1=0: got %s, want 1", got1)
	}
	want2 := new(big.Int).Exp(a, big.NewInt(3), p)
	if got2.Cmp(want2) != 0 {
		t.Fatalf("exp2=3: got %s, want %s", got2, want2)
	}

	// exp2=0
	got1, got2 = ModExpDoubleWindowed(big.NewInt(3), big.NewInt(0), wt, gw, params)
	want1 := new(big.Int).Exp(a, big.NewInt(3), p)
	if got1.Cmp(want1) != 0 {
		t.Fatalf("exp1=3: got %s, want %s", got1, want1)
	}
	if got2.Cmp(big.NewInt(1)) != 0 {
		t.Fatalf("exp2=0: got %s, want 1", got2)
	}

	t.Log("Edge cases: correct")
}

// ============================================================================
// Correctness tests — GCW Fourfold
// ============================================================================

func TestModExpFourfoldWindowed_Random(t *testing.T) {
	skipWithoutAVX512ME(t)

	for _, bits := range []int{64, 128, 256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsStage4(p)
			gw := NewGCWWorkspace(params)

			for i := 0; i < 20; i++ {
				base := randomInRange(p)
				exps := [4]*big.Int{
					randomInRange(p), randomInRange(p),
					randomInRange(p), randomInRange(p),
				}

				wt := NewVROOMWindowTable(base, bits, params)
				got := ModExpFourfoldWindowed(exps, wt, gw, params)

				for j := range exps {
					want := new(big.Int).Exp(base, exps[j], p)
					if got[j].Cmp(want) != 0 {
						t.Fatalf("test %d exp[%d]: got %s, want %s", i, j, got[j], want)
					}
				}
			}
			t.Logf("%d-bit: 20 fourfold GCW exponentiations correct", bits)
		})
	}
}

func TestModExpFourfoldWindowed_MatchesWindowed(t *testing.T) {
	skipWithoutAVX512ME(t)

	p, _ := rand.Prime(rand.Reader, 512)
	params := SetupRNSParamsStage4(p)
	gw := NewGCWWorkspace(params)
	w := NewModExpWorkspace(params)

	for i := 0; i < 20; i++ {
		base := randomInRange(p)
		exps := [4]*big.Int{
			randomInRange(p), randomInRange(p),
			randomInRange(p), randomInRange(p),
		}

		wt := NewVROOMWindowTable(base, 512, params)
		got := ModExpFourfoldWindowed(exps, wt, gw, params)

		for j := range exps {
			want := ModExpVROOMWindowed(exps[j], wt, w, params)
			if got[j].Cmp(want) != 0 {
				t.Fatalf("test %d exp[%d]: GCW=%s, windowed=%s", i, j, got[j], want)
			}
		}
	}
	t.Log("20 tests: fourfold GCW matches independent windowed")
}

// ============================================================================
// Benchmarks
// ============================================================================

func BenchmarkModExpDoubleWindowed_1024bit(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	gw := NewGCWWorkspace(params)
	base := randomInRange(p)
	exp1, _ := rand.Int(rand.Reader, p)
	exp2, _ := rand.Int(rand.Reader, p)
	wt := NewVROOMWindowTable(base, 1024, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpDoubleWindowed(exp1, exp2, wt, gw, params)
	}
}

func BenchmarkModExpFourfoldWindowed_1024bit(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	gw := NewGCWWorkspace(params)
	base := randomInRange(p)
	exps := [4]*big.Int{
		randomInRange(p), randomInRange(p),
		randomInRange(p), randomInRange(p),
	}
	wt := NewVROOMWindowTable(base, 1024, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpFourfoldWindowed(exps, wt, gw, params)
	}
}

func BenchmarkModExpDoubleWindowed_2048bit(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 2048)
	params := SetupRNSParamsStage4(p)
	gw := NewGCWWorkspace(params)
	base := randomInRange(p)
	exp1, _ := rand.Int(rand.Reader, p)
	exp2, _ := rand.Int(rand.Reader, p)
	wt := NewVROOMWindowTable(base, 2048, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpDoubleWindowed(exp1, exp2, wt, gw, params)
	}
}

func BenchmarkModExpFourfoldWindowed_2048bit(b *testing.B) {
	skipWithoutAVX512MEB(b)

	p, _ := rand.Prime(rand.Reader, 2048)
	params := SetupRNSParamsStage4(p)
	gw := NewGCWWorkspace(params)
	base := randomInRange(p)
	exps := [4]*big.Int{
		randomInRange(p), randomInRange(p),
		randomInRange(p), randomInRange(p),
	}
	wt := NewVROOMWindowTable(base, 2048, params)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ModExpFourfoldWindowed(exps, wt, gw, params)
	}
}
