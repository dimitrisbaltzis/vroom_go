// bench_test.go — Fair comparison: VROOM vs jiajunxin/multiexp
//
// Three benchmark scenarios:
//   A. Single base^exp mod p  (precompute + runtime, separated)
//   B. Many exponents, same base (amortized precompute)
//   C. 4 exponents, same base (jiajunxin GCW advantage)
//
// Run:
//   AVX512_TEST=1 go test -bench BenchmarkComparison -benchmem -benchtime 5s -count 3 | tee results.txt
//   benchstat results.txt

package comparison

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"runtime"
	"testing"

	vroom "github.com/dimitrisbaltzis/vroom_go"
	"github.com/jiajunxin/multiexp"
)

// ============================================================================
// Helpers
// ============================================================================

var bigOne = big.NewInt(1)

func randInRange(max *big.Int) *big.Int {
	v, _ := rand.Int(rand.Reader, new(big.Int).Sub(max, big.NewInt(2)))
	v.Add(v, big.NewInt(2)) // ensure v >= 2 (jiajunxin requires x > 1)
	return v
}

func skipNoAVX512(tb testing.TB) {
	// VROOM tests use this env var to gate AVX512 tests
	// If AVX512 isn't available, the VROOM setup will panic
	// We rely on the same gating mechanism as the main repo
	tb.Helper()
}

// ============================================================================
// Correctness — both must match big.Int.Exp
// ============================================================================

func TestCorrectness(t *testing.T) {
	for _, bits := range []int{256, 512, 1024} {
		t.Run(fmt.Sprintf("%dbit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			base := randInRange(p)
			exp := randInRange(p)

			// Reference
			ref := new(big.Int).Exp(base, exp, p)

			// ── Jiajunxin ──
			tableSize := (bits + 63) / 64
			jt := multiexp.NewPrecomputeTable(base, p, tableSize)
			if jt == nil {
				t.Fatal("jiajunxin: NewPrecomputeTable returned nil")
			}
			got1 := multiexp.ExpParallel(base, exp, p, jt, 1, 2)
			if ref.Cmp(got1) != 0 {
				t.Fatalf("jiajunxin WRONG:\n  base=%s\n  exp=%s\n  got=%s\n  want=%s",
					base, exp, got1, ref)
			}

			// ── VROOM ──
			params := vroom.SetupRNSParamsStage4(p)
			ws := vroom.NewModExpWorkspace(params)
			vt := vroom.NewVROOMWindowTable(base, bits, params)
			got2 := vroom.ModExpVROOMWindowed(exp, vt, ws, params)
			if ref.Cmp(got2) != 0 {
				t.Fatalf("VROOM WRONG:\n  base=%s\n  exp=%s\n  got=%s\n  want=%s",
					base, exp, got2, ref)
			}

			// ── Both match ──
			t.Logf("%d-bit: jiajunxin ✓  VROOM ✓  (match big.Int.Exp)", bits)
		})
	}
}

func TestCorrectness_Fourfold(t *testing.T) {
	for _, bits := range []int{256, 512, 1024} {
		t.Run(fmt.Sprintf("%dbit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			base := randInRange(p)
			exps := [4]*big.Int{
				randInRange(p), randInRange(p),
				randInRange(p), randInRange(p),
			}

			// Reference
			var refs [4]*big.Int
			for i := range refs {
				refs[i] = new(big.Int).Exp(base, exps[i], p)
			}

			// ── Jiajunxin FourfoldExp ──
			got := multiexp.FourfoldExp(base, p, exps)
			for i := range got {
				if refs[i].Cmp(got[i]) != 0 {
					t.Fatalf("jiajunxin FourfoldExp[%d] WRONG: got=%s want=%s", i, got[i], refs[i])
				}
			}

			// ── VROOM 4× independent ──
			params := vroom.SetupRNSParamsStage4(p)
			ws := vroom.NewModExpWorkspace(params)
			vt := vroom.NewVROOMWindowTable(base, bits, params)
			for i := range exps {
				got := vroom.ModExpVROOMWindowed(exps[i], vt, ws, params)
				if refs[i].Cmp(got) != 0 {
					t.Fatalf("VROOM exp[%d] WRONG: got=%s want=%s", i, got, refs[i])
				}
			}

			t.Logf("%d-bit: fourfold correctness ✓", bits)
		})
	}
}

// ============================================================================
// Scenario A: Single base^exp mod p
// ============================================================================

func BenchmarkComparison(b *testing.B) {
	for _, bits := range []int{1024, 2048} {
		p, _ := rand.Prime(rand.Reader, bits)
		base := randInRange(p)
		exp := randInRange(p)
		tableSize := (bits + 63) / 64
		nCPU := runtime.NumCPU()

		// ────────────────────────────────────────────────────
		// A1. Precompute cost (one-time per base)
		// ────────────────────────────────────────────────────

		b.Run(fmt.Sprintf("%d/precompute/jiajunxin", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				multiexp.NewPrecomputeTable(base, p, tableSize)
			}
		})

		vroomParams := vroom.SetupRNSParamsStage4(p)
		b.Run(fmt.Sprintf("%d/precompute/vroom_windowed", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				vroom.NewVROOMWindowTable(base, bits, vroomParams)
			}
		})

		// ────────────────────────────────────────────────────
		// A2. Runtime — single exp (table already built)
		// ────────────────────────────────────────────────────

		jt := multiexp.NewPrecomputeTable(base, p, tableSize)
		ws := vroom.NewModExpWorkspace(vroomParams)
		vt := vroom.NewVROOMWindowTable(base, bits, vroomParams)

		// Jiajunxin: 1 thread (fair SIMD vs scalar comparison)
		b.Run(fmt.Sprintf("%d/single_exp/jiajunxin_1T", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				multiexp.ExpParallel(base, exp, p, jt, 1, 2)
			}
		})

		// Jiajunxin: all cores (practical throughput)
		b.Run(fmt.Sprintf("%d/single_exp/jiajunxin_%dT", bits, nCPU), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				multiexp.ExpParallel(base, exp, p, jt, nCPU, 2)
			}
		})

		// VROOM windowed (single-thread, SIMD)
		b.Run(fmt.Sprintf("%d/single_exp/vroom_windowed", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				vroom.ModExpWindowed(exp, vt, ws, vroomParams)
			}
		})

		// VROOM windowed constant-time
		b.Run(fmt.Sprintf("%d/single_exp/vroom_windowed_CT", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				vroom.ModExpWindowedConstTime(exp, vt, ws, vroomParams)
			}
		})

		// Baseline: Go stdlib
		b.Run(fmt.Sprintf("%d/single_exp/stdlib_bigint", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				new(big.Int).Exp(base, exp, p)
			}
		})

		// ────────────────────────────────────────────────────
		// B. Many exponents, same base (amortized precompute)
		// ────────────────────────────────────────────────────

		// Pre-generate 100 random exponents
		numExps := 100
		exponents := make([]*big.Int, numExps)
		for i := range exponents {
			exponents[i] = randInRange(p)
		}

		b.Run(fmt.Sprintf("%d/batch_%d/jiajunxin_1T", bits, numExps), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				for _, e := range exponents {
					multiexp.ExpParallel(base, e, p, jt, 1, 2)
				}
			}
		})

		b.Run(fmt.Sprintf("%d/batch_%d/vroom_windowed", bits, numExps), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				for _, e := range exponents {
					vroom.ModExpWindowed(e, vt, ws, vroomParams)
				}
			}
		})

		// ────────────────────────────────────────────────────
		// C. Fourfold: jiajunxin GCW vs VROOM 4× independent
		// ────────────────────────────────────────────────────

		e4 := [4]*big.Int{
			randInRange(p), randInRange(p),
			randInRange(p), randInRange(p),
		}

		// Jiajunxin: native fourfold with GCW optimization
		b.Run(fmt.Sprintf("%d/fourfold/jiajunxin_GCW", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				multiexp.FourfoldExp(base, p, e4)
			}
		})

		// Jiajunxin: fourfold with precomputed table
		b.Run(fmt.Sprintf("%d/fourfold/jiajunxin_precomputed", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				multiexp.FourfoldExpPrecomputed(base, p, e4, jt)
			}
		})

		// Jiajunxin: fourfold with precomputed table + parallel
		b.Run(fmt.Sprintf("%d/fourfold/jiajunxin_precomputed_parallel", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				multiexp.FourfoldExpPrecomputedParallel(base, p, e4, jt)
			}
		})

		// VROOM: 4 independent windowed exponentiations
		b.Run(fmt.Sprintf("%d/fourfold/vroom_4x_windowed", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				for _, e := range e4 {
					vroom.ModExpWindowed(e, vt, ws, vroomParams)
				}
			}
		})

		// Baseline: 4× big.Int.Exp
		b.Run(fmt.Sprintf("%d/fourfold/stdlib_4x_bigint", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				for _, e := range e4 {
					new(big.Int).Exp(base, e, p)
				}
			}
		})

		// ────────────────────────────────────────────────────
		// D. DoubleExp (jiajunxin only — VROOM has no native)
		// ────────────────────────────────────────────────────

		e2 := [2]*big.Int{randInRange(p), randInRange(p)}

		b.Run(fmt.Sprintf("%d/double/jiajunxin_GCW", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				multiexp.DoubleExp(base, e2, p)
			}
		})

		b.Run(fmt.Sprintf("%d/double/vroom_2x_windowed", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				for _, e := range e2 {
					vroom.ModExpWindowed(e, vt, ws, vroomParams)
				}
			}
		})

		// ────────────────────────────────────────────────────
		// E. RSA verify style (small exponent e=65537)
		// ────────────────────────────────────────────────────

		rsaE := big.NewInt(65537)

		b.Run(fmt.Sprintf("%d/rsa_verify/jiajunxin_1T", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				multiexp.ExpParallel(base, rsaE, p, jt, 1, 2)
			}
		})

		b.Run(fmt.Sprintf("%d/rsa_verify/vroom_windowed", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				vroom.ModExpWindowed(rsaE, vt, ws, vroomParams)
			}
		})

		b.Run(fmt.Sprintf("%d/rsa_verify/stdlib_bigint", bits), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				new(big.Int).Exp(base, rsaE, p)
			}
		})
	}
}
