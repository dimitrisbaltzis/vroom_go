package main

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"os"
	"testing"
)

// --------------------------------------------------------------------------
// Correctness tests — 52-bit moduli
// --------------------------------------------------------------------------

func TestVROOMFast52_Correctness(t *testing.T) {
	for _, bits := range []int{64, 128, 256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsU64_52(p)

			t.Logf("p=%d bits, tM=%d, tN=%d (52-bit moduli)",
				p.BitLen(), len(params.BaseM.Moduli), len(params.BaseN.Moduli))

			for i := 0; i < 50; i++ {
				a := randomInRange(p)
				bv := randomInRange(p)

				expected := new(big.Int).Mul(a, bv)
				expected.Mod(expected, p)

				aM, aN := ToVROOMEncodingFast(a, params)
				bM, bN := ToVROOMEncodingFast(bv, params)
				rM, _ := VROOMFast(aM, aN, bM, bN, params)
				got := FromVROOMEncodingFast(rM, params)

				if got.Cmp(expected) != 0 {
					t.Fatalf("test %d: want %s, got %s", i, expected, got)
				}
			}
		})
	}
}

func TestVROOMStage2_52_Correctness(t *testing.T) {
	for _, bits := range []int{64, 128, 256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			ps := SetupRNSParamsStage2_52(p)

			t.Logf("p=%d bits, tM=%d, tN=%d (52-bit moduli)",
				p.BitLen(), len(ps.BaseM.Moduli), len(ps.BaseN.Moduli))

			for i := 0; i < 50; i++ {
				a := randomInRange(p)
				bv := randomInRange(p)

				expected := new(big.Int).Mul(a, bv)
				expected.Mod(expected, p)

				aM, aN := ToVROOMEncodingStage2(a, ps)
				bM, bN := ToVROOMEncodingStage2(bv, ps)
				rM, _ := VROOMStage2(aM, aN, bM, bN, ps)
				got := FromVROOMEncodingStage2(rM, ps)

				if got.Cmp(expected) != 0 {
					t.Fatalf("test %d: want %s, got %s", i, expected, got)
				}
			}
		})
	}
}

func TestVROOMNoAlloc52_Correctness(t *testing.T) {
	for _, bits := range []int{64, 128, 256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			ps := SetupRNSParamsNoAlloc_52(p)

			t.Logf("p=%d bits, tM=%d, tN=%d (52-bit moduli)",
				p.BitLen(), len(ps.BaseM.Moduli), len(ps.BaseN.Moduli))

			for i := 0; i < 50; i++ {
				a := randomInRange(p)
				bv := randomInRange(p)

				expected := new(big.Int).Mul(a, bv)
				expected.Mod(expected, p)

				aM, aN := ToVROOMEncodingNoAlloc(a, ps)
				bM, bN := ToVROOMEncodingNoAlloc(bv, ps)
				rM, _ := VROOMNoAlloc(aM, aN, bM, bN, ps)

				rMCopy := make([]uint64, len(rM))
				copy(rMCopy, rM)
				got := FromVROOMEncodingNoAlloc(rMCopy, ps)

				if got.Cmp(expected) != 0 {
					t.Fatalf("test %d: want %s, got %s", i, expected, got)
				}
			}
		})
	}
}

func TestVROOMStage2_52_Chained(t *testing.T) {
	p, _ := rand.Prime(rand.Reader, 256)
	ps := SetupRNSParamsStage2_52(p)
	base := randomInRange(p)

	for _, exp := range []int{2, 10, 50, 100} {
		t.Run(fmt.Sprintf("exp=%d", exp), func(t *testing.T) {
			expected := new(big.Int).Exp(base, big.NewInt(int64(exp)), p)

			accM, accN := ToVROOMEncodingStage2(base, ps)
			bM, bN := ToVROOMEncodingStage2(base, ps)
			for i := 1; i < exp; i++ {
				accM, accN = VROOMStage2(accM, accN, bM, bN, ps)
			}
			got := FromVROOMEncodingStage2(accM, ps)

			if got.Cmp(expected) != 0 {
				t.Fatalf("base^%d mod p mismatch", exp)
			}
		})
	}
}

func TestVROOMStage2_52_KnownPrimes(t *testing.T) {
	p25519 := new(big.Int).Exp(bigTwo, big.NewInt(255), nil)
	p25519.Sub(p25519, big.NewInt(19))

	bls381q, _ := new(big.Int).SetString(
		"1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", 16)

	for _, tc := range []struct {
		name string
		p    *big.Int
	}{
		{"Curve25519", p25519},
		{"BLS12-381", bls381q},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ps := SetupRNSParamsStage2_52(tc.p)
			t.Logf("%s: tM=%d, tN=%d (52-bit moduli)",
				tc.name, len(ps.BaseM.Moduli), len(ps.BaseN.Moduli))

			for i := 0; i < 20; i++ {
				a := randomInRange(tc.p)
				b := randomInRange(tc.p)
				expected := new(big.Int).Mul(a, b)
				expected.Mod(expected, tc.p)

				aM, aN := ToVROOMEncodingStage2(a, ps)
				bM, bN := ToVROOMEncodingStage2(b, ps)
				rM, _ := VROOMStage2(aM, aN, bM, bN, ps)
				got := FromVROOMEncodingStage2(rM, ps)

				if got.Cmp(expected) != 0 {
					t.Fatalf("test %d: mismatch", i)
				}
			}
		})
	}
}

// --------------------------------------------------------------------------
// Benchmarks — 32-bit vs 52-bit comparison
// --------------------------------------------------------------------------

func BenchmarkVROOMFast_32bit(b *testing.B) {
	for _, bits := range []int{256, 512, 1024} {
		b.Run(fmt.Sprintf("%d-bit", bits), func(b *testing.B) {
			p, _ := rand.Prime(rand.Reader, bits)
			ps := SetupRNSParamsU64(p)
			a := randomInRange(p)
			bv := randomInRange(p)
			aM, aN := ToVROOMEncodingFast(a, ps)
			bM, bN := ToVROOMEncodingFast(bv, ps)
			b.ReportMetric(float64(len(ps.BaseM.Moduli)), "tM")
			b.ReportMetric(float64(len(ps.BaseN.Moduli)), "tN")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VROOMFast(aM, aN, bM, bN, ps)
			}
		})
	}
}

func BenchmarkVROOMFast_52bit(b *testing.B) {
	for _, bits := range []int{256, 512, 1024} {
		b.Run(fmt.Sprintf("%d-bit", bits), func(b *testing.B) {
			p, _ := rand.Prime(rand.Reader, bits)
			ps := SetupRNSParamsU64_52(p)
			a := randomInRange(p)
			bv := randomInRange(p)
			aM, aN := ToVROOMEncodingFast(a, ps)
			bM, bN := ToVROOMEncodingFast(bv, ps)
			b.ReportMetric(float64(len(ps.BaseM.Moduli)), "tM")
			b.ReportMetric(float64(len(ps.BaseN.Moduli)), "tN")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VROOMFast(aM, aN, bM, bN, ps)
			}
		})
	}
}

func BenchmarkVROOMStage2_32bit(b *testing.B) {
	for _, bits := range []int{256, 512, 1024} {
		b.Run(fmt.Sprintf("%d-bit", bits), func(b *testing.B) {
			p, _ := rand.Prime(rand.Reader, bits)
			ps := SetupRNSParamsStage2(p)
			a := randomInRange(p)
			bv := randomInRange(p)
			aM, aN := ToVROOMEncodingStage2(a, ps)
			bM, bN := ToVROOMEncodingStage2(bv, ps)
			b.ReportMetric(float64(len(ps.BaseM.Moduli)), "tM")
			b.ReportMetric(float64(len(ps.BaseN.Moduli)), "tN")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VROOMStage2(aM, aN, bM, bN, ps)
			}
		})
	}
}

func BenchmarkVROOMStage2_52bit(b *testing.B) {
	for _, bits := range []int{256, 512, 1024} {
		b.Run(fmt.Sprintf("%d-bit", bits), func(b *testing.B) {
			p, _ := rand.Prime(rand.Reader, bits)
			ps := SetupRNSParamsStage2_52(p)
			a := randomInRange(p)
			bv := randomInRange(p)
			aM, aN := ToVROOMEncodingStage2(a, ps)
			bM, bN := ToVROOMEncodingStage2(bv, ps)
			b.ReportMetric(float64(len(ps.BaseM.Moduli)), "tM")
			b.ReportMetric(float64(len(ps.BaseN.Moduli)), "tN")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VROOMStage2(aM, aN, bM, bN, ps)
			}
		})
	}
}

func BenchmarkVROOMNoAlloc_32bit(b *testing.B) {
	for _, bits := range []int{256, 512, 1024} {
		b.Run(fmt.Sprintf("%d-bit", bits), func(b *testing.B) {
			p, _ := rand.Prime(rand.Reader, bits)
			ps := SetupRNSParamsNoAlloc(p)
			a := randomInRange(p)
			bv := randomInRange(p)
			aM, aN := ToVROOMEncodingNoAlloc(a, ps)
			bM, bN := ToVROOMEncodingNoAlloc(bv, ps)
			b.ReportMetric(float64(len(ps.BaseM.Moduli)), "tM")
			b.ReportMetric(float64(len(ps.BaseN.Moduli)), "tN")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VROOMNoAlloc(aM, aN, bM, bN, ps)
			}
		})
	}
}

func BenchmarkVROOMNoAlloc_52bit(b *testing.B) {
	for _, bits := range []int{256, 512, 1024} {
		b.Run(fmt.Sprintf("%d-bit", bits), func(b *testing.B) {
			p, _ := rand.Prime(rand.Reader, bits)
			ps := SetupRNSParamsNoAlloc_52(p)
			a := randomInRange(p)
			bv := randomInRange(p)
			aM, aN := ToVROOMEncodingNoAlloc(a, ps)
			bM, bN := ToVROOMEncodingNoAlloc(bv, ps)
			b.ReportMetric(float64(len(ps.BaseM.Moduli)), "tM")
			b.ReportMetric(float64(len(ps.BaseN.Moduli)), "tN")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VROOMNoAlloc(aM, aN, bM, bN, ps)
			}
		})
	}
}

func BenchmarkVROOMStage3_52bit(b *testing.B) {
	if os.Getenv("AVX512_TEST") == "" {
		b.Skip("Set AVX512_TEST=1 (needs AVX512IFMA or SDE)")
	}
	for _, bits := range []int{256, 512, 1024} {
		b.Run(fmt.Sprintf("%d-bit", bits), func(b *testing.B) {
			p, _ := rand.Prime(rand.Reader, bits)
			ps := SetupRNSParamsStage3_52(p)
			a := randomInRange(p)
			bv := randomInRange(p)
			aM, aN := ToVROOMEncodingStage3(a, ps)
			bM, bN := ToVROOMEncodingStage3(bv, ps)
			b.ReportMetric(float64(len(ps.BaseM.Moduli)), "tM")
			b.ReportMetric(float64(len(ps.BaseN.Moduli)), "tN")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VROOMStage3(aM, aN, bM, bN, ps)
			}
		})
	}
}

func TestModuliCountComparison(t *testing.T) {
	for _, bits := range []int{256, 381, 512, 1024} {
		var p *big.Int
		if bits == 381 {
			p, _ = new(big.Int).SetString(
				"1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", 16)
		} else {
			p, _ = rand.Prime(rand.Reader, bits)
		}

		p32 := SetupRNSParamsStage2(p)
		p52 := SetupRNSParamsStage2_52(p)

		tM32, tN32 := len(p32.BaseM.Moduli), len(p32.BaseN.Moduli)
		tM52, tN52 := len(p52.BaseM.Moduli), len(p52.BaseN.Moduli)

		crns32 := tM32*tN32 + tN32*tM32
		crns52 := tM52*tN52 + tN52*tM52

		t.Logf("%d-bit p: 32-bit tM=%d tN=%d (ops=%d) | 52-bit tM=%d tN=%d (ops=%d) | ratio=%.2fx",
			p.BitLen(), tM32, tN32, crns32, tM52, tN52, crns52,
			float64(crns32)/float64(crns52))
	}
}
