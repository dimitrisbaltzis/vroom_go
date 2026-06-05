package main

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"math/bits"
	"os"
	"testing"
)

func skipWithoutAVX512S4(t *testing.T) {
	if os.Getenv("AVX512_TEST") == "" {
		t.Skip("Set AVX512_TEST=1 to run (needs AVX512IFMA or SDE)")
	}
}

func skipWithoutAVX512S4B(b *testing.B) {
	if os.Getenv("AVX512_TEST") == "" {
		b.Skip("Set AVX512_TEST=1 (needs AVX512IFMA or SDE)")
	}
}

// ============================================================================
// Unit tests for division-free modular arithmetic
// ============================================================================

func TestMulmodShoup52(t *testing.T) {
	// Test against reference mulmod
	rng := mathRand()
	for _, nBits := range []int{30, 40, 50, 51, 52} {
		n := randomPrimeBits(nBits)
		nU := n.Uint64()
		twoPow64 := new(big.Int).Lsh(bigOne, 64)

		for i := 0; i < 1000; i++ {
			// w < n (a constant), a < 2^57 (an accumulator-sized value)
			w := mathRandUint64(rng) % nU
			a := mathRandUint64(rng) % (1 << 57)

			// Precompute Shoup quotient
			wPrime := new(big.Int).SetUint64(w)
			wPrime.Mul(wPrime, twoPow64)
			wPrime.Div(wPrime, n)
			wP := wPrime.Uint64()

			got := mulmodShoup52(a, w, wP, nU)
			expected := new(big.Int).SetUint64(a)
			expected.Mul(expected, new(big.Int).SetUint64(w))
			expected.Mod(expected, n)
			exp := expected.Uint64()

			if got != exp {
				t.Fatalf("n=%d bits, a=%d, w=%d: got %d, want %d", nBits, a, w, got, exp)
			}
		}
	}
	t.Logf("mulmodShoup52: 5000 tests passed")
}

func TestBarrettReduce52(t *testing.T) {
	rng := mathRand()
	for _, nBits := range []int{30, 40, 50, 52} {
		n := randomPrimeBits(nBits)
		nU := n.Uint64()
		twoPow64 := new(big.Int).Lsh(bigOne, 64)
		mu := new(big.Int).Div(twoPow64, n).Uint64()

		for i := 0; i < 1000; i++ {
			// x < 2^58 (worst case for our usage)
			x := mathRandUint64(rng) % (1 << 58)
			got := barrettReduce52(x, nU, mu)
			exp := x % nU
			if got != exp {
				t.Fatalf("n=%d bits, x=%d: got %d, want %d", nBits, x, got, exp)
			}
		}
	}
	t.Logf("barrettReduce52: 4000 tests passed")
}

// ============================================================================
// Assembly kernel tests
// ============================================================================

func TestMatvecAVX512_3g(t *testing.T) {
	skipWithoutAVX512S4(t)

	// 3 groups = 24 target lanes
	tFrom := 5
	padTTo := 24

	// Build random A matrix and r vector
	aFlat := make([]uint64, tFrom*padTTo)
	r := make([]uint64, tFrom)
	for i := 0; i < tFrom; i++ {
		r[i] = uint64(i + 1) & mask52
		for j := 0; j < padTTo; j++ {
			aFlat[i*padTTo+j] = uint64((i+1)*(j+1)) & mask52
		}
	}

	// AVX512 kernel
	accLo := make([]uint64, padTTo)
	accHi := make([]uint64, padTTo)
	matvecAVX512_3g(&accLo[0], &accHi[0], &r[0], &aFlat[0], tFrom, padTTo)

	// Scalar reference
	refLo := make([]uint64, padTTo)
	refHi := make([]uint64, padTTo)
	for i := 0; i < tFrom; i++ {
		for j := 0; j < padTTo; j++ {
			ri := r[i] & mask52
			ai := aFlat[i*padTTo+j] & mask52
			hi, lo := bits.Mul64(ri, ai)
			refLo[j] += lo & mask52
			refHi[j] += (hi << 12) | (lo >> 52)
		}
	}

	for j := 0; j < padTTo; j++ {
		if accLo[j] != refLo[j] {
			t.Errorf("accLo[%d]: got %d, want %d", j, accLo[j], refLo[j])
		}
		if accHi[j] != refHi[j] {
			t.Errorf("accHi[%d]: got %d, want %d", j, accHi[j], refHi[j])
		}
	}
	t.Logf("matvecAVX512_3g: 24 lanes correct for %d source residues", tFrom)
}

func TestMatvecAVX512_3g_Large(t *testing.T) {
	skipWithoutAVX512S4(t)

	// Simulate realistic 1024-bit scenario: tFrom ≈ 22, padTTo = 24
	tFrom := 22
	padTTo := 24
	rng := mathRand()

	aFlat := make([]uint64, tFrom*padTTo)
	r := make([]uint64, tFrom)
	for i := 0; i < tFrom; i++ {
		r[i] = mathRandUint64(rng) & mask52
		for j := 0; j < padTTo; j++ {
			aFlat[i*padTTo+j] = mathRandUint64(rng) & mask52
		}
	}

	accLo := make([]uint64, padTTo)
	accHi := make([]uint64, padTTo)
	matvecAVX512_3g(&accLo[0], &accHi[0], &r[0], &aFlat[0], tFrom, padTTo)

	// Scalar reference
	refLo := make([]uint64, padTTo)
	refHi := make([]uint64, padTTo)
	for i := 0; i < tFrom; i++ {
		for j := 0; j < padTTo; j++ {
			ri := r[i] & mask52
			ai := aFlat[i*padTTo+j] & mask52
			hi, lo := bits.Mul64(ri, ai)
			refLo[j] += lo & mask52
			refHi[j] += (hi << 12) | (lo >> 52)
		}
	}

	for j := 0; j < padTTo; j++ {
		if accLo[j] != refLo[j] {
			t.Errorf("accLo[%d]: got %d, want %d", j, accLo[j], refLo[j])
		}
		if accHi[j] != refHi[j] {
			t.Errorf("accHi[%d]: got %d, want %d", j, accHi[j], refHi[j])
		}
	}
	t.Logf("matvecAVX512_3g: 24 lanes correct for tFrom=%d (1024-bit simulation)", tFrom)
}

func TestMatvecAVX512Gen(t *testing.T) {
	skipWithoutAVX512S4(t)

	// Test generic kernel with 2 groups (16 lanes)
	tFrom := 10
	padTTo := 16
	rng := mathRand()

	aFlat := make([]uint64, tFrom*padTTo)
	r := make([]uint64, tFrom)
	for i := 0; i < tFrom; i++ {
		r[i] = mathRandUint64(rng) & mask52
		for j := 0; j < padTTo; j++ {
			aFlat[i*padTTo+j] = mathRandUint64(rng) & mask52
		}
	}

	// Generic kernel (must pre-zero)
	accLo := make([]uint64, padTTo)
	accHi := make([]uint64, padTTo)
	matvecAVX512Gen(&accLo[0], &accHi[0], &r[0], &aFlat[0], tFrom, padTTo)

	// Scalar reference
	refLo := make([]uint64, padTTo)
	refHi := make([]uint64, padTTo)
	for i := 0; i < tFrom; i++ {
		for j := 0; j < padTTo; j++ {
			ri := r[i] & mask52
			ai := aFlat[i*padTTo+j] & mask52
			hi, lo := bits.Mul64(ri, ai)
			refLo[j] += lo & mask52
			refHi[j] += (hi << 12) | (lo >> 52)
		}
	}

	for j := 0; j < padTTo; j++ {
		if accLo[j] != refLo[j] {
			t.Errorf("accLo[%d]: got %d, want %d", j, accLo[j], refLo[j])
		}
		if accHi[j] != refHi[j] {
			t.Errorf("accHi[%d]: got %d, want %d", j, accHi[j], refHi[j])
		}
	}
	t.Logf("matvecAVX512Gen: %d lanes correct for tFrom=%d", padTTo, tFrom)
}

func TestMatvecAVX512_6g(t *testing.T) {
	skipWithoutAVX512S4(t)

	// 6 groups = 48 target lanes (2048-bit simulation)
	tFrom := 42
	padTTo := 48
	rng := mathRand()

	aFlat := make([]uint64, tFrom*padTTo)
	r := make([]uint64, tFrom)
	for i := 0; i < tFrom; i++ {
		r[i] = mathRandUint64(rng) & mask52
		for j := 0; j < padTTo; j++ {
			aFlat[i*padTTo+j] = mathRandUint64(rng) & mask52
		}
	}

	accLo := make([]uint64, padTTo)
	accHi := make([]uint64, padTTo)
	matvecAVX512_6g(&accLo[0], &accHi[0], &r[0], &aFlat[0], tFrom, padTTo)

	refLo := make([]uint64, padTTo)
	refHi := make([]uint64, padTTo)
	for i := 0; i < tFrom; i++ {
		for j := 0; j < padTTo; j++ {
			ri := r[i] & mask52
			ai := aFlat[i*padTTo+j] & mask52
			hi, lo := bits.Mul64(ri, ai)
			refLo[j] += lo & mask52
			refHi[j] += (hi << 12) | (lo >> 52)
		}
	}

	for j := 0; j < padTTo; j++ {
		if accLo[j] != refLo[j] {
			t.Errorf("accLo[%d]: got %d, want %d", j, accLo[j], refLo[j])
		}
		if accHi[j] != refHi[j] {
			t.Errorf("accHi[%d]: got %d, want %d", j, accHi[j], refHi[j])
		}
	}
	t.Logf("matvecAVX512_6g: 48 lanes correct for tFrom=%d", tFrom)
}

// ============================================================================
// Full pipeline tests (correctness)
// ============================================================================

func TestVROOMStage4(t *testing.T) {
	skipWithoutAVX512S4(t)

	for _, nbits := range []int{64, 128, 256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", nbits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, nbits)
			params := SetupRNSParamsStage4(p)

			for i := 0; i < 30; i++ {
				a := randomInRange(p)
				bv := randomInRange(p)

				expected := new(big.Int).Mul(a, bv)
				expected.Mod(expected, p)

				aM, aN := ToVROOMEncodingStage4(a, params)
				bM, bN := ToVROOMEncodingStage4(bv, params)
				rM, _ := VROOMStage4(aM, aN, bM, bN, params)

				rMCopy := make([]uint64, len(rM))
				copy(rMCopy, rM)
				got := FromVROOMEncodingStage4(rMCopy, params)

				if got.Cmp(expected) != 0 {
					t.Fatalf("test %d: want %s, got %s", i, expected, got)
				}
			}
			t.Logf("%d-bit: 30 multiplications correct", nbits)
		})
	}
}

func TestVROOMStage4Chained(t *testing.T) {
	skipWithoutAVX512S4(t)

	p, _ := rand.Prime(rand.Reader, 256)
	params := SetupRNSParamsStage4(p)
	base := randomInRange(p)
	exp := 100

	expected := new(big.Int).Exp(base, big.NewInt(int64(exp)), p)

	accM, accN := ToVROOMEncodingStage4(base, params)
	bM, bN := ToVROOMEncodingStage4(base, params)
	for i := 1; i < exp; i++ {
		rM, rN := VROOMStage4(accM, accN, bM, bN, params)
		accM = make([]uint64, len(rM))
		copy(accM, rM)
		accN = make([]uint64, len(rN))
		copy(accN, rN)
	}
	got := FromVROOMEncodingStage4(accM, params)

	if got.Cmp(expected) != 0 {
		t.Fatalf("base^%d mod p: want %s, got %s", exp, expected, got)
	}
	t.Logf("base^%d mod p: correct (chained 256-bit)", exp)
}

func TestVROOMStage4_1024Chained(t *testing.T) {
	skipWithoutAVX512S4(t)

	p, _ := rand.Prime(rand.Reader, 1024)
	params := SetupRNSParamsStage4(p)
	base := randomInRange(p)
	exp := 50

	expected := new(big.Int).Exp(base, big.NewInt(int64(exp)), p)

	accM, accN := ToVROOMEncodingStage4(base, params)
	bM, bN := ToVROOMEncodingStage4(base, params)
	for i := 1; i < exp; i++ {
		rM, rN := VROOMStage4(accM, accN, bM, bN, params)
		accM = make([]uint64, len(rM))
		copy(accM, rM)
		accN = make([]uint64, len(rN))
		copy(accN, rN)
	}
	got := FromVROOMEncodingStage4(accM, params)

	if got.Cmp(expected) != 0 {
		t.Fatalf("base^%d mod p: want %s, got %s", exp, expected, got)
	}
	t.Logf("base^%d mod p: correct (chained 1024-bit)", exp)
}

// ============================================================================
// Benchmarks
// ============================================================================

func BenchmarkVROOMStage4(b *testing.B) {
	skipWithoutAVX512S4B(b)
	for _, nbits := range []int{256, 512, 1024, 2048} {
		b.Run(fmt.Sprintf("%d-bit", nbits), func(b *testing.B) {
			p, _ := rand.Prime(rand.Reader, nbits)
			params := SetupRNSParamsStage4(p)
			a := randomInRange(p)
			bv := randomInRange(p)
			aM, aN := ToVROOMEncodingStage4(a, params)
			bM, bN := ToVROOMEncodingStage4(bv, params)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VROOMStage4(aM, aN, bM, bN, params)
			}
		})
	}
}

// BenchmarkApplyStage4_Parts profiles individual steps of ApplyStage4
func BenchmarkApplyStage4_Parts_1024(b *testing.B) {
	skipWithoutAVX512S4B(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	ps := SetupRNSParamsStage4(p)
	tM := len(ps.BaseM.Moduli)
	r := make([]uint64, tM)
	for i := range r {
		r[i] = randomInRange(new(big.Int).SetUint64(ps.BaseM.Moduli[i])).Uint64()
	}

	b.Run("full_Apply", func(b *testing.B) {
		out := make([]uint64, ps.CRNS1.PadTTo)
		lo := make([]uint64, ps.CRNS1.PadTTo)
		hi := make([]uint64, ps.CRNS1.PadTTo)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ps.CRNS1.ApplyStage4(r, out, lo, hi)
		}
	})
	b.Run("step1_matvec_kernel", func(b *testing.B) {
		lo := make([]uint64, ps.CRNS1.PadTTo)
		hi := make([]uint64, ps.CRNS1.PadTTo)
		nGroups := ps.CRNS1.PadTTo / 8
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			switch nGroups {
			case 3:
				matvecAVX512_3g(&lo[0], &hi[0], &r[0], &ps.CRNS1.AFlat[0], tM, ps.CRNS1.PadTTo)
			case 6:
				matvecAVX512_6g(&lo[0], &hi[0], &r[0], &ps.CRNS1.AFlat[0], tM, ps.CRNS1.PadTTo)
			default:
				for j := range lo[:ps.CRNS1.PadTTo] {
					lo[j] = 0
					hi[j] = 0
				}
				matvecAVX512Gen(&lo[0], &hi[0], &r[0], &ps.CRNS1.AFlat[0], tM, ps.CRNS1.PadTTo)
			}
		}
	})
	b.Run("step2_combine_shoup", func(b *testing.B) {
		// Pre-fill realistic accumulator values
		lo := make([]uint64, ps.CRNS1.PadTTo)
		hi := make([]uint64, ps.CRNS1.PadTTo)
		out := make([]uint64, ps.CRNS1.PadTTo)
		for j := 0; j < ps.CRNS1.TTo; j++ {
			lo[j] = uint64(j*1234567) % ps.CRNS1.ToMod[j]
			hi[j] = uint64(j*7654321) % ps.CRNS1.ToMod[j]
		}
		m := ps.CRNS1
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for j := 0; j < m.TTo; j++ {
				nj := m.ToMod[j]
				hiPart := mulmodShoup52(hi[j], m.Pow52Mod[j], m.Pow52Shoup[j], nj)
				sum := hiPart + lo[j]
				out[j] = barrettReduce52(sum, nj, m.BarrettMu[j])
			}
		}
	})
	b.Run("step3_k192", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			computeK192(r, ps.CRNS1.F, ps.CRNS1.FHi, ps.CRNS1.Prec, tM)
		}
	})
	b.Run("step4_correction_shoup", func(b *testing.B) {
		out := make([]uint64, ps.CRNS1.PadTTo)
		for j := 0; j < ps.CRNS1.TTo; j++ {
			out[j] = uint64(j*111) % ps.CRNS1.ToMod[j]
		}
		m := ps.CRNS1
		k := uint64(17) // typical k value
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for j := 0; j < m.TTo; j++ {
				nj := m.ToMod[j]
				kc := mulmodShoup52(k, m.CPadded[j], m.CShoup[j], nj)
				sum := out[j] + kc
				if sum >= nj {
					sum -= nj
				}
				_ = sum
			}
		}
	})
}

// BenchmarkApplyStage3_Parts_1024_Baseline benchmarks the old Stage 3 Apply
// with 52-bit moduli for fair comparison against Stage 4.
func BenchmarkApplyStage3_Parts_1024_Baseline(b *testing.B) {
	skipWithoutAVX512S4B(b)

	p, _ := rand.Prime(rand.Reader, 1024)
	ps3 := SetupRNSParamsStage3_52(p) // same 52-bit moduli as Stage 4
	tM := len(ps3.BaseM.Moduli)
	r := make([]uint64, tM)
	for i := range r {
		r[i] = randomInRange(new(big.Int).SetUint64(ps3.BaseM.Moduli[i])).Uint64()
	}

	b.Run("full_Apply_stage3_52bit", func(b *testing.B) {
		out := make([]uint64, ps3.CRNS1.PadTTo)
		lo := make([]uint64, ps3.CRNS1.PadTTo)
		hi := make([]uint64, ps3.CRNS1.PadTTo)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ps3.CRNS1.ApplyAVX512(r, out, lo, hi)
		}
	})
}

// BenchmarkVROOMStage3vs4_1024 runs Stage 3 and Stage 4 side by side for direct comparison.
func BenchmarkVROOMStage3vs4_1024(b *testing.B) {
	skipWithoutAVX512S4B(b)

	p, _ := rand.Prime(rand.Reader, 1024)

	b.Run("Stage3_52bit", func(b *testing.B) {
		ps3 := SetupRNSParamsStage3_52(p)
		a := randomInRange(p)
		bv := randomInRange(p)
		aM, aN := ToVROOMEncodingStage3(a, ps3)
		bM, bN := ToVROOMEncodingStage3(bv, ps3)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			VROOMStage3(aM, aN, bM, bN, ps3)
		}
	})

	b.Run("Stage4_52bit", func(b *testing.B) {
		ps4 := SetupRNSParamsStage4(p)
		a := randomInRange(p)
		bv := randomInRange(p)
		aM, aN := ToVROOMEncodingStage4(a, ps4)
		bM, bN := ToVROOMEncodingStage4(bv, ps4)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			VROOMStage4(aM, aN, bM, bN, ps4)
		}
	})
}

// ============================================================================
// Test helpers
// ============================================================================

// mathRand returns a deterministic-enough source for test reproducibility
func mathRand() *deterministicRand {
	return &deterministicRand{state: 0x12345678ABCDEF01}
}

type deterministicRand struct {
	state uint64
}

func (r *deterministicRand) Uint64() uint64 {
	// xorshift64
	r.state ^= r.state << 13
	r.state ^= r.state >> 7
	r.state ^= r.state << 17
	return r.state
}

func mathRandUint64(r *deterministicRand) uint64 {
	return r.Uint64()
}

func randomPrimeBits(bits int) *big.Int {
	p, _ := rand.Prime(rand.Reader, bits)
	return p
}
