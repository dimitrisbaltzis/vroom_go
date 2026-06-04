package main

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"math/bits"
	"os"
	"testing"
)

// skipWithoutAVX512 skips tests unless AVX512_TEST=1 is set.
// Run with SDE: go test -c -o vroom.test.exe && sde -spr -- vroom.test.exe -test.run=TestAVX512
func skipWithoutAVX512(t *testing.T) {
	if os.Getenv("AVX512_TEST") == "" {
		t.Skip("Set AVX512_TEST=1 to run (needs AVX512IFMA or SDE)")
	}
}

const mask52 = (1 << 52) - 1

// refMadd52Lo computes the scalar reference for VPMADD52LUQ.
func refMadd52Lo(dst, a, b [8]uint64) [8]uint64 {
	for i := 0; i < 8; i++ {
		ai := a[i] & mask52
		bi := b[i] & mask52
		_, lo := bits.Mul64(ai, bi) // 104-bit product, we only need low 64
		dst[i] += lo & mask52       // add low 52 bits of product
	}
	return dst
}

// refMadd52Hi computes the scalar reference for VPMADD52HUQ.
func refMadd52Hi(dst, a, b [8]uint64) [8]uint64 {
	for i := 0; i < 8; i++ {
		ai := a[i] & mask52
		bi := b[i] & mask52
		hi, lo := bits.Mul64(ai, bi)
		prodHigh52 := (hi << 12) | (lo >> 52) // bits [103:52]
		dst[i] += prodHigh52
	}
	return dst
}

func TestAVX512_VPMADD52LUQ(t *testing.T) {
	skipWithoutAVX512(t)

	a := [8]uint64{
		0xFFFFFFFFFFFFF, 123456789, 0, 1,
		0xABCDE12345678, 999999999999, 2, 0xFFFFFFFFFFFFF,
	}
	b := [8]uint64{
		2, 987654321, 12345, 0xFFFFFFFFFFFFF,
		3, 1, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF,
	}
	dst := [8]uint64{100, 200, 300, 400, 500, 600, 700, 800}

	// Compute expected with scalar reference
	expected := refMadd52Lo(dst, a, b)

	// Run AVX512 version
	got := dst // copy initial values
	vpmadd52luq(&got[0], &a[0], &b[0])

	for i := 0; i < 8; i++ {
		if got[i] != expected[i] {
			t.Errorf("lane %d: got %d, want %d", i, got[i], expected[i])
		}
	}
	t.Logf("VPMADD52LUQ: all 8 lanes correct")
}

func TestAVX512_VPMADD52HUQ(t *testing.T) {
	skipWithoutAVX512(t)

	a := [8]uint64{
		0xFFFFFFFFFFFFF, 123456789, 0, 1,
		0xABCDE12345678, 999999999999, 2, 0xFFFFFFFFFFFFF,
	}
	b := [8]uint64{
		0xFFFFFFFFFFFFF, 987654321, 12345, 0xFFFFFFFFFFFFF,
		3, 1, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF,
	}
	dst := [8]uint64{0, 0, 0, 0, 0, 0, 0, 0}

	expected := refMadd52Hi(dst, a, b)

	got := dst
	vpmadd52huq(&got[0], &a[0], &b[0])

	for i := 0; i < 8; i++ {
		if got[i] != expected[i] {
			t.Errorf("lane %d: got %d, want %d", i, got[i], expected[i])
		}
	}
	t.Logf("VPMADD52HUQ: all 8 lanes correct")
}

func TestAVX512_BroadcastMulAccLo52(t *testing.T) {
	skipWithoutAVX512(t)

	scalar := uint64(0xABCDE12345)
	b := [8]uint64{1, 2, 3, 4, 5, 100, 0xFFFFFFFFFFFFF, 0}
	dst := [8]uint64{0, 0, 0, 0, 0, 0, 0, 0}

	// Expected: dst[i] += low52(scalar * b[i])
	expected := dst
	for i := 0; i < 8; i++ {
		si := scalar & mask52
		bi := b[i] & mask52
		_, lo := bits.Mul64(si, bi)
		expected[i] += lo & mask52
	}

	got := dst
	broadcastMulAccLo52(&got[0], scalar, &b[0])

	for i := 0; i < 8; i++ {
		if got[i] != expected[i] {
			t.Errorf("lane %d: got %d, want %d", i, got[i], expected[i])
		}
	}
	t.Logf("BroadcastMulAccLo52: all 8 lanes correct")
}

func TestAVX512_Accumulation(t *testing.T) {
	skipWithoutAVX512(t)

	// Simulate CRNS inner loop: acc[j] += r[i] * A[i][j] for multiple i
	A := [4][8]uint64{
		{10, 20, 30, 40, 50, 60, 70, 80},
		{11, 21, 31, 41, 51, 61, 71, 81},
		{12, 22, 32, 42, 52, 62, 72, 82},
		{13, 23, 33, 43, 53, 63, 73, 83},
	}
	r := [4]uint64{100, 200, 300, 400}

	// Scalar reference
	expected := [8]uint64{}
	for i := 0; i < 4; i++ {
		for j := 0; j < 8; j++ {
			_, lo := bits.Mul64(r[i]&mask52, A[i][j]&mask52)
			expected[j] += lo & mask52
		}
	}

	// AVX512 version: repeated broadcastMulAccLo52
	got := [8]uint64{}
	for i := 0; i < 4; i++ {
		broadcastMulAccLo52(&got[0], r[i], &A[i][0])
	}

	for j := 0; j < 8; j++ {
		if got[j] != expected[j] {
			t.Errorf("lane %d: got %d, want %d", j, got[j], expected[j])
		}
	}
	t.Logf("Accumulation (CRNS simulation): all 8 lanes correct")
}

func TestAVX512_BroadcastMulAcc52Combined(t *testing.T) {
	skipWithoutAVX512(t)

	scalar := uint64(0xABCDE12345)
	b := [8]uint64{1, 2, 3, 4, 5, 100, mask52, 0}
	lo := [8]uint64{10, 20, 30, 40, 50, 60, 70, 80}
	hi := [8]uint64{0, 0, 0, 0, 0, 0, 0, 0}

	// Expected
	expLo := lo
	expHi := hi
	for i := 0; i < 8; i++ {
		si := scalar & mask52
		bi := b[i] & mask52
		h, l := bits.Mul64(si, bi)
		expLo[i] += l & mask52
		expHi[i] += (h << 12) | (l >> 52)
	}

	broadcastMulAcc52(&lo[0], &hi[0], scalar, &b[0])

	for i := 0; i < 8; i++ {
		if lo[i] != expLo[i] {
			t.Errorf("lo lane %d: got %d, want %d", i, lo[i], expLo[i])
		}
		if hi[i] != expHi[i] {
			t.Errorf("hi lane %d: got %d, want %d", i, hi[i], expHi[i])
		}
	}
	t.Logf("BroadcastMulAcc52 combined: all lanes correct")
}

// TestAVX512_VROOMStage3 tests the full VROOM pipeline with AVX512 CRNS.
func TestAVX512_VROOMStage3(t *testing.T) {
	skipWithoutAVX512(t)

	for _, nbits := range []int{64, 128, 256, 512} {
		t.Run(fmt.Sprintf("%d-bit", nbits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, nbits)
			params := SetupRNSParamsStage3(p)

			for i := 0; i < 30; i++ {
				a := randomInRange(p)
				bv := randomInRange(p)

				expected := new(big.Int).Mul(a, bv)
				expected.Mod(expected, p)

				aM, aN := ToVROOMEncodingStage3(a, params)
				bM, bN := ToVROOMEncodingStage3(bv, params)
				rM, _ := VROOMStage3(aM, aN, bM, bN, params)

				rMCopy := make([]uint64, len(rM))
				copy(rMCopy, rM)
				got := FromVROOMEncodingStage3(rMCopy, params)

				if got.Cmp(expected) != 0 {
					t.Fatalf("test %d: want %s, got %s", i, expected, got)
				}
			}
			t.Logf("%d-bit: 30 multiplications correct", nbits)
		})
	}
}

// TestAVX512_VROOMStage3Chained tests chained multiplications (mod exp).
func TestAVX512_VROOMStage3Chained(t *testing.T) {
	skipWithoutAVX512(t)

	p, _ := rand.Prime(rand.Reader, 256)
	params := SetupRNSParamsStage3(p)
	base := randomInRange(p)
	exp := 100

	expected := new(big.Int).Exp(base, big.NewInt(int64(exp)), p)

	accM, accN := ToVROOMEncodingStage3(base, params)
	bM, bN := ToVROOMEncodingStage3(base, params)
	for i := 1; i < exp; i++ {
		rM, rN := VROOMStage3(accM, accN, bM, bN, params)
		// Copy results (workspace is reused)
		accM = make([]uint64, len(rM))
		copy(accM, rM)
		accN = make([]uint64, len(rN))
		copy(accN, rN)
	}
	got := FromVROOMEncodingStage3(accM, params)

	if got.Cmp(expected) != 0 {
		t.Fatalf("base^%d mod p: want %s, got %s", exp, expected, got)
	}
	t.Logf("base^%d mod p: correct", exp)
}

// BenchmarkVROOMStage3 benchmarks AVX512 VROOM (run under SDE or AVX512 CPU).
func BenchmarkVROOMStage3(b *testing.B) {
	if os.Getenv("AVX512_TEST") == "" {
		b.Skip("Set AVX512_TEST=1 (needs AVX512IFMA or SDE)")
	}
	for _, nbits := range []int{256, 512, 1024} {
		b.Run(fmt.Sprintf("%d-bit", nbits), func(b *testing.B) {
			p, _ := rand.Prime(rand.Reader, nbits)
			params := SetupRNSParamsStage3(p)
			a := randomInRange(p)
			bv := randomInRange(p)
			aM, aN := ToVROOMEncodingStage3(a, params)
			bM, bN := ToVROOMEncodingStage3(bv, params)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VROOMStage3(aM, aN, bM, bN, params)
			}
		})
	}
}

func BenchmarkApplyAVX512_Parts_1024(b *testing.B) {
    p, _ := rand.Prime(rand.Reader, 1024)
    ps := SetupRNSParamsStage3_52(p)
    r := make([]uint64, len(ps.BaseM.Moduli))
    for i := range r { r[i] = rand.Uint64() % ps.BaseM.Moduli[i] }
    
    b.Run("full_Apply", func(b *testing.B) {
        out := make([]uint64, ps.CRNS1.PadTTo)
        lo := make([]uint64, ps.CRNS1.PadTTo)
        hi := make([]uint64, ps.CRNS1.PadTTo)
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            ps.CRNS1.ApplyAVX512(r, out, lo, hi)
        }
    })
    b.Run("step1_avx512_only", func(b *testing.B) {
        lo := make([]uint64, ps.CRNS1.PadTTo)
        hi := make([]uint64, ps.CRNS1.PadTTo)
        tFrom := len(r)
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            for j := 0; j < ps.CRNS1.PadTTo; j++ { lo[j]=0; hi[j]=0 }
            for i := 0; i < tFrom; i++ {
                for g := 0; g < ps.CRNS1.PadTTo; g += 8 {
                    broadcastMulAcc52(&lo[g], &hi[g], r[i], &ps.CRNS1.APadded[i][g])
                }
            }
        }
    })
    b.Run("step3_k192_only", func(b *testing.B) {
        tFrom := len(r)
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            computeK192(r, ps.CRNS1.F, ps.CRNS1.FHi, ps.CRNS1.Prec, tFrom)
        }
    })
}