package main

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"testing"
)

// TestPoschandPosch tests Algorithm 1 with various prime sizes.
func TestPoschandPosch(t *testing.T) {
	for _, bits := range []int{64, 128, 256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParams(p)

			for i := 0; i < 50; i++ {
				a := randomInRange(p)
				b := randomInRange(p)

				expected := new(big.Int).Mul(a, b)
				expected.Mod(expected, p)

				aM, aN := ToMontgomeryRNS(a, params)
				bM, bN := ToMontgomeryRNS(b, params)
				rM, _ := PoschandPosch(aM, aN, bM, bN, params)
				got := FromMontgomeryRNS(rM, params)

				if got.Cmp(expected) != 0 {
					t.Fatalf("a*b mod p mismatch:\n  a=%s\n  b=%s\n  want=%s\n  got=%s",
						a, b, expected, got)
				}
			}
		})
	}
}

// TestVROOM tests Algorithm 2 with various prime sizes.
func TestVROOM(t *testing.T) {
	for _, bits := range []int{64, 128, 256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParams(p)

			for i := 0; i < 50; i++ {
				a := randomInRange(p)
				b := randomInRange(p)

				expected := new(big.Int).Mul(a, b)
				expected.Mod(expected, p)

				aM, aN := ToVROOMEncoding(a, params)
				bM, bN := ToVROOMEncoding(b, params)
				rM, _ := VROOM(aM, aN, bM, bN, params)
				got := FromVROOMEncoding(rM, params)

				if got.Cmp(expected) != 0 {
					t.Fatalf("a*b mod p mismatch:\n  a=%s\n  b=%s\n  want=%s\n  got=%s",
						a, b, expected, got)
				}
			}
		})
	}
}

// TestVROOMKnownPrimes tests with specific cryptographic primes.
func TestVROOMKnownPrimes(t *testing.T) {
	// Curve25519: p = 2^255 - 19
	p25519 := new(big.Int).Exp(bigTwo, big.NewInt(255), nil)
	p25519.Sub(p25519, big.NewInt(19))

	// BLS12-381 field modulus
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
			params := SetupRNSParams(tc.p)
			for i := 0; i < 20; i++ {
				a := randomInRange(tc.p)
				b := randomInRange(tc.p)
				expected := new(big.Int).Mul(a, b)
				expected.Mod(expected, tc.p)

				aM, aN := ToVROOMEncoding(a, params)
				bM, bN := ToVROOMEncoding(b, params)
				rM, _ := VROOM(aM, aN, bM, bN, params)
				got := FromVROOMEncoding(rM, params)

				if got.Cmp(expected) != 0 {
					t.Fatalf("test %d: a*b mod p mismatch", i)
				}
			}
		})
	}
}

// TestChainedVROOM tests sequential multiplications (modular exponentiation).
func TestChainedVROOM(t *testing.T) {
	p, _ := rand.Prime(rand.Reader, 256)
	params := SetupRNSParams(p)
	base := randomInRange(p)

	for _, exp := range []int{2, 10, 50, 100} {
		t.Run(fmt.Sprintf("exp=%d", exp), func(t *testing.T) {
			expected := new(big.Int).Exp(base, big.NewInt(int64(exp)), p)

			accM, accN := ToVROOMEncoding(base, params)
			bM, bN := ToVROOMEncoding(base, params)
			for i := 1; i < exp; i++ {
				accM, accN = VROOM(accM, accN, bM, bN, params)
			}
			got := FromVROOMEncoding(accM, params)

			if got.Cmp(expected) != 0 {
				t.Fatalf("base^%d mod p mismatch:\n  want=%s\n  got=%s",
					exp, expected, got)
			}
		})
	}
}

// TestEdgeCases tests boundary conditions.
func TestEdgeCases(t *testing.T) {
	p, _ := rand.Prime(rand.Reader, 256)
	params := SetupRNSParams(p)

	t.Run("a*0=0", func(t *testing.T) {
		a := randomInRange(p)
		aM, aN := ToVROOMEncoding(a, params)
		bM, bN := ToVROOMEncoding(big.NewInt(0), params)
		rM, _ := VROOM(aM, aN, bM, bN, params)
		got := FromVROOMEncoding(rM, params)
		if got.Sign() != 0 {
			t.Fatalf("a*0 != 0, got %s", got)
		}
	})

	t.Run("a*1=a", func(t *testing.T) {
		a := randomInRange(p)
		aM, aN := ToVROOMEncoding(a, params)
		bM, bN := ToVROOMEncoding(big.NewInt(1), params)
		rM, _ := VROOM(aM, aN, bM, bN, params)
		got := FromVROOMEncoding(rM, params)
		if got.Cmp(a) != 0 {
			t.Fatalf("a*1 != a: got %s, want %s", got, a)
		}
	})

	t.Run("(p-1)*(p-1)=1", func(t *testing.T) {
		pm1 := new(big.Int).Sub(p, bigOne)
		aM, aN := ToVROOMEncoding(pm1, params)
		bM, bN := ToVROOMEncoding(pm1, params)
		rM, _ := VROOM(aM, aN, bM, bN, params)
		got := FromVROOMEncoding(rM, params)
		if got.Cmp(bigOne) != 0 {
			t.Fatalf("(p-1)^2 mod p != 1, got %s", got)
		}
	})

	t.Run("consistency_alg1_vroom", func(t *testing.T) {
		for i := 0; i < 20; i++ {
			a := randomInRange(p)
			b := randomInRange(p)

			aM1, aN1 := ToMontgomeryRNS(a, params)
			bM1, bN1 := ToMontgomeryRNS(b, params)
			rM1, _ := PoschandPosch(aM1, aN1, bM1, bN1, params)
			r1 := FromMontgomeryRNS(rM1, params)

			aM2, aN2 := ToVROOMEncoding(a, params)
			bM2, bN2 := ToVROOMEncoding(b, params)
			rM2, _ := VROOM(aM2, aN2, bM2, bN2, params)
			r2 := FromVROOMEncoding(rM2, params)

			if r1.Cmp(r2) != 0 {
				t.Fatalf("Alg1 != VROOM for test %d", i)
			}
		}
	})
}

// BenchmarkVROOM benchmarks VROOM multiplication.
func BenchmarkVROOM(b *testing.B) {
	for _, bits := range []int{256, 512, 1024} {
		b.Run(fmt.Sprintf("%d-bit", bits), func(b *testing.B) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParams(p)
			a := randomInRange(p)
			bv := randomInRange(p)
			aM, aN := ToVROOMEncoding(a, params)
			bM, bN := ToVROOMEncoding(bv, params)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VROOM(aM, aN, bM, bN, params)
			}
		})
	}
}

// BenchmarkVROOMFast benchmarks Stage 1 (uint64 residues) VROOM.
func BenchmarkVROOMFast(b *testing.B) {
	for _, bits := range []int{256, 512, 1024} {
		b.Run(fmt.Sprintf("%d-bit", bits), func(b *testing.B) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsU64(p)
			a := randomInRange(p)
			bv := randomInRange(p)
			aM, aN := ToVROOMEncodingFast(a, params)
			bM, bN := ToVROOMEncodingFast(bv, params)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				VROOMFast(aM, aN, bM, bN, params)
			}
		})
	}
}

// TestVROOMFastCorrectness verifies Stage 1 gives same results as big.Int reference.
func TestVROOMFastCorrectness(t *testing.T) {
	for _, bits := range []int{64, 128, 256, 512, 1024} {
		t.Run(fmt.Sprintf("%d-bit", bits), func(t *testing.T) {
			p, _ := rand.Prime(rand.Reader, bits)
			params := SetupRNSParamsU64(p)

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
					t.Fatalf("mismatch at test %d:\n  want %s\n  got  %s", i, expected, got)
				}
			}
		})
	}
}

// BenchmarkBigIntMul benchmarks standard math/big modular multiplication.
func BenchmarkBigIntMul(b *testing.B) {
	for _, bits := range []int{256, 512, 1024} {
		b.Run(fmt.Sprintf("%d-bit", bits), func(b *testing.B) {
			p, _ := rand.Prime(rand.Reader, bits)
			a := randomInRange(p)
			bv := randomInRange(p)
			result := new(big.Int)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result.Mul(a, bv)
				result.Mod(result, p)
			}
		})
	}
}
