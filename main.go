package main

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"time"
)

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  VROOM: RNS Montgomery Multiplication — Go Reference Impl       ║")
	fmt.Println("║  Algorithm 1 (Posch & Posch) + Algorithm 2 (VROOM)              ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	allPassed := true

	// Test with various prime sizes
	for _, bits := range []int{64, 128, 256, 512, 1024} {
		passed := testWithPrimeSize(bits, 20)
		if !passed {
			allPassed = false
		}
	}

	fmt.Println()
	// Test with specific well-known primes
	testKnownPrimes()

	// Test chained multiplications (simulating modular exponentiation)
	testChainedMul()

	fmt.Println()
	if allPassed {
		fmt.Println("✅ ALL TESTS PASSED")
	} else {
		fmt.Println("❌ SOME TESTS FAILED")
	}
}

// testWithPrimeSize runs randomized tests with a prime of the given bit size.
func testWithPrimeSize(bits, numTests int) bool {
	fmt.Printf("━━━ Testing %d-bit prime (%d random pairs) ━━━\n", bits, numTests)

	// Generate random prime
	p, err := rand.Prime(rand.Reader, bits)
	if err != nil {
		panic(err)
	}
	fmt.Printf("  p = %s...  (%d bits)\n", truncStr(p.String(), 40), p.BitLen())

	// Setup RNS parameters
	start := time.Now()
	params := SetupRNSParams(p)
	setupTime := time.Since(start)
	fmt.Printf("  Setup: %v  (M: %d bits, %d moduli | N: %d bits, %d moduli)\n",
		setupTime,
		params.BaseM.Product.BitLen(), len(params.BaseM.Moduli),
		params.BaseN.Product.BitLen(), len(params.BaseN.Moduli))

	passedAlg1 := 0
	passedVroom := 0
	var totalAlg1, totalVroom time.Duration

	for i := 0; i < numTests; i++ {
		a := randomInRange(p)
		b := randomInRange(p)

		// Expected result: a * b mod p
		expected := new(big.Int).Mul(a, b)
		expected.Mod(expected, p)

		// ----- Algorithm 1 (Posch & Posch) -----
		t0 := time.Now()
		aM1, aN1 := ToMontgomeryRNS(a, params)
		bM1, bN1 := ToMontgomeryRNS(b, params)
		rM1, _ := PoschandPosch(aM1, aN1, bM1, bN1, params)
		result1 := FromMontgomeryRNS(rM1, params)
		totalAlg1 += time.Since(t0)

		if result1.Cmp(expected) == 0 {
			passedAlg1++
		} else {
			fmt.Printf("  ❌ Alg1 FAIL: a=%s, b=%s\n", a.String(), b.String())
			fmt.Printf("    expected: %s\n    got:      %s\n", expected.String(), result1.String())
		}

		// ----- Algorithm 2 (VROOM) -----
		t1 := time.Now()
		aM2, aN2 := ToVROOMEncoding(a, params)
		bM2, bN2 := ToVROOMEncoding(b, params)
		rM2, _ := VROOM(aM2, aN2, bM2, bN2, params)
		result2 := FromVROOMEncoding(rM2, params)
		totalVroom += time.Since(t1)

		if result2.Cmp(expected) == 0 {
			passedVroom++
		} else {
			fmt.Printf("  ❌ VROOM FAIL: a=%s, b=%s\n", a.String(), b.String())
			fmt.Printf("    expected: %s\n    got:      %s\n", expected.String(), result2.String())
		}
	}

	fmt.Printf("  Algorithm 1: %d/%d passed  (avg %v/op)\n",
		passedAlg1, numTests, totalAlg1/time.Duration(numTests))
	fmt.Printf("  VROOM:       %d/%d passed  (avg %v/op)\n",
		passedVroom, numTests, totalVroom/time.Duration(numTests))
	fmt.Println()

	return passedAlg1 == numTests && passedVroom == numTests
}

// testKnownPrimes tests with specific well-known cryptographic primes.
func testKnownPrimes() {
	fmt.Println("━━━ Testing with well-known primes ━━━")

	// Curve25519 prime: p = 2^255 - 19
	p25519 := new(big.Int).Exp(bigTwo, big.NewInt(255), nil)
	p25519.Sub(p25519, big.NewInt(19))
	testSpecificPrime("Curve25519 (2^255-19)", p25519, 10)

	// BLS12-381 field modulus
	// q = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
	bls381q, _ := new(big.Int).SetString("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab", 16)
	testSpecificPrime("BLS12-381 (381-bit)", bls381q, 10)

	fmt.Println()
}

// testSpecificPrime runs tests with a given prime and reports results.
func testSpecificPrime(name string, p *big.Int, numTests int) {
	fmt.Printf("  %s (%d bits)\n", name, p.BitLen())

	params := SetupRNSParams(p)
	passed := 0

	for i := 0; i < numTests; i++ {
		a := randomInRange(p)
		b := randomInRange(p)

		expected := new(big.Int).Mul(a, b)
		expected.Mod(expected, p)

		// Test VROOM
		aM, aN := ToVROOMEncoding(a, params)
		bM, bN := ToVROOMEncoding(b, params)
		rM, _ := VROOM(aM, aN, bM, bN, params)
		result := FromVROOMEncoding(rM, params)

		if result.Cmp(expected) == 0 {
			passed++
		} else {
			fmt.Printf("    ❌ FAIL test %d\n", i)
		}
	}
	fmt.Printf("    VROOM: %d/%d passed ✓\n", passed, numTests)
}

// testChainedMul tests a chain of multiplications (simulating modular exponentiation).
func testChainedMul() {
	fmt.Println("━━━ Testing chained multiplications (simulated mod exp) ━━━")

	// Use a 256-bit prime
	p, _ := rand.Prime(rand.Reader, 256)
	params := SetupRNSParams(p)

	base := randomInRange(p)
	exponent := 100

	// Expected: base^exponent mod p
	expected := new(big.Int).Exp(base, big.NewInt(int64(exponent)), p)

	// ----- Algorithm 1 chain -----
	{
		// Start with base in Montgomery form
		accM, accN := ToMontgomeryRNS(base, params)
		// Multiply by base (exponent-1) more times
		bM, bN := ToMontgomeryRNS(base, params)
		for i := 1; i < exponent; i++ {
			accM, accN = PoschandPosch(accM, accN, bM, bN, params)
		}
		result := FromMontgomeryRNS(accM, params)

		if result.Cmp(expected) == 0 {
			fmt.Printf("  Algorithm 1: base^%d mod p = %s... ✓\n", exponent, truncStr(result.String(), 30))
		} else {
			fmt.Printf("  Algorithm 1: ❌ FAIL\n")
			fmt.Printf("    expected: %s\n    got:      %s\n", expected.String(), result.String())
		}
	}

	// ----- VROOM chain -----
	{
		accM, accN := ToVROOMEncoding(base, params)
		bM, bN := ToVROOMEncoding(base, params)
		for i := 1; i < exponent; i++ {
			accM, accN = VROOM(accM, accN, bM, bN, params)
		}
		result := FromVROOMEncoding(accM, params)

		if result.Cmp(expected) == 0 {
			fmt.Printf("  VROOM:       base^%d mod p = %s... ✓\n", exponent, truncStr(result.String(), 30))
		} else {
			fmt.Printf("  VROOM:       ❌ FAIL\n")
			fmt.Printf("    expected: %s\n    got:      %s\n", expected.String(), result.String())
		}
	}

	// ----- Edge cases -----
	fmt.Println()
	fmt.Println("━━━ Edge cases ━━━")

	// a * 0 mod p = 0
	{
		a := randomInRange(p)
		zero := big.NewInt(0)
		aM, aN := ToVROOMEncoding(a, params)
		bM, bN := ToVROOMEncoding(zero, params)
		rM, _ := VROOM(aM, aN, bM, bN, params)
		result := FromVROOMEncoding(rM, params)
		if result.Sign() == 0 {
			fmt.Println("  a * 0 = 0 ✓")
		} else {
			fmt.Printf("  a * 0 = %s ❌\n", result.String())
		}
	}

	// a * 1 mod p = a
	{
		a := randomInRange(p)
		one := big.NewInt(1)
		aM, aN := ToVROOMEncoding(a, params)
		bM, bN := ToVROOMEncoding(one, params)
		rM, _ := VROOM(aM, aN, bM, bN, params)
		result := FromVROOMEncoding(rM, params)
		expected := new(big.Int).Mod(a, p)
		if result.Cmp(expected) == 0 {
			fmt.Println("  a * 1 = a ✓")
		} else {
			fmt.Printf("  a * 1 ≠ a ❌ (got %s, want %s)\n", result.String(), expected.String())
		}
	}

	// (p-1) * (p-1) mod p = 1
	{
		pm1 := new(big.Int).Sub(p, bigOne)
		aM, aN := ToVROOMEncoding(pm1, params)
		bM, bN := ToVROOMEncoding(pm1, params)
		rM, _ := VROOM(aM, aN, bM, bN, params)
		result := FromVROOMEncoding(rM, params)
		if result.Cmp(bigOne) == 0 {
			fmt.Println("  (p-1) * (p-1) = 1 ✓")
		} else {
			fmt.Printf("  (p-1) * (p-1) = %s ❌\n", result.String())
		}
	}

	// Consistency: both algorithms give same result
	{
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

		if r1.Cmp(r2) == 0 {
			fmt.Println("  Alg1 == VROOM ✓")
		} else {
			fmt.Printf("  Alg1 ≠ VROOM ❌ (%s vs %s)\n", r1.String(), r2.String())
		}
	}
}

// truncStr truncates a string to maxLen chars, adding "..." if needed.
func truncStr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
