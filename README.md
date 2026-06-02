# vroom-go

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Go](https://img.shields.io/badge/Go-1.22+-00ADD8.svg)

A Go implementation of the **VROOM** RNS Montgomery modular multiplication algorithm, based on the paper:

> _VROOM: Accelerating (Almost All) Number-Theoretic Cryptography Using Vectorization and the Residue Number System_
> Simon Langowski, Kaiwen He, Srinivas Devadas — MIT

---

## What is this

Modular multiplication (`a · b mod p`) is the core bottleneck in number-theoretic cryptography — RSA, ECDSA, BLS signatures, zero-knowledge proofs. The VROOM paper shows how to speed it up significantly on modern CPUs using:

- A **Residue Number System (RNS)** representation that breaks large integers into small residues
- **Montgomery multiplication** adapted to work natively in RNS
- **AVX512IFMA** vector instructions for parallel 52-bit multiplications

This repository implements the algorithm in pure Go using `math/big` for correctness and clarity, with the full AVX512 path as the next step.

---

## Algorithms implemented

### Algorithm 1 — Posch & Posch (1995)

The foundational RNS Montgomery algorithm. Given `a, b` in Montgomery-RNS form:

```
q_M  =  a_M · b_M · (−p⁻¹)  mod M        [elementwise]
q_N  =  CRNS_{M→N}(q_M)
r_N  =  (a_N · b_N + q_N · p) · M⁻¹  mod N
r_M  =  CRNS_{N→M}(r_N)
```

### Algorithm 2 — VROOM

The optimized version. Uses a modified encoding `T ≡ 1 (mod M), T ≡ M⁻¹ (mod N)` to absorb constant multiplications into the CRNS matrix, reducing the number of elementwise multiplications:

```
q_M  =  a_M · b_M  mod M                  [no −p⁻¹ factor]
r_N  =  (a_N · b_N + CRNS^{M·(−p⁻¹)}_{N·(p·M⁻²)}(q_M))  mod N
r_M  =  CRNS^{N·M}_{M·1}(r_N)
```

Multiplication count: `2t² + 13t` vs `2t² + 2t` schoolbook — the gap is closed by AVX512 parallelism.

---

## Project structure

```
vroom-go/
├── rns.go        # Core implementation: RNSBase, MontParams, CRNS, VROOM, Posch & Posch
├── main.go       # Demo: tests across prime sizes, known primes, chained multiplications
├── rns_test.go   # Test suite and benchmarks
└── go.mod
```

---

## Running

```bash
# Demo (random and well-known primes, edge cases)
go run .

# Full test suite
go test -v

# Benchmarks
go test -bench=. -benchmem
```

Requires Go 1.22+.

---

## Test coverage

| Test                   | What it checks                                                 |
| ---------------------- | -------------------------------------------------------------- |
| `TestPoschandPosch`    | Algorithm 1 vs `math/big`, 50 random pairs, 64–1024 bit primes |
| `TestVROOM`            | Algorithm 2 vs `math/big`, 50 random pairs, 64–1024 bit primes |
| `TestVROOMKnownPrimes` | Curve25519 (`2²⁵⁵ − 19`) and BLS12-381 (381-bit)               |
| `TestChainedVROOM`     | `base^100 mod p` via 99 sequential multiplications             |
| `TestEdgeCases`        | `a·0=0`, `a·1=a`, `(p−1)²=1`, consistency Alg1 == VROOM        |

---

## Performance note

This implementation uses `math/big` internally for all residue arithmetic. It is a **reference implementation** — correct, readable, but not fast. The speedups reported in the paper (up to 4× over OpenSSL for RSA-4096) come from:

1. Native `uint64` residue arithmetic with `math/bits.Mul64` for 128-bit products
2. The CRNS matrix-vector formulation (Appendix A of the paper) without CRT reconstruction
3. AVX512IFMA instructions: 8 parallel 52-bit multiply-accumulates per cycle

Stage 2 (rns_stage2.go) eliminates big.Int from the CRNS hot path entirely,
closing the gap with math/big from ~80× to ~4× for 256-bit primes.

---

## Roadmap

- [ ] Replace `math/big` residues with native `uint64` arithmetic
- [x] Implement CRNS via precomputed matrix (no CRT reconstruction)
      Matrix-vector product with fixed-point k estimation (Appendix A).
      Precision: 52 + ceil(log2(t)) bits. Zero big.Int at runtime.
      Result: ~20× speedup over Stage 1, allocations 508 → 7 per op.
- [ ] AVX512IFMA path via Go assembly (`.s` files)
- [ ] Modular exponentiation using VROOM
- [ ] BLS12-381 field extension arithmetic (`F²q`, `F¹²q`)

---

## References

- [VROOM paper (MIT)](https://github.com/SimonLangowski/VROOM)
- [Posch & Posch 1995](https://ieeexplore.ieee.org/document/381846) — original RNS Montgomery algorithm
- [Montgomery 1985](https://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777282-X/) — Montgomery modular multiplication
