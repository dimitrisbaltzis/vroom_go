# vroom-go

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Go](https://img.shields.io/badge/Go-1.22+-00ADD8.svg)
![AVX512IFMA](https://img.shields.io/badge/AVX512-IFMA-orange.svg)

A Go implementation of the **VROOM** RNS Montgomery modular multiplication algorithm, based on the paper:

> _VROOM: Accelerating (Almost All) Number-Theoretic Cryptography Using Vectorization and the Residue Number System_
> Simon Langowski, Kaiwen He, Srinivas Devadas — MIT

---

## What is this

Modular multiplication (`a · b mod p`) is the core bottleneck in number-theoretic cryptography — RSA, ECDSA, BLS signatures, zero-knowledge proofs. The VROOM paper shows how to speed it up significantly on modern CPUs using:

- A **Residue Number System (RNS)** representation that breaks large integers into small residues
- **Montgomery multiplication** adapted to work natively in RNS
- **AVX512IFMA** vector instructions for parallel 52-bit multiply-accumulate

This repository implements the full algorithm in Go with custom AVX512 assembly, achieving **799 ns per 1024-bit modular multiplication** — a 63× speedup over the `math/big` baseline.

---

## Performance

Measured on Intel Xeon Gold 6326 (Ice Lake) @ 2.90 GHz, single core, 1024-bit prime.

| Stage | ns/op | Speedup | Key change |
|-------|-------|---------|------------|
| Baseline (`math/big`) | ~50,000 | 1× | heap-allocated big.Int arithmetic |
| Stage 1 (uint64) | ~15,000 | ~3× | native uint64 residues, zero allocations |
| Stage 2 (matrix CRNS) | 5,671 | ~9× | matrix-vector CRNS, 192-bit k estimator |
| Stage 3 (AVX512IFMA) | 1,750 | ~29× | VPMADD52 vectorized matvec |
| **Stage 4** | **799** | **~63×** | register-resident kernel + Shoup/Barrett |

Stage 4 Apply breakdown (CRNS base change, 269 ns total):

| Step | ns | What changed |
|------|----|-------------|
| Matvec kernel | 77 | 6 ZMM accumulators in registers, 1 kernel call instead of 63 |
| Combine (Shoup+Barrett) | 75 | 2×MULQ replaces DIVQ (~35→10 cycles per lane) |
| k estimation (192-bit) | 46 | unchanged |
| Correction (Shoup) | 21 | precomputed quotients, single conditional subtract |

Zero heap allocations, zero `math/big` in the hot path, constant-time execution.

Comparison with the paper's C++ implementation: ~1.6× slower (799 vs ~500 ns), attributable to Go overhead and different hardware (Ice Lake vs Sapphire Rapids).

---

## Algorithms

### Algorithm 1 — Posch & Posch (1995)

The foundational RNS Montgomery algorithm. Given `a, b` in Montgomery-RNS form:

```
q_M  =  a_M · b_M · (−p⁻¹)  mod M        [elementwise]
q_N  =  CRNS_{M→N}(q_M)
r_N  =  (a_N · b_N + q_N · p) · M⁻¹  mod N
r_M  =  CRNS_{N→M}(r_N)
```

### Algorithm 2 — VROOM

The optimized version. Uses a modified encoding `T ≡ 1 (mod M), T ≡ M⁻¹ (mod N)` to absorb constant multiplications into the CRNS matrix:

```
q_M  =  a_M · b_M  mod M                  [no −p⁻¹ factor]
r_N  =  (a_N · b_N + CRNS^{M·(−p⁻¹)}_{N·(p·M⁻²)}(q_M))  mod N
r_M  =  CRNS^{N·M}_{M·1}(r_N)
```

Each CRNS internally performs: matrix-vector product → fixed-point k estimation → correction.

---

## Optimization stages

### Stage 1 — uint64 residues

Replace `[]*big.Int` with `[]uint64`. Elementwise multiply via `bits.Mul64` + `bits.Div64`. Zero allocations. CRNS still uses full CRT reconstruction via `big.Int` (bottleneck).

### Stage 2 — Matrix CRNS

Replace CRT reconstruction with precomputed matrix A[i][j], fixed-point vector f[i], correction vector c[j] (paper Appendix A). At runtime: matrix-vector product + 192-bit dot product for k + correction. No `big.Int` in hot path.

### Stage 3 — AVX512IFMA

Vectorize the matrix-vector product with `VPMADD52LUQ`/`VPMADD52HUQ`. Each instruction does 8 parallel 52-bit multiply-accumulates. For 1024-bit: 22 source residues × 3 groups of 8 lanes = 63 Go→assembly calls to `broadcastMulAcc52`. Elementwise reductions still use scalar DIVQ.

### Stage 4 — Register-resident kernel + division-free reduction

Two optimizations combined:

**Register-resident matvec kernel** (`matvecAVX512_3g`): All 6 ZMM accumulators (Z0-Z5) stay in registers across the entire source loop. 6 independent VPMADD52 per iteration — zero data dependencies — OOO engine dispatches 2/cycle on Ice Lake. One kernel call replaces 63 function calls, eliminating ~950 cycles of call overhead and 252 memory load/stores.

**Shoup/Barrett reduction**: Replace 42 DIVQ instructions (~35 cycles each) with Shoup's method (2×MULQ ≈ 10 cycles) and Barrett reduction (1×MULQ ≈ 8 cycles). Constants precomputed at setup.

---

## Project structure

```
vroom-go/
├── rns.go              # big.Int reference: RNSBase, MontParams, CRNS, VROOM
├── rns_fast.go         # Stage 1: uint64 residues, mulmod, addmod
├── rns_noalloc.go      # Stage 2.5: zero-allocation workspace
├── rns_52bit.go        # 52-bit prime generation, setup helpers
├── rns_stage2.go       # Stage 2: matrix CRNS, k192 estimator
├── rns_stage3.go       # Stage 3: AVX512 CRNS (broadcastMulAcc52)
├── rns_stage4.go       # Stage 4: register-resident + Shoup/Barrett
├── avx512_amd64.go     # Go stubs — Stage 3 assembly
├── avx512_amd64.s      # Assembly: vpmadd52, broadcastMulAcc52
├── avx512v2_amd64.go   # Go stubs — Stage 4 assembly
├── avx512v2_amd64.s    # Assembly: matvecAVX512_3g, _6g, Gen
├── avx512_test.go      # Stage 3 tests
├── rns_stage4_test.go  # Stage 4 tests + benchmarks
├── rns_test.go         # Reference tests
├── main.go             # Demo
└── go.mod
```

---

## Running

```bash
# Compile
go build ./...

# Unit tests (no AVX512 hardware needed)
go test -v -run "TestMulmodShoup52|TestBarrettReduce52"

# Full tests (requires AVX512IFMA hardware or Intel SDE)
AVX512_TEST=1 go test -v -run "TestVROOMStage4|TestMatvec"

# Benchmarks
AVX512_TEST=1 go test -bench BenchmarkVROOMStage3vs4 -benchmem -count=3
AVX512_TEST=1 go test -bench BenchmarkApplyStage4_Parts -benchmem -count=3
```

Requires Go 1.22+ and AVX512IFMA hardware (Intel Ice Lake / Sapphire Rapids) or [Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) for the vectorized path.

---

## Test coverage

### Correctness tests — 9,522 total

| Test | What it checks | Data points |
|------|---------------|-------------|
| TestMulmodShoup52 | Shoup multiply vs big.Int reference | 5,000 random |
| TestBarrettReduce52 | Barrett reduce vs native modulo | 4,000 random |
| TestMatvecAVX512_3g | 3-group kernel vs scalar reference | 48 values, deterministic |
| TestMatvecAVX512_3g_Large | 3-group kernel, realistic 1024-bit | 48 values, random |
| TestMatvecAVX512Gen | Generic kernel, 2 groups | 32 values, random |
| TestMatvecAVX512_6g | 6-group kernel, 2048-bit | 96 values, random |
| TestVROOMStage4 | Full pipeline, 5 prime sizes (64–1024 bit) | 150 multiplications |
| TestVROOMStage4Chained | base^100 mod p, error accumulation | 99 chained, 256-bit |
| TestVROOMStage4_1024Chained | base^50 mod p, 1024-bit | 49 chained |

### Benchmarks — ~60,000,000 iterations (automatic)

| Benchmark | Iterations × 3 |
|-----------|----------------|
| BenchmarkVROOMStage4/1024-bit | ~4,500,000 |
| BenchmarkApplyStage4_Parts/full_Apply | ~13,400,000 |
| BenchmarkApplyStage4_Parts/step1_matvec | ~47,000,000 |

---

## Roadmap

- [x] Native `uint64` residue arithmetic with `math/bits`
- [x] CRNS via precomputed matrix (no CRT reconstruction)
- [x] 52-bit moduli for VPMADD52 alignment
- [x] AVX512IFMA vectorized matvec via Go assembly
- [x] Register-resident kernel (Z0-Z15, 3g/6g/Gen variants)
- [x] Division-free reduction (Shoup + Barrett, zero DIVQ in hot path)
- [ ] Shoup/Barrett for elementwise ops (remaining ~66 DIVQ per VROOM call)
- [ ] Modular exponentiation (`a^e mod p`)
- [ ] RSA-CRT with interleaved dual CRNS
- [ ] BLS12-381 field extension arithmetic (`F²q`, `F¹²q`)
- [ ] Batching for latency hiding (paper Table 11)

---

## References

- [VROOM paper (MIT)](https://github.com/SimonLangowski/VROOM)
- [Posch & Posch 1995](https://ieeexplore.ieee.org/document/381846) — original RNS Montgomery algorithm
- [Montgomery 1985](https://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777282-X/) — Montgomery modular multiplication
