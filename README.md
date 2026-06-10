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

This repository implements the full algorithm in Go with custom AVX512 assembly, achieving **799 ns per 1024-bit modular multiplication** — a 63× speedup over the `math/big` baseline. On top of multiplication, the library provides **modular exponentiation** (`a^e mod p`) with a precomputed-table strategy that achieves **423 μs for 1024-bit exponentiation** — 1.4× faster than Go's `big.Int.Exp`.

---

## Performance

Measured on Intel Xeon Gold 6326 (Ice Lake) @ 2.90 GHz, single core, 1024-bit prime.

### Modular multiplication (single `a · b mod p`)

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

### Modular exponentiation (`a^e mod p`)

Three strategies, each building on the previous:

| Benchmark | ns/op | allocs | B/op | Strategy |
|-----------|-------|--------|------|----------|
| **Precomputed 1024-bit exp** | **423,000** | **0** | **0** | precomputed table, ~512 VROOM muls |
| Inner naive 1024-bit exp | 1,244,000 | 0 | 0 | square-and-multiply, ~1535 VROOM calls |
| Inner CT 1024-bit exp | 1,684,000 | 0 | 0 | always square+multiply, +35% |
| Inner RSA verify (e=65537) | 13,700 | 0 | 0 | 17 VROOM calls |
| Full RSA verify (e=65537) | 67,800 | 451 | 36 KB | encode + 17 VROOM calls + decode |
| Full 1024-bit exp | 1,303,000 | 451 | 36 KB | encode/decode adds ~59 μs |
| `big.Int.Exp` (baseline) | 586,000 | 22 | 6 KB | windowed Montgomery, scalar |

**Precomputed table (423 μs — 1.4× faster than `big.Int.Exp`):** Precompute `base^(2^i)` for every bit position `i` and store in VROOM form. At runtime, scan the exponent LSB→MSB and multiply only for set bits — no squaring at runtime, only ~512 multiplies for a random 1024-bit exponent. Precompute cost: 1023 squarings (~828 μs, one-time). Memory: ~360 KB. Amortizes after a single exponentiation — ideal when base/modulus are reused (RSA, DH, repeated signatures).

**Naive square-and-multiply (1,244 μs):** Left-to-right binary scan. 1023 squares + ~512 multiplies = ~1535 VROOM calls. Per-call cost: 1,244,000 / 1535 ≈ 810 ns, consistent with standalone VROOMStage4 (799 ns). 2.1× slower than `big.Int.Exp` due to more word-level operations per multiply (22² vs 16² for O(n²) Montgomery) and no windowing.

**RSA verify (e=65537, 13.7 μs):** The exponent 65537 = 2¹⁶+1 has only 2 set bits → 16 squares + 1 multiply = 17 VROOM calls. The Full path spends 54 μs (80% of total) on encode/decode; the Inner path eliminates this.

**Constant-time overhead (+35%):** The CT variant always performs both square and multiply (2 VROOM calls per bit), then uses branchless `ctCondCopy` to select the result. Theoretical overhead: 2.0/1.5 = +33%, measured +35%.

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

### Modular exponentiation — Square-and-multiply

Given base `a` and exponent `e`, compute `a^e mod p` by scanning `e` from MSB to LSB:

```
acc = a                              [MSB is always 1]
for i = bitlen(e)-2 downto 0:
    acc = acc * acc                  [square — 1 VROOM call]
    if bit(e, i) == 1:
        acc = acc * a                [multiply — 1 VROOM call]
return acc
```

For a random n-bit exponent: (n-1) squares + ~(n-1)/2 multiplies ≈ **1.5·(n-1)** VROOM calls total.

**Constant-time variant:** Always executes both square and multiply, then uses branchless conditional copy to select the correct result:

```
for i = bitlen(e)-2 downto 0:
    sq  = acc * acc                  [square — always]
    mul = sq  * a                    [multiply — always]
    acc = bit(e,i) ? mul : sq        [ctCondCopy — branchless]
```

This eliminates data-dependent branches on the exponent bits, at the cost of **2·(n-1)** VROOM calls (vs 1.5·(n-1) for the non-CT path). The `ctCondCopy` function uses bitwise masking (`dst = (ifOne & mask) | (ifZero & ^mask)`) with no branches, no early exits, and no timing variation.

### Precomputed-table exponentiation

Precompute `base^(2^i)` for every bit position, then decompose `base^exp` as a product of table entries:

```
Precompute (one-time):
    table[0] = base                          [encode]
    table[i] = table[i-1]²                   [1023 VROOM squarings]

Runtime (per exponentiation):
    acc = 1
    for i = 0 to bitlen(exp)-1:              [LSB → MSB]
        if bit(exp, i) == 1:
            acc = acc × table[i]             [VROOM multiply]
    return acc
```

Mathematical basis: `base^exp = ∏ base^(2^i)` for each bit `i` where `exp` has a 1. A random 1024-bit exponent has ~512 set bits, so runtime is **~512 VROOM calls** — no squaring, no windowing logic, just multiplies.

**Three API levels:**

| API | Use case | Allocations | Runtime VROOM calls |
|-----|----------|-------------|---------------------|
| `ModExpVROOM` | One-shot, convenience | 451 | ~1535 (naive) |
| `ModExpInner` | Zero-alloc inner loop | 0 | ~1535 (naive) |
| `ModExpPrecomputed` | Pre-built table, zero-alloc | 0 | ~512 (table lookup) |

The precomputed table (`VROOMPreTable`) stores `base^(2^i)` in VROOM encoding with a pre-encoded identity element (1). The runtime path uses only `copy()` and `VROOMStage4` — zero allocations, zero `big.Int` in the hot path.

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

### Modular exponentiation — Square-and-multiply over VROOM

Built on top of Stage 4. Each exponentiation is a sequence of VROOMStage4 calls orchestrated by the square-and-multiply loop. Two key design decisions:

**Zero-allocation inner loop:** The `ModExpWorkspace` pre-allocates all buffers (`accM/accN` for the accumulator, `sqM/sqN` for the CT square result) once. The inner loop uses only `copy()` and VROOMStage4 — no `big.Int`, no `make()`, no GC pressure. Verified at 0 B/op, 0 allocs/op.

**Separation of encoding from computation:** The Full API (`ModExpVROOM`) pays 54 μs for big.Int ↔ VROOM conversion — acceptable for one-shot calls, but 80% of the total time for short exponents like RSA's e=65537. The Inner API (`ModExpInner`) eliminates this entirely, enabling 13.7 μs RSA verify when the base is pre-encoded.

### Precomputed-table exponentiation — trading memory for speed

The key insight: in square-and-multiply, squarings are determined entirely by the exponent length, not its value. By precomputing `base^(2^i)` for every bit position, all squarings move to a one-time setup phase. At runtime, only multiplications remain — and only for set bits (~50% of them).

**Precompute phase (one-time):** 1023 VROOMStage4 squarings, each entry is the square of the previous. Cost: ~828 μs. Memory: 1024 × (tM + tN) uint64s ≈ 360 KB. The identity element (1) is also pre-encoded to avoid runtime allocation.

**Runtime phase (per exponentiation):** Scan exponent LSB→MSB. For each set bit `i`, multiply accumulator by `table[i]`. A random 1024-bit exponent has ~512 set bits → ~512 VROOM calls. Measured: 423 μs, zero allocations.

**Trade-off:** 360 KB memory + 828 μs one-time setup → 3× faster runtime per exponentiation. Amortizes after a single call. Ideal for RSA (same key, many operations), DH (same generator), or any repeated-base scenario.

**Result:** First time VROOM exponentiation beats `big.Int.Exp` at 1024-bit — 423 μs vs 586 μs (1.4× faster), with zero allocations vs 22.

---

## Project structure

```
vroom-go/
├── rns.go                    # big.Int reference: RNSBase, MontParams, CRNS, VROOM
├── rns_fast.go               # Stage 1: uint64 residues, mulmod, addmod
├── rns_noalloc.go            # Stage 2.5: zero-allocation workspace
├── rns_52bit.go              # 52-bit prime generation, setup helpers
├── rns_stage2.go             # Stage 2: matrix CRNS, k192 estimator
├── rns_stage3.go             # Stage 3: AVX512 CRNS (broadcastMulAcc52)
├── rns_stage4.go             # Stage 4: register-resident + Shoup/Barrett
├── modexp.go                 # Modular exponentiation: square-and-multiply
├── modexp_precompute.go      # Precomputed-table exponentiation
├── avx512_amd64.go           # Go stubs — Stage 3 assembly
├── avx512_amd64.s            # Assembly: vpmadd52, broadcastMulAcc52
├── avx512v2_amd64.go         # Go stubs — Stage 4 assembly
├── avx512v2_amd64.s          # Assembly: matvecAVX512_3g, _6g, Gen
├── avx512_test.go            # Stage 3 tests
├── rns_stage4_test.go        # Stage 4 tests + benchmarks
├── modexp_test.go            # Modular exponentiation tests + benchmarks
├── modexp_precompute_test.go # Precomputed-table tests + benchmarks
├── rns_test.go               # Reference tests
├── main.go                   # Demo
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

# Modular exponentiation tests
AVX512_TEST=1 go test -v -run TestModExp

# Precomputed-table exponentiation tests
AVX512_TEST=1 go test -v -run TestModExpPrecomputed

# Benchmarks — multiplication
AVX512_TEST=1 go test -bench BenchmarkVROOMStage3vs4 -benchmem -count=3
AVX512_TEST=1 go test -bench BenchmarkApplyStage4_Parts -benchmem -count=3

# Benchmarks — exponentiation (all strategies)
AVX512_TEST=1 go test -bench BenchmarkModExp -benchmem -count=3

# Benchmarks — 2048-bit
AVX512_TEST=1 go test -bench "Benchmark.*2048" -benchmem -count=3
```

Requires Go 1.22+ and AVX512IFMA hardware (Intel Ice Lake / Sapphire Rapids) or [Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) for the vectorized path.

---

## Test coverage

### Correctness tests — 10,182+ total

**Multiplication pipeline (9,522 tests):**

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

**Naive modular exponentiation (270+ tests):**

| Test | What it checks | Data points |
|------|---------------|-------------|
| TestModExpVROOM_Small | Deterministic small values (3¹³ mod 7) + 16-bit random prime | 2 |
| TestModExpVROOM_EdgeCases | a⁰=1, a¹=a, a²=a·a, Fermat a^(p-1)=1 | 4 |
| TestModExpVROOM_Random | Random (base, exp) vs big.Int.Exp, 5 prime sizes (64–1024 bit) | 100 (20×5) |
| TestModExpVROOM_RSAPublicExponent | e=65537, 3 prime sizes (256–1024 bit) | 30 (10×3) |
| TestModExpVROOMConstTime_Random | CT path vs big.Int.Exp, 5 prime sizes | 100 (20×5) |
| TestModExpVROOMConstTime_MatchesNonConstTime | CT path = non-CT path, 512-bit | 30 |
| TestModExpVROOMConstTime_Fermat | CT a^(p-1) mod p = 1, 256-bit | 10 |

**Precomputed-table exponentiation (390+ tests):**

| Test | What it checks | Data points |
|------|---------------|-------------|
| TestModExpPrecomputed_Small | Deterministic small values (3¹³ mod 7), exp=0, exp=1 | 3 |
| TestModExpPrecomputed_EdgeCases | a⁰=1, a¹=a, a²=a·a, Fermat a^(p-1)=1 | 4 |
| TestModExpPrecomputed_Random | Random (base, exp) vs big.Int.Exp, 5 prime sizes (64–1024 bit) | 100 (20×5) |
| TestModExpPrecomputed_MatchesNaive | Precomputed = naive ModExpVROOM, 512-bit | 30 |
| TestModExpPrecomputed_RSAPublicExponent | e=65537, 3 prime sizes (256–1024 bit) | 30 (10×3) |
| TestModExpPrecomputed_ReuseTable | 1 table, 50 different exponents, 512-bit | 50 |
| TestModExpPrecomputedConstTime_Random | CT precomputed vs big.Int.Exp, 5 sizes | 100 (20×5) |
| TestModExpPrecomputedConstTime_MatchesNonConstTime | CT = non-CT precomputed, 512-bit | 30 |
| TestModExpPrecomputedConstTime_Fermat | CT precomputed Fermat test, 256-bit | 10 |

Every exponentiation test compares against Go's `math/big.Int.Exp` as the reference oracle. The Fermat tests provide an independent mathematical invariant (a^(p-1) ≡ 1 mod p for prime p) that doesn't depend on any reference implementation.

### Benchmarks — ~60,000,000+ iterations (automatic)

**Multiplication:**

| Benchmark | Iterations × 3 |
|-----------|----------------|
| BenchmarkVROOMStage4/1024-bit | ~4,500,000 |
| BenchmarkApplyStage4_Parts/full_Apply | ~13,400,000 |
| BenchmarkApplyStage4_Parts/step1_matvec | ~47,000,000 |

**Exponentiation (naive):**

| Benchmark | Iterations × 3 |
|-----------|----------------|
| BenchmarkModExpInner_RSAVerify_1024 | ~260,000 |
| BenchmarkModExpInner_1024bit_exp | ~2,900 |
| BenchmarkModExpInnerConstTime_1024bit_exp | ~2,100 |
| BenchmarkModExpVROOM_RSAVerify_1024 | ~53,000 |
| BenchmarkModExpVROOM_1024bit_exp | ~2,700 |
| BenchmarkModExpVROOMConstTime_1024bit_exp | ~2,100 |
| BenchmarkModExpBigInt_1024bit_exp (baseline) | ~6,000 |

**Exponentiation (precomputed table):**

| Benchmark | Iterations × 3 |
|-----------|----------------|
| BenchmarkModExpPrecomputed_1024bit_exp | ~8,300 |
| BenchmarkModExpPrecomputedConstTime_1024bit_exp | ~4,100 |
| BenchmarkModExpPrecomputed_RSAVerify_1024 | ~166,000 |
| BenchmarkNewVROOMPreTable_1024 (setup cost) | variable |

---

## Roadmap

- [x] Native `uint64` residue arithmetic with `math/bits`
- [x] CRNS via precomputed matrix (no CRT reconstruction)
- [x] 52-bit moduli for VPMADD52 alignment
- [x] AVX512IFMA vectorized matvec via Go assembly
- [x] Register-resident kernel (Z0-Z15, 3g/6g/Gen variants)
- [x] Division-free reduction (Shoup + Barrett, zero DIVQ in hot path)
- [x] Modular exponentiation (`a^e mod p`) — square-and-multiply, zero-alloc inner loop, constant-time variant
- [x] Precomputed-table exponentiation — 423 μs for 1024-bit, 1.4× faster than `big.Int.Exp`
- [ ] Shoup/Barrett for elementwise ops (remaining ~66 DIVQ per VROOM call)
- [ ] 2048/4096-bit benchmarks (where VROOM's AVX512 parallelism overtakes scalar further)
- [ ] RSA-CRT with interleaved dual CRNS
- [ ] BLS12-381 field extension arithmetic (`F²q`, `F¹²q`)
- [ ] Batching for latency hiding (paper Table 11)

---

## References

- [VROOM paper (MIT)](https://github.com/SimonLangowski/VROOM)
- [Posch & Posch 1995](https://ieeexplore.ieee.org/document/381846) — original RNS Montgomery algorithm
- [Montgomery 1985](https://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777282-X/) — Montgomery modular multiplication
