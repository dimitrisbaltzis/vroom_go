# VROOM vs Notus — Comparison Benchmarks

Fair comparison of two modular exponentiation approaches on the same machine.

## Background

Both libraries compute `base^exp mod p` but with fundamentally different strategies:

**VROOM** ([paper](https://github.com/SimonLangowski/VROOM)) decomposes large integers into small 52-bit residues (RNS representation) and uses AVX512IFMA SIMD instructions (`VPMADD52`) for 8-wide parallel multiply-accumulate. Single-threaded, but exploits instruction-level parallelism.

**Notus** stays in classical Montgomery representation and exploits two things: thread-level parallelism (splits exponent words across goroutines) and GCW (Greatest Common Words — extracts common bits across multiple exponents to avoid redundant multiplications).

They are **not directly iso** in all scenarios — see the notes below.

---

## Prerequisites

**1. AVX512IFMA hardware** — required for VROOM:

```bash
grep -o 'avx512ifma' /proc/cpuinfo | head -1
# Must output: avx512ifma
```

Intel Ice Lake+, Sapphire Rapids, or AMD Zen 4+. If empty → VROOM benchmarks will fail with illegal instruction.

**2. Go 1.22+**

```bash
go version
```

**3. CPU info for your report:**

```bash
lscpu | grep "Model name"
```

---

## Setup (one-time)

```bash
cd comparison/
go mod tidy      # resolves jiajunxin/multiexp version
go build ./...   # compile check — must produce no output
go vet ./...     # static analysis
```

---

## Run

```bash
# Step 1: Correctness — both must match big.Int.Exp (reference oracle)
AVX512_TEST=1 go test -v -run TestCorrectness

# Step 2: Benchmarks
AVX512_TEST=1 go test -bench BenchmarkComparison \
    -benchmem -benchtime 5s -count 3 | tee results.txt

# Step 3: Statistical analysis
go install golang.org/x/perf/cmd/benchstat@latest
benchstat results.txt
```

---

## What is compared

| Scenario       | Notus                   | VROOM                     | Iso?                                                                                  |
| -------------- | ----------------------- | ------------------------- | ------------------------------------------------------------------------------------- |
| **Precompute** | `NewPrecomputeTable`    | `NewVROOMWindowTable`     | ✓                                                                                     |
| **Single exp** | `ExpParallel` (1T + NT) | `ModExpWindowed`          | Partial — see note 1                                                                  |
| **Batch 100**  | 100× `ExpParallel`      | 100× `ModExpWindowed`     | ✓ amortized precompute                                                                |
| **Fourfold**   | `FourfoldExp` (GCW)     | 4× `ModExpWindowed`       | ✗ Notus extracts common bits across 4 exponents (GCW), VROOM runs 4 independent calls |
| **Double**     | `DoubleExp` (GCW)       | 2× `ModExpWindowed`       | ✗ same as above, for 2 exponents2                                                     |
| **RSA verify** | `ExpParallel(e=65537)`  | `ModExpWindowed(e=65537)` | ✓                                                                                     |
| **Baseline**   | —                       | `big.Int.Exp` (stdlib)    | reference                                                                             |

---

## Important notes for interpreting results

**Note 1 — Single exp, parallelism:**
Notus with N threads exploits multiple cores; VROOM is single-threaded but uses AVX512 SIMD (8 lanes). The benchmark runs Notus at both 1T and NT so you can separate the two effects.

**Note 2 — Fourfold/Double is not iso:**
Notus' `FourfoldExp` uses GCW to extract bits common across all 4 exponents and compute them once. VROOM has no native multi-exp — it runs 4 independent exponentiations. This gives Notus a structural advantage in the fourfold scenario that is algorithmic, not just a hardware difference.

**Note 3 — Montgomery vs RNS:**
Notus uses classical Montgomery (one large multi-word integer, scalar arithmetic). VROOM uses RNS (many small 52-bit residues, SIMD arithmetic). They have different constant factors and scale differently with bit size — check both 1024-bit and 2048-bit results.

**Note 4 — When each wins:**

- Single random exp, same base reused → likely VROOM (AVX512 + windowed table)
- 4 exponents with same base, no hardware SIMD → likely Notus (GCW)
- RSA verify (e=65537, 17 multiplications) → likely VROOM (very few VROOM calls needed)
- 2048-bit → check results, VROOM scales better per multiply but has more residues

---

## Go 1.22+ compatibility

jiajunxin/multiexp uses internal `math/big` symbols via assembly trampolines (`shrVU`, `divWW`, `mulWW`, `reciprocalWord`, `nlz`) that were removed in Go 1.22+. To fix this, the library was forked at `github.com/dimitrisbaltzis/multiexp`, the assembly files (`arith_*.s`, `arith_decl.go`) were removed, and replaced with pure Go implementations. The `go.mod` uses a `replace` directive to point to the fork.

---

## Results

Measured on Intel Xeon Gold 6326 (Ice Lake) @ 2.90 GHz. Full analysis in [benchmark_analysis.md](benchmark_analysis.md).

| Scenario                 | VROOM (ns/op)   | Notus 1T (ns/op) | Notus 64T (ns/op) | Winner (single-thread)        |
| ------------------------ | --------------- | ---------------- | ----------------- | ----------------------------- |
| Single exp 1024-bit      | **198,000**     | 541,000          | 167,000           | VROOM 2.7× faster than 1T     |
| Single exp 2048-bit      | **787,000**     | 4,598,000        | 663,000           | VROOM 5.8× faster than 1T     |
| **Single exp 3072-bit**  | **2,813,000**   | 15,747,000       | 1,468,000         | **VROOM 5.6× faster than 1T** |
| Batch 100 × 1024-bit     | **19,856,000**  | 55,057,000       | —                 | VROOM 2.8× faster             |
| Batch 100 × 2048-bit     | **78,446,000**  | 428,547,000      | —                 | VROOM 5.5× faster             |
| **Batch 100 × 3072-bit** | **283,411,000** | 1,600,781,000    | —                 | **VROOM 5.7× faster**         |
| Double 1024-bit          | **393,000**     | 1,904,000        | —                 | VROOM 4.8× faster             |
| Double 2048-bit          | **1,563,000**   | 15,068,000       | —                 | VROOM 9.6× faster             |
| **Double 3072-bit**      | **5,730,000**   | 56,234,000       | —                 | **VROOM 9.8× faster**         |
| Fourfold 1024-bit        | 791,000         | 1,081,000        | **416,000**       | Notus 64T wins                |
| Fourfold 2048-bit        | 3,126,000       | 8,241,000        | **2,476,000**     | Notus 64T marginally          |
| **Fourfold 3072-bit**    | 11,377,000      | 30,414,000       | **8,268,000**     | **Notus 64T 1.4× only**       |
| RSA verify 1024-bit      | **4,615**       | 12,579           | —                 | VROOM 2.7× faster             |
| RSA verify 2048-bit      | **9,119**       | 37,467           | —                 | VROOM 4.1× faster             |
| **RSA verify 3072-bit**  | **17,011**      | 80,541           | —                 | **VROOM 4.7× faster**         |

VROOM wins in every single-thread scenario across all bit sizes. The advantage is most pronounced at RSA-3072: **5.6× faster** for single exponentiation, **9.8× faster** for double, and **4.7× faster** for RSA verify — all with zero heap allocations.

The only scenario where Notus wins is fourfold with 64 threads, combining GCW and thread-level parallelism. However, the gap narrows with key size: from 1.9× at 1024-bit to **1.4× at 3072-bit**, where Notus needs 64 cores to achieve a diminishing advantage over VROOM's single thread.

---

## References

- [VROOM paper (MIT)](https://github.com/SimonLangowski/VROOM) — Langowski, He, Devadas
- [jiajunxin/multiexp](https://github.com/jiajunxin/multiexp)
- [Posch & Posch 1995](https://ieeexplore.ieee.org/document/381846) — original RNS Montgomery
