// rns_noalloc.go — Stage 2.5: zero allocations in the hot path
//
// Pre-allocates all scratch buffers once at setup time.
// VROOMStage2NoAlloc performs 0 heap allocations per multiplication.

package main

import (
	"math/big"
)

// Workspace holds pre-allocated buffers reused across multiplications.
type Workspace struct {
	tM int
	tN int
	// Buffers sized tM
	qM     []uint64
	crns2A []uint64 // CRNS2 Apply accumulator
	rM     []uint64
	// Buffers sized tN
	abN    []uint64
	crns1A []uint64 // CRNS1 Apply accumulator
	crns1R []uint64 // CRNS1 Apply result
	rN     []uint64
}

// NewWorkspace creates a workspace for the given base sizes.
func NewWorkspace(tM, tN int) *Workspace {
	return &Workspace{
		tM:     tM,
		tN:     tN,
		qM:     make([]uint64, tM),
		crns2A: make([]uint64, tM),
		rM:     make([]uint64, tM),
		abN:    make([]uint64, tN),
		crns1A: make([]uint64, tN),
		crns1R: make([]uint64, tN),
		rN:     make([]uint64, tN),
	}
}

// --------------------------------------------------------------------------
// Zero-allocation elementwise operations (write to dst)
// --------------------------------------------------------------------------

func rnsMulTo(dst []uint64, a, b []uint64, moduli []uint64) {
	for i, m := range moduli {
		dst[i] = mulmod(a[i], b[i], m)
	}
}

func rnsAddTo(dst []uint64, a, b []uint64, moduli []uint64) {
	for i, m := range moduli {
		dst[i] = addmod(a[i], b[i], m)
	}
}

// --------------------------------------------------------------------------
// Zero-allocation CRNS Apply (writes to provided buffers)
// --------------------------------------------------------------------------

// ApplyTo performs CRNS base change, writing to acc (scratch) and out (result).
// acc must be len(ToMod), out must be len(ToMod). Both are overwritten.
func (m *CRNSMatrixU64) ApplyTo(r []uint64, out []uint64, acc []uint64) {
	tFrom := len(r)
	tTo := len(m.ToMod)

	// Step 1: matrix-vector product
	for j := 0; j < tTo; j++ {
		nj := m.ToMod[j]
		var a uint64
		for i := 0; i < tFrom; i++ {
			prod := mulmod(r[i], m.A[i][j], nj)
			a = addmod(a, prod, nj)
		}
		acc[j] = a
	}

	// Step 2: compute k via 192-bit accumulator
	k := computeK192(r, m.F, m.FHi, m.Prec, tFrom)

	// Step 3: correction → out
	for j := 0; j < tTo; j++ {
		nj := m.ToMod[j]
		kc := mulmod(k, m.C[j], nj)
		out[j] = addmod(acc[j], kc, nj)
	}
}

// --------------------------------------------------------------------------
// MontParamsNoAlloc — Stage 2 params + workspace
// --------------------------------------------------------------------------

type MontParamsNoAlloc struct {
	P     *big.Int
	BaseM *RNSBaseU64
	BaseN *RNSBaseU64
	T     *big.Int
	TInv  *big.Int
	CRNS1 *CRNSMatrixU64
	CRNS2 *CRNSMatrixU64
	W     *Workspace
}

func SetupRNSParamsNoAlloc(p *big.Int) *MontParamsNoAlloc {
	// Reuse Stage 2 setup
	s2 := SetupRNSParamsStage2(p)
	tM := len(s2.BaseM.Moduli)
	tN := len(s2.BaseN.Moduli)
	return &MontParamsNoAlloc{
		P:     s2.P,
		BaseM: s2.BaseM,
		BaseN: s2.BaseN,
		T:     s2.T,
		TInv:  s2.TInv,
		CRNS1: s2.CRNS1,
		CRNS2: s2.CRNS2,
		W:     NewWorkspace(tM, tN),
	}
}

// --------------------------------------------------------------------------
// VROOMStage2NoAlloc — 0 allocations per call
// --------------------------------------------------------------------------

// VROOMNoAlloc performs VROOM multiplication with zero heap allocations.
// Output is written to params.W.rM and params.W.rN.
// IMPORTANT: not safe for concurrent use — workspace is shared.
func VROOMNoAlloc(aM, aN, bM, bN []uint64, p *MontParamsNoAlloc) ([]uint64, []uint64) {
	w := p.W

	// Step 2: q_M = a_M · b_M % M
	rnsMulTo(w.qM, aM, bM, p.BaseM.Moduli)

	// Step 3: r_N = a_N · b_N + CRNS1(q_M)
	rnsMulTo(w.abN, aN, bN, p.BaseN.Moduli)
	p.CRNS1.ApplyTo(w.qM, w.crns1R, w.crns1A)
	rnsAddTo(w.rN, w.abN, w.crns1R, p.BaseN.Moduli)

	// Step 4: r_M = CRNS2(r_N)
	p.CRNS2.ApplyTo(w.rN, w.rM, w.crns2A)

	return w.rM, w.rN
}

// Encoding/decoding — same as Stage 2
func ToVROOMEncodingNoAlloc(a *big.Int, p *MontParamsNoAlloc) ([]uint64, []uint64) {
	v := new(big.Int).Mul(a, p.BaseM.Product)
	v.Mod(v, p.P)
	MN := new(big.Int).Mul(p.BaseM.Product, p.BaseN.Product)
	aMN := new(big.Int).Mul(v, p.T)
	aMN.Mod(aMN, MN)
	return p.BaseM.Encode(aMN), p.BaseN.Encode(aMN)
}

func FromVROOMEncodingNoAlloc(rM []uint64, p *MontParamsNoAlloc) *big.Int {
	rPrime := p.BaseM.Decode(rM)
	MInvP := new(big.Int).ModInverse(p.BaseM.Product, p.P)
	result := new(big.Int).Mul(rPrime, MInvP)
	return result.Mod(result, p.P)
}
