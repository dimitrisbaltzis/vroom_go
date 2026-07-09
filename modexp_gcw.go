// modexp_gcw.go

package main

import "math/big"

// ModExpDoubleWindowed υπολογίζει base^exp1 και base^exp2 mod p
// χρησιμοποιώντας GCW για να μοιραστούν κοινά bits.
func ModExpDoubleWindowed(
    exp1, exp2 *big.Int,
    wt *VROOMWindowTable,
    w *ModExpWorkspace,
    params *MontParamsStage4,
) (*big.Int, *big.Int) {

    if exp1.Sign() == 0 {
        return big.NewInt(1), ModExpVROOMWindowed(exp2, wt, w, params)
    }
    if exp2.Sign() == 0 {
        return ModExpVROOMWindowed(exp1, wt, w, params), big.NewInt(1)
    }

    // Διαχωρισμός σε κοινά και μοναδικά bits
    common    := new(big.Int).And(exp1, exp2)
    exp1Extra := new(big.Int).AndNot(exp1, exp2)
    exp2Extra := new(big.Int).AndNot(exp2, exp1)

    // Workspace για τα τρία ξεχωριστά exponentiations
    w1 := NewModExpWorkspace(params)
    w2 := NewModExpWorkspace(params)
    wc := NewModExpWorkspace(params)

    // Υπολογισμός — 3 exponentiations αντί για 2
    // αλλά με λιγότερα bits συνολικά
    ModExpWindowed(common,    wt, wc, params)  // base^common
    ModExpWindowed(exp1Extra, wt, w1, params)  // base^exp1Extra
    ModExpWindowed(exp2Extra, wt, w2, params)  // base^exp2Extra

    // Συνδυασμός: base^exp1 = base^exp1Extra · base^common
    rM1, rN1 := VROOMStage4(w1.accM, w1.accN, wc.accM, wc.accN, params)
    copy(w1.accM, rM1)
    copy(w1.accN, rN1)

    rM2, rN2 := VROOMStage4(w2.accM, w2.accN, wc.accM, wc.accN, params)
    copy(w2.accM, rM2)
    copy(w2.accN, rN2)

    return FromVROOMEncodingStage4(w1.accM, params),
           FromVROOMEncodingStage4(w2.accM, params)
}

// ModExpFourfoldWindowed υπολογίζει base^exp[0..3] mod p με GCW.
func ModExpFourfoldWindowed(
    exps [4]*big.Int,
    wt *VROOMWindowTable,
    params *MontParamsStage4,
) [4]*big.Int {

    // Βρες bits κοινά σε όλους τους 4
    common4 := new(big.Int).And(exps[0], exps[1])
    common4.And(common4, exps[2])
    common4.And(common4, exps[3])

    // Αφαίρεσε τα κοινά bits από κάθε exponent
    extras := [4]*big.Int{}
    for i := range exps {
        extras[i] = new(big.Int).AndNot(exps[i], common4)
    }

    // Υπολογισμός
    wc := NewModExpWorkspace(params)
    ModExpWindowed(common4, wt, wc, params)

    ws := [4]*ModExpWorkspace{}
    for i := range ws {
        ws[i] = NewModExpWorkspace(params)
        ModExpWindowed(extras[i], wt, ws[i], params)
    }

    // Συνδυασμός
    var results [4]*big.Int
    for i := range results {
        rM, rN := VROOMStage4(ws[i].accM, ws[i].accN, wc.accM, wc.accN, params)
        copy(ws[i].accM, rM)
        copy(ws[i].accN, rN)
        results[i] = FromVROOMEncodingStage4(ws[i].accM, params)
    }

    return results
}