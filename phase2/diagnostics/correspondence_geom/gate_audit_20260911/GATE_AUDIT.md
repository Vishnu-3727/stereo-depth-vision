# GATE SPECIFICATION AUDIT — can the random-first gate be made rigorous?

Record `phase2/diagnostics/correspondence_geom/gate_audit_20260911/`. 2026-09-11.

**Question:** can the random-first gate be specified so that **every possible
outcome terminates informatively**?

**Audit only. Nothing executed, trained, designed or frozen.** No `N`, `t` levels,
scenes, masks or thresholds are selected. No inference. All historical records
untouched.

**Anti-post-hoc.** Every result below is algebra or logic on the frozen
construction and the source. No GEOM-002 response value is used.

Tags: **MEASURED**, **DERIVED**, **INFERRED**, **UNKNOWN**.

---

## VERDICT

```
GATE-CAN-BE-MADE-RIGOROUS
```

— but **not in the two-stage form proposed in the pre-design audit.** Auditing the
outcome space showed the two-stage gate has a structural defect (§2), and that the
rigorous form is a **standalone random-weight-only measurement whose result is a
rate, not a pass/fail** (§3). In that form every outcome, including every
degenerate one, terminates informatively (§4).

---

## 1. A SECOND ARCHITECTURAL FORCING RESULT, DERIVED HERE

The pre-design audit derived that the **slope** is architecturally forced. The
same substitution settles the **axis-specificity** condition too.

**Horizontal.** `V[k][w] = Lf[w+k] − Lf[w+τ]`. Putting `k = κ + τ` gives
`V = F(k−τ, w+τ)` — a translation **along the candidate axis** exists, so a
shift-equivariant `Agg` translates with it and the readout slope is 1 for any
weights.

**Vertical.** `V[k][y,w] = Lf[y,w+k] − Lf[y−τ, w]`. The candidate axis indexes
**horizontal** displacement only; there is no `κ` that makes the two terms
coincide. **No candidate-axis translation exists, so no slope is forced.**

```
DERIVED:  under RANDOM weights the horizontal arm ramps and the vertical does not.
          ⇒ |α_h| > |α_v| is ALSO architecturally forced.
```

**Consequence.** Both discriminators that survived the GEOM-002 postmortem — the
repaired free-intercept slope **and** axis specificity — are predicted to be
satisfied by untrained weights. **The entire qualitative signature of the
synthetic self-pair may be architectural.** That is the proposition the gate must
measure.

---

## 2. WHY THE TWO-STAGE GATE FAILS AUDIT

The proposed structure was: Stage 1 random arm → if it reproduces the signature,
stop; else Stage 2 trained arm.

### 2.1 Defect A — the gate criterion needs a threshold, or is vacuous

A natural gate is *"is the a-priori prediction `S*` inside the empirical random
range `[S_min, S_max]`?"* — threshold-free, and it works for a **two-sided**
statistic with an interior anchor (the slope, anchor `α* = 1`).

It **degenerates for one-sided statistics with a boundary anchor.** For the affine
residual `1 − R²`, the anchor is `0` and the domain is `[0, ∞)`. Random seeds
essentially never attain exactly `0`, so `S*` is always outside the random range
and the gate **always passes** — with no power to detect architectural forcing.

**DERIVED: the "anchor inside the random range" gate is statistic-dependent and
silently vacuous for an entire class of shape statistics.** A gate whose validity
depends on which statistic it is applied to is not a rigorous gate.

### 2.2 Defect B — with a separation rule, the gate is unnecessary

If Stage 2's rule is *"trained separates from all N random seeds"*, then **both**
Stage-2 outcomes are already informative:

- separates ⇒ weight-dependent;
- does not separate ⇒ not distinguishable from untrained architecture.

**The gate protects against nothing.** Its only remaining function would be to
save the trained arm's "one shot" — which brings us to the decisive point.

### 2.3 Defect C — the trained arm has almost no predictive content left

**MEASURED:** GEOM-002's `RESULTS.md` publishes `m_h(t)` and `m_v(t)` for all
three trained checkpoints at all seven levels. The trained horizontal response is
**public**. On fresh scenes it would be new numbers, but the *shape* is known.

**The random-weight response has never been measured, on any checkpoint, in any
experiment.**

```
DERIVED: essentially all the remaining predictive content of the synthetic
         self-pair construction lies in the RANDOM arm, not the trained arm.
```

**INFERRED:** a two-stage design spends its rigour protecting the arm that is
already spent, and treats as a mere "gate" the only measurement that is genuinely
unknown. **The staging is inverted.**

---

## 3. THE RIGOROUS FORM — the gate as a standalone measurement

Drop the second stage. The scientifically valuable experiment is the random arm
**alone**, and its purpose is to test a sharp, a-priori, algebraically derived
prediction:

> §1 and the pre-design audit §1.6 predict that **untrained weights of this
> architecture reproduce the full qualitative correspondence signature** — ramp,
> unit slope, monotonicity, and axis specificity.

This is a falsifiable claim about the architecture, derived before any
measurement, and it decides whether the synthetic self-pair observable can ever
be diagnostic.

### 3.1 Why this form is rigorous

| property | how it is satisfied |
|---|---|
| **no threshold anywhere** | each seed is classified by the **same threshold-free qualitative conditions** that would qualify a trained model — sign, strict monotonicity of first differences, axis specificity. The result is the **rate** `k/N`, which is a measurement, not a test |
| **no trained arm** | the public GEOM-002 curves cannot contaminate it — they are never used, compared against, or needed |
| **a-priori prediction** | §1 predicts a high rate. The prediction was derived from shift-equivariance before any measurement |
| **resolution fixed in advance** | the rate resolves to `1/N`; `N` is chosen from compute alone, exactly as INDEX-002 fixed its permutation resolution |
| **falsifies in both directions** | a high rate refutes the diagnosticity of the observable; a low rate refutes §1's derivation |
| **cannot be rescued** | there is no pass/fail to argue about. The rate is the result |

### 3.2 What it is not

It is **not** a test of the trained model, and it makes **no claim about
correspondence**. It is a measurement of how much of the signature the
architecture supplies without learning.

---

## 4. OUTCOME SPACE — does every outcome terminate informatively?

Each seed is classified **SATISFIES / DOES-NOT-SATISFY / DEGENERATE(subtype)**.
The result is the rate plus the degeneracy tally.

### 4.1 The rate

| `k/N` | reading | terminal? |
|---|---|---|
| `= N/N` | every untrained seed reproduces the signature ⇒ **the observable is architecturally forced**; the synthetic self-pair cannot test learning. **Closes the construction.** | **yes** |
| high but `< N/N` | the architecture supplies most of the signature; any trained result would be weakly interpretable at best; the rate quantifies how weakly | **yes** |
| intermediate | partial forcing; the rate *is* the quantitative answer | **yes** |
| low | the architecture rarely supplies it ⇒ **the observable may be diagnostic**; a separately preregistered trained-arm experiment becomes warranted | **yes** |
| `= 0` | untrained weights never reproduce it ⇒ §1's derivation is contradicted (or its padding caveats dominate); the observable is diagnostic | **yes** |

**Every value of `k` is a measurement. No value leaves the experiment
uninterpretable.** This is the property the audit was asked to establish, and it
holds because the result is a rate rather than a verdict.

### 4.2 Degenerate and pathological cases

These are where an outcome space usually fails to terminate. Each is enumerable
and each is informative, **provided classification rules are preregistered**:

| case | derived reading | terminal? |
|---|---|---|
| response constant in `τ` (no response at all) | monotonicity fails ⇒ DOES-NOT-SATISFY. The architecture produces no candidate-axis response from untrained weights | **yes** |
| response monotone but **decreasing** | sign fails ⇒ DOES-NOT-SATISFY, but must be **tallied separately**: an anti-correlated response is a different finding from no response | **yes**, if the subtype is recorded |
| response **saturates at a rail** (`k → 0` or `11`) | clipping destroys the translation the derivation depends on; monotonicity fails for a reason unrelated to the mechanism. Must be **flagged as CLIPPED and tallied separately**, never silently merged into DOES-NOT-SATISFY | **yes**, if flagged |
| ties in the first differences | a tie fails *strict* monotonicity; the rule must say so explicitly in advance | **yes** |
| `NaN` / non-finite | **not an outcome — an implementation fault.** HARD STOP | terminates as a fault |
| bimodal / multimodal rate across seeds | the rate still measures the fraction; the modality is reported descriptively | **yes** |

**DERIVED: the outcome space is closed and every cell is informative, conditional
on the four flagged subtypes (constant / anti-correlated / clipped / tied) being
preregistered as distinct classifications rather than collapsed into a single
failure bucket.** That conditional is the one real requirement this audit imposes.

### 4.3 The one genuine non-termination risk, and its remedy

If **clipped** seeds were merged into DOES-NOT-SATISFY, a low rate would be
ambiguous between *"the architecture lacks the mechanism"* and *"the readout
railed"*. That ambiguity would not terminate informatively.

**Remedy, derivable in advance:** clipping is detectable from the response itself
— saturation at the candidate-axis rails — without reference to any threshold on
the statistic. Preregister it as a separate classification. **This is the
GEOM-001 lesson (a range defect masquerading as a null result) applied
prospectively.**

---

## 5. REQUIREMENTS A FREEZE MUST SATISFY

Stated as requirements, not as a design. **Nothing here selects `N`, levels,
scenes, masks or thresholds.**

1. **Random arm only.** No trained checkpoint is loaded, compared against, or
   needed. This is what makes the public GEOM-002 curves irrelevant.
2. **Randomise aggregation weights only.** Feature extractor frozen — randomising
   it destroys the `V = 0 at κ = 0` structure the whole derivation rests on.
   Readout already parameter-free (**MEASURED**).
3. **Framework-default initialisation, seeds frozen before execution**, and the
   scope limitation declared: the result characterises *this* initialisation
   scheme, not all untrained networks.
4. **The result is a rate, not a verdict.** No pass/fail, no threshold, no
   aggregate decision rule. `k/N` plus the degeneracy tally is the finding.
5. **Per-seed classification rules preregistered**, with **four distinct
   subtypes** — constant, anti-correlated, clipped, tied — never collapsed
   (§4.2, §4.3).
6. **The conditions applied to the random arm must be exactly the threshold-free
   conditions that would qualify a trained model.** If they are not identical, the
   measurement does not bear on diagnosticity.
7. **Both arms measured** — horizontal and vertical — since §1 predicts axis
   specificity is forced too, and that prediction is independently falsifiable.
8. **`N` fixed from compute alone**, with the resolution `1/N` stated in advance
   (INDEX-002's finite-resolution discipline).
9. **Hard stops encode derivable properties, not predicted values** — the
   GEOM-002 `t = 0 → 5.5` lesson — and **enforcement asserted in code**, which
   GEOM-002's was not.
10. **The a-priori prediction recorded before execution**: §1 predicts a high
    rate. Recording it makes the measurement a genuine test of the derivation
    rather than an open-ended survey.

---

## 6. WHAT EACH TERMINAL OUTCOME WOULD LICENCE

| outcome | maximum claim | not claimable |
|---|---|---|
| high rate | "The candidate-axis ramp and its axis specificity under synthetic horizontal displacement are produced by this architecture **without training**." → the synthetic self-pair observable is **non-diagnostic**; the construction is closed | anything about correspondence, learned or otherwise; anything about the trained models |
| low rate | "Untrained weights of this architecture do **not** reproduce the signature." ⇒ the observable **may** be diagnostic, and a trained-arm experiment becomes warrantable under a **separate** preregistration | that the trained models *do* exhibit correspondence — the trained arm was never measured |
| intermediate | the rate itself, reported | any binary reading |
| any outcome | — | real-scene correspondence; genuine disparity search; disparity correctness; generalisation |

---

## 7. CONCLUSION

```
GATE-CAN-BE-MADE-RIGOROUS
```

**Yes — every possible outcome terminates informatively**, provided the gate is
specified as a **standalone random-weight-only measurement reporting a rate**,
with the four degeneracy subtypes preregistered as distinct classifications.

Three things the audit changed:

1. **The two-stage form does not survive.** Its criterion is vacuous for an entire
   class of statistics (§2.1), it is redundant under a separation rule (§2.2), and
   it stages the experiment backwards — protecting the already-public trained arm
   while treating the only unmeasured quantity as a preliminary (§2.3).
2. **Axis specificity is architecturally forced too** (§1), so the whole
   qualitative signature is in question, not just the slope.
3. **The single genuine non-termination risk is clipping being merged into
   failure** (§4.3) — removable in advance by preregistering it as a separate
   classification.

**This experiment would be the first in the campaign whose result is a measurement
rather than a verdict, and therefore the first that cannot be argued with after
the fact.** Its expected outcome — a high rate, closing the construction — is the
honest one, and closing a line with a measurement is a better result than six
inconclusive attempts to open one.

**Nothing is designed, selected, preregistered or launched. Freeze only after
these requirements are met.**
