# DESIGN AUDIT — EXP-CORRESPONDENCE-ARCH-RATE-001

**Freeze audit. Nothing executed.** No training, no inference, no model
instantiated, no `RESULTS.md`. GEOM-001, GEOM-002 and every prior record
untouched.

Tags: **MEASURED** (source/frozen record), **DERIVED** (algebra), **PREDICTED**
(an expectation resting on an approximate argument), **UNKNOWN**.

---

## 1. THE BOUNDARY-EQUIVARIANCE QUALIFICATION — the required proof

The operator wrote: *"random weights ramp horizontally and not vertically"* must
be recorded as an **architectural prediction, not an unconditional theorem**,
because the disparity-axis convolutions use a finite `D = 12` with padding.

**That caution is correct, and the audit finds it is stronger than stated.**

### 1.1 What is exactly true (DERIVED)

**MEASURED** (`src/models/stereonet/aggregation.py`): `4 × [Conv3d(32→32, 3×3×3,
padding=1) + LeakyReLU]` then `Conv3d(32→1, 3×3×3, padding=1)` — five 3-tap
convolutions along `D`, receptive field `1 + 2·5 = 11` over `D = 12`.

Output candidate `k` depends on inputs `[k−5, k+5]`. Padding-free in the
`V`-frame requires `k ∈ {5, 6}` — **2 of 12 candidates.**

`Agg(V)(k) = Agg(F)(k−τ)` requires the receptive field to be padding-free in
**both** frames: `k ∈ {5,6}` **and** `k−τ ∈ {5,6}`.

| τ | t px | exactly-equivariant candidates | count |
|---:|---:|---|---:|
| 0 | 0 | `{5, 6}` | 2 |
| 1 | 16 | `{6}` | **1** |
| 2 | 32 | — | **0** |
| 3 | 48 | — | **0** |
| 4 | 64 | — | **0** |
| 5 | 80 | — | **0** |
| 6 | 96 | — | **0** |

```
DERIVED: exact translation equivariance holds at 2 candidates for τ = 0,
         1 candidate for τ = 1, and NONE for τ ≥ 2.
```

**The "profile translates with τ" statement is an interior, approximate result.
On this finite axis it is not an unconditional theorem at any level of the
intended sweep beyond τ = 1.**

### 1.2 A second, independent boundary effect (DERIVED)

The soft-argmin sums over a **linear index grid `0…11`**. Under the prediction
`m(τ) ≈ τ + c` with `τ ∈ {0,…,6}`, the response spans 7 of the 12 candidates. The
intercept `c` is **UNKNOWN a priori**, so whether the sweep reaches a rail is
**UNKNOWN**:

| intercept `c` | predicted span | rails? |
|---:|---|---|
| 0 | 0 … 6 | inside |
| 3 | 3 … 9 | inside |
| 5 | 5 … 11 | **touches the high rail** |

**⇒ clipping must be detected and classified separately, never assumed absent.**
This is the GEOM-001 lesson applied prospectively.

### 1.3 The vertical arm — what is and is not derived

**DERIVED:** the candidate axis indexes *horizontal* displacement only. There is
no `κ` making `Lf[y, w+k]` coincide with `Lf[y−τ, w]` for generic content, so
**no candidate-axis translation is forced** on the vertical arm.

**NOT DERIVED:** that the vertical response is therefore absent, flat or null.
Vertically self-similar content — road surface, lane markings, poles — can
produce partial matches. **The vertical arm is unconstrained, not
predicted-null.** No predicted value is assigned to it anywhere in this design.

### 1.4 The three-tier record, as required

```
DERIVED    : the interior candidate-axis structure translates with τ;
             V(k,w) = F(k−τ, w+τ) exactly, with F independent of τ.
             Exact equivariance survives at 2 / 1 / 0 candidates for τ = 0 / 1 / ≥2.

PREDICTED  : this should produce a HIGH rate of horizontal ramp-like responses
             for random aggregation weights, and a lower rate of vertical ones,
             hence a high QUALIFYING rate.
             This rests on APPROXIMATE equivariance, not on §1.1's exact count.

UNKNOWN    : how strongly the D = 12 padding boundary alters the response in
             practice; the value of the intercept c, hence whether the sweep
             rails; the strength of vertical self-similarity in these scenes.
```

**This distinction is carried verbatim into `PREREGISTRATION.md` §2.** Without
it, an approximate architectural argument would again be dressed as an exact
prediction — the error that produced the GEOM-002 `t = 0 → 5.5` hard stop.

### 1.5 Why the qualification does not weaken the experiment

The experiment does **not** test whether equivariance is exact. It **measures how
often the signature appears** with untrained weights. The derivation supplies a
falsifiable expectation (a high rate); the measurement supplies the answer. A low
rate would simply mean the boundary effects dominate on this axis — itself a
finding, and one that would make the observable diagnostic.

**The weaker the theorem, the more the measurement is worth.**

---

## 2. AUDIT AGAINST THE FOURTEEN FROZEN PRINCIPLES

| # | principle | satisfied by |
|---:|---|---|
| 1 | random aggregation weights only | `randomisation_spec.json`: aggregation, 111 585 params |
| 2 | feature extractor frozen | explicitly listed under `frozen_NOT_randomised`; required, since randomising it destroys the `V = 0 at κ = 0` structure |
| 3 | same synthetic construction | `construction_spec.json` inherits GEOM-002's construction unchanged, including the corrected sign `a_R = a_L + t` |
| 4 | horizontal and vertical conditions | both axes, identical code path |
| 5 | same six disparity levels | `t ∈ {16,…,96}` plus `t = 0` |
| 6 | no trained-arm inspection or execution | `trained_aggregation_measured: false`; no trained aggregation is ever run |
| 7 | score each seed independently | unit = `(feature_extractor, scene, seed)`; 384 units |
| 8 | rate `k/N`, not a binary verdict | `primary_result` is the rate plus the full tally |
| 9 | five separate classifications | `QUALIFYING / FLAT / ANTI_CORRELATED / NON_MONOTONE / NON_SPECIFIC`, plus `CLIPPED` |
| 10 | clipping not folded into ordinary failure | `clipping_must_not_be_folded: true`; `CLIPPED` is evaluated **before** every other class |
| 11 | hard stops mechanically verifiable | §4 — all are hash, shape, count or finiteness checks |
| 12 | `N` fixed before execution | `N = 32`, resolution `1/32 = 0.03125` |
| 13 | no thresholds from any previous response | §3 |
| 14 | no trained model measured | the trained **feature extractor** is loaded because principle 2 requires it; no trained **aggregation** is ever executed |

---

## 3. THRESHOLD PROVENANCE — the one place magnitudes enter

`FLAT` and `CLIPPED` are inherently magnitude notions and cannot be defined
without a scale. Both cuts are **derived from the architecture's own
quantisation**, one candidate:

```
CLIPPED : min_t m_h(t) ≤ 1.0   OR   max_t m_h(t) ≥ 10.0     (one candidate from each rail)
FLAT    : max_t m_h(t) − min_t m_h(t) < 1.0                  (less than one candidate of range)
```

**Neither derives from any observed response, from GEOM-002, or from any previous
experiment.** A sweep spanning six candidates cannot be "tracking" if its whole
response is under one candidate wide; a response sitting within one candidate of
a rail is compressed by the grid's own boundary.

**Safeguard against this being load-bearing:** the specification requires that
**every `m_h(t)` and `m_v(t)` value for every one of the 384 units be published**,
so any reader can reclassify under an alternative definition without re-running
anything. The rate is reported alongside the raw data, not in place of it.

---

## 4. OUTCOME SPACE — every cell terminates informatively

| outcome | reading | terminal |
|---|---|---|
| rate high | the architecture supplies the signature without learning ⇒ **the synthetic self-pair observable is non-diagnostic; the construction is closed** | **yes** |
| rate intermediate | the rate *is* the quantitative answer: this much of the signature is architectural | **yes** |
| rate low / zero | untrained weights rarely reproduce it ⇒ the derivation's boundary caveats dominate, **and the observable may be diagnostic** ⇒ a trained-arm experiment becomes warrantable under a **separate** preregistration | **yes** |
| many `CLIPPED` | the intercept `c` places the sweep on a rail (§1.2); reported as its own count, never merged into failure | **yes** |
| many `FLAT` | untrained weights produce no candidate-axis response at all | **yes** |
| many `ANTI_CORRELATED` | the profile translates but the readout follows it with the opposite sign — a distinct mechanism finding | **yes** |
| many `NON_SPECIFIC` | horizontal ramps but the vertical arm ramps as strongly ⇒ axis specificity is *not* architecturally privileged | **yes** |
| `NONFINITE` | **not an outcome — an implementation fault** ⇒ HARD STOP | terminates as a fault |

**DERIVED: the classification is a total, mutually exclusive, ordered partition of
the unit space. There is no unit that cannot be classified and no rate that
cannot be read.**

---

## 5. WHAT THIS DESIGN CANNOT DO

- It measures **nothing about the trained models.** No trained aggregation is
  executed. It cannot support any claim about correspondence, learned or
  otherwise.
- It characterises **framework-default initialisation at this scale only** — not
  all untrained networks.
- A low rate does **not** establish that trained models do exhibit
  correspondence; it establishes only that the observable is not obviously
  architecturally forced.
- It says nothing about real-scene correspondence, disparity correctness, genuine
  disparity search, or generalisation.

---

## 6. VERDICT

```
ARCH-RATE-001-DESIGN-READY
```

The boundary-equivariance qualification is proven (§1), recorded in three tiers
(§1.4) and carried into the preregistration. All fourteen frozen principles are
satisfied (§2). The only magnitude thresholds are grid-derived and accompanied by
a raw-data publication requirement (§3). The outcome space is a total partition in
which every cell terminates informatively (§4).

**Not executed. Not authorised. Awaiting the operator's go.**
