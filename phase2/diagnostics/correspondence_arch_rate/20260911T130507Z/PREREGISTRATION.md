# EXP-CORRESPONDENCE-ARCH-RATE-001 — PRE-REGISTRATION

**How often does the frozen architecture produce the horizontal correspondence
signature with aggregation weights that contain no learned information?**

Record `phase2/diagnostics/correspondence_arch_rate/20260911T130507Z/`.
Frozen 2026-09-11. **NOT EXECUTED.**

This is a **descriptive measurement**, not a hypothesis test. Its result is a
**rate**, not a verdict. There is no pass/fail criterion anywhere in this
document.

Inference only. No training, no fine-tuning, no weight optimisation, no
architecture change, no new checkpoint. **No trained aggregation is ever
executed.** Phase 1 untouched. GEOM-001 and GEOM-002 untouched; their verdicts
unchanged. No historical record modified.

**Frozen artefacts, written before this document, hashed over FILE BYTES:**

```
construction_spec.json    sha256 7a4e2341fc50e3695648cc286cdb4219390a0378a6669994ec0e4555b75cf9b0
randomisation_spec.json   sha256 e52b5b03c5d4406e5ed9212f6681ee53bb448927eebddd367a7698ce9ce203ad
classification_spec.json  sha256 273f7562b62dd4c29a2d617e3b2b4b24d98532fcc7d99531b82b55b377d0cd28
```

All written with `newline=""` so `sha256sum -c frozen.sha256` reproduces them
externally.

---

## 0. PROVENANCE DISCLOSURE — recorded before execution

1. **The designer has read the response curves of six prior experiments**,
   including GEOM-002's. **None is used here**: no trained model is measured, no
   threshold derives from any observed response, and every quantity in this
   document comes from source, integer geometry, or the frozen specs.
2. **No trained aggregation is executed.** The trained **feature extractor** is
   loaded because §4 requires it frozen — it supplies the cost volume. That is
   the only use of a trained checkpoint.
3. **Scenes are held out from both prior correspondence experiments**
   (§6), chosen by a GT-free index rule.
4. **The result is a rate.** Its interpretation does not depend on choosing a
   post-hoc success threshold. It is a preregistered descriptive measurement.
5. **Scope:** the result characterises **framework-default initialisation at this
   scale**, not all untrained networks.

---

## 1. QUESTION

> Of `N` independently seeded, untrained aggregations placed in the frozen
> pipeline, **what fraction reproduces the qualitative horizontal correspondence
> signature** — correct sign, strict monotonicity across the sweep, and
> horizontal-over-vertical specificity — on synthetic stereo pairs whose
> disparity is known exactly by construction?

The signature is defined by **exactly the threshold-free conditions that would be
used to qualify a trained model** (§7). If they were not identical, the
measurement would not bear on whether those conditions test learning.

---

## 2. THE ARCHITECTURAL ARGUMENT — three tiers, kept separate

This is the qualification the freeze audit was required to establish. It is
recorded here verbatim so that an **approximate** architectural argument is never
again treated as an **exact** prediction.

### DERIVED (algebra on source; unconditional)

With stride-16 equivariance and `τ = t/16`, `Rf[w] = Lf[w+τ]`, so

```
V[k][w] = Lf[w+k] − Lf[w+τ] = F[k−τ][w+τ],     F[κ][w] := Lf[w+κ] − Lf[w]
```

`F` is **independent of `τ`**. The cost volume is `F` translated by `τ` along the
candidate axis. `V` is exactly zero at `k = τ`.

**MEASURED** (`aggregation.py`): five 3-tap padded convolutions along `D`;
receptive field 11 over `D = 12`; padding-free candidates `{5, 6}`.

`Agg(V)(k) = Agg(F)(k−τ)` requires padding-freedom in **both** frames:

| τ | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| exactly-equivariant candidates | `{5,6}` | `{6}` | — | — | — | — | — |
| count | 2 | **1** | **0** | **0** | **0** | **0** | **0** |

**Exact equivariance holds at 2 candidates for `τ = 0`, 1 for `τ = 1`, and NONE
for `τ ≥ 2`. The translation result is an interior, approximate statement — not
an unconditional theorem on this finite axis.**

### PREDICTED (rests on the approximate argument, not on the exact count)

Approximate interior equivariance **should** produce a **high rate** of
horizontal ramp-like responses from untrained aggregations, and a lower rate of
vertical ones — hence a **high QUALIFYING rate**.

**This is the falsifiable expectation this experiment tests.** It is recorded
before execution so the measurement is a genuine test of the derivation rather
than an open-ended survey.

### UNKNOWN

- How strongly the `D = 12` padding boundary alters the response in practice.
- The intercept `c` in `m(τ) ≈ τ + c`, hence **whether the sweep reaches a rail**
  (the response spans 7 of 12 candidates; at `c = 5` it touches the high rail).
- The strength of vertical self-similarity in these scenes. **DERIVED:** no
  candidate-axis translation is *forced* on the vertical arm — but it is **not**
  derived that the vertical response is absent or flat. Road surfaces, lane
  markings and poles are vertically self-similar. **The vertical arm is
  unconstrained, not predicted-null, and no predicted value is assigned to it
  anywhere.**

---

## 3. CONSTRUCTION — frozen in `construction_spec.json`

Inherited **unchanged** from GEOM-002, including the corrected sign.

```
left  crop : rows [0, 272), cols [0, 1136)                FIXED every condition
right crop : horizontal  rows [0, 272),  cols [t, 1136+t)
             vertical    rows [t, 272+t), cols [0, 1136)
crop 1136 × 272 = (71×16) × (17×16)                        identical every condition
t ∈ {0, 16, 32, 48, 64, 80, 96} px                         13 distinct conditions
```

Both images are crops of the **same source image** (`scene.left`); the scene's own
right image is never used. Integer slicing only — **no fill, no padding, no
interpolation**. `d = a_R − a_L = t` exactly and uniformly. Every `t` is a
multiple of the stride; every window lies inside the source.

---

## 4. RANDOMISATION — frozen in `randomisation_spec.json`

```
randomised  : aggregation ONLY
              4 × [Conv3d(32→32, 3×3×3, padding=1) + LeakyReLU(0.01)]
              then Conv3d(32→1, 3×3×3, padding=1)     111 585 parameters
init        : framework default — kaiming_uniform_(weight, a=√5);
              bias ~ U(−1/√fan_in, +1/√fan_in)
seeding     : torch.manual_seed(seed) immediately before construction
seeds       : 0 … 31        N = 32        rate resolution 1/32 = 0.03125
```

**NOT randomised:** the feature extractor (**trained, frozen** — randomising it
would destroy the `V = 0 at κ = 0` structure the whole construction rests on),
the cost volume (no parameters), the readout (**zero parameters**, verified from
source), refinement (never invoked).

**Feature extractors used:** all three of `POS_6b_seed0/1/2`, so the rate is not
tied to one. **Their trained aggregations are never run.**

---

## 5. MASK — frozen in `classification_spec.json`

```
retain (y, x) in CROP coordinates iff   64 ≤ y < 208   AND   64 ≤ x < 1072
145 152 px per scene
```

Purely geometric. **No ground truth is consulted** — the disparity is `t` by
construction. Identical for every `t`, axis, seed, feature extractor and scene.

---

## 6. UNITS AND SCENES

```
unit = (feature extractor, scene, random seed)
     = 3 × 4 × 32 = 384 units
```

**Scenes**, by a **GT-free rule fixed in advance** — the first four `hailo_val`
indices excluding **both** prior correspondence sets `{27,0,31,6}` (GEOM-001) and
`{1,2,3,4}` (GEOM-002):

```
5 000165_10.png   7 000167_10.png   8 000168_10.png   9 000169_10.png
```

**MEASURED:** 40 scenes exist; these four have never appeared in any experiment.
No ground truth was consulted to select them.

Units are **not independent replicates** — they share three feature extractors
and four scenes. This is a descriptive measurement, not a population inference.
**No p-value is computed anywhere.**

---

## 7. STATISTIC AND CONDITIONS — frozen

```
m_axis(t) = np.median( disparity_initial[mask] )                candidate units
α_axis    = free-intercept OLS slope of m_axis(t) on t/16,  t ∈ {16,…,96}
```

**Free intercept, NOT through the origin.** The through-origin form is the proven
GEOM-002 defect: it maps a constant step of height `h` to `0.2308h`, making a step
indistinguishable from a ramp. A free intercept maps every step to `α = 0`
exactly.

Per unit, the three threshold-free conditions:

```
Q1  sign          :  α_h > 0
Q2  monotone      :  m_h(t) strictly increasing over all 7 levels
Q3  axis-specific :  |α_h| > |α_v|
```

**These are exactly the conditions that would be used to qualify a trained
model.** No predicted value is assigned to `α_v`.

---

## 8. CLASSIFICATION — total, ordered, mutually exclusive

Evaluated **in this order**; the first match wins.

| order | class | definition |
|---:|---|---|
| 1 | `NONFINITE` | any `m` value not finite ⇒ **implementation fault ⇒ HARD STOP** (not an outcome) |
| 2 | **`CLIPPED`** | `min_t m_h(t) ≤ 1.0` **or** `max_t m_h(t) ≥ 10.0` |
| 3 | `FLAT` | `max_t m_h(t) − min_t m_h(t) < 1.0` |
| 4 | `ANTI_CORRELATED` | `α_h < 0` |
| 5 | `NON_MONOTONE` | Q2 fails and the unit is not `FLAT` |
| 6 | `NON_SPECIFIC` | Q1 ∧ Q2 hold, Q3 fails |
| 7 | `QUALIFYING` | Q1 ∧ Q2 ∧ Q3 |

**`CLIPPED` is evaluated second — before every ordinary failure class — so a
railed response can never be silently folded into failure.** That conflation is
the GEOM-001 failure mode, prevented here structurally.

**Threshold provenance.** The two magnitude cuts are derived from the
architecture's **candidate grid spacing of one candidate** — one candidate from
each rail for `CLIPPED`, less than one candidate of total range for `FLAT`.
**Neither derives from any observed response, from GEOM-002, or from any prior
experiment.**

**Safeguard:** every `m_h(t)` and `m_v(t)` for all 384 units **must be
published**, so any reader can reclassify under an alternative definition without
re-running anything.

---

## 9. RESULT

```
PRIMARY RESULT = the rate  k / 384  of QUALIFYING units,
                 reported with the full seven-way classification tally,
                 broken down by feature extractor and by scene.
```

**There is no pass/fail rule. There is no threshold on the rate. The rate is the
finding.**

Reported alongside, descriptively: the distributions of `α_h` and `α_v`; the raw
curves; the `CLIPPED` count; the per-extractor and per-scene rates.

---

## 10. INTERPRETATION — fixed before execution

| outcome | maximum claim |
|---|---|
| **high rate** | "The horizontal ramp and its axis specificity under synthetic displacement are produced by this architecture **without training**." ⇒ **the synthetic self-pair observable is non-diagnostic and the construction is closed.** |
| **intermediate rate** | the rate itself: this fraction of the signature is architectural. No binary reading. |
| **low / zero rate** | "Untrained weights of this architecture rarely reproduce the signature." ⇒ the boundary caveats of §2 dominate, **and the observable may be diagnostic** ⇒ a trained-arm experiment becomes warrantable under a **separate** preregistration. |

**Not claimable under any outcome:** anything about the trained models;
correspondence, learned or architectural; genuine disparity search; disparity
correctness; real-scene correspondence; generalisation.

A low rate does **not** establish that trained models exhibit correspondence —
**the trained arm is never measured.**

---

## 11. HARD STOPS — mechanically verifiable only

Each is a hash, shape, count or finiteness check. **None depends on a predicted
numeric value** — the GEOM-002 `t = 0 → 5.5` lesson.

| check | criterion |
|---|---|
| spec digests | all three re-verified **over file bytes** before the first model instantiation, and present verbatim in this document |
| no `RESULTS.md` | must not exist at start |
| checkpoint hashes | the three `POS` sha256 verified at load |
| readout parameter-free | `sum(p.numel() for p in model.regression.parameters()) == 0` |
| aggregation param count | `== 111 585` for every seed |
| seed distinctness | every randomised tensor's sha256 distinct across all 32 seeds |
| crop invariance | every condition yields exactly `1136 × 272` |
| no fill / interpolation | every window asserted inside the source; all `t` multiples of 16 |
| left crop fixed | byte-identical across all 13 conditions within a scene |
| mask invariance | `145 152` px in every condition |
| scenes held out | selected index set disjoint from `{27,0,31,6} ∪ {1,2,3,4}` |
| finiteness | any non-finite `m` ⇒ HARD STOP |
| no trained aggregation executed | asserted: the aggregation module in every forward pass is a freshly seeded random instance |
| determinism | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `no_grad`, `eval`, fp32 |
| forward-pass count | exactly `384 × 13 = 4 992` aggregation/readout arms over `156` cached cost volumes |

**Every hard stop must be enforced in code and halt execution** — GEOM-002's was
recorded but not enforced.

---

## 12. COMPUTE

The cost volume depends on `(feature extractor, scene, axis, t)` and **not** on
the aggregation seed, so it is built once and reused across all 32 seeds:

```
156 cost-volume builds  +  4 992 aggregation/readout arms
```

Estimated well under 5 minutes. No training, no new checkpoint, no architecture
change.

---

## 13. PREREGISTRATION ORDER

1. construction frozen → `construction_spec.json` ✔
2. randomisation frozen → `randomisation_spec.json` ✔
3. classification + mask + scenes frozen → `classification_spec.json` ✔
4. boundary qualification derived → §2, `DESIGN_AUDIT.md` §1 ✔
5. statistic frozen → §7 ✔
6. classification order frozen → §8 ✔
7. `PREREGISTRATION.md` written → this document ✔
8. **execute** → NOT DONE
9. `RESULTS.md` → NOT CREATED

**No result may be inspected between 7 and 8. No protocol modification after
execution begins.**

---

## 14. OUTPUTS

Present now (design artifacts only): `PREREGISTRATION.md`, `DESIGN_AUDIT.md`,
`EXECUTION_PROMPT.md`, `construction_spec.json`, `randomisation_spec.json`,
`classification_spec.json`, `frozen.sha256`, `freeze_rate.py`.

At execution only: `RESULTS.md`, `results.json`, `ENVIRONMENT.txt`, `run.log`,
`RELATED_RUNS.md`, and the harness with its sha256.

**`RESULTS.md` does not exist and must not be created before execution.**
