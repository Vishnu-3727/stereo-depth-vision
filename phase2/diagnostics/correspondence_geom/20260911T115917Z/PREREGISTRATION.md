# EXP-CORRESPONDENCE-GEOM-002 — PRE-REGISTRATION

**Synthetic fronto-parallel self-pair sweep: does `disparity_initial` track an
imposed horizontal displacement whose true disparity is known exactly by
construction?**

Record `phase2/diagnostics/correspondence_geom/20260911T115917Z/`.
Frozen 2026-09-11. **NOT EXECUTED.**

Inference only. No training, no fine-tuning, no weight change, no architecture
change, no new checkpoint. Phase 1 untouched. GEOM-001 untouched; its verdict
remains `GEOMETRIC-CORRESPONDENCE-NOT-DEMONSTRATED`. No historical record
modified.

**Frozen artefacts, written before this document, hashed over FILE BYTES:**

```
construction_spec.json  sha256 a9e8f250bfef49607951fae0b304d5713091553ad2da885b911f4de711bdea10
mask_spec.json          sha256 b274ce037a1fbd6973f8470baf1c38f25a50267c1b740548df4f224598b4b5d7
```

Both written with `newline=""` so `sha256sum -c frozen.sha256` reproduces them
externally — the GEOM-001 CRLF lesson, applied at freeze time.

---

## 0. DISCLOSURES — recorded before execution, not attached afterwards

1. **The designer has read the GEOM-001 response curve.** It is excluded as a
   design input and as evidence. Every quantity in this document derives from
   source, integer geometry, or the frozen specs.
2. **The three `shift="left"` checkpoints are reused and cannot be replaced** —
   training is forbidden and they are the entire available population.
3. **Therefore this experiment cannot be confirmatory in the strict sense.** Its
   achievable status is **consistency evidence with predictive content on the
   data and intervention axes**: the scenes are held out and the intervention has
   never been run on any checkpoint.
4. **A negative result is ambiguous** because the input is out of distribution
   (§9).
5. **A positive result cannot distinguish learned from architecturally-forced
   correspondence** (`DESIGN_AUDIT.md` §6.4). The optional random-weight arm that
   would resolve it is named there and deliberately excluded.
6. **A sign error in the prior draft was corrected.**
   `successor_audit_20260911/RECOMMENDATION.md` specified the right crop at
   `cols [96−t, 1136−t)`, imposing `d = −t` — negative and unrepresentable.
   Corrected here to `a_R = a_L + t`. The draft is preserved unmodified.

---

## 1. QUESTION

When the right image is a pure horizontal translation of the left by `t` px — a
stereo pair whose true disparity is exactly `t` everywhere — does
`disparity_initial` track `t/16` candidates, specifically on the horizontal axis,
beyond a matched vertical control and beyond a provably search-free model?

---

## 2. SOURCE-DERIVED GEOMETRY

**MEASURED**, `src/models/stereonet/cost_volume.py`
(`shift_left(x,k) = F.pad(x,(0,k))[..., k:]`):

```
V[:,c,k,y,u] = Lf[c,y,u+k] − Rf[c,y,u]
```

**MEASURED**, `src/models/stereonet/feature_extractor.py`: fully convolutional,
`4 × Conv2d(5, stride=2, padding=2)` then ResBlocks and `Conv2d(3, stride=1)`; no
pooling, no normalisation, no global operator; `scale = 2⁴ = 16`. An image
translation of `16n` px therefore produces **exactly** an `n`-sample feature
translation in the interior.

**MEASURED**, `src/models/stereonet/regression.py`: `soft_argmin` over the index
grid `0…11` ⇒ `disparity_initial` is in **candidate units**.

**DERIVED — the construction.** With `left[x] = L[a_L+x]` and
`right[x] = L[a_R+x]`, a scene point at source column `X` has `x_L = X − a_L` and
`x_R = X − a_R`, so

```
d = x_L − x_R = a_R − a_L
```

Setting `a_L = 0`, `a_R = t` imposes **`d = t` px exactly, uniform over the whole
crop**. Verified against `phase2/viz/core.correspondence` (`x_R = x_L − d`,
photometrically confirmed in the test suite) at `t = 0, 16, 48, 96`.

**DERIVED — the cost volume under this construction.** Since
`Rf[u] = Lf[u + t/16]`:

```
V[k][u] = Lf[u+k] − Lf[u + t/16]     ⇒  EXACTLY ZERO at k = t/16, non-zero elsewhere
```

The match is unambiguous and sits at a candidate index that moves with `t`.

---

## 3. INTERVENTION — frozen in `construction_spec.json`

Both images are crops of the **same source image**: `scene.left`. **The scene's
own right image is never used.**

```
left  crop :  rows [0, 272), cols [0, 1136)                FIXED for every condition
right crop :  horizontal  rows [0, 272),  cols [t, 1136+t)
              vertical    rows [t, 272+t), cols [0, 1136)
crop size  :  1136 × 272 = (71 × 16) × (17 × 16)           identical every condition
t ∈ {0, 16, 32, 48, 64, 80, 96} px                          13 distinct conditions (t=0 shared)
```

Integer slicing only: **no fill, no padding, no interpolation**. Every `t` is a
multiple of the stride. **DERIVED**: every window lies inside the source image
for every `t`.

**The left crop is fixed in every condition.** This is load-bearing: any
left-image-only function is therefore constant in `t` and cannot pass.

### Representability and rail clearance (DERIVED)

| t px | 0 | 16 | 32 | 48 | 64 | 80 | 96 |
|---|---:|---:|---:|---:|---:|---:|---:|
| true candidate | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| dist. to rail 0 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| dist. to rail 11 | 11 | 10 | 9 | 8 | 7 | 6 | 5 |

All levels representable. **The GEOM-001 defect cannot recur**: the disparity is
*set*, not perturbed, so it cannot fall off the axis. `t_max = 96` is chosen from
the candidate axis and crop arithmetic alone.

---

## 4. MASK — frozen in `mask_spec.json`

```
retain (y, x) in CROP coordinates iff   64 ≤ y < 208   AND   64 ≤ x < 1072
retained region 1008 × 144 = 145 152 px per scene
```

**Purely geometric. No ground truth is consulted** — the true disparity is `t` by
construction and uniform. This removes the GT band, the scene-disparity selection
and the mask-occupancy confound present since INDEX-001.

Single, fixed mask: identical for every `t`, every axis, every model and every
scene. The border reduces, but cannot remove, contamination from the ≈477 px
feature receptive field; the search-free control measures the residual.

---

## 5. CONDITIONS AND CONTROLS

| role | model | shift | seed | sha256 |
|---|---|---|---|---|
| test | `POS_6b_seed0` | `left` | 0 | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` |
| test | `POS_6b_seed1` | `left` | 1 | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` |
| test | `POS_6b_seed2` | `left` | 2 | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` |
| artifact baseline | `NEG_shift_none` | `none` | — | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` |

All frozen, read-only, standardised parameter-free readout.

- **Matched negative axis:** vertical self-pair at the same `t` — same source
  image, same displacement magnitude, same crop size, same code path (the axis is
  a function argument), same mask, same preprocessing. **No predicted slope is
  assigned to it.**
- **Search-free artifact baseline:** `shift="none"`, whose volume is `k`-constant
  by construction, so it carries the perturbation magnitude but **no
  candidate-selective information**. Its `Slope_h`, `Slope_v` are **measured, not
  assumed zero**.

### Wiring check with an exact value (not a scientific criterion)

**DERIVED:** at `t = 0` the self-pair is `(L, L)`, so for `shift="none"`
`V[k] = 0` exactly; `Agg(0)` is constant in `k` with no boundary effect (padding
zeros equal signal zeros); standardisation returns exactly 0; the softmax is
uniform; therefore

```
disparity_initial = (0+1+…+11)/12 = 5.5   EXACTLY
```

A deviation indicates an implementation fault, **not a finding**.

---

## 6. UNITS

```
12 positive units = 3 weight sets × 4 held-out scenes
 4 control units  = 1 search-free model × 4 held-out scenes
```

**Scenes**, selected by a **GT-free rule fixed in advance** — the first four
`hailo_val` indices excluding the GEOM-001 focus set `{27, 0, 31, 6}`:

```
1 000161_10.png   2 000162_10.png   3 000163_10.png   4 000164_10.png
```

**MEASURED:** `hailo_val` holds 40 scenes; 4 were used by GEOM-001; these 4 have
never appeared in any correspondence experiment. No ground truth was consulted to
select them.

Units are **not independent replicates**: units sharing a checkpoint share all
weights; units sharing a scene share the source image. Deterministic causal
diagnostic, not population inference.

---

## 7. PRIMARY STATISTIC — frozen

```
m_axis(t)  = np.median( disparity_initial[mask] )                  candidate units
Slope_axis = Σ_t (m_axis(t) − m_axis(0)) · (t/16)  /  Σ_t (t/16)²   t ∈ {16,32,48,64,80,96}
ANCHOR:  Slope_h = 1 candidate per candidate                        exact, by construction
```

`np.median` — the aggregate used by INDEX-001/002, ARCH-001 and GEOM-001.

**DERIVED anchor proof.** Under correspondence `m(t) = t/16` and `m(0) = 0`, so
`m(t) − m(0) = t/16` and `Slope_h = Σ(t/16)²/Σ(t/16)² = 1`. Subtracting `m(0)`
makes the statistic immune to a constant per-model bias; the fit is through the
origin, matching `t = 0 ⇒ Δd = 0`.

`Slope_v` uses the identical formula on the vertical arm.

**No p-value.** Conditions are not exchangeable; units share weights. Paired and
descriptive throughout.

---

## 8. DECISION RULE — frozen, threshold-free, unit-level

Per positive unit:

```
C1  Slope_h > 0                                     source-derived direction
C2  m_h(t) strictly increasing over t = 0,16,32,48,64,80,96     6-level monotonicity
C3  |Slope_h| > |Slope_v|                           axis specificity
C4  Slope_h > Slope_h(NEG, same scene)              exceeds the search-free baseline
```

```
PASS  iff  C1 ∧ C2 ∧ C3 ∧ C4  at 12/12 positive units
otherwise FAIL, classified by §10
```

No partial counts, no majority rule, no mean rule, no threshold relaxation, no
post-hoc exception. **`Slope_h` is reported against the anchor `1.0`
descriptively and is never a pass criterion**, because no tolerance on it can be
justified a priori.

**DERIVED power:** strict monotonicity over 6 levels has probability `1/6! = 1/720`
per unit under an exchangeable null — against `1/2` for any two-level design.

---

## 9. INTERPRETATION — fixed before execution

**On PASS**, the maximum claim is exactly:

> "On synthetic fronto-parallel stereo pairs, the trained pipeline's
> `disparity_initial` tracks the imposed horizontal displacement, specifically on
> the epipolar axis, beyond a matched vertical control and beyond a provably
> search-free baseline."

Status: **`SYNTHETIC-CORRESPONDENCE-EVIDENCE`**.

**On FAIL**: `SYNTHETIC-CORRESPONDENCE-NOT-DEMONSTRATED`. **It must NOT be
concluded that correspondence is absent** — the input is out of distribution, so
a failure is ambiguous between "no functioning correspondence machinery" and
"the machinery does not generalise to this input".

**Stated asymmetry:** this design is a **strong falsifier and a weak confirmer**.

### Explicitly NOT establishable, whatever the outcome

Real-scene correspondence; disparity correctness or accuracy; sub-candidate
precision; genuine disparity search; learned-versus-architectural origin
(`DESIGN_AUDIT.md` §6.4); generalisation; anything about `disparity_final`
(refinement is never invoked). No EPE, D1 or RMSE is computed.

---

## 10. FAILURE CLASSIFICATION — distinguished, not collapsed

| # | class | signature |
|---|---|---|
| 1 | correspondence-positive | C1∧C2∧C3∧C4 at 12/12 |
| 2 | flat response | `m_h(t)` ≈ constant — consistent with a monocular-on-left pathway |
| 3 | non-specific response | `Slope_h ≈ Slope_v`; C3 fails — generic displacement sensitivity |
| 4 | search-free-equivalent | positives do not exceed `NEG`; C4 fails |
| 5 | non-monotone / mixed | C2 fails, or units disagree |
| 6 | protocol failure | any §11 hard stop |

---

## 11. CONTROLS AND HARD STOPS

| check | criterion |
|---|---|
| spec digests | `construction_spec.json` and `mask_spec.json` re-verified over **file bytes** before the first model instantiation |
| checkpoint hashes | the four §5 sha256, verified at load |
| readout parameter-free | `sum(p.numel() for p in model.regression.parameters()) == 0` |
| crop invariance | every condition yields exactly `1136 × 272` |
| no fill / no interpolation | every crop window asserted inside the source; all `t` multiples of 16 |
| left crop fixed | asserted identical in every condition |
| identical code path | the axis is a function argument, not a branch |
| mask invariance | retained count `145 152` in every condition, axis, model and scene |
| scenes held out | selected index set disjoint from `{27, 0, 31, 6}` |
| `shift="none"` degeneracy | `max|slice_k − slice_0| == 0` re-verified per scene |
| `t = 0` wiring check | `NEG` returns `5.5` exactly |
| no training | no optimizer imported or constructed; no `.backward()`, no `.step()`; weights re-hashed after every scene |
| determinism | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `no_grad`, `eval`, fp32 |
| forward-pass count | exactly `4 × 4 × 13 = 208` |

**HARD STOP** on any failure above, on any protocol modification after execution
begins, on runtime exceeding 10 minutes, or if any historical record would need
modification.

---

## 12. COMPUTE

```
4 models × 4 scenes × 13 conditions = 208 forward passes to disparity_initial
```

Crop `1136 × 272`, smaller than native. Refinement never invoked. **Estimate
under 2 minutes.** No training, no new checkpoint, no new architecture.

---

## 13. PREREGISTRATION ORDER

1. construction frozen → `construction_spec.json` ✔
2. mask frozen → `mask_spec.json` ✔
3. geometry derived from source → §2 ✔
4. statistic frozen → §7 ✔
5. decision rule frozen → §8 ✔
6. controls frozen → §5, §11 ✔
7. `PREREGISTRATION.md` written → this document ✔
8. **execute** → NOT DONE
9. `RESULTS.md` → NOT CREATED

**No result may be inspected between 7 and 8. No protocol modification after
execution begins.**

---

## 14. OUTPUTS

Present now (design artifacts only): `PREREGISTRATION.md`, `DESIGN_AUDIT.md`,
`EXECUTION_PROMPT.md`, `construction_spec.json`, `mask_spec.json`,
`frozen.sha256`, `freeze_geom2.py`.

To be created **at execution only**: `RESULTS.md`, `results.json`,
`ENVIRONMENT.txt`, `run.log`, `RELATED_RUNS.md`, and the harness with its sha256.

**`RESULTS.md` does not exist and must not be created before execution.**
