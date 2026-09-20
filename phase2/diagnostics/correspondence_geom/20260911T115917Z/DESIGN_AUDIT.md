# DESIGN AUDIT — EXP-CORRESPONDENCE-GEOM-002

**Is the synthetic self-pair a trivial image-translation detector rather than a
test of the correspondence machinery?**

Record `phase2/diagnostics/correspondence_geom/20260911T115917Z/`. Date 2026-09-11.

**Audit only. Nothing executed.** No training, no inference, no model
instantiated, no `RESULTS.md`. GEOM-001 untouched, its verdict unchanged. No
historical record modified. **No GEOM-002 response data was inspected — none
exists.**

Tags: **MEASURED** (source/frozen record), **DERIVED** (algebra), **INFERRED**,
**UNKNOWN**.

---

## VERDICT

```
GEOM-002-DESIGN-READY
```

The confound is **real but not fatal**, and it is bounded. Two limitations are
declared in advance and written into the claim ceiling (§6, §13–§15). One
**error in my own prior draft was found and corrected** (§2).

---

## 1. EXACT CONSTRUCTION (objective 1)

Both images are crops of the **same source image** — the scene's left image. The
scene's own right image is never used.

```
left [x] = L[a_L + x]          a_L = 0            rows [0, 272), cols [0, 1136)
right[x] = L[a_R + x]          a_R = t            rows [0, 272), cols [t, 1136+t)
vertical right                                    rows [t, 272+t), cols [0, 1136)
crop 1136 × 272 = (71×16) × (17×16)               identical for every condition
t ∈ {0, 16, 32, 48, 64, 80, 96} px
```

Integer slicing only: **no fill, no padding, no interpolation.** **DERIVED**:
every window lies inside the source for every `t`.

---

## 2. WHICH IMAGE MOVES, AND A SIGN ERROR IN THE PRIOR DRAFT (objective 2)

**DERIVED.** For a scene point at source column `X`:
`x_L = X − a_L`, `x_R = X − a_R`, so

```
d = x_L − x_R = a_R − a_L
```

To impose disparity `t`, the **right crop must start `t` px further RIGHT**:
`a_R = a_L + t`. Verified against `phase2/viz/core.correspondence`
(`x_R = x_L − d`, photometrically confirmed in the test suite) at `t = 0, 16, 48,
96` — all consistent.

> **Error found.** `successor_audit_20260911/RECOMMENDATION.md` specified the
> right crop at `cols [96−t, 1136−t)`, i.e. `a_R = a_L − t`, which imposes
> `d = −t` — a **negative, unrepresentable** disparity. Every level would have
> been invalid and GEOM-002 would have failed for the same class of reason as
> GEOM-001.
>
> **Corrected here to `a_R = a_L + t`.** The prior draft is preserved unmodified;
> the correction is recorded in `construction_spec.json`
> (`sign_correction_note`).

---

## 3. `t` ↔ TRUE DISPARITY (objective 3)

`d = t` px exactly, **uniform over the whole crop**, by construction. In
candidate units `d = t/16`:

| t px | 0 | 16 | 32 | 48 | 64 | 80 | 96 |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidates | 0 | 1 | 2 | 3 | 4 | 5 | 6 |

**No ground truth is consulted.** This is what dissolves the GEOM-001 defect: the
disparity is *set*, not perturbed, so it cannot fall outside the candidate axis.

---

## 4. CROP GEOMETRY — UNINTENDED CUES? (objective 4)

| candidate cue | present? | handling |
|---|---|---|
| fill / padding discontinuity | **no** | integer crops only; all windows inside the source |
| interpolation ringing | **no** | `t` always a multiple of the stride 16 |
| varying output size | **no** | 1136 × 272 in every condition |
| left image varying with `t` | **no** | the left crop is **fixed**; this is load-bearing for §6 |
| right crop content window varying | **yes, unavoidable** | intrinsic to any translation; bounded by the vertical arm and the search-free control |
| receptive-field contamination at the crop edge | **yes** | RF ≈ 477 px exceeds any affordable border (crop is 272 px tall). **Not removable.** Measured by the search-free control. |

---

## 5. DOES IT EXERCISE THE LEFT/RIGHT PATHWAYS? (objective 5)

**MEASURED**, `src/models/stereonet/cost_volume.py`: the *only* place the two
images meet is

```
V[:,c,k,y,u] = Lf[c,y,u+k] − Rf[c,y,u]
```

**DERIVED.** The feature extractor is fully convolutional with total stride 16
(`4 × Conv2d(5, stride=2, padding=2)`, ResBlocks, `Conv2d(3, stride=1)`; no
pooling, no normalisation, no global op), so for `t` a multiple of 16,
`Rf[u] = Lf[u + t/16]` exactly in the interior. Therefore

```
V[k][u] = Lf[u+k] − Lf[u + t/16]
```

**`V` is exactly zero at `k = t/16` and non-zero elsewhere.** The match is
unambiguous, and it sits at a candidate index that moves with `t`. Any response
tracking `t` must pass through this comparison — there is no other left/right
pathway in the graph.

---

## 6. COULD A MONOCULAR / SINGLE-IMAGE SHORTCUT PASS? (objective 6) — **the core question**

### 6.1 Left-only shortcut — **excluded by construction**

The left crop is **identical in every condition**. Any function of the left image
alone is therefore **constant in `t`**, and would fail `C1` and `C2` outright.
This is why the left crop is fixed rather than centred.

### 6.2 Right-only shortcut — **possible in principle, bounded by the vertical arm**

The right crop's content window slides with `t`, so a right-image-only predictor
evaluated on a *fixed* crop-coordinate mask samples different source pixels as
`t` grows. Its median could drift.

**DERIVED:** such a drift is scene-dependent and has no reason to be
`+1 candidate per candidate of t`, monotonically, at six consecutive levels. The
**vertical arm applies the identical displacement to the identical image through
the identical code path**, so it carries the same right-only drift while offering
no horizontal match. `C3` (`|Slope_h| > |Slope_v|`) is precisely the test that
separates them.

### 6.3 "Trivial translation detector" — **the charge, answered**

The sharpest form of the objection: *the model might merely be detecting that the
two images differ by a translation, rather than matching.*

**INFERRED:** in this architecture those are the same operation. Detecting the
translation requires locating the candidate offset at which the two feature maps
agree — that *is* correspondence, and it is the only mechanism available (§5).
There is no alternative route by which a translation could be measured.

### 6.4 The genuine residual — **architecturally forced vs. learned**

**This is the real limitation and it is not removable by this design.**

`V[k] = 0` at `k = t/16` is a property of the *construction*, not of the weights.
Whether the aggregation + soft-argmin maps "the zero slice sits at candidate
`t/16`" to "the output is `t/16`" for **arbitrary** weights is **UNKNOWN**.
ARCH-001 measured that random aggregation weights give near-flat candidate-orbit
responses (range 0.09–1.42 vs trained 5.36–6.39) — suggestive, but that was a
different intervention and does not transfer.

**Consequence:** a GEOM-002 **pass cannot distinguish learned correspondence from
architecturally-forced correspondence.** The claim ceiling (§13) is written to
say only what the design supports.

**Resolvable extension, deliberately NOT included:** a random-weight aggregation
arm, as ARCH-001 used. It would settle §6.4 at ~52 extra forward passes. It is
excluded to keep the design minimal and because instantiating untrained weights
required explicit authorisation in ARCH-001. **Named here so the operator can
add it if they want §6.4 answered.**

---

## 7. DOES THE VERTICAL CONTROL DISCRIMINATE? (objective 7)

**DERIVED.** Vertical self-pair: `V[k][y,u] = Lf[y,u+k] − Lf[y−t/16, u]`. The
cost volume compares **within a row**; a vertical displacement moves content
off-row, where no candidate `k` can reach it. **No zero slice exists**, so no
correspondence signal is available — while the perturbation magnitude, the
displacement, the crop, the mask and the code path are identical.

**Declared weakness.** Road surfaces, lane markings and horizon lines are
vertically self-similar, so a spurious partial match at some `t` is possible. The
vertical arm is therefore a **magnitude-matched null, never a second geometric
test**, and **no predicted slope is assigned to it** — the same rule GEOM-001
adopted after Stage A.

---

## 8. IS THE SEARCH-FREE CONTROL STILL VALID? (objective 8) — **yes, and it gains a provable anchor**

**DERIVED.** For `shift="none"`, `reference_shift` returns its input, so
`V[k] = Lf[u] − Rf[u]` for **all** `k` — `k`-constant. Under the self-pair
`Rf[u] = Lf[u+t/16]`, so `V[k] = Lf[u] − Lf[u+t/16]`: non-zero in content,
**degenerate in `k`.** It carries the perturbation magnitude but **no
candidate-selective information**. Exactly the intended baseline.

**Free wiring check with an exact value.** At `t = 0` the self-pair is `(L, L)`,
so `V[k] = Lf[u] − Lf[u] = 0` **exactly**. Then `Agg(0)` is constant in `k` with
*no* boundary effect (the padding zeros equal the signal zeros), standardisation
returns exactly 0, the softmax is uniform, and

```
disparity_initial = (0+1+…+11)/12 = 5.5   EXACTLY, at every pixel
```

**MEASURED-by-derivation.** This is recorded as a **wiring check, not a
scientific criterion** — a deviation indicates an implementation fault, not a
finding.

---

## 9. ARE ALL SIX LEVELS REPRESENTABLE AND CLEAR OF THE RAILS? (objective 9)

**DERIVED.** Candidate axis `0…11`.

| t px | candidate | dist to rail 0 | dist to rail 11 |
|---:|---:|---:|---:|
| 16 | 1 | 1 | 10 |
| 32 | 2 | 2 | 9 |
| 48 | 3 | 3 | 8 |
| 64 | 4 | 4 | 7 |
| 80 | 5 | 5 | 6 |
| 96 | 6 | 6 | 5 |

All six are representable. **`t = 16` sits one step from the low rail** — the
tightest clearance — and is retained because dropping it would not help: the
aggregation's receptive field is 11 over `D = 12`, so **only candidates 5 and 6
are padding-free** and every level is boundary-affected regardless. This is
declared, not engineered around.

**Note `t` need not be ≤ 96 for representability** (candidates 7–8 would also fit);
96 is chosen so the response stays clear of *both* rails while the crop remains
stride-aligned at 1136 × 272.

---

## 10. IS THE STATISTIC MATHEMATICALLY CORRECT? (objective 10)

```
m_axis(t)  = median( disparity_initial[mask] )
Slope_axis = Σ_t (m_axis(t) − m_axis(0)) · (t/16)  /  Σ_t (t/16)²        t ∈ {16,…,96}
```

**DERIVED.** Under correspondence, `disparity_initial` reports the true disparity
`t/16`, so `m(t) − m(0) = t/16 − 0 = t/16` and

```
Slope_h = Σ (t/16)² / Σ (t/16)² = 1      exactly
```

**The anchor is correct.** Two properties worth stating:

- subtracting `m(0)` makes the statistic **immune to a constant per-model bias**,
  so a model that reports `t/16 + c` still yields `Slope = 1`;
- the fit is through the origin, matching the construction (`t = 0 ⇒ Δd = 0`).

`Slope_v` uses the identical formula on the vertical arm. **No predicted value is
assigned to it** (§7).

---

## 11. SHOULD THE UNITS BE 3 CHECKPOINTS × 4 HELD-OUT SCENES? (objective 11)

**Yes, with the dependence stated.**

- **Weights:** the 3 frozen `shift="left"` checkpoints are the **entire available
  population** — training is forbidden. Reuse is irreducible. This, not the scene
  count, is the binding limit on evidential strength.
- **Scenes:** **MEASURED** — `hailo_val` holds 40 scenes; 4 were used by
  GEOM-001; 36 have never appeared in any correspondence experiment. Four are
  selected by a **GT-free rule fixed in advance**: the first four indices
  excluding the GEOM-001 focus set `{27,0,31,6}` → indices **1, 2, 3, 4**
  (`000161_10`, `000162_10`, `000163_10`, `000164_10`). No ground truth was
  consulted.
- **Structure:** 3 × 4 = **12 positive units**, plus 4 control units — matching
  every prior record for direct comparability.
- **Dependence:** units sharing a checkpoint share all weights; units sharing a
  scene share the source image. **Not independent replicates.** This is a
  deterministic causal diagnostic, not a population inference; no p-value is used
  anywhere.

---

## 12. ARE THE FOUR CONDITIONS PREREGISTRABLE AND NON-POST-HOC? (objective 12)

**Yes. No GEOM-002 data exists, so none could have informed them.**

| condition | a-priori source | threshold-free? |
|---|---|---|
| `C1  Slope_h > 0` | sign of the geometric prediction (§10) | yes |
| `C2  m_h(t)` strictly increasing over the 6 levels | monotonicity of `d = t/16` | yes — `1/6! = 1/720` under a null |
| `C3  \|Slope_h\| > \|Slope_v\|` | axis specificity (§7) | yes |
| `C4  Slope_h > Slope_h(NEG, same scene)` | baseline **measured in-run** (§8) | yes |

**`Slope_h` is reported against the anchor `1.0` descriptively and is NEVER a
pass criterion**, because no tolerance on it can be justified a priori. This is
the GEOM-001 rule carried forward unchanged.

The one number imported from a previous record — `t_max = 96` — was chosen from
the **candidate axis and crop arithmetic**, not from any observed response.

---

## 13. WHAT A POSITIVE RESULT WOULD ESTABLISH (objective 13)

> On synthetic fronto-parallel stereo pairs, the trained pipeline's
> `disparity_initial` tracks the imposed horizontal displacement, specifically on
> the epipolar axis, beyond a matched vertical control and beyond a provably
> search-free baseline.

Status: **`SYNTHETIC-CORRESPONDENCE-EVIDENCE`** — consistency evidence that the
correspondence pathway functions when an unambiguous match is present.

It would establish that the model is **not** monocular-on-left (§6.1), that the
response is **horizontal-axis-specific** (§7), and that it **exceeds a search-free
model** (§8).

---

## 14. WHAT A NEGATIVE RESULT WOULD ESTABLISH (objective 14)

> `SYNTHETIC-CORRESPONDENCE-NOT-DEMONSTRATED`

The pipeline did not track an unambiguous, exactly-known horizontal
correspondence on the easiest possible input. Because the input is
out-of-distribution (constant disparity is never seen in training), a failure is
**ambiguous between "no functioning correspondence machinery" and "the machinery
does not generalise to this input"**.

**It must NOT be concluded that correspondence is absent.**

**Stated asymmetry:** GEOM-002 is a **strong falsifier and a weak confirmer**. A
failure here is the most informative outcome this campaign could obtain, because
every prior experiment has been a weak confirmer.

---

## 15. WHAT GEOM-002 CANNOT ESTABLISH (objective 15)

- **Real-scene correspondence.** A constant-disparity plane has identical
  texture, no occlusion, no slant, no ambiguity, no repeated structure. Real
  stereo has all of them.
- **Disparity correctness.** `Slope_h` is descriptive; magnitude is never a pass
  criterion. Nothing about accuracy, sub-candidate precision or bias.
- **Genuine disparity search.** Locating an exact zero is far weaker than
  searching a real cost landscape.
- **Learned vs. architectural.** §6.4 — unresolved by design, and resolvable only
  by the random-weight arm named there.
- **Generalisation.** Four scenes, three checkpoints, one synthetic construction.
- **Anything about `disparity_final`** — refinement is never invoked.
- **Full confirmation.** The three checkpoints are reused and cannot be replaced
  without training; the achievable status is **consistency evidence with
  predictive content on the data and intervention axes**, not independent
  confirmation.

---

## 16. WHY THE CONFOUND IS NOT FATAL — summary

| objection | resolution |
|---|---|
| trivial image-translation detector | detecting the translation *is* correspondence here; the cost volume is the only left/right pathway (§5, §6.3) |
| monocular-on-left shortcut | **excluded** — the left crop is fixed, so left-only is constant in `t` (§6.1) |
| monocular-on-right shortcut | possible drift, bounded by the matched vertical arm via `C3` (§6.2, §7) |
| architecturally forced rather than learned | **real, unresolved, declared** — bounds the claim, does not invalidate the test (§6.4) |
| out of distribution | **real, declared** — makes a negative ambiguous, not a positive invalid (§14) |
| crop/edge artifacts | present and not removable; measured by the search-free control (§4, §8) |

**`GEOM-002-DESIGN-READY`** — with the sign corrected, the two limitations
declared in advance, and the claim ceiling written to match what the design can
actually support.
