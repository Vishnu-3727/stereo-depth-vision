# PIPELINE-LEVEL CORRESPONDENCE — DESIGN GATE

**Id:** `correspondence_pipeline/design_20260913T055200Z`
**Type:** design gate. Nothing executed, no checkpoint opened, no training, no
code modified, no result generated.
**Supersedes as the active branch:** the module-level synthetic branch
(`correspondence_geom/*`), which is closed: O1 NOT IDENTIFIABLE
(`o1_final_blind_audit_20260913T050232Z`), CP-001 substantively FAILED
(`direction_decision_20260913T052115Z/HS_BAND_RESULT.md`).

---

## 0. The realisation that makes the pipeline question tractable

The counterexample that killed O1 and motivated CP-001 was a **candidate-pointwise
detector** `c(k,w) = Σ_c |V[c,k,w]|` inside the frozen `Aggregation`.

At **module** level that is correspondence-free: the module never compares left
and right, because `V = Lf[w+k] − Rf[w]` is computed upstream in frozen code.

At **pipeline** level the same object is `argmin_k Σ_c |Lf[w+k] − Rf[w]|` —
**winner-take-all block matching**, the canonical stereo correspondence
algorithm. It *is* genuine geometric correspondence.

So the adversary set shrinks when the unit of claim moves from the module to the
pipeline. The pointwise latch leaves `H_ALT` and joins `H_CORR`. This is why the
pipeline question can be identifiable while the module question provably was not,
and it is the only reason this branch is worth opening.

Corollary, binding on every claim below: a positive result is a statement about
the **pipeline**. It must never be reported as "the aggregation performs
correspondence".

---

## 1. Current claim ceiling

> **Level D — genuine geometric correspondence NOT-DEMONSTRATED.**

Evidence for: none. Evidence against: none. Not demonstrated is not absent.

---

## 2. New scientific question

> **At a pixel whose left-image content and its own monocular context are held
> fixed, does the pipeline's pre-refinement disparity (`disparity_initial`)
> follow a deliberately imposed, spatially localised change in the true
> left-right geometric offset — and fail to follow it when that offset is
> imposed with content that has no correspondence to the left image?**

The wording is deliberately narrower than the candidate formulation in the brief
in three ways: it fixes the **stage** (`disparity_initial`, not final), it fixes
the **locality** (a spatially localised offset, which is what defeats the
strongest adversary), and it pairs the treatment with a
**correspondence-destroyed, statistics-matched** arm rather than with mere
right-image corruption.

---

## 3. Hypotheses

* **H_CORR** — disparity selection depends on geometrically correct left-right
  correspondence. Includes block matching, cost-volume WTA, learned
  regularised matching. All predict: the prediction tracks a *local* imposed
  offset, locally, and does not track it when no matching content exists.
* **H_ALT** — the stereo dependence is explained by any combination of:
  monocular depth from the left image; generic (non-geometric) right-image
  dependence; cost-volume global statistics; candidate-axis architecture;
  candidate-index/padding effects; soft-argmin/readout behaviour;
  left-image-guided refinement.

`H_ALT` is not hypothetical here. It has a **trained, accurate instantiation in
this project**: the deployed `shift="none"` checkpoint has a bit-exactly
degenerate cost volume (all 12 slices identical, `max|slice_k − slice_0| = 0.0`)
and still scores **8.15 % D1**. Any pipeline-level design must treat that model
as the null, not as a curiosity.

---

## 4. Candidate interventions considered

| Intervention | Correspondence-discriminating? | Main confound | Decision |
|---|---|---|---|
| Global right-image translation, slope of `d̂` vs shift | **NO** | already rejected on record: the degenerate model imitates sign and monotonicity (measured response up to 0.70 candidates); only calibration separates, and nothing forbids coincidence. A global scalar shift estimate is obtainable from `L−R` statistics without alignment | **REJECT as evidence.** Retain **only as an internal positive control** (see §7) |
| Right-image corruption / noise / blur | **NO** | prohibited standalone; degradation is explained by statistics, not geometry | REJECT |
| Mismatched scene pairing (L_A with R_B) | **WEAK** | content statistics change with the scene; degradation non-specific | REJECT as primary; keep only inside the statistics-matched swap control |
| Vertical shift as a correspondence-breaker | **NO** | on record: response is *larger* vertically (+0.0048 vs −0.0017); and `FH = 17 < 30` cells makes a halo-free vertical band empty at any feasible crop | REJECT |
| Translated-self-pair candidate-order statistics | **NO** | class retired: `τ` is a coordinate change, `(u,d)` reparametrisation | PROHIBITED |
| O1 / candidate-axis permutation | **NO** | verdict C, pointwise latch reproduces the signature | PROHIBITED |
| CP-001 context propagation | **NO (unexecutable)** | HS-BAND: C1 and C2 co-satisfiable only at 3–5e-06 of the readout scale, 6–11 float32 ULPs | PROHIBITED |
| Direct edits to `V` (Type C) | **N/A** | no stereo pair maps to the edited tensor; cannot evidence natural stereo behaviour | REJECT as evidence; permitted only as a Type-B module probe, never reported as pipeline behaviour |
| Cost-volume slice masking / candidate ablation (Type B) | PARTIAL | tells us what the readout does with a volume, not whether the volume's geometry drives selection | REJECT as primary |
| `disparity_initial` vs `disparity_final` comparison | **NO alone** | a stage-attribution diagnostic, not a mechanism test | Keep as mandatory stage separation, not as evidence |
| **Zone-local imposed disparity offset** (this design) | **YES** | seam cue; zone-statistics cue; zone-interior size vs the 477 px extractor receptive field | **ACCEPT** |

---

## 5. Strongest surviving intervention — `EXP-CORRESPONDENCE-ZONE-001`

### 5.1 Construction (locally Type A)

The KITTI frame is **1232×368**; the working crop is **1136×272**. That leaves
**96 px of horizontal and 96 px of vertical slack**, i.e. a right-crop origin can
be moved by up to **96 px = 6 candidates** in either direction **with no missing
data, no synthetic fill and no invalid border**. This is the same mechanism
GEOM-002 already used (`right_origin(axis, t)`), so it is settled machinery.

* Left crop: **fixed**, origin `(0,0)`, never edited, in every arm.
* Right crop: assembled column-wise from the **same real frame** at two origins.
  For a chosen zone `Z` (a contiguous horizontal span of the crop):

      right_crop[:, x] = frame[:, x + 16·Δ]     for x in Z
      right_crop[:, x] = frame[:, x]            for x outside Z

  Inside `Z` the true disparity is `d(x,y) − 16Δ`; outside it is `d(x,y)`.
  Both are exact and known from the existing GT.
* Every pixel used in the statistic lies in the **interior** of `Z` or of its
  complement, at least **318 px** from the seam (238 px extractor radius + 80 px
  = 5 feature cells of aggregation reach). For such pixels the input is
  **bit-indistinguishable** from a genuine rectified stereo pair whose disparity
  is uniformly `d − 16Δ` (or `d`): the seam is outside their entire receptive
  field. The composite is globally stitched, but **locally Type A**, and that is
  the property the claim rests on. `HS-Z` measures it.
* Geometry budget: `1136 − 2·318 = 500 px` of interior across both regions,
  ≈ 250 px ≈ 15 feature cells each. Non-empty with margin. Contrast with CP-001,
  which needed pixels *at* the receptive-field boundary and therefore died; this
  design needs pixels *deep inside* a region, which is the easy direction.

### 5.2 Arms

| Arm | Δ (candidates) | Zone |
|---|---|---|
| treatment | −6, −3, −1, 0, +1, +3, +6 | left half |
| treatment, mirrored | same | right half |
| **swap control** (primary null) | same | zone content taken from a **different scene** at the same origin offset |
| **global positive control** | same | zone = whole crop |

Mirroring exists so that "tracking follows the zone" can be separated from
"tracking follows a side of the image".

### 5.3 Primary statistic

Per checkpoint, per scene, per zone placement, on **GT-valid pixels only**
(KITTI GT is sparse; that is acceptable and reduces `n`, it does not bias):

    β_in  = ∂ d̂_initial / ∂Δ  estimated over the ZONE interior
    β_out = ∂ d̂_initial / ∂Δ  estimated over the COMPLEMENT interior
    S     = β_in − β_out                     ← PRIMARY

Units: candidates per candidate. `H_CORR` predicts `β_in = −1`, `β_out = 0`,
hence `S = −1` (sign fixed by the recorded convention `d = x_L − x_R`, to be
pinned by `HS-FRAME` before freezing). `H_ALT` with any global shift estimate
`ĝ` predicts `β_in = β_out` and hence **`S = 0`**, whatever `ĝ` is.

`S` is a **difference of slopes measured inside one image under one forward
pass**, so it is invariant to: the recorded candidate↔GT frame offset (an
intercept), soft-argmin centre pull, z-score scale, padding contamination,
sharpness, and any global right-image dependence. Those nuisances enter `β_in`
and `β_out` identically and cancel. This is the design's whole point.

### 5.4 Secondary statistic (reported, never the gate)

Saturation breakpoint: for a pixel of GT disparity `d_p` candidates, tracking
must stop when `d_p − Δ` leaves `[0, 11]`, so the breakpoint location in Δ moves
with `d_p` at **unit slope**: `∂Δ*/∂d_p = 1`. Reachable with the available
`Δ ∈ [−6,+6]`: the lower breakpoint for pixels with `d_p ≤ 6`, the upper for
`d_p ≥ 5`. Intercept-free, so it also survives the frame-offset discrepancy.
Secondary because it needs dense GT coverage across `d_p`.

---

## 6. Adversarial alternative, and why it fails

### The strongest correspondence-free mechanism (`A4`)

    d̂(x,y) = clamp( m(x,y) − ĝ , 0, 11 )

* `m(x,y)` — monocular depth from the **left** image alone. Not hypothetical:
  the `shift="none"` checkpoint achieves 8.15 % D1 this way, with a provably
  search-free cost volume.
* `ĝ` — a **single global scalar** read off non-geometric statistics of the
  `k`-constant term `Lf − Rf` (the deployed model's measured 0.70-candidate
  response to right-image translation is exactly this channel).

`A4` defeats every weaker intervention: it produces unit slope under global
right-image translation, correct sign, correct monotonicity, degradation under
right-image corruption, a plausible candidate profile, good EPE — and a
breakpoint that moves with `d_p` at unit slope, since `m ≈ d_p`. It is why the
global-translation slope test was correctly rejected.

### Why it fails against the zone intervention

`A4`'s shift estimate is **one number for the whole image**. The zone
intervention imposes **two different true offsets in one image**, with the left
image and hence `m(x,y)` **identical in every arm**. Therefore `A4` must apply
the same `ĝ` to both regions and predicts `β_in = β_out`, i.e. `S = 0`, for every
`ĝ`, however it is computed and however accurate it is.

To produce `S = −1`, `A4` must be upgraded to a **spatially varying** `ĝ(x,y)`
that recovers, at each location, the local left-right offset — from the left
image (which is unchanged across arms, so carries zero information about Δ) and
the right image (whose local content only reveals Δ **relative to** the left).
Recovering a spatially varying offset between two images **is** local
correspondence. **The upgraded adversary is inside `H_CORR`, not `H_ALT`.**
That is the identification argument, and it is the only one in this project that
closes.

### Residual adversaries, and the control that kills each

| Adversary | Kill |
|---|---|
| `A5` — reads Δ from the seam | tested pixels are ≥ 318 px from the seam, outside extractor RF (238 px) + aggregation reach (80 px). `HS-Z` measures this rather than assuming it — the CP-001 lesson. |
| `A6` — maps zone-level content statistics to a shift | the **swap control**: the zone is filled from a different scene at the same offset, so zone statistics change comparably but no candidate corresponds. `H_CORR` predicts `β_in → 0`; `A6` predicts `β_in` unchanged. |
| `A7` — "tracking follows the left/right half of the frame" | the **mirrored zone** arm. |
| `A8` — architectural/readout artefact | `S` is a difference of slopes within one forward pass; nuisances cancel. Plus the `shift="none"` arm, which cannot search and must give `S ≈ 0`. |
| `A9` — refinement manufactures the effect | primary stage is `disparity_initial`, **pre-refinement**; refinement is left-guided and is measured separately. |

---

## 7. Controls (all mandatory)

1. **Swap control** — primary null. Statistics-matched, correspondence-destroyed,
   same Δ, same zone, same pipeline. This is the nuisance-matched mechanistic
   null every previous experiment in this project lacked.
2. **`NEG_shift_none`** — the deployed degenerate checkpoint. A trained, accurate
   model that **provably cannot search** (all 12 slices bit-identical). It must
   give `S ≈ 0`. It is the empirical instantiation of `H_ALT` and was wrongly
   excluded from GEOM-002's authorisation. Including it is the single largest
   improvement over every prior design here.
3. **Global positive control** — `Δ` applied to the whole crop must give
   `β ≈ −1`. If even this fails, the run is **inconclusive, not negative**: it
   would mean the effect is undetectable at this stage for reasons unrelated to
   locality.
4. **Mirrored zone** — as above.
5. **Random weights** — declared explicitly as a **floor, not a mechanistic
   null**. Reported, never gating.

---

## 8. Stage attribution

| Stage | Role |
|---|---|
| cost volume `V` | verified geometrically, model-free (`HS-FRAME`); not a claim target |
| aggregated cost profile | reported as a mechanistic descriptor only |
| **`disparity_initial`** | **primary** — this is where the claim lives |
| `disparity_final` (refined) | secondary, reported; the refinement is left-guided and supplies nearly all the magnitude, so it may show `S ≈ 0` even on a positive — that would be a finding about the refinement, not a negative |
| EPE / D1 | **not used as evidence at any stage** |

---

## 9. Requirements

| Question | Answer | Why |
|---|---|---|
| New data required? | **NO** | KITTI 2015 `hailo_val` real pairs with GT are present, and the 96 px frame slack supplies Δ ∈ [−6,+6] border-free |
| New training required? | **NO** | inference only |
| Existing checkpoints sufficient? | **YES** | the three `shift="left"` H2 seeds, plus the deployed `shift="none"` checkpoint as the null |
| New architecture required? | **NO** | |

Standing defect, declared: the same three trained checkpoints for the twelfth
time. `n = 3` at the checkpoint level. Do not present scenes, Δ levels or zone
placements as replications.

---

## 10. Inference, gate, stopping

* **Primary inference unit: checkpoint** (`n = 3`). Scenes (6 pairs), Δ levels
  (7) and zone placements (2) are **repeated measures**, not samples. GT-valid
  pixels within a zone interior are ~15 feature cells wide — the pixel count
  overstates independence by ≈16×, as already established for the earlier mask.
* **Gate: rate/quantile, preregistered, never `max`.** `max(h)` over units is a
  statistic of the number of units — the measured GEOM-002 error
  (`P(pass) = 1.4e-14`).
* **Threshold: not set in this record.** It must be derived from the measured
  spread of the **swap control** and the `shift="none"` arm, before any trained
  `shift="left"` arm is read. No post-hoc selection. No threshold tuning.
* **Seeds:** not required beyond the existing three checkpoints; no new seeds.
* **Preregistration must fix, before any trained arm is loaded:** the zone spans,
  the 318 px exclusion, the Δ set, the GT validity rule, the slope estimator,
  the sign convention from `HS-FRAME`, the gate rate and its threshold, and the
  downgrade rule for control failures.

### Hard stops (all model-free, no checkpoint, seconds)

* **`HS-FRAME`** — pin the candidate↔GT frame mapping and the sign of `β`
  numerically, on synthetic data with known disparity, no model. The recorded
  reference-frame discrepancy (`build_cost_volume` stores slice `k` at the
  *right* image's column while GT is left-referenced) affects the intercept;
  `S` is intercept-free, but the **sign** must be pinned, not assumed.
* **`HS-Z`** — measure, do not derive, that a zone-interior pixel's right
  features depend **only** on zone content, and that the interior is non-empty
  after the 318 px exclusion. This is HS-BAND's question in the easy direction.
* **`HS-SLACK`** — verify every `Δ ∈ [−6,+6]` crop is fully inside the frame for
  every scene used: no missing data, no fill, no border.
* **`HS-SYM`** — assert numerically, on random tensors with no model, that `S` is
  **not** invariant under the intervention's symmetry group. The intervention is
  local, so no global `(u,d)` reparametrisation of the GEOM-002 kind exists;
  assert it rather than assert it away.
* **`HS-POSCTRL`** — the global positive control must show `β ≈ −1` before any
  zone result is interpreted.

**Stopping rule:** any hard stop raises, is recorded, and halts the process. The
experiment is never repaired and continued. If `HS-Z` shows an empty interior,
or `HS-POSCTRL` fails, the result is **inconclusive** and the pipeline question
closes as not identifiable with available interventions.

---

## 11. Interpretation, fixed in advance

* **Positive** (`S ≈ −1` in the treatment, `S ≈ 0` in the swap control and in
  `shift="none"`, positive control passing, in ≥ a preregistered rate of units
  and all 3 checkpoints):
  raises the ceiling to **Level E — genuine correspondence / disparity search, at
  the PIPELINE level.** Explicitly **not** Level E for the aggregation module,
  which remains structurally uncreditable because `L − R` is frozen upstream.
  Level F (natural-scene correspondence behaviour) would additionally require the
  effect on **held-out scenes**, which can be added at zero design cost by
  reserving scenes, and is *not* claimed by this experiment.
* **Negative** (`S ≈ 0` in the treatment while the global positive control
  passes): genuine **evidence against** local correspondence-driven selection in
  `disparity_initial` for these checkpoints — the first informative negative in
  this campaign. Ceiling stays at **D**; it does **not** become "no
  correspondence anywhere", because the refinement stage and the
  larger-than-zone spatial scales are untested.
* **Control failure** (swap control also tracks, or `shift="none"` tracks):
  the intervention is confounded, the result is withdrawn, and the pipeline
  question closes as not identifiable. No repair, no second variant.

---

## 12. Explicitly prohibited next experiments

O1 and any permutation variant · candidate permutations on translated self-pairs ·
any further synthetic global self-pair shift · CP-001 · any further
receptive-field-band search · any `S8` statistic or threshold · `max`-over-unit
gates · random weights presented as a mechanistic null · EPE/D1-only arguments ·
right-image corruption as standalone evidence · gradient-presence or
softmax-sharpness arguments · final-refinement accuracy as correspondence
evidence · Type-C direct `V` edits presented as pipeline evidence · any design
whose interpretation still turns on the defeated candidate-pointwise latch —
noting that at pipeline level that latch is *inside* `H_CORR`, so a design must
not be rejected merely for admitting it.

---

## 13. Decision

**EXECUTE `EXP-CORRESPONDENCE-ZONE-001`**, starting with the five model-free hard
stops and a preregistration written from their measured values. Nothing was
executed in this gate.

Ceiling until then: **Level D — genuine geometric correspondence
NOT-DEMONSTRATED.**
