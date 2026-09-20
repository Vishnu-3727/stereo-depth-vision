# NEXT-DIRECTION AUDIT — Phase-2 StereoNet correspondence investigation

**Record:** `phase2/diagnostics/correspondence/next_direction_audit_20260911/`
**Date:** 2026-09-11. **Status: AUDIT + DESIGN ONLY. NOTHING EXECUTED.**
No training, no inference, no model instantiated, no `StereoNet`/`Aggregation`
constructed, no experimental output generated. No `RESULTS.md` exists or may be
created under this record. No historical record modified. Phase 1 untouched.

Evidence tags: **MEASURED** (read directly from source or an existing record),
**DERIVED** (arithmetic/algebra on measured values), **INFERRED**
(interpretation), **UNKNOWN** (not determinable without a new experiment).

---

## 1. COMPLETE EVIDENCE LADDER

Statuses use ONLY: `ESTABLISHED / NOT-DEMONSTRATED / INCONCLUSIVE / INVALID`.

| LEVEL | CLAIM | STATUS | EVIDENCE | LIMITATION (what it does NOT prove) |
|---|---|---|---|---|
| A | right-image dependence | ESTABLISHED | MEASURED: EXP-007 — corrupting the right image costs up to +89.96 D1 points (EPE 1.44 → up to 69.4 px); SHIFT-001 Stage A — the degenerate `shift="none"` checkpoint responds to right-image translation by up to 0.70 candidates. | Does NOT prove matching or geometry. EXP-010 already showed large right-image dependence coexisting with a provably degenerate (search-free) cost volume. Dependence ≠ correspondence. |
| B | binocular / matching-path dependence | ESTABLISHED | MEASURED: EXP-007 (right matters); EXP-010 (the degenerate volume is still `left − right`, a genuine two-image photometric cue at one fixed disparity); H1 substitution (replacing `disparity_initial` moves final EPE by tens of px). The cost-volume→aggregation→readout path is causally load-bearing. | Does NOT prove the path searches over candidates or selects by geometric fit. A fixed-disparity differencing path satisfies all of the above. |
| C | candidate-coordinate sensitivity | INCONCLUSIVE | Ordered re-indexing response `d(+2)>d(+1)>d(0)>d(−1)` at 12/12 units, three seeds, reproduced bit-exactly across two independent executions and a third (INDEX-002) run; degenerate control bit-exactly invariant (0.0) under every permutation (6160 arms in INDEX-002, 816 in ARCH-001 incl. random weights). BUT: single-draw random controls failed in opposite directions (RUN-A separated, RUN-B indistinguishable); empirical-null magnitude test failed 0/12 (INDEX-002); ordering is architecturally expected for arbitrary weights (§3); shape-statistic architecture control had ~no power (§4, ARCH-001 §6). Provenance audit: NO-CONFIRMATORY-EVIDENCE either way. | Does NOT establish coordinate tracking: the decisive criterion was never both preregistered and decidable in any run; spatial-structure half never measured; magnitude and ordering dissociate (ordering passes 12/12 while magnitude is null-typical). |
| D | geometric correspondence | NOT-DEMONSTRATED | SHIFT-001 (the only geometric test) halted at Stage A; no positive-checkpoint translation number exists in any record and none may be quoted. No other record converts candidate indices to physical disparity. | Nothing at D has been tested on non-degenerate weights. The slope anchor −1/16 cand/px is DERIVED from source but never confronted with trained-model data. |
| E | genuine disparity search / correct candidate selection | NOT-DEMONSTRATED | No record scores argmin/soft-argmin candidate selection against GT/16 on non-degenerate weights. | Ordered readout movement under artificial re-indexing (C) would not imply correct selection on natural inputs even if C were established. |
| F | final stereo correctness (as correspondence-derived accuracy) | NOT-DEMONSTRATED | Accuracy numbers exist (Phase-1 baseline EPE ≈1.3–1.4 px, D1 ≈8–9%; Phase-2 `shift="left"`+standardised models train to EPE ≈2.3 px), but the baseline's accuracy is achieved with NO search (EXP-010), and no record attributes any accuracy figure to a demonstrated correspondence mechanism. | Accuracy does NOT evidence correspondence: the frozen baseline proves high accuracy without search. F-as-a-number is measured; F-as-correspondence is undemonstrated. |

### Constituent verdicts carried forward unchanged

- SHIFT-001: `DIAGNOSTIC-INVALID-AS-PREREGISTERED — STOP, NEW PREREGISTRATION REQUIRED`.
- INDEX-001 RUN-A (`…164346Z`): `ESTABLISHED` → downgraded by provenance audit to EXPLORATORY-ONLY.
- INDEX-001 RUN-B (`…164326Z`): `NOT-DEMONSTRATED` → EXPLORATORY-ONLY.
- INDEX-002: `CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED` (stands).
- INDEX-003 design audit: `DESIGN-INCONCLUSIVE` (stands).
- ARCH-001: `CASE-A-ARCHITECTURE-INDUCED` **with recorded near-zero power** (stands; must not be read as "weights don't matter").
- Overall Level C (provenance audit): `NO-CONFIRMATORY-EVIDENCE`.

---

## 2. LEVEL-C IDENTIFIABILITY VERDICT

```
LEVEL-C-NOT-CLEANLY-IDENTIFIABLE
```

**Level C is not salvageable as a confirmatory prerequisite.** Three independent
reasons, any one sufficient:

1. **No predictive statistic remains.** Every statistic confined to the four
   ordered arms `{−1,0,+1,+2}` has a value already published (12/12 strictly
   increasing, magnitudes 1.1–3.1 candidates). Per the INDEX-003 audit §1.2, such
   a statistic can at best *calibrate an already-measured quantity*, never
   predict. The only unmeasured Level-C quantities are the offsets `m∈{3…10}`
   (since measured for trained weights in ARCH-001, now also public) and
   amplitude on unseen scenes — and amplitude was selected post-hoc (ARCH-001
   §6.3), so promoting it on the same checkpoints/scenes is forbidden.
2. **Level C is not necessary** (§3 below): a genuine correspondence mechanism in
   this architecture could be weak, distributed, or shape-coded in ways the
   position-tracking statistic cannot see.
3. **Level C is not sufficient** (§3 below): a degenerate, search-free model
   already produces structured, input-dependent candidate curves and large
   ordered readout movements; approximate equivariance gives ordering for
   arbitrary weights.

Continuing to invent Level-C statistics would violate the task rule against
rescuing a dead hypothesis to preserve the original plan.

---

## 3. ARCHITECTURAL ARTIFACT ANALYSIS (from source, no new execution)

### 3.1 The computational graph (MEASURED)

`src/models/stereonet/stereonet.py:98-111`:
`left,right (B,3,368,1232)` → shared `FeatureExtractor` → `(B,32,23,77)` each
(stride 16) → `CostVolume` (subtract, `shift="none"` default) → `(B,32,12,23,77)`
→ `Aggregation` → `(B,12,23,77)` → upsample cost to `(B,12,368,1232)` →
standardise per-pixel across the 12 candidates (Phase-2 `scaled_regression.py`,
zero learned parameters — verified from source) → `soft_argmin` → candidate
units 0…11 → `disparity_initial` → `Refinement(disparity_initial, left)` →
`+`, ReLU → final.

`aggregation.py`: 4× `Conv3d(32→32,3×3×3,padding=1)` + LeakyReLU(0.01), then
`Conv3d(32→1,3×3×3,padding=1)`. **Five** candidate-axis 3-tap convolutions;
receptive field along D: `1+5·2 = 11`; only outputs `k∈{5,6}` are padding-free.

### 3.2 What each mechanism can generate

1. **Candidate-axis translation/equivariance.** Convolution is equivariant by
   operator, not by weights (LeakyReLU commutes with permutations exactly).
   Interior relative error 0.00–0.07 at `m=1` with *untrained* weights
   (preflight H-2, synthetic, MEASURED). Generates: ordered, sign-consistent
   readout movement under cyclic shifts — for ARBITRARY weights. Cannot by
   itself generate the wrap location, amplitude, or sharpness (weight-dependent).
2. **Zero-padding boundary effects.** Five stacked padded layers propagate the
   D-boundary up to 5 candidates inward from each end; with D=12 essentially the
   whole volume is boundary-affected. Generates (MEASURED, Stage A): a strongly
   non-constant candidate curve (interior spread 9.49e12, per-pixel SD 5.39e12)
   from a bit-exactly degenerate volume, mean max softmax weight 0.307 (vs 0.083
   uniform), soft-argmin 3.90±1.96 instead of 5.5.
3. **Soft-argmin.** Linear index grid 0…11, not circular: under circular input
   shift the response is a *wrapped ramp*, not monotone (ARCH-001 design §5,
   measured orbit: peaks at m=2…7, minimum at m=10 in 12/12 units). Generates:
   non-monotonicity, wrap crashes, and scale compression toward the grid mean.
4. **Standardisation.** Commutes exactly with candidate permutation (multiset
   preserved); rescales arbitrary magnitudes to unit variance. Generates:
   amplification of near-flat random orbits into arbitrary z-scored shapes
   (ARCH-001 §6.1: RR interval 97.4% of [−1,+1]) — i.e. it *destroys* amplitude
   information the scale-free statistic then cannot use, and makes flat noise
   look structured after z-scoring.
5. **Arbitrary learned weights.** Can set cost-curve peakedness/phase (orbit
   amplitude 5.4–6.4 trained vs 0.09–1.42 random — MEASURED but post-hoc, see
   §6.3 caveat), channel mixing, and spatial pooling. Generates: any smooth
   candidate-axis profile, with or without correspondence.
6. **Degenerate shift-none cost volume.** `reference_shift` pads right and slices
   `[0:W]` = identity (cost_volume.py:45-55); all 12 slices bit-identical
   (EXP-010, re-confirmed 0.0 in every later gate). Generates: the algebraic
   permutation-invariance control — the only control in the programme that holds
   by construction rather than expectation.

### 3.3 Separation

| mechanism class | signature | example evidence |
|---|---|---|
| architecture-induced | ordered movement, V-shaped translation magnitude, floor-pinned TV, wrap crashes | Stage A V-response; preflight equivariance; ARCH-001 orbit shape |
| weight-induced | amplitude/sharpness/phase of the candidate curve | trained vs random amplitude gap (post-hoc, not evidence) |
| input/cost-volume-induced | content dependence of the (identical) slice, border fill, feature correlation | Stage A input-dependence (3.65–4.27 across scenes); translation magnitude |
| genuinely correspondence-dependent | signed monotone slope ≈−1/16, H≫V separation, GT-tracking selection | NOT YET OBSERVED (Stage C never ran) |

### 3.4 Level-C premise: necessity and sufficiency

- **Not sufficient.** A search-free degenerate model shows structured
  input-dependent candidate curves, 0.70-candidate translation responses, and
  (via equivariance) ordered re-indexing responses with arbitrary weights.
  Strong coordinate response without correspondence is demonstrated possible.
- **Not necessary.** The graph permits correspondence with weak/distributed
  position-tracking: (a) matching evidence is 32-channel and spatially pooled
  (3×3×3 spans H,W too) — selection could be encoded in curve *shape/width*
  rather than peak *position*; (b) only 2 of 12 outputs are padding-free, so a
  learned solution may avoid position-coding at contaminated indices; (c) the
  refinement stage (left-guided, 76.5% of final magnitude) can re-express
  upstream evidence non-positionally; (d) standardisation+softmax read *relative*
  profiles, so absolute coordinate gain is free to be small while selection is
  correct.
- **Therefore Level C is neither necessary nor sufficient. It is not a valid
  gate for geometry.**

---

## 4. WHY PREVIOUS DIAGNOSTICS FAILED OR WERE LIMITED

**A. Right-image translation (SHIFT-001).** Intervention: translate right-image
content horizontally/vertically (±16,±32 px), left fixed; held fixed:
preprocessing, mask definition, code path both axes. Changed variable actually
tested: whole-image content + borders + features, not just geometry. Null
(analytic: uniform softmax → 5.5, zero response) was FALSE because the anchor
chain skipped the aggregation stage: five padded D-axis convolutions shape the
degenerate volume's identical slices into a non-uniform input-dependent curve.
Alternative mechanism producing the result: D-boundary + content effects
(MEASURED). Statistic (response magnitude / OLS slope) non-discriminative:
degenerate model gives V-shaped magnitude 0.70 with near-zero signed slope
(−0.0017 = 2.7% of anchor) and vertical > horizontal. Max defensible claim: the
analytic anchor is wrong; magnitude alone cannot evidence correspondence.
Stage-A negative control correctly invalidated the preregistration; positives
never run.

**B. Candidate-axis re-indexing.**
- Single random permutations (INDEX-001 both runs): one uncalibrated draw, no
  null distribution; GT-weighted band bias ≈+4.2 for BOTH perms yet responses of
  opposite sign (−0.20 vs +1.98) — control has no demonstrated discriminative
  power either way. RUN-A: verdict hand-written in prose, no decidable rule in
  code, unverifiable control (no generator), fixed point at index 9, "deviations:
  none" contradicted. RUN-B: reproducible control, but decisive `0.5·|Δ+1|`
  threshold exists only in the harness (written 5.2 s after RUN-A's verdict),
  not the preregistration; spatial-structure half never measured. Both
  EXPLORATORY-ONLY.
- INDEX-002 empirical triples: 1536 uniform `S_12` draws grouped into 512
  unrelated triples. Structurally mismatched to the ordered object (powers of one
  12-cycle generator; cycle types [12]/[6,6]/[1¹²] matched at no offset).
  Statistic `S_order=[d(+2)−d(id)]+[d(+1)−d(−1)]` tests magnitude+direction, not
  ordering-across-arms; ordered values (+4.1…+6.8) typical of null (p99
  +5.0…+6.9), ranks 479–508/513. Correct verdict NOT-DEMONSTRATED, but the null
  cannot test ordering by construction.
- Structurally matched generator null (INDEX-003 audit): exists (uniform
  12-cycle generators, powers γᵐ, TV statistic, M=1000) and is non-circular —
  but unusable without an architecture control, because ordering is operator-
  not weight-dependent, and the 4-arm ordered values are already public.
- Four-arm ordering now public (12/12, three independent reproductions,
  bit-exact across runs); TV floor-pinned (trained 22–32 vs random 22–62);
  orbit-shape z-score comparison powerless because random orbits are near-flat
  (range 0.09–1.42, median 0.59) so z-scored correlations fill [−1,+1]
  (RR width 97.4% of attainable).

**C. Architecture random-weight control (ARCH-001).** Intervention: same
tensor-object volume, trained vs 16 random aggregation weight-sets, full
12-orbit; held fixed: features, volume, readout, mask, scenes. CASE A
(INSIDE 12/12) recorded — but the scale-free primary statistic was
pre-declared and nearly unfalsifiable (containment in an interval covering
97.4% of [−1,+1]). **CASE A cannot be interpreted as "trained weights are
irrelevant"**: the statistic discarded amplitude, the dimension that separates
completely (trained 5.36–6.39 vs random max 1.42, 0/192 overlap). That
separation is MEASURED but post-hoc and must not be promoted to evidence.
Weight-dependence remains UNKNOWN. The `CASE A` label reflects the frozen
statistic, not the truth about the model.

---

## 5. CANDIDATE NEXT DIAGNOSTICS CONSIDERED

| # | candidate | causal assessment | disposition |
|---|---|---|---|
| 1 | New Level-C ordering stat (TV/Spearman on 4 arms, fresh null) | Ordered values public → no predictive content; ordering operator-expected → near-guaranteed pass, vacuous | REJECT (§VI) |
| 2 | Amplitude-based Level-C (trained vs random orbit range) | Separates completely on current data BUT selected post-hoc; same checkpoints/scenes now public → cannot preregister predictively; amplitude alone ≠ coordinate tracking (gain confound) | REJECT as confirmatory; recorded as exploratory observation |
| 3 | Level-C on held-out scenes only | Partially restores predictivity, but checkpoints shared (leakage), costs ≥ ARCH-001 scale (~30 min for M=1000 generator null), and still tests a non-necessary property | REJECT (dominated by 6: cheaper, direct) |
| 4 | Synthetic correspondence pairs (gratings/planes) | Precise geometry, matched statistics possible; but OOD for KITTI-trained features — failure uninterpretable (correspondence vs domain gap); claim ceiling low | REJECT as primary; possible follow-up only |
| 5 | Mismatched-scene pairs (right from other scene) as primary | EPE/correlation collapse predicted by BOTH geometry and generic dependence → predictions overlap substantially | REJECT as primary; keep as secondary |
| 6 | **Controlled horizontal translation with signed-slope prediction + vertical comparator + degenerate empirical null (SELECTED)** | Quantitative preregisterable slope −1/16 DERIVED from source; degenerate artifact baseline MEASURED (V-shaped, OLS≈0, V≥H); generic dependence predicts magnitude not signed slope; intercept-free (frame-ambiguity-proof); inference-only seconds | **ACCEPT — cheapest discriminative test** |
| 7 | Candidate-index/disparity remapping as geometry | Assumes the candidate↔pixel mapping under test; circular for a correspondence claim | REJECT |
| 8 | Any retraining design | No inference design proven impossible; forbidden by cheapness rule unless justified | REJECT (not justified) |

---

## 6. CAUSAL IDENTIFIABILITY TABLE (selected design vs confounds)

| hypothesis | predicted primary outcome (median H-slope β_H, monotonicity, H-vs-V) |
|---|---|
| genuine correspondence (`shift="left"` trained) | β_H < 0, monotone decreasing, \|β_H\| ≫ \|β_V\|, closer to −1/16 than to degenerate −0.0017 (i.e. β_H < −0.0321, DERIVED §8) |
| generic binocular dependence | V-shaped magnitude, OLS ≈ 0, non-monotone, V ≈ or > H (as MEASURED in Stage A) |
| architectural artifact (D-conv/padding/softmax/standardisation) | bounded by degenerate measurement: signed OLS ≤ ~3% of anchor, non-monotone, no H/V separation |
| no correspondence (null) | β_H ≈ β_H_degenerate ≈ 0; monotone/axis criteria fail |

Positive and artifact predictions overlap negligibly (3% vs ≥51% of anchor
magnitude with opposite monotonicity structure). Mismatch-pair EPE collapse is
secondary/descriptive precisely because its predictions DO overlap.

---

## 7. CHEAPEST VALID NEXT EXPERIMENT

**EXP-CORRESPONDENCE-GEOM-001 — signed geometric translation test.**
Full preregistration: `design.json` (this record). Summary:

- **Checkpoints:** 3 positive (`shift="left"`+standardised 6-block seeds 0,1,2 —
  same authoritative checkpoints as INDEX/ARCH) + 1 negative (trained
  `shift="none"`+standardised degenerate). Read-only, hash-verified.
- **Scenes:** `FOCUS_SCENES=[27,0,31,6]`, `hailo_val` (continuity; trained slopes
  never measured here → predictive content intact).
- **Intervention:** translate RIGHT image only, zero-fill, identical code path
  both axes: Δx,Δy ∈ {−32,−16,0,+16,+32} px. Left fixed. 9 conditions.
- **Endpoint:** `disparity_initial` ONLY (candidate units). Refinement never
  invoked. No EPE/D1 in primary.
- **Mask (frozen, GT-derived, identical pixel set everywhere):** GT valid ∧
  GT/16∈[2,8] ∧ 32-px border exclusion all sides. Computed once from GT, held
  fixed across checkpoints and conditions.
- **Geometry mapping (source-derived, §8):** `k = d − Δ/16`, slope −1/16
  cand/px, intercept-free → immune to the left/right reference-frame offset.
- **Primary statistic:** per-unit OLS slope of masked-median `disparity_initial`
  vs Δ (horizontal β_H, vertical β_V); global = median over 12 units.
- **Null:** empirical — degenerate checkpoint's MEASURED slopes (β_H=−0.0017,
  V-shaped, V≥H) + vertical comparator arms. No p-values (exchangeability
  unjustified: 3 weight-sets × 4 scenes); descriptive paired pattern rule (§10).
- **Compute:** 16 units × 9 conditions = 144 aggregation/readout passes, no
  refinement, no training. At measured throughput (≈0.007–0.011 s/arm): **tens
  of seconds**. Evaluator: existing `phase2/viz` + INDEX/SHIFT harness pattern
  reused inside the new record; no parallel framework.
- **Claim ceiling:** correspondence-consistent disparity search at candidate
  resolution on `disparity_initial` — nothing about final disparity, refinement,
  sub-candidate precision, occlusions, block counts, or stereo correctness.

---

## 8. EXACT PRIMARY STATISTIC

For unit `u` = (checkpoint, scene), condition Δ:

```
d_u(Δ) = median( disparity_initial[mask_u] )          # candidate units
β_H_u  = OLS slope of d_u vs Δx over 5 horizontal points (incl. Δ=0)
β_V_u  = OLS slope of d_u vs Δy over vertical points + shared Δ=0
β_H    = median_u β_H_u          # global primary (12 positive units)
β_V    = median_u β_V_u
mono_u = d_u strictly decreasing across Δx = −32…+32
sep_u  = |β_H_u| > |β_V_u|
```

Median-based primary (robust; INDEX-line convention). Mean-based slopes
secondary. Spearman ρ(d_u,Δx) reported descriptively per unit.

---

## 9. EXACT NULL

Empirical, frozen before execution:

- **N1 — degenerate artifact baseline (MEASURED, SHIFT-001 Stage A):**
  β_H_degenerate = −0.001713 cand/px (2.7% of anchor), V-shaped non-monotone,
  vertical slope +0.004769 ≥ horizontal in magnitude.
- **N2 — vertical comparator (measured in-run):** same code path, no
  horizontal-disparity information; correspondence predicts no signed slope here.
- No distributional null, no permutation null, no p-value: units share weights
  and scenes, so exchangeability is unjustified (§XII).

---

## 10. EXACT DECISION RULE (pattern-based, frozen)

```
C1  sign:        β_H < 0
C2  monotonicity: mono_u at ≥9/12 positive units
C3  axis:         sep_u at ≥9/12 positive units
C4  artifact separation: |β_H − (−1/16)| < |β_H − (−0.001713)|
      ⟺  β_H < −0.0321  (≈51% of anchor with correct sign; DERIVED, not fitted)
```

- **Pattern C (correspondence-consistent):** C1∧C2∧C3∧C4 → readout exhibits
  correspondence-consistent disparity search at candidate resolution.
- **Pattern B (sensitivity with geometry uncalibrated):** C1∧C2∧C3 ∧ ¬C4 →
  correspondence-shaped response; calibrated candidate selection NOT established.
- **Pattern A (no correspondence evidence):** otherwise.
- 9/12 (75%) supermajority frozen to avoid brittle 12/12 rules; thresholds use
  only source derivation + the already-measured degenerate baseline, never
  trained translation data (which does not exist).

---

## 11. EXACT CLAIM CEILING

If Pattern C: *"`disparity_initial` exhibits correspondence-consistent disparity
search at candidate resolution under controlled horizontal translation."*
Explicitly NOT claimable: final-disparity correctness, refinement
correspondence, sub-candidate precision, occlusion/textureless/repetitive
behaviour, global algorithm correctness, block-count superiority, causality of
parameter counts, latency, transfer. Same ceiling as SHIFT-001 §13.

---

## 12. EXPECTED OUTCOMES UNDER EACH HYPOTHESIS

- **Genuine correspondence:** C1–C4 pass; β_H ≈ −0.06 (≈100% of anchor; boundary
  distortion allows 51–150%); monotone; H≫V; degenerate stays V-shaped ≈0.
- **Generic binocular dependence:** magnitude without sign (V-shape, OLS≈0,
  C1/C2 fail, V≈H) — replicates Stage A on non-degenerate weights.
- **Architectural artifact:** signed component bounded by N1 (≤~3% of anchor);
  C4 fails by an order of magnitude; no H/V separation.
- **No correspondence:** β_H ≈ −0.0017 region; C2/C3 fail; Pattern A.

---

## 13. COMPUTE ESTIMATE

144 aggregation/readout arms + 16 cached volume builds ×(left once + 9 right
variants); refinement never run; no training. **≈20–60 s wall clock** (from
SHIFT-001: 36 passes/4.6 s; INDEX-002: 0.0107 s/arm). Seconds-to-minutes target
met. No retraining required; inference provably sufficient (question is about
frozen-weight behaviour).

---

## 14. PROVENANCE PLAN

Execution (NOT authorised here) must create
`phase2/diagnostics/correspondence/<UTC_TIMESTAMP>/` containing
`PREREGISTRATION.md` (the `design.json` of this record rendered verbatim, plus
frozen checkpoint hashes, scene IDs, mask counts), `results.json`,
`ENVIRONMENT.txt`, `RELATED_RUNS.md`, run log, and harness snapshot with sha256
(INDEX-002 freeze order: artefacts hashed → preregistration → execution).
`git diff phase-1-frozen -- src scripts` must be empty at execution. This audit
record gains no `RESULTS.md`, ever. New scenes (if ever needed for Level-C
follow-ups) must be declared before loading.

---

## 15. FINAL DECISION

```
BYPASS-LEVEL-C
```

Level C is not cleanly identifiable and is neither necessary nor sufficient for
correspondence in this architecture. A direct geometric diagnostic exists that
does not assume Level C, is source-grounded, has a measured (not analytic)
artifact baseline, and costs seconds. Neither Option 1 (no predictive Level-C
statistic exists), Option 3 (a discriminative design DOES exist), nor Option 4
applies.

---

## 16. EXACT ANSWER

**The single scientifically justified next experiment is
EXP-CORRESPONDENCE-GEOM-001: the signed horizontal-translation slope test on
`disparity_initial` for the three frozen `shift="left"` 6-block checkpoints
against the frozen degenerate `shift="none"` empirical null and an in-run
vertical comparator — inference-only (≈144 arms, tens of seconds),
preregistered exactly as `design.json` in this record, and NOT executed here.**

---

## APPENDIX A. SOURCE-DERIVED GEOMETRY MAPPING (required by §VIII)

From `src/models/stereonet/cost_volume.py` (`shift_left`: `F.pad(x,(0,k))[…,k:]`
→ `shifted[u]=left[u+k]`; `build_cost_volume` stacks `shifted−right`):

- `shift="left"` slice k at output column u: `V_k[u] = left_feat[u+k] − right_feat[u]`
  — stored at the **right** image's column u (right-referenced); GT/mask are
  left-referenced. Offset = disparity → affects **intercept only**.
- Feature stride 16 (`stereonet.py: feature_stride`), so candidate k ⟺ disparity
  16k full-resolution px; `disparity_initial` (soft-argmin over 0…11,
  `regression.py:soft_argmin`) is in **candidate units**, comparable to GT/16.
- Perturbation `right′[y,x]=right[y,x−Δ]` (content right by Δ):
  `right′_feat[u] ≈ right_feat[u−δ]`, `δ=Δ/16`. Baseline match at right-column v:
  `left_feat[v+d]≈right_feat[v]`; with `v=u−δ`: `left_feat[u−δ+d]≈right′_feat[u]`
  → matching slice `k=d−δ=d−Δ/16`. **Expected slope −1/16=−0.0625 cand/px**,
  independent of storage frame (frame offset cancels in differences/slopes).
- `shift="none"` (`reference_shift`): pad-right + slice `[0:W]` = identity →
  `V_k=D ∀k`; mapping undefined — correctly so, since no search exists there.

## APPENDIX B. STATISTICAL DISCIPLINE COMPLIANCE

Statistic, null, unit of analysis (12 = 3 checkpoints × 4 scenes, dependent),
and pattern rule frozen before execution; shared weights/checkpoints accounted
(no independence assumed, no Fisher/Stouffer/Bonferroni, no p-values);
descriptive paired comparisons; finite-resolution issues absent (no permutation
null); no statistic chosen from preliminary trained-translation data (none
exists); GT-derived mask identical across arms (counts auditable as in
INDEX-001/002: 33026/42676/37892/62908).
