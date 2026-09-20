# H1 visual mechanism investigation V1 — where does the working cost volume act?

**Scope.** Forensic investigation of the two recorded H1-v2 checkpoints. No
training, no seed replication, no architecture change, no Stage E. Phase 1 is
untouched (`phase-1-frozen` = `b4207e5`, verified: `git diff phase-1-frozen --
src scripts experiments docs` is empty). The registered H1-v2 experiment records
are unmodified; nothing here replaces them.

**Claim tags used throughout:** `MEASURED` (a number this investigation
computed, with its protocol), `DERIVED` (arithmetic on measured numbers),
`OBSERVED VISUALLY` (read off an image, with the image named), `INFERRED` (a
reading of the evidence that the evidence does not force), `UNKNOWN`.

**Evidence files.** `phase2/results/h1_mechanism/*.json` — `stages`, `spatial`,
`ablation`, `intervene`, `pooled`, `cross_shift`, `gradients`,
`saturation`, `tie_check`, `weight_drift`, `feature_matching`; see that
directory's `README.md` for the command that regenerates each one. Figures:
`phase2/visualizations/h1_mechanism/*_{stages,d1_flips}.png` (git-ignored,
regenerable). Tooling: `phase2/scripts/h1_mechanism_probe.py`, plus two additive
figure functions in `phase2/viz/render.py`.

**Protocol note, stated once.** Every number below except where explicitly
labelled "recorded" comes from a probe protocol defined here: inference on full
368x1232 `hailo_val` frames, pooled over `gt > 0` pixels, or single-sample
256x512 crops from `hailo_calib` for gradient work. The registered H1-v2
validation protocol (10 scenes, during training, every 5 epochs) is a different
population; the two are never mixed and the recorded result is not restated as
if this investigation reproduced it.

---

## 0. Preconditions

`MEASURED` — the two checkpoints are the same network with the same parameter
names, shapes and dtypes (72 tensors, all float32); no tensor is bit-identical
between them (they are separate training runs); the only differing key in the
stored training config is `cost_volume_shift` (`none` vs `left`). Nothing else
about the arms differs.

`MEASURED` — scene selection is the pre-existing exploratory ranking
(`phase2/visualizations/scene_ranking_hailo_val.json`): scene 27 (`000187_10`,
WORKING best by 7.54 pt), scene 0 (`000160_10`, typical WORKING gain 4.23 pt),
scene 31 (`000191_10`, near-tie 0.20 pt), scene 6 (`000166_10`, WORKING worst by
6.19 pt). Improvement and degradation examples are inspected in equal measure.

---

## 1. What each stage actually does

Tensor shapes, verified not assumed: features `(1,32,23,77)`; cost volume
`(1,32,12,23,77)`; aggregated cost `(1,12,23,77)`; `disparity_initial`,
`refinement_residual`, `disparity_final` `(1,1,368,1232)`.

`MEASURED` (`stages.json`, four focus scenes):

| quantity | BASE (`shift=none`) | WORKING (`shift=left`) |
|---|---|---|
| max abs deviation of any disparity slice from slice 0 | **exactly 0.0** | 1.7e14 – 2.6e14 |
| max abs value in the cost volume | 4.8e8 – 6.5e8 | 1.7e14 – 2.6e14 |
| max abs value in the aggregated cost | 5.2e12 – 8.6e12 | 3.9e19 – 6.2e19 |
| non-finite values anywhere | none | none |
| mean softmax entropy (max possible 2.485 nats) | **0.0000** | **0.0000** |
| mean max softmax weight | **1.0000** | **1.0000** |
| `disparity_initial` std | **0.0000** | 3.87 – 4.32 |
| `disparity_initial` value | **constant 11.0 everywhere** | varies, 0..11, mean 2.4–4.3 |
| `disparity_initial` correlation with GT | undefined (constant) | +0.28, +0.05, −0.00, +0.24 |
| `disparity_initial` correlation with the final map | undefined | +0.0003, +0.07, +0.03, −0.07 |
| refinement residual correlation with the final map | **+1.0000** | +0.91 – +0.97 |
| refinement residual correlation with GT | +0.72 – +0.85 | +0.53 – +0.79 |

`MEASURED` — the reason the softmax is saturated: the gap between the smallest
and second-smallest aggregated cost per pixel has median 3.4e11 – 8.2e11 (BASE)
and 8.0e15 – 9.8e16 (WORKING), minimum 9.3e7 (BASE) / 1.2e11 (WORKING) over the
inspected frames. `DERIVED` — fp32 `exp()` underflows to exactly zero below
about −88, so at every pixel of these frames the soft-argmin's weights are
exactly `{1, 0, …, 0}`: it is a hard argmin, and its derivative with respect to
the cost is exactly zero.

`DERIVED` — for BASE the argmin index is 11 at all 1771 low-resolution
positions of all four scenes, so `disparity_initial` is the constant 11.0 and the
matching branch contributes no spatial information whatsoever. The only path
from the cost volume to the output is `disparity_initial`, so **for BASE the
entire stereo branch is disconnected from the answer**.

`OBSERVED VISUALLY` (`*_stages.png`) — BASE's `disparity_initial` panel is a
flat field (the figure labels it "CONSTANT 11.000, no spatial information").
WORKING's is a coarse, blocky map of a few large blobs at 1/16 resolution,
bearing no visible relation to scene structure. Both arms' final disparity maps
look like plausible monocular depth: sky far, road near, buildings graded.

---

### 1b. Was the matching branch ever trained, and can its features match?

`MEASURED` (`weight_drift.json`) — both arms seed identically
(`torch.manual_seed(0)` before model construction; the two shift settings
produce bit-identical initial weights, asserted in the probe). Relative distance
from that initialisation, `||w − w0|| / ||w0||`:

| stage | BASE | WORKING |
|---|---:|---:|
| feature extractor | 1.097 | 1.548 |
| aggregation | **1.919** | **2.500** |
| refinement | 1.560 | 1.619 |

`DERIVED` — the matching branch is **not** untrained. Both arms moved their
feature extractor and aggregation far from initialisation — aggregation most of
all — even though at convergence those stages receive no gradient (§5). The
zero-gradient state is where training *ended up*, not where it started.

`MEASURED` (`feature_matching.json`, four focus scenes) — a plain L1 cost volume
built directly from each arm's trained features, over the same 12 shifts,
bypassing aggregation and the soft-argmin entirely, gives an argmin map
correlating with ground truth at +0.016 / +0.229 / +0.237 / +0.177 (BASE, mean
+0.165) and +0.045 / +0.330 / +0.203 / +0.285 (WORKING, mean +0.215).

`DERIVED` — the trained features carry a weak but non-zero matching signal, and
BASE's carry nearly as much as WORKING's despite BASE's stereo branch being
disconnected from its output. `INFERRED` — the matching signal that exists is
being discarded downstream rather than never being formed; but +0.2 is weak in
absolute terms, so the features are not "ready" either.

## 2. Is the right image used at all?

`MEASURED` (`ablation.json`, Phase 1 EXP-007 variant set, 10 scenes, both arms):

| right image replaced by | BASE mean abs change | BASE D1 | WORKING mean abs change | WORKING D1 |
|---|---:|---:|---:|---:|
| (unmodified) | 0.000 px | 77.06 | 0.000 px | 76.02 |
| the left image | **0.000 px** | 77.06 | 0.369 px | 75.92 |
| another scene's right image | **0.000 px** | 77.06 | 1.065 px | 76.60 |
| horizontally flipped | **0.000 px** | 77.06 | 1.080 px | 76.75 |
| black | **0.000 px** | 77.06 | 0.876 px | **75.60** |
| uniform noise | **0.000 px** | 77.06 | 1.204 px | 76.81 |

`MEASURED` — BASE's output is bit-identical (correlation 1.000000, change
0.000 px) under every corruption of the right image. `DERIVED` — **BASE-v2 is a
monocular predictor**; the second camera has no influence on its output at all.

`MEASURED` — WORKING's output moves by about 1 px on a mean disparity of 26 px
when the right image is destroyed, and its D1 moves by at most 0.8 pt in either
direction; feeding a **black** right image gives its *best* D1 of the set
(75.60 vs 76.02 unmodified).

`MEASURED`, for contrast, from the frozen Phase 1 record EXP-007 (the Hailo
reference weights, same variants): corrupting the right image cost **+89.7 D1
points**. The reference model depends heavily on the second camera despite its
degenerate volume; our under-trained arms do not.

---

## 3. Is the matching output used by refinement? (intervention, not correlation)

The only channel from matching to output is `disparity_initial`. Substituting
it while holding the left image and all weights fixed measures the channel's
worth directly.

`MEASURED` (`pooled.json`, all 40 `hailo_val` scenes, pooled over
`gt > 0`):

| variant | EPE (px) | D1 (%) |
|---|---:|---:|
| BASE as trained | 11.357 | 78.69 |
| WORKING as trained | 11.173 | **77.93** |
| WORKING, `disparity_initial` := its own scalar mean | 11.318 | 78.23 |
| WORKING, `disparity_initial` := constant 11 (what BASE emits) | 10.540 | **77.19** |
| WORKING, `disparity_initial` := spatially shuffled | 13.448 | 84.07 |
| BASE, `disparity_initial` := WORKING's map | 22.875 | 92.91 |

`MEASURED` (`intervene.json`, per-scene means over 40 scenes) — for BASE,
replacing `disparity_initial` with a constant or with a shuffled copy of itself
changes the output by **exactly 0.0000 px** (it is already a constant). For
WORKING, flattening it to its own mean changes the output by 1.42 px on average.

`DERIVED` — decomposition of WORKING's 0.76 pt pooled D1 advantage: destroying
all spatial content of WORKING's matching output costs **0.30 pt** (77.93 →
78.23); the remaining **0.46 pt** advantage over BASE survives with a scalar in
place of the matching map, i.e. it lives in the trained feature/refinement
weights rather than in the matching signal at inference.

`MEASURED` — WORKING with a constant 11 fed to refinement scores **0.74 pt
better than WORKING as trained** and 1.50 pt better than BASE. Its own matching
output is worse than a well-chosen constant.

`MEASURED` — shuffling that map costs 6.14 pt, and feeding WORKING's map into
BASE's refinement costs 14.2 pt. `DERIVED` — refinement is *not* ignoring its
disparity input; it is strongly sensitive to it, and each arm's refinement is
tuned to the specific input distribution it was trained with. The matching map
is used, and is barely better than a constant.

`MEASURED` (`cross_shift.json`, all 40 scenes, weights unchanged, only the
inference-time shift flag switched):

| weights | shift at inference | EPE | D1 |
|---|---|---:|---:|
| BASE | none (as trained) | 11.357 | 78.69 |
| BASE | left | 11.551 | 78.75 |
| WORKING | none | 11.528 | 78.68 |
| WORKING | left (as trained) | 11.173 | 77.93 |

`MEASURED` — switching WORKING's shift off at inference removes its whole
advantage (77.93 → 78.68, level with BASE's 78.69); switching BASE's shift on
gives BASE nothing (+0.06 pt).

`MEASURED` — WORKING weights under `shift=none` do **not** collapse to a
constant `disparity_initial` (std 2.06–2.53 vs 3.87–4.32 with the shift on), so
the cross-shift result is not the same intervention as "replace the map with a
constant". A hypothesis that it was — that the shift acts purely by changing a
scalar level — was tested and **rejected**: the maximum difference between
`final(shift=none)` and `final(shift=left, initial := that scalar)` is 9.1–11.8
px, not zero.

---

## 4. Where do the two arms differ spatially?

`MEASURED` (`spatial.json`, all 40 scenes, 3,802,797 valid pixels; strata
defined only from ground truth and the left image, never from the predictions):

- Mean `|BASE − WORKING|` disparity difference on valid pixels: **3.396 px** —
  the arms disagree substantially **everywhere**, on a mean disparity of ~26 px.
- WORKING turns 367,804 BASE outliers into inliers (9.7 % of valid pixels) and
  turns 339,265 BASE inliers into outliers (8.9 %). **Net +28,539 pixels =
  −0.750 pt of D1.** The headline effect is a small residue of two large,
  opposing flows.
- Of BASE's most accurate pixels (error 0–2 px, 543,718 px), WORKING breaks
  **211,838 (39 %)**.

Stratified ΔD1 (negative favours WORKING):

| stratum | ΔD1 | note |
|---|---:|---|
| GT disparity 0–10 px (far) | **+1.97** | WORKING worse |
| GT disparity 10–20 px | −0.58 | |
| GT disparity 20–30 px | −1.37 | |
| GT disparity 30–45 px | **−1.52** | |
| GT disparity 45–200 px (near) | −0.21 | both arms ~95 % outliers here |
| local GT disparity range 0–0.5 px (flat) | −0.76 | |
| local GT disparity range 0.5–2 px | −0.74 | |
| local GT disparity range 2–5 px (discontinuity) | −0.98 | only 1,124 px |
| left-image gradient Q1 (flattest) | −0.88 | |
| left-image gradient Q4 (most textured) | −0.79 | |
| columns 0–64 (left border) | **+0.31** | WORKING worse |
| columns 64–128 | +0.24 | |
| columns 128–400 | −0.13 | |
| columns 400–800 | −0.65 | |
| columns 800–1232 (right side) | **−1.70** | largest single stratum effect |
| rows 92–184 (upper middle) | −1.19 | |
| rows 184–276 | −0.51 | |
| rows 276–368 (road, foreground) | −0.90 | |
| rows 0–92 (sky) | +0.00 | only 343 valid GT pixels |

`DERIVED` — the differences do **not** concentrate at disparity
discontinuities (−0.74 to −0.98 across the local-range bins, a 0.24 pt spread)
nor by texture (−0.64 to −0.88 across gradient quartiles). They do vary by image
column (+0.31 at the left border to −1.70 on the right side, a 2.0 pt spread)
and by ground-truth disparity (+1.97 for the farthest bin to −1.52 for the
30–45 px bin, a 3.5 pt spread).

**Not claimed:** that the working cost volume "improves matching at range" or
"fails at the left border". The strata show *where the two trained networks
disagree*, and both networks differ in every weight, not only in the shift.

`OBSERVED VISUALLY` (`*_d1_flips.png`) — fixed (green) and broken (red) pixels
are interleaved across the same regions in every scene, at ground-truth-sample
density. On scene 27 the parked car at mid-left is a coherent green cluster
(13,045 fixed vs 7,232 broken overall). On scene 6 the leading car is a coherent
**red** cluster (7,262 fixed vs 14,063 broken). The same structure type — a
nearby vehicle — flips in opposite directions in the two scenes.

Per focus scene (`spatial.json`):

| scene | mean abs difference | BASE D1 | WORKING D1 | ΔD1 | fixed | broken |
|---|---:|---:|---:|---:|---:|---:|
| 27 `000187_10` | 3.226 px | 71.66 | 64.12 | −7.54 | 13,045 | 7,232 |
| 0 `000160_10` | 2.882 px | 69.00 | 64.78 | −4.23 | 16,541 | 12,308 |
| 31 `000191_10` | 2.864 px | 78.88 | 78.68 | −0.20 | 7,365 | 7,203 |
| 6 `000166_10` | 3.413 px | 76.79 | 82.99 | +6.19 | 7,262 | 14,063 |

---

## 5. The gradient-norm difference

Recorded values, from the four registered experiments (`metrics.json`), quoted
not recomputed:

| run | median grad norm | **max** grad norm | non-finite |
|---|---:|---:|---:|
| `EXP-H1-BASE` (v1, 20 ep) | 31.795 | **1.302e+05** | 0 |
| `EXP-H1-WORKING` (v1, 20 ep) | 26.845 | 246.5 | 0 |
| `EXP-H1-BASE-v2` (200 ep) | 32.441 | 1.358e+04 | 0 |
| `EXP-H1-WORKING-v2` (200 ep) | 37.253 | **3.567e+13** | 0 |

`MEASURED` — in v1 the huge maximum was in **BASE**, with WORKING three orders
of magnitude smaller; in v2 the direction reverses. `DERIVED` — the spike is not
a systematic property of the shifted cost volume; it occurs in either arm.

`MEASURED` (`saturation.json`, `tie_check.json`; trained weights, 150 identical
256x512 crops per arm, forward+backward only, no optimizer step):

| | BASE | WORKING |
|---|---:|---:|
| median total gradient norm | 54.6 | 58.4 |
| max total gradient norm | 261.8 | **3.978e+12** |
| crops with norm > 1e4 | **0 / 150** | **5 / 150** |
| crops where the matching path (features + aggregation) receives *any* gradient | **0 / 150** | **5 / 150** |
| median norm on crops where it does | — | 7.99e+07 |
| median norm on crops where it does not | 54.6 | 57.3 |
| crops with gradient into the cost tensor | 0 / 150 | 5 / 150 |
| cost pixels receiving gradient, on those crops | — | **exactly 1 of 131,072** |
| top-2 cost gap at that pixel | — | **exactly 0.0 (a tie)** |

`MEASURED` — spikes and matching-path gradient coincide 5/5. `MEASURED` — on
every spike crop the gradient enters the upsampled cost tensor at exactly one
pixel, where two disparity candidates are exactly tied; every other pixel is
saturated to a zero derivative.

`MEASURED` — the boundary-padding idea was tested rather than assumed: sampling
crops anchored at the left edge, at the right edge, and at random interior
columns gives spikes in all three bands (2 / 4 / 2 of 72 crops each for
WORKING, 0 everywhere for BASE). `DERIVED` — no evidence that spikes are
specific to the shift's zero-padded border. The right-edge band has a higher
median norm in both arms (114 vs 59 for WORKING, 85 vs 53 for BASE), i.e. it is
not shift-specific.

`INFERRED` — a mechanism consistent with all of the above: the aggregated cost
grows to 1e12–1e19, so the soft-argmin is saturated and passes exactly zero
gradient on almost every batch; when bilinear upsampling produces an exact tie
at a single pixel, that pixel's derivative becomes non-zero, and the chain rule
multiplies it by the enormous cost scale, producing a 1e7–1e13 parameter
gradient. This is consistent with a rare, sample-specific, either-arm event, and
with `max_norm=1e9` clipping being ineffective. **Not established:** that the
tie *causes* the spike — the association is 5/5 in a 150-crop sample at the end
of training, and no intervention (e.g. removing the tie and re-measuring) was
run.

`UNKNOWN` — whether the specific v2 spike at 3.567e+13 arose this way; it was
recorded during training, at an unrecorded epoch, and was not reproduced here.

---

## 6. Answers to the questions asked

**A. Where does the working cost volume help?**
`MEASURED` — pooled over 40 scenes, WORKING is 0.76 D1 pt better; the gain is
spread over 367,804 fixed pixels covering the whole frame, largest in the right
third of the image (−1.70 pt), in the 20–45 px ground-truth disparity range
(−1.37 / −1.52 pt), and in the upper-middle rows (−1.19 pt). `OBSERVED
VISUALLY` — occasionally a whole object flips: the parked car in scene 27.

**B. Where does it hurt?**
`MEASURED` — 339,265 pixels are broken, including 39 % of BASE's 0–2 px-error
pixels. Strata where WORKING is worse: the farthest ground-truth disparities
(0–10 px, +1.97 pt) and the leftmost 128 columns (+0.24 / +0.31 pt). Whole
scenes go the other way (scene 6, +6.19 pt); `OBSERVED VISUALLY`, the leading
car there is a solid red cluster.

**C. How large are the spatial effects?**
`MEASURED` — the *disagreement* is large and global (mean 3.40 px, ~13 % of mean
disparity); the *net accuracy effect* is small (0.75 pt = 28,539 pixels net out
of 3.8 M). `DERIVED` — the arms are two substantially different predictors whose
error rates happen to be close, not one predictor with a localized correction.

**D. Do effects concentrate around particular scene structures?**
`MEASURED` — no concentration at disparity discontinuities (0.24 pt spread
across local-GT-range bins, and only 1,124 pixels available in the
discontinuity bin) and none by texture (0.24 pt spread across gradient
quartiles). There is column and range dependence (2.0 pt and 3.5 pt spreads).
Sky is unmeasurable here (343 valid GT pixels in the top quarter). `INFERRED` —
the effect looks like a global change of predictor, not a structure-specific
correction.

**E. Does the difference originate in matching, aggregation, soft-argmin, or
refinement?**
`MEASURED` — the soft-argmin is saturated in both arms (entropy 0.0000, top-2
gaps ≥ 9.3e7), so matching and aggregation reach the output only through a hard
argmin index map. `DERIVED` — for BASE that map is a constant, so its stereo
branch contributes nothing; for WORKING, destroying the map's spatial content
costs 0.30 pt of its 0.76 pt advantage, and the remaining 0.46 pt persists.
`INFERRED` — the difference originates *mostly in the trained weights of the
monocular path* (feature extractor and refinement, which took different training
trajectories), and *partly* in a low-information argmin map. `MEASURED` — the
matching stages of both arms did move far from initialisation (§1b), so this is
a statement about where the signal ends up, not about which stages were trained.

**F. Is the working cost volume actually being used meaningfully?**
`MEASURED` — no, on any reasonable reading of "meaningfully": its output
correlates with ground truth at +0.28 to −0.00; replacing it with a scalar costs
0.30 pt; replacing it with the constant 11 *improves* WORKING by 0.74 pt;
corrupting the right image entirely moves WORKING's D1 by at most 0.8 pt and in
one case improves it. `MEASURED` — for BASE the cost volume is provably unused:
the right image can be replaced by noise with bit-identical output. `MEASURED` —
this is not because the features cannot match: an L1 cost volume built by hand
from the same features gives an argmin correlating with ground truth at +0.165
(BASE) / +0.215 (WORKING), weak but non-zero (§1b).

**G. Is there evidence that the refinement network is compensating for weak
stereo matching?**
`MEASURED` — the refinement residual correlates with the final map at +1.0000
(BASE) and +0.91 to +0.97 (WORKING), and with ground truth at +0.72 to +0.85
(BASE) and +0.53 to +0.79 (WORKING); the matching output correlates with ground
truth at ≤ +0.28. `MEASURED` — BASE reaches D1 78.69 with a provably
disconnected stereo branch. `DERIVED` — the refinement network is producing
essentially all of the usable signal in both arms, from the left image alone.
This is consistent with Phase 1's compute finding (refinement = 90.6 % of MACs,
73.3 % of GPU time) but is a separate, independently measured fact.

**H. What could explain the huge gradient-norm difference?**
See §5. `MEASURED`: spikes occur in 5/150 crops for WORKING and 0/150 for BASE
at the end of training; they coincide exactly with the only crops where any
gradient reaches the matching path; on those crops gradient enters the cost
tensor at exactly one tied pixel out of 131,072. `MEASURED`: v1 had the same
phenomenon in the *other* arm, so it is not shift-specific. `MEASURED`: no
evidence of border specificity. `INFERRED`: saturation-gated gradient multiplied
by a 1e12–1e19 activation scale. `UNKNOWN`: causality, and the provenance of the
particular 3.567e+13 value.

**I. What experiment should be performed next?** See §8.

---

## 7. What this investigation did **not** establish

- Nothing here is a statistical claim. One seed per arm; no significance test
  was run on anything, and the checkpoint series were not touched.
- Four scenes were inspected visually and 40 scored numerically. `hailo_val` is
  40 scenes; no claim extends beyond it.
- Both arms are under-trained by design (160 scenes, random init). Every
  statement about "the model" is a statement about *these* checkpoints.
- The strata are proxies: local 3x3 ground-truth range is a weak discontinuity
  proxy under sparse LiDAR ground truth (1,124 pixels in the top bin), image
  gradient is a weak texture proxy, and row bands are a weak sky/road proxy
  (343 valid sky pixels).
- No causal test of the tie→spike link was run.
- Occlusion, repetitive texture and thin structures were **not** measured: this
  investigation has no occlusion mask (left-right consistency was not computed)
  and no repetition detector. Column bands are not an occlusion measurement.
  Reporting them as such would be a label without evidence.

---

## 8. Status and recommendation

    H1 MECHANISM STATUS: PARTIALLY SUPPORTED

Supported: the working cost volume does change the trained model's behaviour,
and switching the shift off at inference removes WORKING's entire measured D1
advantage (77.93 → 78.68 vs BASE 78.69).

Not supported: any account in which the advantage comes from *correspondence
information reaching the answer*. The soft-argmin is saturated to a hard argmin
with an exactly zero derivative, the matching branch receives literally no
gradient on 145/150 sampled crops (BASE: 150/150), BASE's stereo branch is
provably disconnected from its output, and WORKING's matching map is worth
0.30 pt of its 0.76 pt advantage — less than the 0.74 pt gained by replacing
that map with a constant.

    RECOMMENDED NEXT EXPERIMENT:
        EXP-H2-SOFTARGMIN-SCALE — one changed variable: normalise the
        aggregated cost before the soft-argmin (per-pixel standardisation
        across the 12 candidates, or an equivalent fixed temperature chosen so
        the median top-2 gap is O(1)), holding the H1-v2 recipe, seed, split,
        crop, schedule and cost_volume_shift="left" fixed. Compare against the
        existing EXP-H1-WORKING-v2 arm as its control.
        Primary endpoint (mechanistic, not accuracy): the fraction of training
        batches on which the feature extractor and aggregation receive non-zero
        gradient — currently 5/150 measured at convergence — together with the
        gradient-norm distribution.
        Secondary endpoint: validation EPE/D1 under the recorded protocol.

Why this and not seed replication or Stage E: seed replication would measure the
variance of a 0.6 pt effect whose mechanism this investigation shows is *not*
stereo matching; cost-volume optimisation (Stage E) would tune candidate counts
for a branch that receives no gradient in 97 % of batches and whose output the
network is better off ignoring. The saturation is the first thing in the chain
that makes the other questions unanswerable, and it is a one-line, one-variable
change with an existing control arm.

Design note from §1b, already measured: a hand-made L1 cost volume over the
trained features correlates with ground truth at only +0.165 (BASE) / +0.215
(WORKING), and both arms' matching stages moved far from initialisation before
saturating. So EXP-H2 must expect to *re-learn* features, not merely to unblock
a ready signal — its budget and its success criterion should say so, and the
mechanistic endpoint (gradient actually reaching the matching path) is the one
to judge it on, not a fraction of a D1 point.
