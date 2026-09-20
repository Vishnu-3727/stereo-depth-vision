# stage_a_diagnostics — STAGE-A frozen-model diagnostic (D0–D5)

READ-ONLY diagnostic on the frozen MICROCHIP STEREONET P2A checkpoints.
No model was changed, retrained, tuned, or reinterpreted here. No file
outside this directory was modified by this step (verified via `git status`:
only `stage_a_diagnostics/` is new from this step).

## What produced what

| step | script | outputs |
|---|---|---|
| D0 integrity + baseline reproduction | `scripts/d0_verify.py` | `integrity.json`, `baseline.json`, `raw/d0_preds_s{0,1,2}.npz` |
| D1 per-disparity-bin decomposition + gap attribution | `scripts/d1_bins.py` (reuses D0 raw dumps; scores the frozen reference ONNX itself) | `disparity_bins.json`, `raw/d1_reference.npz`, `raw/d1_bins.csv` |

| D2 cost-distribution / readout | `scripts/d2_cost.py` | `cost_distribution.json`, `raw/d2_hist_s{0,1,2}.npz`, `raw/d2_bins.csv` |

D3–D7 are PENDING and are not in this directory.

## D2 — cost-distribution / readout (measured, not inferred)

FROZEN-MODEL DIAGNOSTIC. No weight altered, nothing trained. Recomputing the
refinement with a different `disparity_initial` input is NOT a model change:
the UNCHANGED frozen `self.refinement` is reused as a fixed function with the
same normalized left image as guidance. No file outside this directory was
modified by this step (other `git status` entries pre-date it).

Correction to the step brief (verified in `src/models/stereonet/stereonet.py`):
`forward(..., return_stages=True)` returns a TUPLE `(disparity, stages-dict)`,
not a bare dict. The script unpacks the tuple and uses the flag; no hooks,
no model edits. The script's own readout replication (upsample bilinear
`align_corners=True` -> standardise across disparity axis -> `softmax(-cost)`
-> expectation) matches the model's `disparity_initial` to max-abs-diff
0.00e+00. `disparity_initial` is in candidate units (0..23); `disparity_final`
is in pixels. Pooled per-bin baseline EPEs match D1 to 0.00e+00 on all seeds.

How to re-run: `python stage_a_diagnostics/scripts/d2_cost.py`
(one image at a time; float64 accumulators; raises on NaN/inf/shape mismatch).

- Baseline reproduces exactly: YES. Seed-0 EPE 1.4149795737, diff 0.00e+00
  vs step 1 (tolerance 1e-6; STOP gate passed). Valid pixels 3802797 all seeds.
- Pooled distribution stats, method-mean across 3 seeds (seed0: entropy
  1.8390, top-1 0.5116, margin 0.3170): entropy 1.8531 nats, top-1 prob
  0.5051, top-1 margin 0.3099, variance 25.206 (candidate-index units),
  mean |argmax − soft-argmin| 2.0105 candidates.
- Range reachability: fraction of valid pixels with gt_cand = GT_px/8
  OUTSIDE [0,23] is 0.000000 on all seeds (D1 bins [160,184)/[184,inf) are
  empty, consistent).
- GT-candidate rank: fraction where the nearest-GT candidate IS the argmax
  (rank 1) is 0.1330 pooled (seed0 0.1539, seed1 0.1301, seed2 0.1150);
  mean mass at nearest-GT candidate 0.1786, mean mass in gt_cand ± 1
  neighbourhood 0.3969.
- Readout EPEs through the frozen refinement, pooled method-mean
  (deltas vs baseline): baseline 1.4410; argmax 10.7675 (+9.3265);
  top-2 oracle 8.5297 (+7.0887); top-3 oracle 6.6505 (+5.2095);
  top-4 oracle 5.9277 (+4.4867); continuous oracle (gt_cand clipped to
  [0,23]) 5.7061 (+4.2651).
- GT>=96 pixels specifically (34152 px, 0.898% of valid; method-mean):
  baseline 12.4149; argmax 15.0248; top-2 12.8533; top-3 11.3975;
  top-4 10.2852; continuous 6.4961/7.1509/7.6475 per seed (mean 7.0982).
  Rank-1 fraction there is 0.0745; out-of-range fraction 0.000000.
- Per-bin (seed 0): the continuous oracle scores below baseline only in far
  bins — [64,80) 1.989 vs 3.060, [112,128) 5.741 vs 13.535,
  [128,144) 15.825 vs 33.785 — and above baseline in every bin below 64 px
  (which hold ~99% of valid pixels), so the pooled oracle EPEs above follow.

Factual statement only (no verdict; step 8 does verdicts): no alternative
readout — hard argmax, top-2/3/4 oracle picks, or the exact continuous
gt_cand value — reduces pooled EPE when fed through the unchanged frozen
refinement; all score above the baseline they start from. Protocol property
(not a conclusion): each alternative map replaces `disparity_initial` only
at valid pixels (~21% of the image); invalid pixels keep the model's own
soft-argmin values, so the refinement sees spatially patchy inputs unlike
the full soft maps it processed during training.

Nothing in this step was unmeasurable; every D2 quantity was measured.

> NOTE (D2b correction): the D2 readout numbers above are CONFOUNDED
> (`readout_comparison_status` in `cost_distribution.json`) and are
> superseded by `cost_readout_corrected.json`. See the D2b section below.
> The D2 distribution statistics (entropy, top-1, margin, variance, rank,
> mass, in/out-of-range fractions) are sound and stand unchanged.

## D2b — corrected readout comparison (measured, not inferred)

FROZEN-MODEL DIAGNOSTIC. No weight altered, nothing trained, eval split
unchanged. Script: `scripts/d2b_readout.py` (one image at a time, float64
accumulators, raises on NaN/inf/shape mismatch). Outputs:
`cost_readout_corrected.json`, `raw/d2b_s{0,1,2}.npz`, `raw/d2b_bins.csv`.

The fault it corrects: `d2_cost.py` built every alternative readout map by
scattering replacement values at valid pixels only (~21% of the image) and
pushing that patchy map through the frozen dilated refinement, whose
receptive field is far larger than one pixel. The refinement therefore
received a disparity map full of artificial step discontinuities at the
valid/invalid boundary — an input unlike anything it ever saw. Proof the
delivery was broken, not the readout: the D2 CONTINUOUS ORACLE fed the
refinement the exact ground-truth candidate yet scored 5.7061 px against a
1.4410 px baseline. A perfect input cannot make a frozen model four times
worse unless the delivery mechanism is broken. Those D2 readout numbers are
preserved beside this correction, marked CONFOUNDED in place.

Three confound-free protocols replace them:

- Protocol A (primary, zero confound): the refinement is NOT re-run. The
  residual the model actually produced (`refinement_residual`) is held
  literally fixed; each readout variant scores
  `relu(alt_cand + residual_actual)` at valid pixels only. Gate: the
  soft-argmin variant must reproduce the baseline EPE to 1e-9.
- Protocol B (secondary, no GT needed): the hard argmax is defined at every
  pixel, so a DENSE argmax map over the full image goes through the
  unchanged frozen refinement, scored on valid pixels. GT-based oracles
  cannot be made dense (GT is sparse by nature) and get Protocol A only.
- Protocol C (patchiness control): baseline soft-argmin map + 0.5 candidate
  units, sparse at valid pixels only (C1, same scatter pattern as the bad
  run) vs dense at all pixels (C2), both through the unchanged frozen
  refinement. (C1 − C2) quantifies the sparse-substitution artefact.

How to re-run: `python stage_a_diagnostics/scripts/d2b_readout.py`.
Gates (full runs): baseline reproduces D2 seed-0 EPE 1.4149795737 to
0.00e+00; Protocol A soft-argmin reproduces baseline per seed to
−9.45e-10 / +1.02e-11 / +6.47e-11 px (tolerance 1e-9) — all PASS.

- Protocol A pooled EPE, method-mean (deltas vs baseline 1.4410): soft
  1.4410 (−0.0000, gate); argmax 2.6817 (+1.2408); top-2 2.2981 (+0.8571);
  top-3 1.9946 (+0.5536); top-4 1.8739 (+0.4329); continuous 1.8132
  (+0.3722). Pooled D1: baseline 8.585; argmax 24.489; top-2 16.494;
  top-3 12.465; top-4 10.978; continuous 10.055.
- Protocol A for GT>=96 (34152 px, 0.898% of valid; px-weighted mean;
  baseline 12.4149): argmax 12.9285 (+0.5136); top-2 12.4208 (+0.0059);
  top-3 11.9862 (−0.4287); top-4 11.5897 (−0.8252); continuous 10.3996
  (−2.0153). D1 there improves under every oracle except argmax (seed0:
  baseline 60.737 vs continuous 45.330; seed1: 57.177 vs 44.697; seed2:
  53.528 vs 40.888).
- Protocol A per D1 bin (seed 0, continuous oracle vs baseline): below
  baseline in [64,80) 2.780 vs 3.060, [80,96) 4.255 vs 4.861, [96,112)
  8.101 vs 9.596, [112,128) 10.813 vs 13.535, [128,144) 27.888 vs 33.785,
  [144,160) 32.628 vs 38.689; above baseline in every bin below 64 px
  (which hold ~99% of valid pixels), so the pooled oracle EPEs above
  follow. Top-4 oracle likewise scores below baseline from [80,96)
  upward and above it below 64 px.
- Protocol B dense argmax, method-mean pooled EPE 17.9440 (per seed
  17.8614 / 18.2130 / 17.7577), vs the old sparse-argmax 10.7675. A
  dense, patch-free argmax map through the unchanged refinement scores
  far worse than either — the argmax field is too noisy for the
  refinement to use, with no sparse-substitution artefact involved.
- Protocol C, method-mean pooled EPE: C1 sparse 3.1905, C2 dense 5.4580;
  sparse-substitution effect (C1 − C2) −2.2676 px (per seed −2.2629 /
  −2.2836 / −2.2563). The identical +0.5 readout change scores ~2.3 px
  differently depending only on the substitution pattern — that pattern
  sensitivity is what dominated the confounded D2 numbers.

Factual statement only (no verdict; step 8 does verdicts): as Protocol A
measures it, the frozen cost distribution does contain disparity
information the current readout does not use — but only far out: for
GT>=96 the continuous oracle lands 2.02 px closer than the baseline
(10.3996 vs 12.4149) and the top-4 oracle 0.83 px closer, with matching
per-bin gains from [64,80) upward; pooled over all valid pixels, where
~99% sit below 64 px, no oracle readout (argmax, top-2/3/4, continuous)
lands closer than the baseline given the refinement output the model
actually generated.

Nothing else was unmeasurable. NOT MEASURABLE WITH CURRENT ARTIFACTS:
dense GT-oracle maps through the refinement (GT is sparse by nature).

## How to re-run

```
python stage_a_diagnostics/scripts/d0_verify.py   # GATE: must print GATE: PASS
python stage_a_diagnostics/scripts/d1_bins.py
```

- `d0_verify.py` is a read-only mirror of `phase2/scripts/eval_p2a.py`
  (same dataset split, GT scale, valid mask, pooled metrics, contract guard
  imported unmodified from `phase1.harness.frozen_eval`; same
  `StereoNetConfig`). It never writes under `phase2/`.
- `d1_bins.py` scores `reference/onnx/stereonet.onnx` with the
  `phase1/runs/arm_v_diag/reference_strata.py` mechanism at the wider D1 bin
  edges (no interpolation of `reference_strata.json`, whose bins differ).
- All writes are atomic (`.tmp` + rename). Raw per-pixel vectors are preserved
  under `raw/`. No prior record is overwritten.

## Headline results (measured, not inferred)

- D0 GATE: PASS. Seed-0 EPE 1.4149796 px reproduced to 0.00e+00;
  3-seed mean 1.4409826 px reproduced to 2.0e-11 (tolerance 1e-3 px).
- Reference ONNX re-scored at these bins: 1.3134471 px (diff −2.3e-08).
- Total gap: 0.1275355 px. Gap_contribution column sums to 0.1275355 px
  (diff −2.2e-16): CLOSES.
- Top 3 bins by gap contribution: [112,128) 0.029622, [96,112) 0.026210,
  [16,32) 0.024537 px.
- Empty bins [160,184) and [184,inf) contain 0 valid pixels; [144,160) has
  9 px (EPE reported, regression withheld as not meaningful). Nothing else
  was unmeasurable. No mechanism conclusion is drawn here by design.

## D3 — frozen matching representation (measured, not inferred)

FROZEN-MODEL DIAGNOSTIC. No weight altered, nothing trained, eval split
unchanged, no file outside this directory modified. Script:
`scripts/d3_matching.py` (one image at a time; float64 accumulators; raises
on NaN/inf; aborts on warnings under `-W error::RuntimeWarning`, which is
how the run was executed). Outputs: `matching_diagnostic.json`,
`raw/d3_s{0,1,2}.npz`, `raw/d3_bins.csv`.

Architecture verification (scene 0, all seeds): `forward(...,
return_stages=True)` returns a `(disparity, stages-dict)` tuple and
`stages["left_features"]` / `stages["right_features"]` are `(1, 32, 46,
154)` for `(1, 3, 368, 1232)` input, i.e. stride 8 as briefed.
`feature_normalize` is FALSE in the P2A configs used, so the stages
features are the raw extractor output; representation C normalises them
as a DIAGNOSTIC ONLY. All four representations use the same
shift-right candidate geometry (candidate k = left(x) − right(x−k),
24 candidates). A wiring gate asserts the A_L1 volume varies across
candidates; a smoke run (seed 0, `--limit 2`) preceded the full 3-seed run.

GT at feature resolution: NEAREST sampling at grid centres
(y\*8+4, x\*8+4), no averaging/interpolation; kept cells have sampled
GT > 0; gt_cand = GT_px/8, gt_idx = clip(round(gt_cand), 0, 23).
Left-edge rule: (cell, k) pairs with feature column x < k are excluded
from EVERY statistic (matched against zeros); cells with x < gt_idx are
dropped. Per seed: 59431 GT cells sampled (of 283360), 1313 dropped by
the left-edge cell rule, 79326 pixel-candidate pairs removed by the
pair rule, 58048 defined cells remain (36 without a valid
non-neighbourhood candidate, 70 without a valid negative; identical
masks across reps since validity is geometry-only). Pooled defined
cells over 3 seeds: 174144 — NOT under 10,000, so no low-n caveat on
the pooled numbers. GT>=96 holds 393 cells/seed.

A and B coincide under L1: YES (per-image max abs diff exactly 0.0 on
all seeds, `ab_coincide_under_l1` true). Stated, not hidden: the
headline B numbers below use an L2 (RMS) reduction so A and B are
actually distinct. Group-wise correlation (D) is NEGATED to a
lower-is-better cost scale before every rank/margin computation.
Mathematical property of the specified definition, verified in the
numbers: group-averaging rescales the full channel dot by 1/G, so
D_G4/D_G8/D_G16 ranks and overlaps are IDENTICAL by construction
(rank-1 0.2501 all three) and margins scale as 1/G. Neighbourhood =
{k:|k−gt_idx|<=1}; negatives = valid k with |k−gt_cand|>2.0; gap =
(cell_negmean−cell_GT)/std over the cell's valid candidates.

Headline numbers, method-mean across 3 seeds (per-seed pooled rank-1:
A 0.6565/0.6508/0.6335; B 0.6606/0.6609/0.6369; C 0.6418/0.6325/0.6116;
D_G8 0.2802/0.2476/0.2225 — ordering identical on every seed):

| rep | rank-1 | rank<=3 | margin frac>0 | overlap | gap mean |
|---|---|---|---|---|---|
| A (L1) | 0.6469 | 0.8412 | 0.7320 | 0.0355 | 1.7757 |
| B (L2/RMS) | 0.6528 | 0.8450 | 0.7386 | 0.0350 | 1.8036 |
| C (L2-norm + L1) | 0.6286 | 0.8210 | 0.7035 | 0.0242 | 1.9256 |
| D (grp-corr G=8, negated) | 0.2501 | 0.5358 | 0.3202 | 0.1819 | 1.3308 |

Mean margins on the raw scale (A 7.57e6, B 9.60e6, C 0.0162,
D_G8 −6.49e15) are NOT comparable across representations: the frozen
raw features have ~1e6–1e7 magnitude (e.g. pooled A pos-mean
2.31e7 vs neg-mean 6.75e7), so only the scale-free columns above
(rank, margin-sign fraction, overlap, gap) compare across reps.

GT>=96 px (393 cells/seed): A rank-1 0.3647 margin-frac>0 0.4436
overlap 0.1781 gap 1.1778; B 0.3681 / 0.4580 / 0.1782 / 1.1956;
C 0.3783 / 0.4427 / 0.0990 / 1.3831; D_G8 0.1730 / 0.2188 / 0.2360 /
0.9472. Per-bin (seed-0 rank-1 A/B/C/D8): [0,16) n=8040
0.671/0.676/0.651/0.293; [16,32) n=18053 0.684/0.688/0.670/0.294;
[32,48) n=17160 0.661/0.664/0.646/0.276; [48,64) n=12562
0.627/0.631/0.613/0.268; [64,80) n=1463 0.576/0.580/0.563/0.236;
[80,96) n=377 0.438/0.432/0.411/0.210; [96,112) n=286
0.378/0.385/0.371/0.171; [112,128) n=107 (low-n)
0.383/0.383/0.467/0.336; [128,144) and [144,160) EMPTY at feature
resolution (nearest grid sampling hit none of the 1071/9 full-res px);
[160,184), [184,inf) EMPTY.

Factual statement only (no verdict; step 8 does verdicts): pooled, B
edges A on rank-1/rank<=3/margin-sign (0.6528 vs 0.6469) while C has
the best overlap (0.0242) and gap (1.9256); D separates worst on every
scale-free metric. In the high-disparity bins the ordering changes on
rank-1: C leads in [112,128) on all 3 seeds (0.467/0.327/0.374 vs A
0.383/0.271/0.308, n=107 low-n) and leads the GT>=96 subset on rank-1,
overlap and gap.

The model does NOT score candidates with any of these scalar
reductions. It feeds the full 32-channel signed-difference volume into
a trained 3D aggregation which produces the cost that is actually read
out. The scalar reduction is a DIAGNOSTIC PROXY chosen to make the four
representations comparable, not the model's own scoring function.
Therefore: a representation winning here establishes MECHANISM
PLAUSIBILITY ONLY. It does NOT predict that a trained architecture
using it will improve EPE. No architecture is recommended and no
mechanism verdict is declared here.

NOT MEASURABLE WITH CURRENT ARTIFACTS: what a trained aggregation
would do with any of these representations; nothing else was
unmeasurable.

## D4 — aggregation diagnostic (measured, not inferred)

FROZEN-MODEL DIAGNOSTIC. No weight altered, nothing trained, eval split
unchanged, no file outside this directory modified by this step.
Script: `scripts/d4_aggregation.py` (one image at a time — the PRE volume
is (1,32,24,46,154) ~ 27 MB float32 and is never held for more than one
scene; float64 accumulators; raises on NaN/inf). Outputs:
`aggregation_diagnostic.json`, `raw/d4_s{0,1,2}.npz`, `raw/d4_bins.csv`.

How to re-run: `python stage_a_diagnostics/scripts/d4_aggregation.py`
(seed 0 `--limit 2` smoke test preceded the full 3-seed run).

Conventions reused EXACTLY from D3 (imported logic, not re-derived):
nearest-centre GT sampling, gt_cand = GT_px/8,
gt_idx = clip(round(gt_cand), 0, 23), left-edge x<k exclusion from EVERY
statistic, cells with x<gt_idx dropped, neighbourhood
{k:|k−gt_idx|<=1}, negatives = valid k with |k−gt_cand|>2.0, overlap =
fraction of pooled valid negatives scoring better (lower) than the median
positive, gap = (cell_negmean−cell_GT)/std over valid candidates, same
ok-mask (valid non-neighbourhood AND valid negative). Masks are
geometry-only, hence identical for PRE and POST and identical to D3:
per seed 59431 GT cells sampled, 1313 dropped by the left-edge cell rule,
79326 pairs removed, 36/70 margin/negative-undefined, 58048 defined cells
remain — the same counts D3 reports. Gate: the PRE proxy reproduces the
D3 A_L1 rank-1 per seed to 0.00e+00 on all seeds (0.6565/0.6508/0.6335),
so PRE here IS the D3 A_L1 representation, recomputed from the frozen
volume rather than from the features.

PRE vs POST objects (stated, not hidden): PRE = A_L1 diagnostic proxy
(mean over channels of |cost_volume|); POST = the model's OWN cost
(stages["aggregated_cost"], the readout's softmax(-cost) input). NOT the
same kind of object: only RANKS and ORDERINGS are compared, never raw
magnitudes. Entropy = softmax(-cost) over VALID candidates only (the
left-edge rule covers every statistic); PRE entropy is labelled proxy
entropy. Both entropies collapse: POST valid-only softmax is exactly
one-hot on every cell (entropy 0.0; aggregated costs span ~1e9), PRE has
mean 6.3e-06 with a small tail (max 0.35 on seed 0). The entropy column
is therefore reported but uninformative at raw scale; readout-level
entropy (~1.85 nats, standardised) is D2's quantity, not this step's.

Regimes: OCCLUSION by footprint-max — the full-res occluded-only mask
(valid disp_occ_0 AND invalid disp_noc_0, the reference_strata.py
definition) lifted by max over each stride-8 footprint. Centre-sampling
the sparse binary mask yields ZERO occluded cells (measured), so
max-pooling the dense mask is used and stated; averaging sparse GT would
not be legitimate, max over a binary validity mask is. Full-res context
(measured, not interpolated): 81581 of 3802797 valid pixels (2.144%)
are occluded-only, yet the feature-res OCC stratum holds only 134
cells/seed (402 pooled, low-n flagged). TEXTURE by full-res left
grayscale gradient magnitude, mean per stride-8 cell, per-seed median
split (threshold 5.0557, identical on all seeds since it is
model-independent; 29024/29024 cells per seed). EDGES: gt_idx differs by
>= 2 from any in-bounds 4-neighbour with sampled GT>0 (676 cells/seed).
LOW/HIGH split at gt_cand 8 (55815/2233 per seed). GT96_OCC/GT96_NONOCC
(2/391 per seed) test the occlusion confound on the D3 negative
high-disparity margin directly.

Headline numbers, method-mean across 3 seeds (per-seed pooled rank-1
deltas −0.4759/−0.4943/−0.4989 — same sign on every seed; rank<=3
deltas +0.0597/+0.0270/−0.0160):

| group (n/seed) | PRE rank-1 | POST rank-1 | Δ rank-1 | PRE mgn-frac | POST mgn-frac | Δ | overlap PRE→POST | gap PRE→POST |
|---|---|---|---|---|---|---|---|---|
| POOLED (58048) | 0.6469 | 0.1572 | −0.4897 | 0.7320 | 0.5247 | −0.2073 | 0.0355→0.0101 | 1.7757→2.2801 |
| GT>=96 (393) | 0.3647 | 0.1018 | −0.2629 | 0.4436 | 0.3961 | −0.0475 | 0.1781→0.0419 | 1.1778→1.5702 |
| OCC (134, low-n) | 0.3980 | 0.1468 | −0.2512 | 0.5672 | 0.6194 | +0.0522 | 0.2267→0.1014 | 0.9668→1.1800 |
| NONOCC (57914) | 0.6475 | 0.1572 | −0.4902 | 0.7324 | 0.5245 | −0.2079 | 0.0353→0.0101 | 1.7775→2.2827 |
| TEXLO (29024) | 0.6228 | 0.1457 | −0.4771 | 0.7170 | 0.5244 | −0.1926 | 0.0378→0.0104 | 1.7044→2.2677 |
| TEXHI (29024) | 0.6710 | 0.1687 | −0.5023 | 0.7470 | 0.5251 | −0.2219 | 0.0183→0.0095 | 1.8469→2.2926 |
| EDGE (676) | 0.2751 | 0.2515 | −0.0237 | 0.3264 | 0.4413 | +0.1149 | 0.1847→0.0283 | 1.0022→1.9784 |
| LOW (55815) | 0.6526 | 0.1584 | −0.4942 | 0.7380 | 0.5274 | −0.2106 | 0.0342→0.0098 | 1.7879→2.2955 |
| HIGH (2233) | 0.5044 | 0.1272 | −0.3772 | 0.5820 | 0.4575 | −0.1245 | 0.0873→0.0276 | 1.4701→1.8951 |
| HIGH_OCC (23, low-n) | 0.3188 | 0.0435 | −0.2754 | 0.4058 | 0.4493 | +0.0435 | 0.2239→0.1027 | 0.8720→0.9777 |
| HIGH_NONOCC (2210) | 0.5063 | 0.1281 | −0.3783 | 0.5839 | 0.4576 | −0.1262 | 0.0857→0.0273 | 1.4763→1.9047 |
| GT96_OCC (2, low-n) | 0.3333 | 0.0000 | −0.3333 | 0.5000 | 0.0000 | −0.5000 | measured, n=6 pooled | 1.1599→0.0473 |
| GT96_NONOCC (391, low-n per-seed) | 0.3649 | 0.1023 | −0.2626 | 0.4433 | 0.3981 | −0.0452 | 0.1784→0.0414 | 1.1778→1.5779 |

Per-bin (method-mean rank-1 delta, all populated bins negative):
[0,16) −0.0914, [16,32) −0.6330, [32,48) −0.5664, [48,64) −0.4539,
[64,80) −0.4420, [80,96) −0.2449, [96,112) −0.2786, [112,128) −0.2212
(low-n, n=107); rank<=3 delta is +0.1462 in [0,16) and mixed elsewhere
(−0.19 in low-n [112,128)); [128,144) and above EMPTY at feature
resolution, as in D3.

POST spatial consistency (dense valid-masked argmin vs existing
in-bounds 4-neighbours), method-mean: pooled agreement 0.8670, mean
|diff| 0.1607 candidates, full-4-all-agree 0.5733. By regime: EDGE
0.7114/0.5949/0.3219; GT96_NONOCC 0.7821/0.5543/0.5069; HIGH
0.8315/0.3115; TEXLO 0.8583 vs TEXHI 0.8757; OCC 0.8212 (low-n);
GT96_OCC and HIGH_OCC low-n flagged (69 and 6 cells pooled).

Factual statement only (no verdict; step 8 does verdicts): POST rank-1
is materially LOWER than PRE pooled (−0.4897), in every populated D1
bin, and in every regime including textureless/textured and low/high
disparity; POST margin is less often positive pooled (−0.2073) with the
only margin-frac gains in the low-n OCC (+0.0522) and HIGH_OCC (+0.0435)
strata and at EDGE (+0.1149, n=676/seed); overlap is lower POST than PRE
in every group (pooled 0.0355→0.0101) and gap higher (pooled
1.7757→2.2801); rank<=3 is flat-to-up pooled (+0.0236, +0.1462 in
[0,16)). The POST argmax field is spatially coherent pooled (0.8670
neighbour agreement) and less so at edges (0.7114) and at GT>=96 without
occlusion (0.7821, mean |diff| 0.5543). On the D3 negative-margin
confound: GT96_NONOCC (391 cells/seed, occlusion fully removed) shows
PRE margin-frac 0.4433 and POST 0.3981 — the negative-margin majority
persists without occlusion — while GT96_OCC holds 2 cells/seed (6
pooled), so occlusion cannot account for it as measured. POST rank-1
0.1572 sits in the same range as the independent D2 readout-level
rank-1 0.1330, measured differently.

Prior work: phase1/runs/arm_v_diag/prevs_post_aggregation.py frames the
same question but answers no part of these measurements — ARM-V (not
P2A) checkpoints, raw L2-norm cost, upsampled full resolution, no
left-edge exclusion, no D1 bins or regimes. None of its numbers are
reused; D3's conventions are used instead so PRE here is exactly the D3
A_L1 representation (gate above).

NOT MEASURABLE WITH CURRENT ARTIFACTS: nothing in this brief was
unmeasurable; every listed quantity was measured. GT96_OCC (6 pooled)
and HIGH_OCC (69 pooled) are reported with low-n flags rather than
presented alongside well-populated regimes without comment.

## D5 — refinement diagnostic (measured, not inferred)

FROZEN-MODEL DIAGNOSTIC. No weight altered, nothing trained, eval split
unchanged, no file outside this directory modified by this step.
Script: `scripts/d5_refinement.py` (one image at a time; float64
accumulators; raises on NaN/inf). Outputs:
`refinement_diagnostic.json`, `raw/d5_s{0,1,2}.npz` (per-valid-pixel
init/residual/final/GT/top-1/margin/entropy as float32 store, float64
compute; occluded flag), `raw/d5_bins.csv`.

Population: FULL-RESOLUTION frozen valid mask (GT > 0), 3802797 px/seed
— the same pixels D0/D1/D2 used, NOT the D3/D4 feature-res cell set.
Gate: `disparity_final` reproduces seed-0 EPE 1.4149795737 to 0.00e+00 —
PASS. D2 cross-check: recomputed standardised-softmax top-1/margin/
entropy match D2's pooled values to <=2.9e-11 on seed 0 (all seeds pass
1e-6). Occlusion uses the reference_strata.py definition (valid in
disp_occ_0 AND not valid in disp_noc_0): 81581 px/seed, 2.145% of valid.

How to re-run: `python stage_a_diagnostics/scripts/d5_refinement.py`
(seed-0 `--limit 2` smoke test precedes the full run; `--reuse-raw`
repeats the identical pure-numpy analysis from `raw/d5_s*.npz` without
GPU inference and reproduces every pooled number exactly).

Units (verified in `src/models/stereonet/stereonet.py`):
`disparity_final = relu(disparity_initial + refinement_residual)` with
`disparity_initial` in CANDIDATE units (0..23) and `disparity_final` in
pixels, so the residual fuses candidate->pixel scaling AND correction.
Two coarse conventions side by side; HEADLINE is COARSE_SCALED (8x),
because the soft-argmin output is only interpretable as disparity after
candidate->pixel scaling. Pooled method-mean (per seed 0/1/2):

| convention | EPE (px) | D1 (%) |
|---|---|---|
| COARSE_RAW (units-mismatched literal tensor) | 30.4720 (30.4317/30.4919/30.4926) | 99.32 |
| COARSE_SCALED = 8x init (headline) | 8.7944 (9.0542/8.7083/8.6206) | 90.38 |
| REFINED = final | 1.4410 (1.4150/1.4486/1.4594) | 8.59 |
| delta REFINED − COARSE_SCALED | −7.3534 (−7.6392/−7.2597/−7.1612) | −81.79 pp |

Refinement effect pooled, method-mean (per seed 0/1/2): improved
0.9507 (0.9534/0.9502/0.9485), worsened 0.0493
(0.0466/0.0498/0.0515), unchanged-within-1e-6 ~3.5e-07 (2/0/0 px);
mean improvement on helped 7.895 px (8.151/7.810/7.725); mean
degradation on harmed 3.105 px (2.840/3.239/3.235). Residual:
signed mean +30.287 px, mean-abs 30.287 px (residual >= 0 nearly
everywhere), relu clamp fraction 1.8e-07 pooled (2/0/0 px) — the final
relu does essentially no work.

Per D1 bin (seed 0; same pattern seeds 1–2): refined EPE beats
coarse-scaled EPE in every bin on seeds 0 and 1, and in every bin
except [64,80) on seed 2 (2.96 vs 2.85 px — the single bin-level
regression, honestly recorded). Improved fraction is lowest in [64,80)
on all seeds (0.473/0.552/0.495) vs 0.90–1.00 elsewhere; EPE there is
3.06/2.89/2.96 px. Oracle alpha* per bin (seed 0): 0.97–1.04 below
64 px, 1.10 ([64,80)/[80,96)), 1.32/1.63 ([96,112)/[112,128)), ~4.0
([128,144), n=1071), 3.71 ([144,160), n=9).

Confidence vs benefit (method-mean deciles, equal-width over [0,1];
top-1 decile 9 and margin decile 9 empty — top-1 never reaches 0.9):

| top-1 decile | px (3 seeds) | mean improvement | frac improved | mean \|resid\| |
|---|---|---|---|---|
| 0 [0.0,0.1) | 8409 | 5.28 | 0.676 | 70.7 |
| 1–2 | 44613/158042 | 7.52/5.87 | 0.753/0.758 | 53.6/41.5 |
| 3–5 (bulk) | 1.66M/3.90M/4.02M | 6.19/6.48/6.89 | 0.922/0.950/0.963 | 35.3/34.0/32.2 |
| 6–8 | 768K/563K/291K | 11.29/12.48/13.36 | 0.972/0.978/0.989 | 12.8/7.2/4.3 |

Margin deciles show the same shape: mean |residual| falls 35.4 → 3.9
from decile 0 to 8 while frac improved rises 0.926 → 0.991 and mean
improvement rises 6.42 → 13.51. So the correction is NOT independent
of confidence: the largest corrections land where the cost
distribution was most ambiguous (mean |resid| 71 px at lowest top-1),
while the highest success rate and largest mean improvement land
where it was most peaked.

Headroom probes (frozen, pure rescoring, pooled method-mean):
ORACLE SCALE alpha* = 1.018 (1.039/1.000/1.015), EPE at alpha*
1.4330 vs baseline 1.4410 (gain 0.008); EPE at alpha=8 is 38.6 px.
ORACLE OFFSET beta* = +0.095 (−0.0025..+0.2025 across seeds), EPE
1.4340 (gain 0.007). No systematic GLOBAL mis-scaling (global alpha gain
0.008 px method-mean; per-seed global gains 0.0207/0.0000/0.0031 px at
seeds 0/1/2), but strong DISPARITY-DEPENDENT mis-scaling: per-bin alpha*
is flat near 1.0 below 96 px (seed 0: 0.974/1.022/1.041/1.039) and rises
steeply above it (1.104 at [64,80), 1.319 at [96,112), 1.631 at
[112,128), 4.003 at [128,144)), and the mass-weighted per-bin oracle-alpha
gain — sum over bins of frac_px * (epe_refined − alpha_epe) — is 0.0481 /
0.0263 / 0.0289 px at seeds 0/1/2, about six times the global figure at
seed 0. That per-bin figure is a GT-FITTED ORACLE CEILING, not an
achievable improvement.
Sign-only: the briefed literal formula
`relu(init+|GT-init|*sign(resid))` scores EPE ~7e-05 — reported
literally, but it is DEGENERATE, not a finding: with GT in px >> init
in candidate units and resid > 0 nearly everywhere it collapses to GT
itself. The units-consistent supplementary variant in headline space,
`relu(8*init+|GT-8*init|*sign(resid))`, scores 16.80 px
(17.39/16.56/16.47) — worse than the 8.79 px coarse-scaled baseline —
because 8*init overshoots GT on ~95% of pixels below 64 px (sign
agreement there is 0.000–0.030, rising to 0.96–1.00 above 80 px),
while the residual operates in the raw frame where GT_px > init_cand
makes its positive sign trivially correct. Sign-vs-magnitude
decomposition is therefore frame-dependent and ill-posed here; the
residual fuses scaling and correction inseparably, exactly as briefed.

Regimes (seed 0; seeds 1–2 match): GT<64 (95.4% of valid) improved
0.970/mean +7.92 px; GT 64–96 improved 0.545/mean +0.56 px (weakest
regime on all seeds: 0.545/0.610/0.562); GT 96–128 improved
0.923/mean +7.42; GT>=128 improved 0.906/mean +13.36 (n=1080).
Occluded (2.1%) improved 0.786/mean +4.91 vs non-occluded
0.957/mean +7.70; refined EPE 5.81 vs 1.32 px.

Factual statement only (no verdict; step 8 does verdicts): of the four
options, the numbers fit (1) systematically improves — 95.1% of valid
pixels improve on all seeds, pooled EPE falls 7.35 px vs the headline
coarse baseline, every D1 bin improves on seeds 0–1 — with elements of
(2): the GT 64–96 band ([64,80) bin) is helped least (~0.50–0.61
improved, smallest mean gain), and occluded pixels less than
non-occluded (0.79 vs 0.96). No regime is damaged in EPE terms except
the single seed-2 [64,80) bin above. Remaining headroom under
global scale/offset is ~0.01 px (4).

NOT MEASURABLE WITH CURRENT ARTIFACTS: nothing in this brief was
unmeasurable; every listed quantity was measured (the naive
residual_oracle = GT − init was deliberately not reported as a
finding per the brief).

## D6 — high-disparity reassessment (measured, not inferred)

FROZEN-MODEL DIAGNOSTIC. No weight altered, nothing trained, eval split
unchanged, no file outside this directory modified by this step (other
`git status` entries pre-date it). Script:
`scripts/d6_high_disparity.py` (one image at a time where images are
touched; float64 accumulators; raises on NaN/inf). Outputs:
`high_disparity.json`, `raw/d6_s{0,1,2}.npz` (P/G/occluded/gt_cand per
seed), `raw/d6_bins.csv` (per-D1-bin + per-regime rows).

Population: FULL-RESOLUTION frozen valid mask, 3802797 px/seed — the same
pixels as D0/D1/D2/D5. P2A vectors reused from D0/D5 raw dumps (D5 final
matches D0 P to 1e-4, D5 GT == D0 GT, asserted); reference vectors reused
from `raw/d1_reference.npz` (reference_strata.py mechanism, no
re-inference); matching numbers reuse D3/D4 raw npz (not recomputed).
Occlusion is the established mask (valid in disp_occ_0 AND not valid in
disp_noc_0): 81581/3802797 = 2.1445%, confirming D4's 2.14%.

How to re-run: `python stage_a_diagnostics/scripts/d6_high_disparity.py`.

### Regimes (method-mean; slopes pooled over 3-seed concatenation)

| regime | px | frac | EPE | D1 | signed | slope (per-seed) | R2 | max_pred vs max_gt |
|---|---|---|---|---|---|---|---|---|
| GT>=64 | 175472 | 4.6143% | 5.1478 | 30.87 | −3.73 | 0.751 (0.772/0.719/0.763) | 0.634 | 136.29 vs 153.04 |
| GT>=96 | 34152 | 0.8981% | 12.4149 | 57.15 | −11.72 | 0.570 (0.623/0.537/0.550) | 0.091 | 136.29 vs 153.04 |
| GT>=128 | 1080 (low-n) | 0.0284% | 33.3687 | 100.0 | −33.37 | 0.814 pooled, per-seed 0.502/0.848/1.093, R2 0.030 — not meaningful | — | 127.74 (seed 1) vs 153.04 |
| GT>=184 | 0 | EMPTY | — | — | — | — | — | — |

[160,184) and [184,inf) are EMPTY (0 px) — confirmed, not fabricated.
Mean pred vs mean GT per seed: GT>=64: 76.7/76.5/76.8 vs 80.40;
GT>=96: 98.6/96.8/98.6 vs 109.72 (systematic under-prediction).

Reference ONNX per regime: GT>=64 EPE 3.248, slope 0.930, signed −0.10;
GT>=96 EPE 5.757, slope 0.727, signed −2.83; GT>=128 EPE 19.415,
slope 0.471, signed −19.42; GT>=184 EMPTY. The reference is better in
every populated regime. Developing the D1 [112,128) clue (ref 0.642 vs
P2A 0.982): at GT>=64/GT>=96 the reference slope is HIGHER (0.930 vs
0.751; 0.727 vs 0.570) with near-zero bias (−0.10/−2.83 vs −3.73/−11.72),
i.e. better calibrated there; at GT>=128 the ordering flips (ref 0.471
vs P2A pooled 0.814, but P2A per-seed 0.502/0.848/1.093 with R2 0.030 —
the regression is meaningless on 1080 px, and the D1 [128,144) bin slope
is −7.795), while the reference still wins on EPE (19.42 vs 33.37) via
smaller under-prediction bias (−19.42 vs −33.37).

### The five explanations, separately

1. RANGE — REFUTED. Fraction of valid pixels with gt_cand = GT/8 outside
   [0,23] is 0.000000 overall and in every regime (independent
   confirmation of D2). Max gt_cand is 19.13 < 23. Distance from gt_cand
   to the nearest integer candidate: mean 0.240 (>=64), 0.209 (>=96),
   0.065 (>=128); max 0.5/0.5/0.406. Max prediction never exceeds max GT
   in any regime on any seed. The hypothesis set covers the correct
   disparity everywhere it is evaluated.
2. CALIBRATION — SUPPORTED. Per-regime oracle-alpha rescale (GT-fitted
   ceiling, not achievable): GT>=64 alpha* 1.146/1.100/1.111, gain
   0.30/0.14/0.17 px (6.0/2.7/3.4% of regime EPE); GT>=96 alpha*
   1.538/1.494/1.446, gain 2.06/1.78/1.30 px (17.6/13.2/10.8%);
   GT>=128 alpha* 3.993/3.872/4.402, gain 19.05/20.85/27.06 px
   (56/67/77% — low-n, oracle ceiling). D5 per-bin alpha* restated:
   ~0.97–1.04 below 64 px, 1.10 at [64,80)/[80,96), 1.32/1.63 at
   [96,112)/[112,128), ~4.0 at [128,144). Slope falls 0.751 → 0.570
   from >=64 to >=96 while signed error grows −3.73 → −11.72.
3. MATCHING — SUPPORTED. Feature-res cells (D3/D4 sampling, not
   full-res). GT>=64 (2233 cells/seed): PRE rank-1 0.518/0.505/0.490,
   margin-frac>0 0.596/0.579/0.571; POST rank-1 0.154/0.117/0.111,
   margin-frac 0.519/0.425/0.429. GT>=96 (393 cells/seed): PRE rank-1
   0.379/0.356/0.359 with margin-frac>0 0.440/0.443/0.448 — the
   GT-candidate margin is negative in the majority on all seeds, before
   and after aggregation (POST margin-frac 0.450/0.338/0.399); D4 showed
   this persists with occlusion removed (GT96_NONOCC margin-frac PRE
   0.4433/POST 0.3981). GT>=128/GT>=184: EMPTY at feature resolution
   (nearest grid sampling hit none of the 1080/0 full-res px) — matching
   there is NOT MEASURABLE WITH CURRENT ARTIFACTS.
4. OCCLUSION — SUPPORTED at >=128; WEAK below it. GT>=64: occluded
   fraction 18.15% (31853 px), EPE occluded ~10.9/10.9/9.9 vs
   non-occluded 3.74/4.12/3.99, occluded share of regime error mass
   ~39/37/36%. GT>=96: occluded 26.52% (9058 px), EPE occluded
   21.9/25.4/22.1 vs non-occluded 8.06/9.21/8.38, error-mass share
   ~49/50/49% — half the error mass, but non-occluded EPE stays high, so
   occlusion does not account for the regime. GT>=128: 1071/1080 px
   (99.17%) are occluded — the regime is essentially correspondence-
   invalid as measured; EPE occluded 33.78/31.05/35.28 vs non-occluded
   (9 px) 38.69/29.80/29.88.
5. SUPERVISION COVERAGE — SUPPORTED at >=96; WEAK elsewhere. Raw
   hailo_calib training mass vs hailo_val eval mass: GT>=96 is 0.228%
   vs 0.898% (3.9x under-represented); GT>=64 is 3.93% vs 4.61%;
   GT>=128 is 0.0022% vs 0.0284%. Effective training distribution after
   the P2A augmentation, Monte-Carlo emulated with the exact recipe
   (s ~ logUniform(0.7,1.7) per sample, w=round(512/s), h=round(256/s),
   uniform top-left crop in the 368x1232 source frame, cv2 NEAREST
   disparity resize to 256x512, disparity scaled by realised sx=512/w,
   loss mask gt>0 and gt<184; 5 draws per training scene, 800 augmented
   samples, 24.6M valid px; diagnostic rng seed 0): GT>=96 rises to
   0.444% — still ~2x below eval 0.898%. GT>=64 effective 6.01% vs eval
   4.61% (covered); GT>=128 effective 0.0438% vs eval 0.0284%
   (over-covered after augmentation, but eval-low-n). Per-bin: training
   [96,112)/[112,128)/[128,144) raw 0.194%/0.032%/0.002% vs eval
   0.559%/0.311%/0.028%; effective 0.338%/0.063%/0.034%.

Factual statements only, no overall mechanism verdict (step 8 does
verdicts). Low-n regimes flagged: [128,144) (1071 px), [144,160) (9 px).
Nothing else was unmeasurable except feature-res matching at >=128.

## D7 — spatial / context diagnostic (measured, not inferred)

FROZEN-MODEL DIAGNOSTIC. No weight altered, nothing trained, eval split
unchanged, no file outside this directory modified by this step (other
`git status` entries pre-date it). Script:
`scripts/d7_spatial.py` (one image at a time; float64 accumulators; raises
on NaN/inf). Outputs: `spatial_context.json`, `raw/d7_s{0,1,2}.npz`
(per-valid-pixel abs_err/GT/occluded/top-1/margin/entropy/grad_mag/
edge_dist as float32 store, float64 compute), `raw/d7_bins.csv`.

Population: FULL-RESOLUTION frozen valid mask (GT > 0 in disp_occ_0),
3802797 px/seed — the same pixels as D0/D1/D2/D5/D6. Occlusion is the
established mask (valid in disp_occ_0 AND not valid in disp_noc_0):
81581 px/seed, 2.145% of valid (non-occ 3721216 px). EVERY metric below is
reported on all-valid AND non-occluded-valid; no conclusion flips between
the two (stated per metric, not assumed).

How to re-run: `python stage_a_diagnostics/scripts/d7_spatial.py`
(seed-0 `--limit 2` smoke test precedes the full run).

Gates (full runs): seed-0 EPE reproduces D0 1.4149795737 to 0.00e+00 —
PASS; valid counts 3802797 on all seeds; D2 cross-check (recomputed
standardised-softmax top-1/margin/entropy vs `cost_distribution.json`
pooled): diff 0.0/0.0/0.0 on seed 0, 0.0/0.0/0.0 on seed 1,
-4.4e-16/0.0/0.0 on seed 2 (tolerance 1e-6) — PASS on all seeds.

Receptive field from the P2A architecture (stated arithmetic, full-res
px; rule RF += (k−1)*dilation*stride_acc per conv): feature extractor —
3x 5x5 stride-2 convs RF 1→5→13→29, 6 residual blocks x 2 convs 3x3 at
stride 8 (+192), output 3x3 (+16) = 237 px at stride 8; aggregation
3x3x3 at feature res (+16 full-res px per layer): 4 filter layers → 301,
4+1 incl. output layer → 317; refinement alone at full res: input 3x3
(1→3), 6 blocks x 2 convs at dilations (1,2,4,8,1,1) with BOTH block
convs dilated (+4/+8/+16/+32/+4/+4 → 3→71), output 3x3 (+2) = 73 px;
end-to-end (coarse map upsampled, then refinement combines a 73-wide
neighbourhood of it): 301+72 = 373 (4-filter count) / 317+72 = 389
(5-conv actual). HEADLINE end-to-end RF: 389 full-resolution px
(`aggregation.py` builds 4 filter + 1 output 3x3x3 convs; the brief
counts 4, so both variants are recorded).

### 1. Neighbouring-error correlation (pooled method-mean; per-seed ordering identical)

Pearson r of e(x,y) with e(x+d,y) (h) and e(x,y+d) (v), pairs survive iff
BOTH pixels are in the population; per-D1-bin pairs assigned by the
anchor (left/top) pixel's GT bin. Permuted control: per image, signed-e
values permuted among valid pixels (seeded rng; marginal preserved;
nonocc pop permuted separately within nonocc pixels); the same
permutation drives the permuted components. Seed-0 surviving pairs
(all-valid): h d=1/4/16: 1966690/2210137/1897116; v: 1725719/1834007/
1653327. Nonocc: h 1936122/2169169/1866386; v 1697481/1805116/1625666.

| pop | field | ori | d=1 real (S0/S1/S2) | d=1 perm | d=4 real | d=4 perm | d=16 real | d=16 perm |
|---|---|---|---|---|---|---|---|---|
| all | e | h | 0.9912/0.9929/0.9924 | 0.031 | 0.9513 | 0.031 | 0.6674 | 0.031 |
| all | e | v | 0.9846 | 0.032 | 0.8353 | 0.031 | 0.4262 | 0.031 |
| all | \|e\| | h | 0.9924 | 0.040 | 0.9526 | 0.040 | 0.6998 | 0.040 |
| all | \|e\| | v | 0.9845 | 0.042 | 0.8391 | 0.042 | 0.4631 | 0.041 |
| nonocc | e | h | 0.9904 | 0.017 | 0.9419 | 0.016 | 0.6013 | 0.017 |
| nonocc | e | v | 0.9807 | 0.015 | 0.8051 | 0.017 | 0.3292 | 0.016 |
| nonocc | \|e\| | h | 0.9906 | 0.026 | 0.9426 | 0.026 | 0.6348 | 0.026 |
| nonocc | \|e\| | v | 0.9802 | 0.026 | 0.8047 | 0.026 | 0.3575 | 0.026 |

(d=4/d=16 real columns are method-means; perm columns method-means;
full d=1/2/4/8/16/32 series per seed in `spatial_context.json`.)
The permuted control is small but nonzero (0.015–0.045, lower without
occlusion) rather than exactly zero: permutation is within-image, so each
image's mean error is preserved and per-image mean differences contribute
to the pooled correlation (verified with a synthetic grouped-iid check in
which pooled permuted r tracks the group-mean variance share). The
real-vs-control gap is 0.29–0.96 at every offset/orientation/population.
Per D1 bin (seed 0, all-valid, e, d=1 real h/v): [0,16) 0.984/0.972;
[16,32) 0.987/0.981; [32,48) 0.982/0.965; [48,64) 0.988/0.977;
[64,80) 0.995/0.993; [80,96) 0.999/0.992; [96,112) 0.998/0.994;
[112,128) 0.999/0.997; [128,144) 0.996/0.998 (low-n, 1071 px);
[144,160) NO_PAIRS (9 px, flagged); [160,184)/[184,inf) EMPTY.

Autocorrelation length (first d with r below half r(d=1)), identical on
all 3 seeds, both pops, e and |e|: horizontal 32 px (seed-0 all-valid e:
0.991→0.946→0.836→0.645→0.422 at d=1/4/8/16/32), vertical 16 px
(0.983→0.829→0.601→0.405→0.243). Against the 389 px end-to-end RF (73 px
refinement-only): the error structure sits well INSIDE a local receptive
field — it does not extend beyond it.

### 2. Error clustering / connected regions (|e| > 3, 8-connectivity via scipy.ndimage.label)

Error pixels: 332842/336266/349305 per seed (8.75/8.84/9.19% of valid;
nonocc 298273/304571/315970). Per-image components (method-mean real vs
permuted): all-valid 1897 vs 6604+ (seed 0: 1851.5 vs 6604.4); nonocc
1710.6 vs 6069.2. Median component size 1.0 px both real and permuted
(many isolated pixels either way); p90 real 6.0 vs permuted 2.0; max
real 22037/16340/13363 vs permuted 104/99/192 (nonocc max
18981/13508/12838 vs 75/—/—). Fraction of error mass in components
> 100 px, method-mean: all-valid 0.3131 real vs 0.0004 permuted
(per seed 0.3160/0.3115/0.3118 vs 0.00031/0.00000/0.00096); nonocc
0.2927 real vs 0.0000 permuted (0.2920/0.2934/0.2927 vs 0/0/0).
Real errors form ~3.5x fewer but far larger regions than the scattered
control; the contrast survives occlusion removal unchanged.

### 3. Edge proximity — DEGENERATE AS SPECIFIED, honestly recorded

With the specified rule (neighbouring valid GT differing by >= 3 px,
4-neighbourhood), the discontinuity set is EMPTY: a dataset-level check
over all 40 eval scenes finds the maximum adjacent valid-valid GT
difference is 2.0 px (4- AND 8-connectivity), so no pixel pair qualifies
anywhere in the 3,802,797 eval pixels. Consequently all valid pixels
fall in the >16 px bin (mean |e| = overall EPE 1.415/1.449/1.459;
nonocc 1.319/1.359/1.368) and bins 0-1..8-16 hold 0 px on all seeds,
both pops. Whether error concentrates at depth boundaries is therefore
NOT MEASURABLE WITH CURRENT ARTIFACTS under the >= 3 px neighbouring
definition — the GT's valid-valid adjacency never reaches the
threshold, which is itself a measured property of this sparse GT, not
an analysis failure. No threshold was lowered to manufacture a result.

### 4. Textureless regions (luminance + cv2.Sobel 3x3 magnitude, window 3x3)

Median gradient magnitude over valid pixels (seed 0/1/2):
25.81/25.81/25.81 (nonocc 25.75); lowest-decile threshold ~4.44.
Method-mean mean |e| below/above median: 1.4145 vs 1.4675 px (per seed
1.396/1.434, 1.414/1.483, 1.433/1.486) — the median split barely
separates error; D1 below/above median 8.42/8.42 (seed 0). Lowest
gradient decile IS worse: mean |e| 1.6862 method-mean
(1.686/1.662/1.711), D1 11.54/11.24/12.18 (seed 0/1/2) vs 8.42 pooled.
Nonocc shows the same shape (lowest decile 1.559/1.589/1.615 vs median
split 1.300/1.337). Error concentrates only in the most textureless
~10%, not broadly below the median.

### 5. Local confidence vs error

Pooled means cross-check D2 to 0.0 (see Gates). Correlation of |e| with
confidence (per seed all-valid / non-occ): top-1 −0.157/−0.122,
−0.154/−0.116, −0.130/−0.088; margin −0.077/−0.058, −0.069/−0.048,
−0.057/−0.035; entropy +0.217/+0.169, +0.206/+0.158, +0.171/+0.120.
Weak in magnitude but consistent in sign on all seeds and both pops:
higher |e| goes with flatter cost distributions. Mean |e| per top-1
decile (seed 0, all-valid; decile 9 EMPTY — top-1 never reaches 0.9):
d0 (205 px) 18.43, d1 13.43, d2 7.87, d3 2.07, d4 1.32, d5 1.01, d6
1.44, d7 1.44, d8 0.99 — steep fall over deciles 0–5, flat after.
Confidence is itself spatially smooth: top-1 neighbour r (seed 0)
all-valid h d=1/4/16: 0.9983/0.9799/0.8534; v: 0.9907/0.8811/0.4565
(nonocc h 0.9985/0.9813/0.8578, v 0.9905/0.8780/0.4424) — smoother
horizontally, same anisotropy as the error field.

Factual statement only (no verdict; step 8 does verdicts): residual
error is strongly spatially clustered relative to the permuted control
(component mass >100 px 0.313 vs 0.0004; neighbour r 0.99 vs 0.03 at
d=1); the structure is anisotropic, persisting further along the
horizontal (disparity) axis (autocorr length 32 h vs 16 v); it survives
removing occluded pixels with no flip (nonocc correlations within
~0.06 of all-valid at every offset; component contrast 0.293 vs 0.000);
and its scale (tens of px) lies well within the model's 389 px
end-to-end (73 px refinement-only) receptive field.

NOT MEASURABLE WITH CURRENT ARTIFACTS: edge-proximity-as-specified
(discontinuity set empty under the >= 3 px neighbouring-valid rule —
see §3); nothing else in this brief was unmeasurable ([144,160)
per-bin pairs flagged NO_PAIRS on 9 px; top-1 decile 9 EMPTY).

## Step 8 — mechanism verdict, architecture gate, report (SYNTHESIS ONLY)

No new experiment, no new measurement. See `DIAGNOSTIC_REPORT.md` (the
full 13-section report) and `mechanism_verdict.json` (machine-readable
verdicts). Headline: gate = NO ARCHITECTURE JUSTIFIED.
