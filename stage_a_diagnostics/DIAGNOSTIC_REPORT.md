# STAGE-A Diagnostic Report — P2A vs reference frozen-model gap (D0–D7 synthesis)

SYNTHESIS ONLY. No new experiment was run and nothing new was measured for
this report. Every number below already exists in
`stage_a_diagnostics/*.json` or `stage_a_diagnostics/raw/*.csv`, and every
number cites the file it came from. Where a wanted number does not exist, the
report states NOT MEASURABLE WITH CURRENT ARTIFACTS instead of estimating it.
No `D0`–`D7` JSON, CSV, NPZ or text was modified; `README.md` received an
appended pointer only.

## 1 Executive Summary

- P2A baseline (frozen, 3-seed method mean): EPE **1.4409826 px**
  (`baseline.json`: `method_mean.epe`).
- Frozen reference (ONNX, re-scored at D1 bins): EPE **1.3134471 px**
  (`disparity_bins.json`: `totals.reference_epe`).
- Residual gap: **0.1275355 px** (`disparity_bins.json`:
  `totals.total_gap_px`).
- Diagnostics completed: D0 (integrity + baseline), D1 (per-disparity
  decomposition), D2 (cost distribution, CONFOUNDED readout superseded),
  D2b (corrected readout, 3 protocols), D3 (matching representation),
  D4 (aggregation), D5 (refinement), D6 (high-disparity reassessment),
  D7 (spatial/context). Nine JSON artifacts plus raw dumps; see
  `stage_a_diagnostics/README.md` for the per-step record.
- Mechanism identified: **no single architectural bottleneck was isolated**.
  Mechanism 7 ("no single architectural bottleneck") is SUPPORTED, mechanism
  6 (supervision distribution) is SUPPORTED as a data-side contributor, and
  the architecture gate returns **NO ARCHITECTURE JUSTIFIED** (Section 11).

## 2 Integrity Verification

- D0 gate: **PASS** (`integrity.json`: `gate`; `baseline.json`: `gate`).
- Seed-0 EPE 1.4149796 px reproduced to 0.00e+00; 3-seed mean 1.4409826 px
  reproduced to 2.0e-11 (tolerance 1e-3 px) (`integrity.json`:
  `seed0_epe_reproduces`, `mean3_epe_reproduces`; `baseline.json`:
  `reproduced`, `method_mean`).
- Checkpoints: all three seed SHA-256 match both the P2A record and the
  P2A eval-best; strict load with 0 missing / 0 unexpected keys (70 matched)
  on all seeds (`integrity.json`).
- Config: 397954 params, 3 downsample levels, 24 disparities, right shift,
  regression-normalize true, no BatchNorm modules (`integrity.json`).
- Contract: frozen-eval contract guard imported unmodified; contract match
  true on all seeds; 3802797 valid pixels all seeds; GT scale 256.0;
  occluded GT `disp_occ_0` (`integrity.json`; `baseline.json`).
- Reference re-scored at D1 bins by the `reference_strata.py` mechanism
  (no interpolation): 1.3134471 px, contract match true
  (`disparity_bins.json`: `totals`, `conventions`).

## 3 Per-Disparity Error Decomposition

- The gap attribution column **SUMS to 0.1275355 px (diff -2.2e-16): CLOSES**
  (`disparity_bins.json`: `totals.gap_attribution_closes`).
- The gap is SPLIT, with no single dominant bin
  (`disparity_bins.json`: `gap_attribution`):
  - Below 64 px: contributions 0.0034340 + 0.0245368 + 0.0064988 +
    0.0054086 = 0.03988 px, i.e. **~31% of the gap on 95.4% of pixels**.
  - 64–128 px: 0.0153210 + 0.0125415 + 0.0262096 + 0.0296224 = 0.08369 px,
    i.e. **~66% of the gap on 4.6% of pixels**.
  - >= 128 px: 0.0039600 + 0.0000028 = 0.00396 px, i.e. **~3% of the gap**.
- Top 3 bins by gap contribution: [112,128) 0.029622, [96,112) 0.026210,
  [16,32) 0.024537 px (`disparity_bins.json`:
  `top3_by_gap_contribution`).
- Bins [160,184) and [184,inf) are EMPTY (0 px); [144,160) holds 9 px
  (EPE reported, regression withheld as not meaningful)
  (`disparity_bins.json`: `per_seed_bins`, `method_mean_bins`).
- Pooled P2A EPE per bin (method mean) rises steeply with disparity while
  pixel mass collapses: e.g. [16,32) 1.2518 px at 30.7% mass vs [112,128)
  15.3675 px at 0.31% mass (`disparity_bins.json`: `method_mean_bins`).

## 4 Cost / Readout Analysis

- Distribution statistics (sound, NOT confounded; `cost_distribution.json`:
  `seeds.{0,1,2}.distribution_pooled`; method means in
  `stage_a_diagnostics/README.md` D2): entropy 1.8531 nats, top-1 0.5051,
  top-1 margin 0.3099, variance 25.206 (candidate-index units), mean
  |argmax − soft-argmin| 2.0105 candidates.
- GT-candidate rank: rank-1 fraction 0.1330 pooled (per seed 0.1539 /
  0.1301 / 0.1150); mean mass at nearest-GT candidate 0.1786; mass in
  gt_cand ± 1 neighbourhood 0.3969 (`cost_distribution.json`;
  `stage_a_diagnostics/README.md` D2).
- RANGE REFUTED (first of two independent refutations): fraction of valid
  pixels with gt_cand outside [0,23] is 0.000000 on all seeds
  (`cost_distribution.json`: `distribution_pooled.frac_gt_out_of_range`).
- Original D2 readout numbers are **CONFOUNDED and SUPERSEDED**
  (`cost_distribution.json`: `readout_comparison_status`;
  `cost_readout_corrected.json`: `supersedes`). Cause: sparse substitution
  at valid pixels only (~21% of image) through the frozen dilated
  refinement; proof: the exact-GT continuous oracle scored 5.7061 px vs a
  1.4410 px baseline (`stage_a_diagnostics/README.md` D2b).
- Corrected Protocol A (primary, zero confound; frozen residual held
  fixed; gate soft-reproduces-baseline to -9.45e-10 / +1.02e-11 /
  +6.47e-11 px, `cost_readout_corrected.json`: `gates`): pooled
  method-mean EPEs — baseline 1.4410, A_soft 1.4410, A_argmax 2.6817
  (+1.2408), A_topk2 2.2981 (+0.8571), A_topk3 1.9946 (+0.5536),
  A_topk4 1.8739 (+0.4329), A_continuous 1.8132 (+0.3722)
  (`cost_readout_corrected.json`: `method_mean_epe`,
  `method_mean_epe_delta_vs_baseline`). Pooled, the readout is NOT
  independently improvable: the exact GT candidate with the model's own
  residual scores 1.8132 vs baseline 1.4410 and loses in every bin below
  64 px (`cost_readout_corrected.json`: `seeds.0.per_bin`).
- BUT at GT>=96 (34152 px, 0.898% of valid) the oracles gain: continuous
  10.3996 vs baseline 12.4149 (**-2.0153 px**), top-4 11.5897 (-0.8252),
  top-3 11.9862 (-0.4287); per-bin gains from [64,80) upward on seed 0:
  2.780 vs 3.060, 4.255 vs 4.861, 8.101 vs 9.596, 10.813 vs 13.535,
  27.888 vs 33.785, 32.628 vs 38.689
  (`cost_readout_corrected.json`: `method_mean_gt_ge_96_epe`,
  `seeds.0.per_bin`, `seeds.0.gt_ge_96`).
- Protocol B (dense argmax, patch-free): 17.9440 px method-mean — far
  worse than either sparse baseline; the argmax field is too noisy for
  the refinement (`cost_readout_corrected.json`: `method_mean_epe`).
- Protocol C (patchiness control): C1 sparse 3.1905 vs C2 dense 5.4580;
  sparse-substitution effect (C1 − C2) **-2.2676 px**
  (`cost_readout_corrected.json`:
  `method_mean_sparse_damage_C1_minus_C2_px`). The identical +0.5 readout
  change scores ~2.3 px differently by substitution pattern alone — this
  sensitivity dominated the confounded D2 numbers.
- IDENTIFIABILITY LIMIT (stated, carries to the verdict): Protocol A holds
  the residual fixed, so it CANNOT separate "readout is fine" from
  "readout and refinement are co-adapted and would need joint retraining"
  (`stage_a_diagnostics/README.md` D2b).
- NOT MEASURABLE WITH CURRENT ARTIFACTS: dense GT-oracle maps through the
  refinement (GT is sparse by nature)
  (`cost_readout_corrected.json`: `unmeasurable`).

## 5 Matching Analysis

- Architecture verified: `stages["left_features"]` / `right_features` are
  (1, 32, 46, 154) at stride 8; `feature_normalize` FALSE (raw extractor
  output); 24 shift-right candidates; A/B coincide under L1 exactly (max
  abs diff 0.0, so headline B uses L2/RMS to stay distinct)
  (`matching_diagnostic.json`: `architecture_verification`,
  `ab_coincide_under_l1_all_seeds`; `stage_a_diagnostics/README.md` D3).
- Sampling: 59431 GT cells/seed, 1313 dropped by left-edge rule, 79326
  pairs removed, 58048 defined cells remain (174144 pooled — not low-n);
  GT>=96 holds 393 cells/seed
  (`matching_diagnostic.json`: `counts`; `method_mean.A_L1.gt_ge_96`).
- Pooled method-mean (`matching_diagnostic.json`: `method_mean`):

  | rep | rank-1 | rank<=3 | margin frac>0 | overlap | gap |
  |---|---|---|---|---|---|
  | A (L1) | 0.6469 | 0.8412 | 0.7320 | 0.0355 | 1.7757 |
  | B (L2/RMS) | 0.6528 | 0.8450 | 0.7386 | 0.0350 | 1.8036 |
  | C (L2-norm + L1) | 0.6286 | 0.8210 | 0.7035 | 0.0242 | 1.9256 |
  | D (grp-corr G=8, negated) | 0.2501 | 0.5358 | 0.3202 | 0.1819 | 1.3308 |

- Pooled, B edges A on rank-1/rank<=3/margin-sign while C has the best
  overlap and gap; D separates worst on every scale-free metric. Raw-scale
  mean margins (A 7.57e6, B 9.60e6, C 0.0162, D_G8 −6.49e15) are NOT
  comparable across representations (frozen raw features have ~1e6–1e7
  magnitude); only rank/margin-sign/overlap/gap compare
  (`matching_diagnostic.json`: `method_mean`;
  `stage_a_diagnostics/README.md` D3).
- At GT>=96 the GT-candidate margin is negative in the majority in **ALL
  FOUR** representations: margin-frac>0 A 0.4436 / B 0.4580 / C 0.4427 /
  D_G8 0.2188; rank-1 A 0.3647 / B 0.3681 / C 0.3783 / D_G8 0.1730
  (`matching_diagnostic.json`: `method_mean.*.gt_ge_96`).
- KNOWN GAP (not a refutation): D3's representation D was group-AVERAGED,
  which rescales the full channel dot by 1/G and made it G-invariant —
  ranks and overlaps IDENTICAL for G4/G8/G16 by construction (rank-1
  0.2501 all three), margins scaling as 1/G
  (`matching_diagnostic.json`: `method_mean.D_G4/D_G8/D_G16`). D3 did NOT
  test group-wise correlation as GwcNet uses it (G preserved as channels
  into aggregation). ARM-W (group-wise cost) was separately REFUTED in
  Phase 1 at ~1.776 (`stage_a_diagnostics/README.md` D3).
- The model does NOT score candidates with any scalar reduction; it feeds
  the full 32-channel signed-difference volume into a trained 3D
  aggregation. A representation winning here establishes MECHANISM
  PLAUSIBILITY ONLY (`matching_diagnostic.json`: `honesty_constraint`,
  `interpretation`).
- NOT MEASURABLE WITH CURRENT ARTIFACTS: what a trained aggregation would
  do with any of these representations
  (`matching_diagnostic.json`: `unmeasurable`).

## 6 Aggregation Analysis

- PRE proxy reproduces D3 A_L1 rank-1 per seed to 0.00e+00
  (0.6565/0.6508/0.6335): PRE here IS the D3 A_L1 representation,
  recomputed from the frozen volume
  (`aggregation_diagnostic.json`: `d3_crosscheck`,
  `method_mean.POOLED.pre.rank1_frac` 0.6469).
- PRE vs POST, method-mean (`aggregation_diagnostic.json`: `method_mean`):
  - POOLED (58048/seed): rank-1 0.6469 → 0.1572 (Δ −0.4897, same sign
    all seeds); margin-frac 0.7320 → 0.5247 (Δ −0.2073); overlap
    0.0355 → 0.0101; gap 1.7757 → 2.2801; rank<=3 flat-to-up (+0.0236,
    +0.1462 in [0,16)).
  - GT>=96 (393/seed): rank-1 0.3647 → 0.1018 (Δ −0.2629);
    margin-frac 0.4436 → 0.3961 (Δ −0.0475); overlap 0.1781 → 0.0419.
  - Rank-1 delta negative in every populated D1 bin and every regime
    incl. TEXLO/TEXHI and LOW/HIGH.
  - Margin-frac gains only in low-n OCC (+0.0522, n=134/seed), HIGH_OCC
    (+0.0435, n=23/seed) and at EDGE (+0.1149, n=676/seed).
- Occlusion does NOT explain the high-disparity matching failure:
  GT96_NONOCC (391 cells/seed, occlusion fully removed) margin-frac PRE
  0.4433 → POST 0.3981 — the negative-margin majority persists without
  occlusion — while GT96_OCC holds 2 cells/seed (6 pooled)
  (`aggregation_diagnostic.json`: `method_mean.GT96_NONOCC`,
  `method_mean.GT96_OCC`).
- POST argmax field is spatially coherent pooled (0.8670 neighbour
  agreement, mean |diff| 0.1607 candidates; full-4-all-agree 0.5733), less
  so at edges (0.7114) and GT96_NONOCC (0.7821, mean |diff| 0.5543)
  (`stage_a_diagnostics/README.md` D4).
- POST rank-1 0.1572 sits in the same range as the independent D2
  readout-level rank-1 0.1330, measured differently
  (`stage_a_diagnostics/README.md` D4).
- SAME IDENTIFIABILITY LIMIT AS D2b (carries to the verdict): because
  `disparity_initial` is co-adapted with the residual and is not a
  disparity estimate, POST-ranking the GT candidate first scores the model
  against a property it was never trained to have. Loss vs repurposing is
  NOT IDENTIFIABLE. This is NOT reported as aggregation being the
  bottleneck (`stage_a_diagnostics/README.md` D4).
- Entropies collapse (POST valid-only softmax exactly one-hot, entropy
  0.0; aggregated costs span ~1e9) so the entropy column is reported but
  uninformative at raw scale; readout-level entropy (~1.85 nats,
  standardised) is D2's quantity
  (`aggregation_diagnostic.json`: `method_mean.POOLED`;
  `stage_a_diagnostics/README.md` D4).

## 7 Refinement Analysis

- Units: `disparity_final = relu(disparity_initial + refinement_residual)`
  with init in CANDIDATE units and final in pixels; headline convention is
  COARSE_SCALED = 8x init
  (`refinement_diagnostic.json`: `conventions`, `headline_coarse_convention`).
- Headline (method-mean pooled, `refinement_diagnostic.json`:
  `method_mean_pooled`): coarse-scaled EPE 8.7944 → refined 1.4410
  (**−7.3534 px**, D1 −81.79 pp); frac improved 0.9507, worsened 0.0493;
  mean improvement on helped 7.895 px vs mean degradation on harmed
  3.105 px; residual signed mean +30.287 px (≥ 0 nearly everywhere);
  **relu clamp fraction 1.8e-07** — the relu does no work.
- Of four interpretation options, the numbers fit (1) systematically
  improves, with elements of (2): GT 64–96 helped least, occluded less
  than non-occluded (0.786 vs 0.957; refined EPE 5.81 vs 1.32 px)
  (`refinement_diagnostic.json`: `seeds.0.regimes`... `regimes`;
  `stage_a_diagnostics/README.md` D5).
- Weakest band: [64,80) frac improved 0.473/0.552/0.495 (seeds 0/1/2) —
  on seed 0 refinement HARMS more pixels than it helps there; the single
  bin-level EPE regression is seed-2 [64,80) (2.96 vs 2.85 px)
  (`refinement_diagnostic.json`: `seeds.{0,1,2}.per_bin`).
- Disparity-dependent mis-scaling (seed 0,
  `refinement_diagnostic.json`: `seeds.0.per_bin.*.oracle_scale`):
  alpha* 0.974 ([0,16)), 1.022/1.041/1.039 below 64 px, 1.104
  ([64,80)/[80,96)), 1.319 ([96,112)), 1.631 ([112,128)), 4.003
  ([128,144), n=1071), 3.71 ([144,160), n=9).
- Global headroom ~0.008 px: oracle alpha* 1.018 → EPE 1.4330 vs 1.4410;
  oracle beta* +0.095 → 1.4340 (gain 0.007); EPE at alpha=8 is 38.6 px
  (`refinement_diagnostic.json`: `method_mean_pooled`).
- The mass-weighted per-bin oracle-alpha gain (sum frac_px ×
  (epe_refined − alpha_epe)) is 0.0481 / 0.0263 / 0.0289 px at seeds
  0/1/2 — about 6x the global figure at seed 0 — but it is a GT-FITTED
  ORACLE CEILING, not an achievable improvement
  (`stage_a_diagnostics/README.md` D5).
- Correction tracks confidence: largest |residual| where cost is most
  ambiguous (mean |resid| 71 px at lowest top-1 decile), highest success
  rate and largest mean improvement where most peaked (frac improved
  0.922 → 0.963 → 0.989 across bulk deciles 3–8)
  (`stage_a_diagnostics/README.md` D5).
- Sign-only literal formula scores ~7e-05 but is DEGENERATE (collapses to
  GT given units mismatch); the units-consistent supplementary variant
  scores 16.80 px — worse than the 8.79 px coarse baseline — so
  sign-vs-magnitude decomposition is frame-dependent and ill-posed
  (`refinement_diagnostic.json`: `method_mean_pooled`;
  `stage_a_diagnostics/README.md` D5).

## 8 High-Disparity Analysis

- Regimes, method-mean (`high_disparity.json`: `regimes`):
  - GT>=64: 175472 px (4.6143%), EPE 5.1478, signed −3.73, slope 0.751
    (per-seed 0.772/0.719/0.763), R2 0.634; reference EPE 3.248, slope
    0.930, signed −0.10.
  - GT>=96: 34152 px (0.8981%), EPE **12.4149**, signed **−11.72**,
    slope **0.570** (0.623/0.537/0.550), R2 0.091; reference EPE 5.757,
    slope **0.727**, signed −2.83. Reference better calibrated there.
  - GT>=128: 1080 px (low-n), EPE 33.3687, signed −33.37; pooled slope
    0.814 but per-seed 0.502/0.848/1.093 with R2 0.030 — regression
    meaningless; D1 [128,144) bin slope is −7.795
    (`disparity_bins.json`: `method_mean_bins`).
  - GT>=184: EMPTY (0 px).
- Five explanations, separately (`high_disparity.json`):
  1. RANGE — **REFUTED** (second independent refutation of D2): 0.000000
     outside [0,23] overall and in every regime; max gt_cand 19.13 < 23;
     max prediction never exceeds max GT on any seed (`range`).
  2. CALIBRATION — SUPPORTED: regime oracle-alpha* (GT-fitted ceilings)
     GT>=64 1.146/1.100/1.111 (gain 0.30/0.14/0.17 px), GT>=96
     1.538/1.494/1.446 (gain 2.06/1.78/1.30 px), GT>=128
     3.993/3.872/4.402 (gain 19.05/20.85/27.06 px, low-n); slope falls
     0.751 → 0.570 from >=64 to >=96 while signed error grows −3.73 →
     −11.72 (`calibration`).
  3. MATCHING — SUPPORTED at >=64/>=96: GT>=64 PRE rank-1
     0.518/0.505/0.490, POST 0.154/0.117/0.111; GT>=96 PRE rank-1
     0.379/0.356/0.359 with margin-frac>0 0.440/0.443/0.448 — negative
     majority on all seeds, before and after aggregation; persists with
     occlusion removed (Section 6). GT>=128/>=184 EMPTY at feature
     resolution — NOT MEASURABLE WITH CURRENT ARTIFACTS (`matching`).
  4. OCCLUSION — SUPPORTED at >=128; WEAK below it: GT>=64 occluded
     18.15% (error-mass share ~39/37/36%); GT>=96 occluded 26.52%
     (9058 px, share ~49/50/49%) but non-occluded EPE stays high
     (8.06/9.21/8.38), so occlusion does not account for the regime;
     GT>=128: 1071/1080 px (**99.17%**) occluded — essentially
     correspondence-invalid (`occlusion`).
  5. SUPERVISION COVERAGE — SUPPORTED at >=96; WEAK elsewhere: raw
     training GT>=96 0.228% vs eval 0.898% (3.9x under); GT>=64 3.93%
     vs 4.61%; GT>=128 0.0022% vs 0.0284%. After P2A augmentation
     (Monte-Carlo emulation of the exact recipe, 800 augmented samples,
     24.6M valid px): GT>=96 effective **0.444%** vs eval 0.898% —
     still **~2x under-covered**; GT>=64 effective 6.01% vs 4.61%
     (covered); GT>=128 effective 0.0438% vs 0.0284% (over-covered after
     augmentation, but eval-low-n). Per-bin effective vs eval:
     [96,112) 0.338% vs 0.559%, [112,128) 0.063% vs 0.311%,
     [128,144) 0.034% vs 0.028% (`supervision_coverage`). This is a DATA
     property, not an architecture property.

## 9 Spatial / Context Analysis

- Gates: seed-0 EPE reproduces D0 to 0.00e+00; D2 cross-check diffs ≤
  2.9e-11 seed 0 (1e-6 all seeds) (`spatial_context.json`:
  `seeds.0.d2_crosscheck`; `stage_a_diagnostics/README.md` D7).
- Receptive field (stated arithmetic):
  feature extractor 237 px at stride 8; aggregation 301 (4-filter) /
  317 (5-conv actual); refinement-only 73 px; **headline end-to-end 389
  full-res px** (373 on 4-filter count)
  (`spatial_context.json`: `receptive_field`).
- Neighbouring-error correlation (method-mean; per-seed ordering
  identical): d=1 h/v real 0.9912/0.9846 vs permuted 0.031/0.032;
  d=4 0.9513/0.8353; d=16 0.6674/0.4262 — real-vs-control gap 0.29–0.96
  at every offset/orientation/population, surviving occlusion removal
  (nonocc within ~0.06 of all-valid everywhere)
  (`spatial_context.json`: `seeds.0.correlations`;
  `stage_a_diagnostics/README.md` D7).
- Autocorrelation length (first d with r below half r(d=1)), identical
  all 3 seeds, both pops, e and |e|: **32 px horizontal / 16 px
  vertical** (seed-0 all-valid e: 0.991→0.946→0.836→0.645→0.422 at
  d=1/4/8/16/32) (`spatial_context.json`:
  `seeds.0.autocorr_length`). Against the 389 px end-to-end RF (73 px
  refinement-only): the error structure sits WELL INSIDE the existing
  receptive field. Long-range context therefore has WEAK measured
  support — argued from this model's own numbers, not literature.
- Clustering (|e|>3, 8-connectivity): 31.31% of error mass in components
  >100 px real (per seed 0.3160/0.3115/0.3118) vs 0.0004 permuted
  (0.00031/0.00000/0.00096); nonocc 0.2927 vs 0.0000; ~3.5x fewer but far
  larger regions than control (per-image components 1851.5 vs 6604.4
  seed 0; max 22037 vs 104); median size 1.0 px both
  (`spatial_context.json`: `seeds.0.components`;
  `stage_a_diagnostics/README.md` D7).
- Edge proximity — DEGENERATE AS SPECIFIED: with the >= 3 px
  neighbouring-valid rule the discontinuity set is EMPTY (dataset-level
  max adjacent valid-valid GT diff is 2.0 px); all valid pixels fall in
  the >16 px bin. NOT MEASURABLE WITH CURRENT ARTIFACTS under that
  definition — a measured property of this sparse GT, not an analysis
  failure (`spatial_context.json`: `seeds.0.edge_proximity`;
  `stage_a_diagnostics/README.md` D7).
- Texture: median split barely separates error (1.4145 vs 1.4675 px
  method-mean); only the lowest gradient decile is worse (1.6862 px,
  D1 11.54/11.24/12.18) — concentration in the most textureless ~10%,
  not broadly below median (`spatial_context.json`:
  `seeds.0.texture`).
- Confidence vs error: |e| correlation with top-1 −0.157/−0.154/−0.130,
  margin −0.077/−0.069/−0.057, entropy +0.217/+0.206/+0.171 per seed
  (all-valid) — weak but consistent in sign; confidence itself spatially
  smooth with the same horizontal anisotropy
  (`stage_a_diagnostics/README.md` D7).

## 10 Mechanism Verdict

One label per mechanism from SUPPORTED / PLAUSIBLE / WEAK / REFUTED /
NOT IDENTIFIABLE. Detail and per-number citations are in
`mechanism_verdict.json`.

| # | Mechanism | Verdict | Evidence (source) | Counter-evidence (source) |
|---|---|---|---|---|
| 1 | Matching representation | **PLAUSIBLE** | GT>=96 PRE margin negative in majority in all 4 reps (0.4436 A; D_G8 0.2188); rank-1 0.3647; persists occlusion-removed (0.4433→0.3981) (`matching_diagnostic.json`, `aggregation_diagnostic.json`) | No rep dominates pooled (B 0.6528>A 0.6469>C 0.6286>>D 0.2501); D3-D G-invariant by construction — true group-wise correlation NOT tested (KNOWN GAP); proxy≠model scoring; trained outcome NOT MEASURABLE; ARM-W refuted Phase 1 ~1.776 (`matching_diagnostic.json`, `stage_a_diagnostics/README.md` D3) |
| 2 | Cost-volume ambiguity / readout | **PLAUSIBLE** | GT>=96 oracle gains to −2.02 px (continuous), −0.83 (top-4); per-bin gains from [64,80) up (`cost_readout_corrected.json`) | Pooled oracles LOSE (1.8132 vs 1.4410; lose every bin <64 px = ~99% pixels); D2 CONFOUNDED/superseded (C1−C2 −2.27 px); Protocol A co-adaptation limit (`cost_readout_corrected.json`, `cost_distribution.json`, `stage_a_diagnostics/README.md` D2b) |
| 3 | Cost aggregation / context | **NOT IDENTIFIABLE** | POST rank-1 0.1572 vs PRE 0.6469 (Δ −0.4897, all seeds/bins/regimes); margin-frac −0.2073 (`aggregation_diagnostic.json`) | Overlap halves (0.0355→0.0101), gap rises (1.7757→2.2801), rank<=3 flat-to-up; POST argmax coherent (0.8670); SAME co-adaptation limit — disparity_initial is not a disparity estimate; loss vs repurposing NOT IDENTIFIABLE; not the bottleneck (`aggregation_diagnostic.json`, `stage_a_diagnostics/README.md` D4) |
| 4 | Refinement | **REFUTED** as the mechanism explaining the residual gap. Refinement is not healthy in every regime — it mis-scales with disparity and harms more pixels than it helps at [64,80) — but neither observation can account for the 0.1275355 px gap: global rescale headroom is 0.008 px, and the per-bin 0.0481/0.0263/0.0289 px figure is a GT-FITTED ORACLE CEILING, not achievable | Per-bin alpha* 0.974→1.319→1.631→4.003 with disparity; [64,80) frac_improved 0.473 (harms more than helps) (`refinement_diagnostic.json`) | Systematically improves (95.07%, −7.35 px); relu no-op (1.8e-07); global headroom ~0.008 px; per-bin 0.0481/0.0263/0.0289 px is a GT-FITTED ORACLE CEILING, not achievable (`refinement_diagnostic.json`, `stage_a_diagnostics/README.md` D5) |
| 5a | Residual high-disparity behaviour, GT 64–128 px | **SUPPORTED** as the locus where residual error concentrates; NOT an independent architecture mechanism — causes attributed to mechanisms 1, 2 and 6 (attribution is not refutation) | GT 64–128 band holds ~66% of the 0.1275355 px gap (0.08369 px) (`disparity_bins.json`: `gap_attribution`); GT>=96 slope 0.570 with signed bias −11.72 px vs reference 0.727 / −2.83 (`high_disparity.json`: `regimes`); GT>=96 margin negative in majority in all four representations (A 0.4436 / B 0.4580 / C 0.4427 / D_G8 0.2188) (`matching_diagnostic.json`: `method_mean`), occlusion refuted as the explanation (GT96_NONOCC 0.4433→0.3981) (`aggregation_diagnostic.json`); GT>=96 readout oracles gain up to −2.02 px (continuous) (`cost_readout_corrected.json`); effective training mass 0.444% vs eval 0.898%, ~2x under (`high_disparity.json`: `supervision_coverage`) | Causes attributed to mechanisms 1 (matching, PLAUSIBLE), 2 (readout, PLAUSIBLE with NOT IDENTIFIABLE co-adaptation limit) and 6 (supervision, SUPPORTED DATA fix) rather than an independent architecture mechanism |
| 5b | Residual high-disparity behaviour, GT>=128 px | **REFUTED** as an architecture-fixable cause: GT>=128 only | — (as architecture-fixable cause) | 99.17% occluded → correspondence-invalid, cannot be fixed by architecture; worth 0.0040 px (~3%) of gap; matching there NOT MEASURABLE (`high_disparity.json`, `disparity_bins.json`) |
| 6 | Training / supervision distribution | **SUPPORTED** | Effective GT>=96 mass 0.444% vs eval 0.898% (~2x under; 3.9x raw); reference slope 0.727/bias −2.83 vs P2A 0.570/−11.72 at GT>=96 (`high_disparity.json`) | DATA property, not architecture; GT>=64 covered (6.01% vs 4.61%); does not explain sub-64px ~31% (`high_disparity.json`, `stage_a_diagnostics/README.md` D6) |
| 7 | No single architectural bottleneck | **SUPPORTED** | Gap splits 31/66/3 with no dominant bin and CLOSES (diff −2.2e-16); range REFUTED twice (0.000000; max 19.13<23); readout+aggregation both NOT IDENTIFIABLE-limited; D7 structure (32h/16v) inside 389 px RF; >=128 correspondence-invalid (all JSONs above) | — |

## 11 Architecture Gate

**NO ARCHITECTURE JUSTIFIED**

Justification from the table alone:

- ~31% of the gap sits below 64 px where every diagnostic looks healthy
  (Section 3; oracles lose in every bin below 64 px, Section 4).
- The largest single identified effect — supervision coverage ~2x under
  at GT>=96 — is a DATA fix, not an architecture change (mechanism 6).
- The two architecture-shaped candidates, readout (mechanism 2) and
  aggregation (mechanism 3), BOTH carry an explicit NOT IDENTIFIABLE
  limit from the same readout–residual co-adaptation.
- D7 refutes the usual long-range-context argument on this model's own
  numbers (32h/16v structure vs 389 px RF; mechanism 7).
- The most extreme regime (GT>=128) is 99% occluded,
  correspondence-invalid, and worth 3% of the gap (mechanism 5).

The evidence does not isolate a mechanism. NO ARCHITECTURE JUSTIFIED is
the correct scientific answer. No candidate is invented to keep momentum.
No architecture ranking by numerical score is made; nothing here is
called Phase 3.

## 12 If Justified (candidates)

Not applicable — the gate returned NO ARCHITECTURE JUSTIFIED, so no
architecture candidates are listed. The Hailo screening rule
(40 epochs × 2 seeds, P2A recipe, ONE intervention, method-mean
1.4409826 baseline from `baseline.json`, preregistered ≥ 0.03 px gain,
GT<64 must not regress; Hailo toolchain UNVERIFIED — never claim
compatibility; never let a compiler pass optimise away P2A's 23 Pad
clusters) is recorded here for the controlled experiments in Section 13.

## 13 If Not Justified (what remains unknown)

For each unknown, the ONE diagnostic or controlled experiment that would
resolve it (specific enough to execute; unknowns also recorded in
`mechanism_verdict.json`):

1. **U1 — readout–refinement co-adaptation.** Unknown: whether the
   residual compensates specifically for soft-argmin, i.e. whether an
   alternative readout would win if the refinement were allowed to adapt.
   Frozen weights cannot answer this (`stage_a_diagnostics/README.md`
   D2b). Resolving experiment: P2A recipe, freeze the cost volume,
   retrain ONLY the refinement from dense alternative inits (soft-argmin
   vs sharpened-softmax vs top-k-detached); or retrain the full model
   with an auxiliary GT-candidate loss. Compare method-mean EPE against
   1.4409826 (`baseline.json`). A jointly-retrained alternative readout
   win confirms co-adaptation and makes readout actionable.
2. **U2 — true group-wise correlation effect.** Unknown: what a trained
   aggregation does with group-wise correlation with G preserved as
   channels (GwcNet-style). D3's D was group-averaged and G-invariant by
   construction, so this was never tested
   (`matching_diagnostic.json`: `method_mean.D_G4/D_G8/D_G16`).
   Resolving experiment: retrain a P2A variant replacing the 32-channel
   signed-difference volume with group-wise correlation (G=8, G preserved
   into aggregation), identical recipe and seeds; screen 40 epochs ×
   2 seeds vs 1.4409826, threshold ≥ 0.03 px, GT<64 must not regress.
3. **U3 — the ~31% sub-64px gap with no isolated cause.** Unknown: what
   explains the below-64px gap (0.03988 px of 0.1275355 px on 95.4% of
   pixels, `disparity_bins.json`: `gap_attribution`) where oracles lose
   in every bin (`cost_readout_corrected.json`: `seeds.0.per_bin`).
   Resolving diagnostic (no new training, existing raw dumps only):
   fine-grained below-64 attribution — reference-vs-P2A signed
   error/slope per narrow bin plus texture/confidence deciles restricted
   to GT<64. If still diffuse, a capacity-matched retrain or
   low-disparity loss-reweighting experiment.
4. **U4 — edge proximity.** Unknown and NOT MEASURABLE WITH CURRENT
   ARTIFACTS: the ≥ 3 px discontinuity set is empty (max adjacent
   valid-valid GT diff 2.0 px;
   `spatial_context.json`: `seeds.0.edge_proximity`). Resolving
   diagnostic: a denser GT source, or a relaxed-threshold sensitivity
   analysis that reports threshold-dependence rather than a single
   number.
5. **U5 — causal test of the supervision hypothesis (DATA fix, not
   architecture).** Unknown: whether eval-matched coverage closes the
   high-disparity gap. Resolving experiment: retrain with GT>=96
   oversampled to ~0.9% effective mass under the identical recipe;
   success = GT>=96 EPE/slope moves toward the reference (5.757 /
   0.727, `high_disparity.json`: `regimes.ge96.reference`) with no GT<64
   regression.
