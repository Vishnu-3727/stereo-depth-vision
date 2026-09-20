# C1 PARITY-FAILURE DIAGNOSTIC REPORT (ARM-P seed 1 vs P2A seed 0 control)

Diagnosis only. No training, no retraining, no weight change, no architecture
change, no tolerance change was performed or proposed here. Every claim below
is labelled **OBSERVED**, **INFERRED** or **UNKNOWN**. The word "noise" is not
used for the discrepancy anywhere: what was measured is stated instead.

---

## 1. ORIGINAL FROZEN GATE

**OBSERVED.** The C1 parity gate compares the exported ARM-P seed-1 ONNX
(`stage_c_deploy/armp_stereonet.onnx`) against its PyTorch checkpoint on the
same 5 frozen-contract scenes (`hailo_val` indices 0–4: `000160_10.png` …
`000164_10.png`), with tolerance `max |Δ| < 1e-3 px`
(`stage_c_deploy/C1_EXPORT_PARITY_REPORT.md` Section 5,
`stage_c_deploy/export_armp.py` parity block). Per-scene max |Δ| (px):
000160: 8.3447e-04; 000161: 1.4668e-03; 000162: 8.9645e-04;
000163: 5.7602e-04; 000164: 1.7090e-03. Pooled max |Δ| = **1.708984375e-03 px**,
verdict **FAIL**. This diagnostic re-measured the same quantity with an
independent script and reproduced it exactly (pooled max 1.708984375e-03 px;
per-scene values identical to the last digit), so the FAIL under diagnosis is
the same FAIL. The P2A seed-0 control passed the identical recipe at pooled
max 6.1798e-04 px (`phase2/deploy/P2A_DEPLOYMENT_VALIDATION.json`,
reproduced exactly here too).

---

## 2. DIAGNOSTIC METHOD

**OBSERVED.** Script `stage_c_deploy/diagnostics/c1_parity_diagnostic.py`
(new file; the failed artifacts were only read, never written):

- Both checkpoint hashes and both ONNX hashes re-verified before any
  inference; all four matched (see Provenance). Identical architecture for
  both models (`downsample_levels=3, num_disparities=24,
  cost_volume_shift="right", regression_normalize=True`), so any
  ARM-P-vs-P2A difference is weight-dependent, not structural.
- Inputs: the same 5 scenes via `Kitti2015Stereo(REPO/data/kitti2015,
  split=hailo_val, disparity_scale=256.0, occluded=True)` indices 0–4 with
  `normalize()` from `src/datasets/kitti2015.py`, identical preprocessing for
  both models and both backends.
- PyTorch on CPU, `model.eval()`, `torch.no_grad()`; ORT with
  `CPUExecutionProvider` only. Every difference computed after casting both
  sides to float64. Threads pinned to 1/1 (torch/ORT) for reproducibility.
- Softmax tie statistics reproduce `src/models/stereonet/regression.py`
  exactly at full resolution: `aggregated_cost` (1,24,46,154) is upsampled
  with bilinear `align_corners=True` to (368,1232), standardised across dim=1
  (mean/std + 1e-6, because `regression_normalize=True`), then
  `softmax(-cost)`. Because the distribution is computed at full resolution,
  no approximate low-res cell mapping is used; top-1/top-2/gap/entropy are
  exact per-full-res-pixel quantities of the quantity the regression reads.
- Stage localisation uses `onnx.shape_inference.infer_shapes` shape matching
  against the prescribed shapes, plus a semantic name anchor per stage that
  must additionally pass a value cross-check (sibling-branch or runner-up
  margin ≥ 100×), else the stage is UNKNOWN. A partial localisation is
  reported as-is. `disparity_initial` is UNKNOWN by this method (see §6).
- Occlusion signal: `disp_occ_0` vs `disp_noc_0` from the same dataset class
  (`occluded=True/False`); occluded = occ-valid AND noc-invalid. Where the
  occlusion comparison is not made, occlusion is recorded UNKNOWN — no
  guessing was done.

---

## 3. DISCREPANCY MAGNITUDE DISTRIBUTION

The bins below are **DIAGNOSTIC ONLY and do not redefine the gate**
(tolerance stays 1e-3 px, max-abs). Total pixels pooled: 2,266,880
(5 × 368 × 1232) per model. **OBSERVED.**

ARM-P seed 1 (subject), pooled over 5 scenes:

| \|Δ\| bin (px) | count | fraction |
|---|---|---|
| > 1e-4 | 19968 | 8.81e-03 |
| > 5e-4 | 2524 | 1.11e-03 |
| > 1e-3 | 194 | 8.56e-05 |
| > 1.5e-3 | 4 | 1.76e-06 |
| > 1.7e-3 | 1 | 4.41e-07 |

max 1.7090e-03, mean 7.14e-06, median 2.86e-06; percentiles:
p99 8.77e-05, p99.9 5.26e-04, p99.99 9.78e-04, p99.999 1.33e-03.

ARM-P per scene (max / mean / n>1e-3 / n>1e-4):
000160: 8.34e-04 / 4.45e-06 / 0 / 1655;
000161: 1.47e-03 / 6.34e-06 / 28 / 1645;
000162: 8.96e-04 / 6.19e-06 / 0 / 3152;
000163: 5.76e-04 / 4.83e-06 / 0 / 1551;
000164: 1.71e-03 / 1.39e-05 / 166 / 11965.

P2A seed 0 (control), pooled:

| \|Δ\| bin (px) | count | fraction |
|---|---|---|
| > 1e-4 | 7620 | 3.36e-03 |
| > 5e-4 | 40 | 1.76e-05 |
| > 1e-3 | 0 | 0 |
| > 1.5e-3 | 0 | 0 |
| > 1.7e-3 | 0 | 0 |

max 6.1798e-04, mean 4.43e-06, median 1.91e-06; percentiles:
p99 5.34e-05, p99.9 1.68e-04, p99.99 3.28e-04, p99.999 5.20e-04.

P2A per scene (max / mean / n>1e-4):
000160: 2.21e-04 / 3.15e-06 / 341; 000161: 4.16e-04 / 3.98e-06 / 546;
000162: 3.43e-04 / 4.35e-06 / 1522; 000163: 1.35e-04 / 3.32e-06 / 60;
000164: 6.18e-04 / 7.37e-06 / 5151.

**INFERRED.** The FAIL is a localised tail event, not a broad shift: 194 of
2.27M pixels (0.0086%) exceed 1e-3 px, 85% of them in one scene (000164).
Both models are worst on scene 000164 and the per-scene ordering is similar,
which already suggests a shared, scene-dependent sensitivity with a
weight-dependent amplitude (ratio of pooled maxima ≈ 2.8×).

---

## 4. SPATIAL DISTRIBUTION

**OBSERVED** (ARM-P; P2A has zero pixels > 1e-3, so its >1e-3 rows are null
and its >1e-4 rows are given for the same-scene control):

- Counts: 194 pixels > 1e-3 pooled (28 in 000161, 166 in 000164, 0 in the
  other three scenes); 19968 pixels > 1e-4 pooled.
- Border distance (min over four edges): >1e-3 pixels: mean 23.6, median 28,
  max 90, p90 43 (n=194). **INFERRED** reference: for a uniform pixel draw on
  368×1232, P(distance ≤ 90) ≈ 0.56, so 100%-within-90 is a mild border
  concentration — but the per-edge breakdown (top/bottom vs left/right) was
  not recorded, so which border is **UNKNOWN**.
- Disparity-discontinuity proxy (3×3 gradient magnitude of the PyTorch
  predicted disparity): discrepant median 3.15 vs all-pixel median 0.18
  (≈17×); mean 3.86 vs 0.32; maxima comparable (34.9 vs 35.8).
  **INFERRED.** "Concentrated at edges" as a *comparison*, not an assertion:
  the discrepant set sits at much steeper predicted-disparity gradients than
  background.
- Texture proxy (3×3 gradient magnitude of left-image grayscale intensity):
  discrepant median 0.30 vs all-pixel median 4.25; mean 1.53 vs 10.80.
  **INFERRED.** Discrepant pixels lie in *flatter-than-average* image
  regions, not textured ones.
- Ground truth: only 12/194 discrepant pixels (6.2%) have valid GT, vs 21.1%
  background validity — so most discrepant pixels are GT-invalid
  (**OBSERVED**). The 12 valid ones have GT ≈ 104–105 px (mean 103.96), far
  above the background valid-GT mean of 41.1 (median 36.4).
- Occlusion (disp_occ_0 vs disp_noc_0): 0/194 discrepant pixels are occluded,
  vs 2.45% occluded among background valid-GT pixels. **INFERRED.** No
  occlusion concentration; an occlusion-driven mechanism is refuted by this
  comparison (see §9).

---

## 5. DISPARITY/CANDIDATE DISTRIBUTION

**OBSERVED.** Predicted (PyTorch) disparity at ARM-P discrepant pixels:
mean 46.9, median 51.6, max 84.3, p90 69.3 (n=194) vs all-pixel mean 30.1,
median 20.6, max 126.1. (P2A background: mean 30.2, median 20.5 — same scene
content, as expected.)

Tie statistics from the exact full-res softmax (see §2):

| set | top1 (mean/median) | top2 | top1−top2 gap | entropy |
|---|---|---|---|---|
| ARM-P discrepant (n=194) | 0.303 / 0.290 | 0.142 / 0.125 | 0.161 / 0.148 | 2.468 / 2.552 |
| ARM-P ordinary sample (n=5000, \|Δ\|≤1e-4) | 0.528 / 0.525 | 0.191 / 0.180 | 0.337 / 0.342 | 1.790 / 1.845 |
| P2A ordinary sample (n=5000) | 0.563 / 0.547 | 0.158 / 0.143 | 0.405 / 0.397 | 1.728 / 1.831 |
| P2A discrepant | — (n=0, none exist) | — | — | — |

**INFERRED.** This is the direct test of "does a near-tie amplify a tiny
cost perturbation into a visible disparity difference": discrepant pixels
sit at roughly half the winner-gap (0.16 vs 0.34) and substantially higher
entropy (2.47 vs 1.79) than ordinary pixels of the *same* model on the
*same* scenes. The single worst pixel (000164 y=44 x=1037, |Δ|=1.7090e-03)
has gap 0.0036, entropy 2.43 — a near-exact tie. The max discrepant gap is
0.45, so near-tie is a distributional shift, not a hard rule. P2A's ordinary
pixels are *less* tie-like (gap 0.40) than ARM-P's ordinary pixels
(gap 0.34), consistent with ARM-P's cost landscape being flatter overall.

---

## 6. INTERMEDIATE-TENSOR LOCALIZATION

Criterion (stated BEFORE the numbers, also recorded in the JSON):
a stage is MATERIAL iff max|Δ| > 1e-6 AND relative divergence
(max|Δ| / max|value|) > 1e-7; the first material stage is the earliest in
forward order meeting both. **OBSERVED** results (pooled over 5 scenes;
diffs measured under the instrumented debug graph — see note below):

| PyTorch stage | ARM-P max\|Δ\| / mean\|Δ\| / rel | P2A max\|Δ\| / mean\|Δ\| / rel |
|---|---|---|
| left_features (1,32,46,154) | 2.10e+07 / 1.01e+06 / 5.1e-07 | 4.94e+00 / 3.02e-01 / 3.1e-07 |
| right_features (1,32,46,154) | 1.66e+07 / 9.98e+05 / 3.8e-07 | 5.19e+00 / 3.04e-01 / 3.1e-07 |
| cost_volume (1,32,24,46,154) | 2.94e+07 / 1.41e+06 / 4.1e-07 | 8.63e+00 / 4.25e-01 / 4.2e-07 |
| aggregated_cost (1,24,46,154) | 1.92e+12 / 1.98e+10 / 7.6e-07 | 6.14e+03 / 5.62e+01 / 1.0e-06 |
| disparity_initial (1,1,368,1232) | UNKNOWN | UNKNOWN |
| refinement_residual (1,1,368,1232) | 1.77e-03 / 6.79e-06 / 1.5e-05 | 5.95e-04 / 3.86e-06 / 5.4e-06 |
| disparity_final (1,1,368,1232) | 1.92e-03 / 7.16e-06 / 1.5e-05 | 6.22e-04 / 4.34e-06 / 5.1e-06 |

Activation dynamic range, **OBSERVED** (new measurement
`stage_c_deploy/diagnostics/measure_activation_range.py` →
`activation_range.json`; PyTorch `model(L, R, return_stages=True)` for ARM-P
and P2A, instrumented-ORT for the reference; per-stage `max |value|` pooled
over the same 5 scenes; reference fed the same ImageNet-normalised input the
host-side reproduction uses, because its normalisation is compiler-inserted,
not in the ONNX):

| PyTorch stage | ARM-P max\|value\| | P2A max\|value\| | reference max\|value\| |
|---|---|---|---|
| left_features | 4.12e+13 | 1.58e+07 | 6.28e+01 |
| right_features | 4.35e+13 | 1.67e+07 | 6.39e+01 |
| cost_volume | 7.08e+13 | 2.06e+07 | 8.82e+01 |
| aggregated_cost | 2.53e+18 | 6.16e+09 | 2.39e+01 |
| disparity_initial | 1.59e+01 | 1.64e+01 | 1.05e+01 |
| refinement_residual | 1.15e+02 | 1.10e+02 | 1.36e+02 |
| disparity_final | 1.26e+02 | 1.21e+02 | 1.36e+02 |

Reference anchors: `/res/res.6/Conv_output_0` (left — traced back to
`input.1`), `/res/res.6_1/Conv_output_0` (right — traced to `input.83`),
`/Transpose_output_0`, `/Squeeze_output_0`, `/ReduceSum_output_0`,
`/refine/refine.7/Conv_output_0`, `524`. No stage was UNKNOWN. Caveat: the
reference geometry differs (x16 downsample, 12 candidates; features
(1,32,23,77)) and its cost volume is the degenerate no-op (EXP-010), so this
is a magnitude comparison, not equivalent computation. Scene-4 (`000164`)
per-scene maxima reproduce the manager's independent spot-check to 4 figures
(ARM-P agg 2.0188e18 vs 2.019e18; P2A agg 5.3943e9 vs 5.394e9; all seven
stages likewise), so the scale is confirmed by two implementations.

Stage identities and their evidence (**OBSERVED**):
left/right = `/feature_extractor/output_conv[/_1]/Conv_output_0`, verified by
branch cross-checks (own-branch diff vs sibling-branch diff differs by
~10⁴×: e.g. ARM-P left 1.47e7 vs 5.78e13);
cost_volume = `/cost_volume/Transpose_23_output_0`, unique best of 9
same-shape candidates (runner-up 1.5e15 vs 2.1e7 for ARM-P);
aggregated_cost = `/aggregation/Squeeze_output_0` (unique candidate);
residual = `/refinement/output_conv/Conv_output_0` (decoy margin: true
6.4e-4 vs final-as-decoy 14.9 for ARM-P);
final = `disparity` graph output (structural identity).

`disparity_initial` is UNKNOWN: the `Resize` in the regression path defeats
`onnx.shape_inference` (output shape `(None,None,None,None)`), so every
downstream regression tensor including the suspected
`/regression/ReduceSum_output_0` has an uninferable shape under the
prescribed shape method. A partial localisation is the valid result here.

**INFERRED.** (a) By the predeclared criterion the first material stage is
`left_features` for *both* models — the flag fires immediately because
activations are enormous (PyTorch-side max-abs, **OBSERVED**: ARM-P features
≈ 4.1e13, agg cost ≈ 2.1e18; P2A features ≈ 1.6e7, agg cost ≈ 5.4e9), so any
fp32-level relative divergence trivially exceeds an absolute 1e-6 gate. The
binary flag is therefore uninformative; the informative profile is the
*relative* column: 3–5e-07 at features/volume, growing to ~1e-06 at
aggregated cost, then ~1e-05 at residual/final — the same growth shape in
both models. (b) The residual diff already accounts for essentially the
whole final diff (ARM-P 1.77e-03 vs 1.71–1.92e-03; P2A 5.95e-04 vs
6.18–6.22e-04), so little amplification happens inside refinement; the jump
from ~1e-6-relative cost perturbation to ~1e-3-absolute disparity error
happens in the unobserved normalise→upsample→softmax→soft-argmin readout
(§5 supplies the corroborating tie evidence). The exact Δinitial
decomposition is **UNKNOWN**.

Note: the instrumented graph's final max (1.92e-03 ARM-P) differs slightly
from the unmodified default-session parity number (1.7090e-03); adding
outputs perturbs ORT's fusions — itself consistent with the §7-E fusion
sensitivity finding. Parity verdicts are quoted from the unmodified graph.

---

## 7. ARM-P VS P2A CONTROL

**OBSERVED**, side by side: every quantity in §§3–6 plus §E was computed
identically for both models. Pooled max: 1.7090e-03 (ARM-P) vs 6.1798e-04
(P2A), ratio ≈ 2.8×. Same worst scene (000164) and same runner-up structure.
Same stage-wise *relative* divergence profile (features ≈3–5e-07 → agg
≈1e-06 → final ≈1e-05). Same top-pixel geography: P2A's largest pixels also
cluster in scene 000164 around x≈1029–1030 with gaps 0.006–0.13 and entropy
≈2.7 — the same near-tie signature, weaker amplitude. ARM-P's ordinary
pixels are already slightly more tie-like than P2A's (gap 0.34 vs 0.40).

**INFERRED (answer to Q9).** ARM-P is exposing a **generic exporter
sensitivity that P2A also has in weaker form**, not a specific defect of
these weights: the control shows the same scene selectivity, the same
regional clustering, the same relative-divergence growth, and the same
near-tie correlate. What is weight-specific is the *amplitude path*: ARM-P's
activations are ~10⁶–10⁸× larger than P2A's (features 4e13 vs 1.6e7; agg
cost 2e18 vs 5e9 — **OBSERVED** PyTorch-side scales), so the same ~1e-7–1e-6
relative fp32 divergence becomes a ~1e-3 absolute disparity error for ARM-P
while staying at ~6e-4 for P2A. Why ARM-P training produced 1e13-scale
activations is **UNKNOWN** (training dynamics; outside this task's scope).

---

## 8. LIKELY MECHANISM(S)

**INFERRED.**

- **M1 (primary).** Deterministic fp32 accumulation-order divergence through
  the deep conv graph at extreme activation scales — quantified: ARM-P
  `max |value|` pooled over the 5 scenes is 4.1–4.4e13 at features/volume
  and 2.53e18 at `aggregated_cost`, vs P2A 1.6–2.1e7 and 6.16e9, vs the
  reference 63–88 and 23.9 (**OBSERVED**, §6 table) — with relative
  divergence ~1e-7–1e-6 from the first feature convs growing to ~1e-6 at
  aggregated cost, converted into visible disparity error at the
  standardise→softmax→soft-argmin readout where the affected pixels sit at
  near-ties (gap ≈ 0.16 vs 0.34 background), in flat-texture,
  high-disparity-gradient regions; carried nearly unchanged through
  refinement (residual Δ ≈ final Δ). Stated explicitly: the dynamic range
  does NOT explain the parity gap — relative divergence upstream is
  comparable between the two models (~3e-7 to 1e-6 for both), and fp32
  relative precision is scale-invariant, so the larger absolute magnitudes
  do not by themselves produce the larger final error. The parity gap
  remains attributed to readout amplification at near-ties, as already
  measured (§5). Downstream of the readout all three models live at the
  same scale (residual/final maxima ≈ 110–136 for every model), which is
  consistent with the readout — not the upstream scale — setting the
  disparity error.
- **M2 (modulator, not source).** ORT graph fusion/contraction choices move
  the pooled max at the ~1e-4 level (see §E numbers); they shape which pixel
  is worst, not whether the FAIL occurs.

---

## 9. EVIDENCE AGAINST EACH MECHANISM

Mandatory disconfirmation section; each mechanism gets its refuting test and
the verdict of the measurements. No mechanism is listed without one.

- **Against M1.** It would be refuted if (i) stage-wise relative divergence
  were flat at ~1e-7 with no growth toward the readout, or (ii) discrepant
  pixels had ordinary tie statistics, or (iii) P2A showed no divergence
  anywhere. **OBSERVED:** (i) relative grows 5e-07 → 7.6e-07 → 1.5e-05;
  (ii) discrepant gap/entropy strongly shifted (0.161/2.47 vs 0.337/1.79);
  (iii) P2A shows the same relative profile. M1 survives; its weakest link
  is the unobserved Δinitial (UNKNOWN), so the readout-amplification step
  rests on tie correlates, not a direct input/output perturbation pair.
- **Against M2-as-source.** It would be refuted if disabling all ORT graph
  optimizations removed the FAIL. **OBSERVED:** with
  `ORT_DISABLE_ALL`, ARM-P pooled max is 1.7700e-03 (still FAIL) and P2A is
  7.9346e-04; per-scene maxima move in *both* directions (ARM-P 000160
  8.34e-04→6.29e-04 down, 000163 5.76e-04→1.352e-03 up). M2-as-source is
  **refuted**; M2-as-modulator is supported.
- **Against an occlusion mechanism.** Would be refuted by occ_frac at
  discrepant ≤ background. **OBSERVED:** 0.0 vs 0.0245 — refuted; occlusion
  recorded via disp_occ_0/disp_noc_0, not guessed.
- **Against a high-texture/border-artifact mechanism.** Would be refuted by
  texture-at-discrepant ≤ background. **OBSERVED:** median 0.30 vs 4.25 —
  the high-texture variant is refuted; discrepant pixels are
  flatter-than-average (consistent with matching ambiguity in flat regions).
- **Against an export-correctness bug** (wrong op, dropped branch — the
  STOP-and-report case). Would be *supported* by a structural-scale mismatch
  at some stage (relative ~1, not ~1e-7). **OBSERVED:** every located stage
  matches its PyTorch counterpart to relative ≤ ~1.5e-05 with verified
  identities and ≥100× decoy margins. No export correctness bug found; per
  the brief, none is fixed here (there is nothing to fix).
- **Against nondeterminism/async races.** Would be supported by any
  self-run difference. **OBSERVED:** PyTorch-twice and ORT-twice are both
  bit-identical (max self-diff 0.0). Refuted — the discrepancy is a
  deterministic function of (weights, input, implementation).

---

## 10. WHAT REMAINS UNKNOWN

- The ONNX-side `disparity_initial` (Resize defeats shape inference) and
  hence the exact Δinitial/Δresidual decomposition of the final 1.7e-3.
- Which border (top/bottom/left/right) the mild border concentration belongs
  to (only min-over-edges recorded).
- The exact soft-argmin input perturbation magnitude at discrepant pixels.
- Why ARM-P seed-1 training produced ~1e13-scale feature activations
  (training dynamics; no training analysis in this task).
- Which specific kernel/ordering difference (Conv accumulation order in
  MKL/oneDNN vs ORT, Resize bilinear align_corners rounding, ReduceSum
  reduction order, Softmax) contributes how much — operator-level tracing is
  outside this task. Plausible contributors named with evidence status:
  `Resize` (bilinear, align_corners) — implicated only circumstantially (its
  shape opacity blocks localisation; NOT directly measured);
  `Softmax`-over-disparity-axis — implicated via the tie statistics (§5);
  `ReduceSum` — neither implicated nor exonerated (downstream of Resize,
  unobserved); `Conv` accumulation order — implicated as the carrier of the
  measured ~1e-7–1e-6 relative divergence from the first feature layer
  (matches the expected fp32 accumulation signature at these scales;
  exonerated as a *bug* by the verified stage identities).
- Whether a different ORT/torch version changes the numbers (recorded:
  onnxruntime 1.27.0, torch 2.7.0+cu128 CPU path, onnx 1.22.0, numpy 2.5.1).
- **INFERRED** risk, **UNKNOWN** magnitude: whether ARM-P's activation range
  survives int8 quantisation. The reference's compiled quantisation is
  uniform int8, 8/8/8 across all 242 layers with no mixed precision
  (**VERIFIED:** `docs/hardware_analysis.md` §4a, "Quantisation is uniform
  int8"). An `aggregated_cost` tensor with `max |value|` ≈ 2.5e18
  (**OBSERVED**, §6) must be representable under a per-tensor int8 scale
  before `regression_normalize` standardises it — against a reference whose
  corresponding tensor peaks at 23.9. Whether that range survives the
  quantisation toolchain is **UNKNOWN** and cannot be established without
  the toolchain. No outcome is predicted here; the measurement that would
  settle it is a quantisation-time range/calibration check on these exact
  weights.

---

## 11. ORIGINAL GATE STATUS

**C1 parity gate = FAIL** (max |Δ| 1.709e-3 px > 1e-3 px tolerance), and this
diagnostic did not and cannot change it.

---

## 12. NEXT DECISION

Options for the project owner, without recommendation between them:

- Treat the FAIL as blocking for deployment of these weights and keep the
  target-device work gated behind a passing parity re-measurement.
- Commission a follow-up investigation (a new task, not this one) into the
  unobserved readout step — e.g. named-tensor instrumentation of the
  ReduceSum output, or a perturbation-injection study through the PyTorch
  readout — to convert the tie correlate into a measured perturbation pair.
- In a later decision, examine whether a bare max-abs criterion is the right
  deployment-parity gate (e.g. whether a percentile or region-aware criterion
  better reflects deployment risk). Any such change must be justified in
  writing before a new number is chosen and must leave this FAIL in the
  record. This diagnostic changes no threshold and writes no pass verdict.
- For the project owner's information (not a recommendation, and no
  architecture change is proposed — Stage C forbids it at this point): the
  §10 quantisation-range item is a C3-stage risk worth knowing now rather
  than discovering at quantisation time. ARM-P's `aggregated_cost`
  `max |value|` ≈ 2.5e18 stands against a reference that quantised
  successfully with the corresponding tensor at 23.9; whether that range
  survives the toolchain is UNKNOWN until quantisation is attempted.

---

## PROVENANCE

**OBSERVED** (all values recorded in `c1_parity_diagnostic.json`):

- UTC: 2026-09-19T16:18:13.008105+00:00 (script run; this report written
  after it from the JSON + CSVs). Git HEAD: `58e8a19…` (full
  `58e8a19908ddbd35451652c61aef478f56b51ebd`, same as the C1 export runs).
  Host: Windows-11-10.0.26200-SP0. Python 3.12.9, torch 2.7.0+cu128 (CPU
  path used), onnx 1.22.0, onnxruntime 1.27.0, numpy 2.5.1. Threads:
  torch 1, ORT intra/inter 1. Provider requested and used:
  `CPUExecutionProvider`.
- ARM-P ckpt sha256 `b2f6f5d5…fffeb7454` (full match), P2A ckpt `0868ffd1…`
  (full match), ARM-P ONNX `4277090d…` (full match), P2A ONNX `0995a6c5…`
  (full match) — all four re-verified at script start; mismatch would have
  stopped the run.
- Outputs created (only these, all new under `stage_c_deploy/diagnostics/`):
  `c1_parity_diagnostic.py`, `c1_parity_diagnostic.json`,
  `top_pixels_armp.csv` (20 rows), `top_pixels_p2a.csv` (20 rows),
   `armp_instrumented_debug.onnx`, `p2a_instrumented_debug.onnx`,
   `C1_PARITY_DIAGNOSTIC_REPORT.md` (this file). The `_instrumented_debug`
   files are DEBUG copies, never deployment artifacts. Correction addendum:
   `measure_activation_range.py`, `activation_range.json`,
   `reference_instrumented_debug.onnx` (reference DEBUG copy;
   `reference/onnx/stereonet.onnx` read but never written).
- No existing repo artifact was modified: the working-tree modifications
  (`.gitignore`, `src/models/stereonet/{__init__,cost_volume,regression,
  stereonet}.py`) are exactly the set the C1 parity report recorded as
  pre-existing before the C1 run; this task added only new files.
