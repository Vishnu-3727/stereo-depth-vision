# STAGE C DR-0 — ACTIVATION DYNAMIC-RANGE AUDIT

Diagnosis only. No training, no fine-tuning, no optimizer, no backward pass was
performed. No model, architecture, loss, augmentation, candidate count,
normalization setting, checkpoint, tolerance, or device assumption was changed.
All compute is inference-only, CPU, `torch.no_grad()`.

Label convention: every substantive conclusion carries exactly one label —
**VERIFIED**, **INFERRED**, or **UNKNOWN**. Measured, inferred, and unknown are
kept strictly separate. A correction made during this audit is preserved beside
the original mistake in §10, not deleted.

---

## 1 OBJECTIVE

**VERIFIED.** Determine, by measurement only, where the huge activation dynamic
range in the frozen ARM-P seed-1 model originates; whether it is inherited from
the P2A control or amplified in ARM-P (descriptive only); whether it is broad
or concentrated; whether `regression_normalize` plausibly leaves the
pre-normalization scale unconstrained; whether the scale is related to the C1
ONNX parity failure; whether a single identifiable intervention exists; and
whether any training experiment is justified. No intervention was tested.

---

## 2 FROZEN INPUTS

**VERIFIED.**

- ARM-P seed 1: `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth`,
  sha256 `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454`
  (re-verified at script start; mismatch would have stopped the run).
- P2A control: `phase2/runs/p2a_scale_coverage/p2a_best.pth`, sha256
  `0868ffd137a9985306bf5563685fd2362799bdd630d313e1181c980a7fbb6033`
  (re-verified at script start).
- ARM-P ONNX: `stage_c_deploy/armp_stereonet.onnx` (read-only), sha256
  `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989`.
- Reference: `reference/onnx/stereonet.onnx` (read-only, onnxruntime CPU),
  sha256 `b1a01d855bb22663f06dfd29eae11194e6fde952a319673e036f157c3e10095c`.
- Config: downsample_levels=3, num_disparities=24, cost_volume_shift=right,
  regression_normalize=True.
- Model size facts, unchanged: 397,954 params; 70 tensors; 24 candidates;
  3 downsample levels.
- Accuracy facts, unchanged: ARM-P frozen-KITTI EPE 1.1912168 px; Hailo
  reference 1.3134471 px; C1 parity FAIL.
- Frozen evaluation contract facts, unchanged: 40 scenes / 3,802,797 valid
  pixels / gt_scale 256.0 / source disp_occ_0.
- Target Hailo device UNKNOWN.
- C1 parity tolerance unchanged: max |delta| < 1e-3 px. C1 verdict stays FAIL.

---

## 3 DATA / FIVE-SCENE CONTRACT

**VERIFIED.** KITTI2015 split `hailo_val` indices 0–4, asserted exactly as the
existing script does (`000160_10.png 000161_10.png 000162_10.png 000163_10.png
000164_10.png`); the assertion passed, so the run proceeded. Preprocessing is
the same `normalize()` both models and all backends use. All per-scene numbers
below are on these five scenes; "pooled" means concatenated over all five.

---

## 4 MEASUREMENT METHOD

**VERIFIED.** New scripts only (no existing file touched):

- `stage_c_deploy/dr0_audit/dr0_activation_audit.py` → Tasks 1–6 and 8.
- `stage_c_deploy/dr0_audit/dr0_normalization_probe.py` → Task 7 empirical.
- Debug copies (never deployment artifacts, never modifications of the
  sources): `stage_c_deploy/dr0_audit/armp_instrumented_debug.onnx` (adds
  `/aggregation/Squeeze_output_0` as a graph output of a copy of
  `armp_stereonet.onnx`); the reference reuses the existing read-only debug
  copy `stage_c_deploy/diagnostics/reference_instrumented_debug.onnx`.
- PyTorch path: `model.eval()`, CPU, `torch.no_grad()`, 1 thread;
  `forward(..., return_stages=True)` for the seven named stages, plus two
  regression-internal stages recomputed exactly per
  `src/models/stereonet/regression.py:48,66-67`: `cost_upsampled_prenorm` =
  `F.interpolate(aggregated_cost, bilinear, align_corners=True)`,
  `cost_postnorm` = `(up − mean_d) / (std_d + 1e-6)`, i.e. the tensor actually
  fed to `softmax(−cost)`.
- Statistics per stage, per-scene and pooled: min, max, mean, abs-mean, std,
  median, p90, p95, p99, p99.9, p99.99 of |value|, fraction of |value| above
  each of 1e2 … 1e18, tensor name, shape, dtype. **VERIFIED.** These thresholds
  are descriptive; none of them is a deployment criterion.
- Per-layer growth: forward hooks only, no model modification.
- Reference path: onnxruntime CPU, 1 thread, same five scenes, same normalized
  inputs.
- All numbers: `stage_c_deploy/dr0_audit/dr0_activation_audit.json` and
  `stage_c_deploy/dr0_audit/dr0_normalization_probe.json`, with provenance
  (UTC, git HEAD, host, python/torch/onnx/onnxruntime/numpy/scipy versions,
  sha256 of both checkpoints and of every ONNX read, scene list, tensor names,
  shapes, dtypes, exact commands, deviations).

---

## 5 ARM-P ACTIVATION STATISTICS

**VERIFIED.** Pooled over the five scenes (per-scene tables in the JSON;
shapes: features `(1,32,46,154)`, cost_volume `(1,32,24,46,154)`,
aggregated_cost `(1,24,46,154)`, upsampled/postnorm `(1,24,368,1232)`,
disparities `(1,1,368,1232)`; all float32):

| stage | max | abs-mean | median | p99 | p99.9 | p99.99 |
|---|---|---|---|---|---|---|
| left_features | 4.12e13 | 3.50e12 | 2.46e12 | 1.60e13 | 2.37e13 | 3.03e13 |
| right_features | 4.35e13 | 3.50e12 | — | — | 2.36e13 | — |
| cost_volume | 7.08e13 | 4.15e12 | 2.87e12 | 1.94e13 | 3.00e13 | 4.09e13 |
| aggregated_cost | 2.53e18 | 4.20e16 | 2.14e15 | 8.15e17 | 1.38e18 | 1.87e18 |
| cost_upsampled_prenorm | 2.52e18 | 4.26e16 | 2.07e15 | 8.13e17 | 1.37e18 | 1.84e18 |
| cost_postnorm | 4.62e00 | 5.95e-01 | 3.77e-01 | 3.69e00 | 4.43e00 | 4.52e00 |
| disparity_initial | 1.59e01 | 4.80e00 | 3.98e00 | 1.20e01 | 1.33e01 | 1.48e01 |
| refinement_residual | 1.15e02 | 2.53e01 | 1.66e01 | 9.53e01 | 1.05e02 | 1.08e02 |
| disparity_final | 1.26e02 | 3.01e01 | 2.06e01 | 1.07e02 | 1.18e02 | 1.20e02 |

Fractions of |value| above thresholds, pooled **(VERIFIED)**:

- left_features: above 1e10: 0.9976; above 1e12: 0.7701. The whole feature
  tensor lives above 1e8 (fraction 1.0000 at 1e2–1e6, 0.99998 at 1e8).
- cost_volume: above 1e10: 0.9936; above 1e12: 0.7849.
- aggregated_cost: above 1e12: 0.99985; above 1e14: 0.9855; above 1e16:
  0.2001; above 1e18: 0.0051. ARM-P aggregated cost 2.53e18.
- cost_postnorm: nothing above 1e2 (max 4.62) — the readout compresses ~18
  orders of magnitude to single digits.
- disparity_final: above 1e2: 0.0351; everything else 0.

---

## 6 P2A ACTIVATION STATISTICS

**VERIFIED.** Same method, same scenes:

| stage | max | abs-mean | median | p99 | p99.9 | p99.99 |
|---|---|---|---|---|---|---|
| left_features | 1.58e07 | 1.16e06 | 8.35e05 | 5.18e06 | 7.64e06 | 1.02e07 |
| right_features | 1.67e07 | 1.16e06 | — | — | 7.82e06 | — |
| cost_volume | 2.06e07 | 1.40e06 | 9.98e05 | 6.31e06 | 9.45e06 | 1.24e07 |
| aggregated_cost | 6.16e09 | 1.44e08 | 1.45e07 | 2.56e09 | 4.18e09 | 5.11e09 |
| cost_upsampled_prenorm | 6.15e09 | 1.45e08 | 1.42e07 | 2.57e09 | 4.15e09 | 5.05e09 |
| cost_postnorm | 4.69e00 | 6.00e-01 | 3.67e-01 | 3.96e00 | 4.59e00 | 4.66e00 |
| disparity_initial | 1.64e01 | 5.01e00 | 4.11e00 | 1.21e01 | 1.33e01 | 1.40e01 |
| refinement_residual | 1.10e02 | 2.52e01 | 1.64e01 | 9.37e01 | 1.04e02 | 1.06e02 |
| disparity_final | 1.21e02 | 3.02e01 | 2.05e01 | 1.06e02 | 1.17e02 | 1.19e02 |

P2A 6.16e9 at aggregated_cost. Downstream of the readout both models live at
the same scale (postnorm max 4.6 vs 4.7; final max 126 vs 121).

Descriptive side-by-side, as required: ARM-P exhibits feature maxima ≈4e13
while P2A exhibits ≈1.6e7; ARM-P exhibits aggregated-cost maxima ≈2.5e18 while
P2A exhibits ≈6.2e9; ARM-P exhibits post-readout maxima ≈126 px while P2A
exhibits ≈121 px. **VERIFIED.** No causal claim is made about what produced
either scale.

---

## 7 HAILO REFERENCE STATISTICS

**VERIFIED** where anchors exist; otherwise UNKNOWN (no guessing). Anchors
reused from the existing measurement: `/Transpose_output_0`,
`/Squeeze_output_0`, `/ReduceSum_output_0`,
`/refine/refine.7/Conv_output_0`, output `524`, and the two `res.6` branch
convs (left traced to `input.1`, right to `input.83`). Reference 23.9 at
aggregated_cost. Pooled maxima: features 62.8/63.9, cost_volume 88.2,
aggregated_cost 23.9, disparity_initial 10.5, residual 135.7, final 135.7.
Full percentile/fraction tables are in the JSON; no reference fraction exceeds
1e2 at any pre-readout stage.

Caveats, carried verbatim: different downsample (x16 vs x8), 12 vs 24
candidates, the reference cost volume is degenerate (EXP-010), reference
measured through onnxruntime while ours is PyTorch. The reference is a
DEPLOYMENT-RANGE reference, not a layer-by-layer architectural equivalence.

---

## 8 STAGE-BY-STAGE SCALE GROWTH

**VERIFIED.** Growth factor = max|out| / max|in| on pooled maxima (p99.9
variant alongside):

ARM-P: features→features(sibling) ×1.06; features→cost_volume ×1.63;
cost_volume→aggregated_cost ×3.58e4 (p99.9 ×4.58e4);
aggregated→upsampled ×0.99; upsampled→postnorm ×1.8e-18 (the readout
compression); postnorm→disparity_initial ×3.45;
disparity_initial→residual ×7.22; residual→final ×1.10.

P2A: same shape — cost_volume→aggregated_cost ×2.98e2 (p99.9 ×4.42e2) is the
largest growth in both models; all other transitions are ≤ ×7.

**INFERRED.** The single operation with the largest growth factor is the 3D
aggregation submodule (cost_volume → aggregated_cost), in both models.

One level deeper, hooks only, inside aggregation **(VERIFIED maxima)**:

- The growth is progressive across all four aggregation Conv3d layers, not a
  single-layer event. ARM-P per-scene maxima: filter.0 out ≈1.4–1.8e15 (i.e.
  ×~21 over the 7e13 cost-volume input at the first conv); filter.2 ≈2.7–3.3e16
  (×~20); filter.4 ≈2.7–3.4e16 (≈flat); filter.6 ≈1.4–1.9e17 (×~6); to_cost
  (final Conv3d 32→1) ≈1.6–2.5e18 (×~12, the largest single-layer jump).
- P2A shows the same progression at its own scale: ≈1.8–2.0e8 → ≈5.2–6.0e8 →
  ≈5.5–6.8e8 → ≈2.4–2.8e9 → to_cost ≈5.4–6.2e9.
- The earlier suspicion is corrected: aggregated_cost is not the origin by
  assumption — it was identified as the growth winner by measured
  max|out|/max|in| over the full chain, and the per-layer hooks then named
  the to_cost conv as the largest single jump inside it.
- Note: the `left_features → right_features` chain entry is not a
  computational transition (sibling branches sharing weights); its ×1.06
  reflects branch asymmetry only. **VERIFIED** from the graph structure.

---

## 9 CHANNEL / DISPARITY CONCENTRATION

**VERIFIED.** Pooled over the five scenes; ratio = top-1 channel/slice max ÷
median channel/slice max; n90 = units carrying 90% of total |value| mass:

- ARM-P exhibits left_features ratio 1.51, n90 27/32 → GLOBAL, while P2A
  exhibits 1.75, n90 28/32 → GLOBAL.
- ARM-P exhibits right_features ratio 1.71, n90 27/32 → GLOBAL, while P2A
  exhibits 1.81, n90 28/32 → GLOBAL.
- ARM-P exhibits cost_volume ratio 1.12, n90 22/24 slices → GLOBAL, while P2A
  exhibits 1.19, n90 22/24 → GLOBAL.
- ARM-P exhibits aggregated_cost ratio 3.72, n90 8/24 slices, while P2A
  exhibits 2.97, n90 8/24 slices — both flagged CONCENTRATED by the predeclared
  rule (n90 ≤ n/3), at the exact boundary (8 of 24).

**INFERRED.** The huge range is broad (GLOBAL), not sparse: features and cost
volume are spread across essentially all channels/slices in both models. The
only concentration is moderate — 8 of 24 disparity slices carry 90% of
aggregated-cost mass — with an identical shape in both models, so it does not
distinguish them.

---

## 10 REPRODUCIBILITY CHECK

**VERIFIED.**

a) Determinism: ARM-P forward run twice — bit-identical stage maxima on all
   nine stages (max self-diff 0.0). P2A likewise. The extreme values are not
   run-to-run instrumentation noise.
b) Second path: `stage_c_deploy/armp_stereonet.onnx` (read-only) copied to
   `dr0_audit/armp_instrumented_debug.onnx` with `/aggregation/Squeeze_output_0`
   exposed, run under onnxruntime. Per-scene relative difference vs the PyTorch
   aggregated-cost maximum: 1.29e-07, 8.7e-08, 2.17e-07, 0.0, 2.04e-07. The
   2.5e18 scale reproduces through an independent backend to ~1e-7 relative —
   the fp32 accumulation-order signature, not an instrumentation artifact.
c) float32 throughout (every stage dtype float32, both models); non-finite
   count is 0 at every stage, per-scene and pooled — no overflow to inf/nan.
d) Hand check: max|.| of aggregated_cost recomputed with float64 accumulation
   matches the pipeline value exactly (ARM-P 2.5326005148901704e18 on
   000162_10.png; P2A 6155218944.0 on 000163_10.png).

Correction preserved: the first full-audit run crashed with `RuntimeError:
dictionary changed size during iteration` (deleting `conc_raw` entries while
iterating it); the fix (`for k in list(conc_raw)`) is in the shipped script,
and all numbers above come from the clean re-run. The earlier
`dr0_normalization_probe.py` draft also overstated the static analysis ("for
SMALL a, eps/a dominates, z→0" unconditionally); the corrected condition is
that collapse requires a·std_d ≪ 1e-6 (see §11), and the shipped script carries
the corrected text. Both mistakes are recorded here, not deleted.

---

## 11 NORMALIZATION ANALYSIS

Static **(VERIFIED against code)**. `soft_argmin` with `normalize=True`
(`src/models/stereonet/regression.py:48`) computes
`z = (c − mean_d(c)) / (std_d(c) + 1e-6)`, then `softmax(−z)`. For a > 0,
mean_d(a·c) = a·mean_d(c) and std_d(a·c) = a·std_d(c) (torch std, default
unbiased=True, is positively homogeneous), so
z(a·c) = (c − mean_d) / (std_d + 1e-6/a). Exact invariance holds only at eps=0.
The fixed epsilon breaks exact invariance only where it is not negligible
against a·std_d, i.e. when a·std_d ≪ 1e-6: then z→0, softmax→uniform over the
24 candidates, disparity→mean index 11.5 candidates. In the large-a direction
the formula approaches the scale-free value, but fp32 overflow of a·c itself
(inf/nan) can occur before the normalization rescues it. In `forward()`
(`src/models/stereonet/stereonet.py:150-154`) normalization sits inside
regression, i.e. before refinement and before the loss. The loss,
`masked_smooth_l1` (`src/losses/disparity.py:21-51`), takes only the final
disparity versus GT (valid = target > 0 and target < max_disparity,
`disparity.py:43-45`; smooth-L1 call, `disparity.py:51`); the training recipe
calls it as `masked_smooth_l1(out, disparity, ...)` where `out = model(left,
right)` is disparity_final
(`stage_b_armp/20260919T012646Z_tier2_seed1/scripts/finetune_pilot.py:298-299`).
**VERIFIED.** No term in the training pipeline sees the pre-normalization
magnitude.

Empirical, inference-only **(VERIFIED)**. Rescaling the captured
aggregated_cost by a ∈ {1e-6, 1e-3, 0.1, 1, 10, 1e3, 1e6} and re-running only
regression+refinement (replication self-check 0.0 on all scenes, both models):
pooled max|disparity(a) − disparity(1)| is 6.1e-05 px worst-case for ARM-P and
4.6e-05 px for P2A — fp32 noise at the frozen C1 scale — over the entire
12-order-of-magnitude range, for both models.

**INFERRED.** The graph supports the "scale is unconstrained" hypothesis: the
readout is invariant to the pre-normalization scale over at least a ∈
[1e-6, 1e6], and nothing in the loss penalizes its drift. Untested scales
(a·std_d ≪ 1e-6, i.e. a ≪ ~1e-22 at ARM-P's measured std, or overflow-scale
large a) remain UNKNOWN.

---

## 12 PARITY RELATIONSHIP

**VERIFIED.** The 20 top-disagreement pixels from
`stage_c_deploy/diagnostics/top_pixels_armp.csv` were mapped back to
low-res cost coordinates via the align_corners inverse
i=round(y·45/367), j=round(x·153/1231) (46×154 grid), magnitude = max over 24
candidates of |value| at (i,j), versus 2000 random pixels from the same scenes:

- Spearman(|Δ|, aggregated-cost magnitude): rho = −0.169, p = 0.476.
- Spearman(|Δ|, cost_postnorm magnitude): rho = +0.291, p = 0.213.
- Median aggregated-cost magnitude at discrepant pixels 2.7e15 vs 3.5e17 at
  random pixels; postnorm medians 3.51 vs 3.46 (indistinguishable).

**INFERRED.** No evidence of correlation was found at this sample size
(n = 20 discrepant pixels against 2000 random pixels: rho = −0.169,
p = 0.476 on aggregated-cost magnitude; rho = +0.291, p = 0.213 on
cost_postnorm magnitude); the test is underpowered and cannot exclude a
correlation — correlation only was tested. The two findings remain separate:
the scale does not coincide with the parity-failure locations, and the scale
did not cause the parity failure by any evidence in this audit. The C1
verdict stays FAIL regardless. **OBSERVED.** Discrepant pixels have a median
aggregated-cost magnitude of 2.7e15 against 3.5e17 at random pixels, i.e.
LOWER by roughly two orders of magnitude, which points away from, not
toward, large magnitude coinciding with parity disagreement; no causal
conclusion is drawn from it.

---

## 13 WHAT IS VERIFIED

- The nine-stage pooled maxima/means/percentiles/fractions for ARM-P, P2A, and
  the reference (where located), all float32, zero non-finite values (§§5–7).
- Bit-identical two-run determinism; independent ONNX second-path reproduction
  to ~1e-7 relative; exact float64 hand recomputation (§10).
- The largest measured growth is cost_volume→aggregated_cost in both models
  (ARM-P ×3.58e4, P2A ×2.98e2), progressive across aggregation layers with the
  largest single jump at the to_cost conv (§8).
- Broad (GLOBAL) concentration at features/volume; moderate 8-of-24 readout
  concentration identical in both models (§9).
- Disparity invariance to pre-normalization rescaling over a ∈ [1e-6, 1e6] at
  ≤6.1e-05 px, both models; loss sees only disparity_final (§11).
- No significant parity–magnitude correlation (rho −0.17/+0.29, p 0.48/0.21);
  C1 stays FAIL (§12).

---

## 14 WHAT IS INFERRED

- The scale explosion originates inside the 3D aggregation submodule
  (progressive conv amplification, largest jump at to_cost), not at the
  features or the cost volume — but the features already enter aggregation at
  ~4e13 (ARM-P), so aggregation amplifies an already-huge input.
- The range is broad, not sparse.
- `regression_normalize` plausibly leaves the pre-normalization scale as an
  unconstrained degree of freedom (readout invariance + loss blindness).
- The scale and the parity failure are separate findings.

---

## 15 WHAT REMAINS UNKNOWN

- Why either model's weights produce feature scales of 4e13 (ARM-P) or 1.6e7
  (P2A) — training dynamics; no training analysis is authorized here.
- Whether the 2.53e18 tensor survives per-tensor int8 quantisation before
  `regression_normalize` rescues it — UNKNOWN until quantisation is attempted
  with the toolchain (no outcome predicted).
- Untested rescale regimes (a·std_d ≪ 1e-6; overflow-scale a).
- Whether a different torch/onnxruntime version changes the numbers (recorded:
  torch 2.7.0+cu128 CPU path, onnx 1.22.0, onnxruntime 1.27.0, numpy 2.5.1,
  scipy 1.18.0).
- Reference `cost_upsampled_prenorm`/`cost_postnorm` equivalents: UNKNOWN (no
  anchors; not guessed).
- Whether a parity-magnitude relationship exists at a sample size larger than
  the 20 pixels available in top_pixels_armp.csv: UNKNOWN.

---

## 16 TESTABLE HYPOTHESES

H1 — "The large pre-normalization cost magnitude is an unconstrained degree of
freedom and can be reduced without materially affecting the learned stereo
solution." **Verdict: testable (INFERRED).** The audit supports testability:
(a) the readout is empirically invariant to 12 orders of rescaling at
≤6.1e-05 px; (b) the loss provably never observes the pre-normalization
magnitude; (c) the growth site is localized to the aggregation submodule with
a named largest jump (to_cost). H1 was NOT tested here. No second mechanism is
invented: a negative result on H1 is a valid outcome.

---

## 17 PROPOSED NEXT GATE

Exactly one candidate intervention is defined (H1 is supported; nothing else
is proposed):

- **Intervention:** rescale the frozen ARM-P aggregation outputs toward the
  reference deployment range (e.g. a fixed scalar s ≈ 1e17) via exactly one
  of two distinct variants — no training in either case, and both remain
  unauthorized until the next stage is separately authorized:
  - Variant A (activation rescale): divide the captured aggregated_cost by s
    before regression. No weight change, no new checkpoint.
  - Variant B (weight rescale): divide the `to_cost` conv weight and bias by
    s. This IS a weight modification producing a NEW checkpoint artifact; the
    frozen ARM-P seed-1 checkpoint `b2f6f5d5…fffeb7454` must not be
    overwritten.
  **VERIFIED** against `src/models/stereonet/aggregation.py:35,39-40`: A and
  B are mathematically equivalent at the aggregated_cost output because
  `to_cost` is a linear Conv3d with nothing between it and that output except
  the channel squeeze.
- **Frozen components:** all weights except the single rescaled tensor; all
  architecture, loss, augmentation, candidate count (24), normalization
  setting (`regression_normalize=True`), checkpoints, five-scene contract,
  C1 tolerance (< 1e-3 px, verdict FAIL unchanged).
- **Changed component:** the single scalar gain s applied at exactly one
  point (the aggregation output).
- **Hypothesis:** H1 — disparity output is unchanged to within fp32 noise
  because the readout is scale-invariant and the loss never saw the scale.
- **Expected mechanism:** `z(a·c) = (c−mean)/(std+eps/a)` (§11); with
  a = 1/s, eps/a stays negligible against measured std_d, so postnorm,
  softmax weights, disparity_initial, residual and final are preserved.
- **Cheapest valid gate:** inference-only rescale-and-compare on the same five
  scenes (max|Δ disparity| vs unscaled, plus the §11 probe extended to the
  chosen s) — no training gate is needed to test H1's invariance claim.
- **Accuracy metric:** frozen-contract EPE on the 40 hailo_val scenes
  (disp_occ_0, gt_scale 256.0) vs ARM-P frozen-KITTI EPE 1.1912168 px.
- **Deployment-range metric:** pooled max|aggregated_cost| and the §5
  fraction table recomputed after rescaling, against reference 23.9.
- **Rejection criteria:** the DR-1 rescale equivalence threshold —
  max|Δ disparity| > 1e-3 px (rescaled-vs-unscaled PyTorch) on any of the
  five scenes — or frozen-contract EPE regression ≥ 0.05 px, or
  postnorm/final max leaving the reference order of magnitude — any one
  rejects H1. The DR-1 rescale equivalence threshold is a SEPARATE criterion
  that happens to sit at the same numeric value as the frozen C1 parity
  tolerance; it is NOT the C1 gate, it does not modify, reinterpret or
  re-run C1, and the C1 verdict remains FAIL irrespective of any DR-1
  outcome. The numeric value is unchanged.
- **Provenance requirements:** new files only; record s, the exact rescale
  point (tensor name), UTC, git HEAD, checkpoint/ONNX sha256, per-scene
  max|Δ|, and the unchanged C1 FAIL.

---

## 18 STOP CONDITIONS

- If the rescale-and-compare gate rejects H1, stop: the scale is load-bearing
  and no training experiment is justified by this audit.
- If H1 passes the inference gate, that still justifies no training: the
  finding would be that the scale is removable without retraining.
- No training experiment is justified by this audit under any outcome —
  **INFERRED**, because every question posed here was answerable (and
  answered) by inference-only measurement.
- The C1 gate is untouched: tolerance max |delta| < 1e-3 px, verdict FAIL,
  regardless of any DR-0 outcome.

---

## PROVENANCE

**VERIFIED.** Scripts run (exact commands):
`python stage_c_deploy/dr0_audit/dr0_normalization_probe.py` and
`python stage_c_deploy/dr0_audit/dr0_activation_audit.py`, CPU,
`torch.no_grad()`, torch 1 thread / ORT 1 thread. Outputs (new files only):
`stage_c_deploy/dr0_audit/dr0_activation_audit.py`,
`stage_c_deploy/dr0_audit/dr0_activation_audit.json`,
`stage_c_deploy/dr0_audit/dr0_normalization_probe.py`,
`stage_c_deploy/dr0_audit/dr0_normalization_probe.json`,
`stage_c_deploy/dr0_audit/armp_instrumented_debug.onnx` (debug copy),
`stage_c_deploy/DYNAMIC_RANGE_AUDIT.md` (this file). Deviations from the
brief: none in method; two mistakes made and corrected during execution are
recorded in §10 beside their corrections. UTC, git HEAD
(`58e8a19908ddbd35451652c61aef478f56b51ebd`), host, all package versions, all
 sha256, scene list, tensor names/shapes/dtypes are in the two JSONs.
 Stage C DR-0 correction pass: text-only edits to sections 12, 15 and 17
 (fault 1: split intervention into activation-rescale vs weight-rescale
 variants; fault 2: named the DR-1 rescale equivalence threshold as separate
 from the C1 gate; fault 3: restated the n=20 finding as no evidence with the
 observed direction); no measurement was re-run and no number was changed.
