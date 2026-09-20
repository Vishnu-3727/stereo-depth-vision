# PHASE-2 STEP 1 — ARM-V HIGH-DISPARITY DIAGNOSTIC

Inference only. **No training was run. No ARM-V artifact was modified. No
closed arm's verdict was revisited, re-scored, retuned or reinterpreted.**

Diagnostic record: `phase2/diagnostics/arm_v_high_disparity/`
(`diag_high_disparity.py`, `conditional_probe.py`, `collapse_census.py`,
`apparatus_and_reference.py`, `scale_transfer_probe.py`, `tables.py`,
`make_figures.py`, plus `results.json`, `conditional.json`,
`collapse_census.json`, `apparatus_and_reference.json`, `scale_transfer.json`,
`high_disparity_diagnostic.png`). Reproducibility metadata (UTC, git head,
python/torch/numpy versions, device, checkpoint SHA-256, contract guard) is
embedded in `results.json`.

Note on the pre-existing `phase2/` tree: the H1/H2/E-series correspondence
campaign already in `phase2/docs/` and `phase2/experiments/` belongs to an
earlier, superseded Phase-2 line built on the pre-ARM-V baseline. Nothing in it
was edited. This document opens the ARM-V-based Phase 2 defined by the current
kickoff.

Evidence grades: **A** directly measured · **B** strongly supported by multiple
measurements · **C** plausible inference · **D** speculation.

---

## 1. Executive Summary

Five inference-only probes were run on the three frozen ARM-V checkpoints.

**The high-disparity collapse is real, universal, and is caused by neither the
refinement stage, the readout scale, the readout architecture, nor a failure of
stereo matching at those pixels.** Each of those four was eliminated by a
direct measurement, not by argument.

The decisive result is a test-time scale transfer. Presenting the *same*
evaluation pixels to the *same* frozen weights with the stereo pair downscaled
by 0.5 — which halves every true disparity, moving the high-disparity pixels
into the band the model handles well, at the cost of half the spatial
resolution — collapses the GT ≥ 96 px error on every seed:

| seed | EPE(GT≥96) at scale 1.0 | at scale 0.5 | signed bias 1.0 → 0.5 | slope(pred~GT, GT≥96) 1.0 → 0.5 | max prediction 1.0 → 0.5 |
|---|---|---|---|---|---|
| 0 | 39.287 | **10.879** | −39.29 → **−4.24** | +0.295 → **+1.010** | 91.5 → **152.7** |
| 1 | 21.708 | **8.475** | −21.70 → **−3.69** | +0.239 → **+1.253** | 113.9 → **144.3** |
| 2 | 38.935 | **10.851** | −38.94 → **−4.19** | +0.075 → **+0.775** | 101.7 → **149.0** |

Throwing away half the spatial resolution cannot improve stereo matching. It
can only change which disparity values are presented. Therefore the collapse is
a function of **the disparity value presented, not of those pixels' scene
content or matching difficulty** [Grade A, 3/3 seeds].

A census over every frozen Phase-1 checkpoint that exists — 16 runs spanning
ARM-U, ARM-V, ARM-W, ARM-X, ARM-Y, ARM-Z — finds the collapse in **16 of 16**.
Zero runs track high disparity. Maximum predicted disparity across all 16 lies
in 84.9–118.8 px against a 153.0 px GT range; the high-stratum slope has mean
+0.094 and s.d. 0.217. The frozen reference ONNX, which uses the *identical
graph with a degenerate no-op cost-volume shift and therefore performs no
disparity search at all*, has slope **+0.727** (2.9 s.d. above the entire
population of 16 trained runs) and reaches 135.7 px [Grade A].

One mechanism class survives every elimination: **the disparity band the model
was trained to cover**. Training pixels with GT ≥ 96 px are 0.194 % of the
training split.

**Decision: ONE SPECIFIC OPTIMIZATION JUSTIFIED** — in-domain scale
augmentation of the training pair, specified in
`phase2/docs/PHASE2_HYPOTHESIS_01.md`. **No training has been run. Approval is
required before executing it.**

---

## 2. Frozen ARM-V Baseline

| Property | Value |
|---|---|
| 3-seed frozen EPE | 1.8903392 / 1.5493088 / 1.8785827 |
| Mean / spread | 1.7727436 px / 0.3410304 px |
| Frozen reference | 1.3134471 px (re-verified live here: 1.3134470770188373, contract_match true) |
| Reference gap | 0.4592965 px |
| Parameters | 397,954 |
| Evaluation | frozen contract, 40 scenes, 3,802,797 valid px, `disp_occ_0`, GT/256, top-left 368×1232 |

ARM-V is not modified by anything in this document.

---

## 3. High-Disparity Error Concentration

Per-GT-bin statistics, frozen evaluation split, all three seeds. Bins below
1,000 px are reported as INSUFFICIENT and not summarised.
(`results.json` → `seeds.*.hailo_val.bins`.)

| GT bin | px | frac | EPE s0 / s1 / s2 | signed s0 / s1 / s2 | D1 % s0 / s1 / s2 |
|---|---|---|---|---|---|
| [0,16) | 527,871 | 0.13881 | 1.257 / 1.216 / 1.298 | +0.50 / +0.46 / +0.49 | 7.8 / 7.7 / 8.3 |
| [16,32) | 1,166,214 | 0.30667 | 1.185 / 1.168 / 1.214 | +0.11 / +0.05 / +0.13 | 7.1 / 7.4 / 7.5 |
| [32,48) | 1,118,114 | 0.29402 | 1.325 / 1.236 / 1.314 | −0.00 / −0.15 / +0.07 | 8.6 / 7.7 / 9.1 |
| [48,64) | 815,126 | 0.21435 | 1.535 / 1.334 / 1.490 | −0.05 / −0.18 / −0.10 | 10.6 / 8.0 / 9.8 |
| [64,80) | 112,726 | 0.02964 | 4.664 / 3.637 / 4.469 | −3.41 / −2.99 / −2.83 | 36.4 / 24.5 / 30.0 |
| [80,96) | 28,594 | 0.00752 | 19.009 / 9.338 / 18.377 | −19.00 / −9.17 / −18.38 | 91.6 / 61.2 / 92.0 |
| [96,112) | 21,249 | 0.00559 | 35.120 / 16.658 / 33.361 | −35.12 / −16.65 / −33.36 | 100.0 / 93.2 / 100.0 |
| [112,128) | 11,823 | 0.00311 | 45.483 / 28.064 / 47.196 | −45.48 / −28.06 / −47.20 | 100.0 / 99.1 / 100.0 |
| [128,144) | 1,071 | 0.00028 | 53.353 / 51.424 / 58.088 | −53.35 / −51.42 / −58.09 | 100.0 / 100.0 / 100.0 |
| [144,160) | 9 | 0.00000 | INSUFFICIENT | — | — |

Two facts define the problem [Grade A]:

1. Below 64 px — 95.386 % of pixels — the model is unbiased to within
   ±0.5 px and its D1 is 7–11 %.
2. Above 64 px the signed error is, from [80,96) upward, **equal in magnitude
   to the EPE**. That means essentially every pixel in those bins is
   under-predicted; the error is not scatter, it is a one-sided deficit.

Maximum predicted disparity (this diagnostic reports both populations; the
Phase-1 audit quoted the valid-pixel figure):

| | s0 | s1 | s2 | reference |
|---|---|---|---|---|
| over valid GT pixels | 91.53 | 113.95 | 101.66 | 135.73 |
| over all pixels incl. no-GT | 99.52 | 151.22 | 101.66 | 136.29 |
| GT maximum | \- | \- | \- | 153.04 |

---

## 4. Prediction-vs-GT Compression Analysis

Descriptive least-squares fits only. **No model was trained.** (`results.json`
→ `high_stratum`, `low_stratum_fits`.)

| stratum | fit | s0 | s1 | s2 |
|---|---|---|---|---|
| GT < 64 px | pred = a·GT + b, a | 0.9895 | 0.9865 | 0.9886 |
| | R² | 0.9629 | 0.9660 | 0.9628 |
| GT ≥ 64 px | pred = a·GT + b, a | **0.1375** | **0.5292** | **0.1195** |
| | b | +56.41 | +30.21 | +58.40 |
| | R² | **0.0461** | 0.4796 | **0.0329** |
| | residual RMSE (px) | 10.50 | 9.25 | 10.87 |

GT ≥ 64 px stratum (175,472 px, 4.614 % of the contract): mean GT 80.40,
mean prediction 67.46 / 72.76 / 68.01, mean signed error −12.93 / −7.64 /
−12.39, EPE 13.740 / 8.083 / 13.443, mean pred/GT 0.865 / 0.919 / 0.873.

Reading [Grade A]: below 64 px the model is very nearly an unbiased
disparity estimator (slope ≈ 0.99, R² ≈ 0.96). Above 64 px, on seeds 0 and 2,
**the output is close to a constant** — a slope of 0.12–0.14 with R² ≈ 0.04
means the prediction barely depends on the true disparity at all. This is not
a compressive-but-monotone mapping that a gain correction could undo; it is a
loss of dependence.

---

## 5. Initial-vs-Final Disparity Analysis

The soft-argmin emits a candidate index (0…23), not pixels; the refinement
supplies the scale. To ask whether the ceiling exists before refinement, the
index→pixel map was calibrated on GT < 64 px only (least squares) and then
extrapolated into the high stratum. (`results.json` →
`extrapolated_initial_readout`.)

| seed | calibration fitted on GT<64 | extrapolated EPE on GT≥64 | extrapolated max | **final model EPE** | **final model max** |
|---|---|---|---|---|---|
| 0 | GT = 4.6857·idx − 3.967 | 15.495 | 91.4 | 13.740 | 91.5 |
| 1 | GT = 8.7968·idx − 9.151 | 10.338 | 115.7 | 8.083 | 113.9 |
| 2 | GT = 4.6466·idx − 6.528 | 15.204 | 88.9 | 13.443 | 101.7 |

And the same collapse measured directly on the initial readout, GT ≥ 64:
slope of (8·index) on GT = **0.229 / 0.454 / 0.166**, R² = **0.039 / 0.406 /
0.017** — statistically the same collapse as the final output (§4).

Two conclusions, both [Grade A, 3/3 seeds]:

1. **Refinement does not cause the ceiling.** A purely affine rescaling of the
   initial readout, calibrated where the model works, already reproduces the
   ceiling almost exactly (91.4 vs 91.5; 115.7 vs 113.9). The ceiling is
   present at or before the soft-argmin.
2. **Refinement mildly helps.** Final EPE on the high stratum is *better* than
   the affine extrapolation on every seed (13.74 < 15.50, 8.08 < 10.34,
   13.44 < 15.20).

This **eliminates mechanism class 4 (refinement suppression)**, and it also
**eliminates class 1 (readout calibration / scale compression) as a fix**: with
R² of 0.02–0.41 inside the high stratum there is no surviving disparity signal
in the index for a recalibration to rescale. The learned scales (4.69 / 8.80 /
4.65 px per candidate against a geometric 8.0) are miscalibrated, but
correcting that miscalibration is measurably *worse* than what the model
already does.

---

## 6. Cost/Readout Analysis

Readout statistics computed exactly as the model computes them: aggregated cost
bilinearly upsampled to full resolution, standardised across the disparity axis
(`regression_normalize=True`), then `softmax(-cost)`. `z@true` is the
standardised cost at the true candidate — negative means a genuine
below-average (better-matching) response there. `raw` is the un-aggregated
volume reduced by ‖L(x) − R(x−k)‖₂.

| | s0 GT<64 | s0 GT≥64 | s1 GT<64 | s1 GT≥64 | s2 GT<64 | s2 GT≥64 |
|---|---|---|---|---|---|---|
| readout entropy (max 3.178) | 2.436 | 2.757 | 1.985 | 2.317 | 2.511 | 2.522 |
| p at true candidate ÷ uniform | 3.689 | 0.788 | 0.863 | 1.449 | 3.944 | 0.588 |
| median rank of true candidate (of 24) | 5 | 14 | 5 | 6 | 2 | 16 |
| **z@true, aggregated** | −1.117 | **−0.081** | −0.288 | −0.574 | −1.353 | **+0.088** |
| **z@true, raw volume** | **−2.029** | **−1.458** | −1.988 | −1.434 | −2.051 | −1.465 |
| frac z@true < 0, raw | 0.983 | **0.932** | 0.983 | 0.927 | 0.982 | 0.933 |
| readout distribution s.d. (candidates) | 6.51 | 6.59 | 5.44 | 4.70 | 6.41 | 6.82 |

Three findings [Grade A]:

1. **The raw cost volume still carries a genuine response at the true
   disparity in the high stratum.** z@true(raw) is −1.46 and negative for
   93 % of high-disparity pixels, against −2.03 and 98 % at low disparity. The
   matching evidence weakens but does not vanish.
2. **The aggregated cost loses that response** at high disparity on seeds 0 and
   2 (z −0.08, +0.09; the true candidate ranks 14th–16th of 24).
3. **But absence of a true-candidate response is demonstrably not sufficient to
   cause failure.** In the GT [32,48) and [48,64) bins z@true(aggregated) is
   already ≈ −0.06 to −0.14 and the true candidate ranks 13th–18th, yet EPE
   there is 1.24–1.54 px. In this architecture accuracy comes from the
   *centroid* of a broad distribution, not from a cost minimum at the true
   candidate. So "aggregation loses the peak" cannot be the explanation.

### Breaking the GT/evidence confound

Raw matching evidence and training density both degrade with disparity, so they
are confounded. The conditional probe stratifies **within** a GT bin by raw
evidence strength (`conditional.json`). EPE by z@true(raw) quartile:

| | quartile 1 (strongest) | 2 | 3 | 4 (weakest) |
|---|---|---|---|---|
| GT [64,80), s0 | 2.057 | 3.348 | 4.586 | 8.666 |
| GT [64,80), s1 | 1.931 | 2.494 | 3.294 | 6.828 |
| GT [64,80), s2 | 2.161 | 3.504 | 4.973 | 7.237 |
| GT [96,112), s0 | 34.000 | 35.989 | 35.139 | 35.352 |
| GT [96,112), s1 | 10.161 | 12.804 | 16.443 | 27.221 |
| GT [96,112), s2 | 33.415 | 34.110 | 33.629 | 32.289 |

At GT 64–80 px matching evidence matters on every seed (3–4× EPE spread across
quartiles). At GT 96–112 px on seeds 0 and 2 it does not matter at all — the
prediction is ≈ 68–70 px whatever the evidence says. So the deep collapse is
**decoupled from matching evidence** [Grade A].

### The decisive probe: test-time scale transfer

`scale_transfer.json`. Same weights, same pixels, same GT; the stereo pair is
downscaled by s before inference and the prediction divided by s afterwards.

| seed | scale | global EPE | EPE GT<64 | EPE GT 64–96 | EPE GT≥96 | signed GT≥96 | slope_hi | max pred |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.00 | 1.8903 | 1.317 | 7.567 | 39.287 | −39.29 | +0.295 | 91.5 |
| 0 | 0.75 | 2.2044 | 1.951 | 4.138 | 21.102 | −20.73 | +0.472 | 126.7 |
| 0 | 0.50 | 2.2717 | 2.127 | 3.898 | **10.879** | **−4.24** | **+1.010** | **152.7** |
| 1 | 1.00 | 1.5493 | 1.233 | 4.790 | 21.708 | −21.70 | +0.239 | 113.9 |
| 1 | 0.75 | 2.0340 | 1.885 | 3.304 | 12.605 | −12.14 | +0.537 | 128.0 |
| 1 | 0.50 | 2.1946 | 2.062 | 4.088 | **8.475** | **−3.69** | **+1.253** | **144.3** |
| 2 | 1.00 | 1.8786 | 1.319 | 7.283 | 38.935 | −38.94 | +0.075 | 101.7 |
| 2 | 0.75 | 2.1806 | 1.950 | 3.623 | 20.680 | −19.90 | +0.219 | 123.1 |
| 2 | 0.50 | 2.2267 | 2.092 | 3.611 | **10.851** | **−4.19** | **+0.775** | **149.0** |

Monotone in s on all three seeds, on every high-disparity metric. At s = 0.5
the systematic under-prediction is almost gone (−39.3 → −4.2 px), the model
tracks disparity again (slope 0.075–0.295 → 0.775–1.253) and its predictions
reach the GT range.

Downscaling by 2 destroys half the spatial resolution and coarsens the
effective candidate spacing from 8 px to 16 px of original image. It can only
*hurt* matching. It nevertheless fixes the high stratum. **Therefore the
collapse is not a matching failure at those pixels — it is a function of the
disparity value presented** [Grade A, 3/3 seeds]. This eliminates mechanism
class 3, and — since the readout, aggregation and refinement are bit-identical
across scales — it also eliminates classes 1, 2 and 4.

The low stratum degrades as expected (1.32 → 2.13 px), and global EPE degrades
with it, so scale 0.5 is **not** an improvement to the model; it is a probe.

---

## 7. Training-vs-Validation Evidence

The same per-bin diagnostic was run on the 160 **training** scenes with the
identical forward graph, GT scale, crop and pooled metric. This is a
diagnostic, **not** a frozen-contract score and is never reported as one.

| GT bin | train px frac | val px frac | train EPE s0/s1/s2 | val EPE s0/s1/s2 |
|---|---|---|---|---|
| [0,16) | 0.16819 | 0.13881 | 1.046 / 0.987 / 1.006 | 1.257 / 1.216 / 1.298 |
| [16,32) | 0.33772 | 0.30667 | 0.881 / 0.873 / 0.882 | 1.185 / 1.168 / 1.214 |
| [32,48) | 0.27276 | 0.29402 | 0.939 / 0.919 / 0.918 | 1.325 / 1.236 / 1.314 |
| [48,64) | 0.18197 | 0.21435 | 1.039 / 1.022 / 1.014 | 1.535 / 1.334 / 1.490 |
| [64,80) | 0.02956 | 0.02964 | 2.875 / 2.481 / 2.928 | 4.664 / 3.637 / 4.469 |
| [80,96) | 0.00751 | 0.00752 | 10.020 / 6.155 / 10.526 | 19.009 / 9.338 / 18.377 |
| [96,112) | **0.00194** | 0.00559 | 22.982 / 7.413 / 22.604 | 35.120 / 16.658 / 33.361 |
| [112,128) | **0.00032** | 0.00311 | 36.886 / 9.866 / 34.512 | 45.483 / 28.064 / 47.196 |
| [128,144) | 0.00002 | 0.00028 | INSUFFICIENT (293 px) | 53.353 / 51.424 / 58.088 |

**The ceiling is present on the training data**, on pixels the optimiser saw
200 times [Grade A]. Train GT [112,128) EPE is 34.5–36.9 px on seeds 0 and 2.
It is milder than on validation (roughly 35–45 % lower), so part of the
validation deficit is generalisation — but the collapse itself is not a
generalisation artifact.

**The evaluation split is materially richer in high disparity than the training
split**: GT ≥ 96 px is 0.194 % + 0.032 % + 0.002 % ≈ **0.228 %** of training
pixels against **0.898 %** of evaluation pixels — a 3.9× under-representation
[Grade A].

Maximum prediction on the training split: 108.15 / 161.93 / 112.26 px (all
pixels). The cap is not a fixed constant — it is higher where the model
trained, and it differs per seed.

---

## 8. Competing Mechanisms

The six mechanism classes named in the kickoff, each with its disposition and
the measurement that produced it.

| # | class | disposition | decisive measurement |
|---|---|---|---|
| 1 | readout calibration / scale compression | **ELIMINATED as a fix** | §5: affine recalibration of the initial readout, fitted where the model works, extrapolates to EPE 15.5 / 10.3 / 15.2 on GT≥64 — *worse* than the model's own 13.7 / 8.1 / 13.4. Inside the high stratum the index carries R² 0.02–0.41 against GT; there is no signal left to rescale. Also §6: the readout is bit-identical at scale 0.5, where the stratum is fixed. |
| 2 | cost-distribution saturation | **ELIMINATED as cause** | §6: identical readout and identical distribution machinery at scale 0.5 produce slope +0.78…+1.25 on the same pixels. And the frozen reference uses the same soft-argmin readout and reaches slope +0.727. |
| 3 | high-disparity matching failure | **ELIMINATED as the primary cause** | §6: halving the resolution — strictly less matching information — cuts GT≥96 EPE by 2.6–3.6× on 3/3 seeds. The raw volume also retains a true-candidate response (z −1.46, negative for 93 % of high pixels). Contributing at GT 64–80 (evidence quartiles span 3–4× EPE), but not at GT≥96 on seeds 0 and 2, where error is flat across evidence quartiles. |
| 4 | refinement suppression | **ELIMINATED** | §5: the ceiling is already reproduced by the pre-refinement readout; refinement *improves* the stratum on 3/3 seeds. |
| 5 | training-data coverage / high-disparity supervision | **SURVIVES** | §7: GT≥96 is 0.228 % of training pixels; the collapse is present on training data; §6: the failure is a function of the disparity value presented, not the pixels; §9 census: 16/16 runs collapse across 6 different architectural mechanisms, while the reference — same graph, degenerate cost volume, different training regime — does not. |
| 6 | another directly measured mechanism | **not required** | Class 5 accounts for every measurement without residue. |

### The census that made the elimination possible

`collapse_census.json`. Every frozen Phase-1 checkpoint, scored on GT ≥ 96 px.
Used **only** as samples of a Phase-2 training-dynamics question; no arm's
accuracy verdict is restated, re-ranked or revisited, and ARM-W / ARM-X /
ARM-Y / ARM-Z stay closed.

| arm | seeds | slope(pred~GT, GT≥96) | max predicted (valid px) |
|---|---|---|---|
| ARM-U | 3 | −0.122 / −0.163 / −0.425 | 84.9 / 88.1 / 93.4 |
| ARM-V | 3 | +0.295 / +0.239 / +0.075 | 91.5 / 114.0 / 101.7 |
| ARM-W | 1 | +0.136 | 89.1 |
| ARM-X | 3 | +0.293 / +0.325 / +0.410 | 117.9 / 112.1 / 118.8 |
| ARM-Y | 3 | −0.020 / −0.001 / +0.286 | 86.3 / 97.7 / 94.1 |
| ARM-Z | 3 | +0.090 / +0.089 / +0.003 | 92.2 / 92.1 / 87.1 |
| **all 16 trained runs** | | mean **+0.094**, s.d. 0.217, range −0.425…+0.410 | 84.9…118.8 |
| **frozen reference ONNX** | | **+0.727** (r = +0.597) | **135.7** |

16 of 16 collapse. Zero track. The reference is 2.9 s.d. above the entire
population — and the reference's cost-volume shift is a proven no-op
(EXP-010), i.e. it performs no disparity search whatsoever. **Whatever causes
the collapse, it is not the cost volume, not the aggregation, not the readout
and not the refinement, because the reference shares all four and a strictly
weaker version of the first** [Grade A/B].

What all 16 share and the reference does not: KITTI-only training on
160 scenes whose disparity distribution effectively ends at ~96 px.

---

## 9. Evidence Grades

| # | claim | grade |
|---|---|---|
| 1 | 95.4 % of pixels (GT<64) are predicted with slope 0.99, R² 0.96, bias < 0.5 px | **A** |
| 2 | GT≥64 (4.61 % of px) is under-predicted on every seed; from GT≥80 the signed error equals the EPE | **A** |
| 3 | Within GT≥64 the prediction is near-constant on seeds 0 and 2 (slope 0.12–0.14, R² 0.03–0.05) | **A** |
| 4 | The ceiling is present at/before the soft-argmin; refinement improves rather than causes it | **A** |
| 5 | No affine recalibration of the readout can fix it (extrapolation is worse than the model) | **A** |
| 6 | The raw cost volume retains a true-candidate response in the high stratum (z −1.46, 93 % negative) | **A** |
| 7 | Absence of an aggregated true-candidate response does not cause failure (GT 32–64 proves it) | **A** |
| 8 | Downscaling the input by 0.5 cuts GT≥96 EPE 2.6–3.6× and restores slope ≈ 1 on 3/3 seeds | **A** |
| 9 | Therefore the collapse is a function of the disparity value presented, not of the pixels | **A** |
| 10 | The collapse is present on the training split | **A** |
| 11 | GT≥96 is 0.228 % of training pixels vs 0.898 % of evaluation pixels | **A** |
| 12 | 16/16 frozen checkpoints collapse; the reference, same graph, does not | **A** |
| 13 | The surviving cause is the disparity band covered by training supervision | **B** |
| 14 | Widening that band by in-domain scale augmentation will reduce the collapse | **C** — this is the hypothesis, not a finding |
| 15 | The residual GT 64–80 px evidence-dependence (§6) is a second, smaller mechanism | **C** |

**NOT MEASURED / NOT IDENTIFIABLE**, stated rather than inferred:

- Whether scale augmentation preserves the GT<64 accuracy that currently sits
  at parity with the reference. The scale-0.5 probe shows the frozen model is
  scale-sensitive (low-stratum EPE 1.32 → 2.13), so this is the principal risk
  and is **NOT MEASURED**.
- The reference model's actual training recipe. Upstream StereoNet's published
  recipe is SceneFlow pretraining then KITTI finetune [Grade C,
  `PRETRAIN_RESOURCE_AUDIT`]; the weights in `reference/onnx/stereonet.onnx`
  were not produced here and their provenance is **NOT VERIFIED**.
- Whether the 0.228 % training share is "too small" in any absolute sense. Those
  pixels contribute roughly 9 % of the training loss magnitude, so they are not
  invisible to the optimiser; why they are nonetheless unfit is **NOT
  IDENTIFIABLE** from inference alone.
- ARM-V MAC/activation profile at stride 1/8 — **NOT RE-MEASURED** (the recorded
  90.6 % refinement share is from the 1/16 configuration).

### Measurement apparatus — an honest negative result

The kickoff asked for an instrument that can resolve 0.1–0.3 px in the high
stratum. Measured 3-seed relative spread (spread ÷ mean) of the two candidate
instruments, over every arm with three seeds:

| arm | global EPE | EPE(GT≥96) |
|---|---|---|
| ARM-U | 0.057 | 0.127 |
| ARM-V | 0.192 | **0.528** |
| ARM-X | 0.044 | 0.072 |
| ARM-Y | 0.633 | 0.204 |
| ARM-Z | 0.120 | 0.206 |

**EPE(GT≥96) is relatively noisier than global EPE in 4 of 5 arms.** Stratum
metrics are necessary because that is where the effect is, but they are *not* a
lower-noise instrument and must not be presented as one [Grade A].

What *is* resolvable is the **slope of prediction on GT in the high stratum**,
because the target effect is large compared with its noise: all 16 trained runs
lie in −0.425…+0.410 (s.d. 0.217) while the reference sits at +0.727. A
treatment that moves slope_hi to ≥ 0.60 would be outside the entire observed
population of this project's trained models. That is the instrument
`PHASE2_HYPOTHESIS_01` uses as its primary mechanism check.

---

## 10. Phase-2 Decision

### ONE SPECIFIC OPTIMIZATION JUSTIFIED

The mechanism is identified, not guessed. Four of the six candidate classes
were eliminated by direct measurement (§8), the fifth survives every probe, and
the scale-transfer result (§6) is a positive demonstration on frozen weights —
not an inference — that the same pixels are predicted well when their disparity
is presented inside the covered band.

The justified optimization is **in-domain scale augmentation of the training
pair**: resize the training crop and scale its disparity by the same factor, so
that the 160 available KITTI scenes supply supervision across the disparity
band the evaluation data actually contains. One variable, zero new parameters,
no change to the graph, no deployment impact, and disjoint from every closed
arm — ARM-G was *photometric* augmentation, ARM-D was *out-of-domain*
pretraining, and ARM-W / ARM-X / ARM-Y / ARM-Z are all graph or loss changes.

Specification: **`phase2/docs/PHASE2_HYPOTHESIS_01.md`**.

**Training has NOT been run and will not be run without explicit approval.**

### Diagnostic counterfactual — not an achieved result

If a model kept ARM-V's scale-1.0 accuracy below 64 px and achieved ARM-V's own
measured scale-0.5 accuracy above it, the frozen-contract EPE would be
1.4989 / 1.4044 / 1.4899, mean **1.4644 px** (from ARM-V's mean 1.7727436).
This is a **diagnostic counterfactual computed from measured strata, not an
achieved performance figure, and not a success threshold.** ARM-V's achieved
performance remains **1.7727436 px**. The frozen reference remains
1.3134471 px and is a reference, not a Phase-2 pass mark.

Phase 2 remains the final phase. No Phase 3 was created.
