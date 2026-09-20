# FINAL PHASE-1 BOTTLENECK AUDIT — ARM-V

Read-only audit. No training was run. No existing arm's artifacts were
modified. New files created by this audit, and nothing else:

- `phase1/harness/bottleneck_diag.py` — per-stratum error decomposition
- `phase1/runs/arm_v_diag/` — diagnostic scripts + machine-generated JSON
- this document

Evidence grades used throughout, as briefed:
**A** = directly measured in ARM-V by this audit or an ARM-V artifact ·
**B** = measured in a previous controlled experiment in this project ·
**C** = project audit / public implementation evidence ·
**D** = inference only · **E** = speculation.

---

## 1. Executive Status

The remaining ARM-V error was located. It is not distributed, and it is not
where the last three arms looked.

**85.5 % of the 0.4593 px gap between the ARM-V 3-seed mean and the frozen
reference lives in the 4.61 % of evaluation pixels whose ground-truth
disparity is ≥ 64 px.** Reproduced on all three seeds (83.9 % / 94.6 % /
83.2 %). On the other 95.4 % of pixels ARM-V is at parity with the frozen
reference — the entire sub-64 px range contributes +0.093 / +0.013 / +0.095 px
of gap across seeds. [Grade A]

The mechanism of that failure is also measured: ARM-V's predictions
**saturate**. Maximum predicted disparity over the whole evaluation set is
91.5 / 113.9 / 101.7 px across seeds, against a ground-truth maximum of
153.0 px; the reference ONNX reaches 135.7 px. Signed error by GT bin is
within ±0.2 px up to 64 px and then collapses to −6.6 / −38.8 / −53.4 px
(seed 0) as GT rises. The model does not mis-estimate large disparities; it
cannot emit them. [Grade A]

Three mechanisms that were plausible a priori are **refuted by measurement**,
not by argument: disparity quantisation, disparity range, and model capacity.

Despite the bottleneck being identified unusually cleanly, **no candidate
intervention reaches the evidence bar this brief sets, and — independently —
the project's own measurement apparatus provably cannot confirm any effect
smaller than 74 % of the entire remaining gap.** ARM-X's real 0.1493 px
improvement already failed confirmation under exactly that rule [Grade B].

**Decision: NO FINAL EXPERIMENT JUSTIFIED. Freeze ARM-V, close Phase 1,
carry the measured bottleneck into Phase 2 as its starting brief.**

---

## 2. ARM-V Current State

Unchanged from the record; restated for self-containment.

| Property | Value |
|---|---|
| 3-seed frozen EPE | 1.8903392 / 1.5493088 / 1.8785827 |
| Mean | 1.7727436 px |
| Spread (max − min) | 0.3410304 px |
| Frozen reference | 1.3134471 px (re-verified live by this audit: 1.3134470770188373) |
| Gap to reference | 0.4592965 px |
| Original PyTorch baseline | 15.3958267 px |
| Parameters | 397,954 |
| 3-seed mean D1 | 10.269 % (reference 8.154 %) |

Architecture as recorded: 3 × 5×5 stride-2 convs (no activations between
them, so the downsample stack is linear), 6 residual blocks, 3×3 output conv,
32 channels, stride 8; signed-subtraction cost volume, 24 candidates, 8 px
full-resolution spacing, 184 px represented range, `shift=right`;
4 × Conv3d(32→32) + Conv3d(32→1); soft-argmin with
`regression_normalize=True`, emitting **candidate indices (0…23), not
pixels**; one full-resolution refinement stage guided by the left RGB image,
dilations (1,2,4,8,1,1); ReLU clamp. No BatchNorm anywhere.

Evaluation: the frozen contract only. Every number in this document that is
labelled an EPE over 3,802,797 pixels was produced through the unmodified
`phase1.harness.frozen_eval` dataset, GT scale, valid mask, pooled metric and
contract guard. Training-split numbers in §3.9 are explicitly **not**
frozen-contract scores and are never reported as such.

---

## 3. Bottleneck Evidence

Machine-generated backing: `phase1/runs/arm_v_diag/diag_seed{0,1,2}.json`,
`reference_strata.json`, `train_split_epe.json`, `ceiling_probe.json`,
`pre_vs_post_aggregation.json`. Reproduce the tables with
`python phase1/runs/arm_v_diag/summarize.py` and
`python phase1/runs/arm_v_diag/gap_decomposition.py`.

### 3.1 The error is a tail, not a precision deficit [Grade A]

| error bin (px) | frac of pixels (s0/s1/s2) | share of EPE mass (s0/s1/s2) |
|---|---|---|
| < 1 | 0.629 / 0.655 / 0.629 | 0.142 / 0.178 / 0.144 |
| 1 – 3 | 0.261 / 0.248 / 0.262 | 0.230 / 0.265 / 0.231 |
| 3 – 10 | 0.079 / 0.074 / 0.080 | 0.208 / 0.236 / 0.209 |
| ≥ 10 | 0.030 / 0.023 / 0.029 | 0.420 / 0.321 / 0.416 |

Median error 0.70 / 0.65 / 0.70 px. p99 error 29.8 / 18.7 / 30.9 px.
Roughly 3 % of pixels carry ~40 % of the EPE. Any mechanism that improves
typical-pixel precision by a few tenths of a pixel cannot move the frozen
score materially; the score is set by the tail.

### 3.2 The tail is exactly the large-disparity stratum [Grade A]

Gap decomposition against the frozen reference, per GT-disparity stratum.
`contribution = frac_px × (EPE_ARM-V − EPE_reference)`; contributions sum to
the total gap by construction.

| GT bin (px) | frac px | ref EPE | s0 EPE | s1 EPE | s2 EPE | s0 gap | s1 gap | s2 gap |
|---|---|---|---|---|---|---|---|---|
| [0,8) | 0.02975 | 1.301 | 1.311 | 1.185 | 1.332 | +0.0003 | −0.0035 | +0.0009 |
| [8,16) | 0.10906 | 1.296 | 1.242 | 1.224 | 1.289 | −0.0059 | −0.0078 | −0.0008 |
| [16,32) | 0.30667 | 1.172 | 1.185 | 1.168 | 1.214 | +0.0041 | −0.0011 | +0.0130 |
| [32,64) | 0.50837 | 1.228 | 1.413 | 1.277 | 1.388 | +0.0943 | +0.0251 | +0.0815 |
| **[64,96)** | 0.03716 | 2.642 | **7.567** | **4.790** | **7.283** | **+0.1830** | **+0.0798** | **+0.1725** |
| **[96,128)** | 0.00870 | 5.311 | **38.825** | **20.735** | **38.307** | **+0.2915** | **+0.1341** | **+0.2870** |
| **[128,160)** | 0.00028 | 19.416 | **53.436** | **51.495** | **58.185** | **+0.0097** | **+0.0091** | **+0.0110** |

- Total gap: +0.5769 / +0.2359 / +0.5651 px (3-seed mean +0.4593, matching
  the recorded 0.4592965).
- GT ≥ 64 px contributes **+0.4841 / +0.2231 / +0.4704 px = 83.9 % / 94.6 %
  / 83.2 %** of each seed's gap. 3-seed mean share **85.5 %**.
- Counterfactual: if ARM-V matched the reference's error on GT ≥ 64 px only,
  the 3-seed mean would go **1.7727 → 1.3802 px**, i.e. 0.0668 px from the
  reference. Everything else about the model would be left untouched.
- ARM-V's pooled EPE over GT ≥ 64 px is 13.74 / 8.08 / 13.44 px; the
  reference's is 3.25 px.

### 3.3 The mechanism: output saturation, not mis-estimation [Grade A]

`phase1/runs/arm_v_diag/ceiling_probe.json`.

| | s0 | s1 | s2 | reference | GT |
|---|---|---|---|---|---|
| predicted disparity p50 | 34.65 | 34.60 | 34.70 | 34.54 | 34.80 |
| p90 | 57.79 | 57.67 | 57.70 | 57.79 | 57.75 |
| p99 | 75.33 | 82.51 | 75.43 | 93.80 | 93.23 |
| p99.9 | 83.16 | 102.19 | 83.01 | 119.28 | 123.73 |
| **max** | **91.53** | **113.95** | **101.66** | **135.73** | **153.04** |

Mean signed error by GT bin (seed 0): +1.07, +0.35, +0.11, −0.02, **−6.57,
−38.82, −53.44**. The distribution is matched to the GT up to the p90 point
and then truncates. Seeds 1 and 2 show the same shape with different ceiling
heights. The reference under-predicts only in the last bin (−19.4 px).

This is a **learned** ceiling, not a structural one: the cost volume
represents 184 px, the soft-argmin index range is 0…23, and the refinement
residual is unbounded. Nothing in the graph forbids a 150 px output. The
trained composition simply never produces one.

Associated measurement (same probe): the effective readout scale, obtained by
the best least-squares affine fit of GT on the soft-argmin candidate index
over all valid pixels, is **4.99 / 9.26 / 4.91 px per candidate** against the
geometric 8.0 px per candidate. The soft-argmin index itself reaches
20.35 / 14.19 / 20.53 of a possible 23. Under the fitted scale the implied
maximum expressible disparity is ≈ 96 / 120 / 93 px — which is where the
observed ceilings sit. The one seed whose fitted scale is closest to
geometric (s1, 9.26) is also the seed with the highest ceiling, the lowest
GT ≥ 96 px error and the best overall EPE. **That last association is n = 3
and observational; it is recorded as Grade D and is not used as evidence for
anything below.**

### 3.4 Disparity quantisation is REFUTED as a bottleneck [Grade A]

| | s0 | s1 | s2 |
|---|---|---|---|
| EPE at distance-to-candidate 0.0–0.1 | 1.926 | 1.562 | 1.853 |
| EPE at distance-to-candidate 0.1–0.2 | 1.901 | 1.516 | 1.875 |
| EPE at distance-to-candidate 0.2–0.3 | 1.887 | 1.520 | 1.897 |
| EPE at distance-to-candidate 0.3–0.4 | 1.854 | 1.565 | 1.906 |
| EPE at distance-to-candidate 0.4–0.5 | 1.886 | 1.585 | 1.863 |
| frac-part histogram min / uniform | 0.968 | 0.949 | 0.971 |
| frac-part histogram max / uniform | 1.068 | 1.076 | 1.042 |

If the 8 px candidate spacing were binding, error would rise for predictions
that fall midway between candidates and the readout would show pinning at
integer indices. Neither happens: the error is **flat** in distance-to-candidate
(total variation < 4 %, and on seed 0 it is slightly *lower* midway), and the
fractional-part histogram is uniform to ±7 %. Only 9.6–9.9 % of pixels land
within 0.05 of an integer index, against 10 % expected by chance.

Additionally, **the represented range is not binding**: maximum GT in the
evaluation split is 153.04 px; the fraction of evaluation pixels with
GT > 184 px is exactly 0.000. Extending or refining the candidate grid
attacks nothing that is measured to be broken.

### 3.5 Cost-volume quality: weak before aggregation [Grade A for the proxy]

`pre_vs_post_aggregation.json`. RAW = ‖L(x) − R(x−k)‖₂ over the 32 channels of
the un-aggregated volume; AGG = the Conv3d output the readout consumes.

| | RAW s0/s1/s2 | AGG s0/s1/s2 |
|---|---|---|
| corr(hard-argmin, GT index) | 0.457 / 0.467 / 0.480 | 0.640 / 0.909 / 0.705 |
| corr(soft-argmin, GT index) | 0.367 / 0.348 / 0.380 | **0.938 / 0.962 / 0.937** |
| softmax entropy (max = 3.178) | 2.722 / 2.742 / 2.710 | 2.460 / 2.005 / 2.521 |
| mean local minima along D (of 24) | 7.13 / 7.06 / 7.24 | 6.79 / 6.23 / 6.92 |
| unimodal fraction | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |

Reading, stated carefully:

- The raw matching signal is **weak**: its soft readout correlates only
  0.35–0.38 with the true disparity index. The volume is live (the
  shift→none substitution moves EPE to 39.6 / 30.9 / 34.9 on the record) but
  it is not a discriminative matching cost.
- Aggregation **creates** most of the usable signal rather than destroying
  it: readout correlation rises from ~0.37 to 0.94–0.96.
- Neither stage produces a unimodal cost curve. Zero pixels, on any seed,
  before or after aggregation, have a single local minimum along the
  disparity axis; the mean is 6–7 minima out of 24. The final aggregated
  curve's own *minimum* is a poor disparity estimate (hard-argmin EPE
  25.2 / 13.7 / 31.4 px); the disparity information is carried by the mass
  of the curve, which is what the soft-argmin reads.
- Honesty caveat: ‖·‖₂ is **my** reduction of the signed 32-channel volume.
  The network's aggregation applies learned filters to the signed channels,
  so RAW is a proxy for what the network sees, not an identity. The
  measurement of the proxy is Grade A; the inference "the network's
  pre-aggregation signal is weak" is Grade C.

The matching-representation axis this points at is the axis **ARM-Z already
tested** (per-pixel L2 normalisation of L/R before subtraction): 3-seed mean
1.7435801, verdict INCONCLUSIVE, closed and not reopenable. The
`MATCHING_REPRESENTATION_AUDIT` selected ARM-Z as the single best-isolated
candidate on that axis out of nine surveyed mechanisms; the rest were
excluded on ARM-W overlap, ARM-Y overlap, or shape/sign non-isolation.

### 3.6 Aggregation is not identified as the origin of the remaining error [Grade A / D]

Measured: aggregation raises readout correlation from ~0.37 to 0.94–0.96
(§3.5), and its 3×3×3 kernels over a 24-deep volume at stride 8 give a
disparity-axis receptive field of 11 candidates (88 px) across the four
filter layers plus the output conv — adequate for the candidate grid in use. Its parameter share
is small; the refinement stage carried 90.6 % of MACs and 90.8 % of
activation traffic in the measured architecture [EXP-001, Grade B, measured
on the 1/16 configuration — ARM-V's 8× larger volume shifts that split and it
was **NOT re-measured** here].

What is **not** established: that a stronger aggregation would recover the
large-disparity stratum. Nothing in this audit varied aggregation. Claiming
aggregation as the bottleneck would be Grade D.

### 3.7 Refinement [Grade A]

| | s0 | s1 | s2 |
|---|---|---|---|
| corr(initial readout, GT) | 0.943 | 0.965 | 0.941 |
| corr(residual, final output) | 0.9991 | 0.9998 | 0.9990 |
| residual share of output magnitude | 0.764 | 0.858 | 0.746 |
| EPE of the best affine rescaling of the initial readout | 2.412 | 1.967 | 2.508 |
| EPE after refinement | 1.890 | 1.549 | 1.879 |
| ORACLE: refinement fed GT/8 as its initial input | 11.695 | 3.331 | 13.061 |

Two things changed versus the frozen baseline and should be recorded. In the
reference/baseline model the initial estimate was *anti*-correlated with GT
(r = −0.98) and the refinement inverted it [EXP-006/EXP-008, Grade B]. In
ARM-V the initial readout is **positively** correlated with GT at 0.94–0.97 —
the disparity search is now doing real work. The refinement still supplies
75–86 % of the output magnitude, but that is now mostly the candidate-index →
pixel rescaling, not the estimate itself.

Refinement's own contribution beyond an optimal linear rescale of the initial
readout is **0.52 / 0.42 / 0.63 px**. It is a real contribution but it is not
the dominant remaining limitation, and the model's headroom is not there.

The ORACLE probe is reported and then discounted. Feeding the refinement a
perfect GT-derived initial index makes the output **worse** (11.70 / 3.33 /
13.06 px). That proves the refinement is co-adapted to the specific,
miscalibrated statistics of the initial readout it was trained beside. It
does **not** bound what a jointly-trained better front end could achieve, and
it is not used as evidence for or against any candidate.

### 3.8 Occlusions, discontinuities, edges, texture [Grade A]

| stratum | frac px | ARM-V EPE (s0/s1/s2) | ref EPE | gap contribution (s0/s1/s2) |
|---|---|---|---|---|
| occluded (`disp_occ_0` valid ∧ `disp_noc_0` = 0) | 0.02145 | 9.01 / 8.14 / 8.59 | 3.72 | +0.113 / +0.095 / +0.105 |
| within 2 px of a > 3 px GT jump | 0.00333 | 4.60 / 4.47 / 4.23 | — | 0.8–1.0 % of EPE mass |
| lowest-texture quartile (image gradient) | 0.2518 | 2.28 / 1.78 / 2.28 | — | 29–31 % of EPE mass |
| highest-texture quartile | 0.2500 | 1.91 / 1.63 / 1.84 | — | 24–26 % of EPE mass |

- **Occlusion is a real but secondary contributor**: ~2.1 % of pixels,
  10–11 % of EPE mass, ≈ 0.10 px of the gap. It is also **not disjoint** from
  the large-disparity stratum (occlusions cluster on near objects), so the
  §3.2 and occlusion decompositions must not be added.
- **Discontinuities and edges are not a bottleneck**: 0.33 % of pixels, under
  1 % of EPE mass. Caveat: KITTI GT is ~21 % dense, so this 5×5-window
  discontinuity detector has low recall and the stratum is a weak proxy. It
  bounds the contribution from above only loosely; it does not prove edges
  are clean.
- **Texture matters mildly**: the low-texture quartile is 0.3–0.5 px worse
  than the high-texture quartile, and its EPE-mass share (29–31 %) barely
  exceeds its pixel share (25 %).
- Sky / road / reflective strata: **NOT MEASURED** — the evaluation contract
  carries no semantic labels and none were fabricated.

### 3.9 Capacity and data [Grade A]

Train-split measurement, run with the same forward graph, GT scale, crop and
pooled metric as the frozen contract but on the 160 **training** scenes.
Reported as a diagnostic, never as a contract score.

| | s0 | s1 | s2 |
|---|---|---|---|
| EPE on hailo_calib (train, 14,216,313 px) | **1.136** | **1.035** | **1.123** |
| EPE on hailo_val (frozen contract, 3,802,797 px) | 1.890 | 1.549 | 1.879 |
| generalisation gap | 0.754 | 0.514 | 0.755 |
| EPE on train pixels with GT > 96 px (32,414 px) | 25.31 | 7.92 | 24.62 |
| EPE on val pixels with GT > 96 px (34,150 px) | 39.29 | 21.71 | 38.94 |

Two conclusions, both load-bearing:

1. **Capacity is not the binding constraint at the scale of the remaining
   gap.** The 397,954-parameter model already reaches 1.03–1.14 px on data it
   has seen — *below the 1.3134 px reference score*. The remaining 0.4593 px
   is smaller than the 0.51–0.75 px generalisation gap. It sits inside
   generalisation, not inside capacity. Training loss falls monotonically to
   0.54–0.62 under cosine-to-zero with no divergence or plateau pathology.
2. **The large-disparity stratum is under-fit even on the training data.**
   GT > 96 px is 0.228 % of training pixels (32,414 of 14.2 M) versus
   0.898 % of evaluation pixels — a 3.9× under-representation — and the model
   fails on those pixels *in training* (7.9–25.3 px EPE). So the saturation
   of §3.3 is not pure overfitting.

Whether the ceiling is caused by (i) the readout/refinement composition's
learned scale, (ii) the scarcity of large-disparity training pixels, or (iii)
their interaction is **NOT SEPARABLE** by any measurement in this audit. That
attribution is Grade D.

Domain / data note: upstream StereoNet's published recipe is SceneFlow
pretraining (FlyingThings3D + Monkaa + Driving) followed by KITTI finetune
[Grade C, `PRETRAIN_RESOURCE_AUDIT`]. SceneFlow is **NOT on disk** (zero PFM
files, zero markers — verified in that audit). ARM-D tested the only locally
available proxy, Driving-only (4,400 pairs), and it was REFUTED (10.575 px).
This audit found no reason that differs materially from ARM-D, and there is
no data to run a different one on.

### 3.10 Loss / training target [Grade A / D]

Current loss: masked smooth L1, β = 1.0, valid = `gt > 0 ∧ gt < 184`.
Evaluation: pooled EPE (pixel-count-weighted mean absolute error) and D1.

Measured alignment facts:

- The training mask and the evaluation mask coincide in practice: zero
  evaluation pixels and < 5 × 10⁻⁷ of training pixels exceed 184 px, so the
  `gt < max_disparity` clause removes essentially nothing [Grade A].
- Smooth L1 with β = 1.0 is linear above 1 px, so on the tail that dominates
  the score it **is** the evaluation metric up to a constant; both weight
  every pixel equally. There is no large-error down-weighting to undo.
- The large-disparity stratum is not invisible to the loss. At 0.23 % of
  training pixels with ~25 px error, it contributes on the order of 9 % of
  the total training loss — the optimiser feels it and still does not fit it.

There is therefore **no measured misalignment between the loss and the metric**.
A loss reweighting is a hypothesis about optimisation dynamics, not a
correction of a measured mismatch, and it would be Grade D. ARM-X already
tested the one loss-shape change that was registered (deep supervision):
3-seed mean 1.6234090, a real 0.1493 px improvement over the ARM-V mean,
which nonetheless **failed its preregistered confirmation** and was closed
[Grade B].

### 3.11 Deployment constraints [Grade B / C]

Recorded, unchanged: 397,954 parameters; no BatchNorm; refinement carried
90.6 % of MACs and 90.8 % of activation traffic in the measured
configuration, with every activation inside it at 55.34 MiB, and Hailo's
compiler spatially defusing it into up to 22 pieces [EXP-001 / SR-003,
Grade B/C]. ARM-V's 8× larger cost volume was **NOT re-profiled** here; the
recorded MAC split predates the stride change and should not be quoted for
ARM-V without re-measurement.

Nothing in this audit produced a deployment-driven accuracy hypothesis, and
deployment efficiency was not used to invent one.

---

## 4. What Is NOT Identifiable

Stated plainly, as briefed, rather than resolved by assertion.

1. **Capacity as a bottleneck — NOT IDENTIFIABLE, and positively
   counter-indicated.** Train-split EPE 1.03–1.14 px is below the reference's
   1.3134 px val score; the remaining gap is smaller than the generalisation
   gap. Parameter count tells us nothing here and was not used.
2. **The cause of the output ceiling — NOT SEPARABLE.** Readout scale
   miscalibration, training-pixel scarcity at GT ≥ 64 px, and their
   interaction all predict the same observation. No measurement in this audit
   distinguishes them; only an intervention could, and §7 explains why an
   intervention cannot be resolved here.
3. **Whether aggregation capacity limits the large-disparity stratum — NOT
   MEASURED.** Aggregation was never varied.
4. **Whether a better-conditioned front end would help, given joint
   training — NOT BOUNDED.** The ORACLE probe (§3.7) measures co-adaptation
   of the existing refinement, not the ceiling of a retrained pair.
5. **Semantic strata (sky, road, reflective surfaces) — NOT MEASURED.** No
   labels exist under the frozen contract and none were invented.
6. **ARM-V MAC/activation profile — NOT RE-MEASURED.** The recorded split is
   from the 1/16 configuration.
7. **Edge/discontinuity contribution — measured but weakly bounded.** The
   GT-jump detector has low recall on 21 %-dense GT; 0.8–1.0 % of EPE mass is
   an estimate from a low-recall proxy, not a tight upper bound.
8. **The n = 3 association between fitted readout scale and seed quality
   (§3.3) — NOT EVIDENCE.** Three observational points, correct sign,
   explicitly not used.

---

## 5. Previously Tested Mechanisms

Closed. Not reopened by this audit; no result below was reinterpreted.

| Arm | Mechanism | Outcome |
|---|---|---|
| ARM-K | 200-epoch cosine training | incumbent predecessor, 5.527 |
| ARM-K-LR | step LR | refuted, 12.783 |
| ARM-K600 | 600 epochs | refuted, 8.558 |
| ARM-G | photometric augmentation | refuted, 10.475 |
| ARM-D | Driving-only synthetic pretraining | refuted, 10.575 |
| ARM-S | disparity search without normalised regression | refuted, 10.887 |
| ARM-T | cost normalisation before softmax, alone | refuted, 8.270 |
| ARM-U | search + regression normalisation | accepted, mean 2.3732; superseded |
| **ARM-V** | 24 candidates, 8 px spacing | **incumbent, mean 1.7727** |
| ARM-W | group-wise cost compression | refuted for accuracy, 1.7776 |
| ARM-X | deep supervision | mean 1.6234, preregistered confirmation FAILED, closed |
| ARM-Y | left-image-guided post-volume excitation | strongly refuted, mean 3.4075 |
| ARM-Z | per-pixel L2 feature normalisation | mean 1.7436, INCONCLUSIVE, closed; may not be retuned, rescored or stacked |

Coverage this audit's findings imply:

- The **matching-representation** axis (§3.5) is the axis ARM-Z closed, after
  `MATCHING_REPRESENTATION_AUDIT` surveyed nine public mechanisms and
  excluded the rest on ARM-W overlap, ARM-Y overlap, or shape/sign
  non-isolation.
- The **post-volume conditioning** axis is ARM-Y, strongly refuted.
- The **channel-layout** axis is ARM-W, refuted.
- The **supervision-shape** axis is ARM-X, closed with a real but
  unconfirmable improvement.
- The **candidate-density** axis is ARM-V itself, and §3.4 now measures that
  it has no remaining headroom.
- The **synthetic-pretraining** axis is ARM-D, refuted, with no new data
  available to differentiate a retry.

---

## 6. Remaining Candidate Mechanisms

Three candidates, stated in full and graded against the brief's rule that
only A/B evidence normally justifies the final experiment.

### Candidate 1 — Geometric readout scale

- **Mechanism.** Multiply the soft-argmin output by the feature stride (8.0)
  so `disparity_initial` leaves the readout in full-resolution pixels rather
  than candidate indices; the refinement then corrects a physically scaled
  estimate instead of learning the scale itself.
- **Bottleneck attacked.** The output ceiling of §3.3.
- **Supporting evidence.** Grade **A** for the bottleneck: the ceiling
  (91.5 / 113.9 / 101.7 px vs GT 153.0), the bin-wise bias collapse, and the
  fitted readout scale 4.99 / 9.26 / 4.91 px per candidate against the
  geometric 8.0. Grade **D** for the mechanism: no experiment in this project
  has ever varied the readout scale, and the refinement's first convolution
  can absorb any constant rescaling of its input exactly, so the function
  class is *unchanged* — only initialisation and optimisation conditioning
  change. There is no measured evidence that this conditioning difference
  produces an accuracy difference.
- **Exact intervention.** One line in `DisparityRegression.forward`, gated by
  a config flag defaulting off. Zero parameters.
- **Risk.** Plausibly a near no-op, because the composition is already able
  to express any scale and demonstrably learns one (correlation 0.94–0.97).
- **Parameter impact.** 0 (397,954 unchanged). **Deployment impact.** One
  scalar multiply; ideal.
- **Why prior arms do not test it.** ARM-W changed channel layout; ARM-X
  changed supervision; ARM-Y added a post-volume gate; ARM-Z changed feature
  normalisation; ARM-T standardised cost across the disparity axis before the
  softmax, which is a different operation from scaling the index after it.
  Disjoint.
- **Verdict.** Fails the evidence rule: mechanism evidence is D.

### Candidate 2 — Disparity-dependent loss weighting

- **Mechanism.** Reweight the masked smooth-L1 loss to increase the gradient
  share of pixels with large GT disparity.
- **Bottleneck attacked.** The under-fitting of GT ≥ 64 px (§3.2, §3.9).
- **Supporting evidence.** Grade **A** for the bottleneck and for the
  train-split under-fit. Grade **D** for the mechanism: §3.10 measures that
  the loss is *not* misaligned with the metric, that the stratum already
  contributes ~9 % of the training loss, and that the optimiser therefore
  already feels it. Reweighting is a hypothesis about optimisation dynamics
  with no measured support.
- **Exact intervention.** A weight term in `masked_smooth_l1`. Zero
  parameters.
- **Risk.** High and asymmetric. 95.4 % of pixels are currently **at parity
  with the reference** and contribute only +0.093 / +0.013 / +0.095 px of
  gap. There is almost nothing to gain there and a great deal to lose, and
  any weighting scheme has a free parameter, which the "no tuning, no sweep"
  budget forbids choosing empirically.
- **Parameter impact.** 0. **Deployment impact.** None (training-only).
- **Why prior arms do not test it.** ARM-X changed *where* supervision is
  applied, not *how pixels are weighted*. Disjoint but weakly.
- **Verdict.** Fails the evidence rule: mechanism evidence is D, and the
  intervention is not free of an unconstrained hyperparameter.

### Candidate 3 — Full SceneFlow pretraining

- **Mechanism.** Reproduce upstream's two-stage recipe: pretrain on the full
  SceneFlow corpus, finetune on KITTI.
- **Bottleneck attacked.** The training-pixel scarcity at GT ≥ 64 px (§3.9)
  and, plausibly, the reference's advantage there.
- **Supporting evidence.** Grade **C** that upstream used this recipe. Grade
  **A** that the large-disparity stratum is under-represented in our training
  split by 3.9×. Grade **B**, against, that the only locally available
  synthetic proxy (Driving-only, 4,400 pairs) was tested as ARM-D and
  refuted at 10.575 px.
- **Exact intervention.** Not specifiable: **SceneFlow is not on disk**
  (`PRETRAIN_RESOURCE_AUDIT`: zero PFM files, zero markers), the repository's
  training script has no SceneFlow path, and the acquisition cost was never
  verified.
- **Risk.** Unbounded; a multi-day data acquisition plus a new two-stage
  training path is not one variable and not one experiment.
- **Parameter impact.** 0. **Deployment impact.** None.
- **Why prior arms do not test it.** ARM-D used 4,400 Driving pairs only.
- **Verdict.** Fails on executability, not only on evidence. It is a Phase-2
  scope item, if it is anything.

---

## 7. Decision

### NO FINAL EXPERIMENT JUSTIFIED

Two independent reasons. Either alone is sufficient.

**Reason 1 — no candidate mechanism clears the evidence bar.**

The *bottleneck* is Grade A and reproduced on three seeds: 85.5 % of the
remaining gap is the GT ≥ 64 px stratum, and the proximate mechanism is an
output ceiling at 91–114 px against a 153 px GT range. But every candidate
*mechanism* that attacks it rests on Grade D inference. The brief's rule is
explicit — "D/E alone are insufficient", and "C may support a mechanism but
must not substitute for a measured bottleneck". Candidate 1's function class
is provably unchanged by the intervention; Candidate 2 attacks a
misalignment that §3.10 measures does not exist and carries a free
hyperparameter the budget forbids fitting; Candidate 3 cannot be executed
with the data on disk and was already refuted in its available form.

**Reason 2 — the measurement apparatus cannot resolve any plausible effect.**

This is the harder constraint, and it is Grade B, measured in this project's
own controlled experiments.

- ARM-V's 3-seed spread is **0.3410304 px**. The whole remaining gap to the
  reference is **0.4592965 px**. The spread is 74 % of the gap.
- The inherited ARM-V-anchored confirmation gate is mean < 1.4317132 px. To
  CONFIRM, a new arm must capture **0.3410 px = 74.2 % of the entire
  remaining gap in a single intervention**, with no tuning, no extra seeds
  and no rescue run.
- That bar has already been shown to reject real improvements. **ARM-X moved
  the mean by 0.1493 px — a genuine, reproduced improvement on all three
  seeds — and failed confirmation.** ARM-Z moved it by 0.0292 px and was
  INCONCLUSIVE.
- The leaderboard's own recorded caveat says the same thing arithmetically:
  a bar re-derived at 3× the current spread would be 1.02 px, which exceeds
  the 0.459 px that remains.

So a fourth single-variable arm, at three seeds, under the frozen rule, has
no path to a CONFIRMED verdict short of nearly eliminating the large-disparity
stratum outright — and no candidate has A/B evidence that it would.

Running it anyway would produce a fourth INCONCLUSIVE record. That is not a
result; it is the cost of the result we already have.

**No `phase1/docs/FINAL_PHASE1_HYPOTHESIS.md` is written, because no
experiment is justified. No training is permitted under this decision.**

---

## 8. Phase-1 Closure Plan

1. **Freeze ARM-V as the Phase-1 deliverable.** 3-seed frozen EPE
   1.8903392 / 1.5493088 / 1.8785827, mean 1.7727436 px, spread 0.3410304 px,
   397,954 parameters, gap to the frozen reference 0.4592965 px, versus the
   15.3958267 px original training regime. Seed 1's checkpoint
   (`phase1/runs/arm_v_s1/arm_v_best.pth`, 1.5493088) is the best single
   artifact; the mean is the honest headline. Which of the two is carried into
   Phase 2 is a project-owner decision, not an audit finding.
2. **Close Phase 1.** ARM-W, ARM-X, ARM-Y and ARM-Z stay closed on their
   recorded verdicts. Nothing in this audit reinterprets any of them.
3. **Record this audit as Phase 1's exit evidence.** The bottleneck is
   located and quantified, which is a stronger closing position than a fourth
   unconfirmable arm would have been.
4. **Hand Phase 2 the measured brief**, without pre-committing it to any
   mechanism:
   - The target is the GT ≥ 64 px stratum: 4.61 % of evaluation pixels,
     85.5 % of the remaining gap, ARM-V EPE 13.74 / 8.08 / 13.44 px against
     the reference's 3.25 px.
   - The symptom is an output ceiling at 91–114 px against a 153 px GT range,
     with a fitted readout scale of 4.99 / 9.26 / 4.91 px per candidate
     against a geometric 8.0.
   - Three mechanisms are measured **not** to be the problem: disparity
     quantisation, disparity range, and model capacity.
   - The first thing Phase 2 needs is **not** another arm. It is a
     measurement apparatus that can resolve a 0.1–0.3 px effect — more seeds,
     a variance-reduced protocol, or a paired comparison — because the
     current one demonstrably cannot, and that is what closed ARM-X.
5. **Do not create a Phase 3.** Exactly three phases exist.

---

Appendix — reproduction

```
python phase1/harness/bottleneck_diag.py phase1/runs/arm_v    arm_v_best.pth phase1/runs/arm_v_diag/diag_seed0.json
python phase1/harness/bottleneck_diag.py phase1/runs/arm_v_s1 arm_v_best.pth phase1/runs/arm_v_diag/diag_seed1.json
python phase1/harness/bottleneck_diag.py phase1/runs/arm_v_s2 arm_v_best.pth phase1/runs/arm_v_diag/diag_seed2.json
python phase1/runs/arm_v_diag/reference_strata.py
python phase1/runs/arm_v_diag/train_split_epe.py
python phase1/runs/arm_v_diag/ceiling_probe.py
python phase1/runs/arm_v_diag/prevs_post_aggregation.py
python phase1/runs/arm_v_diag/summarize.py
python phase1/runs/arm_v_diag/gap_decomposition.py
```

All frozen-contract evaluations in this audit report `contract_match true`,
40 scenes, 3,802,797 valid pixels, `disp_occ_0`, GT scale 256.0, split
`hailo_val`. The reference ONNX re-scored live at 1.3134470770188373 px,
matching the recorded frozen reference exactly.
