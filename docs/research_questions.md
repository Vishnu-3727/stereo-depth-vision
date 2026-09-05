# Research questions

Open questions carried out of Phase 1, and the questions Phase 1 answered.

---

## Part 1 — Answered

| # | Question | Answer | Evidence |
|---|---|---|---|
| Q1 | What does the published 8.223 figure mean? | The KITTI D1 outlier rate in percent, not end-point error. The evaluator's own variable is named `three_pixel_correct_rate`. The model's true EPE is 1.313 px. | EXP-005 |
| Q2 | Can the published figure be reproduced from public artifacts? | Yes, exactly: 8.2237 against 8.223, a gap of +0.0007 points. | EXP-005 |
| Q3 | What is the full evaluation protocol? | `disp_occ_0` on KITTI 2015 training scenes 160–199, frame `_10`; zero-pad then top-left 368×1232 crop, no resize; ground truth divided by 255 rather than 256; D1 averaged per image; reported as a percentage. | EXP-005 |
| Q4 | Where does the published 623.1K parameter count come from? | Counting the shared Siamese extractor once per graph occurrence. 423,586 unique + 199,552 = 623,138. | EXP-001 |
| Q5 | Where does the published 112.2G operation count come from? | 2 × MACs. Our 56.04 GMAC doubles to 112.08 G, a −0.11 % gap. | EXP-001 |
| Q6 | Does the model do subtraction or concatenation? | Subtraction, contradicting the upstream README. | EXP-001 |
| Q7 | Crop or resize to reach 368×1232? | Zero-pad then top-left crop. No resize, so no disparity rescaling. | SR-002 repository |
| Q8 | Does the shipped application produce metric depth? | No. Disparity only, cast to 8-bit. No calibration, no baseline, no `Z = fB/d` anywhere. | SR-007 |
| Q9 | Is the cost volume the bottleneck, as the name suggests? | No. Construction is 0 % of MACs and aggregation 4.2 %. Refinement is 90.6 %. | EXP-001 |
| Q10 | Is refinement the *runtime* bottleneck, or only the static-MAC one? | Both, but MAC share overstates it: 90.6 % of MACs, 73.3 % of GPU time, 75.8 % of CPU time. | EXP-013, EXP-014 |
| Q11 | Does fewer FLOPs mean faster? | No. Errors from 0.67× to 142×, and a zero-MAC stage takes 3.6 % of GPU time. The ranking also changes 6× between GPU and CPU. | EXP-013, EXP-014 |
| Q12 | Is the 12-candidate disparity range adequate for KITTI? | Yes. Ground truth reaches 9.56 of 11 units; 0.0 % of pixels exceed the range. An earlier expectation to the contrary was refuted. | EXP-008 |
| Q13 | Does the model use the right image at all? | Yes, decisively. Corrupting it costs up to 90 D1 points. | EXP-007 |
| Q14 | **Does the model perform a disparity search?** | **No.** All twelve slices are bit-identical; the shift is a no-op. It uses `left − right` at zero shift. | **EXP-010** |
| Q15 | Why is the intermediate disparity anti-correlated with truth? | Because of Q14: the cost tensor carries no disparity information, so the reduction has no consistent semantics. | EXP-006, EXP-008, EXP-010 |
| Q16 | Can we implement the architecture independently and match? | Yes, to a relative 1e-7 at every stage — fp32 accumulation noise. | EXP-011 |
| Q17 | How much does a disparity error cost in metres? | 0.065 m at 5 m, 16.65 m at 80 m, per pixel of error — a factor of 256 from geometry alone. | EXP-003 |
| Q18 | How does that show up in the real model? | Disparity error is flat with range (1.02–1.47 px); depth error grows 43×, 0.213 m to 9.158 m. | EXP-012 |
| Q19 | What does quantisation cost? | fp16 is free (−0.014 D1 points, 28 % faster). int8 on our stack costs 2.79 D1 points, with a 45 px maximum per-pixel change. | EXP-015 |
| Q20 | Does the training pipeline work? | Yes. Loss fell 11.49 → 7.61, gradients finite and bounded, checkpoint produced. Validation did **not** improve meaningfully — 160 scenes from scratch is too little. | EXP-016 |

---

## Part 2 — Open

Grouped by what it would take to answer them.

### Answerable with what is already on disk

| # | Question | Why it matters |
|---|---|---|
| O1 | What is in `stereonet_profiler_results_compiled_runtime_data.html` [SR-006]? | 40.9 MB of Hailo's own compiled-runtime profile, already downloaded. The single most likely public source of per-layer on-device behaviour, and it would test B2 and B4 against the vendor's own numbers. **Highest-value open item.** |
| O2 | What does the original StereoNet paper actually specify? | [SR-010] has been read only at abstract level. The paper-versus-implementation comparison in `stereonet_architecture.md` §9 stays incomplete until the full text is read — including whether the refinement is genuinely a multi-scale cascade and whether the downsampling stack should have activations. |
| O3 | How much does the bottom-7-row crop change the accuracy figure? | Those rows hold the nearest road surface and the largest disparities. Measurable by evaluating on the uncropped image. |

### Answerable with a modest download

| # | Question | Needs |
|---|---|---|
| O4 | How does the model fail on textureless, repetitive, reflective, thin and boundary cases? | Middlebury 2014 (~5 GB). The largest remaining gap in `failure_analysis.md`. |
| O5 | Does it generalise off KITTI at all? | Middlebury or ETH3D. Trained and evaluated only on KITTI 2015 [SOURCE: SR-002]. |
| O6 | What does full Scene Flow pre-training reach? | FlyingThings3D subset plus GPU days. The recipe is recorded in EXP-016's config. |

### Answerable only with hardware

| # | Question | Needs |
|---|---|---|
| O7 | What is the per-layer latency on Hailo silicon? | A Hailo-8 device |
| O8 | What does spatial defusion cost? | A Hailo-8 device |
| O9 | Do any operators fall back to host? | A Hailo-8 device, or the mapping report |
| O10 | Power and FPS/W on anything? | Instrumentation |
| O11 | Why does the config list hailo15h/hailo10h while the benchmark is Hailo-8? | Possibly documentation; possibly a device |

### Deliberately not answered in Phase 1

These require changing the baseline, which the phase boundary forbids. They are
Phase 2 experiments on Phase 2 models.

| # | Question |
|---|---|
| O12 | **What is a working cost volume worth?** Fix the shift, retrain, measure. The implementation already carries the corrected shift behind a flag Phase 1 never enables. This is the single most consequential open question in the project. |
| O13 | Would soft-argmin at 1/16 resolution followed by disparity upsampling change accuracy? It would touch 1/192 as much data (B8). |
| O14 | Would adding activations to the linear downsampling stack change anything? |
| O15 | How much of the refinement stage's capacity is spent repairing the degenerate cost volume rather than genuinely refining? Depends on O12. |
| O16 | Which stage loses the accuracy under int8? The 45 px maximum change suggests one tensor, not uniform degradation. |

---

## Part 3 — Research opportunities for Phase 2

Hypotheses and directions. **No model, no implementation, no design.** Each is
stated so it can be falsified.

### H1 — The headroom is unmeasured

**Observation.** The baseline reaches D1 8.154 % and EPE 1.313 px with a cost
volume that performs no search [EXP-005, EXP-010].

**Hypothesis.** A working disparity search, everything else held fixed, changes
accuracy by a measurable amount.

**Why it is not obvious.** The refinement network was trained to compensate. It
may have learned a monocular-plus-photometric-cue solution good enough that
restoring the search helps little — or the search may unlock a large gain. Both
outcomes are informative, and neither can be assumed.

**Smallest disproving experiment.** Train both variants under one identical
protocol and compare. Everything needed already exists.

**This is the first thing Phase 2 should do.** Until it is answered, it is not
known what this architecture is actually capable of, and every comparison against
it in the literature is suspect.

### H2 — The efficiency literature is aimed at the wrong stage

**Observation.** Published efficiency work overwhelmingly targets cost-volume
aggregation [SR-042, SR-043, SR-045, SR-046]. Here aggregation is 4.2 % of MACs
and 2.8 % of GPU time, while refinement is 90.6 % and 73.3 % [EXP-001, EXP-013].

**Hypothesis.** For this architecture class, the compute frontier is set by
full-resolution refinement, so aggregation-focused techniques transfer poorly.

**Smallest disproving experiment.** Apply one aggregation-efficiency technique
and measure end-to-end latency. H2 predicts a negligible change.

### H3 — Peak single activation, not FLOPs, is the edge constraint

**Observation.** Peak activation 55.34 MiB; the compiler tiles exactly those
layers [EXP-001, SR-003]. MAC share mispredicts time by up to 142×, and the stage
ranking changes 6× between two general-purpose devices [EXP-013, EXP-014].

**Hypothesis.** Peak single-tensor size predicts edge deployability better than
parameter count or MACs.

**Smallest disproving experiment.** Requires target silicon. Until then this
stays a hypothesis, and this project should not claim edge efficiency without it.

### H4 — Far-range depth needs its own objective

**Observation.** Disparity error is flat with range while depth error grows 43×
[EXP-012]. A prediction passing the 3 px D1 test at 80 m can be wrong about
distance by more than half [EXP-003].

**Hypothesis.** A loss or metric weighted by depth sensitivity, rather than
uniform disparity error, improves metric depth at range at little cost to the
aggregate disparity figure.

**Smallest disproving experiment.** Retrain with a depth-weighted loss under the
frozen protocol and compare range-binned depth error.

### H5 — Confidence may be recoverable, or may not exist to recover

**Observation.** No confidence output, and by B1 no meaningful cost curve to
derive one from [SR-001, EXP-010].

**Hypothesis.** With a working cost volume (H1), the cost curve's shape supports a
usable confidence estimate; without one, it does not.

**Smallest disproving experiment.** Depends on H1. Measure whether cost-curve
sharpness correlates with error, in both variants.

---

### What Phase 2 must not conclude from Phase 1

- That StereoNet's architecture is limited in the ways this deployment is. The
  deployment has a broken cost volume; the architecture may not.
- That our latency numbers say anything about Hailo silicon. They do not.
- That the competitor comparisons here are measurements. They are abstract-level
  author claims.
- That aggregate accuracy metrics describe far-range behaviour. They do not.

---

*Answers reference EXP-001 through EXP-016 and SR-001 through SR-050 as marked.*
