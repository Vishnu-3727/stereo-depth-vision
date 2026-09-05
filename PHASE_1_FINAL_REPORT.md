# Phase 1 Final Report

**Microchip Stereo Depth Vision — forensics, reproduction and characterisation of
the Hailo StereoNet reference deployment.**

Seventeen experiments, twelve knowledge-base documents, one independent
implementation verified against the reference to a relative 1e-7, and the
published accuracy figure reproduced to four significant figures.

Everything below is traceable to an experiment ID (`EXP-xxx`) or a registered
source (`SR-xxx`). Where something could not be established it says **UNKNOWN**.

---

## The three findings that matter

**1. The deployed model performs no disparity search.** All twelve slices of its
cost volume are bit-identical — maximum absolute difference exactly 0.0
[EXP-010]. The zero-padding is appended on the right and then sliced away, so the
intended left-shift never happens. The published D1 of 8.223 % is achieved
without stereo correspondence search.

**2. The published "EPE 8.223" is not an EPE.** It is the KITTI D1 outlier rate
in percent. Hailo's own evaluator names the variable `three_pixel_correct_rate`.
The model's actual end-point error is 1.313 px [EXP-005]. The apparent sevenfold
disagreement with the StereoNet paper was a units problem, not an accuracy
problem.

**3. FLOPs do not predict latency, and the ranking is device-dependent.**
Per-stage error between MAC share and time share runs from 0.67× to 142×, a
zero-MAC stage consumes 3.6 % of GPU time, and 3D aggregation moves from 2.8 % of
time on GPU to 17.2 % on CPU [EXP-013, EXP-014].

---

## 1. What exactly is the reference stereo-depth system?

A **disparity** pipeline, not a depth pipeline [SR-007].

Two rectified 368×1232 RGB images in; one 368×1232 single-channel disparity map
out, in pixels. The shipped application's entire postprocess is
`cv::Mat(h, w, CV_8U, data).clone()` — the raw buffer wrapped as an 8-bit image.
There is **no calibration, no baseline, no focal length and no `Z = fB/d`**
anywhere in it. Synchronisation, calibration and rectification are all assumed
and never checked.

The metric-depth half of a stereo depth system is absent and would have to be
built. This project built it (`src/geometry/stereo.py`) in order to evaluate
depth at all.

## 2. How does the pipeline work?

```
left, right (368x1232)
  -> pad bottom/right, crop top-left, no resize        [deterministic, host]
  -> normalise, ImageNet statistics                    [compiler-inserted layer]
  -> Siamese feature extractor, shared weights, 1/16   -> 32x23x77 each
  -> "cost volume", 12 candidates, subtraction         -> 32x12x23x77
  -> 3D aggregation, 5 convolutions                    -> 12x23x77
  -> bilinear upsample to full res, then soft-argmin   -> 1x368x1232
  -> refinement guided by the left image, dilations 1/2/4/8/1/1
  -> add, ReLU                                         -> 1x368x1232 disparity
```

168 nodes, fully static shapes, batch fixed at 1 [SR-001]. Normalisation is not
in the ONNX — Hailo's parser inserts it [SR-002, SR-003] — so running the ONNX
directly requires doing it on the host, and omitting it fails silently.

## 3. What differs from the original paper?

The full paper text has **not** been read — only its abstract [SR-010] — so this
is deliberately incomplete and remains open question O2.

Established: the deployed model has **no activations in its downsampling stack**,
making it a linear operator [SR-001, pinned by a linearity test]. Its refinement
is a **single** full-resolution stage, not the multi-scale cascade the paper's
title implies. Its cost volume uses **subtraction**, contradicting the upstream
README's claim of concatenation [EXP-001]. And it performs no search at all
[EXP-010].

**Three artifacts must never be conflated:** the ECCV paper, the upstream
implementation, and the Hailo deployment. This project studied the third.

## 4. How is the cost volume constructed?

Nominally: for each of 12 candidates, shift the left features left by `k` columns
and subtract the right features, at 1/16 resolution and 32 channels, giving
`1×32×12×23×77` = 2.59 MiB [SR-001].

Actually: `Concat([features, zeros], axis=3)` then `Slice(starts=[0], ends=[77])`
returns the features unchanged. **Every slice is `left − right` at zero shift**
[EXP-010]. Construction costs zero MACs and 48 data-movement nodes.

## 5. How is disparity estimated?

Soft-argmin over 12 candidates, `Σ k · softmax(−cost)ₖ` — computed at **full
resolution**, after bilinearly upsampling the whole cost tensor to
`12×368×1232` (20.8 MiB) [SR-001]. The index grid is a baked-in 21.76 MiB
constant holding `arange(12)`, 92 % of the model file [EXP-001, EXP-006].

Because of the degenerate volume, this stage produces an estimate
**anti-correlated** with ground truth, r = −0.976 [EXP-008].

## 6. How is disparity refined?

Concatenate the estimate with the left RGB image (4 channels), then
`Conv(4→32)`, six residual blocks with dilations 1/2/4/8/1/1 at 32 channels and
full resolution, then `Conv(32→1)`, added residually and clamped by ReLU
[SR-001].

The name understates it. The refinement residual correlates with the final output
at **r = +0.9998** and supplies **76.5 %** of its magnitude [EXP-006]. It is not
polishing a nearly-correct estimate; it is inverting, rescaling and largely
producing the disparity.

## 7. How is disparity converted to depth?

In the reference: **it is not** [SR-007].

Here: `Z = fB/d`, with per-scene KITTI calibration (f 721.5 px, B 0.533 m,
fB 384.4 px·m) and explicit invalid-pixel handling — a disparity at or below zero
yields no depth rather than a large finite number [EXP-003, EXP-012].

## 8. What is computationally expensive?

**[EXP-001, EXP-013]** — static MAC share against measured RTX 4060 time:

| Stage | MACs | Time | Ratio |
|---|---:|---:|---:|
| Feature extraction ×2 | 5.1 % | 18.4 % | 3.59× |
| Cost volume construction | 0.0 % | 3.6 % | — |
| 3D aggregation | 4.2 % | 2.8 % | 0.67× |
| Upsample + soft-argmin | 0.01 % | 1.4 % | 142× |
| **Refinement** | **90.6 %** | **73.3 %** | 0.81× |

Total 56.04 GMAC, 44.4 ms, 22.5 FPS on the RTX 4060; 642.3 ms, 1.6 FPS on CPU.

Refinement dominates on both devices. MAC share overstates it.

**A third, independent line of evidence agrees.** Hailo's own compiled profiler
[SR-006, EXP-017] puts refinement at **95.2 %** of the compiled model's MACs and
identifies refinement convolutions as the eight slowest layers, with `conv50`
setting the modelled bottleneck at 43.03 FPS. Static analysis, our own profiling
and the vendor's compiler model all reach the same conclusion by different
routes.

## 9. What consumes memory?

Peak single activation **55.34 MiB** — every tensor inside refinement — against
2.59 MiB for the cost volume [EXP-001]. Total activation traffic 1,899 MiB, 90.8 %
of it in refinement. Measured peak GPU allocation 302.1 MiB [EXP-013].

For an edge accelerator the binding figure is the 55.34 MiB peak, not the total.
That is what forces Hailo's compiler to split those layers by width into up to 22
pieces [SR-003].

Learned weights are only 1.62 MiB. The 21.76 MiB index grid is 13× larger than
the entire model.

## 10. What causes accuracy failures?

Ranked in `failure_analysis.md`. The root cause is F1/B1: **no disparity search**,
which mechanically produces the anti-correlated intermediate estimate and forces
the refinement stage to carry the model.

Secondary: occluded pixels cost 0.62 D1 points and are unmarked [EXP-005]; no
confidence or validity output exists, and the final ReLU makes "no match" and
"zero disparity" the same value [SR-001].

**Refuted:** the disparity range is *not* a limitation. Ground truth reaches 9.56
of 11 candidate units and **0.0 %** of pixels exceed the range [EXP-008].

## 11. What causes depth failures?

Geometry, and it is not avoidable [EXP-012]:

| Band | Disparity EPE | Depth MAE | δ₁ |
|---|---:|---:|---:|
| 0–10 m | 1.467 px | 0.213 m | 99.1 % |
| 20–30 m | 1.290 px | 2.082 m | 93.2 % |
| 50–80 m | 1.311 px | 9.158 m | 75.5 % |

Disparity error is flat; **depth error grows 43×**. Overall δ₁ of 96.6 % is
flattered by the 81 % of pixels inside 20 m. A prediction passing the 3 px D1
test at 80 m can be wrong about distance by more than half [EXP-003].

## 12. What happens during quantisation?

**[EXP-015]**, our stack, our hardware:

| | EPE px | D1 % | Depth Abs Rel | Latency |
|---|---:|---:|---:|---:|
| fp32 CUDA | 1.314 | 8.155 | 0.0489 | 90.1 ms |
| **fp16 CUDA** | 1.310 | 8.141 | 0.0486 | **64.8 ms** |
| **int8 ORT CPU** | **1.655** | **10.945** | 0.0627 | 677.4 ms |

**fp16 is free** — the change is within noise, and it is 28 % faster. **int8
costs 2.79 D1 points**, with a maximum per-pixel change of 45 px, and shrinks the
model 3.9×.

Hailo's own float-to-hardware degradation is about 2.08 points [SR-002, SR-004].
Similar magnitude; **no inference drawn** — different quantiser, calibration and
arithmetic.

## 13. What limits edge deployment?

Peak activation size, and the operator set.

55.34 MiB tensors exceed on-chip memory, forcing spatial defusion into up to 22
pieces [SR-003, EXP-001]. What that costs is **UNKNOWN** without a device.

The operator set is otherwise conservative — no batch norm, no dynamic shapes, no
control flow — and the model compiles for Hailo-8 in full [SR-005]. The one
awkward operator is a **5D transpose** in the cost volume [SR-001].

Hailo's own compiled profiler report [SR-006], decoded in EXP-017, adds three
things that were previously UNKNOWN: the compiled model spans **6 device
contexts** (77/70/27/28/32/8 layers), quantisation is **uniform 8/8/8** across
all 242 compiled layers with no mixed precision, and the modelled throughput
bottleneck is **`conv50` at 43.03 FPS** — with the eight slowest layers in the
model all being refinement convolutions.

**Measured silicon behaviour is still UNKNOWN.** The report's `profiling_mode` is
`post_placement` and its model-level `fps` and `latency` fields are `N/A`: it is
a compiler estimate, not a run. No device was available. Our RTX 4060 and CPU
numbers are our own and are never compared with Hailo's published 10.7 FPS.

## 14. How does Hailo's deployment differ from the original architecture?

Hailo did not modify the architecture. It exported the upstream implementation
faithfully — including the degenerate shift — and compiled it. Hailo's own
contributions are the parser-inserted normalisation, the quantisation calibration
on the first 160 KITTI scenes, and the compiler directives that tile the
refinement layers [SR-002, SR-003].

The gap between "StereoNet the paper" and "StereoNet as deployed" is almost
entirely inherited from the upstream implementation [SR-011], not introduced by
Hailo.

## 15. How does StereoNet compare with other approaches?

Twelve architectures surveyed at abstract level in `competitive_landscape.md`
[SR-040 to SR-050]. **None implemented or measured.**

The field has four answers to expensive 3D convolutions: shrink the volume
(StereoNet), enrich it so aggregation can shrink (GwcNet, ACVNet), replace 3D
aggregation (AANet, CoEx), or abandon the volume (HITNet, RAFT-Stereo,
IGEV-Stereo).

**The observation that matters:** almost all published efficiency work targets
cost-volume aggregation. Here aggregation is already 4.2 % of MACs and 2.8 % of
GPU time. **Most of that literature optimises a stage that is nearly free in this
architecture**, while the expensive stage — full-resolution guided refinement —
receives far less attention.

## 16. What are the strongest research opportunities?

In `research_questions.md` §3, as falsifiable hypotheses. The first is
prerequisite to the rest.

**H1 — the headroom is unmeasured.** This architecture reaches D1 8.15 % with an
inert matching stage. What a working search is worth is unknown, and both
outcomes are informative. The corrected shift already exists behind a flag that
Phase 1 never enabled.

**H2** — aggregation-focused efficiency techniques should transfer poorly here.
**H3** — peak single-tensor size predicts edge deployability better than MACs.
**H4** — far-range depth needs its own objective, not uniform disparity error.
**H5** — confidence may only be recoverable once H1 is resolved.

## 17. What should not be changed?

- **The colour-guided dilated refinement design.** It recovers a usable disparity
  from a degenerate input, r = +0.9998 with the output [EXP-006]. It works.
- **Static shapes and the conservative operator set.** The model compiles in full
  for Hailo-8 [SR-005]. That is not free.
- **fp16.** Accuracy-neutral, 28 % faster [EXP-015].
- **Near-field accuracy.** 0.213 m mean absolute error inside 10 m [EXP-012].
- **Subtraction over concatenation.** Halves the volume at no measured cost.

## 18. Most promising directions for a new architecture

Stated as directions, **not designs**, and none implemented.

1. Resolve H1 first. Nothing else can be sized until it is known what this
   architecture does with a working cost volume.
2. Attack full-resolution refinement, since that is where 90 % of the arithmetic
   and 73 % of the time are — not aggregation, where the literature is.
3. Treat peak single-tensor size as a first-class design constraint.
4. Make far-range sub-pixel precision an explicit objective with its own metric.
5. Add a validity output; for a depth system it is arguably a requirement.
6. Measure on target silicon before making any efficiency claim.

---

## Completion gate

| # | Item | Status |
|---|---|---|
| 1 | Environment reproducible | **done** — `experiments/_env_baseline_primary.json` |
| 2 | Sources registered | **done** — 29 entries, every cited one inspected |
| 3 | Stereo fundamentals documented | **done** — `stereo_fundamentals.md`, EXP-003 |
| 4 | Complete pipeline reconstructed | **done** — `reference_pipeline.md` |
| 5 | Hailo artifacts preserved and inspected | **done** — hashed manifest; profiler report decoded (EXP-017); HEF acquired, not decoded |
| 6 | Architecture reconstructed | **done** — 168-node table, EXP-001 |
| 7 | Cost volume understood | **done** — `cost_volume_analysis.md`, EXP-010 |
| 8 | Independent implementation exists | **done** — `src/models/stereonet` |
| 9 | Unit tests pass | **done** — 29 tests |
| 10 | ONNX equivalence tested | **done** — relative 1e-7 at every stage, EXP-011 |
| 11 | Discrepancies documented | **done** — register in `reproduction_report.md` §6 |
| 12 | KITTI protocol frozen | **done** — four variants, EXP-005 |
| 13 | Hailo figure reproduction attempted | **done** — exact: 8.2237 vs 8.223 |
| 14 | Depth evaluation validated | **done** — EXP-012, range-binned |
| 15 | Short training convergence demonstrated | **done with a correction** — loss fell 1.51×, gradients finite; **validation did not improve**, see `experiments/EXP-016/CORRECTION.md` |
| 16 | Compute profiling complete | **done** — EXP-013, EXP-014 |
| 17 | Memory profiling complete | **done** — EXP-001, EXP-013 |
| 18 | Failure analysis complete | **partial** — ten findings ranked; scene-class failures need Middlebury, **UNKNOWN** |
| 19 | Quantisation analysis complete | **done** — EXP-015 |
| 20 | Hardware/compiler analysis complete | **partial** — all public artifacts now analysed including the profiler report; **measured** on-device behaviour remains **UNKNOWN**, no device |
| 21 | Competitor landscape documented | **done at abstract level** — no competitor implemented or measured |
| 22 | Bottlenecks ranked | **done** — nine, `bottleneck_report.md` |
| 23 | Research questions answered | **done** — 20 answered, 16 open |
| 24 | Unresolved items marked UNKNOWN | **done** |

**Three items are incomplete and are not presented otherwise:** scene-class
failure analysis (18), *measured* on-device hardware behaviour (20), and
competitor measurement (21). Item 15 carries a correction that withdraws an
overstated conclusion.

### Release verification

Run before tagging, all green:

- `python scripts/verify_claims.py` — **53 checks, 0 failures**. Asserts that
  every headline number in `docs/` and this report matches the corresponding
  `experiments/EXP-xxx/metrics.json`, so a transcription error cannot reach the
  release.
- `python -m pytest tests/ -q` from a **fresh clone of the committed tree** —
  28 passed, 1 skipped. The skip is the KITTI calibration test, which correctly
  skips when the dataset is absent; the suite therefore needs no untracked file.
- All 17 experiments present, contiguous, `status: completed`, every recorded
  git commit resolving into this history, no missing charter-required config
  field.
- Secret scan over every tracked file: no keys, tokens, credentials, emails or
  absolute home paths.

---

## Integrity notes

Things that went wrong, recorded rather than tidied away.

- **EXP-002 was superseded by EXP-003.** It reported an undefined depth as
  3.8×10¹¹ m because a non-positive disparity was clamped. Both runs kept.
- **EXP-008's `supports` field is unreliable** — a crude threshold rule produced
  a wrong classification. Annotated in place, metrics unaltered.
- **EXP-009 was superseded by EXP-011.** It ran before the degenerate shift was
  understood and diverged at the cost volume. That divergence is what led to the
  finding. Both kept.
- **EXP-016's conclusion was corrected.** It claimed validation improved; the
  numbers show noise. `experiments/EXP-016/CORRECTION.md`.
- **Two arXiv identifiers were wrong** and returned unrelated papers. Discarded
  rather than cited, and recorded in the registry.
- **`scripts/verify_claims.py` initially passed a check for the wrong reason.**
  A bare `fB` pattern matched "surfboard" in a leftover COCO class table in the
  Hailo app's `common.h`, and a shell heredoc had collapsed `\b` into literal
  backspace bytes, making several regexes unmatchable and their checks vacuous.
  Both were found and fixed before tagging; the underlying finding was
  unaffected. The leftover class table is itself further evidence the
  application is an adapted classification example.
- **Two derived binaries were committed by mistake** — a 6.1 MB quantised ONNX
  and a 1.7 MB training checkpoint, 7.8 MB of the 8.4 MB tracked total. Both are
  regenerable from the committed code and the hashed reference inputs. They are
  now untracked and ignored, with their sizes and SHA-256 recorded in
  `results/derived_artifacts.json`. **The blobs remain in git history**: removing
  them would require rewriting history, which would invalidate the commit hashes
  every experiment records, so the traceability was judged worth more than the
  7.8 MB.

No experiment was deleted. No result was tuned toward a target. The baseline was
frozen after EXP-011 and not modified thereafter.

---

## Reproducing everything

```
python scripts/hash_reference.py            # verify artifacts
python -m pytest tests/ -q                  # 29 tests
python scripts/exp_onnx_forensics.py        # EXP-001
python scripts/depth_error_curve.py         # EXP-003
python scripts/exp_reproduce_hailo.py       # EXP-005, the 8.223 reproduction
python scripts/exp_intermediate_tensors.py  # EXP-006
python scripts/exp_right_image_ablation.py  # EXP-007
python scripts/exp_cost_sign_convention.py  # EXP-008
python scripts/exp_degenerate_cost_volume.py# EXP-010, the central finding
python scripts/exp_onnx_equivalence.py      # EXP-011
python scripts/exp_depth_validation.py      # EXP-012
python scripts/exp_profile.py --device cuda # EXP-013
python scripts/exp_quantization.py          # EXP-015
python scripts/exp_train_convergence.py     # EXP-016
python scripts/extract_profiler_report.py   # decode Hailo's profiler HTML
python scripts/exp_profiler_report.py       # EXP-017
python scripts/verify_claims.py             # 53 claim checks against the records
```

Data: KITTI 2015 (`data_scene_flow.zip`, `data_scene_flow_calib.zip`) into
`data/kitti2015/`. Artifacts download into `reference/` and are hashed.

---

## Phase 2 may begin

The gate is met on 21 of 24 items, with the three shortfalls named and their
causes stated. The reference system is understood well enough to say what it
does, how it works, why it works, where it fails, where it is expensive and why —
and, unexpectedly, that it achieves its published accuracy without doing the one
thing its architecture exists to do.

**Phase 2 opens with H1.** Until it is known what this architecture does with a
working cost volume, nothing about its limits is settled.
