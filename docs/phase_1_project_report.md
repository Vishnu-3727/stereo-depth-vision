# Phase 1 Project Report

**Microchip Stereo Depth Vision — reverse engineering, reproduction and
characterisation of the Hailo StereoNet reference deployment.**

Prepared at the close of Phase 1. Every figure traces to an experiment record
(`EXP-xxx`) or a registered source (`SR-xxx`); anything that could not be
established is marked UNKNOWN rather than estimated.

This is the project-level report. The charter-question deliverable is
[`../PHASE_1_FINAL_REPORT.md`](../PHASE_1_FINAL_REPORT.md); the technical detail
is in the twelve knowledge-base documents alongside this one.

---

## 1. Executive summary

Phase 1 set out to understand the Hailo StereoNet edge deployment well enough to
design against it. It did that, and found something that changes what the
baseline means.

**The deployed model performs no disparity search.** Its cost volume — the
component whose entire purpose is to compare the two camera views at different
candidate shifts — produces twelve bit-identical slices. The maximum difference
between any slice and the first is exactly 0.0, across every scene tested. A
zero-padding block is appended to the right of the feature map and then sliced
straight back off, so the intended shift never happens. What the network actually
receives is a single photometric difference at zero shift, copied twelve times.

The model still reaches a respectable 8.15 % D1 error on KITTI 2015. **It achieves
that without stereo correspondence search.**

Two further findings reframe how the baseline should be read:

**The published accuracy figure is not what its label says.** Hailo publishes
"EPE 8.223" for this model. That number is the KITTI D1 outlier *percentage*, not
an end-point error — Hailo's own evaluator names the variable
`three_pixel_correct_rate`. The model's actual end-point error is **1.313 px**.
Anyone comparing 8.223 against another architecture's EPE would conclude the
model is roughly six times worse than it is.

**Operation counts do not predict speed.** Across pipeline stages, the error
between share-of-MACs and share-of-runtime ranges from 0.67× to 142×; one stage
performing zero multiply-accumulates consumes 3.6 % of GPU time; and the ranking
of which stage matters changes by 6× between a GPU and a CPU running the same
graph. Any efficiency claim about edge silicon made from FLOP counts alone is
unsupported.

Alongside the findings, Phase 1 delivered a working reconstruction: an
independent implementation that matches the reference to a relative 1e-7 at every
stage, and an evaluation framework that reproduces Hailo's published figure to
four significant figures (8.2237 against 8.223).

---

## 2. What was asked, and what was delivered

The brief was explicitly forensic: understand the reference system, do not
improve it. That boundary was held throughout — the baseline was frozen after the
equivalence check and never modified, and every weakness found was recorded
rather than fixed.

| Asked for | Delivered |
|---|---|
| Reverse-engineer the reference system | Complete pipeline reconstructed from the executable artifact, not from documentation |
| Reproduce it independently | Implementation written from the specification, matching to 1e-7 |
| Validate accuracy | Published figure reproduced exactly; full protocol recovered |
| Measure compute and memory | Profiled per stage on two devices, plus static analysis and the vendor's own compiler report |
| Understand failure modes | Ten ranked findings with evidence and confidence |
| Study the competitive landscape | Twelve architectures, each read from its primary source |
| Identify bottlenecks | Nine ranked by evidence, none solved |
| Produce an evidence-backed knowledge base | Twelve documents, eighteen experiment records |

---

## 3. Method

The work was governed by an evidence discipline that turned out to matter more
than any single technique.

**Everything is tagged.** `SOURCE` (someone else published it), `MEASUREMENT` (we
measured it, with an experiment ID), `INFERENCE` (derived, derivation stated),
`HYPOTHESIS` (proposed, not established), `UNKNOWN` (could not be established).
These never merge. Our RTX 4060 timings are never presented as Hailo silicon
performance, and Hailo's published figures are never presented as ours.

**Artifacts outrank documentation.** When sources disagreed, the ordering was:
the executable ONNX first, then the compiled binary, then the compiler script,
then the upstream source, then configuration, then documentation, then the paper.
This mattered repeatedly — it is why the cost-volume defect was found at all. The
upstream README describes a concatenation-based cost volume; the ONNX uses
subtraction; and neither description mentions that the shift does nothing.

**Every measurement is an immutable record.** Eighteen experiment directories,
each stamped with the git commit, configuration, hardware and software versions.
None was deleted, including the ones that turned out to be wrong.

**Reproduction before interpretation.** The published figure was reproduced
before any conclusion was drawn about the model's quality, which is what exposed
the metric mislabelling.

---

## 4. Findings

### 4.1 No disparity search — the central finding

Each disparity level of the cost volume is built as:

```
Concat([left_features (width 77), zeros (width k)], axis=3)   -> width 77+k
Slice(starts=[0], ends=[77], axes=[3])                        -> width 77
Sub(that, right_features)
```

Concatenating on the **right** and then slicing `[0:77]` returns the input
unchanged. Verified by execution: every slice is bit-identical to slice 0,
difference exactly 0.0 [EXP-010]. The same construction is present in the
upstream PyTorch source, so it originates there and was carried faithfully
through export and compilation — neither introduced it.

**It is not that the model ignores the second camera.** Corrupting the right
image costs up to 90 D1 points, driving error from 9.33 % to 99.29 % [EXP-007].
`left − right` at zero shift is a real stereo cue: near surfaces differ a lot
between the views at zero shift, far ones differ little. The network learned to
read depth out of that difference magnitude plus the left image, rather than by
searching for correspondences.

This single cause explains a set of anomalies found earlier and separately:

| Anomaly | Explanation |
|---|---|
| The intermediate disparity estimate is anti-correlated with truth, r = −0.976 [EXP-006, EXP-008] | The cost tensor carries no disparity information, so reducing it has no consistent meaning |
| The refinement stage supplies 76.5 % of the output magnitude and correlates with it at r = +0.9998 [EXP-006] | Nothing upstream of it estimates disparity, so it has to |
| Neither a cost nor a score reading of the aggregated tensor fits [EXP-008] | There is nothing coherent to read |

**Consequence.** The baseline remains valid evidence about *achievable accuracy*
and about *compute behaviour*. It is not valid evidence about cost-volume design:
nothing it demonstrates concerning disparity range, candidate count or 3D
aggregation reflects a working cost volume. It also means the architecture holds
unquantified headroom.

### 4.2 The published metric is mislabelled

Hailo's configuration declares `eval_metric: EPE` with
`full_precision_result: 8.223`. The evaluator computes:

```python
correct = (diff < 3) | (diff < true_disp * 0.05)
three_pixel_correct_rate = 1 - sum(correct) / len(index)
```

That is the KITTI D1 outlier rate, averaged per image and displayed as a
percentage. Reproducing it required recovering the whole protocol — the last 40
training scenes, `disp_occ_0` ground truth, a top-left 368×1232 crop with no
resize, and ground truth divided by 255 rather than KITTI's 256.

**Result: 8.2237 against a published 8.223**, a gap of +0.0007 points [EXP-005].
The model's true EPE on the same predictions is **1.313 px**, which sits
comfortably in the range the original StereoNet paper reports. The apparent
sevenfold disagreement with the literature was a units problem throughout.

### 4.3 Where the computation actually goes

The name "StereoNet" suggests the 3D cost volume is expensive. It is not.

| Stage | Share of MACs | Share of GPU time | Ratio |
|---|---:|---:|---:|
| Feature extraction (×2) | 5.1 % | 18.4 % | 3.59× |
| Cost volume construction | **0.0 %** | 3.6 % | — |
| 3D aggregation | 4.2 % | 2.8 % | 0.67× |
| Upsample + soft-argmin | 0.01 % | 1.4 % | **142×** |
| **Refinement (full resolution)** | **90.6 %** | **73.3 %** | 0.81× |

Total 56.04 GMAC; 44.4 ms and 22.5 FPS on an RTX 4060, 642.3 ms and 1.6 FPS on
CPU [EXP-013, EXP-014].

**Three independent lines of evidence agree that refinement dominates**: our
static analysis (90.6 % of MACs), our measured profiling on two devices
(73–76 % of time), and Hailo's own compiler report, which puts refinement at
90.6 % of the compiled model's operations — matching our ONNX figure to within
0.02 percentage points — and identifies refinement convolutions as the eight
slowest layers [EXP-018].

**Two cautions come out of the same data.** MAC share *overstates* refinement,
and the ranking is device-dependent: 3D aggregation is 2.8 % of GPU time and
17.2 % of CPU time. Two general-purpose processors disagree by 6× about which
stage matters, so extrapolating either to a fixed-function accelerator is
unjustified.

### 4.4 Memory is about peak tensor size, not total footprint

Every activation inside the refinement stage is **55.34 MiB**, and several are
live simultaneously through the residual connections. That is what forces Hailo's
compiler to split those layers by image width into as many as 22 pieces, with
re-concatenation trees and multi-way shortcut fanout.

By contrast the whole cost volume is 2.59 MiB, and the learned weights are
1.62 MiB. Parameter count is not the constraint; peak single-tensor size is.

A related oddity: **92 % of the 23.7 MB model file is one constant** — a
`1×12×368×1232` tensor holding twelve distinct values, `arange(12)`, tiled across
the image. A broadcast against a 12-element vector would be numerically identical
and occupy 48 bytes. It is an export artifact, and because of §4.1 it weights a
softmax over twelve identical inputs.

### 4.5 Depth error is dominated by geometry

The project's target is metric depth, not disparity, so disparity metrics are
only half the picture.

| Depth band | Disparity EPE | Depth error | Within 25 % |
|---|---:|---:|---:|
| 0–10 m | 1.467 px | **0.213 m** | 99.1 % |
| 10–20 m | 1.158 px | 0.639 m | 97.8 % |
| 20–30 m | 1.290 px | 2.082 m | 93.2 % |
| 30–50 m | 1.247 px | 4.384 m | 87.1 % |
| 50–80 m | 1.311 px | **9.158 m** | 75.5 % |

**Disparity error is flat with range; depth error grows 43×** [EXP-012]. This is
the `ΔZ ≈ Z²/(fB)·Δd` relation, derived analytically and confirmed against a
numerical derivative, now observed in a real model's output.

It has a practical edge: aggregate metrics hide it. Overall accuracy looks strong
(96.6 % within 25 %) because 81 % of ground-truth pixels lie inside 20 m. At
50–80 m, one pixel in four fails. A prediction that passes KITTI's 3-pixel
correctness test at 80 m can still be wrong about the distance by more than half.

### 4.6 Precision

**fp16 is free** — accuracy change within noise, 28 % faster. **int8 costs 2.79
D1 points** on our stack, with a maximum per-pixel change of 45 px, and shrinks
the model 3.9× [EXP-015]. Hailo's own float-to-hardware degradation is about 2.08
points; the magnitudes are similar, but the quantiser, calibration and arithmetic
all differ, so no inference is drawn from the resemblance.

### 4.7 The reference is a disparity system, not a depth system

The shipped Hailo application's entire postprocess is one line casting the
network output to an 8-bit image. There is no calibration, no baseline, no focal
length and no `Z = fB/d` anywhere in it. Synchronisation, calibration and
rectification are all assumed and never checked — an unrectified pair would fail
silently, producing a plausible and wrong result.

The 8-bit cast also discards the sub-pixel precision the architecture's
soft-argmin exists to produce, which by §4.5 is exactly what far-range depth
depends on.

This project implemented the missing geometry in order to evaluate depth at all.
That component is an evaluation tool, not a deployable depth service.

---

## 5. What was built

| Component | Purpose |
|---|---|
| Independent StereoNet implementation | Written from the reconstructed specification, not ported. Matches the reference to a relative 1e-7 at every stage [EXP-011] |
| Evaluation framework | Four protocol variants, keeping Hailo's convention and KITTI's official one explicitly apart |
| Stereo geometry module | Calibration parsing, `Z = fB/d`, and error propagation, with invalid pixels excluded rather than clamped |
| Experiment recorder | Allocates immutable `EXP-xxx` directories, captures environment automatically, refuses to overwrite |
| Analysis tooling | ONNX graph inspection, per-layer cost model, profiler decoding, claim verification |
| Test suite | 75 tests covering shapes, cost volume, regression, refinement, geometry, protocol, conclusion logic and profiler stage mapping |

---

## 6. Integrity record

Five errors were made during Phase 1 and corrected. All are recorded rather than
tidied away, because the record of how a conclusion was reached is part of the
evidence.

| Error | Correction |
|---|---|
| An undefined depth reported as 3.8×10¹¹ m, from clamping a non-positive disparity | Geometry returns `inf`; EXP-002 superseded by EXP-003, both kept |
| An automated verdict produced by a crude threshold rule | Annotated in place, metrics unaltered |
| An equivalence check that diverged at the cost volume | Superseded by EXP-011 — and that divergence is what led to the central finding |
| **A training conclusion asserted rather than measured** | See below |
| **A wrong stage mapping in the compiler-profiler analysis**, reporting refinement as 95.2 % of compiled MACs | Caught by an external audit. Corrected in EXP-018 to 90.6 %; EXP-017 preserved and withdrawn |
| Two wrong arXiv identifiers returning unrelated papers | Discarded rather than cited, and recorded in the source registry |

The fourth deserves its own note, because it is the failure mode this whole
discipline exists to prevent.

**EXP-016 recorded that validation improved while its own validation error rose
from 18.497 px to 19.216 px.** The sentence was a string literal in the
experiment script — it did not depend on the run at all, so re-running would have
recreated the false claim indefinitely. Two neighbouring claims had the same
defect.

The corrective audit replaced all three with values derived from the
measurements, added twenty regression tests including a check that the script
contains no hard-coded outcome claim, and extended the claim verifier to detect
the class of error across all experiments. Both the guard and the new verifier
checks were confirmed to fail against the defective version recovered from the
earlier tag, so they are not decorative. EXP-016's original record is unaltered;
its correction document now also explains the fix.

A second correction followed, and it is the more instructive one. An external
audit found that the compiler-profiler analysis had mapped Hailo's compiled
convolutions to the wrong architectural stages, folding the right
feature-extractor branch and the real 3D aggregation into refinement and
reporting refinement as 95.2 % of compiled operations.

**The withdrawn figure disagreed with our own ONNX-derived 90.6 % by nearly five
points, and that discrepancy was visible at the time and was not questioned.**
Both numbers supported the same qualitative conclusion — refinement dominates —
and that agreement in direction masked a disagreement in magnitude. Two
independent routes to the same quantity giving different answers is exactly the
signal this project's evidence discipline exists to surface, and it was missed.

The corrected analysis (EXP-018) puts refinement at 90.6 %, aggregation at 4.2 %
and feature extraction at 5.1 %, agreeing with the ONNX analysis to within 0.02
percentage points. The mapping now lives in a unit-tested module documented
against the layer shapes, the analysis raises rather than records if the two
routes diverge by more than a percentage point, and twenty-six tests pin the
range boundaries and the defusion-name handling. EXP-017's record is unaltered
and carries its withdrawal.

---

## 7. What is incomplete

Stated plainly so it is not mistaken for finished work.

| Gap | Status | What it would take |
|---|---|---|
| Scene-class failure analysis — textureless, repetitive, reflective, thin structures, boundaries | **NOT COMPLETED** | Middlebury 2014, roughly 5 GB |
| Hailo silicon behaviour | **UNKNOWN** | A physical device. The vendor's profiler report is a post-placement compiler estimate, not a measured run — its model-level FPS and latency fields are `N/A` |
| Competitor benchmarking | **NOT PERFORMED** | All twelve entries are abstract-level source reading; none was implemented or run |
| Full-scale training | **NOT PERFORMED** | EXP-016 is a short convergence proof only, and its validation did not improve. The recipe is recorded |
| The original StereoNet paper in full | Read at abstract level only | The paper-versus-deployment comparison stays partial until the full text is read |

---

## 8. What this means for Phase 2

**The first question is not an optimisation question.** It is: what is a working
cost volume worth?

The baseline reaches 8.15 % D1 with its matching stage inert. Whether restoring
the search yields a large gain or almost none is genuinely open, and the answer
determines everything downstream — including how much of the refinement stage's
90 % compute share is spent repairing the defect rather than doing useful work.
The corrected shift already exists in the implementation behind a configuration
flag that Phase 1 never enabled. It should stay disabled until this is measured
deliberately, under the frozen protocol, against the frozen baseline.

Three further directions follow from the measurements:

**Target the right stage.** Published efficiency work overwhelmingly optimises
cost-volume aggregation. In this architecture aggregation is already 4.2 % of
operations and 2.8 % of runtime, while the expensive stage receives far less
attention in the literature. Techniques borrowed from that body of work should be
expected to transfer poorly here, and that expectation is testable.

**Design against peak tensor size.** Not parameter count, not FLOPs. Peak single
activation is what forced the vendor's compiler to tile this model, and it is the
quantity an edge accelerator actually constrains.

**Give far-range depth its own objective.** Uniform disparity error produces
wildly non-uniform depth error. A metric that averages over the image will keep
reporting success while the far field degrades.

Finally, a methodological constraint that Phase 1 established with unusual
clarity: **no claim about edge efficiency should be made without target-silicon
measurement.** Two general-purpose processors already disagree by 6× about which
stage dominates. Obtaining a device is the single highest-value unblocking step
available.

---

## 9. Experiment index

| ID | Subject |
|---|---|
| EXP-001 | Static forensics of the reference ONNX; parameter and operation accounting |
| EXP-002 | Depth error propagation *(superseded by EXP-003)* |
| EXP-003 | Depth error propagation on real KITTI calibration |
| EXP-004 | Protocol reproduction, pilot |
| EXP-005 | **Full reproduction of the published figure; protocol recovery** |
| EXP-006 | Stage-by-stage tensor magnitudes inside the reference |
| EXP-007 | Right-image ablation |
| EXP-008 | Cost tensor sign convention |
| EXP-009 | Equivalence check *(superseded by EXP-011)* |
| EXP-010 | **The degenerate cost volume** |
| EXP-011 | Tensor-level equivalence, independent implementation vs reference |
| EXP-012 | Metric depth accuracy, binned by range |
| EXP-013 | Per-stage latency and memory, RTX 4060 |
| EXP-014 | Per-stage latency, CPU |
| EXP-015 | Precision study: fp32 / fp16 / int8 |
| EXP-016 | Training convergence *(conclusion corrected)* |
| EXP-017 | Hailo compiled profiler report, decoded *(per-stage rollup corrected)* |
| EXP-018 | **Profiler report re-analysed with the corrected stage mapping** |

---

## 10. Verification status

| Check | Result |
|---|---|
| Test suite | **75 passed** (74 passed, 1 intentional dataset skip from a clean clone) |
| Claim verification | **73 checks, 0 failures** — every headline figure asserted against its experiment record |
| Experiment records | 18, contiguous, all completed, every git reference valid, none deleted |
| Secrets and credentials | None |
| Datasets, checkpoints, large binaries | None tracked; 1.16 MB across 173 files |
| Git history | Preserved, never rewritten |
| Tags | `phase-1` (original freeze), `phase-1-final` (corrective release) |

---

*Sources SR-001 to SR-050, experiments EXP-001 to EXP-017. Detailed technical
treatment in the twelve knowledge-base documents alongside this one; charter
questions answered
in [`../PHASE_1_FINAL_REPORT.md`](../PHASE_1_FINAL_REPORT.md).*
