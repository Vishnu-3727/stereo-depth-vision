# Bottleneck report

What actually limits the reference stereo-depth system, ranked by evidence.

This answers *what is limiting the system*, not *what would make it better*. Each
entry names a possible Phase 2 direction, and **none of them is implemented**.
The baseline is frozen.

---

## Ranking

| # | Bottleneck | Category | Severity | Confidence | Evidence |
|---|---|---|---|---|---|
| **B1** | The cost volume performs no disparity search | Accuracy / architectural | **Critical** | VERIFIED | EXP-010 |
| **B2** | Full-resolution refinement dominates compute and memory | Computational + bandwidth | High | MEASUREMENT | EXP-001, EXP-013, EXP-014, EXP-017, SR-003 |
| **B3** | Depth error grows quadratically with range | Depth-range | High | MEASUREMENT | EXP-003, EXP-012 |
| **B4** | Peak activation of 55.34 MiB forces the compiler to tile | Memory / hardware | High | MEASUREMENT + SOURCE | EXP-001, SR-003 |
| **B5** | No confidence or validity signal exists | Occlusion / accuracy | Medium | VERIFIED | SR-001, EXP-005 |
| **B6** | int8 costs 2.79 D1 points on our stack | Quantisation | Medium | MEASUREMENT | EXP-015 |
| **B7** | MAC count does not predict latency | Methodological | Medium | MEASUREMENT | EXP-013, EXP-014 |
| **B8** | The soft-argmin stage is 142× over-represented in time | Bandwidth | Low–Medium | MEASUREMENT | EXP-013 |
| **B9** | 92 % of the model file is a redundant constant | Deployment | Low | VERIFIED | EXP-001, EXP-006 |

---

## B1 — The cost volume performs no disparity search. **Critical.**

**Problem.** All twelve disparity slices are bit-identical. The model has no
correspondence search.

**Evidence.** The shift is built as `Concat([features, zeros], axis=3)` followed
by `Slice(starts=[0], ends=[77])`, which returns the features unchanged
[VERIFIED: SR-001]. Confirmed by execution: maximum absolute difference between
any slice and slice 0 is exactly 0.0 across every scene tested
[MEASUREMENT: EXP-010]. The same construction is in the upstream source
[SOURCE: SR-011].

**Measurement.** The model nonetheless reaches D1 8.154 %, EPE 1.313 px, depth
Abs Rel 0.0489 [MEASUREMENT: EXP-005, EXP-012]. It genuinely uses the right
image — corrupting it costs up to 90 D1 points [MEASUREMENT: EXP-007] — but as a
photometric difference at zero shift, not a search.

**Severity: critical**, for two reasons that pull in opposite directions and both
matter. It invalidates the baseline as evidence about cost-volume design: nothing
learned here about disparity range, candidate count or 3D aggregation reflects a
working cost volume. And it means the architecture has unquantified headroom —
this accuracy was reached with the matching stage inert.

**Reproducibility.** Deterministic and exact. `python
scripts/exp_degenerate_cost_volume.py`.

**Mechanism: verified, not suspected.** The slice bounds are read directly from
the graph.

**Possible Phase 2 direction.** Quantify what a working search is worth by
measuring both variants under one protocol — the implementation already carries
the corrected shift behind a flag that Phase 1 never enables. The result bounds
how much of this architecture's headroom is real. **Not implemented.**

---

## B2 — Full-resolution refinement dominates. **High.**

**Problem.** Thirteen convolutions at 368×1232 × 32 channels carry the model.

**Evidence and measurement.**

| | Refinement share |
|---|---:|
| MACs | 90.6 % (50.79 G of 56.04 G) [EXP-001] |
| Activation traffic | 90.8 % (1,724 MiB of 1,899 MiB) [EXP-001] |
| GPU time | 73.3 % (32.6 ms of 44.4 ms) [EXP-013] |
| CPU time | 75.8 % (487.0 ms of 642.3 ms) [EXP-014] |
| Parameters | 26.5 % (112,449 of 423,586) [EXP-001] |

Hailo's compiler spatially defuses exactly these layers into up to 22 pieces
[SOURCE: SR-003] — independent corroboration from the vendor's own toolchain.

**A third line of evidence, from Hailo's compiled profiler** [SOURCE: SR-006,
EXP-017]: refinement is **95.2 %** of the compiled model's MACs, and the eight
slowest layers by modelled throughput are all refinement convolutions, with
`conv50` setting the bottleneck at 43.03 FPS. The first non-refinement layer
appears ninth. Static analysis, measured profiling on two devices, and the
vendor's own compiler model agree.

**Note the honest caveat:** MAC share *overstates* the runtime share, 90.6 %
against 73.3 %. These are dense 3×3 convolutions, which run near peak efficiency.

**Why it is structural, not incidental.** This is the direct consequence of
StereoNet's core trade: match cheaply at 1/16 resolution, then spend the savings
on a learned upsampler. The upsampler became the model. And by B1, it is doing
even more than intended, because nothing upstream estimates disparity — the
refinement residual alone correlates with the output at r = +0.9998 and supplies
76.5 % of its magnitude [MEASUREMENT: EXP-006].

**Reproducibility.** Stable: σ = 0.069 ms across 30 GPU runs.

**Possible Phase 2 direction.** The refinement stage is where any compute saving
has to come from. Whether it can shrink depends on how much of its work is
repairing B1 rather than genuinely refining — which makes B1 the prerequisite
question. **Not implemented.**

---

## B3 — Depth error grows quadratically with range. **High.**

**Problem.** Uniform disparity accuracy does not give uniform depth accuracy.

**Measurement** [EXP-012], 3.8 M valid pixels:

| Band | Disparity EPE | Depth MAE | δ₁ |
|---|---:|---:|---:|
| 0–10 m | 1.467 px | 0.213 m | 99.1 % |
| 10–20 m | 1.158 px | 0.639 m | 97.8 % |
| 20–30 m | 1.290 px | 2.082 m | 93.2 % |
| 30–50 m | 1.247 px | 4.384 m | 87.1 % |
| 50–80 m | 1.311 px | 9.158 m | 75.5 % |

**Disparity error is flat; depth error grows 43×.** Predicted analytically by
`ΔZ ≈ Z²/(fB)·Δd` [EXP-003] and now observed in a real model.

**Cause: geometry.** No architecture escapes it. What an architecture controls is
`Δd`, and the quadratic penalty means sub-pixel accuracy at small disparities is
worth far more than the same gain up close.

**Severity: high** because it is invisible in aggregate metrics. Overall δ₁ is
96.6 %, but 81 % of pixels sit inside 20 m; at 50–80 m one pixel in four fails.

**Possible Phase 2 direction.** Treat far-range sub-pixel precision as a
first-class objective with its own metric, rather than trusting an image-averaged
disparity number. **Not implemented.**

---

## B4 — Peak activation forces the compiler to tile. **High.**

**Problem.** Every refinement activation is 55.34 MiB, and several are live at
once through the residual connections.

**Evidence.** 55.34 MiB peak single activation, 302.1 MiB measured peak GPU
allocation [MEASUREMENT: EXP-001, EXP-013]. `stereonet.alls` splits `conv42`–
`conv52` by width into up to 22 pieces with re-concatenation trees and multi-way
shortcut fanout [SOURCE: SR-003].

**INFERENCE, clearly labelled:** that the defusion is *caused* by activation size
is not stated by Hailo. It is a strong inference — the defused layers are exactly
those with 55 MiB activations, and no other layer is defused — but it is an
inference.

**UNKNOWN:** what defusion costs in latency. Halo exchange, extra traffic and
scheduling overhead are all plausible, none measurable without the device.

**Possible Phase 2 direction.** Peak single-tensor size, not parameter count or
total footprint, is the memory quantity an edge accelerator cares about. It
deserves to be a design constraint from the start. **Not implemented.**

---

## B5 — No confidence or validity signal. **Medium.**

**Problem.** One output, disparity. No confidence, no occlusion mask, no validity
[VERIFIED: SR-001]. The final ReLU makes "no match" and "zero disparity"
identical — and zero disparity means infinite depth.

**Measurement.** Occluded pixels cost 0.62 D1 points and are unmarked
[EXP-005]. By B1 there is no cost curve to derive confidence from, so it cannot
be bolted on.

**Possible Phase 2 direction.** A validity output is arguably a requirement for a
depth system rather than a feature. **Not implemented.**

---

## B6 — int8 costs 2.79 D1 points. **Medium.**

**Measurement** [EXP-015], onnxruntime static QDQ, per-channel, 16 calibration
scenes:

| | fp32 | fp16 | int8 |
|---|---:|---:|---:|
| EPE px | 1.313 | 1.310 | 1.655 |
| D1 % | 8.154 | 8.141 | 10.945 |
| Depth Abs Rel | 0.0489 | 0.0486 | 0.0627 |
| Max per-pixel change | — | 1.12 px | **45.13 px** |

**fp16 is free.** int8 costs 2.79 D1 points and the damage is not uniform.

**Scope, stated firmly.** This is our quantiser on our runtime. Hailo's own
float-to-hardware degradation is about 2.08 points [SOURCE: SR-002, SR-004]. The
magnitudes are similar; **no inference is drawn from that**, because the
quantiser, calibration and arithmetic all differ.

**Possible Phase 2 direction.** Identify which stage loses the accuracy. The
45 px maximum change suggests a small number of pixels move enormously, which
points at a specific tensor rather than uniform degradation. **Not implemented.**

---

## B7 — MAC count does not predict latency. **Medium, methodological.**

**Problem.** Designing against FLOPs would optimise the wrong things.

**Measurement** [EXP-013], RTX 4060:

| Stage | MAC share | Time share | Ratio |
|---|---:|---:|---:|
| Feature extraction | 5.1 % | 18.4 % | **3.59×** |
| Cost volume | **0.0 %** | 3.6 % | undefined |
| Aggregation | 4.2 % | 2.8 % | 0.67× |
| Soft-argmin | 0.01 % | 1.4 % | **142×** |
| Refinement | 90.6 % | 73.3 % | 0.81× |

Errors run from 0.67× to 142×, in both directions, and one stage with zero MACs
takes measurable time.

**Stronger still, the ranking is device-dependent**: 3D aggregation is 2.8 % of
GPU time and 17.2 % of CPU time [EXP-013, EXP-014]. Two general-purpose
processors disagree by 6× on which stage matters.

**Possible Phase 2 direction.** No architecture claim about edge efficiency is
credible without target-silicon measurement. This is the strongest argument in
the whole report for obtaining a device. **Not implemented.**

---

## B8 — Soft-argmin is 142× over-represented in time. **Low–Medium.**

**Problem.** A 16× bilinear upsample of the cost tensor to `12×368×1232`
(20.8 MiB), then softmax, multiply against a 21.76 MiB constant, and reduce — all
at full resolution [VERIFIED: SR-001].

**Measurement.** 0.633 ms on GPU (1.4 %), 14.495 ms on CPU (2.3 %), for 0.01 % of
the MACs [EXP-013, EXP-014]. Pure memory traffic.

Reducing at 1/16 resolution and upsampling a single-channel result afterwards
would touch 1/192 as much data. **Whether accuracy would differ is UNKNOWN** and
deliberately untested — it would mean changing the frozen baseline.

**Possible Phase 2 direction.** Measure the ordering as a design choice.
**Not implemented.**

---

## B9 — 92 % of the model file is a redundant constant. **Low.**

`/Tile_output_0`, `1×12×368×1232` float32, 21,762,048 bytes — against 1,694,344
bytes of learned weights [MEASUREMENT: EXP-001]. It holds twelve distinct values,
`arange(12)`, constant across height and width [MEASUREMENT: EXP-006]. A
broadcast against a 12-element vector would be numerically identical at 48 bytes.

An export artifact of building the grid with `repeat()` at runtime
[SOURCE: SR-011]. Low severity — the compiler may well fold it — but it inflates
the artifact 14×, and by B1 it weights a softmax over twelve identical inputs.

---

## Categories with no bottleneck found

- **Disparity range.** Ground-truth disparity reaches 9.56 of 11 candidate units
  and **0.0 %** of pixels exceed the range [MEASUREMENT: EXP-008]. An earlier
  expectation that the range truncates the far field was tested and **refuted**.
- **Parameter count.** 423,586 unique weights, 1.62 MiB. Not a constraint.
- **Boundary and thin-structure failures.** **UNKNOWN** — needs Middlebury 2014,
  not yet acquired. Absence of evidence, not evidence of absence.

---

## What should not be changed

- **The colour-guided dilated refinement design.** It recovers a usable disparity
  from a degenerate input, correlating with the final output at r = +0.9998
  [EXP-006]. It works.
- **Static shapes and the conservative operator set.** The model compiles for
  Hailo-8 in full [SR-005]. That is not free and should not be given up casually.
- **fp16.** Accuracy-neutral and 28 % faster [EXP-015].
- **Near-field accuracy.** 0.213 m mean absolute error inside 10 m [EXP-012].

---

*Measurements: EXP-001, EXP-003, EXP-005 to EXP-016. Sources: [SR-001] ONNX,
[SR-002] Model Zoo, [SR-003] compiler script, [SR-004] benchmark table,
[SR-005] HEF, [SR-011] upstream implementation.*
