# Hardware, compiler and deployment analysis

What can be established about deploying this model on edge silicon from public
artifacts and our own experiments — and, just as importantly, what cannot.

**No Hailo device is available.** That is a documented Phase 1 limitation, not a
blocker. It means every statement about Hailo silicon performance is either
`SOURCE` (Hailo published it) or `UNKNOWN` (we cannot establish it). Our RTX 4060
and CPU measurements are **our measurements of our hardware**; they are not
proxies for Hailo silicon and are never compared against Hailo's figures.

---

## 1. The deployment chain

```
PyTorch model  [SR-011]
     |  torch.onnx.export, pytorch 2.0.0, opset 14
     v
ONNX  [SR-001]        168 nodes, static shapes, batch fixed at 1
     |  Hailo parser: normalize_in_net inserts a normalisation layer
     |  quantisation: calibration on the first 160 KITTI 2015 scenes
     |  Dataflow Compiler, directed by stereonet.alls  [SR-003]
     v
HEF  [SR-005]         24,057,165 bytes, Hailo-8
     |  HailoRT
     v
Hailo-8 accelerator   published: 10.7 FPS batch 1, 11.6 batch 8  [SR-004]
```

Every stage before the HEF is public and was inspected. The HEF was acquired but
**not decoded** — its internal format is not publicly documented, and without a
device there is nothing to run it on.

---

## 2. What the compiler script reveals

`stereonet.alls` [SOURCE: SR-003] is the most informative public artifact about
Hailo's hardware constraints, because it records what the compiler had to do to
make this model fit.

### Normalisation is inserted, not exported

```
stereonet/normalization1, stereonet/normalization2 =
    normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
```

The ONNX contains no normalisation [VERIFIED: SR-001]. It becomes the first
on-device layer. Practical consequence: running the ONNX directly requires
normalising on the host first — which our reproduction does, and omitting it
would produce a wrong result with no error.

### The refinement layers are torn apart

```
conv47_ws, conv47_sd0 … conv47_sd21, conv47_sdc =
    defuse(conv47, 22, defuse_type=SPATIAL, splitter=True)
ws_from_conv47_ws_to_conv47_sd0-1 =
    shape_splitter(SPLIT_WIDTH, conv47_ws, [conv47_sd0, conv47_sd1])
…
sh_from_conv46_to_conv48_sd0-3 = shortcut(conv46_sdc, [conv48_sd0, …])
```

Layers `conv42` through `conv52` — the full-resolution refinement block — are
**spatially defused** into as many as 22 pieces, split along width, computed
separately and re-concatenated, with residual connections fanned out to every
piece.

**Why:** every activation in that stage is 55.34 MiB [MEASUREMENT: EXP-001], and
several are live simultaneously through the residual connections. On-chip memory
on this class of accelerator is far smaller than that, so the compiler tiles the
work by image width.

**This is the vendor's own toolchain reporting where the model strains the
hardware**, and it agrees with our independent finding that the same stage
carries 90.6 % of the MACs and 90.8 % of the activation traffic
[MEASUREMENT: EXP-001], and 73.3 % of measured GPU time [MEASUREMENT: EXP-013].

**Status:** the defusion is `SOURCE`. That it is caused by activation size is
`INFERENCE` — a strong one, since the defused layers are exactly the ones with
55 MiB activations and no other layer is defused, but Hailo does not state the
reason.

### What defusion costs

**UNKNOWN.** Splitting a convolution by width requires halo exchange at the
seams, extra memory traffic for the re-concatenation, and more scheduling
overhead. None of that is measurable without the device. It is plausible that
the refinement stage's share of on-device time exceeds its 73.3 % share of our
GPU time, but that is a `HYPOTHESIS`, not a finding.

---

## 3. Operator inventory and portability

**[VERIFIED: SR-001]** — the complete set of operators in the graph:

| Operator | Count | Edge-accelerator risk |
|---|---:|---|
| Conv (2D and 3D) | 53 | Low for 2D. **3D convolution is the notable risk** — many NPUs support it poorly or not at all |
| LeakyRelu | 40 | Low; occasionally fused, occasionally not |
| Add | 19 | Low |
| Concat | 13 | Low, but 13 of them contribute to layout pressure |
| Sub | 12 | Low |
| Unsqueeze | 12 | Layout-only; may be free or may force a copy |
| Slice | 11 | Low |
| Transpose | 1 | **5D transpose**, permutation `[0,2,1,3,4]`. Genuinely awkward — a 5D layout change is exactly what fixed-function tensor engines handle badly |
| Squeeze, Resize, Neg, Softmax, Mul, ReduceSum, Relu | 1 each | Resize (bilinear, align_corners) and Softmax over a non-channel axis at full resolution are both moderate risk |

Notable absences: no `BatchNormalization` (folded at export), no dynamic shapes,
no control flow, no custom operators. That is a well-behaved graph, and it is
part of why it compiles at all.

**The whole model does compile for Hailo-8** — the HEF exists [SR-005] — so none
of these operators is fatal on that target. **Whether any of them falls back to
host execution, and at what cost, is UNKNOWN**: the mapping report would say, and
we do not have it.

### The 5D transpose and the cost volume

Nodes 104–117 build the volume with 12 `Unsqueeze`, a 5D `Concat` and a 5D
`Transpose` [VERIFIED: SR-001]. Zero MACs, 15.3 MiB of traffic
[MEASUREMENT: EXP-001], and 3.6 % of GPU time [MEASUREMENT: EXP-013].

On a device whose cost model is dominated by moving tensors between memories,
this stage is invisible to a FLOP count and potentially expensive. And by
EXP-010, it constructs twelve identical copies of one tensor.

---

## 4. Precision

**[MEASUREMENT: EXP-015]** — 40 scenes, our runtimes, our hardware.

| Configuration | EPE px | D1 % | Depth Abs Rel | Depth RMSE m | δ₁ % | Latency |
|---|---:|---:|---:|---:|---:|---:|
| fp32, onnxruntime CPU | 1.313 | 8.154 | 0.0489 | 2.980 | 96.6 | 549.1 ms |
| fp32, PyTorch CUDA | 1.314 | 8.155 | 0.0489 | 2.980 | 96.6 | 90.1 ms |
| **fp16, PyTorch CUDA** | **1.310** | **8.141** | 0.0486 | 2.978 | 96.6 | **64.8 ms** |
| **int8, onnxruntime CPU** | **1.655** | **10.945** | 0.0627 | 3.310 | 95.8 | 677.4 ms |

Model size: fp32 ONNX 23,690,586 bytes → int8 6,105,895 bytes (3.9× smaller).

### fp16 is free

EPE moves by −0.0037 px and D1 by −0.014 points — both *slightly better*, which
is noise, not an improvement. Maximum per-pixel difference 1.12 px. Latency drops
28 % (90.1 → 64.8 ms).

**fp16 costs nothing measurable in accuracy for this model.** That is not
surprising given the activation ranges observed in EXP-006 (the largest
intermediate magnitudes are in the tens), but it is now measured rather than
assumed.

### int8 costs 2.79 D1 points

EPE 1.313 → 1.655 px (+26 %), D1 8.154 → 10.945 % (+2.79 points), depth Abs Rel
0.0489 → 0.0627, δ₁ 96.6 → 95.8 %. **Maximum per-pixel difference 45.1 px** — so
the damage is not uniform; some pixels move enormously.

The int8 latency is *worse* (677 vs 549 ms). That is a property of onnxruntime's
QDQ representation on this CPU, where quantise/dequantise pairs are inserted and
not all fuse into integer kernels. **It says nothing about int8 on hardware with
native integer support**, where the whole point is that it is faster.

### What this does and does not say about Hailo

Hailo publishes 8.223 float and 10.3 hardware on its metric [SOURCE: SR-002,
SR-004] — a degradation of about 2.08 points. Our fp32→int8 degradation is 2.79
points.

**These are of similar magnitude, and that is as far as the observation may be
taken.** Different quantiser, different calibration procedure, different integer
arithmetic, different aggregation of the metric. The similarity is an
observation, **not** evidence that our stack reproduces Hailo's quantisation, and
it is not used to infer anything.

**Status: `SOURCE` for Hailo's figures, `MEASUREMENT` for ours, `UNKNOWN` for any
relationship between them.**

---

## 4a. Hailo's own compiled profiler report

**[SOURCE: SR-006, decoded in EXP-017]** — `python
scripts/extract_profiler_report.py` then `python scripts/exp_profiler_report.py`.

The 40.9 MB report is a single-page application with a semicolon-separated
payload embedded in it: a 53-field model summary and a **242-row per-layer
table**. Extracted cleanly, zero malformed rows.

**Read the caveat first.** `profiling_mode` is `post_placement`, and the
model-level `fps` and `latency` fields are both `N/A`. This is the compiler's
static post-placement estimate. **It is not a measured run on hardware**, and
nothing in it is a silicon measurement.

### Model-level summary

| Field | Value |
|---|---|
| `weights` | **623,138** |
| `macs_per_image` | 56,258,882,372 |
| `ops_per_image` | 112,111,950,720 |
| `layers` (compiled) | 242 |
| `number_of_contexts` | **6** |
| `number_of_devices` | 4 |
| `hw_arch` | hailo8 |
| `normalization` | True |
| `optimization_level` / `compression_level` | 1 / 0 |
| `l2_data_usage` | 10,514,432 bytes (10.03 MiB) |
| `l2_weights` | 101,376 bytes |
| `net_input_throughput` | 27.98 MB/s |
| `gross_input_throughput` | 1.09 GB/s |
| `net_output_throughput` | 4.66 MB/s |
| `gross_output_throughput` | 786.99 MB/s |
| `fps`, `latency` | **N/A** |

### Three of our findings, confirmed by Hailo's own toolchain

1. **`weights = 623,138` is exactly our per-occurrence parameter count.** We
   derived 423,586 unique learned tensors and observed that adding the shared
   Siamese feature extractor a second time gives 623,138, matching the published
   "623.1K" [MEASUREMENT: EXP-001]. Hailo's compiler reports precisely that
   number. The counting convention is now confirmed at source, not inferred.
2. **`ops_per_image / macs_per_image` = 1.9928 ≈ 2.** The published operation
   count is twice the MAC count, as inferred in EXP-001. Our independent MAC
   total of 56,039,313,792 differs from Hailo's 56,258,882,372 by **−0.39 %**.
3. **Refinement carries 90.6 % of the compiled model's MACs** — 50.98 G of
   56.26 G — against our 90.6 % on the ONNX [EXP-018]. Two independent routes to
   the same split, agreeing to within 0.02 percentage points.

### Per-stage, from Hailo's compiled graph

**[SOURCE: SR-006, EXP-018]** — EXP-017's version of this table used an
incorrect stage mapping and is withdrawn; see
`experiments/EXP-017/CORRECTION.md`.

| Stage | Compiled layers | MACs | Share | Min modelled FPS | Mean effective MAC utilisation |
|---|---:|---:|---:|---:|---:|
| **Refinement** | 14 | 50,981,677,824 | **90.6 %** | **43.0** | 0.491 |
| Aggregation (3D) | 5 | 2,371,404,420 | 4.2 % | 286.0 | 0.691 |
| Feature extraction (left) | 28 | 1,441,849,024 | 2.6 % | 100.6 | 0.086 |
| Feature extraction (right) | 17 | 1,441,849,024 | 2.6 % | 100.6 | 0.141 |
| Regression / elementwise | 22 | 19,381,824 | 0.0 % | 110.2 | 0.064 |
| Normalisation | 2 | 2,720,256 | 0.0 % | 699.4 | 0.021 |
| Data movement | 128 | 0 | 0.0 % | 178.6 | 0.266 |
| I/O and constants | 12 | 0 | 0.0 % | 252.0 | 0.000 |
| Other | 14 | 0 | 0.0 % | 1123.0 | 0.000 |

The compiled shares match our ONNX analysis (90.6 / 4.2 / 5.1) to within 0.02
percentage points, which is what makes the mapping credible rather than merely
plausible.

### The modelled throughput bottleneck is refinement

The report gives a per-layer FPS, and the lowest defines the context's
throughput. **The eight slowest layers in the entire model are all refinement
convolutions:**

| Layer | Modelled FPS | MACs | Context | Effective MAC utilisation |
|---|---:|---:|---|---:|
| `conv50` | **43.03** | 4,192,821,248 | context_4 | 0.828 |
| `conv52` | 53.45 | 4,192,821,248 | context_4 | 0.635 |
| `conv41` | 72.80 | 4,192,821,248 | context_2 | 0.749 |
| `conv49` | 72.80 | 4,192,821,248 | context_4 | 0.443 |
| `conv51` | 72.80 | 4,192,821,248 | context_4 | 0.443 |
| `conv40` | 75.41 | 536,797,184 | context_2 | 0.382 |
| `conv44` | 76.09 | 4,192,821,248 | context_2 | 0.719 |
| `conv42` | 83.89 | 4,192,821,248 | context_2 | 0.701 |

The first non-refinement layer appears in ninth place at 100.63 FPS.

**This closes the B2 question from the vendor's side.** Our static MAC analysis
said refinement dominates, our GPU and CPU profiling measured it dominating, and
Hailo's own compiler model independently identifies refinement convolutions as
the throughput bottleneck. Three independent lines of evidence, one conclusion.

### The model does not fit in one context

`number_of_contexts: 6`, with layers distributed 77 / 70 / 27 / 28 / 32 / 8
across contexts 0 to 5. A context switch reloads network configuration on the
device, and the report documents dedicated cost categories for it
(`RUNTIME_CONFIG`, `RUNTIME_OVERHEAD`, `context_switch_configs = 5,544,448`).

**INFERENCE, labelled:** the modelled bottleneck of 43.03 FPS is a per-layer
throughput figure, while Hailo publishes 10.7 FPS measured at batch 1
[SOURCE: SR-004] — roughly a factor of four lower. Six contexts and their switch
overhead is a plausible explanation for a gap of that size, and it is consistent
with batch 8 being *faster* (11.6 FPS) since batching amortises configuration
loading. **This is a hypothesis, not a finding**: the report supplies no measured
context-switch cost, and no device is available to measure one.

### Quantisation is uniform int8

All 242 layers report `weights_bits`, `input_activation_bits` and
`output_activation_bits` as **8/8/8**. `total_4bit_macs_per_frame` is 0. There is
no mixed precision and no 4-bit enhancement anywhere in the compiled model, and
`optimization_level` is 1 with `compression_level` 0.

That means Hailo's published float-to-hardware degradation of about 2.08 points
[SR-002, SR-004] is the cost of uniform int8 — the same nominal precision as the
int8 row in §4, though still a different quantiser on different arithmetic, so
the two still may not be compared directly.

### Defusion, confirmed and quantified

182 of the 242 rows carry a `defuse_name`. The width-splitting groups
(`ws_from_conv47_ws_to_conv47_sd*`, `ws_from_conv48_ws_to_conv48_sd*` and so on)
appear exactly where `stereonet.alls` directs them — around `conv42`–`conv53`,
the full-resolution refinement block — along with the `mux`/`demux` and
`shortcut` layers that carry residual connections across the split pieces and
across context boundaries.

The report still does **not** state what defusion costs in time. That remains
UNKNOWN.

---

## 5. Our latency measurements

**[MEASUREMENT: EXP-013, EXP-014]** — full detail in `compute_profile.md`.

| Device | Total | FPS | Peak memory |
|---|---:|---:|---|
| RTX 4060 Laptop, fp32, batch 1 | 44.4 ms | 22.5 | 302.1 MiB allocated |
| Ryzen AI 9 HX370 CPU, fp32, batch 1 | 642.3 ms | 1.6 | not measured |

**For reference only, never compared:** Hailo publishes 10.7 FPS at batch 1 on
Hailo-8 [SOURCE: SR-004]. That is a different device, a different precision, a
different runtime and a different measurement methodology. Placing it beside our
numbers in the same column would be exactly the error this project's rules exist
to prevent.

**The most important deployment finding is not a latency number.** It is that the
stage ranking changed substantially between two devices we *do* control: 3D
aggregation is 2.8 % of GPU time and 17.2 % of CPU time [MEASUREMENT: EXP-013,
EXP-014]. If two general-purpose processors disagree by 6× on which stage
matters, extrapolating either to a fixed-function accelerator is unjustified.

---

## 6. Constraints, and their status

| Constraint | Evidence | Status |
|---|---|---|
| Full-resolution refinement activations are 55.34 MiB, several live at once | EXP-001 | MEASUREMENT |
| The compiler defuses those layers into up to 22 spatial pieces | SR-003 | SOURCE |
| Defusion is caused by activation size exceeding on-chip memory | SR-003 + EXP-001 | INFERENCE (strong) |
| The graph uses only well-supported operators plus one 5D transpose | SR-001 | VERIFIED |
| The model compiles for Hailo-8 in full | SR-005 | SOURCE |
| Normalisation is compiler-inserted, not in the ONNX | SR-002, SR-003 | VERIFIED |
| fp16 is accuracy-neutral for this model | EXP-015 | MEASUREMENT |
| int8 on our stack costs 2.79 D1 points | EXP-015 | MEASUREMENT |
| MAC count does not predict latency | EXP-013, EXP-014 | MEASUREMENT |
| Stage ranking is device-dependent | EXP-013, EXP-014 | MEASUREMENT |
| Batch is fixed at 1 in the exported graph | SR-001 | VERIFIED |
| Hailo-8 achieves 10.7 FPS at batch 1 | SR-004 | SOURCE |
| Which operators fall back to host on Hailo, if any | — | **UNKNOWN** |
| What spatial defusion costs in latency | — | **UNKNOWN** |
| Per-layer *modelled* throughput on Hailo | SR-006, EXP-017 | **SOURCE** — refinement convolutions are the bottleneck at 43.03 FPS modelled; the eight slowest layers are all refinement |
| Per-layer *measured* latency on Hailo silicon | — | **UNKNOWN** — the report is a post-placement estimate, fps and latency N/A |
| The compiled model spans 6 device contexts | SR-006, EXP-017 | **SOURCE** |
| Quantisation is uniform 8/8/8 across all 242 layers | SR-006, EXP-017 | **SOURCE** |
| Whether context switching explains the gap between 43 FPS modelled and 10.7 FPS published | — | **HYPOTHESIS** |
| Power, FPS/W on any device | — | **UNKNOWN** |
| Host-device transfer overhead | — | **UNKNOWN** |
| Why the config lists hailo15h/hailo10h while the benchmark is Hailo-8 | — | **UNKNOWN** |

---

## 7. Not done

- **The profiler report [SR-006] has now been decoded** — see §4a. What it does
  not contain is measured silicon behaviour: it is a post-placement compiler
  estimate.
- The HEF has not been decoded.
- No Hailo device, therefore no on-device measurement of anything.
- The Ryzen AI NPU (XDNA2) was not targeted. It would be a separate runtime
  target, not a Hailo proxy.
- TensorRT was not evaluated.

---

*Sources: [SR-001] ONNX, [SR-002] Model Zoo configuration, [SR-003] compiler
script, [SR-004] Hailo-8 benchmark table, [SR-005] HEF, [SR-006] profiler report
(unread). Measurements: [EXP-001], [EXP-013], [EXP-014], [EXP-015].*
