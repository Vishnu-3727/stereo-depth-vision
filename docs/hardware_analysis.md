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
| Per-layer latency on Hailo silicon | — | **UNKNOWN** (the profiler report [SR-006] may contain it; not yet read) |
| Power, FPS/W on any device | — | **UNKNOWN** |
| Host-device transfer overhead | — | **UNKNOWN** |
| Why the config lists hailo15h/hailo10h while the benchmark is Hailo-8 | — | **UNKNOWN** |

---

## 7. Not done

- **`stereonet_profiler_results_compiled_runtime_data.html` [SR-006] has not been
  read.** It is 40.9 MB of Hailo's own compiled-runtime profile and is the single
  most likely public source of per-layer on-device behaviour. Acquired and
  hashed; reading it is outstanding and is the highest-value remaining item in
  this document.
- The HEF has not been decoded.
- No Hailo device, therefore no on-device measurement of anything.
- The Ryzen AI NPU (XDNA2) was not targeted. It would be a separate runtime
  target, not a Hailo proxy.
- TensorRT was not evaluated.

---

*Sources: [SR-001] ONNX, [SR-002] Model Zoo configuration, [SR-003] compiler
script, [SR-004] Hailo-8 benchmark table, [SR-005] HEF, [SR-006] profiler report
(unread). Measurements: [EXP-001], [EXP-013], [EXP-014], [EXP-015].*
