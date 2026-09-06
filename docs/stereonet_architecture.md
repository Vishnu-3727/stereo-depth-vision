# StereoNet architecture, as deployed by Hailo

Reconstructed layer by layer from the exported ONNX [SR-001], cross-read against
the upstream PyTorch source [SR-011], Hailo's Model Zoo configuration [SR-002]
and its compiler script [SR-003].

The reconstruction is essentially complete. The ONNX has fully static shapes on
all 168 nodes, so almost every entry below is `VERIFIED` against the executable
artifact rather than inferred.

Reproduce with:

```
python scripts/inspect_onnx.py          # graph inventory  -> results/onnx_inspection/
python scripts/analyze_architecture.py  # cost model       -> results/architecture/
python scripts/exp_onnx_forensics.py    # EXP-001, the record
```

---

## 0. The one thing to read first

The name "StereoNet" suggests the expensive part is the 3D cost volume. **In
this model it is not.** The cost volume sits at 1/16 resolution with 12
disparity candidates and accounts for 4.2 % of the arithmetic. The
full-resolution refinement network accounts for **90.6 %**.

| Stage | Nodes | Params | MACs | MAC share | Peak act. | Act. traffic |
|---|---:|---:|---:|---:|---:|---:|
| Feature extraction (left) | 35 | 199,552 | 1.44 G | 2.6 % | 13.8 MiB | 25.1 MiB |
| Feature extraction (right) | 35 | shared | 1.44 G | 2.6 % | 13.8 MiB | 25.1 MiB |
| Cost volume construction | 48 | 0 | 0 | 0.0 % | 2.6 MiB | 15.3 MiB |
| Cost volume aggregation (3D) | 10 | 111,585 | 2.37 G | 4.2 % | 2.6 MiB | 20.9 MiB |
| Upsample + soft-argmin | 5 | 0 | 0.005 G | 0.0 % | 20.8 MiB | 84.7 MiB |
| **Refinement (full resolution)** | **33** | **112,449** | **50.79 G** | **90.6 %** | **55.3 MiB** | **1724.3 MiB** |
| Residual add + ReLU | 2 | 0 | 0 | 0.0 % | 1.7 MiB | 3.5 MiB |
| **Total** | **168** | **423,586** | **56.04 G** | | **55.3 MiB** | **1898.9 MiB** |

*[MEASUREMENT: EXP-001], derived from ONNX shapes at batch 1, 368×1232 input.*

This inversion is a consequence of the design: StereoNet's whole premise is to
make matching cheap by doing it at very low resolution and spend the saved
budget on a learned upsampler. The published numbers show what that trade costs
— the upsampler ends up being the model.

---

## 1. Input and output

| | |
|---|---|
| Inputs | two, `input.1` and `input.83`, each `1×3×368×1232` float32 [VERIFIED: SR-001] |
| Output | one, `524`, `1×1×368×1232` float32 — disparity in pixels at full resolution [VERIFIED: SR-001] |
| Layout | NCHW [VERIFIED: SR-001] |
| Batch | fixed at 1 in the graph; no dynamic axes [VERIFIED: SR-001] |
| Producer | pytorch 2.0.0, opset 14, IR version 7 [VERIFIED: SR-001] |
| Normalisation | **not in the ONNX.** Mean `[123.675, 116.28, 103.53]`, std `[58.395, 57.12, 57.375]` are applied by Hailo's parser via `normalize_in_net: true` [SOURCE: SR-002], and `stereonet.alls` inserts them as the first graph layer [SOURCE: SR-003]. Running the ONNX directly therefore requires normalising the input first. |

`368 = 16 × 23` and `1232 = 16 × 77`, so the resolution is chosen to divide
exactly by the 16× downsampling factor. KITTI images are 1242×375, so reaching
this input requires a crop or resize — which one, and the effect on the
evaluation, is a W7 question and currently **UNKNOWN**.

---

## 2. Feature extraction — twice, with shared weights

Runs once per image. Both branches use the **same initializer tensors** — all 17
convolution weight pairs match by name [VERIFIED: SR-001] — so it is a true
Siamese tower, one set of weights executed twice.

**Downsampling: four stride-2 5×5 convolutions, no activation between them.**

| Node | Output | Kernel | Stride | Params | MACs |
|---:|---|---|---|---:|---:|
| 0 | 1×32×184×616 | 5×5 | 2×2 | 2,432 | 272.0 M |
| 1 | 1×32×92×308 | 5×5 | 2×2 | 25,632 | 725.4 M |
| 2 | 1×32×46×154 | 5×5 | 2×2 | 25,632 | 181.4 M |
| 3 | 1×32×23×77 | 5×5 | 2×2 | 25,632 | 45.3 M |

Resolution drops 16× to **23×77**, 32 channels throughout.

Two things stand out. First, the second convolution is the most expensive in the
whole tower (725 M MACs) because it still operates at 1/2 resolution with 32
input channels — the first is cheaper only because its input has 3 channels.
Second, **there is no non-linearity anywhere in this stack** [VERIFIED: SR-001,
confirmed in SR-011 where `nn.Sequential` holds four bare `Conv2d`]. Four
stacked linear convolutions compose to a single linear operator, so the four
layers have no more representational power than one appropriately sized
convolution. They are not equivalent in cost — a single fused kernel would be a
different shape and a different cost — but they add no expressiveness. This
appears to be an oversight in the upstream implementation rather than a design
choice; the paper describes a downsampling stack with activations. **Recorded,
not fixed** — the baseline is frozen.

**Residual stack: six residual blocks then one convolution, all at 23×77.**

Each block is `Conv3×3 → LeakyReLU → Conv3×3 → Add(input) → LeakyReLU`, 32
channels, 16.3 M MACs per convolution. Batch normalisation is present in the
PyTorch source [SR-011] but **absent from the ONNX operator inventory** — zero
`BatchNormalization` nodes [VERIFIED: SR-001] — so it was folded into the
convolution weights and biases at export. This is why the exported convolutions
carry biases while the source declares `bias=False`.

Feature extractor total: **199,552 parameters, 1.436 G MACs per branch.**

---

## 3. Cost volume

Covered in full in `cost_volume_analysis.md`. In brief [VERIFIED: SR-001]:

- **12 disparity candidates** at 1/16 resolution, corresponding to disparities
  0–11 at 23×77, i.e. **0–176 pixels at full resolution** in steps of 16.
- Built by **subtraction**: 12 `Sub` nodes producing `1×32×23×77` each.
- Each shift is implemented as `Concat` with a zero block then `Slice` back to
  width 77 — a zero-padded left shift, 11 times.
- Stacked by 12 `Unsqueeze` + one `Concat` into `1×12×32×23×77`, then
  `Transpose` to `1×32×12×23×77` for 3D convolution.
- **Zero MACs.** The construction is pure data movement, 15.3 MiB of it.

**This contradicts the upstream README**, which states the cost volume uses
concatenation [SOURCE: SR-011]. The source supports both modes; the exported
artifact uses `subtract`. The ONNX is what Hailo compiled and is therefore what
the deployed baseline does. Both values are recorded rather than one being
picked.

## 4. Cost volume aggregation

Four `Conv3d(32→32, 3×3×3)` + LeakyReLU, then `Conv3d(32→1, 3×3×3)`, all at
`12×23×77` [VERIFIED: SR-001].

| Node | Output | Params | MACs |
|---:|---|---:|---:|
| 118, 120, 122, 124 | 1×32×12×23×77 | 27,680 each | 587.6 M each |
| 126 | 1×1×12×23×77 | 865 | 18.4 M |

Total **111,585 parameters, 2.369 G MACs** — 4.2 % of the model. Output is a
`1×12×23×77` cost tensor after `Squeeze`: one scalar cost per disparity
candidate per low-resolution pixel.

The 3D convolutions here are what most stereo architectures spend their budget
on. At this resolution and disparity count they are almost free.

## 5. Upsampling and soft-argmin — the unusual part

| Node | Op | Output | Note |
|---:|---|---|---|
| 128 | `Resize` (bilinear) | 1×12×**368×1232** | the whole cost tensor, upsampled 16× |
| 129 | `Neg` | 1×12×368×1232 | low cost should mean high probability |
| 130 | `Softmax` | 1×12×368×1232 | over the 12 disparity candidates |
| 131 | `Mul` | 1×12×368×1232 | × a baked-in constant index grid |
| 132 | `ReduceSum` | 1×1×368×1232 | expected disparity |

**The soft-argmin is computed at full resolution, not at 1/16.** The cost
tensor is bilinearly upsampled from `12×23×77` to `12×368×1232` — a 20.8 MiB
tensor — and softmax, multiply and reduce all run over it [VERIFIED: SR-001].

The alternative, taking soft-argmin at low resolution and then upsampling the
resulting single-channel disparity map, would do the same work on 1/192 of the
data. Whether the full-resolution version is materially better is not a question
Phase 1 answers; it is noted as a **[HYPOTHESIS]** for W8/W9 and nothing is
changed.

The index grid at node 131 is `/Tile_output_0`, a `1×12×368×1232` float32
constant of **21,762,048 bytes — 92 % of the entire 23.7 MB ONNX file**
[MEASUREMENT: EXP-001]. The learned weights are only 1,694,344 bytes. The
constant is `torch.arange(12)` broadcast to full resolution, materialised by the
exporter because the source builds it with `repeat` at runtime [SR-011]. It is
data, not parameters, and it is why the file size and the parameter count tell
completely different stories.

Note also that the upstream `forward()` computes this upsample-and-soft-argmin
path **twice** — once for `d_initial_l` and once inside `forward_stage3`
[SR-011] — but the exported graph contains only one `Resize` and one `Softmax`.
The exporter eliminated the duplicate. A less thorough export path would have
shipped it twice.

## 6. Refinement — 90.6 % of the model

Input is the full-resolution disparity concatenated with the left RGB image:
`1×1×368×1232` ⊕ `1×3×368×1232` → `1×4×368×1232` [VERIFIED: SR-001]. This is the
"guided" part of *guided hierarchical refinement*: the colour image tells the
network where the edges are.

| Node | Output | Dilation | Params | MACs |
|---:|---|---:|---:|---:|
| 134 | 1×32×368×1232 | 1 | 1,184 | 522.3 M |
| 135, 137 | 1×32×368×1232 | 1 | 9,248 each | 4.178 G each |
| 140, 142 | 1×32×368×1232 | **2** | 9,248 each | 4.178 G each |
| 145, 147 | 1×32×368×1232 | **4** | 9,248 each | 4.178 G each |
| 150, 152 | 1×32×368×1232 | **8** | 9,248 each | 4.178 G each |
| 155, 157 | 1×32×368×1232 | 1 | 9,248 each | 4.178 G each |
| 160, 162 | 1×32×368×1232 | 1 | 9,248 each | 4.178 G each |
| 165 | 1×1×368×1232 | 1 | 289 | 130.6 M |

Six residual blocks with dilations 1, 2, 4, 8, 1, 1, giving a receptive field
wide enough to propagate disparity across textureless regions at full
resolution. Then `Add` with the initial disparity (node 166) and `ReLU`
(node 167) to clamp negatives.

**112,449 parameters — 27 % of the model's weights — generating 50.79 G MACs,
90.6 % of its arithmetic.** Twelve convolutions each cost 4.178 G MACs, more
than the entire cost volume aggregation stage put together. Every activation in
this stage is 55.34 MiB.

This stage is exactly the region `stereonet.alls` performs surgery on: `conv42`
through `conv52` are spatially defused into as many as 22 pieces with width
splitting and re-concatenation [SOURCE: SR-003]. Hailo's compiler could not fit
these layers whole. That is independent evidence, from the vendor's own
toolchain, that the refinement stage is where this model strains the hardware.

**Status: the static cost is [MEASUREMENT: EXP-001]. That it is the *runtime*
bottleneck remains [HYPOTHESIS] until W8 profiling.** Static MAC share does not
determine latency; testing that assumption is an explicit Phase 1 objective.

---

## 7. Full layer table

All 168 nodes with shapes, kernels, strides, dilations, parameters, MACs and
activation sizes: **`results/architecture/layer_table.md`** (generated by
`python scripts/make_layer_table.py`, so it cannot drift from the analysis).
Machine-readable form in `results/architecture/layers.json`.

---

## 8. Parameter and operation accounting

**[MEASUREMENT: EXP-001]**

| Quantity | Value |
|---|---:|
| Unique learned tensors | **423,586** |
| Counted once per graph occurrence | **623,138** |
| Hailo published [SOURCE: SR-002] | 623,100 ("623.1K") |
| Learned weight bytes (fp32) | 1,694,344 |
| Non-learned constant bytes | 21,956,416 |
| Our MAC count | 56,039,313,792 (56.04 G) |
| 2 × MACs | 112,078,627,584 (112.08 G) |
| Hailo published operations [SOURCE: SR-002] | 112,200,000,000 ("112.2G") |

Both published figures are reproduced, and the conventions behind them are now
known:

- **Parameters.** Hailo's 623.1K counts the Siamese feature extractor **twice**,
  once per graph occurrence, rather than once per unique tensor. `423,586 +
  199,552 = 623,138`, matching to 38 parameters (rounding to 623.1K). The model
  ships 423,586 distinct weights.
- **Operations.** Hailo's figure is **2 × MACs**. Our 56.04 G doubles to
  112.08 G against the published 112.2 G, a −0.11 % gap, plausibly from a
  slightly different treatment of the elementwise operations (0.34 G of them, of
  the right order to close the gap).

Neither figure is wrong; they simply use conventions that were not stated. Any
comparison this project makes against another architecture must state which
convention it uses.

---

## 9. Where this differs from the original paper

The paper [SR-010] has **not yet been read in full** — only its abstract and
architecture summary — so this section is deliberately incomplete and is
finished in W4/W9. Recording what is already visible:

| Aspect | Deployed model | Notes |
|---|---|---|
| Cost volume metric | subtraction [VERIFIED: SR-001] | The paper describes a difference-based volume, so the ONNX agrees with the paper and the upstream README does not. |
| Downsampling activations | none [VERIFIED: SR-001] | The paper describes a downsampling stack with non-linearities. Likely an upstream implementation deviation. **[INFERENCE]** pending a full read of SR-010. |
| Refinement cascade | single refinement at full resolution [VERIFIED: SR-001] | The paper's title says *hierarchical* refinement, implying a cascade over several scales. The deployed model refines once, directly at full resolution. Whether the paper's cascade is multi-stage and how it compares in cost is **UNKNOWN** pending SR-010. |
| Disparity candidates | 12 [VERIFIED: SR-001] | Follows from `D=192, k=4` in the upstream default. Whether the paper uses the same is **UNKNOWN**. |
| Accuracy | float EPE 8.223 on KITTI 2015 [SOURCE: SR-002] | The upstream author states their implementation falls short of the paper's accuracy [SOURCE: SR-011]. The size of the gap and its cause are the central W7 question. |

The rule for this project: **"Hailo StereoNet" and "StereoNet (ECCV 2018)" are
different systems** and are never used interchangeably. The baseline under study
is the former.

---

## 10. Open questions carried forward

| # | Question | Status | Resolved in |
|---|---|---|---|
| A1 | Does 368×1232 come from cropping or resizing KITTI's 1242×375, and does that choice affect the published EPE? | UNKNOWN | W3 / W7 |
| A2 | Does the shipped Hailo application convert disparity to metric depth at all, or does it output disparity only? | UNKNOWN | W3 |
| A3 | What does the paper actually specify for the refinement cascade and the downsampling activations? | UNKNOWN | W4 (full read of SR-010) |
| A4 | Is the full-resolution refinement stage the *runtime* bottleneck, or only the static-MAC bottleneck? | HYPOTHESIS | W8 |
| A5 | Would soft-argmin at low resolution followed by disparity upsampling change accuracy? | HYPOTHESIS — **not to be tested by modifying the baseline in Phase 1** | Phase 2 |
| A6 | What accounts for the −0.11 % gap between our 112.08 G and Hailo's 112.2 G? | INFERENCE (elementwise op treatment) | W4 |

---

*All shape, parameter and MAC values: [MEASUREMENT: EXP-001] from [SR-001].
Published comparison figures: [SOURCE: SR-002], [SOURCE: SR-004]. Module
structure cross-read from [SR-011]. Compiler behaviour: [SOURCE: SR-003].*
