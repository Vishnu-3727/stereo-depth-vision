# Compute and memory profile

Where the time and the memory actually go, measured rather than inferred from
operation counts.

**These are our measurements on our hardware.** They are not Hailo numbers, not
proxies for Hailo numbers, and are never compared against Hailo's published
latency or FPS. Hailo's figures — 10.7 FPS at batch 1, 11.6 at batch 8 on
Hailo-8 [SOURCE: SR-004] — remain SOURCE evidence and stay in their own column.

Reproduce with `python scripts/exp_profile.py --device cuda|cpu`.

---

## 1. The assumption under test

EXP-001 found the full-resolution refinement stage carries **90.6 %** of the
model's MACs, and Hailo's compiler spatially defuses exactly that region into up
to 22 pieces [SOURCE: SR-003]. That makes it the obvious runtime bottleneck.

Obvious is not measured. The charter is explicit that *fewer FLOPs = faster* is
an empirical question, so every stage's measured time share is reported beside
its static MAC share.

**Result: MAC share is a poor predictor, and wrong in both directions.**

---

## 2. RTX 4060 Laptop, fp32, batch 1

**[MEASUREMENT: EXP-013]** — 368×1232, median of 30 runs after 5 warm-up
iterations, CUDA synchronised before every timestamp.

| Stage | Median | Time share | MAC share | Time share ÷ MAC share | σ |
|---|---:|---:|---:|---:|---:|
| Feature extraction (both branches) | 8.166 ms | 18.4 % | 5.1 % | **3.59×** | 3.954 ms |
| Cost volume construction | 1.608 ms | 3.6 % | **0.0 %** | — (undefined) | 0.707 ms |
| Cost volume aggregation (3D) | 1.260 ms | 2.8 % | 4.2 % | 0.67× | 0.474 ms |
| Upsample + soft-argmin | 0.633 ms | 1.4 % | 0.01 % | **142×** | 0.058 ms |
| **Refinement** | **32.565 ms** | **73.3 %** | **90.6 %** | 0.81× | 0.069 ms |
| Residual add + ReLU | 0.360 ms | 0.8 % | 0.0 % | — | 0.104 ms |
| **Total** | **44.412 ms** | | | | 4.968 ms |

**22.5 FPS at batch 1.**

Peak GPU memory: **302.1 MiB allocated, 404.0 MiB reserved** on an 8 GiB card.
Memory is not a constraint on this hardware; it is on an edge accelerator, which
is a different question addressed in `hardware_analysis.md`.

The high σ on feature extraction (3.954 ms against a 8.166 ms median) is worth
noting: it is the first stage after the input, so it absorbs clock and scheduling
variation that later stages do not. Refinement, by contrast, has σ = 0.069 ms —
extremely stable, consistent with a stage that saturates the device.

## 3. Ryzen AI 9 HX370 CPU, fp32, batch 1

**[MEASUREMENT: EXP-014]** — same model and input, median of 8 runs after 2
warm-up iterations.

| Stage | Median | Time share | MAC share | Time share ÷ MAC share |
|---|---:|---:|---:|---:|
| Feature extraction | 27.943 ms | 4.4 % | 5.1 % | 0.85× |
| Cost volume construction | 1.317 ms | 0.2 % | 0.0 % | — |
| **Cost volume aggregation (3D)** | **110.251 ms** | **17.2 %** | 4.2 % | **4.06×** |
| Upsample + soft-argmin | 14.495 ms | 2.3 % | 0.01 % | **226×** |
| Refinement | 486.997 ms | 75.8 % | 90.6 % | 0.84× |
| Residual add + ReLU | 0.233 ms | 0.0 % | — | — |
| **Total** | **642.264 ms** | | | |

**1.6 FPS at batch 1** — 14.5× slower than the GPU.

---

## 4. What the measurements show

### Refinement dominates, but by less than its MAC share

73.3 % of GPU time against 90.6 % of MACs. It is unambiguously the largest
stage, and the `.alls` defusion [SOURCE: SR-003] shows Hailo's compiler
struggling with the same region — but a MAC count *overstates* it by about 20 %.
Straightforward dense 3×3 convolutions at 32 channels are what GPUs are best at,
so refinement runs closer to peak efficiency than anything else in the model.

**F1 prediction status: confirmed as the dominant stage, with the caveat that
the MAC share overstates the degree.**

### Feature extraction is 3.6× over-represented in time

5.1 % of MACs, 18.4 % of GPU time. It runs at the highest resolutions in the
model — 184×616 and 92×308 — where activation traffic is large relative to
arithmetic, and it runs **twice**. The 5×5 kernels are also less efficient per
MAC than 3×3.

This is the clearest case in the model of arithmetic intensity, not arithmetic
volume, setting the cost.

### The cost volume costs 3.6 % of the time while performing zero MACs

1.608 ms on the GPU for 48 nodes of pure data movement — concatenation, slicing,
unsqueezing, stacking, transposition [MEASUREMENT: EXP-001, EXP-013]. **A FLOP
count cannot see this stage at all.**

It is a small number here only because the volume is small. In an architecture
matching at 1/4 resolution with a full disparity range, the same construction
scales by 64× (see `cost_volume_analysis.md`) and would become a first-order
cost that no FLOP-based model would predict.

And by F1, this 1.608 ms builds twelve identical copies of one tensor.

### Soft-argmin is 142× over-represented on GPU, 226× on CPU

0.633 ms for 0.01 % of the MACs. Its work is a 16× bilinear upsample of a
`12×23×77` tensor to `12×368×1232` — 20.8 MiB — followed by a softmax, a
multiply against a 21.76 MiB constant, and a reduction, all at full resolution
[VERIFIED: SR-001]. Pure memory traffic, essentially no arithmetic.

Doing the reduction at 1/16 resolution and upsampling a single-channel result
would move the same operations onto 1/192 as much data. That observation is
recorded for Phase 2; nothing is changed.

### The device changes the ranking

| Stage | GPU share | CPU share | Change |
|---|---:|---:|---|
| Feature extraction | 18.4 % | 4.4 % | ↓ 4.2× |
| Aggregation (3D) | 2.8 % | 17.2 % | **↑ 6.1×** |
| Refinement | 73.3 % | 75.8 % | ≈ |

**3D convolution is 6× more expensive relative to everything else on CPU than on
GPU.** A stage that looks negligible on one device is the second-largest on
another, from the same graph and the same weights.

That is the strongest single argument in this document against reasoning about
edge deployment from operation counts. Hailo silicon is neither of these devices,
and its ranking cannot be predicted from either — which is precisely why
`hardware_analysis.md` marks the corresponding entries UNKNOWN rather than
extrapolating.

---

## 5. Static memory figures

**[MEASUREMENT: EXP-001]** — derived from ONNX shapes, batch 1, fp32.

| Quantity | Value |
|---|---:|
| Peak single activation | **55.34 MiB** (every refinement tensor) |
| Total activation traffic | 1,898.9 MiB |
| Refinement share of traffic | 90.8 % |
| Cost volume | 2.59 MiB |
| Upsampled cost tensor | 20.8 MiB |
| Learned weights (fp32) | 1.62 MiB |
| Non-learned constants | **20.94 MiB** (the index grid) |
| Measured peak GPU allocation | 302.1 MiB |

The gap between 55.34 MiB of peak single activation and 302.1 MiB measured is
allocator behaviour — several full-resolution tensors are live at once through
the residual connections, plus workspace.

**The relevant number for an edge accelerator is 55.34 MiB per tensor, not
302 MiB total.** On-chip SRAM on this class of device is measured in single-digit
to low-double-digit megabytes, and that is what forces the compiler to split the
refinement layers by width [SOURCE: SR-003] — it cannot hold one of these
tensors, let alone the several the residual connections require simultaneously.

---

## 6. Conclusions

1. **Refinement is the runtime bottleneck** on both devices, 73–76 % of the
   time. [MEASUREMENT: EXP-013, EXP-014]
2. **MAC share does not predict time share.** Errors range from 0.67× to 226×,
   in both directions, and one stage with zero MACs takes measurable time.
   [MEASUREMENT: EXP-013]
3. **The ranking is device-dependent.** 3D aggregation moves from 2.8 % to 17.2 %
   between GPU and CPU. [MEASUREMENT: EXP-013, EXP-014]
4. **Memory pressure is about peak single activation**, 55.34 MiB, not total
   footprint. [MEASUREMENT: EXP-001]
5. **Nothing here transfers to Hailo silicon.** [UNKNOWN]

---

## 7. Not measured

| Gap | Why |
|---|---|
| Power and FPS/W | No instrumentation available on this laptop |
| GPU utilisation and occupancy | Needs Nsight Compute; wall-clock per stage was sufficient for the question asked |
| Host-device transfer overhead | Inputs were resident on device; a camera pipeline would add transfer cost |
| Preprocessing and postprocessing latency | Measured only inside the network; end-to-end pipeline timing is outstanding |
| Batch > 1 | The reference ONNX has batch fixed at 1 [VERIFIED: SR-001] |
| Any Hailo device measurement | No device available. **UNKNOWN, and a documented Phase 1 limitation.** |

---

*Measurements: [EXP-001] static analysis, [EXP-013] RTX 4060, [EXP-014] CPU.
Sources: [SR-003] compiler script, [SR-004] Hailo-8 benchmark table.*
