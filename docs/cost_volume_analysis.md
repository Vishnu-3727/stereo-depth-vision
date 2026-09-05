# Cost volume analysis

What the cost volume is, what this particular one costs, and why it is not where
the money goes in this model.

Everything numeric is [MEASUREMENT: EXP-001], derived from the exported ONNX
[SR-001] and reproducible with `python scripts/exp_onnx_forensics.py`.

---

## 1. What a cost volume is, and why one exists

After rectification, the match for a left-image pixel lies somewhere along the
same row in the right image. The matching problem is: for each pixel, which of
the candidate shifts is right?

A cost volume answers that by not choosing yet. For every pixel and every
candidate disparity `d` in the search range, it stores a number saying how well
the left feature at `(u, v)` agrees with the right feature at `(u - d, v)`. The
result is a 3D structure indexed by `(disparity, height, width)` — or 4D if each
entry keeps a feature vector rather than a scalar.

Deferring the decision is the point. A per-pixel argmin over raw matching costs
is fragile: textureless regions have no distinguishing cost, repetitive textures
have several equally good minima. Keeping the whole volume lets a subsequent
network aggregate evidence across neighbouring pixels *and* across neighbouring
disparities before committing, which is what makes a blank wall resolvable at
all — the answer is propagated in from the edges where there was texture.

The price is that the volume is one dimension larger than the image, so both
memory and compute scale with the number of disparity candidates. Nearly every
design decision in efficient stereo is about paying that price somewhere
cheaper.

---

## 2. How this one is built

**[VERIFIED: SR-001]** — nodes 70–117 of the ONNX.

Inputs are the two 32-channel feature maps at 1/16 resolution, `1×32×23×77`
each.

For each disparity `k` in `0 … 11`:

1. Shift the **left** features left by `k` columns. In the graph this is a
   `Concat` of the left features with a `1×32×23×k` zero block, widening the
   tensor to `77 + k`, followed by a `Slice` back to the first 77 columns. The
   zero block is a stored constant, one per disparity — eleven of them, sizes
   `1×32×23×1` through `1×32×23×11`.
2. Subtract the right features: `Sub`, giving `1×32×23×77`.

Then 12 `Unsqueeze` operations add a disparity axis and one `Concat` stacks them
into `1×12×32×23×77`, and a `Transpose` reorders to `1×32×12×23×77` so that
`Conv3d` sees channels first and disparity as the depth axis.

```
left  1x32x23x77 ──shift by k──┐
                               ├── Sub ──> 1x32x23x77   (k = 0 .. 11)
right 1x32x23x77 ──────────────┘
                                    │
                    12x Unsqueeze + Concat
                                    │
                            1x12x32x23x77
                                    │
                               Transpose
                                    │
                            1x32x12x23x77   <- the cost volume
```

### The subtraction / concatenation contradiction

The upstream implementation supports two modes. `subtract` produces a
32-channel volume; `concat` stacks both feature vectors for 64 channels
[SOURCE: SR-011]. Its README states concatenation is used.

**The exported ONNX uses subtraction**: 12 `Sub` nodes, 32 channels into the
first 3D convolution, whose weight tensor is `[32, 32, 3, 3, 3]` — a 32-channel
input, not 64 [VERIFIED: SR-001].

Both statements are preserved. The ONNX is the artifact Hailo compiled, so
subtraction is what the deployed baseline does. Had concatenation been used, the
volume would be `1×64×12×23×77` and the first 3D convolution would cost twice as
much — a difference of about 0.59 G MACs, roughly 1 % of the model. The choice
matters less here than it would in an architecture with a larger cost volume.

---

## 3. Dimensions

**[VERIFIED: SR-001]** — none of these are assumed; all are read from the graph.

| | |
|---|---|
| Input resolution | 368 × 1232 |
| Feature resolution | **23 × 77** (1/16, from four stride-2 convolutions) |
| Feature channels | 32 |
| Disparity candidates | **12** |
| Volume shape (post-transpose) | `1 × 32 × 12 × 23 × 77` |
| Elements | 680,064 |
| fp32 size | **2.59 MiB** |
| int8 size | 0.65 MiB |

The disparity count follows from the upstream default `D = 192, k = 4`:
`192 // 2⁴ = 12` [SOURCE: SR-011], consistent with the graph.

### What 12 candidates mean in the real world

At 1/16 resolution, one step of low-resolution disparity is **16 pixels** of
full-resolution disparity. So the search covers full-resolution disparities
**0 to 176 in steps of 16**.

Using the measured KITTI calibration `fB = 384.38 px·m` [MEASUREMENT: EXP-003]:

| Candidate `k` | Full-res disparity | Depth |
|---:|---:|---:|
| 0 | 0 px | undefined (infinity) |
| 1 | 16 px | 24.0 m |
| 2 | 32 px | 12.0 m |
| 3 | 48 px | 8.0 m |
| 4 | 64 px | 6.0 m |
| 6 | 96 px | 4.0 m |
| 8 | 128 px | 3.0 m |
| 11 | 176 px | 2.2 m |

Two consequences:

- **The minimum measurable range is about 2.2 m.** Anything closer has a
  disparity beyond the search and cannot be matched.
- **The candidate grid is brutally coarse where depth resolution matters most.**
  Between candidate 0 and candidate 1 lies the entire range from 24 m to
  infinity. Between 1 and 2 lies 12 m to 24 m. Everything beyond 24 m — most of
  a driving scene — is represented by the interval between two adjacent
  candidates.

This is the structural reason the model's far-range behaviour depends entirely
on interpolation rather than matching, and it is a strong prior on where the
error will be found in W7. Recorded as an expectation to be tested, not as a
result: **[HYPOTHESIS] error concentrates at long range because the candidate
grid does not resolve it**, to be confirmed or refuted by range-binned
measurement.

---

## 4. Memory

The general formula is

```
cost_volume_bytes = B × C × D × H × W × bytes_per_element
```

For this model at batch 1, fp32:

```
1 × 32 × 12 × 23 × 77 × 4 = 2,720,256 bytes = 2.59 MiB
```

Held alongside it during construction are the twelve `Sub` outputs
(`1×32×23×77` = 0.22 MiB each) and the eleven zero-padding constants
(0.30 MiB in total).

**Peak activation in the cost volume stages: 2.59 MiB.** For comparison, peak
activation in the refinement stage is **55.34 MiB**, 21× larger
[MEASUREMENT: EXP-001].

### How that scales

The volume grows linearly in every dimension, so the design choices compound:

| Configuration | Volume size (fp32) | Ratio |
|---|---:|---:|
| As deployed — 1/16 res, 12 disparities, 32 ch | 2.59 MiB | 1× |
| 1/8 resolution, same disparity count | 10.4 MiB | 4× |
| 1/8 resolution, 24 disparities | 20.8 MiB | 8× |
| 1/4 resolution, 48 disparities | 166 MiB | 64× |
| Concatenation instead of subtraction (64 ch) | 5.19 MiB | 2× |

The last row shows what the `concat`/`subtract` question would have cost. The
others show why architectures that match at 1/4 resolution with a full disparity
range need a fundamentally different memory strategy — and why StereoNet's
choice to match at 1/16 is what makes it deployable at all.

---

## 5. Compute

**Construction: zero MACs.** The whole stage is subtraction, concatenation,
slicing and reshaping — 48 nodes, 15.3 MiB of activation traffic, no arithmetic
that a MAC counter registers [MEASUREMENT: EXP-001].

That does not make it free. Forty-eight data-movement nodes on an accelerator
whose cost model is dominated by moving tensors between memories is a real
expense that a FLOP count cannot see. Whether it registers as latency is a W8
measurement, and it is one of the concrete cases where the "fewer FLOPs =
faster" assumption gets tested.

**Aggregation: 2.369 G MACs**, 4.2 % of the model [MEASUREMENT: EXP-001].

| Node | Operation | Output | MACs |
|---:|---|---|---:|
| 118, 120, 122, 124 | Conv3d 32→32, 3×3×3 | 1×32×12×23×77 | 587.6 M each |
| 126 | Conv3d 32→1, 3×3×3 | 1×1×12×23×77 | 18.4 M |

Each of the first four is `output_elements × in_channels × kernel_volume =
680,064 × 32 × 27 = 587,575,296` MACs. Five 3D convolutions, 111,585
parameters, 2,368,662,912 MACs in total — still a rounding error next to the
refinement stage's 50.79 G.

---

## 6. From volume to disparity

The aggregated volume is `1×12×23×77` after `Squeeze`: one scalar cost per
candidate per low-resolution pixel. Turning that into a disparity uses
**soft-argmin**:

```
d̂(u,v) = Σₖ  k · softmax(−cost(k, u, v))ₖ
```

Negating first is what makes it an arg*min* — low cost becomes high weight. The
softmax makes it differentiable and, more importantly, makes the output
**continuous**: the weighted average of integer candidate indices is not an
integer. That is where StereoNet's sub-pixel precision comes from, and it is the
mechanism the paper credits for allowing such a small cost volume [SR-010].

It is worth being precise about what that buys here. The candidates are 16
full-resolution pixels apart, so soft-argmin is interpolating within intervals
of 16 pixels. Its output is continuous, but its *accuracy* between candidates
depends entirely on the aggregation network having shaped the cost curve so that
the interpolation lands correctly. There is no matching evidence at
intermediate disparities — none was ever computed.

**In this model the soft-argmin runs at full resolution, not at 1/16.** The
`1×12×23×77` cost tensor is bilinearly upsampled to `1×12×368×1232` (20.8 MiB)
before `Neg`, `Softmax`, `Mul` and `ReduceSum` [VERIFIED: SR-001]. The index
grid it multiplies by is a baked-in `1×12×368×1232` constant occupying 92 % of
the ONNX file [MEASUREMENT: EXP-001].

Doing the reduction first and upsampling a single-channel disparity map
afterwards would move the same work onto 1/192 as much data. Whether the
accuracy differs is **[HYPOTHESIS]**, noted for Phase 2. The baseline is frozen
and is not being modified to find out.

---

## 7. Summary

| | |
|---|---|
| Volume shape | `1 × 32 × 12 × 23 × 77` [VERIFIED] |
| Memory (fp32) | 2.59 MiB [MEASUREMENT: EXP-001] |
| Construction | subtraction, zero MACs, 48 data-movement nodes [VERIFIED] |
| Aggregation | 5 × Conv3d, 2.369 G MACs, 4.2 % of the model [MEASUREMENT: EXP-001] |
| Disparity recovery | soft-argmin at **full** resolution [VERIFIED] |
| Effective search | 0–176 px in 16 px steps → 2.2 m minimum range, one interval covering 24 m to infinity [MEASUREMENT: EXP-003 calibration] |
| Share of total cost | **4.2 % of MACs, 1.9 % of activation traffic** |

The cost volume is not this model's bottleneck. It was made small enough not to
be, and the cost of that decision was moved into the refinement stage that has
to repair the result — which is where 90.6 % of the arithmetic now lives. That
trade is the single most important thing Phase 1 has established about this
architecture, and it is the natural starting point for Phase 2's questions.

---

*Sources: [SR-001] exported ONNX, [SR-002] Model Zoo config, [SR-011] upstream
implementation. Measurements: [EXP-001] static analysis, [EXP-003] KITTI
calibration.*
