# Failure analysis

Where the reference model's accuracy comes from, where it does not, and what
explains the difference. Ranked by severity, each entry with its evidence and an
explicit confidence status.

Every entry is a measurement on the frozen baseline. Nothing here has been
changed, and no fix has been implemented.

---

## Summary of ranked failures

| # | Failure | Severity | Status | Evidence |
|---|---|---|---|---|
| **F1** | **No disparity search is performed at all** | **Critical** | **VERIFIED** | EXP-010 |
| F2 | Depth error grows 43× from near to far range | High | MEASUREMENT | EXP-012 |
| F3 | The intermediate disparity estimate is anti-correlated with truth | High | MEASUREMENT | EXP-006, EXP-008 |
| F4 | No confidence, validity or occlusion output exists | High | VERIFIED | SR-001, SR-007 |
| F5 | Occluded pixels cost 0.62 D1 points and are not marked | Medium | MEASUREMENT | EXP-005 |
| F6 | The published metric is mislabelled as EPE | Medium | VERIFIED | EXP-005 |
| F7 | 92 % of the model file is a redundant constant | Medium | VERIFIED | EXP-001, EXP-006 |
| F8 | The application discards sub-pixel precision at output | Medium | VERIFIED | SR-007 |
| F9 | The downsampling stack is linear | Low | VERIFIED | SR-001, EXP-011 test |
| F10 | The final ReLU never activates | Low | MEASUREMENT | EXP-006 |

---

## F1 — The model performs no disparity search. **Critical.**

**Observed.** Every one of the twelve disparity slices in the cost volume is
bit-identical. The maximum absolute difference between any slice and slice 0 is
exactly 0.0, in every scene tested [MEASUREMENT: EXP-010].

**Expected.** A cost volume exists to compare the left image against the right
at several candidate shifts. Twelve candidates should produce twelve different
tensors.

**Mechanism — verified, not hypothesised.** Each disparity level is built as
[VERIFIED: SR-001]:

```
Concat([left_features (width 77), zeros (width k)], axis=3)  -> width 77+k
Slice(starts=[0], ends=[77], axes=[3], steps=[1])            -> width 77
Sub(that, right_features)
```

The zero block is appended on the **right**, and the slice then takes columns
`[0:77]` — which are exactly the original left features. The intended left-shift
never happens. To shift left by `k`, the slice would have to start at `k`, or the
padding would have to go on the left.

The identical construction is in the upstream PyTorch source [SOURCE: SR-011,
`utils/cost_volume.py`], so this originates upstream and was carried faithfully
through ONNX export and Hailo compilation. Neither introduced it.

**Consequences, all measured:**

- The cost volume is the single tensor `left_features − right_features`,
  replicated twelve times. It encodes a photometric difference at *zero*
  disparity, not a search.
- The 3D aggregation's 2.369 G MACs and 111,585 parameters operate on twelve
  identical copies of one tensor [MEASUREMENT: EXP-001].
- The 21.76 MB index-grid constant weights a softmax over twelve identical
  inputs [MEASUREMENT: EXP-001, EXP-006].
- F3 follows directly.

**What it does *not* mean.** The model is not ignoring the right image.
Corrupting the right input costs up to **90 D1 points** — D1 rises from 9.33 %
to 99.29 %, EPE from 1.44 px to 69.4 px [MEASUREMENT: EXP-007]. `left − right` at
zero shift is a genuine stereo cue: where a surface is near, the two views differ
a lot at zero shift; where it is far, they differ little. The model learned to
read disparity out of that difference magnitude, plus the left image, rather than
by searching for correspondences.

**Why this matters most.** The published D1 of 8.223 % [SOURCE: SR-002],
reproduced exactly [MEASUREMENT: EXP-005], is achieved **without stereo
correspondence search**. Every conclusion anyone might draw from this model about
cost-volume design, disparity range, or 3D aggregation is drawn from a network
whose cost volume is inert. It also means the architecture has substantial
unrealised headroom, and that any Phase 2 comparison against "StereoNet as
deployed" must state which version it means.

**Severity: critical**, because it invalidates the baseline as evidence about
cost-volume behaviour while leaving it valid as evidence about achievable
accuracy.

**Not fixed.** The baseline is frozen. The independent implementation reproduces
the no-op shift by default; the intended shift exists behind a flag that nothing
in Phase 1 uses.

---

## F2 — Depth error grows 43× from near to far. **High.**

**Measured** [MEASUREMENT: EXP-012], 40 scenes, 3.8 M valid pixels, per-scene
KITTI calibration:

| Depth band | Pixels | Share | Disparity EPE | Depth MAE | Relative | δ₁ |
|---|---:|---:|---:|---:|---:|---:|
| 0–10 m | 1,631,127 | 42.9 % | 1.467 px | **0.213 m** | 2.8 % | 99.1 % |
| 10–20 m | 1,444,383 | 38.0 % | 1.158 px | 0.639 m | 4.4 % | 97.8 % |
| 20–30 m | 378,053 | 9.9 % | 1.290 px | 2.082 m | 8.5 % | 93.2 % |
| 30–50 m | 249,913 | 6.6 % | 1.247 px | 4.384 m | 11.4 % | 87.1 % |
| 50–80 m | 98,641 | 2.6 % | 1.311 px | **9.158 m** | 15.0 % | 75.5 % |
| 80 m+ | 680 | 0.0 % | 1.023 px | 13.498 m | 16.6 % | 74.1 % |

**Disparity error is essentially flat with range** — 1.02 to 1.47 px everywhere —
**while depth error grows 43×.** This is the `ΔZ ≈ Z²/(fB)·Δd` relation derived in
`stereo_fundamentals.md` and quantified in [MEASUREMENT: EXP-003], now observed
in a real model's output rather than predicted.

**Cause: geometry, not the network.** No architecture escapes it. What an
architecture controls is `Δd`, and because the penalty is quadratic in depth,
sub-pixel accuracy at small disparities is worth far more than the same
improvement up close.

**Why the headline numbers look better than the far field is.** Overall depth
Abs Rel is 0.0489 and δ₁ is 96.6 %, but 81 % of valid pixels are inside 20 m. The
aggregate is dominated by the easy near field. At 50–80 m, one pixel in four
fails the δ₁ < 1.25 test.

**Status: MEASUREMENT.** The mechanism is a derivation, confirmed against a
numerical derivative in `tests/test_geometry.py`.

---

## F3 — The intermediate disparity estimate is anti-correlated with truth. **High.**

**Observed** [MEASUREMENT: EXP-006, EXP-008]:

| Quantity | Correlation with truth | Mean (candidate units) |
|---|---:|---:|
| Ground truth | — | 2.51 |
| Soft-argmin output, as the graph computes it | **−0.976** | 7.22 |
| Same with the sign flipped | +0.681 | 1.10 |
| Hard argmin | −0.931 | 7.29 |
| Hard argmax | +0.571 | 0.85 |
| Final model output | +0.989 | 2.50 |

The soft-argmin output is also anti-correlated with the model's own final
disparity at r = −0.975, while the refinement residual matches it at r = +0.9998
and supplies **76.5 %** of the output magnitude [MEASUREMENT: EXP-006].

**Cause: F1.** With every disparity slice identical, the aggregated tensor
carries no disparity information. Whatever variation exists along the disparity
axis comes from the 3D convolutions' boundary handling at the ends of that axis,
not from matching. The `Neg` before the softmax then reduces a tensor that has no
consistent cost-or-score semantics, which is why neither reading fits: the
sign-flipped version correlates positively (+0.68) and lands near the true mean
(1.10 against 2.51), while the graph's own version correlates strongly negatively
and sits far from it (7.22).

**Consequence.** The tensor the architecture presents as a coarse disparity
estimate is not one. Anyone using it as a confidence signal, an early exit, or
the coarse level of a cascade would be building on noise. The refinement stage
must invert it, rescale it from candidate units to pixels, and produce the actual
disparity — which is why it dominates the model.

**Status: MEASUREMENT** for the correlations; **INFERENCE** for the causal link
to F1, though the inference is tight: F1 is verified and mechanically forces the
cost tensor to be uninformative.

---

## F4 — No confidence, validity, or occlusion output. **High.**

**Verified** [SR-001]: the graph has exactly one output, `1×1×368×1232`
disparity. There is no second head, no variance estimate, no matching-quality
signal.

The final `ReLU` clamps negatives to zero, so **"no match here" and "zero
disparity" are the same value**. Zero disparity means infinite depth; an
unmatched pixel means no information. The model cannot distinguish them, and
neither can anything downstream.

Combined with F1 — no search, therefore no cost curve to derive confidence from —
there is no obvious way to add confidence to this model without changing the
architecture.

**Consequence for a depth system.** Occluded regions, textureless regions and
sky all produce confident-looking disparity values with no marker. A consumer
must either trust everything or apply its own heuristics.

---

## F5 — Occluded pixels cost 0.62 D1 points and are unmarked. **Medium.**

**Measured** [MEASUREMENT: EXP-005], same predictions, same protocol otherwise:

| Ground truth | D1 | EPE |
|---|---:|---:|
| `disp_noc_0` (non-occluded) | 7.530 % | 1.261 px |
| `disp_occ_0` (all, includes occluded) | 8.154 % | 1.313 px |

Occluded pixels are 0.62 D1 points harder. Expected — there is no correct
disparity for a surface only one camera can see — but the model emits a value
there with no indication (F4), and Hailo's published figure is the harder
`disp_occ_0` number.

---

## F6 — The published metric is mislabelled. **Medium.**

`eval_metric: EPE`, `full_precision_result: 8.223` [SOURCE: SR-002] and "Float
EPE 8.22" [SOURCE: SR-004] all describe a quantity that is **not** end-point
error. Hailo's evaluator computes the KITTI D1 outlier rate, names the variable
`three_pixel_correct_rate`, averages it per image and reports it as a percentage
[VERIFIED: SR-002 repository, reproduced in EXP-005].

The model's actual EPE is **1.313 px** [MEASUREMENT: EXP-005].

**Why it is a failure and not a curiosity.** Anyone comparing 8.223 against
another architecture's EPE would conclude this model is roughly six times worse
than it is. Two further deviations from the KITTI convention are also unstated:
ground truth divided by 255 rather than 256 (+0.055 points), and per-image rather
than pooled averaging (+0.014 points here) [MEASUREMENT: EXP-005].

---

## F7 — 92 % of the model file is a redundant constant. **Medium.**

`/Tile_output_0` is a `1×12×368×1232` float32 initializer of **21,762,048
bytes** — 92 % of the 23.7 MB ONNX. The learned weights are 1,694,344 bytes
[MEASUREMENT: EXP-001].

It holds **twelve distinct values**: `arange(12)`, constant across height and
width, verified directly [MEASUREMENT: EXP-006]. A broadcast against a 12-element
vector would be numerically identical and occupy 48 bytes.

It is an export artifact — the source builds the grid with `repeat()` at runtime
[SOURCE: SR-011], and the exporter materialised the result. And because of F1, it
weights a softmax over twelve identical inputs.

---

## F8 — Sub-pixel precision is discarded at the output. **Medium.**

The shipped application's entire postprocess is
`cv::Mat(h, w, CV_8U, data).clone()` [VERIFIED: SR-007]. Disparities above 255
saturate and everything between integers is lost.

The soft-argmin exists to produce continuous, sub-pixel disparity — and by F2,
sub-pixel precision is exactly what far-range depth depends on. The application
throws it away. Acceptable for a visualisation demo; not acceptable in a system
that consumes the values, and worth knowing before treating the application as a
reference design.

---

## F9 — The downsampling stack is linear. **Low.**

Four stacked stride-2 5×5 convolutions with **no activation between them**
[VERIFIED: SR-001, and pinned by a linearity test in `tests/test_stereonet.py`].
Four composed linear convolutions have the representational power of one.

They are not free — 1.224 G MACs of the feature extractor's 1.436 G, per branch,
per image [MEASUREMENT: EXP-001] — and feature extraction turned out to be the
most time-disproportionate stage in profiling, 5.1 % of MACs but 18.4 % of
measured time [MEASUREMENT: EXP-013].

**Low severity** only because the network still reaches 8.15 % D1 with it, so
whatever expressiveness is lost is evidently not binding. That it is
*architecturally* wasteful is separate from whether it *costs accuracy*, and the
second question is not answered here.

---

## F10 — The final ReLU never activates. **Low.**

Across every pixel of five scenes, the pre-ReLU sum is negative in **0.0 %** of
cases [MEASUREMENT: EXP-006]. The clamp is dead code on real data. It still
serves as a guarantee that no negative disparity escapes, which matters for the
`Z = fB/d` conversion, so it is not useless — merely never exercised.

---

## What is *not* broken

Worth stating plainly, because Phase 2 should not spend effort on things that
already work.

- **Accuracy is genuinely decent.** 8.15 % D1-all, 1.31 px EPE, 96.6 % δ₁ on
  depth [MEASUREMENT: EXP-005, EXP-012]. For 423,586 parameters that is a strong
  result — and it is achieved without a working cost volume.
- **Near-field depth is good.** 0.213 m mean absolute error inside 10 m, where
  43 % of the valid pixels are [MEASUREMENT: EXP-012].
- **The refinement stage works very well.** Its residual correlates with the
  final output at r = +0.9998 and it recovers a usable disparity from a
  degenerate input [MEASUREMENT: EXP-006]. The colour-guided, dilated,
  full-resolution design is doing real work.
- **The disparity range is adequate for KITTI.** Ground-truth disparity reaches
  9.56 candidate units against a nominal 11, and **0.0 %** of pixels exceed the
  range [MEASUREMENT: EXP-008]. An earlier expectation that the range might be
  truncating the far field was tested and is **refuted**.
- **The published figure is exactly reproducible** once the protocol is known
  [MEASUREMENT: EXP-005].

---

## Not yet examined

Stated so the gaps are visible rather than implied.

| Gap | Status |
|---|---|
| Textureless, repetitive, reflective, thin-structure and boundary failures as a curated scene set | **UNKNOWN** — needs Middlebury 2014, not yet downloaded |
| Low light and exposure mismatch | **UNKNOWN** — KITTI does not isolate these |
| Error maps and qualitative visualisations per failure class | **UNKNOWN** |
| Whether F1 costs accuracy, and how much | **Deliberately not tested.** Requires changing the cost volume, which is a Phase 2 experiment on a Phase 2 model, not a Phase 1 modification of a frozen baseline. |

---

*Measurements: EXP-001, EXP-003, EXP-005 through EXP-013. Sources: [SR-001]
ONNX, [SR-002] Model Zoo configuration and repository, [SR-004] benchmark table,
[SR-007] application source, [SR-011] upstream implementation.*
