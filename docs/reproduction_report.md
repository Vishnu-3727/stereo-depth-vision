# Reproduction report

Whether the published behaviour of the Hailo StereoNet deployment can be
reproduced independently, and what had to be understood to do it.

**Result: yes, exactly.** Hailo publishes 8.223; running the reference ONNX
under the reconstructed protocol gives **8.2237** — a gap of +0.0007 points,
0.008 % relative [MEASUREMENT: EXP-005].

Getting there required discovering that the published figure is not what its
label says it is.

---

## 1. The headline finding: 8.223 is not an EPE

Hailo's configuration declares `eval_metric: EPE` with
`full_precision_result: 8.223` [SOURCE: SR-002], and the Hailo-8 table lists
"Float EPE 8.22" [SOURCE: SR-004]. The StereoNet paper reports KITTI errors
around 1 px. A sevenfold disagreement looked like a serious problem.

It was not a disagreement. Reading the evaluator Hailo actually runs
(`core/eval/stereo_evaluation.py` [SR-002 repository]):

```python
correct = (diff < 3) | (diff < true_disp * 0.05)
three_pixel_correct_rate = 1 - float(np.sum(correct)) / float(len(index[:, 0]))
self.TotalEPE += three_pixel_correct_rate
```

That is the **KITTI D1 outlier rate** — a pixel is an outlier when its error
exceeds 3 px *and* exceeds 5 % of ground truth — accumulated per image and
divided by the image count. `Eval.is_percentage()` returns `True` by default and
the stereo evaluator does not override it, so the fraction 0.08223 is displayed
as **8.223 %**.

The variable is named `three_pixel_correct_rate` in Hailo's own source. Only the
accumulator it is added to is called `TotalEPE`.

**So: 8.223 is a D1 outlier rate in percent, not an end-point error in pixels.**
The apparent conflict with the paper was a units problem. On the same
predictions, the model's actual end-point error is **1.313 px**
[MEASUREMENT: EXP-005] — entirely consistent with the paper's range.

This is exactly the kind of number that, quoted into a comparison table, would
have made every subsequent conclusion in this project wrong.

---

## 2. The reconstructed protocol

Assembled from three files in the Model Zoo repository [SR-002], and
reimplemented in `src/datasets/kitti2015.py` and `src/evaluation/metrics.py`:

| Element | Value | Source |
|---|---|---|
| Dataset | KITTI 2015 **training** set | `create_kitti_stereo_tfrecord.py` |
| Split | sorted names, indices **160–199**; 0–159 are the quantisation calibration set | same |
| Frames | `_10` only (frame 11 has no disparity ground truth) | same |
| Ground truth | `disp_occ_0` — includes occluded pixels, so this is **D1-all** | same |
| Ground-truth scale | divide raw 16-bit values by **255.0** | `parse_kitti_stereo.py` |
| Preprocessing | zero-pad bottom/right to ≥368×1232, crop top-left, **no resize** | `stereonet_preprocessing.py` |
| Valid pixels | `gt > 0` | `stereo_evaluation.py` |
| Metric | D1 outlier rate: error ≥3 px **and** ≥5 % of ground truth | same |
| Aggregation | mean over images, not pooled over pixels | same |
| Reporting | × 100, as a percentage | `eval_base_class.py` |
| Normalisation | ImageNet mean/std, inserted by the compiler, **not in the ONNX** | `stereonet.yaml`, `stereonet.alls` |

Two of these deviate from the KITTI convention and are worth stating plainly:

- **The 1/255 divisor.** KITTI stores `round(disparity × 256)`; the official
  scale is 1/256. Hailo divides by 255, making its ground-truth disparities
  256/255 = **1.0039×** larger than the true values.
- **Per-image averaging.** KITTI pools outliers over all valid pixels. Averaging
  per image weights a scene with few valid pixels as heavily as one with many.

Neither is wrong as long as it is stated. Neither is stated.

---

## 3. Results

**[MEASUREMENT: EXP-005]** — reference ONNX [SR-001], onnxruntime 1.27.0 CPU,
fp32, batch 1, all 40 validation scenes, 3,802,797 valid pixels.

| Protocol variant | Headline % | vs published | True EPE (px) | D1 pooled % |
|---|---:|---:|---:|---:|
| **`hailo_exact`** — 1/255 GT, per-image average | **8.2237** | **+0.0007** | 1.342 | 8.211 |
| `hailo_scale_kitti_gt` — 1/256 GT, per-image average | 8.1685 | −0.0545 | 1.313 | 8.154 |
| `kitti_official_d1_all` — 1/256 GT, pooled | 8.1544 | −0.0686 | 1.313 | 8.154 |
| `kitti_official_d1_noc` — `disp_noc_0`, pooled | 7.5301 | −0.6929 | 1.261 | 7.530 |
| *Hailo published* [SOURCE: SR-002] | *8.223* | — | — | — |

Full metrics under the official KITTI convention (`kitti_official_d1_all`):

| Metric | Value |
|---|---:|
| EPE | **1.313 px** |
| RMSE | 2.583 px |
| D1-all | 8.154 % |
| D1-noc | 7.530 % |
| bad-1 | 38.90 % |
| bad-2 | 16.08 % |
| bad-3 | 8.64 % |

The four variants were fixed **before** running, from what the source says the
evaluator does. None was adjusted to close the gap.

### Isolated effects

- **The 1/255 divisor** costs +0.055 points on the headline. It inflates ground
  truth, so predictions look slightly worse.
- **Per-image averaging versus pooling** costs +0.014 points here — small,
  because KITTI scenes have similar valid-pixel counts. On a dataset with more
  variable coverage it would not be small.
- **Occluded pixels** cost +0.624 points. D1-all is meaningfully harder than
  D1-noc, as expected: occluded pixels have no correct answer.

### The residual +0.0007

Within rounding of a value published to four significant figures. No further
explanation is required, and none is invented. The reproduction is treated as
exact.

---

## 4. What this does and does not establish

**Established:**

- The reference ONNX runs and produces sensible disparity: range 0.61–136.29 px,
  mean 32.67, no zeros [MEASUREMENT: EXP-005].
- The published accuracy figure is reproducible to four significant figures from
  public artifacts alone.
- The complete evaluation protocol is now known and reimplemented, and can be
  frozen as the baseline protocol for the rest of the project.
- The published figure's true meaning is documented, so it will not be
  mis-compared later.

**Not established:**

- **Nothing about Hailo hardware.** The published hardware EPE of 10.3 and 10.7
  FPS [SOURCE: SR-004] are SOURCE values and remain so. No Hailo device is
  available. The onnxruntime CPU latency recorded in EXP-005 (0.56 s mean per
  pair) is our own measurement on our own host and is **not** a Hailo number,
  not a comparison to one, and not a proxy for one.
- **Nothing about our own implementation yet.** EXP-005 runs Hailo's ONNX, not
  code we wrote. The independent implementation and its tensor-level comparison
  against this ONNX are W5/W6, still outstanding.
- **Nothing about depth accuracy.** These are disparity metrics. Depth
  validation is W7.

---

## 5. Baseline freeze

With the reference behaviour reproduced, the baseline is frozen:

- `reference/` is read-only, hashed in `reference/MANIFEST.md`.
- The evaluation protocol is fixed as the four variants in
  `scripts/exp_reproduce_hailo.py`, with `hailo_exact` as the comparison point
  for anything Hailo publishes and `kitti_official_d1_all` for anything else.
- **No architectural change to the baseline for the remainder of Phase 1**, and
  no adjustment of the protocol to improve a number.

Weaknesses found from here on are recorded as
`problem → evidence → measurement → hypothesis → possible Phase 2 direction`
and left in place.

---

## 6. Discrepancies register

Everything found so far where two sources disagree. Nothing here is resolved by
preference; the artifact ordering in `reference_pipeline.md` decides which one
describes the deployed baseline, and both values are kept.

| # | Discrepancy | Resolution | Status |
|---|---|---|---|
| D1 | Config labels the metric `EPE`; the evaluator computes a D1 outlier rate | Evaluator source is authoritative. 8.223 is D1-all in percent. True EPE is 1.313 px. | **RESOLVED** [EXP-005] |
| D2 | Upstream README says the cost volume uses concatenation; the ONNX uses subtraction | ONNX is the compiled artifact and is authoritative. Both recorded. | **RESOLVED** [EXP-001] |
| D3 | Config publishes 623.1K parameters; unique learned tensors total 423,586 | Hailo counts the shared Siamese extractor once per graph occurrence: 423,586 + 199,552 = 623,138. | **RESOLVED** [EXP-001] |
| D4 | Config publishes 112.2G operations; our MAC count is 56.04G | Hailo's figure is 2 × MACs. 112.08G against 112.2G, −0.11 %. | **RESOLVED** [EXP-001] |
| D5 | Ground-truth scale 1/255 versus KITTI's 1/256 | Deviation confirmed in Hailo's parser. Effect measured at +0.055 points. | **RESOLVED** [EXP-005] |
| D6 | The upstream author reports falling short of the paper's accuracy [SR-011] | Cannot be assessed until the paper is read in full [SR-010]. | **UNKNOWN** |
| D7 | Config lists `hailo15h, hailo10h`; the benchmark table and application are Hailo-8 | Not yet investigated. | **UNKNOWN** |
| D8 | Downsampling stack has no activations, unlike the paper's description | Verified absent in both ONNX and upstream source. Whether the paper specifies otherwise needs the full text. | **INFERENCE** pending [SR-010] |

---

## 7. Reproducing this

```
python scripts/hash_reference.py       # verify the artifacts
python scripts/exp_reproduce_hailo.py  # full 40-scene run, ~50 s on CPU
python -m pytest tests/ -q             # protocol and metric checks
```

Raw predictions in `results/reproduction/predictions.npz`, per-variant results
in `results/reproduction/variants.json`, and the full run record in
`experiments/EXP-005/`.

---

*Measurements: [EXP-005]. Sources: [SR-001] ONNX, [SR-002] Model Zoo
configuration and repository, [SR-004] Hailo-8 benchmark table, [SR-011]
upstream implementation, [SR-020] and [SR-021] KITTI 2015.*
