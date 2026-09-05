# The reference stereo-depth pipeline, as Hailo actually ships it

What happens between two camera images and whatever comes out the other end,
with learned operations, fixed geometry and runtime concerns kept strictly
apart.

The short version: **the public Hailo pipeline is a disparity pipeline, not a
depth pipeline.** No calibration, no baseline, no `Z = fB/d`. The metric-depth
half of a stereo depth system is not present in the shipped application and
would have to be built.

---

## 1. Evidence ordering

Where sources disagree, this is the order used to decide what the *deployed
baseline* does. A lower-priority source never silently overrides a higher one;
conflicts are recorded with both values.

1. The Hailo ONNX [SR-001] — the executable graph
2. HEF metadata [SR-005] — acquired, not decoded
3. The `.alls` compiler script [SR-003]
4. The upstream implementation [SR-011]
5. The Model Zoo configuration [SR-002]
6. Hailo documentation [SR-004]
7. The original paper [SR-010]
8. Secondary sources

This ordering has already mattered twice: the cost-volume metric (ONNX says
subtraction, the upstream README says concatenation) and the meaning of the
published accuracy figure (the evaluator source says D1 outlier rate, the
configuration label says EPE).

---

## 2. The pipeline

```
  left camera            right camera
       │                      │
       │   (A) synchronisation: ASSUMED, not performed
       │   (B) calibration:     ASSUMED, not performed
       │   (C) rectification:   ASSUMED, not performed
       │                      │
       ▼                      ▼
  ┌────────────────────────────────┐
  │ C. PREPROCESSING               │   pad bottom/right to >= 368x1232,
  │    (deterministic, host)       │   crop top-left to 368x1232, no resize
  └────────────────────────────────┘
       │                      │
       ▼                      ▼
  ┌────────────────────────────────┐
  │ E. NORMALISATION               │   (x - mean) / std, ImageNet statistics
  │    (compiler-inserted layer)   │   NOT in the ONNX; added by the parser
  └────────────────────────────────┘
       │                      │
       ▼                      ▼
  ┌────────────────────────────────┐
  │ A. NEURAL NETWORK              │   168 nodes, static shapes
  │    Siamese features (1/16)     │
  │    cost volume, 12 candidates  │
  │    3D aggregation              │
  │    upsample + soft-argmin      │
  │    full-resolution refinement  │
  └────────────────────────────────┘
       │
       ▼
   disparity, 1 x 1 x 368 x 1232 float32, in pixels
       │
       ├──────────────► D. POSTPROCESSING (the shipped application)
       │                   cast to 8-bit, display or save as an image.
       │                   That is the entire postprocess.
       │
       └──────────────► B. DETERMINISTIC GEOMETRY
                           Z = fB / d
                           *** NOT PRESENT IN THE PUBLIC PIPELINE ***
```

---

## 3. A — Neural network operations

Fully specified in `stereonet_architecture.md` and `cost_volume_analysis.md`.
Summary [VERIFIED: SR-001]:

| Stage | Output | MAC share |
|---|---|---:|
| Siamese feature extraction, ×2, shared weights | 1×32×23×77 each | 5.2 % |
| Cost volume by subtraction, 12 candidates | 1×32×12×23×77 | 0 % |
| 3D aggregation, 5 convolutions | 1×12×23×77 | 4.2 % |
| Bilinear upsample then soft-argmin at full resolution | 1×1×368×1232 | 0.0 % |
| Refinement, 13 convolutions, dilations 1/2/4/8/1/1 | 1×1×368×1232 | **90.6 %** |
| Residual add and ReLU | 1×1×368×1232 | 0 % |

**Output units: disparity in pixels**, at the cropped 368×1232 resolution. The
final `ReLU` clamps negatives to zero, so the model cannot express "no match
here" — an unmatched pixel and a genuine zero-disparity pixel are the same
value. There is **no confidence or validity output** of any kind
[VERIFIED: SR-001]. Anything downstream that needs to know whether a pixel is
trustworthy has to infer it.

---

## 4. B — Deterministic geometry

**Not present in the public Hailo pipeline.** [VERIFIED: SR-007]

Searching the shipped application source for `baseline`, `focal`, `calib`,
`depth =` or any disparity-to-depth arithmetic returns nothing. There is no
calibration file format, no camera-parameter argument, and no conversion.

This project implements the missing half itself, in `src/geometry/stereo.py`:
calibration parsing, `Z = fB/d`, explicit invalid-pixel handling, and error
propagation. Quantified in `stereo_fundamentals.md` [MEASUREMENT: EXP-003].

### What is assumed and never checked

- **Synchronisation.** The two preprocessing threads read their streams
  independently and the inference stage pairs whatever arrives
  [VERIFIED: SR-007]. Nothing enforces that the two frames are simultaneous.
- **Calibration.** Never read, never used.
- **Rectification.** The architecture only searches horizontally, so an
  unrectified pair breaks the model's core assumption. Nothing detects this.
  The failure mode is silent: a plausible-looking, wrong disparity map.

These are reasonable assumptions for a demonstration application. They are
listed because a *system* has to satisfy them, and satisfying them is work the
public pipeline leaves entirely to the integrator.

---

## 5. C — Preprocessing

From `core/preprocessing/stereonet_preprocessing.py` [SR-002 repository],
reimplemented and tested in `src/datasets/kitti2015.py`:

1. Zero-pad the **bottom and right** edges up to at least 368×1232.
2. Crop to 368×1232 anchored at the **top-left**.
3. No resizing at any point.

For KITTI's 375×1242 images nothing is padded: **7 rows are removed from the
bottom and 10 columns from the right.**

That crop is not neutral. The bottom rows of a KITTI frame are the road surface
immediately in front of the car — the largest disparities in the scene. Removing
them removes part of the near-field distribution the model is scored on. The
size of that effect is **UNKNOWN** and is a W7 question.

Ground truth receives the identical pad-and-crop, so alignment is preserved.

**This answers open question A1**: the 368×1232 input comes from cropping, not
resizing, so disparity values need no rescaling — a resize would have required
scaling disparities by the width ratio, and getting that wrong is a classic
source of silent evaluation error.

---

## 6. D — Postprocessing

The entire postprocess in the shipped application [VERIFIED: SR-007]:

```cpp
frame_to_draw = cv::Mat(s.height, s.width, CV_8U, data).clone();
```

The raw output buffer is wrapped as an 8-bit single-channel image and displayed
or written to a file. Three consequences:

- **No depth conversion**, as above.
- **8-bit output.** Disparities above 255 saturate, and everything between
  integers is lost. The sub-pixel precision the soft-argmin exists to produce is
  discarded at the display stage. For KITTI's disparity range this is mostly a
  visualisation choice; for a system that consumes the values it would be a
  serious information loss.
- **No filtering, no invalid-pixel handling, no confidence.**

The function still carries a docstring describing classifier post-processing and
top-1 overlay [VERIFIED: SR-007] — the app is a thin adaptation of a
classification example, which is worth knowing before treating it as a reference
system design.

---

## 7. E — Runtime and hardware

**Normalisation is a compiler-inserted layer.** The ONNX contains no
normalisation [VERIFIED: SR-001]. `normalize_in_net: true` in the configuration
[SOURCE: SR-002] makes the parser insert it, and `stereonet.alls` places it
first in the graph [SOURCE: SR-003]:

```
stereonet/normalization1, stereonet/normalization2 =
    normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
```

Running the ONNX directly therefore requires doing this on the host first — as
this project's reproduction does. Missing it would produce a wrong result with
no error.

**The compiler restructures the refinement stage heavily.** From
`stereonet.alls` [SOURCE: SR-003]: `conv42`–`conv52` are spatially defused into
up to 22 pieces, with `shape_splitter(SPLIT_WIDTH, …)`, re-concatenation trees,
and multi-way `shortcut` fanout for the residual connections. Those layers are
the full-resolution refinement block, where every activation is 55.34 MiB
[MEASUREMENT: EXP-001].

The compiler could not place those layers whole. That is the vendor's own
toolchain reporting, indirectly, where this model strains the hardware — and it
agrees with our independent static analysis that the same stage carries 90.6 %
of the arithmetic. **That it is the runtime bottleneck remains [HYPOTHESIS]
until W8 profiling.**

Also visible in the configuration [SOURCE: SR-002]:

- Quantisation calibration set: the **first 160** KITTI 2015 training scenes,
  disjoint from the 40-scene validation split.
- `supported_hw_arch: hailo15h, hailo10h` in the current configuration, while
  the published benchmark table is for Hailo-8 [SOURCE: SR-004] and the shipped
  application states Hailo-8 only. The relationship between these is
  **UNKNOWN**.

---

## 8. What a complete depth system needs that this does not have

Not a criticism of the reference — it is a model deployment example, not a
product. It is a list of what Phase 2 has to own.

| Missing | Consequence |
|---|---|
| Calibration ingest | No metric depth is possible without it |
| Rectification, or a check that input is rectified | Silent, plausible, wrong output |
| Frame synchronisation | Motion between the two frames appears as disparity error |
| Confidence or validity output | Occlusions and textureless regions are indistinguishable from measurements |
| Invalid-disparity handling | The final ReLU makes "no match" and "zero disparity" identical |
| Sub-pixel-preserving output | The 8-bit cast discards what the architecture works hardest to produce |
| Disparity-to-depth conversion | The system stops one step short of its stated purpose |

---

## 9. Open questions from this document

| # | Question | Status | Where |
|---|---|---|---|
| A1 | Crop or resize to 368×1232? | **RESOLVED**: pad then top-left crop, no resize [VERIFIED: SR-002 repository] | this document |
| A2 | Does the application produce metric depth? | **RESOLVED**: no. Disparity only, cast to 8-bit [VERIFIED: SR-007] | this document |
| A7 | How much does dropping the bottom 7 rows change the accuracy figure? | UNKNOWN | W7 |
| A8 | Why does the configuration list hailo15h/hailo10h while the benchmark and application are Hailo-8? | UNKNOWN | W9 |
| A4 | Is refinement the runtime bottleneck as well as the static-MAC bottleneck? | HYPOTHESIS | W8 |

---

*Sources: [SR-001] ONNX, [SR-002] Model Zoo configuration and repository,
[SR-003] `.alls`, [SR-004] Hailo-8 benchmark table, [SR-007] application source.
Measurements: [EXP-001] static analysis, [EXP-003] depth geometry,
[EXP-005] reproduction.*
