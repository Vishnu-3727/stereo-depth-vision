# Stage C2 — Metric Depth / 3D Measurement Layer (MICROCHIP STEREONET)

One-line verdict: **PASS** — frozen disparity in, calibrated metric depth and XYZ out,
geometry layer proven not to alter inference, 135/135 validation checks green.

Claim labels used throughout: **VERIFIED** (measured in a run pasted below),
**INFERRED** (follows from verified facts), **UNKNOWN** (not measured),
**NOT APPLICABLE** (out of scope by design).

---

## 1 OBJECTIVE

Add a deterministic geometry POST-PROCESSING layer around the FROZEN ARM-P
StereoNet checkpoint: frozen disparity → calibrated metric depth (metres) →
optional 3D point measurement. **VERIFIED**: no training, no architecture,
weight, loss, augmentation, disparity-count, or normalization change was made
(checkpoint SHA256 identical before and after — see §18).

## 2 FROZEN MODEL STATE

- Checkpoint: `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth`
  SHA256 `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454`
  (**VERIFIED** — asserted before `torch.load`, re-hashed after the run).
- Config: `StereoNetConfig(downsample_levels=3, num_disparities=24,
  cost_volume_shift="right", regression_normalize=True)`,
  `n_params=397954`, run on `cuda`, torch 2.7.0+cu128 (**VERIFIED** — printed
  by the validation run, §13).
- Frozen reference EPE 1.1912168 px on the 40-scene hailo_val split
  (**VERIFIED** — pre-existing frozen record; this stage does not re-measure EPE
  and makes no EPE claim).
- ONNX `stage_c_deploy/armp_stereonet.onnx` SHA256 measured
  `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989`
  (**VERIFIED**). Documentation discrepancy (**VERIFIED**): the task prompt and
  `stage_c_deploy/target_resolution/evidence_summary.json` carry the transposed
  spelling `...deed1dce26...`; 8 other repo artifacts and the file on disk agree
  on `...deed1cde26...`. Disk is authoritative. Recorded as a documentation
  discrepancy, NOT an integrity failure. No existing file was fixed (prohibited).

## 3 INPUT/OUTPUT CONTRACT

- Input: KITTI 2015 rectified stereo pair (`image_2`/`image_3`), hailo_val scene
  name; calibration from `training/calib_cam_to_cam/<seq>.txt` (**VERIFIED**).
- Stage 1 output: raw disparity map, 368×1232 float64, original-image pixels,
  produced by the frozen `net(left, right)` call (**VERIFIED** — bitwise
  identical to a direct `model()` call, §13).
- Stage 2 output: depth map (metres, NaN where invalid) + boolean valid mask;
  no clipping of the data (**VERIFIED**).
- Stage 3 output: per-pixel or NxN-region measurement `(u, v, disparity, Z, X, Y)`
  with PIXEL vs REGION MEDIAN values labelled separately (**VERIFIED** — §14).

## 4 CALIBRATION SOURCE

Parser: the REUSED `parse_kitti_cam_to_cam` from `src/geometry/stereo.py`
(P_rect_02/P_rect_03, baseline = |tx_right − tx_left| / f). No second parser was
written (**VERIFIED** — `grep` shows the only `P_rect` reader is the import).
Measured calibration (the 40 validated scenes span two sequence calibrations):

| scenes | fx (px) | fy (px) | fx==fy | B (m) | cx | cy | S_rect |
|---|---|---|---|---|---|---|---|
| 000160–000169_10.png (10 scenes) | 718.3351 | 718.3351 | yes | 0.5301404873575021 | 600.3891 | 181.5122 | 1238×374 |
| 000170–000199_10.png (30 scenes) | 718.856 | 718.856 | yes | 0.5323318578 | 607.1928 | 185.2157 | 1241×376 |

(**VERIFIED** — `out/c2_validation.json` `scenes` array over all 40 scenes.)

## 5 DISPARITY UNIT ANALYSIS

**VERIFIED**: model output disparity is in original-image pixels (scale factor 1.0).
Evidence: `pad_and_crop` source contains no resize/interp call (inspected via
`inspect.getsource` in the validation run — `no_resize=True`); `normalize()` is
the ImageNet-scale-on-0-255-RGB transform the frozen evaluator uses; the network
emits full-resolution disparity with no downsampling of the output. Ground-truth
scale (256.0 vs Hailo's 255.0) concerns evaluation only and does not touch the
model output path.

## 6 RESOLUTION/RESIZE ANALYSIS

**VERIFIED** (measured, and it corrects the brief): the 40 hailo_val scenes do
NOT measure 375×1242. Measured original shapes over all 40 scenes:

```
{(374, 1238): 10, (376, 1241): 30}   # (h, w); cropped output is (368, 1232) in all cases
```

All originals exceed 368×1232 in both axes, so the run takes the crop-only path:
zero-pad bottom/right is a no-op, then a TOP-LEFT crop to 368×1232. A top-left
crop does not move the image origin, so a pixel `(u,v)` in the crop is the same
camera ray as `(u,v)` in the original — hence fx, fy, cx, cy are UNCHANGED and
the disparity scale factor is exactly 1.0 (**VERIFIED**: claim checked against
`pad_and_crop` source — slice `image[:target_h, :target_w]` after bottom/right
pad — not merely repeated). The "375×1242" figure in the task brief is recorded
as a documentation discrepancy, NOT a blocker: the operative fact (crop-only,
origin unmoved, factor 1.0) holds for the measured sizes.

## 7 DEPTH EQUATION

Actually used (via the IMPORTED `StereoCalibration.depth_from_disparity` —
no second depth formula exists in the new code):

```
Z = fx * B / d          (metres; fB = 380.818520 pixel·m for scenes 000160–000169,
                         fB = 382.669950 pixel·m for scenes 000170–000199)
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy   (fy read from P_rect_02[1,1] by the new module; KITTI fx == fy here)
```

**VERIFIED**: max |imported-method depth − independent fB/d| = 0.000e+00 on
scene 000160_10.png.

## 8 IMPLEMENTATION

New files ONLY under `stage_c_deploy/metric_depth/` (nothing else touched):

| file | role |
|---|---|
| `armp_depth.py` | SHA256-asserted frozen load; inference exactly as `dr1_eval_40scene.py` (same `normalize()`, `net.eval()`, `torch.no_grad()`, `net(tl, tr)`) |
| `metric_depth.py` | disparity→depth via IMPORTED method; `read_fy_from_calib`; XYZ reprojection; stats; NaN+mask invalid handling, no clipping |
| `measurement.py` | pixel measure (PIXEL DEPTH) and odd-window NxN median over valid pixels only (REGION MEDIAN DEPTH) |
| `qcheck.py` | OpenCV-convention Q build + explicit Q matmul + `cv2.reprojectImageTo3D` comparison; tolerance stated BEFORE running (see §12) |
| `visualize.py` | NEW visualization mode: left RGB, disparity, metric depth (colorbar in metres, display 0.5–60 m stated on figure, data unclipped), RGB+depth overlay, invalid transparent |
| `validate_c2.py` | runnable A–I validation; writes `out/c2_validation.json`; exit nonzero on failure |
| `out/` | `c2_validation.json`, `000160_10_c2_viz.png`, `000162_10_c2_viz.png` |

Every non-trivial module has an assert-based `demo()` / `__main__` self-check
(project convention); all pass (**VERIFIED** — §13 transcript).

## 9 INVALID-DATA HANDLING

Rule: d ≤ 1e-3, NaN, +Inf, or non-finite → depth NaN + mask False. No clipping
(**VERIFIED** — synthetic edge vector):

```
in=[10,0,-3,NaN,Inf,1e-9] valid=[[True, False, False, False, False, False]] depths all-NaN except first: True
```

On the 40 real scenes, 38/40 masks are fully valid (453376/453376, 100.00%) —
the frozen net's final ReLU yields strictly positive disparity almost everywhere.
Two scenes exercise the invalid path on real data: 000176_10.png has 6 invalid
pixels and 000199_10.png has 27 invalid pixels (both have disp_min exactly 0.0,
i.e. zero-disparity outputs caught by the d ≤ 1e-3 rule). That is a measured
property of these outputs, NOT a guarantee (**INFERRED**: other scenes could
contain more zeros; the mask path is tested and ready).

## 10 XYZ REPROJECTION

X=(u−cx)Z/fx, Y=(v−cy)Z/fy, Z=Z with fy from the calib file (**VERIFIED** —
exact bitwise match to independent arithmetic at (u=616,v=184), scene 000160:
`d=13.386547 Z=28.447853 X=0.618230 Y=0.098523 exact match … True`).
Coordinate convention: rectified left-camera frame, +X right, +Y down, +Z forward
(**INFERRED** from the KITTI rectified-pinhole model the parser implements).

## 11 KITTI VALIDATION

40/40 hailo_val scenes (000160–000199_10.png). Disparity/depth measured through
the new path. Aggregate stats across all 40 scenes (full per-scene detail in
`out/c2_validation.json` `scenes` array):

| quantity | min | max | mean |
|---|---|---|---|
| depth min (m) | 2.1854 | 6.1792 | 4.2518 |
| depth max (m) | 82.1843 | 12085.5541 | 537.4429 |
| depth mean (m) | 14.2578 | 37.8257 | 24.6350 |
| depth median (m) | 8.9295 | 35.7236 | 17.4832 |
| percent_valid (%) | 99.9940 | 100.0000 | 99.9998 |

38/40 scenes are 100.00% valid (453376/453376); the exceptions are 000176_10.png
(453370/453376, 99.9987%, 6 invalid) and 000199_10.png (453349/453376, 99.9940%,
27 invalid) — both from zero-disparity outputs (disp_min exactly 0.0). The very
large depth-max values (up to 12085.6 m on 000199_10.png) come from valid but
near-zero disparities (~0.03 px); geometrically valid, with huge
disparity-error leverage (see §16). First five scenes for continuity with the
earlier 5-scene record:

| scene | disp min/max/mean (px) | depth min/max/mean/median (m) | valid |
|---|---|---|---|
| 000160_10.png | 4.1496 / 80.0067 / 25.8097 | 4.760 / 91.773 / 23.503 / 19.298 | 453376/453376 (100.00%) |
| 000161_10.png | 3.5933 / 126.1539 / 33.2538 | 3.019 / 105.979 / 21.475 / 18.178 | 100.00% |
| 000162_10.png | 3.6264 / 105.2182 / 26.5231 | 3.619 / 105.012 / 23.086 / 19.541 | 100.00% |
| 000163_10.png | 3.3305 / 72.6336 / 28.2069 | 5.243 / 114.344 / 22.982 / 18.820 | 100.00% |
| 000164_10.png | 3.6946 / 116.3116 / 36.5873 | 3.274 / 103.074 / 23.029 / 15.473 | 100.00% |

(**VERIFIED** — `out/c2_validation.json` `scenes` array, 40 entries.)

Hand-checkable pixel (independently re-verifiable by hand): scene 000162_10.png,
(u=616, v=184): d=10.807845115661621 px, fx=718.3351 px,
B=0.5301404873575021 m, cx=600.3891, cy=181.5122.
fB = 718.3351 × 0.5301404873575021 = 380.818520;
Z = 380.818520 / 10.807845115661621 = 35.235379 m ✓ (matches `pixel_depth_m`
35.235379108843524); X = (616−600.3891)×35.235379/718.3351 = 0.765737 m ✓;
Y = (184−181.5122)×35.235379/718.3351 = 0.122030 m ✓ (**VERIFIED**).

## 12 Q-MATRIX CROSS-CHECK

Tolerance stated BEFORE running (in `qcheck.py`): explicit float64 Q matmul must
match Method A with max abs diff ≤ 1e-6 m for X/Y and ≤ 1e-6 × max(1, Zmax) for Z.
Q used (OpenCV convention, sign verified empirically):

```
Q = [[1,0,0,-cx],[0,1,0,-cy],[0,0,0,fx],[0,0,1/B,0]]   (Q[3,2]=+1/B encodes Z=+fB/d;
  the -1/B spelling found in some references yields Z=-fB/d and is rejected here.)
```

Result: **PASS** on all 40 scenes — explicit-Q vs Method A agrees to ≤ 4.6e-13 m
(worst case dx=4.547e-13, dy=5.684e-14 on 000176_10.png, dz=4.547e-13, both on
000199_10.png except dy) over all valid pixels per scene (453376 compared per
fully-valid scene; 453370 on 000176_10.png, 453349 on 000199_10.png)
(**VERIFIED** — per-scene evidence lines in the §13 transcript and
`out/c2_validation.json`).
Secondary `cv2.reprojectImageTo3D` (float32 path, tolerance 1e-4 m justified by
float32 disparity quantization): PASS on 39/40 scenes (dx,dy,dz ≤ ~6.3e-06 m on
the original 5 scenes; worst over the other 34 passing scenes dx=6.170e-05 m on
000176_10.png). Total cv2 missing-value sentinel count across all 40 scenes:
38 (exactly 1 per scene on 38 scenes; 0 on 000176_10.png and 000199_10.png,
where no global-minimum-disparity sentinel pixel occurs); sentinel pixels are
excluded from the cv2 comparison and counted, not hidden.
One honestly-recorded secondary observation, NOT a check failure: on scene
000199_10.png the secondary cv2 comparison exceeds its informative 1e-4 m
tolerance (dx=1.870e-04, dy=8.718e-05, dz=5.562e-04, sentinel10000=0,
cv2_passed=False), while the PRIMARY explicit-Q criterion on the same scene
passes (dx=4.547e-13, dy=5.684e-14, dz=4.547e-13 against tol_xy=1e-6,
tol_z=1.209e-02) — so the `Q-matrix cross-check 000199_10.png` check itself is
PASS and the run exits 0. Cause (**INFERRED**, consistent with the float32
justification stated in `qcheck.py` before running): 000199_10.png contains
valid near-zero disparities (~0.03 px → Z≈12085 m, hence tol_z=1.209e-02),
where float32 disparity quantization is levered into >1e-4 m metric error via
|dZ| ≈ f·B/d² · |dd|. No tolerance or check was changed; the check definition
(primary = explicit float64 Q, secondary = cv2 informational) is unchanged from
the 5-scene run.
No material disagreement exists between Method A and the explicit-Q cross-check
on any scene, so no STOP was triggered (**VERIFIED**).

## 13 NUMERICAL VALIDATION

- Accuracy preservation: new path vs direct `model()` call max abs disparity
  diff = 0.000e+00 on all 40 scenes — the geometry layer does not alter inference
  (**VERIFIED**).
- Determinism: same scene twice → bitwise-identical disparity on all 40 scenes
  (**VERIFIED**).
- Finiteness: all valid depths finite on all 40 scenes (**VERIFIED**).
- Module self-checks: `metric_depth`, `measurement`, `qcheck`, `visualize`,
  `armp_depth` demos all print `self-check passed` (**VERIFIED** — transcript):

```
metric_depth self-check passed
measurement self-check passed
qcheck self-check passed: dx=1.421e-14 dy=7.105e-15 dz=2.842e-14
visualize self-check passed: .../out/demo_tmp/demo_10_c2_viz.png
armp_depth self-check passed on 000160_10.png: shape=(368, 1232) min=4.1496 max=80.0067 mean=25.8097
```

- Full run: `C2 validation: 135/135 passed; overall=PASS` on CUDA in 29.5 s
  (**VERIFIED** — full per-test transcript, first lines and notable scenes):

```
[PASS] frozen checkpoint unchanged (before) :: sha256(...)=b2f6f5d5...expected=b2f6f5d5...
[PASS] model parameters unchanged :: config=StereoNetConfig(...downsample_levels=3...num_disparities=24...regression_normalize=True...) n_params=397954 device=cuda
[PASS] KITTI hailo_val split has 40 scenes :: len=40 names[0]=000160_10.png
[PASS] disparity units verified :: pad_and_crop has no resize/interp: True; ...
[PASS] resolution scaling verified :: measured original shapes ... {(374, 1238): 10, (376, 1241): 30} ... disparity scale factor=1.0, crop-only, no resize
[PASS] accuracy preservation 000160_10.png (new path vs model()) :: max abs disparity diff=0.000e+00
[PASS] determinism 000160_10.png (bitwise identical) :: identical=True mean=25.809716
[PASS] calibration source verified :: ...000160.txt -> f=718.3351 B=0.5301404873575021 cx=600.3891 cy=181.5122 1238x374
[PASS] KITTI fx == fy for scenes used :: scene 000160_10.png: fx=P[0,0]=718.3351 fy=P[1,1]=718.3351 equal=True
[PASS] depth equation verified (Z=fB/d) :: scene 000160_10.png: max|depth-imported minus fB/d|=0.000e+00, fB=380.818520
[PASS] numerical finiteness :: all 453376 valid depths finite: True
[PASS] XYZ reprojection verified :: pixel (u=616,v=184): d=13.386547 Z=28.447853 X=0.618230 Y=0.098523 exact match to independent arithmetic: True
[PASS] Q-matrix cross-check 000160_10.png :: explicit-Q dx=7.105e-15 dy=7.105e-15 dz=1.421e-14 ... cv2(float32): dx=1.024e-06 dy=1.621e-06 dz=5.803e-06 sentinel10000=1 passed=True
(+ identical accuracy/determinism/Q lines for the remaining 35 scenes, all PASS —
40 accuracy + 40 determinism + 40 Q = 120 per-scene checks; see §12 for the two
scenes whose Q evidence lines differ: 000176_10.png with n=453370 and
sentinel10000=0, and 000199_10.png with n=453349, sentinel10000=0 and the
honestly-recorded secondary cv2 non-pass alongside a passing primary check)
[PASS] metric measurement :: scene=000162_10.png PIXEL (u=616,v=184) d=10.8078px Z=35.2354m X=0.7657 Y=0.1220; REGION5 median Z=35.3435m n=25/25
[PASS] visualization renders (new mode, invalid distinct) :: wrote [...000160_10_c2_viz.png, ...000162_10_c2_viz.png] ...
[PASS] invalid disparity handling :: in=[10,0,-3,NaN,Inf,1e-9] valid=[[True, False, False, False, False, False]] ...
[PASS] frozen checkpoint unchanged (after) :: sha256=b2f6f5d5...
[PASS] ONNX hash matches disk spelling (typo in prompt noted) :: sha256(...)=4277090deed1cde26... authoritative
```

## 14 MEASUREMENT INTERFACE

One REAL measured sample (scene 000162_10.png, u=616, v=184) — PIXEL DEPTH and
REGION MEDIAN DEPTH labelled separately (**VERIFIED**):

```
PIXEL (u=616,v=184): d=10.8078 px, Z=35.2354 m, X=0.7657 m, Y=0.1220 m, valid=True
REGION 5x5 median:   d=10.7748 px, Z=35.3435 m, X=0.7657 m, Y=0.1220 m, n=25/25 valid
```

Note: the region-median X/Y here coincide numerically with the center pixel's X/Y
to printed precision because the median element of this window happens to be the
center column's value (verified by dumping the sorted 5×5 X window — median index
12 = 0.7657373 = center). Genuine property of this window, not an aggregation
bypass: median disparity (10.7748) differs from the center disparity (10.8078),
proving the window was aggregated (**VERIFIED** — see §13-adjacent debug run).

## 15 VISUALIZATION

New mode only; existing diagnostics untouched (**VERIFIED** — no file outside
`stage_c_deploy/metric_depth/` and `C2_METRIC_DEPTH_REPORT.md` was created or
modified by this stage; `git status` shows no other new/modified paths from this
work). Files: `out/000160_10_c2_viz.png`, `out/000162_10_c2_viz.png` — 4 panels
(left RGB / disparity with pixel colorbar and stated 0–176 px display range /
metric depth with metre colorbar and stated 0.5–60 m display range / RGB+depth
overlay), invalid pixels transparent. The disparity display range is fixed at
0–176 px (`DISPLAY_DISP_VMIN_PX=0.0`, `DISPLAY_DISP_VMAX_PX=176.0`) so figures
are comparable by eye (**VERIFIED** — both PNGs use the identical 0–176 px
disparity colorbar). Justification (**VERIFIED** — `out/c2_validation.json`
`disp_min`/`disp_max` over all 40 scenes): measured disparity range is
0.0–175.105 px, so 0–176 px covers the whole split with no clipping of real
data. Display ranges are presentation-only; the underlying data is never
clipped (**VERIFIED** by code:
`vmin/vmax` apply to `imshow`, `masked_depth` only masks invalid; the stored
disparity array is unmodified).

## 16 LIMITATIONS

- Disparity EPE is NOT a depth-accuracy measurement. Depth error scales as
  |dZ| ≈ f·B/d² · |dd|, so relative depth error ≈ relative disparity error;
  at 90 m a 1-px disparity error costs tens of metres (**INFERRED** from the
  verified depth equation; no depth ground truth was evaluated — stated, not measured).
- The laptop webcam is NOT a stereo camera (**NOT APPLICABLE** — no webcam used).
- No Hailo hardware claim is made (**NOT APPLICABLE** — CPU/CUDA run only).
- No real-camera accuracy claim is made (**NOT APPLICABLE** — KITTI only).
- Near-100%-valid masks are a measured property of these 40 outputs (38 fully
  valid; 000176_10.png and 000199_10.png carry 6 and 27 zero-disparity invalids),
  not a guarantee (**INFERRED**).
- Depth values beyond ~100 m from few-pixel disparities are geometrically valid
  but carry huge disparity-error leverage (see error scaling above) (**INFERRED**).

## 17 REAL-CAMERA REQUIREMENTS

How the same geometry layer would accept `(left, right, calibration)` later
(documentation only — no capture implemented, no camera selected):

1. Provide a rectified pair and a `StereoCalibration(focal_px, baseline_m, cx,
   cy, width, height)` — e.g. extend `parse_kitti_cam_to_cam`-style parsing to
   the camera's calibration format, or construct it directly from a standard
   stereo calibration (OpenCV `stereoRectify` outputs map 1:1 onto these fields).
2. Preprocess exactly as validated: NO resizing; if the frame is smaller than the
   network input, zero-pad bottom/right; crop top-left (origin unmoved) or adjust
   cx/cy by the crop offset and scale fx,cx,cy by any resize factor (resize factor
   was 1.0 here — any future resize must re-derive the factor explicitly).
3. Feed rectified pairs through the frozen net, then `disparity_to_depth` +
   `reproject_xyz` unchanged. Re-run `validate_c2.py` on the new rig before any
   accuracy claim. (**INFERRED** — procedure follows from the verified units/scaling
   analysis; no real camera exists in this repo.)

## 18 ARTIFACT INTEGRITY

| artifact | hash before | hash after | verdict |
|---|---|---|---|
| ARM-P checkpoint `p2a_best.pth` | `b2f6f5d5…eb7454` | identical | **PASS** (asserted pre-load AND re-hashed post-run) |
| `armp_stereonet.onnx` | `4277090d…bcb6989` | identical | **PASS** (disk spelling authoritative; prompt typo noted, no file fixed) |

No prohibited path was modified: `src/**`, `phase1/**`, `stage_b_armp/**`,
`stage_c_deploy/dr1_rescale/dr1_eval_40scene.py`, ONNX files, reports, and
`.gitignore` are untouched by this stage (**VERIFIED** — all C2 files live under
`stage_c_deploy/metric_depth/` plus this report; evaluator file unmodified).

## 19 FINAL VERDICT

**PASS.** Frozen disparity → calibrated metric depth → 3D measurement works
end-to-end on all 40 KITTI hailo_val scenes with 135/135 validation checks green,
inference proven bitwise-unaltered on every scene (max abs diff == 0),
explicit-Q cross-check passing to ≤ 4.6e-13 m worst case, one hand-verified
measurement (scene 000162_10.png pixel (616,184):
Z=35.2354 m, X=0.7657 m, Y=0.1220 m; 5×5 median Z=35.3435 m), and all hashes
intact. Two documentation discrepancies recorded (ONNX-hash typo in prompt/
evidence_summary; 375×1242 assumed vs {(374,1238):10,(376,1241):30} measured) —
neither affects integrity or the verdict. One honestly-recorded secondary
observation (NOT a check failure, no STOP): the informational cv2 float32
comparison on 000199_10.png exceeds 1e-4 m (dx=1.870e-04) at valid near-zero
disparities (Z≈12085 m levering float32 quantization); the primary explicit-Q
check on that scene passes at 4.6e-13 m. Nothing is BLOCKED.

Validation table (Status PASS requires pasted evidence above; none BLOCKED):

| Test | Result | Evidence | Status |
|---|---|---|---|
| frozen checkpoint unchanged | PASS | §13/§18 hashes before+after | PASS |
| model parameters unchanged | PASS | config + n_params=397954 | PASS |
| disparity units verified | PASS | no-resize inspection + §13 | PASS |
| calibration source verified | PASS | §4 table, parser reused | PASS |
| resolution scaling verified | PASS | measured shapes + top-left analysis | PASS |
| depth equation verified | PASS | maxdiff 0.000e+00, fB=380.818520 | PASS |
| invalid disparity handling | PASS | edge-vector transcript | PASS |
| XYZ reprojection | PASS | bitwise match at (616,184) | PASS |
| Q-matrix cross-check | PASS | ≤4.6e-13 explicit-Q all 40; cv2 39/40, quirk counted, 000199 secondary noted | PASS |
| numerical finiteness | PASS | all valid depths finite ×40 | PASS |
| determinism | PASS | bitwise-identical ×40 | PASS |
| KITTI scene validation | PASS | §11 table + aggregates, 40 scenes | PASS |
| metric measurement | PASS | §14 PIXEL vs REGION sample | PASS |
