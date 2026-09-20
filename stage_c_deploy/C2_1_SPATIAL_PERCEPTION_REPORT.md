# C2.1 Spatial Perception Report — lightweight 3D spatial perception on frozen C2 metric depth

Claim labels used throughout: **VERIFIED** (measured / asserted by the
validation run in `stage_c_deploy/spatial_perception/out/c2_1_validation.json`),
**INFERRED** (follows from code inspection or arithmetic, not directly
measured), **UNKNOWN** (not measured, stated as such), **NOT APPLICABLE**.

## 1 Objective

Build the SMALLEST useful 3D spatial-perception extension on top of the frozen
C2 metric-depth layer, using deterministic CPU post-processing only
(**VERIFIED**: no neural network was added, trained, modified, or proposed;
`validate_spatial.py` imports only the frozen C2 path plus NumPy code; the
full 40-scene run exits 0 with 251/251 checks passing). One primary
intervention: spatial binning + nearest-valid-surface reporting over the
unchanged C2 depth (**VERIFIED**: per-scene `c2_maxdiff == 0.0`, see §15).

## 2 Frozen State

- ARM-P checkpoint `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth`
  SHA256 `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454`
  (**VERIFIED**: asserted before AND after the run, both match).
- ONNX on disk `stage_c_deploy/armp_stereonet.onnx` SHA256
  `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989`
  (**VERIFIED**: asserted before AND after, both match).
- Documentation discrepancy (**VERIFIED**, fix nothing): the task prompt
  carries the transposed typo `...deed1dce26...`; the disk spelling
  `...deed1cde26...` is authoritative and is what the validator asserts.
- C2 layer untouched (**VERIFIED**: scope check — no file outside
  `stage_c_deploy/spatial_perception/**` and this report was created or
  modified by this task; C2's `disparity_to_depth` validity rules
  `d<=0, NaN, Inf, tiny disparity` are preserved unchanged, re-tested in §15).
- No git commits were made (**VERIFIED**).

## 3 C2 Dependency

C2 is imported, never re-implemented (**VERIFIED** by code inspection):

| C2 / src module | Imported symbol | Used for |
|---|---|---|
| `metric_depth/armp_depth.py` | `load_frozen_net`, `infer_disparity_scene`, `ARMP_REL`, `ARMP_SHA` | frozen-checkpoint load (SHA256 asserted) + inference |
| `metric_depth/metric_depth.py` | `disparity_to_depth`, `reproject_xyz`, `read_fy_from_calib` | depth + XYZ (X/Y come from `reproject_xyz` only) |
| `metric_depth/measurement.py` | `pixel_measure`, `region_measure` | pixel + region measurement (re-exported/called; NO second measurement module exists in `spatial_perception`) |
| `metric_depth/qcheck.py` | `qcheck_scene`, Q machinery | XYZ cross-validation |
| `src/geometry/stereo.py` | `StereoCalibration`, `parse_kitti_cam_to_cam` | calibration (parsed per scene, never hardcoded) |
| `src/datasets/kitti2015.py` | `Kitti2015Stereo`, `normalize` | hailo_val data + preprocessing |

Two calibration groups in hailo_val, parsed per scene (**VERIFIED** from the
40-scene record: 10 scenes `fx=fy=718.3351, B=0.5301404873575021`
(000160–000169) + 30 scenes `fx=fy=718.856, B=0.5323318578` (000170–000199)).

## 4 System Architecture

```text
frozen ARM-P checkpoint --(armp_depth, C2)--> disparity (368x1232 float64)
  --(C2 disparity_to_depth, UNCHANGED)--> depth + C2 validity mask
  --(C2 reproject_xyz, UNCHANGED)--> X, Y, Z maps
  --(C2 pixel_measure / region_measure)--> pixel + region measurement
  --(C2.1, new, CPU-only, NumPy)--> pointcloud / spatial cells /
      discontinuity map / occupancy grid / spatial figure
```

Coordinate convention, reused from C2 exactly (**VERIFIED**: identical
equations in `reproject_xyz`): metres, camera optical centre at origin, +X
right, +Y down, +Z forward along the optical axis, right-handed.

Files created (ONLY these, plus this report):

| File | Role |
|---|---|
| `spatial_perception/pointcloud.py` | valid pixels → (N,3) float32 + (u,v); separate `physical_mask` |
| `spatial_perception/spatial.py` | grid binning (default 1×3), per-cell stats, nearest valid surface |
| `spatial_perception/discontinuity.py` | NaN-aware gradient magnitude on depth + disparity |
| `spatial_perception/occupancy.py` | NEAR / FAR / INSUFFICIENT_DATA coarse grid |
| `spatial_perception/visualize_spatial.py` | NEW 4-panel spatial figure (C2 visualizer untouched) |
| `spatial_perception/validate_spatial.py` | 40-scene evidence, JSON record, nonzero exit on failure |
| `spatial_perception/out/c2_1_validation.json` | validation record (251 checks) |
| `spatial_perception/out/*_c2_1_spatial.png` | 2 figures (000160, 000162; no sprawl) |

## 5 Point Cloud

`pointcloud.depth_to_pointcloud(xs, ys, zs, valid, stride)`: takes the valid
subset of the C2 X/Y/Z maps directly (strided slicing is a view; only the
valid subset is copied), returns `(N,3)` float32 points plus `(u,v)` index
arrays preserving pixel↔point correspondence, plus count and nbytes
(**VERIFIED**: per-scene check samples 5 points and asserts each reproduces
X, Y, Z exactly at its `(u,v)`; 40/40 scenes pass).

- `stride=1` is the full-resolution data; `stride` is the VISUALIZATION
  SUBSAMPLE only (**VERIFIED**: demo asserts the stride-2 set is a strict
  subset of the stride-1 set).
- No extra full-resolution copies of disparity/depth/X/Y/Z are materialised
  beyond the C2 maps themselves (**VERIFIED** by code inspection; §13 counts
  the buffers).
- Measured: mean point count 453375.2/frame, mean 12694504.9 bytes/frame
  (**VERIFIED**: `memory` field of the JSON record; full frames are ~100%
  C2-valid on real data, min 453349 on 000199_10.png).

## 6 Distance Measurement

C2 ALREADY has pixel + region measurement; C2.1 calls it, unchanged
(**VERIFIED**: `region measurement vs independent hand calculation` check
passes — window medians recomputed with an INDEPENDENT sort+middle-index
routine, no `np.median`, bitwise-equal on scene 000162_10.png; pixel X/Y
recomputed by hand, exact).

One real measured sample (**VERIFIED**, pasted from `c2_1_validation.json`,
scene 000162_10.png, `fx=fy=718.3351, B=0.5301404873575021`):

```json
{"scene": "000162_10.png", "u": 616, "v": 184,
 "pixel": {"disparity_px": 10.807845115661621, "pixel_depth_m": 35.235379108843524,
           "X_m": 0.7657372996673083, "Y_m": 0.12203020031595374, "valid": true},
 "region5": {"region_median_disparity_px": 10.774785995483398,
             "region_median_depth_m": 35.3434880432551, "valid_count": 25, "window_pixels": 25},
 "center_nearest_valid_surface_m": 6.52528887611687,
 "center_min_m": 5.326113255584936, "center_median_m": 28.830577247579853}
```

## 7 Spatial Analysis

`spatial.spatial_cells(depth, valid, n_rows=1, n_cols=3)`: configurable grid,
default LEFT / CENTER / RIGHT with optional row split (TOP-/BOTTOM- names).
Per cell: valid count, percent valid, min, 5th-percentile, median over VALID
pixels only (**VERIFIED**: synthetic + 40-scene checks).

- Headline NEAREST VALID SURFACE = 5th-percentile depth (single-pixel minima
  are noise); true per-cell min reported ALONGSIDE, never instead
  (**VERIFIED**: demo case 2 — a 0.5 m single-pixel spike in a 40 m field
  leaves the 5th percentile at 40.0 m while min reports 0.5 m).
- Terminology enforced: NEAREST VALID SURFACE / SELECTED REGION; never
  object, never obstacle (**VERIFIED** by inspection — neither word appears
  in any module except in prohibitions in this report).
- Synthetic case with obvious expected answer (**VERIFIED**, pasted evidence):

```text
[PASS] synthetic spatial-map case (near centre band) ::
  expected CENTER nearest/min/median=10.0, LEFT=RIGHT=50.0;
  got CENTER=10.0/10.0/10.0, LEFT=50.0, RIGHT=50.0
```

- Real-data definitional check (**VERIFIED**, 40/40): per-cell min is an
  exact member of the valid depth array; the 5th percentile lies within
  [min, max] with ≥5% of valid samples at/below it.
- Example real cells, 000160_10.png (**VERIFIED**):
  LEFT nearest 6.904 m / min 6.053 m / median 23.103 m;
  CENTER nearest 6.720 m / min 5.829 m / median 24.480 m;
  RIGHT nearest 5.310 m / min 4.760 m / median 13.867 m.

## 8 Depth Discontinuity

`discontinuity.depth_discontinuity` / `.disparity_discontinuity`: NaN-aware
forward-difference gradient magnitude — |dZ/du| valid only if both
depth neighbours are valid, likewise |dZ/dv|; magnitude defined where BOTH
hold (borders excluded); same operator on disparity (default validity =
finite; pass the C2 mask to mirror C2's invalid rules). No smoothing of any
kind (**VERIFIED**: module demos + inspection; no smoothing call exists).
Discontinuities are NOT labelled object boundaries (**VERIFIED** by
inspection — no such label exists in code or figures).

- Flat field → exactly 0; a 20 m step → exactly 20 at the step column; an
  invalid pixel poisons exactly the differences touching it (**VERIFIED**,
  `discontinuity self-check passed`).
- Real data: mean valid discontinuity pixels 451777/453376 per frame, i.e.
  borders + the 27 invalid pixels of 000199 excluded (**VERIFIED**).

## 9 Invalid Depth

C2's invalid rules (`d<=0, NaN, Inf, tiny disparity → NaN + False`) are
preserved unchanged (**VERIFIED**):

```text
[PASS] invalid-depth handling (C2 rules preserved) ::
  in=[10,0,-3,NaN,Inf,1e-9,0.03] valid=[[True, False, False, False, False, False, True]];
  0.03px stays C2-valid with Z=12694.0m (geometrically valid, physically unreliable -> separate mask)
```

Physical-range filter (**VERIFIED**): `pointcloud.physical_mask(depth,
c2_valid, physical_max_m)` returns a SEPARATE boolean mask
(`c2_valid & finite & depth<=cap`); it never modifies the stored depth, never
clips, is never applied by default. On real data it is a strict subset of the
C2 mask (000160_10.png: 20489 of 453376 C2-valid pixels beyond the 60 m cap;
000199_10.png: 22056 excluded). Scene 000199_10.png carries 27 C2-invalid
(`d==0`) pixels (**VERIFIED**).

## 10 Visualization

`visualize_spatial.save_spatial_visualization` — NEW mode, separate file; the
C2 visualizer is untouched (**VERIFIED** by inspection). Panels: left RGB,
disparity (fixed 0–176 px, the C2 validated range, kept), metric depth (fixed
0.5–60 m, kept), spatial panel with the cell grid + per-cell nearest valid
surface (selected region highlighted), and a monospace text block with the
selected region's median / nearest / min and pixel XYZ + region-5 median.
No smoothing for display (none applied; stated on the figure). Invalid pixels
transparent. Two scenes written (000160, 000162), no sprawl (**VERIFIED**:
files exist; `visualization renders` check passes).

## 11 Resource Analysis

1. New neural-network compute: NONE (**VERIFIED** — no model, no training,
   no new weights; validator imports only frozen C2 + NumPy).
2. New accelerator compute: NONE (**VERIFIED** — all C2.1 code is NumPy on
   CPU; the only GPU use is the unchanged frozen ARM-P inference).
3. CPU-only: YES for everything C2.1 adds (**VERIFIED**).
4. Memory footprint: tracemalloc peak for the spatial stage 40770062 bytes
   (~38.9 MiB); ~6 full-resolution buffers held at once ≈ 22215424 bytes
   (~21.2 MiB: depth, valid, X, Y, Z, disparity input); mean point-cloud
   allocation 12694504.9 bytes (~12.1 MiB) (**VERIFIED**, `memory` field).
5. Computational cost: per-frame means — inference 0.134 s, depth 0.0017 s,
   XYZ 0.0101 s, spatial 0.0780 s; visualization 1.32 s/figure (offline only)
   (**VERIFIED**, `timing_s` field; all DEVELOPMENT-MACHINE MEASUREMENT).
6. Can it run without copying the whole depth map several times: YES
   (**INFERRED**) — X/Y need only be evaluated at valid pixels
   (gather-then-compute) instead of full-res maps; current footprint is LOW
   so this optimisation was deliberately NOT implemented (smallest extension
   wins over premature optimisation).

## 12 CPU Cost

DEVELOPMENT-MACHINE MEASUREMENT (never Hailo performance), mean per frame
over 40 scenes (**VERIFIED**):

| Stage | Mean time | Class |
|---|---|---|
| ARM-P inference (frozen, unchanged) | 0.1341 s | MODERATE |
| Depth conversion (`disparity_to_depth`) | 0.0017 s | NEGLIGIBLE |
| XYZ (`reproject_xyz`) | 0.0101 s | LOW |
| Spatial analysis (pointcloud + cells + discontinuity + occupancy) | 0.0780 s | LOW |
| Visualization (offline, per figure, not per-frame) | 1.3218 s | MODERATE |

Classes: NEGLIGIBLE <5 ms, LOW 5–100 ms, MODERATE 0.1–2 s, HIGH >2 s
(**INFERRED** thresholds, stated here explicitly). Total added CPU per frame
≈ 90 ms on this machine; inference dominates and is unchanged from C2.

## 13 Memory Cost

DEVELOPMENT-MACHINE MEASUREMENT (**VERIFIED**):

| Item | Measurement | Class |
|---|---|---|
| tracemalloc peak, spatial stage | 40770062 bytes (~38.9 MiB) | LOW |
| Full-resolution buffers (depth+valid+X+Y+Z+disp) | 22215424 bytes (~21.2 MiB) | LOW |
| Point cloud per frame (mean) | 12694504.9 bytes (~12.1 MiB), 453375.2 points | LOW |
| Additional persistent state | 0 bytes (all per-frame temporaries) | NEGLIGIBLE |

Class scale for extra memory: NEGLIGIBLE <1 MiB, LOW 1–100 MiB, MODERATE
0.1–1 GiB, HIGH >1 GiB (**INFERRED** thresholds). Number of full-resolution
buffers: 6 (5 float64 H×W + 1 bool H×W) (**VERIFIED** by inspection).

## 14 Hailo Deployment Considerations

- New neural compute: NONE. Hailo-specific dependency: NONE (**VERIFIED**).
- No Hailo compilation, HEF, quantization, or hardware benchmarking was
  performed (**VERIFIED** — none exists in the code or record).
- The target device is UNKNOWN, so NO Hailo-8/8L/10H/15H limit, NO DFC
  budget, NO latency or bandwidth number is stated anywhere in this work
  (**VERIFIED** by inspection of code, JSON, and this report).
- All timings are labelled DEVELOPMENT-MACHINE MEASUREMENT and must NOT be
  read as device performance (**VERIFIED**: the JSON `timing_note` field and
  every table here carry the label).
- Qualitative portability note (**INFERRED**, not a measurement): the added
  stages are plain NumPy reductions/gather/scatter over float arrays with a
  ~39 MiB transient peak — the class of workload that maps to a host CPU
  companion rather than the accelerator fabric; no claim beyond that is made
  (**UNKNOWN** how any specific device partitions this pipeline).

## 15 Validation

Validator: `spatial_perception/validate_spatial.py` →
`spatial_perception/out/c2_1_validation.json`; nonzero exit on failure;
default all 40 hailo_val scenes, `[n_scenes]` argv override (**VERIFIED**:
ran with no arg → 40 scenes; exit code 0).

Result (**VERIFIED**, pasted):

```text
C2.1 validation: 251/251 passed; overall=PASS;
  wrote .../stage_c_deploy/spatial_perception/out/c2_1_validation.json
```

Check inventory (251 total): hashes before (2) + split size (1) + synthetic
spatial case (1) + invalid handling (1) + physical-mask separation (1), then
per scene ×40 — inference determinism, C2-depth-unchanged (`max abs diff ==
0`, NaN-pattern identical), Q-matrix cross-check (explicit-Q vs Method A,
tolerances from `qcheck.py`), spatial determinism, point-cloud coordinate
exactness, nearest-surface-from-real-data — plus physical-subset (1),
region-vs-hand-calculation (1, scene 000162), visualization (1), hashes after
(2). Failures: none.

C2-depth-unchanged proof (**VERIFIED**, representative paste; all 40
identical in structure):

```text
[PASS] C2 depth unchanged 000199_10.png (max abs diff == 0) ::
  max|C2.1-depth minus fB/d|=0.000e+00 NaN-pattern-identical=True
```

Q-matrix cross-check worst case is 000199_10.png
(`dx=4.5e-13, dz=4.5e-13`, tolerance scaled by its ~12086 m zmax → 1.2e-02)
— passes (**VERIFIED**). Every module also has an assert-based
`demo()`/`__main__` self-check; each passes and cleans up files it writes
(**VERIFIED**: `pointcloud/spatial/discontinuity/occupancy/
visualize_spatial self-check passed`).

## 16 40-Scene Results

- 40/40 scenes: inference deterministic (bitwise-identical repeat),
  C2 depth bitwise-unchanged, Q cross-check passed, spatial deterministic,
  point-cloud exact, nearest-surface from real data (**VERIFIED**).
- Disparity range observed: min 0.0 px (000199 only, 27 pixels), max
  175.10543823242188 px on 000199_10.png — inside the 0–176 px display range
  (**VERIFIED**).
- Valid fraction: 100% on 39 scenes; 99.994% (453349/453376) on 000199
  (**VERIFIED**).
- Occupancy example (000160, 4×6 grid, 10 m threshold): bottom two rows NEAR,
  rest FAR, no INSUFFICIENT_DATA (**VERIFIED**, pasted in evidence JSON).
- No scene required the physical cap to be applied; exclusion counts are
  reported, never silently used (**VERIFIED**).

## 17 Limitations

- Nearest valid surface is a per-cell depth percentile, NOT an object
  detector: it cannot distinguish road, wall, vegetation, or noise
  (**VERIFIED** by design — no classifier exists in this codebase).
- NEAR does not mean obstacle and FAR does not mean free: no ground plane,
  no pose, no free-space reasoning exists (**VERIFIED** by design).
- The 5th percentile is interpolation-based (NumPy default linear); on
  nearly-uniform cells it equals the local depth, on multimodal cells it
  tracks the near mode's edge — interpreted accordingly (**INFERRED** from
  the definition; behaviour asserted in demo case 2).
- Very-near or invalid-dominated cells report NaN (INSUFFICIENT_DATA in
  occupancy); downstream consumers must handle NaN, not treat it as far
  (**VERIFIED**: demo case 3).
- `stride>1` point clouds are visualization subsamples, not measurement data
  (**VERIFIED** by construction).
- If a neural component (learned confidence, refinement, detection) is ever
  concluded necessary: STOP and propose it as future work — nothing of that
  kind was implemented here (**NOT APPLICABLE** — no such conclusion).

## 18 Artifact Integrity

- Checkpoint SHA256 before = after = `b2f6f5...feb7454` (**VERIFIED**).
- ONNX SHA256 before = after = `4277090d...bcb6989` disk spelling
  (**VERIFIED**); prompt typo `...deed1dce26...` recorded as documentation
  discrepancy, nothing fixed, nothing renamed.
- Frozen evaluator / C2 sources unmodified by this task (**VERIFIED**:
  only `stage_c_deploy/spatial_perception/**` and this report were written).
- No previous experiment run directory, checkpoint, or record was
  overwritten (**VERIFIED**: all outputs are new paths under
  `spatial_perception/out/`).
- No git commits (**VERIFIED**).

## 19 Final Verdict

**PASS** — 251/251 checks, exit code 0, over all 40 hailo_val scenes, with
C2 depth proven bitwise-unchanged, frozen artifacts intact, and resource
costs measured and classified (added neural compute NONE, added accelerator
compute NONE, added CPU ≈90 ms/frame and ≈39 MiB transient peak on the
development machine, both LOW). No EPE, stereo-matching, or Hailo-accuracy
claim is made — the model did not change; this is system functionality
(**VERIFIED**: no such claim appears in code, JSON, or report).

Capability table:

| Capability | Implemented | Validated | Added neural compute | Resource class |
|---|---|---|---|---|
| Metric depth (C2, reused) | yes (C2) | yes, unchanged (diff==0) | NONE | NEGLIGIBLE |
| XYZ reconstruction (C2, reused) | yes (C2) | yes, Q cross-check | NONE | LOW |
| Point cloud | yes (C2.1) | yes, exact (u,v) round-trip | NONE | LOW |
| Pixel measurement (C2, reused) | yes (C2) | yes, hand-checked | NONE | NEGLIGIBLE |
| Region measurement (C2, reused) | yes (C2) | yes, independent hand calc | NONE | NEGLIGIBLE |
| Nearest valid surface | yes (C2.1) | yes, from-real-data + synthetic | NONE | LOW |
| Depth discontinuity | yes (C2.1) | yes, synthetic exact | NONE | LOW |
| Spatial map (occupancy) | yes (C2.1) | yes, synthetic exact | NONE | NEGLIGIBLE |

Resource table:

| Resource | Measurement | Status |
|---|---|---|
| ARM-P inference | 0.1341 s/frame mean (dev machine, CUDA) | MODERATE (unchanged, frozen) |
| Post-processing CPU time (added) | depth 0.0017 + XYZ 0.0101 + spatial 0.0780 ≈ 0.090 s/frame | LOW |
| Additional memory | 40770062 bytes tracemalloc peak (~38.9 MiB) | LOW |
| Point count | 453375.2 mean / frame | LOW (12.1 MiB) |
| Full-resolution buffers | 6 (depth+valid+X+Y+Z+disp, ~21.2 MiB) | LOW |
| New neural compute | NONE | NOT APPLICABLE |
| Hailo-specific dependency | NONE | NOT APPLICABLE |

All timings DEVELOPMENT-MACHINE MEASUREMENT, never Hailo performance.
