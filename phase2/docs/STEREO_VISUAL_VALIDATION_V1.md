# Stereo Depth Visualizer V1 — visual validation of the H1-v2 models

**Status:** tooling, not an experiment. It adds no record to
`phase2/experiments/`, changes no checkpoint, and touches nothing in Phase 1
(frozen at `phase-1-frozen` = `b4207e5`). A regression test asserts exactly
that: `test_saving_a_report_touches_nothing_outside_the_output_directory`
compares mtimes of every file under `src/`, `phase2/experiments/` and
`phase2/results/training/` across a full report generation.

## 1. Purpose

`EPE = 14.4 px` and `D1 = 81 %` are true and nearly uninterpretable. This tool
answers a different question:

> If I put a stereo pair into this network, what does it actually believe the
> depth structure of the scene is?

and, specifically for H1: **where** the ~0.6 pt D1 gap between `BASE-v2` and
`WORKING-v2` lives on the image.

It is an observation instrument. It does not decide anything about H1.

## 2. Files

| Path | Role |
|---|---|
| `phase2/viz/core.py` | scene loading, checkpoint loading + prediction cache, depth conversion, pixel probe, region statistics, point cloud, metadata |
| `phase2/viz/render.py` | colour maps with honest masking, panels, per-scene artefact set |
| `phase2/scripts/stereo_visualizer.py` | CLI and the interactive matplotlib viewer |
| `phase2/tests/test_visualizer.py` | 25 tests: geometry, invalid handling, correspondence, protocol, write-safety |
| `phase2/scripts/mentor_demo.py` | builds the mentor demonstration package (section 12) |
| `phase2/tests/test_mentor_demo.py` | 7 tests for the demo builder's claim discipline |
| `phase2/visualizations/` | output (git-ignored except `scene_ranking_*.json`) |

Dependencies: only what Phase 1 already pins — numpy, torch, matplotlib,
opencv (via the dataset loader). No new dependency, no web framework.

## 3. Data flow

```
KITTI 2015 training/{image_2,image_3,disp_occ_0,calib_cam_to_cam}
        |  src/datasets/kitti2015  (pad/crop 368x1232 top-left, /256 GT)
        v
   core.Scene  ── left, right (uint8) ── gt_disparity (0 = no GT) ── calibration
        |
        |  normalize() -> NCHW, ImageNet stats on 0-255 RGB
        v
   StereoNet(cost_volume_shift = none | left)  <- EXP-H1-*-v2 checkpoint
        v
   core.Prediction.disparity  (H x W float32, disparity pixels)
        |
        +-- Z = fB/d  ------------------> metric depth (where d > 1e-3 and finite)
        +-- |pred - gt| where gt > 0 ---> error
        +-- x_r = x_l - d --------------> correspondence
        +-- region / pixel statistics
        v
   render.*  -> PNG + metadata.json
```

Inference runs **once** per (model, scene) and is cached in memory; every probe,
region and mode switch reads the cached array.

## 4. Preprocessing and coordinate consistency

The tool reuses `src/datasets/kitti2015` unchanged, which is what
`phase2/scripts/exp_h1_cost_volume.py::validate` uses. Consequences worth
stating because they are the usual source of silent misalignment:

- Images and ground truth undergo the **same** pad/crop, so a pixel index means
  the same thing in `left`, `right`, `gt_disparity` and the prediction.
- The crop is anchored top-left and there is **no resize**, so the KITTI
  rectified intrinsics (`f`, `cx`, `cy`) apply to cropped coordinates unchanged
  — no focal-length rescaling is needed or performed. If preprocessing ever
  starts resizing, `Scene.calibration_note` and this section must change with
  it.
- Ground truth is `disp_occ_0 / 256.0` (KITTI's own scale, the dataset default
  the H1 runs used — *not* Hailo's 255.0 variant, which is 1.0039x larger).
- Training used random 256x512 crops; **evaluation and this tool use the full
  368x1232 frame**, matching the recorded validation protocol.

## 5. Disparity, depth and correspondence mathematics

**Depth.** `Z = f*B/d`, with `f` and `B` read from that scene's
`calib_cam_to_cam` file via `src/geometry/stereo.parse_kitti_cam_to_cam`
(`B = |tx_right - tx_left| / f` from the rectified projection matrices).
Nothing is hard-coded. For `hailo_val` the values are around `f ≈ 718 px`,
`B ≈ 0.53 m`, `fB ≈ 381 px·m`.

**Invalid disparity.** A pixel is converted only when `d > 1e-3` **and** `d` is
finite. Everything else is `NaN` plus an explicit invalid mask, never a clamped
"very large" depth. The finiteness half of that guard lives in
`phase2/viz/core.depth_from_disparity`, because Phase 1's frozen
`StereoCalibration.depth_from_disparity` accepts `+inf` and returns exactly
`0 m` — a fabricated measurement. The frozen baseline is not edited to fix
that; it is guarded at this tool's boundary, and a test pins the behaviour.

**Correspondence.** Left-referenced: `x_right = x_left - d`. This is verified,
not assumed. `test_disparity_sign_convention_holds_on_kitti` warps left pixels
by the ground-truth disparity in both directions and compares intensities:
`x_l - d` gives mean |ΔI| ≈ 7.6 against ≈ 35.5 for `x_l + d` on scene 0. A
synthetic-shift test pins the arithmetic itself, including that the wrong sign
does *not* match.

**Ground-truth depth** exists only where GT disparity exists (~20 % of pixels
on KITTI). Depth error is reported only where prediction and GT are both valid.

## 6. Visualization modes

| Mode | What it shows | Honesty rules applied |
|---|---|---|
| RGB | left / right images | the left image is the reference frame for every overlay |
| Disparity | predicted disparity, px | shared colour scale with GT so the two are comparable; valid range printed |
| Depth | `fB/d`, metres | invalid disparity drawn grey, never as a depth |
| Depth overlay | depth painted on the left image | invalid is transparent (RGB shows through), colour bar in metres |
| Error | `|pred − gt|`, px | pixels without GT are grey and labelled "no ground truth, not zero error" |
| BASE vs WORKING | both arms, same scales | error panels share one colour range so the comparison is visual, not per-panel-normalised |
| Difference | `WORKING − BASE` disparity, and `|err_WORKING| − |err_BASE|` | symmetric diverging scale, titled "a difference, not an improvement" |
| Pixel probe | disparity / depth / errors / `x_right` at one pixel | anything undefined prints `N/A`; nothing non-finite is ever formatted as a number |
| Correspondence | selected left pixel and its predicted match on the right image | predicted match and GT match drawn separately, plus the `x_l` column for reference |
| Region probe | median / mean / p10 / p90 depth, valid-pixel counts, MAE, median AE, D1 in the box | median first — a box straddling a depth boundary is bimodal and its mean names a distance nothing is at |
| Point cloud | back-projection with the real intrinsics | only where calibration exists; points past 80 m dropped rather than drawn at an invented distance |

Every map states its valid data range, its valid pixel count, and — when the
display range clips — how many pixels fell outside it.

## 7. Commands

```bash
# list the 40 validation scenes
python phase2/scripts/stereo_visualizer.py --list

# full artefact set for one scene, both arms
python phase2/scripts/stereo_visualizer.py --scene 0

# several scenes plus a back-projected point cloud
python phase2/scripts/stereo_visualizer.py --scenes 27 0 31 6 --point-cloud 6

# numbers only, nothing written
python phase2/scripts/stereo_visualizer.py --scene 0 --probe 742,191 \
    --region 600,150,900,300 --no-save

# rank every scene by the BASE/WORKING D1 gap, to choose what to look at
python phase2/scripts/stereo_visualizer.py --rank

# interactive: click a pixel, drag a region
python phase2/scripts/stereo_visualizer.py --scene 27 --interactive
#   m mode | n/p scene | b model | c correspondence | 3 point cloud | s save | q quit

python -m pytest phase2/tests -q          # 106 passed (whole Phase 2 suite)
```

Per-scene output (`phase2/visualizations/<scene>/`): `rgb_left.png`,
`rgb_right.png`, `disparity_gt.png`, `depth_gt.png`, `disparity_{base,working}.png`,
`error_{base,working}.png`, `depth_{base,working}.png`,
`depth_overlay_{base,working}.png`, `depth_error_{base,working}.png`,
`difference_disparity.png`, `difference_error.png`, `comparison.png`,
`metadata.json`, plus `correspondence_*.png` / `probe_*.json` when probed.

## 8. Metrics: three levels, kept apart

1. **Recorded global** — read from `phase2/experiments/EXP-H1-*-v2/metrics.json`
   and **not recomputed**. Protocol: first 10 scenes of `hailo_val`, pooled over
   `gt > 0`, every 5 epochs during training. BASE 14.330 px / 81.19 %,
   WORKING 14.340 px / 80.74 % at epoch 199.
2. **Per-scene** — computed here, one scene at a time, pooled over that scene's
   `gt > 0` pixels, full frame. A **new** computation with a **different**
   population from (1); the two are not comparable and the tool says so on every
   printout and in `scene_ranking_*.json`.
3. **Per-region / per-pixel** — computed here, stated as such.

## 9. Reproducibility

Every report writes `metadata.json`: scene name/index/split, git revision,
preprocessing description, calibration source and values, disparity convention,
per-model checkpoint path + **SHA-256** + device + precision + input size + the
training config stored in the checkpoint, per-scene metrics, and the recorded
global metrics with their source path. Re-running the same command at the same
revision reproduces the images; the SHA-256 is what proves the same weights
were used.

## 10. What this tool observed (H1) — observations only

Per-scene ranking over all 40 `hailo_val` scenes
(`phase2/visualizations/scene_ranking_hailo_val.json`):

- WORKING-v2 has the lower D1 in **27 of 40** scenes; mean ΔD1 −0.77 pt, median
  −1.01 pt. Direction matches the recorded late-window result; magnitude is a
  different protocol and is not a restatement of it.
- The spread is wide and two-sided: scene 27 (`000187_10`) favours WORKING by
  7.54 pt, scene 6 (`000166_10`) favours BASE by 6.19 pt. **In this per-scene
  view the effect is not uniform across scenes.**
- Example scenes generated for inspection: 27 (largest WORKING margin),
  0 (typical WORKING margin), 31 (near-tie, ΔD1 −0.20), 6 (largest BASE margin).

Nothing above establishes a mechanism. Reading the difference maps to name one
is the next task, not this one's conclusion.

## 11. Limitations and known issues

- **Both arms are badly under-trained** (160 scenes, random init; EPE ≈ 14 px
  on the recorded protocol). The pictures show what an under-trained model
  believes, which is the point, but no visual impression here transfers to a
  competently trained StereoNet.
- **Ground truth is sparse** (~20 % of pixels, LiDAR-projected) and drawn as
  individual pixels, so `disparity_gt.png` looks like noise at page scale. That
  is the data, not a rendering fault; zoom or probe instead.
- **No automatic failure-case categorisation.** Textureless, repetitive,
  occluded, thin-structure and far-range regions must be chosen by eye —
  `--rank` plus manual scene selection is the provided route. Labelling a scene
  "occlusion failure" without evidence would be a claim, not an observation.
- **No object detector**, so no object labels: a selected region is a depth
  region, never a detection.
- The interactive viewer needs a GUI backend (TkAgg). Headless machines get the
  CLI, which is fully featured except for clicking.
- The point cloud is a matplotlib scatter, subsampled and capped at 80 m.
  Anything more is a 3D viewer, which V1 deliberately is not.
- `--rank` runs 40 scenes x 2 models (~1 min on an RTX 4060, longer on CPU) and
  is not cached across invocations.
- Display ranges use 2nd/98th percentiles of the valid data. This is stated on
  every panel along with the number of pixels outside it, but it does mean the
  colour extremes are not the data extremes.

## 12. Mentor demonstration package (2026-09-08) — additive, no V1 behaviour changed

`phase2/scripts/mentor_demo.py` builds `phase2/demo/` for a non-technical
reader. It is a *consumer* of the V1 primitives above, not a change to them.
The only edit to V1 itself was one option on `render.show_overlay`
(`colorbar=False`, returning the image handle) so a two-panel figure can share
one colour bar and keep both panels the same width. Default behaviour is
unchanged and `test_show_overlay_colorbar_flag_controls_the_extra_axes` pins
both branches.

**Model.** `H2` (`EXP-H2-SOFTARGMIN-SCALE`), not the H1-v2 arms that are still
the CLI's default `--models`. Nothing about the demo touches H1's records.

**Scene selection, stated rather than smuggled in.** All 40 `hailo_val` scenes
are scored by this model's per-scene D1 (the `--rank` computation, *not* the
recorded 10-scene protocol) and the full table goes into
`phase2/demo/manifest.json`. The three demonstration scenes are ranks 3, 9 and
12 of 40; the single worst scene, rank 40, is included as a disclosed failure
case and its figures are labelled `FAILURE CASE`. Every figure prints its own
rank.

**Probe grid.** Four boxes, fixed in `mentor_demo.PROBE_BOXES`, identical for
every scene — no per-scene box placement, so a favourable number cannot be
obtained by moving a box. `test_probe_boxes_are_one_fixed_grid_inside_the_frame`
pins that.

**Stereo dependence figure.** Reuses Phase 1's own corruption set
(`scripts/exp_right_image_ablation.variants`, variant
`right_from_other_scene`) rather than inventing a demo-only corruption, so the
picture shows the same manipulation the experiments measured. Labelled a
diagnostic on the figure itself, never a normal operating mode.

**Observations from this package (observations only, one model, four scenes):**

- Demonstration scenes: per-scene EPE 1.27–1.32 px, D1 7.7–10.3 %. Overlays
  show recognisable road/foreground/background structure.
- Stereo dependence, scene `000191_10`: replacing the right image takes D1 from
  7.69 % to 98.91 % (+91.22 pt) and EPE from 1.294 to 33.253 px, mean |Δd|
  38.34 px.
- Failure case `000161_10`: GT disparity reaches 129.5 px there; this model's
  output never exceeds 80.6 px on any of the four scenes. 15,978 GT pixels
  exceed 100 px and carry MAE 55.5 px; over the remaining GT pixels MAE is
  4.21 px. The overlay still looks plausible — visual plausibility is not
  accuracy, which is why the error maps ship alongside it.

Outputs are git-ignored PNGs plus a committed `README.md` and `manifest.json`,
the same rule as `phase2/visualizations/`.

## 13. Desktop shortcut and the one-knob update path (2026-09-08)

`Stereo Depth Demo.lnk` on the desktop launches the interactive viewer on the
current demonstration model:

```
target : C:\Users\vishn\.pyenv\pyenv-win\versions\3.12.9\python.exe
args   : "phase2\scripts\stereo_visualizer.py" --demo --scene 31 --interactive
cwd    : C:\Users\vishn\stereo_depth_vision
```

**The shortcut names no model.** `--demo` resolves to `core.DEMO_MODEL` at
launch, so the shortcut never has to be edited, re-created or repaired when a
better model arrives.

### When the model improves, edit one file

`phase2/viz/core.py`:

1. add the new checkpoint to `MODELS` (key, `experiment`, `cost_volume_shift`,
   and `regression` if it uses the standardised stage) — the same shape as the
   existing entries, so the loader keeps cross-checking the recorded config
   against what it builds;
2. repoint `DEMO_MODEL` at the new key.

Then `python phase2/scripts/mentor_demo.py` to regenerate `phase2/demo/`.

That is the whole procedure. The demo builder, the `--demo` flag and the
desktop shortcut all follow `DEMO_MODEL`; nothing else names a model.
`test_everything_follows_the_one_demo_model_knob` deliberately does **not**
hardcode `"H2"` — it asserts the knob points at a registered model and that the
builder follows it, so it keeps passing across a model change instead of
becoming the thing that blocks one.

**Two viewer defects fixed while wiring this up**, both pre-existing and both
hit by the single-model shortcut path:

- the viewer figure was a fixed 15x8 in = 1500x800 px, which overflows a
  1280x800 panel and clipped the readout strip — the numbers — off the bottom.
  It has to be sized at figure construction: matplotlib restores the window to
  the figure size on draw, so a later `wm_geometry` or `wm_state("zoomed")` is
  silently undone. Now queries the screen and shrinks to fit, never grows.
- `difference` mode was in the `m` cycle unconditionally, so with one model
  loaded it was a dead stop printing "difference needs two models loaded". The
  mode list is now built from the models actually loaded.
