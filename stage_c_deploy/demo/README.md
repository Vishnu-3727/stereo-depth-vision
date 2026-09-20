# Pipeline demo — what the deployment candidate actually produces

```
python stage_c_deploy/demo/pipeline_demo.py              # scene 12, interactive window
python stage_c_deploy/demo/pipeline_demo.py --scene 31
python stage_c_deploy/demo/pipeline_demo.py --list       # the 40 hailo_val scenes
python stage_c_deploy/demo/pipeline_demo.py --no-window --save demo.png
```

Desktop shortcut: **Stereo Depth Demo - Pipeline**. (The older
**Stereo Depth Demo** shortcut still runs `phase2/scripts/stereo_visualizer.py`,
the research viewer, and is left untouched.)

Needs: the frozen ARM-P checkpoint (in this repo) and `data/kitti2015/`
(download separately — see the README's "Reproducing this work").

## What it shows

One real KITTI stereo pair taken all the way through the deployment graph:

| Panel | Stage | Module |
|---|---|---|
| 1 | left camera image | `src/datasets/kitti2015.py` |
| 2 | ARM-P disparity (px) | `stage_c_deploy/metric_depth/armp_depth.py` — **the only neural stage** |
| 3 | metric depth (m), `Z = fB/d` | `metric_depth/metric_depth.py` (C2, 135/135 PASS) |
| 4 | depth discontinuities (m/px) | `spatial_perception/discontinuity.py` (C2.1.1, 19/19 PASS) |

The text panel adds the spatial layer (C2.1, 251/251 PASS): left/centre/right
median depth, nearest surface ahead, and how many 4×6 occupancy cells are NEAR
by cell-median depth.

**Click any image panel** to measure that point — depth in metres plus X/Y/Z in
the camera frame, via the same `measurement.pixel_measure` the C2 validator uses.

## What it does not do

- It is **not a new algorithm.** Every number comes from a module that Stage C
  already validated; this script only composes them and draws the result.
- It runs the **PyTorch checkpoint on this machine**. It is not Hailo, not
  quantized, and the timings shown are development-machine measurements.
- Per-scene EPE/D1 in the panel are for **that scene only**. The frozen
  40-scene contract score of this checkpoint is **1.1912168 px** and is quoted,
  never recomputed here.
- The ≤60 m physical mask is applied to the **displayed statistics only**; the
  stored depth is never modified or clipped.

## Honesty footer (shown in every run)

The demo prints, and draws, the two facts a viewer must not miss:

- **ONNX export parity C1 = FAIL** (1.708984375e-3 px against the frozen
  1e-3 px criterion).
- **Hailo target device UNSPECIFIED, toolchain NOT EXECUTED, no ARM-P HEF.**

The checkpoint's SHA-256 is asserted before it is loaded; if the file ever
changes, the demo stops instead of showing a different model.
