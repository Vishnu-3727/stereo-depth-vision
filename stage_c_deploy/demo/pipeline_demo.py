"""MICROCHIP STEREONET -- end-to-end pipeline demo (ARM-P seed 1).

Shows what the deployment candidate actually produces, on a real KITTI stereo
pair, through the exact modules that Stage C validated (C2 135/135,
C2.1 251/251, C2.1.1 19/19):

    stereo pair -> ARM-P disparity -> metric depth (m) -> spatial cells,
    occupancy, depth discontinuities

Nothing here is a new algorithm. Every number comes from a frozen, validated
function; this file only composes them and draws the result. The checkpoint's
SHA-256 is asserted before it is loaded.

    python stage_c_deploy/demo/pipeline_demo.py                 # window, click to measure
    python stage_c_deploy/demo/pipeline_demo.py --scene 5
    python stage_c_deploy/demo/pipeline_demo.py --no-window --save demo.png
    python stage_c_deploy/demo/pipeline_demo.py --list

Click any image panel to measure that point: depth in metres, plus X/Y/Z in
the camera frame.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "metric_depth"))
sys.path.insert(0, str(REPO / "stage_c_deploy" / "spatial_perception"))

from src.datasets.kitti2015 import Kitti2015Stereo                # noqa: E402
from src.geometry.stereo import parse_kitti_cam_to_cam            # noqa: E402

import armp_depth as AD                                           # noqa: E402
from metric_depth import (depth_stats, disparity_to_depth,        # noqa: E402
                          read_fy_from_calib, reproject_xyz)
from measurement import pixel_measure                             # noqa: E402
from discontinuity import depth_discontinuity                     # noqa: E402
from occupancy import occupancy_grid                              # noqa: E402
from pointcloud import physical_mask                              # noqa: E402
from spatial import spatial_cells                                 # noqa: E402

# Quoted, never recomputed here: the frozen 40-scene contract score of this
# checkpoint (stage_b_armp/ARMP_CLOSURE_RECORD.md section 5).
FROZEN_EPE_40 = 1.1912168
PHYSICAL_MAX_M = 60.0      # separate mask only; stored depth is never modified
NEAR_THRESHOLD_M = 10.0    # caller-supplied, reported with every occupancy cell


def scene_accuracy(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Per-scene EPE / D1 over gt > 0, the frozen contract's pixel rule."""
    valid = gt > 0
    n = int(valid.sum())
    if not n:
        return {"n": 0, "epe": float("nan"), "d1": float("nan")}
    err = np.abs(pred[valid] - gt[valid])
    d1 = np.mean((err > 3.0) & (err > 0.05 * gt[valid])) * 100.0
    return {"n": n, "epe": float(err.mean()), "d1": float(d1)}


def run_scene(index: int, device: str | None = None) -> dict:
    """Run the whole pipeline on one hailo_val scene. Returns everything drawn."""
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val")
    assert 0 <= index < len(ds), f"scene {index} outside 0..{len(ds) - 1}"
    sample = ds[index]

    net, dev = AD.load_frozen_net(device)

    t0 = time.perf_counter()
    disparity = AD.infer_disparity_scene(net, dev, sample.name)
    t_infer = time.perf_counter() - t0

    cal_path = ds.calibration_path(sample.name)
    calib = parse_kitti_cam_to_cam(cal_path)
    fy = read_fy_from_calib(cal_path)

    t1 = time.perf_counter()
    depth, valid = disparity_to_depth(calib, disparity)
    xs, ys, zs = reproject_xyz(calib, depth, fy)
    phys = physical_mask(depth, valid, PHYSICAL_MAX_M)
    cells = spatial_cells(depth, valid, 1, 3, physical_valid=phys)
    occ = occupancy_grid(depth, phys, 4, 6, near_threshold_m=NEAR_THRESHOLD_M)
    mag, mag_valid = depth_discontinuity(depth, valid)
    t_post = time.perf_counter() - t1

    return {
        "name": sample.name, "index": index, "device": dev,
        "left": sample.left, "gt": sample.disparity,
        "disparity": disparity, "depth": depth, "valid": valid,
        "phys": phys, "xs": xs, "ys": ys, "zs": zs,
        "calib": calib, "fy": fy,
        "cells": cells, "occupancy": occ, "mag": mag, "mag_valid": mag_valid,
        "stats": depth_stats(depth, phys),
        "accuracy": scene_accuracy(disparity, sample.disparity),
        "t_infer": t_infer, "t_post": t_post,
    }


def summary_lines(r: dict) -> list[str]:
    c, a, s = r["calib"], r["accuracy"], r["stats"]
    near = [o for o in r["occupancy"] if o.get("state") == "NEAR"]
    centre = next((x for x in r["cells"] if "centre" in x.get("name", "").lower()
                   or "center" in x.get("name", "").lower()), None)
    cell_txt = "  ".join(
        f"{x['name']}: {x.get('median_m', float('nan')):.1f} m" for x in r["cells"])
    return [
        f"MODEL   ARM-P seed 1  ({AD.ARMP_SHA[:12]}...)  397,954 params  on {r['device']}",
        f"SCENE   {r['name']}   (hailo_val #{r['index']})   1232x368",
        f"CAMERA  f = {c.focal_px:.1f} px   B = {c.baseline_m:.4f} m"
        f"   fB = {c.fB:.1f} px*m",
        "",
        f"ACCURACY, this scene vs LiDAR ground truth ({a['n']:,} valid px)",
        f"        EPE {a['epe']:.3f} px      D1 {a['d1']:.2f} %",
        f"        frozen 40-scene contract score: {FROZEN_EPE_40} px (quoted)",
        "",
        f"DEPTH   median {s.get('median_m', float('nan')):.1f} m   "
        f"range {s.get('min_m', float('nan')):.1f} - {s.get('max_m', float('nan')):.1f} m"
        f"   ({s['percent_valid']:.1f} % of frame valid)",
        f"        <= {PHYSICAL_MAX_M:.0f} m physical mask applied to the stats only;"
        " stored depth never modified",
        f"SCENE   {cell_txt}",
        f"NEAREST {centre['min_m']:.2f} m ahead"
        if centre and np.isfinite(centre.get("min_m", float("nan")))
        else "NEAREST n/a",
        f"OCCUPANCY  {len(near)} of {len(r['occupancy'])} grid cells NEAR"
        f" (cell-median depth <= {NEAR_THRESHOLD_M:.0f} m)",
        "",
        f"TIMING  inference {r['t_infer'] * 1e3:.0f} ms (first call, includes warm-up)"
        f"   depth+spatial {r['t_post'] * 1e3:.0f} ms",
        "        DEVELOPMENT-MACHINE MEASUREMENT -- never a Hailo figure",
        "",
        "STATUS  ONNX export parity C1 = FAIL (1.709e-3 px vs 1e-3 px criterion)",
        "        Hailo target device UNSPECIFIED, toolchain NOT EXECUTED, no HEF",
        "        this runs the PyTorch checkpoint on this machine only",
    ]


def draw(r: dict, save: Path | None, show: bool) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth_show = np.where(r["phys"], r["depth"], np.nan)
    mag_show = np.where(r["mag_valid"], r["mag"], np.nan)
    disp_show = np.where(r["valid"], r["disparity"], np.nan)

    fig = plt.figure(figsize=(17, 9))
    fig.canvas.manager.set_window_title(
        f"MICROCHIP STEREONET -- ARM-P seed 1 -- {r['name']}")
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.25], hspace=0.33, wspace=0.08)

    panels = []
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(r["left"]); panels.append(ax)
    ax.set_title("1. left camera", loc="left", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(disp_show, cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.025, label="px")
    ax.set_title("2. ARM-P disparity", loc="left", fontsize=10); panels.append(ax)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(depth_show, cmap="viridis_r")
    fig.colorbar(im, ax=ax, fraction=0.025, label="metres")
    ax.set_title(f"3. metric depth   Z = fB/d   (white = no disparity or beyond"
                 f" {PHYSICAL_MAX_M:.0f} m)", loc="left", fontsize=10)
    panels.append(ax)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(mag_show, cmap="inferno", vmax=np.nanpercentile(mag_show, 99))
    fig.colorbar(im, ax=ax, fraction=0.025, label="m / px")
    ax.set_title("4. depth discontinuities (object edges)", loc="left", fontsize=10)
    panels.append(ax)

    for a in panels:
        a.set_xticks([]); a.set_yticks([])

    axt = fig.add_subplot(gs[2, :]); axt.axis("off")
    axt.text(0.0, 1.0, "\n".join(summary_lines(r)), family="monospace",
             fontsize=9.5, va="top", ha="left", transform=axt.transAxes)

    readout = axt.text(0.0, 1.10, "click any image panel to measure a point",
                       family="monospace", fontsize=11, color="#1a7f4b", va="bottom",
                       transform=axt.transAxes)
    markers: list = []

    def on_click(event):
        if event.inaxes not in panels or event.xdata is None:
            return
        u, v = int(round(event.xdata)), int(round(event.ydata))
        h, w = r["depth"].shape
        if not (0 <= u < w and 0 <= v < h):
            return
        m = pixel_measure(r["calib"], r["disparity"], r["depth"],
                          r["xs"], r["ys"], u, v, fy=r["fy"])
        if m["valid"]:
            readout.set_text(
                f"pixel ({u},{v}):  depth {m['pixel_depth_m']:.2f} m"
                f"   disparity {m['disparity_px']:.2f} px"
                f"   X {m['X_m']:+.2f} m   Y {m['Y_m']:+.2f} m")
            readout.set_color("#2a6")
        else:
            readout.set_text(f"pixel ({u},{v}): no valid disparity here")
            readout.set_color("#c33")
        while markers:
            markers.pop().remove()
        for a in panels:
            markers.append(a.plot(u, v, "+", color="#ff3", ms=12, mew=2)[0])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=110)
        print(f"saved {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=int, default=31,
                    help="hailo_val index 0..39 (default 31 = 000191_10)")
    ap.add_argument("--list", action="store_true", help="list the scenes and exit")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if present)")
    ap.add_argument("--save", default=None, help="write the figure to this path")
    ap.add_argument("--no-window", action="store_true", help="print numbers, draw offscreen")
    args = ap.parse_args()

    if args.list:
        ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val")
        for i, n in enumerate(ds.names):
            print(f"{i:3d}  {n}")
        return

    r = run_scene(args.scene, args.device)
    print()
    for line in summary_lines(r):
        print("  " + line)
    print()
    draw(r, Path(args.save) if args.save else None, show=not args.no_window)


if __name__ == "__main__":
    main()
