"""Mentor demonstration package for the current demonstration model.

Builds ``phase2/demo/`` from the Stereo Depth Visualizer V1 primitives
(``phase2/viz``). This is presentation only: it loads the frozen demonstration
checkpoint named by ``core.DEMO_MODEL`` read-only, runs inference, and writes
PNGs plus a manifest. It trains nothing, tunes nothing, and writes nothing into
``experiments/``, ``results/`` or Phase 1.

Scene choice is by evidence, not by eye: every scene in the split is scored by
this model's per-scene D1, the demonstration scenes are drawn from the better
half of that ranking, and the single worst scene is included as a disclosed
failure case. Ranks are printed on every figure and recorded in the manifest.

Probe regions are a FIXED grid, identical for every scene, declared in
``PROBE_BOXES`` below -- so no box was moved to make a number look better.

    python phase2/scripts/mentor_demo.py                # full package
    python phase2/scripts/mentor_demo.py --out somewhere --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2.viz import core, render  # noqa: E402

MODEL_KEY = core.DEMO_MODEL   # repoint core.DEMO_MODEL, not this
SPLIT = "hailo_val"
DEFAULT_OUT = REPO_ROOT / "phase2" / "demo"

# Chosen from the ranking below over all 40 hailo_val scenes; ranks recorded in
# the manifest. Three from the better half, one worst case, disclosed as such.
DEMO_SCENES = [31, 27, 0]
FAILURE_SCENES = [1]

# One fixed grid for every scene. Not per-scene tuning.
PROBE_BOXES = {
    "road ahead": (560, 300, 700, 355),
    "left side": (180, 160, 330, 265),
    "right side": (900, 160, 1050, 265),
    "far centre": (590, 150, 700, 205),
}

# The corruption used for the stereo-dependence test. Reuses Phase 1's own
# variant set (scripts/exp_right_image_ablation.variants) rather than inventing
# a new one, so the demonstrated corruption is the measured corruption.
STEREO_TEST_VARIANT = "right_from_other_scene"


def two_panel(scene: core.Scene, values, valid, unit: str, cmap: str,
              headline: str, subtitle: str):
    """The main mentor view: the left camera image, and the same scene coloured
    by the model's estimate. Stacked rather than side by side because the
    frames are 1232x368 -- side by side, each panel would be a thumbnail."""
    fig = render.plt.figure(figsize=(13, 7.4), dpi=130)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.02], wspace=0.015,
                          hspace=0.16)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    render.show_rgb(ax_top, scene.left, "1. WHAT THE LEFT CAMERA SEES")
    im = render.show_overlay(
        ax_bot, scene.left, values, valid,
        "2. WHAT THE MODEL ESTIMATES  --  colour = " + headline,
        unit, cmap=cmap, alpha=0.72, colorbar=False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(unit, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle(subtitle, fontsize=11)
    return fig


def probe_figure(scene: core.Scene, pred: core.Prediction, subtitle: str):
    """The fixed probe grid drawn on the image, each box labelled with its own
    median disparity and the distance derived from it."""
    depth, ok = core.depth_from_disparity(pred.disparity, scene.calibration)
    fig, ax = render.plt.subplots(figsize=(13, 4.8), dpi=130)
    if depth is None:
        render.show_rgb(ax, scene.left, "left image")
        ax.set_title(scene.calibration_note, fontsize=9)
    else:
        render.show_overlay(ax, scene.left, depth, ok,
                            "estimated distance, with probe regions", "m",
                            cmap=render.DEPTH_CMAP, alpha=0.55)

    rows = {}
    for name, box in PROBE_BOXES.items():
        s = core.region_stats(scene, pred, *box)
        rows[name] = s
        x1, y1, x2, y2 = s["region"]
        ax.add_patch(render.plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          fill=False, ec="#39ff14", lw=1.8))
        if depth is None:
            label = "{}\nshift {}".format(
                name, core.fmt(s.get("pred_disparity_median"), "px", 1))
        else:
            label = "{}\nshift {} -> {}".format(
                name,
                core.fmt(s.get("pred_disparity_median"), "px", 1),
                core.fmt(s.get("pred_depth_median_m"), "m", 1))
        ax.text(x1, max(y1 - 6, 12), label, color="#39ff14", fontsize=8,
                family="monospace", va="bottom",
                bbox=dict(fc="black", ec="none", alpha=0.55, pad=1.5))
    fig.suptitle(subtitle + "\nEach box: median pixel shift between the two "
                 "cameras, and the distance that shift implies.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig, rows


def stereo_dependence(scene: core.Scene, runner: core.ModelRunner,
                      out_dir: Path) -> dict:
    """Same left image, second camera replaced by a different scene's right
    image. Reports what changed. A diagnostic, not a normal operating mode."""
    import torch
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from exp_right_image_ablation import variants  # noqa: E402 (read-only import)
    from src.datasets.kitti2015 import normalize  # noqa: E402
    from src.evaluation.metrics import disparity_metrics  # noqa: E402

    other = core.load_scene((scene.index + 7) % len(core.scene_names(SPLIT)),
                            split=SPLIT)
    right_bad = variants(scene.left, scene.right, other.right)[STEREO_TEST_VARIANT]

    model, _, _ = runner._model(MODEL_KEY)
    with torch.no_grad():
        out = model(torch.from_numpy(normalize(scene.left)).to(runner.device),
                    torch.from_numpy(normalize(right_bad)).to(runner.device))
    bad = out[0, 0].cpu().numpy().astype(np.float32)
    good = runner.predict(MODEL_KEY, scene).disparity

    valid = scene.gt_valid
    gt = scene.gt_disparity
    m_good = disparity_metrics(good[valid], gt[valid])
    m_bad = disparity_metrics(bad[valid], gt[valid])

    d_good, ok_good = core.depth_from_disparity(good, scene.calibration)
    d_bad, ok_bad = core.depth_from_disparity(bad, scene.calibration)

    fig, axes = render.plt.subplots(3, 2, figsize=(17, 9.6), dpi=115)
    render.show_rgb(axes[0, 0], scene.right, "second camera: REAL right image")
    render.show_rgb(axes[0, 1], right_bad,
                    "second camera: REPLACED with a different scene "
                    "(deliberate corruption)")
    lo, hi, _ = render._range(good, None)
    render.show_map(axes[1, 0], good, None,
                    "estimated pixel shift (disparity)", "px",
                    render.DISPARITY_CMAP, vmin=lo, vmax=hi)
    render.show_map(axes[1, 1], bad, None,
                    "estimated pixel shift with the corrupted second camera",
                    "px", render.DISPARITY_CMAP, vmin=lo, vmax=hi)
    if d_good is not None:
        dlo, dhi, _ = render._range(d_good, ok_good)
        render.show_map(axes[2, 0], d_good, ok_good, "estimated distance", "m",
                        render.DEPTH_CMAP, vmin=dlo, vmax=dhi)
        render.show_map(axes[2, 1], d_bad, ok_bad,
                        "estimated distance with the corrupted second camera",
                        "m", render.DEPTH_CMAP, vmin=dlo, vmax=dhi)
    else:
        for ax in axes[2]:
            ax.axis("off")
    fig.suptitle(
        "STEREO DEPENDENCE TEST -- {}  (diagnostic, not a normal operating "
        "mode)\nLeft image unchanged. Only the second camera was replaced. "
        "Error rate {:.2f} % -> {:.2f} % ({:+.2f} points), average disparity "
        "error {:.3f} -> {:.3f} px. Mean |change| {:.2f} px.".format(
            scene.name, m_good.d1, m_bad.d1, m_bad.d1 - m_good.d1,
            m_good.epe, m_bad.epe, float(np.abs(bad - good).mean())),
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    stem = Path(scene.name).stem
    fig.savefig(out_dir / "{}_stereo_dependence.png".format(stem),
                bbox_inches="tight")
    render.plt.close(fig)

    render.save_rgb_png(out_dir / "{}_left_unchanged.png".format(stem),
                        scene.left, "first camera (LEFT): unchanged in both runs")
    render.save_rgb_png(out_dir / "{}_right_normal.png".format(stem),
                        scene.right, "second camera: REAL right image")
    render.save_rgb_png(out_dir / "{}_right_corrupted.png".format(stem),
                        right_bad, "second camera: REPLACED (corruption)")

    return {
        "scene": scene.name,
        "corruption": STEREO_TEST_VARIANT,
        "corruption_source": "scripts/exp_right_image_ablation.py::variants",
        "normal": {"epe_px": m_good.epe, "d1_percent": m_good.d1},
        "corrupted_right": {"epe_px": m_bad.epe, "d1_percent": m_bad.d1},
        "d1_penalty_points": m_bad.d1 - m_good.d1,
        "mean_abs_disparity_change_px": float(np.abs(bad - good).mean()),
    }


def rank(runner: core.ModelRunner) -> list[dict]:
    """Per-scene D1/EPE for every scene in the split, so scene choice is
    evidence-based. Same computation as ``stereo_visualizer.py --rank``: one
    scene at a time, pooled over that scene's gt > 0 pixels, full frames. NOT
    the recorded 10-scene training-time protocol and not comparable to it."""
    rows = []
    for index in range(len(core.scene_names(SPLIT))):
        sc = core.load_scene(index, split=SPLIT)
        m = core.scene_metrics(runner.predict(MODEL_KEY, sc).disparity, sc)
        rows.append({"index": index, "scene": sc.name,
                     "epe_px": m["epe"], "d1_percent": m["d1"]})
    rows.sort(key=lambda r: r["d1_percent"])
    return rows


def build(out_root: Path, device: str | None) -> dict:
    runner = core.ModelRunner(device=device)
    dirs = {k: out_root / k for k in
            ("01_input", "02_disparity", "03_depth", "04_overlay", "05_probe",
             "06_stereo_test")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    ranking = rank(runner)
    manifest: dict = {
        "tool": "phase2 mentor demo, built on Stereo Depth Visualizer V1",
        "model": MODEL_KEY,
        "experiment": core.MODELS[MODEL_KEY]["experiment"],
        "split": SPLIT,
        "git_revision": core.git_revision(),
        "probe_boxes": {k: list(v) for k, v in PROBE_BOXES.items()},
        "scene_selection": (
            "ranked all {} {} scenes by this model's per-scene D1; the "
            "demonstration scenes are drawn from the better half and the "
            "failure case is the single worst scene. Representative "
            "demonstration scenes, not a statistical benchmark.".format(
                len(ranking), SPLIT)),
        "ranking": ranking,
        "scenes": {},
    }

    for index in DEMO_SCENES + FAILURE_SCENES:
        is_failure = index in FAILURE_SCENES
        scene = core.load_scene(index, split=SPLIT)
        pred = runner.predict(MODEL_KEY, scene)
        stem = Path(scene.name).stem
        rank_pos = next(i for i, r in enumerate(ranking) if r["index"] == index)
        m = core.scene_metrics(pred.disparity, scene)
        tag = "FAILURE CASE" if is_failure else "demonstration scene"
        subtitle = (
            "{} -- {} ({} #{}), {}\n"
            "this scene: average disparity error {:.2f} px, {:.1f} % of "
            "ground-truth pixels off by more than 3 px  |  rank {}/{} of this "
            "split by that error rate".format(
                tag, scene.name, SPLIT, index,
                "metric distance available" if scene.calibration is not None
                else "METRIC DISTANCE UNAVAILABLE",
                m["epe"], m["d1"], rank_pos + 1, len(ranking)))

        render.save_rgb_png(dirs["01_input"] / "{}_left.png".format(stem),
                            scene.left, "LEFT camera -- " + scene.name)
        render.save_rgb_png(dirs["01_input"] / "{}_right.png".format(stem),
                            scene.right, "RIGHT camera -- " + scene.name)

        render.save_map_png(
            dirs["02_disparity"] / "{}_disparity.png".format(stem),
            pred.disparity, None,
            "ESTIMATED DISPARITY (pixel shift between the two cameras) -- "
            + scene.name, "px", render.DISPARITY_CMAP)
        render.save_map_png(
            dirs["02_disparity"] / "{}_disparity_groundtruth.png".format(stem),
            scene.gt_disparity, scene.gt_valid,
            "MEASURED disparity, LiDAR ground truth (grey = not measured) -- "
            + scene.name, "px", render.DISPARITY_CMAP)
        err, err_valid = render.error_map(pred.disparity, scene)
        render.save_map_png(
            dirs["02_disparity"] / "{}_error.png".format(stem), err, err_valid,
            "DISAGREEMENT with ground truth, |estimate - measured| (grey = no "
            "ground truth, NOT zero error) -- " + scene.name,
            "px", render.ERROR_CMAP)

        depth, ok = core.depth_from_disparity(pred.disparity, scene.calibration)
        entry = {
            "name": scene.name, "index": index, "rank": rank_pos + 1,
            "of": len(ranking), "role": "failure_case" if is_failure else "demo",
            "scene_metrics": m,
            "calibration": scene.calibration_note,
            "checkpoint": str(pred.checkpoint.relative_to(REPO_ROOT)),
            "checkpoint_sha256": pred.checkpoint_sha256,
            "device": pred.device,
        }
        if depth is None:
            entry["metric_depth"] = "unavailable: " + str(scene.calibration_error)
            fig = two_panel(scene, pred.disparity, None, "px",
                            render.DISPARITY_CMAP,
                            "estimated pixel shift in pixels "
                            "(bright = large shift = close)",
                            subtitle + "\nMETRIC DEPTH UNAVAILABLE for this "
                            "scene -- disparity only.")
        else:
            render.save_map_png(dirs["03_depth"] / "{}_depth.png".format(stem),
                                depth, ok,
                                "ESTIMATED DISTANCE (metres) -- " + scene.name,
                                "m", render.DEPTH_CMAP)
            gt_depth, gt_ok = scene.gt_depth()
            if gt_depth is not None:
                render.save_map_png(
                    dirs["03_depth"] / "{}_depth_groundtruth.png".format(stem),
                    gt_depth, gt_ok,
                    "MEASURED distance from ground truth (grey = not measured) "
                    "-- " + scene.name, "m", render.DEPTH_CMAP)
            entry["metric_depth"] = {
                "valid_pixels": int(ok.sum()),
                "p2_m": float(np.nanpercentile(depth[ok], 2)),
                "median_m": float(np.nanmedian(depth[ok])),
                "p98_m": float(np.nanpercentile(depth[ok], 98)),
            }
            fig = two_panel(scene, depth, ok, "m", render.DEPTH_CMAP,
                            "estimated distance in metres "
                            "(yellow = close, purple = far away)", subtitle)
        fig.savefig(dirs["04_overlay"] / "{}_overlay.png".format(stem),
                    bbox_inches="tight")
        render.plt.close(fig)

        fig, rows = probe_figure(scene, pred, subtitle)
        fig.savefig(dirs["05_probe"] / "{}_probe.png".format(stem),
                    bbox_inches="tight")
        render.plt.close(fig)
        entry["probe"] = {
            name: {
                "region": s["region"],
                "median_disparity_px": s.get("pred_disparity_median"),
                "median_depth_m": s.get("pred_depth_median_m"),
                "depth_p10_m": s.get("pred_depth_p10_m"),
                "depth_p90_m": s.get("pred_depth_p90_m"),
                "ground_truth_pixels": s.get("gt_valid_pixels"),
                "disparity_mae_px": s.get("disparity_mae"),
            } for name, s in rows.items()
        }
        manifest["scenes"][scene.name] = entry
        print("built {} (rank {}/{}, EPE {:.2f} px, D1 {:.1f} %)".format(
            scene.name, rank_pos + 1, len(ranking), m["epe"], m["d1"]))

    scene = core.load_scene(DEMO_SCENES[0], split=SPLIT)
    manifest["stereo_dependence_test"] = stereo_dependence(
        scene, runner, dirs["06_stereo_test"])
    print("built stereo dependence test on " + scene.name)

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default=None, help="cuda / cpu")
    args = ap.parse_args()
    out = Path(args.out)
    manifest = build(out, args.device)
    print("\nwrote {} ({} scenes)".format(out, len(manifest["scenes"])))


if __name__ == "__main__":
    main()
