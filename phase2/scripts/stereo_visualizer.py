"""Stereo Depth Visualizer V1 -- CLI and interactive viewer.

Looks at what the H1-v2 checkpoints actually predict. It reads checkpoints,
data and experiment records; it writes only into the output directory you give
it (default ``phase2/visualizations/``, git-ignored). Nothing under
``experiments/``, ``results/training/`` or Phase 1 is touched.

    # list the validation scenes
    python phase2/scripts/stereo_visualizer.py --list

    # full artefact set for one scene, both arms
    python phase2/scripts/stereo_visualizer.py --scene 0

    # several scenes, plus a back-projected point cloud
    python phase2/scripts/stereo_visualizer.py --scenes 0 3 7 --point-cloud 6

    # numbers for one pixel and one region, with a correspondence figure
    python phase2/scripts/stereo_visualizer.py --scene 0 --probe 742,191 \\
        --region 600,150,900,300

    # interactive: click a pixel, drag a region, keys switch mode/scene/model
    python phase2/scripts/stereo_visualizer.py --scene 0 --interactive
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

DEFAULT_OUT = REPO_ROOT / "phase2" / "visualizations"
MODES = ["rgb", "disparity", "depth", "depth_overlay", "error", "difference"]


def print_pixel_probe(probe: dict) -> None:
    x, y = probe["pixel"]
    print("\nPixel ({}, {})".format(x, y))
    print("  GT disparity:      {}".format(core.fmt(probe["gt_disparity"], "px")))
    print("  GT depth:          {}".format(core.fmt(probe["gt_depth_m"], "m")))
    print("  GT match x_right:  {}".format(core.fmt(probe["gt_x_right"], "", 1)))
    for key, m in probe["models"].items():
        print("  [{}]".format(key))
        print("    pred disparity:  {}".format(core.fmt(m["disparity"], "px")))
        print("    disparity error: {}".format(core.fmt(m["disparity_error"], "px")))
        print("    pred depth:      {}".format(core.fmt(m["depth_m"], "m")))
        print("    depth error:     {}".format(core.fmt(m["depth_error_m"], "m")))
        print("    match x_right:   {}".format(core.fmt(m["x_right"], "", 1)))


def print_scene_panel(scene: core.Scene, predictions: dict[str, core.Prediction]) -> None:
    print("\nScene {} ({} #{}), {}x{}".format(
        scene.name, scene.split, scene.index, scene.shape[1], scene.shape[0]))
    print("  calibration: " + scene.calibration_note)
    print("  ground truth: {} valid pixels ({:.1f}% of frame)".format(
        int(scene.gt_valid.sum()), 100.0 * scene.gt_valid.mean()))
    print("\n  scene-level metrics (this scene only, pooled over gt > 0):")
    for key, pred in predictions.items():
        m = core.scene_metrics(pred.disparity, scene)
        print("    {:<8} EPE {:>7.3f} px   D1 {:>6.2f} %   RMSE {:>7.3f} px   "
              "valid {}".format(key, m["epe"], m["d1"], m["rmse"], m["valid_pixels"]))
    print("\n  recorded global validation metrics (not recomputed here):")
    for key in predictions:
        g = core.recorded_global_metrics(key)
        if not g.get("available"):
            print("    {:<8} unavailable ({})".format(key, g["source"]))
            continue
        print("    {:<8} epoch {}: EPE {} px, D1 {} %  [{}]".format(
            key, g["final_epoch"], core.fmt(g["val_epe"], "", 3),
            core.fmt(g["val_d1"], "", 2), g["source"]))
    print("    protocol: 10 scenes of hailo_val during training -- not the same "
          "population as the per-scene numbers above")


class Viewer:
    """Lightweight matplotlib viewer: click a pixel, drag a region, press keys.

    Deliberately not a web app. It reuses the already-computed prediction for
    every interaction; inference runs once per (model, scene).
    """

    HELP = ("keys: m mode | n/p scene | b model | c correspondence | 3 point cloud | "
            "s save report | q quit    mouse: click = pixel probe, drag = region")

    def __init__(self, runner: core.ModelRunner, split: str, index: int,
                 model_keys: list[str], out_root: Path) -> None:
        import matplotlib
        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector

        self.plt = plt
        self.runner = runner
        self.split = split
        self.names = core.scene_names(split)
        self.index = index
        self.model_keys = model_keys
        self.model = model_keys[0]
        self.out_root = out_root
        # "difference" compares two loaded models; with one model it can only
        # print an apology, so it is not in the cycle at all then. The desktop
        # shortcut loads exactly one model, which is how this surfaced.
        self.modes = [m for m in MODES
                      if m != "difference" or len(model_keys) > 1]
        self.mode = "disparity"
        self.pixel: tuple[int, int] | None = None
        self.region: tuple[int, int, int, int] | None = None

        # A 15x8 in figure is 1500x800 px, which overflows a 1280x800 laptop
        # panel and clips the readout strip -- the numbers -- off the bottom.
        # It has to be fixed at construction: matplotlib restores the window to
        # the figure size on draw, so a later wm_geometry or wm_state("zoomed")
        # is simply undone. Shrink only, never grow.
        figsize = (15.0, 8.0)
        try:
            import tkinter
            probe = tkinter.Tk()
            probe.withdraw()
            sw, sh = probe.winfo_screenwidth(), probe.winfo_screenheight()
            probe.destroy()
            dpi = float(plt.rcParams["figure.dpi"])
            figsize = (min(15.0, (sw - 90) / dpi), min(8.0, (sh - 110) / dpi))
        except Exception:
            pass
        self.fig = plt.figure(figsize=figsize)
        try:   # position only -- matplotlib overrides the size, not the origin
            self.fig.canvas.manager.window.wm_geometry("+0+0")
        except Exception:
            pass
        self.ax = self.fig.add_axes((0.03, 0.30, 0.94, 0.62))
        self.text_ax = self.fig.add_axes((0.03, 0.02, 0.94, 0.26))
        self.text_ax.axis("off")
        self.cbar = None
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.selector = RectangleSelector(
            self.ax, self._on_region, useblit=False, button=[1],
            minspanx=5, minspany=5, spancoords="pixels", interactive=True,
        )
        self._load()

    def _load(self) -> None:
        self.scene = core.load_scene(self.index, split=self.split)
        self.preds = {k: self.runner.predict(k, self.scene) for k in self.model_keys}
        self.pixel = None
        self.region = None
        self._draw()

    def _draw(self) -> None:
        self.ax.clear()
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        pred = self.preds[self.model]
        scene = self.scene

        if self.mode == "rgb":
            render.show_rgb(self.ax, scene.left, "left image -- " + scene.name)
        elif self.mode == "disparity":
            render.show_map(self.ax, pred.disparity, None,
                            "predicted disparity -- {}".format(self.model), "px",
                            render.DISPARITY_CMAP, colorbar=False)
        elif self.mode == "depth":
            depth, ok = core.depth_from_disparity(pred.disparity, scene.calibration)
            if depth is None:
                self._unavailable(scene.calibration_note)
            else:
                render.show_map(self.ax, depth, ok,
                                "predicted depth -- {}".format(self.model), "m",
                                render.DEPTH_CMAP, colorbar=False)
        elif self.mode == "depth_overlay":
            depth, ok = core.depth_from_disparity(pred.disparity, scene.calibration)
            if depth is None:
                self._unavailable(scene.calibration_note)
            else:
                render.show_overlay(self.ax, scene.left, depth, ok,
                                    "depth over left image -- " + self.model, "m")
        elif self.mode == "error":
            err, valid = render.error_map(pred.disparity, scene)
            render.show_map(self.ax, err, valid,
                            "|pred - gt| -- {} (grey = no ground truth)".format(self.model),
                            "px", render.ERROR_CMAP, colorbar=False)
        elif self.mode == "difference":
            if len(self.model_keys) < 2:
                self._unavailable("difference needs two models loaded")
            else:
                a, b = self.model_keys[0], self.model_keys[1]
                diff = self.preds[b].disparity.astype(np.float64) - self.preds[a].disparity
                render.show_map(self.ax, diff, None,
                                "{} - {} disparity (a difference, not an improvement)".format(b, a),
                                "px", render.DIFF_CMAP, symmetric=True, colorbar=False)

        if self.pixel is not None:
            self.ax.plot(self.pixel[0], self.pixel[1], "o", mfc="none",
                         mec="#39ff14", ms=11, mew=1.6)
        if self.region is not None:
            x1, y1, x2, y2 = self.region
            self.ax.add_patch(self.plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, ec="#39ff14", lw=1.4))
        self._draw_text()
        self.fig.canvas.draw_idle()

    def _unavailable(self, message: str) -> None:
        self.ax.imshow(self.scene.left)
        self.ax.set_title(message, fontsize=9)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def _draw_text(self) -> None:
        scene, pred = self.scene, self.preds[self.model]
        lines = [
            "scene {} [{}/{}]  mode {}  model {}   {}".format(
                scene.name, self.index, len(self.names) - 1, self.mode,
                self.model, self.HELP),
        ]
        parts = []
        for key, p in self.preds.items():
            m = core.scene_metrics(p.disparity, scene)
            parts.append("{}: EPE {:.3f} px, D1 {:.2f} %".format(key, m["epe"], m["d1"]))
        lines.append("SCENE  " + "   |   ".join(parts)
                     + "   |   valid GT {} px".format(int(scene.gt_valid.sum())))

        if self.pixel is not None:
            x, y = self.pixel
            probe = core.probe_pixel(scene, self.preds, x, y)
            seg = ["PIXEL ({}, {})  GT d {}  GT Z {}".format(
                x, y, core.fmt(probe["gt_disparity"], "px"),
                core.fmt(probe["gt_depth_m"], "m"))]
            for key, m in probe["models"].items():
                seg.append("  {}: d {}  err {}  Z {}  dZ {}  x_r {}".format(
                    key, core.fmt(m["disparity"], "px"),
                    core.fmt(m["disparity_error"], "px"),
                    core.fmt(m["depth_m"], "m"),
                    core.fmt(m["depth_error_m"], "m"),
                    core.fmt(m["x_right"], "", 1)))
            lines += seg
        else:
            lines.append("PIXEL  click the image")

        if self.region is not None:
            s = core.region_stats(scene, pred, *self.region)
            if s.get("pixels"):
                lines.append(
                    "REGION {}  [{}]  {} px, {} with pred depth, {} with GT   "
                    "median Z {}  p10-p90 {}..{}  median d {}".format(
                        s["region"], self.model, s["pixels"],
                        s.get("pred_disparity_valid_pixels", 0), s.get("gt_valid_pixels", 0),
                        core.fmt(s.get("pred_depth_median_m"), "m"),
                        core.fmt(s.get("pred_depth_p10_m")), core.fmt(s.get("pred_depth_p90_m")),
                        core.fmt(s.get("pred_disparity_median"), "px")))
                lines.append(
                    "       region error: MAE {}  median AE {}  D1 {}  depth MAE {}".format(
                        core.fmt(s.get("disparity_mae"), "px"),
                        core.fmt(s.get("disparity_median_ae"), "px"),
                        core.fmt(s.get("d1_percent"), "%"),
                        core.fmt(s.get("depth_mae_m"), "m")))
            else:
                lines.append("REGION empty selection")
        else:
            lines.append("REGION  drag on the image")

        self.text_ax.clear()
        self.text_ax.axis("off")
        self.text_ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                          family="monospace", fontsize=8)

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        h, w = self.scene.shape
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= x < w and 0 <= y < h:
            self.pixel = (x, y)
            self._draw()

    def _on_region(self, press, release) -> None:
        if press.xdata is None or release.xdata is None:
            return
        self.region = (int(press.xdata), int(press.ydata),
                       int(release.xdata), int(release.ydata))
        self._draw()

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == "q":
            self.plt.close(self.fig)
        elif key == "m":
            self.mode = self.modes[
                (self.modes.index(self.mode) + 1) % len(self.modes)]
            self._draw()
        elif key == "b":
            i = self.model_keys.index(self.model)
            self.model = self.model_keys[(i + 1) % len(self.model_keys)]
            self._draw()
        elif key in ("n", "p"):
            step = 1 if key == "n" else -1
            self.index = (self.index + step) % len(self.names)
            self._load()
        elif key == "c" and self.pixel is not None:
            fig = render.figure_correspondence(
                self.scene, self.preds[self.model], *self.pixel)
            fig.show()
        elif key == "3":
            fig = render.figure_point_cloud(self.scene, self.preds[self.model])
            if fig is None:
                print(self.scene.calibration_note)
            else:
                fig.show()
        elif key == "s":
            out = render.save_scene_report(self.scene, self.preds, self.out_root)
            print("saved " + str(out))

    def run(self) -> None:
        print(self.HELP)
        self.plt.show()


def rank_scenes(runner: core.ModelRunner, split: str, model_keys: list[str],
                out_root: Path, no_save: bool) -> None:
    """Score every scene in the split, so scenes can be chosen by evidence.

    NEW COMPUTATION, stated rather than smuggled in: per-scene disparity
    metrics pooled over that scene's gt > 0 pixels, full 368x1232 frames, one
    scene at a time. This is *not* the recorded training-time protocol (10
    scenes, pooled together), and the numbers are not comparable to it. It
    exists to pick scenes to look at, not to restate the H1 result.
    """
    rows = []
    for index in range(len(core.scene_names(split))):
        scene = core.load_scene(index, split=split)
        row = {"index": index, "scene": scene.name,
               "gt_valid_pixels": int(scene.gt_valid.sum())}
        for key in model_keys:
            m = core.scene_metrics(runner.predict(key, scene).disparity, scene)
            row[key] = {"epe": m["epe"], "d1": m["d1"]}
        if len(model_keys) == 2:
            a, b = model_keys
            row["delta_d1"] = row[b]["d1"] - row[a]["d1"]
            row["delta_epe"] = row[b]["epe"] - row[a]["epe"]
        rows.append(row)

    key_fn = (lambda r: r.get("delta_d1", 0.0)) if len(model_keys) == 2 else (
        lambda r: r[model_keys[0]]["d1"])
    rows.sort(key=key_fn)
    header = "  ".join("{:>8} {:>7}".format(k + " EPE", k + " D1") for k in model_keys)
    print("\nper-scene ranking, {} ({} scenes) -- {}".format(
        split, len(rows), "sorted by D1 gap" if len(model_keys) == 2 else "sorted by D1"))
    print("{:>4}  {:<14} {:>9}  {}  {:>9} {:>9}".format(
        "idx", "scene", "gt px", header, "dD1", "dEPE"))
    for r in rows:
        cells = "  ".join("{:>8.3f} {:>7.2f}".format(r[k]["epe"], r[k]["d1"])
                          for k in model_keys)
        print("{:>4}  {:<14} {:>9}  {}  {:>9}  {:>9}".format(
            r["index"], r["scene"], r["gt_valid_pixels"], cells,
            core.fmt(r.get("delta_d1"), "", 2), core.fmt(r.get("delta_epe"), "", 3)))
    if len(model_keys) == 2:
        a, b = model_keys
        deltas = [r["delta_d1"] for r in rows]
        favour = sum(1 for d in deltas if d < 0)
        print("\n{} has lower D1 in {}/{} scenes; mean dD1 {:+.3f} pt, "
              "median {:+.3f} pt (per-scene protocol, not the recorded one)".format(
                  b, favour, len(deltas), float(np.mean(deltas)), float(np.median(deltas))))
    if not no_save:
        out_root.mkdir(parents=True, exist_ok=True)
        path = out_root / "scene_ranking_{}.json".format(split)
        path.write_text(json.dumps(
            {"protocol": ("per-scene disparity metrics, pooled over that scene's "
                          "gt > 0 pixels, full 368x1232 frames; NOT the recorded "
                          "10-scene training-time validation protocol"),
             "split": split, "models": model_keys,
             "git_revision": core.git_revision(), "scenes": rows}, indent=2),
            encoding="utf-8")
        print("wrote " + str(path))


def parse_ints(text: str, count: int, label: str) -> list[int]:
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != count:
        raise SystemExit("--{} needs {} integers, got '{}'".format(label, count, text))
    return [int(p) for p in parts]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=None,
                    help="scene index within the split, or a name like 000160")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="several scenes (indices or names)")
    ap.add_argument("--all", action="store_true", help="every scene in the split")
    ap.add_argument("--split", default="hailo_val", choices=["hailo_val", "hailo_calib", "all"])
    ap.add_argument("--models", nargs="+", default=["BASE", "WORKING"],
                    choices=sorted(core.MODELS))
    ap.add_argument("--demo", action="store_true",
                    help="show the current demonstration model (core.DEMO_MODEL, "
                         "presently {}) instead of the H1 arms; what the desktop "
                         "shortcut uses, so the shortcut never names a "
                         "model".format(core.DEMO_MODEL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--probe", default=None, help="pixel to inspect, 'x,y'")
    ap.add_argument("--region", default=None, help="region to inspect, 'x1,y1,x2,y2'")
    ap.add_argument("--point-cloud", type=int, default=0, metavar="STRIDE",
                    help="also save a back-projected point cloud at this pixel stride")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="print numbers, write no files")
    ap.add_argument("--list", action="store_true", help="list the scenes and exit")
    ap.add_argument("--rank", action="store_true",
                    help="score every scene in the split with each model and rank them "
                         "by the BASE-WORKING D1 gap, to choose scenes to look at")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if present)")
    args = ap.parse_args()
    if args.demo:
        args.models = [core.DEMO_MODEL]

    if args.list:
        names = core.scene_names(args.split)
        for i, name in enumerate(names):
            print("{:>3}  {}".format(i, name))
        print("{} scenes in split '{}'".format(len(names), args.split))
        return

    runner = core.ModelRunner(device=args.device)
    out_root = Path(args.out)

    if args.rank:
        rank_scenes(runner, args.split, list(args.models), out_root, args.no_save)
        return

    if args.interactive:
        first = args.scene if args.scene is not None else 0
        index = int(first) if str(first).isdigit() else (
            core.load_scene(str(first), split=args.split).index)
        Viewer(runner, args.split, index, list(args.models), out_root).run()
        return

    selectors: list = []
    if args.all:
        selectors = list(range(len(core.scene_names(args.split))))
    elif args.scenes:
        selectors = [int(s) if s.isdigit() else s for s in args.scenes]
    elif args.scene is not None:
        selectors = [int(args.scene) if str(args.scene).isdigit() else args.scene]
    else:
        selectors = [0]

    for selector in selectors:
        scene = core.load_scene(selector, split=args.split)
        preds = {k: runner.predict(k, scene) for k in args.models}
        print_scene_panel(scene, preds)

        if args.probe:
            x, y = parse_ints(args.probe, 2, "probe")
            probe = core.probe_pixel(scene, preds, x, y)
            print_pixel_probe(probe)
            if not args.no_save:
                out = out_root / Path(scene.name).stem
                out.mkdir(parents=True, exist_ok=True)
                for key, pred in preds.items():
                    fig = render.figure_correspondence(scene, pred, x, y)
                    path = out / "correspondence_{}_{}x{}.png".format(key.lower(), x, y)
                    fig.savefig(path, bbox_inches="tight")
                    render.plt.close(fig)
                (out / "probe_{}x{}.json".format(x, y)).write_text(
                    json.dumps(probe, indent=2), encoding="utf-8")

        if args.region:
            x1, y1, x2, y2 = parse_ints(args.region, 4, "region")
            for key, pred in preds.items():
                stats = core.region_stats(scene, pred, x1, y1, x2, y2)
                print("\nRegion {} [{}]".format(stats["region"], key))
                for k, v in stats.items():
                    if k in ("region", "model"):
                        continue
                    print("  {:<28} {}".format(
                        k, v if isinstance(v, (int, str)) else core.fmt(v)))

        if not args.no_save:
            out = render.save_scene_report(
                scene, preds, out_root, point_cloud_stride=args.point_cloud)
            print("\nwrote " + str(out))


if __name__ == "__main__":
    main()
