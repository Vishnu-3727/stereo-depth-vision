"""Stereo Depth Visualizer V1 -- the drawing half.

Rules this module enforces, because they are the difference between a picture
and a misleading picture:

- invalid pixels are drawn in a distinct grey and named in the legend; they are
  never drawn as a zero, a black region, or a plausible-looking depth;
- every map carries a colour bar with real units, and the valid data range is
  printed on the panel;
- when the display range is clipped, the panel says so and says how many pixels
  fell outside it;
- a difference is labelled a difference, never an improvement.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # overridden by the interactive viewer before it draws
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import core  # noqa: E402

INVALID_COLOUR = "#5a5a5a"
DISPARITY_CMAP = "magma"
DEPTH_CMAP = "viridis_r"   # near = bright, far = dark
ERROR_CMAP = "inferno"
DIFF_CMAP = "coolwarm"


def _masked(values: np.ndarray, valid: np.ndarray | None) -> np.ma.MaskedArray:
    v = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(v)
    if valid is not None:
        bad |= ~np.asarray(valid, dtype=bool)
    return np.ma.masked_array(v, mask=bad)


def _range(values: np.ndarray, valid: np.ndarray | None, lo=2.0, hi=98.0):
    m = _masked(values, valid).compressed()
    if m.size == 0:
        return 0.0, 1.0, 0
    vmin, vmax = float(np.percentile(m, lo)), float(np.percentile(m, hi))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    clipped = int(((m < vmin) | (m > vmax)).sum())
    return vmin, vmax, clipped


def show_map(
    ax,
    values: np.ndarray,
    valid: np.ndarray | None,
    title: str,
    unit: str,
    cmap: str = DISPARITY_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    symmetric: bool = False,
    colorbar: bool = True,
) -> tuple[float, float]:
    """Draw one scalar map with a colour bar, an explicit range and a clip note."""
    m = _masked(values, valid)
    auto_min, auto_max, clipped = _range(values, valid)
    if symmetric:
        span = max(abs(auto_min), abs(auto_max), 1e-6)
        auto_min, auto_max = -span, span
    vmin = auto_min if vmin is None else vmin
    vmax = auto_max if vmax is None else vmax

    cm = plt.get_cmap(cmap).with_extremes(bad=INVALID_COLOUR)
    im = ax.imshow(m, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    if m.count():
        data_min, data_max = float(m.min()), float(m.max())
        note = "valid {:.2f}..{:.2f} {}  |  {} px valid".format(
            data_min, data_max, unit, m.count()
        )
        if clipped:
            note += "  |  display clipped to {:.2f}..{:.2f}, {} px outside".format(
                vmin, vmax, clipped
            )
    else:
        note = "no valid pixels"
    if m.mask.any():
        note += "  |  grey = invalid/no data"
    ax.set_xlabel(note, fontsize=6)

    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cb.set_label(unit, fontsize=7)
        cb.ax.tick_params(labelsize=6)
    return vmin, vmax


def show_rgb(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def show_overlay(
    ax,
    rgb: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray | None,
    title: str,
    unit: str,
    cmap: str = DEPTH_CMAP,
    alpha: float = 0.75,
    colorbar: bool = True,
):
    """Scalar map painted over the left image, so structure and value line up."""
    ax.imshow(rgb)
    m = _masked(values, valid)
    vmin, vmax, clipped = _range(values, valid)
    # invalid stays fully transparent, so the RGB underneath shows through
    cm = plt.get_cmap(cmap).with_extremes(bad=(0, 0, 0, 0))
    im = ax.imshow(m, cmap=cm, vmin=vmin, vmax=vmax, alpha=alpha, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        "overlay {:.2f}..{:.2f} {}  |  transparent = invalid".format(vmin, vmax, unit)
        + ("  |  {} px outside display range".format(clipped) if clipped else ""),
        fontsize=6,
    )
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cb.set_label(unit, fontsize=7)
        cb.ax.tick_params(labelsize=6)
    return im


def error_map(pred: np.ndarray, scene: core.Scene) -> tuple[np.ndarray, np.ndarray]:
    """|pred - gt|, valid only where ground truth exists."""
    valid = scene.gt_valid
    err = np.abs(pred.astype(np.float64) - scene.gt_disparity.astype(np.float64))
    return np.where(valid, err, np.nan), valid


def save_map_png(
    path: Path,
    values: np.ndarray,
    valid: np.ndarray | None,
    title: str,
    unit: str,
    cmap: str,
    symmetric: bool = False,
) -> None:
    h, w = np.asarray(values).shape
    fig, ax = plt.subplots(figsize=(w / 110.0, h / 110.0 + 0.9), dpi=130)
    show_map(ax, values, valid, title, unit, cmap=cmap, symmetric=symmetric)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_rgb_png(path: Path, image: np.ndarray, title: str) -> None:
    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 110.0, h / 110.0 + 0.5), dpi=130)
    show_rgb(ax, image, title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_comparison(scene: core.Scene, predictions: dict[str, core.Prediction]):
    """The one-page view: inputs, ground truth, both arms, errors, depth, difference."""
    keys = list(predictions)
    rows = 4
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4.0 * rows), dpi=110)
    fig.suptitle(
        "{}  ({} #{})  --  {}\n{}".format(
            scene.name, scene.split, scene.index,
            " vs ".join(keys) if len(keys) > 1 else keys[0],
            scene.calibration_note,
        ),
        fontsize=10,
    )

    show_rgb(axes[0, 0], scene.left, "left image (model reference)")
    show_rgb(axes[0, 1], scene.right, "right image")

    # A shared disparity scale, so the two arms are visually comparable.
    gt_lo, gt_hi, _ = _range(scene.gt_disparity, scene.gt_valid)
    show_map(
        axes[1, 0], scene.gt_disparity, scene.gt_valid,
        "ground-truth disparity (disp_occ_0 / {:g})".format(scene.disparity_scale),
        "px", DISPARITY_CMAP, vmin=gt_lo, vmax=gt_hi,
    )
    first = predictions[keys[0]]
    show_map(
        axes[1, 1], first.disparity, None,
        "predicted disparity -- {}".format(keys[0]), "px",
        DISPARITY_CMAP, vmin=gt_lo, vmax=gt_hi,
    )
    if len(keys) > 1:
        show_map(
            axes[2, 0], predictions[keys[1]].disparity, None,
            "predicted disparity -- {}".format(keys[1]), "px",
            DISPARITY_CMAP, vmin=gt_lo, vmax=gt_hi,
        )
        diff = predictions[keys[1]].disparity.astype(np.float64) - first.disparity
        show_map(
            axes[2, 1], diff, None,
            "difference: {} - {} disparity (a difference, not an improvement)".format(
                keys[1], keys[0]
            ),
            "px", DIFF_CMAP, symmetric=True,
        )
    else:
        depth, ok = core.depth_from_disparity(first.disparity, scene.calibration)
        if depth is None:
            axes[2, 0].text(0.5, 0.5, scene.calibration_note, ha="center", wrap=True)
            axes[2, 0].axis("off")
        else:
            show_map(axes[2, 0], depth, ok, "predicted depth -- " + keys[0], "m", DEPTH_CMAP)
        axes[2, 1].axis("off")

    err0, valid = error_map(first.disparity, scene)
    e_lo, e_hi, _ = _range(err0, valid)
    show_map(
        axes[3, 0], err0, valid,
        "|error| -- {} (grey = no ground truth, not zero error)".format(keys[0]),
        "px", ERROR_CMAP, vmin=e_lo, vmax=e_hi,
    )
    if len(keys) > 1:
        err1, _ = error_map(predictions[keys[1]].disparity, scene)
        show_map(
            axes[3, 1], err1, valid,
            "|error| -- {} (same scale)".format(keys[1]), "px",
            ERROR_CMAP, vmin=e_lo, vmax=e_hi,
        )
    else:
        axes[3, 1].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def figure_correspondence(
    scene: core.Scene, pred: core.Prediction, x: int, y: int, half_width: int = 160
):
    """The predicted match for one pixel, drawn on both images.

    Both panels show the same row band at the same scale, so the horizontal
    offset on the page is the disparity.
    """
    d = float(pred.disparity[y, x])
    xr = core.correspondence(x, d)
    gt_d = float(scene.gt_disparity[y, x])
    gt_xr = core.correspondence(x, gt_d) if gt_d > 0 else None

    h, w = scene.shape
    x0, x1 = max(0, x - half_width), min(w, x + half_width)
    y0, y1 = max(0, y - half_width // 2), min(h, y + half_width // 2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=120)
    axes[0].imshow(scene.left[y0:y1, x0:x1])
    axes[0].plot(x - x0, y - y0, "o", mfc="none", mec="#39ff14", ms=12, mew=2)
    axes[0].set_title("left: selected pixel ({}, {})".format(x, y), fontsize=9)

    axes[1].imshow(scene.right[y0:y1, x0:x1])
    axes[1].plot(
        xr - x0, y - y0, "o", mfc="none", mec="#39ff14", ms=12, mew=2,
        label="predicted match  x_r = x_l - d = {:.1f}  (d = {:.2f} px)".format(xr, d),
    )
    if gt_xr is not None:
        axes[1].plot(
            gt_xr - x0, y - y0, "x", color="#00b0ff", ms=12, mew=2,
            label="ground-truth match  x_r = {:.1f}  (d = {:.2f} px)".format(gt_xr, gt_d),
        )
    axes[1].axvline(x - x0, color="w", lw=0.6, ls=":", label="x_l (no-disparity column)")
    axes[1].legend(fontsize=7, loc="lower left")
    axes[1].set_title("right: predicted correspondence -- {}".format(pred.model_key), fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "correspondence, left-referenced convention x_right = x_left - d", fontsize=10
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_point_cloud(scene: core.Scene, pred: core.Prediction, stride: int = 6):
    """A plain scatter of the back-projected points. V1 needs no 3D engine."""
    cloud = core.point_cloud(pred.disparity, scene, stride=stride)
    if cloud is None:
        return None
    fig = plt.figure(figsize=(10, 7), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        cloud[:, 0], cloud[:, 2], -cloud[:, 1],
        c=np.clip(cloud[:, 3:6], 0, 1), s=3, marker=".", linewidths=0,
    )
    # frame on the bulk of the points; a handful of far outliers would otherwise
    # squeeze the whole scene into a corner
    ax.set_xlim(*np.percentile(cloud[:, 0], [1, 99]))
    ax.set_ylim(0.0, float(np.percentile(cloud[:, 2], 99)))
    ax.set_zlim(*np.percentile(-cloud[:, 1], [1, 99]))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z depth (m)")
    ax.set_zlabel("-Y (m)")
    ax.set_title(
        "back-projection -- {} ({} points, stride {}, <= 80 m)\n{}".format(
            pred.model_key, len(cloud), stride, scene.calibration_note
        ),
        fontsize=9,
    )
    ax.view_init(elev=18, azim=-70)
    return fig


def figure_flip_map(scene: core.Scene, base: np.ndarray, work: np.ndarray,
                    labels: tuple[str, str] = ("BASE", "WORKING")):
    """Which pixels change D1 status between two predictions.

    D1 is a threshold count, so the difference between two arms is entirely a
    set of pixels crossing that threshold in one direction or the other. This
    draws that set: green = the second arm turned an outlier into an inlier,
    red = it did the opposite. Pixels without ground truth are grey; pixels
    where both arms agree are muted so the flips stand out.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    gt = scene.gt_disparity.astype(np.float64)
    valid = scene.gt_valid
    eb, ew = np.abs(base - gt), np.abs(work - gt)
    ob = (eb > 3.0) & (eb > 0.05 * gt)
    ow = (ew > 3.0) & (ew > 0.05 * gt)

    # 0 no GT, 1 both inlier, 2 both outlier, 3 fixed, 4 broken
    code = np.zeros(scene.shape, dtype=np.uint8)
    code[valid & ~ob & ~ow] = 1
    code[valid & ob & ow] = 2
    code[valid & ob & ~ow] = 3
    code[valid & ~ob & ow] = 4
    counts = [int((code == i).sum()) for i in range(5)]

    cmap = ListedColormap(["#3a3a3a", "#cfe8cf", "#7a1f1f", "#22d422", "#ff2d2d"])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), dpi=120)
    show_rgb(axes[0], scene.left, "left image -- " + scene.name)
    axes[1].imshow(code, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title(
        "D1 status change, {} -> {} (net {:+d} pixels)".format(
            labels[0], labels[1], counts[3] - counts[4]), fontsize=10)
    names = ["no ground truth", "both inlier", "both outlier",
             "{} fixed".format(labels[1]), "{} broke".format(labels[1])]
    axes[1].legend(
        handles=[Patch(facecolor=c, label="{} ({} px)".format(n, k))
                 for c, n, k in zip(cmap.colors, names, counts)],
        fontsize=7, loc="upper right", framealpha=0.85, ncol=2)
    fig.tight_layout()
    return fig


def figure_stage_comparison(scene: core.Scene, stages: dict[str, dict[str, np.ndarray]]):
    """What each arm's matching stage hands to refinement, and what comes out.

    Left column: ``disparity_initial`` -- the soft-argmin output, in units of
    disparity candidates (0..11); this is everything the cost volume
    contributes to the answer. Middle: the refinement residual, in pixels.
    Right: the final disparity. A constant left panel means the matching stage
    contributed no spatial information at all for that arm.
    """
    keys = list(stages)
    fig, axes = plt.subplots(len(keys), 3, figsize=(18, 3.4 * len(keys)), dpi=110)
    axes = np.atleast_2d(axes)
    for row, key in enumerate(keys):
        s = stages[key]
        init = s["disparity_initial"]
        title = "{}: disparity_initial (candidates 0-11)".format(key)
        if float(init.std()) == 0.0:
            title += "  --  CONSTANT {:.3f}, no spatial information".format(float(init.mean()))
        show_map(axes[row, 0], init, None, title, "candidate", DISPARITY_CMAP)
        show_map(axes[row, 1], s["refinement_residual"], None,
                 "{}: refinement residual".format(key), "px", DIFF_CMAP, symmetric=True)
        show_map(axes[row, 2], s["disparity_final"], None,
                 "{}: final disparity".format(key), "px", DISPARITY_CMAP)
    fig.suptitle(
        "matching output vs refinement contribution -- {} "
        "(final = initial + residual)".format(scene.name), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def save_scene_report(
    scene: core.Scene,
    predictions: dict[str, core.Prediction],
    out_root: Path,
    point_cloud_stride: int = 0,
) -> Path:
    """Write the per-scene artefact set described in the V1 specification."""
    out = Path(out_root) / Path(scene.name).stem
    out.mkdir(parents=True, exist_ok=True)

    save_rgb_png(out / "rgb_left.png", scene.left, "left image -- " + scene.name)
    save_rgb_png(out / "rgb_right.png", scene.right, "right image -- " + scene.name)
    save_map_png(
        out / "disparity_gt.png", scene.gt_disparity, scene.gt_valid,
        "ground-truth disparity -- " + scene.name, "px", DISPARITY_CMAP,
    )
    gt_depth, gt_depth_ok = scene.gt_depth()
    if gt_depth is not None:
        save_map_png(
            out / "depth_gt.png", gt_depth, gt_depth_ok,
            "ground-truth depth -- " + scene.name, "m", DEPTH_CMAP,
        )

    for key, pred in predictions.items():
        low = key.lower()
        save_map_png(
            out / "disparity_{}.png".format(low), pred.disparity, None,
            "predicted disparity -- {}".format(key), "px", DISPARITY_CMAP,
        )
        err, valid = error_map(pred.disparity, scene)
        save_map_png(
            out / "error_{}.png".format(low), err, valid,
            "|pred - gt| disparity -- {} (grey = no ground truth)".format(key),
            "px", ERROR_CMAP,
        )
        depth, ok = core.depth_from_disparity(pred.disparity, scene.calibration)
        if depth is not None:
            save_map_png(
                out / "depth_{}.png".format(low), depth, ok,
                "predicted depth -- {}".format(key), "m", DEPTH_CMAP,
            )
            h, w = scene.shape
            fig, ax = plt.subplots(figsize=(w / 110.0, h / 110.0 + 0.9), dpi=130)
            show_overlay(
                ax, scene.left, depth, ok,
                "depth over left image -- {}".format(key), "m",
            )
            fig.tight_layout()
            fig.savefig(out / "depth_overlay_{}.png".format(low), bbox_inches="tight")
            plt.close(fig)
        if gt_depth is not None and depth is not None:
            both = ok & (gt_depth_ok if gt_depth_ok is not None else ok)
            save_map_png(
                out / "depth_error_{}.png".format(low),
                np.where(both, np.abs(depth - gt_depth), np.nan), both,
                "|depth error| -- {}".format(key), "m", ERROR_CMAP,
            )
        if point_cloud_stride:
            fig = figure_point_cloud(scene, pred, stride=point_cloud_stride)
            if fig is not None:
                fig.savefig(out / "pointcloud_{}.png".format(low), bbox_inches="tight")
                plt.close(fig)

    keys = list(predictions)
    if len(keys) > 1:
        a, b = predictions[keys[0]], predictions[keys[1]]
        save_map_png(
            out / "difference_disparity.png",
            b.disparity.astype(np.float64) - a.disparity,
            None,
            "{} - {} disparity (difference, not improvement)".format(keys[1], keys[0]),
            "px", DIFF_CMAP, symmetric=True,
        )
        err_a, valid = error_map(a.disparity, scene)
        err_b, _ = error_map(b.disparity, scene)
        save_map_png(
            out / "difference_error.png",
            np.where(valid, err_b - err_a, np.nan), valid,
            "|error| {} - |error| {} (negative = {} closer to ground truth here)".format(
                keys[1], keys[0], keys[1]
            ),
            "px", DIFF_CMAP, symmetric=True,
        )

    fig = figure_comparison(scene, predictions)
    fig.savefig(out / "comparison.png", bbox_inches="tight")
    plt.close(fig)

    meta = core.metadata(scene, predictions)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def demo() -> None:
    """Self-check: invalid pixels stay masked and are never coloured as data."""
    values = np.array([[1.0, 2.0], [np.nan, 4.0]])
    valid = np.array([[True, False], [True, True]])
    m = _masked(values, valid)
    assert m.mask.tolist() == [[False, True], [True, False]], m.mask
    assert m.count() == 2

    lo, hi, clipped = _range(np.array([[0.0, 10.0, 100.0, 1000.0]]), None)
    assert lo < hi and clipped >= 0

    fig, ax = plt.subplots()
    vmin, vmax = show_map(ax, values, valid, "t", "px")
    assert vmin < vmax
    plt.close(fig)

    # an all-invalid map must draw without raising and without inventing a range
    fig, ax = plt.subplots()
    show_map(ax, np.full((4, 4), np.nan), np.zeros((4, 4), bool), "empty", "px")
    plt.close(fig)
    print("render self-check passed")


if __name__ == "__main__":
    demo()
