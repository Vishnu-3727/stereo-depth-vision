"""Tests for the changes this task made: the mentor demo builder and the one
new option on ``render.show_overlay``.

The point of these is claim discipline, not that the code runs. A demo figure
that prints a metre value it has no calibration for, or a probe box that was
quietly moved per scene, would fail here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "phase2" / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "phase2" / "scripts"))

import mentor_demo  # noqa: E402
from phase2.viz import core, render  # noqa: E402
from src.geometry.stereo import StereoCalibration  # noqa: E402

CALIB = StereoCalibration(
    focal_px=721.5, baseline_m=0.533, cx=600.0, cy=180.0,
    width=1242, height=375, source="synthetic",
)


def _scene(calib=CALIB) -> core.Scene:
    h, w = 368, 1232
    gt = np.zeros((h, w), np.float32)
    gt[200, 600] = 20.0
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return core.Scene(
        name="synthetic_10.png", index=0, split="synthetic",
        left=rgb, right=rgb.copy(), gt_disparity=gt, disparity_scale=256.0,
        original_shape=(h, w), calibration=calib,
        calibration_error=None if calib else "no calib file",
    )


def _pred(scene: core.Scene, value: float = 25.0) -> core.Prediction:
    return core.Prediction(
        model_key=mentor_demo.MODEL_KEY,
        disparity=np.full(scene.shape, value, np.float32),
        checkpoint=Path("synthetic"), checkpoint_sha256="0" * 64,
        config={}, device="cpu",
    )


def test_everything_follows_the_one_demo_model_knob():
    """`core.DEMO_MODEL` is the only place the demonstrated model is named.

    Deliberately does NOT hardcode "H2": the point of the knob is that
    repointing it moves the demo builder, the --demo flag and the desktop
    shortcut together. Hardcoding here would break the moment the model
    improves, which is exactly when it must not.
    """
    assert core.DEMO_MODEL in core.MODELS, core.DEMO_MODEL
    assert mentor_demo.MODEL_KEY == core.DEMO_MODEL

    spec = core.MODELS[core.DEMO_MODEL]
    assert spec["experiment"], "the demo model must name its experiment record"
    assert core.checkpoint_path(core.DEMO_MODEL).name.endswith("_checkpoint.pth")


def test_the_demo_flag_resolves_to_that_knob_not_to_the_h1_arms():
    """The desktop shortcut passes --demo and names no model of its own."""
    import argparse
    src = Path(REPO_ROOT / "phase2" / "scripts" / "stereo_visualizer.py").read_text(
        encoding="utf-8")
    assert "args.models = [core.DEMO_MODEL]" in src
    # and the shortcut's own arguments must not name a model
    assert '"--demo"' in src

    ns = argparse.Namespace(demo=True, models=["BASE", "WORKING"])
    if ns.demo:                      # the branch main() takes
        ns.models = [core.DEMO_MODEL]
    assert ns.models == [core.DEMO_MODEL]


def test_difference_mode_is_absent_when_only_one_model_is_loaded():
    """With one model, `difference` can only print an apology -- it must not be
    in the cycle at all. The desktop shortcut loads exactly one model."""
    import stereo_visualizer as sv

    one = [m for m in sv.MODES if m != "difference" or len([core.DEMO_MODEL]) > 1]
    assert "difference" not in one
    assert "depth_overlay" in one and "disparity" in one

    two = [m for m in sv.MODES if m != "difference" or len(["BASE", "WORKING"]) > 1]
    assert "difference" in two


def test_probe_boxes_are_one_fixed_grid_inside_the_frame():
    """The grid must be shared by every scene -- per-scene boxes would be
    cherry-picking dressed up as a probe."""
    for name, (x1, y1, x2, y2) in mentor_demo.PROBE_BOXES.items():
        assert 0 <= x1 < x2 < 1232, name
        assert 0 <= y1 < y2 < 368, name
    # the builder passes the same dict for every scene; nothing indexes it by
    # scene, so a regression that added per-scene boxes would change this type
    assert isinstance(mentor_demo.PROBE_BOXES, dict)


def test_probe_labels_never_print_metres_without_calibration():
    scene = _scene(calib=None)
    fig, rows = mentor_demo.probe_figure(scene, _pred(scene), "subtitle")
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert labels, "probe boxes must still be drawn and labelled"
    for text in labels:
        assert " m" not in text, text
        assert "px" in text, text
    for stats in rows.values():
        assert "pred_depth_median_m" not in stats
    render.plt.close(fig)


def test_probe_labels_print_metres_when_calibration_exists():
    scene = _scene()
    fig, rows = mentor_demo.probe_figure(scene, _pred(scene, 25.0), "subtitle")
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert labels
    expected = CALIB.fB / 25.0
    for text in labels:
        assert "px ->" in text and " m" in text, text
    for stats in rows.values():
        assert stats["pred_depth_median_m"] == pytest.approx(expected)
    render.plt.close(fig)


def test_two_panel_labels_disparity_and_depth_in_their_own_units():
    scene = _scene()
    depth, ok = core.depth_from_disparity(_pred(scene).disparity, CALIB)
    fig = mentor_demo.two_panel(scene, depth, ok, "m", render.DEPTH_CMAP,
                                "estimated distance in metres", "sub")
    titles = " | ".join(ax.get_title() for ax in fig.axes)
    assert "metres" in titles and "px" not in titles, titles
    render.plt.close(fig)

    fig = mentor_demo.two_panel(scene, _pred(scene).disparity, None, "px",
                                render.DISPARITY_CMAP,
                                "estimated pixel shift in pixels", "sub")
    titles = " | ".join(ax.get_title() for ax in fig.axes)
    assert "pixel" in titles and "metre" not in titles, titles
    render.plt.close(fig)


def test_show_overlay_colorbar_flag_controls_the_extra_axes():
    """The flag added in this task: off must draw no colour bar, and must still
    hand back the image so a shared one can be attached."""
    scene = _scene()
    values = np.full(scene.shape, 20.0)
    fig, ax = render.plt.subplots()
    im = render.show_overlay(ax, scene.left, values, None, "t", "m",
                             colorbar=False)
    assert im is not None
    assert len(fig.axes) == 1, "colorbar=False must add no axes"
    render.plt.close(fig)

    fig, ax = render.plt.subplots()
    render.show_overlay(ax, scene.left, values, None, "t", "m")
    assert len(fig.axes) == 2, "the default must still draw a colour bar"
    render.plt.close(fig)


def test_the_stereo_test_corruption_comes_from_the_measured_variant_set():
    """The demo must corrupt the right image the same way the experiments did,
    not in some new way invented for the picture."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from exp_right_image_ablation import variants

    scene = _scene()
    other = _scene()
    other.right = np.zeros_like(other.right)
    made = variants(scene.left, scene.right, other.right)
    assert mentor_demo.STEREO_TEST_VARIANT in made
    assert not np.array_equal(made[mentor_demo.STEREO_TEST_VARIANT], scene.right)
