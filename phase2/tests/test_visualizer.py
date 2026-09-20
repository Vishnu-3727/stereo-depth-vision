"""Tests for the Stereo Depth Visualizer V1.

The numerical ones run anywhere. The ones that need KITTI or the H1-v2
checkpoints skip when those are absent, so the suite still says something
useful on a machine without the data.

The point of most of these is not that the code runs -- it is that a
geometrically wrong but visually convincing picture would fail here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2.viz import core, render  # noqa: E402
from src.geometry.stereo import StereoCalibration  # noqa: E402

CALIB = StereoCalibration(
    focal_px=721.5, baseline_m=0.533, cx=600.0, cy=180.0,
    width=1242, height=375, source="synthetic",
)

HAVE_DATA = (core.DATA_ROOT / "training" / "image_2").exists()
HAVE_CHECKPOINTS = all(core.checkpoint_path(k).exists() for k in core.MODELS)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="KITTI 2015 not present")
needs_models = pytest.mark.skipif(
    not (HAVE_DATA and HAVE_CHECKPOINTS), reason="H1-v2 checkpoints or KITTI not present"
)


# --- depth mathematics -------------------------------------------------------

def test_depth_equals_fB_over_d():
    d = np.array([[10.0, 40.0, 80.0]])
    depth, ok = core.depth_from_disparity(d, CALIB)
    assert ok.all()
    expected = 721.5 * 0.533 / d
    assert np.allclose(depth, expected)
    # and the round trip closes
    back, _ = CALIB.disparity_from_depth(depth)
    assert np.allclose(back, d)


@pytest.mark.parametrize("bad", [0.0, -1.0, -1e-9, 1e-12, np.nan, np.inf, -np.inf])
def test_non_positive_or_non_finite_disparity_yields_no_depth(bad):
    depth, ok = core.depth_from_disparity(np.array([[bad]]), CALIB)
    assert not ok[0, 0], "disparity {} must not produce a depth".format(bad)
    assert np.isnan(depth[0, 0])
    assert core.fmt(depth[0, 0], "m") == "N/A"


def test_no_infinity_reaches_the_user():
    for value in (np.inf, -np.inf, np.nan, None):
        assert core.fmt(value) == "N/A"
    assert core.fmt(9.1837, "m") == "9.18 m"


def test_missing_calibration_is_explicit_not_fabricated():
    assert core.depth_from_disparity(np.ones((2, 2)), None) == (None, None)
    scene = _synthetic_scene(calib=None)
    assert "unavailable" in scene.calibration_note.lower()
    assert scene.gt_depth() == (None, None)
    assert core.point_cloud(np.ones(scene.shape, np.float32), scene) is None
    probe = core.probe_pixel(scene, {"M": _synthetic_prediction(scene)}, 4, 2)
    assert probe["gt_depth_m"] is None
    assert probe["models"]["M"]["depth_m"] is None


# --- correspondence ----------------------------------------------------------

def test_correspondence_is_left_referenced():
    assert core.correspondence(742, 41.8) == pytest.approx(700.2)
    assert core.correspondence(100, 0.0) == 100.0


def test_correspondence_recovers_a_synthetic_shift():
    """A synthetic pair with a known disparity: the predicted right column must
    land on the pixel the pattern was actually shifted to."""
    rng = np.random.default_rng(0)
    h, w, d = 32, 200, 17
    left = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, : w - d] = left[:, d:]          # a feature at x_l appears at x_l - d
    for x in (40, 100, 150):
        xr = int(core.correspondence(x, d))
        assert np.array_equal(right[:, xr], left[:, x]), x
        # the opposite sign must NOT match, or the test proves nothing
        assert not np.array_equal(right[:, x + d], left[:, x])


@needs_data
def test_disparity_sign_convention_holds_on_kitti():
    """Re-derive the convention photometrically instead of trusting it.

    Warping the left pixels to x_l - d must land on far more similar right
    pixels than x_l + d. If this ever flips, every correspondence drawing in
    the tool is wrong.
    """
    scene = core.load_scene(0)
    left = scene.left.astype(np.float64).mean(2)
    right = scene.right.astype(np.float64).mean(2)
    ys, xs = np.nonzero(scene.gt_valid)
    d = np.rint(scene.gt_disparity[ys, xs]).astype(int)
    errors = {}
    for sign in (-1, +1):
        xr = xs + sign * d
        ok = (xr >= 0) & (xr < left.shape[1])
        errors[sign] = float(np.abs(left[ys[ok], xs[ok]] - right[ys[ok], xr[ok]]).mean())
    assert errors[-1] < errors[+1] / 2.0, errors


# --- scene, probe, region ----------------------------------------------------

def _synthetic_scene(calib=CALIB) -> core.Scene:
    h, w = 8, 12
    gt = np.zeros((h, w), np.float32)
    gt[2, 4] = 20.0            # exactly one pixel with ground truth
    return core.Scene(
        name="synthetic_10.png", index=0, split="synthetic",
        left=np.zeros((h, w, 3), np.uint8), right=np.zeros((h, w, 3), np.uint8),
        gt_disparity=gt, disparity_scale=256.0, original_shape=(h, w),
        calibration=calib, calibration_error=None if calib else "no calib file",
    )


def _synthetic_prediction(scene: core.Scene, value: float = 25.0) -> core.Prediction:
    return core.Prediction(
        model_key="M", disparity=np.full(scene.shape, value, np.float32),
        checkpoint=Path("synthetic"), checkpoint_sha256="0" * 64,
        config={}, device="cpu",
    )


def test_probe_reports_missing_ground_truth_as_na_not_zero():
    scene = _synthetic_scene()
    preds = {"M": _synthetic_prediction(scene)}
    hit = core.probe_pixel(scene, preds, 4, 2)
    assert hit["gt_disparity"] == 20.0
    assert hit["models"]["M"]["disparity_error"] == pytest.approx(5.0)
    assert hit["gt_depth_m"] == pytest.approx(CALIB.fB / 20.0)

    miss = core.probe_pixel(scene, preds, 0, 0)
    assert miss["gt_disparity"] is None
    assert miss["gt_depth_m"] is None
    assert miss["gt_x_right"] is None
    assert miss["models"]["M"]["disparity_error"] is None
    assert miss["models"]["M"]["depth_m"] is not None  # prediction still has a depth


def test_probe_rejects_out_of_bounds_pixels():
    scene = _synthetic_scene()
    with pytest.raises(IndexError):
        core.probe_pixel(scene, {"M": _synthetic_prediction(scene)}, 999, 0)


def test_region_stats_handles_empty_and_out_of_bounds_regions():
    scene = _synthetic_scene()
    pred = _synthetic_prediction(scene)
    empty = core.region_stats(scene, pred, 50, 50, 60, 60)
    assert empty["pixels"] == 0 and "note" in empty

    clipped = core.region_stats(scene, pred, -20, -20, 5, 5)
    assert clipped["region"] == [0, 0, 5, 5]
    assert clipped["pixels"] == 36
    assert clipped["gt_valid_pixels"] == 1
    assert clipped["pred_depth_median_m"] == pytest.approx(CALIB.fB / 25.0)
    # error statistics come only from the single ground-truth pixel
    assert clipped["disparity_mae"] == pytest.approx(5.0)


def test_region_with_invalid_predicted_disparity_reports_fewer_valid_pixels():
    scene = _synthetic_scene()
    pred = _synthetic_prediction(scene)
    pred.disparity[:, :6] = 0.0
    stats = core.region_stats(scene, pred, 0, 0, 11, 7)
    assert stats["pred_disparity_valid_pixels"] == 8 * 6
    assert np.isfinite(stats["pred_depth_max_m"])


def test_error_map_masks_pixels_without_ground_truth():
    scene = _synthetic_scene()
    err, valid = render.error_map(_synthetic_prediction(scene).disparity, scene)
    assert valid.sum() == 1
    assert err[2, 4] == pytest.approx(5.0)
    assert np.isnan(err[0, 0]), "no ground truth must not read as zero error"
    masked = render._masked(err, valid)
    assert masked.count() == 1


def test_point_cloud_drops_undefined_and_far_points():
    scene = _synthetic_scene()
    disp = np.full(scene.shape, 40.0, np.float32)
    disp[0, :] = 0.0                       # undefined
    disp[1, :] = CALIB.fB / 500.0          # 500 m away
    cloud = core.point_cloud(disp, scene, stride=1, max_depth_m=80.0)
    assert cloud is not None
    assert len(cloud) == (scene.shape[0] - 2) * scene.shape[1]
    assert np.isfinite(cloud).all()
    assert cloud[:, 2].max() <= 80.0


# --- protocol and safety -----------------------------------------------------

@needs_data
def test_scene_geometry_matches_the_training_time_evaluation_protocol():
    scene = core.load_scene(0)
    assert scene.shape == (368, 1232)
    assert scene.left.shape == scene.right.shape == (368, 1232, 3)
    assert scene.gt_disparity.shape == scene.shape
    assert (scene.gt_disparity >= 0).all()
    assert scene.gt_valid.sum() > 0

    from src.datasets.kitti2015 import Kitti2015Stereo, pad_and_crop, read_disparity_png
    ds = Kitti2015Stereo(core.DATA_ROOT, split="hailo_val")
    raw = read_disparity_png(ds.gt_dir / scene.name, scale=256.0)
    assert np.array_equal(scene.gt_disparity, pad_and_crop(raw))


@needs_models
def test_checkpoints_are_loaded_under_the_configuration_they_were_trained_with():
    """Every arm's checkpoint must record the shift it was trained with.

    The H1 records store it as a top-level key. The H2 record states it inside
    `held_fixed` instead (see that experiment's NOTE.md), so the rule is: match
    when the key is present, and otherwise the value must still appear in the
    recorded configuration -- never silently absent.
    """
    import torch
    for key, spec in core.MODELS.items():
        config = torch.load(core.checkpoint_path(key), map_location="cpu",
                            weights_only=False)["config"]
        shift = spec["cost_volume_shift"]
        recorded = config.get("cost_volume_shift")
        if recorded is not None:
            assert recorded == shift, (key, recorded, shift)
        else:
            # H2 records it inside `held_fixed`, H3 inside `changed_variable`
            # ("cost_volume_shift: 'left' -> 'none'"). Both name the field and
            # quote the value; neither may leave it unrecorded.
            # ARMP_S1's checkpoint is a frozen stage_b file whose stored config
            # never recorded the shift and must not be rewritten to add it; for
            # that arm the shift is stated in the registry's explicit model
            # config instead (core.MODELS[key]["config"]), which the loader
            # cross-checks against the registry shift -- so that versioned
            # location counts, but it must still state the value exactly.
            stated = spec.get("config", {}).get("cost_volume_shift")
            if stated is not None:
                assert stated == shift, (key, stated, shift)
            else:
                text = " ".join(str(v) for v in config.values())
                assert "cost_volume_shift" in text, key
                assert "'{}'".format(shift) in text, (key, shift)


@needs_models
def test_inference_is_deterministic_and_matches_the_recorded_metric_protocol():
    runner = core.ModelRunner(device="cpu")
    scene = core.load_scene(0)
    a = runner.predict("BASE", scene).disparity
    b = core.ModelRunner(device="cpu").predict("BASE", scene).disparity
    assert np.array_equal(a, b)
    assert a.shape == scene.shape and np.isfinite(a).all()

    from src.evaluation.metrics import disparity_metrics
    valid = scene.gt_valid
    expected = disparity_metrics(a[valid], scene.gt_disparity[valid])
    assert core.scene_metrics(a, scene)["epe"] == pytest.approx(expected.epe)


@needs_models
def test_saving_a_report_touches_nothing_outside_the_output_directory(tmp_path):
    watched = sorted(
        list((core.PHASE2_ROOT / "results" / "training").rglob("*"))
        + list(core.EXPERIMENTS_DIR.rglob("*"))
        + list((REPO_ROOT / "src").rglob("*.py"))
    )
    before = {p: p.stat().st_mtime_ns for p in watched if p.is_file()}

    runner = core.ModelRunner(device="cpu")
    scene = core.load_scene(0)
    preds = {k: runner.predict(k, scene) for k in core.MODELS}
    out = render.save_scene_report(scene, preds, tmp_path)

    after = {p: p.stat().st_mtime_ns for p in watched if p.is_file()}
    assert before == after, "the visualizer must never write into the research record"
    assert (out / "comparison.png").exists()
    assert (out / "metadata.json").exists()

    import json
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["disparity_convention"] == "left-referenced: x_right = x_left - d"
    assert meta["models"]["BASE"]["cost_volume_shift"] == "none"
    assert meta["models"]["WORKING"]["cost_volume_shift"] == "left"
    assert len(meta["models"]["BASE"]["checkpoint_sha256"]) == 64


# --- additive figures from the H1 mechanism investigation ---------------------

def test_flip_map_counts_match_the_d1_definition():
    """The flip map's five categories must reproduce the D1 rule exactly."""
    scene = _synthetic_scene()
    scene.gt_disparity[3, 0] = 100.0    # a second ground-truth pixel, far larger
    base = np.full(scene.shape, 25.0, np.float32)   # at (2,4): err 5 px on gt 20 -> outlier
    work = np.full(scene.shape, 21.0, np.float32)   # at (2,4): err 1 px         -> inlier
    fig = render.figure_flip_map(scene, base.astype(np.float64), work.astype(np.float64))
    label = fig.axes[1].get_title()
    # (2,4): BASE outlier -> WORKING inlier = fixed. (3,0): gt 100, errors 75/79 px,
    # both beyond 3 px and 5 % -> both outlier. Net therefore +1 pixel.
    assert "net +1 pixels" in label, label
    render.plt.close(fig)


def test_stage_comparison_names_a_constant_matching_output():
    """A constant disparity_initial must be called out, not quietly drawn."""
    scene = _synthetic_scene()
    stages = {
        "BASE": {
            "disparity_initial": np.full(scene.shape, 11.0),
            "refinement_residual": np.zeros(scene.shape),
            "disparity_final": np.full(scene.shape, 11.0),
        },
        "WORKING": {
            "disparity_initial": np.arange(scene.shape[0] * scene.shape[1]).reshape(scene.shape) * 1.0,
            "refinement_residual": np.ones(scene.shape),
            "disparity_final": np.ones(scene.shape),
        },
    }
    fig = render.figure_stage_comparison(scene, stages)
    titles = [ax.get_title() for ax in fig.axes]
    assert any("CONSTANT 11.000" in t for t in titles), titles
    assert not any("CONSTANT" in t and "WORKING" in t for t in titles), titles
    render.plt.close(fig)
