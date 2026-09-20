"""Regression tests for EXP-H3-VIABILITY-001.

These exist to stop the two ways this experiment could lie: by quietly changing
something other than the cost-volume shift, and by reporting a LEFT-vs-NONE
difference that the complete pipeline does not actually produce.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2.models import scaled_regression  # noqa: E402
from phase2.viz import core  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

PREFLIGHT = REPO_ROOT / "phase2" / "results" / "h3_viability" / "preflight.json"
H3_RECORD = core.EXPERIMENTS_DIR / "EXP-H3-VIABILITY-001"

HAVE_DATA = (core.DATA_ROOT / "training" / "image_2").exists()
HAVE_H2 = core.checkpoint_path("H2").exists()
HAVE_H3 = core.checkpoint_path("H3").exists()
needs_h2 = pytest.mark.skipif(not HAVE_H2, reason="H2 checkpoint not present")
needs_h3 = pytest.mark.skipif(
    not (HAVE_H3 and HAVE_DATA), reason="H3 checkpoint or KITTI not present")


def _model(shift: str, state=None) -> StereoNet:
    model = StereoNet(StereoNetConfig(cost_volume_shift=shift))
    scaled_regression.apply_to(model)
    if state is not None:
        model.load_state_dict(state)
    return model.eval()


# --- 1 & 2: nothing that must not move, moved --------------------------------

@needs_h2
def test_h2_checkpoint_is_unchanged():
    """The control's hash must still match what the H3 record was built against."""
    recorded = json.loads((H3_RECORD / "config.json").read_text(encoding="utf-8"))
    expected = recorded.get("config", recorded).get("control_checkpoint_sha256")
    assert expected, "the H3 record must state which control checkpoint it used"
    assert core.sha256(core.checkpoint_path("H2")) == expected


def test_phase_1_is_unchanged():
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "phase-1-frozen", "--", "src", "scripts"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    except Exception as exc:
        pytest.skip("cannot check the frozen tag here: {}".format(exc))
    assert out.stdout.strip() == "", "Phase 1 changed:\n" + out.stdout


# --- 3: exactly one changed variable -----------------------------------------

def test_h3_differs_from_h2_only_in_the_cost_volume_shift():
    torch.manual_seed(0)
    h2 = _model("left")
    torch.manual_seed(0)
    h3 = _model("none")

    assert h2.cost_volume.shift == "left" and h3.cost_volume.shift == "none"
    # identical initialisation, identical parameter set, identical module types
    a, b = h2.state_dict(), h3.state_dict()
    assert set(a) == set(b)
    assert all(torch.equal(a[k], b[k]) for k in a), "initialisation must match"
    assert h2.parameter_count() == h3.parameter_count() == 423586
    assert type(h2.regression) is type(h3.regression) is (
        scaled_regression.StandardisedDisparityRegression)
    for attr in ("num_disparities", "method"):
        assert getattr(h2.cost_volume, attr) == getattr(h3.cost_volume, attr)
    # and every other stage is the same class
    for stage in ("feature_extractor", "aggregation", "refinement"):
        assert type(getattr(h2, stage)) is type(getattr(h3, stage))


def test_h3_record_states_the_changed_variable_and_the_control():
    config = json.loads((H3_RECORD / "config.json").read_text(encoding="utf-8"))
    config = config.get("config", config)
    assert "cost_volume_shift" in config["changed_variable"]
    assert "'left'" in config["changed_variable"] and "'none'" in config["changed_variable"]
    assert config["control"].startswith("EXP-H2-SOFTARGMIN-SCALE")


# --- 4: the complete LEFT vs NONE path really is different --------------------

@needs_h2
def test_complete_pipeline_distinguishes_left_from_none():
    """Same weights, same input, complete model: features must agree and
    everything downstream of the cost volume must differ."""
    state = torch.load(core.checkpoint_path("H2"), map_location="cpu",
                       weights_only=False)["model"]
    left_arm, none_arm = _model("left", state), _model("none", state)
    torch.manual_seed(0)
    left_img, right_img = torch.randn(1, 3, 64, 128), torch.randn(1, 3, 64, 128)

    with torch.no_grad():
        _, a = left_arm(left_img, right_img, return_stages=True)
        _, b = none_arm(left_img, right_img, return_stages=True)

    assert torch.equal(a["left_features"], b["left_features"])
    assert torch.equal(a["right_features"], b["right_features"])
    for stage in ("cost_volume", "aggregated_cost", "disparity_initial",
                  "disparity_final"):
        assert a[stage].shape == b[stage].shape, stage
        assert float((a[stage] - b[stage]).abs().max()) > 1e-6, stage


@needs_h2
def test_preflight_recorded_the_same_verdict():
    if not PREFLIGHT.exists():
        pytest.skip("preflight has not been run on this machine")
    gate = json.loads(PREFLIGHT.read_text(encoding="utf-8"))["gate"]
    assert gate["checkpoint_unchanged"] and gate["no_nan_or_inf"]
    assert gate["final_disparity_differs"] and gate["cost_volume_differs"]
    assert gate["PASS"] is True


# --- 5 & 6: the stereo probes still measure something -------------------------

@needs_h2
def test_right_image_ablation_is_still_meaningful_on_the_control():
    """If corrupting the right image stopped mattering for H2, every stereo
    conclusion in this project would be measuring nothing."""
    if not PREFLIGHT.exists():
        pytest.skip("preflight has not been run on this machine")
    data = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    penalties = [row[k]["d1_penalty"]
                 for row in data["stereo_gate"]["left"]
                 for k in ("right_black", "right_noise", "right_equals_left")]
    assert min(penalties) > 5.0, penalties


def test_matching_path_receives_gradient_through_the_swapped_stage():
    for shift in ("left", "none"):
        torch.manual_seed(0)
        model = _model(shift)
        model.train()
        model(torch.randn(1, 3, 64, 128), torch.randn(1, 3, 64, 128)).sum().backward()
        norm = sum(float(p.grad.norm()) ** 2
                   for n, p in model.named_parameters()
                   if n.startswith(("feature_extractor", "aggregation"))
                   and p.grad is not None) ** 0.5
        assert norm > 1e-12, (shift, norm)


# --- 7 & 8: outputs and tooling ----------------------------------------------

@needs_h3
def test_h3_predictions_are_finite_and_not_a_constant():
    runner = core.ModelRunner(device="cpu")
    scene = core.load_scene(0)
    pred = runner.predict("H3", scene).disparity
    assert np.isfinite(pred).all(), "no NaN or Inf may reach a prediction"
    assert pred.std() > 0.5, "a collapsed constant-disparity map would fail here"
    assert pred.min() >= 0.0


@needs_h3
def test_visualizer_handles_h3():
    """H3 must be usable through the existing visualizer, not a second one."""
    runner = core.ModelRunner(device="cpu")
    scene = core.load_scene(0)
    preds = {k: runner.predict(k, scene) for k in ("H2", "H3")}
    meta = core.metadata(scene, preds)
    assert meta["models"]["H3"]["cost_volume_shift"] == "none"
    assert meta["models"]["H2"]["cost_volume_shift"] == "left"
    assert len(meta["models"]["H3"]["checkpoint_sha256"]) == 64
    assert meta["scene_metrics"]["H3"]["valid_pixels"] > 0


# --- 9: the record says what a reader needs -----------------------------------

def test_experiment_record_is_complete():
    config = json.loads((H3_RECORD / "config.json").read_text(encoding="utf-8"))
    config = config.get("config", config)
    for key in ("hypothesis", "control", "control_checkpoint_sha256",
                "changed_variable", "held_fixed", "budget", "dataset", "split",
                "seed", "epochs", "schedule_T_max", "optimizer", "learning_rate",
                "loss", "validation_protocol", "preflight",
                "gradient_present_threshold", "known_limitation"):
        assert key in config, key

    metrics = json.loads((H3_RECORD / "metrics.json").read_text(encoding="utf-8"))
    metrics = metrics.get("metrics", metrics)
    for key in ("history", "aborted", "wall_clock_s", "checkpoint",
                "checkpoint_sha256", "parameter_count_unique",
                "primary_endpoint_matching_gradient", "gradient_norms"):
        assert key in metrics, key
    for epoch in (0, 1, 2, 5, 10):
        row = metrics["checkpoint_epoch_{}".format(epoch)]
        for key in ("val_epe", "val_d1", "val_loss", "stage_stats"):
            assert key in row, (epoch, key)
    assert metrics["gradient_norms"]["any_nan"] is False
    assert metrics["aborted"] is None
