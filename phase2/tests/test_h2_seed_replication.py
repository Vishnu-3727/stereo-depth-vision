"""Regression tests for EXP-H2-SEED-REPLICATION-001.

A replication is only worth anything if the thing being replicated did not move.
These tests pin that: the H2 configuration, the shift, the reference checkpoint,
frozen Phase 1, and the probes the verdict depends on.

They also pin the protocol correction, so the invalidated epoch-10 stereo gate
cannot quietly come back.
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
from phase2.scripts import exp_h2_seed_replication as R  # noqa: E402
from phase2.viz import core, render  # noqa: E402

RESULT_DIR = R.RESULT_DIR
REFERENCE_RECORD = core.EXPERIMENTS_DIR / R.REFERENCE_ID
CORRECTION_DOC = (REPO_ROOT / "phase2" / "docs"
                  / "EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md")

HAVE_DATA = (core.DATA_ROOT / "training" / "image_2").exists()
HAVE_H2 = core.checkpoint_path("H2").exists()
needs_h2 = pytest.mark.skipif(not (HAVE_H2 and HAVE_DATA),
                              reason="H2 checkpoint or KITTI not present")


def _record(experiment: str) -> tuple[dict, dict]:
    path = core.EXPERIMENTS_DIR / experiment
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    return config.get("config", config), metrics.get("metrics", metrics)


def _replication_records() -> list[str]:
    """Completed replication records only.

    A record's directory is created when its run starts but `metrics.json` is
    written when it ends, so an in-progress run must not be read as an
    incomplete record.
    """
    return sorted(p.name for p in core.EXPERIMENTS_DIR.iterdir()
                  if p.name.startswith(R.EXPERIMENT_PREFIX)
                  and (p / "metrics.json").exists())


# --- 1, 2, 3: the configuration and the one changed variable -------------------

def test_h2_reference_configuration_is_unchanged():
    """The run being replicated must still say what it said."""
    config, _ = _record(R.REFERENCE_ID)
    assert config["seed"] == 0
    assert config["epochs"] == 200
    assert config["batch_size"] == 2
    assert config["resolution"] == [256, 512]
    assert config["learning_rate"] == 1e-3
    assert config["precision"] == "fp32"
    assert "cost_volume_shift='left'" in config["held_fixed"]


def test_replication_builds_the_h2_architecture_with_shift_left():
    for seed in (0, 1, 2):
        model = R.build_model(seed, "cpu")
        assert model.cost_volume.shift == "left", seed
        assert isinstance(model.regression,
                          scaled_regression.StandardisedDisparityRegression)
        assert model.parameter_count() == 423586, seed


def test_only_the_seed_differs_between_replication_runs():
    a, b = R.recipe_config(1, "cpu"), R.recipe_config(2, "cpu")
    differing = [k for k in a if a[k] != b[k]]
    assert differing == ["seed"], differing
    # and different seeds really do produce different initial weights
    w1 = R.build_model(1, "cpu").state_dict()
    w2 = R.build_model(2, "cpu").state_dict()
    assert not all(torch.equal(w1[k], w2[k]) for k in w1)
    # while the same seed reproduces its own initialisation exactly
    again = R.build_model(1, "cpu").state_dict()
    assert all(torch.equal(w1[k], again[k]) for k in w1)


def test_recorded_replication_configs_differ_from_the_reference_only_by_seed():
    reference, _ = _record(R.REFERENCE_ID)
    for name in _replication_records():
        config, _ = _record(name)
        assert config["cost_volume_shift"] == "left", name
        assert config["epochs"] == reference["epochs"], name
        assert config["batch_size"] == reference["batch_size"], name
        assert config["learning_rate"] == reference["learning_rate"], name
        assert config["crop"] == reference["resolution"], name
        assert config["precision"] == reference["precision"], name
        assert config["changed_variable"].startswith("random seed only"), name


# --- 4, 5: nothing that must not move, moved ----------------------------------

@needs_h2
def test_seed0_checkpoint_is_unchanged():
    """Every replication record must have been built against this exact file."""
    digest = core.sha256(core.checkpoint_path("H2"))
    checked = 0
    for name in _replication_records():
        config, _ = _record(name)
        recorded = config.get("reference_checkpoint_sha256")
        if recorded:
            assert recorded == digest, name
            checked += 1
    assert checked > 0, "no replication record states the reference checkpoint hash"


def test_phase_1_is_unchanged():
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "phase-1-frozen", "--", "src", "scripts"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    except Exception as exc:
        pytest.skip("cannot check the frozen tag here: {}".format(exc))
    assert out.stdout.strip() == "", "Phase 1 changed:\n" + out.stdout


# --- the protocol correction must stay corrected ------------------------------

def test_the_invalidated_epoch_10_stereo_gate_cannot_come_back():
    """The epoch-10 gate must never enforce stereo dependence again.

    Control evidence: seed 0 fails that criterion at epoch 10 while reaching
    +80.8...+90.9 D1 points of dependence by epoch 200.
    """
    checkpoints = {"0": {"val_epe": 35.0}, "10": {"val_epe": 15.0}}
    ablation = [{
        "right_black": {"d1_penalty": -3.3}, "right_noise": {"d1_penalty": -2.0},
        "right_equals_left": {"d1_penalty": -1.0},
        "initial_constant_mean": {"d1_penalty": -5.9},
        "initial_shuffled": {"d1_penalty": -4.0},
    }]
    verdict = R._gate_verdict(checkpoints, ablation, matching_fraction=1.0)
    assert verdict["PASS"] is True, "the seed-0 pattern must pass the epoch-10 gate"
    assert verdict["stereo_dependence_enforced"] is False
    assert verdict["stereo_dependence_at_this_epoch"] is False  # recorded, not acted on
    assert R.STEREO_GATE_EPOCH == 200
    assert R.GATE_MIN_STEREO_D1_POINTS == 20.0


def test_the_correction_is_documented_and_frozen():
    text = CORRECTION_DOC.read_text(encoding="utf-8")
    for required in ("invalid", "20.0", "REPRODUCIBLE", "PARTIALLY REPRODUCIBLE",
                     "NOT REPRODUCIBLE", "same-seed", "SEED0-REFERENCE"):
        assert required in text, required


def test_the_200_epoch_criterion_separates_the_h1_and_h2_regimes():
    """The frozen threshold must fail a monocular model and pass a stereo one."""
    checkpoints = {"200": {"stage_stats": {"entropy": 1.7, "final_std": 9.0}}}
    monocular = [{
        "right_black": {"d1_penalty": 0.5}, "right_noise": {"d1_penalty": -1.0},
        "right_equals_left": {"d1_penalty": 1.9},
        "initial_constant_mean": {"d1_penalty": -0.5},
        "initial_shuffled": {"d1_penalty": 0.1},
    }]
    stereo = [{
        "right_black": {"d1_penalty": 84.4}, "right_noise": {"d1_penalty": 84.2},
        "right_equals_left": {"d1_penalty": 80.8},
        "initial_constant_mean": {"d1_penalty": 71.5},
        "initial_shuffled": {"d1_penalty": 71.4},
    }]
    assert R._stereo_verdict(checkpoints, monocular, 1.0)["STEREO_FUNCTIONAL"] is False
    assert R._stereo_verdict(checkpoints, stereo, 1.0)["STEREO_FUNCTIONAL"] is True
    # and a collapsed or saturated model fails even with strong ablation numbers
    collapsed = {"200": {"stage_stats": {"entropy": 1.7, "final_std": 0.2}}}
    assert R._stereo_verdict(collapsed, stereo, 1.0)["STEREO_FUNCTIONAL"] is False
    saturated = {"200": {"stage_stats": {"entropy": 0.0, "final_std": 9.0}}}
    assert R._stereo_verdict(saturated, stereo, 1.0)["STEREO_FUNCTIONAL"] is False


# --- 6: the visualizer must not mislabel a comparison -------------------------

def test_visualizer_titles_name_the_arms_being_compared():
    from phase2.tests.test_visualizer import _synthetic_prediction, _synthetic_scene

    scene = _synthetic_scene()
    preds = {}
    for key in ("H2", "SEEDX"):
        pred = _synthetic_prediction(scene)
        pred.model_key = key
        preds[key] = pred
    fig = render.figure_comparison(scene, preds)
    title = fig._suptitle.get_text()
    assert "H2 vs SEEDX" in title, title
    assert "BASE vs WORKING" not in title, title
    render.plt.close(fig)


# --- 7, 8: records and finiteness ---------------------------------------------

def test_replication_records_are_complete():
    for name in _replication_records():
        config, metrics = _record(name)
        for key in ("hypothesis", "seed", "changed_variable", "reference_run",
                    "reference_checkpoint_sha256", "dataset", "train_split",
                    "validation_split", "optimizer", "learning_rate", "scheduler",
                    "loss", "epochs", "precision", "preflight", "known_limitation"):
            assert key in config, (name, key)
        for key in ("history", "checkpoints", "aborted", "wall_clock_s",
                    "epochs_completed", "primary_endpoint_matching_gradient",
                    "gradient_norms", "parameter_count_unique"):
            assert key in metrics, (name, key)
        assert metrics["parameter_count_unique"] == 423586, name


def test_no_non_finite_values_in_any_replication_record():
    for name in _replication_records():
        _, metrics = _record(name)
        assert metrics["gradient_norms"]["any_nan"] is False, name
        for row in metrics["history"]:
            assert np.isfinite(row["mean_loss"]), (name, row["epoch"])
            assert np.isfinite(row["median_grad_norm"]), (name, row["epoch"])
        for epoch, entry in metrics["checkpoints"].items():
            for key in ("val_epe", "val_d1", "val_loss"):
                assert np.isfinite(entry[key]), (name, epoch, key)


def test_same_seed_noise_and_harness_self_consistency_are_recorded():
    """Seed-to-seed claims must be bounded by measured run-to-run variance."""
    noise = json.loads((RESULT_DIR / "same_seed_noise.json").read_text(encoding="utf-8"))
    assert noise["train_loss_abs_difference"]["max"] > 0.0
    self_check = json.loads(
        (RESULT_DIR / "harness_self_consistency.json").read_text(encoding="utf-8"))
    assert self_check["bit_identical"] is False, (
        "if the harness ever becomes bit-reproducible, the noise bound in the "
        "protocol correction must be re-derived")


# --- 9, 10: the probes the verdict depends on ---------------------------------

@needs_h2
def test_stereo_ablation_infrastructure_detects_a_known_dependence():
    """The probe must register the reference model's known right-image dependence."""
    runner = core.ModelRunner(device="cpu")
    scene = core.load_scene(27)
    model = runner._model("H2")[0]
    rows = R.stereo_probe(model, [scene], "cpu", np.random.default_rng(0))
    row = rows[0]
    assert row["baseline"]["d1"] < 20.0, row["baseline"]
    assert row["right_black"]["d1_penalty"] > 50.0, row["right_black"]
    assert row["initial_shuffled"]["d1_penalty"] > 50.0, row["initial_shuffled"]
    assert 0.5 < row["softmax_entropy"] < np.log(12)


def test_matching_gradient_infrastructure_reports_a_live_pathway():
    model = R.build_model(1, "cpu")
    probe = R.gradient_probe(model, "cpu", crops=2)
    assert probe["batches"] == 2
    assert probe["matching_present_fraction"] == 1.0
    assert probe["matching_norm"]["median"] > 0.0
    assert probe["any_non_finite"] is False
    assert probe["mean_softmax_entropy"] > 0.5
