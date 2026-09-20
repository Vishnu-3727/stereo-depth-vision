"""Regression tests for EXP-O6-REFINEMENT-DILATION-001.

O6 is a confound-disambiguation experiment: it is only meaningful if the arms
differ in the dilation schedule and in *nothing else*. These tests exist to fail
loudly if that ever stops being true -- if an arm quietly picks up a different
parameter count, a different cost volume, a different readout, a different
initialisation, or a changed tensor contract, the experiment stops answering the
question it was registered to answer.

Model-building tests run on CPU and need no checkpoint. The integrity tests need
the repository; they skip if it is unavailable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import phase2.scripts.exp_o6_refinement_dilation as o6  # noqa: E402
from phase2.viz import core  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402
from src.models.stereonet.refinement import DEFAULT_DILATIONS  # noqa: E402

H2_SHA256 = "e3d48021d7f6a3d4d16fe33031e8ba85fd24389c5ba892f52c7e873e48454927"
DEVICE = "cpu"


@pytest.fixture(scope="module")
def models():
    return {arm: o6.build_arm(arm, DEVICE) for arm in o6.ARMS}


# --- 1. each configuration creates the intended architecture -----------------

def test_each_arm_builds_the_dilation_schedule_it_declares(models):
    """Read the schedule back off the built convolutions, do not trust the spec."""
    assert sorted(o6.ARMS) == ["D124", "D128", "D148"]
    expected = {"D124": [1, 2, 4], "D128": [1, 2, 8], "D148": [1, 4, 8]}
    for arm, model in models.items():
        assert o6.effective_dilations(model) == expected[arm], arm
        assert list(o6.ARMS[arm]["dilations"]) == expected[arm], arm
        assert len(model.refinement.blocks) == 3, arm


def test_padding_tracks_dilation_so_the_residual_add_stays_valid(models):
    """ResBlock adds its input to its output; any padding/dilation mismatch would
    change the spatial size and break that add."""
    for arm, model in models.items():
        for block in model.refinement.blocks:
            for conv in (block.conv1, block.conv2):
                assert conv.padding == conv.dilation, (arm, conv.padding, conv.dilation)
                assert conv.kernel_size == (3, 3), arm
                assert conv.stride == (1, 1), arm


def test_the_three_schedules_are_distinct_and_two_of_them_keep_dilation_8():
    """The experiment is pointless if the arms are not actually different, or if
    the two experimental arms do not both retain the wide dilation."""
    schedules = {a: tuple(o6.ARMS[a]["dilations"]) for a in o6.ARMS}
    assert len(set(schedules.values())) == 3, schedules
    assert 8 not in schedules["D124"]
    assert 8 in schedules["D128"] and 8 in schedules["D148"]


def test_receptive_field_is_ordered_as_the_registration_describes():
    """DERIVED quantity: 1 + 2*(input + output + sum of 2*dilation)."""
    assert o6.receptive_field((1, 2, 4)) == 33
    assert o6.receptive_field((1, 2, 8)) == 49
    assert o6.receptive_field((1, 4, 8)) == 57
    assert o6.receptive_field(DEFAULT_DILATIONS) == 73
    # the arms that keep dilation 8 must have the wider field, or the hypothesis
    # is not being tested at all
    assert o6.receptive_field((1, 2, 4)) < o6.receptive_field((1, 2, 8))
    assert o6.receptive_field((1, 2, 4)) < o6.receptive_field((1, 4, 8))


# --- 2. parameter counts are deterministic and identical ---------------------

def test_parameter_counts_are_identical_across_arms_and_deterministic(models):
    """Dilation changes the sampling grid, never the weight shape. If this ever
    fails, capacity is no longer held constant and O6 cannot separate the two
    causes it exists to separate."""
    counts = {arm: m.parameter_count() for arm, m in models.items()}
    assert len(set(counts.values())) == 1, counts
    assert set(counts.values()) == {o6.EXPECTED_PARAMETERS}, counts
    assert o6.EXPECTED_PARAMETERS == 368098
    # rebuilding must reproduce the same number
    again = o6.build_arm("D128", DEVICE).parameter_count()
    assert again == o6.EXPECTED_PARAMETERS


def test_initial_weights_are_bit_identical_across_arms(models):
    """All three arms keep blocks 0,1,2 of one seed-0 stack and are re-dilated in
    place, so initialisation is not a confound. Weight tensors must be equal
    everywhere, including input_conv and output_conv."""
    names = sorted(models)
    reference = models[names[0]].state_dict()
    for arm in names[1:]:
        state = models[arm].state_dict()
        assert set(state) == set(reference), arm
        for key in reference:
            assert torch.equal(reference[key], state[key]), (arm, key)


def test_re_dilation_changes_only_dilation_and_padding():
    """The mutation must not touch weights -- that is the whole basis for the
    initialisation claim above."""
    a = o6.build_arm("D124", DEVICE)
    before = {k: v.detach().clone() for k, v in a.state_dict().items()}
    o6.set_dilations(a, (1, 4, 8))
    after = a.state_dict()
    assert set(after) == set(before)
    for key in before:
        assert torch.equal(before[key], after[key]), key
    assert o6.effective_dilations(a) == [1, 4, 8]


def test_set_dilations_rejects_a_length_mismatch():
    model = o6.build_arm("D124", DEVICE)
    with pytest.raises(ValueError):
        o6.set_dilations(model, (1, 2))


# --- 3. MAC counts are deterministic and identical ----------------------------

def test_mac_counts_are_identical_across_arms_and_deterministic(models):
    """Dilated convolution has the same multiply-accumulate count as undilated.
    Small input: this asserts the invariant, not the deploy figure."""
    thop = pytest.importorskip("thop")
    left = torch.randn(1, 3, 64, 128)
    right = torch.randn(1, 3, 64, 128)
    macs = {}
    for arm in sorted(o6.ARMS):
        model = o6.build_arm(arm, DEVICE)          # fresh: thop mutates modules
        macs[arm], _ = thop.profile(model, inputs=(left, right), verbose=False)
    assert len(set(macs.values())) == 1, macs

    repeat, _ = thop.profile(o6.build_arm("D148", DEVICE), inputs=(left, right),
                             verbose=False)
    assert repeat == macs["D148"]


# --- 4 & 9. nothing else about the architecture moved -------------------------

def test_arms_differ_from_each_other_in_refinement_dilation_only(models):
    """Every non-refinement stage, and the refinement's own input/output convs,
    must be structurally identical across arms."""
    names = sorted(models)
    reference = models[names[0]]
    for arm in names[1:]:
        model = models[arm]
        for stage in ("feature_extractor", "cost_volume", "aggregation", "regression"):
            assert repr(getattr(model, stage)) == repr(getattr(reference, stage)), (arm, stage)
        for part in ("input_conv", "output_conv"):
            assert repr(getattr(model.refinement, part)) \
                == repr(getattr(reference.refinement, part)), (arm, part)
        assert model.config == reference.config, arm


def test_no_accidental_cost_volume_readout_or_disparity_range_change(models):
    """The pathway H2 established must be exactly what every O6 arm runs."""
    for arm, model in models.items():
        assert model.cost_volume.shift == "left", arm
        assert model.config.cost_volume_method == StereoNetConfig().cost_volume_method, arm
        assert model.config.num_disparities == 12, arm
        assert model.config.upsample_before_argmin is True, arm
        assert model.config.final_relu is True, arm
        # the H2 standardised readout, not the frozen Phase 1 soft-argmin
        assert type(model.regression).__name__ == "StandardisedDisparityRegression", arm


def test_the_unmodified_h2_architecture_is_untouched():
    """Building H2 the way the H2 script does must still give six blocks with the
    original ladder and the baseline parameter count. O6 must not have leaked
    into the default configuration."""
    assert StereoNetConfig().refinement_dilations == (1, 2, 4, 8, 1, 1)
    assert DEFAULT_DILATIONS == (1, 2, 4, 8, 1, 1)
    baseline = StereoNet(StereoNetConfig(cost_volume_shift="left"))
    assert len(baseline.refinement.blocks) == 6
    assert baseline.parameter_count() == 423586
    assert [int(b.conv1.dilation[0]) for b in baseline.refinement.blocks] \
        == [1, 2, 4, 8, 1, 1]


# --- 8. tensor shape contracts ------------------------------------------------

def test_every_arm_preserves_the_stage_shape_contract(models):
    """Same input, same shape at every named stage, for every arm."""
    left = torch.randn(1, 3, 64, 128)
    right = torch.randn(1, 3, 64, 128)
    shapes = {}
    for arm, model in models.items():
        model.eval()
        with torch.no_grad():
            out, stages = model(left, right, return_stages=True)
        assert out.shape == (1, 1, 64, 128), (arm, out.shape)
        assert torch.isfinite(out).all(), arm
        shapes[arm] = {k: tuple(v.shape) for k, v in stages.items()}
    names = sorted(shapes)
    for arm in names[1:]:
        assert shapes[arm] == shapes[names[0]], arm
    assert "disparity_initial" in shapes[names[0]]
    assert "refinement_residual" in shapes[names[0]]


# --- 5, 6. integrity ----------------------------------------------------------

@pytest.mark.skipif(not core.checkpoint_path("H2").exists(),
                    reason="H2 checkpoint not present")
def test_h2_checkpoint_hash_is_unchanged():
    assert core.sha256(core.checkpoint_path("H2")) == H2_SHA256


def test_phase_1_diff_against_the_frozen_tag_is_empty():
    proc = subprocess.run(
        ["git", "diff", "--stat", "phase-1-frozen", "--",
         "src", "scripts", "tests", "experiments", "docs", "results", "reference"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip("git unavailable: " + proc.stderr.strip())
    assert proc.stdout.strip() == "", proc.stdout


# --- 10. experiment records ---------------------------------------------------

def test_arm_records_are_named_uniquely_and_do_not_collide_with_earlier_work():
    ids = {o6.arm_id(a) for a in o6.ARMS}
    assert len(ids) == len(o6.ARMS)
    for eid in ids:
        assert eid.startswith("EXP-O6-REFINEMENT-DILATION-001-ARM-")
        for earlier in ("EXP-H1", "EXP-H2", "EXP-H3", "EXP-E1", "EXP-E2", "EXP-E3"):
            assert not eid.startswith(earlier), eid


def test_arm_config_records_the_registered_fields_machine_readably():
    for arm in o6.ARMS:
        config = o6.arm_config(arm, DEVICE)
        assert config["refinement_dilations"] == list(o6.ARMS[arm]["dilations"])
        assert config["refinement_blocks"] == 3
        assert config["preregistration"] == o6.PREREGISTRATION
        assert config["seed"] == 0
        assert config["epochs"] == 200
        assert config["cost_volume_shift"] == "left"
        assert config["experiment"] == o6.arm_id(arm)
        assert config["band"] == o6.BAND
        assert config["known_limitation"]
        assert config["refinement_receptive_field_px"] == o6.receptive_field(
            o6.ARMS[arm]["dilations"])


def test_recipe_fields_are_identical_across_arm_configs():
    """The single-variable condition, checked on the records that will be written
    rather than only on the models."""
    configs = {a: o6.arm_config(a, DEVICE) for a in o6.ARMS}
    names = sorted(configs)
    for key in o6.RECIPE_KEYS_MUST_MATCH:
        first = configs[names[0]][key]
        for a in names[1:]:
            assert configs[a][key] == first, (key, a, configs[a][key], first)
    # and the one field that must differ, does
    assert len({tuple(configs[a]["refinement_dilations"]) for a in names}) == 3


def test_the_preregistration_exists_and_freezes_the_decision_rule():
    text = (REPO_ROOT / o6.PREREGISTRATION).read_text(encoding="utf-8")
    for token in ("RECEPTIVE-FIELD-IMPLICATED", "CAPACITY-IMPLICATED",
                  "CONFIGURATION-SENSITIVE", "INCONCLUSIVE", "(1, 2, 8)", "(1, 4, 8)"):
        assert token in text, token
    assert str(o6.RECOVERY_STRONG * 100).startswith("50")
    assert str(o6.RECOVERY_WEAK * 100).startswith("20")
    assert o6.BAND["epe"] == pytest.approx(0.21380062638697517)
    assert o6.BAND["d1"] == 1.0


@pytest.mark.skipif(
    not (REPO_ROOT / "phase2" / "experiments" / o6.CONTROL_ID / "metrics.json").exists(),
    reason="control record not present")
def test_the_reused_control_is_the_six_block_arm_it_claims_to_be():
    metrics = json.loads(
        (REPO_ROOT / "phase2" / "experiments" / o6.CONTROL_ID / "metrics.json")
        .read_text(encoding="utf-8"))["metrics"]
    assert metrics["parameter_count_unique"] == 423586
    assert metrics["epochs_completed"] == 200
    assert metrics.get("aborted") is None
    assert metrics["stereo_verdict_epoch_200"]["STEREO_FUNCTIONAL"] is True
