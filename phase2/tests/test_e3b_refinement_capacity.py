"""Checks for the logic E3b's verdict depends on.

Training itself needs GPU-hours; these cover the arm construction (that an arm
differs from the control in nothing but block count) and the frozen band /
material rule, where a silent error would change the reported verdict.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase2.scripts.exp_e3b_refinement_capacity import (  # noqa: E402
    ARMS, BAND_MULTIPLIER, FLOOR_D1, FLOOR_EPE, H2_RECORDED_LATE, build_arm,
)


def band(noise_epe, noise_d1):
    """The frozen band, recomputed the way run_analysis does."""
    return (max(BAND_MULTIPLIER * noise_epe, FLOOR_EPE),
            max(BAND_MULTIPLIER * noise_d1, FLOOR_D1))


def materially_different(d_epe, d_d1, band_epe, band_d1):
    """The frozen conjunction: BOTH metrics must move past their threshold."""
    return abs(d_epe) > band_epe and abs(d_d1) > band_d1


def test_arm_specification_matches_the_registration():
    assert ARMS["A"]["keep"] == (0, 1, 2, 3, 4, 5)
    assert ARMS["B"]["keep"] == (0, 1, 2, 3, 4)
    assert ARMS["C"]["keep"] == (0, 1, 2, 3)
    # The 1-2-4-8 ladder survives in every arm; only trailing dilation-1 blocks go.
    for arm in ARMS.values():
        assert arm["dilations"][:4] == (1, 2, 4, 8)
    assert [ARMS[a]["parameters"] for a in "ABC"] == [423586, 405090, 386594]


def test_arms_differ_only_in_block_count():
    models = {a: build_arm(a, "cpu") for a in ARMS}
    reference = models["A"]
    for arm, model in models.items():
        assert len(model.refinement.blocks) == ARMS[arm]["blocks"]
        assert model.parameter_count() == ARMS[arm]["parameters"]
        assert model.cost_volume.shift == "left"
        for position, index in enumerate(ARMS[arm]["keep"]):
            a = reference.refinement.blocks[index].state_dict()
            b = model.refinement.blocks[position].state_dict()
            assert all(torch.equal(a[k], b[k]) for k in a)
        for name in ("feature_extractor", "aggregation"):
            a = getattr(reference, name).state_dict()
            b = getattr(model, name).state_dict()
            assert all(torch.equal(a[k], b[k]) for k in a)


def test_the_floor_binds_when_measured_noise_is_small():
    band_epe, band_d1 = band(0.1069, 0.1267)          # the values actually measured
    assert abs(band_epe - 0.2138) < 1e-9              # 2x noise wins on EPE
    assert band_d1 == FLOOR_D1                        # the floor wins on D1


def test_material_rule_needs_both_metrics():
    band_epe, band_d1 = band(0.1069, 0.1267)
    # Arm B as measured: EPE past its band, D1 inside -> not materially different.
    assert not materially_different(-0.2333, 0.8759, band_epe, band_d1)
    # Arm C as measured: D1 past its band, EPE inside -> not materially different.
    assert not materially_different(-0.0893, 1.1610, band_epe, band_d1)
    # Both past -> materially different.
    assert materially_different(-0.30, 1.50, band_epe, band_d1)


def test_control_is_compared_against_the_recorded_h2_late_window():
    assert H2_RECORDED_LATE == {"epe": 4.199, "d1": 24.783}
