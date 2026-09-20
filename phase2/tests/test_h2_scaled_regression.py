"""Tests for EXP-H2's single changed variable.

The point of these is not that the module runs. It is that the module does
*only* what the experiment says it does: standardise per pixel across the
disparity axis, change no shapes, add no parameters, mutate nothing, leave the
control path alone, and leave frozen Phase 1 untouched.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2.models import scaled_regression as sr  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402
from src.models.stereonet.regression import DisparityRegression, soft_argmin  # noqa: E402

H1_MAGNITUDE = 1e16   # the scale the H1 investigation measured in the aggregated cost


def _cost(batch=2, d=12, h=5, w=7, scale=1.0, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, d, h, w) * scale


# --- what the scaling computes -----------------------------------------------

def test_standardisation_gives_zero_mean_unit_std_per_pixel():
    out = sr.standardise_across_disparity(_cost(scale=H1_MAGNITUDE))
    assert torch.allclose(out.mean(dim=1), torch.zeros(2, 5, 7), atol=1e-4)
    assert torch.allclose(out.std(dim=1, unbiased=False), torch.ones(2, 5, 7), atol=1e-3)


def test_normalisation_is_per_pixel_across_disparity_only():
    """Each pixel must be standardised on its own statistics.

    A pixel whose costs are 1000x another's must come out with the same
    z-scores; and changing one pixel must not change any other.
    """
    cost = torch.zeros(1, 12, 2, 1)
    cost[0, :, 0, 0] = torch.arange(12, dtype=torch.float32)
    cost[0, :, 1, 0] = torch.arange(12, dtype=torch.float32) * 1000.0 + 5e6
    out = sr.standardise_across_disparity(cost)
    assert torch.allclose(out[0, :, 0, 0], out[0, :, 1, 0], atol=1e-3), out

    changed = cost.clone()
    changed[0, :, 1, 0] += 1e9
    out2 = sr.standardise_across_disparity(changed)
    assert torch.equal(out[0, :, 0, 0], out2[0, :, 0, 0]), "pixels must be independent"

    # and it must not be a batch statistic either
    stacked = torch.cat([cost, cost * 1e6], dim=0)
    out3 = sr.standardise_across_disparity(stacked)
    assert torch.allclose(out3[0], out3[1], atol=1e-3), "must not use batch statistics"


def test_scaling_does_not_mutate_its_input():
    cost = _cost(scale=H1_MAGNITUDE)
    before = cost.clone()
    sr.standardise_across_disparity(cost)
    assert torch.equal(cost, before)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e6, 1e19])
def test_numerically_stable_across_magnitudes(scale):
    out = sr.standardise_across_disparity(_cost(scale=scale))
    assert torch.isfinite(out).all(), scale
    assert float(out.abs().max()) < 100.0, scale


def test_constant_cost_column_does_not_divide_by_zero():
    flat = torch.full((1, 12, 3, 3), 7.0)
    out = sr.standardise_across_disparity(flat)
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) == 0.0
    # and the soft-argmin of a flat cost is the mean candidate index, 5.5
    assert float(soft_argmin(out, dim=1).mean()) == pytest.approx(5.5, abs=1e-5)


# --- the module in place of the frozen one ------------------------------------

def test_module_preserves_shape_and_candidate_range():
    module = sr.StandardisedDisparityRegression()
    out = module(_cost(batch=1, d=12, h=4, w=6), (16, 24))
    assert out.shape == (1, 1, 16, 24)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 11.0


def test_module_adds_no_parameters_and_holds_no_state():
    module = sr.StandardisedDisparityRegression()
    assert list(module.parameters()) == []
    assert list(module.buffers()) == []


def test_a_preferred_candidate_wins_but_confidence_is_capped():
    """Standardisation removes scale, so it also caps how peaked the softmax can be.

    With one clear winner among 12 candidates the winner takes ~0.77 of the
    weight no matter how large its margin was, and the regressed value is pulled
    toward the candidate mean. This is a real consequence of the changed
    variable, not a defect, and it is pinned here so it cannot change silently.
    """
    module = sr.StandardisedDisparityRegression()
    values = []
    for k in range(12):
        cost = torch.zeros(1, 12, 2, 2)
        cost[:, k] = -50.0            # much cheaper than the others
        weights = torch.softmax(-sr.standardise_across_disparity(cost), dim=1)
        assert int(weights[0, :, 0, 0].argmax()) == k
        assert float(weights[0, k, 0, 0]) == pytest.approx(0.77, abs=0.02)
        values.append(float(module(cost, (2, 2)).mean()))

    assert values == sorted(values), values          # monotone in the winner
    assert values[0] > 0.0 and values[-1] < 11.0     # pulled toward the centre
    assert abs(values[6] - 6.0) < 0.6, values[6]     # interior estimates stay near k


def test_scale_invariance_holds_above_the_epsilon_floor_and_fails_below_it():
    """Documented limitation of the fixed eps, pinned rather than hidden.

    The standardisation is invariant to the cost's magnitude only while the
    per-pixel standard deviation is large compared with eps = 1e-5. Below that
    floor the division is dominated by eps and the soft-argmin collapses toward
    the uniform mean of 5.5. The H2 run's own diagnostics record the observed
    cost std so the margin can be checked against this.
    """
    cost = _cost(batch=1, h=3, w=3)
    module = sr.StandardisedDisparityRegression()
    # reference: eps made negligible, i.e. the exact z-score
    exact = soft_argmin(sr.standardise_across_disparity(cost, eps=1e-30), dim=1)

    for factor, tolerance in ((1e-2, 5e-3), (1.0, 1e-4), (1e3, 1e-4), (1e12, 1e-4)):
        # the deviation from the exact z-score is of order eps / std
        deviation = float((module(cost * factor, (3, 3)) - exact).abs().max())
        assert deviation < tolerance, (factor, deviation)

    collapsed = module(cost * 1e-6, (3, 3))      # std ~1e-6, at the eps floor
    assert float((collapsed - exact).abs().max()) > 0.1
    assert float(collapsed.mean()) == pytest.approx(5.5, abs=0.2)


def test_swap_changes_only_the_regression_stage():
    torch.manual_seed(0)
    model = StereoNet(StereoNetConfig(cost_volume_shift="left"))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert isinstance(model.regression, DisparityRegression)

    sr.apply_to(model)
    assert isinstance(model.regression, sr.StandardisedDisparityRegression)
    assert model.regression.upsample_first == model.config.upsample_before_argmin
    after = model.state_dict()
    assert set(before) == set(after), "the swap must not add or remove parameters"
    assert all(torch.equal(before[k], after[k]) for k in before), "weights must not move"
    assert model.parameter_count() == 423586


def test_control_regression_still_saturates_and_h2_does_not():
    """The two paths must differ in exactly the documented way."""
    cost = _cost(batch=1, h=3, w=3, scale=H1_MAGNITUDE)
    control = DisparityRegression()(cost, (3, 3))
    h2 = sr.StandardisedDisparityRegression()(cost, (3, 3))

    control_weights = torch.softmax(-cost, dim=1)
    scaled_weights = torch.softmax(-sr.standardise_across_disparity(cost), dim=1)
    entropy = lambda p: -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)  # noqa: E731
    assert float(entropy(control_weights).mean()) < 1e-6      # saturated hard argmin
    assert float(entropy(scaled_weights).mean()) > 1.0        # usable distribution
    # the control's output is an integer index; H2's is not forced to be
    assert torch.allclose(control, control.round(), atol=1e-4)
    assert not torch.allclose(h2, h2.round(), atol=1e-4)


# --- gradient, the primary endpoint's mechanism -------------------------------

def test_gradient_reaches_the_cost_only_with_scaling():
    cost = _cost(scale=H1_MAGNITUDE).requires_grad_(True)
    soft_argmin(cost, dim=1).sum().backward()
    assert float(cost.grad.abs().max()) == 0.0, "control path must be gradient-dead here"

    cost2 = _cost(scale=H1_MAGNITUDE).requires_grad_(True)
    soft_argmin(sr.standardise_across_disparity(cost2), dim=1).sum().backward()
    assert float(cost2.grad.abs().max()) > 0.0
    assert torch.isfinite(cost2.grad).all()


def test_gradient_reaches_aggregation_and_features_through_the_swapped_stage():
    torch.manual_seed(0)
    model = StereoNet(StereoNetConfig(cost_volume_shift="left"))
    sr.apply_to(model)
    left = torch.randn(1, 3, 64, 64)
    right = torch.randn(1, 3, 64, 64)
    model(left, right).sum().backward()
    for stage in ("feature_extractor", "aggregation"):
        norm = sum(float(p.grad.norm()) ** 2
                   for n, p in model.named_parameters()
                   if n.startswith(stage) and p.grad is not None) ** 0.5
        assert norm > 1e-12, (stage, norm)


# --- the things that must not have moved --------------------------------------

def test_phase_1_source_is_untouched():
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "phase-1-frozen", "--", "src", "scripts"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    except Exception as exc:  # no git, or no such tag on this machine
        pytest.skip("cannot check the frozen tag here: {}".format(exc))
    assert out.stdout.strip() == "", "Phase 1 changed:\n" + out.stdout


def test_h1_records_and_checkpoints_are_not_written_by_h2():
    """The H2 module and script must never target an H1 artefact path."""
    from phase2.scripts import exp_h2_softargmin_scale as h2

    assert h2.EXPERIMENT_ID == "EXP-H2-SOFTARGMIN-SCALE"
    assert h2.CONTROL_ID == "EXP-H1-WORKING-v2"
    assert "H2" in (h2.OUT_DIR / (h2.EXPERIMENT_ID + "_checkpoint.pth")).name
    for h1 in ("EXP-H1-BASE-v2", "EXP-H1-WORKING-v2"):
        assert not (h2.OUT_DIR / (h2.EXPERIMENT_ID + "_checkpoint.pth")).name.startswith(h1)


def test_recipe_constants_match_the_control_run():
    """H2 must inherit the H1 recipe rather than restate it."""
    from phase2.scripts import exp_h1_cost_volume as h1
    from phase2.scripts import exp_h2_softargmin_scale as h2

    assert (h2.CROP_H, h2.CROP_W) == (h1.CROP_H, h1.CROP_W) == (256, 512)
    assert h2.CroppedKitti is h1.CroppedKitti
    assert h2.validate is h1.validate
