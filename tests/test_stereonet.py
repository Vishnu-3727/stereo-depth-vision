"""Unit tests for the independent StereoNet implementation.

These test the implementation against the architecture specification in
docs/stereonet_architecture.md, not against the upstream source. The
tensor-level comparison with the reference ONNX lives in
scripts/exp_onnx_equivalence.py (EXP-011) because it needs the reference
artifact; these run anywhere.
"""

import numpy as np
import pytest
import torch

from src.models.stereonet import (
    Aggregation,
    CostVolume,
    FeatureExtractor,
    Refinement,
    StereoNet,
    StereoNetConfig,
    build_cost_volume,
    build_cost_volume_reference,
    reference_shift,
    shift_left,
    soft_argmin,
)

# A small input that still divides by 16, so shapes behave as at full size.
SMALL = (1, 3, 64, 128)


# -- feature extraction ----------------------------------------------------

def test_feature_extractor_reduces_resolution_by_16():
    fe = FeatureExtractor()
    out = fe(torch.randn(*SMALL))
    assert fe.scale == 16
    assert out.shape == (1, 32, 4, 8)


def test_feature_extractor_shapes_at_deployment_resolution():
    fe = FeatureExtractor()
    out = fe(torch.zeros(1, 3, 368, 1232))
    assert out.shape == (1, 32, 23, 77), out.shape


def test_downsampling_stack_has_no_activations():
    """The deployed model's downsampling stack is purely linear. That is a
    finding about the baseline, so the implementation must reproduce it -- if
    someone later inserts activations, this test should fail and force the
    change to be deliberate."""
    fe = FeatureExtractor()
    x = torch.randn(2, 3, 64, 128)
    with torch.no_grad():
        # Linearity up to the bias terms: f(a+b) - f(a) - f(b) + f(0) == 0
        a, b = x[:1], x[1:]
        fa, fb = fe.downsample(a), fe.downsample(b)
        fab, f0 = fe.downsample(a + b), fe.downsample(torch.zeros_like(a))
        residual = (fab - fa - fb + f0).abs().max()
    assert residual < 1e-3, residual


# -- cost volume -----------------------------------------------------------

def test_reference_shift_is_a_no_op():
    """The deployed model's disparity shift returns its input unchanged
    (EXP-010). Pinning it so the degenerate behaviour cannot be silently
    'fixed' while the baseline is frozen."""
    x = torch.randn(1, 4, 3, 7)
    for k in range(5):
        assert torch.equal(reference_shift(x, k), x), k


def test_shift_left_actually_shifts():
    x = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5)
    got = shift_left(x, 2)
    assert got.flatten().tolist() == [2.0, 3.0, 4.0, 0.0, 0.0]


def test_reference_cost_volume_slices_are_all_identical():
    """The consequence of the no-op shift: no disparity search happens."""
    left, right = torch.randn(1, 8, 5, 11), torch.randn(1, 8, 5, 11)
    vol = build_cost_volume(left, right, 6, shift="none")
    assert vol.shape == (1, 8, 6, 5, 11)
    for k in range(6):
        assert torch.equal(vol[:, :, k], vol[:, :, 0]), k
    assert torch.allclose(vol[:, :, 0], left - right)


def test_shifted_cost_volume_slices_differ():
    left, right = torch.randn(1, 8, 5, 11), torch.randn(1, 8, 5, 11)
    vol = build_cost_volume(left, right, 6, shift="left")
    differing = sum(1 for k in range(1, 6) if not torch.equal(vol[:, :, k], vol[:, :, 0]))
    assert differing == 5


@pytest.mark.parametrize("shift", ["none", "left"])
def test_vectorised_cost_volume_matches_independent_loop(shift):
    """The fast path is checked against a slow, explicitly indexed version
    written separately, so agreement is evidence rather than a tautology."""
    torch.manual_seed(0)
    left, right = torch.randn(1, 3, 4, 9), torch.randn(1, 3, 4, 9)
    fast = build_cost_volume(left, right, 5, shift=shift)
    slow = build_cost_volume_reference(left, right, 5, shift=shift)
    assert torch.allclose(fast, slow, atol=1e-6), (fast - slow).abs().max()


def test_cost_volume_memory_formula():
    cv = CostVolume(num_disparities=12)
    # B x C x D x H x W x 4, the figure quoted in docs/cost_volume_analysis.md
    assert cv.memory_bytes(1, 32, 23, 77) == 1 * 32 * 12 * 23 * 77 * 4 == 2_720_256


def test_concat_method_doubles_channels():
    left, right = torch.randn(1, 8, 5, 11), torch.randn(1, 8, 5, 11)
    vol = build_cost_volume(left, right, 4, method="concat")
    assert vol.shape == (1, 16, 4, 5, 11)


# -- aggregation and regression --------------------------------------------

def test_aggregation_reduces_channels_to_one_cost_per_candidate():
    agg = Aggregation()
    out = agg(torch.randn(1, 32, 12, 4, 8))
    assert out.shape == (1, 12, 4, 8)


def test_soft_argmin_recovers_a_planted_disparity():
    """A cost tensor with a sharp minimum at candidate k must reduce to k."""
    d, h, w = 12, 3, 5
    for planted in (0, 3, 7, 11):
        cost = torch.full((1, d, h, w), 50.0)
        cost[:, planted] = -50.0
        got = soft_argmin(cost, dim=1)
        assert got.shape == (1, 1, h, w)
        assert torch.allclose(got, torch.full_like(got, float(planted)), atol=1e-4), (
            planted, got.flatten()[0].item()
        )


def test_soft_argmin_is_subpixel_between_candidates():
    """Two equally good neighbouring candidates should give the midpoint. This
    is the mechanism the architecture relies on for sub-pixel precision."""
    cost = torch.full((1, 12, 1, 1), 50.0)
    cost[:, 4] = -50.0
    cost[:, 5] = -50.0
    got = soft_argmin(cost, dim=1)
    assert abs(got.item() - 4.5) < 1e-4, got.item()


def test_soft_argmin_output_is_bounded_by_candidate_count():
    """Its range is 0..D-1 in candidate units, which is why the deployed
    model's initial estimate cannot exceed 11 (EXP-006)."""
    got = soft_argmin(torch.randn(4, 12, 6, 6), dim=1)
    assert float(got.min()) >= 0.0
    assert float(got.max()) <= 11.0


# -- refinement ------------------------------------------------------------

def test_refinement_preserves_resolution_and_returns_a_residual():
    ref = Refinement()
    disparity = torch.randn(1, 1, 64, 128)
    out = ref(disparity, torch.randn(1, 3, 64, 128))
    assert out.shape == disparity.shape


def test_refinement_dilations_match_the_specification():
    ref = Refinement()
    dilations = [b.conv1.dilation[0] for b in ref.blocks]
    assert dilations == [1, 2, 4, 8, 1, 1]


# -- end to end ------------------------------------------------------------

def test_end_to_end_output_shape_and_stages():
    model = StereoNet().eval()
    with torch.no_grad():
        out, stages = model(
            torch.randn(*SMALL), torch.randn(*SMALL), return_stages=True
        )
    assert out.shape == (1, 1, 64, 128)
    assert stages["left_features"].shape == (1, 32, 4, 8)
    assert stages["cost_volume"].shape == (1, 32, 12, 4, 8)
    assert stages["aggregated_cost"].shape == (1, 12, 4, 8)
    assert stages["disparity_initial"].shape == (1, 1, 64, 128)


def test_output_is_non_negative_because_of_the_final_relu():
    model = StereoNet().eval()
    with torch.no_grad():
        out = model(torch.randn(*SMALL), torch.randn(*SMALL))
    assert float(out.min()) >= 0.0


def test_parameter_counts_match_the_reference():
    """423,586 unique weights; 623,138 counting the shared Siamese extractor
    once per graph occurrence, which is Hailo's published 623.1K
    [MEASUREMENT: EXP-001]."""
    model = StereoNet()
    assert model.parameter_count() == 423_586
    assert model.parameter_count(per_occurrence=True) == 623_138


def test_config_reports_the_disparity_search_range():
    c = StereoNetConfig()
    assert c.feature_stride == 16
    # 12 candidates 16 px apart: 0 to 176 px
    assert c.max_disparity_px == 176


def test_model_depends_on_the_right_image():
    """A sanity check that the two inputs are not accidentally interchangeable
    in this implementation. The measured version of this on the reference model
    is EXP-007."""
    torch.manual_seed(0)
    model = StereoNet().eval()
    left, right = torch.randn(*SMALL), torch.randn(*SMALL)
    with torch.no_grad():
        a = model(left, right)
        b = model(left, torch.randn(*SMALL))
    assert not torch.allclose(a, b)
