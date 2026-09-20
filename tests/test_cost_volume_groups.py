"""Group-wise cost reduction for the StereoNet cost volume (step 1)."""

import torch
import torch.nn as nn

from src.models.stereonet import (
    CostVolume,
    StereoNet,
    StereoNetConfig,
    build_cost_volume,
)


def test_groups_zero_matches_build_cost_volume_bit_identical():
    torch.manual_seed(0)
    left = torch.randn(1, 32, 23, 77)
    right = torch.randn(1, 32, 23, 77)
    cv = CostVolume(num_disparities=24, method="subtract", shift="right", groups=0)
    assert sum(p.numel() for p in cv.parameters()) == 0
    assert not hasattr(cv, "reduce")
    assert len(cv.state_dict()) == 0
    out = cv(left, right)
    ref = build_cost_volume(left, right, 24, "subtract", "right")
    assert torch.equal(out, ref)


def test_groups_eight_output_shape():
    torch.manual_seed(0)
    cv = CostVolume(num_disparities=24, channels=32, groups=8, shift="right")
    left = torch.randn(1, 32, 23, 77)
    right = torch.randn(1, 32, 23, 77)
    out = cv(left, right)
    assert out.shape == (1, 8, 24, 23, 77)


def test_reduce_conv_shared_grouped():
    cv = CostVolume(num_disparities=24, channels=32, groups=8, shift="right")
    convs = [m for m in cv.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 1
    reduce = cv.reduce
    assert reduce.weight.numel() == 32
    assert tuple(reduce.weight.shape) == (8, 4, 1, 1)
    assert reduce.groups == 8
    assert reduce.bias is None
    # Output channel g depends only on input channels [4g, 4g + 4).
    for g in range(8):
        x = torch.zeros(1, 32, 2, 2)
        x[:, 4 * g:4 * g + 4] = 1.0
        y = reduce(x)
        assert y.shape == (1, 8, 2, 2)
        assert torch.all(y[:, g] != 0)
        mask = torch.ones(8, dtype=torch.bool)
        mask[g] = False
        assert torch.equal(y[:, mask], torch.zeros_like(y[:, mask]))


def test_stereonet_parameter_counts_groups_0_and_8():
    cfg8 = StereoNetConfig(
        downsample_levels=3,
        num_disparities=24,
        cost_volume_shift="right",
        regression_normalize=True,
        cost_volume_groups=8,
    )
    cfg0 = StereoNetConfig(
        downsample_levels=3,
        num_disparities=24,
        cost_volume_shift="right",
        regression_normalize=True,
        cost_volume_groups=0,
    )
    assert StereoNet(cfg8).parameter_count() == 377250
    assert StereoNet(cfg0).parameter_count() == 397954
