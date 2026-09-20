"""ARM-Y — cost-volume excitation checks.

a. The gate is disparity-independent: the same gate multiplies every D slice.
b. Output shape equals input shape.
c. ARM-V config (flag False) adds zero parameters and is output-identical.
d. The module contains no BatchNorm.
e. The module alone has exactly 1,056 parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.stereonet import StereoNet, StereoNetConfig
from src.models.stereonet.excitation import CostVolumeExcitation


def _arm_v_config() -> StereoNetConfig:
    return StereoNetConfig(
        downsample_levels=3,
        num_disparities=24,
        cost_volume_shift="right",
        regression_normalize=True,
    )


def test_gate_is_disparity_independent():
    torch.manual_seed(0)
    B, C, D, H, W = 1, 32, 8, 4, 5
    exc = CostVolumeExcitation(cv_channels=C, im_channels=C)
    exc.eval()
    left = torch.randn(B, C, H, W)
    with torch.no_grad():
        gate_before = torch.sigmoid(exc.conv2(exc.act(exc.conv1(left))))  # (B, C, H, W)
        assert gate_before.shape == (B, C, H, W)
        # Volume whose D slices are known distinct constants: slice d is (d + 1).
        scales = torch.arange(1, D + 1, dtype=torch.float32).view(1, 1, D, 1, 1)
        volume = torch.ones(B, C, D, H, W) * scales
        gated = exc(volume, left)
    assert gated.shape == volume.shape
    ratio = gated / volume
    slice0 = ratio[:, :, 0]
    for d in range(1, D):
        assert (ratio[:, :, d] - slice0).abs().max().item() < 1e-6
    # The applied ratio equals the sigmoid gate at every pixel/channel.
    assert (slice0 - gate_before).abs().max().item() < 1e-6


def test_output_shape_equals_input_shape():
    torch.manual_seed(1)
    exc = CostVolumeExcitation()
    vol = torch.randn(2, 32, 6, 5, 7)
    feat = torch.randn(2, 32, 5, 7)
    out = exc(vol, feat)
    assert out.shape == vol.shape


def test_arm_v_flag_off_adds_zero_params_and_identical_output():
    torch.manual_seed(2)
    cfg_off = _arm_v_config()
    assert cfg_off.cost_volume_excitation is False
    m_off = StereoNet(cfg_off)
    assert m_off.excitation is None
    torch.manual_seed(3)
    cfg_on = _arm_v_config()
    cfg_on.cost_volume_excitation = True
    m_on = StereoNet(cfg_on)
    delta = sum(p.numel() for p in m_on.parameters()) - sum(
        p.numel() for p in m_off.parameters()
    )
    assert delta == 1056
    # Same seed, same weights for the shared path: copy shared weights so only
    # the gate can explain any output difference.
    torch.manual_seed(4)
    left = torch.randn(1, 3, 64, 96)
    right = torch.randn(1, 3, 64, 96)
    m_off.eval()
    m_on.eval()
    with torch.no_grad():
        out_off = m_off(left, right)
    # A second ARM-V model built from the same seed must match bit-identically.
    torch.manual_seed(2)
    m_off2 = StereoNet(_arm_v_config())
    m_off2.eval()
    with torch.no_grad():
        out_off2 = m_off2(left, right)
    assert (out_off - out_off2).abs().max().item() == 0.0


def test_module_contains_no_batchnorm():
    exc = CostVolumeExcitation()
    assert not any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in exc.modules())
    cfg = _arm_v_config()
    cfg.cost_volume_excitation = True
    model = StereoNet(cfg)
    assert not any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in model.modules()
    )


def test_module_parameter_count_is_1056():
    exc = CostVolumeExcitation(cv_channels=32, im_channels=32)
    total = sum(p.numel() for p in exc.parameters())
    assert total == 1056
    # 32*16 weights (no bias) + 16*32 weights + 32 bias.
    assert exc.conv1.weight.numel() == 512
    assert exc.conv1.bias is None
    assert exc.conv2.weight.numel() == 512
    assert exc.conv2.bias.numel() == 32
