"""ARM-Z — per-pixel L2 feature normalization checks.

1. Normalized nonzero vectors have unit L2 norm along dim=1 (atol 1e-5).
2. An exactly-zero feature vector gives a finite result (no NaN/inf) via eps.
3. Norm is over channels only: spatial magnitude removed, direction kept.
4. Param count identical with flag on/off: 397954 at the ARM-V config.
5. Cost volume shape is (B, 32, 24, H, W).
6. Sign: volume equals Lhat - shift_right(Rhat, k), hand-computed (no negate).
7. Flag off reproduces the ARM-V volume exactly (bitwise).
"""
from __future__ import annotations

import torch

from src.models.stereonet import StereoNet, StereoNetConfig
from src.models.stereonet.cost_volume import build_cost_volume, shift_right
from src.models.stereonet.stereonet import normalize_features_l2


def _arm_v_config() -> StereoNetConfig:
    return StereoNetConfig(
        downsample_levels=3,
        num_disparities=24,
        cost_volume_shift="right",
        regression_normalize=True,
    )


def test_normalized_nonzero_vectors_have_unit_norm():
    torch.manual_seed(0)
    x = torch.randn(2, 32, 5, 7)
    out = normalize_features_l2(x)
    norms = out.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_zero_vector_is_finite_via_eps_clamp():
    x = torch.zeros(1, 32, 3, 4)
    out = normalize_features_l2(x)
    assert torch.isfinite(out).all().item()
    assert (out == 0).all().item()  # 0 / 1e-5 == 0


def test_norm_is_over_channels_only():
    torch.manual_seed(1)
    B, C, H, W = 1, 4, 2, 6
    direction = torch.randn(B, C, 1, 1).expand(B, C, H, W).contiguous()
    scale = torch.linspace(0.5, 3.0, W).view(1, 1, 1, W).expand(B, 1, H, W)
    x = direction * scale
    out = normalize_features_l2(x)
    # Spatially varying magnitude removed: same direction -> same output.
    for w in range(1, W):
        d = (out[..., w] - out[..., 0]).abs().max().item()
        assert d < 1e-5, d
    # Direction preserved: output parallel to input (cosine sim == 1).
    cos = (out * x).sum(dim=1) / (
        out.norm(p=2, dim=1) * x.norm(p=2, dim=1)
    ).clamp_min(1e-12)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)


def test_param_count_unchanged_at_arm_v_config():
    cfg_off = _arm_v_config()
    assert cfg_off.feature_normalize is False
    m_off = StereoNet(cfg_off)
    n_off = sum(p.numel() for p in m_off.parameters())
    cfg_on = _arm_v_config()
    cfg_on.feature_normalize = True
    m_on = StereoNet(cfg_on)
    n_on = sum(p.numel() for p in m_on.parameters())
    assert n_off == 397954, n_off
    assert n_on == 397954, n_on
    assert n_on == n_off


def test_cost_volume_shape_with_normalize():
    torch.manual_seed(2)
    B, C, D, H, W = 1, 32, 24, 4, 9
    left = torch.randn(B, C, H, W)
    right = torch.randn(B, C, H, W)
    vol = build_cost_volume(
        normalize_features_l2(left),
        normalize_features_l2(right),
        D, method="subtract", shift="right",
    )
    assert vol.shape == (B, C, D, H, W)


def test_sign_is_lhat_minus_shifted_rhat():
    torch.manual_seed(3)
    B, C, H, W, D = 1, 3, 2, 7, 4
    left = torch.randn(B, C, H, W)
    right = torch.randn(B, C, H, W)
    lh = normalize_features_l2(left)
    rh = normalize_features_l2(right)
    vol = build_cost_volume(lh, rh, D, method="subtract", shift="right")
    for k in range(D):
        expected = lh - shift_right(rh, k)
        got = vol[:, :, k]
        assert (got - expected).abs().max().item() == 0.0
        negated = lh + shift_right(rh, k) if k == 0 else None
        _ = negated
    # Explicit no-negation: slice 0 must equal lh - rh, not lh + rh.
    assert (vol[:, :, 0] - (lh - rh)).abs().max().item() == 0.0
    assert (vol[:, :, 0] - (lh + rh)).abs().max().item() > 1e-6


def test_flag_off_reproduces_arm_v_volume_bitwise():
    torch.manual_seed(4)
    cfg_off = _arm_v_config()
    m_off = StereoNet(cfg_off)
    m_off.eval()
    torch.manual_seed(5)
    left = torch.randn(1, 3, 64, 96)
    right = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        lf = m_off.feature_extractor(left)
        rf = m_off.feature_extractor(right)
        vol_off = m_off.cost_volume(lf, rf)
        vol_raw = build_cost_volume(
            lf, rf, 24, method="subtract", shift="right",
        )
    assert (vol_off - vol_raw).abs().max().item() == 0.0
    # A second identical model matches bit-identically (default unchanged).
    torch.manual_seed(4)
    m_off2 = StereoNet(_arm_v_config())
    m_off2.eval()
    with torch.no_grad():
        lf2 = m_off2.feature_extractor(left)
        rf2 = m_off2.feature_extractor(right)
        vol_off2 = m_off2.cost_volume(lf2, rf2)
    assert (vol_off - vol_off2).abs().max().item() == 0.0
