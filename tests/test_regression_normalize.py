"""ARM T — regression normalisation checks.

a. regression_normalize=False is bit-identical to the current soft_argmin
   (ARM K untouched).
b. Scale-invariance: normalised readout gives the same disparity for C and
   s*C (within 1e-4); the unnormalised one does not.
c. Un-saturation: at magnitudes ~1e7 with a spatially varying argmin, the
   normalised readout has std > 0 across pixels and is NOT pinned to a single
   candidate index.
"""
from __future__ import annotations

import torch

from src.models.stereonet.regression import DisparityRegression, soft_argmin


def test_default_path_untouched():
    torch.manual_seed(0)
    cost = torch.randn(2, 12, 5, 7) * 10.0
    a = soft_argmin(cost)
    b = soft_argmin(cost, normalize=False)
    assert (a - b).abs().max().item() == 0.0
    # Module default must also match the legacy function exactly.
    r0 = DisparityRegression()(cost, (9, 11))
    r1 = DisparityRegression(normalize=False)(cost, (9, 11))
    assert (r0 - r1).abs().max().item() == 0.0


def test_normalized_is_scale_invariant_but_plain_is_not():
    torch.manual_seed(1)
    cost = torch.randn(1, 12, 4, 6)
    s = 37.0
    d_plain_c = soft_argmin(cost)
    d_plain_scaled = soft_argmin(s * cost)
    d_norm_c = soft_argmin(cost, normalize=True)
    d_norm_scaled = soft_argmin(s * cost, normalize=True)
    assert (d_norm_c - d_norm_scaled).abs().max().item() < 1e-4
    # The unnormalised readout must be scale-sensitive on this input.
    assert (d_plain_c - d_plain_scaled).abs().max().item() > 1e-4


def test_normalized_unsaturates_at_1e7():
    torch.manual_seed(2)
    B, D, H, W = 1, 12, 4, 9
    # Cost with magnitude ~1e7 and a spatially varying argmin: at pixel
    # (i, j) candidate (j % D) is the unique minimum, everything else is
    # ~1e7 above it. Plain soft-argmin may still resolve this particular
    # construction (gaps are O(1e7)); the assertions that matter are on the
    # normalised readout: it must vary spatially and must not collapse.
    base = torch.randn(B, D, H, W) * 1e6
    argmin_map = torch.arange(W).unsqueeze(0).repeat(H, 1) % D  # (H, W)
    cost = base + 1e7
    for j in range(W):
        k = int(argmin_map[0, j].item())
        cost[:, k, :, j] -= 1e7
    d_norm = soft_argmin(cost, normalize=True)
    assert d_norm.shape == (B, 1, H, W)
    assert float(d_norm.std().item()) > 0.0
    uniq = torch.unique(d_norm.round())
    assert uniq.numel() > 1, "normalised readout pinned to a single value"
    # Must track the argmin spatially: correlation between readout mean over
    # rows and the true argmin index across columns.
    col_mean = d_norm[0, 0].mean(dim=0)
    assert float(col_mean.max().item() - col_mean.min().item()) > 0.5
