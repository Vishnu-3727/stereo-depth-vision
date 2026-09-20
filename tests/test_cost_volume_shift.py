"""ARM S — cost volume shift-mode checks.

a. shift=none still yields 12 identical slices (ARM K unchanged).
b. shift=right yields slices that are NOT all identical.
c. Sign-convention proof: with right = left rolled by +D columns
   (left(x) == right(x - D) away from borders), per-pixel argmin over
   candidates of the L1 cost equals D in the interior.
"""
from __future__ import annotations

import torch

from src.models.stereonet.cost_volume import build_cost_volume

D = 12


def test_none_produces_identical_slices():
    torch.manual_seed(0)
    left = torch.randn(1, 4, 6, 20)
    right = torch.randn(1, 4, 6, 20)
    vol = build_cost_volume(left, right, D, shift="none")
    assert vol.shape == (1, 4, D, 6, 20)
    slice0 = vol[:, :, 0]
    for k in range(1, D):
        assert (vol[:, :, k] - slice0).abs().max().item() == 0.0


def test_right_produces_nonidentical_slices():
    torch.manual_seed(1)
    left = torch.randn(1, 4, 6, 20)
    right = torch.randn(1, 4, 6, 20)
    vol = build_cost_volume(left, right, D, shift="right")
    assert vol.shape == (1, 4, D, 6, 20)
    diffs = [(vol[:, :, k] - vol[:, :, 0]).abs().max().item() for k in range(1, D)]
    assert any(d > 0.0 for d in diffs), diffs


def test_right_sign_convention_argmin_equals_true_disparity():
    true_d = 5
    assert true_d < D
    B, C, H, W = 1, 2, 6, 24
    torch.manual_seed(2)
    left = torch.randn(B, C, H, W)
    # right(x) = left(x + D) so that left(x) == right(x - D)
    right = torch.zeros_like(left)
    right[..., :-true_d] = left[..., true_d:]
    vol = build_cost_volume(left, right, D, shift="right")  # (B, C, D, H, W)
    l1 = vol.abs().mean(dim=1)  # (B, D, H, W)
    pred = l1.argmin(dim=1)  # (B, H, W)
    # Interior columns where candidate D is exact and no border effect
    # contaminates any candidate: need x - k >= 0 for all k, i.e. x >= D - 1,
    # and x <= W - 1 - true_d so right(x - D) is defined from real data.
    interior = pred[:, :, (D - 1):(W - true_d)]
    assert interior.numel() > 0
    assert (interior == true_d).all(), (
        "expected argmin %d in interior, got unique %s"
        % (true_d, sorted(map(int, interior.unique().tolist())))
    )
