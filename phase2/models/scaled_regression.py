"""EXP-H2 -- the single changed variable: a scale for the aggregated cost.

Phase 1's `src/models/stereonet/regression.py` is frozen and is not touched.
This module provides a drop-in replacement for `DisparityRegression` that
standardises the cost across the disparity dimension immediately before the
soft-argmin, and nothing else:

    C_scaled[d, y, x] = (C[d, y, x] - mean_d C[:, y, x]) / (std_d C[:, y, x] + eps)
    P = softmax(-C_scaled)               <- Phase 1's own soft_argmin, imported
    disparity_initial = sum_d P[d] * d

Decisions that had to be made, recorded before training rather than after:

- **Where.** The deployed model upsamples the whole cost tensor to full
  resolution and takes the soft-argmin there. "Immediately before the
  soft-argmin" therefore means *after* that upsampling, on the exact tensor the
  softmax consumes -- which is also the tensor the H1 investigation found
  saturated (`phase2/docs/H1_VISUAL_MECHANISM_INVESTIGATION_V1.md` §5). The
  `upsample_first=False` ordering is supported for completeness and standardises
  whatever tensor the soft-argmin sees in that ordering too.
- **Statistics.** Per pixel, across the 12 disparity candidates only. Never
  across the batch, never across space, never using ground truth. Population
  standard deviation (`unbiased=False`); with 12 samples the unbiased correction
  would be a fixed 1.045x factor on every pixel, i.e. an arbitrary global
  temperature, which is exactly what this experiment must not smuggle in.
- **Epsilon.** Fixed at 1e-5, chosen before any accuracy was measured, purely to
  keep the division defined when all 12 candidates are equal. It is not tuned and
  must not be tuned.
- **No new parameters.** No learnable temperature, no annealing, no clipping, no
  constant divisor. The module holds no state and no weights, so the parameter
  count is identical to the control's 423,586.

Nothing else in the network changes: the same feature extractor, the same cost
volume construction (`cost_volume_shift="left"`, as in EXP-H1-WORKING-v2), the
same aggregation, the same refinement, the same loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.stereonet.regression import soft_argmin  # noqa: E402  (frozen, read-only)

EPS = 1e-5


def standardise_across_disparity(cost: torch.Tensor, dim: int = 1,
                                 eps: float = EPS) -> torch.Tensor:
    """Per-pixel z-score across the disparity axis. Returns a new tensor."""
    mean = cost.mean(dim=dim, keepdim=True)
    std = cost.std(dim=dim, keepdim=True, unbiased=False)
    return (cost - mean) / (std + eps)


class StandardisedDisparityRegression(nn.Module):
    """`DisparityRegression` with the cost standardised before the soft-argmin.

    Signature, input and output are identical to the frozen module it replaces:
    ``forward(cost: (B, D, h, w), size: (H, W)) -> (B, 1, H, W)`` in units of
    disparity candidates.
    """

    def __init__(self, upsample_first: bool = True, eps: float = EPS) -> None:
        super().__init__()
        self.upsample_first = upsample_first
        self.eps = eps
        # Instrumentation only: when True, the tensors around the softmax are
        # kept (with retain_grad) so a probe can measure saturation and the
        # gradient reaching them. It changes no arithmetic; leave it off in the
        # training loop except on the batches being measured, since holding the
        # references keeps that graph alive.
        self.capture = False
        self.last: dict[str, torch.Tensor] = {}

    def forward(self, cost: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if self.upsample_first:
            cost = F.interpolate(cost, size=size, mode="bilinear", align_corners=True)
            scaled = standardise_across_disparity(cost, dim=1, eps=self.eps)
            self._capture(cost, scaled)
            return soft_argmin(scaled, dim=1)
        scaled = standardise_across_disparity(cost, dim=1, eps=self.eps)
        self._capture(cost, scaled)
        disparity = soft_argmin(scaled, dim=1)
        return F.interpolate(disparity, size=size, mode="bilinear", align_corners=True)

    def _capture(self, raw: torch.Tensor, scaled: torch.Tensor) -> None:
        if not self.capture:
            self.last = {}
            return
        for tensor in (raw, scaled):
            if tensor.requires_grad:
                tensor.retain_grad()
        self.last = {"raw_cost": raw, "softmax_input": scaled}

    def extra_repr(self) -> str:
        return "upsample_first={}, eps={}".format(self.upsample_first, self.eps)


def apply_to(model) -> None:
    """Swap a built StereoNet's regression stage in place.

    The model's own `config.upsample_before_argmin` is carried over, so the only
    difference from the control is the standardisation itself.
    """
    model.regression = StandardisedDisparityRegression(
        upsample_first=model.config.upsample_before_argmin)


def demo() -> None:
    """Self-check: the scaling does what it claims and breaks nothing else."""
    torch.manual_seed(0)
    cost = torch.randn(2, 12, 5, 7) * 1e12 + 3e12   # H1-scale magnitudes

    scaled = standardise_across_disparity(cost)
    assert scaled.shape == cost.shape
    assert torch.allclose(scaled.mean(dim=1), torch.zeros(2, 5, 7), atol=1e-4)
    assert torch.allclose(scaled.std(dim=1, unbiased=False),
                          torch.ones(2, 5, 7), atol=1e-3)
    # the caller's tensor must not be touched
    before = cost.clone()
    standardise_across_disparity(cost)
    assert torch.equal(cost, before)

    # a constant cost column must not divide by zero
    flat = torch.full((1, 12, 2, 2), 5.0)
    out = standardise_across_disparity(flat)
    assert torch.isfinite(out).all() and float(out.abs().max()) == 0.0

    # saturation: the control's softmax collapses, the scaled one does not
    control_entropy = _entropy(torch.softmax(-cost, dim=1))
    scaled_entropy = _entropy(torch.softmax(-scaled, dim=1))
    assert float(control_entropy.mean()) < 1e-6, float(control_entropy.mean())
    assert float(scaled_entropy.mean()) > 1.0, float(scaled_entropy.mean())

    # gradient reaches the cost tensor after scaling, and does not before
    for tensor, expect in ((cost, False), (None, True)):
        c = (cost if tensor is not None else cost).clone().requires_grad_(True)
        head = soft_argmin(standardise_across_disparity(c) if expect else c, dim=1)
        head.sum().backward()
        nonzero = bool((c.grad != 0).any())
        assert nonzero == expect, (expect, nonzero)

    # the module keeps the interface and adds no parameters
    module = StandardisedDisparityRegression()
    assert list(module.parameters()) == []
    out = module(torch.randn(1, 12, 4, 6), (16, 24))
    assert out.shape == (1, 1, 16, 24)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 11.0
    print("scaled regression self-check passed")


def _entropy(p: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=dim)


if __name__ == "__main__":
    demo()
