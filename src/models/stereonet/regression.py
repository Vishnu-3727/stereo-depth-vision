"""Disparity regression by soft-argmin, and the full-resolution upsampling.

From ``docs/stereonet_architecture.md``, section 5 [VERIFIED: SR-001]. The
deployed graph does something unusual: it upsamples the **whole cost tensor**
from 1/16 resolution to full resolution and takes the soft-argmin there, rather
than reducing first and upsampling a single-channel disparity map.

    Resize(bilinear, align_corners) : (B, 12, 23, 77) -> (B, 12, 368, 1232)
    Neg -> Softmax(dim=1) -> Mul(index grid) -> ReduceSum(dim=1, keepdim)

The negation is what makes it an arg*min*: low cost becomes high weight. The
softmax makes the output continuous rather than an integer index, which is where
the model's sub-pixel precision comes from.

One thing worth being precise about: the candidates are 16 full-resolution
pixels apart, so soft-argmin is interpolating inside 16-pixel intervals with no
matching evidence anywhere between them. Its continuity is real; its accuracy
between candidates depends entirely on the aggregation network having shaped the
cost curve correctly.

``upsample_first`` exposes the ordering as a flag, defaulting to the deployed
behaviour. The alternative is available for measurement in Phase 2 and is **not**
used to alter the baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_argmin(cost: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Expected disparity index under softmax(-cost).

    ``cost`` is ``(B, D, H, W)``; the result is ``(B, 1, H, W)`` in units of
    disparity *candidates*. Converting candidates to full-resolution pixels is
    the caller's job, and in this architecture the scale factor is folded into
    the upsampling rather than applied here -- see the note in
    :class:`DisparityRegression`.
    """
    weights = torch.softmax(-cost, dim=dim)
    index = torch.arange(cost.shape[dim], dtype=cost.dtype, device=cost.device)
    shape = [1] * cost.dim()
    shape[dim] = -1
    return (weights * index.view(shape)).sum(dim=dim, keepdim=True)


class DisparityRegression(nn.Module):
    """Cost tensor at 1/16 resolution to disparity at full resolution."""

    def __init__(self, upsample_first: bool = True) -> None:
        super().__init__()
        self.upsample_first = upsample_first

    def forward(self, cost: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if self.upsample_first:
            cost = F.interpolate(cost, size=size, mode="bilinear", align_corners=True)
            return soft_argmin(cost, dim=1)
        disparity = soft_argmin(cost, dim=1)
        return F.interpolate(disparity, size=size, mode="bilinear", align_corners=True)
