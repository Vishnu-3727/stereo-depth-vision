"""Cost volume aggregation with 3D convolutions.

From ``docs/stereonet_architecture.md``, section 4 [VERIFIED: SR-001]: four
``Conv3d(32 -> 32, 3x3x3, padding 1)`` each followed by LeakyReLU with slope
0.01, then a final ``Conv3d(32 -> 1, 3x3x3, padding 1)`` and a squeeze of the
channel axis.

The 3x3x3 kernel spans the disparity axis as well as the spatial axes, which is
the point: a pixel with no texture of its own gets its cost curve shaped by its
neighbours, and each disparity candidate is informed by the ones adjacent to it.
That is how evidence propagates into regions where matching alone would fail.

Input ``(B, C, D, H, W)``, output ``(B, D, H, W)`` -- one scalar cost per
disparity candidate per pixel.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import AGGREGATION_SLOPE


class Aggregation(nn.Module):
    def __init__(self, in_channels: int = 32, channels: int = 32, num_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for _ in range(num_layers):
            layers.append(nn.Conv3d(c_in, channels, 3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(negative_slope=AGGREGATION_SLOPE, inplace=False))
            c_in = channels
        self.filter = nn.Sequential(*layers)
        self.to_cost = nn.Conv3d(channels, 1, 3, stride=1, padding=1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        x = self.filter(volume)
        x = self.to_cost(x)          # (B, 1, D, H, W)
        return x.squeeze(1)           # (B, D, H, W)
