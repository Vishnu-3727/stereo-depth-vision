"""Siamese feature extraction: full resolution down to 1/16.

From ``docs/stereonet_architecture.md``, section 2 [VERIFIED: SR-001]:

- four 5x5 stride-2 convolutions, 3->32 then 32->32 three times, padding 2,
  **with no activation between them**;
- six residual blocks at 32 channels;
- one 3x3 convolution, 32->32.

The absent activations in the downsampling stack are not an oversight in this
reimplementation. Four stacked linear convolutions compose to a single linear
operator, so the stack has no more representational power than one convolution
of the right shape. That is what the deployed artifact does, and the baseline is
reproduced as it is rather than as it arguably should be. It is recorded in the
architecture document as a finding for Phase 2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ResBlock


class FeatureExtractor(nn.Module):
    """Shared tower, applied to both images. 368x1232 -> 23x77, 32 channels."""

    def __init__(
        self, in_channels: int = 3, channels: int = 32,
        downsample_levels: int = 4, residual_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.downsample_levels = downsample_levels
        self.channels = channels

        convs = []
        c_in = in_channels
        for _ in range(downsample_levels):
            convs.append(nn.Conv2d(c_in, channels, 5, stride=2, padding=2))
            c_in = channels
        # No activations here, matching the exported graph.
        self.downsample = nn.Sequential(*convs)

        self.residual = nn.Sequential(*[ResBlock(channels) for _ in range(residual_blocks)])
        self.output_conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

    @property
    def scale(self) -> int:
        """Spatial reduction factor, 16 for the deployed configuration."""
        return 2**self.downsample_levels

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.downsample(image)
        x = self.residual(x)
        return self.output_conv(x)
