"""Left-image-guided cost-volume excitation (ARM-Y).

Injects left-image context before aggregation by reweighting *which channels*
of matching evidence to trust at each pixel. The gate is computed from the
left feature map alone with two 1x1 convolutions and a sigmoid, then
broadcast over the disparity axis, so it is channel-wise, pixel-wise and
DISPARITY-INDEPENDENT by construction: it reweights channels, never
candidates, and cannot prejudge the answer the aggregation network exists
to compute.

Evidence for the mechanism comes from at least four independent public
implementations under three names: CoEx ``channelAtt``, Fast-ACVNet
``channelAtt``, LightStereo ``AttentionModule`` under ``LEFT_ATT``, and the
IGEV/StereoBase/FoundationStereo family (see
``phase1/docs/PUBLIC_STEREO_RESEARCH_AUDIT.md`` sections 8-9 and
``phase1/docs/ARM_Y_PREREGISTRATION.md``).

Deliberate deviation from the audited reference block: BatchNorm is excluded
(the no-BN state matches the BN-folded exported artifact), while the
LeakyReLU is retained so the two 1x1 convolutions do not collapse into a
single linear rank-16 map.

Input ``volume`` is ``(B, C, D, H, W)``, ``left_features`` is ``(B, C, H, W)``;
output matches the volume shape.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CostVolumeExcitation(nn.Module):
    """Left-image-guided cost-volume excitation (ARM-Y)."""

    def __init__(self, cv_channels: int = 32, im_channels: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(im_channels, im_channels // 2, 1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv2d(im_channels // 2, cv_channels, 1)  # bias=True
        for m in (self.conv1, self.conv2):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, volume: torch.Tensor, left_features: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.conv2(self.act(self.conv1(left_features)))).unsqueeze(2)
        return gate * volume
