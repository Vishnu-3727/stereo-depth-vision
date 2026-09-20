"""Guided refinement at full resolution.

From ``docs/stereonet_architecture.md``, section 6 [VERIFIED: SR-001]:

- concatenate the disparity estimate with the left RGB image, 1 + 3 = 4 channels
- ``Conv2d(4 -> 32, 3x3)``
- six residual blocks with dilations 1, 2, 4, 8, 1, 1
- ``Conv2d(32 -> 1, 3x3)``

The colour image is the guidance: it tells the network where object boundaries
are, so disparity edges can be placed at image edges rather than smeared by the
16x upsampling. The dilation ladder widens the receptive field enough to
propagate a disparity across a textureless region at full resolution.

This stage carries 90.6 % of the model's MACs and 90.8 % of its activation
traffic [MEASUREMENT: EXP-001], and every activation inside it is 55.34 MiB. It
is also the region Hailo's compiler spatially defuses into up to 22 pieces
[SOURCE: SR-003].

It is worth being clear about what this stage actually does here, because the
name understates it. EXP-006 and EXP-008 measured that the incoming disparity
estimate is anti-correlated with ground truth (r = -0.98) and on the wrong
scale, while this stage's residual output is almost perfectly correlated with
the final answer (r = +0.9998) and supplies 76.5 % of its magnitude. So the
refinement is not polishing a nearly-correct estimate -- it is inverting,
rescaling and largely producing the disparity. Recorded, not changed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ResBlock

DEFAULT_DILATIONS = (1, 2, 4, 8, 1, 1)


class Refinement(nn.Module):
    def __init__(
        self,
        guidance_channels: int = 3,
        channels: int = 32,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(1 + guidance_channels, channels, 3, stride=1, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels, d) for d in dilations])
        self.output_conv = nn.Conv2d(channels, 1, 3, stride=1, padding=1)

    def forward(self, disparity: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        """Returns the residual to add to ``disparity``, not the final map."""
        x = torch.cat([disparity, guidance], dim=1)
        x = self.input_conv(x)
        x = self.blocks(x)
        return self.output_conv(x)
