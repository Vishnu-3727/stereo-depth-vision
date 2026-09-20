"""Shared building blocks.

Written from the specification in ``docs/stereonet_architecture.md``, which was
derived from the exported ONNX [SR-001]. Deliberately not a copy of the upstream
PyTorch source: the point of the independent implementation is that it is an
independent reading of the same evidence.

One consequence of building from the ONNX rather than the training source: the
exported graph has **no batch normalisation** -- it was folded into the
convolution weights and biases at export time. So the convolutions here carry
biases and there are no normalisation layers, which matches what actually runs.
The training-time model would need batch norm restored; that is handled
separately in the training pipeline rather than by making inference carry
layers the deployed artifact does not have.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# LeakyReLU slopes read from the ONNX attributes [VERIFIED: SR-001]:
# 0.2 in the residual and refinement blocks, 0.01 in the 3D aggregation.
RESIDUAL_SLOPE = 0.2
AGGREGATION_SLOPE = 0.01


class ResBlock(nn.Module):
    """Conv - LeakyReLU - Conv - add input - LeakyReLU, 3x3, stride 1.

    Padding equals the dilation, which keeps the spatial size unchanged for any
    dilation rate -- necessary because the residual add requires the block
    output to match its input exactly.
    """

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, 3, stride=1, padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv2d(
            channels, channels, 3, stride=1, padding=dilation, dilation=dilation
        )
        self.act = nn.LeakyReLU(negative_slope=RESIDUAL_SLOPE, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.act(out + x)
