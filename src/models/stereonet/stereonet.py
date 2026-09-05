"""StereoNet as Hailo deploys it, reimplemented independently.

Assembled from ``docs/stereonet_architecture.md``, which was reconstructed from
the exported ONNX [SR-001]. This is not a port of the upstream PyTorch source;
it is a second implementation written from the specification, so that agreement
with the reference is evidence the specification is right.

Flow, matching the exported graph exactly:

    left, right  (B, 3, 368, 1232), already normalised
        -> shared feature extractor           -> (B, 32, 23, 77) each
        -> cost volume by subtraction, 12     -> (B, 32, 12, 23, 77)
        -> 3D aggregation                     -> (B, 12, 23, 77)
        -> upsample to full res, soft-argmin  -> (B, 1, 368, 1232)
        -> refinement guided by left image    -> residual
        -> add, ReLU                          -> (B, 1, 368, 1232)

Two properties of the deployed model that this reproduces rather than corrects,
because the baseline is frozen:

- the downsampling stack has no activations, so it is linear;
- the soft-argmin runs at full resolution over an upsampled cost tensor, and
  emits values in units of disparity *candidates* (0..11), not pixels -- the
  refinement stage supplies the scale;
- the cost volume's disparity shift is a no-op, so all 12 candidate slices are
  identical and no disparity search takes place (EXP-010).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .aggregation import Aggregation
from .cost_volume import CostVolume
from .feature_extractor import FeatureExtractor
from .refinement import Refinement
from .regression import DisparityRegression


@dataclass
class StereoNetConfig:
    """The deployed configuration. Defaults are the values verified in the ONNX."""

    in_channels: int = 3
    feature_channels: int = 32
    downsample_levels: int = 4        # -> 1/16 resolution
    residual_blocks: int = 6
    num_disparities: int = 12         # 192 // 16
    cost_volume_method: str = "subtract"
    # The reference model's disparity shift is a no-op (EXP-010), so "none" is
    # what reproduces it. "left" is the intended behaviour, available for later
    # measurement and deliberately not the default.
    cost_volume_shift: str = "none"
    aggregation_layers: int = 4
    refinement_dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 1)
    upsample_before_argmin: bool = True
    final_relu: bool = True

    @property
    def feature_stride(self) -> int:
        return 2**self.downsample_levels

    @property
    def max_disparity_px(self) -> int:
        """Largest full-resolution disparity the search can represent."""
        return (self.num_disparities - 1) * self.feature_stride


class StereoNet(nn.Module):
    def __init__(self, config: StereoNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or StereoNetConfig()
        c = self.config

        self.feature_extractor = FeatureExtractor(
            in_channels=c.in_channels,
            channels=c.feature_channels,
            downsample_levels=c.downsample_levels,
            residual_blocks=c.residual_blocks,
        )
        self.cost_volume = CostVolume(
            c.num_disparities, c.cost_volume_method, c.cost_volume_shift
        )
        agg_in = c.feature_channels * (2 if c.cost_volume_method == "concat" else 1)
        self.aggregation = Aggregation(
            in_channels=agg_in, channels=c.feature_channels, num_layers=c.aggregation_layers
        )
        self.regression = DisparityRegression(upsample_first=c.upsample_before_argmin)
        self.refinement = Refinement(
            guidance_channels=c.in_channels,
            channels=c.feature_channels,
            dilations=c.refinement_dilations,
        )

    def forward(
        self, left: torch.Tensor, right: torch.Tensor, return_stages: bool = False
    ):
        size = (left.shape[-2], left.shape[-1])

        left_features = self.feature_extractor(left)
        right_features = self.feature_extractor(right)
        volume = self.cost_volume(left_features, right_features)
        cost = self.aggregation(volume)
        disparity_initial = self.regression(cost, size)
        residual = self.refinement(disparity_initial, left)
        disparity = disparity_initial + residual
        if self.config.final_relu:
            disparity = torch.relu(disparity)

        if not return_stages:
            return disparity
        return disparity, {
            "left_features": left_features,
            "right_features": right_features,
            "cost_volume": volume,
            "aggregated_cost": cost,
            "disparity_initial": disparity_initial,
            "refinement_residual": residual,
            "disparity_final": disparity,
        }

    def parameter_count(self, per_occurrence: bool = False) -> int:
        """Learned parameters.

        ``per_occurrence`` adds the shared feature extractor a second time,
        reproducing Hailo's counting convention -- the one that yields the
        published 623.1K rather than the 423,586 unique weights
        [MEASUREMENT: EXP-001].
        """
        total = sum(p.numel() for p in self.parameters())
        if per_occurrence:
            total += sum(p.numel() for p in self.feature_extractor.parameters())
        return total
