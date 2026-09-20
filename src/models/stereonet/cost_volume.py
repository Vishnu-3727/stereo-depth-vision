"""Cost volume construction, including the reference model's degenerate shift.

The deployed reference builds each disparity level as [VERIFIED: SR-001]:

    Concat([left_features (width W), zeros (width k)], axis=3)   -> width W + k
    Slice(starts=[0], ends=[W], axes=[3])                        -> width W
    Sub(that, right_features)

Concatenating the zero block on the **right** and then slicing ``[0:W]`` returns
the left features unchanged, so **the shift never happens**. Every one of the 12
disparity levels computes the identical tensor ``left - right``. This was proved
directly in EXP-010: across five scenes, the maximum absolute difference between
any disparity slice and slice 0 is exactly 0.0. The same construction is present
in the upstream PyTorch source [SR-011], so it originates there and was carried
faithfully through export and compilation.

The consequence is that the deployed model performs no disparity search at all.
That is a property of the frozen baseline, so ``shift="none"`` is the default
here and is what every Phase 1 experiment uses. The intended behaviour is
available as ``shift="left"`` purely so the difference can be measured later; it
is **not** used to alter the baseline in Phase 1.

Output shape is ``(B, C, D, H, W)``: channels first, disparity as the depth
axis, ready for ``Conv3d``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def shift_left(x: torch.Tensor, k: int) -> torch.Tensor:
    """Shift ``x`` left by ``k`` columns, filling the right edge with zeros.

    This is what the construction is meant to do; it is not what the reference
    model does. See :func:`reference_shift`.
    """
    if k == 0:
        return x
    return F.pad(x, (0, k))[..., k:]


def shift_right(x: torch.Tensor, k: int) -> torch.Tensor:
    """Shift ``x`` right by ``k`` columns, filling the left edge with zeros.

    Pads ``k`` zeros on the left then slices ``[..., :width]``. Applied to the
    RIGHT features, candidate ``k`` computes ``left(x) - right(x - k)``, i.e.
    the LEFT-frame matching cost for disparity ``k``.
    """
    if k == 0:
        return x
    width = x.shape[-1]
    padded = F.pad(x, (k, 0))          # [zeros | features]
    return padded[..., :width]         # drop the rightmost k columns


def reference_shift(x: torch.Tensor, k: int) -> torch.Tensor:
    """The reference model's shift, reproduced exactly -- a no-op.

    Written out rather than replaced by ``return x`` so the mechanism stays
    visible: pad on the right, then slice from zero, which recovers the input.
    """
    width = x.shape[-1]
    if k == 0:
        return x
    padded = F.pad(x, (0, k))        # [features | zeros]
    return padded[..., :width]       # slicing from 0 gives the features back


def build_cost_volume(
    left: torch.Tensor,
    right: torch.Tensor,
    num_disparities: int,
    method: str = "subtract",
    shift: str = "none",
) -> torch.Tensor:
    """Cost volume. Returns ``(B, C, D, H, W)``.

    ``method`` accepts ``"subtract"`` (what the deployed model uses) and
    ``"concat"`` (what the upstream README claims it uses, kept so the
    difference can be studied without editing this file). Concatenation doubles
    the channel dimension.

    ``shift`` selects the disparity shift. ``"none"`` reproduces the reference
    model, whose shift is a no-op, and is the default. ``"left"`` performs the
    shift the construction was intended to perform (left features shifted left,
    RIGHT-frame indexing). ``"right"`` shifts the RIGHT features right, giving
    ``cost_k(x) = left(x) - right(x - k)`` indexed in the LEFT frame, which is
    what KITTI left-view ground truth requires. The modes produce very
    different volumes; ``"none"`` produces one whose slices are all identical.
    """
    if left.shape != right.shape:
        raise ValueError(
            "feature maps must match: " + str(tuple(left.shape))
            + " vs " + str(tuple(right.shape))
        )
    if shift not in {"none", "left", "right"}:
        raise ValueError("unknown shift mode: " + shift)
    levels = []
    for k in range(num_disparities):
        if shift == "none":
            shifted_left = reference_shift(left, k)
            anchor, other = shifted_left, right
        elif shift == "left":
            anchor, other = shift_left(left, k), right
        else:  # shift == "right"
            anchor, other = left, shift_right(right, k)
        if method == "subtract":
            levels.append(anchor - other)
        elif method == "concat":
            levels.append(torch.cat([anchor, other], dim=1))
        else:
            raise ValueError("unknown cost volume method: " + method)
    # stack on a new axis after batch, then move channels back in front so the
    # disparity axis becomes Conv3d's depth dimension
    volume = torch.stack(levels, dim=1)          # (B, D, C, H, W)
    return volume.permute(0, 2, 1, 3, 4)          # (B, C, D, H, W)


def build_cost_volume_reference(
    left: torch.Tensor, right: torch.Tensor, num_disparities: int, shift: str = "left"
) -> torch.Tensor:
    """Deliberately slow, explicitly indexed version, for testing only.

    Written independently of the vectorised path so that agreement between the
    two is evidence rather than a tautology. Loops over every disparity, row and
    column and writes one element at a time. Defaults to the *intended* shift,
    since its purpose is to state the indexing explicitly; pass ``shift="none"``
    to describe the reference model instead.
    """
    b, c, h, w = left.shape
    out = torch.zeros(b, c, num_disparities, h, w, dtype=left.dtype, device=left.device)
    for k in range(num_disparities):
        offset = k if shift == "left" else 0
        for y in range(h):
            for x in range(w):
                src = x + offset
                lhs = left[:, :, y, src] if src < w else torch.zeros_like(left[:, :, y, 0])
                out[:, :, k, y, x] = lhs - right[:, :, y, x]
    return out


class CostVolume(nn.Module):
    def __init__(
        self,
        num_disparities: int = 12,
        method: str = "subtract",
        shift: str = "none",
        channels: int = 32,
        groups: int = 0,
    ) -> None:
        super().__init__()
        self.num_disparities = num_disparities
        self.method = method
        self.shift = shift
        self.channels = channels
        self.groups = groups
        if groups > 0:
            assert channels % groups == 0
            self.reduce = nn.Conv2d(channels, groups, 1, groups=groups, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        volume = build_cost_volume(
            left, right, self.num_disparities, self.method, self.shift
        )
        if self.groups > 0:
            reduced = [self.reduce(volume[:, :, d]) for d in range(volume.shape[2])]
            return torch.stack(reduced, dim=2)
        return volume

    def memory_bytes(self, batch: int, channels: int, h: int, w: int, dtype_bytes: int = 4) -> int:
        """B x C x D x H x W x bytes, the figure used in the analysis document."""
        if self.groups > 0:
            c = self.groups
        else:
            c = channels * (2 if self.method == "concat" else 1)
        return batch * c * self.num_disparities * h * w * dtype_bytes
