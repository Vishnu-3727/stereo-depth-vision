"""Disparity training loss.

The upstream training scripts use a smooth L1 loss on valid ground-truth pixels
[SOURCE: SR-011], which is the standard choice across the stereo literature: L2
near zero so small errors are penalised smoothly, L1 in the tail so a handful of
badly wrong pixels do not dominate the gradient. Stereo ground truth has exactly
that structure -- mostly small errors with occasional gross outliers at
occlusions and depth discontinuities.

Masking is the part that is easy to get wrong. KITTI ground truth is sparse and
a stored zero means *no data*; including those pixels would train the network to
predict zero disparity across the sky.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_disparity: float | None = None,
    beta: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Smooth L1 over valid ground-truth pixels only.

    Valid means ``target > 0``, and additionally ``target < max_disparity`` when
    that is given -- a pixel whose true disparity is outside the model's search
    range cannot be predicted correctly and supplies a gradient that only
    teaches the network to saturate.

    Returns ``(loss, valid_pixel_count)``. The count is returned so a caller can
    tell an easy batch from one that was almost entirely masked out, rather than
    reading a small loss as good news.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must match: "
            + str(tuple(prediction.shape)) + " vs " + str(tuple(target.shape))
        )
    valid = target > 0
    if max_disparity is not None:
        valid = valid & (target < max_disparity)
    count = int(valid.sum())
    if count == 0:
        # No supervision in this batch. Return a zero that still carries a
        # gradient path, so the optimiser step is a no-op rather than a crash.
        return (prediction.sum() * 0.0), 0
    return F.smooth_l1_loss(prediction[valid], target[valid], beta=beta), count


def demo() -> None:
    """Self-check: masking excludes invalid pixels and the loss behaves."""
    pred = torch.zeros(1, 1, 4, 4, requires_grad=True)
    target = torch.zeros(1, 1, 4, 4)

    # no valid pixels at all -> zero loss, no crash, gradient path intact
    loss, n = masked_smooth_l1(pred, target)
    assert n == 0 and float(loss) == 0.0
    loss.backward()

    # a perfect prediction on the valid pixels scores zero
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :2] = 10.0
    p = target.clone().requires_grad_(True)
    loss, n = masked_smooth_l1(p, target)
    assert n == 8, n
    assert float(loss) == 0.0

    # invalid pixels are genuinely ignored: changing them changes nothing
    p2 = target.clone()
    p2[0, 0, 2:] = 999.0
    loss2, n2 = masked_smooth_l1(p2.requires_grad_(True), target)
    assert n2 == 8 and float(loss2) == 0.0

    # out-of-range ground truth is excluded when a range is given
    target3 = torch.full((1, 1, 2, 2), 500.0)
    _, n3 = masked_smooth_l1(torch.zeros(1, 1, 2, 2), target3, max_disparity=176.0)
    assert n3 == 0, n3

    # the loss is smooth at zero and linear in the tail
    t = torch.zeros(1, 1, 1, 1)
    t[:] = 10.0
    small, _ = masked_smooth_l1(t + 0.1, t)
    large, _ = masked_smooth_l1(t + 10.0, t)
    assert float(small) < 0.01          # quadratic region
    assert 9.0 < float(large) < 10.0    # linear region, not quadratic

    print("disparity loss self-check passed")


if __name__ == "__main__":
    demo()
