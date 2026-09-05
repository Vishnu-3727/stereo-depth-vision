"""Disparity and depth metrics, with the evaluation protocol made explicit.

Two protocols live here and they are deliberately kept apart:

``hailo_protocol``
    An independent reimplementation of what Hailo's Model Zoo evaluator
    actually computes for this model. It is *not* end-point error, despite
    being labelled ``EPE`` in the configuration and the published tables. It is
    the KITTI D1 outlier rate -- the fraction of valid pixels whose disparity
    error exceeds both 3 px and 5 % of ground truth -- averaged per image and
    reported as a percentage. Reconstructed from
    ``hailo_model_zoo/core/eval/stereo_evaluation.py`` [SR-002 repository].

``kitti_official``
    The KITTI 2015 convention: ground truth scaled by 1/256, outliers pooled
    over all valid pixels rather than averaged per image, plus genuine EPE.

They give different numbers on the same predictions. Reporting one under the
other's name is the single easiest way to produce a meaningless comparison, so
every function here states which convention it uses.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# KITTI stores disparity in 16-bit PNGs as round(disparity * 256).
KITTI_DISPARITY_SCALE = 256.0
# Hailo's dataset parser divides the raw PNG values by 255 instead, making its
# ground-truth disparities 256/255 = 1.0039x larger than the official values.
HAILO_DISPARITY_SCALE = 255.0


@dataclass
class DisparityMetrics:
    """Disparity accuracy over one set of valid pixels."""

    valid_pixels: int
    epe: float           # mean absolute disparity error, pixels
    rmse: float          # root mean square disparity error, pixels
    d1: float            # KITTI outlier rate, percent
    bad1: float          # error > 1 px, percent
    bad2: float          # error > 2 px, percent
    bad3: float          # error > 3 px, percent

    def as_dict(self) -> dict:
        return asdict(self)


def disparity_metrics(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None
) -> DisparityMetrics:
    """Disparity metrics pooled over every valid pixel.

    ``valid`` defaults to ``gt > 0``, which is KITTI's convention: a stored
    zero means *no ground truth here*, not *zero disparity*. Treating those
    pixels as measurements is a standard way to get a wrong number.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if valid is None:
        valid = gt > 0
    valid = np.asarray(valid, dtype=bool)

    n = int(valid.sum())
    if n == 0:
        raise ValueError("no valid ground-truth pixels to evaluate")

    err = np.abs(pred[valid] - gt[valid])
    # KITTI D1: an outlier fails BOTH tests -- more than 3 px off AND more than
    # 5 % off. A pixel passing either test counts as correct.
    outlier = (err > 3.0) & (err > 0.05 * gt[valid])

    return DisparityMetrics(
        valid_pixels=n,
        epe=float(err.mean()),
        rmse=float(np.sqrt((err**2).mean())),
        d1=float(100.0 * outlier.mean()),
        bad1=float(100.0 * (err > 1.0).mean()),
        bad2=float(100.0 * (err > 2.0).mean()),
        bad3=float(100.0 * (err > 3.0).mean()),
    )


def hailo_image_outlier_rate(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hailo's per-image figure, as a fraction in [0, 1].

    Reimplemented to match ``StereoNetEval.update_op``: valid pixels are
    ``gt > 0``; a pixel is correct when the absolute error is below 3 px *or*
    below 5 % of ground truth; the returned value is one minus the correct
    fraction. Hailo accumulates this per image, divides by the image count, and
    displays the result as a percentage -- which is where the published 8.223
    comes from.

    Note the strictness difference from :func:`disparity_metrics`: Hailo uses
    ``<`` where the KITTI convention uses ``<=`` at the boundary. The effect is
    confined to pixels landing exactly on the threshold and is negligible on
    real data, but it is a difference and it is recorded rather than smoothed
    over.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    index = gt > 0
    if not index.any():
        raise ValueError("no valid ground-truth pixels to evaluate")
    diff = np.abs(gt[index] - pred[index])
    correct = (diff < 3.0) | (diff < gt[index] * 0.05)
    return float(1.0 - correct.sum() / index.sum())


@dataclass
class DepthMetrics:
    """Depth accuracy in metres, over pixels valid in both prediction and truth."""

    valid_pixels: int
    abs_rel: float
    sq_rel: float
    rmse: float
    rmse_log: float
    delta1: float  # percent of pixels within a 1.25 ratio
    delta2: float
    delta3: float

    def as_dict(self) -> dict:
        return asdict(self)


def depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid: np.ndarray | None = None,
    min_depth: float = 1e-3,
    max_depth: float = np.inf,
) -> DepthMetrics:
    """Standard depth metrics.

    A pixel counts only when both prediction and ground truth are finite,
    positive and inside the depth window. Disparity at or below zero produces
    an undefined depth, and those pixels are dropped here rather than clamped
    to a large finite value.
    """
    pred_depth = np.asarray(pred_depth, dtype=np.float64)
    gt_depth = np.asarray(gt_depth, dtype=np.float64)
    ok = (
        np.isfinite(pred_depth)
        & np.isfinite(gt_depth)
        & (pred_depth > min_depth)
        & (gt_depth > min_depth)
        & (gt_depth <= max_depth)
    )
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)

    n = int(ok.sum())
    if n == 0:
        raise ValueError("no pixels valid in both prediction and ground truth")

    p = pred_depth[ok]
    g = gt_depth[ok]
    ratio = np.maximum(p / g, g / p)

    return DepthMetrics(
        valid_pixels=n,
        abs_rel=float((np.abs(p - g) / g).mean()),
        sq_rel=float((((p - g) ** 2) / g).mean()),
        rmse=float(np.sqrt(((p - g) ** 2).mean())),
        rmse_log=float(np.sqrt(((np.log(p) - np.log(g)) ** 2).mean())),
        delta1=float(100.0 * (ratio < 1.25).mean()),
        delta2=float(100.0 * (ratio < 1.25**2).mean()),
        delta3=float(100.0 * (ratio < 1.25**3).mean()),
    )


def bin_by_depth(
    values: np.ndarray, gt_depth: np.ndarray, edges: list[float], valid: np.ndarray
) -> dict[str, dict]:
    """Summarise per-pixel values inside depth bands.

    A single average over the whole image hides the thing that matters most
    here: the same disparity error is worth centimetres up close and tens of
    metres far away. Every accuracy result in this project is reported binned.
    """
    out: dict[str, dict] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = valid & (gt_depth >= lo) & (gt_depth < hi)
        n = int(band.sum())
        label = "{:g}-{:g}m".format(lo, hi)
        if n == 0:
            out[label] = {"pixels": 0, "mean": None, "median": None, "p95": None}
            continue
        v = values[band]
        out[label] = {
            "pixels": n,
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
        }
    return out


def demo() -> None:
    """Self-check: the metrics agree with hand-computed values on planted data,
    and the two protocols are shown to disagree."""
    rng = np.random.default_rng(0)

    # perfect prediction
    gt = rng.uniform(1.0, 100.0, size=(64, 64))
    m = disparity_metrics(gt.copy(), gt)
    assert m.epe == 0.0 and m.d1 == 0.0 and m.bad1 == 0.0

    # zeros in ground truth are excluded, not scored
    gt2 = gt.copy()
    gt2[:32] = 0.0
    m2 = disparity_metrics(np.zeros_like(gt2), gt2)
    assert m2.valid_pixels == 32 * 64, m2.valid_pixels

    # a planted 4 px error: above the 3 px test, so D1 depends on the 5 % test
    gt3 = np.full((10, 10), 50.0)      # 5 % of 50 is 2.5, so 4 px fails both
    pred3 = gt3 + 4.0
    assert disparity_metrics(pred3, gt3).d1 == 100.0
    gt4 = np.full((10, 10), 200.0)     # 5 % of 200 is 10, so 4 px passes
    assert disparity_metrics(gt4 + 4.0, gt4).d1 == 0.0

    # the D1 rate and Hailo's per-image rate agree on a single image
    gt5 = rng.uniform(1.0, 100.0, size=(48, 48))
    pred5 = gt5 + rng.normal(0.0, 4.0, size=gt5.shape)
    d1 = disparity_metrics(pred5, gt5).d1
    hailo = 100.0 * hailo_image_outlier_rate(pred5, gt5)
    assert abs(d1 - hailo) < 1e-9, (d1, hailo)

    # ... but averaging per image is not the same as pooling over pixels when
    # images have different valid-pixel counts, which is the real protocol
    # difference. Two images, very different valid areas:
    a_gt = np.full((10, 10), 50.0)
    a_gt[:, 5:] = 0.0                  # 50 valid pixels, all wrong
    a_pred = a_gt + 10.0
    b_gt = np.full((100, 100), 50.0)   # 10000 valid pixels, all right
    b_pred = b_gt.copy()
    per_image = 100.0 * np.mean(
        [hailo_image_outlier_rate(a_pred, a_gt), hailo_image_outlier_rate(b_pred, b_gt)]
    )
    pooled = disparity_metrics(
        np.concatenate([a_pred.ravel(), b_pred.ravel()]),
        np.concatenate([a_gt.ravel(), b_gt.ravel()]),
    ).d1
    assert abs(per_image - 50.0) < 1e-9, per_image
    assert pooled < 1.0, pooled
    assert abs(per_image - pooled) > 40.0, (per_image, pooled)

    # depth metrics on a perfect prediction
    depth = rng.uniform(2.0, 80.0, size=(32, 32))
    dm = depth_metrics(depth.copy(), depth)
    assert dm.abs_rel == 0.0 and dm.delta1 == 100.0

    # undefined depths are dropped, never scored as huge errors
    pred_depth = depth.copy()
    pred_depth[:16] = np.inf
    dm2 = depth_metrics(pred_depth, depth)
    assert dm2.valid_pixels == 16 * 32, dm2.valid_pixels
    assert dm2.abs_rel == 0.0

    # binning splits the range, and the bands sum to the valid total
    valid = np.ones_like(depth, dtype=bool)
    bins = bin_by_depth(np.abs(depth - depth), depth, [0, 20, 50, 80, 1000], valid)
    assert sum(b["pixels"] for b in bins.values()) == valid.sum()

    print("metrics self-check passed")


if __name__ == "__main__":
    demo()
