"""Stage C2.1: valid-pixel point cloud on top of the frozen C2 metric depth.

Deterministic CPU post-processing only. No neural network is added, trained,
modified, or proposed here. ARM-P stays frozen. C2 stays frozen.

Coordinate convention (reused from C2 exactly, not re-invented): metres, camera
optical centre at origin, +X right, +Y down, +Z forward along the optical axis,
right-handed.

X and Y come from C2's ``reproject_xyz``; this module takes the valid subset
directly and does NOT materialise extra full-resolution copies of
disparity/depth/X/Y/Z. The (u, v) index arrays preserve pixel<->point
correspondence: point i was imaged at pixel (us[i], vs[i]).

The ``stride`` argument produces the VISUALIZATION SUBSAMPLE only: stride=1 is
the full-resolution data. Strided sampling slices views (no copies) and then
copies only the valid subset.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np


def physical_mask(depth: np.ndarray, c2_valid: np.ndarray,
                  physical_max_m: float) -> np.ndarray:
    """SEPARATE physical-range mask, alongside (never instead of) the C2 mask.

    Very small positive disparity yields geometrically valid but physically
    unreliable depth (e.g. ~0.03 px -> Z ~= 12086 m on 000199_10.png). This
    returns ``c2_valid & isfinite(depth) & (depth <= physical_max_m)`` as a NEW
    boolean array. The stored depth is never modified, never clipped, and this
    filter is never applied by default -- callers must pass it explicitly and
    report that they did.
    """
    depth = np.asarray(depth, dtype=np.float64)
    c2_valid = np.asarray(c2_valid, dtype=bool)
    assert physical_max_m > 0, physical_max_m
    return c2_valid & np.isfinite(depth) & (depth <= physical_max_m)


def depth_to_pointcloud(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
                        valid: np.ndarray, stride: int = 1) -> dict:
    """Valid pixels to an (N, 3) float32 point array plus (u, v) index arrays.

    xs/ys/zs are the full-resolution maps from C2's ``reproject_xyz`` (X and Y
    are NOT recomputed here). Only the valid subset is copied; the strided
    slicing beforehand is a view. Returns point count and nbytes.
    """
    assert stride >= 1 and int(stride) == stride, stride
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    zs = np.asarray(zs)
    valid = np.asarray(valid, dtype=bool)
    assert xs.shape == ys.shape == zs.shape == valid.shape, (
        xs.shape, ys.shape, zs.shape, valid.shape)
    h, w = valid.shape

    # Views, not copies: strided subsample for VISUALIZATION only.
    sv = valid[::stride, ::stride]
    sx = xs[::stride, ::stride]
    sy = ys[::stride, ::stride]
    sz = zs[::stride, ::stride]

    # Pixel coordinates of the subsampled grid (small index arrays only).
    vs_sub, us_sub = np.nonzero(sv)
    # Stride-k subsample of a 0-based grid starts at 0: full-res coords are k*i.
    us = (us_sub * stride).astype(np.int64)
    vs = (vs_sub * stride).astype(np.int64)

    # Single copy of the valid subset, nothing else.
    points = np.stack([sx[sv], sy[sv], sz[sv]], axis=-1).astype(np.float32)
    assert points.shape == (int(sv.sum()), 3), points.shape
    return {"points": points, "us": us, "vs": vs,
            "count": int(points.shape[0]),
            "nbytes": int(points.nbytes + us.nbytes + vs.nbytes),
            "points_nbytes": int(points.nbytes),
            "stride": int(stride),
            "full_shape": (int(h), int(w))}


def demo() -> None:
    """Self-check: correspondence, stride, physical mask separation."""
    sys.path.insert(0, str(REPO / "stage_c_deploy" / "metric_depth"))
    from metric_depth import reproject_xyz
    from src.geometry.stereo import StereoCalibration

    calib = StereoCalibration(focal_px=718.3351, baseline_m=0.5301404873575021,
                              cx=600.3891, cy=181.5122, width=1238, height=374,
                              source="synthetic")
    rng = np.random.default_rng(0)
    disp = rng.uniform(2.0, 60.0, size=(24, 32))
    disp[0, 0] = 0.0  # one C2-invalid pixel
    disp[1, 1] = 0.03  # tiny but positive -> huge, physically unreliable depth
    depth, valid = calib.depth_from_disparity(disp)
    depth[~valid] = np.nan
    xs, ys, zs = reproject_xyz(calib, depth)

    # full-res: every valid pixel becomes exactly one point
    rec = depth_to_pointcloud(xs, ys, zs, valid, stride=1)
    assert rec["count"] == int(valid.sum()) == 24 * 32 - 1, rec["count"]
    assert rec["points"].dtype == np.float32
    assert rec["nbytes"] == rec["points_nbytes"] + rec["us"].nbytes + rec["vs"].nbytes
    # point i maps back to its (u, v) and reproduces X, Y, Z exactly
    for i in (0, 7, rec["count"] - 1):
        u, v = int(rec["us"][i]), int(rec["vs"][i])
        assert valid[v, u], (i, u, v)
        assert rec["points"][i, 0] == np.float32(xs[v, u]), (i, u, v)
        assert rec["points"][i, 1] == np.float32(ys[v, u]), (i, u, v)
        assert rec["points"][i, 2] == np.float32(zs[v, u]), (i, u, v)

    # stride=2 is a strict subsample of the stride=1 set
    rec2 = depth_to_pointcloud(xs, ys, zs, valid, stride=2)
    assert rec2["count"] == int(valid[::2, ::2].sum()), rec2["count"]
    assert rec2["count"] <= rec["count"]
    assert set(map(tuple, np.stack([rec2["us"], rec2["vs"]], -1).tolist())) <= set(
        map(tuple, np.stack([rec["us"], rec["vs"]], -1).tolist()))

    # physical mask is SEPARATE: stored depth untouched, C2 mask untouched
    before = depth.copy()
    pm = physical_mask(depth, valid, 60.0)
    assert pm.dtype == bool and pm.shape == valid.shape
    assert (pm <= valid).all(), "physical mask must be a subset of the C2 mask"
    assert not pm[1, 1], "the ~12000 m pixel must be excluded by a 60 m cap"
    assert valid[1, 1], "but it stays C2-valid"
    assert np.array_equal(depth, before, equal_nan=True), "stored depth modified!"

    print(f"pointcloud self-check passed: count={rec['count']} "
          f"nbytes={rec['nbytes']} stride2_count={rec2['count']}")


if __name__ == "__main__":
    demo()
