"""Stage C2: disparity -> metric depth + XYZ reprojection.

REUSES src.geometry.stereo.StereoCalibration.depth_from_disparity for the
depth conversion (no second depth formula exists in this file). XYZ uses
X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy.

Note on fy: StereoCalibration stores a single focal_px = P_rect_02[0,0].
KITTI rectified matrices normally have P[0,0] == P[1,1]; this module reads fy
= P_rect_02[1,1] from the calib file itself (read_fy_from_calib) WITHOUT
modifying src/geometry/stereo.py, and reports whether fx == fy.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from src.geometry.stereo import StereoCalibration


def read_fy_from_calib(path: str | Path) -> float:
    """Read fy = P_rect_02[1,1] from a KITTI calib_cam_to_cam file."""
    for line in Path(path).read_text().splitlines():
        if line.startswith("P_rect_02:"):
            vals = [float(v) for v in line.partition(":")[2].split()]
            p = np.array(vals, dtype=np.float64).reshape(3, 4)
            return float(p[1, 1])
    raise ValueError(f"P_rect_02 not found in {path}")


def disparity_to_depth(calib: StereoCalibration,
                       disparity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth via the IMPORTED StereoCalibration.depth_from_disparity.

    Invalid (d <= min_disparity, NaN, Inf, out-of-range in the sense of
    non-positive) becomes NaN plus a False mask entry. No clipping.
    """
    disp = np.asarray(disparity, dtype=np.float64)
    # Non-finite entries can never be valid; depth_from_disparity already
    # rejects d <= min_disparity, and NaN/Inf comparisons are False there,
    # but be explicit so the contract is obvious.
    finite = np.isfinite(disp)
    depth, valid = calib.depth_from_disparity(disp)
    valid = valid & finite
    depth[~valid] = np.nan
    return depth, valid


def reproject_xyz(calib: StereoCalibration, depth: np.ndarray,
                  fy: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject a depth map to X, Y, Z maps. Invalid depth stays NaN."""
    z = np.asarray(depth, dtype=np.float64)
    if fy is None:
        fy = calib.focal_px
    fx = calib.focal_px
    h, w = z.shape
    us = np.arange(w, dtype=np.float64)[None, :].repeat(h, axis=0)
    vs = np.arange(h, dtype=np.float64)[:, None].repeat(w, axis=1)
    x = (us - calib.cx) * z / fx
    y = (vs - calib.cy) * z / fy
    return x, y, z.copy()


def depth_stats(depth: np.ndarray, valid: np.ndarray) -> dict:
    """Summary stats over valid pixels only. No clipping of the data."""
    d = {"num_pixels": int(depth.size),
         "num_valid": int(np.count_nonzero(valid)),
         "num_invalid": int(depth.size - np.count_nonzero(valid))}
    d["percent_valid"] = 100.0 * d["num_valid"] / d["num_pixels"]
    v = depth[valid] if d["num_valid"] else np.array([], dtype=np.float64)
    if v.size:
        d.update({"min_m": float(v.min()), "max_m": float(v.max()),
                  "mean_m": float(v.mean()), "median_m": float(np.median(v))})
    else:
        d.update({"min_m": float("nan"), "max_m": float("nan"),
                  "mean_m": float("nan"), "median_m": float("nan")})
    return d


def demo() -> None:
    calib = StereoCalibration(focal_px=718.3351, baseline_m=0.5301404873575021,
                              cx=600.3891, cy=181.5122, width=1238, height=374,
                              source="synthetic")
    # invalid handling: d<=0, NaN, Inf -> NaN + invalid
    d = np.array([[10.0, 0.0, -2.0, np.nan, np.inf]])
    z, valid = disparity_to_depth(calib, d)
    assert valid.tolist() == [[True, False, False, False, False]]
    assert np.isnan(z[0, 1:]).all()
    assert abs(z[0, 0] - calib.focal_px * calib.baseline_m / 10.0) < 1e-12
    # XYZ at a hand-checkable pixel
    zt = np.full((3, 4), 10.0)
    x, y, zz = reproject_xyz(calib, zt)
    u, v = 2, 1
    assert abs(x[v, u] - (u - calib.cx) * 10.0 / calib.focal_px) < 1e-12
    assert abs(y[v, u] - (v - calib.cy) * 10.0 / calib.focal_px) < 1e-12
    s = depth_stats(z, valid)
    assert s["num_valid"] == 1 and s["num_invalid"] == 4
    print("metric_depth self-check passed")


if __name__ == "__main__":
    demo()
