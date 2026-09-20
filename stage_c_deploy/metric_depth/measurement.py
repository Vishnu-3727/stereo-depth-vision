"""Stage C2: pixel and NxN region measurement on (disparity, depth, XYZ)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from src.geometry.stereo import StereoCalibration


def pixel_measure(calib: StereoCalibration, disparity: np.ndarray,
                  depth: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                  u: int, v: int, fy: float | None = None) -> dict:
    """Measure a single pixel. Returns PIXEL values (no aggregation)."""
    if fy is None:
        fy = calib.focal_px
    h, w = disparity.shape
    assert 0 <= u < w and 0 <= v < h, f"pixel ({u},{v}) outside {w}x{h}"
    d = float(disparity[v, u])
    z = float(depth[v, u])
    if np.isfinite(z):
        x = (u - calib.cx) * z / calib.focal_px
        y = (v - calib.cy) * z / fy
    else:
        x, y = float("nan"), float("nan")
    return {"u": u, "v": v, "disparity_px": d,
            "pixel_depth_m": z, "X_m": float(x), "Y_m": float(y),
            "valid": bool(np.isfinite(z))}


def region_measure(calib: StereoCalibration, disparity: np.ndarray,
                   depth: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                   u: int, v: int, size: int = 5,
                   fy: float | None = None) -> dict:
    """Median over VALID pixels only in a size x size window centred on (u,v).

    Returns REGION MEDIAN values, labelled separately from pixel values.
    If no valid pixel exists in the window, medians are NaN and valid_count is 0.
    """
    if fy is None:
        fy = calib.focal_px
    assert size % 2 == 1 and size >= 1, "size must be odd and >= 1"
    h, w = disparity.shape
    r = size // 2
    u0, u1 = max(u - r, 0), min(u + r + 1, w)
    v0, v1 = max(v - r, 0), min(v + r + 1, h)
    zwin = depth[v0:v1, u0:u1]
    xwin = xs[v0:v1, u0:u1]
    ywin = ys[v0:v1, u0:u1]
    m = np.isfinite(zwin)
    n = int(m.sum())
    if n == 0:
        return {"u": u, "v": v, "window": size,
                "region_median_disparity_px": float("nan"),
                "region_median_depth_m": float("nan"),
                "region_median_X_m": float("nan"),
                "region_median_Y_m": float("nan"),
                "valid_count": 0,
                "window_pixels": int(zwin.size)}
    return {"u": u, "v": v, "window": size,
            "region_median_disparity_px": float(np.median(disparity[v0:v1, u0:u1][m])),
            "region_median_depth_m": float(np.median(zwin[m])),
            "region_median_X_m": float(np.median(xwin[m])),
            "region_median_Y_m": float(np.median(ywin[m])),
            "valid_count": n,
            "window_pixels": int(zwin.size)}


def demo() -> None:
    calib = StereoCalibration(focal_px=100.0, baseline_m=0.5,
                              cx=5.0, cy=5.0, width=11, height=11,
                              source="synthetic")
    disp = np.full((11, 11), 10.0)
    disp[0, 0] = 0.0  # one invalid pixel
    depth, _ = calib.depth_from_disparity(disp)
    xs = (np.arange(11)[None, :].repeat(11, 0) - calib.cx) * depth / calib.focal_px
    ys = (np.arange(11)[:, None].repeat(11, 1) - calib.cy) * depth / calib.focal_px
    p = pixel_measure(calib, disp, depth, xs, ys, 5, 5)
    assert abs(p["pixel_depth_m"] - 5.0) < 1e-12, p
    assert abs(p["X_m"]) < 1e-12 and abs(p["Y_m"]) < 1e-12, p
    q = pixel_measure(calib, disp, depth, xs, ys, 0, 0)
    assert q["valid"] is False and np.isnan(q["pixel_depth_m"])
    r = region_measure(calib, disp, depth, xs, ys, 1, 1, size=3)
    # window has 8 valid + 1 invalid -> median over valid only == 5.0 m
    assert r["valid_count"] == 8, r
    assert abs(r["region_median_depth_m"] - 5.0) < 1e-12, r
    print("measurement self-check passed")


if __name__ == "__main__":
    demo()
