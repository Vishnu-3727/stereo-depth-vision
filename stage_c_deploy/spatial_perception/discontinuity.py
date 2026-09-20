"""Stage C2.1: DEPTH DISCONTINUITY MAP (deterministic, CPU-only).

NaN-aware gradient magnitude on depth (|dZ/du|, |dZ/dv| via forward
differences, invalid if either neighbour is invalid) and the same on
disparity. Outputs the magnitude map plus its validity mask.

No smoothing of any kind is applied to the base data. Discontinuities are NOT
labelled as object boundaries -- no detector exists; a large gradient is a
depth step in the data, nothing more.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np


def _forward_mag(field: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|dF/du|, |dF/dv| by forward differences; magnitude where BOTH hold."""
    field = np.asarray(field, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    assert field.shape == valid.shape
    h, w = field.shape
    mag = np.full(field.shape, np.nan, dtype=np.float64)
    mag_valid = np.zeros(field.shape, dtype=bool)
    if h < 2 or w < 2:
        return mag, mag_valid
    du_ok = valid[:, :-1] & valid[:, 1:]
    dv_ok = valid[:-1, :] & valid[1:, :]
    du = np.full(field.shape, np.nan)
    dv = np.full(field.shape, np.nan)
    du[:, :-1][du_ok] = np.abs(field[:, 1:][du_ok] - field[:, :-1][du_ok])
    dv[:-1, :][dv_ok] = np.abs(field[1:, :][dv_ok] - field[:-1, :][dv_ok])
    both = np.zeros(field.shape, dtype=bool)
    both[:-1, :-1] = du_ok[:-1, :] & dv_ok[:, :-1]
    m = both & np.isfinite(du) & np.isfinite(dv)
    mag[m] = np.hypot(du[m], dv[m])
    mag_valid = m
    return mag, mag_valid


def depth_discontinuity(depth: np.ndarray, valid: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Gradient magnitude (m/px) of the metric depth map + validity mask."""
    return _forward_mag(depth, valid)


def disparity_discontinuity(disparity: np.ndarray,
                            valid: np.ndarray | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Same operator on disparity (px/px). Default validity is finite pixels;
    pass the C2 mask to mirror C2's invalid rules (d<=0 excluded)."""
    disparity = np.asarray(disparity, dtype=np.float64)
    if valid is None:
        valid = np.isfinite(disparity)
    return _forward_mag(disparity, valid)


def demo() -> None:
    """Self-check: flat field -> zero; step edge -> exact step; NaN kills."""
    # flat field: zero gradient everywhere it is defined
    f = np.full((6, 8), 10.0)
    mag, mv = _forward_mag(f, np.ones_like(f, bool))
    assert mv[:-1, :-1].all() and not mv[-1, :].any() and not mv[:, -1].any()
    assert np.all(mag[mv] == 0.0)

    # vertical step of exactly 20 m: forward |dZ/du| == 20 at the step column
    g = np.full((6, 10), 10.0)
    g[:, 5:] = 30.0
    mag2, mv2 = _forward_mag(g, np.ones_like(g, bool))
    assert mv2[:-1, :-1].all()
    assert np.all(mag2[:-1, 4] == 20.0), mag2[:-1, 4]
    assert np.all(mag2[:-1, :4] == 0.0) and np.all(mag2[:-1, 5:-1] == 0.0)

    # NaN-aware: an invalid pixel poisons exactly the differences touching it
    h = np.full((4, 5), 7.0)
    vh = np.ones_like(h, dtype=bool)
    vh[1, 2] = False
    h[1, 2] = np.nan
    _, mv3 = _forward_mag(h, vh)
    assert not mv3[1, 2] and not mv3[1, 1] and not mv3[0, 2], mv3.astype(int)
    assert mv3[0, 0] and mv3[2, 3], mv3.astype(int)

    # disparity twin honours a supplied C2-style mask (d<=0 excluded)
    d = np.full((4, 5), 20.0)
    d[0, 0] = 0.0
    m4, mv4 = disparity_discontinuity(d, valid=d > 1e-3)
    assert not mv4[0, 0], mv4.astype(int)

    print("discontinuity self-check passed")


if __name__ == "__main__":
    demo()
