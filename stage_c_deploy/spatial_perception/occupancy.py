"""Stage C2.1: DEPTH OCCUPANCY / NEAR-SURFACE MAP only.

A coarse cell grid classified by a documented, caller-supplied depth threshold
into NEAR / FAR / INSUFFICIENT_DATA. A cell with too few valid pixels is
INSUFFICIENT_DATA -- it is NOT free.

There is no ground plane, no camera-pose assumption, and no free-space claim
in this module. NEAR means "the cell's valid depth samples are predominantly
closer than the caller's threshold" -- it is NOT an obstacle: a near surface
can be road, a wall, vegetation, or noise, and this module cannot tell which
because no detector exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

NEAR = "NEAR"
FAR = "FAR"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


def occupancy_grid(depth: np.ndarray, valid: np.ndarray,
                   n_rows: int = 4, n_cols: int = 6,
                   near_threshold_m: float = 10.0,
                   min_valid_fraction: float = 0.1) -> list[dict]:
    """Classify each coarse cell. ``near_threshold_m`` is caller-supplied and
    recorded in every record; a cell is NEAR when its valid-median depth is at
    or below the threshold, FAR when above it, INSUFFICIENT_DATA when its
    valid fraction is below ``min_valid_fraction``.
    """
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    assert depth.shape == valid.shape
    assert near_threshold_m > 0, near_threshold_m
    assert 0.0 <= min_valid_fraction <= 1.0, min_valid_fraction
    h, w = depth.shape
    row_edges = np.linspace(0, h, n_rows + 1).astype(int)
    col_edges = np.linspace(0, w, n_cols + 1).astype(int)
    out = []
    for r in range(n_rows):
        for c in range(n_cols):
            v0, v1 = int(row_edges[r]), int(row_edges[r + 1])
            u0, u1 = int(col_edges[c]), int(col_edges[c + 1])
            cell_valid = valid[v0:v1, u0:u1]
            frac = float(cell_valid.mean())
            vals = depth[v0:v1, u0:u1][cell_valid]
            med = float(np.median(vals)) if vals.size else float("nan")
            if frac < min_valid_fraction or vals.size == 0:
                label = INSUFFICIENT_DATA
            elif med <= near_threshold_m:
                label = NEAR
            else:
                label = FAR
            out.append({"row": r, "col": c, "bounds": (v0, v1, u0, u1),
                        "label": label, "median_m": med,
                        "valid_fraction": frac,
                        "near_threshold_m": float(near_threshold_m),
                        "min_valid_fraction": float(min_valid_fraction)})
    return out


def demo() -> None:
    """Self-check: obvious near/far halves, sparse cell, threshold recorded."""
    depth = np.full((20, 40), 30.0)
    depth[:, :20] = 5.0  # left half near
    grid = occupancy_grid(depth, np.ones_like(depth, bool),
                          n_rows=1, n_cols=2, near_threshold_m=10.0)
    assert [g["label"] for g in grid] == [NEAR, FAR], grid
    assert all(g["near_threshold_m"] == 10.0 for g in grid)

    # a cell with almost no valid pixels is INSUFFICIENT_DATA, not FAR/free
    valid = np.zeros_like(depth, dtype=bool)
    valid[0, 0] = True
    grid2 = occupancy_grid(depth, valid, n_rows=1, n_cols=2,
                           near_threshold_m=10.0, min_valid_fraction=0.1)
    assert grid2[0]["label"] == INSUFFICIENT_DATA, grid2[0]
    assert grid2[1]["label"] == INSUFFICIENT_DATA, grid2[1]

    # boundary: median exactly at the threshold counts as NEAR (documented)
    grid3 = occupancy_grid(np.full((4, 4), 10.0), np.ones((4, 4), bool),
                           n_rows=1, n_cols=1, near_threshold_m=10.0)
    assert grid3[0]["label"] == NEAR, grid3[0]

    print("occupancy self-check passed")


if __name__ == "__main__":
    demo()
