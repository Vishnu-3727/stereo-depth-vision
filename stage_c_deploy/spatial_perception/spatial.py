"""Stage C2.1: spatial binning and nearest valid surface.

Deterministic CPU post-processing on the frozen C2 depth. No detector exists:
terminology is NEAREST VALID SURFACE / SELECTED REGION. Never object, never
obstacle.

Binning: configurable grid, default 1 row x 3 cols (LEFT / CENTER / RIGHT) with
an optional row split (e.g. 2x3 -> TOP-LEFT ... BOTTOM-RIGHT). Cells tile the
image without overlap; border pixels go to the last row/col.

Per cell: valid count, percent valid, min, 5th-percentile, median depth
(computed over VALID pixels only; empty cells report NaN).

The headline "nearest valid surface" of a cell is the 5th-percentile depth
(single-pixel minima are noise). The true per-cell min is reported ALONGSIDE
it, never instead of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np


def cell_name(r: int, c: int, n_rows: int, n_cols: int) -> str:
    row = "" if n_rows == 1 else ("TOP-" if r == 0 else
                                  ("BOTTOM-" if r == n_rows - 1 else f"ROW{r}-"))
    if n_cols == 3 and n_rows == 1:
        col = ("LEFT", "CENTER", "RIGHT")[c]
    elif n_cols == 3:
        col = ("LEFT", "CENTER", "RIGHT")[c]
    else:
        col = f"COL{c}"
    return f"{row}{col}"


def spatial_cells(depth: np.ndarray, valid: np.ndarray,
                  n_rows: int = 1, n_cols: int = 3,
                  physical_valid: np.ndarray | None = None) -> list[dict]:
    """Bin depth into an n_rows x n_cols grid. See module docstring.

    ``physical_valid`` is an OPTIONAL separate mask (see pointcloud.
    physical_mask). When given, statistics are computed over
    ``valid & physical_valid`` and the record notes that the physical-range
    filter was applied; the stored depth is never modified. Default (None)
    uses the C2 validity mask unchanged.
    """
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    assert depth.shape == valid.shape, (depth.shape, valid.shape)
    assert n_rows >= 1 and n_cols >= 1, (n_rows, n_cols)
    h, w = depth.shape
    if physical_valid is not None:
        physical_valid = np.asarray(physical_valid, dtype=bool)
        assert physical_valid.shape == valid.shape
        use = valid & physical_valid
        filt = True
    else:
        use = valid
        filt = False

    row_edges = np.linspace(0, h, n_rows + 1).astype(int)
    col_edges = np.linspace(0, w, n_cols + 1).astype(int)
    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            v0, v1 = int(row_edges[r]), int(row_edges[r + 1])
            u0, u1 = int(col_edges[c]), int(col_edges[c + 1])
            cell_use = use[v0:v1, u0:u1]
            n_pix = int(cell_use.size)
            n_valid = int(cell_use.sum())
            vals = depth[v0:v1, u0:u1][cell_use] if n_pix else np.array([])
            if n_valid:
                mn = float(vals.min())
                p5 = float(np.percentile(vals, 5))
                med = float(np.median(vals))
            else:
                mn = p5 = med = float("nan")
            cells.append({"name": cell_name(r, c, n_rows, n_cols),
                          "row": r, "col": c,
                          "bounds": (v0, v1, u0, u1),
                          "pixels": n_pix, "valid_count": n_valid,
                          "percent_valid": 100.0 * n_valid / n_pix if n_pix else 0.0,
                          "min_m": mn,
                          "nearest_valid_surface_m": p5,
                          "median_m": med,
                          "physical_filter_applied": filt})
    return cells


def demo() -> None:
    """Self-check on synthetic cases with obvious expected answers."""
    # Case 1: three vertical bands, near in the centre. 1x3 must find it.
    depth = np.full((30, 90), 50.0)
    depth[:, 30:60] = 10.0
    valid = np.ones_like(depth, dtype=bool)
    cells = spatial_cells(depth, valid)
    assert [c["name"] for c in cells] == ["LEFT", "CENTER", "RIGHT"]
    got = {c["name"]: (c["nearest_valid_surface_m"], c["min_m"],
                       c["median_m"]) for c in cells}
    assert got["CENTER"] == (10.0, 10.0, 10.0), got
    assert got["LEFT"] == (50.0, 50.0, 50.0), got
    assert got["RIGHT"] == (50.0, 50.0, 50.0), got
    assert all(c["valid_count"] == 30 * 30 for c in cells)

    # Case 2: single-pixel spike must NOT move the 5th percentile, but the
    # true min is still reported alongside.
    depth2 = np.full((20, 20), 40.0)
    depth2[10, 10] = 0.5
    cells2 = spatial_cells(depth2, np.ones_like(depth2, bool))
    assert cells2[1]["min_m"] == 0.5, cells2[1]
    assert cells2[1]["nearest_valid_surface_m"] == 40.0, cells2[1]

    # Case 3: optional row split names + an all-invalid cell reports NaN.
    depth3 = np.full((20, 30), 12.0)
    valid3 = np.zeros_like(depth3, dtype=bool)
    valid3[:10] = True
    cells3 = spatial_cells(depth3, valid3, n_rows=2, n_cols=3)
    assert cells3[0]["name"] == "TOP-LEFT", cells3[0]["name"]
    assert cells3[3]["name"] == "BOTTOM-LEFT", cells3[3]["name"]
    assert np.isnan(cells3[3]["nearest_valid_surface_m"])
    assert cells3[3]["valid_count"] == 0

    # Case 4: physical filter is a separate, reported mask.
    depth4 = np.array([[10.0, 5000.0]])
    cells4 = spatial_cells(depth4, np.ones((1, 2), bool),
                           physical_valid=np.array([[True, False]]))
    assert cells4[1]["physical_filter_applied"] is True
    assert cells4[1]["nearest_valid_surface_m"] == 10.0, cells4[1]

    print("spatial self-check passed")


if __name__ == "__main__":
    demo()
