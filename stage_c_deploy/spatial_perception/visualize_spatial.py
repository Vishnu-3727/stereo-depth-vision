"""Stage C2.1: NEW spatial visualization mode (does not touch the C2 visualizer).

Panels: left RGB, disparity (fixed 0-176 px -- the C2 validated range, kept),
metric depth (fixed 0.5-60 m -- kept), plus a spatial panel showing the cell
grid with per-cell nearest valid surface, and a text block with the selected
region's median depth / nearest valid depth / XYZ.

No smoothing is applied to the data for display. Invalid pixels are shown
transparent / distinct, as in C2.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

OUT_DIR = REPO / "stage_c_deploy" / "spatial_perception" / "out"

# Display ranges ONLY (data itself is never clipped): kept from C2.
DISPLAY_DISP_VMIN_PX = 0.0
DISPLAY_DISP_VMAX_PX = 176.0
DISPLAY_VMIN_M = 0.5
DISPLAY_VMAX_M = 60.0


def save_spatial_visualization(scene_name: str, left_rgb: np.ndarray,
                               disparity: np.ndarray, depth: np.ndarray,
                               valid: np.ndarray, cells: list[dict],
                               selected_name: str,
                               region_text: str,
                               out_dir: Path = OUT_DIR) -> Path:
    """Render the 4-panel spatial figure and write the PNG. Returns the path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(scene_name).stem
    masked_depth = np.ma.masked_where(~np.asarray(valid, dtype=bool),
                                      np.asarray(depth, dtype=np.float64))

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"C2.1 spatial perception -- {scene_name} "
                 f"(depth display {DISPLAY_VMIN_M}-{DISPLAY_VMAX_M} m; "
                 f"data unclipped, no smoothing)")
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(left_rgb)
    ax1.set_title("left RGB (cropped 368x1232)")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    im1 = ax2.imshow(disparity, cmap="plasma",
                     vmin=DISPLAY_DISP_VMIN_PX, vmax=DISPLAY_DISP_VMAX_PX)
    ax2.set_title(f"raw disparity (px), display "
                  f"{DISPLAY_DISP_VMIN_PX:.0f}-{DISPLAY_DISP_VMAX_PX:.0f} px")
    ax2.axis("off")
    fig.colorbar(im1, ax=ax2, label="disparity (px)")

    ax3 = fig.add_subplot(2, 2, 3)
    im2 = ax3.imshow(masked_depth, cmap="inferno",
                     vmin=DISPLAY_VMIN_M, vmax=DISPLAY_VMAX_M)
    ax3.set_title(f"metric depth (m), display "
                  f"{DISPLAY_VMIN_M}-{DISPLAY_VMAX_M} m (invalid transparent)")
    ax3.axis("off")
    fig.colorbar(im2, ax=ax3, label="depth (metres)")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(masked_depth, cmap="inferno",
               vmin=DISPLAY_VMIN_M, vmax=DISPLAY_VMAX_M)
    for cell in cells:
        v0, v1, u0, u1 = cell["bounds"]
        edge = "cyan" if cell["name"] == selected_name else "white"
        lw = 2.0 if cell["name"] == selected_name else 1.0
        ax4.add_patch(patches.Rectangle(
            (u0, v0), u1 - u0, v1 - v0, fill=False, edgecolor=edge, linewidth=lw))
        ax4.text((u0 + u1) / 2, (v0 + v1) / 2,
                 f"{cell['name']}\n{cell['nearest_valid_surface_m']:.1f} m",
                 color="white", fontsize=8, ha="center", va="center",
                 bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))
    ax4.set_title("spatial grid: per-cell NEAREST VALID SURFACE (5th pct, m)")
    ax4.axis("off")

    fig.text(0.5, 0.01, region_text, ha="center", va="bottom", fontsize=8,
             family="monospace",
             bbox=dict(facecolor="white", alpha=0.85, pad=4, edgecolor="gray"))
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = out_dir / f"{stem}_c2_1_spatial.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def demo() -> None:
    """Self-check: renders a synthetic figure, then cleans it up."""
    sys.path.insert(0, str(REPO / "stage_c_deploy" / "spatial_perception"))
    from spatial import spatial_cells
    rng = np.random.default_rng(2)
    left = (rng.uniform(0, 255, size=(48, 64, 3))).astype(np.uint8)
    disp = rng.uniform(2.0, 40.0, size=(48, 64))
    depth = 380.0 / disp  # synthetic fB; data unclipped
    valid = disp > 1e-3
    cells = spatial_cells(depth, valid)
    out_tmp = OUT_DIR / "demo_tmp"
    p = save_spatial_visualization(
        "demo_10.png", left, disp, depth, valid, cells, "CENTER",
        "SELECTED REGION CENTER: median=9.99 m nearest=9.50 m X=0.00 Y=0.00",
        out_dir=out_tmp)
    assert p.exists(), p
    p.unlink()
    out_tmp.rmdir()
    print(f"visualize_spatial self-check passed: {p}")


if __name__ == "__main__":
    demo()
