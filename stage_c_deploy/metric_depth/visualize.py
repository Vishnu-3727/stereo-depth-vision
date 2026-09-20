"""Stage C2: NEW visualization mode (does not touch existing diagnostics).

Panels: left RGB, disparity map, metric depth map (colourbar in metres;
display range stated and separate from the data), RGB+depth overlay.
Invalid pixels visually distinct (masked transparent / magenta).
Saves PNGs under stage_c_deploy/metric_depth/out/.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

OUT_DIR = REPO / "stage_c_deploy" / "metric_depth" / "out"

# Display range ONLY (data itself is never clipped): 0.5 m to 60 m.
DISPLAY_VMIN_M = 0.5
DISPLAY_VMAX_M = 60.0

# Disparity display range ONLY (data itself is never clipped): 0-176 px.
# Measured disparity range over all 40 hailo_val scenes is min 0.0 px,
# max 175.10543823242188 px (source: out/c2_validation.json, fields
# disp_min/disp_max), so 0-176 px covers the whole split with no clipping
# of real data.
DISPLAY_DISP_VMIN_PX = 0.0
DISPLAY_DISP_VMAX_PX = 176.0


def save_visualization(scene_name: str, left_rgb: np.ndarray,
                       disparity: np.ndarray, depth: np.ndarray,
                       valid: np.ndarray,
                       out_dir: Path = OUT_DIR) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(scene_name).stem

    masked_depth = np.ma.masked_where(~valid, depth)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"C2 metric depth — {scene_name} "
                 f"(display {DISPLAY_VMIN_M}-{DISPLAY_VMAX_M} m; data unclipped)")
    axes[0, 0].imshow(left_rgb)
    axes[0, 0].set_title("left RGB (cropped 368x1232)")
    axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(disparity, cmap="plasma",
                             vmin=DISPLAY_DISP_VMIN_PX, vmax=DISPLAY_DISP_VMAX_PX)
    axes[0, 1].set_title(
        f"raw disparity (px), display {DISPLAY_DISP_VMIN_PX:.0f}-{DISPLAY_DISP_VMAX_PX:.0f} px")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], label="disparity (px)")

    im2 = axes[1, 0].imshow(masked_depth, cmap="inferno",
                            vmin=DISPLAY_VMIN_M, vmax=DISPLAY_VMAX_M)
    axes[1, 0].set_title(
        f"metric depth (m), display {DISPLAY_VMIN_M}-{DISPLAY_VMAX_M} m")
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], label="depth (metres)")

    axes[1, 1].imshow(left_rgb)
    ov = axes[1, 1].imshow(masked_depth, cmap="inferno", alpha=0.55,
                           vmin=DISPLAY_VMIN_M, vmax=DISPLAY_VMAX_M)
    axes[1, 1].set_title("RGB + depth overlay (invalid transparent)")
    axes[1, 1].axis("off")
    fig.colorbar(ov, ax=axes[1, 1], label="depth (metres)")

    fig.tight_layout()
    path = out_dir / f"{stem}_c2_viz.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def demo() -> None:
    from src.geometry.stereo import StereoCalibration
    calib = StereoCalibration(focal_px=718.3351, baseline_m=0.5301404873575021,
                              cx=600.3891, cy=181.5122, width=1238, height=374,
                              source="synthetic")
    rng = np.random.default_rng(1)
    left = (rng.uniform(0, 255, size=(32, 40, 3))).astype(np.uint8)
    disp = rng.uniform(0, 40, size=(32, 40))
    depth, m = calib.depth_from_disparity(disp)
    out_tmp = OUT_DIR / "demo_tmp"
    p = save_visualization("demo_10.png", left, disp, depth, m,
                           out_dir=out_tmp)
    assert p.exists()
    p.unlink()
    out_tmp.rmdir()
    print(f"visualize self-check passed: {p}")


if __name__ == "__main__":
    demo()
