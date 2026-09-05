"""EXP: how disparity error turns into depth error, on real KITTI calibration.

Surveys the calibration of every KITTI 2015 training scene, then quantifies
what a disparity error is worth in metres across the depth range. The point is
to replace the usual hand-wave ("depth error grows with distance") with numbers
taken from the calibration the benchmark is actually defined on.

Writes a figure and a table to results/depth_error/ and records the run as an
experiment.

    python scripts/depth_error_curve.py
"""

from __future__ import annotations

import json
import statistics
import sys
from glob import glob
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.experiment import Experiment  # noqa: E402
from src.geometry.stereo import StereoCalibration, parse_kitti_cam_to_cam  # noqa: E402

CALIB_GLOB = str(REPO_ROOT / "data/kitti2015/training/calib_cam_to_cam/*.txt")
OUT_DIR = REPO_ROOT / "results" / "depth_error"

# The depths the charter asks to be quantified explicitly.
PROBE_DEPTHS = [5.0, 10.0, 20.0, 50.0, 80.0]
# Disparity errors worth separating: subpixel, the D1 threshold, and the
# published float EPE of the Hailo model.
PROBE_ERRORS = [0.25, 1.0, 3.0, 8.223]


def median_calibration(calibs: list[StereoCalibration]) -> StereoCalibration:
    return StereoCalibration(
        focal_px=statistics.median(c.focal_px for c in calibs),
        baseline_m=statistics.median(c.baseline_m for c in calibs),
        cx=statistics.median(c.cx for c in calibs),
        cy=statistics.median(c.cy for c in calibs),
        width=calibs[0].width,
        height=calibs[0].height,
        source="median over {} KITTI 2015 training scenes".format(len(calibs)),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob(CALIB_GLOB))
    if not files:
        raise SystemExit(
            "No KITTI calibration found at " + CALIB_GLOB + ". Download "
            "data_scene_flow_calib.zip into data/kitti2015/ first."
        )

    calibs = [parse_kitti_cam_to_cam(f) for f in files]
    calib = median_calibration(calibs)

    config = {
        "dataset": "kitti2015",
        "split": "training (calibration only)",
        "resolution": [calib.height, calib.width],
        "crop": None,
        "disparity_range": None,
        "batch_size": None,
        "precision": "fp64 (numpy)",
        "seed": None,
        "calibration_files": len(files),
        "focal_px_median": calib.focal_px,
        "baseline_m_median": calib.baseline_m,
    }

    with Experiment(
        "Disparity-to-depth error propagation on real KITTI 2015 calibration",
        config=config,
    ) as exp:
        spread = {
            "focal_px": [min(c.focal_px for c in calibs), calib.focal_px,
                         max(c.focal_px for c in calibs)],
            "baseline_m": [min(c.baseline_m for c in calibs), calib.baseline_m,
                           max(c.baseline_m for c in calibs)],
            "fB_pixel_metres": [min(c.fB for c in calibs), calib.fB,
                                max(c.fB for c in calibs)],
        }
        exp.metric("calibration_spread_min_median_max", spread)
        exp.metric("fB_pixel_metres", calib.fB)
        exp.metric(
            "range_at_one_pixel_disparity_m", calib.max_range_for_disparity(1.0)
        )
        exp.metric(
            "disparity_at_probe_depths_px",
            {str(z): calib.fB / z for z in PROBE_DEPTHS},
        )

        # -- the table ------------------------------------------------------
        table = []
        for err in PROBE_ERRORS:
            for z in PROBE_DEPTHS:
                d = calib.fB / z
                lin = float(calib.depth_error_from_disparity_error(z, err))
                far, near = calib.depth_error_exact(z, err)
                table.append(
                    {
                        "disparity_error_px": err,
                        "depth_m": z,
                        "true_disparity_px": d,
                        "linear_depth_error_m": lin,
                        "linear_depth_error_pct": 100.0 * lin / z,
                        "depth_if_disparity_underestimated_m": (
                            float(far) if np.isfinite(far) else None
                        ),
                        "depth_if_disparity_overestimated_m": float(near),
                    }
                )
        (OUT_DIR / "table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
        # None in the table means the disparity underestimate drives disparity to
        # zero or below: depth is undefined there, not large.
        exp.metric("error_table", table)

        one_px = {r["depth_m"]: r["linear_depth_error_m"]
                  for r in table if r["disparity_error_px"] == 1.0}
        exp.metric("one_pixel_error_metres_by_depth", one_px)

        # -- figure ---------------------------------------------------------
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            z = np.linspace(2.0, 100.0, 500)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

            for err in PROBE_ERRORS:
                axes[0].plot(
                    z, calib.depth_error_from_disparity_error(z, err),
                    label="{} px disparity error".format(err),
                )
            axes[0].set_xlabel("true depth Z (m)")
            axes[0].set_ylabel("depth error |dZ| (m)")
            axes[0].set_title(
                "Depth error from disparity error\nKITTI 2015, fB = {:.1f} px m".format(
                    calib.fB
                )
            )
            axes[0].set_yscale("log")
            axes[0].grid(alpha=0.3)
            axes[0].legend(fontsize=8)

            d = np.linspace(1.0, 200.0, 500)
            axes[1].plot(d, calib.fB / d, color="k")
            for zp in PROBE_DEPTHS:
                axes[1].axhline(zp, color="0.8", lw=0.6)
                axes[1].plot([calib.fB / zp], [zp], "o", ms=4)
                axes[1].annotate(
                    "{:.0f} m at {:.1f} px".format(zp, calib.fB / zp),
                    (calib.fB / zp, zp), fontsize=7,
                    xytext=(6, 4), textcoords="offset points",
                )
            axes[1].set_xlabel("disparity d (px)")
            axes[1].set_ylabel("depth Z (m)")
            axes[1].set_title("Z = fB / d is strongly non-linear at small disparity")
            axes[1].set_ylim(0, 120)
            axes[1].grid(alpha=0.3)

            fig.tight_layout()
            out = OUT_DIR / "depth_error_curve.png"
            fig.savefig(out, dpi=140)
            plt.close(fig)
            exp.note("figure written to " + str(out.relative_to(REPO_ROOT)))
        except Exception as e:  # a missing backend must not lose the numbers
            exp.note("figure not produced: " + repr(e))

        exp.note(
            "KITTI 2015 calibration is close to constant across scenes: focal "
            "length {:.1f}-{:.1f} px and baseline {:.4f}-{:.4f} m, giving "
            "fB = {:.1f} px m.".format(
                spread["focal_px"][0], spread["focal_px"][2],
                spread["baseline_m"][0], spread["baseline_m"][2], calib.fB,
            )
        )
        exp.note(
            "One pixel of disparity error costs "
            + ", ".join(
                "{:.2f} m at {:.0f} m".format(one_px[z], z) for z in PROBE_DEPTHS
            )
            + ". The 80 m figure is {:.0f}x the 5 m figure for the same one-pixel "
            "error, purely from geometry.".format(one_px[80.0] / one_px[5.0])
        )
        exp.note(
            "At 80 m the true disparity is only {:.2f} px, so the entire range "
            "beyond {:.0f} m is encoded in less than one pixel of disparity. No "
            "amount of network capacity recovers depth there without subpixel "
            "precision.".format(
                calib.fB / 80.0, calib.max_range_for_disparity(1.0)
            )
        )

        epe_row = [r for r in table if r["disparity_error_px"] == 8.223]
        undefined_beyond = calib.fB / 8.223
        exp.note(
            "For scale: Hailo publishes a float EPE of 8.223 px for this model. "
            "Applied uniformly as an underestimate, a true 20 m point would land "
            "at {:.1f} m, and beyond {:.1f} m the disparity would be driven to "
            "zero or below, so depth becomes undefined rather than merely large "
            "-- that is what the inf entries in the table mean. Average EPE is "
            "not uniform across the image, so this is an illustration of what "
            "the figure is worth in metres, not a prediction of the model's "
            "depth error.".format(
                next(r["depth_if_disparity_underestimated_m"]
                     for r in epe_row if r["depth_m"] == 20.0),
                undefined_beyond,
            )
        )
        exp.metric("depth_beyond_which_epe_8223_is_undefined_m", undefined_beyond)
        exp.conclude(
            "Depth error scales as Z^2/(fB) for a fixed disparity error, "
            "confirmed against a numerical derivative in the geometry unit test "
            "and quantified on the real KITTI calibration. Any disparity metric "
            "reported without a depth range attached is close to meaningless for "
            "this project."
        )

        # readable table
        lines = ["Depth error from disparity error, KITTI 2015 median calibration",
                 "f = {:.4f} px   B = {:.4f} m   fB = {:.4f} px m".format(
                     calib.focal_px, calib.baseline_m, calib.fB),
                 ""]
        lines.append("{:>10} {:>9} {:>10} {:>12} {:>9} {:>14} {:>14}".format(
            "disp err", "depth", "true disp", "depth err", "as %",
            "Z if d under", "Z if d over"))
        for r in table:
            far = r["depth_if_disparity_underestimated_m"]
            far_txt = "undefined" if far is None else "{:.1f}m".format(far)
            lines.append(
                "{:>9.3f}p {:>8.1f}m {:>9.2f}p {:>11.3f}m {:>8.1f}% "
                "{:>14} {:>13.1f}m".format(
                    r["disparity_error_px"], r["depth_m"], r["true_disparity_px"],
                    r["linear_depth_error_m"], r["linear_depth_error_pct"],
                    far_txt, r["depth_if_disparity_overestimated_m"],
                )
            )
        text = "\n".join(lines)
        (OUT_DIR / "table.txt").write_text(text, encoding="utf-8")
        print("\n" + text + "\n")
        print("recorded as " + exp.id)


if __name__ == "__main__":
    main()
