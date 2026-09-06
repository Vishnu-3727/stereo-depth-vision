"""EXP: metric depth accuracy of the reference model, binned by range.

The project's target is stereo *depth*, not disparity, so the disparity metrics
in EXP-005 are only half the picture. This converts both prediction and ground
truth to metres using each scene's own KITTI calibration and measures depth
accuracy in bands, because a single average hides the thing that matters most:
the same disparity error is worth centimetres up close and tens of metres far
away (EXP-003).

Invalid pixels are dropped, never clamped. A disparity at or below zero has no
depth, and turning it into a large finite number would quietly manufacture data.

    python scripts/exp_depth_validation.py [--scenes N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.experiment import Experiment  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bin_by_depth,
    depth_metrics,
    disparity_metrics,
)
from src.geometry.stereo import parse_kitti_cam_to_cam  # noqa: E402

MODEL = REPO_ROOT / "reference" / "onnx" / "stereonet.onnx"
OUT_DIR = REPO_ROOT / "results" / "depth_validation"

DEPTH_BANDS = [0.0, 10.0, 20.0, 30.0, 50.0, 80.0, 1000.0]
DISPARITY_BANDS = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 1000.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=40)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    ds = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    n = min(args.scenes, len(ds))

    config = {
        "dataset": "kitti2015",
        "split": "hailo_val",
        "resolution": [368, 1232],
        "crop": "Hailo pad-and-crop",
        "disparity_range": "0-176 px nominal; no search performed (EXP-010)",
        "batch_size": 1,
        "precision": "fp32",
        "seed": None,
        "scenes": n,
        "ground_truth": "disp_occ_0, KITTI official 1/256 scale",
        "calibration": "per scene, from calib_cam_to_cam",
    }

    with Experiment(
        "Metric depth accuracy of the reference model on KITTI 2015, binned by "
        "range",
        config=config,
    ) as exp:
        pred_d, gt_d, pred_z, gt_z, fb_values = [], [], [], [], []

        for i in range(n):
            s = ds[i]
            calib = parse_kitti_cam_to_cam(ds.calibration_path(s.name))
            fb_values.append(calib.fB)

            out = session.run(
                [output_name],
                {
                    input_names[0]: normalize(s.left),
                    input_names[1]: normalize(s.right),
                },
            )[0][0, 0].astype(np.float64)

            valid = s.disparity > 0
            p = out[valid]
            g = s.disparity[valid].astype(np.float64)

            # Depth for both, with the same validity rule applied to each.
            zp, ok_p = calib.depth_from_disparity(p)
            zg, ok_g = calib.depth_from_disparity(g)
            ok = ok_p & ok_g

            pred_d.append(p)
            gt_d.append(g)
            pred_z.append(np.where(ok, zp, np.nan))
            gt_z.append(np.where(ok, zg, np.nan))
            if (i + 1) % 10 == 0:
                exp.log("scene {}/{}".format(i + 1, n))

        pred_d = np.concatenate(pred_d)
        gt_d = np.concatenate(gt_d)
        pred_z = np.concatenate(pred_z)
        gt_z = np.concatenate(gt_z)
        finite = np.isfinite(pred_z) & np.isfinite(gt_z)

        exp.metric("valid_disparity_pixels", int(gt_d.size))
        exp.metric("pixels_with_defined_depth", int(finite.sum()))
        exp.metric(
            "pixels_dropped_undefined_depth",
            int(gt_d.size - finite.sum()),
        )
        exp.metric("fB_min_max", [float(min(fb_values)), float(max(fb_values))])

        dm = disparity_metrics(pred_d, gt_d)
        zm = depth_metrics(pred_z[finite], gt_z[finite])
        exp.metric("disparity_metrics_all", dm.as_dict())
        exp.metric("depth_metrics_all", zm.as_dict())

        # -- binned by true depth -------------------------------------------
        abs_disp_err = np.abs(pred_d - gt_d)
        abs_depth_err = np.abs(pred_z - gt_z)
        rel_depth_err = abs_depth_err / np.where(gt_z > 0, gt_z, np.nan)

        by_depth = {}
        for lo, hi in zip(DEPTH_BANDS[:-1], DEPTH_BANDS[1:]):
            band = finite & (gt_z >= lo) & (gt_z < hi)
            label = "{:g}-{:g}m".format(lo, hi)
            if band.sum() < 100:
                by_depth[label] = {"pixels": int(band.sum())}
                continue
            band_zm = depth_metrics(pred_z[band], gt_z[band])
            by_depth[label] = {
                "pixels": int(band.sum()),
                "share_of_pixels": float(band.sum() / finite.sum()),
                "disparity_epe_px": float(abs_disp_err[band].mean()),
                "depth_mae_m": float(abs_depth_err[band].mean()),
                "depth_median_ae_m": float(np.median(abs_depth_err[band])),
                "depth_rel_error_pct": float(100.0 * np.nanmean(rel_depth_err[band])),
                "abs_rel": band_zm.abs_rel,
                "rmse_m": band_zm.rmse,
                "delta1_pct": band_zm.delta1,
                "mean_true_disparity_px": float(gt_d[band].mean()),
            }
        exp.metric("by_true_depth", by_depth)

        # -- binned by true disparity ---------------------------------------
        by_disp = bin_by_depth(abs_disp_err, gt_d, DISPARITY_BANDS, np.ones_like(gt_d, bool))
        exp.metric("disparity_error_by_true_disparity", by_disp)

        (OUT_DIR / "depth_validation.json").write_text(
            json.dumps(
                {
                    "disparity_metrics_all": dm.as_dict(),
                    "depth_metrics_all": zm.as_dict(),
                    "by_true_depth": by_depth,
                    "disparity_error_by_true_disparity": by_disp,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # -- readable table --------------------------------------------------
        lines = [
            "Depth accuracy by range, KITTI 2015 Hailo validation split, "
            "{} scenes".format(n),
            "",
            "{:>10} {:>10} {:>7} {:>12} {:>12} {:>12} {:>10} {:>9}".format(
                "band", "pixels", "share", "disp EPE px", "depth MAE m",
                "median AE m", "rel err %", "delta1 %"
            ),
        ]
        for label, v in by_depth.items():
            if "depth_mae_m" not in v:
                lines.append("{:>10} {:>10,} (too few pixels)".format(label, v["pixels"]))
                continue
            lines.append(
                "{:>10} {:>10,} {:>6.1f}% {:>12.3f} {:>12.3f} {:>12.3f} "
                "{:>10.1f} {:>9.1f}".format(
                    label, v["pixels"], 100 * v["share_of_pixels"],
                    v["disparity_epe_px"], v["depth_mae_m"],
                    v["depth_median_ae_m"], v["depth_rel_error_pct"],
                    v["delta1_pct"],
                )
            )
        text = "\n".join(lines)
        (OUT_DIR / "by_depth.txt").write_text(text, encoding="utf-8")

        near = by_depth.get("0-10m", {})
        far = by_depth.get("50-80m", {})
        if "depth_mae_m" in near and "depth_mae_m" in far:
            exp.note(
                "Disparity error is roughly flat with range ({:.2f} px at "
                "0-10 m against {:.2f} px at 50-80 m) but depth error is not: "
                "{:.2f} m against {:.2f} m, a factor of {:.0f}. That is the "
                "Z^2/(fB) amplification of EXP-003 showing up in a real "
                "model's output.".format(
                    near["disparity_epe_px"], far["disparity_epe_px"],
                    near["depth_mae_m"], far["depth_mae_m"],
                    far["depth_mae_m"] / near["depth_mae_m"],
                )
            )
        exp.note(
            "Overall: disparity EPE {:.3f} px, D1 {:.2f} %; depth Abs Rel "
            "{:.4f}, RMSE {:.3f} m, delta1 {:.1f} %.".format(
                dm.epe, dm.d1, zm.abs_rel, zm.rmse, zm.delta1
            )
        )
        exp.note(
            "{:,} of {:,} valid ground-truth pixels were dropped because "
            "prediction or truth gave an undefined depth. They are excluded, "
            "not clamped.".format(int(gt_d.size - finite.sum()), int(gt_d.size))
        )
        exp.conclude(
            "Depth accuracy measured and binned; the range dependence is "
            "recorded above and written up in docs/failure_analysis.md."
        )
        print("\n" + text + "\n\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
