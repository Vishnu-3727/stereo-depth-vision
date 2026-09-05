"""EXP: what precision costs, in accuracy, latency and size.

Compares fp32, fp16 and int8 on our own stack, measuring disparity accuracy,
metric depth accuracy, latency and model size for each.

**These are onnxruntime and PyTorch results on our hardware.** They are not
Hailo int8 results and must never be presented as such: Hailo's quantiser,
calibration procedure and integer arithmetic are different, and its published
hardware figure of 10.3 [SOURCE: SR-004] stays SOURCE evidence in its own
column.

The int8 calibration set is Hailo's -- the first 160 KITTI 2015 training scenes,
disjoint from the 40-scene validation split [SOURCE: SR-002] -- so at least the
data split matches even though the quantiser does not.

    python scripts/exp_quantization.py [--scenes N] [--calib N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.experiment import Experiment  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.evaluation.metrics import depth_metrics, disparity_metrics  # noqa: E402
from src.geometry.stereo import parse_kitti_cam_to_cam  # noqa: E402
from src.models.stereonet import StereoNet  # noqa: E402
from src.models.stereonet.onnx_weights import load_onnx_weights  # noqa: E402

MODEL = REPO_ROOT / "reference" / "onnx" / "stereonet.onnx"
OUT_DIR = REPO_ROOT / "results" / "quantization"
WORK = REPO_ROOT / "results" / "quantization" / "artifacts"
H, W = 368, 1232


def evaluate(predict, ds, n, exp, label):
    """Run a prediction callable over the split and score disparity and depth."""
    pred_d, gt_d, pred_z, gt_z, lat = [], [], [], [], []
    for i in range(n):
        s = ds[i]
        calib = parse_kitti_cam_to_cam(ds.calibration_path(s.name))
        t0 = time.perf_counter()
        out = predict(s.left, s.right)
        lat.append(time.perf_counter() - t0)
        valid = s.disparity > 0
        p = out[valid].astype(np.float64)
        g = s.disparity[valid].astype(np.float64)
        zp, ok_p = calib.depth_from_disparity(p)
        zg, ok_g = calib.depth_from_disparity(g)
        ok = ok_p & ok_g
        pred_d.append(p)
        gt_d.append(g)
        pred_z.append(np.where(ok, zp, np.nan))
        gt_z.append(np.where(ok, zg, np.nan))
    pred_d = np.concatenate(pred_d)
    gt_d = np.concatenate(gt_d)
    pred_z = np.concatenate(pred_z)
    gt_z = np.concatenate(gt_z)
    finite = np.isfinite(pred_z) & np.isfinite(gt_z)
    dm = disparity_metrics(pred_d, gt_d)
    zm = depth_metrics(pred_z[finite], gt_z[finite])
    exp.log(
        "{:<26} EPE {:6.3f} px  D1 {:6.3f}%  AbsRel {:.4f}  RMSE {:6.3f} m  "
        "{:7.1f} ms".format(
            label, dm.epe, dm.d1, zm.abs_rel, zm.rmse, 1000 * np.median(lat)
        )
    )
    return {
        "disparity": dm.as_dict(),
        "depth": zm.as_dict(),
        "latency_median_ms": float(1000 * np.median(lat)),
        "predictions": pred_d,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--calib", type=int, default=32)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    ds = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    n = min(args.scenes, len(ds))
    calib_ds = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")

    config = {
        "dataset": "kitti2015",
        "split": "hailo_val",
        "resolution": [H, W],
        "crop": "Hailo pad-and-crop",
        "disparity_range": "12 candidates at 1/16",
        "batch_size": 1,
        "precision": "fp32 / fp16 / int8 compared",
        "seed": 0,
        "scenes": n,
        "int8_calibration": "hailo_calib split, {} scenes".format(args.calib),
    }

    with Experiment(
        "Precision study: fp32, fp16 and int8 accuracy, latency and size on our "
        "own runtime stack",
        config=config,
    ) as exp:
        exp.note(
            "onnxruntime and PyTorch results on our hardware. Not Hailo int8 "
            "results: Hailo uses a different quantiser, a different calibration "
            "procedure and different integer arithmetic. Its published hardware "
            "figure of 10.3 stays SOURCE evidence and is not compared against "
            "these."
        )

        results = {}
        sizes = {"fp32_onnx_bytes": MODEL.stat().st_size}

        # -- fp32 reference, onnxruntime ------------------------------------
        sess32 = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
        names = [i.name for i in sess32.get_inputs()]
        out_name = sess32.get_outputs()[0].name

        def predict_ort(sess):
            def f(left, right):
                return sess.run(
                    [out_name],
                    {names[0]: normalize(left), names[1]: normalize(right)},
                )[0][0, 0]
            return f

        results["fp32_onnxruntime_cpu"] = evaluate(
            predict_ort(sess32), ds, n, exp, "fp32 onnxruntime CPU"
        )

        # -- fp32 and fp16 in PyTorch on GPU ---------------------------------
        if torch.cuda.is_available():
            model = StereoNet().eval()
            load_onnx_weights(model, MODEL)
            model = model.cuda()

            def predict_torch(dtype):
                m = model.half() if dtype == torch.float16 else model.float()

                def f(left, right):
                    with torch.no_grad():
                        l = torch.from_numpy(normalize(left)).cuda().to(dtype)
                        r = torch.from_numpy(normalize(right)).cuda().to(dtype)
                        o = m(l, r)
                        torch.cuda.synchronize()
                    return o[0, 0].float().cpu().numpy()
                return f

            results["fp32_torch_cuda"] = evaluate(
                predict_torch(torch.float32), ds, n, exp, "fp32 torch CUDA"
            )
            results["fp16_torch_cuda"] = evaluate(
                predict_torch(torch.float16), ds, n, exp, "fp16 torch CUDA"
            )
            model = model.float()
        else:
            exp.note("CUDA unavailable; fp16 not measured. UNKNOWN.")

        # -- int8, onnxruntime static quantisation ---------------------------
        int8_path = WORK / "stereonet_int8.onnx"
        try:
            from onnxruntime.quantization import (
                CalibrationDataReader,
                QuantFormat,
                QuantType,
                quantize_static,
            )

            class Reader(CalibrationDataReader):
                def __init__(self, dataset, count):
                    self.items = []
                    for i in range(min(count, len(dataset))):
                        s = dataset[i]
                        self.items.append(
                            {names[0]: normalize(s.left), names[1]: normalize(s.right)}
                        )
                    self.it = iter(self.items)

                def get_next(self):
                    return next(self.it, None)

            exp.log("calibrating int8 on {} scenes".format(args.calib))
            quantize_static(
                str(MODEL),
                str(int8_path),
                Reader(calib_ds, args.calib),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
            sizes["int8_onnx_bytes"] = int8_path.stat().st_size
            sess8 = ort.InferenceSession(
                str(int8_path), providers=["CPUExecutionProvider"]
            )
            results["int8_onnxruntime_cpu"] = evaluate(
                predict_ort(sess8), ds, n, exp, "int8 onnxruntime CPU"
            )
        except Exception as e:
            exp.note("int8 quantisation failed: " + repr(e)[:400])
            exp.metric("int8_status", "FAILED: " + type(e).__name__)

        # -- report ----------------------------------------------------------
        base = results["fp32_onnxruntime_cpu"]
        table = {}
        for key, r in results.items():
            table[key] = {
                "epe_px": r["disparity"]["epe"],
                "d1_percent": r["disparity"]["d1"],
                "depth_abs_rel": r["depth"]["abs_rel"],
                "depth_rmse_m": r["depth"]["rmse"],
                "depth_delta1_percent": r["depth"]["delta1"],
                "latency_median_ms": r["latency_median_ms"],
                "epe_delta_vs_fp32_px": r["disparity"]["epe"] - base["disparity"]["epe"],
                "d1_delta_vs_fp32_points": r["disparity"]["d1"] - base["disparity"]["d1"],
                "max_abs_prediction_diff_vs_fp32_px": float(
                    np.abs(r["predictions"] - base["predictions"]).max()
                ),
                "mean_abs_prediction_diff_vs_fp32_px": float(
                    np.abs(r["predictions"] - base["predictions"]).mean()
                ),
            }
        exp.metric("precision_comparison", table)
        exp.metric("model_sizes", sizes)
        exp.metric(
            "hailo_published_for_context",
            {
                "float_metric": 8.223,
                "hardware_metric": 10.3,
                "note": "SOURCE: SR-002 and SR-004. Hailo's own quantiser on "
                "Hailo-8, on the D1 metric mislabelled EPE. Not comparable with "
                "the int8 row above.",
            },
        )
        (OUT_DIR / "precision_comparison.json").write_text(
            json.dumps({"table": table, "sizes": sizes}, indent=2), encoding="utf-8"
        )

        lines = [
            "Precision comparison, KITTI 2015 Hailo validation split, "
            "{} scenes".format(n),
            "",
            "{:<24} {:>9} {:>9} {:>10} {:>10} {:>9} {:>11}".format(
                "configuration", "EPE px", "D1 %", "AbsRel", "RMSE m",
                "delta1 %", "latency ms"
            ),
        ]
        for key, v in table.items():
            lines.append(
                "{:<24} {:>9.3f} {:>9.3f} {:>10.4f} {:>10.3f} {:>9.1f} "
                "{:>11.1f}".format(
                    key, v["epe_px"], v["d1_percent"], v["depth_abs_rel"],
                    v["depth_rmse_m"], v["depth_delta1_percent"],
                    v["latency_median_ms"],
                )
            )
        text = "\n".join(lines)
        (OUT_DIR / "report.txt").write_text(text, encoding="utf-8")

        if "fp16_torch_cuda" in table:
            f16 = table["fp16_torch_cuda"]
            exp.note(
                "fp16 changes EPE by {:+.4f} px and D1 by {:+.4f} points against "
                "fp32 on the same device, with a maximum per-pixel difference of "
                "{:.3f} px.".format(
                    f16["epe_px"] - table["fp32_torch_cuda"]["epe_px"],
                    f16["d1_percent"] - table["fp32_torch_cuda"]["d1_percent"],
                    f16["max_abs_prediction_diff_vs_fp32_px"],
                )
            )
        if "int8_onnxruntime_cpu" in table:
            i8 = table["int8_onnxruntime_cpu"]
            exp.note(
                "int8 changes EPE by {:+.3f} px and D1 by {:+.3f} points against "
                "fp32 on the same runtime, with a maximum per-pixel difference "
                "of {:.3f} px.".format(
                    i8["epe_delta_vs_fp32_px"], i8["d1_delta_vs_fp32_points"],
                    i8["max_abs_prediction_diff_vs_fp32_px"],
                )
            )
        exp.conclude(
            "Precision results recorded above; interpretation in "
            "docs/hardware_analysis.md. Nothing here describes Hailo silicon."
        )
        print("\n" + text + "\n\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
