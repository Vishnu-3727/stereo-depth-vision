"""EXP: run the Hailo StereoNet ONNX and try to reproduce its published figure.

Executes the reference ONNX with onnxruntime over Hailo's own 40-scene KITTI
2015 validation slice, under several protocol variants, and reports what each
one produces. The published value is compared against, never optimised toward:
the variants are fixed in advance by what the Model Zoo source says it does,
not adjusted until something matches.

    python scripts/exp_reproduce_hailo.py [--limit N] [--provider cpu|cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.experiment import Experiment  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    disparity_metrics,
    hailo_image_outlier_rate,
)

MODEL = REPO_ROOT / "reference" / "onnx" / "stereonet.onnx"
DATA = REPO_ROOT / "data" / "kitti2015"
OUT_DIR = REPO_ROOT / "results" / "reproduction"

# Hailo publishes 8.223 as "full_precision_result" with "eval_metric: EPE"
# [SR-002], and 8.22 as "Float EPE" in the Hailo-8 table [SR-004].
HAILO_PUBLISHED = 8.223

# The protocol variants, fixed before running. Each is a defensible reading of
# the evidence; the point is to see which one the published number corresponds
# to, and how much the choices matter.
VARIANTS = {
    "hailo_exact": {
        "description": (
            "Hailo's evaluator reimplemented exactly: disp_occ_0 ground truth "
            "scaled by 1/255, per-image outlier rate averaged over images."
        ),
        "disparity_scale": 255.0,
        "occluded": True,
        "aggregate": "per_image",
    },
    "hailo_scale_kitti_gt": {
        "description": (
            "As above but with KITTI's official 1/256 ground-truth scale, to "
            "isolate the effect of Hailo's non-standard divisor."
        ),
        "disparity_scale": 256.0,
        "occluded": True,
        "aggregate": "per_image",
    },
    "kitti_official_d1_all": {
        "description": (
            "KITTI convention: 1/256 ground truth, outliers pooled over all "
            "valid pixels rather than averaged per image. D1-all."
        ),
        "disparity_scale": 256.0,
        "occluded": True,
        "aggregate": "pooled",
    },
    "kitti_official_d1_noc": {
        "description": "As above on disp_noc_0, excluding occluded pixels. D1-noc.",
        "disparity_scale": 256.0,
        "occluded": False,
        "aggregate": "pooled",
    },
}


def build_session(provider: str):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if provider == "cuda":
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print("CUDA provider not available in this onnxruntime build; using CPU")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(MODEL), sess_options=opts, providers=providers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="evaluate only N scenes")
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session(args.provider)
    input_names = [i.name for i in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    base = Kitti2015Stereo(DATA, split="hailo_val")
    n_scenes = len(base) if args.limit is None else min(args.limit, len(base))

    config = {
        "dataset": "kitti2015",
        "split": "hailo_val (training scenes 160-199, frame _10)",
        "resolution": [368, 1232],
        "crop": "pad bottom/right then crop top-left, per Hailo preprocessing",
        "disparity_range": "0-176 px in 16 px steps (12 candidates at 1/16)",
        "batch_size": 1,
        "precision": "fp32",
        "seed": None,
        "model": "reference/onnx/stereonet.onnx",
        "runtime": "onnxruntime " + __import__("onnxruntime").__version__,
        "providers": session.get_providers(),
        "scenes": n_scenes,
    }

    with Experiment(
        "Reproduce Hailo's published StereoNet KITTI figure by running the "
        "reference ONNX under several protocol variants",
        config=config,
    ) as exp:
        exp.note(
            "Hailo's Model Zoo evaluator for this model does not compute "
            "end-point error, despite 'eval_metric: EPE' in the configuration. "
            "Its update_op computes the KITTI D1 outlier rate -- error above "
            "3 px AND above 5 % of ground truth -- names the variable "
            "three_pixel_correct_rate, averages it per image, and reports it as "
            "a percentage because Eval.is_percentage() defaults to True and the "
            "stereo evaluator does not override it."
        )

        # -- inference, once; every variant scores the same predictions -----
        preds: dict[str, np.ndarray] = {}
        latencies = []
        for i in range(n_scenes):
            s = base[i]
            feed = {
                input_names[0]: normalize(s.left),
                input_names[1]: normalize(s.right),
            }
            t0 = time.perf_counter()
            out = session.run([output_name], feed)[0]
            latencies.append(time.perf_counter() - t0)
            preds[s.name] = out[0, 0].astype(np.float32)
            if (i + 1) % 10 == 0:
                exp.log("inferred {}/{}".format(i + 1, n_scenes))

        exp.metric("scenes_evaluated", n_scenes)
        exp.metric("onnxruntime_providers", session.get_providers())
        exp.metric("latency_mean_s", float(np.mean(latencies)))
        exp.metric("latency_median_s", float(np.median(latencies)))
        exp.metric(
            "note_on_latency",
            "onnxruntime on this host under no thermal or memory control. "
            "Indicative only; the profiling experiments in W8 are the "
            "authority, and neither is comparable to Hailo silicon.",
        )

        pred_stats = np.concatenate([p.ravel() for p in preds.values()])
        exp.metric(
            "prediction_distribution",
            {
                "min": float(pred_stats.min()),
                "max": float(pred_stats.max()),
                "mean": float(pred_stats.mean()),
                "p99": float(np.percentile(pred_stats, 99)),
                "fraction_zero": float((pred_stats == 0).mean()),
            },
        )

        # -- score under each variant ---------------------------------------
        results = {}
        for key, v in VARIANTS.items():
            ds = Kitti2015Stereo(
                DATA,
                split="hailo_val",
                disparity_scale=v["disparity_scale"],
                occluded=v["occluded"],
            )
            per_image_rates = []
            all_pred, all_gt = [], []
            for i in range(n_scenes):
                s = ds[i]
                pred = preds[s.name]
                per_image_rates.append(hailo_image_outlier_rate(pred, s.disparity))
                valid = s.disparity > 0
                all_pred.append(pred[valid])
                all_gt.append(s.disparity[valid])

            pooled = disparity_metrics(
                np.concatenate(all_pred), np.concatenate(all_gt)
            )
            headline = (
                100.0 * float(np.mean(per_image_rates))
                if v["aggregate"] == "per_image"
                else pooled.d1
            )
            results[key] = {
                "description": v["description"],
                "aggregate": v["aggregate"],
                "disparity_scale": v["disparity_scale"],
                "ground_truth": "disp_occ_0" if v["occluded"] else "disp_noc_0",
                "headline_percent": headline,
                "gap_vs_published": headline - HAILO_PUBLISHED,
                "pooled_metrics": pooled.as_dict(),
                "per_image_outlier_rate_percent_mean": 100.0 * float(
                    np.mean(per_image_rates)
                ),
                "per_image_outlier_rate_percent_std": 100.0 * float(
                    np.std(per_image_rates)
                ),
            }
            exp.log(
                "{:<24} headline {:6.3f}%   true EPE {:6.3f} px   "
                "D1 pooled {:6.3f}%".format(
                    key, headline, pooled.epe, pooled.d1
                )
            )

        exp.metric("variants", results)
        exp.metric("hailo_published", HAILO_PUBLISHED)
        (OUT_DIR / "variants.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            OUT_DIR / "predictions.npz", **{k: v for k, v in preds.items()}
        )

        exact = results["hailo_exact"]
        true_epe = results["kitti_official_d1_all"]["pooled_metrics"]["epe"]
        exp.note(
            "Under Hailo's exact protocol the model scores {:.3f} %, against a "
            "published {:.3f} % -- a gap of {:+.3f} points.".format(
                exact["headline_percent"], HAILO_PUBLISHED,
                exact["gap_vs_published"],
            )
        )
        exp.note(
            "The model's actual end-point error on the same predictions is "
            "{:.3f} px under the official KITTI ground-truth scale. The "
            "published 8.223 and this figure are different quantities and must "
            "never be compared with each other or with another model's EPE.".format(
                true_epe
            )
        )
        exp.note(
            "Hailo's 1/255 ground-truth divisor inflates ground-truth "
            "disparities by 256/255. Its effect on the headline number is the "
            "difference between the hailo_exact and hailo_scale_kitti_gt "
            "variants: {:+.3f} points.".format(
                results["hailo_scale_kitti_gt"]["headline_percent"]
                - exact["headline_percent"]
            )
        )
        exp.conclude(
            "The published figure is a KITTI D1 outlier rate in percent, not an "
            "end-point error in pixels. Reproduction result and residual gap "
            "recorded above; the apparent seven-fold disagreement with the "
            "StereoNet paper's KITTI numbers was a units problem, not an "
            "accuracy problem."
        )
        print("\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
