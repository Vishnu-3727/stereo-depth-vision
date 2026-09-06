"""EXP: what each stage of the reference ONNX actually contributes.

Exposes internal tensors of the Hailo StereoNet graph as extra outputs and runs
real KITTI pairs through it, to establish the magnitude and role of each stage
rather than inferring them from the topology.

The specific question that motivated this: the baked-in soft-argmin index grid
contains the values 0..11, so the regression stage emits a disparity in units of
low-resolution *candidates*, bounded by 11. Yet the model's final output reaches
136 px. Something between those two facts has to supply the remaining
magnitude, and the topology alone cannot say what.

The reference ONNX on disk is never modified; the extra outputs are added to an
in-memory copy.

    python scripts/exp_intermediate_tensors.py [--scenes N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.experiment import Experiment  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402

MODEL = REPO_ROOT / "reference" / "onnx" / "stereonet.onnx"
OUT_DIR = REPO_ROOT / "results" / "intermediates"

# Node index -> a name for the tensor it produces. Taken from the node walk in
# results/onnx_inspection/report.txt.
TAPS = {
    34: "left_features",           # 1x32x23x77
    69: "right_features",          # 1x32x23x77
    117: "cost_volume",            # 1x32x12x23x77
    127: "aggregated_cost",        # 1x12x23x77
    128: "cost_upsampled",         # 1x12x368x1232
    130: "cost_softmax",           # 1x12x368x1232
    132: "disparity_initial",      # 1x1x368x1232  <- soft-argmin output
    165: "refinement_residual",    # 1x1x368x1232  <- what refinement adds
    166: "disparity_sum",          # 1x1x368x1232
    167: "disparity_final",        # 1x1x368x1232
}


def stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {
        "shape": list(a.shape),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "abs_mean": float(np.abs(a).mean()),
        "p01": float(np.percentile(a, 1)),
        "p99": float(np.percentile(a, 99)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=5)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(MODEL))
    graph = model.graph
    existing = {o.name for o in graph.output}
    tap_names = {}
    for idx, label in TAPS.items():
        name = graph.node[idx].output[0]
        tap_names[label] = name
        if name not in existing:
            graph.output.extend([onnx.helper.ValueInfoProto(name=name)])

    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in session.get_inputs()]
    fetch = list(tap_names.values())

    ds = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    n = min(args.scenes, len(ds))

    config = {
        "dataset": "kitti2015",
        "split": "hailo_val",
        "resolution": [368, 1232],
        "crop": "Hailo pad-and-crop",
        "disparity_range": "12 candidates at 1/16",
        "batch_size": 1,
        "precision": "fp32",
        "seed": None,
        "scenes": n,
        "model": "reference/onnx/stereonet.onnx with internal tensors exposed",
    }

    with Experiment(
        "Stage-by-stage tensor magnitudes inside the reference StereoNet ONNX",
        config=config,
    ) as exp:
        accum: dict[str, list] = {k: [] for k in TAPS.values()}
        for i in range(n):
            s = ds[i]
            feed = {
                input_names[0]: normalize(s.left),
                input_names[1]: normalize(s.right),
            }
            outs = session.run(fetch, feed)
            for label, arr in zip(tap_names.keys(), outs):
                accum[label].append(arr)
            exp.log("scene " + s.name)

        summary = {}
        for label, arrays in accum.items():
            stacked = np.concatenate([a.ravel() for a in arrays])
            summary[label] = stats(stacked)
            summary[label]["shape"] = list(arrays[0].shape)
        exp.metric("stage_statistics", summary)

        init = summary["disparity_initial"]
        resid = summary["refinement_residual"]
        final = summary["disparity_final"]

        # How much of the final disparity does each part actually supply?
        contribution = {
            "soft_argmin_abs_mean": init["abs_mean"],
            "refinement_abs_mean": resid["abs_mean"],
            "final_abs_mean": final["abs_mean"],
            "refinement_share_of_magnitude": resid["abs_mean"]
            / (init["abs_mean"] + resid["abs_mean"]),
        }
        exp.metric("magnitude_contribution", contribution)

        # correlation between the initial estimate and the final output
        a = np.concatenate([x.ravel() for x in accum["disparity_initial"]])
        b = np.concatenate([x.ravel() for x in accum["disparity_final"]])
        c = np.concatenate([x.ravel() for x in accum["refinement_residual"]])
        exp.metric(
            "correlations",
            {
                "initial_vs_final": float(np.corrcoef(a, b)[0, 1]),
                "refinement_vs_final": float(np.corrcoef(c, b)[0, 1]),
            },
        )
        exp.metric(
            "final_output_negative_fraction_before_relu",
            float((np.concatenate([x.ravel() for x in accum["disparity_sum"]]) < 0).mean()),
        )

        (OUT_DIR / "stage_statistics.json").write_text(
            json.dumps({"stages": summary, "contribution": contribution}, indent=2),
            encoding="utf-8",
        )

        exp.note(
            "The soft-argmin output spans {:.2f} to {:.2f} with mean magnitude "
            "{:.2f}. That is bounded by 11 by construction: the baked-in index "
            "grid holds the integers 0..11, so the regression emits disparity in "
            "units of low-resolution candidates, not full-resolution pixels.".format(
                init["min"], init["max"], init["abs_mean"]
            )
        )
        exp.note(
            "The refinement residual spans {:.2f} to {:.2f} with mean magnitude "
            "{:.2f}, against a final output spanning {:.2f} to {:.2f}. The "
            "refinement supplies {:.1%} of the output magnitude.".format(
                resid["min"], resid["max"], resid["abs_mean"],
                final["min"], final["max"],
                contribution["refinement_share_of_magnitude"],
            )
        )
        exp.note(
            "The index grid /Tile_output_0 is 21,762,048 bytes and is constant "
            "across height and width -- it holds twelve distinct values tiled "
            "368x1232 times. A broadcast against a 12-element vector would be "
            "numerically identical and occupy 48 bytes. The tiling is an export "
            "artifact of building the grid with repeat() at runtime, and it is "
            "92% of the model file."
        )
        exp.conclude(
            "Recorded above. The stage magnitudes settle what the topology alone "
            "could not: which part of the network actually determines the output "
            "disparity."
        )

        print("\n" + json.dumps(summary, indent=2))
        print("\ncontribution: " + json.dumps(contribution, indent=2))
        print("\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
