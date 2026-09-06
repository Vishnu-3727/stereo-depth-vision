"""EXP: does the independent implementation match the reference ONNX?

Loads the reference weights into the implementation written from the
architecture specification, runs both on identical real inputs, and compares
the final disparity and every intermediate tensor that can be tapped on both
sides.

Agreement is the evidence that the specification in
``docs/stereonet_architecture.md`` is correct. Disagreement anywhere is
reported with its magnitude and location, not smoothed over.

    python scripts/exp_onnx_equivalence.py [--scenes N] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.experiment import Experiment  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.models.stereonet import StereoNet  # noqa: E402
from src.models.stereonet.onnx_weights import load_onnx_weights  # noqa: E402

MODEL = REPO_ROOT / "reference" / "onnx" / "stereonet.onnx"
OUT_DIR = REPO_ROOT / "results" / "equivalence"

# Our stage name -> the ONNX node whose output should hold the same tensor.
TAPS = {
    "left_features": 34,
    "cost_volume": 117,
    "aggregated_cost": 127,
    "disparity_initial": 132,
    "refinement_residual": 165,
    "disparity_final": 167,
}


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    diff = np.abs(a - b)
    scale = max(float(np.abs(b).mean()), 1e-12)
    return {
        "elements": int(a.size),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "relative_mean_abs_diff": float(diff.mean() / scale),
        "reference_abs_mean": scale,
        "max_abs_diff_index": int(diff.argmax()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=5)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    # reference, with intermediate tensors exposed on an in-memory copy
    ref_model = onnx.load(str(MODEL))
    graph = ref_model.graph
    existing = {o.name for o in graph.output}
    tap_names = {}
    for label, idx in TAPS.items():
        name = graph.node[idx].output[0]
        tap_names[label] = name
        if name not in existing:
            graph.output.extend([onnx.helper.ValueInfoProto(name=name)])
    session = ort.InferenceSession(
        ref_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in session.get_inputs()]

    # ours
    model = StereoNet().eval()
    report = load_onnx_weights(model, MODEL)
    model = model.to(device)

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
        "reference": "reference/onnx/stereonet.onnx via onnxruntime CPU",
        "ours": "src/models/stereonet on " + str(device),
    }

    with Experiment(
        "Tensor-level equivalence between the independent implementation and "
        "the reference ONNX",
        config=config,
    ) as exp:
        exp.metric("weight_load_report", {
            k: v for k, v in report.items() if k != "layers"
        })
        exp.metric("model_parameters_unique", model.parameter_count())
        exp.metric(
            "model_parameters_per_occurrence", model.parameter_count(per_occurrence=True)
        )
        exp.note(
            "Weights are mapped by execution order, not by name -- batch-norm "
            "folding renamed most tensors at export. Every shape is asserted "
            "during the load, and the two feature-extractor branches are "
            "checked to reference identical initializers before the mapping is "
            "applied."
        )

        per_stage: dict[str, list] = {k: [] for k in TAPS}
        per_stage["output"] = []

        for i in range(n):
            s = ds[i]
            ln, rn = normalize(s.left), normalize(s.right)
            ref_out = session.run(
                [tap_names[k] for k in TAPS],
                {input_names[0]: ln, input_names[1]: rn},
            )
            ref = dict(zip(TAPS.keys(), ref_out))

            with torch.no_grad():
                _, ours = model(
                    torch.from_numpy(ln).to(device),
                    torch.from_numpy(rn).to(device),
                    return_stages=True,
                )
            ours = {k: v.detach().cpu().numpy() for k, v in ours.items()}

            for label in TAPS:
                per_stage[label].append(compare(ours[label], ref[label]))
            exp.log("scene {}/{} {}".format(i + 1, n, s.name))

        summary = {}
        for label, rows in per_stage.items():
            if not rows:
                continue
            summary[label] = {
                "max_abs_diff": max(r["max_abs_diff"] for r in rows),
                "mean_abs_diff": float(np.mean([r["mean_abs_diff"] for r in rows])),
                "relative_mean_abs_diff": float(
                    np.mean([r["relative_mean_abs_diff"] for r in rows])
                ),
                "reference_abs_mean": float(
                    np.mean([r["reference_abs_mean"] for r in rows])
                ),
            }
            exp.log(
                "{:<22} max |diff| {:.3e}   mean |diff| {:.3e}   "
                "relative {:.3e}".format(
                    label, summary[label]["max_abs_diff"],
                    summary[label]["mean_abs_diff"],
                    summary[label]["relative_mean_abs_diff"],
                )
            )

        exp.metric("stage_differences", summary)
        (OUT_DIR / "stage_differences.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        final = summary["disparity_final"]
        # fp32 accumulation over ~56 GMAC in two different runtimes will not be
        # bit-identical; what matters is that the difference stays at the level
        # of arithmetic noise rather than indicating a structural divergence.
        exp.metric(
            "final_disparity_max_abs_diff_px", final["max_abs_diff"]
        )
        exp.metric(
            "final_disparity_relative_mean_abs_diff", final["relative_mean_abs_diff"]
        )
        exp.note(
            "Final disparity differs by at most {:.3e} px, mean {:.3e} px, "
            "against a reference mean magnitude of {:.2f} px -- a relative "
            "difference of {:.2e}.".format(
                final["max_abs_diff"], final["mean_abs_diff"],
                final["reference_abs_mean"], final["relative_mean_abs_diff"],
            )
        )
        exp.conclude(
            "Differences recorded per stage above. Whether they represent "
            "floating-point accumulation order or a structural disagreement is "
            "argued from the per-stage progression in "
            "docs/reproduction_report.md."
        )
        print("\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
