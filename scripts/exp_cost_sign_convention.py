"""EXP: why is the soft-argmin output anti-correlated with the answer?

EXP-006 measured r = -0.975 between the soft-argmin output and the final
disparity. EXP-007 ruled out the explanation that the matching stage is unused:
corrupting the right image costs 90 D1 points, so the stereo signal is
load-bearing.

That leaves a question about representation. The graph computes
``softmax(-cost)``, which assumes the aggregation network emits a *cost* -- low
at the correct disparity. If the network instead learned to emit a *score* --
high at the correct disparity -- then negating before the softmax would weight
the wrong candidates, and the resulting soft-argmin would run backwards.

Three predictions distinguish these:

  C1 (cost)   soft-argmin of  softmax(-cost) correlates positively with truth
  C2 (score)  soft-argmin of  softmax(+cost) correlates positively with truth,
              and the graph's own output correlates negatively
  C3 (neither) both are weakly correlated and the representation is not a
              per-candidate disparity likelihood at all

Compared against ground-truth disparity in candidate units (true disparity / 16,
since candidates are 16 full-resolution pixels apart).

    python scripts/exp_cost_sign_convention.py [--scenes N]
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
OUT_DIR = REPO_ROOT / "results" / "ablation"
FEATURE_STRIDE = 16  # candidates are 16 full-resolution pixels apart


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def soft_index(weights: np.ndarray, axis: int = 0) -> np.ndarray:
    idx = np.arange(weights.shape[axis], dtype=np.float64)
    shape = [1] * weights.ndim
    shape[axis] = -1
    return (weights * idx.reshape(shape)).sum(axis=axis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=10)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(MODEL))
    graph = model.graph
    # node 128 is the upsampled cost tensor, node 167 the final disparity
    tap_cost = graph.node[128].output[0]
    existing = {o.name for o in graph.output}
    if tap_cost not in existing:
        graph.output.extend([onnx.helper.ValueInfoProto(name=tap_cost)])
    final_name = graph.output[0].name

    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in session.get_inputs()]

    ds = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    n = min(args.scenes, len(ds))

    config = {
        "dataset": "kitti2015",
        "split": "hailo_val",
        "resolution": [368, 1232],
        "crop": "Hailo pad-and-crop",
        "disparity_range": "12 candidates at 1/16, 16 px apart",
        "batch_size": 1,
        "precision": "fp32",
        "seed": None,
        "scenes": n,
        "model": "reference/onnx/stereonet.onnx (unmodified; cost tensor exposed)",
    }

    with Experiment(
        "Sign convention of the aggregated cost tensor: is it a cost or a score?",
        config=config,
    ) as exp:
        gt_all, as_cost_all, as_score_all, argmin_all, argmax_all, final_all = (
            [], [], [], [], [], []
        )

        for i in range(n):
            s = ds[i]
            outs = session.run(
                [final_name, tap_cost],
                {
                    input_names[0]: normalize(s.left),
                    input_names[1]: normalize(s.right),
                },
            )
            final = outs[0][0, 0].astype(np.float64)
            cost = outs[1][0].astype(np.float64)  # (12, H, W)

            valid = s.disparity > 0
            gt_candidates = s.disparity[valid] / FEATURE_STRIDE

            gt_all.append(gt_candidates)
            as_cost_all.append(soft_index(softmax(-cost, axis=0))[valid])
            as_score_all.append(soft_index(softmax(+cost, axis=0))[valid])
            argmin_all.append(np.argmin(cost, axis=0)[valid].astype(np.float64))
            argmax_all.append(np.argmax(cost, axis=0)[valid].astype(np.float64))
            final_all.append(final[valid] / FEATURE_STRIDE)
            exp.log("scene " + s.name)

        gt = np.concatenate(gt_all)
        series = {
            "soft_argmin_as_cost (what the graph computes)": np.concatenate(as_cost_all),
            "soft_argmin_as_score (sign flipped)": np.concatenate(as_score_all),
            "hard_argmin": np.concatenate(argmin_all),
            "hard_argmax": np.concatenate(argmax_all),
            "final_output (candidate units)": np.concatenate(final_all),
        }

        results = {}
        for name, v in series.items():
            results[name] = {
                "pearson_r_with_ground_truth": float(np.corrcoef(v, gt)[0, 1]),
                "mean": float(v.mean()),
                "std": float(v.std()),
                "mean_abs_error_candidates": float(np.abs(v - gt).mean()),
            }
            exp.log(
                "{:<46} r = {:+.4f}   mean {:5.2f}  MAE {:6.2f} candidates".format(
                    name, results[name]["pearson_r_with_ground_truth"],
                    results[name]["mean"],
                    results[name]["mean_abs_error_candidates"],
                )
            )

        exp.metric("ground_truth_candidate_units", {
            "mean": float(gt.mean()), "std": float(gt.std()),
            "min": float(gt.min()), "max": float(gt.max()),
            "fraction_beyond_11_candidates": float((gt > 11).mean()),
        })
        exp.metric("series", results)
        (OUT_DIR / "cost_sign_convention.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

        r_cost = results["soft_argmin_as_cost (what the graph computes)"][
            "pearson_r_with_ground_truth"
        ]
        r_score = results["soft_argmin_as_score (sign flipped)"][
            "pearson_r_with_ground_truth"
        ]
        exp.note(
            "Against ground truth in candidate units: the graph's own "
            "soft-argmin gives r = {:+.4f}, and the same reduction with the "
            "sign flipped gives r = {:+.4f}.".format(r_cost, r_score)
        )
        exp.note(
            "Hard argmin r = {:+.4f}, hard argmax r = {:+.4f}. These do not "
            "depend on the softmax temperature, so they separate the sign "
            "question from any saturation effect.".format(
                results["hard_argmin"]["pearson_r_with_ground_truth"],
                results["hard_argmax"]["pearson_r_with_ground_truth"],
            )
        )
        # No automated verdict here. A single correlation threshold cannot
        # decide between the hypotheses -- with 12 candidates the argmin is
        # close to a reflection of the argmax, so the wrong reading produces a
        # large |r| with the wrong sign. The diagnostic that actually separates
        # them is the sign together with the mean: only one reading lands near
        # the ground-truth mean. Both are recorded and the interpretation is
        # argued in docs/failure_analysis.md.
        exp.metric(
            "diagnostic",
            {
                "ground_truth_mean_candidates": float(gt.mean()),
                "as_cost_mean_candidates": results[
                    "soft_argmin_as_cost (what the graph computes)"]["mean"],
                "as_score_mean_candidates": results[
                    "soft_argmin_as_score (sign flipped)"]["mean"],
                "r_as_cost": r_cost,
                "r_as_score": r_score,
            },
        )
        exp.conclude(
            "Correlations recorded above; the interpretation is written up "
            "against these numbers in docs/failure_analysis.md. The baseline is "
            "unchanged -- this experiment only reads internal tensors."
        )
        print("\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
