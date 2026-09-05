"""EXP: does the reference model actually use the right image?

EXP-006 found that the soft-argmin output is negatively correlated with the
final disparity (r = -0.975) while the refinement residual is almost perfectly
correlated with it (r = +0.9998). Two readings are consistent with that:

  H1  The matching stage carries real stereo information, and the refinement
      network transforms it into pixel units in a way that happens to invert the
      linear correlation.
  H2  The matching stage carries little usable information, and the model is
      predicting disparity mostly from the left image alone -- closer to
      monocular depth estimation with a weak stereo prior than to stereo
      matching.

These make opposite predictions about what happens when the right image is
corrupted. Under H1 accuracy should collapse; under H2 it should barely move.
So: run the frozen baseline with the right input replaced in several ways and
measure both how far the output moves and how much accuracy is lost.

Nothing is modified -- this probes the baseline as it stands.

    python scripts/exp_right_image_ablation.py [--scenes N]
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
from src.evaluation.metrics import disparity_metrics  # noqa: E402

MODEL = REPO_ROOT / "reference" / "onnx" / "stereonet.onnx"
OUT_DIR = REPO_ROOT / "results" / "ablation"


def variants(left: np.ndarray, right: np.ndarray, other: np.ndarray) -> dict:
    """Ways of destroying the stereo signal, from mildest to most complete."""
    rng = np.random.default_rng(0)
    return {
        "baseline": right,
        # The correct match for every pixel is now at disparity zero. A working
        # stereo matcher should report a near-uniform, near-zero disparity.
        "right_equals_left": left.copy(),
        # A real image with real structure, but from a different scene: no
        # correspondence exists anywhere.
        "right_from_other_scene": other,
        # Structure destroyed along the epipolar direction specifically.
        "right_flipped_horizontally": right[:, ::-1].copy(),
        # No structure at all.
        "right_black": np.zeros_like(right),
        "right_noise": rng.integers(0, 256, size=right.shape, dtype=np.uint8),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=20)
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
        "disparity_range": "12 candidates at 1/16",
        "batch_size": 1,
        "precision": "fp32",
        "seed": 0,
        "scenes": n,
        "model": "reference/onnx/stereonet.onnx (unmodified)",
    }

    with Experiment(
        "Right-image ablation: how much of the reference model's output depends "
        "on the second camera at all",
        config=config,
    ) as exp:
        exp.note(
            "Tests H1 (matching carries real stereo information) against H2 "
            "(the model predicts mostly from the left image). H1 predicts "
            "accuracy collapses when the right image is corrupted; H2 predicts "
            "it barely moves."
        )

        per_variant: dict[str, dict] = {}
        preds: dict[str, list] = {}
        gts: list = []

        for i in range(n):
            s = ds[i]
            other = ds[(i + 7) % n].right  # a different scene, deterministic
            vs = variants(s.left, s.right, other)
            left_in = normalize(s.left)
            for name, right_img in vs.items():
                out = session.run(
                    [output_name],
                    {input_names[0]: left_in, input_names[1]: normalize(right_img)},
                )[0][0, 0]
                preds.setdefault(name, []).append(out.astype(np.float32))
            gts.append(s.disparity)
            exp.log("scene {}/{} {}".format(i + 1, n, s.name))

        base_stack = np.concatenate([p.ravel() for p in preds["baseline"]])
        valid = np.concatenate([(g > 0).ravel() for g in gts])
        gt_stack = np.concatenate([g.ravel() for g in gts])

        for name, plist in preds.items():
            stack = np.concatenate([p.ravel() for p in plist])
            m = disparity_metrics(stack[valid], gt_stack[valid])
            per_variant[name] = {
                "metrics": m.as_dict(),
                "mean_disparity": float(stack.mean()),
                "std_disparity": float(stack.std()),
                # how far this variant's output moved from the baseline output
                "mean_abs_change_vs_baseline_px": float(np.abs(stack - base_stack).mean()),
                "correlation_with_baseline": float(np.corrcoef(stack, base_stack)[0, 1]),
                "d1_penalty_vs_baseline_points": (
                    m.d1 - disparity_metrics(base_stack[valid], gt_stack[valid]).d1
                ),
            }
            exp.log(
                "{:<28} EPE {:6.3f}  D1 {:6.2f}%  mean {:6.2f}px  "
                "moved {:6.3f}px  corr {:+.4f}".format(
                    name, m.epe, m.d1, stack.mean(),
                    per_variant[name]["mean_abs_change_vs_baseline_px"],
                    per_variant[name]["correlation_with_baseline"],
                )
            )

        exp.metric("variants", per_variant)
        (OUT_DIR / "right_image_ablation.json").write_text(
            json.dumps(per_variant, indent=2), encoding="utf-8"
        )

        base = per_variant["baseline"]["metrics"]
        worst = max(
            (v for k, v in per_variant.items() if k != "baseline"),
            key=lambda v: v["metrics"]["d1"],
        )
        same_lr = per_variant["right_equals_left"]

        exp.metric(
            "largest_d1_penalty_points",
            worst["metrics"]["d1"] - base["d1"],
        )
        exp.note(
            "Baseline EPE {:.3f} px, D1 {:.2f} %. The worst corruption of the "
            "right image reaches EPE {:.3f} px, D1 {:.2f} % -- a penalty of "
            "{:+.2f} D1 points.".format(
                base["epe"], base["d1"],
                worst["metrics"]["epe"], worst["metrics"]["d1"],
                worst["metrics"]["d1"] - base["d1"],
            )
        )
        exp.note(
            "With the right image replaced by the left, every true "
            "correspondence sits at disparity zero, so a stereo matcher should "
            "output near-zero disparity everywhere. The model instead outputs a "
            "mean of {:.2f} px and stays correlated with its baseline output at "
            "r = {:+.4f}.".format(
                same_lr["mean_disparity"], same_lr["correlation_with_baseline"]
            )
        )
        exp.conclude(
            "Numbers recorded above. Which of H1 and H2 they support is stated "
            "in docs/failure_analysis.md against these measurements; the "
            "baseline itself is unchanged."
        )
        print("\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
