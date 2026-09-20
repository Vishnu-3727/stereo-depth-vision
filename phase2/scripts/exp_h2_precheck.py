"""EXP-H2 numerical pre-check -- does the scaling actually desaturate anything?

Runs BEFORE any H2 training, and gates it. Both arms of this diagnostic use the
**same weights** (the unmodified `EXP-H1-WORKING-v2` checkpoint) and the **same
inputs**; the only difference is whether the aggregated cost is standardised
before the soft-argmin. Nothing is trained, no checkpoint is written, no
experiment record is created by this script.

What it cannot tell us, stated up front: this measures what the scaling does to
a network trained *without* it. It is a numerical diagnostic of the operator,
not a prediction of what an H2-trained model will do.

    python phase2/scripts/exp_h2_precheck.py            # 4 val scenes + 40 crops
    python phase2/scripts/exp_h2_precheck.py --batches 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2.models.scaled_regression import standardise_across_disparity  # noqa: E402
from phase2.scripts.exp_h1_cost_volume import CROP_H, CROP_W  # noqa: E402
from phase2.viz import core  # noqa: E402
from src.datasets.kitti2015 import normalize  # noqa: E402
from src.losses.disparity import masked_smooth_l1  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402
from src.models.stereonet.regression import soft_argmin  # noqa: E402

CONTROL_EXPERIMENT = "EXP-H1-WORKING-v2"
OUT_DIR = REPO_ROOT / "phase2" / "results" / "h2_precheck"
FOCUS_SCENES = [27, 0, 31, 6]

# A gradient counts as present when its norm exceeds this. fp32's smallest
# normal magnitude is ~1.2e-38, so 1e-12 is far above the noise floor and far
# below any gradient that could influence an Adam step. Exact-zero counts are
# recorded separately, so the threshold never hides the difference between
# "tiny" and "structurally zero".
GRADIENT_PRESENT_THRESHOLD = 1e-12

# fp32 exp() underflows to exactly zero below about -88; a top-2 cost gap above
# that makes the soft-argmin's derivative exactly zero.
SATURATION_GAP = 88.0


def build_control(device: str) -> StereoNet:
    """The control network: EXP-H1-WORKING-v2 exactly as trained."""
    ckpt = torch.load(core.checkpoint_path("WORKING"), map_location="cpu",
                      weights_only=False)
    assert ckpt["config"]["cost_volume_shift"] == "left", ckpt["config"]
    model = StereoNet(StereoNetConfig(cost_volume_shift="left"))
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def cost_report(cost: torch.Tensor) -> dict:
    """Saturation statistics for the tensor the softmax actually consumes.

    ``cost`` is ``(B, D, H, W)``. Everything is computed in float64 so that the
    statistics of a 1e19-magnitude tensor are not themselves an overflow.
    """
    c = cost.detach().double()
    ordered, _ = torch.sort(c, dim=1)
    gap = ordered[:, 1] - ordered[:, 0]
    weights = torch.softmax(-c, dim=1)
    entropy = -(weights * torch.log(weights.clamp_min(1e-300))).sum(dim=1)
    return {
        "min": float(c.min()), "max": float(c.max()),
        "mean": float(c.mean()), "std": float(c.std()),
        "median_top2_gap": float(gap.median()),
        "min_top2_gap": float(gap.min()),
        "fraction_pixels_gap_below_88": float((gap < SATURATION_GAP).double().mean()),
        "exact_ties": int((gap == 0).sum()),
        "mean_softmax_entropy_nats": float(entropy.mean()),
        "max_possible_entropy_nats": float(np.log(c.shape[1])),
        "mean_max_softmax_weight": float(weights.max(dim=1).values.mean()),
        "fraction_zero_softmax_probabilities": float((weights == 0).double().mean()),
    }


def disparity_report(initial: torch.Tensor) -> dict:
    d = initial.detach().double()
    return {"min": float(d.min()), "max": float(d.max()),
            "mean": float(d.mean()), "std": float(d.std())}


def parameter_gradient_norms(model: StereoNet) -> dict:
    """Gradient norms per stage, with the presence test spelled out."""
    groups = {"feature_extractor": 0.0, "aggregation": 0.0, "refinement": 0.0}
    exact_zero = {k: 0 for k in groups}
    counted = {k: 0 for k in groups}
    for name, p in model.named_parameters():
        stage = name.split(".")[0]
        if stage not in groups or p.grad is None:
            continue
        n = float(p.grad.norm())
        groups[stage] += n ** 2
        counted[stage] += 1
        if n == 0.0:
            exact_zero[stage] += 1
    norms = {k: v ** 0.5 for k, v in groups.items()}
    matching = (norms["feature_extractor"] ** 2 + norms["aggregation"] ** 2) ** 0.5
    return {
        "stage_norms": norms,
        "matching_path_norm": matching,
        "matching_path_present": bool(matching > GRADIENT_PRESENT_THRESHOLD),
        "tensors_with_exactly_zero_gradient": exact_zero,
        "tensors_counted": counted,
        "total_norm": sum(v ** 2 for v in norms.values()) ** 0.5,
    }


def run_one(model: StereoNet, left: torch.Tensor, right: torch.Tensor,
            gt: torch.Tensor, scaled: bool, max_disparity: float) -> dict:
    """One forward+backward, with the regression stage written out explicitly.

    The stages are re-expressed here (rather than calling ``model.regression``)
    so that the cost tensor entering the softmax can be retained and its
    gradient counted. The arithmetic is the frozen `soft_argmin` either way.
    """
    model.zero_grad(set_to_none=True)
    size = (left.shape[-2], left.shape[-1])
    volume = model.cost_volume(model.feature_extractor(left),
                              model.feature_extractor(right))
    cost = model.aggregation(volume)
    upsampled = F.interpolate(cost, size=size, mode="bilinear", align_corners=True)
    upsampled.retain_grad()
    consumed = standardise_across_disparity(upsampled, dim=1) if scaled else upsampled
    # Two different tensors, and the difference matters: `upsampled` is the raw
    # aggregated cost (magnitude ~1e19 in the control), `consumed` is what the
    # softmax actually sees. Standardisation divides by that magnitude, so the
    # gradient w.r.t. the raw cost is scaled down by ~1/std while the gradient
    # w.r.t. the parameters is not. Reporting only one of them would be
    # misleading in opposite directions for the two arms.
    consumed.retain_grad()
    initial = soft_argmin(consumed, dim=1)
    prediction = initial + model.refinement(initial, left)
    if model.config.final_relu:
        prediction = torch.relu(prediction)
    loss, valid_pixels = masked_smooth_l1(prediction, gt, max_disparity=max_disparity)
    loss.backward()

    def tensor_grad_report(grad: torch.Tensor) -> dict:
        return {
            "norm": float(grad.norm()),
            "max_abs": float(grad.abs().max()),
            "pixels_with_gradient": int((grad != 0).any(dim=1).sum()),
            "pixels_total": int(grad.shape[0] * grad.shape[2] * grad.shape[3]),
            "present": bool(float(grad.norm()) > GRADIENT_PRESENT_THRESHOLD),
        }

    report = {
        "loss": float(loss),
        "valid_pixels": int(valid_pixels),
        "cost_entering_softmax": cost_report(consumed),
        "disparity_initial": disparity_report(initial),
        # the tensor the softmax consumes (standardised in the H2 arm)
        "softmax_input_gradient": tensor_grad_report(consumed.grad),
        # the raw aggregated cost before any standardisation
        "cost_tensor_gradient": tensor_grad_report(upsampled.grad),
        "parameter_gradients": parameter_gradient_norms(model),
    }
    model.zero_grad(set_to_none=True)
    return report


def summarise(rows: list[dict]) -> dict:
    def pick(path):
        out = []
        for r in rows:
            v = r
            for key in path:
                v = v[key]
            out.append(float(v))
        return np.array(out)

    matching = pick(["parameter_gradients", "matching_path_norm"])
    total = pick(["parameter_gradients", "total_norm"])
    return {
        "batches": len(rows),
        "matching_path_gradient_present_fraction": float(np.mean(
            [r["parameter_gradients"]["matching_path_present"] for r in rows])),
        "softmax_input_gradient_present_fraction": float(np.mean(
            [r["softmax_input_gradient"]["present"] for r in rows])),
        "raw_cost_gradient_present_fraction": float(np.mean(
            [r["cost_tensor_gradient"]["present"] for r in rows])),
        "median_softmax_input_gradient_norm": float(np.median(
            [r["softmax_input_gradient"]["norm"] for r in rows])),
        "median_pixels_with_cost_gradient": float(np.median(
            pick(["softmax_input_gradient", "pixels_with_gradient"]))),
        "pixels_total": int(rows[0]["cost_tensor_gradient"]["pixels_total"]),
        "mean_softmax_entropy_nats": float(np.mean(
            pick(["cost_entering_softmax", "mean_softmax_entropy_nats"]))),
        "mean_max_softmax_weight": float(np.mean(
            pick(["cost_entering_softmax", "mean_max_softmax_weight"]))),
        "mean_fraction_pixels_gap_below_88": float(np.mean(
            pick(["cost_entering_softmax", "fraction_pixels_gap_below_88"]))),
        "median_top2_gap": float(np.median(
            pick(["cost_entering_softmax", "median_top2_gap"]))),
        "disparity_initial_std_mean": float(np.mean(
            pick(["disparity_initial", "std"]))),
        "matching_path_gradient_norm": {
            "median": float(np.median(matching)), "max": float(matching.max()),
            "p90": float(np.percentile(matching, 90)),
        },
        "total_gradient_norm": {
            "median": float(np.median(total)), "max": float(total.max()),
            "p90": float(np.percentile(total, 90)),
            "batches_above_1e4": int((total > 1e4).sum()),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=40,
                    help="training-recipe crops from hailo_calib")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_control(device)
    model.train()
    max_disparity = float(model.config.max_disparity_px)

    # identical crops for both arms, drawn once
    rng = np.random.default_rng(0)
    calib = [core.load_scene(i, split="hailo_calib") for i in range(20)]
    h, w = calib[0].shape
    crops = [(calib[int(rng.integers(0, len(calib)))],
              int(rng.integers(0, h - CROP_H + 1)),
              int(rng.integers(0, w - CROP_W + 1))) for _ in range(args.batches)]

    result: dict = {
        "purpose": (
            "numerical pre-check for EXP-H2-SOFTARGMIN-SCALE: does per-pixel "
            "standardisation of the aggregated cost across the disparity axis "
            "change the saturation regime, using the control's own weights"
        ),
        "protocol": {
            "weights": "EXP-H1-WORKING-v2 checkpoint, unmodified, for BOTH arms",
            "checkpoint_sha256": core.sha256(core.checkpoint_path("WORKING")),
            "inputs": "{} random {}x{} crops from hailo_calib (identical for both "
                      "arms) plus full frames of hailo_val scenes {}".format(
                          args.batches, CROP_H, CROP_W, FOCUS_SCENES),
            "loss": "masked smooth L1, max_disparity={}".format(max_disparity),
            "backward": "yes, gradients measured; no optimizer step, nothing saved",
            "gradient_present_threshold": GRADIENT_PRESENT_THRESHOLD,
            "saturation_gap_threshold": SATURATION_GAP,
            "device": device,
        },
        "git_revision": core.git_revision(),
        "crops": {}, "scenes": {},
    }

    for label, scaled in (("control", False), ("h2_scaled", True)):
        rows = []
        for scene, y, x in crops:
            sl = (slice(y, y + CROP_H), slice(x, x + CROP_W))
            rows.append(run_one(
                model,
                torch.from_numpy(normalize(scene.left[sl])).to(device),
                torch.from_numpy(normalize(scene.right[sl])).to(device),
                torch.from_numpy(scene.gt_disparity[sl][None, None].astype(np.float32)).to(device),
                scaled, max_disparity))
        result["crops"][label] = {"summary": summarise(rows), "batches": rows}
        print("precheck: {} crops done".format(label))

    for index in FOCUS_SCENES:
        scene = core.load_scene(index)
        entry = {"name": scene.name}
        for label, scaled in (("control", False), ("h2_scaled", True)):
            entry[label] = run_one(
                model,
                torch.from_numpy(normalize(scene.left)).to(device),
                torch.from_numpy(normalize(scene.right)).to(device),
                torch.from_numpy(scene.gt_disparity[None, None].astype(np.float32)).to(device),
                scaled, max_disparity)
        result["scenes"][str(index)] = entry
        print("precheck: scene {:>2} {} done".format(index, scene.name))

    path = OUT_DIR / "precheck.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("wrote " + str(path))

    c = result["crops"]["control"]["summary"]
    h = result["crops"]["h2_scaled"]["summary"]
    print("\n{:<46} {:>16} {:>16}".format("", "CONTROL", "H2 (scaled)"))
    for key, fmt in (("mean_softmax_entropy_nats", "{:>16.4f}"),
                     ("mean_max_softmax_weight", "{:>16.6f}"),
                     ("mean_fraction_pixels_gap_below_88", "{:>16.4f}"),
                     ("median_top2_gap", "{:>16.4g}"),
                     ("disparity_initial_std_mean", "{:>16.4f}"),
                     ("matching_path_gradient_present_fraction", "{:>16.3f}"),
                     ("softmax_input_gradient_present_fraction", "{:>16.3f}"),
                     ("raw_cost_gradient_present_fraction", "{:>16.3f}"),
                     ("median_softmax_input_gradient_norm", "{:>16.4g}"),
                     ("median_pixels_with_cost_gradient", "{:>16.1f}")):
        print(("{:<46}" + fmt + fmt).format(key, c[key], h[key]))
    print("{:<46}{:>16.4g}{:>16.4g}".format(
        "matching-path grad norm (median)",
        c["matching_path_gradient_norm"]["median"], h["matching_path_gradient_norm"]["median"]))
    print("{:<46}{:>16.4g}{:>16.4g}".format(
        "total grad norm (median)",
        c["total_gradient_norm"]["median"], h["total_gradient_norm"]["median"]))


if __name__ == "__main__":
    main()
