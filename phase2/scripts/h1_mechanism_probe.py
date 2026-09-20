"""H1 mechanism probe -- where, if anywhere, does the working cost volume act?

Diagnostic tooling for the H1-v2 forensic investigation. It runs the two frozen
H1-v2 checkpoints in inference mode, records what every stage of the network
does, and breaks the BASE/WORKING difference down spatially. It trains nothing,
writes no experiment record, and touches no checkpoint.

    stages           per-stage statistics for the focus scenes
    spatial          where the two arms differ, stratified over all 40 scenes
    ablation         does either arm use the right image (Phase 1 EXP-007 set)
    intervene        substitute the matching output and measure the effect
    pooled           the same substitutions, pooled over every valid pixel
    cross_shift      each arm's weights under both cost-volume shift settings
    gradients        per-parameter gradient norms under the trained weights
    saturation       softmax saturation vs gradient reaching the matching path
    tie_check        where gradient enters the cost tensor on spiking crops
    weight_drift     how far each stage moved from the shared seed-0 init
    feature_matching can the trained features match without the learned head
    figures          D1-flip maps and stage-comparison figures
    all              every job above

    python phase2/scripts/h1_mechanism_probe.py stages

Results are written as JSON under ``phase2/results/h1_mechanism/``. Every number
carries the protocol that produced it; none of them is the recorded H1-v2
validation protocol, and none of them replaces it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2.viz import core  # noqa: E402
from src.datasets.kitti2015 import normalize  # noqa: E402
from src.evaluation.metrics import disparity_metrics  # noqa: E402
from src.losses.disparity import masked_smooth_l1  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "phase2" / "results" / "h1_mechanism"
FOCUS_SCENES = [27, 0, 31, 6]   # strong WORKING gain, typical gain, near-tie, strong loss

# Which arms a run compares, and where it writes. Defaults reproduce the H1
# investigation exactly; --models / --out-dir let the same probes be pointed at
# a later arm without overwriting the H1 evidence.
ACTIVE_MODELS = ["BASE", "WORKING"]


def load_model(model_key: str, device: str) -> StereoNet:
    ckpt = torch.load(core.checkpoint_path(model_key), map_location="cpu", weights_only=False)
    spec = core.MODELS[model_key]
    shift = spec["cost_volume_shift"]
    # The H1 records store the shift as a top-level config key; the H2 record
    # states it inside "held_fixed" instead, so absence is not a mismatch.
    recorded = ckpt["config"].get("cost_volume_shift")
    assert recorded in (None, shift), (model_key, recorded, shift)
    model = StereoNet(StereoNetConfig(cost_volume_shift=shift))
    if spec.get("regression") == "standardised":
        from phase2.models import scaled_regression
        scaled_regression.apply_to(model)
    model.load_state_dict(ckpt["model"])
    return model.eval().to(device)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# --- 1. stage-by-stage statistics --------------------------------------------

def stage_stats(model: StereoNet, scene: core.Scene, device: str) -> dict:
    """Everything the forward pass reveals, stage by stage, for one scene.

    Shapes (verified, not assumed): features (1,32,23,77); cost volume
    (1,32,12,23,77); aggregated cost (1,12,23,77); disparity_initial,
    refinement_residual and disparity_final (1,1,368,1232).
    """
    # An arm whose regression stage rescales the cost feeds the softmax a
    # different tensor than `aggregated_cost`. Capture it when the stage offers
    # one, so saturation is measured on what the softmax actually consumes
    # rather than on the raw cost.
    capturing = hasattr(model.regression, "capture")
    if capturing:
        model.regression.capture = True
    with torch.no_grad():
        _, st = model(
            torch.from_numpy(normalize(scene.left)).to(device),
            torch.from_numpy(normalize(scene.right)).to(device),
            return_stages=True,
        )
    consumed = None
    if capturing:
        consumed = model.regression.last.get("softmax_input")
        model.regression.capture = False
        model.regression.last = {}

    volume = st["cost_volume"][0]              # (C, D, H, W)
    slice0 = volume[:, :1]
    slice_dev = (volume - slice0).abs()        # how far each slice is from slice 0
    cost = st["aggregated_cost"][0]            # (D, H, W)
    # `softmax_cost` is the tensor the soft-argmin sees: the raw cost for the
    # frozen regression stage, the standardised one for EXP-H2's stage.
    softmax_cost = cost if consumed is None else consumed[0]
    weights = torch.softmax(-softmax_cost, dim=0)
    entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(0)
    argmin = softmax_cost.argmin(0)
    ordered, _ = torch.sort(softmax_cost.double(), dim=0)
    top2_gap = ordered[1] - ordered[0]

    initial = st["disparity_initial"][0, 0].cpu().numpy().astype(np.float64)
    residual = st["refinement_residual"][0, 0].cpu().numpy().astype(np.float64)
    final = st["disparity_final"][0, 0].cpu().numpy().astype(np.float64)

    valid = scene.gt_valid
    gt = scene.gt_disparity.astype(np.float64)
    d = int(cost.shape[0])
    interior = slice(1, d - 1)

    def finite_stats(t: torch.Tensor) -> dict:
        f = torch.isfinite(t)
        vals = t[f].abs()
        return {
            "non_finite_count": int((~f).sum()),
            "elements": int(t.numel()),
            "max_abs_finite": float(vals.max()) if vals.numel() else float("nan"),
            "p99_abs_finite": float(torch.quantile(
                vals[:: max(1, vals.numel() // 100000)].float(), 0.99)) if vals.numel() else float("nan"),
        }

    return {
        "cost_volume": {
            "shape": list(st["cost_volume"].shape),
            "magnitude": finite_stats(volume),
            # 0 for a degenerate volume: every disparity slice equals slice 0
            "max_abs_slice_deviation_from_slice0": float(slice_dev.max()),
            "mean_abs_slice_deviation_from_slice0": float(slice_dev.mean()),
            "mean_std_across_disparity": float(volume.std(dim=1).mean()),
            "mean_abs_value": float(volume.abs().mean()),
        },
        "aggregated_cost": {
            "shape": list(st["aggregated_cost"].shape),
            "magnitude": finite_stats(cost),
            "mean_std_across_disparity": float(cost.std(dim=0).mean()),
            # interior vs edge: with identical slices, a Conv3d can still vary
            # along D purely from its zero padding at the two ends
            "mean_std_across_disparity_interior": float(cost[interior].std(dim=0).mean()),
            "softmax_input_is_standardised": bool(consumed is not None),
            "softmax_input_median_top2_gap": float(top2_gap.median()),
            "softmax_input_fraction_gap_below_88": float((top2_gap < 88).double().mean()),
            "mean_softmax_entropy_nats": float(entropy.mean()),
            "max_possible_entropy_nats": float(np.log(d)),
            "mean_max_softmax_weight": float(weights.max(dim=0).values.mean()),
            "argmin_index_histogram": np.bincount(
                argmin.flatten().cpu().numpy(), minlength=d).tolist(),
            "fraction_pixels_argmin_not_0": float((argmin != 0).float().mean()),
        },
        "disparity_initial": {
            "units": "disparity candidates (0..{})".format(d - 1),
            "min": float(initial.min()), "max": float(initial.max()),
            "mean": float(initial.mean()), "std": float(initial.std()),
            "corr_with_gt_on_valid": corr(initial[valid], gt[valid]),
            "corr_with_final": corr(initial, final),
        },
        "refinement_residual": {
            "mean_abs": float(np.abs(residual).mean()),
            "std": float(residual.std()),
            "corr_with_gt_on_valid": corr(residual[valid], gt[valid]),
            "corr_with_final": corr(residual, final),
            # the final map is initial + residual, so this says which term the
            # magnitude of the answer comes from
            "share_of_final_magnitude": float(
                np.abs(residual).sum() / max(np.abs(residual).sum() + np.abs(initial).sum(), 1e-9)
            ),
        },
        "disparity_final": {
            "min": float(final.min()), "max": float(final.max()),
            "mean": float(final.mean()), "std": float(final.std()),
            "corr_with_gt_on_valid": corr(final[valid], gt[valid]),
            "metrics": disparity_metrics(final[valid], gt[valid]).as_dict(),
        },
    }


def run_stages(device: str, scenes: list[int]) -> dict:
    out = {
        "protocol": (
            "inference only, full 368x1232 frames, hailo_val; per-scene stage "
            "statistics from StereoNet.forward(return_stages=True). Not the "
            "recorded H1-v2 validation protocol."
        ),
        "git_revision": core.git_revision(),
        "scenes": {},
    }
    models = {k: load_model(k, device) for k in ACTIVE_MODELS}
    for index in scenes:
        scene = core.load_scene(index)
        out["scenes"][str(index)] = {
            "name": scene.name,
            **{k: stage_stats(m, scene, device) for k, m in models.items()},
        }
        print("stages: scene {:>2} {} done".format(index, scene.name))
    return out


# --- 2. where the two arms differ --------------------------------------------

def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    grey = image.astype(np.float64).mean(axis=2)
    gy, gx = np.gradient(grey)
    return np.hypot(gx, gy)


def _gt_disparity_gradient(gt: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local ground-truth disparity range, and where it can be measured.

    A 3x3 max-minus-min over pixels that all have ground truth. Sparse LiDAR
    ground truth means this is only defined on a subset, which is reported.
    """
    from scipy.ndimage import maximum_filter, minimum_filter

    filled_hi = np.where(valid, gt, -np.inf)
    filled_lo = np.where(valid, gt, np.inf)
    hi = maximum_filter(filled_hi, size=3)
    lo = minimum_filter(filled_lo, size=3)
    ok = valid & np.isfinite(hi) & np.isfinite(lo)
    return np.where(ok, hi - lo, np.nan), ok


def _bin_report(mask: np.ndarray, base: np.ndarray, work: np.ndarray,
                gt: np.ndarray) -> dict:
    """D1 and mean error for both arms inside one mask, plus flip counts."""
    n = int(mask.sum())
    if n == 0:
        return {"pixels": 0}
    eb = np.abs(base[mask] - gt[mask])
    ew = np.abs(work[mask] - gt[mask])
    g = gt[mask]
    ob = (eb > 3.0) & (eb > 0.05 * g)
    ow = (ew > 3.0) & (ew > 0.05 * g)
    return {
        "pixels": n,
        "base_d1": float(100.0 * ob.mean()),
        "work_d1": float(100.0 * ow.mean()),
        "delta_d1": float(100.0 * (ow.mean() - ob.mean())),
        "base_epe": float(eb.mean()),
        "work_epe": float(ew.mean()),
        "delta_epe": float(ew.mean() - eb.mean()),
        "fixed": int((ob & ~ow).sum()),      # BASE outlier, WORKING inlier
        "broken": int((~ob & ow).sum()),     # BASE inlier, WORKING outlier
        "mean_abs_disparity_difference": float(np.abs(work[mask] - base[mask]).mean()),
    }


def run_spatial(device: str, scenes: list[int]) -> dict:
    """Break the BASE/WORKING difference down by scene structure.

    Every bin is defined from data available without the model: ground-truth
    disparity, its local range, left-image gradient, pixel position. Nothing is
    labelled from the prediction being explained.
    """
    runner = core.ModelRunner(device=device)
    strata: dict[str, dict] = {}
    per_scene = {}
    threshold_bands: dict[str, dict] = {}

    def add(group: str, label: str, mask, base, work, gt):
        rep = _bin_report(mask, base, work, gt)
        if not rep.get("pixels"):
            return
        acc = strata.setdefault(group, {}).setdefault(label, {
            "pixels": 0, "base_outliers": 0, "work_outliers": 0,
            "base_err_sum": 0.0, "work_err_sum": 0.0, "fixed": 0, "broken": 0,
            "abs_diff_sum": 0.0})
        acc["pixels"] += rep["pixels"]
        acc["base_outliers"] += int(round(rep["base_d1"] / 100.0 * rep["pixels"]))
        acc["work_outliers"] += int(round(rep["work_d1"] / 100.0 * rep["pixels"]))
        acc["base_err_sum"] += rep["base_epe"] * rep["pixels"]
        acc["work_err_sum"] += rep["work_epe"] * rep["pixels"]
        acc["fixed"] += rep["fixed"]
        acc["broken"] += rep["broken"]
        acc["abs_diff_sum"] += rep["mean_abs_disparity_difference"] * rep["pixels"]

    for index in scenes:
        scene = core.load_scene(index)
        first, second = ACTIVE_MODELS[0], ACTIVE_MODELS[1]
        base = runner.predict(first, scene).disparity.astype(np.float64)
        work = runner.predict(second, scene).disparity.astype(np.float64)
        gt = scene.gt_disparity.astype(np.float64)
        valid = scene.gt_valid
        h, w = scene.shape

        diff = np.abs(work - base)
        per_scene[str(index)] = {
            "name": scene.name,
            "mean_abs_difference_all_pixels": float(diff.mean()),
            "mean_abs_difference_valid_gt": float(diff[valid].mean()),
            "p99_abs_difference": float(np.percentile(diff, 99)),
            "max_abs_difference": float(diff.max()),
            "overall": _bin_report(valid, base, work, gt),
        }

        # ground-truth disparity magnitude: near vs far
        for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 45), (45, 200)]:
            add("gt_disparity_px", "{}-{}".format(lo, hi),
                valid & (gt >= lo) & (gt < hi), base, work, gt)

        # local ground-truth disparity range: flat surface vs discontinuity
        gt_range, range_ok = _gt_disparity_gradient(gt, valid)
        for lo, hi, label in [(0.0, 0.5, "0-0.5"), (0.5, 2.0, "0.5-2"),
                              (2.0, 5.0, "2-5"), (5.0, 1e9, ">5")]:
            add("local_gt_disparity_range_px", label,
                range_ok & (gt_range >= lo) & (gt_range < hi), base, work, gt)

        # left-image gradient: textureless vs textured (quartiles of this scene)
        grad = _gradient_magnitude(scene.left)
        q = np.percentile(grad[valid], [25, 50, 75])
        for label, mask in [
            ("Q1_flattest", valid & (grad <= q[0])),
            ("Q2", valid & (grad > q[0]) & (grad <= q[1])),
            ("Q3", valid & (grad > q[1]) & (grad <= q[2])),
            ("Q4_most_textured", valid & (grad > q[2])),
        ]:
            add("left_image_gradient_quartile", label, mask, base, work, gt)

        # image columns: the left band is where left-referenced stereo has no
        # match inside the frame for large disparities
        cols = np.arange(w)[None, :].repeat(h, 0)
        for lo, hi in [(0, 64), (64, 128), (128, 400), (400, 800), (800, 1232)]:
            add("column_band", "{}-{}".format(lo, hi),
                valid & (cols >= lo) & (cols < hi), base, work, gt)

        # image rows: KITTI's sky is the top band, the road the bottom one
        rows = np.arange(h)[:, None].repeat(w, 1)
        for lo, hi in [(0, 92), (92, 184), (184, 276), (276, 368)]:
            add("row_band", "{}-{}".format(lo, hi),
                valid & (rows >= lo) & (rows < hi), base, work, gt)

        # how close BASE sits to the D1 threshold, since D1 is a threshold count
        eb = np.abs(base - gt)
        for lo, hi, label in [(0.0, 2.0, "0-2px"), (2.0, 3.0, "2-3px"),
                              (3.0, 4.0, "3-4px"), (4.0, 6.0, "4-6px"),
                              (6.0, 1e9, ">6px")]:
            band = valid & (eb >= lo) & (eb < hi)
            n = int(band.sum())
            if not n:
                continue
            ew = np.abs(work - gt)
            ob = (eb > 3.0) & (eb > 0.05 * gt)
            ow = (ew > 3.0) & (ew > 0.05 * gt)
            acc = threshold_bands.setdefault(label, {
                "pixels": 0, "fixed": 0, "broken": 0, "abs_diff_sum": 0.0})
            acc["pixels"] += n
            acc["fixed"] += int((band & ob & ~ow).sum())
            acc["broken"] += int((band & ~ob & ow).sum())
            acc["abs_diff_sum"] += float(np.abs(work[band] - base[band]).sum())
        print("spatial: scene {:>2} {} done".format(index, scene.name))

    def finish(acc: dict) -> dict:
        n = acc["pixels"]
        return {
            "pixels": n,
            "base_d1": 100.0 * acc["base_outliers"] / n,
            "work_d1": 100.0 * acc["work_outliers"] / n,
            "delta_d1": 100.0 * (acc["work_outliers"] - acc["base_outliers"]) / n,
            "base_epe": acc["base_err_sum"] / n,
            "work_epe": acc["work_err_sum"] / n,
            "delta_epe": (acc["work_err_sum"] - acc["base_err_sum"]) / n,
            "fixed": acc["fixed"],
            "broken": acc["broken"],
            "net_fixed": acc["fixed"] - acc["broken"],
            "mean_abs_disparity_difference": acc["abs_diff_sum"] / n,
        }

    return {
        "protocol": (
            "inference only, full frames, pooled over the listed hailo_val "
            "scenes; strata defined from ground truth and the left image alone, "
            "never from the predictions being compared. Not the recorded H1-v2 "
            "validation protocol."
        ),
        "git_revision": core.git_revision(),
        "scenes_used": scenes,
        "per_scene": per_scene,
        "strata": {g: {k: finish(v) for k, v in labels.items()}
                   for g, labels in strata.items()},
        "base_error_bands": {
            k: {"pixels": v["pixels"], "fixed": v["fixed"], "broken": v["broken"],
                "net_fixed": v["fixed"] - v["broken"],
                "mean_abs_disparity_difference": v["abs_diff_sum"] / v["pixels"]}
            for k, v in threshold_bands.items()
        },
    }


# --- 3. does either model use the right image --------------------------------

def run_ablation(device: str, scenes: list[int]) -> dict:
    """Phase 1's EXP-007 protocol, applied to the two H1-v2 checkpoints.

    Same variant set, same measurements. Phase 1 ran it on the reference ONNX
    weights; this reruns it on our own trained arms so the two are comparable in
    kind, not in value.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from exp_right_image_ablation import variants  # noqa: E402  (read-only import)

    models = {k: load_model(k, device) for k in ACTIVE_MODELS}
    out: dict = {
        "protocol": (
            "Phase 1 EXP-007 variant set, applied to EXP-H1-*-v2 checkpoints; "
            "inference only, {} hailo_val scenes, pooled over gt > 0 "
            "pixels.".format(len(scenes)),
        ),
        "git_revision": core.git_revision(),
        "scenes_used": scenes,
        "models": {},
    }
    loaded = [core.load_scene(i) for i in scenes]
    for key, model in models.items():
        preds: dict[str, list] = {}
        gts, valids = [], []
        for i, scene in enumerate(loaded):
            other = loaded[(i + 7) % len(loaded)].right
            for name, right in variants(scene.left, scene.right, other).items():
                with torch.no_grad():
                    o = model(
                        torch.from_numpy(normalize(scene.left)).to(device),
                        torch.from_numpy(normalize(right)).to(device),
                    )[0, 0].cpu().numpy()
                preds.setdefault(name, []).append(o.astype(np.float32).ravel())
            gts.append(scene.gt_disparity.ravel())
            valids.append(scene.gt_valid.ravel())
        gt = np.concatenate(gts).astype(np.float64)
        valid = np.concatenate(valids)
        baseline = np.concatenate(preds["baseline"]).astype(np.float64)
        base_d1 = disparity_metrics(baseline[valid], gt[valid]).d1
        out["models"][key] = {}
        for name, chunks in preds.items():
            stack = np.concatenate(chunks).astype(np.float64)
            m = disparity_metrics(stack[valid], gt[valid])
            out["models"][key][name] = {
                "epe": m.epe, "d1": m.d1,
                "mean_disparity": float(stack.mean()),
                "mean_abs_change_vs_baseline_px": float(np.abs(stack - baseline).mean()),
                "correlation_with_baseline": corr(stack, baseline),
                "d1_penalty_vs_baseline_points": m.d1 - base_d1,
            }
        print("ablation: {} done".format(key))
    return out


# --- 3b. is the matching output actually used by refinement -------------------

def _forward_parts(model: StereoNet, scene: core.Scene, device: str):
    """Run the network in two halves so the middle can be intervened on."""
    left = torch.from_numpy(normalize(scene.left)).to(device)
    right = torch.from_numpy(normalize(scene.right)).to(device)
    with torch.no_grad():
        lf = model.feature_extractor(left)
        rf = model.feature_extractor(right)
        cost = model.aggregation(model.cost_volume(lf, rf))
        initial = model.regression(cost, (left.shape[-2], left.shape[-1]))
    return left, initial


def _finish(model: StereoNet, left: torch.Tensor, initial: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = initial + model.refinement(initial, left)
        if model.config.final_relu:
            out = torch.relu(out)
    return out[0, 0].cpu().numpy().astype(np.float64)


def run_intervene(device: str, scenes: list[int]) -> dict:
    """Replace the matching stage's output and see whether the answer moves.

    A correlation of zero between ``disparity_initial`` and the final map would
    be only linear evidence. This is the intervention: hold the left image
    fixed, substitute what the matching stage hands to refinement, and measure
    how far the final disparity moves. If it barely moves, refinement is not
    using the matching result, whatever the correlation says.

    Variants (per arm, same scenes):
      as_is                  the arm's own pipeline (control, delta must be 0)
      initial_constant_mean  disparity_initial replaced by its own scalar mean
      initial_constant_0/11  the two ends of the candidate range
      initial_shuffled       the same values, spatially permuted
      initial_from_other_arm the other arm's matching output
    """
    models = {k: load_model(k, device) for k in ACTIVE_MODELS}
    rng = np.random.default_rng(0)
    out: dict = {
        "protocol": (
            "inference-only intervention on the matching -> refinement boundary; "
            "left image and all weights unchanged, only disparity_initial is "
            "substituted. hailo_val scenes {}.".format(scenes)
        ),
        "git_revision": core.git_revision(),
        "models": {k: {} for k in models},
    }
    for index in scenes:
        scene = core.load_scene(index)
        gt = scene.gt_disparity.astype(np.float64)
        valid = scene.gt_valid
        parts = {k: _forward_parts(m, scene, device) for k, m in models.items()}
        for key, model in models.items():
            left, initial = parts[key]
            other_key = [k for k in models if k != key][0]
            flat = initial.flatten()
            perm = torch.from_numpy(rng.permutation(flat.numel())).to(flat.device)
            variants = {
                "as_is": initial,
                "initial_constant_mean": torch.full_like(initial, float(initial.mean())),
                "initial_constant_0": torch.zeros_like(initial),
                "initial_constant_11": torch.full_like(initial, 11.0),
                "initial_shuffled": flat[perm].view_as(initial),
                "initial_from_other_arm": parts[other_key][1],
            }
            baseline = _finish(model, left, initial)
            base_metrics = disparity_metrics(baseline[valid], gt[valid])
            for name, variant in variants.items():
                pred = _finish(model, left, variant)
                m = disparity_metrics(pred[valid], gt[valid])
                out["models"][key].setdefault(name, []).append({
                    "scene": index,
                    "mean_abs_change_px": float(np.abs(pred - baseline).mean()),
                    "max_abs_change_px": float(np.abs(pred - baseline).max()),
                    "epe": m.epe, "d1": m.d1,
                    "delta_epe_vs_as_is": m.epe - base_metrics.epe,
                    "delta_d1_vs_as_is": m.d1 - base_metrics.d1,
                })
        print("intervene: scene {:>2} {} done".format(index, scene.name))

    out["summary"] = {
        key: {
            name: {
                "mean_abs_change_px": float(np.mean([r["mean_abs_change_px"] for r in rows])),
                "max_abs_change_px": float(np.max([r["max_abs_change_px"] for r in rows])),
                "mean_delta_epe": float(np.mean([r["delta_epe_vs_as_is"] for r in rows])),
                "mean_delta_d1": float(np.mean([r["delta_d1_vs_as_is"] for r in rows])),
            }
            for name, rows in variants.items()
        }
        for key, variants in out["models"].items()
    }
    return out


def run_pooled_intervention(device: str, scenes: list[int]) -> dict:
    """The intervention variants scored the way the arms themselves are scored.

    ``run_intervene`` averages per-scene deltas; this pools every valid pixel of
    every scene into one D1/EPE, so the variants sit on the same scale as the
    BASE/WORKING comparison itself.
    """
    models = {k: load_model(k, device) for k in ACTIVE_MODELS}
    rng = np.random.default_rng(0)
    acc: dict[str, dict] = {}

    def add(tag: str, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> None:
        err = np.abs(pred[valid] - gt[valid])
        g = gt[valid]
        outlier = (err > 3.0) & (err > 0.05 * g)
        a = acc.setdefault(tag, {"pixels": 0, "outliers": 0, "error_sum": 0.0})
        a["pixels"] += int(err.size)
        a["outliers"] += int(outlier.sum())
        a["error_sum"] += float(err.sum())

    for index in scenes:
        scene = core.load_scene(index)
        gt = scene.gt_disparity.astype(np.float64)
        valid = scene.gt_valid
        parts = {k: _forward_parts(m, scene, device) for k, m in models.items()}
        for key, model in models.items():
            add(key + " as trained", _finish(model, *parts[key]), gt, valid)
        first, second = ACTIVE_MODELS[0], ACTIVE_MODELS[1]
        left, initial = parts[second]
        work = models[second]
        add("{} initial:=own mean".format(second),
            _finish(work, left, torch.full_like(initial, float(initial.mean()))), gt, valid)
        add("{} initial:=constant 11".format(second),
            _finish(work, left, torch.full_like(initial, 11.0)), gt, valid)
        flat = initial.flatten()
        perm = torch.from_numpy(rng.permutation(flat.numel())).to(flat.device)
        add("{} initial:=shuffled".format(second), _finish(work, left, flat[perm].view_as(initial)),
            gt, valid)
        add("{} initial:={}'s map".format(first, second),
            _finish(models[first], parts[first][0], initial), gt, valid)
        print("pooled: scene {:>2} {} done".format(index, scene.name))

    return {
        "protocol": (
            "inference-only interventions on the matching -> refinement boundary, "
            "pooled over every gt > 0 pixel of hailo_val scenes {}".format(scenes)
        ),
        "git_revision": core.git_revision(),
        "results": {
            tag: {"epe": a["error_sum"] / a["pixels"],
                  "d1": 100.0 * a["outliers"] / a["pixels"],
                  "pixels": a["pixels"]}
            for tag, a in acc.items()
        },
    }


def run_cross_shift(device: str, scenes: list[int]) -> dict:
    """Run each arm's weights under the other arm's cost-volume shift.

    Weights are never modified; only the inference-time shift flag changes. If
    WORKING's accuracy survives its shift being switched off, the trained
    weights -- not the shift acting at inference -- carry whatever difference
    exists.
    """
    rows: dict = {}
    for key in ACTIVE_MODELS:
        ckpt = torch.load(core.checkpoint_path(key), map_location="cpu", weights_only=False)
        spec = core.MODELS[key]
        # Each arm keeps its own regression stage here; only the shift changes.
        # An arm's cost scaling is a different variable and is ablated
        # separately (see the "_no_standardisation" rows below).
        variants = [(shift, spec.get("regression")) for shift in ("none", "left")]
        if spec.get("regression") == "standardised":
            variants.append((spec["cost_volume_shift"], None))
        for shift, regression in variants:
            model = StereoNet(StereoNetConfig(cost_volume_shift=shift))
            if regression == "standardised":
                from phase2.models import scaled_regression
                scaled_regression.apply_to(model)
            model.load_state_dict(ckpt["model"])
            model.eval().to(device)
            preds, gts = [], []
            for index in scenes:
                scene = core.load_scene(index)
                with torch.no_grad():
                    o = model(
                        torch.from_numpy(normalize(scene.left)).to(device),
                        torch.from_numpy(normalize(scene.right)).to(device),
                    )[0, 0].cpu().numpy()
                valid = scene.gt_valid
                preds.append(o[valid].astype(np.float64))
                gts.append(scene.gt_disparity[valid].astype(np.float64))
            m = disparity_metrics(np.concatenate(preds), np.concatenate(gts))
            label = "{}_weights_shift_{}{}".format(
                key, shift, "" if regression == spec.get("regression")
                else "_no_standardisation")
            rows[label] = m.as_dict()
            print("cross_shift: {} done".format(label))
    return {
        "protocol": (
            "inference-only: each arm's trained weights run under both cost-volume "
            "shift settings, pooled over gt > 0 pixels of hailo_val scenes {}. "
            "Trained-with settings are BASE/none and WORKING/left.".format(scenes)
        ),
        "git_revision": core.git_revision(),
        "results": rows,
    }


# --- 4. gradient norms under the trained weights ------------------------------

def run_gradients(device: str, batches: int = 40, seed: int = 0) -> dict:
    """Per-parameter gradient norms for both arms, on identical inputs.

    Uses the H1 training recipe's crop, loss and batch size, but the *trained*
    weights, and never steps the optimizer -- weights are read, gradients are
    measured, nothing is saved. This says what the loss surface looks like at
    the end of training; it is not a replay of the training-time spike.
    """
    from phase2.scripts.exp_h1_cost_volume import CROP_H, CROP_W  # noqa: E402

    ds_scenes = [core.load_scene(i, split="hailo_calib") for i in range(20)]
    rng = np.random.default_rng(seed)
    crops = []
    for scene in ds_scenes:
        for _ in range(max(1, batches // len(ds_scenes))):
            h, w = scene.shape
            y = int(rng.integers(0, h - CROP_H + 1))
            x = int(rng.integers(0, w - CROP_W + 1))
            crops.append((scene, y, x))
    crops = crops[:batches]

    results: dict = {
        "protocol": (
            "trained EXP-H1-*-v2 weights, {} single-sample 256x512 crops from "
            "hailo_calib (identical crops for both arms), masked smooth L1 as in "
            "the H1 recipe, backward only -- no optimizer step, nothing "
            "saved.".format(len(crops)),
        ),
        "git_revision": core.git_revision(),
        "models": {},
    }
    max_disparity = float(StereoNetConfig().max_disparity_px)

    for key in ACTIVE_MODELS:
        model = load_model(key, device)
        model.train()  # the recipe's mode; there is no BN or dropout in this net
        totals, per_param_max, worst = [], {}, None
        for i, (scene, y, x) in enumerate(crops):
            sl = (slice(y, y + CROP_H), slice(x, x + CROP_W))
            left = torch.from_numpy(normalize(scene.left[sl])).to(device)
            right = torch.from_numpy(normalize(scene.right[sl])).to(device)
            gt = torch.from_numpy(
                scene.gt_disparity[sl][None, None].astype(np.float32)).to(device)
            model.zero_grad(set_to_none=True)
            loss, _ = masked_smooth_l1(model(left, right), gt, max_disparity=max_disparity)
            loss.backward()
            total = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9))
            totals.append(total)
            for name, p in model.named_parameters():
                if p.grad is not None:
                    n = float(p.grad.norm())
                    per_param_max[name] = max(per_param_max.get(name, 0.0), n)
            if worst is None or total > worst["total_norm"]:
                worst = {
                    "batch": i, "scene": scene.name, "crop_yx": [y, x],
                    "total_norm": total, "loss": float(loss),
                    "top_parameters": sorted(
                        ((float(p.grad.norm()), n) for n, p in model.named_parameters()
                         if p.grad is not None), reverse=True)[:5],
                }
        model.zero_grad(set_to_none=True)
        results["models"][key] = {
            "median_total_norm": float(np.median(totals)),
            "max_total_norm": float(np.max(totals)),
            "min_total_norm": float(np.min(totals)),
            "any_non_finite": bool(not np.all(np.isfinite(totals))),
            "worst_batch": worst,
            "top_parameters_by_max_norm": sorted(
                ((v, k) for k, v in per_param_max.items()), reverse=True)[:8],
        }
        print("gradients: {} done".format(key))
    return results


# --- 4b. saturation, and what the gradient spikes coincide with ---------------

def run_saturation(device: str, per_scene: int = 5, scenes: int = 30) -> dict:
    """Does gradient reach the matching path at all, and when it does, how big?

    For each crop: the softmax saturation of the aggregated cost, the gradient
    norm reaching feature extraction + aggregation (the matching path), and the
    total gradient norm. Also repeats the sampling in three column bands so the
    "boundary padding" idea gets a test instead of an assumption.

    Backward passes only; no optimizer step, no weight is ever written.
    """
    from phase2.scripts.exp_h1_cost_volume import CROP_H, CROP_W  # noqa: E402

    loaded = [core.load_scene(i, split="hailo_calib") for i in range(scenes)]
    h, w = loaded[0].shape
    max_disparity = float(StereoNetConfig().max_disparity_px)
    rng = np.random.default_rng(2)
    crops = [(sc, int(rng.integers(0, h - CROP_H + 1)), int(rng.integers(0, w - CROP_W + 1)))
             for sc in loaded for _ in range(per_scene)]
    bands = {
        "x=0_left_edge": lambda: 0,
        "x=max_right_edge": lambda: w - CROP_W,
        "x=random_interior": lambda: int(rng.integers(1, w - CROP_W)),
    }

    def one_pass(model, sc, y, x, want_saturation: bool):
        sl = (slice(y, y + CROP_H), slice(x, x + CROP_W))
        left = torch.from_numpy(normalize(sc.left[sl])).to(device)
        right = torch.from_numpy(normalize(sc.right[sl])).to(device)
        gt = torch.from_numpy(sc.gt_disparity[sl][None, None].astype(np.float32)).to(device)
        model.zero_grad(set_to_none=True)
        pred, st = model(left, right, return_stages=True)
        row = {}
        if want_saturation:
            # The soft-argmin runs on the cost *after* bilinear upsampling to
            # full resolution, so that is the tensor whose saturation decides
            # whether any gradient reaches the matching path. Measuring the
            # 1/16-resolution cost instead would miss the interpolated pixels
            # between argmin regions, which are the only unsaturated ones.
            import torch.nn.functional as F

            low = st["aggregated_cost"]
            up = F.interpolate(low, size=(CROP_H, CROP_W), mode="bilinear",
                               align_corners=True)[0].double()
            for tag, cost in (("low_res", low[0].double()), ("upsampled", up)):
                ordered, _ = torch.sort(cost, dim=0)
                gap = ordered[1] - ordered[0]   # margin between best and runner-up
                row["min_cost_gap_" + tag] = float(gap.min())
                row["median_cost_gap_" + tag] = float(gap.median())
                # fp32 exp() underflows to exactly zero below about -88, so any
                # gap above that makes the soft-argmin's derivative exactly zero
                row["fraction_pixels_gap_below_88_" + tag] = float((gap < 88).double().mean())
        loss, _ = masked_smooth_l1(pred, gt, max_disparity=max_disparity)
        loss.backward()
        row["total_norm"] = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9))
        row["matching_path_grad_norm"] = float(sum(
            p.grad.norm() ** 2 for n, p in model.named_parameters()
            if p.grad is not None and n.startswith(("feature_extractor", "aggregation"))
        ) ** 0.5)
        model.zero_grad(set_to_none=True)
        return row

    out: dict = {
        "protocol": (
            "trained EXP-H1-*-v2 weights, {} single-sample {}x{} crops from "
            "hailo_calib, identical crops for both arms; forward + backward "
            "only, no optimizer step, nothing saved.".format(
                len(crops), CROP_H, CROP_W)
        ),
        "git_revision": core.git_revision(),
        "models": {}, "column_bands": {},
    }
    for key in ACTIVE_MODELS:
        model = load_model(key, device)
        model.train()
        rows = [one_pass(model, sc, y, x, True) for sc, y, x in crops]
        total = np.array([r["total_norm"] for r in rows])
        match = np.array([r["matching_path_grad_norm"] for r in rows])
        spike = total > 1e4
        out["models"][key] = {
            "crops": len(rows),
            "total_norm_median": float(np.median(total)),
            "total_norm_max": float(total.max()),
            "spike_count_above_1e4": int(spike.sum()),
            "matching_path_grad_zero_on": int((match == 0).sum()),
            "matching_path_grad_max": float(match.max()),
            "spikes_with_nonzero_matching_gradient": int((match[spike] > 0).sum()) if spike.any() else 0,
            "median_total_norm_when_matching_gradient_nonzero": (
                float(np.median(total[match > 0])) if (match > 0).any() else None),
            "median_total_norm_when_matching_gradient_zero": (
                float(np.median(total[match == 0])) if (match == 0).any() else None),
            "cost_gap": {
                tag: {
                    "min_over_crops": float(min(r["min_cost_gap_" + tag] for r in rows)),
                    "median_over_crops": float(np.median(
                        [r["median_cost_gap_" + tag] for r in rows])),
                    "mean_fraction_pixels_below_88": float(np.mean(
                        [r["fraction_pixels_gap_below_88_" + tag] for r in rows])),
                    "max_fraction_pixels_below_88": float(max(
                        r["fraction_pixels_gap_below_88_" + tag] for r in rows)),
                }
                for tag in ("low_res", "upsampled")
            },
            "unsaturated_fraction_upsampled": {
                "on_spike_crops": [r["fraction_pixels_gap_below_88_upsampled"]
                                   for r, sp in zip(rows, spike) if sp],
                "median_on_non_spike_crops": float(np.median(
                    [r["fraction_pixels_gap_below_88_upsampled"]
                     for r, sp in zip(rows, spike) if not sp])),
            },
        }
        band_rows = {}
        for name, pick in bands.items():
            band_crops = [(sc, int(rng.integers(0, h - CROP_H + 1)), pick())
                          for sc in loaded for _ in range(3)]
            t = np.array([one_pass(model, sc, y, x, False)["total_norm"]
                          for sc, y, x in band_crops])
            band_rows[name] = {
                "crops": int(t.size), "median": float(np.median(t)),
                "p90": float(np.percentile(t, 90)), "max": float(t.max()),
                "spikes_above_1e4": int((t > 1e4).sum()),
            }
        out["column_bands"][key] = band_rows
        print("saturation: {} done".format(key))
    return out


def run_tie_check(device: str, per_scene: int = 5, scenes: int = 30) -> dict:
    """Where does gradient enter the cost tensor, on the crops that spike?

    Reproduces the network's own regression stage explicitly (upsample, softmax,
    index-weighted sum) so ``d(loss)/d(aggregated cost)`` can be retained and
    counted. Spikes and the count of cost-tensor pixels receiving gradient are
    recorded together; neither is assumed from the other.
    """
    import torch.nn.functional as F

    from phase2.scripts.exp_h1_cost_volume import CROP_H, CROP_W  # noqa: E402

    loaded = [core.load_scene(i, split="hailo_calib") for i in range(scenes)]
    h, w = loaded[0].shape
    max_disparity = float(StereoNetConfig().max_disparity_px)
    rng = np.random.default_rng(2)
    crops = [(sc, int(rng.integers(0, h - CROP_H + 1)), int(rng.integers(0, w - CROP_W + 1)))
             for sc in loaded for _ in range(per_scene)]

    out: dict = {
        "protocol": (
            "trained weights, {} crops of {}x{} from hailo_calib; the regression "
            "stage is re-expressed inline so the gradient w.r.t. the upsampled "
            "aggregated cost can be counted. Backward only, nothing "
            "saved.".format(len(crops), CROP_H, CROP_W)
        ),
        "git_revision": core.git_revision(),
        "models": {},
    }
    for key in ACTIVE_MODELS:
        model = load_model(key, device)
        model.train()
        rows = []
        for sc, y, x in crops:
            sl = (slice(y, y + CROP_H), slice(x, x + CROP_W))
            left = torch.from_numpy(normalize(sc.left[sl])).to(device)
            right = torch.from_numpy(normalize(sc.right[sl])).to(device)
            gt = torch.from_numpy(sc.gt_disparity[sl][None, None].astype(np.float32)).to(device)
            model.zero_grad(set_to_none=True)
            cost = model.aggregation(model.cost_volume(
                model.feature_extractor(left), model.feature_extractor(right)))
            up = F.interpolate(cost, size=(CROP_H, CROP_W), mode="bilinear",
                               align_corners=True)
            up.retain_grad()
            weights = torch.softmax(-up, dim=1)
            index = torch.arange(up.shape[1], device=up.device, dtype=up.dtype).view(1, -1, 1, 1)
            initial = (weights * index).sum(1, keepdim=True)
            pred = torch.relu(initial + model.refinement(initial, left))
            loss, _ = masked_smooth_l1(pred, gt, max_disparity=max_disparity)
            loss.backward()
            ordered, _ = torch.sort(up[0].double(), dim=0)
            gap = ordered[1] - ordered[0]
            rows.append({
                "scene": sc.name, "crop_yx": [y, x],
                "total_norm": float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1e9)),
                "cost_pixels_with_gradient": int((up.grad != 0).any(dim=1).sum()),
                "cost_pixels_total": CROP_H * CROP_W,
                "min_top2_cost_gap": float(gap.min()),
                "exact_ties": int((gap == 0).sum()),
            })
            model.zero_grad(set_to_none=True)
        with_grad = [r for r in rows if r["cost_pixels_with_gradient"] > 0]
        spikes = [r for r in rows if r["total_norm"] > 1e4]
        out["models"][key] = {
            "crops": len(rows),
            "crops_with_gradient_into_cost": len(with_grad),
            "crops_with_spike_above_1e4": len(spikes),
            "spikes_that_also_have_cost_gradient": sum(
                1 for r in spikes if r["cost_pixels_with_gradient"] > 0),
            "detail": with_grad,
        }
        print("tie_check: {} done".format(key))
    return out


# --- 4c. did the matching branch train at all, and can its features match? ----

def run_weight_drift() -> dict:
    """How far each stage moved from the initialisation both arms shared.

    Both H1 arms seed with ``torch.manual_seed(0)`` before constructing the
    model, and the architecture is identical, so the initial weights are
    identical too (asserted here, not assumed). Distance from that
    initialisation says whether a stage was trained -- which is not the same
    question as whether it receives gradient at convergence.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    init = {k: v.clone() for k, v in
            StereoNet(StereoNetConfig(cost_volume_shift="none")).state_dict().items()}
    torch.manual_seed(0)
    np.random.seed(0)
    init_left = StereoNet(StereoNetConfig(cost_volume_shift="left")).state_dict()
    same_init = all(torch.equal(init[k], init_left[k]) for k in init)

    out = {
        "protocol": ("recreate the seed-0 initialisation and measure "
                     "||w - w0|| / ||w0|| per stage for each trained arm"),
        "git_revision": core.git_revision(),
        "both_arms_share_the_initialisation": bool(same_init),
        "models": {},
    }
    for key in ACTIVE_MODELS:
        state = torch.load(core.checkpoint_path(key), map_location="cpu",
                           weights_only=False)["model"]
        groups: dict[str, list[float]] = {}
        for k, v in state.items():
            g = k.split(".")[0]
            acc = groups.setdefault(g, [0.0, 0.0])
            acc[0] += float((v - init[k]).norm()) ** 2
            acc[1] += float(init[k].norm()) ** 2
        out["models"][key] = {g: (d ** 0.5) / (n ** 0.5) for g, (d, n) in groups.items()}
        print("weight_drift: {} done".format(key))
    return out


def run_feature_matching(device: str, scenes: list[int]) -> dict:
    """Can the trained features match at all, without the learned aggregation?

    Builds a plain L1 cost volume over the same 12 shifts directly from each
    arm's trained features, takes its argmin, and correlates that with ground
    truth. This bypasses aggregation and the saturated soft-argmin entirely, so
    it separates "the features cannot match" from "the matching head throws the
    information away".
    """
    import torch.nn.functional as F

    models = {k: load_model(k, device) for k in ACTIVE_MODELS}
    out: dict = {
        "protocol": (
            "inference only; L1 cost volume built directly from the trained "
            "feature maps over the same 12 shifts, argmin -> candidate * 16 px, "
            "nearest-upsampled to full resolution, correlated with ground truth "
            "on gt > 0 pixels"
        ),
        "git_revision": core.git_revision(),
        "models": {k: {} for k in models},
    }
    for index in scenes:
        scene = core.load_scene(index)
        gt = scene.gt_disparity.astype(np.float64)
        valid = scene.gt_valid
        for key, model in models.items():
            with torch.no_grad():
                lf = model.feature_extractor(
                    torch.from_numpy(normalize(scene.left)).to(device))
                rf = model.feature_extractor(
                    torch.from_numpy(normalize(scene.right)).to(device))
                cost = torch.stack([
                    ((F.pad(lf, (0, k))[..., k:] if k else lf) - rf).abs().mean(1)
                    for k in range(12)], dim=1)[0]
                argmin = cost.argmin(0).float()[None, None]
                disparity = F.interpolate(
                    argmin, size=scene.shape, mode="nearest")[0, 0].cpu().numpy() * 16.0
                _, st = model(
                    torch.from_numpy(normalize(scene.left)).to(device),
                    torch.from_numpy(normalize(scene.right)).to(device),
                    return_stages=True)
                net_initial = st["disparity_initial"][0, 0].cpu().numpy()
            out["models"][key][str(index)] = {
                "scene": scene.name,
                "corr_with_gt": corr(disparity[valid], gt[valid]),
                "mean_disparity_px": float(disparity.mean()),
                "fraction_argmin_not_0": float((argmin > 0).float().mean()),
                "corr_with_network_initial": corr(disparity, net_initial),
            }
        print("feature_matching: scene {:>2} {} done".format(index, scene.name))
    return out


# --- 5. figures ---------------------------------------------------------------

FIGURE_DIR = REPO_ROOT / "phase2" / "visualizations" / "h1_mechanism"


def run_figures(device: str, scenes: list[int]) -> dict:
    """Two additive figures per scene: where D1 flips, and what each stage did."""
    from phase2.viz import render

    models = {k: load_model(k, device) for k in ACTIVE_MODELS}
    figure_dir = FIGURE_DIR if ACTIVE_MODELS == ["BASE", "WORKING"] else (
        REPO_ROOT / "phase2" / "visualizations" /
        ("_vs_".join(m.lower() for m in ACTIVE_MODELS)))
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index in scenes:
        scene = core.load_scene(index)
        stages, preds = {}, {}
        for key, model in models.items():
            with torch.no_grad():
                _, st = model(
                    torch.from_numpy(normalize(scene.left)).to(device),
                    torch.from_numpy(normalize(scene.right)).to(device),
                    return_stages=True,
                )
            stages[key] = {k: st[k][0, 0].cpu().numpy().astype(np.float64)
                           for k in ("disparity_initial", "refinement_residual",
                                     "disparity_final")}
            preds[key] = stages[key]["disparity_final"]

        stem = Path(scene.name).stem
        fig = render.figure_flip_map(
            scene, preds[ACTIVE_MODELS[0]], preds[ACTIVE_MODELS[1]],
            labels=(ACTIVE_MODELS[0], ACTIVE_MODELS[1]))
        p = figure_dir / "{}_d1_flips.png".format(stem)
        fig.savefig(p, bbox_inches="tight"); render.plt.close(fig); written.append(str(p))

        fig = render.figure_stage_comparison(scene, stages)
        p = figure_dir / "{}_stages.png".format(stem)
        fig.savefig(p, bbox_inches="tight"); render.plt.close(fig); written.append(str(p))
        print("figures: scene {:>2} {} done".format(index, scene.name))
    return {"git_revision": core.git_revision(), "files": written}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stages", "spatial", "ablation", "intervene",
                                     "pooled", "cross_shift", "gradients", "saturation",
                                     "tie_check", "weight_drift",
                                     "feature_matching", "figures", "all"])
    ap.add_argument("--scenes", nargs="+", type=int, default=None,
                    help="scene indices; default is the four focus scenes for "
                         "'stages' and all 40 for 'spatial'")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--models", nargs="+", default=None,
                    help="arms to compare (default: BASE WORKING, the H1 pair)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the JSON (default: phase2/results/h1_mechanism)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    global ACTIVE_MODELS, OUT_DIR
    if args.models:
        unknown = [m for m in args.models if m not in core.MODELS]
        if unknown:
            raise SystemExit("unknown model(s): " + ", ".join(unknown))
        ACTIVE_MODELS = list(args.models)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_scenes = list(range(len(core.scene_names("hailo_val"))))
    jobs = (["stages", "spatial", "ablation", "intervene", "pooled", "cross_shift", "gradients",
             "saturation", "tie_check", "weight_drift",
             "feature_matching", "figures"]
            if args.what == "all" else [args.what])

    for job in jobs:
        if job == "stages":
            result = run_stages(device, args.scenes or FOCUS_SCENES)
        elif job == "spatial":
            result = run_spatial(device, args.scenes or all_scenes)
        elif job == "ablation":
            result = run_ablation(device, args.scenes or FOCUS_SCENES)
        elif job == "intervene":
            result = run_intervene(device, args.scenes or all_scenes)
        elif job == "pooled":
            result = run_pooled_intervention(device, args.scenes or all_scenes)
        elif job == "cross_shift":
            result = run_cross_shift(device, args.scenes or all_scenes)
        elif job == "saturation":
            result = run_saturation(device)
        elif job == "tie_check":
            result = run_tie_check(device)
        elif job == "weight_drift":
            result = run_weight_drift()
        elif job == "feature_matching":
            result = run_feature_matching(device, args.scenes or FOCUS_SCENES)
        elif job == "figures":
            result = run_figures(device, args.scenes or FOCUS_SCENES)
        else:
            result = run_gradients(device, batches=args.batches)
        path = OUT_DIR / "{}.json".format(job)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("wrote " + str(path))


if __name__ == "__main__":
    main()
