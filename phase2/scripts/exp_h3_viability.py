"""EXP-H3-VIABILITY-001 -- is the disparity shift worth expensive training, now
that the matching pathway actually works?

H2 established a functioning stereo pathway (`cost_volume_shift="left"` plus the
standardised soft-argmin) and is the control here. H3 changes exactly one thing:

    cost_volume_shift: "left"  ->  "none"

Everything else -- architecture, regression stage, feature extractor,
aggregation, refinement, loss, optimizer, schedule, seed, dataset, split, crop,
augmentation, batch size, initialisation, parameter count, precision -- is the
H2 recipe, imported rather than restated.

This is a cheap viability gate, not an accuracy experiment. Three stages, each
of which can stop the next:

    preflight   Part A + B: does the complete LEFT vs NONE pipeline actually
                differ on the existing H2 checkpoint, and does H2 still pass its
                own stereo tests? Writes a gate verdict.
    train       Part C + D: 10 epochs only, refuses to start unless the gate
                passed. The cosine schedule keeps T_max = 200 so these are the
                *same first ten epochs* H2 ran, not a compressed schedule.
    (visuals)   Part E is done with the existing visualizer, not with new code.

    python phase2/scripts/exp_h3_viability.py preflight
    python phase2/scripts/exp_h3_viability.py train

Nothing here writes to any H1 or H2 path. The H2 checkpoint is opened read-only
and its hash is checked before and after every stage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase2.models import scaled_regression  # noqa: E402
from phase2.scripts.exp_h1_cost_volume import (  # noqa: E402
    CROP_H, CROP_W, CroppedKitti, validate,
)
from phase2.scripts.exp_h2_softargmin_scale import (  # noqa: E402
    GRADIENT_PRESENT_THRESHOLD, saturation_snapshot, stage_gradients,
    validation_stage_stats,
)
from phase2.viz import core  # noqa: E402
from src.common.experiment import Experiment  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.evaluation.metrics import disparity_metrics  # noqa: E402
from src.losses.disparity import masked_smooth_l1  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

PHASE2_ROOT = REPO_ROOT / "phase2"
EXPERIMENTS_DIR = PHASE2_ROOT / "experiments"
OUT_DIR = PHASE2_ROOT / "results" / "training"
PREFLIGHT_DIR = PHASE2_ROOT / "results" / "h3_viability"
EXPERIMENT_ID = "EXP-H3-VIABILITY-001"
CONTROL_ID = "EXP-H2-SOFTARGMIN-SCALE"
CONTROL_KEY = "H2"                      # as registered in phase2/viz/core.MODELS
FOCUS_SCENES = [27, 0, 31, 6]

# Two arms differ "materially" at a stage when their outputs differ by more than
# this. fp32 forward passes of the *same* weights on the *same* input are
# bit-identical here (no non-determinism is introduced), so anything above the
# fp32 epsilon is a real difference, and this threshold is far above it.
DIFFERENCE_THRESHOLD = 1e-6
# The right image / matching map must matter by at least this many D1 points for
# the arm to count as stereo-functional. Chosen from the H1 investigation, where
# the broken control moved by at most 0.8 points under the same tests.
STEREO_DEPENDENCE_MIN_D1 = 5.0


def build(shift: str, state: dict, device: str) -> StereoNet:
    """A complete H2-architecture model with the given shift and H2's weights."""
    model = StereoNet(StereoNetConfig(cost_volume_shift=shift))
    scaled_regression.apply_to(model)          # the H2 regression stage
    model.load_state_dict(state)
    return model.eval().to(device)


def full_forward(model: StereoNet, left_img: np.ndarray, right_img: np.ndarray,
                 device: str) -> dict:
    """Run the COMPLETE model and return every stage, including the tensor the
    soft-argmin actually consumes."""
    model.regression.capture = True
    with torch.no_grad():
        _, stages = model(
            torch.from_numpy(normalize(left_img)).to(device),
            torch.from_numpy(normalize(right_img)).to(device),
            return_stages=True,
        )
    captured = dict(model.regression.last)
    model.regression.capture = False
    model.regression.last = {}
    out = {k: stages[k] for k in (
        "left_features", "right_features", "cost_volume", "aggregated_cost",
        "disparity_initial", "refinement_residual", "disparity_final")}
    out["softmax_input"] = captured["softmax_input"]
    return out


def entropy_of(cost: torch.Tensor) -> float:
    weights = torch.softmax(-cost.double(), dim=1)
    return float(-(weights * torch.log(weights.clamp_min(1e-300))).sum(dim=1).mean())


def finite_report(tensor: torch.Tensor) -> dict:
    return {"shape": list(tensor.shape),
            "nan": int(torch.isnan(tensor).sum()),
            "inf": int(torch.isinf(tensor).sum()),
            "max_abs": float(tensor.abs().max())}


def d1_of(pred: np.ndarray, scene: core.Scene) -> dict:
    valid = scene.gt_valid
    m = disparity_metrics(pred[valid], scene.gt_disparity[valid])
    return {"epe": m.epe, "d1": m.d1}


def run_preflight(device: str, scenes: list[int]) -> dict:
    """Part A (complete LEFT vs NONE) and Part B (H2 still passes stereo tests)."""
    checkpoint = core.checkpoint_path(CONTROL_KEY)
    digest_before = core.sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]

    models = {"left": build("left", state, device), "none": build("none", state, device)}

    # the two arms must be the same network with the same weights
    identical = all(
        torch.equal(a, b) for a, b in zip(models["left"].state_dict().values(),
                                          models["none"].state_dict().values()))
    matches_checkpoint = all(
        torch.equal(models["left"].state_dict()[k].cpu(), v) for k, v in state.items())
    config_diff = [
        (models["left"].config.cost_volume_shift, models["none"].config.cost_volume_shift)]

    result: dict = {
        "purpose": ("Part A/B gate: does the complete pipeline distinguish "
                    "cost_volume_shift left vs none on the trained H2 weights, "
                    "and is H2 still stereo-functional"),
        "protocol": {
            "weights": CONTROL_ID + " checkpoint, read-only, both arms",
            "checkpoint_sha256": digest_before,
            "scenes": scenes,
            "difference_threshold": DIFFERENCE_THRESHOLD,
            "stereo_dependence_min_d1_points": STEREO_DEPENDENCE_MIN_D1,
            "gradient_present_threshold": GRADIENT_PRESENT_THRESHOLD,
            "device": device,
        },
        "git_revision": core.git_revision(),
        "identity_checks": {
            "arms_share_every_weight": bool(identical),
            "weights_match_the_checkpoint": bool(matches_checkpoint),
            "cost_volume_shift_left_vs_none": config_diff,
            "parameter_count": models["left"].parameter_count(),
        },
        "stage_differences": {}, "stereo_gate": {}, "gradients": {},
    }

    # --- Part A: stage-by-stage, complete model -------------------------------
    for index in scenes:
        scene = core.load_scene(index)
        out = {k: full_forward(m, scene.left, scene.right, device)
               for k, m in models.items()}
        row = {"name": scene.name, "stages": {}}
        for stage in out["left"]:
            a, b = out["left"][stage], out["none"][stage]
            same_shape = a.shape == b.shape
            diff = float((a - b).abs().max()) if same_shape else float("nan")
            row["stages"][stage] = {
                "shape_match": bool(same_shape),
                "max_abs_difference": diff,
                "differs": bool(same_shape and diff > DIFFERENCE_THRESHOLD),
                "left": finite_report(a), "none": finite_report(b),
            }
        for arm in ("left", "none"):
            pred = out[arm]["disparity_final"][0, 0].cpu().numpy().astype(np.float64)
            row.setdefault("metrics", {})[arm] = d1_of(pred, scene)
            row.setdefault("softmax_entropy_nats", {})[arm] = entropy_of(
                out[arm]["softmax_input"])
            initial = out[arm]["disparity_initial"][0, 0].cpu().numpy().astype(np.float64)
            valid = scene.gt_valid
            row.setdefault("disparity_initial", {})[arm] = {
                "std": float(initial.std()), "min": float(initial.min()),
                "max": float(initial.max()),
                "corr_with_gt": (float(np.corrcoef(
                    initial[valid], scene.gt_disparity[valid])[0, 1])
                    if initial[valid].std() > 0 else float("nan")),
            }
        result["stage_differences"][str(index)] = row
        print("preflight: stages, scene {:>2} {} done".format(index, scene.name))

    # --- Part B: does H2 still depend on the right image and on its own map? --
    rng = np.random.default_rng(0)
    for arm, model in models.items():
        rows = []
        for index in scenes:
            scene = core.load_scene(index)
            base = full_forward(model, scene.left, scene.right, device)
            base_pred = base["disparity_final"][0, 0].cpu().numpy().astype(np.float64)
            entry = {"scene": scene.name, "baseline": d1_of(base_pred, scene)}

            # right-image dependence: black, noise, and the left image itself
            for name, right in (
                ("right_black", np.zeros_like(scene.right)),
                ("right_noise", rng.integers(0, 256, size=scene.right.shape,
                                             dtype=np.uint8)),
                ("right_equals_left", scene.left.copy()),
            ):
                pred = full_forward(model, scene.left, right, device)[
                    "disparity_final"][0, 0].cpu().numpy().astype(np.float64)
                entry[name] = d1_of(pred, scene)
                entry[name]["d1_penalty"] = entry[name]["d1"] - entry["baseline"]["d1"]
                entry[name]["mean_abs_change_px"] = float(np.abs(pred - base_pred).mean())

            # matching-map dependence: substitute disparity_initial only
            left_t = torch.from_numpy(normalize(scene.left)).to(device)
            initial = base["disparity_initial"]
            flat = initial.flatten()
            perm = torch.from_numpy(rng.permutation(flat.numel())).to(flat.device)
            for name, replacement in (
                ("initial_constant_mean", torch.full_like(initial, float(initial.mean()))),
                ("initial_shuffled", flat[perm].view_as(initial)),
            ):
                with torch.no_grad():
                    pred_t = replacement + model.refinement(replacement, left_t)
                    if model.config.final_relu:
                        pred_t = torch.relu(pred_t)
                pred = pred_t[0, 0].cpu().numpy().astype(np.float64)
                entry[name] = d1_of(pred, scene)
                entry[name]["d1_penalty"] = entry[name]["d1"] - entry["baseline"]["d1"]
                entry[name]["mean_abs_change_px"] = float(np.abs(pred - base_pred).mean())
            rows.append(entry)
        result["stereo_gate"][arm] = rows
        print("preflight: stereo gate, arm '{}' done".format(arm))

    # --- Part B: matching-path gradient, both arms ----------------------------
    calib = [core.load_scene(i, split="hailo_calib") for i in range(12)]
    h, w = calib[0].shape
    grng = np.random.default_rng(0)
    crops = [(sc, int(grng.integers(0, h - CROP_H + 1)), int(grng.integers(0, w - CROP_W + 1)))
             for sc in calib]
    max_disparity = float(models["left"].config.max_disparity_px)
    for arm, model in models.items():
        model.train()
        rows = []
        for sc, y, x in crops:
            sl = (slice(y, y + CROP_H), slice(x, x + CROP_W))
            model.zero_grad(set_to_none=True)
            model.regression.capture = True
            pred = model(torch.from_numpy(normalize(sc.left[sl])).to(device),
                         torch.from_numpy(normalize(sc.right[sl])).to(device))
            gt = torch.from_numpy(
                sc.gt_disparity[sl][None, None].astype(np.float32)).to(device)
            loss, _ = masked_smooth_l1(pred, gt, max_disparity=max_disparity)
            loss.backward()
            grads = stage_gradients(model)
            snap = saturation_snapshot(model)
            model.regression.capture = False
            model.regression.last = {}
            model.zero_grad(set_to_none=True)
            rows.append({"scene": sc.name, "crop_yx": [y, x], "loss": float(loss),
                         **{k: v for k, v in grads.items()},
                         "softmax_entropy": snap.get("mean_softmax_entropy_nats"),
                         "softmax_input_grad_norm": snap.get("softmax_input_grad_norm")})
        model.eval()
        present = [r["matching_present"] for r in rows]
        norms = np.array([r["matching_path"] for r in rows])
        result["gradients"][arm] = {
            "crops": len(rows),
            "matching_present_fraction": float(np.mean(present)),
            "matching_norm_median": float(np.median(norms)),
            "matching_norm_max": float(norms.max()),
            "mean_softmax_entropy": float(np.mean(
                [r["softmax_entropy"] for r in rows if r["softmax_entropy"] is not None])),
            "any_non_finite_loss": bool(not np.all(np.isfinite([r["loss"] for r in rows]))),
            "batches": rows,
        }
        print("preflight: gradients, arm '{}' done".format(arm))

    # --- verdict ---------------------------------------------------------------
    digest_after = core.sha256(checkpoint)
    stages_that_differ = {
        stage: all(row["stages"][stage]["differs"]
                   for row in result["stage_differences"].values())
        for stage in result["stage_differences"][str(scenes[0])]["stages"]
    }
    finite = all(
        s["left"]["nan"] == 0 and s["left"]["inf"] == 0
        and s["none"]["nan"] == 0 and s["none"]["inf"] == 0
        for row in result["stage_differences"].values() for s in row["stages"].values())
    shapes_ok = all(
        s["shape_match"] for row in result["stage_differences"].values()
        for s in row["stages"].values())

    def worst_penalty(arm: str, keys) -> float:
        return min(min(row[k]["d1_penalty"] for k in keys)
                   for row in result["stereo_gate"][arm])

    right_dependence = worst_penalty(
        "left", ("right_black", "right_noise", "right_equals_left"))
    map_dependence = worst_penalty("left", ("initial_constant_mean", "initial_shuffled"))

    checks = {
        "checkpoint_unchanged": digest_before == digest_after,
        "arms_share_every_weight": bool(identical),
        "weights_match_the_checkpoint": bool(matches_checkpoint),
        "no_nan_or_inf": bool(finite),
        "no_shape_mismatch": bool(shapes_ok),
        "final_disparity_differs": bool(stages_that_differ["disparity_final"]),
        "cost_volume_differs": bool(stages_that_differ["cost_volume"]),
        "features_identical_as_expected": not stages_that_differ["left_features"],
        "h2_right_image_dependence_d1_points": right_dependence,
        "h2_matching_map_dependence_d1_points": map_dependence,
        "h2_right_image_gate": right_dependence >= STEREO_DEPENDENCE_MIN_D1,
        "h2_matching_map_gate": map_dependence >= STEREO_DEPENDENCE_MIN_D1,
        "h2_matching_gradient_gate": result["gradients"]["left"][
            "matching_present_fraction"] == 1.0,
    }
    checks["PASS"] = all(v for k, v in checks.items()
                         if isinstance(v, bool) and k != "features_identical_as_expected")
    result["stage_summary"] = stages_that_differ
    result["gate"] = checks
    result["checkpoint_sha256_after"] = digest_after
    return result


def gate_passed() -> tuple[bool, dict]:
    path = PREFLIGHT_DIR / "preflight.json"
    if not path.exists():
        return False, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return bool(data.get("gate", {}).get("PASS")), data


def run_training(device: str, epochs: int, schedule_epochs: int, seed: int) -> None:
    """Part C/D: the same first `epochs` epochs H2 ran, with the shift removed."""
    ok, preflight = gate_passed()
    if not ok:
        raise SystemExit(
            "preflight gate has not passed -- run 'preflight' first and read "
            "its verdict. Training is deliberately blocked.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    train = CroppedKitti(train_base, seed=seed)
    loader = DataLoader(train, batch_size=2, shuffle=True, num_workers=0)

    config = StereoNetConfig(cost_volume_shift="none")     # <- the one change
    model = StereoNet(config).to(device)
    scaled_regression.apply_to(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    # T_max stays at the control's 200 so these are the control's first epochs,
    # not a compressed schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_epochs)
    max_disparity = float(config.max_disparity_px)

    exp_config = {
        "hypothesis": (
            "H3: with the matching pathway functional (H2's standardised "
            "soft-argmin), how much does the disparity shift itself contribute? "
            "This run is a cheap viability gate, not the answer."),
        "control": CONTROL_ID + " (trained; its first {} epochs are the "
                   "comparison points, not retrained here)".format(epochs),
        "control_checkpoint_sha256": preflight["protocol"]["checkpoint_sha256"],
        "changed_variable": "cost_volume_shift: 'left' -> 'none'",
        "held_fixed": (
            "standardised soft-argmin regression stage, feature extractor, "
            "aggregation, refinement, loss, optimizer, learning rate, cosine "
            "schedule with T_max={}, seed, dataset, split, crop, augmentation, "
            "batch size, initialisation, parameter count, precision -- the H2 "
            "recipe imported from phase2/scripts/exp_h2_softargmin_scale.py and "
            "phase2/scripts/exp_h1_cost_volume.py".format(schedule_epochs)),
        "budget": "{} epochs of the control's {}-epoch schedule (viability gate; "
                  "NOT a full run)".format(epochs, schedule_epochs),
        "dataset": "kitti2015",
        "split": "hailo_calib (scenes 0-159) train, hailo_val (160-199) validate",
        "resolution": [CROP_H, CROP_W],
        "batch_size": 2, "precision": "fp32", "seed": seed,
        "epochs": epochs, "schedule_T_max": schedule_epochs,
        "optimizer": "Adam betas=(0.9, 0.999)", "learning_rate": 1e-3,
        "loss": "masked smooth L1, beta=1.0, valid = gt > 0 and gt < max_disparity",
        "validation_protocol": (
            "first 10 scenes of hailo_val, full 368x1232 frames, pooled over "
            "gt > 0 (exp_h1_cost_volume.validate, imported); evaluated at "
            "epochs 0, 1, 2, 5, 10 for this gate"),
        "validation_loss": "same masked smooth L1, on the same 10 validation scenes",
        "preflight": "phase2/results/h3_viability/preflight.json",
        "gradient_present_threshold": GRADIENT_PRESENT_THRESHOLD,
        "device": str(device),
        "known_limitation": (
            "10 epochs, one seed, 160 training scenes. This can show that H3 is "
            "trainable and roughly where it sits early; it cannot say what H3 "
            "converges to."),
    }

    if (EXPERIMENTS_DIR / EXPERIMENT_ID).exists():
        raise SystemExit(EXPERIMENT_ID + " already exists -- records are never "
                         "overwritten.")

    checkpoint_epochs = {0, 1, 2, 5, epochs}
    with Experiment("H3 viability: cost_volume_shift='none' under H2's working "
                    "soft-argmin", config=exp_config, experiment_id=EXPERIMENT_ID,
                    experiments_dir=EXPERIMENTS_DIR) as exp:
        exp.note(exp_config["hypothesis"])
        exp.note(exp_config["known_limitation"])
        exp.note("Early-exit gates are active: the run stops on any non-finite "
                 "loss or gradient.")

        history, grad_norms, matching_norms = [], [], []
        matching_present = 0
        batches_total = 0
        t0 = time.time()
        aborted = None

        for epoch in range(epochs + 1):
            if epoch in checkpoint_epochs:
                metrics = validate(model, val_base, device, limit=10)
                stats = validation_stage_stats(model, val_base, device)
                val_loss = _validation_loss(model, val_base, device, max_disparity)
                row = {"epoch": epoch, "val_epe": metrics.epe, "val_d1": metrics.d1,
                       "val_loss": val_loss, "stage_stats": stats,
                       "train_loss_of_previous_epoch": history[-1]["mean_loss"]
                       if history else None}
                if grad_norms:
                    row["grad"] = {
                        "median_total": float(np.median(grad_norms)),
                        "max_total": float(np.max(grad_norms)),
                        "median_matching": float(np.median(matching_norms)),
                        "matching_present_fraction": matching_present / max(batches_total, 1),
                    }
                exp.metric("checkpoint_epoch_{}".format(epoch), row)
                exp.log("checkpoint epoch {:>3}  val EPE {:.3f}  D1 {:.2f}%  "
                        "val loss {:.4f}  entropy {:.3f}  init std {:.3f}  "
                        "init r(GT) {:+.3f}".format(
                            epoch, metrics.epe, metrics.d1, val_loss,
                            stats["entropy"], stats["initial_std"],
                            stats["initial_corr_gt"]))
            if epoch == epochs:
                break

            epoch_loss, batches = 0.0, 0
            for left, right, disparity in loader:
                left, right = left.to(device), right.to(device)
                disparity = disparity.to(device)
                out = model(left, right)
                loss, _ = masked_smooth_l1(out, disparity, max_disparity=max_disparity)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9))
                stages = stage_gradients(model)
                optimizer.step()

                if not np.isfinite(float(loss)) or not np.isfinite(gn):
                    aborted = ("non-finite loss or gradient at epoch {} batch {}: "
                               "loss={} grad={}".format(epoch, batches_total,
                                                        float(loss), gn))
                    break
                epoch_loss += float(loss)
                batches += 1
                batches_total += 1
                grad_norms.append(gn)
                matching_norms.append(stages["matching_path"])
                matching_present += int(stages["matching_present"])
            if aborted:
                exp.note("ABORTED: " + aborted)
                break

            scheduler.step()
            history.append({
                "epoch": epoch, "mean_loss": epoch_loss / max(batches, 1),
                "lr": float(scheduler.get_last_lr()[0]),
                "median_grad_norm": float(np.median(grad_norms[-batches:])),
                "median_matching_grad_norm": float(np.median(matching_norms[-batches:])),
                "matching_present_fraction_this_epoch": float(np.mean(
                    [m > GRADIENT_PRESENT_THRESHOLD for m in matching_norms[-batches:]])),
            })
            exp.log("epoch {:>3}  loss {:8.4f}  grad {:9.3f}  match {:9.3e} "
                    "({:.0%} of batches)  lr {:.2e}".format(
                        epoch, history[-1]["mean_loss"], history[-1]["median_grad_norm"],
                        history[-1]["median_matching_grad_norm"],
                        history[-1]["matching_present_fraction_this_epoch"],
                        history[-1]["lr"]))

        totals = np.array(grad_norms) if grad_norms else np.array([np.nan])
        matching = np.array(matching_norms) if matching_norms else np.array([np.nan])
        exp.metric("history", history)
        exp.metric("aborted", aborted)
        exp.metric("wall_clock_s", time.time() - t0)
        exp.metric("primary_endpoint_matching_gradient", {
            "definition": "batches where ||grad(feature_extractor)|| + "
                          "||grad(aggregation)|| exceeds {}".format(
                              GRADIENT_PRESENT_THRESHOLD),
            "batches": batches_total,
            "present_fraction": matching_present / max(batches_total, 1),
            "norm": {"median": float(np.median(matching)), "max": float(matching.max()),
                     "p90": float(np.percentile(matching, 90))},
        })
        exp.metric("gradient_norms", {
            "median": float(np.median(totals)), "max": float(totals.max()),
            "p90": float(np.percentile(totals, 90)),
            "any_nan": bool(np.any(~np.isfinite(totals))),
            "batches_above_1e4": int((totals > 1e4).sum()),
        })
        exp.metric("parameter_count_unique", model.parameter_count())

        ckpt = OUT_DIR / (EXPERIMENT_ID + "_checkpoint.pth")
        torch.save({"model": model.state_dict(), "config": exp_config}, ckpt)
        exp.metric("checkpoint", str(ckpt.relative_to(REPO_ROOT)))
        exp.metric("checkpoint_sha256", core.sha256(ckpt))
        (OUT_DIR / (EXPERIMENT_ID + "_history.json")).write_text(
            json.dumps(history, indent=2), encoding="utf-8")

        exp.conclude(
            "Viability gate only ({} epochs of a {}-epoch schedule). Matching-path "
            "gradient present on {:.1%} of {} batches; total gradient max {:.4g}; "
            "{}. Read the checkpoint progression before any conclusion about "
            "accuracy.".format(
                epochs, schedule_epochs, matching_present / max(batches_total, 1),
                batches_total, float(totals.max()),
                "no abort" if aborted is None else "ABORTED: " + aborted))
        print("\nrecorded as " + exp.id)


def _validation_loss(model: StereoNet, dataset, device: str,
                     max_disparity: float, limit: int = 10) -> float:
    """The training loss, evaluated on the validation scenes."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(min(limit, len(dataset))):
            s = dataset[i]
            pred = model(torch.from_numpy(normalize(s.left)).to(device),
                         torch.from_numpy(normalize(s.right)).to(device))
            gt = torch.from_numpy(s.disparity[None, None].astype(np.float32)).to(device)
            loss, _ = masked_smooth_l1(pred, gt, max_disparity=max_disparity)
            losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preflight", "train"])
    ap.add_argument("--scenes", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--schedule-epochs", type=int, default=200,
                    help="cosine T_max; keep at the control's 200 so this is the "
                         "control's first epochs rather than a compressed schedule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.stage == "preflight":
        PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
        result = run_preflight(device, args.scenes or FOCUS_SCENES)
        path = PREFLIGHT_DIR / "preflight.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("\nwrote " + str(path))
        print("\nGATE: " + ("PASS" if result["gate"]["PASS"] else "FAIL"))
        for key, value in result["gate"].items():
            if key != "PASS":
                print("  {:<44} {}".format(key, value))
        if not result["gate"]["PASS"]:
            raise SystemExit(1)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_training(device, args.epochs, args.schedule_epochs, args.seed)


if __name__ == "__main__":
    main()
