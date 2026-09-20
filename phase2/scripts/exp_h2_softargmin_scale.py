"""EXP-H2-SOFTARGMIN-SCALE -- can the matching path learn once the soft-argmin
is no longer numerically saturated?

Control: `EXP-H1-WORKING-v2` (already trained; NOT retrained here).
Changed variable, exactly one: the aggregated cost is standardised per pixel
across the 12 disparity candidates immediately before the soft-argmin
(`phase2/models/scaled_regression.py`). No new parameter, no temperature, no
clipping, no annealing.

Held fixed, byte-for-byte from the H1-v2 recipe by importing it rather than
re-typing it: dataset, split, crop size and sampling, augmentation, seed,
optimizer, learning rate, schedule, batch size, epoch count, loss, maximum
disparity, precision, validation protocol, and `cost_volume_shift="left"`.

Primary endpoint is mechanistic, not accuracy: the fraction of training batches
on which the matching path (feature extractor + aggregation) receives a
gradient whose norm exceeds a documented threshold. Accuracy is secondary and
is not used to steer anything.

    python phase2/scripts/exp_h2_softargmin_scale.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase2.models import scaled_regression  # noqa: E402
from phase2.scripts.exp_h1_cost_volume import (  # noqa: E402  (the H1 recipe, reused)
    CROP_H, CROP_W, CroppedKitti, validate,
)
from src.common.conclusions import (  # noqa: E402
    classify_series_trend, classify_validation_change, describe_training_outcome,
    describe_validation,
)
from src.common.experiment import Experiment  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.losses.disparity import masked_smooth_l1  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

PHASE2_ROOT = REPO_ROOT / "phase2"
EXPERIMENTS_DIR = PHASE2_ROOT / "experiments"
OUT_DIR = PHASE2_ROOT / "results" / "training"
EXPERIMENT_ID = "EXP-H2-SOFTARGMIN-SCALE"
CONTROL_ID = "EXP-H1-WORKING-v2"

# Identical to the pre-check, and documented there: fp32's smallest normal is
# ~1.2e-38, so 1e-12 is far above the noise floor and far below anything that
# could move an Adam step. Exact zeros are counted separately.
GRADIENT_PRESENT_THRESHOLD = 1e-12
SATURATION_GAP = 88.0
# Saturation statistics need the tensors around the softmax kept alive, so they
# are measured on a periodic subset of batches rather than all of them.
DIAGNOSTIC_EVERY = 20


def stage_gradients(model: StereoNet) -> dict:
    groups = {"feature_extractor": 0.0, "aggregation": 0.0, "refinement": 0.0}
    exact_zero = 0
    for name, p in model.named_parameters():
        stage = name.split(".")[0]
        if stage in groups and p.grad is not None:
            n = float(p.grad.norm())
            groups[stage] += n ** 2
            exact_zero += int(n == 0.0)
    norms = {k: v ** 0.5 for k, v in groups.items()}
    matching = (norms["feature_extractor"] ** 2 + norms["aggregation"] ** 2) ** 0.5
    return {
        "feature_extractor": norms["feature_extractor"],
        "aggregation": norms["aggregation"],
        "refinement": norms["refinement"],
        "matching_path": matching,
        "matching_present": matching > GRADIENT_PRESENT_THRESHOLD,
        "matching_exactly_zero": matching == 0.0,
        "tensors_with_exactly_zero_gradient": exact_zero,
    }


def saturation_snapshot(model: StereoNet) -> dict:
    """Softmax saturation and cost-tensor gradient for the last captured batch."""
    captured = model.regression.last
    if not captured:
        return {}
    scaled = captured["softmax_input"]
    c = scaled.detach().double()
    ordered, _ = torch.sort(c, dim=1)
    gap = ordered[:, 1] - ordered[:, 0]
    weights = torch.softmax(-c, dim=1)
    entropy = -(weights * torch.log(weights.clamp_min(1e-300))).sum(dim=1)
    raw = captured["raw_cost"].detach().double()
    out = {
        "softmax_input_median_top2_gap": float(gap.median()),
        "softmax_input_min_top2_gap": float(gap.min()),
        "fraction_pixels_gap_below_88": float((gap < SATURATION_GAP).double().mean()),
        "exact_ties": int((gap == 0).sum()),
        "mean_softmax_entropy_nats": float(entropy.mean()),
        "mean_max_softmax_weight": float(weights.max(dim=1).values.mean()),
        "raw_cost_max_abs": float(raw.abs().max()),
        "raw_cost_std": float(raw.std()),
    }
    grad = scaled.grad
    if grad is not None:
        out["softmax_input_grad_norm"] = float(grad.norm())
        out["softmax_input_pixels_with_gradient"] = int((grad != 0).any(dim=1).sum())
        out["softmax_input_pixels_total"] = int(
            grad.shape[0] * grad.shape[2] * grad.shape[3])
    raw_grad = captured["raw_cost"].grad
    if raw_grad is not None:
        out["raw_cost_grad_norm"] = float(raw_grad.norm())
    return out


def validation_stage_stats(model: StereoNet, dataset, device, limit: int = 10) -> dict:
    """Stage statistics on the same scenes the recorded protocol validates on."""
    model.eval()
    model.regression.capture = True
    rows = []
    with torch.no_grad():
        for i in range(min(limit, len(dataset))):
            s = dataset[i]
            _, st = model(
                torch.from_numpy(normalize(s.left)).to(device),
                torch.from_numpy(normalize(s.right)).to(device),
                return_stages=True,
            )
            scaled = model.regression.last["softmax_input"].double()
            ordered, _ = torch.sort(scaled, dim=1)
            gap = ordered[:, 1] - ordered[:, 0]
            weights = torch.softmax(-scaled, dim=1)
            entropy = -(weights * torch.log(weights.clamp_min(1e-300))).sum(dim=1)
            initial = st["disparity_initial"][0, 0].cpu().numpy().astype(np.float64)
            residual = st["refinement_residual"][0, 0].cpu().numpy().astype(np.float64)
            final = st["disparity_final"][0, 0].cpu().numpy().astype(np.float64)
            valid = s.disparity > 0
            gt = s.disparity.astype(np.float64)
            rows.append({
                "entropy": float(entropy.mean()),
                "max_weight": float(weights.max(dim=1).values.mean()),
                "median_top2_gap": float(gap.median()),
                "initial_std": float(initial.std()),
                "initial_mean": float(initial.mean()),
                "initial_corr_gt": (
                    float(np.corrcoef(initial[valid], gt[valid])[0, 1])
                    if initial[valid].std() > 0 else float("nan")),
                "residual_mean_abs": float(np.abs(residual).mean()),
                "residual_corr_final": float(np.corrcoef(residual.ravel(),
                                                          final.ravel())[0, 1]),
                "final_std": float(final.std()),
                "final_corr_gt": float(np.corrcoef(final[valid], gt[valid])[0, 1]),
            })
    model.regression.capture = False
    model.regression.last = {}
    model.train()
    keys = rows[0]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    train = CroppedKitti(train_base, seed=args.seed)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0)

    config = StereoNetConfig(cost_volume_shift="left")
    model = StereoNet(config).to(device)
    scaled_regression.apply_to(model)          # <- the one changed variable
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    exp_config = {
        "hypothesis": (
            "H2: the matching pathway is untrainable because the soft-argmin is "
            "numerically saturated. Standardising the aggregated cost across the "
            "disparity axis immediately before the soft-argmin, and changing "
            "nothing else, should restore gradient flow to the feature extractor "
            "and aggregation."
        ),
        "control": CONTROL_ID + " (already trained; not retrained for this experiment)",
        "changed_variable": (
            "regression stage: DisparityRegression -> "
            "phase2.models.scaled_regression.StandardisedDisparityRegression "
            "(per-pixel z-score across the 12 candidates, eps=1e-5, applied to "
            "the upsampled cost the softmax consumes)"
        ),
        "held_fixed": (
            "cost_volume_shift='left', feature extractor, cost volume, "
            "aggregation, refinement, loss, optimizer, schedule, seed, data "
            "split, crop, augmentation, epochs, batch size, precision -- the H1 "
            "recipe imported from phase2/scripts/exp_h1_cost_volume.py"
        ),
        "new_parameters": 0,
        "dataset": "kitti2015",
        "split": "hailo_calib (scenes 0-159) train, hailo_val (160-199) validate",
        "resolution": [CROP_H, CROP_W],
        "crop": "random {}x{}".format(CROP_H, CROP_W),
        "disparity_range": config.max_disparity_px,
        "batch_size": args.batch,
        "precision": "fp32",
        "seed": args.seed,
        "epochs": args.epochs,
        "optimizer": "Adam betas=(0.9, 0.999)",
        "learning_rate": args.lr,
        "schedule": "cosine annealing to 0 over {} epochs".format(args.epochs),
        "loss": "masked smooth L1, beta=1.0, valid = gt > 0 and gt < max_disparity",
        "augmentation": "random crop, independent per-image gain jitter sigma=0.1; "
                        "no horizontal flip (it inverts disparity)",
        "initialisation": "PyTorch defaults, random, seed 0 -- identical to the "
                          "H1 arms (the changed module holds no parameters)",
        "validation_protocol": "first 10 scenes of hailo_val, full 368x1232 "
                               "frames, pooled over gt > 0, every 5 epochs "
                               "(exp_h1_cost_volume.validate, imported)",
        "primary_endpoint": (
            "fraction of training batches on which ||grad(feature_extractor)|| "
            "+ ||grad(aggregation)|| (2-norm combined) exceeds {}".format(
                GRADIENT_PRESENT_THRESHOLD)
        ),
        "gradient_present_threshold": GRADIENT_PRESENT_THRESHOLD,
        "numerical_precheck": "phase2/results/h2_precheck/precheck.json",
        "device": str(device),
        "known_limitation": (
            "one seed, 160 training scenes from random initialisation -- the "
            "same budget as both H1 arms. The comparison against "
            + CONTROL_ID + " is a controlled A/B of the changed variable; "
            "neither arm's absolute accuracy is a ceiling."
        ),
    }

    exp_id = EXPERIMENT_ID + args.suffix
    if (EXPERIMENTS_DIR / exp_id).exists():
        raise SystemExit(
            "{} already exists -- records are never overwritten. Pass a new "
            "--suffix.".format(exp_id))

    with Experiment(
        "H2: standardise the aggregated cost before the soft-argmin",
        config=exp_config, experiment_id=exp_id, experiments_dir=EXPERIMENTS_DIR,
    ) as exp:
        exp.note(exp_config["hypothesis"])
        exp.note(exp_config["known_limitation"])
        exp.note("Primary endpoint is mechanistic (gradient reachability); EPE "
                 "and D1 are secondary and are not used to steer the run.")

        history, grad_norms, matching_norms = [], [], []
        matching_present, matching_exact_zero = 0, 0
        batches_total = 0
        diagnostics = []
        t0 = time.time()

        for epoch in range(args.epochs):
            epoch_loss, epoch_pixels, batches = 0.0, 0, 0
            for left, right, disparity in loader:
                left = left.to(device)
                right = right.to(device)
                disparity = disparity.to(device)

                measure = (batches_total % DIAGNOSTIC_EVERY == 0)
                model.regression.capture = measure

                out = model(left, right)
                loss, n = masked_smooth_l1(
                    out, disparity, max_disparity=float(config.max_disparity_px))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9))
                stages = stage_gradients(model)
                if measure:
                    snap = saturation_snapshot(model)
                    snap.update({"epoch": epoch, "batch": batches_total,
                                 "total_norm": gn,
                                 "matching_path_norm": stages["matching_path"]})
                    diagnostics.append(snap)
                model.regression.capture = False
                model.regression.last = {}
                optimizer.step()

                epoch_loss += float(loss)
                epoch_pixels += n
                batches += 1
                batches_total += 1
                grad_norms.append(gn)
                matching_norms.append(stages["matching_path"])
                matching_present += int(stages["matching_present"])
                matching_exact_zero += int(stages["matching_exactly_zero"])

            scheduler.step()
            mean_loss = epoch_loss / max(batches, 1)
            row = {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "valid_pixels": epoch_pixels,
                "lr": float(scheduler.get_last_lr()[0]),
                "median_grad_norm": float(np.median(grad_norms[-batches:])),
                "median_matching_grad_norm": float(np.median(matching_norms[-batches:])),
                "matching_present_fraction_this_epoch": float(np.mean(
                    [m > GRADIENT_PRESENT_THRESHOLD for m in matching_norms[-batches:]])),
            }
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                m = validate(model, val_base, device, limit=10)
                row["val_epe"] = m.epe
                row["val_d1"] = m.d1
                row["stage_stats"] = validation_stage_stats(model, val_base, device)
            history.append(row)
            exp.log(
                "epoch {:>3}  loss {:8.4f}  grad {:9.3f}  match {:9.3e} "
                "({:.0%} of batches)  lr {:.2e}{}".format(
                    epoch, mean_loss, row["median_grad_norm"],
                    row["median_matching_grad_norm"],
                    row["matching_present_fraction_this_epoch"], row["lr"],
                    "  val EPE {:.3f}  D1 {:.2f}%".format(row["val_epe"], row["val_d1"])
                    if "val_epe" in row else "",
                )
            )

        exp.metric("history", history)
        exp.metric("wall_clock_s", time.time() - t0)
        losses = [r["mean_loss"] for r in history]
        exp.metric("first_epoch_loss", losses[0])
        exp.metric("final_epoch_loss", losses[-1])
        exp.metric("loss_reduction_factor", losses[0] / max(losses[-1], 1e-9))
        exp.metric("min_loss", min(losses))

        matching = np.array(matching_norms)
        totals = np.array(grad_norms)
        exp.metric("primary_endpoint_matching_gradient", {
            "definition": exp_config["primary_endpoint"],
            "batches": batches_total,
            "present_fraction": matching_present / max(batches_total, 1),
            "exactly_zero_fraction": matching_exact_zero / max(batches_total, 1),
            "norm": {
                "median": float(np.median(matching)), "mean": float(matching.mean()),
                "p90": float(np.percentile(matching, 90)),
                "p95": float(np.percentile(matching, 95)),
                "p99": float(np.percentile(matching, 99)),
                "max": float(matching.max()),
            },
            "batches_above": {
                "1e4": int((matching > 1e4).sum()), "1e6": int((matching > 1e6).sum()),
                "1e8": int((matching > 1e8).sum()), "1e10": int((matching > 1e10).sum()),
            },
        })
        exp.metric("gradient_norms", {
            "median": float(np.median(totals)), "mean": float(totals.mean()),
            "p90": float(np.percentile(totals, 90)),
            "p95": float(np.percentile(totals, 95)),
            "p99": float(np.percentile(totals, 99)),
            "max": float(totals.max()),
            "any_nan": bool(np.any(~np.isfinite(totals))),
            "batches_above": {
                "1e4": int((totals > 1e4).sum()), "1e6": int((totals > 1e6).sum()),
                "1e8": int((totals > 1e8).sum()), "1e10": int((totals > 1e10).sum()),
            },
        })
        exp.metric("saturation_diagnostics", diagnostics)
        exp.metric("parameter_count_unique", model.parameter_count())

        ckpt = OUT_DIR / (exp_id + "_checkpoint.pth")
        torch.save({"model": model.state_dict(), "config": exp_config}, ckpt)
        exp.metric("checkpoint", str(ckpt.relative_to(REPO_ROOT)))
        (OUT_DIR / (exp_id + "_history.json")).write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        (OUT_DIR / (exp_id + "_diagnostics.json")).write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8")

        val_rows = [r for r in history if "val_epe" in r]
        val_initial = val_rows[0]["val_epe"] if val_rows else None
        val_final = val_rows[-1]["val_epe"] if val_rows else None
        val_best_row = min(val_rows, key=lambda r: r["val_epe"]) if val_rows else None
        finite = [g for g in grad_norms if math.isfinite(g)]

        exp.metric("validation_epe_series",
                   [{"epoch": r["epoch"], "val_epe": r["val_epe"], "val_d1": r["val_d1"]}
                    for r in val_rows])
        if val_rows:
            exp.metric("validation_epe_initial", val_initial)
            exp.metric("validation_epe_final", val_final)
            exp.metric("validation_epe_best", val_best_row["val_epe"])
            exp.metric("validation_epe_best_epoch", val_best_row["epoch"])
            exp.metric("validation_change",
                       classify_validation_change(val_initial, val_final))
        exp.metric("loss_change",
                   classify_series_trend(losses) if len(losses) >= 2 else None)
        exp.metric("non_finite_gradient_count", len(grad_norms) - len(finite))

        exp.note(
            "Matching-path gradient exceeded {} on {}/{} batches ({:.1%}); it was "
            "exactly zero on {} batches. Median matching-path norm {:.4g}, max "
            "{:.4g}.".format(
                GRADIENT_PRESENT_THRESHOLD, matching_present, batches_total,
                matching_present / max(batches_total, 1), matching_exact_zero,
                float(np.median(matching)), float(matching.max())))
        if val_rows:
            exp.note(describe_validation(
                val_initial, val_final, best=val_best_row["val_epe"],
                best_label="epoch {}".format(val_best_row["epoch"])))

        exp.conclude(
            describe_training_outcome(
                losses=losses, grad_norms=grad_norms,
                val_initial=val_initial, val_final=val_final,
                val_best=val_best_row["val_epe"] if val_rows else None,
                val_best_label="epoch {}".format(val_best_row["epoch"]) if val_rows else "",
                checkpoint_written=ckpt.exists(),
            )
            + " Primary endpoint: the matching path received gradient on "
              "{:.1%} of batches (control EXP-H1-WORKING-v2 measured at 5/150 "
              "crops at convergence). Accuracy is secondary and must be read "
              "against the control, not against Phase 1 numbers.".format(
                  matching_present / max(batches_total, 1)))
        print("\nrecorded as " + exp.id)


if __name__ == "__main__":
    main()
