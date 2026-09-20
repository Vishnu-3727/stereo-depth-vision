"""EXP-H2-SEED-REPLICATION-001 -- is the H2 stereo pathway reproducible across seeds?

A replication, not an architecture or tuning experiment. The only intentional
difference from `EXP-H2-SOFTARGMIN-SCALE` (seed 0) is the random seed.

    python phase2/scripts/exp_h2_seed_replication.py config              # Part A
    python phase2/scripts/exp_h2_seed_replication.py preflight --seed 1  # Parts B/C/D
    python phase2/scripts/exp_h2_seed_replication.py train --seed 1      # Parts E/F/I

Design decision, recorded before running: the task's pipeline is
short-training-then-full-training. Running 10 epochs and then restarting for 200
would either change the cosine schedule (`T_max` tracks the epoch budget in the
H2 recipe) or throw away the 10 epochs. So `train` runs the single 200-epoch H2
recipe and evaluates a **gate at epoch 10**: if the seed is not learning, not
receiving matching-path gradient, or not stereo-dependent, the run aborts there
and the record says so. Same compute as short-then-full, no schedule confound,
and the early exit is real.

Nothing here writes to any seed-0, H1 or H3 path. Phase 1 is untouched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
from phase2.scripts.exp_h3_viability import d1_of, entropy_of, full_forward  # noqa: E402
from phase2.viz import core  # noqa: E402
from src.common.experiment import Experiment  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.losses.disparity import masked_smooth_l1  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

PHASE2_ROOT = REPO_ROOT / "phase2"
EXPERIMENTS_DIR = PHASE2_ROOT / "experiments"
OUT_DIR = PHASE2_ROOT / "results" / "training"
RESULT_DIR = PHASE2_ROOT / "results" / "EXP-H2-SEED-REPLICATION-001"
REFERENCE_ID = "EXP-H2-SOFTARGMIN-SCALE"          # seed 0, the reference run
EXPERIMENT_PREFIX = "EXP-H2-SEED-REPLICATION-001"
FOCUS_SCENES = [27, 0, 31, 6]

EPOCHS = 200
GATE_EPOCH = 10
CHECKPOINT_EPOCHS = (1, 2, 5, 10, 20, 50, 100, 150, 200)
WEIGHT_SNAPSHOT_EPOCHS = (10, 20, 50, 100, 150, 200)

# Gate thresholds. The epoch-10 stereo criterion used in the first attempt was
# INVALIDATED by control evidence -- seed 0 itself fails it at epoch 10 -- and the
# corrected protocol is frozen in
# phase2/docs/EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md. Stereo
# dependence is now recorded at epochs 10 and 100 and judged only at epoch 200.
GATE_MIN_EPE_IMPROVEMENT = 3.0     # px, epoch-10 validation vs initialisation
GATE_MIN_MATCHING_FRACTION = 0.95  # batches with matching-path gradient
GATE_MIN_STEREO_D1_POINTS = 20.0   # D1 cost of corrupting the right image, AT EPOCH 200
STEREO_GATE_EPOCH = 200            # the only epoch at which the criterion is calibrated
MIN_FINAL_DISPARITY_STD = 1.0      # px, collapse check
MIN_SOFTMAX_ENTROPY = 0.5          # nats; H1's broken regime was 0.0000


def experiment_id(seed: int, label: str = "") -> str:
    return "{}-SEED{}{}".format(EXPERIMENT_PREFIX, seed, label)


def build_model(seed: int, device: str) -> StereoNet:
    """The H2 architecture, initialised exactly the way the H2 script does it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = StereoNet(StereoNetConfig(cost_volume_shift="left")).to(device)
    scaled_regression.apply_to(model)
    assert model.cost_volume.shift == "left", "H2 must not inherit H3's shift"
    return model


def recipe_config(seed: int, device: str) -> dict:
    """The configuration this replication will train under."""
    return {
        "architecture": "StereoNet (Phase 1 src/models/stereonet) + phase2 "
                        "StandardisedDisparityRegression",
        "cost_volume_shift": "left",
        "cost_volume_method": StereoNetConfig().cost_volume_method,
        "num_disparities": StereoNetConfig().num_disparities,
        "regression": "standardised soft-argmin (phase2.models.scaled_regression)",
        "seed": seed,
        "dataset": "kitti2015",
        "train_split": "hailo_calib (scenes 0-159)",
        "validation_split": "hailo_val (scenes 160-199), first 10 scenes validated",
        "crop": [CROP_H, CROP_W],
        "augmentation": "random crop, per-image gain jitter sigma=0.1, no flip",
        "optimizer": "Adam betas=(0.9, 0.999)",
        "learning_rate": 1e-3,
        "scheduler": "CosineAnnealingLR T_max={}".format(EPOCHS),
        "loss": "masked smooth L1, beta=1.0, valid = gt > 0 and gt < max_disparity",
        "batch_size": 2,
        "epochs": EPOCHS,
        "precision": "fp32",
        "device": device,
    }


def measure_macs(model: StereoNet, device: str) -> dict:
    """MAC count for one 256x512 stereo pair, or an explicit failure note."""
    try:
        from thop import profile
        left = torch.randn(1, 3, CROP_H, CROP_W, device=device)
        right = torch.randn(1, 3, CROP_H, CROP_W, device=device)
        macs, params = profile(model, inputs=(left, right), verbose=False)
        return {"macs": float(macs), "thop_params": float(params),
                "input": [1, 3, CROP_H, CROP_W]}
    except Exception as exc:
        return {"macs": None, "error": "{}: {}".format(type(exc).__name__, exc)}


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception as exc:
            return "unavailable: {}".format(exc)
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "phase_1_diff_vs_frozen": run(
            "git", "diff", "--name-only", "phase-1-frozen", "--", "src", "scripts"),
        "tracked_changes": run("git", "status", "--short"),
    }


# --- Part A -------------------------------------------------------------------

def run_config_check(device: str, seeds: list[int]) -> dict:
    """Compare this replication's configuration against the recorded seed-0 run."""
    recorded = json.loads(
        (EXPERIMENTS_DIR / REFERENCE_ID / "config.json").read_text(encoding="utf-8"))
    recorded = recorded.get("config", recorded)

    model = build_model(seeds[0], device)
    macs = measure_macs(model, device)
    here = recipe_config(seeds[0], device)

    # Field-by-field comparison against what the seed-0 record states.
    comparisons = {
        "cost_volume_shift": ("left" if "cost_volume_shift='left'" in
                              str(recorded.get("held_fixed", "")) else "NOT FOUND",
                              here["cost_volume_shift"]),
        "dataset": (recorded["dataset"], here["dataset"]),
        "split": (recorded["split"], here["train_split"] + " / " + here["validation_split"]),
        "crop": (recorded["resolution"], here["crop"]),
        "batch_size": (recorded["batch_size"], here["batch_size"]),
        "optimizer": (recorded["optimizer"], here["optimizer"]),
        "learning_rate": (recorded["learning_rate"], here["learning_rate"]),
        "scheduler": (recorded["schedule"], here["scheduler"]),
        "loss": (recorded["loss"], here["loss"]),
        "augmentation": (recorded["augmentation"], here["augmentation"]),
        "epochs": (recorded["epochs"], here["epochs"]),
        "precision": (recorded["precision"], here["precision"]),
        "seed": (recorded["seed"], "1 and 2 (the only intended difference)"),
    }
    # `seed` is the intended difference. `split`, `scheduler` and `augmentation`
    # are phrased differently in the two records but denote the same thing, so
    # they are checked by substance below instead of by string equality --
    # `augmentation` in particular is not a description this script re-implements:
    # it is the same imported `CroppedKitti` class the reference run used, which
    # is asserted rather than compared.
    mismatches = [
        k for k, (a, b) in comparisons.items()
        if k not in ("seed", "split", "scheduler", "augmentation")
        and str(a).strip().lower() != str(b).strip().lower()
    ]
    scheduler_ok = ("cosine" in str(comparisons["scheduler"][0]).lower()
                    and str(EPOCHS) in str(comparisons["scheduler"][0])
                    and "T_max={}".format(EPOCHS) in comparisons["scheduler"][1])
    split_ok = ("hailo_calib" in str(comparisons["split"][0])
                and "hailo_val" in str(comparisons["split"][0]))
    reference_augmentation = str(comparisons["augmentation"][0])
    augmentation_ok = (
        CroppedKitti.__module__ == "phase2.scripts.exp_h1_cost_volume"
        and "random crop" in reference_augmentation
        and "0.1" in reference_augmentation
        and "no horizontal flip" in reference_augmentation
        and abs(CroppedKitti(Kitti2015Stereo(
            REPO_ROOT / "data" / "kitti2015", split="hailo_calib"), seed=0).jitter
            - 0.1) < 1e-12)

    reference_ckpt = core.checkpoint_path("H2")
    return {
        "purpose": "Part A: the replication configuration must match the recorded "
                   "seed-0 H2 run in everything except the seed",
        "git": git_state(),
        "reference_experiment": REFERENCE_ID,
        "reference_checkpoint_sha256": core.sha256(reference_ckpt),
        "recorded_seed0_config": recorded,
        "replication_config": here,
        "field_comparison": {k: {"seed0_record": a, "replication": b}
                             for k, (a, b) in comparisons.items()},
        "parameter_count": model.parameter_count(),
        "parameter_count_per_occurrence": model.parameter_count(per_occurrence=True),
        "macs": macs,
        "mismatches": mismatches,
        "scheduler_substance_matches": bool(scheduler_ok),
        "split_substance_matches": bool(split_ok),
        "augmentation_is_the_reference_class": bool(augmentation_ok),
        "augmentation_note": ("the record's wording differs from this script's "
                              "one-line summary; the augmentation itself is the "
                              "same imported CroppedKitti (jitter 0.1, no flip)"),
        "PASS": bool(not mismatches and scheduler_ok and split_ok and augmentation_ok),
    }


# --- Parts B, C, D ------------------------------------------------------------

def stereo_probe(model: StereoNet, scenes: list[core.Scene], device: str,
                 rng: np.random.Generator) -> list[dict]:
    """Right-image and matching-map dependence for one model."""
    rows = []
    for scene in scenes:
        base = full_forward(model, scene.left, scene.right, device)
        base_pred = base["disparity_final"][0, 0].cpu().numpy().astype(np.float64)
        entry = {"scene": scene.name, "baseline": d1_of(base_pred, scene),
                 "softmax_entropy": entropy_of(base["softmax_input"])}
        for name, right in (
            ("right_black", np.zeros_like(scene.right)),
            ("right_noise", rng.integers(0, 256, size=scene.right.shape, dtype=np.uint8)),
            ("right_equals_left", scene.left.copy()),
        ):
            pred = full_forward(model, scene.left, right, device)[
                "disparity_final"][0, 0].cpu().numpy().astype(np.float64)
            entry[name] = d1_of(pred, scene)
            entry[name]["d1_penalty"] = entry[name]["d1"] - entry["baseline"]["d1"]
            entry[name]["mean_abs_change_px"] = float(np.abs(pred - base_pred).mean())

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
    return rows


def gradient_probe(model: StereoNet, device: str, crops: int = 12,
                   seed: int = 0) -> dict:
    """Part D: does the matching path receive gradient, and how large is it?"""
    calib = [core.load_scene(i, split="hailo_calib") for i in range(crops)]
    h, w = calib[0].shape
    rng = np.random.default_rng(seed)
    max_disparity = float(model.config.max_disparity_px)
    was_training = model.training
    model.train()
    rows = []
    for scene in calib:
        y = int(rng.integers(0, h - CROP_H + 1))
        x = int(rng.integers(0, w - CROP_W + 1))
        sl = (slice(y, y + CROP_H), slice(x, x + CROP_W))
        model.zero_grad(set_to_none=True)
        model.regression.capture = True
        pred = model(torch.from_numpy(normalize(scene.left[sl])).to(device),
                     torch.from_numpy(normalize(scene.right[sl])).to(device))
        gt = torch.from_numpy(scene.gt_disparity[sl][None, None].astype(np.float32)).to(device)
        loss, _ = masked_smooth_l1(pred, gt, max_disparity=max_disparity)
        loss.backward()
        grads = stage_gradients(model)
        snap = saturation_snapshot(model)
        model.regression.capture = False
        model.regression.last = {}
        model.zero_grad(set_to_none=True)
        rows.append({"scene": scene.name, "crop_yx": [y, x], "loss": float(loss),
                     **grads, "softmax_entropy": snap.get("mean_softmax_entropy_nats"),
                     "exact_ties": snap.get("exact_ties")})
    if not was_training:
        model.eval()
    norms = np.array([r["matching_path"] for r in rows])
    totals = np.array([np.sqrt(r["feature_extractor"] ** 2 + r["aggregation"] ** 2
                               + r["refinement"] ** 2) for r in rows])
    losses = np.array([r["loss"] for r in rows])
    return {
        "batches": len(rows),
        "matching_present": int(sum(r["matching_present"] for r in rows)),
        "matching_present_fraction": float(np.mean([r["matching_present"] for r in rows])),
        "matching_norm": {"median": float(np.median(norms)), "min": float(norms.min()),
                          "max": float(norms.max())},
        "total_norm": {"median": float(np.median(totals)), "max": float(totals.max())},
        "spikes_above_1e4": int((totals > 1e4).sum()),
        "mean_softmax_entropy": float(np.mean([r["softmax_entropy"] for r in rows])),
        "exact_ties_total": int(sum(r["exact_ties"] or 0 for r in rows)),
        "any_non_finite": bool(not np.all(np.isfinite(losses))
                               or not np.all(np.isfinite(norms))),
        "crops": rows,
    }


def forward_sanity(model: StereoNet, scenes: list[core.Scene], device: str) -> list[dict]:
    """Part B: shapes, finiteness and ranges of every stage, before training."""
    rows = []
    for scene in scenes:
        out = full_forward(model, scene.left, scene.right, device)
        row = {"scene": scene.name, "stages": {}}
        for name, tensor in out.items():
            row["stages"][name] = {
                "shape": list(tensor.shape),
                "nan": int(torch.isnan(tensor).sum()),
                "inf": int(torch.isinf(tensor).sum()),
                "min": float(tensor.min()), "max": float(tensor.max()),
                "std": float(tensor.std()),
            }
        row["softmax_entropy_nats"] = entropy_of(out["softmax_input"])
        pred = out["disparity_final"][0, 0].cpu().numpy().astype(np.float64)
        row["metrics_untrained"] = d1_of(pred, scene)
        rows.append(row)
    return rows


def run_preflight(device: str, seed: int, scenes: list[int]) -> dict:
    model = build_model(seed, device)
    model.eval()
    loaded = [core.load_scene(i) for i in scenes]
    rng = np.random.default_rng(0)

    result = {
        "purpose": "Parts B/C/D for seed {}: forward sanity, stereo probe and "
                   "gradient probe on the *untrained* initialisation".format(seed),
        "seed": seed,
        "git": git_state(),
        "config": recipe_config(seed, device),
        "parameter_count": model.parameter_count(),
        "cost_volume_shift": model.cost_volume.shift,
        "forward_sanity": forward_sanity(model, loaded, device),
        "stereo_probe_untrained": stereo_probe(model, loaded, device, rng),
        "gradient_probe": gradient_probe(model, device),
    }

    finite = all(s["nan"] == 0 and s["inf"] == 0
                 for row in result["forward_sanity"] for s in row["stages"].values())
    entropies = [row["softmax_entropy_nats"] for row in result["forward_sanity"]]
    g = result["gradient_probe"]
    result["gate"] = {
        "shift_is_left": model.cost_volume.shift == "left",
        "no_nan_or_inf": bool(finite),
        "parameter_count_matches_reference": model.parameter_count() == 423586,
        "softmax_not_saturated": bool(min(entropies) > 0.5),
        "min_softmax_entropy_nats": float(min(entropies)),
        "matching_gradient_fraction": g["matching_present_fraction"],
        "matching_gradient_gate": g["matching_present_fraction"] >= GATE_MIN_MATCHING_FRACTION,
        "no_gradient_spike": g["spikes_above_1e4"] == 0,
        "no_non_finite_gradient": not g["any_non_finite"],
    }
    result["gate"]["PASS"] = all(
        v for k, v in result["gate"].items()
        if isinstance(v, bool) and k != "PASS")
    return result


# --- Parts E, F, I ------------------------------------------------------------

def run_training(device: str, seed: int, scenes: list[int],
                 stop_after: int | None = None, label: str = "") -> None:
    preflight_path = RESULT_DIR / "preflight_seed{}.json".format(seed)
    if not preflight_path.exists():
        raise SystemExit("run preflight for seed {} first".format(seed))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight["gate"]["PASS"]:
        raise SystemExit("seed {} failed preflight; training is blocked".format(seed))

    exp_id = experiment_id(seed, label)
    if (EXPERIMENTS_DIR / exp_id).exists():
        raise SystemExit(exp_id + " already exists -- records are never overwritten.")

    train_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    model = build_model(seed, device)           # seeds torch/np exactly as H2 does
    train = CroppedKitti(train_base, seed=seed)
    loader = DataLoader(train, batch_size=2, shuffle=True, num_workers=0)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    max_disparity = float(model.config.max_disparity_px)
    probe_scenes = [core.load_scene(i) for i in scenes]

    budget = EPOCHS if stop_after is None else stop_after
    config = dict(recipe_config(seed, device))
    config.update({
        "hypothesis": ("H2's functional stereo pathway is reproducible across "
                       "random seeds and is not an artifact of seed 0."),
        "experiment": "EXP-H2-SEED-REPLICATION-001, replication arm seed {}".format(seed),
        "reference_run": REFERENCE_ID + " (seed 0; not retrained)",
        "reference_checkpoint_sha256": preflight.get("reference_checkpoint_sha256")
        or core.sha256(core.checkpoint_path("H2")),
        "changed_variable": "random seed only ({} instead of 0)".format(seed),
        "preflight": str(preflight_path.relative_to(REPO_ROOT)),
        "gate_at_epoch": GATE_EPOCH,
        "gate_thresholds": {
            "min_val_epe_improvement_px": GATE_MIN_EPE_IMPROVEMENT,
            "min_matching_gradient_fraction": GATE_MIN_MATCHING_FRACTION,
            "min_right_image_d1_points": GATE_MIN_STEREO_D1_POINTS,
        },
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "stop_after": stop_after,
        "stop_after_note": (None if stop_after is None else
                            "reference measurement only: the cosine schedule "
                            "still uses T_max={} so these are the reference "
                            "run's first {} epochs, but training halts there "
                            "instead of continuing".format(EPOCHS, stop_after)),
        "validation_protocol": ("first 10 scenes of hailo_val, full 368x1232 "
                                "frames, pooled over gt > 0 "
                                "(exp_h1_cost_volume.validate, imported); every 5 "
                                "epochs as in the reference run, plus epochs 1 and 2"),
        "known_limitation": ("one run per seed, 160 training scenes; three seeds "
                             "in total across the experiment cannot establish "
                             "statistical significance."),
    })

    with Experiment("H2 seed replication, seed {}".format(seed), config=config,
                    experiment_id=exp_id, experiments_dir=EXPERIMENTS_DIR) as exp:
        exp.note(config["hypothesis"])
        exp.note(config["known_limitation"])
        exp.note("Corrected protocol, frozen before this run in phase2/docs/"
                 "EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md: epoch {} "
                 "enforces learning (>= {} px improvement) and matching-path "
                 "gradient (>= {:.0%} of batches) only; epoch 100 enforces a "
                 "collapse check; stereo dependence is RECORDED at epochs 10 and "
                 "100 and JUDGED only at epoch {}, against a frozen {} D1-point "
                 "threshold. The epoch-10 stereo gate used in the first attempt "
                 "was invalidated by control evidence -- seed 0 fails it too.".format(
                     GATE_EPOCH, GATE_MIN_EPE_IMPROVEMENT, GATE_MIN_MATCHING_FRACTION,
                     STEREO_GATE_EPOCH, GATE_MIN_STEREO_D1_POINTS))

        history, grad_norms, matching_norms = [], [], []
        matching_present, batches_total = 0, 0
        checkpoints, ablations = {}, {}
        t0 = time.time()
        aborted = None
        rng = np.random.default_rng(0)

        initial = validate(model, val_base, device, limit=10)
        checkpoints["0"] = {
            "epoch": 0, "val_epe": initial.epe, "val_d1": initial.d1,
            "val_loss": _validation_loss(model, val_base, device, max_disparity),
            "stage_stats": validation_stage_stats(model, val_base, device),
            "train_loss_of_previous_epoch": None}
        exp.log("checkpoint epoch   0 (untrained)  val EPE {:.3f}  D1 {:.2f}%".format(
            initial.epe, initial.d1))

        for epoch in range(budget):
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
                    aborted = ("non-finite loss or gradient at epoch {} batch {}"
                               .format(epoch, batches_total))
                    break
                epoch_loss += float(loss)
                batches += 1
                batches_total += 1
                grad_norms.append(gn)
                matching_norms.append(stages["matching_path"])
                matching_present += int(stages["matching_present"])
            if aborted:
                break

            scheduler.step()
            row = {"epoch": epoch, "mean_loss": epoch_loss / max(batches, 1),
                   "lr": float(scheduler.get_last_lr()[0]),
                   "median_grad_norm": float(np.median(grad_norms[-batches:])),
                   "median_matching_grad_norm": float(np.median(matching_norms[-batches:])),
                   "matching_present_fraction_this_epoch": float(np.mean(
                       [m > GRADIENT_PRESENT_THRESHOLD for m in matching_norms[-batches:]]))}
            done = epoch + 1                       # epochs completed
            if done % 5 == 0 or done in CHECKPOINT_EPOCHS:
                metrics = validate(model, val_base, device, limit=10)
                row["val_epe"], row["val_d1"] = metrics.epe, metrics.d1
            history.append(row)
            exp.log("epoch {:>3}  loss {:8.4f}  grad {:9.3f}  match {:9.3e} "
                    "({:.0%})  lr {:.2e}{}".format(
                        epoch, row["mean_loss"], row["median_grad_norm"],
                        row["median_matching_grad_norm"],
                        row["matching_present_fraction_this_epoch"], row["lr"],
                        "  val EPE {:.3f}  D1 {:.2f}%".format(row["val_epe"], row["val_d1"])
                        if "val_epe" in row else ""))

            if done in CHECKPOINT_EPOCHS:
                metrics = validate(model, val_base, device, limit=10)
                entry = {"epoch": done, "val_epe": metrics.epe, "val_d1": metrics.d1,
                         "val_loss": _validation_loss(model, val_base, device, max_disparity),
                         "stage_stats": validation_stage_stats(model, val_base, device),
                         "train_loss_of_previous_epoch": row["mean_loss"],
                         "grad": {
                             "median_total": float(np.median(grad_norms)),
                             "max_total": float(np.max(grad_norms)),
                             "median_matching": float(np.median(matching_norms)),
                             "matching_present_fraction":
                                 matching_present / max(batches_total, 1)}}
                checkpoints[str(done)] = entry
                exp.metric("checkpoint_epoch_{}".format(done), entry)
                if done in WEIGHT_SNAPSHOT_EPOCHS:
                    snap = OUT_DIR / "{}_epoch{}.pth".format(exp_id, done)
                    torch.save({"model": model.state_dict(), "epoch": done,
                                "config": config}, snap)

            if done in (GATE_EPOCH, 100, EPOCHS) or done == budget:
                model.eval()
                ablations[str(done)] = stereo_probe(model, probe_scenes, device, rng)
                model.train()
                exp.metric("stereo_ablation_epoch_{}".format(done), ablations[str(done)])
                if done == GATE_EPOCH and stop_after is not None:
                    verdict = _gate_verdict(
                        checkpoints, ablations[str(GATE_EPOCH)],
                        matching_present / max(batches_total, 1))
                    exp.metric("gate_epoch_{}_not_enforced".format(GATE_EPOCH), verdict)
                    exp.log("GATE at epoch {} (recorded, NOT enforced for this "
                            "reference run): {}".format(
                                GATE_EPOCH, "PASS" if verdict["PASS"] else "FAIL"))

            if done == 100 and stop_after is None:
                collapse = _collapse_check(
                    checkpoints, 100, matching_present / max(batches_total, 1))
                exp.metric("collapse_check_epoch_100", collapse)
                exp.log("COLLAPSE CHECK at epoch 100: {}".format(
                    "PASS" if collapse["PASS"] else "FAIL"))
                if not collapse["PASS"]:
                    aborted = "failed the epoch-100 collapse check: {}".format(collapse)
                    exp.note("ABORTED: " + aborted)
                    break

            if done == EPOCHS and stop_after is None:
                verdict = _stereo_verdict(
                    checkpoints, ablations[str(EPOCHS)],
                    matching_present / max(batches_total, 1))
                exp.metric("stereo_verdict_epoch_200", verdict)
                exp.log("STEREO VERDICT at epoch 200: {}".format(
                    "STEREO-FUNCTIONAL" if verdict["STEREO_FUNCTIONAL"]
                    else "NOT STEREO-FUNCTIONAL"))

            if done == GATE_EPOCH and stop_after is None:
                gate = _gate_verdict(checkpoints, ablations[str(GATE_EPOCH)],
                                     matching_present / max(batches_total, 1))
                exp.metric("gate_epoch_{}".format(GATE_EPOCH), gate)
                exp.log("GATE at epoch {}: {}".format(
                    GATE_EPOCH, "PASS" if gate["PASS"] else "FAIL"))
                if not gate["PASS"]:
                    aborted = "failed the epoch-{} viability gate: {}".format(
                        GATE_EPOCH, gate)
                    exp.note("ABORTED: " + aborted)
                    break

        totals = np.array(grad_norms) if grad_norms else np.array([np.nan])
        matching = np.array(matching_norms) if matching_norms else np.array([np.nan])
        exp.metric("history", history)
        exp.metric("checkpoints", checkpoints)
        exp.metric("stereo_ablations", ablations)
        exp.metric("aborted", aborted)
        exp.metric("wall_clock_s", time.time() - t0)
        exp.metric("epochs_completed", len(history))
        exp.metric("primary_endpoint_matching_gradient", {
            "batches": batches_total,
            "present_fraction": matching_present / max(batches_total, 1),
            "norm": {"median": float(np.median(matching)), "max": float(matching.max()),
                     "p90": float(np.percentile(matching, 90))}})
        exp.metric("gradient_norms", {
            "median": float(np.median(totals)), "max": float(totals.max()),
            "p90": float(np.percentile(totals, 90)),
            "any_nan": bool(np.any(~np.isfinite(totals))),
            "batches_above_1e4": int((totals > 1e4).sum())})
        exp.metric("parameter_count_unique", model.parameter_count())

        ckpt = OUT_DIR / (exp_id + "_checkpoint.pth")
        torch.save({"model": model.state_dict(), "config": config}, ckpt)
        exp.metric("checkpoint", str(ckpt.relative_to(REPO_ROOT)))
        exp.metric("checkpoint_sha256", core.sha256(ckpt))
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / "history_seed{}.json".format(seed)).write_text(
            json.dumps({"history": history, "checkpoints": checkpoints,
                        "stereo_ablations": ablations}, indent=2), encoding="utf-8")

        val_rows = [r for r in history if "val_epe" in r]
        exp.metric("validation_epe_series",
                   [{"epoch": r["epoch"], "val_epe": r["val_epe"], "val_d1": r["val_d1"]}
                    for r in val_rows])
        if val_rows:
            best = min(val_rows, key=lambda r: r["val_epe"])
            exp.metric("validation_epe_best", best["val_epe"])
            exp.metric("validation_epe_best_epoch", best["epoch"])
        exp.conclude(
            "Seed {} replication of {}. {} epochs completed; matching-path "
            "gradient on {:.1%} of {} batches; total gradient max {:.4g}; {}.".format(
                seed, REFERENCE_ID, len(history),
                matching_present / max(batches_total, 1), batches_total,
                float(totals.max()),
                "no abort" if aborted is None else "ABORTED: " + str(aborted)))
        print("\nrecorded as " + exp.id)


def _gate_verdict(checkpoints: dict, ablation: list[dict],
                  matching_fraction: float) -> dict:
    start = checkpoints["0"]["val_epe"]
    now = checkpoints[str(GATE_EPOCH)]["val_epe"]
    right_penalty = min(row[k]["d1_penalty"] for row in ablation
                        for k in ("right_black", "right_noise", "right_equals_left"))
    map_penalty = min(row[k]["d1_penalty"] for row in ablation
                      for k in ("initial_constant_mean", "initial_shuffled"))
    checks = {
        "val_epe_start": start, "val_epe_at_gate": now,
        "val_epe_improvement": start - now,
        "learning": (start - now) >= GATE_MIN_EPE_IMPROVEMENT,
        "matching_gradient_fraction": matching_fraction,
        "matching_gradient": matching_fraction >= GATE_MIN_MATCHING_FRACTION,
        "right_image_d1_penalty": right_penalty,
        "matching_map_d1_penalty": map_penalty,
        # RECORDED, never enforced at this epoch: the control fails it here too
        # (protocol correction, section 1), so enforcing it kills runs that would
        # have become stereo-functional later.
        "stereo_dependence_at_this_epoch":
            right_penalty >= GATE_MIN_STEREO_D1_POINTS,
        "stereo_dependence_enforced": False,
    }
    checks["PASS"] = bool(checks["learning"] and checks["matching_gradient"])
    return checks


def _collapse_check(checkpoints: dict, epoch: int, matching_fraction: float) -> dict:
    """Enforced mid-run check: no collapse, gradient still flowing."""
    now, ten = checkpoints[str(epoch)], checkpoints[str(GATE_EPOCH)]
    checks = {
        "epoch": epoch,
        "val_epe_at_epoch_10": ten["val_epe"],
        "val_epe_now": now["val_epe"],
        "no_regression_vs_epoch_10": now["val_epe"] <= ten["val_epe"],
        "matching_gradient_fraction": matching_fraction,
        "matching_gradient": matching_fraction >= GATE_MIN_MATCHING_FRACTION,
        "final_disparity_std": now["stage_stats"]["final_std"],
        "not_collapsed": now["stage_stats"]["final_std"] > MIN_FINAL_DISPARITY_STD,
        "softmax_entropy": now["stage_stats"]["entropy"],
        "entropy_ok": now["stage_stats"]["entropy"] > MIN_SOFTMAX_ENTROPY,
    }
    checks["PASS"] = bool(
        checks["no_regression_vs_epoch_10"] and checks["matching_gradient"]
        and checks["not_collapsed"] and checks["entropy_ok"])
    return checks


def _stereo_verdict(checkpoints: dict, ablation: list[dict],
                    matching_fraction: float) -> dict:
    """The frozen 200-epoch stereo criterion (protocol correction, section 4.2)."""
    right = min(row[k]["d1_penalty"] for row in ablation
                for k in ("right_black", "right_noise", "right_equals_left"))
    matching_map = min(row[k]["d1_penalty"] for row in ablation
                       for k in ("initial_constant_mean", "initial_shuffled"))
    final = checkpoints[str(EPOCHS)]
    checks = {
        "threshold_d1_points": GATE_MIN_STEREO_D1_POINTS,
        "right_image_min_d1_penalty": right,
        "right_image_dependence": right >= GATE_MIN_STEREO_D1_POINTS,
        "matching_map_min_d1_penalty": matching_map,
        "matching_map_dependence": matching_map >= GATE_MIN_STEREO_D1_POINTS,
        "matching_gradient_fraction": matching_fraction,
        "matching_gradient": matching_fraction >= GATE_MIN_MATCHING_FRACTION,
        "softmax_entropy": final["stage_stats"]["entropy"],
        "entropy_ok": final["stage_stats"]["entropy"] > MIN_SOFTMAX_ENTROPY,
        "final_disparity_std": final["stage_stats"]["final_std"],
        "not_collapsed": final["stage_stats"]["final_std"] > MIN_FINAL_DISPARITY_STD,
    }
    checks["STEREO_FUNCTIONAL"] = bool(
        checks["right_image_dependence"] and checks["matching_map_dependence"]
        and checks["matching_gradient"] and checks["entropy_ok"]
        and checks["not_collapsed"])
    return checks


def _validation_loss(model: StereoNet, dataset, device: str,
                     max_disparity: float, limit: int = 10) -> float:
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


def run_visuals(device: str, model_keys: list[str], scenes: list[int]) -> list[str]:
    """Part G: normal vs corrupted right, drawn with the existing visualizer.

    No new visualization system: this composes `phase2.viz.render`'s own panels.
    """
    from phase2.viz import render

    out_dir = PHASE2_ROOT / "visualizations" / "seed_replication_stereo_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = core.ModelRunner(device=device)
    written = []
    for key in model_keys:
        model = runner._model(key)[0]
        for index in scenes:
            scene = core.load_scene(index)
            normal = full_forward(model, scene.left, scene.right, device)[
                "disparity_final"][0, 0].cpu().numpy().astype(np.float64)
            black = full_forward(model, scene.left, np.zeros_like(scene.right), device)[
                "disparity_final"][0, 0].cpu().numpy().astype(np.float64)
            err_n, valid = render.error_map(normal, scene)
            err_b, _ = render.error_map(black, scene)
            m_n, m_b = d1_of(normal, scene), d1_of(black, scene)

            fig, axes = render.plt.subplots(3, 2, figsize=(16, 10), dpi=110)
            render.show_rgb(axes[0, 0], scene.left, "left image -- " + scene.name)
            render.show_map(axes[0, 1], scene.gt_disparity, scene.gt_valid,
                            "ground-truth disparity", "px", render.DISPARITY_CMAP)
            lo, hi, _ = render._range(scene.gt_disparity, scene.gt_valid)
            render.show_map(axes[1, 0], normal, None,
                            "{}: normal right image  (EPE {:.2f}, D1 {:.1f}%)".format(
                                key, m_n["epe"], m_n["d1"]), "px",
                            render.DISPARITY_CMAP, vmin=lo, vmax=hi)
            render.show_map(axes[1, 1], black, None,
                            "{}: right image BLACK  (EPE {:.2f}, D1 {:.1f}%)".format(
                                key, m_b["epe"], m_b["d1"]), "px",
                            render.DISPARITY_CMAP, vmin=lo, vmax=hi)
            e_lo, e_hi, _ = render._range(err_n, valid)
            render.show_map(axes[2, 0], err_n, valid,
                            "|error| with the right image", "px", render.ERROR_CMAP,
                            vmin=e_lo, vmax=e_hi)
            render.show_map(axes[2, 1], err_b, valid,
                            "|error| without it (same scale)", "px", render.ERROR_CMAP,
                            vmin=e_lo, vmax=e_hi)
            fig.suptitle("{} -- stereo dependence, D1 {:.1f}% -> {:.1f}% "
                         "({:+.1f} points) when the right image is destroyed".format(
                             key, m_n["d1"], m_b["d1"], m_b["d1"] - m_n["d1"]),
                         fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            path = out_dir / "{}_{}_right_image_ablation.png".format(
                key.lower(), Path(scene.name).stem)
            fig.savefig(path, bbox_inches="tight")
            render.plt.close(fig)
            written.append(str(path))
            print("visuals: {} scene {} done".format(key, index))
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["config", "preflight", "train", "visuals"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--scenes", nargs="+", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stop-after", type=int, default=None,
                    help="halt training after this many epochs (reference "
                         "measurements only; the schedule is unchanged and the "
                         "epoch-10 gate is recorded but not enforced)")
    ap.add_argument("--models", nargs="+", default=["H2", "SEED1"],
                    help="registered model keys for the 'visuals' stage")
    ap.add_argument("--label", default="",
                    help="suffix for the experiment id, e.g. '-REFERENCE'")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    scenes = args.scenes or FOCUS_SCENES

    if args.stage == "config":
        result = run_config_check(device, args.seeds)
        (RESULT_DIR / "config_check.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print("PART A -- configuration check vs " + REFERENCE_ID)
        for key, pair in result["field_comparison"].items():
            print("  {:<16} record: {:<48} replication: {}".format(
                key, str(pair["seed0_record"])[:48], str(pair["replication"])[:48]))
        print("\n  parameters {} (per-occurrence {}), MACs {}".format(
            result["parameter_count"], result["parameter_count_per_occurrence"],
            result["macs"].get("macs")))
        print("  git commit {}".format(result["git"]["commit"][:12]))
        print("  Phase 1 diff vs phase-1-frozen: '{}'".format(
            result["git"]["phase_1_diff_vs_frozen"]))
        print("  mismatches: {}".format(result["mismatches"] or "none"))
        print("\nPART A: " + ("PASS" if result["PASS"] else "FAIL"))
        if not result["PASS"]:
            raise SystemExit(1)
        return

    if args.stage == "preflight":
        result = run_preflight(device, args.seed, scenes)
        path = RESULT_DIR / "preflight_seed{}.json".format(args.seed)
        result["reference_checkpoint_sha256"] = core.sha256(core.checkpoint_path("H2"))
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("\nwrote " + str(path))
        print("\nPREFLIGHT seed {}: {}".format(
            args.seed, "PASS" if result["gate"]["PASS"] else "FAIL"))
        for key, value in result["gate"].items():
            if key != "PASS":
                print("  {:<38} {}".format(key, value))
        if not result["gate"]["PASS"]:
            raise SystemExit(1)
        return

    if args.stage == "visuals":
        written = run_visuals(device, args.models, scenes)
        print("wrote {} figures".format(len(written)))
        return

    run_training(device, args.seed, scenes, args.stop_after, args.label)


if __name__ == "__main__":
    main()
