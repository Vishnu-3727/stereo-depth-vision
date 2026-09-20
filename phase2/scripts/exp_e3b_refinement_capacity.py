"""EXP-E3B-REFINEMENT-CAPACITY-001 -- does a *trained* smaller refinement stack match H2?

E3 measured that pruning a trained six-block stack is never free, and recorded the
retrained case as UNKNOWN. This trains it.

    python phase2/scripts/exp_e3b_refinement_capacity.py preflight
    python phase2/scripts/exp_e3b_refinement_capacity.py train --arm A   # 6 blocks, control
    python phase2/scripts/exp_e3b_refinement_capacity.py train --arm B   # 5 blocks
    python phase2/scripts/exp_e3b_refinement_capacity.py train --arm C   # 4 blocks
    python phase2/scripts/exp_e3b_refinement_capacity.py analyse

Arms, blocks removed, gates, the noise band, the verdict rule and the conditional
stage 2 are all frozen in
`phase2/docs/EXP_E3B_REFINEMENT_CAPACITY_001_PREREGISTRATION.md`, approved by
`phase2/docs/PHASE_2_STAGE_E_DECISION_RECORD.md`. This file only executes them.

Arm A is a **same-seed repeat of the unmodified H2 recipe**, not the existing H2
checkpoint: the difference between the two is the same-seed run-to-run noise at
200 epochs, which is what every comparison here is judged against.

Everything except the block count is imported from the H2 / seed-replication
recipe rather than restated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase2.scripts.exp_h1_cost_volume import CroppedKitti, validate  # noqa: E402
from phase2.scripts.exp_h2_seed_replication import (  # noqa: E402
    CHECKPOINT_EPOCHS, EPOCHS, EXPERIMENTS_DIR, FOCUS_SCENES, GATE_EPOCH,
    GRADIENT_PRESENT_THRESHOLD, OUT_DIR, WEIGHT_SNAPSHOT_EPOCHS, _collapse_check,
    _gate_verdict, _stereo_verdict, _validation_loss, build_model, git_state,
    measure_macs, recipe_config, stereo_probe,
)
from phase2.scripts.exp_h2_softargmin_scale import (  # noqa: E402
    stage_gradients, validation_stage_stats,
)
from phase2.scripts.exp_h3_viability import full_forward  # noqa: E402
from phase2.viz import core  # noqa: E402
from src.common.experiment import Experiment  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo  # noqa: E402
from src.losses.disparity import masked_smooth_l1  # noqa: E402

EXPERIMENT_PREFIX = "EXP-E3B-REFINEMENT-CAPACITY-001"
RESULT_DIR = REPO_ROOT / "phase2" / "results" / EXPERIMENT_PREFIX
SEED = 0

# Frozen in the pre-registration section 2: the trailing dilation-1 blocks are
# removed, the 1-2-4-8 ladder is kept intact.
ARMS = {
    "A": {"blocks": 6, "keep": (0, 1, 2, 3, 4, 5), "dilations": (1, 2, 4, 8, 1, 1),
          "parameters": 423586, "macs_deploy": 56.034e9, "role": "control (H2 repeat)"},
    "B": {"blocks": 5, "keep": (0, 1, 2, 3, 4), "dilations": (1, 2, 4, 8, 1),
          "parameters": 405090, "macs_deploy": 47.677e9, "role": "ablated"},
    "C": {"blocks": 4, "keep": (0, 1, 2, 3), "dilations": (1, 2, 4, 8),
          "parameters": 386594, "macs_deploy": 39.321e9, "role": "ablated"},
}

# The recorded seed-0 H2 late-window figures the control is compared against
# (EXP-H2-SOFTARGMIN-SCALE, epochs 150-199, recorded validation protocol).
H2_RECORDED_LATE = {"epe": 4.199, "d1": 24.783}
LATE_WINDOW = (150, 199)
# Frozen floors (pre-registration section 4), reused verbatim from E2/E3.
FLOOR_EPE = 0.10
FLOOR_D1 = 1.0
BAND_MULTIPLIER = 2.0


# Set from --label. Records are never overwritten, so a rerun of the same arms
# carries a suffix; the pre-registration, preflight and thresholds are unchanged.
ARM_LABEL = ""


def arm_id(arm: str) -> str:
    return "{}-ARM-{}{}".format(EXPERIMENT_PREFIX, arm, ARM_LABEL)


def build_arm(arm: str, device: str):
    """H2's model at seed 0, truncated to this arm's block count.

    The full stack is built first and then truncated, so every surviving block
    holds exactly the weights it would have held in the control -- which the
    preflight asserts.
    """
    model = build_model(SEED, device)              # seeds torch/np, shift=left, H2 readout
    keep = ARMS[arm]["keep"]
    model.refinement.blocks = nn.Sequential(
        *[model.refinement.blocks[i] for i in keep])
    return model


def run_preflight(device: str) -> dict:
    """Blocking: the arms may differ in nothing but the block count."""
    models = {arm: build_arm(arm, device) for arm in ARMS}
    scenes = [core.load_scene(i) for i in FOCUS_SCENES]
    out = {"git": git_state(), "arms": {}, "checks": {}}

    # Pass 1 -- weight identity. This must finish before any thop call, because
    # thop attaches total_ops/total_params buffers to the modules it profiles.
    reference = models["A"]
    identical = True
    for arm, model in models.items():
        for position, index in enumerate(ARMS[arm]["keep"]):
            a = reference.refinement.blocks[index].state_dict()
            b = model.refinement.blocks[position].state_dict()
            identical = identical and all(torch.equal(a[k], b[k]) for k in a)
        for name in ("feature_extractor", "aggregation"):
            a = getattr(reference, name).state_dict()
            b = getattr(model, name).state_dict()
            identical = identical and all(torch.equal(a[k], b[k]) for k in a)
        for name in ("input_conv", "output_conv"):
            a = getattr(reference.refinement, name).state_dict()
            b = getattr(model.refinement, name).state_dict()
            identical = identical and all(torch.equal(a[k], b[k]) for k in a)

    # Pass 2 -- forward sanity and MACs.
    for arm, model in models.items():
        model.eval()
        finite, ranges = True, {}
        for scene in scenes:
            stages = full_forward(model, scene.left, scene.right, device)
            for stage, tensor in stages.items():
                if not torch.isfinite(tensor).all():
                    finite = False
                ranges.setdefault(stage, []).append(float(tensor.abs().max()))
        out["arms"][arm] = {
            "blocks": len(model.refinement.blocks),
            "dilations": list(ARMS[arm]["dilations"]),
            "parameters": model.parameter_count(),
            "parameters_expected": ARMS[arm]["parameters"],
            "parameters_match": model.parameter_count() == ARMS[arm]["parameters"],
            "finite_at_init": finite,
            "stage_max_abs": {k: max(v) for k, v in ranges.items()},
            "cost_volume_shift": model.cost_volume.shift,
            "macs": measure_macs(model, device),      # last: thop mutates the model
        }

    out["checks"] = {
        "surviving_weights_bit_identical": bool(identical),
        "parameter_counts_match_registration": all(
            out["arms"][a]["parameters_match"] for a in ARMS),
        "block_counts_correct": all(
            out["arms"][a]["blocks"] == ARMS[a]["blocks"] for a in ARMS),
        "all_finite_at_init": all(out["arms"][a]["finite_at_init"] for a in ARMS),
        "shift_is_left": all(out["arms"][a]["cost_volume_shift"] == "left" for a in ARMS),
        "h2_checkpoint_sha256": core.sha256(core.checkpoint_path("H2")),
        "phase_1_diff_empty": out["git"]["phase_1_diff_vs_frozen"] == "",
    }
    out["checks"]["PASS"] = all(
        v for k, v in out["checks"].items()
        if isinstance(v, bool))
    return out


def arm_config(arm: str, device: str) -> dict:
    spec = ARMS[arm]
    config = dict(recipe_config(SEED, device))
    config.update({
        "experiment": arm_id(arm),
        "hypothesis": ("A refinement stack trained at {} residual blocks matches "
                       "six-block H2 at the same budget, seed and recipe, and "
                       "stays stereo-functional.".format(spec["blocks"])),
        "preregistration": "phase2/docs/EXP_E3B_REFINEMENT_CAPACITY_001_PREREGISTRATION.md",
        "approved_by": "phase2/docs/PHASE_2_STAGE_E_DECISION_RECORD.md",
        "arm": arm,
        "arm_role": spec["role"],
        "refinement_blocks": spec["blocks"],
        "refinement_dilations": list(spec["dilations"]),
        "changed_variable": ("number of residual blocks in Refinement.blocks "
                             "({} instead of 6); trailing dilation-1 blocks "
                             "removed, the 1-2-4-8 ladder kept intact".format(
                                 spec["blocks"])),
        "control": ("arm A of this experiment -- a same-seed repeat of the "
                    "unmodified H2 recipe, NOT the existing H2 checkpoint; the "
                    "A-vs-EXP-H2-SOFTARGMIN-SCALE difference is the same-seed "
                    "run-to-run noise every comparison is judged against"),
        "gate_at_epoch": GATE_EPOCH,
        "stereo_gate_epoch": EPOCHS,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "validation_protocol": ("first 10 scenes of hailo_val, full 368x1232 "
                                "frames, pooled over gt > 0 "
                                "(exp_h1_cost_volume.validate, imported); every 5 "
                                "epochs, plus epochs 1 and 2"),
        "known_limitation": ("160 training scenes from random initialisation: a "
                             "capacity result here does NOT transfer to a "
                             "pretrained model. MACs are arithmetic, not latency "
                             "(Phase 1 measured 0.67x-142x misprediction on this "
                             "stack). One seed in stage 1; no significance test "
                             "and none claimed."),
    })
    return config


def run_training(arm: str, device: str) -> None:
    preflight_path = RESULT_DIR / "preflight.json"
    if not preflight_path.exists():
        raise SystemExit("run preflight first")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight["checks"]["PASS"]:
        raise SystemExit("preflight failed; training is blocked")

    exp_id = arm_id(arm)
    if (EXPERIMENTS_DIR / exp_id).exists():
        raise SystemExit(exp_id + " already exists -- records are never overwritten.")

    train_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")
    model = build_arm(arm, device)
    loader = DataLoader(CroppedKitti(train_base, seed=SEED), batch_size=2,
                        shuffle=True, num_workers=0)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    max_disparity = float(model.config.max_disparity_px)
    probe_scenes = [core.load_scene(i) for i in FOCUS_SCENES]
    config = arm_config(arm, device)

    with Experiment("E3b arm {} -- refinement at {} blocks".format(
                    arm, ARMS[arm]["blocks"]), config=config, experiment_id=exp_id,
                    experiments_dir=EXPERIMENTS_DIR) as exp:
        exp.note(config["hypothesis"])
        exp.note(config["known_limitation"])
        exp.note("Stereo dependence is RECORDED at epochs 10 and 100 and JUDGED "
                 "only at epoch 200 -- never gated before epoch 20, per "
                 "EXP-H2-EMERGENCE-001.")

        history, grad_norms, matching_norms = [], [], []
        matching_present, batches_total = 0, 0
        checkpoints, ablations = {}, {}
        aborted, t0 = None, time.time()
        rng = np.random.default_rng(0)

        initial = validate(model, val_base, device, limit=10)
        checkpoints["0"] = {
            "epoch": 0, "val_epe": initial.epe, "val_d1": initial.d1,
            "val_loss": _validation_loss(model, val_base, device, max_disparity),
            "stage_stats": validation_stage_stats(model, val_base, device),
            "train_loss_of_previous_epoch": None}
        exp.log("checkpoint epoch   0 (untrained)  val EPE {:.3f}  D1 {:.2f}%".format(
            initial.epe, initial.d1))

        for epoch in range(EPOCHS):
            epoch_loss, batches = 0.0, 0
            for left, right, disparity in loader:
                left, right = left.to(device), right.to(device)
                disparity = disparity.to(device)
                loss, _ = masked_smooth_l1(model(left, right), disparity,
                                           max_disparity=max_disparity)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9))
                stages = stage_gradients(model)
                optimizer.step()
                if not np.isfinite(float(loss)) or not np.isfinite(gn):
                    aborted = "non-finite loss or gradient at epoch {} batch {}".format(
                        epoch, batches_total)
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
            done = epoch + 1
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
                checkpoints[str(done)] = {
                    "epoch": done, "val_epe": metrics.epe, "val_d1": metrics.d1,
                    "val_loss": _validation_loss(model, val_base, device, max_disparity),
                    "stage_stats": validation_stage_stats(model, val_base, device),
                    "train_loss_of_previous_epoch": row["mean_loss"],
                    "grad": {"median_total": float(np.median(grad_norms)),
                             "max_total": float(np.max(grad_norms)),
                             "median_matching": float(np.median(matching_norms)),
                             "matching_present_fraction":
                                 matching_present / max(batches_total, 1)}}
                exp.metric("checkpoint_epoch_{}".format(done), checkpoints[str(done)])
                if done in WEIGHT_SNAPSHOT_EPOCHS:
                    torch.save({"model": model.state_dict(), "epoch": done,
                                "config": config},
                               OUT_DIR / "{}_epoch{}.pth".format(exp_id, done))

            if done in (GATE_EPOCH, 100, EPOCHS):
                model.eval()
                ablations[str(done)] = stereo_probe(model, probe_scenes, device, rng)
                model.train()
                exp.metric("stereo_ablation_epoch_{}".format(done), ablations[str(done)])

            if done == GATE_EPOCH:
                gate = _gate_verdict(checkpoints, ablations[str(GATE_EPOCH)],
                                     matching_present / max(batches_total, 1))
                exp.metric("gate_epoch_{}".format(GATE_EPOCH), gate)
                exp.log("GATE at epoch {}: {}".format(
                    GATE_EPOCH, "PASS" if gate["PASS"] else "FAIL"))
                if not gate["PASS"]:
                    aborted = "failed the epoch-{} viability gate: {}".format(GATE_EPOCH, gate)
                    exp.note("ABORTED: " + aborted)
                    break

            if done == 100:
                collapse = _collapse_check(checkpoints, 100,
                                           matching_present / max(batches_total, 1))
                exp.metric("collapse_check_epoch_100", collapse)
                exp.log("COLLAPSE CHECK at epoch 100: {}".format(
                    "PASS" if collapse["PASS"] else "FAIL"))
                if not collapse["PASS"]:
                    aborted = "failed the epoch-100 collapse check: {}".format(collapse)
                    exp.note("ABORTED: " + aborted)
                    break

            if done == EPOCHS:
                verdict = _stereo_verdict(checkpoints, ablations[str(EPOCHS)],
                                          matching_present / max(batches_total, 1))
                exp.metric("stereo_verdict_epoch_200", verdict)
                exp.log("STEREO VERDICT at epoch 200: {}".format(
                    "STEREO-FUNCTIONAL" if verdict["STEREO_FUNCTIONAL"]
                    else "NOT STEREO-FUNCTIONAL"))

        totals = np.array(grad_norms) if grad_norms else np.array([np.nan])
        matching = np.array(matching_norms) if matching_norms else np.array([np.nan])
        val_rows = [r for r in history if "val_epe" in r]
        late = [r for r in val_rows if LATE_WINDOW[0] <= r["epoch"] <= LATE_WINDOW[1]]
        late_stats = {
            "points": len(late),
            "epe_mean": float(np.mean([r["val_epe"] for r in late])) if late else None,
            "epe_sd": float(np.std([r["val_epe"] for r in late])) if late else None,
            "d1_mean": float(np.mean([r["val_d1"] for r in late])) if late else None,
            "d1_sd": float(np.std([r["val_d1"] for r in late])) if late else None,
        }
        exp.metric("history", history)
        exp.metric("checkpoints", checkpoints)
        exp.metric("stereo_ablations", ablations)
        exp.metric("late_window", late_stats)
        exp.metric("aborted", aborted)
        exp.metric("wall_clock_s", time.time() - t0)
        exp.metric("epochs_completed", len(history))
        exp.metric("parameter_count_unique", model.parameter_count())
        exp.metric("primary_endpoint_matching_gradient", {
            "batches": batches_total,
            "present_fraction": matching_present / max(batches_total, 1),
            "norm": {"median": float(np.median(matching)), "max": float(matching.max())}})
        exp.metric("gradient_norms", {
            "median": float(np.median(totals)), "max": float(totals.max()),
            "any_nan": bool(np.any(~np.isfinite(totals))),
            "batches_above_1e4": int((totals > 1e4).sum())})

        ckpt = OUT_DIR / (exp_id + "_checkpoint.pth")
        torch.save({"model": model.state_dict(), "config": config}, ckpt)
        exp.metric("checkpoint", str(ckpt.relative_to(REPO_ROOT)))
        exp.metric("checkpoint_sha256", core.sha256(ckpt))
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / "history_arm_{}{}.json".format(arm, ARM_LABEL)).write_text(json.dumps(
            {"history": history, "checkpoints": checkpoints,
             "stereo_ablations": ablations, "late_window": late_stats}, indent=2),
            encoding="utf-8")
        exp.conclude("Arm {} ({} blocks): {} epochs; late-window EPE {} D1 {}; "
                     "matching gradient {:.1%} of {} batches; {}.".format(
                         arm, ARMS[arm]["blocks"], len(history),
                         late_stats["epe_mean"], late_stats["d1_mean"],
                         matching_present / max(batches_total, 1), batches_total,
                         "no abort" if aborted is None else "ABORTED: " + str(aborted)))
        print("\nrecorded as " + exp.id)


def _arm_metrics(arm: str) -> dict:
    path = EXPERIMENTS_DIR / arm_id(arm) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def run_analysis(device: str) -> None:
    exp_id = EXPERIMENT_PREFIX + "-ANALYSIS" + ARM_LABEL
    if (EXPERIMENTS_DIR / exp_id).exists():
        raise SystemExit(exp_id + " already exists -- records are never overwritten.")
    arms = {a: _arm_metrics(a) for a in ARMS}

    noise = {
        "epe": abs(arms["A"]["late_window"]["epe_mean"] - H2_RECORDED_LATE["epe"]),
        "d1": abs(arms["A"]["late_window"]["d1_mean"] - H2_RECORDED_LATE["d1"]),
    }
    band = {"epe": max(BAND_MULTIPLIER * noise["epe"], FLOOR_EPE),
            "d1": max(BAND_MULTIPLIER * noise["d1"], FLOOR_D1)}

    comparison, stereo_ok = {}, {}
    for arm in ("B", "C"):
        d_epe = arms[arm]["late_window"]["epe_mean"] - arms["A"]["late_window"]["epe_mean"]
        d_d1 = arms[arm]["late_window"]["d1_mean"] - arms["A"]["late_window"]["d1_mean"]
        material = abs(d_epe) > band["epe"] and abs(d_d1) > band["d1"]
        verdict = arms[arm].get("stereo_verdict_epoch_200", {})
        stereo_ok[arm] = bool(verdict.get("STEREO_FUNCTIONAL"))
        comparison[arm] = {
            "delta_epe_vs_A": d_epe, "delta_d1_vs_A": d_d1,
            "materially_different": bool(material),
            "worse": bool(material and d_epe > 0 and d_d1 > 0),
            "stereo_functional": stereo_ok[arm],
            "stereo_verdict": verdict,
            "aborted": arms[arm].get("aborted"),
        }

    a_stereo = bool(arms["A"].get("stereo_verdict_epoch_200", {}).get("STEREO_FUNCTIONAL"))
    gate = {
        "control_completed": arms["A"].get("aborted") is None
        and arms["A"]["epochs_completed"] == EPOCHS,
        "control_stereo_functional": a_stereo,
        "control_converged": (arms["A"]["checkpoints"]["0"]["val_epe"]
                              - arms["A"]["late_window"]["epe_mean"]) > 3.0,
        "arms_completed": all(arms[a].get("aborted") is None for a in ARMS),
    }
    gate["PASS"] = all(gate.values())

    viable = [a for a in ("B", "C")
              if not comparison[a]["worse"] and stereo_ok[a]]
    accuracy_ok_but_broken = [a for a in ("B", "C")
                              if not comparison[a]["worse"] and not stereo_ok[a]]
    if not gate["PASS"]:
        verdict = "INVALID"
    elif accuracy_ok_but_broken:
        verdict = "STEREO-BROKEN"
    elif viable:
        verdict = "CAPACITY-REDUCIBLE"
    elif all(comparison[a]["worse"] for a in ("B", "C")):
        verdict = "CAPACITY-REQUIRED"
    else:
        verdict = "INCONCLUSIVE"

    stage2 = [a for a in ("B", "C") if comparison[a]["materially_different"]]

    config = {
        "experiment": exp_id,
        "hypothesis": "Analysis of the three E3b stage-1 arms against the frozen rule.",
        "preregistration": "phase2/docs/EXP_E3B_REFINEMENT_CAPACITY_001_PREREGISTRATION.md",
        "measurement_only": True, "training": "none -- reads the three arm records",
        "dataset": "kitti2015", "split": "hailo_val (first 10 scenes, recorded protocol)",
        "resolution": "368x1232", "crop": [256, 512], "disparity_range": 12,
        "batch_size": 2, "precision": "fp32", "seed": SEED,
        "known_limitation": ("the noise band is a SINGLE-SAMPLE lower bound from one "
                             "same-seed repeat, not a confidence interval; no p-value "
                             "is claimed. 160 training scenes; no transfer to a "
                             "pretrained model is implied."),
    }
    with Experiment("E3b stage 1 analysis", config=config, experiment_id=exp_id,
                    experiments_dir=EXPERIMENTS_DIR) as exp:
        exp.note(config["known_limitation"])
        exp.log("same-seed noise (arm A vs recorded H2): EPE {:.4f} px, D1 {:.4f} pt"
                .format(noise["epe"], noise["d1"]))
        exp.log("material band: EPE > {:.4f} px and D1 > {:.4f} pt".format(
            band["epe"], band["d1"]))
        for arm in ARMS:
            lw = arms[arm]["late_window"]
            exp.log("arm {} ({} blocks): late EPE {:.3f} +- {:.3f}  D1 {:.3f} +- {:.3f}  "
                    "params {}  stereo {}".format(
                        arm, ARMS[arm]["blocks"], lw["epe_mean"], lw["epe_sd"],
                        lw["d1_mean"], lw["d1_sd"], arms[arm]["parameter_count_unique"],
                        arms[arm].get("stereo_verdict_epoch_200", {})
                        .get("STEREO_FUNCTIONAL")))
        exp.metric("same_seed_noise", noise)
        exp.metric("material_band", band)
        exp.metric("late_window", {a: arms[a]["late_window"] for a in ARMS})
        exp.metric("comparison", comparison)
        exp.metric("gates", gate)
        exp.metric("verdict", verdict)
        exp.metric("stage_2_triggered_for", stage2)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / "stage1_analysis{}.json".format(
            ARM_LABEL.lower())).write_text(json.dumps(
            {"same_seed_noise": noise, "material_band": band,
             "late_window": {a: arms[a]["late_window"] for a in ARMS},
             "comparison": comparison, "gates": gate, "verdict": verdict,
             "stage_2_triggered_for": stage2}, indent=2), encoding="utf-8")
        exp.conclude("E3b stage 1: verdict {}; stage 2 triggered for {}.".format(
            verdict, stage2 or "no arm"))
        print("\nVERDICT: {}   stage 2 for: {}".format(verdict, stage2 or "no arm"))
        print("recorded as " + exp.id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preflight", "train", "analyse"])
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--label", default="",
                    help="suffix for the arm record ids, e.g. '-RUN2'; records "
                         "are never overwritten, so a rerun needs new ids")
    args = ap.parse_args()
    global ARM_LABEL
    ARM_LABEL = args.label
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "preflight":
        result = run_preflight(device)
        (RESULT_DIR / "preflight.json").write_text(json.dumps(result, indent=2),
                                                   encoding="utf-8")
        for arm in sorted(ARMS):
            row = result["arms"][arm]
            print("arm {}  {} blocks  {} params (expected {})  {}".format(
                arm, row["blocks"], row["parameters"], row["parameters_expected"],
                "ok" if row["parameters_match"] else "MISMATCH"))
        for key, value in result["checks"].items():
            print("  {:<40} {}".format(key, value))
        print("\nPREFLIGHT: " + ("PASS" if result["checks"]["PASS"] else "FAIL"))
        if not result["checks"]["PASS"]:
            raise SystemExit(1)
        return

    if args.stage == "train":
        if not args.arm:
            raise SystemExit("--arm is required")
        run_training(args.arm, device)
        return

    run_analysis(device)


if __name__ == "__main__":
    main()
