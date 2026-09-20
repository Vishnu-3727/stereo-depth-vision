"""EXP-BLOCKCOUNT-FULL-001 -- the 200-epoch block-count experiment, BATCH 1 ONLY.

EXP-BLOCKCOUNT-SEEDSCREEN-001 returned CONTINUE-FULL and warned that a naive
3-arm x 2-seed 200-epoch campaign is at risk of being underpowered: at 30 epochs
the seed spread was the same order as the architecture spread. The research lead
therefore authorised a SEQUENTIAL campaign with a hard review gate:

    BATCH 1   6 blocks seed 1      200 epochs
              6 blocks seed 2      200 epochs
              5 blocks seed 1      200 epochs      -> then STOP and inspect

Rationale, from the lead: the two 6-block seeds establish the seed trajectory
range at fixed architecture, and the single 5-block arm says whether an
architecture difference survives past the 30-epoch emergence region. Batch 2
(5 blocks seed 2, 4 blocks seeds 1 and 2) is NOT authorised and this script
REFUSES to run it -- see BATCH1 and the batch guard in cmd_train.

    python .../exp_blockcount_full.py preflight  --out DIR
    python .../exp_blockcount_full.py train      --out DIR --arm A --seed 1
    python .../exp_blockcount_full.py evaluate40 --out DIR --arm A --seed 1
    python .../exp_blockcount_full.py probes     --out DIR
    python .../exp_blockcount_full.py compare    --out DIR

Writes only under --out. Phase 1, every Phase 2 scientific script and every
historical record (H1, H2, H3, E1, E2, E3, E3b, E3c, O6, Stage A, Stage B, the
deterministic block-count series, the seed screen) are read-only here.

WHAT IS DIFFERENT FROM THE SCREEN
---------------------------------
The screen shortened the epoch BUDGET to 30 while pinning the SCHEDULE at
T_max=200, which made it a strict prefix of the run it was forecasting. This
experiment runs the frozen recipe unmodified: budget 200, schedule T_max 200,
the native e3b/H2 checkpoint epochs, the epoch-10 viability gate, the epoch-100
collapse check and the epoch-200 stereo verdict all active. Nothing about the
recipe is re-fitted; only `refinement_blocks` and `seed` change.

Seed 0 is NOT retrained. Its 200-epoch records exist and are quoted read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing the screen module sets CUBLAS_WORKSPACE_CONFIG before torch is used
# and gives us its already-reviewed provenance/environment/probe helpers. Its
# own constants are never applied here: this module does its own repointing.
import phase2.factorial.block_count_seed_screen.exp_blockcount_seed_screen as screen  # noqa: E402,F401

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

import phase2.scripts.exp_e3b_refinement_capacity as e3b        # noqa: E402
import phase2.scripts.exp_h2_seed_replication as h2             # noqa: E402
from phase2.scripts.exp_h2_seed_replication import (            # noqa: E402
    FOCUS_SCENES, measure_macs, stereo_probe,
)
from phase2.scripts.exp_h2_softargmin_scale import (            # noqa: E402
    validation_stage_stats,
)
from phase2.viz import core                                     # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo              # noqa: E402

from phase2.factorial.block_count_seed_screen.exp_blockcount_seed_screen import (  # noqa: E402
    CUBLAS_CONFIG, SEED0_INIT_SHA, FrozenCosineAnnealingLR, _eval40,
    enable_determinism, environment, file_sha256, provenance, sha, weight_sha,
)

EXPERIMENT_PREFIX = "EXP-BLOCKCOUNT-FULL-001"

# THE AUTHORISED BATCH. Nothing outside this tuple may be trained by this script.
BATCH1 = (("A", 1), ("A", 2), ("B", 1))
BATCH1_LABEL = ("batch 1 of a sequential campaign: 6b/seed1, 6b/seed2, 5b/seed1; "
                "batch 2 (5b/seed2, 4b/seed1, 4b/seed2) is NOT authorised and is "
                "refused by this script until the research lead reviews batch 1")

FULL_EPOCHS = 200                 # the BUDGET *and* the SCHEDULE -- both native
PROBE_EPOCHS = (10, 20, 50, 100, 150, 200)   # the native weight-snapshot epochs

# Seed-0 200-epoch records, READ-ONLY. Not retrained, not overwritten.
SEED0 = {
    "A": {"blocks": 6,
          "record": "phase2/diagnostics/determinism/20260909T041500Z_baseline",
          "experiment": "STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA",
          "frozen_40_scene": {"epe": 2.2955181809208267, "d1": 15.6045142562172}},
    "B": {"blocks": 5,
          "record": "phase2/factorial/block_count_deterministic/20260909T131500Z",
          "experiment": "EXP-BLOCKCOUNT-DETERMINISTIC-001-ARM-B",
          "frozen_40_scene": {"epe": 2.6458, "d1": 21.3712}},
    "C": {"blocks": 4,
          "record": "phase2/factorial/block_count_deterministic/20260909T131500Z",
          "experiment": "EXP-BLOCKCOUNT-DETERMINISTIC-001-ARM-C",
          "frozen_40_scene": {"epe": 2.3387, "d1": 16.9797}},
}


def run_id(arm: str, seed: int) -> str:
    return "{}-ARM-{}-SEED{}".format(EXPERIMENT_PREFIX, arm, seed)


# --------------------------------------------------------------------- repoint
def repoint(out: Path, seed) -> None:
    """Point the frozen E3b training code at this experiment's identity.

    Unlike the screen, the epoch budget and the checkpoint schedule are LEFT
    NATIVE: 200 epochs, CHECKPOINT_EPOCHS (1,2,5,10,20,50,100,150,200),
    WEIGHT_SNAPSHOT_EPOCHS (10,20,50,100,150,200), gate at 10, collapse check at
    100, stereo verdict at 200. Only paths, identity and seed are changed.
    """
    out = out.resolve()
    e3b.EXPERIMENT_PREFIX = EXPERIMENT_PREFIX
    e3b.RESULT_DIR = out / "results"
    e3b.EXPERIMENTS_DIR = out / "experiments"
    e3b.OUT_DIR = out / "checkpoints"
    for d in (e3b.RESULT_DIR, e3b.EXPERIMENTS_DIR, e3b.OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # The budget and the snapshot schedule are the imported originals. Asserted,
    # not assigned, so that a change upstream fails loudly instead of silently.
    assert e3b.EPOCHS == FULL_EPOCHS, e3b.EPOCHS
    assert h2.EPOCHS == FULL_EPOCHS, h2.EPOCHS
    assert e3b.CHECKPOINT_EPOCHS == (1, 2, 5, 10, 20, 50, 100, 150, 200), \
        e3b.CHECKPOINT_EPOCHS
    assert e3b.WEIGHT_SNAPSHOT_EPOCHS == (10, 20, 50, 100, 150, 200), \
        e3b.WEIGHT_SNAPSHOT_EPOCHS

    # A no-op at 200 -- installed only so the T_max actually used is recorded the
    # same way the screen recorded it, and so a re-fitted T_max cannot slip in.
    torch.optim.lr_scheduler.CosineAnnealingLR = FrozenCosineAnnealingLR

    if seed is not None:
        e3b.SEED = seed
        e3b.ARM_LABEL = "-SEED{}".format(seed)

    original = e3b.arm_config

    def full_arm_config(arm: str, device: str) -> dict:
        config = dict(original(arm, device))
        config.update({
            "experiment": e3b.arm_id(arm),
            "experiment_kind": ("200-EPOCH BLOCK-COUNT EXPERIMENT, BATCH 1 OF A "
                                "SEQUENTIAL CAMPAIGN WITH A HARD REVIEW GATE. "
                                + BATCH1_LABEL),
            "seed": e3b.SEED,
            "epochs": FULL_EPOCHS,
            "epoch_budget": FULL_EPOCHS,
            "scheduler": ("CosineAnnealingLR T_max={} -- the native frozen "
                          "schedule, neither shortened nor re-fitted".format(
                              FULL_EPOCHS)),
            "control": ("the seed-0 200-epoch records, READ-ONLY and NOT "
                        "retrained: Stage B deterministic run A (6 blocks) and "
                        "EXP-BLOCKCOUNT-DETERMINISTIC-001 arm B (5 blocks). Within "
                        "this batch, the two 6-block seeds are the seed-noise "
                        "reference the 5-block arm is read against."),
            "changed_variables": ["refinement block count", "seed"],
            "deterministic_controls": {
                "torch.use_deterministic_algorithms": True,
                "torch.backends.cudnn.deterministic": True,
                "torch.backends.cudnn.benchmark": False,
                "CUBLAS_WORKSPACE_CONFIG": CUBLAS_CONFIG,
            },
            "materiality_band": ("NONE. The historical E3b/E3c band is invalidated. "
                                 "No band is reused and none is invented here, "
                                 "before or after seeing the results. Establishing "
                                 "the seed-noise scale at 6 blocks is part of what "
                                 "this batch measures."),
            "claim_ceiling": ("Batch 1 has three runs: two seeds at 6 blocks and "
                              "ONE seed at 5 blocks. A single 5-block seed cannot "
                              "separate an architecture effect from a seed effect. "
                              "No statement of the form 'N blocks are "
                              "required/equivalent/superior' is supportable from "
                              "batch 1 alone."),
            "prior_screen": ("EXP-BLOCKCOUNT-SEEDSCREEN-001 "
                             "(phase2/factorial/block_count_seed_screen/"
                             "20260909T145659Z), verdict CONTINUE-FULL"),
        })
        return config

    e3b.arm_config = full_arm_config


# ------------------------------------------------------------------- preflight
def cmd_preflight(args) -> int:
    out = Path(args.out).resolve()
    repoint(out, None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controls = enable_determinism()

    runs, fresh_ok, identical_ok = {}, True, True
    for arm, seed in BATCH1:
        e3b.SEED = seed
        ref = e3b.build_arm("A", device)          # same seed, full six-block stack
        model = e3b.build_arm(arm, device)
        ok = True
        for position, index in enumerate(e3b.ARMS[arm]["keep"]):
            a = ref.refinement.blocks[index].state_dict()
            b = model.refinement.blocks[position].state_dict()
            ok = ok and all(torch.equal(a[k], b[k]) for k in a)
        for name in ("feature_extractor", "aggregation"):
            a, b = getattr(ref, name).state_dict(), getattr(model, name).state_dict()
            ok = ok and all(torch.equal(a[k], b[k]) for k in a)
        for name in ("input_conv", "output_conv"):
            a = getattr(ref.refinement, name).state_dict()
            b = getattr(model.refinement, name).state_dict()
            ok = ok and all(torch.equal(a[k], b[k]) for k in a)
        identical_ok = identical_ok and ok
        init = weight_sha(model)
        fresh_ok = fresh_ok and init != SEED0_INIT_SHA
        runs[run_id(arm, seed)] = {
            "arm": arm, "seed": seed,
            "blocks": len(model.refinement.blocks),
            "blocks_expected": e3b.ARMS[arm]["blocks"],
            "dilations": [int(b.conv1.dilation[0]) for b in model.refinement.blocks],
            "dilations_expected": list(e3b.ARMS[arm]["dilations"]),
            "parameters": model.parameter_count(),
            "parameters_expected": e3b.ARMS[arm]["parameters"],
            "parameters_match": model.parameter_count() == e3b.ARMS[arm]["parameters"],
            "init_weight_sha": init,
            "init_differs_from_seed0": init != SEED0_INIT_SHA,
            "surviving_weights_bit_identical_within_seed": ok,
            "cost_volume_shift": model.cost_volume.shift,
            "num_disparities": model.cost_volume.num_disparities,
            "regression": type(model.regression).__name__,
            "epoch_budget": e3b.EPOCHS,
            "loaded_from_stage_b_checkpoint": False,
            "in_authorised_batch": True,
            "macs_256x512": measure_macs(e3b.build_arm(arm, device), device),
        }
        del model, ref

    probe_model = e3b.build_arm("A", device)
    probe_opt = torch.optim.Adam(probe_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    probe_sched = torch.optim.lr_scheduler.CosineAnnealingLR(probe_opt, T_max=e3b.EPOCHS)
    schedule = {
        "class": type(probe_sched).__name__,
        "T_max_requested_by_training_code": FrozenCosineAnnealingLR.requested_T_max,
        "T_max_actually_used": probe_sched.T_max,
        "lr_epoch_0": float(probe_opt.param_groups[0]["lr"]),
        "optimizer_state_empty_at_construction": len(probe_opt.state) == 0,
    }
    del probe_sched, probe_opt, probe_model

    result = {
        "experiment": EXPERIMENT_PREFIX,
        "purpose": ("batch 1 of the sequential 200-epoch block-count campaign: "
                    "establish the 200-epoch seed range at 6 blocks and take one "
                    "5-block seed, then STOP for review"),
        "authorised_batch": [list(p) for p in BATCH1],
        "batch_note": BATCH1_LABEL,
        "provenance": provenance(), "environment": environment(),
        "deterministic_controls": controls,
        "schedule": schedule,
        "seed0_records_readonly": SEED0,
        "runs": runs,
    }
    result["checks"] = {
        "phase_1_diff_empty": result["provenance"]["phase_1_diff_vs_frozen"] == "",
        "deterministic_controls_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
            and torch.backends.cudnn.deterministic
            and not torch.backends.cudnn.benchmark
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_CONFIG),
        "shift_is_left": all(r["cost_volume_shift"] == "left" for r in runs.values()),
        "readout_is_standardised": all(
            r["regression"] == "StandardisedDisparityRegression" for r in runs.values()),
        "candidates_are_12": all(r["num_disparities"] == 12 for r in runs.values()),
        "block_counts_correct": all(
            r["blocks"] == r["blocks_expected"] for r in runs.values()),
        "dilations_correct": all(
            r["dilations"] == r["dilations_expected"] for r in runs.values()),
        "parameter_counts_match_registration": all(
            r["parameters_match"] for r in runs.values()),
        "seeds_are_1_and_2_only": sorted({r["seed"] for r in runs.values()}) == [1, 2],
        "seed_0_not_scheduled": all(r["seed"] != 0 for r in runs.values()),
        "epoch_budget_is_200": all(r["epoch_budget"] == 200 for r in runs.values()),
        "schedule_T_max_is_200": schedule["T_max_actually_used"] == 200,
        "schedule_not_refitted": (schedule["T_max_requested_by_training_code"]
                                  == FULL_EPOCHS),
        "initialisation_is_fresh_not_stage_b": bool(fresh_ok),
        "no_stage_b_checkpoint_loaded": all(
            not r["loaded_from_stage_b_checkpoint"] for r in runs.values()),
        "optimizer_state_fresh": bool(schedule["optimizer_state_empty_at_construction"]),
        "surviving_weights_bit_identical_within_seed": bool(identical_ok),
        "run_count_is_3": len(runs) == 3,
        "batch_is_A1_A2_B1": (sorted((r["arm"], r["seed"]) for r in runs.values())
                              == sorted(BATCH1)),
        "batch_2_not_scheduled": all(
            (r["arm"], r["seed"]) in BATCH1 for r in runs.values()),
    }
    result["checks"]["PASS"] = all(v for v in result["checks"].values()
                                   if isinstance(v, bool))
    (e3b.RESULT_DIR / "preflight.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"schedule": schedule,
                      "runs": {k: {kk: v[kk] for kk in
                                   ("arm", "seed", "blocks", "parameters",
                                    "init_weight_sha", "init_differs_from_seed0",
                                    "epoch_budget")}
                               for k, v in runs.items()},
                      "checks": result["checks"]}, indent=2))
    if not result["checks"]["PASS"]:
        print("\nPREFLIGHT FAILED -- STOP. Nothing was trained.")
        return 1
    print("\nPREFLIGHT PASS.")
    return 0


# ----------------------------------------------------------------------- train
def cmd_train(args) -> int:
    out = Path(args.out).resolve()
    arm, seed = args.arm, int(args.seed)
    if seed == 0:
        print("STOP: seed 0 is frozen and must not be retrained.")
        return 1
    if (arm, seed) not in BATCH1:
        print("STOP: arm {} seed {} is NOT in the authorised batch {}. "
              "Batch 2 needs the research lead's review of batch 1 first."
              .format(arm, seed, list(BATCH1)))
        return 1
    repoint(out, seed)
    controls = enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = e3b.arm_id(arm)
    assert exp_id == run_id(arm, seed), (exp_id, run_id(arm, seed))

    model = e3b.build_arm(arm, device)
    meta = {"arm": arm, "seed": seed, "experiment_id": exp_id, "controls": controls,
            "batch": 1, "batch_note": BATCH1_LABEL,
            "blocks": len(model.refinement.blocks),
            "parameters": model.parameter_count(),
            "init_weight_sha": weight_sha(model),
            "epoch_budget": e3b.EPOCHS,
            "schedule_T_max": FULL_EPOCHS,
            "changed_variables": {"refinement_blocks": e3b.ARMS[arm]["blocks"],
                                  "seed": seed},
            "initialised_from": "fresh, per the frozen recipe; no checkpoint loaded",
            "provenance": provenance(), "environment": environment(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    stop = None
    if meta["parameters"] != e3b.ARMS[arm]["parameters"]:
        stop = "parameter count {}".format(meta["parameters"])
    if meta["init_weight_sha"] == SEED0_INIT_SHA:
        stop = "initial weights match the seed-0 initialisation"
    if e3b.EPOCHS != FULL_EPOCHS:
        stop = "epoch budget {}".format(e3b.EPOCHS)
    if stop:
        (out / "ABORTED_{}.json".format(exp_id)).write_text(
            json.dumps(dict(meta, stop=stop), indent=2), encoding="utf-8")
        print("STOP:", stop)
        return 1
    del model

    t0 = time.time()
    e3b.run_training(arm, device)
    meta["wall_clock_s"] = time.time() - t0
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["schedule_T_max_requested_by_training_code"] = \
        FrozenCosineAnnealingLR.requested_T_max

    ckpt = e3b.OUT_DIR / (exp_id + "_checkpoint.pth")
    meta["checkpoint"] = str(ckpt.relative_to(REPO_ROOT))
    meta["checkpoint_sha256"] = file_sha256(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    tensors = state["model"] if "model" in state else state
    meta["final_weight_sha"] = sha(torch.cat(
        [v.reshape(-1).float() for v in tensors.values() if v.is_floating_point()]))
    meta["snapshots"] = sorted(
        p.name for p in e3b.OUT_DIR.glob(exp_id + "_epoch*.pth"))
    (out / "run_meta_{}.json".format(exp_id)).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print("{} done in {:.1f}s; final weight sha {}".format(
        exp_id, meta["wall_clock_s"], meta["final_weight_sha"]))
    return 0


# ------------------------------------------------------- 40-scene evaluation
def cmd_evaluate40(args) -> int:
    out = Path(args.out).resolve()
    arm, seed = args.arm, int(args.seed)
    repoint(out, seed)
    enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_id = e3b.arm_id(arm)
    model = e3b.build_arm(arm, device)
    ckpt = e3b.OUT_DIR / (exp_id + "_checkpoint.pth")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    result = _eval40(model, device)
    result.update({"experiment_id": exp_id, "arm": arm, "seed": seed,
                   "epoch": FULL_EPOCHS,
                   "blocks": len(model.refinement.blocks),
                   "checkpoint_sha256": file_sha256(ckpt),
                   "environment": environment(), "provenance": provenance()})
    (out / "eval40_{}.json".format(exp_id)).write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    p = result["pooled"]
    print("{} 40-scene @200ep  EPE {:.7f}  D1 {:.7f}  RMSE {:.7f}".format(
        exp_id, p["epe"], p["d1"], p["rmse"]))
    return 0


# ---------------------------------------------------------------------- probes
def cmd_probes(args) -> int:
    out = Path(args.out).resolve()
    repoint(out, None)
    enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenes = [core.load_scene(i) for i in FOCUS_SCENES]
    val_base = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val")

    payload = {"probe_epochs": list(PROBE_EPOCHS), "focus_scenes": FOCUS_SCENES,
               "probe_implementation": ("exp_h2_seed_replication.stereo_probe and "
                                        "exp_h2_softargmin_scale."
                                        "validation_stage_stats, imported unchanged "
                                        "-- no new stereo metric was introduced"),
               "caveat": ("right-image and matching-map probes establish binocular "
                          "DEPENDENCE, not correct disparity search: Phase 1 "
                          "EXP-007 measured +89.7 D1 on the reference weights "
                          "despite a provably degenerate shift."),
               "runs": {}}
    for arm, seed in BATCH1:
        e3b.SEED = seed
        e3b.ARM_LABEL = "-SEED{}".format(seed)
        exp_id = e3b.arm_id(arm)
        print("{} ({} blocks, seed {}):".format(exp_id, e3b.ARMS[arm]["blocks"], seed))
        rows = {}
        for epoch in PROBE_EPOCHS:
            path = e3b.OUT_DIR / "{}_epoch{}.pth".format(exp_id, epoch)
            if not path.exists():
                rows[str(epoch)] = {"error": "snapshot missing: " + path.name}
                print("    epoch {:>3}  MISSING".format(epoch))
                continue
            model = e3b.build_arm(arm, device)
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"] if "model" in state else state)
            model.eval()
            rng = np.random.default_rng(0)
            ab = stereo_probe(model, scenes, device, rng)
            right = min(r[k]["d1_penalty"] for r in ab
                        for k in ("right_black", "right_noise", "right_equals_left"))
            mmap = min(r[k]["d1_penalty"] for r in ab
                       for k in ("initial_constant_mean", "initial_shuffled"))
            st = validation_stage_stats(model, val_base, device)
            rows[str(epoch)] = {
                "epoch": epoch,
                "right_image_min_d1_penalty": right,
                "matching_map_min_d1_penalty": mmap,
                "softmax_entropy": st["entropy"],
                "final_disparity_std": st["final_std"],
                "disparity_initial_corr_gt": st["initial_corr_gt"],
                "final_corr_gt": st["final_corr_gt"],
            }
            print("    epoch {:>3}  right {:+8.3f}  map {:+8.3f}  entropy {:.4f}  "
                  "std {:6.3f}  init r(GT) {:+.4f}".format(
                      epoch, right, mmap, st["entropy"], st["final_std"],
                      st["initial_corr_gt"]))
            del model
        payload["runs"][exp_id] = {"arm": arm, "seed": seed,
                                   "blocks": e3b.ARMS[arm]["blocks"],
                                   "series": rows}
    payload["environment"] = environment()
    payload["provenance"] = provenance()
    (out / "probes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out / "probes.json")
    return 0


# --------------------------------------------------------------------- compare
COMPARE_EPOCHS = (10, 20, 50, 100, 150, 200)


def _val_at(history, done: int):
    """Validation row for the epoch after `done` completed epochs (rows are 0-based)."""
    for row in history:
        if row.get("epoch") == done - 1 and "val_epe" in row:
            return {"epe": row["val_epe"], "d1": row["val_d1"],
                    "train_loss": row.get("mean_loss"),
                    "median_grad_norm": row.get("median_grad_norm"),
                    "median_matching_grad_norm": row.get("median_matching_grad_norm"),
                    "matching_present_fraction_this_epoch":
                        row.get("matching_present_fraction_this_epoch")}
    return None


def cmd_compare(args) -> int:
    out = Path(args.out).resolve()
    repoint(out, None)
    res = {"experiment": EXPERIMENT_PREFIX,
           "authorised_batch": [list(p) for p in BATCH1],
           "batch_note": BATCH1_LABEL,
           "compare_epochs": list(COMPARE_EPOCHS),
           "seed0_records_readonly": SEED0,
           "provenance": provenance(), "runs": {}}

    for arm, seed in BATCH1:
        e3b.SEED = seed
        e3b.ARM_LABEL = "-SEED{}".format(seed)
        exp_id = e3b.arm_id(arm)
        hist_path = e3b.RESULT_DIR / "history_arm_{}-SEED{}.json".format(arm, seed)
        record = json.loads(hist_path.read_text(encoding="utf-8"))
        history = record["history"]
        res["runs"][exp_id] = {
            "arm": arm, "seed": seed, "blocks": e3b.ARMS[arm]["blocks"],
            "epochs_completed": len(history),
            "at_epoch": {str(e): _val_at(history, e) for e in COMPARE_EPOCHS},
            "late_window": record.get("late_window"),
            "final_train_loss": history[-1]["mean_loss"] if history else None,
        }
        eval_path = out / "eval40_{}.json".format(exp_id)
        if eval_path.exists():
            res["runs"][exp_id]["eval40"] = json.loads(
                eval_path.read_text(encoding="utf-8"))["pooled"]

    # Seed spread at fixed architecture (6 blocks, seeds 1 vs 2) -- the reference
    # scale this batch exists to measure.
    a1, a2 = res["runs"][run_id("A", 1)], res["runs"][run_id("A", 2)]
    seed_spread = {}
    for e in COMPARE_EPOCHS:
        v1, v2 = a1["at_epoch"][str(e)], a2["at_epoch"][str(e)]
        if v1 and v2:
            seed_spread[str(e)] = {
                "seed1_epe": v1["epe"], "seed2_epe": v2["epe"],
                "abs_delta_epe": abs(v1["epe"] - v2["epe"]),
                "seed1_d1": v1["d1"], "seed2_d1": v2["d1"],
                "abs_delta_d1": abs(v1["d1"] - v2["d1"])}
    res["seed_spread_6_blocks"] = seed_spread

    # The 5-block arm read against that spread. One seed only -- stated as such.
    b1 = res["runs"][run_id("B", 1)]
    arch = {}
    for e in COMPARE_EPOCHS:
        v1, v2, vb = (a1["at_epoch"][str(e)], a2["at_epoch"][str(e)],
                      b1["at_epoch"][str(e)])
        if v1 and v2 and vb:
            lo, hi = min(v1["epe"], v2["epe"]), max(v1["epe"], v2["epe"])
            lo_d1, hi_d1 = min(v1["d1"], v2["d1"]), max(v1["d1"], v2["d1"])
            arch[str(e)] = {
                "blocks5_seed1_epe": vb["epe"],
                "blocks6_seed_range_epe": [lo, hi],
                "epe_inside_6block_seed_range": lo <= vb["epe"] <= hi,
                "blocks5_seed1_d1": vb["d1"],
                "blocks6_seed_range_d1": [lo_d1, hi_d1],
                "d1_inside_6block_seed_range": lo_d1 <= vb["d1"] <= hi_d1}
    res["blocks5_vs_6block_seed_range"] = arch
    res["interpretation_limit"] = (
        "One 5-block seed against two 6-block seeds cannot separate an "
        "architecture effect from a seed effect. 'Inside the 6-block seed range' "
        "is a descriptive statement about two seeds, not a test.")

    (out / "comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    def fmt(blocks, seed, at):
        def f(e, k):
            return "{:.4f}".format(at[str(e)][k]) if at.get(str(e)) else "n/a"
        return "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            blocks, seed, f(10, "epe"), f(50, "epe"), f(100, "epe"),
            f(150, "epe"), f(200, "epe"), f(200, "d1"))

    print("\n| Blocks | Seed | EPE@10 | EPE@50 | EPE@100 | EPE@150 | EPE@200 | D1@200 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm, seed in BATCH1:
        r = res["runs"][run_id(arm, seed)]
        print(fmt(r["blocks"], seed, r["at_epoch"]))
    print("\n40-scene at epoch 200:")
    for arm, seed in BATCH1:
        r = res["runs"][run_id(arm, seed)]
        if "eval40" in r:
            print("  {} blocks seed {}:  EPE {:.4f}  D1 {:.4f}  RMSE {:.4f}".format(
                r["blocks"], seed, r["eval40"]["epe"], r["eval40"]["d1"],
                r["eval40"]["rmse"]))
    print("\nseed-0 frozen 40-scene records (read-only, NOT retrained):")
    for a in ("A", "B", "C"):
        print("  {} blocks seed 0:  EPE {:.4f}  D1 {:.4f}".format(
            SEED0[a]["blocks"], SEED0[a]["frozen_40_scene"]["epe"],
            SEED0[a]["frozen_40_scene"]["d1"]))
    print("\nseed spread at 6 blocks (the reference scale):")
    print(json.dumps(seed_spread, indent=2))
    print("\n5 blocks seed 1 vs the 6-block seed range:")
    print(json.dumps(arch, indent=2))
    print("\n" + res["interpretation_limit"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "train", "evaluate40", "probes", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name in ("train", "evaluate40"):
            p.add_argument("--arm", required=True, choices=list(e3b.ARMS))
            p.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()
    return {"preflight": cmd_preflight, "train": cmd_train,
            "evaluate40": cmd_evaluate40, "probes": cmd_probes,
            "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
