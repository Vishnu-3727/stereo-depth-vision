"""EXP-BLOCKCOUNT-FULL-001 BATCH 2 -- ONE resolution run: 5 blocks, seed 2.

Batch 1 (`phase2/factorial/block_count_full/20260910T005550Z/`) measured the
200-epoch 6-block seed range on the authoritative 40-scene protocol as
2.2955-2.3982 EPE (spread 0.1027 px) and put 5b/seed1 at 2.4730 -- above every
6-block draw, but by less than the 6-block spread itself, and inside the
6-block D1 range. Two 5-block draws cannot separate architecture from seed.

The research lead authorised exactly ONE further run to resolve it:

    5 blocks, seed 2, 200 epochs

That brings the 5-block arm to three seeds {0, 1, 2}, matching the 6-block
arm's three seeds on the same protocol. NOTHING else is authorised -- not
4b/seed1, not 4b/seed2, not another 5b or 6b seed, not seed 0, not a rerun.
`BATCH2` below is the guard and `cmd_train` refuses everything outside it.

    python .../exp_blockcount_batch2.py preflight  --out DIR
    python .../exp_blockcount_batch2.py train      --out DIR
    python .../exp_blockcount_batch2.py evaluate40 --out DIR
    python .../exp_blockcount_batch2.py probes     --out DIR
    python .../exp_blockcount_batch2.py compare    --out DIR

The training/evaluation/probe machinery is imported unchanged from
`exp_blockcount_full` (which imports it unchanged from the frozen E3b/H2
recipe). This module contributes the batch-2 guard, a preflight whose checks
match a one-run batch, and the three-seed comparison. Batch 1's script and
record directory are never written.

Run identity stays in the same family: `EXP-BLOCKCOUNT-FULL-001-ARM-B-SEED2`,
written to this batch's own timestamped directory.
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

import phase2.factorial.block_count_full.exp_blockcount_full as full  # noqa: E402

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

import phase2.scripts.exp_e3b_refinement_capacity as e3b        # noqa: E402
from phase2.scripts.exp_h2_seed_replication import measure_macs  # noqa: E402

from phase2.factorial.block_count_seed_screen.exp_blockcount_seed_screen import (  # noqa: E402
    CUBLAS_CONFIG, SEED0_INIT_SHA, FrozenCosineAnnealingLR,
    enable_determinism, environment, provenance, weight_sha,
)

# THE AUTHORISED BATCH -- one run. Everything else is refused.
BATCH2 = (("B", 2),)
BATCH2_LABEL = ("batch 2 of the sequential campaign: ONE resolution run, "
                "5 blocks / seed 2 / 200 epochs, to bring the 5-block arm to "
                "three seeds {0,1,2} against the 6-block arm's three. No other "
                "run is authorised -- not 4b/seed1, not 4b/seed2, not another "
                "5b or 6b seed, not seed 0, not a rerun of anything")

FULL_EPOCHS = full.FULL_EPOCHS               # 200
PROBE_EPOCHS = full.PROBE_EPOCHS             # 10, 20, 50, 100, 150, 200

# Install the batch-2 identity into the shared module. Every function imported
# from `full` reads these at call time, so the guard, the recorded config text
# and the probe loop all follow.
full.BATCH1 = BATCH2
full.BATCH1_LABEL = BATCH2_LABEL

# --------------------------------------------------------------- frozen records
# READ-ONLY. Not retrained, not overwritten. Pooled 40-scene figures at epoch 200.
FROZEN_40 = {
    (6, 0): {"epe": 2.2955182, "d1": 15.6045143,
             "eval40": "phase2/diagnostics/determinism/20260909T041500Z_baseline/eval40_run_A.json"},
    (6, 1): {"epe": 2.3982217, "d1": 18.3357145,
             "eval40": "phase2/factorial/block_count_full/20260910T005550Z/"
                       "eval40_EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1.json"},
    (6, 2): {"epe": 2.3950488, "d1": 17.4699044,
             "eval40": "phase2/factorial/block_count_full/20260910T005550Z/"
                       "eval40_EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2.json"},
    (5, 0): {"epe": 2.6457595, "d1": 21.3711907,
             "eval40": "phase2/factorial/block_count_deterministic/20260909T131500Z/eval40_arm_B.json"},
    (5, 1): {"epe": 2.4729830, "d1": 18.2061782,
             "eval40": "phase2/factorial/block_count_full/20260910T005550Z/"
                       "eval40_EXP-BLOCKCOUNT-FULL-001-ARM-B-SEED1.json"},
    (4, 0): {"epe": 2.3387143, "d1": 16.9797389,
             "eval40": "phase2/factorial/block_count_deterministic/20260909T131500Z/eval40_arm_C.json"},
}

ARM, SEED = BATCH2[0]
BLOCKS = 5


def run_id() -> str:
    return full.run_id(ARM, SEED)


# ------------------------------------------------------------------- preflight
def cmd_preflight(args) -> int:
    out = Path(args.out).resolve()
    full.repoint(out, None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controls = enable_determinism()

    e3b.SEED = SEED
    ref = e3b.build_arm("A", device)          # same seed, untruncated six-block stack
    model = e3b.build_arm(ARM, device)
    identical_ok = True
    for position, index in enumerate(e3b.ARMS[ARM]["keep"]):
        a = ref.refinement.blocks[index].state_dict()
        b = model.refinement.blocks[position].state_dict()
        identical_ok = identical_ok and all(torch.equal(a[k], b[k]) for k in a)
    for name in ("feature_extractor", "aggregation"):
        a, b = getattr(ref, name).state_dict(), getattr(model, name).state_dict()
        identical_ok = identical_ok and all(torch.equal(a[k], b[k]) for k in a)
    for name in ("input_conv", "output_conv"):
        a = getattr(ref.refinement, name).state_dict()
        b = getattr(model.refinement, name).state_dict()
        identical_ok = identical_ok and all(torch.equal(a[k], b[k]) for k in a)

    init = weight_sha(model)
    train_base_len = None
    try:
        from phase2.scripts.exp_h1_cost_volume import CroppedKitti
        from src.datasets.kitti2015 import Kitti2015Stereo
        tb = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
        cropped = CroppedKitti(tb, seed=SEED)
        train_base_len = len(cropped)
        sample = cropped[0]
        crop_shape = tuple(sample[0].shape)
    except Exception as exc:                                    # pragma: no cover
        crop_shape = "unavailable: {}".format(exc)

    run = {
        "arm": ARM, "seed": SEED,
        "blocks": len(model.refinement.blocks),
        "blocks_expected": e3b.ARMS[ARM]["blocks"],
        "dilations": [int(b.conv1.dilation[0]) for b in model.refinement.blocks],
        "dilations_expected": list(e3b.ARMS[ARM]["dilations"]),
        "parameters": model.parameter_count(),
        "parameters_expected": e3b.ARMS[ARM]["parameters"],
        "parameters_match": model.parameter_count() == e3b.ARMS[ARM]["parameters"],
        "init_weight_sha": init,
        "init_differs_from_seed0": init != SEED0_INIT_SHA,
        "surviving_weights_bit_identical_within_seed": identical_ok,
        "cost_volume_shift": model.cost_volume.shift,
        "num_disparities": model.cost_volume.num_disparities,
        "regression": type(model.regression).__name__,
        "epoch_budget": e3b.EPOCHS,
        "loaded_from_stage_b_checkpoint": False,
        "loaded_from_any_checkpoint": False,
        "train_split": "hailo_calib",
        "val_split": "hailo_val",
        "train_scenes": train_base_len,
        "crop_shape": crop_shape,
        "batch_size": 2,
        "optimizer": "Adam(lr=1e-3, betas=(0.9, 0.999))",
        "macs_256x512": measure_macs(e3b.build_arm(ARM, device), device),
    }
    del model, ref

    probe_model = e3b.build_arm(ARM, device)
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
        "experiment": full.EXPERIMENT_PREFIX,
        "batch": 2,
        "purpose": ("ONE resolution run: 5 blocks / seed 2 / 200 epochs, bringing "
                    "the 5-block arm to three seeds against the 6-block arm's three"),
        "authorised_batch": [list(p) for p in BATCH2],
        "batch_note": BATCH2_LABEL,
        "provenance": provenance(), "environment": environment(),
        "deterministic_controls": controls,
        "schedule": schedule,
        "frozen_records_readonly": {"{}b_seed{}".format(b, s): v
                                    for (b, s), v in FROZEN_40.items()},
        "runs": {run_id(): run},
    }
    result["checks"] = {
        "phase_1_diff_empty": result["provenance"]["phase_1_diff_vs_frozen"] == "",
        "deterministic_controls_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
            and torch.backends.cudnn.deterministic
            and not torch.backends.cudnn.benchmark
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_CONFIG),
        "seed_is_2": run["seed"] == 2,
        "block_count_is_5": run["blocks"] == 5,
        "dilations_are_1_2_4_8_1": run["dilations"] == [1, 2, 4, 8, 1],
        "dilations_match_registration": run["dilations"] == run["dilations_expected"],
        "shift_is_left": run["cost_volume_shift"] == "left",
        "readout_is_standardised": run["regression"] == "StandardisedDisparityRegression",
        "candidates_are_12": run["num_disparities"] == 12,
        "parameter_count_matches_registration": run["parameters_match"],
        "epoch_budget_is_200": run["epoch_budget"] == 200,
        "schedule_T_max_is_200": schedule["T_max_actually_used"] == 200,
        "schedule_not_refitted": (schedule["T_max_requested_by_training_code"]
                                  == FULL_EPOCHS),
        "initialisation_is_fresh_not_stage_b": run["init_differs_from_seed0"],
        "no_checkpoint_loaded": not run["loaded_from_any_checkpoint"],
        "optimizer_state_fresh": bool(schedule["optimizer_state_empty_at_construction"]),
        "surviving_weights_bit_identical_within_seed": bool(identical_ok),
        "train_split_is_hailo_calib": run["train_split"] == "hailo_calib",
        "val_split_is_hailo_val": run["val_split"] == "hailo_val",
        "batch_size_is_2": run["batch_size"] == 2,
        "run_count_is_1": len(result["runs"]) == 1,
        "batch_is_B2_only": sorted(BATCH2) == [("B", 2)],
        "seed_0_not_scheduled": run["seed"] != 0,
        "four_block_not_scheduled": all(a != "C" for a, _ in BATCH2),
        "six_block_not_scheduled": all(a != "A" for a, _ in BATCH2),
    }
    result["checks"]["PASS"] = all(v for v in result["checks"].values()
                                   if isinstance(v, bool))
    (e3b.RESULT_DIR / "preflight.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    (out / "preflight.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"schedule": schedule,
                      "run": {k: run[k] for k in
                              ("arm", "seed", "blocks", "dilations", "parameters",
                               "init_weight_sha", "init_differs_from_seed0",
                               "epoch_budget", "train_scenes", "crop_shape",
                               "batch_size", "optimizer")},
                      "checks": result["checks"]}, indent=2, default=str))
    if not result["checks"]["PASS"]:
        print("\nPREFLIGHT FAILED -- STOP. Nothing was trained.")
        return 1
    print("\nPREFLIGHT PASS.")
    return 0


# ------------------------------------------------------- thin argument adapters
class _Args:
    def __init__(self, out, arm=ARM, seed=SEED):
        self.out, self.arm, self.seed = out, arm, seed


def cmd_train(args) -> int:
    return full.cmd_train(_Args(args.out))


def cmd_evaluate40(args) -> int:
    return full.cmd_evaluate40(_Args(args.out))


def cmd_probes(args) -> int:
    return full.cmd_probes(_Args(args.out))


# --------------------------------------------------------------------- compare
def _stats(values):
    a = np.array(values, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "min": float(a.min()), "max": float(a.max()),
            "range": float(a.max() - a.min())}


def cmd_compare(args) -> int:
    out = Path(args.out).resolve()
    full.repoint(out, None)

    new = json.loads((out / "eval40_{}.json".format(run_id())).read_text(encoding="utf-8"))
    records = {(5, 2): {"epe": new["pooled"]["epe"], "d1": new["pooled"]["d1"],
                        "rmse": new["pooled"]["rmse"], "per_scene": new["per_scene"],
                        "source": "this batch"}}
    for (b, s), meta in FROZEN_40.items():
        doc = json.loads((REPO_ROOT / meta["eval40"]).read_text(encoding="utf-8"))
        records[(b, s)] = {"epe": doc["pooled"]["epe"], "d1": doc["pooled"]["d1"],
                           "rmse": doc["pooled"].get("rmse"),
                           "per_scene": doc["per_scene"],
                           "source": meta["eval40"]}

    six = [records[(6, s)] for s in (0, 1, 2)]
    five = [records[(5, s)] for s in (0, 1, 2)]
    ranges = {
        "6_blocks": {"seeds": {str(s): {"epe": records[(6, s)]["epe"],
                                        "d1": records[(6, s)]["d1"]} for s in (0, 1, 2)},
                     "epe": _stats([r["epe"] for r in six]),
                     "d1": _stats([r["d1"] for r in six])},
        "5_blocks": {"seeds": {str(s): {"epe": records[(5, s)]["epe"],
                                        "d1": records[(5, s)]["d1"]} for s in (0, 1, 2)},
                     "epe": _stats([r["epe"] for r in five]),
                     "d1": _stats([r["d1"] for r in five])},
        "4_blocks_seed0_only": {"epe": records[(4, 0)]["epe"], "d1": records[(4, 0)]["d1"]},
    }

    epe_sep = ranges["5_blocks"]["epe"]["min"] > ranges["6_blocks"]["epe"]["max"]
    d1_sep = ranges["5_blocks"]["d1"]["min"] > ranges["6_blocks"]["d1"]["max"]
    pairings = {}
    for s5 in (0, 1, 2):
        for s6 in (0, 1, 2):
            pairings["5b_seed{}_vs_6b_seed{}".format(s5, s6)] = {
                "epe_5b_worse": records[(5, s5)]["epe"] > records[(6, s6)]["epe"],
                "d1_5b_worse": records[(5, s5)]["d1"] > records[(6, s6)]["d1"],
                "delta_epe": records[(5, s5)]["epe"] - records[(6, s6)]["epe"],
                "delta_d1": records[(5, s5)]["d1"] - records[(6, s6)]["d1"]}

    # Paired per-scene comparison: every model saw the same 40 scenes.
    def paired(a, b):
        ea = {r["scene"]: r["epe"] for r in a["per_scene"]}
        eb = {r["scene"]: r["epe"] for r in b["per_scene"]}
        da = {r["scene"]: r["d1"] for r in a["per_scene"]}
        db = {r["scene"]: r["d1"] for r in b["per_scene"]}
        keys = sorted(set(ea) & set(eb))
        de = np.array([ea[k] - eb[k] for k in keys])
        dd = np.array([da[k] - db[k] for k in keys])
        return {"scenes": len(keys),
                "epe_5b_worse_in": int((de > 0).sum()),
                "epe_mean_delta": float(de.mean()),
                "epe_median_delta": float(np.median(de)),
                "epe_max_delta": float(de.max()), "epe_min_delta": float(de.min()),
                "d1_5b_worse_in": int((dd > 0).sum()),
                "d1_mean_delta": float(dd.mean())}

    per_scene = {"5b_seed2_vs_6b_seed2": paired(records[(5, 2)], records[(6, 2)]),
                 "5b_seed2_vs_6b_seed1": paired(records[(5, 2)], records[(6, 1)]),
                 "5b_seed2_vs_6b_seed0": paired(records[(5, 2)], records[(6, 0)]),
                 "5b_seed1_vs_6b_seed1": paired(records[(5, 1)], records[(6, 1)]),
                 "5b_seed0_vs_6b_seed0": paired(records[(5, 0)], records[(6, 0)])}

    res = {"experiment": full.EXPERIMENT_PREFIX, "batch": 2,
           "batch_note": BATCH2_LABEL,
           "new_run": {"id": run_id(), "blocks": 5, "seed": 2,
                       "epe": records[(5, 2)]["epe"], "d1": records[(5, 2)]["d1"],
                       "rmse": records[(5, 2)]["rmse"]},
           "seed_ranges": ranges,
           "non_overlap": {
               "min_5b_epe_gt_max_6b_epe": bool(epe_sep),
               "min_5b_epe": ranges["5_blocks"]["epe"]["min"],
               "max_6b_epe": ranges["6_blocks"]["epe"]["max"],
               "epe_margin": ranges["5_blocks"]["epe"]["min"] - ranges["6_blocks"]["epe"]["max"],
               "min_5b_d1_gt_max_6b_d1": bool(d1_sep),
               "min_5b_d1": ranges["5_blocks"]["d1"]["min"],
               "max_6b_d1": ranges["6_blocks"]["d1"]["max"],
               "d1_margin": ranges["5_blocks"]["d1"]["min"] - ranges["6_blocks"]["d1"]["max"]},
           "all_pairings": pairings,
           "all_nine_pairings_5b_worse_epe": all(p["epe_5b_worse"] for p in pairings.values()),
           "all_nine_pairings_5b_worse_d1": all(p["d1_5b_worse"] for p in pairings.values()),
           "per_scene_paired": per_scene,
           "interpretation_limit": (
               "Descriptive non-overlap over three seeds per arm on one dataset, "
               "one recipe and one deterministic protocol. Not a statistical test, "
               "and not a universal law about refinement capacity. No materiality "
               "band is used: the historical E3b/E3c band stays invalidated and no "
               "new threshold was invented."),
           "provenance": provenance()}
    (out / "comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n| Blocks | Seed |    EPE |      D1 | source |")
    print("|---:|---:|---:|---:|---|")
    for b in (6, 5):
        for s in (0, 1, 2):
            r = records[(b, s)]
            print("| {} | {} | {:.4f} | {:.4f} | {} |".format(
                b, s, r["epe"], r["d1"], "this batch" if (b, s) == (5, 2) else "frozen"))
    r = records[(4, 0)]
    print("| 4 | 0 | {:.4f} | {:.4f} | frozen (not retrained) |".format(r["epe"], r["d1"]))

    print("\nseed ranges:")
    print(json.dumps({k: {m: ranges[k][m] for m in ("epe", "d1")}
                      for k in ("6_blocks", "5_blocks")}, indent=2))
    print("\nnon-overlap:")
    print(json.dumps(res["non_overlap"], indent=2))
    print("\nall nine 5b-vs-6b pairings, 5b worse on EPE:",
          res["all_nine_pairings_5b_worse_epe"])
    print("all nine 5b-vs-6b pairings, 5b worse on D1: ",
          res["all_nine_pairings_5b_worse_d1"])
    print("\npaired per-scene:")
    print(json.dumps(per_scene, indent=2))
    print("\n" + res["interpretation_limit"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "train", "evaluate40", "probes", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        # --arm/--seed are accepted so a stray call is REFUSED by the guard
        # rather than silently retargeted.
        if name in ("train", "evaluate40"):
            p.add_argument("--arm", default=ARM)
            p.add_argument("--seed", default=SEED, type=int)
    args = ap.parse_args()
    if getattr(args, "arm", ARM) != ARM or int(getattr(args, "seed", SEED)) != SEED:
        print("STOP: batch 2 authorises only arm {} seed {} (5 blocks, seed 2). "
              "Refusing arm {} seed {}.".format(ARM, SEED, args.arm, args.seed))
        return 1
    return {"preflight": cmd_preflight, "train": cmd_train,
            "evaluate40": cmd_evaluate40, "probes": cmd_probes,
            "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
