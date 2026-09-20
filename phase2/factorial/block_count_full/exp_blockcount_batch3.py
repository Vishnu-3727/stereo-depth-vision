"""EXP-BLOCKCOUNT-FULL-001 BATCH 3 -- the final 4-block runs: seeds 1 and 2.

Batch 2 (`phase2/factorial/block_count_full/20260910T052736Z/`) closed the
5-block question with `5-BLOCK-PENALTY-AT-SEEDS-0-1-2`: on the authoritative
40-scene protocol every 5-block seed (2.6458 / 2.4730 / 2.6373) is worse on EPE
than every 6-block seed (2.2955 / 2.3982 / 2.3950), margin +0.0748 px against a
6-block seed range of 0.1027 px.

The 4-block arm rests on ONE seed -- 2.3387 -- which lies *inside* the 6-block
range. That is the same position the 5-block arm occupied after batch 1, and
batch 2 demonstrated that a two-point picture can misplace an arm's centre by
0.22 px. The research lead authorised exactly TWO further runs:

    4 blocks, seed 1, 200 epochs
    4 blocks, seed 2, 200 epochs

That brings the 4-block arm to three seeds {0, 1, 2}, matching both other arms
on the same protocol and the same deterministic controls. NOTHING else is
authorised -- not a 4b/seed0 rerun, not a 4b/seed3, not another 5b or 6b seed,
not a 3-block arm, not a dilation or candidate-count variant, not longer
training, not a changed recipe. `BATCH3` below is the guard and `cmd_train`
refuses everything outside it.

    python .../exp_blockcount_batch3.py preflight     --out DIR
    python .../exp_blockcount_batch3.py train         --out DIR --arm C --seed 1
    python .../exp_blockcount_batch3.py evaluate40    --out DIR --arm C --seed 1
    python .../exp_blockcount_batch3.py train         --out DIR --arm C --seed 2
    python .../exp_blockcount_batch3.py evaluate40    --out DIR --arm C --seed 2
    python .../exp_blockcount_batch3.py verify_prefix --out DIR
    python .../exp_blockcount_batch3.py probes        --out DIR
    python .../exp_blockcount_batch3.py compare       --out DIR

The training / evaluation / probe machinery is imported unchanged from
`exp_blockcount_full` (which imports it unchanged from the frozen E3b/H2
recipe). This module contributes: the batch-3 guard, a preflight sized for a
two-run batch that additionally checks each seed's initialisation against the
frozen 30-epoch seed screen, an exact prefix verifier, and the three-arm
comparison with the pre-registered decision rule applied mechanically.

Run identity stays in the same family: `EXP-BLOCKCOUNT-FULL-001-ARM-C-SEED{1,2}`,
written to this batch's own timestamped directory. Batch 1's and batch 2's
scripts and record directories are read, never written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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

# ------------------------------------------------------------ THE AUTHORISED BATCH
# Two runs. Everything else is refused before training.
BATCH3 = (("C", 1), ("C", 2))
BATCH3_LABEL = ("batch 3, the final batch of the sequential campaign: TWO runs, "
                "4 blocks / seed 1 and 4 blocks / seed 2, 200 epochs each, to "
                "bring the 4-block arm to three seeds {0,1,2} against the "
                "6-block and 5-block arms' three each. No other run is "
                "authorised -- not a 4b/seed0 rerun, not 4b/seed3, not another "
                "5b or 6b seed, not a 3-block arm, not a dilation, "
                "candidate-count, receptive-field, Scene Flow or pretrained "
                "variant, not longer training, not an altered recipe")

FULL_EPOCHS = full.FULL_EPOCHS               # 200
PROBE_EPOCHS = full.PROBE_EPOCHS             # 10, 20, 50, 100, 150, 200
ARM = "C"
BLOCKS = 4
DILATIONS = [1, 2, 4, 8]

# Install the batch-3 identity into the shared module. Every function imported
# from `full` reads these at call time, so the guard and the probe loop follow.
full.BATCH1 = BATCH3
full.BATCH1_LABEL = BATCH3_LABEL

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
    (5, 2): {"epe": 2.6372994, "d1": 20.6593989,
             "eval40": "phase2/factorial/block_count_full/20260910T052736Z/"
                       "eval40_EXP-BLOCKCOUNT-FULL-001-ARM-B-SEED2.json"},
    (4, 0): {"epe": 2.3387143, "d1": 16.9797389,
             "eval40": "phase2/factorial/block_count_deterministic/20260909T131500Z/eval40_arm_C.json"},
}

# ------------------------------------------- FROZEN 6-BLOCK REFERENCE RANGE (EPE)
# The pre-registered decision boundary. Read from the frozen 6-block eval40
# records at registration time; asserted against them again in `compare`.
SIX_BLOCK_EPE_MIN = 2.2955181809208267
SIX_BLOCK_EPE_MAX = 2.3982217171269924

# ------------------------------------------------- FROZEN SEED-SCREEN PREFIX DATA
# From EXP-BLOCKCOUNT-SEEDSCREEN-001 (phase2/factorial/block_count_seed_screen/
# 20260909T145659Z), the 30-epoch screen. The screen pinned T_max at 200 while
# shortening only the budget, so its first 30 epochs are a strict prefix of a
# 200-epoch run at the same seed. Under the deterministic protocol the new runs
# must reproduce these numbers EXACTLY. Frozen here before training.
# CORRECTION 2026-09-10, applied BEFORE `verify_prefix` ran and while seed 1 was
# still training: the values first frozen here were transcribed from a listing
# printed at 10 decimal places, and their trailing digits were wrong. They are
# replaced below by `repr()` of the authoritative screen records themselves --
# `block_count_seed_screen/20260909T145659Z/results/history_arm_C-SEED{1,2}.json`.
# This makes the check STRICTER and correct, not looser: the live seed-1 run had
# already reproduced the true value (epoch 2, 18.94565264092234) exactly. The
# correction is recorded in `CORRECTION_screen_prefix.md` in the record
# directory. No threshold, gate or decision boundary is affected -- SCREEN_PREFIX
# feeds only the initialisation-hash preflight check (whose SHA strings were
# always correct and passed) and `verify_prefix`.
SCREEN_PREFIX = {
    1: {"init_weight_sha": "2cac2916ed802495",
        "epoch0_mean_loss": 11.187826424837112,
        "checkpoints": {
            "1":  {"val_epe": 16.031864940965352, "val_d1": 89.06086644453445},
            "2":  {"val_epe": 18.94565264092234, "val_d1": 87.00300963707065},
            "10": {"val_epe": 18.36530086611511, "val_d1": 88.2830455679038},
            "20": {"val_epe": 12.078822974549746, "val_d1": 78.73558710487097}}},
    2: {"init_weight_sha": "e622226a5a22fc9b",
        "epoch0_mean_loss": 10.646129465103149,
        "checkpoints": {
            "1":  {"val_epe": 21.45397410346398, "val_d1": 89.77658199353804},
            "2":  {"val_epe": 17.84664041672191, "val_d1": 89.12050045289612},
            "10": {"val_epe": 16.342920881565185, "val_d1": 87.35554870271855},
            "20": {"val_epe": 15.506311573290349, "val_d1": 85.64163506780548}}},
}


def run_id(seed: int) -> str:
    return full.run_id(ARM, seed)


# --------------------------------------------------------------------- repoint
def repoint(out: Path, seed) -> None:
    """`full.repoint`, then correct the recorded batch text to batch 3.

    `full.repoint` stamps every recorded config with batch-1 wording. Batch 2
    inherited that wording verbatim, which left a stale `claim_ceiling` inside a
    batch-2 record. Batch 3 overrides the text fields that describe *this*
    batch. No numeric, architectural or recipe field is touched: they all still
    come from the frozen `e3b.arm_config`.
    """
    full.repoint(out, seed)
    inner = e3b.arm_config

    def batch3_arm_config(arm: str, device: str) -> dict:
        config = dict(inner(arm, device))
        config.update({
            "experiment_kind": (
                "200-EPOCH BLOCK-COUNT EXPERIMENT, BATCH 3 (FINAL) OF A "
                "SEQUENTIAL CAMPAIGN WITH A HARD REVIEW GATE. " + BATCH3_LABEL),
            "control": (
                "the frozen 200-epoch 40-scene records, READ-ONLY and NOT "
                "retrained: 6 blocks at seeds 0/1/2 (2.2955 / 2.3982 / 2.3950) "
                "and 5 blocks at seeds 0/1/2 (2.6458 / 2.4730 / 2.6373). The "
                "6-block three-seed EPE range [2.2955182, 2.3982217] is the "
                "pre-registered decision boundary and was fixed before this "
                "batch was launched."),
            "claim_ceiling": (
                "Batch 3 completes the 4-block arm at three seeds. The verdict "
                "is one of three pre-registered labels decided on 40-scene EPE "
                "alone. Three seeds per arm is descriptive evidence about this "
                "tested configuration, not a statistical test and not a "
                "universal law. An empirical block-count effect is NEVER to be "
                "restated as a parameter-count or receptive-field CAUSE."),
            "prior_batches": (
                "batch 1 phase2/factorial/block_count_full/20260910T005550Z "
                "(verdict CONTINUE-BATCH-2); batch 2 "
                "phase2/factorial/block_count_full/20260910T052736Z "
                "(verdict 5-BLOCK-PENALTY-AT-SEEDS-0-1-2)"),
        })
        return config

    e3b.arm_config = batch3_arm_config


# ------------------------------------------------------------------- preflight
def cmd_preflight(args) -> int:
    out = Path(args.out).resolve()
    repoint(out, None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    controls = enable_determinism()

    runs, fresh_ok, identical_ok, screen_ok = {}, True, True, True
    for arm, seed in BATCH3:
        e3b.SEED = seed
        ref = e3b.build_arm("A", device)       # same seed, untruncated six-block stack
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
        matches_screen = init == SCREEN_PREFIX[seed]["init_weight_sha"]
        screen_ok = screen_ok and matches_screen

        train_base_len, crop_shape = None, None
        try:
            from phase2.scripts.exp_h1_cost_volume import CroppedKitti
            from src.datasets.kitti2015 import Kitti2015Stereo
            tb = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_calib")
            cropped = CroppedKitti(tb, seed=seed)
            train_base_len = len(cropped)
            crop_shape = tuple(cropped[0][0].shape)
        except Exception as exc:                                # pragma: no cover
            crop_shape = "unavailable: {}".format(exc)

        runs[run_id(seed)] = {
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
            "init_matches_seed_screen": matches_screen,
            "init_weight_sha_expected_from_screen":
                SCREEN_PREFIX[seed]["init_weight_sha"],
            "surviving_weights_bit_identical_within_seed": ok,
            "cost_volume_shift": model.cost_volume.shift,
            "num_disparities": model.cost_volume.num_disparities,
            "regression": type(model.regression).__name__,
            "epoch_budget": e3b.EPOCHS,
            "loaded_from_any_checkpoint": False,
            "train_split": "hailo_calib",
            "val_split": "hailo_val",
            "train_scenes": train_base_len,
            "crop_shape": crop_shape,
            "batch_size": 2,
            "optimizer": "Adam(lr=1e-3, betas=(0.9, 0.999))",
            "macs_256x512": measure_macs(e3b.build_arm(arm, device), device),
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

    # Re-read the frozen 6-block range from the records rather than trusting the
    # constants: the decision boundary must equal what the records actually say.
    six = {}
    for s in (0, 1, 2):
        doc = json.loads((REPO_ROOT / FROZEN_40[(6, s)]["eval40"]).read_text(
            encoding="utf-8"))
        six[s] = doc["pooled"]["epe"]
    boundary_ok = (min(six.values()) == SIX_BLOCK_EPE_MIN
                   and max(six.values()) == SIX_BLOCK_EPE_MAX)

    result = {
        "experiment": full.EXPERIMENT_PREFIX,
        "batch": 3,
        "purpose": ("TWO final runs: 4 blocks / seed 1 and 4 blocks / seed 2, "
                    "200 epochs each, bringing the 4-block arm to three seeds "
                    "against the 6-block and 5-block arms' three each"),
        "authorised_batch": [list(p) for p in BATCH3],
        "batch_note": BATCH3_LABEL,
        "provenance": provenance(), "environment": environment(),
        "deterministic_controls": controls,
        "schedule": schedule,
        "frozen_six_block_epe_range": {
            "min": SIX_BLOCK_EPE_MIN, "max": SIX_BLOCK_EPE_MAX,
            "read_back_from_records": {str(s): six[s] for s in (0, 1, 2)},
            "note": ("the pre-registered decision boundary, fixed before this "
                     "batch was launched")},
        "frozen_records_readonly": {"{}b_seed{}".format(b, s): v
                                    for (b, s), v in FROZEN_40.items()},
        "seed_screen_prefix_expectations": SCREEN_PREFIX,
        "runs": runs,
    }
    result["checks"] = {
        "phase_1_diff_empty": result["provenance"]["phase_1_diff_vs_frozen"] == "",
        "deterministic_controls_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
            and torch.backends.cudnn.deterministic
            and not torch.backends.cudnn.benchmark
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_CONFIG),
        "seeds_are_1_and_2": sorted(r["seed"] for r in runs.values()) == [1, 2],
        "block_count_is_4": all(r["blocks"] == 4 for r in runs.values()),
        "dilations_are_1_2_4_8": all(r["dilations"] == DILATIONS
                                     for r in runs.values()),
        "dilations_match_registration": all(r["dilations"] == r["dilations_expected"]
                                            for r in runs.values()),
        "shift_is_left": all(r["cost_volume_shift"] == "left" for r in runs.values()),
        "readout_is_standardised": all(
            r["regression"] == "StandardisedDisparityRegression" for r in runs.values()),
        "candidates_are_12": all(r["num_disparities"] == 12 for r in runs.values()),
        "parameter_count_matches_registration": all(r["parameters_match"]
                                                    for r in runs.values()),
        "epoch_budget_is_200": all(r["epoch_budget"] == 200 for r in runs.values()),
        "schedule_T_max_is_200": schedule["T_max_actually_used"] == 200,
        "schedule_not_refitted": (schedule["T_max_requested_by_training_code"]
                                  == FULL_EPOCHS),
        "initialisation_is_fresh_not_seed0": fresh_ok,
        "initialisation_matches_seed_screen": screen_ok,
        "no_checkpoint_loaded": all(not r["loaded_from_any_checkpoint"]
                                    for r in runs.values()),
        "optimizer_state_fresh": bool(schedule["optimizer_state_empty_at_construction"]),
        "surviving_weights_bit_identical_within_seed": bool(identical_ok),
        "train_split_is_hailo_calib": all(r["train_split"] == "hailo_calib"
                                          for r in runs.values()),
        "val_split_is_hailo_val": all(r["val_split"] == "hailo_val"
                                      for r in runs.values()),
        "batch_size_is_2": all(r["batch_size"] == 2 for r in runs.values()),
        "run_count_is_2": len(runs) == 2,
        "batch_is_C1_and_C2_only": sorted(BATCH3) == [("C", 1), ("C", 2)],
        "seed_0_not_scheduled": all(s != 0 for _, s in BATCH3),
        "seed_3_not_scheduled": all(s != 3 for _, s in BATCH3),
        "five_block_not_scheduled": all(a != "B" for a, _ in BATCH3),
        "six_block_not_scheduled": all(a != "A" for a, _ in BATCH3),
        "decision_boundary_matches_frozen_records": bool(boundary_ok),
    }
    result["checks"]["PASS"] = all(v for v in result["checks"].values()
                                   if isinstance(v, bool))
    (e3b.RESULT_DIR / "preflight.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    (out / "preflight.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"schedule": schedule,
                      "six_block_boundary": result["frozen_six_block_epe_range"],
                      "runs": {k: {f: v[f] for f in
                                   ("arm", "seed", "blocks", "dilations",
                                    "parameters", "init_weight_sha",
                                    "init_matches_seed_screen", "epoch_budget",
                                    "train_scenes", "crop_shape", "batch_size",
                                    "optimizer")}
                               for k, v in runs.items()},
                      "checks": result["checks"]}, indent=2, default=str))
    n = sum(1 for v in result["checks"].values() if isinstance(v, bool)) - 1
    if not result["checks"]["PASS"]:
        print("\nPREFLIGHT FAILED -- STOP. Nothing was trained.")
        return 1
    print("\nPREFLIGHT PASS ({}/{} checks).".format(n, n))
    return 0


# ------------------------------------------------------- thin argument adapters
class _Args:
    def __init__(self, out, arm, seed):
        self.out, self.arm, self.seed = out, arm, int(seed)


def cmd_train(args) -> int:
    repoint(Path(args.out).resolve(), int(args.seed))
    return full.cmd_train(_Args(args.out, args.arm, args.seed))


def cmd_evaluate40(args) -> int:
    repoint(Path(args.out).resolve(), int(args.seed))
    return full.cmd_evaluate40(_Args(args.out, args.arm, args.seed))


def cmd_probes(args) -> int:
    return full.cmd_probes(_Args(args.out, ARM, 0))


# -------------------------------------------------------------- prefix verifier
def cmd_verify_prefix(args) -> int:
    """Exact comparison of each new run against the frozen 30-epoch seed screen.

    The screen shortened only the epoch BUDGET and pinned T_max at 200, so its
    first 30 epochs are a strict prefix of a 200-epoch run at the same seed.
    Under the deterministic protocol the reproduction must be exact -- not
    close. Any mismatch is an unexplained discrepancy and stops the batch.
    """
    out = Path(args.out).resolve()
    report = {"source_screen": ("phase2/factorial/block_count_seed_screen/"
                                "20260909T145659Z"),
              "expectations_frozen_before_training": True,
              "comparison": "exact float equality, not tolerance-based",
              "runs": {}}
    all_ok = True
    for _, seed in BATCH3:
        exp_id = run_id(seed)
        hist = json.loads((out / "results" / "history_arm_C-SEED{}.json".format(seed))
                          .read_text(encoding="utf-8"))
        meta = json.loads((out / "run_meta_{}.json".format(exp_id))
                          .read_text(encoding="utf-8"))
        want = SCREEN_PREFIX[seed]
        rows, ok = {}, True
        init_ok = meta["init_weight_sha"] == want["init_weight_sha"]
        ok = ok and init_ok
        loss0 = hist["history"][0]["mean_loss"]
        loss_ok = loss0 == want["epoch0_mean_loss"]
        ok = ok and loss_ok
        for epoch, exp in want["checkpoints"].items():
            got = hist["checkpoints"][epoch]
            e_ok = got["val_epe"] == exp["val_epe"]
            d_ok = got["val_d1"] == exp["val_d1"]
            ok = ok and e_ok and d_ok
            rows[epoch] = {"expected_val_epe": exp["val_epe"],
                           "observed_val_epe": got["val_epe"], "val_epe_exact": e_ok,
                           "expected_val_d1": exp["val_d1"],
                           "observed_val_d1": got["val_d1"], "val_d1_exact": d_ok}
        report["runs"][exp_id] = {
            "seed": seed,
            "init_weight_sha_expected": want["init_weight_sha"],
            "init_weight_sha_observed": meta["init_weight_sha"],
            "init_weight_sha_exact": init_ok,
            "epoch0_mean_loss_expected": want["epoch0_mean_loss"],
            "epoch0_mean_loss_observed": loss0,
            "epoch0_mean_loss_exact": loss_ok,
            "checkpoints": rows,
            "PREFIX_REPRODUCED_EXACTLY": ok}
        all_ok = all_ok and ok
        print("{}  prefix reproduced exactly: {}".format(exp_id, ok))
        for epoch, r in sorted(rows.items(), key=lambda kv: int(kv[0])):
            print("    epoch {:>2}  val_epe {:.10f} {}  val_d1 {:.10f} {}".format(
                int(epoch), r["observed_val_epe"],
                "OK" if r["val_epe_exact"] else "MISMATCH",
                r["observed_val_d1"],
                "OK" if r["val_d1_exact"] else "MISMATCH"))
    report["ALL_PREFIXES_REPRODUCED"] = all_ok
    report["provenance"] = provenance()
    (out / "prefix_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if not all_ok:
        print("\nPREFIX MISMATCH -- STOP. Preserve the runs and report; do not "
              "continue analysis under an unexplained discrepancy.")
        return 1
    print("\nALL PREFIXES REPRODUCED EXACTLY.")
    return 0


# --------------------------------------------------------------------- compare
def _stats(values):
    a = np.array(values, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "min": float(a.min()), "max": float(a.max()),
            "range": float(a.max() - a.min())}


def _stereo_from_metrics(path: Path):
    """Read the frozen epoch-200 stereo verdict out of an experiment record.

    The experiment recorder nests everything under a `metrics` object; the
    verdict is NOT at the document's top level. Reading the top level silently
    yields `{}` and therefore a false `STEREO_FUNCTIONAL=False` for every run,
    which is how the first `compare` of this batch produced a spurious
    `4-BLOCK-STEREO-FUNCTIONALITY-FAILURE`. Fixed 2026-09-10; see
    `CORRECTION_stereo_reader.md`. The stereo criteria themselves are the frozen
    ones and are untouched -- only the lookup path changed.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    metrics = doc.get("metrics", doc)
    v = metrics.get("stereo_verdict_epoch_200")
    if v is None:
        raise KeyError(
            "stereo_verdict_epoch_200 missing from {} -- refusing to treat an "
            "absent verdict as a stereo failure".format(path))
    return bool(v.get("STEREO_FUNCTIONAL")), v


def cmd_compare(args) -> int:
    out = Path(args.out).resolve()
    repoint(out, None)

    records = {}
    for _, seed in BATCH3:
        doc = json.loads((out / "eval40_{}.json".format(run_id(seed)))
                         .read_text(encoding="utf-8"))
        records[(4, seed)] = {"epe": doc["pooled"]["epe"], "d1": doc["pooled"]["d1"],
                              "rmse": doc["pooled"]["rmse"],
                              "per_scene": doc["per_scene"], "source": "this batch"}
    for (b, s), meta in FROZEN_40.items():
        doc = json.loads((REPO_ROOT / meta["eval40"]).read_text(encoding="utf-8"))
        records[(b, s)] = {"epe": doc["pooled"]["epe"], "d1": doc["pooled"]["d1"],
                           "rmse": doc["pooled"].get("rmse"),
                           "per_scene": doc["per_scene"], "source": meta["eval40"]}

    ranges = {}
    for b in (6, 5, 4):
        arm_r = [records[(b, s)] for s in (0, 1, 2)]
        ranges["{}_blocks".format(b)] = {
            "seeds": {str(s): {"epe": records[(b, s)]["epe"],
                               "d1": records[(b, s)]["d1"],
                               "rmse": records[(b, s)]["rmse"]} for s in (0, 1, 2)},
            "epe": _stats([r["epe"] for r in arm_r]),
            "d1": _stats([r["d1"] for r in arm_r])}

    # The decision boundary is the frozen 6-block range. Assert it still equals
    # what the frozen records say before it is used.
    assert ranges["6_blocks"]["epe"]["min"] == SIX_BLOCK_EPE_MIN
    assert ranges["6_blocks"]["epe"]["max"] == SIX_BLOCK_EPE_MAX

    four_epe = [records[(4, s)]["epe"] for s in (0, 1, 2)]

    stereo = {}
    for _, seed in BATCH3:
        ok, verdict = _stereo_from_metrics(
            out / "experiments" / run_id(seed) / "metrics.json")
        stereo[str(seed)] = {"STEREO_FUNCTIONAL": ok, "verdict": verdict,
                             "source": "this batch"}
    ok0, v0 = _stereo_from_metrics(
        REPO_ROOT / "phase2/factorial/block_count_deterministic/20260909T131500Z/"
        "experiments/EXP-BLOCKCOUNT-DETERMINISTIC-001-ARM-C/metrics.json")
    stereo["0"] = {"STEREO_FUNCTIONAL": ok0, "verdict": v0,
                   "source": "frozen deterministic record, read-only"}
    all_stereo = all(stereo[str(s)]["STEREO_FUNCTIONAL"] for s in (0, 1, 2))

    all_above = all(e > SIX_BLOCK_EPE_MAX for e in four_epe)
    all_inside = all(SIX_BLOCK_EPE_MIN <= e <= SIX_BLOCK_EPE_MAX for e in four_epe)

    if not all_stereo:
        verdict = "4-BLOCK-STEREO-FUNCTIONALITY-FAILURE"
    elif all_above:
        verdict = "4-BLOCK-PENALTY-AT-SEEDS-0-1-2"
    elif all_inside:
        verdict = "4-BLOCK-NO-PENALTY-AT-SEEDS-0-1-2"
    else:
        verdict = "4-BLOCK-INCONCLUSIVE-AT-SEEDS-0-1-2"

    pairings = {}
    for s4 in (0, 1, 2):
        for s6 in (0, 1, 2):
            pairings["4b_seed{}_vs_6b_seed{}".format(s4, s6)] = {
                "epe_4b_worse": records[(4, s4)]["epe"] > records[(6, s6)]["epe"],
                "d1_4b_worse": records[(4, s4)]["d1"] > records[(6, s6)]["d1"],
                "delta_epe": records[(4, s4)]["epe"] - records[(6, s6)]["epe"],
                "delta_d1": records[(4, s4)]["d1"] - records[(6, s6)]["d1"]}
    pairings_45 = {}
    for s4 in (0, 1, 2):
        for s5 in (0, 1, 2):
            pairings_45["4b_seed{}_vs_5b_seed{}".format(s4, s5)] = {
                "epe_4b_worse": records[(4, s4)]["epe"] > records[(5, s5)]["epe"],
                "delta_epe": records[(4, s4)]["epe"] - records[(5, s5)]["epe"]}

    def paired(a, b, label):
        ea = {r["scene"]: r["epe"] for r in a["per_scene"]}
        eb = {r["scene"]: r["epe"] for r in b["per_scene"]}
        da = {r["scene"]: r["d1"] for r in a["per_scene"]}
        db = {r["scene"]: r["d1"] for r in b["per_scene"]}
        keys = sorted(set(ea) & set(eb))
        de = np.array([ea[k] - eb[k] for k in keys])
        dd = np.array([da[k] - db[k] for k in keys])
        return {"pairing": label, "scenes": len(keys),
                "epe_first_worse_in": int((de > 0).sum()),
                "epe_mean_delta": float(de.mean()),
                "epe_median_delta": float(np.median(de)),
                "epe_max_delta": float(de.max()), "epe_min_delta": float(de.min()),
                "d1_first_worse_in": int((dd > 0).sum()),
                "d1_mean_delta": float(dd.mean())}

    per_scene = {}
    for s in (0, 1, 2):
        per_scene["4b_seed{0}_vs_6b_seed{0}".format(s)] = paired(
            records[(4, s)], records[(6, s)], "4b/s{0} vs 6b/s{0}".format(s))
        per_scene["4b_seed{0}_vs_5b_seed{0}".format(s)] = paired(
            records[(4, s)], records[(5, s)], "4b/s{0} vs 5b/s{0}".format(s))

    res = {"experiment": full.EXPERIMENT_PREFIX, "batch": 3,
           "batch_note": BATCH3_LABEL,
           "new_runs": {run_id(s): {"blocks": 4, "seed": s,
                                    "epe": records[(4, s)]["epe"],
                                    "d1": records[(4, s)]["d1"],
                                    "rmse": records[(4, s)]["rmse"]}
                        for _, s in BATCH3},
           "seed_ranges": ranges,
           "decision_rule": {
               "primary_metric": "40-scene pooled EPE",
               "frozen_six_block_range": [SIX_BLOCK_EPE_MIN, SIX_BLOCK_EPE_MAX],
               "boundary_fixed_before_launch": True,
               "verdict_A_requires": "all three 4b EPE > max(6b EPE)",
               "verdict_B_requires": "all three 4b EPE inside the 6b range",
               "verdict_C_otherwise": True,
               "d1_may_not_override_epe": True},
           "decision_inputs": {
               "four_block_epe_by_seed": {str(s): records[(4, s)]["epe"]
                                          for s in (0, 1, 2)},
               "all_four_block_epe_above_six_block_max": bool(all_above),
               "all_four_block_epe_inside_six_block_range": bool(all_inside),
               "all_four_block_runs_stereo_functional": bool(all_stereo)},
           "non_overlap": {
               "min_4b_epe": ranges["4_blocks"]["epe"]["min"],
               "max_4b_epe": ranges["4_blocks"]["epe"]["max"],
               "min_6b_epe": SIX_BLOCK_EPE_MIN, "max_6b_epe": SIX_BLOCK_EPE_MAX,
               "min_4b_epe_minus_max_6b_epe":
                   ranges["4_blocks"]["epe"]["min"] - SIX_BLOCK_EPE_MAX,
               "epe_ranges_disjoint_4b_above_6b":
                   bool(ranges["4_blocks"]["epe"]["min"] > SIX_BLOCK_EPE_MAX),
               "min_4b_d1": ranges["4_blocks"]["d1"]["min"],
               "max_4b_d1": ranges["4_blocks"]["d1"]["max"],
               "min_6b_d1": ranges["6_blocks"]["d1"]["min"],
               "max_6b_d1": ranges["6_blocks"]["d1"]["max"],
               "min_4b_d1_minus_max_6b_d1":
                   ranges["4_blocks"]["d1"]["min"] - ranges["6_blocks"]["d1"]["max"],
               "d1_ranges_disjoint_4b_above_6b":
                   bool(ranges["4_blocks"]["d1"]["min"]
                        > ranges["6_blocks"]["d1"]["max"])},
           "mean_differences": {
               "epe_4b_minus_6b": (ranges["4_blocks"]["epe"]["mean"]
                                   - ranges["6_blocks"]["epe"]["mean"]),
               "d1_4b_minus_6b": (ranges["4_blocks"]["d1"]["mean"]
                                  - ranges["6_blocks"]["d1"]["mean"]),
               "epe_4b_minus_5b": (ranges["4_blocks"]["epe"]["mean"]
                                   - ranges["5_blocks"]["epe"]["mean"]),
               "d1_4b_minus_5b": (ranges["4_blocks"]["d1"]["mean"]
                                  - ranges["5_blocks"]["d1"]["mean"])},
           "all_pairings_4b_vs_6b": pairings,
           "all_nine_pairings_4b_worse_epe": all(p["epe_4b_worse"]
                                                 for p in pairings.values()),
           "all_nine_pairings_4b_worse_d1": all(p["d1_4b_worse"]
                                                for p in pairings.values()),
           "all_pairings_4b_vs_5b": pairings_45,
           "all_nine_pairings_4b_better_than_5b_epe":
               all(not p["epe_4b_worse"] for p in pairings_45.values()),
           "per_scene_paired": per_scene,
           "stereo": stereo,
           "VERDICT": verdict,
           "interpretation_limit": (
               "Descriptive comparison of three seeds per arm on one dataset, "
               "one recipe and one deterministic protocol. Not a statistical "
               "test. An empirical block-count effect is not a parameter-count "
               "or receptive-field CAUSE. The historical E3b/E3c materiality "
               "band stays invalidated; no band was reused and no threshold was "
               "invented after seeing the result."),
           "provenance": provenance()}
    (out / "comparison.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("\n| Blocks | Seed |       EPE |        D1 |   RMSE | source |")
    print("|---:|---:|---:|---:|---:|---|")
    for b in (6, 5, 4):
        for s in (0, 1, 2):
            r = records[(b, s)]
            src = "this batch" if (b == 4 and s in (1, 2)) else "frozen"
            print("| {} | {} | {:.7f} | {:.7f} | {:.4f} | {} |".format(
                b, s, r["epe"], r["d1"],
                r["rmse"] if r["rmse"] is not None else float("nan"), src))
    print("\nseed ranges:")
    print(json.dumps({k: {m: ranges[k][m] for m in ("epe", "d1")} for k in ranges},
                     indent=2))
    print("\ndecision inputs:")
    print(json.dumps(res["decision_inputs"], indent=2))
    print("\nnon-overlap:")
    print(json.dumps(res["non_overlap"], indent=2))
    print("\nmean differences:")
    print(json.dumps(res["mean_differences"], indent=2))
    print("\nper-scene paired:")
    print(json.dumps(per_scene, indent=2))
    print("\nstereo functional: " + json.dumps(
        {s: stereo[s]["STEREO_FUNCTIONAL"] for s in ("0", "1", "2")}))
    print("\nVERDICT: " + verdict)
    print("\n" + res["interpretation_limit"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "train", "evaluate40", "verify_prefix", "probes",
                 "compare"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name in ("train", "evaluate40"):
            p.add_argument("--arm", default=ARM)
            p.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()
    if args.cmd in ("train", "evaluate40"):
        if (args.arm, int(args.seed)) not in BATCH3:
            print("STOP: batch 3 authorises only arm C seeds 1 and 2 "
                  "(4 blocks). Refusing arm {} seed {}.".format(
                      args.arm, args.seed))
            return 1
    return {"preflight": cmd_preflight, "train": cmd_train,
            "evaluate40": cmd_evaluate40, "verify_prefix": cmd_verify_prefix,
            "probes": cmd_probes, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
