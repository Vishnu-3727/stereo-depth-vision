"""EXP-E1-ERROR-PARTITION-003-HELDOUT-ORACLE -- the partition, without the leak.

E1 has produced two oracles that disagree by an order of magnitude:

  * `EXP-E1-ERROR-PARTITION-001` -- affine in ground truth, one slope and offset
    per model. Removes **7.8-12.1 %** of pooled EPE. It is the smoothest possible
    oracle and therefore a **lower bound** on what a perfect matching stage buys.
  * `EXP-E1-ERROR-PARTITION-002-...-RUN3` -- per-pixel fixed point, properly
    scaled. Removes **61.4-69.6 %** and is still descending at iteration 8. But it
    tunes `disparity_initial` at exactly the `gt > 0` pixels it is then scored on,
    so it is an **upper bound**, and a leaky one.

Neither is the answer, and the pre-registered verdict rule returns
DOWNSTREAM-LIMITED on one and MIXED/MATCHING-LIMITED on the other. This run
removes the leak.

    python phase2/scripts/exp_e1_heldout_oracle.py run

**Pre-registered before running** (this docstring is the registration; the record
is `EXP-E1-ERROR-PARTITION-003-HELDOUT-ORACLE`):

  * The `gt > 0` pixels of each scene are split 50/50 at random, seed 0, into a
    FIT half and an EVAL half, fixed for all models.
  * The fixed-point search updates `disparity_initial` using **only the FIT
    half**: `initial <- clamp(initial + a*(gt - final), 0, 11)` there, the
    model's own value everywhere else. `a` is the candidates-per-pixel slope
    measured in run 001; alpha = 1.0, the best damping in RUN3.
  * 16 iterations, because RUN3 was still descending at 8.
  * EPE/D1 are pooled separately over the FIT half and the EVAL half at every
    iteration. **The verdict reads the EVAL half only.**
  * Verdict rule as frozen in `EXP_E1_ERROR_PARTITION_001_PREREGISTRATION.md` §4,
    applied to the best EVAL-half iterate: ratio <= 0.35 MATCHING-LIMITED,
    >= 0.65 DOWNSTREAM-LIMITED, otherwise MIXED. Per seed; overall only if all
    three agree.
  * Gates: finite throughout; iteration 0 reproduces the model as trained on both
    halves; and the search must improve the FIT half (otherwise the step is
    mis-scaled and the run is VOID, the gate RUN2 lacked).

What this still is not, stated in advance: refinement is convolutional, so a
value corrected at a FIT pixel propagates to its EVAL neighbours. The EVAL-half
number is therefore still an **upper** bound on what a realizable matching stage
buys -- it is bounded above by "an oracle that knows half the ground truth
densely" -- but it is no longer self-scoring. `UNKNOWN` -- what a refinement
stage retrained against such an input would do.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase2.scripts.exp_e1_error_partition import (  # noqa: E402
    DOWNSTREAM_LIMITED_MIN_RATIO, MATCHING_LIMITED_MAX_RATIO, MODELS, Pool, SCENES,
)
from phase2.scripts.exp_e1_inverse_oracle import CANDIDATE_MAX, CANDIDATE_MIN  # noqa: E402
from phase2.scripts.exp_h2_seed_replication import EXPERIMENTS_DIR  # noqa: E402
from phase2.scripts.h1_mechanism_probe import (  # noqa: E402
    _finish, _forward_parts, load_model,
)
from phase2.viz import core  # noqa: E402
from src.common.experiment import Experiment  # noqa: E402

EXPERIMENT_ID = "EXP-E1-ERROR-PARTITION-003-HELDOUT-ORACLE"
RESULT_DIR = REPO_ROOT / "phase2" / "results" / "EXP-E1-ERROR-PARTITION-001"
RUN_001 = RESULT_DIR / "partition.json"
ITERATIONS = 16
ALPHA = 1.0
SPLIT_SEED = 0


def split_masks(scene) -> tuple:
    """Half the ground-truth pixels to steer with, half to be judged on."""
    valid = scene.gt_valid
    rng = np.random.default_rng(SPLIT_SEED)
    coin = rng.random(valid.shape) < 0.5
    return valid & coin, valid & ~coin


def heldout_oracle(model_key: str, device: str, scenes, slope: float) -> dict:
    model = load_model(model_key, device)
    fit_pools = [Pool() for _ in range(ITERATIONS + 1)]
    eval_pools = [Pool() for _ in range(ITERATIONS + 1)]
    finite = True
    for scene in scenes:
        gt = scene.gt_disparity.astype(np.float64)
        fit_mask, eval_mask = split_masks(scene)
        left, initial = _forward_parts(model, scene, device)
        current = initial.clone()
        for step in range(ITERATIONS + 1):
            final = _finish(model, left, current)
            fit_pools[step].add(final, gt, fit_mask)
            eval_pools[step].add(final, gt, eval_mask)
            if not np.all(np.isfinite(final)):
                finite = False
            if step == ITERATIONS:
                break
            update = np.where(fit_mask, ALPHA * slope * (gt - final), 0.0)
            nxt = current[0, 0].cpu().numpy().astype(np.float64) + update
            nxt = np.clip(nxt, CANDIDATE_MIN, CANDIDATE_MAX)
            current = torch.from_numpy(nxt.astype(np.float32)).to(initial).view_as(initial)
    return {"slope": slope, "finite": finite,
            "fit_trace": [p.result() for p in fit_pools],
            "eval_trace": [p.result() for p in eval_pools]}


def verdict_for(trace: list) -> dict:
    as_trained = trace[0]["epe"]
    best = min(range(len(trace)), key=lambda i: trace[i]["epe"])
    ratio = trace[best]["epe"] / as_trained
    label = ("MATCHING-LIMITED" if ratio <= MATCHING_LIMITED_MAX_RATIO
             else "DOWNSTREAM-LIMITED" if ratio >= DOWNSTREAM_LIMITED_MIN_RATIO
             else "MIXED")
    return {"as_trained_epe": as_trained, "best_iteration": best,
            "best_epe": trace[best]["epe"], "best_d1": trace[best]["d1"],
            "epe_ratio": ratio, "error_removed_fraction": 1.0 - ratio,
            "verdict": label}


def table(results: dict) -> str:
    lines = ["| model | iteration | FIT EPE | FIT D1 % | EVAL EPE | EVAL D1 % |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in MODELS:
        for i, (f, e) in enumerate(zip(results[key]["fit_trace"],
                                       results[key]["eval_trace"])):
            lines.append("| {} | {} | {:.3f} | {:.2f} | {:.3f} | {:.2f} |".format(
                key, i, f["epe"], f["d1"], e["epe"], e["d1"]))
    return "\n".join(lines)


def run(device: str) -> None:
    config = {
        "experiment": EXPERIMENT_ID,
        "hypothesis": ("With the ground truth that steers the oracle held apart "
                       "from the ground truth it is scored on, how much of H2's "
                       "error would a perfect matching stage remove?"),
        "preregistration": "the module docstring of phase2/scripts/exp_e1_heldout_oracle.py",
        "parent_runs": ["EXP-E1-ERROR-PARTITION-001",
                        "EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE-RUN3"],
        "measurement_only": True,
        "training": "none -- trained H2 checkpoints, inference and substitution only",
        "models": list(MODELS),
        "method": ("gt>0 pixels split 50/50 (seed {}); initial <- clamp(initial + "
                   "{}*a*(gt-final), 0, 11) on the FIT half only, {} iterations; "
                   "EPE/D1 pooled separately on both halves; the verdict reads the "
                   "EVAL half".format(SPLIT_SEED, ALPHA, ITERATIONS)),
        "verdict_thresholds": {
            "matching_limited_max_epe_ratio": MATCHING_LIMITED_MAX_RATIO,
            "downstream_limited_min_epe_ratio": DOWNSTREAM_LIMITED_MIN_RATIO},
        "dataset": "kitti2015",
        "split": "hailo_val, all 40 scenes",
        "resolution": "368x1232 full frames",
        "crop": None,
        "disparity_range": 12,
        "batch_size": 1,
        "precision": "fp32",
        "seed": [0, 1, 2],
        "known_limitation": ("refinement is convolutional, so a corrected FIT "
                             "pixel propagates to its EVAL neighbours; the EVAL "
                             "number remains an upper bound on what a realizable "
                             "matching stage buys, and says nothing about a "
                             "refinement stage retrained against such an input."),
    }
    if (EXPERIMENTS_DIR / EXPERIMENT_ID).exists():
        raise SystemExit(EXPERIMENT_ID + " already exists -- records are never overwritten.")

    parent = json.loads(RUN_001.read_text(encoding="utf-8"))
    scenes = [core.load_scene(i) for i in SCENES]
    hashes_before = {k: core.sha256(core.checkpoint_path(k)) for k in MODELS}

    with Experiment("E1 partition with a held-out oracle", config=config,
                    experiment_id=EXPERIMENT_ID, experiments_dir=EXPERIMENTS_DIR) as exp:
        exp.note(config["hypothesis"])
        exp.note(config["known_limitation"])

        t0 = time.time()
        results, verdicts, fit_verdicts = {}, {}, {}
        for key in MODELS:
            slope = parent["affine_fits"][key]["a"]
            results[key] = heldout_oracle(key, device, scenes, slope)
            verdicts[key] = verdict_for(results[key]["eval_trace"])
            fit_verdicts[key] = verdict_for(results[key]["fit_trace"])
            exp.log("{:<6} FIT  EPE trace {}".format(key, "  ".join(
                "{:.3f}".format(r["epe"]) for r in results[key]["fit_trace"])))
            exp.log("{:<6} EVAL EPE trace {}".format(key, "  ".join(
                "{:.3f}".format(r["epe"]) for r in results[key]["eval_trace"])))
            exp.log("{} EVAL best iteration {} ratio {:.3f} -> {}".format(
                key, verdicts[key]["best_iteration"], verdicts[key]["epe_ratio"],
                verdicts[key]["verdict"]))

        hashes_after = {k: core.sha256(core.checkpoint_path(k)) for k in MODELS}
        integrity = hashes_before == hashes_after
        gate = {
            "all_finite": all(results[k]["finite"] for k in MODELS),
            "iteration_0_matches_as_trained": all(
                abs(results[k]["fit_trace"][0]["epe"]
                    - results[k]["eval_trace"][0]["epe"]) < 0.05 for k in MODELS),
            "search_improves_fit_half": all(
                fit_verdicts[k]["best_iteration"] > 0 for k in MODELS),
            "checkpoints_unchanged": integrity,
        }
        gate["PASS"] = all(gate.values())
        labels = {verdicts[k]["verdict"] for k in MODELS}
        overall = (labels.pop() if len(labels) == 1 else "NO CONSISTENT VERDICT") \
            if gate["PASS"] else "VOID -- a pre-registered gate failed"

        exp.metric("results", results)
        exp.metric("verdicts_eval_half", verdicts)
        exp.metric("verdicts_fit_half", fit_verdicts)
        exp.metric("gates", gate)
        exp.metric("overall_verdict", overall)
        exp.metric("checkpoint_integrity", {"before": hashes_before,
                                            "after": hashes_after, "unchanged": integrity})
        exp.metric("wall_clock_s", time.time() - t0)

        md = table(results)
        exp.path("heldout_oracle_table.md").write_text(md + "\n", encoding="utf-8")
        (RESULT_DIR / "heldout_oracle.json").write_text(json.dumps(
            {"results": results, "verdicts_eval_half": verdicts,
             "verdicts_fit_half": fit_verdicts, "gates": gate,
             "overall_verdict": overall}, indent=2), encoding="utf-8")

        print("\n" + md + "\n")
        for key in MODELS:
            v, f = verdicts[key], fit_verdicts[key]
            print("{:<6} FIT  best {:.3f} (ratio {:.3f})   EVAL best {:.3f} "
                  "(ratio {:.3f}, iteration {})  -> {}".format(
                      key, f["best_epe"], f["epe_ratio"], v["best_epe"],
                      v["epe_ratio"], v["best_iteration"], v["verdict"]))
        print("\nOVERALL (EVAL half): " + overall)
        exp.conclude("Held-out oracle removes {} of pooled EVAL-half EPE; overall "
                     "verdict {}.".format(", ".join("{:.1%}".format(
                         verdicts[k]["error_removed_fraction"]) for k in MODELS), overall))
        print("\nrecorded as " + exp.id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"], nargs="?", default="run")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    run(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
