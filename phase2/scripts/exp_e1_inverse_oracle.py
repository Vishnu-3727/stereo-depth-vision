"""EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE -- is E1's verdict an artifact of a
weak oracle?

`EXP-E1-ERROR-PARTITION-001` returned DOWNSTREAM-LIMITED on all three seeds: an
affine-in-ground-truth oracle at the matching -> refinement boundary removed only
7.8-12.1 % of pooled EPE. That verdict is only as strong as the oracle, and the
affine oracle has two visible weaknesses, both of which bias *toward* the verdict
it produced:

  1. the fit is `R^2` 0.86-0.89 with a residual of 0.40-0.69 candidates, so it is
     perfect only up to a linear convention;
  2. `a*gt + b` can leave the [0, 11] interval the soft-argmin can actually
     output, so some substituted pixels are off-manifold inputs the refinement
     stage never saw in training.

This run replaces the assumed convention with a **targeted** oracle that needs no
convention at all. Refinement is a residual stage, `final = initial +
refine(initial, left)`, so the input that would make the output correct can be
searched for directly by fixed-point iteration:

    initial <- clamp(initial + alpha * a * (gt - final), 0, 11)   # only where gt > 0

where `a` is the candidate-per-pixel slope run 001 already measured for this
model (one candidate is ~10-16 px, so an unscaled pixel-valued step overshoots by
an order of magnitude -- that mistake is recorded in the NOTE.md of
`EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE`, whose diverging trace produced a
*false* confirmation of E1's verdict).

If even this cannot drive the final map to ground truth, the refinement stage
genuinely cannot express the answer and E1's verdict is not an oracle artifact.
If it can, the affine oracle was under-powered and the partition must be re-read.

    python phase2/scripts/exp_e1_inverse_oracle.py run

**Pre-registered before running** (this docstring is the registration; the record
is `EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE` and is never overwritten):

  * 8 iterations at each damping factor alpha in {1.0, 0.5, 0.25}, clamped to the
    soft-argmin's own [0, 11] output range, applied only where `gt > 0`; the
    model's own `disparity_initial` elsewhere. The best trace over alpha is the
    one the verdict reads; all traces are recorded.
  * the full iteration trace is reported, not just the best point.
  * verdict rule is the one already frozen in
    `EXP_E1_ERROR_PARTITION_001_PREREGISTRATION.md` §4, applied to the best
    iterate: EPE ratio <= 0.35 MATCHING-LIMITED, >= 0.65 DOWNSTREAM-LIMITED,
    otherwise MIXED. Verdict per seed; overall only if all three agree.
  * gates: the iteration must be finite throughout; iteration 0 must reproduce
    the model as trained exactly; and **the search must actually search** -- at
    least one alpha must produce an iterate better than iteration 0 for every
    model. If none does, the step is mis-scaled and the run reports VOID rather
    than a verdict. This gate is the one the first attempt lacked.
  * this run also reports, as a diagnostic on run 001, the fraction of that
    run's affine-oracle values that fell outside [0, 11].

`UNKNOWN`, stated in advance: a clamped fixed point is a search over inputs to
*these* refinement weights. It says nothing about what a refinement stage
retrained against a perfect matching input could do.
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
from phase2.scripts.exp_h2_seed_replication import EXPERIMENTS_DIR  # noqa: E402
from phase2.scripts.h1_mechanism_probe import (  # noqa: E402
    _finish, _forward_parts, load_model,
)
from phase2.viz import core  # noqa: E402
from src.common.experiment import Experiment  # noqa: E402

EXPERIMENT_ID = "EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE-RUN3"
RESULT_DIR = REPO_ROOT / "phase2" / "results" / "EXP-E1-ERROR-PARTITION-001"
RUN_001 = REPO_ROOT / "phase2" / "results" / "EXP-E1-ERROR-PARTITION-001" / "partition.json"
ITERATIONS = 8
DAMPING = (1.0, 0.5, 0.25)
CANDIDATE_MIN, CANDIDATE_MAX = 0.0, 11.0


def affine_oracle_range_diagnostic(fits: dict, scenes) -> dict:
    """How much of run 001's oracle fell outside the soft-argmin's own range."""
    out = {}
    for key, fit in fits.items():
        outside = total = 0
        lo = hi = 0.0
        for scene in scenes:
            gt = scene.gt_disparity.astype(np.float64)[scene.gt_valid]
            oracle = fit["a"] * gt + fit["b"]
            outside += int(((oracle < CANDIDATE_MIN) | (oracle > CANDIDATE_MAX)).sum())
            total += oracle.size
            lo = min(lo, float(oracle.min()))
            hi = max(hi, float(oracle.max()))
        out[key] = {"fraction_outside_0_11": outside / total,
                    "oracle_min": lo, "oracle_max": hi}
    return out


def inverse_oracle(model_key: str, device: str, scenes, slope: float,
                   alpha: float) -> dict:
    """Fixed-point search for the matching output this refinement stage wants.

    The update is scaled by ``slope`` (candidates per pixel, measured in run 001)
    because ``initial`` is in candidate units and the residual is in pixels.
    """
    model = load_model(model_key, device)
    pools = [Pool() for _ in range(ITERATIONS + 1)]
    finite = True
    for scene in scenes:
        gt = scene.gt_disparity.astype(np.float64)
        valid = scene.gt_valid
        left, initial = _forward_parts(model, scene, device)
        current = initial.clone()
        for step in range(ITERATIONS + 1):
            final = _finish(model, left, current)
            pools[step].add(final, gt, valid)
            if not np.all(np.isfinite(final)):
                finite = False
            if step == ITERATIONS:
                break
            update = np.where(valid, alpha * slope * (gt - final), 0.0)
            nxt = current[0, 0].cpu().numpy().astype(np.float64) + update
            nxt = np.clip(nxt, CANDIDATE_MIN, CANDIDATE_MAX)
            current = torch.from_numpy(nxt.astype(np.float32)).to(initial).view_as(initial)
    return {"alpha": alpha, "slope": slope,
            "trace": [p.result() for p in pools], "finite": finite}


def verdict_for(trace: list) -> dict:
    as_trained = trace[0]["epe"]
    best = min(range(len(trace)), key=lambda i: trace[i]["epe"])
    ratio = trace[best]["epe"] / as_trained
    if ratio <= MATCHING_LIMITED_MAX_RATIO:
        label = "MATCHING-LIMITED"
    elif ratio >= DOWNSTREAM_LIMITED_MIN_RATIO:
        label = "DOWNSTREAM-LIMITED"
    else:
        label = "MIXED"
    return {"as_trained_epe": as_trained, "best_iteration": best,
            "best_epe": trace[best]["epe"], "best_d1": trace[best]["d1"],
            "epe_ratio": ratio, "error_removed_fraction": 1.0 - ratio,
            "verdict": label}


def table(runs: dict) -> str:
    lines = ["| model | alpha | iteration | EPE (px) | D1 (%) |",
             "|---|---:|---:|---:|---:|"]
    for key in MODELS:
        for run_row in runs[key]:
            for i, row in enumerate(run_row["trace"]):
                lines.append("| {} | {} | {} | {:.3f} | {:.2f} |".format(
                    key, run_row["alpha"], i, row["epe"], row["d1"]))
    return "\n".join(lines)


def best_run(run_rows: list) -> dict:
    """The alpha whose trace reaches the lowest EPE."""
    return min(run_rows, key=lambda r: min(row["epe"] for row in r["trace"]))


def run(device: str) -> None:
    config = {
        "experiment": EXPERIMENT_ID,
        "hypothesis": ("EXP-E1-ERROR-PARTITION-001's DOWNSTREAM-LIMITED verdict is "
                       "not an artifact of its affine oracle: a targeted "
                       "fixed-point search for the matching output that would "
                       "make the refinement stage correct also fails to remove "
                       "the error."),
        "preregistration": "the module docstring of phase2/scripts/exp_e1_inverse_oracle.py",
        "parent_run": "EXP-E1-ERROR-PARTITION-001",
        "measurement_only": True,
        "training": "none -- trained H2 checkpoints, inference and substitution only",
        "models": list(MODELS),
        "method": ("initial <- clamp(initial + alpha*a*(gt - final), 0, 11) where "
                   "gt > 0, {} iterations at each alpha in {}, a = the "
                   "candidates-per-pixel slope measured in run 001; model's own "
                   "initial elsewhere".format(ITERATIONS, list(DAMPING))),
        "supersedes": ("EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE, whose update "
                       "was not scaled by a and diverged -- see its NOTE.md"),
        "verdict_thresholds": {
            "matching_limited_max_epe_ratio": MATCHING_LIMITED_MAX_RATIO,
            "downstream_limited_min_epe_ratio": DOWNSTREAM_LIMITED_MIN_RATIO},
        "dataset": "kitti2015",
        "split": "hailo_val, all 40 scenes, pooled over gt > 0",
        "resolution": "368x1232 full frames",
        "crop": None,
        "disparity_range": 12,
        "batch_size": 1,
        "precision": "fp32",
        "seed": [0, 1, 2],
        "known_limitation": ("a clamped fixed point searches inputs to THESE "
                             "refinement weights; it says nothing about a "
                             "refinement stage retrained against a perfect "
                             "matching input."),
    }
    if (EXPERIMENTS_DIR / EXPERIMENT_ID).exists():
        raise SystemExit(EXPERIMENT_ID + " already exists -- records are never overwritten.")

    parent = json.loads(RUN_001.read_text(encoding="utf-8"))
    scenes = [core.load_scene(i) for i in SCENES]
    hashes_before = {k: core.sha256(core.checkpoint_path(k)) for k in MODELS}

    with Experiment("E1 robustness: inverse (fixed-point) oracle at the "
                    "matching -> refinement boundary", config=config,
                    experiment_id=EXPERIMENT_ID, experiments_dir=EXPERIMENTS_DIR) as exp:
        exp.note(config["hypothesis"])
        exp.note(config["known_limitation"])

        t0 = time.time()
        diag = affine_oracle_range_diagnostic(parent["affine_fits"], scenes)
        for key, row in diag.items():
            exp.log("run 001 affine oracle, {}: {:.2%} of substituted pixels outside "
                    "[0, 11] (range {:.2f} .. {:.2f})".format(
                        key, row["fraction_outside_0_11"], row["oracle_min"],
                        row["oracle_max"]))

        runs, verdicts, best = {}, {}, {}
        for key in MODELS:
            slope = parent["affine_fits"][key]["a"]
            runs[key] = [inverse_oracle(key, device, scenes, slope, alpha)
                         for alpha in DAMPING]
            best[key] = best_run(runs[key])
            verdicts[key] = verdict_for(best[key]["trace"])
            verdicts[key]["alpha"] = best[key]["alpha"]
            verdicts[key]["slope_candidates_per_px"] = slope
            for run_row in runs[key]:
                exp.log("{:<6} alpha {:<5} EPE trace {}".format(
                    key, run_row["alpha"],
                    "  ".join("{:.3f}".format(r["epe"]) for r in run_row["trace"])))
            exp.log("{} best alpha {} iteration {} ratio {:.3f} -> {}".format(
                key, verdicts[key]["alpha"], verdicts[key]["best_iteration"],
                verdicts[key]["epe_ratio"], verdicts[key]["verdict"]))

        hashes_after = {k: core.sha256(core.checkpoint_path(k)) for k in MODELS}
        integrity = hashes_before == hashes_after
        parent_as_trained = {k: parent["scores"][k]["as trained"]["epe"] for k in MODELS}
        gate = {
            "all_finite": all(r["finite"] for k in MODELS for r in runs[k]),
            "iteration_0_matches_parent_as_trained": all(
                abs(r["trace"][0]["epe"] - parent_as_trained[k]) <= 1e-9
                for k in MODELS for r in runs[k]),
            # The gate the first attempt lacked: a diverging search must not be
            # allowed to return "no improvement" as if it were a finding.
            "search_actually_improves": all(
                verdicts[k]["best_iteration"] > 0 for k in MODELS),
            "checkpoints_unchanged": integrity,
        }
        gate["PASS"] = all(gate.values())
        labels = {verdicts[k]["verdict"] for k in MODELS}
        overall = (labels.pop() if len(labels) == 1 else "NO CONSISTENT VERDICT") \
            if gate["PASS"] else "VOID -- a pre-registered gate failed"

        exp.metric("affine_oracle_range_diagnostic", diag)
        exp.metric("runs", runs)
        exp.metric("verdicts", verdicts)
        exp.metric("gates", gate)
        exp.metric("overall_verdict", overall)
        exp.metric("checkpoint_integrity", {"before": hashes_before, "after": hashes_after,
                                            "unchanged": integrity})
        exp.metric("wall_clock_s", time.time() - t0)

        md = table(runs)
        exp.path("inverse_oracle_table.md").write_text(md + "\n", encoding="utf-8")
        (RESULT_DIR / "inverse_oracle_run3.json").write_text(json.dumps(
            {"affine_oracle_range_diagnostic": diag, "runs": runs,
             "verdicts": verdicts, "gates": gate, "overall_verdict": overall},
            indent=2), encoding="utf-8")

        print("\n" + md + "\n")
        for key in MODELS:
            v = verdicts[key]
            print("{:<6} best alpha {}  iteration {}  EPE {:.3f} vs {:.3f} as "
                  "trained  ratio {:.3f}  -> {}".format(
                      key, v["alpha"], v["best_iteration"], v["best_epe"],
                      v["as_trained_epe"], v["epe_ratio"], v["verdict"]))
        print("\nOVERALL: " + overall)
        exp.conclude("Fixed-point oracle removes {} of pooled EPE; overall verdict "
                     "{}.".format(", ".join("{:.1%}".format(
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
