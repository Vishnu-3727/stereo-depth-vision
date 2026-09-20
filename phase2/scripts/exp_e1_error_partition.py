"""EXP-E1-ERROR-PARTITION-001 -- is H2's residual error upstream or downstream?

Measurement only, no training. Substitutes an oracle matching output into the
matching -> refinement boundary of the three trained H2 seeds and asks how much
of the remaining error a *perfect* matching stage would remove.

    python phase2/scripts/exp_e1_error_partition.py run

Everything about the protocol -- variants, gates, verdict thresholds -- is
frozen in `phase2/docs/EXP_E1_ERROR_PARTITION_001_PREREGISTRATION.md`, written
before this script ran. Read that first; this file only executes it.

The intervention harness (`_forward_parts`, `_finish`) is imported unchanged
from `h1_mechanism_probe`, which produced the recorded pooled numbers this run
is checked against.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase2.scripts.exp_h2_seed_replication import EXPERIMENTS_DIR  # noqa: E402
from phase2.scripts.h1_mechanism_probe import (  # noqa: E402
    _finish, _forward_parts, load_model,
)
from phase2.viz import core  # noqa: E402
from src.common.experiment import Experiment  # noqa: E402

EXPERIMENT_ID = "EXP-E1-ERROR-PARTITION-001"
RESULT_DIR = REPO_ROOT / "phase2" / "results" / "EXP-E1-ERROR-PARTITION-001"
MODELS = ("H2", "SEED1", "SEED2")
SCENES = tuple(range(40))

# Frozen in the pre-registration. Not to be touched.
MATCHING_LIMITED_MAX_RATIO = 0.35
DOWNSTREAM_LIMITED_MIN_RATIO = 0.65
RECORDED_H2_POOLED = {"epe": 2.400, "d1": 16.72}   # h2_mechanism/pooled.json
RECORD_TOLERANCE = {"epe": 0.01, "d1": 0.05}
IDENTITY_TOLERANCE_PX = 1e-9

AS_TRAINED = "as trained"
IDENTITY = "initial := itself"
ORACLE_SPARSE = "initial := a*gt+b where gt>0"
ORACLE_DENSE = "initial := a*gt+b densified"
ORACLE_SHUFFLED = "initial := a*gt+b shuffled"


class Pool:
    """Pixel-pooled EPE/D1, accumulated across scenes exactly as the recorded
    pooled protocol does it (KITTI D1: err > 3 px AND err > 5 % of gt)."""

    def __init__(self) -> None:
        self.pixels = 0
        self.outliers = 0
        self.error_sum = 0.0

    def add(self, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> None:
        err = np.abs(pred[valid] - gt[valid])
        g = gt[valid]
        self.pixels += int(err.size)
        self.outliers += int(((err > 3.0) & (err > 0.05 * g)).sum())
        self.error_sum += float(err.sum())

    def result(self) -> dict:
        return {"epe": self.error_sum / self.pixels,
                "d1": 100.0 * self.outliers / self.pixels,
                "pixels": self.pixels}


def densify(gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Nearest-valid-pixel fill. Interpolation, and labelled as such."""
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return gt[iy, ix]


def fit_affine(model_key: str, device: str, scenes) -> dict:
    """Recover the model's own candidate-vs-ground-truth convention.

    `disparity_initial` is in candidate units and the candidate->pixel scale is
    learned, so the oracle must be expressed in the model's convention rather
    than in an assumed x16.
    """
    model = load_model(model_key, device)
    sx = sy = sxx = sxy = 0.0
    n = 0
    for scene in scenes:
        _, initial = _forward_parts(model, scene, device)
        y = initial[0, 0].cpu().numpy().astype(np.float64)[scene.gt_valid]
        x = scene.gt_disparity.astype(np.float64)[scene.gt_valid]
        sx += x.sum(); sy += y.sum(); sxx += (x * x).sum(); sxy += (x * y).sum()
        n += x.size
    a = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b = (sy - a * sx) / n
    # Second pass for the residual: needed for R^2 and it is cheap.
    ss_res = ss_tot = 0.0
    mean_y = sy / n
    for scene in scenes:
        _, initial = _forward_parts(model, scene, device)
        y = initial[0, 0].cpu().numpy().astype(np.float64)[scene.gt_valid]
        x = scene.gt_disparity.astype(np.float64)[scene.gt_valid]
        ss_res += float(((y - (a * x + b)) ** 2).sum())
        ss_tot += float(((y - mean_y) ** 2).sum())
    return {"a": a, "b": b, "r2": 1.0 - ss_res / ss_tot,
            "residual_std_candidates": float(np.sqrt(ss_res / n)),
            "pixels": n, "initial_mean_candidates": mean_y}


def score_model(model_key: str, device: str, scenes, fit: dict) -> dict:
    model = load_model(model_key, device)
    pools = {tag: Pool() for tag in
             (AS_TRAINED, IDENTITY, ORACLE_SPARSE, ORACLE_DENSE, ORACLE_SHUFFLED)}
    rng = np.random.default_rng(0)
    a, b = fit["a"], fit["b"]
    for scene in scenes:
        gt = scene.gt_disparity.astype(np.float64)
        valid = scene.gt_valid
        left, initial = _forward_parts(model, scene, device)
        pools[AS_TRAINED].add(_finish(model, left, initial), gt, valid)
        pools[IDENTITY].add(_finish(model, left, initial.clone()), gt, valid)

        oracle = a * gt + b
        own = initial[0, 0].cpu().numpy().astype(np.float64)
        sparse = np.where(valid, oracle, own)
        dense = a * densify(gt, valid) + b

        def as_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.astype(np.float32)).to(initial).view_as(initial)

        pools[ORACLE_SPARSE].add(_finish(model, left, as_tensor(sparse)), gt, valid)
        pools[ORACLE_DENSE].add(_finish(model, left, as_tensor(dense)), gt, valid)
        flat = sparse.flatten()
        shuffled = flat[rng.permutation(flat.size)].reshape(sparse.shape)
        pools[ORACLE_SHUFFLED].add(_finish(model, left, as_tensor(shuffled)), gt, valid)
    return {tag: pool.result() for tag, pool in pools.items()}


def verdict_for(scores: dict) -> dict:
    ratio = scores[ORACLE_SPARSE]["epe"] / scores[AS_TRAINED]["epe"]
    if ratio <= MATCHING_LIMITED_MAX_RATIO:
        label = "MATCHING-LIMITED"
    elif ratio >= DOWNSTREAM_LIMITED_MIN_RATIO:
        label = "DOWNSTREAM-LIMITED"
    else:
        label = "MIXED"
    return {"epe_ratio": ratio, "error_removed_fraction": 1.0 - ratio,
            "verdict": label}


def gates(scores: dict, model_key: str) -> dict:
    identity_delta = abs(scores[IDENTITY]["epe"] - scores[AS_TRAINED]["epe"])
    checks = {
        "substitution_fidelity": identity_delta <= IDENTITY_TOLERANCE_PX,
        "identity_epe_delta": identity_delta,
        "negative_control": scores[ORACLE_SHUFFLED]["epe"] > scores[AS_TRAINED]["epe"],
    }
    if model_key == "H2":
        checks["record_fidelity"] = (
            abs(scores[AS_TRAINED]["epe"] - RECORDED_H2_POOLED["epe"])
            <= RECORD_TOLERANCE["epe"]
            and abs(scores[AS_TRAINED]["d1"] - RECORDED_H2_POOLED["d1"])
            <= RECORD_TOLERANCE["d1"])
        checks["record_epe_delta"] = scores[AS_TRAINED]["epe"] - RECORDED_H2_POOLED["epe"]
        checks["record_d1_delta"] = scores[AS_TRAINED]["d1"] - RECORDED_H2_POOLED["d1"]
    checks["PASS"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks


def table(rows: dict) -> str:
    lines = ["| model | variant | EPE (px) | D1 (%) |", "|---|---|---:|---:|"]
    for key in MODELS:
        for tag in (AS_TRAINED, IDENTITY, ORACLE_SPARSE, ORACLE_DENSE, ORACLE_SHUFFLED):
            r = rows[key][tag]
            lines.append("| {} | {} | {:.3f} | {:.2f} |".format(key, tag, r["epe"], r["d1"]))
    return "\n".join(lines)


def run(device: str) -> None:
    config = {
        "experiment": EXPERIMENT_ID,
        "hypothesis": ("H2's residual error is partitioned by substituting a "
                       "perfect (oracle) matching output at the matching -> "
                       "refinement boundary: the fraction of error it removes "
                       "says whether the ceiling is upstream or downstream."),
        "preregistration": "phase2/docs/EXP_E1_ERROR_PARTITION_001_PREREGISTRATION.md",
        "measurement_only": True,
        "training": "none -- trained H2 checkpoints, inference and substitution only",
        "models": list(MODELS),
        "harness": ("phase2.scripts.h1_mechanism_probe._forward_parts / ._finish, "
                    "imported unchanged"),
        "oracle": ("disparity_initial := a*gt + b, with a, b fitted by least "
                   "squares per model over every gt>0 pixel of the 40 scenes; "
                   "the candidate->pixel scale is learned, so it is measured, "
                   "not assumed"),
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
        "known_limitation": ("measures the ceiling of THESE trained refinement "
                             "weights; a refinement stage retrained against a "
                             "perfect matching input could behave differently. "
                             "One dataset slice, three seeds, no significance "
                             "test and none claimed."),
    }
    if (EXPERIMENTS_DIR / EXPERIMENT_ID).exists():
        raise SystemExit(EXPERIMENT_ID + " already exists -- records are never overwritten.")

    scenes = [core.load_scene(i) for i in SCENES]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    hashes_before = {k: core.sha256(core.checkpoint_path(k)) for k in MODELS}

    with Experiment("E1 error partition: what would a perfect matching stage buy?",
                    config=config, experiment_id=EXPERIMENT_ID,
                    experiments_dir=EXPERIMENTS_DIR) as exp:
        exp.note(config["hypothesis"])
        exp.note(config["known_limitation"])
        exp.note("Thresholds and variants were frozen in " + config["preregistration"]
                 + " before this run.")

        t0 = time.time()
        fits, scores, gate_rows, verdicts = {}, {}, {}, {}
        for key in MODELS:
            fits[key] = fit_affine(key, device, scenes)
            exp.log("{} affine fit: initial = {:.6f}*gt + {:.6f}  R2 {:.4f}  "
                    "residual {:.3f} candidates".format(
                        key, fits[key]["a"], fits[key]["b"], fits[key]["r2"],
                        fits[key]["residual_std_candidates"]))
            scores[key] = score_model(key, device, scenes, fits[key])
            gate_rows[key] = gates(scores[key], key)
            verdicts[key] = verdict_for(scores[key])
            for tag, r in scores[key].items():
                exp.log("{:<6} {:<32} EPE {:7.3f}  D1 {:6.2f} %".format(
                    key, tag, r["epe"], r["d1"]))
            exp.log("{} ratio {:.3f} -> {} (gates {})".format(
                key, verdicts[key]["epe_ratio"], verdicts[key]["verdict"],
                "PASS" if gate_rows[key]["PASS"] else "FAIL"))

        hashes_after = {k: core.sha256(core.checkpoint_path(k)) for k in MODELS}
        integrity = hashes_before == hashes_after
        labels = {verdicts[k]["verdict"] for k in MODELS}
        all_gates = all(gate_rows[k]["PASS"] for k in MODELS) and integrity
        overall = (labels.pop() if len(labels) == 1 else "NO CONSISTENT VERDICT") \
            if all_gates else "VOID -- a pre-registered gate failed"

        exp.metric("affine_fits", fits)
        exp.metric("scores", scores)
        exp.metric("gates", gate_rows)
        exp.metric("checkpoint_integrity", {"before": hashes_before,
                                            "after": hashes_after, "unchanged": integrity})
        exp.metric("verdicts", verdicts)
        exp.metric("overall_verdict", overall)
        exp.metric("wall_clock_s", time.time() - t0)

        md = table(scores)
        exp.path("partition_table.md").write_text(md + "\n", encoding="utf-8")
        (RESULT_DIR / "partition.json").write_text(json.dumps(
            {"affine_fits": fits, "scores": scores, "gates": gate_rows,
             "verdicts": verdicts, "overall_verdict": overall,
             "checkpoint_sha256": hashes_after}, indent=2), encoding="utf-8")

        print("\n" + md + "\n")
        for key in MODELS:
            print("{:<6} oracle/as-trained EPE ratio {:.3f}  "
                  "error removed {:.1%}  -> {}".format(
                      key, verdicts[key]["epe_ratio"],
                      verdicts[key]["error_removed_fraction"], verdicts[key]["verdict"]))
        print("\nOVERALL: " + overall)
        exp.conclude("Oracle matching output removes {} of H2's pooled EPE across "
                     "seeds; overall verdict {}.".format(
                         ", ".join("{:.1%}".format(verdicts[k]["error_removed_fraction"])
                                   for k in MODELS), overall))
        print("\nrecorded as " + exp.id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"], nargs="?", default="run")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    run(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
