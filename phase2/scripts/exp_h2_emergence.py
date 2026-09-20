"""EXP-H2-EMERGENCE-001 -- when during training does stereo dependence appear?

Measurement only. No training, no architecture change, no new recipe: every
number here comes from weight snapshots that `EXP-H2-SEED-REPLICATION-001`
already saved (epochs 10/20/50/100/150/200 for seeds 1 and 2).

    python phase2/scripts/exp_h2_emergence.py run

Why it exists: the seed-replication run established that seed 0 shows *no*
right-image dependence at epoch 10 (-3.33 D1 points) yet +83.95 points at epoch
200, so stereo dependence is late-emerging in this recipe. That is the whole
reason its epoch-10 gate was invalid. This experiment measures the curve
between those two endpoints instead of guessing at it.

Frozen before running, and NOT invented here -- reused verbatim from
`phase2/docs/EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md`:

  * "stereo-dependent" at an epoch means the WORST-CASE D1 cost of corrupting
    the right image, over 3 corruptions x 4 probe scenes, is >= 20.0 points
    (`GATE_MIN_STEREO_D1_POINTS`, imported, not restated).
  * the probe itself is `exp_h2_seed_replication.stereo_probe`, imported
    unchanged, on the same `FOCUS_SCENES`.
  * emergence epoch = the first snapshot epoch meeting that criterion. The
    snapshot grid is coarse (10/20/50/100/150/200), so the answer is an
    interval between two saved epochs, never a precise epoch, and this script
    reports it that way.

Deliberate protocol deviation, recorded rather than hidden: during training the
ablation drew from one `np.random.default_rng(0)` that advanced across epochs,
so the epoch-100 and epoch-200 draws are not reproducible from the snapshots.
Here every checkpoint gets a FRESH `default_rng(0)`, which makes all twelve
measurements mutually comparable at the cost of not being bit-identical to the
three recorded ones. Epochs 10/100/200 overlap with the recorded ablation and
are cross-checked against it below; the difference is a measurement of the
probe's own noise.

Validation EPE/D1 are NOT recomputed -- they are read from each run's recorded
`metrics.json`.
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

from phase2.scripts.exp_h2_seed_replication import (  # noqa: E402
    EXPERIMENTS_DIR, FOCUS_SCENES, GATE_MIN_STEREO_D1_POINTS, OUT_DIR,
    WEIGHT_SNAPSHOT_EPOCHS, experiment_id, gradient_probe, stereo_probe,
)
from phase2.scripts.exp_h3_viability import build  # noqa: E402
from phase2.viz import core  # noqa: E402
from src.common.experiment import Experiment  # noqa: E402

RESULT_DIR = REPO_ROOT / "phase2" / "results" / "EXP-H2-EMERGENCE-001"
EXPERIMENT_ID = "EXP-H2-EMERGENCE-001"
SEEDS = (1, 2)
LABEL = "-RUN2"                      # the runs that reached 200 epochs
RIGHT_KEYS = ("right_black", "right_noise", "right_equals_left")
MAP_KEYS = ("initial_constant_mean", "initial_shuffled")


def snapshot_path(seed: int, epoch: int) -> Path:
    return OUT_DIR / "{}_epoch{}.pth".format(experiment_id(seed, LABEL), epoch)


def recorded_metrics(seed: int) -> dict:
    """The recorded metrics of one source run.

    `Experiment` nests everything it recorded under a top-level "metrics" key;
    reading the file's top level instead silently yields an empty dict, which
    is what defect (1) in EXP-H2-EMERGENCE-001's NOTE.md was.
    """
    path = EXPERIMENTS_DIR / experiment_id(seed, LABEL) / "metrics.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    metrics = blob["metrics"]
    assert "checkpoints" in metrics and "stereo_ablations" in metrics, \
        "unexpected record layout in " + str(path)
    return metrics


def worst_penalty(ablation: list, keys) -> float:
    """The weakest evidence of dependence: smallest D1 cost over all scenes."""
    return min(row[k]["d1_penalty"] for row in ablation for k in keys)


def mean_penalty(ablation: list, keys) -> float:
    return float(np.mean([row[k]["d1_penalty"] for row in ablation for k in keys]))


def probe_checkpoint(seed: int, epoch: int, scenes: list, device: str) -> dict:
    path = snapshot_path(seed, epoch)
    blob = torch.load(path, map_location=device, weights_only=False)
    assert blob["epoch"] == epoch, "snapshot {} claims epoch {}".format(path, blob["epoch"])
    assert blob["config"]["seed"] == seed, "snapshot {} claims seed {}".format(
        path, blob["config"]["seed"])
    assert blob["config"]["cost_volume_shift"] == "left", "not the H2 shift"
    model = build("left", blob["model"], device)          # H2 architecture + regression
    ablation = stereo_probe(model, scenes, device, np.random.default_rng(0))
    gradient = gradient_probe(model, device, crops=12, seed=0)
    baseline_d1 = float(np.mean([row["baseline"]["d1"] for row in ablation]))
    baseline_epe = float(np.mean([row["baseline"]["epe"] for row in ablation]))
    return {
        "seed": seed, "epoch": epoch,
        "snapshot": str(path.relative_to(REPO_ROOT)),
        "snapshot_sha256": core.sha256(path),
        "probe_scene_d1": baseline_d1, "probe_scene_epe": baseline_epe,
        "right_dependence_worst_pt": worst_penalty(ablation, RIGHT_KEYS),
        "right_dependence_mean_pt": mean_penalty(ablation, RIGHT_KEYS),
        "map_dependence_worst_pt": worst_penalty(ablation, MAP_KEYS),
        "map_dependence_mean_pt": mean_penalty(ablation, MAP_KEYS),
        "softmax_entropy": float(np.mean([row["softmax_entropy"] for row in ablation])),
        "matching_gradient_fraction": gradient["matching_present_fraction"],
        "matching_grad_median": gradient["matching_norm"]["median"],
        "total_grad_max": gradient["total_norm"]["max"],
        "grad_spikes_above_1e4": gradient["spikes_above_1e4"],
        "ablation": ablation, "gradient": gradient,
    }


def emergence_interval(rows: list) -> dict:
    """First snapshot epoch meeting the frozen criterion, as an interval."""
    epochs = [r["epoch"] for r in rows]
    met = [r["epoch"] for r in rows
           if r["right_dependence_worst_pt"] >= GATE_MIN_STEREO_D1_POINTS]
    if not met:
        return {"emerged": False, "first_epoch_meeting_criterion": None,
                "interval": [epochs[-1], None]}
    first = min(met)
    earlier = [e for e in epochs if e < first]
    return {"emerged": True, "first_epoch_meeting_criterion": first,
            "interval": [max(earlier) if earlier else 0, first],
            "monotone_after": all(
                r["right_dependence_worst_pt"] >= GATE_MIN_STEREO_D1_POINTS
                for r in rows if r["epoch"] >= first)}


def cross_check(rows: list, seed: int) -> list:
    """Recorded training-time ablation vs this re-measurement, same weights."""
    recorded = recorded_metrics(seed).get("stereo_ablations", {})
    out = []
    for row in rows:
        key = str(row["epoch"])
        if key not in recorded:
            continue
        was = worst_penalty(recorded[key], RIGHT_KEYS)
        out.append({
            "seed": seed, "epoch": row["epoch"],
            "recorded_right_worst_pt": was,
            "remeasured_right_worst_pt": row["right_dependence_worst_pt"],
            "difference_pt": row["right_dependence_worst_pt"] - was,
        })
    return out


def table(rows: list, val: dict) -> str:
    head = ("| seed | epoch | val EPE | val D1 % | right-image dep. worst / mean (pt) "
            "| matching-map dep. worst (pt) | softmax entropy | matching grad. | "
            "grad max |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines = [head]
    for r in rows:
        v = val.get((r["seed"], r["epoch"]), {})
        lines.append(
            "| {} | {} | {} | {} | {:+.2f} / {:+.2f} | {:+.2f} | {:.3f} | {:.0%} | "
            "{:.4g} |".format(
                r["seed"], r["epoch"],
                "{:.3f}".format(v["val_epe"]) if v else "n/a",
                "{:.2f}".format(v["val_d1"]) if v else "n/a",
                r["right_dependence_worst_pt"], r["right_dependence_mean_pt"],
                r["map_dependence_worst_pt"], r["softmax_entropy"],
                r["matching_gradient_fraction"], r["total_grad_max"]))
    return "\n".join(lines)


def run(device: str, scenes: list, label: str = "") -> None:
    for seed in SEEDS:
        for epoch in WEIGHT_SNAPSHOT_EPOCHS:
            if not snapshot_path(seed, epoch).exists():
                raise SystemExit("missing snapshot: " + str(snapshot_path(seed, epoch)))

    exp_id = EXPERIMENT_ID + label
    config = {
        "experiment": exp_id,
        "hypothesis": ("Stereo dependence in the H2 recipe is late-emerging: it is "
                       "absent early in training and appears at some point before "
                       "epoch 200. This experiment locates that point on the saved "
                       "snapshot grid."),
        "measurement_only": True,
        "training": "none -- weights are read from EXP-H2-SEED-REPLICATION-001 snapshots",
        "source_runs": [experiment_id(s, LABEL) for s in SEEDS],
        "snapshot_epochs": list(WEIGHT_SNAPSHOT_EPOCHS),
        "probe": "phase2.scripts.exp_h2_seed_replication.stereo_probe (imported unchanged)",
        "probe_scenes": scenes,
        "criterion_frozen_in":
            "phase2/docs/EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md",
        "criterion_right_image_d1_points": GATE_MIN_STEREO_D1_POINTS,
        "rng_deviation": ("fresh np.random.default_rng(0) per checkpoint, where "
                          "training advanced a single generator across epochs; "
                          "epochs 10/100/200 are cross-checked against the "
                          "recorded ablation"),
        "validation_metrics": "read from each run's recorded metrics.json, not recomputed",
        "dataset": "kitti2015",
        "split": "hailo_val scenes {} (probe), first 10 scenes (recorded validation)"
                 .format(scenes),
        "resolution": "368x1232 full frames",
        "crop": [256, 512],
        "disparity_range": 12,
        "batch_size": 1,
        "precision": "fp32",
        "seed": list(SEEDS),
        "known_limitation": ("two seeds, four probe scenes, and a six-point epoch "
                             "grid; the emergence point is an interval between two "
                             "saved epochs, not a measured epoch, and the harness "
                             "is not bit-reproducible (fp32 cuDNN)."),
    }

    loaded = [core.load_scene(i) for i in scenes]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with Experiment("H2 stereo-dependence emergence over training",
                    config=config, experiment_id=exp_id,
                    experiments_dir=EXPERIMENTS_DIR) as exp:
        exp.note(config["hypothesis"])
        exp.note(config["known_limitation"])
        exp.note(config["rng_deviation"])

        t0 = time.time()
        rows, val = [], {}
        for seed in SEEDS:
            recorded = recorded_metrics(seed).get("checkpoints", {})
            for epoch in WEIGHT_SNAPSHOT_EPOCHS:
                row = probe_checkpoint(seed, epoch, loaded, device)
                rows.append(row)
                if str(epoch) in recorded:
                    val[(seed, epoch)] = {
                        "val_epe": recorded[str(epoch)]["val_epe"],
                        "val_d1": recorded[str(epoch)]["val_d1"]}
                exp.log("seed {} epoch {:>3}  right-dep worst {:+8.2f} pt  "
                        "mean {:+8.2f} pt  map-dep worst {:+8.2f} pt  entropy {:.3f}  "
                        "matching grad {:.0%}".format(
                            seed, epoch, row["right_dependence_worst_pt"],
                            row["right_dependence_mean_pt"],
                            row["map_dependence_worst_pt"], row["softmax_entropy"],
                            row["matching_gradient_fraction"]))

        per_seed = {}
        for seed in SEEDS:
            seed_rows = [r for r in rows if r["seed"] == seed]
            per_seed[str(seed)] = emergence_interval(seed_rows)
            exp.metric("emergence_seed{}".format(seed), per_seed[str(seed)])

        checks = [c for seed in SEEDS for c in cross_check(
            [r for r in rows if r["seed"] == seed], seed)]
        exp.metric("cross_check_vs_recorded_ablation", checks)
        exp.metric("rows", rows)
        exp.metric("validation_from_record",
                   {"{}:{}".format(*k): v for k, v in val.items()})
        exp.metric("wall_clock_s", time.time() - t0)

        md = table(rows, val)
        exp.path("emergence_table.md").write_text(md + "\n", encoding="utf-8")
        (RESULT_DIR / "emergence{}.json".format(label.lower())).write_text(
            json.dumps({"rows": rows, "emergence": per_seed, "cross_check": checks,
                        "validation_from_record":
                            {"{}:{}".format(*k): v for k, v in val.items()}},
                       indent=2), encoding="utf-8")
        print("\n" + md + "\n")
        for seed in SEEDS:
            print("seed {}: {}".format(seed, per_seed[str(seed)]))
        print("\ncross-check vs recorded ablation (same weights, different rng draw):")
        for c in checks:
            print("  seed {} epoch {:>3}  recorded {:+8.2f} pt  remeasured {:+8.2f} pt"
                  "  diff {:+.2f} pt".format(c["seed"], c["epoch"],
                                             c["recorded_right_worst_pt"],
                                             c["remeasured_right_worst_pt"],
                                             c["difference_pt"]))

        intervals = ", ".join("seed {} in ({}, {}]".format(
            s, per_seed[str(s)]["interval"][0], per_seed[str(s)]["interval"][1])
            for s in SEEDS if per_seed[str(s)]["emerged"])
        exp.conclude(
            "Stereo dependence (>= {:.0f} D1 points of right-image cost, worst case) "
            "first appears at: {}. Measured on saved snapshots only; no training."
            .format(GATE_MIN_STEREO_D1_POINTS,
                    intervals or "no snapshot met the criterion"))
        print("\nrecorded as " + exp.id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"], nargs="?", default="run")
    ap.add_argument("--scenes", nargs="+", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--label", default="",
                    help="suffix for the experiment id, e.g. '-RUN2'; records "
                         "are never overwritten, so a rerun needs a new id")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run(device, args.scenes or FOCUS_SCENES, args.label)


if __name__ == "__main__":
    main()
