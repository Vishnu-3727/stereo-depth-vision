"""EXP-E3-REFINEMENT-ABLATION-001 -- how much of the refinement stack does H2 need?

Measurement only, no training. Removes residual blocks from the trained H2 seeds'
refinement stage at inference and measures what each removal costs in accuracy
and saves in MACs.

    python phase2/scripts/exp_e3_refinement_ablation.py run

Everything -- variants, gates, acceptance bands, verdict rule -- is frozen in
`phase2/docs/EXP_E3_REFINEMENT_ABLATION_001_PREREGISTRATION.md`, written before
this ran. Read that first; this file only executes it.

Why the blocks can be removed at all: each is
`out = act(conv2(act(conv1(x))) + x)`, residual and shape-preserving, so a subset
is a valid network with the surviving weights untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from phase2.scripts.exp_e1_error_partition import MODELS, Pool, SCENES  # noqa: E402
from phase2.scripts.exp_h2_seed_replication import (  # noqa: E402
    EXPERIMENTS_DIR, FOCUS_SCENES, GATE_MIN_STEREO_D1_POINTS, stereo_probe,
)
from phase2.scripts.h1_mechanism_probe import (  # noqa: E402
    _finish, _forward_parts, load_model,
)
from phase2.viz import core  # noqa: E402
from src.common.experiment import Experiment  # noqa: E402

EXPERIMENT_ID = "EXP-E3-REFINEMENT-ABLATION-001-RUN2"
RESULT_DIR = REPO_ROOT / "phase2" / "results" / "EXP-E3-REFINEMENT-ABLATION-001"
E1_RESULTS = REPO_ROOT / "phase2" / "results" / "EXP-E1-ERROR-PARTITION-001" / "partition.json"
DILATIONS = (1, 2, 4, 8, 1, 1)
N_BLOCKS = len(DILATIONS)
FULL = tuple(range(N_BLOCKS))
DEPLOY_HW = (368, 1232)
CROP_HW = (256, 512)

# Frozen acceptance bands (pre-registration section 4).
NEGLIGIBLE = {"d1": 1.0, "epe": 0.10}
TOLERABLE = {"d1": 3.0, "epe": 0.30}
OVERSIZED_MIN_SAVING = 0.25
RIGHT_SIZED_MAX_SAVING = 0.10
RECORD_TOLERANCE = {"epe": 0.01, "d1": 0.05}


def variants() -> dict:
    """Frozen variant set: baseline, single drops, prefixes, ladder."""
    out = {"full": FULL}
    for i in range(N_BLOCKS):
        out["drop block {} (dilation {})".format(i, DILATIONS[i])] = tuple(
            j for j in FULL if j != i)
    for k in range(N_BLOCKS):
        out["keep first {}".format(k)] = tuple(range(k))
    out["drop ladder (blocks 1,2,3)"] = tuple(j for j in FULL if j not in (1, 2, 3))
    return out


def ablate(model, keep: tuple):
    """A copy of the model with only the listed refinement blocks."""
    reduced = copy.deepcopy(model)
    reduced.refinement.blocks = nn.Sequential(
        *[model.refinement.blocks[i] for i in keep])
    return reduced.eval()


def score(model, scenes, device: str) -> dict:
    pool = Pool()
    finite = True
    for scene in scenes:
        left, initial = _forward_parts(model, scene, device)
        pred = _finish(model, left, initial)
        if not np.all(np.isfinite(pred)):
            finite = False
        pool.add(pred, scene.gt_disparity.astype(np.float64), scene.gt_valid)
    return {**pool.result(), "finite": finite}


def macs_for(keep: tuple, device: str) -> dict:
    """MACs of a freshly built model with this block subset, both resolutions.

    thop mutates the module it profiles, so every measurement gets a new model
    and none of them is the model that produced the accuracy numbers.
    """
    from thop import profile

    from phase2.models import scaled_regression
    from src.models.stereonet import StereoNet, StereoNetConfig

    out = {}
    for label, (h, w) in (("deploy_368x1232", DEPLOY_HW), ("crop_256x512", CROP_HW)):
        fresh = StereoNet(StereoNetConfig(cost_volume_shift="left"))
        scaled_regression.apply_to(fresh)
        fresh.refinement.blocks = nn.Sequential(
            *[fresh.refinement.blocks[i] for i in keep])
        fresh = fresh.to(device).eval()
        left = torch.randn(1, 3, h, w, device=device)
        right = torch.randn(1, 3, h, w, device=device)
        macs, params = profile(fresh, inputs=(left, right), verbose=False)
        out[label] = {"macs": float(macs), "thop_params": float(params)}
        del fresh
    return out


def band_for(delta_epe: float, delta_d1: float) -> str:
    if delta_d1 <= NEGLIGIBLE["d1"] and delta_epe <= NEGLIGIBLE["epe"]:
        return "NEGLIGIBLE"
    if delta_d1 <= TOLERABLE["d1"] and delta_epe <= TOLERABLE["epe"]:
        return "TOLERABLE"
    return "COSTLY"


def classify(scores: dict, macs: dict) -> dict:
    """Worst-case band across seeds, plus the MAC saving, per variant."""
    out = {}
    for name in scores[MODELS[0]]:
        deltas = {k: {"epe": scores[k][name]["epe"] - scores[k]["full"]["epe"],
                      "d1": scores[k][name]["d1"] - scores[k]["full"]["d1"]}
                  for k in MODELS}
        worst_epe = max(d["epe"] for d in deltas.values())
        worst_d1 = max(d["d1"] for d in deltas.values())
        band = band_for(worst_epe, worst_d1)
        saving = 1.0 - (macs[name]["deploy_368x1232"]["macs"]
                        / macs["full"]["deploy_368x1232"]["macs"])
        out[name] = {"worst_delta_epe": worst_epe, "worst_delta_d1": worst_d1,
                     "band": band, "mac_saving_fraction": saving,
                     "macs_deploy": macs[name]["deploy_368x1232"]["macs"],
                     "per_seed_delta": deltas}
    return out


def table(summary: dict, scores: dict) -> str:
    lines = ["| variant | blocks | MACs @368x1232 (G) | MAC saving | worst dEPE | "
             "worst dD1 | band |", "|---|---:|---:|---:|---:|---:|---|"]
    for name, row in summary.items():
        lines.append("| {} | {} | {:.3f} | {:.1%} | {:+.3f} | {:+.2f} | {} |".format(
            name, len(variants()[name]), row["macs_deploy"] / 1e9,
            row["mac_saving_fraction"], row["worst_delta_epe"],
            row["worst_delta_d1"], row["band"]))
    lines += ["", "| model | variant | EPE (px) | D1 (%) |", "|---|---|---:|---:|"]
    for key in MODELS:
        for name, row in scores[key].items():
            lines.append("| {} | {} | {:.3f} | {:.2f} |".format(
                key, name, row["epe"], row["d1"]))
    return "\n".join(lines)


def run(device: str) -> None:
    config = {
        "experiment": EXPERIMENT_ID,
        "hypothesis": ("With the matching path working, a substantial part of "
                       "H2's six-block refinement stack can be removed at "
                       "inference for little accuracy cost -- the largest "
                       "available efficiency win, since refinement is 90.6 % of "
                       "MACs and carries 58-71 % of the residual error."),
        "preregistration": "phase2/docs/EXP_E3_REFINEMENT_ABLATION_001_PREREGISTRATION.md",
        "measurement_only": True,
        "training": "none -- trained H2 checkpoints, block removal at inference only",
        "models": list(MODELS),
        "variants": {k: list(v) for k, v in variants().items()},
        "acceptance_bands": {"negligible": NEGLIGIBLE, "tolerable": TOLERABLE},
        "verdict_thresholds": {"oversized_min_mac_saving": OVERSIZED_MIN_SAVING,
                               "right_sized_max_mac_saving": RIGHT_SIZED_MAX_SAVING},
        "stereo_constraint": "C2: right-image dependence >= {} D1 points".format(
            GATE_MIN_STEREO_D1_POINTS),
        "dataset": "kitti2015",
        "split": "hailo_val, all 40 scenes, pooled over gt > 0",
        "resolution": "368x1232 full frames",
        "crop": None,
        "disparity_range": 12,
        "batch_size": 1,
        "precision": "fp32",
        "seed": [0, 1, 2],
        "known_limitation": ("inference-time ablation of weights trained as a "
                             "six-block stack: a LOWER bound on how small a "
                             "retrained refinement stage could be. MACs are not "
                             "latency; no latency is measured here."),
    }
    if (EXPERIMENTS_DIR / EXPERIMENT_ID).exists():
        raise SystemExit(EXPERIMENT_ID + " already exists -- records are never overwritten.")

    e1 = json.loads(E1_RESULTS.read_text(encoding="utf-8"))
    scenes = [core.load_scene(i) for i in SCENES]
    probe_scenes = [core.load_scene(i) for i in FOCUS_SCENES]
    hashes_before = {k: core.sha256(core.checkpoint_path(k)) for k in MODELS}
    plan = variants()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with Experiment("E3: how much of H2's refinement stack is load-bearing?",
                    config=config, experiment_id=EXPERIMENT_ID,
                    experiments_dir=EXPERIMENTS_DIR) as exp:
        exp.note(config["hypothesis"])
        exp.note(config["known_limitation"])

        t0 = time.time()
        macs = {name: macs_for(keep, device) for name, keep in plan.items()}
        for name, row in macs.items():
            exp.log("{:<28} {:>2} blocks  {:8.3f} GMAC @368x1232".format(
                name, len(plan[name]), row["deploy_368x1232"]["macs"] / 1e9))

        scores, stereo = {}, {}
        for key in MODELS:
            base = load_model(key, device)
            scores[key], stereo[key] = {}, {}
            for name, keep in plan.items():
                model = ablate(base, keep)
                scores[key][name] = score(model, scenes, device)
                exp.log("{:<6} {:<28} EPE {:7.3f}  D1 {:6.2f} %".format(
                    key, name, scores[key][name]["epe"], scores[key][name]["d1"]))
                if name == "full" or name.startswith("keep first"):
                    rows = stereo_probe(model, probe_scenes, device,
                                        np.random.default_rng(0))
                    worst = min(row[k]["d1_penalty"] for row in rows
                                for k in ("right_black", "right_noise",
                                          "right_equals_left"))
                    stereo[key][name] = {"right_dependence_worst_pt": worst,
                                         "meets_c2": worst >= GATE_MIN_STEREO_D1_POINTS}
                    exp.log("{:<6} {:<28} right-image dependence {:+.2f} pt "
                            "(C2 {})".format(key, name, worst,
                                             "met" if worst >= GATE_MIN_STEREO_D1_POINTS
                                             else "FAILED"))
                del model

        summary = classify(scores, macs)
        for name, row in summary.items():
            c2 = [stereo[k][name]["meets_c2"] for k in MODELS if name in stereo[k]]
            row["meets_c2"] = all(c2) if c2 else None
            row["viable"] = row["band"] in ("NEGLIGIBLE", "TOLERABLE") and row["meets_c2"] is not False

        hashes_after = {k: core.sha256(core.checkpoint_path(k)) for k in MODELS}
        gate = {
            "harness_fidelity": all(
                abs(scores[k]["full"]["epe"] - e1["scores"][k]["as trained"]["epe"])
                <= RECORD_TOLERANCE["epe"]
                and abs(scores[k]["full"]["d1"] - e1["scores"][k]["as trained"]["d1"])
                <= RECORD_TOLERANCE["d1"] for k in MODELS),
            "keep5_equals_drop5": all(
                abs(scores[k]["keep first 5"]["epe"]
                    - scores[k]["drop block 5 (dilation 1)"]["epe"]) <= 1e-9
                for k in MODELS),
            "all_finite": all(scores[k][n]["finite"] for k in MODELS for n in plan),
            "checkpoints_unchanged": hashes_before == hashes_after,
        }
        gate["PASS"] = all(gate.values())

        viable = {n: r for n, r in summary.items() if r["viable"] and n != "full"}
        best = max(viable.values(), key=lambda r: r["mac_saving_fraction"], default=None)
        best_saving = best["mac_saving_fraction"] if best else 0.0
        if not gate["PASS"]:
            verdict = "VOID -- a pre-registered gate failed"
        elif best_saving >= OVERSIZED_MIN_SAVING:
            verdict = "REFINEMENT IS OVERSIZED"
        elif best_saving < RIGHT_SIZED_MAX_SAVING:
            verdict = "REFINEMENT IS RIGHT-SIZED"
        else:
            verdict = "PARTIALLY OVERSIZED"

        exp.metric("macs", macs)
        exp.metric("scores", scores)
        exp.metric("stereo_dependence", stereo)
        exp.metric("summary", summary)
        exp.metric("gates", gate)
        exp.metric("best_viable_mac_saving", best_saving)
        exp.metric("verdict", verdict)
        exp.metric("checkpoint_integrity", {"before": hashes_before, "after": hashes_after})
        exp.metric("wall_clock_s", time.time() - t0)

        md = table(summary, scores)
        exp.path("ablation_table.md").write_text(md + "\n", encoding="utf-8")
        (RESULT_DIR / "ablation.json").write_text(json.dumps(
            {"macs": macs, "scores": scores, "stereo_dependence": stereo,
             "summary": summary, "gates": gate, "verdict": verdict}, indent=2),
            encoding="utf-8")

        print("\n" + md + "\n")
        print("gates: " + ("PASS" if gate["PASS"] else "FAIL " + str(gate)))
        print("best viable MAC saving: {:.1%}".format(best_saving))
        print("VERDICT: " + verdict)
        exp.conclude("Best viable refinement ablation saves {:.1%} of whole-model "
                     "MACs; verdict {}.".format(best_saving, verdict))
        print("\nrecorded as " + exp.id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"], nargs="?", default="run")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    run(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
