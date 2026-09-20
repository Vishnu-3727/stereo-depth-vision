"""EXP-E3C-REFINEMENT-FLOOR-001 -- does a *trained* three-block refinement stack match H2?

E3b found four blocks indistinguishable from six. This takes one step further
down and looks for the floor.

    python phase2/scripts/exp_e3c_refinement_floor.py preflight
    python phase2/scripts/exp_e3c_refinement_floor.py train
    python phase2/scripts/exp_e3c_refinement_floor.py analyse

Registered separately from E3b, and deliberately so: E3b's stage 2 was
pre-committed to trigger only on an arm falling outside the material band (none
did) and to add no new arms. Reusing its ID for a new arm would mean editing a
frozen protocol after seeing its results. See
`phase2/docs/EXP_E3C_REFINEMENT_FLOOR_001_PREREGISTRATION.md` section 0.

What is reused from E3b, as measured inputs rather than as code to re-run: its
**control arm** (`...-ARM-A-RUN2`, 6 blocks, seed 0) and its **noise band**
(0.2138 px / 1.0000 D1 point). The training loop, gates and probes are the same
objects, imported.

The confound this arm cannot avoid, stated in the pre-registration before the
run: `keep first 3` is dilations 1, 2, 4, so it drops dilation 8 and shrinks the
receptive field as well as the capacity. A negative result here cannot separate
the two.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import phase2.scripts.exp_e3b_refinement_capacity as e3b  # noqa: E402
from phase2.viz import core  # noqa: E402
from src.common.experiment import Experiment  # noqa: E402

EXPERIMENT_PREFIX = "EXP-E3C-REFINEMENT-FLOOR-001"
RESULT_DIR = REPO_ROOT / "phase2" / "results" / EXPERIMENT_PREFIX
CONTROL_ID = "EXP-E3B-REFINEMENT-CAPACITY-001-ARM-A-RUN2"
PREREGISTRATION = "phase2/docs/EXP_E3C_REFINEMENT_FLOOR_001_PREREGISTRATION.md"

# Frozen in the pre-registration section 2.
ARM_D = {"blocks": 3, "keep": (0, 1, 2), "dilations": (1, 2, 4),
         "parameters": 368098, "macs_deploy": 30.964e9,
         "role": "ablated (three blocks; drops dilation 8 -- see the confound)"}
# Frozen in section 4: reused from E3b, not recomputed.
BAND = {"epe": 0.21380062638697517, "d1": 1.0}

_original_arm_config = e3b.arm_config


def _arm_config(arm: str, device: str) -> dict:
    """E3b's config, re-pointed at this experiment's registration."""
    config = dict(_original_arm_config(arm, device))
    config.update({
        "experiment": e3b.arm_id(arm),
        "preregistration": PREREGISTRATION,
        "approved_by": None,
        "hypothesis": ("A refinement stack trained at 3 residual blocks matches "
                       "six-block H2 at the same budget, seed and recipe, and "
                       "stays stereo-functional."),
        "control": ("{} (E3b arm A, 6 blocks, seed 0, not retrained); the "
                    "material band 0.2138 px / 1.0000 D1 point is E3b's measured "
                    "same-seed noise band, reused not recomputed".format(CONTROL_ID)),
        "changed_variable": ("number of residual blocks in Refinement.blocks (3 "
                             "instead of 6), keep first 3"),
        "known_confound": ("keep-first-3 is dilations 1,2,4 and therefore drops "
                           "dilation 8: this arm reduces the receptive field as "
                           "well as the capacity, so a negative result cannot "
                           "separate the two causes. Stated before the run; the "
                           "disambiguating (1,2,8)/(1,4,8) arm is named in the "
                           "pre-registration section 8 and is NOT run here."),
        "known_limitation": ("one seed; 160 training scenes from random "
                             "initialisation, so no transfer to a pretrained "
                             "model is implied. MACs are arithmetic, not latency."),
    })
    return config


def _install() -> None:
    """Point E3b's machinery at this experiment. No E3b record is touched."""
    e3b.ARMS["D"] = ARM_D
    e3b.EXPERIMENT_PREFIX = EXPERIMENT_PREFIX
    e3b.RESULT_DIR = RESULT_DIR
    e3b.arm_config = _arm_config
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_preflight(device: str) -> dict:
    result = e3b.run_preflight(device)          # checks every arm, D included
    result["control"] = {
        "experiment": CONTROL_ID,
        "checkpoint_sha256": core.sha256(
            e3b.OUT_DIR / (CONTROL_ID + "_checkpoint.pth")),
        "band": BAND,
    }
    return result


def _control_metrics() -> dict:
    path = e3b.EXPERIMENTS_DIR / CONTROL_ID / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


def run_analysis(device: str) -> None:
    exp_id = EXPERIMENT_PREFIX + "-ANALYSIS"
    if (e3b.EXPERIMENTS_DIR / exp_id).exists():
        raise SystemExit(exp_id + " already exists -- records are never overwritten.")
    arm = json.loads((e3b.EXPERIMENTS_DIR / e3b.arm_id("D") / "metrics.json")
                     .read_text(encoding="utf-8"))["metrics"]
    control = _control_metrics()

    d_epe = arm["late_window"]["epe_mean"] - control["late_window"]["epe_mean"]
    d_d1 = arm["late_window"]["d1_mean"] - control["late_window"]["d1_mean"]
    material = abs(d_epe) > BAND["epe"] and abs(d_d1) > BAND["d1"]
    worse = bool(material and d_epe > 0 and d_d1 > 0)
    verdict_200 = arm.get("stereo_verdict_epoch_200", {})
    stereo = bool(verdict_200.get("STEREO_FUNCTIONAL"))

    gate = {
        "arm_completed": arm.get("aborted") is None
        and arm["epochs_completed"] == e3b.EPOCHS,
        "control_stereo_functional": bool(
            control.get("stereo_verdict_epoch_200", {}).get("STEREO_FUNCTIONAL")),
        "phase_1_diff_empty": e3b.git_state()["phase_1_diff_vs_frozen"] == "",
    }
    gate["PASS"] = all(gate.values())

    if not gate["PASS"]:
        verdict = "INVALID"
    elif not worse and not stereo:
        verdict = "STEREO-BROKEN"
    elif not worse and stereo:
        verdict = "CAPACITY-REDUCIBLE-TO-3"
    elif worse:
        verdict = "CAPACITY-REQUIRED-ABOVE-3"
    else:
        verdict = "INCONCLUSIVE"

    config = {
        "experiment": exp_id,
        "hypothesis": "Analysis of the three-block arm against E3b's control and band.",
        "preregistration": PREREGISTRATION,
        "measurement_only": True, "training": "none -- reads two existing records",
        "control": CONTROL_ID, "band": BAND,
        "dataset": "kitti2015", "split": "hailo_val (first 10 scenes, recorded protocol)",
        "resolution": "368x1232", "crop": [256, 512], "disparity_range": 12,
        "batch_size": 2, "precision": "fp32", "seed": e3b.SEED,
        "known_limitation": ("one seed; the band is E3b's single-sample lower "
                             "bound on same-seed noise, not a confidence "
                             "interval; the arm also reduces the receptive field, "
                             "so a negative result is ambiguous."),
    }
    with Experiment("E3c three-block refinement analysis", config=config,
                    experiment_id=exp_id, experiments_dir=e3b.EXPERIMENTS_DIR) as exp:
        exp.note(config["known_limitation"])
        for name, m in (("control (6 blocks)", control), ("arm D (3 blocks)", arm)):
            lw = m["late_window"]
            exp.log("{:<20} late EPE {:.4f} +- {:.4f}  D1 {:.4f} +- {:.4f}  params {}"
                    .format(name, lw["epe_mean"], lw["epe_sd"], lw["d1_mean"],
                            lw["d1_sd"], m["parameter_count_unique"]))
        exp.log("delta vs control: EPE {:+.4f} (band {:.4f})  D1 {:+.4f} (band {:.4f})"
                " -> materially different {}".format(
                    d_epe, BAND["epe"], d_d1, BAND["d1"], material))
        exp.metric("late_window", {"control": control["late_window"],
                                   "arm_D": arm["late_window"]})
        exp.metric("comparison", {"delta_epe_vs_control": d_epe,
                                  "delta_d1_vs_control": d_d1,
                                  "materially_different": bool(material),
                                  "worse": worse, "band": BAND})
        exp.metric("stereo_verdict_epoch_200", verdict_200)
        exp.metric("gates", gate)
        exp.metric("verdict", verdict)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / "analysis.json").write_text(json.dumps(
            {"late_window": {"control": control["late_window"],
                             "arm_D": arm["late_window"]},
             "comparison": {"delta_epe_vs_control": d_epe, "delta_d1_vs_control": d_d1,
                            "materially_different": bool(material), "worse": worse,
                            "band": BAND},
             "stereo_verdict_epoch_200": verdict_200, "gates": gate,
             "verdict": verdict}, indent=2), encoding="utf-8")
        exp.conclude("Three-block refinement: dEPE {:+.4f} px, dD1 {:+.4f} pt vs the "
                     "six-block control; verdict {}.".format(d_epe, d_d1, verdict))
        print("\nVERDICT: " + verdict)
        print("recorded as " + exp.id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preflight", "train", "analyse"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _install()

    if args.stage == "preflight":
        result = run_preflight(device)
        (RESULT_DIR / "preflight.json").write_text(json.dumps(result, indent=2),
                                                   encoding="utf-8")
        for arm in sorted(result["arms"]):
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
        e3b.run_training("D", device)
        return

    run_analysis(device)


if __name__ == "__main__":
    main()
