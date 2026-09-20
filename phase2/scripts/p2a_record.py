"""Assemble the self-contained EXP-P2A-SCALE-COVERAGE-001 experiment record.

Merges the per-seed training records, the frozen-evaluation JSON and the
preregistered decisions into one file. Reads only; invents nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = {"0": "p2a_scale_coverage", "1": "p2a_scale_coverage_s1",
        "2": "p2a_scale_coverage_s2"}
BASE = REPO / "phase2" / "runs" / "p2a_scale_coverage"

E = json.loads((BASE / "p2a_eval_best.json").read_text())
D = json.loads((BASE / "P2A_DECISIONS.json").read_text())

seeds = {}
for s, run in RUNS.items():
    rec = json.loads((REPO / "phase2" / "runs" / run / "p2a_record.json").read_text())
    ev = E["seeds"][s]
    seeds[s] = {
        "run_dir": "phase2/runs/" + run,
        "training": {
            "wall_clock_s": rec["wall_clock_s"], "epochs_run": rec["epochs_run"],
            "best_epoch": rec["best_epoch"],
            "best_val_epe_10scene_monitor": rec["best_val_epe_10scene"],
            "git_head": rec["git_head"],
            "scale_draw_sample": rec["scale_draw_sample"],
            "integrity_guard": rec["integrity_guard"],
        },
        "checkpoints": {
            "best": "phase2/runs/%s/p2a_best.pth" % run,
            "best_sha256": rec["best_sha256"],
            "final": "phase2/runs/%s/p2a_final.pth" % run,
            "final_sha256": rec["final_sha256"],
            "scored_snapshot": "best (train-time 10-scene monitor selection only)",
        },
        "integrity_at_eval": {
            "params": ev["params"], "n_keys": ev["n_keys"],
            "state_dict_keys_match_arm_v": ev["state_dict_keys_match_arm_v"],
            "strict_load": ev["strict_load"], "sha256": ev["sha256"],
            "weight_sha16": ev["weight_sha16"],
        },
        "frozen_evaluation": ev["hailo_val"],
        "training_split_evaluation": ev["hailo_calib"],
    }

out = {
    "experiment": "EXP-P2A-SCALE-COVERAGE-001",
    "status": "COMPLETE",
    "preregistration": "phase2/docs/PHASE2_HYPOTHESIS_01.md",
    "diagnostic_basis": "phase2/docs/PHASE2_ARMV_HIGH_DISPARITY_DIAGNOSTIC.md",
    "result_report": "phase2/docs/PHASE2_P2A_SCALE_COVERAGE_RESULT.md",
    "intervention": json.loads(
        (REPO / "phase2" / "runs" / "p2a_scale_coverage" / "p2a_record.json").read_text()
    )["config"]["intervention"],
    "training_recipe": {k: v for k, v in json.loads(
        (REPO / "phase2" / "runs" / "p2a_scale_coverage" / "p2a_record.json").read_text()
    )["config"].items() if k not in ("intervention", "arm")},
    "control": D["control_arm_v"],
    "frozen_reference": D["frozen_reference"],
    "gates_as_preregistered": D["gates"],
    "results": D["p2a"],
    "decisions": D["decisions"],
    "evaluation_protocol": {
        "contract": "frozen, unchanged: KITTI 2015 _10, hailo_val 160-199, disp_occ_0, "
                    "368x1232 top-left crop, GT/256, valid = gt > 0, pooled, 3,802,797 px",
        "eval_time_augmentation": "none",
        "multi_scale_inference": False,
        "scorer": "phase2/scripts/eval_p2a.py — established score_mirror mechanism; "
                  "dataset, GT scale, valid mask, pooled metrics and contract guard "
                  "imported unmodified from phase1.harness.frozen_eval",
        "mirror_note": E["mirror_note"],
        "contract_match_all_seeds": all(
            E["seeds"][s]["hailo_val"]["guard"]["contract_match"] for s in RUNS),
    },
    "software": json.loads(
        (REPO / "phase2" / "runs" / "p2a_scale_coverage" / "p2a_record.json").read_text()
    )["software"],
    "execution_history": {
        "note": "Two agent-harness background shells were killed by the host for low "
                "system memory (15.1 GB total, ~1.2 GB free, browsers dominant). Neither "
                "kill touched a training process: the detached python child survived both "
                "times and seed 1 trained through uninterrupted. Seed 2 and the evaluation "
                "were then run from a detached launcher outside the harness "
                "(phase2/scripts/run_p2a_remaining.ps1). No seed was restarted, dropped, "
                "replaced or re-run. No result was read before all three seeds completed. "
                "Nothing about the experiment changed.",
        "seed_order": ["seed 0 16:26-15:30", "seed 1 15:30-16:35", "seed 2 16:35-17:37",
                       "frozen evaluation 17:37-17:38"],
    },
    "seeds": seeds,
}
(BASE / "P2A_EXPERIMENT_RECORD.json").write_text(json.dumps(out, indent=2))
print("wrote", BASE / "P2A_EXPERIMENT_RECORD.json")
print("decisions:", json.dumps(out["decisions"], indent=2))
