#!/usr/bin/env python
"""Stage E4 one-shot finish: run once seed 2's finetune kernel is done.

When `stage-e-e4-finetune-seed2` finishes, a single command completes E4:

    python stage_e_recipe/kaggle/finish_e4.py

Steps, in order, stopping at the first failure with the exact error:

    (a) status-finetune 2: if the kernel is not COMPLETE or ERROR (e.g.
        still RUNNING), print that and exit 0 — nothing else runs.
    (b) pull-finetune 2 into stage_e_recipe/kaggle/e4_output/seed2/.
    (c) recover_e4 --seeds 2 (only when there is training evidence to
        recover: a STOPPED record, or a COMPLETE pull that still ships the
        repo/ training tree), then complete_e4 --seeds 2, which scores seed 2
        locally on the same device as seeds 0/1 and handles both Kaggle
        outcomes (STOPPED -> kaggle_stopped_seed2.json; COMPLETE ->
        kaggle_record_seed2.json plus kaggle_t4 provenance).
    (d) int8_control.py e4 (re-runs all three seeds; overwrites int8_e4.json
        by design).
    (e) verdict.py e4.
    (f) summary: per-seed local best/final EPE, Kaggle-vs-local diff for
        seed 2 when Kaggle T4 provenance exists, INT8 P, verdict + reason.

No training, no Kaggle pushes, no commits, no doc edits. Each step shells
out to the script that owns it; this script adds no scoring or guard logic.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = sys.executable

SEED = 2
SEED_DIR = HERE / "e4_output" / f"seed{SEED}"
PULLED_RECORD = SEED_DIR / f"e4_seed{SEED}.json"
ARMP_LOG = (SEED_DIR / "repo" / "stage_e_recipe" / "run" / "armp"
            / "training_log.jsonl")


def run_step(label: str, cmd: list[str]) -> str:
    """Run one step, echoing the command and its output. Returns output.

    Exits with the step's return code and the exact output on failure.
    """
    print(f"=== [{label}] {' '.join(cmd)} ===", flush=True)
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                       errors="replace")
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    if out:
        print(out[-3000:], flush=True)
    if p.returncode != 0:
        raise SystemExit(
            f"FINISH_E4 FAIL at step {label} (exit {p.returncode}):\n{out}")
    return out


def step_status() -> str:
    """Step (a): report the seed-2 kernel state; '' if not ready."""
    out = run_step("a/status-finetune-2",
                   [PY, str(HERE / "push_e4.py"), "status-finetune",
                    str(SEED)])
    low = out.lower()
    if "error" in low:
        print("seed 2 kernel state: ERROR", flush=True)
        return "ERROR"
    if "complete" in low:
        print("seed 2 kernel state: COMPLETE", flush=True)
        return "COMPLETE"
    for state in ("running", "queued", "pending", "retry"):
        if state in low:
            print(f"seed 2 kernel still {state.upper()}; E4 finish waits — "
                  f"re-run this script once it is COMPLETE/ERROR", flush=True)
            return ""
    raise SystemExit(
        f"FINISH_E4 FAIL at step a/status-finetune-2: could not determine "
        f"kernel state from output:\n{out}")


def step_pull() -> None:
    run_step("b/pull-finetune-2",
             [PY, str(HERE / "push_e4.py"), "pull-finetune", str(SEED)])
    if not PULLED_RECORD.is_file():
        raise SystemExit(
            f"FINISH_E4 FAIL at step b/pull-finetune-2: pulled output has no "
            f"{PULLED_RECORD}")


def step_recover_and_complete() -> None:
    rec = json.loads(PULLED_RECORD.read_text(encoding="utf-8"))
    status = rec.get("status")
    if status == "STOPPED" or ARMP_LOG.is_file():
        run_step("c/recover-seed2",
                 [PY, str(HERE / "recover_e4.py"), "--seeds", str(SEED)])
    elif status == "COMPLETE":
        print("seed 2 Kaggle record already COMPLETE with no repo/ training "
              "tree to recover; skipping recover_e4", flush=True)
    else:
        raise SystemExit(
            f"FINISH_E4 FAIL at step c: unexpected pulled status "
            f"{status!r} in {PULLED_RECORD}")
    run_step("c/complete-seed2",
             [PY, str(HERE / "complete_e4.py"), "--seeds", str(SEED)])


def main() -> None:
    state = step_status()
    if not state:
        return
    step_pull()
    step_recover_and_complete()
    run_step("d/int8-e4", [PY, str(HERE.parents[0] / "int8_control.py"), "e4"])
    run_step("e/verdict-e4", [PY, str(HERE.parents[0] / "verdict.py"), "e4"])
    print_summary()


def per_seed_scores() -> dict[int, dict]:
    out = {}
    for f in sorted((HERE / "e4_output").glob("seed*/e4_seed*.json")):
        if f.name.startswith("kaggle_"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("status") != "COMPLETE" or "checkpoints" not in d:
            continue
        out[int(d["seed"])] = {
            "best": d["checkpoints"]["best"]["hailo_val"]["metrics"]["epe"],
            "final": d["checkpoints"]["final"]["hailo_val"]["metrics"]["epe"],
        }
    return out


def print_summary() -> None:
    print("=== [f/summary] ===", flush=True)
    scores = per_seed_scores()
    for seed in sorted(scores):
        print(f"seed{seed}: local best EPE {scores[seed]['best']:.7f}  "
              f"final EPE {scores[seed]['final']:.7f}", flush=True)
    record = SEED_DIR / f"kaggle_record_seed{SEED}.json"
    if record.is_file():
        kag = json.loads(record.read_text(encoding="utf-8"))
        kb = kag["checkpoints"]["best"]["hailo_val"]["metrics"]["epe"]
        kf = kag["checkpoints"]["final"]["hailo_val"]["metrics"]["epe"]
        lb, lf = scores[SEED]["best"], scores[SEED]["final"]
        print(f"seed{SEED} Kaggle T4 best {kb:.7f} vs local {lb:.7f} "
              f"(diff {lb - kb:+.7f})", flush=True)
        print(f"seed{SEED} Kaggle T4 final {kf:.7f} vs local {lf:.7f} "
              f"(diff {lf - kf:+.7f})", flush=True)
    else:
        stopped = SEED_DIR / f"kaggle_stopped_seed{SEED}.json"
        if stopped.is_file():
            print(f"seed{SEED}: Kaggle STOPPED (no T4 scores); preserved in "
                  f"{stopped.name}, scored locally", flush=True)
    int8_path = HERE.parents[0] / "int8_e4" / "int8_e4.json"
    int8 = json.loads(int8_path.read_text(encoding="utf-8"))
    print(f"INT8 P_candidate {int8['P_candidate']:.7f} vs threshold "
          f"{int8['gate_threshold']:.7f} -> {int8['int8_gate']}", flush=True)
    verdict = json.loads(
        (HERE.parents[0] / "e4_verdict.json").read_text(encoding="utf-8"))
    print(f"VERDICT: {verdict['verdict']}\n{verdict['reason']}", flush=True)


if __name__ == "__main__":
    main()
