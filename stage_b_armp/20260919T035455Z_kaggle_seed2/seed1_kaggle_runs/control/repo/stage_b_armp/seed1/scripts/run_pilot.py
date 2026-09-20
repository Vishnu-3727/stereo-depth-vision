"""TIER2-PILOT sequential orchestrator. Runs CONTROL then ARM-P (never concurrent),
each as a FRESH process (fresh optimizer/scheduler by construction), writes the
per-arm record immediately after each arm, then frozen-contract eval of both
arms' best+final checkpoints, then pilot_results.json + README append.

All writes atomic (.tmp then os.replace). No existing repo file is modified;
all output lands under the pilot dir. No P2A artifact is touched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
REPO = PILOT.parents[1]
RUN_ARM = PILOT / "scripts" / "run_arm.py"
EVAL = PILOT / "scripts" / "eval_tier2.py"
STAGE1_CKPT = (REPO / "stage_b_armp" / "20260918T062146Z_stage1_pretrain"
               / "checkpoints" / "armp_stage1_best.pth")
TIMEOUT_S = 7200.0
SPREAD_REF = 0.0444  # P2A three-seed EPE spread, reference only (no significance claim)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run_arm(arm: str, init: str) -> int:
    """One fresh process per arm. Returns exit code; record.json check is separate."""
    outdir = PILOT / arm
    log = open(PILOT / ("pipeline_%s.log" % arm), "a", encoding="utf-8")
    cmd = [sys.executable, str(RUN_ARM), "--arm", arm, "--init", init,
           "--outdir", str(outdir), "--timeout_s", str(TIMEOUT_S)]
    print("PIPELINE launching arm=%s init=%s" % (arm, init), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    dt = time.time() - t0
    log.write("arm=%s rc=%d orchestrator_wall_s=%.1f\n" % (arm, rc, dt))
    log.close()
    print("PIPELINE arm=%s exited rc=%d wall=%.1fs" % (arm, rc, dt), flush=True)
    return rc


def classify(delta_best: float, delta_final: float, stopped: list) -> tuple[str, str]:
    """Purely descriptive, mechanical rule stated verbatim in the output file.
    ONE seed: no significance, no CI, no extrapolation. Deltas smaller than the
    0.0444 reference spread are inside known run-to-run variation."""
    if stopped:
        return ("PILOT INVALID",
                "stop condition triggered on %s; no comparison is valid." % ",".join(stopped))
    if delta_best <= -SPREAD_REF and delta_final <= -SPREAD_REF:
        return ("PILOT SUPPORTS ARM-P INITIALIZATION",
                "both best-checkpoint and final-checkpoint deltas are negative beyond "
                "the %.4f reference spread (single-seed observation only)." % SPREAD_REF)
    if delta_best >= SPREAD_REF or delta_final >= SPREAD_REF:
        return ("PILOT CONTRADICTS ARM-P INITIALIZATION BENEFIT",
                "at least one selection shows ARM-P higher by >= %.4f (single-seed "
                "observation only)." % SPREAD_REF)
    return ("PILOT DOES NOT DEMONSTRATE INITIALIZATION BENEFIT",
            "delta(s) inside the %.4f reference spread, i.e. within known run-to-run "
            "variation; single seed, no significance claimed." % SPREAD_REF)


def main() -> int:
    t_all = time.time()
    stopped: list[str] = []

    rc = run_arm("control", "random")
    if rc != 0 or not (PILOT / "control" / "record.json").is_file():
        stopped.append("control(rc=%d)" % rc)
    else:
        print("PIPELINE control record.json present; proceeding to armp.", flush=True)

    rc = run_arm("armp", str(STAGE1_CKPT))
    if rc != 0 or not (PILOT / "armp" / "record.json").is_file():
        stopped.append("armp(rc=%d)" % rc)
    else:
        print("PIPELINE armp record.json present; proceeding to eval.", flush=True)

    eval_ok = False
    if not stopped:
        r = subprocess.run([sys.executable, str(EVAL)], cwd=str(REPO)).returncode
        eval_ok = (r == 0 and (PILOT / "tier2_eval.json").is_file())
        if not eval_ok:
            stopped.append("eval(rc=%d)" % r)

    # ---- assemble pilot_results.json (measurements only where measurable) ----
    ctrl_rec = json.loads((PILOT / "control" / "record.json").read_text(encoding="utf-8")) \
        if (PILOT / "control" / "record.json").is_file() else None
    armp_rec = json.loads((PILOT / "armp" / "record.json").read_text(encoding="utf-8")) \
        if (PILOT / "armp" / "record.json").is_file() else None
    ev = json.loads((PILOT / "tier2_eval.json").read_text(encoding="utf-8")) \
        if (PILOT / "tier2_eval.json").is_file() else None

    def frozen(arm: str, tag: str, key: str):
        if ev is None:
            return "NOT MEASURABLE WITH CURRENT ARTIFACTS"
        try:
            return ev["checkpoints"]["%s/%s" % (arm, tag)]["hailo_val"]["metrics"][key]
        except KeyError:
            return "NOT MEASURABLE WITH CURRENT ARTIFACTS"

    c_best_epe, c_best_d1 = frozen("control", "best", "epe"), frozen("control", "best", "d1")
    c_fin_epe, c_fin_d1 = frozen("control", "final", "epe"), frozen("control", "final", "d1")
    p_best_epe, p_best_d1 = frozen("armp", "best", "epe"), frozen("armp", "best", "d1")
    p_fin_epe, p_fin_d1 = frozen("armp", "final", "epe"), frozen("armp", "final", "d1")

    def delta(a, b):
        if isinstance(a, str) or isinstance(b, str):
            return "NOT MEASURABLE WITH CURRENT ARTIFACTS"
        return a - b

    d_best, d_final = delta(p_best_epe, c_best_epe), delta(p_fin_epe, c_fin_epe)
    if isinstance(d_best, str):
        classification, why = "PILOT INVALID", "frozen eval missing; deltas not measurable."
    else:
        classification, why = classify(d_best, d_final, stopped)

    # convergence observations, kept SEPARATE from final-error observations
    def curve(arm: str):
        p = PILOT / arm / "epoch_log.jsonl"
        if not p.is_file():
            return "NOT MEASURABLE WITH CURRENT ARTIFACTS"
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        val_pts = [{"epoch": r["epoch"], "val_epe": r["val_epe"], "val_d1": r["val_d1"]}
                   for r in rows if r.get("val_epe") is not None]
        return {"val_every_5_epochs": val_pts,
                "epoch_of_monitor_best": (ctrl_rec if arm == "control" else armp_rec or {})
                .get("train_monitor_best_epoch") if (ctrl_rec or armp_rec) else None}

    results = {
        "experiment": "ARM-P TIER-2 PILOT (single seed 0, 200 epochs/arm, sequential)",
        "seed": 0, "epochs_per_arm": 200,
        "control": {"record": ctrl_rec, "frozen_best": {"epe": c_best_epe, "d1": c_best_d1},
                    "frozen_final": {"epe": c_fin_epe, "d1": c_fin_d1}},
        "armp": {"record": armp_rec, "frozen_best": {"epe": p_best_epe, "d1": p_best_d1},
                 "frozen_final": {"epe": p_fin_epe, "d1": p_fin_d1}},
        "delta_EPE_best": d_best, "delta_EPE_final": d_final,
        "delta_sign_note": "negative means ARM-P lower.",
        "reference_spread": {"p2a_three_seed_epe": [1.4149796, 1.4485736, 1.4593946],
                             "spread": SPREAD_REF,
                             "note": ("single-seed deltas smaller than this are inside known "
                                      "run-to-run variation; no significance claimed.")},
        "convergence_observations": {"control": curve("control"), "armp": curve("armp"),
                                     "note": ("convergence (epoch of best, early val EPE) is "
                                              "reported SEPARATELY from final/best error; curve "
                                              "shape is not interpreted as mechanism.")},
        "frozen_eval": ev,
        "classification": classification, "classification_why": why,
        "classification_rule": ("mechanical only: INVALID if any stop; SUPPORTS iff best AND final "
                                "deltas <= -0.0444; CONTRADICTS iff any delta >= +0.0444; else DOES "
                                "NOT DEMONSTRATE. One seed; no CI; no extrapolation."),
        "stop_conditions_triggered": stopped if stopped else "none",
        "pipeline_wall_s": time.time() - t_all,
        "interpretation_limits": ("ONE seed. No statistical significance. No confidence intervals. "
                                  "No extrapolation to general behaviour. Small differences are not "
                                  "called meaningful. Faster/lower-final/lower-best/delayed-"
                                  "convergence/overfitting reported as SEPARATE observations. No "
                                  "follow-up experiment started. Not Phase 3; no historical verdict "
                                  "rewritten."),
    }
    atomic_write_text(PILOT / "pilot_results.json", json.dumps(results, indent=2))
    print("PIPELINE wrote pilot_results.json classification=%s" % classification, flush=True)

    # ---- README append (atomic: read, append, .tmp, rename) ----
    rd = PILOT / "README.md"
    sec = ("\n## Tier-2 pilot run (seed 0, 200 epochs/arm, sequential)\n\n"
           "UTC: %s. CONTROL first (random init), then ARM-P (Stage-1 init `%s`).\n\n"
           "Per-arm: `control/` and `armp/` hold `stdout.log`, `training_log.jsonl` (script output),\n"
           "`epoch_log.jsonl` (same rows plus `epoch_wall_s`; val null except val epochs),\n"
           "`p2a_best.pth`/`p2a_final.pth` with `.sha256` sidecars, `integrity_guard.json`,\n"
           "`p2a_record.json` (script output) and `record.json` (written immediately post-arm).\n"
           "Frozen-contract scores (mirror of `phase2/scripts/eval_p2a.py`, unmodified):\n"
           "`tier2_eval.json`. Aggregate: `pilot_results.json`.\n\n"
           "CONTROL frozen best EPE/D1: %s / %s; final EPE/D1: %s / %s; monitor-best epoch: %s; wall: %ss.\n\n"
           "ARM-P frozen best EPE/D1: %s / %s; final EPE/D1: %s / %s; monitor-best epoch: %s; wall: %ss.\n\n"
           "delta_EPE best: %s; delta_EPE final: %s (negative = ARM-P lower; reference spread 0.0444).\n\n"
           "Classification: **%s** — %s\n\nStop conditions: %s. Pipeline wall: %.1f s.\n" % (
               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               STAGE1_CKPT.name,
               c_best_epe, c_best_d1, c_fin_epe, c_fin_d1,
               (ctrl_rec or {}).get("train_monitor_best_epoch"),
               round((ctrl_rec or {}).get("training_duration_s", 0) or 0, 1),
               p_best_epe, p_best_d1, p_fin_epe, p_fin_d1,
               (armp_rec or {}).get("train_monitor_best_epoch"),
               round((armp_rec or {}).get("training_duration_s", 0) or 0, 1),
               d_best, d_final, classification, why,
               stopped if stopped else "none", time.time() - t_all))
    cur = rd.read_text(encoding="utf-8")
    atomic_write_text(rd, cur + sec)
    print("PIPELINE README appended.", flush=True)
    return 0 if classification != "PILOT INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
