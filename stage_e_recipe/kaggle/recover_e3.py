#!/usr/bin/env python
"""Recover the three ERRORed E3 Kaggle seeds offline (seeds 0, 1 and 2).

All three `stage-e-e3-seed0`, `stage-e-e3-seed1` and `stage-e-e3-seed2` trained all 400 epochs normally
and then STOPped with `epochs_incomplete / {rows: 400}` / returncode 3, because
run_arm.py's two post-hoc guards hardcoded the literal 200. The guards are
read-only checks that run AFTER training, so the training itself is unaffected
and the results are valid.

This script re-runs the SAME post-hoc guards as run_arm.py — the guard block
is sliced verbatim out of run_arm.py (canonical source plus e3_patch, i.e.
exactly what bundle_e3 ships) and exec'd, with the expected epoch count 400
supplied as the arm's --epochs argument — against the downloaded evidence:

    stage_e_recipe/kaggle/e3_output/seed{0,1,2}/repo/stage_e_recipe/run/armp/

and writes `stage_e_recipe/e3_recovery.json` recording, per seed: every guard
checked and whether it passed, the row count, the best10 EPE and the epoch it
came from, and the final verdict (RECOVERED or genuinely STOPPED).

Read-only against the evidence: the only file this script writes is
e3_recovery.json. run_arm.py's two write side-effects inside the guard block
(stop() -> STOP.json, atomic_write_text -> *.sha256 sidecars) are replaced in
the exec namespace by in-memory recorders, so e3_output/ is never touched.
The decision logic — every condition, comparison, file read and hash check —
is verbatim run_arm.py.

No training runs, no GPU work, no Kaggle calls.

    python stage_e_recipe/kaggle/recover_e3.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "stage_e_recipe" / "e3_recovery.json"

# The same source build_bundle.py assembles the bundle from.
RUN_ARM_SRC = (REPO / "stage_b_armp" / "20260919T012646Z_tier2_seed1"
               / "scripts" / "run_arm.py")

EXPECTED_EPOCHS = 400  # E3 = 400 epochs; supplied as the arm's --epochs argument
SEEDS = (0, 1, 2)

GUARD_START = "# ---- post-hoc guards (all read-only checks on the finished arm) ----"
GUARD_END = "# ---- enriched per-epoch log: required columns + epoch_wall_s ----"

# stop() reason -> guard name, in the order run_arm.py checks them. The two
# epochs_incomplete guards are told apart by their detail payload.
GUARDS_IN_ORDER = [
    "rows_count",
    "nan_inf_loss_in_log",
    "zero_valid_pixels",
    "nan_inf_val_metric",
    "param_or_architecture_changed",
    "epochs_run",
    "wrong_checkpoint_resume",
    "checkpoint_sha_mismatch",
]


class _Stopped(Exception):
    def __init__(self, reason: str, detail: dict):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patched_run_arm_text() -> str:
    """run_arm.py as bundle_e3 ships it: canonical source plus the e3_patch.

    The orchestration edits build_bundle.py also applies touch only the
    argparse/env preamble, never the guard block, so patching the canonical
    source in memory yields the identical post-hoc guards.
    """
    e3_patch = load_module("e3_patch", HERE / "e3_patch.py")
    return e3_patch.apply(RUN_ARM_SRC.read_text(encoding="utf-8"))


def guard_block(text: str) -> str:
    """Slice the verbatim post-hoc guard block out of run_arm.py text."""
    if GUARD_START not in text or GUARD_END not in text:
        raise SystemExit("RECOVER FAIL: guard block markers not found in run_arm.py")
    inner = text.split(GUARD_START, 1)[1].split(GUARD_END, 1)[0]
    # The block lives inside main(), so it carries a 4-space indent; dedent
    # it for top-level exec. Content is otherwise byte-identical.
    return textwrap.dedent(inner)


def run_posthoc_guards(outdir: Path, expected_epochs: int,
                       run_arm_text: str | None = None) -> dict:
    """Exec run_arm.py's verbatim post-hoc guards against outdir.

    Returns per-guard pass/fail/not_reached plus the stop reason when a guard
    fires. Never writes into outdir: stop() and atomic_write_text() are
    replaced by in-memory recorders; every check itself is unmodified.
    """
    if run_arm_text is None:
        run_arm_text = patched_run_arm_text()
    run_arm = load_module("run_arm_guards_ref", RUN_ARM_SRC)

    rec_pre = json.loads((outdir / "p2a_record.json").read_text(encoding="utf-8"))
    init_source = rec_pre.get("config", {}).get("init_record", {}).get("source")
    args = SimpleNamespace(epochs=expected_epochs, arm="armp",
                           init="random" if init_source == "random" else "stage1-ckpt")

    captured_writes: dict = {}

    def stop(outdir_, reason: str, detail: dict):
        raise _Stopped(reason, detail)

    def atomic_write_text(path, text_):
        captured_writes[str(path)] = text_

    ns = {"json": json, "math": math, "Path": Path,
          "outdir": outdir, "args": args,
          "EXPECTED_PARAMS": run_arm.EXPECTED_PARAMS,
          "EXPECTED_SHA_ARMP_STAGE1": run_arm.EXPECTED_SHA_ARMP_STAGE1,
          "sha256_file": run_arm.sha256_file,
          "stop": stop, "atomic_write_text": atomic_write_text}

    fired: _Stopped | None = None
    try:
        # The block is main()'s body (it uses `return`), so exec it wrapped
        # in a function shim and call it. The guard statements themselves are
        # verbatim; only this scaffolding is added. stop() raises _Stopped,
        # so the `return` values are never used.
        shim = ("def _posthoc_guards():\n"
                + textwrap.indent(guard_block(run_arm_text), "    ")
                + "\n_posthoc_guards()\n")
        exec(compile(shim, "run_arm_posthoc_guards", "exec"), ns)
    except _Stopped as s:
        fired = s

    if fired is None:
        return {"guards": {g: "pass" for g in GUARDS_IN_ORDER},
                "stop": None, "captured_writes": captured_writes}

    if fired.reason == "epochs_incomplete" and "rows" in fired.detail:
        failed = "rows_count"
    elif fired.reason == "epochs_incomplete" and "epochs_run" in fired.detail:
        failed = "epochs_run"
    else:
        reason_to_guard = {
            "nan_inf_loss_in_log": "nan_inf_loss_in_log",
            "zero_valid_pixels": "zero_valid_pixels",
            "nan_inf_val_metric": "nan_inf_val_metric",
            "param_or_architecture_changed": "param_or_architecture_changed",
            "wrong_checkpoint_resume": "wrong_checkpoint_resume",
            "checkpoint_sha_mismatch": "checkpoint_sha_mismatch",
        }
        if fired.reason not in reason_to_guard:
            raise SystemExit(f"RECOVER FAIL: unknown stop reason {fired.reason!r}")
        failed = reason_to_guard[fired.reason]

    guards = {}
    seen_fail = False
    for g in GUARDS_IN_ORDER:
        if g == failed:
            guards[g] = "fail"
            seen_fail = True
        elif not seen_fail:
            guards[g] = "pass"
        else:
            guards[g] = "not_reached"
    return {"guards": guards,
            "stop": {"reason": fired.reason, "detail": fired.detail},
            "captured_writes": captured_writes}


def recover_seed(seed: int, run_arm_text: str) -> dict:
    outdir = (HERE / "e3_output" / f"seed{seed}" / "repo"
              / "stage_e_recipe" / "run" / "armp")
    for f in ("training_log.jsonl", "p2a_record.json", "integrity_guard.json",
              "p2a_best.pth"):
        if not (outdir / f).is_file():
            raise SystemExit(f"RECOVER FAIL: seed{seed} missing {f}")

    result = run_posthoc_guards(outdir, EXPECTED_EPOCHS, run_arm_text)

    rows = [json.loads(x) for x in
            (outdir / "training_log.jsonl").read_text(encoding="utf-8")
            .splitlines() if x.strip()]
    # The in-training monitor logs val_epe/val_d1 every 5 epochs; the best10
    # comes from the rows that carry it.
    scored = [r for r in rows if "val_epe" in r]
    best_row = min(scored, key=lambda r: r["val_epe"])
    rec = json.loads((outdir / "p2a_record.json").read_text(encoding="utf-8"))
    prior_stop = json.loads((outdir / "STOP.json").read_text(encoding="utf-8"))

    if result["stop"] is None:
        verdict = "RECOVERED"
        stop_field = None
    else:
        verdict = "STOPPED"
        stop_field = result["stop"]

    return {
        "seed": seed,
        "expected_epochs": EXPECTED_EPOCHS,
        "guards": result["guards"],
        "row_count": len(rows),
        "scored_rows": len(scored),
        "best10_epe": best_row["val_epe"],
        "best10_epoch": best_row["epoch"],
        "record_best10_epe": rec.get("best_val_epe_10scene"),
        "record_best_epoch": rec.get("best_epoch"),
        "kaggle_stop": {"reason": prior_stop.get("reason"),
                        "detail": prior_stop.get("detail")},
        "verdict": verdict,
        "stop": stop_field,
    }


def main() -> None:
    run_arm_text = patched_run_arm_text()
    # Sanity: the guards under test really are the epoch-parameterised ones.
    if "if len(rows) != args.epochs:" not in run_arm_text:
        raise SystemExit("RECOVER FAIL: patched row-count guard not found")
    if 'if rec.get("epochs_run") != args.epochs:' not in run_arm_text:
        raise SystemExit("RECOVER FAIL: patched epochs_run guard not found")

    record = {
        "experiment": "stage-e-e3",
        "method": ("verbatim run_arm.py post-hoc guard block (canonical source "
                   "+ e3_patch, as bundle_e3 ships it), exec'd with "
                   "args.epochs=400; write side-effects captured in memory, "
                   "evidence dirs untouched"),
        "seeds": {str(s): recover_seed(s, run_arm_text) for s in SEEDS},
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    for s in SEEDS:
        r = record["seeds"][str(s)]
        print(f"seed{s}: rows={r['row_count']} "
              f"best10={r['best10_epe']:.4f}@{r['best10_epoch']} "
              f"verdict={r['verdict']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
