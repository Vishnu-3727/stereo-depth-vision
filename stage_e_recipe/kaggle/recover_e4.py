#!/usr/bin/env python
"""Recover the two STOPPED E4 finetune Kaggle seeds offline (seeds 0 and 1).

Both `stage-e-e4-finetune-seed0` and `stage-e-e4-finetune-seed1` trained all
200 epochs normally and then STOPped with `wrong_checkpoint_resume` /
returncode 3, because run_arm.py's post-hoc init-sha guard still compared the
init record's sha against the E0 Stage-1 constant (3ae6fb3b...) while the E4
finetune starts from the E4 pretrain's best-by-pretrain-val checkpoint
(4c16fbe3...). The guard is a read-only check that runs AFTER training, so
the training itself is unaffected and the results are valid.

This script re-runs the SAME post-hoc guards as run_arm.py — the guard block
is sliced verbatim out of the PATCHED rebuilt bundle's run_arm.py (i.e.
exactly what the fixed bundle_e4ft ships: E3's epoch guards plus the
e4ft_patch init-sha retarget) and exec'd, with the expected epoch count 200
supplied as the arm's --epochs argument — against the downloaded evidence:

    stage_e_recipe/kaggle/e4ft_err_seed{0,1}/repo/stage_e_recipe/run/armp/

and writes `stage_e_recipe/e4_recovery.json` recording, per seed: every guard
checked and whether it passed, the row count, the best10 EPE and the epoch it
came from, and the final verdict (RECOVERED or genuinely STOPPED).

Read-only against the evidence: the only file this script writes is
e4_recovery.json. run_arm.py's two write side-effects inside the guard block
(stop() -> STOP.json, atomic_write_text -> *.sha256 sidecars) are replaced in
the exec namespace by in-memory recorders, so e4ft_err_seed*/ is never
touched. The decision logic — every condition, comparison, file read and hash
check — is verbatim run_arm.py.

No training runs, no GPU work, no Kaggle calls.

    python stage_e_recipe/kaggle/recover_e4.py
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
OUT = REPO / "stage_e_recipe" / "e4_recovery.json"

# The patched artefact itself: exactly what the rebuilt bundle_e4ft ships.
RUN_ARM_PATCHED = HERE / "bundle_e4ft" / "scripts" / "run_arm.py"

EXPECTED_EPOCHS = 200  # E4 finetune = E0 recipe; supplied as --epochs
EXPECTED_INIT_SHA = "4c16fbe32724d0c1157aa91b0503f67e7e1576faf597325d62c87510aa6b829f"
SEEDS = (0, 1)

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
    """run_arm.py as the rebuilt bundle_e4ft ships it, read verbatim.

    The patch (e3 epoch guards + e4ft init-sha retarget) is baked into the
    file by build_bundle_e4ft.py, so no in-memory patching is needed — what
    is sliced below is byte-for-byte the shipped artefact.
    """
    return RUN_ARM_PATCHED.read_text(encoding="utf-8")


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
    run_arm = load_module("run_arm_guards_ref", RUN_ARM_PATCHED)

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
    outdir = (HERE / f"e4ft_err_seed{seed}" / "repo"
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
        "expected_init_sha": EXPECTED_INIT_SHA,
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
    # Sanity: the guards under test really are the patched ones.
    if "if len(rows) != args.epochs:" not in run_arm_text:
        raise SystemExit("RECOVER FAIL: patched row-count guard not found")
    if 'if rec.get("epochs_run") != args.epochs:' not in run_arm_text:
        raise SystemExit("RECOVER FAIL: patched epochs_run guard not found")
    if f'if init_rec.get("sha256") != "{EXPECTED_INIT_SHA}":' not in run_arm_text:
        raise SystemExit("RECOVER FAIL: retargeted init-sha guard not found")

    record = {
        "experiment": "stage-e-e4",
        "method": ("verbatim run_arm.py post-hoc guard block from the rebuilt "
                   "bundle_e4ft (canonical source + e3_patch epoch guards + "
                   "e4ft_patch init-sha retarget, as the fixed bundle ships "
                   "it), exec'd with args.epochs=200; write side-effects "
                   "captured in memory, evidence dirs untouched"),
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
