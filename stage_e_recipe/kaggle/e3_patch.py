"""The E3 intervention: 400 epochs instead of 200.

Two surgical edits to the bundle's copy of `scripts/run_arm.py`, and nothing
else. The E3 epoch count itself needs no patch: run_experiment.py already
passes it as `--epochs` (EPOCHS[e3] = 400 in push_run.py), the orchestration
edits already forward it as P2A_EPOCHS, and finetune_pilot.py reads it from
the environment — so the recipe file stays byte-identical to E0, and the
architecture, initialization, data, augmentation, loss, optimizer, learning
rate, batch size and scheduler are unchanged.

What IS patched is the defect the 400-epoch runs exposed: run_arm.py's two
post-hoc `epochs_incomplete` guards hardcode the literal 200, so a normally
completed 400-epoch arm STOPs with `{rows: 400}` / returncode 3 after training
has already finished. Both guards become comparisons against `args.epochs` —
the arm's own configured epoch count, already in scope in main() via the
`--epochs` argument the orchestration edits declare (default 200, so omitting
the flag reproduces the frozen recipe). No second literal is introduced: a
hardcoded 400 would merely move the defect to the next epoch count, while
`args.epochs` is correct for 200, 400, and any future value. The alternative
considered — comparing against `p2a_record.json`'s `config.epochs` — would
compare two fields of the same file against each other (self-consistency)
rather than against what the arm was actually asked to run, so it is weaker
and is not used.

Imported by build_bundle.py; not executed directly.
"""
from __future__ import annotations

EPOCHS = 400

EDITS = [
    # 1. The training-log row-count guard: expect the configured epoch count.
    ("""    if len(rows) != 200:
        return stop(outdir, "epochs_incomplete", {"rows": len(rows)})
""",
     """    if len(rows) != args.epochs:
        return stop(outdir, "epochs_incomplete", {"rows": len(rows)})
""",
     "Row-count guard compares against the arm's --epochs argument, not the "
     "literal 200, so a completed 400-epoch log passes."),

    # 2. The record epochs_run guard: same derivation, same reason.
    ("""    if rec.get("epochs_run") != 200:
        return stop(outdir, "epochs_incomplete", {"epochs_run": rec.get("epochs_run")})
""",
     """    if rec.get("epochs_run") != args.epochs:
        return stop(outdir, "epochs_incomplete", {"epochs_run": rec.get("epochs_run")})
""",
     "Record epochs_run guard compares against the arm's --epochs argument, "
     "not the literal 200."),
]


def apply(text: str) -> str:
    """Apply every edit exactly once, failing loudly if a target is missing."""
    for old, new, why in EDITS:
        if old not in text:
            raise SystemExit(
                "E3 PATCH FAIL: target not found -> " + why + "\n" + repr(old[:90]))
        if text.count(old) != 1:
            raise SystemExit(
                f"E3 PATCH FAIL: target appears {text.count(old)} times -> {why}")
        text = text.replace(old, new, 1)
    return text
