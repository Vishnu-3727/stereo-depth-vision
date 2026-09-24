"""The E4 finetune intervention: retarget the init-sha guard to the E4 init.

One surgical edit to the bundle's copy of `scripts/run_arm.py`, and nothing
else. The E4 finetune recipe is byte-identical to E0 (batch 2, 200 epochs,
T_max 200, KITTI hailo_calib only); ONLY the init changes, from the Stage-1
checkpoint (sha 3ae6fb3b...) to the E4 pretrain's best-by-pretrain-val
checkpoint (sha 4c16fbe3...). But run_arm.py's post-hoc `wrong_checkpoint_resume`
guard still compares the init record's sha against the E0 constant
EXPECTED_SHA_ARMP_STAGE1, so a normally completed 200-epoch E4 finetune arm
STOPs with `wrong_checkpoint_resume` / returncode 3 after training has
already finished (seeds 0 and 1, 2026-09-23). The guard becomes a comparison
against the E4 init sha passed in as a build parameter — the same sha the
kernel asserts before training and the bundle records as expected_init_sha256,
never a second hardcoded constant in this file. The alternative considered —
comparing against `p2a_record.json`'s own init sha field — would compare the
file against itself (self-consistency) rather than against the init the
finetune was actually authorized to start from, so it is weaker and is not
used.

The E0 constant definition itself is left untouched: it is documentation of
what the reference arm means, and no training code reads it after this edit.
No other guard and no training code is changed.

Imported by build_bundle_e4ft.py; not executed directly.
"""
from __future__ import annotations

EDITS = [
    # The init-sha guard: expect the E4 pretrain checkpoint this finetune
    # was authorized to start from. {init_sha} is filled in by apply().
    ("""        if init_rec.get("sha256") != EXPECTED_SHA_ARMP_STAGE1:
""",
     """        if init_rec.get("sha256") != "{init_sha}":
""",
     "Init-sha guard compares against the E4 pretrain checkpoint sha the "
     "finetune kernel asserts before training, not the E0 Stage-1 constant, "
     "so a completed E4 finetune arm passes."),
]


def apply(text: str, init_sha: str) -> str:
    """Apply every edit exactly once, failing loudly if a target is missing."""
    init_sha = (init_sha or "").strip().lower()
    if len(init_sha) != 64 or any(c not in "0123456789abcdef" for c in init_sha):
        raise SystemExit("E4FT PATCH FAIL: init_sha must be a 64-char hex sha256")
    for old, new_tmpl, why in EDITS:
        if old not in text:
            raise SystemExit(
                "E4FT PATCH FAIL: target not found -> " + why + "\n" + repr(old[:90]))
        if text.count(old) != 1:
            raise SystemExit(
                f"E4FT PATCH FAIL: target appears {text.count(old)} times -> {why}")
        text = text.replace(old, new_tmpl.format(init_sha=init_sha), 1)
    return text
