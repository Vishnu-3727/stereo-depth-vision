"""The E2 intervention: native physical batch 8.

One edit. `BATCH = 2` becomes `BATCH = 8`, and nothing else moves: same
architecture, initialization, data, augmentation, loss, optimizer, learning
rate (1e-3, deliberately NOT scaled), epochs and scheduler as E0.

Native batch 8 rather than 4x accumulated batch 2, per amendment A1.1: ARM-P
has no BatchNorm, so the two are mathematically identical, and the native form
removes accumulation semantics as an implementation variable that could be got
wrong or quietly redefined later. Gate G6 measured batch 8 on the Kaggle T4 at
5,766 MiB reserved of 14,911, a 61.3% headroom against the 20% threshold fixed
before the measurement.

Consequence of the intervention, not a side change: an epoch now contains 20
optimizer steps instead of 80, because the 160-scene dataset is unchanged and
each step consumes four times as many samples. The scheduler still steps once
per epoch with T_max = 200.

Imported by build_bundle.py; not executed directly.
"""
from __future__ import annotations

BATCH = 8

EDITS = [
    ("BATCH = 2\n",
     "BATCH = 8  # STAGE E / E2 INTERVENTION: native batch 8 (was 2)\n",
     "Physical batch 2 -> 8; LR, epochs, scheduler and everything else unchanged."),
]


def apply(text: str) -> str:
    """Apply every edit exactly once, failing loudly if a target is missing."""
    for old, new, why in EDITS:
        if old not in text:
            raise SystemExit(
                "E2 PATCH FAIL: target not found -> " + why + "\n" + repr(old))
        if text.count(old) != 1:
            raise SystemExit(
                f"E2 PATCH FAIL: target appears {text.count(old)} times -> {why}")
        text = text.replace(old, new, 1)
    return text
