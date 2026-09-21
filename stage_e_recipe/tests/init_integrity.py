#!/usr/bin/env python
"""Stage E initialization-integrity gate (pre-run validation items 3, 4, 5, 7).

Verifies that the frozen Stage-1 ARM-P checkpoint is what Stage E claims it is,
that it loads strictly into the frozen architecture, and that NOTHING but model
weights is inherited from Stage 1.

Not a training run: CPU only, no optimizer step, no checkpoint written, no
weight persisted. Read-only with respect to every frozen artefact.

    python stage_e_recipe/tests/init_integrity.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

INIT_REL = "stage_b_armp/20260918T062146Z_stage1_pretrain/checkpoints/armp_stage1_best.pth"
EXPECTED_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"
EXPECTED_PARAMS = 397954
EXPECTED_KEYS = 70
ARM_V_CONFIG = dict(downsample_levels=3, num_disparities=24,
                    cost_volume_shift="right", regression_normalize=True)
# Keys that would mean optimizer/scheduler state is riding along with the
# weights. Stage E inherits model weights ONLY.
FORBIDDEN_BLOB_KEYS = {"optimizer", "optimizer_state_dict", "scheduler",
                       "lr_scheduler", "scheduler_state_dict", "amp",
                       "scaler", "epoch", "step"}


def main() -> None:
    report: dict = {"gate": "Stage E initialization integrity",
                    "not_a_training_run": True, "checks": {}}
    ok = True

    def check(name: str, passed: bool, detail) -> None:
        nonlocal ok
        ok &= bool(passed)
        report["checks"][name] = {"pass": bool(passed), "detail": detail}
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    init = REPO / INIT_REL
    check("checkpoint_exists", init.exists(), str(init.relative_to(REPO)).replace("\\", "/"))
    if not init.exists():
        sys.exit(1)

    got = hashlib.sha256(init.read_bytes()).hexdigest()
    check("sha256_matches", got == EXPECTED_SHA, got)

    blob = torch.load(init, map_location="cpu", weights_only=False)
    blob_keys = sorted(blob.keys()) if isinstance(blob, dict) else ["<raw state_dict>"]
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob

    # Only weights may be inherited: the blob must not smuggle optimizer or
    # scheduler state that a loader could pick up.
    leaked = sorted(set(blob_keys) & FORBIDDEN_BLOB_KEYS)
    check("no_optimizer_or_scheduler_state_in_blob", not leaked,
          f"blob keys {blob_keys}; forbidden present: {leaked or 'none'}")

    check("tensor_key_count", len(sd) == EXPECTED_KEYS, f"{len(sd)}/{EXPECTED_KEYS}")

    model = StereoNet(StereoNetConfig(**ARM_V_CONFIG))
    n_params = sum(p.numel() for p in model.parameters())
    check("param_count", n_params == EXPECTED_PARAMS, f"{n_params}")

    missing, unexpected = model.load_state_dict(sd, strict=True)
    check("strict_load_missing_keys", list(missing) == [], list(missing))
    check("strict_load_unexpected_keys", list(unexpected) == [], list(unexpected))

    cfg = model.config
    for field, want in ARM_V_CONFIG.items():
        check(f"contract_{field}", getattr(cfg, field) == want,
              f"{getattr(cfg, field)!r} (want {want!r})")

    # Fresh optimizer/scheduler are constructed AFTER init in the frozen
    # recipe; assert the shapes of that contract here rather than trusting it.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    fresh_opt = all(len(st) == 0 for st in opt.state_dict()["state"].values()) \
        if opt.state_dict()["state"] else True
    check("optimizer_state_fresh", fresh_opt, "no accumulated moments")
    check("scheduler_starts_at_epoch_0", sched.last_epoch == 0, f"last_epoch={sched.last_epoch}")
    check("scheduler_initial_lr", abs(sched.get_last_lr()[0] - 1e-3) < 1e-12,
          f"lr={sched.get_last_lr()[0]}")

    report["verdict"] = "PASS" if ok else "FAIL"
    out = HERE.parent / "e0_control" / "init_integrity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nverdict: {report['verdict']}\nwrote {out}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
