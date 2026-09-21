#!/usr/bin/env python
"""E1 correctness gate: the EMA update does what it claims.

Exercises the exact statements the E1 patch inserts, on the real architecture,
against an independently computed reference. CPU only, no data, no optimizer,
no checkpoint, no training - this is arithmetic, not a run.

Checks:
  1. the update matches ema <- decay*ema + (1-decay)*param exactly
  2. it covers all 70 state entries, leaving nothing stale
  3. the shadow copy starts equal to the initialised weights
  4. training weights are NOT modified by the EMA update
  5. the EMA model keeps 397,954 parameters (no architectural addition)
  6. after N steps on a constant target the EMA converges as the closed form
     predicts, so the decay is applied per step and not per epoch

    python stage_e_recipe/tests/test_ema.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

CFG = dict(downsample_levels=3, num_disparities=24,
           cost_volume_shift="right", regression_normalize=True)
EMA_DECAY = 0.999
EXPECTED_PARAMS = 397954
EXPECTED_KEYS = 70


def ema_step(model, ema_model, decay=EMA_DECAY):
    """Byte-for-byte the statements the E1 patch inserts after optimizer.step()."""
    with torch.no_grad():
        _msd = model.state_dict()
        for _k, _v in ema_model.state_dict().items():
            _v.mul_(decay).add_(_msd[_k], alpha=1.0 - decay)


def main() -> None:
    torch.manual_seed(0)
    model = StereoNet(StereoNetConfig(**CFG))
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    failures = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # 3. shadow starts equal
    check("shadow_starts_equal",
          all(torch.equal(a, b) for a, b in
              zip(model.state_dict().values(), ema_model.state_dict().values())))

    # 5. no architectural addition
    check("ema_param_count", sum(p.numel() for p in ema_model.parameters()) == EXPECTED_PARAMS)
    check("state_entries", len(ema_model.state_dict()) == EXPECTED_KEYS,
          f"{len(ema_model.state_dict())}")

    # 1 + 2 + 4: one update against an independent reference
    before_ema = {k: v.clone() for k, v in ema_model.state_dict().items()}
    with torch.no_grad():          # stand in for an optimizer step
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    before_model = {k: v.clone() for k, v in model.state_dict().items()}

    ema_step(model, ema_model)

    worst, covered = 0.0, 0
    for k, v in ema_model.state_dict().items():
        expected = before_ema[k] * EMA_DECAY + before_model[k] * (1.0 - EMA_DECAY)
        worst = max(worst, float((v - expected).abs().max()))
        if not torch.equal(before_ema[k], v):
            covered += 1
    # Tolerance, not equality: the patch computes `v.mul_(d).add_(x, alpha=1-d)`
    # in place, while the reference above computes `a*d + b*(1-d)` through
    # separate temporaries. Both are correct; float32 rounds them differently.
    # Weights here are O(1), and float32 eps is ~1.2e-7, so 1e-6 absolute is a
    # bound that a genuine logic error could not slip under.
    check("update_matches_formula", worst < 1e-6, f"max abs diff {worst:.3e}")
    check("all_entries_updated", covered == EXPECTED_KEYS, f"{covered}/{EXPECTED_KEYS}")
    check("training_weights_untouched",
          all(torch.equal(before_model[k], v) for k, v in model.state_dict().items()))

    # 6. per-step decay: hold the target constant, check the closed form
    #    ema_n = target + (ema_0 - target) * decay**n
    torch.manual_seed(1)
    m2 = StereoNet(StereoNetConfig(**CFG))
    e2 = copy.deepcopy(m2)
    with torch.no_grad():
        for p in m2.parameters():
            p.fill_(1.0)
    start = {k: v.clone() for k, v in e2.state_dict().items()}
    n_steps = 50
    for _ in range(n_steps):
        ema_step(m2, e2)
    worst_cf = 0.0
    for k, v in e2.state_dict().items():
        target = m2.state_dict()[k]
        expected = target + (start[k] - target) * (EMA_DECAY ** n_steps)
        worst_cf = max(worst_cf, float((v - expected).abs().max()))
    check("per_step_decay_closed_form", worst_cf < 1e-6, f"max abs diff {worst_cf:.3e}")

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
