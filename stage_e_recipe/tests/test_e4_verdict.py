#!/usr/bin/env python
"""E4 acceptance gate: verdict.py scores exp e4 per the pre-registered table.

Spec: docs/superpowers/specs/2026-09-22-stage-e-e4-pretrain-design.md §5.
E0 remains the control; its instantiated constants apply unchanged:

    M0_best = 1.2037841, S0_best = 0.0174488, min(control best) = 1.1962889,
    INT8 gate: P_candidate <= 5.6365268.

Every row of the §5 table is exercised with synthetic seed numbers through
verdict.decide (no checkpoints, no data, no Kaggle):

    VOID                   | fewer than 3 completed seeds, or INT8 not measured
    REJECT                 | delta_best <= 0, or the INT8 gate fails
    INCONCLUSIVE           | 0 < delta_best < S0_best
    ACCEPT                 | delta_best >= S0_best
    ACCEPT (non-overlapping) | additionally max(E4 best) < min(control best)

Plus the E4-specific rule: the E3-only delta_final requirement does NOT
apply to E4 — delta_final is recorded but never gates the verdict.

    pytest stage_e_recipe/tests/test_e4_verdict.py
    python stage_e_recipe/tests/test_e4_verdict.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "stage_e_recipe"))

import verdict  # noqa: E402

# Spec §5 instantiated constants (E0 control).
M0_BEST, S0_BEST = 1.2037841, 0.0174488
MIN_CONTROL_BEST = 1.1962889
INT8_THRESHOLD = 5.6365268

PASS_INT8 = {"P_candidate": 5.0}   # <= 5.6365268
FAIL_INT8 = {"P_candidate": 6.0}   # >  5.6365268


def seeds_of(bests: list[float], finals: list[float] | None = None) -> list[dict]:
    finals = finals if finals is not None else list(bests)
    return [{"seed": i, "best": b, "final": f, "contract": True}
            for i, (b, f) in enumerate(zip(bests, finals))]


def test_constants_match_spec():
    assert verdict.M0_BEST == M0_BEST
    assert verdict.S0_BEST == S0_BEST
    assert min(verdict.CONTROL_BEST) == MIN_CONTROL_BEST
    assert verdict.P_CONTROL + verdict.INT8_MARGIN == INT8_THRESHOLD or \
        abs((verdict.P_CONTROL + verdict.INT8_MARGIN) - INT8_THRESHOLD) < 1e-9
    assert "e4" in verdict.SUPPORTED_EXPS


def test_void_fewer_than_3_seeds():
    out = verdict.decide("e4", seeds_of([1.18, 1.183]), PASS_INT8)
    assert out["verdict"] == "VOID", out
    assert out["seeds_completed"] == 2


def test_void_int8_not_measured():
    out = verdict.decide("e4", seeds_of([1.18, 1.183, 1.185]), None)
    assert out["verdict"] == "VOID", out
    assert "NOT MEASURED" in out["reason"]


def test_reject_no_improvement():
    # mean 1.21 > M0_best: delta_best = -0.0062 <= 0.
    out = verdict.decide("e4", seeds_of([1.208, 1.21, 1.212]), PASS_INT8)
    assert out["verdict"] == "REJECT", out
    assert out["delta_best"] <= 0


def test_reject_int8_gate_fails():
    out = verdict.decide("e4", seeds_of([1.18, 1.183, 1.185]), FAIL_INT8)
    assert out["verdict"] == "REJECT", out
    assert out["int8_gate"] == "FAIL"


def test_inconclusive_inside_control_noise():
    # mean 1.191: delta = +0.0128, inside (0, S0_best).
    out = verdict.decide("e4", seeds_of([1.190, 1.191, 1.192]), PASS_INT8)
    assert out["verdict"] == "INCONCLUSIVE", out
    assert 0 < out["delta_best"] < S0_BEST


def test_accept_overlapping():
    # mean 1.182333: delta = +0.02145 >= S0_best, but max 1.197 overlaps.
    out = verdict.decide("e4", seeds_of([1.170, 1.180, 1.197]), PASS_INT8)
    assert out["verdict"] == "ACCEPT", out
    assert out["delta_best"] >= S0_BEST
    assert max(out["per_seed_best"]) >= MIN_CONTROL_BEST


def test_accept_non_overlapping():
    # mean 1.1826667: delta >= S0_best and worst seed still beats control.
    out = verdict.decide("e4", seeds_of([1.180, 1.183, 1.185]), PASS_INT8)
    assert out["verdict"] == "ACCEPT (non-overlapping)", out
    assert max(out["per_seed_best"]) < MIN_CONTROL_BEST


def test_e4_records_but_does_not_require_delta_final():
    # Clears the best bar while final REGRESSES past the E3 bar (and past
    # zero): E4 must still ACCEPT, with delta_final recorded.
    bests = [1.180, 1.183, 1.185]
    finals = [1.230, 1.231, 1.229]  # mean 1.23: delta_final strongly negative
    out = verdict.decide("e4", seeds_of(bests, finals), PASS_INT8)
    assert out["verdict"] == "ACCEPT (non-overlapping)", out
    assert "delta_final" in out, out
    assert out["delta_final"] < verdict.S0_FINAL
    assert out["delta_final_required"] is None
    assert out["verdict"] != "BEST PASS / FINAL FAIL"


def test_int8_control_declares_e4_support():
    text = (REPO / "stage_e_recipe" / "int8_control.py").read_text(encoding="utf-8")
    assert '"e4"' in text or "'e4'" in text, "int8_control must name exp e4"
    assert "5.5865268" in text, "int8_control P_control reference must match verdict"
    assert float(text.split("P_CONTROL = ")[1].split()[0]) == verdict.P_CONTROL


def main() -> None:
    names = [n for n in sorted(dir(sys.modules[__name__]))
             if n.startswith("test_")]
    failures = []
    for name in names:
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {name}: {e!r}")
            failures.append(name)
        else:
            print(f"PASS  {name}")
    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
