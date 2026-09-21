#!/usr/bin/env python
"""Stage E verdict — computed from the recorded results, never by hand.

Applies the pre-registered ladder in order, first match wins:

    1. fewer than 3 completed seeds        -> VOID
    2. INT8 gate fails                     -> REJECT
    3. delta_best <= 0                     -> REJECT
    4. 0 < delta_best < S0_best            -> INCONCLUSIVE
    5. delta_best >= S0_best AND
       max(candidate) < min(control)       -> ACCEPT (non-overlapping)
    6. delta_best >= S0_best               -> ACCEPT

E3 additionally requires delta_final >= S0_final; clearing best but not final
is recorded as BEST PASS / FINAL FAIL and is not an acceptance.

    python stage_e_recipe/verdict.py e1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Instantiated from E0 before any candidate ran. See PREREGISTRATION.md.
M0_BEST, S0_BEST = 1.2037841, 0.0174488
M0_FINAL, S0_FINAL = 1.2125409, 0.0090235
CONTROL_BEST = [1.2137377, 1.1962889, 1.2013258]
P_CONTROL, INT8_MARGIN = 5.5865268, 0.05


def load(exp: str) -> tuple[list, dict | None]:
    seeds = []
    for f in sorted((HERE / "kaggle" / f"{exp}_output").glob(f"seed*/{exp}_seed*.json")):
        d = json.loads(f.read_text())
        if d.get("status") != "COMPLETE":
            continue
        seeds.append({
            "seed": d["seed"],
            "best": d["checkpoints"]["best"]["hailo_val"]["metrics"]["epe"],
            "final": d["checkpoints"]["final"]["hailo_val"]["metrics"]["epe"],
            "contract": (d["checkpoints"]["best"]["hailo_val"]["guard"]["contract_match"]
                         and d["checkpoints"]["final"]["hailo_val"]["guard"]["contract_match"]),
        })
    int8_path = HERE / f"int8_{exp}" / f"int8_{exp}.json"
    int8 = json.loads(int8_path.read_text()) if int8_path.is_file() else None
    return seeds, int8


def decide(exp: str, seeds: list, int8: dict | None) -> dict:
    n = len(seeds)
    bests = [s["best"] for s in seeds]
    finals = [s["final"] for s in seeds]
    out: dict = {"experiment": exp, "seeds_completed": n,
                 "per_seed_best": bests, "per_seed_final": finals,
                 "contract_all": all(s["contract"] for s in seeds)}
    if n:
        out["mean_best"] = sum(bests) / n
        out["mean_final"] = sum(finals) / n
        out["spread_best"] = max(bests) - min(bests)
        out["delta_best"] = M0_BEST - out["mean_best"]
        out["delta_final"] = M0_FINAL - out["mean_final"]

    # 1. seed count
    if n < 3:
        out["verdict"] = "VOID"
        out["reason"] = f"{n} of 3 seeds completed; a candidate is never scored on fewer"
        return out

    # 2. INT8 gate
    if int8 is None:
        out["verdict"] = "VOID"
        out["reason"] = ("INT8 gate NOT MEASURED; clearing the EPE bar with the "
                         "gate unmeasured is an unfinished comparison")
        return out
    P = int8.get("P_candidate")
    out["P_candidate"] = P
    out["int8_threshold"] = P_CONTROL + INT8_MARGIN
    out["int8_gate"] = "PASS" if P <= P_CONTROL + INT8_MARGIN else "FAIL"
    if out["int8_gate"] == "FAIL":
        out["verdict"] = "REJECT"
        out["reason"] = f"INT8 gate: {P:.7f} > {P_CONTROL + INT8_MARGIN:.7f}"
        return out

    d_b, d_f = out["delta_best"], out["delta_final"]
    # 3. no improvement
    if d_b <= 0:
        out["verdict"] = "REJECT"
        out["reason"] = f"delta_best {d_b:+.7f} <= 0: no improvement over control"
        return out
    # 4. inside the control's own noise
    if d_b < S0_BEST:
        out["verdict"] = "INCONCLUSIVE"
        out["reason"] = (f"delta_best {d_b:+.7f} is positive but below S0_best "
                         f"{S0_BEST:.7f}: inside the control's own seed spread, "
                         f"so not distinguishable from noise")
        return out
    # E3 only: the final checkpoint must clear its bar too
    if exp == "e3":
        out["delta_final_required"] = S0_FINAL
        if d_f < S0_FINAL:
            out["verdict"] = "BEST PASS / FINAL FAIL"
            out["reason"] = (f"delta_best {d_b:+.7f} clears {S0_BEST:.7f} but "
                             f"delta_final {d_f:+.7f} does not clear {S0_FINAL:.7f}; "
                             f"400 epochs gets 81 monitor selections against the "
                             f"control's 41, so best alone is not sufficient")
            return out
    # 5/6
    if max(bests) < min(CONTROL_BEST):
        out["verdict"] = "ACCEPT (non-overlapping)"
        out["reason"] = (f"delta_best {d_b:+.7f} >= {S0_BEST:.7f} and worst "
                         f"candidate seed {max(bests):.7f} < best control seed "
                         f"{min(CONTROL_BEST):.7f}")
    else:
        out["verdict"] = "ACCEPT"
        out["reason"] = f"delta_best {d_b:+.7f} >= S0_best {S0_BEST:.7f}"
    return out


def main() -> None:
    exp = sys.argv[1] if len(sys.argv) > 1 else "e1"
    seeds, int8 = load(exp)
    out = decide(exp, seeds, int8)
    dst = HERE / f"{exp}_verdict.json"
    dst.write_text(json.dumps(out, indent=2))

    print(f"{exp.upper()}  seeds {out['seeds_completed']}/3  "
          f"contract_all={out['contract_all']}")
    if "mean_best" in out:
        print(f"  mean best  {out['mean_best']:.7f}  (control {M0_BEST:.7f})  "
              f"delta {out['delta_best']:+.7f}  bar {S0_BEST:.7f}")
        print(f"  mean final {out['mean_final']:.7f}  (control {M0_FINAL:.7f})  "
              f"delta {out['delta_final']:+.7f}")
        print(f"  spread     {out['spread_best']:.7f}")
    if "P_candidate" in out:
        print(f"  INT8       P {out['P_candidate']:.7f} vs threshold "
              f"{out['int8_threshold']:.7f} -> {out['int8_gate']}")
    print(f"\n  VERDICT: {out['verdict']}")
    print(f"  {out['reason']}")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
