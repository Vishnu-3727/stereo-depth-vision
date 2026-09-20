"""Verify P2A per-bin overlap/mfrac/rank1 mm values cited in README. Read-only."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
with open(REPO / "stage_a_diagnostics" / "matching_diagnostic.json") as fh:
    p2a = json.load(fh)
for r in ("A_L1", "B_L2", "C_L1norm", "D_G8"):
    print(f"== {r} ==")
    for b in ("[64,80)", "[80,96)", "[96,112)", "[112,128)"):
        idx = next(i for i, bb in enumerate(p2a["seeds"]["0"]["reps"][r]["per_bin"]) if bb["bin"] == b)
        for key in ("rank1_frac", "margin_frac_positive", "neg_pos_overlap"):
            v = [p2a["seeds"][s]["reps"][r]["per_bin"][idx].get(key) for s in ("0", "1", "2")]
            print(f"  {b} {key}: {[round(x, 4) for x in v]} mm={sum(v)/3:.4f}")
