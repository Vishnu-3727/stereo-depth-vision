"""Verify P2A pooled method-mean numbers cited in README. Read-only."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
with open(REPO / "stage_a_diagnostics" / "matching_diagnostic.json") as fh:
    p2a = json.load(fh)
for r in ("A_L1", "B_L2", "C_L1norm", "D_G8"):
    mm = p2a["method_mean"][r]
    s = [p2a["seeds"][k]["reps"][r] for k in ("0", "1", "2")]
    print(r, "pooled rank1:", [round(x["rank1_frac"], 4) for x in s], "mm:", round(mm["rank1_frac"], 4))
    print(r, "pooled mfrac:", [round(x["margin_frac_positive"], 4) for x in s], "mm:", round(mm["margin_frac_positive"], 4))
    g = [p2a["seeds"][k]["reps"][r]["gt_ge_96"] for k in ("0", "1", "2")]
    print(r, "GT96 rank1:", [round(x["rank1_frac"], 4) for x in g], "mm:", round(mm["gt_ge_96"]["rank1_frac"], 4))
    print(r, "GT96 mfrac:", [round(x["margin_frac_positive"], 4) for x in g], "mm:", round(mm["gt_ge_96"]["margin_frac_positive"], 4))
