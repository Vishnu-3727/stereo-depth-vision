"""Print per-bin rows for B/C/D (wanted bins). Read-only."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUTDIR = REPO / "stage_b_armp" / "20260918T105050Z_d3_transfer"

with open(OUTDIR / "armp_d3_matching.json") as fh:
    armp = json.load(fh)
with open(REPO / "stage_a_diagnostics" / "matching_diagnostic.json") as fh:
    p2a = json.load(fh)

A = armp["seed0_summary"]
for r in ("B_L2", "C_L1norm", "D_G8"):
    print(f"== {r} ==")
    for b in A["reps"][r]["per_bin"]:
        if b["bin"] not in ("[64,80)", "[80,96)", "[96,112)", "[112,128)"):
            continue
        idx = next(i for i, bb in enumerate(p2a["seeds"]["0"]["reps"][r]["per_bin"]) if bb["bin"] == b["bin"])
        pr = sum(p2a["seeds"][s]["reps"][r]["per_bin"][idx].get("rank1_frac") for s in ("0", "1", "2")) / 3
        pm = sum(p2a["seeds"][s]["reps"][r]["per_bin"][idx].get("margin_frac_positive") for s in ("0", "1", "2")) / 3
        po = sum(p2a["seeds"][s]["reps"][r]["per_bin"][idx].get("neg_pos_overlap") for s in ("0", "1", "2")) / 3
        print(f"  {b['bin']}: n={b['n_cells']} rank1={b['rank1_frac']:.4f} (p2a {pr:.4f}) "
              f"mfrac={b['margin_frac_positive']:.4f} (p2a {pm:.4f}) "
              f"overlap={b['neg_pos_overlap']:.4f} (p2a {po:.4f})")
    # pooled + gt>=96 pos/neg means for scale note
    rep = A["reps"][r]
    print(f"   pooled pos_mean={rep['pos_mean_score']:.3g} neg_mean={rep['neg_mean_score']:.3g}")
    g = rep["gt_ge_96"]
    print(f"   GT>=96 pos_mean={g['pos_mean_score']:.3g} neg_mean={g['neg_mean_score']:.3g}")
print("== A_L1 scale ==")
rep = A["reps"]["A_L1"]
print(f"   pooled pos_mean={rep['pos_mean_score']:.3g} neg_mean={rep['neg_mean_score']:.3g}")
g = rep["gt_ge_96"]
print(f"   GT>=96 pos_mean={g['pos_mean_score']:.3g} neg_mean={g['neg_mean_score']:.3g}")
