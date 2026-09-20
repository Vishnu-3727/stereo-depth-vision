"""Print headline numbers for the README/report. Read-only."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUTDIR = REPO / "stage_b_armp" / "20260918T105050Z_d3_transfer"

with open(OUTDIR / "armp_d3_matching.json") as fh:
    armp = json.load(fh)
with open(OUTDIR / "comparison_vs_p2a.json") as fh:
    comp = json.load(fh)

A = armp["seed0_summary"]
print("device:", armp["device"], "wall_s:", round(armp["wall_clock_s"], 1))
print("feature cells total:", A["n_feature_cells_total"],
      "gt cells:", A["n_gt_cells_sampled_gt_gt0"],
      "dropped leftedge:", A["n_dropped_leftedge_gt_candidate_invalid"],
      "pairs removed:", A["n_pairs_removed_by_leftedge_rule"])
for r in ("A_L1", "B_L2", "C_L1norm", "D_G8"):
    rep = A["reps"][r]
    g = rep["gt_ge_96"]
    print(f"== {r} pooled: n={rep['n_defined_cells']} rank1={rep['rank1_frac']:.4f} "
          f"mfrac={rep['margin_frac_positive']:.4f}")
    print(f"   GT>=96: n={g['n_cells']} rank1={g['rank1_frac']:.4f} "
          f"mfrac={g['margin_frac_positive']:.4f} mmean={g['margin_mean']:.1f} "
          f"overlap={g['neg_pos_overlap']:.4f} gap={g['gap_mean']:.4f}")
print("--- deltas vs P2A method-mean ---")
for r in ("A_L1", "B_L2", "C_L1norm", "D_G8"):
    g = comp["gt_ge_96"][r]
    print(f"{r} GT>=96 d_rank1={g['delta_rank1_armp_minus_p2a']:+.4f} "
          f"d_mfrac={g['delta_marginfrac_armp_minus_p2a']:+.4f} "
          f"(armp {g['armp_rank1']:.4f}/{g['armp_margin_fracpos']:.4f} vs "
          f"p2a {g['p2a_methodmean_rank1']:.4f}/{g['p2a_methodmean_margin_fracpos']:.4f})")
print("--- A_L1 per-bin ---")
for b, row in comp["per_bin_A_L1"].items():
    print(f"{b}: n={row['armp_n_cells']} rank1={row['armp_rank1']:.4f} "
          f"(p2a mm {row['p2a_methodmean_rank1']:.4f}, d {row['delta_rank1']:+.4f}) "
          f"mfrac={row['armp_margin_fracpos']:.4f} "
          f"(p2a mm {row['p2a_methodmean_margin_fracpos']:.4f}, d {row['delta_marginfrac']:+.4f}) "
          f"overlap={row['armp_overlap']:.4f} low_n={row['low_n_armp']}")
