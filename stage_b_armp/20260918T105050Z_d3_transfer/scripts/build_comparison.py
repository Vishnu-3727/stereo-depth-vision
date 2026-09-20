"""Build comparison_vs_p2a.json from ARM-P output + Stage-A artifacts. Read-only on existing files."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUTDIR = REPO / "stage_b_armp" / "20260918T105050Z_d3_transfer"

with open(OUTDIR / "armp_d3_matching.json") as fh:
    armp = json.load(fh)
with open(REPO / "stage_a_diagnostics" / "matching_diagnostic.json") as fh:
    p2a = json.load(fh)

A = armp["seed0_summary"]
REPS = ("A_L1", "B_L2", "C_L1norm", "D_G8")
BINS_WANTED = ("[64,80)", "[80,96)", "[96,112)", "[112,128)")

# P2A per-seed GT>=96 + pooled for headline rep, read from artifact
p2a_seeds = p2a["seeds"]
p2a_mm = p2a["method_mean"]

# P2A per-bin values from d3_bins.csv (seed,rep,bin rows), cross-check vs json
csv_rows = {}
with open(REPO / "stage_a_diagnostics" / "raw" / "d3_bins.csv", newline="") as fh:
    for row in csv.DictReader(fh):
        csv_rows[(row["seed"], row["rep"], row["bin"])] = row


def num(v):
    return float(v) if v not in (None, "") else None


comp = {
    "task": "ARM-P D3 transfer vs P2A comparison (diagnostic only)",
    "asymmetry_limitation": "P2A values are a 3-seed METHOD MEAN; ARM-P is a SINGLE checkpoint "
                            "(seed0 slot). Direct delta is descriptive, not a significance test.",
    "p2a_sources": ["stage_a_diagnostics/matching_diagnostic.json",
                    "stage_a_diagnostics/raw/d3_bins.csv"],
    "armp_source": "stage_b_armp/20260918T105050Z_d3_transfer/armp_d3_matching.json",
    "armp_checkpoint": armp["checkpoint"],
    "global": {},
    "gt_ge_96": {},
    "per_bin_A_L1": {},
    "all_reps": {},
}

for r in REPS:
    a_rep = A["reps"][r]
    p_rep = p2a_mm[r]
    comp["global"][r] = {
        "armp_n_defined_cells": a_rep["n_defined_cells"],
        "armp_rank1": a_rep["rank1_frac"],
        "armp_margin_fracpos": a_rep["margin_frac_positive"],
        "p2a_methodmean_rank1": p_rep["rank1_frac"],
        "p2a_methodmean_margin_fracpos": p_rep["margin_frac_positive"],
        "delta_rank1_armp_minus_p2a": a_rep["rank1_frac"] - p_rep["rank1_frac"],
        "delta_marginfrac_armp_minus_p2a": (a_rep["margin_frac_positive"]
                                            - p_rep["margin_frac_positive"]),
    }
    a96 = a_rep["gt_ge_96"]
    p96 = p_rep["gt_ge_96"]
    comp["gt_ge_96"][r] = {
        "armp_n_cells": a96["n_cells"],
        "armp_rank1": a96["rank1_frac"],
        "armp_margin_fracpos": a96["margin_frac_positive"],
        "armp_margin_mean": a96["margin_mean"],
        "armp_overlap": a96["neg_pos_overlap"],
        "armp_gap_mean": a96["gap_mean"],
        "armp_rankle3": a96["rankle3_frac"],
        "p2a_methodmean_rank1": p96["rank1_frac"],
        "p2a_methodmean_margin_fracpos": p96["margin_frac_positive"],
        "p2a_methodmean_margin_mean": p96["margin_mean"],
        "p2a_methodmean_overlap": p96["neg_pos_overlap"],
        "p2a_methodmean_gap": p96["gap_mean"],
        "p2a_perseed_rank1": [p2a_seeds[s]["reps"][r]["gt_ge_96"]["rank1_frac"] for s in ("0", "1", "2")],
        "p2a_perseed_margin_fracpos": [p2a_seeds[s]["reps"][r]["gt_ge_96"]["margin_frac_positive"] for s in ("0", "1", "2")],
        "p2a_perseed_n_cells": [p2a_seeds[s]["reps"][r]["gt_ge_96"]["n_cells"] for s in ("0", "1", "2")],
        "delta_rank1_armp_minus_p2a": a96["rank1_frac"] - p96["rank1_frac"],
        "delta_marginfrac_armp_minus_p2a": (a96["margin_frac_positive"]
                                            - p96["margin_frac_positive"]),
    }
    comp["all_reps"][r] = {
        "per_bin_armp": [
            {"bin": b["bin"], "n_cells": b.get("n_cells"),
             "rank1": b.get("rank1_frac"), "margin_fracpos": b.get("margin_frac_positive"),
             "overlap": b.get("neg_pos_overlap"),
             "low_n": b.get("n_cells", 0) < 1000}
            for b in a_rep["per_bin"] if b["bin"] in BINS_WANTED or b.get("n_cells", 0) > 0 and b["bin"] in BINS_WANTED
        ],
    }

# A_L1 per-bin table with P2A method-mean + csv cross-check
a_bins = {b["bin"]: b for b in A["reps"]["A_L1"]["per_bin"]}
for b in BINS_WANTED:
    ab = a_bins[b]
    # method-mean across seeds from json
    mm_r1 = sum(p2a_seeds[s]["reps"]["A_L1"]["per_bin"][i]["rank1_frac"]
                for s in ("0", "1", "2")
                for i in [0]
                )  # placeholder, replaced below
    # find bin index by label
    idx = next(i for i, bb in enumerate(p2a_seeds["0"]["reps"]["A_L1"]["per_bin"]) if bb["bin"] == b)
    p_r1 = [p2a_seeds[s]["reps"]["A_L1"]["per_bin"][idx].get("rank1_frac") for s in ("0", "1", "2")]
    p_mf = [p2a_seeds[s]["reps"]["A_L1"]["per_bin"][idx].get("margin_frac_positive") for s in ("0", "1", "2")]
    p_ov = [p2a_seeds[s]["reps"]["A_L1"]["per_bin"][idx].get("neg_pos_overlap") for s in ("0", "1", "2")]
    p_n = [p2a_seeds[s]["reps"]["A_L1"]["per_bin"][idx].get("n_cells") for s in ("0", "1", "2")]
    csv_check = {s: csv_rows.get((s, "A_L1", b), {}) for s in ("0", "1", "2")}
    comp["per_bin_A_L1"][b] = {
        "armp_n_cells": ab.get("n_cells"),
        "armp_rank1": ab.get("rank1_frac"),
        "armp_margin_fracpos": ab.get("margin_frac_positive"),
        "armp_overlap": ab.get("neg_pos_overlap"),
        "low_n_armp": ab.get("n_cells", 0) < 1000,
        "p2a_perseed_rank1": p_r1,
        "p2a_perseed_margin_fracpos": p_mf,
        "p2a_perseed_overlap": p_ov,
        "p2a_perseed_n_cells": p_n,
        "p2a_methodmean_rank1": sum(p_r1) / 3,
        "p2a_methodmean_margin_fracpos": sum(p_mf) / 3,
        "delta_rank1": ab.get("rank1_frac") - sum(p_r1) / 3,
        "delta_marginfrac": ab.get("margin_frac_positive") - sum(p_mf) / 3,
        "csv_rank1_check": {s: csv_check[s].get("rank1_frac") for s in ("0", "1", "2")},
    }

comp["global_counts"] = {
    "armp_n_feature_cells_total": A["n_feature_cells_total"],
    "armp_n_gt_cells": A["n_gt_cells_sampled_gt_gt0"],
    "armp_n_dropped_leftedge": A["n_dropped_leftedge_gt_candidate_invalid"],
    "armp_n_pairs_removed_leftedge": A["n_pairs_removed_by_leftedge_rule"],
    "p2a_n_gt_cells_per_seed": p2a["counts"]["n_gt_cells_per_seed"],
    "p2a_n_pairs_removed_per_seed": p2a["counts"]["n_pairs_removed_by_leftedge_rule_per_seed"],
}
comp["representation_D_note"] = ("D was group-AVERAGED and is therefore G-invariant; "
                                 "known Stage-A limitation, PRESERVED, not fixed.")
comp["observed_vs_mechanism"] = {
    "observed_effect": "to be written in README/report from the numbers above",
    "mechanism_evidence": "to be written in README/report; D3 alone neither validates nor falsifies ARM-P",
}

fd, tmp = tempfile.mkstemp(dir=str(OUTDIR), suffix=".tmp")
try:
    with os.fdopen(fd, "w") as fh:
        json.dump(comp, fh, indent=2)
    os.replace(tmp, OUTDIR / "comparison_vs_p2a.json")
except BaseException:
    try:
        os.unlink(tmp)
    except OSError:
        pass
    raise
print("wrote comparison_vs_p2a.json", flush=True)
