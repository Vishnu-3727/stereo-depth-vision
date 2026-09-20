"""EXP-CORRESPONDENCE-INDEX-002 -- null-quality summary (PREREGISTRATION.md section 12 / task section K).

Reads the frozen permutation file, the ground-truth masks and results.json, and
emits `null_summary.json`. Removes nothing, screens nothing, re-scores nothing.

Only the items enumerated in section K are produced. No association between
permutation displacement and observed response is computed: no such statistic was
preregistered, and inventing one after inspecting the results is forbidden by
the protocol (task rule 11, PREREGISTRATION.md section 15).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPO_ROOT = OUT.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2.viz import core  # noqa: E402

FOCUS_SCENES = [27, 0, 31, 6]
SPLIT = "hailo_val"
BAND = list(range(2, 9))


def main() -> int:
    frozen = json.loads((OUT / "permutations.json").read_text(encoding="utf-8"))
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    meta = frozen["random_permutation_metadata"]

    # ---- retained candidate occupancy (GT only; model-independent) ----------
    per_scene, pooled = {}, []
    for si in FOCUS_SCENES:
        sc = core.load_scene(si, split=SPLIT)
        gt = sc.gt_disparity.astype(np.float64)
        m = (gt > 0) & (gt / 16.0 >= 2.0) & (gt / 16.0 <= 8.0)
        d = gt[m] / 16.0
        pooled.append(d)
        per_scene[sc.name] = {"retained_pixels": int(m.sum()),
                              "mean_candidate_units": float(d.mean()),
                              "median_candidate_units": float(np.median(d))}
    dd = np.concatenate(pooled)
    hist, _ = np.histogram(dd, bins=np.arange(1.5, 9.5, 1.0))
    w = hist / hist.sum()
    occupancy = {
        "note": "GT expressed in candidate units (GT/16). Descriptive only; not an "
                "input to any decision rule. The mask was NOT changed for this experiment.",
        "per_scene": per_scene,
        "pooled_retained_pixels": int(dd.size),
        "pooled_mean_candidate_units": float(dd.mean()),
        "pooled_median_candidate_units": float(np.median(dd)),
        "occupancy_by_candidate_bin_d2_to_d8": [float(x) for x in w],
        "fraction_in_d2_to_d4": float(w[:3].sum()),
    }

    # ---- displacement distribution across the frozen null ------------------
    disp_all = np.array([mm["mean_displacement_all12"] for mm in meta])
    disp_band = np.array([mm["mean_displacement_band_2_8_uniform"] for mm in meta])
    disp_gtw = np.array([float(np.dot(w, [mm["inverse_permutation"][d] - d for d in BAND]))
                         for mm in meta])
    fixed = np.array([mm["n_fixed_points"] for mm in meta])
    per_d = np.array([[mm["inverse_permutation"][d] - d for d in BAND] for mm in meta])

    def desc(a: np.ndarray) -> dict:
        return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
                "std": float(a.std(ddof=1)), "min": float(a.min()), "max": float(a.max()),
                "p01": float(np.percentile(a, 1.0, method="linear")),
                "p99": float(np.percentile(a, 99.0, method="linear"))}

    displacement = {
        "mean_displacement_over_all_12_candidates": desc(disp_all),
        "mean_displacement_over_retained_band_uniform": desc(disp_band),
        "mean_displacement_over_retained_band_gt_weighted": desc(disp_gtw),
        "n_fixed_points": {"mean": float(fixed.mean()),
                           "histogram": {str(k): int((fixed == k).sum())
                                         for k in range(0, int(fixed.max()) + 1)}},
        "per_candidate_displacement_pooled": desc(per_d.reshape(-1)),
        "reference_frame_caveat":
            "Mapping GT/16 onto candidate index d assumes candidate magnitude equals "
            "disparity in feature-grid units. The volume indexes that magnitude at the "
            "RIGHT image column while GT indexes the LEFT column. These displacement "
            "figures are therefore descriptive design metadata, not results, and were "
            "not used by the primary statistic, which takes only between-arm differences "
            "on an identical pixel set.",
    }

    # ---- null response distribution per positive unit ----------------------
    pos = [u for u in res["units"] if not u["is_negative_control"]]
    units = []
    for u in pos:
        n, t = u["null"], u["test"]
        units.append({
            "checkpoint": u["checkpoint"], "seed": u["seed"], "scene": u["scene"],
            "retained_pixels": u["valid_pixels"],
            "ordered": {"d_minus1": u["d_minus1"], "d_identity": u["d_identity"],
                        "d_plus1": u["d_plus1"], "d_plus2": u["d_plus2"],
                        "S_order": u["S_order"], "S_abs": u["S_abs"]},
            "null_response_distribution": {
                "n": n["n"], "mean": n["mean"], "median": n["median"], "std": n["std"],
                "min": n["min"], "max": n["max"], "p01": n["p01"], "p99": n["p99"],
                "degenerate": n["degenerate"]},
            "per_permutation_d_spread": {
                "min": u["d_random_min"], "median": u["d_random_median"],
                "mean": u["d_random_mean"], "max": u["d_random_max"]},
            "empirical_rank": t["empirical_rank"], "rank_of": t["rank_of"],
            "n_null_ge_ordered": t["n_null_ge_ordered"],
            "n_null_le_ordered": t["n_null_le_ordered"],
            "p_two_sided": t["p_two_sided"],
            "outside_central_99": t["outside_central_99"],
            "null_pass": t["null_pass"], "order_pass": u["order_pass"],
        })

    summary = {
        "experiment": "EXP-CORRESPONDENCE-INDEX-002",
        "verdict": res["VERDICT"],
        "frozen_seed": frozen["frozen_seed"],
        "permutations_sha256": res["permutations_sha256"],
        "m_triples": frozen["m_triples"],
        "n_random_permutations": frozen["n_random_permutations"],
        "duplicates_in_null": frozen["n_duplicates"],
        "null_draws_removed": 0,
        "screening_applied": False,
        "retained_candidate_occupancy": occupancy,
        "displacement_distribution": displacement,
        "per_unit": units,
        "aggregate": {
            "positive_units": len(pos),
            "S_order_min": min(u["S_order"] for u in pos),
            "S_order_median": float(np.median([u["S_order"] for u in pos])),
            "S_order_max": max(u["S_order"] for u in pos),
            "null_mean_min": min(u["null"]["mean"] for u in pos),
            "null_mean_max": max(u["null"]["mean"] for u in pos),
            "null_sd_min": min(u["null"]["std"] for u in pos),
            "null_sd_max": max(u["null"]["std"] for u in pos),
            "null_p99_min": min(u["null"]["p99"] for u in pos),
            "null_p99_max": max(u["null"]["p99"] for u in pos),
            "p_two_min": min(u["test"]["p_two_sided"] for u in pos),
            "p_two_median": float(np.median([u["test"]["p_two_sided"] for u in pos])),
            "p_two_max": max(u["test"]["p_two_sided"] for u in pos),
            "alpha": res["alpha"],
            "min_attainable_two_sided_p": res["summary"]["min_attainable_two_sided_p"],
            "null_pass_units": res["summary"]["null_pass_units"],
            "order_pass_units": res["summary"]["order_pass_units"],
        },
        "inverse_permutation_mapping_location":
            "permutations.json -> random_permutation_metadata[*].inverse_permutation "
            "(all 1536 recorded, none removed)",
        "association_analysis_performed": False,
        "association_analysis_note":
            "No association between permutation displacement and observed response was "
            "computed. No such statistic was preregistered; inventing one after "
            "inspecting results is forbidden (task rule 11).",
    }

    (OUT / "null_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote null_summary.json")
    print("occupancy d2..d8:", ["%.4f" % x for x in w])
    print("fraction d2..d4 : %.4f" % w[:3].sum())
    print("band displacement (uniform)     :", json.dumps(desc(disp_band)))
    print("band displacement (GT-weighted) :", json.dumps(desc(disp_gtw)))
    print("fixed-point histogram           :", displacement["n_fixed_points"]["histogram"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
