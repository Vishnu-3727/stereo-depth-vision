"""Render the ARM-V high-disparity diagnostic JSON as markdown tables. Read-only."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
S = R["seeds"]
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "hailo_val"


def g(seed, *path, default=None):
    o = S[str(seed)][SPLIT]
    for p in path:
        if o is None:
            return default
        o = o.get(p) if isinstance(o, dict) else None
    return default if o is None else o


print("### split:", SPLIT)
print()
print("| seed | scenes | valid px | contract_match | max pred px | init idx max | init*8 px |")
print("|---|---|---|---|---|---|---|")
for s in (0, 1, 2):
    d = S[str(s)][SPLIT]
    print("| %d | %d | %d | %s | %.2f | %.3f | %.1f |" % (
        s, d["scenes"], d["valid_pixels"], d["guard"].get("contract_match"),
        d["pred_max_all_px"], d["init_max_candidates"], d["init_max_x8_px"]))

print()
print("#### A. per-GT-bin statistics")
for s in (0, 1, 2):
    print()
    print("seed", s)
    print("| GT bin | px | frac | EPE | signed | RMSE | D1 % | mean GT | mean pred | pred/GT | mean init*8 | init*8/GT |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in S[str(s)][SPLIT]["bins"]:
        if b["status"] != "ok":
            print("| [%d,%d) | %d | %.5f | — | — | — | — | — | — | — | — | — |  <!-- %s -->"
                  % (b["lo"], b["hi"], b["px"], b["frac_px"], b["status"]))
            continue
        print("| [%d,%d) | %d | %.5f | %.3f | %+.3f | %.3f | %.2f | %.2f | %.2f | %.3f | %.2f | %.3f |" % (
            b["lo"], b["hi"], b["px"], b["frac_px"], b["epe"], b["signed_err"],
            b["rmse"], b["d1_pct"], b["mean_gt"], b["mean_pred"], b["pred_over_gt"],
            b["mean_init_x8_px"], b["init_x8_over_gt"]))

print()
print("#### D/E. readout + cost response per GT bin")
for s in (0, 1, 2):
    print()
    print("seed", s)
    print("| GT bin | entropy | p_top1 | p@true | rank@true | z@true agg | z@true raw | argmax within1 |")
    print("|---|---|---|---|---|---|---|---|")
    for b in S[str(s)][SPLIT]["bins"]:
        if b["status"] != "ok":
            continue
        print("| [%d,%d) | %.3f | %.4f | %.4f | %.2f | %+.4f | %+.4f | %.4f |" % (
            b["lo"], b["hi"], b["readout_entropy"], b["p_top1"],
            b["p_at_true_candidate"], b["mean_rank_of_true_candidate"],
            b["z_at_true_candidate_aggregated"], b["z_at_true_candidate_raw"],
            b["frac_argmax_within_1_of_true"]))

print()
print("#### B. GT >= 64 stratum")
print("| seed | px | frac | mean GT | mean pred | med GT | med pred | signed | EPE | pred/GT | slope | intercept | R2 | resid RMSE |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for s in (0, 1, 2):
    h = S[str(s)][SPLIT]["high_stratum"]
    f = h["fit_pred_on_gt"]
    print("| %d | %d | %.5f | %.2f | %.2f | %.2f | %.2f | %+.2f | %.3f | %.3f | %.4f | %+.2f | %.4f | %.2f |" % (
        s, h["px"], h["frac_px"], h["mean_gt"], h["mean_pred"], h["median_gt"],
        h["median_pred"], h["mean_signed_err"], h["epe"], h["mean_pred_over_gt"],
        f["slope"], f["intercept"], f["r2"], f["residual_rmse_px"]))

print()
print("#### B2. GT >= 64 stratum: initial readout x8 vs GT (descriptive)")
print("| seed | slope | intercept | R2 | pearson r | resid RMSE |")
print("|---|---|---|---|---|---|")
for s in (0, 1, 2):
    f = S[str(s)][SPLIT]["high_stratum"]["fit_initx8_on_gt"]
    print("| %d | %.4f | %+.2f | %.4f | %.4f | %.2f |" % (
        s, f["slope"], f["intercept"], f["r2"], f["pearson_r"], f["residual_rmse_px"]))

if SPLIT == "hailo_val":
    print()
    print("#### C. low-GT-calibrated initial readout, extrapolated to GT >= 64")
    print("| seed | calibration (GT = a*idx + b, fit on GT<64) | mean GT | extrap mean pred | extrap signed | extrap EPE | extrap max | final EPE | final max pred |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in (0, 1, 2):
        e = S[str(s)][SPLIT]["extrapolated_initial_readout"]
        print("| %d | a=%.4f b=%+.4f | %.2f | %.2f | %+.2f | %.3f | %.1f | %.3f | %.1f |" % (
            s, S[str(s)][SPLIT]["low_stratum_fits"]["fit_gt_on_init_index"]["slope"],
            S[str(s)][SPLIT]["low_stratum_fits"]["fit_gt_on_init_index"]["intercept"],
            e["mean_gt_on_high_stratum"], e["mean_extrapolated_pred_on_high_stratum"],
            e["mean_signed_err"], e["epe"], e["max_extrapolated"],
            e["final_model_epe_same_pixels"], e["final_model_max_pred"]))

    print()
    print("#### low-stratum descriptive fits (GT < 64)")
    print("| seed | pred~GT slope | pred~GT R2 | init*8~GT slope | init*8~GT R2 | GT~idx slope |")
    print("|---|---|---|---|---|---|")
    for s in (0, 1, 2):
        L = S[str(s)][SPLIT]["low_stratum_fits"]
        print("| %d | %.4f | %.4f | %.4f | %.4f | %.4f |" % (
            s, L["fit_pred_on_gt"]["slope"], L["fit_pred_on_gt"]["r2"],
            L["fit_initx8_on_gt"]["slope"], L["fit_initx8_on_gt"]["r2"],
            L["fit_gt_on_init_index"]["slope"]))

print()
print("#### D. readout, GT<64 vs GT>=64")
keys = ["px", "entropy", "p_top1", "p_top3_mass", "p_at_true_candidate",
        "p_at_true_over_uniform", "median_rank_of_true_candidate",
        "frac_true_candidate_in_top1", "frac_true_candidate_in_top3",
        "z_at_true_candidate_aggregated", "z_at_true_candidate_raw",
        "frac_z_true_below_zero_aggregated", "frac_z_true_below_zero_raw",
        "readout_distribution_std_candidates", "mean_argmax_index",
        "mean_true_index", "mean_soft_index"]
print("| metric | " + " | ".join("s%d %s" % (s, t) for s in (0, 1, 2)
                                 for t in ("lo", "hi")) + " |")
print("|" + "---|" * 7)
for k in keys:
    cells = []
    for s in (0, 1, 2):
        for tag in ("gt_lt_64", "gt_ge_64"):
            v = S[str(s)][SPLIT]["readout_low_vs_high"][tag].get(k)
            cells.append("—" if v is None else ("%d" % v if k == "px" else "%.4f" % v))
    print("| " + k + " | " + " | ".join(cells) + " |")
