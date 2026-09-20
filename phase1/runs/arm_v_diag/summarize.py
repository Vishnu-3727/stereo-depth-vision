"""Cross-seed summary of the ARM-V bottleneck diagnostic. Read-only."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
runs = {s: json.loads((D / f"diag_seed{s}.json").read_text()) for s in (0, 1, 2)}


def row(label, fn):
    vals = [fn(runs[s]) for s in (0, 1, 2)]
    fmt = []
    for v in vals:
        fmt.append("None" if v is None else format(v, ".4f"))
    print(label.ljust(46) + " | " + " | ".join(x.rjust(9) for x in fmt))


print("metric".ljust(46) + " | " + " | ".join(("seed" + str(s)).rjust(9) for s in (0, 1, 2)))
print("-" * 82)
row("EPE", lambda d: d["epe"])
row("D1 %", lambda d: d["d1"])
print("-- error mass --")
for i, b in enumerate(runs[0]["error_bins"]):
    row(f"frac_px err[{b['lo']},{b['hi']})", lambda d, i=i: d["error_bins"][i]["frac_px"])
    row(f"  EPE-mass share", lambda d, i=i: d["error_bins"][i]["epe_mass_frac"])
for q in ("50", "75", "90", "95", "99", "99.9"):
    row("err p" + q, lambda d, q=q: d["error_percentiles"][q])
print("-- GT magnitude --")
for i, b in enumerate(runs[0]["gt_bins"][:7]):
    row(f"EPE | gt in [{b['lo']},{b['hi']})", lambda d, i=i: d["gt_bins"][i]["epe"])
    row("  EPE-mass share", lambda d, i=i: d["gt_bins"][i]["epe_mass_frac"])
row("frac_px gt>64", lambda d: sum(b["frac_px"] for b in d["gt_bins"] if b["lo"] >= 64))
row("EPE-mass share gt>64", lambda d: sum(b["epe_mass_frac"] for b in d["gt_bins"] if b["lo"] >= 64))
row("frac_px gt>184 (out of range)", lambda d: d["gt_stats"]["frac_over_184"])
row("max gt px", lambda d: d["gt_stats"]["max"])
print("-- strata --")
for k in ("occluded", "discontinuity_r2_3px", "texture_low_q25", "texture_high_q75"):
    row(k + " frac_px", lambda d, k=k: d[k]["frac_px"])
    row(k + " EPE", lambda d, k=k: d[k]["epe"])
    row(k + " EPE-mass share", lambda d, k=k: d[k]["epe_mass_frac"])
print("-- stage --")
for k in ("corr_init_gt", "best_affine_a", "best_affine_b",
          "epe_of_best_affine_init", "epe_of_8x_init", "epe_final",
          "resid_share_of_magnitude", "corr_resid_pred", "oracle_gt_init_epe",
          "init_mean", "init_std", "init_min", "init_max"):
    row("stage." + k, lambda d, k=k: d["stage"][k])
print("-- quantisation --")
row("frac near integer (<0.05)", lambda d: d["quantisation"]["frac_near_integer_within_0p05"])
for i in range(5):
    row("EPE | dist-to-candidate bin " + str(i),
        lambda d, i=i: d["quantisation"]["epe_vs_dist_to_candidate"][i]["epe"])
row("frac-part hist min/expect ratio",
    lambda d: min(d["quantisation"]["frac_part_hist"]) / d["quantisation"]["frac_part_uniform_expect"])
row("frac-part hist max/expect ratio",
    lambda d: max(d["quantisation"]["frac_part_hist"]) / d["quantisation"]["frac_part_uniform_expect"])
print("-- cost volume --")
for k in ("entropy_mean", "entropy_max_possible", "entropy_p10", "entropy_p90",
          "local_minima_mean", "unimodal_frac", "runner_up_margin_mean_sd_units",
          "argmin_index_mean", "epe_of_hard_argmin_x8", "corr_argmin_gt",
          "argmin_within_1_candidate_of_gt"):
    row("cv." + k, lambda d, k=k: d["cost_volume"][k])
row("cv.argmin frac in {1,2,3}",
    lambda d: sum(d["cost_volume"]["argmin_index_hist"][1:4]) / d["valid_pixels"])
row("cv.argmin frac at index 23",
    lambda d: d["cost_volume"]["argmin_index_hist"][23] / d["valid_pixels"])
