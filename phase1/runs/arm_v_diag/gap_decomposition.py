"""Decompose the ARM-V -> frozen-reference EPE gap by GT-disparity stratum.

gap_contribution(stratum) = frac_px(stratum) * (EPE_armv(stratum) - EPE_ref(stratum))
The contributions sum to the total EPE gap by construction (pooled-pixel EPE is
a pixel-count-weighted mean). Read-only.
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
ref = json.loads((D / "reference_strata.json").read_text())
seeds = {s: json.loads((D / f"diag_seed{s}.json").read_text()) for s in (0, 1, 2)}
rb = {(b["lo"], b["hi"]): b for b in ref["gt_bins"]}

print("frozen reference EPE = %.7f   D1 = %.4f%%" % (ref["epe"], ref["d1"]))
print("reference pred percentiles:", {k: round(v, 2) for k, v in ref["pred_pct"].items()})
print()
hdr = "gt bin".ljust(13) + "frac_px".rjust(9) + "  " + "  ".join(
    x.rjust(10) for x in ("ref EPE", "s0 EPE", "s1 EPE", "s2 EPE",
                          "s0 gap", "s1 gap", "s2 gap"))
print(hdr)
print("-" * len(hdr))
tot = {0: 0.0, 1: 0.0, 2: 0.0}
big = {0: 0.0, 1: 0.0, 2: 0.0}
for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 96), (96, 128), (128, 160)]:
    r = rb[(lo, hi)]
    row = [r["frac_px"], r["epe"]]
    gaps = []
    for s in (0, 1, 2):
        b = [x for x in seeds[s]["gt_bins"] if x["lo"] == lo][0]
        row.append(b["epe"])
        g = r["frac_px"] * (b["epe"] - r["epe"])
        gaps.append(g)
        tot[s] += g
        if lo >= 64:
            big[s] += g
    print(("[%d,%d)" % (lo, hi)).ljust(13)
          + ("%.5f" % row[0]).rjust(9) + "  "
          + "  ".join(("%.4f" % v).rjust(10) for v in row[1:])
          + "  " + "  ".join(("%+.4f" % g).rjust(10) for g in gaps))
print()
for s in (0, 1, 2):
    print("seed %d: total gap to reference = %+.4f px | gt>=64 contributes %+.4f px (%.1f%%)"
          % (s, tot[s], big[s], 100 * big[s] / tot[s]))
mean_gap = sum(tot.values()) / 3
mean_big = sum(big.values()) / 3
print("3-seed mean: total gap %+.4f px | gt>=64 share %.1f%%"
      % (mean_gap, 100 * mean_big / mean_gap))
print()
print("counterfactual: ARM-V EPE if gt>=64 pixels matched the REFERENCE's error there")
for s in (0, 1, 2):
    print("  seed %d: %.4f -> %.4f px" % (s, seeds[s]["epe"], seeds[s]["epe"] - big[s]))
print("  3-seed mean: %.4f -> %.4f px"
      % (sum(seeds[s]["epe"] for s in (0, 1, 2)) / 3,
         sum(seeds[s]["epe"] - big[s] for s in (0, 1, 2)) / 3))
print()
print("occlusion stratum")
print("  reference: frac %.5f EPE %.4f mass %.4f" % (
    ref["occluded"]["frac_px"], ref["occluded"]["epe"], ref["occluded"]["epe_mass_frac"]))
for s in (0, 1, 2):
    o = seeds[s]["occluded"]
    print("  seed %d  : frac %.5f EPE %.4f mass %.4f | gap contribution %+.4f px"
          % (s, o["frac_px"], o["epe"], o["epe_mass_frac"],
             ref["occluded"]["frac_px"] * (o["epe"] - ref["occluded"]["epe"])))
