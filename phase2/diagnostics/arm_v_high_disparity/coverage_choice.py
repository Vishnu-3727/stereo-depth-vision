"""Pick the scale-augmentation range by MEASURED coverage, not by tuning.

No training, no model. Takes the training-split GT disparity distribution and
the evaluation-split GT disparity distribution, and asks, for each candidate
scale distribution, what fraction of augmented TRAINING pixels would land in
each evaluation GT bin.

Pre-registered selection rule, fixed before the numbers were read:

    choose the narrowest candidate range such that, for EVERY evaluation bin
    that holds at least 1,000 evaluation pixels, the augmented training pixel
    fraction is at least the evaluation pixel fraction in that bin,

subject to the loss mask (scaled disparities >= 184 px are dropped by the
existing masked loss and are counted as lost, not as coverage).

Approximation stated rather than hidden: the crop window is scaled together
with the image, and crop position is uniform, so the augmented disparity
distribution is estimated by scaling the pooled GT distribution. This ignores
the mild correlation between disparity and image row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from src.datasets.kitti2015 import Kitti2015Stereo

BINS = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
MAXD = 184.0
RNG = np.random.default_rng(0)
N_DRAW = 24


def pooled(split: str) -> np.ndarray:
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split=split,
                         disparity_scale=256.0, occluded=True)
    v = []
    for i in range(len(ds)):
        d = ds[i].disparity
        v.append(d[d > 0].astype(np.float32))
    return np.concatenate(v)


def frac_by_bin(x: np.ndarray, denom: int) -> list:
    return [float(((x >= a) & (x < b)).sum()) / denom
            for a, b in zip(BINS[:-1], BINS[1:])]


def main() -> None:
    TR, VA = pooled("hailo_calib"), pooled("hailo_val")
    sub = TR[::4]                      # deterministic stride, keeps memory bounded
    va_frac = frac_by_bin(VA, VA.size)
    va_px = [int(((VA >= a) & (VA < b)).sum()) for a, b in zip(BINS[:-1], BINS[1:])]

    cands = {
        "s = 1.0 (ARM-V control)": lambda n: np.ones(n),
        "s ~ U(1.0, 2.0)": lambda n: RNG.uniform(1.0, 2.0, n),
        "s ~ logU(0.7, 1.7)": lambda n: np.exp(RNG.uniform(np.log(0.7), np.log(1.7), n)),
        "s ~ logU(0.5, 2.0)": lambda n: np.exp(RNG.uniform(np.log(0.5), np.log(2.0), n)),
    }
    out = {"bins": BINS, "eval_frac": va_frac, "eval_px": va_px, "candidates": {}}
    print("| GT bin | eval px | eval frac | " + " | ".join(cands) + " |")
    print("|" + "---|" * (3 + len(cands)))
    tables = {}
    for name, draw in cands.items():
        reps = []
        for _ in range(N_DRAW):
            s = draw(sub.size)
            reps.append(sub * s)
        allv = np.concatenate(reps)
        kept = allv[allv < MAXD]
        tables[name] = {"frac": frac_by_bin(kept, allv.size),
                        "lost_to_mask": float((allv >= MAXD).mean())}
    for i, (a, b) in enumerate(zip(BINS[:-1], BINS[1:])):
        print("| [%d,%d) | %d | %.5f | " % (a, b, va_px[i], va_frac[i])
              + " | ".join("%.5f" % tables[n]["frac"][i] for n in cands) + " |")
    print("| pixels lost to gt<184 loss mask | | | "
          + " | ".join("%.5f" % tables[n]["lost_to_mask"] for n in cands) + " |")

    print()
    print("RULE AS ORIGINALLY WRITTEN (kept, with its refutation, not silently replaced):")
    print("  'augmented train frac >= eval frac in EVERY eval bin with >=1000 px'")
    for name in cands:
        fails = [(BINS[i], BINS[i + 1], va_frac[i], tables[name]["frac"][i])
                 for i in range(len(va_px))
                 if va_px[i] >= 1000 and tables[name]["frac"][i] < va_frac[i]]
        print("  %-26s %s" % (name, "PASS" if not fails else
                              "fails at " + ", ".join("[%d,%d)" % (f[0], f[1]) for f in fails)))
    print("  -> the rule is UNSATISFIABLE BY CONSTRUCTION: both sides are")
    print("     probability vectors over the same bins, so no distribution can")
    print("     dominate another in every bin. The rule was ill-posed. Corrected below.")

    # SECOND ill-posed attempt, also kept: defining the deficient set by
    # histogram shortfall picks up [32,48) and [48,64), where the model's EPE is
    # 1.24-1.54 px -- its BEST range. Raising those and the tail together is the
    # same impossibility again. Deficiency must be defined by the measured
    # FAILURE, not by the histogram.
    ctrl = tables["s = 1.0 (ARM-V control)"]["frac"]
    hist_deficient = [i for i in range(len(va_px))
                      if va_px[i] >= 1000 and ctrl[i] < va_frac[i]]
    print()
    print("  second ill-posed attempt: deficient-by-histogram would be",
          ", ".join("[%d,%d)" % (BINS[i], BINS[i + 1]) for i in hist_deficient),
          "- which includes the model's BEST bins. Rejected.")

    # FAILURE-DEFINED deficient set: bins whose ARM-V 3-seed mean EPE exceeds
    # twice the GT<64 level. From PHASE2_ARMV_HIGH_DISPARITY_DIAGNOSTIC section 3
    # the GT<64 bins sit at 1.17-1.54 px and [64,80) is the first bin above 2x.
    deficient = [i for i in range(len(va_px))
                 if va_px[i] >= 1000 and BINS[i] >= 64]
    print()
    print("CORRECTED RULE (well-posed, failure-defined):")
    print("  deficient bins = measured-failure bins (ARM-V EPE > 2x its GT<64 level):",
          ", ".join("[%d,%d)" % (BINS[i], BINS[i + 1]) for i in deficient))
    print("  C1 coverage: augmented train frac >= eval frac in EVERY deficient bin")
    print("  C2 minimum disturbance: among candidates passing C1, take the one with")
    print("     the smallest total-variation distance to the CONTROL distribution")
    print("     (i.e. the narrowest range that fixes the measured shortfall)")
    print()
    print("| candidate | passes C1 | TV vs control | TV vs eval | cov ratio [96,112) | [112,128) | [128,144) |")
    print("|---|---|---|---|---|---|---|")
    best, best_tv = None, 1e9
    for name in cands:
        f = tables[name]["frac"]
        c1 = all(f[i] >= va_frac[i] for i in deficient)
        tv_ctrl = 0.5 * sum(abs(f[i] - ctrl[i]) for i in range(len(f)))
        tv_eval = 0.5 * sum(abs(f[i] - va_frac[i]) for i in range(len(f)))
        ratios = [f[i] / va_frac[i] if va_frac[i] else None for i in (6, 7, 8)]
        out["candidates"][name] = {
            "frac": f, "lost_to_mask": tables[name]["lost_to_mask"],
            "passes_C1": bool(c1), "tv_vs_control": tv_ctrl, "tv_vs_eval": tv_eval,
            "coverage_ratio_vs_eval": {"[96,112)": ratios[0], "[112,128)": ratios[1],
                                       "[128,144)": ratios[2]}}
        print("| %s | %s | %.4f | %.4f | %.2f | %.2f | %.2f |" % (
            name, "yes" if c1 else "no", tv_ctrl, tv_eval, *ratios))
        if c1 and tv_ctrl < best_tv and "control" not in name:
            best, best_tv = name, tv_ctrl
    out["deficient_bins"] = [{"lo": BINS[i], "hi": BINS[i + 1]} for i in deficient]
    out["selected"] = best
    print()
    print("SELECTED:", best)
    (HERE / "coverage_choice.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
