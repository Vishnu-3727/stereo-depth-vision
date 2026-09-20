"""SYNTHETIC MASK-VALIDITY CHECK — DESIGN ONLY.

ANALYTICAL / SYNTHETIC ONLY.
  * no trained checkpoint is loaded, no trained model instantiated
  * no real image, no dataset
  * no training, no optimizer, no .backward(), no .step()
  * this is NOT the experiment's random gate: it computes no h_rand over the
    preregistered units, produces no gate.json and makes no gate decision.
    It is the section-10 mechanism check, on synthetic fields, whose only job
    is to establish LOGICAL VALIDITY of the corrected mask.

WHAT IT TESTS  (section 10, items 1-6)

 S1 matched self-pair anchor       V[tau] == 0 exactly, and ONLY on the
                                    predicted band once a halo is injected
 S2 cross-pair construction        AB/BA built from the identical crops
 S3 interaction cancellation       V_AA + V_BB - V_AB - V_BA == 0 identically,
                                    INCLUDING inside the halo and the fill
 S4 affine aggregation             the aggregated-cost interaction is zero to
                                    round-off for any weights
 S5 nonlinear aggregation          the interaction is non-zero (the signal
                                    channel is the nonlinearity)
 S6 BOUNDARY CONTAMINATION EXCLUSION -- the decisive test:
        inject TWO DIFFERENT random halos into the same fields and require the
        statistic on the CORRECTED mask to be BIT-IDENTICAL between them,
        while the statistic on the GEOM-001 mask is not.

The halo is injected by overwriting the feature columns that the measured
extractor boundary contaminates.  That is a faithful stand-in: the real halo is
exactly "these columns do not obey the translation law", and its values are
arbitrary.

Run:  python synthetic_mask_check.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
import torch.nn as nn                                                # noqa: E402
import torch.nn.functional as Fn                                     # noqa: E402

from src.models.stereonet.aggregation import Aggregation             # noqa: E402
from src.models.stereonet.cost_volume import build_cost_volume       # noqa: E402
from src.models.stereonet.regression import soft_argmin              # noqa: E402
from phase2.models.scaled_regression import (                        # noqa: E402
    standardise_across_disparity,
)

STRIDE, D = 16, 12
CROP_W, CROP_H = 1136, 272
FW, FH = CROP_W // STRIDE, CROP_H // STRIDE          # 71 x 17
CHANNELS = 32
TAUS = [1, 2, 3, 4, 5, 6]
SEEDS = [0, 1, 2, 3]
SMOOTH = 1.5

# ---- the adopted CONSERVATIVE domain (domain_derivation.md) ---------------
HALO = 15
EXT_LO, EXT_HI = HALO, FW - 1 - HALO                 # [15, 55]
CV_LO, CV_HI = EXT_LO, EXT_HI - (D - 1)              # [15, 44]
AGG_R = 5
ST_LO, ST_HI = CV_LO + AGG_R, CV_HI - AGG_R          # [20, 39]

# ---- feature -> pixel, align_corners=True, in=71 out=1136 ----------------
# x maps to u = x*(FW-1)/(CROP_W-1); bilinear uses floor(u) and floor(u)+1.
#   need floor(u)   >= ST_LO  <=>  u >= ST_LO
#   need floor(u)+1 <= ST_HI  <=>  u <  ST_HI
def pixel_band(lo: int, hi: int) -> tuple[int, int]:
    num, den = FW - 1, CROP_W - 1                     # 70 / 1135
    x_lo = -(-lo * den // num)                        # ceil(lo*den/num)
    x_hi = (hi * den - 1) // num                      # largest x with x*num < hi*den
    while x_lo * num < lo * den:
        x_lo += 1
    while (x_hi + 1) * num < hi * den:
        x_hi += 1
    while x_hi * num >= hi * den:
        x_hi -= 1
    return x_lo, x_hi


PX_LO, PX_HI = pixel_band(ST_LO, ST_HI)               # expected 325, 632
ROW_LO, ROW_HI = 64, 208                              # inherited, conservative
OLD_PX_LO, OLD_PX_HI = 64, 864                        # the GEOM-001 mask


def smoothed_field(seed: int, h: int, w: int, c: int, sigma: float) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, c, h, w, generator=g)
    r = max(1, int(3 * sigma))
    co = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (co / sigma) ** 2)
    k = k / k.sum()
    x = Fn.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), padding=(0, r), groups=c)
    x = Fn.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), padding=(r, 0), groups=c)
    return x / x.std()


def inject_halo(f: torch.Tensor, seed: int) -> torch.Tensor:
    """Overwrite the boundary columns the extractor contaminates.

    Columns [0, HALO-1] and [FW-HALO, FW-1] of the crop's OWN feature map.
    Values are arbitrary -- that is the point.
    """
    out = f.clone()
    g = torch.Generator().manual_seed(seed)
    lo = torch.randn(f.shape[0], f.shape[1], f.shape[2], HALO, generator=g)
    hi = torch.randn(f.shape[0], f.shape[1], f.shape[2], HALO, generator=g)
    out[..., :HALO] = lo
    out[..., FW - HALO:] = hi
    return out


def statistic(agg, cells: dict, px_lo: int, px_hi: int):
    """Ibar(k) and argmin on a pixel band. The frozen readout order."""
    zs = {}
    for name, v in cells.items():
        cost = agg(v)
        up = Fn.interpolate(cost, size=(CROP_H, CROP_W), mode="bilinear",
                            align_corners=True)
        z = standardise_across_disparity(up, dim=1)
        zs[name] = z[0, :, ROW_LO:ROW_HI, px_lo:px_hi + 1].reshape(D, -1)
    inter = 0.5 * (zs["AA"] + zs["BB"]) - 0.5 * (zs["AB"] + zs["BA"])
    a = inter.detach().float().cpu().numpy().astype(np.float64)
    ibar = np.median(a, axis=1)
    return ibar, int(np.argmin(ibar))


def main() -> int:
    t0 = time.time()
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    print("feature grid %dx%d  adopted domain: ext [%d,%d] cv [%d,%d] stat [%d,%d]"
          % (FW, FH, EXT_LO, EXT_HI, CV_LO, CV_HI, ST_LO, ST_HI))
    print("corrected pixel band x in [%d,%d]  (%d cols);  GEOM-001 band [%d,%d)"
          % (PX_LO, PX_HI, PX_HI - PX_LO + 1, OLD_PX_LO, OLD_PX_HI), flush=True)

    base = [smoothed_field(1000, FH, FW + max(TAUS), CHANNELS, SMOOTH),
            smoothed_field(2000, FH, FW + max(TAUS), CHANNELS, SMOOTH)]
    PLAN = {"AA": (0, 0), "BB": (1, 1), "AB": (0, 1), "BA": (1, 0)}

    def build(tau: int, halo_seed: int | None):
        """The four cells at one tau. halo_seed=None -> no halo injected."""
        lf, rf = {}, {}
        for i in (0, 1):
            L = base[i][..., 0:FW]
            R = base[i][..., tau:tau + FW]
            if halo_seed is not None:
                L = inject_halo(L, halo_seed + 100 * i)
                R = inject_halo(R, halo_seed + 100 * i + 50)
            lf[i], rf[i] = L, R
        return {c: build_cost_volume(lf[li], rf[ri], D, "subtract", "left")
                for c, (li, ri) in PLAN.items()}

    res = {"domain": {"ext": [EXT_LO, EXT_HI], "cv": [CV_LO, CV_HI],
                      "stat": [ST_LO, ST_HI],
                      "pixel_band": [PX_LO, PX_HI],
                      "rows": [ROW_LO, ROW_HI],
                      "old_pixel_band": [OLD_PX_LO, OLD_PX_HI]},
           "S1": [], "S3": [], "S4": [], "S5": [], "S6": []}

    # ---------------- S1 : matched anchor, clean and with halo ------------
    for tau in TAUS:
        clean = build(tau, None)
        hal = build(tau, 7000)
        row = {"tau": tau}
        v = clean["AA"][0, :, tau, :, :]
        zc = np.flatnonzero(v.abs().amax(dim=0).amax(dim=0).cpu().numpy() == 0.0) \
            if v.dim() == 3 else None
        row["clean_zero_cols"] = [int(zc.min()), int(zc.max())] if zc.size else None
        vh = hal["AA"][0, :, tau, :, :]
        zh = np.flatnonzero(vh.abs().amax(dim=0).amax(dim=0).cpu().numpy() == 0.0)
        row["halo_zero_cols"] = [int(zh.min()), int(zh.max())] if zh.size else None
        row["halo_predicted_zero_cols"] = [EXT_LO, EXT_HI - tau]
        row["anchor_exact_on_cv_domain"] = bool(
            float(vh[:, :, CV_LO:CV_HI + 1].abs().max()) == 0.0)
        res["S1"].append(row)
        print("S1 tau=%d  zero cols with halo %s  predicted %s  anchor on cv domain %s"
              % (tau, row["halo_zero_cols"], row["halo_predicted_zero_cols"],
                 row["anchor_exact_on_cv_domain"]), flush=True)

    # ---------------- S3 : cost-volume cancellation, everywhere -----------
    for tau in TAUS:
        h = build(tau, 7000)
        resid = h["AA"] + h["BB"] - h["AB"] - h["BA"]
        scale = float(h["AB"].abs().max())
        res["S3"].append({"tau": tau, "max_abs_residual": float(resid.abs().max()),
                          "relative": float(resid.abs().max()) / max(scale, 1e-12),
                          "region": "ENTIRE tensor, halo and fill included"})
    print("S3 cancellation over the ENTIRE tensor: max rel %.3e"
          % max(r["relative"] for r in res["S3"]), flush=True)

    # ---------------- S4 / S5 : affine vs nonlinear aggregation ----------
    for seed in SEEDS:
        for arm in ("affine", "nonlinear"):
            torch.manual_seed(seed)
            agg = Aggregation(32, 32, 4).eval()
            if arm == "affine":
                for m in agg.modules():
                    if isinstance(m, nn.LeakyReLU):
                        m.negative_slope = 1.0
            for tau in (1, 4):
                cells = build(tau, 7000)
                costs = {c: agg(v) for c, v in cells.items()}
                r = costs["AA"] + costs["BB"] - costs["AB"] - costs["BA"]
                rel = float(r.abs().max()) / max(float(costs["AB"].abs().max()), 1e-12)
                res["S4" if arm == "affine" else "S5"].append(
                    {"seed": seed, "tau": tau, "relative_interaction": rel})
    print("S4 affine    aggregated-cost interaction: max rel %.3e"
          % max(r["relative_interaction"] for r in res["S4"]))
    print("S5 nonlinear aggregated-cost interaction: min rel %.3e  median %.3e"
          % (min(r["relative_interaction"] for r in res["S5"]),
             float(np.median([r["relative_interaction"] for r in res["S5"]]))),
          flush=True)

    # ---------------- S6 : BOUNDARY CONTAMINATION EXCLUSION --------------
    n_new_ident = n_new = n_old_ident = n_old = 0
    for seed in SEEDS:
        torch.manual_seed(seed)
        agg = Aggregation(32, 32, 4).eval()
        for tau in TAUS:
            a = build(tau, 7000)
            b = build(tau, 9000)          # a DIFFERENT arbitrary halo
            i_new_a, am_new_a = statistic(agg, a, PX_LO, PX_HI)
            i_new_b, am_new_b = statistic(agg, b, PX_LO, PX_HI)
            i_old_a, am_old_a = statistic(agg, a, OLD_PX_LO, OLD_PX_HI - 1)
            i_old_b, am_old_b = statistic(agg, b, OLD_PX_LO, OLD_PX_HI - 1)
            new_ident = bool(np.array_equal(i_new_a, i_new_b))
            old_ident = bool(np.array_equal(i_old_a, i_old_b))
            n_new += 1
            n_old += 1
            n_new_ident += int(new_ident)
            n_old_ident += int(old_ident)
            res["S6"].append({
                "seed": seed, "tau": tau,
                "corrected_mask_bit_identical": new_ident,
                "corrected_mask_max_abs_diff": float(np.abs(i_new_a - i_new_b).max()),
                "corrected_argmin": [am_new_a, am_new_b],
                "geom001_mask_bit_identical": old_ident,
                "geom001_mask_max_abs_diff": float(np.abs(i_old_a - i_old_b).max()),
                "geom001_argmin": [am_old_a, am_old_b],
            })
        print("S6 seed %d done (%.1fs)" % (seed, time.time() - t0), flush=True)

    res["S6_summary"] = {
        "corrected_mask_bit_identical": "%d/%d" % (n_new_ident, n_new),
        "geom001_mask_bit_identical": "%d/%d" % (n_old_ident, n_old),
        "corrected_max_abs_diff_over_all":
            max(r["corrected_mask_max_abs_diff"] for r in res["S6"]),
        "geom001_max_abs_diff_over_all":
            max(r["geom001_mask_max_abs_diff"] for r in res["S6"]),
        "geom001_argmin_changed_by_halo":
            int(sum(1 for r in res["S6"] if r["geom001_argmin"][0] != r["geom001_argmin"][1])),
        "corrected_argmin_changed_by_halo":
            int(sum(1 for r in res["S6"] if r["corrected_argmin"][0] != r["corrected_argmin"][1])),
    }
    res["meta"] = {"kind": "synthetic mask-validity check -- NOT an experiment result",
                   "trained_model_used": False, "real_data_used": False,
                   "training_performed": False, "is_the_experiment_gate": False,
                   "n_random_seeds": len(SEEDS), "wall_clock_s": time.time() - t0}
    (OUT / "synthetic_mask_check.json").write_text(json.dumps(res, indent=2),
                                                   encoding="utf-8")

    print("\n============ S6  BOUNDARY CONTAMINATION EXCLUSION ============")
    print("corrected mask  bit-identical under two different halos : %s"
          % res["S6_summary"]["corrected_mask_bit_identical"])
    print("   max |diff| over all cells                            : %.3e"
          % res["S6_summary"]["corrected_max_abs_diff_over_all"])
    print("GEOM-001 mask   bit-identical under two different halos : %s"
          % res["S6_summary"]["geom001_mask_bit_identical"])
    print("   max |diff| over all cells                            : %.3e"
          % res["S6_summary"]["geom001_max_abs_diff_over_all"])
    print("   argmin CHANGED by the halo, GEOM-001 mask            : %d/%d"
          % (res["S6_summary"]["geom001_argmin_changed_by_halo"], n_old))
    print("   argmin changed by the halo, corrected mask           : %d/%d"
          % (res["S6_summary"]["corrected_argmin_changed_by_halo"], n_new))
    print("\nwrote synthetic_mask_check.json  (%.1fs)" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
