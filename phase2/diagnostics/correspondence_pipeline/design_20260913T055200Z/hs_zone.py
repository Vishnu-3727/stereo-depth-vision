"""Pre-freeze hard stops for EXP-CORRESPONDENCE-ZONE-001.

MODEL-FREE.  No checkpoint is opened, no training, no optimizer.  Frozen modules
are imported and used verbatim.  A randomly initialised FeatureExtractor is used
only where the quantity measured is a RECEPTIVE-FIELD SUPPORT property, which is
a property of the architecture and not of any trained weights (same justification
as HS-BAND).

STAGES
  --stage 1   HS-SLACK  crop origins in range for every Delta, no zero-fill
              HS-GT     GT coverage inside the two interiors
              HS-FRAME1 pin the disparity SIGN on real data with GT, no model
  --stage 2   HS-FRAME2 feature-level: matched candidate location under a right
                        crop-origin offset
              HS-Z      seam blindness: zone-interior features depend ONLY on
                        zone content (measured, not derived)
              HS-SYM    the zone-shifted cost volume is NOT a global
                        candidate-axis translate of the unshifted one

Any failure is reported as FAIL and halts the design; nothing is repaired.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("C:/Users/vishn/stereo_depth_vision")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from phase2.viz import core                                              # noqa: E402

# ------------------------------------------------------------ frozen geometry
FRAME_H, FRAME_W = 368, 1232
CROP_H, CROP_W = 272, 1136
STRIDE, D = 16, 12
FW, FH = CROP_W // STRIDE, CROP_H // STRIDE          # 71 x 17
Y0 = 96                       # bottom 272 rows of the frame: where KITTI GT lives
XL = 0                        # LEFT crop origin, fixed in every arm
DELTAS = [0, 1, 2, 3, 4, 5, 6]                        # x_R = XL + 16*Delta in [0,96]
# AMENDED after HS-GT stage 1 run A: measured GT in these crops is 0.5-7 candidates,
# so the offset must INCREASE disparity (d' = d + 16*Delta), not decrease it.
SEAM = CROP_W // 2            # 568
EXCL = 318                    # 238 px extractor radius + 80 px (5 cells) aggregation
ZONE_INT = (0, SEAM - EXCL)            # [0, 250)
COMP_INT = (SEAM + EXCL, CROP_W)       # [886, 1136)
SPLIT = "hailo_val"
PAIRS = [(1, 2), (3, 4), (10, 11), (12, 13), (14, 15), (16, 17)]
SCENES = sorted({i for p in PAIRS for i in p})


def crop(img: np.ndarray, y0: int, x0: int) -> np.ndarray:
    return img[y0:y0 + CROP_H, x0:x0 + CROP_W]


def stage1() -> dict:
    out: dict = {"stage": 1, "y0": Y0, "xl": XL, "deltas": DELTAS,
                 "seam": SEAM, "exclusion_px": EXCL,
                 "zone_interior": list(ZONE_INT), "comp_interior": list(COMP_INT)}

    # ---------------------------------------------------------------- HS-SLACK
    origins = [XL + STRIDE * d for d in DELTAS]
    in_range = all(0 <= x and x + CROP_W <= FRAME_W for x in origins) and \
               0 <= Y0 and Y0 + CROP_H <= FRAME_H
    out["HS-SLACK"] = {"right_origins": origins,
                       "all_in_range": bool(in_range),
                       "frame": [FRAME_H, FRAME_W], "crop": [CROP_H, CROP_W]}

    shapes, zerofill = set(), []
    scenes = {}
    for i in SCENES:
        sc = core.load_scene(i, split=SPLIT)
        scenes[i] = sc
        shapes.add(sc.left.shape)
        # zero-fill detection: pad_and_crop zero-fills bottom/right for small frames
        band = sc.left[Y0:Y0 + CROP_H, :]
        zerofill.append({"scene": sc.name,
                         "all_zero_rows": int((band.sum(axis=(1, 2)) == 0).sum()),
                         "all_zero_cols": int((sc.left[:, :, :].sum(axis=(0, 2)) == 0).sum())})
    out["HS-SLACK"]["frame_shapes"] = sorted(str(s) for s in shapes)
    out["HS-SLACK"]["zero_fill"] = zerofill
    out["HS-SLACK"]["PASS"] = bool(
        in_range and shapes == {(FRAME_H, FRAME_W, 3)}
        and all(z["all_zero_rows"] == 0 and z["all_zero_cols"] == 0 for z in zerofill))

    # ------------------------------------------------------------------- HS-GT
    cov = []
    for i in SCENES:
        sc = scenes[i]
        g = crop(sc.gt_disparity, Y0, XL)
        v = g > 0
        zi = v[:, ZONE_INT[0]:ZONE_INT[1]]
        ci = v[:, COMP_INT[0]:COMP_INT[1]]
        cov.append({"scene": sc.name,
                    "zone_interior_valid_px": int(zi.sum()),
                    "zone_interior_frac": float(zi.mean()),
                    "comp_interior_valid_px": int(ci.sum()),
                    "comp_interior_frac": float(ci.mean()),
                    "d_candidates_p05_p95": [float(np.percentile(g[v] / STRIDE, 5)),
                                             float(np.percentile(g[v] / STRIDE, 95))]})
    out["HS-GT"] = {"per_scene": cov,
                    "min_zone_valid": min(c["zone_interior_valid_px"] for c in cov),
                    "min_comp_valid": min(c["comp_interior_valid_px"] for c in cov)}
    # a usable slope needs GT in both interiors in every scene; 500 px ~ 30 cells
    out["HS-GT"]["PASS"] = bool(out["HS-GT"]["min_zone_valid"] >= 500
                                and out["HS-GT"]["min_comp_valid"] >= 500)

    # -------------------------------------------------------------- HS-FRAME1
    # Which convention holds:  right[x] ~ left[x+d]  (H_PLUS)  or  left[x-d] (H_MINUS)?
    # Photometric error over GT-valid pixels, real data, NO MODEL.
    rows = []
    for i in SCENES:
        sc = scenes[i]
        L = sc.left.astype(np.float64).mean(axis=2)
        R = sc.right.astype(np.float64).mean(axis=2)
        g = sc.gt_disparity
        ys, xs = np.nonzero(g > 0)
        dd = np.rint(g[ys, xs]).astype(int)
        keep = (xs + dd < FRAME_W) & (xs - dd >= 0)
        ys, xs, dd = ys[keep], xs[keep], dd[keep]
        e_plus = np.abs(R[ys, xs] - L[ys, xs + dd]).mean()
        e_minus = np.abs(R[ys, xs] - L[ys, xs - dd]).mean()
        e_zero = np.abs(R[ys, xs] - L[ys, xs]).mean()
        rows.append({"scene": sc.name, "n": int(len(dd)),
                     "MAE_right_vs_left_x_plus_d": float(e_plus),
                     "MAE_right_vs_left_x_minus_d": float(e_minus),
                     "MAE_right_vs_left_x": float(e_zero),
                     "winner": "x+d" if e_plus < e_minus else "x-d"})
    out["HS-FRAME1"] = {"per_scene": rows,
                        "unanimous": len({r["winner"] for r in rows}) == 1,
                        "convention": rows[0]["winner"]}
    # And the implied effect of the right-crop origin offset:
    #   right_crop[x] = frame_R[xR + x],  left_crop[x] = frame_L[XL + x]
    #   with frame_R[u] = frame_L[u + d]:  content at right_crop[x] sits at
    #   frame_L[xR + x + d] = left_crop[xR - XL + x + d]
    #   so d' = (xR - XL) + d = d + 16*Delta        (for xR = XL + 16*Delta)
    out["HS-FRAME1"]["implied_d_prime"] = "d + 16*Delta  (candidates: d/16 + Delta), for xR = XL + 16*Delta"
    out["HS-FRAME1"]["PASS"] = bool(out["HS-FRAME1"]["unanimous"]
                                    and rows[0]["winner"] == "x+d")
    out["VERDICT"] = "PASS" if all(out[k]["PASS"] for k in
                                   ("HS-SLACK", "HS-GT", "HS-FRAME1")) else "FAIL"
    return out


def stage2() -> dict:
    import torch
    from src.models.stereonet.feature_extractor import FeatureExtractor
    from src.models.stereonet.cost_volume import build_cost_volume
    from src.datasets.kitti2015 import normalize
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    out: dict = {"stage": 2}
    ex_seeds = [11, 12, 13]
    exts = []
    for s in ex_seeds:
        torch.manual_seed(s)
        exts.append(FeatureExtractor().eval())
    out["extractor_seeds"] = ex_seeds

    def feats(ex, img):
        return ex(torch.from_numpy(normalize(img)))

    # ------------------------------------------------------------- HS-FRAME2
    # synthetic, no data: Rf[w] = Lf[w+tau]  ->  the matched candidate is k=tau
    torch.manual_seed(3)
    lf = torch.randn(1, 32, FH, FW)
    f2 = []
    for tau in (1, 3, 5):
        rf = torch.nn.functional.pad(lf, (0, tau))[..., tau:]
        v = build_cost_volume(lf, rf, D, "subtract", "left")
        prof = [float(v[:, :, k, :, 15:45].abs().mean()) for k in range(D)]
        f2.append({"tau": tau, "argmin_k": int(np.argmin(prof)),
                   "profile_min": min(prof), "profile_second": sorted(prof)[1]})
    out["HS-FRAME2"] = {"synthetic": f2,
                        "PASS": all(r["argmin_k"] == r["tau"] and r["profile_min"] == 0.0
                                    for r in f2)}

    # ------------------------------------------------------------------ HS-Z
    sc = core.load_scene(SCENES[0], split=SPLIT)
    delta = 3
    xr = XL + STRIDE * delta
    r_none = crop(sc.right, Y0, XL)
    r_all = crop(sc.right, Y0, xr)
    r_zone = r_none.copy()
    r_zone[:, :SEAM] = r_all[:, :SEAM]           # zone = left half gets the offset
    zc = (ZONE_INT[0] // STRIDE, -(-ZONE_INT[1] // STRIDE))     # cell range, ceil
    cc = (-(-COMP_INT[0] // STRIDE), COMP_INT[1] // STRIDE)
    rows = []
    with torch.no_grad():
        for ex in exts:
            fz, fa, fn = feats(ex, r_zone), feats(ex, r_all), feats(ex, r_none)
            d_zone = float((fz[..., zc[0]:zc[1]] - fa[..., zc[0]:zc[1]]).abs().max())
            d_comp = float((fz[..., cc[0]:cc[1]] - fn[..., cc[0]:cc[1]]).abs().max())
            # sanity: the arms are genuinely different somewhere
            d_any = float((fz - fn).abs().max())
            rows.append({"zone_interior_vs_all_shifted": d_zone,
                         "comp_interior_vs_unshifted": d_comp,
                         "max_diff_anywhere": d_any})
    out["HS-Z"] = {"delta": delta, "zone_cells": list(zc), "comp_cells": list(cc),
                   "per_seed": rows,
                   "PASS": all(r["zone_interior_vs_all_shifted"] == 0.0
                               and r["comp_interior_vs_unshifted"] == 0.0
                               and r["max_diff_anywhere"] > 0.0 for r in rows)}

    # ---------------------------------------------------------------- HS-SYM
    # is V_zone a GLOBAL candidate-axis translate of V_none anywhere?
    with torch.no_grad():
        ex = exts[0]
        lfr = feats(ex, crop(sc.left, Y0, XL))
        v_none = build_cost_volume(lfr, feats(ex, r_none), D, "subtract", "left")
        v_zone = build_cost_volume(lfr, feats(ex, r_zone), D, "subtract", "left")
        best = []
        for s in range(-6, 7):
            ks = [k for k in range(D) if 0 <= k - s < D]
            a = v_zone[:, :, [k - s for k in ks], :, :]
            b = v_none[:, :, ks, :, :]
            for nm, (lo, hi) in (("zone", zc), ("comp", cc)):
                dif = float((a[..., lo:hi] - b[..., lo:hi]).abs().max())
                best.append({"s": s, "region": nm, "max_abs_diff": dif})
    zmin = min(r["max_abs_diff"] for r in best if r["region"] == "zone")
    cmin = min(r["max_abs_diff"] for r in best if r["region"] == "comp")
    out["HS-SYM"] = {"min_over_shift_zone": zmin, "min_over_shift_comp": cmin,
                     "PASS": bool(zmin > 0.0)}
    out["VERDICT"] = "PASS" if all(out[k]["PASS"] for k in
                                   ("HS-FRAME2", "HS-Z", "HS-SYM")) else "FAIL"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    a = ap.parse_args()
    print(json.dumps(stage1() if a.stage == 1 else stage2(), indent=1))
