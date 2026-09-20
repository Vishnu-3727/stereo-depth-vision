"""VALIDITY VALIDATION for the corrected geometric diagnostic — DESIGN ONLY.

WHAT THIS RUNS
--------------
The FEATURE EXTRACTOR and the COST VOLUME only.  Both are parameter-frozen
stages whose outputs are WIRING quantities.

WHAT THIS DOES NOT RUN
----------------------
  * no aggregation forward pass, trained or random
  * no standardisation, no softmax, no soft-argmin, no readout
  * no interaction I_tau(k), no argmin, no h, no h_rand, no h_trained
  * no gate, no G-PASS, no vertical arm, no search-free arm
  * no training, no optimizer, no .backward(), no .step()
  * no gate.json / trained.json / results.json is produced

The task specification (section 4) requires this validation to "extract
features" and to verify R_f[w] = L_f[w+tau] and V[tau,w] = 0 on every
preregistered scene and every tau, recording exact counts.  Section 12
prohibits running the EXPERIMENT arms.  This file executes exactly the former
and none of the latter, and stops before aggregation, as section 4 directs.

OUTPUT: validate_domain.json  (+ stdout captured to validate_domain.log)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CUBLAS = ":4096:8"
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402

from src.models.stereonet.stereonet import StereoNet, StereoNetConfig    # noqa: E402
from src.datasets.kitti2015 import normalize                             # noqa: E402
from phase2.models import scaled_regression                              # noqa: E402
from phase2.viz import core                                              # noqa: E402

W, H, STRIDE, D = 1232, 368, 16, 12
CROP_W, CROP_H = 1136, 272
FW, FH = CROP_W // STRIDE, CROP_H // STRIDE                 # 71 x 17
TAUS = [1, 2, 3, 4, 5, 6]
SPLIT = "hailo_val"
PAIRS = [(1, 2), (3, 4), (10, 11), (12, 13), (14, 15), (16, 17)]

# ---- the DERIVED domain under test (domain_derivation.md) ------------------
HALO = 15                       # extractor one-sided halo, feature cells
EXT_LO, EXT_HI = HALO, FW - 1 - HALO          # [15, 55]  extractor-clean
CV_LO, CV_HI = EXT_LO, EXT_HI - (D - 1)       # [15, 44]  every k in 0..11 clean
AGG_R = 5                                     # 5 x Conv3d(3x3x3) spatial radius
ST_LO, ST_HI = CV_LO + AGG_R, CV_HI - AGG_R   # [20, 39]  aggregation-clean
CHECKPOINTS = {
    "H2_seed0": ("phase2/diagnostics/determinism/20260909T041500Z_baseline/"
                 "checkpoints/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA_checkpoint.pth",
                 "581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a"),
    "H2_seed1": ("phase2/factorial/block_count_full/20260910T005550Z/checkpoints/"
                 "EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth",
                 "58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf"),
    "H2_seed2": ("phase2/factorial/block_count_full/20260910T005550Z/checkpoints/"
                 "EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth",
                 "245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69"),
}


class ValidationFailure(RuntimeError):
    pass


def sha256_bytes(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def crop(img: np.ndarray, y0: int, x0: int) -> np.ndarray:
    if y0 < 0 or x0 < 0 or y0 + CROP_H > H or x0 + CROP_W > W:
        raise ValidationFailure("crop (%d,%d) leaves the source" % (y0, x0))
    return img[y0:y0 + CROP_H, x0:x0 + CROP_W]


def load_extractor(key: str, device: str):
    rel, want = CHECKPOINTS[key]
    path = REPO_ROOT / rel
    got = sha256_bytes(path)
    if got != want:
        raise ValidationFailure("%s sha256 %s != %s" % (key, got, want))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = StereoNet(StereoNetConfig(cost_volume_shift="left"))
    scaled_regression.apply_to(model)
    model.load_state_dict(ckpt["model"])
    # the aggregation and the readout are NOT used anywhere in this file;
    # the aggregation is discarded so it cannot be invoked by accident.
    model.aggregation = None
    model.refinement = None
    return model.eval().to(device), got


def contiguous_true_run(flags: np.ndarray) -> tuple[int, int, int]:
    """Longest contiguous run of True. Returns (lo, hi, length); (-1,-1,0) if none."""
    best = (-1, -1, 0)
    i = 0
    n = flags.size
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            if j - i + 1 > best[2]:
                best = (i, j, j - i + 1)
            i = j + 1
        else:
            i += 1
    return best


def main() -> int:
    t0 = time.time()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device=%s  FW=%d FH=%d D=%d" % (device, FW, FH, D), flush=True)
    print("DERIVED domain under test:  extractor [%d,%d]  cost-volume [%d,%d]  "
          "statistic [%d,%d]" % (EXT_LO, EXT_HI, CV_LO, CV_HI, ST_LO, ST_HI),
          flush=True)

    scenes = {}
    for pi, (ia, ib) in enumerate(PAIRS):
        for who, idx in (("A", ia), ("B", ib)):
            s = core.load_scene(idx, split=SPLIT)
            scenes[(pi, who)] = {"src": s.left, "left": crop(s.left, 0, 0),
                                 "name": s.name, "index": idx}

    records, cancel, zero_checks = [], [], []
    halo_lo_seen, halo_hi_seen = set(), set()

    for ext_key in CHECKPOINTS:
        model, digest = load_extractor(ext_key, device)
        print("\nextractor %s (%s...)" % (ext_key, digest[:12]), flush=True)
        for pi in range(len(PAIRS)):
            lf, rf_all = {}, {}
            for who in ("A", "B"):
                sc = scenes[(pi, who)]
                lf[who] = model.feature_extractor(
                    torch.from_numpy(normalize(sc["left"])).to(device))
            for tau in TAUS:
                t = tau * STRIDE
                for who in ("A", "B"):
                    sc = scenes[(pi, who)]
                    rc = crop(sc["src"], 0, t)
                    # pixel-level identity, from the arrays
                    if not np.array_equal(rc[:, :CROP_W - t], sc["left"][:, t:]):
                        raise ValidationFailure(
                            "pixel identity failed %s tau=%d" % (sc["name"], tau))
                    rf_all[(who, tau)] = model.feature_extractor(
                        torch.from_numpy(normalize(rc)).to(device))

                    # ---- STEP 3/4 : where does Rf[w] == Lf[w+tau] hold? ----
                    a = rf_all[(who, tau)][0, :, :, :FW - tau]
                    b = lf[who][0, :, :, tau:]
                    eq = (a == b)                       # exact float equality
                    col_ok = eq.all(dim=0).all(dim=0).detach().cpu().numpy()
                    lo, hi, ln = contiguous_true_run(col_ok)
                    halo_lo_seen.add(int(lo))
                    halo_hi_seen.add(int(hi))
                    n_ok = int(col_ok.sum())
                    inside_all_ok = bool(col_ok[EXT_LO:EXT_HI - tau + 1].all())
                    # failure OUTSIDE must actually be present (step 4)
                    fails_left = bool(not col_ok[EXT_LO - 1]) if EXT_LO >= 1 else None
                    right_probe = EXT_HI - tau + 1
                    fails_right = (bool(not col_ok[right_probe])
                                   if right_probe < col_ok.size else None)
                    records.append({
                        "extractor": ext_key, "pair": pi, "scene": who,
                        "scene_name": scenes[(pi, who)]["name"], "tau": tau,
                        "valid_run_lo": int(lo), "valid_run_hi": int(hi),
                        "valid_run_len": int(ln), "n_valid_cols": n_ok,
                        "derived_band": [EXT_LO, EXT_HI - tau],
                        "derived_band_all_exact": inside_all_ok,
                        "fails_just_left_of_band": fails_left,
                        "fails_just_right_of_band": fails_right,
                    })
                    if not inside_all_ok:
                        raise ValidationFailure(
                            "HS1-A: Rf[w] != Lf[w+tau] inside the derived band, "
                            "%s %s tau=%d" % (ext_key, scenes[(pi, who)]["name"], tau))

                # ---- STEP 5 : V[tau, w] == 0 on the cost-volume domain -----
                plan = {"AA": ("A", "A"), "BB": ("B", "B"),
                        "AB": ("A", "B"), "BA": ("B", "A")}
                vols = {}
                for cell, (li, ri) in plan.items():
                    vols[cell] = model.cost_volume(lf[li], rf_all[(ri, tau)])
                for cell in ("AA", "BB"):
                    v = vols[cell][0, :, tau, :, CV_LO:CV_HI + 1]
                    mx = float(v.abs().max())
                    zero_checks.append({"extractor": ext_key, "pair": pi,
                                        "cell": cell, "tau": tau,
                                        "domain": [CV_LO, CV_HI],
                                        "max_abs": mx, "exact_zero": mx == 0.0,
                                        "n_cells": int(v.numel())})
                    if mx != 0.0:
                        raise ValidationFailure(
                            "HS1-B: V[tau=%d] != 0 on [%d,%d] for %s %s pair %d "
                            "(max %r)" % (tau, CV_LO, CV_HI, ext_key, cell, pi, mx))
                # full-width zero profile, for the record
                vfull = vols["AA"][0, :, tau, :, :].abs().amax(dim=(0, 1))
                zcols = np.flatnonzero(vfull.detach().cpu().numpy() == 0.0)
                zero_checks[-2]["full_width_zero_cols"] = [int(zcols.min()),
                                                           int(zcols.max())] if zcols.size else None

                # ---- 2x2 cancellation on REAL features (never measured before)
                resid = vols["AA"] + vols["BB"] - vols["AB"] - vols["BA"]
                scale = float(vols["AB"].abs().max())
                cancel.append({"extractor": ext_key, "pair": pi, "tau": tau,
                               "max_abs_residual": float(resid.abs().max()),
                               "volume_scale": scale,
                               "relative": float(resid.abs().max()) / max(scale, 1e-12)})
                del vols
            del lf, rf_all
            torch.cuda.empty_cache()
            print("  pair %d (%s/%s) ok  (%.1fs)"
                  % (pi, scenes[(pi, 'A')]["name"], scenes[(pi, 'B')]["name"],
                     time.time() - t0), flush=True)
        del model

    # ------------------------------------------------------------- summary
    runs_lo = sorted(halo_lo_seen)
    runs_hi = sorted(halo_hi_seen)
    band_exact = all(r["derived_band_all_exact"] for r in records)
    band_tight = all(r["valid_run_lo"] == EXT_LO
                     and r["valid_run_hi"] == EXT_HI - r["tau"] for r in records)
    fails_l = [r["fails_just_left_of_band"] for r in records]
    fails_r = [r["fails_just_right_of_band"] for r in records
               if r["fails_just_right_of_band"] is not None]
    summary = {
        "kind": "VALIDITY VALIDATION -- extractor + cost volume only; no aggregation, "
                "no readout, no statistic, no gate",
        "aggregation_run": False, "readout_run": False, "statistic_computed": False,
        "training_performed": False,
        "device": device,
        "geometry": {"crop": [CROP_W, CROP_H], "feature_grid": [FW, FH],
                     "stride": STRIDE, "D": D, "taus": TAUS},
        "derived_domain": {
            "extractor_halo_cells": HALO,
            "extractor_clean": [EXT_LO, EXT_HI],
            "cost_volume_clean": [CV_LO, CV_HI],
            "aggregation_radius_cells": AGG_R,
            "statistic_clean": [ST_LO, ST_HI],
        },
        "n_records": len(records),
        "n_zero_checks": len(zero_checks),
        "n_cancellation_checks": len(cancel),
        "translation_validity": {
            "derived_band_exact_everywhere": band_exact,
            "measured_band_equals_derived_band_everywhere": band_tight,
            "observed_valid_run_lo_values": runs_lo,
            "observed_valid_run_hi_values": runs_hi,
            "fails_immediately_left_of_band": {"all": all(x is True for x in fails_l),
                                               "n": len(fails_l)},
            "fails_immediately_right_of_band": {"all": all(x is True for x in fails_r),
                                                "n": len(fails_r)},
            "min_n_valid_cols": min(r["n_valid_cols"] for r in records),
            "max_n_valid_cols": max(r["n_valid_cols"] for r in records),
        },
        "cost_volume_anchor": {
            "all_exact_zero_on_domain": all(z["exact_zero"] for z in zero_checks),
            "max_abs_over_all": max(z["max_abs"] for z in zero_checks),
            "cells_checked_per_check": zero_checks[0]["n_cells"],
        },
        "cancellation_real_features": {
            "max_relative_residual": max(c["relative"] for c in cancel),
            "min_relative_residual": min(c["relative"] for c in cancel),
            "median_relative_residual": float(np.median([c["relative"] for c in cancel])),
        },
        "records": records, "zero_checks": zero_checks, "cancellation": cancel,
        "wall_clock_s": time.time() - t0,
    }
    (OUT / "validate_domain.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")

    print("\n================ VALIDATION SUMMARY ================")
    print("records (extractor x pair x scene x tau) : %d" % len(records))
    print("derived band exact everywhere            : %s" % band_exact)
    print("measured band == derived band everywhere : %s" % band_tight)
    print("observed valid-run lo values             : %s" % runs_lo)
    print("observed valid-run hi values             : %s" % runs_hi)
    print("fails immediately left  of band (all)    : %s" % all(x is True for x in fails_l))
    print("fails immediately right of band (all)    : %s" % all(x is True for x in fails_r))
    print("V[tau] exact zero on [%d,%d] everywhere  : %s"
          % (CV_LO, CV_HI, all(z["exact_zero"] for z in zero_checks)))
    print("   checks=%d  cells per check=%d  max|V|=%r"
          % (len(zero_checks), zero_checks[0]["n_cells"],
             max(z["max_abs"] for z in zero_checks)))
    print("2x2 cancellation on REAL features        : max rel %.4e  median %.4e"
          % (max(c["relative"] for c in cancel),
             float(np.median([c["relative"] for c in cancel]))))
    print("wall_clock_s=%.1f" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
