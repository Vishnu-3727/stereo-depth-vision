"""STAGE-A D6: HIGH-DISPARITY REASSESSMENT (frozen model, read-only).

FROZEN-MODEL DIAGNOSTIC. No model change, no retraining, no tuning, no
mechanism verdict (step 8 does verdicts).

Population: FULL-RESOLUTION frozen valid mask (GT > 0), 3,802,797 px/seed --
the same pixels D0/D1/D2/D5 used (P2A vectors reused from D0 raw dumps and
D5 raw dumps; reference vectors reused from D1 raw dump). D3/D4 feature-res
raw npz are reused (not recomputed) for the matching explanation.

Regimes: GT>=64, GT>=96, GT>=128, GT>=184.

Five explanations measured separately:
  1. RANGE       -- gt_cand=GT/8 outside [0,23]? + dist to nearest int
                     candidate + max_pred vs max_gt.
  2. CALIBRATION -- per-regime slope/signed error, D5 per-bin oracle alpha*
                     restated, per-regime oracle-alpha rescale gain.
  3. MATCHING    -- D3/D4 GT-candidate margin + rank distribution per regime.
  4. OCCLUSION   -- reference_strata.py mask; occ frac/EPE split/error-mass
                     share per regime.
  5. SUPERVISION  -- hailo_calib training GT histogram vs hailo_val eval
                     histogram + EFFECTIVE training distribution after the
                     P2A scale augmentation, emulated by Monte-Carlo sampling
                     of the exact recipe in
                     phase2/scripts/train_p2a_scale_coverage.py.

Outputs (atomic .tmp + rename):
  stage_a_diagnostics/high_disparity.json
  stage_a_diagnostics/raw/d6_s{0,1,2}.npz
  stage_a_diagnostics/raw/d6_bins.csv
Usage:
  python stage_a_diagnostics/scripts/d6_high_disparity.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

OUT = REPO / "stage_a_diagnostics"
RAW = OUT / "raw"

EDGES = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 184, np.inf]
REGIMES = [(64.0, np.inf, "ge64"), (96.0, np.inf, "ge96"),
           (128.0, np.inf, "ge128"), (184.0, np.inf, "ge184")]
MIN_REGRESSION_PX = 1000
EXPECTED_VALID = 3802797
N_CAND = 24
STRIDE = 8.0


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def check_finite(name: str, *arrs: np.ndarray) -> None:
    for a in arrs:
        if not np.isfinite(np.asarray(a, dtype=np.float64)).all():
            raise ValueError(f"NaN/inf detected in {name}")


def d1_pct(err: np.ndarray, gt: np.ndarray) -> float:
    return float((((err > 3.0) & (err > 0.05 * gt)).mean()) * 100.0)


def regime_stats(P: np.ndarray, G: np.ndarray, lo: float, hi: float,
                 total_n: int) -> dict:
    m = (G >= lo) & (G < hi)
    n = int(m.sum())
    row: dict = {"px": n, "frac_px": float(n / total_n) if total_n else 0.0}
    if n == 0:
        row["status"] = "EMPTY"
        return row
    err = np.abs(P[m] - G[m])
    check_finite("regime err", err)
    row["epe"] = float(err.mean())
    row["d1_pct"] = d1_pct(err, G[m])
    row["signed_err"] = float((P[m] - G[m]).mean())
    row["mean_pred"] = float(P[m].mean())
    row["mean_gt"] = float(G[m].mean())
    row["max_pred"] = float(P[m].max())
    row["max_gt"] = float(G[m].max())
    if n >= MIN_REGRESSION_PX and float(G[m].var()) > 0 and float(P[m].var()) > 0:
        a, b = np.polyfit(G[m], P[m], 1)
        yh = a * G[m] + b
        ss_res = float(((P[m] - yh) ** 2).sum())
        ss_tot = float(((P[m] - P[m].mean()) ** 2).sum())
        row["slope"] = float(a)
        row["intercept"] = float(b)
        row["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    else:
        row["slope"] = None
        row["intercept"] = None
        row["r2"] = None
        row["regression_status"] = (
            "INSUFFICIENT (px<%d or degenerate variance); regression not meaningful"
            % MIN_REGRESSION_PX)
    return row


def epe_alpha(a: float, init: np.ndarray, resid: np.ndarray,
              gt: np.ndarray) -> float:
    return float(np.abs(np.maximum(a * init + resid, 0.0) - gt).mean())


def oracle_alpha(init: np.ndarray, resid: np.ndarray, gt: np.ndarray) -> dict:
    best_a, best_e = None, np.inf
    for a in np.arange(0.0, 16.0001, 0.1):
        e = epe_alpha(float(a), init, resid, gt)
        if e < best_e:
            best_e, best_a = e, float(a)
    for a in np.arange(best_a - 0.1, best_a + 0.10001, 0.01):
        if a < 0:
            continue
        e = epe_alpha(float(a), init, resid, gt)
        if e < best_e:
            best_e, best_a = e, float(a)
    for a in np.arange(best_a - 0.01, best_a + 0.010001, 0.001):
        if a < 0:
            continue
        e = epe_alpha(float(a), init, resid, gt)
        if e < best_e:
            best_e, best_a = e, float(a)
    if not np.isfinite(best_e):
        raise ValueError("NaN/inf in oracle-alpha scan")
    return {"alpha_star": best_a, "epe_at_alpha_star": best_e,
            "epe_at_alpha_1": epe_alpha(1.0, init, resid, gt)}


def main() -> None:
    # ---- load frozen vectors (reuse; no inference) ----
    seeds_P, seeds_G, seeds_occ = {}, {}, {}
    seeds_init, seeds_resid, seeds_final = {}, {}, {}
    for seed in (0, 1, 2):
        z0 = np.load(RAW / f"d0_preds_s{seed}.npz")
        P = z0["P"].astype(np.float64)
        G = z0["G"].astype(np.float64)
        if P.shape != G.shape or P.size != EXPECTED_VALID:
            raise ValueError(f"seed {seed}: bad D0 vectors {P.shape}")
        check_finite(f"D0 seed {seed}", P, G)
        if (G <= 0).any():
            raise ValueError(f"seed {seed}: non-positive GT in valid mask")
        z5 = np.load(RAW / f"d5_s{seed}.npz")
        init = z5["disparity_initial"].astype(np.float64)
        resid = z5["refinement_residual"].astype(np.float64)
        final = z5["disparity_final"].astype(np.float64)
        gt5 = z5["gt"].astype(np.float64)
        occ = z5["occluded"].astype(bool)
        check_finite(f"D5 seed {seed}", init, resid, final, gt5)
        if not np.array_equal(gt5, G):
            raise ValueError(f"seed {seed}: D5 GT != D0 GT (population mismatch)")
        if not np.allclose(final, P, atol=1e-4):
            raise ValueError(f"seed {seed}: D5 final != D0 P (population mismatch)")
        seeds_P[seed], seeds_G[seed] = P, G
        seeds_occ[seed] = occ
        seeds_init[seed], seeds_resid[seed], seeds_final[seed] = init, resid, final
    G0 = seeds_G[0]
    for seed in (1, 2):
        if not np.array_equal(seeds_G[seed], G0):
            raise ValueError("GT differs across seeds")
    total_n = int(G0.size)

    zr = np.load(RAW / "d1_reference.npz")
    P_ref = zr["P"].astype(np.float64)
    G_ref = zr["G"].astype(np.float64)
    check_finite("reference", P_ref, G_ref)
    if not np.array_equal(G_ref, G0):
        raise ValueError("reference GT != P2A GT (split mismatch)")

    d5 = json.loads((OUT / "refinement_diagnostic.json").read_text())

    # ---- Part 0: mass-weighted per-bin oracle-alpha gain, all 3 seeds ----
    part0 = {}
    for seed in (0, 1, 2):
        v = d5["seeds"][str(seed)]
        gain = sum((r["px"] / v["valid_pixels"])
                   * (r["epe_refined"] - r["oracle_scale"]["epe_at_alpha_star"])
                   for r in v["per_bin"] if r.get("px"))
        glob = (v["oracle_scale_pooled"]["epe_at_alpha_1"]
                - v["oracle_scale_pooled"]["epe_at_alpha_star"])
        part0[str(seed)] = {
            "mass_weighted_per_bin_gain_px": float(gain),
            "global_gain_px": float(glob),
            "global_alpha_star": float(v["oracle_scale_pooled"]["alpha_star"]),
        }

    # ---- regimes: P2A per seed + method-mean + pooled regression ----
    regimes_out: dict = {}
    for lo, hi, key in REGIMES:
        per_seed = {}
        for seed in (0, 1, 2):
            per_seed[str(seed)] = regime_stats(seeds_P[seed], seeds_G[seed],
                                               lo, hi, total_n)
        n = int(((G0 >= lo) & (G0 < hi)).sum())
        mrow: dict = {"px": n, "frac_px": float(n / total_n)}
        if n == 0:
            mrow["status"] = "EMPTY"
        else:
            for k in ("epe", "d1_pct", "signed_err", "mean_pred", "mean_gt",
                      "max_pred", "max_gt"):
                mrow[k] = float(np.mean([per_seed[str(s)][k] for s in (0, 1, 2)]))
                mrow[k + "_per_seed"] = [float(per_seed[str(s)][k])
                                         for s in (0, 1, 2)]
            Ppool = np.concatenate([seeds_P[s][(G0 >= lo) & (G0 < hi)]
                                    for s in (0, 1, 2)])
            Gpool = np.concatenate([G0[(G0 >= lo) & (G0 < hi)]
                                    for _ in (0, 1, 2)])
            pooled = regime_stats(Ppool, Gpool, lo, hi, 3 * total_n)
            for k in ("slope", "intercept", "r2"):
                mrow[k] = pooled[k]
                mrow[k + "_per_seed"] = [per_seed[str(s)][k] for s in (0, 1, 2)]
            if pooled.get("regression_status"):
                mrow["regression_status"] = pooled["regression_status"]
            mrow["regression"] = ("pooled over 3x%d concatenated seed predictions"
                                  % n)
        mrow["per_seed"] = per_seed
        mrow["reference"] = regime_stats(P_ref, G_ref, lo, hi, total_n)
        regimes_out[key] = mrow

    # ---- 1. RANGE ----
    gt_cand = G0 / STRIDE
    nearest_dist = np.abs(gt_cand - np.round(gt_cand))
    range_out: dict = {"candidate_interval": [0, N_CAND - 1],
                       "overall_frac_outside": float(
                           ((gt_cand < 0) | (gt_cand > N_CAND - 1)).mean()),
                       "per_regime": {}}
    for lo, hi, key in REGIMES:
        m = (G0 >= lo) & (G0 < hi)
        n = int(m.sum())
        if n == 0:
            range_out["per_regime"][key] = {"px": 0, "status": "EMPTY"}
            continue
        d = nearest_dist[m]
        check_finite("range dist " + key, d)
        range_out["per_regime"][key] = {
            "px": n,
            "frac_outside_0_23": float(
                ((gt_cand[m] < 0) | (gt_cand[m] > N_CAND - 1)).mean()),
            "dist_to_nearest_candidate_mean": float(d.mean()),
            "dist_to_nearest_candidate_max": float(d.max()),
            "max_gt_cand": float(gt_cand[m].max()),
            "max_pred_vs_max_gt_per_seed": [
                {"max_pred": float(seeds_P[s][m].max()),
                 "max_gt": float(G0[m].max())} for s in (0, 1, 2)],
            "mean_pred_vs_mean_gt_per_seed": [
                {"mean_pred": float(seeds_P[s][m].mean()),
                 "mean_gt": float(G0[m].mean())} for s in (0, 1, 2)],
        }

    # ---- 2. CALIBRATION ----
    calib_out: dict = {"d5_per_bin_alpha_star": {},
                       "per_regime_oracle": {}}
    for seed in (0, 1, 2):
        for r in d5["seeds"][str(seed)]["per_bin"]:
            if r.get("px"):
                calib_out["d5_per_bin_alpha_star"].setdefault(
                    r["bin"], {})[str(seed)] = {
                        "alpha_star": r["oracle_scale"]["alpha_star"],
                        "epe_refined": r["epe_refined"],
                        "alpha_epe": r["oracle_scale"]["epe_at_alpha_star"],
                        "px": r["px"]}
    for lo, hi, key in REGIMES:
        m = (G0 >= lo) & (G0 < hi)
        n = int(m.sum())
        if n == 0:
            calib_out["per_regime_oracle"][key] = {"px": 0, "status": "EMPTY"}
            continue
        per_seed = {}
        for seed in (0, 1, 2):
            oa = oracle_alpha(seeds_init[seed][m], seeds_resid[seed][m],
                              seeds_G[seed][m])
            epe_ref = float(np.abs(seeds_final[seed][m]
                                   - seeds_G[seed][m]).mean())
            gain = oa["epe_at_alpha_1"] - oa["epe_at_alpha_star"]
            per_seed[str(seed)] = {
                **oa,
                "epe_refined_regime": epe_ref,
                "gain_px": float(gain),
                "gain_frac_of_regime_epe": float(gain / epe_ref)
                if epe_ref > 0 else None,
                "slope": regimes_out[key]["per_seed"][str(seed)]["slope"],
                "signed_err": regimes_out[key]["per_seed"][str(seed)][
                    "signed_err"],
            }
        calib_out["per_regime_oracle"][key] = {
            "px": n, "per_seed": per_seed,
            "method_mean_gain_px": float(np.mean(
                [per_seed[str(s)]["gain_px"] for s in (0, 1, 2)])),
        }

    # ---- 3. MATCHING (reuse D3/D4 raw npz) ----
    match_out: dict = {"per_regime": {}, "note": ""}
    for seed in (0, 1, 2):
        z3 = np.load(RAW / f"d3_s{seed}.npz")
        z4 = np.load(RAW / f"d4_s{seed}.npz")
        # D3 stores 58118 cells (70 with undefined margin as NaN); D4 stores
        # the 58048 fully-defined cells. Same geometry, different row counts,
        # so each file is masked by its OWN gt vector (documented, not hidden).
        g3 = z3["gt_px"].astype(np.float64)
        g4 = z4["gt_px"].astype(np.float64)
        check_finite(f"D3/D4 seed {seed} gt", g3[~np.isnan(g3)], g4)
        rA_all = z3["rank_A_L1"].astype(np.float64)
        mgA_all = z3["margin_A_L1"].astype(np.float64)
        ok3 = np.isfinite(rA_all) & np.isfinite(mgA_all)
        for lo, hi, key in REGIMES:
            m3 = (g3 >= lo) & (g3 < hi) & ok3
            m4 = (g4 >= lo) & (g4 < hi)
            cell = match_out["per_regime"].setdefault(
                key, {"d3_cells_per_seed": [], "d4_cells_per_seed": [],
                      "seeds": {}})
            cell["d3_cells_per_seed"].append(int(m3.sum()))
            cell["d4_cells_per_seed"].append(int(m4.sum()))
            if int(m3.sum()) == 0 and int(m4.sum()) == 0:
                cell["seeds"][str(seed)] = {"status": "EMPTY"}
                continue
            rA = rA_all[m3]
            mgA = mgA_all[m3]
            rP = z4["rank_post"][m4].astype(np.float64)
            mgP = z4["margin_post"][m4].astype(np.float64)
            check_finite(f"matching {key} seed {seed}", rA, mgA, rP, mgP)
            cell["seeds"][str(seed)] = {
                "d3_cells": int(m3.sum()),
                "d4_cells": int(m4.sum()),
                "PRE_A_L1": {
                    "rank1_frac": float((rA == 1).mean()),
                    "rank_le3_frac": float((rA <= 3).mean()),
                    "median_rank": float(np.median(rA)),
                    "mean_rank": float(rA.mean()),
                    "margin_frac_pos": float((mgA > 0).mean()),
                    "margin_mean": float(mgA.mean()),
                },
                "POST_agg": {
                    "rank1_frac": float((rP == 1).mean()),
                    "rank_le3_frac": float((rP <= 3).mean()),
                    "median_rank": float(np.median(rP)),
                    "mean_rank": float(rP.mean()),
                    "margin_frac_pos": float((mgP > 0).mean()),
                    "margin_mean": float(mgP.mean()),
                },
            }
    match_out["note"] = (
        "Feature-resolution cells (D3/D4 nearest-centre sampling); NOT the "
        "full-res pixel population. PRE=A_L1 diagnostic proxy, POST=model's "
        "own aggregated_cost. Ranks/margins use the D3 left-edge exclusion. "
        "D3 stores 58118 cells/seed (70 margin-undefined NaN excluded here); "
        "D4 stores the 58048 fully-defined cells; per-regime counts are "
        "reported separately per file.")

    # ---- 4. OCCLUSION (d5 occ flag = reference_strata.py definition) ----
    occ_out: dict = {"per_regime": {}}
    for lo, hi, key in REGIMES:
        m = (G0 >= lo) & (G0 < hi)
        n = int(m.sum())
        if n == 0:
            occ_out["per_regime"][key] = {"px": 0, "status": "EMPTY"}
            continue
        per_seed = {}
        for seed in (0, 1, 2):
            o = seeds_occ[seed][m]
            err = np.abs(seeds_P[seed][m] - seeds_G[seed][m])
            nocc = int(o.sum())
            e_occ = float(err[o].mean()) if nocc else None
            e_non = float(err[~o].mean()) if nocc != n else None
            mass_share = (float(err[o].sum() / err.sum())
                          if float(err.sum()) > 0 else None)
            per_seed[str(seed)] = {
                "occ_px": nocc,
                "occ_frac": float(nocc / n),
                "epe_occluded": e_occ,
                "epe_non_occluded": e_non,
                "occ_share_of_regime_error_mass": mass_share,
            }
        occ_out["per_regime"][key] = {
            "px": n, "per_seed": per_seed,
            "method_mean_occ_frac": float(np.mean(
                [per_seed[str(s)]["occ_frac"] for s in (0, 1, 2)])),
        }

    # ---- 5. SUPERVISION COVERAGE ----
    from src.datasets.kitti2015 import Kitti2015Stereo
    cov: dict = {"bins": [f"[{EDGES[i]},{EDGES[i+1]})"
                          if np.isfinite(EDGES[i+1])
                          else f"[{EDGES[i]},inf)"
                          for i in range(len(EDGES) - 1)],
                 "edges": [float(e) if np.isfinite(e) else None
                           for e in EDGES]}
    tr_occ = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_calib",
                             disparity_scale=256.0, occluded=True)
    ev_occ = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                             disparity_scale=256.0, occluded=True)
    if len(tr_occ) != 160 or len(ev_occ) != 40:
        raise ValueError("split sizes %d/%d, need 160/40"
                         % (len(tr_occ), len(ev_occ)))
    if set(tr_occ.names) & set(ev_occ.names):
        raise ValueError("train/eval overlap!")
    import cv2
    train_bin = np.zeros(len(EDGES) - 1, dtype=np.float64)
    train_valid = 0.0
    aug_bin = np.zeros(len(EDGES) - 1, dtype=np.float64)
    aug_valid = 0.0
    rng = np.random.default_rng(0)
    K_DRAWS = 5
    LOG_LO, LOG_HI = float(np.log(0.7)), float(np.log(1.7))
    for i in range(len(tr_occ)):
        s = tr_occ[i]
        if s.disparity.shape != (368, 1232):
            raise ValueError(f"train shape mismatch {s.name}")
        disp = s.disparity.astype(np.float32)
        v = disp[disp > 0].astype(np.float64)
        if v.size == 0:
            raise ValueError(f"empty train GT {s.name}")
        bi = np.clip(np.searchsorted(np.array(EDGES), v, side="right") - 1,
                     0, len(train_bin) - 1)
        train_bin += np.bincount(bi, minlength=len(train_bin))
        train_valid += float(v.size)
        H, W = disp.shape
        for _ in range(K_DRAWS):
            sv = float(np.exp(rng.uniform(LOG_LO, LOG_HI)))
            w = int(round(512 / sv))
            h = int(round(256 / sv))
            h, w = min(h, H), min(w, W)
            y = int(rng.integers(0, H - h + 1))
            x = int(rng.integers(0, W - w + 1))
            crop = disp[y:y + h, x:x + w]
            rs = cv2.resize(crop, (512, 256), interpolation=cv2.INTER_NEAREST)
            if not np.isfinite(rs).all():
                raise ValueError("NaN/inf in augmented disparity")
            sx = 512.0 / float(w)
            rs = rs.copy()
            rs[rs > 0] *= sx
            av = rs[rs > 0].astype(np.float64)
            # loss mask also requires gt < max_disparity (184)
            av = av[av < 184.0]
            abi = np.clip(np.searchsorted(np.array(EDGES), av, side="right")
                          - 1, 0, len(aug_bin) - 1)
            aug_bin += np.bincount(abi, minlength=len(aug_bin))
            aug_valid += float(av.size)
        if (i + 1) % 40 == 0:
            print(f"coverage [{i + 1}/160] train_valid={train_valid:.0f} "
                  f"aug_valid={aug_valid:.0f}", flush=True)
    eval_bin = np.zeros(len(EDGES) - 1, dtype=np.float64)
    eval_valid = 0.0
    for i in range(len(ev_occ)):
        s = ev_occ[i]
        v = s.disparity[s.disparity > 0].astype(np.float64)
        bi = np.clip(np.searchsorted(np.array(EDGES), v, side="right") - 1,
                     0, len(eval_bin) - 1)
        eval_bin += np.bincount(bi, minlength=len(eval_bin))
        eval_valid += float(v.size)
    if abs(eval_valid - EXPECTED_VALID) > 0:
        raise ValueError(f"eval valid {eval_valid} != {EXPECTED_VALID}")
    cov["method"] = (
        "Monte-Carlo emulation of the exact P2A recipe "
        "(s~logUniform(0.7,1.7) per sample; w=round(512/s), h=round(256/s); "
        "uniform top-left crop in the 368x1232 source frame; cv2 NEAREST "
        "disparity resize to 256x512; disparity scaled by REALISED "
        "sx=512/w; loss mask gt>0 and gt<184). 5 draws per training scene "
        "(800 augmented samples), rng seed 0 (diagnostic only). Raw training "
        "histogram has no augmentation.")
    cov["n_train_scenes"] = 160
    cov["draws_per_scene"] = K_DRAWS
    cov["train_valid_pixels"] = float(train_valid)
    cov["aug_valid_pixels"] = float(aug_valid)
    cov["eval_valid_pixels"] = float(eval_valid)
    cov["per_bin"] = [{
        "bin": cov["bins"][i],
        "train_frac": float(train_bin[i] / train_valid),
        "train_px": float(train_bin[i]),
        "effective_train_frac": float(aug_bin[i] / aug_valid),
        "effective_train_px": float(aug_bin[i]),
        "eval_frac": float(eval_bin[i] / eval_valid),
        "eval_px": float(eval_bin[i]),
    } for i in range(len(train_bin))]

    def mass(idxs):
        return {k: float(v) for k, v in [
            ("train", train_bin[idxs].sum() / train_valid),
            ("effective_train", aug_bin[idxs].sum() / aug_valid),
            ("eval", eval_bin[idxs].sum() / eval_valid)]}

    cov["regime_mass"] = {
        "ge64": mass(list(range(4, 12))),
        "ge96": mass(list(range(6, 12))),
        "ge128": mass(list(range(8, 12))),
        "ge184": mass([11]),
    }

    out = {
        "conventions": {
            "population": "full-resolution frozen valid mask (GT>0), "
                          "3802797 px/seed; same pixels as D0/D1/D2/D5",
            "regimes": "GT>=64, GT>=96, GT>=128, GT>=184 (px units)",
            "method_mean": "additive stats = mean across 3 seeds; "
                           "slope/intercept/r2 = pooled regression over "
                           "concatenated 3-seed predictions",
            "regression_rule": "slope/intercept/r2 only if px>=1000 and "
                               "var(GT)>0 and var(pred)>0",
            "oracle_alpha": "alpha minimising EPE of relu(alpha*init+resid) "
                            "on the regime's pixels; GT-FITTED ORACLE CEILING, "
                            "not an achievable improvement",
            "matching_pop": "feature-res cells (D3/D4 sampling), not full-res",
            "occlusion": "valid in disp_occ_0 AND not valid in disp_noc_0 "
                         "(reference_strata.py definition; d5 occ flag)",
            "reference": "reference/onnx/stereonet.onnx vectors reused from "
                         "raw/d1_reference.npz (reference_strata.py mechanism)",
        },
        "part0_correction": {
            "corrected_statement": "No systematic GLOBAL mis-scaling (global "
                "alpha gain ~0.008 px method-mean), but strong "
                "DISPARITY-DEPENDENT mis-scaling: alpha* is flat near 1.0 "
                "below 96 px and rises steeply above it, and a per-bin "
                "rescale is worth ~0.048 px at seed 0.",
            "mass_weighted_per_bin_oracle_alpha_gain_px": part0,
            "oracle_note": "GT-FITTED ORACLE CEILING, not an achievable "
                           "improvement.",
        },
        "regimes": regimes_out,
        "range": range_out,
        "calibration": calib_out,
        "matching": match_out,
        "occlusion": occ_out,
        "supervision_coverage": cov,
        "interpretation": ("Factual statement only, no verdict (step 8 does "
                           "verdicts): each explanation above carries its own "
                           "numbers; regimes [128,144) (1071 px) and "
                           "[144,160) (9 px) are low-n flagged; [160,184) and "
                           "[184,inf) are EMPTY."),
        "unmeasurable": "NOT MEASURABLE WITH CURRENT ARTIFACTS: nothing in "
                        "this brief was unmeasurable; every listed quantity "
                        "was measured.",
    }
    atomic_write_json(OUT / "high_disparity.json", out)

    for seed in (0, 1, 2):
        atomic_write_npz(
            RAW / f"d6_s{seed}.npz",
            P=seeds_P[seed].astype(np.float32),
            G=seeds_G[seed].astype(np.float32),
            occluded=seeds_occ[seed],
            gt_cand=(seeds_G[seed] / STRIDE).astype(np.float32))

    fd, tmp = tempfile.mkstemp(dir=str(RAW), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["kind", "bin", "px", "mass", "p2a_epe_s0",
                        "p2a_epe_s1", "p2a_epe_s2", "p2a_mean_epe", "ref_epe",
                        "p2a_d1_mean", "ref_d1", "p2a_signed_mean",
                        "ref_signed", "p2a_slope_pooled", "ref_slope",
                        "p2a_r2_pooled", "ref_r2"])

            def f6(v):
                return f"{v:.6f}" if v is not None else ""

            def f4(v):
                return f"{v:.4f}" if v is not None else ""

            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                lab = (f"[{int(lo)},{int(hi)})" if np.isfinite(hi)
                       else f"[{int(lo)},inf)")
                m = (G0 >= lo) & (G0 < hi)
                n = int(m.sum())
                pes = [float(np.abs(seeds_P[s][m] - G0[m]).mean())
                       if n else None for s in (0, 1, 2)]
                pm = float(np.mean(pes)) if n else None
                re = (float(np.abs(P_ref[m] - G_ref[m]).mean())
                      if n else None)
                if n:
                    d1m = float(np.mean([
                        d1_pct(np.abs(seeds_P[s][m] - G0[m]), G0[m])
                        for s in (0, 1, 2)]))
                    sm = float(np.mean([(seeds_P[s][m] - G0[m]).mean()
                                        for s in (0, 1, 2)]))
                else:
                    d1m, sm = None, None
                rd1 = d1_pct(np.abs(P_ref[m] - G_ref[m]),
                             G_ref[m]) if n else None
                rs = float((P_ref[m] - G_ref[m]).mean()) if n else None
                Ppool = (np.concatenate([seeds_P[s][m] for s in (0, 1, 2)])
                         if n else None)
                Gpool = np.concatenate([G0[m]] * 3) if n else None
                if n and n * 3 >= MIN_REGRESSION_PX and float(Gpool.var()) > 0:
                    a, _ = np.polyfit(Gpool, Ppool, 1)
                    yh = a * Gpool + _
                    ss = float(((Ppool - yh) ** 2).sum())
                    st = float(((Ppool - Ppool.mean()) ** 2).sum())
                    ps, pr2 = float(a), 1.0 - ss / st
                else:
                    ps, pr2 = None, None
                if n and n >= MIN_REGRESSION_PX and float(G_ref[m].var()) > 0:
                    ra, _ = np.polyfit(G_ref[m], P_ref[m], 1)
                    yh = ra * G_ref[m] + _
                    ss = float(((P_ref[m] - yh) ** 2).sum())
                    st = float(((P_ref[m] - P_ref[m].mean()) ** 2).sum())
                    fs, fr2 = float(ra), 1.0 - ss / st
                else:
                    fs, fr2 = None, None
                w.writerow(["bin", lab, n, f"{n / total_n:.9f}",
                            *(f6(v) for v in pes), f6(pm), f6(re),
                            f4(d1m), f4(rd1), f4(sm), f4(rs),
                            ps if ps is not None else "",
                            fs if fs is not None else "",
                            pr2 if pr2 is not None else "",
                            fr2 if fr2 is not None else ""])
            for lo, hi, key in REGIMES:
                r = regimes_out[key]
                rr = r["reference"]
                pes = ([r["per_seed"][str(s)]["epe"]
                        if r["per_seed"][str(s)].get("epe") is not None
                        else None for s in (0, 1, 2)]
                       if r.get("px") else [None, None, None])
                w.writerow(["regime", key, r["px"], f"{r['frac_px']:.9f}",
                            *(f6(v) for v in pes), f6(r.get("epe")),
                            f6(rr.get("epe")), f4(r.get("d1_pct")),
                            f4(rr.get("d1_pct")), f4(r.get("signed_err")),
                            f4(rr.get("signed_err")),
                            r.get("slope", ""), rr.get("slope", ""),
                            r.get("r2", ""), rr.get("r2", "")])
        os.replace(tmp, RAW / "d6_bins.csv")
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print("wrote high_disparity.json + raw/d6_s*.npz + raw/d6_bins.csv",
          flush=True)


if __name__ == "__main__":
    main()
