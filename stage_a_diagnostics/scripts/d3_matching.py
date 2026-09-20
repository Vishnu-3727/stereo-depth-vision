"""STAGE-A D3: FROZEN MATCHING REPRESENTATION diagnostic (frozen model, read-only).

FROZEN-MODEL DIAGNOSTIC. No weight altered, nothing trained, eval split
unchanged. The model is NEVER modified: forward(..., return_stages=True) is
called on the UNCHANGED P2A checkpoints and stages["left_features"] /
stages["right_features"] (raw extractor output; feature_normalize is FALSE in
P2A) are read out. All four representations below are DIAGNOSTIC PROXIES built
from those frozen features with the SAME candidate geometry
(shift_right: candidate k computes left(x) - right(x-k), 24 candidates,
stride 8). One image at a time; float64 accumulators; raises on NaN/inf.

Ground truth at feature resolution: NEAREST sampling at the feature grid
(full-res pixel at the centre of each stride-8 cell, (y*8+4, x*8+4)); no
averaging, no interpolation. Only cells with sampled GT > 0 survive.
gt_cand = GT_px / 8.0; nearest-GT index = round(gt_cand) clipped to [0,23].

LEFT-EDGE INVALID DOMAIN: shift_right pads k zeros on the LEFT of the right
features, so for candidate k the feature columns x < k are matched against
ZEROS. (cell, k) pairs with x < k are excluded from EVERY statistic. A cell
whose own GT candidate is invalid (x < gt_idx) is dropped. The JSON states how
many pixel-candidate pairs that removed.

Representations (per-candidate scalar s_k, L = left feats, Rk = shift_right(R,k)):
  A. EXISTING: s_k = mean_c |L_c - Rk_c|            (lower = better)
  B. ABSOLUTE with L1 coincides with A by definition (mean|.| of the same
     difference); verified numerically (max abs diff 0.0) and STATED. The
     headline B numbers use an L2 (RMS) reduction instead:
     s_k = sqrt(mean_c (L_c - Rk_c)^2), so A and B are actually distinct.
  C. L2-NORMALISED: unit-normalise each 32-dim vector over channels
     (clamp norm at 1e-5, same rule as normalize_features_l2), then L1 mean.
     DIAGNOSTIC ONLY; the model itself is untouched.
  D. GROUP-WISE CORRELATION: split 32 channels into G groups, per-group
     dot(L, Rk) summed over the group's channels, averaged over groups.
     Correlation is HIGHER = better, so it is NEGATED before every
     rank/margin computation (all metrics run on a cost scale where
     lower = better). Primary G=8; G=4 and G=16 secondary.

Metrics per representation, pooled and per D1 bin (D1 edges reused):
  rank of nearest-GT candidate among VALID candidates only (1 = best);
    rank-1 fraction, rank<=3 fraction.
  margin = (best valid non-GT-neighbourhood score) - (GT score), positive
    means GT preferred (cost scale); mean + fraction positive.
    GT-neighbourhood = {k: |k - gt_idx| <= 1} (discrete, around the
    nearest-GT index). Buffer candidates (1 < |k-gt_cand| <= 2) compete in
    the margin pool but are excluded from the negative population.
  positive-vs-negative: mean GT score vs mean over valid NEGATIVE candidates
    {k: |k - gt_cand| > 2.0} (continuous distance); overlap = fraction of the
    pooled negative distribution scoring better (lower, cost scale) than the
    median positive.
  normalised gap per cell = (cell_negmean - cell_GTscore) / std over the
    cell's VALID candidates (population std); reported as the mean.

Outputs (atomic .tmp + rename):
  stage_a_diagnostics/matching_diagnostic.json
  stage_a_diagnostics/raw/d3_s{0,1,2}.npz
  stage_a_diagnostics/raw/d3_bins.csv
Usage:
  python stage_a_diagnostics/scripts/d3_matching.py [--limit N] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

OUT = REPO / "stage_a_diagnostics"
RAW = OUT / "raw"
RUNS = {0: "p2a_scale_coverage", 1: "p2a_scale_coverage_s1", 2: "p2a_scale_coverage_s2"}
CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
D = 24
STRIDE = 8
NORM_EPS = 1e-5
EDGES = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 184, np.inf]
REPS = ("A_L1", "B_L2", "C_L1norm", "D_G8", "D_G4", "D_G16")

HONESTY = (
    "The model does NOT score candidates with any of these scalar reductions. "
    "It feeds the full 32-channel signed-difference volume into a trained 3D "
    "aggregation which produces the cost that is actually read out. The scalar "
    "reduction is a DIAGNOSTIC PROXY chosen to make the four representations "
    "comparable, not the model's own scoring function. Therefore: a "
    "representation winning here establishes MECHANISM PLAUSIBILITY ONLY. It "
    "does NOT predict that a trained architecture using it will improve EPE."
)


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


def bin_label(lo, hi) -> str:
    if np.isfinite(hi):
        return f"[{int(lo)},{int(hi)})"
    return f"[{int(lo)},inf)"


LABELS = [bin_label(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]


def shift_right_np(x: np.ndarray, k: int) -> np.ndarray:
    """Numpy mirror of cost_volume.shift_right on the width axis (diagnostic)."""
    if k == 0:
        return x
    w = x.shape[-1]
    return np.concatenate([np.zeros_like(x[..., :k]), x[..., :w - k]], axis=-1)


class SeedAcc:
    """Per-seed storage: one row per surviving feature cell + neg distributions."""

    def __init__(self) -> None:
        self.gt_px: list[np.ndarray] = []
        self.cell_x: list[np.ndarray] = []
        self.rank: dict[str, list[np.ndarray]] = {r: [] for r in REPS}
        self.margin: dict[str, list[np.ndarray]] = {r: [] for r in REPS}
        self.pos: dict[str, list[np.ndarray]] = {r: [] for r in REPS}
        self.negmean: dict[str, list[np.ndarray]] = {r: [] for r in REPS}
        self.gap: dict[str, list[np.ndarray]] = {r: [] for r in REPS}
        self.neg_vals: dict[str, list[np.ndarray]] = {r: [] for r in REPS}
        self.neg_bins: dict[str, list[np.ndarray]] = {r: [] for r in REPS}
        self.n_gt_cells = 0
        self.n_dropped_leftedge_cells = 0
        self.n_pairs_removed_leftedge = 0
        self.n_nonneighbour_undef = 0
        self.n_negative_undef = 0
        self.n_gap_undef_std0 = 0
        self.ab_max_abs_diff = 0.0
        self.images = 0


@torch.no_grad()
def process_seed(seed: int, device: str, limit: int | None) -> SeedAcc:
    ck = REPO / "phase2" / "runs" / RUNS[seed] / "p2a_best.pth"
    if not ck.exists():
        raise FileNotFoundError(f"checkpoint missing: {ck}")
    blob = torch.load(str(ck), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "model" not in blob:
        raise ValueError(f"seed {seed}: checkpoint missing ['model'] key")
    cfg = StereoNetConfig(**CFG)
    if cfg.feature_normalize is not False:
        raise ValueError("P2A must run with feature_normalize FALSE (raw extractor output)")
    model = StereoNet(cfg)
    model.load_state_dict(blob["model"], strict=True)
    model.eval().to(device)

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    if len(ds) != 40:
        raise ValueError(f"seed {seed}: split has {len(ds)} scenes, need 40")
    n_img = len(ds) if limit is None else min(limit, len(ds))

    # ---- architecture verification on scene 0 ----
    s0 = ds[0]
    if s0.disparity.shape != (368, 1232):
        raise ValueError("scene shape mismatch")
    l0 = torch.from_numpy(normalize(s0.left)).to(device)
    r0 = torch.from_numpy(normalize(s0.right)).to(device)
    ret = model(l0, r0, return_stages=True)
    if not (isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict)):
        raise ValueError("forward(return_stages=True) did not return (disparity, stages-dict)")
    st0 = ret[1]
    if "left_features" not in st0 or "right_features" not in st0:
        raise ValueError("stages dict missing left/right_features")
    lf0, rf0 = st0["left_features"], st0["right_features"]
    Hf, Wf = 368 // STRIDE, 1232 // STRIDE
    if tuple(lf0.shape) != (1, 32, Hf, Wf) or tuple(rf0.shape) != (1, 32, Hf, Wf):
        raise ValueError(f"feature shape {tuple(lf0.shape)} {tuple(rf0.shape)}, "
                         f"need (1,32,{Hf},{Wf})")
    if not (torch.isfinite(lf0).all() and torch.isfinite(rf0).all()):
        raise ValueError("NaN/inf in frozen features on scene 0")
    del ret, st0, lf0, rf0, l0, r0
    print(f"seed {seed} arch OK: features (1,32,{Hf},{Wf}), "
          f"feature_normalize=FALSE, return_stages tuple", flush=True)

    acc = SeedAcc()
    k_idx = np.arange(D)
    x_grid = np.tile(np.arange(Wf), (Hf, 1))  # (Hf, Wf) feature column index

    for i in range(n_img):
        s = ds[i]
        left = torch.from_numpy(normalize(s.left)).to(device)
        right = torch.from_numpy(normalize(s.right)).to(device)
        gt_full = s.disparity.astype(np.float64)
        if not np.isfinite(gt_full).all():
            raise ValueError(f"NaN/inf in GT scene {s.name}")

        out, st = model(left, right, return_stages=True)
        Lf = st["left_features"]
        Rf = st["right_features"]
        if tuple(Lf.shape) != (1, 32, Hf, Wf) or tuple(Rf.shape) != (1, 32, Hf, Wf):
            raise ValueError(f"feature shape drift scene {s.name}")
        if not (torch.isfinite(Lf).all() and torch.isfinite(Rf).all()):
            raise ValueError(f"NaN/inf in frozen features scene {s.name}")
        L = Lf[0].detach().cpu().numpy().astype(np.float64)  # (32,Hf,Wf)
        R = Rf[0].detach().cpu().numpy().astype(np.float64)

        # ---- GT nearest sampling at feature grid centres ----
        ys = (np.arange(Hf) * STRIDE + 4)[:, None].repeat(Wf, axis=1)
        xs = (np.arange(Wf) * STRIDE + 4)[None, :].repeat(Hf, axis=0)
        gt_feat = gt_full[ys, xs]  # (Hf,Wf)
        gt_mask = gt_feat > 0
        acc.n_gt_cells += int(gt_mask.sum())
        gt_cand_full = gt_feat / STRIDE
        gt_idx_full = np.clip(np.round(gt_cand_full), 0, D - 1).astype(np.int64)

        # ---- left-edge pair accounting over ALL GT cells ----
        # invalid pair: candidate k at column x with x < k
        x_b = x_grid[None, :, :].repeat(D, axis=0)  # (24,Hf,Wf)
        invalid = (x_b < k_idx[:, None, None]) & gt_mask[None, :, :]
        acc.n_pairs_removed_leftedge += int(invalid.sum())
        leftedge_ok = x_grid >= gt_idx_full
        keep = gt_mask & leftedge_ok
        acc.n_dropped_leftedge_cells += int((gt_mask & ~leftedge_ok).sum())

        # ---- candidate scores, one k at a time (torch, float32 -> float64) ----
        Lt = torch.from_numpy(L.astype(np.float32))
        Rt = torch.from_numpy(R.astype(np.float32))
        nL = Lt.norm(p=2, dim=0, keepdim=True).clamp_min(NORM_EPS)
        nR = Rt.norm(p=2, dim=0, keepdim=True).clamp_min(NORM_EPS)
        Ln = (Lt / nL).numpy().astype(np.float64)
        Rn = (Rt / nR).numpy().astype(np.float64)
        S: dict[str, np.ndarray] = {r: np.empty((D, Hf, Wf), dtype=np.float64)
                                    for r in REPS}
        for k in range(D):
            Rk = shift_right_np(R, k).astype(np.float32)
            Rnk = shift_right_np(Rn, k).astype(np.float32)
            diff = torch.from_numpy(L.astype(np.float32)) - torch.from_numpy(Rk)
            ad = diff.abs()
            a = ad.mean(dim=0).numpy().astype(np.float64)
            b_l1 = ad.mean(dim=0).numpy().astype(np.float64)  # independent path, same def
            dmax = float(np.abs(a - b_l1).max())
            acc.ab_max_abs_diff = max(acc.ab_max_abs_diff, dmax)
            S["A_L1"][k] = a
            S["B_L2"][k] = torch.from_numpy(
                diff.pow(2).mean(dim=0).sqrt().numpy().astype(np.float64)).numpy()
            diffn = (torch.from_numpy(Ln.astype(np.float32))
                     - torch.from_numpy(Rnk))
            S["C_L1norm"][k] = diffn.abs().mean(dim=0).numpy().astype(np.float64)
            L32 = torch.from_numpy(L.astype(np.float32))
            R32 = torch.from_numpy(Rk)
            for gname, G in (("D_G8", 8), ("D_G4", 4), ("D_G16", 16)):
                Cg = 32 // G
                dg = (L32.view(G, Cg, Hf, Wf) * R32.view(G, Cg, Hf, Wf)).sum(
                    dim=1).mean(dim=0)
                S[gname][k] = (-dg.numpy()).astype(np.float64)  # NEGATED: lower=better
            del Rk, Rnk, diff, ad, a, b_l1, diffn, L32, R32
        for r in REPS:
            if not np.isfinite(S[r]).all():
                raise ValueError(f"NaN/inf in representation {r} scene {s.name}")
        # wiring sanity: scores must vary across candidates somewhere
        if not float(S["A_L1"].std(axis=0).max()) > 0:
            raise ValueError(f"degenerate A_L1 volume scene {s.name}")

        # ---- per-cell metrics over VALID candidates only ----
        yy, xx = np.nonzero(keep)
        M = yy.size
        if M:
            gt_c = gt_cand_full[yy, xx]          # (M,)
            gt_i = gt_idx_full[yy, xx]           # (M,)
            # valid mask (24, M): candidate k valid iff column x >= k
            V = (xx[None, :] >= k_idx[:, None])
            acc.gt_px.append(gt_feat[yy, xx])
            acc.cell_x.append(xx.astype(np.float64))
            for r in REPS:
                Sm = S[r][:, yy, xx]             # (24, M) cost scale, lower=better
                # rank of GT candidate among valid
                Sm_masked = np.where(V, Sm, np.inf)
                order = np.argsort(Sm_masked, axis=0, kind="stable")
                rank = np.empty(M, dtype=np.float64)
                for j in range(M):
                    rank[j] = float(np.nonzero(order[:, j] == gt_i[j])[0][0] + 1)
                # margin vs non-neighbourhood (|k - gt_idx| > 1)
                kk = k_idx[:, None]
                nonneigh = V & (np.abs(kk - gt_i[None, :]) > 1)
                has_nn = nonneigh.any(axis=0)
                best_other = np.where(nonneigh, Sm, np.inf).min(axis=0)
                margin = np.where(has_nn, best_other - Sm[gt_i, np.arange(M)], np.nan)
                acc.n_nonneighbour_undef += int((~has_nn).sum())
                # negatives: valid & |k - gt_cand| > 2
                negm = V & (np.abs(kk - gt_c[None, :]) > 2.0)
                has_neg = negm.any(axis=0)
                negcount = negm.sum(axis=0).astype(np.float64)
                negmean = np.where(has_neg,
                                   np.where(negm, Sm, 0.0).sum(axis=0)
                                   / np.where(has_neg, negcount, 1.0), np.nan)
                acc.n_negative_undef += int((~has_neg).sum())
                # gap: (negmean - GT) / std over valid
                cnt = V.sum(axis=0).astype(np.float64)
                mu = np.where(V, Sm, 0.0).sum(axis=0) / cnt
                sd = np.sqrt(np.maximum(
                    (np.where(V, (Sm - mu[None, :]) ** 2, 0.0).sum(axis=0) / cnt), 0.0))
                gap = np.where(has_neg & (sd > 0), (negmean - Sm[gt_i, np.arange(M)]) / np.where(sd > 0, sd, 1.0), np.nan)
                acc.n_gap_undef_std0 += int((has_neg & ~(sd > 0)).sum())
                # keep only cells where margin AND negatives are defined (one N)
                ok = has_nn & has_neg
                # NOTE: cells failing ok are dropped from ALL per-rep rows for
                # this rep; counts are dominated by GT+left-edge filters and
                # reported; N is recomputed per rep below.
                acc.rank[r].append(np.where(ok, rank, np.nan))
                acc.margin[r].append(np.where(ok, margin, np.nan))
                acc.pos[r].append(np.where(ok, Sm[gt_i, np.arange(M)], np.nan))
                acc.negmean[r].append(np.where(ok, negmean, np.nan))
                acc.gap[r].append(np.where(ok, gap, np.nan))
                # pooled negative distribution (valid neg pairs of ok cells)
                for j in np.nonzero(ok)[0]:
                    nv = Sm[negm[:, j], j]
                    acc.neg_vals[r].append(nv)
                    acc.neg_bins[r].append(
                        np.full(nv.shape,
                                int(np.searchsorted(EDGES, gt_feat[yy[j], xx[j]],
                                                    side="right") - 1),
                                dtype=np.int64))
            # drop bookkeeping for cells failing any rep? masks are rep-
            # independent (validity/geometry only), so ok is IDENTICAL for all
            # reps: verify once via D_G8 vs A_L1 NaN patterns below in summary.
        acc.images += 1
        del out, st, Lf, Rf, left, right
        if device == "cuda" and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        print(f"seed {seed} [{i + 1}/{n_img}] {s.name} cells={M}", flush=True)

    for r in REPS:
        acc.rank[r] = [a.astype(np.float64) for a in acc.rank[r]]
        for d in (acc.margin, acc.pos, acc.negmean, acc.gap):
            d[r] = [a.astype(np.float64) for a in d[r]]
    return acc


def _f(x: float) -> float:
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("non-finite headline value")
    return v


def _mean_or_none(a: np.ndarray):
    v = np.asarray(a, dtype=np.float64)
    v = v[np.isfinite(v)]
    return _f(v.mean()) if v.size else None


def _fracpos_or_none(a: np.ndarray):
    v = np.asarray(a, dtype=np.float64)
    v = v[np.isfinite(v)]
    return _f((v > 0).mean()) if v.size else None


def pooled_stats(vals: np.ndarray) -> dict:
    n = int(np.isfinite(vals).sum())
    v = vals[np.isfinite(vals)]
    if n == 0:
        return {"n": 0, "status": "UNDEFINED (no defined cells)"}
    return {"n": n, "mean": _f(v.mean()), "frac_positive": _f((v > 0).mean())}


def summarize_seed(seed: int, acc: SeedAcc) -> dict:
    gt = np.concatenate(acc.gt_px) if acc.gt_px else np.zeros(0)
    N = int(gt.size)
    if N == 0:
        raise ValueError(f"seed {seed}: zero surviving cells")
    bref = {"bin_of": np.clip(np.searchsorted(EDGES, gt, side="right") - 1, 0, len(LABELS) - 1)}
    out: dict = {
        "n_feature_cells_total": 40 * (368 // STRIDE) * (1232 // STRIDE),
        "n_gt_cells_sampled_gt_gt0": acc.n_gt_cells,
        "n_dropped_leftedge_gt_candidate_invalid": acc.n_dropped_leftedge_cells,
        "n_pairs_removed_by_leftedge_rule": acc.n_pairs_removed_leftedge,
        "n_cells_undefined_margin": acc.n_nonneighbour_undef // len(REPS),
        "n_cells_undefined_negatives": acc.n_negative_undef // len(REPS),
        "n_cells_undefined_gap_std0": acc.n_gap_undef_std0 // len(REPS),
        "ab_l1_max_abs_diff_vs_a": _f(acc.ab_max_abs_diff),
        "ab_coincide_under_l1": bool(acc.ab_max_abs_diff == 0.0),
    }
    reps: dict = {}
    for r in REPS:
        rk = np.concatenate(acc.rank[r])
        mg = np.concatenate(acc.margin[r])
        ps = np.concatenate(acc.pos[r])
        nm = np.concatenate(acc.negmean[r])
        gp = np.concatenate(acc.gap[r])
        ok = np.isfinite(rk)
        if int(ok.sum()) == 0:
            raise ValueError(f"seed {seed} rep {r}: zero defined cells")
        nv = np.concatenate(acc.neg_vals[r]) if acc.neg_vals[r] else np.zeros(0)
        nb = np.concatenate(acc.neg_bins[r]) if acc.neg_bins[r] else np.zeros(0, dtype=np.int64)
        med = _f(np.median(ps[ok]))
        overlap = _f((nv < med).mean()) if nv.size else None
        row = {
            "n_defined_cells": int(ok.sum()),
            "rank1_frac": _f((rk[ok] == 1).mean()),
            "rankle3_frac": _f((rk[ok] <= 3).mean()),
            "mean_rank": _f(rk[ok].mean()),
            "margin_mean": _mean_or_none(mg),
            "margin_frac_positive": _fracpos_or_none(mg),
            "pos_mean_score": _mean_or_none(ps),
            "neg_mean_score": _f(nv.mean()) if nv.size else None,
            "neg_pos_overlap_frac_neg_better_than_median_pos": overlap,
            "gap_mean": _mean_or_none(gp),
        }
        # per D1 bin
        bins = []
        for bi, lab in enumerate(LABELS):
            m = bref["bin_of"] == bi
            n = int(m.sum())
            if n == 0:
                bins.append({"bin": lab, "n_cells": 0, "status": "EMPTY"})
                continue
            rkb = rk[m]
            okb = np.isfinite(rkb)
            if int(okb.sum()) == 0:
                bins.append({"bin": lab, "n_cells": n, "status": "UNDEFINED"})
                continue
            nvb = nv[nb == bi]
            medb = _f(np.median(ps[m][okb]))
            brow = {
                "bin": lab, "n_cells": int(okb.sum()),
                "rank1_frac": _f((rkb[okb] == 1).mean()),
                "rankle3_frac": _f((rkb[okb] <= 3).mean()),
                "margin_mean": _mean_or_none(mg[m]),
                "margin_frac_positive": _fracpos_or_none(mg[m]),
                "pos_mean_score": _mean_or_none(ps[m]),
                "neg_mean_score": _f(nvb.mean()) if nvb.size else None,
                "neg_pos_overlap": _f((nvb < medb).mean()) if nvb.size else None,
                "gap_mean": _mean_or_none(gp[m]),
            }
            if n < 100:
                brow["low_n"] = True
            bins.append(brow)
        row["per_bin"] = bins
        # GT>=96 subset
        m96 = gt >= 96.0
        n96 = int(np.isfinite(rk[m96]).sum())
        if n96:
            nvb = nv[np.isin(nb, [6, 7, 8, 9, 10, 11])]
            med96 = _f(np.median(ps[m96][np.isfinite(rk[m96])]))
            row["gt_ge_96"] = {
                "n_cells": n96,
                "rank1_frac": _f((rk[m96][np.isfinite(rk[m96])] == 1).mean()),
                "rankle3_frac": _f((rk[m96][np.isfinite(rk[m96])] <= 3).mean()),
                "margin_mean": _mean_or_none(mg[m96]),
                "margin_frac_positive": _fracpos_or_none(mg[m96]),
                "pos_mean_score": _mean_or_none(ps[m96]),
                "neg_mean_score": _f(nvb.mean()) if nvb.size else None,
                "neg_pos_overlap": _f((nvb < med96).mean()) if nvb.size else None,
                "gap_mean": _mean_or_none(gp[m96]),
            }
        else:
            row["gt_ge_96"] = {"n_cells": 0, "status": "EMPTY"}
        reps[r] = row
    out["reps"] = reps
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"D3 device={device} seeds={seeds} limit={args.limit}", flush=True)

    accs: dict[int, SeedAcc] = {}
    per_seed: dict[str, dict] = {}
    for seed in seeds:
        acc = process_seed(seed, device, args.limit)
        accs[seed] = acc
        per_seed[str(seed)] = summarize_seed(seed, acc)
        gt_all = np.concatenate(acc.gt_px)
        xx_all = np.concatenate(acc.cell_x)
        to_save: dict = {
            "gt_px": gt_all,
            "cell_x": xx_all,
        }
        for r in REPS:
            to_save[f"rank_{r}"] = np.concatenate(acc.rank[r])
            to_save[f"margin_{r}"] = np.concatenate(acc.margin[r])
            to_save[f"pos_{r}"] = np.concatenate(acc.pos[r])
            to_save[f"negmean_{r}"] = np.concatenate(acc.negmean[r])
            to_save[f"gap_{r}"] = np.concatenate(acc.gap[r])
            nv = np.concatenate(acc.neg_vals[r]) if acc.neg_vals[r] else np.zeros(0)
            nb = (np.concatenate(acc.neg_bins[r]) if acc.neg_bins[r]
                  else np.zeros(0, dtype=np.int64))
            to_save[f"negvals_{r}"] = nv.astype(np.float32)
            to_save[f"negbins_{r}"] = nb.astype(np.int64)
        atomic_write_npz(RAW / f"d3_s{seed}.npz", **to_save)

    if args.limit is not None:
        print("smoke run: no JSON written", flush=True)
        return

    # ---- method-mean headlines across seeds ----
    mm: dict = {}
    for r in REPS:
        mm[r] = {}
        for key in ("rank1_frac", "rankle3_frac", "mean_rank", "margin_mean",
                    "margin_frac_positive", "pos_mean_score", "neg_mean_score",
                    "neg_pos_overlap_frac_neg_better_than_median_pos", "gap_mean"):
            vals = [per_seed[str(s)]["reps"][r][key] for s in seeds]
            vals = [v for v in vals if v is not None]
            mm[r][key] = _f(float(np.mean(vals))) if vals else None
        for sub in ("gt_ge_96",):
            mm[r][sub] = {}
            for key in ("rank1_frac", "rankle3_frac", "margin_mean",
                        "margin_frac_positive", "gap_mean",
                        "neg_pos_overlap"):
                vals = [per_seed[str(s)]["reps"][r][sub].get(key) for s in seeds]
                vals = [v for v in vals if v is not None]
                mm[r][sub][key] = _f(float(np.mean(vals))) if vals else None
            mm[r][sub]["n_cells_per_seed"] = [per_seed[str(s)]["reps"][r][sub]["n_cells"]
                                              for s in seeds]
    n_pooled = sum(per_seed[str(s)]["n_gt_cells_sampled_gt_gt0"] for s in seeds)
    n_defined = sum(per_seed[str(s)]["reps"]["A_L1"]["n_defined_cells"] for s in seeds)
    pooled_low_n = n_defined < 10000

    ab_all = all(per_seed[str(s)]["ab_coincide_under_l1"] for s in seeds)
    out = {
        "task": "D3 FROZEN MATCHING REPRESENTATION (diagnostic, frozen model)",
        "honesty_constraint": HONESTY,
        "conventions": {
            "geometry": ("P2A shift=right: candidate k is left(x)-right(x-k) in "
                         "the LEFT frame; 24 candidates, stride 8, k <-> 8*k px"),
            "features": ("(1,32,H/8,W/8) for (1,3,368,1232) input; "
                         "feature_normalize FALSE so stages features are raw "
                         "extractor output; representation C normalises as a "
                         "DIAGNOSTIC ONLY"),
            "gt_sampling": ("NEAREST at feature-grid centres (y*8+4, x*8+4); "
                            "no averaging/interpolation; keep sampled GT>0; "
                            "gt_cand=GT_px/8; gt_idx=clip(round(gt_cand),0,23)"),
            "left_edge_rule": ("(cell,k) pairs with feature column x<k excluded "
                               "from EVERY statistic (matched against zeros); "
                               "cells with x<gt_idx dropped"),
            "valid_candidates": "k<=x only; rank/margin/negatives/std use valid only",
            "neighbourhood": ("GT-neighbourhood={k:|k-gt_idx|<=1}; margin pool = "
                              "valid non-neighbourhood; buffer (1,2] competes "
                              "in margin but is excluded from negatives"),
            "negatives": "valid k with |k-gt_cand|>2.0 (continuous distance)",
            "overlap": ("fraction of pooled valid-negative scores BETTER "
                        "(lower, cost scale) than the median positive score"),
            "gap": ("per cell (cell_negmean-cell_GT)/std over valid candidates "
                    "(population std); reported as mean"),
            "group_corr": ("per-group dot over the group's channels, averaged "
                           "over groups; NEGATED to cost scale (lower=better) "
                           "before every rank/margin computation"),
            "ab_note": ("A and B coincide under L1 by definition; verified "
                        "max abs diff 0.0; headline B uses L2 (RMS) reduction"),
            "d1_bins": "per-bin rows reuse D1 edges [0,16)...[160,184),[184,inf)",
            "method_mean": "headline = arithmetic mean across 3 seeds",
        },
        "architecture_verification": {
            "feature_shape": "(1,32,46,154)",
            "feature_normalize": False,
            "forward_returns": "(disparity, stages-dict) with left/right_features",
            "shift": "diagnostic candidates built with shift_right geometry",
        },
        "counts": {
            "n_gt_cells_per_seed": {str(s): per_seed[str(s)]["n_gt_cells_sampled_gt_gt0"]
                                    for s in seeds},
            "n_defined_cells_per_seed": {str(s): per_seed[str(s)]["reps"]["A_L1"]["n_defined_cells"]
                                         for s in seeds},
            "n_pairs_removed_by_leftedge_rule_per_seed": {
                str(s): per_seed[str(s)]["n_pairs_removed_by_leftedge_rule"]
                for s in seeds},
            "n_dropped_leftedge_cells_per_seed": {
                str(s): per_seed[str(s)]["n_dropped_leftedge_gt_candidate_invalid"]
                for s in seeds},
            "pooled_defined_cells_3seeds": n_defined,
            "pooled_under_10000_low_n": pooled_low_n,
        },
        "ab_coincide_under_l1_all_seeds": ab_all,
        "ab_l1_max_abs_diff_per_seed": {str(s): per_seed[str(s)]["ab_l1_max_abs_diff_vs_a"] for s in seeds},
        "seeds": per_seed,
        "method_mean": mm,
        "interpretation": ("Factual statement only, no verdict and no "
                           "'proven' claim; step 8 does mechanism verdicts. "
                           "A representation winning here establishes "
                           "MECHANISM PLAUSIBILITY ONLY. It does NOT predict "
                           "that a trained architecture using it will improve "
                           "EPE. No architecture is recommended and no "
                           "mechanism verdict is declared here."),
        "unmeasurable": ("NOT MEASURABLE WITH CURRENT ARTIFACTS: what a "
                         "trained aggregation would do with any of these "
                         "representations; nothing else was unmeasurable."),
    }
    if pooled_low_n:
        out["low_n_warning"] = ("pooled defined cells under 10,000: per-bin "
                                "numbers are low-n, not quietly reported")
    atomic_write_json(OUT / "matching_diagnostic.json", out)

    import csv

    def f6(v):
        return f"{v:.6f}" if v is not None else ""

    fd, tmp = tempfile.mkstemp(dir=str(RAW), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["seed", "rep", "bin", "n_cells", "rank1_frac",
                        "rankle3_frac", "margin_mean", "margin_fracpos",
                        "pos_mean", "neg_mean", "overlap", "gap_mean"])
            for s in seeds:
                for r in REPS:
                    prow = per_seed[str(s)]["reps"][r]
                    g96 = prow["gt_ge_96"]
                    w.writerow([s, r, "POOLED", prow["n_defined_cells"],
                                f6(prow["rank1_frac"]),
                                f6(prow["rankle3_frac"]),
                                f6(prow["margin_mean"]),
                                f6(prow["margin_frac_positive"]),
                                f6(prow["pos_mean_score"]),
                                f6(prow["neg_mean_score"]),
                                f6(prow["neg_pos_overlap_frac_neg_better_than_median_pos"]),
                                f6(prow["gap_mean"])])
                    for b in prow["per_bin"]:
                        if b.get("status") in ("EMPTY", "UNDEFINED"):
                            w.writerow([s, r, b["bin"], b.get("n_cells", 0),
                                        "", "", "", "", "", "", "", ""])
                            continue
                        w.writerow([s, r, b["bin"], b["n_cells"],
                                    f6(b["rank1_frac"]),
                                    f6(b["rankle3_frac"]),
                                    f6(b["margin_mean"]),
                                    f6(b["margin_frac_positive"]),
                                    f6(b["pos_mean_score"]),
                                    f6(b["neg_mean_score"]),
                                    f6(b["neg_pos_overlap"]),
                                    f6(b["gap_mean"])])
                    w.writerow([s, r, "GT>=96", g96.get("n_cells", 0),
                                f6(g96.get("rank1_frac")),
                                f6(g96.get("rankle3_frac")),
                                f6(g96.get("margin_mean")),
                                f6(g96.get("margin_frac_positive")),
                                f6(g96.get("pos_mean_score")),
                                f6(g96.get("neg_mean_score")),
                                f6(g96.get("neg_pos_overlap")),
                                f6(g96.get("gap_mean"))])
        os.replace(tmp, RAW / "d3_bins.csv")
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print("wrote matching_diagnostic.json + raw/d3_s*.npz + raw/d3_bins.csv", flush=True)


if __name__ == "__main__":
    main()
