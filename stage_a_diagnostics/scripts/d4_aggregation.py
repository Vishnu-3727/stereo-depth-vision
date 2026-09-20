"""STAGE-A D4: AGGREGATION DIAGNOSTIC (frozen model, read-only).

FROZEN-MODEL DIAGNOSTIC. No weight altered, nothing trained, eval split
unchanged, no file outside stage_a_diagnostics/ modified. The model is NEVER
modified: forward(..., return_stages=True) is called on the UNCHANGED P2A
checkpoints and stages["cost_volume"] (PRE-aggregation, (1,32,24,H/8,W/8))
plus stages["aggregated_cost"] (POST-aggregation, (1,24,H/8,W/8)) are read
out. One image at a time (the PRE volume is ~27 MB float32); float64
accumulators; raises on NaN/inf.

Scientific question: do the four Conv3D aggregation blocks RESOLVE
correspondence ambiguity, or merely preserve/propagate it?

Conventions reused EXACTLY from D3 (scripts/d3_matching.py) so the two steps
are comparable:
  GT nearest sampling at feature-grid centres (y*8+4, x*8+4); keep sampled
  GT > 0; gt_cand = GT_px/8; gt_idx = clip(round(gt_cand), 0, 23).
  Left-edge rule: (cell, k) pairs with feature column x < k are excluded
  from EVERY statistic (matched against zeros); cells with x < gt_idx are
  dropped. Masks are geometry-only, hence IDENTICAL for PRE and POST.
  GT-neighbourhood = {k: |k-gt_idx| <= 1}; margin pool = valid
  non-neighbourhood; negatives = valid k with |k-gt_cand| > 2.0
  (continuous distance); overlap = fraction of pooled valid-negative scores
  BETTER (lower, cost scale) than the median positive; gap per cell =
  (cell_negmean - cell_GT)/std over the cell's valid candidates.
  Only cells with a valid non-neighbourhood candidate AND a valid negative
  are kept (same ok-mask as D3).

PRE vs POST objects (stated, not hidden):
  PRE scalar = A_L1 diagnostic proxy: mean over channels of |cost_volume|,
  exactly the D3 A_L1 reduction, recomputed from the frozen volume (a
  cross-check asserts it reproduces the D3 A_L1 rank-1 per seed).
  POST scalar = the model's OWN cost: stages["aggregated_cost"], whose
  readout applies softmax(-cost). Low = better for both.
  The PRE scalar is a DIAGNOSTIC PROXY and the POST one is the model's
  actual cost - they are NOT the same kind of object, so only RANKS and
  ORDERINGS are compared, never raw magnitudes.

Entropy: softmax(-cost) over the cell's VALID candidates only (the
left-edge rule excludes x<k pairs from EVERY statistic, entropy included).
PRE entropy uses softmax(-proxy) and is labelled proxy entropy throughout.

Regimes (per surviving cell):
  OCCLUSION: occluded-only = valid in disp_occ_0 AND not valid in
    disp_noc_0 at the same grid centre (cf.
    phase1/runs/arm_v_diag/reference_strata.py); else non-occluded.
  TEXTURE: full-res left-image grayscale gradient magnitude, averaged
    (mean) over each stride-8 cell; per-seed median split into
    textureless vs textured; threshold reported.
  EDGES: cell's gt_idx differs by >= 2 from any in-bounds 4-neighbour cell
    that itself has sampled GT > 0 (disparity discontinuity); count reported.
  DISPARITY: low = gt_cand < 8 (GT < 64 px), high = gt_cand >= 8.

Spatial consistency (POST only): dense POST argmax field (argmin over valid
candidates at every feature cell); per surviving cell, agreement with each
existing in-bounds 4-neighbour + mean |diff| in candidate units.

Outputs (atomic .tmp + rename):
  stage_a_diagnostics/aggregation_diagnostic.json
  stage_a_diagnostics/raw/d4_s{0,1,2}.npz
  stage_a_diagnostics/raw/d4_bins.csv
Usage:
  python stage_a_diagnostics/scripts/d4_aggregation.py [--limit N] [--seeds 0,1,2]
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

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

OUT = REPO / "stage_a_diagnostics"
RAW = OUT / "raw"
RUNS = {0: "p2a_scale_coverage", 1: "p2a_scale_coverage_s1", 2: "p2a_scale_coverage_s2"}
CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
D = 24
STRIDE = 8
EDGES = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 184, np.inf]
STAGES = ("pre", "post")

HONESTY = (
    "The pre-aggregation scalar is a DIAGNOSTIC PROXY (mean over channels of "
    "the frozen signed-difference volume, i.e. the D3 A_L1 reduction) and the "
    "post-aggregation scalar is the model's ACTUAL cost (the trained 3D "
    "aggregation output that the readout consumes). They are NOT the same kind "
    "of object. Only RANKS and ORDERINGS are compared, never raw magnitudes. "
    "A POST-OVER-PRE improvement establishes that aggregation RESOLVES "
    "ambiguity as measured; near-zero deltas mean it PRESERVES it. No "
    "long-range-aggregation claim from the literature is used as evidence, "
    "no conclusive-claim language is used, and no mechanism verdict is "
    "declared here (step 8 does verdicts)."
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


def cell_metrics(Sm: np.ndarray, V: np.ndarray, gt_c: np.ndarray,
                 gt_i: np.ndarray):
    """Rank / margin / pos / negmean / gap / entropy / argmax for one stage.

    Sm: (24, M) cost scale (lower = better), float64. V: (24, M) bool valid.
    Returns dict of (M,) float64 arrays with NaN where undefined.
    Entropy uses softmax(-cost) over VALID candidates only.
    """
    M = gt_c.shape[0]
    k_idx = np.arange(D)
    kk = k_idx[:, None]
    Sm_masked = np.where(V, Sm, np.inf)
    if not np.isfinite(np.where(V, Sm, 0.0)).all():
        raise ValueError("NaN/inf in candidate scores")
    order = np.argsort(Sm_masked, axis=0, kind="stable")
    rank = np.empty(M, dtype=np.float64)
    argmax = np.empty(M, dtype=np.float64)
    for j in range(M):
        rank[j] = float(np.nonzero(order[:, j] == gt_i[j])[0][0] + 1)
        argmax[j] = float(order[0, j])
    nonneigh = V & (np.abs(kk - gt_i[None, :]) > 1)
    has_nn = nonneigh.any(axis=0)
    best_other = np.where(nonneigh, Sm, np.inf).min(axis=0)
    gt_score = Sm[gt_i, np.arange(M)]
    margin = np.where(has_nn, best_other - gt_score, np.nan)
    negm = V & (np.abs(kk - gt_c[None, :]) > 2.0)
    has_neg = negm.any(axis=0)
    negcount = negm.sum(axis=0).astype(np.float64)
    negmean = np.where(has_neg,
                       np.where(negm, Sm, 0.0).sum(axis=0)
                       / np.where(has_neg, negcount, 1.0), np.nan)
    cnt = V.sum(axis=0).astype(np.float64)
    mu = np.where(V, Sm, 0.0).sum(axis=0) / cnt
    sd = np.sqrt(np.maximum(
        (np.where(V, (Sm - mu[None, :]) ** 2, 0.0).sum(axis=0) / cnt), 0.0))
    gap = np.where(has_neg & (sd > 0),
                   (negmean - gt_score) / np.where(sd > 0, sd, 1.0), np.nan)
    # valid-only softmax entropy (invalid pairs excluded, left-edge rule)
    mx = np.where(V, Sm, np.inf).min(axis=0)
    L = np.zeros_like(Sm)
    L[V] = (-(Sm - mx[None, :]))[V]  # <= 0 on valid entries: no exp overflow
    E = np.zeros_like(Sm)
    E[V] = np.exp(L[V])
    Z = E.sum(axis=0)
    if not (Z > 0).all():
        raise ValueError("non-positive softmax normaliser")
    P = E / Z[None, :]
    with np.errstate(divide="ignore", invalid="raise"):
        ent = -(np.where(V, P * np.log(np.where(P > 0, P, 1.0)), 0.0)).sum(axis=0)
    if not np.isfinite(ent).all():
        raise ValueError("NaN/inf in entropy")
    return {"rank": rank, "margin": margin, "pos": gt_score, "negmean": negmean,
            "gap": gap, "entropy": ent, "argmax": argmax,
            "has_nn": has_nn, "has_neg": has_neg, "negm": negm}


@torch.no_grad()
def process_seed(seed: int, device: str, limit: int | None) -> dict:
    ck = REPO / "phase2" / "runs" / RUNS[seed] / "p2a_best.pth"
    if not ck.exists():
        raise FileNotFoundError(f"checkpoint missing: {ck}")
    blob = torch.load(str(ck), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "model" not in blob:
        raise ValueError(f"seed {seed}: checkpoint missing ['model'] key")
    model = StereoNet(StereoNetConfig(**CFG))
    model.load_state_dict(blob["model"], strict=True)
    model.eval().to(device)

    ds_occ = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                             disparity_scale=256.0, occluded=True)
    ds_noc = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                             disparity_scale=256.0, occluded=False)
    if len(ds_occ) != 40 or len(ds_noc) != 40:
        raise ValueError("split must have 40 scenes")
    n_img = len(ds_occ) if limit is None else min(limit, len(ds_occ))
    Hf, Wf = 368 // STRIDE, 1232 // STRIDE

    # ---- architecture verification on scene 0 ----
    s0 = ds_occ[0]
    l0 = torch.from_numpy(normalize(s0.left)).to(device)
    r0 = torch.from_numpy(normalize(s0.right)).to(device)
    ret = model(l0, r0, return_stages=True)
    if not (isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict)):
        raise ValueError("forward(return_stages=True) did not return (disparity, stages)")
    st0 = ret[1]
    if tuple(st0["cost_volume"].shape) != (1, 32, D, Hf, Wf):
        raise ValueError(f"cost_volume shape {tuple(st0['cost_volume'].shape)}")
    if tuple(st0["aggregated_cost"].shape) != (1, D, Hf, Wf):
        raise ValueError(f"aggregated_cost shape {tuple(st0['aggregated_cost'].shape)}")
    if not (torch.isfinite(st0["cost_volume"]).all()
            and torch.isfinite(st0["aggregated_cost"]).all()):
        raise ValueError("NaN/inf in frozen volumes on scene 0")
    del ret, st0, l0, r0
    print(f"seed {seed} arch OK: cost_volume (1,32,{D},{Hf},{Wf}), "
          f"aggregated (1,{D},{Hf},{Wf})", flush=True)

    k_idx = np.arange(D)
    x_grid = np.tile(np.arange(Wf), (Hf, 1))
    acc: dict[str, list] = {k: [] for k in (
        "gt_px", "cell_x", "cell_y", "occ", "tex", "edge",
        "rank_pre", "rank_post", "margin_pre", "margin_post",
        "pos_pre", "pos_post", "negmean_pre", "negmean_post",
        "gap_pre", "gap_post", "ent_pre", "ent_post",
        "argmax_pre", "argmax_post",
        "nb_argmax_post", "nb_exists")}
    neg_pre_vals: list[np.ndarray] = []
    neg_pre_cell: list[np.ndarray] = []
    neg_post_vals: list[np.ndarray] = []
    neg_post_cell: list[np.ndarray] = []
    counts = {"n_gt_cells": 0, "n_dropped_leftedge": 0,
              "n_pairs_removed": 0, "n_nonneg_undef": 0, "n_neg_undef": 0,
              "n_fullres_valid_occ": 0, "n_fullres_occ_only": 0,
              "images": 0}
    base = 0  # cell offset for neg pools

    for i in range(n_img):
        s = ds_occ[i]
        sn = ds_noc[i]
        if sn.name != s.name:
            raise ValueError("occ/noc scene order mismatch")
        left = torch.from_numpy(normalize(s.left)).to(device)
        right = torch.from_numpy(normalize(s.right)).to(device)
        gt_full = s.disparity.astype(np.float64)
        gt_noc_full = sn.disparity.astype(np.float64)
        if not (np.isfinite(gt_full).all() and np.isfinite(gt_noc_full).all()):
            raise ValueError(f"NaN/inf in GT scene {s.name}")
        # texture: full-res grayscale gradient magnitude, mean per 8x8 cell
        grey = s.left.astype(np.float64).mean(axis=2)
        gy, gx = np.gradient(grey)
        mag = np.hypot(gy, gx)
        tex_grid = mag.reshape(Hf, STRIDE, Wf, STRIDE).mean(axis=(1, 3))
        if not np.isfinite(tex_grid).all():
            raise ValueError(f"NaN/inf in texture scene {s.name}")

        # occlusion, lifted to feature resolution by footprint-max: a
        # surviving cell counts as occluded if ANY full-res pixel in its
        # stride-8 footprint is occluded-only (valid in disp_occ_0 AND
        # invalid in disp_noc_0). Centre-sampling the sparse binary mask
        # instead yields zero occluded cells, so max-pooling the dense
        # full-res mask is the legitimate lift (averaging a sparse GT would
        # not be; max over a binary validity mask is).
        occ_only_full = (gt_full > 0) & (gt_noc_full <= 0)
        occ_grid = occ_only_full.reshape(Hf, STRIDE, Wf, STRIDE).max(axis=(1, 3))
        counts["n_fullres_valid_occ"] += int((gt_full > 0).sum())
        counts["n_fullres_occ_only"] += int(occ_only_full.sum())

        out, st = model(left, right, return_stages=True)
        V = st["cost_volume"]
        A = st["aggregated_cost"]
        if tuple(V.shape) != (1, 32, D, Hf, Wf) or tuple(A.shape) != (1, D, Hf, Wf):
            raise ValueError(f"shape drift scene {s.name}")
        if not (torch.isfinite(V).all() and torch.isfinite(A).all()):
            raise ValueError(f"NaN/inf in frozen volumes scene {s.name}")
        # PRE proxy A_L1: mean_c |.| per candidate, one k at a time (float32
        # torch, then float64 numpy; never holds the full float64 volume)
        pre = np.empty((D, Hf, Wf), dtype=np.float64)
        Vt = V[0]  # (32, 24, Hf, Wf)
        for k in range(D):
            pre[k] = Vt[:, k].abs().mean(dim=0).detach().cpu().numpy().astype(np.float64)
        post = A[0].detach().cpu().numpy().astype(np.float64)
        if not (np.isfinite(pre).all() and np.isfinite(post).all()):
            raise ValueError(f"NaN/inf in scalar costs scene {s.name}")
        if not float(pre.std(axis=0).max()) > 0:
            raise ValueError(f"degenerate PRE volume scene {s.name}")
        del out, st, V, A, Vt, left, right

        ys = (np.arange(Hf) * STRIDE + 4)[:, None].repeat(Wf, axis=1)
        xs = (np.arange(Wf) * STRIDE + 4)[None, :].repeat(Hf, axis=0)
        gt_feat = gt_full[ys, xs]
        noc_feat = gt_noc_full[ys, xs]
        gt_mask = gt_feat > 0
        counts["n_gt_cells"] += int(gt_mask.sum())
        gt_cand_full = gt_feat / STRIDE
        gt_idx_full = np.clip(np.round(gt_cand_full), 0, D - 1).astype(np.int64)

        x_b = x_grid[None, :, :].repeat(D, axis=0)
        invalid = (x_b < k_idx[:, None, None]) & gt_mask[None, :, :]
        counts["n_pairs_removed"] += int(invalid.sum())
        leftedge_ok = x_grid >= gt_idx_full
        keep = gt_mask & leftedge_ok
        counts["n_dropped_leftedge"] += int((gt_mask & ~leftedge_ok).sum())

        yy, xx = np.nonzero(keep)
        M = yy.size
        if M == 0:
            counts["images"] += 1
            continue
        gt_c = gt_cand_full[yy, xx]
        gt_i = gt_idx_full[yy, xx]
        Vmask = (xx[None, :] >= k_idx[:, None])
        res_pre = cell_metrics(pre[:, yy, xx], Vmask, gt_c, gt_i)
        res_post = cell_metrics(post[:, yy, xx], Vmask, gt_c, gt_i)
        ok = res_pre["has_nn"] & res_pre["has_neg"]
        # geometry-only mask: identical for PRE/POST by construction; verify
        if not bool(((res_post["has_nn"] == res_pre["has_nn"])
                     & (res_post["has_neg"] == res_pre["has_neg"])).all()):
            raise ValueError("PRE/POST validity masks differ (must be geometry-only)")
        counts["n_nonneg_undef"] += int((~res_pre["has_nn"]).sum())
        counts["n_neg_undef"] += int((~res_pre["has_neg"]).sum())
        keep_idx = np.nonzero(ok)[0]
        n_ok = keep_idx.size
        if n_ok == 0:
            counts["images"] += 1
            continue
        # occlusion from the footprint-max grid (see above)
        occ_flag = occ_grid[yy, xx].astype(np.float64)
        # edge: gt_idx differs by >= 2 from any in-bounds 4-neighbour with GT>0
        edge_flag = np.zeros(M, dtype=np.float64)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yn, xn = yy + dy, xx + dx
            inb = (yn >= 0) & (yn < Hf) & (xn >= 0) & (xn < Wf)
            has = np.zeros(M, dtype=bool)
            df = np.zeros(M, dtype=np.float64)
            ii = np.nonzero(inb)[0]
            has[ii] = gt_feat[yn[ii], xn[ii]] > 0
            df[ii] = np.abs(gt_idx_full[yn[ii], xn[ii]].astype(np.float64)
                            - gt_i[ii].astype(np.float64))
            edge_flag[(has & (df >= 2.0))] = 1.0
        # dense POST argmax field (valid-masked argmin everywhere)
        post_full_masked = np.where(x_grid[None, :, :] >= k_idx[:, None, None],
                                    post, np.inf)
        arg_grid = np.argmin(post_full_masked, axis=0).astype(np.int64)
        nb_arg = np.full((M, 4), -1, dtype=np.float64)
        nb_ex = np.zeros((M, 4), dtype=np.float64)
        for d, (dy, dx) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            yn, xn = yy + dy, xx + dx
            inb = (yn >= 0) & (yn < Hf) & (xn >= 0) & (xn < Wf)
            ii = np.nonzero(inb)[0]
            nb_ex[ii, d] = 1.0
            nb_arg[ii, d] = arg_grid[yn[ii], xn[ii]].astype(np.float64)

        j = keep_idx
        acc["gt_px"].append(gt_feat[yy[j], xx[j]])
        acc["cell_x"].append(xx[j].astype(np.float64))
        acc["cell_y"].append(yy[j].astype(np.float64))
        acc["occ"].append(occ_flag[j])
        acc["tex"].append(tex_grid[yy[j], xx[j]])
        acc["edge"].append(edge_flag[j])
        for tag, res in (("pre", res_pre), ("post", res_post)):
            acc[f"rank_{tag}"].append(res["rank"][j])
            acc[f"margin_{tag}"].append(res["margin"][j])
            acc[f"pos_{tag}"].append(res["pos"][j])
            acc[f"negmean_{tag}"].append(res["negmean"][j])
            acc[f"gap_{tag}"].append(res["gap"][j])
            acc[f"ent_{tag}"].append(res["entropy"][j])
            acc[f"argmax_{tag}"].append(res["argmax"][j])
        acc["nb_argmax_post"].append(nb_arg[j])
        acc["nb_exists"].append(nb_ex[j])
        # neg pools (valid negatives of ok cells only)
        for tag, res, vlist, clist in (
                ("pre", res_pre, neg_pre_vals, neg_pre_cell),
                ("post", res_post, neg_post_vals, neg_post_cell)):
            Sm = (pre if tag == "pre" else post)[:, yy, xx]
            negm = res["negm"]
            for t, jj in enumerate(j):
                nv = Sm[negm[:, jj], jj]
                vlist.append(nv.astype(np.float32))
                clist.append(np.full(nv.shape, base + t, dtype=np.int64))
        base += n_ok
        counts["images"] += 1
        if device == "cuda" and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        print(f"seed {seed} [{i + 1}/{n_img}] {s.name} cells={n_ok}", flush=True)

    out = {"counts": counts}
    for k, v in acc.items():
        out[k] = np.concatenate(v) if v else np.zeros(0, dtype=np.float64)
    out["neg_pre_vals"] = (np.concatenate(neg_pre_vals).astype(np.float32)
                           if neg_pre_vals else np.zeros(0, dtype=np.float32))
    out["neg_pre_cell"] = (np.concatenate(neg_pre_cell).astype(np.int64)
                           if neg_pre_cell else np.zeros(0, dtype=np.int64))
    out["neg_post_vals"] = (np.concatenate(neg_post_vals).astype(np.float32)
                            if neg_post_vals else np.zeros(0, dtype=np.float32))
    out["neg_post_cell"] = (np.concatenate(neg_post_cell).astype(np.int64)
                            if neg_post_cell else np.zeros(0, dtype=np.int64))
    if out["gt_px"].size == 0:
        raise ValueError(f"seed {seed}: zero surviving cells")
    return out


def subset_stats(gt, rank, margin, pos, negmean, gap, ent, neg_vals, neg_cell,
                 idx: np.ndarray) -> dict:
    """Pooled metrics for one stage over cell subset idx (NaN-safe)."""
    n = int(idx.size)
    if n == 0:
        return {"n_cells": 0, "status": "EMPTY"}
    r = rank[idx]
    if not np.isfinite(r).all():
        raise ValueError("NaN in rank (ok-mask guarantees defined)")
    med = _f(np.median(pos[idx]))
    nvm = neg_vals[np.isin(neg_cell, idx)]
    row = {
        "n_cells": n,
        "rank1_frac": _f((r == 1).mean()),
        "rankle3_frac": _f((r <= 3).mean()),
        "margin_mean": _mean_or_none(margin[idx]),
        "margin_frac_positive": _fracpos_or_none(margin[idx]),
        "overlap_frac_neg_better_than_median_pos":
            _f((nvm.astype(np.float64) < med).mean()) if nvm.size else None,
        "gap_mean": _mean_or_none(gap[idx]),
        "entropy_mean": _mean_or_none(ent[idx]),
    }
    if n < 1000:
        row["low_n"] = True
    return row


def summarize_seed(seed: int, d: dict) -> dict:
    gt = d["gt_px"]
    N = int(gt.size)
    bin_of = np.clip(np.searchsorted(EDGES, gt, side="right") - 1, 0, len(LABELS) - 1)
    tex_med = _f(np.median(d["tex"]))
    groups: dict[str, np.ndarray] = {
        "POOLED": np.arange(N),
        "GT>=96": np.nonzero(gt >= 96.0)[0],
        "OCC": np.nonzero(d["occ"] > 0.5)[0],
        "NONOCC": np.nonzero(d["occ"] <= 0.5)[0],
        "TEXLO": np.nonzero(d["tex"] <= tex_med)[0],
        "TEXHI": np.nonzero(d["tex"] > tex_med)[0],
        "EDGE": np.nonzero(d["edge"] > 0.5)[0],
        "NONEDGE": np.nonzero(d["edge"] <= 0.5)[0],
        "LOW": np.nonzero(gt / STRIDE < 8.0)[0],
        "HIGH": np.nonzero(gt / STRIDE >= 8.0)[0],
        "HIGH_OCC": np.nonzero((gt / STRIDE >= 8.0) & (d["occ"] > 0.5))[0],
        "HIGH_NONOCC": np.nonzero((gt / STRIDE >= 8.0) & (d["occ"] <= 0.5))[0],
        "GT96_OCC": np.nonzero((gt >= 96.0) & (d["occ"] > 0.5))[0],
        "GT96_NONOCC": np.nonzero((gt >= 96.0) & (d["occ"] <= 0.5))[0],
    }
    for bi, lab in enumerate(LABELS):
        groups[f"BIN {lab}"] = np.nonzero(bin_of == bi)[0]
    per: dict[str, dict] = {}
    for gname, idx in groups.items():
        row = {}
        for tag in STAGES:
            row[tag] = subset_stats(
                gt, d[f"rank_{tag}"], d[f"margin_{tag}"], d[f"pos_{tag}"],
                d[f"negmean_{tag}"], d[f"gap_{tag}"], d[f"ent_{tag}"],
                d[f"neg_{tag}_vals"], d[f"neg_{tag}_cell"], idx)
        # deltas POST - PRE where both defined
        delta = {}
        for key in ("rank1_frac", "rankle3_frac", "margin_mean",
                    "margin_frac_positive",
                    "overlap_frac_neg_better_than_median_pos", "gap_mean",
                    "entropy_mean"):
            a, b = row["pre"].get(key), row["post"].get(key)
            delta[key] = _f(b - a) if (a is not None and b is not None) else None
        row["delta_post_minus_pre"] = delta
        if gname not in ("POOLED",) and row["pre"].get("n_cells", 0) < 1000:
            row["low_n_warning"] = "fewer than 1000 cells: do not present alongside well-populated regimes without comment"
        per[gname] = row

    # ---- POST spatial consistency, pooled + per regime ----
    spat = {}
    am = d["argmax_post"].astype(np.int64)
    nb_a = d["nb_argmax_post"].astype(np.int64)  # (N,4), -1 if OOB
    nb_e = d["nb_exists"] > 0.5
    regmap = {"POOLED": np.arange(N), "OCC": groups["OCC"],
              "NONOCC": groups["NONOCC"], "TEXLO": groups["TEXLO"],
              "TEXHI": groups["TEXHI"], "EDGE": groups["EDGE"],
              "NONEDGE": groups["NONEDGE"], "LOW": groups["LOW"],
              "HIGH": groups["HIGH"], "HIGH_OCC": groups["HIGH_OCC"],
              "HIGH_NONOCC": groups["HIGH_NONOCC"],
              "GT96_OCC": groups["GT96_OCC"],
              "GT96_NONOCC": groups["GT96_NONOCC"]}
    for gname, idx in regmap.items():
        if idx.size == 0:
            spat[gname] = {"n_cells": 0, "status": "EMPTY"}
            continue
        ex = nb_e[idx]  # (n,4)
        ag = (nb_a[idx] == am[idx][:, None]) & ex
        n_pairs = int(ex.sum())
        ad = np.abs(nb_a[idx].astype(np.float64) - am[idx][:, None].astype(np.float64))
        srow: dict = {
            "n_cells": int(idx.size),
            "n_neighbour_pairs": n_pairs,
            "agree_frac_mean_over_existing_neighbours":
                _f(ag.sum() / n_pairs) if n_pairs else None,
            "mean_abs_diff_candidates":
                _f(ad[ex].mean()) if n_pairs else None,
        }
        perdir = {}
        for dname, dd in (("up", 0), ("down", 1), ("left", 2), ("right", 3)):
            e = ex[:, dd]
            perdir[dname] = {
                "n_pairs": int(e.sum()),
                "agree_frac": _f(ag[:, dd].sum() / e.sum()) if e.sum() else None,
            }
        srow["per_direction"] = perdir
        all4 = ex.all(axis=1)
        srow["full4_n_cells"] = int(all4.sum())
        srow["full4_all_agree_frac"] = (
            _f((ag[all4].all(axis=1)).mean()) if all4.sum() else None)
        if idx.size < 1000:
            srow["low_n"] = True
        spat[gname] = srow

    return {
        "n_defined_cells": N,
        "texture_median_threshold": tex_med,
        "n_edge_cells": int((d["edge"] > 0.5).sum()),
        "n_occluded_cells": int((d["occ"] > 0.5).sum()),
        "groups": per,
        "post_spatial_consistency": spat,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"D4 device={device} seeds={seeds} limit={args.limit}", flush=True)

    datas: dict[int, dict] = {}
    per_seed: dict[str, dict] = {}
    for seed in seeds:
        d = process_seed(seed, device, args.limit)
        datas[seed] = d
        if args.limit is None:
            per_seed[str(seed)] = summarize_seed(seed, d)
            keep = {k: v for k, v in d.items() if k != "counts"}
            keep["counts_json"] = json.dumps(d["counts"])
            atomic_write_npz(RAW / f"d4_s{seed}.npz", **keep)
        else:
            print(f"seed {seed} smoke: {d['gt_px'].size} cells", flush=True)

    if args.limit is not None:
        print("smoke run: no JSON written", flush=True)
        return

    # ---- PRE-vs-D3 cross-check: PRE proxy must reproduce D3 A_L1 rank-1 ----
    mj_path = OUT / "matching_diagnostic.json"
    if mj_path.exists():
        mj = json.loads(mj_path.read_text())
        for seed in seeds:
            d3r = mj["seeds"][str(seed)]["reps"]["A_L1"]["rank1_frac"]
            d4r = per_seed[str(seed)]["groups"]["POOLED"]["pre"]["rank1_frac"]
            diff = abs(d3r - d4r)
            print(f"seed {seed} PRE-vs-D3-A_L1 rank1: D3={d3r:.6f} "
                  f"D4={d4r:.6f} diff={diff:.2e}", flush=True)
            if not diff < 1e-9:
                raise ValueError(
                    f"PRE proxy does not reproduce D3 A_L1 (diff {diff:.2e}): "
                    "conventions diverged, refusing to write JSON")

    # ---- method-mean headlines across seeds ----
    mm: dict = {}
    glist = list(next(iter(per_seed.values()))["groups"].keys())
    for gname in glist:
        mm[gname] = {}
        for tag in STAGES:
            mm[gname][tag] = {}
            for key in ("rank1_frac", "rankle3_frac", "margin_mean",
                        "margin_frac_positive",
                        "overlap_frac_neg_better_than_median_pos", "gap_mean",
                        "entropy_mean"):
                vals = [per_seed[str(s)]["groups"][gname][tag].get(key)
                        for s in seeds]
                vals = [v for v in vals if v is not None]
                mm[gname][tag][key] = _f(float(np.mean(vals))) if vals else None
        mm[gname]["delta_post_minus_pre"] = {}
        for key in ("rank1_frac", "rankle3_frac", "margin_mean",
                    "margin_frac_positive",
                    "overlap_frac_neg_better_than_median_pos", "gap_mean",
                    "entropy_mean"):
            vals = [per_seed[str(s)]["groups"][gname]["delta_post_minus_pre"].get(key)
                    for s in seeds]
            vals = [v for v in vals if v is not None]
            mm[gname]["delta_post_minus_pre"][key] = (
                _f(float(np.mean(vals))) if vals else None)
        mm[gname]["n_cells_per_seed"] = [
            per_seed[str(s)]["groups"][gname]["pre"].get("n_cells", 0)
            for s in seeds]
    mm_spat = {}
    for gname in ("POOLED", "OCC", "NONOCC", "TEXLO", "TEXHI", "EDGE",
                  "NONEDGE", "LOW", "HIGH", "HIGH_OCC", "HIGH_NONOCC",
                  "GT96_OCC", "GT96_NONOCC"):
        rows = [per_seed[str(s)]["post_spatial_consistency"][gname] for s in seeds]
        okrows = [r for r in rows
                  if r.get("agree_frac_mean_over_existing_neighbours") is not None]
        def _mm(key):
            vals = [r.get(key) for r in okrows if r.get(key) is not None]
            return _f(float(np.mean(vals))) if vals else None
        mm_spat[gname] = {
            "agree_frac_mean_over_existing_neighbours":
                _mm("agree_frac_mean_over_existing_neighbours"),
            "mean_abs_diff_candidates": _mm("mean_abs_diff_candidates"),
            "full4_all_agree_frac": _mm("full4_all_agree_frac"),
            "n_cells_per_seed": [r["n_cells"] for r in rows],
        }
        if all(r.get("n_cells", 0) == 0 for r in rows):
            mm_spat[gname]["status"] = "EMPTY on all seeds"
        elif all(r.get("n_cells", 0) < 1000 for r in rows):
            mm_spat[gname]["low_n"] = True

    out = {
        "task": "D4 AGGREGATION DIAGNOSTIC (frozen model, no retraining)",
        "honesty_constraint": HONESTY,
        "conventions": {
            "geometry": ("P2A shift=right, 24 candidates, stride 8; "
                         "identical D3 conventions: nearest-centre GT "
                         "sampling, gt_cand=GT_px/8, "
                         "gt_idx=clip(round(gt_cand),0,23), left-edge x<k "
                         "exclusion from EVERY statistic, cells with "
                         "x<gt_idx dropped, ok-mask (valid non-neighbourhood "
                         "AND valid negative) identical for PRE/POST"),
            "pre_object": ("DIAGNOSTIC PROXY: mean_c |cost_volume| (D3 A_L1 "
                           "reduction), recomputed from the frozen volume; "
                           "rank-1 cross-checked against D3 A_L1 per seed"),
            "post_object": ("model's ACTUAL cost: stages['aggregated_cost']; "
                            "readout applies softmax(-cost)"),
            "comparison_rule": "RANKS and ORDERINGS only, never raw magnitudes",
            "margin": ("best valid non-neighbourhood cost minus GT cost; "
                       "positive = GT candidate preferred"),
            "overlap": ("fraction of pooled valid negatives scoring better "
                        "(lower) than the median positive"),
            "entropy": ("softmax(-cost) over VALID candidates only; PRE value "
                        "is a PROXY entropy, POST is the model's own"),
            "texture": ("full-res left grayscale gradient magnitude, mean "
                        "over each stride-8 cell; per-seed median split"),
            "edge": ("gt_idx differs by >=2 from any in-bounds 4-neighbour "
                     "with sampled GT>0"),
            "regimes": "low = gt_cand<8 (GT<64px); high = gt_cand>=8",
            "occlusion": ("full-res occluded-only mask (valid disp_occ_0 AND "
                          "invalid disp_noc_0, the reference_strata.py "
                          "definition) lifted to feature resolution by "
                          "footprint-max: a surviving cell is occluded if ANY "
                          "pixel in its stride-8 footprint is occluded-only. "
                          "Centre-sampling the sparse binary mask yields zero "
                          "occluded cells, so max-pooling the dense mask is "
                          "used and stated; HIGH_OCC / HIGH_NONOCC groups "
                          "test the occlusion confound on the D3 negative "
                          "high-disparity margin directly"),
            "spatial": ("POST dense argmax (valid-masked argmin) vs its "
                        "existing in-bounds 4-neighbours"),
            "low_n": "any regime with fewer than 1000 cells is flagged low-n",
            "method_mean": "headline = arithmetic mean across 3 seeds",
        },
        "prior_work": {
            "prevs_post_aggregation.py": (
                "phase1/runs/arm_v_diag/prevs_post_aggregation.py compares a "
                "raw L2-norm cost against the aggregated cost, but on the "
                "ARM-V checkpoints (not P2A), at upsampled full resolution "
                "with nearest interpolation, standardised softmax, no "
                "left-edge exclusion and no D1-bin / occlusion / texture "
                "regimes. It frames the same question but answers no part of "
                "this brief's measurements; none of its numbers are reused. "
                "D3's conventions are used instead so PRE here is exactly "
                "comparable to the D3 A_L1 representation."),
        },
        "counts_per_seed": {str(s): datas[s]["counts"] for s in seeds},
        "d3_crosscheck": ("PRE rank-1 reproduces D3 A_L1 rank-1 per seed to "
                          "<1e-9 (gate passed before this JSON was written)"),
        "seeds": per_seed,
        "method_mean": mm,
        "method_mean_post_spatial_consistency": mm_spat,
        "interpretation": ("Factual statement only, no verdict and no "
                           "conclusive-claim language; step 8 does mechanism "
                           "verdicts. POST rank-1 materially above PRE with more "
                           "positive margins and lower overlap = aggregation "
                           "RESOLVING ambiguity as measured; deltas near zero "
                           "= aggregation PRESERVING it. No literature claim "
                           "about long-range aggregation is admitted as "
                           "evidence; only these deltas count."),
        "unmeasurable": ("NOT MEASURABLE WITH CURRENT ARTIFACTS: nothing in "
                         "this brief was unmeasurable; every listed quantity "
                         "was measured."),
    }
    atomic_write_json(OUT / "aggregation_diagnostic.json", out)

    import csv

    def f6(v):
        return f"{v:.6f}" if v is not None else ""

    fd, tmp = tempfile.mkstemp(dir=str(RAW), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["seed", "stage", "group", "n_cells", "rank1_frac",
                        "rankle3_frac", "margin_mean", "margin_fracpos",
                        "overlap", "gap_mean", "entropy_mean"])
            for s in seeds:
                for tag in STAGES + ("delta",):
                    for gname in glist:
                        g = per_seed[str(s)]["groups"][gname]
                        src = g["delta_post_minus_pre"] if tag == "delta" else g[tag]
                        n = g["pre"].get("n_cells", 0)
                        if tag == "delta":
                            w.writerow([s, tag, gname, n, f6(src.get("rank1_frac")),
                                        f6(src.get("rankle3_frac")),
                                        f6(src.get("margin_mean")),
                                        f6(src.get("margin_frac_positive")),
                                        f6(src.get("overlap_frac_neg_better_than_median_pos")),
                                        f6(src.get("gap_mean")), f6(src.get("entropy_mean"))])
                        else:
                            w.writerow([s, tag, gname, src.get("n_cells", n),
                                        f6(src.get("rank1_frac")),
                                        f6(src.get("rankle3_frac")),
                                        f6(src.get("margin_mean")),
                                        f6(src.get("margin_frac_positive")),
                                        f6(src.get("overlap_frac_neg_better_than_median_pos")),
                                        f6(src.get("gap_mean")), f6(src.get("entropy_mean"))])
        os.replace(tmp, RAW / "d4_bins.csv")
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print("wrote aggregation_diagnostic.json + raw/d4_s*.npz + raw/d4_bins.csv",
          flush=True)


if __name__ == "__main__":
    main()
