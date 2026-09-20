"""STAGE-A D2: COST-DISTRIBUTION / READOUT (frozen model, read-only).

FROZEN-MODEL DIAGNOSTIC. No model change, no retraining, no tuning.
Recomputing the refinement module with a different disparity_initial input
alters NO weight and trains NOTHING; it reuses the UNCHANGED frozen
refinement (self.refinement) as a fixed function. Stated explicitly.

Architecture corrections to the brief (verified in code, not trusted):
- StereoNet.forward(left, right, return_stages=True) returns a TUPLE
  (disparity, stages-dict), NOT a bare dict. stages holds left_features,
  right_features, cost_volume, aggregated_cost, disparity_initial,
  refinement_residual, disparity_final. This script unpacks the tuple.
- P2A config: downsample_levels=3 (stride 8), num_disparities=24,
  cost_volume_shift='right', regression_normalize=True (same CFG as D0).
- Candidate k = 8*k full-res px. 24 candidates cover 0..184 px.
- Readout: upsample aggregated_cost (B,24,H/8,W/8) bilinearly
  (align_corners=True) to (B,24,368,1232), standardise across disparity axis
  (regression_normalize=True), softmax(-cost), expectation vs index grid.
  disparity_initial is in CANDIDATE UNITS (0..23), NOT pixels.
- disparity_final = relu(disparity_initial + refinement_residual), in pixels.

Scientific question: is correct disparity info present in the frozen cost
distribution but lost by the current readout? This step produces
measurements + a plain factual statement. No verdict (step 8 does verdicts).

Conventions:
- gt_cand = GT_px / 8.0. nearest GT candidate = clip(round(gt_cand), 0, 23).
  dist_frac = |gt_cand - round(gt_cand)| with unclipped round.
  in_range = 0 <= gt_cand <= 23.
- rank: candidates sorted by w_k descending; rank 1 = argmax IS the nearest
  GT candidate. Ties broken arbitrarily by argsort (stable); recorded.
- mass_nearest = w[nearest]. mass_neigh = sum of w_k over |k-gt_cand|<=1.
- Readout variants rebuild disparity_initial maps (valid pixels replaced,
  invalid pixels keep the model's own soft-argmin value so invalid regions
  are unchanged), push through the SAME frozen refinement with the SAME
  normalized left image as guidance, relu, score on the same valid pixels.

Outputs (atomic .tmp + rename):
  stage_a_diagnostics/cost_distribution.json
  stage_a_diagnostics/raw/d2_hist_s{0,1,2}.npz
  stage_a_diagnostics/raw/d2_bins.csv
Usage:
  python stage_a_diagnostics/scripts/d2_cost.py [--limit N] [--seeds 0,1,2]
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
EDGES = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 184, np.inf]
EXPECTED_SEED0_EPE = 1.4149795736625577
EXPECTED_VALID = 3802797
TOL_BASELINE_PX = 1e-6  # same code path + same weights: must match ~exactly


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


def edges_of(label: str):
    i = LABELS.index(label)
    return EDGES[i], EDGES[i + 1]


class Acc:
    """Float64 accumulators: pooled sums + per-D1-bin sums + histograms."""

    VARIANTS = ("baseline", "argmax", "topk2", "topk3", "topk4", "cont")

    def __init__(self) -> None:
        self.n = 0
        self.n96 = 0
        # distribution-stat sums (pooled)
        self.s = {k: 0.0 for k in (
            "entropy", "top1", "margin", "var", "argmax_idx", "soft",
            "abs_argmax_soft", "dist_frac", "mass_nearest", "mass_neigh")}
        self.s96 = {k: 0.0 for k in self.s}
        self.c_in_range = 0
        self.c_rank1 = 0
        self.c_in_range96 = 0
        self.c_rank196 = 0
        # readout abs-err sums (pooled)
        self.e = {v: 0.0 for v in self.VARIANTS}
        self.e96 = {v: 0.0 for v in self.VARIANTS}
        # per-bin
        nb = len(LABELS)
        self.bn = np.zeros(nb, dtype=np.int64)
        self.bs = {k: np.zeros(nb, dtype=np.float64) for k in self.s}
        self.bc_in_range = np.zeros(nb, dtype=np.int64)
        self.bc_rank1 = np.zeros(nb, dtype=np.int64)
        self.be = {v: np.zeros(nb, dtype=np.float64) for v in self.VARIANTS}
        # rank histogram 1..24, argmax histogram 0..23 (pooled)
        self.rank_hist = np.zeros(D, dtype=np.float64)
        self.argmax_hist = np.zeros(D, dtype=np.float64)
        # fine histograms (pooled)
        self.h = {
            "entropy": (np.zeros(100, dtype=np.float64), 0.0, float(np.log(D))),
            "top1": (np.zeros(100, dtype=np.float64), 0.0, 1.0),
            "margin": (np.zeros(100, dtype=np.float64), 0.0, 1.0),
            "var": (np.zeros(100, dtype=np.float64), 0.0, 144.0),
            "abs_argmax_soft": (np.zeros(100, dtype=np.float64), 0.0, 12.0),
            "mass_nearest": (np.zeros(100, dtype=np.float64), 0.0, 1.0),
            "mass_neigh": (np.zeros(100, dtype=np.float64), 0.0, 1.0),
            "dist_frac": (np.zeros(51, dtype=np.float64), 0.0, 0.5),
        }
        # per-image rows
        self.images: list[dict] = []

    def add_hist(self, key: str, x: np.ndarray) -> None:
        arr, lo, hi = self.h[key]
        idx = np.clip(((x - lo) / (hi - lo) * len(arr)).astype(np.int64),
                      0, len(arr) - 1)
        np.add.at(arr, idx, 1)


@torch.no_grad()
def process_seed(seed: int, device: str, limit: int | None) -> tuple[dict, Acc]:
    ck = REPO / "phase2" / "runs" / RUNS[seed] / "p2a_best.pth"
    if not ck.exists():
        raise FileNotFoundError(f"checkpoint missing: {ck}")
    blob = torch.load(str(ck), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "model" not in blob:
        raise ValueError(f"seed {seed}: checkpoint missing ['model'] key")
    model = StereoNet(StereoNetConfig(**CFG))
    model.load_state_dict(blob["model"], strict=True)
    model.eval().to(device)

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    if len(ds) != 40:
        raise ValueError(f"seed {seed}: split has {len(ds)} scenes, need 40")
    n_img = len(ds) if limit is None else min(limit, len(ds))

    # ---- pre-checks on scene 0 BEFORE the expensive loop ----
    s0 = ds[0]
    if s0.disparity.shape != (368, 1232):
        raise ValueError(f"shape mismatch scene {s0.name}: {s0.disparity.shape}")
    left0 = torch.from_numpy(normalize(s0.left)).to(device)
    right0 = torch.from_numpy(normalize(s0.right)).to(device)
    ret = model(left0, right0, return_stages=True)
    if not (isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict)):
        raise ValueError("forward(return_stages=True) did not return "
                         "(disparity, stages-dict); refusing to continue")
    _out0, st0 = ret
    for k in ("left_features", "right_features", "cost_volume",
              "aggregated_cost", "disparity_initial", "refinement_residual",
              "disparity_final"):
        if k not in st0:
            raise ValueError(f"stages dict missing key: {k}")
    agg0 = st0["aggregated_cost"]
    if tuple(agg0.shape) != (1, D, 368 // STRIDE, 1232 // STRIDE):
        raise ValueError(f"aggregated_cost shape {tuple(agg0.shape)} != "
                         f"(1,{D},{368 // STRIDE},{1232 // STRIDE})")
    dinit0 = st0["disparity_initial"]
    if tuple(dinit0.shape) != (1, 1, 368, 1232):
        raise ValueError(f"disparity_initial shape {tuple(dinit0.shape)}")
    if not (torch.isfinite(agg0).all() and torch.isfinite(dinit0).all()
            and torch.isfinite(st0["refinement_residual"]).all()
            and torch.isfinite(st0["disparity_final"]).all()):
        raise ValueError("NaN/inf in stages on scene 0")
    dmin, dmax = float(dinit0.min()), float(dinit0.max())
    if not (0.0 - 1e-3 <= dmin and dmax <= D - 1 + 1e-3 + 1.0):
        raise ValueError(f"disparity_initial out of candidate-unit range: "
                         f"[{dmin},{dmax}]")
    fmax = float(st0["disparity_final"].max())
    if not fmax > 20.0:
        raise ValueError(f"disparity_final max {fmax}: not in pixels?")
    # readout reconstruction check: upsample -> standardise -> softmax -> E[k]
    with torch.no_grad():
        cu = F.interpolate(agg0, size=(368, 1232), mode="bilinear",
                           align_corners=True)
        if not torch.isfinite(cu).all():
            raise ValueError("NaN/inf in upsampled cost")
        cus = (cu - cu.mean(dim=1, keepdim=True)) / (cu.std(dim=1, keepdim=True) + 1e-6)
        if not torch.isfinite(cus).all():
            raise ValueError("NaN/inf in standardised cost")
        w0 = torch.softmax(-cus, dim=1)
        grid = torch.arange(D, dtype=w0.dtype, device=w0.device).view(1, D, 1, 1)
        soft0 = (w0 * grid).sum(dim=1, keepdim=True)
        recon = float((soft0 - dinit0).abs().max())
    print(f"seed {seed} pre-checks OK: recon_maxdiff={recon:.2e} "
          f"dinit_range=[{dmin:.3f},{dmax:.3f}] final_max={fmax:.2f}", flush=True)
    if recon > 1e-4:
        raise ValueError(f"readout reconstruction mismatch {recon}: refusing loop")
    del ret, st0, agg0, dinit0, cu, cus, w0, soft0, grid

    acc = Acc()
    idx_grid = torch.arange(D, dtype=torch.float32, device=device)
    kk = idx_grid.view(D, 1)

    for i in range(n_img):
        s = ds[i]
        if s.disparity.shape != (368, 1232):
            raise ValueError(f"shape mismatch scene {s.name}")
        left = torch.from_numpy(normalize(s.left)).to(device)
        right = torch.from_numpy(normalize(s.right)).to(device)
        gt_np = s.disparity.astype(np.float64)
        valid_np = gt_np > 0
        nv = int(valid_np.sum())
        if nv == 0:
            raise ValueError(f"empty valid mask scene {s.name}")
        gt_valid_np = gt_np[valid_np]
        gt_cand_np = gt_valid_np / STRIDE

        out, st = model(left, right, return_stages=True)
        agg = st["aggregated_cost"]
        d_init = st["disparity_initial"]
        final = st["disparity_final"]
        if not (torch.isfinite(agg).all() and torch.isfinite(d_init).all()
                and torch.isfinite(final).all()):
            raise ValueError(f"NaN/inf in forward scene {s.name}")
        if tuple(agg.shape) != (1, D, 368 // STRIDE, 1232 // STRIDE):
            raise ValueError(f"shape mismatch agg scene {s.name}: {tuple(agg.shape)}")

        # frozen readout weights, exactly mirroring DisparityRegression
        cu = F.interpolate(agg, size=(368, 1232), mode="bilinear", align_corners=True)
        cus = (cu - cu.mean(dim=1, keepdim=True)) / (cu.std(dim=1, keepdim=True) + 1e-6)
        w = torch.softmax(-cus, dim=1)
        if not torch.isfinite(w).all():
            raise ValueError(f"NaN/inf in softmax weights scene {s.name}")
        soft = (w * idx_grid.view(1, D, 1, 1)).sum(dim=1, keepdim=True)

        valid_t = torch.from_numpy(valid_np).to(device)
        W = w[0, :, valid_t]            # (24, N) float32
        N = W.shape[1]
        if N != nv:
            raise ValueError("valid count mismatch")
        soft_v = soft[0, 0, valid_t].double()          # (N,)
        gt_t = torch.from_numpy(gt_valid_np).to(device)          # float64 px
        gtc = torch.from_numpy(gt_cand_np).to(device)            # float64 cand
        Wd = W.double()

        ent = -(Wd * torch.log(Wd + 1e-12)).sum(0)
        top2v, top2i = torch.topk(Wd, 2, dim=0)
        top1 = top2v[0]
        margin = top2v[0] - top2v[1]
        argmax = torch.argmax(Wd, dim=0)               # int64 (N,)
        am_f = argmax.double()
        abs_as = (am_f - soft_v).abs()
        var = (Wd * (kk.double() - soft_v.unsqueeze(0)) ** 2).sum(0)

        gt_round = torch.round(gtc)
        nearest = torch.clamp(gt_round.long(), 0, D - 1)
        dist_frac = (gtc - gt_round).abs()
        in_range = (gtc >= 0.0) & (gtc <= D - 1)

        order = torch.argsort(Wd, dim=0, descending=True)  # (24, N)
        rank = torch.empty(N, dtype=torch.int64, device=device)
        for r in range(D):
            rank[(order[r] == nearest)] = r + 1
        if int((rank < 1).sum()) != 0:
            raise ValueError("unassigned rank")
        mass_nearest = Wd[nearest, torch.arange(N, device=device)]
        neigh_mask = (kk.double() - gtc.unsqueeze(0)).abs() <= (1.0 + 1e-9)
        mass_neigh = (Wd * neigh_mask).sum(0)

        # ---- readout variants through the UNCHANGED frozen refinement ----
        base_map = d_init.detach().clone()  # (1,1,H,W), candidate units
        am_map = base_map.clone()
        am_map[0, 0, valid_t] = am_f.float()
        topk_idx = torch.topk(Wd, 4, dim=0).indices  # (4, N) int64
        alt_maps = {"argmax": am_map}
        for k in (2, 3, 4):
            cand = topk_idx[:k, :]                                # (k, N)
            diff = (cand.double() - gtc.unsqueeze(0)).abs()
            pick = cand[torch.argmin(diff, dim=0), torch.arange(N, device=device)]
            m = base_map.clone()
            m[0, 0, valid_t] = pick.float()
            alt_maps[f"topk{k}"] = m
        cont_map = base_map.clone()
        cont_map[0, 0, valid_t] = torch.clamp(gtc, 0.0, D - 1).float()
        alt_maps["cont"] = cont_map

        base_err = (final[0, 0, valid_t].double() - gt_t).abs()
        errs: dict[str, torch.Tensor] = {"baseline": base_err}
        for key in ("argmax", "topk2", "topk3", "topk4", "cont"):
            res_alt = model.refinement(alt_maps[key], left)
            if not torch.isfinite(res_alt).all():
                raise ValueError(f"NaN/inf refinement-alt {key} scene {s.name}")
            fin_alt = torch.relu(alt_maps[key] + res_alt)
            errs[key] = (fin_alt[0, 0, valid_t].double() - gt_t).abs()
            del res_alt, fin_alt
        for key in ("argmax", "topk2", "topk3", "topk4", "cont"):
            del alt_maps[key]
        del alt_maps, am_map, cont_map, base_map

        # ---- accumulate (float64) ----
        to_np = lambda t: t.detach().cpu().numpy().astype(np.float64)
        a_entropy, a_top1, a_margin = to_np(ent), to_np(top1), to_np(margin)
        a_var, a_soft = to_np(var), to_np(soft_v)
        a_am = argmax.detach().cpu().numpy().astype(np.int64)
        a_abs_as = to_np(abs_as)
        a_dist, a_mn, a_mw = to_np(dist_frac), to_np(mass_nearest), to_np(mass_neigh)
        a_in = in_range.detach().cpu().numpy()
        a_rank = rank.detach().cpu().numpy().astype(np.int64)
        e_np = {k: to_np(v) for k, v in errs.items()}
        if not all(np.isfinite(a).all() for a in
                   (a_entropy, a_top1, a_margin, a_var, a_soft, a_abs_as,
                    a_dist, a_mn, a_mw, *e_np.values())):
            raise ValueError(f"NaN/inf in accumulated stats scene {s.name}")
        if ((a_rank < 1) | (a_rank > D)).any():
            raise ValueError("rank out of [1,24]")

        b = np.searchsorted(EDGES, gt_valid_np, side="right") - 1
        b = np.clip(b, 0, len(LABELS) - 1)
        m96 = gt_valid_np >= 96.0

        acc.n += N
        for key, arr in (("entropy", a_entropy), ("top1", a_top1),
                         ("margin", a_margin), ("var", a_var),
                         ("argmax_idx", a_am.astype(np.float64)), ("soft", a_soft),
                         ("abs_argmax_soft", a_abs_as), ("dist_frac", a_dist),
                         ("mass_nearest", a_mn), ("mass_neigh", a_mw)):
            acc.s[key] += float(arr.sum())
            acc.s96[key] += float(arr[m96].sum())
            np.add.at(acc.bs[key], b, arr)
        acc.c_in_range += int(a_in.sum())
        acc.c_rank1 += int((a_rank == 1).sum())
        acc.c_in_range96 += int(a_in[m96].sum())
        acc.c_rank196 += int((a_rank[m96] == 1).sum())
        for v in Acc.VARIANTS:
            acc.e[v] += float(e_np[v].sum())
            acc.e96[v] += float(e_np[v][m96].sum())
            np.add.at(acc.be[v], b, e_np[v])
        np.add.at(acc.bn, b, 1)
        acc.n96 += int(m96.sum())
        rh, _ = np.histogram(a_rank, bins=np.arange(1, D + 2))
        ah, _ = np.histogram(a_am, bins=np.arange(0, D + 1))
        acc.rank_hist += rh.astype(np.float64)
        acc.argmax_hist += ah.astype(np.float64)
        acc.add_hist("entropy", a_entropy)
        acc.add_hist("top1", a_top1)
        acc.add_hist("margin", a_margin)
        acc.add_hist("var", a_var)
        acc.add_hist("abs_argmax_soft", a_abs_as)
        acc.add_hist("mass_nearest", a_mn)
        acc.add_hist("mass_neigh", a_mw)
        acc.add_hist("dist_frac", a_dist)
        acc.images.append({"scene": s.name, "valid": N, **{
            f"epe_{v}": float(e_np[v].mean()) for v in Acc.VARIANTS}})

        del out, st, agg, d_init, final, cu, cus, w, soft, W, Wd
        del ent, top1, margin, argmax, var, rank, mass_nearest, mass_neigh
        if device == "cuda" and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        print(f"seed {seed} [{i + 1}/{n_img}] {s.name} valid={N} "
              f"base_epe={float(e_np['baseline'].mean()):.4f} "
              f"argmax_epe={float(e_np['argmax'].mean()):.4f} "
              f"topk4_epe={float(e_np['topk4'].mean()):.4f} "
              f"cont_epe={float(e_np['cont'].mean()):.4f}", flush=True)

    if acc.n != EXPECTED_VALID and limit is None:
        raise ValueError(f"seed {seed}: valid pixels {acc.n} != {EXPECTED_VALID}")
    return model.config, acc


def summarize(seed: int, acc: Acc) -> dict:
    n = acc.n
    if n == 0:
        raise ValueError("zero valid pixels")
    mean = {k: acc.s[k] / n for k in acc.s}
    out: dict = {
        "valid_pixels": n,
        "distribution_pooled": {
            **{k: float(v) for k, v in mean.items()},
            "frac_gt_in_range_0_23": float(acc.c_in_range / n),
            "frac_gt_out_of_range": float(1 - acc.c_in_range / n),
            "frac_rank1": float(acc.c_rank1 / n),
        },
        "readout_epe_pooled": {v: float(acc.e[v] / n) for v in Acc.VARIANTS},
    }
    base = acc.e["baseline"] / n
    out["readout_delta_vs_baseline_pooled"] = {
        v: float(acc.e[v] / n - base) for v in Acc.VARIANTS if v != "baseline"}
    n96 = acc.n96
    out["gt_ge_96"] = {
        "valid_pixels": int(n96),
        "frac_of_valid": float(n96 / n),
        ** ({} if n96 == 0 else {
            "distribution": {
                **{k: float(acc.s96[k] / n96) for k in acc.s96},
                "frac_gt_in_range_0_23": float(acc.c_in_range96 / n96),
                "frac_gt_out_of_range": float(1 - acc.c_in_range96 / n96),
                "frac_rank1": float(acc.c_rank196 / n96),
            },
            "readout_epe": {v: float(acc.e96[v] / n96) for v in Acc.VARIANTS},
            "readout_delta_vs_baseline": {
                v: float(acc.e96[v] / n96 - acc.e96["baseline"] / n96)
                for v in Acc.VARIANTS if v != "baseline"},
        }),
    }
    bins = []
    for bi, lab in enumerate(LABELS):
        cnt = int(acc.bn[bi])
        if cnt == 0:
            bins.append({"bin": lab, "px": 0, "status": "EMPTY"})
            continue
        lo, hi = edges_of(lab)
        row = {
            "bin": lab, "px": cnt, "frac_px": float(cnt / n),
            "distribution": {
                **{k: float(acc.bs[k][bi] / cnt) for k in acc.bs},
                "frac_gt_in_range_0_23": float(acc.bc_in_range[bi] / cnt),
                "frac_gt_out_of_range": float(1 - acc.bc_in_range[bi] / cnt),
                "frac_rank1": float(acc.bc_rank1[bi] / cnt),
            },
            "readout_epe": {v: float(acc.be[v][bi] / cnt) for v in Acc.VARIANTS},
        }
        bb = acc.be["baseline"][bi] / cnt
        row["readout_delta_vs_baseline"] = {
            v: float(acc.be[v][bi] / cnt - bb)
            for v in Acc.VARIANTS if v != "baseline"}
        bins.append(row)
    out["per_bin"] = bins
    out["rank_hist_1_to_24"] = [int(x) for x in acc.rank_hist]
    out["argmax_hist_0_to_23"] = [int(x) for x in acc.argmax_hist]
    out["per_image_epe"] = acc.images
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"D2 device={device} seeds={seeds} limit={args.limit}", flush=True)

    results: dict = {}
    accs: dict[int, Acc] = {}
    for seed in seeds:
        _, acc = process_seed(seed, device, args.limit)
        accs[seed] = acc
        results[str(seed)] = summarize(seed, acc)
        # raw dump per seed (histograms + per-bin tables + per-image EPEs)
        hists = {f"hist_{k}": v[0] for k, v in acc.h.items()}
        atomic_write_npz(
            RAW / f"d2_hist_s{seed}.npz",
            rank_hist=acc.rank_hist, argmax_hist=acc.argmax_hist,
            bin_counts=acc.bn.astype(np.float64),
            **{f"binmean_{k}": np.divide(acc.bs[k], np.maximum(acc.bn, 1))
               for k in acc.bs},
            **{f"binepe_{v}": np.divide(acc.be[v], np.maximum(acc.bn, 1))
               for v in acc.be},
            **hists)

    # baseline reproduction gate (full runs only)
    if args.limit is None and 0 in accs:
        e0 = results["0"]["readout_epe_pooled"]["baseline"]
        ok = abs(e0 - EXPECTED_SEED0_EPE) < TOL_BASELINE_PX
        print(f"BASELINE seed0={e0:.10f} expected={EXPECTED_SEED0_EPE:.10f} "
              f"diff={e0 - EXPECTED_SEED0_EPE:.2e} -> "
              f"{'REPRODUCES' if ok else 'MISMATCH: STOP'}", flush=True)
        if not ok:
            raise SystemExit("D2 STOP: baseline does not reproduce step-1 number")
        for seed in seeds:
            z = np.load(OUT / "raw" / f"d0_preds_s{seed}.npz")
            P = z["P"].astype(np.float64)
            if P.size != accs[seed].n:
                raise ValueError(f"seed {seed}: D2 valid count != D0 dump")

    # method-mean headline over the run seeds
    if args.limit is None:
        mm = {v: float(np.mean([results[str(s)]["readout_epe_pooled"][v]
                                for s in seeds])) for v in Acc.VARIANTS}
        base = mm["baseline"]
        out = {
            "conventions": {
                "gt_cand": "GT_px / 8.0; candidate units 0..23 cover 0..184 px",
                "nearest_gt_candidate": "clip(round(gt_cand), 0, 23)",
                "rank": "candidates sorted by w_k desc; rank 1 = argmax IS nearest-GT",
                "mass_neigh": "sum w_k over |k - gt_cand| <= 1",
                "refinement_reuse": "recomputing refinement with a different "
                    "disparity_initial input is NOT a model change: no weight "
                    "altered, nothing trained; frozen self.refinement reused "
                    "with the same normalized left image as guidance",
                "forward_returns": "StereoNet.forward(return_stages=True) "
                    "returns TUPLE (disparity, stages-dict); brief said dict",
                "d1_bins": "per-bin rows reuse D1 edges "
                    "[0,16)...[160,184),[184,inf) from disparity_bins.json",
            },
            "config": {"downsample_levels": 3, "num_disparities": 24,
                       "cost_volume_shift": "right",
                       "regression_normalize": True, "stride_px": 8,
                       "candidate_range_px": [0, 184]},
            "seeds": results,
            "method_mean_readout_epe": mm,
            "method_mean_delta_vs_baseline": {
                v: mm[v] - base for v in Acc.VARIANTS if v != "baseline"},
            "interpretation": ("Factual statement only, no verdict: "
                "the pooled and per-bin numbers above show how much of the "
                "baseline EPE is removed when the frozen cost distribution's "
                "top-k candidates (oracle-picked per pixel) or the exact "
                "continuous gt_cand value are fed through the unchanged "
                "frozen refinement, versus the model's own soft-argmin and "
                "hard-argmax readouts on the same valid pixels."),
            "unmeasurable": "NOT MEASURABLE WITH CURRENT ARTIFACTS: none; "
                "all D2 quantities were measured.",
        }
        atomic_write_json(OUT / "cost_distribution.json", out)
        # flat per-bin CSV across seeds (atomic)
        import csv
        fd, tmp = tempfile.mkstemp(dir=str(RAW), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["seed", "bin", "px", "mass", "entropy", "top1",
                            "margin", "var", "abs_argmax_soft", "frac_rank1",
                            "frac_out_of_range", "mass_nearest", "mass_neigh",
                            "epe_baseline", "epe_argmax", "epe_topk2",
                            "epe_topk3", "epe_topk4", "epe_cont"])
                for s in seeds:
                    for row in results[str(s)]["per_bin"]:
                        if row.get("status") == "EMPTY":
                            w.writerow([s, row["bin"], 0, 0, *[("") for _ in range(16)]])
                            continue
                        d = row["distribution"]
                        e = row["readout_epe"]
                        w.writerow([s, row["bin"], row["px"],
                                    f"{row['frac_px']:.9f}",
                                    f"{d['entropy']:.6f}", f"{d['top1']:.6f}",
                                    f"{d['margin']:.6f}", f"{d['var']:.6f}",
                                    f"{d['abs_argmax_soft']:.6f}",
                                    f"{d['frac_rank1']:.6f}",
                                    f"{d['frac_gt_out_of_range']:.6f}",
                                    f"{d['mass_nearest']:.6f}",
                                    f"{d['mass_neigh']:.6f}",
                                    f"{e['baseline']:.6f}", f"{e['argmax']:.6f}",
                                    f"{e['topk2']:.6f}", f"{e['topk3']:.6f}",
                                    f"{e['topk4']:.6f}", f"{e['cont']:.6f}"])
            os.replace(tmp, RAW / "d2_bins.csv")
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        print("wrote cost_distribution.json + raw/d2_hist_s*.npz + raw/d2_bins.csv",
              flush=True)


if __name__ == "__main__":
    main()
