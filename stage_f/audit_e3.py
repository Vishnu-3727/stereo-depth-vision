"""Stage-F F1: E3 zero-training error audit (inference only, trains nothing).

Subjects: E3 FINAL seeds 0/1/2 (primary) + BEST seeds 0/1/2 (secondary,
monitor-selected-on-160..169). Checkpoints are sha256-asserted against
stage_f/STAGE_F_READINESS.md before any measurement.

GATE (full runs): each final's pooled contract EPE must equal the
e3_verdict.json per-seed final within 1e-6, valid_pixels must be 3802797 and
refuse_unless_contract must pass; else the JSON records status STOP and the
process exits 1.

Reuse (imported, never reimplemented):
  src.models.stereonet (StereoNet/StereoNetConfig), eval_tier2.CFG/score
  (ev.REPO pointed at the repo root exactly as e3_measure.py does),
  frozen_eval.{pooled_metrics, refuse_unless_contract, sha256_file},
  bottleneck_diag.{SPACING, NDISP, MAXD, gt_discontinuity},
  src.evaluation.metrics.disparity_metrics,
  d5_refinement.{core_stats, oracle_alpha, oracle_beta, sign_only_scaled_epe},
  d7_spatial.{CorrAcc, edge_distance_map, texture_magnitude},
  Kitti2015Stereo (incl. occluded=False for noc), pad_and_crop for obj_map.
New probe code (warp sampler, top-k/dominant-mode readouts, bimodality rate,
ordinal-rank Spearman, reservoir sampler, bad0.5) is documented under
out["methods"]; the JSON "reimplemented" list records functions copied because
they could not be imported (empty: every reuse above imports cleanly).

Memory: one scene at a time, float32 inference, GPU tensors freed per scene;
per-checkpoint concat vectors are float32 (init_cand, resid, final, gt, tex)
plus a <=2M-pair warp reservoir and GT<64 (err, bimodal) pairs.

Usage (from repo root):
  <host-python> stage_f/audit_e3.py [--limit N] [--device cpu|cuda]
      [--subjects final|all]
Writes stage_f/audit/audit_e3.json + stage_f/audit/per_scene.csv (atomic).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F

from phase1.harness import bottleneck_diag as bdiag
from phase1.harness.frozen_eval import (
    pooled_metrics,
    refuse_unless_contract,
    sha256_file,
)
from src.datasets.kitti2015 import Kitti2015Stereo, normalize, pad_and_crop
from src.evaluation.metrics import disparity_metrics
from src.models.stereonet import StereoNet, StereoNetConfig

DATA_ROOT = REPO / "data" / "kitti2015"
OUT_DIR = REPO / "stage_f" / "audit"
EV_PATH = REPO / "stage_e_recipe" / "kaggle" / "bundle_e3" / "scripts" / "eval_tier2.py"
D5_PATH = REPO / "stage_a_diagnostics" / "scripts" / "d5_refinement.py"
D7_PATH = REPO / "stage_a_diagnostics" / "scripts" / "d7_spatial.py"
VERDICT_PATH = REPO / "stage_e_recipe" / "e3_verdict.json"

EXPECTED_CFG = dict(
    downsample_levels=3,
    num_disparities=24,
    cost_volume_shift="right",
    regression_normalize=True,
)
EXPECTED_PARAMS = 397954
EXPECTED_VALID = 3802797
GATE_TOL = 1e-6
SCORE_XCHECK_TOL = 1e-9
BIN_CONTRIB_TOL = 1e-9
BINS16 = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
EDGE_STRATA = [0, 1, 2, 4, 8, 16, np.inf]
RESERVOIR_CAP = 2_000_000
UNCH_THRESH = 1e-6

EXPECTED_SHA = {
    ("seed0", "best"): "7d3c2109c11339b6fb30cc33f5bd042667740f3bc7345768f4e4b04704836209",
    ("seed0", "final"): "82e58bc441a4382ec449479fe26bf479f6b45e9ace4d79b530abeeb40ea79c6d",
    ("seed1", "best"): "133c19c41585445da9aabf18919e62f5805b5f8abcf80232808d26d33aa293b0",
    ("seed1", "final"): "7d26844d511b646c96189e44e77f8da15ecaa0818cb1e1edd2d76420c017415e",
    ("seed2", "best"): "1ba8ca09bfabf2c689b6ec7fbbbae1aa70367d33562b1aad06521ba71904cbcf",
    ("seed2", "final"): "228487aff2f80bd3514f228f6bc095f18e97b845d7c4fd92da360fc43d229169",
}

CANDIDATE_DATA_DIRS = [
    "data/sceneflow",
    "data/kitti2012",
    "data/kitti2015_extra",
    "data/middlebury",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = load_module("stage_f_eval_tier2", EV_PATH)
ev.REPO = REPO  # same retarget e3_measure.py applies for local runs
d5 = load_module("stage_f_d5_refinement", D5_PATH)
d7 = load_module("stage_f_d7_spatial", D7_PATH)

if dict(ev.CFG) != EXPECTED_CFG:
    raise SystemExit("ABORT: eval_tier2.CFG != E3 CFG: %r" % (ev.CFG,))


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


def git_head() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True, timeout=15,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def ckpt_path(seed: int, tag: str) -> Path:
    return (REPO / "stage_e_recipe" / "kaggle" / "e3_output" / f"seed{seed}"
            / f"e3_seed{seed}_{tag}.pth")


def load_model(seed: int, tag: str, device: str):
    ck = ckpt_path(seed, tag)
    sha = sha256_file(ck)
    exp = EXPECTED_SHA[(f"seed{seed}", tag)]
    if sha != exp:
        raise ValueError(
            f"sha256 mismatch seed{seed}/{tag}: got {sha} want {exp}")
    blob = torch.load(str(ck), map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model = StereoNet(StereoNetConfig(**ev.CFG))
    model.load_state_dict(state, strict=True)
    params = sum(p.numel() for p in model.parameters())
    if params != EXPECTED_PARAMS:
        raise ValueError(f"param count {params} != {EXPECTED_PARAMS}")
    model.eval().to(device)
    return model, sha


class Reservoir:
    """Vectorised reservoir sampler for (x, y) float32 pairs."""

    def __init__(self, cap: int, rng: np.random.Generator):
        self.cap = cap
        self.rng = rng
        self.xs: list[np.ndarray] = []
        self.ys: list[np.ndarray] = []
        self.n_stored = 0
        self.n_seen = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).ravel()
        y = np.asarray(y, dtype=np.float32).ravel()
        n = x.size
        if n == 0:
            return
        t = self.n_seen
        if t + n <= self.cap:
            self.xs.append(x)
            self.ys.append(y)
            self.n_stored += n
        else:
            r = self.rng.integers(0, t + n, size=n)
            hit = r < self.cap
            if np.any(hit):
                buf_x = self._buf_x()
                buf_y = self._buf_y()
                buf_x[r[hit]] = x[hit]
                buf_y[r[hit]] = y[hit]
                self.xs = [buf_x]
                self.ys = [buf_y]
                self.n_stored = self.cap
        self.n_seen += n

    def _buf_x(self) -> np.ndarray:
        if self.xs:
            b = np.concatenate(self.xs)
            if b.size < self.cap:
                pad = np.zeros(self.cap - b.size, dtype=np.float32)
                b = np.concatenate([b, pad])
            return b[:self.cap].copy()
        return np.zeros(self.cap, dtype=np.float32)

    def _buf_y(self) -> np.ndarray:
        if self.ys:
            b = np.concatenate(self.ys)
            if b.size < self.cap:
                pad = np.zeros(self.cap - b.size, dtype=np.float32)
                b = np.concatenate([b, pad])
            return b[:self.cap].copy()
        return np.zeros(self.cap, dtype=np.float32)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.xs:
            return (np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32))
        x = np.concatenate(self.xs)
        y = np.concatenate(self.ys)
        if self.n_seen <= self.cap:
            return x, y
        return self.xs[0], self.ys[0]


def spearman_ordinal(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rho with ordinal ranks (ties broken by order; recorded)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 3:
        return {"n_pairs": int(n), "rho": None, "status": "INSUFFICIENT"}
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    xm, ym = rx.mean(), ry.mean()
    vx = ((rx - xm) ** 2).sum()
    vy = ((ry - ym) ** 2).sum()
    if vx <= 0 or vy <= 0:
        return {"n_pairs": int(n), "rho": None, "status": "ZERO_VARIANCE"}
    rho = float(((rx - xm) * (ry - ym)).sum() / np.sqrt(vx * vy))
    if not np.isfinite(rho):
        raise ValueError("NaN/inf in spearman rho")
    return {"n_pairs": int(n), "rho": rho, "status": "OK",
            "rank_method": "ordinal (ties broken by order)"}


@torch.no_grad()
def audit_subject(seed: int, tag: str, device: str, limit: int | None,
                  scene_rows: list[dict]) -> dict:
    model, sha = load_model(seed, tag, device)
    ds_occ = Kitti2015Stereo(DATA_ROOT, split="hailo_val",
                             disparity_scale=256.0, occluded=True)
    ds_noc = Kitti2015Stereo(DATA_ROOT, split="hailo_val",
                             disparity_scale=256.0, occluded=False)
    n_img = len(ds_occ) if limit is None else min(limit, len(ds_occ))

    nB = len(BINS16) - 1
    bin_n = np.zeros(nB, dtype=np.int64)
    bin_sum = np.zeros(nB, dtype=np.float64)
    acc = {
        "n": 0, "sum_err": 0.0, "sum_sq": 0.0,
        "bad05": 0, "bad1": 0, "bad2": 0, "bad3": 0, "d1": 0,
    }
    strat = {k: {"n": 0, "sum": 0.0} for k in
             ("occ", "noc", "fg", "bg", "disc", "nondisc")}
    edge_n = np.zeros(len(EDGE_STRATA) - 1, dtype=np.int64)
    edge_sum = np.zeros(len(EDGE_STRATA) - 1, dtype=np.float64)
    corr_all = d7.CorrAcc()
    corr_bin = [d7.CorrAcc() for _ in range(nB)]
    auto = {(o, d): d7.CorrAcc()
            for o in ("h", "v") for d in (1, 2, 4, 8, 16, 32)}

    sum_oracle = 0.0
    sum_dom, sum_top2, sum_top3, sum_wta = 0.0, 0.0, 0.0, 0.0
    sum_dom_ref, sum_top2_ref, sum_top3_ref = 0.0, 0.0, 0.0
    sanity_max = 0.0
    wta_hit = 0
    fg_ok = True

    L_init, L_resid, L_final, L_gt, L_tex = [], [], [], [], []
    L_berr, L_bim = [], []
    rng = np.random.default_rng(1000 + seed * 131 + (0 if tag == "final" else 57))
    res = Reservoir(RESERVOIR_CAP, rng)
    H_img, W_img = 368, 1232

    for i in range(n_img):
        s = ds_occ[i]
        sn = ds_noc[i]
        if s.name != sn.name:
            raise ValueError(f"scene order mismatch {s.name} vs {sn.name}")
        gt = s.disparity.astype(np.float64)
        valid = gt > 0
        nv = int(valid.sum())
        noc_map = sn.disparity
        occluded = valid & (noc_map <= 0)

        try:
            import cv2

            obj = cv2.imread(
                str(DATA_ROOT / "training" / "obj_map" / s.name),
                cv2.IMREAD_UNCHANGED,
            )
            if obj is None:
                raise FileNotFoundError(s.name)
            obj = pad_and_crop(obj)
            fg_full = (obj > 0) & valid
        except Exception:
            fg_ok = False
            fg_full = np.zeros_like(valid)

        left_t = torch.from_numpy(normalize(s.left)).to(device)
        right_t = torch.from_numpy(normalize(s.right)).to(device)
        pred_t, st = model(left_t, right_t, return_stages=True)
        cost = st["aggregated_cost"]
        init_t = st["disparity_initial"]
        resid_t = st["refinement_residual"]
        final_t = st["disparity_final"]
        if not (torch.isfinite(pred_t).all() and torch.isfinite(init_t).all()
                and torch.isfinite(resid_t).all()
                and torch.isfinite(final_t).all()
                and torch.isfinite(cost).all()):
            raise ValueError(f"NaN/inf in stages scene {s.name}")

        pred = final_t[0, 0].cpu().numpy().astype(np.float64)
        init_c = init_t[0, 0].cpu().numpy().astype(np.float64)
        resid = resid_t[0, 0].cpu().numpy().astype(np.float64)
        err = np.abs(pred - gt)
        gv = gt[valid]
        evec = err[valid]

        acc["n"] += nv
        acc["sum_err"] += float(evec.sum())
        acc["sum_sq"] += float((evec ** 2).sum())
        acc["bad05"] += int((evec > 0.5).sum())
        acc["bad1"] += int((evec > 1.0).sum())
        acc["bad2"] += int((evec > 2.0).sum())
        acc["bad3"] += int((evec > 3.0).sum())
        acc["d1"] += int(((evec > 3.0) & (evec > 0.05 * gv)).sum())

        b = np.clip(np.searchsorted(BINS16, gv, side="right") - 1, 0, nB - 1)
        for bi in range(nB):
            m = b == bi
            c = int(m.sum())
            if c:
                bin_n[bi] += c
                bin_sum[bi] += float(evec[m].sum())

        for key, mfull in (("occ", occluded), ("noc", valid & ~occluded),
                           ("fg", fg_full & valid),
                           ("bg", valid & ~fg_full),
                           ("disc", None), ("nondisc", None)):
            if key in ("disc", "nondisc"):
                continue
            mv = mfull[valid]
            strat[key]["n"] += int(mv.sum())
            strat[key]["sum"] += float(evec[mv].sum())
        disc_full = bdiag.gt_discontinuity(gt.astype(np.float32), valid)
        dm = disc_full[valid]
        strat["disc"]["n"] += int(dm.sum())
        strat["disc"]["sum"] += float(evec[dm].sum())
        strat["nondisc"]["n"] += int((~dm).sum())
        strat["nondisc"]["sum"] += float(evec[~dm].sum())

        edist = d7.edge_distance_map(gt, valid)
        ed = edist[valid]
        ei = np.clip(np.searchsorted(np.array(EDGE_STRATA), ed, side="right") - 1,
                     0, len(edge_n) - 1)
        for k in range(len(edge_n)):
            m = ei == k
            c = int(m.sum())
            if c:
                edge_n[k] += c
                edge_sum[k] += float(evec[m].sum())

        e_init_signed = 8.0 * init_c[valid] - gv
        corr_all.add(resid[valid], e_init_signed)
        for bi in range(nB):
            m = b == bi
            if m.any():
                corr_bin[bi].add(resid[valid][m], e_init_signed[m])

        e_full = np.zeros_like(gt)
        e_full[valid] = pred[valid] - gt[valid]
        for d in (1, 2, 4, 8, 16, 32):
            mh = valid[:, :-d] & valid[:, d:]
            if mh.any():
                auto[("h", d)].add(e_full[:, :-d][mh], e_full[:, d:][mh])
            mv = valid[:-d, :] & valid[d:, :]
            if mv.any():
                auto[("v", d)].add(e_full[:-d, :][mv], e_full[d:, :][mv])

        vt = torch.from_numpy(valid).to(device)[None, None]
        gi = torch.from_numpy(
            np.where(valid, np.clip(gt, 0, bdiag.MAXD) / bdiag.SPACING, 0.0)
        ).to(device).float()[None, None]
        gi = torch.where(vt, gi, init_t)
        oracle_t = torch.relu(gi + model.refinement(gi, left_t))
        oracle = oracle_t[0, 0].cpu().numpy().astype(np.float64)
        oerr = np.abs(oracle - gt)[valid]
        sum_oracle += float(oerr.sum())

        cu = F.interpolate(cost.float(), size=(H_img, W_img),
                           mode="bilinear", align_corners=True)
        cn = ((cu - cu.mean(dim=1, keepdim=True))
              / (cu.std(dim=1, keepdim=True) + 1e-6))
        pw = torch.softmax(-cn, dim=1)
        if not torch.isfinite(pw).all():
            raise ValueError(f"NaN/inf in readout softmax scene {s.name}")

        # F2 sanity gate: the audit's full soft-argmin reconstruction must
        # equal the model's own disparity_initial within max-abs 1e-4.
        nD = pw.shape[1]
        cand = torch.arange(nD, dtype=pw.dtype, device=device).view(1, nD, 1, 1)
        init_recon = (pw * cand).sum(dim=1, keepdim=True)
        sanity = float((init_recon - init_t).abs().max())
        sanity_max = max(sanity_max, sanity)
        if sanity > 1e-4:
            raise ValueError(
                f"soft-argmin sanity scene {s.name}: max-abs {sanity:.3e} "
                f"> 1e-4")

        # A1.2: alternative initial maps on the FULL image (candidate units,
        # on device, same standardise/softmax as the model), passed through
        # the model's own refinement:
        # final_alt = relu(init_alt + refinement(init_alt, left)).
        a_full = pw.argmax(dim=1, keepdim=True)
        km1 = (a_full - 1).clamp(0, nD - 1)
        kp1 = (a_full + 1).clamp(0, nD - 1)
        w_dom = torch.cat([pw.gather(1, km1), pw.gather(1, a_full),
                           pw.gather(1, kp1)], dim=1)
        w_dom = w_dom / w_dom.sum(dim=1, keepdim=True)
        k_dom = torch.cat([km1, a_full, kp1], dim=1).to(pw.dtype)
        init_dom = (w_dom * k_dom).sum(dim=1, keepdim=True)
        t2_v, t2_i = torch.topk(pw, 2, dim=1)
        t2_v = t2_v / t2_v.sum(dim=1, keepdim=True)
        init_top2 = (t2_v * t2_i.to(pw.dtype)).sum(dim=1, keepdim=True)
        t3_v, t3_i = torch.topk(pw, 3, dim=1)
        t3_v = t3_v / t3_v.sum(dim=1, keepdim=True)
        init_top3 = (t3_v * t3_i.to(pw.dtype)).sum(dim=1, keepdim=True)
        if not (torch.isfinite(init_dom).all()
                and torch.isfinite(init_top2).all()
                and torch.isfinite(init_top3).all()):
            raise ValueError(f"NaN/inf in readout init_alt scene {s.name}")
        final_dom = torch.relu(init_dom + model.refinement(init_dom, left_t))
        final_t2 = torch.relu(init_top2 + model.refinement(init_top2, left_t))
        final_t3 = torch.relu(init_top3 + model.refinement(init_top3, left_t))
        dom_r = final_dom[0, 0].cpu().numpy().astype(np.float64)
        t2_r = final_t2[0, 0].cpu().numpy().astype(np.float64)
        t3_r = final_t3[0, 0].cpu().numpy().astype(np.float64)
        dom_re = np.abs(dom_r[valid] - gv)
        t2_re = np.abs(t2_r[valid] - gv)
        t3_re = np.abs(t3_r[valid] - gv)
        sum_dom_ref += float(dom_re.sum())
        sum_top2_ref += float(t2_re.sum())
        sum_top3_ref += float(t3_re.sum())

        Pv = pw[0, :, vt[0, 0]].double().cpu().numpy()  # (24, N)
        D = Pv.shape[0]
        a = Pv.argmax(axis=0)
        wta_hit += int((np.abs(a - np.clip(gv, 0, bdiag.MAXD)
                               / bdiag.SPACING) <= 1).sum())
        sum_wta += float(np.abs(8.0 * a - gv).sum())

        idx = np.arange(D)[:, None]
        lo = np.maximum(a - 1, 0)
        win = np.stack([np.clip(a - 1, 0, D - 1), a,
                        np.clip(a + 1, 0, D - 1)], axis=0)
        w3 = np.take_along_axis(Pv, win, axis=0)
        w3 = w3 / w3.sum(axis=0, keepdims=True)
        exp_dom = (w3 * win).sum(axis=0)
        sum_dom += float(np.abs(8.0 * exp_dom - gv).sum())
        for k, acc_key in ((2, "t2"), (3, "t3")):
            order = np.argsort(-Pv, axis=0)[:k]
            vk = np.take_along_axis(Pv, order, axis=0)
            vk = vk / vk.sum(axis=0, keepdims=True)
            expk = (vk * order).sum(axis=0)
            if k == 2:
                sum_top2 += float(np.abs(8.0 * expk - gv).sum())
            else:
                sum_top3 += float(np.abs(8.0 * expk - gv).sum())

        left_nb = np.concatenate([np.full((1, Pv.shape[1]), -np.inf), Pv[:-1]], axis=0)
        right_nb = np.concatenate([Pv[1:], np.full((1, Pv.shape[1]), -np.inf)], axis=0)
        is_peak = (Pv > left_nb) & (Pv >= right_nb)
        order_p = np.argsort(-Pv, axis=0)
        top1v = Pv[order_p[0], np.arange(Pv.shape[1])]
        top1k = order_p[0]
        peak_rank = np.argsort(np.argsort(-Pv, axis=0), axis=0)
        second_v = np.full(Pv.shape[1], -np.inf)
        second_k = np.full(Pv.shape[1], -1, dtype=np.int64)
        for r in range(1, D):
            cand_k = order_p[r, np.arange(Pv.shape[1])]
            cand_v = Pv[cand_k, np.arange(Pv.shape[1])]
            fill = (second_k < 0) & is_peak[cand_k, np.arange(Pv.shape[1])]
            second_v[fill] = cand_v[fill]
            second_k[fill] = cand_k[fill]
        bim = ((second_k >= 0) & (second_v >= 0.5 * top1v)
               & (np.abs(second_k - top1k) >= 2))
        gt64 = gv < 64.0
        L_berr.append(evec[gt64].astype(np.float32))
        L_bim.append(bim[gt64])

        dmap = final_t[0, 0]
        ys, xs = torch.meshgrid(
            torch.arange(H_img, device=device),
            torch.arange(W_img, device=device), indexing="ij")
        x_src = xs.unsqueeze(0) - dmap[0]
        gx = 2.0 * x_src / (W_img - 1) - 1.0
        gy = 2.0 * ys.unsqueeze(0).float() / (H_img - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)
        right_f = torch.from_numpy(s.right.transpose(2, 0, 1)).to(device).float()
                  
        left_f = torch.from_numpy(s.left.transpose(2, 0, 1)).to(device).float()
        warped = F.grid_sample(right_f[None], grid, mode="bilinear",
                               padding_mode="zeros", align_corners=True)[0]
        werr = torch.abs(left_f - warped).sum(0).cpu().numpy().astype(np.float64)
        xs_np = x_src[0].cpu().numpy()
        wm = valid & (gt < 64.0) & (xs_np >= 0)
        if wm.any():
            res.add(werr[wm].astype(np.float32), err[wm].astype(np.float32))

        L_init.append(init_t[0, 0].cpu().numpy()[valid].astype(np.float32))
        L_resid.append(resid_t[0, 0].cpu().numpy()[valid].astype(np.float32))
        L_final.append(final_t[0, 0].cpu().numpy()[valid].astype(np.float32))
        L_gt.append(gt[valid].astype(np.float32))
        L_tex.append(d7.texture_magnitude(s.left)[valid].astype(np.float32))

        gts = float(gv.sum())
        scene_rows.append({
            "seed": seed, "tag": tag, "scene": s.name,
            "valid_px": nv,
            "epe": float(evec.mean()),
            "d1": float((((evec > 3.0) & (evec > 0.05 * gv)).mean()) * 100.0),
            "rmse": float(np.sqrt((evec ** 2).mean())),
            "bad1": float((evec > 1.0).mean() * 100.0),
            "gtlt64_epe": (float(evec[gv < 64.0].mean())
                           if (gv < 64.0).any() else None),
            "init8_epe": float(np.abs(8.0 * init_c[valid] - gv).mean()),
            "oracle_epe": float(oerr.mean()),
            "dom_ref_epe": float(dom_re.mean()),
            "top2_ref_epe": float(t2_re.mean()),
            "top3_ref_epe": float(t3_re.mean()),
            "sanity_max_abs": sanity,
            "occ_epe": (float(evec[occluded[valid]].mean())
                        if occluded[valid].any() else None),
            "noc_epe": (float(evec[~occluded[valid]].mean())
                        if (~occluded[valid]).any() else None),
            "occ_frac": float(occluded[valid].mean()),
            "mean_gt": float(gv.mean()),
        })
        del (pred_t, st, cost, init_t, resid_t, final_t, cu, cn, pw, Pv,
              gi, oracle_t, warped, grid, right_f, left_f,
              init_recon, init_dom, init_top2, init_top3,
              final_dom, final_t2, final_t3)
        if device == "cuda" and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or (i + 1) == n_img:
            print(f"seed {seed}/{tag} [{i + 1}/{n_img}] "
                  f"epe_sofar={acc['sum_err'] / acc['n']:.4f}", flush=True)

    n = acc["n"]
    epe = acc["sum_err"] / n
    dm = disparity_metrics(
        np.concatenate(L_final).astype(np.float64),
        np.concatenate(L_gt).astype(np.float64))
    bad05 = 100.0 * acc["bad05"] / n
    assert abs(dm.epe - epe) < 1e-9, (dm.epe, epe)

    bins16 = []
    for bi, (lo, hi) in enumerate(zip(BINS16[:-1], BINS16[1:])):
        px = int(bin_n[bi])
        row = {"lo": lo, "hi": hi, "px": px,
               "frac_px": float(px / n) if n else 0.0}
        if px:
            row["epe"] = float(bin_sum[bi] / px)
            row["contribution"] = float(bin_sum[bi] / n)
        else:
            row["status"] = "EMPTY"
        bins16.append(row)
    contrib_sum = sum(r.get("contribution", 0.0) for r in bins16)
    assert abs(contrib_sum - epe) < BIN_CONTRIB_TOL, (contrib_sum, epe)

    strata = {}
    for key in ("occ", "noc", "fg", "bg", "disc", "nondisc"):
        px = strat[key]["n"]
        cell = {"px": px, "frac_px": float(px / n) if n else 0.0}
        cell["epe"] = float(strat[key]["sum"] / px) if px else None
        if key == "fg" and not fg_ok:
            cell["status"] = "OBJ_MAP_UNAVAILABLE"
        strata[key] = cell
    edge_rows = []
    for k, (lo, hi) in enumerate(zip(EDGE_STRATA[:-1], EDGE_STRATA[1:])):
        px = int(edge_n[k])
        lab = f"[{int(lo)},{int(hi)})" if np.isfinite(hi) else f"[{int(lo)},inf)"
        cell = {"bin": lab, "px": px,
                "frac_px": float(px / n) if n else 0.0}
        cell["epe"] = float(edge_sum[k] / px) if px else None
        edge_rows.append(cell)

    init = np.concatenate(L_init).astype(np.float64)
    resid_v = np.concatenate(L_resid).astype(np.float64)
    final = np.concatenate(L_final).astype(np.float64)
    gtv = np.concatenate(L_gt).astype(np.float64)
    tex = np.concatenate(L_tex).astype(np.float64)
    if not (np.isfinite(init).all() and np.isfinite(resid_v).all()
            and np.isfinite(final).all() and np.isfinite(gtv).all()
            and np.isfinite(tex).all()):
        raise ValueError("NaN/inf in concat vectors")
    del L_init, L_resid, L_final, L_gt, L_tex

    bb = np.clip(np.searchsorted(BINS16, gtv, side="right") - 1, 0, nB - 1)
    per_bin = []
    for bi, (lo, hi) in enumerate(zip(BINS16[:-1], BINS16[1:])):
        m = bb == bi
        px = int(m.sum())
        if px == 0:
            per_bin.append({"lo": lo, "hi": hi, "px": 0, "status": "EMPTY"})
            continue
        cs = d5.core_stats(init[m], resid_v[m], final[m], gtv[m])
        xcheck_epe = float(bin_sum[bi] / px)
        if abs(cs["epe_refined"] - xcheck_epe) > 1e-9:
            raise ValueError(f"bin {lo}-{hi} core_stats xcheck failed")
        row = {"lo": lo, "hi": hi, "px": px,
               "frac_px": float(px / n),
               "init8_epe": cs["epe_coarse_scaled"],
               "final_epe": cs["epe_refined"],
               "resid_mean_abs": cs["residual_mean_abs"],
               "corr_resid_initerr": corr_bin[bi].result().get("r"),
               "corr_pairs": corr_bin[bi].result().get("n_pairs"),
               "frac_improved": cs["frac_improved"],
               "frac_worsened": cs["frac_worsened"],
               "contribution": float(bin_sum[bi] / n)}
        per_bin.append(row)

    fall = np.abs(final - gtv).astype(np.float64)
    order = np.argsort(fall)
    tot = float(fall.sum())
    tail = {}
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(n * frac))
        tail[f"top_{int(frac * 100)}pct_share"] = float(fall[order[-k:]].sum() / tot)
    tail["quantiles"] = {str(q): float(np.percentile(fall, q))
                         for q in (50, 75, 90, 95, 99, 99.9)}

    tq = np.quantile(tex, np.linspace(0, 1, 11))
    tex_rows = []
    for d in range(10):
        m = (tex >= tq[d]) & (tex <= tq[d + 1] if d == 9 else tex < tq[d + 1])
        px = int(m.sum())
        cell = {"decile": d, "lo": float(tq[d]), "hi": float(tq[d + 1]),
                "px": px}
        cell["epe"] = float(fall[m].mean()) if px else None
        tex_rows.append(cell)

    acl = {}
    for ori in ("h", "v"):
        r1 = auto[(ori, 1)].result().get("r")
        if r1 is None or not (r1 > 0):
            acl[ori] = {"r_d1": r1, "length_px": None,
                        "status": "R_D1_NONPOSITIVE_OR_UNDEFINED"}
            continue
        L = None
        for d in (1, 2, 4, 8, 16, 32):
            rr = auto[(ori, d)].result().get("r")
            if rr is not None and rr < 0.5 * r1:
                L = d
                break
        acl[ori] = {"r_d1": r1, "length_px": L,
                    "status": ("falls_below_half_within_32" if L is not None
                               else "beyond_32px")}
    spatial = {
        "method": "Pearson r of signed error on valid pairs, "
                  "CorrAcc accumulators reused from d7_spatial",
        "autocorr_length_signed_err": acl,
        "pair_counts": {f"{o}_{d}": auto[(o, d)].result().get("n_pairs")
                        for o in ("h", "v") for d in (1, 2, 4, 8, 16, 32)},
    }

    oracle_epe = sum_oracle / n
    oracle_reduction = epe - oracle_epe
    dom_coarse = sum_dom / n
    top2_coarse = sum_top2 / n
    top3_coarse = sum_top3 / n
    dom_epe = sum_dom_ref / n
    top2_epe = sum_top2_ref / n
    top3_epe = sum_top3_ref / n
    wta_acc = float(wta_hit / n)

    berr = np.concatenate(L_berr) if L_berr else np.zeros(0, dtype=np.float32)
    bim_all = np.concatenate(L_bim) if L_bim else np.zeros(0, dtype=bool)
    del L_berr, L_bim
    if berr.size:
        q90 = float(np.quantile(berr.astype(np.float64), 0.9))
        q50 = float(np.quantile(berr.astype(np.float64), 0.5))
        top = berr >= q90
        low = berr <= q50
        r_top = float(bim_all[top].mean()) if top.any() else None
        r_low = float(bim_all[low].mean()) if low.any() else None
    else:
        q90 = q50 = r_top = r_low = None
    bimod = {
        "population": "GT<64 valid pixels",
        "px": int(berr.size),
        "peak_def": "strictly greater than left neighbour, >= right "
                    "neighbour (single-sided at edges)",
        "bimodal_def": "second local softmax peak >= 0.5x first, "
                       "|candidate diff| >= 2",
        "top_decile_thresh": q90, "lower_half_thresh": q50,
        "rate_top_decile": r_top, "rate_lower_half": r_low,
        "ratio": (float(r_top / r_low) if r_top is not None
                  and r_low not in (None, 0) else None),
    }

    wx, wy = res.arrays()
    sp = spearman_ordinal(wx, wy)
    warp = {
        "method": "torch grid_sample bilinear align_corners=True, "
                  "padding zeros; warp_err=|I_L-warp(I_R,d_final)| summed "
                  "over RGB; mask=valid & GT<64 & x-d_final>=0; "
                  "ordinal-rank Spearman on reservoir sample",
        "spearman": sp,
        "reservoir_cap": RESERVOIR_CAP,
        "reservoir_seen": int(res.n_seen),
        "sampling": "per-checkpoint vectorised reservoir, rng seed "
                    f"{1000 + seed * 131 + (0 if tag == 'final' else 57)}",
    }

    oa = d5.oracle_alpha(init, resid_v, gtv)
    ob = d5.oracle_beta(init, resid_v, gtv)
    sos = d5.sign_only_scaled_epe(init, resid_v, gtv)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "seed": seed, "tag": tag, "sha256": sha,
        "scenes": n_img,
        "subset": limit is not None,
        "valid_pixels": n,
        "epe": epe,
        "rmse": float(np.sqrt(acc["sum_sq"] / n)),
        "d1": float(100.0 * acc["d1"] / n),
        "bad05": bad05,
        "bad1": float(100.0 * acc["bad1"] / n),
        "bad2": float(100.0 * acc["bad2"] / n),
        "bad3": float(100.0 * acc["bad3"] / n),
        "bins16": bins16,
        "strata": strata,
        "edge_distance": edge_rows,
        "texture_deciles": tex_rows,
        "texture_method": "d7_spatial.texture_magnitude (Sobel 3x3 on "
                          "luminance); equal-count deciles over valid pixels",
        "per_bin_init_refine": per_bin,
        "corr_resid_initerr_pooled": corr_all.result(),
        "tail": tail,
        "spatial": spatial,
        "oracle": {"epe": oracle_epe, "reduction": oracle_reduction,
                   "method": "bottleneck_diag GT-index oracle, unchanged: "
                             "clip(GT,0,184)/8 where valid else init, "
                             "relu(gi + refinement(gi, left))"},
        "readouts": {
            "dominant_mode_pm1_epe": dom_epe,
            "top2_epe": top2_epe, "top3_epe": top3_epe,
            "dominant_mode_pm1_coarse_epe": dom_coarse,
            "top2_coarse_epe": top2_coarse,
            "top3_coarse_epe": top3_coarse,
            "wta_hard_argmin_x8_epe": float(sum_wta / n),
            "wta_within_1_candidate_frac": wta_acc,
            "bimodality": bimod,
            "sanity_softargmin_max_abs": sanity_max,
            "sanity_status": "PASS",
            "method": "full-res cost bilinear align_corners=True, "
                      "unbiased-std standardise eps 1e-6 (model readout), "
                      "softmax(-cost); A1.2: dominant=argmax+-1 renormalised "
                      "and top-k renormalised built on the FULL image in "
                      "candidate units, then final_alt = "
                      "relu(init_alt + refinement(init_alt, left)) and "
                      "pooled contract EPE on valid pixels (gating); "
                      "coarse pre-refinement EPEs (x8 to pixels) secondary "
                      "only; F2 sanity: full soft-argmin reconstruction "
                      "equals disparity_initial within max-abs 1e-4",
        },
        "warp": warp,
        "d5_oracle_scale_pooled": oa,
        "d5_oracle_offset_pooled": ob,
        "sign_only_scaled_epe": sos,
        "vec_store": "float32 concat freed per subject after d5 probes",
    }


@torch.no_grad()
def train_split_epe(seed: int, tag: str, device: str) -> dict:
    """hailo_calib 0-159 pooled + GT<64 EPE. Same metric code family
    (disparity_metrics/valid>0/GT/256); labelled NOT the contract."""
    model, _ = load_model(seed, tag, device)
    ds = Kitti2015Stereo(DATA_ROOT, split="hailo_calib",
                         disparity_scale=256.0, occluded=True)
    n = 0
    s_err = 0.0
    n64 = 0
    s64 = 0.0
    for i in range(len(ds)):
        samp = ds[i]
        gt = samp.disparity.astype(np.float64)
        valid = gt > 0
        out = model(torch.from_numpy(normalize(samp.left)).to(device),
                    torch.from_numpy(normalize(samp.right)).to(device))
        pred = out[0, 0].cpu().numpy().astype(np.float64)
        e = np.abs(pred[valid] - gt[valid])
        gv = gt[valid]
        n += int(valid.sum())
        s_err += float(e.sum())
        m = gv < 64.0
        n64 += int(m.sum())
        s64 += float(e[m].sum())
        del out
        if device == "cuda" and (i + 1) % 40 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 40 == 0 or (i + 1) == len(ds):
            print(f"train seed {seed}/{tag} [{i + 1}/{len(ds)}]", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "split": "hailo_calib",
        "scenes": len(ds),
        "contract": False,
        "note": "NOT the frozen contract; same metric code",
        "valid_pixels": n,
        "epe": float(s_err / n),
        "gtlt64_px": n64,
        "gtlt64_epe": float(s64 / n64),
    }


def stop(path: Path, prov: dict, reason: str) -> "NoReturn":
    out = {"status": "STOP", "provenance": prov, "reason": reason,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_write_json(path, out)
    print("STOP: " + reason, flush=True)
    raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--subjects", type=str, default="final",
                    choices=("final", "all"))
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tags = ("final",) if args.subjects == "final" else ("final", "best")
    smoke = args.limit is not None

    verdict = json.loads(VERDICT_PATH.read_text(encoding="utf-8"))
    prov = {
        "git_head": git_head(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": device,
        "checkpoint_sha256": {
            f"seed{s}_{t}": EXPECTED_SHA[(f"seed{s}", t)]
            for s in (0, 1, 2) for t in tags
        },
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = OUT_DIR / "audit_e3.json"

    subjects: dict[str, dict] = {}
    scene_rows: list[dict] = []
    for s in (0, 1, 2):
        for t in tags:
            key = f"seed{s}_{t}"
            print(f"=== subject {key} "
                  f"({'SMOKE' if smoke else 'FULL'}) ===", flush=True)
            try:
                rec = audit_subject(s, t, device, args.limit, scene_rows)
            except ValueError as ex:
                stop(out_path, prov, f"{key}: {ex}")
            if not smoke:
                model, _ = load_model(s, t, device)
                v = ev.score(model, "hailo_val", device)
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
                rec["contract_score_reuse"] = {
                    "epe": v["metrics"]["epe"],
                    "metrics": v["metrics"],
                    "bins": v["bins"],
                    "guard": v["guard"],
                }
                if abs(v["metrics"]["epe"] - rec["epe"]) > SCORE_XCHECK_TOL:
                    stop(out_path, prov,
                         f"{key}: ev.score/my-loop mismatch "
                         f"{v['metrics']['epe']:.10f} vs {rec['epe']:.10f}")
                if t == "final":
                    want = verdict["per_seed_final"][s]
                    if abs(rec["epe"] - want) > GATE_TOL:
                        stop(out_path, prov,
                             f"GATE {key}: epe {rec['epe']:.10f} != "
                             f"verdict {want:.10f} (tol 1e-6)")
                    if rec["valid_pixels"] != EXPECTED_VALID:
                        stop(out_path, prov,
                             f"GATE {key}: valid {rec['valid_pixels']} != "
                             f"{EXPECTED_VALID}")
                    try:
                        g = refuse_unless_contract(
                            40, rec["valid_pixels"], 256.0, "hailo_val",
                            "disp_occ_0")
                    except ValueError as ex:
                        stop(out_path, prov, f"GATE {key}: {ex}")
                    rec["gate"] = {"verdict_match_1e_6": True,
                                   "valid_pixels_ok": True,
                                   "contract": g,
                                   "status": "PASS"}
                rec["train_split"] = train_split_epe(s, t, device)
            else:
                rec["contract_score_reuse"] = None
                rec["train_split"] = None
                rec["gate"] = {"status": "SKIPPED_SMOKE"}
            subjects[key] = rec

    finals = [subjects[f"seed{s}_final"] for s in (0, 1, 2)]
    rules: dict[str, dict] = {}
    if not smoke:
        # A1.1 probe validity gate: a ceiling probe counts only if it
        # improves on the unmodified model. oracle EPE >= model EPE on a
        # seed -> INCONCLUSIVE (off-manifold), not FAIL. Overall
        # INCONCLUSIVE if any seed is INCONCLUSIVE.
        oracle_rows = []
        for f in finals:
            oe, me = f["oracle"]["epe"], f["epe"]
            red = me - oe
            if oe >= me:
                st = "INCONCLUSIVE"
            elif red >= 0.168:
                st = "PASS"
            else:
                st = "FAIL"
            oracle_rows.append({"seed": f["seed"], "model_epe": me,
                               "oracle_epe": oe, "reduction": red,
                               "status": st})
        if any(r["status"] == "INCONCLUSIVE" for r in oracle_rows):
            oracle_overall = "INCONCLUSIVE"
        elif all(r["status"] == "PASS" for r in oracle_rows):
            oracle_overall = "PASS"
        else:
            oracle_overall = "FAIL"
        oracle_b = [r["status"] == "PASS" for r in oracle_rows]
        readout_low_b = []
        for f in finals:
            r = f["readouts"]
            readout_low_b.append(bool(
                r["dominant_mode_pm1_epe"] < f["epe"]
                or r["top2_epe"] < f["epe"] or r["top3_epe"] < f["epe"]))
        bim_b = []
        for f in finals:
            bim_b.append(bool(
                f["readouts"]["bimodality"]["ratio"] is not None
                and f["readouts"]["bimodality"]["ratio"] >= 2.0))
        readout_b = [bool(a or b) for a, b in zip(readout_low_b, bim_b)]
        warp_b = [(f["warp"]["spearman"]["rho"] is not None
                   and f["warp"]["spearman"]["rho"] >= 0.20) for f in finals]
        gap_b = [(f["train_split"]["gtlt64_epe"]
                  <= 0.70 * gtlt64_of(f)) for f in finals]
        rules = {
            "R-ORACLE": {"per_seed": oracle_b, "rule": all(oracle_b),
                         "overall": oracle_overall,
                         "seeds": oracle_rows},
            "R-READOUT": {"per_seed": readout_b, "rule": all(readout_b),
                          "readout_low_per_seed": readout_low_b,
                          "bimodal_per_seed": bim_b},
            "R-WARP": {"per_seed": warp_b, "rule": all(warp_b)},
            "R-GAP": {"per_seed": gap_b, "rule": all(gap_b)},
        }
        leakage = {d: (REPO / d).exists() for d in CANDIDATE_DATA_DIRS}
        if oracle_overall == "INCONCLUSIVE":
            branch = "A1.3 (R-ORACLE INCONCLUSIVE)"
            if all(readout_low_b):
                sel = ("M1", branch + ": an alternative refined readout "
                       "(A1.2) lowers pooled contract EPE on 3/3 seeds "
                       "(bimodality limb alone not sufficient here)")
            elif all(warp_b):
                sel = ("M2", branch + ": R-WARP holds on 3/3 seeds")
            elif all(gap_b) and any(leakage.values()):
                sel = ("DATA", branch + ": R-GAP holds and leakage-free "
                       "source on disk")
            else:
                sel = ("STOP", branch + ": SUB-1.0 NOT ACHIEVED; E3 frozen")
        else:
            branch = "frozen (§10; R-ORACLE PASS or FAIL)"
            if all(oracle_b) and all(readout_b):
                sel = ("M1", branch + ": R-ORACLE and R-READOUT hold "
                       "on 3/3 seeds")
            elif (not all(oracle_b)) and all(warp_b):
                sel = ("M2", branch + ": NOT R-ORACLE and R-WARP holds "
                       "on 3/3 seeds")
            elif all(gap_b) and any(leakage.values()):
                sel = ("DATA", branch + ": R-GAP holds and leakage-free "
                       "source on disk")
            else:
                sel = ("STOP", branch + ": SUB-1.0 NOT ACHIEVED; E3 frozen")
        selection = {"mechanism": sel[0], "reason": sel[1],
                     "branch": branch,
                     "leakage_free_source_checked": leakage}

        def mean(vals):
            return float(np.mean(vals))

        mech = [
            ("A", "distribution supervision",
             max(0.0, mean([max(f["epe"] - f["readouts"]["dominant_mode_pm1_epe"],
                                f["epe"] - f["readouts"]["top2_epe"],
                                f["epe"] - f["readouts"]["top3_epe"])
                            for f in finals])),
             "max refined-readout (A1.2, through refinement) EPE reduction, "
             "3-seed mean (finals)"),
            ("B", "matching ceiling (oracle-init)",
             max(0.0, mean([f["oracle"]["reduction"] for f in finals])),
             "R-ORACLE reduction, 3-seed mean (finals)"),
            ("C", "warp-error refinement input",
             mean([((f["warp"]["spearman"]["rho"] or 0.0) ** 2) * f["epe"]
                   for f in finals]),
             "Spearman^2 x pooled EPE, 3-seed mean (finals)"),
            ("D", "high-disparity coverage",
             mean([sum(r["contribution"] for r in f["bins16"]
                       if r["lo"] >= 64) for f in finals]),
             "GT>=64 contribution px (REJECTED as primary by arithmetic)"),
            ("E", "GWC / matching WTA",
             mean([wta_miss_mass(f) for f in finals]),
             "EPE mass on WTA-miss pixels (|argmin-GT/8|>1)"),
            ("F", "edge/discontinuity",
             mean([edge_excess(f) for f in finals]),
             "excess EPE mass within <2px of discontinuity vs >=2px"),
            ("G", "deep supervision of init",
             mean([f["sign_only_scaled_epe"] for f in finals]),
             "d5 sign-only-scaled residual sign-error mass"),
            ("H", "data/regularisation gap",
             max(0.0, mean([f["epe"] - f["train_split"]["epe"]
                            for f in finals])),
             "max(0, contract-train EPE), 3-seed mean (finals)"),
            ("I", "occlusion handling",
             mean([occ_excess(f) for f in finals]),
             "max(0,EPE_occ-EPE_noc) x frac_occ"),
        ]
        ranked = sorted(
            ({"id": m, "name": n, "contribution_px": c, "basis": b}
             for m, n, c, b in mech),
            key=lambda r: r["contribution_px"], reverse=True)
    else:
        rules = {"note": "SMOKE run: R quantities computed on subset, "
                         "non-gating; R-READOUT/R-WARP/R-GAP need full runs"}
        for key, f in subjects.items():
            r = f["readouts"]
            rules[key] = {
                "oracle_reduction": f["oracle"]["reduction"],
                "oracle_epe": f["oracle"]["epe"],
                "model_epe": f["epe"],
                "dom_mode_delta": f["epe"] - r["dominant_mode_pm1_epe"],
                "top2_delta": f["epe"] - r["top2_epe"],
                "top3_delta": f["epe"] - r["top3_epe"],
                "dom_mode_coarse_delta":
                    f["epe"] - r["dominant_mode_pm1_coarse_epe"],
                "top2_coarse_delta": f["epe"] - r["top2_coarse_epe"],
                "top3_coarse_delta": f["epe"] - r["top3_coarse_epe"],
                "sanity_softargmin_max_abs":
                    r["sanity_softargmin_max_abs"],
                "bimodality_ratio": r["bimodality"]["ratio"],
                "warp_rho": f["warp"]["spearman"]["rho"],
            }
        selection = {"mechanism": "SMOKE",
                     "reason": "subset run; no selection"}
        ranked = []

    out = {
        "status": "SMOKE" if smoke else "OK",
        "experiment": "stage-f F1 zero-training E3 audit",
        "provenance": prov,
        "subjects": subjects,
        "rules": rules,
        "selection": selection,
        "mechanisms_ranked": ranked,
        "reused": [
            {"name": "StereoNet/StereoNetConfig",
             "source": "src/models/stereonet"},
            {"name": "CFG/score",
             "source": "stage_e_recipe/kaggle/bundle_e3/scripts/eval_tier2.py"},
            {"name": "pooled_metrics/refuse_unless_contract/sha256_file",
             "source": "phase1/harness/frozen_eval.py"},
            {"name": "SPACING/NDISP/MAXD/gt_discontinuity/GT-index oracle",
             "source": "phase1/harness/bottleneck_diag.py"},
            {"name": "disparity_metrics",
             "source": "src/evaluation/metrics.py"},
            {"name": "core_stats/oracle_alpha/oracle_beta/sign_only_scaled_epe",
             "source": "stage_a_diagnostics/scripts/d5_refinement.py"},
            {"name": "CorrAcc/edge_distance_map/texture_magnitude",
             "source": "stage_a_diagnostics/scripts/d7_spatial.py"},
            {"name": "Kitti2015Stereo(occluded=False)/normalize/pad_and_crop",
             "source": "src/datasets/kitti2015.py"},
            {"name": "obj_map fg rule (fg = raw>0 after pad_and_crop)",
             "source": "data/kitti2015/training/obj_map + cv2 IMREAD_UNCHANGED"},
        ],
        "reimplemented": [],
        "reimplemented_note": "empty: every reuse above imports without side "
                              "effects; warp sampler, top-k/dominant-mode "
                              "readouts, bimodality rate, ordinal-rank "
                              "Spearman, reservoir sampler and bad0.5 are new "
                              "probe code documented under methods",
        "methods": {
            "bins16": "eval_tier2.BINS edges; contribution=bin_err_sum/"
                      "total_n; assert sums to pooled EPE",
            "readouts": subjects[next(iter(subjects))]["readouts"]["method"],
            "warp": subjects[next(iter(subjects))]["warp"]["method"],
            "tail": "float64 sort of pooled |err|; top-frac share of EPE sum",
            "quantiles": "numpy percentile of pooled |err|",
            "bad05": "100*mean(|err|>0.5)",
            "gap_population": "train hailo_calib pooled + GT<64; NOT contract",
        },
        "sampling": {
            "reservoir_cap_per_checkpoint": RESERVOIR_CAP,
            "note": "Spearman/quantile inputs sampled or full-sort as "
                    "recorded per subject; error tail uses full pooled sort",
        },
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(out_path, out)

    csv_path = OUT_DIR / "per_scene.csv"
    fd, tmp = tempfile.mkstemp(dir=str(OUT_DIR), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(scene_rows[0].keys()))
            w.writeheader()
            w.writerows(scene_rows)
        os.replace(tmp, csv_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print(f"wrote {out_path} + {csv_path} status={out['status']}", flush=True)


def gtlt64_of(f: dict) -> float:
    px = sum(r["px"] for r in f["bins16"] if r["hi"] <= 64)
    s = sum(r["epe"] * r["px"] for r in f["bins16"]
            if r["hi"] <= 64 and "epe" in r)
    return float(s / px)


def wta_miss_mass(f: dict) -> float:
    wta_epe = f["readouts"]["wta_hard_argmin_x8_epe"]
    acc = f["readouts"]["wta_within_1_candidate_frac"]
    return float(max(0.0, (wta_epe - f["epe"]) * (1.0 - acc)))


def edge_excess(f: dict) -> float:
    ed = f["edge_distance"]
    near = [c for c in ed if c["bin"] in ("[0,1)", "[1,2)")]
    far = [c for c in ed if c["bin"] not in ("[0,1)", "[1,2)")]
    nn = sum(c["px"] for c in near)
    ns = sum(c["epe"] * c["px"] for c in near if c["epe"] is not None)
    fn = sum(c["px"] for c in far)
    fs = sum(c["epe"] * c["px"] for c in far if c["epe"] is not None)
    if not nn or not fn:
        return 0.0
    return float(max(0.0, ns / nn - fs / fn) * nn / f["valid_pixels"])


def occ_excess(f: dict) -> float:
    o, v = f["strata"]["occ"], f["strata"]["noc"]
    if o["epe"] is None or v["epe"] is None:
        return 0.0
    return float(max(0.0, o["epe"] - v["epe"]) * o["frac_px"])


if __name__ == "__main__":
    main()