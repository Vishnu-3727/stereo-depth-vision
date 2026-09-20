"""STAGE-A D7: SPATIAL / CONTEXT DIAGNOSTIC (frozen model, read-only).

FROZEN-MODEL DIAGNOSTIC. No model change, no retraining, no tuning, no
mechanism verdict (step 8 does verdicts).

Population: FULL-RESOLUTION frozen valid mask (GT > 0 in disp_occ_0),
3,802,797 px/seed -- the same pixels D0/D1/D2/D5/D6 used. Every metric is
reported twice: on all valid pixels, and on NON-OCCLUDED valid pixels only.
Occlusion mask (established definition, as in
phase1/runs/arm_v_diag/reference_strata.py and D5/D6): valid in disp_occ_0
AND NOT valid in disp_noc_0.

Scientific question: are residual errors spatially STRUCTURED enough that
additional context propagation could plausibly address them, or are they
largely independent noise?

Works one image at a time (float64 accumulators; raises on NaN/inf).
A seed-0 --limit 2 smoke test precedes full runs.

Outputs (atomic .tmp + rename):
  stage_a_diagnostics/spatial_context.json
  stage_a_diagnostics/raw/d7_s{0,1,2}.npz
  stage_a_diagnostics/raw/d7_bins.csv
Usage:
  python stage_a_diagnostics/scripts/d7_spatial.py [--limit N] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse
import csv
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
EDGES = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 184, np.inf]
OFFSETS = [1, 2, 4, 8, 16, 32]
CONF_OFFSETS = [1, 4, 16]
EDGE_BIN_EDGES = [0, 1, 2, 4, 8, 16, np.inf]
EXPECTED_VALID = 3802797
EXPECTED_SEED0_EPE = 1.4149795736625577
TOL_BASELINE_PX = 1e-6
TOL_D2_XCHECK = 1e-6
ERR_THRESH = 3.0  # D1 criterion on |e|
LOW_N = 1000

try:
    from scipy import ndimage as _ndi
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


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


def bin_label(lo, hi) -> str:
    if np.isfinite(hi):
        return f"[{int(lo)},{int(hi)})"
    return f"[{int(lo)},inf)"


LABELS = [bin_label(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]


def d1_pct(err: np.ndarray, gt: np.ndarray) -> float:
    return float((((err > 3.0) & (err > 0.05 * gt)).mean()) * 100.0)


class CorrAcc:
    """Float64 sufficient statistics for Pearson correlation."""

    __slots__ = ("n", "sx", "sy", "sxx", "syy", "sxy")

    def __init__(self) -> None:
        self.n = 0.0
        self.sx = 0.0
        self.sy = 0.0
        self.sxx = 0.0
        self.syy = 0.0
        self.sxy = 0.0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size == 0:
            return
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("NaN/inf in correlation pairs")
        self.n += float(x.size)
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxx += float((x * x).sum())
        self.syy += float((y * y).sum())
        self.sxy += float((x * y).sum())

    def result(self) -> dict:
        n = self.n
        if n < 2:
            return {"n_pairs": int(n), "r": None, "status": "INSUFFICIENT"}
        cov = self.sxy - self.sx * self.sy / n
        vx = self.sxx - self.sx * self.sx / n
        vy = self.syy - self.sy * self.sy / n
        if not (np.isfinite(cov) and np.isfinite(vx) and np.isfinite(vy)):
            raise ValueError("NaN/inf in correlation reduction")
        if vx <= 0 or vy <= 0:
            return {"n_pairs": int(n), "r": None, "status": "ZERO_VARIANCE"}
        r = cov / float(np.sqrt(vx * vy))
        if not np.isfinite(r):
            raise ValueError("NaN/inf in correlation r")
        return {"n_pairs": int(n), "r": float(r)}


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connectivity connected components of a bool mask.

    Uses scipy.ndimage.label with a 3x3 ones structuring element when scipy
    is present (it is: scipy 1.18.0); otherwise a union-find fallback.
    Returns (label_image, n_components). Background (False) is 0.
    """
    mask = np.ascontiguousarray(mask, dtype=bool)
    if _HAVE_SCIPY:
        lab, n = _ndi.label(mask, structure=np.ones((3, 3), dtype=int))
        return lab, int(n)
    # Union-find fallback (8-connectivity), two-pass.
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int64)
    parent: dict[int, int] = {}
    nxt = 0

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for y in range(H):
        for x in range(W):
            if not mask[y, x]:
                continue
            neigh = []
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and lab[yy, xx] > 0:
                    neigh.append(int(lab[yy, xx]))
            if not neigh:
                nxt += 1
                parent[nxt] = nxt
                lab[y, x] = nxt
            else:
                m = min(neigh)
                lab[y, x] = m
                for v in neigh:
                    if v != m:
                        union(m, v)
    # second pass: relabel roots compactly
    remap: dict[int, int] = {}
    nn = 0
    for y in range(H):
        for x in range(W):
            if lab[y, x] > 0:
                r = find(int(lab[y, x]))
                if r not in remap:
                    nn += 1
                    remap[r] = nn
                lab[y, x] = remap[r]
    return lab, nn


def component_size_stats(sizes: np.ndarray, n_images: int) -> dict:
    sizes = np.asarray(sizes, dtype=np.float64)
    n_err = int(sizes.sum()) if sizes.size else 0
    if sizes.size == 0:
        return {"n_components_total": 0, "n_components_per_image": 0.0,
                "n_error_px": 0, "median_size": None, "p90_size": None,
                "max_size": None, "frac_err_mass_in_gt100": None,
                "status": "NO_ERROR_PIXELS"}
    out = {
        "n_components_total": int(sizes.size),
        "n_components_per_image": float(sizes.size / n_images),
        "n_error_px": n_err,
        "median_size": float(np.median(sizes)),
        "p90_size": float(np.percentile(sizes, 90)),
        "max_size": float(sizes.max()),
        "frac_err_mass_in_gt100": float(sizes[sizes > 100].sum() / n_err)
        if n_err else None,
    }
    check_finite("comp stats",
                 np.asarray([v for v in out.values() if isinstance(v, float)]))
    return out


def edge_distance_map(gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) to nearest GT disparity discontinuity.

    Discontinuity: valid pixel with a 4-neighbour (up/down/left/right) that
    is also valid and differs in GT by >= 3 px. Both pixels of such a pair
    are marked. Distance via scipy.ndimage.distance_transform_edt on the
    complement (scipy is present; stated).
    """
    H, W = gt.shape
    disc = np.zeros((H, W), dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb_v = np.zeros_like(valid)
        nb_g = np.zeros_like(gt)
        if dy == -1:
            nb_v[:-1, :] = valid[1:, :]
            nb_g[:-1, :] = gt[1:, :]
        elif dy == 1:
            nb_v[1:, :] = valid[:-1, :]
            nb_g[1:, :] = gt[:-1, :]
        elif dx == -1:
            nb_v[:, :-1] = valid[:, 1:]
            nb_g[:, :-1] = gt[:, 1:]
        else:
            nb_v[:, 1:] = valid[:, :-1]
            nb_g[:, 1:] = gt[:, :-1]
        disc |= valid & nb_v & (np.abs(gt - nb_g) >= 3.0)
    if _HAVE_SCIPY:
        dist = _ndi.distance_transform_edt(~disc)
    else:  # pragma: no cover - scipy is present in this env
        raise RuntimeError("scipy required for edge distance transform")
    return dist.astype(np.float64)


def texture_magnitude(left_u8: np.ndarray) -> np.ndarray:
    """Left-image local gradient magnitude at full resolution.

    Operator (stated): grayscale luminance Y = 0.299R+0.587G+0.114B
    (float64), cv2.Sobel 3x3 (dx=1,dy=0 and dx=0,dy=1, ksize=3, scale=1),
    magnitude sqrt(gx^2+gy^2). Window 3x3.
    """
    import cv2

    y = (0.299 * left_u8[:, :, 0] + 0.587 * left_u8[:, :, 1]
         + 0.114 * left_u8[:, :, 2]).astype(np.float64)
    gx = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)
    if not (np.isfinite(gx).all() and np.isfinite(gy).all()):
        raise ValueError("NaN/inf in Sobel gradients")
    return np.sqrt(gx * gx + gy * gy)


def compute_receptive_field() -> dict:
    """Theoretical receptive field from the P2A architecture in full-res px.

    P2A config: downsample_levels=3 (stride 8), residual_blocks=6,
    aggregation 4 Conv3D(3x3x3) filter layers + 1 Conv3D output layer,
    refinement input Conv3x3 + 6 ResBlocks (2 convs each) with dilations
    (1,2,4,8,1,1) + output Conv3x3. ResBlock convs both carry the block
    dilation (blocks.py: padding=dilation, dilation=dilation for conv1
    AND conv2). All refinement convs stride 1 at full resolution.
    Rule: RF += (k-1)*dilation*stride_acc per conv; stride_acc *= stride.
    """
    # --- feature extractor (full-res px) ---
    rf = 1
    s_acc = 1
    steps = []
    for _ in range(3):  # 5x5 stride-2 downsample convs
        rf += (5 - 1) * 1 * s_acc
        s_acc *= 2
        steps.append(rf)
    # 6 residual blocks x 2 convs, k=3 d=1 s=1 at stride_acc=8
    for _ in range(6 * 2):
        rf += (3 - 1) * 1 * s_acc
    steps.append(rf)
    rf += (3 - 1) * 1 * s_acc  # output conv
    feat_rf = rf  # stride_acc = 8
    # --- aggregation at feature res (stride 8 -> 16 full-res px per cell) ---
    agg4 = feat_rf + 4 * 2 * 8   # 4 filter layers as briefed
    agg5 = feat_rf + 5 * 2 * 8   # 4 filter + 1 output layer (actual code)
    # --- refinement alone at full res ---
    r = 1
    r += 2  # input conv k3 d1
    for dd in (1, 2, 4, 8, 1, 1):
        r += 2 * dd * 2  # 2 convs per block at dilation dd
    r += 2  # output conv
    ref_rf = r
    # --- end-to-end: coarse map (feat+agg RF) upsampled, then refinement
    # combines a ref_rf x ref_rf neighbourhood of it ---
    total4 = agg4 + (ref_rf - 1)
    total5 = agg5 + (ref_rf - 1)
    return {
        "method": "RF += (k-1)*dilation*stride_acc per conv; stride_acc *= stride",
        "feature_downsample_rf_after_3x5x5s2": steps[0:3],
        "feature_after_resblocks": steps[3],
        "feature_extractor_rf_fullres_px": feat_rf,
        "feature_stride": s_acc,
        "aggregation_4filter_rf_fullres_px": agg4,
        "aggregation_5conv_incl_output_rf_fullres_px": agg5,
        "refinement_only_rf_fullres_px": ref_rf,
        "end_to_end_4filter_fullres_px": total4,
        "end_to_end_5conv_fullres_px": total5,
        "headline_end_to_end_fullres_px": total5,
        "note": ("aggregation.py builds 4 Conv3d filter layers + 1 Conv3d "
                 "output layer = 5 3x3x3 convs; the brief counts 4, so both "
                 "variants are stated and the 5-conv total is headlined."),
    }


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
        raise ValueError(f"seed {seed}: split sizes {len(ds_occ)}/{len(ds_noc)}")
    n_img = len(ds_occ) if limit is None else min(limit, len(ds_occ))
    idx_grid = torch.arange(D, dtype=torch.float32, device=device)

    pops = ("all", "nonocc")
    # correlation accumulators: key (pop, field, ori, d, kind) + per-bin
    acc: dict = {}
    # top-1 neighbour correlation: key (pop, d)
    tacc: dict = {}

    def get_acc(key):
        a = acc.get(key)
        if a is None:
            a = acc[key] = CorrAcc()
        return a

    def get_tacc(key):
        a = tacc.get(key)
        if a is None:
            a = tacc[key] = CorrAcc()
        return a

    nB = len(LABELS)
    # per-bin accumulators: (pop, field, ori, d, kind, bin) -> CorrAcc
    bacc: dict = {}

    def get_bacc(key):
        a = bacc.get(key)
        if a is None:
            a = bacc[key] = CorrAcc()
        return a

    comp_sizes = {p: [] for p in pops}
    comp_sizes_perm = {p: [] for p in pops}
    comp_n_per_image = {p: [] for p in pops}
    comp_n_per_image_perm = {p: [] for p in pops}

    # per-valid-pixel vectors for edge/texture/confidence stages
    L_ae, L_gt, L_occ = [], [], []
    L_top1, L_margin, L_ent = [], [], []
    L_grad, L_edist = [], []
    n_valid = 0
    n_nonocc = 0
    epe_sum = 0.0

    for i in range(n_img):
        s = ds_occ[i]
        sn = ds_noc[i]
        if s.name != sn.name:
            raise ValueError(f"scene order mismatch: {s.name} vs {sn.name}")
        gt = s.disparity.astype(np.float64)
        if gt.shape != (368, 1232):
            raise ValueError(f"shape mismatch scene {s.name}")
        valid = gt > 0
        nv = int(valid.sum())
        if nv == 0:
            raise ValueError(f"empty valid mask scene {s.name}")
        occ_full = valid & (sn.disparity <= 0)
        nonocc_full = valid & (~occ_full)

        left = torch.from_numpy(normalize(s.left)).to(device)
        right = torch.from_numpy(normalize(s.right)).to(device)
        ret = model(left, right, return_stages=True)
        if not (isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict)):
            raise ValueError("forward(return_stages=True) did not return tuple")
        out, st = ret
        for k in ("aggregated_cost", "disparity_initial",
                  "refinement_residual", "disparity_final"):
            if k not in st:
                raise ValueError(f"stages dict missing key: {k}")
        final = st["disparity_final"][0, 0].double().cpu().numpy()
        if not np.isfinite(final).all():
            raise ValueError(f"NaN/inf in final scene {s.name}")
        cu = F.interpolate(st["aggregated_cost"], size=(368, 1232),
                           mode="bilinear", align_corners=True)
        cus = ((cu - cu.mean(dim=1, keepdim=True))
               / (cu.std(dim=1, keepdim=True) + 1e-6))
        w = torch.softmax(-cus, dim=1)
        if not torch.isfinite(w).all():
            raise ValueError(f"NaN/inf in softmax scene {s.name}")
        valid_t = torch.from_numpy(valid).to(device)
        Wd = w[0, :, valid_t].double()
        ent_v = -(Wd * torch.log(Wd + 1e-12)).sum(0)
        top2v, _ = torch.topk(Wd, 2, dim=0)
        top1_full = np.zeros_like(gt)
        top1_full[valid] = top2v[0].cpu().numpy()
        marg_full = np.zeros_like(gt)
        marg_full[valid] = (top2v[0] - top2v[1]).cpu().numpy()
        ent_full = np.zeros_like(gt)
        ent_full[valid] = ent_v.cpu().numpy()

        e_full = np.zeros_like(gt)
        e_full[valid] = final[valid] - gt[valid]
        ae_full = np.abs(e_full)
        check_finite(f"seed {seed} scene {s.name} fields",
                     e_full[valid], top1_full[valid])
        epe_sum += float(ae_full[valid].sum())
        n_valid += nv
        n_nonocc += int(nonocc_full.sum())

        # anchor GT bins (full-res map, -1 at invalid)
        bmap = np.full(gt.shape, -1, dtype=np.int64)
        bmap[valid] = np.clip(np.searchsorted(np.array(EDGES), gt[valid],
                                              side="right") - 1, 0, nB - 1)

        # ---- permuted control: permute signed e among valid pixels ----
        rng = np.random.default_rng(1000 + seed * 100000 + i)
        v_idx = np.flatnonzero(valid)
        perm = rng.permutation(v_idx.size)
        e_perm_flat = e_full.flat[v_idx[perm]]
        e_perm_full = np.zeros_like(gt)
        e_perm_full.flat[v_idx] = e_perm_flat
        ae_perm_full = np.abs(e_perm_full)
        # non-occ population permutation (within nonocc pixels)
        n_idx = np.flatnonzero(nonocc_full)
        rng2 = np.random.default_rng(2000 + seed * 100000 + i)
        perm2 = rng2.permutation(n_idx.size)
        e_perm_no = np.zeros_like(gt)
        if n_idx.size:
            e_perm_no.flat[n_idx] = e_full.flat[n_idx[perm2]]
        ae_perm_no = np.abs(e_perm_no)

        # ---- 1. neighbour correlations (valid pairs only) ----
        fields = {"e": (e_full, e_perm_full), "ae": (ae_full, ae_perm_full)}
        fields_no = {"e": (e_full, e_perm_no), "ae": (ae_full, ae_perm_no)}
        for d in OFFSETS:
            # horizontal pairs (x, x+d)
            mh = valid[:, :-d] & valid[:, d:]
            mh_no = nonocc_full[:, :-d] & nonocc_full[:, d:]
            bh = bmap[:, :-d]  # anchor bin
            for fkey, (Ff, Fp) in fields.items():
                x = Ff[:, :-d][mh]
                y = Ff[:, d:][mh]
                get_acc(("all", fkey, "h", d, "real")).add(x, y)
                xp = Fp[:, :-d][mh]
                yp = Fp[:, d:][mh]
                get_acc(("all", fkey, "h", d, "perm")).add(xp, yp)
                bb = bh[mh]
                for bi in range(nB):
                    m = bb == bi
                    if m.any():
                        get_bacc(("all", fkey, "h", d, "real", bi)).add(x[m], y[m])
                        get_bacc(("all", fkey, "h", d, "perm", bi)).add(xp[m], yp[m])
            for fkey, (Ff, Fp) in fields_no.items():
                x = Ff[:, :-d][mh_no]
                y = Ff[:, d:][mh_no]
                get_acc(("nonocc", fkey, "h", d, "real")).add(x, y)
                xp = Fp[:, :-d][mh_no]
                yp = Fp[:, d:][mh_no]
                get_acc(("nonocc", fkey, "h", d, "perm")).add(xp, yp)
                bb = bh[mh_no]
                for bi in range(nB):
                    m = bb == bi
                    if m.any():
                        get_bacc(("nonocc", fkey, "h", d, "real", bi)).add(x[m], y[m])
                        get_bacc(("nonocc", fkey, "h", d, "perm", bi)).add(xp[m], yp[m])
            # vertical pairs (y, y+d)
            mv = valid[:-d, :] & valid[d:, :]
            mv_no = nonocc_full[:-d, :] & nonocc_full[d:, :]
            bv = bmap[:-d, :]
            for fkey, (Ff, Fp) in fields.items():
                x = Ff[:-d, :][mv]
                y = Ff[d:, :][mv]
                get_acc(("all", fkey, "v", d, "real")).add(x, y)
                xp = Fp[:-d, :][mv]
                yp = Fp[d:, :][mv]
                get_acc(("all", fkey, "v", d, "perm")).add(xp, yp)
                bb = bv[mv]
                for bi in range(nB):
                    m = bb == bi
                    if m.any():
                        get_bacc(("all", fkey, "v", d, "real", bi)).add(x[m], y[m])
                        get_bacc(("all", fkey, "v", d, "perm", bi)).add(xp[m], yp[m])
            for fkey, (Ff, Fp) in fields_no.items():
                x = Ff[:-d, :][mv_no]
                y = Ff[d:, :][mv_no]
                get_acc(("nonocc", fkey, "v", d, "real")).add(x, y)
                xp = Fp[:-d, :][mv_no]
                yp = Fp[d:, :][mv_no]
                get_acc(("nonocc", fkey, "v", d, "perm")).add(xp, yp)
                bb = bv[mv_no]
                for bi in range(nB):
                    m = bb == bi
                    if m.any():
                        get_bacc(("nonocc", fkey, "v", d, "real", bi)).add(x[m], y[m])
                        get_bacc(("nonocc", fkey, "v", d, "perm", bi)).add(xp[m], yp[m])
            # top-1 neighbour correlation (real only, valid pairs)
            for dd in CONF_OFFSETS:
                if dd != d:
                    continue
                xh = top1_full[:, :-d][mh]
                yh = top1_full[:, d:][mh]
                get_tacc(("all", d)).add(xh, yh)
                xh2 = top1_full[:, :-d][mh_no]
                yh2 = top1_full[:, d:][mh_no]
                get_tacc(("nonocc", d)).add(xh2, yh2)
                xv = top1_full[:-d, :][mv]
                yv = top1_full[d:, :][mv]
                get_tacc(("all_v", d)).add(xv, yv)
                xv2 = top1_full[:-d, :][mv_no]
                yv2 = top1_full[d:, :][mv_no]
                get_tacc(("nonocc_v", d)).add(xv2, yv2)

        # ---- 2. connected components of |e| > 3 ----
        for pop, pmask, pperm_ae in (("all", valid, ae_perm_full),
                                     ("nonocc", nonocc_full, ae_perm_no)):
            errm = pmask & (ae_full > ERR_THRESH)
            lab, nn = label_components(errm)
            comp_n_per_image[pop].append(nn)
            if nn:
                cnt = np.bincount(lab[errm])
                # bincount includes background only if label 0 present in
                # errm pixels -- it is not (errm pixels have labels >= 1)
                comp_sizes[pop].extend([int(c) for c in cnt if c > 0])
            perrm = pmask & (pperm_ae > ERR_THRESH)
            labp, nnp = label_components(perrm)
            comp_n_per_image_perm[pop].append(nnp)
            if nnp:
                cntp = np.bincount(labp[perrm])
                comp_sizes_perm[pop].extend([int(c) for c in cntp if c > 0])

        # ---- 3/4. edge distance + texture (per-valid-pixel vectors) ----
        edist = edge_distance_map(gt, valid)
        grad = texture_magnitude(s.left)
        check_finite(f"seed {seed} scene {s.name} edist/grad",
                     edist[valid], grad[valid])
        L_ae.append(ae_full[valid])
        L_gt.append(gt[valid])
        L_occ.append(occ_full[valid])
        L_top1.append(top1_full[valid])
        L_margin.append(marg_full[valid])
        L_ent.append(ent_full[valid])
        L_grad.append(grad[valid])
        L_edist.append(edist[valid])

        del ret, out, st, final, cu, cus, w, Wd, ent_v, top2v
        if device == "cuda" and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or (i + 1) == n_img:
            print(f"seed {seed} [{i + 1}/{n_img}] valid={n_valid}", flush=True)

    ae = np.concatenate(L_ae).astype(np.float64)
    gtv = np.concatenate(L_gt).astype(np.float64)
    occ = np.concatenate(L_occ).astype(bool)
    top1 = np.concatenate(L_top1).astype(np.float64)
    margin = np.concatenate(L_margin).astype(np.float64)
    ent = np.concatenate(L_ent).astype(np.float64)
    grad = np.concatenate(L_grad).astype(np.float64)
    edist = np.concatenate(L_edist).astype(np.float64)
    check_finite(f"seed {seed} concat", ae, gtv, top1, margin, ent, grad, edist)
    del L_ae, L_gt, L_occ, L_top1, L_margin, L_ent, L_grad, L_edist
    nonocc = ~occ
    epe = float(epe_sum / n_valid)

    res: dict = {"valid_pixels": int(n_valid),
                 "nonocc_pixels": int(nonocc.sum()),
                 "occ_pixels": int(occ.sum()),
                 "occ_frac_of_valid": float(occ.mean()),
                 "epe": epe,
                 "d1": d1_pct(ae, gtv)}

    # ---- 1. correlations out ----
    corr_out: dict = {}
    for pop in pops:
        for fkey in ("e", "ae"):
            for ori in ("h", "v"):
                for d in OFFSETS:
                    for kind in ("real", "perm"):
                        r = acc.get((pop, fkey, ori, d, kind))
                        corr_out.setdefault(pop, {}).setdefault(fkey, {}) \
                            .setdefault(ori, {})[f"{d}_{kind}"] = (
                                r.result() if r is not None
                                else {"n_pairs": 0, "r": None,
                                      "status": "NO_PAIRS"})
    res["correlations"] = corr_out
    # per-bin correlations
    bin_corr = []
    for bi, lab in enumerate(LABELS):
        row: dict = {"bin": lab}
        m = (np.clip(np.searchsorted(np.array(EDGES), gtv, side="right") - 1,
                                    0, nB - 1) == bi)
        row["px"] = int(m.sum())
        if int(m.sum()) == 0:
            row["status"] = "EMPTY"
            bin_corr.append(row)
            continue
        if int(m.sum()) < LOW_N:
            row["low_n"] = True
        for pop in pops:
            for fkey in ("e", "ae"):
                for ori in ("h", "v"):
                    for d in CONF_OFFSETS:
                        for kind in ("real", "perm"):
                            a = bacc.get((pop, fkey, ori, d, kind, bi))
                            row.setdefault(pop, {}).setdefault(fkey, {}) \
                                .setdefault(ori, {})[f"{d}_{kind}"] = (
                                    a.result() if a is not None
                                    else {"n_pairs": 0, "r": None,
                                          "status": "NO_PAIRS"})
        bin_corr.append(row)
    res["correlations_per_bin"] = bin_corr

    # autocorrelation length: first d where r < half r(d=1), real only
    acl: dict = {}
    for pop in pops:
        for fkey in ("e", "ae"):
            for ori in ("h", "v"):
                r1 = corr_out[pop][fkey][ori]["1_real"]["r"]
                if r1 is None or not (r1 > 0):
                    acl.setdefault(pop, {}).setdefault(fkey, {})[ori] = {
                        "r_d1": r1, "length_px": None,
                        "status": "R_D1_NONPOSITIVE_OR_UNDEFINED"}
                    continue
                L = None
                for d in OFFSETS:
                    rr = corr_out[pop][fkey][ori][f"{d}_real"]["r"]
                    if rr is not None and rr < 0.5 * r1:
                        L = d
                        break
                acl.setdefault(pop, {}).setdefault(fkey, {})[ori] = {"r_d1": r1, "length_px": L,
                                       "length_status": ("falls_below_half_within_32"
                                                         if L is not None else
                                                         "beyond_32px")}
    res["autocorr_length"] = acl

    # ---- 2. components out ----
    comp_out: dict = {}
    for pop in pops:
        arr = np.asarray(comp_sizes[pop], dtype=np.float64)
        arrp = np.asarray(comp_sizes_perm[pop], dtype=np.float64)
        ni = np.asarray(comp_n_per_image[pop], dtype=np.float64)
        nip = np.asarray(comp_n_per_image_perm[pop], dtype=np.float64)
        n_err_px = int(((ae > ERR_THRESH) & (nonocc if pop == "nonocc" else np.ones_like(nonocc, dtype=bool))).sum())
        comp_out[pop] = {
            "real": {**component_size_stats(arr, n_img),
                     "mean_components_per_image": float(ni.mean()),
                     "n_error_px_pop": n_err_px},
            "permuted": {**component_size_stats(arrp, n_img),
                         "mean_components_per_image": float(nip.mean())},
            "connectivity": ("scipy.ndimage.label 8-connectivity"
                             if _HAVE_SCIPY else "union-find 8-connectivity fallback"),
        }
    res["components"] = comp_out

    # ---- 3. edge proximity ----
    edge_rows = []
    for j in range(len(EDGE_BIN_EDGES) - 1):
        lo, hi = EDGE_BIN_EDGES[j], EDGE_BIN_EDGES[j + 1]
        lab = f"[{int(lo)},{int(hi)})" if np.isfinite(hi) else f"[{int(lo)},inf)"
        row = {"bin": lab, "lo": float(lo),
               "hi": float(hi) if np.isfinite(hi) else None}
        for pop, pm in (("all", np.ones_like(nonocc, dtype=bool)),
                        ("nonocc", nonocc)):
            m = pm & (edist >= lo) & (edist < hi if np.isfinite(hi) else True)
            n = int(m.sum())
            cell: dict = {"px": n}
            if n == 0:
                cell["status"] = "EMPTY"
            else:
                if n < LOW_N:
                    cell["low_n"] = True
                cell["mean_abs_err"] = float(ae[m].mean())
                cell["d1"] = d1_pct(ae[m], gtv[m])
            row[pop] = cell
        edge_rows.append(row)
    res["edge_proximity"] = edge_rows

    # ---- 4. texture ----
    tex_out: dict = {}
    for pop, pm in (("all", np.ones_like(nonocc, dtype=bool)),
                    ("nonocc", nonocc)):
        g = grad[pm]
        a = ae[pm]
        gv = gtv[pm]
        med = float(np.median(g))
        dec = float(np.percentile(g, 10))
        lo = g < med
        hi_m = ~lo
        ld = g <= dec
        cell = {
            "median_thresh": med, "lowest_decile_thresh": dec,
            "below_median": {"px": int(lo.sum()),
                             "mean_abs_err": float(a[lo].mean()),
                             "d1": d1_pct(a[lo], gv[lo])},
            "above_median": {"px": int(hi_m.sum()),
                             "mean_abs_err": float(a[hi_m].mean()),
                             "d1": d1_pct(a[hi_m], gv[hi_m])},
            "lowest_decile": {"px": int(ld.sum()),
                              "mean_abs_err": float(a[ld].mean()),
                              "d1": d1_pct(a[ld], gv[ld])},
        }
        tex_out[pop] = cell
    res["texture"] = tex_out
    res["texture_operator"] = ("grayscale luminance Y=0.299R+0.587G+0.114B "
                               "(float64), cv2.Sobel 3x3 ksize=3, "
                               "magnitude sqrt(gx^2+gy^2); window 3x3")

    # ---- 5. confidence ----
    def pearson(x: np.ndarray, y: np.ndarray) -> dict:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = x.size
        if n < 2:
            return {"n": int(n), "r": None, "status": "INSUFFICIENT"}
        xm, ym = x.mean(), y.mean()
        vx = ((x - xm) ** 2).sum()
        vy = ((y - ym) ** 2).sum()
        if vx <= 0 or vy <= 0:
            return {"n": int(n), "r": None, "status": "ZERO_VARIANCE"}
        r = float(((x - xm) * (y - ym)).sum() / np.sqrt(vx * vy))
        if not np.isfinite(r):
            raise ValueError("NaN/inf in confidence correlation")
        return {"n": int(n), "r": r}

    conf_out: dict = {}
    for pop, pm in (("all", np.ones_like(nonocc, dtype=bool)),
                    ("nonocc", nonocc)):
        a = ae[pm]
        t = top1[pm]
        mg = margin[pm]
        en = ent[pm]
        conf_out[pop] = {
            "mean_top1": float(t.mean()), "mean_margin": float(mg.mean()),
            "mean_entropy": float(en.mean()),
            "corr_abserr_top1": pearson(a, t),
            "corr_abserr_margin": pearson(a, mg),
            "corr_abserr_entropy": pearson(a, en),
        }
    res["confidence"] = conf_out
    # mean |e| per top-1 decile (equal-width over [0,1])
    dec_edges = np.linspace(0.0, 1.0, 11)
    for pop, pm in (("all", np.ones_like(nonocc, dtype=bool)),
                    ("nonocc", nonocc)):
        a = ae[pm]
        t = top1[pm]
        di = np.clip(np.digitize(t, dec_edges[1:-1], right=False), 0, 9)
        rows = []
        for dd in range(10):
            m = di == dd
            n = int(m.sum())
            if n == 0:
                rows.append({"decile": dd, "px": 0, "status": "EMPTY"})
                continue
            cell = {"decile": dd, "lo": float(dec_edges[dd]),
                    "hi": float(dec_edges[dd + 1]), "px": n,
                    "mean_abs_err": float(a[m].mean()),
                    "d1": d1_pct(a[m], gtv[pm][m])}
            if n < LOW_N:
                cell["low_n"] = True
            rows.append(cell)
        res.setdefault("confidence_deciles", {})[pop] = rows
    # top-1 spatial smoothness
    tsmooth: dict = {}
    for key, a in tacc.items():
        tsmooth[f"{key[0]}_{key[1]}" if len(key) == 2 else f"{key[0]}_v{key[1]}"] = a.result()
    res["top1_neighbour_corr"] = tsmooth

    # D2 cross-check (full runs only)
    res["d2_crosscheck"] = None
    if limit is None:
        d2 = json.loads((OUT / "cost_distribution.json").read_text())
        d2p = d2["seeds"][str(seed)]["distribution_pooled"]
        xcheck = {}
        for k, arr in (("entropy", ent), ("top1", top1), ("margin", margin)):
            diff = float(arr.mean() - d2p[k])
            xcheck[k] = {"d7": float(arr.mean()), "d2": float(d2p[k]),
                         "diff": diff}
            if abs(diff) > TOL_D2_XCHECK:
                raise ValueError(f"seed {seed}: D2 cross-check failed for {k}: "
                                 f"diff={diff:.2e}")
        res["d2_crosscheck"] = xcheck

    # raw dump (float32 store, float64 compute)
    if limit is None:
        atomic_write_npz(
            RAW / f"d7_s{seed}.npz",
            abs_err=ae.astype(np.float32), gt=gtv.astype(np.float32),
            occluded=occ, top1_prob=top1.astype(np.float32),
            top1_margin=margin.astype(np.float32),
            entropy=ent.astype(np.float32),
            grad_mag=grad.astype(np.float32),
            edge_dist=edist.astype(np.float32))

    print(f"seed {seed} epe={epe:.7f} d1={res['d1']:.4f} "
          f"occ_frac={res['occ_frac_of_valid']:.6f}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"D7 device={device} seeds={seeds} limit={args.limit} "
          f"scipy={_HAVE_SCIPY}", flush=True)

    if args.limit is None:
        smoke = process_seed(0, device, 2)
        print(f"SMOKE seed0 limit=2 epe={smoke['epe']:.7f} (informational)",
              flush=True)
        del smoke

    results: dict[int, dict] = {}
    for seed in seeds:
        results[seed] = process_seed(seed, device, args.limit)

    if args.limit is not None:
        print("limit run: no files written", flush=True)
        return

    # GATE: seed-0 EPE reproduces D0
    e0 = results[0]["epe"]
    ok = abs(e0 - EXPECTED_SEED0_EPE) < TOL_BASELINE_PX
    print(f"GATE seed0 epe={e0:.10f} expected={EXPECTED_SEED0_EPE:.10f} "
          f"diff={e0 - EXPECTED_SEED0_EPE:.2e} -> "
          f"{'PASS' if ok else 'FAIL: STOP'}", flush=True)
    if not ok:
        raise SystemExit("D7 STOP: EPE does not reproduce 1.4149796 on seed 0")
    for seed in seeds:
        if results[seed]["valid_pixels"] != EXPECTED_VALID:
            raise ValueError(f"seed {seed}: valid "
                             f"{results[seed]['valid_pixels']} != {EXPECTED_VALID}")

    def mm(get):
        return float(np.mean([get(results[s]) for s in seeds]))

    rf = compute_receptive_field()
    out = {
        "conventions": {
            "population": "full-resolution frozen valid mask (GT>0 in "
                          "disp_occ_0), 3802797 px/seed; same pixels as "
                          "D0/D1/D2/D5/D6",
            "occlusion": "valid in disp_occ_0 AND not valid in disp_noc_0 "
                         "(reference_strata.py definition); EVERY metric "
                         "reported on all-valid AND non-occluded-valid",
            "signed_error": "e = pred - GT (px); abs error |e|",
            "neighbour_pairs": "offset d pairs (x,y)-(x+d,y) horizontal and "
                               "(x,y)-(x,y+d) vertical; a pair survives iff "
                               "BOTH pixels are in the population (valid, or "
                               "non-occ-valid for the nonocc pop); per-D1-bin "
                               "pairs are assigned by the ANCHOR (left/top) "
                               "pixel's GT bin",
            "permuted_control": "per image, signed-e values randomly "
                                "permuted among valid pixels (seeded rng "
                                "1000+seed*100000+img), preserving the "
                                "marginal; |e|-perm derived from it; "
                                "nonocc pop permuted separately within "
                                "nonocc pixels (rng 2000+...); the same "
                                "permutation drives permuted correlations "
                                "and permuted components",
            "components": "|e|>3 px binary mask over the population "
                          "(invalid/occ-excluded pixels are background); "
                          "8-connectivity (scipy.ndimage.label, 3x3 ones; "
                          "union-find fallback stated if used)",
            "edge": "discontinuity = valid pixel with a 4-neighbour also "
                    "valid and |GT diff| >= 3 px (both marked); Euclidean "
                    "distance via scipy distance_transform_edt; geometry "
                    "from all valid GT, stats restricted per population",
            "texture": ("grayscale luminance + cv2.Sobel 3x3 magnitude; "
                        "window 3x3; thresholds = per-seed median / 10th "
                        "percentile over the population"),
            "confidence": "standardised-softmax readout recomputed exactly "
                          "as D2/D5 (upsample bilinear align_corners=True, "
                          "standardise, softmax(-cost), top-1/margin/entropy)",
            "method_mean": "arithmetic mean across the 3 seeds",
            "low_n": f"bins with px < {LOW_N} flagged low_n",
        },
        "config": {"downsample_levels": 3, "num_disparities": 24,
                   "cost_volume_shift": "right", "regression_normalize": True,
                   "stride_px": 8},
        "receptive_field": rf,
        "seeds": {str(s): results[s] for s in seeds},
        "method_mean": {
            "epe": mm(lambda r: r["epe"]),
            "d1": mm(lambda r: r["d1"]),
            "occ_frac": mm(lambda r: r["occ_frac_of_valid"]),
        },
        "interpretation": ("Factual statement only, no verdict (step 8 does "
                           "verdicts): compare real vs permuted correlations "
                           "and component statistics, horizontal vs vertical, "
                           "all-valid vs non-occ, and the autocorrelation "
                           "length against the receptive field above."),
        "unmeasurable": "NOT MEASURABLE WITH CURRENT ARTIFACTS: none; every D7 quantity was measured.",
    }
    atomic_write_json(OUT / "spatial_context.json", out)

    fd, tmp = tempfile.mkstemp(dir=str(RAW), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["seed", "pop", "kind", "bin", "px", "frac_px",
                        "mean_abs_err", "d1", "corr_e_h1", "corr_e_h4",
                        "corr_e_h16", "corr_ae_h1", "corr_ae_h4",
                        "corr_ae_h16", "edge_note", "tex_note"])

            def f(x, spec=".6f"):
                return "" if x is None else format(x, spec)

            for s in seeds:
                r = results[s]
                n = r["valid_pixels"]
                for pop, pm_n in (("all", n), ("nonocc", r["nonocc_pixels"])):
                    # pooled row
                    w.writerow([s, pop, "pooled", "ALL", pm_n, "1.000000000",
                                "", "", "", "", "", "", "", "",
                                "", ""])
                for row in r["correlations_per_bin"]:
                    if row.get("status") == "EMPTY":
                        for pop in ("all", "nonocc"):
                            w.writerow([s, pop, "bin", row["bin"], 0, "0",
                                        *["" for _ in range(10)]])
                        continue
                    for pop in ("all", "nonocc"):
                        c = row[pop]
                        w.writerow([
                            s, pop, "bin", row["bin"], row["px"],
                            f"{row['px'] / n:.9f}", "", "",
                            f(c["e"]["h"].get("1_real", {}).get("r")),
                            f(c["e"]["h"].get("4_real", {}).get("r")),
                            f(c["e"]["h"].get("16_real", {}).get("r")),
                            f(c["ae"]["h"].get("1_real", {}).get("r")),
                            f(c["ae"]["h"].get("4_real", {}).get("r")),
                            f(c["ae"]["h"].get("16_real", {}).get("r")),
                            "", ""])
                for row in r["edge_proximity"]:
                    for pop in ("all", "nonocc"):
                        c = row[pop]
                        w.writerow([s, pop, "edge", row["bin"], c.get("px", 0),
                                    f"{c.get('px', 0) / n:.9f}" if n else "",
                                    f(c.get("mean_abs_err")),
                                    f(c.get("d1"), ".4f"),
                                    "", "", "", "", "", "", "", ""])
                for dd in r["confidence_deciles"]["all"]:
                    w.writerow([s, "all", "top1dec", f"d{dd.get('decile')}",
                                dd.get("px", 0),
                                f"{dd.get('px', 0) / n:.9f}" if n else "",
                                f(dd.get("mean_abs_err")),
                                f(dd.get("d1"), ".4f"),
                                "", "", "", "", "", "", "", ""])
        os.replace(tmp, RAW / "d7_bins.csv")
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print("wrote spatial_context.json + raw/d7_s*.npz + raw/d7_bins.csv",
          flush=True)


if __name__ == "__main__":
    main()
