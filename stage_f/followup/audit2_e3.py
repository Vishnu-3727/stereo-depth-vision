"""Stage-F F2 follow-up: audit-2 (E3 error-mass + cost-volume diagnostics).

Zero-compute: inference only, trains nothing. Implements plan.md sections 2-5.

Subjects: E3 FINAL seeds 0/1/2 on hailo_val (40 scenes, contract) with the
section-3 error-mass tables; plus the train split hailo_calib (160 scenes) for
seed0 FINAL with the section-3 error-mass tables and section-4 diagnostics.
Train-fit numbers use valid = 0<gt<184; the plain gt>0 EPE is also reported
for gate G6. Contract subjects use the contract mask (gt>0, disp_occ_0).

Reuse by import from stage_f/audit_e3.py (never reimplemented here):
  load_model, ckpt_path, EXPECTED_SHA, EXPECTED_PARAMS, EXPECTED_VALID,
  DATA_ROOT, Kitti2015Stereo, normalize, pad_and_crop, atomic_write_json,
  git_head, sha256_file, and (via audit_e3's own imports) bdiag
  (gt_discontinuity, SPACING, MAXD) and d7 (texture Sobel).
NOTE on texture: audit_e3 measures texture with d7.texture_magnitude (Sobel
  3x3 on luminance) -- it IS a plain function, so it is reused as
  a3.d7.texture_magnitude; nothing is copied.
Peak/mode convention replicates audit_e3's is_peak rule exactly:
  (Pv > left_nb) & (Pv >= right_nb) with -inf-padded ends, i.e. an end bin
  counts as a peak iff strictly greater than its single neighbour.
Standardised softmax replicates audit_e3 exactly: cost upsampled bilinear
  align_corners=True to 368x1232, standardised with the torch default std
  (unbiased) + 1e-6 eps, softmax(-cn).

Gates G1-G6 (plan section 5): G1 SHA256 == EXPECTED_SHA (asserted inside
load_model); G2 params == 397954 (asserted inside load_model); G3 recomputed
final EPE per seed == stage_e_recipe/e3_verdict.json per_seed_final within
1e-6 and valid px == 3802797; G4 recomputed soft-argmin == model init within
max-abs 1e-6; G5 error-bin and GT-bin masses sum to pooled EPE within 1e-9;
G6 train-split seed0 gt>0 EPE reproduces audit_e3.json train value
0.9170082187690973 within 1e-6. Any gate failure -> STOP.json + exit 1.
Smoke runs (--limit) still compute every gate value and record them in the
smoke JSON; G3/G6 are recorded as SMOKE_NOT_ENFORCED (expected to fail on a
1-scene subset, no stop()), while G4/G5 are enforced in smoke too (STOP.json
on failure -- they must hold on any subset).

Memory: scene by scene, torch.no_grad(), float32 on GPU, GPU tensors freed
per scene; pooled valid-pixel vectors are float32 / uint8 numpy (~4M px per
contract subject, ~15M for train -- well under 3 GB host RAM).

Usage (from repo root, with the project python):
  python stage_f/followup/audit2_e3.py [--device cuda|cpu] [--limit N]
         [--selfcheck]
Outputs stage_f/followup/audit2/{audit2.json, per_scene.csv, run.log}
(or stage_f/followup/audit2_smoke/ with --limit). run.log is written by an
in-script tee of stdout, so plain `python ...` (no outer tee) suffices.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage_f"))
import audit_e3 as a3  # noqa: E402  (reuse by import; see docstring)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

bdiag = a3.bdiag
d7 = a3.d7

OUT_DIR = REPO / "stage_f" / "followup" / "audit2"
SMOKE_DIR = REPO / "stage_f" / "followup" / "audit2_smoke"
VERDICT_PATH = REPO / "stage_e_recipe" / "e3_verdict.json"

GATE_TOL = 1e-6
MASS_TOL = 1e-9
G6_TARGET = 0.9170082187690973  # audit_e3.json train split epe (seed0 final)
G6_TOL = 1e-6

ERR_EDGES = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0, np.inf])
ERR_NAMES = ["[0,0.25)", "[0.25,0.5)", "[0.5,1)", "[1,2)", "[2,3)", "[3,inf)"]
GT_EDGES = np.array([0.0, 4.0, 8.0, 16.0, 32.0, 64.0, np.inf])
GT_NAMES = ["[0,4)", "[4,8)", "[8,16)", "[16,32)", "[32,64)", "[64,inf)"]
H_IMG, W_IMG = 368, 1232
UNCH_THRESH = 1e-6
MODE_P_THRESH = 0.05
SVD_CAP = 200_000
SVD_RNG_SEED = 1234
M1_B = 0.5
M1_PEAK_T_INT = 8.0
M1_PEAK_T_HALF = 8.5

if float(bdiag.SPACING) != 8.0:  # coarse = 8 * disparity_initial; factor check
    raise SystemExit("ABORT: bdiag.SPACING=%r != 8.0" % (bdiag.SPACING,))


def err_bin_idx(e):
    return np.clip(np.searchsorted(ERR_EDGES, e, side="right") - 1, 0, 5)


def gt_bin_idx(gt):
    return np.clip(np.searchsorted(GT_EDGES, gt, side="right") - 1, 0, 5)


def count_modes(Pv, thr=MODE_P_THRESH):
    """Local-maxima count along k with p >= thr (audit_e3 is_peak rule)."""
    Pv = np.asarray(Pv, dtype=np.float64)
    n = Pv.shape[1]
    left_nb = np.concatenate([np.full((1, n), -np.inf), Pv[:-1]], axis=0)
    right_nb = np.concatenate([Pv[1:], np.full((1, n), -np.inf)], axis=0)
    is_peak = (Pv > left_nb) & (Pv >= right_nb) & (Pv >= thr)
    return is_peak.sum(axis=0).astype(np.int64)


def target_ranks(Pv, ks):
    """1-based rank of the target bin under p (1 = top)."""
    Pv = np.asarray(Pv, dtype=np.float64)
    ks = np.asarray(ks, dtype=np.int64)
    col = np.arange(Pv.shape[1])
    return (1 + (Pv > Pv[ks, col][None, :]).sum(axis=0)).astype(np.int64)


def column_entropy(Pv):
    Pv = np.asarray(Pv, dtype=np.float64)
    return np.where(Pv > 0, -Pv * np.log(Pv), 0.0).sum(axis=0)


def analytic_peak_cap(n=24, unbiased=True):
    """Attainable peak softmax prob under the exact standardisation +
    softmax(-z) the model uses (torch default unbiased std, eps 1e-6):
    build a one-hot cost vector in both orientations, run the exact ops,
    take the overall max. The max is attained at the single LOW-cost bin
    (high cost -> low probability, so orientation [1,0,...] peaks at only
    ~0.04 on the zero bins). NOTE: the plan quotes approx 0.867, which is
    the population-std (biased) value 0.8663; the exact unbiased-std value
    is ~0.8535. Both are recorded; both exceed the M1 peaks either way."""
    best = 0.0
    for v in (torch.tensor([1.0] + [0.0] * (n - 1), dtype=torch.float64),
              torch.tensor([0.0] + [1.0] * (n - 1), dtype=torch.float64)):
        m = v.mean()
        s = v.std(unbiased=unbiased)  # torch default unbiased=True
        z = (v - m) / (s + 1e-6)
        p = torch.softmax(-z, dim=0)
        best = max(best, float(p.max()))
    return best


def m1_peak(t, b=M1_B, n=24):
    """Peak probability of the M1 target q_k propto exp(-|k-t|/b)."""
    k = np.arange(n, dtype=np.float64)
    q = np.exp(-np.abs(k - t) / b)
    q /= q.sum()
    return float(q.max())


def shift_right_features(x, k):
    """Shift RIGHT features right by k (zeros on the left). Matches the E3
    model cost_volume shift='right' convention (bundle_e3
    cost_volume.shift_right: cost_k(x) = left(x) - right(x-k)); this 3-line
    pad/slice is written locally with that citation."""
    if k == 0:
        return x
    w = x.shape[-1]
    return F.pad(x, (k, 0))[..., :w]


def svd_report(X):
    """Singular spectrum + effective rank + energy ranks of centered X."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    e = s * s
    tot = float(e.sum())
    out = {"n_samples": int(X.shape[0]), "n_channels": int(X.shape[1]),
           "singular_values": [float(v) for v in s]}
    if tot > 0:
        p = e / tot
        pnz = p[p > 0]
        out["effective_rank"] = float(np.exp(-(pnz * np.log(pnz)).sum()))
        cum = np.cumsum(p)
        for thr, key in ((0.90, "rank_90"), (0.95, "rank_95"), (0.99, "rank_99")):
            out[key] = int(np.searchsorted(cum, thr) + 1)
    else:
        out["effective_rank"] = 0.0
        out["rank_90"] = out["rank_95"] = out["rank_99"] = 0
    return out


def stop(out_dir, prov, reason):
    out = {"status": "STOP", "provenance": prov, "reason": reason,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    a3.atomic_write_json(out_dir / "STOP.json", out)
    print("STOP: " + reason, flush=True)
    raise SystemExit(1)


@torch.no_grad()
def run_subject(seed, split, train_fit, device, limit, scene_rows,
                want_ops, want_svd):
    """One subject = one ckpt on one split. Returns (rec, pool, diag)."""
    model, sha = a3.load_model(seed, "final", device)
    ds_occ = a3.Kitti2015Stereo(a3.DATA_ROOT, split=split,
                                disparity_scale=256.0, occluded=True)
    ds_noc = a3.Kitti2015Stereo(a3.DATA_ROOT, split=split,
                                disparity_scale=256.0, occluded=False)
    n_img = len(ds_occ) if limit is None else min(limit, len(ds_occ))

    try:
        import cv2  # noqa: F401
        have_cv2 = True
    except Exception:
        have_cv2 = False
    fg_ok = True

    E, GT, SE, CE, TEX, IDX, FG, OCC, DIS = ([] for _ in range(9))
    D_ENT, D_RANK, D_PM, D_LM = ([] for _ in range(4))
    D_MOD, D_EA, D_OUT, D_P0, E_SUB = ([] for _ in range(5))
    max_p = 0.0
    e_init_xcheck = 0.0
    n_init_xcheck = 0
    sanity_max = 0.0
    n_gt0, s_gt0 = 0, 0.0
    n_fit, s_fit = 0, 0.0
    op_acc = {k: {"n": 0, "rank1": 0, "marg": 0.0} for k in ("l1", "dot", "cos")}
    feat_list, feat_valid_total = [], 0

    for i in range(n_img):
        s = ds_occ[i]
        sn = ds_noc[i]
        if s.name != sn.name:
            raise ValueError("scene order mismatch %s vs %s" % (s.name, sn.name))
        if s.left.shape[:2] != (H_IMG, W_IMG):
            raise ValueError("unexpected frame size %r" % (s.left.shape[:2],))
        gt = s.disparity.astype(np.float64)
        if train_fit:
            valid_fit = (gt > 0) & (gt < float(bdiag.MAXD))
        else:
            valid_fit = gt > 0
        valid_gt0 = gt > 0
        nv_fit = int(valid_fit.sum())
        noc_map = sn.disparity
        occluded = valid_gt0 & (noc_map <= 0)

        if have_cv2:
            import cv2
            obj = cv2.imread(str(a3.DATA_ROOT / "training" / "obj_map" / s.name),
                             cv2.IMREAD_UNCHANGED)
            if obj is None:
                fg_ok = False
                fg_full = np.zeros_like(valid_gt0)
            else:
                fg_full = (a3.pad_and_crop(obj) > 0) & valid_gt0
        else:
            fg_ok = False
            fg_full = np.zeros_like(valid_gt0)

        left_t = torch.from_numpy(a3.normalize(s.left)).to(device)
        right_t = torch.from_numpy(a3.normalize(s.right)).to(device)
        pred_t, st = model(left_t, right_t, return_stages=True)
        cost = st["aggregated_cost"]
        init_t = st["disparity_initial"]
        final_t = st["disparity_final"]
        if not (torch.isfinite(pred_t).all() and torch.isfinite(init_t).all()
                and torch.isfinite(final_t).all() and torch.isfinite(cost).all()):
            raise ValueError("NaN/inf in stages scene %s" % s.name)

        # G4 sanity: identical standardise + softmax(-cn) reconstruction
        # (replicates audit_e3 lines 411-424).
        cu = F.interpolate(cost.float(), size=(H_IMG, W_IMG),
                           mode="bilinear", align_corners=True)
        cn = ((cu - cu.mean(dim=1, keepdim=True))
              / (cu.std(dim=1, keepdim=True) + 1e-6))
        pw = torch.softmax(-cn, dim=1)
        if not torch.isfinite(pw).all():
            raise ValueError("NaN/inf in readout softmax scene %s" % s.name)
        nD = pw.shape[1]
        cand = torch.arange(nD, dtype=pw.dtype, device=device).view(1, nD, 1, 1)
        init_recon = (pw * cand).sum(dim=1, keepdim=True)
        sanity = float((init_recon - init_t).abs().max())
        sanity_max = max(sanity_max, sanity)

        final = final_t[0, 0].cpu().numpy().astype(np.float64)
        init_c = init_t[0, 0].cpu().numpy().astype(np.float64)
        coarse = 8.0 * init_c  # factor 8 == bdiag.SPACING (checked at import)
        gv_fit = gt[valid_fit]
        e_fit = np.abs(final[valid_fit] - gv_fit)
        se_fit = final[valid_fit] - gv_fit
        ce_fit = coarse[valid_fit] - gv_fit
        n_fit += nv_fit
        s_fit += float(e_fit.sum())
        e_gt0 = np.abs(final[valid_gt0] - gt[valid_gt0])
        n_gt0 += int(valid_gt0.sum())
        s_gt0 += float(e_gt0.sum())

        E.append(e_fit.astype(np.float32))
        GT.append(gv_fit.astype(np.float32))
        SE.append(se_fit.astype(np.float32))
        CE.append(ce_fit.astype(np.float32))
        TEX.append(d7.texture_magnitude(s.left)[valid_fit].astype(np.float32))
        IDX.append(np.flatnonzero(valid_fit).astype(np.int64))
        FG.append(fg_full[valid_fit].astype(np.uint8))
        OCC.append(occluded[valid_fit].astype(np.uint8))
        disc_full = bdiag.gt_discontinuity(gt.astype(np.float32), valid_gt0)
        DIS.append(disc_full[valid_fit].astype(np.uint8))

        # ---- cost-volume diagnostics on valid & GT<64 ----
        sub = valid_fit & (gt < 64.0)
        has_sub = bool(sub.any())
        if has_sub:
            gsub = gt[sub]
            pwf = pw[0].reshape(nD, -1)
            cnf = cn[0].reshape(nD, -1)
            cols = torch.from_numpy(sub.reshape(-1)).to(device)
            Pv = pwf[:, cols].double().cpu().numpy()
            Lg = (-cnf[:, cols].double()).cpu().numpy()
            ks = np.clip(np.round(gsub / float(bdiag.SPACING)),
                         0, nD - 1).astype(np.int64)
            col = np.arange(Pv.shape[1])
            ent = column_entropy(Pv)
            ranks = target_ranks(Pv, ks)
            o1 = np.argsort(-Pv, axis=0)
            pm = Pv[o1[0], col] - Pv[o1[1], col]
            lo1 = np.argsort(-Lg, axis=0)
            lm = Lg[lo1[0], col] - Lg[lo1[1], col]
            modes = count_modes(Pv)
            Ek = (Pv * np.arange(nD)[:, None]).sum(axis=0)
            amax = Pv.argmax(axis=0)
            ea = np.abs(Ek - amax)
            win = np.abs(np.arange(nD)[:, None] - amax[None, :]) <= 1
            outm = 1.0 - (Pv * win).sum(axis=0)
            p0 = Pv[0].copy()
            max_p = max(max_p, float(Pv.max()))
            e_init_xcheck += float(np.abs(Ek - init_c[sub]).sum())
            n_init_xcheck += int(sub.sum())
            D_ENT.append(ent.astype(np.float32))
            D_RANK.append(ranks.astype(np.int64))
            D_PM.append(pm.astype(np.float32))
            D_LM.append(lm.astype(np.float32))
            D_MOD.append(modes.astype(np.int64))
            D_EA.append(ea.astype(np.float32))
            D_OUT.append(outm.astype(np.float32))
            D_P0.append(p0.astype(np.float32))
            E_SUB.append(np.abs(final[sub] - gsub).astype(np.float32))

        # ---- frozen operator comparison (seed0 contract only) ----
        if want_ops:
            fl = st["left_features"]
            fr = st["right_features"]
            gtf = torch.from_numpy(gt).to(device).float()[None, None]
            gds = F.interpolate(gtf, size=fl.shape[-2:], mode="nearest")[0, 0]
            gds_np = gds.cpu().numpy().astype(np.float64)
            vloc = (gds_np > 0) & (gds_np < 64.0)
            if vloc.any():
                kt = np.clip(np.round(gds_np[vloc] / 8.0),
                             0, nD - 1).astype(np.int64)
                vm = torch.from_numpy(vloc).to(device)
                l1v = torch.empty((nD,) + fl.shape[-2:], device=device,
                                  dtype=torch.float32)
                dotv = torch.empty_like(l1v)
                cosv = torch.empty_like(l1v)
                nL = (fl * fl).sum(1, keepdim=True).sqrt()
                for kk in range(nD):
                    frs = shift_right_features(fr, kk)
                    l1v[kk] = -(fl - frs).abs().sum(1)[0]
                    dv = (fl * frs).sum(1)[0]
                    dotv[kk] = dv
                    nR = (frs * frs).sum(1, keepdim=True).sqrt()[0]
                    cosv[kk] = dv / (nL[0] * nR + 1e-6)
                kt_t = torch.from_numpy(kt).to(device)
                for name, vv in (("l1", l1v), ("dot", dotv), ("cos", cosv)):
                    sc = vv[:, vm].double().cpu().numpy()
                    sd = float(sc.std())
                    r1 = int((sc.argmax(axis=0) == kt).sum())
                    o = np.argsort(-sc, axis=0)
                    mg = (sc[o[0], np.arange(sc.shape[1])]
                          - sc[o[1], np.arange(sc.shape[1])])
                    op_acc[name]["n"] += int(vm.sum())
                    op_acc[name]["rank1"] += r1
                    op_acc[name]["marg"] += float((mg / (sd + 1e-12)).sum())
                    del sc
                del l1v, dotv, cosv, kt_t
            del gtf, gds

        # ---- SVD feature collector (seed0 only) ----
        if want_svd:
            fl_np = st["left_features"].float().cpu().numpy()[0]  # (32,h,w)
            gtf2 = torch.from_numpy(gt).float()[None, None]
            gds2 = F.interpolate(gtf2, size=fl_np.shape[-2:],
                                 mode="nearest")[0, 0].numpy()
            if train_fit:
                vds = (gds2 > 0) & (gds2 < float(bdiag.MAXD))
            else:
                vds = gds2 > 0
            feat_valid_total += int(vds.sum())
            if vds.any():
                feat_list.append(fl_np[:, vds].T.copy())
            del gtf2, gds2

        scene_rows.append({
            "split": split, "seed": seed, "tag": "final", "scene": s.name,
            "valid_px": nv_fit,
            "epe": float(e_fit.mean()) if nv_fit else None,
            "gtlt64_epe": (float(e_fit[gv_fit < 64.0].mean())
                           if (gv_fit < 64.0).any() else None),
            "init8_epe": (float(np.abs(coarse[valid_fit] - gv_fit).mean())
                          if nv_fit else None),
            "sanity_max_abs": sanity,
            "occ_epe": (float(e_gt0[occluded[valid_gt0]].mean())
                        if occluded[valid_gt0].any() else None),
            "noc_epe": (float(e_gt0[(~occluded)[valid_gt0]].mean())
                        if (~occluded[valid_gt0]).any() else None),
            "occ_frac": (float(occluded[valid_gt0].mean())
                         if valid_gt0.any() else None),
            "mean_gt": float(gv_fit.mean()) if nv_fit else None,
        })
        del pred_t, st, cost, init_t, final_t, cu, cn, pw
        del left_t, right_t, init_recon
        if has_sub:
            del Pv, Lg
        if device == "cuda" and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or (i + 1) == n_img:
            print("seed %d/final %s [%d/%d] epe_sofar=%.4f"
                  % (seed, split, i + 1, n_img, s_fit / n_fit), flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    pool = {
        "e": np.concatenate(E).astype(np.float64),
        "gt": np.concatenate(GT).astype(np.float64),
        "se": np.concatenate(SE).astype(np.float64),
        "ce": np.concatenate(CE).astype(np.float64),
        "tex": np.concatenate(TEX).astype(np.float64),
        "idx": np.concatenate(IDX),
        "fg": np.concatenate(FG),
        "occ": np.concatenate(OCC),
        "dis": np.concatenate(DIS),
    }
    diag = None
    if D_ENT:
        diag = {
            "ent": np.concatenate(D_ENT).astype(np.float64),
            "rank": np.concatenate(D_RANK),
            "pm": np.concatenate(D_PM).astype(np.float64),
            "lm": np.concatenate(D_LM).astype(np.float64),
            "mod": np.concatenate(D_MOD),
            "ea": np.concatenate(D_EA).astype(np.float64),
            "out": np.concatenate(D_OUT).astype(np.float64),
            "p0": np.concatenate(D_P0).astype(np.float64),
            "esub": np.concatenate(E_SUB).astype(np.float64),
        }
    svd = None
    if want_svd and feat_list:
        X = np.concatenate(feat_list, axis=0).astype(np.float32)
        if X.shape[0] > SVD_CAP:
            rng = np.random.default_rng(SVD_RNG_SEED)
            X = X[rng.choice(X.shape[0], SVD_CAP, replace=False)]
        svd = svd_report(X)
        svd["valid_locations_total"] = int(feat_valid_total)
        svd["rng_seed"] = SVD_RNG_SEED
        del X
    del E, GT, SE, CE, TEX, IDX, FG, OCC, DIS
    del D_ENT, D_RANK, D_PM, D_LM, D_MOD, D_EA, D_OUT, D_P0, E_SUB, feat_list
    rec = {
        "seed": seed, "tag": "final", "sha256": sha, "split": split,
        "scenes": n_img, "subset": limit is not None,
        "valid_rule": "0<gt<184" if train_fit else "gt>0",
        "valid_pixels": int(n_fit), "epe": float(s_fit / n_fit),
        "valid_px_gt0": int(n_gt0), "epe_gt0": float(s_gt0 / n_gt0),
        "sanity_softargmin_max_abs": sanity_max,
        "E_vs_init_mean_abs": (float(e_init_xcheck / n_init_xcheck)
                               if n_init_xcheck else None),
        "max_p_observed": max_p,
        "fg_ok": bool(fg_ok),
        "op_acc": op_acc if want_ops else None,
        "svd": svd,
    }
    return rec, pool, diag


def describe(a):
    a = np.asarray(a, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()) if a.size else None,
            "median": float(np.median(a)) if a.size else None}


def section3(pool, n_valid):
    """Error-magnitude + GT-bin audit. Masses are EPE masses (sum/N_valid)."""
    e, gt = pool["e"], pool["gt"]
    eb = err_bin_idx(e)
    gb = gt_bin_idx(gt)
    err_bins = []
    for b in range(6):
        m = eb == b
        c = int(m.sum())
        row = {"bin": ERR_NAMES[b], "px": c,
               "frac_px": float(c / n_valid) if n_valid else 0.0}
        if c:
            row["mean_e"] = float(e[m].mean())
            row["mass"] = float(e[m].sum() / n_valid)
        else:
            row["status"] = "EMPTY"
            row["mass"] = 0.0
        err_bins.append(row)
    gt_bins = []
    for b in range(6):
        m = gb == b
        c = int(m.sum())
        row = {"bin": GT_NAMES[b], "px": c,
               "frac_px": float(c / n_valid) if n_valid else 0.0}
        if c:
            row["epe"] = float(e[m].mean())
            row["mass"] = float(e[m].sum() / n_valid)
        else:
            row["status"] = "EMPTY"
            row["mass"] = 0.0
        gt_bins.append(row)
    mass_matrix = np.zeros((6, 6))
    for gi in range(6):
        for ei in range(6):
            m = (gb == gi) & (eb == ei)
            if m.any():
                mass_matrix[gi, ei] = e[m].sum() / n_valid
    thr = float(np.quantile(e, 0.9))
    top = e >= thr
    dec = {"threshold_p90": thr, "frac_px": float(top.mean()),
           "mass_share": float(e[top].sum() / e.sum())}
    tex = pool["tex"]
    tq = np.quantile(tex, np.linspace(0, 1, 11))
    tdec = np.clip(np.searchsorted(tq[1:-1], tex, side="right"), 0, 9)
    fg = pool["fg"].astype(bool)
    occ = pool["occ"].astype(bool)
    dis = pool["dis"].astype(bool)
    yy = pool["idx"] // W_IMG
    xx = pool["idx"] % W_IMG
    reg = ((yy * 3) // H_IMG) * 3 + ((xx * 3) // W_IMG)
    by = {"gtbin": [
        {"bin": GT_NAMES[b],
         "px_share": float(((gb == b) & top).sum() / top.sum()),
         "mass_share": float(e[(gb == b) & top].sum() / e[top].sum())}
        for b in range(6)]}
    for name, pos in (("fg", fg), ("bg", ~fg), ("occ", occ), ("noc", ~occ),
                      ("disc", dis), ("nondisc", ~dis)):
        m = pos & top
        by[name] = {"px_share": float(m.sum() / top.sum()),
                    "mass_share": float(e[m].sum() / e[top].sum())}
    by["texdec"] = [
        {"decile": d, "lo": float(tq[d]), "hi": float(tq[d + 1]),
         "px_share": float(((tdec == d) & top).sum() / top.sum()),
         "mass_share": float(e[(tdec == d) & top].sum() / e[top].sum())}
        for d in range(10)]
    by["region3x3"] = [
        {"region": int(r),
         "px_share": float(((reg == r) & top).sum() / top.sum()),
         "mass_share": float(e[(reg == r) & top].sum() / e[top].sum())}
        for r in range(9)]
    dec["by"] = by
    dec["tex_quantiles"] = [float(v) for v in tq]
    # coarse vs final: raw delta d = final - coarse; improved/worsened compare
    # |err| and require |d| >= 1e-6 (else unchanged).
    fin_abs = np.abs(pool["se"])
    ce_abs = np.abs(pool["ce"])
    raw = (pool["se"] + pool["gt"]) - (pool["ce"] + pool["gt"])
    unch = np.abs(raw) < UNCH_THRESH
    impr = (~unch) & (fin_abs < ce_abs)
    wors = (~unch) & (fin_abs > ce_abs)
    cf = {"epe_coarse_x8": float(ce_abs.mean()), "epe_final": float(fin_abs.mean()),
          "delta_mean": float(raw.mean()),
          "frac_improved": float(impr.mean()),
          "frac_worsened": float(wors.mean()),
          "frac_unchanged": float(unch.mean()),
          "worsened_mass": float(fin_abs[wors].sum() / n_valid),
          "worsened_excess": float((fin_abs[wors] - ce_abs[wors]).sum() / n_valid),
          "per_gtbin": [], "per_errbin": []}
    for b in range(6):
        m = gb == b
        cf["per_gtbin"].append({
            "bin": GT_NAMES[b], "px": int(m.sum()),
            "epe_coarse_x8": float(ce_abs[m].mean()) if m.any() else None,
            "epe_final": float(fin_abs[m].mean()) if m.any() else None,
            "delta_mean": float(raw[m].mean()) if m.any() else None})
    for b in range(6):
        m = eb == b  # final-error bin
        cf["per_errbin"].append({
            "bin": ERR_NAMES[b], "px": int(m.sum()),
            "epe_coarse_x8": float(ce_abs[m].mean()) if m.any() else None,
            "epe_final": float(fin_abs[m].mean()) if m.any() else None,
            "delta_mean": float(raw[m].mean()) if m.any() else None})
    # signed error tables
    se = pool["se"]
    intbin = np.clip(np.floor(gt).astype(np.int64), 0, 63)
    int_rows = []
    for b in range(64):
        m = (intbin == b) & (gt < 64.0)
        c = int(m.sum())
        row = {"gt_int": b, "px": c}
        if c:
            row["mean_signed"] = float(se[m].mean())
            row["median_signed"] = float(np.median(se[m]))
        int_rows.append(row)
    phi8 = (gt / 8.0) % 1.0
    phi1 = gt % 1.0
    phi8_rows, phi1_rows = [], []
    for b in range(10):
        m8 = (phi8 >= b / 10.0) & (phi8 < (b + 1) / 10.0)
        m1 = (phi1 >= b / 10.0) & (phi1 < (b + 1) / 10.0)
        r8 = {"bin": b, "lo": b / 10.0, "hi": (b + 1) / 10.0, "px": int(m8.sum())}
        if m8.any():
            r8["mean_signed"] = float(se[m8].mean())
            r8["median_signed"] = float(np.median(se[m8]))
        r1 = {"bin": b, "lo": b / 10.0, "hi": (b + 1) / 10.0, "px": int(m1.sum())}
        if m1.any():
            r1["mean_signed"] = float(se[m1].mean())
            r1["median_signed"] = float(np.median(se[m1]))
        phi8_rows.append(r8)
        phi1_rows.append(r1)
    sig_means = np.array([r.get("mean_signed", np.nan) for r in phi8_rows])
    signed = {"vs_int_gt": int_rows, "vs_phi8": phi8_rows, "vs_phi1": phi1_rows,
              "phi8_amplitude": (float(np.nanmax(sig_means) - np.nanmin(sig_means))
                                 if np.isfinite(sig_means).any() else None)}
    return {"err_bins": err_bins, "gt_bins": gt_bins,
            "mass_matrix_6x6": mass_matrix.tolist(),
            "mass_matrix_rows_gt": list(GT_NAMES),
            "mass_matrix_cols_err": list(ERR_NAMES),
            "top_decile": dec, "coarse_final": cf, "signed": signed,
            "p90_threshold": thr}


def section4(diag, err_of_sub, p90):
    """Cost-volume diagnostics: overall, by error bin, top-decile vs rest."""
    keys = (("entropy", "ent"), ("rank", "rank"), ("prob_margin", "pm"),
            ("logit_margin", "lm"), ("modes", "mod"), ("E_minus_argmax", "ea"),
            ("outside_mass", "out"), ("p_k0", "p0"))
    out = {"overall": {n: describe(diag[k]) for n, k in keys},
           "n_sub": int(diag["esub"].size),
           "by_errbin": [], "decile_vs_rest": {}}
    eb = err_bin_idx(err_of_sub)
    for b in range(6):
        m = eb == b
        row = {"bin": ERR_NAMES[b], "px": int(m.sum())}
        for n, k in keys:
            row[n] = describe(diag[k][m]) if m.any() else None
        out["by_errbin"].append(row)
    top = err_of_sub >= p90
    for name, m in (("top_decile", top), ("rest", ~top)):
        cell = {"px": int(m.sum())}
        for n, k in keys:
            cell[n] = describe(diag[k][m]) if m.any() else None
        out["decile_vs_rest"][name] = cell
    return out


def selfcheck():
    print("selfcheck: synthetic asserts, no data", flush=True)
    # 1. masses sum to pooled mean
    rng = np.random.default_rng(0)
    e = np.abs(rng.normal(0, 1.5, 50_000))
    eb = err_bin_idx(e)
    masses = np.array([e[eb == b].sum() / e.size for b in range(6)])
    assert abs(masses.sum() - e.mean()) < 1e-9, (masses.sum(), e.mean())
    gb = gt_bin_idx(np.abs(rng.normal(20, 15, 50_000)) + 0.5)
    print("  [1] err-bin masses sum to mean: PASS", flush=True)
    # 2. mode count on known vectors
    Pv = np.array([[0.6], [0.2], [0.2]])
    assert count_modes(Pv).tolist() == [1]
    Pv = np.array([[0.05], [0.5], [0.05], [0.4], [0.0]])
    assert count_modes(Pv).tolist() == [2], count_modes(Pv)
    Pv = np.array([[0.1], [0.2], [0.3], [0.25], [0.15]])
    assert count_modes(Pv).tolist() == [1]
    Pv = np.array([[0.04], [0.049], [0.04]])  # below thr -> 0
    assert count_modes(Pv).tolist() == [0]
    print("  [2] mode count on known vectors: PASS", flush=True)
    # 3. analytic cap: exact unbiased-std ~0.8535; biased-std ~0.8663
    cap = analytic_peak_cap(24)
    cap_biased = analytic_peak_cap(24, unbiased=False)
    assert 0.84 < cap < 0.87, cap
    assert abs(cap - 0.8535) < 0.005, cap
    assert abs(cap_biased - 0.867) < 0.005, cap_biased
    print("  [3] analytic cap n=24 exact=%.6f biased=%.6f: PASS"
          % (cap, cap_biased), flush=True)
    # 4. M1 peaks below cap
    pi_, ph = m1_peak(M1_PEAK_T_INT), m1_peak(M1_PEAK_T_HALF)
    assert pi_ < cap and ph < cap, (pi_, ph, cap)
    assert abs(pi_ - 0.76) < 0.02, pi_
    print("  [4] M1 peaks int=%.4f half=%.4f < cap: PASS" % (pi_, ph), flush=True)
    # 5. ranks + entropy
    Pv = np.array([[0.1, 0.5], [0.7, 0.3], [0.2, 0.2]])
    assert target_ranks(Pv, np.array([1, 0])).tolist() == [1, 1]
    assert target_ranks(Pv, np.array([0, 2])).tolist() == [3, 3]
    uni = np.full((24, 100), 1.0 / 24)
    assert abs(column_entropy(uni).mean() - np.log(24)) < 1e-9
    print("  [5] ranks + uniform entropy ln24: PASS", flush=True)
    # 6. svd_report on known spectrum
    X = np.eye(4)[:3]
    rep = svd_report(np.tile(X, (10, 8)))
    assert rep["rank_90"] >= 1 and rep["n_channels"] == 32
    print("  [6] svd_report: PASS", flush=True)
    print("selfcheck: ALL PASS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    smoke = args.limit is not None
    out_dir = SMOKE_DIR if smoke else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    log_fh = open(log_path, "w")

    class Tee:
        def __init__(self, *fhs):
            self.fhs = fhs

        def write(self, s):
            for fh in self.fhs:
                fh.write(s)

        def flush(self):
            for fh in self.fhs:
                fh.flush()

    sys.stdout = Tee(sys.stdout, log_fh)
    try:
        _run(device, args.limit, smoke, out_dir)
    finally:
        sys.stdout = sys.stdout.fhs[0]
        log_fh.close()


def _run(device, limit, smoke, out_dir):
    verdict = json.loads(VERDICT_PATH.read_text(encoding="utf-8"))
    prov = {
        "git_head": a3.git_head(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "device": device,
        "argv": list(sys.argv),
        "dataset_splits": ["hailo_val", "hailo_calib"],
        "checkpoint_sha256": {
            "seed%d_final" % s: a3.EXPECTED_SHA[("seed%d" % s, "final")]
            for s in (0, 1, 2)},
        # ckpt file paths via a3.ckpt_path; per-file sha256 via
        # a3.sha256_file is asserted inside a3.load_model (G1) and returned.
        "ckpt_paths": {("seed%d_final" % s): str(a3.ckpt_path(s, "final"))
                       for s in (0, 1, 2)},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("plan: stage-f followup audit2; device=%s smoke=%s" % (device, smoke),
          flush=True)
    subjects = {}
    scene_rows = []
    # contract subjects: seeds 0/1/2 final (ops on seed0, svd on seed0)
    for s in (0, 1, 2):
        print("=== subject seed%d_final contract (%s) ==="
              % (s, "SMOKE" if smoke else "FULL"), flush=True)
        rec, pool, diag = run_subject(
            s, "hailo_val", False, device, limit, scene_rows,
            want_ops=(s == 0), want_svd=(s == 0))
        s3 = section3(pool, rec["valid_pixels"])
        rec["section3"] = s3
        rec["section4"] = section4(diag, pool["e"][(pool["gt"] < 64.0)],
                                   s3["p90_threshold"])
        rec["cap_check"] = {
            "max_p_observed": rec.pop("max_p_observed"),
            "analytic_cap_n24": analytic_peak_cap(24),
            "analytic_cap_n24_biased_std": analytic_peak_cap(24, unbiased=False),
            "cap_note": "plan quotes ~0.867 = biased-std value; exact "
                        "unbiased-std value ~0.8535 (see analytic_peak_cap)",
            "m1_peak_t8_int": m1_peak(M1_PEAK_T_INT),
            "m1_peak_t8p5_half": m1_peak(M1_PEAK_T_HALF),
            "m1_b": M1_B,
        }
        if rec["op_acc"] is not None:
            ops = {}
            for name, a in rec["op_acc"].items():
                ops[name] = {
                    "locations": a["n"],
                    "target_rank1_rate": (a["rank1"] / a["n"] if a["n"] else None),
                    "mean_norm_top1_top2_margin": (a["marg"] / a["n"]
                                                   if a["n"] else None)}
            rec["operators"] = ops
            del rec["op_acc"]
        else:
            del rec["op_acc"]
        subjects["seed%d_final" % s] = (rec, pool, diag)
    # train subject: seed0 final
    print("=== subject seed0_final train (%s) ===" % ("SMOKE" if smoke else "FULL"),
          flush=True)
    rec, pool, diag = run_subject(
        0, "hailo_calib", True, device, limit, scene_rows,
        want_ops=False, want_svd=True)
    s3 = section3(pool, rec["valid_pixels"])
    rec["section3"] = s3
    rec["section4"] = section4(diag, pool["e"][(pool["gt"] < 64.0)],
                               s3["p90_threshold"])
    rec["cap_check"] = {
        "max_p_observed": rec.pop("max_p_observed"),
        "analytic_cap_n24": analytic_peak_cap(24),
        "m1_peak_t8_int": m1_peak(M1_PEAK_T_INT),
        "m1_peak_t8p5_half": m1_peak(M1_PEAK_T_HALF),
        "m1_b": M1_B,
    }
    eb = s3["err_bins"]
    sub1 = eb[0]["mass"] + eb[1]["mass"] + eb[2]["mass"]
    rec["l1_eligibility"] = {"sub1_mass": sub1, "threshold": 0.1676,
                             "eligible": bool(sub1 >= 0.1676)}
    del rec["op_acc"]
    subjects["train_seed0_final"] = (rec, pool, diag)

    # paired seed-to-seed scene deltas (contract, by scene index order)
    by_scene = {}
    for r in scene_rows:
        if r["split"] == "hailo_val":
            by_scene.setdefault(r["scene"], {})[r["seed"]] = r["epe"]
    paired = [{"scene": sc,
               "epe_s0": d.get(0), "epe_s1": d.get(1), "epe_s2": d.get(2),
               "d_s1_s0": (d[1] - d[0] if d.get(0) is not None
                           and d.get(1) is not None else None),
               "d_s2_s0": (d[2] - d[0] if d.get(0) is not None
                           and d.get(2) is not None else None),
               "d_s2_s1": (d[2] - d[1] if d.get(1) is not None
                           and d.get(2) is not None else None)}
              for sc, d in sorted(by_scene.items())]
    prov["valid_pixel_counts"] = {k: v[0]["valid_pixels"]
                                  for k, v in subjects.items()}

    # ---- gates (G3/G6 full-only; G4/G5 enforced always, incl. smoke) ----
    gates = {}
    for s in (0, 1, 2):
        rec = subjects["seed%d_final" % s][0]
        want = verdict["per_seed_final"][s]
        g3e = abs(rec["epe"] - want) <= GATE_TOL
        g3v = rec["valid_pixels"] == a3.EXPECTED_VALID
        g4 = rec["sanity_softargmin_max_abs"] <= GATE_TOL
        e5 = sum(b["mass"] for b in rec["section3"]["err_bins"])
        g5 = sum(b["mass"] for b in rec["section3"]["gt_bins"])
        g5e = abs(e5 - rec["epe"]) <= MASS_TOL
        g5g = abs(g5 - rec["epe"]) <= MASS_TOL
        if smoke:
            gates["seed%d_final" % s] = {
                "G3_epe_match_1e_6": "SMOKE_NOT_ENFORCED",
                "G3_valid_px": "SMOKE_NOT_ENFORCED",
                "G4_softargmin_1e_6": bool(g4),
                "G5_errmass_1e_9": bool(g5e),
                "G5_gtmass_1e_9": bool(g5g)}
        else:
            gates["seed%d_final" % s] = {
                "G3_epe_match_1e_6": bool(g3e),
                "G3_valid_px": bool(g3v),
                "G4_softargmin_1e_6": bool(g4),
                "G5_errmass_1e_9": bool(g5e),
                "G5_gtmass_1e_9": bool(g5g)}
        if not smoke and not (g3e and g3v):
            stop(out_dir, prov,
                 "G3 seed%d: epe %.10f vs verdict %.10f, valid %d"
                 % (s, rec["epe"], want, rec["valid_pixels"]))
        if not g4:
            stop(out_dir, prov,
                 "G4 seed%d: sanity max-abs %.3e > 1e-6" % (s, rec[
                     "sanity_softargmin_max_abs"]))
        if not (g5e and g5g):
            stop(out_dir, prov, "G5 seed%d: mass closure fail" % s)
    trec = subjects["train_seed0_final"][0]
    g4t = trec["sanity_softargmin_max_abs"] <= GATE_TOL
    e5 = sum(b["mass"] for b in trec["section3"]["err_bins"])
    g5t = (abs(e5 - trec["epe"]) <= MASS_TOL
           and abs(sum(b["mass"] for b in trec["section3"]["gt_bins"])
                   - trec["epe"]) <= MASS_TOL)
    g6 = abs(trec["epe_gt0"] - G6_TARGET) <= G6_TOL
    if smoke:
        gates["train_seed0_final"] = {
            "G4_softargmin_1e_6": bool(g4t),
            "G5_mass_1e_9": bool(g5t),
            "G6_train_epe_match_1e_6": "SMOKE_NOT_ENFORCED"}
    else:
        gates["train_seed0_final"] = {
            "G4_softargmin_1e_6": bool(g4t),
            "G5_mass_1e_9": bool(g5t),
            "G6_train_epe_match_1e_6": bool(g6)}
    if not g4t:
        stop(out_dir, prov, "G4 train: sanity max-abs %.3e > 1e-6"
             % trec["sanity_softargmin_max_abs"])
    if not g5t:
        stop(out_dir, prov, "G5 train: mass closure fail")
    if not smoke and not g6:
        stop(out_dir, prov,
             "G6 train: epe_gt0 %.10f != %.10f" % (trec["epe_gt0"], G6_TARGET))
    gates["G1_sha256"] = True  # asserted inside load_model per subject
    gates["G2_params_397954"] = True  # asserted inside load_model
    gates["status"] = "SMOKE_G3_G6_NOT_ENFORCED" if smoke else "PASS"

    out = {
        "status": "SMOKE" if smoke else "OK",
        "experiment": "stage-f F2 follow-up audit2 (zero-compute)",
        "provenance": prov,
        "subjects": {k: v[0] for k, v in subjects.items()},
        "paired_scene_deltas": paired,
        "gates": gates,
        "thresholds": {
            "GATE_TOL": GATE_TOL, "MASS_TOL": MASS_TOL,
            "G6_TARGET": G6_TARGET, "G6_TOL": G6_TOL,
            "L1_SUB1_FLOOR": 0.1676, "M1_B": M1_B, "M1_LAMBDA": 0.1},
        "methods": {
            "valid_contract": "gt>0 (disp_occ_0)",
            "valid_train_fit": "0<gt<184",
            "coarse": "8.0 * disparity_initial (bdiag.SPACING==8.0 checked)",
            "softmax": "cost bilinear align_corners=True to 368x1232, "
                       "torch-default-std standardise eps 1e-6, softmax(-cn)",
            "modes": "local maxima along k with p>=0.05, ends count iff "
                     "strictly greater than single neighbour (audit_e3 rule)",
            "target_bin": "k*=round(gt/8) clipped 0..23",
            "operators": "24-candidate volumes from 1/8 features, "
                         "shift=right (cost_k=left(x)-right(x-k)); margins "
                         "normalised by each operator's own std",
            "svd": "left 1/8 features at nearest-downsampled valid "
                   "locations, <=200k rng 1234, centered, float64 SVD",
            "texture": "a3.d7.texture_magnitude reused (Sobel 3x3); "
                       "equal-count deciles over pooled valid pixels",
        },
        "reused_from_audit_e3": [
            "load_model", "ckpt_path", "EXPECTED_SHA", "EXPECTED_PARAMS",
            "EXPECTED_VALID", "DATA_ROOT", "Kitti2015Stereo", "normalize",
            "pad_and_crop", "bdiag (gt_discontinuity, SPACING, MAXD)",
            "d7 (texture Sobel via d7.texture_magnitude)",
            "atomic_write_json", "git_head", "sha256_file",
            "standardised-softmax + soft-argmin sanity construction"],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    a3.atomic_write_json(out_dir / "audit2.json", out)

    with open(out_dir / "per_scene.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scene_rows[0].keys()))
        w.writeheader()
        w.writerows(scene_rows)
    print("wrote %s + per_scene.csv status=%s" % (out_dir / "audit2.json",
                                                  out["status"]), flush=True)


if __name__ == "__main__":
    main()
