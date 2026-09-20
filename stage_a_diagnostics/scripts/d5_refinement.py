"""STAGE-A D5: REFINEMENT DIAGNOSTIC (frozen model, read-only).

FROZEN-MODEL DIAGNOSTIC. No model change, no retraining, no tuning, no
mechanism verdict (step 8 does verdicts).

Works at FULL RESOLUTION on the frozen valid mask (GT > 0), the same
3,802,797 valid pixels D0/D1/D2 used - NOT the feature-resolution cell set
D3/D4 used. Populations are never mixed in one table.

Units (verified in src/models/stereonet/stereonet.py):
  disparity_initial is in CANDIDATE units (0..23).
  disparity_final   is in PIXELS.
  disparity_final = relu(disparity_initial + refinement_residual).
Two coarse conventions, reported side by side, never silently picked:
  COARSE_RAW    = disparity_initial as-is vs GT in px (units-mismatched,
                  the literal pre-refinement tensor).
  COARSE_SCALED = 8.0 * disparity_initial vs GT in px (physically meaningful;
                  candidate k corresponds to 8k px). HEADLINE convention.
  REFINED       = disparity_final (must reproduce seed-0 1.4149796; GATE).

Outputs (atomic .tmp + rename):
  stage_a_diagnostics/refinement_diagnostic.json
  stage_a_diagnostics/raw/d5_s{0,1,2}.npz
  stage_a_diagnostics/raw/d5_bins.csv
Usage:
  python stage_a_diagnostics/scripts/d5_refinement.py [--limit N] [--seeds 0,1,2]
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
SCALE = 8.0
EDGES = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 184, np.inf]
EXPECTED_SEED0_EPE = 1.4149795736625577
EXPECTED_VALID = 3802797
TOL_BASELINE_PX = 1e-6
TOL_D2_XCHECK = 1e-6
UNCH_THRESH = 1e-6


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


def fmt(x, spec: str) -> str:
    return "" if x is None else format(x, spec)


def bin_label(lo, hi) -> str:
    if np.isfinite(hi):
        return f"[{int(lo)},{int(hi)})"
    return f"[{int(lo)},inf)"


LABELS = [bin_label(lo, hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]


def d1_pct(err: np.ndarray, gt: np.ndarray) -> float:
    return float((((err > 3.0) & (err > 0.05 * gt)).mean()) * 100.0)


def check_finite(name: str, *arrs: np.ndarray) -> None:
    for a in arrs:
        if not np.isfinite(a).all():
            raise ValueError(f"NaN/inf detected in {name}")


def core_stats(init: np.ndarray, resid: np.ndarray, final: np.ndarray,
               gt: np.ndarray) -> dict:
    """All-float64 pooled stats for one pixel population. No model use."""
    err_raw = np.abs(init - gt)
    err_scaled = np.abs(SCALE * init - gt)
    err_ref = np.abs(final - gt)
    improv = err_scaled - err_ref  # >0 means refinement helped
    helped = improv > UNCH_THRESH
    harmed = improv < -UNCH_THRESH
    unch = ~(helped | harmed)
    n = gt.size
    out = {
        "px": int(n),
        "epe_coarse_raw": float(err_raw.mean()),
        "epe_coarse_scaled": float(err_scaled.mean()),
        "epe_refined": float(err_ref.mean()),
        "delta_refined_minus_coarse_scaled": float(err_ref.mean() - err_scaled.mean()),
        "d1_coarse_raw": d1_pct(err_raw, gt),
        "d1_coarse_scaled": d1_pct(err_scaled, gt),
        "d1_refined": d1_pct(err_ref, gt),
        "frac_improved": float(helped.mean()),
        "frac_worsened": float(harmed.mean()),
        "frac_unchanged_1e6": float(unch.mean()),
        "mean_improvement_on_helped": float(improv[helped].mean()) if helped.any() else None,
        "mean_degradation_on_harmed": float((-improv[harmed]).mean()) if harmed.any() else None,
        "residual_mean_signed": float(resid.mean()),
        "residual_mean_abs": float(np.abs(resid).mean()),
        "relu_clamp_frac": float(((init + resid) < 0.0).mean()),
        "frac_resid_sign_agrees_with_needed": float(
            (((resid > 0) & ((gt - SCALE * init) > 0))
             | ((resid < 0) & ((gt - SCALE * init) < 0))
             | ((resid == 0) & ((gt - SCALE * init) == 0))).mean()),
    }
    check_finite("core_stats",
                 np.asarray([v for v in out.values() if isinstance(v, float)]))
    return out


def epe_alpha(a: float, init: np.ndarray, resid: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(np.maximum(a * init + resid, 0.0) - gt).mean())


def epe_beta(b: float, init: np.ndarray, resid: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(np.maximum(init + resid + b, 0.0) - gt).mean())


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
            "epe_at_alpha_1": epe_alpha(1.0, init, resid, gt),
            "epe_at_alpha_8": epe_alpha(8.0, init, resid, gt)}


def oracle_beta(init: np.ndarray, resid: np.ndarray, gt: np.ndarray) -> dict:
    best_b, best_e = None, np.inf
    for b in np.arange(-30.0, 30.0001, 0.25):
        e = epe_beta(float(b), init, resid, gt)
        if e < best_e:
            best_e, best_b = e, float(b)
    for b in np.arange(best_b - 0.25, best_b + 0.25001, 0.025):
        e = epe_beta(float(b), init, resid, gt)
        if e < best_e:
            best_e, best_b = e, float(b)
    for b in np.arange(best_b - 0.025, best_b + 0.025001, 0.0025):
        e = epe_beta(float(b), init, resid, gt)
        if e < best_e:
            best_e, best_b = e, float(b)
    if not np.isfinite(best_e):
        raise ValueError("NaN/inf in oracle-beta scan")
    return {"beta_star": best_b, "epe_at_beta_star": best_e,
            "epe_at_beta_0": epe_beta(0.0, init, resid, gt)}


def sign_only_epe(init: np.ndarray, resid: np.ndarray, gt: np.ndarray) -> float:
    corrected = init + np.abs(gt - init) * np.sign(resid)
    e = float(np.abs(np.maximum(corrected, 0.0) - gt).mean())
    if not np.isfinite(e):
        raise ValueError("NaN/inf in sign-only ceiling")
    return e


def sign_only_scaled_epe(init: np.ndarray, resid: np.ndarray, gt: np.ndarray) -> float:
    """SUPPLEMENTARY (not briefed): units-consistent sign-only probe.

    corrected = relu(8*init + |GT - 8*init| * sign(resid)).
    Added after observing the literal briefed probe is degenerate (~0 EPE):
    with init in candidate units and GT in px, GT >> init on ~all pixels and
    resid > 0 nearly everywhere, so the literal formula collapses to GT
    itself. This variant asks the intended question in the headline
    (COARSE_SCALED) space: EPE > 0 here comes ONLY from pixels where
    sign(resid) disagrees with sign(GT - 8*init) (or the relu bites), so it
    quantifies sign-error mass vs magnitude-error mass.
    """
    coarse = SCALE * init
    corrected = coarse + np.abs(gt - coarse) * np.sign(resid)
    e = float(np.abs(np.maximum(corrected, 0.0) - gt).mean())
    if not np.isfinite(e):
        raise ValueError("NaN/inf in supplementary sign-only-scaled probe")
    return e


@torch.no_grad()
def process_seed(seed: int, device: str, limit: int | None,
                 reuse_raw: bool = False) -> dict:
    if reuse_raw:
        # Skip GPU inference; reload the frozen raw dumps (float32 store,
        # float64 compute). Raw files must already exist from a full run.
        z = np.load(RAW / f"d5_s{seed}.npz")
        init = z["disparity_initial"].astype(np.float64)
        resid = z["refinement_residual"].astype(np.float64)
        final = z["disparity_final"].astype(np.float64)
        gt = z["gt"].astype(np.float64)
        top1 = z["top1_prob"].astype(np.float64)
        margin = z["top1_margin"].astype(np.float64)
        ent = z["entropy"].astype(np.float64)
        occ = z["occluded"].astype(bool)
        check_finite(f"seed {seed} reused vectors",
                     init, resid, final, gt, top1, margin, ent)
        n_valid = int(gt.size)
        return analyze_seed(seed, init, resid, final, gt, top1, margin, ent,
                            occ, n_valid, limit=None, write_raw=False)
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
        raise ValueError(f"seed {seed}: split sizes {len(ds_occ)}/{len(ds_noc)}, need 40/40")
    n_img = len(ds_occ) if limit is None else min(limit, len(ds_occ))
    idx_grid = torch.arange(D, dtype=torch.float32, device=device)

    # per-image lists, concatenated at end (float64)
    L_init, L_resid, L_final, L_gt = [], [], [], []
    L_top1, L_margin, L_ent, L_occ = [], [], [], []
    n_valid = 0

    for i in range(n_img):
        s = ds_occ[i]
        sn = ds_noc[i]
        if s.name != sn.name:
            raise ValueError(f"scene order mismatch: {s.name} vs {sn.name}")
        if s.disparity.shape != (368, 1232) or sn.disparity.shape != (368, 1232):
            raise ValueError(f"shape mismatch scene {s.name}")
        left = torch.from_numpy(normalize(s.left)).to(device)
        right = torch.from_numpy(normalize(s.right)).to(device)
        gt_np = s.disparity.astype(np.float64)
        valid_np = gt_np > 0
        nv = int(valid_np.sum())
        if nv == 0:
            raise ValueError(f"empty valid mask scene {s.name}")
        occ_np = (valid_np & (sn.disparity <= 0))

        ret = model(left, right, return_stages=True)
        if not (isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[1], dict)):
            raise ValueError("forward(return_stages=True) did not return (disparity, stages-dict)")
        out, st = ret
        for k in ("aggregated_cost", "disparity_initial",
                  "refinement_residual", "disparity_final"):
            if k not in st:
                raise ValueError(f"stages dict missing key: {k}")
        d_init = st["disparity_initial"]
        resid = st["refinement_residual"]
        final = st["disparity_final"]
        if not (torch.isfinite(d_init).all() and torch.isfinite(resid).all()
                and torch.isfinite(final).all()):
            raise ValueError(f"NaN/inf in stages scene {s.name}")

        # D2 confidence readout, recomputed (standardised softmax, NOT raw scale)
        cu = F.interpolate(st["aggregated_cost"], size=(368, 1232),
                           mode="bilinear", align_corners=True)
        cus = ((cu - cu.mean(dim=1, keepdim=True))
               / (cu.std(dim=1, keepdim=True) + 1e-6))
        w = torch.softmax(-cus, dim=1)
        if not torch.isfinite(w).all():
            raise ValueError(f"NaN/inf in softmax weights scene {s.name}")
        valid_t = torch.from_numpy(valid_np).to(device)
        Wd = w[0, :, valid_t].double()  # (24, N)
        ent = -(Wd * torch.log(Wd + 1e-12)).sum(0)
        top2v, _ = torch.topk(Wd, 2, dim=0)

        L_init.append(d_init[0, 0, valid_t].double().cpu().numpy())
        L_resid.append(resid[0, 0, valid_t].double().cpu().numpy())
        L_final.append(final[0, 0, valid_t].double().cpu().numpy())
        L_gt.append(gt_np[valid_np])
        L_top1.append(top2v[0].cpu().numpy())
        L_margin.append((top2v[0] - top2v[1]).cpu().numpy())
        L_ent.append(ent.cpu().numpy())
        L_occ.append(occ_np[valid_np])
        n_valid += nv
        del ret, out, st, d_init, resid, final, cu, cus, w, Wd, ent, top2v
        if device == "cuda" and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0 or (i + 1) == n_img:
            print(f"seed {seed} [{i + 1}/{n_img}] valid={n_valid}", flush=True)

    init = np.concatenate(L_init).astype(np.float64)
    resid = np.concatenate(L_resid).astype(np.float64)
    final = np.concatenate(L_final).astype(np.float64)
    gt = np.concatenate(L_gt).astype(np.float64)
    top1 = np.concatenate(L_top1).astype(np.float64)
    margin = np.concatenate(L_margin).astype(np.float64)
    ent = np.concatenate(L_ent).astype(np.float64)
    occ = np.concatenate(L_occ).astype(bool)
    check_finite(f"seed {seed} vectors", init, resid, final, gt, top1, margin, ent)
    if (gt <= 0).any():
        raise ValueError(f"seed {seed}: non-positive GT in valid mask")
    del L_init, L_resid, L_final, L_gt, L_top1, L_margin, L_ent, L_occ

    return analyze_seed(seed, init, resid, final, gt, top1, margin, ent,
                        occ, n_valid, limit, write_raw=not reuse_raw)


def analyze_seed(seed: int, init: np.ndarray, resid: np.ndarray,
                 final: np.ndarray, gt: np.ndarray, top1: np.ndarray,
                 margin: np.ndarray, ent: np.ndarray, occ: np.ndarray,
                 n_valid: int, limit: int | None, write_raw: bool = True) -> dict:
    """Pure-numpy analysis of one seed's valid-pixel vectors (float64)."""
    if write_raw and limit is None:
        atomic_write_npz(
            RAW / f"d5_s{seed}.npz",
            disparity_initial=init.astype(np.float32),
            refinement_residual=resid.astype(np.float32),
            disparity_final=final.astype(np.float32),
            gt=gt.astype(np.float32),
            top1_prob=top1.astype(np.float32),
            top1_margin=margin.astype(np.float32),
            entropy=ent.astype(np.float32),
            occluded=occ,
        )

    res: dict = {"valid_pixels": int(n_valid)}
    res["pooled"] = core_stats(init, resid, final, gt)

    # per-D1-bin
    b = np.clip(np.searchsorted(EDGES, gt, side="right") - 1, 0, len(LABELS) - 1)
    bins = []
    for bi, lab in enumerate(LABELS):
        m = b == bi
        if int(m.sum()) == 0:
            bins.append({"bin": lab, "px": 0, "status": "EMPTY"})
            continue
        row = {"bin": lab, **core_stats(init[m], resid[m], final[m], gt[m])}
        row["oracle_scale"] = oracle_alpha(init[m], resid[m], gt[m])
        row["oracle_offset"] = oracle_beta(init[m], resid[m], gt[m])
        row["sign_only_epe"] = sign_only_epe(init[m], resid[m], gt[m])
        row["sign_only_scaled_supplementary"] = sign_only_scaled_epe(
            init[m], resid[m], gt[m])
        bins.append(row)
    res["per_bin"] = bins

    # headroom probes, pooled
    res["oracle_scale_pooled"] = oracle_alpha(init, resid, gt)
    res["oracle_offset_pooled"] = oracle_beta(init, resid, gt)
    res["sign_only_epe_pooled"] = sign_only_epe(init, resid, gt)
    res["sign_only_scaled_supplementary_pooled"] = sign_only_scaled_epe(init, resid, gt)

    # confidence deciles (equal-width over [0,1])
    dec_edges = np.linspace(0.0, 1.0, 11)
    err_scaled = np.abs(SCALE * init - gt)
    err_ref = np.abs(final - gt)
    improv = err_scaled - err_ref
    for key, conf in (("top1_prob", top1), ("top1_margin", margin)):
        di = np.clip(np.digitize(conf, dec_edges[1:-1], right=False), 0, 9)
        rows = []
        for d in range(10):
            m = di == d
            n = int(m.sum())
            if n == 0:
                rows.append({"decile": d, "lo": float(dec_edges[d]),
                             "hi": float(dec_edges[d + 1]), "px": 0,
                             "status": "EMPTY"})
                continue
            rows.append({
                "decile": d, "lo": float(dec_edges[d]), "hi": float(dec_edges[d + 1]),
                "px": n, "frac_px": float(n / gt.size),
                "mean_refinement_improvement": float(improv[m].mean()),
                "frac_improved": float((improv[m] > UNCH_THRESH).mean()),
                "mean_abs_residual": float(np.abs(resid[m]).mean()),
            })
        res[f"deciles_{key}"] = rows

    # regimes
    regs = {
        "gt_lt_64": gt < 64.0,
        "gt_64_96": (gt >= 64.0) & (gt < 96.0),
        "gt_96_128": (gt >= 96.0) & (gt < 128.0),
        "gt_ge_128": gt >= 128.0,
        "occluded": occ,
        "non_occluded": ~occ,
    }
    reg_out = {}
    for name, m in regs.items():
        n = int(m.sum())
        if n == 0:
            reg_out[name] = {"px": 0, "status": "EMPTY"}
            continue
        r = core_stats(init[m], resid[m], final[m], gt[m])
        reg_out[name] = {
            "px": n, "frac_px": float(n / gt.size),
            "frac_improved": r["frac_improved"],
            "frac_worsened": r["frac_worsened"],
            "mean_refinement_improvement": float(improv[m].mean()),
            "epe_coarse_scaled": r["epe_coarse_scaled"],
            "epe_refined": r["epe_refined"],
            "d1_coarse_scaled": r["d1_coarse_scaled"],
            "d1_refined": r["d1_refined"],
            "residual_mean_signed": r["residual_mean_signed"],
            "residual_mean_abs": r["residual_mean_abs"],
            "relu_clamp_frac": r["relu_clamp_frac"],
        }
    res["regimes"] = reg_out
    res["occ_frac_of_valid"] = float(occ.mean())

    # D2 cross-check (full runs only: pooled stats must match D2's 40-image
    # pooled values; a --limit subset has a different pixel population)
    res["d2_crosscheck"] = None
    if limit is None:
        d2 = json.loads((OUT / "cost_distribution.json").read_text())
        d2p = d2["seeds"][str(seed)]["distribution_pooled"]
        xcheck = {}
        for k, arr in (("entropy", ent), ("top1", top1), ("margin", margin)):
            diff = float(arr.mean() - d2p[k])
            xcheck[k] = {"d5": float(arr.mean()), "d2": float(d2p[k]), "diff": diff}
            if abs(diff) > TOL_D2_XCHECK:
                raise ValueError(f"seed {seed}: D2 cross-check failed for {k}: diff={diff:.2e}")
        res["d2_crosscheck"] = xcheck

    print(f"seed {seed} pooled: raw={res['pooled']['epe_coarse_raw']:.4f} "
          f"scaled={res['pooled']['epe_coarse_scaled']:.4f} "
          f"ref={res['pooled']['epe_refined']:.7f} "
          f"improved={res['pooled']['frac_improved']:.4f} "
          f"alpha*={res['oracle_scale_pooled']['alpha_star']:.3f} "
          f"beta*={res['oracle_offset_pooled']['beta_star']:.4f}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--reuse-raw", action="store_true",
                    help="reload raw/d5_s*.npz instead of GPU inference; "
                         "analysis is identical pure-numpy code")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"D5 device={device} seeds={seeds} limit={args.limit} "
          f"reuse_raw={args.reuse_raw}", flush=True)

    # seed 0 smoke test first when doing full runs (skipped with --reuse-raw:
    # raw dumps already cover all 40 scenes)
    if args.limit is None and not args.reuse_raw:
        smoke = process_seed(0, device, 2)
        e0 = smoke["pooled"]["epe_refined"]
        print(f"SMOKE seed0 limit=2 refined_epe={e0:.7f} (informational)", flush=True)
        del smoke

    results: dict[int, dict] = {}
    for seed in seeds:
        results[seed] = process_seed(seed, device, args.limit,
                                     reuse_raw=args.reuse_raw)

    if args.limit is None and 0 in results:
        e0 = results[0]["pooled"]["epe_refined"]
        ok = abs(e0 - EXPECTED_SEED0_EPE) < TOL_BASELINE_PX
        print(f"GATE seed0 refined={e0:.10f} expected={EXPECTED_SEED0_EPE:.10f} "
              f"diff={e0 - EXPECTED_SEED0_EPE:.2e} -> "
              f"{'PASS' if ok else 'FAIL: STOP'}", flush=True)
        if not ok:
            raise SystemExit("D5 STOP: REFINED does not reproduce 1.4149796 on seed 0")
        for seed in seeds:
            z = np.load(OUT / "raw" / f"d0_preds_s{seed}.npz")
            if int(z["P"].size) != results[seed]["valid_pixels"]:
                raise ValueError(f"seed {seed}: D5 valid count != D0 dump")
            if results[seed]["valid_pixels"] != EXPECTED_VALID:
                raise ValueError(f"seed {seed}: valid {results[seed]['valid_pixels']} "
                                 f"!= {EXPECTED_VALID}")

    if args.limit is None:
        def mm(get):
            return float(np.mean([get(results[s]) for s in seeds]))

        pooled_mm = {
            "epe_coarse_raw": mm(lambda r: r["pooled"]["epe_coarse_raw"]),
            "epe_coarse_scaled": mm(lambda r: r["pooled"]["epe_coarse_scaled"]),
            "epe_refined": mm(lambda r: r["pooled"]["epe_refined"]),
            "delta_refined_minus_coarse_scaled": mm(
                lambda r: r["pooled"]["delta_refined_minus_coarse_scaled"]),
            "d1_coarse_raw": mm(lambda r: r["pooled"]["d1_coarse_raw"]),
            "d1_coarse_scaled": mm(lambda r: r["pooled"]["d1_coarse_scaled"]),
            "d1_refined": mm(lambda r: r["pooled"]["d1_refined"]),
            "frac_improved": mm(lambda r: r["pooled"]["frac_improved"]),
            "frac_worsened": mm(lambda r: r["pooled"]["frac_worsened"]),
            "frac_unchanged_1e6": mm(lambda r: r["pooled"]["frac_unchanged_1e6"]),
            "mean_improvement_on_helped": mm(
                lambda r: r["pooled"]["mean_improvement_on_helped"]),
            "mean_degradation_on_harmed": mm(
                lambda r: r["pooled"]["mean_degradation_on_harmed"]),
            "residual_mean_signed": mm(lambda r: r["pooled"]["residual_mean_signed"]),
            "residual_mean_abs": mm(lambda r: r["pooled"]["residual_mean_abs"]),
            "relu_clamp_frac": mm(lambda r: r["pooled"]["relu_clamp_frac"]),
            "frac_resid_sign_agrees_with_needed": mm(
                lambda r: r["pooled"]["frac_resid_sign_agrees_with_needed"]),
            "oracle_alpha_star": mm(lambda r: r["oracle_scale_pooled"]["alpha_star"]),
            "oracle_alpha_epe": mm(
                lambda r: r["oracle_scale_pooled"]["epe_at_alpha_star"]),
            "oracle_beta_star": mm(lambda r: r["oracle_offset_pooled"]["beta_star"]),
            "oracle_beta_epe": mm(
                lambda r: r["oracle_offset_pooled"]["epe_at_beta_star"]),
            "sign_only_epe": mm(lambda r: r["sign_only_epe_pooled"]),
            "sign_only_scaled_supplementary": mm(
                lambda r: r["sign_only_scaled_supplementary_pooled"]),
        }
        out = {
            "conventions": {
                "population": "full-resolution frozen valid mask (GT>0), "
                              "3802797 px/seed; NOT the D3/D4 feature-res cell set",
                "coarse_raw": "disparity_initial as-is vs GT in px "
                              "(units-mismatched literal pre-refinement tensor)",
                "coarse_scaled": "8.0*disparity_initial vs GT in px (physically "
                                 "meaningful; candidate k = 8k px). HEADLINE coarse "
                                 "convention: the soft-argmin output is only "
                                 "interpretable as disparity after candidate->pixel "
                                 "scaling, and the refinement adds its residual to "
                                 "the unscaled tensor, so 8x is the reading under "
                                 "which 'correction' is well-defined",
                "refined": "disparity_final = relu(disparity_initial + "
                           "refinement_residual), in pixels",
                "improved": "|refined err| < |coarse-scaled err|; unchanged "
                            "within 1e-6 on the abs-err difference",
                "improvement": "coarse-scaled abs err minus refined abs err "
                               "(positive = helped)",
                "deciles": "equal-width bins over [0,1] for top-1 prob / margin",
                "oracle_scale": "alpha minimising EPE of relu(alpha*init+resid); "
                                "no model re-run, pure rescoring",
                "oracle_offset": "beta minimising EPE of relu(init+resid+beta)",
                "sign_only": "relu(init + |GT-init|*sign(resid)); naive "
                             "residual_oracle = GT-init is trivially 0 and is NOT "
                             "reported as a finding. NOTE: the briefed sign-only "
                             "formula itself collapses to ~GT (EPE ~1e-4) because "
                             "GT_px >> init_cand and resid>0 nearly everywhere; "
                             "reported literally with this caveat",
                "sign_only_scaled_supplementary": "SUPPLEMENTARY, not briefed: "
                             "relu(8*init+|GT-8*init|*sign(resid)); EPE>0 comes "
                             "only from sign disagreements, so it quantifies "
                             "sign-error vs magnitude-error mass",
                "occlusion": "valid in disp_occ_0 AND not valid in disp_noc_0 "
                             "(reference_strata.py definition)",
                "method_mean": "arithmetic mean across the 3 seeds",
            },
            "config": {"downsample_levels": 3, "num_disparities": 24,
                       "cost_volume_shift": "right", "regression_normalize": True,
                       "stride_px": 8, "candidate_range_px": [0, 184]},
            "headline_coarse_convention": "COARSE_SCALED",
            "seeds": {str(s): results[s] for s in seeds},
            "method_mean_pooled": pooled_mm,
            "interpretation_options": ["(1) systematically improves",
                                       "(2) mainly fixes particular regimes",
                                       "(3) damages particular regimes",
                                       "(4) has little remaining headroom"],
            "interpretation": ("Factual statement only, no verdict: see the pooled "
                               "improved/worsened fractions, per-bin and per-regime "
                               "improvement, oracle alpha/beta vs 1.0/0.0, and the "
                               "confidence-decile slopes in the numbers above."),
            "unmeasurable": "NOT MEASURABLE WITH CURRENT ARTIFACTS: none; "
                            "every D5 quantity was measured.",
        }
        atomic_write_json(OUT / "refinement_diagnostic.json", out)

        import csv
        fd, tmp = tempfile.mkstemp(dir=str(RAW), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["seed", "bin", "px", "frac_px", "epe_raw", "epe_scaled",
                            "epe_refined", "delta_ref_scaled", "d1_raw", "d1_scaled",
                            "d1_refined", "frac_improved", "frac_worsened",
                            "frac_unchanged", "mean_impr_helped", "mean_degr_harmed",
                            "resid_mean", "resid_mean_abs", "clamp_frac",
                            "alpha_star", "alpha_epe", "beta_star", "beta_epe",
                            "sign_only_epe", "sign_only_scaled_suppl"])
                for s in seeds:
                    n = results[s]["valid_pixels"]
                    for row in results[s]["per_bin"]:
                        if row.get("status") == "EMPTY":
                            w.writerow([s, row["bin"], 0, 0, *[("") for _ in range(21)]])
                            continue
                        w.writerow([s, row["bin"], row["px"],
                                    f"{row['px'] / n:.9f}",
                                    f"{row['epe_coarse_raw']:.6f}",
                                    f"{row['epe_coarse_scaled']:.6f}",
                                    f"{row['epe_refined']:.6f}",
                                    f"{row['delta_refined_minus_coarse_scaled']:.6f}",
                                    f"{row['d1_coarse_raw']:.4f}",
                                    f"{row['d1_coarse_scaled']:.4f}",
                                    f"{row['d1_refined']:.4f}",
                                    f"{row['frac_improved']:.6f}",
                                    f"{row['frac_worsened']:.6f}",
                                    f"{row['frac_unchanged_1e6']:.6f}",
                                    f"{fmt(row['mean_improvement_on_helped'], '.6f')}",
                                    f"{fmt(row['mean_degradation_on_harmed'], '.6f')}",
                                    f"{row['residual_mean_signed']:.6f}",
                                    f"{row['residual_mean_abs']:.6f}",
                                    f"{row['relu_clamp_frac']:.6f}",
                                    f"{row['oracle_scale']['alpha_star']:.4f}",
                                    f"{row['oracle_scale']['epe_at_alpha_star']:.6f}",
                                    f"{row['oracle_offset']['beta_star']:.4f}",
                                    f"{row['oracle_offset']['epe_at_beta_star']:.6f}",
                                    f"{row['sign_only_epe']:.6f}",
                                    f"{row['sign_only_scaled_supplementary']:.6f}"])
            os.replace(tmp, RAW / "d5_bins.csv")
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        print("wrote refinement_diagnostic.json + raw/d5_s*.npz + raw/d5_bins.csv",
              flush=True)


if __name__ == "__main__":
    main()
