"""PHASE-2 STEP 1 — ARM-V high-disparity diagnostic. INFERENCE ONLY, NO TRAINING.

Answers, per frozen ARM-V seed (0, 1, 2), on both the frozen evaluation split
and the training split:

  A  per-GT-bin pooled statistics (count, fraction, EPE, signed error, RMSE, D1)
  B  prediction-vs-GT relationship in the GT >= 64 px stratum (descriptive fits)
  C  initial readout vs final output in the same stratum
  D  readout distribution statistics stratified by GT
  E  whether the cost volume carries a response at the TRUE disparity candidate
  F  train-split versus validation-split high-disparity behaviour

The readout distribution reproduced here is exactly the one the model consumes:
the aggregated cost is bilinearly upsampled to full resolution (align_corners),
standardised across the disparity axis (regression_normalize=True), then
softmax(-cost). No ground truth enters any model computation; GT is used only to
index and stratify the measurements afterwards.

Nothing is trained. No ARM-V artifact is modified.

Usage: python phase2/diagnostics/arm_v_high_disparity/diag_high_disparity.py
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F

from phase1.harness.frozen_eval import refuse_unless_contract, sha256_file
from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

SPACING = 8.0            # full-resolution px per disparity candidate (ARM-V)
NDISP = 24
MAXD_PX = (NDISP - 1) * SPACING          # 184.0
BIN_EDGES = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
MIN_BIN_PX = 1000        # below this a bin is reported as INSUFFICIENT, not summarised
HIGH_CUT = 64.0
SEEDS = {0: "arm_v", 1: "arm_v_s1", 2: "arm_v_s2"}

# Scalars accumulated per pixel. Kept as float32 arrays for the eval split (3.8 M
# px) and, on the train split (14.2 M px), only for the GT >= 64 stratum; the
# low-GT train bins use running sums, which is all question F needs.
FIELDS = ("gt", "pred", "init", "ent", "p_top1", "p_top3", "p_true",
          "rank_true", "z_true_agg", "z_true_raw", "argmax_idx", "dist_std")


def build_model(run: str) -> StereoNet:
    blob = torch.load(REPO / "phase1" / "runs" / run / "arm_v_best.pth",
                      map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    m = StereoNet(StereoNetConfig(downsample_levels=3, num_disparities=NDISP,
                                  cost_volume_shift="right",
                                  regression_normalize=True))
    m.load_state_dict(state, strict=True)
    return m.eval()


def readout_stats(cost_full: torch.Tensor, true_idx: torch.Tensor) -> dict:
    """cost_full: (1, D, H, W) upsampled aggregated cost. true_idx: (1,1,H,W)."""
    mu = cost_full.mean(1, keepdim=True)
    sd = cost_full.std(1, keepdim=True)
    z = (cost_full - mu) / (sd + 1e-6)          # what soft_argmin(normalize=True) sees
    p = torch.softmax(-z, dim=1)
    idx = torch.arange(NDISP, device=p.device, dtype=p.dtype).view(1, -1, 1, 1)
    exp_idx = (p * idx).sum(1, keepdim=True)
    ent = -(p * torch.log(p + 1e-12)).sum(1, keepdim=True)
    var = (p * (idx - exp_idx) ** 2).sum(1, keepdim=True)
    top3 = p.topk(3, dim=1).values
    k = true_idx.round().clamp(0, NDISP - 1).long()
    p_true = p.gather(1, k)
    z_true = z.gather(1, k)
    # rank of the true candidate by probability, 1 = most probable
    rank_true = (p > p_true).sum(1, keepdim=True).to(p.dtype) + 1.0
    return {"ent": ent, "p_top1": top3[:, :1], "p_top3": top3.sum(1, keepdim=True),
            "p_true": p_true, "rank_true": rank_true, "z_true_agg": z_true,
            "argmax_idx": p.argmax(1, keepdim=True).to(p.dtype),
            "dist_std": var.sqrt(), "exp_idx": exp_idx}


def run_split(model: StereoNet, split: str, device: str, keep_all: bool) -> dict:
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split=split,
                         disparity_scale=256.0, occluded=True)
    store = {f: [] for f in FIELDS}
    # running per-bin sums, used for every bin (and the only source for the
    # train split's low-GT bins, where full arrays are not retained)
    nb = len(BIN_EDGES) - 1
    S = {k: np.zeros(nb, dtype=np.float64) for k in
         ("n", "abs", "signed", "sq", "d1", "gt", "pred", "init8",
          "ent", "p_true", "p_top1", "z_true_agg", "z_true_raw", "rank_true",
          "argmax_hit")}
    pred_max_all = -1e30
    init_max_all = -1e30
    n_valid = 0
    for i in range(len(ds)):
        s = ds[i]
        gt_np = s.disparity.astype(np.float32)
        valid_np = gt_np > 0
        with torch.no_grad():
            L = torch.from_numpy(normalize(s.left)).to(device)
            R = torch.from_numpy(normalize(s.right)).to(device)
            out, st = model(L, R, return_stages=True)
            cost_full = F.interpolate(st["aggregated_cost"], size=gt_np.shape,
                                      mode="bilinear", align_corners=True)
            gt_t = torch.from_numpy(gt_np).to(device)[None, None]
            true_idx = (gt_t.clamp(0, MAXD_PX) / SPACING)
            rs = readout_stats(cost_full, true_idx)
            # raw (pre-aggregation) response at the true candidate
            raw = st["cost_volume"].norm(p=2, dim=1)              # (1,D,h,w)
            raw_full = F.interpolate(raw, size=gt_np.shape, mode="bilinear",
                                     align_corners=True)
            zr = (raw_full - raw_full.mean(1, keepdim=True)) / (raw_full.std(1, keepdim=True) + 1e-6)
            rs["z_true_raw"] = zr.gather(1, true_idx.round().clamp(0, NDISP - 1).long())
            rs["pred"] = out
            rs["init"] = st["disparity_initial"]
            rs["gt"] = gt_t

        v = torch.from_numpy(valid_np).to(device)[None, None]
        pred_max_all = max(pred_max_all, float(out.max()))
        init_max_all = max(init_max_all, float(st["disparity_initial"].max()))
        flat = {f: rs[f][v].detach().cpu().numpy().astype(np.float32) for f in FIELDS}
        gt_v, pred_v = flat["gt"].astype(np.float64), flat["pred"].astype(np.float64)
        err = np.abs(pred_v - gt_v)
        n_valid += gt_v.size
        bidx = np.digitize(gt_v, BIN_EDGES) - 1
        for b in range(nb):
            m = bidx == b
            if not m.any():
                continue
            S["n"][b] += m.sum()
            S["abs"][b] += err[m].sum()
            S["signed"][b] += (pred_v[m] - gt_v[m]).sum()
            S["sq"][b] += ((pred_v[m] - gt_v[m]) ** 2).sum()
            S["d1"][b] += ((err[m] > 3) & (err[m] > 0.05 * gt_v[m])).sum()
            S["gt"][b] += gt_v[m].sum()
            S["pred"][b] += pred_v[m].sum()
            S["init8"][b] += (SPACING * flat["init"][m].astype(np.float64)).sum()
            for k in ("ent", "p_true", "p_top1", "z_true_agg", "z_true_raw", "rank_true"):
                S[k][b] += flat[k][m].astype(np.float64).sum()
            S["argmax_hit"][b] += (np.abs(flat["argmax_idx"][m].astype(np.float64)
                                          - np.clip(gt_v[m], 0, MAXD_PX) / SPACING) <= 1).sum()
        if keep_all:
            for f in FIELDS:
                store[f].append(flat[f])
        else:
            hi = gt_v >= HIGH_CUT
            if hi.any():
                for f in FIELDS:
                    store[f].append(flat[f][hi])
        print("  %s %2d/%d %s" % (split, i + 1, len(ds), s.name), flush=True)

    A = {f: np.concatenate(store[f]).astype(np.float64) for f in FIELDS}
    guard = refuse_unless_contract(len(ds), n_valid, 256.0, split, "disp_occ_0") \
        if split == "hailo_val" else {"contract_match": None,
                                      "note": "training split; NOT a frozen-contract score"}
    return {"S": S, "A": A, "n_valid": int(n_valid), "scenes": len(ds),
            "pred_max_all_px": pred_max_all, "init_max_candidates": init_max_all,
            "guard": guard, "arrays_are_high_only": not keep_all}


def summarise_bins(S: dict, A: dict, high_only: bool, n_valid: int) -> list:
    rows = []
    for b, (lo, hi) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
        n = int(S["n"][b])
        row = {"lo": lo, "hi": hi, "px": n, "frac_px": n / n_valid if n_valid else 0.0}
        if n < MIN_BIN_PX:
            row["status"] = "INSUFFICIENT (<%d px) — not summarised" % MIN_BIN_PX
            rows.append(row)
            continue
        row["status"] = "ok"
        row["epe"] = S["abs"][b] / n
        row["signed_err"] = S["signed"][b] / n
        row["rmse"] = (S["sq"][b] / n) ** 0.5
        row["d1_pct"] = 100.0 * S["d1"][b] / n
        row["mean_gt"] = S["gt"][b] / n
        row["mean_pred"] = S["pred"][b] / n
        row["pred_over_gt"] = row["mean_pred"] / row["mean_gt"] if row["mean_gt"] else None
        row["mean_init_x8_px"] = S["init8"][b] / n
        row["init_x8_over_gt"] = row["mean_init_x8_px"] / row["mean_gt"] if row["mean_gt"] else None
        row["readout_entropy"] = S["ent"][b] / n
        row["p_top1"] = S["p_top1"][b] / n
        row["p_at_true_candidate"] = S["p_true"][b] / n
        row["mean_rank_of_true_candidate"] = S["rank_true"][b] / n
        row["z_at_true_candidate_aggregated"] = S["z_true_agg"][b] / n
        row["z_at_true_candidate_raw"] = S["z_true_raw"][b] / n
        row["frac_argmax_within_1_of_true"] = S["argmax_hit"][b] / n
        if not high_only or lo >= HIGH_CUT:
            m = (A["gt"] >= lo) & (A["gt"] < hi)
            if m.sum() >= MIN_BIN_PX:
                row["median_gt"] = float(np.median(A["gt"][m]))
                row["median_pred"] = float(np.median(A["pred"][m]))
        else:
            row["median_gt"] = None
            row["median_pred"] = None
            row["median_note"] = "NOT COMPUTED (train split retains full arrays only for GT>=64)"
        rows.append(row)
    return rows


def descriptive_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares y ~ a*x + b. DESCRIPTIVE ONLY; no model is trained."""
    if x.size < MIN_BIN_PX:
        return {"status": "INSUFFICIENT", "n": int(x.size)}
    a, b = np.polyfit(x, y, 1)
    yh = a * x + b
    ss_res = float(((y - yh) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"status": "ok", "n": int(x.size), "slope": float(a), "intercept": float(b),
            "r2": 1.0 - ss_res / ss_tot if ss_tot else None,
            "residual_rmse_px": float(np.sqrt(ss_res / x.size)),
            "pearson_r": float(np.corrcoef(x, y)[0, 1])}


def analyse(res: dict) -> dict:
    A, S = res["A"], res["S"]
    high_only = res["arrays_are_high_only"]
    out = {"scenes": res["scenes"], "valid_pixels": res["n_valid"],
           "guard": res["guard"],
           "pred_max_all_px": res["pred_max_all_px"],
           "init_max_candidates": res["init_max_candidates"],
           "init_max_x8_px": SPACING * res["init_max_candidates"],
           "bins": summarise_bins(S, A, high_only, res["n_valid"])}

    hi = A["gt"] >= HIGH_CUT
    lo = ~hi
    out["high_stratum"] = {
        "cut_px": HIGH_CUT, "px": int(hi.sum()),
        "frac_px": float(hi.sum()) / res["n_valid"],
        "mean_gt": float(A["gt"][hi].mean()), "mean_pred": float(A["pred"][hi].mean()),
        "median_gt": float(np.median(A["gt"][hi])),
        "median_pred": float(np.median(A["pred"][hi])),
        "mean_signed_err": float((A["pred"][hi] - A["gt"][hi]).mean()),
        "epe": float(np.abs(A["pred"][hi] - A["gt"][hi]).mean()),
        "mean_pred_over_gt": float((A["pred"][hi] / A["gt"][hi]).mean()),
        "fit_pred_on_gt": descriptive_fit(A["gt"][hi], A["pred"][hi]),
        "fit_initx8_on_gt": descriptive_fit(A["gt"][hi], SPACING * A["init"][hi]),
    }
    if not high_only:
        out["low_stratum_fits"] = {
            "fit_pred_on_gt": descriptive_fit(A["gt"][lo], A["pred"][lo]),
            "fit_initx8_on_gt": descriptive_fit(A["gt"][lo], SPACING * A["init"][lo]),
            "fit_gt_on_init_index": descriptive_fit(A["init"][lo], A["gt"][lo]),
        }
        # C: calibrate the index->pixel map on GT<64 ONLY, then extrapolate to
        # the high stratum. Tests whether the INITIAL readout already carries
        # the high disparities, independent of the learned output scale.
        f = out["low_stratum_fits"]["fit_gt_on_init_index"]
        a, b = f["slope"], f["intercept"]
        ext = a * A["init"][hi] + b
        out["extrapolated_initial_readout"] = {
            "calibration": "GT = %.6f * soft_argmin_index + %.6f, fitted on GT < 64 px only"
                           % (a, b),
            "geometric_slope_would_be": SPACING,
            "mean_extrapolated_pred_on_high_stratum": float(ext.mean()),
            "mean_gt_on_high_stratum": float(A["gt"][hi].mean()),
            "mean_signed_err": float((ext - A["gt"][hi]).mean()),
            "epe": float(np.abs(ext - A["gt"][hi]).mean()),
            "max_extrapolated": float(ext.max()),
            "final_model_epe_same_pixels": float(np.abs(A["pred"][hi] - A["gt"][hi]).mean()),
            "final_model_max_pred": float(A["pred"][hi].max()),
        }
    # D/E stratified readout comparison
    out["readout_low_vs_high"] = {}
    for tag, m in (("gt_lt_64", lo), ("gt_ge_64", hi)):
        if m.sum() < MIN_BIN_PX:
            out["readout_low_vs_high"][tag] = {"status": "INSUFFICIENT"}
            continue
        out["readout_low_vs_high"][tag] = {
            "px": int(m.sum()),
            "entropy": float(A["ent"][m].mean()),
            "entropy_max_possible": float(np.log(NDISP)),
            "p_top1": float(A["p_top1"][m].mean()),
            "p_top3_mass": float(A["p_top3"][m].mean()),
            "p_at_true_candidate": float(A["p_true"][m].mean()),
            "p_at_true_over_uniform": float(A["p_true"][m].mean() * NDISP),
            "median_rank_of_true_candidate": float(np.median(A["rank_true"][m])),
            "frac_true_candidate_in_top1": float((A["rank_true"][m] <= 1).mean()),
            "frac_true_candidate_in_top3": float((A["rank_true"][m] <= 3).mean()),
            "z_at_true_candidate_aggregated": float(A["z_true_agg"][m].mean()),
            "z_at_true_candidate_raw": float(A["z_true_raw"][m].mean()),
            "frac_z_true_below_zero_aggregated": float((A["z_true_agg"][m] < 0).mean()),
            "frac_z_true_below_zero_raw": float((A["z_true_raw"][m] < 0).mean()),
            "readout_distribution_std_candidates": float(A["dist_std"][m].mean()),
            "mean_argmax_index": float(A["argmax_idx"][m].mean()),
            "mean_true_index": float((np.clip(A["gt"][m], 0, MAXD_PX) / SPACING).mean()),
            "mean_soft_index": float(A["init"][m].mean()),
        }
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        head = None
    meta = {"utc": datetime.now(timezone.utc).isoformat(), "git_head": head,
            "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "device": device, "training_performed": False,
            "spacing_px_per_candidate": SPACING, "num_disparities": NDISP,
            "represented_range_px": MAXD_PX, "bin_edges": BIN_EDGES,
            "high_cut_px": HIGH_CUT, "min_bin_px": MIN_BIN_PX}
    results = {"metadata": meta, "seeds": {}}
    for seed, run in SEEDS.items():
        ckpt = REPO / "phase1" / "runs" / run / "arm_v_best.pth"
        print("seed %d (%s)" % (seed, run), flush=True)
        model = build_model(run).to(device)
        rec = {"run_dir": "phase1/runs/" + run,
               "checkpoint": "phase1/runs/%s/arm_v_best.pth" % run,
               "sha256": sha256_file(ckpt),
               "params": sum(p.numel() for p in model.parameters())}
        rec["hailo_val"] = analyse(run_split(model, "hailo_val", device, keep_all=True))
        rec["hailo_calib"] = analyse(run_split(model, "hailo_calib", device, keep_all=False))
        results["seeds"][str(seed)] = rec
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print("wrote", HERE / "results.json")


if __name__ == "__main__":
    main()
