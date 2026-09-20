"""TIER2-PILOT frozen-contract eval. Line-for-line mirror of the scoring core of
phase2/scripts/eval_p2a.py (NOT modified), applied to the Tier-2 pilot arms'
best+final checkpoints. Dataset split, GT scale, valid mask, pooled metrics
and contract guard are imported UNMODIFIED from phase1.harness.frozen_eval.

Adds only the required extra strata GT<64, 64-96, 96-128, >=128 alongside
P2A's own gt_lt_64/gt_ge_64/gt_ge_96 (kept for comparability).

Usage: <python> eval_tier2.py
Writes: <pilot>/tier2_eval.json (atomic). Never touches P2A artifacts.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
REPO = PILOT.parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from phase1.harness.frozen_eval import (pooled_metrics, refuse_unless_contract,
                                        sha256_file, strict_load_report, weight_sha)
from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

BINS = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
MIN_BIN = 1000
CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
EXPECTED_PARAMS = 397954

TARGETS = [("control", "best"), ("control", "final"),
           ("armp", "best"), ("armp", "final")]


def score(model, split: str, device: str) -> dict:
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split=split,
                         disparity_scale=256.0, occluded=True)
    preds, gts = [], []
    pmax_all = -1e30
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            out = model(torch.from_numpy(normalize(s.left)).to(device),
                        torch.from_numpy(normalize(s.right)).to(device))
            pred = out[0, 0].cpu().numpy().astype(np.float64)
            pmax_all = max(pmax_all, float(pred.max()))
            valid = s.disparity > 0
            preds.append(pred[valid])
            gts.append(s.disparity[valid].astype(np.float64))
    P, G = np.concatenate(preds), np.concatenate(gts)
    m = pooled_metrics(preds, gts)
    err = np.abs(P - G)

    def stratum(mask):
        if mask.sum() < MIN_BIN:
            return {"px": int(mask.sum()), "status": "INSUFFICIENT"}
        a, b = np.polyfit(G[mask], P[mask], 1)
        yh = a * G[mask] + b
        ss_res = float(((P[mask] - yh) ** 2).sum())
        ss_tot = float(((P[mask] - P[mask].mean()) ** 2).sum())
        return {"px": int(mask.sum()), "frac_px": float(mask.mean()),
                "epe": float(err[mask].mean()),
                "signed_err": float((P[mask] - G[mask]).mean()),
                "mean_gt": float(G[mask].mean()), "mean_pred": float(P[mask].mean()),
                "slope": float(a), "intercept": float(b),
                "r2": 1.0 - ss_res / ss_tot if ss_tot else None,
                "pearson_r": float(np.corrcoef(G[mask], P[mask])[0, 1])}

    bins = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mm = (G >= lo) & (G < hi)
        row = {"lo": lo, "hi": hi, "px": int(mm.sum()), "frac_px": float(mm.mean())}
        if mm.sum() >= MIN_BIN:
            row.update({"epe": float(err[mm].mean()),
                        "signed_err": float((P[mm] - G[mm]).mean()),
                        "rmse": float(np.sqrt(((P[mm] - G[mm]) ** 2).mean())),
                        "d1_pct": float((((err[mm] > 3) & (err[mm] > 0.05 * G[mm])).mean()) * 100),
                        "mean_gt": float(G[mm].mean()), "mean_pred": float(P[mm].mean())})
        else:
            row["status"] = "INSUFFICIENT (<%d px)" % MIN_BIN
        bins.append(row)

    guard = refuse_unless_contract(len(ds), m["valid_pixels"], 256.0, split,
                                   "disp_occ_0") if split == "hailo_val" else {
        "contract_match": None, "note": "training split; NOT a frozen-contract score"}
    return {"split": split, "scenes": len(ds),
            "metrics": m, "guard": guard,
            "gt_lt_64": stratum(G < 64),
            "gt_64_96": stratum((G >= 64) & (G < 96)),
            "gt_96_128": stratum((G >= 96) & (G < 128)),
            "gt_ge_128": stratum(G >= 128),
            "gt_ge_64": stratum(G >= 64), "gt_ge_96": stratum(G >= 96),
            "bins": bins,
            "max_pred_valid_px": float(P.max()), "max_pred_all_px": pmax_all,
            "max_gt": float(G.max())}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except Exception:
        head = None
    out = {"experiment": "TIER2-PILOT frozen eval (mirror of eval_p2a scoring core)",
           "mirror_note": ("scoring core mirrors phase2/scripts/eval_p2a.py, which was NOT "
                           "modified; P2A numbers NOT re-evaluated. Extra strata "
                           "gt_64_96/gt_96_128/gt_ge_128 added per pilot brief."),
           "metadata": {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "git_head": head, "python": sys.version.split()[0],
                        "torch": torch.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "device": device},
           "checkpoints": {}}
    for arm, tag in TARGETS:
        ck = PILOT / arm / ("p2a_%s.pth" % tag)
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        state = blob["model"]
        model = StereoNet(StereoNetConfig(**CFG))
        compat = strict_load_report(model, state)
        params = sum(p.numel() for p in model.parameters())
        if compat["missing"] or compat["unexpected"] or params != EXPECTED_PARAMS:
            raise SystemExit("ABORT eval: strict-load/architecture guard failed for %s/%s: %s"
                             % (arm, tag, json.dumps(compat)))
        model.eval().to(device)
        rec = {"checkpoint": "stage_b_armp/20260919T012646Z_tier2_seed1/%s/p2a_%s.pth" % (arm, tag),
               "sha256": sha256_file(ck), "weight_sha16": weight_sha(state),
               "params": params, "n_keys": len(state), "strict_load": compat}
        rec["hailo_val"] = score(model, "hailo_val", device)
        out["checkpoints"]["%s/%s" % (arm, tag)] = rec
        v = rec["hailo_val"]
        print("%s/%s EPE %.7f D1 %.4f%% | <64 %.4f 64-96 %.4f 96-128 %.4f >=128 %.4f | contract %s"
              % (arm, tag, v["metrics"]["epe"], v["metrics"]["d1"],
                 v["gt_lt_64"].get("epe", float("nan")), v["gt_64_96"].get("epe", float("nan")),
                 v["gt_96_128"].get("epe", float("nan")), v["gt_ge_128"].get("epe", float("nan")),
                 v["guard"]["contract_match"]), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    tmp = PILOT / "tier2_eval.json.tmp"
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, PILOT / "tier2_eval.json")
    print("wrote", PILOT / "tier2_eval.json")


if __name__ == "__main__":
    main()
