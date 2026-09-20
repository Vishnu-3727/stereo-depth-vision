"""EXP-P2A-SCALE-COVERAGE-001 — frozen evaluation + preregistered metrics.

Scores the three P2A checkpoints under the UNCHANGED frozen contract via the
established score_mirror mechanism for this architecture (dataset, GT scale,
valid mask, pooled metrics and contract guard imported unmodified from
phase1.harness.frozen_eval), then computes every metric the preregistration
requires. NO evaluation-time scale augmentation. NO multi-scale inference.

Usage: python phase2/scripts/eval_p2a.py [best|final]
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from phase1.harness.frozen_eval import (pooled_metrics, refuse_unless_contract,
                                        sha256_file, strict_load_report, weight_sha)
from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

RUNS = {0: "p2a_scale_coverage", 1: "p2a_scale_coverage_s1", 2: "p2a_scale_coverage_s2"}
BINS = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
MIN_BIN = 1000
CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)

MIRROR_NOTE = (
    "frozen_eval.score_checkpoint hardcodes StereoNetConfig() (downsample_levels=4, "
    "num_disparities=12, cost_volume_shift=none, regression_normalize=False), which "
    "would evaluate these weights through a different forward graph than trained. "
    "Scores use the established line-for-line mirror of score_checkpoint with "
    "StereoNetConfig(downsample_levels=3, num_disparities=24, cost_volume_shift='right', "
    "regression_normalize=True) -- the same mirror ARM-V seeds 1 and 2 used. Dataset "
    "split, GT scale, valid mask, pooled metrics and contract guard are imported "
    "unmodified from phase1.harness.frozen_eval.")


def score(model, split: str, device: str) -> dict:
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split=split,
                         disparity_scale=256.0, occluded=True)
    preds, gts, names = [], [], []
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
            names.append(s.name)
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
        row = {"lo": lo, "hi": hi, "px": int(mm.sum()),
               "frac_px": float(mm.mean())}
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
        "contract_match": None,
        "note": "training split; NOT a frozen-contract score"}
    return {"split": split, "scenes": len(ds), "names": names if split == "hailo_val" else None,
            "metrics": m, "guard": guard,
            "gt_lt_64": stratum(G < 64), "gt_ge_64": stratum(G >= 64),
            "gt_ge_96": stratum(G >= 96), "bins": bins,
            "max_pred_valid_px": float(P.max()), "max_pred_all_px": pmax_all,
            "max_gt": float(G.max())}


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "best"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except Exception:
        head = None
    out = {"experiment": "EXP-P2A-SCALE-COVERAGE-001", "checkpoint_tag": tag,
           "mirror_note": MIRROR_NOTE,
           "metadata": {"utc": datetime.now(timezone.utc).isoformat(), "git_head": head,
                        "python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform(),
                        "device": device, "eval_time_augmentation": "none",
                        "multi_scale_inference": False},
           "seeds": {}}
    for seed, run in RUNS.items():
        ck = REPO / "phase2" / "runs" / run / ("p2a_%s.pth" % tag)
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        state = blob["model"]
        model = StereoNet(StereoNetConfig(**CFG))
        compat = strict_load_report(model, state)
        model.eval().to(device)
        ref = torch.load(REPO / "phase1" / "runs" / "arm_v" / "arm_v_best.pth",
                         map_location="cpu", weights_only=False)["model"]
        rec = {"run_dir": "phase2/runs/" + run,
               "checkpoint": "phase2/runs/%s/p2a_%s.pth" % (run, tag),
               "sha256": sha256_file(ck), "weight_sha16": weight_sha(state),
               "params": sum(p.numel() for p in model.parameters()),
               "state_dict_keys_match_arm_v": sorted(state.keys()) == sorted(ref.keys()),
               "n_keys": len(state), "strict_load": compat,
               "wall_clock_s": json.loads((REPO / "phase2" / "runs" / run
                                           / "p2a_record.json").read_text())["wall_clock_s"],
               "integrity_guard": json.loads((REPO / "phase2" / "runs" / run
                                              / "integrity_guard.json").read_text())}
        rec["hailo_val"] = score(model, "hailo_val", device)
        rec["hailo_calib"] = score(model, "hailo_calib", device)
        out["seeds"][str(seed)] = rec
        v = rec["hailo_val"]
        print("seed %d  EPE %.7f  D1 %.4f%%  | GT>=64 %.3f  GT>=96 %.3f  slope_hi %+.4f  "
              "R2_hi %.4f | GT<64 %.4f | maxpred %.2f | contract %s"
              % (seed, v["metrics"]["epe"], v["metrics"]["d1"], v["gt_ge_64"]["epe"],
                 v["gt_ge_96"]["epe"], v["gt_ge_96"]["slope"], v["gt_ge_96"]["r2"],
                 v["gt_lt_64"]["epe"], v["max_pred_valid_px"],
                 v["guard"]["contract_match"]), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    dst = REPO / "phase2" / "runs" / "p2a_scale_coverage" / ("p2a_eval_%s.json" % tag)
    dst.write_text(json.dumps(out, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
