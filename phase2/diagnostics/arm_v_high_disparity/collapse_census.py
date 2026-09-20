"""PHASE-2 STEP 1c — census of the high-disparity collapse across FROZEN Phase-1
checkpoints. INFERENCE ONLY, NO TRAINING.

Purpose, stated precisely so it cannot be mistaken for reopening anything:
ARM-V's three seeds do not behave the same way above ~80 px. Two of them
decouple their output from the true disparity; one does not. With n = 3 that
cannot be told apart from coincidence. This probe measures the SAME collapse
statistic on every frozen Phase-1 checkpoint that exists, purely to find out
whether the collapse is a reproducible training outcome or a one-off.

It does NOT re-score, re-rank, retune, reinterpret or revisit the verdict of any
arm. No arm's accuracy verdict appears here. ARM-W, ARM-X, ARM-Y and ARM-Z stay
closed on their recorded verdicts; their checkpoints are used only as additional
samples of a training-dynamics question that is new in Phase 2.

Collapse statistic, fixed before running:
    r_hi = Pearson correlation between predicted and true disparity over the
           evaluation pixels with GT >= 96 px (21,249 + 11,823 + 1,071 px).
    slope_hi = least-squares slope of pred on GT over the same pixels.
A run is called COLLAPSED when slope_hi < 0.25, RETAINED when slope_hi > 0.50,
and INTERMEDIATE in between. Thresholds are descriptive labels for reporting,
not an acceptance rule for any experiment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

V = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
         regression_normalize=True)
RUNS = [
    ("ARM-U", 0, "arm_u", "arm_u_best.pth",
     dict(downsample_levels=4, num_disparities=12, cost_volume_shift="right",
          regression_normalize=True)),
    ("ARM-U", 1, "arm_u_s1", "arm_u_best.pth",
     dict(downsample_levels=4, num_disparities=12, cost_volume_shift="right",
          regression_normalize=True)),
    ("ARM-U", 2, "arm_u_s2", "arm_u_best.pth",
     dict(downsample_levels=4, num_disparities=12, cost_volume_shift="right",
          regression_normalize=True)),
    ("ARM-V", 0, "arm_v", "arm_v_best.pth", dict(V)),
    ("ARM-V", 1, "arm_v_s1", "arm_v_best.pth", dict(V)),
    ("ARM-V", 2, "arm_v_s2", "arm_v_best.pth", dict(V)),
    ("ARM-W", 0, "arm_w", "arm_w_best.pth", dict(V, cost_volume_groups=8)),
    ("ARM-X", 0, "arm_x", "arm_x_best.pth", dict(V)),
    ("ARM-X", 1, "arm_x_s1", "arm_x_best.pth", dict(V)),
    ("ARM-X", 2, "arm_x_s2", "arm_x_best.pth", dict(V)),
    ("ARM-Y", 0, "arm_y", "arm_y_best.pth", dict(V, cost_volume_excitation=True)),
    ("ARM-Y", 1, "arm_y_s1", "arm_y_best.pth", dict(V, cost_volume_excitation=True)),
    ("ARM-Y", 2, "arm_y_s2", "arm_y_best.pth", dict(V, cost_volume_excitation=True)),
    ("ARM-Z", 0, "arm_z", "arm_z_best.pth", dict(V, feature_normalize=True)),
    ("ARM-Z", 1, "arm_z_s1", "arm_z_best.pth", dict(V, feature_normalize=True)),
    ("ARM-Z", 2, "arm_z_s2", "arm_z_best.pth", dict(V, feature_normalize=True)),
]
CUT_HI = 96.0


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    rows = []
    for arm, seed, run, ck, cfg in RUNS:
        path = REPO / "phase1" / "runs" / run / ck
        if not path.exists():
            rows.append({"arm": arm, "seed": seed, "status": "CHECKPOINT MISSING"})
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        m = StereoNet(StereoNetConfig(**cfg))
        m.load_state_dict(state, strict=True)
        m.eval().to(dev)
        G, P = [], []
        pmax = -1e30
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]
                o = m(torch.from_numpy(normalize(s.left)).to(dev),
                      torch.from_numpy(normalize(s.right)).to(dev))
                p = o[0, 0].cpu().numpy().astype(np.float64)
                pmax = max(pmax, float(p.max()))
                v = s.disparity > 0
                G.append(s.disparity[v].astype(np.float64)); P.append(p[v])
        G, P = np.concatenate(G), np.concatenate(P)
        hi = G >= CUT_HI
        a, b = np.polyfit(G[hi], P[hi], 1)
        r = float(np.corrcoef(G[hi], P[hi])[0, 1])
        label = "COLLAPSED" if a < 0.25 else ("RETAINED" if a > 0.50 else "INTERMEDIATE")
        row = {"arm": arm, "seed": seed, "run": "phase1/runs/" + run,
               "status": "ok", "global_epe": float(np.abs(P - G).mean()),
               "px_ge_96": int(hi.sum()),
               "epe_ge_96": float(np.abs(P[hi] - G[hi]).mean()),
               "mean_pred_ge_96": float(P[hi].mean()),
               "mean_gt_ge_96": float(G[hi].mean()),
               "slope_hi": float(a), "intercept_hi": float(b), "pearson_hi": r,
               "max_pred_valid_px": float(P.max()),
               "max_pred_all_px": pmax, "label": label}
        rows.append(row)
        print("%-6s s%d  EPE %7.4f | GT>=96: EPE %7.3f pred %6.2f slope %+.4f r %+.4f  max_pred %6.2f  %s"
              % (arm, seed, row["global_epe"], row["epe_ge_96"], row["mean_pred_ge_96"],
                 a, r, row["max_pred_valid_px"], label), flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = {"labels": {lab: sum(1 for r in ok if r["label"] == lab)
                          for lab in ("COLLAPSED", "INTERMEDIATE", "RETAINED")},
               "n_runs": len(ok)}
    (HERE / "collapse_census.json").write_text(json.dumps(
        {"cut_px": CUT_HI, "summary": summary, "runs": rows}, indent=2))
    print()
    print("labels:", summary)


if __name__ == "__main__":
    main()
