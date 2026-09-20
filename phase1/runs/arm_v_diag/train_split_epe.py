"""Train-split (hailo_calib) vs eval-split (hailo_val) EPE for ARM-V. Read-only.

Same forward graph, same GT scale (1/256), same top-left 368x1232 crop, same
valid = gt > 0 pooled-pixel metric as the frozen contract -- applied to the 160
TRAINING scenes, purely to separate "capacity-limited / underfit" from
"generalisation-limited / overfit". This is NOT a frozen-contract score and is
never reported as one.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np
import torch

from phase1.harness.frozen_eval import pooled_metrics
from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

dev = "cuda" if torch.cuda.is_available() else "cpu"
out = {}
for run in ("arm_v", "arm_v_s1", "arm_v_s2"):
    blob = torch.load(REPO / "phase1" / "runs" / run / "arm_v_best.pth",
                      map_location="cpu", weights_only=False)
    m = StereoNet(StereoNetConfig(downsample_levels=3, num_disparities=24,
                                  cost_volume_shift="right",
                                  regression_normalize=True))
    m.load_state_dict(blob["model"], strict=True)
    m.eval().to(dev)
    row = {}
    for split in ("hailo_calib", "hailo_val"):
        ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split=split,
                             disparity_scale=256.0, occluded=True)
        preds, gts, big_e, big_n = [], [], 0.0, 0
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]
                o = m(torch.from_numpy(normalize(s.left)).to(dev),
                      torch.from_numpy(normalize(s.right)).to(dev))
                p = o[0, 0].cpu().numpy().astype(np.float64)
                v = s.disparity > 0
                g = s.disparity[v].astype(np.float64)
                preds.append(p[v]); gts.append(g)
                bm = g > 96
                big_e += float(np.abs(p[v][bm] - g[bm]).sum()); big_n += int(bm.sum())
        mm = pooled_metrics(preds, gts)
        row[split] = {"epe": mm["epe"], "d1": mm["d1"], "px": mm["valid_pixels"],
                      "epe_gt_over_96": (big_e / big_n) if big_n else None,
                      "px_gt_over_96": big_n}
    row["generalisation_gap_px"] = row["hailo_val"]["epe"] - row["hailo_calib"]["epe"]
    out[run] = row
    print(run, json.dumps(row))

(Path(__file__).resolve().parent / "train_split_epe.json").write_text(json.dumps(out, indent=2))
