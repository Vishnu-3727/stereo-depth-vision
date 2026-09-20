"""Train-curve + GT-distribution probe for the ARM-V bottleneck audit. Read-only."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np

print("== ARM-V training curves (train loss vs 10-scene val EPE) ==")
for run in ("arm_v", "arm_v_s1", "arm_v_s2"):
    rows = [json.loads(l) for l in (REPO / "phase1" / "runs" / run
                                    / "training_log.jsonl").read_text().splitlines() if l.strip()]
    keys = sorted(rows[0].keys())
    print(run, "n=", len(rows), "keys=", keys)
    for i in (0, 1, 9, 24, 49, 99, 149, 179, 199):
        if i < len(rows):
            r = rows[i]
            print("  ", {k: (round(v, 5) if isinstance(v, float) else v)
                         for k, v in r.items()})

print()
print("== GT disparity distribution: train split vs eval split ==")
from src.datasets.kitti2015 import Kitti2015Stereo

for split in ("hailo_calib", "hailo_val"):
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split=split,
                         disparity_scale=256.0, occluded=True)
    vals = []
    for i in range(len(ds)):
        d = ds[i].disparity
        vals.append(d[d > 0].astype(np.float32))
    v = np.concatenate(vals)
    edges = [0, 8, 16, 32, 64, 96, 128, 160, 184, 1e9]
    h = [float(((v >= a) & (v < b)).mean()) for a, b in zip(edges[:-1], edges[1:])]
    print(split, "scenes", len(ds), "px", v.size,
          "mean", round(float(v.mean()), 3), "p99", round(float(np.percentile(v, 99)), 2),
          "max", round(float(v.max()), 2))
    print("   frac by gt bin", [round(x, 5) for x in h])
    print("   frac gt>64", round(float((v > 64).mean()), 5),
          " frac gt>96", round(float((v > 96).mean()), 5),
          " frac gt>184", round(float((v > 184).mean()), 6))
