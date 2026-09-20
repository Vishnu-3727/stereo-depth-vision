"""Disparity-range ceiling of the current architecture, measured on hailo_val.

The cost volume has 12 candidates at stride 16, so max_disparity_px = 176.
This asks what that alone costs, before any model quality question.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from phase2.viz import core
from src.evaluation.metrics import disparity_metrics

vals = []
for i in range(40):
    s = core.load_scene(i, split="hailo_val")
    vals.append(s.gt_disparity[s.gt_valid].astype(np.float64))
gt = np.concatenate(vals)
clipped = np.clip(gt, 0.0, 176.0)
m = disparity_metrics(clipped, gt)
out = {
    "valid_pixels": int(gt.size),
    "max_gt_disparity_px": float(gt.max()),
    "percentiles": {p: float(np.percentile(gt, p)) for p in (50, 90, 99, 99.9)},
    "fraction_above_176px": float((gt > 176.0).mean()),
    "fraction_above_88px": float((gt > 88.0).mean()),
    "oracle_clip_to_176_epe": m.epe,
    "oracle_clip_to_176_d1": m.d1,
}
print(json.dumps(out, indent=2))
Path("phase2/results/architecture_ceilings").mkdir(parents=True, exist_ok=True)
Path("phase2/results/architecture_ceilings/gt_range_hailo_val.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
