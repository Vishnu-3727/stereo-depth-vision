"""Output-range / saturation probe for ARM-V. Read-only, trains nothing.

Question: does the prediction saturate below the GT range, and is that where the
large-disparity error mass lives? Reports, pooled over the frozen eval contract
pixels (and over the training split for contrast): prediction percentiles,
signed bias per GT bin, and the initial-readout ceiling implied by the
soft-argmin index range.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np
import torch

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

dev = "cuda" if torch.cuda.is_available() else "cpu"
GEDGES = [0, 8, 16, 32, 64, 96, 128, 160]
out = {}
for run in ("arm_v", "arm_v_s1", "arm_v_s2"):
    blob = torch.load(REPO / "phase1" / "runs" / run / "arm_v_best.pth",
                      map_location="cpu", weights_only=False)
    m = StereoNet(StereoNetConfig(downsample_levels=3, num_disparities=24,
                                  cost_volume_shift="right",
                                  regression_normalize=True))
    m.load_state_dict(blob["model"], strict=True)
    m.eval().to(dev)
    rec = {}
    for split in ("hailo_val", "hailo_calib"):
        ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split=split,
                             disparity_scale=256.0, occluded=True)
        P, G, I = [], [], []
        pmax_all = -1e9
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]
                o, st = m(torch.from_numpy(normalize(s.left)).to(dev),
                          torch.from_numpy(normalize(s.right)).to(dev),
                          return_stages=True)
                p = o[0, 0].cpu().numpy().astype(np.float64)
                pmax_all = max(pmax_all, float(p.max()))
                v = s.disparity > 0
                P.append(p[v]); G.append(s.disparity[v].astype(np.float64))
                I.append(st["disparity_initial"][0, 0].cpu().numpy()[v].astype(np.float64))
        P, G, I = np.concatenate(P), np.concatenate(G), np.concatenate(I)
        bias = []
        for a, b in zip(GEDGES[:-1], GEDGES[1:]):
            k = (G >= a) & (G < b)
            if k.any():
                bias.append({"gt_lo": a, "gt_hi": b, "n": int(k.sum()),
                             "mean_pred": float(P[k].mean()),
                             "mean_gt": float(G[k].mean()),
                             "mean_signed_err": float((P[k] - G[k]).mean()),
                             "p95_pred": float(np.percentile(P[k], 95))})
        rec[split] = {
            "pred_pct": {q: float(np.percentile(P, q)) for q in (50, 90, 99, 99.9, 100)},
            "pred_max_over_all_px_incl_invalid": pmax_all,
            "gt_pct": {q: float(np.percentile(G, q)) for q in (50, 90, 99, 99.9, 100)},
            "init_index_max": float(I.max()),
            "frac_pred_over_96": float((P > 96).mean()),
            "frac_gt_over_96": float((G > 96).mean()),
            "bias_by_gt_bin": bias,
        }
    out[run] = rec
    print(run, "val pred pct", {k: round(v, 2) for k, v in rec["hailo_val"]["pred_pct"].items()},
          "| gt pct", {k: round(v, 2) for k, v in rec["hailo_val"]["gt_pct"].items()})
    for b in rec["hailo_val"]["bias_by_gt_bin"]:
        print("   gt[%3d,%3d) n=%8d  mean_gt=%7.2f  mean_pred=%7.2f  bias=%8.2f  p95_pred=%7.2f"
              % (b["gt_lo"], b["gt_hi"], b["n"], b["mean_gt"], b["mean_pred"],
                 b["mean_signed_err"], b["p95_pred"]))

(Path(__file__).resolve().parent / "ceiling_probe.json").write_text(json.dumps(out, indent=2))
