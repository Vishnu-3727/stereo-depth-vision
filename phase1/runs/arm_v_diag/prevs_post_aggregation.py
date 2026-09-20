"""Is the disparity signal weak BEFORE aggregation, or destroyed BY it?

Compares two cost curves per pixel over the 24 candidates, on the frozen
evaluation scenes, for each ARM-V seed:

  RAW   : ||L(x) - R(x-k)||_2 over the 32 channels of the un-aggregated
          cost volume (the natural matching cost the volume encodes)
  AGG   : the aggregated Conv3d output the readout actually consumes

and reports, for each, how well the curve's minimum locates the true
disparity (GT/8 candidates) plus the curve's sharpness and modality.
Read-only; trains nothing.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

dev = "cuda" if torch.cuda.is_available() else "cpu"
SP, ND = 8.0, 24


def curve_stats(c, gt_idx, valid):
    """c: (1, 24, h, w) at 1/8 res; gt_idx/valid: full-res numpy."""
    cn = (c - c.mean(1, keepdim=True)) / (c.std(1, keepdim=True) + 1e-6)
    p = torch.softmax(-cn, 1)
    ent = -(p * torch.log(p + 1e-12)).sum(1, keepdim=True)
    amin = c.argmin(1, keepdim=True).float()
    ln = torch.cat([c[:, :1], c[:, :-1]], 1)
    rn = torch.cat([c[:, 1:], c[:, -1:]], 1)
    modes = ((c <= ln) & (c <= rn)).sum(1, keepdim=True).float()
    soft = (p * torch.arange(ND, device=c.device, dtype=c.dtype).view(1, -1, 1, 1)).sum(1, keepdim=True)
    up = lambda t: F.interpolate(t, size=gt_idx.shape, mode="nearest")[0, 0].cpu().numpy().astype(np.float64)
    a, e, m, s = up(amin), up(ent), up(modes), up(soft)
    g = gt_idx[valid]
    return {"argmin": a[valid], "ent": e[valid], "modes": m[valid],
            "soft": s[valid], "gt_idx": g}


out = {}
for run in ("arm_v", "arm_v_s1", "arm_v_s2"):
    blob = torch.load(REPO / "phase1" / "runs" / run / "arm_v_best.pth",
                      map_location="cpu", weights_only=False)
    mdl = StereoNet(StereoNetConfig(downsample_levels=3, num_disparities=ND,
                                    cost_volume_shift="right",
                                    regression_normalize=True))
    mdl.load_state_dict(blob["model"], strict=True)
    mdl.eval().to(dev)
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    acc = {"raw": [], "agg": []}
    for i in range(len(ds)):
        s = ds[i]
        gt = s.disparity.astype(np.float64)
        valid = gt > 0
        gt_idx = np.clip(gt, 0, (ND - 1) * SP) / SP
        with torch.no_grad():
            _, st = mdl(torch.from_numpy(normalize(s.left)).to(dev),
                        torch.from_numpy(normalize(s.right)).to(dev),
                        return_stages=True)
            raw = st["cost_volume"].norm(p=2, dim=1)        # (1,24,h,w)
            agg = st["aggregated_cost"]                      # (1,24,h,w)
        acc["raw"].append(curve_stats(raw, gt_idx, valid))
        acc["agg"].append(curve_stats(agg, gt_idx, valid))
    rec = {}
    for tag in ("raw", "agg"):
        A = {k: np.concatenate([d[k] for d in acc[tag]]) for k in acc[tag][0]}
        rec[tag] = {
            "argmin_within_1_candidate": float((np.abs(A["argmin"] - A["gt_idx"]) <= 1).mean()),
            "argmin_within_2_candidates": float((np.abs(A["argmin"] - A["gt_idx"]) <= 2).mean()),
            "epe_of_hard_argmin_x8": float((np.abs(A["argmin"] - A["gt_idx"]) * SP).mean()),
            "corr_argmin_gtidx": float(np.corrcoef(A["argmin"], A["gt_idx"])[0, 1]),
            "corr_softargmin_gtidx": float(np.corrcoef(A["soft"], A["gt_idx"])[0, 1]),
            "epe_of_softargmin_x8": float((np.abs(A["soft"] - A["gt_idx"]) * SP).mean()),
            "entropy_mean": float(A["ent"].mean()),
            "entropy_max_possible": float(np.log(ND)),
            "local_minima_mean": float(A["modes"].mean()),
            "unimodal_frac": float((A["modes"] <= 1).mean()),
        }
    out[run] = rec
    print(run)
    for tag in ("raw", "agg"):
        print("  " + tag.upper().ljust(4),
              " ".join(k + "=" + format(v, ".4f") for k, v in rec[tag].items()))

(Path(__file__).resolve().parent / "pre_vs_post_aggregation.json").write_text(json.dumps(out, indent=2))
