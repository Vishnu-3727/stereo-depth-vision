"""PHASE-2 STEP 1e — test-time scale transfer. INFERENCE ONLY, NO TRAINING.

The decisive question the per-bin diagnostic leaves open: is the high-disparity
collapse a property of the PIXELS (scene content, matching difficulty at those
locations) or of the DISPARITY VALUE those pixels happen to carry?

Test: present the SAME evaluation pixels to the SAME frozen ARM-V weights with
the stereo pair spatially downscaled by a factor s. Downscaling by s multiplies
every true disparity by s, so a 110 px pixel is presented to the network as a
55 px pixel at s = 0.5 — inside the range the model handles well. The prediction
is then divided by s to return to original-image units and scored against the
unchanged ground truth on the unchanged frozen-contract pixel set.

Downscaling destroys spatial detail, so it should HURT. If high-disparity error
nevertheless falls sharply, the collapse is a function of the disparity value
presented, not of those pixels' matching difficulty.

No weights change. No training. GT is used only to score and stratify.
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
import torch.nn.functional as F

from src.datasets.kitti2015 import Kitti2015Stereo, NORM_MEAN, NORM_STD
from src.models.stereonet import StereoNet, StereoNetConfig

SEEDS = {0: "arm_v", 1: "arm_v_s1", 2: "arm_v_s2"}
SCALES = [1.0, 0.75, 0.5]
BINS = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
MIN_BIN = 1000


def to_tensor(img: np.ndarray, dev: str) -> torch.Tensor:
    x = (np.asarray(img, dtype=np.float32) - NORM_MEAN) / NORM_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None])).to(dev)


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    out = {"scales": SCALES, "bins": BINS, "note": "inference only; no training",
           "seeds": {}}
    for seed, run in SEEDS.items():
        blob = torch.load(REPO / "phase1" / "runs" / run / "arm_v_best.pth",
                          map_location="cpu", weights_only=False)
        m = StereoNet(StereoNetConfig(downsample_levels=3, num_disparities=24,
                                      cost_volume_shift="right",
                                      regression_normalize=True))
        m.load_state_dict(blob["model"], strict=True)
        m.eval().to(dev)
        G = []
        P = {s: [] for s in SCALES}
        pmax = {s: -1e30 for s in SCALES}
        for i in range(len(ds)):
            smp = ds[i]
            gt = smp.disparity.astype(np.float64)
            v = gt > 0
            H, W = gt.shape
            L0 = to_tensor(smp.left, dev)
            R0 = to_tensor(smp.right, dev)
            G.append(gt[v])
            for s in SCALES:
                with torch.no_grad():
                    if s == 1.0:
                        L, R = L0, R0
                    else:
                        h, w = int(round(H * s)), int(round(W * s))
                        L = F.interpolate(L0, size=(h, w), mode="bilinear",
                                          align_corners=False)
                        R = F.interpolate(R0, size=(h, w), mode="bilinear",
                                          align_corners=False)
                    o = m(L, R)
                    # back to original-image disparity units: undo the spatial
                    # downscale, then undo the disparity scaling
                    if s != 1.0:
                        o = F.interpolate(o, size=(H, W), mode="bilinear",
                                          align_corners=False) / s
                p = o[0, 0].cpu().numpy().astype(np.float64)
                pmax[s] = max(pmax[s], float(p.max()))
                P[s].append(p[v])
        G = np.concatenate(G)
        rec = {"per_scale": {}}
        for s in SCALES:
            p = np.concatenate(P[s])
            err = np.abs(p - G)
            hi = G >= 96
            mid = (G >= 64) & (G < 96)
            a, _ = np.polyfit(G[hi], p[hi], 1)
            bins = []
            for lo, h_ in zip(BINS[:-1], BINS[1:]):
                mm = (G >= lo) & (G < h_)
                bins.append({"lo": lo, "hi": h_, "px": int(mm.sum()),
                             "epe": float(err[mm].mean()) if mm.sum() >= MIN_BIN else None,
                             "signed": float((p[mm] - G[mm]).mean()) if mm.sum() >= MIN_BIN else None})
            rec["per_scale"][str(s)] = {
                "global_epe": float(err.mean()),
                "global_d1": float((((err > 3) & (err > 0.05 * G)).mean()) * 100),
                "epe_gt_lt_64": float(err[G < 64].mean()),
                "epe_gt_64_96": float(err[mid].mean()),
                "epe_gt_ge_96": float(err[hi].mean()),
                "signed_gt_ge_96": float((p[hi] - G[hi]).mean()),
                "slope_hi": float(a),
                "pearson_hi": float(np.corrcoef(G[hi], p[hi])[0, 1]),
                "max_pred_valid_px": float(p.max()),
                "max_pred_all_px": pmax[s],
                "bins": bins}
        out["seeds"][str(seed)] = rec
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
        print("seed", seed, "done", flush=True)

    (HERE / "scale_transfer.json").write_text(json.dumps(out, indent=2))

    print()
    print("| seed | scale | global EPE | EPE GT<64 | EPE GT 64-96 | EPE GT>=96 | signed GT>=96 | slope_hi | r_hi | max pred |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for seed in SEEDS:
        for s in SCALES:
            r = out["seeds"][str(seed)]["per_scale"][str(s)]
            print("| %d | %.2f | %.4f | %.3f | %.3f | %.3f | %+.2f | %+.4f | %+.4f | %.1f |" % (
                seed, s, r["global_epe"], r["epe_gt_lt_64"], r["epe_gt_64_96"],
                r["epe_gt_ge_96"], r["signed_gt_ge_96"], r["slope_hi"],
                r["pearson_hi"], r["max_pred_valid_px"]))


if __name__ == "__main__":
    main()
