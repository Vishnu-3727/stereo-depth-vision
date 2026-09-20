"""PHASE-2 STEP 1b — break the GT/evidence confound. INFERENCE ONLY, NO TRAINING.

The per-bin diagnostic shows two things that both worsen monotonically with GT
disparity, and are therefore confounded:

  (i)  raw matching evidence at the true candidate weakens
  (ii) training-pixel density collapses

This probe conditions WITHIN a GT bin on the strength of the raw matching
evidence (z at the true candidate in the un-aggregated volume). If the
high-disparity collapse were driven by matching failure, then inside a fixed GT
bin the pixels with STRONG raw evidence should be predicted much better than
those with weak evidence. If error is flat in evidence strength inside the bin,
matching strength is not what is driving the collapse there.

Also reports, for the same pixels, the within-bin correlation between signed
error and the aggregated/raw evidence, and the per-scene relationship between a
scene's high-disparity pixel count and its high-disparity error.

Nothing is trained. No ARM-V artifact is modified.
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

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

SPACING, NDISP = 8.0, 24
MAXD = (NDISP - 1) * SPACING
SEEDS = {0: "arm_v", 1: "arm_v_s1", 2: "arm_v_s2"}
BINS = [(32, 48), (48, 64), (64, 80), (80, 96), (96, 112), (112, 128)]
MIN_CELL = 500


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"note": "inference only; no training; GT used only to stratify",
           "seeds": {}}
    for seed, run in SEEDS.items():
        blob = torch.load(REPO / "phase1" / "runs" / run / "arm_v_best.pth",
                          map_location="cpu", weights_only=False)
        m = StereoNet(StereoNetConfig(downsample_levels=3, num_disparities=NDISP,
                                      cost_volume_shift="right",
                                      regression_normalize=True))
        m.load_state_dict(blob["model"], strict=True)
        m.eval().to(dev)
        ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                             disparity_scale=256.0, occluded=True)
        G, P, ZR, ZA, IX, SCENE = [], [], [], [], [], []
        per_scene = []
        for i in range(len(ds)):
            s = ds[i]
            gt = s.disparity.astype(np.float32)
            v = gt > 0
            with torch.no_grad():
                L = torch.from_numpy(normalize(s.left)).to(dev)
                R = torch.from_numpy(normalize(s.right)).to(dev)
                o, st = m(L, R, return_stages=True)
                gt_t = torch.from_numpy(gt).to(dev)[None, None]
                k = (gt_t.clamp(0, MAXD) / SPACING).round().clamp(0, NDISP - 1).long()
                agg = F.interpolate(st["aggregated_cost"], size=gt.shape,
                                    mode="bilinear", align_corners=True)
                za = ((agg - agg.mean(1, True)) / (agg.std(1, True) + 1e-6)).gather(1, k)
                raw = F.interpolate(st["cost_volume"].norm(p=2, dim=1), size=gt.shape,
                                    mode="bilinear", align_corners=True)
                zr = ((raw - raw.mean(1, True)) / (raw.std(1, True) + 1e-6)).gather(1, k)
            f = lambda t: t[0, 0].cpu().numpy()[v].astype(np.float64)
            g_, p_ = gt[v].astype(np.float64), f(o)
            G.append(g_); P.append(p_); ZR.append(f(zr)); ZA.append(f(za))
            IX.append(f(st["disparity_initial"]))
            SCENE.append(np.full(g_.size, i, dtype=np.int32))
            hi = g_ >= 64
            per_scene.append({"scene": s.name, "high_px": int(hi.sum()),
                              "high_frac": float(hi.mean()),
                              "high_epe": float(np.abs(p_[hi] - g_[hi]).mean()) if hi.any() else None,
                              "epe": float(np.abs(p_ - g_).mean())})
        G, P, ZR, ZA, IX = map(np.concatenate, (G, P, ZR, ZA, IX))
        rec = {"bins": [], "per_scene": per_scene}
        for lo, hi in BINS:
            msk = (G >= lo) & (G < hi)
            if msk.sum() < 4 * MIN_CELL:
                rec["bins"].append({"lo": lo, "hi": hi, "px": int(msk.sum()),
                                    "status": "INSUFFICIENT"})
                continue
            g, p, zr, za, ix = G[msk], P[msk], ZR[msk], ZA[msk], IX[msk]
            err = np.abs(p - g)
            qs = np.quantile(zr, [0.25, 0.5, 0.75])
            cells = []
            edges = [-np.inf] + list(qs) + [np.inf]
            for a, b in zip(edges[:-1], edges[1:]):
                c = (zr >= a) & (zr < b)
                if c.sum() < MIN_CELL:
                    cells.append({"z_raw_lo": None, "status": "INSUFFICIENT"})
                    continue
                cells.append({
                    "z_raw_lo": None if a == -np.inf else float(a),
                    "z_raw_hi": None if b == np.inf else float(b),
                    "px": int(c.sum()), "mean_z_raw": float(zr[c].mean()),
                    "epe": float(err[c].mean()),
                    "signed_err": float((p[c] - g[c]).mean()),
                    "mean_gt": float(g[c].mean()), "mean_pred": float(p[c].mean()),
                    "mean_init_idx": float(ix[c].mean()),
                })
            rec["bins"].append({
                "lo": lo, "hi": hi, "px": int(msk.sum()), "status": "ok",
                "epe": float(err.mean()),
                "z_raw_quartile_cells": cells,
                "epe_spread_across_z_raw_quartiles":
                    (max(c["epe"] for c in cells if "epe" in c)
                     - min(c["epe"] for c in cells if "epe" in c)),
                "corr_signed_err_vs_z_raw": float(np.corrcoef(zr, p - g)[0, 1]),
                "corr_signed_err_vs_z_agg": float(np.corrcoef(za, p - g)[0, 1]),
                "corr_init_idx_vs_gt_within_bin": float(np.corrcoef(ix, g)[0, 1]),
                "corr_pred_vs_gt_within_bin": float(np.corrcoef(p, g)[0, 1]),
            })
        out["seeds"][str(seed)] = rec
        print("seed", seed, "done", flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    (HERE / "conditional.json").write_text(json.dumps(out, indent=2))

    for s in (0, 1, 2):
        print()
        print("seed", s)
        print("| GT bin | px | EPE | EPE spread across z_raw quartiles | r(signed err, z_raw) | r(signed err, z_agg) | r(init idx, GT) | r(pred, GT) |")
        print("|---|---|---|---|---|---|---|---|")
        for b in out["seeds"][str(s)]["bins"]:
            if b["status"] != "ok":
                print("| [%d,%d) | %d | INSUFFICIENT | | | | | |" % (b["lo"], b["hi"], b["px"]))
                continue
            print("| [%d,%d) | %d | %.3f | %.3f | %+.4f | %+.4f | %+.4f | %+.4f |" % (
                b["lo"], b["hi"], b["px"], b["epe"],
                b["epe_spread_across_z_raw_quartiles"],
                b["corr_signed_err_vs_z_raw"], b["corr_signed_err_vs_z_agg"],
                b["corr_init_idx_vs_gt_within_bin"], b["corr_pred_vs_gt_within_bin"]))
        print()
        print("  z_raw quartile cells, GT [64,80) and [96,112):")
        for b in out["seeds"][str(s)]["bins"]:
            if b["status"] == "ok" and b["lo"] in (64, 96):
                for c in b["z_raw_quartile_cells"]:
                    if "epe" in c:
                        print("    GT[%d,%d) z_raw~%.3f px=%d EPE=%.3f signed=%+.3f pred=%.2f gt=%.2f"
                              % (b["lo"], b["hi"], c["mean_z_raw"], c["px"], c["epe"],
                                 c["signed_err"], c["mean_pred"], c["mean_gt"]))


if __name__ == "__main__":
    main()
