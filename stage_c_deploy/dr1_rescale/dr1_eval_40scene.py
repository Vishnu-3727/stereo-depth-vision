"""DR-1 frozen 40-scene accuracy gate (STAGE C, PHASE DR-1). INFERENCE ONLY.

Uses CUDA (the frozen EPE 1.1912168 px was measured on cuda per tier2_eval.json
metadata). Original path calls model() directly -- scoring core mirrors
stage_b_armp/.../scripts/eval_tier2.py score() + phase1.harness.frozen_eval
(pooled_metrics, refuse_unless_contract, gt_scale 256.0, disp_occ_0).
Rescaled path: same captured aggregated_cost / 1e17 through the model's OWN
regression/refinement modules (Variant A, no weight change).

FIRST validates: module path at s=1 reproduces model() before any rescaled
measurement is trusted.

Writes: stage_c_deploy/dr1_rescale/dr1_eval_40scene.json (new file only).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from phase1.harness.frozen_eval import pooled_metrics, refuse_unless_contract
from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

OUT = REPO / "stage_c_deploy" / "dr1_rescale"
ARMP_REL = "stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth"
ARMP_SHA = "b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454"
CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
S = 1e17
FROZEN_EPE = 1.1912168


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def downstream(net, agg, left):
    with torch.no_grad():
        size = (left.shape[-2], left.shape[-1])
        disp_init = net.regression(agg, size)
        residual = net.refinement(disp_init, left)
        disp = disp_init + residual
        if net.config.final_relu:
            disp = torch.relu(disp)
    return disp


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True,
                                  timeout=15).stdout.strip()
    except Exception:
        git_head = None
    ckpt = REPO / ARMP_REL
    assert sha256(ckpt) == ARMP_SHA, "checkpoint hash changed -> STOP"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = StereoNet(StereoNetConfig(**CFG))
    net.load_state_dict(blob["model"], strict=True)
    net.eval().to(device)

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    assert len(ds) == 40, f"expected 40 scenes, got {len(ds)} -> STOP"

    t0 = time.time()
    preds_orig, preds_rs, gts = [], [], []
    s1_maxdiff = 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            sm = ds[i]
            tl = torch.from_numpy(normalize(sm.left)).to(device)
            tr = torch.from_numpy(normalize(sm.right)).to(device)
            out_orig = net(tl, tr)
            _, st = net(tl, tr, return_stages=True)
            dA = downstream(net, st["aggregated_cost"], tl)
            if i < 5:
                s1_maxdiff = max(s1_maxdiff,
                                 float((dA - out_orig).abs().max().item()))
            dB = downstream(net, st["aggregated_cost"] / S, tl)
            valid = sm.disparity > 0
            preds_orig.append(out_orig[0, 0].cpu().numpy().astype(np.float64)[valid])
            preds_rs.append(dB[0, 0].cpu().numpy().astype(np.float64)[valid])
            gts.append(sm.disparity[valid].astype(np.float64))
            if (i + 1) % 10 == 0:
                print(f"scene {i + 1}/40 done", flush=True)
    m_orig = pooled_metrics(preds_orig, gts)
    m_rs = pooled_metrics(preds_rs, gts)
    g_orig = refuse_unless_contract(len(ds), m_orig["valid_pixels"], 256.0,
                                    "hailo_val", "disp_occ_0")
    g_rs = refuse_unless_contract(len(ds), m_rs["valid_pixels"], 256.0,
                                  "hailo_val", "disp_occ_0")
    rec = {
        "provenance": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "git_head": git_head, "host": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "device": device,
            "command": "python stage_c_deploy/dr1_rescale/dr1_eval_40scene.py",
            "argv": sys.argv,
            "checkpoint": ARMP_REL, "checkpoint_sha256": ARMP_SHA,
            "scalar_s": S, "variant": "A (activation rescale)",
            "scoring_core": ("mirror of eval_tier2.py score() + "
                             "phase1.harness.frozen_eval; evaluator unmodified"),
            "elapsed_s": time.time() - t0,
            "c1_verdict_unchanged": "FAIL",
        },
        "s1_path_validation_maxdiff_5scenes": s1_maxdiff,
        "s1_reproduces_forward": bool(s1_maxdiff == 0.0),
        "frozen_epe_reference": FROZEN_EPE,
        "original_reproduces_frozen": bool(abs(m_orig["epe"] - FROZEN_EPE) < 5e-7),
        "original": {"epe": m_orig["epe"], "d1": m_orig["d1"],
                     "valid_pixels": m_orig["valid_pixels"], "guard": g_orig,
                     "full_metrics": m_orig},
        "rescaled": {"epe": m_rs["epe"], "d1": m_rs["d1"],
                     "valid_pixels": m_rs["valid_pixels"], "guard": g_rs,
                     "full_metrics": m_rs},
        "epe_delta_rescaled_minus_original": m_rs["epe"] - m_orig["epe"],
        "d1_delta_rescaled_minus_original": m_rs["d1"] - m_orig["d1"],
    }
    with open(OUT / "dr1_eval_40scene.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"s1 validation maxdiff={s1_maxdiff:.3e}; orig EPE={m_orig['epe']:.7f} "
          f"(frozen {FROZEN_EPE}); rs EPE={m_rs['epe']:.7f}; "
          f"delta={m_rs['epe'] - m_orig['epe']:.6e}; valid={m_orig['valid_pixels']}",
          flush=True)


if __name__ == "__main__":
    main()
