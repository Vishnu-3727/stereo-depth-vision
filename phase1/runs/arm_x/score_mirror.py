"""Seed-replication scorer: line-for-line mirror of frozen_eval.score_checkpoint
with StereoNetConfig(downsample_levels=3, num_disparities=24,
cost_volume_shift="right", regression_normalize=True).

Dataset, GT scale, metrics, and contract guard are imported UNMODIFIED from
phase1.harness.frozen_eval. Usage: python <run_dir>/score_mirror.py <run_dir>
(e.g. phase1/runs/arm_x). Writes frozen_eval_best.json / frozen_eval_final.json
into the run dir and prints the liveness control (shift=none at eval) EPE for best.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from phase1.harness.frozen_eval import (  # noqa: E402  (unmodified harness imports)
    pooled_metrics,
    record_provenance,
    refuse_unless_contract,
    sha256_file,
    strict_load_report,
    weight_sha,
)
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402
from src.models.stereonet import StereoNet, StereoNetConfig  # noqa: E402

MIRROR_NOTE = (
    "frozen_eval.score_checkpoint hardcodes StereoNetConfig() "
    "(downsample_levels=4, num_disparities=12, cost_volume_shift=none, "
    "regression_normalize=False), which would evaluate "
    "ARM X weights through a different forward graph than trained. Faithful scores "
    "use a line-for-line mirror of score_checkpoint with "
    "StereoNetConfig(downsample_levels=3, num_disparities=24, "
    "cost_volume_shift='right', regression_normalize=True); "
    "dataset split, GT scale, metrics, and contract guard are imported unmodified "
    "from phase1.harness.frozen_eval."
)


def score_ckpt(ckpt_rel: str, shift: str, regnorm: bool, device: str) -> dict:
    ckpt_path = REPO_ROOT / ckpt_rel
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    config = blob.get("config", {}) if isinstance(blob, dict) else {}
    model = StereoNet(StereoNetConfig(downsample_levels=3, num_disparities=24,
                                       cost_volume_shift=shift,
                                       regression_normalize=regnorm))
    compat = strict_load_report(model, state)
    model.eval().to(device)
    ds = Kitti2015Stereo(REPO_ROOT / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    preds, gts, names = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            out = model(torch.from_numpy(normalize(s.left)).to(device),
                        torch.from_numpy(normalize(s.right)).to(device))
            pred = out[0, 0].cpu().numpy().astype(np.float64)
            valid = s.disparity > 0
            preds.append(pred[valid])
            gts.append(s.disparity[valid].astype(np.float64))
            names.append(s.name)
    m = pooled_metrics(preds, gts)
    guard = refuse_unless_contract(len(ds), m["valid_pixels"], 256.0,
                                   "hailo_val", "disp_occ_0")
    return {"checkpoint": ckpt_rel, "sha256": sha256_file(ckpt_path),
            "weight_sha16": weight_sha(state), "config": config,
            "compat": compat, "device": device, "scenes": len(ds),
            "names": names, "metrics": m, "guard": guard,
            "provenance": record_provenance(),
            "eval_graph": {"cost_volume_shift": shift,
                           "regression_normalize": regnorm},
            "mirror_note": MIRROR_NOTE}


def main() -> None:
    run_dir = Path(sys.argv[1])
    ckpt_best = (run_dir / "arm_x_best.pth").as_posix()
    ckpt_final = (run_dir / "arm_x_final.pth").as_posix()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ckpt_rel, tag in ((ckpt_best, "best"), (ckpt_final, "final")):
        rec = score_ckpt(ckpt_rel, "right", True, device)
        (run_dir / f"frozen_eval_{tag}.json").write_text(json.dumps(rec, indent=2))
        print(f"{tag}: EPE {rec['metrics']['epe']:.7f} "
              f"D1 {rec['metrics']['d1']:.7f}% RMSE {rec['metrics']['rmse']:.7f} "
              f"px {rec['metrics']['valid_pixels']} "
              f"contract_match {rec['guard']['contract_match']}", flush=True)
    live = score_ckpt(ckpt_best, "none", True, device)
    print(f"LIVENESS shift=none @best: EPE {live['metrics']['epe']:.7f} "
          f"D1 {live['metrics']['d1']:.7f}%", flush=True)
    (run_dir / "liveness_shift_none.json").write_text(json.dumps(
        {"checkpoint": ckpt_best, "eval_graph": live["eval_graph"],
         "metrics": live["metrics"],
         "guard": live["guard"]}, indent=2))


if __name__ == "__main__":
    main()
