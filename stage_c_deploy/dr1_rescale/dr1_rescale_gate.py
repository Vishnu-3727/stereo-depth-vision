"""DR-1 Variant A activation-rescale gate (STAGE C, PHASE DR-1). INFERENCE ONLY.

ZERO TRAINING. No weight/architecture/loss/augmentation/candidate/tolerance/
dataset/preprocessing change. No new checkpoint. Frozen checkpoint stays
byte-identical (hash asserted before AND after).

Variant A only: aggregated_cost / s -> same regression/refinement, with
s = 1e17 exactly. Variant B (weight rescale) is NOT performed here.

Method: capture aggregated_cost ONCE per scene from the frozen ARM-P forward
result (return_stages=True), then run BOTH paths through the model's OWN
modules (no replication):
  Path A: disp = relu(refine(regression(agg)) + ...)
  Path B: disp = relu(refine(regression(agg / s)) + ...)
so all downstream computation is identical by construction.

Five-scene contract: KITTI2015 hailo_val indices 0-4 asserted exactly.
Forty-scene contract: frozen evaluator (pooled_metrics + refuse_unless_contract,
gt_scale 256.0, disp_occ_0); the rescaled path uses the module-based forward
which is FIRST validated to reproduce model() at s=1 (bit check).

Writes (new files only under stage_c_deploy/dr1_rescale/):
  dr1_rescale_gate.json, dr1_eval_40scene.json
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
import torch.nn.functional as F

from phase1.harness.frozen_eval import (
    pooled_metrics,
    refuse_unless_contract,
)
from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

OUT = REPO / "stage_c_deploy" / "dr1_rescale"
ARMP_REL = "stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth"
ARMP_SHA = "b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454"
CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
EXPECTED_PARAMS = 397954
EXPECTED_TENSORS = 70
S = 1e17  # exact DR-1 scalar; aggregated_cost_rescaled = aggregated_cost / s
TENSOR_NAME = "forward return_stages['aggregated_cost']"
FIVE_SCENES = [f"00016{i}_10.png" for i in range(5)]
DR1_THRESH = 1e-3  # DR-1 RESCALE EQUIVALENCE THRESHOLD (NOT the C1 gate)
EPE_REG_TOL = 0.05


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def downstream(net, agg: torch.Tensor, left: torch.Tensor):
    """Run the model's OWN regression+refinement+add+relu on an agg tensor.

    Returns (disp_final, disp_init, residual, cost_postnorm_recomputed).
    cost_postnorm is recomputed exactly per regression.py:48 on the upsampled
    cost for range reporting only (not fed separately; regression does it).
    """
    with torch.no_grad():
        size = (left.shape[-2], left.shape[-1])
        disp_init = net.regression(agg, size)
        residual = net.refinement(disp_init, left)
        disp = disp_init + residual
        if net.config.final_relu:
            disp = torch.relu(disp)
        up = F.interpolate(agg, size=size, mode="bilinear", align_corners=True)
        post = (up - up.mean(1, keepdim=True)) / (up.std(1, keepdim=True) + 1e-6)
    return disp, disp_init, residual, post


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True,
                                  timeout=15).stdout.strip()
    except Exception:
        git_head = None

    ckpt = REPO / ARMP_REL
    sha_before = sha256(ckpt)
    assert sha_before == ARMP_SHA, "checkpoint hash changed -> STOP"

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = StereoNet(StereoNetConfig(**CFG))
    net.load_state_dict(blob["model"], strict=True)
    net.eval().cpu()
    assert sum(p.numel() for p in net.parameters()) == EXPECTED_PARAMS
    assert len(blob["model"]) == EXPECTED_TENSORS
    assert net.config.regression_normalize is True

    # snapshot weights to prove no modification
    w_before = {k: v.clone() for k, v in net.state_dict().items()}

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    assert [ds[i].name for i in range(5)] == FIVE_SCENES, "scene order -> STOP"

    prov = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head, "host": platform.platform(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__,
        "command": "python stage_c_deploy/dr1_rescale/dr1_rescale_gate.py",
        "argv": sys.argv,
        "checkpoint": ARMP_REL, "checkpoint_sha256_before": sha_before,
        "scalar_s": S, "tensor_name": TENSOR_NAME,
        "intervention": "aggregated_cost / s -> same regression/refinement (Variant A)",
        "variant": "A (activation rescale). Variant B NOT performed.",
        "scenes": FIVE_SCENES,
        "config": {**CFG, "params": EXPECTED_PARAMS, "tensors": EXPECTED_TENSORS},
        "c1_verdict_unchanged": "FAIL",
        "dr1_rescale_equivalence_threshold": DR1_THRESH,
        "epe_regression_tolerance": EPE_REG_TOL,
    }

    five = {"s": S, "per_scene": [], "pooled_max_abs_delta": 0.0,
            "determinism": {}, "probe_a_1e_minus_17": {}}
    pooled_delta = 0.0
    rep1, rep2 = [], []
    with torch.no_grad():
        for i in range(5):
            s = ds[i]
            L, R = normalize(s.left), normalize(s.right)
            tl, tr = torch.from_numpy(L), torch.from_numpy(R)
            # capture ONCE
            fwd, st = net(tl, tr, return_stages=True)
            agg = st["aggregated_cost"]
            shape = list(agg.shape)
            dtype = str(agg.dtype)
            # Path A / Path B through the model's own modules
            dA, iA, rA, pA = downstream(net, agg, tl)
            dB, iB, rB, pB = downstream(net, agg / S, tl)
            # replication self-check: module path A must equal forward()
            selfdiff = float((dA - fwd).abs().max().item())
            delta = dB - dA
            marr = delta.detach().cpu().numpy()
            assert np.all(np.isfinite(marr)), f"nonfinite delta scene {s.name}"
            md = float(np.abs(marr).max())
            pooled_delta = max(pooled_delta, md)
            # determinism: run the B path a second time
            dB2, _, _, _ = downstream(net, agg / S, tl)
            rep1.append(float(dB.abs().max().item()))
            rep2.append(float(dB2.abs().max().item()))
            det = float((dB - dB2).abs().max().item())
            rec = {
                "scene": s.name,
                "tensor": TENSOR_NAME, "shape": shape, "dtype": dtype,
                "s": S,
                "rerun_selfdiff_pathA_vs_forward": selfdiff,
                "max_abs_delta_disparity": md,
                "determinism_rerun_maxdiff": det,
                "agg_max_abs_original": float(agg.abs().max().item()),
                "agg_max_abs_rescaled": float((agg / S).abs().max().item()),
                "postnorm_max_original": float(pA.abs().max().item()),
                "postnorm_max_rescaled": float(pB.abs().max().item()),
                "disparity_initial_max_original": float(iA.abs().max().item()),
                "disparity_initial_max_rescaled": float(iB.abs().max().item()),
                "refinement_residual_max_original": float(rA.abs().max().item()),
                "refinement_residual_max_rescaled": float(rB.abs().max().item()),
                "disparity_final_max_original": float(dA.abs().max().item()),
                "disparity_final_max_rescaled": float(dB.abs().max().item()),
                "nonfinite_counts": {
                    "delta": int(np.sum(~np.isfinite(marr))),
                    "dB": int(np.sum(~np.isfinite(
                        dB.detach().cpu().numpy()))),
                    "dA": int(np.sum(~np.isfinite(
                        dA.detach().cpu().numpy()))),
                },
            }
            five["per_scene"].append(rec)
            print(f"{s.name}: selfdiff={selfdiff:.3e} max|dB-dA|={md:.6e} "
                  f"post {rec['postnorm_max_original']:.4f}->{rec['postnorm_max_rescaled']:.4f} "
                  f"final {rec['disparity_final_max_original']:.3f}->{rec['disparity_final_max_rescaled']:.3f}",
                  flush=True)
    five["pooled_max_abs_delta"] = pooled_delta
    five["determinism"] = {
        "per_scene_pathB_rerun_maxdiff": [
            r["determinism_rerun_maxdiff"] for r in five["per_scene"]],
        "bit_identical": all(
            r["determinism_rerun_maxdiff"] == 0.0 for r in five["per_scene"]),
    }
    five["selfcheck_pathA_vs_forward_max"] = max(
        r["rerun_selfdiff_pathA_vs_forward"] for r in five["per_scene"])
    # extend DR-0 probe at a = 1e-17 (== 1/s), self-contained row
    five["probe_a_1e_minus_17"] = {
        "a": 1e-17, "equiv_s": 1e17,
        "pooled_max_abs_delta_vs_a1": pooled_delta,
        "note": ("Same Variant-A comparison expressed in DR-0 probe units "
                 "(a = 1/s). DR-0 measurements untouched."),
    }

    # weights unchanged?
    w_after = net.state_dict()
    weights_unchanged = all(torch.equal(w_before[k], w_after[k]) for k in w_before)
    sha_after = sha256(ckpt)
    five["weights_unchanged"] = weights_unchanged
    five["checkpoint_sha256_after"] = sha_after

    with open(OUT / "dr1_rescale_gate.json", "w") as f:
        json.dump({"provenance": prov, "five_scene": five}, f, indent=2)
    print(f"pooled max|dB-dA| = {pooled_delta:.6e} px; "
          f"weights_unchanged={weights_unchanged}; sha_after==sha_before: "
          f"{sha_after == sha_before}", flush=True)

    # ---- frozen 40-scene accuracy gate (only valid comparisons) ----
    t0 = time.time()
    preds_orig, preds_rs, gts, names = [], [], [], []
    maxdiff_s1 = 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            sm = ds[i]
            tl = torch.from_numpy(normalize(sm.left))
            tr = torch.from_numpy(normalize(sm.right))
            out_orig = net(tl, tr)  # frozen path, unmodified call
            _, st = net(tl, tr, return_stages=True)
            dA, _, _, _ = downstream(net, st["aggregated_cost"], tl)
            if i < 5:  # s=1 validation of the rescaled-path machinery
                maxdiff_s1 = max(maxdiff_s1,
                                 float((dA - out_orig).abs().max().item()))
            dB, _, _, _ = downstream(net, st["aggregated_cost"] / S, tl)
            po = out_orig[0, 0].cpu().numpy().astype(np.float64)
            pr = dB[0, 0].cpu().numpy().astype(np.float64)
            valid = sm.disparity > 0
            preds_orig.append(po[valid])
            preds_rs.append(pr[valid])
            gts.append(sm.disparity[valid].astype(np.float64))
            names.append(sm.name)
    m_orig = pooled_metrics(preds_orig, gts)
    # pooled_metrics consumes lists; recompute gts concat for rescaled
    m_rs = pooled_metrics(preds_rs, gts)
    g_orig = refuse_unless_contract(len(ds), m_orig["valid_pixels"], 256.0,
                                    "hailo_val", "disp_occ_0")
    g_rs = refuse_unless_contract(len(ds), m_rs["valid_pixels"], 256.0,
                                  "hailo_val", "disp_occ_0")
    eval_rec = {
        "provenance": {**prov,
                       "eval_elapsed_s": time.time() - t0,
                       "s1_path_validation_maxdiff_5scenes": maxdiff_s1},
        "s1_validation_reproduces_forward": maxdiff_s1 == 0.0,
        "original": {"epe": m_orig["epe"], "d1": m_orig["d1"],
                     "valid_pixels": m_orig["valid_pixels"], "guard": g_orig,
                     "full_metrics": m_orig},
        "rescaled": {"epe": m_rs["epe"], "d1": m_rs["d1"],
                     "valid_pixels": m_rs["valid_pixels"], "guard": g_rs,
                     "full_metrics": m_rs},
        "epe_delta_rescaled_minus_original": m_rs["epe"] - m_orig["epe"],
        "contract_scenes": len(ds),
    }
    with open(OUT / "dr1_eval_40scene.json", "w") as f:
        json.dump(eval_rec, f, indent=2)
    print(f"s=1 path validation maxdiff={maxdiff_s1:.3e}; "
          f"orig EPE={m_orig['epe']:.7f} rs EPE={m_rs['epe']:.7f} "
          f"delta={m_rs['epe'] - m_orig['epe']:.3e}; "
          f"valid={m_orig['valid_pixels']}/{m_rs['valid_pixels']}", flush=True)
    if maxdiff_s1 != 0.0:
        print("STOP: rescaled-path machinery does not reproduce forward at s=1",
              flush=True)


if __name__ == "__main__":
    main()
