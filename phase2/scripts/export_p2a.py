"""PHASE-2 DEPLOYMENT — export P2A seed 0 to ONNX and validate it.

Steps, in order:
  1. export the deployed checkpoint to ONNX at the deployment input size
  2. numerical parity: exported ONNX vs the trained PyTorch graph
  3. DEPLOYED ACCURACY: score the ONNX itself under the unchanged frozen contract
  4. runtime + resource measurement (PyTorch CUDA/CPU, ONNX Runtime CPU)
  5. operator-level compatibility check against the reference deployment ONNX

Nothing is retrained. The frozen contract is untouched: no evaluation-time
augmentation, no multi-scale inference, no crop change.
"""
from __future__ import annotations

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
import onnx
import onnxruntime as ort
import torch

from phase1.harness.frozen_eval import (pooled_metrics, refuse_unless_contract,
                                        sha256_file, weight_sha)
from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

CKPT = REPO / "phase2" / "runs" / "p2a_scale_coverage" / "p2a_best.pth"
OUT = REPO / "phase2" / "deploy"
ONNX_PATH = OUT / "p2a_stereonet.onnx"
H, W = 368, 1232
CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
REF_ONNX = REPO / "reference" / "onnx" / "stereonet.onnx"


def build():
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = StereoNet(StereoNetConfig(**CFG))
    m.load_state_dict(blob["model"], strict=True)
    return m.eval(), blob


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"artifact": "P2A seed 0", "source_checkpoint": str(CKPT.relative_to(REPO)).replace("\\", "/"),
           "source_sha256": sha256_file(CKPT),
           "utc": datetime.now(timezone.utc).isoformat()}
    try:
        rec["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                         capture_output=True, text=True,
                                         timeout=15).stdout.strip()
    except Exception:
        rec["git_head"] = None
    rec["software"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                       "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                       "numpy": np.__version__, "platform": platform.platform()}

    model, blob = build()
    rec["weight_sha16"] = weight_sha(blob["model"])
    rec["params"] = int(sum(p.numel() for p in model.parameters()))
    rec["config"] = CFG

    # ---- 1. export ----
    dummy_l = torch.zeros(1, 3, H, W)
    dummy_r = torch.zeros(1, 3, H, W)
    torch.onnx.export(model, (dummy_l, dummy_r), str(ONNX_PATH),
                      input_names=["left", "right"], output_names=["disparity"],
                      opset_version=13, do_constant_folding=True, dynamo=False)
    g = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(g)
    ops = sorted({n.op_type for n in g.graph.node})
    rec["export"] = {"path": str(ONNX_PATH.relative_to(REPO)).replace("\\", "/"),
                     "sha256": sha256_file(ONNX_PATH),
                     "size_bytes": ONNX_PATH.stat().st_size,
                     "opset": int(g.opset_import[0].version),
                     "input_shape": [1, 3, H, W], "n_nodes": len(g.graph.node),
                     "op_types": ops, "checker": "pass"}
    print("exported:", ONNX_PATH.name, len(g.graph.node), "nodes, ops:", ops, flush=True)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    inames = [i.name for i in sess.get_inputs()]
    oname = sess.get_outputs()[0].name

    # ---- 2. numerical parity on real data ----
    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    diffs = []
    for i in range(5):
        s = ds[i]
        L, R = normalize(s.left), normalize(s.right)
        with torch.no_grad():
            pt = model(torch.from_numpy(L), torch.from_numpy(R))[0, 0].numpy()
        on = sess.run([oname], {inames[0]: L, inames[1]: R})[0][0, 0]
        diffs.append(float(np.abs(pt.astype(np.float64) - on.astype(np.float64)).max()))
    rec["parity_vs_pytorch"] = {"scenes": 5, "max_abs_diff_px": max(diffs),
                               "per_scene_max_abs_diff_px": diffs,
                               "tolerance_px": 1e-3,
                               "pass": bool(max(diffs) < 1e-3)}
    print("parity max |diff| = %.3e px" % max(diffs), flush=True)

    # ---- 3. deployed accuracy: score the ONNX itself, frozen contract ----
    preds, gts = [], []
    pmax_all = -1e30
    for i in range(len(ds)):
        s = ds[i]
        out = sess.run([oname], {inames[0]: normalize(s.left),
                                 inames[1]: normalize(s.right)})[0]
        p = out[0, 0].astype(np.float64)
        pmax_all = max(pmax_all, float(p.max()))
        v = s.disparity > 0
        preds.append(p[v]); gts.append(s.disparity[v].astype(np.float64))
    P, G = np.concatenate(preds), np.concatenate(gts)
    m = pooled_metrics(preds, gts)
    err = np.abs(P - G)
    hi96 = G >= 96
    a96, _ = np.polyfit(G[hi96], P[hi96], 1)
    bins = []
    for lo, hi in zip([0, 16, 32, 48, 64, 80, 96, 112, 128, 144],
                      [16, 32, 48, 64, 80, 96, 112, 128, 144, 160]):
        mm = (G >= lo) & (G < hi)
        bins.append({"lo": lo, "hi": hi, "px": int(mm.sum()),
                     "epe": float(err[mm].mean()) if mm.sum() >= 1000 else None,
                     "signed_err": float((P[mm] - G[mm]).mean()) if mm.sum() >= 1000 else None})
    rec["deployed_accuracy_onnx"] = {
        "protocol": "frozen contract, unchanged; no eval-time augmentation; "
                    "no multi-scale inference",
        "scenes": len(ds), "metrics": m,
        "guard": refuse_unless_contract(len(ds), m["valid_pixels"], 256.0,
                                        "hailo_val", "disp_occ_0"),
        "gt_lt_64_epe": float(err[G < 64].mean()),
        "gt_ge_64_epe": float(err[G >= 64].mean()),
        "gt_ge_64_signed": float((P[G >= 64] - G[G >= 64]).mean()),
        "gt_ge_96_epe": float(err[hi96].mean()),
        "gt_ge_96_signed": float((P[hi96] - G[hi96]).mean()),
        "slope_hi": float(a96),
        "max_pred_valid_px": float(P.max()), "max_pred_all_px": pmax_all,
        "bins": bins}
    print("ONNX frozen-contract EPE %.7f  D1 %.4f%%  contract %s"
          % (m["epe"], m["d1"], rec["deployed_accuracy_onnx"]["guard"]["contract_match"]),
          flush=True)

    # ---- 4. runtime + resource ----
    runtime = {}
    L0, R0 = normalize(ds[0].left), normalize(ds[0].right)
    for _ in range(3):
        sess.run([oname], {inames[0]: L0, inames[1]: R0})
    t = []
    for _ in range(20):
        t0 = time.perf_counter()
        sess.run([oname], {inames[0]: L0, inames[1]: R0})
        t.append((time.perf_counter() - t0) * 1000)
    runtime["onnxruntime_cpu_ms"] = {"mean": float(np.mean(t)), "median": float(np.median(t)),
                                     "min": float(np.min(t)), "n": len(t)}
    tl, tr = torch.from_numpy(L0), torch.from_numpy(R0)
    with torch.no_grad():
        for _ in range(3):
            model(tl, tr)
        t = []
        for _ in range(20):
            t0 = time.perf_counter()
            model(tl, tr)
            t.append((time.perf_counter() - t0) * 1000)
    runtime["pytorch_cpu_ms"] = {"mean": float(np.mean(t)), "median": float(np.median(t)),
                                 "min": float(np.min(t)), "n": len(t)}
    if torch.cuda.is_available():
        mc = model.cuda()
        gl, gr = tl.cuda(), tr.cuda()
        with torch.no_grad():
            for _ in range(5):
                mc(gl, gr)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = []
            for _ in range(50):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                mc(gl, gr)
                torch.cuda.synchronize()
                t.append((time.perf_counter() - t0) * 1000)
        runtime["pytorch_cuda_ms"] = {"mean": float(np.mean(t)), "median": float(np.median(t)),
                                      "min": float(np.min(t)), "n": len(t),
                                      "gpu": torch.cuda.get_device_name(0)}
        runtime["cuda_peak_alloc_mb"] = float(torch.cuda.max_memory_allocated() / 2**20)
        model.cpu()
    runtime["params"] = rec["params"]
    runtime["onnx_size_bytes"] = rec["export"]["size_bytes"]
    runtime["checkpoint_size_bytes"] = CKPT.stat().st_size
    rec["runtime"] = runtime
    print("runtime:", json.dumps({k: (round(v["median"], 2) if isinstance(v, dict) and "median" in v else v)
                                  for k, v in runtime.items()}), flush=True)

    # ---- 5. operator compatibility vs the reference deployment graph ----
    ref = onnx.load(str(REF_ONNX))
    ref_ops = sorted({n.op_type for n in ref.graph.node})
    new_ops = set(ops)
    rec["deployment_compatibility"] = {
        "reference_onnx": "reference/onnx/stereonet.onnx",
        "reference_opset": int(ref.opset_import[0].version),
        "reference_op_types": ref_ops,
        "reference_n_nodes": len(ref.graph.node),
        "p2a_op_types": ops, "p2a_n_nodes": len(g.graph.node),
        "ops_not_in_reference": sorted(new_ops - set(ref_ops)),
        "ops_only_in_reference": sorted(set(ref_ops) - new_ops),
        "op_set_is_subset_of_reference": bool(new_ops <= set(ref_ops)),
        "hailo_toolchain_available": False,
        "hailo_compilation": "NOT VERIFIED — the Hailo SDK / dataflow compiler is not "
                             "installed in this environment, so no .hef was produced and "
                             "no on-device measurement was taken. The operator comparison "
                             "above is structural evidence only, not a compilation result.",
    }
    print("ops not in reference:", rec["deployment_compatibility"]["ops_not_in_reference"],
          flush=True)

    (OUT / "P2A_DEPLOYMENT_VALIDATION.json").write_text(json.dumps(rec, indent=2))
    print("wrote", OUT / "P2A_DEPLOYMENT_VALIDATION.json")


if __name__ == "__main__":
    main()
