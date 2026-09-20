"""PHASE-2 DEPLOYMENT — static-shape re-export of P2A seed 0.

Export-only mitigation for the seven dynamic-shape operators flagged in
PHASE2_DEPLOYMENT_VALIDATION.md section 5. No weight is touched, no training,
no change to the evaluation contract.

Two sources of the dynamic ops, both in the regression stage:

  1. soft_argmin builds its candidate index with
     torch.arange(cost.shape[1]) -> Range + Shape + Gather + Cast
  2. regression_normalize uses cost.std(1, keepdim=True), whose unbiased
     denominator N-1 is derived from the tensor shape
     -> ReduceProd + Shape + Gather

The deployment module replaces both with literals: a constant index buffer of
length 24, and the explicit unbiased form

     std = sqrt( sum_d (x - mean)^2 * (1 / (D - 1)) ),  D = 24

which is the identity torch.std computes, with the extent baked in. The
regression stage holds NO parameters, so swapping it cannot alter any weight.

Validation performed here:
  - op-level: confirm the dynamic operators are gone
  - numerical parity of the static PyTorch graph vs the trained graph
  - numerical parity of the static ONNX vs the original ONNX
  - frozen-contract accuracy of the static ONNX, scored end to end

Both artifacts are kept. The original export remains the reference artifact.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
print("repo:", REPO, flush=True)
sys.path.insert(0, str(REPO))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1.harness.frozen_eval import (pooled_metrics, refuse_unless_contract,
                                        sha256_file)
from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

CKPT = REPO / "stage_b_armp" / "20260919T012646Z_tier2_seed1" / "armp" / "p2a_best.pth"
OUT = REPO / "stage_c_deploy"
ORIG_ONNX = OUT / "armp_stereonet.onnx"
STATIC_ONNX = OUT / "armp_stereonet_static.onnx"
H, W, NDISP = 368, 1232, 24
CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
DYNAMIC_OPS = ["Range", "ReduceProd", "Shape", "Gather", "ConstantOfShape"]


class StaticDisparityRegression(nn.Module):
    """Deployment-only regression stage. Parameter-free, like the original.

    Numerically the same function as DisparityRegression(upsample_first=True,
    normalize=True); the only difference is that the candidate index grid and
    the unbiased-variance denominator are constants instead of shape-derived.
    """

    def __init__(self, num_disparities: int = NDISP, eps: float = 1e-6,
                 upsample_factor: int = 8) -> None:
        super().__init__()
        self.eps = eps
        self.num_disparities = num_disparities
        self.inv_dof = 1.0 / float(num_disparities - 1)      # unbiased: 1/(D-1)
        # The deployment input is fixed at 368x1232 and the feature stride is 8,
        # and 368 = 46*8, 1232 = 154*8 exactly, so a constant scale factor gives
        # bit-identical output geometry to the shape-derived `size` argument --
        # and emits Resize with constant scales instead of Shape -> Slice.
        self.upsample_factor = float(upsample_factor)
        self.register_buffer(
            "index",
            torch.arange(num_disparities, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False)

    def forward(self, cost: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        cost = F.interpolate(cost, scale_factor=self.upsample_factor,
                             mode="bilinear", align_corners=True,
                             recompute_scale_factor=False)
        mean = cost.mean(1, keepdim=True)
        centred = cost - mean
        var = (centred * centred).sum(1, keepdim=True) * self.inv_dof
        cost = centred / (var.sqrt() + self.eps)
        weights = torch.softmax(-cost, dim=1)
        return (weights * self.index).sum(1, keepdim=True)


def build(static: bool):
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = StereoNet(StereoNetConfig(**CFG))
    m.load_state_dict(blob["model"], strict=True)
    if static:
        before = {k: v.clone() for k, v in m.state_dict().items()}
        m.regression = StaticDisparityRegression()
        after = m.state_dict()
        assert sorted(after.keys()) == sorted(before.keys()), "state_dict changed"
        assert all(torch.equal(after[k], before[k]) for k in before), "weights changed"
    return m.eval()


def main() -> None:
    rec = {"artifact": "ARM-P seed 1, static-shape re-export",
           "source_checkpoint": "stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth",
           "source_sha256": sha256_file(CKPT),
           "utc": datetime.now(timezone.utc).isoformat(),
           "weights_touched": False, "training_performed": False}
    try:
        rec["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                         capture_output=True, text=True,
                                         timeout=15).stdout.strip()
    except Exception:
        rec["git_head"] = None
    rec["software"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                       "onnx": onnx.__version__, "onnxruntime": ort.__version__,
                       "numpy": np.__version__, "platform": platform.platform()}

    ref_model = build(static=False)
    st_model = build(static=True)
    rec["params"] = int(sum(p.numel() for p in st_model.parameters()))
    rec["params_unchanged"] = rec["params"] == 397954
    rec["state_dict_keys"] = len(st_model.state_dict())

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)

    # ---- parity of the static PyTorch graph against the trained graph ----
    d = []
    dm = []
    dr = []
    for i in range(5):
        s = ds[i]
        L, R = torch.from_numpy(normalize(s.left)), torch.from_numpy(normalize(s.right))
        with torch.no_grad():
            a = ref_model(L, R)[0, 0].numpy().astype(np.float64)
            b = st_model(L, R)[0, 0].numpy().astype(np.float64)
        ad = np.abs(a - b)
        d.append(float(ad.max()))
        dm.append(float(ad.mean()))
        m = a > 1.0
        dr.append(float((ad[m] / a[m]).max()) if m.any() else 0.0)
    rec["parity_static_vs_trained_pytorch"] = {
        "scenes": 5, "max_abs_diff_px": max(d), "per_scene": d,
        "mean_abs_diff_px": float(np.mean(dm)), "per_scene_mean_abs_diff_px": dm,
        "max_rel_diff": max(dr), "per_scene_max_rel_diff": dr,
        "rel_guard_px": 1.0,
        "tolerance_px": 1e-3, "pass": bool(max(d) < 1e-3)}
    print("static PyTorch vs trained PyTorch: max |diff| = %.3e px" % max(d), flush=True)

    # ---- export ----
    torch.onnx.export(st_model,
                      (torch.zeros(1, 3, H, W), torch.zeros(1, 3, H, W)),
                      str(STATIC_ONNX),
                      input_names=["left", "right"], output_names=["disparity"],
                      opset_version=13, do_constant_folding=True, dynamo=False)
    g = onnx.load(str(STATIC_ONNX))
    onnx.checker.check_model(g)
    ops = sorted({n.op_type for n in g.graph.node})
    orig = onnx.load(str(ORIG_ONNX))
    orig_ops = sorted({n.op_type for n in orig.graph.node})
    remaining = [o for o in DYNAMIC_OPS if o in ops]
    rec["export"] = {"path": "stage_c_deploy/armp_stereonet_static.onnx",
                     "sha256": sha256_file(STATIC_ONNX),
                     "size_bytes": STATIC_ONNX.stat().st_size,
                     "opset": int(g.opset_import[0].version),
                     "n_nodes": len(g.graph.node), "op_types": ops,
                     "checker": "pass"}
    rec["dynamic_op_removal"] = {
        "targeted": DYNAMIC_OPS,
        "present_in_original": [o for o in DYNAMIC_OPS if o in orig_ops],
        "present_in_static": remaining,
        "all_removed": not remaining,
        "original_n_nodes": len(orig.graph.node),
        "static_n_nodes": len(g.graph.node),
        "ops_dropped": sorted(set(orig_ops) - set(ops)),
        "ops_added": sorted(set(ops) - set(orig_ops))}
    print("static export: %d nodes (was %d); dynamic ops remaining: %s"
          % (len(g.graph.node), len(orig.graph.node), remaining or "none"), flush=True)

    sess = ort.InferenceSession(str(STATIC_ONNX), providers=["CPUExecutionProvider"])
    osess = ort.InferenceSession(str(ORIG_ONNX), providers=["CPUExecutionProvider"])
    inn = [i.name for i in sess.get_inputs()]
    on = sess.get_outputs()[0].name
    oinn = [i.name for i in osess.get_inputs()]
    oon = osess.get_outputs()[0].name

    # ---- parity: static ONNX vs original ONNX, and vs trained PyTorch ----
    dv, dp = [], []
    dvm, dpm = [], []
    dvr, dpr = [], []
    for i in range(5):
        s = ds[i]
        L, R = normalize(s.left), normalize(s.right)
        a = sess.run([on], {inn[0]: L, inn[1]: R})[0][0, 0].astype(np.float64)
        b = osess.run([oon], {oinn[0]: L, oinn[1]: R})[0][0, 0].astype(np.float64)
        with torch.no_grad():
            c = ref_model(torch.from_numpy(L), torch.from_numpy(R))[0, 0].numpy().astype(np.float64)
        adv = np.abs(a - b)
        adp = np.abs(a - c)
        dv.append(float(adv.max()))
        dp.append(float(adp.max()))
        dvm.append(float(adv.mean()))
        dpm.append(float(adp.mean()))
        mv = b > 1.0
        mp = c > 1.0
        dvr.append(float((adv[mv] / b[mv]).max()) if mv.any() else 0.0)
        dpr.append(float((adp[mp] / c[mp]).max()) if mp.any() else 0.0)
    rec["parity_static_onnx_vs_original_onnx"] = {
        "scenes": 5, "max_abs_diff_px": max(dv), "per_scene": dv,
        "mean_abs_diff_px": float(np.mean(dvm)), "per_scene_mean_abs_diff_px": dvm,
        "max_rel_diff": max(dvr), "per_scene_max_rel_diff": dvr,
        "rel_guard_px": 1.0, "rel_reference": "original ONNX output",
        "tolerance_px": 1e-3, "pass": bool(max(dv) < 1e-3)}
    rec["parity_static_onnx_vs_trained_pytorch"] = {
        "scenes": 5, "max_abs_diff_px": max(dp), "per_scene": dp,
        "mean_abs_diff_px": float(np.mean(dpm)), "per_scene_mean_abs_diff_px": dpm,
        "max_rel_diff": max(dpr), "per_scene_max_rel_diff": dpr,
        "rel_guard_px": 1.0, "rel_reference": "trained PyTorch output",
        "tolerance_px": 1e-3, "pass": bool(max(dp) < 1e-3)}
    print("static ONNX vs original ONNX : max |diff| = %.3e px" % max(dv), flush=True)
    print("static ONNX vs trained torch : max |diff| = %.3e px" % max(dp), flush=True)

    # ---- deployed accuracy of the static ONNX, frozen contract ----
    preds, gts = [], []
    pmax_all = -1e30
    for i in range(len(ds)):
        s = ds[i]
        out = sess.run([on], {inn[0]: normalize(s.left), inn[1]: normalize(s.right)})[0]
        p = out[0, 0].astype(np.float64)
        pmax_all = max(pmax_all, float(p.max()))
        v = s.disparity > 0
        preds.append(p[v]); gts.append(s.disparity[v].astype(np.float64))
    P, G = np.concatenate(preds), np.concatenate(gts)
    m = pooled_metrics(preds, gts)
    err = np.abs(P - G)
    hi = G >= 96
    a96, _ = np.polyfit(G[hi], P[hi], 1)
    rec["deployed_accuracy_static_onnx"] = {
        "protocol": "frozen contract, unchanged; no eval-time augmentation; "
                    "no multi-scale inference",
        "scenes": len(ds), "metrics": m,
        "guard": refuse_unless_contract(len(ds), m["valid_pixels"], 256.0,
                                        "hailo_val", "disp_occ_0"),
        "gt_lt_64_epe": float(err[G < 64].mean()),
        "gt_ge_64_epe": float(err[G >= 64].mean()),
        "gt_ge_96_epe": float(err[hi].mean()),
        "gt_ge_96_signed": float((P[hi] - G[hi]).mean()),
        "slope_hi": float(a96),
        "max_pred_valid_px": float(P.max()), "max_pred_all_px": pmax_all}
    print("static ONNX frozen-contract EPE %.7f  D1 %.4f%%  contract %s"
          % (m["epe"], m["d1"],
             rec["deployed_accuracy_static_onnx"]["guard"]["contract_match"]), flush=True)

    (OUT / "ARMP_STATIC_EXPORT_VALIDATION.json").write_text(json.dumps(rec, indent=2))
    print("wrote", OUT / "ARMP_STATIC_EXPORT_VALIDATION.json")


if __name__ == "__main__":
    main()
