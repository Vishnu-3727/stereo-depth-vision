"""DR-0 activation dynamic-range audit (STAGE C, PHASE DR-0). DIAGNOSIS ONLY.

ZERO TRAINING. No fine-tuning, no optimizer, no backward pass. No change to any
model, architecture, loss, augmentation, candidate count, normalization setting
or checkpoint. Inference-only compute, CPU, torch.no_grad(). No existing repo
file is modified; this script and its JSON outputs are new files only.

Reuses the structure of stage_c_deploy/diagnostics/measure_activation_range.py:
dataset loading, checkpoint loading, scene assertion, ONNX instrumentation
helpers and provenance block.

Covers brief TASKS 1, 2, 3, 4, 5, 6, 8. TASK 7 (normalization probe) lives in
dr0_normalization_probe.py. TASK 9 is the markdown report (written separately).

Stages (forward(..., return_stages=True)) plus the two extra regression stages:
  left_features, right_features, cost_volume, aggregated_cost,
  cost_upsampled_prenorm, cost_postnorm,
  disparity_initial, refinement_residual, disparity_final
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import helper
from scipy.stats import spearmanr

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

OUT = REPO / "stage_c_deploy" / "dr0_audit"
ARMP_ONNX = REPO / "stage_c_deploy" / "armp_stereonet.onnx"
REF_ONNX = REPO / "reference" / "onnx" / "stereonet.onnx"
REF_DBG_EXISTING = REPO / "stage_c_deploy" / "diagnostics" / "reference_instrumented_debug.onnx"
ARMP_DBG = OUT / "armp_instrumented_debug.onnx"
TOP_PIXELS = REPO / "stage_c_deploy" / "diagnostics" / "top_pixels_armp.csv"

CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
CKPTS = {
    "armp": ("stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth",
             "b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454"),
    "p2a": ("phase2/runs/p2a_scale_coverage/p2a_best.pth",
            "0868ffd137a9985306bf5563685fd2362799bdd630d313e1181c980a7fbb6033"),
}
STAGES9 = ["left_features", "right_features", "cost_volume",
           "aggregated_cost", "cost_upsampled_prenorm", "cost_postnorm",
           "disparity_initial", "refinement_residual", "disparity_final"]
CHAIN = STAGES9  # the growth chain order
THRESHOLDS = [1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16, 1e18]

REF_ANCHORS = {
    "left_features": None,  # resolved by backward trace (see below)
    "right_features": None,
    "cost_volume": "/Transpose_output_0",
    "aggregated_cost": "/Squeeze_output_0",
    "disparity_initial": "/ReduceSum_output_0",
    "refinement_residual": "/refine/refine.7/Conv_output_0",
    "disparity_final": "524",
}
REF_BRANCH_TENSORS = {"/res/res.6/Conv_output_0", "/res/res.6_1/Conv_output_0"}

# C1 parity tolerance, restated only (never redefined, never changed).
C1_TOL = 1e-3


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def full_stats(absflat64: np.ndarray, signed: np.ndarray,
               name: str, shape: list, dtype: str) -> dict:
    a = absflat64
    return {
        "tensor": name, "shape": shape, "dtype": dtype, "n": int(a.size),
        "min_abs": float(a.min()), "max_abs": float(a.max()),
        "mean_abs": float(a.mean()), "abs_mean": float(a.mean()),
        "mean_signed": float(signed.mean()),
        "std_abs": float(a.std()), "std_signed": float(signed.std()),
        "median_abs": float(np.median(a)),
        "p90_abs": float(np.percentile(a, 90.0)),
        "p95_abs": float(np.percentile(a, 95.0)),
        "p99_abs": float(np.percentile(a, 99.0)),
        "p999_abs": float(np.percentile(a, 99.9)),
        "p9999_abs": float(np.percentile(a, 99.99)),
        "frac_above": {str(t): float((a > t).mean()) for t in THRESHOLDS},
        "nonfinite": int(np.sum(~np.isfinite(a))),
    }


def trace_to_input(model: onnx.ModelProto, tensor: str) -> str | None:
    """Backward BFS from `tensor` through the graph to a graph input name."""
    prod: dict[str, onnx.NodeProto] = {}
    for n in model.graph.node:
        for o in n.output:
            prod[o] = n
    seen, stack = set(), [tensor]
    while stack:
        t = stack.pop()
        if t in ("input.1", "input.83"):
            return t
        if t in seen:
            continue
        seen.add(t)
        n = prod.get(t)
        if n is not None:
            stack.extend(n.input)
    return None


def capture_nine(net, tl: torch.Tensor, tr: torch.Tensor) -> dict[str, torch.Tensor]:
    """return_stages=True plus the two regression-internal tensors.

    cost_upsampled_prenorm: aggregated_cost AFTER bilinear upsample
    (align_corners=True), BEFORE standardization.
    cost_postnorm: AFTER standardization across dim=1 with eps 1e-6, i.e. the
    tensor actually fed to softmax(-cost) inside soft_argmin with
    normalize=True (src/models/stereonet/regression.py:48,66-67).
    """
    with torch.no_grad():
        _, st = net(tl, tr, return_stages=True)
        agg = st["aggregated_cost"]
        size = (tl.shape[-2], tl.shape[-1])
        up = F.interpolate(agg, size=size, mode="bilinear", align_corners=True)
        post = (up - up.mean(1, keepdim=True)) / (up.std(1, keepdim=True) + 1e-6)
    out = dict(st)
    out["cost_upsampled_prenorm"] = up
    out["cost_postnorm"] = post
    return out


def concentration_of(tensors: list[np.ndarray], kind: str) -> dict:
    """Per-channel (C=32, features) or per-disparity-slice (D=24, costs) stats.

    tensors: list of per-scene float32 arrays. kind 'channel' aggregates over
    everything but axis=1; kind 'slice' aggregates over everything but the
    disparity axis (axis=2 of (B,C,D,H,W), axis=1 of (B,D,H,W)).
    """
    info = {}
    per_unit_max, per_unit_std, per_unit_mass = [], [], []
    for arr in tensors:
        if kind == "channel":
            n_units = arr.shape[1]
            r = arr.reshape(arr.shape[0], n_units, -1)
        elif kind == "slice":
            d_axis = 2 if arr.ndim == 5 else 1
            n_units = arr.shape[d_axis]
            r = np.moveaxis(arr, d_axis, 1).reshape(arr.shape[0], n_units, -1)
        else:
            raise ValueError(kind)
        a = np.abs(r.astype(np.float64))
        per_unit_max.append(a.max(axis=(0, 2)))
        per_unit_std.append(r.astype(np.float64).std(axis=(0, 2)))
        per_unit_mass.append(a.sum(axis=(0, 2)))
    mx = np.max(per_unit_max, axis=0)
    mass = np.sum(per_unit_mass, axis=0)
    order = np.argsort(-mass)
    cum = np.cumsum(mass[order]) / mass.sum()
    n90 = int(np.searchsorted(cum, 0.9) + 1)
    med = float(np.median(mx))
    top1 = float(mx.max())
    info = {"n_units": int(len(mx)),
            "per_unit_max_abs": [float(v) for v in mx],
            "per_unit_std": [float(v) for v in np.max(per_unit_std, axis=0)],
            "top1_to_median_max_ratio": float(top1 / med) if med > 0 else None,
            "units_for_90pct_mass": n90,
            "verdict": "CONCENTRATED" if (n90 <= len(mx) // 3) else "GLOBAL"}
    return info


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    np.random.seed(0)

    try:
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True,
                                  timeout=15).stdout.strip()
    except Exception:
        git_head = None

    # provenance: versions of everything read
    import onnx as _onnx_mod
    prov = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head, "host": platform.platform(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "onnx": _onnx_mod.__version__, "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "command": "python stage_c_deploy/dr0_audit/dr0_activation_audit.py",
        "argv": sys.argv,
        "c1_tolerance_restated_unchanged": C1_TOL,
        "c1_verdict_unchanged": "FAIL",
    }
    for mname, (rel, exp) in CKPTS.items():
        got = sha256(REPO / rel)
        assert got == exp, f"{mname} checkpoint hash changed -> STOP"
        prov[mname + "_ckpt_sha256"] = got
    prov["armp_onnx_sha256"] = sha256(ARMP_ONNX)
    prov["reference_onnx_sha256"] = sha256(REF_ONNX)
    if REF_DBG_EXISTING.exists():
        prov["reference_debug_copy_sha256"] = sha256(REF_DBG_EXISTING)
    prov["note"] = ("No existing repo artifact was modified. New files only "
                    "under stage_c_deploy/dr0_audit/. Both .onnx files and "
                    "both .pth files read but never written.")

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    assert [ds[i].name for i in range(5)] == \
        [f"00016{i}_10.png" for i in range(5)], "scene order changed -> STOP"
    scenes = [ds[i].name for i in range(5)]

    out: dict = {"provenance": prov, "scenes": scenes,
                 "config": CFG, "models": {}}
    # tensor-name map for the record
    out["tensor_names"] = {
        "left_features": "forward return_stages['left_features']",
        "right_features": "forward return_stages['right_features']",
        "cost_volume": "forward return_stages['cost_volume']",
        "aggregated_cost": "forward return_stages['aggregated_cost']",
        "cost_upsampled_prenorm": "F.interpolate(aggregated_cost, bilinear, align_corners=True)",
        "cost_postnorm": "(up-mean_d)/(std_d+1e-6), i.e. softmax input in soft_argmin",
        "disparity_initial": "forward return_stages['disparity_initial']",
        "refinement_residual": "forward return_stages['refinement_residual']",
        "disparity_final": "forward return_stages['disparity_final']",
    }

    nets = {}
    for mname, (rel, _) in CKPTS.items():
        blob = torch.load(REPO / rel, map_location="cpu", weights_only=False)
        net = StereoNet(StereoNetConfig(**CFG))
        net.load_state_dict(blob["model"], strict=True)
        net.eval().cpu()
        nets[mname] = net

    # ---- TASKS 1,2,4,6a,6c,6d: per-model full stats ----
    for mname, net in nets.items():
        mrec: dict = {"backend": "pytorch-cpu-eval-no_grad",
                      "stages": {}, "concentration": {},
                      "per_layer": None, "reproducibility": {}}
        pool: dict[str, list] = {s: [] for s in STAGES9}
        pool_signed: dict[str, list] = {s: [] for s in STAGES9}
        conc_raw: dict[str, list] = {"left_features": [], "right_features": [],
                                     "cost_volume": [], "aggregated_cost": []}
        rep_max_run1: dict[str, list] = {s: [] for s in STAGES9}
        rep_max_run2: dict[str, list] = {s: [] for s in STAGES9}
        with torch.no_grad():
            for i in range(5):
                s = ds[i]
                L, R = normalize(s.left), normalize(s.right)
                tl, tr = torch.from_numpy(L), torch.from_numpy(R)
                for rep, store in ((1, rep_max_run1), (2, rep_max_run2)):
                    cap = capture_nine(net, tl, tr)
                    for stage in STAGES9:
                        t = cap[stage]
                        if rep == 1 and i == 0:
                            mrec["stages"].setdefault(
                                stage, {"per_scene": [], "pooled": None})
                        store[stage].append(
                            float(t.detach().abs().max().item()))
                        if rep == 1:
                            f32 = t.detach().cpu().numpy()
                            pool[stage].append(
                                np.abs(f32.astype(np.float64)).ravel())
                            pool_signed[stage].append(
                                f32.astype(np.float64).ravel())
                            if stage in conc_raw:
                                conc_raw[stage].append(f32)
                            if i == 0:
                                pass
                # per-scene stats from run 1 only
                cap = capture_nine(net, tl, tr)
                for stage in STAGES9:
                    f32 = cap[stage].detach().cpu().numpy()
                    a = np.abs(f32.astype(np.float64)).ravel()
                    sgn = f32.astype(np.float64).ravel()
                    mrec["stages"][stage]["per_scene"].append(
                        {"scene": s.name,
                         **full_stats(a, sgn, stage, list(f32.shape),
                                      str(f32.dtype))})
                del cap
        for stage in STAGES9:
            a = np.concatenate(pool[stage])
            g = np.concatenate(pool_signed[stage])
            shape = list(conc_raw[stage][0].shape) if stage in conc_raw else None
            if shape is None:
                # recover from per_scene[0]
                shape = mrec["stages"][stage]["per_scene"][0]["shape"]
            mrec["stages"][stage]["pooled"] = full_stats(
                a, g, stage, shape,
                mrec["stages"][stage]["per_scene"][0]["dtype"])
            del pool[stage], pool_signed[stage]
        # Task 2 concentration (pooled over 5 scenes)
        mrec["concentration"]["left_features"] = concentration_of(
            conc_raw["left_features"], "channel")
        mrec["concentration"]["right_features"] = concentration_of(
            conc_raw["right_features"], "channel")
        mrec["concentration"]["cost_volume"] = concentration_of(
            conc_raw["cost_volume"], "slice")
        mrec["concentration"]["cost_volume"]["note"] = (
            "5-D (B,C,D,H,W): per-disparity-slice over B,C,H,W; "
            "per-channel-over-B,D,H,W not computed (channel axis entangled "
            "with subtract construction); slice verdict only.")
        mrec["concentration"]["aggregated_cost"] = concentration_of(
            conc_raw["aggregated_cost"], "slice")
        for k in list(conc_raw):
            del conc_raw[k]
        # Task 6a determinism
        selfdiff = {s: max(abs(a - b) for a, b in
                            zip(rep_max_run1[s], rep_max_run2[s]))
                    for s in STAGES9}
        mrec["reproducibility"]["determinism_two_runs_max_self_diff"] = selfdiff
        mrec["reproducibility"]["determinism_bit_identical"] = all(
            v == 0.0 for v in selfdiff.values())
        # Task 6d float64 hand recompute on pooled-max tensor (aggregated_cost)
        aggmax_scene = max(range(5), key=lambda i: mrec["stages"][
            "aggregated_cost"]["per_scene"][i]["max_abs"])
        s = ds[aggmax_scene]
        L, R = normalize(s.left), normalize(s.right)
        with torch.no_grad():
            cap = capture_nine(net, torch.from_numpy(L), torch.from_numpy(R))
            t = cap["aggregated_cost"].detach().cpu().numpy().astype(np.float64)
        hand_max = float(np.abs(t, dtype=np.float64).max())
        mrec["reproducibility"]["float64_hand_recompute"] = {
            "stage": "aggregated_cost", "scene": s.name,
            "pipeline_max_abs": mrec["stages"]["aggregated_cost"][
                "per_scene"][aggmax_scene]["max_abs"],
            "hand_max_abs_float64": hand_max,
        }
        out["models"][mname] = mrec
        p = mrec["stages"]
        print(f"{mname}: pooled max|v|: " + ", ".join(
            f"{s}={p[s]['pooled']['max_abs']:.4e}" for s in STAGES9),
            flush=True)

    # ---- TASK 3: growth chain + per-layer hooks in winner ----
    for mname in nets:
        p = out["models"][mname]["stages"]
        growth = []
        for a, b in zip(CHAIN[:-1], CHAIN[1:]):
            ma, mb = p[a]["pooled"]["max_abs"], p[b]["pooled"]["max_abs"]
            pa, pb = p[a]["pooled"]["p999_abs"], p[b]["pooled"]["p999_abs"]
            growth.append({"from": a, "to": b,
                           "growth_max": mb / ma if ma else None,
                           "growth_p999": pb / pa if pa else None})
        out["models"][mname]["growth"] = growth
        win = max(growth, key=lambda g: g["growth_max"])
        out["models"][mname]["largest_growth_transition"] = win
        print(f"{mname}: largest growth {win['from']}->{win['to']} "
              f"x{win['growth_max']:.3e} (p999 x{win['growth_p999']:.3e})",
              flush=True)

    # per-layer hooks ONLY inside the winner output-stage submodule (ARM-P
    # decides; P2A recorded for the same submodule for the side-by-side).
    win_to = out["models"]["armp"]["largest_growth_transition"]["to"]
    net = nets["armp"]
    layer_max: dict[str, list] = {}
    if win_to == "aggregated_cost":
        targets = {f"aggregation.filter.{i}": net.aggregation.filter[i]
                   for i in range(len(net.aggregation.filter))}
        targets["aggregation.to_cost"] = net.aggregation.to_cost
        submod = "aggregation (filter convs/activations + to_cost)"
    elif win_to in ("left_features", "right_features", "cost_volume"):
        targets = {}
        for i, c in enumerate(net.feature_extractor.downsample):
            targets[f"feature_extractor.downsample.{i}"] = c
        for i, b in enumerate(net.feature_extractor.residual):
            targets[f"feature_extractor.residual.{i}"] = b
        targets["feature_extractor.output_conv"] = net.feature_extractor.output_conv
        submod = ("feature_extractor (producer of the winning stage; "
                  "cost_volume itself is parameter-free subtraction)")
    else:
        targets = {}
        submod = "no submodule (post-readout stage); per-layer UNKNOWN"
    if targets:
        handles = []
        store: dict[str, torch.Tensor] = {}

        def _mk(name):
            def _h(mod, inp, o):
                store[name] = o.detach()
            return _h

        for name, mod in targets.items():
            handles.append(mod.register_forward_hook(_mk(name)))
        with torch.no_grad():
            for i in range(5):
                s = ds[i]
                L, R = normalize(s.left), normalize(s.right)
                net(torch.from_numpy(L), torch.from_numpy(R))
                for name, t in store.items():
                    layer_max.setdefault(name, []).append(
                        float(t.abs().max().item()))
        for h in handles:
            h.remove()
        # order layers by first-scene max for readability
        order = sorted(layer_max, key=lambda k: layer_max[k][0])
        out["models"]["armp"]["per_layer"] = {
            "winner_output_stage": win_to, "submodule": submod,
            "per_scene_max_abs": {k: layer_max[k] for k in order},
            "scenes": scenes,
        }
        # same submodule on P2A for the descriptive side-by-side
        pnet = nets["p2a"]
        pt: dict[str, torch.Tensor] = {}
        if win_to == "aggregated_cost":
            ptargets = {f"aggregation.filter.{i}": pnet.aggregation.filter[i]
                        for i in range(len(pnet.aggregation.filter))}
            ptargets["aggregation.to_cost"] = pnet.aggregation.to_cost
        else:
            ptargets = {}
            for i, c in enumerate(pnet.feature_extractor.downsample):
                ptargets[f"feature_extractor.downsample.{i}"] = c
            for i, b in enumerate(pnet.feature_extractor.residual):
                ptargets[f"feature_extractor.residual.{i}"] = b
            ptargets["feature_extractor.output_conv"] = \
                pnet.feature_extractor.output_conv
        pt, ph, player_max = {}, [], {}
        for name, mod in ptargets.items():
            def _mk2(mod, inp, o, _name=name):
                pt[_name] = o.detach()
            ph.append(mod.register_forward_hook(_mk2))
        with torch.no_grad():
            for i in range(5):
                s = ds[i]
                L, R = normalize(s.left), normalize(s.right)
                pnet(torch.from_numpy(L), torch.from_numpy(R))
                for name, t in pt.items():
                    player_max.setdefault(name, []).append(
                        float(t.abs().max().item()))
        for h in ph:
            h.remove()
        out["models"]["p2a"]["per_layer"] = {
            "winner_output_stage": win_to,
            "submodule": submod + " (same submodule, descriptive side-by-side)",
            "per_scene_max_abs": {k: player_max[k] for k in
                                  sorted(player_max,
                                         key=lambda k: player_max[k][0])},
            "scenes": scenes,
        }
    else:
        out["models"]["armp"]["per_layer"] = {"winner_output_stage": win_to,
                                              "submodule": submod,
                                              "status": "UNKNOWN"}
        out["models"]["p2a"]["per_layer"] = {"winner_output_stage": win_to,
                                             "submodule": submod,
                                             "status": "UNKNOWN"}

    # ---- TASK 6b: second-path reproduction of aggregated_cost max via ONNX --
    model = onnx.load(str(ARMP_ONNX))
    existing = {o.name for o in model.graph.output}
    anchor = "/aggregation/Squeeze_output_0"
    assert any(anchor in (list(n.output)) for n in model.graph.node), \
        "aggregation anchor missing from armp onnx -> STOP"
    if anchor not in existing:
        model.graph.output.append(helper.make_tensor_value_info(
            anchor, onnx.TensorProto.FLOAT, None))
    onnx.save(model, str(ARMP_DBG))
    prov["armp_debug_copy"] = str(ARMP_DBG.name)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(ARMP_DBG), sess_options=so,
                                providers=["CPUExecutionProvider"])
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]
    pmax = [out["models"]["armp"]["stages"]["aggregated_cost"][
        "per_scene"][i]["max_abs"] for i in range(5)]
    second, reldiff = [], []
    for i in range(5):
        s = ds[i]
        L, R = normalize(s.left), normalize(s.right)
        outs = sess.run(onames, {inames[0]: L, inames[1]: R})
        omap = dict(zip(onames, outs))
        m = float(np.abs(np.asarray(omap[anchor],
                                    dtype=np.float64)).max())
        second.append(m)
        reldiff.append(abs(m - pmax[i]) / pmax[i] if pmax[i] else None)
    out["models"]["armp"]["reproducibility"]["second_path_onnx"] = {
        "debug_copy": str(ARMP_DBG.name),
        "anchor": anchor,
        "source_onnx": "stage_c_deploy/armp_stereonet.onnx (read-only)",
        "pytorch_max_abs_per_scene": pmax,
        "onnx_max_abs_per_scene": second,
        "relative_difference_per_scene": reldiff,
    }
    print("armp second-path rel diff per scene:", reldiff, flush=True)

    # ---- TASK 5: Hailo reference column (extended table where located) ----
    if REF_DBG_EXISTING.exists():
        ref_dbg_path = REF_DBG_EXISTING
        prov["reference_debug_copy_reused"] = \
            "stage_c_deploy/diagnostics/reference_instrumented_debug.onnx (read-only)"
    else:
        model = onnx.load(str(REF_ONNX))
        branch_src = {t: trace_to_input(model, t) for t in REF_BRANCH_TENSORS}
        anchors = dict(REF_ANCHORS)
        sess0 = ort.InferenceSession(str(REF_ONNX),
                                     providers=["CPUExecutionProvider"])
        left_src = sess0.get_inputs()[0].name
        for t, src in branch_src.items():
            if src == left_src:
                anchors["left_features"] = t
            else:
                anchors["right_features"] = t
        existing = {o.name for o in model.graph.output}
        for stage, nm in anchors.items():
            if nm is not None and nm not in existing:
                model.graph.output.append(helper.make_tensor_value_info(
                    nm, onnx.TensorProto.FLOAT, None))
                existing.add(nm)
        ref_dbg_path = OUT / "reference_instrumented_debug_copy.onnx"
        onnx.save(model, str(ref_dbg_path))
    model = onnx.load(str(ref_dbg_path))
    branch_src = {t: trace_to_input(model, t) for t in REF_BRANCH_TENSORS}
    anchors = dict(REF_ANCHORS)
    sess0 = ort.InferenceSession(str(REF_ONNX),
                                 providers=["CPUExecutionProvider"])
    left_src = sess0.get_inputs()[0].name
    for t, src in branch_src.items():
        if src == left_src:
            anchors["left_features"] = t
        else:
            anchors["right_features"] = t
    located = {s: nm for s, nm in anchors.items() if nm is not None}
    rrec: dict = {
        "backend": "onnxruntime-CPU",
        "input_note": ("fed the same ImageNet-normalised input the host-side "
                       "reproduction uses (normalisation is compiler-inserted, "
                       "NOT in the ONNX); input names input.1/input.83, left "
                       "image as first input."),
        "caveat": ("different downsample (x16 vs x8), 12 vs 24 candidates, "
                   "the reference cost volume is degenerate (EXP-010), "
                   "reference measured through onnxruntime while ours is "
                   "PyTorch. The reference is a DEPLOYMENT-RANGE reference, "
                   "not a layer-by-layer architectural equivalence."),
        "anchors": {s: anchors.get(s) for s in
                    ["left_features", "right_features", "cost_volume",
                     "aggregated_cost", "disparity_initial",
                     "refinement_residual", "disparity_final"]},
        "branch_trace": branch_src,
        "stages": {},
    }
    sess = ort.InferenceSession(str(ref_dbg_path), sess_options=so,
                                providers=["CPUExecutionProvider"])
    inames = [i.name for i in sess.get_inputs()]
    assert inames[0] == "input.1" and inames[1] == "input.83", \
        f"reference input names changed: {inames} -> STOP"
    onames = [o.name for o in sess.get_outputs()]
    rpool: dict[str, list] = {}
    rpool_s: dict[str, list] = {}
    for i in range(5):
        s = ds[i]
        L, R = normalize(s.left), normalize(s.right)
        outs = sess.run(onames, {inames[0]: L, inames[1]: R})
        omap = dict(zip(onames, outs))
        for stage in ["left_features", "right_features", "cost_volume",
                      "aggregated_cost", "disparity_initial",
                      "refinement_residual", "disparity_final"]:
            nm = located.get(stage)
            if nm is None or nm not in omap or omap[nm] is None:
                if i == 0:
                    rrec["stages"][stage] = {"status": "UNKNOWN",
                                             "reason": "anchor not located; "
                                                       "no guessing"}
                continue
            f32 = np.asarray(omap[nm])
            a = np.abs(f32.astype(np.float64)).ravel()
            rpool.setdefault(stage, []).append(a)
            rpool_s.setdefault(stage, []).append(f32.astype(np.float64).ravel())
            rrec["stages"].setdefault(
                stage, {"onnx_tensor": nm, "per_scene": [], "pooled": None,
                        "status": "located"})
            rrec["stages"][stage]["per_scene"].append(
                {"scene": s.name,
                 **full_stats(a, f32.astype(np.float64).ravel(), nm,
                              list(f32.shape), str(f32.dtype))})
    for stage, arrs in rpool.items():
        nm = located[stage]
        f0 = np.asarray(sess.run(
            onames, {inames[0]: normalize(ds[0].left),
                     inames[1]: normalize(ds[0].right)})[onames.index(nm)])
        rrec["stages"][stage]["pooled"] = full_stats(
            np.concatenate(arrs), np.concatenate(rpool_s[stage]), nm,
            list(f0.shape), str(f0.dtype))
    out["models"]["reference"] = rrec
    print("reference pooled max|v|: " + ", ".join(
        f"{s}={rrec['stages'][s]['pooled']['max_abs']:.4e}"
        if rrec["stages"].get(s, {}).get("status") == "located"
        else f"{s}=UNKNOWN" for s in
        ["left_features", "right_features", "cost_volume", "aggregated_cost",
         "disparity_initial", "refinement_residual", "disparity_final"]),
        flush=True)

    # ---- TASK 8: parity correlation (correlation ONLY) ----
    # Mapping: full-res pixel (y,x) in 368x1232 back to low-res cost coords in
    # 46x154. The regression path upsamples (B,24,46,154)->(B,24,368,1232)
    # with bilinear align_corners=True, so low-res grid point (i,j) sits at
    # full-res (i*(367/45), j*(1231/153)). Inverse mapping used here:
    #   i = round(y*45/367), j = round(x*153/1231), clipped to bounds.
    # Magnitude at a pixel = max over the 24 disparity candidates of |value|
    # at (i,j), for aggregated_cost and cost_postnorm.
    import csv as _csv
    rows = []
    with open(TOP_PIXELS) as f:
        for r in _csv.DictReader(f):
            rows.append(r)
    net = nets["armp"]
    mag_disc_agg, mag_disc_post, diffs = [], [], []
    with torch.no_grad():
        cache = {}
        for r in rows:
            scene, y, x = r["scene"], int(r["y"]), int(r["x"])
            if scene not in cache:
                s = ds[[d.name for d in
                        (ds[i] for i in range(5))].index(scene)]
                L, R = normalize(s.left), normalize(s.right)
                cache[scene] = capture_nine(
                    net, torch.from_numpy(L), torch.from_numpy(R))
            cap = cache[scene]
            i_lr = min(45, max(0, round(y * 45 / 367)))
            j_lr = min(153, max(0, round(x * 153 / 1231)))
            a = cap["aggregated_cost"][0, :, i_lr, j_lr].abs().max().item()
            p = cap["cost_postnorm"][0, :, i_lr, j_lr].abs().max().item()
            mag_disc_agg.append(a)
            mag_disc_post.append(p)
            diffs.append(float(r["abs_diff"]))
    rng = np.random.default_rng(0)
    mag_rand_agg, mag_rand_post = [], []
    with torch.no_grad():
        for i in range(5):
            s = ds[i]
            L, R = normalize(s.left), normalize(s.right)
            cap = capture_nine(net, torch.from_numpy(L),
                               torch.from_numpy(R))
            ys = rng.integers(0, 368, size=400)
            xs = rng.integers(0, 1232, size=400)
            for y, x in zip(ys, xs):
                i_lr = min(45, max(0, round(int(y) * 45 / 367)))
                j_lr = min(153, max(0, round(int(x) * 153 / 1231)))
                mag_rand_agg.append(float(
                    cap["aggregated_cost"][0, :, i_lr, j_lr].abs().max().item()))
                mag_rand_post.append(float(
                    cap["cost_postnorm"][0, :, i_lr, j_lr].abs().max().item()))
    sp_agg = spearmanr(diffs, mag_disc_agg)
    sp_post = spearmanr(diffs, mag_disc_post)
    out["parity_correlation"] = {
        "source": "stage_c_deploy/diagnostics/top_pixels_armp.csv (n=20)",
        "mapping": ("i=round(y*45/367) clip[0,45], j=round(x*153/1231) "
                    "clip[0,153]; align_corners=True inverse; magnitude = "
                    "max over 24 disparity candidates of |value| at (i,j)"),
        "n_discrepant": len(rows),
        "n_random": len(mag_rand_agg),
        "spearman_absdiff_vs_agg_magnitude": {
            "rho": float(sp_agg.statistic), "pvalue": float(sp_agg.pvalue)},
        "spearman_absdiff_vs_postnorm_magnitude": {
            "rho": float(sp_post.statistic), "pvalue": float(sp_post.pvalue)},
        "agg_magnitude_at_discrepant": {
            "median": float(np.median(mag_disc_agg)),
            "mean": float(np.mean(mag_disc_agg)),
            "max": float(np.max(mag_disc_agg))},
        "agg_magnitude_at_random": {
            "median": float(np.median(mag_rand_agg)),
            "mean": float(np.mean(mag_rand_agg)),
            "max": float(np.max(mag_rand_agg))},
        "postnorm_magnitude_at_discrepant": {
            "median": float(np.median(mag_disc_post)),
            "mean": float(np.mean(mag_disc_post)),
            "max": float(np.max(mag_disc_post))},
        "postnorm_magnitude_at_random": {
            "median": float(np.median(mag_rand_post)),
            "mean": float(np.mean(mag_rand_post)),
            "max": float(np.max(mag_rand_post))},
        "claim": ("correlation ONLY. No causal claim about the parity "
                  "failure is made here."),
    }
    print("parity spearman (agg): rho=%.3f p=%.3g; (post): rho=%.3f p=%.3g"
          % (sp_agg.statistic, sp_agg.pvalue,
             sp_post.statistic, sp_post.pvalue), flush=True)

    with open(OUT / "dr0_activation_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT / "dr0_activation_audit.json", flush=True)


if __name__ == "__main__":
    main()
