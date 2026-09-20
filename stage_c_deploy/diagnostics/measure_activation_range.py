"""Activation dynamic-range measurement for the C1 parity-diagnostic correction.

ZERO TRAINING. Diagnosis only: no weight change, no architecture change, no
tolerance change, no modification of any existing repo artifact.

What it does:
  1. ARM-P seed 1 + P2A seed 0 (PyTorch, CPU, eval, no_grad): per-stage
     max|value|, mean|value|, 99.9th percentile of |value| on each of the 5
     frozen-contract scenes (hailo_val indices 0-4), via
     model(L, R, return_stages=True), plus pooled stats.
  2. Reference StereoNet (reference/onnx/stereonet.onnx, READ-ONLY, never
     modified): builds a DEBUG copy
     stage_c_deploy/diagnostics/reference_instrumented_debug.onnx exposing
     the 6 stage tensors as extra graph outputs, runs it under onnxruntime
     CPU, and records max|value| (/mean/p99.9) for the same stages on the
     same 5 scenes.

Reference input notes (recorded honestly in the JSON):
  - input names are `input.1` / `input.83`;
  - normalisation is compiler-inserted, NOT in the ONNX, so the reference is
    fed the same ImageNet-normalised input the host-side reproduction uses;
  - the reference cost volume is the degenerate no-op (EXP-010) and its
    geometry differs (downsample x16 -> (1,32,23,77) features, 12 candidates
    vs ARM-P/P2A x8 -> (1,32,46,154), 24 candidates), so stage semantics are
    not identical; the comparison is about magnitudes the quantiser had to
    cope with, not equivalent computation.

Reference stage anchors (graph-topology grounded, verified by inspection):
  - left/right features: /res/res.6/Conv_output_0 and /res/res.6_1/Conv_output_0
    (the two branch-final feature tensors feeding the Sub/Slice cost-volume
    subgraph); left/right assignment by backward trace to input.1/input.83.
  - cost_volume: /Transpose_output_0 (unique (1,32,12,23,77) semantic anchor).
  - aggregated_cost: /Squeeze_output_0 (unique (1,12,23,77) candidate).
  - disparity_initial: /ReduceSum_output_0 (shape (1,1,368,1232) inferable).
  - refinement_residual: /refine/refine.7/Conv_output_0 (last refine conv).
  - disparity_final: 524 (graph output).
If any anchor is absent from the instrumented outputs, that stage is UNKNOWN
rather than guessed.
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
from onnx import helper

from src.datasets.kitti2015 import Kitti2015Stereo, normalize
from src.models.stereonet import StereoNet, StereoNetConfig

OUT = REPO / "stage_c_deploy" / "diagnostics"
REF_ONNX = REPO / "reference" / "onnx" / "stereonet.onnx"
REF_DBG = OUT / "reference_instrumented_debug.onnx"

CFG = dict(downsample_levels=3, num_disparities=24, cost_volume_shift="right",
           regression_normalize=True)
CKPTS = {
    "armp": "stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth",
    "p2a": "phase2/runs/p2a_scale_coverage/p2a_best.pth",
}
STAGES = ["left_features", "right_features", "cost_volume",
          "aggregated_cost", "disparity_initial",
          "refinement_residual", "disparity_final"]

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


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stats(absflat: np.ndarray) -> dict:
    return {"max_abs": float(absflat.max()),
            "mean_abs": float(absflat.mean()),
            "p999_abs": float(np.percentile(absflat, 99.9)),
            "n": int(absflat.size)}


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

    ds = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                         disparity_scale=256.0, occluded=True)
    assert [ds[i].name for i in range(5)] == \
        [f"00016{i}_10.png" for i in range(5)], "scene order changed -> STOP"
    scenes = [ds[i].name for i in range(5)]

    out: dict = {
        "provenance": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "git_head": git_head, "host": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "onnxruntime": ort.__version__, "numpy": np.__version__,
            "reference_onnx_sha256": sha256(REF_ONNX),
            "note": "No existing repo artifact was modified. New files only: "
                    "measure_activation_range.py, activation_range.json, "
                    "reference_instrumented_debug.onnx (debug copy). "
                    "reference/onnx/stereonet.onnx read but never written.",
        },
        "scenes": scenes,
        "models": {},
    }

    # ---- 1. ARM-P + P2A (PyTorch) ----
    for mname, rel in CKPTS.items():
        blob = torch.load(REPO / rel, map_location="cpu", weights_only=False)
        net = StereoNet(StereoNetConfig(**CFG))
        net.load_state_dict(blob["model"], strict=True)
        net.eval().cpu()
        mrec: dict = {"backend": "pytorch-cpu-eval-no_grad",
                      "stages": {s: {"per_scene": [], "pooled": None}
                                 for s in STAGES}}
        pool: dict[str, list] = {s: [] for s in STAGES}
        with torch.no_grad():
            for i in range(5):
                s = ds[i]
                L, R = normalize(s.left), normalize(s.right)
                _, st = net(torch.from_numpy(L), torch.from_numpy(R),
                            return_stages=True)
                for stage in STAGES:
                    a = np.abs(st[stage].numpy().astype(np.float64)).ravel()
                    mrec["stages"][stage]["per_scene"].append(
                        {"scene": s.name, **stats(a)})
                    pool[stage].append(a)
        for stage in STAGES:
            mrec["stages"][stage]["pooled"] = stats(
                np.concatenate(pool[stage]))
        out["models"][mname] = mrec
        p = mrec["stages"]
        print(f"{mname}: scene4 max|v|: " +
              ", ".join(f"{s}={p[s]['per_scene'][4]['max_abs']:.4e}"
                        for s in STAGES), flush=True)

    # ---- 2. reference (ORT, instrumented debug copy) ----
    model = onnx.load(str(REF_ONNX))
    # resolve left/right branch assignment by backward trace
    branch_src = {t: trace_to_input(model, t) for t in REF_BRANCH_TENSORS}
    print("branch trace:", branch_src, flush=True)
    anchors = dict(REF_ANCHORS)
    # input.1 is the left image (first session input); verify by order
    sess0 = ort.InferenceSession(str(REF_ONNX), providers=["CPUExecutionProvider"])
    first_input = sess0.get_inputs()[0].name
    left_src = first_input  # left image fed as first input
    for t, src in branch_src.items():
        if src == left_src:
            anchors["left_features"] = t
        else:
            anchors["right_features"] = t
    # fallback: if trace failed, leave UNKNOWN (no guessing)
    existing = {o.name for o in model.graph.output}
    added = []
    for stage, nm in anchors.items():
        if nm is not None and nm not in existing:
            # shape unknown statically; use None dims (dynamic) to avoid lying
            model.graph.output.append(helper.make_tensor_value_info(
                nm, onnx.TensorProto.FLOAT, None))
            existing.add(nm)
            added.append(nm)
    onnx.save(model, str(REF_DBG))
    print("reference debug copy:", REF_DBG.name, "added:", added, flush=True)

    # record which anchors were actually located
    located = {s: (nm if nm in existing else None)
               for s, nm in anchors.items()}
    out["models"]["reference"] = {
        "backend": "onnxruntime-CPU",
        "input_note": "fed the same ImageNet-normalised input the host-side "
                      "reproduction uses (normalisation is compiler-inserted, "
                      "NOT in the ONNX); input names input.1/input.83, left "
                      "image as first input.",
        "caveat": "reference cost volume is the degenerate no-op (EXP-010); "
                  "geometry differs (x16 downsample, 12 candidates vs x8, 24 "
                  "for ARM-P/P2A). Stage semantics not identical; comparison "
                  "is about magnitudes the quantiser had to cope with.",
        "anchors": located,
        "branch_trace": branch_src,
        "stages": {},
    }
    rrec = out["models"]["reference"]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(REF_DBG), sess_options=so,
                                providers=["CPUExecutionProvider"])
    inames = [i.name for i in sess.get_inputs()]
    assert inames[0] == "input.1" and inames[1] == "input.83", \
        f"reference input names changed: {inames} -> STOP"
    onames = [o.name for o in sess.get_outputs()]
    rpool: dict[str, list] = {}
    for i in range(5):
        s = ds[i]
        L, R = normalize(s.left), normalize(s.right)
        outs = sess.run(onames, {inames[0]: L, inames[1]: R})
        omap = dict(zip(onames, outs))
        for stage in STAGES:
            nm = located[stage]
            if nm is None or nm not in omap or omap[nm] is None:
                if i == 0:
                    rrec["stages"][stage] = {"status": "UNKNOWN",
                                             "reason": "anchor not located; "
                                                       "no guessing"}
                continue
            a = np.abs(np.asarray(omap[nm], dtype=np.float64)).ravel()
            rpool.setdefault(stage, []).append(a)
            rrec["stages"].setdefault(
                stage, {"onnx_tensor": nm, "per_scene": [], "pooled": None,
                        "status": "located"})
            rrec["stages"][stage]["per_scene"].append(
                {"scene": s.name, **stats(a)})
    for stage, arrs in rpool.items():
        rrec["stages"][stage]["pooled"] = stats(np.concatenate(arrs))
    print("reference: scene4 max|v|: " +
          ", ".join(f"{s}={rrec['stages'][s]['per_scene'][4]['max_abs']:.4e}"
                    if rrec["stages"].get(s, {}).get("status") == "located"
                    else f"{s}=UNKNOWN" for s in STAGES), flush=True)

    with open(OUT / "activation_range.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT / "activation_range.json", flush=True)


if __name__ == "__main__":
    main()
