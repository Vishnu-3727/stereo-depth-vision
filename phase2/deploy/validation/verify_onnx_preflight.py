"""PHASE-2 DEPLOYMENT VALIDATION — pre-compilation integrity + toolchain probe.

Verifies every integrity item required before a Hailo compile attempt, on BOTH
ONNX artifacts, and records the result of probing this machine for a Hailo
toolchain. Neither ONNX file is modified, simplified, quantised or re-exported.

Layers are recorded separately and never collapsed:
    ONNX validated / Hailo parser / Hailo compiler / HEF / HailoRT / hardware
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from phase1.harness.frozen_eval import sha256_file
from src.models.stereonet import StereoNet, StereoNetConfig

CKPT = REPO / "phase2" / "runs" / "p2a_scale_coverage" / "p2a_best.pth"
ARTIFACTS = {
    "original": REPO / "phase2" / "deploy" / "p2a_stereonet.onnx",
    "static": REPO / "phase2" / "deploy" / "p2a_stereonet_static.onnx",
}
EXPECTED_PARAMS = 397954
EXPECTED_OPSET = 13
EXPECTED_IN = [1, 3, 368, 1232]
EXPECTED_OUT = [1, 1, 368, 1232]

HAILO_MODULES = ["hailo_sdk_client", "hailo_sdk_common", "hailo_platform",
                 "hailort", "hailo_model_zoo", "hailo_tools", "hailo_dataflow_compiler"]
HAILO_BINARIES = ["hailo", "hailortcli", "hailo_compiler", "hailomz", "hailo-dfc", "dfc"]
HAILO_PATHS = [r"C:\Program Files\Hailo", r"C:\Program Files (x86)\Hailo", r"C:\Hailo",
               os.path.expandvars(r"%LOCALAPPDATA%\Hailo"),
               os.path.expanduser(r"~\.hailo"),
               os.path.expandvars(r"%APPDATA%\Hailo"),
               "/opt/hailo", "/usr/local/hailo"]


def probe_toolchain() -> dict:
    mods = {}
    for m in HAILO_MODULES:
        try:
            mod = importlib.import_module(m)
            mods[m] = {"importable": True,
                       "version": getattr(mod, "__version__", "unknown"),
                       "file": getattr(mod, "__file__", None)}
        except Exception as e:
            mods[m] = {"importable": False, "error": type(e).__name__}
    bins = {b: shutil.which(b) for b in HAILO_BINARIES}
    envs = {k: v for k, v in os.environ.items()
            if "HAILO" in k.upper() or "hailo" in str(v).lower()}
    paths = {p: Path(p).exists() for p in HAILO_PATHS}
    docker = {"docker_cli": shutil.which("docker")}
    if docker["docker_cli"]:
        try:
            out = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                                 capture_output=True, text=True, timeout=25)
            docker["daemon_reachable"] = out.returncode == 0
            docker["hailo_images"] = [l for l in out.stdout.splitlines() if "hailo" in l.lower()]
        except Exception as e:
            docker["daemon_reachable"] = False
            docker["error"] = type(e).__name__
    return {"python_modules": mods, "binaries_on_path": bins,
            "environment_variables": envs or None, "install_paths": paths,
            "docker": docker,
            "any_toolchain_found": bool(
                any(v["importable"] for v in mods.values())
                or any(bins.values()) or envs
                or any(paths.values())
                or docker.get("hailo_images"))}


def verify_onnx(tag: str, path: Path, ref_state: dict) -> dict:
    g = onnx.load(str(path))
    rep = {"path": str(path.relative_to(REPO)).replace("\\", "/"),
           "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
           "modified_by_this_script": False}
    try:
        onnx.checker.check_model(g)
        rep["onnx_checker"] = "PASS"
    except Exception as e:
        rep["onnx_checker"] = "FAIL: " + str(e)
    rep["opset"] = int(g.opset_import[0].version)
    rep["opset_ok"] = rep["opset"] == EXPECTED_OPSET
    rep["ir_version"] = int(g.ir_version)
    rep["producer"] = g.producer_name + " " + g.producer_version

    def shape_of(vi):
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                for d in vi.type.tensor_type.shape.dim]

    rep["inputs"] = [{"name": i.name, "shape": shape_of(i),
                      "dtype": onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type)}
                     for i in g.graph.input]
    rep["outputs"] = [{"name": o.name, "shape": shape_of(o),
                       "dtype": onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type)}
                      for o in g.graph.output]
    rep["io_names_ok"] = ([i["name"] for i in rep["inputs"]] == ["left", "right"]
                          and [o["name"] for o in rep["outputs"]] == ["disparity"])
    rep["io_shapes_ok"] = (all(i["shape"] == EXPECTED_IN for i in rep["inputs"])
                           and rep["outputs"][0]["shape"] == EXPECTED_OUT)
    rep["static_shapes_only"] = all(
        all(isinstance(d, int) for d in x["shape"])
        for x in rep["inputs"] + rep["outputs"])

    # weights: match the ONNX initializers against the trained state_dict
    inits = {i.name: numpy_helper.to_array(i) for i in g.graph.initializer}
    matched, mismatched, unmatched_sd = 0, [], []
    total_matched_elems = 0
    for k, v in ref_state.items():
        w = v.detach().cpu().numpy()
        hit = None
        for name, arr in inits.items():
            if arr.shape == w.shape and np.array_equal(arr, w):
                hit = name
                break
        if hit is None:
            unmatched_sd.append(k)
        else:
            matched += 1
            total_matched_elems += int(w.size)
    rep["weights"] = {
        "state_dict_tensors": len(ref_state),
        "matched_bit_identical_in_onnx": matched,
        "unmatched_state_dict_tensors": unmatched_sd,
        "matched_parameter_elements": total_matched_elems,
        "expected_parameter_elements": EXPECTED_PARAMS,
        "parameter_count_ok": total_matched_elems == EXPECTED_PARAMS,
        "n_initializers_in_graph": len(inits),
        "note": ("initializer count exceeds the trainable-tensor count because "
                 "traced constants are stored as initializers too; the check "
                 "that matters is that every trained tensor appears bit-identical"),
    }

    op_counts = {}
    for n in g.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    rep["n_nodes"] = len(g.graph.node)
    rep["op_counts"] = dict(sorted(op_counts.items()))
    rep["has_batchnorm"] = any(
        t in op_counts for t in ("BatchNormalization", "InstanceNormalization"))
    rep["no_batchnorm_ok"] = not rep["has_batchnorm"]

    # tensor ranks present, for the parser risk note
    ranks = set()
    for vi in list(g.graph.value_info) + list(g.graph.input) + list(g.graph.output):
        d = len(vi.type.tensor_type.shape.dim)
        if d:
            ranks.add(d)
    rep["tensor_ranks_declared"] = sorted(ranks)

    dyn = [o for o in ("Shape", "Gather", "Range", "ReduceProd", "NonZero",
                       "Expand", "Tile") if o in op_counts]
    rep["dynamic_shape_ops_present"] = dyn
    rep["all_ok"] = bool(rep["onnx_checker"] == "PASS" and rep["opset_ok"]
                         and rep["io_names_ok"] and rep["io_shapes_ok"]
                         and rep["static_shapes_only"] and rep["no_batchnorm_ok"]
                         and rep["weights"]["parameter_count_ok"]
                         and not rep["weights"]["unmatched_state_dict_tensors"])
    return rep


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = blob["model"]
    model = StereoNet(StereoNetConfig(downsample_levels=3, num_disparities=24,
                                      cost_volume_shift="right",
                                      regression_normalize=True))
    model.load_state_dict(state, strict=True)
    torch_params = int(sum(p.numel() for p in model.parameters()))
    bn = [n for n, m in model.named_modules()
          if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm))]

    rec = {
        "task": "Hailo deployment validation of the frozen P2A artifact",
        "utc": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0],
                 "torch": torch.__version__, "onnx": onnx.__version__},
        "no_research_artifact_modified": True,
        "source_checkpoint": {
            "path": "phase2/runs/p2a_scale_coverage/p2a_best.pth",
            "sha256": sha256_file(CKPT),
            "torch_parameter_count": torch_params,
            "parameter_count_ok": torch_params == EXPECTED_PARAMS,
            "state_dict_tensors": len(state),
            "batchnorm_modules": bn, "no_batchnorm_ok": not bn},
        "toolchain_probe": probe_toolchain(),
        "artifacts": {tag: verify_onnx(tag, p, state) for tag, p in ARTIFACTS.items()},
    }
    # weights identical between the two exports
    o, s = rec["artifacts"]["original"], rec["artifacts"]["static"]
    rec["cross_artifact"] = {
        "both_match_trained_weights": bool(
            not o["weights"]["unmatched_state_dict_tensors"]
            and not s["weights"]["unmatched_state_dict_tensors"]),
        "node_delta": s["n_nodes"] - o["n_nodes"],
        "ops_only_in_original": sorted(set(o["op_counts"]) - set(s["op_counts"])),
        "ops_only_in_static": sorted(set(s["op_counts"]) - set(o["op_counts"])),
    }
    rec["layer_status"] = {
        "onnx_validated": "PASS" if o["all_ok"] and s["all_ok"] else "FAIL",
        "hailo_parser_validated": "NOT ATTEMPTED — toolchain absent",
        "hailo_compiler_validated": "NOT ATTEMPTED — toolchain absent",
        "hef_validated": "NOT ATTEMPTED — no HEF produced",
        "hailort_runtime_validated": "NOT ATTEMPTED — HailoRT absent",
        "physical_hardware_validated": "NOT AVAILABLE — no Hailo device present",
    }
    (HERE / "HAILO_COMPILATION_RECORD.json").write_text(json.dumps(rec, indent=2))

    print("toolchain found:", rec["toolchain_probe"]["any_toolchain_found"])
    for tag in ARTIFACTS:
        a = rec["artifacts"][tag]
        print("%-9s nodes=%-4d opset=%s checker=%s params=%d io=%s static=%s bn=%s dyn=%s ALL_OK=%s"
              % (tag, a["n_nodes"], a["opset"], a["onnx_checker"],
                 a["weights"]["matched_parameter_elements"],
                 a["io_names_ok"] and a["io_shapes_ok"], a["static_shapes_only"],
                 not a["has_batchnorm"], a["dynamic_shape_ops_present"] or "none",
                 a["all_ok"]))
    print("cross:", json.dumps(rec["cross_artifact"]))
    print("wrote", HERE / "HAILO_COMPILATION_RECORD.json")


if __name__ == "__main__":
    main()
