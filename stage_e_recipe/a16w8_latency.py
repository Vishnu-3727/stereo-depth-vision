#!/usr/bin/env python
"""A16W8 CPU latency next to fp32/R0/R2, host part: A16 regen + CPU latency.

Implements docs/superpowers/specs/2026-09-24-a16w8-latency-design.md
(host, single process). Characterization only: no latency thresholds.

Steps: sha-assert fp32/R0 subjects; reuse-or-regen R2 with sha-assert;
regenerate A16 via the int8_a16w8 guarded child-process quantization;
inspect the kept A16 graph (opset + QuantizeLinear zero-point dtypes);
reuse-or-precompute 10-scene latency inputs; time all four configs
sequentially in THIS process (10 warm-up + 10 scenes x 5 repeats,
median/p90, ORT CPU EP, ORT_ENABLE_ALL); re-score all four on the
40-scene contract (EPE gates 1e-6, any miss is STOP).

Run with the host python, with no LLM process resident (quantization
peaks ~5.6 GB RAM). Writes only under stage_e_recipe/a16w8_latency/.
This script writes its JSON record itself; do not parse stdout.
"""

from __future__ import annotations

import ctypes
import datetime
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import int8_sensitivity as S  # noqa: E402
import int8_a16w8 as A16Q  # noqa: E402
import int8_regression_head as RH  # noqa: E402
from mixed_precision_cost import p90, prefix_names  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402

WORK = HERE / "a16w8_latency"
OUT = WORK / "a16w8_latency.json"
A16_DIR = WORK / "a16_regen"
NPY_FALLBACK_DIR = WORK / "inputs_npy"
NPY_REUSE_DIR = HERE / "mixed_precision_cost" / "inputs_npy"
R2_REUSE = (HERE / "mixed_precision_cost" / "r2_regen"
            / "seed0_R2_head_refinement_fp32.onnx")

ARTEFACTS = {
    "fp32": {
        "path": "stage_e_recipe/int8_e3/e3_seed0_best_fp32.onnx",
        "sha256": "7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645",
        "bytes": 1701550,
        "ref_epe": 1.18598534271453,
        "ref_src": "stage_e_recipe/int8_e3/int8_e3.json seed 0 fp32",
    },
    "R0": {
        "path": "stage_e_recipe/int8_e3/e3_seed0_best_int8.onnx",
        "sha256": "8b31f94921878f7f08ae122e00a1e36983fc2fb19dae5af3c07566589598c68a",
        "bytes": 764520,
        "ref_epe": 5.248941413313285,
        "ref_src": "stage_e_recipe/int8_e3/int8_e3.json seed 0 int8",
    },
    "R2": {
        "path": None,  # resolved below: reuse R2_REUSE or regenerate
        "reuse_path": "stage_e_recipe/mixed_precision_cost/r2_regen/seed0_R2_head_refinement_fp32.onnx",
        "sha256": "b12d1e0dd591750d8711ef77d528fe5756ba0f05cfdfe3bd390fc0b6dfb644ef",
        "bytes": 1050225,
        "ref_epe": 1.2908231335718434,
        "ref_src": "stage_e_recipe/mixed_precision_cost/mixed_precision_cost.json R2",
    },
    "A16": {
        "path": None,  # regenerated below
        "recorded_sha256": "8dd30bca2d0b7ce272a1635cbfdf6aa500fe897c0b6df46e78ed0a3528465988",
        "recorded_bytes": 764752,
        "ref_epe": 1.2603345881778643,
        "ref_src": "stage_e_recipe/int8_a16w8/int8_a16w8.json seed 0 e_A16",
    },
}

N_SCENES = 10
N_REPEATS = 5
N_WARMUP = 10
REPRO_TOL = 1e-6

ELEM_TYPE_NAMES = {
    0: "UNDEFINED", 1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16",
    5: "INT16", 6: "INT32", 7: "INT64", 8: "STRING", 9: "BOOL",
    10: "FLOAT16", 11: "DOUBLE", 12: "UINT32", 13: "UINT64",
    14: "COMPLEX64", 15: "COMPLEX128", 16: "BFLOAT16",
    17: "FLOAT8E4M3FN", 18: "FLOAT8E4M3FNUZ", 19: "FLOAT8E5M2",
    20: "FLOAT8E5M2FNUZ", 21: "UINT4", 22: "INT4", 23: "FLOAT4E2M1",
}


def fail(reason: str, report: dict) -> "NoReturn":
    report["status"] = f"STOP: {reason}"
    report["utc_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    OUT.write_text(json.dumps(report, indent=2))
    print(f"STOP: {reason} (partial report -> {OUT})", flush=True)
    raise SystemExit(1)


def cpu_name() -> dict:
    info: dict = {"platform_processor": platform.processor(),
                  "machine": platform.machine()}
    try:
        r = subprocess.run(["wmic", "cpu", "get", "name"],
                           capture_output=True, text=True, timeout=15)
        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        info["wmic_cpu_name"] = lines[1] if len(lines) > 1 else None
    except Exception:
        info["wmic_cpu_name"] = None
    return info


def free_ram_gb() -> float | None:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullAvailPhys / (1024 ** 3), 3)
    except Exception:
        return None


def inspect_a16_graph(path: Path) -> dict:
    import onnx

    m = onnx.load(str(path))
    opset = None
    for o in m.opset_import:
        if o.domain in ("", "ai.onnx"):
            opset = o.version
    dtype_by_name: dict[str, int] = {}
    for i in m.graph.initializer:
        dtype_by_name[i.name] = i.data_type
    for gi in m.graph.input:
        tt = gi.type.tensor_type
        if tt.HasField("elem_type"):
            dtype_by_name.setdefault(gi.name, tt.elem_type)
    counts: dict[str, int] = {}
    n_ql = 0
    for n in m.graph.node:
        if n.op_type != "QuantizeLinear":
            continue
        n_ql += 1
        zp = n.input[2] if len(n.input) > 2 else ""
        if not zp:
            counts["absent"] = counts.get("absent", 0) + 1
        elif zp not in dtype_by_name:
            counts["unknown"] = counts.get("unknown", 0) + 1
        else:
            v = dtype_by_name[zp]
            name = ELEM_TYPE_NAMES.get(v, f"elem_type_{v}")
            counts[name] = counts.get(name, 0) + 1
    return {"opset": opset, "n_quantizelinear": n_ql,
            "zero_point_dtype_counts": counts,
            "size_bytes": path.stat().st_size,
            "sha256": S.sha256(path)}


def main() -> None:
    import onnxruntime as ort

    WORK.mkdir(parents=True, exist_ok=True)
    A16_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "spec": "docs/superpowers/specs/2026-09-24-a16w8-latency-design.md",
        "part": "host",
        "utc_start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head": S.git_head(),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "cpu_count": os.cpu_count(),
            "cpu": cpu_name(),
            "free_ram_gb_at_start": free_ram_gb(),
        },
        "protocol": {
            "scenes": f"0-{N_SCENES - 1} (head of hailo_val)",
            "warmup_runs": N_WARMUP,
            "repeats_per_scene": N_REPEATS,
            "timed_samples": N_SCENES * N_REPEATS,
            "scope": "sess.run only (preprocessing excluded, inputs precomputed .npy)",
            "clock": "time.perf_counter",
            "graph_optimization": "ORT_ENABLE_ALL",
            "process": "all 4 configs timed sequentially in ONE process",
        },
        "context": {
            "note": ("Recorded medians from mixed_precision_cost.json "
                     "(cross-session context, NOT comparisons)."),
            "fp32_median_ms": 766.75, "R0_median_ms": 918.09,
            "R2_median_ms": 818.31,
        },
        "measured": {},
        "inferred": {},
        "unknown": {},
    }

    # --- spec §1 subject sha asserts (fp32, R0 on disk) ---
    for tag in ("fp32", "R0"):
        cfg = ARTEFACTS[tag]
        p = REPO / cfg["path"]
        if not p.exists():
            fail(f"subject {tag} missing on disk: {p}", report)
        got = S.sha256(p)
        if got != cfg["sha256"]:
            fail(f"subject {tag} sha mismatch: {got} != recorded {cfg['sha256']}",
                 report)
        if p.stat().st_size != cfg["bytes"]:
            fail(f"subject {tag} size {p.stat().st_size} != recorded {cfg['bytes']}",
                 report)
        print(f"subjects: {tag} OK (sha match, {cfg['bytes']} bytes)", flush=True)
    report["measured"]["subject_asserts"] = {
        tag: {"sha256": ARTEFACTS[tag]["sha256"], "bytes": ARTEFACTS[tag]["bytes"],
              "match": True} for tag in ("fp32", "R0")
    }

    # --- spec §1 R2: reuse the mixed-precision-cost file, else regen ---
    r2_reused = False
    if R2_REUSE.exists():
        got_r2 = S.sha256(R2_REUSE)
        if got_r2 != ARTEFACTS["R2"]["sha256"]:
            fail(f"R2 reuse sha mismatch: {got_r2} != recorded "
                 f"{ARTEFACTS['R2']['sha256']}", report)
        if R2_REUSE.stat().st_size != ARTEFACTS["R2"]["bytes"]:
            fail("R2 reuse size differs from recorded "
                 f"{ARTEFACTS['R2']['bytes']}", report)
        ARTEFACTS["R2"]["path"] = ARTEFACTS["R2"]["reuse_path"]
        r2_reused = True
        print(f"R2: reused sha-verified file ({ARTEFACTS['R2']['bytes']} bytes)",
              flush=True)
        report["measured"]["r2_source"] = {"reused": True,
                                           "path": ARTEFACTS["R2"]["reuse_path"]}
    else:
        fp32 = REPO / ARTEFACTS["fp32"]["path"]
        calib = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_calib")
        head, _ = RH.kept_set(fp32)
        head_rt = prefix_names(fp32, "/regression/")
        assert head == head_rt, "head set differs from runtime prefix enumeration"
        refin = prefix_names(fp32, "/refinement/")
        for m in ("/regression/Mul", "/regression/Mul_1", "/regression/Mul_2"):
            assert m in head, f"required head node {m} missing"
        for m in ("/refinement/input_conv/Conv", "/refinement/output_conv/Conv"):
            assert m in refin, f"required refinement node {m} missing"
        combined = sorted(set(head) | set(refin))
        dst_r2 = WORK / "r2_regen" / "seed0_R2_head_refinement_fp32.onnx"
        dst_r2.parent.mkdir(parents=True, exist_ok=True)
        qrec = S.quantize_with_excludes(fp32, dst_r2, calib, combined)
        got_r2 = S.sha256(dst_r2)
        if got_r2 != ARTEFACTS["R2"]["sha256"]:
            fail(f"regenerated R2 sha mismatch: {got_r2} != recorded "
                 f"{ARTEFACTS['R2']['sha256']}", report)
        print(f"R2: regenerated sha match ({dst_r2.stat().st_size} bytes, "
              f"quantize {qrec['quantize_s']}s)", flush=True)
        report["measured"]["r2_source"] = {"reused": False, **qrec,
                                           "n_excluded": len(combined)}
        ARTEFACTS["R2"]["path"] = str(dst_r2.relative_to(REPO)).replace("\\", "/")

    val = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                          disparity_scale=256.0, occluded=True)
    assert len(val) == 40, f"hailo_val has {len(val)} scenes, expected 40"

    # --- spec §2 A16 regeneration via the int8_a16w8 guarded child ---
    fp32 = REPO / ARTEFACTS["fp32"]["path"]
    dst_a16 = A16_DIR / "seed0_A16.onnx"
    gc.collect()
    qrec = A16Q.quantize_with_guard(fp32, dst_a16, None, use_a16=True)
    if "error" in qrec:
        fail(f"A16 guarded quantization failed: {qrec}", report)
    got_a16 = S.sha256(dst_a16)
    sha_match = got_a16 == ARTEFACTS["A16"]["recorded_sha256"]
    print(f"A16: regenerated sha {'match' if sha_match else 'MISMATCH'} "
          f"({dst_a16.stat().st_size} bytes, quantize {qrec.get('quantize_s')}s, "
          f"child peak {qrec.get('peak_child_mem_gb')} GB)", flush=True)
    report["measured"]["a16_regen"] = {
        **qrec, "sha256": got_a16,
        "recorded_sha256": ARTEFACTS["A16"]["recorded_sha256"],
        "sha_match": sha_match,
        "kept_path": str(dst_a16.relative_to(REPO)).replace("\\", "/"),
    }
    ARTEFACTS["A16"]["path"] = str(dst_a16.relative_to(REPO)).replace("\\", "/")

    # --- spec §6 A16 graph inspection (kept graph, read-only) ---
    graph_info = inspect_a16_graph(dst_a16)
    print(f"A16 graph: opset {graph_info['opset']}, "
          f"QuantizeLinear {graph_info['n_quantizelinear']}, "
          f"zp dtypes {graph_info['zero_point_dtype_counts']}", flush=True)
    report["measured"]["a16_graph"] = graph_info

    # --- spec §4 inputs: reuse bit-identical .npy, else precompute ---
    inputs: list[tuple[np.ndarray, np.ndarray]] = []
    if NPY_REUSE_DIR.is_dir() and all(
            (NPY_REUSE_DIR / f"scene{i}_{s}.npy").exists()
            for i in range(N_SCENES) for s in ("left", "right")):
        for i in range(N_SCENES):
            l = np.load(str(NPY_REUSE_DIR / f"scene{i}_left.npy"))
            r = np.load(str(NPY_REUSE_DIR / f"scene{i}_right.npy"))
            assert l.shape == (1, 3, 368, 1232) and l.dtype == np.float32
            assert r.shape == (1, 3, 368, 1232) and r.dtype == np.float32
            assert l.flags["C_CONTIGUOUS"] and r.flags["C_CONTIGUOUS"]
            inputs.append((l, r))
        print(f"inputs: reused {N_SCENES} scenes from {NPY_REUSE_DIR}", flush=True)
        report["measured"]["inputs"] = {
            "source": "reused",
            "dir": str(NPY_REUSE_DIR.relative_to(REPO)).replace("\\", "/"),
            "scenes": N_SCENES, "shape": [1, 3, 368, 1232], "dtype": "float32",
        }
    else:
        NPY_FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(N_SCENES):
            s = val[i]
            l, r = normalize(s.left), normalize(s.right)
            assert l.shape == (1, 3, 368, 1232) and l.dtype == np.float32
            assert r.shape == (1, 3, 368, 1232) and r.dtype == np.float32
            assert l.flags["C_CONTIGUOUS"] and r.flags["C_CONTIGUOUS"]
            np.save(NPY_FALLBACK_DIR / f"scene{i}_left.npy", l)
            np.save(NPY_FALLBACK_DIR / f"scene{i}_right.npy", r)
            inputs.append((l, r))
        print(f"inputs: precomputed {N_SCENES} scenes into {NPY_FALLBACK_DIR}",
              flush=True)
        report["measured"]["inputs"] = {
            "source": "precomputed",
            "dir": str(NPY_FALLBACK_DIR.relative_to(REPO)).replace("\\", "/"),
            "scenes": N_SCENES, "shape": [1, 3, 368, 1232], "dtype": "float32",
        }

    # --- per-artefact (ONE process, sequential): H-CPU latency + re-score ---
    rows: dict[str, dict] = {}
    for tag in ("fp32", "R0", "R2", "A16"):
        cfg = ARTEFACTS[tag]
        p = REPO / cfg["path"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(p), sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        ins = [i.name for i in sess.get_inputs()]
        assert len(ins) == 2, f"{tag}: expected 2 inputs, got {ins}"
        out_name = sess.get_outputs()[0].name
        thread_settings = {
            "cpu_count": os.cpu_count(),
            "intra_op_num_threads": opts.intra_op_num_threads,
            "inter_op_num_threads": opts.inter_op_num_threads,
            "note": "defaults, no explicit values set",
        }

        for _ in range(N_WARMUP):  # untimed warm-up on scene-0 input
            sess.run([out_name], {ins[0]: inputs[0][0], ins[1]: inputs[0][1]})

        samples: list[float] = []
        per_scene_median: list[float] = []
        for i in range(N_SCENES):
            scene_samples: list[float] = []
            for _ in range(N_REPEATS):
                t0 = time.perf_counter()
                sess.run([out_name], {ins[0]: inputs[i][0], ins[1]: inputs[i][1]})
                dt_ms = (time.perf_counter() - t0) * 1e3
                samples.append(dt_ms)
                scene_samples.append(dt_ms)
            per_scene_median.append(float(np.median(scene_samples)))

        med = float(np.median(samples))
        row = {
            "size_bytes": p.stat().st_size,
            "cpu_latency": {
                "n_samples": len(samples),
                "median_ms": med,
                "p90_ms": float(p90(samples)),
                "min_ms": float(min(samples)),
                "max_ms": float(max(samples)),
                "fps_from_median": 1000.0 / med if med > 0 else float("nan"),
                "per_scene_median_ms": per_scene_median,
                "raw_samples_ms": samples,
            },
            "thread_settings": thread_settings,
            "session": {"providers": sess.get_providers(), "inputs": ins},
        }
        del sess
        gc.collect()

        scored = S.score_contract(p, val)
        epe = scored["metrics"]["epe"]
        row["cpu_40scene"] = {
            "epe": epe, "d1": scored["metrics"]["d1"],
            "rmse": scored["metrics"]["rmse"],
            "valid_pixels": scored["metrics"]["valid_pixels"],
            "nonfinite_pixels": scored["nonfinite_pixels"],
            "contract_match": scored["guard"]["contract_match"],
            "score_s": scored["score_s"],
            "recorded_ref_epe": cfg["ref_epe"],
            "ref_src": cfg["ref_src"],
            "abs_diff_vs_recorded": abs(epe - cfg["ref_epe"]),
        }
        if not scored["guard"]["contract_match"]:
            fail(f"{tag}: contract_match false on CPU re-score", report)
        if abs(epe - cfg["ref_epe"]) > REPRO_TOL:
            fail(f"{tag}: EPE {epe} != recorded {cfg['ref_epe']} within 1e-6",
                 report)
        print(f"{tag}: size {row['size_bytes']} median {med:.2f}ms "
              f"p90 {row['cpu_latency']['p90_ms']:.2f}ms "
              f"EPE {epe:.7f} (ref {cfg['ref_epe']:.7f})", flush=True)
        rows[tag] = row

    report["measured"]["rows"] = rows
    report["measured"]["epe_gates"] = {"pass": True, "tol_px": REPRO_TOL}
    report["inferred"]["note"] = (
        "Latency rows describe ORT CPU EP inference on THIS host in ONE "
        "process; they do not predict NPU/Hailo behaviour. EPE rows "
        "re-measure recorded references.")
    report["unknown"]["items"] = [
        "Repeatability of every latency number (single run each).",
        "Any transfer beyond this machine/installation.",
    ]
    report["status"] = "COMPLETE: A16 regen + 4-config CPU latency + CPU re-scores"
    report["utc_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    OUT.write_text(json.dumps(report, indent=2))
    print(f"COMPLETE wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
