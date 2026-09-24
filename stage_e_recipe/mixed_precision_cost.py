#!/usr/bin/env python
"""Mixed-precision cost, host part: R2 regen + CPU latency + CPU re-scores.

Implements docs/superpowers/specs/2026-09-24-mixed-precision-cost-design.md
sections 1-5 (host half) and 7-8. Characterization only: no thresholds.

Steps: sha-assert fp32/R0 subjects; regenerate R2 via import and sha-assert;
precompute 10-scene latency inputs to .npy; time H-CPU inference
(10 warm-up + 10 scenes x 5 repeats, median/p90); check H-CUDA availability;
re-score all three artefacts on the 40-scene contract (R2 EPE gate 1e-6).

Run with the host python. Writes only under stage_e_recipe/mixed_precision_cost/.
"""

from __future__ import annotations

import datetime
import json
import math
import os
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
import int8_regression_head as RH  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402

WORK = HERE / "mixed_precision_cost"
OUT = WORK / "mixed_precision_cost.json"
NPY_DIR = WORK / "inputs_npy"
R2_DIR = WORK / "r2_regen"

ARTEFACTS = {
    "fp32": {
        "path": "stage_e_recipe/int8_e3/e3_seed0_best_fp32.onnx",
        "sha256": "7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645",
        "bytes": 1701550,
        "ref_epe": 1.18598534271453,
    },
    "R0": {
        "path": "stage_e_recipe/int8_e3/e3_seed0_best_int8.onnx",
        "sha256": "8b31f94921878f7f08ae122e00a1e36983fc2fb19dae5af3c07566589598c68a",
        "bytes": 764520,
        "ref_epe": 5.248941413313285,
    },
    "R2": {
        "path": None,  # regenerated below
        "sha256": "b12d1e0dd591750d8711ef77d528fe5756ba0f05cfdfe3bd390fc0b6dfb644ef",
        "bytes": 1050225,
        "ref_epe": 1.2908231335718434,
    },
}

N_SCENES = 10
N_REPEATS = 5
N_WARMUP = 10
REPRO_TOL = 1e-6


def fail(reason: str, report: dict) -> "NoReturn":
    report["status"] = f"STOP: {reason}"
    OUT.write_text(json.dumps(report, indent=2))
    print(f"STOP: {reason} (partial report -> {OUT})", flush=True)
    raise SystemExit(1)


def p90(xs: list[float]) -> float:
    s = sorted(xs)
    k = max(1, math.ceil(0.9 * len(s)))
    return s[k - 1]


def prefix_names(fp32_path: Path, prefix: str) -> list[str]:
    import onnx

    m = onnx.load(str(fp32_path))
    return sorted(n.name for n in m.graph.node if n.name.startswith(prefix))


def main() -> None:
    import onnxruntime as ort

    WORK.mkdir(parents=True, exist_ok=True)
    NPY_DIR.mkdir(parents=True, exist_ok=True)
    R2_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "spec": "docs/superpowers/specs/2026-09-24-mixed-precision-cost-design.md",
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
        },
        "protocol": {
            "scenes": f"0-{N_SCENES - 1} (head of hailo_val)",
            "warmup_runs": N_WARMUP,
            "repeats_per_scene": N_REPEATS,
            "timed_samples": N_SCENES * N_REPEATS,
            "scope": "sess.run only (preprocessing excluded, inputs precomputed .npy)",
            "clock": "time.perf_counter",
            "graph_optimization": "ORT_ENABLE_ALL",
        },
        "measured": {},
        "inferred": {},
        "unknown": {},
    }

    # --- §1 subject sha asserts (fp32, R0 on disk) ---
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
        print(f"{tag} OK: sha match, {cfg['bytes']} bytes", flush=True)
    report["measured"]["subject_asserts"] = {
        tag: {"sha256": ARTEFACTS[tag]["sha256"], "bytes": ARTEFACTS[tag]["bytes"],
              "match": True} for tag in ("fp32", "R0")
    }

    # --- §2 R2 regeneration via import ---
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
    dst_r2 = R2_DIR / "seed0_R2_head_refinement_fp32.onnx"
    qrec = S.quantize_with_excludes(fp32, dst_r2, calib, combined)
    got_r2 = S.sha256(dst_r2)
    if got_r2 != ARTEFACTS["R2"]["sha256"]:
        fail(f"regenerated R2 sha mismatch: {got_r2} != recorded "
             f"{ARTEFACTS['R2']['sha256']}", report)
    print(f"R2 regen OK: sha match, {dst_r2.stat().st_size} bytes, "
          f"quantize {qrec['quantize_s']}s", flush=True)
    report["measured"]["r2_regen"] = {
        **qrec, "sha_match": True,
        "n_excluded": len(combined),
        "n_head": len(head), "n_refinement": len(refin),
    }
    ARTEFACTS["R2"]["path"] = str(dst_r2.relative_to(REPO)).replace("\\", "/")

    val = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                          disparity_scale=256.0, occluded=True)
    assert len(val) == 40, f"hailo_val has {len(val)} scenes, expected 40"

    # --- §4 input precompute (10 head scenes, normalize, .npy) ---
    inputs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(N_SCENES):
        s = val[i]
        l, r = normalize(s.left), normalize(s.right)
        assert l.shape == (1, 3, 368, 1232) and l.dtype == np.float32
        assert r.shape == (1, 3, 368, 1232) and r.dtype == np.float32
        assert l.flags["C_CONTIGUOUS"] and r.flags["C_CONTIGUOUS"]
        np.save(NPY_DIR / f"scene{i}_left.npy", l)
        np.save(NPY_DIR / f"scene{i}_right.npy", r)
        inputs.append((l, r))
    report["measured"]["inputs"] = {
        "dir": str(NPY_DIR.relative_to(REPO)).replace("\\", "/"),
        "scenes": N_SCENES, "shape": [1, 3, 368, 1232], "dtype": "float32",
    }

    # --- H-CUDA availability check (§3: expected unavailable) ---
    avail = ort.get_available_providers()
    report["measured"]["cuda"] = {
        "available": "CUDAExecutionProvider" in avail,
        "observed_providers": avail,
        "note": ("CUDAExecutionProvider absent from host onnxruntime; "
                 "no torch-CUDA substitute (would not be an ORT measurement)."),
    }
    print(f"H-CUDA available: {'CUDAExecutionProvider' in avail} "
          f"(providers={avail})", flush=True)

    # --- per-artefact: H-CPU latency + 40-scene CPU re-score ---
    rows: dict[str, dict] = {}
    for tag in ("fp32", "R0", "R2"):
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
            "abs_diff_vs_recorded": abs(epe - cfg["ref_epe"]),
        }
        if not scored["guard"]["contract_match"]:
            fail(f"{tag}: contract_match false on CPU re-score", report)
        print(f"{tag}: size {row['size_bytes']} median {med:.2f}ms "
              f"p90 {row['cpu_latency']['p90_ms']:.2f}ms "
              f"EPE {epe:.7f} (ref {cfg['ref_epe']:.7f})", flush=True)
        rows[tag] = row

    # --- §1 R2 EPE gate (regeneration behaviour check) ---
    if abs(rows["R2"]["cpu_40scene"]["epe"] - ARTEFACTS["R2"]["ref_epe"]) > REPRO_TOL:
        fail("regenerated R2 EPE does not reproduce recorded "
             f"{ARTEFACTS['R2']['ref_epe']} within 1e-6", report)
    report["measured"]["r2_epe_gate"] = {"pass": True, "tol_px": REPRO_TOL}

    report["measured"]["rows"] = rows
    report["inferred"]["note"] = (
        "Latency rows describe ORT CPU EP inference on THIS host; they do not "
        "predict NPU/Hailo behaviour. EPE rows re-measure recorded references.")
    report["unknown"]["items"] = [
        "NPU compile/latency/partition (covered by the NPU part).",
        "H-CUDA latency (provider unavailable in host onnxruntime).",
    ]
    report["status"] = "COMPLETE: host part (regen + CPU latency + CPU re-scores)"
    report["utc_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    OUT.write_text(json.dumps(report, indent=2))
    print(f"COMPLETE wrote {OUT}")


if __name__ == "__main__":
    main()
