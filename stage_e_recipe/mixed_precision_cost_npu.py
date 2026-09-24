#!/usr/bin/env python
"""Mixed-precision cost, NPU part: VitisAI EP compile + latency + NPU EPE.

Implements docs/superpowers/specs/2026-09-24-mixed-precision-cost-design.md
sections 3-6 (NPU half). Characterization only: no thresholds.

Driver mode (no args): loops artefacts fp32/R0/R2, launching one worker
subprocess per artefact (25-min compile+measure cap each, 90-min total
budget), then merges partials into mixed_precision_cost_npu.json.
Worker mode (--worker TAG): builds the VitisAI session (stderr captured to
<tag>_npu_compile.log), runs §4 latency on the shared .npy inputs, scores
the 40-scene contract through the NPU session, writes npu_<tag>_partial.json.

Run with the NPU python (Ryzen AI 1.8.0 env) with
RYZEN_AI_INSTALLATION_PATH set. CWD must be the output dir so any
EP-emitted files land under it. Writes only under
stage_e_recipe/mixed_precision_cost/.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

WORK = HERE / "mixed_precision_cost"
OUT = WORK / "mixed_precision_cost_npu.json"
NPY_DIR = WORK / "inputs_npy"

TAGS = ("fp32", "R0", "R2")
N_SCENES = 10
N_REPEATS = 5
N_WARMUP = 10
PER_ARTEFACT_TIMEOUT_S = 25 * 60
TOTAL_BUDGET_S = 90 * 60
PARTITION_KEYWORDS = ("subgraph", "partition", "npu",
                      "vitisai", "xclbin", "fuse")


def artefact_paths() -> dict[str, Path]:
    return {
        "fp32": REPO / "stage_e_recipe/int8_e3/e3_seed0_best_fp32.onnx",
        "R0": REPO / "stage_e_recipe/int8_e3/e3_seed0_best_int8.onnx",
        "R2": WORK / "r2_regen" / "seed0_R2_head_refinement_fp32.onnx",
    }


def detect_npu_type() -> str:
    """Mirror quicktest.py: pnputil PCI enumeration -> PHX/HPT, STX, KRK."""
    try:
        cmd = r"pnputil /enum-devices /bus PCI /deviceids "
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout, _ = proc.communicate()
        text = stdout.decode(errors="replace")
    except Exception as e:
        return f"unknown (pnputil failed: {e})"
    npu_type = ""
    if "PCI\\VEN_1022&DEV_1502&REV_00" in text:
        npu_type = "PHX/HPT"
    for rev in ("DEV_17F0&REV_00", "DEV_17F0&REV_10",
                "DEV_17F0&REV_11", "DEV_17F0&REV_20"):
        if f"PCI\\VEN_1022&{rev}" in text:
            npu_type = "STX" if "REV_20" not in rev else "KRK"
    return npu_type if npu_type else "unknown (no matching PCI device id)"


def provider_options_for(npu_type: str) -> list[dict]:
    """Mirror quicktest.py: only PHX/HPT gets xclbin/target options."""
    import os as _os

    if npu_type == "PHX/HPT":
        install_dir = _os.environ.get("RYZEN_AI_INSTALLATION_PATH", "")
        xclbin = str(Path(install_dir) / "voe-4.0-win_amd64" / "xclbins"
                     / "phoenix" / "4x4.xclbin")
        return [{"target": "X1", "xlnx_enable_py3_round": 0, "xclbin": xclbin},
                {}]
    return [{}, {}]


def p90(xs: list[float]) -> float:
    s = sorted(xs)
    k = max(1, math.ceil(0.9 * len(s)))
    return s[k - 1]


def load_latency_inputs() -> list[tuple]:
    import numpy as np

    inputs = []
    for i in range(N_SCENES):
        l = np.load(NPY_DIR / f"scene{i}_left.npy")
        r = np.load(NPY_DIR / f"scene{i}_right.npy")
        assert l.shape == (1, 3, 368, 1232) and l.dtype == np.float32, \
            f"scene{i} left shape/dtype {l.shape}/{l.dtype}"
        assert r.shape == (1, 3, 368, 1232) and r.dtype == np.float32, \
            f"scene{i} right shape/dtype {r.shape}/{r.dtype}"
        inputs.append((l, r))
    return inputs


def parse_partition_lines(log_text: str) -> list[str]:
    hits = []
    for line in log_text.splitlines():
        low = line.lower()
        if any(k in low for k in PARTITION_KEYWORDS):
            hits.append(line.strip()[:500])
    return hits


def worker(tag: str) -> None:
    import numpy as np
    import onnxruntime as ort
    from phase1.harness.frozen_eval import pooled_metrics, refuse_unless_contract
    from src.datasets.kitti2015 import Kitti2015Stereo, normalize

    t_part_start = time.time()
    paths = artefact_paths()
    model = paths[tag]
    if not model.exists():
        print(f"WORKER {tag} FAILED: artefact missing: {model}", flush=True)
        raise SystemExit(2)
    inputs = load_latency_inputs()

    npu_type = detect_npu_type()
    prov_opts = provider_options_for(npu_type)
    providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]

    before = {p.name for p in WORK.iterdir()}
    compile_log = WORK / f"{tag}_npu_compile.log"

    opts = ort.SessionOptions()
    opts.log_severity_level = 1  # as quicktest.py
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # fd-level stderr capture around session creation (EP compile messages)
    err_fd = os.dup(2)
    log_fh = open(compile_log, "w", encoding="utf-8", errors="replace")
    t0 = time.time()
    try:
        os.dup2(log_fh.fileno(), 2)
        try:
            sess = ort.InferenceSession(
                str(model), sess_options=opts, providers=providers,
                provider_options=prov_opts)
        finally:
            os.dup2(err_fd, 2)
    except Exception as e:
        os.dup2(err_fd, 2)
        raise
    finally:
        log_fh.close()
        os.close(err_fd)
    compile_s = time.time() - t0
    log_text = compile_log.read_text(encoding="utf-8", errors="replace")
    after = {p.name for p in WORK.iterdir()}
    ep_files = sorted(after - before - {compile_log.name})

    ins = [i.name for i in sess.get_inputs()]
    assert len(ins) == 2, f"{tag}: expected 2 inputs, got {ins}"
    out_name = sess.get_outputs()[0].name

    for _ in range(N_WARMUP):
        sess.run([out_name], {ins[0]: inputs[0][0], ins[1]: inputs[0][1]})

    samples: list[float] = []
    per_scene_median: list[float] = []
    for i in range(N_SCENES):
        scene_samples = []
        for _ in range(N_REPEATS):
            s = time.perf_counter()
            sess.run([out_name], {ins[0]: inputs[i][0], ins[1]: inputs[i][1]})
            dt_ms = (time.perf_counter() - s) * 1e3
            samples.append(dt_ms)
            scene_samples.append(dt_ms)
        per_scene_median.append(float(np.median(scene_samples)))
    med = float(np.median(samples))

    # 40-scene contract through the NPU session (score_contract behaviour,
    # only the session differs; reused pooled_metrics/refuse_unless_contract).
    val = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                          disparity_scale=256.0, occluded=True)
    assert len(val) == 40, f"hailo_val has {len(val)} scenes, expected 40"
    preds, gts = [], []
    nonfinite = 0
    t1 = time.time()
    for i in range(len(val)):
        sc = val[i]
        out = sess.run([out_name], {ins[0]: normalize(sc.left),
                                    ins[1]: normalize(sc.right)})[0]
        pred = out[0, 0].astype(np.float64)
        nonfinite += int((~np.isfinite(pred)).sum())
        valid = sc.disparity > 0
        preds.append(pred[valid].astype(np.float64))
        gts.append(sc.disparity[valid].astype(np.float64))
    m = pooled_metrics(preds, gts)
    guard = refuse_unless_contract(len(val), int(sum(p.size for p in preds)),
                                   256.0, "hailo_val", "disp_occ_0")
    score_s = time.time() - t1

    part = {
        "tag": tag,
        "model": str(model.relative_to(REPO)).replace("\\", "/"),
        "size_bytes": model.stat().st_size,
        "npu_type_detected": npu_type,
        "providers_requested": providers,
        "provider_options": prov_opts,
        "RyzenAI_installation_path": os.environ.get(
            "RYZEN_AI_INSTALLATION_PATH", ""),
        "session_providers": sess.get_providers(),
        "compile_s": round(compile_s, 1),
        "compile_log": str(compile_log.relative_to(REPO)).replace("\\", "/"),
        "partition_lines": parse_partition_lines(log_text),
        "partition_parse_rule": f"lines containing any of {PARTITION_KEYWORDS}",
        "ep_emitted_files": ep_files,
        "latency": {
            "n_samples": len(samples),
            "median_ms": med,
            "p90_ms": float(p90(samples)),
            "min_ms": float(min(samples)),
            "max_ms": float(max(samples)),
            "fps_from_median": 1000.0 / med if med > 0 else float("nan"),
            "per_scene_median_ms": per_scene_median,
            "raw_samples_ms": samples,
        },
        "thread_settings": {
            "cpu_count": os.cpu_count(),
            "intra_op_num_threads": opts.intra_op_num_threads,
            "inter_op_num_threads": opts.inter_op_num_threads,
            "note": "defaults, no explicit values set",
        },
        "npu_40scene": {
            "epe": m["epe"], "d1": m["d1"], "rmse": m["rmse"],
            "valid_pixels": m["valid_pixels"],
            "nonfinite_pixels": nonfinite,
            "contract_match": guard["contract_match"],
            "score_s": round(score_s, 1),
        },
        "part_elapsed_s": round(time.time() - t_part_start, 1),
    }
    if not guard["contract_match"]:
        print(f"WORKER {tag} FAILED: contract_match false", flush=True)
        (WORK / f"npu_{tag}_partial.json").write_text(json.dumps(part, indent=2))
        raise SystemExit(3)
    (WORK / f"npu_{tag}_partial.json").write_text(json.dumps(part, indent=2))
    print(f"WORKER {tag} DONE median {med:.2f}ms EPE {m['epe']:.7f} "
          f"compile {compile_s:.1f}s", flush=True)


def driver() -> None:
    import onnxruntime as ort

    WORK.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "spec": "docs/superpowers/specs/2026-09-24-mixed-precision-cost-design.md",
        "part": "npu",
        "utc_start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "cpu_count": os.cpu_count(),
            "RYZEN_AI_INSTALLATION_PATH": os.environ.get(
                "RYZEN_AI_INSTALLATION_PATH", ""),
            "npu_type_detected": detect_npu_type(),
        },
        "budgets": {
            "per_artefact_timeout_s": PER_ARTEFACT_TIMEOUT_S,
            "total_budget_s": TOTAL_BUDGET_S,
        },
        "measured": {},
        "inferred": {},
        "unknown": {},
    }
    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        report["status"] = ("STOP: VitisAIExecutionProvider not in available "
                            f"providers {ort.get_available_providers()}")
        OUT.write_text(json.dumps(report, indent=2))
        print(report["status"], flush=True)
        raise SystemExit(1)

    paths = artefact_paths()
    missing = [t for t, p in paths.items() if not p.exists()]
    if missing:
        report["status"] = (f"STOP: missing artefacts {missing} "
                            "(run the host part first to regen R2)")
        OUT.write_text(json.dumps(report, indent=2))
        print(report["status"], flush=True)
        raise SystemExit(1)
    if not (NPY_DIR / "scene0_left.npy").exists():
        report["status"] = ("STOP: latency inputs missing "
                            "(run the host part first to precompute .npy)")
        OUT.write_text(json.dumps(report, indent=2))
        print(report["status"], flush=True)
        raise SystemExit(1)

    t_all = time.time()
    rows: dict[str, dict] = {}
    for tag in TAGS:
        elapsed = time.time() - t_all
        if elapsed > TOTAL_BUDGET_S:
            rows[tag] = {"failed": True,
                         "error": f"total 90-min budget exceeded ({elapsed:.0f}s); "
                                  "not attempted"}
            continue
        cmd = [sys.executable, str(HERE / "mixed_precision_cost_npu.py"),
               "--worker", tag]
        try:
            r = subprocess.run(cmd, cwd=str(WORK), capture_output=True,
                               text=True, timeout=PER_ARTEFACT_TIMEOUT_S)
        except subprocess.TimeoutExpired as e:
            rows[tag] = {"failed": True,
                         "error": f"timeout after {PER_ARTEFACT_TIMEOUT_S}s "
                                  f"(25-min per-artefact cap): {e}"}
            print(f"{tag}: TIMEOUT", flush=True)
            continue
        tail = (r.stderr or "")[-2000:] + (r.stdout or "")[-2000:]
        if r.returncode != 0:
            rows[tag] = {"failed": True,
                         "error": f"worker exit {r.returncode}: {tail}"}
            print(f"{tag}: worker exit {r.returncode}", flush=True)
            continue
        part_file = WORK / f"npu_{tag}_partial.json"
        if not part_file.exists():
            rows[tag] = {"failed": True,
                         "error": f"worker exit 0 but {part_file.name} missing: "
                                  f"{tail}"}
            continue
        rows[tag] = json.loads(part_file.read_text())
        rows[tag]["worker_tail"] = tail[-1000:]
        print(f"{tag}: OK", flush=True)

    report["measured"]["rows"] = rows
    report["measured"]["total_elapsed_s"] = round(time.time() - t_all, 1)
    report["inferred"]["note"] = (
        "Rows describe VitisAI EP behaviour on THIS machine/installation; "
        "CPU fallback of unsupported ops is recorded, not worked around.")
    report["unknown"]["items"] = [
        "Per-op NPU/CPU assignment beyond what the EP's compile log and "
        "emitted files reveal (not exposed via the ORT API).",
    ]
    failed = [t for t, r in rows.items() if r.get("failed")]
    report["status"] = ("COMPLETE: NPU part" if not failed
                        else f"PARTIAL: NPU part, failed artefacts {failed}")
    report["utc_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    OUT.write_text(json.dumps(report, indent=2))
    print(f"{report['status']} wrote {OUT}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        assert sys.argv[2] in TAGS
        worker(sys.argv[2])
    elif len(sys.argv) == 1:
        driver()
    else:
        raise SystemExit("usage: mixed_precision_cost_npu.py [--worker TAG]")
