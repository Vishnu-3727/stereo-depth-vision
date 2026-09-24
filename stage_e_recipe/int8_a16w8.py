import os
import sys
import time
import json
import ctypes
import datetime
import multiprocessing as mp
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import onnx
import onnxruntime as ort
import torch
import numpy as np

import int8_sensitivity as S
import int8_head_refinement as HR

WORK = HERE / "int8_a16w8"
WORK.mkdir(exist_ok=True, parents=True)
OUT = WORK / "int8_a16w8.json"
import gc

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

def get_free_gb():
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys / (1024**3)


SEEDS = {
    0: {"fp32_onnx": HERE / "int8_e3" / "e3_seed0_best_fp32.onnx",
        "fp32_sha": "7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645",
        "ref_fp32_epe": 1.18598534271453, "ref_int8_epe": 5.248941413313285},
    1: {"fp32_onnx": HERE / "int8_e3" / "e3_seed1_best_fp32.onnx",
        "fp32_sha": "9b4922c91367a2c41254795e0c7160dcb05313f7fb948e2c0d56f05d9ebe9ac2",
        "ref_fp32_epe": 1.178821279341308, "ref_int8_epe": 5.3849187722115985},
    2: {"fp32_onnx": HERE / "int8_e3" / "e3_seed2_best_fp32.onnx",
        "fp32_sha": "7c0c16e4cf66a7ca14e59f5be31801f19815ae1ba6ca72014cb7dd3e99e50418",
        "ref_fp32_epe": 1.1713166599983933, "ref_int8_epe": 5.87156272118187}
}
REPRO_TOL = 1e-6

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]

_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010

def get_child_peak_gb(pid):
    """Peak working set (GB) of pid via GetProcessMemoryInfo. None if unreadable."""
    try:
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        h = kernel32.OpenProcess(_PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, False, pid)
        if not h:
            return None
        try:
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            ok = psapi.GetProcessMemoryInfo(h, ctypes.byref(counters), counters.cb)
            if not ok:
                return None
            return counters.PeakWorkingSetSize / (1024**3)
        finally:
            kernel32.CloseHandle(h)
    except Exception:
        return None

def _child_quantize(src_path, dst_path, exclude, use_a16):
    import sys
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(HERE))
    import onnxruntime.quantization as ortq
    import int8_sensitivity as child_S
    from src.datasets.kitti2015 import Kitti2015Stereo
    
    orig_qs = ortq.quantize_static
    def patched_qs(*args, **kwargs):
        if use_a16:
            kwargs["activation_type"] = ortq.QuantType.QInt16
        else:
            kwargs["activation_type"] = ortq.QuantType.QInt8
        return orig_qs(*args, **kwargs)
    ortq.quantize_static = patched_qs
    child_S.CALIB_N = 32
    calib = Kitti2015Stereo(Path(REPO) / "data" / "kitti2015", split="hailo_calib")
    qrec = child_S.quantize_with_excludes(Path(src_path), Path(dst_path), calib, exclude)
    return qrec

def _child_target(src_path, dst_path, exclude, use_a16, q):
    try:
        qrec = _child_quantize(src_path, dst_path, exclude, use_a16)
        q.put({"ok": qrec})
    except Exception:
        q.put({"error": "CHILD_FAILED", "traceback": traceback.format_exc()})

def quantize_with_guard(src, dst, exclude, use_a16):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    proc = ctx.Process(target=_child_target, args=(str(src), str(dst), exclude, use_a16, q))
    proc.start()
    child_pid = proc.pid

    peak_gb = 0.0
    peak_seen = False
    min_free_gb = get_free_gb()
    while proc.is_alive():
        free_gb = get_free_gb()
        if free_gb < min_free_gb:
            min_free_gb = free_gb
        if child_pid is not None:
            child_peak = get_child_peak_gb(child_pid)
            if child_peak is not None:
                peak_seen = True
                if child_peak > peak_gb:
                    peak_gb = child_peak
        if free_gb < 0.5:
            proc.terminate()
            proc.join()
            return {"error": "MEMORY_GUARD_TRIPPED", "free_gb": free_gb,
                    "peak_child_mem_gb": peak_gb, "peak_child_mem": peak_gb,
                    "min_free_gb": min_free_gb, "peak_seen": peak_seen}
        time.sleep(2)

    # Final peak read (PeakWorkingSetSize is monotonic; one last sample
    # in case the child exited between polls). Process may already be
    # gone, in which case the last polled value stands.
    if child_pid is not None:
        try:
            child_peak = get_child_peak_gb(child_pid)
        except Exception:
            child_peak = None
        if child_peak is not None:
            peak_seen = True
            if child_peak > peak_gb:
                peak_gb = child_peak
    proc.join()
    if proc.exitcode != 0:
        try:
            msg = q.get_nowait()
        except Exception:
            msg = {}
        err = msg.get("traceback", f"exitcode={proc.exitcode}") if isinstance(msg, dict) else f"exitcode={proc.exitcode}"
        return {"error": "CHILD_FAILED", "traceback": err,
                "peak_child_mem_gb": peak_gb, "peak_child_mem": peak_gb,
                "min_free_gb": min_free_gb, "peak_seen": peak_seen}
    msg = q.get()
    if isinstance(msg, dict) and "ok" in msg:
        res = msg["ok"]
    else:
        return {"error": "CHILD_FAILED", "traceback": str(msg),
                "peak_child_mem_gb": peak_gb, "peak_child_mem": peak_gb,
                "min_free_gb": min_free_gb, "peak_seen": peak_seen}
    res["peak_child_mem_gb"] = peak_gb
    res["peak_child_mem"] = peak_gb
    res["min_free_gb"] = min_free_gb
    res["peak_seen"] = peak_seen
    return res

def fail(msg, report):
    report["status"] = f"STOP: {msg}"
    OUT.write_text(json.dumps(report, indent=2))
    print(report["status"], file=sys.stderr)
    sys.exit(1)

def main():
    report = {
        "provenance": {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_head": S.git_head(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "onnxruntime": ort.__version__
        },
        "thresholds": {
            "R_ACCEPT": 0.50,
            "N_SEEDS_REQUIRED": 3,
            "RESID_MAX_PX": 0.50,
            "note": "inherited from 6a6cf6b, frozen"
        },
        "measured": {},
        "inferred": {},
        "unknown": {},
        "status": "RUNNING"
    }
    
    from src.datasets.kitti2015 import Kitti2015Stereo
    val = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val")
    
    seeds_data = {}
    S1 = None
    
    for seed in (0, 1, 2):
        cfg = SEEDS[seed]
        fp32 = cfg["fp32_onnx"]
        if not fp32.exists():
            fail(f"Missing {fp32}", report)
        sha = S.sha256(fp32)
        if sha != cfg["fp32_sha"]:
            fail(f"sha mismatch on seed {seed} fp32: {sha}", report)
        model = onnx.load(str(fp32), load_external_data=False)
        n_nodes = len(model.graph.node)
        if n_nodes != 712:
            fail(f"n_nodes {n_nodes} != 712 on seed {seed} fp32", report)
        
        print(f"--- seed {seed} ---", flush=True)
        d = {}
        seeds_data[str(seed)] = d
        
        # fp32 arm
        t0_fp32 = time.time()
        scored = S.score_contract(fp32, val)
        scored["score_s"] = time.time() - t0_fp32
        d["fp32_scored"] = scored
        d["fp32_file_bytes"] = fp32.stat().st_size
        print(f"seed {seed} fp32: EPE {scored['metrics']['epe']:.4f}", flush=True)
        
        # R0 arm
        dst = WORK / f"seed{seed}_R0.onnx"
        t0 = time.time()
        gc.collect()
        qrec = quantize_with_guard(fp32, dst, None, use_a16=False)
        if "error" in qrec:
            fail(f"Memory guard tripped during R0: {qrec}", report)
        artifact_sha = S.sha256(dst)
        d["R0_file_bytes"] = dst.stat().st_size
        scored_r0 = S.score_contract(dst, val)
        scored_r0["score_s"] = time.time() - (t0 + qrec["quantize_s"])
        dst.unlink(missing_ok=True)
        d["R0_scored"] = scored_r0
        d["R0_quantize"] = {**qrec, "sha256": artifact_sha}
        
        print(f"seed {seed} R0: EPE {scored_r0['metrics']['epe']:.4f}", flush=True)
        
        if S1 is None:
            S1 = time.time() - t0_fp32
            T_est = 9 * S1
            print(f"Budget check: S1 = {S1:.1f}s -> T_est = {T_est:.1f}s", flush=True)
            report["measured"]["S1_s"] = S1
            report["measured"]["T_est_s"] = T_est
            if T_est > 7200:
                report["status"] = "STOP: budget exceeded"
                OUT.write_text(json.dumps(report, indent=2))
                fail("Budget exceeded", report)
                
    # Gate check
    gate_ok_all = True
    gate_rows = {}
    for seed in (0, 1, 2):
        cfg = SEEDS[seed]
        d = seeds_data[str(seed)]
        e_f = d["fp32_scored"]["metrics"]["epe"]
        e_0 = d["R0_scored"]["metrics"]["epe"]
        ok_f = abs(e_f - cfg["ref_fp32_epe"]) <= REPRO_TOL
        ok_0 = abs(e_0 - cfg["ref_int8_epe"]) <= REPRO_TOL
        ok_g = (d["fp32_scored"]["guard"]["contract_match"] is True and 
                d["R0_scored"]["guard"]["contract_match"] is True)
        gate_rows[str(seed)] = {"fp32_ok": ok_f, "R0_ok": ok_0, "guards_ok": ok_g}
        if not (ok_f and ok_0 and ok_g):
            gate_ok_all = False
            
    report["measured"]["baseline_gate_per_seed"] = gate_rows
    if not gate_ok_all:
        report["measured"]["seeds_baselines"] = seeds_data
        fail("Baseline reproduction gate failed", report)
        
    for seed in (0, 1, 2):
        print(f"--- seed {seed} A16 ---", flush=True)
        cfg = SEEDS[seed]
        fp32 = cfg["fp32_onnx"]
        d = seeds_data[str(seed)]
        
        # A16 arm
        dst = WORK / f"seed{seed}_A16.onnx"
        gc.collect()
        qrec = quantize_with_guard(fp32, dst, None, use_a16=True)
        if "error" in qrec:
            fail(f"Memory guard tripped during A16: {qrec}", report)
        artifact_sha = S.sha256(dst)
        d["A16_file_bytes"] = dst.stat().st_size
        scored_a16 = S.score_contract(dst, val)
        dst.unlink(missing_ok=True)
        d["A16_scored"] = scored_a16
        d["A16_quantize"] = {**qrec, "sha256": artifact_sha}
        print(f"seed {seed} A16: EPE {scored_a16['metrics']['epe']:.4f}", flush=True)

    th = report["thresholds"]
    r_acc, n_req, resid_max = th["R_ACCEPT"], th["N_SEEDS_REQUIRED"], th["RESID_MAX_PX"]
    rows = []
    for seed in (0, 1, 2):
        d = seeds_data[str(seed)]
        e_f = d["fp32_scored"]["metrics"]["epe"]
        e_0 = d["R0_scored"]["metrics"]["epe"]
        e_a16 = d["A16_scored"]["metrics"]["epe"]
        gap = e_0 - e_f
        r_a16 = (e_0 - e_a16) / gap if gap else None
        resid_a16 = e_a16 - e_f
        
        row = {
            "seed": seed,
            "e_fp32": e_f,
            "e_R0": e_0,
            "e_A16": e_a16,
            "G_seed": gap,
            "R_A16": r_a16,
            "resid_A16": resid_a16,
            "size_fp32": d["fp32_file_bytes"],
            "size_R0": d["R0_file_bytes"],
            "size_A16": d["A16_file_bytes"],
            "d1_A16": d["A16_scored"]["metrics"]["d1"],
            "nonfinite_A16": d["A16_scored"]["nonfinite_pixels"],
            "contract_match": {
                "fp32": d["fp32_scored"]["guard"]["contract_match"],
                "R0": d["R0_scored"]["guard"]["contract_match"],
                "A16": d["A16_scored"]["guard"]["contract_match"]
            },
            "metrics_A16": d["A16_scored"]["metrics"],
            "A16_quantize": d["A16_quantize"],
            "A16_artifact_sha256": d["A16_quantize"]["sha256"],
        }
        row["pass_A16_R"] = r_a16 is not None and r_a16 >= r_acc
        row["pass_A16_resid"] = resid_a16 <= resid_max
        row["pass_A16_seed"] = bool(row["pass_A16_R"] and row["pass_A16_resid"])
        rows.append(row)
        print(f"seed {seed}: A16 EPE {e_a16:.4f} R_A16={r_a16:.3f} resid={resid_a16:+.4f}", flush=True)

    n_pass = sum(1 for r in rows if r["pass_A16_seed"])
    report["measured"]["rows"] = rows
    report["measured"]["n_pass_A16"] = n_pass
    if n_pass >= n_req:
        verdict = f"SUPPORTED: R_A16 >= {r_acc} and residual <= {resid_max}px on {n_pass}/3 seeds (required {n_req})"
    else:
        failing = [r["seed"] for r in rows if not r["pass_A16_seed"]]
        verdict = f"FALSIFIED: A16W8 whole graph does not meet the criterion on all 3 seeds (seeds {failing} failed)"
    report["measured"]["verdict"] = verdict
    report["status"] = "COMPLETE"
    OUT.write_text(json.dumps(report, indent=2))
    print(f"COMPLETE verdict={verdict}")

if __name__ == "__main__":
    main()
