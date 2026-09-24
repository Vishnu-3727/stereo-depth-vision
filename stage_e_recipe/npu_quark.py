#!/usr/bin/env python
"""Quark INT8 on the Ryzen AI NPU: quantize (Quark) + CPU reference + NPU cost.

Implements docs/superpowers/specs/2026-09-24-npu-quark-int8-design.md
sections 1-8 (all arms). Characterization only: no thresholds.

Arms (all from the same fp32 ONNX + same 32-scene hailo_calib calibration):
  Q1 = Quark XINT8 (default config), Q2 = Quark A16W8 (if exposed),
  Q3 = XINT8 with /regression/* + /refinement/* excluded (kept fp32).

Driver mode (no args): sha-assert subject; enumerate node sets; then run arms
strictly one at a time, each in its own child process (--arm-full) under a
watchdog that polls free RAM every 2 s and kills the child below 2.0 GB free
(2 h total box from first quantization attempt; Q1 first, Q2/Q3 only if Q1
finishes within the guard). Attempt-3 deviations (Amendment A1): 8 calib
scenes, CalibrationMethod.MinMax, use_external_data_format=True,
include_cle=False, XINT8 power-of-two scale type kept.

Run with the clone python (ryzen-ai-1.8.0-quark env, amd-quark installed).
CWD must be the output dir so EP-emitted files land under it. Writes only
under stage_e_recipe/npu_quark/. Reuses inputs_npy/ from mixed_precision_cost/
and existing scoring/node-set code by import. Modifies no existing file.
"""

from __future__ import annotations

import copy
import datetime
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

WORK = HERE / "npu_quark"
OUT = WORK / "npu_quark.json"
NPY_DIR = HERE / "mixed_precision_cost" / "inputs_npy"

SPEC = "docs/superpowers/specs/2026-09-24-npu-quark-int8-design.md"
PROTOCOL_COMMIT = "6397508"

FP32_PATH = "stage_e_recipe/int8_e3/e3_seed0_best_fp32.onnx"
FP32_SHA = "7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645"
FP32_BYTES = 1701550

ARMS = ("Q1", "Q2", "Q3")
ARM_CONFIG = {"Q1": "XINT8", "Q2": "A16W8", "Q3": "XINT8"}

N_SCENES = 10
N_REPEATS = 5
N_WARMUP = 10
PER_ARTEFACT_TIMEOUT_S = 25 * 60
TOTAL_BUDGET_S = 2 * 3600
# Attempt-3 (Amendment A1) low-memory settings.
CALIB_N_A3 = 8  # first 8 hailo_calib scenes (cut from 32)
MEM_GUARD_FREE_GB = 2.0  # kill child if free physical RAM drops below this
MEM_POLL_S = 2.0  # watchdog poll interval
PARTITION_KEYWORDS = ("subgraph", "partition", "npu",
                      "vitisai", "xclbin", "fuse")
KERNEL_RULE = ("distinct alphanumeric tokens containing int8/int16/uint8 "
               "(INT8-family) vs bf16 (Bf16-family) vs fp16/fp32 (other), "
               "case-insensitive; counts are distinct-token counts with "
               "total substring hits alongside")

REQUIRED_HEAD = ["/regression/Mul", "/regression/Mul_1", "/regression/Mul_2"]
REQUIRED_REFIN = ["/refinement/input_conv/Conv",
                  "/refinement/output_conv/Conv"]


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def p90(xs: list[float]) -> float:
    s = sorted(xs)
    k = max(1, math.ceil(0.9 * len(s)))
    return s[k - 1]


def prefix_names(fp32_path: Path, prefix: str) -> list[str]:
    import onnx

    m = onnx.load(str(fp32_path))
    return sorted(n.name for n in m.graph.node if n.name.startswith(prefix))


def count_kernels(log_text: str) -> dict:
    toks = re.findall(
        r"[A-Za-z0-9_.\-/]*?(?:int8|int16|uint8|bf16|fp16|fp32)[A-Za-z0-9_]*",
        log_text, flags=re.IGNORECASE)
    fams: dict[str, set[str]] = {"int8_family": set(), "bf16_family": set(),
                                 "other_float": set()}
    for t in toks:
        low = t.lower()
        if "int8" in low or "int16" in low or "uint8" in low:
            fams["int8_family"].add(t)
        elif "bf16" in low:
            fams["bf16_family"].add(t)
        else:
            fams["other_float"].add(t)
    return {
        "rule": KERNEL_RULE,
        "int8_family": {"distinct": len(fams["int8_family"]),
                        "examples": sorted(fams["int8_family"])[:20]},
        "bf16_family": {"distinct": len(fams["bf16_family"]),
                        "examples": sorted(fams["bf16_family"])[:20]},
        "other_float": {"distinct": len(fams["other_float"]),
                        "examples": sorted(fams["other_float"])[:20]},
        "total_token_hits": len(toks),
    }


def parse_partition_lines(log_text: str) -> list[str]:
    hits = []
    for line in log_text.splitlines():
        low = line.lower()
        if "vitis ai ep" in low or any(k in low for k in PARTITION_KEYWORDS):
            hits.append(line.strip()[:500])
    return hits


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
        assert l.flags["C_CONTIGUOUS"] and r.flags["C_CONTIGUOUS"]
        inputs.append((l, r))
    return inputs


def quark_reader(calib_ds, count: int):
    from onnxruntime.quantization import CalibrationDataReader

    import onnxruntime as ort

    sess = ort.InferenceSession(str(REPO / FP32_PATH),
                                providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    assert len(names) == 2, f"fp32 inputs: {names}"

    from src.datasets.kitti2015 import normalize

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = []
            for i in range(min(count, len(calib_ds))):
                s = calib_ds[i]
                self.items.append({names[0]: normalize(s.left),
                                   names[1]: normalize(s.right)})
            self.it = iter(self.items)

        def get_next(self):
            return next(self.it, None)

    return Reader()


def quantize_arm(fp32: Path, arm: str, combined: list[str],
                 calib_ds, count: int) -> dict:
    from quark.onnx import ModelQuantizer, QConfig
    from onnxruntime.quantization.calibrate import CalibrationMethod

    cfg_name = ARM_CONFIG[arm]
    try:
        base = QConfig.get_default_config(cfg_name)
    except Exception as e:
        raise RuntimeError(
            f"Quark default config {cfg_name!r} not exposed: {e}") from e
    g = base.global_quant_config
    cfg_desc = {
        "name": cfg_name,
        "calibrate_method": str(g.calibrate_method),
        "activation_type": str(g.activation_type),
        "weight_type": str(g.weight_type),
        "quant_format": str(g.quant_format),
        "enable_npu_cnn": bool(getattr(g, "enable_npu_cnn", False)),
        "extra_options": dict(getattr(g, "extra_options", {}) or {}),
    }
    # ATTEMPT-3 (Amendment A1, 2026-09-24) deviations from spec §2:
    # - include_cle=False (carried from attempt 2): default flow crashes in
    #   the CLE pre-pass (cle/equalization.py _cross_layer_equalize
    #   tail_w_data.transpose(1, 0, 2, 3) assumes 4-D conv weights, but 5 of
    #   the 51 Convs are Conv3d with 5-D weights -> ValueError: axes don't
    #   match array; attempt 1 failed all 3 arms in 18.5 s).
    # - calibrate_method=CalibrationMethod.MinMax (explicit): replaces the
    #   XINT8 default PowerOfTwoMethod.MinMSE. Installed amd-quark==0.11.2 /
    #   ORT==1.27.0 expose CalibrationMethod.{MinMax, Entropy, Percentile,
    #   Distribution} and PowerOfTwoMethod.{NonOverflow, MinMSE}; no separate
    #   "running-range" name exists — MinMax IS the min/max running-range
    #   method. XINT8 power-of-two scale type is kept (QUInt8/QInt8,
    #   ActivationSymmetric, QDQ).
    # - use_external_data_format=True: installed Quark 0.11.2 exposes NO
    #   dedicated per-batch/streaming/low-memory calibration flag
    #   (inspected all QuantizationConfig fields); this is the only
    #   memory-relevant option — quantized tensors go to an external data
    #   file instead of being held inline. Reader already yields one scene
    #   at a time (batch 1).
    # - calibration scenes: first 8 of hailo_calib (cut from 32).
    cfg = copy.deepcopy(base)
    cfg.global_quant_config.include_cle = False
    cfg.global_quant_config.calibrate_method = CalibrationMethod.MinMax
    cfg.global_quant_config.use_external_data_format = True
    cfg_desc["include_cle"] = False
    cfg_desc["calibrate_method_override"] = "CalibrationMethod.MinMax"
    cfg_desc["use_external_data_format"] = True
    cfg_desc["calib_scenes"] = int(count)
    cfg_desc["deviation"] = ("Amendment A1: include_cle=False (CLE crashes "
                             "on Conv3d 5-D weights); MinMax calibration "
                             "(no running-range name exists in Quark "
                             "0.11.2); 8 calib scenes; external-data format "
                             "(no streaming flag exists); XINT8 power-of-two "
                             "scale type kept")
    if arm == "Q3":
        cfg.global_quant_config.nodes_to_exclude = list(combined)
        cfg_desc["nodes_to_exclude_n"] = len(combined)
    else:
        cfg_desc["nodes_to_exclude_n"] = 0
    dst = WORK / f"{arm}_{cfg_name.lower()}.onnx"
    quantizer = ModelQuantizer(cfg)
    reader = quark_reader(calib_ds, count)
    t0 = time.time()
    quantizer.quantize_model(str(fp32), str(dst), reader)
    quantize_s = time.time() - t0

    import int8_sensitivity as S

    # With use_external_data_format=True Quark may emit sidecar data files
    # next to the .onnx; record every sibling sharing the stem.
    sidecars = []
    total_bytes = 0
    for p in WORK.glob(dst.stem + "*"):
        if p.is_file() and p.suffix != ".log" and p != dst and \
                p.name.startswith(dst.stem):
            sidecars.append({"name": p.name, "bytes": p.stat().st_size})
            total_bytes += p.stat().st_size
    onnx_bytes = dst.stat().st_size if dst.exists() else 0
    return {"config": cfg_desc, "quantize_s": round(quantize_s, 1),
            "sha256": S.sha256(dst), "bytes": onnx_bytes,
            "sidecar_files": sidecars,
            "total_bytes_incl_sidecars": onnx_bytes + total_bytes,
            "path": str(dst.relative_to(REPO)).replace("\\", "/")}


def _write_stage(arm: str, stage: str) -> None:
    try:
        (WORK / f"attempt3_{arm}_stage.txt").write_text(stage)
    except Exception:
        pass


def worker(arm: str, model_rel: str) -> None:
    import numpy as np
    import onnxruntime as ort
    from phase1.harness.frozen_eval import pooled_metrics, refuse_unless_contract
    from src.datasets.kitti2015 import Kitti2015Stereo, normalize

    t_part_start = time.time()
    model = REPO / model_rel
    if not model.exists():
        print(f"WORKER {arm} FAILED: artefact missing: {model}", flush=True)
        raise SystemExit(2)
    inputs = load_latency_inputs()

    providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]
    cache_dir = WORK / f"cache_{arm.lower()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prov_opts = [{"cacheDir": str(cache_dir),
                  "cacheKey": f"npu-quark-{arm.lower()}"}, {}]

    before = {p.name for p in WORK.iterdir()}
    compile_log = WORK / f"{arm}_npu_compile.log"

    opts = ort.SessionOptions()
    opts.log_severity_level = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # fd-level stdout+stderr capture (VitisAI partition lines go to stdout)
    saved_out, saved_err = os.dup(1), os.dup(2)
    log_fh = open(compile_log, "w", encoding="utf-8", errors="replace")
    t0 = time.time()
    try:
        os.dup2(log_fh.fileno(), 1)
        os.dup2(log_fh.fileno(), 2)
        try:
            sess = ort.InferenceSession(
                str(model), sess_options=opts, providers=providers,
                provider_options=prov_opts)
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
    except Exception:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        raise
    finally:
        log_fh.close()
        os.close(saved_out)
        os.close(saved_err)
    compile_s = time.time() - t0
    log_text = compile_log.read_text(encoding="utf-8", errors="replace")
    after = {p.name for p in WORK.iterdir()}
    ep_files = sorted(after - before - {compile_log.name})

    ins = [i.name for i in sess.get_inputs()]
    assert len(ins) == 2, f"{arm}: expected 2 inputs, got {ins}"
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
        "arm": arm,
        "model": model_rel,
        "size_bytes": model.stat().st_size,
        "providers_requested": providers,
        "provider_options": [{"cacheDir": str(cache_dir.relative_to(REPO))
                              .replace("\\", "/"),
                              "cacheKey": f"npu-quark-{arm.lower()}"}, {}],
        "RyzenAI_installation_path": os.environ.get(
            "RYZEN_AI_INSTALLATION_PATH", ""),
        "session_providers": sess.get_providers(),
        "compile_s": round(compile_s, 1),
        "compile_log": str(compile_log.relative_to(REPO)).replace("\\", "/"),
        "partition_lines": parse_partition_lines(log_text),
        "partition_parse_rule": ("lines containing '[Vitis AI EP]' or any of "
                                 f"{PARTITION_KEYWORDS}"),
        "kernel_counts": count_kernels(log_text),
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
        print(f"WORKER {arm} FAILED: contract_match false", flush=True)
        (WORK / f"npu_{arm}_partial.json").write_text(json.dumps(part, indent=2))
        raise SystemExit(3)
    (WORK / f"npu_{arm}_partial.json").write_text(json.dumps(part, indent=2))
    print(f"WORKER {arm} DONE median {med:.2f}ms EPE {m['epe']:.7f} "
          f"compile {compile_s:.1f}s", flush=True)


def aianalyzer_probe() -> dict:
    info: dict = {}
    try:
        import importlib.metadata as md

        info["version"] = md.version("aianalyzer")
    except Exception as e:
        info["version"] = f"unknown ({e})"
    try:
        r = subprocess.run([sys.executable, "-m", "aianalyzer", "--help"],
                           capture_output=True, text=True, timeout=60)
        info["help_exit"] = r.returncode
        info["help_head"] = ((r.stdout or "") + (r.stderr or ""))[:1500]
    except Exception as e:
        info["help_exit"] = f"probe failed: {e}"
        info["help_head"] = ""
    return info


def full_arm(arm: str) -> None:
    """Attempt-3 child: quantize + Q-CPU score + NPU (one arm, one process).

    Reads the node-exclusion list from WORK/attempt3_combined.json (written
    by the parent driver). Writes WORK/attempt3_<ARM>_partial.json and exits:
    0 = partial written (NPU failure is recorded data, not a crash);
    2/3 = quantize / CPU-score crash (partial still written).
    The parent watchdog enforces the memory guard and the time boxes.
    """
    import int8_sensitivity as S
    from src.datasets.kitti2015 import Kitti2015Stereo

    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(str(WORK))  # Quark/EP side files land under the output dir
    fp32 = REPO / FP32_PATH
    combined = json.loads(
        (WORK / "attempt3_combined.json").read_text())
    calib = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_calib")
    val = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                          disparity_scale=256.0, occluded=True)
    assert len(val) == 40, f"hailo_val has {len(val)} scenes, expected 40"
    row: dict = {"arm": arm}

    def dump():
        (WORK / f"attempt3_{arm}_partial.json").write_text(
            json.dumps(row, indent=2))

    _write_stage(arm, "quantize")
    try:
        qrec = quantize_arm(fp32, arm, combined, calib, CALIB_N_A3)
    except Exception as e:
        row["quantize"] = {"failed": True,
                           "error": f"{type(e).__name__}: {e}"}
        dump()
        print(f"FULL {arm}: QUANTIZE FAILED: {type(e).__name__}: {e}",
              flush=True)
        raise SystemExit(2)
    row["quantize"] = qrec
    dump()
    print(f"FULL {arm}: quantized {qrec['bytes']} bytes "
          f"(+{qrec['total_bytes_incl_sidecars'] - qrec['bytes']} sidecar) sha "
          f"{qrec['sha256'][:12]} in {qrec['quantize_s']}s", flush=True)

    _write_stage(arm, "cpu_score")
    try:
        scored = S.score_contract(REPO / qrec["path"], val)
    except Exception as e:
        row["cpu_40scene"] = {"failed": True,
                              "error": f"{type(e).__name__}: {e}"}
        dump()
        print(f"FULL {arm}: CPU SCORE FAILED: {type(e).__name__}: {e}",
              flush=True)
        raise SystemExit(3)
    row["cpu_40scene"] = {
        "epe": scored["metrics"]["epe"], "d1": scored["metrics"]["d1"],
        "rmse": scored["metrics"]["rmse"],
        "valid_pixels": scored["metrics"]["valid_pixels"],
        "nonfinite_pixels": scored["nonfinite_pixels"],
        "contract_match": scored["guard"]["contract_match"],
        "score_s": scored["score_s"],
    }
    dump()
    print(f"FULL {arm}: CPU EPE {scored['metrics']['epe']:.7f} "
          f"contract={scored['guard']['contract_match']}", flush=True)

    _write_stage(arm, "npu_compile")
    cmd = [sys.executable, str(HERE / "npu_quark.py"),
           "--worker", arm, qrec["path"]]
    try:
        r = subprocess.run(cmd, cwd=str(WORK), capture_output=True,
                           text=True, timeout=PER_ARTEFACT_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        row["npu"] = {"failed": True,
                      "error": f"timeout after {PER_ARTEFACT_TIMEOUT_S}s "
                               f"(25-min per-arm NPU cap): {e}"}
        dump()
        print(f"FULL {arm}: NPU TIMEOUT", flush=True)
        return
    _write_stage(arm, "npu_done")
    tail = (r.stderr or "")[-2000:] + (r.stdout or "")[-2000:]
    if r.returncode != 0:
        row["npu"] = {"failed": True,
                      "error": f"worker exit {r.returncode}: {tail}"}
        dump()
        print(f"FULL {arm}: NPU worker exit {r.returncode}", flush=True)
        return
    part_file = WORK / f"npu_{arm}_partial.json"
    if not part_file.exists():
        row["npu"] = {"failed": True,
                      "error": f"worker exit 0 but {part_file.name} "
                               f"missing: {tail}"}
        dump()
        return
    row["npu"] = json.loads(part_file.read_text())
    row["npu"]["worker_tail"] = tail[-1000:]
    npu_epe = row["npu"]["npu_40scene"]["epe"]
    cpu_epe = row["cpu_40scene"]["epe"]
    row["npu"]["delta_vs_own_cpu"] = npu_epe - cpu_epe
    dump()
    print(f"FULL {arm}: NPU OK EPE {npu_epe:.7f} delta {npu_epe - cpu_epe:+.7f}",
          flush=True)


def _tree_rss_gb(proc) -> float:
    try:
        total = proc.memory_info().rss
        for c in proc.children(recursive=True):
            try:
                total += c.memory_info().rss
            except Exception:
                pass
        return total / 1e9
    except Exception:
        return 0.0


def _kill_tree(proc) -> None:
    try:
        for c in proc.children(recursive=True):
            try:
                c.kill()
            except Exception:
                pass
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _free_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / 1e9


def run_arm_guarded(arm: str, timeout_s: float) -> dict:
    """Run one arm in a child process under the memory-guard watchdog.

    Polls free physical RAM every MEM_POLL_S seconds; kills the whole child
    tree if free RAM drops below MEM_GUARD_FREE_GB. Returns the merged row
    dict including peak_rss_gb / min_free_gb / guard fields.
    """
    import psutil

    stage_file = WORK / f"attempt3_{arm}_stage.txt"
    try:
        stage_file.unlink()
    except Exception:
        pass
    part_file = WORK / f"attempt3_{arm}_partial.json"
    try:
        part_file.unlink()
    except Exception:
        pass
    log_path = WORK / f"attempt3_{arm}_driver.log"
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
    cmd = [sys.executable, str(HERE / "npu_quark.py"), "--arm-full", arm]
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=log_fh,
                            stderr=subprocess.STDOUT)
    p = psutil.Process(proc.pid)
    peak_rss = 0.0
    min_free = _free_gb()
    aborted = None
    while proc.poll() is None:
        time.sleep(MEM_POLL_S)
        try:
            free = _free_gb()
        except Exception:
            free = min_free
        min_free = min(min_free, free)
        rss = _tree_rss_gb(p)
        peak_rss = max(peak_rss, rss)
        elapsed = time.time() - t0
        if free < MEM_GUARD_FREE_GB:
            stage = "unknown"
            try:
                stage = stage_file.read_text().strip() or "unknown"
            except Exception:
                pass
            _kill_tree(p)
            try:
                proc.wait(timeout=60)
            except Exception:
                pass
            aborted = (f"aborted: memory guard (free < 2 GB), peak child "
                       f"RSS {peak_rss:.2f} GB at stage {stage}")
            break
        if elapsed > timeout_s:
            _kill_tree(p)
            try:
                proc.wait(timeout=60)
            except Exception:
                pass
            aborted = (f"aborted: arm timeout after {timeout_s:.0f}s "
                       f"(peak child RSS {peak_rss:.2f} GB)")
            break
    try:
        log_fh.close()
    except Exception:
        pass
    wall_s = round(time.time() - t0, 1)
    try:
        log_tail = log_path.read_text(
            encoding="utf-8", errors="replace")[-2000:]
    except Exception:
        log_tail = ""
    row: dict = {"arm": arm, "wall_s": wall_s,
                 "peak_rss_gb": round(peak_rss, 2),
                 "min_free_gb": round(min_free, 2),
                 "memory_guard_gb": MEM_GUARD_FREE_GB}
    if aborted is not None:
        row["memory_guard_abort"] = aborted
        print(f"{arm}: {aborted} (wall {wall_s}s)", flush=True)
        if part_file.exists():
            try:
                row.update(json.loads(part_file.read_text()))
                row["memory_guard_abort"] = aborted
            except Exception:
                pass
        row["child_log_tail"] = log_tail[-1000:]
        return row
    row["child_exit"] = proc.returncode
    if part_file.exists():
        try:
            row.update(json.loads(part_file.read_text()))
        except Exception as e:
            row["partial_read_error"] = f"{type(e).__name__}: {e}"
    else:
        row["no_partial"] = (f"child exit {proc.returncode} without writing "
                             f"{part_file.name}")
    row["child_log_tail"] = log_tail[-1000:]
    print(f"{arm}: child exit {proc.returncode} wall {wall_s}s "
          f"peakRSS {peak_rss:.2f}GB minFree {min_free:.2f}GB", flush=True)
    return row


def driver() -> None:
    import onnxruntime as ort

    import int8_sensitivity as S
    import int8_regression_head as RH
    from src.datasets.kitti2015 import Kitti2015Stereo

    WORK.mkdir(parents=True, exist_ok=True)
    try:
        import quark

        quark_v = getattr(quark, "__version__", "unknown")
    except Exception as e:
        quark_v = f"import failed: {e}"

    def git_time(rev: str) -> str:
        try:
            r = subprocess.run(["git", "show", "-s", "--format=%ci", rev],
                               cwd=REPO, capture_output=True, text=True,
                               timeout=15)
            return r.stdout.strip()
        except Exception:
            return "unknown"

    report: dict = {
        "spec": SPEC,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_commit_time": git_time(PROTOCOL_COMMIT),
        "utc_start": utcnow(),
        "git_head": S.git_head(),
        "environment": {
            "python": sys.version.split()[0],
            "quark": quark_v,
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "cpu_count": os.cpu_count(),
            "RYZEN_AI_INSTALLATION_PATH": os.environ.get(
                "RYZEN_AI_INSTALLATION_PATH", ""),
        },
        "aianalyzer": aianalyzer_probe(),
        "amendment_commit": "497b55c",
        "amendment": "Amendment A1 (2026-09-24): include_cle=False; "
                     "CalibrationMethod.MinMax (no running-range name exists "
                     "in Quark 0.11.2 — MinMax IS the min/max running-range "
                     "method); first 8 hailo_calib scenes batch-1; "
                     "use_external_data_format=True (no streaming flag "
                     "exists); one child process per arm, Q1 first, Q2/Q3 "
                     "only if Q1 finishes within the 2 GB memory guard; "
                     "XINT8 power-of-two scale type kept for Q1",
        "prior_attempts": {
            "attempt1": ("2026-09-24T07:00:52Z, 18.5 s: all 3 arms "
                         "ValueError: axes don't match array in the default "
                         "CLE pre-pass (5 Conv3d 5-D weights)"),
            "attempt2": ("include_cle=False reached calibration "
                         "(quantized_info.csv: pre process 2.26 s, collect "
                         "data 208.52 s), then committed memory ~10.5 GB, "
                         "free RAM 0 on the 15 GB laptop; killed by the "
                         "manager on user instruction; no model produced"),
        },
        "deviation": {
            "spec_text": "spec §2 said default configs only (no AdaRound / "
                         "AdaQuant / CLE / fast-finetune), 32 calib scenes",
            "what_changed": ("include_cle=False; "
                             "calibrate_method=CalibrationMethod.MinMax "
                             "(XINT8 default was PowerOfTwoMethod.MinMSE); "
                             "use_external_data_format=True; "
                             "calib scenes 32 -> first 8; one guarded child "
                             "process per arm"),
            "why": ("attempt 1: untouched defaults failed all 3 arms "
                    "(ValueError: axes don't match array, CLE 4-D "
                    "assumption vs 5 Conv3d); attempt 2: 32-scene "
                    "calibration exhausted RAM (10.5 GB committed, free 0)"),
            "effect": ("CLE is an accuracy pre-pass; MinMax keeps XINT8 "
                       "power-of-two scales (QUInt8/QInt8, symmetric) with "
                       "a different threshold rule; 8 scenes + external "
                       "data + guard keep the run inside 15 GB RAM"),
        },
        "budgets": {
            "total_box_s": TOTAL_BUDGET_S,
            "per_arm_npu_timeout_s": PER_ARTEFACT_TIMEOUT_S,
        },
        "measured": {},
        "inferred": {},
        "unknown": {},
    }

    # --- §1 subject sha assert ---
    fp32 = REPO / FP32_PATH
    if not fp32.exists():
        report["status"] = f"STOP: subject missing: {fp32}"
        OUT.write_text(json.dumps(report, indent=2))
        raise SystemExit(1)
    got, nbytes = S.sha256(fp32), fp32.stat().st_size
    if got != FP32_SHA or nbytes != FP32_BYTES:
        report["status"] = (f"STOP: subject mismatch sha={got} bytes={nbytes}")
        OUT.write_text(json.dumps(report, indent=2))
        raise SystemExit(1)
    print(f"subject OK: sha match, {nbytes} bytes", flush=True)
    report["measured"]["subject_assert"] = {"sha256": got, "bytes": nbytes,
                                            "match": True}

    # --- node sets (§0 precedent) ---
    head, n_nodes = RH.kept_set(fp32)
    head_rt = prefix_names(fp32, "/regression/")
    assert head == head_rt, "head set differs from runtime prefix enumeration"
    refin = prefix_names(fp32, "/refinement/")
    for m in REQUIRED_HEAD:
        assert m in head, f"required head node {m} missing"
    for m in REQUIRED_REFIN:
        assert m in refin, f"required refinement node {m} missing"
    combined = sorted(set(head) | set(refin))
    print(f"node sets OK: {n_nodes} nodes, {len(head)} /regression/*, "
          f"{len(refin)} /refinement/*, {len(combined)} combined", flush=True)
    report["measured"]["node_sets"] = {
        "n_nodes": n_nodes, "n_head": len(head), "n_refinement": len(refin),
        "n_combined": len(combined), "head": head, "refinement": refin,
    }

    if not (NPY_DIR / "scene0_left.npy").exists():
        report["status"] = "STOP: latency inputs missing (run host part first)"
        OUT.write_text(json.dumps(report, indent=2))
        raise SystemExit(1)

    val = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                          disparity_scale=256.0, occluded=True)
    assert len(val) == 40, f"hailo_val has {len(val)} scenes, expected 40"

    # --- arms strictly one at a time, each in its own child process ---
    # 2 h box from first quantization attempt (attempt-3 start).
    t_box0 = time.time()
    report["measured"]["first_quantize_utc"] = utcnow()
    (WORK / "attempt3_combined.json").write_text(json.dumps(combined))
    rows: dict[str, dict] = {}
    # Q2/Q3 run only if Q1 finishes within the memory guard.
    rows["Q1"] = run_arm_guarded(
        "Q1", min(TOTAL_BUDGET_S - (time.time() - t_box0), 5400))
    q1_guard = rows["Q1"].get("memory_guard_abort")
    q1_no_partial = rows["Q1"].get("no_partial")
    if q1_guard is None and q1_no_partial is None:
        for arm in ("Q2", "Q3"):
            remaining = TOTAL_BUDGET_S - (time.time() - t_box0)
            if remaining <= 60:
                rows[arm] = {"arm": arm, "npu": {
                    "failed": True,
                    "error": "skipped: 2 h total box exceeded"}}
                continue
            rows[arm] = run_arm_guarded(arm, min(remaining, 5400))
    else:
        for arm in ("Q2", "Q3"):
            rows[arm] = {
                "arm": arm,
                "skipped": ("Q1 did not finish within the memory guard "
                            f"({q1_guard or q1_no_partial}); per Amendment "
                            "A1 Q2/Q3 run only if Q1 finishes clean"),
            }
            print(f"{arm}: SKIPPED (Q1 guard)", flush=True)

    report["measured"]["rows"] = rows
    report["measured"]["total_elapsed_s"] = round(time.time() - t_box0, 1)
    report["inferred"]["note"] = (
        "Rows describe Quark-quantized graphs and VitisAI EP behaviour on "
        "THIS machine/installation; CPU fallback of unsupported ops is "
        "recorded, not worked around.")
    report["unknown"]["items"] = [
        "Per-op NPU/CPU assignment beyond what the compile log and "
        "EP-emitted files reveal (not exposed via the ORT API).",
    ]
    failed = [a for a, r in rows.items()
              if r.get("quantize", {}).get("failed") or
              r.get("cpu_40scene", {}).get("failed") or
              r.get("npu", {}).get("failed") or
              r.get("memory_guard_abort") or r.get("no_partial") or
              r.get("skipped")]
    report["status"] = ("COMPLETE: Quark INT8 NPU characterization"
                        if not failed else
                        f"PARTIAL: failed arms/rows {failed}")
    report["utc_end"] = utcnow()
    OUT.write_text(json.dumps(report, indent=2))
    print(f"{report['status']} wrote {OUT}")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        assert sys.argv[2] in ARMS
        worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "--arm-full":
        assert sys.argv[2] in ARMS
        full_arm(sys.argv[2])
    elif len(sys.argv) == 1:
        driver()
    else:
        raise SystemExit("usage: npu_quark.py [--worker ARM MODEL_REL] "
                         "or [--arm-full ARM]")
