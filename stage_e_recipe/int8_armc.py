#!/usr/bin/env python
"""Arm C — INT8 op-type sensitivity follow-up (Brief 5, §11 of the spec).

Pre-registered in docs/superpowers/specs/2026-09-23-int8-layer-sensitivity-design.md
§11 (amendment committed BEFORE this ran). Diagnostic only. No training.

Reuses the §2 quantizer and §5 scoring from stage_e_recipe/int8_sensitivity.py
verbatim in behavior (imported, not copied). Same subject, calibration
(first 32 hailo_calib scenes), 40-scene contract scoring with contract_match
guard. Baselines re-scored with the §6 1e-6 px reproduction gate (STOP rule).

Configs (17): C0 conv-only int8, C1 non-conv-only int8, plus one per
non-Conv DequantizeLinear-consumer op type T (T stays fp32). Each config's
.onnx is DELETED as soon as it is scored (sha256 recorded first).

Budget: time the first config; T_est = 17 x S1; STOP if T_est > 2 h.

Outputs ONLY in stage_e_recipe/int8_sensitivity/ (armc.json + per-config
jsons). Never pip installs / never touches environments.
"""
from __future__ import annotations

import datetime
import json
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

WORK = S.WORK
COVERAGE = WORK / "qdq_coverage.json"
OUT = WORK / "armc.json"
BUDGET_S = 2 * 3600


def fail(reason: str, report: dict) -> "NoReturn":
    report["status"] = f"STOP: {reason}"
    OUT.write_text(json.dumps(report, indent=2))
    print(f"STOP: {reason} (partial report -> {OUT})", flush=True)
    raise SystemExit(1)


def main() -> None:
    from src.datasets.kitti2015 import Kitti2015Stereo  # noqa: E402

    WORK.mkdir(parents=True, exist_ok=True)
    try:
        import onnxruntime as ort
        ort_v = ort.__version__
    except Exception as e:
        sys.exit(f"Arm C NOT MEASURABLE: {e}")

    # --- subject assertions (same as §1) ---
    assert S.FP32_SRC.exists(), f"missing subject {S.FP32_SRC}"
    got_sha = S.sha256(S.FP32_SRC)
    assert got_sha == S.REC_FP32_SHA, (
        f"fp32 sha mismatch: {got_sha} != recorded {S.REC_FP32_SHA}")
    conv_names, n_nodes = S.get_conv_names(S.FP32_SRC)
    assert n_nodes == S.REC_N_NODES, f"n_nodes {n_nodes} != recorded"

    import onnx
    fp32 = onnx.load(str(S.FP32_SRC))
    all_names = [n.name for n in fp32.graph.node]
    assert all(all_names), "empty node name in fp32 graph"
    by_type: dict[str, list[str]] = {}
    for n in fp32.graph.node:
        by_type.setdefault(n.op_type, []).append(n.name)
    n_conv = len(by_type.get("Conv", []))
    assert n_conv == len(conv_names), (
        f"Conv count {n_conv} != enumerated {len(conv_names)}")

    # --- op-type list grounded in measured qdq_coverage.json ---
    assert COVERAGE.exists(), f"missing {COVERAGE}"
    cov = json.loads(COVERAGE.read_text())
    measured_types = sorted(
        cov["baseline"]["dequant_consumer_edges_by_op_type"].keys())
    ctypes = [t for t in measured_types if t != "Conv"]
    expected = sorted(["Add", "LeakyRelu", "Sub", "Concat", "Pad", "Slice",
                       "Mul", "ReduceMean", "Div", "Shape", "Softmax",
                       "ReduceSum", "Resize", "Squeeze", "Transpose"])
    assert ctypes == expected, (
        f"consumer-type list drifted: {ctypes} != pre-registered {expected}")
    for t in ctypes:
        assert by_type.get(t), f"op type {t} has no node in fp32 graph"

    non_conv_names = [n for n in all_names
                      if n not in set(conv_names)]
    assert len(non_conv_names) == n_nodes - len(conv_names)

    configs: list[tuple[str, str, list[str]]] = [
        ("C0_conv_only_int8", "conv-only int8 (exclude every non-Conv node)",
         non_conv_names),
        ("C1_nonconv_only_int8", "non-conv-only int8 (exclude every Conv node)",
         list(conv_names)),
    ]
    for t in ctypes:
        configs.append(
            (f"C_{t}", f"keep {t} fp32 (exclude all {t} nodes)",
             list(by_type[t])))

    report: dict = {
        "spec": ("docs/superpowers/specs/2026-09-23-int8-layer-sensitivity-design.md "
                 "§11 (amendment 14b7f91, pre-registered before Arm C ran)"),
        "diagnostic_only": True,
        "not_hailo": ("NOT Hailo validation, compatibility, HEF validation or "
                       "hardware validation. Stage D remains blocked."),
        "subject": {"fp32_onnx": str(S.FP32_SRC.relative_to(REPO)).replace("\\", "/"),
                    "sha256": got_sha, "n_nodes": n_nodes,
                    "n_conv": len(conv_names), "n_non_conv": len(non_conv_names)},
        "quantizer": ("quantize_static, QDQ, QInt8 act, QInt8 weight, "
                      f"per_channel, first {S.CALIB_N} scenes of hailo_calib"),
        "consumer_op_types": ctypes,
        "n_configs": len(configs),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head": S.git_head(),
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "numpy": np.__version__, "onnxruntime": ort_v},
        "measured": {},
        "inferred": {},
        "unknown": {},
    }

    val = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                          disparity_scale=256.0, occluded=True)
    calib = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_calib")
    assert len(val) == 40, f"hailo_val has {len(val)} scenes, expected 40"

    # --- baselines + §6 reproduction gate (STOP rule) ---
    fp32_scored = S.score_contract(S.FP32_SRC, val)
    base_i8 = WORK / "baseline_int8.onnx"
    if not base_i8.exists():
        fail("baseline_int8.onnx missing; refusing to re-quantize silently", report)
    int8_scored = S.score_contract(base_i8, val)
    e_fp32 = fp32_scored["metrics"]["epe"]
    e_int8 = int8_scored["metrics"]["epe"]
    ok = (abs(e_fp32 - S.REC_FP32_EPE) <= S.REPRO_TOL
          and abs(e_int8 - S.REC_INT8_EPE) <= S.REPRO_TOL
          and fp32_scored["guard"]["contract_match"] is True
          and int8_scored["guard"]["contract_match"] is True)
    print(f"baselines: fp32 {e_fp32:.7f} int8 {e_int8:.7f} gate_ok={ok}",
          flush=True)
    if not ok:
        fail("baseline reproduction gate failed (fp32/int8 EPE or guard)", report)
    gap = e_int8 - e_fp32
    report["measured"]["baseline_fp32"] = fp32_scored
    report["measured"]["baseline_int8"] = int8_scored
    report["measured"]["gap_G"] = gap

    # --- sweep with 2 h budget guard on the first config ---
    rows: list[dict] = []
    t_first: float | None = None
    for k, (tag, label, excl) in enumerate(configs):
        dst = WORK / f"armc_{S.sanitize(tag)}.onnx"
        t0 = time.time()
        qrec = S.quantize_with_excludes(S.FP32_SRC, dst, calib, excl)
        artifact_sha = S.sha256(dst)
        sc = S.score_contract(dst, val)
        dt = time.time() - t0
        if t_first is None:
            t_first = dt
            t_est = len(configs) * t_first
            report["timing_first_config_s"] = round(t_first, 1)
            report["t_est_s"] = round(t_est, 1)
            report["t_est_h"] = round(t_est / 3600, 2)
            print(f"first config {tag}: {dt:.1f}s x{len(configs)} "
                  f"-> est {t_est / 3600:.2f}h", flush=True)
            if t_est > BUDGET_S:
                dst.unlink(missing_ok=True)
                fail(f"budget exceeded: est {t_est / 3600:.2f}h > 2h", report)
        epe = sc["metrics"]["epe"]
        row = {"config": tag, "label": label,
               "n_excluded": len(excl),
               "artifact_sha256": artifact_sha,
               "epe": epe,
               "delta_vs_fp32": epe - e_fp32,
               "delta_vs_int8": epe - e_int8,
               "recovered_fraction_R": (e_int8 - epe) / gap if gap else None,
               "d1": sc["metrics"]["d1"],
               "metrics": sc["metrics"], "guard": sc.get("guard"),
               "nonfinite_pixels": sc["nonfinite_pixels"],
               "score_s": sc["score_s"],
               "quantize_s": qrec.get("quantize_s")}
        rows.append(row)
        (WORK / f"armc_{S.sanitize(tag)}.json").write_text(
            json.dumps(row, indent=2))
        dst.unlink(missing_ok=True)  # delete .onnx as soon as scored
        print(f"ArmC {k + 1}/{len(configs)} {tag}: EPE {epe:.4f} "
              f"R={row['recovered_fraction_R']:.3f} (onnx deleted)",
              flush=True)

    ranked = sorted(rows, key=lambda r: r["epe"])
    report["measured"]["armc_ranked"] = ranked
    dominant = [r for r in ranked
                if r["recovered_fraction_R"] is not None
                and r["recovered_fraction_R"] >= 0.50]
    report["measured"]["dominant_configs_R_ge_0_5"] = [
        {"config": r["config"], "epe": r["epe"],
         "R": r["recovered_fraction_R"]} for r in dominant]
    report["inferred"]["note"] = (
        "Ranking is descriptive under THIS quantizer. A high-R config is where "
        "keeping that node set in fp32 recovers most of the gap; it does not "
        "identify a fix and bears no Hailo implication.")
    report["unknown"]["items"] = [
        "Mechanism behind any op-type difference.",
        "Whether results transfer to other seeds, procedures, or hardware.",
    ]
    report["status"] = "COMPLETE: all Arm C configs scored on the 40-scene contract"
    OUT.write_text(json.dumps(report, indent=2))
    print(f"Arm C COMPLETE gap={gap:.4f} dominant={len(dominant)} wrote {OUT}")


if __name__ == "__main__":
    main()
