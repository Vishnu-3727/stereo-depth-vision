#!/usr/bin/env python
"""INT8 per-layer sensitivity for E3 seed-0 — diagnostic sweep (no training).

Implements docs/superpowers/specs/2026-09-23-int8-layer-sensitivity-design.md
exactly. Pre-registered BEFORE measurement (commit c53ebe7 scope).

Subject: reuse stage_e_recipe/int8_e3/e3_seed0_best_fp32.onnx (provenance
asserted against int8_e3.json). Quantizer: exact int8_control.py procedure
(QDQ, QInt8 act+weight, per_channel, first-32 hailo_calib scenes).
Sweeps: leave-one-out (nodes_to_exclude=[L]) + only-one
(nodes_to_exclude=[all Conv except L]) over every Conv node.
Scoring: frozen 40-scene contract, contract_match required.
Baseline gate: reproduce recorded E3-seed0 numbers within 1e-6 px else STOP.
Budget: time one scoring + one quantization; >6h estimate -> 10-scene monitor
sweep + top-5 40-scene re-score.

Outputs ONLY in stage_e_recipe/int8_sensitivity/.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from phase1.harness.frozen_eval import pooled_metrics, refuse_unless_contract  # noqa: E402
from src.datasets.kitti2015 import Kitti2015Stereo, normalize  # noqa: E402

FP32_SRC = HERE / "int8_e3" / "e3_seed0_best_fp32.onnx"
WORK = HERE / "int8_sensitivity"
CALIB_N = 32

# Recorded references from stage_e_recipe/int8_e3/int8_e3.json seed "0"
# (labeled "recorded reference", NOT results of this run).
REC_FP32_EPE = 1.18598534271453
REC_INT8_EPE = 5.248941413313285
REC_FP32_SHA = "7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645"
REC_N_NODES = 712
REPRO_TOL = 1e-6
BUDGET_S = 6 * 3600


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def sanitize(name: str) -> str:
    return name.strip("/").replace("/", "__").replace(".", "_")


def get_conv_names(fp32_path: Path) -> list[str]:
    import onnx
    m = onnx.load(str(fp32_path))
    names = sorted(n.name for n in m.graph.node if n.op_type == "Conv")
    assert names, "no Conv nodes found"
    assert all(names), "empty Conv node name found"
    assert len(names) == len(set(names)), "duplicate Conv node names"
    return names, len(m.graph.node)


def quantize_with_excludes(src: Path, dst: Path, calib_ds,
                           exclude: list[str] | None) -> dict:
    from onnxruntime.quantization import (QuantFormat, QuantType,
                                          quantize_static,
                                          CalibrationDataReader)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]

    class Reader(CalibrationDataReader):
        def __init__(self, ds, count):
            self.items = []
            for i in range(min(count, len(ds))):
                s = ds[i]
                self.items.append({names[0]: normalize(s.left),
                                   names[1]: normalize(s.right)})
            self.it = iter(self.items)

        def get_next(self):
            return next(self.it, None)

    t0 = time.time()
    kw = dict(quant_format=QuantFormat.QDQ,
              activation_type=QuantType.QInt8,
              weight_type=QuantType.QInt8,
              per_channel=True)
    if exclude:
        quantize_static(str(src), str(dst), Reader(calib_ds, CALIB_N),
                        nodes_to_exclude=exclude, **kw)
    else:
        quantize_static(str(src), str(dst), Reader(calib_ds, CALIB_N), **kw)
    return {"path": str(dst.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256(dst), "bytes": dst.stat().st_size,
            "parent_onnx_sha256": sha256(src),
            "procedure": "quantize_static, QDQ, QInt8 act, QInt8 weight, per_channel",
            "calibration": f"first {CALIB_N} scenes of hailo_calib",
            "nodes_to_exclude": list(exclude) if exclude else [],
            "quantize_s": round(time.time() - t0, 1)}


def score_contract(path: Path, ds) -> dict:
    """Full 40-scene frozen-contract scoring. Raises on contract mismatch."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    out_name = sess.get_outputs()[0].name
    preds, gts = [], []
    nonfinite = 0
    t0 = time.time()
    for i in range(len(ds)):
        s = ds[i]
        out = sess.run([out_name], {ins[0]: normalize(s.left),
                                    ins[1]: normalize(s.right)})[0]
        pred = out[0, 0].astype(np.float64)
        nonfinite += int((~np.isfinite(pred)).sum())
        valid = s.disparity > 0
        preds.append(pred[valid].astype(np.float64))
        gts.append(s.disparity[valid].astype(np.float64))
    m = pooled_metrics(preds, gts)
    guard = refuse_unless_contract(len(ds), int(sum(p.size for p in preds)),
                                   256.0, "hailo_val", "disp_occ_0")
    return {"metrics": m, "guard": guard, "nonfinite_pixels": nonfinite,
            "score_s": round(time.time() - t0, 1)}


def score_monitor(path: Path, ds, limit: int = 10) -> dict:
    """10-scene monitor scoring. Diagnostic only, NO contract guard."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    ins = [i.name for i in sess.get_inputs()]
    out_name = sess.get_outputs()[0].name
    preds, gts = [], []
    nonfinite = 0
    t0 = time.time()
    n = min(limit, len(ds))
    for i in range(n):
        s = ds[i]
        out = sess.run([out_name], {ins[0]: normalize(s.left),
                                    ins[1]: normalize(s.right)})[0]
        pred = out[0, 0].astype(np.float64)
        nonfinite += int((~np.isfinite(pred)).sum())
        valid = s.disparity > 0
        preds.append(pred[valid].astype(np.float64))
        gts.append(s.disparity[valid].astype(np.float64))
    m = pooled_metrics(preds, gts)
    return {"metrics": m,
            "contract": "10-scene-monitor, not-frozen-contract, diagnostic-only",
            "scenes": n, "nonfinite_pixels": nonfinite,
            "score_s": round(time.time() - t0, 1)}


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)

    # --- subject assertions (STOP if provenance broken) ---
    assert FP32_SRC.exists(), f"missing subject {FP32_SRC}"
    got_sha = sha256(FP32_SRC)
    assert got_sha == REC_FP32_SHA, (
        f"fp32 sha mismatch: {got_sha} != recorded {REC_FP32_SHA}")
    conv_names, n_nodes = get_conv_names(FP32_SRC)
    assert n_nodes == REC_N_NODES, f"n_nodes {n_nodes} != recorded {REC_N_NODES}"
    n = len(conv_names)
    print(f"subject OK: sha match, {n_nodes} nodes, {n} Conv groups", flush=True)

    val = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                          disparity_scale=256.0, occluded=True)
    calib = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_calib")
    assert len(val) == 40, f"hailo_val has {len(val)} scenes, expected 40"

    try:
        import onnxruntime as ort
        ort_v = ort.__version__
    except Exception as e:
        sys.exit(f"INT8 sensitivity NOT MEASURABLE: {e}")

    report: dict = {
        "spec": "docs/superpowers/specs/2026-09-23-int8-layer-sensitivity-design.md",
        "diagnostic_only": True,
        "not_hailo": ("NOT Hailo validation, compatibility, HEF validation or "
                       "hardware validation. Stage D remains blocked."),
        "subject": {"fp32_onnx": str(FP32_SRC.relative_to(REPO)).replace("\\", "/"),
                    "sha256": got_sha, "n_nodes": n_nodes,
                    "provenance": "reused; recorded in stage_e_recipe/int8_e3/int8_e3.json"},
        "quantizer": ("quantize_static, QDQ, QInt8 act, QInt8 weight, "
                      f"per_channel, first {CALIB_N} scenes of hailo_calib"),
        "recorded_reference_e3_seed0": {"fp32_epe": REC_FP32_EPE,
                                        "int8_epe": REC_INT8_EPE,
                                        "P": REC_INT8_EPE - REC_FP32_EPE},
        "conv_groups": conv_names,
        "n_groups": n,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head": git_head(),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "onnxruntime": ort_v},
        "measured": {},
        "inferred": {},
        "unknown": {},
    }

    # --- budget timing: one scoring + one quantization ---
    t0 = time.time()
    fp32_scored = score_contract(FP32_SRC, val)
    s_score = time.time() - t0
    base_i8 = WORK / "baseline_int8.onnx"
    t0 = time.time()
    if not base_i8.exists():
        qinfo = quantize_with_excludes(FP32_SRC, base_i8, calib, None)
    else:
        # still time a fresh quantization for the estimate on a temp file
        tmp = WORK / "_timing_quant.onnx"
        qinfo = quantize_with_excludes(FP32_SRC, tmp, calib, None)
        tmp.unlink(missing_ok=True)
    s_quant = time.time() - t0
    n_configs = 2 + 2 * n
    t_est = n_configs * (s_score + s_quant)
    branch = "contract40" if t_est <= BUDGET_S else "monitor10+top5"
    report["timing"] = {"single_score_s": round(s_score, 1),
                        "single_quantize_s": round(s_quant, 1),
                        "n_configs": n_configs,
                        "t_est_s": round(t_est, 1),
                        "t_est_h": round(t_est / 3600, 2),
                        "budget_s": BUDGET_S, "branch": branch}
    print(f"timing: score {s_score:.1f}s quant {s_quant:.1f}s "
          f"x{n_configs} -> est {t_est / 3600:.2f}h -> branch {branch}",
          flush=True)

    # --- baselines + reproduction gate (STOP rule) ---
    if base_i8.exists() and "sha256" not in qinfo:
        pass
    if not base_i8.exists():
        raise RuntimeError("baseline quantization did not produce a file")
    base_qinfo = {"path": str(base_i8.relative_to(REPO)).replace("\\", "/"),
                  "sha256": sha256(base_i8), "bytes": base_i8.stat().st_size,
                  "parent_onnx_sha256": got_sha,
                  "procedure": report["quantizer"],
                  "calibration": f"first {CALIB_N} scenes of hailo_calib",
                  "nodes_to_exclude": [],
                  "quantize_s": qinfo.get("quantize_s")}
    int8_scored = score_contract(base_i8, val)
    report["measured"]["baseline_fp32"] = fp32_scored
    report["measured"]["baseline_int8"] = int8_scored
    report["measured"]["baseline_int8_artifact"] = base_qinfo

    e_fp32 = fp32_scored["metrics"]["epe"]
    e_int8 = int8_scored["metrics"]["epe"]
    ok_fp32 = abs(e_fp32 - REC_FP32_EPE) <= REPRO_TOL
    ok_int8 = abs(e_int8 - REC_INT8_EPE) <= REPRO_TOL
    ok_guard = (fp32_scored["guard"]["contract_match"] is True
                and int8_scored["guard"]["contract_match"] is True)
    print(f"baselines: fp32 {e_fp32:.7f} (rec {REC_FP32_EPE:.7f}, ok={ok_fp32}) | "
          f"int8 {e_int8:.7f} (rec {REC_INT8_EPE:.7f}, ok={ok_int8}) | "
          f"guards ok={ok_guard}", flush=True)
    if not (ok_fp32 and ok_int8 and ok_guard):
        report["baseline_gate"] = "STOP: reproduction failed"
        out = WORK / "int8_sensitivity.json"
        out.write_text(json.dumps(report, indent=2))
        sys.exit("STOP: baseline reproduction gate failed; see "
                 f"{out}. fp32_ok={ok_fp32} int8_ok={ok_int8} guards_ok={ok_guard}")
    report["baseline_gate"] = "PASS: reproduction within 1e-6 px, guards true"
    gap = e_int8 - e_fp32
    report["measured"]["gap_G"] = gap

    loo_rows, only_rows = [], []

    def finalize_row(arm, idx, lname, artifact, qrec, scored):
        epe = scored["metrics"]["epe"]
        return {"arm": arm, "index": idx, "group": lname,
                "artifact": artifact, "quantize": qrec,
                "epe": epe,
                "delta_vs_fp32": epe - e_fp32,
                "delta_vs_int8": epe - e_int8,
                "recovered_fraction_R": (e_int8 - epe) / gap if gap else None,
                "d1": scored["metrics"]["d1"],
                "metrics": scored["metrics"], "guard": scored.get("guard"),
                "nonfinite_pixels": scored["nonfinite_pixels"],
                "score_s": scored["score_s"]}

    if branch == "contract40":
        for i, lname in enumerate(conv_names):
            # Arm A: leave-one-out
            dst = WORK / f"loo_{i:02d}_{sanitize(lname)}.onnx"
            qrec = quantize_with_excludes(FP32_SRC, dst, calib, [lname])
            sc = score_contract(dst, val)
            row = finalize_row("leave-one-out", i, lname,
                               str(dst.relative_to(REPO)).replace("\\", "/"),
                               qrec, sc)
            loo_rows.append(row)
            (WORK / f"loo_{i:02d}.json").write_text(json.dumps(row, indent=2))
            print(f"LOO {i}/{n - 1} {lname}: EPE {row['epe']:.4f} "
                  f"R={row['recovered_fraction_R']:.3f}", flush=True)
            # Arm B: only-one
            dst = WORK / f"only_{i:02d}_{sanitize(lname)}.onnx"
            qrec = quantize_with_excludes(
                FP32_SRC, dst, calib, [c for c in conv_names if c != lname])
            sc = score_contract(dst, val)
            row = finalize_row("only-one", i, lname,
                               str(dst.relative_to(REPO)).replace("\\", "/"),
                               qrec, sc)
            only_rows.append(row)
            (WORK / f"only_{i:02d}.json").write_text(json.dumps(row, indent=2))
            print(f"ONLY {i}/{n - 1} {lname}: EPE {row['epe']:.4f}", flush=True)
    else:
        # 10-scene monitor sweep, then top-5 re-score on contract.
        mon_loo, mon_only = [], []
        for i, lname in enumerate(conv_names):
            dst = WORK / f"loo_{i:02d}_{sanitize(lname)}.onnx"
            qrec = quantize_with_excludes(FP32_SRC, dst, calib, [lname])
            sc = score_monitor(dst, val)
            mon_loo.append(finalize_row(
                "leave-one-out-monitor", i, lname,
                str(dst.relative_to(REPO)).replace("\\", "/"), qrec,
                {**sc, "metrics": sc["metrics"]}))
            print(f"MON-LOO {i}/{n - 1}: EPE {mon_loo[-1]['epe']:.4f}", flush=True)
            dst = WORK / f"only_{i:02d}_{sanitize(lname)}.onnx"
            qrec = quantize_with_excludes(
                FP32_SRC, dst, calib, [c for c in conv_names if c != lname])
            sc = score_monitor(dst, val)
            mon_only.append(finalize_row(
                "only-one-monitor", i, lname,
                str(dst.relative_to(REPO)).replace("\\", "/"), qrec,
                {**sc, "metrics": sc["metrics"]}))
            print(f"MON-ONLY {i}/{n - 1}: EPE {mon_only[-1]['epe']:.4f}", flush=True)
        report["measured"]["monitor_rows"] = {"loo": mon_loo, "only": mon_only}
        top_loo = sorted(mon_loo, key=lambda r: r["epe"])[:5]
        top_only = sorted(mon_only, key=lambda r: r["epe"], reverse=True)[:5]
        for r in top_loo:
            sc = score_contract(REPO / r["artifact"], val)
            loo_rows.append(finalize_row("leave-one-out", r["index"],
                                         r["group"], r["artifact"],
                                         r["quantize"], sc))
        for r in top_only:
            sc = score_contract(REPO / r["artifact"], val)
            only_rows.append(finalize_row("only-one", r["index"],
                                          r["group"], r["artifact"],
                                          r["quantize"], sc))
        report["measured"]["rescored"] = {"top5_loo_contract": loo_rows,
                                          "top5_only_contract": only_rows}

    loo_ranked = sorted(loo_rows, key=lambda r: r["epe"])
    only_ranked = sorted(only_rows, key=lambda r: r["epe"], reverse=True)
    report["measured"]["loo_ranked"] = loo_ranked
    report["measured"]["only_ranked"] = only_ranked
    dominant = [r for r in loo_ranked
                if r["recovered_fraction_R"] is not None
                and r["recovered_fraction_R"] >= 0.50]
    report["measured"]["dominant_layers_R_ge_0_5"] = [
        {"group": r["group"], "epe": r["epe"],
         "R": r["recovered_fraction_R"]} for r in dominant]
    report["inferred"]["note"] = (
        "Ranking is descriptive. A high-R leave-one-out layer is where fp32 "
        "restoration helps most under THIS quantizer; it does not identify a "
        "fix and bears no Hailo implication.")
    report["unknown"]["items"] = [
        "Layers not re-scored on the contract (monitor branch only).",
        "Any mechanism behind per-layer differences.",
        "Whether results transfer to other seeds, procedures, or hardware.",
    ]

    out = WORK / "int8_sensitivity.json"
    out.write_text(json.dumps(report, indent=2))

    def table(rows):
        lines = ["| rank | layer | EPE | d_fp32 | d_int8 | R | D1 | nonfinite |",
                 "|---|---|---|---|---|---|---|---|"]
        for k, r in enumerate(rows, 1):
            lines.append(
                f"| {k} | `{r['group']}` | {r['epe']:.4f} | "
                f"{r['delta_vs_fp32']:+.4f} | {r['delta_vs_int8']:+.4f} | "
                f"{r['recovered_fraction_R']:.3f} | {r['d1']:.2f} | "
                f"{r['nonfinite_pixels']} |")
        return "\n".join(lines)

    md = [
        "# INT8 per-layer sensitivity — E3 seed-0 (measured)",
        "",
        "Diagnostic only. No accept/reject. No training. NOT Hailo evidence; "
        "Stage D remains blocked.",
        "",
        f"- Branch: `{branch}` (est {report['timing']['t_est_h']} h vs 6 h budget).",
        f"- Single-score timing: {report['timing']['single_score_s']} s; "
        f"single-quantize: {report['timing']['single_quantize_s']} s; "
        f"configs: {n_configs}.",
        f"- Baseline fp32 EPE (measured): {e_fp32:.7f} "
        f"(recorded ref {REC_FP32_EPE:.7f}).",
        f"- Baseline int8 EPE (measured): {e_int8:.7f} "
        f"(recorded ref {REC_INT8_EPE:.7f}).",
        f"- Gap G (measured): {gap:.7f} px. Baseline gate: "
        f"{report['baseline_gate']}.",
        f"- Guards: fp32 contract_match={fp32_scored['guard']['contract_match']}, "
        f"int8 contract_match={int8_scored['guard']['contract_match']}.",
        "",
        "## Arm A — leave-one-out (all int8 except L; sorted by EPE ascending)",
        "",
        table(loo_ranked),
        "",
        "## Arm B — only-one (only L int8; sorted by EPE descending)",
        "",
        table(only_ranked),
        "",
        "## Pre-registered readout",
        "",
        ("Dominant layers (R >= 0.50): "
         + (", ".join(f"`{d['group']}` (R={d['R']:.3f})" for d in
                        report["measured"]["dominant_layers_R_ge_0_5"])
            if dominant else "none measured.")),
        "",
        "## Provenance",
        "",
        f"- UTC: {report['utc']}; git HEAD: {report['git_head']}.",
        f"- Env: {report['environment']}.",
        f"- Subject sha256 asserted: `{got_sha}`; Conv groups: {n}.",
        "- Full rows: `int8_sensitivity.json` (+ per-layer json).",
    ]
    (WORK / "REPORT.md").write_text("\n".join(md) + "\n")
    print(f"branch={branch} gap={gap:.4f} dominant={len(dominant)} wrote {out}")


if __name__ == "__main__":
    main()
