#!/usr/bin/env python
"""INT8 regression-head-fp32 recovery — E3 seed-0/1/2 (pre-registered diagnostic).

Implements docs/superpowers/specs/2026-09-24-int8-regression-head-fp32-design.md
exactly. Pre-registered BEFORE measurement (spec commit 6a6cf6b scope).

Subjects: reuse stage_e_recipe/int8_e3/e3_seed{0,1,2}_best_fp32.onnx (provenance
asserted per seed against int8_e3.json). Quantizer: exact int8_control.py
procedure via int8_sensitivity.py (imported, not copied — see REUSE_METHOD).
Arms per seed: R0 all-int8 baseline (nodes_to_exclude=None, fresh), R1 readout
fp32 rest int8 (nodes_to_exclude=[all /regression/* names]), plus fp32 re-score.
Scoring: frozen 40-scene contract, contract_match required.
Gates: §6 baseline reproduction (1e-6 px, STOP rule) and §3 no-op precondition
(kept set must carry QDQ in the R0 baseline, STOP rule). Budget §9: T_est = 6×S1.

Outputs ONLY in stage_e_recipe/int8_regression_head/. Each quantized .onnx is
DELETED immediately after scoring (sha256 recorded before deletion).
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

# Reuse method (spec: say which): IMPORT from stage_e_recipe/int8_sensitivity.py.
# Quantizer (S.quantize_with_excludes), scoring (S.score_contract), sha/git
# helpers (S.sha256, S.git_head) and calibration constant (S.CALIB_N) are used
# unmodified — quantizer settings, calibration set and scoring are unchanged.
REUSE_METHOD = "import from stage_e_recipe/int8_sensitivity.py (no copy, no changes)"

WORK = HERE / "int8_regression_head"
OUT = WORK / "int8_regression_head.json"
BUDGET_S = 2 * 3600
REPRO_TOL = 1e-6

# User-set threshold block (§7, set by the user on 2026-09-24 before any R1
# measurement). Written at the top of the output JSON; the verdict below reads
# report["thresholds"] mechanically and never chooses values.
THRESHOLDS = {
    "R_ACCEPT": 0.50,
    "N_SEEDS_REQUIRED": 3,
    "RESID_MAX_PX": 0.50,
    "set_by": "user on 2026-09-24 before any R1 measurement",
    "semantics": {
        "R_ACCEPT": "per-seed recovered fraction R_R1 must be >= 0.50",
        "N_SEEDS_REQUIRED": "must hold on all 3 E3 seeds",
        "RESID_MAX_PX": "EPE_R1 - EPE_fp32 must be <= 0.50 px on each of those seeds",
    },
}

SEEDS = {
    0: {
        "fp32_onnx": "stage_e_recipe/int8_e3/e3_seed0_best_fp32.onnx",
        "sha256": "7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645",
        "bytes": 1701550, "opset": 13, "n_nodes": 712,
        "parent_checkpoint_sha256": "7d3c2109c11339b6fb30cc33f5bd042667740f3bc7345768f4e4b04704836209",
        "params": 397954,
        "ref_fp32_epe": 1.18598534271453,
        "ref_int8_epe": 5.248941413313285,
    },
    1: {
        "fp32_onnx": "stage_e_recipe/int8_e3/e3_seed1_best_fp32.onnx",
        "sha256": "9b4922c91367a2c41254795e0c7160dcb05313f7fb948e2c0d56f05d9ebe9ac2",
        "bytes": 1701550, "opset": 13, "n_nodes": 712,
        "parent_checkpoint_sha256": "133c19c41585445da9aabf18919e62f5805b5f8abcf80232808d26d33aa293b0",
        "params": 397954,
        "ref_fp32_epe": 1.178821279341308,
        "ref_int8_epe": 5.3849187722115985,
    },
    2: {
        "fp32_onnx": "stage_e_recipe/int8_e3/e3_seed2_best_fp32.onnx",
        "sha256": "7c0c16e4cf66a7ca14e59f5be31801f19815ae1ba6ca72014cb7dd3e99e50418",
        "bytes": 1701550, "opset": 13, "n_nodes": 712,
        "parent_checkpoint_sha256": "1ba8ca09bfabf2c689b6ec7fbbbae1aa70367d33562b1aad06521ba71904cbcf",
        "params": 397954,
        "ref_fp32_epe": 1.1713166599983933,
        "ref_int8_epe": 5.87156272118187,
    },
}

REQUIRED_MUL = ["/regression/Mul", "/regression/Mul_1", "/regression/Mul_2"]


def fail(reason: str, report: dict) -> "NoReturn":
    report["status"] = f"STOP: {reason}"
    OUT.write_text(json.dumps(report, indent=2))
    for p in sorted(WORK.glob("*.onnx")):
        p.unlink(missing_ok=True)
    print(f"STOP: {reason} (partial report -> {OUT})", flush=True)
    raise SystemExit(1)


def kept_set(fp32_path: Path) -> list[str]:
    import onnx

    m = onnx.load(str(fp32_path))
    names = sorted(n.name for n in m.graph.node if n.name.startswith("/regression/"))
    return names, len(m.graph.node)


def coverage_audit(baseline_int8_path: Path, kept: list[str]) -> dict:
    """§3 no-op precondition audit against the seed's fresh R0 baseline.

    For each kept /regression/* node: does it consume at least one
    DequantizeLinear output in the quantized baseline? Also records whether
    the R0 artifact differs from its fp32 parent (otherwise altered).
    """
    import onnx

    m = onnx.load(str(baseline_int8_path))
    dql_outputs = set()
    for n in m.graph.node:
        if n.op_type == "DequantizeLinear":
            dql_outputs.update(n.output)
    per_node = {}
    for n in m.graph.node:
        if n.name in set(kept):
            hits = sum(1 for i in n.input if i in dql_outputs)
            per_node[n.name] = {"op_type": n.op_type, "dql_input_edges": hits}
    for name in kept:
        per_node.setdefault(name, {"op_type": None, "dql_input_edges": 0,
                                   "note": "kept node absent from quantized graph"})
    covered = sorted(n for n, r in per_node.items() if r["dql_input_edges"] > 0)
    return {
        "baseline": str(baseline_int8_path.relative_to(REPO)).replace("\\", "/"),
        "baseline_sha256": S.sha256(baseline_int8_path),
        "n_kept": len(kept),
        "n_kept_consuming_dql": len(covered),
        "kept_consuming_dql": covered,
        "per_node": per_node,
    }


def main() -> None:
    from src.datasets.kitti2015 import Kitti2015Stereo  # noqa: E402

    WORK.mkdir(parents=True, exist_ok=True)
    try:
        import onnxruntime as ort
        ort_v = ort.__version__
    except Exception as e:
        sys.exit(f"INT8 regression-head NOT MEASURABLE: {e}")

    report: dict = {
        "spec": "docs/superpowers/specs/2026-09-24-int8-regression-head-fp32-design.md",
        "spec_pre_register_commit": "6a6cf6b",
        "thresholds": THRESHOLDS,
        "diagnostic_only": True,
        "not_hailo": ("NOT Hailo validation, compatibility, HEF validation or "
                       "hardware validation. Stage D remains blocked."),
        "reuse_method": REUSE_METHOD,
        "quantizer": ("quantize_static, QDQ, QInt8 act, QInt8 weight, "
                      f"per_channel, first {S.CALIB_N} scenes of hailo_calib"),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head": S.git_head(),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "numpy": np.__version__, "onnxruntime": ort_v},
        "measured": {},
        "inferred": {},
        "unknown": {},
    }

    val = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_val",
                          disparity_scale=256.0, occluded=True)
    calib = Kitti2015Stereo(REPO / "data" / "kitti2015", split="hailo_calib")
    assert len(val) == 40, f"hailo_val has {len(val)} scenes, expected 40"

    # --- §1 subject assertions + §3 kept-set enumeration (per seed) ---
    kept_per_seed: dict[str, list[str]] = {}
    for seed in (0, 1, 2):
        cfg = SEEDS[seed]
        fp32 = REPO / cfg["fp32_onnx"]
        assert fp32.exists(), f"missing subject {fp32}"
        got_sha = S.sha256(fp32)
        assert got_sha == cfg["sha256"], (
            f"seed {seed} fp32 sha mismatch: {got_sha} != recorded {cfg['sha256']}")
        kept, n_nodes = kept_set(fp32)
        assert n_nodes == cfg["n_nodes"] == 712, (
            f"seed {seed} n_nodes {n_nodes} != recorded 712")
        assert kept, f"seed {seed}: empty /regression/* kept set"
        for m in REQUIRED_MUL:
            if m not in kept:
                fail(f"seed {seed}: required readout node {m} missing from kept set",
                     report)
        kept_per_seed[str(seed)] = kept
        print(f"seed {seed} OK: sha match, {n_nodes} nodes, "
              f"{len(kept)} /regression/* nodes", flush=True)
    report["measured"]["kept_set_per_seed"] = kept_per_seed
    if len({tuple(v) for v in kept_per_seed.values()}) > 1:
        report["measured"]["kept_set_note"] = (
            "kept sets differ across seeds; recorded all three, proceeding per seed")
    else:
        report["measured"]["kept_set_note"] = "identical /regression/* set on all 3 seeds"

    # --- per-seed R0 quantize + fp32 score; §9 budget on first config ---
    seeds_data: dict[str, dict] = {}
    s1: float | None = None
    for seed in (0, 1, 2):
        cfg = SEEDS[seed]
        fp32 = REPO / cfg["fp32_onnx"]
        kept = kept_per_seed[str(seed)]
        dst_r0 = WORK / f"seed{seed}_R0_all_int8.onnx"

        t0 = time.time()
        qrec_r0 = S.quantize_with_excludes(fp32, dst_r0, calib, None)
        artifact_sha_r0 = S.sha256(dst_r0)

        # --- §3 no-op precondition (before scoring) ---
        audit = coverage_audit(dst_r0, kept)
        audit["fp32_parent_sha256"] = cfg["sha256"]
        audit["r0_differs_from_fp32"] = audit["baseline_sha256"] != cfg["sha256"]
        print(f"seed {seed} coverage: {audit['n_kept_consuming_dql']}/"
              f"{audit['n_kept']} kept nodes consume DQL in R0", flush=True)
        if audit["n_kept_consuming_dql"] == 0:
            report["measured"]["coverage_audit_seed%d" % seed] = audit
            dst_r0.unlink(missing_ok=True)
            fail(f"seed {seed}: §3 no-op precondition failed — kept set carries "
                 f"no QDQ in the R0 baseline; arm cannot move the score", report)
        report["measured"][f"coverage_audit_seed{seed}"] = audit

        fp32_scored = S.score_contract(fp32, val)
        r0_scored = S.score_contract(dst_r0, val)
        dt = time.time() - t0
        dst_r0.unlink(missing_ok=True)  # delete .onnx as soon as scored

        if s1 is None:
            s1 = dt
            t_est = 6 * s1
            report["timing"] = {"S1_first_config_s": round(s1, 1),
                                "T_est_s": round(t_est, 1),
                                "T_est_h": round(t_est / 3600, 2),
                                "budget_s": BUDGET_S,
                                "branch": ("proceed"
                                           if t_est <= BUDGET_S else "STOP over budget")}
            print(f"budget: S1 {s1:.1f}s x6 -> est {t_est / 3600:.2f}h", flush=True)
            if t_est > BUDGET_S:
                report["measured"]["partial_seed0"] = {
                    "fp32": fp32_scored, "R0": r0_scored, "quantize": qrec_r0}
                fail(f"budget exceeded: est {t_est / 3600:.2f}h > 2h", report)

        seeds_data[str(seed)] = {
            "fp32_scored": fp32_scored, "R0_scored": r0_scored,
            "R0_quantize": {**qrec_r0, "sha256": artifact_sha_r0},
        }
        e_f = fp32_scored["metrics"]["epe"]
        e_0 = r0_scored["metrics"]["epe"]
        print(f"seed {seed}: fp32 {e_f:.7f} (ref {cfg['ref_fp32_epe']:.7f}) | "
              f"R0 {e_0:.7f} (ref {cfg['ref_int8_epe']:.7f})", flush=True)

    # --- §6 baseline reproduction gate (STOP rule, all seeds) ---
    gate_rows = {}
    gate_ok_all = True
    for seed in (0, 1, 2):
        cfg = SEEDS[seed]
        d = seeds_data[str(seed)]
        e_f = d["fp32_scored"]["metrics"]["epe"]
        e_0 = d["R0_scored"]["metrics"]["epe"]
        ok_f = abs(e_f - cfg["ref_fp32_epe"]) <= REPRO_TOL
        ok_0 = abs(e_0 - cfg["ref_int8_epe"]) <= REPRO_TOL
        ok_g = (d["fp32_scored"]["guard"]["contract_match"] is True
                and d["R0_scored"]["guard"]["contract_match"] is True)
        gate_rows[str(seed)] = {"fp32_ok": ok_f, "R0_ok": ok_0, "guards_ok": ok_g,
                                "e_fp32": e_f, "e_R0": e_0}
        if not (ok_f and ok_0 and ok_g):
            gate_ok_all = False
    report["measured"]["baseline_gate_per_seed"] = gate_rows
    if not gate_ok_all:
        report["measured"]["seeds_baselines"] = {
            s: {"fp32": v["fp32_scored"], "R0": v["R0_scored"],
                "R0_quantize": v["R0_quantize"]} for s, v in seeds_data.items()}
        report["baseline_gate"] = "STOP: reproduction failed"
        fail("baseline reproduction gate failed (§6); see baseline_gate_per_seed",
             report)
    report["baseline_gate"] = "PASS: reproduction within 1e-6 px, guards true (all seeds)"

    # --- R1 arm per seed (readout fp32, rest int8) ---
    rows = []
    for seed in (0, 1, 2):
        cfg = SEEDS[seed]
        fp32 = REPO / cfg["fp32_onnx"]
        kept = kept_per_seed[str(seed)]
        d = seeds_data[str(seed)]
        e_f = d["fp32_scored"]["metrics"]["epe"]
        e_0 = d["R0_scored"]["metrics"]["epe"]
        gap = e_0 - e_f
        dst_r1 = WORK / f"seed{seed}_R1_regression_fp32.onnx"
        qrec_r1 = S.quantize_with_excludes(fp32, dst_r1, calib, kept)
        artifact_sha_r1 = S.sha256(dst_r1)
        r1_scored = S.score_contract(dst_r1, val)
        dst_r1.unlink(missing_ok=True)  # delete .onnx as soon as scored
        e_1 = r1_scored["metrics"]["epe"]
        r_r1 = (e_0 - e_1) / gap if gap else None
        resid = e_1 - e_f
        row = {"seed": seed,
               "e_fp32": e_f, "e_R0": e_0, "e_R1": e_1,
               "G_seed": gap,
               "delta_R1_vs_fp32": resid,
               "delta_R1_vs_R0": e_1 - e_0,
               "R_R1": r_r1,
               "d1_R1": r1_scored["metrics"]["d1"],
               "d1_fp32": d["fp32_scored"]["metrics"]["d1"],
               "d1_R0": d["R0_scored"]["metrics"]["d1"],
               "nonfinite_R1": r1_scored["nonfinite_pixels"],
               "nonfinite_fp32": d["fp32_scored"]["nonfinite_pixels"],
               "nonfinite_R0": d["R0_scored"]["nonfinite_pixels"],
               "contract_match_R1": r1_scored["guard"]["contract_match"],
               "metrics_R1": r1_scored["metrics"],
               "metrics_fp32": d["fp32_scored"]["metrics"],
               "metrics_R0": d["R0_scored"]["metrics"],
               "guard_R1": r1_scored["guard"],
               "R1_quantize": {**qrec_r1, "sha256": artifact_sha_r1},
               "R1_artifact_sha256": artifact_sha_r1,
               "n_excluded": len(kept),
               "score_s_R1": r1_scored["score_s"],
               "quantize_s_R1": qrec_r1.get("quantize_s")}
        rows.append(row)
        (WORK / f"seed{seed}_R1.json").write_text(json.dumps(row, indent=2))
        (WORK / f"seed{seed}_R0.json").write_text(json.dumps(
            {"seed": seed, "e_fp32": e_f, "e_R0": e_0, "G_seed": gap,
             "metrics_fp32": d["fp32_scored"]["metrics"],
             "metrics_R0": d["R0_scored"]["metrics"],
             "guard_fp32": d["fp32_scored"]["guard"],
             "guard_R0": d["R0_scored"]["guard"],
             "R0_quantize": d["R0_quantize"]}, indent=2))
        print(f"seed {seed}: R1 EPE {e_1:.4f} R_R1={r_r1:.3f} resid={resid:+.4f} "
              f"(onnx deleted)", flush=True)

    # --- §7 mechanical verdict, thresholds read from report["thresholds"] ---
    th = report["thresholds"]
    r_acc, n_req, resid_max = th["R_ACCEPT"], th["N_SEEDS_REQUIRED"], th["RESID_MAX_PX"]
    for r in rows:
        r["pass_R"] = r["R_R1"] is not None and r["R_R1"] >= r_acc
        r["pass_resid"] = r["delta_R1_vs_fp32"] <= resid_max
        r["pass_seed"] = bool(r["pass_R"] and r["pass_resid"])
    n_pass = sum(1 for r in rows if r["pass_seed"])
    report["measured"]["rows"] = rows
    report["measured"]["n_pass"] = n_pass
    if n_pass >= n_req:
        verdict = (f"readout-localized (diagnostic label only, no fix claim, no Hailo "
                   f"implication): R_R1 >= {r_acc} and residual <= {resid_max}px "
                   f"on {n_pass}/3 seeds (required {n_req})")
    elif n_pass == 0:
        verdict = (f"FALSIFIED §8(a): R_R1 below R_ACCEPT={r_acc} (or residual breach) "
                   f"on all three seeds")
    elif rows[0]["pass_seed"] and not rows[1]["pass_seed"] and not rows[2]["pass_seed"]:
        verdict = (f"FALSIFIED §8(b): at/above threshold on seed 0 only "
                   f"(R_R1={rows[0]['R_R1']:.3f}) but below on seeds 1 and 2 — "
                   f"seed-specific, hypothesis as stated falsified")
    else:
        failing = [r["seed"] for r in rows if not r["pass_seed"]]
        verdict = (f"FALSIFIED: below threshold on seeds {failing} "
                   f"({n_pass}/3 pass, required {n_req}); hypothesis as stated "
                   f"(general readout-localized gap) falsified")
    report["measured"]["verdict"] = verdict
    report["inferred"]["note"] = (
        "Verdict is mechanical application of the user-set §7 thresholds. "
        "Ranking is descriptive under THIS quantizer; it does not identify a fix "
        "and bears no Hailo implication. §8(c) (C0-level comparison) not tested "
        "in this run — no all-non-Conv-fp32 control was scored here.")
    report["unknown"]["items"] = [
        "Mechanism behind any readout difference (out of scope, mechanism unknown).",
        "Whether results transfer to other procedures or hardware.",
        "§8(c) C0-level comparison (no C0 arm in this run).",
    ]
    report["status"] = "COMPLETE: all 9 scorings (3 fp32 + 3 R0 + 3 R1) on the 40-scene contract"
    OUT.write_text(json.dumps(report, indent=2))
    print(f"COMPLETE n_pass={n_pass}/3 verdict={verdict} wrote {OUT}")


if __name__ == "__main__":
    main()
