#!/usr/bin/env python
"""INT8 head-plus-refinement-fp32 recovery — E3 seed-0/1/2 (pre-registered diagnostic).

Implements docs/superpowers/specs/2026-09-24-int8-head-plus-refinement-fp32-design.md
exactly. Pre-registered BEFORE measurement (spec commit da64722 scope).

Subjects: reuse stage_e_recipe/int8_e3/e3_seed{0,1,2}_best_fp32.onnx (provenance
asserted per seed against int8_e3.json). Quantizer: exact int8_control.py
procedure via int8_sensitivity.py (imported, not copied — see REUSE_METHOD).
Node sets enumerated at runtime by name prefix: /regression/* (head),
/refinement/* (refinement). Arms per seed: R0 all-int8 baseline (fresh),
R1 head fp32 (must reproduce bbe37fd within 1e-6), R2 head+refinement fp32
(PRIMARY), R3 refinement-only fp32 (ATTRIBUTION, no verdict), plus fp32
re-score. Scoring: frozen 40-scene contract, contract_match required.
Gates: §6 baseline+R1 reproduction (1e-6 px, STOP rule) and §3 no-op
precondition per kept set (STOP rule). Budget §9: T_est = 12xS1.

Outputs ONLY in stage_e_recipe/int8_head_refinement/. Each quantized .onnx is
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
import int8_regression_head as RH  # noqa: E402

# Reuse method (spec: say which): IMPORT from both precedent modules.
# - from stage_e_recipe/int8_sensitivity.py: quantizer
#   (S.quantize_with_excludes), scoring (S.score_contract), sha/git helpers
#   (S.sha256, S.git_head), calibration constant (S.CALIB_N) — unmodified.
# - from stage_e_recipe/int8_regression_head.py: kept-set enumeration
#   (RH.kept_set, /regression/* prefix logic) and no-op coverage audit
#   (RH.coverage_audit, generic over any kept list) — unmodified, called
#   for head / refinement / combined sets.
# No copies of quantizer or scoring logic; node-set helpers reused by import.
REUSE_METHOD = (
    "import from stage_e_recipe/int8_sensitivity.py "
    "(quantizer, scoring, sha/git, CALIB_N; no copy, no changes) and "
    "import from stage_e_recipe/int8_regression_head.py "
    "(kept_set enumeration + coverage_audit; no copy, no changes)"
)

WORK = HERE / "int8_head_refinement"
OUT = WORK / "int8_head_refinement.json"
BUDGET_S = 2 * 3600
REPRO_TOL = 1e-6

# Inherited threshold block (§7, inherited from 6a6cf6b, frozen — the SAME
# frozen thresholds the user set for the previous experiment, carried over
# unchanged). Written at the top of the output JSON; the verdict below reads
# report["thresholds"] mechanically and never chooses values.
THRESHOLDS = {
    "R_ACCEPT": 0.50,
    "N_SEEDS_REQUIRED": 3,
    "RESID_MAX_PX": 0.50,
    "inherited_from": "6a6cf6b",
    "frozen": True,
    "statement": "inherited from 6a6cf6b, frozen",
    "semantics": {
        "R_ACCEPT": "per-seed recovered fraction R_R2 must be >= 0.50",
        "N_SEEDS_REQUIRED": "must hold on all 3 E3 seeds",
        "RESID_MAX_PX": "EPE_R2 - EPE_fp32 must be <= 0.50 px on each of those seeds",
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
        "ref_R1_epe": 1.7021716010157655,
    },
    1: {
        "fp32_onnx": "stage_e_recipe/int8_e3/e3_seed1_best_fp32.onnx",
        "sha256": "9b4922c91367a2c41254795e0c7160dcb05313f7fb948e2c0d56f05d9ebe9ac2",
        "bytes": 1701550, "opset": 13, "n_nodes": 712,
        "parent_checkpoint_sha256": "133c19c41585445da9aabf18919e62f5805b5f8abcf80232808d26d33aa293b0",
        "params": 397954,
        "ref_fp32_epe": 1.178821279341308,
        "ref_int8_epe": 5.3849187722115985,
        "ref_R1_epe": 1.5605500813958062,
    },
    2: {
        "fp32_onnx": "stage_e_recipe/int8_e3/e3_seed2_best_fp32.onnx",
        "sha256": "7c0c16e4cf66a7ca14e59f5be31801f19815ae1ba6ca72014cb7dd3e99e50418",
        "bytes": 1701550, "opset": 13, "n_nodes": 712,
        "parent_checkpoint_sha256": "1ba8ca09bfabf2c689b6ec7fbbbae1aa70367d33562b1aad06521ba71904cbcf",
        "params": 397954,
        "ref_fp32_epe": 1.1713166599983933,
        "ref_int8_epe": 5.87156272118187,
        "ref_R1_epe": 1.7001293943249853,
    },
}

REQUIRED_MUL = ["/regression/Mul", "/regression/Mul_1", "/regression/Mul_2"]
REQUIRED_REFIN = ["/refinement/input_conv/Conv", "/refinement/output_conv/Conv"]


def fail(reason: str, report: dict) -> "NoReturn":
    report["status"] = f"STOP: {reason}"
    OUT.write_text(json.dumps(report, indent=2))
    for p in sorted(WORK.glob("*.onnx")):
        p.unlink(missing_ok=True)
    print(f"STOP: {reason} (partial report -> {OUT})", flush=True)
    raise SystemExit(1)


def prefix_set(fp32_path: Path, prefix: str) -> list[str]:
    import onnx

    m = onnx.load(str(fp32_path))
    names = sorted(n.name for n in m.graph.node if n.name.startswith(prefix))
    return names, len(m.graph.node)


def main() -> None:
    from src.datasets.kitti2015 import Kitti2015Stereo  # noqa: E402

    WORK.mkdir(parents=True, exist_ok=True)
    try:
        import onnxruntime as ort
        ort_v = ort.__version__
    except Exception as e:
        sys.exit(f"INT8 head+refinement NOT MEASURABLE: {e}")

    report: dict = {
        "spec": "docs/superpowers/specs/2026-09-24-int8-head-plus-refinement-fp32-design.md",
        "spec_pre_register_commit": "da64722",
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

    # --- §1 subject assertions + §3 node-set enumeration (per seed) ---
    head_per_seed: dict[str, list[str]] = {}
    refin_per_seed: dict[str, list[str]] = {}
    for seed in (0, 1, 2):
        cfg = SEEDS[seed]
        fp32 = REPO / cfg["fp32_onnx"]
        assert fp32.exists(), f"missing subject {fp32}"
        got_sha = S.sha256(fp32)
        assert got_sha == cfg["sha256"], (
            f"seed {seed} fp32 sha mismatch: {got_sha} != recorded {cfg['sha256']}")
        head, n_nodes = prefix_set(fp32, "/regression/")
        # cross-check head enumeration against the precedent helper (imported)
        head_rh, n_rh = RH.kept_set(fp32)
        assert head == head_rh and n_nodes == n_rh == cfg["n_nodes"] == 712, (
            f"seed {seed} head-set mismatch vs int8_regression_head.kept_set "
            f"or n_nodes {n_nodes}/{n_rh} != recorded 712")
        refin, n2 = prefix_set(fp32, "/refinement/")
        assert n2 == cfg["n_nodes"] == 712, (
            f"seed {seed} n_nodes {n2} != recorded 712")
        assert head, f"seed {seed}: empty /regression/* set"
        assert refin, f"seed {seed}: empty /refinement/* set"
        for m in REQUIRED_MUL:
            if m not in head:
                fail(f"seed {seed}: required head node {m} missing", report)
        for m in REQUIRED_REFIN:
            if m not in refin:
                fail(f"seed {seed}: required refinement node {m} missing", report)
        head_per_seed[str(seed)] = head
        refin_per_seed[str(seed)] = refin
        print(f"seed {seed} OK: sha match, {n_nodes} nodes, "
              f"{len(head)} /regression/*, {len(refin)} /refinement/*",
              flush=True)
    report["measured"]["head_set_per_seed"] = head_per_seed
    report["measured"]["refinement_set_per_seed"] = refin_per_seed
    if len({tuple(v) for v in head_per_seed.values()}) > 1:
        report["measured"]["head_set_note"] = (
            "head sets differ across seeds; recorded all three, proceeding per seed")
    else:
        report["measured"]["head_set_note"] = "identical /regression/* set on all 3 seeds"
    if len({tuple(v) for v in refin_per_seed.values()}) > 1:
        report["measured"]["refinement_set_note"] = (
            "refinement sets differ across seeds; recorded all three, proceeding per seed")
    else:
        report["measured"]["refinement_set_note"] = (
            "identical /refinement/* set on all 3 seeds")

    # --- per-seed R0 quantize + fp32 score; §9 budget on first config ---
    seeds_data: dict[str, dict] = {}
    s1: float | None = None
    for seed in (0, 1, 2):
        cfg = SEEDS[seed]
        fp32 = REPO / cfg["fp32_onnx"]
        head = head_per_seed[str(seed)]
        refin = refin_per_seed[str(seed)]
        combined = sorted(set(head) | set(refin))
        dst_r0 = WORK / f"seed{seed}_R0_all_int8.onnx"

        t0 = time.time()
        qrec_r0 = S.quantize_with_excludes(fp32, dst_r0, calib, None)
        artifact_sha_r0 = S.sha256(dst_r0)

        # --- §3 no-op preconditions (before scoring), via imported audit ---
        for label, kept in (("head", head), ("refinement", refin),
                            ("combined", combined)):
            audit = RH.coverage_audit(dst_r0, kept)
            audit["fp32_parent_sha256"] = cfg["sha256"]
            audit["r0_differs_from_fp32"] = audit["baseline_sha256"] != cfg["sha256"]
            audit["set_label"] = label
            print(f"seed {seed} coverage {label}: "
                  f"{audit['n_kept_consuming_dql']}/{audit['n_kept']} kept "
                  f"nodes consume DQL in R0", flush=True)
            report["measured"][f"coverage_audit_seed{seed}_{label}"] = audit
            if audit["n_kept_consuming_dql"] == 0:
                dst_r0.unlink(missing_ok=True)
                fail(f"seed {seed}: §3 no-op precondition failed for {label} set — "
                     f"kept set carries no QDQ in the R0 baseline; arm cannot "
                     f"move the score", report)

        fp32_scored = S.score_contract(fp32, val)
        r0_scored = S.score_contract(dst_r0, val)
        dt = time.time() - t0
        dst_r0.unlink(missing_ok=True)  # delete .onnx as soon as scored

        if s1 is None:
            s1 = dt
            t_est = 12 * s1
            report["timing"] = {"S1_first_config_s": round(s1, 1),
                                "T_est_s": round(t_est, 1),
                                "T_est_h": round(t_est / 3600, 2),
                                "budget_s": BUDGET_S,
                                "branch": ("proceed"
                                           if t_est <= BUDGET_S else "STOP over budget")}
            print(f"budget: S1 {s1:.1f}s x12 -> est {t_est / 3600:.2f}h", flush=True)
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

    # --- R1/R2/R3 arms per seed (quantize + score, delete immediately) ---
    for seed in (0, 1, 2):
        cfg = SEEDS[seed]
        fp32 = REPO / cfg["fp32_onnx"]
        head = head_per_seed[str(seed)]
        refin = refin_per_seed[str(seed)]
        combined = sorted(set(head) | set(refin))
        d = seeds_data[str(seed)]
        for arm, kept, tag in (("R1", head, "regression_fp32"),
                               ("R2", combined, "head_refinement_fp32"),
                               ("R3", refin, "refinement_fp32")):
            dst = WORK / f"seed{seed}_{arm}_{tag}.onnx"
            qrec = S.quantize_with_excludes(fp32, dst, calib, kept)
            artifact_sha = S.sha256(dst)
            scored = S.score_contract(dst, val)
            dst.unlink(missing_ok=True)  # delete .onnx as soon as scored
            d[f"{arm}_scored"] = scored
            d[f"{arm}_quantize"] = {**qrec, "sha256": artifact_sha}
            d[f"{arm}_artifact_sha256"] = artifact_sha
            d[f"{arm}_n_excluded"] = len(kept)
            print(f"seed {seed} {arm}: EPE {scored['metrics']['epe']:.4f} "
                  f"(onnx deleted)", flush=True)

    # --- §6 baseline + R1 reproduction gate (STOP rule, all seeds) ---
    gate_rows = {}
    gate_ok_all = True
    for seed in (0, 1, 2):
        cfg = SEEDS[seed]
        d = seeds_data[str(seed)]
        e_f = d["fp32_scored"]["metrics"]["epe"]
        e_0 = d["R0_scored"]["metrics"]["epe"]
        e_1 = d["R1_scored"]["metrics"]["epe"]
        ok_f = abs(e_f - cfg["ref_fp32_epe"]) <= REPRO_TOL
        ok_0 = abs(e_0 - cfg["ref_int8_epe"]) <= REPRO_TOL
        ok_1 = abs(e_1 - cfg["ref_R1_epe"]) <= REPRO_TOL
        ok_g = (d["fp32_scored"]["guard"]["contract_match"] is True
                and d["R0_scored"]["guard"]["contract_match"] is True
                and d["R1_scored"]["guard"]["contract_match"] is True
                and d["R2_scored"]["guard"]["contract_match"] is True
                and d["R3_scored"]["guard"]["contract_match"] is True)
        gate_rows[str(seed)] = {"fp32_ok": ok_f, "R0_ok": ok_0, "R1_ok": ok_1,
                                "guards_ok": ok_g, "e_fp32": e_f, "e_R0": e_0,
                                "e_R1": e_1}
        if not (ok_f and ok_0 and ok_1 and ok_g):
            gate_ok_all = False
    report["measured"]["baseline_gate_per_seed"] = gate_rows
    if not gate_ok_all:
        report["measured"]["seeds_baselines"] = {
            s: {"fp32": v["fp32_scored"], "R0": v["R0_scored"],
                "R1": v["R1_scored"], "R2": v["R2_scored"], "R3": v["R3_scored"]}
            for s, v in seeds_data.items()}
        report["baseline_gate"] = "STOP: reproduction failed"
        fail("baseline+R1 reproduction gate failed (§6); see baseline_gate_per_seed",
             report)
    report["baseline_gate"] = (
        "PASS: fp32/R0 reproduction within 1e-6 px vs int8_e3.json and R1 "
        "within 1e-6 px vs bbe37fd, guards true (all seeds)")

    # --- §7 rows + mechanical verdict (thresholds from report["thresholds"]) ---
    th = report["thresholds"]
    r_acc, n_req, resid_max = th["R_ACCEPT"], th["N_SEEDS_REQUIRED"], th["RESID_MAX_PX"]
    rows = []
    for seed in (0, 1, 2):
        d = seeds_data[str(seed)]
        e_f = d["fp32_scored"]["metrics"]["epe"]
        e_0 = d["R0_scored"]["metrics"]["epe"]
        e_1 = d["R1_scored"]["metrics"]["epe"]
        e_2 = d["R2_scored"]["metrics"]["epe"]
        e_3 = d["R3_scored"]["metrics"]["epe"]
        gap = e_0 - e_f
        r_r1 = (e_0 - e_1) / gap if gap else None
        r_r2 = (e_0 - e_2) / gap if gap else None
        r_r3 = (e_0 - e_3) / gap if gap else None
        resid_r1 = e_1 - e_f
        resid_r2 = e_2 - e_f
        row = {"seed": seed,
               "e_fp32": e_f, "e_R0": e_0, "e_R1": e_1, "e_R2": e_2, "e_R3": e_3,
               "G_seed": gap,
               "R_R1": r_r1, "R_R2": r_r2, "R_R3": r_r3,
               "resid_R1": resid_r1, "resid_R2": resid_r2,
               "delta_R1_to_R2": e_2 - e_1,
               "delta_resid_R1_to_R2": resid_r2 - resid_r1,
               "r2_below_r1": bool(resid_r2 < resid_r1),
               "d1_fp32": d["fp32_scored"]["metrics"]["d1"],
               "d1_R0": d["R0_scored"]["metrics"]["d1"],
               "d1_R1": d["R1_scored"]["metrics"]["d1"],
               "d1_R2": d["R2_scored"]["metrics"]["d1"],
               "d1_R3": d["R3_scored"]["metrics"]["d1"],
               "nonfinite_fp32": d["fp32_scored"]["nonfinite_pixels"],
               "nonfinite_R0": d["R0_scored"]["nonfinite_pixels"],
               "nonfinite_R1": d["R1_scored"]["nonfinite_pixels"],
               "nonfinite_R2": d["R2_scored"]["nonfinite_pixels"],
               "nonfinite_R3": d["R3_scored"]["nonfinite_pixels"],
               "contract_match": {
                   "fp32": d["fp32_scored"]["guard"]["contract_match"],
                   "R0": d["R0_scored"]["guard"]["contract_match"],
                   "R1": d["R1_scored"]["guard"]["contract_match"],
                   "R2": d["R2_scored"]["guard"]["contract_match"],
                   "R3": d["R3_scored"]["guard"]["contract_match"]},
               "metrics_fp32": d["fp32_scored"]["metrics"],
               "metrics_R0": d["R0_scored"]["metrics"],
               "metrics_R1": d["R1_scored"]["metrics"],
               "metrics_R2": d["R2_scored"]["metrics"],
               "metrics_R3": d["R3_scored"]["metrics"],
               "R1_quantize": d["R1_quantize"],
               "R2_quantize": d["R2_quantize"],
               "R3_quantize": d["R3_quantize"],
               "R1_artifact_sha256": d["R1_artifact_sha256"],
               "R2_artifact_sha256": d["R2_artifact_sha256"],
               "R3_artifact_sha256": d["R3_artifact_sha256"],
               "n_excluded_R1": d["R1_n_excluded"],
               "n_excluded_R2": d["R2_n_excluded"],
               "n_excluded_R3": d["R3_n_excluded"]}
        row["pass_R2_R"] = r_r2 is not None and r_r2 >= r_acc
        row["pass_R2_resid"] = resid_r2 <= resid_max
        row["pass_R2_seed"] = bool(row["pass_R2_R"] and row["pass_R2_resid"])
        rows.append(row)
        (WORK / f"seed{seed}_R2.json").write_text(json.dumps(row, indent=2))
        (WORK / f"seed{seed}_arms.json").write_text(json.dumps(
            {"seed": seed, "e_fp32": e_f, "e_R0": e_0, "e_R1": e_1,
             "e_R2": e_2, "e_R3": e_3, "G_seed": gap,
             "R_R1": r_r1, "R_R2": r_r2, "R_R3": r_r3,
             "resid_R1": resid_r1, "resid_R2": resid_r2}, indent=2))
        print(f"seed {seed}: R2 EPE {e_2:.4f} R_R2={r_r2:.3f} "
              f"resid={resid_r2:+.4f} | R3 EPE {e_3:.4f} R_R3={r_r3:.3f}",
              flush=True)

    n_pass = sum(1 for r in rows if r["pass_R2_seed"])
    all_r2_below_r1 = all(r["r2_below_r1"] for r in rows)
    report["measured"]["rows"] = rows
    report["measured"]["n_pass_R2"] = n_pass
    report["measured"]["r2_below_r1_all_seeds"] = all_r2_below_r1
    if n_pass >= n_req:
        verdict = (f"SUPPORTED: R_R2 >= {r_acc} and residual <= {resid_max}px "
                   f"on {n_pass}/3 seeds (required {n_req})")
    else:
        failing = [r["seed"] for r in rows if not r["pass_R2_seed"]]
        verdict = (f"FALSIFIED: below threshold on seeds {failing} "
                   f"({n_pass}/3 pass, required {n_req}); hypothesis as stated "
                   f"(head+refinement fp32 meets the criterion) falsified")
    report["measured"]["verdict"] = verdict
    report["inferred"]["note"] = (
        "Verdict is mechanical application of the inherited §7 thresholds to "
        "the PRIMARY R2 arm. R3 is attribution only (no verdict). Ranking is "
        "descriptive under THIS quantizer; it does not identify a fix and "
        "bears no Hailo implication.")
    report["unknown"]["items"] = [
        "Mechanism behind any head/refinement difference (out of scope, mechanism unknown).",
        "Whether results transfer to other procedures or hardware.",
    ]
    report["status"] = ("COMPLETE: all 15 scorings (3 fp32 + 12 quantized: "
                        "3 R0 + 3 R1 + 3 R2 + 3 R3) on the 40-scene contract")
    OUT.write_text(json.dumps(report, indent=2))
    print(f"COMPLETE n_pass_R2={n_pass}/3 verdict={verdict} wrote {OUT}")


if __name__ == "__main__":
    main()
