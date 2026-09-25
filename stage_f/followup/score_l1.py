"""Stage-F L1 continuation pilot scoring: score the L1 pilot (2 arms) and apply the frozen gate.

Zero-compute: local inference only, trains nothing.

Reuse (imported, never reimplemented here):
  - score_f1 (stage_f/followup/score_f1.py — itself unedited): the E3 seed0
    FINAL sanity references (REF_CONTRACT_EPE / REF_TRAIN_GT64 / REF_TRAIN_EPE /
    SANITY_TOL) and score_arm (both L1 arms are width-32, so the unmodified
    f1 loader applies directly).
  - stage_f/audit_e3.py (via score_f1's imports): sha256_file, StereoNet /
    StereoNetConfig, ev.score (frozen 40-scene hailo_val contract), git_head,
    atomic_write_json.

Sanity (mandatory, runs first): score_f1's pipeline on E3 seed0 FINAL must
reproduce the audit2 contract final EPE within 1e-4 and the audit_e3 train
GT<64 EPE within 1e-4 (the same check as score_f1 / score_twin). On failure
the script writes the scores JSON with sanity FAIL, writes a NO_VERDICT
verdict file, and exits 1 — no verdict is produced.

Gate (frozen, design section 8, launch note section 5):
  dfinal = contract_final_EPE(l1m) - contract_final_EPE(l1c)
  PASS iff dfinal <= -0.030 px, else FAIL.
Kill (section 9, pilot-applicable): NaN/Inf, zero valid pixels, strict-load
failure. A single-seed PASS only admits to 3-seed validation.

Each arm's delta vs the E3 seed0 FINAL starting point (contract
1.1628882757801255) is reported for information only; it does not enter
the gate.

Usage (from repo root, with the project python):
  python stage_f/followup/score_l1.py [--device cuda|cpu]
Outputs stage_f/followup/l1_pilot/{l1_scores.json, l1_verdict.json,
L1_PILOT_REPORT.md, run.log}.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stage_f"))
sys.path.insert(0, str(REPO / "stage_f" / "followup"))
import audit_e3 as a3  # noqa: E402
import score_f1 as f1  # noqa: E402  (reused, never edited)

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT_DIR = REPO / "stage_f" / "followup" / "l1_pilot"

ARMS = {
    "l1c": {
        "label": "Smooth-L1 control (E0 loss, unchanged)",
        "ckpt": (REPO / "stage_f" / "followup" / "kaggle" / "l1c_output"
                 / "seed0" / "l1c_seed0_final.pth"),
        "owner": "vishnuvardhanksks",
    },
    "l1m": {
        "label": "masked L1",
        "ckpt": (REPO / "stage_f" / "followup" / "kaggle" / "l1m_output"
                 / "seed0" / "l1m_seed0_final.pth"),
        "owner": "vishnuvardhanksks",
    },
}

ENV_DIFFERENCE = (
    "Declared environment sameness (per L1 launch note section 4): l1c and l1m "
    "both run on Kaggle account vishnuvardhanksks — same account, no "
    "cross-account difference. Same GPU type (T4), same continuation "
    "recipe/seed/epochs (E3 seed0 FINAL init, 40 epochs, fresh Adam lr 1e-4, "
    "cosine to 0); the arms differ ONLY in the loss knob (Smooth-L1 vs "
    "masked L1)."
)

# Frozen references live in score_f1 (same check as score_f1 / score_twin).
REF_CONTRACT_EPE = f1.REF_CONTRACT_EPE
REF_TRAIN_GT64 = f1.REF_TRAIN_GT64
REF_TRAIN_EPE = f1.REF_TRAIN_EPE
SANITY_TOL = f1.SANITY_TOL

# E3 seed0 FINAL starting point (continuation init), information only.
START_CONTRACT_EPE = 1.1628882757801255

PASS_DFINAL = -0.030


def finite_or_none(x) -> bool:
    return x is not None and math.isfinite(x)


def apply_verdict(arms: dict) -> dict:
    """Frozen L1 gate: dfinal = contract_final_EPE(l1m) - contract_final_EPE(l1c)."""
    c, m = arms["l1c"], arms["l1m"]
    dfinal = m["contract"]["final_epe"] - c["contract"]["final_epe"]
    nums = [m["contract"]["final_epe"], m["contract"]["d1"],
            m["contract"]["gtlt64_epe"], m["train"]["epe"],
            m["train"]["gtlt64_epe"],
            c["contract"]["final_epe"], c["contract"]["d1"],
            c["contract"]["gtlt64_epe"], c["train"]["epe"],
            c["train"]["gtlt64_epe"]]
    kill = None
    if not (m["strict_load"] and c["strict_load"]):
        bad = [k for k in ("l1c", "l1m") if not arms[k]["strict_load"]]
        kill = ("strict-load failure on " + ",".join(bad) + ": "
                + "; ".join(f"{k}={arms[k]['strict_error']}" for k in bad))
    elif not all(finite_or_none(x) for x in nums):
        kill = "NaN/Inf in scored numbers"
    elif (m["contract"]["valid_pixels"] == 0 or m["train"]["valid_pixels"] == 0
          or c["contract"]["valid_pixels"] == 0
          or c["train"]["valid_pixels"] == 0):
        kill = "zero valid pixels"
    # Information only: delta vs the E3 seed0 FINAL continuation start.
    info = {
        "l1c_vs_start": c["contract"]["final_epe"] - START_CONTRACT_EPE,
        "l1m_vs_start": m["contract"]["final_epe"] - START_CONTRACT_EPE,
        "start_contract_final_epe": START_CONTRACT_EPE,
    }
    reasons = [
        f"dfinal=contract_final_EPE(l1m)-contract_final_EPE(l1c)={dfinal:.6f} "
        f"(l1m {m['contract']['final_epe']:.6f} vs "
        f"l1c {c['contract']['final_epe']:.6f})",
    ]
    if kill is not None:
        return {"dfinal": dfinal, "result": "KILL",
                "reasons": [kill] + reasons, "info_vs_start": info}
    if dfinal <= PASS_DFINAL:
        return {"dfinal": dfinal, "result": "PASS",
                "reasons": reasons + ["dfinal <= -0.030"],
                "info_vs_start": info}
    return {"dfinal": dfinal, "result": "FAIL",
            "reasons": reasons + ["dfinal > -0.030"],
            "info_vs_start": info}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUT_DIR / "run.log"
    log_fh = open(log_path, "w")  # noqa: PTH123

    class Tee:
        def __init__(self, *fhs):
            self.fhs = fhs

        def write(self, s):
            for fh in self.fhs:
                fh.write(s)

        def flush(self):
            for fh in self.fhs:
                fh.flush()

    sys.stdout = Tee(sys.stdout, log_fh)
    try:
        _run(device)
    finally:
        sys.stdout = sys.stdout.fhs[0]
        log_fh.close()


def _run(device: str) -> None:
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("plan: stage-f L1 continuation pilot scoring (l1c vs l1m); "
          f"device={device}", flush=True)

    # ---- mandatory sanity check on E3 seed0 FINAL (score_f1's pipeline) ----
    e3_ckpt = a3.ckpt_path(0, "final")
    print(f"=== sanity: E3 seed0 FINAL {e3_ckpt} ===", flush=True)
    sanity_rec = f1.score_arm("e3_sanity", e3_ckpt, device)
    d_contract = sanity_rec["contract"]["final_epe"] - REF_CONTRACT_EPE
    d_train64 = sanity_rec["train"]["gtlt64_epe"] - REF_TRAIN_GT64
    d_train = sanity_rec["train"]["epe"] - REF_TRAIN_EPE
    sanity_ok = abs(d_contract) <= SANITY_TOL and abs(d_train64) <= SANITY_TOL
    sanity = {
        "ckpt": str(e3_ckpt),
        "sha256": sanity_rec["sha256"],
        "recomputed_contract_final_epe": sanity_rec["contract"]["final_epe"],
        "reference_contract_final_epe": REF_CONTRACT_EPE,
        "reference_source": "stage_f/followup/audit2/audit2.json seed0_final.epe "
                            "(== e3_verdict.json per_seed_final[0]; via score_f1)",
        "diff_contract": d_contract,
        "recomputed_train_gtlt64_epe": sanity_rec["train"]["gtlt64_epe"],
        "reference_train_gtlt64_epe": REF_TRAIN_GT64,
        "reference_train_source": "stage_f/audit/audit_e3.json "
                                  "seed0_final.train_split.gtlt64_epe (via score_f1)",
        "diff_train_gtlt64": d_train64,
        "recomputed_train_epe": sanity_rec["train"]["epe"],
        "reference_train_epe": REF_TRAIN_EPE,
        "diff_train_epe": d_train,
        "tolerance": SANITY_TOL,
        "status": "PASS" if sanity_ok else "FAIL",
    }
    print(f"sanity contract EPE={sanity_rec['contract']['final_epe']:.10f} "
          f"ref={REF_CONTRACT_EPE:.10f} diff={d_contract:.3e}", flush=True)
    print(f"sanity train GT<64={sanity_rec['train']['gtlt64_epe']:.10f} "
          f"ref={REF_TRAIN_GT64:.10f} diff={d_train64:.3e}", flush=True)
    print(f"sanity status: {sanity['status']}", flush=True)

    prov = {
        "git_head": a3.git_head(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "numpy": np.__version__,
        "device": device,
        "env_difference": ENV_DIFFERENCE,
        "utc": utc,
    }

    arms: dict = {}
    if sanity_ok:
        for name in ("l1c", "l1m"):
            arms[name] = f1.score_arm(name, ARMS[name]["ckpt"], device)
            arms[name]["owner_account"] = ARMS[name]["owner"]
            arms[name]["label"] = ARMS[name]["label"]

    scores = {
        "experiment": "stage-f L1 continuation pilot scoring (l1c Smooth-L1 "
                      "control vs l1m masked L1; seed0 FINAL checkpoints)",
        "provenance": prov,
        "sanity": sanity,
        "arms": arms,
        "thresholds": {"PASS_DFINAL": PASS_DFINAL,
                       "SANITY_TOL": SANITY_TOL},
        "start_point": {
            "note": "E3 seed0 FINAL continuation init; information only, "
                    "not part of the gate",
            "contract_final_epe": START_CONTRACT_EPE,
        },
    }
    a3.atomic_write_json(OUT_DIR / "l1_scores.json", scores)

    if not sanity_ok:
        nov = {"status": "NO_VERDICT_SANITY_FAIL",
               "reason": "E3 seed0 FINAL sanity check failed; no gate verdict "
                         "produced (see l1_scores.json sanity diffs).",
               "sanity": sanity}
        a3.atomic_write_json(OUT_DIR / "l1_verdict.json", nov)
        _write_report(scores, None)
        print("STOP: sanity check failed; no verdict produced", flush=True)
        raise SystemExit(1)

    body = apply_verdict(arms)
    if body["result"] == "PASS":
        admission = ("Single-seed PASS: admitted to 3-seed validation only, "
                     "never acceptance.")
    elif body["result"] == "FAIL":
        admission = "Single-seed FAIL: nothing is admitted to 3-seed validation."
    else:
        admission = "KILL: STOP, no verdict."
    verdict = {
        "status": "OK",
        "experiment": "stage-f L1 continuation pilot frozen gate "
                      "(l1m contract final EPE vs l1c)",
        "gates": {"PASS": "dfinal <= -0.030",
                  "FAIL": "otherwise",
                  "KILL": "NaN/Inf, zero valid px, strict-load fail",
                  "note": "deltas vs the E3 seed0 FINAL start are information "
                          "only and do not enter the gate"},
        "dfinal": body["dfinal"],
        "result": body["result"],
        "reasons": body["reasons"],
        "info_vs_start": body["info_vs_start"],
        "admission": admission,
        "env_difference": ENV_DIFFERENCE,
        "provenance": prov,
    }
    a3.atomic_write_json(OUT_DIR / "l1_verdict.json", verdict)
    _write_report(scores, verdict)
    print(f"wrote {OUT_DIR / 'l1_scores.json'} + l1_verdict.json + "
          f"L1_PILOT_REPORT.md status=OK result={verdict['result']}",
          flush=True)


def _write_report(scores: dict, verdict: dict | None) -> None:
    arms = scores["arms"]
    san = scores["sanity"]
    lines = ["# L1 pilot report (Stage F continuation, seed 0)",
             "",
             "[FACT] Checkpoints scored (FINAL only, seed 0):",
             f"- control l1c: `{ARMS['l1c']['ckpt'].name}` "
             "(Smooth-L1 control)",
             f"- masked L1 l1m: `{ARMS['l1m']['ckpt'].name}`",
             "- Both width 32, 397954 params (same architecture).",
             "",
             "[FACT] " + ENV_DIFFERENCE,
             "",
             "## Sanity check (E3 seed0 FINAL, mandatory)",
             "",
             "[MEASUREMENT] "
             f"recomputed contract final EPE = {san['recomputed_contract_final_epe']:.10f} "
             f"(reference {san['reference_contract_final_epe']:.10f}, "
             f"diff {san['diff_contract']:.3e}, tol {san['tolerance']})",
             "[MEASUREMENT] "
             f"recomputed train GT<64 EPE = {san['recomputed_train_gtlt64_epe']:.10f} "
             f"(reference {san['reference_train_gtlt64_epe']:.10f}, "
             f"diff {san['diff_train_gtlt64']:.3e}, tol {san['tolerance']})",
             f"[VERDICT] sanity status: {san['status']}",
             "",
             ]
    if verdict is None:
        lines += ["## Verdict",
                  "",
                  "[VERDICT] NO_VERDICT_SANITY_FAIL — sanity check failed; "
                  "no gate verdict produced.",
                  ""]
    else:
        lines += ["## Numbers",
                  "",
                  "| arm | contract final EPE | contract D1 | contract GT<64 EPE | "
                  "train EPE | train GT<64 EPE |",
                  "|---|---|---|---|---|---|"]
        for name in ("l1c", "l1m"):
            a = arms[name]
            c, t = a["contract"], a["train"]
            lines.append(
                f"| {name} | {c['final_epe']:.6f} | {c['d1']:.4f} | "
                f"{c['gtlt64_epe']:.6f} | {t['epe']:.6f} | {t['gtlt64_epe']:.6f} |")
        lines += ["",
                  "[MEASUREMENT] per-arm sha256 / strict-load / params:",
                  f"- l1c sha256={arms['l1c']['sha256']} "
                  f"strict_load={arms['l1c']['strict_load']} "
                  f"params={arms['l1c']['params']}",
                  f"- l1m sha256={arms['l1m']['sha256']} "
                  f"strict_load={arms['l1m']['strict_load']} "
                  f"params={arms['l1m']['params']}",
                  "",
                  "## Gate outcome (frozen section 8)",
                  "",
                  f"[MEASUREMENT] dfinal={verdict['dfinal']:.6f} "
                  f"(l1m {arms['l1m']['contract']['final_epe']:.6f} vs "
                  f"l1c {arms['l1c']['contract']['final_epe']:.6f})",
                  f"[VERDICT] {verdict['result']} — "
                  + "; ".join(verdict["reasons"]),
                  f"[VERDICT] {verdict['admission']}",
                  "",
                  "## Information only (not part of the gate)",
                  "",
                  "[MEASUREMENT] "
                  f"l1c vs E3 seed0 FINAL start: "
                  f"{verdict['info_vs_start']['l1c_vs_start']:.6f} "
                  f"(l1c {arms['l1c']['contract']['final_epe']:.6f} vs start "
                  f"{verdict['info_vs_start']['start_contract_final_epe']:.6f})",
                  "[MEASUREMENT] "
                  f"l1m vs E3 seed0 FINAL start: "
                  f"{verdict['info_vs_start']['l1m_vs_start']:.6f} "
                  f"(l1m {arms['l1m']['contract']['final_epe']:.6f} vs start "
                  f"{verdict['info_vs_start']['start_contract_final_epe']:.6f})",
                  ""]
    lines += ["## Interpretation (separate from measured facts above)",
              "",
              ("[INFERENCE] A single-seed PASS admits to 3-seed validation "
               "only; it is not acceptance. A FAIL closes the L1 mechanism "
               "under this controlled pilot. See l1_verdict.json for the "
               "frozen gate record."),
              ""]
    (OUT_DIR / "L1_PILOT_REPORT.md").write_text("\n".join(lines),
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
