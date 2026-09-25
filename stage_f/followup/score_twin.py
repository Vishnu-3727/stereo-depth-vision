"""Stage-F capacity-twin scoring: score the twin pilot (2 arms) and apply the frozen verdict.

Zero-compute: local inference only, trains nothing.

Reuse (imported, never reimplemented here):
  - score_f1 (stage_f/followup/score_f1.py — itself unedited): the E3 seed0
    FINAL sanity references (REF_CONTRACT_EPE / REF_TRAIN_GT64 / REF_TRAIN_EPE /
    SANITY_TOL), score_arm for the sanity subject (E3 final is width-32, so the
    unmodified f1 loader applies), and train_split_epe_from_model (takes a
    loaded model, so it applies to either width).
  - stage_f/audit_e3.py (via score_f1's imports): sha256_file, StereoNet /
    StereoNetConfig, ev.score (frozen 40-scene hailo_val contract), git_head,
    atomic_write_json.

The twin arms need width-parametrised loading (t48 = 48 channels, t48c = 32),
which score_f1's loader hard-wires to 32 — so this script builds the model
with the per-arm feature_channels and reuses everything else. No section-4
cost-volume diagnostics: the frozen twin record (§6) needs contract final EPE,
D1, contract GT<64, train global EPE, train GT<64, params, s/epoch (from the
epoch log), sha256 and strict load — nothing else.

Sanity (mandatory, runs first): score_f1's pipeline on E3 seed0 FINAL must
reproduce the audit2 contract final EPE within 1e-4 and the audit_e3 train
GT<64 EPE within 1e-4 (the same check as score_f1). On failure the script
writes the scores JSON with sanity FAIL, writes a NO_VERDICT verdict file,
and exits 1 — no verdict is produced.

Verdict (frozen §6, unchanged by A2):
  d = train_GT<64(t48) - train_GT<64(t48c)
  d <= -0.050 -> CAPACITY SUPPORTED
  |d| < 0.020 -> CAPACITY WEAKENED
  else INCONCLUSIVE (neither is proof)
Contract dfinal is recorded for information only.
Kill: NaN/Inf in scored numbers, zero valid pixels, strict-load failure.

Usage (from repo root, with the project python):
  python stage_f/followup/score_twin.py [--device cuda|cpu]
  python stage_f/followup/score_twin.py --selftest [--device cuda|cpu]
Outputs stage_f/followup/twin_pilot/{twin_scores.json, twin_verdict.json,
TWIN_PILOT_REPORT.md, run.log}; --selftest writes to
stage_f/followup/twin_pilot/selftest/ instead, standing in the f1c FINAL
checkpoint (width 32) for BOTH arms, and must report d = 0.0 exactly.
"""

from __future__ import annotations

import argparse
import json
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

OUT_DIR = REPO / "stage_f" / "followup" / "twin_pilot"
SELFTEST_DIR = OUT_DIR / "selftest"

ARMS = {
    "t48": {
        "label": "capacity twin (width 48)",
        "width": 48,
        "expected_params": 891074,
        "ckpt": (REPO / "stage_f" / "followup" / "kaggle" / "t48_output"
                 / "seed0" / "t48_seed0_final.pth"),
        "epoch_log": (REPO / "stage_f" / "followup" / "kaggle" / "t48_output"
                      / "seed0" / "t48_seed0_epoch_log.jsonl"),
        "owner": "vishnuvardhanksece",
    },
    "t48c": {
        "label": "fresh concurrent control (width 32, E0 recipe)",
        "width": 32,
        "expected_params": 397954,
        "ckpt": (REPO / "stage_f" / "followup" / "kaggle" / "t48c_output"
                 / "seed0" / "t48c_seed0_final.pth"),
        "epoch_log": (REPO / "stage_f" / "followup" / "kaggle" / "t48c_output"
                      / "seed0" / "t48c_seed0_epoch_log.jsonl"),
        "owner": "vishnuvardhanksece",
    },
}

SELFTEST_CKPT = (REPO / "stage_f" / "followup" / "kaggle" / "f1c_output"
                 / "seed0" / "f1c_seed0_final.pth")
SELFTEST_LOG = (REPO / "stage_f" / "followup" / "kaggle" / "f1c_output"
                / "seed0" / "f1c_seed0_epoch_log.jsonl")

ENV_DIFFERENCE = (
    "Declared environment sameness (per amendment A2 section 3): t48 and t48c "
    "both run on Kaggle account vishnuvardhanksece — same account, no "
    "cross-account difference. Same GPU type (T4), same E0 recipe/seed/epochs; "
    "the arms differ ONLY in the single width knob and its init (twin48 embed "
    "vs ARM-P Stage-1)."
)

# Frozen references live in score_f1 (same check as score_f1).
REF_CONTRACT_EPE = f1.REF_CONTRACT_EPE
REF_TRAIN_GT64 = f1.REF_TRAIN_GT64
REF_TRAIN_EPE = f1.REF_TRAIN_EPE
SANITY_TOL = f1.SANITY_TOL

SUPPORT_D = -0.050
WEAKEN_ABS = 0.020


def load_model_from_path(ckpt: Path, width: int, device: str):
    """Load a StereoNet of the given width from an arbitrary checkpoint path.

    Same construction as score_f1.load_model_from_path except the model width
    is parametrised (score_f1 hard-wires the 32-channel E0 config). Returns
    (model, sha, params, strict_ok, strict_error).
    """
    sha = a3.sha256_file(ckpt)
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    cfg = dict(a3.ev.CFG)
    cfg["feature_channels"] = width
    model = a3.StereoNet(a3.StereoNetConfig(**cfg))
    try:
        model.load_state_dict(state, strict=True)
        strict_ok, strict_error = True, None
    except Exception as ex:  # noqa: BLE001 — recorded, not raised
        strict_ok, strict_error = False, f"{type(ex).__name__}: {ex}"
        return model, sha, None, strict_ok, strict_error
    params = sum(p.numel() for p in model.parameters())
    model.eval().to(device)
    return model, sha, params, strict_ok, strict_error


def secs_per_epoch(epoch_log: Path) -> dict:
    """Mean s/epoch from the kernel-enriched epoch log (epoch_wall_s column).

    The log is written by run_arm.py (read-only here); entries with a null
    wall time are excluded honestly, never fabricated.
    """
    if not epoch_log.is_file():
        return {"epoch_log": str(epoch_log), "present": False,
                "n_epochs": 0, "secs_per_epoch": None, "secs": []}
    secs = []
    n_rows = 0
    with open(epoch_log, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            r = json.loads(line)
            w = r.get("epoch_wall_s")
            if w is not None and math.isfinite(w):
                secs.append(float(w))
    return {"epoch_log": str(epoch_log), "present": True,
            "n_epochs": n_rows,
            "secs_per_epoch": (sum(secs) / len(secs)) if secs else None,
            "secs_timed": len(secs)}


def score_arm(name: str, ckpt: Path, width: int, expected_params: int,
              epoch_log: Path, device: str) -> dict:
    print(f"=== arm {name} (width {width}): {ckpt} ===", flush=True)
    if not ckpt.is_file():
        raise SystemExit(f"ABORT: checkpoint missing for {name}: {ckpt}")
    model, sha, params, strict_ok, strict_error = load_model_from_path(
        ckpt, width, device)
    rec: dict = {
        "ckpt": str(ckpt),
        "feature_channels": width,
        "sha256": sha,
        "strict_load": strict_ok,
        "strict_error": strict_error,
        "params": params,
        "expected_params": expected_params,
        "params_match": (params == expected_params) if strict_ok else False,
    }
    if not strict_ok:
        rec.update({"contract": None, "train": None, "timing": None})
        del model
        return rec
    with torch.no_grad():
        v = a3.ev.score(model, "hailo_val", device)
    rec["contract"] = {
        "split": v["split"],
        "scenes": v["scenes"],
        "valid_pixels": v["metrics"]["valid_pixels"],
        "final_epe": v["metrics"]["epe"],
        "d1": v["metrics"]["d1"],
        "gtlt64_epe": v["gt_lt_64"].get("epe"),
        "gtlt64_px": v["gt_lt_64"].get("px"),
        "guard": v["guard"],
    }
    print(f"arm {name} contract EPE={v['metrics']['epe']:.10f} "
          f"D1={v['metrics']['d1']:.6f} "
          f"GT<64={v['gt_lt_64'].get('epe')}", flush=True)
    with torch.no_grad():
        rec["train"] = f1.train_split_epe_from_model(model, device)
    print(f"arm {name} train EPE={rec['train']['epe']:.10f} "
          f"GT<64={rec['train']['gtlt64_epe']:.10f}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    rec["timing"] = secs_per_epoch(epoch_log)
    print(f"arm {name} s/epoch={rec['timing']['secs_per_epoch']} "
          f"(n_epochs={rec['timing']['n_epochs']})", flush=True)
    return rec


def finite_or_none(x) -> bool:
    return x is not None and math.isfinite(x)


def apply_verdict(arms: dict) -> dict:
    """Frozen §6 verdict on d = train GT<64(t48) - train GT<64(t48c)."""
    a, c = arms["t48"], arms["t48c"]
    dfinal = a["contract"]["final_epe"] - c["contract"]["final_epe"]
    d = a["train"]["gtlt64_epe"] - c["train"]["gtlt64_epe"]
    nums = [a["contract"]["final_epe"], a["contract"]["d1"],
            a["contract"]["gtlt64_epe"], a["train"]["epe"],
            a["train"]["gtlt64_epe"],
            c["contract"]["final_epe"], c["contract"]["d1"],
            c["contract"]["gtlt64_epe"], c["train"]["epe"],
            c["train"]["gtlt64_epe"]]
    kill = None
    if not (a["strict_load"] and c["strict_load"]):
        bad = [k for k in ("t48", "t48c") if not arms[k]["strict_load"]]
        kill = ("strict-load failure on " + ",".join(bad) + ": "
                + "; ".join(f"{k}={arms[k]['strict_error']}" for k in bad))
    elif not all(finite_or_none(x) for x in nums):
        kill = "NaN/Inf in scored numbers"
    elif (a["contract"]["valid_pixels"] == 0 or a["train"]["valid_pixels"] == 0
          or c["contract"]["valid_pixels"] == 0
          or c["train"]["valid_pixels"] == 0):
        kill = "zero valid pixels"
    reasons = [
        f"d=train_GT<64(t48)-train_GT<64(t48c)={d:.6f} "
        f"(t48 {a['train']['gtlt64_epe']:.6f} vs "
        f"t48c {c['train']['gtlt64_epe']:.6f})",
        f"contract dfinal for information: {dfinal:.6f} "
        f"(t48 {a['contract']['final_epe']:.6f} vs "
        f"t48c {c['contract']['final_epe']:.6f})",
    ]
    if kill is not None:
        return {"d": d, "dfinal_contract": dfinal, "result": "KILL",
                "reasons": [kill] + reasons}
    if d <= SUPPORT_D:
        return {"d": d, "dfinal_contract": dfinal,
                "result": "CAPACITY SUPPORTED",
                "reasons": reasons + ["d <= -0.050"]}
    if abs(d) < WEAKEN_ABS:
        return {"d": d, "dfinal_contract": dfinal,
                "result": "CAPACITY WEAKENED",
                "reasons": reasons + ["|d| < 0.020"]}
    return {"d": d, "dfinal_contract": dfinal, "result": "INCONCLUSIVE",
            "reasons": reasons + ["neither is proof"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="stand the f1c FINAL checkpoint (width 32) in for BOTH "
                         "arms; write to twin_pilot/selftest/; expect d == 0.0")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = SELFTEST_DIR if args.selftest else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
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
        _run(device, out_dir, args.selftest)
    finally:
        sys.stdout = sys.stdout.fhs[0]
        log_fh.close()


def _run(device: str, out_dir: Path, selftest: bool) -> None:
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mode = "SELFTEST (f1c FINAL stands in for both arms)" if selftest \
        else "twin pilot scoring (t48 vs t48c)"
    print(f"plan: stage-f capacity-twin scoring; {mode}; device={device}",
          flush=True)

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
        "selftest": selftest,
        "utc": utc,
    }

    arms: dict = {}
    if sanity_ok:
        if selftest:
            for name in ("t48", "t48c"):
                arms[name] = score_arm(name, SELFTEST_CKPT, 32, 397954,
                                       SELFTEST_LOG, device)
                arms[name]["stand_in"] = (
                    "f1c_seed0_final.pth (width 32) stands in for the "
                    "not-yet-pulled twin checkpoint")
                arms[name]["owner_account"] = "vishnu3727 (stand-in source)"
                arms[name]["label"] = ARMS[name]["label"] + " [SELFTEST STAND-IN]"
        else:
            for name in ("t48", "t48c"):
                cfg = ARMS[name]
                arms[name] = score_arm(name, cfg["ckpt"], cfg["width"],
                                       cfg["expected_params"], cfg["epoch_log"],
                                       device)
                arms[name]["owner_account"] = cfg["owner"]
                arms[name]["label"] = cfg["label"]

    exp = ("stage-f capacity-twin selftest (f1c FINAL for both arms)"
           if selftest else
           "stage-f capacity-twin scoring (t48 width-48 vs t48c width-32 "
           "control; seed0 FINAL checkpoints)")
    scores = {
        "experiment": exp,
        "provenance": prov,
        "sanity": sanity,
        "arms": arms,
        "thresholds": {"SUPPORTED_D": SUPPORT_D,
                       "WEAKENED_ABS": WEAKEN_ABS,
                       "SANITY_TOL": SANITY_TOL},
    }
    a3.atomic_write_json(out_dir / "twin_scores.json", scores)

    if not sanity_ok:
        nov = {"status": "NO_VERDICT_SANITY_FAIL",
               "reason": "E3 seed0 FINAL sanity check failed; no verdict "
                         "produced (see twin_scores.json sanity diffs).",
               "sanity": sanity}
        a3.atomic_write_json(out_dir / "twin_verdict.json", nov)
        _write_report(scores, None, selftest, out_dir)
        print("STOP: sanity check failed; no verdict produced", flush=True)
        raise SystemExit(1)

    verdict_body = apply_verdict(arms)
    verdict = {
        "status": "OK",
        "experiment": ("stage-f capacity-twin frozen §6 verdict "
                       "(selftest stand-in)" if selftest else
                       "stage-f capacity-twin frozen §6 verdict "
                       "(t48 train GT<64 vs concurrent t48c)"),
        "gates": {"CAPACITY SUPPORTED": "d <= -0.050",
                  "CAPACITY WEAKENED": "|d| < 0.020",
                  "INCONCLUSIVE": "otherwise (neither is proof)",
                  "KILL": "NaN/Inf, zero valid px, strict-load fail",
                  "note": "contract dfinal recorded for information only"},
        "d": verdict_body["d"],
        "dfinal_contract": verdict_body["dfinal_contract"],
        "result": verdict_body["result"],
        "reasons": verdict_body["reasons"],
        "env_difference": ENV_DIFFERENCE,
        "provenance": prov,
    }
    a3.atomic_write_json(out_dir / "twin_verdict.json", verdict)
    _write_report(scores, verdict, selftest, out_dir)
    print(f"wrote {out_dir / 'twin_scores.json'} + twin_verdict.json + "
          f"TWIN_PILOT_REPORT.md status=OK result={verdict['result']}",
          flush=True)
    if selftest:
        if verdict["d"] != 0.0:
            print(f"SELFTEST FAIL: d={verdict['d']!r} is not exactly 0.0",
                  flush=True)
            raise SystemExit(1)
        print("SELFTEST PASS: d == 0.0 exactly (same checkpoint both arms)",
              flush=True)


def _write_report(scores: dict, verdict: dict | None, selftest: bool,
                  out_dir: Path) -> None:
    arms = scores["arms"]
    san = scores["sanity"]
    title = ("# Twin pilot report — SELFTEST (f1c FINAL stands in for both arms)"
             if selftest else "# Twin pilot report (Stage F capacity twin, seed 0)")
    lines = [title,
             "",
             "[FACT] Checkpoints scored (FINAL only, seed 0):"]
    if selftest:
        lines += [f"- t48 (stand-in): `{SELFTEST_CKPT.name}` (width 32)",
                  f"- t48c (stand-in): `{SELFTEST_CKPT.name}` (width 32)",
                  "",
                  "[FACT] Selftest stands the f1c FINAL checkpoint in for BOTH "
                  "arms (the twin checkpoints do not exist yet); d must be "
                  "0.0 exactly.",
                  ""]
    else:
        lines += [f"- twin t48: `{ARMS['t48']['ckpt'].name}` (width 48)",
                  f"- control t48c: `{ARMS['t48c']['ckpt'].name}` (width 32)",
                  "",
                  "[FACT] " + ENV_DIFFERENCE,
                  ""]
    lines += ["## Sanity check (E3 seed0 FINAL, mandatory)",
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
                  "no verdict produced.",
                  ""]
    else:
        lines += ["## Numbers",
                  "",
                  "| arm | width | params | contract final EPE | contract D1 | "
                  "contract GT<64 EPE | train EPE | train GT<64 EPE | s/epoch |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name in ("t48", "t48c"):
            a = arms[name]
            c, t, s = a["contract"], a["train"], a["timing"]
            lines.append(
                f"| {name} | {a['feature_channels']} | {a['params']} | "
                f"{c['final_epe']:.6f} | {c['d1']:.4f} | "
                f"{c['gtlt64_epe']:.6f} | {t['epe']:.6f} | {t['gtlt64_epe']:.6f} | "
                f"{s['secs_per_epoch']} |")
        lines += ["",
                  "## Verdict (frozen §6)",
                  "",
                  f"[MEASUREMENT] d={verdict['d']:.6f}, "
                  f"contract dfinal={verdict['dfinal_contract']:.6f} (information only)",
                  f"[VERDICT] {verdict['result']} — " + "; ".join(verdict["reasons"]),
                  ""]
    lines += ["## Interpretation (separate from measured facts above)",
              "",
              ("[INFERENCE] SUPPORTED/WEAKENED/INCONCLUSIVE is the frozen §6 "
               "reading of the train-floor move; neither is proof. See "
               "twin_verdict.json for the frozen gate record."),
              ""]
    (out_dir / "TWIN_PILOT_REPORT.md").write_text("\n".join(lines),
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
