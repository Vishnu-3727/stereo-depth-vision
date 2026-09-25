"""Stage-F F1 pilot scoring: score the Stage F M1 pilot (3 arms) and apply the frozen gates.

Zero-compute: local inference only, trains nothing.

Reuse (imported, never reimplemented here):
  - from stage_f/audit_e3.py: sha256_file, StereoNet/StereoNetConfig via ev.CFG,
    ev.score (frozen 40-scene hailo_val contract — the exact path used for E3),
    train_split_epe metric code family (DATA_ROOT, Kitti2015Stereo, normalize),
    atomic_write_json, git_head.
  - from stage_f/followup/audit2_e3.py: run_subject (section-4 cost-volume
    diagnostics) + section3/section4 builders, via a temporary a3.load_model
    monkeypatch (audit2_e3.run_subject is hard-wired to E3 paths; the patch
    only redirects the checkpoint load, the diagnostic code runs unmodified).

Neither audit_e3.py nor audit2_e3.py is edited.

Sanity (mandatory, runs first): the same pipeline on E3 seed0 FINAL must
reproduce audit2.json contract final EPE (1.1628882757801255) within 1e-4 and
the audit_e3.json train GT<64 EPE (0.8620391636463651) within 1e-4. On failure
the script writes the scores JSON with sanity FAIL, writes a NO_VERDICT
verdict file, and exits 1 — no gate verdict is produced.

Gates (frozen, per M1 arm vs shared control f1c):
  dfinal = contract_final_EPE(arm) - contract_final_EPE(f1c)
  dtrain = train_GT<64(arm) - train_GT<64(f1c)
  PASS iff dfinal <= -0.030
  else FLOOR-MOVE iff dtrain <= -0.050 and dfinal <= 0
  else FAIL
  KILL on any NaN/Inf, zero valid px, strict-load failure, or pooled
  entropy < 0.1 nats.

Usage (from repo root, with the project python):
  python stage_f/followup/score_f1.py [--device cuda|cpu]
Outputs stage_f/followup/f1_pilot/{f1_scores.json, f1_verdict.json,
F1_PILOT_REPORT.md, run.log}.
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
import audit2_e3 as a2  # noqa: E402  (imports audit_e3 itself; same module)
import audit_e3 as a3  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT_DIR = REPO / "stage_f" / "followup" / "f1_pilot"

ARMS = {
    "f1c": {
        "label": "control (Smooth-L1)",
        "ckpt": (REPO / "stage_f" / "followup" / "kaggle" / "f1c_output"
                 / "seed0" / "f1c_seed0_final.pth"),
        "owner": "vishnu3727",
    },
    "f1m": {
        "label": "M1 lambda=0.1",
        "ckpt": (REPO / "stage_f" / "followup" / "kaggle" / "f1m_output"
                 / "seed0" / "f1m_seed0_final.pth"),
        "owner": "vishnuvardhanksece",
    },
    "f1m10": {
        "label": "M1 lambda=1.0",
        "ckpt": (REPO / "stage_f" / "followup" / "kaggle" / "f1m10_output"
                 / "seed0" / "f1m10_seed0_final.pth"),
        "owner": "vishnuvardhanksece",
    },
}

ENV_DIFFERENCE = (
    "Declared environment difference (per task brief / amendment A1 section 3): "
    "f1c ran on Kaggle account vishnu3727; both M1 arms (f1m, f1m10) ran on "
    "account vishnuvardhanksece. Same GPU type (T4), same E0 recipe/seed/epochs."
)

# Frozen references (recorded artefacts, not recomputed here).
REF_CONTRACT_EPE = 1.1628882757801255  # audit2.json seed0_final.epe ==
# e3_verdict.json per_seed_final[0] == audit_e3.json seed0_final.epe
REF_TRAIN_GT64 = 0.8620391636463651  # audit_e3.json seed0_final.train_split.gtlt64_epe
# (audit2.json train section3 GT-bin derived value 0.8620391636542211 agrees
# to ~1e-11)
REF_TRAIN_EPE = 0.9170082187690973  # audit_e3.json train_split.epe (gt>0)
SANITY_TOL = 1e-4

PASS_DFINAL = -0.030
FLOOR_DTRAIN = -0.050
ENTROPY_KILL = 0.1


def load_model_from_path(ckpt: Path, device: str):
    """Load a StereoNet from an arbitrary checkpoint path.

    Same construction as a3.load_model (same CFG/arch) but parametrised by
    path instead of hard-wired to E3 paths. Returns (model, sha, params,
    strict_ok, strict_error).
    """
    sha = a3.sha256_file(ckpt)
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model = a3.StereoNet(a3.StereoNetConfig(**a3.ev.CFG))
    try:
        model.load_state_dict(state, strict=True)
        strict_ok, strict_error = True, None
    except Exception as ex:  # noqa: BLE001 — recorded, not raised
        strict_ok, strict_error = False, f"{type(ex).__name__}: {ex}"
        return model, sha, None, strict_ok, strict_error
    params = sum(p.numel() for p in model.parameters())
    model.eval().to(device)
    return model, sha, params, strict_ok, strict_error


def train_split_epe_from_model(model, device: str) -> dict:
    """Train-split global EPE + GT<64 EPE on the no-aug train split.

    Thin path-taking wrapper around a3.train_split_epe: identical dataset,
    mask, model call and accumulation (a3.DATA_ROOT / a3.Kitti2015Stereo /
    a3.normalize); only the checkpoint source differs (takes a loaded model
    instead of an E3 (seed, tag)).
    """
    ds = a3.Kitti2015Stereo(a3.DATA_ROOT, split="hailo_calib",
                            disparity_scale=256.0, occluded=True)
    n = 0
    s_err = 0.0
    n64 = 0
    s64 = 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            samp = ds[i]
            gt = samp.disparity.astype(np.float64)
            valid = gt > 0
            out = model(torch.from_numpy(a3.normalize(samp.left)).to(device),
                        torch.from_numpy(a3.normalize(samp.right)).to(device))
            pred = out[0, 0].cpu().numpy().astype(np.float64)
            e = np.abs(pred[valid] - gt[valid])
            gv = gt[valid]
            n += int(valid.sum())
            s_err += float(e.sum())
            m = gv < 64.0
            n64 += int(m.sum())
            s64 += float(e[m].sum())
            del out
            if device == "cuda" and (i + 1) % 40 == 0:
                torch.cuda.empty_cache()
            if (i + 1) % 40 == 0 or (i + 1) == len(ds):
                print(f"train [{i + 1}/{len(ds)}]", flush=True)
    return {
        "split": "hailo_calib",
        "scenes": len(ds),
        "contract": False,
        "note": "NOT the frozen contract; same metric code",
        "valid_pixels": n,
        "epe": float(s_err / n),
        "gtlt64_px": n64,
        "gtlt64_epe": float(s64 / n64),
    }


def section4_for_path(ckpt: Path, device: str) -> dict:
    """Section-4 cost-volume diagnostics for an arbitrary checkpoint.

    Runs a2.run_subject unmodified on the contract split; only the E3
    hard-wired load (a3.load_model) is temporarily redirected to `ckpt`.
    Returns overall pooled entropy / target-rank / mode-count summaries.
    """
    orig_load = a3.load_model

    def patched(seed, tag, dev):
        model, sha, params, strict_ok, strict_error = load_model_from_path(
            ckpt, dev)
        if not strict_ok:
            raise ValueError(f"strict-load failed for {ckpt}: {strict_error}")
        if params != a3.EXPECTED_PARAMS:
            raise ValueError(f"param count {params} != {a3.EXPECTED_PARAMS}")
        return model, sha

    a3.load_model = patched
    try:
        scene_rows: list = []
        rec, pool, diag = a2.run_subject(
            0, "hailo_val", False, device, None, scene_rows,
            want_ops=False, want_svd=False)
    finally:
        a3.load_model = orig_load
    s3 = a2.section3(pool, rec["valid_pixels"])
    s4 = a2.section4(diag, pool["e"][(pool["gt"] < 64.0)],
                     s3["p90_threshold"])
    return {
        "n_sub": s4["n_sub"],
        "entropy_mean": s4["overall"]["entropy"]["mean"],
        "entropy_median": s4["overall"]["entropy"]["median"],
        "rank_mean": s4["overall"]["rank"]["mean"],
        "rank_median": s4["overall"]["rank"]["median"],
        "modes_mean": s4["overall"]["modes"]["mean"],
        "modes_median": s4["overall"]["modes"]["median"],
        "sanity_softargmin_max_abs": rec["sanity_softargmin_max_abs"],
        "xcheck_contract_epe": rec["epe"],
        "xcheck_valid_pixels": rec["valid_pixels"],
    }


def score_arm(name: str, ckpt: Path, device: str) -> dict:
    print(f"=== arm {name}: {ckpt} ===", flush=True)
    if not ckpt.is_file():
        raise SystemExit(f"ABORT: checkpoint missing for {name}: {ckpt}")
    model, sha, params, strict_ok, strict_error = load_model_from_path(
        ckpt, device)
    rec: dict = {
        "ckpt": str(ckpt),
        "sha256": sha,
        "strict_load": strict_ok,
        "strict_error": strict_error,
        "params": params,
        "expected_params": a3.EXPECTED_PARAMS,
    }
    if not strict_ok:
        rec.update({"contract": None, "train": None, "diagnostics": None})
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
        rec["train"] = train_split_epe_from_model(model, device)
    print(f"arm {name} train EPE={rec['train']['epe']:.10f} "
          f"GT<64={rec['train']['gtlt64_epe']:.10f}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    rec["diagnostics"] = section4_for_path(ckpt, device)
    d = rec["diagnostics"]
    print(f"arm {name} entropy_mean={d['entropy_mean']:.6f} "
          f"rank_median={d['rank_median']} modes_mean={d['modes_mean']:.6f}",
          flush=True)
    return rec


def finite_or_none(x) -> bool:
    return x is not None and math.isfinite(x)


def apply_gates(arms: dict) -> dict:
    """Frozen M1 gates per arm vs shared control f1c (decision only)."""
    c = arms["f1c"]
    c_epe = c["contract"]["final_epe"]
    c_tr = c["train"]["gtlt64_epe"]
    verdicts = {}
    for name in ("f1m", "f1m10"):
        a = arms[name]
        dfinal = a["contract"]["final_epe"] - c_epe
        dtrain = a["train"]["gtlt64_epe"] - c_tr
        nums = [a["contract"]["final_epe"], a["contract"]["d1"],
                a["contract"]["gtlt64_epe"], a["train"]["epe"],
                a["train"]["gtlt64_epe"], a["diagnostics"]["entropy_mean"]]
        reasons: list[str] = []
        kill = None
        if not a["strict_load"]:
            kill = f"strict-load failure: {a['strict_error']}"
        elif not all(finite_or_none(x) for x in nums):
            kill = "NaN/Inf in scored numbers"
        elif (a["contract"]["valid_pixels"] == 0
              or a["train"]["valid_pixels"] == 0):
            kill = "zero valid pixels"
        elif a["diagnostics"]["entropy_mean"] < ENTROPY_KILL:
            kill = (f"pooled entropy collapse: "
                    f"{a['diagnostics']['entropy_mean']} < {ENTROPY_KILL} nats")
        if kill is not None:
            verdicts[name] = {"dfinal": dfinal, "dtrain": dtrain,
                              "result": "KILL", "reasons": [kill]}
            continue
        reasons.append(f"dfinal={dfinal:.6f} (arm {a['contract']['final_epe']:.6f} "
                       f"vs f1c {c_epe:.6f})")
        reasons.append(f"dtrain={dtrain:.6f} (arm {a['train']['gtlt64_epe']:.6f} "
                       f"vs f1c {c_tr:.6f})")
        if dfinal <= PASS_DFINAL:
            verdicts[name] = {"dfinal": dfinal, "dtrain": dtrain,
                              "result": "PASS",
                              "reasons": reasons + ["dfinal <= -0.030"]}
        elif dtrain <= FLOOR_DTRAIN and dfinal <= 0:
            verdicts[name] = {"dfinal": dfinal, "dtrain": dtrain,
                              "result": "FLOOR-MOVE",
                              "reasons": reasons + ["dtrain <= -0.050 and dfinal <= 0"]}
        else:
            verdicts[name] = {"dfinal": dfinal, "dtrain": dtrain,
                              "result": "FAIL",
                              "reasons": reasons + ["not supported under this "
                                                    "controlled pilot"]}
    return verdicts


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
    print(f"plan: stage-f F1 pilot scoring (3 arms + E3 sanity); device={device}",
          flush=True)

    # ---- mandatory sanity check on E3 seed0 FINAL ----
    e3_ckpt = a3.ckpt_path(0, "final")
    print(f"=== sanity: E3 seed0 FINAL {e3_ckpt} ===", flush=True)
    sanity_rec = score_arm("e3_sanity", e3_ckpt, device)
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
                            "(== e3_verdict.json per_seed_final[0])",
        "diff_contract": d_contract,
        "recomputed_train_gtlt64_epe": sanity_rec["train"]["gtlt64_epe"],
        "reference_train_gtlt64_epe": REF_TRAIN_GT64,
        "reference_train_source": "stage_f/audit/audit_e3.json "
                                  "seed0_final.train_split.gtlt64_epe",
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
        for name in ("f1c", "f1m", "f1m10"):
            arms[name] = score_arm(name, ARMS[name]["ckpt"], device)
            arms[name]["owner_account"] = ARMS[name]["owner"]
            arms[name]["label"] = ARMS[name]["label"]

    scores = {
        "experiment": "stage-f F1 M1 pilot scoring (f1c control, f1m lam=0.1, "
                      "f1m10 lam=1.0; seed0 FINAL checkpoints)",
        "provenance": prov,
        "sanity": sanity,
        "arms": arms,
        "thresholds": {"PASS_DFINAL": PASS_DFINAL,
                       "FLOOR_DTRAIN": FLOOR_DTRAIN,
                       "ENTROPY_KILL_NATS": ENTROPY_KILL,
                       "SANITY_TOL": SANITY_TOL},
    }
    a3.atomic_write_json(OUT_DIR / "f1_scores.json", scores)

    if not sanity_ok:
        nov = {"status": "NO_VERDICT_SANITY_FAIL",
               "reason": "E3 seed0 FINAL sanity check failed; no gate verdict "
                         "produced (see f1_scores.json sanity diffs).",
               "sanity": sanity}
        a3.atomic_write_json(OUT_DIR / "f1_verdict.json", nov)
        _write_report(scores, None)
        print("STOP: sanity check failed; no verdict produced", flush=True)
        raise SystemExit(1)

    verdicts = apply_gates(arms)
    passed = [k for k, v in verdicts.items() if v["result"] == "PASS"]
    if len(passed) == 2:
        better = min(passed, key=lambda k: verdicts[k]["dfinal"])
        mult = (f"Both M1 arms pass at single seed: admission to 3-seed "
                f"validation only, never acceptance. Larger improvement "
                f"({better}, dfinal={verdicts[better]['dfinal']:.6f}) goes forward.")
    elif len(passed) == 1:
        mult = (f"Single-seed pass by {passed[0]}: admission to 3-seed "
                f"validation only, never acceptance.")
    else:
        mult = ("No single-seed pass: nothing is admitted to 3-seed validation.")
    verdict = {
        "status": "OK",
        "experiment": "stage-f F1 M1 pilot frozen gates (per arm vs shared f1c)",
        "gates": {"PASS": "dfinal <= -0.030",
                  "FLOOR-MOVE": "dtrain <= -0.050 and dfinal <= 0",
                  "FAIL": "otherwise (not supported under this controlled pilot)",
                  "KILL": "NaN/Inf, zero valid px, strict-load fail, "
                          "pooled entropy < 0.1 nats"},
        "per_arm": verdicts,
        "multiplicity_A1": mult,
        "env_difference": ENV_DIFFERENCE,
        "provenance": prov,
    }
    a3.atomic_write_json(OUT_DIR / "f1_verdict.json", verdict)
    _write_report(scores, verdict)
    print(f"wrote {OUT_DIR / 'f1_scores.json'} + f1_verdict.json + "
          f"F1_PILOT_REPORT.md status=OK", flush=True)


def _write_report(scores: dict, verdict: dict | None) -> None:
    arms = scores["arms"]
    san = scores["sanity"]
    lines = ["# F1 pilot report (Stage F M1, seed 0)",
             "",
             "[FACT] Checkpoints scored (FINAL only, seed 0):",
             f"- control f1c: `{ARMS['f1c']['ckpt'].name}`",
             f"- M1 lam=0.1 f1m: `{ARMS['f1m']['ckpt'].name}`",
             f"- M1 lam=1.0 f1m10: `{ARMS['f1m10']['ckpt'].name}`",
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
                  "train EPE | train GT<64 EPE | entropy mean | rank median | "
                  "modes mean |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name in ("f1c", "f1m", "f1m10"):
            a = arms[name]
            c, t, d = a["contract"], a["train"], a["diagnostics"]
            lines.append(
                f"| {name} | {c['final_epe']:.6f} | {c['d1']:.4f} | "
                f"{c['gtlt64_epe']:.6f} | {t['epe']:.6f} | {t['gtlt64_epe']:.6f} | "
                f"{d['entropy_mean']:.4f} | {d['rank_median']} | {d['modes_mean']:.4f} |")
        lines += ["",
                  "## Gate outcomes (per arm vs shared control f1c)",
                  ""]
        for name in ("f1m", "f1m10"):
            v = verdict["per_arm"][name]
            lines.append(f"[MEASUREMENT] {name}: dfinal={v['dfinal']:.6f}, "
                         f"dtrain={v['dtrain']:.6f}")
            lines.append(f"[VERDICT] {name}: {v['result']} — "
                         + "; ".join(v["reasons"]))
        lines += ["",
                  "[VERDICT] multiplicity (A1): " + verdict["multiplicity_A1"],
                  ""]
    lines += ["## Interpretation (separate from measured facts above)",
              "",
              ("[INFERENCE] A single-seed PASS/FLOOR-MOVE admits an arm to 3-seed "
               "validation only; it is not acceptance. See f1_verdict.json for "
               "the frozen gate record."),
              ""]
    (OUT_DIR / "F1_PILOT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
