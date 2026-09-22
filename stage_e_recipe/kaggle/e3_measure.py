#!/usr/bin/env python
"""Stage E3 measurements: log-metric comparison (step 3) + E0 device recheck (step 4).

Step 3: best10 EPE from per-epoch training logs with the IDENTICAL definition
recover_e3.py uses: min over log rows carrying val_epe. E0's epoch logs write
val_epe:null on non-monitor epochs while E3's omit the key, so the filter is
`r.get("val_epe") is not None` -- byte-identical result on E3's logs (which
carry no nulls), null-safe on E0's. No verdict, no interpretation.

Step 4: re-score E0's pulled checkpoints LOCALLY on the frozen 40-scene
contract, reusing run_experiment.py's scoring code path verbatim in structure:
eval_tier2.score(model, "hailo_val", device) with strict_load_report /
weight_sha / sha256_file from phase1.harness.frozen_eval. E3 checkpoints are
NOT scored here.

Writes stage_e_recipe/e3_logmetric_compare.json and
stage_e_recipe/e0_device_recheck.json. Read-only against e0_output/ and
e3_output/.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT_COMPARE = REPO / "stage_e_recipe" / "e3_logmetric_compare.json"
OUT_RECHECK = REPO / "stage_e_recipe" / "e0_device_recheck.json"
EV_PATH = REPO / "stage_e_recipe" / "kaggle" / "bundle_e3" / "scripts" / "eval_tier2.py"
EXPECTED_PARAMS = 397954


def best10_from_log(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    scored = [r for r in rows if r.get("val_epe") is not None]
    best = min(scored, key=lambda r: r["val_epe"])
    return {"log_file": str(path.relative_to(REPO)).replace("\\", "/"),
            "row_count": len(rows), "scored_rows": len(scored),
            "best10_epe": best["val_epe"], "best10_epoch": best["epoch"]}


def step3() -> dict:
    out = {"experiment": "stage-e E0-vs-E3 training-log metric comparison",
           "definition": ("min over per-epoch log rows carrying non-null val_epe "
                          "(identical to recover_e3.py; null-safe for E0 logs)"),
           "note": "training-log 10-scene monitor metric, NOT frozen-contract EPE; "
                   "no verdict, no interpretation",
           "seeds": {}}
    for n in (0, 1, 2):
        p = REPO / "stage_e_recipe" / "kaggle" / "e0_output" / f"seed{n}" / \
            f"e0_seed{n}_epoch_log.jsonl"
        out["seeds"][f"E0_seed{n}"] = best10_from_log(p)
    for n in (0, 1, 2):
        p = REPO / "stage_e_recipe" / "kaggle" / "e3_output" / f"seed{n}" / \
            "repo" / "stage_e_recipe" / "run" / "armp" / "training_log.jsonl"
        out["seeds"][f"E3_seed{n}"] = best10_from_log(p)
    OUT_COMPARE.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def step4() -> dict:
    import torch

    from phase1.harness.frozen_eval import (sha256_file, strict_load_report,
                                            weight_sha)
    from src.models.stereonet import StereoNet, StereoNetConfig

    spec = importlib.util.spec_from_file_location("eval_tier2", EV_PATH)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    # ev.REPO derives from the bundle layout (bundle_e3/scripts -> parents[1]),
    # which on Kaggle resolves to the repo root because the bundle is assembled
    # there. Locally it does not, so point it at the real repo root. score()
    # itself -- dataset split, GT scale, mask, pooled metrics, contract guard --
    # is used unmodified.
    ev.REPO = REPO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"experiment": "stage-e E0 device comparability calibration (T4 vs local)",
           "note": ("E0 pulled checkpoints re-scored LOCALLY on the frozen 40-scene "
                    "contract via run_experiment.py's scoring path "
                    "(eval_tier2.score + strict_load_report/weight_sha); E3 NOT scored"),
           "local_device": (torch.cuda.get_device_name(0)
                            if device == "cuda" else "cpu"),
           "local_torch": torch.__version__,
           "seeds": {}}
    for n in (0, 1, 2):
        e0dir = REPO / "stage_e_recipe" / "kaggle" / "e0_output" / f"seed{n}"
        rec = json.loads((e0dir / f"e0_seed{n}.json").read_text(encoding="utf-8"))
        seed_rec = {}
        for tag in ("best", "final"):
            ck = e0dir / f"e0_seed{n}_{tag}.pth"
            t4_epe = rec["checkpoints"][tag]["hailo_val"]["metrics"]["epe"]
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            state = blob["model"]
            model = StereoNet(StereoNetConfig(**ev.CFG))
            compat = strict_load_report(model, state)
            params = sum(p_.numel() for p_ in model.parameters())
            guard_ok = (not compat["missing"] and not compat["unexpected"]
                        and params == EXPECTED_PARAMS)
            entry = {"checkpoint": str(ck.relative_to(REPO)).replace("\\", "/"),
                     "t4_epe": t4_epe,
                     "strict_load_ok": guard_ok,
                     "weight_sha16_match": (weight_sha(state)
                                            == rec["checkpoints"][tag]["weight_sha16"])}
            if not guard_ok:
                entry["local_epe"] = None
                entry["local_minus_t4"] = None
                entry["contract_match_local"] = False
            else:
                model.eval().to(device)
                with torch.no_grad():
                    v = ev.score(model, "hailo_val", device)
                local_epe = v["metrics"]["epe"]
                entry["local_epe"] = local_epe
                entry["local_minus_t4"] = local_epe - t4_epe
                entry["contract_match_local"] = v["guard"]["contract_match"]
                entry["local_valid_pixels"] = v["metrics"]["valid_pixels"]
                entry["local_scenes"] = v["scenes"]
                del v
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            seed_rec[tag] = entry
            print(f"seed{n}/{tag}: t4={t4_epe:.7f} "
                  f"local={entry['local_epe'] if entry['local_epe'] is not None else None} "
                  f"contract={entry['contract_match_local']}", flush=True)
        out["seeds"][str(n)] = seed_rec
    out["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT_RECHECK.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    c = step3()
    for k, v in c["seeds"].items():
        print(f"{k}: best10={v['best10_epe']:.7f}@{v['best10_epoch']} "
              f"rows={v['row_count']}")
    print(f"wrote {OUT_COMPARE}")
    step4()
    print(f"wrote {OUT_RECHECK}")


if __name__ == "__main__":
    sys.exit(main())
