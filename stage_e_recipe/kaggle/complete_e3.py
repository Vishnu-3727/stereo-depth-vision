#!/usr/bin/env python
"""Stage E3 completion: materialize canonical layout + score locally (Task brief 2).

Step 1 (additive copy only): for each seed N in 0,1,2 copy from
    stage_e_recipe/kaggle/e3_output/seed<N>/repo/stage_e_recipe/run/armp/
        p2a_best.pth       -> seed<N>/e3_seed<N>_best.pth
        p2a_final.pth      -> seed<N>/e3_seed<N>_final.pth
        training_log.jsonl -> seed<N>/e3_seed<N>_epoch_log.jsonl
        integrity_guard.json -> seed<N>/e3_seed<N>_integrity_guard.json
        stdout.log         -> seed<N>/e3_seed<N>_stdout.log
Never moves, deletes, or overwrites: a destination that already exists is
left untouched (its sha256 is verified against the source and reported).

Step 2 (local scoring): score each seed's best and final checkpoint LOCALLY
on the frozen 40-scene contract, reusing run_experiment.py's scoring block
(phase1.harness.frozen_eval strict_load_report / weight_sha / sha256_file
plus eval_tier2.score(model, "hailo_val", device)) verbatim in structure.
eval_tier2 is loaded from bundle_e3/scripts/eval_tier2.py with the same
ev.REPO root fix documented in e3_measure.py; score() itself is unmodified.

For each seed the original Kaggle STOPPED record is first preserved via copy
to seed<N>/kaggle_stopped_seed<N>.json (that name must NOT match the
e3_seed*.json glob), and only then is the canonical seed<N>/e3_seed<N>.json
written with status COMPLETE, the checkpoints block in exactly the shape
verdict.py reads, plus honest local-scoring provenance fields.

No training, no Kaggle calls, no edits to any existing file's contents.

    python stage_e_recipe/kaggle/complete_e3.py
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

SEEDS = (0, 1, 2)
EXP = "e3"
EXPECTED_PARAMS = 397954
EV_PATH = REPO / "stage_e_recipe" / "kaggle" / "bundle_e3" / "scripts" / "eval_tier2.py"

SCORING_NOTE = (
    "contract scoring never ran on Kaggle (false epochs_incomplete STOP "
    "aborted run_experiment.py); scored offline on the development box. "
    "Device comparability calibrated in stage_e_recipe/e0_device_recheck.json, "
    "max |local-T4| = 0.000129 px vs S0_best 0.0174488."
)
RECOVERY_REF = "stage_e_recipe/e3_recovery.json"

COPY_MAP = (
    ("p2a_best.pth", "e3_seed{N}_best.pth"),
    ("p2a_final.pth", "e3_seed{N}_final.pth"),
    ("training_log.jsonl", "e3_seed{N}_epoch_log.jsonl"),
    ("integrity_guard.json", "e3_seed{N}_integrity_guard.json"),
    ("stdout.log", "e3_seed{N}_stdout.log"),
)


def sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def step1_copy(seed: int) -> dict:
    """Additive copy of the canonical layout for one seed. Returns a report."""
    srcdir = (REPO / "stage_e_recipe" / "kaggle" / "e3_output" / f"seed{seed}"
              / "repo" / "stage_e_recipe" / "run" / "armp")
    dstdir = REPO / "stage_e_recipe" / "kaggle" / "e3_output" / f"seed{seed}"
    report = {"seed": seed, "files": {}}
    for src_name, dst_tmpl in COPY_MAP:
        src = srcdir / src_name
        dst = dstdir / dst_tmpl.format(N=seed)
        if not src.is_file():
            raise SystemExit(f"COMPLETE_E3 FAIL: seed{seed} source missing {src}")
        if dst.is_file():
            # Never overwrite: verify the existing file matches the source.
            same = sha256_of(src) == sha256_of(dst)
            report["files"][dst.name] = {
                "action": "kept-existing",
                "sha_match_source": same,
                "bytes": dst.stat().st_size,
            }
            if not same:
                raise SystemExit(
                    f"COMPLETE_E3 FAIL: seed{seed} existing {dst.name} "
                    f"differs from source; refusing to overwrite")
        else:
            shutil.copy2(src, dst)
            report["files"][dst.name] = {
                "action": "copied",
                "sha_match_source": sha256_of(src) == sha256_of(dst),
                "bytes": dst.stat().st_size,
            }
    return report


def load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_tier2", EV_PATH)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    # Same root fix as e3_measure.py: the bundle layout's derived REPO does
    # not resolve locally, so point it at the real repo root. score() itself
    # -- dataset split, GT scale, mask, pooled metrics, contract guard -- is
    # used unmodified.
    ev.REPO = REPO
    return ev


def step2_score(seed: int, ev, device: str, scoring_device: str,
                scoring_torch: str) -> dict:
    import torch

    from phase1.harness.frozen_eval import (sha256_file, strict_load_report,
                                            weight_sha)
    from src.models.stereonet import StereoNet, StereoNetConfig

    dstdir = REPO / "stage_e_recipe" / "kaggle" / "e3_output" / f"seed{seed}"
    orig_path = dstdir / f"e3_seed{seed}.json"
    if not orig_path.is_file():
        raise SystemExit(f"COMPLETE_E3 FAIL: seed{seed} missing {orig_path}")
    orig = json.loads(orig_path.read_text(encoding="utf-8"))

    # Preserve the Kaggle STOPPED record BEFORE writing anything. The backup
    # name must not match the e3_seed*.json glob verdict.py reads.
    backup = dstdir / f"kaggle_stopped_seed{seed}.json"
    if backup.is_file():
        existing = json.loads(backup.read_text(encoding="utf-8"))
        if existing != orig and orig.get("status") == "STOPPED":
            raise SystemExit(
                f"COMPLETE_E3 FAIL: seed{seed} backup {backup.name} exists "
                f"but differs from current STOPPED record; refusing to touch")
        backup_action = "kept-existing-backup"
    else:
        shutil.copy2(orig_path, backup)
        backup_action = "copied"

    if orig.get("status") != "STOPPED":
        # The canonical file was already written by a previous run of this
        # script: re-scoring would only rewrite it, so report and stop.
        raise SystemExit(
            f"COMPLETE_E3 FAIL: seed{seed} e3_seed{seed}.json status is "
            f"{orig.get('status')!r}, not STOPPED; refusing to overwrite a "
            f"completed canonical record")

    t0 = time.time()
    checkpoints = {}
    for tag in ("best", "final"):
        ck = dstdir / f"e3_seed{seed}_{tag}.pth"
        if not ck.is_file():
            raise SystemExit(f"COMPLETE_E3 FAIL: seed{seed} missing {ck.name}")
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        model = StereoNet(StereoNetConfig(**ev.CFG))
        compat = strict_load_report(model, state)
        params = sum(p_.numel() for p_ in model.parameters())
        if compat["missing"] or compat["unexpected"] or params != EXPECTED_PARAMS:
            raise SystemExit(
                f"COMPLETE_E3 FAIL: seed{seed}/{tag} guard failed: "
                f"{json.dumps(compat)} params={params}")
        model.eval().to(device)
        # run_experiment.py's scoring block, verbatim in structure.
        entry = {"sha256": sha256_file(ck), "weight_sha16": weight_sha(state),
                 "params": params, "n_keys": len(state), "strict_load": compat,
                 "hailo_val": ev.score(model, "hailo_val", device)}
        checkpoints[tag] = entry
        v = entry["hailo_val"]
        print(f"  seed{seed}/{tag} EPE {v['metrics']['epe']:.7f} "
              f"D1 {v['metrics']['d1']:.4f}% "
              f"contract {v['guard']['contract_match']}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    rec = dict(orig)  # preserve every Kaggle field (experiment/seed/epochs/
                      # intervention/init/environment/train_wall_s/STOP/...)
    rec["checkpoints"] = checkpoints
    rec["status"] = "COMPLETE"
    rec["scored_locally"] = True
    rec["scoring_device"] = scoring_device
    rec["scoring_torch"] = scoring_torch
    rec["scoring_note"] = SCORING_NOTE
    rec["recovery"] = RECOVERY_REF
    rec["scoring_wall_s"] = round(time.time() - t0, 1)
    rec["scoring_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    orig_path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return {"seed": seed, "backup_action": backup_action,
            "best_epe": checkpoints["best"]["hailo_val"]["metrics"]["epe"],
            "final_epe": checkpoints["final"]["hailo_val"]["metrics"]["epe"],
            "contract_best": checkpoints["best"]["hailo_val"]["guard"]["contract_match"],
            "contract_final": checkpoints["final"]["hailo_val"]["guard"]["contract_match"]}


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scoring_device = (torch.cuda.get_device_name(0)
                      if device == "cuda" else "cpu")
    scoring_torch = torch.__version__
    print(f"device={device} scoring_device={scoring_device} "
          f"torch={scoring_torch}", flush=True)

    for seed in SEEDS:
        rep = step1_copy(seed)
        for name, info in rep["files"].items():
            print(f"seed{seed} {name}: {info['action']} "
                  f"sha_match={info['sha_match_source']} "
                  f"bytes={info['bytes']}", flush=True)

    ev = load_eval_module()
    results = []
    for seed in SEEDS:
        print(f"--- scoring seed {seed} ---", flush=True)
        results.append(step2_score(seed, ev, device, scoring_device,
                                   scoring_torch))
    for r in results:
        print(f"seed{r['seed']}: best EPE {r['best_epe']:.7f} "
              f"final EPE {r['final_epe']:.7f} "
              f"contract {r['contract_best']}/{r['contract_final']} "
              f"(backup {r['backup_action']})")
    print("wrote canonical e3_seed<N>.json for seeds 0,1,2")


if __name__ == "__main__":
    main()
