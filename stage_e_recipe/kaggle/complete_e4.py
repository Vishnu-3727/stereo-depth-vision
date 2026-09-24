#!/usr/bin/env python
"""Stage E4 completion: materialize canonical layout + score locally.

Step 1 (additive copy only): for each requested seed N copy from the pulled
Kaggle output's training dir

    <evidence-seed-dir>/repo/stage_e_recipe/run/armp/
        p2a_best.pth       -> seed<N>/e4_seed<N>_best.pth
        p2a_final.pth      -> seed<N>/e4_seed<N>_final.pth
        training_log.jsonl -> seed<N>/e4_seed<N>_epoch_log.jsonl
        integrity_guard.json -> seed<N>/e4_seed<N>_integrity_guard.json
        stdout.log         -> seed<N>/e4_seed<N>_stdout.log

into <output-root>/seed<N>/ (created if missing; the probe files already in
e4_output/ are left untouched). Never moves, deletes, or overwrites: a
destination that already exists is left untouched (its sha256 is verified
against the source and reported). When the pull already landed directly in
the destination (seed 2's pull-finetune lands in e4_output/seed2/), the
files are already in place and are verified, not copied.

Step 2 (local scoring): score each seed's best and final checkpoint LOCALLY
on the frozen 40-scene contract, reusing run_experiment.py's scoring block
(phase1.harness.frozen_eval strict_load_report / weight_sha / sha256_file
plus eval_tier2.score(model, "hailo_val", device)) verbatim in structure.
eval_tier2 is loaded from bundle_e4ft/scripts/eval_tier2.py with the same
ev.REPO root fix documented in e3_measure.py; score() itself is unmodified.
All E4 seeds are scored on the same local device, including a seed whose
Kaggle run already passed its guards.

Two seed-2 outcomes are handled:

- Kaggle STOPPED: the original Kaggle STOPPED record
  (<evidence>/e4_seed<N>.json) is first preserved via copy to
  seed<N>/kaggle_stopped_seed<N>.json (that name must NOT match the
  e4_seed*.json glob), and only then is the canonical seed<N>/e4_seed<N>.json
  written with status COMPLETE, the checkpoints block in exactly the shape
  verdict.py reads, plus honest local-scoring provenance fields.
- Kaggle COMPLETE (guard passed on Kaggle): Kaggle's own record is first
  preserved via copy to seed<N>/kaggle_record_seed<N>.json, and only then is
  the canonical seed<N>/e4_seed<N>.json replaced by the locally scored one
  (the sole permitted overwrite: the pulled copy is replaced only after it
  is preserved). Kaggle's T4 best/final EPEs are kept in a `kaggle_t4`
  provenance field for comparison against the local scores.

Options:

    python stage_e_recipe/kaggle/complete_e4.py [--seeds 0,1]
           [--evidence-root <dir>] [--output-root <dir>] [--dry-run]

--seeds defaults to "0,1" (existing behaviour unchanged).
--evidence-root overrides where the pulled Kaggle output of each seed lives
(default: e4ft_err_seed<N>/ for seeds 0/1, e4_output/seed2/ for seed 2;
either the seed evidence dir itself or a parent dir holding per-seed dirs).
--output-root overrides the canonical destination parent (default
e4_output/). --dry-run verifies copies, backups and canonical records and
reports whether everything would be byte-identical, without copying,
scoring, or writing anything.

No training, no Kaggle calls, no edits to any existing file's contents.

    python stage_e_recipe/kaggle/complete_e4.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_SEEDS = (0, 1)
EXP = "e4"
EXPECTED_PARAMS = 397954
EV_PATH = REPO / "stage_e_recipe" / "kaggle" / "bundle_e4ft" / "scripts" / "eval_tier2.py"
DEFAULT_OUTPUT_ROOT = REPO / "stage_e_recipe" / "kaggle" / "e4_output"

SCORING_NOTE = (
    "contract scoring never ran on Kaggle (false wrong_checkpoint_resume STOP "
    "aborted run_experiment_e4ft.py); scored offline on the development box. "
    "Device comparability calibrated in stage_e_recipe/e0_device_recheck.json, "
    "max |local-T4| = 0.000129 px vs S0_best 0.0174488."
)
RECOVERY_REF = "stage_e_recipe/e4_recovery.json"

COPY_MAP = (
    ("p2a_best.pth", "e4_seed{N}_best.pth"),
    ("p2a_final.pth", "e4_seed{N}_final.pth"),
    ("training_log.jsonl", "e4_seed{N}_epoch_log.jsonl"),
    ("integrity_guard.json", "e4_seed{N}_integrity_guard.json"),
    ("stdout.log", "e4_seed{N}_stdout.log"),
)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = []
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seed = int(part)
        except ValueError:
            raise SystemExit(f"COMPLETE_E4 FAIL: bad --seeds value {value!r}")
        if seed not in (0, 1, 2):
            raise SystemExit(
                f"COMPLETE_E4 FAIL: seed must be 0, 1 or 2, got {seed}")
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise SystemExit(f"COMPLETE_E4 FAIL: bad --seeds value {value!r}")
    return tuple(seeds)


def evidence_dir_for(seed: int, evidence_root: str | None) -> Path:
    """Resolve the pulled Kaggle output dir for one seed.

    Defaults preserve the existing layout: seeds 0/1 live in
    `e4ft_err_seed<N>/`, while seed 2's pull-finetune lands directly in
    `e4_output/seed2/`. --evidence-root overrides the location: either the
    seed evidence dir itself (it contains `repo/` or `e4_seed<N>.json`), or
    a parent dir holding per-seed dirs (`e4ft_err_seed<N>`, `seed<N>`, or
    `e4_output/seed<N>`).
    """
    if evidence_root is not None:
        root = Path(evidence_root)
        if (root / "repo").is_dir() or (root / f"e4_seed{seed}.json").is_file():
            return root
        for cand in (root / f"e4ft_err_seed{seed}", root / f"seed{seed}",
                     root / "e4_output" / f"seed{seed}"):
            if cand.is_dir():
                return cand
        return root / f"e4ft_err_seed{seed}"
    if seed == 2:
        return REPO / "stage_e_recipe" / "kaggle" / "e4_output" / f"seed{seed}"
    return REPO / "stage_e_recipe" / "kaggle" / f"e4ft_err_seed{seed}"


def srcdir_for(evidence_dir: Path) -> Path:
    return evidence_dir / "repo" / "stage_e_recipe" / "run" / "armp"


def dstdir_for(seed: int, output_root: Path) -> Path:
    return output_root / f"seed{seed}"


def orig_path_for(seed: int, evidence_dir: Path) -> Path:
    return evidence_dir / f"e4_seed{seed}.json"


def sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_for_record(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(path)


def step1_copy(seed: int, srcdir: Path, dstdir: Path,
               dry_run: bool = False) -> dict:
    """Additive copy of the canonical layout for one seed. Returns a report."""
    if not dry_run:
        dstdir.mkdir(parents=True, exist_ok=True)
    report = {"seed": seed, "files": {}}
    for src_name, dst_tmpl in COPY_MAP:
        src = srcdir / src_name
        dst = dstdir / dst_tmpl.format(N=seed)
        if not src.is_file():
            if dst.is_file():
                # The pull already landed directly in the destination (seed
                # 2's pull-finetune lands in e4_output/seed2/): nothing to
                # copy, just verify presence.
                report["files"][dst.name] = {
                    "action": "kept-existing-pulled",
                    "sha_match_source": True,
                    "bytes": dst.stat().st_size,
                }
                continue
            raise SystemExit(f"COMPLETE_E4 FAIL: seed{seed} source missing {src}")
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
                    f"COMPLETE_E4 FAIL: seed{seed} existing {dst.name} "
                    f"differs from source; refusing to overwrite")
        else:
            if dry_run:
                report["files"][dst.name] = {
                    "action": "would-copy",
                    "sha_match_source": True,
                    "bytes": src.stat().st_size,
                }
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


def plan_step2(seed: int, evidence_dir: Path, dstdir: Path,
               dry_run: bool = False) -> dict:
    """Read the Kaggle record, preserve it, and decide whether scoring may proceed.

    Returns a plan dict with the orig record, its status branch ("STOPPED" or
    "COMPLETE"), the backup path/action, the canon path, and whether this is a
    fresh (unprocessed) record. Exits on any provenance violation. In dry-run
    mode nothing is copied; the plan only reports what would happen.
    """
    orig_path = orig_path_for(seed, evidence_dir)
    if not orig_path.is_file():
        raise SystemExit(f"COMPLETE_E4 FAIL: seed{seed} missing {orig_path}")
    orig = json.loads(orig_path.read_text(encoding="utf-8"))
    status = orig.get("status")

    canon_path = dstdir / f"e4_seed{seed}.json"
    same_file = canon_path.resolve() == orig_path.resolve()

    if status == "STOPPED":
        # Preserve the Kaggle STOPPED record BEFORE writing anything. The
        # backup name must not match the e4_seed*.json glob verdict.py reads.
        backup = dstdir / f"kaggle_stopped_seed{seed}.json"
        branch = "STOPPED"
    elif status == "COMPLETE":
        # Preserve Kaggle's own COMPLETE record BEFORE replacing the pulled
        # copy with the locally scored canonical file.
        backup = dstdir / f"kaggle_record_seed{seed}.json"
        branch = "COMPLETE"
    else:
        raise SystemExit(
            f"COMPLETE_E4 FAIL: seed{seed} e4_seed{seed}.json status is "
            f"{status!r}, not STOPPED or COMPLETE; refusing to write a "
            f"completed canonical record from it")

    if backup.is_file():
        existing = json.loads(backup.read_text(encoding="utf-8"))
        # The backup must match the record it was preserved from. When the
        # pull lives in the destination dir the live file moves on (STOPPED
        # -> locally COMPLETE), so a mismatch against an already-preserved
        # backup is only a violation while the live record is still the
        # unprocessed Kaggle original (no local scoring fields yet).
        if existing != orig and not orig.get("scored_locally"):
            raise SystemExit(
                f"COMPLETE_E4 FAIL: seed{seed} backup {backup.name} exists "
                f"but differs from current {status} record; refusing to touch")
        backup_action = "kept-existing-backup"
    elif dry_run:
        backup_action = "would-copy-backup"
    else:
        dstdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(orig_path, backup)
        backup_action = "copied"

    if canon_path.is_file():
        if same_file and not orig.get("scored_locally"):
            # Fresh pull sitting at the canonical path (seed 2): the pulled
            # copy is replaced by the locally scored record below — the sole
            # permitted overwrite, allowed only after the backup above.
            fresh = True
        elif dry_run:
            # Already completed: a real run would refuse here, so the file
            # would stay byte-identical. Report that instead of failing.
            return {"seed": seed, "branch": branch, "orig": orig,
                    "orig_path": orig_path, "canon_path": canon_path,
                    "backup": backup, "backup_action": backup_action,
                    "same_file": same_file, "fresh": False}
        else:
            # The canonical file was already written by a previous run of
            # this script: re-scoring would only rewrite it, so stop.
            raise SystemExit(
                f"COMPLETE_E4 FAIL: seed{seed} e4_seed{seed}.json already "
                f"exists; refusing to overwrite a completed canonical record")
    else:
        fresh = True

    return {"seed": seed, "branch": branch, "orig": orig,
            "orig_path": orig_path, "canon_path": canon_path,
            "backup": backup, "backup_action": backup_action,
            "same_file": same_file, "fresh": True}


def kaggle_t4_provenance(seed: int, orig: dict, backup: Path) -> dict:
    """Extract Kaggle's T4 best/final EPEs from a COMPLETE record."""
    try:
        best = orig["checkpoints"]["best"]["hailo_val"]["metrics"]
        final = orig["checkpoints"]["final"]["hailo_val"]["metrics"]
        prov = {"best_epe": best["epe"], "final_epe": final["epe"],
                "best_d1": best["d1"], "final_d1": final["d1"]}
    except (KeyError, TypeError):
        raise SystemExit(
            f"COMPLETE_E4 FAIL: seed{seed} Kaggle COMPLETE record has no "
            f"checkpoints best/final hailo_val metrics; refusing to replace "
            f"it with a locally scored record")
    prov["note"] = ("T4 scores from Kaggle's own COMPLETE record, preserved "
                    f"in {rel_for_record(backup)}; replaced below by same-"
                    "device local scoring")
    return prov


def recovery_ref_for(seed: int) -> str:
    recovery_file = REPO / "stage_e_recipe" / "e4_recovery.json"
    if recovery_file.is_file():
        try:
            seeds = json.loads(recovery_file.read_text(encoding="utf-8")).get("seeds", {})
            if str(seed) in seeds:
                return RECOVERY_REF
        except (json.JSONDecodeError, OSError):
            pass
    return ("kaggle-guard-passed; no local recovery entry "
            "(see kaggle_record_seed<N>.json)")


def step2_score(seed: int, ev, device: str, scoring_device: str,
                scoring_torch: str, evidence_dir: Path,
                output_root: Path) -> dict:
    import torch

    from phase1.harness.frozen_eval import (sha256_file, strict_load_report,
                                            weight_sha)
    from src.models.stereonet import StereoNet, StereoNetConfig

    dstdir = dstdir_for(seed, output_root)
    plan = plan_step2(seed, evidence_dir, dstdir)
    orig, backup = plan["orig"], plan["backup"]
    backup_action, branch = plan["backup_action"], plan["branch"]
    canon_path = plan["canon_path"]

    t0 = time.time()
    checkpoints = {}
    for tag in ("best", "final"):
        ck = dstdir / f"e4_seed{seed}_{tag}.pth"
        if not ck.is_file():
            raise SystemExit(f"COMPLETE_E4 FAIL: seed{seed} missing {ck.name}")
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        model = StereoNet(StereoNetConfig(**ev.CFG))
        compat = strict_load_report(model, state)
        params = sum(p_.numel() for p_ in model.parameters())
        if compat["missing"] or compat["unexpected"] or params != EXPECTED_PARAMS:
            raise SystemExit(
                f"COMPLETE_E4 FAIL: seed{seed}/{tag} guard failed: "
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
    if branch == "COMPLETE":
        rec["kaggle_record"] = rel_for_record(backup)
        rec["kaggle_t4"] = kaggle_t4_provenance(seed, orig, backup)
    rec["checkpoints"] = checkpoints
    rec["status"] = "COMPLETE"
    rec["scored_locally"] = True
    rec["scoring_device"] = scoring_device
    rec["scoring_torch"] = scoring_torch
    rec["scoring_note"] = SCORING_NOTE
    rec["recovery"] = recovery_ref_for(seed)
    rec["scoring_wall_s"] = round(time.time() - t0, 1)
    rec["scoring_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    canon_path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return {"seed": seed, "branch": branch, "backup_action": backup_action,
            "best_epe": checkpoints["best"]["hailo_val"]["metrics"]["epe"],
            "final_epe": checkpoints["final"]["hailo_val"]["metrics"]["epe"],
            "contract_best": checkpoints["best"]["hailo_val"]["guard"]["contract_match"],
            "contract_final": checkpoints["final"]["hailo_val"]["guard"]["contract_match"]}


def main(argv: list[str] | None = None) -> None:
    import torch

    ap = argparse.ArgumentParser(description="Complete E4 seeds locally.")
    ap.add_argument("--seeds", default="0,1",
                    help="comma-separated seeds to complete (default 0,1)")
    ap.add_argument("--evidence-root", default=None,
                    help="where the pulled Kaggle output of each seed lives: "
                         "either the seed evidence dir itself or a parent dir "
                         "holding per-seed dirs (default: e4ft_err_seed<N>/ for "
                         "seeds 0/1, e4_output/seed2/ for seed 2)")
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                    help="canonical destination parent (default e4_output/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify copies, backups and canonical records and "
                         "report whether everything would be byte-identical; "
                         "copy, score and write nothing")
    args = ap.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    output_root = Path(args.output_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scoring_device = (torch.cuda.get_device_name(0)
                      if device == "cuda" else "cpu")
    scoring_torch = torch.__version__
    print(f"device={device} scoring_device={scoring_device} "
          f"torch={scoring_torch}", flush=True)

    layouts = [(s, evidence_dir_for(s, args.evidence_root),
                dstdir_for(s, output_root)) for s in seeds]
    for seed, evidence_dir, dstdir in layouts:
        rep = step1_copy(seed, srcdir_for(evidence_dir), dstdir,
                         dry_run=args.dry_run)
        for name, info in rep["files"].items():
            print(f"seed{seed} {name}: {info['action']} "
                  f"sha_match={info['sha_match_source']} "
                  f"bytes={info['bytes']}", flush=True)

    if args.dry_run:
        for seed, evidence_dir, dstdir in layouts:
            plan = plan_step2(seed, evidence_dir, dstdir, dry_run=True)
            canon = plan["canon_path"]
            if not plan["fresh"]:
                print(f"seed{seed}: canonical {canon.name} already completed "
                      f"(branch {plan['branch']}, backup "
                      f"{plan['backup_action']}); would keep byte-identical, "
                      f"no rescoring", flush=True)
            elif canon.is_file():
                print(f"seed{seed}: fresh pull at canonical path "
                      f"(branch {plan['branch']}, backup "
                      f"{plan['backup_action']}); would score locally and "
                      f"replace after preservation; nothing written", flush=True)
            else:
                print(f"seed{seed}: would score locally "
                      f"(branch {plan['branch']}, backup "
                      f"{plan['backup_action']}); nothing written", flush=True)
        print("dry-run: nothing copied, scored or written")
        return

    ev = load_eval_module()
    results = []
    for seed, evidence_dir, dstdir in layouts:
        print(f"--- scoring seed {seed} ---", flush=True)
        results.append(step2_score(seed, ev, device, scoring_device,
                                   scoring_torch, evidence_dir, output_root))
    for r in results:
        print(f"seed{r['seed']}: best EPE {r['best_epe']:.7f} "
              f"final EPE {r['final_epe']:.7f} "
              f"contract {r['contract_best']}/{r['contract_final']} "
              f"(branch {r['branch']}, backup {r['backup_action']})")
    print(f"wrote canonical e4_seed<N>.json for seeds "
          f"{','.join(str(s) for s in seeds)}")


if __name__ == "__main__":
    main()
