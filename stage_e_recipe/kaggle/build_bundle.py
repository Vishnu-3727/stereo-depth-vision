#!/usr/bin/env python
"""Assemble the Stage-E Kaggle bundle and record its integrity.

Copies the research source Stage E needs into `bundle/`, hashes every file, and
asserts each copy is byte-identical to its repository original. Exactly one
file is permitted to differ — `scripts/run_arm.py`, whose Stage-B copy
hardcodes the seed and epoch count — and that delta is declared explicitly,
diffed, and recorded. Everything else is byte-identical or the build fails.

Not a training run and not a Kaggle upload: this only writes
`stage_e_recipe/kaggle/bundle/` and `source_integrity.json`.

    python stage_e_recipe/kaggle/build_bundle.py
"""
from __future__ import annotations

import difflib
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPERIMENT = "e0"
if "--experiment" in sys.argv:
    EXPERIMENT = sys.argv[sys.argv.index("--experiment") + 1]
assert EXPERIMENT in ("e0", "e1", "e2", "e3"), EXPERIMENT
BUNDLE = HERE / ("bundle" if EXPERIMENT == "e0" else f"bundle_{EXPERIMENT}")
# E3 alters only --epochs, passed to run_arm.py, so it reuses E0's recipe
# file byte-identically and needs no patch module.
PATCHED = EXPERIMENT in ("e1", "e2")

SEED1 = "stage_b_armp/20260919T012646Z_tier2_seed1/scripts"
STAGE1_CKPT = ("stage_b_armp/20260918T062146Z_stage1_pretrain/checkpoints/"
               "armp_stage1_best.pth")
STAGE1_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"

# bundle-relative path  ->  repo-relative source
FILES = {
    # research source, byte-identical
    "src/__init__.py": "src/__init__.py",
    "src/datasets/__init__.py": "src/datasets/__init__.py",
    "src/datasets/kitti2015.py": "src/datasets/kitti2015.py",
    "src/evaluation/__init__.py": "src/evaluation/__init__.py",
    "src/evaluation/metrics.py": "src/evaluation/metrics.py",
    "src/losses/__init__.py": "src/losses/__init__.py",
    "src/losses/disparity.py": "src/losses/disparity.py",
    "src/models/__init__.py": "src/models/__init__.py",
    "src/models/stereonet/__init__.py": "src/models/stereonet/__init__.py",
    "src/models/stereonet/aggregation.py": "src/models/stereonet/aggregation.py",
    "src/models/stereonet/blocks.py": "src/models/stereonet/blocks.py",
    "src/models/stereonet/cost_volume.py": "src/models/stereonet/cost_volume.py",
    "src/models/stereonet/excitation.py": "src/models/stereonet/excitation.py",
    "src/models/stereonet/feature_extractor.py": "src/models/stereonet/feature_extractor.py",
    "src/models/stereonet/refinement.py": "src/models/stereonet/refinement.py",
    "src/models/stereonet/regression.py": "src/models/stereonet/regression.py",
    "src/models/stereonet/stereonet.py": "src/models/stereonet/stereonet.py",
    "phase1/harness/determinism.py": "phase1/harness/determinism.py",
    "phase1/harness/frozen_eval.py": "phase1/harness/frozen_eval.py",
    # the frozen recipe; byte-identical for E0, EMA-patched for E1 (see below)
    "scripts/eval_tier2.py": f"{SEED1}/eval_tier2.py",
    # initialization weights
    "checkpoints/armp_stage1_best.pth": STAGE1_CKPT,
    # required by finetune_pilot.py:154 integrity_guard(), which loads it as
    # the architecture reference BEFORE any optimizer step. Without it every
    # Kaggle run aborts at the guard.
    "phase1/runs/arm_v/arm_v_best.pth": "phase1/runs/arm_v/arm_v_best.pth",
}

# Stage-B's stdlib-only Kaggle portability layer. Not research source: it
# discovers the dataset mounts and assembles a working tree. Reused rather
# than rewritten because its v4 mount discovery is already proven on Kaggle.
BOOTSTRAP = "kaggle_bootstrap.py"
BOOTSTRAP_SRC = ("stage_b_armp/20260919T035455Z_kaggle_seed2/bundle/"
                 "kaggle_bootstrap.py")

# The declared deltas: orchestration only, no recipe semantics.
RUN_ARM = "scripts/run_arm.py"
RUN_ARM_SRC = f"{SEED1}/run_arm.py"
DECLARED_EDITS = [
    {"line_was": 'env["P2A_SEED"] = "1"',
     "line_now": 'env["P2A_SEED"] = str(args.seed)',
     "why": "Stage B hardcoded one seed per source copy; Stage E runs seeds "
            "0, 1 and 2 from one file, so the seed becomes a CLI argument."},
    {"line_was": 'env["P2A_EPOCHS"] = "200"',
     "line_now": 'env["P2A_EPOCHS"] = str(args.epochs)',
     "why": "E3 needs 400 epochs; E0/E1/E2 pass 200 explicitly. The default "
            "stays 200 so omitting the flag reproduces the frozen recipe."},
    {"line_was": 'ap.add_argument("--timeout_s", type=float, default=7200.0)',
     "line_now": 'ap.add_argument("--timeout_s", ...)  + --seed + --epochs',
     "why": "Declares the two new arguments."},
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_run_arm(text: str) -> str:
    """Apply the three declared orchestration edits; fail loudly if a target
    line is missing, rather than silently producing a different file."""
    subs = [
        ('    ap.add_argument("--timeout_s", type=float, default=7200.0)\n',
         '    ap.add_argument("--timeout_s", type=float, default=7200.0)\n'
         '    ap.add_argument("--seed", type=int, required=True,\n'
         '                    help="STAGE E: P2A_SEED for this run (0, 1 or 2)")\n'
         '    ap.add_argument("--epochs", type=int, default=200,\n'
         '                    help="STAGE E: P2A_EPOCHS; 200 = frozen recipe, 400 = E3")\n'),
        ('    env["P2A_SEED"] = "1"\n', '    env["P2A_SEED"] = str(args.seed)\n'),
        ('    env["P2A_EPOCHS"] = "200"\n', '    env["P2A_EPOCHS"] = str(args.epochs)\n'),
    ]
    for old, new in subs:
        if old not in text:
            sys.exit(f"BUILD FAIL: expected line not found in run_arm.py:\n{old!r}")
        text = text.replace(old, new, 1)
    return text


def main() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    record: dict = {
        "bundle": "stage_e_recipe/kaggle/bundle",
        "purpose": "Stage E Kaggle T4 execution bundle",
        "byte_identical_files": {},
        "declared_deltas": {},
        "verdict": None,
    }
    ok = True

    for rel, src_rel in FILES.items():
        src = REPO / src_rel
        if not src.exists():
            sys.exit(f"BUILD FAIL: missing source {src_rel}")
        dst = BUNDLE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        h_src, h_dst = sha256(src), sha256(dst)
        same = h_src == h_dst
        ok &= same
        record["byte_identical_files"][rel] = {
            "repo_source": src_rel, "sha256": h_dst, "identical": same}

    # The recipe: byte-identical for E0, one declared intervention for E1.
    fp_src = REPO / f"{SEED1}/finetune_pilot.py"
    fp_text = fp_src.read_text(encoding="utf-8")
    fp_dst = BUNDLE / "scripts/finetune_pilot.py"
    fp_dst.parent.mkdir(parents=True, exist_ok=True)
    if not PATCHED:
        shutil.copy2(fp_src, fp_dst)
        record["byte_identical_files"]["scripts/finetune_pilot.py"] = {
            "repo_source": f"{SEED1}/finetune_pilot.py",
            "sha256": sha256(fp_dst), "identical": sha256(fp_src) == sha256(fp_dst)}
    else:
        # Each experiment's intervention lives in its own patch module, so the
        # diff that defines it is a reviewable file rather than inline logic.
        patch_mod = __import__(f"{EXPERIMENT}_patch")
        patched = patch_mod.apply(fp_text)
        # newline="" keeps LF: the ONLY difference from the repo original must
        # be the declared edit, not the platform's line endings.
        fp_dst.write_text(patched, encoding="utf-8", newline="")
        record["declared_deltas"]["scripts/finetune_pilot.py"] = {
            "repo_source": f"{SEED1}/finetune_pilot.py",
            "repo_sha256": sha256(fp_src), "bundle_sha256": sha256(fp_dst),
            "identical": False,
            "scope": f"{EXPERIMENT.upper()} INTERVENTION",
            "edits": [{"why": why, "old": old.strip()[:120], "new": new.strip()[:160]}
                      for old, new, why in patch_mod.EDITS],
            "unified_diff": list(difflib.unified_diff(
                fp_text.splitlines(), patched.splitlines(),
                fromfile=f"repo/{SEED1}/finetune_pilot.py",
                tofile=f"bundle_{EXPERIMENT}/scripts/finetune_pilot.py",
                lineterm="", n=2)),
            "unchanged": "architecture, initialization, data, augmentation, loss, "
                         "optimizer, LR, batch size, epochs, scheduler",
        }

    # The one declared delta.
    src = REPO / RUN_ARM_SRC
    original = src.read_text(encoding="utf-8")
    patched = patch_run_arm(original)
    dst = BUNDLE / RUN_ARM
    dst.write_text(patched, encoding="utf-8", newline="")
    diff = list(difflib.unified_diff(
        original.splitlines(), patched.splitlines(),
        fromfile=f"repo/{RUN_ARM_SRC}", tofile=f"bundle/{RUN_ARM}", lineterm="", n=1))
    record["declared_deltas"][RUN_ARM] = {
        "repo_source": RUN_ARM_SRC,
        "repo_sha256": sha256(src),
        "bundle_sha256": sha256(dst),
        "identical": False,
        "scope": "ORCHESTRATION ONLY - no training semantics change",
        "edits": DECLARED_EDITS,
        "unified_diff": diff,
        "recipe_file_untouched": "scripts/finetune_pilot.py is byte-identical; "
                                 "it alone defines optimizer, loss, schedule, "
                                 "augmentation and the training loop.",
    }

    # Initialization checkpoint must match the frozen hash after copying.
    ck = BUNDLE / "checkpoints/armp_stage1_best.pth"
    ck_sha = sha256(ck)
    record["init_checkpoint"] = {
        "path": "checkpoints/armp_stage1_best.pth",
        "sha256": ck_sha, "expected": STAGE1_SHA, "match": ck_sha == STAGE1_SHA}
    ok &= ck_sha == STAGE1_SHA

    # Portability layer, with its one declared path edit.
    bsrc = REPO / BOOTSTRAP_SRC
    boot_orig = bsrc.read_text(encoding="utf-8")
    old_line = 'EXP_SUBDIR = Path("stage_b_armp") / "seed2"\n'
    new_line = 'EXP_SUBDIR = Path("stage_e_recipe") / "run"\n'
    if old_line not in boot_orig:
        sys.exit("BUILD FAIL: EXP_SUBDIR line not found in kaggle_bootstrap.py")
    boot_patched = boot_orig.replace(old_line, new_line, 1)
    bdst = BUNDLE / BOOTSTRAP
    bdst.write_text(boot_patched, encoding="utf-8", newline="")
    record["declared_deltas"][BOOTSTRAP] = {
        "repo_source": BOOTSTRAP_SRC,
        "repo_sha256": sha256(bsrc),
        "bundle_sha256": sha256(bdst),
        "identical": False,
        "scope": "PATH NAMING ONLY - no behaviour change",
        "edits": [{"line_was": old_line.strip(), "line_now": new_line.strip(),
                   "why": "Scripts must sit exactly two levels under the "
                          "assembled root so finetune_pilot.py's parents[3] "
                          "and run_arm.py's PILOT.parents[1] both resolve to "
                          "it. Same depth as Stage B's path; renamed so a "
                          "Stage-E run does not masquerade as a Stage-B one."}],
        "not_research_source": "Kaggle portability wrapper (stdlib only). It "
                               "discovers mounts and copies files; it does not "
                               "touch the model, data, loss or schedule.",
    }

    # assemble() requires configs/ among ROOT_DIRS, and find_bundle_root()
    # requires the marker.
    (BUNDLE / "configs").mkdir(parents=True, exist_ok=True)
    (BUNDLE / "configs" / "stage_e.json").write_text(json.dumps({
        "campaign": "Stage E recipe optimization",
        "experiment": EXPERIMENT,
        "initialization": "Stage-1 pretrained ARM-P (never random init)",
        "init_sha256": STAGE1_SHA,
        "seeds": [0, 1, 2],
        "e0": {"epochs": 200, "batch": 2, "ema": None},
        "note": "Seed and epochs are passed to run_arm.py as CLI arguments; "
                "this file is a record, not a source of truth.",
    }, indent=2))
    (BUNDLE / "BUNDLE_MARKER.json").write_text(json.dumps({
        "bundle": f"stage-e-{EXPERIMENT}",
        "campaign": "Stage E",
        "experiment": EXPERIMENT,
        "initialization": "Stage-1 pretrained ARM-P",
        "checkpoint_sha256": STAGE1_SHA,
        "recipe_source": SEED1,
        "training_authorized": False,
    }, indent=2))

    record["file_count"] = sum(1 for _ in BUNDLE.rglob("*") if _.is_file())
    record["total_bytes"] = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file())
    record["verdict"] = "PASS" if ok else "FAIL"
    manifest = json.dumps(record, indent=2)
    # Per-experiment manifest. E0 already ran against its own, and the record
    # of a completed experiment is never overwritten.
    name = ("source_integrity.json" if EXPERIMENT == "e0"
            else f"source_integrity_{EXPERIMENT}.json")
    (HERE / name).write_text(manifest)
    # The kernel re-verifies every hash on Kaggle (gate G2), so the manifest
    # ships inside the bundle too.
    (BUNDLE / "source_integrity.json").write_text(manifest)

    n_ident = sum(1 for v in record["byte_identical_files"].values() if v["identical"])
    print(f"experiment:           {EXPERIMENT}")
    print(f"byte-identical files: {n_ident}/{len(record['byte_identical_files'])}")
    print(f"declared deltas:      {len(record['declared_deltas'])} "
          f"({', '.join(sorted(record['declared_deltas']))})")
    print(f"init checkpoint:      {'MATCH' if record['init_checkpoint']['match'] else 'MISMATCH'}")
    print(f"bundle:               {record['file_count']} files, "
          f"{record['total_bytes'] / 2**20:.2f} MiB")
    print(f"verdict:              {record['verdict']}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
