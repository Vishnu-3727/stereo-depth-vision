#!/usr/bin/env python
"""Assemble the Stage-E E4 finetune Kaggle bundle and record its integrity.

The E4 finetune recipe is byte-identical to E0 (batch 2, 200 epochs, T_max
200, KITTI hailo_calib only); ONLY the init changes, from the Stage-1
checkpoint to the E4 pretrain's best-by-pretrain-val checkpoint. That
checkpoint does not live in this bundle: it ships as its own dataset
(vishnuvardhanksece/stage-e-e4-init, see build_init_dataset_e4.py) and the
finetune kernel resolves it by file name, verifying the sha below.

The expected init sha is a REQUIRED build parameter, not a literal in this
file, because the pretrain checkpoint does not exist yet (pretrain running
on Kaggle). Rebuild with the real sha once it is pulled; never hand-edit a
sha into a committed bundle.

    python stage_e_recipe/kaggle/build_bundle_e4ft.py --init-sha <hex>

Writes `stage_e_recipe/kaggle/bundle_e4ft/` and `source_integrity_e4ft.json`.
Not a training run and not a Kaggle upload.

run_arm.py ships E3's corrected guards (comparisons against args.epochs,
via e3_patch): with EPOCHS=200 they accept exactly what E0's literals
accepted, and they stay correct for any future epoch count. It also ships
the E4 init-sha retarget (via e4ft_patch): the post-hoc wrong_checkpoint_resume
guard compares against the E4 pretrain checkpoint sha passed as --init-sha
instead of the E0 Stage-1 constant. Those are the only recipe-adjacent
deltas versus the E0 bundle; finetune_pilot.py is byte-identical to E0's.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = HERE / "bundle_e4ft"
INIT_FILE_NAME = "e4_pretrain_best.pth"

sys.path.insert(0, str(HERE))
import e3_patch  # noqa: E402  (E3's corrected epoch guards, reused verbatim)
import e4ft_patch  # noqa: E402  (E4 init-sha guard retarget)

SEED1 = "stage_b_armp/20260919T012646Z_tier2_seed1/scripts"

# bundle-relative path  ->  repo-relative source. Identical set to the E0
# bundle MINUS the init checkpoint, which ships as its own dataset.
FILES = {
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
    "scripts/eval_tier2.py": f"{SEED1}/eval_tier2.py",
    # required by finetune_pilot.py:154 integrity_guard(), which loads it as
    # the architecture reference BEFORE any optimizer step. Unchanged: the
    # reference arm is architecture, not initialization.
    "phase1/runs/arm_v/arm_v_best.pth": "phase1/runs/arm_v/arm_v_best.pth",
}

BOOTSTRAP = "kaggle_bootstrap.py"
BOOTSTRAP_SRC = ("stage_b_armp/20260919T035455Z_kaggle_seed2/bundle/"
                 "kaggle_bootstrap.py")

RUN_ARM = "scripts/run_arm.py"
RUN_ARM_SRC = f"{SEED1}/run_arm.py"
DECLARED_EDITS = [
    {"line_was": 'env["P2A_SEED"] = "1"',
     "line_now": 'env["P2A_SEED"] = str(args.seed)',
     "why": "Stage B hardcoded one seed per source copy; Stage E runs seeds "
            "0, 1 and 2 from one file, so the seed becomes a CLI argument."},
    {"line_was": 'env["P2A_EPOCHS"] = "200"',
     "line_now": 'env["P2A_EPOCHS"] = str(args.epochs)',
     "why": "The epoch count becomes a CLI argument (default 200 = frozen "
            "recipe); E4 finetune passes 200 explicitly."},
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
    """The three declared orchestration edits; fail loudly if a target line
    is missing, rather than silently producing a different file."""
    subs = [
        ('    ap.add_argument("--timeout_s", type=float, default=7200.0)\n',
         '    ap.add_argument("--timeout_s", type=float, default=7200.0)\n'
         '    ap.add_argument("--seed", type=int, required=True,\n'
         '                    help="STAGE E: P2A_SEED for this run (0, 1 or 2)")\n'
         '    ap.add_argument("--epochs", type=int, default=200,\n'
         '                    help="STAGE E: P2A_EPOCHS; 200 = frozen recipe")\n'),
        ('    env["P2A_SEED"] = "1"\n', '    env["P2A_SEED"] = str(args.seed)\n'),
        ('    env["P2A_EPOCHS"] = "200"\n', '    env["P2A_EPOCHS"] = str(args.epochs)\n'),
    ]
    for old, new in subs:
        if old not in text:
            sys.exit(f"BUILD FAIL: expected line not found in run_arm.py:\n{old!r}")
        text = text.replace(old, new, 1)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--init-sha", required=True,
                    help="sha256 of the E4 pretrain's best-by-pretrain-val "
                         "checkpoint (from init_integrity.json after pull)")
    ap.add_argument("--init-file", default=INIT_FILE_NAME,
                    help="init file name inside the init dataset "
                         f"(default: {INIT_FILE_NAME})")
    args = ap.parse_args()
    init_sha = args.init_sha.strip().lower()
    if len(init_sha) != 64 or any(c not in "0123456789abcdef" for c in init_sha):
        sys.exit("BUILD FAIL: --init-sha must be a 64-char hex sha256")

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    record: dict = {
        "bundle": "stage_e_recipe/kaggle/bundle_e4ft",
        "purpose": ("Stage E E4 finetune Kaggle bundle: byte-identical E0 "
                    "recipe on the E4 pretrain init (init shipped separately)"),
        "byte_identical_files": {},
        "declared_deltas": {},
        "init": {
            "in_bundle": False,
            "dataset": "vishnuvardhanksece/stage-e-e4-init",
            "file": args.init_file,
            "expected_sha256": init_sha,
            "note": ("Init is external: the finetune kernel resolves it by "
                     "file name under /kaggle/input and asserts this sha "
                     "before training. Rebuild this bundle with the real sha "
                     "once the pretrain checkpoint is pulled."),
        },
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

    # The recipe: byte-identical to E0. No finetune intervention.
    fp_src = REPO / f"{SEED1}/finetune_pilot.py"
    fp_text = fp_src.read_text(encoding="utf-8")
    fp_dst = BUNDLE / "scripts/finetune_pilot.py"
    fp_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fp_src, fp_dst)
    record["byte_identical_files"]["scripts/finetune_pilot.py"] = {
        "repo_source": f"{SEED1}/finetune_pilot.py",
        "sha256": sha256(fp_dst),
        "identical": sha256(fp_src) == sha256(fp_dst)}

    # run_arm.py: orchestration edits PLUS E3's corrected epoch guards PLUS
    # the E4 init-sha guard retarget.
    src = REPO / RUN_ARM_SRC
    original = src.read_text(encoding="utf-8")
    patched = patch_run_arm(original)
    patched = e3_patch.apply(patched)
    patched = e4ft_patch.apply(patched, init_sha)
    run_arm_edits = list(DECLARED_EDITS) + [
        {"line_was": old.strip()[:120], "line_now": new.strip()[:160],
         "why": why} for old, new, why in e3_patch.EDITS] + [
        {"line_was": old.strip()[:120],
         "line_now": new.format(init_sha=init_sha).strip()[:160],
         "why": why} for old, new, why in e4ft_patch.EDITS]
    dst = BUNDLE / RUN_ARM
    dst.write_text(patched, encoding="utf-8", newline="")
    diff = list(difflib.unified_diff(
        original.splitlines(), patched.splitlines(),
        fromfile=f"repo/{RUN_ARM_SRC}", tofile=f"bundle_e4ft/{RUN_ARM}",
        lineterm="", n=1))
    record["declared_deltas"][RUN_ARM] = {
        "repo_source": RUN_ARM_SRC,
        "repo_sha256": sha256(src),
        "bundle_sha256": sha256(dst),
        "identical": False,
        "scope": ("ORCHESTRATION + E3 GUARD FIX + E4 INIT-SHA RETARGET - the "
                  "seed/epochs CLI forwarding, the two post-hoc epoch guards "
                  "comparing against args.epochs instead of the literal 200, "
                  "and the post-hoc init-sha guard comparing against the E4 "
                  "pretrain checkpoint sha instead of the E0 Stage-1 "
                  "constant; no training semantics change"),
        "edits": run_arm_edits,
        "unified_diff": diff,
        "recipe_file_untouched": "scripts/finetune_pilot.py is byte-identical; "
                                 "it alone defines optimizer, loss, schedule, "
                                 "augmentation and the training loop.",
    }

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

    # assemble() requires checkpoints/ among ROOT_DIRS. The init ships
    # separately, so the bundle carries an empty checkpoints dir plus a
    # pointer — never a stale or placeholder checkpoint file.
    ckdir = BUNDLE / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    (ckdir / "README_E4_INIT_IS_EXTERNAL.txt").write_text(
        "The E4 finetune init is NOT in this bundle.\n"
        "It ships as dataset vishnuvardhanksece/stage-e-e4-init "
        f"({args.init_file});\n"
        "the kernel resolves it under /kaggle/input and asserts sha256\n"
        f"{init_sha}\n"
        "before training. This directory exists only because assemble()\n"
        "requires checkpoints/ among ROOT_DIRS.\n", encoding="utf-8")

    # assemble() requires configs/ among ROOT_DIRS, and find_bundle_root()
    # requires the marker.
    (BUNDLE / "configs").mkdir(parents=True, exist_ok=True)
    (BUNDLE / "configs" / "stage_e.json").write_text(json.dumps({
        "campaign": "Stage E recipe optimization",
        "experiment": "e4",
        "initialization": "E4 pretrain best-by-pretrain-val (external init dataset)",
        "init_dataset": "vishnuvardhanksece/stage-e-e4-init",
        "init_file": args.init_file,
        "expected_init_sha256": init_sha,
        "seeds": [0, 1, 2],
        "e4": {"epochs": 200, "batch": 2, "ema": None},
        "note": "Seed is passed to run_arm.py as a CLI argument; epochs is "
                "fixed at 200 (E0 recipe). This file is a record, not a "
                "source of truth.",
    }, indent=2))
    (BUNDLE / "BUNDLE_MARKER.json").write_text(json.dumps({
        "bundle": "stage-e-e4ft",
        "campaign": "Stage E",
        "experiment": "e4",
        "initialization": "E4 pretrain best-by-pretrain-val (external init dataset)",
        "init_dataset": "vishnuvardhanksece/stage-e-e4-init",
        "init_file": args.init_file,
        "expected_init_sha256": init_sha,
        "recipe_source": SEED1,
        "training_authorized": False,
    }, indent=2))

    record["file_count"] = sum(1 for _ in BUNDLE.rglob("*") if _.is_file())
    record["total_bytes"] = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file())
    record["verdict"] = "PASS" if ok else "FAIL"
    manifest = json.dumps(record, indent=2)
    (HERE / "source_integrity_e4ft.json").write_text(manifest)
    # The kernel re-verifies every hash on Kaggle (gate G2 pattern), so the
    # manifest ships inside the bundle too.
    (BUNDLE / "source_integrity.json").write_text(manifest)

    n_ident = sum(1 for v in record["byte_identical_files"].values() if v["identical"])
    print(f"experiment:           e4 (finetune, E0 recipe)")
    print(f"init sha (param):     {init_sha[:16]}...")
    print(f"byte-identical files: {n_ident}/{len(record['byte_identical_files'])}")
    print(f"declared deltas:      {len(record['declared_deltas'])} "
          f"({', '.join(sorted(record['declared_deltas']))})")
    print(f"bundle:               {record['file_count']} files, "
          f"{record['total_bytes'] / 2**20:.2f} MiB")
    print(f"verdict:              {record['verdict']}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
