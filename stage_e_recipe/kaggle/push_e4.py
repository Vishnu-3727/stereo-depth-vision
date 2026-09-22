#!/usr/bin/env python
"""Push / poll / pull the E4 Kaggle kernels: probe and pretrain.

    python stage_e_recipe/kaggle/push_e4.py dataset
    python stage_e_recipe/kaggle/push_e4.py push            # probe
    python stage_e_recipe/kaggle/push_e4.py status          # probe
    python stage_e_recipe/kaggle/push_e4.py pull            # probe
    python stage_e_recipe/kaggle/push_e4.py push-pretrain   # pretrain
    python stage_e_recipe/kaggle/push_e4.py status-pretrain # pretrain
    python stage_e_recipe/kaggle/push_e4.py pull-pretrain   # pretrain

Two kernels: the probe is a measurement (200 optimizer steps, no
checkpoints), not a pretrain; the pretrain is the 8-epoch resume-chained
run (checkpoints saved, --max-wall-s 37800, optional --resume from a
resume.pt dataset). Each kernel has its own slug, its own kernel dir, and
its own metadata writer, and is pushed / polled / pulled separately.

The pretrain kernel's dataset_sources are the two corpus datasets plus the
bundle, plus the resume dataset `<user>/stage-e-e4-resume` when it exists.
Existence is signalled locally via STAGE_E_E4_WITH_RESUME=1 (set once the
resume dataset has been uploaded); the resume dataset is never mandatory
and the default push omits it.

LOCAL-ONLY file writer: like push_run.py, this only writes kernel metadata
files and shells out to `kaggle`. It performs NO Kaggle API call on import,
and running it with no arguments prints usage and exits non-zero.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def resolve_user() -> str:
    """Kaggle dataset/kernel owner, overridable via STAGE_E_KAGGLE_USER."""
    u = os.environ.get("STAGE_E_KAGGLE_USER", "")
    u = u.strip() if u else ""
    return u if u else "vishnu3727"


USER = resolve_user()
_USER_ANNOUNCED = False

BUNDLE_SLUG = f"{USER}/stage-e-e4-bundle"
RESUME_SLUG = f"{USER}/stage-e-e4-resume"

PROBE_SLUG = f"{USER}/stage-e-e4-probe"
PROBE_DIR = HERE / "kernel_e4_probe"
PROBE_CODE_FILE = "stage-e-e4-probe.py"

PRETRAIN_SLUG = f"{USER}/stage-e-e4-pretrain"
PRETRAIN_DIR = HERE / "kernel_e4_pretrain"
PRETRAIN_CODE_FILE = "stage-e-e4-pretrain.py"
PRETRAIN_OUT = HERE / "e4_pretrain_output"

# E4's corpus: two public datasets (read-only mounts on Kaggle).
IMAGES_SLUG = "arjun12367/sceneflow-flyingthings-images"
DISPARITY_SLUG = "arjun12367/sceneflow-flyingthings-disparity"


def run(args: list[str]) -> int:
    global _USER_ANNOUNCED
    if not _USER_ANNOUNCED:
        print(f"[kaggle user: {USER}]", flush=True)
        _USER_ANNOUNCED = True
    print("+", " ".join(args), flush=True)
    p = subprocess.run([sys.executable, "-m", "kaggle"] + args,
                       capture_output=True, text=True, errors="replace")
    print(((p.stdout or "") + (p.stderr or "")).strip()[:3000], flush=True)
    return p.returncode


def cmd_dataset() -> int:
    """Upload the E4 bundle as its own immutable dataset."""
    b = HERE / "bundle_e4"
    (b / "dataset-metadata.json").write_text(json.dumps(
        {"title": "Stage E E4 bundle", "id": BUNDLE_SLUG,
         "licenses": [{"name": "other"}]}, indent=2))
    rc = run(["datasets", "create", "-p", str(b), "-r", "zip", "--dir-mode", "zip"])
    if rc != 0:
        rc = run(["datasets", "version", "-p", str(b), "-m", "Stage E e4",
                  "-r", "zip", "--dir-mode", "zip"])
    return rc


def pretrain_dataset_sources() -> list[str]:
    """Corpus + bundle, plus the resume dataset slug when it exists.

    "When it exists" is signalled locally: set STAGE_E_E4_WITH_RESUME=1
    once the resume dataset has been uploaded. Default omits it, so the
    resume dataset is never mandatory.
    """
    sources = [IMAGES_SLUG, DISPARITY_SLUG, BUNDLE_SLUG]
    if os.environ.get("STAGE_E_E4_WITH_RESUME", "").strip().lower() in ("1", "true", "yes"):
        sources.append(RESUME_SLUG)
    return sources


def cmd_push() -> int:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    (PROBE_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": PROBE_SLUG,
        "title": "Stage E E4 probe",
        "code_file": PROBE_CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [IMAGES_SLUG, DISPARITY_SLUG, BUNDLE_SLUG],
        "kernel_sources": [],
    }, indent=2))
    return run(["kernels", "push", "-p", str(PROBE_DIR)])


def cmd_status() -> int:
    return run(["kernels", "status", PROBE_SLUG])


def cmd_pull() -> int:
    out = HERE / "e4_output"
    out.mkdir(parents=True, exist_ok=True)
    return run(["kernels", "output", PROBE_SLUG, "-p", str(out)])


def cmd_push_pretrain() -> int:
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    (PRETRAIN_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": PRETRAIN_SLUG,
        "title": "Stage E E4 pretrain",
        "code_file": PRETRAIN_CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": pretrain_dataset_sources(),
        "kernel_sources": [],
    }, indent=2))
    return run(["kernels", "push", "-p", str(PRETRAIN_DIR)])


def cmd_status_pretrain() -> int:
    return run(["kernels", "status", PRETRAIN_SLUG])


def cmd_pull_pretrain() -> int:
    PRETRAIN_OUT.mkdir(parents=True, exist_ok=True)
    return run(["kernels", "output", PRETRAIN_SLUG, "-p", str(PRETRAIN_OUT)])


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in ("push", "status", "pull", "dataset",
                                        "push-pretrain", "status-pretrain",
                                        "pull-pretrain"):
        print(__doc__.strip())
        return 2
    return {"push": cmd_push, "status": cmd_status,
            "pull": cmd_pull, "dataset": cmd_dataset,
            "push-pretrain": cmd_push_pretrain,
            "status-pretrain": cmd_status_pretrain,
            "pull-pretrain": cmd_pull_pretrain}[argv[1]]()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
