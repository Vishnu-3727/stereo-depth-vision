#!/usr/bin/env python
"""Push / poll / pull the E4 rate-probe kernel on Kaggle.

    python stage_e_recipe/kaggle/push_e4.py dataset
    python stage_e_recipe/kaggle/push_e4.py push
    python stage_e_recipe/kaggle/push_e4.py status
    python stage_e_recipe/kaggle/push_e4.py pull

One kernel only: the E4 probe is a measurement (200 optimizer steps, no
checkpoints), not a pretrain. The real pretrain gets its own helper in
step 4.

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
KERNEL_SLUG = f"{USER}/stage-e-e4-probe"
KERNEL_DIR = HERE / "kernel_e4_probe"
CODE_FILE = "stage-e-e4-probe.py"

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


def cmd_push() -> int:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_SLUG,
        "title": "Stage E E4 probe",
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [IMAGES_SLUG, DISPARITY_SLUG, BUNDLE_SLUG],
        "kernel_sources": [],
    }, indent=2))
    return run(["kernels", "push", "-p", str(KERNEL_DIR)])


def cmd_status() -> int:
    return run(["kernels", "status", KERNEL_SLUG])


def cmd_pull() -> int:
    out = HERE / "e4_output"
    out.mkdir(parents=True, exist_ok=True)
    return run(["kernels", "output", KERNEL_SLUG, "-p", str(out)])


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in ("push", "status", "pull", "dataset"):
        print(__doc__.strip())
        return 2
    return {"push": cmd_push, "status": cmd_status,
            "pull": cmd_pull, "dataset": cmd_dataset}[argv[1]]()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
