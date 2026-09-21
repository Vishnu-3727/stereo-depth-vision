#!/usr/bin/env python
"""Upload the Stage-E bundle as a private Kaggle dataset and push the gate
kernel. Does NOT start E0 training - the kernel it pushes runs gates only.

    python stage_e_recipe/kaggle/push_gates.py dataset   # create/version bundle
    python stage_e_recipe/kaggle/push_gates.py kernel    # push + run gates
    python stage_e_recipe/kaggle/push_gates.py status    # poll
    python stage_e_recipe/kaggle/push_gates.py pull      # fetch output
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"
KERNEL_DIR = HERE / "kernel"
USER = "vishnu3727"
DATASET_SLUG = f"{USER}/stage-e-recipe-bundle"
DATA_SLUG = f"{USER}/kitti2015-tier2-seed2-subset"   # reused, not re-uploaded
KERNEL_SLUG = f"{USER}/stage-e-pre-run-gates"


def run(args: list[str]) -> int:
    print("+", " ".join(args), flush=True)
    p = subprocess.run([sys.executable, "-m", "kaggle"] + args,
                       capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print(out.strip()[:4000], flush=True)
    return p.returncode


def cmd_dataset() -> int:
    meta = {
        "title": "Stage E recipe bundle",
        "id": DATASET_SLUG,
        "licenses": [{"name": "other"}],
    }
    (BUNDLE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    # Try create first; if it exists, push a new version instead.
    rc = run(["datasets", "create", "-p", str(BUNDLE), "-r", "zip", "--dir-mode", "zip"])
    if rc != 0:
        rc = run(["datasets", "version", "-p", str(BUNDLE),
                  "-m", "Stage E gates bundle", "-r", "zip", "--dir-mode", "zip"])
    return rc


def cmd_kernel() -> int:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    script = KERNEL_DIR / "stage-e-pre-run-gates.py"
    script.write_text((HERE / "stage_e_gates.py").read_text(encoding="utf-8"),
                      encoding="utf-8")
    meta = {
        "id": KERNEL_SLUG,
        "title": "Stage E pre-run gates",
        "code_file": script.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [DATA_SLUG, DATASET_SLUG],
        "kernel_sources": [],
    }
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return run(["kernels", "push", "-p", str(KERNEL_DIR)])


def cmd_status() -> int:
    return run(["kernels", "status", KERNEL_SLUG])


def cmd_pull() -> int:
    out = HERE / "gate_output"
    out.mkdir(parents=True, exist_ok=True)
    return run(["kernels", "output", KERNEL_SLUG, "-p", str(out)])


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("dataset", "kernel", "status", "pull"):
        sys.exit(__doc__)
    sys.exit({"dataset": cmd_dataset, "kernel": cmd_kernel,
              "status": cmd_status, "pull": cmd_pull}[sys.argv[1]]())
