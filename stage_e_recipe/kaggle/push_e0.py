#!/usr/bin/env python
"""Push / poll / pull one E0 control seed on Kaggle.

    python stage_e_recipe/kaggle/push_e0.py push 0
    python stage_e_recipe/kaggle/push_e0.py status 0
    python stage_e_recipe/kaggle/push_e0.py pull 0

One kernel per seed: each is ~1.03 h (measured), so a failure costs one seed
rather than the whole control, and each run's output is committed separately.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER = "vishnu3727"
DATA_SLUG = f"{USER}/kitti2015-tier2-seed2-subset"
BUNDLE_SLUG = f"{USER}/stage-e-recipe-bundle"


def slug(seed: int) -> str:
    return f"{USER}/stage-e-e0-control-seed{seed}"


def run(args: list[str]) -> int:
    print("+", " ".join(args), flush=True)
    p = subprocess.run([sys.executable, "-m", "kaggle"] + args,
                       capture_output=True, text=True, errors="replace")
    print(((p.stdout or "") + (p.stderr or "")).strip()[:3000], flush=True)
    return p.returncode


def cmd_push(seed: int) -> int:
    d = HERE / f"kernel_e0_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    src = (HERE / "e0_run.py").read_text(encoding="utf-8")
    if "__SEED__" not in src:
        sys.exit("e0_run.py has no __SEED__ placeholder")
    name = f"stage-e-e0-control-seed{seed}.py"
    (d / name).write_text(src.replace("__SEED__", str(seed)), encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": slug(seed),
        "title": f"Stage E E0 control seed{seed}",
        "code_file": name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [DATA_SLUG, BUNDLE_SLUG],
        "kernel_sources": [],
    }, indent=2))
    return run(["kernels", "push", "-p", str(d)])


def cmd_status(seed: int) -> int:
    return run(["kernels", "status", slug(seed)])


def cmd_pull(seed: int) -> int:
    out = HERE / "e0_output" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    return run(["kernels", "output", slug(seed), "-p", str(out)])


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("push", "status", "pull"):
        sys.exit(__doc__)
    s = int(sys.argv[2])
    sys.exit({"push": cmd_push, "status": cmd_status, "pull": cmd_pull}[sys.argv[1]](s))
