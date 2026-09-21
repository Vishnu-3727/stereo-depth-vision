#!/usr/bin/env python
"""Push / poll / pull one E0 control seed on Kaggle.

    python stage_e_recipe/kaggle/push_run.py dataset e1
    python stage_e_recipe/kaggle/push_run.py push   0 e1
    python stage_e_recipe/kaggle/push_run.py status 0 e1
    python stage_e_recipe/kaggle/push_run.py pull   0 e1

The experiment defaults to e0 when omitted, matching the E0 runs already made.

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

# Each experiment gets its own immutable bundle: E0 ran against a byte-identical
# recipe, E1 against the EMA-patched one, and neither may be overwritten.
BUNDLES = {"e0": f"{USER}/stage-e-recipe-bundle",
           "e1": f"{USER}/stage-e-e1-bundle",
           "e2": f"{USER}/stage-e-e2-bundle",
           "e3": f"{USER}/stage-e-recipe-bundle"}   # E3 reuses E0's recipe
EPOCHS = {"e0": 200, "e1": 200, "e2": 200, "e3": 400}
INTERVENTION = {"e0": "NONE (frozen incumbent recipe)",
                "e1": "weight EMA, decay 0.999",
                "e2": "native batch 8",
                "e3": "400 epochs"}


def slug(seed: int, exp: str) -> str:
    return f"{USER}/stage-e-{exp}-seed{seed}"


def run(args: list[str]) -> int:
    print("+", " ".join(args), flush=True)
    p = subprocess.run([sys.executable, "-m", "kaggle"] + args,
                       capture_output=True, text=True, errors="replace")
    print(((p.stdout or "") + (p.stderr or "")).strip()[:3000], flush=True)
    return p.returncode


def cmd_push(seed: int, exp: str) -> int:
    d = HERE / f"kernel_{exp}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    src = (HERE / "run_experiment.py").read_text(encoding="utf-8")
    for ph in ("__SEED__", "__EXP__", "__EPOCHS__", "__INTERVENTION__"):
        if ph not in src:
            sys.exit(f"run_experiment.py has no {ph} placeholder")
    src = (src.replace("__SEED__", str(seed)).replace("__EXP__", exp)
              .replace("__EPOCHS__", str(EPOCHS[exp]))
              .replace("__INTERVENTION__", INTERVENTION[exp]))
    name = f"stage-e-{exp}-seed{seed}.py"
    (d / name).write_text(src, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": slug(seed, exp),
        "title": f"Stage E {exp.upper()} seed{seed}",
        "code_file": name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [DATA_SLUG, BUNDLES[exp]],
        "kernel_sources": [],
    }, indent=2))
    return run(["kernels", "push", "-p", str(d)])


def cmd_status(seed: int, exp: str) -> int:
    return run(["kernels", "status", slug(seed, exp)])


def cmd_pull(seed: int, exp: str) -> int:
    out = HERE / f"{exp}_output" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    return run(["kernels", "output", slug(seed, exp), "-p", str(out)])


def cmd_dataset(exp: str) -> int:
    """Upload this experiment's bundle as its own immutable dataset."""
    b = HERE / ("bundle" if exp == "e0" else f"bundle_{exp}")
    (b / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"Stage E {exp} bundle", "id": BUNDLES[exp],
         "licenses": [{"name": "other"}]}, indent=2))
    rc = run(["datasets", "create", "-p", str(b), "-r", "zip", "--dir-mode", "zip"])
    if rc != 0:
        rc = run(["datasets", "version", "-p", str(b), "-m", f"Stage E {exp}",
                  "-r", "zip", "--dir-mode", "zip"])
    return rc


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("push", "status", "pull", "dataset"):
        sys.exit(__doc__)
    exp_ = sys.argv[3] if len(sys.argv) > 3 else "e0"
    if sys.argv[1] == "dataset":
        sys.exit(cmd_dataset(sys.argv[2]))
    s = int(sys.argv[2])
    sys.exit({"push": cmd_push, "status": cmd_status,
              "pull": cmd_pull}[sys.argv[1]](s, exp_))
