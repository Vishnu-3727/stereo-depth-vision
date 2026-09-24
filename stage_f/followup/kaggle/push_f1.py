#!/usr/bin/env python
"""Push / poll / pull the Stage-F F1 pilot seeds on Kaggle.

    python stage_f/followup/kaggle/push_f1.py dataset [arm]
    python stage_f/followup/kaggle/push_f1.py push   f1c
    python stage_f/followup/kaggle/push_f1.py status f1c
    python stage_f/followup/kaggle/push_f1.py pull   f1c

Three arms, seed 0 only, 200 epochs each:

- f1c: concurrent control, the unchanged E0 recipe (E0 bundle).
- f1m: M1 distribution CE, Laplacian b=0.5 candidates, lambda=0.1 (F1M bundle).
- f1m10: M1 distribution CE, Laplacian b=0.5 candidates, lambda=1.0
  (F1M10 bundle, amendment A1).

One kernel per arm. Kernel code is `stage_e_recipe/kaggle/run_experiment.py`
templated exactly like Stage E's push_run.cmd_push. run_experiment.py needs no
functional change for the F1 arms (see module note in cmd_push); only its stale
`e0 | e1 | e2 | e3` comment is refreshed by the templating below.

Owner / auth model (replicates stage_e_recipe/kaggle/push_e4.py exactly):
per-arm OWNER dict below selects the slug owner. STAGE_E_KAGGLE_USER, when set,
overrides ALL arms (global override, same semantics as push_run/push_e4).
There is NO KAGGLE_CONFIG_DIR handling in code -- push_e4.py has none either:
its run() shells out to `python -m kaggle` with ambient credentials and only
varies the slug owner string (resolve_finetune_user defaults to
vishnuvardhanksece). The operator must therefore make the active kaggle.json
match the arm's owner before pushing (verify with `kaggle config view`);
an externally-set KAGGLE_CONFIG_DIR is inherited by the subprocess as-is.
Per-arm owners: f1c and f1m -> vishnu3727; f1m10 -> vishnuvardhanksece
(Kaggle concurrency is 2 GPU sessions per account, so the third arm runs on
the second account).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_EXPERIMENT = REPO / "stage_e_recipe" / "kaggle" / "run_experiment.py"

ARMS = ("f1c", "f1m", "f1m10")
SEED = 0
EPOCHS = {"f1c": 200, "f1m": 200, "f1m10": 200}
INTERVENTION = {"f1c": "NONE - concurrent control (E0 recipe)",
                "f1m": "M1 distribution CE, Laplacian b=0.5 candidates, lambda=0.1",
                "f1m10": "M1 distribution CE, Laplacian b=0.5 candidates, "
                         "lambda=1.0 (amendment A1)"}

OWNER = {"f1c": "vishnu3727",
         "f1m": "vishnu3727",
         "f1m10": "vishnuvardhanksece"}


def owner(arm: str) -> str:
    """Slug owner for an arm.

    STAGE_E_KAGGLE_USER, when set and non-blank, overrides ALL arms (global
    override, same semantics as stage_e push_run/push_e4). Otherwise the
    per-arm OWNER dict applies.
    """
    u = os.environ.get("STAGE_E_KAGGLE_USER", "")
    u = u.strip() if u else ""
    if u:
        return u
    return OWNER[arm]


def resolve_user() -> str:
    """Legacy single-user entrypoint: default owner (kept for compatibility)."""
    u = os.environ.get("STAGE_E_KAGGLE_USER", "")
    u = u.strip() if u else ""
    return u if u else "vishnu3727"


USER = resolve_user()
_ANNOUNCED: set[str] = set()


def data_slug(arm: str) -> str:
    return f"{owner(arm)}/kitti2015-tier2-seed2-subset"


def bundle_slug(arm: str) -> str:
    base = {"f1c": "stage-e-recipe-bundle",
            "f1m": "stage-f-m1-bundle",
            "f1m10": "stage-f-m1-lam1-bundle"}[arm]
    return f"{owner(arm)}/{base}"


DATA_SLUG = f"{USER}/kitti2015-tier2-seed2-subset"
BUNDLES = {"f1c": f"{USER}/stage-e-recipe-bundle",
           "f1m": f"{USER}/stage-f-m1-bundle",
           "f1m10": f"{OWNER['f1m10']}/stage-f-m1-lam1-bundle"}

BUNDLE_DIR = {"f1c": None, "f1m": "bundle_f1m", "f1m10": "bundle_f1m10"}
BUNDLE_TITLE = {"f1m": "Stage F M1 bundle",
                "f1m10": "Stage F M1 lam1 bundle"}


def slug(arm: str) -> str:
    # Kaggle derives the kernel slug from the TITLE below
    # ("Stage F F1M10 seed0" -> stage-f-f1m10-seed0); the id must equal that.
    return f"{owner(arm)}/stage-f-{arm}-seed{SEED}"


def run(args: list[str]) -> int:
    print(f"[kaggle owner: ambient kaggle.json must match slug owner]",
          flush=True)
    print("+", " ".join(args), flush=True)
    p = subprocess.run([sys.executable, "-m", "kaggle"] + args,
                       capture_output=True, text=True, errors="replace")
    print(((p.stdout or "") + (p.stderr or "")).strip()[:3000], flush=True)
    return p.returncode


def cmd_push(arm: str) -> int:
    print(f"[arm {arm} owner: {owner(arm)}]", flush=True)
    d = HERE / f"kernel_{arm}_seed{SEED}"
    d.mkdir(parents=True, exist_ok=True)
    src = RUN_EXPERIMENT.read_text(encoding="utf-8")
    for ph in ("__SEED__", "__EXP__", "__EPOCHS__", "__INTERVENTION__"):
        if ph not in src:
            sys.exit(f"run_experiment.py has no {ph} placeholder")
    src = (src.replace("__SEED__", str(SEED)).replace("__EXP__", arm)
              .replace("__EPOCHS__", str(EPOCHS[arm]))
              .replace("__INTERVENTION__", INTERVENTION[arm]))
    # run_experiment.py needs NO functional change for the F1 arms: EXP only
    # flows into output paths, the record's experiment/intervention strings and
    # checkpoint copy names; there is no branch on EXP in (e0..e3), the param
    # count (397954) and init sha are unchanged by M1, and load_bootstrap finds
    # the single bundle mount via BUNDLE_MARKER.json (present in bundle_f1m).
    # Only the stale experiment-list comment is refreshed, if present.
    old_comment = '# e0 | e1 | e2 | e3'
    if old_comment in src:
        src = src.replace(old_comment, '# f1c | f1m | f1m10', 1)
    name = f"stage-f-{arm}-seed{SEED}.py"
    (d / name).write_text(src, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": slug(arm),
        "title": f"Stage F {arm.upper()} seed{SEED}",
        "code_file": name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [data_slug(arm), bundle_slug(arm)],
        "kernel_sources": [],
    }, indent=2))
    return run(["kernels", "push", "-p", str(d)])


def cmd_status(arm: str) -> int:
    print(f"[arm {arm} owner: {owner(arm)}]", flush=True)
    return run(["kernels", "status", slug(arm)])


def cmd_pull(arm: str) -> int:
    print(f"[arm {arm} owner: {owner(arm)}]", flush=True)
    out = HERE / f"{arm}_output" / f"seed{SEED}"
    out.mkdir(parents=True, exist_ok=True)
    return run(["kernels", "output", slug(arm), "-p", str(out)])


def cmd_dataset(arm: str = "f1m") -> int:
    """Upload an F1M bundle as its own immutable dataset.

    f1m -> vishnu3727/stage-f-m1-bundle from bundle_f1m/;
    f1m10 -> vishnuvardhanksece/stage-f-m1-lam1-bundle from bundle_f1m10/.
    The ambient kaggle.json must authenticate as the arm's owner.
    """
    if arm not in ("f1m", "f1m10"):
        sys.exit(f"dataset arm must be one of ('f1m', 'f1m10'), got {arm!r}")
    print(f"[dataset {arm} owner: {owner(arm)}]", flush=True)
    b = HERE / BUNDLE_DIR[arm]
    (b / "dataset-metadata.json").write_text(json.dumps(
        {"title": BUNDLE_TITLE[arm], "id": bundle_slug(arm),
         "licenses": [{"name": "other"}]}, indent=2))
    rc = run(["datasets", "create", "-p", str(b), "-r", "zip", "--dir-mode", "zip"])
    if rc != 0:
        rc = run(["datasets", "version", "-p", str(b), "-m", BUNDLE_TITLE[arm],
                  "-r", "zip", "--dir-mode", "zip"])
    return rc


def check_arm(arm: str) -> str:
    if arm not in ARMS:
        sys.exit(f"arm must be one of {ARMS}, got {arm!r}")
    return arm


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("push", "status", "pull", "dataset"):
        sys.exit(__doc__)
    if sys.argv[1] == "dataset":
        a = sys.argv[2] if len(sys.argv) > 2 else "f1m"
        if a not in ("f1m", "f1m10"):
            sys.exit(__doc__ + "\ndataset arm must be one of ('f1m', 'f1m10')")
        sys.exit(cmd_dataset(a))
    if len(sys.argv) < 3:
        sys.exit(__doc__ + f"\narm must be one of {ARMS}")
    a = check_arm(sys.argv[2])
    sys.exit({"push": cmd_push, "status": cmd_status,
              "pull": cmd_pull}[sys.argv[1]](a))
