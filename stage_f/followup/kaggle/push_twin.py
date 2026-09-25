#!/usr/bin/env python
"""Push / poll / pull the Stage-F capacity-twin seeds on Kaggle.

    python stage_f/followup/kaggle/push_twin.py dataset [t48]
    python stage_f/followup/kaggle/push_twin.py push   t48
    python stage_f/followup/kaggle/push_twin.py status t48
    python stage_f/followup/kaggle/push_twin.py pull   t48

Two arms, seed 0 only, 200 epochs each:

- t48: capacity twin, width-48 function-preserving embed of ARM-P Stage-1
  (amendment A2; bundle_twin/, marker `stage-f-twin48`, init
  `checkpoints/twin48_init.pth`).
- t48c: FRESH concurrent control, the unchanged E0 recipe (E0 bundle, ARM-P
  Stage-1 init). NOT a reuse of f1c: f1c was concurrent with M1, not with
  the twin (standing rule: a control is never a historical record).

One kernel per arm. Kernel code is `stage_e_recipe/kaggle/run_experiment.py`
templated exactly like Stage E's push_run.cmd_push (and push_f1.cmd_push).
For t48 the template additionally carries the twin init sha, the 891074
param gate, and the twin48 init path; scoring builds the model from the
bundle's own `eval_tier2.CFG` (48 channels in bundle_twin, 32 in E0), so no
scoring logic is forked here.

Owner / auth model (replicates push_f1.py exactly):
per-arm OWNER dict below selects the slug owner. STAGE_E_KAGGLE_USER, when
set, overrides ALL arms (global override, same semantics as push_run/push_e4).
There is NO KAGGLE_CONFIG_DIR handling in code -- push_f1.py has none
either: the kaggle CLI authenticates via the ambient `kaggle.json` and only
the slug owner string varies. The operator must therefore make the active
kaggle.json match the arm's owner before pushing (verify with
`kaggle config view`); an externally-set KAGGLE_CONFIG_DIR is inherited by
the subprocess as-is.
Per-arm owners: t48 and t48c -> vishnuvardhanksece (both arms on one
account; Kaggle allows 2 concurrent GPU sessions, so both fit).

Dataset lineage (recorded, not assumed):
f1c used `vishnu3727/kitti2015-tier2-seed2-subset` +
`vishnu3727/stage-e-recipe-bundle` (see kernel_f1c_seed0/kernel-metadata.json).
A `vishnuvardhanksece` copy of the KITTI subset EXISTS
(kernel_f1m_seed0/kernel-metadata.json uses
`vishnuvardhanksece/kitti2015-tier2-seed2-subset`; gate6_kitti_verify.json
confirms the slug). NO `vishnuvardhanksece` copy of `stage-e-recipe-bundle`
was found anywhere in push tooling or records: before pushing t48c, the
manager must ensure `vishnuvardhanksece/stage-e-recipe-bundle` exists
(upload one E0-bundle copy under that account; content identical to
`stage_e_recipe/kaggle/bundle/`). The t48 twin bundle uploads via
`dataset t48`.
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
PREFLIGHT_A2 = REPO / "stage_f" / "followup" / "twin" / "preflight_a2.json"

ARMS = ("t48", "t48c")
SEED = 0
EPOCHS = {"t48": 200, "t48c": 200}
INTERVENTION = {"t48": "capacity twin width-48 embed of ARM-P Stage-1 (amendment A2)",
                "t48c": "NONE - fresh concurrent control (E0 recipe, amendment A2)"}

OWNER = {"t48": "vishnuvardhanksece",
         "t48c": "vishnuvardhanksece"}

TWIN_PARAMS = 891074
E0_PARAMS = 397954
ARMP_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"


def twin_sha() -> str:
    """A2-finalized twin init sha (read from preflight_a2.json, must be A2_PASS)."""
    pre = json.loads(PREFLIGHT_A2.read_text(encoding="utf-8"))
    if pre.get("verdict") != "A2_PASS":
        raise SystemExit(f"twin_sha: preflight_a2 verdict is {pre.get('verdict')!r}, not A2_PASS")
    sha = pre["twin_init"]["sha256"]
    if not sha:
        raise SystemExit("twin_sha: empty sha in preflight_a2.json")
    return sha


def owner(arm: str) -> str:
    """Slug owner for an arm.

    STAGE_E_KAGGLE_USER, when set and non-blank, overrides ALL arms (global
    override, same semantics as stage_e push_run/push_e4/push_f1). Otherwise
    the per-arm OWNER dict applies.
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
    return u if u else "vishnuvardhanksece"


USER = resolve_user()


def data_slug(arm: str) -> str:
    return f"{owner(arm)}/kitti2015-tier2-seed2-subset"


def bundle_slug(arm: str) -> str:
    base = {"t48": "stage-f-twin48-bundle",
            "t48c": "stage-e-recipe-bundle"}[arm]
    return f"{owner(arm)}/{base}"


DATA_SLUG = f"{USER}/kitti2015-tier2-seed2-subset"
BUNDLES = {"t48": f"{USER}/stage-f-twin48-bundle",
           "t48c": f"{USER}/stage-e-recipe-bundle"}

BUNDLE_DIR = {"t48": "bundle_twin", "t48c": None}
BUNDLE_TITLE = {"t48": "Stage F twin48 bundle"}


def slug(arm: str) -> str:
    # Kaggle derives the kernel slug from the TITLE below
    # ("Stage F T48 seed0" -> stage-f-t48-seed0); the id must equal that.
    return f"{owner(arm)}/stage-f-{arm}-seed{SEED}"


def run(args: list[str]) -> int:
    print("[kaggle owner: ambient kaggle.json must match slug owner]",
          flush=True)
    print("+", " ".join(args), flush=True)
    p = subprocess.run([sys.executable, "-m", "kaggle"] + args,
                       capture_output=True, text=True, errors="replace")
    print(((p.stdout or "") + (p.stderr or "")).strip()[:3000], flush=True)
    return p.returncode


def write_kernel(arm: str) -> Path:
    """Write the kernel dir for an arm (LOCAL ONLY, no Kaggle call)."""
    check_arm(arm)
    d = HERE / f"kernel_{arm}_seed{SEED}"
    d.mkdir(parents=True, exist_ok=True)
    src = RUN_EXPERIMENT.read_text(encoding="utf-8")
    for ph in ("__SEED__", "__EXP__", "__EPOCHS__", "__INTERVENTION__"):
        if ph not in src:
            sys.exit(f"run_experiment.py has no {ph} placeholder")
    src = (src.replace("__SEED__", str(SEED)).replace("__EXP__", arm)
              .replace("__EPOCHS__", str(EPOCHS[arm]))
              .replace("__INTERVENTION__", INTERVENTION[arm]))
    # run_experiment.py needs NO functional change for the twin arms beyond
    # the init/params identity below: EXP only flows into output paths, the
    # record's experiment/intervention strings and checkpoint copy names;
    # load_bootstrap finds the single bundle mount via BUNDLE_MARKER.json;
    # scoring builds StereoNet from the bundle's own eval_tier2.CFG (48 in
    # bundle_twin, 32 in E0). Only the stale experiment-list comment is
    # refreshed, if present.
    old_comment = "# e0 | e1 | e2 | e3"
    if old_comment in src:
        src = src.replace(old_comment, "# t48 | t48c", 1)
    if arm == "t48":
        sha = twin_sha()
        if ARMP_SHA not in src:
            sys.exit("run_experiment.py has no ARM-P init sha literal to replace")
        src = src.replace(ARMP_SHA, sha)
        if "EXPECTED_PARAMS = 397954" not in src:
            sys.exit("run_experiment.py has no EXPECTED_PARAMS literal to replace")
        src = src.replace("EXPECTED_PARAMS = 397954",
                          f"EXPECTED_PARAMS = {TWIN_PARAMS}")
        old_init = 'init = repo / "checkpoints" / "armp_stage1_best.pth"'
        if old_init not in src:
            sys.exit("run_experiment.py has no armp init path literal to replace")
        src = src.replace(old_init,
                          'init = repo / "checkpoints" / "twin48_init.pth"')
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
    return d


def cmd_push(arm: str) -> int:
    print(f"[arm {arm} owner: {owner(arm)}]", flush=True)
    d = write_kernel(arm)
    print(f"kernel dir: {d}", flush=True)
    return run(["kernels", "push", "-p", str(d)])


def cmd_status(arm: str) -> int:
    print(f"[arm {arm} owner: {owner(arm)}]", flush=True)
    return run(["kernels", "status", slug(arm)])


def cmd_pull(arm: str) -> int:
    print(f"[arm {arm} owner: {owner(arm)}]", flush=True)
    out = HERE / f"{arm}_output" / f"seed{SEED}"
    out.mkdir(parents=True, exist_ok=True)
    return run(["kernels", "output", slug(arm), "-p", str(out)])


def cmd_dataset(arm: str = "t48") -> int:
    """Upload the twin bundle as its own immutable dataset.

    t48 -> vishnuvardhanksece/stage-f-twin48-bundle from bundle_twin/.
    The ambient kaggle.json must authenticate as the arm's owner.
    t48c has no dataset command: the control reuses the unchanged E0 bundle
    (see module docstring for the vishnuvardhanksece/stage-e-recipe-bundle
    prerequisite).
    """
    if arm != "t48":
        sys.exit(f"dataset arm must be 't48' (control t48c reuses the E0 bundle; "
                 f"see module docstring), got {arm!r}")
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
        a = sys.argv[2] if len(sys.argv) > 2 else "t48"
        if a != "t48":
            sys.exit(__doc__ + "\ndataset arm must be 't48'")
        sys.exit(cmd_dataset(a))
    if len(sys.argv) < 3:
        sys.exit(__doc__ + f"\narm must be one of {ARMS}")
    a = check_arm(sys.argv[2])
    sys.exit({"push": cmd_push, "status": cmd_status,
              "pull": cmd_pull}[sys.argv[1]](a))
