#!/usr/bin/env python
"""Push / poll / pull the Stage-F L1-continuation pilot seeds on Kaggle.

    python stage_f/followup/kaggle/push_l1.py dataset [l1c|l1m]
    python stage_f/followup/kaggle/push_l1.py kernels
    python stage_f/followup/kaggle/push_l1.py push   l1c
    python stage_f/followup/kaggle/push_l1.py status l1c
    python stage_f/followup/kaggle/push_l1.py pull   l1c

Two arms, seed 0 only, 40 epochs each (frozen §8):

- l1c: Smooth-L1 control (the E0 loss, unchanged), continued from E3 seed0
  FINAL with a FRESH Adam lr 1e-4, cosine to 0 over 40 epochs.
- l1m: masked L1 arm (same valid mask/max-disparity, plain L1 instead of
  Smooth-L1), otherwise identical continuation.

One kernel per arm. Kernel code is `stage_e_recipe/kaggle/run_experiment.py`
templated exactly like Stage E's push_run.cmd_push (and push_twin.cmd_push).
For L1 the template additionally carries the E3-final init sha, the
e3_seed0_final init path, the honest "E3 seed0 final (L1 continuation)"
initialization record, and 40 epochs; scoring builds the model from the
bundle's own `eval_tier2.CFG` (32 channels in both L1 bundles — unchanged),
so no scoring logic is forked here. EXPECTED_PARAMS stays 397954.

Owner / auth model (replicates push_twin.py exactly):
per-arm OWNER dict below selects the slug owner. STAGE_E_KAGGLE_USER, when
set, overrides ALL arms (global override, same semantics as
push_run/push_e4/push_f1/push_twin). There is NO KAGGLE_CONFIG_DIR handling
in code: the kaggle CLI authenticates via the ambient `kaggle.json` and only
the slug owner string varies. The operator must therefore make the active
kaggle.json match the arm's owner before pushing (verify with
`kaggle config view`); an externally-set KAGGLE_CONFIG_DIR is inherited by
the subprocess as-is.
Per-arm owners: l1c and l1m -> vishnuvardhanksks (both arms on one
account; Kaggle allows 2 concurrent GPU sessions, so both fit — same
account, no cross-account difference).

Bundles are per-arm (stage-f-l1c-bundle, stage-f-l1m-bundle): the two bundle
contents differ by the loss knob, so they cannot share one immutable dataset
slug (same convention as f1m/f1m10). `kernels` (LOCAL ONLY, no Kaggle call)
writes kernel_l1c_seed0/ and kernel_l1m_seed0/.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_EXPERIMENT = REPO / "stage_e_recipe" / "kaggle" / "run_experiment.py"
E3_FINAL = REPO / "stage_e_recipe" / "kaggle" / "e3_output" / "seed0" / "e3_seed0_final.pth"

ARMS = ("l1c", "l1m")
SEED = 0
EPOCHS = {"l1c": 40, "l1m": 40}
INTERVENTION = {"l1c": "NONE - Smooth-L1 control continued from E3 seed0 final "
                       "(fresh Adam lr 1e-4, cosine 40 epochs; section 8)",
                "l1m": "masked L1 (same valid mask/max-disparity, plain L1 "
                       "instead of Smooth-L1), continued from E3 seed0 final "
                       "(fresh Adam lr 1e-4, cosine 40 epochs; section 8)"}

OWNER = {"l1c": "vishnuvardhanksks",
         "l1m": "vishnuvardhanksks"}

E3_SHA = "82e58bc441a4382ec449479fe26bf479f6b45e9ace4d79b530abeeb40ea79c6d"
ARMP_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"
E0_PARAMS = 397954


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def e3_sha() -> str:
    """E3 seed0 final sha (verified against the local file, must match)."""
    if not E3_FINAL.is_file():
        raise SystemExit(f"e3_sha: E3 seed0 final not found: {E3_FINAL}")
    actual = sha256(E3_FINAL)
    if actual != E3_SHA:
        raise SystemExit(f"e3_sha: local E3 final sha {actual} != {E3_SHA}")
    return E3_SHA


def owner(arm: str) -> str:
    """Slug owner for an arm.

    STAGE_E_KAGGLE_USER, when set and non-blank, overrides ALL arms (global
    override, same semantics as stage_e push_run/push_e4/push_f1/push_twin).
    Otherwise the per-arm OWNER dict applies.
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
    return u if u else "vishnuvardhanksks"


USER = resolve_user()


def data_slug(arm: str) -> str:
    return f"{owner(arm)}/kitti2015-tier2-seed2-subset"


def bundle_slug(arm: str) -> str:
    base = {"l1c": "stage-f-l1c-bundle",
            "l1m": "stage-f-l1m-bundle"}[arm]
    return f"{owner(arm)}/{base}"


DATA_SLUG = f"{USER}/kitti2015-tier2-seed2-subset"
BUNDLES = {"l1c": f"{USER}/stage-f-l1c-bundle",
           "l1m": f"{USER}/stage-f-l1m-bundle"}

BUNDLE_DIR = {"l1c": "bundle_l1c", "l1m": "bundle_l1m"}
BUNDLE_TITLE = {"l1c": "Stage F L1C bundle",
                "l1m": "Stage F L1M bundle"}


def slug(arm: str) -> str:
    # Kaggle derives the kernel slug from the TITLE below
    # ("Stage F L1C seed0" -> stage-f-l1c-seed0); the id must equal that.
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
    sha = e3_sha()
    d = HERE / f"kernel_{arm}_seed{SEED}"
    d.mkdir(parents=True, exist_ok=True)
    src = RUN_EXPERIMENT.read_text(encoding="utf-8")
    for ph in ("__SEED__", "__EXP__", "__EPOCHS__", "__INTERVENTION__"):
        if ph not in src:
            sys.exit(f"run_experiment.py has no {ph} placeholder")
    src = (src.replace("__SEED__", str(SEED)).replace("__EXP__", arm)
              .replace("__EPOCHS__", str(EPOCHS[arm]))
              .replace("__INTERVENTION__", INTERVENTION[arm]))
    # L1 needs no functional change beyond the init identity below: EXP only
    # flows into output paths, the record's experiment/intervention strings
    # and checkpoint copy names; load_bootstrap finds the single bundle mount
    # via BUNDLE_MARKER.json; scoring builds StereoNet from the bundle's own
    # eval_tier2.CFG (32 channels in both L1 bundles, params unchanged at
    # 397954). Only the stale experiment-list comment is refreshed, if
    # present; the init path + sha + initialization record carry the
    # continuation identity.
    old_comment = "# e0 | e1 | e2 | e3"
    if old_comment in src:
        src = src.replace(old_comment, "# l1c | l1m", 1)
    if ARMP_SHA not in src:
        sys.exit("run_experiment.py has no ARM-P init sha literal to replace")
    src = src.replace(ARMP_SHA, sha)
    if f"EXPECTED_PARAMS = {E0_PARAMS}" not in src:
        sys.exit("run_experiment.py has no EXPECTED_PARAMS literal to check")
    old_init = 'init = repo / "checkpoints" / "armp_stage1_best.pth"'
    if old_init not in src:
        sys.exit("run_experiment.py has no armp init path literal to replace")
    src = src.replace(old_init,
                      'init = repo / "checkpoints" / "e3_seed0_final.pth"')
    old_rec = '"initialization": "Stage-1 pretrained ARM-P"'
    if old_rec not in src:
        sys.exit("run_experiment.py has no initialization record literal to replace")
    src = src.replace(old_rec,
                      '"initialization": "E3 seed0 final (L1 continuation, section 8)"')
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


def cmd_kernels() -> int:
    """Write both kernel dirs (LOCAL ONLY, no Kaggle call)."""
    for arm in ARMS:
        d = write_kernel(arm)
        print(f"kernel dir: {d} slug={slug(arm)}", flush=True)
    return 0


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


def cmd_dataset(arm: str = "l1c") -> int:
    """Upload an L1 bundle as its own immutable dataset.

    l1c -> vishnuvardhanksks/stage-f-l1c-bundle from bundle_l1c/;
    l1m -> vishnuvardhanksks/stage-f-l1m-bundle from bundle_l1m/.
    The ambient kaggle.json must authenticate as the arm's owner.
    """
    if arm not in ("l1c", "l1m"):
        sys.exit(f"dataset arm must be one of ('l1c', 'l1m'), got {arm!r}")
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
    if len(sys.argv) < 2 or sys.argv[1] not in ("push", "status", "pull", "dataset", "kernels"):
        sys.exit(__doc__)
    if sys.argv[1] == "kernels":
        sys.exit(cmd_kernels())
    if sys.argv[1] == "dataset":
        a = sys.argv[2] if len(sys.argv) > 2 else "l1c"
        if a not in ("l1c", "l1m"):
            sys.exit(__doc__ + "\ndataset arm must be one of ('l1c', 'l1m')")
        sys.exit(cmd_dataset(a))
    if len(sys.argv) < 3:
        sys.exit(__doc__ + f"\narm must be one of {ARMS}")
    a = check_arm(sys.argv[2])
    sys.exit({"push": cmd_push, "status": cmd_status,
              "pull": cmd_pull}[sys.argv[1]](a))
