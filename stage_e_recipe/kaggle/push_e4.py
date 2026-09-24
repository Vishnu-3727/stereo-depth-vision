#!/usr/bin/env python
"""Push / poll / pull the E4 Kaggle kernels: probe and pretrain.

    python stage_e_recipe/kaggle/push_e4.py dataset
    python stage_e_recipe/kaggle/push_e4.py push            # probe
    python stage_e_recipe/kaggle/push_e4.py status          # probe
    python stage_e_recipe/kaggle/push_e4.py pull            # probe
    python stage_e_recipe/kaggle/push_e4.py push-pretrain   # pretrain
    python stage_e_recipe/kaggle/push_e4.py status-pretrain # pretrain
    python stage_e_recipe/kaggle/push_e4.py pull-pretrain   # pretrain
    python stage_e_recipe/kaggle/push_e4.py init-dataset    # init dataset
    python stage_e_recipe/kaggle/push_e4.py push-finetune 0 <init-sha>
    python stage_e_recipe/kaggle/push_e4.py status-finetune 0
    python stage_e_recipe/kaggle/push_e4.py pull-finetune 0

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

# --- E4 finetune -----------------------------------------------------------
# The finetune runs under vishnuvardhanksece (that account holds the KITTI
# subset since the Gate-6 re-upload), byte-identical E0 recipe on the E4
# pretrain init. Three inputs: the finetune bundle, the init dataset built
# by build_init_dataset_e4.py, and the KITTI subset.
def resolve_finetune_user() -> str:
    u = os.environ.get("STAGE_E_KAGGLE_USER", "")
    u = u.strip() if u else ""
    return u if u else "vishnuvardhanksece"


FT_USER = resolve_finetune_user()
FT_BUNDLE_SLUG = f"{FT_USER}/stage-e-e4ft-bundle"
FT_INIT_SLUG = f"{FT_USER}/stage-e-e4-init"
FT_KITTI_SLUG = f"{FT_USER}/kitti2015-tier2-seed2-subset"
FT_INIT_DIR = HERE / "e4_init_dataset"
FT_TEMPLATE = HERE / "run_experiment_e4ft.py"
FT_EPOCHS = 200
FT_INTERVENTION = "E4 broader-pretrain init; E0 recipe"


def ft_slug(seed: int) -> str:
    # Kaggle derives the slug from the title "Stage E E4 finetune seed<N>".
    return f"{FT_USER}/stage-e-e4-finetune-seed{seed}"


def check_init_sha(value: str) -> str:
    v = (value or "").strip().lower()
    if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
        sys.exit("BUILD FAIL: init-sha must be a 64-char hex sha256 "
                 "(from init_integrity.json after pull)")
    return v


def write_kernel_finetune(seed: int, init_sha: str) -> Path:
    """Render kernel_e4ft_seed<seed>/ from the template. File writes only,
    no Kaggle call: the dry run and cmd_push_finetune share this."""
    if seed not in (0, 1, 2):
        sys.exit(f"BUILD FAIL: finetune seed must be 0, 1 or 2, got {seed}")
    init_sha = check_init_sha(init_sha)
    if not FT_TEMPLATE.is_file():
        sys.exit(f"BUILD FAIL: missing template {FT_TEMPLATE}")
    src = FT_TEMPLATE.read_text(encoding="utf-8")
    for ph in ("__SEED__", "__INIT_SHA__"):
        if ph not in src:
            sys.exit(f"run_experiment_e4ft.py has no {ph} placeholder")
    src = src.replace("__SEED__", str(seed)).replace("__INIT_SHA__", init_sha)
    if "__SEED__" in src or "__INIT_SHA__" in src:
        sys.exit("BUILD FAIL: unreplaced placeholder left in finetune kernel")
    d = HERE / f"kernel_e4ft_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    name = f"stage-e-e4ft-seed{seed}.py"
    (d / name).write_text(src, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": ft_slug(seed),
        "title": f"Stage E E4 finetune seed{seed}",
        "code_file": name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [FT_BUNDLE_SLUG, FT_INIT_SLUG, FT_KITTI_SLUG],
        "kernel_sources": [],
    }, indent=2))
    return d


def cmd_init_dataset() -> int:
    """Upload the E4 init folder (built by build_init_dataset_e4.py) as its
    own immutable private dataset."""
    if not (FT_INIT_DIR / "e4_pretrain_best.pth").is_file():
        sys.exit(f"BUILD FAIL: {FT_INIT_DIR} has no e4_pretrain_best.pth; "
                 f"run build_init_dataset_e4.py on the pulled checkpoint first")
    (FT_INIT_DIR / "dataset-metadata.json").write_text(json.dumps(
        {"title": "Stage E E4 finetune init", "id": FT_INIT_SLUG,
         "licenses": [{"name": "other"}]}, indent=2))
    rc = run(["datasets", "create", "-p", str(FT_INIT_DIR),
              "-r", "zip", "--dir-mode", "zip"])
    if rc != 0:
        rc = run(["datasets", "version", "-p", str(FT_INIT_DIR),
                  "-m", "Stage E e4 init", "-r", "zip", "--dir-mode", "zip"])
    return rc


def cmd_push_finetune(seed: int, init_sha: str) -> int:
    d = write_kernel_finetune(seed, init_sha)
    return run(["kernels", "push", "-p", str(d)])


def cmd_status_finetune(seed: int) -> int:
    return run(["kernels", "status", ft_slug(seed)])


def cmd_pull_finetune(seed: int) -> int:
    out = HERE / "e4_output" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    return run(["kernels", "output", ft_slug(seed), "-p", str(out)])


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
                                        "pull-pretrain", "init-dataset",
                                        "push-finetune", "status-finetune",
                                        "pull-finetune"):
        print(__doc__.strip())
        return 2
    if argv[1] == "init-dataset":
        return cmd_init_dataset()
    if argv[1] in ("push-finetune", "status-finetune", "pull-finetune"):
        if len(argv) < 3:
            print(f"usage: push_e4.py {argv[1]} <seed 0|1|2>"
                  + (" <init-sha>" if argv[1] == "push-finetune" else ""))
            return 2
        seed = int(argv[2])
        if argv[1] == "push-finetune":
            sha = (argv[3] if len(argv) > 3 else
                   os.environ.get("STAGE_E_E4_INIT_SHA", ""))
            return cmd_push_finetune(seed, sha)
        return {"status-finetune": cmd_status_finetune,
                "pull-finetune": cmd_pull_finetune}[argv[1]](seed)
    return {"push": cmd_push, "status": cmd_status,
            "pull": cmd_pull, "dataset": cmd_dataset,
            "push-pretrain": cmd_push_pretrain,
            "status-pretrain": cmd_status_pretrain,
            "pull-pretrain": cmd_pull_pretrain}[argv[1]]()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
