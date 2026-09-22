#!/usr/bin/env python
"""Stage E - E4 resume-chained pretrain kernel (8 epochs) on Kaggle.

Runs train_e4_pretrain.py as a real pretrain (checkpoints ARE saved) with
--epochs 8 and --max-wall-s 37800 (10.5 h), leaving margin under Kaggle's
~12 h session cap for bundle assembly and manifest enumeration (measured
7-21 min depending on the session draw). If the wall budget is reached the
trainer stops itself cleanly at an epoch boundary, exits 0 with resume.pt
intact, and this kernel reports that another resume session is needed.

RESUME: searches /kaggle/input recursively for a file named exactly
resume.pt (the same any-depth walk find_mounts uses, since Kaggle mounts
datasets nested at /kaggle/input/datasets/<owner>/<slug>). Zero hits means
a fresh run, which is normal and must not fail. Two or more hits fail
loudly rather than picking one.

A pretrain produces an init; the finetune is a separate experiment. This
kernel does NO contract scoring.

Kaggle script kernels take no arguments, so all parameters are constants.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- pretrain constants (not tunables; see module docstring) -----------------
SEED = 0
EPOCHS = 8
MAX_WALL_S = 37800.0    # 10.5 h; trainer stops itself cleanly before the cap
TIMEOUT_S = 40500.0     # 11.25 h; backstop above the wall budget, below the cap
WORKING = Path("/kaggle/working")
OUT = WORKING / "e4_pretrain.json"
MANIFEST = WORKING / "e4_subset_manifest.json"
PRETRAIN_DIR = WORKING / "e4pretrain"

IMAGE_SENTINEL = Path("FlyingThings3D_subset/train/image_clean/left")
DISP_SENTINEL = Path("FlyingThings3D_subset/train/disparity/left")
KAGGLE_INPUT = Path("/kaggle/input")
RESUME_NAME = "resume.pt"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_mounts(base: Path = KAGGLE_INPUT) -> tuple[Path, Path]:
    """Resolve the images mount and the disparity mount.

    Walks base at any depth for a directory R such that R/IMAGE_SENTINEL
    is a directory (images) or R/DISP_SENTINEL is a directory
    (disparity). Prunes the walk at any FlyingThings3D_subset directory
    so it never descends into the thousands of frame directories. Fails
    loudly with a depth-2 listing if either root is missing: a silent
    wrong mount is the failure mode this guards against.
    """
    base = Path(base)
    images_root = disp_root = None
    if base.is_dir():
        for dp, dn, _fn in os.walk(base):
            dn.sort()
            if "FlyingThings3D_subset" in dn:
                dn.remove("FlyingThings3D_subset")
                candidate = Path(dp)
                if images_root is None and (candidate / IMAGE_SENTINEL).is_dir():
                    images_root = candidate
                if disp_root is None and (candidate / DISP_SENTINEL).is_dir():
                    disp_root = candidate
                if images_root is not None and disp_root is not None:
                    break
                continue
            candidate = Path(dp)
            if images_root is None and (candidate / IMAGE_SENTINEL).is_dir():
                images_root = candidate
            if disp_root is None and (candidate / DISP_SENTINEL).is_dir():
                disp_root = candidate
            if images_root is not None and disp_root is not None:
                break
    if images_root is None or disp_root is None:
        listing: list[str] = []
        if base.is_dir():
            for child in sorted(base.iterdir()):
                listing.append(str(child))
                if child.is_dir():
                    try:
                        for grand in sorted(child.iterdir()):
                            listing.append(str(grand))
                    except OSError:
                        continue
        raise FileNotFoundError(
            "E4 PRETRAIN ABORT: could not resolve both FlyingThings3D_subset mounts "
            f"(images={images_root}, disparity={disp_root}); "
            f"/kaggle/input listing: {listing}")
    return images_root, disp_root


def find_bundle() -> Path:
    """Locate the E4 bundle mount via its BUNDLE_MARKER.json."""
    hits = []
    base = Path("/kaggle/input")
    if base.is_dir():
        for dp, _dn, fn in os.walk(base):
            marker = Path(dp) / "BUNDLE_MARKER.json"
            if "BUNDLE_MARKER.json" in fn:
                try:
                    if json.loads(marker.read_text()).get("bundle") == "stage-e-e4":
                        hits.append(Path(dp))
                except Exception:
                    continue
    if len(hits) != 1:
        raise FileNotFoundError(
            f"E4 PRETRAIN ABORT: expected 1 stage-e-e4 bundle mount, found "
            f"{len(hits)}: {hits}")
    return hits[0]


def find_resume(base: Path = KAGGLE_INPUT) -> Path | None:
    """Locate a resume.pt anywhere under the Kaggle input tree.

    Same any-depth walk discipline as find_mounts: Kaggle mounts datasets
    nested at /kaggle/input/datasets/<owner>/<slug>. Zero hits means a
    fresh run, which is normal and must not fail. Two or more hits fail
    loudly rather than picking one.
    """
    hits: list[Path] = []
    base = Path(base)
    if base.is_dir():
        for dp, _dn, fn in os.walk(base):
            if RESUME_NAME in fn:
                candidate = Path(dp) / RESUME_NAME
                if candidate.is_file():
                    hits.append(candidate)
    if len(hits) == 0:
        return None
    if len(hits) > 1:
        raise FileExistsError(
            f"E4 PRETRAIN ABORT: expected 0 or 1 resume.pt under {base}, "
            f"found {len(hits)}: {hits} (refusing to pick one silently)")
    return hits[0]


def assemble(bundle: Path) -> Path:
    """Copy the bundle into a writable working tree, preserving repo-relative
    layout so train_e4_pretrain.py's parents[3] repo-root resolution holds,
    and put the tree root on sys.path (mirrors how the E0-E3 kernels expose
    the bundle's src/ and phase1/). Re-verifies every bundled hash first."""
    manifest = json.loads((bundle / "source_integrity.json").read_text())
    for rel, entry in manifest["byte_identical_files"].items():
        f = bundle / rel
        if not f.is_file() or sha256(f) != entry["sha256"]:
            raise SystemExit(f"E4 PRETRAIN ABORT: bundle integrity failed at {rel}")
    tree = WORKING / "e4tree"
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(bundle, tree)
    sys.path.insert(0, str(tree))
    return tree


def main() -> None:
    t_all = time.time()
    rec: dict = {"experiment": "STAGE E - E4 PRETRAIN", "seed": SEED,
                 "epochs": EPOCHS, "max_wall_s": MAX_WALL_S,
                 "note": "real run: checkpoints saved; resume-chained via resume.pt"}

    images_root, disp_root = find_mounts()
    rec["images_mount"] = str(images_root)
    rec["disparity_mount"] = str(disp_root)
    print(f"images mount:     {images_root}", flush=True)
    print(f"disparity mount:  {disp_root}", flush=True)

    tree = assemble(find_bundle())
    scripts = tree / "stage_e_recipe" / "e4_pretrain" / "scripts"
    rec["bundle_files_verified"] = len(json.loads(
        (tree / "source_integrity.json").read_text())["byte_identical_files"])

    import torch
    rec["environment"] = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    try:
        import numpy, cv2
        rec["environment"]["numpy"] = numpy.__version__
        rec["environment"]["opencv"] = cv2.__version__
    except Exception:
        pass
    print(json.dumps(rec["environment"]), flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("E4 PRETRAIN ABORT: no GPU (pretrain needs the T4)")

    # --- resume discovery --------------------------------------------------
    resume_src = find_resume()
    if resume_src is None:
        print("E4 PRETRAIN: no resume.pt under /kaggle/input; fresh run", flush=True)
        rec["resumed"] = False
        rec["resume_src"] = None
    else:
        print(f"E4 PRETRAIN: resuming from {resume_src}", flush=True)
        rec["resumed"] = True
        rec["resume_src"] = str(resume_src)

    # --- manifest ---------------------------------------------------------
    cmd = [sys.executable, str(scripts / "enumerate_subset_triplets.py"),
           "--images-root", str(images_root),
           "--disparity-root", str(disp_root),
           "--output", str(MANIFEST)]
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(tree))
    if p.returncode != 0 or not MANIFEST.is_file():
        raise SystemExit(f"E4 PRETRAIN ABORT: manifest build exited {p.returncode}")
    manifest = json.loads(MANIFEST.read_text())
    rec["n_triplets"] = manifest["n_triplets"]
    rec["n_incomplete"] = manifest["n_incomplete"]
    rec["n_train"] = manifest["n_train"]
    rec["n_holdout"] = manifest["n_holdout"]
    print(f"triplets={rec['n_triplets']} incomplete={rec['n_incomplete']} "
          f"train={rec['n_train']} holdout={rec['n_holdout']}", flush=True)

    # --- pretrain ---------------------------------------------------------
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(scripts / "train_e4_pretrain.py"),
           "--manifest", str(MANIFEST), "--out-dir", str(PRETRAIN_DIR),
           "--epochs", str(EPOCHS), "--max-wall-s", str(MAX_WALL_S),
           "--seed", str(SEED)]
    if resume_src is not None:
        cmd += ["--resume", str(resume_src)]
    print("+ " + " ".join(cmd), flush=True)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(tree), timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"E4 PRETRAIN ABORT: trainer exceeded {TIMEOUT_S}s timeout")
    rec["pretrain_wall_s"] = round(time.time() - t0, 1)
    rec["trainer_returncode"] = p.returncode
    if p.returncode != 0:
        OUT.write_text(json.dumps(rec, indent=2))
        raise SystemExit(f"E4 PRETRAIN ABORT: trainer exited {p.returncode}")

    # --- artefacts: hash, copy into /kaggle/working, run-state ------------
    train_record = PRETRAIN_DIR / "e4_pretrain_record.json"
    if not train_record.is_file():
        raise SystemExit("E4 PRETRAIN ABORT: run record missing at "
                         f"{train_record}")
    prec = json.loads(train_record.read_text())
    rec["stopped_for_wall_budget"] = bool(prec.get("stopped_for_wall_budget", False))
    rec["epochs_run"] = prec.get("epochs_run")
    rec["best_epoch"] = prec.get("best_epoch")
    rec["best_pretrain_val_epe"] = prec.get("best_pretrain_val_epe")

    best_ckpt = PRETRAIN_DIR / "checkpoints" / "e4_pretrain_best.pth"
    resume_ckpt = PRETRAIN_DIR / "checkpoints" / "resume.pt"
    jsonl_log = PRETRAIN_DIR / "pretrain_log.jsonl"
    for expect in (best_ckpt, resume_ckpt, jsonl_log, train_record):
        if not expect.is_file():
            raise SystemExit(f"E4 PRETRAIN ABORT: expected artefact missing: {expect}")
    rec["best_sha256"] = sha256(best_ckpt)
    rec["resume_sha256"] = sha256(resume_ckpt)
    print(f"best checkpoint sha256:   {rec['best_sha256']}", flush=True)
    print(f"resume checkpoint sha256: {rec['resume_sha256']}", flush=True)
    for src in (best_ckpt, resume_ckpt, jsonl_log, train_record):
        dst = WORKING / src.name
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}", flush=True)
    rec["copied_to_working"] = [best_ckpt.name, resume_ckpt.name,
                                jsonl_log.name, train_record.name]

    if rec["stopped_for_wall_budget"]:
        rec["status"] = "WALL_BUDGET_STOP"
        verdict = (f"ran {rec['epochs_run']}/{EPOCHS} epochs then stopped on "
                   f"the wall budget; another resume session is needed")
    elif rec["epochs_run"] == EPOCHS:
        rec["status"] = "COMPLETE"
        verdict = (f"finished all {EPOCHS} epochs; "
                   f"no further session needed")
    else:
        raise SystemExit(
            f"E4 PRETRAIN ABORT: epochs_run={rec['epochs_run']} but "
            f"stopped_for_wall_budget={rec['stopped_for_wall_budget']} "
            f"(expected {EPOCHS} epochs or a wall-budget stop)")

    rec["wall_s"] = round(time.time() - t_all, 1)
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(rec, indent=2))
    print(f"\nE4 PRETRAIN {rec['status']} in {rec['wall_s']}s: {verdict}")
    print(f"wrote {OUT}")


def _demo_find_resume() -> None:
    """Self-check: resume.pt nested two levels deep resolves via find_resume()."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        assert find_resume(base) is None
        nested = base / "datasets" / "owner" / "slug"
        nested.mkdir(parents=True)
        want = nested / RESUME_NAME
        want.write_bytes(b"x")
        assert find_resume(base) == want
        other = base / "datasets" / "owner" / "other"
        other.mkdir(parents=True)
        (other / RESUME_NAME).write_bytes(b"y")
        try:
            find_resume(base)
        except FileExistsError:
            pass
        else:
            raise AssertionError("two resume.pt hits must fail loudly")


if __name__ == "__main__":
    if os.environ.get("E4_PRETRAIN_SELF_CHECK") == "1" and not Path("/kaggle/input").exists():
        _demo_find_resume()
    else:
        main()
