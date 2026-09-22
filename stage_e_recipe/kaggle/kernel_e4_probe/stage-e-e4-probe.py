#!/usr/bin/env python
"""Stage E - E4 rate-probe kernel (measurement, NOT a pretrain) on Kaggle.

Discovers the two FlyingThings3D_subset mounts (images + disparity) under
/kaggle/input/, builds the triplet manifest, then runs train_e4_pretrain.py
in probe mode (--max-steps 200 --epochs 1) and records the section-6 gate
number (s/step, s/epoch) into /kaggle/working/e4_probe.json.

PROBE-ONLY BY CONSTRUCTION, not by flag discipline: after the trainer exits
this kernel asserts the probe record exists, the real-run record does NOT,
and no checkpoint was saved. Raising MAX_STEPS below still cannot produce a
pretrain artefact -- probe mode (--max-steps > 0) saves no checkpoints by
design, and this script refuses any other mode. The real pretrain gets its
own kernel in step 4.

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

# --- probe constants (not tunables; see module docstring) -------------------
SEED = 0
EPOCHS = 1
MAX_STEPS = 200          # probe mode: trainer saves NO checkpoints
TIMEOUT_S = 3600.0       # 1 h; a 200-step probe should take minutes
WORKING = Path("/kaggle/working")
OUT = WORKING / "e4_probe.json"
MANIFEST = WORKING / "e4_subset_manifest.json"
PROBE_DIR = WORKING / "e4probe"

IMAGE_SENTINEL = Path("FlyingThings3D_subset/train/image_clean/left")
DISP_SENTINEL = Path("FlyingThings3D_subset/train/disparity/left")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_mounts() -> tuple[Path, Path]:
    """Resolve the images mount and the disparity mount.

    Searches /kaggle/input/* for a FlyingThings3D_subset tree containing
    train/image_clean/left (images) and one containing
    train/disparity/left (disparity). Fails loudly with the directory
    listing if either is missing: a silent wrong mount is the failure mode
    this guards against.
    """
    base = Path("/kaggle/input")
    listing = sorted(str(p) for p in base.iterdir()) if base.is_dir() else []
    images_root = disp_root = None
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if (child / IMAGE_SENTINEL).is_dir() and images_root is None:
                images_root = child
            if (child / DISP_SENTINEL).is_dir() and disp_root is None:
                disp_root = child
    if images_root is None or disp_root is None:
        raise FileNotFoundError(
            "E4 PROBE ABORT: could not resolve both FlyingThings3D_subset mounts "
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
            f"E4 PROBE ABORT: expected 1 stage-e-e4 bundle mount, found "
            f"{len(hits)}: {hits}")
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
            raise SystemExit(f"E4 PROBE ABORT: bundle integrity failed at {rel}")
    tree = WORKING / "e4tree"
    if tree.exists():
        shutil.rmtree(tree)
    shutil.copytree(bundle, tree)
    sys.path.insert(0, str(tree))
    return tree


def main() -> None:
    t_all = time.time()
    rec: dict = {"experiment": "STAGE E - E4 RATE PROBE", "seed": SEED,
                 "max_steps": MAX_STEPS, "epochs": EPOCHS,
                 "note": "measurement only: probe mode saves no checkpoints"}

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
        raise SystemExit("E4 PROBE ABORT: no GPU (rate numbers need the T4)")

    # --- manifest ---------------------------------------------------------
    cmd = [sys.executable, str(scripts / "enumerate_subset_triplets.py"),
           "--images-root", str(images_root),
           "--disparity-root", str(disp_root),
           "--output", str(MANIFEST)]
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(tree))
    if p.returncode != 0 or not MANIFEST.is_file():
        raise SystemExit(f"E4 PROBE ABORT: manifest build exited {p.returncode}")
    manifest = json.loads(MANIFEST.read_text())
    rec["n_triplets"] = manifest["n_triplets"]
    rec["n_incomplete"] = manifest["n_incomplete"]
    rec["n_train"] = manifest["n_train"]
    rec["n_holdout"] = manifest["n_holdout"]
    print(f"triplets={rec['n_triplets']} incomplete={rec['n_incomplete']} "
          f"train={rec['n_train']} holdout={rec['n_holdout']}", flush=True)

    # --- probe ------------------------------------------------------------
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(scripts / "train_e4_pretrain.py"),
           "--manifest", str(MANIFEST), "--out-dir", str(PROBE_DIR),
           "--epochs", str(EPOCHS), "--max-steps", str(MAX_STEPS),
           "--seed", str(SEED)]
    print("+ " + " ".join(cmd), flush=True)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(tree), timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"E4 PROBE ABORT: trainer exceeded {TIMEOUT_S}s timeout")
    rec["probe_wall_s"] = round(time.time() - t0, 1)
    rec["trainer_returncode"] = p.returncode
    if p.returncode != 0:
        OUT.write_text(json.dumps(rec, indent=2))
        raise SystemExit(f"E4 PROBE ABORT: trainer exited {p.returncode}")

    # --- probe-only assertions: a measurement, not a pretrain -------------
    probe_record = PROBE_DIR / "e4_pretrain_rate_probe.json"
    real_record = PROBE_DIR / "e4_pretrain_record.json"
    ckpts = sorted((PROBE_DIR / "checkpoints").glob("*.pth")) \
        if (PROBE_DIR / "checkpoints").is_dir() else []
    if not probe_record.is_file():
        raise SystemExit("E4 PROBE ABORT: probe record missing")
    if real_record.is_file() or ckpts:
        raise SystemExit("E4 PROBE ABORT: probe saved pretrain artefacts "
                         f"(record={real_record.is_file()}, ckpts={ckpts})")
    prec = json.loads(probe_record.read_text())
    rec["rate"] = prec["rate"]
    rec["gpu"] = prec["software"]["gpu"]
    rec["probe_record_sha256"] = sha256(probe_record)

    rec["status"] = "COMPLETE"
    rec["wall_s"] = round(time.time() - t_all, 1)
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(rec, indent=2))
    r = rec["rate"]
    print(f"\nE4 PROBE COMPLETE in {rec['wall_s']}s: "
          f"{r['optimizer_steps']} steps, {r['sec_per_step']:.3f} s/step "
          f"({rec['gpu']})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
