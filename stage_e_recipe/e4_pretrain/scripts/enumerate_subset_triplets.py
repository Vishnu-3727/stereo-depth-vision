"""Enumerate FlyingThings3D_subset TRAIN triplets for E4 pretraining.

E4 corpus layout (two separate Kaggle datasets, so two separate roots):

    <images-root>/FlyingThings3D_subset/train/image_clean/left/<id>.png
    <images-root>/FlyingThings3D_subset/train/image_clean/right/<id>.png
    <disparity-root>/FlyingThings3D_subset/train/disparity/left/<id>.pfm

with <id> = 0000000..0021817 (21,818 train indices).

Writes a manifest with the SAME schema Stage 1 used
(stage_b_armp/20260918T062146Z_stage1_pretrain/dataset_manifest.json):
description, data_root, rng_seed, holdout_frac, n_triplets, n_train,
n_holdout, holdout_ids, train_ids, triplets (each with id/left/right/disp).

Differences from Stage 1, stated explicitly in the manifest `description`:
this is the official FlyingThings3D_subset slice, not TRAIN A+C; and because
the images and disparity mount from two separate Kaggle datasets, triplet
paths are stored ABSOLUTE so the loader resolves them without guessing
(see also images_root / disparity_root recorded in the manifest).

Same holdout protocol as Stage 1: rng_seed 0, holdout_frac 0.05,
np.random.default_rng(0) permutation, n_hold = round(n * frac).

Train split only: the corpus `val` split is ignored; the pretrain's own
holdout comes out of train, exactly as Stage 1 did.

Read-only probe otherwise: no training, no copies, no downloads.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

SEED = 0
HOLDOUT_FRAC = 0.05

# Candidate sub-layouts under each dataset root, in preference order: the
# official FlyingThings3D_subset nesting first, then flatter fallbacks so a
# user may also pass the train dir (or the image_clean / disparity dir) itself.
IMAGE_LEFT_CANDIDATES = (
    Path("FlyingThings3D_subset/train/image_clean/left"),
    Path("train/image_clean/left"),
    Path("image_clean/left"),
    Path("left"),
    Path("."),
)
IMAGE_RIGHT_CANDIDATES = (
    Path("FlyingThings3D_subset/train/image_clean/right"),
    Path("train/image_clean/right"),
    Path("image_clean/right"),
    Path("right"),
)
DISP_CANDIDATES = (
    Path("FlyingThings3D_subset/train/disparity/left"),
    Path("train/disparity/left"),
    Path("disparity/left"),
    Path("left"),
    Path("."),
)


def _resolve(subdir_candidates: tuple[Path, ...], root: Path, what: str) -> Path:
    for sub in subdir_candidates:
        cand = root / sub
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "could not find %s dir under root %s (tried: %s)"
        % (what, root, ", ".join(str(c) for c in subdir_candidates))
    )


def _stems(d: Path, suffix: str) -> set[str]:
    return {p.stem for p in d.glob("*" + suffix) if p.is_file() and p.stat().st_size > 0}


def enumerate_triplets(images_root: Path, disparity_root: Path) -> tuple[list[dict], int]:
    """Return (complete triplets sorted by id, incomplete index count)."""
    left_dir = _resolve(IMAGE_LEFT_CANDIDATES, images_root, "left-image")
    # Right dir must sit next to the resolved left dir when the layout nests
    # them together; otherwise resolve it independently under the root.
    sibling_right = left_dir.parent / "right"
    right_dir = (
        sibling_right
        if sibling_right.is_dir()
        else _resolve(IMAGE_RIGHT_CANDIDATES, images_root, "right-image")
    )
    disp_dir = _resolve(DISP_CANDIDATES, disparity_root, "disparity")

    left_ids = _stems(left_dir, ".png")
    right_ids = _stems(right_dir, ".png")
    disp_ids = _stems(disp_dir, ".pfm")
    complete = sorted(left_ids & right_ids & disp_ids)
    incomplete = len((left_ids | right_ids | disp_ids) - set(complete))
    triplets = [
        {
            "id": idx,
            "left": str((left_dir / (idx + ".png")).resolve()),
            "right": str((right_dir / (idx + ".png")).resolve()),
            "disp": str((disp_dir / (idx + ".pfm")).resolve()),
        }
        for idx in complete
    ]
    return triplets, incomplete


def build_manifest(
    images_root: Path, disparity_root: Path, output_path: Path
) -> dict:
    triplets, n_incomplete = enumerate_triplets(images_root, disparity_root)
    n = len(triplets)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_hold = int(round(n * HOLDOUT_FRAC))
    hold_idx = sorted(perm[:n_hold].tolist())
    train_idx = sorted(perm[n_hold:].tolist())
    manifest = {
        "description": (
            "E4 pretrain (official FlyingThings3D_subset TRAIN, NOT Stage-1 "
            "TRAIN A+C): triplet enumeration + fixed 5% pretrain-validation "
            "holdout. Triplet paths are ABSOLUTE because images and disparity "
            "mount from two separate Kaggle datasets (see images_root / "
            "disparity_root). Pretrain monitoring ONLY; unrelated to the "
            "frozen KITTI contract."
        ),
        "data_root": {
            "images_root": str(images_root.resolve()),
            "disparity_root": str(disparity_root.resolve()),
        },
        "images_root": str(images_root.resolve()),
        "disparity_root": str(disparity_root.resolve()),
        "corpus": "FlyingThings3D_subset",
        "corpus_split": "train",
        "path_style": "absolute",
        "rng_seed": SEED,
        "holdout_frac": HOLDOUT_FRAC,
        "n_triplets": n,
        "n_train": len(train_idx),
        "n_holdout": len(hold_idx),
        "n_incomplete": n_incomplete,
        "holdout_ids": [triplets[i]["id"] for i in hold_idx],
        "train_ids": [triplets[i]["id"] for i in train_idx],
        "triplets": triplets,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, output_path)
    print(
        "triplets=%d incomplete=%d train=%d holdout=%d -> %s"
        % (n, n_incomplete, len(train_idx), len(hold_idx), output_path),
        flush=True,
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images-root", required=True, help="Kaggle mount root of the images dataset")
    ap.add_argument("--disparity-root", required=True, help="Kaggle mount root of the disparity dataset")
    ap.add_argument("--output", required=True, help="output manifest path")
    args = ap.parse_args()
    build_manifest(Path(args.images_root), Path(args.disparity_root), Path(args.output))


if __name__ == "__main__":
    main()
