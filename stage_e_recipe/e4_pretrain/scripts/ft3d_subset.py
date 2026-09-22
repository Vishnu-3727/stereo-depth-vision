"""FlyingThings3D_subset triplets for E4 pretraining.

Same class interface as Stage-1
(stage_b_armp/20260918T062146Z_stage1_pretrain/scripts/ft3d.py):
split='train' -> the 95% train ids; split='pretrain_val' -> the fixed random
5% holdout (rng seed 0). __len__ / __getitem__ return the same StereoSample
shape, so the Stage-1 trainer can use this with a one-line swap.

TWO REQUIRED DIFFERENCES FROM ft3d.py, both loud, not silent:

1. The E4 corpus PFMs store disparity NEGATIVE (measured:
   stage_e_recipe/E4_DATA_AVAILABILITY.md section 1, raw ranges around
   -139..-1.1, 100% finite, 540x960 single-channel). The reading is therefore
   |d| with the standard right(x - d) direction (verified 3.4x-4.8x better
   than the unshifted control, opposite sign worse than control;
   E4_DATA_AVAILABILITY.md section 4). Applied as-is, the training mask
   (gt > 0 AND gt < 184) would yield ZERO valid pixels, so the abs() below
   is load-bearing. It lives in exactly one place, marked SIGN FIX.

2. The loader REFUSES positive-signed input: any positive finite value in a
   loaded PFM raises instead of being silently abs()-ed. The corpus is
   uniformly negative, so positive data means the wrong data is mounted and
   the run must stop.

The colour-PFM refusal from ft3d.py is kept.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

import sys

sys.path.insert(0, str(REPO_ROOT))

from src.datasets.kitti2015 import StereoSample  # noqa: E402
from src.datasets.pfm import read_pfm  # noqa: E402


def read_image(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return bgr[:, :, ::-1].copy()  # to RGB, mirrors kitti2015.read_image


def _resolve_triplet_path(p: str | Path) -> Path:
    """E4 manifests store ABSOLUTE paths (dual Kaggle roots); accept a
    repo-relative path as well so Stage-1-style manifests still resolve."""
    path = Path(p)
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / p


class FlyingThings3DSubset:
    """Usable FlyingThings3D_subset TRAIN triplets from a manifest.

    split='train' -> the 95% train ids; split='pretrain_val' -> the fixed
    random 5% holdout (rng seed 0). Pretrain monitoring ONLY; unrelated to
    the frozen KITTI contract. No KITTI data is touched here.
    """

    def __init__(self, manifest_path: str | Path, split: str = "train") -> None:
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        by_id = {t["id"]: t for t in manifest["triplets"]}
        if split == "train":
            ids = manifest["train_ids"]
        elif split == "pretrain_val":
            ids = manifest["holdout_ids"]
        else:
            raise ValueError("unknown split: " + split)
        self.split = split
        self.ids = list(ids)
        self.records = [by_id[i] for i in self.ids]
        self.nonfinite_sanitized = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> StereoSample:
        rec = self.records[i]
        left = read_image(_resolve_triplet_path(rec["left"]))
        right = read_image(_resolve_triplet_path(rec["right"]))
        disp = read_pfm(_resolve_triplet_path(rec["disp"]))
        if disp.ndim == 3:  # colour PFM has no meaning as disparity; fail loud
            raise ValueError("colour PFM as disparity: " + rec["disp"])
        finite = np.isfinite(disp)
        if bool((disp[finite] > 0).any()):
            # The E4 corpus is uniformly negative-signed (see module docstring
            # and stage_e_recipe/E4_DATA_AVAILABILITY.md); positive values mean
            # the wrong data is mounted. Stop the run instead of abs()-ing it.
            raise ValueError(
                "positive-signed disparity PFM (expected uniformly negative "
                "FlyingThings3D_subset data): " + rec["disp"]
            )
        # SIGN FIX (exactly one place): this corpus stores disparity negative
        # (E4_DATA_AVAILABILITY.md section 1); the trainer's valid mask needs
        # positive px, so read |d|. The refusal above guarantees we only ever
        # abs() genuinely negative-signed data.
        disp = np.abs(disp)
        disp = np.ascontiguousarray(disp, dtype=np.float32)
        bad = ~np.isfinite(disp)
        if bool(bad.any()):
            self.nonfinite_sanitized += int(bad.sum())
            disp = disp.copy()
            disp[bad] = 0.0  # 0 = no ground truth, excluded by the valid mask
        h, w = disp.shape
        assert left.shape[:2] == (h, w) and right.shape[:2] == (h, w), (
            rec["id"],
            left.shape,
            right.shape,
            disp.shape,
        )
        return StereoSample(
            name=rec["id"],
            left=left,
            right=right,
            disparity=disp,
            original_shape=(h, w),
        )
