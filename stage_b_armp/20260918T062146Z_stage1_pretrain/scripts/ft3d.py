"""FlyingThings3D TRAIN triplets (A+C subset staged on C:).

NOT full SceneFlow: B wholly missing, Monkaa empty. See README.md.

Read-only dataset class for ARM P Stage 1 (pretrain only). Imports the
frozen PFM reader (src/datasets/pfm.py); nothing in src/ is modified.
Triplet layout under data/sceneflow/flyingthings3d/extracted/:
  frames_cleanpass/TRAIN/{A,C}/{seq}/{left,right}/{frame}.png
  disparity/TRAIN/{A,C}/{seq}/left/{frame}.pfm
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


class FlyingThings3DTrain:
    """Usable FT3D TRAIN triplets from a dataset_manifest.json.

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
        self.root = REPO_ROOT
        self.nonfinite_sanitized = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> StereoSample:
        rec = self.records[i]
        left = read_image(self.root / rec["left"])
        right = read_image(self.root / rec["right"])
        disp = read_pfm(self.root / rec["disp"])
        if disp.ndim == 3:  # colour PFM has no meaning as disparity; fail loud
            raise ValueError("colour PFM as disparity: " + rec["disp"])
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
