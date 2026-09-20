"""SceneFlow/Driving stereo (synthetic pretraining source for ARM D).

Real extracted layout (VERIFIED 2026-09-14 from
``data/sceneflow/driving/driving__frames_cleanpass.tar`` +
``driving__disparity.tar.bz2`` extracted under
``data/sceneflow/driving/extracted/``):

- frames root: ``frames_cleanpass/<focal>/<direction>/<speed>/{left,right}/<frame>.png``
- disparity root: ``disparity/<focal>/<direction>/<speed>/{left,right}/<frame>.pfm``

with ``focal`` in {35mm_focallength, 15mm_focallength}, ``direction`` in
{scene_backwards, scene_forwards}, ``speed`` in {fast, slow}.
4400 stereo pairs total (fast subsets 300 pairs each, slow subsets 800 each).

Resolution of the earlier guesses (see git history): the archive top-level
names do NOT contain 'driving' (guess 1 wrong: matching is on 'frames' /
'cleanpass' / 'disparity' substrings instead); disparity exists for BOTH
views, not left-only (guess 4 wrong -- the class enumerates left-view
triplets and ignores the right-view PFMs); frames are .png (guess 2
confirmed); the 2x2x2 subpath names match listflowfile.py verbatim (guess 3
confirmed).

Enumeration logic itself still mirrors the upstream enumeration
[SR-upstream]
``reference/upstream/extracted/StereoNet-master/dataloader/listflowfile.py``
(driving section, READ ONLY -- not imported).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.datasets.kitti2015 import StereoSample, read_image
from src.datasets.pfm import read_pfm

# VERIFIED against the real extraction 2026-09-14 (see module docstring).
# Remaining notes:
#  1. Top-level directory names. The real archives extract to
#     'frames_cleanpass' and 'disparity' (no 'driving' in either name), so
#     matching keys off 'frames'/'cleanpass' and 'disparity'. Anything else
#     is an error -- a guessed layout must fail loudly.
#  2. Frame file extension. Real frames are .png; the enumeration still
#     accepts upstream's full jpg/png/ppm/bmp set for tolerance.
#  3. The 2x2x2 subpath set (focal/direction/speed names) VERIFIED verbatim.
#  4. Disparity exists for BOTH views (left/*.pfm and right/*.pfm); the class
#     enumerates left-view triplets only and ignores the right-view PFMs.
#  5. PFM disparity is in PIXELS already (upstream sceneflow-pretrain.py uses
#     it raw against a maxdisp of 160). It must NOT be divided by 256 --
#     that divisor is KITTI-only.
#  6. Frame size is 540x960 (HxW) per the SceneFlow paper -- VERIFIED on the
#     real extraction via the one-sample check (see task report).

FOCAL_LENGTHS = ("35mm_focallength", "15mm_focallength")
DIRECTIONS = ("scene_backwards", "scene_forwards")
SPEEDS = ("fast", "slow")

IMG_EXTENSIONS = (
    ".jpg", ".JPG", ".jpeg", ".JPEG",
    ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
)


@dataclass
class DrivingPair:
    left_image: Path
    right_image: Path
    disparity_pfm: Path


def _find_subdir(
    root: Path, required: tuple[str, ...], preferred: tuple[str, ...] = ()
) -> Path:
    """Mirror listflowfile.py's substring matching for the archive top level.

    ``required`` substrings must all appear in the directory name; when
    several match, ``preferred`` substrings narrow the field (first one that
    isolates a single candidate wins). Anything else is an error, because a
    guessed layout must fail loudly, not silently train on the wrong files.
    """
    candidates = [
        p for p in root.iterdir()
        if p.is_dir() and all(s in p.name for s in required)
    ]
    if not candidates:
        raise FileNotFoundError(
            "no *{}* subdirectory under {}".format("*".join(required), root)
            + ". Expected the Driving archives extracted here; see module docstring."
        )
    if len(candidates) == 1:
        return candidates[0]
    for s in preferred:
        narrowed = [p for p in candidates if s in p.name]
        if len(narrowed) == 1:
            return narrowed[0]
    raise FileNotFoundError(
        "ambiguous Driving layout under {}: {}".format(
            root, sorted(p.name for p in candidates))
        + ". Expected exactly one *{}* subdirectory.".format("*".join(required))
    )


class DrivingStereo:
    """SceneFlow/Driving frames + left-view PFM disparity, enumerated over the
    2x2x2 (focal x direction x speed) grid from listflowfile.py."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                "SceneFlow/Driving not found at " + str(self.root)
                + ". Expected the Driving archives (driving__frames_cleanpass.tar +"
                " driving__disparity.tar.bz2) extracted under this directory."
            )
        frames_root = _find_subdir(self.root, ("frames",), ("cleanpass",))
        disparity_root = _find_subdir(self.root, ("disparity",))
        self.frames_root = frames_root
        self.disparity_root = disparity_root

        pairs: list[DrivingPair] = []
        skipped = 0
        for focal in FOCAL_LENGTHS:
            for direction in DIRECTIONS:
                for speed in SPEEDS:
                    left_dir = self.frames_root / focal / direction / speed / "left"
                    right_dir = self.frames_root / focal / direction / speed / "right"
                    disp_dir = self.disparity_root / focal / direction / speed / "left"
                    if not left_dir.is_dir():
                        skipped += 1
                        continue
                    for img in sorted(left_dir.iterdir()):
                        if img.suffix not in IMG_EXTENSIONS or not img.is_file():
                            continue
                        right = right_dir / img.name
                        disp = disp_dir / (img.stem + ".pfm")
                        if right.is_file() and disp.is_file():
                            pairs.append(DrivingPair(img, right, disp))
                        else:
                            skipped += 1
        if not pairs:
            raise FileNotFoundError(
                "no complete Driving triplets under " + str(self.root)
                + " (frames root {}, disparity root {}). Expected "
                "<focal>/<direction>/<speed>/{{left,right}}/<frame>.png plus "
                "<focal>/<direction>/<speed>/left/<frame>.pfm; "
                "the archive layout is UNVERIFIED -- check the real extraction.".format(
                    self.frames_root, self.disparity_root)
            )
        self.pairs = pairs
        self.skipped = skipped

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> StereoSample:
        pair = self.pairs[i]
        left = read_image(pair.left_image)
        right = read_image(pair.right_image)
        # PFM disparity is already in pixels -- do NOT divide by 256.
        # That divisor belongs to KITTI's 16-bit PNG convention only.
        disp = read_pfm(pair.disparity_pfm)
        if disp.ndim == 3:  # colour PFM: keep the first channel
            disp = disp[:, :, 0]
        disp = np.ascontiguousarray(disp, dtype=np.float32)
        return StereoSample(
            name=str(pair.left_image.relative_to(self.frames_root)),
            left=left,
            right=right,
            disparity=disp,
            original_shape=(left.shape[0], left.shape[1]),
        )
