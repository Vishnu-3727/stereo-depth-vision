"""KITTI 2015 stereo, with Hailo's preprocessing reproduced exactly.

The Hailo Model Zoo evaluates this model on a specific 40-scene slice of the
KITTI 2015 *training* set, with a specific crop and a non-standard ground-truth
scale. Reproducing the published figure requires reproducing all of it, so the
protocol is spelled out here rather than approximated.

Reconstructed from the Model Zoo repository [SR-002]:

- ``datasets/create_kitti_stereo_tfrecord.py`` -- selects ``image_2`` /
  ``image_3`` and ``disp_occ_0`` ground truth, frames matching ``_10`` only,
  sorted; indices 0-159 become the quantisation calibration set and 160-199 the
  validation set.
- ``core/preprocessing/stereonet_preprocessing.py`` -- pads the bottom and
  right edges with zeros up to at least 368x1232, then crops to 368x1232
  anchored at the top-left. Ground truth is padded and cropped identically. No
  resizing anywhere.
- ``core/datasets/parse_kitti_stereo.py`` -- divides the raw 16-bit ground
  truth by **255.0**. KITTI's own convention is 256.0, so Hailo's ground-truth
  disparities are 256/255 = 1.0039x larger than the official values.

That last point is a real deviation, not a rounding detail, so both scales are
available here and the choice is always explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Hailo's input size, and the split boundary its tfrecord script uses.
TARGET_H, TARGET_W = 368, 1232
CALIB_SPLIT_END = 160

# ImageNet statistics on a 0-255 scale, from stereonet.yaml [SR-002].
NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
NORM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


@dataclass
class StereoSample:
    name: str
    left: np.ndarray       # H x W x 3, uint8
    right: np.ndarray      # H x W x 3, uint8
    disparity: np.ndarray  # H x W, float32; 0 means no ground truth
    original_shape: tuple[int, int]


def pad_and_crop(
    image: np.ndarray, target_h: int = TARGET_H, target_w: int = TARGET_W
) -> np.ndarray:
    """Hailo's ``pad_and_crop_tensor``: zero-pad bottom and right if the image
    is smaller than the target, then crop from the top-left corner.

    For KITTI's 375x1242 images no padding happens; 7 rows are removed from the
    bottom and 10 columns from the right. The bottom rows are where the road
    surface closest to the car appears -- the largest disparities in the scene
    -- so this crop is not neutral with respect to the disparity distribution.
    """
    h, w = image.shape[:2]
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h or pad_w:
        pads = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (image.ndim - 2)
        image = np.pad(image, pads, mode="constant", constant_values=0)
    return image[:target_h, :target_w]


def read_disparity_png(path: str | Path, scale: float = 256.0) -> np.ndarray:
    """Read a KITTI 16-bit disparity PNG.

    Stored values are ``round(disparity * scale)``; zero means no ground truth
    at that pixel and must never be treated as a disparity of zero. Pass
    ``scale=255.0`` to reproduce Hailo's convention.
    """
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(str(path))
    if raw.dtype != np.uint16:
        raise ValueError(
            "expected a 16-bit disparity PNG, got " + str(raw.dtype) + " for " + str(path)
        )
    return raw.astype(np.float32) / scale


def read_image(path: str | Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return bgr[:, :, ::-1].copy()  # to RGB


class Kitti2015Stereo:
    """The KITTI 2015 training set, sliced the way Hailo's evaluator slices it."""

    def __init__(
        self,
        root: str | Path,
        split: str = "hailo_val",
        disparity_scale: float = 256.0,
        occluded: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.disparity_scale = disparity_scale
        # disp_occ_0 includes occluded pixels (KITTI "all"); disp_noc_0 excludes
        # them. Hailo's tfrecord script uses disp_occ_0.
        self.gt_dir = self.root / "training" / ("disp_occ_0" if occluded else "disp_noc_0")

        left_dir = self.root / "training" / "image_2"
        if not left_dir.exists():
            raise FileNotFoundError(
                "KITTI 2015 not found at " + str(self.root)
                + ". Expected training/image_2 etc."
            )
        # Frame 10 is the reference frame; frame 11 has no disparity ground truth.
        names = sorted(p.name for p in left_dir.iterdir() if p.name.endswith("_10.png"))

        if split == "hailo_val":
            names = names[CALIB_SPLIT_END:]
        elif split == "hailo_calib":
            names = names[:CALIB_SPLIT_END]
        elif split == "all":
            pass
        else:
            raise ValueError("unknown split: " + split)
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> StereoSample:
        name = self.names[i]
        left = read_image(self.root / "training" / "image_2" / name)
        right = read_image(self.root / "training" / "image_3" / name)
        disp = read_disparity_png(self.gt_dir / name, scale=self.disparity_scale)
        return StereoSample(
            name=name,
            left=pad_and_crop(left),
            right=pad_and_crop(right),
            disparity=pad_and_crop(disp),
            original_shape=(left.shape[0], left.shape[1]),
        )

    def calibration_path(self, name: str) -> Path:
        return self.root / "training" / "calib_cam_to_cam" / (name[:6] + ".txt")


def normalize(image: np.ndarray) -> np.ndarray:
    """Apply the normalisation Hailo's compiler bakes into the network.

    The exported ONNX contains no normalisation -- Hailo inserts it at parse
    time via ``normalize_in_net: true`` [SR-002], and ``stereonet.alls`` places
    it as the first layer [SR-003]. Running the ONNX directly therefore means
    doing this first, on 0-255 RGB, and returning NCHW.
    """
    x = np.asarray(image, dtype=np.float32)
    x = (x - NORM_MEAN) / NORM_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def demo() -> None:
    """Self-check: the crop matches Hailo's rule and ground truth stays aligned."""
    # a 375x1242 image, like KITTI: cropped, never padded
    img = np.arange(375 * 1242 * 3, dtype=np.uint8).reshape(375, 1242, 3)
    out = pad_and_crop(img)
    assert out.shape == (368, 1232, 3), out.shape
    assert np.array_equal(out, img[:368, :1232]), "crop must be anchored top-left"

    # a smaller image is zero-padded on the bottom and right, not centred
    small = np.ones((100, 100, 3), dtype=np.uint8)
    out2 = pad_and_crop(small)
    assert out2.shape == (368, 1232, 3)
    assert out2[:100, :100].all(), "original content must stay at the top-left"
    assert not out2[100:].any() and not out2[:, 100:].any(), "padding must be zero"

    # disparity keeps its alignment with the image under the same operation
    disp = np.arange(375 * 1242, dtype=np.float32).reshape(375, 1242)
    assert np.array_equal(pad_and_crop(disp), disp[:368, :1232])

    # the two ground-truth scales differ by exactly 256/255
    raw = np.uint16(2560)
    assert abs(raw / 256.0 - 10.0) < 1e-9
    assert abs((raw / 255.0) / (raw / 256.0) - 256.0 / 255.0) < 1e-9

    # normalisation produces NCHW with the documented statistics
    n = normalize(np.full((368, 1232, 3), 123.675, dtype=np.float32))
    assert n.shape == (1, 3, 368, 1232), n.shape
    # channel 0 sits exactly on its mean, so it normalises to zero; the other
    # channels do not, which confirms the statistics are applied per channel
    assert abs(float(n[0, 0].mean())) < 1e-5, float(n[0, 0].mean())
    assert float(n[0, 1].mean()) > 0.1 and float(n[0, 2].mean()) > 0.3

    print("kitti2015 self-check passed")


if __name__ == "__main__":
    demo()
