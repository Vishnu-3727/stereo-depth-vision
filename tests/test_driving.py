"""Self-check for the ARM D data layer: PFM round-trip + Driving errors.

Runs today with no Driving data on disk (CPU only, no GPU): the synthetic
PFM is written by the test itself, and the Driving assertions cover the
absent-data paths plus a synthetic mini-tree exercising enumeration.
"""

import numpy as np
import pytest

from src.datasets.driving import DrivingStereo
from src.datasets.pfm import read_pfm


def _write_pfm(path, data, little_endian=True):
    """Minimal test-local PFM writer (bottom-to-top rows, signed scale)."""
    data = np.asarray(data, dtype=np.float32)
    color = data.ndim == 3
    h, w = data.shape[:2]
    endian = "<" if little_endian else ">"
    with open(path, "wb") as fh:
        fh.write(b"PF\n" if color else b"Pf\n")
        fh.write("{} {}\n".format(w, h).encode("utf-8"))
        fh.write(("-1.0\n" if little_endian else "1.0\n").encode("utf-8"))
        fh.write(np.flipud(data).astype(endian + "f4").tobytes())


def test_pfm_roundtrip_shape_dtype_orientation_values(tmp_path):
    # non-square + asymmetric values: a flip/swap bug cannot hide
    data = np.arange(3 * 5, dtype=np.float32).reshape(3, 5) * 0.5 + 0.25
    for little_endian in (True, False):
        p = tmp_path / ("t_{}.pfm".format("le" if little_endian else "be"))
        _write_pfm(p, data, little_endian=little_endian)
        got = read_pfm(p)
        assert got.dtype == np.float32, got.dtype
        assert got.shape == (3, 5), got.shape
        assert np.array_equal(got, data), (little_endian, got)


def test_pfm_rejects_bad_magic(tmp_path):
    p = tmp_path / "bad.pfm"
    p.write_bytes(b"XX\n2 2\n-1.0\n" + b"\x00" * 16)
    with pytest.raises(ValueError):
        read_pfm(p)


def test_driving_raises_clear_error_when_absent(tmp_path):
    missing = tmp_path / "no_such_dir"
    with pytest.raises(FileNotFoundError, match="[Dd]riving"):
        DrivingStereo(missing)
    # existing but empty: still a clear error, not an empty silent dataset
    with pytest.raises(FileNotFoundError):
        DrivingStereo(tmp_path)


def test_driving_enumerates_synthetic_tree(tmp_path):
    cv2 = pytest.importorskip("cv2")
    frames = tmp_path / "driving_frames_cleanpass"
    disp = tmp_path / "driving_disparity"
    left_d = frames / "35mm_focallength" / "scene_forwards" / "fast" / "left"
    right_d = frames / "35mm_focallength" / "scene_forwards" / "fast" / "right"
    disp_d = disp / "35mm_focallength" / "scene_forwards" / "fast" / "left"
    for d in (left_d, right_d, disp_d):
        d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in ("0001.png", "0002.png"):
        img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        assert cv2.imwrite(str(left_d / name), img[:, :, ::-1])
        assert cv2.imwrite(str(right_d / name), img[:, :, ::-1])
        _write_pfm(disp_d / (name[:-4] + ".pfm"),
                   rng.random((48, 64), dtype=np.float32) * 100.0 + 1.0)

    ds = DrivingStereo(tmp_path)
    assert len(ds) == 2, len(ds)
    s = ds[0]
    assert s.left.shape == (48, 64, 3) and s.left.dtype == np.uint8
    assert s.right.shape == (48, 64, 3) and s.right.dtype == np.uint8
    assert s.disparity.shape == (48, 64) and s.disparity.dtype == np.float32
    # disparity raw in pixels: independent re-read of the PFM must match
    from pathlib import Path as _Path
    raw = read_pfm(disp_d / (_Path(s.name).stem + ".pfm"))
    assert np.array_equal(s.disparity, raw)
    assert (s.disparity > 0).all()
