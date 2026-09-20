"""Portable Float Map (PFM) reader for SceneFlow disparity ground truth.

SceneFlow ships disparity as ``.pfm`` files (one per left frame); the repo
previously had no reader for them. Logic follows the upstream reference
implementation [SR-upstream]
``reference/upstream/extracted/StereoNet-master/dataloader/readpfm.py``
(READ ONLY -- not imported, reimplemented here so this repo stands alone):

- magic ``Pf`` (greyscale) or ``PF`` (colour);
- ``<width> <height>`` line;
- scale line whose SIGN encodes endianness (negative = little-endian);
- rows stored bottom-to-top, so the data must be flipped vertically.

Disparity values in these files are already in PIXELS. They must NOT be
divided by 256 -- that divisor is KITTI-only (see ``kitti2015.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_pfm(path: str | Path) -> np.ndarray:
    """Read a PFM file into a float32 array.

    Greyscale files return H x W; colour files return H x W x 3. The image is
    returned top-row-first (the on-disk bottom-to-top order is flipped back).
    """
    with open(path, "rb") as fh:
        magic = fh.readline().rstrip()
        if magic == b"Pf":
            color = False
        elif magic == b"PF":
            color = True
        else:
            raise ValueError(
                "not a PFM file: bad magic " + repr(magic) + " in " + str(path)
            )

        dims = fh.readline().decode("utf-8").strip().split()
        if len(dims) != 2:
            raise ValueError("malformed PFM header in " + str(path))
        width, height = int(dims[0]), int(dims[1])

        scale = float(fh.readline().rstrip())
        if scale == 0.0:
            raise ValueError("PFM scale must be nonzero in " + str(path))
        endian = "<" if scale < 0 else ">"  # sign encodes endianness
        scale = abs(scale)

        data = np.fromfile(fh, endian + "f4")
        expected = height * width * (3 if color else 1)
        if data.size != expected:
            raise ValueError(
                "PFM payload size mismatch in {}: header says {}x{} ({}) "
                "but file holds {} floats".format(path, width, height, expected, data.size)
            )
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # on-disk rows run bottom-to-top
        return np.ascontiguousarray(data * scale, dtype=np.float32)


def demo() -> None:
    """Self-check: header parsing, endianness, orientation. (Payload checks
    live in tests/test_driving.py, which writes its own synthetic PFM.)"""
    import io

    # little-endian greyscale 2x1: scale line is negative
    buf = io.BytesIO(b"Pf\n2 1\n-1.0\n" + np.array([1.0, 2.0], dtype="<f4").tobytes())
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pfm", delete=False) as tmp:
        tmp.write(buf.getvalue())
        name = tmp.name
    got = read_pfm(name)
    assert got.shape == (1, 2), got.shape
    assert got.dtype == np.float32
    assert np.array_equal(got, [[1.0, 2.0]]), got

    print("pfm self-check passed")


if __name__ == "__main__":
    demo()
