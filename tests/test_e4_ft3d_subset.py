"""E4 data plumbing: subset manifest builder + sign-fixed loader."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "stage_e_recipe" / "e4_pretrain" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import enumerate_subset_triplets as builder
from ft3d_subset import FlyingThings3DSubset

PROBE_DIR = Path(
    "C:/Users/vishn/AppData/Local/Temp/claude/C--WINDOWS-system32/"
    "a0388a57-0d0f-405d-b95b-a731f9c83d98/scratchpad/ft3d_probe"
)

H, W = 6, 8


def _write_pfm(path: Path, data: np.ndarray) -> None:
    """Minimal greyscale PFM writer (little-endian); mirrors src/datasets/pfm.py."""
    assert data.ndim == 2
    h, w = data.shape
    with open(path, "wb") as fh:
        fh.write(b"Pf\n%d %d\n-1.0\n" % (w, h))
        fh.write(np.flipud(np.ascontiguousarray(data, dtype="<f4")).tobytes())


def _write_pfm_color(path: Path, data: np.ndarray) -> None:
    assert data.ndim == 3 and data.shape[2] == 3
    h, w, _ = data.shape
    with open(path, "wb") as fh:
        fh.write(b"PF\n%d %d\n-1.0\n" % (w, h))
        fh.write(np.flipud(np.ascontiguousarray(data, dtype="<f4")).tobytes())


def _write_png(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (H, W, 3), dtype=np.uint8))


def _make_corpus(root: Path, n_complete=30, neg=True, color_idx=None):
    """Fake FlyingThings3D_subset layout; returns (images_root, disp_root, ids)."""
    left_d = root / "img" / "FlyingThings3D_subset" / "train" / "image_clean" / "left"
    right_d = root / "img" / "FlyingThings3D_subset" / "train" / "image_clean" / "right"
    disp_d = root / "dsp" / "FlyingThings3D_subset" / "train" / "disparity" / "left"
    for d in (left_d, right_d, disp_d):
        d.mkdir(parents=True, exist_ok=True)
    ids = ["%07d" % i for i in range(n_complete)]
    base = np.linspace(1.0, 10.0, H * W, dtype=np.float64).reshape(H, W)
    for k, idx in enumerate(ids):
        _write_png(left_d / (idx + ".png"), seed=k)
        _write_png(right_d / (idx + ".png"), seed=1000 + k)
        vals = -(base + k) if neg else (base + k)
        if color_idx is not None and k == color_idx:
            _write_pfm_color(disp_d / (idx + ".pfm"), np.stack([vals] * 3, axis=-1).astype(np.float32))
        else:
            _write_pfm(disp_d / (idx + ".pfm"), vals.astype(np.float32))
    # Incomplete triplets: missing right, and missing disp.
    _write_png(left_d / "9999998.png", seed=7)
    _write_pfm(disp_d / "9999998.pfm", -np.ones((H, W), dtype=np.float32))
    _write_png(left_d / "9999999.png", seed=8)
    _write_png(right_d / "9999999.png", seed=9)
    return root / "img", root / "dsp", ids


def _build(root: Path, **kw):
    img_root, dsp_root, ids = _make_corpus(root, **kw)
    out = root / "manifest.json"
    manifest = builder.build_manifest(img_root, dsp_root, out)
    return manifest, out, ids


def test_builder_enumerates_only_complete_and_counts_incomplete(tmp_path):
    manifest, _, ids = _build(tmp_path)
    assert manifest["n_triplets"] == len(ids)
    assert manifest["n_incomplete"] == 2  # one missing right, one missing disp
    got = {t["id"] for t in manifest["triplets"]}
    assert got == set(ids)
    assert "9999998" not in got and "9999999" not in got
    for t in manifest["triplets"]:
        assert set(t) == {"id", "left", "right", "disp"}
        for key in ("left", "right", "disp"):
            assert Path(t[key]).is_absolute() and Path(t[key]).is_file()


def test_holdout_deterministic_seed0_and_disjoint(tmp_path):
    m1, _, ids = _build(tmp_path)
    m2, _, _ = _build(tmp_path / "again")
    assert m1["rng_seed"] == 0 and m1["holdout_frac"] == 0.05
    assert m1["holdout_ids"] == m2["holdout_ids"]  # deterministic under seed 0
    assert m1["train_ids"] == m2["train_ids"]
    assert set(m1["holdout_ids"]).isdisjoint(m1["train_ids"])
    assert sorted(m1["holdout_ids"] + m1["train_ids"]) == sorted(ids)
    assert m1["n_holdout"] == len(m1["holdout_ids"]) > 0
    assert m1["n_train"] + m1["n_holdout"] == m1["n_triplets"]


def test_loader_returns_abs_of_negative_input(tmp_path):
    _, manifest_path, _ = _build(tmp_path)
    ds = FlyingThings3DSubset(manifest_path, split="train")
    assert len(ds) > 0
    sample = ds[0]
    d = np.asarray(sample.disparity)
    assert d.shape == (H, W)
    assert np.all(np.isfinite(d)) and bool((d > 0).all())


def test_loader_raises_on_positive_input(tmp_path):
    _, manifest_path, _ = _build(tmp_path, n_complete=4, neg=False)
    ds = FlyingThings3DSubset(manifest_path, split="train")
    with pytest.raises(ValueError, match="positive-signed"):
        ds[0]


def test_loader_raises_on_colour_pfm(tmp_path):
    _, manifest_path, _ = _build(tmp_path, n_complete=4, neg=True, color_idx=0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rec = next(t for t in manifest["triplets"] if t["id"] == "0000000")
    manifest["triplets"] = [rec]
    manifest["train_ids"] = ["0000000"]
    manifest["holdout_ids"] = []
    manifest["n_triplets"] = manifest["n_train"] = 1
    manifest["n_holdout"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ds = FlyingThings3DSubset(manifest_path, split="train")
    with pytest.raises(ValueError, match="colour PFM"):
        ds[0]


def test_loader_real_sample_0002000(tmp_path):
    left = PROBE_DIR / "left" / "0002000.png"
    right = PROBE_DIR / "right" / "0002000.png"
    disp = PROBE_DIR / "0002000.pfm"
    if not (left.is_file() and right.is_file() and disp.is_file()):
        pytest.skip("ft3d_probe files absent")
    manifest = {
        "description": "adhoc single-triplet manifest for the real-data check",
        "data_root": {"images_root": str(PROBE_DIR), "disparity_root": str(PROBE_DIR)},
        "rng_seed": 0,
        "holdout_frac": 0.05,
        "n_triplets": 1,
        "n_train": 1,
        "n_holdout": 0,
        "holdout_ids": [],
        "train_ids": ["0002000"],
        "triplets": [{"id": "0002000", "left": str(left), "right": str(right), "disp": str(disp)}],
    }
    mp = tmp_path / "real_manifest.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    ds = FlyingThings3DSubset(mp, split="train")
    sample = ds[0]
    d = np.asarray(sample.disparity, dtype=np.float64)
    assert d.shape == (540, 960)
    assert bool(np.isfinite(d).all()) and bool((d > 0).all())
    assert 80.0 <= float(d.max()) <= 90.0
