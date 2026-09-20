"""ARM-P D3 transfer diagnostic wrapper. Frozen inference only.

Does NOT modify stage_a_diagnostics/scripts/d3_matching.py. It imports that
module, monkey-patches ONLY d3.RUNS so the hardcoded checkpoint expression
resolves to the ARM-P checkpoint copy, then calls d3.process_seed(0, ...)
and d3.summarize_seed(0, ...) UNCHANGED.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
assert (REPO / "stage_a_diagnostics" / "scripts" / "d3_matching.py").exists(), REPO
D3DIR = REPO / "stage_a_diagnostics" / "scripts"
sys.path.insert(0, str(D3DIR))
sys.path.insert(0, str(REPO))

import numpy as np
import torch

import d3_matching as d3

OUTDIR_NAME = "20260918T105050Z_d3_transfer"
OUTDIR = REPO / "stage_b_armp" / OUTDIR_NAME
COPY = OUTDIR / "ckpt" / "p2a_best.pth"
SRC = REPO / "stage_b_armp" / "20260918T062146Z_stage1_pretrain" / "checkpoints" / "armp_stage1_best.pth"
EXPECTED_SHA = "3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7"

from src.models.stereonet import StereoNet, StereoNetConfig


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main() -> None:
    t0 = time.time()
    # ---- sha256 assertions ----
    src_sha = sha256_file(SRC)
    copy_sha = sha256_file(COPY)
    assert src_sha.lower() == EXPECTED_SHA.lower(), f"source sha mismatch: {src_sha}"
    assert copy_sha.lower() == EXPECTED_SHA.lower(), f"copy sha mismatch: {copy_sha}"
    assert src_sha == copy_sha, "source/copy sha differ"

    # ---- monkey-patch ONLY RUNS ----
    d3.RUNS = {0: "../../stage_b_armp/" + OUTDIR_NAME + "/ckpt"}
    resolved = (d3.REPO / "phase2" / "runs" / d3.RUNS[0] / "p2a_best.pth").resolve()
    assert resolved == COPY.resolve(), f"RUNS patch resolves wrong: {resolved} vs {COPY.resolve()}"
    assert resolved.exists(), f"checkpoint copy missing: {resolved}"

    # ---- config assertions (byte-identical conventions otherwise) ----
    assert d3.CFG == dict(downsample_levels=3, num_disparities=24,
                          cost_volume_shift="right", regression_normalize=True), \
        f"CFG drift: {d3.CFG}"
    cfg = StereoNetConfig(**d3.CFG)
    assert cfg.feature_normalize is False, "feature_normalize must be FALSE"
    probe = StereoNet(cfg)
    n_params = sum(p.numel() for p in probe.parameters())
    assert n_params == 397954, f"param count {n_params} != 397954"
    blob = torch.load(str(COPY), map_location="cpu", weights_only=False)
    assert isinstance(blob, dict) and "model" in blob, "checkpoint missing ['model'] key"
    probe.load_state_dict(blob["model"], strict=True)  # strict load asserted
    del probe, blob

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ARM-P D3 wrapper: device={device} ckpt={COPY}", flush=True)
    print(f"d3 module: {d3.__file__}", flush=True)

    acc = d3.process_seed(0, device, None)
    summary = d3.summarize_seed(0, acc)

    wall_s = time.time() - t0

    # ---- raw per-cell archive (same layout as d3 main) ----
    gt_all = np.concatenate(acc.gt_px)
    xx_all = np.concatenate(acc.cell_x)
    to_save: dict = {"gt_px": gt_all, "cell_x": xx_all}
    for r in d3.REPS:
        to_save[f"rank_{r}"] = np.concatenate(acc.rank[r])
        to_save[f"margin_{r}"] = np.concatenate(acc.margin[r])
        to_save[f"pos_{r}"] = np.concatenate(acc.pos[r])
        to_save[f"negmean_{r}"] = np.concatenate(acc.negmean[r])
        to_save[f"gap_{r}"] = np.concatenate(acc.gap[r])
        nv = np.concatenate(acc.neg_vals[r]) if acc.neg_vals[r] else np.zeros(0)
        nb = (np.concatenate(acc.neg_bins[r]) if acc.neg_bins[r]
              else np.zeros(0, dtype=np.int64))
        to_save[f"negvals_{r}"] = nv.astype(np.float32)
        to_save[f"negbins_{r}"] = nb.astype(np.int64)
    atomic_write_npz(OUTDIR / "raw" / "armp_d3_s0.npz", **to_save)

    out = {
        "task": "ARM-P D3 TRANSFER DIAGNOSTIC (frozen inference only, no training)",
        "checkpoint": {
            "source": str(SRC.relative_to(REPO)),
            "copy": str(COPY.relative_to(REPO)),
            "sha256_source": src_sha,
            "sha256_copy": copy_sha,
            "sha256_expected": EXPECTED_SHA,
            "sha_match": True,
        },
        "d3_conventions": "byte-identical to stage_a_diagnostics/scripts/d3_matching.py "
                          "via process_seed/summarize_seed UNCHANGED; only d3.RUNS patched. "
                          "Group-averaged representation D (G-invariant) PRESERVED as known limitation.",
        "model_config_asserted": dict(d3.CFG),
        "model_param_count": 397954,
        "strict_load": True,
        "feature_normalize": False,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "wall_clock_s": wall_s,
        "seed0_summary": summary,
    }
    atomic_write_json(OUTDIR / "armp_d3_matching.json", out)
    print(f"wrote armp_d3_matching.json + raw/armp_d3_s0.npz in {wall_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
