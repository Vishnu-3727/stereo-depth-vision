"""Enumerate FT3D TRAIN triplets (left png + right png + left pfm) on C:.

Writes dataset_manifest.json atomically (.tmp then rename) into the
Stage-1 output dir. Read-only probe: no training, no copies.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "sceneflow" / "flyingthings3d" / "extracted"
SEED = 0
HOLDOUT_FRAC = 0.05

import numpy as np


def main() -> None:
    triplets = []
    for subset in ("A", "C"):
        left_root = DATA_ROOT / "frames_cleanpass" / "TRAIN" / subset
        if not left_root.exists():
            continue
        for seq_dir in sorted(left_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            ldir, rdir = seq_dir / "left", seq_dir / "right"
            ddir = DATA_ROOT / "disparity" / "TRAIN" / subset / seq_dir.name / "left"
            if not (ldir.is_dir() and rdir.is_dir() and ddir.is_dir()):
                continue
            for lp in sorted(ldir.glob("*.png")):
                rp = rdir / lp.name
                dp = ddir / (lp.stem + ".pfm")
                if rp.is_file() and rp.stat().st_size > 0 and dp.is_file() and dp.stat().st_size > 0 and lp.stat().st_size > 0:
                    triplets.append({
                        "id": subset + "/" + seq_dir.name + "/" + lp.stem,
                        "left": str(lp.relative_to(REPO_ROOT)),
                        "right": str(rp.relative_to(REPO_ROOT)),
                        "disp": str(dp.relative_to(REPO_ROOT)),
                    })
    triplets.sort(key=lambda t: t["id"])
    n = len(triplets)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_hold = int(round(n * HOLDOUT_FRAC))
    hold_idx = sorted(perm[:n_hold].tolist())
    train_idx = sorted(perm[n_hold:].tolist())
    manifest = {
        "description": "ARM P Stage 1 (FT3D A+C subset, NOT full SceneFlow): triplet enumeration + fixed 5% pretrain-validation holdout. Pretrain monitoring ONLY; unrelated to the frozen KITTI contract.",
        "data_root": str(DATA_ROOT.relative_to(REPO_ROOT)),
        "rng_seed": SEED,
        "holdout_frac": HOLDOUT_FRAC,
        "n_triplets": n,
        "n_train": len(train_idx),
        "n_holdout": len(hold_idx),
        "holdout_ids": [triplets[i]["id"] for i in hold_idx],
        "train_ids": [triplets[i]["id"] for i in train_idx],
        "triplets": triplets,
    }
    tmp = OUT_DIR / "dataset_manifest.json.tmp"
    final = OUT_DIR / "dataset_manifest.json"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, final)
    by_sub = {}
    for t in triplets:
        by_sub[t["id"].split("/")[0]] = by_sub.get(t["id"].split("/")[0], 0) + 1
    print("triplets=%d by_subset=%s train=%d holdout=%d" % (n, by_sub, len(train_idx), len(hold_idx)), flush=True)


if __name__ == "__main__":
    main()
