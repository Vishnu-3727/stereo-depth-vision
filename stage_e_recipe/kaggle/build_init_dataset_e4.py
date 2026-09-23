#!/usr/bin/env python
"""Package a pulled E4 pretrain checkpoint as a private Kaggle dataset folder.

The E4 pretrain kernel writes `e4_pretrain_best.pth` (best-by-pretrain-val) to
/kaggle/working. After `push_e4.py pull-pretrain`, point this script at the
pulled file; it copies the checkpoint into a dataset folder, records its
sha256, and writes the folder's dataset-metadata.json. Folder builder ONLY:
it performs no upload (upload is `push_e4.py init-dataset`).

    python stage_e_recipe/kaggle/build_init_dataset_e4.py --checkpoint <path> [--out-dir <dir>]

The init dataset id is fixed: vishnuvardhanksece/stage-e-e4-init. The E4
finetune kernels (kernel_e4ft_seed*) mount this dataset and resolve the init
by file name, verifying the recorded sha before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

INIT_DATASET_SLUG = "vishnuvardhanksece/stage-e-e4-init"
INIT_FILE_NAME = "e4_pretrain_best.pth"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(checkpoint: Path, out_dir: Path) -> dict:
    if not checkpoint.is_file():
        raise SystemExit(f"BUILD FAIL: checkpoint not found: {checkpoint}")
    if checkpoint.name != INIT_FILE_NAME:
        raise SystemExit(
            f"BUILD FAIL: expected the pretrain's best-by-pretrain-val file "
            f"{INIT_FILE_NAME}, got {checkpoint.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / INIT_FILE_NAME
    # Copy even when checkpoint already lives inside out_dir (same file):
    # read bytes once so the recorded sha is of exactly what ships.
    data = checkpoint.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if not dst.exists() or sha256(dst) != digest:
        dst.write_bytes(data)
    record = {
        "dataset": INIT_DATASET_SLUG,
        "purpose": ("Stage E E4 finetune init: best-by-pretrain-val checkpoint "
                    "of the 8-epoch resume-chained pretrain"),
        "file": INIT_FILE_NAME,
        "sha256": digest,
        "bytes": len(data),
        "source": str(checkpoint),
    }
    (out_dir / "init_integrity.json").write_text(json.dumps(record, indent=2))
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True,
                    help="pulled pretrain file e4_pretrain_best.pth")
    ap.add_argument("--out-dir", default=str(HERE / "e4_init_dataset"),
                    help="dataset folder to write (default: e4_init_dataset/)")
    args = ap.parse_args()
    record = build(Path(args.checkpoint), Path(args.out_dir))
    print(f"file:   {record['file']}")
    print(f"sha256: {record['sha256']}")
    print(f"bytes:  {record['bytes']}")
    print(f"folder: {args.out_dir}")


if __name__ == "__main__":
    main()
