#!/usr/bin/env python
"""Assemble the Stage-E E4 Kaggle bundle and record its integrity.

Copies the research source E4 pretraining needs into `bundle_e4/`, hashes
every file, and asserts each copy is byte-identical to its repository
original. E4 needs NO patched copy (unlike E1/E2/E3): every file is
byte-identical or the build fails.

Not a training run and not a Kaggle upload: this only writes
`stage_e_recipe/kaggle/bundle_e4/` and `source_integrity_e4.json`.

    python stage_e_recipe/kaggle/build_bundle_e4.py

E4 differs from E0-E3 in ways that shape this bundle:
- it trains FROM SCRATCH, so no init checkpoint is bundled;
- it touches NO KITTI data, so no KITTI archives are referenced;
- its corpus arrives as TWO public Kaggle datasets (images + disparity),
  mounted read-only under /kaggle/input/ and resolved by the probe kernel;
- its scripts keep repo-relative layout inside the bundle
  (stage_e_recipe/e4_pretrain/scripts/...) so train_e4_pretrain.py's
  parents[3] repo-root resolution keeps working on Kaggle.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = HERE / "bundle_e4"

# bundle-relative path -> repo-relative source. Bundle-relative EQUALS
# repo-relative throughout, so the trainer's REPO_ROOT = parents[3] still
# resolves to the assembled root on Kaggle.
FILES = {
    # StereoNet model package (all modules the package __init__ re-exports;
    # onnx_weights.py is NOT imported by any of them, so it is excluded).
    "src/__init__.py": "src/__init__.py",
    "src/models/__init__.py": "src/models/__init__.py",
    "src/models/stereonet/__init__.py": "src/models/stereonet/__init__.py",
    "src/models/stereonet/aggregation.py": "src/models/stereonet/aggregation.py",
    "src/models/stereonet/blocks.py": "src/models/stereonet/blocks.py",
    "src/models/stereonet/cost_volume.py": "src/models/stereonet/cost_volume.py",
    "src/models/stereonet/excitation.py": "src/models/stereonet/excitation.py",
    "src/models/stereonet/feature_extractor.py": "src/models/stereonet/feature_extractor.py",
    "src/models/stereonet/refinement.py": "src/models/stereonet/refinement.py",
    "src/models/stereonet/regression.py": "src/models/stereonet/regression.py",
    "src/models/stereonet/stereonet.py": "src/models/stereonet/stereonet.py",
    # loss (train_e4_pretrain imports masked_smooth_l1).
    "src/losses/__init__.py": "src/losses/__init__.py",
    "src/losses/disparity.py": "src/losses/disparity.py",
    # metrics (train_e4_pretrain imports disparity_metrics for the monitor).
    "src/evaluation/__init__.py": "src/evaluation/__init__.py",
    "src/evaluation/metrics.py": "src/evaluation/metrics.py",
    # datasets: pfm.py (read_pfm, used by ft3d_subset) and the WHOLE
    # kitti2015.py (train_e4_pretrain imports normalize from it;
    # ft3d_subset imports StereoSample from it; the file has no other
    # dependencies, so the whole module ships byte-identical).
    "src/datasets/__init__.py": "src/datasets/__init__.py",
    "src/datasets/kitti2015.py": "src/datasets/kitti2015.py",
    "src/datasets/pfm.py": "src/datasets/pfm.py",
    # determinism helper from phase1/harness/ (seed_all, make_generator,
    # worker_init_fn). phase1/ has no __init__.py files (namespace packages).
    "phase1/harness/determinism.py": "phase1/harness/determinism.py",
    # the three E4 pretrain scripts, kept at repo-relative paths.
    "stage_e_recipe/e4_pretrain/scripts/enumerate_subset_triplets.py":
        "stage_e_recipe/e4_pretrain/scripts/enumerate_subset_triplets.py",
    "stage_e_recipe/e4_pretrain/scripts/ft3d_subset.py":
        "stage_e_recipe/e4_pretrain/scripts/ft3d_subset.py",
    "stage_e_recipe/e4_pretrain/scripts/train_e4_pretrain.py":
        "stage_e_recipe/e4_pretrain/scripts/train_e4_pretrain.py",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    record: dict = {
        "bundle": "stage_e_recipe/kaggle/bundle_e4",
        "purpose": "Stage E E4 pretrain Kaggle bundle (from-scratch, subset corpus)",
        "byte_identical_files": {},
        "declared_deltas": {},
        "verdict": None,
    }
    ok = True

    for rel, src_rel in FILES.items():
        src = REPO / src_rel
        if not src.exists():
            raise SystemExit(f"BUILD FAIL: missing source {src_rel}")
        dst = BUNDLE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        h_src, h_dst = sha256(src), sha256(dst)
        same = h_src == h_dst
        ok &= same
        record["byte_identical_files"][rel] = {
            "repo_source": src_rel, "sha256": h_dst, "identical": same}

    (BUNDLE / "BUNDLE_MARKER.json").write_text(json.dumps({
        "bundle": "stage-e-e4",
        "campaign": "Stage E",
        "experiment": "e4",
        "initialization": "random (from scratch; no init checkpoint)",
        "training_authorized": False,
    }, indent=2))

    record["file_count"] = sum(1 for _ in BUNDLE.rglob("*") if _.is_file())
    record["total_bytes"] = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file())
    record["verdict"] = "PASS" if ok else "FAIL"
    manifest = json.dumps(record, indent=2)
    (HERE / "source_integrity_e4.json").write_text(manifest)
    # The kernel re-verifies every hash on Kaggle, so the manifest ships
    # inside the bundle too.
    (BUNDLE / "source_integrity.json").write_text(manifest)

    n_ident = sum(1 for v in record["byte_identical_files"].values() if v["identical"])
    print(f"byte-identical files: {n_ident}/{len(record['byte_identical_files'])}")
    print(f"declared deltas:      {len(record['declared_deltas'])}")
    print(f"bundle:               {record['file_count']} files, "
          f"{record['total_bytes'] / 2**20:.2f} MiB")
    print(f"verdict:              {record['verdict']}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
