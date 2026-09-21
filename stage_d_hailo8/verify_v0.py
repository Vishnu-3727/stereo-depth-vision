#!/usr/bin/env python
"""Stage D artifact-integrity gate (spec section 7).

ARM-P-H8-V0 is defined as the EXACT frozen ARM-P. This script proves the
Stage D copies are byte-identical to the frozen originals, and writes the
result to ``hashes/v0_integrity.json``.

A mismatch is a STOP condition: the copy is never "fixed" silently.

Read-only with respect to every frozen artefact.

The ``v0/`` binaries are deliberately NOT committed: they are byte-identical
duplicates of artefacts git already tracks. ``--create`` recreates them from
those tracked originals, so the branch is reproducible without storing the
same bytes twice.

    python stage_d_hailo8/verify_v0.py [--create]
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# The frozen originals and their spec-declared SHA-256 (spec section 7).
FROZEN = {
    "checkpoint": (
        "stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth",
        "b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454"),
    "onnx": (
        "stage_c_deploy/armp_stereonet.onnx",
        "4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989"),
    # The static export carries no spec-declared hash; it is recorded, not asserted.
    "onnx_static": (
        "stage_c_deploy/armp_stereonet_static.onnx", None),
}

COPIES = {
    "checkpoint": "stage_d_hailo8/v0/armp_h8_v0.pth",
    "onnx": "stage_d_hailo8/v0/armp_h8_v0.onnx",
    "onnx_static": "stage_d_hailo8/v0/armp_h8_v0_static.onnx",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    create = "--create" in sys.argv
    report: dict = {"gate": "ARM-P-H8-V0 == EXACT FROZEN ARM-P (spec section 7)",
                    "artifacts": {}}
    ok = True
    for key, (rel_frozen, declared) in FROZEN.items():
        f, c = REPO / rel_frozen, REPO / COPIES[key]
        if create and f.exists() and not c.exists():
            c.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, c)
            print(f"created {COPIES[key]} from {rel_frozen}")
        if not f.exists() or not c.exists():
            ok = False
            report["artifacts"][key] = {"status": "MISSING",
                                        "frozen_exists": f.exists(),
                                        "copy_exists": c.exists()}
            continue
        hf, hc = sha256(f), sha256(c)
        checks = {
            "frozen_path": rel_frozen, "copy_path": COPIES[key],
            "frozen_sha256": hf, "copy_sha256": hc,
            "copy_matches_frozen": hf == hc,
            "declared_sha256": declared,
            "frozen_matches_declared": None if declared is None else hf == declared,
        }
        good = checks["copy_matches_frozen"] and (
            declared is None or checks["frozen_matches_declared"])
        checks["status"] = "MATCH" if good else "MISMATCH"
        ok &= good
        report["artifacts"][key] = checks
    report["verdict"] = "PASS" if ok else "STOP - HASH MISMATCH"
    out = HERE / "hashes" / "v0_integrity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for k, v in report["artifacts"].items():
        print(f"{v['status']:<9} {k:<12} {v.get('copy_sha256', '')}")
    print(f"verdict: {report['verdict']}")
    print(f"wrote {out}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
