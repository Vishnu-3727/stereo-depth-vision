"""Freeze EXP-CORRESPONDENCE-GEOM-001 before execution.

Writes:
  frozen.sha256      -- the protocol artifacts in this record
  historical.sha256  -- every prior record and source file this experiment
                        depends on, so HS8 ("no historical record modified")
                        can be verified before AND after execution.

Reads only. Creates nothing outside this record.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]

FROZEN = ["PREREGISTRATION.md", "spec.json", "source_trace.md", "run_geom.py",
          "freeze_geom.py"]

HISTORICAL = [
    # design authority
    "phase2/diagnostics/correspondence_geom/design_audit_20260912/DESIGN_AUDIT.md",
    "phase2/diagnostics/correspondence_geom/design_audit_20260912/design.json",
    "phase2/diagnostics/correspondence_geom/design_audit_20260912/RELATED_RUNS.md",
    "phase2/diagnostics/correspondence_geom/design_audit_20260912/source_trace.md",
    "phase2/diagnostics/correspondence_geom/design_audit_20260912/synthetic_mechanism_check.py",
    "phase2/diagnostics/correspondence_geom/design_audit_20260912/synthetic_mechanism_check.json",
    # prior correspondence records
    "phase2/diagnostics/correspondence/20260910T142622Z/RESULTS.md",
    "phase2/diagnostics/correspondence_index/20260911T015124Z/RESULTS.md",
    "phase2/diagnostics/correspondence_arch/20260911T025312Z/RESULTS.md",
    "phase2/diagnostics/correspondence_geom/20260911T111809Z/RESULTS.md",
    "phase2/diagnostics/correspondence_geom/20260911T115917Z/RESULTS.md",
    "phase2/diagnostics/correspondence_geom/20260911T115917Z/construction_spec.json",
    "phase2/diagnostics/correspondence_arch_rate/20260911T130507Z/results.json",
    "phase2/diagnostics/correspondence_arch_rate/20260911T130507Z/construction_spec.json",
    "phase2/diagnostics/correspondence_arch_rate/20260911T130507Z/randomisation_spec.json",
    "phase2/diagnostics/correspondence_arch_rate/20260911T130507Z/classification_spec.json",
    "phase2/diagnostics/correspondence_tr/20260911T134500Z/RESULTS.md",
    "phase2/diagnostics/correspondence_tr/20260911T134500Z/spec.json",
    "phase2/diagnostics/correspondence_tr/20260911T134500Z/results.json",
    "phase2/diagnostics/correspondence_geom/postmortem_20260911/POSTMORTEM.md",
    "phase2/diagnostics/correspondence_geom/postmortem_20260911/STATISTIC_AUDIT.md",
    "phase2/diagnostics/correspondence_geom/postmortem_20260911/SUCCESSOR_ASSESSMENT.md",
    "phase2/diagnostics/correspondence_geom/predesign_audit_20260911/ALGEBRA_AUDIT.md",
    "phase2/diagnostics/correspondence_geom/successor_audit_20260911/AUDIT.md",
    "phase2/diagnostics/correspondence_geom/successor_audit_20260911/RECOMMENDATION.md",
    "phase2/diagnostics/correspondence_geom/successor_audit_20260911/SUCCESSOR_OPTIONS.md",
    "phase2/diagnostics/correspondence_geom/gate_audit_20260911/GATE_AUDIT.md",
    # frozen source under test
    "src/models/stereonet/cost_volume.py",
    "src/models/stereonet/aggregation.py",
    "src/models/stereonet/feature_extractor.py",
    "src/models/stereonet/regression.py",
    "src/models/stereonet/stereonet.py",
    "src/models/stereonet/blocks.py",
    "src/models/stereonet/refinement.py",
    "phase2/models/scaled_regression.py",
    "src/datasets/kitti2015.py",
]


def sha256_bytes(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    for name in ("gate.json", "trained.json", "results.json", "RESULTS.md"):
        if (OUT / name).exists():
            print("REFUSING TO FREEZE: %s already exists" % name)
            return 1
    lines = []
    for name in FROZEN:
        p = OUT / name
        if not p.exists():
            print("MISSING protocol artifact: %s" % name)
            return 1
        lines.append("%s  %s" % (sha256_bytes(p), name))
    (OUT / "frozen.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hist = []
    for name in HISTORICAL:
        p = REPO_ROOT / name
        if not p.exists():
            print("MISSING historical record: %s" % name)
            return 1
        hist.append("%s  %s" % (sha256_bytes(p), name))
    (OUT / "historical.sha256").write_text("\n".join(hist) + "\n", encoding="utf-8")

    print("frozen.sha256      : %d protocol artifacts" % len(lines))
    print("historical.sha256  : %d historical records" % len(hist))
    for line in lines:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
