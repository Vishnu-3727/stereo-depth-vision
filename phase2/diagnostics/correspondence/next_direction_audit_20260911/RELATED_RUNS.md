# RELATED RUNS — next_direction_audit_20260911

Design/audit record only. Nothing here was executed; nothing here modifies any
record listed below.

## Correspondence diagnostics audited

| record | verdict (unchanged) | role in this audit |
|---|---|---|
| `phase2/diagnostics/correspondence/20260910T142622Z/` (EXP-CORRESPONDENCE-SHIFT-001, Stage A) | DIAGNOSTIC-INVALID-AS-PREREGISTERED | source of degenerate empirical null (β_H=−0.0017, V-shape, V≥H); slope derivation −1/16 reused |
| `phase2/diagnostics/correspondence_index/20260910T164346Z/` (INDEX-001 RUN-A) | ESTABLISHED → EXPLORATORY-ONLY (provenance audit) | ordered 12/12 reproduction; control unverifiable |
| `phase2/diagnostics/correspondence_index/20260910T164326Z/` (INDEX-001 RUN-B) | NOT-DEMONSTRATED → EXPLORATORY-ONLY | ordered 12/12 reproduction; single-draw control indistinguishable |
| `phase2/diagnostics/correspondence_index/audit_20260911T011113Z/` (provenance audit) | NO-CONFIRMATORY-EVIDENCE | basis for INCONCLUSIVE Level-C status; mask-occupancy finding (99.4% in d∈[2,4]) |
| `phase2/diagnostics/correspondence_index/20260911T015124Z/` (INDEX-002) | CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED | magnitude-null failure; structural-mismatch analysis reused |
| `phase2/diagnostics/correspondence_index/design_audit_20260911/` (INDEX-003 design audit) | DESIGN-INCONCLUSIVE | generator-null construction; equivariance argument; publicity-of-ordered-values constraint |
| `phase2/diagnostics/correspondence_arch/design_20260911/` (ARCH-001 design) | CLEAN-DESIGN-AVAILABLE (not authorised) | orbit protocol, 16-seed control, TV rejection |
| `phase2/diagnostics/correspondence_arch/20260911T025312Z/` (ARCH-001 execution) | CASE-A-ARCHITECTURE-INDUCED (near-zero power, recorded) | full-orbit data; post-hoc amplitude gap (NOT evidence); z-score power failure |

## Phase-1 evidence

| record | finding reused |
|---|---|
| `experiments/EXP-007/` | right-image dependence (up to +89.96 D1 points); right=left anomaly (mean 82.8 px vs expected ~0) |
| `experiments/EXP-010/` | degenerate cost volume (max inter-slice diff 0.0); construction mechanism |

## Source audited (read-only)

- `src/models/stereonet/cost_volume.py` (shift_left, reference_shift, build_cost_volume)
- `src/models/stereonet/aggregation.py` (5 candidate-axis 3-tap convolutions)
- `src/models/stereonet/regression.py` (soft_argmin, upsample-first ordering)
- `src/models/stereonet/stereonet.py` (graph, stride 16, candidate units)
- `src/models/stereonet/feature_extractor.py`, `blocks.py` (siamese tower, slopes)
- `phase2/models/scaled_regression.py` (parameter-free standardisation)
- `phase2/viz/core.py`, `render.py` (existing evaluator to reuse)
- `phase2/diagnostics/correspondence/exp_correspondence_shift.py` (harness pattern)

## Checkpoints referenced (not loaded)

- NEG `shift=none+standardised` `d40d805b…`, POS 6b seeds `581d62e6…`, `58117f26…`, `245a3f5b…` (INDEX-001/002 hashes).

## Lineage statement

No run here merges, re-scores, or re-interprets any listed record. All verdicts
stand as written. The only quantities carried forward as established are those
with threshold-free, twice-reproduced support: degenerate permutation
invariance (0.0), degenerate translation artifact shape (Stage A), and the
ordered re-indexing response as an exploratory observation.
