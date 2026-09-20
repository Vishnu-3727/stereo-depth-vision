# EXP-CORRESPONDENCE-INDEX-001 — RESULTS

## Gate

Negative control (`shift=none+standardized`): all 4 scenes × 4 non-trivial
permutations → `max|V_perm - V| == 0.0` (16/16) and
`max|d_init_perm - d_init_id| == 0.0` (16/16, full-frame). Identity via the
same `index_select` path reproduces the original volume exactly.
GATE PASSED → positives run.

## Checkpoint-level interior-masked medians (GT/16 in [2,8])

Mean of per-scene medians; Δ = vs identity. Full per-scene table in results.json.

| checkpoint | -1 | identity | +1 | +2 | random |
|---|---|---|---|---|---|
| NEG_shift_none | 6.536 (+0.000) | 6.536 | 6.536 (+0.000) | 6.536 (+0.000) | 6.536 (+0.000) |
| POS_6b_seed0 | 3.757 (-1.780) | 5.536 | 7.470 (+1.786) | 8.357 (+2.965) | 4.780 (-0.810) |
| POS_6b_seed1 | 2.923 (-1.203) | 4.145 | 5.270 (+1.093) | 6.066 (+1.934) | 3.935 (-0.360) |
| POS_6b_seed2 | 3.203 (-1.465) | 4.695 | 6.141 (+1.409) | 7.179 (+2.554) | 5.741 (+0.560) |

Per-scene ordering `d(+2) > d(+1) > d(0) > d(-1)` holds in 12/12
(checkpoint, scene) cases for the median statistic. Random permutation never
reproduces the ordered directional pattern (offsets -0.81/-0.36/+0.56 with
large unstructured full-frame changes, max|Δ| 5.4–7.0 vs identity).

## Verdict

Candidate-coordinate sensitivity (C) established for all three positive
checkpoints. Negative control algebraically invariant. No correspondence (D)
or search (E) claimed. Reference-frame discrepancy recorded, not corrected.

## Files

PREREGISTRATION.md, run_index.py, summarize.py, results.json (80 rows),
run.log, ENVIRONMENT.txt, this file. No historical records modified.
Compute: 16 volume builds reused across 80 arms; ~7.7 s wall clock (cuda).
No training. Protocol deviations: none.
