# EXP-CORRESPONDENCE-INDEX-001 — PREREGISTRATION

Inference-only. Frozen before execution (2026-09-10T16:43:46Z).

## Research question

Does the frozen aggregation/readout path track candidate-slice coordinate identity (level C only)?

## Tensor location

`volume = cost_volume(left_features, right_features)`, shape `(B,32,12,23,77)`,
candidate axis `dim=2`. Built once per checkpoint/scene. All arms derived from
the SAME tensor via `index_select(dim=2)`. No image/feature/volume rebuild.
Primary endpoint `disparity_initial` only (aggregation → standardized regression).
No refinement in primary measurement.

## Permutations (frozen)

`V'(k) = V(π(k))`, `k ∈ 0..11`.

- identity: `[0,1,2,3,4,5,6,7,8,9,10,11]`
- plus1 (`π(k)=(k-1) mod 12`): `[11,0,1,2,3,4,5,6,7,8,9,10]`
- plus2 (`π(k)=(k-2) mod 12`): `[10,11,0,1,2,3,4,5,6,7,8,9]`
- minus1 (`π(k)=(k+1) mod 12`): `[1,2,3,4,5,6,7,8,9,10,11,0]`
- random (fixed, single draw, reused for all scenes/checkpoints): `[7,2,10,0,5,11,1,8,4,9,6,3]`

Content at `j` moves to `j+m mod 12` for ordered shift `m`. Positive `m` moves
content toward larger candidate indices. Same `index_select` mechanism for ALL
arms including identity.

## Checkpoints (read-only)

- NEG `shift=none+standardized`: `phase2/factorial/shift_none_standardized/20260909T071500Z/checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A_checkpoint.pth`
- POS0 seed0: `phase2/diagnostics/determinism/20260909T041500Z_baseline/checkpoints/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA_checkpoint.pth`
- POS1 seed1: `phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth`
- POS2 seed2: `phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth`

Scenes (minimum focus set, `hailo_val`): `[27, 0, 31, 6]`.

## Interior mask (frozen, GT-derived)

Retain pixels with valid GT and `GT/16 ∈ [2,8]`. Same mask for every
checkpoint/permutation. No border mask (images never translated). No post-hoc
mask changes. No fill values.

## Predictions (frozen)

- NEG gate: `max|V_id - V_perm| == 0` and `max|d_id - d_perm| == 0` for every
  permutation under deterministic protocol. Any violation → STOP, INVALID,
  no positives.
- POS: no unit-slope demand. Qualitative: interior-masked `d(+2) > d(+1) > d(0) > d(-1)`
  (median/aggregate sense); +2 generally larger than +1; random = substantial but
  unstructured (no directional prediction, no threshold).
- Random ≈ ordered in magnitude+structure → do NOT claim C.

## Claim ceiling (frozen)

Success claims at most candidate-coordinate sensitivity (C). Never
correspondence / matching / disparity search (D/E). Reference-frame discrepancy
recorded, not corrected.

## Determinism

`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cudnn.deterministic=True`,
`cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`.
`torch.no_grad()`, eval mode. fp32.

## Outputs

`PREREGISTRATION.md` (this file), `results.json`, `RESULTS.md`,
`ENVIRONMENT.txt`, execution log, script snapshot in this directory only.
