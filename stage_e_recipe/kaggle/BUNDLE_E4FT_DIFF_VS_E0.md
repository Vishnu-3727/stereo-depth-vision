# bundle_e4ft vs bundle (E0): byte-level differences

Measured 2026-09-23 by hashing every file in both working-tree bundles
(`bundle/` = E0, `bundle_e4ft/` = E4 finetune). Newline-normalized content
comparison for the files that differ in raw bytes.

## Status (read first)

- The `expected_init_sha256` recorded in `bundle_e4ft/`,
  `source_integrity_e4ft.json` and the `kernel_e4ft_seed*` kernels is the
  **real E4 pretrain checkpoint** (`4c16fbe3...aa6b829f`, from
  `init_integrity.json` after `pull-pretrain` + `build_init_dataset_e4.py`).
  The earlier dry-run placeholder (`fbbab289...fab7c83`) is gone; the bundle
  was rebuilt with the real sha.
- Seeds 0 and 1 ran on Kaggle against the previous bundle revision and
  trained all 200 epochs normally, then STOPped with
  `wrong_checkpoint_resume` / returncode 3: that revision's `run_arm.py`
  still compared the init record's sha against the E0 Stage-1 constant
  (`3ae6fb3b...`), not the E4 init. The retarget below (`e4ft_patch`) fixes
  exactly that guard; seeds 0/1 are recovered offline (see
  `stage_e_recipe/e4_recovery.json`), seed 2 runs on the fixed bundle.
- `bundle_e4ft/` and `kernel_e4ft_seed*/` are gitignored regenerables (see
  `stage_e_recipe/kaggle/.gitignore`); only the builders, the template,
  `source_integrity_e4ft.json` and this note are committed.

## Identical (22 files, sha-equal)

All of `src/`, `phase1/harness/`, `phase1/runs/arm_v/arm_v_best.pth`,
`scripts/eval_tier2.py` and `scripts/finetune_pilot.py` — i.e. the entire
recipe (optimizer, loss, schedule, augmentation, training loop) is
byte-identical to E0. Why: E4 changes only the init (spec §4).

## Differing (measured)

| file | difference | why |
|---|---|---|
| `checkpoints/armp_stage1_best.pth` | present in E0, **absent** in e4ft | init ships as its own dataset (`vishnuvardhanksece/stage-e-e4-init`); the kernel resolves `e4_pretrain_best.pth` by name and asserts its sha |
| `checkpoints/README_E4_INIT_IS_EXTERNAL.txt` | e4ft-only | pointer file; exists only because `assemble()` requires `checkpoints/` among `ROOT_DIRS`. Never a checkpoint |
| `scripts/run_arm.py` | 4 lines (newline-normalized diff): two post-hoc guards `!= 200` -> `!= args.epochs`; `--epochs` help drops `, 400 = E3`; post-hoc guard `!= EXPECTED_SHA_ARMP_STAGE1` -> `!= "4c16fbe3...aa6b829f"` | E3's corrected guards via `e3_patch` (verbatim reuse) plus the E4 init-sha retarget via `e4ft_patch`. With `--epochs 200` the epoch guards accept exactly what E0's literals accepted; the init-sha guard accepts the E4 pretrain checkpoint the kernel asserts before training |
| `kaggle_bootstrap.py` | raw bytes differ (CRLF in E0 working tree vs LF in e4ft); text-identical after newline normalization | same one-line `EXP_SUBDIR` path edit; the byte gap is a line-ending artefact of when each bundle was written, not a content change |
| `BUNDLE_MARKER.json`, `configs/stage_e.json` | record `stage-e-e4ft`, `experiment: e4`, external init dataset + expected sha | provenance record, not training semantics |
| `source_integrity.json` | manifest of e4ft (22 identical files, 2 declared deltas, external-init record) | kernel re-verifies these hashes on Kaggle |

## Not compared

`dataset-metadata.json` (upload-time artefact, renewed at each dataset
create/version) is excluded from the comparison.

## Inference (not measured)

With `--epochs 200` the corrected epoch guards are behaviorally equivalent to
E0's literals; this follows from the guard expressions, not from a run.
The init-sha retarget is likewise expression-level: the guard now compares
against the same sha the kernel asserts before training, so an arm that
starts from the authorized E4 init passes it. Seeds 0 and 1's pulled
evidence (200 log rows, `epochs_run` 200, init sha `4c16fbe3...` in
`e4ft_err_seed{0,1}/`) shows both arms trained fully before the false STOP —
no E4 finetune needs re-running for seeds 0/1.
