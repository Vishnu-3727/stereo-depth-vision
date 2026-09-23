# bundle_e4ft vs bundle (E0): byte-level differences

Measured 2026-09-23 by hashing every file in both working-tree bundles
(`bundle/` = E0, `bundle_e4ft/` = E4 finetune). Newline-normalized content
comparison for the files that differ in raw bytes.

## Status (read first)

- The `expected_init_sha256` recorded in `bundle_e4ft/`,
  `source_integrity_e4ft.json` and the `kernel_e4ft_seed*` kernels is a
  **dry-run placeholder** (`fbbab289...fab7c83`, sha of a 1 MiB dummy file
  in the scratchpad). The real E4 pretrain checkpoint does not exist yet.
- After `push_e4.py pull-pretrain`: run `build_init_dataset_e4.py` on the
  pulled `e4_pretrain_best.pth`, then **rebuild** the bundle
  (`build_bundle_e4ft.py --init-sha <real>`) and re-render the kernels
  (`push_e4.py push-finetune <seed> <real>`). Nothing here is final.
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
| `scripts/run_arm.py` | 3 lines (newline-normalized diff): two post-hoc guards `!= 200` -> `!= args.epochs`; `--epochs` help drops `, 400 = E3` | E3's corrected guards via `e3_patch` (verbatim reuse). With `--epochs 200` they accept exactly what E0's literals accepted |
| `kaggle_bootstrap.py` | raw bytes differ (CRLF in E0 working tree vs LF in e4ft); text-identical after newline normalization | same one-line `EXP_SUBDIR` path edit; the byte gap is a line-ending artefact of when each bundle was written, not a content change |
| `BUNDLE_MARKER.json`, `configs/stage_e.json` | record `stage-e-e4ft`, `experiment: e4`, external init dataset + expected sha | provenance record, not training semantics |
| `source_integrity.json` | manifest of e4ft (22 identical files, 2 declared deltas, external-init record) | kernel re-verifies these hashes on Kaggle |

## Not compared

`dataset-metadata.json` (upload-time artefact, renewed at each dataset
create/version) is excluded from the comparison.

## Inference (not measured)

With `--epochs 200` the corrected guards are behaviorally equivalent to
E0's literals; this follows from the guard expressions, not from a run —
no E4 finetune has run anywhere yet.
