# ARM P — Stage 1 of 2: PRETRAIN ONLY (FT3D A+C subset, NOT full SceneFlow)

**Pretrain only. No fine-tuning. No KITTI training. No `hailo_val` evaluation.
P2A (`train_p2a_scale_coverage.py`) and every existing repo file are untouched;
`phase2/scripts/train_p2a_scale_coverage.py` was mirrored into `scripts/` here
(the established score_mirror precedent). No file outside this directory was
created or modified.**

## Naming integrity

Only FT3D TRAIN A+C are staged on C: — **14,460 usable triplets**
(A: 6,990; B: 0 wholly missing; C: 7,470; Monkaa empty). So this is
**ARM P on the FT3D A+C subset**, NOT full SceneFlow. Nothing was copied from
`D:\sceneflow_archives`. A later FAIL on this subset does not fully discharge
the original ARM P; a PASS would.

## Architecture — exactly P2A, no deviation

`StereoNetConfig(downsample_levels=3, num_disparities=24,
cost_volume_shift=right, regression_normalize=True)`, 397,954 params, no
BatchNorm, 6 refinement blocks, built from `src/models/stereonet` unmodified.
Param count `== 397954` is asserted before training starts; any other value aborts.

## Recipe — P2A mirror, dataset swapped

See `pretrain_config.json` for the field-by-field mirror and the explicit diff
vs `phase2/scripts/train_p2a_scale_coverage.py`: dataset (KITTI→FT3D) and
epochs (200→20) only; loss mask formula identical (`gt>0 AND gt<184`).

## Layout

- `scripts/train_armp_stage1.py` — the mirrored trainer (new file)
- `scripts/ft3d.py` — FT3D dataset class (imports `src/datasets/pfm.py`)
- `scripts/enumerate_triplets.py` — manifest probe (read-only)
- `scripts/launch_stage1.ps1` — detached launcher (tee's stdout to log)
- `dataset_manifest.json` — 14,460 triplets; 13,737 train / 723 holdout (seed 0)
- `pretrain_log.jsonl` — per-epoch: mean loss, pretrain-val EPE, lr, wall clock, valid pixels
- `checkpoints/` — best-by-pretrain-val + final, sha256 in `stage1_record.json`
- `checkpoint_loading_contract_result.json` — strict-load verification (end of run)
- `raw/` — reserved (no NPY cache: per-epoch PFM decode fits the 14 h budget; see run report)

## Run status

Detached background run; survives this session. See the manager's run report
for epoch-1 timing, the 14 h gate decision, and what is NOT MEASURABLE yet
(final/best EPE, loss curve, contract result — pending run completion).
