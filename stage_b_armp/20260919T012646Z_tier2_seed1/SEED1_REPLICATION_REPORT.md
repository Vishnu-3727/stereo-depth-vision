# Seed-1 Replication Report — Tier-2 ARM-P (read-only verification)

Run dir: `stage_b_armp/20260919T012646Z_tier2_seed1/` (seed-1 replication, COMPLETE).
Prior run: `stage_b_armp/20260918T142128Z_tier2_pilot/` (seed 0, closed — read only, untouched).
Method: no training, no re-evaluation, no tuning. No existing file was modified; the only
writes of this verification are this report and `CORRECTIONS.json` in the run dir
(scratch scripts lived in the system Temp dir, never in the repo).

Scope note: per instructions this report states no classification, no next action, no
significance claim, no confidence interval, and no pooled/averaged statistic across seeds.
The 0.0444 px single-seed spread is context only. The control is RANDOM INITIALIZATION
under the original P2A recipe (never called "P2A checkpoint" initialization here).

## 1. Integrity

- `git status --porcelain` (run from repo root): 5 modified tracked files —
  `M .gitignore`, `M src/models/stereonet/__init__.py`, `M src/models/stereonet/cost_volume.py`,
  `M src/models/stereonet/regression.py`, `M src/models/stereonet/stereonet.py` —
  plus untracked entries (`CLAUDE.md`, `PHASE_2_FINAL_REPORT.md`, `phase0/`, `phase1/`,
  `phase2/`, `src/datasets/driving.py`, `src/datasets/pfm.py`,
  `src/models/stereonet/excitation.py`, `stage_a_diagnostics/`, `stage_b_armp/`,
  `tests/test_*.py`, and a `.audit_pytest/` permission warning). None of the modified
  tracked files is inside the run dir; all five have mtimes 2026-09-15 through 2026-09-17,
  i.e. before the seed-1 run start (2026-09-19T06:56:56+05:30,
  `stage_b_armp/20260919T012646Z_tier2_seed1/launch_tier2.ps1`). They are pre-existing
  working-tree changes, not products of this run. `git diff --stat`: 5 files, +113/−15.
- Seed-0 pilot dir: newest file is
  `stage_b_armp/20260918T142128Z_tier2_pilot/TIER2_FROZEN_AUDIT.md` (2026-09-19 06:43:58),
  the authorised prior-turn artefact; every other file in that dir is 2026-09-18 evening.
  No file in the seed-0 pilot dir is newer than the seed-1 run start.
- `phase2/`: newest file `phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md` (2026-09-17
  20:57). No file in `phase2/` is newer than the seed-1 run start.
- `stage_a_diagnostics/scripts/d3_matching.py` MD5 recomputed: `AB29A921C121F5E404014989532172A6`
  — matches the expected `ab29a921c121f5e404014989532172a6`.

## 2. Seed

- `stage_b_armp/20260919T012646Z_tier2_seed1/scripts/run_arm.py:83`:
  `env["P2A_SEED"] = "1"`.
- `stage_b_armp/20260919T012646Z_tier2_seed1/control/p2a_record.json > config.seed` == `1`.
- `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_record.json > config.seed` == `1`.
- `stage_b_armp/20260919T012646Z_tier2_seed1/control/stdout.log:203`:
  `P2A seed 1 done: wall 3812.7s best10 EPE 1.9931 @epoch 185`.
- `stage_b_armp/20260919T012646Z_tier2_seed1/armp/stdout.log:203`:
  `P2A seed 1 done: wall 3728.1s best10 EPE 1.6027 @epoch 185`.
- `resolved_config.json > seed` == `1`; `control.generator_seed` == `1`,
  `armp.generator_seed` == `1`; `gate.json > items.1_seed_verified.verdict` == `PASS`
  (child-process echo confirmed `P2A_SEED=1` for both arms).
- Counter-record (defect, cosmetic only — see §13 and `CORRECTIONS.json`):
  `control/record.json:4` and `armp/record.json:4` both read `"seed": 0`, from the stale
  literal at `scripts/run_arm.py:199`. Both arms affected identically.

## 3. Configuration identity

- Diff of `control/p2a_record.json > config` vs `armp/p2a_record.json > config` (seed 1):
  the ONLY differing keys are the initialization fields —
  `initialisation` (`'PyTorch defaults, random (P2A recipe)'` vs
  `'strict load of C:\Users\vishn\stereo_depth_vision\stage_b_armp\20260918T062146Z_stage1_pretrain\checkpoints\armp_stage1_best.pth'`)
  and `init_record` (`{'source': 'random', ...}` vs `{'source': '<...armp_stage1_best.pth>',
  'strict': True, 'sha256': '3ae6fb3b...', 'n_keys': 70, 'param_count': 397954}`).
- Diff of each seed-1 config against the seed-0 pilot's corresponding config: the ONLY
  differing key on EACH arm is `seed` (`1` vs `0`). Recipe unchanged between seeds.
  No STOP condition.

## 4. Initialization identity

- CONTROL: random — `control/p2a_record.json > config.initialisation` ==
  `'PyTorch defaults, random (P2A recipe)'`, `init_record` ==
  `{'source': 'random', 'note': 'PyTorch defaults, random (P2A recipe)'}`.
  (RANDOM INITIALIZATION under the P2A recipe; not a checkpoint.)
- ARM-P: strict load of
  `stage_b_armp/20260918T062146Z_stage1_pretrain/checkpoints/armp_stage1_best.pth`;
  recomputed sha256 == `3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7` —
  matches the expected value and the `init_sha256` recorded in `armp/record.json:6`.
- Both arms' `integrity_guard.json` (identical files): `param_count` 397954
  (`param_count_ok` true), `n_keys` 70, `state_dict_keys_match_arm_v` true,
  `state_dict_shapes_match_arm_v` true, `config` `{downsample_levels: 3, num_disparities: 24,
  cost_volume_shift: "right", regression_normalize: true}`, `no_batchnorm` true,
  `all_ok` true.
- No optimizer/scheduler state inherited: `gate.json > items.8_no_optimizer_inherited`
  verdict PASS — Stage-1 blob keys are exactly `[config, model]`, both `torch.save` calls
  persist only `{"model": ..., "config": ...}`, init is applied before optimizer/scheduler
  construction (offsets guard 9691 < init 10384 < opt 10483 < sched 10563). Corroborated by
  fresh processes: control `train_pid` 9008 vs `runner_pid` 18280; armp `train_pid` 23244
  vs `runner_pid` 16544 (`control/record.json:15-16`, `armp/record.json:15-16`).

## 5. Training completion

- `control/epoch_log.jsonl`: 200 rows, epochs 0–199, last row carries val metrics.
  `control/record.json > epochs_completed` == `200`.
- `armp/epoch_log.jsonl`: 200 rows, epochs 0–199, last row carries val metrics.
  `armp/record.json > epochs_completed` == `200`.
- Exactly 200 epochs on both arms. Durations: control `training_duration_s`
  3826.524335384369 (`p2a_record.json > wall_clock_s` 3812.7223193645477); armp
  `training_duration_s` 3742.4882850646973 (`wall_clock_s` 3728.067950487137).
- Device: `cuda` on both arms (`p2a_record.json > config.device`);
  `software.gpu` == `NVIDIA GeForce RTX 4060 Laptop GPU` (torch 2.7.0+cu128, CUDA 12.8).

## 6. Checkpoint selection

- Rule confirmed as implemented: minimum 10-scene in-training validation EPE over the grid
  {every 5th epoch 0–195} plus epoch 199 (41 scored epochs; verified `grid_ok=True` for
  both arms by re-scanning `p2a_record.json > history`).
- CONTROL: grid minimum epoch 185, EPE 1.993131879083116 ==
  `best_epoch` 185 / `best_val_epe_10scene` 1.993131879083116. Matches
  `record.json > train_monitor_best_epoch` 185.
- ARM-P: grid minimum epoch 185, EPE 1.602660483368526 ==
  `best_epoch` 185 / `best_val_epe_10scene` 1.602660483368526. Matches record.
- Frozen 40-scene score played no part: selection is written by the trainer into
  `p2a_record.json` during training; `control/p2a_best.pth` (mtime 1789785058) and
  `armp/p2a_best.pth` (mtime 1789788809) both predate `tier2_eval.json` (mtime 1789789100).
- Hashes recomputed independently (sha256 of the `.pth` bytes):
  control best `f3e71d12bffc202f03c9ba00ea0b0967459b53c3834d97765a9c8b18adf9474d`,
  control final `5505a047939670883b76744b384e0b9b44b1e0d87925e0ac12c3b5ce8195240c`,
  armp best `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454`,
  armp final `95e2def926288a320ac4500d7d228168227e891d2eaa8218b7ceef42c617ebe9` —
  each matches its `.sha256` sidecar, its `p2a_record.json` entry, its `record.json`
  entry, and its `tier2_eval.json` entry (all four agree on all four checkpoints).

## 7. Frozen-contract verification

All four scored checkpoints in `tier2_eval.json` (`control/best`, `control/final`,
`armp/best`, `armp/final`): `guard.scenes` == `40`, `guard.valid_pixels` == `3802797`,
`guard.gt_scale` == `256.0`, `guard.gt_source` == `disp_occ_0`
(split `hailo_val`, `contract_match` == `true`); `metrics.valid_pixels` == `3802797`.
`metadata.device` == `cuda`, `git_head` == `58e8a19908ddbd35451652c61aef478f56b51ebd`.
- Script provenance: independent difflib diff of the seed-0 pilot
  `scripts/eval_tier2.py` vs the seed-1 copy shows exactly ONE differing line — line 134,
  the checkpoint provenance label
  (`stage_b_armp/20260918T142128Z_tier2_pilot/...` → `stage_b_armp/20260919T012646Z_tier2_seed1/...`).
  `finetune_pilot.py` and `run_pilot.py` copies are byte-identical (ratio 1.0);
  `run_arm.py` differs only at line 83 (`P2A_SEED` 0→1). Consistent with the run's own
  `harness_diff.json` verdict CLEAN (3 classified blocks, zero UNEXPECTED).

## 8. CONTROL results (seed 1, frozen 40-scene)

- Best (`control/best`, epoch 185, sha256 `f3e71d12…9474d`): EPE 1.4440242127250567,
  D1 8.61534286473877, valid pixels 3802797, scenes 40.
- Final (`control/final`, sha256 `5505a047…95240c`): EPE 1.4517076971230227,
  D1 8.63056849997515, valid pixels 3802797, scenes 40.

## 9. ARM-P results (seed 1, frozen 40-scene)

- Best (`armp/best`, epoch 185, sha256 `b2f6f5d5…eb7454`): EPE 1.191216765057325,
  D1 6.211033615520366, valid pixels 3802797, scenes 40.
- Final (`armp/final`, sha256 `95e2def9…17ebe9`): EPE 1.2017375876950878,
  D1 6.239354874846067, valid pixels 3802797, scenes 40.

## 10. Paired differences (seed 1; delta = ARM-P − CONTROL, full precision; negative = ARM-P lower)

- `delta_best = 1.191216765057325 - 1.4440242127250567 = -0.25280744766773156`.
- `delta_final = 1.2017375876950878 - 1.4517076971230227 = -0.24997010942793496`.
- Sign only: both negative (ARM-P lower on best and on final). No further interpretation
  offered here; the 0.0444 px single-seed spread is context only.

## 11. Per-bin results (seed 1; all four bins, best AND final, both arms)

Bin pixel counts/fractions (identical for all four checkpoints — same frozen 40 scenes):
`<64`: 3627325 px (0.9538571214818987); `64-96`: 141320 px (0.037162120407689396);
`96-128`: 33072 px (0.008696756624137445); `>=128`: 1080 px (0.00028400148627444484).

| bin | control best | armp best | delta best | control final | armp final | delta final |
|---|---|---|---|---|---|---|
| <64 | 1.2457348829434665 | 1.0478466016949992 | -0.19788828124846725 | 1.2427480828353306 | 1.053314475752373 | -0.18943360708295764 |
| 64-96 | 3.692190249837427 | 2.748458721790913 | -0.9437315280465137 | 3.7917937435511315 | 2.800875857906844 | -0.9909178856442877 |
| 96-128 | 11.6295194414196 | 9.58654764469084 | -2.042971796728759 | 12.32172967361608 | 9.931008959946912 | -2.390720713669168 |
| >=128 | 61.34617289967007 | 21.867477544148763 | -39.47869535552131 | 64.20180585296066 | 23.14079440081561 | -41.06101145214505 |

- The `>=128` bin holds 1080 px at seed 1 — the same count as at seed 0 (1080 px).
  Deltas there rest on ~0.03% of valid pixels; quoted for completeness, not weighted.
- The 64–128 range is the Stage-A-relevant regime, and ARM-P is lower there in every
  listed cell — however, a bin improvement does NOT establish a mechanism; curve/bin
  shape is reported separately from error and is not interpreted as mechanism.

## 12. Comparison with seed 0 (side by side; seeds kept separate — no pooling, no averaging, no combined statistic)

Seed-0 values (quoted exactly as specified; verified against
`stage_b_armp/20260918T142128Z_tier2_pilot/tier2_eval.json` full-precision values
1.5587715928467816, 1.2057590024628537, 1.4809434000217176, 1.2087612591692831):
CONTROL best 1.5587716, ARM-P best 1.2057590, delta -0.3530126;
CONTROL final 1.4809434, ARM-P final 1.2087613, delta -0.2721821.

| selection | seed 0 CONTROL | seed 0 ARM-P | seed 0 delta | seed 1 CONTROL | seed 1 ARM-P | seed 1 delta |
|---|---|---|---|---|---|---|
| best | 1.5587716 | 1.2057590 | -0.3530126 | 1.4440242127250567 | 1.191216765057325 | -0.25280744766773156 |
| final | 1.4809434 | 1.2087613 | -0.2721821 | 1.4517076971230227 | 1.2017375876950878 | -0.24997010942793496 |

Seed-0 per-bin reference (same bins, from the seed-0 pilot `tier2_eval.json`):
control best 1.3791478 / 3.5811728 / 11.4630732 / 36.9227622;
control final 1.2849291 / 3.5528602 / 12.3719786 / 55.1993712;
armp best 1.0583472 / 2.8716970 / 9.5243954 / 23.5818674;
armp final 1.0636665 / 2.8527439 / 9.3851977 / 23.0300698 (all with `>=128` = 1080 px).
No seed-0 result is restated as corrected; seed-0 figures above are the pilot's own.
Two seeds do not establish significance; no significance is claimed.

## 13. Deviations

1. **`scripts/run_arm.py:199` stale-seed literal (confirmed, cosmetic only, symmetric).**
   The `record.json` writer hardcodes `"seed": 0`, so `control/record.json:4` and
   `armp/record.json:4` both read `0` while the actual training seed is 1 (evidence §2).
   Proved cosmetic: the training seed is consumed exclusively via `P2A_SEED` env
   (`finetune_pilot.py:59`, set to `"1"` at `run_arm.py:83`); `record.json` is written at
   `run_arm.py:210` after training completes and its `seed` field is never read by any
   training, init, data-ordering, or evaluation path (only readers are
   `run_pilot.py:94-97`, which consume `train_monitor_best_epoch` and
   `training_duration_s`; `eval_tier2.py` never reads `record.json`). Both arms affected
   identically — the comparison is unbiased by it. Full write-up: `CORRECTIONS.json`.
   `run_arm.py` was not edited, per instruction.
2. **Stale seed-0 labels propagated into seed-1 aggregates (same root cause, also cosmetic).**
   `pilot_results.json:2` `"experiment": "ARM-P TIER-2 PILOT (single seed 0, ...)"` and
   `:3` `"seed": 0` are hardcoded at `run_pilot.py:138-139` (byte-identical to the pilot
   copy); the embedded `control.record` / `armp.record` sub-objects carry the stale
   `seed: 0` verbatim. All numeric content in the file is seed-1 values (e.g. frozen best
   1.4440242127250567 / 1.191216765057325). `README.md:58` section header likewise reads
   `## Tier-2 pilot run (seed 0, ...)` from the hardcoded `run_pilot.py:173` string, while
   the numbers beneath it are seed-1 values.
3. **Stale pre-launch status text in `README.md:8-13`.** The file's correct seed-1 header
   (lines 1–6) and gate documentation coexist with the lines
   `**BUILD AND GATE ONLY. The 200-epoch training was NOT launched.**` /
   `control/ and armp/ do not exist` — false after training completed (both dirs exist
   with 200-epoch logs and checkpoints). Pre-existing artefact text, not edited.
4. **Observed only (not a deviation by this verification):** `pilot_results.json` and
   `README.md:75` contain harness-generated verdict strings produced by the pilot's
   mechanical rule. This report adds no verdict of its own, per instructions.
5. **No other deviations found.** Config identity (§3), init identity (§4), 200-epoch
   completion on both arms (§5), argmin checkpoint selection with frozen eval excluded
   (§6), and the frozen contract on all four checkpoints (§7) all verify clean.
   `gate.json` overall: `GATE PASS (10/10)`.
