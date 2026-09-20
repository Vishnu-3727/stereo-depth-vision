# Seed-2 Replication Report — Tier-2 ARM-P (frozen evaluation)

Run dir: `stage_b_armp/20260919T092454Z_tier2_seed2/` (seed-2, COMPLETE).
Prior runs (read only, untouched): `stage_b_armp/20260918T142128Z_tier2_pilot/` (seed 0),
`stage_b_armp/20260919T012646Z_tier2_seed1/` (seed 1).
Seed-2 training ran on Kaggle (Tesla T4); the frozen evaluation below ran LOCALLY on
this machine (torch 2.7.0+cu128, same evaluator as seeds 0 and 1) so that the three
seeds remain comparable under one constant evaluator. No research source, eval logic,
mask, bin, preprocessing, or disparity convention was modified. No classification, no
next action, no significance claim, no pooled/averaged statistic, and no confidence
interval appear in this report. The 0.0444 px single-seed spread is context only.
The control is RANDOM INITIALIZATION under the original P2A recipe (never called
"P2A checkpoint" initialization here).

## 1. Integrity

- `git status --porcelain` (run from repo root at report time): 5 modified tracked files —
  `M .gitignore`, `M src/models/stereonet/__init__.py`, `M src/models/stereonet/cost_volume.py`,
  `M src/models/stereonet/regression.py`, `M src/models/stereonet/stereonet.py` —
  plus the same untracked entries as at seed 1 (`CLAUDE.md`, `PHASE_2_FINAL_REPORT.md`,
  `phase0/`, `phase1/`, `phase2/`, `src/datasets/driving.py`, `src/datasets/pfm.py`,
  `src/models/stereonet/excitation.py`, `stage_a_diagnostics/`, `stage_b_armp/`,
  `tests/test_*.py`, `.audit_pytest/` permission warning). `git diff --stat`: 5 files,
  +113/−15 — identical to the seed-1 reading. File mtimes:
  `.gitignore` 2026-09-08 14:34:34, `src/models/stereonet/__init__.py` 2026-09-16 16:42:55,
  `src/models/stereonet/cost_volume.py` 2026-09-16 09:09:20,
  `src/models/stereonet/regression.py` 2026-09-15 07:47:44,
  `src/models/stereonet/stereonet.py` 2026-09-17 06:43:19 — all predate the seed-2
  assembly (2026-09-19 ~14:55 IST). Pre-existing working-tree changes, not products
  of this run; no tracked file was modified by this run.
- Seed-0 pilot dir: newest file remains
  `stage_b_armp/20260918T142128Z_tier2_pilot/TIER2_FROZEN_AUDIT.md` (2026-09-19 06:43:58).
- Seed-1 dir: newest file remains
  `stage_b_armp/20260919T012646Z_tier2_seed1/SEED1_REPLICATION_REPORT.md` (2026-09-19 09:14:00).
- `phase2/`: newest file remains
  `phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md` (2026-09-17 20:57:00).
- No file in the seed-0 dir, the seed-1 dir, or `phase2/` is newer than the seed-2
  assembly. Neither prior run dir nor `phase2/` was touched.
- `stage_a_diagnostics/scripts/d3_matching.py` MD5 recomputed: `ab29a921c121f5e404014989532172a6`
  — matches the expected `ab29a921c121f5e404014989532172a6`.

## 2. Seed

- `stage_b_armp/20260919T092454Z_tier2_seed2/control/record.json`: `"seed": 2`,
  `"arm": "control"`, `"status": "COMPLETE"`.
- `stage_b_armp/20260919T092454Z_tier2_seed2/armp/record.json`: `"seed": 2`,
  `"arm": "armp"`, `"status": "COMPLETE"`.
- `stage_b_armp/20260919T092454Z_tier2_seed2/control/p2a_record.json > config.seed` == `2`.
- `stage_b_armp/20260919T092454Z_tier2_seed2/armp/p2a_record.json > config.seed` == `2`.
- The Kaggle copy of `run_arm.py` fixed the stale `seed: 0` literal (line 199 now writes
  `"seed": 2`; line 83 sets `P2A_SEED` to `"2"`) — confirmed: `record.json` reads 2 on
  BOTH arms, matching `p2a_record.json > config.seed`. The seed-1 cosmetic defect is gone.
- `control/stdout.log` last line: `P2A seed 2 done: wall 4662.3s best10 EPE 2.0473 @epoch 180`.
- `armp/stdout.log` last line: `P2A seed 2 done: wall 4604.4s best10 EPE 1.5419 @epoch 145`.

## 3. Configuration identity

- Diff of seed-2 `control/p2a_record.json > config` vs `armp/p2a_record.json > config`:
  the ONLY differing keys are the initialization fields —
  `initialisation` (`'PyTorch defaults, random (P2A recipe)'` vs
  `'strict load of /kaggle/working/repo/checkpoints/armp_stage1_best.pth'`)
  and `init_record` (`{'source': 'random', ...}` vs
  `{'source': '/kaggle/working/repo/checkpoints/armp_stage1_best.pth', 'strict': True,
  'sha256': '3ae6fb3b...', 'n_keys': 70, 'param_count': 397954}`).
- Diff of each seed-2 config against the seed-1 corresponding config: CONTROL differs ONLY
  in `seed` (`2` vs `1`) — recipe unchanged. ARM-P differs in `seed` (`2` vs `1`) plus the
  `initialisation` / `init_record` `source` path string
  (`/kaggle/working/repo/checkpoints/armp_stage1_best.pth` vs the local
  `C:\Users\vishn\stereo_depth_vision\stage_b_armp\20260918T062146Z_stage1_pretrain\checkpoints\armp_stage1_best.pth`)
  — same file content (sha256 `3ae6fb3b…29be7` identical on both), path string only,
  reflecting the Kaggle tree location. No STOP condition.

## 4. Initialization identity

- CONTROL: random — `control/p2a_record.json > config.initialisation` ==
  `'PyTorch defaults, random (P2A recipe)'`, `init_record` ==
  `{'source': 'random', 'note': 'PyTorch defaults, random (P2A recipe)'}`.
  (RANDOM INITIALIZATION under the P2A recipe; not a checkpoint.)
- ARM-P: strict load of `checkpoints/armp_stage1_best.pth`;
  `armp/p2a_record.json > config.init_record.sha256` ==
  `3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7` —
  matches the expected value and `armp/record.json > init_sha256`
  (`3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7`).
  (`control/record.json > init_sha256` == `"random"`.)
- Both arms' `integrity_guard.json` (identical files): `param_count` 397954
  (`param_count_ok` true), `n_keys` 70, `state_dict_keys_match_arm_v` true,
  `state_dict_shapes_match_arm_v` true, `config` `{downsample_levels: 3, num_disparities: 24,
  cost_volume_shift: "right", regression_normalize: true}`, `no_batchnorm` true,
  `all_ok` true. 397954 params confirmed.
- No optimizer/scheduler state inherited: fresh Kaggle processes per arm —
  control `train_pid` 23 vs `runner_pid` 22; armp `train_pid` 24 vs `runner_pid` 23
  (`control/record.json`, `armp/record.json`); `final_lr` 0.0 both arms (cosine schedule
  ran its full course from a fresh optimizer). Init is applied before optimizer/scheduler
  construction per the unchanged `finetune_pilot.py` order.

## 5. Training completion

- `control/epoch_log.jsonl`: 200 rows, epochs 0–199. `control/record.json > epochs_completed` == `200`.
- `armp/epoch_log.jsonl`: 200 rows, epochs 0–199. `armp/record.json > epochs_completed` == `200`.
- Exactly 200 epochs on both arms. Durations: control `training_duration_s`
  4670.959505081177 (`p2a_record.json > wall_clock_s` 4662.283803462982); armp
  `training_duration_s` 4614.262996912003 (`wall_clock_s` 4604.424800634384).
- Device: `cuda` on both arms (`p2a_record.json > config.device`);
  `software.gpu` == `Tesla T4` (Kaggle), torch 2.10.0+cu128, CUDA 12.8, python 3.12.13,
  numpy 2.0.2, platform `Linux-6.12.90+-x86_64-with-glibc2.35` — from both arms'
  `p2a_record.json > software`.

## 6. Checkpoint selection

- Rule as implemented: minimum 10-scene in-training validation EPE over the grid
  {every 5th epoch 0–195} plus epoch 199 (41 scored epochs; verified `grid size 41` for
  both arms by re-scanning `p2a_record.json > history` on `val_epe`).
- CONTROL: grid minimum epoch 180, EPE 2.047308491653386 ==
  `best_epoch` 180 / `best_val_epe_10scene` 2.047308491653386. Matches
  `record.json > train_monitor_best_epoch` 180 / `train_monitor_best_epe_10scene` 2.047308491653386.
- ARM-P: grid minimum epoch 145, EPE 1.5419281402011356 ==
  `best_epoch` 145 / `best_val_epe_10scene` 1.5419281402011356. Matches record
  (`train_monitor_best_epoch` 145).
- Frozen 40-scene score played no part: selection is written by the trainer into
  `p2a_record.json` during training on Kaggle, before the local frozen eval existed;
  `tier2_eval.json` (`metadata.utc` 2026-09-19T09:26:06Z) postdates both trainings.
- Hashes recomputed independently (sha256 of the `.pth` bytes in the new seed-2 dir):
  control best `0e993773030a95a5a06098356d035ac3c3d1391dcbf08e108a270bfc775b8bb2`,
  control final `20363e1f62c02a08478f0ac6fce07071b7d10355a0acfc1bb820747e63bdf7e1`,
  armp best `47a2d3e288ad91a6ce448934b0aed08a415bd8ce11904df602e181f1348f8b6d`,
  armp final `97d2fb064bb4572545a3b7da9aebcf66815fcbd3006431255fe0d5045e4ef9a5` —
  each matches its `.sha256` sidecar (new dir and Kaggle source agree), its
  `p2a_record.json` entry, its `record.json` entry, and its `tier2_eval.json` entry
  (all four agree on all four checkpoints).

## 7. Frozen-contract verification

All four scored checkpoints in the seed-2 `tier2_eval.json` (`control/best`,
`control/final`, `armp/best`, `armp/final`): `guard.scenes` == `40`,
`guard.valid_pixels` == `3802797`, `guard.gt_scale` == `256.0`,
`guard.gt_source` == `disp_occ_0` (split `hailo_val`, `contract_match` == `true`);
`metrics.valid_pixels` == `3802797`. The valid-pixel population is exactly 3,802,797 —
no STOP. `metadata` == `{utc 2026-09-19T09:26:06Z, git_head
58e8a19908ddbd35451652c61aef478f56b51ebd, python 3.12.9, torch 2.7.0+cu128,
numpy 2.5.1, platform Windows-11-10.0.26200-SP0, device cuda}` — the same local
evaluator as seeds 0 and 1.
- Script provenance: difflib diff of the seed-1 `scripts/eval_tier2.py` vs the seed-2
  copy (ratio 0.9935483870967742) shows exactly ONE differing block — line 134, the
  checkpoint provenance label
  (`stage_b_armp/20260919T012646Z_tier2_seed1/...` → `stage_b_armp/20260919T092454Z_tier2_seed2/...`),
  a label only; the load path `PILOT/arm/p2a_<tag>.pth` self-resolves and is unchanged.

## 8. CONTROL results (seed 2, frozen 40-scene)

- Best (`control/best`, epoch 180, sha256 `0e993773…b8bb2`): EPE 1.4932978077501271,
  D1 9.3541411755610415, valid pixels 3802797, scenes 40.
- Final (`control/final`, sha256 `20363e1f…bdf7e1`): EPE 1.4850972621507696,
  D1 9.2964993924208947, valid pixels 3802797, scenes 40.

## 9. ARM-P results (seed 2, frozen 40-scene)

- Best (`armp/best`, epoch 145, sha256 `47a2d3e2…f8b6d`): EPE 1.1996447345404795,
  D1 6.3139315614270233, valid pixels 3802797, scenes 40.
- Final (`armp/final`, sha256 `97d2fb06…4ef9a5`): EPE 1.1963930184331233,
  D1 6.2816658370141765, valid pixels 3802797, scenes 40.

## 10. Paired differences (seed 2; delta = ARM-P − CONTROL, full precision; negative = ARM-P lower)

- `delta_best = 1.1996447345404795 - 1.4932978077501271 = -0.2936530732096476`.
- `delta_final = 1.1963930184331233 - 1.4850972621507696 = -0.28870424371764636`.
- D1 deltas (same sign convention): best `-3.0402096141340182`, final `-3.0148335554067183`.
- Sign only: both negative (ARM-P lower on best and on final). No further interpretation
  offered here; the 0.0444 px single-seed spread is context only.

## 11. Per-bin results (seed 2; all four bins, best AND final, both arms)

Bin pixel counts/fractions (identical for all four checkpoints — same frozen 40 scenes):
`<64`: 3627325 px (0.9538571214818987); `64-96`: 141320 px (0.037162120407689396);
`96-128`: 33072 px (0.008696756624137445); `>=128`: 1080 px (0.00028400148627444484).

| bin | control best | armp best | delta best | control final | armp final | delta final |
|---|---|---|---|---|---|---|
| <64 | 1.2951043220795895 | 1.080659518555703 | -0.2144448035238864 | 1.2779979614086445 | 1.0615071705923362 | -0.2164907908163083 |
| 64-96 | 3.876675041923771 | 2.4990093632322044 | -1.3776656786915669 | 3.9374620633323705 | 2.7327747145287553 | -1.2046873488036152 |
| 96-128 | 11.966208144121257 | 8.010309947505919 | -3.9558981966153386 | 12.555477313189098 | 8.756972058821717 | -3.798505254367381 |
| >=128 | 34.57961635589599 | 22.24549595868146 | -12.334120397214534 | 37.159790145026314 | 21.66850107687491 | -15.491289068151403 |

- The `>=128` bin holds 1080 px at seed 2 — the same count as at seeds 0 and 1 (1080 px).
  Deltas there rest on ~0.03% of valid pixels; quoted for completeness, not weighted.
- The 64–128 range is the Stage-A-relevant regime, and ARM-P is lower there in every
  listed cell — however, a bin improvement does NOT establish a mechanism; curve/bin
  shape is reported separately from error and is not interpreted as mechanism.

## 12. Comparison with seeds 0 and 1 (side by side; seeds kept separate — no pooling, no averaging, no combined statistic)

Seed-0 values (quoted exactly as specified; verified against
`stage_b_armp/20260918T142128Z_tier2_pilot/tier2_eval.json` full-precision values
1.5587715928467816, 1.2057590024628537, 1.4809434000217176, 1.2087612591692831):
CONTROL best 1.5587716, ARM-P best 1.2057590, delta -0.3530126;
CONTROL final 1.4809434, ARM-P final 1.2087613, delta -0.2721821.
Seed-1 values (quoted exactly as specified):
CONTROL best 1.4440242, ARM-P best 1.1912168, delta -0.2528074;
CONTROL final 1.4517077, ARM-P final 1.2017376, delta -0.2499701.

| selection | seed 0 CONTROL | seed 0 ARM-P | seed 0 delta | seed 1 CONTROL | seed 1 ARM-P | seed 1 delta | seed 2 CONTROL | seed 2 ARM-P | seed 2 delta |
|---|---|---|---|---|---|---|---|---|---|
| best | 1.5587716 | 1.2057590 | -0.3530126 | 1.4440242127250567 | 1.191216765057325 | -0.25280744766773156 | 1.4932978077501271 | 1.1996447345404795 | -0.2936530732096476 |
| final | 1.4809434 | 1.2087613 | -0.2721821 | 1.4517076971230227 | 1.2017375876950878 | -0.24997010942793496 | 1.4850972621507696 | 1.1963930184331233 | -0.28870424371764636 |

No seed-0 or seed-1 result is restated as corrected; seed-0/seed-1 figures above are
those runs' own. Three seeds do not establish significance; no significance is claimed.

Plain observation of each arm's spread ACROSS the three seeds (not a variance estimate):
CONTROL best spans 1.4440242127250567–1.5587715928467816 (range ~0.1147);
CONTROL final spans 1.4517076971230227–1.4850972621507696 (range ~0.0334);
ARM-P best spans 1.191216765057325–1.2057590024628537 (range ~0.0145);
ARM-P final spans 1.1963930184331233–1.2087612591692831 (range ~0.0124).
The seed-0/seed-1 control movement exceeds the 0.0444 px reference spread (context only).

## 13. Deviations

1. **Seed 2 trained in a different environment (Kaggle Tesla T4 / torch 2.10.0+cu128 /
   python 3.12.13 / Linux) while seeds 0 and 1 trained locally (RTX 4060 Laptop GPU /
   torch 2.7.0+cu128).** This is an environment difference between seeds that is NOT
   part of the intended independent variable (seed + initialization); stated plainly.
   The recipe (config identity, §3) is unchanged across seeds except the seed.
2. **Evaluation deliberately run locally for all three seeds** (seed-2 `tier2_eval.json >
   metadata`: torch 2.7.0+cu128, device cuda, Windows) — holding the evaluator constant
   keeps the three seeds comparable.
3. **The bundle-v6 addition of `phase1/runs/arm_v/arm_v_best.pth` and the wrapper nesting
   fix** (bundle dataset B v6; wrapper `assemble()` now nests scripts at
   `<root>/stage_b_armp/seed2/scripts/` so `parents[3]` resolves correctly), **and the fact
   that the first portability gate missed both because it never exercised the
   `finetune_pilot.py` entrypoint** — documented in the Kaggle dir's `portability_notes.md`
   (§11–12, honest note on the v1–v4 smoke test). Wrapper-side only; no research file
   was edited for portability (only the two declared `run_arm.py` seed-2 line edits).
4. **ARM-P `initialisation`/`init_record.source` path string differs from seed 1**
   (`/kaggle/working/repo/checkpoints/armp_stage1_best.pth` vs the local absolute path) —
   same file content (sha256 `3ae6fb3b…29be7` identical); cosmetic path-string consequence
   of the Kaggle tree. The seed-1 stale-`seed: 0` `record.json` literal is fixed at seed 2
   (`record.json` reads 2 on both arms, §2).
5. **No other deviations found.** Config identity (§3), init identity (§4), 200-epoch
   completion on both arms (§5), argmin checkpoint selection with frozen eval excluded
   (§6), and the frozen contract on all four checkpoints (§7) all verify clean.
