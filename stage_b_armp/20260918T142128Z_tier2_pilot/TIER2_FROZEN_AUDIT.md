# TIER2 FROZEN AUDIT — read-only audit of completed experiment
Pilot dir: `stage_b_armp/20260918T142128Z_tier2_pilot/` (UTC 2026-09-18T14:21:28Z).
Auditor changed nothing except writing this file. No training, no re-evaluation,
no teammate-repo access. Single seed 0 throughout; no significance is claimed
(reference single-seed spread 0.0444 px, per pilot brief).

## A1. RUN COMPLETION — PASS

Both arms reached epoch 200 (0-indexed last epoch 199, 200 rows each).

- `control/epoch_log.jsonl`: 200 rows, first epoch 0, last epoch 199, last lr 0.0.
- `armp/epoch_log.jsonl`: 200 rows, first epoch 0, last epoch 199, last lr 0.0.
- `control/record.json`: `"status": "COMPLETE"`, `"seed": 0`, `"epochs_completed": 200`,
  `"training_duration_s": 3782.7461800575256`, `"train_monitor_best_epoch": 150`,
  `"train_pid": 26148`.
- `armp/record.json`: `"status": "COMPLETE"`, `"seed": 0`, `"epochs_completed": 200`,
  `"training_duration_s": 3839.4806356430054`, `"train_monitor_best_epoch": 185`,
  `"train_pid": 22112`.
- `control/p2a_record.json`: `epochs_run: 200`, `wall_clock_s: 3771.3353197574615`,
  `config.seed: 0`, `config.epochs: 200`, `config.device: "cuda"`, `best_epoch: 150`.
- `armp/p2a_record.json`: `epochs_run: 200`, `wall_clock_s: 3826.8314871788025`,
  `config.seed: 0`, `config.epochs: 200`, `config.device: "cuda"`, `best_epoch: 185`.
- `tier2_eval.json > metadata`: `"device": "cuda"`, `"git_head": "58e8a19908ddbd35451652c61aef478f56b51ebd"`.
- Same recipe evidence: `recipe_diff.json > conclusion`: "The ONLY deltas are the
  two new flags (--init, --arm) and their direct consequences. No hyperparameter,
  dataset, augmentation, loss, optimizer, scheduler, seed, device, gradient, or
  checkpoint-policy line differs." Auditor-verified `p2a_record.json > config`
  comparison: every config field identical between arms EXCEPT `initialisation`
  and `init_record` (control: "PyTorch defaults, random (P2A recipe)"; armp:
  strict load of `.../20260918T062146Z_stage1_pretrain/checkpoints/armp_stage1_best.pth`,
  sha256 `3ae6fb3b…9be7`).
- Note: two wall-clock fields exist per arm and differ slightly (`record.json >
  training_duration_s` 3782.75/3839.48 vs `p2a_record.json > wall_clock_s`
  3771.34/3826.83); README quotes the `record.json` values (3782.7/3839.5).
  Both confirm full 200-epoch completion; FLAG-free, recorded here for precision.

## A2. CHECKPOINT SELECTION — PASS

Declared-before-training rule (identical in P2A source and pilot mirror):

- `phase2/scripts/train_p2a_scale_coverage.py:250-251,277-281`:
  `best_val_epe = float("inf")` / `best_epoch = -1`, then
  `if m.epe < best_val_epe: best_val_epe = m.epe; best_epoch = epoch;
  torch.save({"model": ..., "config": ...}, OUT_DIR / "p2a_best.pth")`
  inside `if epoch % 5 == 0 or epoch == EPOCHS - 1: m = validate(model, val_base,
  device, limit=10)`.
- `stage_b_armp/20260918T142128Z_tier2_pilot/scripts/finetune_pilot.py:286-287,309-317`:
  byte-identical logic (`best_val_epe = float("inf")`, `if m.epe < best_val_epe: ...
  torch.save(..., OUT_DIR / "p2a_best.pth")`), and the module docstring
  (`finetune_pilot.py:9-11`) states "checkpoint-selection rule — is unchanged
  from P2A".
- Criterion verbatim: **minimum 10-scene in-training validation EPE**
  (`validate(..., limit=10)`), evaluated every 5th epoch plus epoch 199; the
  running minimum is saved as `p2a_best.pth`. Final weights saved unconditionally
  as `p2a_final.pth` (`finetune_pilot.py:328-329`).

Selected checkpoints (sha256 recomputed by auditor with `hashlib.sha256` over
the `.pth` bytes; all match both the `.sha256` sidecar and `tier2_eval.json`):

- CONTROL best: `control/p2a_best.pth`, monitor epoch 150,
  `2940801540ded88ed2ef8e46b1e1a51f37af6cafc00f2273c2021235abe233fe` — match.
- CONTROL final: `control/p2a_final.pth`, epoch 199,
  `572447cafb7284073348f6509c797f67b6388f143f97cbf301032cedcb598b86` — match.
- ARM-P best: `armp/p2a_best.pth`, monitor epoch 185,
  `58968771d987acb0f6ad03743d8b8d952b28dc6b43cb72bc5dd723d6a3419818` — match.
- ARM-P final: `armp/p2a_final.pth`, epoch 199,
  `03e2bee86824ecbdb59ae5ef19f149c794bf31af6cc8ce2d3291dd599af24bc0` — match.

The criterion is monitor-based (10-scene train-time monitor) and independent of
the frozen 40-scene contract score: `record.json > note` and `pilot_results.json`
both state "train_monitor numbers are the 10-scene in-training monitor, NOT the
frozen contract". The monitor values (control 2.0388@150; armp 1.6340@185) differ
from the frozen best EPEs (1.5588 / 1.2058), confirming independence.

## A3. CHECKPOINT INTEGRITY — PASS with FLAG (stale expectation constants in shared harness; NOT a contract violation)

Per-arm integrity (`control/integrity_guard.json`, `armp/integrity_guard.json` —
identical values):

- `param_count: 397954`, `param_count_expected: 397954`, `param_count_ok: true`
- `n_keys: 70`, `state_dict_keys_match_arm_v: true`,
  `state_dict_shapes_match_arm_v: true`, `all_ok: true`
- `config: {downsample_levels: 3, num_disparities: 24, cost_volume_shift: "right",
  regression_normalize: true}`, `max_disparity_px: 184`, `feature_stride: 8`,
  `no_batchnorm: true`, `config_matches_arm_v: true`.

Selected-checkpoint strict load (`tier2_eval.json > strict_load`, all four
checkpoints identical in form): `matched: 70`, `missing: []`, `unexpected: []`,
`strict_ok: true`, `params: 397954`, `n_keys: 70`. Key count 70, 24 disparity
candidates (`num_disparities: 24`), 3 downsample levels, shift direction "right",
`regression_normalize: true` (from eval CFG `eval_tier2.py:36-37`, identical to
`finetune_pilot.py:71-72 > ARM_V_CONFIG`).

*** CRITICAL SUB-TASK — resolved: STALE/WRONG expectation constants, NOT a real
violation. ***

- `tier2_eval.json` reports for every checkpoint `expected_keys: 72`,
  `keys_ok: false`, `params_ok: false` alongside `strict_ok: true, matched: 70`.
- Exact source: `phase1/harness/frozen_eval.py:49-50`:
  `EXPECTED_STATE_KEYS = 72` / `EXPECTED_PARAMS = 423586`, consumed by
  `strict_load_report` at `frozen_eval.py:83-86`:
  `"expected_keys": EXPECTED_STATE_KEYS,`
  `"keys_ok": len(want & got) == EXPECTED_STATE_KEYS,`
  `"params": n_params,`
  `"params_ok": n_params == EXPECTED_PARAMS,`.
- These are stale constants for the OLD default architecture (cf.
  `phase2/scripts/eval_p2a.py:37-45 > MIRROR_NOTE`: `frozen_eval.score_checkpoint
  hardcodes StereoNetConfig() (downsample_levels=4, num_disparities=12, ...)`).
  The pilot architecture (downsample 3 / 24 disparities) has the preregistered
  contract **70 keys / 397954 params** (`finetune_pilot.py:71-73 >
  ARM_V_CONFIG / ARM_V_PARAMS`; `checkpoint_contracts.json` pre-gate:
  `key_count_ok: true`, `param_count_ok: true`, `strict_load_ok: true`;
  `integrity_guard.json > all_ok: true`).
- The operative eval-time guard is local and correct —
  `scripts/eval_tier2.py:38,130-132`: `EXPECTED_PARAMS = 397954` and
  `if compat["missing"] or compat["unexpected"] or params != EXPECTED_PARAMS:
  raise SystemExit("ABORT eval: ...")`. Evaluation completed for all four
  checkpoints, so this guard passed. The `keys_ok: false / params_ok: false`
  flags are downstream informational comparisons against the stale harness
  constants and do not reflect the checkpoint bytes (strict load clean,
  70/70 keys, 397954 params).
- FLAG: `phase1/harness/frozen_eval.py:49-50` constants were not updated for the
  P2A/pilot architecture; any consumer reading `strict_load.keys_ok/params_ok`
  at face value would misread a clean checkpoint as failing. No artefact was
  modified by the auditor.

## A4. FROZEN CONTRACT VERIFICATION — PASS

All four scored checkpoints (`tier2_eval.json > checkpoints > {control,best |
control/final | armp/best | armp/final} > hailo_val`):

- `split: "hailo_val"`, `scenes: 40`, `metrics.valid_pixels: 3802797`,
  `guard.gt_scale: 256.0`, `guard.gt_source: "disp_occ_0"`,
  `guard.contract_match: true` (guard.contract repeats the same five values).
  Verified for all four entries by auditor script.

Scoring-core diff (`scripts/eval_tier2.py` vs `phase2/scripts/eval_p2a.py`):

- Shared/identical: `CFG = dict(downsample_levels=3, num_disparities=24,
  cost_volume_shift="right", regression_normalize=True)`; `BINS =
  [0,16,...,160]`, `MIN_BIN = 1000`; dataset construction
  (`Kitti2015Stereo(..., disparity_scale=256.0, occluded=True)`), valid mask
  (`s.disparity > 0`), `pooled_metrics`, `refuse_unless_contract(...,
  "disp_occ_0")`, bin/stratum math (polyfit slope/intercept, r2, pearson_r,
  rmse, d1_pct formula) — line-for-line same.
- Precise differences only:
  1. `eval_tier2.py:97-100` adds three extra strata `gt_64_96`, `gt_96_128`,
     `gt_ge_128` alongside the kept `gt_lt_64 / gt_ge_64 / gt_ge_96`
     (`eval_p2a.py:104` has only the latter three).
  2. `eval_tier2.py:40-41,123-136` scores `TARGETS = [(control,best),
     (control,final),(armp,best),(armp,final)]` with an abort guard
     (`params != EXPECTED_PARAMS = 397954`, lines 38/130-132) and records
     `params / n_keys / strict_load`; `eval_p2a.py:31,110-147` loops
     `RUNS = {0,1,2}` seeds × `hailo_val` + `hailo_calib` splits, tracks per-image
     `names`, and records `state_dict_keys_match_arm_v / wall_clock_s /
     integrity_guard` instead.
  3. `eval_tier2.py` drops the `names` list (`eval_p2a.py:51,63,102`).
- `eval_p2a.py` NOT modified: `git status --porcelain` shows zero `M phase2/...`
  entries; `git diff --stat` lists only `.gitignore` + 4 `src/models/stereonet/`
  files (no `phase2/` entry); `git diff -- phase2/scripts/eval_p2a.py` is empty.
  Caveat: `phase2/` is untracked in this checkout (`?? phase2/`), so git cannot
  attest to its history; the check performed is "no tracked modification
  recorded". `tier2_eval.json > metadata.git_head` (`58e8a19…`) equals
  `git rev-parse HEAD` (`58e8a19908ddbd35451652c61aef478f56b51ebd`).

## A5. RESULTS TABLE — figures as read (no significance claimed)

| arm | ckpt | epoch | EPE | D1 | valid_pixels | scenes | sha256 |
|-----|------|-------|-----|----|----|----|----|
| control | best | 150 (monitor) | 1.5587715928467816 | 9.94207684501697 | 3802797 | 40 | 29408015…233fe |
| control | final | 199 | 1.4809434000217176 | 9.236333151624974 | 3802797 | 40 | 572447ca…598b86 |
| armp | best | 185 (monitor) | 1.2057590024628537 | 6.337282794742922 | 3802797 | 40 | 58968771…419818 |
| armp | final | 199 | 1.2087612591692831 | 6.349615822248729 | 3802797 | 40 | 03e2bee8…af24bc0 |

Full sha256 values in A2. EPE/D1/valid_pixels/scenes from `tier2_eval.json >
checkpoints > */hailo_val > metrics`; epochs from `record.json >
train_monitor_best_epoch` (best) and epoch_log last row 199 (final);
sha256 cross-checked against sidecars (A2).

- `delta_EPE_best = armp_best − control_best = 1.2057590024628537 −
  1.5587715928467816 = -0.35301259038392785` (matches `pilot_results.json >
  delta_EPE_best`).
- `delta_EPE_final = armp_final − control_final = 1.2087612591692831 −
  1.4809434000217176 = -0.27218214085243453` (matches `delta_EPE_final`).
- Negative = ARM-P lower. Reference single-seed spread 0.0444 noted; per the
  brief, no statistical significance is declared (one seed).

## A6. DISPARITY-BIN TABLE — all four bins, no cherry-picking

Strata from `tier2_eval.json > hailo_val > gt_lt_64 / gt_64_96 / gt_96_128 /
gt_ge_128`. Pixel counts identical across arms/checkpoints (same frozen split):
`<64: 3627325 px (0.95385712)`, `64–96: 141320 px (0.03716212)`,
`96–128: 33072 px (0.00869676)`, `>=128: 1080 px (0.00028400)`.

BEST checkpoints:

| bin | control EPE | armp EPE | delta (armp−control) | px | frac |
|-----|----|----|----|----|----|
| <64 | 1.3791477909085046 | 1.058347205498335 | -0.32080058541016965 | 3627325 | 0.95385712 |
| 64–96 | 3.581172766639597 | 2.871696996790068 | -0.7094757698495293 | 141320 | 0.03716212 |
| 96–128 | 11.463073225631538 | 9.524395371902145 | -1.9386778537293932 | 33072 | 0.00869676 |
| >=128 | 36.92276222794144 | 23.58186738755968 | -13.340894840381761 | 1080 | 0.00028400 |

FINAL checkpoints:

| bin | control EPE | armp EPE | delta (armp−control) | px | frac |
|-----|----|----|----|----|----|
| <64 | 1.284929118711576 | 1.0636665274044859 | -0.22126259130709003 | 3627325 | 0.95385712 |
| 64–96 | 3.5528602287919804 | 2.852743923741699 | -0.7001163050502814 | 141320 | 0.03716212 |
| 96–128 | 12.371978557305919 | 9.385197656392243 | -2.9867809009136757 | 33072 | 0.00869676 |
| >=128 | 55.19937116834853 | 23.03006978918005 | -32.16930137916847 | 1080 | 0.00028400 |

Notes: `>=128` holds only 1080 px (0.028%); `144–160` 16-px bins are
`INSUFFICIENT (<1000 px)` (9 px) in all four evals and excluded from the
strata above by the `MIN_BIN = 1000` rule. No mechanism is inferred.

## A7. REPRODUCIBILITY / FILE INTEGRITY — PASS with FLAG (pre-existing tracked modifications unrelated to the pilot)

- `git status --porcelain` (tracked modifications): `M .gitignore`,
  `M src/models/stereonet/__init__.py`, `M src/models/stereonet/cost_volume.py`,
  `M src/models/stereonet/regression.py`, `M src/models/stereonet/stereonet.py`.
  `git diff --stat`: same 5 files (`113 insertions, 15 deletions` total).
  Untracked (`??`): `CLAUDE.md`, `PHASE_2_FINAL_REPORT.md`, `phase0/`, `phase1/`,
  `phase2/`, `src/datasets/driving.py`, `src/datasets/pfm.py`,
  `src/models/stereonet/excitation.py`, `stage_a_diagnostics/`, `stage_b_armp/`,
  `tests/test_*` (incl. all pilot, phase-2, and harness files).
- NO file under `phase2/` is a modified tracked file (no `M phase2/...` line;
  `git diff --stat` has no `phase2/` entry; `git diff -- phase2/scripts/eval_p2a.py`
  empty). NO eval script carries a tracked modification. No historical artefact
  overwrite is recorded by git (tracked set contains only the 5 files above;
  pilot/phase-2 artefacts are untracked so git cannot prove absence of
  overwrite — stated as a limit, not as a failure).
- `git rev-parse HEAD = 58e8a19908ddbd35451652c61aef478f56b51ebd`, equal to
  `tier2_eval.json > metadata.git_head` and both arms' `p2a_record.json >
  git_head`.
- D3 script MD5: `stage_a_diagnostics/scripts/d3_matching.py` →
  `AB29A921C121F5E404014989532172A6` (via `Get-FileHash -Algorithm MD5`),
  **match** against expected `ab29a921c121f5e404014989532172a6` (case-insensitive).
- No optimizer state reused between arms: `finetune_pilot.py:204-241` constructs
  the model fresh, runs `integrity_guard` on the random-init model, then
  `apply_init(model, INIT)` — which for `random` is a documented no-op and for
  ARM-P calls `model.load_state_dict(sd, strict=True)` loading ONLY the
  `["model"]` weights (`finetune_pilot.py:185-201`) — and only afterwards creates
  `optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))`
  plus `CosineAnnealingLR(optimizer, T_max=EPOCHS)` (lines 240-241) inside
  `main()`, i.e. fresh per process. `record.json` confirms separate processes
  (`train_pid` 26148 vs 22112) with the note "Fresh process => fresh
  optimizer/scheduler". No checkpoint loads optimizer/scheduler state anywhere
  (checkpoints store only `{"model", "config"}` per lines 316/329).
- FLAG: the 5 tracked modifications (`.gitignore`, 4 `src/models/stereonet/`
  files) pre-date this audit and are outside the pilot; the auditor did not
  verify their provenance — recorded so the manager can disposition them.

## OPEN DEFECTS

1. `phase1/harness/frozen_eval.py:49-50` — stale `EXPECTED_STATE_KEYS = 72` /
   `EXPECTED_PARAMS = 423586` produce `keys_ok: false / params_ok: false` on
   clean 70-key / 397954-param checkpoints (all four pilot checkpoints affected
   informationally; operative guard in `eval_tier2.py:130-132` unaffected).
2. Dual wall-clock fields per arm (`record.json > training_duration_s` vs
   `p2a_record.json > wall_clock_s`, ~11–13 s apart) — both confirm completion;
   briefs should cite which field they quote.
3. Five tracked working-tree modifications (`.gitignore`,
   `src/models/stereonet/{__init__,cost_volume,regression,stereonet}.py`)
   exist outside the pilot scope; provenance unverified by this audit.
4. `phase2/`, `phase1/`, `stage_b_armp/` are untracked in this checkout, so
   "eval script unmodified / no artefact overwritten" rests on absence of
   tracked diffs + `git_head` agreement, not on a tracked clean tree.
