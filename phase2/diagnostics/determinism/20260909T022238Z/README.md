# Stage A — determinism diagnostic, run `20260909T022238Z`

**DIAGNOSTIC RECORD. Not an experiment.** No architecture was changed, no
hyperparameter was tuned, no scientific record was created, and nothing outside
this directory was written.

**Question.** When the same configuration is run twice with the same seed, where
does the first reproducibility failure occur, and is it caused by the data/RNG
pipeline or by computation?

**Configuration used.** The H2 seed-0 training configuration — identical to
`EXP-E3B-REFINEMENT-CAPACITY-001-ARM-A`: `StereoNet` with
`cost_volume_shift="left"`, `StandardisedDisparityRegression`, six refinement
blocks (dilations 1,2,4,8,1,1), 423,586 parameters, `hailo_calib` (160 scenes),
random 256×512 crops with σ=0.1 gain jitter, Adam lr 1e-3, cosine `T_max=200`,
batch 2, fp32, seed 0. **One epoch (80 batches), never 200.** The recipe pieces
are imported from the existing scripts, not restated:
`build_model` from `exp_h2_seed_replication`, `CroppedKitti` and `validate` from
`exp_h1_cost_volume`, `masked_smooth_l1` from `src.losses.disparity`.

## Runs

| run | flags | fresh process |
|---|---|---|
| `run_A` | default (as every Phase 2 experiment ran) | yes |
| `run_B` | default — byte-identical invocation to A | yes |
| `run_C_deterministic` | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | yes |
| `run_D_deterministic` | identical to C | yes |
| `locate_op` | operator-level identification probe | yes |

## Files

| file | contents |
|---|---|
| `README.md` | this file |
| `RESULTS.md` | **the answers, and the verdict** |
| `RUN_A.md`, `RUN_B.md` | the two default-flag runs, side by side |
| `RUN_C_D.md` | the two determinism-controlled runs |
| `ENVIRONMENT.txt` | git provenance + full environment for all four runs |
| `run_*/record.json` | raw machine-generated records (per-batch losses, RNG hashes, data order, batch-0 provenance) |
| `locate_op/locate_op.json` | operator-level identification |
| `comparison_*.json` | machine-generated diffs between two records |

Scripts (**diagnostic-only**, marked as such in their own docstrings) live one
level up in `phase2/diagnostics/determinism/`:
`diag_determinism.py`, `compare_runs.py`, `diag_locate_op.py`.

## What was *not* touched

- `src/` and every Phase 1 path — `git diff phase-1-frozen -- src scripts` is
  empty before and after.
- Every Phase 2 scientific script, preregistration, report and experiment
  record. **No E3b, E3c or O6 verdict, metric or text was altered.** Where those
  conclusions are affected, `RESULTS.md` says so as a new statement here; it does
  not edit history.
- No checkpoint was written. No `Experiment` record was registered.

## Claim tags

`MEASURED` — produced by these runs. `DERIVED` — arithmetic on them.
`INFERRED` — a reading the evidence does not force. `UNKNOWN`.
