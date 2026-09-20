# EXP-BLOCKCOUNT-SEEDSCREEN-001 — `20260909T145659Z`

**A cheap 30-epoch seed/trajectory screen. Not the block-count experiment.**
Stage E stays CLOSED.

**Question.** At seeds 1 and 2, do the 6-, 5- and 4-block configurations already
show meaningful trajectory variation or changing architecture ordering within the
first 30 epochs — enough to justify a full 6-run × 200-epoch campaign (~8
GPU-hours)?

**Answer.** See `RESULTS.md` (`VERDICT`).

## Why

The deterministic seed-0 series `EXP-BLOCKCOUNT-DETERMINISTIC-001`
(`20260909T131500Z`) ordered `6 < 5 > 4` on 40-scene EPE — non-monotone in block
count, so capacity cannot be read off one seed. Rather than spend 8 GPU-hours to
find out whether the campaign is even informative, this spends ~1.5 on six
30-epoch runs that answer only that question.

## Design

- **Matrix:** block count ∈ {6, 5, 4} × seed ∈ {1, 2} × 30 epochs = **6 runs**.
- **Seed 0 is not rerun.** Its epoch-10/20/30 validation points are read out of
  the frozen records (Stage B deterministic run A for 6 blocks;
  `EXP-BLOCKCOUNT-DETERMINISTIC-001` arms B and C for 5 and 4) and quoted as
  read-only context. Nothing historical is retrained or overwritten.
- **Only variables:** block count and seed. Everything else — readout
  (standardised), `cost_volume_shift = left`, 12 candidates, feature extractor,
  cost volume, aggregation, refinement implementation, loss, optimizer, LR
  schedule, dataset, crop, augmentation, batch size, initialisation procedure —
  is imported unchanged from the Stage B / E3b recipe.
- **Protocol:** the Stage B deterministic controls
  (`use_deterministic_algorithms`, `cudnn.deterministic`, no `cudnn.benchmark`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`).
- **No materiality band.** The historical E3b/E3c band is invalidated; none was
  reused and none was invented, before or after seeing the results.

### The one protocol subtlety

The epoch **budget** is 30. The learning-rate **schedule** stays at its frozen
`T_max = 200` and is deliberately *not* re-fitted to 30 epochs. Re-fitting it
would change the learning rate at every epoch — which the design forbids — and
would turn each run into a fully-annealed 30-epoch training that says nothing
about the 200-epoch trajectory it exists to screen. Each run here is a strict
**prefix** of the 200-epoch run it forecasts, which is also what makes the
seed-0 epoch-10/20/30 points directly comparable at zero cost.

Enforced by `FrozenCosineAnnealingLR`; asserted by the preflight check
`schedule_T_max_frozen_at_200`. Registered in advance in `PREREGISTRATION.md`
§5.1.

## Files

| file | contents |
|---|---|
| `PREREGISTRATION.md` | frozen before the first run — question, matrix, frozen variables, the T_max decision, preflight, claim ceiling, verdict rule, no-band rule, hard stop |
| `RESULTS.md` | the six-run matrix, trajectories, seed effect, ordering, probes, verdict |
| `ENVIRONMENT.txt` | git provenance before and after, preflight summary, per-run environments |
| `CONFIG.json` | the recorded configuration of the screen |
| `results/preflight.json` | all 18 blocking checks, per-run init weight SHAs, MACs |
| `run_meta_<run>.json` | per-run wall clock, controls, init/final weight SHAs, checkpoint SHA-256, snapshot list |
| `experiments/<run>/{config,env,log,metrics}.json` | harness records from the imported code path |
| `eval40_<run>.json` | 40-scene `hailo_val` evaluation at epoch 30, all per-scene rows |
| `probes.json` | the established stereo probes at epochs 10/20/30 for all six runs |
| `comparison.json` | machine-generated matrix, ordering, seed effect, architecture spread, seed-0 context |
| `checkpoints/` | final checkpoint + epoch 10/20/30 snapshots per run |
| `train_*_stdout.log`, `eval40_*_stdout.log`, `probes_stdout.log`, `run_screen.log` | raw stdout |

Scripts: `phase2/factorial/block_count_seed_screen/exp_blockcount_seed_screen.py`
and `run_screen.sh`.

## Known cosmetic wart

The per-run `config.json` carries a `hypothesis` string inherited verbatim from
the imported E3b recipe ("…matches six-block H2 at the same budget…"). It is a
leftover of the code path, not this experiment's hypothesis, and is superseded
by the `experiment_kind`, `epoch_budget`, `scheduler` and `claim_ceiling` fields
written into the same config. It was left alone rather than edited mid-series,
so all six runs carry identical provenance.

## Provenance

`HEAD` `58e8a19`; `phase-1-frozen^{commit}` `b4207e5`;
`git diff phase-1-frozen -- src scripts` **empty** before and after;
working tree `M .gitignore`, `?? phase2/`.

**Phase 2 is not committed to git.** No external immutability is claimed for any
Phase 2 record, including this one.

## What was not touched

H1, H2, H3, E1, E2, E3, E3b, E3c, O6, Stage A, Stage B, the deterministic
block-count series, the factorial shift cell, every historical report, metric
and pre-registration, and all of Phase 1. Additive only.
