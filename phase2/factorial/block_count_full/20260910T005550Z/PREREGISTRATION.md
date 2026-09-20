# EXP-BLOCKCOUNT-FULL-001 — pre-registration (BATCH 1)

Written and frozen **before the first run**. Record directory
`phase2/factorial/block_count_full/20260910T005550Z/`.

Authorised by the research lead on 2026-09-10, in response to the
`CONTINUE-FULL` verdict of `EXP-BLOCKCOUNT-SEEDSCREEN-001`
(`phase2/factorial/block_count_seed_screen/20260909T145659Z/`).

---

## 1. WHY THIS SHAPE

The screen returned `CONTINUE-FULL` and, in the same record, warned that a naive
3-arm × 2-seed 200-epoch campaign is **at risk of being underpowered**: at 30
epochs the seed spread at fixed architecture was the same order as — and at 4
blocks larger than — the architecture spread within a seed.

The lead's decision is therefore **not** to launch all six 200-epoch runs. The
campaign is sequential, with a hard review gate after batch 1:

> Run 6b/seed1, 6b/seed2 and 5b/seed1 first. The two 6-block seeds establish the
> seed trajectory range at fixed architecture; the one 5-block arm says whether
> an architecture difference survives past the 30-epoch emergence region. If the
> 6-block seed variation remains enormous, or 5b behaves wildly differently, we
> reassess before spending the remaining ~4 GPU-hours.

This is the maximum information per GPU-hour, not a claim that the other three
runs are unnecessary.

## 2. EXPERIMENT MATRIX — BATCH 1 ONLY

| Arm | Blocks | Seed | Epochs | Role |
| --- | -----: | ---: | -----: | ---- |
| A   | 6 | 1 | 200 | seed-range reference at fixed architecture |
| A   | 6 | 2 | 200 | seed-range reference at fixed architecture |
| B   | 5 | 1 | 200 | the one architecture arm in this batch |

Run **sequentially**, one process each, in that order.

**NOT authorised, NOT scheduled, and refused by the harness:** 5b/seed2,
4b/seed1, 4b/seed2 (`BATCH1` guard in `cmd_train` returns exit 1 for anything
outside the three pairs above), and seed 0 at any block count.

Estimated cost from the screen's measured throughput (~780 s for 30 epochs):
**~1.4–1.6 GPU-h per run, ~4.5–5 GPU-h for the batch**. Stated here so the
figure is on record before, not after.

## 3. FROZEN CONTROLS — READ-ONLY, NOT RETRAINED

- Stage B deterministic run A — 6 blocks, seed 0, 200 epochs, 40-scene EPE
  2.2955 / D1 15.6045.
- `EXP-BLOCKCOUNT-DETERMINISTIC-001` arm B — 5 blocks, seed 0, 200 epochs,
  40-scene EPE 2.6458 / D1 21.3712.
- `EXP-BLOCKCOUNT-DETERMINISTIC-001` arm C — 4 blocks, seed 0, 200 epochs,
  40-scene EPE 2.3387 / D1 16.9797.
- The seed screen's six 30-epoch runs.

These are quoted for context and **never overwritten**. All output of this
experiment is confined to this timestamped directory.

## 4. ONLY VARIABLES

```
block count ∈ {6, 5}      (4 blocks is not in batch 1)
seed        ∈ {1, 2}      (6 blocks only; 5 blocks is seed 1 only)
```

Everything else is the frozen recipe, imported unchanged: H2 standardised
readout, `cost_volume_shift = left`, 12 disparity candidates, same feature
extractor, cost volume, aggregation, refinement implementation, loss, optimizer
(Adam lr 1e-3), dataset (`hailo_calib` train / `hailo_val` val), crop policy,
augmentation, batch size 2, initialisation procedure.

**Unlike the screen, nothing about the budget or the schedule is modified.**
The screen shortened the budget to 30 while pinning `T_max = 200`; here budget
and schedule are both the native 200. `CHECKPOINT_EPOCHS` (1, 2, 5, 10, 20, 50,
100, 150, 200) and `WEIGHT_SNAPSHOT_EPOCHS` (10, 20, 50, 100, 150, 200) are the
imported originals and are **asserted**, not reassigned.

## 5. DETERMINISM

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Recorded per run in `run_meta_*.json`.

## 6. INITIALISATION

Fresh, per the frozen recipe. **No checkpoint is loaded** — not the Stage B
checkpoint, not the screen's 30-epoch checkpoints (which in any case store only
`{"model", "epoch", "config"}` and cannot be resumed). Initial weight SHA is
recorded for every run and is asserted to differ from the seed-0 initialisation.
Seed-1 and seed-2 hashes are not forced to match anything.

## 7. PRE-FLIGHT — BLOCKING

`preflight` must print `PREFLIGHT PASS` before any training. `run_training`
independently refuses to start if `results/preflight.json` is missing or failed.
Checks: Phase 1 diff empty; deterministic controls enabled; shift = left;
standardised readout; 12 candidates; block counts and dilations correct;
parameter counts match registration; seeds are 1/2 only; seed 0 not scheduled;
epoch budget 200; `T_max` 200 and not re-fitted; fresh initialisation; no Stage B
checkpoint; optimizer state fresh; surviving weights bit-identical within a seed;
run count 3; the batch is exactly {A1, A2, B1}.

**STOP if** a run would use seed 0, a budget other than 200, the wrong block
count, a loaded checkpoint, non-deterministic settings, or any arm/seed outside
batch 1.

## 8. TRAINING

Exactly 200 epochs per run. No tuning, no learning-rate change, no augmentation
change, no dataset change. **Failures are preserved**: an aborted or unfavourable
run is recorded and analysed, never rerun and never deleted. The driver
continues to the next run after a non-zero exit rather than unwinding.

The frozen in-run gates stay active and may legitimately end a run early:
- **epoch-10 viability gate** — abort on failure (this is the recipe's own gate);
- **epoch-100 collapse check** — abort on failure;
- **epoch-200 stereo verdict** — recorded, never an abort.

An abort by one of these is a result, not an error.

## 9. CHECKPOINTS

The project's existing mechanism, at the native epochs (1, 2, 5, 10, 20, 50,
100, 150, 200), with weight snapshots at 10, 20, 50, 100, 150, 200. No new
architecture, no new checkpoint format.

## 10. METRICS

Recorded at every checkpoint epoch: validation EPE, validation D1, validation
loss, training loss, gradient median/max, NaN/Inf status, matching-path gradient
presence fraction, and the softmax/disparity stage statistics.

## 11. STEREO PROBES

The established probes only (`exp_h2_seed_replication.stereo_probe`,
`exp_h2_softargmin_scale.validation_stage_stats`), imported unchanged, at epochs
10, 20, 50, 100, 150, 200: right-image dependence, matching-map dependence,
initial-disparity correlation with GT, softmax entropy, disparity standard
deviation. No new stereo metric.

They establish binocular **dependence**, not correct disparity search — Phase 1
EXP-007 measured +89.7 D1 right-image penalty on provably degenerate weights.

## 12. EVALUATION

The project's validation protocol during training (first 10 `hailo_val` scenes,
full frames, pooled over `gt > 0`), plus the existing 40-scene evaluator on all
40 `hailo_val` scenes at epoch 200 for each run. Same evaluator, unchanged.

## 13. ANALYSIS — PRE-SPECIFIED

1. **Seed range at 6 blocks.** |seed1 − seed2| for EPE and D1 at epochs 10, 20,
   50, 100, 150, 200, plus the 40-scene figures at 200, plus the frozen seed-0
   6-block record → a three-seed picture at fixed architecture.
2. **The 5-block arm read against that range.** Whether 5b/seed1 falls inside or
   outside the 6-block seed interval, at each checkpoint. Reported descriptively.
3. **Trajectory shape.** Whether the ordering seen at 30 epochs survives to 200,
   and where the curves cross.
4. **Stereo functionality** at each probe epoch.

## 14. NO POST-HOC THRESHOLD

No materiality band is defined, before or after seeing the results. The
invalidated E3b/E3c band is neither reused nor replaced. Measuring the 200-epoch
seed scale is part of what this batch is *for*; inventing a threshold from the
numbers it produces would defeat that.

## 15. CLAIM CEILING — BINDING

Batch 1 is **two seeds at 6 blocks and one seed at 5 blocks**. A single 5-block
seed cannot separate an architecture effect from a seed effect.

The following statements are **not supportable from batch 1** and will not be
written:

- "5 blocks are required." / "5 blocks are sufficient."
- "6 blocks are superior."
- "4 blocks are equivalent." (4 blocks is not even in this batch.)

Any ordering statement must be scoped to the named seeds and checkpoints.

## 16. DECISION RULE — chosen after the analysis, from this fixed set

Exactly one of:

- **CONTINUE-BATCH-2** — the 6-block seed range is small enough that a
  single-seed architecture arm is readable, and 5b/seed1 sits far enough outside
  it to be worth resolving. Recommend the smallest batch 2 that resolves it.
- **REDUCE-SCOPE** — the 6-block seed range is small **and** 5b/seed1 sits
  inside it, so the remaining three runs would likely add little. Recommend the
  smaller experiment, do not launch it.
- **STOP-UNDERPOWERED** — the 6-block seed range at 200 epochs is large enough
  that no realistic number of seeds separates these architectures on this
  harness. Recommend one diagnostic that changes that, not more seeds.

Recommend exactly one next experiment. **Do not execute it.**

## 17. OUTPUT

```
phase2/factorial/block_count_full/20260910T005550Z/
  PREREGISTRATION.md   README.md   RESULTS.md   ENVIRONMENT.txt   CONFIG.json
  results/preflight.json  results/history_arm_*.json
  run_meta_*.json  train_*_stdout.log  eval40_*.json  probes.json
  comparison.json  checkpoints/  experiments/
```

Nothing outside this directory is written.

## 18. PROVENANCE

`git rev-parse HEAD`, `git rev-parse phase-1-frozen^{commit}`,
`git status --short` and `git diff phase-1-frozen -- src scripts` recorded before
and after. **The Phase 1 diff must be empty at both ends.** Historical E3b / E3c
/ O6 / H2 / Stage A / Stage B / deterministic-block-count / seed-screen records
are read, never written.

## 19. HARD STOP

After the three runs, the 40-scene evaluations, the probes and the analysis:
**STOP.** Do not launch batch 2, do not add a seed, do not extend a run, do not
change the architecture, do not open Stage E, do not rerun H2 / E3b / E3c / O6,
do not run dilation experiments. The research lead decides what happens next.
