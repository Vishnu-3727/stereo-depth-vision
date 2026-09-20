# EXP-BLOCKCOUNT-FULL-001 — pre-registration (BATCH 2)

Written and frozen **before the run**. Record directory
`phase2/factorial/block_count_full/20260910T052736Z/`.

Authorised by the research lead on 2026-09-10, after reviewing batch 1
(`phase2/factorial/block_count_full/20260910T005550Z/`, verdict
`CONTINUE-BATCH-2`).

---

## 1. WHY THIS RUN, AND ONLY THIS RUN

Batch 1 measured the 200-epoch 6-block seed range on the authoritative 40-scene
protocol:

```
6-block EPE:  2.2955 (seed 0) / 2.3982 (seed 1) / 2.3950 (seed 2)
spread     :  0.1027 px
5b / seed 1:  2.4730 EPE, 18.206 D1
```

5b/seed1 is above every 6-block draw, but by 0.075 px — less than the 6-block
spread itself — and it sits *inside* the 6-block D1 range. Two 5-block draws
(seeds 0 and 1) cannot separate an architecture effect from a seed effect.

One run resolves it: a third 5-block seed makes the arms 3-vs-3 on the same
protocol, against a measured reference scale.

## 2. HARD AUTHORISATION

```
blocks = 5      seed = 2      epochs = 200
```

**Exactly one training run. Nothing else is authorised**, specifically not:
4b/seed1, 4b/seed2, any other 5-block seed, any 6-block seed, seed 0, any
30-epoch run, any rerun of a completed run, any architecture modification.

Enforced in code, not only in prose: `BATCH2 = (("B", 2),)` in
`exp_blockcount_batch2.py`; `cmd_train` refuses anything outside it (inherited
guard) and `main()` refuses a mismatched `--arm`/`--seed` before dispatch.

Estimated cost from batch 1's measured 5-block throughput: **~1.34 GPU-hours**.

## 3. SCIENTIFIC QUESTION

Does the 5-block arm remain separated from the 6-block seed distribution when a
third seed is added?

This is a **resolution run**, not an architecture search.

## 4. FROZEN REFERENCES — READ-ONLY, NOT RETRAINED

40-scene pooled figures at epoch 200:

| Blocks | Seed | EPE | D1 | record |
| -----: | ---: | --: | -: | ------ |
| 6 | 0 | 2.2955182 | 15.6045143 | Stage B deterministic run A |
| 6 | 1 | 2.3982217 | 18.3357145 | batch 1 |
| 6 | 2 | 2.3950488 | 17.4699044 | batch 1 |
| 5 | 0 | 2.6457595 | 21.3711907 | deterministic block-count series arm B |
| 5 | 1 | 2.4729830 | 18.2061782 | batch 1 |
| 5 | 2 | **UNKNOWN** | **UNKNOWN** | this run |
| 4 | 0 | 2.3387143 | 16.9797389 | deterministic block-count series arm C |

All six existing records carry full per-scene arrays, so the paired per-scene
comparison in §12 is possible without retraining anything.

## 5. MODEL — ONLY `seed` IS NEW

5-block arm, dilations `1, 2, 4, 8, 1`. H2 standardised readout,
`cost_volume_shift = left`, 12 disparity candidates, same feature extractor,
cost volume, aggregation, refinement implementation, loss, Adam at lr 1e-3,
`hailo_calib` train / `hailo_val` val, same crop policy, same augmentation,
batch size 2, same initialisation procedure, native 200-epoch schedule
(`CosineAnnealingLR T_max=200`, neither shortened nor re-fitted).

The only new experimental state is `blocks = 5, seed = 2`. The 5-block
architecture itself is unchanged from the frozen series and from batch 1.

## 6. DETERMINISM

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Deterministic errors are **not**
suppressed. If deterministic execution fails, STOP and preserve the failure.

Batch 1 established that this protocol reproduces trajectories bit-for-bit
across processes, so the expected initial weight SHA for 5 blocks / seed 2 is
`540d81706f0d4850` (recorded by the seed screen at the same block count and
seed). It is recorded, not enforced.

## 7. INITIALISATION

Fresh, per the frozen recipe. **No checkpoint is loaded** — not Stage B, not any
6-block checkpoint, not 5b/seed0, not 5b/seed1, not any seed-screen checkpoint.
The initial weight SHA is recorded and asserted to differ from the seed-0
initialisation.

## 8. PRE-FLIGHT — BLOCKING

`preflight` must print `PREFLIGHT PASS`. `run_training` independently refuses to
start if `results/preflight.json` is missing or failed. Checks: Phase 1 diff
empty; deterministic controls enabled; seed = 2; block count = 5; dilations
`1,2,4,8,1` and matching registration; shift = left; standardised readout; 12
candidates; parameter count matches registration; 200 epochs; native `T_max`
200 and not re-fitted; fresh initialisation; no checkpoint loaded; optimizer
state fresh; surviving weights bit-identical within the seed; train split
`hailo_calib`; val split `hailo_val`; batch size 2; run count 1; batch is B2
only; seed 0 not scheduled; no 4-block scheduled; no 6-block scheduled.

**STOP on any mismatch.**

## 9. TRAINING

Exactly 200 epochs. No tuning, no LR change, no augmentation change, no schedule
change, no initialisation change, no extension past 200, and **no restart
because the trajectory looks unfavourable**. The frozen in-run gates stay active
and may end the run early — epoch-10 viability gate, epoch-100 collapse check.
**A gate-triggered abort is a legitimate result** and is analysed, not rerun.

## 10. RECORDING

Training loss, validation loss, EPE, D1, gradient median/max, NaN/Inf status,
matching-path gradient fraction, checkpoints, final weight hash, wall clock —
the same set as batch 1. Weight snapshots at epochs 10, 20, 50, 100, 150, 200.

## 11. STEREO PROBES

The established probes only (`exp_h2_seed_replication.stereo_probe`,
`exp_h2_softargmin_scale.validation_stage_stats`), imported unchanged, at
epochs 10, 20, 50, 100, 150, 200: right-image dependence, matching-map
dependence, initial-disparity correlation with GT, entropy, disparity standard
deviation. No new probe.

They establish binocular **dependence**, not correct disparity search.

## 12. EVALUATION AND COMPARISON

The full 40-scene `hailo_val` evaluation of the final checkpoint is the
**authoritative** metric: pooled EPE, pooled D1, RMSE, and per-scene EPE / D1 /
valid-pixel counts. The 10-scene training monitor is recorded but is **not** the
primary final metric — batch 1 measured that it overstates the 5-vs-6 gap by
roughly 5×.

Pre-specified analysis:

1. Both three-seed ranges reported with **mean, SD, min, max, range and every
   individual seed value**. Individual seeds are never hidden behind a mean.
2. `min(5-block EPE) > max(6-block EPE)` and, separately,
   `min(5-block D1) > max(6-block D1)` — descriptive non-overlap, not a test.
3. Whether the same ordering holds for **all nine** 5b×6b seed pairings.
4. Paired per-scene comparison (every model evaluated on the same 40 scenes).
5. The stereo probes.

## 13. NO POST-HOC THRESHOLD

The historical E3b/E3c materiality band stays **invalidated** and is not
resurrected. No new numerical threshold is invented after seeing the result. The
question is whether the two three-seed distributions overlap and how large the
observed architecture and seed variation are.

## 14. INTERPRETATION RULE — one of exactly three

- **`5-BLOCK-PENALTY-AT-SEEDS-0-1-2`** — all three 5-block seeds worse than all
  three 6-block seeds on the 40-scene metric. Supports a capacity-related effect
  *for this tested setup*.
- **`CAPACITY-EFFECT-INCONCLUSIVE-AT-SEEDS-0-1-2`** — the distributions overlap
  substantially. Do **not** claim 5 blocks are equivalent; do **not** claim 6
  blocks are superior.
- **`TRAJECTORY-SENSITIVE-AT-SEEDS-0-1-2`** — seed 2 substantially reverses the
  previous 5-block pattern. Explain the evidence, do not rerun seed 2, do not
  launch 4-block runs.

`RESULTS.md` separates **MEASURED / DERIVED / INFERRED / UNKNOWN**.

## 15. CLAIM CEILING — BINDING

> This result concerns the tested StereoNet configuration, deterministic
> protocol, three seeds, and evaluated dataset. It does not establish a
> universal law that more refinement blocks always improve stereo depth.

Three seeds per arm is evidence about this configuration, not a general law, and
not a statistical test.

## 16. THE 4-BLOCK ARM STAYS CLOSED

4b/seed1 and 4b/seed2 are **not** trained here. 4b/seed0 (2.3387) already lies
inside the observed 6-block range (2.2955–2.3982); the immediate priority is
whether the 5-block effect survives a third seed.

## 17. OUTPUT

```
phase2/factorial/block_count_full/20260910T052736Z/
  PREREGISTRATION.md  README.md  RESULTS.md  ENVIRONMENT.txt  CONFIG.json
  preflight.json  run_meta_*.json  train_B_seed2_stdout.log
  eval40_*.json  probes.json  comparison.json  checkpoints/  experiments/  results/
```

Batch 1's directory is never written. Nothing outside this directory is written.

## 18. PROVENANCE

`git rev-parse HEAD`, `git rev-parse phase-1-frozen^{commit}`,
`git status --short`, `git diff phase-1-frozen -- src scripts` recorded before
and after. **The Phase 1 diff must be empty at both ends.** Stage A, Stage B,
H2, E3b, E3c, O6, the deterministic block-count series, the seed screen and
batch 1 are read, never written.

## 19. HARD STOP

After the run, the 40-scene evaluation, the probes, the comparison and the
report: **STOP.** Do not launch 4b/seed1, 4b/seed2, another 5-block seed,
another 6-block seed, Scene Flow, dilation experiments, candidate-count
experiments, or Stage E. Recommend exactly one next experiment and do not
execute it. The research lead decides what happens next.
