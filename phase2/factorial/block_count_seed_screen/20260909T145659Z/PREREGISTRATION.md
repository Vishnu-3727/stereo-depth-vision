# EXP-BLOCKCOUNT-SEEDSCREEN-001 — pre-registration

**Frozen before any run in this directory was launched.** Nothing below was
edited after the first training process started.

Record: `phase2/factorial/block_count_seed_screen/20260909T145659Z/`
Script: `phase2/factorial/block_count_seed_screen/exp_blockcount_seed_screen.py`

---

## 1. Why this exists

The deterministic block-count series at seed 0
(`EXP-BLOCKCOUNT-DETERMINISTIC-001`, `20260909T131500Z`) produced, on the
40-scene evaluation at epoch 200:

| blocks | EPE | D1 |
|---|---:|---:|
| 6 | 2.2955 | 15.6045 |
| 5 | 2.6458 | 21.3712 |
| 4 | 2.3387 | 16.9797 |

The ordering is **non-monotone in block count** (`6 < 5 > 4` on EPE), so the
differences at one seed cannot be attributed to capacity. Settling it properly
means a 6-run × 200-epoch campaign at roughly **8 GPU-hours**.

This experiment does **not** attempt to settle it. It is a **cheap screening
experiment**: six 30-epoch runs (~1.5 GPU-hours) whose only job is to decide
whether the 8-hour campaign is scientifically justified.

## 2. Question

> At seeds 1 and 2, do the 6-, 5- and 4-block configurations already show
> meaningful trajectory variation or changing architecture ordering within the
> first 30 epochs?

## 3. Matrix — exactly six runs

| Seed | 6 blocks (arm A) | 5 blocks (arm B) | 4 blocks (arm C) |
|---|---|---|---|
| 1 | 30 epochs | 30 epochs | 30 epochs |
| 2 | 30 epochs | 30 epochs | 30 epochs |

Seed 0 is **not** rerun. No configuration is run for 200 epochs. No seventh run
is added for any reason.

## 4. Frozen control and frozen history

The Stage B 6-block seed-0 checkpoint remains the authoritative 6-block control
for the completed series and is **not retrained**. The seed-0 5-block and
4-block results are historical measurements and are **not overwritten**.

This experiment is **additive only**. It reads the seed-0 records to quote their
epoch-10/20/30 validation points as context; it writes nothing outside its own
timestamped directory.

## 5. The only variables

```
block count ∈ {6, 5, 4}
seed        ∈ {1, 2}
```

Everything else is frozen and imported unchanged from the Stage B / E3b recipe:
H2 standardised readout, `cost_volume_shift = left`, 12 disparity candidates,
the same feature extractor, cost volume, aggregation, refinement implementation,
loss (masked smooth L1, β=1.0), optimizer (Adam, lr 1e-3, β=(0.9, 0.999)),
learning-rate schedule, dataset (KITTI 2015, `hailo_calib` train / `hailo_val`
validation), crop (256×512), augmentation (random crop, per-image gain jitter
σ=0.1, no flip), batch size 2, fp32, and the initialisation procedure.

Deterministic protocol, mandatory:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

```
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### 5.1 The epoch budget is 30; the LR schedule stays at T_max = 200

This is the one non-obvious protocol decision and it is registered here, in
advance, because it is a judgement call.

The training recipe uses `CosineAnnealingLR(T_max=EPOCHS)`. Shortening the
budget to 30 epochs would, if taken naively, also re-fit the cosine to 30
epochs — which changes the learning rate at **every** epoch relative to the
frozen recipe.

That is rejected, for two reasons:

1. The design freezes the learning-rate schedule and forbids changing the
   learning rate. Re-fitting `T_max` changes it.
2. A screen that forecasts a 200-epoch experiment must be a **prefix** of that
   experiment. With `T_max = 30` these six runs would be fully-annealed
   30-epoch trainings — a different recipe — and their trajectories would say
   nothing about the 200-epoch trajectories they exist to screen.

Therefore: **budget 30 epochs, schedule `T_max = 200`, unchanged.** Each run is
a strict prefix of the 200-epoch run it screens for. This also makes the seed-0
epoch-10/20/30 points from the existing 200-epoch records directly comparable,
at zero additional cost.

Enforced in code by `FrozenCosineAnnealingLR` and asserted by the preflight
check `schedule_T_max_frozen_at_200`.

## 6. Initialisation

Fresh initialisation per the established recipe (`build_model(seed)` →
`torch.manual_seed` / `np.random.seed`, then the standardised readout).

**No run is initialised from the Stage B trained checkpoint.** The optimizer and
scheduler are constructed fresh. The initial weight SHA of every run is
recorded. Seed-1 and seed-2 hashes will differ from seed 0's
(`c4d02385e3da28a5`) and are **not** forced to match it — the preflight asserts
that they differ.

## 7. Pre-flight — blocking

Before the first run, the preflight must show all of:

```
phase 1 diff vs phase-1-frozen        = empty
deterministic controls                = enabled (all four)
cost_volume_shift                     = left
readout                               = StandardisedDisparityRegression
disparity candidates                  = 12
block counts                          = 6 / 5 / 4 as registered
dilations                             = 1-2-4-8-1-1 / 1-2-4-8-1 / 1-2-4-8
parameter counts                      = 423586 / 405090 / 386594
seeds                                 = {1, 2} only; seed 0 not scheduled
epoch budget                          = 30 (and ≠ 200)
LR schedule T_max                     = 200 (frozen, not re-fitted)
initialisation                        = fresh; differs from the seed-0 init sha
optimizer state                       = empty at construction
run count                             = 6
```

If any check fails, **STOP** — nothing is trained.

A run is also aborted before training if its parameter count is wrong, its
initial weights match seed 0's, or the epoch budget is not 30.

## 8. Training

Exactly 30 epochs per run. No run is extended. No run is rerun because its
result is unfavourable. No tuning, no learning-rate change, no augmentation
change, no dataset change. **Failures are preserved**, including a run that
trips the frozen epoch-10 viability gate, which stays enforced exactly as in the
imported recipe.

## 9. Checkpoints

The project's existing checkpoint mechanism, with weight snapshots at epochs
**10, 20, 30** and the richer checkpoint record at epochs 1, 2, 5, 10, 20, 30.
No new architecture is created.

## 10. Metrics recorded

Validation EPE and D1 at epochs 10, 20 and 30 (the recorded protocol: first 10
scenes of `hailo_val`, full 368×1232 frames, pooled over `gt > 0`), plus
training loss, gradient median/max, NaN/Inf status, and matching-path gradient
percentage.

## 11. Stereo probes

The established probes only — `exp_h2_seed_replication.stereo_probe` and
`exp_h2_softargmin_scale.validation_stage_stats`, imported unchanged — at epochs
10, 20 and 30: right-image dependence, matching-map dependence,
disparity-initial correlation with GT, softmax entropy, disparity standard
deviation. **No new stereo metric is introduced.**

These establish binocular *dependence*, not correct disparity search (Phase 1
EXP-007 measured +89.7 D1 on the reference weights despite a provably degenerate
shift). They are supporting evidence only.

## 12. Evaluation

The project's established validation protocol during training. At epoch 30 each
of the six runs is additionally evaluated on the **full 40-scene** `hailo_val`
set with the existing evaluator, unchanged — it is cheap and requires no new
procedure. The 30-epoch screen remains primarily a trajectory diagnostic.

## 13. Analysis, fixed in advance

1. The six-run matrix: EPE and D1 at epochs 10, 20, 30.
2. **Architecture ordering** — best → middle → worst on EPE and on D1, at each
   checkpoint, for each seed. Monotonicity in block count is **not** assumed.
3. **Seed effect** — seed 1 vs seed 2 at each block count and checkpoint.
4. **Ordering consistency** — whether any ordering holds across seeds and
   checkpoints.
5. The architecture spread within a seed, alongside the seed spread at fixed
   architecture, so the two can be compared on the same scale.

## 14. Claim ceiling — binding

A 30-epoch screen cannot establish the 200-epoch architecture ranking. The
report will not state "5 blocks are required", "4 blocks are equivalent" or
"6 blocks are superior" except explicitly scoped to the 30-epoch screen.

## 15. Verdict rule — exactly one

- **CONTINUE-FULL** — seeds 1 and 2 show materially different trajectories, or
  the block ordering changes across seeds/checkpoints, or the architectures are
  separating in a way that makes a multi-seed 200-epoch experiment informative.
  Recommend the smallest sufficient full experiment.
- **REDUCE-SCOPE** — all six trajectories are very similar, ordering is stable,
  seed effects are small, and a smaller targeted experiment could answer the
  question. Recommend it. Do **not** launch it.
- **INCONCLUSIVE-SCREEN** — 30 epochs is too early to distinguish the
  trajectories, metrics stay close, no reliable ordering emerges. Recommend
  exactly one next diagnostic.

## 16. No post-hoc threshold

**No numerical materiality threshold is defined, before or after seeing the
results.** The historical E3b/E3c band (0.2138 px / 1.0000 pt) is invalidated
and is neither reused nor replaced. The screen is qualitative: it compares the
*seed* spread against the *architecture* spread on the same measured scale and
reports what it sees, without declaring either "significant".

## 17. Outputs

`PREREGISTRATION.md` (this file), `README.md`, `RESULTS.md`, `ENVIRONMENT.txt`,
`CONFIG.json`, `results/preflight.json`, per-run `run_meta_*.json`,
`experiments/…/{config,env,log,metrics}.json`, `eval40_*.json`, `probes.json`,
`comparison.json`, `checkpoints/` (final + epoch 10/20/30 snapshots per run),
and the per-run stdout logs.

## 18. Provenance

`git rev-parse HEAD`, `git rev-parse phase-1-frozen^{commit}`,
`git status --short` and `git diff phase-1-frozen -- src scripts` are recorded
before and after. **The Phase 1 diff must be empty at both ends.** Historical
E3b / E3c / O6 / H2 records are not modified.

## 19. Hard stop

After the six 30-epoch runs and the analysis: **STOP.** No 200-epoch launch, no
automatic continuation, no third seed, no architecture change, no Stage E, no
H2 rerun, no E3b/E3c rerun, no dilation experiment. The research lead decides
whether the full experiment happens.
