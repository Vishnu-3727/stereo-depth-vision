# EXP-BLOCKCOUNT-SEEDSCREEN-001 — results

Record `phase2/factorial/block_count_seed_screen/20260909T145659Z/`.
Pre-registered in `PREREGISTRATION.md`, frozen before the first run.

Six 30-epoch runs, all completed, exit 0, no aborts, no NaN/Inf.
Total training wall clock **4582.2 s (1.27 GPU-hours)**.

---

## QUESTION

> Do seeds 1 and 2 reveal trajectory variation sufficient to justify a full
> multi-seed block-count experiment?

---

## MATRIX

Validation during training — the recorded protocol: first 10 scenes of
`hailo_val`, full 368×1232 frames, pooled over `gt > 0`.

| Blocks | Seed | EPE @10 | EPE @20 | EPE @30 | D1 @10 | D1 @20 | D1 @30 |
| ------ | ---: | ------: | ------: | ------: | -----: | -----: | -----: |
| 6      |    1 | 16.3545 | 11.5311 |  8.0942 | 86.713 | 77.132 | 64.682 |
| 5      |    1 | 16.5181 | 11.7946 |  7.3619 | 87.153 | 78.857 | 55.682 |
| 4      |    1 | 18.3653 | 12.0788 |  8.7907 | 88.283 | 78.736 | 64.468 |
| 6      |    2 | 15.3428 | 13.2647 |  7.7840 | 86.422 | 83.311 | 53.142 |
| 5      |    2 | 15.6977 | 12.3364 |  8.4022 | 86.584 | 78.760 | 59.401 |
| 4      |    2 | 16.3429 | 15.5063 | 10.7875 | 87.356 | 85.642 | 77.331 |

### Seed 0 — read-only context, not retrained

Read out of the frozen records (Stage B deterministic run A for 6 blocks;
`EXP-BLOCKCOUNT-DETERMINISTIC-001` arms B and C for 5 and 4), same validation
protocol, same recipe. A field-by-field config comparison confirms these records
differ from the screen's runs only in `seed`, `refinement_blocks` and the epoch
budget — architecture, shift, candidates, dataset, splits, crop, augmentation,
optimizer, learning rate, `CosineAnnealingLR T_max=200`, loss, batch size and
precision are identical strings.

| Blocks | Seed | EPE @10 | EPE @20 | EPE @30 | D1 @10 | D1 @20 | D1 @30 |
| ------ | ---: | ------: | ------: | ------: | -----: | -----: | -----: |
| 6      |    0 |  9.0015 |  7.3321 |  7.1893 | 64.552 | 51.106 | 51.880 |
| 5      |    0 | 15.4927 | 17.5082 | 13.0777 | 85.795 | 88.211 | 82.066 |
| 4      |    0 |  9.3812 |  6.9404 |  6.4530 | 68.229 | 53.099 | 46.005 |

### 40-scene evaluation at epoch 30

All 40 `hailo_val` scenes, full frames, pooled over `gt > 0` — the existing
evaluator, unchanged. Run for all six screen runs; it required no new procedure
and cost ~17 s each.

| Blocks | Seed |    EPE |      D1 |    RMSE |
| ------ | ---: | -----: | ------: | ------: |
| 6      |    1 | 5.5463 | 53.9880 |  9.7367 |
| 5      |    1 | 4.7466 | 43.4034 |  9.0397 |
| 4      |    1 | 6.0233 | 55.9166 | 10.7989 |
| 6      |    2 | 4.8990 | 42.0544 | 10.0797 |
| 5      |    2 | 5.3959 | 50.9266 | 10.4335 |
| 4      |    2 | 7.8345 | 69.7483 | 12.3655 |

The 40-scene ordering at epoch 30 agrees with the 10-scene ordering at epoch 30
for both seeds, so the orderings below are not an artefact of the 10-scene
subset.

For scale only, and **not** as a comparison: the frozen seed-0 runs reached
40-scene EPE 2.2955 / 2.6458 / 2.3387 at epoch **200**. Every run in this screen
is between 4.75 and 7.83 at epoch 30 — these models are nowhere near their
end state.

---

## TRAJECTORY

**All six runs are still descending steeply at epoch 30. Nothing has plateaued.**

Every run shares the same two-phase shape: a noisy near-plateau through roughly
epoch 10–15, then a steep descent that has not finished by epoch 30. What differs
between runs is *when* the descent starts and *how fast* it goes.

Full validation-EPE series (epochs 1, 2, 5, 10, 15, 20, 25, 30):

| Blocks | Seed | 1 | 2 | 5 | 10 | 15 | 20 | 25 | 30 | drop 10→30 |
| ------ | ---: | --: | --: | --: | --: | --: | --: | --: | --: | ---: |
| 6 | 1 | 15.91 | 15.82 | 18.66 | 16.35 | 15.72 | 11.53 |  9.26 |  8.09 | −50 % |
| 5 | 1 | 15.66 | 15.96 | 17.90 | 16.52 | **18.42** | 11.79 |  8.70 |  7.36 | −55 % |
| 4 | 1 | 16.03 | 18.95 | 18.34 | 18.37 | 17.97 | 12.08 | 11.76 |  8.79 | −52 % |
| 6 | 2 | 20.77 | 15.40 | 17.92 | 15.34 | 13.90 | 13.26 |  9.07 |  7.78 | −49 % |
| 5 | 2 | 21.93 | 18.08 | 16.97 | 15.70 | 14.19 | 12.34 | 10.40 |  8.40 | −46 % |
| 4 | 2 | 21.45 | 17.85 | 17.40 | 16.34 | **16.93** | 15.51 | 13.01 | 10.79 | −34 % |

- **5 blocks, seed 1** *rises* from 16.52 to 18.42 between epochs 10 and 15 —
  the worst run in the screen at epoch 15 — and then falls faster than anything
  else, ending best at epoch 30 (7.36). Its ordering rank inverts inside 15
  epochs.
- **4 blocks, seed 2** also rises between 10 and 15, descends latest and
  slowest, and is the only run still above 10 px at epoch 30.
- **5 blocks, seed 2** is the only run of the six whose validation EPE never
  increases at any recorded point.
- Training loss is **not** monotone in any run: every run has 6–10 epoch-to-epoch
  increases out of 29 steps, which is ordinary batch noise at batch size 2. The
  trend is downward in all six; the loss at epoch 30 is 3.30–4.02 except for
  4 blocks / seed 2 at 5.57.

The steep phase coincides with the emergence of binocular dependence (see
*STEREO FUNCTIONALITY*): the right-image probe is negative in all six runs at
epoch 10 and strongly positive in five of six by epoch 30. The screen window
straddles a phase change rather than sampling a settled regime.

The seed-0 records show a different shape again at the same block counts — see
below.

The seed-0 records show a qualitatively different pattern again: 6 and 4 blocks
were already at EPE 9.00 / 9.38 by epoch 10 — better than *any* seed-1 or seed-2
run at epoch **30** — while 5 blocks at seed 0 went 15.49 → 17.51 → 13.08, i.e.
**got worse between epoch 10 and 20**, and still ended at 40-scene EPE 2.6458 by
epoch 200.

Gradients, all six runs: median total norm 30.1–37.7, max 224.9–619.0, **no NaN
or Inf anywhere**, matching-path gradient present in **100.00 %** of batches.
The frozen epoch-10 viability gate **passed in all six runs**.

---

## SEED EFFECT

Absolute seed-1 vs seed-2 difference at fixed block count:

| Blocks | ΔEPE @10 | ΔEPE @20 | ΔEPE @30 | ΔD1 @10 | ΔD1 @20 | ΔD1 @30 |
| ------ | -------: | -------: | -------: | ------: | ------: | ------: |
| 6      |    1.012 |    1.734 |    0.310 |   0.292 |   6.179 |  11.540 |
| 5      |    0.820 |    0.542 |    1.040 |   0.568 |   0.097 |   3.720 |
| 4      |    2.022 |    3.427 |    1.997 |   0.927 |   6.906 |  12.863 |

Architecture spread (max − min across 6/5/4 blocks) **within** a seed, for
comparison on the same scale:

| Seed | EPE range @10 | @20 | @30 | D1 range @10 | @20 | @30 |
| ---- | ------------: | --: | --: | -----------: | --: | --: |
| 1    | 2.011 | 0.548 | 1.429 | 1.570 | 1.725 |  9.000 |
| 2    | 1.000 | 3.170 | 3.004 | 0.934 | 6.882 | 24.189 |

**The seed spread and the architecture spread are the same order of magnitude
at every checkpoint.** At 4 blocks the seed-1/seed-2 EPE gap at epoch 20 (3.427)
is larger than the entire 6-vs-5-vs-4 spread within seed 1 at any checkpoint.

Bringing seed 0 in makes it starker. At **fixed** architecture (6 blocks), epoch
10 validation EPE across the three seeds is 9.0015 / 16.3545 / 15.3428 — a spread
of **7.353 px** from the seed alone, against a largest measured architecture
spread of 3.170 px. Same for 4 blocks: 9.3812 / 18.3653 / 16.3429.

No threshold is applied to any of these numbers. They are reported as measured,
and the comparison being drawn is between two spreads on the same scale, not
against a band.

---

## ARCHITECTURE ORDERING

Best → worst. Screen seeds 1 and 2 measured here; seed 0 read from the frozen
records.

**Validation EPE**

| Epoch | Seed 0 | Seed 1 | Seed 2 |
| ----- | ------ | ------ | ------ |
| 10 | 6b, 4b, 5b | 6b, 5b, 4b | 6b, 5b, 4b |
| 20 | 4b, 6b, 5b | 6b, 5b, 4b | 5b, 6b, 4b |
| 30 | 4b, 6b, 5b | **5b, 6b, 4b** | **6b, 5b, 4b** |

**Validation D1**

| Epoch | Seed 0 | Seed 1 | Seed 2 |
| ----- | ------ | ------ | ------ |
| 10 | 6b, 4b, 5b | 6b, 5b, 4b | 6b, 5b, 4b |
| 20 | 6b, 4b, 5b | 6b, 4b, 5b | 5b, 6b, 4b |
| 30 | 4b, 6b, 5b | **5b, 4b, 6b** | **6b, 5b, 4b** |

**40-scene EPE at epoch 30:** seed 1 → 5b, 6b, 4b; seed 2 → 6b, 5b, 4b.

Findings:

1. **The ordering is not stable across seeds.** At epoch 30 the three seeds give
   three different EPE orderings. Five blocks is the *worst* arm at seed 0 (EPE
   13.08 vs 6.45 and 7.19) and the *best* arm at seed 1 (7.36). Four blocks is
   the *best* arm at seed 0 and the *worst* arm at both seed 1 and seed 2.
2. **The ordering is not stable across checkpoints within a seed either.** Seed
   1 orders 6b, 5b, 4b at epochs 10 and 20 and then 5b, 6b, 4b at epoch 30. Seed
   2 orders 6b, 5b, 4b at 10, 5b, 6b, 4b at 20, and 6b, 5b, 4b again at 30.
3. **EPE and D1 do not always agree** even at the same seed and epoch: at seed 1,
   epoch 30, EPE says 5b, 6b, 4b while D1 says 5b, 4b, 6b.
4. The only regularity anywhere in the table: **6 blocks is never last** — it is
   first or second in all nine seed × epoch cells. That is a weak observation
   from three seeds and is not a capacity claim.
5. **Monotonicity in block count does not hold**, and was not assumed. No
   checkpoint at any seed produced `6 ≥ 5 ≥ 4` or its reverse on both metrics.

---

## STEREO FUNCTIONALITY

The established probes only (`exp_h2_seed_replication.stereo_probe`,
`exp_h2_softargmin_scale.validation_stage_stats`), imported unchanged; no new
stereo metric was introduced.

`right` = minimum D1 penalty over the three right-image corruptions;
`map` = minimum D1 penalty over the two matching-map corruptions;
`ent` = softmax entropy (nats); `std` = final disparity standard deviation (px);
`r(GT)` = correlation of the *initial* disparity with ground truth.

| Blocks | Seed | Ep | right | map | ent | std | r(GT) |
| ------ | ---: | -: | ----: | --: | --: | --: | ----: |
| 6 | 1 | 10 |  −3.576 |  −1.883 | 2.093 |  8.396 | +0.633 |
| 6 | 1 | 20 | +12.595 | +10.218 | 1.813 | 12.520 | +0.672 |
| 6 | 1 | 30 | +31.963 | +25.077 | 1.783 | 15.044 | +0.823 |
| 5 | 1 | 10 |  −2.089 |  +2.168 | 2.085 |  7.944 | +0.587 |
| 5 | 1 | 20 | +13.473 | +15.000 | 1.997 | 11.975 | +0.680 |
| 5 | 1 | 30 | +39.957 | +28.887 | 1.824 | 16.048 | +0.814 |
| 4 | 1 | 10 |  −7.462 |  −3.525 | 2.067 |  8.117 | +0.436 |
| 4 | 1 | 20 |  +5.353 |  +7.199 | 1.923 | 12.162 | +0.639 |
| 4 | 1 | 30 | +37.624 | +34.821 | 1.821 | 14.438 | +0.773 |
| 6 | 2 | 10 |  −3.824 |  −4.320 | 2.051 | 10.339 | +0.586 |
| 6 | 2 | 20 |  −4.030 |  +6.875 | 1.877 | 11.036 | +0.687 |
| 6 | 2 | 30 | +46.388 | +43.213 | 1.820 | 15.350 | +0.769 |
| 5 | 2 | 10 |  −2.651 |  −4.088 | 1.973 |  8.776 | +0.518 |
| 5 | 2 | 20 |  −0.026 |  +3.568 | 1.865 | 10.912 | +0.709 |
| 5 | 2 | 30 | +37.962 | +34.911 | 1.719 | 16.001 | +0.733 |
| 4 | 2 | 10 |  −3.864 |  −7.034 | 2.065 |  9.438 | +0.490 |
| 4 | 2 | 20 |  −3.195 |  −1.564 | 2.114 |  9.323 | +0.537 |
| 4 | 2 | 30 |  +5.816 |  +7.784 | 1.959 | 13.129 | +0.689 |

Observations:

- **Binocular dependence emerges inside the screen window**, between epochs 10
  and 30. Right-image dependence is *negative* — corrupting the right image made
  D1 slightly better — in **all six runs at epoch 10** and in three of six at
  epoch 20. This reproduces the EXP-H2-EMERGENCE-001 finding that early epochs
  are not a valid place to judge stereo behaviour, and is why the epoch-10 gate
  does not enforce a stereo criterion.
- **The emergence epoch is itself seed- and architecture-dependent.** At epoch 30
  the right-image penalty spans +5.8 to +46.4 — an eightfold range across six
  runs of the same recipe.
- **Five of six runs meet the imported 20-point stereo criterion at epoch 30.**
  The exception is **4 blocks, seed 2** (+5.8), the same run that descended
  latest and slowest on validation EPE. This is reported as a fact about that run at epoch 30. It is **not**
  evidence that 4 blocks cannot become stereo-functional: 4 blocks at seed 1
  reached +37.6 at the same epoch, and the criterion is calibrated for epoch 200,
  not epoch 30.
- **No collapse anywhere.** Softmax entropy 1.72–2.11 nats (floor 0.5), final
  disparity std 8.1–16.0 px (floor 1.0) in every run at every probe epoch.
- Initial-disparity correlation with GT rises monotonically in every run
  (+0.44…+0.63 at epoch 10 → +0.69…+0.82 at epoch 30), so the matching path is
  carrying real signal in all six.

**Caution, restated:** these probes establish binocular *dependence*, not correct
disparity search. Phase 1 EXP-007 measured a +89.7 D1 right-image penalty on the
reference weights despite a provably degenerate shift. They are supporting
evidence, nothing more.

---

## VERDICT

```
CONTINUE-FULL
```

Under the pre-registered rule (§15), CONTINUE-FULL applies when seeds show
materially different trajectories **or** the block ordering changes across
seeds/checkpoints. Both hold, and not marginally:

- Three seeds produce **three different epoch-30 EPE orderings**, with 5 blocks
  moving from worst to best and 4 blocks from best to worst depending only on the
  seed.
- The ordering also flips **between checkpoints within a single seed**.
- The seed-to-seed spread at fixed architecture is the **same order as, and at 4
  blocks larger than**, the architecture spread within a seed; including seed 0
  it is more than twice as large.
- Two of six runs (5 blocks / seed 1 and 4 blocks / seed 2) *increased* their
  validation EPE between epochs 10 and 15 and then diverged in opposite
  directions — one finished best of the screen, the other worst.
- One run had not yet become stereo-functional at epoch 30 while its
  same-architecture sibling at the other seed was at +37.6.

What this verdict does and does not mean:

- It means the seed-0 non-monotonicity that motivated this screen is **not
  usable as evidence about capacity**; a single seed cannot rank these three
  architectures, at 30 epochs or at 200.
- It means a multi-seed 200-epoch experiment is the only instrument that could
  answer the capacity question, and that no cheap proxy substitutes for it.
- It also means a naive 6-run (3 arms × 2 seeds) 200-epoch campaign is **at risk
  of being underpowered**: at the spreads measured here, two seeds per arm would
  not separate an architecture effect from a seed effect. That is why the
  recommended next experiment below measures the reference scale first rather
  than launching the campaign.
- It does **not** mean the architectures differ in capacity. Nothing here
  measures that.

---

## CLAIM CEILING

> **This 30-epoch screen does not establish the final 200-epoch capacity
> ranking.**

Specifically, nothing in this record supports "5 blocks are required", "4 blocks
are equivalent" or "6 blocks are superior". Every ordering statement above is
scoped to the 30-epoch screen at a named seed and checkpoint. The models are far
from their end state at epoch 30 (40-scene EPE 4.75–7.83 here versus 2.30–2.65
for the frozen seed-0 runs at epoch 200), binocular dependence only emerged
during the screen window, and the seed-0 record demonstrates that a run can be
the *worst* arm at epoch 30 (5 blocks, EPE 13.08) and still finish within 0.35 px
of the best at epoch 200.

No materiality threshold was defined, before or after seeing the results. The
invalidated E3b/E3c band was neither reused nor replaced.

---

## NEXT EXPERIMENT

Exactly one, and it is **not** executed here.

**Measure the 200-epoch seed-noise scale at a fixed architecture: 6 blocks,
seeds 1 and 2, 200 epochs, deterministic protocol. Two new runs (~2.7
GPU-hours), because seed 0 already exists.**

Why this one:

- It is the smallest experiment that makes anything else interpretable. Every
  block-count comparison at 200 epochs is currently being read against *no*
  reference scale — the E3b/E3c band that used to serve that role is invalidated,
  and the Stage B bitwise-reproducibility result measured *execution* noise (zero)
  rather than seed noise.
- It yields a three-seed distribution at 6 blocks against which the **already
  frozen** seed-0 5-block and 4-block results can be read immediately, with no
  further training.
- It is decisive in both directions. If the 200-epoch seed spread turns out
  small — plausible, since all three seed-0 arms converged into a 0.35 px band by
  epoch 200 despite wildly different 30-epoch trajectories — then the existing
  frozen series may already answer the capacity question and the remaining four
  arms are unnecessary. If it is large, the full campaign is known in advance to
  need more than two seeds per arm, and running the 6-run version would have
  wasted 8 GPU-hours on an unresolvable comparison.

Practical note for costing it: the 30-epoch runs in this screen **cannot be
extended**. The checkpoint mechanism stores `{"model", "epoch", "config"}` only —
no optimizer or scheduler state — so the recommended runs are full 200-epoch
runs from fresh initialisation. (Snapshotting optimizer state would make future
screens extendable; that is an observation, not part of the recommendation.)

---

## HARD STOP

Stopping here, per §19 of the pre-registration. No 200-epoch run was launched, no
run was continued, no third seed was added, no architecture was changed, Stage E
stays CLOSED, and H2 / E3b / E3c / O6 / dilation work was not touched. The
research lead decides whether the recommended experiment happens.

---

## PROVENANCE

`HEAD` `58e8a19`; `phase-1-frozen^{commit}` `b4207e5`;
`git diff phase-1-frozen -- src scripts` **empty before and after**;
working tree `M .gitignore`, `?? phase2/` at both ends.
Preflight: **18/18 checks PASS** (`results/preflight.json`).

Phase 2 is not committed to git; no external immutability is claimed for any
Phase 2 record, including this one.

Historical E3b / E3c / O6 / H2 / Stage A / Stage B records and the deterministic
block-count series were read but never written. All output is confined to this
timestamped directory.
