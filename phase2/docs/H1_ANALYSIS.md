# H1 analysis — what is a working cost volume worth?

**Experiments:** `EXP-H1-BASE` (`cost_volume_shift="none"`, frozen Phase 1
degeneracy) vs `EXP-H1-WORKING` (`cost_volume_shift="left"`, genuine
correspondence search). Identical protocol otherwise: KITTI 2015
hailo_calib/hailo_val split, random 256x512 crop, Adam lr 1e-3 cosine anneal,
batch 2, seed 0, 20 epochs, from random initialisation — Phase 1 EXP-016's
recipe, reused byte-for-byte. GPU: RTX 4060, fp32.

Raw records: `phase2/experiments/EXP-H1-BASE/metrics.json`,
`phase2/experiments/EXP-H1-WORKING/metrics.json`.

## Mechanism check (prerequisite, done before training — MEASURED)

`phase2/tests/test_h1_cost_volume_shift.py`, 4/4 passing:
- `shift="none"` still reproduces Phase 1's exact degeneracy (all 12 slices
  bit-identical) — the two arms differ in nothing except the changed variable.
- `shift="left"` makes every one of the 11 non-zero candidate slices differ
  from slice 0, agrees with an independently-written indexed reference to
  1e-6, and the boundary (disparity candidates reading past the right image
  edge) is explicitly zero-padded as documented.

## Accuracy — MEASURED

| | BASE (none) | WORKING (left) | Delta |
|---|---:|---:|---:|
| Final training loss (epoch 19) | 8.4836 | 8.6169 | WORKING +0.133 (worse) |
| Val EPE, epoch 0 | 18.414 px | 17.214 px | WORKING −1.200 (better) |
| Val EPE, epoch 19 (final) | 19.191 px | 19.212 px | +0.021 (WORKING marginally worse) |
| Val EPE, best-seen | 17.413 px (ep 15) | 17.214 px (ep 0) | WORKING's "best" is its untrained start |
| Val D1, epoch 19 (final) | 88.83 % | 88.99 % | +0.16 pt (WORKING marginally worse) |
| Median grad norm | 31.795 | 26.845 | — |
| Max grad norm | 130,158 (one spike; no non-finite values) | 246 | BASE had one large but finite spike |

Both arms: **validation did not improve** over training, matching Phase 1's
own EXP-016 finding under the identical recipe. Final-epoch EPE and D1 are
statistically indistinguishable between BASE and WORKING (0.02 px EPE, 0.16
D1 points) — far smaller than the epoch-to-epoch noise within either arm's
own validation series (±3.5 px swings, e.g. BASE epoch 5's 20.959).

## Complexity — DERIVED

Enabling the shift changes zero architecture: same feature extractor,
aggregation, refinement; same parameter count (423,586 unique); same cost
volume construction cost (0 MACs either way — `shift_left` is padding +
slicing, not a matmul, exactly like Phase 1's degenerate `reference_shift`).
**A working cost volume costs nothing extra in params, MACs, or model size at
this architecture's current disparity-candidate count.** The only cost is a
handful of additional data-movement ops (pad + slice per candidate), already
present in the degenerate path too (EXP-001 recorded 48 data-movement nodes
for the no-op version).

## Runtime — MEASURED

| | BASE | WORKING |
|---|---:|---:|
| Wall clock, 20 epochs | ~375 s | ~354 s |
| Device | RTX 4060, cuda, fp32 | RTX 4060, cuda, fp32 |

No meaningful runtime difference (WORKING was marginally faster, within
scheduling noise — consistent with the DERIVED zero-MAC-delta finding above).

## Interpretation

**Result: no measurable accuracy delta from a genuinely working cost volume,
under this training budget.**

This is a real, recorded null result — not evidence that a working cost
volume has no value in general. It is confounded by the same limitation
Phase 1 named for EXP-016 and this experiment inherited on purpose (for exact
comparability): **160 scenes from random initialisation, 20 epochs, is a
convergence-pipeline budget, not enough data for either arm's validation
curve to leave the noise floor.** Neither BASE nor WORKING improved
validation at all — the refinement network in both arms is still near its
random-init behaviour by epoch 19. A signal that a working correspondence
volume provides cannot be detected by a training regime that has not yet
taught the network to use *any* signal, including a broken one.

**Classification (per PHASE_2_TASK.md §27):**
- **MEASURED:** under this exact 160-scene/20-epoch/seed-0 protocol, final
  validation EPE/D1 are statistically indistinguishable between the two
  cost-volume conditions.
- **DERIVED:** the working cost volume adds zero parameters, zero MACs, and
  no measurable latency cost versus the degenerate one.
- **INFERRED:** this training budget is too small to detect whatever
  accuracy effect a working cost volume has, because neither arm's
  validation improved from initialisation at all — the experiment lacks the
  statistical power to answer H1, not evidence that H1's hypothesis is false.
- **UNKNOWN:** what a working cost volume is worth under a training budget
  large enough for the refinement network to actually converge (Scene Flow
  pretraining, deferred by Phase 1's own charter — research_questions.md O6).

## Decision

**H1 is inconclusive at this training scale — neither keep nor reject the
working cost volume on this evidence.** The controlled experiment ran
correctly (mechanism verified pre-training by regression test, both arms
identical except the changed variable, both wall-clock timed on the same
GPU) and produced a real null result worth recording, but the null result's
cause is confounded with inadequate training signal, not disambiguated from
"the shift genuinely doesn't matter."

**Next step, in order (does not skip to architecture design):**
1. Re-run H1 with a training budget large enough for validation to actually
   move off initialisation for at least one arm — either more epochs on the
   same 160 scenes, or a larger scene count, before concluding anything about
   the shift's value. This is a data/compute-budget experiment, not an
   architecture experiment, and should be run as `EXP-H1-BASE-v2` /
   `EXP-H1-WORKING-v2` rather than overwriting these records.
2. Only once a training regime exists where at least the frozen baseline
   (`shift="none"`) shows real validation improvement should the two arms be
   compared again for H1's accuracy question. Comparing under a regime where
   nothing improves cannot distinguish "shift doesn't help" from "nothing
   would help at this budget."
3. Do not proceed to Stage E (cost-volume optimization: candidate count,
   representation, hierarchical search) until H1 produces a decisive
   (not merely null) result, per the task's own Stage order.

Both `EXP-H1-BASE` and `EXP-H1-WORKING` records are preserved unmodified as
the first, budget-limited attempt at H1 — not deleted, not overwritten by any
v2 rerun.
