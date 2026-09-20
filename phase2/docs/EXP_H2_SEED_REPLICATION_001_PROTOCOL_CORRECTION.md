# EXP-H2-SEED-REPLICATION-001 — protocol correction, frozen before the reruns

**Status: pre-registration.** Everything in this document was written and frozen
*before* seeds 1 and 2 were relaunched, and before any 200-epoch result from a
replication seed existed. It exists because the first attempt used an invalid
gate, and the correction must not be shaped by the results it will judge.

**Claim tags:** `MEASURED`, `DERIVED`, `INFERRED`, `UNKNOWN`.

---

## 1. What went wrong

The first attempt gated each seed at epoch 10 on three criteria: validation
improvement, matching-path gradient presence, and **right-image dependence ≥ 5.0
D1 points**. Seeds 1 and 2 passed the first two and failed the third, and both
runs aborted at epoch 10 as designed.

`MEASURED` — the third criterion was never calibrated. The reference run
(`EXP-H2-SOFTARGMIN-SCALE`, seed 0) measured its +80.8…+84.4 D1 right-image
dependence at **200** epochs; no 10-epoch measurement existed. That gap was
already recorded as a known limitation in
`phase2/docs/EXP_H3_VIABILITY_001_REPORT.md` §7 before this experiment began.

`MEASURED` — the missing measurement, taken afterwards as
`EXP-H2-SEED-REPLICATION-001-SEED0-REFERENCE` (10 epochs, gate recorded but not
enforced):

| at epoch 10 | seed 0 (reference) | seed 1 | seed 2 |
|---|---:|---:|---:|
| val EPE, initialisation → epoch 10 | 31.96 → 15.24 | 35.28 → 15.76 | 35.16 → 14.81 |
| matching-path gradient | 100 % | 100 % | 100 % |
| right-image ΔD1 (worst variant) | **−3.33** | **−1.45** | **−2.71** |
| matching-map ΔD1 (worst substitution) | **−5.97** | +1.36 | −1.61 |
| would the epoch-10 gate pass? | **no** | no | no |

`DERIVED` — **the gate was invalid.** The reference run — the one that reaches
+81…+84 D1 points of stereo dependence by epoch 200 — fails the same gate at
epoch 10. Stereo dependence is a **late-emerging property** of this recipe, not a
property visible at 10 epochs. Seeds 1 and 2 were stopped by a mis-calibrated
instrument, and their aborts say nothing about whether they replicate H2.

`INFERRED` — this does not make seed 0 a failure and does not weaken H2's
200-epoch result. It changes *when* stereo dependence can be tested.

## 2. Frozen decisions

1. The aborted records `EXP-H2-SEED-REPLICATION-001-SEED1` and `-SEED2` are
   **historical and untouched**. No file inside them is edited; each receives an
   additive `NOTE.md` pointing here.
2. Every rerun gets a **new experiment ID** (`…-SEED1-RUN2`, `…-SEED2-RUN2`).
3. `EXP-H2-SOFTARGMIN-SCALE` (seed 0, 200 epochs) remains the **primary
   reference**. It is not retrained and not replaced.
4. `EXP-H2-SEED-REPLICATION-001-SEED0-REFERENCE` (10 epochs) is an
   **infrastructure/control measurement only**. It is never quoted as seed 0's
   result, and its own 200-epoch behaviour is unknown because it was stopped at
   10 by design.
5. The epoch-10 stereo gate is **recorded as invalidated by control evidence**,
   in this document and in the affected records.
6. The 200-epoch stereo criterion is defined in §4 **now**, before any new seed
   result exists.
7. Same-seed run-to-run variance is measured (§3) and bounds every seed-to-seed
   claim this experiment makes.

## 3. Same-seed reproducibility noise — `MEASURED`

Two independent questions were asked, because a difference between the reference
rerun and the original run could have been either non-determinism or a harness
defect.

**(a) Does the replication harness reproduce itself?**
`MEASURED` (`harness_self_consistency.json`): the same harness, the same seed 0,
two consecutive 2-epoch runs — epoch-0 loss **10.8798 vs 10.5762** (Δ −0.3036),
epoch-1 loss **10.4597 vs 10.0278** (Δ −0.4318). **Not bit-identical.**

`DERIVED` — the divergence is fp32 GPU non-determinism (cuDNN algorithm
selection is not deterministic by default and was not constrained, matching the
original H2 run), **not** a harness difference. The harness is therefore accepted
as running the same recipe.

**(b) How large is the noise?**
`MEASURED` (`same_seed_noise.json`, seed 0's original 200-epoch run vs the
10-epoch reference rerun, epochs 0–9):

| epoch | 0 | 4 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|
| train loss, original | 10.635 | 8.990 | 6.490 | 5.919 | 5.006 |
| train loss, rerun | 10.738 | 8.619 | 7.835 | 8.285 | 7.449 |
| difference | +0.102 | −0.370 | +1.345 | +2.366 | **+2.443** |

Mean |Δ| over the ten epochs 0.802; validation EPE at the one shared checkpoint
differs by 0.571 px. Four independent seed-0 runs (original, reference, two
self-checks) span **10.576–10.880** in epoch-0 loss.

`DERIVED` — same-seed divergence is small at initialisation and **compounds
quickly**: by epoch 9 two runs of the identical configuration differ by 2.44 in
train loss (≈33 % relative). `INFERRED` — early-epoch trajectories are chaotic
under this recipe, so *no* early-epoch difference between seeds can be
attributed to the seed. `UNKNOWN` — the same-seed spread at epoch 200; measuring
it would need a second full seed-0 run, which this experiment does not spend.

## 4. Corrected protocol — frozen

### Gates during training

| epoch | enforced (abort on failure) | recorded only |
|---|---|---|
| 10 | validation EPE improved ≥ 3.0 px from initialisation; matching-path gradient on ≥ 95 % of batches; no NaN/Inf | right-image and matching-map ablations |
| 100 | no NaN/Inf; matching gradient ≥ 95 %; validation EPE no worse than at epoch 10 (collapse check) | right-image and matching-map ablations |
| 200 | — (run completes) | full evaluation, §4.2 |

The stereo-dependence criterion is **not enforced before epoch 200**, because
§1 shows it is not satisfied by the reference at epoch 10 and its epoch-100
value is `UNKNOWN`.

### 4.2 The 200-epoch stereo criterion — defined before the results exist

A replication seed is **stereo-functional** at 200 epochs if all of:

1. **Right-image dependence ≥ +20.0 D1 points** — the minimum D1 penalty across
   the five corruption variants (black, noise, flipped, other scene, left image)
   over the four focus scenes 27/0/31/6.
2. **Matching-map dependence ≥ +20.0 D1 points** — the minimum D1 penalty across
   the two substitutions (own-mean constant, spatial shuffle) over the same
   scenes.
3. **Matching-path gradient** present on ≥ 95 % of training batches.
4. **Softmax entropy > 0.5 nats** on validation frames (H1's broken regime was
   0.0000).
5. **No NaN/Inf**, and final disparity standard deviation > 1.0 px (no constant
   collapse).

Where +20.0 comes from, stated so it cannot be read as tuned: the reference at
200 epochs measures **+80.8…+90.9**; the epoch-10 noise band across all three
seeds is **|ΔD1| ≤ 6.0**. The threshold sits ~3× above the noise and ~4× below
the reference — any H1-like monocular model fails it, any H2-like stereo model
clears it by a wide margin.

### 4.3 Reproducibility verdict — defined before the results exist

- **REPRODUCIBLE** — all three seeds meet every criterion in §4.2, and their
  late-window (epochs 150–199) validation D1 values span less than **10.0
  points**.
- **PARTIALLY REPRODUCIBLE** — all seeds meet §4.2 but the late-window D1 span
  is ≥ 10.0 points.
- **NOT REPRODUCIBLE** — any seed fails any criterion in §4.2 at 200 epochs.

The 10-point span is a **stated convention, not a statistical test**: it is ~10×
seed 0's own late-window D1 standard deviation (1.02) and ~5× below the 56-point
H2-vs-H1 effect. `UNKNOWN` — the same-seed spread at 200 epochs, so this band
cannot be claimed to separate seed effects from run-to-run noise. Three seeds
cannot establish statistical significance and no significance will be claimed.

### 4.4 What may not change from here

No hyperparameter, architecture, normalisation, shift, augmentation, loss,
optimizer, learning rate, crop, dataset or schedule may be altered for any seed.
The thresholds in §4.2 and §4.3 are frozen by this document. If a seed fails,
the failure is the result.

## 5. Files

| file | role |
|---|---|
| `phase2/results/EXP-H2-SEED-REPLICATION-001/config_check.json` | Part A, configuration identity vs the seed-0 record |
| `…/preflight_seed{0,1,2}.json` | Parts B/C/D per seed |
| `…/same_seed_noise.json` | §3(b) |
| `…/harness_self_consistency.json` | §3(a) |
| `phase2/experiments/EXP-H2-SEED-REPLICATION-001-SEED{1,2}` | the aborted first attempt, untouched |
| `phase2/experiments/EXP-H2-SEED-REPLICATION-001-SEED0-REFERENCE` | the 10-epoch control measurement |
| `phase2/experiments/EXP-H2-SEED-REPLICATION-001-SEED{1,2}-RUN2` | the reruns under this protocol |
