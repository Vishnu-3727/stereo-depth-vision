# Stage B ARM-P — Environment-ConFound Control: Pre-registration

**File timestamp (UTC):** 2026-09-19T10:03:02Z
**Status at time of writing:** The two Kaggle kernels (seed-1 environment control, both arms) are still RUNNING. No result from either kernel has been observed. This document was written before any result was available, and it contains no result values beyond previously recorded reference numbers restated as context.

**Purpose of this file:** To fix the analysis and interpretation rules for the seed-1 Kaggle environment-confound control in advance, so that they cannot be adjusted after the numbers are seen.

---

## 1. Purpose

Seed 2 was trained on Kaggle (Tesla T4, torch 2.10.0+cu128, Python 3.12.13), while seeds 0 and 1 were trained locally (torch 2.7.0+cu128). That is a between-seed environment difference, and it is NOT the intended independent variable. The intended independent variable is initialization (ARM-P versus random initialization under the original P2A recipe).

This control re-runs SEED 1, BOTH ARMS, in the seed-2 Kaggle environment, in order to test whether the ARM-P advantage is robust to that environment change.

This control is an ENVIRONMENT-CONFOUND CONTROL. It is NOT a new seed. It is NOT seed 3. It is NOT seed 4. It does NOT replace the original seed-1 result. Its only job is to measure how much of the observed delta moves when the same seed is trained in the two different environments.

---

## 2. The four distinct runs

The following four runs are recorded as separate experiments, permanently. They are not merged, not averaged, and not collapsed into fewer entries:

1. **Seed 0, local** — original replication.
2. **Seed 1, local** — original replication.
3. **Seed 2, Kaggle** — third replication, with the environment differing from seeds 0 and 1.
4. **Seed 1, Kaggle rerun** — environment-confound control, not a new seed.

The original seed-1 local numbers are NEVER replaced by the seed-1 Kaggle numbers. Both stand in the record side by side: the local seed-1 result remains the seed-1 replication, and the Kaggle seed-1 rerun stands alongside it purely as the environment-confound control. Any table or leaderboard entry must show them as two distinct rows, never overwriting one with the other.

---

## 3. The quantity of interest: the within-environment delta

The quantity of interest is the WITHIN-ENVIRONMENT DELTA (ARM-P minus CONTROL, measured inside one environment), not the absolute EPE. Absolute EPEs may legitimately differ between environments even if the delta holds — a change in absolute level alone does not, by itself, undermine the ARM-P effect.

The reference values for the local environment are fixed here:

- `delta_local_best = -0.2528074` (ARM-P 1.1912168 − CONTROL 1.4440242)
- `delta_local_final = -0.2499701` (ARM-P 1.2017376 − CONTROL 1.4517077)

The corresponding Kaggle-environment quantities are defined, but their values are as yet unobserved and are NOT filled in here:

- `delta_kaggle_best = ARMP_kaggle_best − CONTROL_kaggle_best`
- `delta_kaggle_final = ARMP_kaggle_final − CONTROL_kaggle_final`

The environment shift — the change in the delta between environments — is defined as:

- `environment_shift_best = delta_kaggle_best − delta_local_best`
- `environment_shift_final = delta_kaggle_final − delta_local_final`

When the Kaggle results arrive, the environment shift MUST be reported as a number at full precision, for both selections (best and final). It must not be described only in words such as "close" or "reproduces" without the number beside them. In addition, the absolute per-arm EPE changes between environments (CONTROL Kaggle versus CONTROL local; ARM-P Kaggle versus ARM-P local, for both best and final selections) must be reported as separate, clearly-labelled quantities, so that a reader can see whether any shift in the delta comes from one arm or from both.

---

## 4. The 0.0444 px figure and the cross-seed spreads

The 0.0444 px figure is stated plainly for what it is: a PREVIOUSLY OBSERVED run-to-run spread, quoted here as a reference number. It is NOT a formal statistical significance threshold. It is NOT a confidence interval. It is NOT a decision boundary derived from this experiment. When the measured environment shift is reported, the 0.0444 px figure may be quoted as context beside it, and nothing more. No pass/fail verdict is derived from it.

Already-observed cross-seed spread is likewise recorded as context: across seeds 0/1/2, the random-init CONTROL best-checkpoint values span about 0.1147 px, while the ARM-P best values span about 0.0145 px. In other words, the control is the noisier arm. That asymmetry is relevant when reading any shift — a movement driven largely by the CONTROL arm is consistent with its already-observed noisiness and should be read that way, not as evidence about ARM-P specifically.

---

## 5. Reporting rules (fixed in advance)

1. **Both selections are reported.** Best-checkpoint and final-checkpoint results are both reported in full. Only the flattering one is never reported alone.
2. **All four disparity bins are reported for both arms** (`<64`, `64–96`, `96–128`, `>=128` px). It is noted in advance that the `>=128` bin has held exactly 1080 px (0.028%) at every seed so far and is too thin to carry weight; it is reported for completeness, not used as the basis of any substantive claim.
3. **No pooling.** There is no pooling across seeds, no averaging across seeds, and no combined statistic. Each seed's delta stands on its own.
4. **No significance claims.** There is no claim of statistical significance and no fabricated confidence interval. The 0.0444 px reference number (Section 4) is context, not a test.
5. **Bins do not establish mechanism.** A bin improvement is a description of where the EPE difference sits, not evidence of why it sits there. No mechanistic claim follows from a bin table alone.
6. **Naming of the CONTROL.** The CONTROL is RANDOM INITIALIZATION under the original P2A recipe. It is never to be called P2A "checkpoint" initialization. The distinction matters: the comparison is against random initialization, not against any pretrained checkpoint.
7. **Frozen evaluation.** Evaluation of the Kaggle checkpoints is performed LOCALLY with the same frozen evaluator used for seeds 0, 1 and 2, keeping the measurement layer constant. The frozen contract — 40 scenes, 3,802,797 valid pixels, gt_scale 256.0, gt_source disp_occ_0 — must hold exactly, or the run is invalid and nothing is concluded from it.

---

## 6. Interpretation branches (fixed before the numbers are seen)

The branch that applies is selected by the measured shift, not by preference. What will be concluded under each branch is written here now:

- **Small shift relative to the quoted context spreads.** If the environment shift is small relative to the context spreads quoted in Section 4, then the environment-confound concern on seed 2 is substantially reduced, and the three seeds stand as they are. This is not proof that the environment has no effect — it is only the statement that this control did not reveal a material one.
- **Material shift.** If the environment shift is material, then seed 2's delta carries an environment component. That caveat is attached to seed 2 explicitly — in the leaderboard entry and in any summary of the three seeds — and seed 2 is not silently dropped. The three-seed direction (ARM-P ahead on all three seeds) may still hold descriptively, but seed 2 is then no longer a clean same-environment replication, and it must be described that way.
- **Invalid control.** If the frozen contract fails, or if either arm does not complete the full 200 epochs, the control is INVALID and nothing is concluded from it. An invalid control neither supports nor undermines any seed; it is recorded as invalid and set aside.

Under every branch: NO next experiment is launched automatically. Any follow-up, if proposed at all, is a separate decision recorded separately — it is never treated as implied by the outcome of this control.

---

## 7. Declaration

This file was written at 2026-09-19T10:03:02Z (real current UTC), while the two seed-1 Kaggle kernels (CONTROL arm and ARM-P arm) were still RUNNING and before any result from either kernel had been observed. No result value is invented, estimated, simulated, or predicted in this document. The only numbers it contains are previously recorded reference values restated as context (Sections 3 and 4). The analysis rule above is fixed before the numbers are seen.
