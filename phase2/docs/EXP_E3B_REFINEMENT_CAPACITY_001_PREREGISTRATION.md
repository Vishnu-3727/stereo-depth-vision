# EXP-E3B-REFINEMENT-CAPACITY-001 — pre-registration, frozen before the run

**Status: pre-registration. REGISTERED, NOT RUN.** Written and frozen before any
arm was trained. Approved by `PHASE_2_STAGE_E_DECISION_RECORD.md` §3, including
the required design amendment in §3.2 and the conditions in §3.3.

**Claim tags:** `MEASURED`, `DERIVED`, `INFERRED`, `UNKNOWN`.

---

## 1. Question

> Does a refinement stack **trained** at four or five residual blocks match the
> six-block H2, at the same budget, seed and recipe — and does it stay
> stereo-functional?

`MEASURED` (`EXP-E3-REFINEMENT-ABLATION-001-RUN2`) — pruning a *trained*
six-block stack is never free: the cheapest single removal costs +0.746 px EPE
and +11.40 D1 points for 14.9 % of MACs, 11× the frozen NEGLIGIBLE band; every
prefix below five blocks fails constraint C2 on every seed. That result is a
**lower bound** by construction, and E3 recorded the retrained case as `UNKNOWN`.
This experiment measures it.

`MEASURED` (E3, §3) — which block is load-bearing differs by seed: dropping
block 4 costs seed 0 +0.19 px but seeds 1 and 2 +24.16 and +16.37 px. `INFERRED` —
the six blocks divide the work differently on every run, so a retrained smaller
stack may redistribute it rather than simply lose it. That is the reason to run
this at all.

## 2. Arms — stage 1, seed 0 only

Three arms, 200 epochs each, one changed variable (the number of residual blocks
in `Refinement.blocks`):

| arm | blocks | dilations | unique parameters | MACs @368×1232 |
|---|---:|---|---:|---:|
| **A — control** | 6 | 1, 2, 4, 8, 1, 1 | 423,586 | 56.034 G |
| **B** | 5 | 1, 2, 4, 8, 1 | 405,090 | 47.677 G |
| **C** | 4 | 1, 2, 4, 8 | 386,594 | 39.321 G |

`MEASURED` — parameter counts computed from the constructed models; MACs re-cited
from E3's `thop` measurement, not re-derived. One `ResBlock` is 18,496
parameters and 8.357 GMAC at deployment resolution.

**Which blocks are removed, and why, fixed before the run:** the trailing
dilation-1 blocks. The 1-2-4-8 ladder is the architecture's receptive-field
design and is kept intact; the two repeated dilation-1 blocks after it are the
redundancy-by-repetition the capacity question is about. This is *not* chosen
from E3's per-block costs — E3 found the cheapest individual removals to be
blocks 2 and 3 (dilations 4 and 8), and those are deliberately **not** the ones
removed here, because removing them would break the ladder and confound capacity
with receptive field.

**Arm A is a same-seed repeat of the unmodified H2 recipe, not the existing H2
checkpoint.** This is the amendment required by the decision record: the
difference between arm A and `EXP-H2-SOFTARGMIN-SCALE` is the **same-seed
run-to-run noise at 200 epochs**, which constraint C8 has flagged as `UNKNOWN`
since the seed replication. It is measured here as the control, and every
comparison below is judged against it.

## 3. Held fixed

Imported from the H2 recipe rather than restated, exactly as the seed replication
does it: dataset (KITTI 2015, `hailo_calib` 160 scenes train / `hailo_val`
validate), random 256×512 crop with σ=0.1 gain jitter, no flip, Adam(0.9, 0.999)
lr 1e-3, `CosineAnnealingLR T_max=200`, batch 2, masked smooth-L1 (β=1.0),
fp32, seed 0, `cost_volume_shift="left"`, the standardised soft-argmin readout
with `eps=1e-5`, and the recorded validation protocol (first 10 `hailo_val`
scenes, full frames, pooled over `gt > 0`, every 5 epochs).

Phase 1 is not touched; `phase2/models/scaled_regression.py` is not modified; no
record is overwritten. `git diff phase-1-frozen -- src scripts` must be empty
before the run starts and is asserted again at the end.

## 4. The noise band — the amendment's whole point

`DERIVED` — define, from arm A against the recorded `EXP-H2-SOFTARGMIN-SCALE`:

    noise_epe = |A.late_window_mean_epe − 4.199|      (recorded H2 seed-0 value)
    noise_d1  = |A.late_window_mean_d1  − 24.783|

using the late window (epochs 150–199, 11 validation points) on the recorded
protocol.

An ablated arm differs **materially** from arm A only if **both**:

    |Δ EPE vs A| > max(2 × noise_epe, 0.10 px)
    |Δ D1  vs A| > max(2 × noise_d1,  1.0 point)

The floors are E3's and E2's frozen bands, reused verbatim. The 2× factor is a
stated convention fixed here, not a statistical construction.

`UNKNOWN`, stated in advance — **one repeat gives one difference, not a
distribution.** `noise_epe` and `noise_d1` are a single-sample estimate and are a
**lower bound** on same-seed noise, never a confidence interval. No p-value will
be claimed. If an ablated arm lands inside the band, the honest statement is "not
separable at this budget", not "equal".

## 5. Enforced gates during training

Per arm, using the corrected protocol frozen in
`EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md`:

| epoch | enforced (abort on failure) | recorded only |
|---|---|---|
| 10 | validation EPE improved ≥ 3.0 px from initialisation; matching-path gradient on ≥ 95 % of batches; no NaN/Inf | right-image and matching-map ablations |
| 100 | no NaN/Inf; matching gradient ≥ 95 %; validation EPE no worse than at epoch 10 | both ablations |
| 200 | — (run completes) | full evaluation, §6 |

**Stereo dependence is not gated before epoch 200**, and never before epoch 20 in
any form: `MEASURED` (`EXP-H2-EMERGENCE-001-RUN2`) — onset is seed 1 in (10, 20]
and seed 2 in (20, 50], and seed 0 shows −3.33 D1 points at epoch 10 despite
+83.95 at 200. Constraint C7: no gate on a criterion the control has not been
measured against at that budget.

**Preflight, before any training** (blocking): the three models are built from
seed 0 and differ only in block count; every surviving weight tensor is
bit-identical across arms at initialisation; parameter counts equal the table in
§2; no NaN/Inf at any stage; the existing H2 checkpoint's SHA-256 is unchanged.

## 6. Endpoints

**Primary** — validation EPE and D1 on the recorded protocol, late window
(150–199), judged against the §4 band.

**Constraint compliance at epoch 200** (C1–C5, all frozen, all required for an
arm to count as viable):

1. softmax entropy > 0.5 nats on the tensor the softmax consumes;
2. right-image dependence ≥ +20.0 D1 points (worst case, four focus scenes);
3. matching-map dependence ≥ +20.0 D1 points;
4. matching-path gradient on ≥ 95 % of batches, no batch above 1e4 total norm;
5. no NaN/Inf, final disparity std > 1.0 px.

**Secondary, reported never used as the verdict** — parameters, MACs at both
resolutions, wall clock, pooled 40-scene EPE/D1 (the E1/E3 protocol) at the final
checkpoint, and the `disparity_initial`-vs-ground-truth correlation.

`MEASURED` (Phase 1) — MACs mispredict latency on this stack by 0.67×–142×. **No
latency claim will be made**, and MACs are reported as arithmetic only.

## 7. Verdict rule — frozen

Exactly one primary verdict:

| verdict | condition |
|---|---|
| **CAPACITY-REDUCIBLE** | an ablated arm is **not** materially worse than A under §4 **and** meets all of C1–C5. Reported with the block count it holds for. |
| **CAPACITY-REQUIRED** | both ablated arms are materially worse than A under §4. |
| **STEREO-BROKEN** | an ablated arm is not materially worse on accuracy but **fails C2 or C3**. This is the expected failure mode given E3 and E2, and it is a rejection, not a success: the arm is not viable and must not be reported as an efficiency result. |
| **INCONCLUSIVE** | the noise band is wider than every measured difference, or the two metrics disagree for an arm, or an arm aborts at a gate. |
| **INVALID** | preflight fails, arm A fails to reproduce H2's qualitative trajectory (does not converge, or is not stereo-functional at 200), a checkpoint hash changes, or the Phase 1 diff is non-empty. |

## 8. Stage 2 — conditional, and pre-committed now

Seeds 1 and 2 are run for an arm **only if** that arm's seed-0 result falls
outside the §4 band (in either direction). Otherwise stage 2 is not run and the
result stands as "not separable at seed 0". Stage 2, if triggered, is a
separately logged stage of this experiment with its own records
(`…-SEED1`, `…-SEED2`), and adds no new arms.

`DERIVED` — this is a pre-commitment, not a decision to be made after seeing the
numbers.

## 9. What may not change

No hyperparameter, architecture detail other than the block count, normalisation,
shift, augmentation, loss, optimizer, learning rate, crop, dataset, schedule,
band, floor, gate or threshold in this document may be altered once the run
starts. Records are never overwritten; a rerun takes a new ID.

## 10. Cost

`DERIVED` — three 200-epoch arms at the measured 3,627–3,727 s per H2-recipe run
≈ **3 GPU-hours** on the RTX 4060, plus minutes of preflight and probes. The
ablated arms are cheaper per epoch than the control.

## 11. Limitations, stated in advance

- `UNKNOWN` — this runs in the 160-scene, random-initialisation regime. A
  capacity result here **does not transfer** to a pretrained model, and the
  report must say so rather than implying a deployment conclusion.
- `UNKNOWN` — latency, on any device.
- `UNKNOWN` — whether a different *shape* of refinement (fewer channels, other
  dilations, a different guidance path) beats fewer blocks. This experiment
  changes block count only.
- One seed in stage 1; the harness is not bit-reproducible, which is precisely
  why arm A exists. No significance test, none claimed.
- E3b does **not** open Stage E: it is a refinement-capacity experiment on the
  downstream half of the network, and Stage E remains closed.
