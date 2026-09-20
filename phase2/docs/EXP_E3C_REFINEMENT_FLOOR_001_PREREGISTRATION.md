# EXP-E3C-REFINEMENT-FLOOR-001 — pre-registration, frozen before the run

**Status: pre-registration.** Written and frozen before the arm was trained.

**Claim tags:** `MEASURED`, `DERIVED`, `INFERRED`, `UNKNOWN`.

---

## 0. Why this is a new experiment and not "E3b stage 2"

`EXP-E3B-REFINEMENT-CAPACITY-001` §8 pre-committed stage 2 to run seeds 1 and 2
**only for an arm whose seed-0 result fell outside the material band**, and stated
that stage 2 "adds no new arms". `MEASURED` — no arm fell outside the band, so
stage 2 did not trigger; and a three-block arm is a new arm either way. Running it
under E3b's ID would mean editing a protocol after seeing its results, which is
the failure mode this project's record discipline exists to prevent.

So it is registered here, separately, and E3b's records and verdict are untouched.
E3b supplies two things this experiment reuses as **measured inputs**: its control
arm and its noise band.

## 1. Question

> Four refinement blocks matched six. Does **three** still match, and does it stay
> stereo-functional?

`MEASURED` (E3b) — at 200 epochs, seed 0, the same recipe: 6 blocks 4.0921 px /
24.9097 %, 5 blocks 3.8588 / 25.7856, 4 blocks 4.0028 / 26.0708 — none materially
different, all stereo-functional. `UNKNOWN` (E3b §6) — anything below four blocks.
This experiment takes one step further down and looks for the floor.

## 2. The arm

| | blocks | dilations | unique parameters | MACs @368x1232 |
|---|---:|---|---:|---:|
| **D** (this experiment) | 3 | 1, 2, 4 | **368,098** | **30.964 G** |
| A — control (E3b, already trained) | 6 | 1, 2, 4, 8, 1, 1 | 423,586 | 56.034 G |

`MEASURED` — the MAC figure is E3's `thop` measurement of the "keep first 3"
configuration, re-cited, not re-derived. The parameter count is 386,594 - 18,496
(one `ResBlock`), and the preflight asserts it against the constructed model.

**Which blocks are removed, fixed before the run:** `keep first 3`, exactly the
rule E3b used — trailing blocks go, leading blocks stay.

**A confound this arm cannot avoid, stated now rather than discovered later.**
E3b's arms B and C dropped only the *repeated dilation-1* blocks, so the 1-2-4-8
receptive-field ladder survived intact and the only variable was capacity. At
three blocks that is impossible: `keep first 3` is dilations 1, 2, 4, which
**drops dilation 8** and therefore shrinks the receptive field as well as the
capacity. `DERIVED` — a negative result here cannot distinguish "not enough
capacity" from "receptive field too small", and must not be reported as if it
could. A positive result is unambiguous (both were reduced and it still worked);
a negative result requires the follow-up in section 8.

## 3. Held fixed

Everything except the block count, imported from the H2 / E3b recipe rather than
restated: dataset (`hailo_calib` 160 scenes train / `hailo_val` validate), random
256x512 crop with sigma=0.1 gain jitter, no flip, Adam(0.9, 0.999) lr 1e-3,
`CosineAnnealingLR T_max=200`, batch 2, masked smooth-L1 (beta=1.0), fp32,
**seed 0**, `cost_volume_shift="left"`, the standardised soft-argmin readout with
`eps=1e-5`, and the recorded validation protocol.

Phase 1 untouched; `phase2/models/scaled_regression.py` unmodified; no record
overwritten; `git diff phase-1-frozen -- src scripts` empty before and after.

## 4. Control and band — reused, not recomputed

`MEASURED` (E3b) — control is **arm A of E3b** (`...-ARM-A-RUN2`, 6 blocks, seed 0,
200 epochs), late window 4.0921 px / 24.9097 %.

`MEASURED` (E3b) — the same-seed 200-epoch noise band, from arm A against the
recorded `EXP-H2-SOFTARGMIN-SCALE`:

    material band:  |delta EPE| > 0.2138 px   AND   |delta D1| > 1.0000 point

Both thresholds must be exceeded, as in E3b. The band is **not** recomputed here:
re-measuring it would need another control run, and reusing a measured quantity is
the point of having measured it.

`UNKNOWN`, carried over unchanged — the band is a single-sample lower bound on
same-seed noise, not a confidence interval. No p-value is claimed.

## 5. Gates

**Preflight (blocking):** the 3-block model differs from E3b's control in nothing
but block count; every surviving weight tensor is bit-identical at initialisation;
parameter count equals 368,098; no NaN/Inf at any stage; E3b's arm-A checkpoint
SHA-256 unchanged.

**During training,** the corrected protocol from
`EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md`, identical to E3b:

| epoch | enforced (abort) | recorded only |
|---|---|---|
| 10 | val EPE improved >= 3.0 px from init; matching gradient >= 95 % of batches; no NaN/Inf | both ablations |
| 100 | no NaN/Inf; matching gradient >= 95 %; no regression vs epoch 10 | both ablations |
| 200 | — | full evaluation |

Stereo dependence is never gated before epoch 20 and is judged only at 200.

## 6. Endpoints

**Primary** — late-window (epochs 150-199) validation EPE and D1 against arm A,
judged by the section 4 band.

**Constraints C1-C5 at epoch 200**, all required for viability: entropy > 0.5 nats;
right-image dependence >= +20 D1 points; matching-map dependence >= +20; matching
gradient >= 95 % of batches with no batch above 1e4; no NaN/Inf and final disparity
std > 1.0 px.

**Secondary, never the verdict** — parameters, MACs, wall clock,
`disparity_initial`-vs-ground-truth correlation. MACs are arithmetic; `MEASURED`
(Phase 1) they mispredict latency on this stack by 0.67x-142x, and **no latency
claim will be made**.

## 7. Verdict rule — frozen

| verdict | condition |
|---|---|
| **CAPACITY-REDUCIBLE-TO-3** | arm D is not materially worse than arm A **and** meets C1-C5. |
| **CAPACITY-REQUIRED-ABOVE-3** | arm D is materially worse than arm A on both metrics. |
| **STEREO-BROKEN** | arm D is not materially worse on accuracy but fails C2 or C3. A rejection, not a result. |
| **INCONCLUSIVE** | the metrics disagree in a way the conjunction rule cannot resolve, or the arm aborts at a gate. |
| **INVALID** | preflight fails, a checkpoint hash changes, or the Phase 1 diff is non-empty. |

## 8. Pre-committed follow-up

`DERIVED` — if arm D is materially worse **or** STEREO-BROKEN, the section 2
confound makes the cause ambiguous, and the disambiguating experiment is a
**second three-block arm that keeps the wide dilation** — `(1, 2, 8)` or
`(1, 4, 8)`, same parameter count and same MACs. That arm is *not* registered now
and is not run in this experiment; it is named here so the choice cannot look
opportunistic later.

If arm D is viable, no follow-up is needed at three blocks and the open question
becomes two blocks.

## 9. What may not change

No hyperparameter, architecture detail other than block count, band, threshold or
gate may be altered once the run starts. Records are never overwritten; a rerun
takes a new ID.

## 10. Cost and limitations

`DERIVED` — one 200-epoch arm at E3b's measured rates ~ **1 GPU-hour** (arm C, 4
blocks, took 3,578 s).

- **One seed.** The honest phrasing for a non-material result is "not separable at
  seed 0", never "equal".
- `UNKNOWN` — transfer to a pretrained model. 160 training scenes from random
  initialisation, ~4 px EPE against the pretrained reference's 1.31 px.
- `UNKNOWN` — latency, on any device.
- The receptive-field confound of section 2.
- This experiment does **not** open Stage E; the cost volume is untouched.
