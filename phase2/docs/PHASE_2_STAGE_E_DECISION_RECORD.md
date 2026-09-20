# Stage E — evidence review and decision record

**Date:** 2026-09-08. **Scope:** everything the Stage E candidate programme has
produced — E1, E3, E2 — read against the constraints frozen in
`PHASE_2_H2_EVIDENCE_REVIEW.md`, and a decision on whether to open Stage E and
whether to register **E3b**.

**Claim tags:** `MEASURED`, `DERIVED`, `INFERRED`, `UNKNOWN`.

**Nothing new is measured here.** Every number is re-cited from a named record.
`MEASURED` — `git diff phase-1-frozen -- src scripts` is empty; Phase 1 is
untouched. 21 experiment records exist under `phase2/experiments/`; the full test
suite is **169 passing**.

---

## 1. What the three candidate experiments established

| | question | verdict | cost |
|---|---|---|---|
| **E1** (`EXP-E1-ERROR-PARTITION-001` + `-002-RUN3` + `-003-HELDOUT-ORACLE`) | how much of H2's residual error would a perfect matching stage remove? | **NO CONSISTENT VERDICT** | ~4 min |
| **E3** (`EXP-E3-REFINEMENT-ABLATION-001-RUN2`) | how much of the six-block refinement stack is load-bearing? | **REFINEMENT IS RIGHT-SIZED** | 154 s |
| **E2** (`EXP-E2-READOUT-TEMPERATURE-001`) | is the error sensitive to readout sharpness? | **READOUT-ROBUST** (case B) | 160 s |

`MEASURED` — the three together cost under ten minutes of GPU time and closed
three of the five candidates named in the evidence review.

### 1.1 What is now established

1. `MEASURED` — **the disparity range is not a constraint.** 0 % of `hailo_val`
   ground truth exceeds the architecture's 176 px ceiling (max 153.04 px).
2. `MEASURED` — **the readout temperature is not an exploitable lever.** Zero of
   24 non-control (seed, T) cells improves either metric; the trained solution
   sits at a sharp optimum of its own training temperature.
3. `MEASURED` — **the refinement stack cannot be pruned for free.** The cheapest
   single-block removal costs +0.746 px and +11.40 D1 points for 14.9 % of MACs —
   11× the frozen NEGLIGIBLE band.
4. `MEASURED` — **which block is redundant is a property of the run, not the
   architecture.** Dropping block 4 costs seed 0 +0.19 px and seeds 1 and 2
   +24.16 and +16.37 px.
5. `MEASURED` — **stereo dependence is destroyed from both ends.** Truncating
   refinement below five blocks (E3) and moving the readout temperature off 1.0
   in either direction (E2) both drive right-image dependence to zero or
   negative, with the cost volume, aggregation and standardisation untouched.
   `DERIVED` — H2's use of the second camera is a property of the *whole trained
   configuration*, not of the cost volume alone.
6. `MEASURED` — the analytic ≈0.77 peak-confidence cap the H2 report derived is
   confirmed by direct measurement: max peak softmax probability 0.769–0.772.

### 1.2 What is still unknown

1. `UNKNOWN` — **what a retrained smaller refinement stack does.** E3 measured
   weights trained as six blocks; it is a lower bound by construction, and
   finding 1.1.4 is positive evidence that a retrained stack would redistribute
   the work differently.
2. `UNKNOWN` — **what a model trained at a different readout temperature does.**
   E2 is inference-only and says so.
3. `UNKNOWN` — **the same-seed run-to-run spread at 200 epochs.** Measured at 10
   epochs (compounding to ≈33 % of train loss by epoch 9); never measured at the
   full budget. This is the outstanding half of constraint **C8**.
4. `UNKNOWN` — **behaviour at real data scale.** Every Phase 2 number rests on
   160 training scenes from random initialisation; pooled EPE 2.32–2.56 px and
   D1 16.7–19.8 % against the pretrained reference's 1.31 px / 8.15 %.
5. `UNKNOWN` — **latency.** MACs mispredict latency on this stack by 0.67×–142×,
   and no Phase 2 experiment has measured any.
6. `UNKNOWN` — failure modes by scene class; physical device behaviour. Both
   deferred since Phase 1.

## 2. Decision 1 — does Stage E open?

    STAGE E REMAINS FORMALLY CLOSED.

`DERIVED` — Stage E is *cost-volume optimization*. After E1, E2 and E3, **not one
measurement implicates the cost volume in H2's remaining error**:

- E1 could not attribute a majority of the error upstream — its honest, held-out
  estimate puts ≈⅓ upstream and ≈⅔ downstream, and even that is bracketed
  between an 8–12 % lower bound and a 61–70 % leaky upper bound;
- the range ceiling is excluded (§1.1.1);
- the read-out, which is the cost volume's consumer, is robust (§1.1.2);
- the matching path's own known deficiency — H2's trained features reach only
  +0.414 hand-made-volume correlation with ground truth — was never shown to be
  the binding constraint on the *output*.

`DERIVED` — the classic Stage E levers (candidate count, volume representation,
hierarchical search) also target 4.2 % of arithmetic and 2.8 % of GPU time, and
would be optimising a component this programme cannot yet show is limiting.

`INFERRED` — E4 (volume representation) remains the only Stage E candidate with a
live rationale, and its cheap form (alternative volumes on H2's already-trained
features) is worth running *before* any Stage E training. It is **not** registered
here; §4 explains why it is second in line.

## 3. Decision 2 — is E3b approved?

    APPROVED, WITH A REQUIRED DESIGN AMENDMENT.

### 3.1 Why it is approved

`DERIVED` — E3b answers a documented `UNKNOWN` (§1.2.1) that E3 explicitly could
not, with a clean single-variable hypothesis, an existing trained control, an
existing recipe, and an existing probe suite. It targets the stage that carries
**90.6 % of MACs** and **58–71 % of the residual error** — the only place in this
architecture where either quantity is concentrated.

`DERIVED` — it is also the cheapest of the three remaining candidates by an order
of magnitude (~3 GPU-hours against O6's ~22 h), and its result is informative in
both directions: a retrained four-block stack matching H2 is a real efficiency
finding; one that does not match reinforces E3 and closes the capacity question.

### 3.2 The amendment, and why it is required

`MEASURED` (E3, finding 1.1.4) — which block is load-bearing differs by seed.
`MEASURED` (seed replication §3) — the harness is not bit-reproducible and
same-seed divergence compounds quickly. `UNKNOWN` (§1.2.3) — the same-seed spread
at 200 epochs.

`DERIVED` — **a naive E3b would be uninterpretable.** A four-block arm scoring
1 D1 point away from H2 could not be distinguished from run-to-run noise, because
that noise has never been measured at this budget. Constraint **C8** forbids the
claim, and constraint **C7** forbids gating on an uncalibrated criterion. This
programme has already been bitten twice by exactly this: the epoch-10 stereo gate
that killed two healthy seeds, and the diverging fixed-point probe that returned
a false confirmation.

**Required amendment:** E3b's control arm is a **same-seed repeat of the unmodified
six-block H2 recipe**, not the existing H2 checkpoint. That repeat *is* the missing
same-seed noise measurement, it comes free as the control, and every E3b
comparison is judged against the band it produces. No difference smaller than that
band may be reported as an effect.

`DERIVED` — this converts E3b from "an experiment that needs C8 satisfied first"
into "the experiment that satisfies C8 while answering its own question", and it
removes the need for a separate 1-hour noise run.

### 3.3 Conditions attached

1. Seed 0 only in the first stage: three arms (6-block repeat, 5-block, 4-block).
   Extension to seeds 1 and 2 is gated on the seed-0 result exceeding the noise
   band, and is a separately recorded stage of the same experiment.
2. The pre-registered primary endpoint is **accuracy relative to the same-seed
   noise band**, not MACs. MACs are reported as arithmetic, never as latency
   (§1.2.5).
3. Constraints **C1–C5** are enforced on every trained arm: entropy > 0.5 nats on
   the tensor the softmax consumes; right-image and matching-map dependence
   ≥ +20 D1 points; matching gradient ≥ 95 % of batches with no spike above 1e4;
   no NaN/Inf; final disparity std > 1.0 px. `MEASURED` (E3) — truncation destroys
   stereo dependence, so an ablated arm that scores well on accuracy while failing
   C2/C3 is the expected failure mode here, and must be caught rather than
   celebrated.
4. Stereo dependence may not be gated before epoch 20 (emergence: seed 1 in
   (10, 20], seed 2 in (20, 50]).
5. E3b does **not** open Stage E. It is a refinement-capacity experiment on the
   downstream half of the network.

### 3.4 What E3b will not settle

`UNKNOWN` — E3b runs in the 160-scene regime. A capacity result there does not
transfer to a pretrained model, and the report must say so. `UNKNOWN` — latency.
`UNKNOWN` — whether a *different shape* of refinement (fewer channels, different
dilations) would do better than fewer blocks; E3b changes block count only.

## 4. Ordering of everything else

| rank | candidate | why here | cost |
|---|---|---|---|
| 1 | **E3b** — registered by this record | §3 | ~3 GPU-h |
| 2 | **E4-cheap** — alternative cost-volume representations on H2's trained features, scored against the recorded +0.414 baseline | no training, minutes, and the only remaining live Stage E rationale; a negative result closes Stage E on evidence rather than on absence | minutes |
| 3 | **O6** — Scene Flow pretraining | largest expected effect on the absolute numbers and the single largest confound behind every Phase 2 result, but blocked on dataset size, not compute | ~22 h |
| 4 | **E2b** — train at a different or learnable temperature | E2 makes it a legitimate question but a low-prior one: the trained solution is already at a sharp optimum of its own temperature | ~2 GPU-h |
| 5 | **E5** — constrain the raw cost scale at source | unchanged from the evidence review: last | ~4 min gate |

`INFERRED` — running E4-cheap before or alongside E3b would be defensible; it is
ranked second only because E3b has the sharper hypothesis and E4-cheap's negative
result is the more likely one.

## 5. Decision summary

1. **Stage E: CLOSED.** No measurement implicates the cost volume.
2. **E3b: APPROVED**, with the same-seed control amendment of §3.2 and the
   conditions of §3.3. Registered as
   `EXP-E3B-REFINEMENT-CAPACITY-001`; pre-registration:
   `phase2/docs/EXP_E3B_REFINEMENT_CAPACITY_001_PREREGISTRATION.md`.
3. Registration is not execution. No training has been run by this record.
