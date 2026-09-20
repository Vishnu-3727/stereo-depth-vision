# EXP-E3B-REFINEMENT-CAPACITY-001 — does a *trained* smaller refinement stack match H2?

**Date:** 2026-09-08. **Compute:** 3.2 GPU-hours (4,361 + 3,646 + 3,578 s), plus a
discarded partial run. **Protocol frozen in advance:**
`EXP_E3B_REFINEMENT_CAPACITY_001_PREREGISTRATION.md`, approved by
`PHASE_2_STAGE_E_DECISION_RECORD.md`.

**Claim tags:** `MEASURED`, `DERIVED`, `INFERRED`, `UNKNOWN`.

**Records** (none overwritten): `…-ARM-A-RUN2`, `…-ARM-B-RUN2`, `…-ARM-C-RUN2`,
`…-ANALYSIS-RUN2` (all authoritative), and `…-ARM-A` — a first attempt killed by
host memory pressure at epoch 146, preserved with a `NOTE.md` and **not evidence**.
Raw data: `phase2/results/EXP-E3B-REFINEMENT-CAPACITY-001/`. Script:
`phase2/scripts/exp_e3b_refinement_capacity.py`.

---

## 1. Verdict

    CAPACITY-REDUCIBLE

`MEASURED` — **neither ablated arm is materially different from the control, and
both are stereo-functional.** Four refinement blocks reach the same accuracy as
six, at the same seed, recipe and budget, for **30 % fewer MACs** (56.034 →
39.321 G at 368×1232) and 36,992 fewer parameters.

All four pre-registered gates passed: the control completed 200 epochs, converged
(31.958 → 4.092 px), and is stereo-functional; no arm aborted.

## 2. The same-seed noise band — constraint C8's missing measurement

`MEASURED` — arm A is a repeat of the unmodified H2 recipe at seed 0. Against the
recorded `EXP-H2-SOFTARGMIN-SCALE` late window (4.199 px / 24.783 %):

| quantity | value |
|---|---:|
| same-seed noise, EPE | **0.1069 px** |
| same-seed noise, D1 | **0.1267 points** |
| material band (2× noise, floored at 0.10 px / 1.0 pt) | **0.2138 px / 1.0000 pt** |

`DERIVED` — same-seed run-to-run noise at 200 epochs is **small**, despite the
same two runs differing by 6.30 D1 points at epoch 59. The trajectories are
chaotic mid-run and converge as the cosine schedule decays. This is the first
measurement of that quantity in the project and it applies to every future
training comparison, whatever E3b itself concluded.

`UNKNOWN` — one repeat gives one difference. The band is a **single-sample lower
bound**, not a confidence interval, and no p-value is claimed.

## 3. Results

`MEASURED` — late window = epochs 150–199 on the recorded validation protocol
(first 10 `hailo_val` scenes, full frames, pooled over `gt > 0`).

| arm | blocks | dilations | parameters | MACs @368×1232 | late EPE (px) | late D1 (%) | ΔEPE vs A | ΔD1 vs A | materially different? |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| **A** control | 6 | 1,2,4,8,1,1 | 423,586 | 56.034 G | 4.0921 ± 0.1000 | 24.9097 ± 1.0821 | — | — | — |
| **B** | 5 | 1,2,4,8,1 | 405,090 | 47.677 G | **3.8588 ± 0.0644** | 25.7856 ± 0.7130 | **−0.2333** | +0.8759 | **no** |
| **C** | 4 | 1,2,4,8 | 386,594 | 39.321 G | 4.0028 ± 0.0753 | 26.0708 ± 0.6425 | −0.0893 | +1.1610 | **no** |

`MEASURED` — stereo functionality at epoch 200, all three arms **STEREO-FUNCTIONAL**:

| arm | right-image dependence | matching-map dependence | entropy (nats) | disparity std (px) | matching gradient | grad max | spikes >1e4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | **+82.00** | **+68.07** | 1.780 | 17.66 | 100 % of 16,000 | 326.6 | 0 |
| B | **+79.71** | **+63.52** | 1.814 | 18.27 | 100 % of 16,000 | 317.1 | 0 |
| C | **+75.42** | **+64.19** | 1.815 | 18.03 | 100 % of 16,000 | 319.3 | 0 |

`DERIVED` — every arm clears C2 and C3 by a factor of ~3–4, with no H1 pathology
anywhere: no saturation, no tie-driven spikes, no collapse, no dead matching path.

## 4. What the numbers say, precisely

1. `DERIVED` — **the conjunction rule is what makes both arms "not different", and
   one arm came close to failing it.** Arm C is +1.1610 D1 points worse than the
   control, which **exceeds** the 1.0-point threshold; it is saved only because its
   EPE difference (−0.0893 px) is inside the 0.2138 px band, and the frozen rule
   requires **both** metrics to move. A disjunctive rule would have called arm C
   materially worse. This is stated because the rule was frozen before the data
   existed and must not be re-read afterwards.
2. `DERIVED` — **the verdict is robust to replacing the D1 floor with measured
   noise.** The 1.0-point floor is ~8× the measured D1 noise (0.127). Using
   2 × noise = 0.2535 instead: arm B becomes materially *different* but not
   materially *worse* (its EPE moves in its own favour), and arm C stays inside
   the EPE band. Both remain viable, and the verdict is unchanged.
3. `MEASURED` — **both ablated arms beat the control on EPE and lose to it on D1**
   (B: −0.233 px / +0.876 pt; C: −0.089 px / +1.161 pt). `INFERRED` — the
   direction is consistent across the two arms but each rests on a single run, and
   the same metric disagreement has appeared in H1 v2, in E3b's own mid-run
   checkpoints and in the control against itself. It is not established as a
   capacity effect.
4. `MEASURED` — the ablated arms show **lower late-window variance** than the
   control (EPE sd 0.064 / 0.075 vs 0.100; D1 sd 0.713 / 0.643 vs 1.082).
   `UNKNOWN` — whether that is a property of smaller stacks or a one-run accident.
5. `MEASURED` — train-loss ordering inverted mid-run: C < B < A up to epoch ~79,
   then A < B < C from epoch 99 onward. `INFERRED` — the larger stacks fit the 160
   training scenes harder in the second half while validation stays level, which
   is what extra capacity does on a small dataset.

## 5. The finding that matters: pruning and retraining disagree

`MEASURED` (`EXP-E3-REFINEMENT-ABLATION-001-RUN2`) — **pruning** a trained
six-block stack to four blocks cost +24.15 px EPE and +81.49 D1 points, and drove
right-image dependence to **−16.9 points** — the H1 monocular signature.

`MEASURED` (here) — **retraining** at four blocks costs −0.089 px and +1.161
points, and keeps right-image dependence at **+75.42**.

`DERIVED` — **the two operations give opposite answers about the same
architecture.** E3's "refinement is right-sized" verdict was correct for the
question it asked (can a trained stack be pruned?) and is not a statement about
how much capacity the architecture needs. E3's report recorded the retrained case
as `UNKNOWN` and declined to extrapolate; that caution was warranted.

`DERIVED` — this also explains E3's finding that *which* block is redundant
differs by seed: the six blocks divide work that a four-block stack simply divides
differently when trained from scratch. Capacity questions in this architecture
cannot be answered by pruning.

## 6. What this does not establish

- `UNKNOWN` — **anything below four blocks.** Three and fewer were not trained.
- `UNKNOWN` — **transfer to a pretrained model.** Everything here is 160 training
  scenes from random initialisation, at 3.9–4.1 px EPE against the pretrained
  reference's 1.31 px. A capacity result at this budget says nothing about the
  capacity a converged model needs, and this is the single largest confound in
  Phase 2 (O6).
- `UNKNOWN` — **latency.** `MEASURED` (Phase 1) — MACs mispredict latency on this
  stack by 0.67×–142×. The 30 % MAC saving is arithmetic. Wall clock did fall
  (4,361 → 3,578 s, −18 %) but that is training throughput on one GPU, not
  inference latency on a target device.
- `UNKNOWN` — whether a different *shape* of refinement (fewer channels, other
  dilations) beats fewer blocks. Only block count was changed.
- **One seed per arm.** Stage 2 (seeds 1 and 2) was pre-committed to run only for
  an arm falling outside the band; none did, so it was not run. The honest
  phrasing is **"not separable at seed 0"**, not "equal".
- `MEASURED` — a protocol asymmetry: each arm's late window holds **10** validation
  points (epochs 154–199) while the recorded H2 run's held 11 (epochs 150–199);
  the two scripts align validation epochs differently by one. The noise band is
  computed across that asymmetry. Small against the ±0.10 px spread, but real.

## 7. Anomalies

1. **The first attempt was killed by the host at arm A epoch 146** for system
   memory pressure — not an experiment failure. The record was never finalised (no
   `metrics.json`), is preserved with a `NOTE.md`, and is not cited. It was not
   resumed: the harness saves model weights only, not optimizer moments or
   scheduler position, so resuming would have been a different recipe, which §9 of
   the pre-registration forbids. The rerun restarted from epoch 0 with new IDs.
2. `MEASURED` — the interruption produced an unplanned benefit: **three** same-seed
   samples at epoch 19 (7.970 / 8.099 px, 53.87 / 57.22 % plus the recorded run's
   8.070 / 58.54 %), spanning 0.13 px and 4.67 points from fp non-determinism
   alone.
3. The `Experiment` harness logged `WARNING config fields not supplied: split,
   resolution, disparity_range` for every arm — the imported `recipe_config` does
   not carry those keys. A metadata gap in the record header, not a recipe
   difference; the split and resolution are stated in `validation_protocol` and
   the disparity range is fixed by `num_disparities: 12`.

## 8. Next step — recommendation only

`DERIVED` — E3b answers its question and **does not open Stage E**: nothing here
touches the cost volume, and Stage E remains formally CLOSED.

The result changes the ordering of what remains. A four-block refinement stack at
30 % fewer MACs with intact stereo function is the first efficiency finding in
Phase 2 — but it was measured in a regime three times worse than the reference,
which is exactly the confound **O6** exists to remove. Re-asking the capacity
question after pretraining is more valuable than extending it now, because a
capacity result that does not survive real training budget is not a deployment
result.

Candidates, unchanged in content and re-ordered by this result:

1. **O6** — Scene Flow pretraining (~22 h, blocked on dataset size). Now doubly
   motivated: it is both the largest confound and the condition under which this
   capacity finding would have to be re-tested.
2. **E3b stage 2 / lower block counts** — seeds 1 and 2 for arms B and C, and a
   three-block arm. Cheap (~1 h per run) and would convert "not separable at seed
   0" into a claim about runs.
3. **E4-cheap** — alternative cost-volume representations on H2's trained features
   (minutes, no training); still the only live Stage E rationale.
4. **E2b**, **E5** — unchanged, last.
