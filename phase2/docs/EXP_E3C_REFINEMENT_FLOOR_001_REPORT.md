# EXP-E3C-REFINEMENT-FLOOR-001 — does a trained three-block refinement stack match H2?

**Date:** 2026-09-08. **Compute:** 3,592 s (1.0 GPU-hour). **Protocol frozen in
advance:** `EXP_E3C_REFINEMENT_FLOOR_001_PREREGISTRATION.md`.

**Claim tags:** `MEASURED`, `DERIVED`, `INFERRED`, `UNKNOWN`.

**Records:** `EXP-E3C-REFINEMENT-FLOOR-001-ARM-D` and `…-ANALYSIS`. Control and
band reused as measured inputs from `EXP-E3B-REFINEMENT-CAPACITY-001`. Raw data:
`phase2/results/EXP-E3C-REFINEMENT-FLOOR-001/`. Script:
`phase2/scripts/exp_e3c_refinement_floor.py`.

Registered separately from E3b, not as its stage 2: E3b pre-committed stage 2 to
trigger only on an arm outside the band (none was) and to add no new arms.

---

## 1. Verdict, and the caveat that must travel with it

    CAPACITY-REDUCIBLE-TO-3   (per the frozen rule)

`MEASURED` — arm D is **materially different** from the six-block control on both
metrics, but in **opposite directions**: −0.3226 px EPE (better) and +2.0304 D1
points (worse), against a band of 0.2138 px / 1.0000 point. The frozen rule
defines "worse" as both deltas positive, so D is not worse, meets C1–C5, and the
verdict is CAPACITY-REDUCIBLE-TO-3.

**The label overstates what happened, and the number that matters is in the table
below.** Three blocks do not reduce capacity for free: they trade **2.03 D1 points
for 0.32 px of EPE**. That is a real tradeoff crossing a pre-registered threshold,
not a free 45 % MAC saving, and any use of this result must carry it.

All gates passed: arm completed 200 epochs, control stereo-functional, Phase 1
diff empty.

## 2. Results, with E3b's arms for context

`MEASURED` — late window = epochs 150–199, recorded validation protocol, seed 0,
identical recipe; deltas against the six-block control.

| arm | blocks | dilations | params | MACs @368×1232 | late EPE | late D1 | ΔEPE | ΔD1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A control | 6 | 1,2,4,8,1,1 | 423,586 | 56.034 G | 4.0921 ± 0.1000 | 24.9097 ± 1.0821 | — | — |
| B | 5 | 1,2,4,8,1 | 405,090 | 47.677 G | 3.8588 ± 0.0644 | 25.7856 ± 0.7130 | −0.2333 | **+0.8759** |
| C | 4 | 1,2,4,8 | 386,594 | 39.321 G | 4.0028 ± 0.0753 | 26.0708 ± 0.6425 | −0.0893 | **+1.1610** |
| **D** | **3** | **1,2,4** | **368,098** | **30.964 G** | **3.7695 ± 0.0580** | **26.9402 ± 0.6617** | **−0.3226** | **+2.0304** |

`MEASURED` — stereo functionality at epoch 200, all four **STEREO-FUNCTIONAL**:

| arm | right-image | matching-map | entropy | disparity std | matching gradient | wall clock |
|---|---:|---:|---:|---:|---:|---:|
| A (6) | +82.00 | +68.07 | 1.780 | 17.66 | 100 % | 4,361 s |
| B (5) | +79.71 | +63.52 | 1.814 | 18.27 | 100 % | 3,646 s |
| C (4) | +75.42 | +64.19 | 1.815 | 18.03 | 100 % | 3,578 s |
| D (3) | **+81.41** | **+65.79** | 1.718 | 18.72 | 100 % | 3,592 s |

`DERIVED` — three blocks is still emphatically a stereo model: destroying the
right image costs it 81.41 D1 points, more than the six-block control's 82.00 to
within noise, and its matching-map dependence is the second highest of the four.
Whatever three blocks costs, it is not the second camera.

## 3. The finding this run actually produced: D1 degrades monotonically

`MEASURED` — ΔD1 against the control, by block count: **+0.88 (5), +1.16 (4),
+2.03 (3)**. Monotone, and the step from four to three is larger than either
earlier step.

`MEASURED` — ΔEPE over the same series: −0.23, −0.09, −0.32. **Not** monotone and
not ordered by block count.

`DERIVED` — the two metrics are telling different stories, and the D1 story is the
consistent one. Each individual comparison passed E3b's and E3c's rules, but the
**trend across four points is a steady outlier-rate cost that the per-arm
conjunction rule was never designed to detect**. E3b's headline — "four blocks
match six" — remains true under its frozen rule and is now visibly the middle of a
gradient rather than a flat region.

`INFERRED` — removing refinement capacity costs the model precisely where D1
measures and EPE does not: a thin band of pixels near the 3 px / 5 % threshold.
That is the same mechanism H1 v2 identified for the opposite sign, and it is
consistent with less capacity producing slightly blunter disparity edges while the
bulk error distribution improves.

`UNKNOWN` — one seed per arm. Four monotone points from four single runs, with a
measured same-seed D1 noise of 0.127 points, is suggestive; it is not a
significance claim and none is made.

## 4. The confound, unresolved by design

`MEASURED` — arm D is `keep first 3` = dilations 1, 2, 4, so it **drops dilation
8** and shrinks the receptive field as well as the capacity. Arms B and C dropped
only repeated dilation-1 blocks and left the 1-2-4-8 ladder intact.

`DERIVED` — D's +2.03 D1 points therefore **cannot be attributed to capacity
alone**. A blunter wide-context path would produce exactly this signature. The
pre-registration named the disambiguating arm before the run: a three-block
`(1,2,8)` or `(1,4,8)` at identical parameters and MACs.

That follow-up was pre-committed to trigger only if D was materially *worse* or
STEREO-BROKEN. `DERIVED` — neither fired, so it is **not** auto-triggered — but
§3's monotone D1 trend now motivates it on evidence rather than on rule, and it is
the cheapest way to learn whether the cost is capacity or receptive field.

## 5. Limitations

- **One seed per arm**, four arms. The honest phrasing for any single comparison
  is "not separable at seed 0"; for the trend, "monotone across four single runs".
- `UNKNOWN` — two blocks and below.
- `UNKNOWN` — transfer to a pretrained model. 160 training scenes from random
  initialisation, 3.8–4.1 px against the pretrained reference's 1.31 px. A
  capacity floor found at this budget is not the floor a converged model has.
- `UNKNOWN` — latency. `MEASURED` (Phase 1) MACs mispredict latency on this stack
  by 0.67×–142×. The 45 % MAC reduction from six to three blocks is arithmetic;
  training wall clock fell only 4,361 → 3,592 s (−18 %), which is itself evidence
  that MAC savings do not convert one-for-one.
- The band is E3b's single-sample lower bound on same-seed noise, reused here, not
  a confidence interval.
- The verdict label, as §1 states, does not encode the direction split. The
  underlying deltas do.

## 6. Next step — recommendation only

Unchanged in substance from E3b, with one addition:

1. **O6 — Scene Flow pretraining.** Still the largest confound, and now also the
   condition under which both the capacity floor and the D1 trend would have to be
   re-measured.
2. **The (1,2,8) / (1,4,8) three-block arm** (~1 GPU-hour) — separates capacity
   from receptive field, and is the direct follow-up §3 motivates.
3. **Seeds 1 and 2 for the existing arms** (~1 h each) — would turn the monotone
   D1 trend from four single runs into a claim about runs.
4. E4-cheap, then E2b, E5 — unchanged.

Stage E remains formally **CLOSED**; nothing here touches the cost volume.
