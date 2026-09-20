# EXP-E1-ERROR-PARTITION-001 — where does H2's remaining error come from?

**Date:** 2026-09-08. **Compute:** ~4 minutes total, no training. **Protocol
frozen in advance:** `EXP_E1_ERROR_PARTITION_001_PREREGISTRATION.md`.

**Claim tags:** `MEASURED`, `DERIVED`, `INFERRED`, `UNKNOWN`.

**Records** (none overwritten):

| ID | Role |
|---|---|
| `EXP-E1-ERROR-PARTITION-001` | the pre-registered run: affine oracle |
| `EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE` | **VOID** — diverging fixed point, see its `NOTE.md` |
| `EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE-RUN2` | crashed before measuring, see its `NOTE.md` |
| `EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE-RUN3` | self-scoring fixed-point oracle |
| `EXP-E1-ERROR-PARTITION-003-HELDOUT-ORACLE` | held-out fixed-point oracle — **the load-bearing run** |

Raw data: `phase2/results/EXP-E1-ERROR-PARTITION-001/{partition,inverse_oracle,inverse_oracle_run3,heldout_oracle}.json`.
Scripts: `exp_e1_error_partition.py`, `exp_e1_inverse_oracle.py`,
`exp_e1_heldout_oracle.py`. Logic tests: `phase2/tests/test_e1_error_partition.py`
(5 tests; full suite now **158 passing**).

---

## 1. Headline

    E1 DOES NOT RETURN A CLEAN PARTITION.

`MEASURED` — a perfect matching output removes **29.3 % / 41.5 / 32.9 %** of H2's
pooled EPE (seeds 0 / 1 / 2) on the honest, held-out measurement. The
pre-registered rule reads that as DOWNSTREAM-LIMITED, MIXED, DOWNSTREAM-LIMITED —
**NO CONSISTENT VERDICT**, with all three EPE ratios (0.707, 0.585, 0.671)
clustered around the 0.65 threshold.

`DERIVED` — roughly **one third** of H2's remaining error is attributable to the
matching stage and roughly **two thirds** is imposed downstream, by these
refinement weights, on every seed. That is a real number and it is not a verdict.

## 2. Why there are three runs

The answer depends entirely on how much capacity the oracle is given, and that
was not appreciated when the protocol was frozen. The three oracles bracket it:

| oracle | what it is | EPE removed | pre-registered verdict |
|---|---|---:|---|
| affine (`001`, primary as pre-registered) | one slope + offset per model, fitted to the model's own convention — the **smoothest** oracle, hence a **lower bound** | 7.8 / 12.1 / 8.7 % | DOWNSTREAM-LIMITED ×3 |
| self-scoring fixed point (`002-RUN3`) | per-pixel search, but tuned at exactly the `gt > 0` pixels it is then scored on — an **upper bound**, and leaky | 61.4 / 69.6 / 61.5 % (still descending) | MIXED, MATCHING-LIMITED, MIXED |
| **held-out fixed point (`003`)** | per-pixel search steered by a random half of the `gt > 0` pixels, scored on the other half; converges by iteration 12–16 | **29.3 / 41.5 / 32.9 %** | DOWNSTREAM-LIMITED, MIXED, DOWNSTREAM-LIMITED |

`DERIVED` — the pre-registered primary (affine) was **under-powered by a factor of
three to four**. Its DOWNSTREAM-LIMITED verdict is reported here as recorded, and
is **not** the conclusion of this experiment.

## 3. Run 001 — the affine oracle, as pre-registered

`MEASURED` (`partition.json`, all 40 `hailo_val` scenes, pooled over `gt > 0`,
3,802,797 pixels):

| model | variant | EPE (px) | D1 (%) |
|---|---|---:|---:|
| H2 | as trained | 2.400 | 16.72 |
| H2 | `initial :=` itself | 2.400 | 16.72 |
| H2 | `initial := a·gt+b` where gt>0 | 2.213 | 16.35 |
| H2 | `initial := a·gt+b` densified | 2.597 | 34.40 |
| H2 | `initial := a·gt+b` shuffled | 15.152 | 88.06 |
| SEED1 | as trained | 2.319 | 17.25 |
| SEED1 | `initial :=` itself | 2.319 | 17.25 |
| SEED1 | `initial := a·gt+b` where gt>0 | 2.039 | 14.60 |
| SEED1 | `initial := a·gt+b` densified | 2.291 | 26.19 |
| SEED1 | `initial := a·gt+b` shuffled | 15.152 | 88.34 |
| SEED2 | as trained | 2.556 | 19.76 |
| SEED2 | `initial :=` itself | 2.556 | 19.76 |
| SEED2 | `initial := a·gt+b` where gt>0 | 2.334 | 18.63 |
| SEED2 | `initial := a·gt+b` densified | 2.649 | 34.24 |
| SEED2 | `initial := a·gt+b` shuffled | 15.031 | 88.07 |

**All four pre-registered gates passed.** `MEASURED` — H2's as-trained figures
reproduce the recorded `h2_mechanism/pooled.json` values to **4.2e-4 px** and
**2.2e-3 D1 points**; identity substitution is bit-exact (ΔEPE 0.0) for every
model; the shuffled negative control is far worse than as trained; all three
checkpoint SHA-256s are unchanged.

`MEASURED` — the fitted conventions: H2 `initial = 0.09516·gt + 1.014`
(R² 0.863, residual 0.686 candidates), SEED1 `0.06332·gt + 1.101` (R² 0.890,
residual 0.402), SEED2 `0.07683·gt + 1.017` (R² 0.862, residual 0.556).
`DERIVED` — one candidate is ≈10.5 px (H2), ≈15.8 px (SEED1), ≈13.0 px (SEED2)
in each model's own learned convention.

`MEASURED` — the densified variant is **worse** than the sparse one on D1
(34.40 vs 16.35 for H2), so the nearest-valid fill injects real error, exactly as
the pre-registration warned. `MEASURED` — 0.53 % (H2), 0.00 % (SEED1), 0.0002 %
(SEED2) of affine-oracle values fell outside the soft-argmin's own [0, 11] range,
so off-manifold inputs are **not** a material weakness of run 001.

## 4. The failed run, kept because of how it failed

`EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE` implemented the fixed point as
`initial ← clamp(initial + (gt − final), 0, 11)`, adding a **pixel**-valued
residual to a **candidate**-valued input. `MEASURED` — one candidate is 10–16 px,
so the step overshot by an order of magnitude and the iteration diverged
monotonically: EPE 2.400 → 22.482 (H2), 2.319 → 29.153 (SEED1), 2.556 → 23.156
(SEED2).

`DERIVED`, and this is the part worth keeping: because every iterate was worse
than the starting point, the best iterate *was* iteration 0, the ratio was
exactly 1.000, and the frozen rule returned **DOWNSTREAM-LIMITED** — apparently
confirming run 001. **A diverging probe produced a false confirmation.** The
missing safeguard was a gate asserting the search actually improves before its
output is read as a verdict; it was added, and both later runs carry it. The
record is kept with a `NOTE.md` and must never be cited as evidence.

This is the second time in Phase 2 that an uncalibrated instrument nearly wrote
a conclusion (the first was the epoch-10 stereo gate). Constraint **C7** of the
evidence review already covered it; this run shows C7 applies to probes, not only
to training gates.

## 5. Run 003 — the held-out oracle (load-bearing)

`MEASURED` — `gt > 0` pixels split 50/50 at random (seed 0); the fixed point is
steered by the FIT half only and scored separately on both halves; α = 1.0,
16 iterations, step scaled by each model's measured slope.

| model | as trained | FIT best | EVAL best (iteration) | EVAL ratio | verdict |
|---|---:|---:|---:|---:|---|
| H2 | 2.398 | 0.765 | **1.696** (12) | 0.707 | DOWNSTREAM-LIMITED |
| SEED1 | 2.317 | 0.608 | **1.356** (16) | 0.585 | MIXED |
| SEED2 | 2.552 | 0.795 | **1.712** (12) | 0.671 | DOWNSTREAM-LIMITED |

`MEASURED` — the EVAL trace **converges** (H2: 2.398, 2.098, 1.952, …, 1.696 at
12, then rises slightly to 1.703 at 16), so this is not a truncated descent like
RUN3's. All gates passed: finite throughout, iteration 0 agrees across halves to
0.005 px, the search improves the FIT half, checkpoints unchanged.

`DERIVED` — the gap between FIT (0.61–0.80 px) and EVAL (1.36–1.71 px) is the
part of the oracle's advantage that does **not** generalise one pixel sideways.
`INFERRED` — that gap is what a real matching stage would have to earn with
structure rather than with per-pixel knowledge.

`UNKNOWN` — refinement is convolutional, so a corrected FIT pixel still helps its
EVAL neighbours; the EVAL number therefore remains an **upper** bound on what a
realizable matching stage buys. The true value is at most 29–42 % and at least the
affine oracle's 8–12 %.

## 6. What this changes

1. `DERIVED` — **Stage E cannot be opened on E1's verdict, because there isn't
   one.** The pre-registered rule returns NO CONSISTENT VERDICT on the honest
   measurement.
2. `DERIVED` — **the downstream share is the larger one on every seed** (58–71 %
   of pooled EPE survives a perfect matching input). If effort is to be spent
   anywhere, the evidence points downstream — at the refinement stage that is
   also 90.6 % of MACs and 73.3 % of GPU time — rather than at the cost volume.
3. `DERIVED` — **cost-volume candidate work is now doubly unattractive**: it
   addresses at most a third of the error and 4.2 % of the arithmetic.
4. `UNKNOWN`, and this is the honest limit — every number here is about *these*
   refinement weights. A refinement stage **retrained** against a better matching
   input could move the partition in either direction. Nothing measured here
   bounds that.
5. `MEASURED` — the disparity-range ceiling was already excluded (0 % of
   `hailo_val` ground truth above 176 px), and off-manifold oracle inputs are
   excluded too (≤ 0.53 %). Two candidate explanations are now closed.

## 7. Next experiment

    NEXT: E3 -- structured inference-time ablation of H2's refinement stack.

Drop residual blocks singly and in suffixes from the trained H2/SEED1/SEED2
weights, score pooled EPE/D1 on the 40 scenes, recount MACs. It needs **no
training**, it targets the 58–71 % share this experiment just localised and the
90.6 % of MACs Phase 1 measured, and its output is directly a deployment number.

Deliberately **not** next: E2 (read-out temperature) and E4 (volume
representation) both address the smaller share and both need a retrain to be
decisive. E5 stays last.

The decisive form of E1 — retrain refinement against a strong matching input and
re-measure the partition — is a real training experiment and should not be run
until E3 has said how much refinement capacity is actually needed.

**Stage E stays closed.** E1 was supposed to open or close it and did neither;
what it produced instead is a measured split (≈⅓ upstream, ≈⅔ downstream) and a
better-aimed next question.

## 8. Limitations

- Three seeds, one dataset slice (40 `hailo_val` scenes), one budget. No
  significance test, none claimed.
- Every oracle here is an upper bound of some kind; the affine one is a lower
  bound. The truth is bracketed, not pinned.
- The oracles substitute at the matching → refinement boundary only. Errors that
  originate in the feature extractor and survive into the aggregated cost are
  counted as "matching", not separated further.
- `gt > 0` covers ~20 % of pixels; both fixed-point oracles act only there.
- The pre-registered verdict thresholds (0.35 / 0.65) were frozen before any
  oracle number existed, and are reported as they fell, including where they
  disagree across seeds.
