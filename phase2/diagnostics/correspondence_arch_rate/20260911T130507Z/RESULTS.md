# EXP-CORRESPONDENCE-ARCH-RATE-001 — RESULTS

**How often do untrained aggregation weights reproduce the synthetic horizontal
correspondence signature?**

Record `phase2/diagnostics/correspondence_arch_rate/20260911T130507Z/`.
Pre-registered in `PREREGISTRATION.md`, frozen before execution and **not
modified** (last write 13:08:09Z; execution began 13:11Z).

**This is a descriptive measurement. The result is a rate. There is no pass/fail
rule anywhere in this document.**

Inference only — 156 cached cost volumes, 5 376 aggregation/readout passes,
56.2 s. **No training. No trained aggregation was ever executed.** Phase 1
untouched; GEOM-001 and GEOM-002 untouched, verdicts unchanged.

---

## THE RESULT

```
QUALIFYING RATE  =  0 / 384  =  0.0000        (resolution 1/32 = 0.03125)

CLIPPED            0
FLAT             378
ANTI_CORRELATED    0
NON_MONOTONE       6
NON_SPECIFIC       0
QUALIFYING         0
```

**Not one of 384 untrained aggregations reproduced the signature. 378 of 384
produced essentially no candidate-axis response at all.**

---

## 1. FREEZE VERIFICATION — 10/10 OK before the first model instantiation

`PREREGISTRATION.md` present · `construction_spec.json` `7a4e2341…` OK ·
`randomisation_spec.json` `e52b5b03…` OK · `classification_spec.json`
`273f7562…` OK · `sha256sum -c frozen.sha256` all three **OK** · all three
digests verbatim in the preregistration · no `RESULTS.md` · `DESIGN_AUDIT.md`
records the boundary qualification · three feature-extractor sha256 verify ·
scenes `[5,7,8,9]` disjoint from `{27,0,31,6} ∪ {1,2,3,4}`.

All three digests re-verified **over file bytes** inside the harness before any
model was built.

## 2. SANITY CHECKS

| check | result |
|---|---|
| crops | one shape only, `(272, 1136)` |
| volumes | `(1, 32, 12, 17, 71)`, **156 built** (cached; seed-independent) |
| left crop fixed | **1 distinct hash per scene** across all 13 conditions |
| mask | 145 152 px, invariant |
| seed distinctness | all 32 seeds' every parameter tensor hash distinct |
| aggregation is random | asserted **384/384** times against the trained hash |
| **trained aggregation executed** | **false** |
| training | **none** — no optimizer, no `.backward()`, no `.step()` |
| no fill / interpolation | every window asserted inside the source; all `t` multiples of 16 |
| `CLIPPED` count | **0** — see §4 |
| determinism | full block held |

## 3. PRIMARY RESULT, BROKEN DOWN

**Uniform. No breakdown shows any qualifying unit.**

| feature extractor | rate | | scene | rate |
|---|---|---|---|---|
| POS_6b_seed0 | 0 / 128 | | 000165_10 | 0 / 96 |
| POS_6b_seed1 | 0 / 128 | | 000167_10 | 0 / 96 |
| POS_6b_seed2 | 0 / 128 | | 000168_10 | 0 / 96 |
| | | | 000169_10 | 0 / 96 |

### Per-condition tallies

```
Q1  α_h > 0                      195 / 384      ≈ a coin flip
Q2  strictly increasing            2 / 384
Q3  |α_h| > |α_v|                324 / 384      see the caveat below
```

**Caveat on Q3 (descriptive).** Q3 passes at 324/384 while **every** `α` is
within noise of zero. It is comparing two near-zero numbers, so its high pass
rate carries no information here. Reported for completeness, not as a finding.

### Free-intercept slopes — the anchor under ideal correspondence is **1.0**

| | min | median | mean | max | sd |
|---|---:|---:|---:|---:|---:|
| `α_h` | −0.1613 | **+0.0032** | +0.0088 | **+0.2441** | 0.0782 |
| `α_v` | −0.0781 | −0.0000 | +0.0009 | +0.0738 | 0.0218 |

```
|α_h| ≥ 0.5 :   0 / 384
|α_h| ≥ 1.0 :   0 / 384        (1.0 is the correspondence anchor)
```

**The largest slope produced by any untrained aggregation is 0.244 — under a
quarter of the anchor.**

### Response range over the 7 levels (a 6-candidate sweep)

```
min 0.0818 · p25 0.2475 · median 0.3535 · p75 0.4909 · max 1.2133   candidates
units with range ≥ 1.0 candidate:  6 / 384
```

Absolute bounds: `min_h ∈ [2.806, 7.525]`, `max_h ∈ [2.980, 7.862]`, against
rails at 0 and 11.

## 4. THE ZERO RATE IS NOT A RAILING ARTEFACT

```
CLIPPED = 0 / 384
```

`CLIPPED` was evaluated **second in the frozen precedence, before every ordinary
failure class**, precisely so a railed response could never be folded into
failure. **No unit came within one candidate of either rail** — the observed
responses sit at 2.8–7.9 on a 0–11 axis, comfortably interior.

**The low rate is therefore a genuine absence of response, not saturation.** This
was the one identified non-termination risk in the gate audit, and the frozen
precedence removed it.

## 5. SAMPLE CURVES

`m_h(t)` in candidate units; the true disparity is `t/16 = 0,1,2,3,4,5,6`.

```
POS_6b_seed0 000165_10 seed 0   m_h  5.275 5.031 5.070 5.087 5.134 5.180 5.217
                                m_v  5.275 5.210 5.163 5.142 5.153 5.111 5.145
POS_6b_seed0 000165_10 seed 3   m_h  3.830 3.771 3.877 3.832 3.786 3.833 4.079
                                m_v  3.830 3.750 3.925 3.798 3.901 3.894 3.944
```

The imposed disparity sweeps six candidates; the response moves by roughly a
third of one.

### The six non-flat units

All six exceed the 1.0-candidate flat cut but fail strict monotonicity. Five of
the six are **seed 18**, across different extractors and scenes.

| extractor | scene | seed | range | α_h | `m_h(t)` |
|---|---|---:|---:|---:|---|
| seed0 | 000168_10 | 18 | 1.029 | +0.2288 | 6.00 5.57 5.71 6.08 6.45 6.54 6.60 |
| seed0 | 000169_10 | 18 | 1.090 | +0.2441 | 5.74 5.40 5.57 5.96 6.34 6.49 6.48 |
| seed1 | 000165_10 | 18 | 1.213 | +0.2379 | 5.59 5.24 5.51 5.94 6.16 6.19 6.46 |
| seed1 | 000167_10 | 15 | 1.054 | +0.1933 | 5.97 5.27 5.57 5.74 5.89 6.02 6.33 |
| seed1 | 000167_10 | 18 | 1.156 | +0.2086 | 5.64 5.23 5.66 5.83 5.98 6.12 6.39 |
| seed1 | 000168_10 | 18 | 1.069 | +0.1858 | 5.75 5.37 5.81 6.08 6.19 6.15 6.44 |

Every one dips at `t = 16` before rising — failing monotonicity at the first
step. **MEASURED**, reported without interpretation.

---

## 6. THE PREREGISTERED PREDICTION IS FALSIFIED

`PREREGISTRATION.md` §2 recorded three tiers. The measurement resolves them:

| tier | content | outcome |
|---|---|---|
| **DERIVED** | `V(k,w) = F(k−τ, w+τ)` with `F` independent of `τ`; exact equivariance at 2 / 1 / 0 candidates for `τ = 0 / 1 / ≥2` | unaffected — this is algebra and remains true |
| **PREDICTED** | approximate interior equivariance **should** produce a **high** qualifying rate | **FALSIFIED. The measured rate is 0/384.** |
| **UNKNOWN** | how strongly the `D = 12` padding boundary alters the response in practice | **RESOLVED: it dominates.** Untrained aggregations produce a median response range of 0.354 candidates across a 6-candidate sweep |

**The three-tier framing did its job.** The prediction was mine, it was recorded
before execution as resting on an *approximate* argument, and it was wrong. Had
it been frozen as an exact expectation — as the GEOM-002 `t = 0 → 5.5` anchor
was — this would have fired a hard stop against correct behaviour.

**Why the derivation over-predicted (INFERRED, from the frozen §2 count):** exact
equivariance survives at **zero** candidates for `τ ≥ 2`, i.e. five of the six
sweep levels. The receptive field is 11 over `D = 12`; the padding boundary
touches almost every candidate. With untrained weights there is nothing to
sharpen the cost profile at the match, so the boundary term dominates the
translation term entirely.

This is consistent with ARCH-001's independent measurement on a different
intervention — random aggregations gave candidate-orbit ranges of 0.09–1.42
against trained 5.36–6.39 — though that was a different experiment and is cited
only as context.

---

## 7. INTERPRETATION — per the frozen §10

The measured rate is **low**, so the frozen interpretation is:

> "Untrained weights of this architecture rarely reproduce the signature."
> ⇒ the boundary caveats of §2 dominate, **and the observable may be
> diagnostic** ⇒ a trained-arm experiment becomes warrantable **under a separate
> preregistration**.

**Not claimable, under any outcome and explicitly not now:** anything about the
trained models; correspondence, learned or architectural; genuine disparity
search; disparity correctness; real-scene correspondence; generalisation.

**A rate of 0/384 does not establish that trained models exhibit correspondence.
The trained arm was never measured.** It establishes only that the horizontal
ramp is **not** routinely produced by arbitrary aggregation weights — which is
what the synthetic self-pair line needed to know before spending a trained-arm
experiment on it.

**Scope:** framework-default initialisation at this scale, three feature
extractors, four scenes, one construction.

---

## 8. PROTOCOL DEVIATIONS — one, minor, with zero numerical impact

**Forward-pass count: 5 376 executed against 4 992 specified in §12.**

The readout loop iterates `2 axes × 7 levels = 14` per unit and does not skip the
shared `t = 0` condition for the vertical axis — only the *volume build* skips
it. The extra 384 passes re-run the **same cached volume** through the **same
aggregation**.

**Verified:** `m_v(0) == m_h(0)` **bit-identically in 384/384 units**. The
redundant pass changed no value and could not have: it is the same computation on
the same tensor.

**Reported rather than corrected.** No statistic, classification, tally or rate
is affected. Every other frozen count matched exactly: 156 volumes, 384 units,
32 seeds, 145 152 mask pixels, one crop shape.

No other deviation. No checkpoint, seed, scene, crop, mask, `t` level,
construction, axis definition, statistic, classification order, threshold or
interpretation rule was changed.

## 9. PROVENANCE

`PREREGISTRATION.md` unmodified. No training. No trained aggregation executed. No
historical record modified; GEOM-001 and GEOM-002 untouched. `HEAD 58e8a19`;
`phase-1-frozen b4207e5`; `phase2/` untracked, so no cryptographic versioning is
claimed.

**Raw-data requirement satisfied:** every `m_h(t)` and `m_v(t)` for all 384 units
is published in `results.json`, so any reader can reclassify under an alternative
definition of `FLAT` or `CLIPPED` without re-running anything.

## 10. FILES

`PREREGISTRATION.md`, `DESIGN_AUDIT.md`, `EXECUTION_PROMPT.md`, `RESULTS.md`
(this file), `results.json`, `run.log`, `ENVIRONMENT.txt`, `RELATED_RUNS.md`,
`construction_spec.json`, `randomisation_spec.json`, `classification_spec.json`,
`frozen.sha256`, `freeze_rate.py`, `run_arch_rate.py`.
