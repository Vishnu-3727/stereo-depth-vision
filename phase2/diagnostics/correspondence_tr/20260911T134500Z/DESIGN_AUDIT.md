# DESIGN AUDIT — EXP-CORRESPONDENCE-TR-001

**Freeze audit. No trained aggregation has been executed.** No training, no
`results.json`, no `RESULTS.md`. Phase 1 untouched; no historical record
modified or overwritten.

Tags: **DERIVED**, **PREREGISTERED**, **MEASURED**, **INFERRED**, **UNKNOWN**.

---

## 1. THE PROVENANCE GATE (§6) — passed before anything else

**MEASURED.** The random reference was hash-verified *before* the record
directory existed:

```
results.json              1f0463ec08b950d3c1b69804407299f66b4b6cee5c04af51419a061dba7458f1
construction_spec.json    7a4e2341…  OK
randomisation_spec.json   e52b5b03…  OK
classification_spec.json  273f7562…  OK
spec hashes recorded INSIDE results.json   -> all three match
384 / 384 units carry full raw m_h and m_v curves
32 seeds (0–31) all present · scenes [5,7,8,9] · mask 145 152 px
```

Had any check failed, the protocol required **HARD STOP — NO DETERMINATION**
with no silent regeneration. It did not fail. The reference is frozen verbatim
into `random_reference.json` (`8d58c412…`), including **every** raw `α_h`, `α_v`
and `range_h` value, so no later step can quietly reconstruct it.

**PREREGISTERED:** regeneration of the random arm is prohibited, whatever the
trained results look like.

---

## 2. STATISTIC — why it is not post-hoc (§4)

| statistic | status |
|---|---|
| GEOM-002 through-origin slope | **prohibited** — provably maps a step of height `h` to `0.2308h`, indistinguishable from a ramp |
| ARCH-001 z-scored orbit-shape correlation | **prohibited as primary** — its reference interval spanned 97.4 % of the attainable range |
| **free-intercept `α_h`** | **adopted** |

**DERIVED, and the decisive point:** `α_h` was defined, frozen and applied in
ARCH-RATE-001 **with no trained result in view**, and produced a meaningful,
non-degenerate distribution there (384 values, range `[−0.161, +0.244]`,
sd 0.078). Reusing it here is inheritance, not selection. Its definition is
hash-bound to `classification_spec.json` `273f7562…`.

**A free intercept maps every constant step to `α = 0` exactly** — the property
the through-origin form lacked.

---

## 3. UNIT STRUCTURE — a resolved tension (§7 vs §8)

§7 says *"use the **corresponding** frozen feature extractors"* (matched
pairing). §8 says *"preserve the ARCH-RATE structure as closely as possible"* and
names `feature extractor × scene × trained checkpoint` (a full cross).

**These conflict.** ARCH-RATE crossed 3 extractors × 32 random aggregations
because a random aggregation has no natural extractor. A *trained* aggregation
does.

**PREREGISTERED resolution — both, never pooled:**

```
PRIMARY   MATCHED  : checkpoint c supplies BOTH extractor and aggregation
                     3 × 4 scenes = 12 units.  THE REAL MODELS.
SECONDARY CROSS    : 3 extractors × 3 aggregations × 4 scenes = 36 units.
                     Closest analogue of the ARCH-RATE structure.
                     Off-diagonal pairs are CHIMERAS, not real models.
```

**INFERRED:** reporting only the cross would let 24 chimeric units dilute or
inflate the 12 real ones; reporting only the matched set would abandon the
structural analogue §8 asks for. Freezing both, with pooling prohibited, honours
both instructions without inflating N.

**PREREGISTERED (§8, §13):** units are **not** independent replicates — they
share three checkpoints and four scenes, and those checkpoints originate from
related training conditions. **No p-value is computed anywhere.** The effect size
is rank-biserial from Mann-Whitney U, reported **as an effect size only**.

---

## 4. WHAT COULD INVALIDATE A SEPARATION, AND HOW IT IS GUARDED

| threat | guard |
|---|---|
| trained response railed at a candidate boundary, faking a slope | `CLIPPED` evaluated **before every ordinary class**, threshold inherited unchanged from ARCH-RATE-001 |
| a flat response turned into a large slope by numerical instability | `FLAT` evaluated before slope-based classes; raw curves reported regardless |
| separation driven by one checkpoint | heterogeneity rule (§19): one checkpoint separating while others overlap ⇒ **MIXED ⇒ CASE C**, never success |
| separation driven by one scene | per-scene grouping mandatory; explicitly reported |
| "0/384 random qualified ⇒ any trained qualification proves learning" | **explicitly rejected as invalid** in `spec.json`; the comparison is distributional |
| a finite random sample mistaken for the space of random weights | mandated phrasing **"outside the observed random population"**; the alternatives are prohibited |
| construction sign assumed rather than checked | **W2 asserts the crop origin at every level**; the sign is verified from the implementation, not assumed |
| a numeric `t = 0` expectation reinstated | **explicitly forbidden**; only structural checks W1–W4 |
| non-determinism | **W4**: a complete trained unit is repeated **after process restart** and must be **bit-identical**; failure ⇒ HARD STOP |

---

## 5. THE DISCLOSED CONTAMINATION

**MEASURED:** trained curves on scenes `{5,7,8,9}` have never been measured by
any experiment — they are genuinely unknown at freeze time.

**Disclosed:** GEOM-002 published trained `m_h(t)` on **different** scenes
`{1,2,3,4}` under the **same** construction, and the designer has read them.
Those values are not used as a threshold, comparand or reference anywhere — but
the designer is **not naive** about the approximate shape a trained curve takes.

**INFERRED:** this bounds what the experiment can claim. It cannot be
confirmatory in the strict sense. It is a **preregistered descriptive comparison
against a pre-existing, hash-verified reference** — which is the strongest form
available given that the three checkpoints are the entire population and cannot
be replaced without training.

**The strongest mitigations are structural, not rhetorical:** the reference
existed before this experiment was conceived; the statistic was inherited from
it; the scenes are its scenes; and no threshold in this design originates from
any trained observation.

---

## 6. SEARCH-FREE CONTROL (§15)

§15 permits it *"if it is already available under the exact frozen conditions."*
**It is not** — GEOM-002 ran `shift="none"` on scenes `{1,2,3,4}`, not these.

**PREREGISTERED:** run it fresh under identical conditions, report it
**separately**, never mix it into the random-weight null, and derive no
threshold from it. Its purpose is narrow: to test whether the trained geometric
statistic **requires** the candidate-dependent construction.

---

## 7. COMPUTE AND THE MEASUREMENT PATH

```
156 cost volumes   (3 extractors × 4 scenes × 13 conditions; seed-independent, cached)
504 trained readout passes        (36 cross units × 14)
 56 search-free readout passes    (4 scenes × 14)
 14 deterministic-repeat passes   (W4, after process restart)
```

The 14-per-unit readout count **replicates ARCH-RATE-001's procedure exactly**,
including its redundant recomputation of the shared `t = 0` condition on the
vertical axis. **DERIVED:** that pass re-runs the same cached volume through the
same aggregation, so `m_v(0) ≡ m_h(0)`; ARCH-RATE verified this bit-identically
in 384/384 units. Replicating the procedure keeps the measurement path identical
between the two populations, which the comparison requires.

---

## 8. VERDICT

```
TR-001-DESIGN-READY
```

Provenance gate passed (§1). Statistic inherited, not selected (§2). Unit-structure
conflict resolved without inflating N (§3). Every identified threat to a
separation claim is guarded by a rule frozen before execution (§4). Contamination
disclosed rather than mitigated away (§5). Search-free control scoped and
isolated (§6).

**No trained aggregation has been executed. Awaiting the frozen
`EXECUTION_PROMPT.md`.**
