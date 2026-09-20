# EXP-CORRESPONDENCE-TR-001 — PRE-REGISTRATION

**Trained-vs-Random Geometric Separation**

Record `phase2/diagnostics/correspondence_tr/20260911T134500Z/`.
Frozen 2026-09-11. **NO TRAINED AGGREGATION HAS BEEN EXECUTED.**

Inference only. No training, no fine-tuning, no weight change, no architecture
change, no new checkpoint. Phase 1 untouched. No historical record modified or
overwritten. ARCH-RATE-001, GEOM-001 and GEOM-002 untouched; their verdicts
unchanged.

**Frozen artefacts, written before this document, hashed over FILE BYTES:**

```
spec.json               sha256 5e6a0bca39408018454fb69e3287f26f8524fc89475ee564a5da5c54a5fa0280
random_reference.json   sha256 8d58c412db5244dc16a8b59de7223ab4def8b70681837cb86928c4a5352e4722
```

---

## 0. PROVENANCE DISCLOSURE — recorded before execution

1. **Trained response curves on THESE scenes `{5,7,8,9}` have never been
   measured** by any experiment. They are genuinely unknown at freeze time.
2. **Disclosed contamination.** GEOM-002 published trained `m_h(t)` curves on
   **different** scenes `{1,2,3,4}` under the **same** construction. The designer
   has read them. They are not used here — not as a threshold, not as a
   comparand, not as a reference — but the designer is not naive about the
   approximate shape a trained curve takes. **This is disclosed, not
   mitigated away.**
3. **The random reference is pre-existing and hash-verified** (§6). It was
   measured before this experiment was conceived and is used exactly as recorded.
4. **No p-value is computed anywhere.** The units are not independent.
5. **The statistic is inherited verbatim** from ARCH-RATE-001, where it was
   defined and applied without any trained result in view.

---

## 1. RANDOM-REFERENCE PROVENANCE — verified before freeze (§6)

`phase2/diagnostics/correspondence_arch_rate/20260911T130507Z/`

| check | result |
|---|---|
| `results.json` sha256 | `1f0463ec08b950d3c1b69804407299f66b4b6cee5c04af51419a061dba7458f1` |
| `construction_spec.json` | `7a4e2341…` **OK** |
| `randomisation_spec.json` | `e52b5b03…` **OK** |
| `classification_spec.json` | `273f7562…` **OK** |
| spec hashes recorded *inside* `results.json` | all three **match** |
| construction | `t = {0,16,32,48,64,80,96}`, crop `1136×272`, `a_R = a_L + t` |
| mask | `64 ≤ y < 208 ∧ 64 ≤ x < 1072`, 145 152 px |
| scenes | `[5,7,8,9]` |
| feature extractors | `POS_6b_seed0/1/2`, role recorded as FEATURE EXTRACTOR ONLY |
| seed list | 32 seeds, 0–31, all present in the raw units |
| units with full raw curves | **384 / 384** |

**ALL PROVENANCE CHECKS PASS.** The reference is frozen into
`random_reference.json` with every raw `α_h`, `α_v` and `range_h` value.

**Regeneration is prohibited.** If trained results are difficult to interpret,
the random arm is **not** re-run.

### The reference distribution (MEASURED, ARCH-RATE-001, N = 384)

| | n | min | p25 | median | p75 | max | mean | sd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `α_h` | 384 | **−0.161307** | −0.049707 | +0.003156 | +0.061659 | **+0.244127** | +0.008772 | 0.078233 |
| `α_v` | 384 | **−0.078097** | −0.012772 | −0.000040 | +0.016012 | **+0.073841** | +0.000940 | 0.021794 |
| `R_h` | 384 | 0.081781 | 0.247512 | 0.353512 | 0.490899 | 1.213284 | 0.391068 | 0.199189 |

Classification tally: `CLIPPED 0 · FLAT 378 · ANTI_CORRELATED 0 ·
NON_MONOTONE 6 · NON_SPECIFIC 0 · QUALIFYING 0`.

### §14 — the seed-18 observation

Five of the six non-flat random units were seed 18, and all six dipped at
`t = 16` before rising. **This is descriptive only.** Seed 18 is **retained in
the reference population exactly as measured** — not excluded, not downweighted,
not treated as an anomaly, not used to construct a special null, and not used to
alter the statistic.

---

## 2. PRIMARY QUESTION

> Under the same synthetic geometric construction, does the trained aggregation
> produce a response distinguishable from the empirically observed random-weight
> population?

**The comparison is TRAINED RESPONSE DISTRIBUTION vs RANDOM RESPONSE
DISTRIBUTION** under identical input construction and measurement. It is not
whether the trained model "looks geometric", not whether the response is
monotone on its own, and no non-zero response is treated as correspondence.

---

## 3. CONSTRUCTION — inherited unchanged

```
left  crop : rows [0,272), cols [0,1136)                FIXED every condition
right crop : horizontal  rows [0,272),  cols [t, 1136+t)
             vertical    rows [t, 272+t), cols [0,1136)
crop 1136 × 272          t ∈ {0,16,32,48,64,80,96}      u = t/16 ∈ {0,…,6}
a_R = a_L + t  ⇒  known disparity d = t px, k_true = t/16 candidates
```

Integer slicing only — **no fill, no padding, no interpolation**. **The sign is
verified from the actual implementation at run time, not assumed.**

**Mask:** `64 ≤ y < 208 ∧ 64 ≤ x < 1072`, 145 152 px, no ground truth. Identical
to the reference.

**Scenes:** `[5,7,8,9]` — **identical to the random reference**, which is required
for an apples-to-apples comparison.

---

## 4. PRIMARY STATISTIC

```
m_h(t)  = median predicted candidate disparity over the fixed mask
α_h, β_h from  m_h(t) = α_h·u + β_h    OLS over ALL SEVEN levels,  u = t/16
```

**FREE INTERCEPT.** The GEOM-002 through-origin form is **prohibited**. The
ARCH-001 z-scored orbit-shape correlation is **prohibited** as primary.

`α_h` is the primary magnitude statistic. **Its definition is inherited verbatim
from ARCH-RATE-001**, where it was fixed with no trained result in view.

### Secondary (recorded, never converted to a new binary claim)

- `α_v`, `β_v` — identical fit on the vertical arm. **A control axis. No
  expectation that `α_v = 0`.**
- `R_h = max(m_h) − min(m_h)` — recorded; **no unregistered threshold applied**.
- First differences `Δ_i = m_h(t_i) − m_h(t_{i−1})`, all six; positive count,
  negative count, min, max.

---

## 5. CLASSIFICATION — inherited verbatim, precedence preserved

```
1 NONFINITE  -> HARD STOP     2 CLIPPED   min m_h ≤ 1.0 or max m_h ≥ 10.0
3 FLAT       range < 1.0      4 ANTI_CORRELATED  α_h < 0
5 NON_MONOTONE               6 NON_SPECIFIC      7 QUALIFYING
```

Thresholds are the **candidate-grid-derived** values frozen in ARCH-RATE-001.
**No new threshold is invented.** `CLIPPED` is evaluated before every ordinary
class, so a clipped response can never be classified monotonic, and a flat
response is never converted into a large slope. **Raw curves are reported
regardless of classification.**

---

## 6. UNIT STRUCTURE

```
PRIMARY   MATCHED: checkpoint c supplies BOTH its feature extractor and its
          aggregation.  3 checkpoints × 4 scenes = 12 units. THE REAL MODELS.

SECONDARY FULL CROSS: 3 extractors × 3 trained aggregations × 4 scenes = 36.
          Closest analogue of the ARCH-RATE 3 × 32 × 4 structure.
          Off-diagonal pairs are chimeras, not real models.
```

**The two populations are never pooled.**

**Units are NOT independent replicates.** They share three checkpoints and four
scenes, and the three checkpoints originate from related training conditions.
**No p-value is computed anywhere.** Results are reported per unit, grouped by
checkpoint, and grouped by scene.

---

## 7. COMPARISON PROCEDURE

Per trained unit, against the 384 random `α_h` values:

- empirical percentile rank;
- empirical CDF position;
- nearest random observations above and below;
- **outside-observed-random-range flag.**

```
KEY DESCRIPTIVE QUANTITY
trained_outside_random_range_rate
   = (# trained units outside the complete observed random α_h range) / (total trained units)
```

Computed identically for `α_v`, `R_h` and the positive-first-difference count —
**reported, never converted into a new binary claim.**

**Distributional reporting is mandatory:** trained median / IQR / min / max,
random median / IQR / min / max, overlap, outside-range fraction, and a
**nonparametric effect size** (rank-biserial correlation from the Mann-Whitney
U statistic) **reported as an effect size only, with no p-value.**

**Explicitly rejected as invalid:** *"0/384 random qualified, therefore any
trained qualification proves learning."* That inference is not used.

---

## 8. WIRING CHECKS — structural only

**No numerical value is required at `t = 0`. The invalid 5.5 expectation is
never reinstated.**

| | check |
|---|---|
| **W1** | the left input is **byte-identical** across all translation conditions (hash-compared) |
| **W2** | the right-image construction matches the frozen spec at every level; crop origin asserted `(0,t)` horizontal, `(t,0)` vertical |
| **W3** | record `k_true = t/16`; **the model output is NOT required to equal it** |
| **W4** | **deterministic repeat** of one complete trained unit **after process restart**; the complete response vector must be **bit-identical**. Failure ⇒ HARD STOP |

---

## 9. SEARCH-FREE CONTROL

`NEG_shift_none`, run fresh under identical conditions (not previously measured
on these scenes). **Purpose:** to test whether the trained geometric statistic
requires the candidate-dependent construction. **Reported separately. Never mixed
into the random-weight null. It generates no significance threshold.**

---

## 10. DECISION LOGIC — no arbitrary pass threshold

| case | condition | interpretation |
|---|---|---|
| **CASE A — CLEAR SEPARATION** | trained distribution substantially displaced, little/no overlap, high outside-range fraction, **and consistent across all three checkpoints and all four scenes** | "The trained aggregation exhibits a geometric response that is not reproduced by the sampled random aggregation population under this synthetic construction." Warrants a stronger investigation. **Does not prove correct disparity search.** |
| **CASE B — OVERLAP** | trained responses substantially overlap the random population | "Not distinguishable from the sampled random population under this statistic/construction." **No correspondence claim.** |
| **CASE C — AMBIGUOUS** | partial separation, mixed checkpoints, strong scene dependence, clipping, or other structure | "Trained-vs-random separation is inconclusive under this measurement." **Do not tune or re-run.** |

**Heterogeneity rule (§19):** if one checkpoint separates while others overlap ⇒
**MIXED ⇒ CASE C**, never overall success. If one scene drives the result, report
it explicitly. **Heterogeneity is never averaged away.**

---

## 11. CRITICAL LANGUAGE CONSTRAINT (§18)

A trained value outside the finite random range is **not** automatically evidence
of learning. The reference contains **384 sampled parameterisations, not the
space of random weights**.

Required phrasing: **"outside the observed random population."**
Prohibited: *"impossible under random weights"*, *"proves learned
correspondence."*

---

## 12. HARD STOPS

Trained weights modified · trained aggregation retrained · random reference not
hash-verifiable · construction differs · translation levels differ · mask differs
· preprocessing differs · **deterministic repeat fails** · trained curves
inspected before freeze · post-hoc statistic introduced · post-hoc threshold
introduced · historical results overwritten.

```
On any hard stop:  VERDICT = NO DETERMINATION
Do not continue merely to obtain a result. Do not rescue post hoc.
```

Every hard stop must **halt execution**, not merely be recorded.

---

## 13. CLAIM CEILING

Maximum available claim:

> "Under the specified synthetic translation construction, the trained
> aggregation response **is / is not** distinguishable from the empirically
> sampled random aggregation population."

**Unavailable from this experiment alone:** genuine stereo correspondence;
correct disparity estimation; true disparity search; correct correspondence on
natural scenes; metric depth correctness; generalisation; biological/physical
scene correspondence.

---

## 14. OUTPUTS

Present now (design artifacts only): `PREREGISTRATION.md`, `DESIGN_AUDIT.md`,
`EXECUTION_PROMPT.md`, `spec.json`, `random_reference.json`, `frozen.sha256`,
`freeze_tr.py`.

At execution only: `results.json`, `results.csv`, `AUDIT.md`, `RESULTS.md`,
`run.log`, `ENVIRONMENT.txt`, `RELATED_RUNS.md`, and the harness with its sha256.

**No follow-up experiment is designed or launched after this one.**
