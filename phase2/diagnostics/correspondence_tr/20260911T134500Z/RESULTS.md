# RESULTS — EXP-CORRESPONDENCE-TR-001

## Trained-vs-Random Geometric Separation

**Verdict: `CLEAR-SEPARATION`** — on the preregistered primary statistic, with
qualifications stated in §6 and §7 that a reader must carry with the number.

Tags: **MEASURED** (came out of this run), **DERIVED** (follows from source or
algebra), **INFERRED** (my reading), **UNKNOWN**.

---

## 1. WHAT WAS FROZEN, AND WHEN

| artifact | sha256 |
|---|---|
| `spec.json` | `5e6a0bca39408018454fb69e3287f26f8524fc89475ee564a5da5c54a5fa0280` |
| `random_reference.json` | `8d58c412db5244dc16a8b59de7223ab4def8b70681837cb86928c4a5352e4722` |
| `PREREGISTRATION.md` | `a9dd4bba226c6a4fd9e9767bab96a401f21c74890d21f181c922f3fe35be243b` |
| `DESIGN_AUDIT.md` | `21856ae1851761424fbfd716e036915668c5cf42cc91e3548700be7bf971b715` |
| `EXECUTION_PROMPT.md` | `671e6546dc5243d371669971c05ba7aa7804337289d7bfd1c9367874aed1718b` |
| `PRE_EXECUTION_ADDENDUM.md` | `9d7d64704b7b7697bb647a198eaadd25b6dc7bce4e2acda937f4b1cbee376cce` |

Every one verified over **file bytes** before the first model was instantiated.
`ARCH-RATE-001/results.json` re-verified at its source
(`1f0463ec08b950d3c1b69804407299f66b4b6cee5c04af51419a061dba7458f1`), its three
internal spec hashes checked, 384 units confirmed.

**The random population was not re-run, re-scored or regenerated.** Seed 18 was
not excluded, downweighted or specially treated.

**One freeze defect was found and resolved BEFORE execution** — see
`PRE_EXECUTION_ADDENDUM.md` and §8 below. Nothing was rewritten to hide it.

---

## 2. THE PRIMARY RESULT

`α_h` = free-intercept OLS slope of the median predicted candidate disparity on
`u = t/16`, over `T_FIT = {16,32,48,64,80,96}` — **bit-identical in definition to
the 384 frozen random values**, verified by reproducing 32 of them exactly.

### PRIMARY — 12 MATCHED units (checkpoint supplies both extractor and aggregation)

```
                      median      IQR                 min        max
trained  alpha_h      +1.0014   [+0.9644, +1.2010]  +0.9534   +1.2246
random   alpha_h      +0.0032   [-0.0497, +0.0617]  -0.1613   +0.2441

outside the observed random population :  12 / 12  = 1.0000
overlap with the observed random range :   0 / 12  = 0.0000
rank-biserial (effect size only)       :  +1.0000
```

The smallest trained value (`+0.9534`) exceeds the largest of 384 random values
(`+0.2441`) by a factor of **3.9**. There is no adjacent random observation above
any trained unit — the nearest random value below is the population maximum for
all twelve.

### Consistency — the CASE A requirement, checked explicitly

| checkpoint | n | α_h range | outside |
|---|---|---|---|
| `H2_seed0` | 4 | `+1.1994 … +1.2246` | 4/4 |
| `H2_seed1` | 4 | `+0.9534 … +0.9663` | 4/4 |
| `H2_seed2` | 4 | `+0.9651 … +1.0156` | 4/4 |

| scene | n | α_h range | outside |
|---|---|---|---|
| `000165_10` | 3 | `+0.9622 … +1.1994` | 3/3 |
| `000167_10` | 3 | `+0.9546 … +1.2095` | 3/3 |
| `000168_10` | 3 | `+0.9663 … +1.2058` | 3/3 |
| `000169_10` | 3 | `+0.9534 … +1.2246` | 3/3 |

**No checkpoint overlaps. No scene overlaps. No unit overlaps.** The
heterogeneity rule (§19: one checkpoint separating while others overlap ⇒ MIXED
⇒ CASE C) is **not** triggered — all three separate, and by similar margins.

### SECONDARY — 36 CROSS units (reported separately, never pooled)

```
alpha_h   trained median +1.0187   IQR [+0.9864, +1.1206]   min +0.9534   max +1.3666
outside the observed random population :  36 / 36 = 1.0000      overlap 0.0000
rank-biserial (effect size only)       :  +1.0000
```

24 of these 36 are **chimeras** (one checkpoint's extractor with another's
aggregation) and are not real models. The separation does not depend on the
pairing.

---

## 3. THE RAW RESPONSE CURVES — 12 MATCHED UNITS

`m_h(t)` at `t = 0,16,32,48,64,80,96`; `k_true = t/16 = 0,1,2,3,4,5,6`.

```
H2_seed0 000165   1.582  2.312  4.034  5.801  7.250  8.615  7.669   NON_MONOTONE
H2_seed0 000167   1.736  2.323  4.027  5.779  7.248  8.544  7.785   NON_MONOTONE
H2_seed0 000168   1.593  2.315  4.049  5.818  7.284  8.636  7.710   NON_MONOTONE
H2_seed0 000169   1.568  2.329  4.039  5.802  7.244  8.633  7.857   NON_MONOTONE
H2_seed1 000165   1.655  1.952  3.077  4.280  5.441  6.066  6.661   QUALIFYING
H2_seed1 000167   1.823  1.965  3.069  4.269  5.394  6.041  6.639   QUALIFYING
H2_seed1 000168   1.734  1.962  3.079  4.305  5.467  6.098  6.682   QUALIFYING
H2_seed1 000169   1.721  1.973  3.077  4.284  5.403  6.037  6.647   QUALIFYING
H2_seed2 000165   1.423  2.047  3.465  4.834  6.243  6.877  6.685   NON_MONOTONE
H2_seed2 000167   1.431  2.046  3.460  4.857  6.243  6.759  6.546   NON_MONOTONE
H2_seed2 000168   1.418  2.049  3.456  4.853  6.286  6.943  6.779   NON_MONOTONE
H2_seed2 000169   1.436  2.044  3.453  4.848  6.223  6.816  6.805   NON_MONOTONE
```

**MEASURED.** Every curve rises steeply and near-linearly from `t = 16` to
`t = 64`, then flattens; `H2_seed0` and `H2_seed2` **turn down at the last step**
(`t = 80 → 96`), which is why 8 of 12 are `NON_MONOTONE`. All 12 have `Q1`
(`α_h > 0`) and `Q3` (`|α_h| > |α_v|`). None is `CLIPPED`. None is `FLAT`.

`k_true` is recorded per unit and **was not required to be matched** (W3). The
curves do not equal `k_true`: the intercept is ≈ 1.4–1.8 candidates at `t = 0`
and the response saturates near the top of the 12-candidate grid.

---

## 4. CLASSIFICATION TALLY — heterogeneous, and reported as such

| | CLIPPED | FLAT | ANTI_CORR | NON_MONOTONE | NON_SPECIFIC | QUALIFYING |
|---|---|---|---|---|---|---|
| MATCHED (12) | 0 | 0 | 0 | **8** | 0 | **4** |
| CROSS (36) | 0 | 0 | 0 | 31 | 0 | 5 |
| SEARCH-FREE (4) | 0 | 0 | 1 | 3 | 0 | 0 |
| **random (384)** | **0** | **378** | **0** | **6** | **0** | **0** |

**All four `QUALIFYING` matched units are `H2_seed1`; `H2_seed0` and `H2_seed2`
contribute zero.** This is real heterogeneity and is **not** averaged away. It is
driven by one step (`t = 80 → 96`) in two of three checkpoints, not by the
separation quantity.

**DERIVED, and important:** the classification tally is *not* the decision
quantity. The frozen decision logic (§19) is the distributional comparison of
`α_h`. The `QUALIFYING`/`NON_MONOTONE` split is required reporting, and it
qualifies the result — it does not overturn a separation that holds 12/12.

---

## 5. THE OTHER THREE QUANTITIES — reported, never converted to a new claim

| quantity (MATCHED, n=12) | trained median | random median | outside | overlap | rank-biserial |
|---|---|---|---|---|---|
| `α_h` **(primary)** | `+1.0014` | `+0.0032` | **12/12** | 0.000 | `+1.0000` |
| `α_v` (control axis) | `+0.0030` | `−0.0000` | 4/12 | 0.667 | `−0.1111` |
| `R_h` (response range) | `5.4171` | `0.3535` | **12/12** | 0.000 | `+1.0000` |
| positive first differences | `5` | `3` | 0/12 | 1.000 | `+0.9219` |

**MEASURED:** `α_v`, the vertical control axis, is **not** separated — trained and
random overlap 2:1 and the effect size is near zero and *negative*. The
preregistration set **no expectation** that `α_v` be zero, and none is imposed
now. Its non-separation is what makes the horizontal separation axis-specific
rather than a global offset.

**MEASURED:** the positive-first-difference count separates in effect size
(`+0.92`) but **0/12 fall outside the observed random range** — six random units
reached 6/6 positive differences. Reported as-is; it produces no claim.

---

## 6. THE SEARCH-FREE CONTROL — the most informative secondary result

`NEG_shift_none`, run fresh on these four scenes under its own `shift="none"`
cost-volume construction. **Never mixed into the random-weight null. No threshold
is derived from it.**

```
000165   7.603  2.844  2.800  2.802  2.860  2.906  2.875   a_h = +0.0153   NON_MONOTONE
000167   7.603  2.668  2.715  2.763  2.881  2.913  2.804   a_h = +0.0397   NON_MONOTONE
000168   7.603  2.936  2.834  2.763  2.727  2.763  2.696   a_h = -0.0413   ANTI_CORRELATED
000169   7.603  2.946  2.982  3.015  3.110  3.216  3.210   a_h = +0.0604   NON_MONOTONE
```

**MEASURED:** `α_h ∈ [−0.041, +0.060]`, **0/4 outside the observed random range**,
overlap 1.000. With a candidate-constant cost volume, a trained aggregation
produces **no translation slope at all**.

**DERIVED:** the `t = 0` value is `7.603` to three decimals and **identical across
all four scenes**. Under `shift="none"` with a self-pair at `t = 0` the two
feature maps are equal, so the cost volume is exactly zero everywhere and the
output is a pure architecture constant — scene-independent by construction. This
is a property of the construction, not a measurement of the scene.

**INFERRED, with the confound stated plainly:** this control differs from the
trained arm in **two** ways at once — a different checkpoint *and* a different
cost-volume construction. It therefore **cannot** isolate the construction as the
cause. What it does establish is that the `α_h ≈ 1` signature is **not** a generic
property of any trained aggregation reading any cost volume.

**A warning about `R_h`:** the search-free control is `4/4` outside the observed
random `R_h` range (median 4.86) — **while having no slope whatsoever**. Its
entire range comes from the single degenerate `t = 0 → 16` drop of ≈ `−4.7`.
**`R_h` is therefore not a clean separator and must not be read as one.** Only
`α_h` distinguishes the two behaviours.

---

## 7. WHAT THIS DOES **NOT** SHOW — the qualification that matters most

**DERIVED, from the campaign's own algebra** (`predesign_audit_20260911`): with a
self-pair, `V(k,w) = F(k−τ, w+τ)` where `F` is τ-independent, so
`m(τ) ≈ τ + const` — **a unit slope is architecturally forced in the padding-free
interior, for ANY weights.** The measured trained slopes are `0.95 … 1.22`:
**exactly that prediction**.

**MEASURED (ARCH-RATE-001):** 378 of 384 random units were nonetheless `FLAT`
(median range 0.354 candidates) because the finite `D = 12` padding boundary
dominates them.

**INFERRED — the honest reading of the separation:**

> The trained aggregation **realises the architecturally available unit ramp**;
> the 384 sampled random aggregations **do not**. That is a real, large, uniform
> difference between trained and sampled-random weights — and it is a difference
> in *whether the architectural equivariance survives the padding boundary*, not
> evidence that the network searches for or computes correspondence.

This is precisely why the claim ceiling below is where it is.

---

## 8. THE PRE-EXECUTION DEFECT (disclosed, not repaired away)

Three frozen documents described the fit as spanning **all seven** levels. The
same `spec.json` asserted `identical_to_reference: true`, hash-bound to a
definition that uses **six** (`T_FIT`, excluding `t = 0`).

**MEASURED, before any trained aggregation ran:** recomputing both fits from the
reference's own published raw curves —

```
max | recorded alpha_h  -  six-level fit  |  =  0.0                    (all 384, exact)
max | recorded alpha_h  -  seven-level fit |  =  0.09443637132644654
```

The frozen population is a six-level fit. The "ALL SEVEN" phrasing was **my
transcription error when drafting this freeze**. Resolved in
`PRE_EXECUTION_ADDENDUM.md` by the criterion that predates any observation:
the comparison is meaningless unless the statistic is identical to the
reference's. No document was modified; the defect stands beside its correction.

**The literal seven-level reading was computed anyway and recorded** as `α_h7`
(MATCHED median `+1.0115`, min `+0.8902`, max `+1.2385`). It is compared to
nothing — no comparable random population exists — and enters no classification
and no decision. **It does not change the direction or magnitude of the result.**

---

## 9. WIRING CHECKS

| check | result |
|---|---|
| **W1** left crop byte-identical across every condition | **PASS** — 1 distinct hash per scene, all 4 scenes |
| **W2** right-crop origin `(0,t)` / `(t,0)` asserted at every level | **PASS** — 24 distinct origins asserted |
| **W2** sign verified **from the arrays**, not assumed | **PASS** — 84 checks: `right[:, :-t] == left[:, t:]` exactly, so `d = a_L − a_R = +t` px |
| **W3** `k_true = t/16` recorded, output not required to equal it | recorded per unit; the curves do **not** equal `k_true` |
| **W4** complete unit repeated **after process restart** | **PASS** — all 14 values **bit-identical** (IEEE-754 byte comparison) |
| no numeric `t = 0` expectation | **none existed**; the invalid `5.5` anchor was never reinstated |

---

## 10. DECISION

Against the frozen §19 criteria, applied without adjustment:

| CASE A requirement | met? |
|---|---|
| trained distribution substantially displaced from random | **yes** — median `+1.00` vs `+0.003`; min trained 3.9× the random max |
| little or no overlap | **yes** — overlap `0.000` |
| high outside-range fraction | **yes** — `12/12 = 1.0000` |
| consistent across all three checkpoints | **yes** — 4/4, 4/4, 4/4 |
| consistent across all four scenes | **yes** — 3/3, 3/3, 3/3, 3/3 |

CASE C triggers checked and **not** present: no partial separation, no mixed
checkpoints *on the separation*, no scene dependence, **no clipping** (0 CLIPPED
anywhere). The one structure that *is* present — the `QUALIFYING`/`NON_MONOTONE`
split concentrated in `H2_seed1` — is reported in §4 and §3 and does not affect
any unit's outside-range status.

```
VERDICT :  CLEAR-SEPARATION
```

---

## 11. CLAIM CEILING — what may be said

**Permitted, and the strongest available:**

> Under the specified synthetic translation construction, the trained aggregation
> response is **distinguishable** from the empirically sampled random aggregation
> population: all 12 matched units and all 36 cross units fall **outside the
> observed random population** of 384 `α_h` values, with zero overlap and
> consistency across three checkpoints and four scenes.

**Required phrasing:** *"outside the observed random population."*
The reference is **384 sampled parameterisations**, not the space of random
weights.

**Prohibited and not claimed anywhere:** "impossible under random weights";
"proves learned correspondence."

**NOT established by this experiment:** genuine stereo correspondence; correct
disparity estimation; true disparity search; correct correspondence on natural
scenes; metric depth correctness; generalisation; biological or physical scene
correspondence. **Level D remains NOT-DEMONSTRATED.**

---

## 12. STANDING STATE OF THE CAMPAIGN

- **ESTABLISHED:** right-image dependence; binocular pathway engaged; final
  accuracy as a metric — for a model with provably zero disparity search
  (EXP-010). *Unchanged.*
- **INCONCLUSIVE:** Level C candidate-coordinate sensitivity, bypassed as
  structurally the wrong axis. *Unchanged.*
- **NOT-DEMONSTRATED:** Level D geometric correspondence. **Unchanged by this
  experiment.**
- **NEW, MEASURED:** under this synthetic construction the trained aggregation's
  horizontal translation slope lies entirely outside the observed 384-unit random
  population, uniformly across three checkpoints and four scenes; the same
  statistic shows no separation on the vertical control axis, and none at all
  when the cost volume is candidate-constant.

**No follow-up experiment is designed or launched.**
