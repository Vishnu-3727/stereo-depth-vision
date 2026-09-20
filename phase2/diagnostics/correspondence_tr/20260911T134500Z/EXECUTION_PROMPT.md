# EXECUTION PROMPT — EXP-CORRESPONDENCE-TR-001

The frozen handoff. **Execution begins only after every check in §A passes.**

---

```
EXECUTE EXP-CORRESPONDENCE-TR-001

Frozen at: phase2/diagnostics/correspondence_tr/20260911T134500Z/

==================================================
A. PRE-EXECUTION FREEZE VERIFICATION
==================================================

1.  PREREGISTRATION.md, DESIGN_AUDIT.md, spec.json, random_reference.json exist.
2.  spec.json              sha256 = 5e6a0bca39408018454fb69e3287f26f8524fc89475ee564a5da5c54a5fa0280
3.  random_reference.json  sha256 = 8d58c412db5244dc16a8b59de7223ab4def8b70681837cb86928c4a5352e4722
4.  sha256sum -c frozen.sha256 -> both OK  (digests over FILE BYTES)
5.  Both digests appear verbatim in PREREGISTRATION.md.
6.  No results.json, results.csv, RESULTS.md or AUDIT.md exists.
7.  RANDOM REFERENCE re-verified: ARCH-RATE results.json sha256 =
        1f0463ec08b950d3c1b69804407299f66b4b6cee5c04af51419a061dba7458f1
    and its three spec hashes match those recorded inside it.
8.  All four checkpoint sha256 verify:
        H2_seed0        581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a
        H2_seed1        58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf
        H2_seed2        245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69
        NEG_shift_none  d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1
9.  Scenes are [5,7,8,9] -- IDENTICAL to the random reference.
10. Mask is 64<=y<208 AND 64<=x<1072, 145152 px -- IDENTICAL to the reference.

ANY failure -> HARD STOP, VERDICT = NO DETERMINATION.
Do not repair. Do not regenerate the random reference.

==================================================
B. ABSOLUTE RULES
==================================================

DO NOT train, fine-tune, or modify any weight.
DO NOT change: construction, t levels, crop, mask, preprocessing, resolution,
        dtype, deterministic settings, standardization, shift construction,
        refinement, statistic, classification order, thresholds.
DO NOT re-run the random arm. It is hash-verified and final.
DO NOT introduce a post-hoc statistic or threshold.
DO NOT exclude, downweight or specially treat seed 18.
DO NOT overwrite any historical record.
DO NOT compute a p-value. Units are not independent.
DO NOT pool the MATCHED and CROSS populations.

==================================================
C. CONSTRUCTION  (inherited, unchanged)
==================================================

    left  crop : rows [0,272), cols [0,1136)        FIXED every condition
    right crop : horizontal  rows [0,272),  cols [t, 1136+t)
                 vertical    rows [t, 272+t), cols [0,1136)
    t in {0,16,32,48,64,80,96}      u = t/16      a_R = a_L + t
    known disparity d = t px        k_true = t/16 candidates

Integer slicing only. No fill, no padding, no interpolation.
VERIFY THE SIGN FROM THE IMPLEMENTATION. Do not assume it.

==================================================
D. WIRING CHECKS -- STRUCTURAL ONLY
==================================================

W1  left input byte-identical across all translation conditions (hash-compared).
W2  right-crop origin asserted == (0,t) horizontal, (t,0) vertical, every level.
W3  record k_true = t/16. The model output is NOT required to equal it.
W4  repeat one COMPLETE trained unit AFTER PROCESS RESTART.
    The complete response vector must be BIT-IDENTICAL. Failure -> HARD STOP.

NO NUMERIC EXPECTATION AT t=0. The invalid 5.5 expectation is never reinstated.

==================================================
E. UNITS
==================================================

MATCHED (PRIMARY)  checkpoint c supplies BOTH extractor and aggregation
                   3 checkpoints x 4 scenes = 12 units. The real models.
CROSS (SECONDARY)  3 extractors x 3 trained aggregations x 4 scenes = 36 units.
                   Off-diagonal pairs are chimeras. Report separately.
SEARCH-FREE        NEG_shift_none, 4 scenes. SEPARATE. Never mixed into the null.

==================================================
F. STATISTIC
==================================================

    m_h(t) = median predicted candidate disparity over the fixed mask
    alpha_h, beta_h : OLS of m_h(t) on u = t/16 over ALL SEVEN levels
    FREE INTERCEPT. Through-origin is PROHIBITED.

Also record: alpha_v, beta_v (same fit, vertical arm -- a CONTROL AXIS, with NO
expectation that it is zero); R_h = max(m_h) - min(m_h); all six first
differences with positive count, negative count, min and max.

Classification, precedence preserved, thresholds inherited unchanged:
    1 NONFINITE -> HARD STOP
    2 CLIPPED   min m_h <= 1.0 OR max m_h >= 10.0
    3 FLAT      max m_h - min m_h < 1.0
    4 ANTI_CORRELATED  5 NON_MONOTONE  6 NON_SPECIFIC  7 QUALIFYING

==================================================
G. COMPARISON
==================================================

Per trained unit, against the 384 frozen random alpha_h values:
    empirical percentile rank, empirical CDF position,
    nearest random observations above and below,
    outside-observed-random-range flag.

KEY QUANTITY
    trained_outside_random_range_rate
      = (# trained units outside the complete observed random alpha_h range)
        / (total trained units)

Same calculation for alpha_v, R_h and the positive-first-difference count --
REPORTED, never converted into a new binary claim.

Distributional reporting MANDATORY: trained median/IQR/min/max, random
median/IQR/min/max, overlap, outside-range fraction, and a nonparametric effect
size (rank-biserial from Mann-Whitney U) AS AN EFFECT SIZE ONLY, NO p-value.

INVALID AND NOT USED: "0/384 random qualified therefore any trained
qualification proves learning."

==================================================
H. DECISION
==================================================

CASE A  CLEAR SEPARATION  substantially displaced, little/no overlap, high
                          outside-range fraction, AND consistent across all
                          three checkpoints and all four scenes.
CASE B  OVERLAP           trained substantially overlaps random.
CASE C  AMBIGUOUS         partial separation, mixed checkpoints, strong scene
                          dependence, clipping, or other structure.

HETEROGENEITY RULE: one checkpoint separating while others overlap -> MIXED ->
CASE C, never overall success. If one scene drives it, say so. Never average
heterogeneity away.

==================================================
I. LANGUAGE
==================================================

REQUIRED : "outside the observed random population"
PROHIBITED: "impossible under random weights"
PROHIBITED: "proves learned correspondence"

The reference is 384 sampled parameterisations, NOT the space of random weights.

==================================================
J. OUTPUTS
==================================================

Only AFTER execution: results.json, results.csv, AUDIT.md, RESULTS.md, run.log,
ENVIRONMENT.txt, RELATED_RUNS.md.

results.json must contain, per unit: every m_h(t), every m_v(t), alpha_h, beta_h,
alpha_v, beta_v, first differences, classification, clipping status, flatness
status, random percentile, outside-range flags.

AUDIT.md must carry exact hashes and provenance.

==================================================
K. FINAL REPORT -- exact format from the system prompt section 23
==================================================

Freeze / Execution / Random population / Trained population / Separation /
Raw-data integrity / Verdict / Claim ceiling.

VERDICT is exactly one of:
    CLEAR-SEPARATION   OVERLAP   AMBIGUOUS   NO-DETERMINATION

==================================================
L. STOP
==================================================

After the report, STOP. Do not design or execute any follow-up experiment.
The experiment owner audits the result independently.
```

---

## Note to the executing session

The random reference is **final and hash-verified**. If the trained results are
hard to interpret, the correct action is `AMBIGUOUS` — **not** re-running the
random arm, not adding seeds, not adjusting a threshold.

A trained value outside the observed random range is a **descriptive fact about a
finite sample**. It is not proof of learning, and the report must not read as
though it were.
