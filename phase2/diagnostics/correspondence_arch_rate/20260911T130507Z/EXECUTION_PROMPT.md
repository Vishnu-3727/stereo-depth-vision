# EXECUTION PROMPT — EXP-CORRESPONDENCE-ARCH-RATE-001

Hand this to the executing session **verbatim** when, and only when, execution is
authorised. **It is not authorised by this document.**

---

```
EXECUTE EXP-CORRESPONDENCE-ARCH-RATE-001

AUTHORIZATION: <state explicitly, or do not proceed>

Frozen at:
    phase2/diagnostics/correspondence_arch_rate/20260911T130507Z/

Execute EXACTLY as frozen.

==================================================
ABSOLUTE RULES
==================================================

DO NOT change the protocol.
DO NOT edit PREREGISTRATION.md, DESIGN_AUDIT.md, or any *_spec.json.
DO NOT change: construction, crop geometry, mask, t levels, axes, seeds, N,
        feature extractors, scenes, statistic, classification order, thresholds.
DO NOT train, fine-tune, or optimise any weight.
DO NOT execute any TRAINED aggregation. Only freshly seeded random
        aggregations are ever run.
DO NOT measure, load for measurement, or compare against any trained-model
        response, including GEOM-002's published curves.
DO NOT modify GEOM-001, GEOM-002, or any historical record.
DO NOT invent a pass/fail rule. THE RESULT IS A RATE.

==================================================
PRE-EXECUTION FREEZE VERIFICATION
==================================================

Before the FIRST model instantiation:

1.  PREREGISTRATION.md exists.
2.  construction_spec.json   sha256 = 7a4e2341fc50e3695648cc286cdb4219390a0378a6669994ec0e4555b75cf9b0
3.  randomisation_spec.json  sha256 = e52b5b03c5d4406e5ed9212f6681ee53bb448927eebddd367a7698ce9ce203ad
4.  classification_spec.json sha256 = 273f7562b62dd4c29a2d617e3b2b4b24d98532fcc7d99531b82b55b377d0cd28
5.  sha256sum -c frozen.sha256  -> all three OK   (digests are over FILE BYTES)
6.  All three digests appear verbatim in PREREGISTRATION.md.
7.  No RESULTS.md exists.
8.  DESIGN_AUDIT.md exists and records the boundary-equivariance qualification
    (exact equivariance at 2 / 1 / 0 candidates for tau = 0 / 1 / >=2).
9.  The three POS checkpoint sha256 verify.
10. Scenes are hailo_val indices [5,7,8,9], disjoint from {27,0,31,6} and {1,2,3,4}.

ANY failure -> HARD STOP. Do not repair the protocol.

==================================================
CONSTRUCTION
==================================================

Both images are crops of the SAME source image: scene.left.
The scene's own right image is NEVER used.

    left  crop : rows [0,272), cols [0,1136)          FIXED every condition
    right crop : horizontal  rows [0,272),  cols [t, 1136+t)
                 vertical    rows [t,272+t), cols [0,1136)
    t in {0,16,32,48,64,80,96}       13 distinct conditions (t=0 shared)

Integer slicing ONLY. No fill. No padding. No interpolation.
Assert 1136 x 272 in EVERY condition; every window inside the 1232 x 368 source;
left crop byte-identical across all 13 conditions within a scene.
The axis must be a FUNCTION ARGUMENT, not a branch.

Imposed disparity is EXACTLY t px: d = a_R - a_L = t.

==================================================
RANDOMISATION
==================================================

Randomise the AGGREGATION ONLY:
    Aggregation(in_channels=32, channels=32, num_layers=4)   111585 params
    torch.manual_seed(seed) immediately before construction
    seeds 0..31   (N = 32)

FROZEN, NOT randomised: feature extractor (trained), cost volume, readout
(0 parameters), refinement (never invoked).

Feature extractors: POS_6b_seed0, POS_6b_seed1, POS_6b_seed2.
Load them ONLY to obtain the frozen feature extractor. NEVER run their
trained aggregation. Assert that the aggregation module in every forward
pass is a freshly seeded random instance.

Verify every randomised tensor's sha256 is distinct across all 32 seeds.

CACHING: the cost volume depends on (feature extractor, scene, axis, t) and
NOT on the aggregation seed. Build 156 volumes once and reuse across seeds.

==================================================
MASK
==================================================

Crop coordinates: 64 <= y < 208 AND 64 <= x < 1072
Retained: 145152 px -- assert exactly, every condition.
No ground truth is consulted.

==================================================
STATISTIC
==================================================

    m_axis(t) = np.median(disparity_initial[mask])
    alpha_axis = FREE-INTERCEPT OLS slope of m_axis(t) on t/16, t in {16..96}

FREE INTERCEPT. NOT through the origin. The through-origin form is the proven
GEOM-002 defect (a step of height h maps to 0.2308h).

    Q1  alpha_h > 0
    Q2  m_h(t) strictly increasing over all 7 levels
    Q3  |alpha_h| > |alpha_v|

No predicted value is assigned to alpha_v.
No EPE. No D1. No RMSE. No p-value. Refinement never invoked.

==================================================
CLASSIFICATION  (ordered, first match wins)
==================================================

1  NONFINITE        any non-finite m -> HARD STOP (implementation fault)
2  CLIPPED          min m_h <= 1.0  OR  max m_h >= 10.0
3  FLAT             max m_h - min m_h < 1.0
4  ANTI_CORRELATED  alpha_h < 0
5  NON_MONOTONE     Q2 fails and not FLAT
6  NON_SPECIFIC     Q1 and Q2 hold, Q3 fails
7  QUALIFYING       Q1 and Q2 and Q3

CLIPPED IS EVALUATED BEFORE EVERY ORDINARY FAILURE CLASS.
Never fold a railed response into ordinary failure.

==================================================
RESULT
==================================================

PRIMARY RESULT = the rate k/384 of QUALIFYING units, with the full seven-way
tally, broken down by feature extractor and by scene.

THERE IS NO PASS/FAIL. THERE IS NO THRESHOLD ON THE RATE.

MANDATORY: publish every m_h(t) and m_v(t) for all 384 units in results.json,
so any reader can reclassify under an alternative definition without re-running.

==================================================
SANITY CHECKS BEFORE REPORTING
==================================================

1  All crops 1136 x 272.          2  No interpolation.
3  No fill/padding.               4  Left crop byte-identical per scene.
5  Mask count 145152 everywhere.  6  Axes share the code path.
7  No training occurred.          8  No trained aggregation executed.
9  Checkpoint hashes unchanged.  10  Seed tensor hashes all distinct.
11 Determinism holds.            12  Historical records untouched.
13 Exactly 4992 arms over 156 cached volumes.

Any unexpected execution path -> HARD STOP.
EVERY hard stop must HALT execution, not merely be recorded.

==================================================
INTERPRETATION
==================================================

high rate         -> the signature is produced WITHOUT training; the synthetic
                     self-pair observable is NON-DIAGNOSTIC; the construction is
                     closed.
intermediate rate -> the rate itself is the answer. No binary reading.
low / zero rate   -> untrained weights rarely reproduce it; the observable MAY be
                     diagnostic; a trained-arm experiment becomes warrantable
                     under a SEPARATE preregistration.

NOT claimable under any outcome: anything about the trained models;
correspondence, learned or architectural; genuine disparity search; disparity
correctness; real-scene correspondence; generalisation.

A low rate does NOT establish that trained models exhibit correspondence.
The trained arm is never measured.

==================================================
FINAL REPORT
==================================================

1 Freeze verification:        2 Forward passes / volumes:
3 Training occurred:          4 Trained aggregation executed:
5 Seed distinctness:          6 QUALIFYING rate k/384:
7 Full seven-way tally:       8 CLIPPED count (reported separately):
9 Rate by feature extractor: 10 Rate by scene:
11 alpha_h and alpha_v distributions (descriptive):
12 Was the PREDICTED high rate observed:
13 EXACT CLAIM CEILING:      14 Protocol deviations:
15 Historical records modified:
16 Is a subsequent experiment warranted:

Do not automatically design or launch any subsequent experiment. STOP.
```

---

## Note for the executing session

This experiment has **no pass/fail rule by design**. If you find yourself wanting
to declare a verdict, you have misread it. The deliverable is a rate and a tally.

Its interpretation does not depend on choosing a post-hoc success threshold —
that is the property it was built for, and the reason it is worth running.
