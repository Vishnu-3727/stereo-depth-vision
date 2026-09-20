# EXECUTION PROMPT — EXP-CORRESPONDENCE-GEOM-002

Hand this to the executing session **verbatim** when, and only when, execution is
authorised. **It is not authorised by this document.**

---

```
EXECUTE EXP-CORRESPONDENCE-GEOM-002

AUTHORIZATION: <state explicitly, or do not proceed>

The preregistration is frozen at:
    phase2/diagnostics/correspondence_geom/20260911T115917Z/

Execute EXACTLY as frozen.

==================================================
ABSOLUTE RULES
==================================================

DO NOT change the protocol.
DO NOT edit PREREGISTRATION.md, DESIGN_AUDIT.md, construction_spec.json,
        mask_spec.json or frozen.sha256.
DO NOT change: checkpoints, seeds, scenes, crop geometry, mask, t levels,
        construction, axis definitions, statistic, controls, decision rule,
        claim ceiling.
DO NOT train, fine-tune, or instantiate untrained weights.
DO NOT modify GEOM-001 or any historical record.
DO NOT rerun any previous experiment.
DO NOT inspect any result between writing the preregistration and executing.

==================================================
PRE-EXECUTION FREEZE VERIFICATION
==================================================

Before the FIRST model instantiation:

1.  PREREGISTRATION.md exists.
2.  construction_spec.json sha256 =
        a9e8f250bfef49607951fae0b304d5713091553ad2da885b911f4de711bdea10
3.  mask_spec.json sha256 =
        b274ce037a1fbd6973f8470baf1c38f25a50267c1b740548df4f224598b4b5d7
4.  sha256sum -c frozen.sha256  -> both OK   (digests are over FILE BYTES)
5.  Both digests appear verbatim in PREREGISTRATION.md.
6.  No RESULTS.md exists.
7.  DESIGN_AUDIT.md exists and records the sign correction (a_R = a_L + t).
8.  The four checkpoint sha256 in PREREGISTRATION.md section 5 all verify.
9.  Scenes are indices [1,2,3,4] of hailo_val, disjoint from {27,0,31,6}.
10. No protocol modification occurred after the preregistration.

ANY failure -> HARD STOP. Do not repair the protocol.

==================================================
CONSTRUCTION  (construction_spec.json)
==================================================

Both images are crops of the SAME source image: scene.left.
The scene's own right image is NEVER used.

    left  crop : rows [0,272), cols [0,1136)          FIXED every condition
    right crop : horizontal  rows [0,272),  cols [t, 1136+t)
                 vertical    rows [t,272+t), cols [0,1136)

    t in {0,16,32,48,64,80,96} px      13 distinct conditions (t=0 shared)

Integer slicing ONLY. No fill. No padding. No interpolation.
Assert crop is 1136 x 272 in EVERY condition.
Assert every window lies inside the 1232 x 368 source.
Assert the left crop is byte-identical across all conditions.
The axis must be a FUNCTION ARGUMENT, not a branch in the forward path.

Imposed disparity is EXACTLY t px, uniform. d = a_R - a_L = t.
Do NOT use the sign from successor_audit_20260911/RECOMMENDATION.md; it is
wrong and is corrected in DESIGN_AUDIT.md section 2.

==================================================
MASK  (mask_spec.json)
==================================================

In CROP coordinates:  64 <= y < 208  AND  64 <= x < 1072
Retained pixels per scene: 145152   -- assert exactly, every condition.
No ground truth is consulted. The mask is identical for every t, axis,
model and scene.

==================================================
MODELS
==================================================

POS_6b_seed0 / seed1 / seed2   shift="left" + standardised readout
NEG_shift_none                 shift="none" + standardised readout

Verify all four sha256 at load. Readout must have 0 parameters.
Re-hash all weights after every scene; any change -> HARD STOP.

==================================================
OUTPUT AND STATISTIC
==================================================

For every model x scene x condition compute disparity_initial, then

    m_axis(t)  = np.median(disparity_initial[mask])
    Slope_axis = sum_t (m_axis(t) - m_axis(0)) * (t/16) / sum_t (t/16)^2
                 over t in {16,32,48,64,80,96}

Anchor Slope_h = 1 candidate per candidate -- the source-derived expected
value ONLY. Do NOT turn it into a threshold.

No EPE. No D1. No RMSE. No p-value. Refinement never invoked.

==================================================
WIRING CHECK (not a scientific criterion)
==================================================

NEG_shift_none at t=0 must return disparity_initial = 5.5 EXACTLY.
A deviation indicates an implementation fault, not a finding.

==================================================
FOUR CONDITIONS, EVERY UNIT
==================================================

C1  Slope_h > 0
C2  m_h(t) strictly increasing over t = 0,16,32,48,64,80,96
C3  |Slope_h| > |Slope_v|
C4  Slope_h > Slope_h(NEG, same scene)

PASS iff C1 and C2 and C3 and C4 at 12/12 units.
No 9/12. No majority. No mean. No relaxation. No post-hoc exception.
If 12/12 fails, do NOT rescue the experiment.

==================================================
SANITY CHECKS BEFORE THE VERDICT
==================================================

1.  All crops 1136 x 272.
2.  No interpolation.
3.  No fill/padding.
4.  Left crop identical in every condition.
5.  Mask count 145152 everywhere.
6.  Horizontal and vertical arms share the code path.
7.  No training occurred.
8.  No weights changed.
9.  Checkpoint hashes unchanged.
10. Deterministic execution holds.
11. Historical records untouched.
12. Exactly 208 forward passes accounted for.
13. shift="none" volume degeneracy re-verified per scene.

Any unexpected execution path -> HARD STOP.

==================================================
RESULTS
==================================================

Only AFTER execution, create in the frozen directory:
    RESULTS.md   results.json   run.log   ENVIRONMENT.txt   RELATED_RUNS.md

Do not modify PREREGISTRATION.md.

Report per unit: Slope_h, Slope_v, Slope_h(NEG), m_h(t) and m_v(t) for all
seven t, C1..C4, unit PASS/FAIL. Also report the raw curves, the search-free
response, the t=0 wiring check, and any protocol deviations.

Classify any failure as one of:
    1 correspondence-positive        2 flat response
    3 non-specific response          4 search-free-equivalent
    5 non-monotone / mixed           6 protocol failure
Do not collapse failures into "stereo failed".

==================================================
CLAIM CEILING
==================================================

PASS -> use ONLY:
  "On synthetic fronto-parallel stereo pairs, the trained pipeline's
   disparity_initial tracks the imposed horizontal displacement, specifically
   on the epipolar axis, beyond a matched vertical control and beyond a
   provably search-free baseline."
  Classify as SYNTHETIC-CORRESPONDENCE-EVIDENCE.

FAIL -> SYNTHETIC-CORRESPONDENCE-NOT-DEMONSTRATED.
  Do NOT conclude correspondence is absent; the input is out of distribution.

Do NOT claim under any outcome: real-scene correspondence, disparity
correctness, sub-candidate precision, genuine disparity search, learned
versus architectural origin, generalisation, or anything about
disparity_final.

==================================================
FINAL REPORT
==================================================

1 Freeze verification:      2 Forward passes:        3 Training occurred:
4 Wiring check (t=0):       5 Search-free control:   6 Horizontal response:
7 Vertical response:        8 C1:  9 C2:  10 C3:  11 C4:
12 12/12 global decision:  13 FINAL VERDICT:        14 EXACT CLAIM CEILING:
15 Failure classification: 16 Protocol deviations:  17 Historical records modified:
18 Is a subsequent experiment justified:

Do not automatically launch any subsequent experiment. STOP.
```

---

## Optional extension — NOT part of the frozen protocol

`DESIGN_AUDIT.md` §6.4 records that a PASS cannot distinguish **learned** from
**architecturally-forced** correspondence. Adding one untrained random-weight
aggregation arm (as ARCH-001 used) would settle it at ~52 extra forward passes.

**It is deliberately excluded** from the frozen protocol to keep the design
minimal, and because instantiating untrained weights required explicit
authorisation in ARCH-001. **If the operator wants §6.4 answered, that arm must
be added to a FRESH preregistration — not bolted onto this one.**
