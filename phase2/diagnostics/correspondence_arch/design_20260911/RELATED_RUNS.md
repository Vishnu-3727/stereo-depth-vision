# RELATED RUNS — DESIGN FOR EXP-CORRESPONDENCE-ARCH-001

This is a **design + pre-flight record**, not an experiment. No `RESULTS.md`
exists because nothing was executed. Every record listed below is preserved
exactly as produced; **none was modified, re-run, re-scored or reinterpreted.**

This is the first record under `phase2/diagnostics/correspondence_arch/`. The
timestamped experiment directory required by §M has deliberately **not** been
created, per §M's instruction not to create it while this prompt is used for
planning.

---

## Lineage

```
EXP-CORRESPONDENCE-SHIFT-001            (Stage A — image-translation sweep)
  phase2/diagnostics/correspondence/20260910T142622Z/
  VERDICT: DIAGNOSTIC-INVALID-AS-PREREGISTERED — halted at Stage A
        |
        |  a degenerate cost volume still produced a structured, input-dependent
        |  candidate curve  ->  response magnitude alone cannot evidence
        |  correspondence
        v
EXP-CORRESPONDENCE-INDEX-001            (single random permutation as control)
  .../correspondence_index/20260910T164346Z/   ESTABLISHED       -> EXPLORATORY-ONLY
  .../correspondence_index/20260910T164326Z/   NOT-DEMONSTRATED  -> EXPLORATORY-ONLY
        |
        v
  .../correspondence_index/audit_20260911T011113Z/
    Provenance audit. Overall: NO-CONFIRMATORY-EVIDENCE.
        |
        v
EXP-CORRESPONDENCE-INDEX-002            (empirical permutation null, M = 512)
  .../correspondence_index/20260911T015124Z/
  VERDICT: CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED
    null_pass 0/12, order_pass 12/12
        |
        v
  .../correspondence_index/design_audit_20260911/
    INDEX-003 design audit. VERDICT: DESIGN-INCONCLUSIVE.
    Blocking finding: the ordered response is architecturally expected and
    would be produced by arbitrary weights. Recommended an architecture-only
    control as the mandatory precondition.
        |
        v
DESIGN — EXP-CORRESPONDENCE-ARCH-001                    (this record)
  phase2/diagnostics/correspondence_arch/design_20260911/
  VERDICT: CLEAN-DESIGN-AVAILABLE     ** nothing executed **
```

---

## 1. Stage A — `EXP-CORRESPONDENCE-SHIFT-001`

**Record:** `phase2/diagnostics/correspondence/20260910T142622Z/`
**Verdict:** `DIAGNOSTIC-INVALID-AS-PREREGISTERED`

Showed that the trained `shift="none"` model — whose cost volume is bit-exactly
degenerate — nevertheless produced a non-uniform, input-dependent candidate curve
(soft-argmin 3.90 ± 1.96), responding to right-image translation by up to 0.70
candidates. Its own claim ceiling recorded, as **INFERRED**, that "the D-axis
convolution boundary effect is the cause — consistent with the architecture (five
stacked padded 3×3×3 convolutions over 12 candidates) … but not isolated by an
ablation here."

**Direct relevance.** ARCH-001 is that missing ablation. Stage A identified the
mechanism but could not isolate it; §5 of `ARCHITECTURE_CONTROL_DESIGN.md`
establishes analytically and synthetically that the mechanism is
weight-independent, and ARCH-001 would measure it directly on the real pipeline.

**Not modified.**

---

## 2. `EXP-CORRESPONDENCE-INDEX-001` — two executions

**Records:** `.../20260910T164346Z/` (ESTABLISHED) and `.../20260910T164326Z/`
(NOT-DEMONSTRATED). Both `EXPLORATORY-ONLY` per the provenance audit.

**What ARCH-001 takes from them.** The preregistration of `20260910T164326Z` §8
already derived the ordered prediction *from the architecture* — "five
disparity-axis 3-tap convolutions with zero padding, so output candidate k
depends on input candidates [k−5, k+5] and only k ∈ {5,6} are padding-free" —
and correctly declined to predict a unit slope. §5 of this design confirms that
reasoning from source and from a synthetic operator test, and extends it to the
decisive point those records could not reach: **the same holds for untrained
weights.**

ARCH-001 also inherits from them, unchanged: the four checkpoints, the four focus
scenes, the mask, the `π_m(k) = (k − m) mod 12` convention, the
`index_select`-for-every-arm requirement including identity, the
`d(x) = np.median(disparity_initial[mask])` aggregate, and the algebraic
negative-control gate.

**Neither record was modified, re-run or re-scored.**

---

## 3. Provenance audit — `audit_20260911T011113Z`

**Record:** `phase2/diagnostics/correspondence_index/audit_20260911T011113Z/`

Established that a control must be *calibrated* rather than assumed, and that
neither INDEX-001 execution was admissible as confirmatory evidence. ARCH-001
carries that discipline forward: its random-weight family is a calibrated
reference with 120 internal baseline pairs per unit, not a single draw, and its
decision rule is fixed before any output exists.

**Not modified.**

---

## 4. `EXP-CORRESPONDENCE-INDEX-002`

**Record:** `phase2/diagnostics/correspondence_index/20260911T015124Z/`
**Verdict:** `CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED`

### Values ARCH-001's design read from it (already published; no new output)

| quantity | value | used for |
|---|---|---|
| unit structure | 3 checkpoints × 4 scenes = 12 | §3 unit of analysis |
| ordered arms, all 12 units | `d(−1) < d(0) < d(+1) < d(+2)` strictly | §4, why a four-arm statistic is not predictive |
| `d(+1) − d(0)` | +1.2 … +1.9 | §5, larger than a unit shift |
| `d(+2) − d(0)` | +1.8 … +3.1 | §5, same |
| measured throughput | 24 640 arms in 264.1 s ⇒ 0.0107 s/arm | §12 budget |
| checkpoint sha256 ×4 | see §11 of the design | §10 control K6 |

### What ARCH-001 inherits and what it changes

| element | INDEX-002 | ARCH-001 |
|---|---|---|
| intervention | `π_m(k) = (k−m) mod 12` | **unchanged** |
| arms | `m ∈ {−1,0,+1,+2}` | **full orbit `m = 0…11`** (recovers the four) |
| response `d(x)` | `np.median(disparity_initial[mask])` | **unchanged** |
| mask, scenes, checkpoints | frozen | **unchanged** |
| what varies | the permutation | **the aggregation weights** |
| null | 512 random-permutation triples | **none — paired trained-vs-random, no p-value** |
| verdict logic | `p ≤ 0.01` + 99 % envelope | **descriptive paired separation** |

**Not modified, not re-run, not re-scored.** Its `null_pass 0/12` remains its
result.

---

## 5. INDEX-003 design audit — the direct parent

**Record:** `phase2/diagnostics/correspondence_index/design_audit_20260911/`
**Verdict:** `DESIGN-INCONCLUSIVE`

That audit found a clean ordering-vs-permutation null exists but blocked the
design because "the ordered response is an expected consequence of the
aggregation's five zero-padded 3-tap candidate-axis convolutions and would be
produced by arbitrary weights", and made an architecture-only control its
**mandatory precondition** (its §9.14), with a pre-fixed interpretation table.

**ARCH-001 is that precondition, executed as a standalone experiment.** Its
CASE A / B / C map onto the audit's interpretation table exactly.

**One statistic is carried over and then rejected.** The audit proposed circular
rank total variation as INDEX-003's primary. §G of the ARCH-001 brief required
that proposal to be re-audited rather than inherited, and it **fails** for this
comparison: §6 of `ARCHITECTURE_CONTROL_DESIGN.md` shows it is floor-pinned at
22 for every structured orbit, because a wrapped ramp is monotone around the
circle whatever its phase or sharpness. TV is retained as a caveated descriptive
only.

This is **not** a criticism of the audit: TV was designed for a different
contrast (natural generator vs random generator), where it is appropriate. It is
unfit for trained-vs-random under the *same* generator.

**The audit record was not modified.**

---

## 6. Shared provenance

| key | sha256 | used by |
|---|---|---|
| `NEG_shift_none` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` | Stage A, INDEX-001 ×2, INDEX-002, ARCH-001 (planned) |
| `POS_6b_seed0` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` | INDEX-001 ×2, INDEX-002, ARCH-001 (planned) |
| `POS_6b_seed1` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` | INDEX-001 ×2, INDEX-002, ARCH-001 (planned) |
| `POS_6b_seed2` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` | INDEX-001 ×2, INDEX-002, ARCH-001 (planned) |

**No checkpoint was loaded by this design task.** No checkpoint has ever been
retrained, modified or added anywhere in this line, and no extra trained seed has
ever been introduced.

ARCH-001 would add **16 untrained random aggregations** (control seeds 0–15).
These are not checkpoints, are never saved as model state, receive no training,
and exist only as a within-run control.

---

## 7. Source files read by this design task (read-only)

| file | what was verified |
|---|---|
| `src/models/stereonet/aggregation.py` | `4 × Conv3d(32→32, 3×3×3, padding=1)` + LeakyReLU(0.01), then `Conv3d(32→1, 3×3×3, padding=1)`; 111 585 parameters |
| `src/models/stereonet/stereonet.py` | `agg_in = 32`, `num_disparities = 12`, `feature_stride = 16` |
| `src/models/stereonet/regression.py` | `soft_argmin` over the candidate index; index grid `0…11`, linear not circular |
| `phase2/models/scaled_regression.py` | `StandardisedDisparityRegression` has **zero** parameters; `forward` is interpolate → standardise → soft_argmin |
| `src/models/stereonet/cost_volume.py` | `shift="left"`: level `k` at column `u` = `left_feat[u+k] − right_feat[u]`; `shift="none"` is a no-op |
| `20260911T015124Z/results.json` | unit structure, ordered-arm values, throughput |

`src/` is bit-identical to the frozen Phase-1 tag and was not touched.

---

## 8. Artifacts in this design record

| file | role |
|---|---|
| `ARCHITECTURE_CONTROL_DESIGN.md` | the design + pre-flight document |
| `design.json` | machine-readable form |
| `RELATED_RUNS.md` | this file |
| `preflight_architecture_analysis.py` | the §H / §G synthetic analysis, re-runnable; loads no checkpoint, no scene, no repo model class |

---

## 9. Standing state of the claim

Unchanged by this design task, which produced no evidence about the trained
model:

- **Established:** right-image dependence; the algebraic negative-control gate
  (bit-exact 0.0 under every permutation tested); the ordered directional
  response, reproduced at 12/12 units in three independent executions.
- **Unresolved:** candidate-coordinate sensitivity (level C). INDEX-002's
  preregistered test did not pass and is not overturned.
- **Not justified:** geometric correspondence (level D) and genuine disparity
  search (level E). Neither is designed, preregistered nor launched.
- **Newly specified, not yet measured:** whether the ordering survives untrained
  weights. §5 of the design argues analytically and synthetically that it should;
  **that is an expectation, not a measurement**, and ARCH-001 exists to replace it
  with one.
