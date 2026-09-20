# RECOMMENDATION — post-GEOM-001 successor

Companion to `AUDIT.md` and `SUCCESSOR_OPTIONS.md`. **Nothing executed. Not
launched. GEOM-001 unmodified.**

---

## DECISION

```
SUCCESSOR-DESIGN-JUSTIFIED
```

**Recommended design: Option B — synthetic fronto-parallel self-pair sweep,
on held-out scenes.**
Proposed identifier: **`EXP-CORRESPONDENCE-GEOM-002`**.

---

## WHY A SUCCESSOR IS JUSTIFIED

1. **The question is unanswered, not answered negatively.** GEOM-001 failed on a
   representability defect: **DERIVED**, 99.8 % of its mask could not exhibit the
   hypothesis at `Δ = 48`. Its `+32` and `+48` conditions were not a test that
   failed; they were not a test. Level D has still never been measured on
   non-degenerate weights.
2. **The defect is removable by construction, not by patching.** Option B *sets*
   the disparity instead of perturbing it, so the constraint of `AUDIT.md` §1
   cannot bind at any level of the sweep.
3. **Genuinely unmeasured on both available axes.** The synthetic self-pair
   response has never been measured on any checkpoint in this campaign, and 36 of
   40 `hailo_val` scenes have never been used (**MEASURED**).
4. **It is cheap and falsifiable.** ~1–2 minutes of inference, six levels, a
   monotonicity constraint of `1/720` per unit against Option A's `1/2`, and an
   anchor of exactly 1 candidate per candidate that follows from the construction
   rather than from any observation.

## WHY NOT THE OBVIOUS ALTERNATIVE

Rebuilding the symmetric translation sweep is **not** recommended. **DERIVED**
from GT alone: the maximum symmetric offset with ≥ 50 % retention is ±16 px on
the frozen scenes, ±32 px only with a band mask and scene selection, and ±48 px
is unreachable on **all 40 scenes**. That leaves at most two signed offsets, so
the monotonicity condition degenerates to a single coin flip. **The
symmetric-translation statistic class is exhausted on this dataset** — a
successor there would be weaker than the experiment it replaces.

---

## HONEST CEILING ON WHAT IT CAN ESTABLISH

**No experiment available with the current models can be confirmatory in the
strict sense.** Training is forbidden, so the three checkpoints are the entire
population and their reuse is irreducible; and the auditor has read the GEOM-001
response curve (`AUDIT.md` §0). The achievable status is:

> **strong consistency evidence with real predictive content on the data and
> intervention axes — not independent confirmation.**

This must be written into the preregistration *before* execution, not attached to
the result afterwards.

**A negative result is ambiguous** (constant disparity is out of distribution).
**A positive result is informative but bounded**: it would show the pipeline can
locate a horizontal correspondence on the easiest possible input, not that it
does so on real stereo.

---

## THE PROPOSED EXPERIMENT — specification for a fresh preregistration

**Not preregistered here. Not authorised. Not executed.**

### Experimental question

When the right image is a pure horizontal translation of the left image by `t`
pixels — a stereo pair whose true disparity is exactly `t` everywhere — does
`disparity_initial` track `t/16` candidates, specifically on the horizontal axis,
beyond what a provably search-free model produces?

### Independent variable

`t ∈ {16, 32, 48, 64, 80, 96}` px (1…6 candidates), plus `t = 0`.
Axis ∈ {horizontal, vertical}. 13 distinct conditions (`t = 0` shared).

Crop-based construction, **no fill, no padding, no interpolation**:

```
left  crop : rows [96, 272), cols [96, 1136)                fixed
right crop : horizontal  rows [96, 272), cols [96−t, 1136−t)
             vertical    rows [96−t, 272−t), cols [96, 1136)
crop size  : 1040 × 176 = (65 × 16) × (11 × 16)             identical every condition
```

**DERIVED**: stride-aligned, every `t` a multiple of 16 so the feature
translation is exact; every crop window inside the image; predicted response 1…6
on a 0…11 axis, clear of both rails.

### Units and scenes

3 weight sets (`POS_6b_seed0/1/2`, `shift="left"`, standardised readout) ×
**N held-out scenes drawn from the 36 never used**, selected by a GT-free rule
fixed in advance (e.g. the first N by index excluding `{27, 0, 31, 6}`). Units
are **not** independent replicates: they share three weight sets. Deterministic
causal diagnostic, not population inference.

### Controls

- **Matched negative axis:** vertical self-pair at the same `t` — same image,
  same displacement, same crop, same code path, same mask. **No predicted slope
  is assigned to it** (`AUDIT.md` §8 failure mode 7).
- **Search-free artifact baseline:** the frozen `shift="none" + standardised`
  checkpoint on the identical sweep. **Measured, never assumed zero.**

### Mask

Purely geometric: the crop interior minus a fixed border, frozen in advance.
**No ground truth is used** — the true disparity is `t` by construction and
uniform over the crop. This eliminates the band mask, the scene-disparity
confound and the mask-occupancy issue that have recurred since INDEX-001.

### Statistic

```
m_axis(t)  = median( disparity_initial[mask] )                 candidate units
Slope_axis = Σ_t (m_axis(t) − m_axis(0)) · (t/16)  /  Σ_t (t/16)²
ANCHOR     : Slope_h = 1 candidate per candidate               exact, by construction
```

No p-value: the conditions are not exchangeable and the units share weights.
Paired and descriptive throughout.

### Decision rule — threshold-free, unit-level

Per unit:

```
C1  Slope_h > 0                                      correct direction
C2  m_h(t) strictly increasing over t = 0,16,…,96     6-level monotonicity  (1/720 under the null)
C3  |Slope_h| > |Slope_v|                             axis specificity
C4  Slope_h > Slope_h(NEG, same scene)                exceeds the search-free baseline
```

PASS iff `C1 ∧ C2 ∧ C3 ∧ C4` at **every** unit. No partial counts, no majority
rule, no threshold relaxation, no post-hoc exception. `Slope_h` is reported
against the anchor `1.0` **descriptively only** — magnitude is never a pass
criterion.

### Claim ceiling — to be frozen verbatim

On PASS:

> "On synthetic fronto-parallel stereo pairs, the trained model's initial
> disparity tracks the imposed horizontal displacement, specifically on the
> epipolar axis, beyond a matched vertical control and a provably search-free
> baseline."

Status: **consistency evidence for horizontal correspondence on synthetic
input.** It does **not** establish correct disparity on natural stereo, genuine
full disparity search, sub-candidate precision, final stereo correctness, or
generalisation.

On FAIL: `SYNTHETIC-CORRESPONDENCE-NOT-DEMONSTRATED` — **and it must not be
concluded that correspondence is absent**, because the input is out of
distribution.

### Preregistration requirements

1. Freeze the `t` set, axes and crop geometry → `translation_spec.json`, hashed
   **over file bytes** (the GEOM-001 CRLF lesson).
2. Freeze the mask → `mask_spec.json`, hashed over file bytes.
3. **Freeze a representability check that consults the frozen specs themselves**,
   and state the predicted response range against the 0…11 axis. GEOM-001's
   defect was importing a figure from a different mask; the check must use the
   artifacts in its own record.
4. Freeze the held-out scene list by an index rule fixed before any scene is
   loaded.
5. Freeze the geometry derivation from source; freeze the statistic; freeze the
   decision rule; freeze the controls.
6. Write `PREREGISTRATION.md` containing all hashes — **then** execute. No result
   inspected between writing and execution.
7. Record in the preregistration, before execution:
   - that the designer has read the GEOM-001 response curve (`AUDIT.md` §0);
   - that the three checkpoints are reused and cannot be replaced;
   - that the achievable status is **consistency evidence, not confirmation**;
   - that a negative result is ambiguous because the input is OOD.
8. New immutable record `phase2/diagnostics/correspondence_geom/<UTC>/`.
   `RESULTS.md` created only after execution. GEOM-001 untouched.

### Compute

4 models × N scenes × 13 conditions forward passes. At N = 4, **208 passes,
~1–2 minutes**, inference only, no training, no new checkpoint.

---

## WHAT THIS WOULD ADD

**UNKNOWN today, and answerable for about a minute of compute:** whether these
checkpoints can locate a horizontal correspondence *at all* when one is
unambiguously present. Every prior experiment in this campaign either measured a
quantity confounded with architecture (INDEX-001/002/003, ARCH-001) or failed on
a design defect (Stage A, GEOM-001). **None has ever established whether the
correspondence machinery functions on an input where the answer is known and
unambiguous.**

That is a genuine gap, it is cheap to close, and — unusually for this
investigation — the experiment is a strong falsifier rather than a weak
confirmer.

---

## FINAL

```
SUCCESSOR-DESIGN-JUSTIFIED
Recommended: EXP-CORRESPONDENCE-GEOM-002 — synthetic fronto-parallel self-pair
sweep, held-out scenes, six levels, matched vertical and search-free controls.
```

**Not preregistered. Not authorised. Not executed. Awaiting your decision.**
