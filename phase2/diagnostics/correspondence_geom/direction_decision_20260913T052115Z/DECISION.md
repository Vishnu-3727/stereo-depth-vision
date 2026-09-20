# FINAL RESEARCH-DIRECTION DECISION GATE — record

**Id:** `direction_decision_20260913T052115Z`
**Type:** decision gate. No experiment was executed, no checkpoint opened, no
weight trained, no historical record modified.
**Authority for O1:** `o1_final_blind_audit_20260913T050232Z` (verdict
C — NOT IDENTIFIABLE, DO NOT RUN). O1 is closed and is not revisited here.

---

## 1. Current scientific state

### Demonstrated (trustworthy, measured)

* **A — right-image dependence.** Established.
* **B — a binocular pathway exists.** Established.
* **Cost-volume geometry.** `V[k,w] = Lf[w+k] − Rf[w]`; for a translated
  self-pair `V[τ] = 0` **exactly** (3,525,120 cells, tolerance exactly zero,
  3 extractors × 12 scenes × 6 τ). Re-measured independently in the O1 audit.
* **Extractor halo.** Receptive field 477 px; measured halo 15 cells left,
  14 right; corrected mask `x ∈ [325,633) ∧ y ∈ [64,208)`, validated by an
  invariance test (bit-identical 24/24 under injected halo content).
* **Padding reach.** Five padded 3-taps ⇒ candidates `k ∈ {5,6}` are the only
  padding-free ones; `k=5`/`k=6` aggregated costs are bit-identical.
* **2×2 pairing-swap cancellation** is exact and unconditional
  (`1.809e-07` on real features); the interaction's signal channel is the
  LeakyReLU, not the padding.
* **Random-weight behaviour.** Near-uniform softmax, `m → ~5.1–5.5`,
  `0/576` units tracking within 1.0 candidate; `R² = 0.867` for `Φ(k−τ)` vs
  `0.289` for `Ψ(k)` in 574/576 units.
* **Frozen-architecture expressibility.** A candidate-**pointwise**
  `Σ_{c<16}|V_c|` detector is exactly representable in the unmodified
  `Aggregation` (`LReLU(x)+LReLU(−x) = 0.99|x|`, residual 4.44e-16) and
  reproduces the full apparent O1 signature.

### Only suggested (never established)

* That the trained aggregation performs matching. `EXP-CORRESPONDENCE-TR-001`
  reached **C′** only: the trained candidate-axis ramp (slope 0.95–1.22) is
  exactly the value the architecture forces anyway.
* That sharpness, candidate-index structure, or right-image response indicate
  correspondence. None of these implies it.

### Falsified / retired (do not silently upgrade)

* **Image-translation slope test** — REJECTED; the degenerate mechanism imitates
  sign and monotonicity.
* **Vertical shift as a null** — INVALID on two independent grounds (no axis
  asymmetry for `shift="none"`; and `FH = 17 < 30` cells makes a halo-free
  vertical band empty, unfixable at any crop size below 496 px).
* **Cyclic candidate re-indexing with unit-slope predictions** — INVALID.
* **The whole synthetic global-shift self-pair class** — retired by the GEOM-002
  failure audit: with `Rf[w] = Lf[w+τ]` the substitution `u = w+τ`, `d = k−τ`
  makes every cell a function of `(u,d)`, so **τ is a coordinate change, not an
  experimental factor**; every candidate-axis-translation-invariant statistic is
  dead on it.
* **O1 / candidate-axis permutation on that class** — NOT IDENTIFIABLE.
* **"Permutation is the one airtight intervention"** — corrected; airtight only
  as a null for the degenerate `shift="none"` volume.
* **`max`-over-units gate rules** — invalid; `P(pass) = 1.4e-14` measured for
  GEOM-002 before it ran.
* **Weight-randomised nulls as mechanistic controls** — a floor, not a control;
  they fail for flatness, so any sharper model inherits a pass.

### Invalid / no determination

* **GEOM-001** — halted at HS1, no admissible result either way.
* **GEOM-002** — G-STOPped at Stage 1, `H*` undefined, trained weights never
  loaded; its single null may not be reused for a different statistic.

### Unidentified

Everything at Level D and above. **Evidence for genuine geometric
correspondence: none. Evidence against: none.**

---

## 2. Unresolved question (narrowest form)

> Does the trained aggregation treat the candidate index as a **spatially
> consistent geometric displacement** — i.e. does it propagate a *matched*
> neighbour's candidate position onto a pixel that has no candidate-resolving
> evidence of its own — as opposed to acting as a per-pixel candidate-axis
> function whose behaviour is explained by architecture, readout, or
> correspondence-free image dependence?

This is the narrowest question whose answer moves the ceiling off D.

---

## 3. Path A assessment — **VIABLE, and the strongest available**

### 3.1 A candidate-local decoy is impossible. Proven, and data-independent.

`ΔV[k,w] = ΔLf[w+k] − ΔRf[w]`. The map from image space to `(k,w)` is a
**shear**:

* a right-image edit at column `w` is **k-constant** — identical on every
  candidate;
* a left-image edit at column `u` lands at `(k, w = u−k)` for **every** `k` —
  one cell in each candidate, a diagonal;
* a candidate-local mark at fixed `w` requires editing `V` directly, which is no
  longer a stereo pair.

This obstruction is a property of `build_cost_volume`, **not** of the self-pair
construction, so it survives any change of dataset. Further, any edit that *does*
make `‖V[k_d,w]‖` small is by definition a photometric match at `k_d` — a genuine
ambiguity a real matcher must also face, not a decoy. **The literal Path-A decoy
requirement is unachievable and is formally rejected.**

### 3.2 The equivalent mechanism that *is* achievable

A decoy exists to make the correspondence-free class fail. The same effect is
obtained analytically instead: **hold the tested pixel's own cost slice
bit-identical across arms and vary only the imposed disparity of its
spatial context.**

Construction (CONTEXT-PROPAGATION, "CP"):

1. Real KITTI 2015 pair, left image **untouched** in every arm.
2. In the right image, replace a band at feature-column distance `Δ` from the
   tested column `w` with the left image's content translated by `t = d_s·16` px.
   The band therefore carries an exactly known disparity `d_s` candidates
   (`V[d_s] = 0` inside it, by the already-verified identity).
3. Choose `Δ` in the **separation band**: far enough that the extractor halo
   never reaches `Rf[w]`, close enough that the aggregation's 5-cell spatial
   reach does reach the band's features. Nominally `Δ ∈ [15,19]` cells for a
   right-side band (`Rf[w]` unchanged needs `Δ > 14`; `Rf[w±5]` affected needs
   `Δ − 5 ≤ 14`). **Width ~5 cells (~80 px). This bound is the single highest-risk
   premise in the design and MUST be measured, not derived** — GEOM-001 halted on
   exactly this class of assumed bound.
4. Tested pixels are selected **model-free**, from the cost volume alone, as
   those whose own candidate profile is flat (no self-evidence): a textureless
   patch. A contrast arm uses unambiguous (textured) tested pixels.
5. Arms: `d_s ∈ {a contiguous interior set}`, plus (i) a **decorrelated-band**
   control (band filled with unrelated content translated by the same `t`, so no
   candidate in the band is a match) and (ii) a **direction/vertical** control.

### 3.3 The ten required tests

| # | Test | Result |
|---|---|---|
| 1 | Distinguishes correspondence from a candidate-latch? | **YES, analytically.** Any candidate-pointwise mechanism receives a **bit-identical** input at the tested pixel across arms, so its slope is **exactly 0** by construction — not empirically, provably. This is the exact class that defeated O1. |
| 2 | Survives the frozen cost-volume construction? | **YES.** Only images are edited; `build_cost_volume`, `Aggregation`, `scaled_regression` and the mask are untouched. |
| 3 | Avoids the translated-self-pair zero-slice degeneracy? | **YES at the measurement point.** The degeneracy is confined to the *context band*, where it is the delivery mechanism for a known `d_s`; the tested pixel is never a self-pair cell. |
| 4 | Avoids candidate-index / readout confounding? | **LARGELY.** Arms share identical candidate indices, identical readout, identical mask, and an identical tested-pixel slice, so centre-pull and padding contribute *equally* to every arm and cancel in the slope. Residual risk: padding contamination is `d_s`-dependent (only `k∈{5,6}` are clean) — must be bounded by restricting `d_s` to an interior set and by an explicit pre-check. |
| 5 | Correspondence-preserving vs correspondence-breaking comparison? | **YES.** Matched band (`V[d_s]=0`) vs decorrelated band (no matching candidate), with identical band content statistics and identical translation. |
| 6 | Null generated under the same nuisance structure? | **YES, and it is analytic.** Primary null = the pointwise-latch class, slope identically 0. Secondary nulls: decorrelated band (same edit, same translation, no match), random weights (floor only), vertical band (contaminated null / downgrade trigger only). |
| 7 | Executable without changing the question? | **YES.** Same model, same checkpoints, same pipeline, inference only. |
| 8 | Interpretable under both hypotheses? | **YES.** H_match/H_propagate: slope ≈ +1 candidate per candidate of `d_s`, large on ambiguous pixels, **small on unambiguous ones** (an internal control with the opposite predicted sign). H_latch: slope 0 on both. |
| 9 | Can a correspondence-free frozen-architecture function reproduce it? | **NO for the pointwise class** (proved). The surviving adversary is "spatially smooth, then latch" — but that mechanism must read a *neighbour's candidate position* and re-use it at this pixel, which is precisely geometric candidate-axis use, i.e. it is inside the hypothesis, not outside it. |
| 10 | Would a positive result raise the ceiling above D? | **YES — to D (geometric candidate-axis use).** A slope of ≈ +1 that appears only for a matched band, only for ambiguous tested pixels, with a provably-zero analytic null, is not explainable by candidate-index weighting, soft-argmin centring, padding, sharpness, or generic right-image dependence. |

### 3.4 Why the previously fatal symmetry does not apply

GEOM-002 died because a *global* translation makes `τ` a coordinate change. The
CP intervention is **local**: the tested pixel and the band are at different
columns, so no global substitution `u = w+τ` exists, and the statistic is not
invariant under candidate-axis translation (translating the whole profile changes
the slope's intercept, not its value — the slope is taken *across* arms that share
one tested pixel, so a common translation cancels and a `d_s`-linked translation
is the signal). `HS-SYM` is expected to pass, and must be run before freezing.

### 3.5 Honest ceiling of Path A

Even a perfect CP result **cannot** reach Level E / Level 5 for the aggregation
module, because the only binocular operation (`L − R`) is performed by frozen,
un-learned code upstream of it. That cap is structural and applies to every
Phase-2 module-level experiment. Path A buys **D**, not E.

---

## 4. Path B assessment — **REJECTED as the next step (not needed, and weaker)**

* **New data are not required.** Real KITTI 2015 stereo pairs with ground-truth
  disparity, non-identical left/right images, spatially varying disparity,
  repeated texture and occlusions are **already in the project** (`hailo_val`
  split, used throughout Phase 1). The "missing data" premise is false.
* On *unedited* real pairs the tested pixel's own evidence cannot be held
  constant across arms, so the analytic null of §3.3(1) is unavailable and the
  latch counterexample survives — strictly weaker than Path A.
* Every statistic available on unedited real pairs without an intervention
  (EPE, D1, sharpness, profile shape) is prohibited as correspondence evidence
  by the hard rules. Path B without an intervention is not an experiment.
* Path B therefore reduces to "Path A on data we already have", plus an
  unnecessary acquisition and training cost.

---

## 5. Path C assessment — **REJECTED**

A feasible, decisive intervention exists (§3) within project scope, requires no
new data, no training and no architecture change, and would raise the ceiling.
Closing the branch now would discard that. Closure is not warranted; note also
that the *synthetic global-shift class* is already closed, and Path A does not
belong to it.

---

## 6. Chosen path

# PATH A — NEW EXISTING-DATA INTERVENTION

**Name:** `EXP-CORRESPONDENCE-CP-001` — context-propagation with a bit-identical
tested-pixel slice.

**Strongest because** the correspondence-free class that defeated O1 is reduced
to an *exact zero* by construction rather than by argument; the intervention is
local, so the translation symmetry that killed GEOM-002 does not exist; and the
tested pixel's own evidence is held bit-identical, which cancels padding,
centring and sharpness between arms instead of requiring them to be modelled.

### Work package (minimum)

* **Intervention.** Left image untouched. Right-image band at `Δ` cells from the
  tested column replaced by left content translated by `t = 16·d_s`.
* **Primary measurement.** Slope of the tested pixel's predicted candidate
  (frozen `scaled_regression` readout, frozen mask discipline) against the
  imposed context disparity `d_s`, on **ambiguous** tested pixels.
* **Hypothesis distinguished.** H_propagate (geometric candidate-axis use)
  vs H_latch (per-pixel candidate-axis function).
* **Controls, all mandatory:** (a) analytic pointwise-latch null — slope must be
  exactly 0; (b) decorrelated band, same translation, no matching candidate;
  (c) unambiguous tested pixels — predicted *smaller* slope, opposite direction
  to a generic-response confound; (d) random weights, floor only; (e) vertical
  band, contaminated null / downgrade trigger only.
* **Null.** Pointwise-latch (analytic, exactly 0) as primary; decorrelated band
  as the empirical nuisance-matched null.
* **Inference unit.** Checkpoint (`n = 3`, the same exhausted three — declare
  it). Scenes and `d_s` levels are repeated measures, **not** replications.
* **Gate.** Rate/quantile, preregistered, never `max`. No threshold is set in
  this record — it must be derived from the decorrelated-band null's measured
  spread before any trained arm is loaded.
* **Mandatory pre-freeze hard stops (model-free, seconds to run):**
  `HS-BAND` — measure the separation band; assert `V[:, :, w_test]` is
  **bit-identical** with and without the band edit, and assert the band's
  features do change within the 5-cell aggregation reach. If the band is empty,
  **halt** — the design is dead and Path C becomes correct.
  `HS-SYM` — assert the primary statistic is not invariant under the
  intervention's symmetry group, numerically, on random tensors, no model.
  `HS-AMBIG` — assert the tested pixels' own profiles are flat, from the cost
  volume alone.
  `HS-NULL-RATE`, `HS-NULL-MATCH`, `HS-DEGENERATE`, `HS-CEILING` as already
  specified in the GEOM-002 failure audit.
* **Stopping rule.** Any hard stop halts the process and the experiment is not
  repaired and continued. If `HS-BAND` fails, record "correspondence not
  demonstrated, class exhausted" and close.

### Requirements

| Question | Answer |
|---|---|
| New data required? | **NO** — KITTI 2015 `hailo_val` already present. |
| New training required? | **NO** — inference only. |
| Existing checkpoints sufficient? | **YES** — the three `shift="left"` H2 seeds. |
| New architecture required? | **NO.** |

### EXECUTION STATUS: **NOT READY**

Not ready for one reason only: `HS-BAND` is unmeasured. The `Δ ∈ [15,19]` band is
derived from a 15/14-cell halo and a 5-cell aggregation reach — the same kind of
derived bound that halted GEOM-001. It must be measured before a preregistration
is written, and it is ~5 cells wide, so it may not survive measurement.

---

## 7. What would raise the claim ceiling

| Level | Required evidence |
|---|---|
| 1 — candidate ordering / coordinate sensitivity | Output changes when candidate coordinates change. **Already available, and partly architectural** (3-tap + padding). Not worth measuring again. |
| 2 — candidate-neighbourhood dependence | Sensitivity that survives coordinate restoration. **Already present at random initialisation** (measured non-zero in 48/48 cells). Not evidence of anything learned. |
| 3 — learned candidate-index structure | Effect exceeding the untrained architectural baseline on a candidate-resolved readout. **Already held** by TR-001's C′ and GEOM-002's `R² = 0.867`. Re-measuring it adds nothing. |
| 4 — **geometric candidate-axis use (= Level D)** | A candidate-axis effect that (i) requires a *matched* neighbour, (ii) vanishes when the neighbour is decorrelated, (iii) is provably zero for every candidate-pointwise mechanism, and (iv) is measured with the tested pixel's own evidence held bit-identical. **This is exactly CP-001's primary statistic.** |
| 5 — genuine correspondence / disparity search | Requires the tested module to *perform* the left/right comparison. **Structurally unreachable** while `L − R` is frozen upstream of the aggregation. Reachable only as a claim about the *pipeline*, and only with 4 plus an independent demonstration that the cost volume's minimum is the geometric match on non-degenerate pairs. |
| 6 — natural-scene disparity behaviour | Held-out real scenes, GT disparity, occlusions and repeated texture, with the candidate-axis result of 4/5 shown to govern `disparity_initial` — not `disparity_final`, which is left-image-guided and supplies nearly all the magnitude. |

The 3 → 4 step is the whole remaining question. **Candidate-axis sensitivity is
not correspondence**, and no amount of 1–3 evidence aggregates into 4.

---

## 8. Experiments explicitly prohibited next

| Prohibited | Why |
|---|---|
| Any O1 / candidate-axis permutation variant | Verdict C, non-identifiable; the pointwise latch reproduces the signature. |
| More O1 seeds, permutations, or checkpoints | `S8`'s sign and magnitude are set by `π`, `τ` and readout blur; more samples measure the confound more precisely. |
| Any `S8` threshold, new or tuned | No principled derivation exists; `S8 ≥ 1` is a statement about `τ`. |
| Another random-init null on the same statistic | It is a flatness floor, not a mechanistic control; already measured. |
| Any candidate-order statistic on translated self-pairs | The class is retired: `τ` is a coordinate change. |
| More translated-self-pair repetitions | Adds τ-coordinate views, not samples. |
| Another audit of the same identifiability failure | Three days spent; the failure is established and closed. |
| A literal candidate-local decoy on `V` | Impossible by the shear algebra; editing `V` directly is not a stereo pair. |
| EPE / D1 / sharpness / gradient-magnitude arguments | Prohibited inferences 3, 4, 5. |
| Any experiment whose purpose is a more impressive positive signature | Rule 14. |

---

## 9. Final state

**Current claim ceiling: Level D — genuine geometric correspondence
NOT-DEMONSTRATED.** Not demonstrated is not absent; the O1 counterexample shows
what *could* produce the signature, never that the trained network does.

### EXECUTION DECISION

**DO NOT EXECUTE YET — DECISION GATE ONLY.**

Next action: measure `HS-BAND` (model-free, no checkpoint, seconds) and, if the
separation band is non-empty, write the `EXP-CORRESPONDENCE-CP-001`
preregistration. If it is empty, Path C becomes correct and the branch closes.
