# ALGEBRA-ONLY PRE-DESIGN AUDIT — successor to GEOM-002

Record `phase2/diagnostics/correspondence_geom/predesign_audit_20260911/`.
2026-09-11.

**Audit only. Nothing executed, nothing trained, no inference, no model
instantiated, no experiment designed.** No `t` levels, scenes, masks, thresholds
or execution details are chosen. GEOM-001, GEOM-002 and every historical record
are untouched.

**Anti-post-hoc compliance.** Every result below is derived from the frozen
construction, the source code, and symbolic algebra. **No GEOM-002 response value
is used to fit, choose, calibrate or justify any statistic.** Observed values
appear nowhere in this document's derivations.

Tags: **MEASURED** (source/frozen record), **DERIVED** (algebra), **INFERRED**,
**UNKNOWN**.

---

## HEADLINE FINDING

The audit found a result that changes what a successor can hope to test:

> **Under the frozen synthetic construction, a unit-slope ramp is
> architecturally forced. Ideal correspondence (A) and random aggregation weights
> (E) predict the *same* leading-order observable, `m(τ) ≈ τ + const`. The slope
> cannot discriminate them, whatever its value.**

This is §1.6 below. It is derived from shift-equivariance alone and holds for
arbitrary weights. It means the random-weight arm is **not a control but a
gate**: it determines whether the observable is diagnostic at all.

---

## 1. RESPONSE SHAPES, DERIVED SYMBOLICALLY

### 1.1 The construction, restated

**MEASURED** (`construction_spec.json`, `cost_volume.py`, `feature_extractor.py`):
right image is `L` shifted so the imposed disparity is exactly `t` px; stride is
`2⁴ = 16`; the extractor is fully convolutional, so for `t` a multiple of 16 and
`τ := t/16`,

```
Rf[w] = Lf[w + τ]                                (exact, interior)
V[k][w] = Lf[w+k] − Rf[w] = Lf[w+k] − Lf[w+τ]
```

### 1.2 The key substitution (DERIVED)

Put `k = κ + τ`:

```
V[κ+τ][w] = Lf[w+τ+κ] − Lf[w+τ] = F[κ][w+τ]
where  F[κ][w] := Lf[w+κ] − Lf[w]     is INDEPENDENT of τ
```

so

```
V(k, w) = F(k − τ, w + τ)
```

**The cost volume is a τ-independent object `F`, translated by `τ` along the
candidate axis and by `τ` along the spatial axis.** `V` is exactly zero at
`κ = 0`, i.e. `k = τ`.

### 1.3 A — ideal correspondence

The readout reports the candidate at which the features match, `k = τ`:

```
m_h(τ) = τ + c          slope exactly 1 in candidate units
```

**Refinement (DERIVED):** the soft-argmin is a softmax-weighted mean over a finite
support, so if the cost profile is a dip of finite width the response is pulled
toward the support's centre. The prediction is therefore **affine in τ with
slope 1 in the padding-free interior, compressing toward the rails**, not the
identity map. The invariant that survives is the **constancy of the first
difference**, `m(τ+1) − m(τ) ≈ 1`.

### 1.4 B — generic horizontal displacement sensitivity

`m_h(τ) = g(τ)`, unconstrained. No prediction on first differences, sign,
monotonicity or linearity.

### 1.5 C — vertical displacement

**DERIVED.** `V[k][y,w] = Lf[y,w+k] − Lf[y−τ, w]`. The cost volume compares
**within a row**; a vertical shift moves content off-row, where no `k` can reach
it. The substitution of §1.2 has no vertical analogue: there is no `κ` making `V`
vanish. **No candidate-axis correspondence exists**, so no mechanism ties `m_v` to
`τ`. Any shape — step, flat, non-monotone — is admissible.

**Declared caveat:** roads, lane markings and horizon lines are vertically
self-similar, so a partial spurious match at some `τ` is possible. The vertical
arm is a magnitude-matched null and must never be assigned a predicted slope.

### 1.6 E — random aggregation weights — **the decisive case**

`Agg` is `Conv3d` layers with a pointwise nonlinearity: **shift-equivariant in
`(k, y, x)` up to padding, for any weights**. Applying it to §1.2:

```
Agg_W(V)(k, w) = Agg_W(F)(k − τ, w + τ)        padding-free interior, ANY W
```

Standardisation is per-pixel across `k` and commutes with a translation of `k`.
A soft-argmin of a profile translated by `τ` returns the untranslated value plus
`τ`. Hence

```
m(τ) = τ + g(w + τ)
```

where `g` depends on the weights and spatial position but **not on `τ`** except
through the spatial drift `w + τ`.

```
⇒  m(τ) ≈ τ + const     FOR ANY AGGREGATION WEIGHTS, TRAINED OR RANDOM
```

**Therefore A and E predict the same slope. A unit-slope ramp on the horizontal
arm is architecturally forced and is not evidence of learning.**

Caveats, all of which *weaken* the effect and none of which create it: only
`k ∈ {5,6}` are padding-free (receptive field 11 over `D = 12`); clipping at the
rails breaks the translation; `g` drifts with the spatial shift.

**Where a discriminator could still live (INFERRED).** Under A the trained
aggregation makes the cost *minimal* at `κ = 0`, so the profile is a sharp dip,
the soft-argmin concentrates, and `g` is small and stable. Under E the sign and
shape of `Agg_W(F)` at `κ = 0` are arbitrary and **seed-dependent**. The
discriminator is therefore the **dispersion and shape-fidelity across random
seeds**, not the slope value.

### 1.7 D — search-free `k`-constant volume

**DERIVED.** For `shift="none"`, `V[k][w] = Lf[w] − Rf[w]` for every `k`:
`k`-constant. The substitution of §1.2 is unavailable — there is no `κ` — so the
profile **does not translate with `τ`**. Any `k`-dependence in the output comes
solely from the aggregation's padding boundary, whose shape in `k` is fixed by the
weights; the content only scales and offsets it.

```
⇒ m(τ) may vary with τ through content, but the cost profile does NOT translate
⇒ no unit-slope mechanism exists
```

This is the one case that is cleanly separated by the slope, and it is separated
**by construction**, which is why it was never the informative control.

### 1.8 Summary

| case | mechanism | predicted `m_h(τ)` | slope |
|---|---|---|---|
| **A** ideal correspondence | profile translates, dip at `κ=0` | `τ + c`, affine | **1** |
| **B** generic sensitivity | none | arbitrary `g(τ)` | arbitrary |
| **C** vertical | no `κ` exists | arbitrary (step/flat/noisy) | undefined |
| **D** search-free | profile does not translate | content-driven, no translation | ≉1 |
| **E** random weights | **profile translates — same as A** | **`τ + c'`, affine** | **1** |

**A and E are indistinguishable by slope. D is distinguishable. C is
distinguishable in principle but only by shape, not magnitude.**

---

## 2. SHAPE-SENSITIVE STATISTICS

Let `u = τ ∈ {1,…,6}` and `r_j = m(u_j)`. **All entries below are algebra on
symbolic shapes; no observation is involved.**

### 2.1 Why the frozen statistic failed, in one line

```
GEOM-002:  Slope = Σ (m(t) − m(0))·u / Σ u²          THROUGH THE ORIGIN
           ramp r = a·u      →  a
           step r = h        →  h·Σu/Σu² = (21/91)h = 0.2308 h
```

**The through-origin constraint absorbs a step's height into the slope.** Freeing
the intercept removes the defect exactly, and this follows from the definition —
not from any observed curve.

### 2.2 Audit of candidate families

| # | statistic | definition | ideal corr. | constant step `h` | arbitrary monotone | additive bias | mult. scale | rail clipping | defeated by an artifact? | a-priori threshold? |
|---|---|---|---|---|---|---|---|---|---|---|
| **S1** | free-intercept slope `α` | OLS of `r` on `u`, intercept free | `α = 1` | **`α = 0` exactly, ∀h** | `α > 0`, uninformative value | **invariant** | scales by `s` | reduced | **yes — §1.6: random weights also give `α ≈ 1`** | **no** (a tolerance on 1 cannot be justified) |
| **S2** | normalised affine residual `1 − R²` | `‖r − (αu+β)‖² / ‖r − r̄‖²` | `0` | `0` (a constant is affine) | `> 0` | invariant | **invariant** | increases | partially | no |
| **S3** | first-difference constancy | `D_j = r_{j+1} − r_j`; require `D_j > 0 ∀j` | all `D_j ≈ 1` | **all `D_j = 0`** | `D_j` varies | invariant | scales | last `D_j` collapse | yes (§1.6) | **YES — strict positivity is threshold-free** |
| **S4** | second difference / curvature | `r_{j+2} − 2r_{j+1} + r_j` | `0` | **`0`** — does **not** separate step from ramp | `≠ 0` | invariant | scales | large | — | no |
| **S5** | Spearman `ρ(u, r)` | rank correlation | `+1` | **ties — undefined/0** | `+1` — does **not** separate from A | invariant | invariant | mild | yes | **YES** (`ρ = 1` is attainable and threshold-free) but it cannot separate A from B |
| **S6** | `min_j D_j / max_j D_j` | scale-free difference ratio | `≈ 1` | `0/0` — undefined | `< 1` | invariant | **invariant** | drops | yes | no |
| **S7** | endpoint/interior consistency | `(r_6 − r_1)/5` vs interior mean `D` | equal | both `0` | differ | invariant | scales | large | yes | no |

### 2.3 What the table establishes

1. **S1 (free-intercept slope) fixes the ramp-vs-step defect by construction** —
   `α = 0` for every step, `α = 1` for the ideal ramp, and the separation is
   exact and data-free. **This is the correct repair of C3, and it is derivable
   without looking at any curve.**
2. **S4 is useless for ramp-vs-step** — a constant has zero curvature, exactly
   like a ramp. Curvature tests *linearity*, not *slope*.
3. **S5 cannot separate A from B** — any monotone response gives `ρ = 1`.
4. **S3 (strict positivity of all first differences) is the only family in the
   table with an a-priori, threshold-free criterion.** It is exactly GEOM-002's
   C2, which was **valid** and was applied to the horizontal arm only.
5. **Every magnitude-bearing statistic (S1, S2, S6, S7) lacks an a-priori
   threshold.** A tolerance around `α = 1`, or a cut on `1 − R²`, cannot be
   justified from geometry. Such statistics can only enter a decision rule
   through **threshold-free comparisons against an empirical control
   distribution**.
6. **Crucially: S1 is defeated by the artifact of §1.6.** Random weights produce
   `α ≈ 1` too. So even the repaired slope is **not** a discriminator between
   trained and random — only between *translating* and *non-translating* volumes,
   i.e. against case D.

### 2.4 The consequence for any successor

> **No statistic of the response curve alone can separate learned correspondence
> from architecturally-forced correspondence, because both produce the same
> affine ramp. Separation requires comparing the trained response against an
> empirically measured distribution of random-weight responses.**

That is a requirement, not a design.

---

## 3. THE RANDOM-WEIGHT ARM

### 3.1 Minimal scientifically useful intervention (DERIVED)

| element | requirement | why |
|---|---|---|
| randomise **aggregation only** | **required** | it is the only module with learned parameters between the cost volume and the readout; 111 585 params |
| **feature extractor frozen** | **mandatory** | randomising it changes `Lf`, hence `F`, hence the `V = 0` structure at `κ = 0`. The comparison would no longer hold the cost volume fixed and §1.2 would no longer apply |
| **readout parameter-free** | **already true** (MEASURED) | nothing to randomise; it must stay identical |
| **identical synthetic inputs** | **mandatory** | the cost volume must be bit-identical between trained and random arms, so that the only difference is the aggregation weights — the same pairing discipline ARCH-001 used |
| **full multi-level curves** | **required** | the statistic is a shape statistic; a single level cannot express shape |
| **multiple seeds** | **required** | the discriminator is the *dispersion* across seeds (§1.6), which a single seed cannot express |

### 3.2 How a trained-vs-random comparison avoids observed-magnitude thresholds

**Threshold-free forms exist**: complete separation (trained statistic outside the
entire range of the random seeds), or containment (trained inside that range).
Both are ordering comparisons and require no numeric cut.

**The ARCH-001 hazard must be avoided.** ARCH-001 used a containment rule whose
reference interval turned out to span 97.4 % of the statistic's attainable range,
making `INSIDE` near-unfalsifiable. **Mitigation, derived not chosen:** use
**separation** rather than containment. If the random distribution is wide,
separation simply fails — and a wide random distribution *is* the finding that the
architecture can produce the signature. Separation degrades gracefully;
containment degrades into vacuity.

### 3.3 Does the random arm resolve "architecturally forced vs learned"?

**PARTIALLY.** It answers: *can untrained weights of this architecture produce
this observable?* — which §1.6 says is the live question.

It does **not** answer: *is the trained behaviour correspondence, or a learned
shortcut that is not correspondence?* A trained network could produce a sharper,
more stable ramp than random weights for reasons unrelated to matching.

**UNKNOWN and not addressable by this arm.**

### 3.4 The arm is a gate, not a control

**INFERRED, and the most consequential recommendation in this audit.**

Because §1.6 shows the primary observable may be architecturally forced, the
random arm determines whether the experiment has any discriminative power at all.
That argues for a **staged, gated structure** of the kind INDEX-001 used for its
algebraic negative control:

```
Stage 1 : random-weight arm ONLY.
          If random weights reproduce the correspondence signature,
          the observable is NON-DIAGNOSTIC -> STOP and report that.
          The trained arms are never inspected.
Stage 2 : only if Stage 1 clears, measure the trained arms.
```

This makes "the observable is architecturally forced" a **preregistered terminal
outcome** instead of a post-hoc excuse, and prevents the trained result from being
seen before its interpretability is established.

---

## 4. CORRECT HANDLING OF `t = 0`

### 4.1 What is false

`disparity_initial = 5.5` at `t = 0` for `shift="none"`. **Refuted** in
`postmortem_20260911/POSTMORTEM.md` Task B: the first `Conv3d` on a zero input
yields the bias, LeakyReLU makes it a **non-zero constant**, and from layer 2 the
`padding=1` zeros differ from that constant, reintroducing D-axis boundary
structure. The profile is non-constant and generally asymmetric, so the
soft-argmin is not `5.5`.

### 4.2 Legitimate uses of `t = 0`, derived from the network equations

| use | verdict | reason |
|---|---|---|
| **baseline subtraction** in a through-origin fit | **NO** | this is precisely what made a step indistinguishable from a ramp (§2.1). A free-intercept fit over `t > 0` needs no baseline |
| **shape normalisation** | **NO** | for `shift="none"`, `t = 0` is a structurally distinct regime (`V ≡ 0` exactly, so the aggregation receives *no image information*). Normalising by a point from another regime is unsound — this is what made C4 vacuous |
| **wiring validation** | **YES — with a correctly derived prediction** | see §4.3 |
| as a response level in the fit | **NO for `shift="none"`**; admissible for `shift="left"`, where `V = 0` only at `k = 0` and the regime is not degenerate | |

### 4.3 Two wiring checks that ARE derivable

**W1 — no model required.** At `t = 0` the left and right crops are the *same
slice of the same source image*, so they must be **byte-identical**. Checkable
directly on the arrays. A failure means the crop or the axis argument is wrong.

**W2 — derived from the network equations.** For `shift="none"` at `t = 0`,
`Lf = Rf` exactly, so `V ≡ 0`. The aggregation then receives **no image
information whatsoever**, and its output is a function of weights and voxel
position only. Therefore:

```
disparity_initial at t=0 for shift="none" must be IDENTICAL FOR EVERY SCENE
```

**This is falsifiable, derivable, and does not require predicting the value.**
A scene-dependent result would indicate that image information is leaking in —
a genuine implementation fault. The *value* itself is **UNKNOWN** a priori and
must not be predicted.

**The general lesson:** a hard stop must be written against a property that is
*derivable*, not against a *numeric value* that merely seems derivable. GEOM-002's
stop encoded a false premise and therefore fired on correct behaviour.

---

## 5. CONTROL HIERARCHY

### 5.1 Sufficiency of the four arms

| arm | ablates | status |
|---|---|---|
| horizontal synthetic | — (the test) | necessary |
| vertical synthetic | correspondence in the **input**, magnitude matched | **necessary and sound** |
| search-free (`shift="none"`) | candidate structure in the **volume** | **necessary but confounded** — §5.2 |
| random-weight aggregation | learning in the **aggregation** | **necessary, and now the gate** — §3.4 |

**DERIVED: the four arms span the three points where the correspondence signal
could be destroyed — input, volume, weights — and are jointly sufficient for the
question as posed.**

### 5.2 One confound identified in an existing arm

The search-free arm uses a **different checkpoint** (`shift="none"`, separately
trained). It therefore conflates *"no candidate structure in the volume"* with
*"different weights"*. A strictly cleaner ablation would hold the weights fixed
and force the volume `k`-constant — but that requires modifying the cost-volume
code path, which the frozen protocol forbids.

**Recorded as a limitation of the existing arm. No replacement is proposed.**

### 5.3 Missing controls

**None is mathematically necessary for the stated question.** No additional arm is
invented here.

**One irreducible gap (UNKNOWN):** nothing in this hierarchy separates *learned
correspondence* from *a learned non-correspondence shortcut that happens to
produce the same ramp*. That is not addressable by any ablation of this
architecture on synthetic input.

---

## 6. CLAIM CEILING BY OUTCOME

Assuming a gated design of the §3.4 form.

| outcome | maximum claim | explicitly NOT claimable |
|---|---|---|
| **Stage 1: random weights reproduce the signature** | "The candidate-axis ramp under synthetic horizontal displacement is **architecturally forced** and carries no information about training." **This closes the observable.** | anything about correspondence, learned or otherwise |
| **Stage 2 passes: trained separates from random** | "Under synthetic fronto-parallel displacement, the trained aggregation's candidate-axis response is **shape-distinguishable from the same architecture with untrained weights**, and is specific to the horizontal axis." → **`SYNTHETIC-GEOMETRIC-RESPONSE-IS-WEIGHT-DEPENDENT`** | correspondence; learned correspondence; genuine disparity search; real-scene correspondence; disparity correctness; generalisation |
| **Stage 2 fails: no separation** | "Not demonstrated under this design." | that correspondence is absent — the input is OOD |
| **any outcome** | — | **real-scene correspondence**, which no synthetic construction can establish; **genuine disparity search**, which requires a real cost landscape, not an exact zero; **disparity correctness**, since magnitude is never a pass criterion |

**Terminology, fixed:**

- **synthetic geometric response** — the response tracks an imposed displacement
  on synthetic input. *Attainable.*
- **architectural correspondence** — the response is produced by the operator
  irrespective of weights. *The Stage-1 outcome.*
- **learned correspondence** — weight-dependent and attributable to matching.
  *Only partially attainable: the random arm shows weight-dependence, never that
  the mechanism is matching.*
- **correspondence** (unqualified) — **not attainable** by this construction.
- **genuine disparity search** — **not attainable**.
- **real-scene correspondence** — **not attainable**.

---

## 7. DECISION

```
DESIGN-PROBLEM-SOLVABLE
```

Solvable **in the specific sense that matters**: a design exists whose every
possible outcome is informative, including the outcome "the observable is
architecturally forced". It is **not** solvable in the sense of guaranteeing an
answer to Level D — §1.6 shows that may be foreclosed, and the gated structure
makes that a preregistered finding rather than a disappointment.

### Mathematical requirements a future preregistration must satisfy

1. **No through-origin fit.** The primary shape statistic must have a **free
   intercept**. A step yields `α = 0` exactly; the through-origin constraint
   yields `0.2308 h` and is the proven defect (§2.1).
2. **`t = 0` may not serve as a baseline** for any fit or normalisation, for any
   arm. For `shift="none"` it is a degenerate regime (`V ≡ 0`).
3. **Threshold-free gating conditions only.** Conditions that gate the verdict
   must be sign, ordering, strict monotonicity, or complete separation. Any
   magnitude criterion (`α ≈ 1`, `1 − R²`) is **descriptive only** unless
   calibrated against an empirical control distribution.
4. **Shape conditions must be applied symmetrically to every arm.** GEOM-002's C2
   was shape-sensitive and valid but was evaluated on the horizontal arm alone.
5. **Non-injectivity must be proven absent for each statistic.** For every
   proposed statistic, the preregistration must exhibit the alternative shapes
   mapping to the same value and show the rule separates them. *This is the
   transferable lesson from C3 and is the single strongest anti-fooling
   requirement.*
6. **The random-weight arm is mandatory and must gate.** Aggregation only;
   feature extractor frozen; readout untouched; bit-identical cost volumes;
   multiple seeds; full curves. Stage 1 runs first and can terminate the
   experiment before any trained arm is inspected (§3.4).
7. **Use separation, not containment**, against the random distribution — ARCH-001
   showed containment degrades into vacuity when the reference is wide (§3.2).
8. **Hard stops must encode derivable properties, not predicted values.** W1 and
   W2 of §4.3 qualify; "equals 5.5" did not. Enforcement must be asserted in code
   and verified — GEOM-002's was not.
9. **Rail clearance and clipping detectability.** The predicted response must stay
   clear of `k = 0` and `k = 11`, and the protocol must be able to detect
   saturation, since clipping destroys the equivariance the design depends on.
10. **Contamination disclosure.** The designer has read six response curves. The
    preregistration must disclose this and state the achievable status in advance
    as **consistency evidence at best** — and should preferably be specified by a
    party that has not read them.

### What makes experiment 7 harder to fool than 1–6

Requirement **5** is the core of it. Experiments 1–6 each verified that their
statistic gave the *right* answer under the hypothesis; none verified that it gave
a *different* answer under the alternative. C3 passed the first test and failed
the second, and that failure was provable by algebra before execution.

Requirement **6** is the second half: §1.6 shows the primary observable may be
architecturally determined, so the experiment must be able to discover that
**about itself, first, and terminate** — rather than measure the trained model and
then argue about what the number meant.

**No design is produced. No `t` levels, scenes, masks, thresholds, statistics or
execution details are selected. Nothing is launched.**
