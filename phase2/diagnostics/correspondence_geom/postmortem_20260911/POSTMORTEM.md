# POSTMORTEM — EXP-CORRESPONDENCE-GEOM-002

Record `phase2/diagnostics/correspondence_geom/postmortem_20260911/`. 2026-09-11.

**Audit only. Nothing executed.** No training, no inference, no model
instantiated. GEOM-001 untouched. **The GEOM-002 execution record is untouched**
— `RESULTS.md`, `results.json`, `run.log`, `PREREGISTRATION.md` and both frozen
specs are unmodified. No GEOM-003 created or launched.

Tags: **MEASURED** (from source or an existing record), **DERIVED** (algebra),
**INFERRED**, **UNKNOWN**.

---

## TASK A — FORMAL CLOSURE OF GEOM-002

### A.1 The four statuses, kept separate

| category | status | detail |
|---|---|---|
| **protocol-admissible result** | **NONE** | the frozen protocol declared a HARD STOP that fired and was not honoured (§A.2). Under the protocol, execution should have halted before the positive checkpoints. No preregistered determination exists. |
| **observed execution data** | **VALID AS MEASUREMENT** | 208 forward passes, determinism held, all other sanity checks passed, crops/mask/weights invariant, no training. The numbers are real and reproducible. |
| **procedurally compromised execution** | **YES** | the run continued past a declared hard stop. |
| **scientific evidence** | **NONE for or against correspondence** | the observations may be used **only** to diagnose design flaws. They may not be promoted to preregistered evidence. |

### A.2 Why the run is compromised

`PREREGISTRATION.md` §11 lists, in the controls table governed by "**HARD STOP**
on any failure above":

> `t = 0` wiring check · `NEG` returns `5.5` exactly

**MEASURED:** `NEG_shift_none` at `t = 0` returned **7.603198** on all four
scenes. The condition fired. **The harness recorded it and did not halt.** That
is a deviation from the frozen protocol, committed by me in the harness, not a
property of the data.

### A.3 A second, independent reason no admissible result exists

Even had the hard stop not fired, **two of the four decision conditions were
incapable of doing their job**, for reasons provable from the frozen definitions
alone and without reference to any observed value:

- **C3 is misspecified** — `Slope` is a linear functional and is not injective on
  response shape, so `|Slope_h| > |Slope_v|` cannot distinguish a ramp from a
  step (`STATISTIC_AUDIT.md` Task C).
- **C4 is vacuous** — its comparand is an artefact of a structural discontinuity
  in the baseline at `t = 0`, and under C1 it cannot fail
  (`STATISTIC_AUDIT.md` Task D).

A decision rule in which half the conditions cannot discriminate is not a valid
test regardless of how it executes.

### A.4 Formal status

```
GEOM-002 FORMAL STATUS

    PROCEDURALLY-COMPROMISED            (declared hard stop breached, §A.2)
  + STATISTIC-MISSPECIFIED              (C3 non-injective, C4 vacuous, §A.3)
  ==================================================================
  => NO PROTOCOL-ADMISSIBLE RESULT

  Observed data:  preserved in full, valid as measurement,
                  usable ONLY for design diagnosis.
```

### A.5 What this does and does not mean

The recorded verdict in `RESULTS.md` —
`SYNTHETIC-CORRESPONDENCE-NOT-DEMONSTRATED`, 3/12 — was computed by the frozen
rule on the observed data, but **outside the protocol**, because the protocol
required a halt first. It is therefore **withdrawn as a formal preregistered
outcome** and reclassified per §A.4.

**This withdrawal is not a rescue.** GEOM-002 *failed*. Withdrawing a failure
converts it into "no determination", not into a success. The correct reading is:

> GEOM-002 produced **no preregistered determination either way** on synthetic
> correspondence. It is not evidence that correspondence is present, and it is
> not evidence that correspondence is absent.

`RESULTS.md` is **not modified** — it stands as the execution record. This
document is the closure note that sits beside it.

### A.6 Observed numbers, preserved

Preserved in full in `20260911T115917Z/results.json` and `RESULTS.md`: all
`m_h(t)` and `m_v(t)` for 16 cells × 7 levels × 2 axes, all `Slope` values, all
C1–C4 booleans, the 3/12 count, the wiring-check values. **Nothing is deleted.
Nothing is re-scored.** Under the anti-post-hoc rule they may be cited only to
show that a design choice was inadequate.

---

## TASK B — WHY `V[k] = 0 ∀k` DOES NOT IMPLY `disparity_initial = 5.5`

### B.1 The claim that failed

`PREREGISTRATION.md` §5 and `DESIGN_AUDIT.md` §8 asserted:

> at `t = 0` … `V[k] = 0` **exactly**; `Agg(0)` is constant in `k` with *no*
> boundary effect (**the padding zeros equal the signal zeros**); standardisation
> returns exactly 0; the softmax is uniform; therefore
> `disparity_initial = (0+1+…+11)/12 = 5.5` **EXACTLY**.

**The parenthetical is true of the first convolution only.**

### B.2 The architecture (MEASURED, `src/models/stereonet/aggregation.py`)

```
4 × [ Conv3d(32→32, kernel 3×3×3, padding=1) + LeakyReLU(0.01) ]
then  Conv3d(32→1, kernel 3×3×3, padding=1)
```

Readout (MEASURED, `regression.py` + `phase2/models/scaled_regression.py`):
bilinear upsample of the cost to full resolution → `standardise_across_disparity`
(per-pixel, across `d`, population sd, `eps = 1e-5`) → `softmax(−z)` → soft-argmin
over the index grid `0…11`.

### B.3 Stage-by-stage trace (DERIVED — algebra only, nothing executed)

Let `x₀ = 0` (the all-zero cost volume).

**Layer 1.** A convolution of the zero tensor is the bias:
`x₁[c,d,h,w] = b₁[c]`, **constant over `(d,h,w)`**. Here the padded zeros *do*
equal the signal zeros, so no boundary effect arises. **This step is correct as
preregistered.**

**LeakyReLU.** `y₁[c] = φ(b₁[c])`, still constant over `(d,h,w)`, and **in general
non-zero**.

**Layer 2 — where the argument breaks.** `Conv3d` now convolves a tensor whose
value is the **non-zero constant** `φ(b₁)`, but its `padding=1` still inserts
**zeros**. Therefore:

- at an interior voxel, all 27 taps see `φ(b₁)`:
  `x₂[o] = (Σ_{i,j,k} W₂[o,:,i,j,k])·φ(b₁) + b₂[o]`;
- at a voxel on the `d = 0` face, the taps at relative offset `−1` along `d` see
  **0** instead of `φ(b₁)`, so those terms drop:
  `x₂[o] = (Σ_{i,j,k≠−1} W₂[o,:,i,j,k])·φ(b₁) + b₂[o]`.

**`x₂(d=0) ≠ x₂(interior)` whenever the `d = −1` kernel slice is non-zero.**
Likewise at `d = 11` with the `+1` slice. And since `W₂[…,−1,…] ≠ W₂[…,+1,…]` in
general, **`x₂(0) ≠ x₂(11)` — the profile is asymmetric.**

**Layers 3–5.** Each further `3×3×3` convolution propagates the boundary shell one
step inward. Four layers follow layer 2, so candidates `d ∈ {0,1,2,3}` and
`{8,9,10,11}` are contaminated and `d ∈ {4,5,6,7}` retain the interior constant.

**Standardisation.** The per-pixel cost profile `c(d)` is now **non-constant**, so
`std_d > 0` and `(c − mean_d)/(std_d + ε)` is **not** the zero vector. (Contrast
`scaled_regression.demo`, which verifies that standardising a *constant* column
returns exactly 0 — true, but the column is not constant here.)

**Softmax and soft-argmin.** `softmax(−z)` is therefore non-uniform, and
`Σ_d p(d)·d = 5.5` only if `p` is symmetric about `5.5`. It is not, because the
`d = 0` and `d = 11` boundary corrections differ (asymmetric kernel slices).

**Conclusion (DERIVED).** `V ≡ 0` implies a *weights-determined, position-
dependent* cost profile — **not** a flat one. `disparity_initial ≠ 5.5`.

### B.4 The exact mechanism of the observed value

**MEASURED:** `7.603198`, with per-pixel range `1.888440 … 9.183665`,
**bit-identical on all four scenes** (spread `0.00e+00`).

**DERIVED:** identity across four different images is the signature of a map that
depends on **weights and voxel position only** — exactly what §B.3 predicts, since
with `V ≡ 0` no image information enters the aggregation at all. The spatial
spread `1.89 … 9.18` arises because the `(h,w)` padding boundaries produce their
own profile, and the bilinear upsample to full resolution mixes neighbouring
feature columns.

**INFERRED:** the median exceeding `5.5` means the standardised profile assigns
more softmax mass to high candidates — i.e. the aggregated cost is lower at large
`d`. This is a property of these particular trained weights.

**UNKNOWN:** the exact decomposition of `7.603198` into per-layer contributions.
Establishing it would require running the aggregation on a zero tensor, which is
**not done here** (no inference).

### B.5 Was this an implementation fault? No.

**INFERRED, with high confidence.** The implementation is correct; the
**preregistered prediction was wrong**. The hard stop was written against the
wrong trigger — "a deviation means a bug" — when in fact a deviation meant the
derivation was mistaken. A hard stop whose trigger encodes a false premise will
fire on correct behaviour, which is what happened.

### B.6 Propagation and containment (MEASURED)

The erroneous claim appears in exactly two places:
`correspondence/next_direction_audit_20260911/ADDENDUM.md` §A.5 (as a proposed
secondary) and GEOM-002's own `PREREGISTRATION.md` §5 / `DESIGN_AUDIT.md` §8.

It did **not** enter any executed decision rule before GEOM-002: ARCH-001's
negative control was a `k`-constant *non-zero* volume (permutation-invariant by
algebra, so the zero case never arose), and GEOM-001 never ran an `(L, L)` pair.
**The error was latent in three documents and could only surface where `V` is
exactly zero — which GEOM-002 was the first to construct.**

**No historical record is corrected by this audit.** The error is recorded here.

---

## TASK E — FULL DESIGN REVIEW

No observed response curve was used to select any judgement below; each rests on
the frozen definitions or on structural properties of the construction.

| element | status | reason |
|---|---|---|
| synthetic construction | **VALID** | `d = a_R − a_L = t` exactly, uniform; sign verified twice from source; no fill, padding or interpolation; it genuinely dissolved GEOM-001's representability defect |
| `t` levels `{16…96}` | **VALID** for representability | candidates 1…6 on a 0…11 axis; all multiples of the stride; no clipping occurred |
| **the `t = 0` condition** | **AMBIGUOUS — newly identified** | it makes left and right **byte-identical**, so `V ≡ 0` exactly. For `shift="none"` this is a *structurally distinct regime* from every `t > 0`. Using it as the baseline in `Slope` is unsound (Task D) |
| crop geometry | **VALID** | `1136 × 272`, stride-aligned, identical in every condition; all windows inside the source |
| mask | **VALID** | purely geometric, no GT, invariant across `t`, axis, model and scene; the ≈477 px receptive-field contamination was declared in advance |
| horizontal arm | **VALID** as an intervention | the only left/right pathway is the cost volume; `V = 0` exactly at `k = t/16` |
| vertical arm | **VALID as an intervention, INEFFECTIVE as used** | the arm is a properly matched null; it was rendered useless by the statistic that consumed it (C3) |
| search-free arm | **WEAK→INVALID as used** | valid as a `k`-constant baseline, but its `t = 0` point is degenerate, which corrupts the `Slope` computed for it (Task D) |
| `Slope_h`, `Slope_v` | **WEAK** | the anchor of 1 is correct and the `m(0)` subtraction correctly removes constant bias, but the statistic is a **linear functional, not injective on shape** (Task C) |
| **C1** sign | **VALID** | threshold-free, a priori, and it does exclude a monocular-on-left pathway |
| **C2** 6-level monotonicity | **VALID — and the only shape-sensitive condition in the design** | `1/720` under an exchangeable null. **Newly identified asymmetry: C2 was applied to the horizontal arm only.** The design contained a shape discriminator and never applied it to the control |
| **C3** `\|Slope_h\| > \|Slope_v\|` | **INVALID — misspecified** | cannot distinguish ramp from step (Task C) |
| **C4** vs search-free | **INVALID — vacuous** | comparand is a baseline artefact; under C1 it cannot fail (Task D) |
| 12/12 decision rule | **VALID in form, unsound in content** | all-or-nothing with no partial counts is correct discipline, but two of its four inputs cannot discriminate |
| claim ceiling | **VALID** | correctly conservative; the OOD ambiguity of a negative was declared before execution and held |
| **hard-stop specification** | **INVALID** | the `t = 0` trigger encoded a false premise (Task B) |
| **hard-stop enforcement** | **FAILED** | the harness did not halt (§A.2) |

### E.1 The three independent defects

1. **A false premise frozen into a hard stop** (Task B) — a derivation error.
2. **A statistic that cannot see the distinction it was built to test** (Task C) —
   a mathematical error, provable without any data.
3. **A control whose baseline point is structurally special** (Task D) — a design
   error that made C4 vacuous.

**INFERRED:** defects 2 and 3 were both discoverable *before execution* by pure
algebra on the frozen definitions. Neither required running the experiment. The
design audit that preceded GEOM-002 examined the construction thoroughly and did
**not** examine the statistic's discriminative capacity — it verified that the
*anchor* was correct (`Slope = 1` for a perfect ramp) but never asked what *else*
maps to 1.

**That is the transferable lesson: verifying that a statistic gives the right
answer under the hypothesis is not the same as verifying it gives a different
answer under the alternative.**
