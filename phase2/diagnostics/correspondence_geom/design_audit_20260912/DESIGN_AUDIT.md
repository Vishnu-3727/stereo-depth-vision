# DESIGN AUDIT — EXP-CORRESPONDENCE-GEOM-001 (successor line)

Record `phase2/diagnostics/correspondence_geom/design_audit_20260912/`.
2026-09-12.

**DESIGN AND AUDIT ONLY. NOTHING EXECUTED.**
No training. No StereoNet instantiated. No trained checkpoint loaded. No
inference with any trained weights. No `RESULTS.md`. Phase 1 untouched.
`INDEX-001/002/003`, `ARCH-001`, `ARCH-RATE-001`, `TR-001`, `GEOM-001`,
`GEOM-002`, `SHIFT-001` and every historical record are **unmodified** —
verified by re-reading them, never by writing to them.

The one thing that was *run* is `synthetic_mechanism_check.py`: framework-default
**random** `Aggregation` weights on **synthetic** feature fields, with a
structural cost-volume identity check. It loads no checkpoint, reads no image,
touches no dataset and performs no training. It is a mechanism demonstration,
not an experiment, and **no number it produced is used as a threshold anywhere
in the decision rule below.**

Tags: **MEASURED** (from source or an existing record) · **DERIVED** (algebra) ·
**INFERRED** · **UNKNOWN**.

---

## 0. CONTAMINATION DISCLOSURE (stated before anything else)

I have read the response curves of all seven prior correspondence experiments,
including `TR-001 §3` and `GEOM-002 §4–5`. Any statistic I select is selected by
someone who has seen those shapes. This is **irreducible for this auditor** and
is disclosed, not mitigated away.

**What is clean.** The statistic proposed below is fixed by an *algebraic
identity in the source* — the cost volume's 2×2 interaction is identically zero
because the cost volume is linear in both feature maps (`source_trace.md` §6) —
together with the *sign convention of the readout* (`softmax(−z)` ⇒ a match is a
**minimum**, `source_trace.md` §7). Neither input is an observed curve. The
derivation is reproducible by anyone reading `cost_volume.py` and
`regression.py` without access to any result.

**Status achievable, declared in advance:** **CONSISTENCY EVIDENCE AT BEST.**
Three checkpoints are irreducibly reused; the designer is contaminated. This
cannot be a confirmatory experiment and must not be reported as one.

---

## 1. COMPLETE CURRENT EVIDENCE LADDER

| level | claim | status | established / blocked by |
|---|---|---|---|
| **A** | right-image dependence | **ESTABLISHED** | SHIFT-001 Stage A and every subsequent run; the right image enters the cost volume |
| **B** | matching-path / binocular dependence engaged | **ESTABLISHED** | the only left↔right pathway is the cost volume; `shift="none"` vs `shift="left"` differ |
| **C** | candidate-coordinate sensitivity | **INCONCLUSIVE — bypassed as structurally the wrong axis** | with `shift="none"` the right image is a `k`-constant term, so the candidate axis cannot carry it. An **in-principle** barrier (INDEX-001/002/003, ARCH-001) |
| **C′** | the trained aggregation *realises* an architecturally available candidate-axis ramp | **MEASURED** | TR-001: `CLEAR-SEPARATION`, 12/12 matched and 36/36 cross units outside the observed 384-unit random population; `α_v` not separated; search-free arm flat |
| **D** | **geometric correspondence** — the response tracks actual image geometry | **NOT-DEMONSTRATED** | **this is the open question** |

### 1.1 Why C′ does not reach D (MEASURED + DERIVED)

`predesign_audit_20260911` §1.6 derives, from shift-equivariance alone and for
**any** weights:

```
V(k, w) = F(k − τ, w + τ)        ⇒        m(τ) ≈ τ + const
```

A unit-slope ramp is **architecturally forced**. TR-001 measured trained slopes
`0.95 … 1.22` — *exactly that prediction*. TR-001's own §7 states the honest
reading: the trained aggregation realises the architecturally available ramp
where 378/384 sampled random aggregations do not, and that is a statement about
**whether the equivariance survives the D=12 padding boundary**, not about
matching.

### 1.2 Prior determinations at Level D — none admissible

| experiment | outcome | why it settled nothing at D |
|---|---|---|
| SHIFT-001 | **HALTED** at Stage A | analytic anchor (`5.5`) falsified; a provably search-free model still responded to translation by up to 0.70 candidates |
| INDEX-001 ×2 | exploratory only | uncalibrated single-draw control |
| INDEX-002 | not demonstrated | structurally mismatched null |
| ARCH-001 | CASE A | reference interval spanned 97.4 % of the statistic's range — ~no power |
| GEOM-001 | not demonstrated | representability defect: 99.8 % of the mask could not exhibit the hypothesis |
| GEOM-002 | **no admissible result** | declared hard stop breached; C3 non-injective; C4 vacuous |
| ARCH-RATE-001 | rate measured | descriptive; 0/384 random units QUALIFYING, 378 FLAT |
| TR-001 | CLEAR-SEPARATION | a **C′** result; §11 states Level D remains NOT-DEMONSTRATED |

**Eight experiments, zero admissible determinations at Level D.** Only the
candidate-axis re-indexing class was refuted *in principle*; the rest failed on
execution, statistics, or non-identifiability.

---

## 2. EXACT SOURCE-DERIVED GEOMETRY MAPPING

Full derivation in `source_trace.md`. The chain, with the sign convention:

```
 t px imposed displacement (16 | t)
   │  extractor: fully convolutional, four stride-2 convs, scale 16, exact interior
   ▼
 τ = t/16 feature cells
   │  matched self-pair:  Rf[w] = Lf[w + τ]
   │  V[k][w] = shift_left(Lf,k)[w] − Rf[w] = Lf[w+k] − Lf[w+τ] = F[k−τ][w+τ]
   ▼
 V = 0  EXACTLY at  k = τ           (interior; F is τ-independent)
   │  aggregation → bilinear upsample → z = (c − mean_k c)/(sd_k c + 1e-5) → softmax(−z)
   ▼
 a match must be a MINIMUM of z(k)          ← the negation in softmax(−z)
   │
   ▼
 1 candidate = 16 full-resolution px ;  imposed disparity d = x_L − x_R = +t px
```

**Sign, verified independently.** `right[y,x] = left[y, x+t]`, so a feature at
left column `x+t` sits at right column `x` ⇒ `d = +t` px, `k_true = τ`. This
matches the frozen `run_tr.py::verify_sign_from_data` assertion
(`right[:, :-t] == left[:, t:]`, 84 array checks, PASS).

**Disparity representation (§VIII of the brief).** `disparity_initial` is **not**
a physical pixel disparity: `soft_argmin` emits candidate units `0…11` and the
scale is supplied by the refinement stage, which is never run. The scalar is
additionally compressed toward the grid centre by the softmax-weighted mean.
**This design therefore states its primary statistic in native candidate units
as an integer `argmin_k`**, which is invariant to the per-pixel affine
standardisation and immune to softmax compression. The exact conversion
`1 candidate = 16 px` is recorded and **used by no step of the decision rule**.

**Marked UNKNOWN and not built upon:** the value of `disparity_initial` for any
input; whether the trained aggregation maps small `|V|` to low cost (that is the
hypothesis, not a premise).

---

## 3. WHY THE ORIGINAL TRANSLATION SWEEP WAS NON-DISCRIMINATIVE

Three independent reasons, in increasing order of severity.

**3.1 Empirical (SHIFT-001 Stage A, MEASURED).** A model with a **provably
degenerate** cost volume — `max|slice_k − slice_0| = 0.0`, no disparity search
whatsoever — nevertheless produced a structured, input-dependent candidate curve
and responded to right-image translation by up to **0.70 candidates**. So
"response ≠ 0" is confounded by the aggregation's D-axis padding boundary.

**3.2 Algebraic (`predesign_audit_20260911` §1.6, DERIVED).** The aggregation is
shift-equivariant in `(k, y, w)` up to padding **for any weights**. With
`V(k,w) = F(k−τ, w+τ)`, a translation of `τ` along `k` passes straight through
to the soft-argmin. Ideal correspondence (**A**) and random weights (**E**)
predict the **same** leading-order observable `m(τ) ≈ τ + const`.
**The slope cannot discriminate them, whatever its value.**

**3.3 The reason that closes the class (DERIVED here, new).** Put
`V_AB[k][w] = Af[w+k] − Bf[w+τ]` for a **non-corresponding** pair and substitute
`k = κ + τ`:

```
V_AB[κ+τ][w] = Af[w+τ+κ] − Bf[w+τ] = G[κ][w+τ],      G τ-independent
```

**A cross-paired volume translates with `τ` exactly as a matched one does.**
Therefore the slope cannot separate *corresponding* from *non-corresponding*
either — not merely trained from random. **The whole translation-slope class is
dead as a correspondence discriminator**, and adding offsets, scenes, seeds,
statistics or mask area cannot revive it.

**MEASURED corroboration (`synthetic_mechanism_check.py` S1, random weights,
synthetic fields — reported as mechanism, used as no threshold):** matched
free-intercept slope median `+0.172`, crossed median `+0.003`. Both arms are
translating objects; the observed gap is the padding boundary's differential
survival, not correspondence. This is precisely the C′ phenomenon, reproduced
with **no trained weights at all**.

---

## 4. ALL CANDIDATE GEOMETRIC INTERVENTIONS CONSIDERED

Nine were evaluated. Each is judged against one question: **does it change
correspondence while holding the translation structure of the cost volume
fixed?** If not, §3.3 kills it.

| # | intervention | changes correspondence? | changes the `k`-translation structure? | verdict |
|---|---|---|---|---|
| 1 | geometry-consistent horizontal translation (the GEOM-002 sweep) | no — it *sets* it | yes, by `τ` | **REJECT** — §3.3; the artifact reproduces it |
| 2 | geometry-inconsistent translation (translate right by `τ`, claim `τ′`) | no; only the label changes | yes, by the actual `τ` | **REJECT** — the network never sees the label. Vacuous |
| 3 | phase-preserving transform of the right image (random-phase / phase-scramble) | yes | **yes and uncontrollably** — destroys `F`'s structure | **REJECT** — this is "destroy the right image" in disguise; §V of the brief forbids it, and a generic-binocular explanation survives |
| 4 | spatially shuffled correspondence (per-row shift jitter `τ(y)`) | no — it makes correspondence *non-uniform*, still true | yes, **row-locally** | **REJECT** — `V(k,y,w) = F_y(k−τ(y), w+τ(y))`; the equivariance is row-local, so the artifact tracks `τ(y)` for free. Superficially attractive, fully confounded |
| 5 | block permutation of the right image | yes | yes, block-locally, **plus** hard block seams | **REJECT** — seams are a large uncontrolled main effect of the right image |
| 6 | controlled stereo-pair synthesis (rendered scene, known disparity) | yes | yes | **REJECT for now** — severely off-manifold relative to KITTI; a negative would test OOD behaviour, not correspondence (§VI of the brief). Also needs a renderer the repo does not have |
| 7 | left/right patch reassignment within a pair | yes | yes, patch-locally, plus seams | **REJECT** — same seam objection as 5, and it breaks the exact `V = 0` anchor |
| 8 | **cross-scene pairing at fixed `τ` (the 2×2)** | **yes — and only this** | **NO — identical `k`-translation structure (§3.3)** | **ACCEPT** — §7 |
| 9 | **vertical-axis 2×2** (same images, same displacement magnitude, wrong axis) | yes, by removing the epipolar relation | there is **no `κ`** making `V` vanish (`predesign_audit` §1.5) | **ACCEPT as the secondary negative** — §7.3 |

**The decisive observation.** Interventions 1–7 all modify the right image
geometrically, which necessarily modifies the cost volume's translation
structure, which the shift-equivariant aggregation converts into candidate-axis
response *without any matching*. Only **8** changes *which two images are paired*
while leaving the translation structure exactly as it was. That is the unique
handle.

---

## 5. CAUSAL PREDICTION TABLE

### 5.1 The intervention

At each `τ ∈ {1,…,6}`, from two source frames `A`, `B`, four cells:

| cell | left | right | corresponds? | imposed `d` |
|---|---|---|---|---|
| **AA** | `A[:, 0:1136]` | `A[:, 16τ : 1136+16τ]` | **YES**, exactly, uniformly | `+16τ` px |
| **BB** | `B[:, 0:1136]` | `B[:, 16τ : 1136+16τ]` | **YES**, exactly, uniformly | `+16τ` px |
| **AB** | `A[:, 0:1136]` | `B[:, 16τ : 1136+16τ]` | **NO**, at any `k` | undefined |
| **BA** | `B[:, 0:1136]` | `A[:, 16τ : 1136+16τ]` | **NO**, at any `k` | undefined |

```
POSITIVE  =  { AA , BB }       geometry obeyed
NEGATIVE  =  { AB , BA }       geometry broken, everything else held
```

**Interaction profile** (per pixel, then median over the fixed mask):

```
I_τ(k)  =  ½[ z_AA(k) + z_BB(k) ]  −  ½[ z_AB(k) + z_BA(k) ]
```

`z` is the standardised cost the softmax consumes (`source_trace.md` §7).

### 5.2 What changes and what is held fixed

| | changes across the contrast | held **exactly** fixed |
|---|---|---|
| POSITIVE vs NEGATIVE | **only which left image is paired with which right image** | `τ`; both crop windows; the crop code path; the mask; the model; the candidate grid; the set of images used (each of `A`, `B` appears once as a left and once as a right); every marginal image statistic; the shift-fill geometry; the padding boundary |

This is not approximate statistical matching — it is **exact additive
cancellation**. Every main effect (left identity, right identity, `τ`, crop
geometry, receptive-field edge contamination, architecture, padding) enters all
four cells and vanishes.

### 5.3 The prediction table (§VII of the brief)

Predictions are **precomputed from the source algebra**, not from any observed
curve.

| mechanism | is `I_τ` non-zero? | where is its extremum? | does the extremum track `τ`? | per-unit hit count `h` (of 6) |
|---|---|---|---|---|
| **H1 — genuine geometric correspondence** | **yes** | a **minimum at `k = τ`** — `V = 0` there in the POSITIVE cells only, and `softmax(−z)` makes a match a minimum | **yes, exactly, unit slope, zero offset** | **6** |
| **H2 — generic binocular / right-image dependence** | **≈ 0** — any effect of the right image alone is a main effect and cancels identically | undefined / noise-located | **no** | chance-like |
| **H3 — candidate-axis architectural artifact** | **exactly 0 for any affine aggregation** (§6); non-zero only through the LeakyReLU | weight-determined, `τ`-independent up to the forced translation | **no reliable tracking** | chance-like |
| **H4 — no useful correspondence** | arbitrary | arbitrary | no | chance-like |

**H1 is separated from H2, H3 and H4 by both the location and the tracking of
the extremum.** H2 and H3 are additionally separated from H1 by amplitude, which
is reported but is **not** part of the decision rule (no a-priori amplitude
threshold exists — `predesign_audit` §2.3 item 5).

### 5.4 Why the same table is not satisfiable by the artifact

Under H3 the response is the linear translation-equivariance plus the padding
boundary. Both are properties of the **linear** operator. §6 shows they cancel
**exactly** in the interaction. The artifact's only surviving channel is the
nonlinearity — which has no reason to place a minimum at `k = τ` in the
POSITIVE cells specifically, because it cannot see which pair corresponds except
through `V` itself.

---

## 6. ARCHITECTURAL-ARTIFACT ANALYSIS (§XI of the brief)

### 6.1 The analytic result

```
V_AA + V_BB − V_AB − V_BA
  = (Af[w+k] − Af[w+τ]) + (Bf[w+k] − Bf[w+τ])
  − (Af[w+k] − Bf[w+τ]) − (Bf[w+k] − Af[w+τ])
  = 0                                     IDENTICALLY, ∀ k, w, τ
```

because `build_cost_volume` is **linear in both feature maps**
(`levels[k] = shift_left(left,k) − right`).

**Therefore:**

1. a **candidate-constant** volume gives an identically-zero interaction (it is
   the `shift="none"` special case of the same identity);
2. an **affine** aggregation gives an identically-zero aggregated-cost
   interaction for **any** weights — *including the entire D-axis padding
   boundary*, which is part of the linear operator;
3. the forced translation `V(k,w) = F(k−τ, w+τ)` applies to matched **and**
   crossed cells alike (§3.3) and so contributes to the interaction only through
   the nonlinearity.

**The intervention is therefore discriminative: the known architectural artifact
cannot produce the primary observable, because the artifact lives in the linear
part of the operator and the linear part cancels exactly.**

### 6.2 The tiny synthetic tensor test (run; no trained model)

`synthetic_mechanism_check.py` — 8 framework-default random `Aggregation`s,
synthetic smoothed feature fields, geometry identical in shape to the frozen
construction, mask as derived in `source_trace.md` §10.

| id | question | result |
|---|---|---|
| **S0** | is the matched volume exactly zero at `k = τ`? | `max|V| = 0.0` — **exact** |
| **S0** | is the cost-volume 2×2 interaction zero? | max **relative** residual `1.566e-07` — float32 round-off |
| **S1** | does the slope separate matched from crossed under random weights? | matched median `+0.172`, crossed `+0.003` — **the slope class is confounded**, §3.3 |
| **S2** | does the **matched arm alone** locate `τ` under random weights? | `6 / 96` hits (chance is `1/12 ≈ 8/96`) — **no**; the interaction is *necessary*, not decorative |
| **S3** | does the **interaction** locate `τ` under random **nonlinear** weights? | `5 / 48`; per-unit `h ∈ {0,0,0,0,0,1,1,3}`; anchors scattered over `{−2,−1,0,1,2,4}` |
| **S3** | …under an **affine** aggregation (LeakyReLU slope 1.0)? | `2 / 48`; `h ≤ 2`; anchors scattered over 14 distinct values |
| **S3** | interaction amplitude, nonlinear vs affine | median `1.124` vs `0.0789` — **~14×**, exactly the §6.1 prediction |

**Reading (INFERRED).** The architecture does **not** reliably place the
interaction minimum at `k = τ`: no random unit reached `h = 6`, and the best
reached 3. The residual affine-arm amplitude is the per-condition
standardisation, the second (small) nonlinear channel identified in
`source_trace.md` §7.

**These numbers set no threshold.** Eight seeds on synthetic fields cannot
calibrate anything. Their only job was to answer *yes/no*: **can the artifact
produce the signature?** The answer is *not reliably* — which is exactly the
condition under which the real, pre-registered random-weight **gate** (§10.1) is
worth running.

---

## 7. BEST POSITIVE / NEGATIVE GEOMETRIC PAIR

### 7.1 POSITIVE — geometry-obeying

`AA` and `BB`: a matched synthetic fronto-parallel self-pair at displacement
`16τ` px. True disparity `= 16τ` px **exactly and uniformly**, with no fill, no
padding and no interpolation — integer slicing of one source frame. `V = 0`
exactly at `k = τ`. This construction is inherited unchanged from
`ARCH-RATE-001/construction_spec.json`; `SUCCESSOR_ASSESSMENT.md` §5 records that
it is **sound** and that it genuinely dissolved GEOM-001's representability
defect. It is the only part of GEOM-002 that survived its post-mortem.

### 7.2 NEGATIVE-1 — geometry-breaking, appearance-controlled (the primary)

`AB` and `BA`: **the identical crops**, the identical displacement, the identical
code path — paired with the *other* scene. Correspondence is destroyed; nothing
else is touched.

**Why this satisfies §V of the brief better than any listed alternative.** The
brief asks that the negative "preserve relevant low-level/statistical properties
sufficiently well that generic binocular dependence cannot trivially explain the
difference". The 2×2 does better than *sufficiently well*: because every image
appears **once as a left and once as a right**, every first-order property is
**exactly cancelled by construction**, not approximately matched. Generic
binocular dependence is not merely weakened — it is *algebraically removed*
(§6.1).

The right image is **not destroyed**: `AB`'s right image is a real KITTI frame,
identically preprocessed, at the identical displacement, and it appears as the
*positive* right image in cell `BB`.

### 7.3 NEGATIVE-2 — geometry-breaking, content-identical (secondary)

The **vertical 2×2**: right crop `src[16τ : 272+16τ, 0:1136]`. The *same two
images*, the *same displacement magnitude*, the *same code path* — but displaced
off the epipolar axis. The cost volume compares within a row, and
`predesign_audit` §1.5 derives that **no `κ` exists** making `V` vanish. Under
H1 the primary statistic must therefore show **no tracking**. This is the
strongest possible content-preservation (content identity, not content
matching), and it is a pure null on the primary statistic.

Declared caveat, inherited: roads, lane markings and horizon lines are
vertically self-similar, so a partial spurious match at some `τ` is possible.
The vertical arm is a **magnitude-matched null** and is never assigned a
predicted slope or a predicted location.

### 7.4 Why real KITTI frames rather than a synthetic distribution (§VI)

**Real frames, synthetic pairing.** Both positive and negative arms draw on the
same 12 real, identically preprocessed KITTI left images, so the contrast is
**internally balanced** with respect to the manifold: whatever OOD penalty the
construction carries, it is carried equally by POSITIVE and NEGATIVE and is
therefore cancelled by the interaction. A rendered synthetic pair would move
*both* arms off-manifold together and would test OOD behaviour without buying
any additional control (§4 item 6).

**The OOD caveat stands and is declared in advance:** a constant-disparity plane
is not natural stereo. **A negative result remains ambiguous.**

---

## 8. PRIMARY STATISTIC

Frozen before execution. Chosen from the algebra of §6.1 and the readout sign
convention of `source_trace.md` §7 — **not** from any observed curve.

```
for each unit (checkpoint c, scene pair p) and each τ ∈ {1,…,6}:

    z_cell(k, y, x)  =  standardised cost the softmax consumes,   cell ∈ {AA,BB,AB,BA}

    I_τ(k, y, x)     =  ½[z_AA + z_BB](k,y,x)  −  ½[z_AB + z_BA](k,y,x)

    Ī_τ(k)           =  median over the fixed mask of I_τ(k, ·, ·)        k = 0…11

    hit_τ            =  [ argmin_k Ī_τ(k)  ==  τ ]                        BOOLEAN

    h(c, p)          =  Σ_{τ=1..6} hit_τ                                  ∈ {0,…,6}
```

### 8.1 Why each choice is forced rather than selected

| choice | forced by |
|---|---|
| **interaction**, not the raw response | §6.1 — it is the unique functional in which the linear artifact cancels exactly |
| **`argmin`**, not `argmax` | `softmax(−z)` — the negation makes a match a **minimum** (`source_trace.md` §7). Read from source |
| **`k = τ`**, not a fitted offset | `V = 0` exactly at `k = τ` — the construction's own anchor |
| **integer `argmin`**, not the soft-argmin scalar | the scalar is softmax-compressed and carries an unknown monotone distortion (§2, §VIII); `argmin` is invariant to the per-pixel affine standardisation |
| **`τ = 1…6`, `τ = 0` excluded** | at `τ = 0` the matched cells are byte-identical ⇒ `V ≡ 0` ⇒ a structurally distinct regime (`POSTMORTEM.md` Task B, `ALGEBRA_AUDIT` §4). Excluded, not baselined |
| **per-pixel interaction, then median** | the cancellation identity is pointwise; taking the median first would cancel medians, not effects |
| **no through-origin fit anywhere** | `ALGEBRA_AUDIT` requirement 1 — the proven GEOM-002 defect |

### 8.2 Non-injectivity proof (the `ALGEBRA_AUDIT` requirement-5 anti-fooling test)

*What else maps to `h = 6`?* The statistic is the **location** of an extremum,
so the question is which mechanisms place `argmin_k Ī_τ` at `τ` for all six `τ`.

| alternative shape / mechanism | value of `h` | separated? |
|---|---|---|
| `Ī_τ` a constant in `k` (no interaction at all) | `argmin` is the first index ⇒ `h ≤ 1` (only `τ=0`, which is excluded ⇒ `h = 0`) | **yes** |
| `Ī_τ` a monotone ramp in `k` | `argmin` is pinned at an endpoint, `τ`-independent ⇒ `h ≤ 1` | **yes** |
| `Ī_τ` a step of any height | `argmin` at one edge of the step, `τ`-independent ⇒ `h ≤ 1` | **yes** |
| `Ī_τ` a dip at a **fixed** `k₀ ≠ τ` | `h ≤ 1` | **yes** |
| `Ī_τ` a dip tracking `τ` with a **constant offset** `c ≠ 0` | `h = 0` | **yes — and it is captured as outcome class D**, §11 |
| `Ī_τ` pure noise | `h ~ Binomial(6, ~1/12)` | **yes**, and the gate calibrates it empirically |
| a **maximum** at `k = τ` (sign-inverted mechanism) | `h = 0` | **yes**; `argmax` is recorded separately |

This is the test C3 failed: `Slope` was verified to give the right answer under
the hypothesis and never checked for what *else* mapped to it. `argmin`-location
is not a linear functional and is **injective on the property it encodes** —
"where is the extremum" — by definition.

### 8.3 Secondary quantities — reported, never converted into a claim

`argmax_k Ī_τ`; the anchor `c_τ = argmin_k Ī_τ − τ`; the amplitude
`max_k Ī_τ − min_k Ī_τ`; the soft-argmin response `m(τ)` and its free-intercept
slope `α` for every cell (for continuity with TR-001 — **compared to nothing**,
since §3.3 shows the slope is not a correspondence discriminator).
**No secondary quantity enters the decision rule.**

---

## 9. NULL / CONTROL

### 9.1 No manufactured p-value

The 18 trained units share 3 checkpoints and 12 scenes; the checkpoints
originate from related training conditions. **Units are not independent
replicates, exchangeability is not defensible, and no p-value is computed
anywhere.** (Same discipline as TR-001.)

A combinatorial null (`P(h = 6) = 12⁻⁶` under uniform `argmin`) is **explicitly
rejected**: `synthetic_mechanism_check.py` S3 shows `argmin` is *not* uniform
under this architecture, so the combinatorial figure would be a fiction. It is
recorded here only to say it is not used.

### 9.2 The null is the empirical random-weight population — and it **gates**

Per `ALGEBRA_AUDIT` requirement 6 and §3.4, the random-weight arm is not a
control but a **gate**: it determines whether the observable is diagnostic at
all, and it runs **first**, before any trained aggregation is loaded.

Randomisation is inherited **verbatim** from
`correspondence_arch_rate/20260911T130507Z/randomisation_spec.json`:

- **aggregation only** (111 585 params), framework-default init
  (`kaiming_uniform_(a=√5)`, bias `U(±1/√fan_in)`), `torch.manual_seed(seed)`
  immediately before construction, seeds `0…31`;
- **feature extractor frozen and trained** — randomising it would destroy the
  `V = 0`-at-`κ = 0` structure and the comparison would no longer hold the cost
  volume fixed;
- **readout parameter-free** (asserted at load), **refinement never invoked**;
- **bit-identical cost volumes** between the random and trained arms.

All three trained extractors are used, so the gate is not tied to one.

### 9.3 Separation, never containment

`ALGEBRA_AUDIT` §3.2: ARCH-001's containment rule degenerated into vacuity
because its reference interval spanned 97.4 % of the statistic's range.
**This design uses complete separation.** If the random distribution is wide,
separation simply fails — and a wide random distribution *is itself the finding*
that the architecture can produce the signature.

### 9.4 The other controls — kept or dismissed, with reasons (§X of the brief)

| control | status | reason |
|---|---|---|
| **1. trained H2 arm** | **REQUIRED** | it is the test arm: 3 checkpoints × 6 pairs |
| **2. `NEG_shift_none` search-free arm** | **REQUIRED, reported separately** | with a `k`-constant volume no `κ` exists, so no tracking mechanism exists. **Confound declared and inherited:** it differs from the trained arm in *two* ways at once — a different checkpoint **and** a different cost-volume construction — so it cannot isolate the construction as the cause (`ALGEBRA_AUDIT` §5.2). It is evidence that the signature is not generic to *any* trained aggregation, nothing more |
| **3. right-image corruption control** | **NOT NEEDED — and the reason is the design's core property** | corrupting the right image changes a **main effect** of the right image, and §6.1 shows every main effect cancels *identically* in the interaction. Such a control would test a channel the statistic has already closed algebraically. Adding it would imply the cancellation is approximate, which it is not |
| **4. geometry-preserving control** | **PRESENT** — it is the POSITIVE arm (`AA`, `BB`), §7.1 |
| **5. geometry-breaking control** | **PRESENT, and doubled** — NEGATIVE-1 cross-pairing (§7.2) and NEGATIVE-2 vertical axis (§7.3) |
| **6. random-weight aggregation** | **REQUIRED, and it GATES** — §9.2 |
| affine-aggregation arm (LeakyReLU slope 1.0) | **synthetic only, already run** | it is a mechanism demonstration (§6.2); running it with trained weights would measure a model that does not exist |

---

## 10. UNIT OF ANALYSIS

### 10.1 Two stages, two populations

```
STAGE 1 — GATE (random aggregations; NO trained aggregation is loaded)
    unit = (feature extractor e, scene pair p, seed s)
         = 3 extractors × 6 pairs × 32 seeds        =  576 units
    each unit yields h ∈ {0,…,6}

STAGE 2 — TRAINED (only if the gate clears)
    unit = (checkpoint c, scene pair p)
         = 3 checkpoints × 6 pairs                  =   18 units    PRIMARY
    search-free arm: 1 checkpoint × 6 pairs         =    6 units    reported separately
    vertical 2×2   : same units, secondary negative
```

### 10.2 Shared weights — accounted for explicitly

**Three checkpoints are the entire population.** Training is forbidden, so
weight reuse is irreducible and caps evidential strength. Reporting is therefore
required **per unit, grouped by checkpoint, and grouped by scene pair**, and the
`ALGEBRA_AUDIT`/TR-001 **heterogeneity rule** applies unchanged: if one
checkpoint separates while others do not ⇒ **MIXED ⇒ CASE C**, never an overall
success. Heterogeneity is never averaged away.

### 10.3 Scene pairs — fresh, and selected by a rule with no freedom

**MEASURED:** `hailo_val` holds 40 scenes. Indices used by prior correspondence
experiments: `{0, 5, 6, 7, 8, 9, 27, 31}` (GEOM-001/SHIFT-001 used `27, 0, 31, 6`;
GEOM-002/ARCH-RATE-001/TR-001 used `5, 7, 8, 9`). **32 are unused.**

**Rule, fixed before any measurement and admitting no choice:** take the twelve
lowest unused indices in ascending order and pair them consecutively.

```
1 (000161)  2 (000162) │ 3 (000163)  4 (000164) │ 10 (000170) 11 (000171)
12 (000172) 13 (000173) │ 14 (000174) 15 (000175) │ 16 (000176) 17 (000177)
```

Six disjoint pairs, twelve never-measured scenes, **no GT-based or
response-based selection anywhere**.

**Declared limitation (inherited, MEASURED from source).** The extractor's
receptive field is ≈ 477 px (`source_trace.md` §3), so a 64-px mask border does
not remove crop-edge contamination. It is a geometric main effect, identical in
all four cells, and is cancelled by the interaction to the extent that §6.1's
cancellation is exact. Declared in advance, exactly as GEOM-002 declared it.

---

## 11. DECISION RULE

Frozen. Threshold-free throughout: sign, ordering and complete separation only.
No magnitude criterion gates anything.

### Stage 1 — the gate (trained aggregations are never loaded)

```
G-STOP :  if  max_over_576_units  h_rand  ==  6
          -> TERMINAL OUTCOME  "STATISTIC-ARCHITECTURALLY-AVAILABLE"
          -> HALT. Stage 2 is NOT run. The trained arms are never inspected.

G-PASS :  otherwise, record  H* := max_over_576_units h_rand   (an ORDERING
          reference for complete separation, not a chosen threshold), and
          proceed to Stage 2.
```

### Stage 2 — the trained arm, 18 units

| outcome | condition | interpretation, fixed in advance |
|---|---|---|
| **A · CORRESPONDENCE-CONSISTENT** | `min h_trained > H*` **and** every one of the 3 checkpoints and every one of the 6 pairs separates (no unit at or below `H*`) | the trained response is sensitive to **which** left and right images are paired, and locates the imposed candidate — §13 |
| **B · NO-SEPARATION** | the trained `h` distribution overlaps the random population | not demonstrated under this design. **Not** evidence that correspondence is absent — the input is OOD (§7.4) |
| **C · MIXED** | partial separation, or one checkpoint / one pair separating while others do not | ambiguous; never an overall success. Report the heterogeneity explicitly |
| **D · TRACKING-WITH-OFFSET** | `h_trained` does not separate, **but** the anchor `c_τ = argmin_k Ī_τ − τ` is **identical across all six `τ`** with `c ≠ 0`, for units that separate from the random anchor-constancy rate | the response tracks the imposed geometry with a **systematic candidate offset**. Informative about equivariance; **NOT** a correspondence pass. Reported as its own class |

**Required alongside every outcome:** the vertical 2×2 (NEGATIVE-2) result and
the search-free arm result, both reported separately and neither mixed into the
gate. Under outcome A, a vertical arm that *also* tracks would downgrade the
result to **C**, since it would indicate a non-epipolar mechanism.

### 11.1 Hard stops — written against derivable properties, and enforced

GEOM-002's stop encoded a *predicted value* (`5.5`) that was false, so it fired
on correct behaviour; and the harness did not halt. Both failures are addressed.

| id | condition — all **derivable**, none a predicted disparity value | derivation |
|---|---|---|
| **HS1** | the left crop is byte-identical across every `τ` and every cell sharing that left source | same slice of the same array |
| **HS2** | the right-crop origin is exactly `(0, 16τ)` horizontally / `(16τ, 0)` vertically, asserted at every level | frozen construction |
| **HS3** | for every POSITIVE cell, `max|V[:, τ, :, w < 54]| == 0.0` **exactly** | `source_trace.md` §4.2; **confirmed `0.0`** by the synthetic check |
| **HS4** | cost-volume 2×2 interaction: `max|residual| / max|V| ≤ 1e-5` | zero in exact arithmetic (§6.1). The bound comes from **float32 round-off**: `eps = 1.19e-7`, accumulated over 32 channels and 4 terms ⇒ `~1e-6`; `1e-5` carries a stated 10× margin. Derived from machine precision, **not** from any observation (the synthetic check's `1.57e-07` corroborates but does not define it) |
| **HS5** | one complete unit recomputed after a **process restart** is **bit-identical** (IEEE-754 byte comparison) | determinism |
| **HS6** | no optimizer imported or constructed; no `.backward()`; no `.step()`; no weight modified; refinement never invoked; readout parameter count `== 0` | source assertions |
| **HS7** | the Stage-1 `G-STOP` condition **halts the process** and Stage-2 code is unreachable in the same run | enforcement |
| **HS8** | mask retained-pixel count `== 115 200`; mask identical in all cells, all `τ`, both axes, every model | §10.3 |
| — | **no numeric expectation is placed on any disparity value anywhere** | the GEOM-002 lesson |

Every hard stop must `raise` and the harness must be shown to have exited —
"recorded and continued" is the GEOM-002 breach and is forbidden.

---

## 12. COMPUTE ESTIMATE

**Inference only. No training. No gradients.**

| stage | passes |
|---|---|
| extractor passes | 3 extractors × 6 pairs × 14 crops (2 left + 12 right) = **252** |
| cost volumes (aggregation-independent, cached) | 3 × 6 × 4 cells × 6 `τ` = **432** horizontal (+432 vertical, trained arm only) |
| **Stage 1 gate** readouts | 3 × 6 × 32 seeds × 4 × 6 = **13 824** |
| **Stage 2** trained readouts | 3 × 6 × 4 × 6 = **432** horizontal + **432** vertical |
| search-free readouts | 1 × 6 × 4 × 6 = **144** + **144** vertical |
| **total** | **≈ 15 000 readout passes** |

**MEASURED reference rates:** ARCH-RATE-001 ran 5 376 arms in **56.2 s**;
TR-001 ran 560 readouts + 208 volumes in **23.7 s**, on an RTX 4060 Laptop.

```
ESTIMATE:  5 – 10 minutes wall clock, single GPU, deterministic settings.
```

Peak memory is bounded by holding four `z` tensors at once:
`4 × 12 × 272 × 1136 × 4 B ≈ 59 MB`.

**No training is proposed, and none is required** (§XIV of the brief): the
question is entirely about the response of *existing* weights to a *changed
input pairing*.

---

## 13. CLAIM CEILING

### Under outcome A — the maximum permitted claim

> Under the specified synthetic fronto-parallel 2×2 pairing intervention, the
> location of the trained aggregation's **nonlinear left↔right pairing response**
> coincides with the imposed matching candidate `k = τ` at every tested `τ`, for
> all three checkpoints and all six scene pairs, and lies outside the observed
> random-weight population. The response is therefore sensitive to **which** left
> and right images are actually paired — not only to the translation applied to
> the right image.

**Required phrasing:** *"outside the observed random population"* — the
reference is 576 sampled parameterisations, **not** the space of random weights.

### Explicitly NOT claimed, under any outcome

complete correspondence · correct matching everywhere · correct disparity ·
sub-candidate precision · genuine disparity search · stereo correctness ·
real-scene correspondence · metric depth correctness · generalisation ·
"impossible under random weights" · "proves learned correspondence".

### Terminology, fixed (inherited from `ALGEBRA_AUDIT` §6)

- **synthetic geometric pairing response** — the response depends on which
  images are paired, at the imposed candidate. *Attainable.*
- **architectural correspondence** — produced by the operator irrespective of
  weights. *The `G-STOP` outcome.*
- **learned correspondence** — weight-dependent and attributable to matching.
  *Only partially attainable: the gate shows weight-dependence, never that the
  mechanism is matching.*
- **correspondence** (unqualified), **genuine disparity search**, **real-scene
  correspondence** — **not attainable** by this construction.

### The irreducible gap, restated

Nothing in this design separates *learned correspondence* from *a learned
non-correspondence shortcut that happens to place its nonlinear pairing minimum
at `k = τ`*. `ALGEBRA_AUDIT` §5.3 marks this **UNKNOWN** and not addressable by
any ablation of this architecture on synthetic input. It remains UNKNOWN.

---

## 14. FINAL DESIGN VERDICT

```
DIRECT-GEOMETRIC-DIAGNOSTIC-VALID          (gated)
```

**Because**, and only because, the three mechanisms make **precomputable,
different** predictions:

| | `I_τ` non-zero? | extremum at `k = τ` for all six `τ`? |
|---|---|---|
| **genuine correspondence (H1)** | yes | **yes** |
| **generic binocular dependence (H2)** | **no — cancels identically** | no |
| **architectural candidate-axis artifact (H3)** | **exactly zero in the linear part**; small nonlinear residue | **no reliable tracking** — `h ≤ 3` over 8 synthetic random seeds, `h ≤ 2` affine |

The separation rests on an **identity in the source**
(`V_AA + V_BB − V_AB − V_BA ≡ 0`, because the cost volume is linear in both
feature maps), not on an expectation about magnitudes. That identity is the one
thing every previous design lacked: a functional in which the known artifact
cancels *exactly* rather than *hopefully*.

**"Gated" is load-bearing.** Stage 1 can terminate the experiment with
`STATISTIC-ARCHITECTURALLY-AVAILABLE` before any trained weight is inspected.
That is a **preregistered terminal outcome**, not a disappointment — it is the
ninth experiment discovering a defect in its own design *before* producing a
number to argue about, instead of after.

**Not `LEVEL-D-NOT-YET-IDENTIFIABLE`:** the identifiability barrier that closed
the translation class (§3.3 — crossed volumes translate exactly as matched ones
do) is *dissolved*, not worked around, by holding the translation fixed and
varying only the pairing.

**Not `CORRESPONDENCE-BRANCH-STOP`:** a defensible cheap diagnostic exists,
it needs no new conceptual apparatus, and it costs minutes.

**Verdict conditions.** This verdict is void unless the preregistration carries,
without exception: the gate running first (§9.2, §11); separation rather than
containment (§9.3); hard stops written against derivable properties **and
enforced in code** (§11.1); the non-injectivity proof for the statistic (§8.2);
shape conditions applied symmetrically to every arm; the contamination
disclosure of §0; and the status declared in advance as **consistency evidence
at best**.

---

## 15. THE EXACT ANSWER

> **What is the single cheapest experiment that can genuinely distinguish
> geometric correspondence from the architectural candidate-axis artifact?**

**Hold the translation fixed and swap the pairing. Measure where the 2×2
interaction of the candidate-cost profile has its minimum.**

Concretely, at each `τ ∈ {1,…,6}` and for two real KITTI frames `A`, `B`, run
four forward passes — `(A, A_τ)`, `(B, B_τ)`, `(A, B_τ)`, `(B, A_τ)` — and test
the single boolean

```
argmin_k  Ī_τ(k)  ==  τ ,        Ī_τ = ½[z_AA + z_BB] − ½[z_AB + z_BA]
```

with the random-weight arm run **first** as a gate.

**Why this and nothing cheaper works.** The architectural artifact is the
aggregation's linear shift-equivariance plus its D-axis padding boundary. Both
live in the **linear** part of the operator, and the cost volume is **linear in
both feature maps**, so

```
V_AA + V_BB − V_AB − V_BA  ≡  0                    identically, ∀ k, w, τ
```

The artifact therefore **cancels exactly** in the interaction — as does every
main effect of the left image, the right image, `τ`, the crop geometry and the
padding. What remains can only have been produced by the network responding to
*which two images were paired*, and under genuine correspondence it must sit at
the one candidate where `V` vanishes, `k = τ`, because `softmax(−z)` makes a
match a minimum.

Every cheaper intervention — any variant of translating, corrupting, scrambling
or re-rendering the right image — modifies the cost volume's translation
structure, which the shift-equivariant aggregation converts into candidate-axis
response with **no matching whatsoever** (§3.3: a *cross-paired* volume
translates exactly as a matched one does). Those designs cannot separate the
mechanisms at any sample size.

**Cost:** ~15 000 inference passes, **5–10 minutes**, no training, four extra
forward passes per condition over what GEOM-002 already ran.

---

## 16. FILES IN THIS RECORD

| file | contents |
|---|---|
| `DESIGN_AUDIT.md` | this document — the §XX report and the frozen design |
| `design.json` | the machine-readable preregistration specification |
| `RELATED_RUNS.md` | provenance of every record relied on, with hashes |
| `source_trace.md` | the complete source-derived geometry mapping |
| `synthetic_mechanism_check.py` | analytical/synthetic mechanism test — **no trained model, no real data, no training** |
| `synthetic_mechanism_check.json` | its output |

**No `RESULTS.md`. Nothing executed. The experiment is NOT launched.**
