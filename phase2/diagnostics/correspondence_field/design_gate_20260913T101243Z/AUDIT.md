# Design / feasibility gate — spatially varying disparity-field intervention

**Id:** `correspondence_field/design_gate_20260913T101243Z`
**Type:** design / feasibility / identification gate. **Nothing was executed.**
No checkpoint opened, no inference, no training, no threshold, no historical
record modified, no image warped. No new measurement was taken; this gate
reasons from records already on file.

**Predecessor, authoritative:**
`correspondence_pipeline/identification_audit_20260913T083245Z` — the
right-crop-translation family is CLOSED AS NOT IDENTIFIABLE. Nothing from that
family (`S`, `S*`, zone translations, swap controls, global translations, O1,
CP-001, translated self-pairs, candidate permutations) is revisited here except
where a result of that audit is *cited*.

**Claim ceiling, unchanged and unaffected by anything below:**
**Level D — genuine geometric correspondence NOT-DEMONSTRATED.**

**Question:** can a spatially varying disparity-field intervention be built from
existing KITTI 2015 data that satisfies conditions 1–6 of the brief?

**Answer in one line:** conditions 1, 2, 3, 5 are satisfiable, and one
construction (`D-y`, integer per-row shift) satisfies condition 4 far better than
anything in the closed family — but condition 6 fails, for the same two
adversaries that closed the previous branch, because the obstruction lives in the
**null**, and the null obstruction proved in the previous audit **does not
mention translation anywhere** and therefore transfers to this family unchanged.

---

## 1. Intervention families

### 1.0 Fixed frame and notation

Frozen geometry, all measured on file (`HS-SLACK`, `HS-GT`): frames
`(368, 1232, 3)`, crop `272 x 1136` at `Y0 = 96`, stride 16, `D = 12`
candidates, feature grid `17 x 71`, dependency window at `disparity_initial`
about **573 px** (238 extractor radius + 80 aggregation + 16 upsample), measured
bit-exact in `prefreeze_audit_20260913T075420Z`.

* `F_L`, `F_R` — base scene frames; `F'_R` — donor scene right frame.
* `L = C(F_L, 0)`, `L[y, x] = F_L[96 + y, x]` — **fixed in every arm, every
  construction, every value of the field.** Condition 1 is satisfied by
  construction throughout and is not discussed again.
* `d(u, y)` — true disparity indexed by **right**-image column `u`, so that
  `F_R[y, u] ≈ F_L[y, u + d(u, y)]`. The sign is the one measured by `HS-FRAME1`
  (unanimous 12/12, about 1.06 M GT pixels).
* `delta(x, y)` — the **imposed** field, in pixels. Candidate units are
  `delta / 16`.

### 1.1 Family A — warp an existing real right image

    R_A[y, x] = F_R[96 + y, x + delta(x, y)]

Content at output column `x` is the scene point that sat at right column
`x + delta`; that point's left column is `x + delta + d(x + delta, y)`. So the
pair `(L, R_A)` carries the disparity field

    d'(x, y) = delta(x, y) + d(x + delta(x, y), y)                         (1)

i.e. an **approximately additive imposed perturbation**, exactly additive when
`d` is locally constant over `delta`.

* **Changed:** the geometric arrangement of real right-image content.
* **Fixed:** the left image; the right image's photometry, sensor noise, scene
  identity, illumination and content inventory (up to resampling and to columns
  skipped or duplicated by the warp).
* **Not used at all:** ground truth. The treatment needs no disparity map. GT is
  needed only for pixel admissibility (keeping `d + delta` inside `[0, 11]`),
  which is a selection rule, not part of the construction. This is a real
  advantage over Family B and removes audit item 9 from the treatment arm.

### 1.2 Family B — resynthesise the right image from the left image and a modified disparity map

    R_B[y, x] = L[y, x + d_target(x, y)]

* **Changed:** everything. `R_B` contains no right-image data at all.
* **Fixed:** only `L`.
* **Requires** a *dense* `d_target`. KITTI 2015 GT is sparse — measured 3.3 %–41 %
  valid in the tested interiors (`HS-GT`, per-scene) — so `d_target` must be
  completed by interpolation or by a model, which makes the intervention depend
  on an **estimated** disparity field (audit item 9) and invents geometry in the
  59–97 % of pixels with no GT.
* **Also** `(L, R_B)` is a self-pair: `R_B` is a pointwise resampling of `L`,
  with zero independent sensor noise, zero view-dependent shading and perfect
  photometric identity at the true offset. The campaign's standing prohibition on
  **translated self-pairs** was issued for exactly this property (the anomaly is
  not separable from the geometric match). A *spatially varying* self-warp is a
  generalisation of the prohibited object, not an exception to it.
* **Verdict: REJECT at the construction stage.** Not audited further.

### 1.3 Family C — piecewise (step) field

`delta` piecewise constant. Steps along `x` are occlusion events (section 2.4);
steps along `y` are not (epipolar lines are horizontal). Sharp seams maximise
the intra-image cue exploited by adversary A10.

### 1.4 Family D — smoothly varying field

`delta` smooth. Two sub-cases must be kept apart because their artefact
signatures are completely different:

* **`D-x`** — `delta` varies along `x`. The sampling map `u = x + delta(x)` has
  Jacobian `1 + ∂delta/∂x`, so the right image is locally **compressed or
  stretched**. Sub-pixel values require interpolation.
* **`D-y`** — `delta` varies along `y` only, with **integer** values:
  `R[y, x] = F_R[96 + y, x + delta(y)]`, `delta(y)` in `Z`. Each row is an exact
  integer pixel copy. **No interpolation, no aliasing, no stretch, no
  duplication, no holes, within a row.** This is the only artefact-free
  spatially varying construction available, and it is the one carried forward.

---

## 2. Image-formation validity audit

Applied to Families A / C / D. "Looks plausible" is not used as a criterion
anywhere below.

| # | Issue | `D-x` (smooth in x) | `C-x` (step in x) | **`D-y` (integer, per row)** |
|---|---|---|---|---|
| 1 | **Interpolation artefacts** | Present everywhere. Bilinear resampling low-passes by an amount that depends on the **fractional part of `delta`**, so the blur field is *spatially correlated with the imposed field*. This is an artefact that carries the signal. | Absent if `delta` integer | **Absent.** Exact pixel copy. |
| 2 | **Aliasing** | Present wherever `∂delta/∂x > 0` (decimation) | At steps only | **Absent.** |
| 3 | **Holes / disocclusion** | Where the map stretches (`∂delta/∂x < 0`) content must be *newly revealed background*; the warp instead duplicates or interpolates foreground. **Physically invalid.** | Same at down-steps | **Absent.** A per-row shift never stretches. |
| 4 | **Occlusion boundaries** | Where the map compresses (`∂delta/∂x > 0`) right-image columns are skipped. That *is* what occlusion does, so this direction is valid. The consequence is severe: validity holds only for **monotone non-decreasing `delta` in `x`** (section 2.1). | Same | **Not applicable.** Occlusion is a within-row phenomenon; a per-row shift induces none. |
| 5 | **Duplicated / missing pixels** | Duplication in stretch regions (invalid), loss in compression regions (valid) | Same | **Neither.** Every row is a bijective copy of a contiguous run of source columns. |
| 6 | **Texture stretching / compression** | Present by construction; magnitude `|∂delta/∂x|` | At steps | **Absent.** |
| 7 | **Invalid disparity regions** | `d' = delta + d` must stay in `[0, 11]` candidates; GT `d_c` measured p05–p95 = 0.36–7.5 candidates, so admissibility must cap `d_c` | Same | Same — a preregistrable selection rule, not a construction defect |
| 8 | **Border requirements** | `x + delta <= 1231` with `x < 1136` gives `delta <= 96` px = **6 candidates**, the same budget `HS-SLACK` already verified in range with **zero all-zero rows or columns** in all 12 scenes | Same | Same |
| 9 | **Dependence on the original disparity estimate** | **None** — `delta` is prescribed; GT enters only as a selection rule | None | **None** |
| 10 | **Within the image-formation model?** | Only for monotone `delta`; and then the "spatial variation" is a monotone bounded ramp | Only one step direction | **Yes within a row**, as a stack of fronto-parallel horizontal bands. **But** such a banded-depth scene is far from the KITTI depth prior — see A8. |

### 2.1 The occlusion-validity constraint on `D-x`, stated as a finding

For `R_A[x] = F_R[x + delta(x)]` to invent no content, the sampling map must not
stretch: `∂delta/∂x >= 0` everywhere. Combined with the border budget
`0 <= delta <= 96`, the occlusion-valid `x`-fields are exactly the **monotone
non-decreasing functions `[0, 1136) -> [0, 96]`**. They cannot oscillate, and
their total excursion across the whole crop is **6 candidates**, so within a
573 px dependency window the field typically varies by at most 2–3 candidates.

This is a genuine structural narrowing, and it matters for a reason beyond
richness: a monotone bounded ramp is a **low-dimensional, globally
parameterised** deformation — for the linear case, a single scale parameter,
which is the standard slanted-plane transformation. The more one forces `D-x` to
be physically valid, the closer its treatment arm moves back toward a globally
parameterised transformation of the right crop. It does not literally re-enter
the closed family (the field is not constant, so `ZI-2` does not apply), but the
adversary's task shrinks from "estimate a field" to "estimate one or two
scalars".

`D-y` does not suffer this: `delta(y)` may be arbitrary and non-monotone, with
the full 0–96 px range, and remains artefact-free. **`D-y` is therefore the
strongest construction in the family and the only one worth auditing further.**

---

## 3. Conditions 1–5 of the brief, adjudicated for `D-y`

| Condition | Status for `D-y` |
|---|---|
| 1. left image fixed | **Satisfied** by construction |
| 2. prescribed perturbation imposed on the right image | **Satisfied**, exactly, with no interpolation |
| 3. perturbation varies within the same receptive field | **Satisfied, and strongly.** The extractor's vertical radius is 238 px and the crop is only 272 px tall, so *every* tested pixel's window spans the full crop height. A field varying in `y` therefore varies inside every window. `ZI-2`, the identity that collapsed the previous family, cannot hold here: no single scalar offset describes any window. |
| 4. locally natural enough to be pipeline evidence | **Partially.** Photometrically exact, artefact-free, real content, valid within each epipolar row. But the implied scene is a stack of fronto-parallel bands, which is off the KITTI depth prior, and the band seams are structures with no natural counterpart. Naturalness is claimed at the *epipolar-row* level only, and must never be claimed at the scene level. |
| 5. correspondence predicts a spatially varying response | **Satisfied.** `H_CORR` predicts `d̂(x, y)` tracks `d + delta(y)`, i.e. `∂d̂/∂delta = +1` row-wise, with the response pattern reproducing the imposed band pattern. |
| 6. strongest correspondence-free alternatives cannot reproduce it | **FAILS.** Section 4 and section 5. |

Conditions 1–5 being satisfiable is a real result and is recorded as such. It is
also not sufficient, and the brief's sixth condition is the one that decides.

---

## 4. Adversary table

Two mechanisms from the closed branch are carried over by name because they are
the ones that decide this gate as well:

* **A4s** — *scene-identity conditioning*: any response magnitude conditioned on
  content class. Fatal whenever the arm label is confounded with scene identity.
* **A5r** — *readout pinning*: on non-corresponding content the aggregated cost
  profile is unstructured, the 12-candidate soft-argmin sits near `E[k]`, and the
  output's response to any input change is noise-like with zero mean. A null's
  "no response" then follows from readout geometry rather than from absent
  geometry, and it *predicts* the `H_CORR` null result.

"Defeated?" below means defeated **by the `D-y` intervention paired with the best
null available for it** (N1f, section 5.1), not by the intervention alone.

| # | Adversary | Explicit mechanism | Defeated? |
|---|---|---|---|
| A1 | **Monocular left-image predictor** | `d̂ = m_L(x, y)`. `L` is fixed, so `∂d̂/∂delta = 0`. Cannot produce any response. | **YES** |
| A2 | **Local image-warp / statistics response without correspondence** | Response driven by low-order statistics of the warped right patch | Partly — the null carries the *same* `delta` and the same warp, so the artefact-borne part cancels; the content-borne part does not (that is A4s) |
| A3 | **Interpolation / warp-artefact detector** | Reads the resampling signature, which in `D-x` is correlated with `delta` | **YES, twice over**: `D-y` has *no* resampling signature at all, and the null carries an identical field so any signature would cancel |
| A4 | **Spatial texture / edge response** | Shifting a row changes which edges occupy the window | Partly, as A2 |
| **A4s** | **Scene-identity conditioning** | The treatment right image is *always* the base scene; any content-conditioned response differs between arms with no geometry involved | **NO — fatal** |
| A5 | **Disparity-field magnitude detector responding to the warp itself** | A detector of the imposed deformation, not of disparity | **YES** — the null carries the identical field, so this cancels exactly. This is the one place the `delta`-field design is genuinely stronger than the closed family |
| **A5r** | **Candidate-axis / readout pinning** | Flat profile on non-corresponding content pins the soft-argmin near `E[k]`; the spatial pattern fails to appear in the null for readout reasons | **NO — fatal** |
| A6 | *(as A5r above)* | — | — |
| A7 | **Receptive-field / convolutional response** | Window geometry identical in both arms | **YES** |
| A8 | **Synthetic-image distribution shift** | Band seams and banded depth are off-prior | Partly — both arms are warped identically, so the shift is common; but the null is *additionally* shifted by content mismatch, so the arms are not equally off-distribution |
| A9 | **Occlusion / disocclusion cues** | `D-y` induces no within-row occlusion, so there is no cue to read | **YES** for `D-y`; **NO** for `D-x` and `C-x` |
| **A10** | **Intra-image deformation reading** (mechanism `M1`, below) | Recover `delta` from the right image alone, never comparing `L` and `R` | **YES** — see 4.1. This is the design's second genuine win |

### 4.1 `M1`, the intra-image deformation reader — stated explicitly, then killed

`M1` is the strongest *new* adversary the `delta`-field direction creates, so it
is written out rather than asserted.

Under `D-y`, rows `y` and `y + 1` of the warped right image show almost the same
scene content displaced horizontally by `delta(y + 1) - delta(y)`. Wherever the
scene is locally vertically coherent — road surface, walls, vegetation, sky, i.e.
most of a KITTI frame — that displacement is recoverable **from the warped right
image alone** by an intra-image displacement estimator, which a bank of oriented
2D convolutions can implement (a sheared edge produces a response monotone in the
shear). Then

    delta_hat(y) = sum over y' < y of  estimated row-to-row displacement
    d_hat(x, y)  = m_L(x, y) + delta_hat(y) + c

reproduces the full spatially varying `H_CORR` signature — correct pattern,
correct sign, unit gain if `c` is calibrated — **with no left-right comparison of
any kind**. `c` need not be invented: the deployed degenerate checkpoint already
has a measured non-geometric global response of up to 0.70 candidates to
right-image translation.

`M1` is killed, cleanly, by the paired null: the null carries the **same
`delta(y)`** applied to a donor right image, which is also a natural, vertically
coherent image, so `M1` produces the **same** row pattern in both arms and
cancels exactly in the differential.

That is a real result. It is also not enough, because A4s and A5r do not care.

---

## 5. Candidate null constructions

Hierarchy required by the brief: (1) `L` identical, (2) tested receptive field
identical in marginal content, (3) `delta` identical, (4) zone/field location
identical, (5) candidate-axis exposure identical, (6) the only intended
difference is whether the translated right content corresponds to the tested left
content. **Nothing below was executed.**

### N1f — same `delta` field, donor right image

* **Intervention:** `R[y, x] = F_R[96 + y, x + delta(y)]`, `delta(y)` integer.
* **Null:** `R[y, x] = F'_R[96 + y, x + delta(y)]`, identical `delta`, identical
  kernel (none), identical field location, identical candidate exposure.
* **Held fixed:** `L`; `delta`; the warp operator; the row banding; the
  candidate-axis exposure; the pixel set and admissibility rule.
* **Changed:** the right image's scene identity, and with it every joint statistic
  it has with `L`.
* **`H_CORR`:** treatment response tracks `delta(y)` at unit gain; null response
  is flat in `delta`.
* **Strongest `H_ALT`:** **A5r** — in the null the profile is unstructured, the
  soft-argmin pins near `E[k]`, and the row pattern fails to appear for readout
  reasons; **A4s** — the base scene is always the treatment and the donor always
  the null, so any content-conditioned response magnitude differs between arms
  with no geometry.
* **Nuisance:** wholesale change of right-image content.
* **Statistic invariant to it?** **No.** Differencing removes level offsets
  between arms; it does not remove differences in *sensitivity*, which is exactly
  what A4s and A5r are.
* **Adversary reproduces the expected result?** **Yes** — A5r does not merely
  permit the predicted null, it predicts it.
* **Null physically valid?** Yes. **Images in the model's valid input domain?**
  Marginally — pixel-value distribution preserved exactly, band seams
  off-distribution, equally in both arms.
* **Closes the adversary set?** **No.**
* **Verdict: REJECT.**

### N2f — same `delta`, same true right image, correspondence broken by an out-of-window constant

* Adds a large constant so the true match leaves `[0, 11]`.
* `H_CORR` predicts a flat response by **saturation**; A5r predicts a flat
  response by **pinning**. The two hypotheses make the same prediction.
* **Verdict: REJECT** — zero discriminating power by construction.

### N3f — `delta` versus a spatially permuted or sign-flipped `delta`, same true pair

* Both arms retain correspondence; there is no correspondence contrast at all.
  The statistic asks only whether the response pattern follows the imposed
  pattern, which `M1` also does.
* This is precisely the prohibited inference "the response follows the imposed
  field, therefore correspondence", wearing a control's clothing.
* **Verdict: REJECT.**

### N4f — `delta` applied to the true right image, with `L` replaced by a donor left

* Violates hierarchy item 1, and changes the monocular pathway that supplies most
  of the prediction.
* **Verdict: REJECT.**

### N5f — `delta` applied to a statistics-matched real donor patch (histogram/spectrum matched)

* Matches the statistics the designer chose to match, not the ones the mechanism
  uses. A4s survives (identity still differs), A5r survives (joint structure with
  `L` still destroyed).
* **Verdict: REJECT.**

### N6f — `delta` applied to the true right image, correspondence broken by a *within-window* content substitution that preserves the match

* This is the construction the brief's FOURTH section is really asking for: same
  imposed field, same warp statistics, same interpolation statistics, same local
  right-image marginal structure, different geometric relationship.
* The first four clauses are constructible. The fifth requires different content.
  "Same marginal structure" is therefore the strongest form of sameness
  available, and **marginal matching is not the property A4s and A5r respond to**
  — they respond to content identity and to the joint structure with `L`.
* **Verdict: REJECT.** The requested contrast is **constructible but
  insufficient**, and section 6 shows why that is structural rather than a
  failure of ingenuity.

| Null | (1) L fixed | (2) RF marginals matched | (3) same `delta` | (4) same location | (5) same candidate exposure | (6) only correspondence differs | Fatal adversary | Verdict |
|---|---|---|---|---|---|---|---|---|
| N1f donor image | yes | marginals only | yes | yes | yes | no — identity also differs | A4s, A5r | **REJECT** |
| N2f out-of-window constant | yes | yes | yes | yes | no (saturated) | no | A5r; same prediction under both hypotheses | **REJECT** |
| N3f permuted field | yes | yes | no (that is the contrast) | yes | yes | no — both correspond | M1 / prohibited inference | **REJECT** |
| N4f donor left | **no** | — | yes | yes | yes | no | hierarchy violation | **REJECT** |
| N5f statistics-matched donor | yes | marginals only | yes | yes | yes | no | A4s, A5r, accidental match | **REJECT** |
| N6f the brief's ideal contrast | yes | marginals only | yes | yes | yes | **unattainable** | A4s, A5r | **REJECT** |

---

## 6. Identification analysis

### 6.1 The field decomposes, and the two parts have opposite problems

Write `delta = delta_bar + delta_tilde`, the constant part and the
spatially varying part.

* **Response to `delta_bar`** is the global right-image translation slope. That
  statistic is closed as not identifiable
  (`identification_audit_20260913T083245Z`). Nothing here reopens it, and it must
  not be used as primary evidence.
* **Response to `delta_tilde`** is the new content of this direction. But
  `delta_tilde` is an **intra-image geometric deformation**: it deforms the right
  image relative to itself, and a deformation of a single image is in principle
  observable from that image alone. `M1` (section 4.1) is the explicit
  construction.

So the varying part is, a priori, readable without correspondence, and the
non-varying part is the closed statistic. The design's entire hope therefore
rests on the null cancelling `M1` — which it does — leaving the contrast to be
decided by whatever else differs between arms. And what else differs is content.

### 6.2 The null obstruction transfers, because it never mentioned translation

The previous audit proved: *for a fixed left crop, a null cannot simultaneously
satisfy N4 (identical right content inside the tested window) and N5 (different
geometric relationship to the left content), because the geometric relationship
is a function of the pair `(L|_W, R|_W)`, and fixing both fixes it.*

Read the proof again with the present family in mind. It quantifies over the
**content of the window**, and says nothing about how that content was produced.
Whether `R|_W` is a crop, a translated crop, a per-row shifted crop, or a warped
crop is irrelevant to the argument. **The obstruction is
intervention-agnostic.** Imposing a spatially varying field changes the treatment
arm; it does not create a loophole in the null.

Consequently the fatal pair survives intact:

* **A4s** survives because the null must change content, and content identity is
  then perfectly confounded with the arm label.
* **A5r** survives because destroying correspondence is precisely what flattens
  the cost profile, so "the imposed pattern does not appear in the null" is a
  consequence of the manipulation's side effect rather than of its intended
  effect. No arm in any of N1f–N6f separates them, and none can, because the flat
  profile and the absent correspondence are the same act.

### 6.3 Why the prohibited inferences are indeed prohibited here

The brief forbids both *"the corresponding image follows the imposed field,
therefore correspondence"* and *"the non-corresponding image does not follow it,
therefore correspondence"*. This audit does not use either:

* the first is defeated by `M1`, which follows the imposed field with the right
  pattern, sign and gain while never comparing `L` and `R`;
* the second is defeated by `A5r`, which produces a flat null response from
  readout geometry alone.

The differential of the two is defeated by the conjunction of `A4s` and `A5r`,
which is a *single explicit correspondence-free account of the whole contrast*:
the arms differ in content, content conditions both the response magnitude and
the structure of the cost profile, and a structured profile moves while an
unstructured one pins.

### 6.4 What was actually gained

Recorded so the ledger is honest, because this direction is genuinely better than
the closed one on the treatment side:

* `ZI-2` does not apply — no single scalar describes a tested window, so the
  treatment arm is *not* the global-translation statistic in disguise.
* `D-y` is interpolation-free, alias-free, hole-free, occlusion-free within a
  row, and uses no ground truth in its construction.
* `A3`, `A5`, `A9` and `A10`/`M1` are all genuinely killed.

None of that reaches the null, and the null is where the branch fails.

---

## 7. Feasibility constraints, for the record

| Quantity | Value | Source |
|---|---|---|
| Field budget | `delta` in `[0, 96]` px = **0–6 candidates** | 96 px frame slack, `HS-SLACK` verified in range, zero all-zero rows/columns in 12/12 scenes |
| Vertical variation inside every window | full crop height, since extractor vertical radius 238 px > crop height 272 px / 2 | architecture |
| Horizontal variation inside a window, occlusion-valid | at most about 2–3 candidates (monotone `delta`, 573 px window) | section 2.1 |
| GT disparity in the crop | p05–p95 = 0.36–7.5 candidates | `HS-GT`, 12 scenes |
| Admissibility needed to avoid the candidate ceiling | `d_c <= 11 - max(delta)/16` | arithmetic |
| GT coverage in tested interiors | 3.3 %–41 % per scene; worst case 2 222 px | `HS-GT` |
| Dense disparity map available? | **No** — required by Family B, fatal to it | `HS-GT` |
| Inference unit | still 3 checkpoints | standing defect, unchanged |

Feasibility is therefore **not** the binding constraint. The data supports the
construction. Identification is the binding constraint.

---

## 8. Verdicts

| Item | Verdict |
|---|---|
| Family A (warp real right image) | **Feasible**; occlusion-valid only for monotone `delta` in `x` |
| Family B (resynthesis from `L`) | **REJECT** — generalised self-pair, prohibited class; and requires a dense disparity map KITTI does not provide |
| Family C (piecewise) | **REJECT in `x`** (invalid down-steps, strong seam cues); reduces to `D-y` in `y` |
| Family D-x (smooth in `x`) | **REJECT** — `delta`-correlated interpolation and aliasing artefacts, invalid stretch regions, and occlusion-validity forces a globally parameterised ramp |
| **Family D-y (integer per-row shift)** | **Construction ACCEPTED as valid and near-artefact-free; design REJECTED on identification** |
| N1f donor image | **REJECT** |
| N2f out-of-window constant | **REJECT** |
| N3f permuted field | **REJECT** |
| N4f donor left | **REJECT** |
| N5f statistics-matched donor | **REJECT** |
| N6f the brief's ideal contrast | **REJECT** — constructible in four clauses of five, insufficient in the one that matters |
| Brief conditions 1, 2, 3, 5 | **Satisfiable**, by `D-y` |
| Brief condition 4 | **Partially** — natural per epipolar row, off-prior as a scene |
| Brief condition 6 | **FAILS** |

---

## 9. Final decision

### 4 — CLOSE PIPELINE BRANCH

The reasoning, in order, so the operator can stop it at whichever step they
decline to accept:

1. For **this direction**, the finding is exactly the brief's option 2:
   *feasible but not identifiable*. A physically valid, artefact-free, spatially
   varying intervention exists (`D-y`), it satisfies conditions 1, 2, 3 and 5,
   and it defeats four adversaries the closed family could not — but no valid
   mechanistic null closes `H_ALT`, because A4s and A5r survive every null in
   section 5.
2. The obstruction is **not specific to this direction**. The null-impossibility
   argument quantifies over the content of the tested window and never mentions
   how that content was produced, so it applies to *any* design of the form
   "fixed left image, modified right image, paired null" — and by the symmetry of
   the argument, to the mirrored form with `L` and `R` exchanged.
3. This direction was on record as **the only explicitly unaudited one**. With it
   falling to an obstruction that generalises, no identifiable route remains
   under the campaign's evidence standard with existing data. Hence option 4
   rather than option 2.

**If the operator does not accept step 2's generalisation**, the verdict
downgrades to **2 — FEASIBLE BUT NOT IDENTIFIABLE**, scoped to the spatially
varying disparity-field family. Steps 1 and 3 are unaffected either way; only the
branch-level consequence changes.

### What would reopen the branch

Stated so the closure is not mistaken for a claim that correspondence is absent,
and **not** as a recommendation:

* Data in which correspondence can be varied **independently of content** — a
  multi-view or multi-baseline capture of the same scene, or a renderer with
  ground-truth control. Both are *new data*, outside the existing KITTI 2015
  material, and a renderer would concede the Type-A property explicitly and would
  require its own ceiling.
* An architectural readout that is not a bounded soft-argmin over 12 candidates,
  which is what makes A5r unfalsifiable here. That is a change to the artefact
  under study, not an experiment on it.

Neither is proposed, and no experiment is created by this gate.

**Level D — genuine geometric correspondence NOT-DEMONSTRATED.** Unchanged.
"Not demonstrated" remains not the same as "absent", and nothing in this audit
is evidence either way about the mechanism itself.
