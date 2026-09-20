# Identification audit of the swap-referenced statistic `S*`

**Id:** `correspondence_pipeline/identification_audit_20260913T083245Z`
**Type:** design / identification audit. **Nothing was executed.** No checkpoint
opened, no inference of any arm, no training, no threshold derived, no
historical record modified. No new measurement was taken; this audit reasons
from records already on file.

**Claim ceiling, unchanged and unaffected by anything below:**
**Level D — genuine geometric correspondence NOT-DEMONSTRATED.**

**Question asked:** can a correspondence-free mechanism `H_ALT` produce a
nonzero `S* = beta_treatment - beta_swap` without performing local left-right
correspondence?

**Answer: YES — explicitly, by at least two named mechanisms (A4s, A5r below),
neither of which performs correspondence.** `S*` is therefore rejected. Every
alternative paired null examined is also rejected, and an impossibility argument
(section C) shows why the rejections are structural rather than incidental.

**Final decision: 2 — DESIGN NOT IDENTIFIABLE.** Scope of that verdict is stated
exactly in section G.

---

## A. Formalisation

### A.1 Objects

Frames, after the project's `pad_and_crop` (measured `(368, 1232, 3)` in all 12
scenes, `HS-SLACK`):

* `F_L`, `F_R` — the base scene's rectified left and right frames.
* `F'_L`, `F'_R` — a donor scene's frames (`F'` is used only by the swap arm;
  `F'_L` is never fed to the model).

Crop operator, with the frozen geometry (`Y0 = 96`, crop `272 x 1136`,
`STRIDE = 16`, `D = 12`, feature grid `17 x 71`):

    C(F, x0)[y, x] = F[96 + y, x0 + x],   0 <= y < 272,  0 <= x < 1136

### A.2 The four inputs

**Fixed left crop**, identical in every arm, at every `Delta`, in both zone
placements:

    L = C(F_L, 0)

**Delta intervention**, one-sided, as amended on record after `HS-GT`:

    Delta in {0, 1, 2, 3, 4, 5, 6},   right-crop origin offset = 16 * Delta

    x_R = x_L + 16*Delta   =>   d' = d + 16*Delta   =>   k' = d/16 + Delta

so `H_CORR` predicts `beta = +1`, **not** `-1`. (`DESIGN.md` sections 5.3 and 7
still carry the superseded `-1`.)

**Zone and complement**, seam at `SEAM = 568`:

    Z   = { x : 0   <= x < 568 }
    Z^c = { x : 568 <= x < 1136 }

**Treatment right crop:**

    R_T^Delta[y, x] = F_R[96 + y, x + 16*Delta]   if x in Z
                    = F_R[96 + y, x]              if x in Z^c

**Swap-control right crop** (donor scene, same origin offset):

    R_S^Delta[y, x] = F'_R[96 + y, x + 16*Delta]  if x in Z
                    = F_R[96 + y, x]              if x in Z^c

**Tested interior** (bit-exact bounds measured in
`prefreeze_audit_20260913T075420Z`, *not* `DESIGN.md`'s `[0, 250)`):

    I = { x : 0 <= x < 224 }        (zone placement = left half)

Pixel admissibility, fixed per `(scene, placement)` and identical across all
`Delta` and both arms: GT valid (`gt > 0`) and `d_c = gt/16` in `[0.5, 5.0]`, so
that `k' = d_c + Delta` stays inside `[0, 11]` for every `Delta <= 6` and the
upper candidate ceiling is never reached.

### A.3 Exactly what differs

| Region | `R_T^Delta` vs `R_S^Delta` |
|---|---|
| `Z` (568 columns, all 272 rows) | **differ at every `Delta`, including `Delta = 0`** — donor content versus base content |
| `Z^c` | **bit-identical** — both are `C(F_R, 0)` restricted to `Z^c` |

At the feature and readout level, for `x` in `I`:

* the entire dependency window of `disparity_initial(x)` lies inside `Z`
  (measured: `ZI-2 = 0.0` exactly, 3 seeds, `Delta` in {1,3,6});
* therefore a tested pixel sees, in each arm, a natural right-image patch that
  is translated by **exactly `16*Delta`** as `Delta` increases;
* the group action on the right content is **identical** in the two arms;
* the left content inside the window is **identical** in the two arms and
  **invariant in `Delta`**;
* the **only** difference inside the tested window is *which natural patch* the
  right content is — equivalently, whether it is the epipolar-corresponding
  patch for `L`.

That last line is the design's intended difference. It is also, stated plainly,
a **wholesale change of right-image content**, and that is the opening every
adversary below walks through.

### A.4 The inherited collapse

`ZI-2` was measured on the treatment construction, and the `S*` proposal does
not change the treatment arm. So the identity survives intact:

    beta_treatment  ==  beta_global  restricted to the tested interior

The treatment term of `S*` **is** the global right-image translation slope. Only
the reference term changes. Anything `S*` claims must therefore be carried by the
reference, not by the treatment.

---

## B. Correspondence-free adversaries

Convention used throughout, taken from the campaign's own definition
(`DESIGN.md` section 0 and section 6): a mechanism that recovers the local
left-right offset — by discrete search, by soft-argmin over candidates, or by any
continuous estimator — is **inside `H_CORR`** and is admitted. `H_ALT` is
everything else. An adversary only counts if it produces `S* != 0` **without**
recovering the local offset.

Two gate readings have to be audited separately, because they have different
adversary sets:

* **Gate-N ("nonzero"):** pass if `S*` exceeds a threshold derived from null
  spread. This is the gate form the campaign has used before.
* **Gate-C ("calibrated"):** pass only if `beta_T` lies in a band around `+1`
  **and** `beta_S` lies in a band around `0`.

| # | Adversary | Mechanism | Sign reproducible? | Killed by `S*` under Gate-N? | Under Gate-C? |
|---|---|---|---|---|---|
| 1 | **A1 — content-dependent translation response** `beta = f(local R stats, local L stats)`, `f` not a matcher | The two arms carry different right content, so any content-conditional `Delta`-response differs between them | Yes, either sign; magnitude scene-dependent | **NO** | Yes — cannot deliver unit gain without offset recovery |
| 2 | **A2 — monocular-from-left plus right statistics** | `m_L(x)` is `Delta`-invariant (L fixed) and contributes slope 0; the right-statistics term reduces to A1 | Yes | **NO** (via its A1 term) | Yes |
| 3 | **A3 — local texture / edge / gradient response** | Translating a patch changes which edges occupy the window; the readout responds to edge density, contrast, gradient energy | Yes; zero-mean across scenes, not zero per unit | **NO** | Yes |
| 4 | **A4s — zone-content / scene-identity conditioning** | Response magnitude conditioned on scene or texture class. Treatment is *always* the base scene, swap is *always* a different scene, so the conditioning variable is perfectly confounded with the arm label | Yes | **NO — fatal** | Only partially: symmetric pairing (A donates to B and B to A) makes the effect zero-mean but does **not** make it zero per unit, and it inflates the null spread that the threshold is derived from |
| 5 | **A5r — candidate-axis / readout pinning** | Soft-argmin output is `sum_k k * softmax(-c_k)`, bounded in `[0, 11]`. On non-corresponding content the aggregated profile is flat, the softmax is near-uniform, the output sits near `E[k]` and its `Delta`-response is noise-like with **zero mean slope**. `beta_S ≈ 0` then follows from readout geometry, not from absence of a match | Yes — it *manufactures exactly the null result `H_CORR` predicts* | **NO — fatal** | **NO.** Gate-C requires `beta_S ≈ 0`, which is precisely what this adversary supplies for free |
| 6 | **A6 — convolutional receptive-field effects** | Window geometry identical in both arms; tested interior depends only on zone content | — | **YES** — measured, `ZI-2 = 0.0` | Yes |
| 7 | **A7 — boundary / seam leakage** | Composite seam at 568 reaching a tested pixel | — | **YES**, *provided* the corrected interiors `[0,224)` / `[912,1136)` are used. `DESIGN.md`'s `[0,250)` / `[886,1136)` leak at 1–2 float32 ULPs (measured 4.8e-07 to 9.5e-07) | Yes |
| 8 | **A8 — repeated texture / accidental correspondence in the swap** | KITTI scenes share road surface, lane markings, vegetation, sky; a donor patch can contain content that genuinely matches `L` at some candidate, giving `beta_S != 0` | Biases `S*` **toward 0** | Not applicable as a false positive; it is a **false-negative** generator and it means the swap is not a clean null | Same |
| 9 | **A9 — spatial prior / absolute-position dependence** | Response conditioned on image column. Identical between arms at a fixed placement | Cancels in `S*` | **YES** | Yes |
| 10 | **A10 — right-image-alone reading of `Delta`** (recognise content, infer its true frame column, subtract the observed column) | Requires no left image at all | Yes, with unit gain in principle | **YES** — the donor content translates by the same `16*Delta`, so the mechanism acts identically in both arms and cancels exactly. *This is the one adversary the swap genuinely kills, and it is the strongest argument in `S*`'s favour* | Yes |
| 11 | **A11 — single-alignment residual magnitude** `h(||Lf[w] - Rf[w]||)` | Exploits image autocorrelation to read offset without searching the candidate axis | Yes, but only over the feature autocorrelation width (1–2 cells); the tested offsets span 0.5–11 candidates (8–176 px) | Inside `H_CORR` by the campaign's own definition (it *is* a local offset estimator), so it is admitted, not an adversary | Admitted |

### B.1 The two fatal adversaries, stated explicitly

**A4s — scene-identity conditioning.** The swap arm changes the scene. The
treatment arm never does. Any mechanism whose `Delta`-response magnitude depends
on content class — texture density, dominant orientation, contrast, depth-range
prior, anything — produces `beta_T != beta_S` with no geometric matching
whatsoever. The brief's instruction *"do not assume that a 'different scene' swap
is automatically a mechanistic null"* is exactly right: a different scene is not
a null, it is **a second treatment applied to a different confounded variable**.

**A5r — readout pinning.** This one is worse, because it does not merely permit
the positive result, it *predicts* it. Under any `H_ALT`, the aggregated cost
profile on non-corresponding content is unstructured; the soft-argmin over 12
candidates then returns a value near the profile-weighted mean with no systematic
`Delta` trend, i.e. `beta_S ≈ 0`. The experiment would report "the response
vanishes when correspondence is destroyed" when what actually happened is "the
readout has nothing to move and therefore does not move". No control in the
design separates these, and none can be built from the swap arm, because the flat
profile is a *consequence* of destroying correspondence — the same act that is
supposed to be the manipulation.

### B.2 Why Gate-C does not rescue `S*`

Gate-C kills A1–A3 (they cannot deliver a calibrated unit slope without becoming
offset estimators) but it does not touch A5r, and it survives A4s only in mean.
Worse, Gate-C relocates the entire evidential weight onto `beta_T ≈ +1`. By the
`ZI-2` identity (A.4), `beta_T` **is** the global right-image translation slope.
`DESIGN.md` section 4 rejects that statistic with a specific stated reason:

> *"only calibration separates, and nothing forbids coincidence."*

So Gate-C is the rejected calibration argument, re-adopted verbatim and measured
over a sub-region.

**And the swap does not address the stated ground of that rejection.** The
adversary the campaign named is a mechanism fitted to, and valid on, corresponding
natural stereo pairs, which reproduces sign, monotonicity and possibly gain by
coincidence. Feeding such a mechanism non-corresponding content puts it far
outside the distribution it was fitted on; its `Delta`-response is then
unconstrained and generically collapses to zero. **`beta_S ≈ 0` is predicted by
the named `H_ALT` adversary exactly as strongly as by `H_CORR`.** A control with
equal likelihood under both hypotheses has zero discriminating power.

Therefore:

    S* = beta_T - beta_S
       = (rejected statistic) - (a term with zero likelihood ratio)

which carries no more evidence than `beta_T` alone, and `beta_T` alone is
rejected. **`S*` is REJECTED.**

---

## C. The required null property, and an impossibility argument

Not "a different scene". Write the requirement out properly. For the slope
contrast to be attributable to geometry, the null must satisfy all of:

* **N1** identical left crop;
* **N2** identical `Delta` group action on the right content;
* **N3** identical zone location and identical candidate-axis exposure;
* **N4** **identical local right-image content inside the tested receptive
  window**, so that no content-conditional response (A1–A4s) can differ;
* **N5** a different geometric relationship between that right content and the
  tested left content.

**Claim.** For a fixed left crop, **N4 and N5 cannot both hold.**

*Proof.* The geometric relationship between the tested left content and the
tested right content is a function of the pair `(L|_W, R|_W)` restricted to the
tested window `W`. N1 fixes `L|_W`. N4 fixes `R|_W`. A function of two fixed
arguments has one value, so the geometric relationship is the same in both arms,
contradicting N5. ∎

The only available relaxation is to demand N4 **as a multiset over the `Delta`
sweep** rather than pointwise: let the null present the *same seven right crops*
as the treatment, assigned to different `Delta` labels (the permutation null,
N6 below). That satisfies content-matching exactly. But then the statistic
reduces to *"is the output monotone in crop index?"*, which is the global
translation slope again, and the `ZI-2` collapse applies unchanged.

So the null design space is bounded by a genuine dilemma:

* **content-matched nulls** re-derive the rejected global-translation statistic;
* **content-different nulls** admit A4s and A5r.

There is no third option for a fixed left crop.

### C.1 Same-frame counterfactual patches under the 96 px slack

The brief asks specifically whether a counterfactual right patch can be built
from the same frame using the available slack. It cannot, and the reason is
structural rather than a shortage of pixels:

Take the zone's right content from the same frame at a horizontal displacement
`H`: `R[y, x] = F_R[96+y, x + 16*Delta + H]`. The content at tested column `x` is
then the scene point whose left-image column is `x + 16*Delta + H + d`, so its
apparent disparity against the fixed left crop is `d + H/16 + Delta` candidates.

* `H = 0` — the treatment.
* `H` small enough to keep `d + H/16 + Delta` inside `[0, 11]` — this is **still a
  valid correspondence**, merely at a different disparity. `H_CORR` predicts
  `beta = +1` here, identically to the treatment. Useless as a null.
* `H` large enough to leave the window (`|H| >= ~100 px`) — **no candidate can
  match, so the readout saturates or pins**, and `beta ≈ 0` follows from A5r
  rather than from the absence of geometry. Useless as a null, for the same
  reason the different-scene swap is.

A vertical displacement is prohibited on record (`DESIGN.md` section 4: the
measured response is *larger* vertically, `+0.0048` versus `-0.0017`; and
`FH = 17` cells makes a halo-free vertical band empty at any feasible crop), and
it additionally invites A8 through vertically extended structure.

The 96 px of slack is therefore sufficient for the *treatment* and useless for
the *null*. The constraint is not data volume; it is that every same-frame
counterfactual is either still-corresponding or out-of-window.

---

## D. Candidate paired nulls

Every candidate below preserves N1, N2, N3 by construction; the audit is
therefore about N4/N5 and about which adversaries survive. **None was executed.**

### N1c — different-scene swap (the proposed `S*`)

* **Intervention** `R_T^Delta` as in A.2. **Null** `R_S^Delta`, donor scene.
* **`H_CORR`** `beta_T = +1`, `beta_S ≈ 0`, `S* ≈ +1`.
* **Strongest `H_ALT`** A5r supplies `beta_S ≈ 0` from readout pinning; A4s
  supplies a `beta_T != beta_S` gap from scene-identity conditioning; under
  Gate-C the burden falls entirely on `beta_T ≈ +1`, which is the rejected
  global-translation calibration.
* **Nuisance** wholesale change of right-image content between arms.
* **Statistic invariant to it?** **No.** Slope-differencing removes constant
  level differences between arms; it does not remove differences in
  `Delta`-sensitivity, which is what A1–A4s are.
* **Adversary can reproduce the sign?** Yes.
* **Closes the adversary set?** No.
* **Verdict: REJECT.**

### N2c — vertical displacement within the same frame

* Null content `F_R[96 + y + v, x + 16*Delta]`.
* **`H_CORR`** `beta ≈ 0`. **`H_ALT`** unchanged response; plus A8 through
  vertically extended structure (poles, buildings, road texture) giving spurious
  valid matches.
* Prohibited on record; content class shifts systematically with `v` in a
  horizontally stratified scene; the recorded vertical response is *larger* than
  the horizontal one, which is the opposite of what a null requires.
* **Verdict: REJECT** (also prohibited).

### N3c — large horizontal displacement within the same frame

* Null content `F_R[96 + y, x + 16*Delta + H]`, `|H|` large.
* Satisfies N4 in *content class* (same scene, same row, same statistics family)
  better than any other candidate.
* **`H_CORR`** `beta ≈ 0` by saturation. **`H_ALT`** A5r gives the same, for
  readout reasons. The two hypotheses make the **same** prediction.
* **Verdict: REJECT** — a null whose predicted value is identical under both
  hypotheses has no discriminating power, by construction.

### N4c — phase-randomised / spectrum-matched synthetic patch

* Matches amplitude spectrum, destroys phase, then translates by `16*Delta`.
* Matches low-order marginals; destroys edge structure entirely, so the extractor
  operates far off-distribution and A5r dominates. Also degrades the arm from a
  natural-image intervention to a synthetic one, weakening the Type-A property
  the whole design rests on.
* **Verdict: REJECT.**

### N5c — real donor patch with matched low-order statistics

* A donor patch selected or histogram/spectrum-matched to the true patch.
* Matches the statistics *the designer chose to match*, not the statistics the
  mechanism actually uses. A4s survives (scene identity still differs), A5r
  survives (joint structure with `L` is still destroyed), A8 survives.
* **Verdict: REJECT.**

### N6c — `Delta`-permutation relabelling (same input multiset)

* Null presents the identical seven treatment crops under a permuted
  `Delta` labelling `pi(Delta)`.
* **The only candidate that satisfies N4 exactly** — every marginal, every joint,
  every detection statistic, every content statistic is identical between arms as
  a set, because the inputs *are* the same set.
* But the statistic then asks only whether the output is monotone in crop index.
  Any mechanism monotone in crop index — including the weak global response the
  degenerate model already measurably has — yields `beta_T != beta_perm`. This is
  the global-translation monotonicity test, and `ZI-2` applies.
* **Verdict: REJECT.**

### N7c — patch permutation / block shuffle inside the zone

* Destroys detection statistics as well as localisation; non-natural input;
  A5r dominates.
* **Verdict: REJECT.**

### N8c — epipolar patch from another location in the same frame

* A relabelling of N3c with the same in-window/out-of-window dilemma.
* **Verdict: REJECT.**

| Candidate | Satisfies N4? | Satisfies N5? | Fatal adversary | Verdict |
|---|---|---|---|---|
| N1c different-scene swap | No | Yes | A4s, A5r | **REJECT** |
| N2c vertical displacement | No | Partly | A8, prohibited on record | **REJECT** |
| N3c large horizontal displacement | Nearly | Yes | A5r; same prediction under both hypotheses | **REJECT** |
| N4c phase-randomised patch | Marginals only | Yes | A5r | **REJECT** |
| N5c statistics-matched donor | Marginals only | Yes | A4s, A5r, A8 | **REJECT** |
| N6c `Delta`-permutation | **Yes (as multiset)** | Yes | collapses to global monotonicity; `ZI-2` | **REJECT** |
| N7c block shuffle | No | Yes | A5r | **REJECT** |
| N8c displaced epipolar patch | Nearly | Yes | A5r | **REJECT** |

---

## E. The identification argument, stated in full

Write `f` for the map from the tested window's content to `disparity_initial` at
a tested pixel, `L_W` for the fixed left content in that window, `R_W` for the
right content, and `T_t` for translation by `t` candidates. The treatment sweep
evaluates `f(L_W, T_Delta R_W)`; the null sweep evaluates `f(L_W, T_Delta R'_W)`.

**What a valid design would need.** If one could establish

    d/dDelta f(L_W, T_Delta R_W) = +1   for all Delta in an extended range,
                                        whenever R_W corresponds to L_W
    d/dDelta f(L_W, T_Delta R'_W) = 0   whenever R'_W does not

**with `R_W` and `R'_W` otherwise indistinguishable**, then integrating the first
line gives `f(L_W, T_d R_W) = d + c(L_W)`: `f` returns the offset up to a term
independent of the offset, i.e. `f` **is** a local offset estimator, hence inside
`H_CORR`. That is a clean and closing argument, and it is the argument `S*` is
reaching for.

**Why it does not apply.** It requires the clause *"with `R_W` and `R'_W`
otherwise indistinguishable"* — precisely property N4 — and section C proves N4
and N5 are jointly unsatisfiable for a fixed left crop. Without N4, the two
derivatives are taken at two different points of content space, and their
difference is attributable to the content change (A1–A4s) or to the readout's
behaviour at an unstructured operating point (A5r) with no appeal to geometry.

**And the residual weight falls back on the rejected statistic.** Because
`ZI-2` holds, `d/dDelta f(L_W, T_Delta R_W)` is identically the global
right-image translation slope restricted to the tested interior. Any design in
this family therefore inherits the rejection recorded in `DESIGN.md` section 4,
and the null cannot lift it, because — as section B.2 shows — the null's
predicted value is the same under the named `H_ALT` adversary as under `H_CORR`.

That is the whole audit in three lines:

1. the treatment term is the rejected statistic (`ZI-2`, measured);
2. the only null that would repair it is content-matched, and content-matched
   nulls are either impossible (section C) or re-derive the rejected statistic
   (N6c);
3. every content-different null admits A4s and A5r.

---

## F. Compliance

Not done, and not proposed: no trained checkpoint opened, no training, no
treatment inference, no swap inference, no threshold derivation, no EPE/D1
argument, no revival of `S`, no O1 or permutation-of-candidates variant, no
translated self-pairs, no global translation as primary evidence, no random
weights presented as a mechanistic null, no `S8`, no `max`-over-unit gate, no
further receptive-field boundary search, no direct edits to `V`, and no claim
that tracking `Delta` by itself demonstrates correspondence.

Corrections carried forward for any future record, both already established and
neither requiring new measurement:

* `H_CORR` predicts `beta = +1` under the amended one-sided
  `x_R = x_L + 16*Delta`; `DESIGN.md` sections 5.3 and 7 still say `-1`.
* Bit-exact tested interiors are `x` in `[0, 224)` and `x` in `[912, 1136)`,
  not `[0, 250)` and `[886, 1136)`.

---

## G. Decision

**2 — DESIGN NOT IDENTIFIABLE.**

`S*` is rejected, and so is every paired null examined: for a fixed left crop,
every feasible null either re-derives the global-translation statistic that this
campaign has already rejected, or leaves a correspondence-free explanation
standing in the form of scene-identity conditioning (A4s) or readout pinning
under an unstructured cost profile (A5r).

**Scope of the verdict, stated so it is not over-read.** This audit closes the
family of designs whose treatment arm is a **translation of the right crop**,
read at a **bounded-receptive-field stage**, referenced against a **paired
null**. By the `ZI-2` identity that family includes every zone-local variant, and
by section C it includes every paired null constructible from the existing KITTI
frames and the 96 px slack.

It does **not** establish that no pipeline-level intervention of any kind exists.
One family was neither audited here nor executed nor endorsed: an intervention
that imposes a **spatially varying** disparity field rather than a translation —
for example by resynthesising the right image from the left image and a modified
ground-truth disparity field. It is named only so that the verdict above is not
mistaken for a stronger claim than it is. It would introduce resampling
artefacts, occlusion holes and synthetic content, none of which is audited here,
and nothing in this document should be read as recommending it.

**Ceiling: Level D — genuine geometric correspondence NOT-DEMONSTRATED.**
Unchanged. No result in this audit raises or lowers it.
