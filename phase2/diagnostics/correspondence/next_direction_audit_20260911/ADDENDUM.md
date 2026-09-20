# ADDENDUM — independent second audit pass

**Added 2026-09-11, after `AUDIT.md`, `design.json` and `RELATED_RUNS.md` were
already present in this directory.** Those three files were **not modified**.
This addendum only adds.

**Provenance note.** This addendum was written by a second, independent audit
pass that reached the same top-level decision (`BYPASS-LEVEL-C`) and the same
next experiment (`EXP-CORRESPONDENCE-GEOM-001`, right-image translation, slope
anchor `−1/16` cand/px, vertical comparator, degenerate model as artifact
baseline) **before** reading the existing files. The convergence is independent
and is itself evidence that the design is the right one. What follows is only
what the second pass adds or disputes.

Nothing was executed. No training, no inference, no `StereoNet`/`Aggregation`
instantiated. No historical record modified.

Tags: **MEASURED**, **DERIVED**, **INFERRED**.

---

## A.1 The structural reason Level C was doomed — stated explicitly

`AUDIT.md` records the slice equation at line 404 but does not draw this
consequence. **MEASURED**, `src/models/stereonet/cost_volume.py`:

```
V[:, c, k, y, u]  =  Lf[c, y, u+k]  −  Rf[c, y, u]
                     ^^^^^^^^^^^^^     ^^^^^^^^^^^
                     depends on k      NO k DEPENDENCE
```

**The right image enters the cost volume as a term that is constant along the
candidate axis.** Three consequences (**DERIVED**):

1. Any purely candidate-axis intervention — re-indexing, permutation, orbit —
   permutes a tensor in which the right-image contribution is a `k`-constant
   offset. It therefore cannot isolate the binocular matching pathway; it mostly
   rearranges the *left*-derived term and the padding boundary.
2. `standardise_across_disparity` subtracts the per-pixel mean across `k`, so a
   `k`-constant term is **exactly removed** at the readout. The right image
   survives to the output only through the aggregation's nonlinearity and its
   zero-padding boundary — the weakest available channel.
3. Therefore INDEX-001/002/003 and ARCH-001 were not merely under-powered; they
   intervened on **the one axis along which the variable of interest is
   constant**. No refinement of the statistic could have fixed that.

This is the crispest available justification for `BYPASS-LEVEL-C`, and it is
source-derived rather than outcome-derived.

---

## A.2 Odd/even decomposition — a strengthening of the primary statistic

`AUDIT.md` correctly identifies that the degenerate model's horizontal response
is V-shaped and that the signed slope is the discriminator. It handles the
V-shape by requiring **monotonicity** (C2) alongside the OLS slope (C1).

The second pass proposes going one step further, for a reason Stage A itself
stated (`20260910T142622Z/RESULTS.md`):

> "The near-zero OLS slope is an artefact of fitting a line through a V."

**A plain OLS slope over a symmetric response is exactly the estimator Stage A
warned about.** It returns ≈0 for a V — which happens to be the right answer for
the degenerate model, but only by cancellation, and it discards the information
that would show *why*.

### The decomposition

For a symmetric sweep, split the response into parts that are odd and even in Δ:

```
O(Δ) = [ r(Δ) − r(−Δ) ] / 2          odd   —  where a SIGNED geometric response lives
E(Δ) = [ r(Δ) + r(−Δ) ] / 2          even  —  where a MAGNITUDE artifact lives
```

Under `H1` (correspondence) the prediction is **entirely odd**:
`O_h(Δ) = −Δ/16` candidates, `E_h` unconstrained.
Under a perturbation-magnitude artifact the response is **entirely even**:
`O ≈ 0`, `E` growing with `|Δ|`.

The two hypotheses are orthogonal in this basis. This is not a post-hoc choice:
a signed prediction *is* odd by definition, and Stage A had already preregistered
a signed slope as primary. **Declared limitation:** the second pass selected the
decomposition knowing Stage A's published negative control is even-dominated, so
the *negative* side carries no predictive content. The *positive* side carries
full predictive content — no positive-checkpoint translation number exists in any
record.

### Applied to Stage A's already-published negative-control table

**DERIVED**, arithmetic only, on the five published horizontal and five published
vertical values. No new measurement.

| \|Δ\| | odd_h | even_h | odd_v | even_v |
|---:|---:|---:|---:|---:|
| 16 | **−0.0929** | +0.0946 | +0.1302 | +0.3472 |
| 32 | **−0.0221** | +0.2424 | +0.1256 | +0.4207 |

Correspondence predicts `odd_h(16) = −1.0000` and `odd_h(32) = −2.0000`. The
provably search-free model delivers **9.3 %** and **1.1 %** of the anchor, while
its **even** part grows monotonically with `|Δ|`.

**The artifact is even. Geometry is odd. They separate exactly, and the
separation is already visible in public data.**

### Suggested amendment to the primary statistic

Replace the plain OLS slope on the five raw points with the OLS-through-origin
slope of the **odd part**:

```
S_axis = Σ_Δ O_axis(Δ)·Δ / Σ_Δ Δ²        Δ ∈ {16, 32, …}
F_axis = S_axis / (−1/16)                 fraction of the geometric anchor
```

`F_h = 1` under correspondence, `F_h ≈ 0` under every artifact. The even parts
are reported descriptively — they are the artifact's own signature and their
magnitude tells the reader how hard the intervention pushed.

This is a strict refinement: `S_h` and the existing `β_H` coincide when the
response is purely odd, and `S_h` is immune to the V-shape contamination that
biases `β_H` whenever it is not.

---

## A.3 Two disputes with the frozen decision rule in `design.json`

Raised for the record. **Neither file was modified**; resolution is the operator's
call before any preregistration is finalised.

### A.3.1 `C4` imports a published measurement into a threshold

```json
"C4_artifact_separation":
  "|beta_H - (-0.0625)| < |beta_H - (-0.001713)|, i.e. beta_H < -0.0321"
```

The cut at `−0.0321` is the midpoint between the geometric anchor and the
**measured** degenerate slope `−0.001713`, which is public in Stage A's record.
The project's own standing rule is that previous results must not set the
threshold. A "closer to which anchor" rule is defensible — but half of that rule
is an observed number, so it should be declared as such rather than presented as
a priori.

**Suggested alternative, threshold-free:** require `F_h(unit) >
F_h(NEG, same scene)` — strict dominance over the artifact baseline measured
in-run, with no numeric cut anywhere.

### A.3.2 `≥9/12` partial counts are an arbitrary aggregation level

`C2` and `C3` require the criterion at `≥9/12` positive units. Nine is not
derived from anything. The precedent in this line (INDEX-002 §10.3, ARCH-001 §7)
is **all units or nothing**, adopted precisely to foreclose a post-hoc choice of
aggregation level. `9/12` reintroduces exactly that degree of freedom.

**Suggested alternative:** `12/12`, with the per-unit counts reported so a
partial result is fully visible and can motivate a *fresh* preregistration rather
than being absorbed into this one.

---

## A.4 Two additions to the design that cost nothing

1. **Extend the sweep to `Δ ∈ {−64, −48, −32, −16, 0, +16, +32, +48, +64}`.**
   The odd part needs ≥3 signed points to distinguish a line from noise;
   `±16, ±32` gives only two. Border exclusion rises from 32 px to 64 px; compute
   rises from 144 to 272 forward passes — still well under two minutes. All `Δ`
   remain multiples of 16, which is required: the feature extractor is
   fully convolutional with total stride `2⁴ = 16` (**MEASURED**,
   `feature_extractor.py`: 4 × `Conv2d(5, stride=2, padding=2)`, then
   `ResBlock`s and `Conv2d(3, stride=1)`, no pooling, no normalisation, no global
   op), so a 16·n px image shift produces **exactly** an n-sample feature shift in
   the interior. Non-multiples introduce an interpolation confound outside the
   hypothesis.

2. **Drop the `GT/16 ∈ [2,8]` band from the primary mask.**
   `design.json` keeps it. The provenance audit measured that this band is
   effectively `[2,4]` — 99.4 % of retained pixels — so it narrows the sample for
   no benefit here: the statistic is a *difference* of medians over a fixed pixel
   set, so the band affects magnitude but not structure. A mask of
   `GT > 0 ∧ ≥64 px from every edge` is broader, purely geometric and
   ground-truth-derived, and identical across every condition and model. Keep the
   `[2,8]` band as a preregistered secondary for continuity with INDEX-001/002.

---

## A.5 Two preregistered secondaries worth adding

- **Left-image translation sign-reversal cross-check.** Under `H1` the odd slope
  must flip sign and match magnitude. Under the `k`-constant artifact of §A.1 the
  two axes are structurally *asymmetric* — the left image enters
  candidate-selectively, the right image does not — so this is a genuine extra
  discriminator, not a restatement.
- **The `(L, L)` zero-disparity pair, which has a provable anchor.** For
  `shift="none"` the volume is **exactly zero**, so `Agg(0)` is constant in `k`
  with *no* boundary effect (padding zeros equal signal zeros), standardisation
  returns exactly 0, the softmax is uniform and `disparity_initial = 5.5`
  **exactly**. One of very few provable values anywhere in this project, and a
  free check that the harness is wired correctly. **Caveat:** `(L, L)` is
  out-of-distribution, so only a *positive* result on the trained models would be
  informative; and `right_equals_left` is already public for the Phase-1
  reference model (EXP-007: EPE 47.96, mean disparity 82.79 px).

---

## A.6 What the second pass does **not** dispute

- `BYPASS-LEVEL-C` — agreed, independently.
- `EXP-CORRESPONDENCE-GEOM-001` as the next and only justified experiment —
  agreed, independently.
- The geometry mapping and the `−1/16 cand/px` anchor — independently re-derived
  from `cost_volume.py`, `feature_extractor.py`, `regression.py`,
  `kitti2015.py` and `phase2/viz/core.correspondence` (whose docstring records
  that the sign was confirmed photometrically against KITTI GT, "the opposite
  sign is roughly five times worse on real data").
- The vertical comparator as a matched control that preserves appearance
  statistics rather than destroying structure — agreed; this is what §IX of the
  brief requires and what EXP-007's ablations are not.
- No p-values; descriptive paired comparison; 12 dependent units from 3 weight
  sets × 4 scenes — agreed.
- Inference-only, seconds to minutes, existing evaluator reused — agreed.
- **Nothing may be executed.** `EXP-CORRESPONDENCE-GEOM-001` is not authorised.
