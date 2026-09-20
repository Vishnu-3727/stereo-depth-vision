# EXP-CORRESPONDENCE-INDEX-001 — RESULTS

Record `phase2/diagnostics/correspondence_index/20260910T164326Z/`.
Pre-registered in `PREREGISTRATION.md`, frozen before any arm ran.
Inference only — 16 volume builds, 80 aggregation/readout arms, 7.5 s total.
No training, no optimizer, no weight or architecture change, no image or feature
modification, refinement never invoked.

```
VERDICT: CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED
```

Driven by success criterion **6** (§12 of the task specification): the random
permutation *did* reproduce the ordered directional pattern. Criteria 1–5 all
passed.

---

## NEGATIVE-CONTROL GATE — PASS

The `shift="none"` volume satisfies `V(k) = D` for every k, so `V[:,:,π] = V`
element-wise for any π. This is an algebraic identity, and the measurement
confirms it exactly:

| arm | max abs diff, volume vs identity | max abs diff, `disparity_initial` vs identity |
|---|---:|---:|
| −1 | **0.0** | **0.0** |
| identity | **0.0** | **0.0** |
| +1 | **0.0** | **0.0** |
| +2 | **0.0** | **0.0** |
| random | **0.0** | **0.0** |

`index_select`-identity vs the original tensor: **max abs diff 0.0**. The
mandatory same-indexing-operation requirement worked — no layout or algorithm
artefact appeared. Pooled `disparity_initial` was **6.4822** in every arm,
bit-identical.

This is the property no image-perturbation diagnostic ever had: the degenerate
model cannot respond, by algebra rather than by expectation.

---

## CHECKPOINT-LEVEL COMPARISON

Pooled over the four focus scenes; **median** `disparity_initial` delta vs
identity, same statistic throughout.

| checkpoint | −1 | identity | +1 | +2 | random |
|---|---:|---:|---:|---:|---:|
| NEG_shift_none | +0.0000 | 0.0000 | +0.0000 | +0.0000 | +0.0000 |
| 6b_seed0 | **−1.7788** | 0.0000 | **+1.9338** | **+2.8208** | +2.3269 |
| 6b_seed1 | **−1.2219** | 0.0000 | **+1.1251** | **+1.9217** | +1.5548 |
| 6b_seed2 | **−1.4928** | 0.0000 | **+1.4455** | **+2.4833** | +2.0561 |

Identity absolute medians: NEG 6.5365, seed0 5.5358, seed1 4.1448, seed2 4.6954.

Per-scene rows are in `results.json` and `positive_checkpoints.json`; retained
pixel counts are 33,026 / 42,676 / 37,892 / 62,908 for scenes 000187_10,
000160_10, 000191_10, 000166_10 — **identical for every checkpoint and every
arm**, because the mask comes from ground truth alone.

---

## ORDERED PERMUTATION RESPONSE — criteria 3, 4, 5 PASS

For all three positive checkpoints, on every scene:

```
d(+2) > d(+1) > d(identity) > d(−1)
```

- **+1 positive** on 3/3 checkpoints and 12/12 scene-arm pairs.
- **−1 negative** on 3/3 and 12/12.
- **+2 larger than +1** on 3/3 and 12/12.

The direction is exactly as pre-registered: `π_m(k) = (k−m) mod 12` moves content
toward larger candidate indices for positive m, and `disparity_initial` follows.
The response is large — 1.1 to 2.8 candidates — against a negative control that
is bit-exactly zero.

**No unit slope is claimed.** As pre-registered, `Aggregation` is five
disparity-axis 3-tap convolutions with zero padding, so output candidate k
depends on inputs [k−5, k+5] and only k ∈ {5,6} are padding-free; the stack is
not cyclically equivariant and `Δd = m` was never predicted. Observed magnitudes
(≈ +1.1…+1.9 at m=+1, ≈ +1.9…+2.8 at m=+2) are reported as descriptive.

---

## RANDOM PERMUTATION CONTROL — criterion 6 FAILS

Frozen permutation `π = [1, 10, 9, 5, 3, 8, 11, 0, 6, 7, 2, 4]`, generated once
before execution, no fixed points.

The scramble produced **+2.3269 / +1.5548 / +2.0561** — the same sign as the
ordered arms and a magnitude sitting **between +1 and +2** for every checkpoint.
It is not distinguishable from the ordered response by magnitude.

Per §12 criterion 6 and §13 of the specification, this blocks the claim: the
intervention may be measuring generic candidate-axis disturbance rather than
ordered candidate tracking. **Level C is not claimed.**

---

## POST-HOC OBSERVATION — flagged, NOT used to change the verdict

Recorded because it materially affects the next design, and labelled as what it
is: reasoning performed **after** seeing the result.

`π⁻¹ = [7, 0, 10, 4, 11, 3, 8, 9, 5, 2, 1, 6]`. A perfect position-tracking
readout under `V'(k) = V(π(k))` would move a signature at candidate d to
`π⁻¹(d)`. Over the retained GT band d ∈ [2,8]:

```
π⁻¹(d) for d = 2..8  =  [10, 4, 11, 3, 8, 9, 5]
mean π⁻¹ over band   =  7.1429
mean d over band     =  5.0000
expected delta       =  +2.1429
observed mean delta  =  +1.9793
```

**The frozen random permutation was a badly chosen control.** It is not
directionally neutral over the retained band — it maps mid-range candidates to
high indices, so a genuine position-tracker is *expected* to produce a large
positive shift under it. Its agreement with the ordered arms is therefore
uninformative about ordered-vs-unordered tracking.

This is a **design flaw in my own pre-registration**, not a result. It is
reported, not acted upon. Re-scoring the experiment against a differently chosen
control would be exactly the goalpost move the protocol forbids. **Any attempt to
resolve level C requires a new pre-registration** specifying a control whose
`π⁻¹` is directionally neutral over the retained band (for example, a permutation
constrained to have `mean π⁻¹(d) ≈ mean d` across the mask), together with the
spatial-structure comparison listed below.

---

## CLAIM CEILING

**MEASURED.** The degenerate `shift="none"` readout is bit-exactly invariant
under every candidate permutation tested (0.0 on both volume and
`disparity_initial`). All three 6-block checkpoints show a large, consistently
ordered `d(+2) > d(+1) > d(0) > d(−1)` response on 12/12 scene-arm pairs. The
frozen scramble produced a response of the same sign and comparable magnitude.

**DERIVED.** Criteria 1–5 pass; criterion 6 fails; therefore the pre-registered
verdict is NOT-DEMONSTRATED.

**INFERRED, post-hoc.** The scramble's `π⁻¹` carries a +2.14 directional bias
over the retained band, which is close to the observed +1.98 — so the control
cannot separate the two hypotheses it was meant to separate.

**UNKNOWN.** Whether the aggregation/readout path is candidate-coordinate
sensitive. **This experiment did not settle it.**

Not established, and not claimed: **D geometric correspondence**, **E genuine
disparity search**. The words *correspondence*, *matching* and *disparity search*
are not used as synonyms for any result here. Candidate indices are **not**
reinterpreted as physical disparity. Nothing is claimed about `disparity_final`,
the refinement path, sub-candidate precision, occluded/textureless/repetitive
regions, any block count, parameter-count or receptive-field causality, latency,
transfer, or optimal architecture.

---

## REFERENCE-FRAME NOTE — recorded, not corrected

The cost volume stores slice k = `left_feat[u+k] − right_feat[u]` at output
column u (right-referenced), while GT, `viz/core.correspondence` and the
refinement guidance are left-referenced. **Preserved as found.** It does not
affect this intervention, which acts directly on candidate indices and never
converts them to physical disparity.

---

## PROVENANCE

`HEAD` `58e8a19`; `phase-1-frozen^{commit}` `b4207e5`;
`git diff phase-1-frozen -- src scripts` **empty**; working tree `M .gitignore`,
`?? phase2/`. All output confined to this directory; no historical record
modified; checkpoints opened read-only. Phase 2 remains untracked in git — no
cryptographic versioning is claimed.

The closed block-count campaign was not reopened, no image-translation or
vertical-shift arm was run, no fill values were introduced, no mask was altered
after results were seen, and no threshold was tuned post hoc.
