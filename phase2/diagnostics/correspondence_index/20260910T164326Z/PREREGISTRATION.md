# EXP-CORRESPONDENCE-INDEX-001 — PRE-REGISTRATION

**Frozen 2026-09-10, before any arm was executed.**
Record: `phase2/diagnostics/correspondence_index/20260910T164326Z/`.

Inference only. No training, no optimizer, no weight change, no architecture
change, no image change, no feature change. The closed block-count campaign
stays closed. This experiment targets **candidate-coordinate sensitivity (level
C)** and nothing above it.

---

## 1. QUESTION

> Does the frozen aggregation/readout path track the **coordinate identity** of
> candidate slices?

Intervention on the already-built cost volume `V ∈ R^(B,C,D,H,W)`, `D=12`,
candidate axis `dim=2`:

```
V'(k) = V(π(k))
```

Images and features are untouched in every arm.

## 2. TENSOR LOCATION

Build the cost volume **once** per (checkpoint, scene) at
`volume = model.cost_volume(left_features, right_features)`, shape
`(1, 32, 12, 23, 77)`. Cache it. Every arm is constructed from that same cached
tensor. Nothing upstream is altered; feature extraction is never repeated.

Then per arm: `permuted volume → model.aggregation → model.regression` and read
**`disparity_initial`**. **Refinement is never invoked.** `disparity_final` is
not measured and is not an endpoint.

## 3. PERMUTATIONS — frozen here

Indices `0…11`. `V'(k) = V(π(k))`, so `πm(k) = (k−m) mod 12` moves content
originally at candidate `j` to `j+m mod 12`: **positive m moves content toward
larger candidate indices.**

| arm | π as a literal 12-element index list |
| --- | --- |
| identity (m=0) | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` |
| m = +1 | `[11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` |
| m = +2 | `[10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` |
| m = −1 | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]` |
| **random (frozen)** | **`[1, 10, 9, 5, 3, 8, 11, 0, 6, 7, 2, 4]`** |

The random permutation was generated **once**, before execution, with
`numpy.random.default_rng(20260910).permutation(12)`, and is written literally
above. It has **no fixed points**. It is not regenerated per scene or per
checkpoint.

## 4. SAME INDEXING OPERATION FOR ALL ARMS — mandatory

Every arm, **including identity**, is built with

```
volume.index_select(dim=2, index=<perm tensor>)
```

The original tensor is **never** passed through directly for identity.
`build_cost_volume` returns a permuted *view* with non-contiguous strides;
`index_select` materialises a contiguous tensor. Mixing the two across arms could
select a different cuDNN algorithm and change the last bits, which would corrupt
the gate for a purely numerical reason. Routing every arm through the identical
op holds layout constant so that the only difference between arms is the value
assignment.

Recorded per arm: shape, dtype, device, contiguity, and
`max_abs_diff(identity-indexed volume, original volume)` — **expected exactly 0**.

Deterministic campaign controls: `torch.use_deterministic_algorithms(True)`,
`cudnn.deterministic=True`, `cudnn.benchmark=False`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `no_grad`.

## 5. NEGATIVE-CONTROL GATE — first and gating

Checkpoint: the trained `shift="none" + standardised` model
(`phase2/factorial/shift_none_standardized/20260909T071500Z/checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A_checkpoint.pth`,
200 epochs, converged).

Its volume is degenerate — `V(k) = D` for every k (`reference_shift` is a no-op;
re-measured at 0.0 max inter-slice difference). Therefore for **any** π,
`V[:,:,π] = V` element-wise. This is an **algebraic identity, not an empirical
expectation.**

Run all five arms. Required, for every arm against identity:

```
max_abs_diff(volume_identity, volume_permuted)                   == 0
max_abs_diff(disparity_initial_identity, disparity_initial_perm) == 0
```

**Any non-zero difference ⇒ STOP IMMEDIATELY.** Do not run positive
checkpoints. Report as an implementation / determinism / layout failure and
**do not interpret it scientifically.**

## 6. POSITIVE CHECKPOINTS

Only after the gate passes: 6-block seeds 0, 1, 2 — the authoritative
deterministic checkpoints from the closed campaign, loaded read-only, unretrained,
preprocessing unaltered.

## 7. INTERIOR MASK — geometry-derived, frozen

Retain pixels with valid GT and

```
GT / 16 ∈ [2, 8]        i.e.  32 px ≤ GT ≤ 128 px
```

Defined from **ground truth, never from the model's own prediction**, so the
identical pixel set applies to every checkpoint and every arm. It keeps candidate
content away from the extreme indices for |m| ≤ 2. **No spatial border mask is
needed — no image is ever translated.** The mask is not modified after results
are seen.

## 8. PRIMARY PREDICTION — no unit slope

`Aggregation` is five disparity-axis 3-tap convolutions with zero padding, so
output candidate k depends on input candidates [k−5, k+5] and only k ∈ {5,6} are
padding-free. **The stack is not cyclically equivariant and `Δd = m` is not
predicted.**

Pre-registered qualitative prediction, over interior-masked pixels, in the
aggregate/median sense:

```
d(+2) > d(+1) > d(0) > d(−1)
```

with a non-zero response and, where signal is retained, a larger response at +2
than at +1. **No numerical slope threshold is preregistered and none may be
invented afterwards.** Failure to obtain unit slope is *not* evidence against
candidate-coordinate sensitivity.

## 9. RANDOM-PERMUTATION CONTROL

Order-destruction control only. **No numerical expected output is preregistered**
— the aggregation smooths across an 11-candidate support, so a scrambled profile
is smeared and `π⁻¹(d)` is not recoverable. Interpreted qualitatively: ordered
arms should give coherent directional responses; the scramble should not
reproduce that ordered pattern. If ordered and random are indistinguishable in
both magnitude and spatial structure, candidate-coordinate tracking is **not**
established.

## 10. SUCCESS CRITERIA (level C only)

1. negative control algebraically invariant;
2. identity exactly reproducible;
3. +1 gives a systematic positive directional response;
4. +2 same direction, generally larger than +1;
5. −1 gives the opposite direction where candidate support permits;
6. random does not reproduce the ordered directional pattern.

Magnitudes are descriptive. No unit-slope claim.

## 11. CLAIM LEVEL

A. right-image dependence — already established, known confounds.
B. candidate dependence — already established; Stage A showed the padding
artefact alone produces it.
**C. candidate-coordinate sensitivity — what this experiment targets.**
D. geometric correspondence — **not** established here.
E. genuine disparity search — **not** established here.

The words *correspondence*, *matching* and *disparity search* must not be used as
synonyms for a positive result.

## 12. REFERENCE-FRAME ISSUE — recorded, not corrected

The source audit found the cost volume stores slice k =
`left_feat[u+k] − right_feat[u]` at output column u (right-referenced) while the
visualiser convention, GT and refinement guidance are left-referenced. It is
**preserved, not corrected**. It does not invalidate this intervention, which
acts directly on candidate indices. Candidate indices are **not** reinterpreted
as physical disparity anywhere in this experiment.

## 13. COMPUTE

Per (checkpoint, scene): one feature-extraction + volume build, cached; then five
aggregation+readout arms. 4 checkpoints × 4 scenes = 16 volume builds, 80
aggregation/readout arms. No refinement, no training. Scenes: the existing
`FOCUS_SCENES = [27, 0, 31, 6]`, split `hailo_val`. No extra scenes unless a
failed gate must be diagnosed. **If the gate fails, positive-control compute is
zero.**

## 14. OUTPUT

`PREREGISTRATION.md`, `RESULTS.md`, `results.json`, `ENVIRONMENT.txt`, execution
log, and the harness source, all under this directory only. Records the frozen
random permutation, checkpoint hashes, scene IDs and valid-pixel counts. No
historical record is modified.

## 15. HARD STOP

The experiment ends when candidate-coordinate sensitivity is determined. No
synthetic-patch, natural-image correspondence, candidate-count, dilation,
architecture, block-count, Scene Flow, pretrained or O6 work follows. Any attempt
to establish geometric correspondence requires a separate pre-registration.
