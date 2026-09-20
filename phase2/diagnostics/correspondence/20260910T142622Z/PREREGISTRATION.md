# EXP-CORRESPONDENCE-SHIFT-001 — PRE-REGISTRATION

**Frozen 2026-09-10, before any sweep was run.**
Record: `phase2/diagnostics/correspondence/20260910T142622Z/`.

Inference only. No training. No architecture change. No modification of any
historical record. The block-count campaign is CLOSED and is not reopened.

---

## 1. QUESTION

> Does `disparity_initial` respond to controlled horizontal translation of the
> right image with the geometrically predicted candidate shift?

Scored on **`disparity_initial`** (units: disparity candidates 0…11), never on
`disparity_final`.

## 2. WHY NOT THE EXISTING PROBES

`disparity_final = disparity_initial + refinement(disparity_initial, left)`
(`src/models/stereonet/stereonet.py:107-109`). The refinement is guided by the
**left image only** and supplies most of the final magnitude, since
`disparity_initial ≤ 11` while GT disparities reach ~192 px. The existing probes
score `disparity_final` under right-image *corruption*, which establishes
binocular dependence and nothing about correspondence. EXP-010 already showed a
provably degenerate cost volume coexisting with large downstream right-image
dependence.

## 3. GEOMETRY — traced from source, not assumed

`src/models/stereonet/cost_volume.py`:

- `shift_left(x, k)` = `F.pad(x, (0, k))[..., k:]` → `shifted[u] = left[u+k]`
- `build_cost_volume` with `shift="left"` → slice *k* = `left_feat[u+k] − right_feat[u]`

Define the horizontal perturbation as moving right-image **content** right by Δ
full-resolution pixels: `right'[y, x] = right[y, x−Δ]`, zero-fill. Feature stride
is 16, so `right'_feat[u] ≈ right_feat[u−δ]`, `δ = Δ/16`.

Baseline match for right-column *v*: `left_feat[v+d] ≈ right_feat[v]`.
Substituting `v = u−δ`: `left_feat[u−δ+d] ≈ right'_feat[u]`, so the matching
slice satisfies `u+k = u−δ+d`:

```
k = d − Δ/16          →  Δk = −Δ/16
expected slope = −1 candidate per 16 full-resolution pixels = −0.0625 cand/px
```

## 4. CONDITIONS

Left image **fixed** in every condition. Only the right image is translated,
zero-filled, by an identical code path for both axes.

- Horizontal: `Δx ∈ {−32, −16, 0, +16, +32}` px
- Vertical null: `Δy ∈ {−32, −16, +16, +32}` px

Nine conditions. The vertical shift is a nuisance control with comparable image
displacement and no horizontal-disparity information — **not** a second
correspondence task.

## 5. ANALYTICAL ANCHORS

**Zero anchor.** For `shift="none"` all 12 slices are identical
(`cost_volume.py:45-55`, EXP-010). `standardise_across_disparity` then returns
`(c−c)/(0+1e-5) = 0` for every candidate → uniform softmax → soft-argmin =
`(0+1+…+11)/12 = 5.5`, independent of input. Expected response to any right-image
translation: **exactly zero**.

**Correspondence anchor.** `−1/16 = −0.0625` candidate per pixel.

Measured horizontal slope is reported as a fraction of the correspondence anchor.

## 6. EXECUTION ORDER — BINDING

**Stage A — falsification first.** Run the complete sweep on the trained
`shift="none" + standardised` negative control
(`phase2/factorial/shift_none_standardized/20260909T071500Z/checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A_checkpoint.pth`,
200 epochs, converged, 40-scene EPE 8.3717). Positive-control results are **not
inspected** until Stage A is recorded.

If the negative control shows a systematic non-zero horizontal slope: **STOP.**
Do not run or interpret the positive controls. Investigate preprocessing,
padding, interpolation, normalisation, borders. The diagnostic is invalid unless
the known-degenerate model behaves as predicted.

**Stage B — matched vertical null**, same run, same code path.

**Stage C — positive checkpoints**: 6 blocks, seeds 0, 1, 2, existing
authoritative checkpoints, not retrained.

## 7. SCENES

`FOCUS_SCENES = [27, 0, 31, 6]` from `phase2/scripts/exp_h2_seed_replication.py`,
split `hailo_val`. The 10-scene monitor subset is **not** used.

## 8. MASK — frozen before any positive result

Applied identically in every condition and for every checkpoint:

1. **Candidate-extreme exclusion:** baseline (Δ=0) `disparity_initial` must lie
   in the open interior, `1.0 ≤ d₀ ≤ 10.0`. Candidates are 0…11 by construction,
   so a pixel pinned at an end cannot move in one direction.
2. **Border exclusion:** drop a **32-pixel** margin on all four edges — the
   maximum translation, applied on every side because both axes are swept.

The mask is computed from each checkpoint's own Δ=0 baseline and is then held
fixed across that checkpoint's nine conditions. The **definition** is identical
for every checkpoint; the realised pixel set is necessarily checkpoint-dependent
because it is defined on the model's own baseline. Retained-pixel counts are
reported per checkpoint so any difference is visible. The mask is never selected
by which pixels produce the strongest slope.

## 9. MEASUREMENT

Per condition, over retained pixels: mean and median of `disparity_initial`.
Response = summary(shifted) − summary(baseline). Slopes fitted by ordinary least
squares on the five horizontal points (including Δ=0) and separately on the
vertical points plus the shared Δ=0 baseline. Both mean-based and median-based
slopes are reported; the **mean-based horizontal slope is primary**, fixed here.

No classification threshold is invented after seeing results.

## 10. VERDICT LOGIC — pre-registered, pattern-based

- **Pattern A — no correspondence evidence.** Horizontal slope approximately
  matches the degenerate control, especially if the vertical response is
  comparable. → No evidence of geometric disparity search.
- **Pattern B — right-image sensitivity with horizontal geometry.** Correct
  negative sign and monotonicity, magnitude substantially different from
  −1/16. → Correspondence-shaped response; geometrically calibrated candidate
  selection **not** established.
- **Pattern C — correspondence-consistent candidate search.** Negative sign,
  monotone in Δ, substantially distinct from the vertical null, consistent with
  the −1/16 anchor, and absent in the degenerate control. → Evidence of disparity
  search at candidate resolution. Claim nothing beyond this.

## 11. REFERENCE-FRAME CHECK

`viz/core.correspondence` states the left-referenced convention
`x_right = x_left − d`. `build_cost_volume` stores slice *k* =
`left_feat[u+k] − right_feat[u]` **at output column u**, i.e. indexed by the
right image's column, while GT and the refinement guidance are left-referenced.
This is traced, documented and **preserved, not corrected**. The derivation in §3
shows the matching slice is `k = d − Δ/16` regardless of storage frame, so an
offset of this kind affects the **intercept**, not the **slope**. Reported, not
silently fixed.

## 12. HARD STOPS

Stop immediately if: the negative control shows a systematic non-zero horizontal
slope; the response is found to depend on the refinement rather than
`disparity_initial`; the vertical null is implemented by a different code path
from the horizontal; the mask definition changes between checkpoints;
preprocessing changes between conditions; the model must be modified; or any
result is thresholded retrospectively.

On failure: report the failure. Do **not** patch and rerun positive controls
without a new pre-registration.

## 13. CLAIM CEILING

If Pattern C: the readout path exhibits correspondence-consistent disparity
search at candidate resolution — and nothing more. Explicitly **not** claimable:
that the final disparity is entirely correspondence-derived; that the refinement
performs correspondence; sub-candidate correspondence; correctness in occlusions,
textureless or repetitive regions; global algorithm correctness; superiority of
any block count; parameter-count or receptive-field causality; latency; transfer;
optimal architecture.

## 14. OUTPUT

`results.json` (machine-readable, per condition and per fit), `RESULTS.md`,
`ENVIRONMENT.txt`, this file, and stdout logs. Written only under this directory.
