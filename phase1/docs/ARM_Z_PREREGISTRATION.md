# ARM-Z — preregistration: per-pixel L2-normalized feature difference

**Written before any ARM-Z code exists and before any ARM-Z training run.**
Everything below is frozen at registration. Nothing in sections 1–13 may be
edited after results are seen; corrections go in a separate note beside the
record, and any deviation from this document marks the run INVALID rather than
silently changing the protocol.

Status: registered 2026-09-17. Executes the intervention specified in
`phase1/docs/MATCHING_REPRESENTATION_AUDIT.md` sections 6–7, on the ARM-V base
recorded in `phase1/docs/AUDIT_STAGE1_BASELINE_FACTS.md` and
`phase1/scripts/train_arm_v.py` / `phase1/scripts/train_arm_v_seed.py`.

Base model structure cited from `src/models/stereonet/stereonet.py`.

Source-audit provenance: `phase1/docs/MATCHING_REPRESENTATION_AUDIT.md`
(sections 6–9 specify this exact intervention; section 8 lists the resolved
ambiguities; section 9 freezes the verdict rule inherited here).

---

## 0. Why this experiment exists

The matching-representation audit (`phase1/docs/MATCHING_REPRESENTATION_AUDIT.md`)
concludes our cost volume is a raw, unnormalized feature difference — the only
matcher in the comparison set with neither matching-time normalization nor
normalization layers — so a channel's contribution to the cost is proportional
to its activation magnitude, not its informativeness for correspondence.

The identified bottleneck: with no BatchNorm anywhere and an unnormalized
convolution stack, raw channel magnitudes are arbitrary scale artifacts; the
signed difference lets the loudest channel decide the cost, and the 3D
aggregation (same 32 filter weights at every pixel) cannot undo a
magnitude confound that varies per pixel. Every public matcher inspected
normalizes the matching operation or the features feeding it (Fast-ACVNet
`groupwise_correlation_norm` / `norm_correlation`, CoEx `AttentionCostVolume`
L2 normalization, IGEV `norm_correlation`).

ARM-W (group-wise cost, a compression that removes information) and ARM-X
(loss only, forward graph untouched) and ARM-Y (post-volume
disparity-independent gate, refuted at mean 3.4075053) did not test this
mechanism. This intervention is disjoint from ARM-W (no groups, no reduction,
32 channels preserved) and from ARM-Y (disparity-dependent, pre-volume,
gateless, zero parameters).

---

## 1. Hypothesis

`H_Z` — PER-PIXEL L2-NORMALIZED FEATURE DIFFERENCE improves stereo
correspondence in our regime: the ARM-Z 3-seed mean EPE, measured under the
frozen evaluation contract (section 7), will fall below the pre-registered
CONFIRMED threshold of 1.4317132 px (section 8).

One mechanism only. Falsifiable: an ARM-Z 3-seed mean at or above the ARM-V
3-seed mean of 1.7727436 px REFUTES `H_Z` for our regime (section 8).

---

## 2. The intervention, exact and frozen

The single intervention is per-pixel L2 unit-normalization of the left and
right feature maps, independently, before the existing shift-right
subtraction. No other graph change is permitted.

```python
eps = 1e-5
Lhat = L / torch.clamp(L.norm(p=2, dim=1, keepdim=True), min=eps)
Rhat = R / torch.clamp(R.norm(p=2, dim=1, keepdim=True), min=eps)
volume = build_cost_volume(Lhat, Rhat, num_disparities=24,
                           method="subtract", shift="right")
```

Frozen details, each load-bearing:

- L2 norm over the 32-CHANNEL axis ONLY (dim=1 of (B,32,H,W)). Not spatial.
  Not global. Not per-channel over space.
- LEFT and RIGHT normalized INDEPENDENTLY (each from its own norm).
- Normalization happens BEFORE disparity shifting (normalize L and R in
  their own pixel coordinates, then shift Rhat).
- Use max(norm, eps) division, i.e. clamp the norm with
  `torch.clamp(norm, min=1e-5)`. Never raw `v / norm`.
- eps = 1e-5 (the Fast-ACVNet / IGEV value).
- Then build the EXISTING ARM-V cost volume unchanged:
  `cost_k(x) = Lhat(x) - Rhat(x - k)`, k = 0..23.
- Keep signed subtraction. DO NOT negate.
- DO NOT convert to cosine similarity or cosine distance.
- DO NOT reduce, group, or project the 32 channels.
- Cost volume shape stays (B, 32, 24, H, W).
- +0 trainable parameters. No nn.Module, no parameters, no buffers.
- Keep shift = right, regression_normalize = True.

---

## 3. Insertion point, exact and frozen

In `StereoNet.forward` (`src/models/stereonet/stereonet.py`), between these
two existing lines and nowhere else:

```python
left_features = self.feature_extractor(left)
right_features = self.feature_extractor(right)
volume = self.cost_volume(left_features, right_features)
```

becomes:

```python
left_features = self.feature_extractor(left)
right_features = self.feature_extractor(right)
if self.config.feature_normalize:   # ARM-Z, default False
    left_features = normalize(left_features)
    right_features = normalize(right_features)
volume = self.cost_volume(left_features, right_features)
```

`normalize` is a small pure function (no nn.Module, no parameters, no
buffers) with `eps = 1e-5` and the clamp documented in its docstring. It is
NOT placed inside `build_cost_volume` (`src/models/stereonet/cost_volume.py`
untouched), so the subtraction semantics there are untouched. No other
insertion point, no second normalization, no change to any other module
wiring. `regression.py`, `aggregation.py`, `refinement.py`,
`feature_extractor.py` untouched.

---

## 4. Config flag, frozen, part of the intervention

ONE new flag on `StereoNetConfig`, defaulting to False so every existing arm
stays bit-identical:

```python
feature_normalize: bool = False   # ARM Z
```

Default False keeps ARM-V (and every earlier arm) exactly reproducible:
`feature_normalize=False` reproduces the ARM-V volume bitwise. No other
default changes.

---

## 5. Parameter budget

- Expected new parameters: 0.
- Expected total: 397,954 (identical to ARM-V at
  `StereoNetConfig(downsample_levels=3, num_disparities=24,
  cost_volume_shift="right", regression_normalize=True)`).

Rule, frozen: the count is verified programmatically before training, and if
it differs from 397,954 the run STOPS and is investigated first. No training
begins on a parameter count that has not been reconciled.

---

## 6. Everything held identical to ARM-V

| Held fixed | ARM-V value |
|---|---|
| Disparity candidates | 24 |
| Candidate spacing | 8 px full-resolution (downsample_levels=3) |
| Represented range | 184 px (23 x 8) |
| Cost-volume shift | right |
| Cost method / shape | subtract, (B, 32, 24, H, W) |
| Readout normalization | regression_normalize=True |
| Aggregation | 4x Conv3d(32 to 32) plus Conv3d(32 to 1), unchanged |
| Refinement | one stage, dilations (1,2,4,8,1,1), unchanged |
| Loss | masked smooth L1 with beta 1.0 |
| Optimizer | Adam, lr 1e-3, betas (0.9, 0.999) |
| Schedule | CosineAnnealingLR (cosine annealing to 0 over 200 epochs) |
| Epochs | 200 |
| Batch size | 2 |
| Crop | random 256x512 crop |
| Augmentation | gain jitter sigma 0.1, no horizontal flip |
| Split | hailo_calib scenes 0-159 for training, hailo_val scenes 160-199 for evaluation |
| Seeds | 0, 1, 2 |
| BatchNorm | NO BatchNorm anywhere in the network |

Sources: `phase1/scripts/train_arm_v.py:47-52`, `:131-136`, `:150-157`;
`phase1/scripts/train_arm_v_seed.py`; `phase1/docs/AUDIT_STAGE1_BASELINE_FACTS.md`.

Explicit DO NOT list — none of the following may change:

- no change to loss, optimizer, LR or schedule;
- no change to epochs, crop, augmentation, disparity range, candidate count,
  refinement, or readout;
- no negation of the subtraction; no cosine similarity or cosine distance;
- no grouping, reduction, or projection of the 32 channels;
- no BN, attention, correlation, deep supervision, or any learnable parameter;
- no change to candidates, spacing, shift, aggregation, refinement or readout.

---

## 7. Frozen evaluation contract

ARM-Z does NOT change the contract. Restated verbatim:

KITTI 2015, `_10` frames, hailo_val scenes 160-199, disp_occ_0, fixed
368x1232 top-left crop after bottom/right padding, no resize, GT scale 1/256,
valid = gt > 0, pooled-pixel evaluation, EPE = mean absolute disparity error
in px, D1 = ((err > 3 px) AND (err > 5% gt)) * 100, raw output scored, ReLU
clamp in graph, exactly 3,802,797 valid pixels.

---

## 8. Decision rule, frozen before any result is seen

ARM-V 3 seeds: 1.8903392, 1.5493088, 1.8785827.
ARM-V mean EPE = 1.7727436
ARM-V spread   = 0.3410304
anchored threshold = 1.4317132

```
EPE < 1.4317132                -> CONFIRMED
1.4317132 <= EPE < 1.7727436   -> INCONCLUSIVE
EPE >= 1.7727436               -> REFUTED
```

Statistical caveat, recorded now: 1.4317132 is the project's pre-registered
confirmation gate inherited from the ARM-V measured seed uncertainty. It is
NOT a claim of universal statistical significance.

The threshold is inherited from the preregistered ARM-V mean-minus-spread
criterion and MUST NOT be altered after seeing results. The thresholds may
not be moved after results are seen, and no new threshold may be derived
from ARM-Z after training.

Reference point: frozen reference EPE = 1.3134470770188373 px. This is
context only; it plays no role in the CONFIRMED / INCONCLUSIVE / REFUTED
verdict, which turns exclusively on the ARM-V-anchored thresholds above.

---

## 9. Seed discipline

Seeds exactly 0, 1, 2, run in that order, all three to completion under the
identical recipe. No seed may be replaced, dropped, selectively rerun, or
used to tune. Intermediate validation numbers may not be used to terminate,
retune, alter seeds or change the recipe.

All three seeds run to completion before any number is read for the verdict.
Checkpoint selection uses the train-time curve and final-epoch weights only,
never the frozen 40-scene score — same rule as ARM-V
(`phase1/scripts/train_arm_v.py:10-16`).

---

## 10. Smoke test to be passed BEFORE training

The following checks must all pass before the 200-epoch runs begin
(`phase1/scripts/smoke_arm_z.py`, output saved to
`phase1/runs/arm_z_smoke/smoke.json`):

1. left/right feature tensor shapes at the ARM-V config;
2. normalization is over channels only;
3. nonzero feature vectors have ~unit L2 norm after normalization;
4. zero vectors are numerically safe via max(norm, 1e-5);
5. cost volume shape == (B, 32, 24, H, W);
6. cost semantics == left_normalized - shifted_right_normalized;
7. no sign negation introduced;
8. parameter count unchanged vs ARM-V (397954 both);
9. no BN module anywhere in the model (module-type scan);
10. gradients propagate through normalization -> cost volume -> aggregation ->
    refinement (non-None, non-zero grad on feature_extractor weights);
11. shift liveness is real: with feature_normalize=True, disparity slices of
    the cost volume are NOT all identical (max abs difference between slice k
    and slice 0 is > 0 for some k).

Unit tests (`tests/test_feature_normalize.py`): unit L2 norm along dim=1
(atol 1e-5); zero-vector finiteness; channel-only norm (spatial magnitude
removed, direction preserved); param-count equality at 397954; volume shape;
hand-computed sign equality `Lhat - shift_right(Rhat, k)`; flag-off bitwise
reproduction of the ARM-V volume.

---

## 11. Liveness test

The existing liveness check: switching shift=right to shift=none must change
the output, proving the stereo path is live; plus the section-10 item 11
check that cost-volume slices differ with normalization on. If the outputs
are identical, the run STOPS — the disparity search path is not exercised
and no accuracy verdict is meaningful.

---

## 12. Per-seed record to be captured

For each of seeds 0, 1, 2, the run record captures:

- best-contract EPE;
- final EPE;
- D1;
- RMSE;
- bad1;
- bad3;
- valid pixels;
- parameter count;
- strict-load result;
- contract_match;
- checkpoint SHA256;
- weight SHA16;
- best epoch.

---

## 13. Reporting and leaderboard rules

- Results go to `phase1/docs/ARM_Z_RESULT.md` after all three seeds are
  complete. No result document is written from a subset of seeds.
- `LEADERBOARD.md` is NOT edited until the three-seed verdict is complete.
- If REFUTED: close ARM-Z, no retuning.
- If INCONCLUSIVE: no retuning; report the result; decide separately whether
  the ARM-V baseline needs tightening (e.g. additional seeds) before any
  follow-up on this mechanism.
- If CONFIRMED: ARM-Z becomes the candidate/incumbent per existing project
  procedure, and no further modification is stacked on top immediately.
- Any deviation from this preregistration marks the run INVALID rather than
  silently changing the protocol.
- The project stays in Phase 1. No Phase 3 is created.
- Regressions, if observed, are recorded honestly and never reinterpreted
  as successes.

---

## 14. What may not change

The hypothesis, the intervention (including before-shift placement,
independent L/R normalization, channel-axis L2 norm, eps = 1e-5,
max(norm, eps) division, subtraction cost with no negation, zero trainable
parameters, (B,32,24,H,W) volume), the insertion point, the config-flag
default, the parameter-budget stop rule, the frozen ARM-V items and DO NOT
list, the evaluation contract, the decision thresholds, the seed discipline,
the smoke-test gate, the liveness check, the per-seed record fields, and the
reporting rules in sections 1–13. If any of them turns out to be wrong, the
run is recorded as-is and a new experiment ID is registered.
