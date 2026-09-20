# ARM-Y — preregistration: left-image-guided cost-volume excitation

**Written before any ARM-Y code exists and before any ARM-Y training run.**
Everything below is frozen at registration. Nothing in sections 1–13 may be
edited after results are seen; corrections go in a separate note beside the
record, and any deviation from this document marks the run INVALID rather than
silently changing the protocol.

Status: registered 2026-09-16. Executes the intervention specified in
`phase1/docs/PUBLIC_STEREO_RESEARCH_AUDIT.md` sections 8–9, on the ARM-V base
recorded in `phase1/docs/AUDIT_STAGE1_BASELINE_FACTS.md` and
`phase1/scripts/train_arm_v.py` / `phase1/scripts/train_arm_v_seed.py`.

Base model structure cited from `src/models/stereonet/stereonet.py`.

---

## 0. Why this experiment exists

The public-stereo audit (`phase1/docs/PUBLIC_STEREO_RESEARCH_AUDIT.md`)
concludes we are not at a StereoNet architectural ceiling: there is one
specific, small, well-evidenced mechanism we have never tested —
left-image-guided cost-volume excitation — appearing in at least four
independent public implementations under three different names (CoEx
`channelAtt`, Fast-ACVNet `channelAtt`, LightStereo `AttentionModule` under
`LEFT_ATT`, and the IGEV/StereoBase/FoundationStereo family).

The identified bottleneck: left-image context currently enters our graph
exactly once — as refinement guidance, after readout
(`src/models/stereonet/refinement.py:51-53`, via
`src/models/stereonet/stereonet.py:122-125`). At that point the cost curve is
already collapsed to a scalar disparity and the matching evidence is gone.
Excitation injects image context before aggregation, reweighting *which
channels* of matching evidence to trust per pixel, before the decision is made.

ARM-W (group-wise cost, a compression that removes information) and ARM-X
(loss only, forward graph untouched) did not test this mechanism.

---

## 1. Hypothesis

`H_Y` — LEFT-IMAGE-GUIDED COST-VOLUME EXCITATION improves stereo
correspondence in our regime: the ARM-Y 3-seed mean EPE, measured under the
frozen evaluation contract (section 7), will fall below the pre-registered
CONFIRMED threshold of 1.4317132 px (section 8).

One mechanism only. Falsifiable: an ARM-Y 3-seed mean at or above the ARM-V
3-seed mean of 1.7727436 px REFUTES `H_Y` for our regime (section 8).

---

## 2. The intervention, exact and frozen

The single intervention is a left-image-guided cost-volume excitation gate,
inserted exactly as written here. No other graph change is permitted.

```python
gate   = Conv2d(32, 16, kernel_size=1, bias=False)(left_features)
gate   = LeakyReLU(negative_slope=0.01)(gate)
gate   = Conv2d(16, 32, kernel_size=1)(gate)          # bias=True
gate   = sigmoid(gate).unsqueeze(2)
volume = gate * volume
```

Shapes:

- `left_features`: (B, 32, H, W)
- `volume`: (B, 32, D, H, W)
- `gate` before unsqueeze: (B, 32, H, W)
- `gate` after unsqueeze: (B, 32, 1, H, W)

The gate broadcasts over D, so it is channel-wise, pixel-wise and
DISPARITY-INDEPENDENT: it reweights channels, never candidates. A
disparity-dependent gate would prejudge the answer that the aggregation
network exists to compute; this gate explicitly does not.

The LeakyReLU is part of the frozen intervention. It is present in the
audited reference block: CoEx BasicConv is Conv2d(bias=False) + BatchNorm2d
+ LeakyReLU, at
`reference/public_repos/OpenStereo/stereo/modeling/models/coex/submodule.py:43-70`.
BatchNorm is deliberately EXCLUDED because BN is a separate,
deployment-relevant change (our no-BN state is a recorded deviation matching
the BN-folded exported artifact, `phase1/scripts/train_arm_v.py:157`). The
LeakyReLU is RETAINED because without it the two 1x1 convolutions collapse
algebraically into a single linear rank-16 map, which would not be the
audited mechanism.

---

## 3. Insertion point, exact and frozen

In `StereoNet.forward` (`src/models/stereonet/stereonet.py`), between these
two existing lines and nowhere else:

```python
volume = self.cost_volume(left_features, right_features)
cost   = self.aggregation(volume)
```

`left_features` is already in scope at that point
(`src/models/stereonet/stereonet.py:115`). Nothing else moves. No other
insertion point, no second gate, no change to any other module wiring.

---

## 4. Initialization, frozen, part of the intervention

The audited CoEx initialization is reproduced for the TWO NEW convolutions
only:

```python
n = kernel_size[0] * kernel_size[1] * out_channels
m.weight.data.normal_(0, math.sqrt(2.0 / n))
```

Source:
`reference/public_repos/OpenStereo/stereo/modeling/models/coex/coex_cost_processor.py:83-88`.

Concretely:

- conv1 (32 -> 16, 1x1): n = 1*1*16 = 16, std = sqrt(2/16).
- conv2 (16 -> 32, 1x1): n = 1*1*32 = 32, std = sqrt(2/32).

The reference loop initializes WEIGHTS only, so the bias of conv2 keeps the
PyTorch default. Recorded explicitly: conv2 bias is NOT zeroed, NOT
initialized by the reference rule; it keeps whatever PyTorch default
initialization produces.

No pre-existing ARM-V parameter initialization may be altered. Every other
parameter in the network keeps exactly the ARM-V initialization
(PyTorch defaults, random — `phase1/scripts/train_arm_v.py:155`).

Only ONE ARM-Y variant will be run. There is no PyTorch-defaults arm. The
initialization choice above is fixed now and is not tuned afterwards; the
open decision noted in the audit (reference init vs PyTorch defaults) is
hereby closed in favour of the reference init.

---

## 5. Parameter budget

- Expected new parameters: 1,056 = (32*16) + (16*32 + 32).
  - conv1: 32*16 weights, no bias = 512.
  - conv2: 16*32 weights + 32 bias = 544.
  - Total: 512 + 544 = 1,056.
- Expected total: 397,954 + 1,056 = 399,010.

Rule, frozen: the count is verified programmatically before training, and if
it differs materially from 1,056 the run STOPS and is investigated first.
No training begins on a parameter count that has not been reconciled.

---

## 6. Everything held identical to ARM-V

| Held fixed | ARM-V value |
|---|---|
| Disparity candidates | 24 |
| Candidate spacing | 8 px full-resolution (downsample_levels=3) |
| Cost-volume shift | right |
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

- no change to cost construction, candidates, spacing, shift, aggregation,
  refinement or readout;
- no BN;
- no feature-extractor normalization;
- no temperature;
- no group-wise cost;
- no hourglass;
- no attention over the disparity axis;
- no extra refinement stage;
- no extra supervision;
- no loss, optimizer, LR or schedule change;
- no crop-sampling change;
- no SceneFlow;
- no augmentation change.

---

## 7. Frozen evaluation contract

ARM-Y does NOT change the contract. Restated verbatim:

KITTI 2015, `_10` frames, hailo_val scenes 160-199, disp_occ_0, fixed
368x1232 top-left crop after bottom/right padding, no resize, GT scale 1/256,
valid = gt > 0, pooled-pixel evaluation, EPE = mean absolute disparity error
in px, D1 = ((err > 3 px) AND (err > 5% gt)) * 100, raw output scored, ReLU
clamp in graph, exactly 3,802,797 valid pixels.

---

## 8. Decision rule, frozen before any result is seen

ARM-V 3 seeds: 1.8903392, 1.5493088, 1.8785827.
ARM-V mean   = 1.7727436 px
ARM-V spread = 0.3410304 px
CONFIRMED threshold = 1.7727436 - 0.3410304 = 1.4317132 px

```
ARM-Y mean <  1.4317132                 -> CONFIRMED
1.4317132 <= ARM-Y mean <  1.7727436    -> INCONCLUSIVE
ARM-Y mean >= 1.7727436                 -> REFUTED
```

Statistical caveat, recorded now: 1.4317132 is the project's pre-registered
confirmation gate inherited from the ARM-V measured seed uncertainty. It is
NOT a claim of universal statistical significance.

The thresholds may not be moved after results are seen, and no new threshold
may be derived from ARM-Y after training.

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

The following 10 checks must all pass before the 200-epoch runs begin:

1. model construction succeeds;
2. exact tensor shapes (section 2 shapes hold: gate before unsqueeze
   (B,32,H,W), after unsqueeze (B,32,1,H,W), gated volume (B,32,D,H,W));
3. forward pass succeeds;
4. backward pass succeeds;
5. no NaNs or Infs in outputs or gradients;
6. gradient reaches excitation conv 1;
7. gradient reaches excitation conv 2;
8. gradient reaches the feature extractor and the cost aggregation
   (i.e. gradients flow through the gate into both upstream parameters and
   the gated volume path — the stereo path is differentiable end to end);
9. final inference output shape is correct ((B,1,H,W) full resolution);
10. parameter count matches section 5 (new parameters 1,056; total 399,010);
    plus: ARM-V and ARM-Y outputs differ on the same input (the gate is not
    a numerical no-op); and no BatchNorm layer was introduced (module census
    finds zero BN layers).

Plus gate statistics (min, max, mean, std) over a few batches, recorded as
DIAGNOSTIC ONLY. They may not be used to modify the architecture or tune the
run — not to rescale the init, not to add normalization, not to alter the
recipe. If the gate has collapsed (near-constant output), that is recorded
as a diagnostic observation and the run proceeds unchanged.

---

## 11. Liveness test

The existing liveness check: switching shift=right to shift=none must change
the output, proving the stereo path is live. If the outputs are identical,
the run STOPS — the disparity search path is not exercised and no accuracy
verdict is meaningful.

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

- Results go to `phase1/docs/ARM_Y_RESULT.md` after all three seeds are
  complete. No result document is written from a subset of seeds.
- `LEADERBOARD.md` is NOT edited until the three-seed verdict is complete.
- If REFUTED: close ARM-Y, no retuning.
- If INCONCLUSIVE: no retuning; report the result; decide separately whether
  the ARM-V baseline needs tightening (e.g. additional seeds) before any
  follow-up on this mechanism.
- If CONFIRMED: ARM-Y becomes the candidate/incumbent per existing project
  procedure, and no further modification is stacked on top immediately.
- Any deviation from this preregistration marks the run INVALID rather than
  silently changing the protocol.
- The project stays in Phase 1. No Phase 3 is created.
- Regressions, if observed, are recorded honestly and never reinterpreted
  as successes.

---

## 14. What may not change

The hypothesis, the intervention (including the LeakyReLU and the exclusion
of BatchNorm), the insertion point, the initialization, the parameter-budget
stop rule, the frozen ARM-V items and DO NOT list, the evaluation contract,
the decision thresholds, the seed discipline, the smoke-test gate, the
liveness check, the per-seed record fields, and the reporting rules in
sections 1–13. If any of them turns out to be wrong, the run is recorded
as-is and a new experiment ID is registered.
