# ARM-Y — result record: left-image-guided cost-volume excitation

**Verdict: `REFUTED`.**

Preregistration: `phase1/docs/ARM_Y_PREREGISTRATION.md`, frozen before any
ARM-Y code existed and before any ARM-Y training run. This record is judged
against that contract. All three seeds (0, 1, 2) ran to completion under the
identical recipe before any number was read for the verdict. No seed was
replaced, dropped, selectively rerun, or used to tune.

Mechanism reference: `phase1/docs/PUBLIC_STEREO_RESEARCH_AUDIT.md`
sections 7–9. Machine-generated metrics: `phase1/runs/arm_y/`,
`phase1/runs/arm_y_s1/`, `phase1/runs/arm_y_s2/`
(`frozen_eval_best.json`, `frozen_eval_final.json`,
`liveness_shift_none.json`, `arm_y_record.json` per run).
Scorer: `phase1/runs/arm_y/score_mirror.py`.

---

## 1. Exact intervention

The single intervention, exactly as preregistered
(`ARM_Y_PREREGISTRATION.md` section 2), inserted in `StereoNet.forward`
between `volume = self.cost_volume(left_features, right_features)` and
`cost = self.aggregation(volume)`, and nowhere else:

```python
gate   = Conv2d(32, 16, kernel_size=1, bias=False)(left_features)
gate   = LeakyReLU(negative_slope=0.01)(gate)
gate   = Conv2d(16, 32, kernel_size=1)(gate)          # bias=True
gate   = sigmoid(gate).unsqueeze(2)
volume = gate * volume
```

Shapes: `left_features` (B, 32, H, W); `volume` (B, 32, D, H, W);
`gate` before unsqueeze (B, 32, H, W); after unsqueeze (B, 32, 1, H, W).
The gate broadcasts over D, so it is channel-wise, pixel-wise and
DISPARITY-INDEPENDENT: it reweights channels, never candidates.

The LeakyReLU is part of the frozen intervention. It is present in the
audited reference block: CoEx BasicConv is Conv2d(bias=False) +
BatchNorm2d + LeakyReLU, at
`reference/public_repos/OpenStereo/stereo/modeling/models/coex/submodule.py:43-70`.
BatchNorm is deliberately EXCLUDED because BN is a separate,
deployment-relevant change (our no-BN state is a recorded deviation matching
the BN-folded exported artifact, `phase1/scripts/train_arm_v.py:157`). The
LeakyReLU is RETAINED because without it the two 1x1 convolutions collapse
algebraically into a single linear rank-16 map, which would not be the
audited mechanism.

No other graph change was made.

## 2. Exact initialization

The audited CoEx initialization, reproduced for the TWO NEW convolutions
only (source:
`reference/public_repos/OpenStereo/stereo/modeling/models/coex/coex_cost_processor.py:83-88`):

```python
n = kernel_size[0] * kernel_size[1] * out_channels
m.weight.data.normal_(0, math.sqrt(2.0 / n))
```

Concretely:

- conv1 (32 -> 16, 1x1): n = 1*1*16 = 16, std = sqrt(2/16).
- conv2 (16 -> 32, 1x1): n = 1*1*32 = 32, std = sqrt(2/32).

The reference loop initializes WEIGHTS only, so the bias of conv2 keeps the
PyTorch default. Recorded explicitly: conv2 bias was NOT zeroed, NOT
initialized by the reference rule; it keeps whatever PyTorch default
initialization produces.

No pre-existing ARM-V parameter initialization was altered. Only ONE ARM-Y
variant was run. There is no PyTorch-defaults arm.

## 3. Exact parameter count

- ARM-Y total: 399,010.
- Baseline (ARM-V): 397,954.
- Delta: 1,056 = (32*16) + (16*32 + 32): conv1 512 weights no bias;
  conv2 512 weights + 32 bias; 512 + 544 = 1,056.

Budget rule enforced in code before training on every seed: the count was
verified programmatically, and it matched the preregistered budget
(arm_y 399,010, baseline 397,954, delta 1,056). Training began on a
reconciled count on all three seeds.

Strict load on all three seeds: strict_ok true, missing [], unexpected [],
matched 73 tensors, params 399,010. See section 14 for the compat-block
`keys_ok`/`params_ok` false flags, which are NOT an ARM-Y defect.

## 4. Smoke-test results

Run BEFORE training. All checks passed: 11 of 11 PASS.

- Model construction succeeds.
- Exact tensor shapes: gate (1,32,32,64), gate unsqueezed (1,32,1,32,64),
  gated volume (1,32,24,32,64).
- Forward pass succeeds; backward pass succeeds.
- No NaN/Inf in outputs or gradients.
- Gradient reaches excitation conv1 (norm 0.0104) and conv2 (0.0122).
- Gradient reaches the feature extractor (8.94) and the aggregation (655):
  the stereo path is differentiable end to end.
- No BatchNorm anywhere (module census finds zero BN layers).
- ARM-V vs ARM-Y outputs differ on the same input (max abs diff 15.5061):
  the gate is not a numerical no-op.
- Liveness at smoke time: max abs diff 1.36969.
- `pytest tests/test_excitation.py`: 5 passed.

## 5. Gate statistics — DIAGNOSTIC ONLY

Gate statistics at initialization, DIAGNOSTIC ONLY:

- min 0.312200, max 0.670645, mean 0.484820, std 0.048219.

These were recorded as a diagnostic observation per preregistration
section 10. They were NOT used to tune anything — not to rescale the init,
not to add normalization, not to alter the recipe. The run proceeded
unchanged.

## 6. Seed-level results

Best checkpoint per seed under the frozen evaluation contract
(KITTI 2015, `_10` frames, hailo_val scenes 160-199, disp_occ_0, fixed
368x1232 top-left crop after bottom/right padding, no resize, GT scale
1/256, valid = gt > 0, pooled-pixel evaluation, raw output scored, ReLU
clamp in graph; contract_match true, valid_pixels 3,802,797, 40 scenes,
gt_scale 256.0, split hailo_val, gt_source disp_occ_0 on all three seeds).

| seed | run dir | best EPE (px) | D1 (%) | RMSE | bad1 | bad3 | final-epoch EPE | final-epoch D1 | best epoch | wall (s) | 10-scene monitor best |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `phase1/runs/arm_y` | 3.7910550 | 34.8116137 | 7.7229856 | 71.48136 | 34.95524 | 3.7980249 | 34.8638121 | 190 | 4041.7 | 5.5274 |
| 1 | `phase1/runs/arm_y_s1` | 4.2937846 | 40.4628225 | 8.3373881 | 76.52352 | 40.56170 | 4.2296322 | 38.3016501 | 155 | 4000.5 | 6.0372 |
| 2 | `phase1/runs/arm_y_s2` | 2.1376764 | 12.7557427 | 6.1082178 | 41.65692 | 12.92557 | 2.1402015 | 12.7721254 | 195 | 3827.5 | 3.5613 |

Checkpoint selection used the train-time curve and final-epoch weights
only, never the frozen 40-scene score — same rule as ARM-V.

git_head for all three runs: `58e8a19908ddbd35451652c61aef478f56b51ebd`.

## 7. Mean

ARM-Y 3-seed mean (best-checkpoint EPE) = 3.4075053 px.

## 8. Spread

ARM-Y spread (max minus min over the three best-checkpoint EPEs) =
2.1561082 px.

## 9. Comparison with ARM-V

ARM-V 3 seeds: 1.8903392, 1.5493088, 1.8785827.
ARM-V 3-seed mean = 1.7727436 px. ARM-V spread = 0.3410304 px.

ARM-Y mean (3.4075053) is worse than the ARM-V mean by 1.6347617 px —
nearly the entire ARM-V mean again.

The result is unanimous: every ARM-Y seed (2.1376764, 3.7910550,
4.2937846) is worse than ARM-V's WORST seed (1.8903392). The mean is not
dragged down by a single outlier.

ARM-Y's seed spread (2.1561082) is 6.3x ARM-V's (0.3410304). The
intervention destabilised training as well as worsening the mean.

## 10. Comparison with the frozen reference

Frozen reference EPE = 1.3134470770188373 px. This is context only; per the
preregistration it plays no role in the CONFIRMED / INCONCLUSIVE / REFUTED
verdict, which turns exclusively on the ARM-V-anchored thresholds.

For the record: ARM-Y mean (3.4075053) is 2.0940582 px above the frozen
reference, and even ARM-Y's best seed (2.1376764) is 0.8242293 px above it.

## 11. Liveness results

Preregistered liveness check (section 11): switching shift=right to
shift=none must change the output, proving the stereo path is live. If the
outputs were identical, the run would STOP.

All three seeds are live — the shift=none EPE differs greatly from the
shift=right best EPE on every seed:

| seed | shift=right best EPE | shift=none EPE (same best checkpoint) |
|---:|---:|---:|
| 0 | 3.7910550 | 23.6258314 |
| 1 | 4.2937846 | 17.9360868 |
| 2 | 2.1376764 | 31.9351917 |

The disparity search path is exercised on every seed. No accuracy verdict
is vacated on liveness grounds.

## 12. Verdict according to the frozen rule

Preregistered rule (`ARM_Y_PREREGISTRATION.md` section 8), applied exactly
as written, with no reinterpretation:

```
ARM-Y mean <  1.4317132              -> CONFIRMED
1.4317132 <= ARM-Y mean <  1.7727436 -> INCONCLUSIVE
ARM-Y mean >= 1.7727436              -> REFUTED
```

3.4075053 >= 1.7727436, therefore the verdict is REFUTED.

The preregistration named this failure mode in advance (section 7 risks):
every public implementation of this block sits in a BatchNorm network with
SceneFlow pretraining, and ARM-Y has neither.

Scientific conclusion, stated carefully and without overclaiming:
left-image-guided cost-volume excitation, which is repeatedly useful in
larger SceneFlow-pretrained public stereo architectures, does NOT transfer
to this ~398k-parameter, BatchNorm-free, KITTI-only (160-scene) regime. Do
not claim the mechanism is useless in general. Do not claim any cause that
was not measured — the BN/pretraining explanation is a HYPOTHESIS
consistent with the preregistered risk, not a measured finding.

## 13. Checkpoint hashes

| seed | best checkpoint | SHA256 | weight SHA16 |
|---|---|---|---|
| 0 | `phase1/runs/arm_y/arm_y_best.pth` | `5fcbf0862195704252f0c8cbc0c053258af8e30dc0c78f32835d68e9c60dbe33` | `a4b1aab55e506f76` |
| 1 | `phase1/runs/arm_y_s1/arm_y_best.pth` | `5836f914758fe57db0ba73cbd6e5427851a04865a6846b09c130c05760f08814` | `15dc285736ba8b90` |
| 2 | `phase1/runs/arm_y_s2/arm_y_best.pth` | `dce6587861be7c995c4919888e247ced515b23483aff8b69f96a1f9e6ac31f4a` | `d88fa268bd4b5310` |

## 14. Any deviations

- One test tolerance was loosened during implementation: the
  disparity-independence test asserted exact equality (== 0.0) on a float
  ratio and saw ~6e-08 residue, so it was changed to a < 1e-6 tolerance.
  Architecture, initialization and preregistration were untouched.
- `frozen_eval.py` hardcodes an older StereoNetConfig, so scoring used a
  per-run mirror scorer (`phase1/runs/arm_y/score_mirror.py`), following the
  existing project pattern (`phase1/runs/arm_v_s1/score_mirror.py`). The
  ONLY differences from that template are the model construction
  (cost_volume_excitation=True), the checkpoint filenames, and the
  docstring. Dataset, split, GT scale, valid mask, pooled metrics and
  contract guard are imported unmodified from `phase1.harness.frozen_eval`.
- The compat block reports keys_ok false and params_ok false. This is NOT
  an ARM-Y defect: those flags compare against a hardcoded 72-key
  original-model baseline, and ARM-V seed 1 and ARM-X report the same false
  values. ARM-V has 70 tensors / 397,954 params; ARM-Y has 73 tensors /
  399,010 params; the +3 tensors are exactly conv1.weight, conv2.weight,
  conv2.bias.
- No other deviation. The recipe, seeds, contract and thresholds were
  unchanged throughout, and no intermediate result was used to alter
  anything.

---

## Closure

Per preregistration section 13, REFUTED means: ARM-Y is closed, no
retuning, and no modification is stacked on top of it.

`phase1/results/LEADERBOARD.md` was deliberately NOT edited. Reason: the
two most recent refuted/inconclusive arms (ARM-W and ARM-X) have no
leaderboard rows, so the closest precedent is that recent arms are not
added. Whether ARM-Y (or ARM-W/ARM-X) should appear on the leaderboard is
flagged as a decision for the project owner rather than something already
done.

The project stays in Phase 1. No Phase 3 was created.
