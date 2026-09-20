# ARM-Z — result record: per-pixel L2-normalized feature difference

**Verdict: `INCONCLUSIVE`.**

Preregistration: `phase1/docs/ARM_Z_PREREGISTRATION.md`, frozen before any
ARM-Z code existed and before any ARM-Z training run. This record is judged
against that contract. All three seeds (0, 1, 2) ran to completion under the
identical recipe before any number was read for the verdict. No seed was
replaced, dropped, selectively rerun, or used to tune.

Mechanism reference: `phase1/docs/MATCHING_REPRESENTATION_AUDIT.md`
sections 6–9. Machine-generated metrics: `phase1/runs/arm_z/`,
`phase1/runs/arm_z_s1/`, `phase1/runs/arm_z_s2/`
(`frozen_eval_best.json`, `frozen_eval_final.json`,
`liveness_shift_none.json`, `arm_z_record.json` per run).
Scorer: `phase1/runs/arm_z/score_mirror.py`.
Self-contained record: `phase1/runs/arm_z/ARM_Z_EXPERIMENT_RECORD.json`.

---

## FACTS

The single intervention, exactly as preregistered
(`ARM_Z_PREREGISTRATION.md` section 2): per-pixel L2 unit-normalization of
the left and right feature maps, independently, before the existing
shift-right subtraction. No other graph change was made.

```python
eps = 1e-5
Lhat = L / torch.clamp(L.norm(p=2, dim=1, keepdim=True), min=eps)
Rhat = R / torch.clamp(R.norm(p=2, dim=1, keepdim=True), min=eps)
volume = build_cost_volume(Lhat, Rhat, num_disparities=24,
                           method="subtract", shift="right")
```

Frozen details, each as preregistered: L2 norm over the 32-CHANNEL axis only
(dim=1 of (B,32,H,W)); LEFT and RIGHT normalized INDEPENDENTLY, each from its
own norm, BEFORE disparity shifting; `max(norm, 1e-5)` clamp division, never
raw `v / norm`; eps = 1e-5; existing ARM-V cost volume unchanged
(`cost_k(x) = Lhat(x) - Rhat(x - k)`, k = 0..23); signed subtraction kept, no
negation, no cosine similarity/distance, no grouping/reduction/projection;
volume shape stays (B, 32, 24, H, W); +0 trainable parameters; shift = right,
regression_normalize = True.

Implementation identity: pure function `normalize_features_l2` in
`src/models/stereonet/stereonet.py:44-53` (no nn.Module, no parameters, no
buffers; eps = 1e-5 documented in its docstring); config flag
`feature_normalize: bool = False` on `StereoNetConfig`
(`src/models/stereonet/stereonet.py:84`), default False so every earlier arm
stays bit-identical; forward hook in `StereoNet.forward` between
`right_features = self.feature_extractor(right)` and
`volume = self.cost_volume(left_features, right_features)`
(`src/models/stereonet/stereonet.py:143-145`). `cost_volume.py`,
`regression.py`, `aggregation.py`, `refinement.py`, `feature_extractor.py`
untouched.

Parameter counts: ARM-Z 397,954; baseline (ARM-V) 397,954; delta 0. Budget
rule enforced in code before training on every seed: training began on a
reconciled count on all three seeds.

Preregistration path: `phase1/docs/ARM_Z_PREREGISTRATION.md`
(status: registered 2026-09-17).

git_head for all three runs: `58e8a19908ddbd35451652c61aef478f56b51ebd`.

Held fixed vs ARM-V (preregistration section 6): 24 disparity candidates;
8 px full-resolution spacing (downsample_levels=3); 184 px represented range;
shift right; subtract cost method, (B, 32, 24, H, W);
regression_normalize=True; aggregation 4x Conv3d(32 to 32) plus Conv3d(32 to
1); one-stage refinement, dilations (1,2,4,8,1,1); masked smooth-L1 loss
beta 1.0; Adam lr 1e-3 betas (0.9, 0.999); cosine annealing to 0 over 200
epochs; 200 epochs; batch size 2; random 256x512 crop; gain jitter sigma 0.1,
no horizontal flip; split hailo_calib scenes 0-159 train / hailo_val scenes
160-199 eval; seeds 0, 1, 2; no BatchNorm anywhere.

---

## MEASUREMENTS

Smoke test (`phase1/runs/arm_z_smoke/smoke.json`), run BEFORE training. All
checks passed: 11 of 11 PASS.

- Feature shapes (B,32,H/8,W/8): left (1, 32, 32, 64), right (1, 32, 32, 64).
- Normalization over channels only (max col diff 5.96e-08, min cosine
  0.99999982).
- Nonzero vectors ~unit norm (min 1.000000, max 1.000000, mean 1.000000).
- Zero vectors numerically safe via max(norm, 1e-5) (max abs 0, finite).
- Cost volume shape (B, 32, 24, H, W): got (1, 32, 24, 32, 64).
- Cost semantics == Lhat - shift_right(Rhat, k) (max diff 0).
- No sign negation (sub 0, add 1.99).
- Param count unchanged vs ARM-V (397954 both).
- No BatchNorm anywhere (module-type scan).
- Gradients propagate through normalization -> cost volume -> aggregation ->
  refinement (feat norm 44.2, agg norm 412, loss 10.220499; non-None,
  non-zero grad on feature_extractor weights).
- Shift liveness at smoke time: cost-volume slices differ (max slice diff
  0.799778).

Unit tests (`tests/test_feature_normalize.py`) per preregistration
section 10: unit L2 norm along dim=1 (atol 1e-5); zero-vector finiteness;
channel-only norm; param-count equality at 397954; volume shape;
hand-computed sign equality `Lhat - shift_right(Rhat, k)`; flag-off bitwise
reproduction of the ARM-V volume.

Seed-level results — best checkpoint per seed under the frozen evaluation
contract (KITTI 2015, `_10` frames, hailo_val scenes 160-199, disp_occ_0,
fixed 368x1232 top-left crop after bottom/right padding, no resize, GT scale
1/256, valid = gt > 0, pooled-pixel evaluation, raw output scored, ReLU
clamp in graph; contract_match true, valid_pixels 3,802,797, 40 scenes,
gt_scale 256.0, split hailo_val, gt_source disp_occ_0 on all three seeds):

| seed | run dir | best EPE (px) | D1 (%) | RMSE | bad1 | bad3 | final-epoch EPE | best epoch | wall (s) | 10-scene monitor best |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `phase1/runs/arm_z` | 1.7126133 | 9.6806114 | 4.8903369 | 34.69415 | 9.86881 | 1.7178527 | 185 | 3675.4 | 2.9618 |
| 1 | `phase1/runs/arm_z_s1` | 1.6543386 | 9.6198403 | 4.6734884 | 34.64032 | 9.91094 | 1.6537950 | 195 | 3620.5 | 2.7855 |
| 2 | `phase1/runs/arm_z_s2` | 1.8637884 | 10.1617573 | 5.6969331 | 36.06030 | 10.36745 | 1.8721789 | 180 | 3717.9 | 3.1835 |

Full per-seed metric detail (best checkpoint): seed 0 bad2 16.0319891;
seed 1 bad2 16.0825834; seed 2 bad2 16.6191622. Final-checkpoint D1:
seed 0 9.6942067; seed 1 9.6197878; seed 2 10.3531164.

Checkpoint selection used the train-time curve and final-epoch weights
only, never the frozen 40-scene score — same rule as ARM-V.

Three-seed statistics (best-checkpoint frozen EPE):

- Mean EPE = 1.7435801 px.
- Seed spread (max minus min) = 0.2094497 px.
- Standard deviation: population 0.0882667 px; sample 0.1081041 px.
- Mean D1 = 9.8207363 %.
- Reference gap (mean EPE minus frozen reference 1.3134470770188373) =
  0.4301330 px.
- Improvement vs ARM-V mean (1.7727436 minus mean EPE) = 0.0291635 px.

Comparison with ARM-V: ARM-V 3 seeds 1.8903392, 1.5493088, 1.8785827;
ARM-V mean 1.7727436 px; ARM-V spread 0.3410304 px. ARM-Z mean (1.7435801)
is 0.0291635 px better than the ARM-V mean. ARM-Z's seed spread
(0.2094497) is narrower than ARM-V's (0.3410304). Two ARM-Z seeds beat the
ARM-V mean; seed 2 (1.8637884) is worse than the ARM-V mean but better than
ARM-V's worst seed (1.8903392).

Comparison with the frozen reference: frozen reference EPE =
1.3134470770188373 px. This is context only; per the preregistration it
plays no role in the CONFIRMED / INCONCLUSIVE / REFUTED verdict, which
turns exclusively on the ARM-V-anchored thresholds. For the record: ARM-Z
mean (1.7435801) is 0.4301330 px above the frozen reference, and even
ARM-Z's best seed (1.6543386) is 0.3408916 px above it.

Frozen-eval contract verification per seed (read from
`frozen_eval_best.json` guard blocks): 40 scenes, 3,802,797 valid pixels,
disp_occ_0, 368x1232 top-left crop, GT 1/256, gt > 0, pooled,
contract_match true on all three seeds. Final-checkpoint evals also report
contract_match true with 3,802,797 valid pixels on all three seeds.

Liveness results — preregistered liveness check (section 11): switching
shift=right to shift=none must change the output, proving the stereo path
is live. If the outputs were identical, the run would STOP.

All three seeds are live — the shift=none EPE differs greatly from the
shift=right best EPE on every seed:

| seed | shift=right best EPE | shift=none EPE (same best checkpoint) |
|---:|---:|---:|
| 0 | 1.7126133 | 39.3784298 |
| 1 | 1.6543386 | 31.1347133 |
| 2 | 1.8637884 | 36.6206358 |

The disparity search path is exercised on every seed. No accuracy verdict
is vacated on liveness grounds.

Checkpoint hashes:

| seed | best checkpoint | SHA256 | weight SHA16 |
|---|---|---|---|
| 0 | `phase1/runs/arm_z/arm_z_best.pth` | `206207cd6e5f5daf1806082ce0ed2635a0dc566e143b21688476e847e1298f8e` | `6c9bfb9a6cd5fd01` |
| 1 | `phase1/runs/arm_z_s1/arm_z_best.pth` | `8ec9e9c0f2abdd08b2bd8cb15cf028573c8223ac81e5f4b2eeee229001472550` | `2e40fd1f022925f2` |
| 2 | `phase1/runs/arm_z_s2/arm_z_best.pth` | `3269b5ae598615532b7ac84f708095b564fd4d55c3916528d0f3b62fbe28f785` | `a27fcba63ec6f13a` |

Final checkpoints: seed 0 SHA256
`b93902f2b939fcfa30e3f294b0a347c0010ad240020aeb9daec101b1481aa9b2`
(weight `4c881754eff06922`); seed 1 SHA256
`1a676236f71424ac84d93feda5f39176a359a15b97a939efdc7ab29bb2300ba7`
(weight `690f303abe58fc12`); seed 2 SHA256
`3968187ec8067fefda0f2cd8d81fbcd224e1d4b034889c2f85b0d58a2a4959a0`
(weight `b17292c644d2052b`).

Strict load on all three seeds: strict_ok true, missing [], unexpected [],
matched 70 tensors, params 397,954. The compat block reports keys_ok false
and params_ok false. This is NOT an ARM-Z defect: those flags compare
against a hardcoded 72-key original-model baseline, and ARM-V seed 1
reports the same false values.

Scorer note: `frozen_eval.py` hardcodes an older StereoNetConfig, so
scoring used a per-run mirror scorer (`phase1/runs/arm_z/score_mirror.py`),
following the existing project pattern
(`phase1/runs/arm_v_s1/score_mirror.py`). The ONLY differences from that
template are the model construction
(`StereoNetConfig(downsample_levels=3, num_disparities=24,
cost_volume_shift="right", regression_normalize=True,
feature_normalize=True)`), the checkpoint filenames, and the docstring.
Dataset, split, GT scale, valid mask, pooled metrics and contract guard are
imported unmodified from `phase1.harness.frozen_eval`.

---

## DECISION

Preregistered rule (`ARM_Z_PREREGISTRATION.md` section 8), applied exactly
as written, with no reinterpretation:

```
EPE < 1.4317132                -> CONFIRMED
1.4317132 <= EPE < 1.7727436   -> INCONCLUSIVE
EPE >= 1.7727436               -> REFUTED
```

Arithmetic: ARM-Z three-seed mean EPE = 1.7435801 px
((1.7126132540515526 + 1.6543386448840673 + 1.8637883767845886) / 3).
1.4317132 <= 1.7435801 < 1.7727436, therefore the verdict is INCONCLUSIVE.

This is not a win. The mean clears the REFUTED boundary by only 0.0291635
px and sits 0.3118669 px above the CONFIRMED gate. A single good seed
establishes nothing: seed 1 (1.6543386) alone would also be INCONCLUSIVE,
and seed 2 (1.8637884) alone would be REFUTED.

---

## INTERPRETATION

The experiment establishes ONLY this: ARM-V plus per-pixel L2 feature
normalization changed the matching representation and produced the measured
change — a three-seed mean EPE of 1.7435801 px versus the ARM-V mean of
1.7727436 px — under the frozen KITTI evaluation protocol. No universal
superiority is claimed. No theoretical proof is claimed. No correspondence
proof is claimed. No generalization beyond the tested distribution is
claimed.

The verdict is not CONFIRMED, so ARM-V remains the incumbent. ARM-Z is not
to be retuned or rescued with another mechanism: per preregistration
section 13, INCONCLUSIVE means no retuning — report the result, and decide
separately whether the ARM-V baseline needs tightening (e.g. additional
seeds) before any follow-up on this mechanism. No modification is stacked
on top of ARM-Z.

---

## CLOSURE

ARM-Z status: INCONCLUSIVE, closed pending any separately registered
follow-up. What is closed: the `H_Z` hypothesis test as preregistered —
the three-seed mean was computed, the frozen rule was applied unchanged,
and the verdict is recorded. What must not be reopened: the ARM-Z decision
thresholds, the seed set (0, 1, 2), checkpoint selection, or the
intervention itself — none may be revisited to move this verdict.

`phase1/results/LEADERBOARD.md` was deliberately NOT edited, per the hard
prohibition in the governing brief. The three-seed decision is now complete
(verdict: INCONCLUSIVE, mean EPE 1.7435801 px) — whether and how ARM-Z
should appear on the leaderboard is flagged as a decision for the manager /
project owner rather than something already done.

No other arm's artifacts were touched. No training was run, no code was
modified, no tuning was performed.

The project stays in Phase 1. No Phase 3 was created.
