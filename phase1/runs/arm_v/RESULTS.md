# ARM V — results (finer disparity sampling: stride 1/8, 24 candidates)

## Hypothesis

The cost volume samples disparity every 16 full-resolution pixels over 12
candidates, covering 0..176 px. KITTI carries ground truth beyond 176 px, and
soft-argmin must interpolate across 16 px gaps with no matching evidence inside
them. Halving the matching stride to 1/8 and doubling the candidates to 24 gives
8 px spacing over 0..184 px — finer evidence and fuller range — reducing
frozen-contract EPE by at least 0.4 px versus the ARM U 3-seed mean 2.3732058.

The teammate model matches at 1/8 with 24 hypotheses at 8 px and reports 1.659
EPE. That is the external evidence motivating this specific change.

## Intervention (one coherent change)

`StereoNetConfig(downsample_levels=3, num_disparities=24,
cost_volume_shift="right", regression_normalize=True)` versus ARM U
downsample_levels=4, num_disparities=12. Stride halves and candidate count
doubles TOGETHER so the disparity RANGE stays about constant (176 -> 184 px)
and only the SAMPLING DENSITY changes. No source change was needed (both knobs
already honoured end to end); no `StereoNetConfig` default changed; no other
hyperparameter changed. Diff of `phase1/scripts/train_arm_u.py` vs
`phase1/scripts/train_arm_v.py` confirms the config line, naming, env-var
plumbing and record strings are the only differences.

## Frozen variables

Seed 0, batch 2, random 256x512 crops, gain jitter sigma=0.1, Adam
(0.9,0.999), LR 1e-3, cosine annealing to 0 over 200 epochs, 200 epochs,
masked smooth-L1 beta=1.0 valid gt>0 and gt<184 (max_disparity derived from the
config: 184), ImageNet norm, no BN, fp32, split hailo_calib (160) train /
hailo_val (40) eval with the overlap-0 assertion, same DataLoader generator +
worker_init_fn, same best+final checkpoint policy.

## Pre-training checks (measured)

- Derived properties: feature_stride 8, max_disparity_px 184. Confirmed before
  training.
- Shapes: feature maps 23x77 -> 46x154 at eval resolution (368x1232); cost
  volume (B,32,12,23,77) -> (B,32,24,46,154), exactly 8x the elements.
  Forward pass verified at 368x1232 and at the 256x512 training crop
  (features 32x64, volume (B,32,24,32,64)).
- Memory: peak GPU for batch 2 (forward+backward, 256x512) = 1.088 GB on the
  8 GB card. Fits. Batch size NOT reduced.
- Parameters: 397,954 vs ARM U 423,586 — NOT parameter-free. The entire delta
  (−25,632) is the removed fourth 5x5 downsample conv (32*32*25+32);
  aggregation, refinement and regression are untouched (num_disparities changes
  the cost depth axis, not channel counts). State dict: 70 keys vs 72,
  strict_ok true under the matching config.
- Step time: 20 training steps timed at 0.102 s/step on random tensors
  (compute only) -> extrapolated 200-epoch wall ~1629 s; 2-epoch smoke on real
  data (dataloader + val curve) took 41.3 s -> extrapolated ~4130 s. Both far
  below the 4 h limit, so the full run proceeded. Actual wall: 3659.4 s
  (ARM U took 3670 s — the 8x cost volume did not slow training; dataloading
  dominates).

## Training

200 epochs, wall 3659.4 s. Train loss 10.89 -> ~0.62; 10-scene val EPE best
3.2360 @epoch 180 (ARM U train-time best was 3.9355 @199 — ARM V optimised
better on the train-time curve too).

## Frozen scores (contract_match true, 3,802,797 px, 40 scenes)

| Snapshot | EPE | D1 | RMSE | sha256 |
|---|---|---|---|---|
| best-val (ep 180) | 1.8903392 | 10.7425666% | 5.5483581 | f2dcf8a2…b452cc4 |
| final (ep 199) | 1.9043133 | 10.7847198% | 5.6609772 | d72c6840…cd61d010 |

Eval-config note (read before trusting the numbers): `score_checkpoint` in the
unmodified `phase1/harness/frozen_eval.py` builds `StereoNet(StereoNetConfig())`,
i.e. stride 1/16, 12 candidates, shift=none, regression_normalize=False, and
checkpoints store weights only — so the literal harness would evaluate ARM V
weights through a different forward graph (and a mismatched state dict) than
the one trained. The scores above use a line-for-line mirror of
`score_checkpoint` with `StereoNetConfig(downsample_levels=3,
num_disparities=24, cost_volume_shift="right", regression_normalize=True)`;
dataset, GT scale, metrics, and contract guard are imported unmodified from
`frozen_eval.py` (`frozen_eval_best.json` / `frozen_eval_final.json`, with
`strict_ok: true`, 70 keys, 397954 params).

## Saturation report + stereo-path liveness (hailo_val scene 0, best checkpoint)

| aggregated_cost |max| | disp_init mean | disp_init std | disp_init range |
|---|---|---|---|---|
| 2.80e9 | 6.4109 | 3.4895 | 2.58–17.13 |

`disparity_initial` varies spatially (std 3.49, spanning candidates ~2.6–17.1):
not pinned to a candidate, not collapsed. The aggregation costs sit at ~1e9
scale — the saturated regime — and the normalisation keeps the readout alive.

Liveness: substituting shift=none at eval time (keeping everything else, same
weights, scene 0) changes the final output by up to ~69.2 px max abs diff —
NOT bit-identical — and the full-40 EPE under shift=none substitution collapses
to 39.55. The stereo path is LIVE: the disparity search contributes to the
prediction.

## Delta vs ARM U mean + verdict

Signed delta (best-val EPE vs ARM U 3-seed MEAN): 1.8903392 − 2.3732058 =
**−0.4828666 px** (better).

Decision rule: the old 1.0 px materiality bar was calibrated against a 4.2 px
gap to the reference. That gap is now 1.06 px, so a 1.0 px bar would count only
reaching the reference as success. The bar for ARM V is set instead at 3x the
MEASURED ARM U 3-seed spread (0.1341484 px), i.e. 0.40 px — the level at which
an improvement is separable from run-to-run noise. Compared against the ARM U
3-seed MEAN (2.3732058), not seed-0 best 2.2866369: single-seed ARM V versus
best-of-three ARM U would be unfair in our favour.

- EPE <= 1.9732058 => ACCEPTED; flag for multi-seed confirmation.
- 1.9732058 < EPE < 2.3732058 => improvement below the bar; INCONCLUSIVE.
- EPE >= 2.3732058 => REFUTED versus the ARM U mean.

**Verdict: ACCEPTED — flag for multi-seed confirmation.** Finer sampling beats
the ARM U mean by 0.48 px, clearing the 0.40 px bar. The improvement over the
mean is 3.6x the run-to-run spread. Stereo-path liveness reported separately:
path LIVE (shift substitution moves output ~69.2 px, EPE 1.89 -> 39.55), and
EPE accepted. No retuning was done and no follow-up run is launched here.
