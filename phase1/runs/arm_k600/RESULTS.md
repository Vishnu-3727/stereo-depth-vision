# ARM K600 — 600-epoch KITTI budget extension (SCORED)

## Hypothesis

Remaining gap is budget-bound: extending 200 -> 600 KITTI epochs (ARM K recipe,
cosine horizon matched) reduces frozen EPE by >= 1.0 px vs ARM K best (5.5271927).

## Single variable changed

Budget 200 -> 600 epochs, with the cosine-annealing horizon matched
(T_max 200 -> 600). Nothing else changed vs ARM K.

## Frozen variables

- Initialisation: PyTorch defaults, random; seed 0
- Dataset: KITTI 2015; split hailo_calib (scenes 0-159) train,
  hailo_val (160-199) eval, overlap 0 (asserted in code)
- Split enforcement: train_base split='hailo_calib' only; no hailo_val sample
  enters the optimizer loop
- Resolution/crop: random 256x512; disparity_range 176; batch 2; fp32
- Optimizer Adam betas=(0.9, 0.999); LR 1e-3 cosine-annealed to 0
- Loss: masked smooth L1, beta=1.0, valid = gt > 0 and gt < max_disparity
- Augmentation: random crop, independent per-image gain jitter sigma=0.1;
  no horizontal flip; device cuda
- Known deviation: no batch normalisation (BN-folded exported artifact),
  carried into all arms

## Frozen scores (phase1/harness/frozen_eval.py, unmodified)

Guard: contract_match true, 3,802,797 valid pixels, 40 scenes, both snapshots.
Scored exactly as ARM K was (phase1/scripts/score_arm.py).

| Snapshot | Checkpoint | Frozen EPE | Frozen D1 | RMSE | Valid px | Scenes |
|---|---|---|---|---|---|---|
| Best (PRIMARY, epoch 350, 10-scene val 10.7240) | arm_k600_best.pth | 8.5583436 | 70.1146814% | 13.0444696 | 3,802,797 | 40 |
| Final (epoch 599, 10-scene val 11.2395) | arm_k600_final.pth | 8.8405404 | 69.3951321% | 13.5988021 | 3,802,797 | 40 |

Full JSON: `phase1/runs/arm_k600/frozen_eval_best.json`,
`phase1/runs/arm_k600/frozen_eval_final.json`.

Checkpoints (unmodified):
best sha256 ca29e4f3f41d20808a78b159de2e04c51a098d3ef4352a921f4d75110d6358be;
final sha256 cbd189584820333ba35e2a5a1eddc508afd6a4bb7fb7d734059f042b7be5532f.
Wall clock 10710.5 s (2.975 h), 600 epochs.
Script `phase1/scripts/train_arm_k600.py`, log
`phase1/runs/arm_k600/training_log.jsonl`, record
`phase1/runs/arm_k600/arm_k600_record.json`.

## Decision vs the >= 1.0 px materiality bar

delta = EPE_ARMK(5.5271927) - EPE_ARMK600-best(8.5583436) = -3.0311509 px.

The sign is NEGATIVE: ARM K600 best is 3.0311509 px WORSE (higher EPE) than
ARM K. The required >= 1.0 px improvement did not occur; EPE moved the wrong
way by a material margin.

Verdict: hypothesis REFUTED. 200 -> 600 epochs under this recipe does not
reduce frozen EPE -- it degrades it materially. ARM K (5.5271927) remains the
incumbent. Best-vs-final snap gap here is 0.2822 px (below the bar); snapshot
choice moves nothing.

The 10-scene training-time val figures (best 10.7240 @ epoch 350, final 11.239)
are the monitoring curve, NOT the score, and are not presented as the score.

## Train/val divergence (reported as observed)

OBSERVED: mean training loss fell from ~8.8 (epoch 35: 8.8387693) to 2.71
(epoch 599: 2.7100153) while the 10-scene val EPE never beat 10.7240 after
epoch 350 (final-epoch val 11.2395).

INTERPRETATION (labelled as such, not a measured result): the falling train
loss against a flat val curve is consistent with the model continuing to fit
the 160-scene hailo_calib training distribution without further transfer to
the held-out hailo_val scenes -- extra budget past ~epoch 350 bought
optimisation progress, not generalisation.
