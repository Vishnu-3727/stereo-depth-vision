# ARM V-S1 — results (ARM V recipe, seed 1)

## Hypothesis

Seed replication of ARM V: the exact ARM V recipe
(`StereoNetConfig(downsample_levels=3, num_disparities=24,
cost_volume_shift="right", regression_normalize=True)`) on a different seed,
scored on the frozen contract. No acceptance decision is made here; the manager
decides.

## Intervention

None. Identical recipe to ARM V seed 0; only SEED changed (0 → 1). The seed
feeds `seed_all()`, the DataLoader generator, `worker_init_fn` and the
`CroppedKitti` crop/jitter RNG exactly as in `train_arm_v.py` — the diff of
`phase1/scripts/train_arm_v_seed.py` vs `phase1/scripts/train_arm_v.py` is
exactly two lines (`ARM_V_SEED` / `ARM_V_OUT_DIR` plumbing). Everything else
frozen: 200 epochs, batch 2, Adam, LR 1e-3, cosine annealing, 256x512 random
crops, gain jitter sigma 0.1, masked smooth-L1 (valid gt>0 and gt<184),
fp32, train split hailo_calib (scenes 0-159) only.

## Training

200 epochs, wall 3716.1 s. Train loss 10.24 → ~0.54; 10-scene val EPE best
2.2723 @epoch 199 (seed 0 reached 3.2360 @epoch 180).

## Frozen scores (contract_match true, 3,802,797 px, 40 scenes)

| Snapshot | EPE | D1 | RMSE | sha256 |
|---|---|---|---|---|
| best-val (ep 199) | 1.5493088 | 9.3507752% | 4.1329 | c40e697f…8f2d78 |
| final (ep 199) | 1.5493088 | 9.3507752% | 4.1329 | adcf80fd…81c67ee |

Scored through a line-for-line mirror of `score_checkpoint` with
`StereoNetConfig(downsample_levels=3, num_disparities=24,
cost_volume_shift="right", regression_normalize=True)`
(`phase1/runs/arm_v_s1/score_mirror.py`); dataset, GT scale, metrics, and
contract guard imported unmodified from the unmodified
`phase1/harness/frozen_eval.py` (`frozen_eval_best.json` /
`frozen_eval_final.json`, `strict_ok: true`, 70 keys, 397954 params).

Note: best-val epoch is 199, so the best-val and final-epoch snaps carry
identical weights (weight_sha16 2a3aaedd125f696f for both) and identical
frozen scores. Checkpoint selection used the train-time 10-scene curve only,
never the 40-scene score.

## Stereo-path liveness (full 40-scene control, best checkpoint)

Substituting shift=none at eval time (keeping regression_normalize=True, same
weights) degrades the frozen score from 1.5493 EPE / 9.35% D1 to **30.8966
EPE / 98.10% D1** (`liveness_shift_none.json`). The disparity search is
carrying the model, as for seed 0 (1.8903 → 39.5550).

## Reference numbers (no verdict)

ARM V seed 0 best-val EPE: 1.8903392. Observed difference (best-val EPE):
1.5493088 − 1.8903392 = **−0.3410304 px**. No acceptance decision is made
here — the manager decides. No tuning was done.
