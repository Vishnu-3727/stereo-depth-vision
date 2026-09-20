# ARM V-S2 — results (ARM V recipe, seed 2)

## Hypothesis

Seed replication of ARM V: the exact ARM V recipe
(`StereoNetConfig(downsample_levels=3, num_disparities=24,
cost_volume_shift="right", regression_normalize=True)`) on a different seed,
scored on the frozen contract. No acceptance decision is made here; the manager
decides.

## Intervention

None. Identical recipe to ARM V seed 0; only SEED changed (0 → 2). The seed
feeds `seed_all()`, the DataLoader generator, `worker_init_fn` and the
`CroppedKitti` crop/jitter RNG exactly as in `train_arm_v.py` — the diff of
`phase1/scripts/train_arm_v_seed.py` vs `phase1/scripts/train_arm_v.py` is
exactly two lines (`ARM_V_SEED` / `ARM_V_OUT_DIR` plumbing). Everything else
frozen: 200 epochs, batch 2, Adam, LR 1e-3, cosine annealing, 256x512 random
crops, gain jitter sigma 0.1, masked smooth-L1 (valid gt>0 and gt<184),
fp32, train split hailo_calib (scenes 0-159) only.

## Training

200 epochs, wall 3715.3 s. Train loss 10.34 → ~0.61; 10-scene val EPE best
3.1809 @epoch 180 (seed 0 reached 3.2360 @epoch 180).

## Frozen scores (contract_match true, 3,802,797 px, 40 scenes)

| Snapshot | EPE | D1 | RMSE | sha256 |
|---|---|---|---|---|
| best-val (ep 180) | 1.8785827 | 10.7135353% | 5.5613 | 44a429b2…545359 |
| final (ep 199) | 1.8894581 | 10.7080657% | 5.6908 | f3729641…77c3b75 |

Scored through a line-for-line mirror of `score_checkpoint` with
`StereoNetConfig(downsample_levels=3, num_disparities=24,
cost_volume_shift="right", regression_normalize=True)`
(`phase1/runs/arm_v_s2/score_mirror.py`, a copy of the s1 script); dataset, GT
scale, metrics, and contract guard imported unmodified from the unmodified
`phase1/harness/frozen_eval.py` (`frozen_eval_best.json` /
`frozen_eval_final.json`, `strict_ok: true`, 70 keys, 397954 params).

Note: the final-epoch snap (1.8895) scores within 0.011 of the best-val snap
(1.8786). The best-val snap remains the primary per the frozen checkpoint
policy (selection used the train-time 10-scene curve only, never the 40-scene
score); both are recorded.

## Stereo-path liveness (full 40-scene control, best checkpoint)

Substituting shift=none at eval time (keeping regression_normalize=True, same
weights) degrades the frozen score from 1.8786 EPE / 10.71% D1 to **34.9067
EPE / 99.05% D1** (`liveness_shift_none.json`) — consistent with seeds 0 and 1.

## Reference numbers (no verdict)

ARM V seed 0 best-val EPE: 1.8903392. Observed difference (best-val EPE):
1.8785827 − 1.8903392 = **−0.0117565 px**. No acceptance decision is made
here — the manager decides. No tuning was done.
