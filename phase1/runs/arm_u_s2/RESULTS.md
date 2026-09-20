# ARM U-S2 — results (ARM U recipe, seed 2)

## Hypothesis

Seed replication of ARM U: the exact ARM U recipe
(`StereoNetConfig(cost_volume_shift="right", regression_normalize=True)`) on a
different seed still beats ARM K (5.5271927) by at least 1.0 px on the frozen
contract.

## Intervention

None. Identical recipe to ARM U seed 0; only SEED changed (0 → 2). The seed
feeds `seed_all()`, the DataLoader generator, `worker_init_fn` and the
`CroppedKitti` crop/jitter RNG exactly as in `train_arm_u.py` — the diff of
`phase1/scripts/train_arm_u_seed.py` vs `phase1/scripts/train_arm_u.py` is
exactly two lines (`ARM_U_SEED` / `ARM_U_OUT_DIR` plumbing). Everything else
frozen: 200 epochs, batch 2, Adam, LR 1e-3, cosine annealing, 256x512 random
crops, gain jitter sigma 0.1, masked smooth-L1, fp32, train split hailo_calib
(scenes 0-159) only.

## Training

200 epochs, wall 3513.5 s. Train loss → ~0.93; 10-scene val EPE best 3.9081
@epoch 145 (seed 0 reached 3.9355 @epoch 199).

## Frozen scores (contract_match true, 3,802,797 px, 40 scenes)

| Snapshot | EPE | D1 | RMSE | sha256 |
|---|---|---|---|---|
| best-val (ep 145) | 2.4121953 | 17.3953803% | 6.0905 | 3d760d1a…38a08 |
| final (ep 199) | 2.3018475 | 15.7013640% | 6.2557 | a6c26d2c…1d03ac |

Scored through a line-for-line mirror of `score_checkpoint` with
`StereoNetConfig(cost_volume_shift="right", regression_normalize=True)`
(`phase1/runs/arm_u_s2/score_mirror.py`, a copy of the s1 script); dataset, GT
scale, metrics, and contract guard imported unmodified from the unmodified
`phase1/harness/frozen_eval.py` (`frozen_eval_best.json` /
`frozen_eval_final.json`, `strict_ok: true`, 72 keys, 423586 params).

Note: the final-epoch snap scores slightly better (2.3018) than the best-val
snap (2.4122). The best-val snap remains the primary per the frozen checkpoint
policy (selection used the train-time 10-scene curve only, never the 40-scene
score); both are recorded.

## Stereo-path liveness (full 40-scene control, best checkpoint; measured though not required)

Substituting shift=none at eval time (keeping regression_normalize=True, same
weights) degrades the frozen score from 2.4122 EPE / 17.40% D1 to **26.3412
EPE / 97.97% D1** (`liveness_shift_none.json`) — consistent with seeds 0 and 1.

## Delta vs ARM K + verdict

Signed delta (best-val EPE): 2.4121953 − 5.5271927 = **−3.1149974 px**
(better).

**Verdict: ACCEPTED vs the ARM K bar (≤ 4.5271927); counts toward the
three-seed confirmation.** No tuning was done.
