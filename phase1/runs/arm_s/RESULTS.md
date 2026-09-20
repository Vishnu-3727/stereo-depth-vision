# ARM S — results (ARM K + real disparity search via cost_volume_shift=right)

## Hypothesis

Restoring an actual disparity search in the cost volume gives the aggregation
network genuine per-candidate matching evidence, and reduces frozen-contract
EPE by at least 1.0 px versus ARM K (5.5271927).

## Intervention (one)

cost_volume_shift: none -> right, and nothing else. New `shift_right(x, k)`
helper (`F.pad(x, (k, 0))[..., :width]`) applied to the RIGHT features, so
candidate k computes `cost_k(x) = left(x) - right(x - k)`, indexed in the LEFT
frame as KITTI left-view GT requires. Modes none and left are byte-identical
in behaviour; `StereoNetConfig` default stays none. Diff of
`phase1/scripts/train_arm_k.py` vs `phase1/scripts/train_arm_s.py` confirms
only the shift, run/file names, record strings, and the step-5-mandated
`ARM_S_EPOCHS`/`ARM_S_OUT_DIR` env overrides differ (unset in the real run:
200 epochs, `phase1/runs/arm_s`).

## Frozen variables

Seed 0, batch 2, random 256x512 crops, gain jitter sigma=0.1, Adam
(0.9,0.999), LR 1e-3, cosine annealing to 0 over 200 epochs, 200 epochs,
masked smooth-L1 beta=1.0 valid gt>0 and gt<176, ImageNet norm, no BN, fp32,
split hailo_calib (160) train / hailo_val (40) eval with the overlap-0
assertion, same DataLoader generator + worker_init_fn, same best+final
checkpoint policy.

## Self-check (before training) — ALL PASS

`tests/test_cost_volume_shift.py`, 3 passed: (a) shift=none still yields 12
bit-identical slices (max abs diff to slice 0 exactly 0.0 — ARM K unchanged);
(b) shift=right slices are not all identical; (c) on a synthetic pair with
right = left rolled by +5 columns, per-pixel argmin of the L1 cost over
candidates equals 5 in the interior — sign convention proved, LEFT-frame
indexed as required.

## Training

200 epochs, wall 3669.9 s. Train loss 11.80 -> ~4.62; 10-scene val EPE never
below 13.94 (best @epoch 140). For comparison ARM K reached 10-scene 8.12
@epoch 180 — the shift made optimization harder, not easier.

## Frozen scores (contract_match true, 3,802,797 px, 40 scenes)

| Snapshot | EPE | D1 | RMSE | sha256 |
|---|---|---|---|---|
| best-val (ep 140) | 10.8866094 | 77.0809486% | 15.8076999 | b28256f8…54874f176 |
| final (ep 199) | 10.8518055 | 76.9303226% | 15.7458957 | c244b907…73f75de857 |

Eval-config note (read before trusting the numbers): `score_checkpoint` in the
unmodified `phase1/harness/frozen_eval.py` builds `StereoNet(StereoNetConfig())`,
i.e. shift=none, and checkpoints store weights only — so the literal harness
would evaluate ARM S weights through a different forward graph than the one
trained. The scores above use a line-for-line mirror of `score_checkpoint`
with `StereoNetConfig(cost_volume_shift="right")`; dataset, GT scale, metrics,
and contract guard are imported unmodified from `frozen_eval.py`
(`frozen_eval_best.json` / `frozen_eval_final.json`, with `strict_ok: true`,
72 keys, 423586 params). The literal-harness numbers were also recorded
(`frozen_eval_{best,final}_shift-none.json`): they are BIT-IDENTICAL to the
faithful scores on all 40 scenes. So the comparison against ARM K is valid
either way.

Why identical: stage dump of the trained weights shows the cost volumes DO
differ by shift (probe means 10.65 vs 21.97), but `aggregated_cost` explodes
to ~1e6 scale, the soft-argmin saturates, and `disparity_initial` is exactly
11.0 everywhere (std 0.0). The refinement then predicts from the left image
plus a constant. The trained ARM S model is a monocular-from-constant
predictor whose stereo path contributes nothing — operating from constant
11.0 where ARM K's degenerate volume yields uniform 5.5. Genuine matching
evidence was supplied; the network saturated it away and learned around it.

## Delta vs ARM K + verdict

Signed delta (best-val EPE): 10.8866094 − 5.5271927 = **+5.3594167 px**
(worse). Final-epoch delta: 10.8518055 − 5.5271927 = +5.3246128 px (worse).

Decision rule: EPE <= 4.5271927 => accepted; 4.5271927 < EPE < 5.5271927 =>
inconclusive; EPE >= 5.5271927 => REFUTED.

**Verdict: hypothesis REFUTED.** Restoring the disparity search degrades
frozen EPE by ~5.4 px rather than improving it. Nothing was retuned to rescue
it and no follow-up run is launched.
