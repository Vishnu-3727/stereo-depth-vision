# ARM T — results (ARM K + scale-invariant soft-argmin via regression_normalize)

## Hypothesis

The soft-argmin is scale-sensitive: at the cost magnitudes the aggregation
network actually produces it saturates, destroying sub-pixel precision and the
gradient along the disparity axis. Normalising the cost across the disparity
axis before the softmax makes the readout scale-invariant and un-saturates it,
reducing frozen-contract EPE by at least 1.0 px versus ARM K (5.5271927).

## Intervention (one)

`regression_normalize`: none -> True, and nothing else. In
`src/models/stereonet/regression.py` the cost is standardised across the
disparity axis immediately before the softmax:

    cost_n = (cost - cost.mean(dim, keepdim=True)) / (cost.std(dim, keepdim=True) + 1e-6)
    weights = softmax(-cost_n, dim)

Parameter-free (423,586 params, 72 state keys, `strict_ok: true`), gated behind
`StereoNetConfig.regression_normalize: bool = False` whose default is
unchanged, so ARM K is exactly reproducible. `cost_volume_shift` stays none
(ARM K's value). Diff of `phase1/scripts/train_arm_k.py` vs
`phase1/scripts/train_arm_t.py` confirms only the config line, run/file names,
record strings, and the step-mandated `ARM_T_EPOCHS`/`ARM_T_OUT_DIR` env
overrides differ (unset in the real run: 200 epochs, `phase1/runs/arm_t`).

## Frozen variables

Seed 0, batch 2, random 256x512 crops, gain jitter sigma=0.1, Adam
(0.9,0.999), LR 1e-3, cosine annealing to 0 over 200 epochs, 200 epochs,
masked smooth-L1 beta=1.0 valid gt>0 and gt<176, ImageNet norm, no BN, fp32,
split hailo_calib (160) train / hailo_val (40) eval with the overlap-0
assertion, same DataLoader generator + worker_init_fn, same best+final
checkpoint policy.

## Self-check (before training) — ALL PASS

`tests/test_regression_normalize.py`, 3 passed: (a) with
`regression_normalize=False` the output is bit-identical to the current
`soft_argmin` on random input (ARM K untouched); (b) the normalised readout
gives the same disparity for C and 37*C within 1e-4 while the unnormalised one
does not; (c) at magnitudes ~1e7 with a spatially varying argmin, the
normalised readout has std > 0 and is not pinned to a single candidate.

## Training

200 epochs, wall 3679.9 s. Train loss 11.07 -> ~3.15; 10-scene val EPE best
11.5774 @epoch 165. For comparison ARM K reached 10-scene 8.12 @epoch 180 —
the normalised readout optimised worse on the train-time curve too.

## Frozen scores (contract_match true, 3,802,797 px, 40 scenes)

| Snapshot | EPE | D1 | RMSE | sha256 |
|---|---|---|---|---|
| best-val (ep 165) | 8.2702542 | 68.2488705% | 12.7972958 | bc11fe63…53d5b |
| final (ep 199) | 8.4647908 | 68.4849862% | 13.0317812 | f7c6610c…9dcb65 |

Eval-config note (read before trusting the numbers): `score_checkpoint` in the
unmodified `phase1/harness/frozen_eval.py` builds `StereoNet(StereoNetConfig())`,
i.e. `regression_normalize=False`, and checkpoints store weights only — so the
literal harness would evaluate ARM T weights through a different forward graph
than the one trained. The scores above use a line-for-line mirror of
`score_checkpoint` with `StereoNetConfig(regression_normalize=True)`; dataset,
GT scale, metrics, and contract guard are imported unmodified from
`frozen_eval.py` (`frozen_eval_best.json` / `frozen_eval_final.json`, with
`strict_ok: true`, 72 keys, 423586 params). The literal-harness numbers were
also recorded (`frozen_eval_{best,final}_default.json`): best 13.2878947 /
final 13.2599504. They DIFFER from the faithful scores, confirming the flag
changes the forward graph — the opposite of ARM S, whose scores were
bit-identical across shifts because its readout was saturated.

## Saturation report (hailo_val scene 0, best checkpoint)

| Eval graph | agg_cost |max| | disp_init mean | disp_init std | disp_init range |
|---|---|---|---|---|
| normalize=True (trained) | 2.12e15 | 4.2378 | 2.1981 | 1.48–9.53 |
| normalize=False (literal) | 2.12e15 | 3.0242 | 4.1223 | 0.00–11.00 |

The saturation is GONE under the trained graph: `disparity_initial` varies
spatially (std 2.20, spanning candidates ~1.5–9.5) instead of collapsing to the
constant 11.0 everywhere as in ARM S. But two observations cut against the
hypothesis: (1) the aggregation network, freed from any scale penalty, emits
costs of order 1e15 — ~50,000x larger than ARM S's 1e8 — so the normalisation
is doing all the work and the learned cost shape is extreme; (2) the
un-saturated readout still scores 8.27 EPE, far above ARM K's 5.53. This is the
"useful negative" outcome: the readout un-saturates but EPE does not improve —
the bottleneck is not (only) the readout scale.

## Delta vs ARM K + verdict

Signed delta (best-val EPE): 8.2702542 − 5.5271927 = **+2.7430615 px**
(worse). Final-epoch delta: 8.4647908 − 5.5271927 = +2.9375981 px (worse).

Decision rule: EPE <= 4.5271927 => accepted; 4.5271927 < EPE < 5.5271927 =>
inconclusive; EPE >= 5.5271927 => REFUTED.

**Verdict: hypothesis REFUTED.** Scale-invariant readout un-saturates the
initial disparity but degrades frozen EPE by ~2.7 px rather than improving it.
Nothing was retuned to rescue it and no follow-up run is launched.
