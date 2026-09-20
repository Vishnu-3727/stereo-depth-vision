# ARM U — results (ARM K + disparity search AND scale-invariant readout)

## Hypothesis

A real disparity search is only testable through a readout that does not
saturate. With the scale-invariant readout in place, restoring the LEFT-frame
disparity search gives the aggregation network usable per-candidate matching
evidence, reducing frozen-contract EPE by at least 1.0 px versus ARM K
(5.5271927).

## Intervention (two flags, and why that is legitimate here)

`StereoNetConfig(cost_volume_shift="right", regression_normalize=True)`.
This is deliberately TWO changes versus ARM K. That is justified here and only
here: each flag has already been measured alone against ARM K (ARM S:
cost_volume_shift=right alone, frozen EPE 10.8866, REFUTED; ARM T:
regression_normalize=True alone, frozen EPE 8.2703, REFUTED), so the pair is
the mechanism under test, not an uncontrolled combination. No third change of
any kind: no source changes (both flags already exist in
`src/models/stereonet`), no default changed, no hyperparameter changed.
Diff of `phase1/scripts/train_arm_k.py` vs `phase1/scripts/train_arm_u.py`
confirms the config line and naming are the only functional differences
(plus the step-mandated `ARM_U_EPOCHS`/`ARM_U_OUT_DIR` env overrides, unset in
the real run: 200 epochs, `phase1/runs/arm_u`).

## Frozen variables

Seed 0, batch 2, random 256x512 crops, gain jitter sigma=0.1, Adam
(0.9,0.999), LR 1e-3, cosine annealing to 0 over 200 epochs, 200 epochs,
masked smooth-L1 beta=1.0 valid gt>0 and gt<176, ImageNet norm, no BN, fp32,
split hailo_calib (160) train / hailo_val (40) eval with the overlap-0
assertion, same DataLoader generator + worker_init_fn, same best+final
checkpoint policy.

## Training

200 epochs, wall 3679.1 s. Train loss 10.82 -> ~0.87; 10-scene val EPE best
3.9355 @epoch 199. For comparison the train-time curves reached 10-scene 8.12
(ARM K @180), 13.94 (ARM S @140), 11.58 (ARM T @165) — ARM U optimised better
on the train-time curve too, unlike either parent flag alone.

## Frozen scores (contract_match true, 3,802,797 px, 40 scenes)

| Snapshot | EPE | D1 | RMSE | sha256 |
|---|---|---|---|---|
| best-val (ep 199) | 2.2866369 | 15.3067597% | 6.1837683 | b115dae0…e32596e5167 |
| final (ep 199) | 2.2866369 | 15.3067597% | 6.1837683 | 113e3865…3637a10c91a99f965 |

Best-val and final carry identical weights (weight_sha16 929e621e53aa2c0a in
both frozen-eval records; file sha differs by pickle serialisation noise
only), so the two scores coincide. Epoch 199 was both the last epoch and the
10-scene optimum.

Eval-config note (read before trusting the numbers): `score_checkpoint` in the
unmodified `phase1/harness/frozen_eval.py` builds `StereoNet(StereoNetConfig())`,
i.e. shift=none and regression_normalize=False, and checkpoints store weights
only — so the literal harness would evaluate ARM U weights through a different
forward graph than the one trained. The scores above use a line-for-line mirror
of `score_checkpoint` with `StereoNetConfig(cost_volume_shift="right",
regression_normalize=True)`; dataset, GT scale, metrics, and contract guard are
imported unmodified from `frozen_eval.py` (`frozen_eval_best.json` /
`frozen_eval_final.json`, with `strict_ok: true`, 72 keys, 423586 params).

## Saturation report + stereo-path liveness (hailo_val scene 0, best checkpoint)

| aggregated_cost |max| | disp_init mean | disp_init std | disp_init range |
|---|---|---|---|---|
| 9.39e8 | 3.3966 | 1.8187 | 1.50–8.74 |

`disparity_initial` varies spatially (std 1.82, spanning candidates ~1.5–8.7):
not pinned to a candidate, not collapsed to the ARM S constant 11.0. The
aggregation costs sit at ~1e9 scale — the same regime that saturated ARM S's
readout — and the normalisation is what keeps the readout alive.

Liveness (the direct test ARM S failed): substituting shift=none at eval time
(keeping regression_normalize=True, same weights, scene 0) changes the final
output by up to ~57.5 px max abs diff — NOT bit-identical. The stereo path is
LIVE: the disparity search contributes to the prediction.

## Delta vs ARM K + verdict

Signed delta (best-val EPE): 2.2866369 − 5.5271927 = **−3.2405558 px**
(better). Final-epoch delta is identical (same weights).

Decision rule: EPE <= 4.5271927 => accepted; 4.5271927 < EPE < 5.5271927 =>
inconclusive; EPE >= 5.5271927 => REFUTED.

**Verdict: ACCEPTED — flag for multi-seed confirmation.** The composition beats
ARM K by 3.24 px, clears the 1.0 px bar with margin, and is the first result in
the programme to do so. Neither parent flag helped alone (ARM S +5.36,
ARM T +2.74); together they compose: the normalisation removes exactly the
saturation that made ARM S's disparity search untestable, and the search gives
the un-saturated readout genuine matching evidence to work with.

Stereo-path liveness reported separately from the EPE verdict, as required:
path LIVE (shift substitution moves output ~57.5 px), and EPE accepted. No
retuning was done and no follow-up run is launched here — the flagged next step
is a multi-seed confirmation, which is a new experiment, not this one.

## Seed confirmation (ARM U-S1, ARM U-S2 — appended, seed-0 numbers above untouched)

The exact ARM U recipe was re-run on seeds 1 and 2 via
`phase1/scripts/train_arm_u_seed.py` (diff vs `train_arm_u.py` is exactly the
`ARM_U_SEED` / `ARM_U_OUT_DIR` plumbing; seed 1 → `phase1/runs/arm_u_s1`, seed
2 → `phase1/runs/arm_u_s2`). Frozen best-val-snap scores (contract_match true,
3,802,797 px):

| Seed | EPE | D1 | Best epoch | Liveness (shift=none EPE) |
|---|---|---|---|---|
| 0 (this run) | 2.2866369 | 15.3067597% | 199 | 24.7448 |
| 1 | 2.4207853 | 18.4118689% | 175 | 21.3743 |
| 2 | 2.4121953 | 17.3953803% | 145 | 26.3412 |

Mean best-val EPE across 3 seeds: 2.3732058. Spread (max − min): 0.1341484 px.

Run-to-run spread across 3 seeds is 0.1341 px; the improvement over ARM K is
3.2405558 px, which is 24.2 times the spread.

Decision rule: all 3 seeds ≤ 4.5271927 EPE ⇒ ARM U CONFIRMED as the new
incumbent and the Phase-2 entry candidate; some pass / some fail ⇒ NOT
confirmed (seed 0 may have been fortunate); spread ≥ 1.0 px ⇒ the improvement
is not separable from seed noise at this bar. Outcome: all three seeds clear
the bar (2.2866, 2.4208, 2.4122 ≤ 4.5271927) and the spread (0.1341 px) is far
below 1.0 px — **ARM U CONFIRMED**. The 3.24 px improvement over ARM K is ~24×
the run-to-run spread, so it is separable from seed noise. Seed 1's liveness
control replicates structurally (2.4208 → 21.3743 EPE under shift=none
substitution); seed 2 agrees (2.4122 → 26.3412).
