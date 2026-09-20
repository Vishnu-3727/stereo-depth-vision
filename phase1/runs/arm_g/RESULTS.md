# ARM G — results (ARM K + asymmetric photometric augmentation)

## Hypothesis

Asymmetric photometric augmentation (independent left/right
brightness/contrast/gamma draws) regularizes the ARM K600 overfitting
signature — train loss 8.8 → 2.71 while 10-scene val never improved after
epoch 350 on 160 training scenes — without touching geometry, so it stays
exact under left-view-only GT. Expected: frozen 40-scene EPE improves vs
ARM K (5.5271927) by a material margin (bar: ≥ 1.0 px).

## Variable (one)

Per sample, independently to left and right (asymmetric — that is the
point): brightness × U(0.8,1.2) → contrast (x−mean)×U(0.8,1.2)+mean →
gamma x^U(0.8,1.2), in [0,1] space BEFORE ImageNet normalization, clamped
to [0,1] after each step. Geometry untouched: no flip/scale/rotation/crop
change; disparity + valid mask unmodified. The gain jitter sigma=0.1
STAYS as in ARM K (frozen recipe); this augmentation is added ON TOP and
is NOT a gain-jitter replacement. `--photo-prob` 1.0 (always on).

Frozen: seed 0, batch 2, random 256x512 crop, Adam (0.9,0.999), LR 1e-3,
cosine to 0 over 200 epochs, 200 epochs, masked smooth-L1 beta=1.0 valid
gt>0 and gt<176, ImageNet norm, no BN, fp32, same split + overlap
assertion, same DataLoader generator/worker_init_fn, same best+final
checkpoint policy. Diff of `phase1/scripts/train_arm_k.py` vs
`phase1/scripts/train_arm_g.py` confirms only the augmentation, `--epochs`/
`--photo-prob` flags, and run/file names differ.

## Self-check (before training) — ALL PASS, 20 samples each

1. Disparity tensor and valid mask bit-identical before/after augmentation
   (photo_prob 1.0 vs 0.0 on the same crop): PASS — the arm is exact under
   left-view-only GT.
2. Augmented images finite (no NaN/inf) and inside the valid
   ImageNet-normalized range: PASS.
3. Left and right receive DIFFERENT parameter draws on some sample: PASS —
   the augmentation is genuinely asymmetric.
4. `--photo-prob 0.0` pipeline bit-identical to ARM K's, sample for sample
   (left, right, disparity): PASS — the only change is the augmentation.

## Training

200 epochs, wall 4255.5 s. Train loss 11.40 → ~5.0; 10-scene val EPE never
below 13.67 (best @epoch 140). For comparison ARM K reached 10-scene 8.12
@epoch 180 with final train loss ~2.6 — the augmentation made optimization
markedly harder (photometric mismatch weakens the matching cue), not just
better-regularized.

## Frozen scores (unmodified `phase1/harness/frozen_eval.py`, contract_match true, 3,802,797 px, 40 scenes)

| Snapshot | EPE | D1 | RMSE | sha256 |
|---|---|---|---|---|
| best-val (ep 140) | 10.4747403 | 78.6592605% | 14.9776238 | 8a60a907…3e68303 |
| final (ep 199) | 11.2448508 | 79.3379452% | 15.9708695 | dfa0d923…7255336d |

## Delta vs ARM K + verdict

Signed delta (best-val EPE): 10.4747403 − 5.5271927 = **+4.9475476 px**
(worse). Materiality bar ≥ 1.0 px is exceeded in the wrong direction.

**Verdict: hypothesis REFUTED.** Always-on ±20% asymmetric photometric
mismatch at this data scale (160 scenes) degrades frozen EPE by ~4.9 px
rather than regularizing. The augmentation destroyed more matching signal
than the overfitting it was meant to cure.

## Rejected experiment — ARM F (horizontal flip, never executed)

ARM F (horizontal flip augmentation) was rejected before execution because
KITTI provides left-view-only ground truth: a horizontal flip mirrors the
image but the disparity field cannot be mirrored into a valid right-view
consistent target — the flipped pair's geometry no longer matches the
stored left-view GT, so the arm is inexact by construction. No code was
written and no training was run; the rejection with this reason is the
complete ARM F record.
