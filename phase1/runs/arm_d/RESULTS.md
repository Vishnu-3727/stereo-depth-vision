# ARM D — Driving-only synthetic pretraining -> KITTI fine-tune (SCORED)

## Hypothesis

Driving-domain synthetic pretraining (4400 SceneFlow Driving pairs) followed by
the ARM K KITTI recipe reduces frozen EPE by >= 1.0 px vs ARM K best (5.5271927).

## THE CEILING (read first — not softened)

ARM D is a PROXY. It is evidence ONLY about driving-domain synthetic
pretraining on 4400 pairs. It is NOT evidence about full SceneFlow
pretraining, and ARM P remains BLOCKED/NOT RUN regardless of this outcome.

A negative ARM D does NOT establish that SceneFlow pretraining fails:
Driving is ~12% of the upstream corpus, single-domain, and the upstream
recipe trains on FlyingThings3D + Monkaa + Driving together. No conclusion
about full-corpus pretraining follows from this result in either direction.

## Stage 1 — pretraining (Driving only)

- Corpus: SceneFlow Driving only, 4400 pairs
  (frames_cleanpass + disparity PFM, 0 skipped triplets).
  Source on disk at run time: data/sceneflow/driving/extracted/.
- Recipe as run (phase1/scripts/pretrain_arm_d.py, 20 epochs, seed 0, cuda,
  fp32): random 256x512 crops; batch 4; RMSprop lr=1e-3 weight_decay=1e-4;
  ExponentialLR gamma=0.9; smooth L1 with valid = gt < 160 (upstream mask,
  no gt>0 floor); random crop only, no jitter, no flip (upstream
  augment=False); PyTorch default random initialisation; checkpoint
  selection by lowest mean train-epoch loss.
- Every deviation from upstream sceneflow-pretrain.py (as recorded in
  arm_d_pretrain_record.json):
  - batch 16 -> 4: upstream batch 16 OOMs 8 GB VRAM at 256x512
  - num_workers 12 -> 0: frozen determinism recipe, as ARM K
  - seed 1 -> 0: frozen seed, as ARM K
  - no DataParallel: single-GPU run
  - scheduler stepped after each epoch; upstream steps before
    (LR decays before epoch 0 there)
  - checkpoint selection by lowest mean train-epoch loss; upstream reports
    FlyingThings3D TEST EPE instead
  - no checkpoint resume: fresh run, as ARM K
  - model is this repo's StereoNet reimplementation, not upstream stereonet()
  - no gain jitter / no augmentation: upstream pretrain path applies none
    (augment=False)
- Known deviation carried into all arms: no batch normalisation
  (BN-folded exported artifact).
- Checkpoints (unmodified):
  best sha256 247185800faec90f30029c1d0c8cea923ae4f5f05a653cb5578b3efb49b42e12
  (epoch 19); final sha256
  d563455f786fc2034c0314d0c8b88c0a52ec36335fd3b62074e65e77f9069c2f
  (epoch 19 of 20 -- best is final epoch here).
  Wall clock 9283.4 s (2.579 h), 20 epochs.
  Record `phase1/runs/arm_d_pretrain/arm_d_pretrain_record.json`, log
  `phase1/runs/arm_d_pretrain/training_log.jsonl`.

## Stage 2 — fine-tune (ARM K recipe, pretrained init)

Recipe identical to ARM K except initialization. Stated explicitly: the ONLY
differences between the Stage-2 config and the ARM K config are the
`initialisation` field (PyTorch defaults, random -> ARM D pretrained
checkpoint phase1\runs\arm_d_pretrain\arm_d_pretrain_best.pth, strict load,
key set asserted), the added `init_sha256` field, and the arm label.

Fields verified identical (compared field-by-field against
phase1/runs/arm_k/arm_k_record.json):

- dataset: kitti2015
- split: hailo_calib (scenes 0-159) train, hailo_val (160-199) eval, overlap 0
- split_enforcement: train_base split='hailo_calib' only; no hailo_val sample
  enters the optimizer loop
- resolution: [256, 512]; crop: random 256x512; disparity_range: 176
- batch_size: 2; precision: fp32; seed: 0; epochs: 200; device: cuda
- optimizer: Adam betas=(0.9, 0.999); learning_rate: 0.001
- schedule: cosine annealing to 0 over 200 epochs
- loss: masked smooth L1, beta=1.0, valid = gt > 0 and gt < max_disparity
- augmentation: random crop, independent per-image gain jitter sigma=0.1;
  no horizontal flip
- known_deviation: no batch normalisation (BN-folded exported artifact)

Init checkpoint sha256:
247185800faec90f30029c1d0c8cea923ae4f5f05a653cb5578b3efb49b42e12
(= Stage-1 best checkpoint).

Fine-tune checkpoints (unmodified):
best sha256 19a4721ee4546c09fd1a066f0beedf7eb0d5d766952ebc79f6a98c198fc46034;
final sha256 74da331db8f80de520e95b650c91edfff988eca7bd7769e82e2ab1bec9a88101.
Wall clock 3567.8 s (0.991 h), 200 epochs.
Script `phase1/scripts/finetune_arm_d.py`, log
`phase1/runs/arm_d/training_log.jsonl`, record
`phase1/runs/arm_d/arm_d_record.json`.

## Frozen scores (phase1/harness/frozen_eval.py, unmodified)

Guard: contract_match true, 3,802,797 valid pixels, 40 scenes, both snapshots.
Scored exactly as ARM K and ARM K600 were (phase1/scripts/score_arm.py).

| Snapshot | Checkpoint | Frozen EPE | Frozen D1 | RMSE | Valid px | Scenes |
|---|---|---|---|---|---|---|
| Best (PRIMARY, epoch 155, 10-scene val 13.2249) | arm_d_best.pth | 10.5748918 | 77.0048204% | 15.2754479 | 3,802,797 | 40 |
| Final (epoch 199, 10-scene val 13.3667) | arm_d_final.pth | 10.6663718 | 77.0223075% | 15.4485627 | 3,802,797 | 40 |

Full JSON: `phase1/runs/arm_d/frozen_eval_best.json`,
`phase1/runs/arm_d/frozen_eval_final.json`.

## Decision vs the >= 1.0 px materiality bar

delta = EPE_ARMK(5.5271927) - EPE_ARMD-best(10.5748918) = -5.0476991 px.

The sign is NEGATIVE: ARM D best is 5.0476991 px WORSE (higher EPE) than
ARM K. The required >= 1.0 px improvement did not occur; EPE moved the wrong
way by a material margin (5x the bar).

Verdict: Driving-domain synthetic pretraining HURT. Under this recipe,
4400-pair Driving-only pretraining followed by the ARM K fine-tune degrades
frozen EPE materially vs random initialisation. ARM K (5.5271927) remains the
incumbent. Best-vs-final snap gap here is 0.0915 px (below the bar); snapshot
choice moves nothing.

The 10-scene training-time val figures (best 13.2249 @ epoch 155, final
13.3667 @ epoch 199) are the monitoring curve, NOT the score, and are not
presented as the score.

## Observed training curves (as measured)

Stage 1 pretrain (mean train-epoch smooth-L1 loss, 20 epochs):
36.9533 (epoch 0) -> 26.3387 -> 21.7500 -> 20.0059 -> 18.4547 -> 17.0388
-> 15.7900 -> 15.0052 -> 14.3614 -> 13.8338 -> 13.1341 -> 13.0256
-> 12.6472 -> 12.4155 -> 12.1024 -> 11.8929 -> 11.7820 -> 11.5822
-> 11.4101 -> 11.2560 (epoch 19, best and final). Monotone decrease every
epoch; no val curve exists for this stage (checkpoint selection by lowest
mean train loss).

Stage 2 fine-tune (mean train-epoch loss; 10-scene val EPE/D1 every 5th epoch):
train loss 12.2897 (epoch 0) -> ~8.3 (epoch 10) -> ~7.1 (epoch 30)
-> ~5.9 (epoch 80) -> ~5.4 (epoch 90) -> ~4.8 (epoch 120) -> ~4.4 (epoch 140)
-> 4.3572 (epoch 155, best-val epoch) -> ~4.2 (epochs 160-184)
-> 4.1303 (epoch 199). 10-scene val EPE: 17.3914 (epoch 0), 19.3894 (epoch 5),
20.5797 (epoch 10), 15.7814 (epoch 30), 15.8186 (epoch 40), 15.2216 (epoch 50),
15.2046 (epoch 60), 13.5982 (epoch 80), 13.9909 (epoch 95), 13.3232 (epoch 140),
13.2249 (epoch 155, best), 13.6128 (epoch 160), 13.2254 (epoch 165),
13.4798 (epoch 180), 13.2486 (epoch 190), 13.3699 (epoch 195),
13.3667 (epoch 199, final).

INTERPRETATION (labelled as such, not a measured result): the fine-tune val
curve never approaches ARM K's 10-scene val levels (ARM K best 8.1202 @
epoch 180); the pretrained init starts Stage 2 at a similar train loss to
random init but converges to a worse held-out solution on the frozen 40-scene
contract.
