# AUDIT STAGE 1 — BASELINE FACTS (our own model + training recipe)

Scope: READ-ONLY facts from the Step B files. No training, no inference.
Citations are `file:line` into the Step B files only.
Where `phase1/scripts/train_arm_v.py` overrides the deployed baseline defaults,
both values are recorded explicitly (baseline default first, ARM V override second).

## 1. Feature extractor (type, layers, channels, downsample factor)

- Type: shared / Siamese tower applied to both images; `368x1232 -> 23x77, 32 channels`.
  - `src/models/stereonet/feature_extractor.py:26-27`
  - `src/models/stereonet/stereonet.py:10-11`
- Layers:
  - Four `5x5 stride-2` convolutions, `3->32` then `32->32` three times, padding 2,
    with **no activation between them**.
    - `src/models/stereonet/feature_extractor.py:5-6`
    - `src/models/stereonet/feature_extractor.py:37-43`
  - Six residual blocks at 32 channels.
    - `src/models/stereonet/feature_extractor.py:7`
    - `src/models/stereonet/feature_extractor.py:45`
    - Defaults `downsample_levels=4, residual_blocks=6`:
      `src/models/stereonet/stereonet.py:49-50`
  - One `3x3` convolution, `32->32`.
    - `src/models/stereonet/feature_extractor.py:8`
    - `src/models/stereonet/feature_extractor.py:46`
- Channels: 32 (`feature_channels=32`).
  - `src/models/stereonet/stereonet.py:48`
  - `src/models/stereonet/feature_extractor.py:30-31`
- Downsample factor: 16 (`2**4`, `1/16` resolution).
  - `src/models/stereonet/stereonet.py:49`
  - `src/models/stereonet/feature_extractor.py:48-51`
  - `src/models/stereonet/stereonet.py:68-69`
- ARM V override (training recipe only, not baseline): `downsample_levels=3` → stride 8.
  - `phase1/scripts/train_arm_v.py:131-132`

## 2. Feature normalization (BN / GN / none)

- None. The downsample stack lists only `nn.Conv2d` layers with no norm layers,
  followed by `ResBlock` sequence and output conv.
  - `src/models/stereonet/feature_extractor.py:37-46`
- Training recipe records the deviation explicitly: `no batch normalisation
  (BN-folded exported artifact), carried into all arms`.
  - `phase1/scripts/train_arm_v.py:157`
- No GroupNorm / BatchNorm / LayerNorm appears in any Step B file.

## 3. Cost volume construction (subtraction / correlation / concat / group-wise)

- Baseline default: subtraction (`cost_volume_method="subtract"`), shift `"none"`.
  - `src/models/stereonet/stereonet.py:52-56`
  - `src/models/stereonet/cost_volume.py:151-159`
- `subtract` computes `anchor - other` per level; `concat` (`torch.cat([anchor, other], dim=1)`,
  doubling channels) is implemented but not the default.
  - `src/models/stereonet/cost_volume.py:110-115`
- Baseline shift `"none"` reproduces the reference no-op: pad zeros on the right
  then slice `[0:W]`, returning input unchanged, so all 12 slices are identical
  `left - right` (no disparity search).
  - `src/models/stereonet/cost_volume.py:59-69`
  - `src/models/stereonet/cost_volume.py:103-105`
  - `src/models/stereonet/stereonet.py:21-26`
- Group-wise path exists (`groups`, optional `Conv2d(channels, groups, 1)`) but
  baseline default is `cost_volume_groups=0` (disabled).
  - `src/models/stereonet/stereonet.py:65`
  - `src/models/stereonet/cost_volume.py:160-171`
- ARM V override: `cost_volume_shift="right"`.
  - `phase1/scripts/train_arm_v.py:131-132`

## 4. Cost volume tensor shape and dimensionality (4D or 3D)

- 4D cost volume (channels + disparity + 2D spatial) stored as a 5D tensor
  `(B, C, D, H, W)`, ready for `Conv3d`. Baseline example: `(B, 32, 12, 23, 77)`.
  - `src/models/stereonet/cost_volume.py:72-79`
  - `src/models/stereonet/cost_volume.py:118-119`
  - `src/models/stereonet/stereonet.py:12`
- Aggregation consumes `(B, C, D, H, W)` and emits `(B, D, H, W)`.
  - `src/models/stereonet/aggregation.py:37-40`

## 5. Number of disparity hypotheses and their spacing in full-resolution pixels

- Baseline: 12 hypotheses (`num_disparities=12`, noted as `192 // 16`).
  - `src/models/stereonet/stereonet.py:51`
- Spacing: one feature-stride = 16 full-resolution pixels; candidates are
  `16px` apart.
  - `src/models/stereonet/regression.py:15-16`
  - `src/models/stereonet/stereonet.py:68-69`
- Range: largest representable disparity `(num_disparities - 1) * feature_stride`
  = `11 * 16 = 176px`.
  - `src/models/stereonet/stereonet.py:72-74`
- ARM V override: `num_disparities=24` with stride 8 → spacing `8px`, range `184px`
  (`23*8`), recorded in the file header and config.
  - `phase1/scripts/train_arm_v.py:3-8`
  - `phase1/scripts/train_arm_v.py:131-132`
  - `phase1/scripts/train_arm_v.py:138-139`

## 6. Aggregation (2D or 3D conv, depth, channels)

- 3D conv (`Conv3d`), not 2D.
  - `src/models/stereonet/aggregation.py:31-35`
- Depth: 4 filtering layers + 1 output layer. Four `Conv3d(32->32, 3x3x3,
  padding 1)` each followed by LeakyReLU (slope 0.01), then final
  `Conv3d(32->1, 3x3x3, padding 1)` and squeeze of channel axis.
  - `src/models/stereonet/aggregation.py:1-6`
  - `src/models/stereonet/aggregation.py:26-35`
- Channels: 32 throughout filtering (`in_channels=32, channels=32, num_layers=4`).
  - `src/models/stereonet/aggregation.py:26`
  - Baseline wiring `aggregation_layers=4`:
    `src/models/stereonet/stereonet.py:57`

## 7. Readout (soft-argmin / classification / regression, any temperature)

- Soft-argmin: expected disparity index under `softmax(-cost)`.
  - `src/models/stereonet/regression.py:33-36`
  - `src/models/stereonet/regression.py:49-53`
- No temperature parameter in code (implicit temperature = 1; negation makes it
  arg*min*, softmax makes it continuous/sub-pixel).
  - `src/models/stereonet/regression.py:8-13`
  - `src/models/stereonet/regression.py:49`
- Ordering: upsample the **whole cost tensor** first (`bilinear, align_corners`)
  from `1/16` (`(B,12,23,77)`) to full res (`(B,12,368,1232)`), then
  `Neg -> Softmax(dim=1) -> Mul(index grid) -> ReduceSum`.
  - `src/models/stereonet/regression.py:3-9`
  - `src/models/stereonet/regression.py:64-67`
  - Flag default `upsample_first=True`:
    `src/models/stereonet/regression.py:59-62`
  - Baseline flag `upsample_before_argmin=True`:
    `src/models/stereonet/stereonet.py:59`
- Output units are disparity *candidates* (`0..11`), not pixels; scale is left to
  upsampling/refinement.
  - `src/models/stereonet/stereonet.py:21-26`
  - `src/models/stereonet/regression.py:33-41`
- Optional standardisation across disparity axis (`normalize=True`, ARM T) exists
  but baseline default is `regression_normalize=False`.
  - `src/models/stereonet/regression.py:42-48`
  - `src/models/stereonet/stereonet.py:64`
- ARM V override: `regression_normalize=True`.
  - `phase1/scripts/train_arm_v.py:131-132`

## 8. Refinement (structure, number of stages)

- Single refinement stage (one `Refinement` module producing a residual that is
  added to the initial disparity, followed by ReLU).
  - `src/models/stereonet/stereonet.py:106-110`
  - `src/models/stereonet/stereonet.py:122-125`
- Structure:
  - Concatenate disparity estimate with left RGB: `1 + 3 = 4` channels.
    - `src/models/stereonet/refinement.py:6`
    - `src/models/stereonet/refinement.py:51-53`
  - `Conv2d(4->32, 3x3)`.
    - `src/models/stereonet/refinement.py:7`
    - `src/models/stereonet/refinement.py:47`
  - Six residual blocks with dilations `(1, 2, 4, 8, 1, 1)`.
    - `src/models/stereonet/refinement.py:8`
    - `src/models/stereonet/refinement.py:36`
    - `src/models/stereonet/refinement.py:48`
    - Baseline `refinement_dilations=(1, 2, 4, 8, 1, 1)`:
      `src/models/stereonet/stereonet.py:58`
  - `Conv2d(32->1, 3x3)` emitting the residual (not the final map).
    - `src/models/stereonet/refinement.py:9`
    - `src/models/stereonet/refinement.py:49`
    - `src/models/stereonet/refinement.py:51-56`
- Final `ReLU` after add (baseline `final_relu=True`).
  - `src/models/stereonet/stereonet.py:60`
  - `src/models/stereonet/stereonet.py:123-125`

## 9. Loss function

- Masked smooth L1, `beta=1.0`, valid mask `gt > 0 and gt < max_disparity`.
  - `phase1/scripts/train_arm_v.py:153`
  - Call site `masked_smooth_l1(out, disparity, max_disparity=float(config.max_disparity_px))`:
    `phase1/scripts/train_arm_v.py:177`

## 10. Optimizer, learning rate, schedule, epochs, batch size

- Optimizer: Adam, `betas=(0.9, 0.999)`.
  - `phase1/scripts/train_arm_v.py:135`
  - Config record `optimizer: Adam betas=(0.9, 0.999)`:
    `phase1/scripts/train_arm_v.py:150`
- Weight decay: UNKNOWN (not stated in the Step B files; no `weight_decay`
  argument appears in the Adam construction at `phase1/scripts/train_arm_v.py:135`).
- Learning rate: `1e-3`.
  - `phase1/scripts/train_arm_v.py:50`
  - `phase1/scripts/train_arm_v.py:151`
- Schedule: cosine annealing to 0 over 200 epochs (`CosineAnnealingLR, T_max=EPOCHS`).
  - `phase1/scripts/train_arm_v.py:52`
  - `phase1/scripts/train_arm_v.py:136`
  - `phase1/scripts/train_arm_v.py:152`
- Epochs: 200 (overridable by `ARM_V_EPOCHS`, smoke test `=2`).
  - `phase1/scripts/train_arm_v.py:48`
- Batch size: 2.
  - `phase1/scripts/train_arm_v.py:49`
  - `phase1/scripts/train_arm_v.py:146`

## 11. Crop size and crop sampling strategy

- Crop size: `256x512` (`CROP_H=256, CROP_W=512`).
  - `phase1/scripts/train_arm_v.py:47`
  - `phase1/scripts/train_arm_v.py:143-144`
- Sampling: uniform random crop per sample (`rng.integers(0, h - CROP_H + 1)`,
  `rng.integers(0, w - CROP_W + 1)`), seeded (`seed=0`).
  - `phase1/scripts/train_arm_v.py:58-70`
  - Seed `SEED=0`: `phase1/scripts/train_arm_v.py:51`

## 12. Augmentation

- Random crop + independent per-image gain jitter (`sigma=0.1`); no horizontal flip.
  - `phase1/scripts/train_arm_v.py:56`
  - `phase1/scripts/train_arm_v.py:74-76`
  - `phase1/scripts/train_arm_v.py:154`
- Normalization wrapper `normalize(...)` applied to left/right crops.
  - `phase1/scripts/train_arm_v.py:72-73`

## CLONE RESULTS

- OpenStereo (`https://github.com/XiandaGuo/OpenStereo`): CLONED
- aanet (`https://github.com/haofeixu/aanet`): CLONED
- Fast-ACVNet (`https://github.com/gangweiX/Fast-ACVNet`): CLONED
- Selective-Stereo (`https://github.com/Windsrain/Selective-Stereo`): CLONED
- mobilestereonet (`https://github.com/cogsys-tuebingen/mobilestereonet`): CLONED
- StereoNet (`https://github.com/neka-nat/StereoNet`): CLONED
- Failed: none. No failing clone line.
