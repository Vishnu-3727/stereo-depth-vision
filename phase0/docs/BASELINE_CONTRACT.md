# PHASE 0 — FROZEN EVALUATION CONTRACT (Objective A)

Status: FROZEN. Every item below was verified against repository source code
(file + line cited). Nothing here is copied from a prior report without a
source check; where a prior claim was checked, the check is the evidence.

Method note: the candidate contract (KITTI 2015, 40-scene `hailo_val` slice,
`disp_occ_0`, 368x1232 top-left crop, 1/256 GT scaling, `gt > 0`, pooled pixel
evaluation, EPE in pixels, D1-all) was verified item by item. The repository
proves this contract with two corrections recorded below: (1) the vendor's
published 8.223 figure uses 1/255 + per-image averaging and is NOT this
contract — it is recorded separately and must never be substituted; (2) the
crop discards 7 bottom rows + 10 right columns of every KITTI frame, which is
part of the contract, not a neutral detail.

## PRIMARY DATASET

KITTI 2015 stereo training set, frames matching `_10` only (frame 11 has no
disparity ground truth).
Evidence: `src/datasets/kitti2015.py:122` (name filter `_10.png`).

## PRIMARY SCENES

40 scenes: the `hailo_val` split = training scenes index 160–199
(`names[160:]`, `CALIB_SPLIT_END = 160`).
Evidence: `src/datasets/kitti2015.py:34-35` (split boundary),
`src/datasets/kitti2015.py:125-126` (`hailo_val` slicing).
Verified live: the local copy holds 200 `_10` left images and 200
`disp_occ_0` files (`data/kitti2015/training/image_2`, `disp_occ_0`).

## IMAGE SIZE

368 x 1232 (height x width), fixed.
Evidence: `src/datasets/kitti2015.py:34` (`TARGET_H, TARGET_W = 368, 1232`);
`reference/hailo_model_zoo/stereonet.yaml:6-9,36` (input shape 368x1232x3 x2).

## CROP-PADDING

Pad bottom/right with zeros up to at least 368x1232, then crop 368x1232
anchored at the top-left. No resizing anywhere. Ground truth is padded and
cropped identically. For native KITTI 375x1242 frames no padding happens: 7
rows are removed from the bottom and 10 columns from the right.
Evidence: `src/datasets/kitti2015.py:51-68` (`pad_and_crop`);
cross-checked against the vendor source
`reference/upstream/extracted/hailo_model_zoo-master/hailo_model_zoo/core/preprocessing/stereonet_preprocessing.py:23-42`
(`pad_and_crop_tensor`, crop to bounding box at origin 0,0).
Contract consequence (recorded, not corrected):
`src/datasets/kitti2015.py:57-60` — the removed bottom rows carry the
closest road surface, i.e. the largest disparities; the crop is not neutral
with respect to the disparity distribution.

## GT SOURCE

`disp_occ_0` (occluded pixels INCLUDED — KITTI "D1-all").
A `disp_noc_0` variant exists for D1-noc only and is NOT part of this contract.
Evidence: `src/datasets/kitti2015.py:112-114` (gt_dir selection, Hailo uses
`disp_occ_0`); `scripts/exp_reproduce_hailo.py:72-76` (noc variant defined
separately).

## GT SCALE

1/256 (KITTI official: stored value = round(disparity * 256)).
The vendor's 1/255 divisor is a different protocol and is EXCLUDED from this
contract (effect measured in isolation: -0.055 headline points).
Evidence: `src/evaluation/metrics.py:29-33` (both scale constants + the
1.0039x statement); `src/datasets/kitti2015.py:71-87` (`read_disparity_png`
default `scale=256.0`, `scale=255.0` only to reproduce Hailo's convention).

## VALID MASK

`gt > 0`. A stored zero means no ground truth, never zero disparity. No
occlusion masking beyond the `occ` file choice; no max-disparity masking at
evaluation time (the `< max_disparity` mask exists in the TRAINING loss only).
Evidence: `src/evaluation/metrics.py:52-66` (pooled; `valid = gt > 0`
default); `src/evaluation/metrics.py:105,108-109` (per-image; `gt > 0`);
`src/losses/disparity.py:43-45` (training-only range mask).

## EPE DEFINITION

Mean absolute disparity error in pixels over all valid pixels (pooled):
`mean(|pred - gt|)`.
Evidence: `src/evaluation/metrics.py:71,78` (`err.mean()`).

## D1 DEFINITION

KITTI outlier rate in percent: fraction of valid pixels with
(err > 3.0 px) AND (err > 5% of gt), times 100.
Evidence: `src/evaluation/metrics.py:72-74,80` (both-tests-fail rule).

## AVERAGING METHOD

Pooled over all valid pixels of all 40 scenes (single fraction over
3,802,797 pixels). Per-image averaging is the vendor's protocol, NOT this
contract (effect measured in isolation: +0.014 points on this data).
Evidence: `src/evaluation/metrics.py:52-60` docstring (pooled);
`src/evaluation/metrics.py:203-250` (`demo()` proves per-image mean != pooled
on unequal pixel counts); `scripts/exp_reproduce_hailo.py:200-204`
(headline = per-image mean vs pooled.d1 selected by variant).

## DISPARITY UNITS

Pixels. Model output `(B,1,368,1232)` float32 disparity.
Evidence: `src/models/stereonet/stereonet.py:98-114` (forward returns
disparity); `src/models/stereonet/regression.py:33-47` (readout emits
candidate units 0..11; pixel scale is carried by the refinement, per
`src/models/stereonet/stereonet.py:18-27`); `scripts/exp_reproduce_hailo.py:151`
(`out[0,0]` used directly as disparity).

## OUTPUT POSTPROCESSING

ReLU clamp inside the graph; nothing else. No depth conversion, no 8-bit
cast, no resizing in the evaluation path.
Evidence: `src/models/stereonet/stereonet.py:110-111` (`torch.relu`);
`scripts/exp_reproduce_hailo.py:149-151` (raw output scored directly).

## CHECKPOINT

The contract pins the protocol, not the weights. Under this contract two
checkpoints are locked in Phase 0: the reference
`reference/onnx/stereonet.onnx` (accuracy reference) and
`results/training/convergence_run.pth` (from-scratch PyTorch baseline).
Both were evaluated live under exactly this contract in Phase 0; numbers are
in `REFERENCE_BASELINE.md` and the final report, never mixed with any other
protocol.

## NO-METRIC-MIXING RULE (binding)

Never mixed into one baseline number: Hailo-exact D1 (1/255, per-image),
KITTI official D1 (1/256, pooled), the EXP-007 subset (1,837,304 px,
EPE 1.4422 / D1 9.3315% — a different pixel set), 10-scene validation
(EXP-016 train-time val), 40-scene evaluation, late-window metrics,
different GT scaling, different valid-pixel sets. Every reported number must
carry dataset, scene set, valid pixels, GT scale, metric, protocol,
checkpoint. A number that cannot be reconstructed from its record is
NOT VERIFIED.

---

This is the single primary evaluation contract for Phase 0 and Phase 1 model
comparisons.
