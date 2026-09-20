# Stage-B pre-flight audit for ARM P (SceneFlow pretraining) — AUDIT ONLY

Date (UTC): 20260918T053135Z. Repo: C:\Users\vishn\stereo_depth_vision.

**NO training was run. NO pretraining, NO fine-tuning, NO checkpoint was created.
No GPU training run was started. No existing file was modified.**

## Verdicts (compressed)

1. **Upstream-checkpoint path: STOPPED. `PROVENANCE NOT ESTABLISHED`.**
   No checkpoint file exists anywhere in
   `reference/upstream/extracted/StereoNet-master/` (verified by full recursive
   file listing: 22 files, zero `.tar`/`.pth`/checkpoint artifacts). The only
   checkpoint named anywhere,
   `checkpoints/checkpoint_sceneflow.tar`, is an undistributed path defaulted
   by `finetune-kitti15/finetune-kitti15.py:32` (and written by
   `pretrain-sceneflow/sceneflow-pretrain.py:155-160`, never shipped). The
   architecture that would have produced it also differs from P2A everywhere
   load-bearing (see `architecture_compatibility.json`). There is nothing to
   load, and nothing loadable.
2. **Viable path: pretrain P2A's OWN architecture (ARM P).** A checkpoint
   produced by training this repo's P2A architecture satisfies strict loading
   by construction — but that MUST still be verified at the time (see
   `checkpoint_loading_contract.json`), never assumed.
3. **Training-data gate: FT3D on C: is INCOMPLETE and ASYMMETRIC.**
   Usable stereo triplets (left png + right png + left pfm) on C::
   **14,460** (A: 6,990; B: 0; C: 7,470). Disparity/TRAIN has A and C only —
   **B is entirely missing** (7,460 left frames orphaned); A lacks 47
   sequences (470 frames). `monkaa/` on C: is EMPTY. Full-archives live only on
   `D:\sceneflow_archives` (archive-only is NOT a training path; nothing was
   copied from D:).
4. **Disparity distribution (400 random usable TRAIN left-PFM, seed 0,
   `src/datasets/pfm.py`, 207,360,000 finite px): fraction >= 184 px =
   0.00446 — SMALL.** The >=184 mask removes ~0.45% of FT3D pixels. No loud
   warning: most of FlyingThings3D IS representable by P2A's 24 candidates.
5. **Three-way comparison: YES — FT3D supplies the missing supervision.**
   Per-bin FT3D_frac / KITTI_eval_frac at [64,80)/[80,96)/[96,112)/[112,128)
   = **2.02 / 7.66 / 5.78 / 4.70** (vs raw KITTI training fracs: 2.03 / 7.67 /
   16.65 / 45.83). Estimated staged FT3D pixels in [64,128): ~1.23 BILLION
   vs 559,057 raw KITTI training pixels (~2,200x). After >=184 masking the
   premise still holds — the masked-away mass is negligible.

## ARM D position (prior art, unchanged)

ARM D was Driving-only (4,400 pairs) and was REFUTED at 10.5748918 px
(`phase1/runs/arm_d/arm_d_record.json`: best 10-scene val EPE 13.2249 at
epoch 155; frozen-contract figure 10.5748918 per audit record). Recorded here
as prior-art risk only. Its verdict is NOT restated and NOT reinterpreted.

## P2A frozen reference (context, re-verified live where noted)

397,954 params (verified live by construction), downsample_levels=3
(stride 8), num_disparities=24, 8 px spacing, cost_volume_shift=right (REAL),
regression_normalize=True, 6 refinement blocks, max representable 184 px.

## Contents

- `provenance_audit.json` — ten provenance answers with file:line evidence.
- `architecture_compatibility.json` — upstream vs P2A table; P2A must NOT be
  modified to accept the upstream checkpoint; no such change is proposed.
- `checkpoint_loading_contract.json` — SPEC the future ARM-P checkpoint must
  satisfy (not a result; nothing was loaded).
- `training_distribution_audit.json` — staging integrity, FT3D histogram,
  three-way comparison, crop/resize analysis, not-measurable list.
- `raw/` — machine-generated probe outputs backing the JSONs.
- `scripts/` — read-only measurement scripts used (no training code).

## Method / prohibitions honored

- Read-only probes only (`pathlib` counts, `read_pfm` histogram sampling,
  model construction for param count). No optimizer, no backward, no save.
- No file outside this directory was created or modified.
- No invented numbers: anything undeterminable without a new training path is
  marked `NOT MEASURABLE WITH CURRENT ARTIFACTS`.
- Raises armed: zero unreadable PFMs, zero NaN/inf maps, 14,460 usable pairs
  (no zero-pair trigger).
