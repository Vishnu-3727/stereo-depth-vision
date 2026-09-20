# PHASE_2_BASELINE_HAILO_STEREONET

Reconstructed reference condition Phase 2 compares against. All numbers below
are **MEASURED** in Phase 1 and re-cited here verbatim (source EXP id given
per row) — none re-measured, none altered. Phase 1 is frozen at tag
`phase-1-frozen` (commit `b4207e5`); nothing in this document changes it.

## Architecture

| Property | Value | Source |
|---|---|---|
| Input resolution | 368 x 1232, RGB, rectified | SR-001 |
| Feature resolution | 23 x 77 (1/16), 32 channels, per branch | SR-001 |
| Disparity candidates | 12 (`num_disparities`) | SR-001 |
| Cost volume | subtraction, `shift="none"` (degenerate — no-op shift) | EXP-010 |
| Aggregation | 5 Conv3d layers | SR-001 |
| Disparity estimation | soft-argmin, full-resolution (cost upsampled first) | SR-001 |
| Refinement | 6 dilated residual blocks (1/2/4/8/1/1), 32 ch, guided by left RGB | SR-001 |
| Output | disparity, ReLU-clamped, full resolution | SR-001 |
| Downsampling activations | none (linear operator) | SR-001 |

## Complexity (MEASURED, EXP-001 / EXP-013 / EXP-017-18)

| Quantity | Value |
|---|---|
| Unique parameters | 423,586 |
| Parameters (Hailo per-occurrence convention) | 623,138 |
| Total MACs/image | 56.04 GMAC (Hailo compiler: 56,258,882,372 within 0.4%) |
| Total ops/image (2xMACs) | 112.08 G |
| Model size (learned weights) | 1.62 MiB |
| Cost volume tensor | 2.59 MiB |
| Peak single activation | 55.34 MiB (inside refinement) |
| Total activation traffic | 1,899 MiB (90.8% in refinement) |
| Measured peak GPU allocation | 302.1 MiB (RTX 4060) |

### Stage split (MEASURED, two independent routes agreeing to 0.02pp — EXP-001/EXP-013/EXP-018)

| Stage | MACs % | GPU time % | CPU time % |
|---|---:|---:|---:|
| Feature extraction (x2) | 5.1 | 18.4 | UNKNOWN (see EXP-014 for CPU split) |
| Cost volume construction | 0.0 | 3.6 | — |
| 3D aggregation | 4.2 | 2.8 | 17.2 |
| Upsample + soft-argmin | 0.01 | 1.4 | — |
| Refinement | 90.6 | 73.3 | 75.8 |

## Accuracy (MEASURED, EXP-005, pretrained Hailo-exported weights)

| Metric | Value | Protocol |
|---|---:|---|
| D1 (Hailo's published-figure protocol) | 8.2237 % | reproduces published 8.223 |
| EPE (official KITTI protocol) | 1.3134 px | disp_occ_0, scenes 160-199 |
| D1 (official protocol) | 8.154 % | same |

## Depth (MEASURED, EXP-012, KITTI calibration f=721.5px, B=0.533m)

| Range band | Disparity EPE | Depth MAE | delta1 |
|---|---:|---:|---:|
| 0-10 m | 1.467 px | 0.213 m | 99.1% |
| 20-30 m | 1.290 px | 2.082 m | 93.2% |
| 50-80 m | 1.311 px | 9.158 m | 75.5% |

## Latency (MEASURED, EXP-013/EXP-014, this project's own stack — NOT Hailo silicon)

| Device | End-to-end latency | FPS |
|---|---:|---:|
| RTX 4060, fp32, batch 1 | 44.4 ms | 22.5 |
| RTX 4060, fp16, batch 1 | 64.8 ms (fp16 slower on this measurement — see EXP-015 note: latency figure is separate from the fp16 accuracy-neutral finding) | — |
| CPU, fp32, batch 1 | 642.3 ms | 1.6 |

Hailo published 10.7 FPS on physical Hailo-8 silicon. **Never compared
directly** — different hardware, compiler, and precision (EXP-013's
methodology note, preserved here).

## Quantization (MEASURED, EXP-015, this project's stack)

| Precision | EPE px | D1 % | Depth Abs Rel | Latency |
|---|---:|---:|---:|---:|
| fp32 CUDA | 1.314 | 8.155 | 0.0489 | 90.1 ms |
| fp16 CUDA | 1.310 | 8.141 | 0.0486 | 64.8 ms |
| int8 ORT CPU | 1.655 | 10.945 | 0.0627 | 677.4 ms |

## Robustness / failure modes

Middlebury scene-class analysis (textureless, repetitive, reflective, thin
structure, boundary) is **UNKNOWN / DEFERRED** — Phase 1 did not download the
dataset (research_questions.md O4). Not fabricated here.

## Training configuration (MEASURED, EXP-016 — convergence proof only, not a
competitive accuracy baseline)

| Field | Value |
|---|---|
| Dataset / split | KITTI 2015, hailo_calib (scenes 0-159) train / hailo_val (160-199) val |
| Resolution / crop | random 256x512 |
| Disparity range | 176 px |
| Augmentation | random crop, per-image gain jitter sigma=0.1, no h-flip |
| Optimizer | Adam betas=(0.9,0.999), lr 1e-3, cosine anneal to 0 |
| Batch size | 2 |
| Epochs | 20 |
| Loss | masked smooth L1, beta=1.0 |
| Initialization | PyTorch defaults, random |
| Seed | 0 |
| Precision | fp32 |
| Known deviation | no batch norm (BN folded into the exported artifact) |
| Result | loss 11.49->7.61; **validation did not improve** (18.497->19.216 px EPE) — 160 scenes from scratch is too little data; see `experiments/EXP-016/CORRECTION.md` |

This training recipe is what Phase 2's H1 experiment (`EXP-H1-BASE` /
`EXP-H1-WORKING`) reuses byte-for-byte, with only `cost_volume_shift` changed.

## Hardware for this document

All figures above except where noted "MEASURED, this project's own stack" are
Phase 1 GPU/CPU numbers on an RTX 4060 + host CPU — see
`experiments/_env_baseline_primary.json`. Phase 2's own H1 training run (see
`phase2/docs/PHASE_2_REGISTRY.md`) executed on **CPU only** —
`nvidia-smi` returned a permissions error in this session's environment
(`NVIDIA-SMI has failed because you do not have sufficient permissions`),
so `torch.cuda.is_available()` is `False` here. This is recorded as **UNKNOWN
why** (driver/service issue, not investigated further) rather than assumed
absent hardware. Phase 2 latency/memory measurements for the H1 comparison
should not be treated as GPU figures.
