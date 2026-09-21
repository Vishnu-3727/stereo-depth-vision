# Host Runtime Benchmarking and Fast Runtime Path (`stage_c_deploy/runtime/`)

## 1 Scope and status

This directory holds host runtime benchmarking and a faster runtime path for
running the pipeline on the development machine.

This is NEW work, added after the Stage C validation closed. It is not part of
the frozen Stage C record, it changes no frozen artefact, and it does not bear
on Hailo deployment, which remains blocked.

Measurement host for all figures in this report: NVIDIA GeForce RTX 4060 Laptop
GPU, torch 2.7.0+cu128, fp32, 368x1232, 10 scenes from the head of hailo_val,
2 warmup iterations, medians.

## 2 Files

| File | Role |
|---|---|
| `bench_host.py` | Baseline harness |
| `runtime_path.py` | Fast path, including `fast_run_scene` |
| `fp16_probe.py` | Overflow probe |
| `preprocess_breakdown.py` | Preprocess sub-timing |
| `out/*.json` | Results |

Result-to-file mapping:

| JSON file | Result it holds |
|---|---|
| `out/bench_cpu_fp32_s10_w2.json` | CPU baseline |
| `out/bench_cuda_fp32_s10_w2.json` | CUDA baseline |
| `out/bench_cuda_fp16_s10_w2.json` | fp16, INVALID output (see §6) |
| `out/bench_cuda_bf16_s10_w2.json` | bf16, rejected (see §7) |
| `out/runtime_cuda_fp32_s10_w2.json` | Preprocess-optimised |
| `out/runtime_cuda_fp32_opt_s10_w2.json` | +gpu-geometry +cudnn.benchmark, REJECTED config (see §8) |
| `out/runtime_cuda_fp32_det_s10_w2.json` | +gpu-geometry, deterministic, the recommended path |
| `out/fast_path_40scene_cuda.json` | 40-scene contract score of the fast path on cuda (see §9) |

## 3 Measured results

Median ms per frame, and FPS from the total median. Same host and protocol as
§1 (10 scenes from the head of hailo_val, 2 warmup iterations).

| config | preproc | infer | depth | cloud | disc | total | FPS |
|---|---|---|---|---|---|---|---|
| cpu fp32 baseline | 87.2 | 763.3 | 13.1 | 19.3 | 28.7 | 912.5 | 1.10 |
| cuda fp32 baseline | 88.2 | 64.9 | 16.4 | 18.3 | 32.1 | 221.7 | 4.51 |
| cuda fp32, preprocess optimised | 30.5 | 55.3 | 13.5 | 18.4 | 24.3 | 140.8 | 7.10 |
| cuda fp32, + gpu geometry (deterministic) | 36.9 | 50.3 | 1.43 | 14.95 | 1.94 | 105.7 | 9.46 |
| cuda fp32, + cudnn.benchmark [REJECTED] | 28.2 | 45.8 | 1.25 | 12.21 | 1.66 | 89.7 | 11.14 |

## 4 Preprocess optimisation (free)

The preprocess optimisation reduces the CUDA baseline preprocess stage from
88.2 ms to 30.5 ms. It is free in the sense that it changes no computed value:

- The ground-truth disparity read (13.1 ms) is dropped because live operation
  has no ground truth.
- The two PNGs are decoded concurrently with `cv2.imread`, verified
  bit-identical to the original loader on 10/10 scenes.
- Normalisation is moved to the GPU in float32, giving input max-abs-diff 0.0
  and disparity max-abs-diff 0.0 px on 10/10 scenes.

## 5 GPU geometry ports (opt-in)

The GPU geometry ports are opt-in via `--gpu-geometry`; the frozen numpy
implementations stay the default.

Equivalence worst case over 10 scenes: depth 0.0 m exact, valid-mask mismatch
0, discontinuity label mismatch 0 with magnitude difference <= 1.42e-14,
spatial 0.0, cloud point-count difference 0.

The occupancy port was measured at 15.01 ms against 6.38 ms for numpy, so
occupancy stays on numpy.

## 6 fp16 is unusable

fp16 is unusable. Activations grow 2.6 -> 23.7 -> 1351 -> 43264 through the
downsample stack and reach the fp16 ceiling of 65504 at `residual.0.conv2`,
the first non-finite stage; the final disparity is 100 percent NaN.

The apparently fast geometry stages in the fp16 run (3.4 and 4.3 ms) were an
artefact of empty masks, not a speed-up. This is consistent with the large
fp32 intermediates recorded in DYNAMIC_RANGE_AUDIT.md.

## 7 bf16 runs but is rejected

bf16 runs finite (191.7 ms, 5.22 FPS) but moves the scene-0 disparity by
2.317255 px against the fp32 CPU reference. Rejected; fp32 remains the runtime
dtype.

## 8 cudnn.benchmark is rejected

cudnn.benchmark is rejected despite being 16 ms faster: it changed disparity
by up to 2.39 px and was nondeterministic across processes, while the plain
path is bit-identical run to run (two separate processes, max abs disparity
difference 0.0). 2.39 px is about twice the model's own EPE, so the speed is
not worth a nondeterministic output.

## 9 Accuracy of the recommended path

Accuracy of the recommended path (the deterministic +gpu-geometry config),
measured on the full frozen contract on cuda fp32
(`out/fast_path_40scene_cuda.json`):

- EPE 1.191216765057325 px
- D1 6.211033615520366
- RMSE 3.1369637733547697
- valid pixels 3,802,797
- contract_match true
- checkpoint SHA-256 b2f6f5d5...eb7454

This equals the frozen CPU figure of 1.1912168 px. This is a NEW measurement
of the same checkpoint on a different device and does NOT replace, edit or
supersede the frozen CPU record.

## 10 Demo integration

`pipeline_demo.py` gained one opt-in flag, `--fast`, which routes through
`runtime_path.fast_run_scene`; the default path is unchanged.

Verified against `run_scene` on scenes 0, 5 and 31: identical keys; disparity,
depth, point cloud, cells, occupancy, stats, EPE and D1 all differ by exactly
0.0; valid-mask, physical-mask and discontinuity-label mismatches all 0; the
only difference is the discontinuity magnitude at <= 7.1e-15, which is
floating-point association order.

A single-shot demo run is dominated by interpreter and matplotlib start-up
(about 11.5 s either way); the per-frame gain shows in a sequence loop, not in
one invocation.

## 11 How to run

```bash
python stage_c_deploy/runtime/bench_host.py --device cuda --scenes 10
python stage_c_deploy/runtime/runtime_path.py --gpu-geometry --scenes 10
python stage_c_deploy/demo/pipeline_demo.py --scene 31 --fast
```

## 12 R2: inference-stage attacks, both rejected or not runnable

Full report: `R2_INFERENCE_SPEED.md`.

The inference stage is the largest single stage of the recommended path, so the two
unmeasured inference flags already in `runtime_path.py` were measured against a
pre-registered gate: accept only if the disparity is bit-identical to the control on 10/10
scenes AND bit-identical across two separate processes. Same standard as section 8.

| config | verdict | basis |
|---|---|---|
| `--channels-last` | **REJECTED** | deterministic (0.0 px across processes) but not equivalent: **0.5596160888671875 px** vs the control, 0/10 scenes bit-identical |
| `--compile` | **NOT RUNNABLE ON THIS HOST** | `TritonMissing` on first run; `runtime_path.py` fell back to the uncompiled net, so those runs timed the control path, not a compiled one |

No configuration was accepted. **The recommended path in section 3 is unchanged**, and no
row was added to its table: the R2 runs used warmup 5, while section 3 is warmup 2, so the
two are not numerically comparable.

Also established by R2: the recommended path is bit-identical across two separate processes
on 10/10 scenes (0.0 px) — a second confirmation of the determinism claimed in section 8.
Measurements: `out/r2_equivalence_determinism.json`, `out/r2_*.json`.

## 13 R3: decode prefetch — accepted, 10.3 to 13.0 FPS on a sequence

Full report: `R3_SEQUENCE_THROUGHPUT.md`.

Sections 3 to 12 all measure one frame at a time, which serialises the CPU-bound PNG decode
against the GPU-bound inference. Over a sequence they need not be serialised: one background
thread decoding frame N+1 while the GPU runs frame N overlaps the two.

The per-frame work is unchanged — this is a schedule, not a computation — so the disparity is
bit-identical, and it passes the same gate that rejected cudnn.benchmark and channels_last:

| check | result |
|---|---|
| prefetch vs serial | **0.0 px**, 10/10 scenes bit-identical |
| prefetch across two processes | **0.0 px**, 10/10 scenes |
| the R3 loop vs the R2 control dump | **0.0 px**, 10/10 (the loop reproduces the timed path) |
| serial, 4 independent processes | 94.82 – 97.15 ms/frame (10.29 – 10.55 FPS) |
| prefetch, 4 independent processes | 75.26 – 77.45 ms/frame (12.91 – 13.29 FPS) |
| gain | **17.96 – 21.89 ms/frame**, against a cross-process median spread of 2.33 ms |

**ACCEPTED**, but not wired in: prefetch only pays off over a sequence, and the demo is
single-shot, so `runtime_path.py`, `fast_run_scene` and `pipeline_demo.py` are unchanged.
This is a throughput result, not a latency result — the first frame still costs full serial
time — and the decode measured is a KITTI PNG read, not a live camera.

Measurements: `out/r3_gate.json`, `out/r3_stream_*.json`.
