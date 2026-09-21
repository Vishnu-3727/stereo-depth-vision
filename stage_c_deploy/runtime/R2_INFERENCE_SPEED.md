# R2 — inference-stage speed, with a pre-registered equivalence gate

## 1 Scope and status

**VERDICT: no configuration accepted. The recommended runtime path is unchanged.**

This experiment attacks the inference stage of the host runtime path only. It is host
runtime work, added after the Stage C validation closed. It changes no frozen artefact, no
checkpoint, no evaluation contract and no accuracy number, and it does not bear on Hailo
deployment, which remains blocked.

The two candidate interventions were already present as unmeasured command-line flags in
`runtime_path.py`: `--channels-last` and `--compile`. This experiment measured them.

## 2 Host and protocol

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| torch | 2.7.0+cu128 |
| Python | 3.12.9 |
| dtype / resolution | fp32, 368x1232 |
| Scenes | first 10 of `hailo_val` |
| Warmup | **5** in every configuration, control included |
| Statistic | median over the 10 scenes |
| Checkpoint | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` (asserted in every run) |

Every configuration, control included, holds `--decode par --norm gpu --gpu-geometry` and
cudnn.benchmark OFF. Warmup 5 was chosen so that the control is calibrated against the
treatments at the same budget; `torch.compile` needs warmup, and a control measured at a
smaller warmup would not be a fair comparison. This means the control here is *not*
numerically comparable to the warmup-2 figure of 105.7 ms in `README.md` section 3.

## 3 Hypothesis

H1: `--channels-last`, `--compile`, or both together reduce the median inference-stage time
below the control, while leaving the computed disparity bit-identical.

## 4 Pre-registered gate (decided before any measurement)

ACCEPT a configuration only if BOTH hold:

1. **Equivalence** — max absolute disparity difference against the R2-CTL control is exactly
   `0.0` on 10/10 scenes.
2. **Determinism** — the same configuration, run in two separate processes, gives max absolute
   disparity difference exactly `0.0` on 10/10 scenes.

REJECT otherwise, whatever the speed. This is the standard that already rejected
cudnn.benchmark (`README.md` section 8). Speed is reported for rejected configurations as
context only, labelled REJECTED.

## 5 Configurations

| id | flags on top of `--gpu-geometry` | outcome |
|---|---|---|
| R2-CTL | (none) | control |
| R2-CL | `--channels-last` | **REJECTED** — fails equivalence |
| R2-CO | `--compile` | **NOT RUNNABLE ON THIS HOST** |
| R2-CLCO | `--channels-last --compile` | **NOT RUNNABLE ON THIS HOST** (compile part) |

## 6 `torch.compile` is not runnable on this host

Measured, both times `--compile` was requested:

```
torch.compile first run failed (TritonMissing: Cannot find a working triton installation.
Either the package is not installed or it is too old.)
```

`runtime_path.py` caught this and fell back to a fresh uncompiled net, as it is written to do
(`compile_status: "fallback_run_TritonMissing"` in both JSONs). Also emitted, before the
failure: `Not enough SMs to use max_autotune_gemm mode`.

**Consequence for the numbers:** the R2-CO and R2-CLCO runs did not time a compiled model.
R2-CO timed the control path and R2-CLCO timed the channels_last path. They are therefore
reported below as extra repeats of those two paths, not as separate configurations. No
attempt was made to install Triton; installing packages was out of scope for this experiment.

H1 for `torch.compile` is **NOT TESTED** on this host, not refuted.

## 7 Gate measurements

Source: `out/r2_equivalence_determinism.json`, produced by `r2_equiv.py`, which dumps the
disparity of the 10 scenes from one process per dump and compares dumps element-wise.

| Check | Bit-identical scenes | Max abs disparity diff | Verdict |
|---|---|---|---|
| R2-CTL, process A vs process B | 10/10 | **0.0 px** | deterministic |
| R2-CL, process A vs process B | 10/10 | **0.0 px** | deterministic |
| R2-CL vs R2-CTL | **0/10** | **0.5596160888671875 px** (scene `000161_10`) | **equivalence FAIL** |

channels_last is internally deterministic — it is not a cudnn.benchmark-style instability —
but it selects different kernels and therefore computes a different disparity. 0.56 px is
about half the model's own EPE of 1.19 px. Under the pre-registered gate this is a rejection,
and it would be a rejection even if channels_last were much faster than it is.

The scene-0 diagnostic in the benchmark JSONs shows the same thing from the other side:
0.227900 px against the fp32 CPU reference on the control path, 0.461359 px on the
channels_last path.

## 8 Timing (context only; nothing here passed the gate)

Median ms per frame. Three repeats per code path, warmup 5 each.

| code path | run | preproc | infer | depth | cloud | disc | total | FPS | source JSON |
|---|---|---|---|---|---|---|---|---|---|
| control | 1 | 33.08 | 53.13 | 1.31 | 12.84 | 1.74 | 101.56 | 9.85 | `out/r2_ctl_s10_w5.json` |
| control | 2 | 28.07 | 53.42 | 1.26 | 12.24 | 1.75 | 97.27 | 10.28 | `out/r2_co_s10_w5.json` (compile fell back) |
| control | 3 | 28.55 | 53.16 | 1.41 | 13.77 | 2.11 | 100.22 | 9.98 | `out/r2_ctl_rep3_s10_w5.json` |
| channels_last [REJECTED] | 1 | 31.37 | 53.19 | 1.85 | 17.47 | 2.57 | 109.83 | 9.11 | `out/r2_cl_s10_w5.json` |
| channels_last [REJECTED] | 2 | 27.67 | 49.95 | 1.27 | 12.04 | 1.69 | 93.18 | 10.73 | `out/r2_clco_s10_w5.json` (compile fell back) |
| channels_last [REJECTED] | 3 | 30.45 | 49.80 | 1.55 | 13.91 | 1.80 | 97.75 | 10.23 | `out/r2_cl_rep3_s10_w5.json` |

Run-to-run noise, measured here:

- Control inference median across 3 runs: 53.13 / 53.42 / 53.16 ms — spread **0.29 ms**.
- channels_last inference median across 3 runs: 53.19 / 49.95 / 49.80 ms — spread **3.39 ms**.
- Control total across 3 runs: 97.27 – 101.56 ms — spread **4.29 ms**.
- channels_last total across 3 runs: 93.18 – 109.83 ms — spread **16.65 ms**.

Read honestly: two of the three channels_last runs are about 3.3 ms faster at the inference
stage than every control run, but the third matches the control, and the total-time spread of
either path is larger than the difference between the paths. At n=3 this experiment **cannot
separate** a channels_last inference speed-up from run ordering and thermal state. That
question is moot for the decision — channels_last is rejected on equivalence regardless of
its speed — so no further repeats were run to settle it.

## 9 Result

- H1 **not supported** for channels_last: it is rejected by the equivalence gate at
  0.5596160888671875 px, and its speed advantage is not resolvable above the measured
  run-to-run noise at n=3.
- H1 **not tested** for torch.compile: not runnable on this host (TritonMissing).
- The recommended host runtime path is **unchanged**: cuda, fp32, `--decode par --norm gpu
  --gpu-geometry`, cudnn.benchmark OFF, channels_last OFF, compile OFF.
- Newly established by this experiment: the recommended path is **bit-identical across
  separate processes on 10/10 scenes** (0.0 px). The previous determinism evidence in
  `README.md` section 8 was a two-process check reported alongside the cudnn.benchmark
  rejection; this is a second, independent confirmation at the same standard.

## 10 What was NOT measured

- Any compiled-model timing on this host — NOT MEASURED (Triton absent).
- channels_last accuracy on the 40-scene contract — NOT MEASURED, and deliberately so: a
  configuration that fails the bit-identity gate is rejected without spending a contract run
  on it.
- Whether channels_last is genuinely faster at inference — NOT SEPARABLE at n=3 (section 8).
- Anything about Hailo hardware — out of scope, unchanged, still blocked.

## 11 Files

| File | Role |
|---|---|
| `r2_equiv.py` | Disparity dump + element-wise comparison (the gate instrument) |
| `out/r2_ctl_s10_w5.json`, `out/r2_ctl_rep3_s10_w5.json` | Control, runs 1 and 3 |
| `out/r2_cl_s10_w5.json`, `out/r2_cl_rep3_s10_w5.json` | channels_last, runs 1 and 3 |
| `out/r2_co_s10_w5.json`, `out/r2_clco_s10_w5.json` | Compile requested; fell back to uncompiled (runs 2) |
| `out/r2_equivalence_determinism.json` | The gate measurements of section 7 |

## 12 How to reproduce

```bash
python stage_c_deploy/runtime/runtime_path.py --device cuda --gpu-geometry --scenes 10 --warmup 5 --out stage_c_deploy/runtime/out/r2_ctl_s10_w5.json
python stage_c_deploy/runtime/runtime_path.py --device cuda --gpu-geometry --channels-last --scenes 10 --warmup 5 --out stage_c_deploy/runtime/out/r2_cl_s10_w5.json
python stage_c_deploy/runtime/r2_equiv.py dump <tmp>/ctl_a
python stage_c_deploy/runtime/r2_equiv.py dump <tmp>/cl_a --channels-last
python stage_c_deploy/runtime/r2_equiv.py cmp <tmp>/ctl_a <tmp>/cl_a
```

---

Host runtime work only. No frozen artefact, checkpoint, contract or accuracy figure was
created or changed to produce this report. Hailo deployment remains blocked.
