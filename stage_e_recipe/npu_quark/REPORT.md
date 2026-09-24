# REPORT: Quark INT8 on the Ryzen AI NPU (characterization, low-memory calibration)

Date: 2026-09-24. Status: PARTIAL (Q1/Q3 full rows; Q2 NPU failed — see exact
errors). This is CHARACTERIZATION, not an experiment: no thresholds, no
verdicts, no deployment claim. No training. No Kaggle.

Spec: `docs/superpowers/specs/2026-09-24-npu-quark-int8-design.md`,
protocol commit `6397508` (2026-09-24 12:20:23 +0530), Amendment A1 commit
`497b55c` (2026-09-24 13:04:01 +0530 = 07:34:01Z). Attempt-3 first
quantization `2026-09-24T07:39:03.692914Z`, end `2026-09-24T07:46:36.811519Z`,
total 453.1 s — inside the 2 h box, and after the amendment commit
(pre-registered). Full record: `stage_e_recipe/npu_quark/npu_quark.json`.
Script: `stage_e_recipe/npu_quark.py` (driver + `--arm-full` child per arm +
`--worker` NPU grandchild).

## Result table (measured)

| arm | quantized size | CPU EPE (40-scene, own arm) | NPU compile + s + kernel evidence | NPU median ms | NPU EPE | NPU−CPU |
|---|---|---|---|---|---|---|
| Q1 (XINT8, MinMax, full graph) | 305380 B + 395744 B sidecar = 701124 B, sha `d2772ef5a89b…` | 7.0967987 (d1 80.35, rmse 12.45, nonfinite 0, contract True) | success, 57.0 s; INT8-family 0 / Bf16-family 0 / other 0 (log 278 B, 3 lines, 0 partition hits, 0 EP files) | 359.18 (p90 375.42, n=50) | 35.8634376 (d1 94.99, rmse 43.20, nonfinite 0, contract True) | +28.7666389 |
| Q2 (A16W8, MinMax, full graph) | 272472 B + 395744 B sidecar = 668216 B, sha `77ba7197c848…` | 1.3683450 (d1 7.60, rmse 3.38, nonfinite 0, contract True) | FAILED (see errors) | failed | failed | failed |
| Q3 (XINT8, MinMax, 73 head+refinement nodes fp32) | 260722 B + 732128 B sidecar = 992850 B, sha `d6f1d32e86c3…` | 1.3319438 (d1 7.03, rmse 3.44, nonfinite 0, contract True) | success, 31.9 s; INT8-family 0 / Bf16-family 0 / other 0 (log 278 B, 3 lines, 0 partition hits, 0 EP files) | 642.44 (p90 657.90, n=50) | 27.3178597 (d1 69.19, rmse 38.71, nonfinite 0, contract True) | +25.9859158 |

Quantize wall: Q1 19.1 s / Q2 18.9 s / Q3 18.0 s. CPU score wall:
41.6 / 43.2 / 41.5 s. NPU score wall: Q1 18.2 s, Q3 29.5 s.
Per-arm child wall (quantize+CPU+NPU): 169.1 / 114.7 / 169.1 s.
40-scene contract: `hailo_val`, 3802797 valid pixels every scored row.
Recorded references (not re-claimed): fp32 1.18598534271453 / R0
5.248941413313285 / R2 1.2908231335718434 (commit `65ac304`).

## Measured

- Subject assert: sha256 `7453b2be…32b9645`, 1701550 bytes — match.
- Node sets: 712 nodes; 40 `/regression/*`, 33 `/refinement/*`, 73 combined;
  required `/regression/Mul{,_1,_2}` + `/refinement/{input_conv,output_conv}/Conv`
  present; runtime prefix enumeration matched.
- Environment: clone python 3.12.11, `amd-quark==0.11.2`,
  `onnxruntime==1.27.0`, providers `[VitisAIExecutionProvider,
  DmlExecutionProvider, CPUExecutionProvider]`, cpu_count 24,
  `RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\1.8.0`. No conda clone
  and no pip install were performed in attempt 3 per the brief (the
  `ryzen-ai-1.8.0-quark` env was reused as-is); `aianalyzer` probe: no module
  named aianalyzer (see JSON).
- Per-arm quantization provenance (exact): `include_cle=False`,
  `calibrate_method=CalibrationMethod.MinMax` (explicit override; XINT8
  default was `PowerOfTwoMethod.MinMSE`), `use_external_data_format=True`,
  first 8 `hailo_calib` scenes batch-1, QDQ, Q1 XINT8 QUInt8/QInt8 symmetric
  (power-of-two scale type kept) / Q2 A16W8 QInt16/QInt8 / Q3 XINT8 with 73
  `nodes_to_exclude`. No per-batch/streaming calibration flag exists in
  installed Quark 0.11.2 (all `QuantizationConfig` fields inspected) —
  `use_external_data_format` is the only memory-relevant option.
- Memory guard: parent polled free RAM every 2 s, kill threshold 2.0 GB free.
  No arm tripped it. Peak child-tree RSS: Q1 2.42 / Q2 2.87 / Q3 1.92 GB.
  Min free observed: 3.43 / 3.65 / 5.56 GB. Free RAM never reached 0
  (attempt-2 failure mode eliminated).
- NPU sessions per spec §3 verbatim: providers
  `["VitisAIExecutionProvider", "CPUExecutionProvider"]`, provider_options
  `[{"cacheDir": "stage_e_recipe/npu_quark/cache_<arm>", "cacheKey":
  "npu-quark-<arm>"}, {}]`, `log_severity_level=1`, `ORT_ENABLE_ALL`,
  CWD = output dir. Session provider lists confirmed
  `[VitisAIExecutionProvider, CPUExecutionProvider]` on Q1/Q3.
- NPU latency per cost-spec §4 (shared `inputs_npy`, 10 scenes × 5 = 50
  `sess.run` samples): Q1 median 359.18 ms (FPS 2.78), Q3 median 642.44 ms
  (FPS 1.56). Raw + per-scene samples in JSON.
- Prior attempts: attempt 1 — all arms `ValueError: axes don't match array`
  in 18.5 s (default CLE pre-pass, Conv3d 5-D weights). Attempt 2 —
  `include_cle=False`, calibration collect-data 208.52 s (32 scenes), then
  ~10.5 GB committed / free RAM 0, killed by manager on user instruction, no
  model produced. Both recorded in Amendment A1 + JSON `prior_attempts`.

## Inferred

- Rows describe Quark-quantized graphs and VitisAI EP behaviour on THIS
  machine/installation only; CPU fallback of unsupported ops is recorded, not
  worked around.
- Q1/Q3 compile logs match the cost-spec R0/R2 precedent shape (3 lines, no
  partition lines, no EP-emitted files) while the sessions still list the
  VitisAI provider first — compile "success" means session creation returned,
  not that any operator ran on the NPU fabric.

## Unknown

- Per-op NPU/CPU assignment beyond the compile log and EP-emitted files (not
  exposed via the ORT API): UNKNOWN on all arms.
- Whether Q2's aiecompiler shared-buffer error caused the subsequent native
  crash (temporal order only): UNKNOWN (hypothesis, not evidence).
- Why Q1 XINT8 full-graph CPU EPE (7.10) exceeds Q2 (1.37) and Q3 (1.33) CPU
  EPEs on the same 40 scenes: UNKNOWN (numbers only, no causal claim).

## Claim status

- MEASURED: every number in the table; guard readings; kernel counts (zero);
  exact error texts below; environment versions; amendment/prior-attempt
  record.
- OPEN: per-op placement; Q2 crash causality; Q1-vs-Q2/Q3 CPU gap cause.
- HYPOTHESIS: Q2's native crash followed its failed AIE compile (order
  observed, mechanism unverified). Nothing else is claimed.

## Exact errors

- Q2 NPU: grandchild `--worker Q2` exited `3221225477` (0xC0000005) with
  empty captured tail (native crash, nothing flushed). Its compile log
  (`Q2_npu_compile.log`, 485 B) retains:
  `ERROR: [aiecompiler 77-5379] Access pattern for shared buffer port
  model.layers[11].single_layer.ofm_half0[0].out[0] is not specified or
  empty.` / `Compilation Failed` / `(WARNING:0, CRITICAL-WARNING:0, ERROR:1)`.
- No other arm errored. Q2/Q3 ran only because Q1 finished within the guard
  (Amendment A1 condition satisfied).

## Repro

Clone python: `ryzen-ai-1.8.0-quark` (`C:\Users\vishn\anaconda3\envs\ryzen-ai-1.8.0-quark\python.exe`),
`RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\1.8.0`, CWD = repo root:
`python stage_e_recipe/npu_quark.py` (drives Q1→Q2→Q3 guarded children; ~8 min
observed, 2 h box). Deviations from the frozen spec are exactly Amendment A1
(§ above + spec file); no other deviation.
