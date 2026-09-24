# Mixed-precision cost — host ORT + Ryzen AI NPU (report)

Spec: `docs/superpowers/specs/2026-09-24-mixed-precision-cost-design.md`
(protocol commit `6218e6b`). Characterization only: no thresholds, no verdict.
Subjects: E3 seed-0 as fp32 / R0 all-int8 QDQ / R2 head+refinement-fp32.

## 1 Results table (artefact × backend)

Latency: 10 warm-up + 10 scenes (head of `hailo_val`) × 5 repeats = 50 timed
`sess.run` samples, 368x1232 batch 1, `time.perf_counter`, `ORT_ENABLE_ALL`,
default thread settings (cpu_count 24, intra/inter-op 0 = default).
EPE: full 40-scene frozen contract, `contract_match` true everywhere below.

| artefact | size (bytes) | backend | median ms | p90 ms | FPS | EPE (px) |
|---|---|---|---|---|---|---|
| fp32 | 1701550 | host CPU EP | 766.75 | 774.96 | 1.30 | 1.18598534271453 |
| fp32 | 1701550 | host CUDA EP | — (unavailable) | — | — | — |
| fp32 | 1701550 | NPU VitisAI EP | — (compile timed out) | — | — | — |
| R0 all-int8 | 764520 | host CPU EP | 918.09 | 957.50 | 1.09 | 5.248941413313285 |
| R0 all-int8 | 764520 | host CUDA EP | — (unavailable) | — | — | — |
| R0 all-int8 | 764520 | NPU VitisAI EP | 652.59 | 668.44 | 1.53 | 70.85625512190461 |
| R2 head+refin fp32 | 1050225 | host CPU EP | 818.31 | 836.63 | 1.22 | 1.2908231335718434 |
| R2 head+refin fp32 | 1050225 | host CUDA EP | — (unavailable) | — | — | — |
| R2 head+refin fp32 | 1050225 | NPU VitisAI EP | 771.64 | 789.19 | 1.30 | 11.145662553725586 |

Numerics check (NPU-run EPE vs host CPU EPE, same artefact):

| artefact | EPE_npu | EPE_cpu | delta (npu − cpu) | NPU nonfinite px | valid px |
|---|---|---|---|---|---|
| R0 | 70.85625512190461 | 5.248941413313285 | +65.60731370859133 | 0 | 3802797 |
| R2 | 11.145662553725586 | 1.2908231335718434 | +9.854839420153743 | 0 | 3802797 |

Host CPU EPEs reproduce the recorded references bit-near (abs diff 0.0 vs
`int8_e3.json` / `int8_head_refinement.json` for all three artefacts,
including the regenerated R2 — the §1 R2 EPE gate passed).

The R0 NPU median (652.59 ms) reads faster than its CPU median (918.09 ms),
but its output is numerically wrong (EPE 70.86, D1 99.60): this is NOT a
usable speed-up, recorded here as a descriptive fact only.

## 2 NPU partition summary per artefact

Environment: NPU python onnxruntime 1.27.0, providers
`['VitisAIExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']`,
NPU type detected `STX`, provider options `[{}, {}]` (quicktest STX branch),
`log_severity_level = 1`, `ORT_ENABLE_ALL`. Session providers observed
`['VitisAIExecutionProvider', 'CPUExecutionProvider']` on R0 and R2.

- fp32: compile FAILED — timeout at the 25-min per-artefact cap. The
  10801-line compile log shows AIE kernel compilation in progress at the
  cap (kernels `Conv2DBf16`, `AddBf16`, `LeakyReluBf16`, `SubBf16`) with
  ZERO error/fail lines: still compiling, not erroring. EP-emitted files
  `original-info-signature.txt` / `original-model-signature.txt`
  (197609 bytes each) appeared under the output dir during this attempt.
  Latency/EPE: unmeasured.
- R0: compile OK in 65.1 s. Compile log is 3 lines (glog init + op
  registration) — no partition lines. EP-emitted files: none.
- R2: compile OK in 30.4 s. Compile log is 3 lines — no partition lines.
  EP-emitted files: none.

Per-op NPU-vs-CPU assignment: UNKNOWN for all three artefacts — not exposed
via the ORT API, and neither the compile logs nor EP-emitted files state it
(parse rule recorded in JSON: lines containing
`subgraph/partition/npu/vitisai/xclbin/fuse`; 0 hits on R0/R2; fp32 log is
kernel-compile chatter only). What is known: both R0 and R2 executed
end-to-end through sessions with VitisAI first + CPU fallback and returned
finite, contract-count-matching outputs.

## 3 Measured vs inferred vs unknown

MEASURED (this run): all table cells above; R2 regen sha
`b12d1e0d…` match + EPE reproduction (diff 0.0); H-CUDA unavailable
(observed providers `['AzureExecutionProvider', 'CPUExecutionProvider']`);
NPU compile outcomes/times; NPU latencies (R0/R2); NPU EPEs + deltas;
NPU type STX; session provider lists; VitisAI-session input order
`['left', 'right']` (verified post hoc — rules out swapped feeds as the
source of the EPE divergence); fp32 compile-log kernel list + 0-error
observation; EP-emitted signature files (fp32 attempt only).

INFERRED (kept apart from measurements): nothing in this report — the
numbers are stated without causal interpretation.

UNKNOWN / OPEN: per-op partition (§2); why the fp32 compile exceeds
25 min (still compiling at cap, no errors — whether it would finish with
more time is untested); why the NPU EPE diverges so far on the QDQ graphs
(70.86 / 11.15, all-finite, contract-matching — mechanism untested);
repeatability of every NPU number (single run each); all fp32 NPU cells;
any transfer beyond this machine/installation.

HYPOTHESIS (untested, not a finding): the pre-run expectation that
Conv3d/5-D ops and fp32 islands fall back to CPU is unconfirmed (no
partition data either way); the QDQ EPE divergence may come from int8/QDQ
handling on the NPU path (mechanism unknown).

## 4 Claim status

- MEASURED: size/latency/EPE cells in §1; CUDA-unavailable; fp32 NPU
  compile timeout with in-progress kernel compilation; R0/R2 NPU compile
  success; R0/R2 NPU EPE divergence (+65.61 / +9.85 px vs CPU);
  input-order verification.
- OPEN: everything listed under UNKNOWN above.
- HYPOTHESIS: the two untested statements in §3.

## 5 Provenance

- Protocol commit `6218e6b` ("docs(stage-e): measurement protocol for
  mixed-precision cost"); host run 2026-09-24T06:06:13Z–06:11:19Z;
  NPU run 2026-09-24T06:11:39Z–06:40:48Z (1749 s total, within the 90-min
  box; fp32 consumed the 25-min per-artefact cap).
- Records: `mixed_precision_cost.json` (host), `mixed_precision_cost_npu.json`
  (NPU, incl. per-artefact compile logs `*_npu_compile.log` kept alongside).
- Reuse: quantizer/scoring/sha (`int8_sensitivity`), kept-set/coverage
  (`int8_regression_head`), metrics/guard (`frozen_eval`), loader/normalize
  (`Kitti2015Stereo`) — all by import; no existing file modified.
- No training, no Kaggle, no environment changes. `.onnx`/`.npy` outputs
  are git-ignored; EP compile artefacts/logs/partials likewise ignored —
  only the scripts, the two JSONs and this report are committed.
