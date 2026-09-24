# Quark INT8 on the Ryzen AI NPU (characterization)

Date: 2026-09-24. Status: PRE-REGISTERED before any quantization or NPU run
under this spec. This is CHARACTERIZATION, not an experiment: no accept/reject
thresholds, no gates with verdicts, no deployment claim. No training.
No Kaggle.

## 0 Purpose (frozen)

Measure what Quark-quantized INT8 variants of the seed-0 fp32 model cost to
run: quantization success, quantized file size, host CPU EP 40-scene contract
EPE (numeric reference per arm), Ryzen AI NPU (VitisAI EP) compile behaviour
(success + wall seconds + INT8-vs-Bf16 kernel evidence from the compile log),
NPU latency, NPU 40-scene contract EPE and its delta vs the arm's own CPU EPE,
plus any per-op/partition info the EP or aianalyzer exposes.

Grounds (recorded references, NOT re-claimed here):

- `stage_e_recipe/mixed_precision_cost/mixed_precision_cost.json`
  (commit `65ac304`): E3 seed-0 fp32 sha256
  `7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645`
  (1701550 bytes, opset 13, 712 nodes), 40-scene contract EPEs
  fp32 1.18598534271453 / R0 5.248941413313285 / R2 1.2908231335718434.
- `stage_e_recipe/mixed_precision_cost/mixed_precision_cost_npu.json`
  (commit `65ac304`): NPU precedent on THIS machine (STX, VitisAI EP,
  provider options `[{}, {}]`): R0 compile 65.1 s, NPU EPE 70.85625512190461
  (delta +65.61 vs CPU); R2 compile 30.4 s, NPU EPE 11.145662553725586
  (delta +9.85 vs CPU); fp32 compile timed out at the 25-min cap with only
  `*Bf16` kernels (`Conv2DBf16`, `AddBf16`, `LeakyReluBf16`, `SubBf16`) in
  the compile log; R0/R2 compile logs were 3 lines (no partition lines);
  per-op NPU/CPU assignment UNKNOWN (not exposed via the ORT API).
- `stage_e_recipe/int8_head_refinement.py`: node-set precedent —
  `/regression/*` (head) and `/refinement/*` (refinement) enumerated at
  runtime by name prefix; required nodes `/regression/Mul`, `/regression/Mul_1`,
  `/regression/Mul_2`, `/refinement/input_conv/Conv`,
  `/refinement/output_conv/Conv`; calibration = first 32 scenes of
  `hailo_calib` (`S.CALIB_N`).

## 1 Subject (frozen)

E3 seed-0 best fp32 ONNX,
`stage_e_recipe/int8_e3/e3_seed0_best_fp32.onnx`, sha256 asserted before any
quantization against `mixed_precision_cost.json`:
`7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645`
(1701550 bytes). On mismatch: STOP, record, no quantization.

## 2 Arms (frozen)

All arms quantized with Quark from the same fp32 ONNX (§1) and the same
calibration set: first 32 scenes of `hailo_calib`, `normalize()`
preprocessing, both inputs (`left`, `right`), via a Quark
`CalibrationDataReader` whose `get_next()` yields `{left: ..., right: ...}`
per scene (same pixels the ORT-QDQ precedent calibrates on).

| arm | Quark config | meaning |
|---|---|---|
| Q1 | `QConfig.get_default_config("XINT8")` | AMD's recommended NPU int8 config: symmetric INT8 activations/weights/bias, power-of-two scales |
| Q2 | `QConfig.get_default_config("A16W8")` (name per installed Quark; `A8W8` fallback name if the installed version spells it differently — record the exact name used) | 16-bit activations, 8-bit weights, float scales — attempt IFF the installed Quark exposes it for ONNX Ryzen AI flow; if not exposed, record the exact error/available names and mark Q2 NOT IMPLEMENTABLE |
| Q3 | Q1 config with the `/regression/*` and `/refinement/*` nodes excluded (kept fp32) — same node sets as `int8_head_refinement.py` R2 (runtime prefix enumeration, required nodes asserted per §0) | XINT8 with head+refinement in fp32 |

Quark API (per Quark 0.12.post1 docs for Ryzen AI):

```python
from quark.onnx import ModelQuantizer, QConfig
quantization_config = QConfig.get_default_config("XINT8")  # or "A16W8"
quantizer = ModelQuantizer(quantization_config)
quantizer.quantize_model(float_model_path, quantized_model_path, calib_data_reader)
```

Node exclusion for Q3 uses whatever node-exclusion mechanism the installed
amd-quark exposes (e.g. a `Config` exclude list or per-layer opt-out). The
exact kwarg is resolved at implement time from the installed package and
recorded verbatim. If Quark exposes no node-name exclusion for this flow,
Q3 is marked NOT IMPLEMENTABLE with the exact evidence (API listing/error),
and Q1/Q2 proceed independently. No AdaRound/AdaQuant/CLE/fast-finetune in
any arm (default configs only — characterization of the defaults).

amd-quark version: the version AMD's Ryzen AI 1.8 doc set pairs with
(Quark docs `latest` = 0.12.post1 at spec time; candidate
`amd-quark==0.12.post1`, the newest on PyPI). The install step MUST run
`pip install --dry-run` first and pick the newest amd-quark whose
requirements do NOT force changes to `onnxruntime-vitisai` (1.27.0),
`onnxruntime`, or `numpy` (1.26.4) already in the clone; the dry-run output
and the final `pip list` diff are recorded in the report. Install target is
the clone env ONLY (see §7).

## 3 Backends and metrics (frozen)

Quantization + CPU reference run in the clone env
(`ryzen-ai-1.8.0-quark`, §7; onnxruntime-vitisai 1.27.0 ships the CPU EP).

- Q-CPU (numeric reference per arm): `CPUExecutionProvider`,
  `ORT_ENABLE_ALL`, full 40-scene frozen contract by importing
  `S.score_contract`-equivalent behaviour (`pooled_metrics` +
  `refuse_unless_contract`, `contract_match` REQUIRED). Quark performs graph
  optimization automatically, so the Quark graph is scored as emitted — no
  byte-identity expectation vs ORT-QDQ graphs.
- NPU: VitisAI EP on STX with the provider options AMD documents for INT8
  models. `quicktest.py` uses empty options `[{}]` for STX; the INT8
  getting-started flow (`predict.py`, Ryzen AI 1.8 docs) uses
  `providers=['VitisAIExecutionProvider']` with
  `provider_options=[{'cacheDir': <run dir>, 'cacheKey': <key>}]` and prints
  `[Vitis AI EP] No. of Operators : CPU x IPU y z%` /
  `[Vitis AI EP] No. of Subgraphs : ...` lines. This spec uses, per arm:
  `providers=["VitisAIExecutionProvider", "CPUExecutionProvider"]`,
  `provider_options=[{"cacheDir": <arm subdir under the output dir>, "cacheKey": "npu-quark-<arm>"}, {}]`
  (two-provider list mirrors the cost-spec precedent; cache options mirror
  the documented INT8 flow; STX gets no `target`/`xclbin` block per
  quicktest). `SessionOptions.log_severity_level = 1`, `ORT_ENABLE_ALL`, as
  quicktest. `RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\1.8.0`.
  CWD is the output dir, so EP-emitted cache/report files land under it.
  The exact options used are recorded verbatim per arm in the JSON.

Metrics per arm (all recorded, none thresholded):

- quantization success (boolean + wall seconds + exact error text on
  failure + output sha256/bytes).
- quantized file size (bytes, `st_size`).
- Q-CPU 40-scene contract EPE (numeric reference) + `contract_match`.
- NPU compile success (boolean + wall seconds + exact error text on
  failure).
- kernel-type evidence: count of kernel names by type in the compile log —
  names containing `nt8`/`NT8`/`nt16`/`NT16` (case-insensitive) counted as
  INT8-family, names containing `bf16`/`Bf16`/`BF16` counted as Bf16-family,
  with the count rule recorded alongside. (Precedent: fp32 log showed only
  `*Bf16` kernels.) Zero hits is recorded as zero, not as failure.
- NPU latency: median ms per §4.
- NPU 40-scene EPE and `delta = EPE_npu − EPE_cpu_own_arm` (px), plus
  nonfinite pixel counts and `contract_match` on both sides.
- per-op/partition info: the `[Vitis AI EP] No. of Operators / No. of
  Subgraphs` lines (they print to stdout — capture stdout AND stderr to
  `<arm>_npu_compile.log` via fd-level redirect around session creation,
  exactly as the cost-spec NPU script does for stderr), before/after
  directory diff of EP-emitted files under the output dir, and anything
  `aianalyzer` (wheel shipped in the Ryzen AI 1.8.0 install; installed into
  the clone only, §7) exposes about the compiled artefact. Anything not
  observable is `unknown` with the exact observation, never inferred.

## 4 Latency protocol (frozen)

Identical to the cost-spec §4 so numbers are comparable: inputs reused from
`stage_e_recipe/mixed_precision_cost/inputs_npy/` (10 head scenes 0–9,
`normalize()`, 368x1232 batch 1, float32 contiguous — asserted on load, NOT
regenerated); timing scope `sess.run` ONLY; `time.perf_counter()`; 10 untimed
warm-up runs on scene-0 input; timed runs 10 scenes × 5 repeats = 50 samples
(scene0×5, scene1×5, … fixed order); statistics median + p90 (nearest-rank)
+ FPS (= 1000/median); raw samples + per-scene medians recorded; no explicit
thread values (record `os.cpu_count()` + intra/inter-op as observed).
NPU sessions per §3; compile wall seconds recorded separately from the
latency loop.

## 5 Accuracy / numerics protocol (frozen)

- Q-CPU reference (§3) doubles as the per-arm numerics baseline.
- NPU script scores each compiled arm on the same 40 scenes through the
  VitisAI session, reusing `pooled_metrics`, `refuse_unless_contract` and
  `normalize` by import; only session construction differs (§3). Dataset via
  the existing `Kitti2015Stereo` import under the clone python; if the import
  fails at runtime, STOP that part and record the exact error — do not
  silently score a subset (`contract_match` must be 40 scenes or the row is
  marked failed).
- Numerics check per arm: `EPE_npu`, `EPE_cpu_own_arm`,
  `delta = EPE_npu − EPE_cpu_own_arm`, nonfinite counts, `contract_match`
  both sides. No threshold — record the numbers.

## 6 Time boxes and STOP rules (frozen)

- Steps 3–4 (implement + run) share a 2 h box from first quantization
  attempt; 25 min per-arm NPU compile cap (same driver/worker split as
  `mixed_precision_cost_npu.py`: one worker subprocess per arm).
- STOP rules: §1 sha mismatch (whole run); per-arm quantization failure
  (STOP that arm with exact error, continue independent arms); NPU compile
  failure/timeout (STOP that arm, record error + elapsed, continue
  independent arms; latency/EPE rows of a failed arm marked `failed`, never
  filled from another arm/backend); any `contract_match == false` (STOP that
  row, record, continue independent rows only); Q2/Q3 NOT IMPLEMENTABLE
  handling per §2 (record evidence, continue).

## 7 Environment (frozen)

ONE new conda env, created by cloning ONLY:
`conda create -y -n ryzen-ai-1.8.0-quark --clone ryzen-ai-1.8.0`, then
pip-install amd-quark (version per §2, dry-run first) plus, optionally, the
`aianalyzer-1.8.0-py3-none-any.whl` wheel copied from
`C:\Program Files\RyzenAI\1.8.0` (read-only — copy before install) into THAT
clone only. NEVER modify the original `ryzen-ai-1.8.0` env, the host pyenv
python, PATH, drivers or system files. If the clone or install fails, STOP
and report the exact error. Host python
(`C:\Users\vishn\.pyenv\pyenv-win\versions\3.12.9\python.exe`) runs nothing
new in this spec (inputs are reused, not regenerated).

## 8 Outputs (new files only)

- Script (new): `stage_e_recipe/npu_quark.py` (+ helpers iff needed) —
  Quark quantization per §2 (clone python), Q-CPU re-scores per §3/§5,
  VitisAI sessions per §3, compile logs (stdout+stderr), NPU latency per §4,
  40-scene NPU EPE per §5, partition summary per §3. Reuses
  `mixed_precision_cost/inputs_npy/` inputs and existing scoring by import;
  no copies unless stated.
- Output dir (new): `stage_e_recipe/npu_quark/` containing per-arm quantized
  `.onnx` (git-ignored; kept under the output dir for the NPU run, deleted
  or kept per §9 note — record whichever), `npu_quark.json` (full record:
  UTC, git HEAD, clone environment incl. `ort.get_available_providers()`,
  amd-quark version, sha asserts, per-arm quantize provenance + sizes +
  CPU EPEs + compile results + log paths + kernel-type counts + latency +
  EPE rows, provider options verbatim), `<arm>_npu_compile.log` files, and
  `REPORT.md` (see §9).
- Measured / inferred / unknown are kept in separate labeled sections.
  Never claim a number not measured in this run; cited reference numbers
  are labeled "recorded reference".

## 9 Report (frozen shape)

`stage_e_recipe/npu_quark/REPORT.md` holds: one table
(arm → quantized size / CPU EPE / NPU compile result + seconds + INT8-vs-Bf16
kernel evidence / NPU median ms / NPU EPE / NPU-minus-CPU delta), a
measured-vs-inferred-vs-unknown split, and a claim-status section
(`MEASURED` / `OPEN` / `HYPOTHESIS`) — with no verdict language
(characterization only). The report also records: the clone + install
commands and what pip changed (dry-run vs final diff); the protocol commit
hash/time vs the first-quantization time (proving pre-registration); the
exact NPU provider options used; kernel-type evidence per arm; exact errors
for any failed/NOT IMPLEMENTABLE arm.

## 10 Hard prohibitions (binding)

- Do not modify any existing `stage_e_recipe/*.py` script, existing run
  outputs, `phase1/`, `src/`, `.onnx`/caches/inputs outside the new output
  dir. Do not touch the original `ryzen-ai-1.8.0` env.
- Do not commit `.onnx` files, NPU cache dirs, or `.npy` inputs
  (`*.onnx`, `*.npy` already git-ignored; extend `.gitignore` iff a Quark or
  EP cache dir would be picked up).
- No training. No Kaggle. No `--force` anywhere.
- This spec's commit and the implementation/report commit stay LOCAL — do
  NOT push them.

## Amendment A1 (2026-09-24, after the attempt-1/2 failures, before attempt 3)

Prior failures recorded here; §§0–10 above stay frozen.

- Attempt 1 (2026-09-24T07:00:52.578915Z start, first quantize
  2026-09-24T07:00:52.883547Z, end 2026-09-24T07:01:11.403659Z, 18.5 s):
  all 3 arms failed in quantization with the exact per-arm error
  `ValueError: axes don't match array` (recorded in the run's
  `npu_quark.json` rows Q1/Q2/Q3). Likely cause: the default flow's CLE
  pre-pass (`quark/onnx/algorithm/cle/equalization.py`
  `_cross_layer_equalize`: `tail_w_data.transpose(1, 0, 2, 3)` assumes 4-D
  conv weights; the model has 5 Conv3d ops with 5-D weights among 51 Convs).
- Attempt 2 (same fp32 subject, `include_cle=False` deviation already in
  `stage_e_recipe/npu_quark.py`): reached calibration —
  `stage_e_recipe/npu_quark/quantized_info.csv` records `pre process,2.26`
  and `calibration: collect data (onnx inference + numpy statistics),208.52`
  — then committed memory grew to ~10.5 GB and free physical RAM on this
  ~15 GB laptop fell to 0; no quantized model was produced and no CPU/NPU
  rows completed. The manager killed the run on the user's instruction.
  No NPU compile/latency/EPE was reached in attempt 1 or 2.

Attempt-3 deviations (all recorded in code + JSON + REPORT):

- `include_cle=False` on all arms (carried over from attempt 2; default
  flow crashes on Conv3d 5-D weights per attempt 1).
- Calibration method: `onnxruntime.quantization.calibrate.CalibrationMethod.MinMax`
  on all arms (explicitly set, replacing the XINT8 default
  `PowerOfTwoMethod.MinMSE`). Installed `amd-quark==0.11.2` /
  `onnxruntime==1.27.0` expose `CalibrationMethod.{MinMax, Entropy,
  Percentile, Distribution}` and `PowerOfTwoMethod.{NonOverflow, MinMSE}`;
  there is no separate "running-range" method name — MinMax IS the
  min/max running-range method and is named exactly as above.
- Calibration set: first 8 scenes of `hailo_calib` (cut from 32),
  `normalize()` preprocessing, both inputs, reader yields one scene at a
  time (batch 1, same pixels as before).
- Low-memory calibration option: inspected all `QuantizationConfig`
  fields in installed amd-quark 0.11.2 — there is NO dedicated
  per-batch / streaming / low-memory calibration flag. The only
  memory-relevant option is `use_external_data_format` (default False);
  attempt 3 sets `use_external_data_format=True` so quantized tensors go
  to an external data file instead of being held inline. Recorded
  verbatim in per-arm config; stated here as "no streaming flag exists".
- Execution: arms run strictly one at a time, each in its own separate
  child process, Q1 first; Q2 and Q3 run only if Q1 finishes within the
  memory guard. Parent watchdog polls free physical RAM every 2 s and
  kills the child if free < 2.0 GB.
- Q1 keeps the power-of-two XINT8 scale type: activation `QUInt8`,
  weight `QInt8`, `ActivationSymmetric=True`, `QDQ` format — only the
  calibration threshold method changes to MinMax.
