# Mixed-precision cost — host ORT + Ryzen AI NPU (characterization)

Date: 2026-09-24. Status: PRE-REGISTERED before any measurement under this
spec. This is CHARACTERIZATION, not an experiment: no accept/reject
thresholds, no gates with verdicts, no deployment claim. No training.
No Kaggle.

## 0 Purpose (frozen)

Measure what the seed-0 mixed-precision artefacts cost to run: file size,
host ONNX Runtime latency (CPU EP; CUDA EP if available), and Ryzen AI NPU
(VitisAI EP) compile behaviour, node partition, latency, plus a numerics
check (NPU-run 40-scene contract EPE vs host CPU EPE of the same artefact).

Grounds (recorded references, NOT re-claimed here):

- `stage_e_recipe/int8_head_refinement/int8_head_refinement.json`
  (commit `0fbceda`): seed-0 arms fp32 EPE 1.18598534271453, R0 EPE
  5.248941413313285, R2 EPE 1.2908231335718434, all `contract_match: true`,
  3802797 valid pixels.
- `stage_e_recipe/int8_e3/int8_e3.json` seed "0": fp32 export sha256
  `7453b2be…9645` (1701550 bytes, opset 13, 712 nodes), int8 quantize
  sha256 `8b31f949…68a` (764520 bytes), same procedure + calibration as
  the R0 baseline (the head-refinement R0 baseline reproduces this sha
  exactly, so the quantizer is deterministic on this machine).
- `stage_c_deploy/runtime/README.md` §§1–3: host latency precedent —
  medians over 10 scenes from the head of `hailo_val`, 368x1232,
  `time.perf_counter` around inference.

## 1 Subjects (frozen)

E3 seed-0 best checkpoint, as three ONNX artefacts:

| tag | artefact | recorded sha256 | bytes | location |
|---|---|---|---|---|
| fp32 | E3 seed-0 fp32 export | `7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645` | 1701550 | `stage_e_recipe/int8_e3/e3_seed0_best_fp32.onnx` (on disk) |
| R0 | all-int8 QDQ baseline | `8b31f94921878f7f08ae122e00a1e36983fc2fb19dae5af3c07566589598c68a` | 764520 | `stage_e_recipe/int8_e3/e3_seed0_best_int8.onnx` (on disk) |
| R2 | head+refinement fp32, rest int8 | `b12d1e0dd591750d8711ef77d528fe5756ba0f05cfdfe3bd390fc0b6dfb644ef` | 1050225 | NOT on disk (deleted after scoring per precedent) — regenerate (see §2) |

- fp32 and R0 MUST be sha-verified against the recorded values before any
  measurement. If either differs: STOP, record, no measurement.
- R2 MUST be regenerated with the existing code (§2) into the new output
  dir only (`stage_e_recipe/mixed_precision_cost/r2_regen/`), then
  sha-verified against `b12d1e0d…`. If the sha differs: STOP, do not
  measure a non-identical graph. (Regeneration uses the host python and
  the §2 quantizer; the `.onnx` stays under the new output dir, which is
  git-ignored for `*.onnx`.)
- Extra safety re-score (host script, §5): the regenerated R2 is scored
  once on the 40-scene contract; `|EPE − 1.2908231335718434| <= 1e-6` px
  is REQUIRED to proceed to latency. If it fails: STOP (regeneration did
  not reproduce behaviour despite matching bytes — record both facts).

## 2 R2 regeneration procedure (frozen)

Exact `stage_e_recipe/int8_head_refinement.py` R2 path, seed 0 only, via
import (no copies, no edits to existing files):

- `import int8_sensitivity as S`, `import int8_regression_head as RH`.
- fp32 subject `stage_e_recipe/int8_e3/e3_seed0_best_fp32.onnx`,
  sha-asserted per §1.
- excludes = `sorted(set(head) | set(refin))` where `head` =
  `RH.kept_set(fp32)` (cross-checked against runtime `/regression/`
  prefix enumeration) and `refin` = runtime `/refinement/` prefix
  enumeration; assert `/regression/Mul`, `/regression/Mul_1`,
  `/regression/Mul_2`, `/refinement/input_conv/Conv`,
  `/refinement/output_conv/Conv` all present.
- `S.quantize_with_excludes(fp32, dst, calib, excludes)` with
  `calib` = first 32 scenes of `hailo_calib` (`S.CALIB_N`).
- Assert `S.sha256(dst) == b12d1e0d…`; record bytes + quantize seconds.

## 3 Backends and metrics (frozen)

Backends per artefact:

- H-CPU: host python ONNX Runtime, `CPUExecutionProvider` only,
  `ORT_ENABLE_ALL`, inference-only timing (§4).
- H-CUDA: host ONNX Runtime `CUDAExecutionProvider` — EXPECTED
  UNAVAILABLE: host onnxruntime 1.27.0 reports available providers
  `['AzureExecutionProvider', 'CPUExecutionProvider']` (checked
  2026-09-24). The script MUST check `ort.get_available_providers()` at
  runtime; if CUDA is absent, record `cuda: unavailable` with the
  observed provider list. No torch-CUDA timing substitute (that would
  not be an ORT measurement).
- NPU: NPU python (`C:\Users\vishn\anaconda3\envs\ryzen-ai-1.8.0\python.exe`,
  onnxruntime 1.27.0 with `VitisAIExecutionProvider`), session providers
  `["VitisAIExecutionProvider", "CPUExecutionProvider"]`, provider
  options `[{}, {}]` — this mirrors `C:\Program Files\RyzenAI\1.8.0\quicktest\quicktest.py`
  exactly for the STX path (Ryzen AI 9 HX 370 is Strix Point; quicktest
  specifies empty options, no config file, no cache_dir for STX; only
  PHX/HPT gets an xclbin/target block). The script replicates quicktest's
  `pnputil` NPU-type detection, records the detected type, and applies
  the matching branch. `RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\1.8.0`.
  `SessionOptions.log_severity_level = 1`, `ORT_ENABLE_ALL`, as quicktest.
  CWD is the output dir, so any EP-emitted cache/report files land under
  it (Program Files is read-only; nothing is written there).

Metrics per (artefact × backend):

- file size (bytes, `st_size`).
- latency: median ms, p90 ms, FPS (= 1000/median) per §4.
- NPU only: compile success (boolean + wall seconds + exact error text on
  failure), node partition summary from the EP's report files (see §6),
  40-scene contract EPE of the NPU-run output and its delta vs the host
  CPU EPE of the same artefact (numerics check, §5).

## 4 Latency protocol (frozen)

- Inputs: 10 scenes from the head of `hailo_val` (scenes 0–9),
  `normalize()` preprocessing, 368x1232, batch 1. Inputs are precomputed
  ONCE with the host python into `.npy` under the output dir
  (`inputs_npy/scene{i}_{left,right}.npy`) and loaded from there by BOTH
  scripts, so every backend times bit-identical inputs. The script
  asserts each input is `(1, 3, 368, 1232)` float32 contiguous.
- Timing scope: `sess.run` ONLY (preprocessing excluded — inputs are
  already normalized arrays; output postprocessing excluded — the raw
  output array is returned, no EPE math inside the timed region).
- Wall clock: `time.perf_counter()` around each `sess.run`.
- Warm-up: 10 untimed `sess.run` calls on the scene-0 input per
  artefact/backend, then timed runs.
- Timed runs: 10 scenes × N repeats, N = 5 → 50 timed samples per
  artefact/backend, looped scene0×5, scene1×5, … (fixed order, recorded).
- Statistics: median and p90 (nearest-rank) over the 50 samples; FPS =
  1000/median_ms. Raw samples + per-scene medians recorded in JSON.
- Thread settings: NO explicit values set (EP/session defaults); the
  script records `os.cpu_count()` and the session's
  `intra_op_num_threads` / `inter_op_num_threads` as observed.
- Host CPU sessions: `providers=["CPUExecutionProvider"]`,
  `graph_optimization_level=ORT_ENABLE_ALL` (same as the scoring
  precedent).
- NPU sessions: per §3. Compile wall seconds recorded separately from
  inference latency (session creation timed apart from the §4 loop).

## 5 Accuracy / numerics protocol (frozen)

- Host script re-scores all three artefacts on the FULL 40-scene frozen
  contract by importing `S.score_contract` (unchanged behaviour:
  `ORT_ENABLE_ALL`, CPU EP, `pooled_metrics` +
  `refuse_unless_contract`, `contract_match` REQUIRED). Expected cost
  ~35–40 s per artefact (recorded `score_s` precedent). The R2 re-score
  doubles as the §1 regeneration gate.
- NPU script scores all three artefacts on the same 40 scenes through the
  VitisAI session, reusing `pooled_metrics`, `refuse_unless_contract`
  and `normalize` by import; only session construction differs (§3).
  Dataset loads via the existing `Kitti2015Stereo` import (verified
  importable under the NPU python on 2026-09-24); if the import fails at
  runtime, STOP that part and record the exact error (the host-precomputed
  `.npy` covers the 10 latency scenes only, NOT a 40-scene fallback —
  do not silently score a subset: `contract_match` must be 40 scenes or
  the row is marked failed).
- Numerics check per artefact: `EPE_npu`, `EPE_cpu`,
  `delta = EPE_npu − EPE_cpu` (px), plus nonfinite pixel counts and
  `contract_match` on both sides. No threshold — record the numbers.

## 6 NPU partition protocol (frozen)

Expectation (recorded upfront, NOT a result): Conv3d / 5-D ops and fp32
islands fall back to CPU; that fallback IS a result — record it, do not
work around it.

- During VitisAI session creation, the script redirects fd-level stderr
  to `<artefact>_npu_compile.log` (restored afterwards) to capture the
  EP's compile/partition messages; the log path is recorded and the file
  kept under the output dir.
- After session creation the script records: `session.get_providers()`,
  any new files the EP wrote under the output dir (before/after
  directory listing diff), and any subgraph/partition lines parsed from
  the compile log (with the parse rule recorded alongside).
- Anything not observable through the above is recorded as `unknown`
  with the exact observation (e.g. "EP wrote no report file; per-op
  assignment not exposed via the ORT API"), never inferred.
- Per-artefact compile budget 25 min; whole-NPU-part budget 90 min from
  first NPU session creation. On compile failure or timeout: STOP that
  artefact, record the exact error text + elapsed seconds, and continue
  with the next artefact only (latency/EPE rows for the failed artefact
  marked `failed`, never filled from another backend).

## 7 Time boxes and STOP rules (frozen)

- Host part: no fixed box (expected ~10 min: R2 regen ~40 s quantize +
  3 × ~40 s contract re-scores + 3 × 50 timed runs at ~0.8 s).
- NPU part: 90 min total; 25 min per-artefact compile cap (§6).
- STOP rules: §1 sha mismatch (any artefact); §1 R2 EPE gate; H-CUDA
  handling per §3 (record unavailable, not a failure); NPU compile
  failure/timeout per §6 (part-level STOP, record, continue only with
  independent artefacts); any `contract_match == false` on a re-score
  (STOP that row, record, continue with independent rows only).

## 8 Outputs (new files only)

- Script (new): `stage_e_recipe/mixed_precision_cost.py` — host part
  (R2 regen + sha asserts + `.npy` precompute + H-CPU latency + H-CUDA
  availability check + 40-scene CPU re-scores). Reuses
  `int8_sensitivity` / `int8_regression_head` / `frozen_eval` /
  `Kitti2015Stereo` / `normalize` by import; no copies unless stated.
- Script (new): `stage_e_recipe/mixed_precision_cost_npu.py` — NPU part
  (VitisAI sessions per §3, compile logs, NPU latency per §4 on the same
  `.npy` inputs, 40-scene NPU EPE per §5, partition summary per §6).
  Run with the NPU python.
- Output dir (new): `stage_e_recipe/mixed_precision_cost/` containing:
  `mixed_precision_cost.json` (host part record: UTC, git HEAD, host
  environment incl. `ort.get_available_providers()`, sha asserts, R2
  regen provenance, thread settings, per-artefact size/latency/EPE rows,
  raw samples), `mixed_precision_cost_npu.json` (NPU record: UTC,
  NPU-python environment, NPU-type detection, provider options used,
  per-artefact compile result + log paths + partition summary +
  latency + EPE rows), `inputs_npy/` (10-scene precomputed inputs),
  `r2_regen/` (regenerated R2 `.onnx`, git-ignored), `*_npu_compile.log`
  files, and `REPORT.md` (see §9).
- Measured / inferred / unknown are kept in separate labeled sections
  in both JSONs. Never claim a number not measured in this run; cited
  reference numbers are labeled "recorded reference".

## 9 Report (frozen shape)

`stage_e_recipe/mixed_precision_cost/REPORT.md` holds: one table
(artefact × backend → size / median ms / p90 ms / FPS / EPE), the NPU
partition summary per artefact, a measured-vs-inferred-vs-unknown split,
and a claim-status section (`MEASURED` / `OPEN` / `HYPOTHESIS`) — with no
verdict language (characterization only).

## 10 Hard prohibitions (binding)

- Do not modify `phase1/harness/frozen_eval.py`, any `stage_e_recipe/*.py`
  existing script, existing experiment outputs, bundles, checkpoints, or
  `e3_*` files. Do not touch `phase1/`, `src/`.
- Do not commit `.onnx` files or NPU cache dirs to git (`*.onnx`,
  `*.npy` already git-ignored; add ignore lines if an EP cache dir would
  be picked up).
- No training. No Kaggle. No environment changes (no installs, upgrades,
  driver changes, PATH edits).
- This spec's commit and the implementation/report commit stay LOCAL —
  do NOT push them.
