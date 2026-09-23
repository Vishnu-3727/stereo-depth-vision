# INT8 per-layer sensitivity — E3 seed-0 (pre-registered diagnostic)

Date: 2026-09-23. Status: PRE-REGISTERED before any measurement under this spec.
No accept/reject. Diagnostic only. No training.

## 1 Subject (frozen)

- Model: E3 seed-0 best checkpoint, `stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_best.pth`.
- FP32 ONNX: reuse `stage_e_recipe/int8_e3/e3_seed0_best_fp32.onnx` **only because**
  its provenance is recorded in `stage_e_recipe/int8_e3/int8_e3.json`
  (seed "0" → export: sha256
  `7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645`,
  bytes 1701550, opset 13, n_nodes 712,
  parent_checkpoint_sha256
  `7d3c2109c11339b6fb30cc33f5bd042667740f3bc7345768f4e4b04704836209`,
  params 397954).
- The script MUST assert at start: file exists, sha256 equals the recorded
  value, `n_nodes == 712`, params implied by reuse (no re-export, no weight
  change). If any assertion fails: STOP, do not re-export silently — re-export
  only via the existing `export_onnx` path from `stage_e_recipe/int8_control.py`
  and record fresh provenance in the output JSON.
- Recorded E3 seed-0 reference numbers (from `int8_e3.json`, NOT re-claimed here):
  fp32 ONNX EPE 1.18598534271453, int8 ONNX EPE 5.248941413313285,
  P = 4.062956070598755 px, both `contract_match: true`, 3802797 valid pixels.

## 2 Quantizer (frozen, identical to control)

- Exact `stage_e_recipe/int8_control.py` `quantize()` procedure, no deviations:
  `quantize_static`, `QuantFormat.QDQ`, `activation_type=QInt8`,
  `weight_type=QInt8`, `per_channel=True`, default `op_types_to_quantize`
  (None), default calibrate method (MinMax), no pre-processing step
  (same omission as control and EXP-015 — like-for-like).
- Calibration data: first 32 scenes of `hailo_calib` split
  (`Kitti2015Stereo(... split="hailo_calib")`, `CALIB_N = 32`), identical
  `Reader` construction and `normalize()` preprocessing. `hailo_val` is never
  used for calibration.

## 3 Layer groups (frozen definition)

- Quantizable node group = one ONNX node with `op_type == "Conv"`.
- Groups are enumerated at runtime from the fp32 ONNX graph
  (`[n.name for n in model.graph.node if n.op_type == "Conv"]`), sorted.
  Observed at spec time: 51 Conv nodes out of 712 total nodes in
  `e3_seed0_best_fp32.onnx`. The script MUST assert the enumerated list is
  non-empty and record the full list; if the count differs from 51, record the
  actual count and proceed with the actual list (count is a measured property
  of the graph, not a gate).
- No other op types are swept. Rationale (inferred, not a measured claim):
  Conv nodes are the quantized compute in this graph under the default
  `op_types_to_quantize`; sweeping them isolates per-layer contribution.

## 4 Sweeps (frozen)

For each Conv node name L in the enumerated list, produce one quantized model
per sweep arm by passing through to `quantize_static`:

- Arm A — leave-one-out (all int8 except L fp32): `nodes_to_exclude=[L]`,
  all other quantizer arguments identical to §2.
- Arm B — only-one (only L int8, everything else fp32):
  `nodes_to_exclude=[all Conv names except L]`.
- Plus two baseline rows: all-fp32 (the reused fp32 ONNX, scored directly, no
  quantization) and all-int8 (`nodes_to_exclude=None`, exact §2 procedure).
- Total configs: 2 + 2 × N (N = 51 observed → 104 scorings).
- Each quantized artifact is written under `stage_e_recipe/int8_sensitivity/`
  only, never beside existing bundles. Filenames deterministic:
  `loo_<i>_<sanitized>.onnx`, `only_<i>_<sanitized>.onnx`, plus the two
  baselines (`baseline_fp32` = reference to reused file by sha, `baseline_int8`
  = freshly quantized `baseline_int8.onnx`).

## 5 Scoring (frozen)

- Scoring code reuses the existing functions verbatim in behavior:
  `score_onnx_model` + `pooled_metrics` + `refuse_unless_contract` from
  `stage_e_recipe/int8_control.py` / `phase1/harness/frozen_eval.py`.
  `phase1/harness/frozen_eval.py` MUST NOT be modified.
- Full-contract scoring: all 40 `hailo_val` scenes, `disparity_scale=256.0`,
  `occluded=True`, `normalize()` inputs, ORT CPU with
  `ORT_ENABLE_ALL`, `contract_match` REQUIRED (raise / STOP on mismatch).
- Record per config: EPE, RMSE, D1, BAD1–3, valid pixels, guard block,
  nonfinite pixel count, score seconds, quantized artifact sha256, node-group
  name, arm label. Scored EPE is the ranking metric.

## 6 Baseline reproduction gate (STOP rule, frozen)

- Before ranking any sweep row, the script scores both baselines on the
  40-scene contract and checks:
  - `|EPE_fp32_measured − 1.18598534271453| <= 1e-6` px, and
  - `|EPE_int8_measured − 5.248941413313285| <= 1e-6` px, and
  - both guards `contract_match == true`.
- If ANY check fails: STOP. Write what was measured, do not proceed to sweep
  ranking, do not reinterpret. (Tolerance rationale: same machine, same
  deterministic CPU procedure — bit-near reproducibility expected; 1e-6
  absorbs fp32 reduction-order noise only.)

## 7 Pre-registered readout (frozen)

- EPE gap G = EPE(int8 baseline, measured) − EPE(fp32 baseline, measured).
- For each layer L in Arm A: recovered fraction
  `R_L = (EPE_int8 − EPE_loo_L) / G`.
- Dominant layer (diagnostic label only): any L with `R_L >= 0.50`, i.e.
  keeping L in fp32 recovers at least half the int8→fp32 EPE gap.
- Report two ranked tables (Arm A sorted by EPE ascending = most recovered
  first; Arm B sorted by EPE descending = most damaged by single-layer
  quantization first), each row with EPE, Δ vs fp32 baseline, Δ vs int8
  baseline, R_L (Arm A), D1, nonfinite count, contract_match.
- No accept/reject, no gate, no deployment claim. NOT Hailo evidence
  (same disclaimer as `int8_control.py`: Stage D remains blocked).

## 8 Budget rule (frozen before starting)

- The script MUST time one full 40-scene scoring pass first (single ONNX
  artifact, wall seconds S_score) and one quantization pass (wall seconds
  S_quant), then estimate
  `T_est = (2 + 2N) × (S_quant + S_score)` seconds for the full sweep.
- If `T_est <= 6 h` (21600 s): run the entire sweep on the 40-scene contract.
- If `T_est > 6 h`: run the sweep on the 10-scene monitor instead — defined as
  the FIRST 10 scenes of the `hailo_val` dataset in dataset order (same
  `limit=10` convention as the train-time monitor), scored with
  `pooled_metrics` WITHOUT the contract guard (explicitly labeled
  `contract: 10-scene-monitor, not-frozen-contract, diagnostic-only`), then
  re-score ONLY the top 5 layers of each arm on the full 40-scene contract
  with `contract_match` required, and rank the report on the 40-scene
  re-scores. Monitor rows are never presented as contract numbers.
- The chosen branch, S_score, S_quant, N, and T_est are recorded in the output
  JSON and REPORT.md.

## 9 Outputs (new files only)

- Script: `stage_e_recipe/int8_sensitivity.py` (new file).
- Output dir: `stage_e_recipe/int8_sensitivity/` (new dir) containing:
  `int8_sensitivity.json` (per-layer rows + provenance: UTC, git HEAD, ORT /
  torch / numpy versions, fp32 sha256 assertion, Conv list, timing, branch
  decision, full metric blocks), one small per-layer JSON each (optional), and
  `REPORT.md` with the ranked tables, measured numbers only.
- Measured / inferred / unknown are kept in separate labeled sections.
  Never claim a number not measured in this run. Baseline reference numbers
  from `int8_e3.json` are labeled "recorded reference", not results.

## 10 Hard prohibitions (binding)

- Do not modify `phase1/harness/frozen_eval.py`, existing experiment outputs,
  existing bundles (`bundle/`, `bundle_e1..e4`), or checkpoints.
- Do not commit ONNX files > 5 MB. Do not `git push`.
- Do not push any Kaggle kernel or dataset. Do not touch the running Kaggle
  kernel `vishnuvardhanksece/stage-e-e4-pretrain`.
- New work goes in the files listed in §9 plus this spec only.

---

## 11 Amendment 2026-09-23 — QDQ coverage correction and Arm C (pre-registered before any Arm C measurement)

This section is APPENDED on 2026-09-23 after Arms A/B were measured. The
original text above (§1–§10) is not edited. Measured / inferred / unknown are
kept separate. Nothing in this amendment re-claims an Arm A/B number; all Arm
A/B numbers live in `stage_e_recipe/int8_sensitivity/REPORT.md` and
`int8_sensitivity.json`.

### 11.1 False premise in §3 (measured correction)

§3 rationale (line ~49) states Conv nodes are the quantized compute in this
graph under the default `op_types_to_quantize`. That premise is MEASURED
FALSE. Reproduction script `stage_e_recipe/int8_qdq_coverage.py` (read-only
audit of the already-quantized artifacts; no training, no quantization)
records into `stage_e_recipe/int8_sensitivity/qdq_coverage.json`:

- `baseline_int8.onnx`: 200 QuantizeLinear nodes, 286 DequantizeLinear nodes
  (1197 graph nodes total). DequantizeLinear outputs are consumed through 374
  input-slot edges by 201 distinct nodes across 16 op types (edges /
  distinct nodes): Conv 153 / 51, Add 40 / 20, LeakyRelu 40 / 40, Sub 51 /
  27, Concat 26 / 2, Pad 23 / 23, Slice 23 / 23, Mul 6 / 3, ReduceMean 3 / 3,
  Div 2 / 2, Shape 2 / 2, Softmax 1 / 1, ReduceSum 1 / 1, Resize 1 / 1,
  Squeeze 1 / 1, Transpose 1 / 1. Zero dangling DequantizeLinear outputs.
- `only_00_*.onnx` (Arm B config for the first Conv group): still 191
  QuantizeLinear nodes, 193 DequantizeLinear nodes, with the non-Conv
  activation QDQ fully retained (e.g. Add 40/40, LeakyRelu 40/40, Concat
  26/26, Pad 23/23, Slice 23/23 edges — identical to baseline).
- One-edge bookkeeping note (measured, not a gate): the Mul edge count is 6
  here (3 distinct Mul nodes); the review brief quoted 5. All other edge
  counts reproduce the brief exactly.

### 11.2 What Arms A/B therefore do and do not show (inference, stated as one)

- Arm A (leave-one-out) rows keep exactly one Conv in fp32; every other
  Conv AND every quantizable non-Conv node stays int8. Arm A therefore
  measures "how much does restoring one Conv to fp32 recover", NOT "how much
  damage does quantizing one Conv cause".
- Arm B (only-one) rows are mislabelled by the original spec: with
  `nodes_to_exclude=[all Conv except L]`, every non-Conv activation stays
  quantized (191 of 200 QuantizeLinear nodes retained in `only_00`). Arm B
  therefore does NOT isolate single-layer int8; it measures "single Conv
  int8 on top of full non-Conv int8".
- Valid reading of the measured Arms A/B data (inference): excluding or
  isolating any single Conv changes EPE by < 0.06 px of the 4.06 px gap, so
  the damage is not attributable to any single Conv; it must sit in non-Conv
  QDQ and/or be distributed. Arm C below tests the non-Conv half of that
  inference directly.

### 11.3 Arm C — pre-registered op-type follow-up (frozen on commit of this amendment)

Same subject (§1, same sha/n_nodes assertions, STOP on mismatch), same
quantizer and calibration (§2, first 32 `hailo_calib` scenes), same 40-scene
contract scoring with `contract_match` guard (§5; STOP on mismatch — every
Arm C row must be contract numbers, no monitor branch), same
`frozen_eval.py` prohibition. Baselines: reuse the §4 fp32 file and the
already-produced `baseline_int8.onnx` (assert it exists; re-score both on
the contract; require the §6 reproduction gate within 1e-6 px and both
guards true, else STOP). R is computed against the same measured fp32/int8
baselines (`R = (EPE_int8 − EPE_config) / G`); the dominant rule is unchanged
(any config with R >= 0.50 is labelled dominant, diagnostic only).

Configs (each written under `stage_e_recipe/int8_sensitivity/`, scored, then
its `.onnx` DELETED immediately after scoring; sha256 recorded before
deletion):

- C0 "conv-only int8": `nodes_to_exclude` = every named non-Conv node of the
  fp32 graph (only Conv quantizes; all non-Conv fp32).
- C1 "non-conv-only int8": `nodes_to_exclude` = every Conv node (all Conv
  fp32; all quantizable non-Conv int8).
- C-type, one config per op type T: for each op type T that consumes a
  DequantizeLinear output in `baseline_int8.onnx` per `qdq_coverage.json`
  (Add, LeakyRelu, Sub, Concat, Pad, Slice, Mul, ReduceMean, Div, Shape,
  Softmax, ReduceSum, Resize, Squeeze, Transpose — i.e. every measured
  consumer type EXCLUDING Conv),
  `nodes_to_exclude` = all fp32-graph nodes with `op_type == T`
  (T stays fp32, everything else quantizes as in §2).
- Total: 2 + 15 = 17 configs.
- Preconditions (STOP and report if violated): every fp32-graph node has a
  non-empty name (exclude lists are name-based); every listed T has at least
  one node in the fp32 graph.

Budget rule: time the first Arm C config end-to-end (quantize + score wall
seconds S1); estimate `T_est = 17 × S1`. If `T_est > 2 h` (7200 s): STOP,
write the partial JSON, delete any unscored `.onnx`, and report without
further configs.

### 11.4 Arm C readout (pre-registered)

One ranked table (sorted by EPE ascending) with the same columns as Arms
A/B: EPE, Δ vs fp32 baseline, Δ vs int8 baseline, R, D1, nonfinite count,
contract_match. REPORT.md gains a correction section plus this table;
Arms A/B tables are kept. Pre-registered interpretations (to be confirmed or
rejected by measurement): C1 EPE near the int8 baseline implicates non-Conv
QDQ as carrying the gap; C0 EPE near the fp32 baseline says the same from
the other side; a C-type config with R >= 0.50 names a dominant op type
(diagnostic label only, no fix claim, no Hailo implication).

### 11.5 Outputs and prohibitions (extends §9–§10)

- New: `stage_e_recipe/int8_armc.py` (or a sibling; must reuse the §2
  quantizer and §5 scoring functions verbatim in behavior),
  `stage_e_recipe/int8_sensitivity/armc.json` (+ small per-config jsons),
  `qdq_coverage.json` (already written by `int8_qdq_coverage.py`), updated
  `REPORT.md`. No `.onnx` file may remain in `int8_sensitivity/` afterwards.
- All §10 prohibitions still bind (frozen records, bundles, checkpoints,
  no push, no Kaggle contact, no > 5 MB ONNX commits, no environment
  changes).
