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
