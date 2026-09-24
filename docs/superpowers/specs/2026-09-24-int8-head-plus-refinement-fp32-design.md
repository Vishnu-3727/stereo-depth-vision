# INT8 head-plus-refinement-fp32 recovery — E3 seed-0/1/2 (pre-registered diagnostic)

Date: 2026-09-24. Status: PRE-REGISTERED before any measurement under this spec.
No accept/reject. Diagnostic only. No training.

## 0 Hypothesis (frozen)

Keeping the refinement stage in fp32 in addition to the regression head
reduces the residual enough to meet the criterion on all 3 E3 best
checkpoints.

Grounds (recorded references, NOT re-claimed here):

- `stage_e_recipe/int8_regression_head/REPORT.md` (commit `bbe37fd`): R1
  (head fp32, 40 `/regression/*` nodes) recovers most of each seed's gap —
  R_R1 0.873 / 0.909 / 0.887 — but the residual ceiling (0.50 px) fails on
  seeds 0 and 2: residuals +0.5162 / +0.3817 / +0.5288 px, 1/3 seeds pass
  the pre-registered R>=0.50 AND residual<=0.50 px criterion. Remaining
  remainder ~0.38–0.53 px, source OPEN.
- `stage_c_deploy/diagnostics/C1_SECTION10_EXPORT_INVESTIGATION.md` §5:
  the export-mismatch FAIL is carried by the residual term (Δresidual
  1.690e-3 ≈ Δfinal 1.709e-3 px; Δinitial 7.14e-4 alone would PASS).
  Correlation only — the residual input-sensitivity vs own-divergence split
  is NOT ISOLATED there, so this motivates but does not establish the
  refinement as the INT8 remainder source.

What counts as "enough" is the frozen inherited threshold (§7), not an
author-set one.

## 1 Subjects (frozen)

- Models: E3 seed-0/1/2 best checkpoints:
  `stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_best.pth`,
  `stage_e_recipe/kaggle/e3_output/seed1/e3_seed1_best.pth`,
  `stage_e_recipe/kaggle/e3_output/seed2/e3_seed2_best.pth`.
- FP32 ONNX: reuse the recorded graphs in `stage_e_recipe/int8_e3/` **only
  because** their provenance is recorded in
  `stage_e_recipe/int8_e3/int8_e3.json` (seed → export):
  - seed 0: sha256
    `7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645`,
    bytes 1701550, opset 13, n_nodes 712, parent_checkpoint_sha256
    `7d3c2109c11339b6fb30cc33f5bd042667740f3bc7345768f4e4b04704836209`,
    params 397954.
  - seed 1: sha256
    `9b4922c91367a2c41254795e0c7160dcb05313f7fb948e2c0d56f05d9ebe9ac2`,
    bytes 1701550, opset 13, n_nodes 712, parent_checkpoint_sha256
    `133c19c41585445da9aabf18919e62f5805b5f8abcf80232808d26d33aa293b0`,
    params 397954.
  - seed 2: sha256
    `7c0c16e4cf66a7ca14e59f5be31801f19815ae1ba6ca72014cb7dd3e99e50418`,
    bytes 1701550, opset 13, n_nodes 712, parent_checkpoint_sha256
    `1ba8ca09bfabf2c689b6ec7fbbbae1aa70367d33562b1aad06521ba71904cbcf`,
    params 397954.
- The script MUST assert at start, per seed: file exists, sha256 equals the
  recorded value, `n_nodes == 712`, params implied by reuse (no re-export, no
  weight change). If any assertion fails: STOP, do not re-export silently —
  re-export only via the existing `export_onnx` path from
  `stage_e_recipe/int8_control.py` and record fresh provenance in the output
  JSON.
- Recorded E3 reference numbers (from `int8_e3.json`, NOT re-claimed here),
  all `contract_match: true`, 3802797 valid pixels:
  - seed 0: fp32 ONNX EPE 1.18598534271453, int8 ONNX EPE 5.248941413313285,
    P = 4.062956070598755 px.
  - seed 1: fp32 ONNX EPE 1.178821279341308, int8 ONNX EPE 5.3849187722115985,
    P = 4.20609749287029 px.
  - seed 2: fp32 ONNX EPE 1.1713166599983933, int8 ONNX EPE 5.87156272118187,
    P = 4.700246061183476 px.
- Recorded R1 reference numbers (from `bbe37fd`'s
  `stage_e_recipe/int8_regression_head/int8_regression_head.json` →
  `measured.rows`, NOT re-claimed here), for the §6 R1 reproduction gate:
  - seed 0: R1 EPE 1.7021716010157655, R_R1 0.8729530299289785,
    resid +0.5161862583012355.
  - seed 1: R1 EPE 1.5605500813958062, R_R1 0.9092439481724895,
    resid +0.3817288020544982.
  - seed 2: R1 EPE 1.7001293943249853, R_R1 0.8874925424237381,
    resid +0.528812734326592.

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

## 3 Node sets (frozen definition)

- Both kept sets are defined by graph position, not by hardcoded lists,
  enumerated at runtime per seed graph, sorted:
  - head set: every node in the fp32 ONNX graph whose name starts with
    `/regression/`
    (`[n.name for n in model.graph.node if n.name.startswith("/regression/")]`);
  - refinement set: every node whose name starts with `/refinement/`
    (`[n.name for n in model.graph.node if n.name.startswith("/refinement/")]`).
- Read-only pre-registration inspection (existing export-code graph, no
  quantization, no scoring): seed-0 fp32 ONNX
  (`7453b2be…`, 712 nodes) decomposes as `/feature_extractor` 68,
  `/cost_volume` 557, `/aggregation` 11, `/regression` 40,
  `/refinement` 33, plus 1 no-slash Constant, `/Add` 1, `/Relu` 1
  (68+557+11+40+33+1+1+1 = 712). The `/refinement/*` prefix (33 nodes:
  1 Concat, 14 Conv, 12 LeakyRelu, 6 Add — `/refinement/Concat`,
  `/refinement/input_conv/Conv`, 6 blocks × (conv1, act, conv2, Add,
  act_1), `/refinement/output_conv/Conv`) is the refinement stage; the
  `/regression/*` prefix (40 nodes, op types 11 Constant / 3 Shape /
  3 ReduceMean / 3 Sub / 3 Mul / 2 Gather / 2 Cast / 2 Div / 1 Slice /
  1 Concat / 1 Resize / 1 ReduceProd / 1 Sqrt / 1 Add / 1 Neg /
  1 Softmax / 1 Range / 1 Reshape / 1 ReduceSum) is the head. Seeds 1 and 2
  show the identical counts (33 refinement, 40 regression, 712 total).
  The full per-seed lists are recorded at runtime; if the sets differ
  across seeds, record all three and proceed per seed (set membership is a
  measured property of each graph, not a gate).
- The script MUST assert the enumerated head set contains
  `/regression/Mul`, `/regression/Mul_1` and `/regression/Mul_2`, and the
  enumerated refinement set contains `/refinement/input_conv/Conv` and
  `/refinement/output_conv/Conv`; if any is missing: STOP and report.
- No-op precondition (same as the precedent, applied per kept set): before
  scoring, the script MUST verify against the seed's freshly quantized
  all-int8 baseline that every kept set used below (head set for R1,
  head+refinement union for R2, refinement set for R3) consumes at least
  one DequantizeLinear output (or is otherwise altered by quantization).
  If a kept set carries no QDQ in the baseline, that arm cannot move the
  score: STOP, write the coverage audit, do not rank.

## 4 Arms (frozen)

Per seed (3 seeds), produce by passing through to `quantize_static`:

- Arm R0 — all-int8 baseline: `nodes_to_exclude=None`, exact §2 procedure
  (freshly quantized, then reproduction-gated per §6).
- Arm R1 — head fp32, rest int8: `nodes_to_exclude=[all /regression/*
  names from §3]`, all other quantizer arguments identical to §2.
  Must reproduce `bbe37fd` within 1e-6 (§6).
- Arm R2 — head + refinement fp32, rest int8 (PRIMARY):
  `nodes_to_exclude=[all /regression/* names + all /refinement/* names
  from §3]`, all other quantizer arguments identical to §2.
- Arm R3 — refinement-only fp32, rest int8 (ATTRIBUTION, no verdict):
  `nodes_to_exclude=[all /refinement/* names from §3]`, all other
  quantizer arguments identical to §2.
- Plus the recorded-reference row: all-fp32 (the reused §1 ONNX, scored
  directly, no quantization).
- Total configs: 3 seeds × 4 quantized + 3 fp32 re-scores = 15 scorings.
- Each quantized artifact is written under a new sibling output dir only
  (§10), never beside existing bundles. Its `.onnx` is DELETED immediately
  after scoring; sha256 recorded before deletion (same rule as the
  precedent).

## 5 Scoring (frozen)

- Scoring code reuses the existing functions verbatim in behavior:
  `score_onnx_model` + `pooled_metrics` + `refuse_unless_contract` from
  `stage_e_recipe/int8_control.py` / `phase1/harness/frozen_eval.py`.
  `phase1/harness/frozen_eval.py` MUST NOT be modified.
- Full-contract scoring: all 40 `hailo_val` scenes, `disparity_scale=256.0`,
  `occluded=True`, `normalize()` inputs, ORT CPU with
  `ORT_ENABLE_ALL`, `contract_match` REQUIRED (raise / STOP on mismatch).
  No monitor branch — every row must be a contract number.
- Record per config: EPE, RMSE, D1, BAD1–3, valid pixels, guard block,
  nonfinite pixel count, score seconds, quantized artifact sha256 (pre-
  deletion), kept-set node lists, arm label, seed. Scored EPE is the ranking
  metric.

## 6 Baseline reproduction gate (STOP rule, frozen)

- Before ranking any arm row, the script re-scores reference points per
  seed on the 40-scene contract and checks:
  - `|EPE_fp32_measured − <seed fp32 reference from §1>| <= 1e-6` px, and
  - `|EPE_int8_measured (Arm R0) − <seed int8 reference from §1>| <= 1e-6` px,
    and
  - `|EPE_R1_measured − <seed R1 reference from §1 (bbe37fd)>| <= 1e-6` px, and
  - all guards `contract_match == true`.
- If ANY check fails on ANY seed: STOP. Write what was measured, do not
  proceed to arm ranking, do not reinterpret. (Tolerance rationale: same
  machine, same deterministic CPU procedure — bit-near reproducibility
  expected; 1e-6 absorbs fp32 reduction-order noise only.)

## 7 Pre-registered readout (frozen; thresholds inherited)

- Per-seed EPE gap G_seed = EPE(R0, measured) − EPE(fp32, measured).
- Per-seed recovered fractions:
  `R_R2,seed = (EPE_R0 − EPE_R2) / G_seed`;
  `R_R3,seed = (EPE_R0 − EPE_R3) / G_seed` (attribution only);
  `R_R1,seed = (EPE_R0 − EPE_R1) / G_seed` (reproduction row).
- Residuals: `resid_R2,seed = EPE_R2 − EPE_fp32`;
  `resid_R1,seed = EPE_R1 − EPE_fp32`.
- Report one table (3 rows, one per seed) with EPE (fp32 / R0 / R1 / R2 /
  R3), R_R2, R_R1, residuals for R1 and R2, D1, nonfinite count,
  contract_match, plus the kept-set node lists.
- Pre-registered decision thresholds for R2 (inherited from 6a6cf6b,
  frozen — the SAME frozen thresholds the user set for the previous
  experiment, carried over unchanged; the script MUST read them from a
  config block at the top of its output JSON and apply them mechanically,
  never choose them):
  - Recovery threshold `R_ACCEPT`: 0.50 (per-seed recovered fraction R_R2 must be >= 0.50).
  - Cross-seed consistency rule `N_SEEDS_REQUIRED`: 3 (must hold on all 3 E3 seeds).
  - Residual ceiling `RESID_MAX_PX`: 0.50 (EPE_R2 - EPE_fp32 must be <= 0.50 px on each of those seeds).
- Thresholds inherited from 6a6cf6b, frozen.
- Pre-registered interpretations (to be confirmed or rejected by measurement,
  conditional on the inherited thresholds):
  - R_R2 at/above threshold on the required seeds: SUPPORTED — keeping
    head+refinement in fp32 meets the criterion (diagnostic label only, no
    fix claim, no Hailo implication).
  - Otherwise: FALSIFIED — head+refinement fp32 does not meet the criterion
    as pre-registered.
- Pre-registered readouts with NO verdict attached (descriptive only):
  - delta residual R1→R2 per seed (`EPE_R2 − EPE_R1`, and
    `resid_R2 − resid_R1`);
  - R3 recovered fraction per seed (`R_R3,seed`);
  - whether the R2 residual is below the R1 residual on all seeds.
- No accept/reject of any model, no gate, no deployment claim. NOT Hailo
  evidence (same disclaimer as `int8_control.py`: Stage D remains blocked).

## 8 Falsification (frozen: what would kill the hypothesis)

The hypothesis (§0) is FALSIFIED if, with all §6 gates passing and the §3
no-op preconditions satisfied:

- (a) R_R2 falls below the inherited `R_ACCEPT` (or breaches the residual
  ceiling) on any of the three seeds — head+refinement in fp32 does not
  meet the criterion on all 3 seeds; or
- (b) R_R2 meets the criterion but the R1→R2 delta residual is ~0 on all
  seeds — the refinement adds nothing and the hypothesis that the
  refinement contributes materially is falsified (criterion met by the
  head alone); or
- (c) R_R3 alone meets the criterion at/above the R2 level — the head adds
  nothing beyond the refinement, qualifying (not overturning) the joint
  claim; recorded as an attribution observation, the PRIMARY R2 verdict
  still applies mechanically.

Any falsification is reported as FALSIFIED with the measured table, not
reinterpreted. Mechanism claims ("why the refinement matters") are out of
scope regardless of outcome — mechanism unknown.

## 9 Budget rule (frozen before starting)

- The script MUST time the first config end-to-end (quantize + score wall
  seconds S1); estimate `T_est = 12 × S1` (3 seeds × 4 quantized configs;
  fp32 re-scores are score-only and bounded by the same S1).
- If `T_est > 2 h` (7200 s): STOP, write the partial JSON, delete any
  unscored `.onnx`, and report without further configs.
- S1, T_est and the chosen branch are recorded in the output JSON.

## 10 Outputs (new files only)

- Script: `stage_e_recipe/int8_head_refinement.py` (new file; must reuse the
  §2 quantizer and §5 scoring functions verbatim in behavior via import
  from `int8_regression_head.py` / `int8_sensitivity.py` — no copies
  unless unavoidable, and say which).
- Output dir: `stage_e_recipe/int8_head_refinement/` (new dir) containing:
  `int8_head_refinement.json` (per-seed rows + provenance: UTC, git HEAD,
  ORT / torch / numpy versions, per-seed fp32 sha256 assertions, kept-set
  lists, §3 coverage audits, inherited threshold block with
  `R_ACCEPT` 0.50 / `N_SEEDS_REQUIRED` 3 / `RESID_MAX_PX` 0.50 stated as
  "inherited from 6a6cf6b, frozen", R1-reproduction gate rows, timing,
  full metric blocks), one small per-config JSON each (optional), and
  `REPORT.md` with the measured table only.
- Thresholds read from a config block at the top of the output JSON,
  applied mechanically.
- Measured / inferred / unknown are kept in separate labeled sections.
  Never claim a number not measured in this run. REPORT.md Arm C and R1
  numbers cited in §0–§1 are labeled "recorded reference", not results.

## 11 Hard prohibitions (binding)

- Do not modify `phase1/harness/frozen_eval.py`, `stage_e_recipe/verdict.py`,
  `stage_e_recipe/int8_control.py`, `stage_e_recipe/int8_sensitivity.py`,
  `stage_e_recipe/int8_regression_head.py`,
  existing experiment outputs, existing bundles (`bundle/`,
  `bundle_e1..e4`), checkpoints, or `e3_*` files.
- Do not commit ONNX files > 5 MB. Do not `git push`.
- Do not push any Kaggle kernel or dataset. Do not touch Kaggle at all —
  in particular the RUNNING E4 seed-2 kernel
  (`vishnuvardhanksece/stage-e-e4-finetune-seed2`).
- No training. No environment changes.
- New work goes in the files listed in §10 plus this spec only.
