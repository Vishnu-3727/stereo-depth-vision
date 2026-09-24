# INT8 regression-head-fp32 recovery — E3 seed-0/1/2 (pre-registered diagnostic)

Date: 2026-09-24. Status: PRE-REGISTERED before any measurement under this spec.
No accept/reject. Diagnostic only. No training.

## 0 Hypothesis (frozen)

The Stage-E INT8 EPE gap sits in the disparity-regression readout: keeping the
readout (`/regression/Mul`, `/regression/Mul_1`, `/regression/Mul_2` and the
other ops feeding the regression head) in fp32 while the rest of the graph is
INT8 recovers a substantial fraction of the per-seed int8→fp32 EPE gap on each
of the three E3 best checkpoints.

Grounds (recorded reference, NOT re-claimed here — all from
`stage_e_recipe/int8_sensitivity/REPORT.md`, Arm C table + readout):

- `C0_conv_only_int8` (only Conv int8, all non-Conv fp32): EPE 1.4963,
  d_fp32 +0.3104, d_int8 −3.7526, R 0.924, D1 7.66, nonfinite 0 —
  "keeping all non-Conv in fp32 recovers 92% of the gap" (REPORT.md,
  "Pre-registered readout — Arm C").
- `C_Mul` (Mul fp32, 3 Mul nodes kept fp32): EPE 2.9241, d_fp32 +1.7381,
  d_int8 −2.3249, R 0.572, D1 38.91, nonfinite 0 — "the only dominant op
  type" (REPORT.md, Arm C table rank 2 + readout).
- `C1_nonconv_only_int8` (all Conv fp32, non-Conv int8): EPE 5.2435 ≈ int8
  baseline 5.2489, R 0.001 — "non-Conv QDQ alone reproduces the full gap"
  (REPORT.md, readout).
- Baselines re-scored for Arm C: fp32 EPE 1.1859853, int8 EPE 5.2489414,
  gap G 4.0629561 px (REPORT.md, "Arm C — op-type follow-up").
- Location (REPORT.md, "Review corrections", §2): the 3 Mul nodes are
  `/regression/Mul`, `/regression/Mul_1`, `/regression/Mul_2`, and the single
  Softmax is `/regression/Softmax` — all four sit in the disparity-regression
  readout, not in the feature extractor or the cost volume / aggregation.
  `baseline_int8.onnx` carries 200 QuantizeLinear nodes with Mul consuming 6
  DequantizeLinear input-slot edges across the 3 distinct Mul nodes
  (REPORT.md, CORRECTION section).

This spec narrows the measured broad claim (damage is non-Conv, R=0.924 for
all-non-Conv-fp32) to the readout subset: if the gap is readout damage, one
`/regression/*`-fp32 arm should recover most of what C0 recovers. What counts
as "most" is a user-set threshold (§7), not an author-set one.

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

## 3 Readout node set (frozen definition)

- The fp32-kept set is defined by graph position, not by a hardcoded list:
  every node in the fp32 ONNX graph whose name starts with `/regression/`
  (`[n.name for n in model.graph.node if n.name.startswith("/regression/")]`),
  enumerated at runtime per seed graph, sorted. The script MUST assert the
  enumerated set contains `/regression/Mul`, `/regression/Mul_1` and
  `/regression/Mul_2` (the REPORT.md-identified readout Mul nodes); if any is
  missing: STOP and report. The full per-seed list is recorded; if the sets
  differ across seeds, record all three and proceed per seed (set membership
  is a measured property of each graph, not a gate).
- No-op precondition (measured lesson from REPORT.md "Review corrections"
  §1: six Arm C configs were byte-identical no-ops because the excluded op
  types carried no QDQ): before scoring, the script MUST verify against the
  seed's freshly quantized all-int8 baseline that every `/regression/*` node
  in the kept set consumes at least one DequantizeLinear output (or is
  otherwise altered by quantization). If the kept set carries no QDQ in the
  baseline, the arm cannot move the score: STOP, write the coverage audit, do
  not rank.

## 4 Arms (frozen)

Per seed (3 seeds), produce by passing through to `quantize_static`:

- Arm R0 — all-int8 baseline: `nodes_to_exclude=None`, exact §2 procedure
  (freshly quantized, then reproduction-gated per §6).
- Arm R1 — readout fp32, rest int8: `nodes_to_exclude=[all /regression/*
  names from §3]`, all other quantizer arguments identical to §2.
- Plus the recorded-reference row: all-fp32 (the reused §1 ONNX, scored
  directly, no quantization).
- Total configs: 3 seeds × 2 quantized + 3 fp32 re-scores = 9 scorings.
- Each quantized artifact is written under a new sibling output dir only
  (§9), never beside existing bundles. Its `.onnx` is DELETED immediately
  after scoring; sha256 recorded before deletion (same rule as Arm C).

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
  deletion), kept-set node list, arm label, seed. Scored EPE is the ranking
  metric.

## 6 Baseline reproduction gate (STOP rule, frozen)

- Before ranking any arm row, the script re-scores both reference points per
  seed on the 40-scene contract and checks:
  - `|EPE_fp32_measured − <seed fp32 reference from §1>| <= 1e-6` px, and
  - `|EPE_int8_measured (Arm R0) − <seed int8 reference from §1>| <= 1e-6` px,
    and
  - both guards `contract_match == true`.
- If ANY check fails on ANY seed: STOP. Write what was measured, do not
  proceed to arm ranking, do not reinterpret. (Tolerance rationale: same
  machine, same deterministic CPU procedure — bit-near reproducibility
  expected; 1e-6 absorbs fp32 reduction-order noise only.)

## 7 Pre-registered readout (frozen; thresholds set by user)

- Per-seed EPE gap G_seed = EPE(R0, measured) − EPE(fp32, measured).
- Per-seed recovered fraction for the readout arm:
  `R_R1,seed = (EPE_R0 − EPE_R1) / G_seed`.
- Report one table (3 rows, one per seed) with EPE, Δ vs fp32, Δ vs R0,
  R_R1, D1, nonfinite count, contract_match, plus the kept-set node lists.
- Pre-registered decision thresholds (set by the user on 2026-09-24 before any
  R1 measurement; the script MUST read them from a config block at the
  top of its output JSON and apply them mechanically, never choose them):
  - Recovery threshold `R_ACCEPT`: 0.50 (per-seed recovered fraction R_R1 must be >= 0.50).
  - Cross-seed consistency rule `N_SEEDS_REQUIRED`: 3 (must hold on all 3 E3 seeds).
  - Residual ceiling `RESID_MAX_PX`: 0.50 (EPE_R1 - EPE_fp32 must be <= 0.50 px on each of those seeds).
- Thresholds set by the user on 2026-09-24 before any R1 measurement.
- Pre-registered interpretations (to be confirmed or rejected by measurement,
  conditional on the user-set thresholds):
  - R_R1 at/above threshold on the required seeds: the INT8 gap is
    readout-localized (diagnostic label only, no fix claim, no Hailo
    implication).
  - R_R1 near C_Mul-level (seed-0 recorded R=0.572 at EPE 2.9241) but far
    below C0-level (recorded R=0.924 at EPE 1.4963): the Mul nodes carry part
    of the gap and the remainder sits in other non-Conv nodes — the narrowed
    readout claim is weakened but the broader non-Conv finding stands.
  - R_R1 near 0 on all seeds: §8 falsification applies.
- No accept/reject of any model, no gate, no deployment claim. NOT Hailo
  evidence (same disclaimer as `int8_control.py`: Stage D remains blocked).

## 8 Falsification (frozen: what would kill the hypothesis)

The hypothesis (§0) is FALSIFIED if, with all §6 gates passing and the §3
no-op precondition satisfied:

- (a) R_R1 falls below the user-set `R_ACCEPT` on all three seeds — the
  readout in fp32 does not recover the gap, so the gap is not readout
  damage; or
- (b) R_R1 is at/above threshold on seed 0 only (reproducing the recorded
  C_Mul direction) but below threshold on seeds 1 and 2 — the effect is
  seed-specific and the hypothesis as stated (a general readout-localized
  gap) is falsified; or
- (c) R_R1 stays far below the recorded C0 recovery (R=0.924) while C0-level
  recovery is reproduced by an all-non-Conv-fp32 control — the damage is
  non-Conv but OUTSIDE the readout, falsifying the narrowed claim while
  leaving the broader REPORT.md non-Conv finding intact.

Any of (a)–(c) is reported as FALSIFIED with the measured table, not
reinterpreted. Mechanism claims ("why the readout collapses") are out of
scope regardless of outcome — mechanism unknown, same as REPORT.md.

## 9 Budget rule (frozen before starting)

- The script MUST time the first config end-to-end (quantize + score wall
  seconds S1); estimate `T_est = 6 × S1` (3 seeds × 2 quantized configs;
  fp32 re-scores are score-only and bounded by the same S1).
- If `T_est > 2 h` (7200 s): STOP, write the partial JSON, delete any
  unscored `.onnx`, and report without further configs.
- S1, T_est and the chosen branch are recorded in the output JSON.

## 10 Outputs (new files only)

- Script: `stage_e_recipe/int8_regression_head.py` (new file; must reuse the
  §2 quantizer and §5 scoring functions verbatim in behavior).
- Output dir: `stage_e_recipe/int8_regression_head/` (new dir) containing:
  `int8_regression_head.json` (per-seed rows + provenance: UTC, git HEAD,
  ORT / torch / numpy versions, per-seed fp32 sha256 assertions, kept-set
  lists, §3 coverage audit, user-set threshold block, timing, full metric
  blocks), one small per-config JSON each (optional), and `REPORT.md` with
  the measured table only.
- Measured / inferred / unknown are kept in separate labeled sections.
  Never claim a number not measured in this run. REPORT.md Arm C numbers
  cited in §0 are labeled "recorded reference", not results.

## 11 Hard prohibitions (binding)

- Do not modify `phase1/harness/frozen_eval.py`, `stage_e_recipe/verdict.py`,
  `stage_e_recipe/int8_control.py`, existing experiment outputs, existing
  bundles (`bundle/`, `bundle_e1..e4`), checkpoints, or `e3_*` files.
- Do not commit ONNX files > 5 MB. Do not `git push`.
- Do not push any Kaggle kernel or dataset. Do not touch Kaggle at all —
  in particular the RUNNING E4 seed-2 kernel
  (`vishnuvardhanksece/stage-e-e4-finetune-seed2`).
- No training. No environment changes.
- New work goes in the files listed in §10 plus this spec only.
