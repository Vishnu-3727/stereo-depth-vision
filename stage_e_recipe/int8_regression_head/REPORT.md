# INT8 regression-head fp32 — E3 seed-0/1/2 (measured)

Diagnostic only. No accept/reject. No training. NOT Hailo evidence;
Stage D remains blocked.

- Thresholds (user-set, pre-registered before any R1 measurement;
  `int8_regression_head.json` → `thresholds`): `R_ACCEPT` 0.50,
  `N_SEEDS_REQUIRED` 3, `RESID_MAX_PX` 0.50 px.
- Reuse method: import from `stage_e_recipe/int8_sensitivity.py`
  (quantizer, calibration, scoring used unmodified — see
  `int8_regression_head.json` → `reuse_method`).
- Budget: first config (seed-0 R0 quantize + fp32/R0 scores) S1 120.2 s × 6
  → est 0.20 h < 2 h → full run proceeded (9 scorings: 3 fp32 + 3 R0 + 3 R1).
- Baseline gate (§6): PASS on all three seeds — measured fp32/R0 EPE equal
  the `int8_e3.json` references to ≤ 1e-6 px, all guards
  `contract_match=true` (per-seed rows below, full blocks in
  `int8_regression_head.json` → `baseline_gate_per_seed`).
- No-op precondition (§3): PASS — 17/40 kept `/regression/*` nodes consume a
  DequantizeLinear output in each seed's freshly quantized R0 baseline, so
  the arm is not a no-op (per-seed `coverage_audit_seed<N>` in the JSON).
- Kept set: 40 `/regression/*` nodes, identical on all 3 seeds, containing
  `/regression/Mul`, `/regression/Mul_1`, `/regression/Mul_2`
  (full lists in `int8_regression_head.json` →
  `measured.kept_set_per_seed`).
- Each quantized `.onnx` deleted immediately after scoring (sha256 recorded
  before deletion). No `.onnx` remains in this directory.

## Baseline reproduction gate (measured, per seed)

| seed | EPE fp32 measured | ref | EPE R0 measured | ref | guards |
|---|---|---|---|---|---|
| 0 | 1.1859853 | 1.1859853 | 5.2489414 | 5.2489414 | fp32 true, R0 true |
| 1 | 1.1788213 | 1.1788213 | 5.3849188 | 5.3849188 | fp32 true, R0 true |
| 2 | 1.1713167 | 1.1713167 | 5.8715627 | 5.8715627 | fp32 true, R0 true |

Exact measured values (`int8_regression_head.json`): seed 0 fp32
1.1859853427 / R0 5.2489414133 (G 4.0629560706); seed 1 fp32 1.1788212793 /
R0 5.3849187722 (G 4.2060974929); seed 2 fp32 1.1713166600 / R0
5.8715627212 (G 4.7002460612). Fresh R0 artifact shas reproduce the recorded
`int8_e3.json` quantize shas (`8b31f949…`, `3abd266a…`, `07b28a15…`).

## Measured readout table (3 rows, one per seed)

| seed | EPE R1 | Δ vs fp32 | Δ vs R0 | R_R1 | D1 R1 | nonfinite | contract | pass_R | pass_resid | pass_seed |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.7022 | +0.5162 | -3.5468 | 0.873 | 9.90 | 0 | true | true | false | false |
| 1 | 1.5606 | +0.3817 | -3.8244 | 0.909 | 8.57 | 0 | true | true | true | true |
| 2 | 1.7001 | +0.5288 | -4.1714 | 0.887 | 9.54 | 0 | true | true | false | false |

Exact measured values: seed 0 R1 1.7021716010, R 0.872953, resid
+0.5161862583; seed 1 R1 1.5605500814, R 0.909244, resid +0.3817288021;
seed 2 R1 1.7001293943, R 0.887493, resid +0.5288127343. R1 D1: 9.89819 /
8.57177 / 9.53879 (fp32 D1 6.13972 / 6.11311 / 6.13046; R0 D1 67.79405 /
70.17979 / 76.20806). Nonfinite pixels 0 on all 9 scorings. R1 artifact shas:
seed 0 `9ec37011…`, seed 1 `629b5120…`, seed 2 `82556a23…` (40 nodes
excluded each). Full metric blocks in `int8_regression_head.json` →
`measured.rows` (+ `seed<N>_R0.json`, `seed<N>_R1.json`).

## Pre-registered verdict (mechanical)

Thresholds applied mechanically from the JSON config block (`R_R1 >= 0.50`
AND `EPE_R1 − EPE_fp32 <= 0.50 px` per seed; required on all 3 seeds):

**FALSIFIED: below threshold on seeds [0, 2] (1/3 pass, required 3);
hypothesis as stated (general readout-localized gap) falsified.**

Seed detail: R passes on all three seeds (0.873 / 0.909 / 0.887 ≥ 0.50) but
the residual ceiling fails on seeds 0 (+0.5162 px, margin 0.0162 px over
ceiling) and 2 (+0.5288 px, margin 0.0288 px over ceiling); only seed 1
passes both (+0.3817 px). This is the §8 falsification branch for
below-threshold seeds (neither the all-three §8(a) pattern nor the
seed-0-only §8(b) pattern — 1/3 pass with residual breach on the other two).
No reinterpretation. Mechanism unknown.

## Inference (labelled, not measured)

- R_R1 0.87–0.91 recovers most of each seed's gap — near the recorded
  C0-level reference (R=0.924 at EPE 1.4963, seed-0 recorded reference, not a
  result of this run) and above the recorded C_Mul-level reference (R=0.572
  at EPE 2.9241) — but the pre-registered residual ceiling still fails on
  2/3 seeds, so the narrowed readout-localized claim as pre-registered fails
  while high recovery is descriptively observed.
- §8(c) (C0-level comparison with a reproduced all-non-Conv-fp32 control)
  was not tested in this run — no C0 arm was scored here (recorded as
  unknown, not as evidence either way).

## Unknown (not measured in this run)

- Mechanism behind any readout difference (out of scope, mechanism unknown).
- Whether results transfer to other procedures or hardware.
- §8(c) C0-level comparison (no C0 arm in this run).

## Provenance

- UTC: 2026-09-24T05:15:03.848658+00:00; git HEAD:
  6a6cf6bfad96d9b194915a8978e432645fa803e8.
- Env (measured, same machine): python 3.12.9, torch 2.7.0+cu128, numpy
  2.5.1, onnxruntime 1.27.0.
- Subjects asserted per seed: sha256 `7453b2be…` / `9b4922c9…` /
  `7c0c16e4…`, 712 nodes, params 397954 (reuse; recorded in
  `stage_e_recipe/int8_e3/int8_e3.json`).
- Full rows: `int8_regression_head.json` (+ `seed<N>_R0.json`,
  `seed<N>_R1.json`). Kept-set lists: JSON → `measured.kept_set_per_seed`.
  Coverage audits: JSON → `measured.coverage_audit_seed<N>`.
- Recorded Arm C numbers cited in the spec (§0) are recorded references,
  not results of this run.

## Claim status (added 2026-09-24, after the verdict)

- FALSIFIED: fp32 regression-head retention as a complete INT8 solution (1/3 seeds pass the pre-registered R>=0.50 AND residual<=0.50 px criterion).
- SUPPORTED: the regression head accounts for most of the observed INT8 degradation (R_R1 0.873 / 0.909 / 0.887 on all three seeds).
- OPEN: the source of the remaining ~0.38-0.53 px residual.
- HYPOTHESIS: the refinement stage contributes materially to that remainder (motivated by the C1 export investigation localizing the export mismatch to the refinement residual; correlation, not a causal result).
