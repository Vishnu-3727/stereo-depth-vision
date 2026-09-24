# INT8 head+refinement fp32 — E3 seed-0/1/2 (measured)

Diagnostic only. No accept/reject. No training. NOT Hailo evidence;
Stage D remains blocked.

- Thresholds (inherited from 6a6cf6b, frozen — the SAME frozen thresholds
  the user set for the previous experiment, carried over unchanged;
  `int8_head_refinement.json` → `thresholds`): `R_ACCEPT` 0.50,
  `N_SEEDS_REQUIRED` 3, `RESID_MAX_PX` 0.50 px.
- Reuse method: import from `stage_e_recipe/int8_sensitivity.py`
  (quantizer, calibration, scoring, sha/git — unmodified) and import from
  `stage_e_recipe/int8_regression_head.py` (kept-set enumeration +
  coverage audit — unmodified); see `int8_head_refinement.json` →
  `reuse_method`.
- Budget: first config (seed-0 R0 quantize + fp32/R0 scores) S1 117.7 s × 12
  → est 0.39 h < 2 h → full run proceeded (15 scorings: 3 fp32 + 3 R0 +
  3 R1 + 3 R2 + 3 R3).
- Baseline gate (§6): PASS on all three seeds — measured fp32/R0 EPE equal
  the `int8_e3.json` references to ≤ 1e-6 px, measured R1 EPE equals the
  `bbe37fd` R1 references to ≤ 1e-6 px, all guards `contract_match=true`
  (per-seed rows below, full blocks in `int8_head_refinement.json` →
  `baseline_gate_per_seed`). R1 artifact shas reproduce `bbe37fd`
  (`9ec37011…`, `629b5120…`, `82556a23…`).
- No-op precondition (§3): PASS — head 17/40, refinement 33/33, combined
  50/73 kept nodes consume a DequantizeLinear output in each seed's
  freshly quantized R0 baseline, so no arm is a no-op (per-seed
  `coverage_audit_seed<N>_<head|refinement|combined>` in the JSON).
- Kept sets: 40 `/regression/*` nodes, identical on all 3 seeds,
  containing `/regression/Mul`, `/regression/Mul_1`, `/regression/Mul_2`;
  33 `/refinement/*` nodes, identical on all 3 seeds (1 Concat, 14 Conv,
  12 LeakyRelu, 6 Add), containing `/refinement/input_conv/Conv` and
  `/refinement/output_conv/Conv` (full lists in
  `int8_head_refinement.json` → `measured.head_set_per_seed`,
  `measured.refinement_set_per_seed`). R2 excludes 73 nodes, R1 excludes
  40, R3 excludes 33 per seed.
- Each quantized `.onnx` deleted immediately after scoring (sha256 recorded
  before deletion). No `.onnx` remains in this directory.

## Baseline + R1 reproduction gate (measured, per seed)

| seed | EPE fp32 measured | ref | EPE R0 measured | ref | EPE R1 measured | ref (bbe37fd) | guards |
|---|---|---|---|---|---|---|---|
| 0 | 1.1859853 | 1.1859853 | 5.2489414 | 5.2489414 | 1.7021716 | 1.7021716 | fp32 true, R0 true, R1 true, R2 true, R3 true |
| 1 | 1.1788213 | 1.1788213 | 5.3849188 | 5.3849188 | 1.5605501 | 1.5605501 | fp32 true, R0 true, R1 true, R2 true, R3 true |
| 2 | 1.1713167 | 1.1713167 | 5.8715627 | 5.8715627 | 1.7001294 | 1.7001294 | fp32 true, R0 true, R1 true, R2 true, R3 true |

Exact measured values (`int8_head_refinement.json`): seed 0 fp32
1.1859853427 / R0 5.2489414133 (G 4.0629560706) / R1 1.7021716010;
seed 1 fp32 1.1788212793 / R0 5.3849187722 (G 4.2060974929) / R1
1.5605500814; seed 2 fp32 1.1713166600 / R0 5.8715627212 (G 4.7002460612)
 / R1 1.7001293943. Diffs vs references are 0.0 at 1e-6 px on all nine
checks; all 15 guards `contract_match=true`.

## Measured arms table (per seed)

| seed | EPE fp32 | EPE R0 | EPE R1 | EPE R2 | EPE R3 | R_R1 | resid R1 | R_R2 | resid R2 | R_R3 | Δ R1→R2 | D1 R2 | nonfinite | contract | pass_R2 | pass_resid | pass_seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.1860 | 5.2489 | 1.7022 | 1.2908 | 5.2798 | 0.873 | +0.5162 | 0.974 | +0.1048 | -0.008 | -0.4113 | 6.71 | 0 | true | true | true | true |
| 1 | 1.1788 | 5.3849 | 1.5606 | 1.3016 | 5.3011 | 0.909 | +0.3817 | 0.971 | +0.1228 | 0.020 | -0.2589 | 6.80 | 0 | true | true | true | true |
| 2 | 1.1713 | 5.8716 | 1.7001 | 1.3349 | 5.8399 | 0.887 | +0.5288 | 0.965 | +0.1636 | 0.007 | -0.3652 | 7.08 | 0 | true | true | true | true |

Exact measured values: seed 0 R2 1.2908231336, R 0.974197, resid
+0.1048377909; R3 5.2797850460, R -0.007591; Δ R1→R2 -0.4113484674;
seed 1 R2 1.3016331314, R 0.970801, resid +0.1228118520; R3 5.3011155465,
R 0.019924; Δ R1→R2 -0.2589169500; seed 2 R2 1.3349449471, R 0.965187,
resid +0.1636282871; R3 5.8398835897, R 0.006740; Δ R1→R2 -0.3651844472.
R2 D1: 6.71251 / 6.79992 / 7.07540 (fp32 D1 6.13972 / 6.11311 / 6.13046;
R0 D1 67.79405 / 70.17979 / 76.20806; R1 D1 9.89819 / 8.57177 / 9.53879;
R3 D1 68.82776 / 72.21406 / 76.30294). Nonfinite pixels 0 on all 15
scorings. Artifact shas: R1 `9ec37011…` / `629b5120…` / `82556a23…` (40
nodes excluded each, reproducing `bbe37fd`); R2 `b12d1e0d…` /
`9aabbe3e…` / `553ff410…` (73 nodes excluded each); R3 `cc3ab78c…` /
`b8f66707…` / `8978acc2…` (33 nodes excluded each). R2 residual is below
R1 on all seeds (true / true / true). Full metric blocks in
`int8_head_refinement.json` → `measured.rows` (+ `seed<N>_R2.json`,
`seed<N>_arms.json`).

## Pre-registered verdict (mechanical)

Thresholds applied mechanically from the JSON config block (inherited from
6a6cf6b, frozen: `R_R2 >= 0.50` AND `EPE_R2 − EPE_fp32 <= 0.50 px` per
seed; required on all 3 seeds):

**SUPPORTED: R_R2 >= 0.5 and residual <= 0.5px on 3/3 seeds (required 3).**

Seed detail: R passes on all three seeds (0.974 / 0.971 / 0.965 ≥ 0.50)
and the residual ceiling passes on all three (+0.1048 / +0.1228 /
+0.1636 px ≤ 0.50 px). R3 (attribution, no verdict) recovers ~nothing
(R_R3 -0.008 / 0.020 / 0.007). No reinterpretation. Mechanism unknown.

## Claim status

- SUPPORTED: head+refinement fp32 meets the pre-registered criterion on all 3 seeds (R_R2 0.974 / 0.971 / 0.965; resid +0.1048 / +0.1228 / +0.1636 px).
- FALSIFIED: refinement-only fp32 as a standalone recovery of the INT8 gap (R_R3 -0.008 / 0.020 / 0.007 — no recovery; the joint R2 effect is not refinement-alone).
- OPEN: the source of the remaining ~0.10–0.16 px R2 residual.
- HYPOTHESIS: the remainder sits outside the head+refinement union (feature extractor / cost volume / aggregation or quantizer procedure itself) — untested, no causal result claimed here.

## Inference (labelled, not measured)

- R2 recovers nearly all of each seed's gap (R 0.965–0.974) and cuts the
  R1 residual by 0.26–0.41 px per seed — descriptively the refinement
  carries the bulk of the R1 remainder under THIS quantizer — but this is
  a descriptive ranking, not a fix claim, and bears no Hailo implication.
- R3 ≈ 0 on all seeds shows the refinement alone moves nothing without
  the head; the R2 effect is a joint (head+refinement) effect, not
  refinement-alone. §8(b) (R2 meets but delta ~0) does not occur; §8(c)
  (R3 meets at R2 level) does not occur.
- The C1 export-investigation correlation (export mismatch carried by the
  refinement residual) is consistent with but not established by this
  result — correlation, not a causal result.

## Unknown (not measured in this run)

- Mechanism behind any head/refinement difference (out of scope, mechanism unknown).
- Whether results transfer to other procedures or hardware.
- Source of the remaining 0.10–0.16 px R2 residual.

## Provenance

- UTC: 2026-09-24T05:35:37.155372+00:00; git HEAD at run:
  da64722b29420b3d18d3e780676d61d3b4cfd464.
- Pre-registration commit: da64722 (2026-09-24 11:04:33 +0530); run start
  (JSON UTC) 2026-09-24T05:35:37Z — spec committed BEFORE any measurement
  under this spec.
- Env (measured, same machine): python 3.12.9, torch 2.7.0+cu128, numpy
  2.5.1, onnxruntime 1.27.0.
- Subjects asserted per seed: sha256 `7453b2be…` / `9b4922c9…` /
  `7c0c16e4…`, 712 nodes, params 397954 (reuse; recorded in
  `stage_e_recipe/int8_e3/int8_e3.json`); R1 references from `bbe37fd`.
- Full rows: `int8_head_refinement.json` (+ `seed<N>_R2.json`,
  `seed<N>_arms.json`). Kept-set lists: JSON →
  `measured.head_set_per_seed`, `measured.refinement_set_per_seed`.
  Coverage audits: JSON → `measured.coverage_audit_seed<N>_<head|refinement|combined>`.
- Recorded R1 / Arm C numbers cited in the spec (§0–§1) are recorded
  references, not results of this run.
