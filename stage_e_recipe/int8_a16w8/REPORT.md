# A16W8 whole-graph — E3 seed-0/1/2 (measured)

Diagnostic only. No accept/reject. No training. NOT Hailo evidence;
Stage D remains blocked.

- Thresholds (inherited from 6a6cf6b, frozen — the SAME frozen thresholds
  the user set for the previous experiment, carried over unchanged;
  `int8_a16w8.json` → `thresholds`): `R_ACCEPT` 0.50,
  `N_SEEDS_REQUIRED` 3, `RESID_MAX_PX` 0.50 px.
- Quantizer: `quantize_static`, QDQ, per-channel, first 32 scenes of
  `hailo_calib` (CALIB_N 32, restored after the amendment round-trip —
  see Timeline); intended `activation_type=QInt16`, `weight_type=QInt8`
  (§2 of the spec). The `procedure` string recorded in
  `int8_a16w8.json` → `A16_quantize` reads `QInt8 act, QInt8 weight` —
  that label is inherited from `int8_sensitivity`'s fixed label and is
  WRONG for this run (see Fault (a)). Whole graph quantized, no fp32
  islands (`nodes_to_exclude: []`).
- Budget: first config (seed-0 fp32/R0) S1 127.56552648544312 s × 9
  → est 1148.089738368988 s < 2 h → full run proceeded (9 scorings:
  3 fp32 + 3 R0 + 3 A16).
- Baseline gate (§5): PASS on all three seeds — measured fp32/R0 EPE equal
  the `int8_e3.json` references to ≤ 1e-6 px, all guards
  `contract_match=true` (per-seed rows below, full blocks in
  `int8_a16w8.json` → `measured.baseline_gate_per_seed`).
- Each quantized `.onnx` deleted immediately after scoring (sha256 recorded
  before deletion). No `.onnx` remains in this directory.

## Baseline reproduction gate (measured, per seed)

| seed | EPE fp32 measured | ref | EPE R0 measured | ref | guards |
|---|---|---|---|---|---|
| 0 | 1.1859853 | 1.1859853 | 5.2489414 | 5.2489414 | fp32 true, R0 true, guards true |
| 1 | 1.1788213 | 1.1788213 | 5.3849188 | 5.3849188 | fp32 true, R0 true, guards true |
| 2 | 1.1713167 | 1.1713167 | 5.8715627 | 5.8715627 | fp32 true, R0 true, guards true |

Exact measured values (`int8_a16w8.json`): seed 0 fp32
1.18598534271453 / R0 5.248941413313285 (G 4.062956070598755);
seed 1 fp32 1.178821279341308 / R0 5.3849187722115985 (G
4.20609749287029); seed 2 fp32 1.1713166599983933 / R0
5.87156272118187 (G 4.700246061183476). Diffs vs references are 0.0 at
1e-6 px on all six checks; all 9 guards `contract_match=true`.

## Measured arms table (per seed)

| seed | EPE fp32 | EPE R0 | EPE A16 | R_A16 | resid A16 | EPE R2 (ref) | resid R2 (ref) | Δ resid A16−R2 | D1 A16 | nonfinite | contract | pass_A16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.1860 | 5.2489 | 1.2603 | 0.982 | +0.0743 | 1.2908 | +0.1048 | -0.0305 | 6.74 | 0 | true | true |
| 1 | 1.1788 | 5.3849 | 1.2678 | 0.979 | +0.0890 | 1.3016 | +0.1228 | -0.0338 | 6.94 | 0 | true | true |
| 2 | 1.1713 | 5.8716 | 1.2458 | 0.984 | +0.0745 | 1.3349 | +0.1636 | -0.0891 | 6.91 | 0 | true | true |

Exact measured values: seed 0 A16 1.2603345881778643, R 0.9817007016144339,
resid +0.07434924546333432; R2 (recorded reference from
`int8_head_refinement.json` → `measured.rows`) EPE 1.2908231335718434, R
0.9741966713310136, resid +0.10483779085731348; Δ resid A16−R2
-0.03048854539397916; seed 1 A16 1.2678008248981807, R 0.9788451062516499,
resid +0.08897954555687271; R2 EPE 1.3016331313575487, R
0.9708014728083652, resid +0.12281185201624067; Δ resid
-0.03383230645936797; seed 2 A16 1.245819769957704, R 0.9841491043257103,
resid +0.07450310995931075; R2 EPE 1.3349449471216213, R
0.9651872933899066, resid +0.16362828712322797; Δ resid
-0.08912517716391721. A16 D1: 6.744377888170207 / 6.943205224996234 /
6.908467635795443 (fp32 D1 6.13972 / 6.11311 / 6.13046; R0 D1 67.79405 /
70.17979 / 76.20806 — fp32/R0 D1 recorded references from
`int8_head_refinement.json` → `measured.rows`). A16 RMSE: 3.2924205926925056
/ 3.3930599930057297 / 3.215440492498058. Nonfinite pixels 0 on all 9
scorings. Artifact shas: A16 `8dd30bca…` / `8cf509e9…` / `56416e4d…`
(no nodes excluded each). A16 residual is below R2 residual on all seeds
(true / true / true). Full metric blocks in `int8_a16w8.json` →
`measured.rows`.

## Sizes, timing, memory (measured, per seed)

| seed | fp32 bytes | R0 bytes | A16 bytes | quantize s (A16) | peak child GB | min free GB |
|---|---|---|---|---|---|---|
| 0 | 1701550 | 764520 | 764752 | 40.9 | 5.6057281494140625 | 2.9447555541992188 |
| 1 | 1701550 | 764529 | 764758 | 40.7 | 5.6041412353515625 | 2.935638427734375 |
| 2 | 1701550 | 764538 | 764768 | 40.4 | 5.604438781738281 | 2.8898544311523438 |

## Pre-registered verdict (mechanical)

Thresholds applied mechanically from the JSON config block (inherited from
6a6cf6b, frozen: `R_A16 >= 0.50` AND `EPE_A16 − EPE_fp32 <= 0.50 px` per
seed; required on all 3 seeds):

**SUPPORTED: R_A16 >= 0.5 and residual <= 0.5px on 3/3 seeds (required 3).**

Seed detail: R passes on all three seeds (0.982 / 0.979 / 0.984 ≥ 0.50)
and the residual ceiling passes on all three (+0.0743 / +0.0890 /
+0.0745 px ≤ 0.50 px). No reinterpretation. Mechanism unknown.

## Faults (verified against the run record, stated plainly)

(a) The A16 `procedure` string in `int8_a16w8.json` says `QInt8 act` —
it is inherited from `int8_sensitivity`'s fixed label and does NOT
describe this run. Evidence the run was 16-bit: `run.log` shows the ORT
"does not support 16-bit integer quantization natively" warning exactly
on the three A16 steps (once per seed, after each `--- seed <N> A16 ---`
marker) and never on any fp32/R0 step, and A16 EPE differs from R0 by
~4 px per seed (3.9886 / 4.1171 / 4.6257 px — an 8-bit re-run would
reproduce R0 to 1e-6). Direct dtype inspection of the A16 graphs is
IMPOSSIBLE because the script deletes each A16 `.onnx` after scoring
(only sha256 kept) — activation dtype is therefore UNKNOWN-by-construction
beyond the log + EPE evidence above.
(b) ORT auto-upgraded the A16 graphs from opset 13 to opset 21 (three
"Automatically update the model to opset 21" warnings in `run.log`, one
per A16 step); the spec does not mention this.
(c) Comparison to Quark A16W8 (`stage_e_recipe/npu_quark/REPORT.md`, Q2
CPU EPE 1.368 on seed 0): different quantizer and calibration settings
(Quark MinMax, 8 calibration scenes vs ORT MinMax, 32 scenes here), so
descriptive only — no claim that one procedure beats the other.

## Measured (this run only)

- Every number in the tables above; guard readings; quantize wall
  seconds; artifact shas and sizes; S1/T_est budget numbers.
- 40-scene contract: `hailo_val`, 3802797 valid pixels every scored row.
- A16 residual below the recorded R2 residual on all 3 seeds
  (Δ -0.0305 / -0.0338 / -0.0891 px) with the whole graph quantized —
  descriptive readout per spec §7, no verdict attached.

## Inferred (labelled, not measured)

- A16W8 whole-graph meets the pre-registered criterion with smaller
  residuals than the R2 fp32-island arm under THESE two procedures — but
  the procedures differ (16-bit activations vs fp32 islands), so this is
  a descriptive ranking, not a mechanism claim, and bears no Hailo
  implication.
- The §8 falsification branches (R below threshold / residual breach on
  any seed) do not occur.

## Unknown (not measured in this run)

- Activation dtype of the deleted A16 graphs by direct inspection
  (UNKNOWN-by-construction — see Fault (a)).
- Mechanism behind the A16 recovery (out of scope, mechanism unknown).
- Whether results transfer to other procedures or hardware.
- Per-op placement, NPU behaviour, latency — none measured here; no NPU
  or latency claims are made.

## Claim status

- SUPPORTED: A16W8 whole graph meets the pre-registered criterion on all
  3 seeds (R_A16 0.982 / 0.979 / 0.984; resid +0.0743 / +0.0890 /
  +0.0745 px).
- OPEN: the mechanism of the recovery; the source of the remaining
  ~0.07–0.09 px A16 residual.
- HYPOTHESIS: none — no causal result claimed here.

## Timeline (measured record)

- Spec pre-registered: `fc3747a` (2026-09-24 14:09:30 +0530 = 08:39:30Z),
  before any measurement under this spec.
- Amendments (committed before the measurements they govern): CALIB_N
  32→16 `fbf8858`, 16→8 `58e9ec9`, revert to 32 after the parent-GC fix
  `0c697a4` (2026-09-24 14:54:01 +0530), guard 2.0→0.5 GB `988939b`
  (2026-09-24 14:55:31 +0530).
- Erratum (R2 reference numbers corrected): `a0c7be4` (2026-09-24
  16:56:46 +0530). Guard fix (real child peak memory): `b1e9506`
  (2026-09-24 16:57:49 +0530).
- STOP attempt 1: `int8_a16w8_stop_20260924T092425Z.json`
  (2026-09-24T09:24:25Z) — 2.0 GB guard tripped during R0 at 1.78 GB free
  with no child load recorded (`peak_child_mem: 0.0`).
- STOP attempt 2: `int8_a16w8_stop_20260924T112831Z.json`
  (2026-09-24T11:28:31Z, `run_stop_20260924T112831Z.log`) — 0.5 GB guard
  tripped during R0, child peak 4.99 GB, free 0.0005 GB, while an LLM
  worker process was also resident.
- Completing run: first-measurement UTC 2026-09-24T11:36:16.881689+00:00
  (`int8_a16w8.json` → `provenance.utc`), git HEAD `b1e9506` at run.

## Provenance

- UTC: 2026-09-24T11:36:16.881689+00:00; git HEAD at run:
  b1e950668b0745743fd3399965edaf092d6e810f.
- Pre-registration commit: fc3747a (2026-09-24 14:09:30 +0530); run start
  (JSON UTC) 2026-09-24T11:36:16Z — spec + amendments + erratum + guard
  fix all committed BEFORE any measurement of the completing run.
- Env (measured, same machine): python 3.12.9, torch 2.7.0+cu128, numpy
  2.5.1, onnxruntime 1.27.0.
- Subjects asserted per seed: sha256 `7453b2be…` / `9b4922c9…` /
  `7c0c16e4…`, 712 nodes, params 397954 (reuse; recorded in
  `stage_e_recipe/int8_e3/int8_e3.json`); fp32/R0 references from
  `int8_e3.json`.
- Full rows: `int8_a16w8.json` → `measured.rows`. R2 numbers cited are
  recorded references from `int8_head_refinement.json` → `measured.rows`,
  not results of this run.

## Addendum (2026-09-24)

Fault (a) resolved by `stage_e_recipe/a16w8_latency/` (protocol
`docs/superpowers/specs/2026-09-24-a16w8-latency-design.md`, record
`a16w8_latency.json`, report `stage_e_recipe/a16w8_latency/REPORT.md`):
the regenerated seed-0 A16 is byte-identical to this study's seed-0 A16
(sha256 `8dd30bca…` match true) and its kept graph shows opset 21 with
INT16 zero-points on all 200 QuantizeLinear nodes — so this study's
activations were 16-bit (measured, no longer UNKNOWN). The `QInt8 act`
procedure label stays wrong. Same run: A16W8 is the slowest config on ORT
CPU EP (median 1178.9456500000597 ms vs fp32 817.9047999999511 / R2
870.315150000124 / R0 981.9808499998999 ms, one process) while keeping the
best quantized EPE (1.2603345881778643 px) — descriptive only.
