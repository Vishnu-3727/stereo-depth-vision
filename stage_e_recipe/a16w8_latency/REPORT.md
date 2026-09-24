# A16W8 CPU latency next to fp32 / R0 / R2 — host ORT (report)

Spec: `docs/superpowers/specs/2026-09-24-a16w8-latency-design.md`
(characterization only: no thresholds, no verdict).
Subjects: E3 seed-0 as fp32 / R0 all-int8 QDQ / R2 head+refinement-fp32 /
A16 whole-graph A16W8 QDQ (regenerated, kept under `a16_regen/`, git-ignored).

## 1 Results table (artefact x backend)

Latency: 10 warm-up + 10 scenes (head of `hailo_val`, scenes 0-9) x 5 repeats
= 50 timed `sess.run` samples, 368x1232 batch 1, `time.perf_counter`,
`ORT_ENABLE_ALL`, default thread settings (cpu_count 24, intra/inter-op 0 =
default). All four configs timed sequentially in ONE process
(fp32 -> R0 -> R2 -> A16), so they share machine state. Sessions released
between configs. Timing scope is `sess.run` only (preprocessing excluded,
inputs precomputed .npy, reused from
`stage_e_recipe/mixed_precision_cost/inputs_npy`). EPE: full 40-scene frozen
contract, `contract_match` true everywhere below.

| artefact | size (bytes) | backend | median ms | p90 ms | FPS | ratio to fp32 median | EPE (px) | EPE ref (px) | abs diff vs ref | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| fp32 | 1701550 | host CPU EP | 817.9047999999511 | 839.0150000000176 | 1.2226361796630363 | 1.000000 | 1.18598534271453 | 1.18598534271453 (`int8_e3.json` seed 0 fp32) | 0.0 | pass |
| R0 all-int8 | 764520 | host CPU EP | 981.9808499998999 | 995.1914999996916 | 1.01834979775838 | 1.200605 | 5.248941413313285 | 5.248941413313285 (`int8_e3.json` seed 0 int8) | 0.0 | pass |
| R2 head+refin fp32 | 1050225 | host CPU EP | 870.315150000124 | 890.480000000025 | 1.1490090687262624 | 1.064079 | 1.2908231335718434 | 1.2908231335718434 (`mixed_precision_cost.json` R2) | 0.0 | pass |
| A16 whole-graph A16W8 | 764752 | host CPU EP | 1178.9456500000597 | 1188.1917000000612 | 0.8482155220640997 | 1.441422 | 1.2603345881778643 | 1.2603345881778643 (`int8_a16w8.json` seed 0 e_A16) | 0.0 | pass |

EPE gate block (`measured.epe_gates`): pass true, tolerance 1e-6 px.
All four rows `contract_match` true, 3802797 valid pixels, 0 nonfinite pixels
each. Per-row D1 / RMSE / score_s (40-scene CPU re-score, this run):

| artefact | D1 | RMSE | score_s |
|---|---|---|---|
| fp32 | 6.139717686744783 | 3.075254588401288 | 36.5 |
| R0 | 67.79404738144056 | 7.266354245453155 | 43.3 |
| R2 | 6.712506610266075 | 3.4843294741682245 | 39.5 |
| A16 | 6.744377888170207 | 3.2924205926925056 | 51.2 |

A16W8 is the SLOWEST config on the ORT CPU EP here: A16 median
1178.9456500000597 ms > R0 981.9808499998999 ms > R2 870.315150000124 ms >
fp32 817.9047999999511 ms (same-process ordering, +196.9648000001598 ms vs
R0, +308.63049999993564 ms vs R2, ratio 1.441422 to fp32). It wins accuracy
among quantized configs (best quantized EPE: A16 1.2603345881778643 px vs R2
1.2908231335718434 px vs R0 5.248941413313285 px; A16 resid above fp32 only
+0.07434924546333432 px) but costs latency on this backend. Descriptive
only; no threshold, no verdict.

Cross-session numbers from `mixed_precision_cost.json` (fp32 766.75 ms /
R0 918.09 ms / R2 818.31 ms) are CONTEXT ONLY, recorded in this run's JSON
under `context` as recorded references, NOT comparisons. The same-process
ordering above is the comparison.

## 2 A16 regen + graph inspection (measured)

Regen (seed 0 only, guarded child via `quantize_with_guard`, whole graph,
`exclude=None`, CALIB_N 32, procedure string `quantize_static, QDQ, QInt8
act, QInt8 weight, per_channel` — that `QInt8 act` label is inherited from
`int8_sensitivity`'s fixed label and is WRONG for this run, see below):

- Output: `stage_e_recipe/a16w8_latency/a16_regen/seed0_A16.onnx` (KEPT,
  git-ignored — never commit; the study deleted its A16 graphs after scoring,
  this run keeps the regen for inspection).
- sha256 `8dd30bca2d0b7ce272a1635cbfdf6aa500fe897c0b6df46e78ed0a3528465988`,
  764752 bytes, parent fp32 sha `7453b2be…` match; recorded study sha
  `8dd30bca2d0b7ce272a1635cbfdf6aa500fe897c0b6df46e78ed0a3528465988` —
  sha_match true (byte-identical to the study's seed-0 A16).
- quantize_s 42.6; child peak mem 5.604854583740234 GB; min free
  3.138721466064453 GB; peak_seen true.
- R2 source: reused
  `stage_e_recipe/mixed_precision_cost/r2_regen/seed0_R2_head_refinement_fp32.onnx`
  (not regenerated). fp32/R0 subjects sha-verified before measurement
  (`7453b2be…` / `8b31f949…`, match true, sizes 1701550 / 764520).

Graph inspection (read-only `onnx` import on the kept regen):

- opset 21 (ORT auto-upgrade from opset 13, same as the study's Fault (b)).
- QuantizeLinear nodes: 200.
- zero-point dtype counts: INT16 200 (all QuantizeLinear nodes carry INT16
  zero-points; zero INT8 zero-points).

This resolves Fault (a) of `stage_e_recipe/int8_a16w8/REPORT.md`: the
byte-identical regenerated A16 (sha match) has INT16 zero-points on ALL
QuantizeLinear nodes, so the study's activations were 16-bit — measured now,
no longer UNKNOWN. The study's JSON `procedure` label `QInt8 act` remains
wrong.

## 3 Measured vs inferred vs unknown

MEASURED (this run): all table cells in §1; EPE gate pass (tol 1e-6, abs
diffs 0.0 on all four rows); fp32/R0 sha asserts (match true); R2 reuse
source; A16 regen sha match true + quantize_s/peak-child/min-free numbers
above; A16 graph census (opset 21, 200 QuantizeLinear, 200x INT16
zero-point); environment/threads/provenance below; session providers
`['CPUExecutionProvider']` and inputs `['left', 'right']` on all four rows;
per-scene medians + raw samples in the JSON (not repeated here).

INFERRED (labelled, not measured): why ORT CPU runs 16-bit QDQ slower is
INFERENCE/HYPOTHESIS ONLY — e.g. fallback of int16 QDQ to fp32 kernels with
extra Q/DQ ops. Mechanism untested; no causal claim is made.

UNKNOWN / OPEN: repeatability of every latency number (single run each);
any transfer beyond this machine/installation; NPU / edge-device latency;
whether an int16-capable accelerator reverses the ordering.

HYPOTHESIS (untested, not a finding): an int16-capable accelerator may
reverse the CPU ordering; the ORT-CPU slowdown mechanism above is
hypothesis only.

## 4 Claim status

- MEASURED: §1 latency/accuracy/size cells; EPE gates; A16 byte-identity
  (sha match) + 16-bit activation census; slowest-on-ORT-CPU ordering;
  best-quantized-EPE readout (A16 1.2603345881778643 < R2 1.2908231335718434
  < R0 5.248941413313285).
- OPEN: everything listed under UNKNOWN above.
- HYPOTHESIS: the two untested statements in §3.

## 5 Provenance

- Protocol commit `ebff709` (spec `2026-09-24-a16w8-latency-design.md`);
  script commit `9763aa4`; run git HEAD `9763aa420102b31fd97058cf9d1bef0b3cbfdec9`.
- Run UTC 2026-09-24T11:59:03.497154+00:00 – 2026-09-24T12:06:37.482411+00:00
  (launched 11:58:59Z, COMPLETE: A16 regen + 4-config CPU latency + CPU
  re-scores).
- Env (measured, same machine): python 3.12.9, torch 2.7.0+cu128, numpy
  2.5.1, onnxruntime 1.27.0; available providers
  `['AzureExecutionProvider', 'CPUExecutionProvider']`; cpu_count 24,
  platform_processor `AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD`,
  machine `AMD64`; free RAM at start 7.845 GB; thread settings defaults
  (intra_op 0, inter_op 0) on all four rows; graph optimization
  `ORT_ENABLE_ALL`.
- Records: `a16w8_latency.json` (authoritative), `run.log`,
  `a16_regen/seed0_A16.onnx` (kept, git-ignored, never commit).
- Reuse by import only: `p90` from `mixed_precision_cost`,
  `quantize_with_guard` from `int8_a16w8`, quantize/score/sha/git_head from
  `int8_sensitivity`, dataset loader/normalize from `Kitti2015Stereo` — no
  existing file modified; no script, JSON, protocol, or `run.log` edited for
  this report.
- No training, no Kaggle, no environment changes. Only this report (plus the
  `int8_a16w8/REPORT.md` addendum) is new prose; `.onnx`/`a16_regen/` stay
  uncommitted.
