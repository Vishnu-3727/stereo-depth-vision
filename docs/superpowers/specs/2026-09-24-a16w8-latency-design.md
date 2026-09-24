# A16W8 CPU latency next to fp32 / R0 / R2 — host ORT (characterization)

Date: 2026-09-24. Status: PRE-REGISTERED before any measurement under this
spec. This is CHARACTERIZATION, not an experiment: no accept/reject
thresholds on latency, no deployment claim. No training. No Kaggle.

## 0 Purpose (frozen)

Measure what the E3 seed-0 A16W8 whole-graph artefact costs on the host
ONNX Runtime CPU EP, next to the three already-timed seed-0 artefacts
(fp32, R0 all-int8, R2 head+refinement-fp32), under the SAME timing
protocol as `stage_e_recipe/mixed_precision_cost.py`, in ONE process so
all four configs share machine state. Descriptive only.

Grounds (recorded references, NOT re-claimed here):

- `stage_e_recipe/int8_e3/int8_e3.json` seed "0": fp32 export sha256
  `7453b2be…9645` (1701550 bytes, opset 13, 712 nodes), EPE
  1.18598534271453; int8 quantize sha256 `8b31f949…68a` (764520 bytes),
  EPE 5.248941413313285; both `contract_match: true`, 3802797 valid
  pixels.
- `stage_e_recipe/mixed_precision_cost/mixed_precision_cost.json`
  (protocol commit `6218e6b`, host run 2026-09-24T06:06:13Z–06:11:19Z):
  R2 regen sha256 `b12d1e0d…644ef` (1050225 bytes), EPE
  1.2908231335718434; host CPU medians fp32 766.75 ms / R0 918.09 ms /
  R2 818.31 ms (10 warm-up + 10 scenes x 5 repeats, `ORT_ENABLE_ALL`,
  CPU EP, `time.perf_counter`). These three numbers are CROSS-SESSION
  CONTEXT, not comparisons: the new run re-times all three configs in
  its own single session and reports its own numbers.
- `stage_e_recipe/int8_a16w8/int8_a16w8.json` (report commit `2d865d1`):
  seed-0 A16 EPE 1.2603345881778643, artifact sha256
  `8dd30bca2d0b7ce272a1635cbfdf6aa500fe897c0b6df46e78ed0a3528465988`
  (764752 bytes), quantized in a guarded child process (0.5 GB guard,
  CALIB_N 32, peak child 5.6 GB). The A16 `.onnx` was deleted after
  scoring, so activation dtype is UNKNOWN-by-construction there; this
  spec fixes that by KEEPING the regenerated graph (§6).

## 1 Subjects (frozen)

E3 seed-0 best checkpoint, as four ONNX artefacts:

| tag | artefact | recorded sha256 | bytes | location |
|---|---|---|---|---|
| fp32 | E3 seed-0 fp32 export | `7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645` | 1701550 | `stage_e_recipe/int8_e3/e3_seed0_best_fp32.onnx` (on disk) |
| R0 | all-int8 QDQ baseline | `8b31f94921878f7f08ae122e00a1e36983fc2fb19dae5af3c07566589598c68a` | 764520 | `stage_e_recipe/int8_e3/e3_seed0_best_int8.onnx` (on disk) |
| R2 | head+refinement fp32, rest int8 | `b12d1e0dd591750d8711ef77d528fe5756ba0f05cfdfe3bd390fc0b6dfb644ef` | 1050225 | `stage_e_recipe/mixed_precision_cost/r2_regen/seed0_R2_head_refinement_fp32.onnx` (on disk, produced by the mixed-precision-cost run) |
| A16 | whole-graph A16W8 QDQ | `8dd30bca2d0b7ce272a1635cbfdf6aa500fe897c0b6df46e78ed0a3528465988` | 764752 | NOT on disk (deleted after scoring per precedent) — regenerate (§2) |

- fp32 and R0 MUST be sha-verified (plus byte size) against the recorded
  values before any measurement. If either differs: STOP, record, no
  measurement.
- R2 MUST be the file the mixed-precision-cost run produced, sha-verified
  against `b12d1e0d…` (plus byte size). If the file is absent from disk,
  regenerate it with the EXACT procedure `mixed_precision_cost.py` used
  (import `int8_sensitivity.quantize_with_excludes` +
  `int8_regression_head.kept_set`, excludes =
  `sorted(set(head) | set(refin))`, CALIB_N 32 — see
  `mixed_precision_cost.py` R2-regen block) into the new output dir and
  sha-verify before use. If the sha differs either way: STOP, do not
  measure a non-identical graph. The script records whether R2 was
  reused or regenerated.
- A16 MUST be regenerated per §2 into the new output dir only
  (`stage_e_recipe/a16w8_latency/a16_regen/`), then scored through the
  §5 gate. Its sha256 is RECORDED as match/mismatch vs `8dd30bca…`
  (§7): a mismatch alone is NOT a stop if the EPE gate passes.

## 2 A16 regeneration procedure (frozen)

Exact `stage_e_recipe/int8_a16w8.py` A16 path, seed 0 only, via import
(no copies, no edits to existing files):

- `from int8_a16w8 import quantize_with_guard` (guarded child-process
  quantization, 0.5 GB free-memory guard).
- `quantize_with_guard(fp32, dst, None, use_a16=True)` with `fp32` the
  §1 sha-asserted subject and `dst` =
  `stage_e_recipe/a16w8_latency/a16_regen/seed0_A16.onnx`.
- Whole graph, no fp32 islands (`exclude=None`); CALIB_N 32 (set inside
  the guarded child, as in `int8_a16w8.py`).
- If the guard trips or the child fails: STOP, record the guard record.
- The regenerated `.onnx` is KEPT (do not delete; do not commit —
  `*.onnx` is git-ignored). It feeds latency (§4), the EPE gate (§5),
  and the dtype inspection (§6).

## 3 Backends and metrics (frozen)

One backend only:

- H-CPU: host python ONNX Runtime, `CPUExecutionProvider` only,
  `ORT_ENABLE_ALL`, inference-only timing (§4). No CUDA check repetition
  beyond recording `ort.get_available_providers()` (H-CUDA was
  unavailable in the host onnxruntime; this run does not depend on it).

Metrics per artefact:

- file size (bytes, `st_size`).
- latency: median ms, p90 ms (nearest-rank), FPS (= 1000/median) per §4,
  plus per-scene medians and raw samples.
- accuracy: full 40-scene frozen-contract EPE (plus D1/RMSE) per §5.
- A16 only: sha256 vs recorded (§7), activation-dtype census + opset
  (§6).

## 4 Latency protocol (frozen)

Identical to `mixed_precision_cost.py` (same session options, same loop,
same statistics):

- Inputs: 10 scenes from the head of `hailo_val` (scenes 0–9),
  `normalize()` preprocessing, 368x1232, batch 1. REUSE the
  bit-identical precomputed inputs under
  `stage_e_recipe/mixed_precision_cost/inputs_npy/` if present (assert
  each is `(1, 3, 368, 1232)` float32 contiguous); otherwise precompute
  once into `stage_e_recipe/a16w8_latency/inputs_npy/` with the same
  procedure and assert the same. The script records which source was
  used.
- Timing scope: `sess.run` ONLY (preprocessing/postprocessing excluded).
- Wall clock: `time.perf_counter()` around each `sess.run`.
- Warm-up: 10 untimed `sess.run` calls on the scene-0 input per artefact,
  then timed runs.
- Timed runs: 10 scenes × 5 repeats → 50 timed samples per artefact,
  looped scene0×5, scene1×5, … (fixed order, recorded).
- Statistics: median and p90 (nearest-rank, same `p90` as
  `mixed_precision_cost.py`) over the 50 samples; FPS = 1000/median_ms.
  Raw samples + per-scene medians recorded in JSON.
- Thread settings: NO explicit values set (session defaults); the script
  records `os.cpu_count()` and the session's `intra_op_num_threads` /
  `inter_op_num_threads` as observed.
- Sessions: `providers=["CPUExecutionProvider"]`,
  `graph_optimization_level=ORT_ENABLE_ALL` (same as the scoring
  precedent).
- ALL FOUR configs are timed in ONE process, sequentially
  (fp32 → R0 → R2 → A16), so they share machine state. Sessions are
  released between configs. No cross-session comparison is made: the
  previous medians (767 / 918 / 818 ms) are context, not baselines.

## 5 Accuracy protocol (frozen)

- The script re-scores all four artefacts on the FULL 40-scene frozen
  contract by importing `S.score_contract` (unchanged behaviour:
  `ORT_ENABLE_ALL`, CPU EP, `pooled_metrics` +
  `refuse_unless_contract`, `contract_match` REQUIRED). Expected cost
  ~35–40 s per artefact.
- Gates (all REQUIRED, tolerance 1e-6 px, else STOP with a `STOP:`
  status in the JSON and a nonzero exit):
  - fp32 EPE equals `int8_e3.json` seed-0 fp32 1.18598534271453;
  - R0 EPE equals `int8_e3.json` seed-0 int8 5.248941413313285;
  - R2 EPE equals `mixed_precision_cost.json` R2 1.2908231335718434;
  - A16 EPE equals `int8_a16w8.json` seed-0 e_A16 1.2603345881778643.
- Any `contract_match == false`: STOP that row, record, no further use
  of the row.

## 6 A16 graph inspection (frozen)

On the REGENERATED (kept) A16 graph, by import of `onnx` (no
re-quantization, read-only):

- opset version of the graph (record as observed; the `int8_a16w8`
  REPORT notes ORT auto-upgraded A16 graphs 13 → 21 — record what this
  graph shows, do not assert).
- activation-dtype census: for every `QuantizeLinear` node, the dtype of
  its zero-point input (initializer element type), counted per dtype
  (e.g. INT16 vs INT8). Record counts per dtype plus the node total.
- This census replaces the UNKNOWN-by-construction dtype note in the
  `int8_a16w8` REPORT for seed 0. No causal claim is attached to it.

## 7 A16 identity rule (frozen)

- Record `sha256(regenerated A16)` and whether it matches recorded
  `8dd30bca2d0b7ce272a1635cbfdf6aa500fe897c0b6df46e78ed0a3528465988`.
- MISMATCH ALONE IS NOT A STOP if the §5 A16 EPE gate passes (the EPE
  gate is the behaviour check; the sha is provenance either way).
- If the EPE gate fails: STOP regardless of the sha outcome, record both
  facts.

## 8 Time boxes and STOP rules (frozen)

- Whole run: no fixed box (expected ~8–12 min: A16 regen ~41 s quantize
  + 4 × ~40 s contract re-scores + 4 × 50 timed runs at ~0.8–0.9 s;
  R2 reuse costs nothing, R2 regen fallback ~40 s).
- Quantization peaks ~5.6 GB RAM (measured child peak in `int8_a16w8`):
  the run MUST execute with no LLM process resident.
- STOP rules: §1 sha/size mismatch (fp32/R0/R2); A16 guard/child failure
  (§2); any §5 EPE gate miss; any `contract_match == false` (§5).
  Every STOP writes a `status: "STOP: <reason>"` JSON and exits nonzero.
- No acceptance threshold on latency: descriptive only.

## 9 Outputs (new files only)

- Spec (new): this file.
- Script (new): `stage_e_recipe/a16w8_latency.py` — subject asserts +
  A16 regen (imported guard) + graph inspection + input reuse/precompute
  + single-process 4-config H-CPU latency + 40-scene CPU re-scores +
  gates. Imports `p90` / `prefix_names` from `mixed_precision_cost`,
  `quantize_with_guard` from `int8_a16w8`, `quantize_with_excludes` /
  `score_contract` / `sha256` / `git_head` from `int8_sensitivity`,
  `kept_set` from `int8_regression_head` (R2-regen fallback only),
  `Kitti2015Stereo` / `normalize` from the dataset module — all by
  import; no copies, no edits to existing files.
- Output dir (new): `stage_e_recipe/a16w8_latency/` containing:
  `a16w8_latency.json` (UTC start/end, git HEAD, environment incl.
  ORT version, `ort.get_available_providers()`, CPU name, thread
  settings, free RAM at start, sha asserts incl. R2 reuse/regen source,
  A16 regen provenance + sha match/mismatch, dtype census + opset,
  per-artefact size/latency/EPE rows with raw samples, context numbers
  labeled as recorded references), `a16_regen/` (regenerated A16
  `.onnx`, KEPT, git-ignored), `inputs_npy/` (only if the fallback
  precompute was needed).
- Measured / inferred / unknown are kept in separate labeled sections
  in the JSON. Never claim a number not measured in this run; cited
  reference numbers are labeled "recorded reference".
- Report: none required by this spec (characterization record is the
  JSON); a later REPORT.md may summarize it without re-measuring.

## 10 Hard prohibitions (binding)

- Do not modify any existing `stage_e_recipe/*.py`, existing run
  outputs, `phase1/`, `src/`, or any spec other than this new file.
- Do not commit `.onnx` files (already git-ignored) or `.npy` files.
- No training. No Kaggle. No environment changes (no installs, upgrades,
  driver changes, PATH edits).
- This spec's commit and the script commit stay LOCAL — do NOT push
  them.
