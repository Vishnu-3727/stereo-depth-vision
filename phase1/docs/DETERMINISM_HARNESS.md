# PHASE 1, STEP 0, TASK 1 — DETERMINISM / EVALUATION HARNESS

Status: COMPLETE. All three verification checks measured live 2026-09-14;
results below are actual harness output, not expectations.

Scope: two identical runs must be reproducible enough that model comparisons are
meaningful. Infrastructure only — no training, no architecture change, no new
loss/dataset/variant. Phase 0 contract (`phase0/docs/BASELINE_CONTRACT.md`) is
binding. Frozen numbers: reference ONNX EPE 1.3134471 px / D1 8.1543664% /
3,802,797 px; `convergence_run.pth` EPE 15.3958267 / D1 88.3922807
(40 scenes KITTI 2015 hailo_val, disp_occ_0, GT 1/256, pooled).

## 1. What already existed and is reused (paths)

- Determinism controls: `phase2/diagnostics/determinism/stageb_deterministic_baseline.py:132-143`
  (`enable_determinism`: cudnn.deterministic=True, cudnn.benchmark=False,
  `torch.use_deterministic_algorithms(True)`, asserts `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  set at `stageb_deterministic_baseline.py:42-44` before CUDA context).
- RNG/data-order diagnosis + same-process forward/backward repeat probes:
  `phase2/diagnostics/determinism/diag_determinism.py:245-302`.
- Non-deterministic op locator (warn_only + per-stage + isolated bilinear-backward probe):
  `phase2/diagnostics/determinism/diag_locate_op.py:78-152`.
- Run-vs-run comparator: `phase2/diagnostics/determinism/compare_runs.py`.
- Frozen-contract evaluator pieces: `src/datasets/kitti2015.py` (slices, crop,
  scales), `src/evaluation/metrics.py:52-84` (`disparity_metrics`, pooled EPE/D1/RMSE/BAD1-3),
  reference driver `scripts/exp_reproduce_hailo.py`, convergence recipe
  `scripts/exp_train_convergence.py:124-162`, provenance/config capture
  `src/common/experiment.py:65-129,195-218`, checkpoint hashing + strict-load
  pattern `phase2/viz/core.py:201-255`.
- The harness below thinly wraps these; new code exists only for genuine gaps
  (single entry points, contract guard, seed helper).

## 2. What was added and why

- `phase1/harness/determinism.py` — one `seed_all(seed)` + `enable_determinism()`
  entry point (re-exports Stage B settings), plus `worker_init_fn` /
  `make_generator(seed)` for DataLoader determinism. Why: repo had the settings
  scattered across diagnostics; no single importable helper.
- `phase1/harness/frozen_eval.py` — frozen-contract constants + `refuse_unless_contract`
  protocol-mixing guard + `score_onnx()` / `score_checkpoint()` + `sha256_file` /
  `strict_load_report` / `record_provenance`. Why: repo had the pieces but no
  single evaluator that refuses cross-protocol comparison.
- `phase1/harness/rescore.py` — CLI running checks 1–2 through `frozen_eval`.
- `phase1/harness/repro_probe.py` — CLI running check 3 (two identical seeded
  forward passes, bitwise comparison).

## 3. Exact settings

- Seeds: `random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)`,
  `torch.cuda.manual_seed_all(seed)` — see `phase1/harness/determinism.py`.
- Deterministic PyTorch: `torch.use_deterministic_algorithms(True)`,
  `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`,
  env `CUBLAS_WORKSPACE_CONFIG=:4096:8` (must be set before CUDA context;
  harness sets it via os.environ if absent and asserts it).
- Non-deterministic op: `F.interpolate(mode="bilinear")` CUDA backward (atomicAdd
  scatter) in the regression upsampler — identified in
  `phase2/diagnostics/determinism/diag_locate_op.py:117-152`
  (`regression.py:56-61`). See §6.
- DataLoader: `num_workers=0` (matches `scripts/exp_train_convergence.py:131`);
  seeded `generator=torch.Generator().manual_seed(seed)`, `shuffle` via generator,
  `worker_init_fn` seeds python/numpy per worker. Deterministic augmentation:
  `CroppedKitti`-style crops draw from an explicit `np.random.default_rng(seed)`.
- Checkpoint hashing: sha256 of file bytes (`sha256_file`), plus weight-sha
  (sha256 of concatenated fp32 parameter bytes, first 16 hex — Stage A method).
- Config capture: full config dict saved with every checkpoint
  (`torch.save({"model":..., "config":...})` per `scripts/exp_train_convergence.py:248`);
  harness re-records it in its output.
- Provenance: git HEAD + subject + status, seed, recipe, dataset/split, torch/numpy/
  cuda/cudnn versions, flags, env — via `record_provenance()` wrapping
  `src/common/experiment.py:git_state/software_versions/hardware`.
- Evaluator contract: KITTI2015 / hailo_val 40 scenes / disp_occ_0 / 1/256 /
  368x1232 top-left crop / gt>0 / pooled; emits EPE, D1, RMSE, BAD1, BAD2, BAD3.
- Guard: `refuse_unless_contract` records scene set, pixel count, GT scale and
  REFUSES comparison against the reference unless scenes==40 AND
  pixels==3,802,797 AND scale==256.0 AND split=="hailo_val" AND gt=="disp_occ_0".
- Compat check: `strict=True` load; harness reports matched/missing/unexpected
  keys (expect 72/72, 423,586 params).
- Checkpoint identity: file sha256 + weight-sha + config-hash + git commit jointly.

## 4. Verification results (real numbers, measured 2026-09-14)

1. Reference ONNX re-score through harness
   (`python phase1\harness\rescore.py --target onnx`, CPUExecutionProvider,
   onnxruntime 1.27.0; full record `phase1/harness/rescore_onnx.json`):
   EPE 1.3134470770188373 / D1 8.154366378221082 / RMSE 2.5829573145561677 /
   BAD1 38.89637022433751 / BAD2 16.079375259841637 / BAD3 8.639640769675584 /
   3,802,797 px, guard `contract_match: true`. Rounded: EPE 1.3134471 /
   D1 8.1543664 — exact match to the binding reference (every digit) and to
   `experiments/EXP-005/metrics.json:62-80`. Checkpoint sha256
   `b1a01d855bb22663f06dfd29eae11194e6fde952a319673e036f157c3e10095c`
   (matches `reference/manifest.json:25`). Git HEAD 58e8a19 at time of run.
2. `convergence_run.pth` re-score through harness
   (`python phase1\harness\rescore.py --target convergence`, CUDA device;
   full record `phase1/harness/rescore_convergence.json`):
   EPE 15.395826671257934 / D1 88.39228073441733 / RMSE 20.002283814769715 /
   BAD1 96.10744407340177 / BAD2 92.25535835859763 / BAD3 88.39228073441733 /
   3,802,797 px, guard `contract_match: true`. Rounded: EPE 15.3958267 /
   D1 88.3922807 — exact match to the binding PyTorch baseline (every digit).
   Strict-load compat: matched 72 / missing [] / unexpected [] / strict_ok true /
   423,586 params. File sha256
   `19f3d6df84d1d7af12a0a656296f8456d6f59d744efe3f54b498e17d67923673`,
   weight-sha16 `5e53f1307044a5d8` (Stage A method).
3. Reproducibility bound (`python phase1\harness\repro_probe.py`; full record
   `phase1/harness/repro_probe.json`): two identical seeded (seed 0) CPU
   forward passes of `StereoNet(StereoNetConfig())` on hailo_val scene 0
   (`000160_10.png`) with determinism controls on — bitwise_identical True,
   max_abs_diff 0.000e+00, out_mean 5.893548.

## 5. Reproducibility bound

Bitwise identical (max_abs_diff 0.0) for repeated forward inference on CPU
with fixed seed + determinism controls (check 3). Bound claimed: inference
repeats are bitwise reproducible; training-step repeats are NOT claimed
bitwise on CUDA because of the bilinear-backward op (§6) — smallest practical
bound for training is the Stage B two-process comparison (final weight-sha +
loss-series equality per `stageb_deterministic_baseline.py:370-436`), which
this task did not re-run (no training allowed).

## 6. Known non-determinism sources left unresolved

- `F.interpolate(mode="bilinear", align_corners=True)` CUDA backward in
  `src/models/stereonet/regression.py:56-61` — atomicAdd scatter, no
  deterministic implementation; confirmed by
  `phase2/diagnostics/determinism/diag_locate_op.py:117-152`. Forward-only
  eval (no backward) is unaffected; training runs cannot be bitwise
  deterministic on CUDA while this op is in the graph. CPU forward repeats are
  bitwise identical (see check 3).
- cuDNN benchmark autotuning (must stay False), TF32 matmul precision drift
  (harness records `allow_tf32` flags; NOT forced — see NOT VERIFIED),
  `PYTHONHASHSEED` process hash order (recorded, not pinned), wall-clock/latency
  nondeterminism (never compared).

## 7. NOT VERIFIED (with reason)

- Multi-process / multi-GPU determinism: NOT VERIFIED (single-GPU/CPU only;
  no training permitted in this task, so Stage B A-vs-B rerun was out of scope).
- TF32 pinned off: NOT VERIFIED as a forced setting (recorded only; forcing it
  is a training-recipe change, out of scope).
- DataLoader multi-worker determinism: NOT VERIFIED live (helpers provided and
  match the num_workers=0 frozen recipe; multi-worker path untested).
- HEF/int8 artifacts: NOT VERIFIED (no device; out of scope, per Phase 0).
- Upstream exact training recipe for the reference export: NOT VERIFIED
  (no training log held; per Phase 0).
