# FORENSIC MODEL INSPECTION — the existing StereoNet/Hailo model

**Type:** inspection/audit only. No training, retraining, evaluation, inference, checkpoint loading, export regeneration, or source modification was performed for this report.
**Repo:** `C:\Users\vishn\stereo_depth_vision`
**Experimental ceiling (binding):** LEVEL D — genuine geometric correspondence NOT-DEMONSTRATED. This report makes no correspondence claim; it establishes what model/pipeline exists.
**Honesty markers used literally:** `NOT VERIFIED`, `CONTRADICTION - REQUIRES RESOLUTION`, `PREVIOUS CLAIM NOT SUPPORTED BY CURRENT EVIDENCE`.

---

## 1. Scope and method (short)

Read-only inspection of code, configs, logs, metrics JSON, docs, and file metadata (sizes, paths, git log). No checkpoint/ONNX/HEF/weight file was opened or loaded; no script that loads a model was run. Every claim cites a path and, where possible, a line number or quoted line. Where a question can only be answered by execution, the answer is `NOT VERIFIED` with the reason stated.

---

## 2. REPOSITORY INVENTORY

Top-level layout (verified by directory listing): `.git/`, `src/`, `scripts/`, `tests/`, `experiments/` (Phase 1, EXP-001…EXP-018), `phase2/` (untracked; see git status below), `reference/`, `results/`, `docs/`, `data/`, `logs/`, plus root reports (`AUDIT_REPORT.md`, `PHASE_1_FINAL_REPORT.md`, `PHASE_1_OVERVIEW.md`, `README.md`, `requirements.txt`).

Git state (read-only `git log` / `git status`): HEAD `58e8a19 "docs: add the Phase 1 overview"`; `phase-1-frozen` commit referenced in docs is `b4207e5`. Working tree shows `M .gitignore` and untracked `phase2/`. Phase 1 `experiments/` are stated frozen; `phase2/` is the post-closure workspace and is NOT part of the frozen Phase 1 tree.

### 2.1 Source dirs (`src/`) — source, used by current pipeline

- `src/models/stereonet/` — the PyTorch model. Source; actively used (imported by `scripts/exp_train_convergence.py:52`, `phase2/scripts/*`).
  - `stereonet.py` — assembles the model; documents the degenerate baseline explicitly (lines 1–27, 53–60).
  - `feature_extractor.py` — Siamese tower (lines 1–16, 26–56).
  - `cost_volume.py` — `shift="none"` (no-op) default reproducing the reference; `shift="left"` available but not default (lines 1–25, 72–76, 123–138).
  - `aggregation.py` — 3D aggregation (lines 1–15, 25–40).
  - `regression.py` — upsample-first soft-argmin (lines 1–24, 49–61).
  - `refinement.py` — full-resolution guided refinement (lines 1–27, 39–56).
  - `blocks.py` — `ResBlock`, slopes `RESIDUAL_SLOPE=0.2`, `AGGREGATION_SLOPE=0.01` (lines 22–48); states no batch-norm in export (lines 8–15).
  - `onnx_weights.py` — positional weight-mapping reference→implementation (lines 1–17, 27–34, 60–133).
- `src/datasets/kitti2015.py` — KITTI 2015 loader + Hailo protocol (pad/crop, 1/255 vs 1/256 scales, ImageNet norm). Source; used. Key lines 1–24, 34–40, 51–68, 99–149, 155–165.
- `src/evaluation/metrics.py` — two protocols (`hailo_protocol` per-image D1 mislabeled EPE vs `kitti_official` pooled + true EPE). Source; used. Lines 1–21, 29–34, 52–84, 87–110.
- `src/losses/disparity.py` — masked smooth-L1, `valid = target > 0` (+ optional `max_disparity`). Source; used by training. Lines 21–51.
- `src/geometry/stereo.py`, `src/common/*` (`experiment.py`, `conclusions.py`, `profiler_stages.py`), `src/profiling/`, `src/deployment/` — supporting source. Evidence: file listing; `conclusions.py` is cited by EXP-016 correction as the derived-verdict mechanism.
- Evidence for "actually uses it": direct imports in `scripts/exp_train_convergence.py:42–52` and the `phase2/scripts/exp_h1_cost_volume.py` reuse statement in `phase2/docs/PHASE_2_REGISTRY.md:33–36`.

### 2.2 Training / inference / eval scripts (`scripts/`, `phase2/scripts/`) — source, mixed use

- `scripts/exp_train_convergence.py` — authoritative Phase 1 training recipe (EXP-016). Source; used for the convergence run. Full recipe lines 56–162; validation on 10 scenes lines 98–112.
- `scripts/exp_reproduce_hailo.py` — Hailo reproduction (EXP-005). Source; used. Cited in `docs/reproduction_report.md:191–202`.
- `scripts/exp_onnx_forensics.py` (EXP-001/010), `exp_degenerate_cost_volume.py`, `exp_right_image_ablation.py`, `exp_cost_sign_convention.py`, `exp_intermediate_tensors.py`, `exp_onnx_equivalence.py`, `exp_profile.py`, `exp_profiler_report.py`, `exp_quantization.py`, `depth_error_curve.py`, `inspect_onnx.py`, `analyze_architecture.py`, `verify_claims.py`, `fetch_reference.py`, `hash_reference.py` — source; each backs a named experiment or audit step. Evidence: `docs/stereonet_architecture.md:11–17` (reproduce commands for the ONNX-inventory scripts), `docs/phase_1_closure_audit.md:237–248` (18 contiguous experiment records with configs).
- `phase2/scripts/` (`exp_h1_cost_volume.py`, `exp_h2_*`, `exp_e1_*`, `exp_e2_*`, `exp_e3*`, `exp_o6*`, `h1_mechanism_probe.py`, `stereo_visualizer.py`, `gt_range_ceiling.py`) — Phase 2 source; used for H1/H2/H3/E-series. Evidence: `phase2/docs/PHASE_2_REGISTRY.md` throughout.
- `phase2/models/scaled_regression.py` — H2's one changed variable (standardised readout). Source; used only in H2 and descendants, NOT in the frozen baseline. Lines 1–36, 56–115.
- `phase2/models/__init__.py` — single-line additive marker. Source.

### 2.3 Model files / checkpoints / ONNX / Hailo artifacts

- `reference/onnx/stereonet.onnx` — 23,690,586 bytes (measured by file listing; hash `b1a01d85…` in `reference/manifest.json:22–27`). Source artifact (vendor export, held read-only). Used as the reference for architecture, weights, and EXP-005/010/011. Whether its tensors load successfully in this session: `NOT VERIFIED` (would require opening/loading the file, prohibited).
- `reference/stereonet.hef` — 24,057,165 bytes (hash `541ee6cb…`, `reference/manifest.json:29–34`). Held for completeness; per `docs/source_registry.md:33` (SR-005) "acquired, **not** decoded" and no Hailo device available. Compiled-model behavior: `NOT VERIFIED`.
- `reference/stereonet.zip` (1,617,369 bytes), profiler HTML (40,883,621 bytes), upstream zips + `reference/upstream/extracted/` — vendor/upstream sources, read-only. Manifest `reference/MANIFEST.md:1–28`.
- `results/quantization/artifacts/stereonet_int8.onnx` — 6,105,895 bytes. Generated output of `scripts/exp_quantization.py`; NOT part of the deployed baseline pipeline. Numerical behavior vs Hailo: `NOT VERIFIED` beyond the recorded `precision_comparison.json` figures (which are cited, not re-verified here).
- `results/training/convergence_run.pth` — 1,723,643 bytes. Generated output of EXP-016. Whether it loads / matches current source: `NOT VERIFIED` (requires loading; prohibited). It was produced by the script at `scripts/exp_train_convergence.py:247–252`; its config is embedded per line 248.
- `phase2/results/training/*.pth` (H1/H2/H3/E3b/E3c/O6 arms, ~1.65–1.73 MB each by listing) — generated outputs of Phase 2 runs. Loadability / weight-architecture correspondence: `NOT VERIFIED` for the same reason. Identity evidence available without loading: recorded checkpoint hashes and init-hash matches cited in registry addenda (e.g. Stage B weight SHA `3ad382c50d413ad2`, `PHASE_2_REGISTRY.md:906–921`).
- `phase2/diagnostics/`, `phase2/factorial/` checkpoints — generated; same `NOT VERIFIED` status on loadability.

### 2.4 Config files

- `reference/hailo_model_zoo/stereonet.yaml` — vendor config: input `368x1232x3` ×2, output `368x1232x1`, `operations: 112.2G`, `parameters: 623.1K`, `eval_metric: EPE`, `full_precision_result: 8.223`, normalisation lists, source link (lines 1–50). Source (vendor); used as the published-figure reference, NOT as training config.
- `reference/hailo_model_zoo/stereonet.alls` — compiler script (spatial defusion etc.). Source (vendor); evidence of hardware constraints, not executed here.
- `experiments/EXP-*/config.json`, `phase2/experiments/*/config.json` — per-run generated configs; authoritative for what was ACTUALLY USED in each run (see §7).

### 2.5 Datasets / data loaders

- Loader: `src/datasets/kitti2015.py` (above). Dataset bytes on disk: `data/` exists but was not inventoried file-by-file in this session; dataset identity (KITTI 2015 training, splits `hailo_calib` 0–159 / `hailo_val` 160–199, `_10` frames, `disp_occ_0`) is established by code (`kitti2015.py:99–136`) and `docs/reproduction_report.md:50–68`. Whether the local `data/` copy is complete/correct: `NOT VERIFIED` (would require reading image/GT files beyond allowed metadata inspection; checksums not checked here).
- Scene Flow / Middlebury: per `docs/source_registry.md:50–52` (SR-022/SR-023), marked `**no** — not yet downloaded or inspected`. Status: absent. Any claim depending on them: `NOT VERIFIED`.

### 2.6 Tests

- `tests/` (7 files: `test_stereonet.py`, `test_metrics.py`, `test_geometry.py`, `test_experiment.py`, `test_conclusions.py`, `test_profiler_stages.py`, `conftest.py`) and `phase2/tests/` — source; guardrails, not pipeline. Closure audit records 75 tests / 73 claim checks (`docs/phase_1_closure_audit.md:82–96`). Re-run here: none (execution not required for this audit; current pass status `NOT VERIFIED` by this inspection).

### 2.7 Experiment directories and audit/diagnostic records

- Phase 1: `experiments/EXP-001…EXP-018`, contiguous, each with `metrics.json`/`log.txt`/`env.json`/`config.json` (+ `CORRECTION.md` where applicable). Evidence: directory listing + closure audit `docs/phase_1_closure_audit.md:237–248`.
- Phase 2: `phase2/experiments/` (32 entries incl. H1/H2/H3/E1/E2/E3/E3b/E3c/O6 + RUN2 reruns), `phase2/results/`, `phase2/diagnostics/`, `phase2/factorial/`, `phase2/visualizations/`, `phase2/viz/`. Evidence: §2 listings + `phase2/docs/PHASE_2_REGISTRY.md`.
- Docs: `docs/*.md` (architecture, reproduction, cost-volume, closure audit, source registry, etc.) and `phase2/docs/` (27 files). These are claims-records, NOT primary evidence; used in this report only as pointers to records, never as proof.

---

## 3. THE ACTUAL MODEL (traced from code, not summarised)

All values below are read from `src/models/stereonet/*.py` and cross-checked against the ONNX-derived doc `docs/stereonet_architecture.md` (which itself cites SR-001 node-level evidence). Where the doc adds node numbers/shapes, that is cited as corroboration, not as re-verification.

### 3.1 Feature extractor (`feature_extractor.py:26–56`, `blocks.py:28–48`)

- Architecture: 4 × `Conv2d(5×5, stride 2, padding 2)` — first `3→32` (2,432 params), next three `32→32` (25,632 each) — with **no activation between them** (linear stack; `feature_extractor.py:39–43` comment "No activations here, matching the exported graph"); then 6 × `ResBlock(32)` (`Conv3×3→LeakyReLU(0.2)→Conv3×3→Add→LeakyReLU`, stride 1, padding=dilation); then 1 × `Conv2d(32→32, 3×3, stride 1, padding 1)` (`output_conv`). No batch-norm anywhere (folded at export per `blocks.py:8–15`).
- Channels: 32 throughout after the first conv. Strides: 2,2,2,2 → total stride 16 (`scale` property, lines 48–51).
- Output resolution / dims: input `(B,3,368,1232)` → `(B,32,23,77)` per branch (docblock `stereonet.py:10–11`, class docstring `feature_extractor.py:27`).
- Corroboration: `docs/stereonet_architecture.md:66–105` (nodes 0–3 table, residual-block form, BN-folding, 199,552 params / 1.436 GMAC per branch).

### 3.2 Cost volume (`cost_volume.py:1–143`)

- Inputs: left/right features `(B,32,23,77)` each.
- Candidates: `num_disparities=12` (`stereonet.py:44–51`), range index `k=0…11`, spacing 16 full-res px (0–176 px; see §3.2 of `docs/cost_volume_analysis.md:107–124`), i.e. low-res steps of 1 column.
- Construction per candidate: `reference_shift(left,k)` then `shifted − right` (subtract; `concat` mode exists but is NOT the deployed path — `build_cost_volume:58–97`, default `method="subtract"`, `shift="none"`; class defaults `CostVolume:123–134`).
- Shift direction / padding / slicing: the source files themselves call left-shift with right-edge zero fill the intended construction (`cost_volume.py:19–21`, `shift_left:34–38`, `build_cost_volume:72–75`); the reference implementation pads on the **right** (`F.pad(x,(0,k))`) then slices `[..., :width]` from index 0 (`reference_shift:45–55`), which is algebraically a **no-op returning the input unchanged**. Slicing behavior in the ONNX: every slice `starts=[0], ends=[77], axes=[3]` with `[features|padding]` operand order (EXP-010 metrics: `experiments/EXP-010/metrics.json:9–274`).
- Whether candidates differ: NO in the deployed configuration — all 12 slices bit-identical (`max_abs_diff_vs_slice_0 = 0.0` all scenes, `metrics.json:276–408`). Equivalent to a repeated tensor: YES — the single tensor `(left−right)` replicated 12× (EXP-010 conclusion line 417).
- Before vs after export: the no-op exists in the PyTorch source path reproduced here AND in the upstream source (SR-011) per `cost_volume.py:11–15`; carried through export/compilation per EXP-010 conclusion. Whether the compiled HEF preserves it: `NOT VERIFIED` (HEF not decoded, no device).
- PyTorch source behavior: the current repo's `shift="none"` reproduces it by default; `shift="left"` (`shift_left:34–42`, `F.pad(x,(0,k))[...,k:]`) is the behavior the source files describe as intended (`cost_volume.py:17–21`, `build_cost_volume:72–75`), available for measurement and deliberately not default (`stereonet.py:53–56`).

### 3.3 Aggregation (`aggregation.py:25–40`)

- Blocks: `num_layers=4` → 4 × `Conv3d(32→32, 3×3×3, stride 1, padding 1)` + `LeakyReLU(0.01)` (`AGGREGATION_SLOPE`, `blocks.py:22–25`), then final `Conv3d(32→1, 3×3×3, padding 1)` + squeeze. No residual connections. No striding (all stride 1, padding 1, so D/H/W preserved).
- Tensors: in `(B,32,12,23,77)` → filter → `(B,32,12,23,77)` → `to_cost` → `(B,1,12,23,77)` → squeeze → `(B,12,23,77)` (one scalar cost per candidate per pixel).
- Operates over candidate disparity: YES — 3×3×3 kernels span the disparity (depth) axis (`aggregation.py:8–14`).
- Corroboration: `docs/stereonet_architecture.md:128–143` (nodes 118–126, 111,585 params, 2.369 GMAC).

### 3.4 Disparity readout (`regression.py:33–61`)

- Ops: `softmax(−cost, dim=1)` (negation makes it arg-min), index grid `arange(D)`, weighted sum → `(B,1,H,W)` in **candidate units 0…11** (docstring lines 33–47).
- Temperature: NONE (plain softmax; no τ parameter in `soft_argmin` or `DisparityRegression`).
- Upsampling: `upsample_first=True` default — bilinear `F.interpolate(cost, size, mode="bilinear", align_corners=True)` on the whole `(B,12,23,77)` cost tensor → `(B,12,368,1232)`, THEN soft-argmin (`regression.py:56–61`). Alternative order supported via flag, not used in baseline.
- `align_corners`: `True` (line 58 and 61). Candidate indexing: `torch.arange(D)` broadcast (lines 43–46); ONNX constant `/Tile_output_0` `1×12×368×1232` per architecture doc §5.
- Normalization: none in the frozen path. (H2's `StandardisedDisparityRegression` adds a per-pixel z-score; that is a Phase 2 variant, NOT this model — `phase2/models/scaled_regression.py:1–36`.)

### 3.5 Refinement (`refinement.py:39–56`)

- Exists: YES, single stage at full resolution (not a multi-scale cascade).
- Inputs: `torch.cat([disparity_initial (B,1,368,1232), left RGB guidance (B,3,368,1232)], dim=1)` → 4 channels (line 53). Feature guidance: NO (image guidance only).
- Structure: `Conv2d(4→32,3×3)` → 6 × `ResBlock(32, dilation)` with `DEFAULT_DILATIONS=(1,2,4,8,1,1)` → `Conv2d(32→1,3×3)` returning a **residual** (lines 47–56; docstring lines 51–52).
- Resolution changes: none inside (padding=dilation preserves size; `blocks.py:28–34`). Final: `disparity = disparity_initial + residual`, then `ReLU` if `final_relu` (`stereonet.py:98–111`).
- Weight: carries 90.6% of MACs per architecture doc §6; code itself states no MAC figures (figures are measurements, cited in §14).

---

## 4. COMPLETE FORWARD PASS (actual implementation, `stereonet.py:98–123`)

Provenance for §4 shapes: (b) derived by hand from source code (module definitions and `forward`), NOT measured by execution here. No tensor was materialised in this session.

Input: `left, right: (B,3,H,W)`, normalisation NOT in-graph (applied by caller; `kitti2015.normalize`, compiler-inserted in deployment).

1. `left_features = FeatureExtractor(left)` → `(B,32,H/16,W/16)`; `right_features = FeatureExtractor(right)` → same. Shared weights (single module applied twice). At `368×1232`: `(B,32,23,77)`.
2. `volume = CostVolume(left_f, right_f)` (`shift="none"`, `method="subtract"`) → `(B,32,12,23,77)`; all 12 disparity slices identical (degenerate; §3.2).
3. `cost = Aggregation(volume)` → `(B,12,23,77)` (scalar cost per candidate per low-res pixel).
4. `disparity_initial = DisparityRegression(cost, size=(H,W))` — bilinear-upsample cost to `(B,12,H,W)`, soft-argmin over dim 1 → `(B,1,H,W)` in candidate units 0…11.
5. `residual = Refinement(disparity_initial, left)` — concat to `(B,4,H,W)` → convs at full res → `(B,1,H,W)`.
6. `disparity = ReLU(disparity_initial + residual)` → `(B,1,H,W)` full-resolution disparity in pixels (scale carried by refinement/training, not by the readout; `stereonet.py:18–27`).
7. Optional `return_stages=True` exposes `left_features, right_features, cost_volume, aggregated_cost, disparity_initial, refinement_residual, disparity_final` (lines 113–123).

Branch structure: two Siamese branches rejoining at subtraction; single downstream chain (no auxiliary heads, no second readout in the graph — the upstream double-compute was exporter-eliminated per architecture doc §5).

---

## 5. WHAT "HAILO MODEL" ACTUALLY MEANS HERE

Verdict: we hold **(A) + (B) + (C) + (D-as-unchecked-binary)**. We do **NOT** hold (E) for the HEF/int8 artifacts. Never write "reproduced" below without the letter.

- (A) PyTorch reproduction of the architecture: YES. `src/models/stereonet/` is a second implementation written from the ONNX-derived spec (`stereonet.py:1–6`), with a 6-stage tapped equivalence against the reference at relative ~1e-7 worst case (closure audit Finding 3: "Worst relative mean absolute difference across **all six** tapped stages: **7.596 × 10⁻⁷**", `docs/phase_1_closure_audit.md:122–126`). Parameter counts match exactly (423,586 unique / 623,138 per-occurrence, same lines).
- (B) Reproduction of Hailo's exported model: YES at the numerical-behavior level for the ONNX (not the HEF). EXP-005 runs the reference ONNX and matches the published figure to +0.0007 pt (`experiments/EXP-005/metrics.json:24–42`; `docs/reproduction_report.md:6–9`).
- (C) ONNX reproduction: the reference ONNX itself is HELD (`reference/onnx/stereonet.onnx`, 23,690,586 bytes, 168 nodes, summary `results/onnx_inspection/summary.json:46–70`); a derived int8 ONNX exists but is a separate quantization artifact, not a reproduction claim.
- (D) Actual Hailo-compiled model: HELD AS BYTES ONLY (`reference/stereonet.hef`, 24,057,165 bytes). No compilation was done here; no device run exists. Its numerical behavior is `NOT VERIFIED`.
- (E) Numerical match against Hailo: YES for the published **accuracy figure** (8.2237 vs 8.223, Hailo-exact protocol) and for the independent implementation vs the ONNX (tapped stages). NO for HEF silicon numbers (float EPE 8.22 / hardware EPE 10.3 / FPS in SR-004 remain SOURCE values, per `docs/reproduction_report.md:140–153`).

Conversion scripts / comparison scripts / logs: `scripts/onnx_weights.py`-side loader (`onnx_weights.py`), `scripts/exp_onnx_equivalence.py` + `results/equivalence/stage_differences.json`, `scripts/inspect_onnx.py` + `results/onnx_inspection/`, weight-load order check (Siamese sharing asserted in code `onnx_weights.py:76–82`). Exact validation status of each: validated per closure audit (Finding 3 PASS), except anything requiring executing silicon or opening binaries now, which is `NOT VERIFIED` in this session.

---

## 6. CHECKPOINT AUDIT

Rule applied: anything requiring execution (load the file, compare tensors) is marked `NOT VERIFIED` with the reason. No checkpoint was loaded.

| Checkpoint | Size / location | Arch expected | Epoch / dataset / seed / loss / metrics | Loads? | Weights↔source? | Used for reported results? |
|---|---|---|---|---|---|---|
| `results/training/convergence_run.pth` (EXP-016) | 1,723,643 bytes | current `StereoNet` default (`shift="none"`, no BN) | 20 epochs, KITTI `hailo_calib` train / `hailo_val` val, seed 0; loss 11.4937→7.6137; val EPE 18.497→19.216 (did NOT improve; `EXP-016/CORRECTION.md`) | `NOT VERIFIED` (loading prohibited) | `NOT VERIFIED` (requires loading + arch compare) | YES — but only as a pipeline proof, explicitly NOT an accuracy baseline (`history.json`, `PHASE_2_BASELINE.md:88–108`) |
| `phase2/results/training/EXP-H1-{BASE,WORKING}(_-v2)*.pth` | ~1.7 MB each (listing) | same + `cost_volume_shift` per arm | H1 v1 20 ep / v2 200 ep, same recipe, seed 0; v2 late-window EPE ~14.42/14.41, D1 ~81.4/80.8 (`H1_V2_ANALYSIS.md:35–67`) | `NOT VERIFIED` | `NOT VERIFIED` | YES — as the H1 A/B records (later narrowed by H2: pathway unlearnable; registry H2 entry) |
| `EXP-H2-SOFTARGMIN-SCALE*.pth`, seed-replication, H3, E3b/E3c/O6, factorial/diagnostic checkpoints | ~1.65–1.73 MB by listing | same + standardised readout (H2 onward) and/or block-count/dilation arms | per-arm epochs/seeds in registry (§10; addenda A/B) | `NOT VERIFIED` | `NOT VERIFIED` | YES — per their experiment records; authoritative-vs-superseded noted per record (RUN2 pattern) |
| Hailo `stereonet.zip` pretrained weights (inside `reference/stereonet.zip`) | 1,617,369 bytes (archive) | upstream training-time model (with BN, per source) | `NOT VERIFIED` (archive not unpacked/inspected here; training provenance per SR-011 only) | `NOT VERIFIED` | `NOT VERIFIED` | INDIRECTLY — the exported ONNX derived from this lineage is what EXP-005 runs; the zip itself is not the evaluated artifact |

Authoritative/current checkpoint: there is NO single "current model checkpoint" by content. By role: the **reference ONNX weights** (via `onnx_weights.py` mapping) are the authoritative weights for the *deployed-behavior* baseline (EXP-005 accuracy, EXP-010 degeneracy); `convergence_run.pth` is authoritative only as the *training-pipeline proof*. Anything beyond that (e.g. "weights correspond to current source") is `NOT VERIFIED` without execution.

---

## 7. TRAINING PIPELINE (EXP-016 recipe; H1/H2 reuse it)

Source: `scripts/exp_train_convergence.py` (code) + `experiments/EXP-016/config.json` + `PHASE_2_BASELINE.md:88–108`. Status words: IMPLEMENTED = in code; CONFIGURED = in a run's `config.json`; ACTUALLY USED = in the authoritative run's record.

| Element | IMPLEMENTED | CONFIGURED (EXP-016) | ACTUALLY USED (authoritative run) |
|---|---|---|---|
| Dataset / split | `Kitti2015Stereo` splits `hailo_calib`/`hailo_val` (`kitti2015.py:99–136`) | hailo_calib (0–159) train, hailo_val (160–199) val | YES, EXP-016 (160 train scenes) |
| Image preprocessing | `normalize()` ImageNet stats (`kitti2015.py:38–40,155–165`); pad/crop top-left 368×1232 | via loader | YES |
| Disparity preprocessing | raw/256 in loader default; training mask `gt>0 & gt<max_disparity` (`disparity.py:28–46`; call site `exp_train_convergence.py:188–191`) | `max_disparity=176` (`config.max_disparity_px` = 11×16) | YES |
| Crop | random 256×512, must divide by 16 (`exp_train_convergence.py:56–60,76–95`) | 256×512 | YES |
| Augmentation | random crop + independent per-image gain jitter σ=0.1; NO h-flip (`:61–95`, config lines 155–157) | as implemented | YES |
| Optimizer / LR / schedule | Adam betas (0.9,0.999), lr 1e-3, cosine anneal T_max=epochs (`:136–139`) | lr 1e-3, cosine | YES |
| Batch / epochs | args `--epochs 20 --batch 2` (`:116–121`) | 20 / 2 | YES (H1-v2/H2 use 200; separate configs) |
| Loss | masked smooth-L1 β=1.0 (`disparity.py:21–51`) | as implemented | YES |
| Loss weighting | none beyond masking (no multi-scale weighting in code) | n/a | n/a |
| Initialization | PyTorch defaults, random (`exp_config` lines 157–162) | random, seed 0 | YES |
| Seed handling | `torch.manual_seed + np.random.seed` (`:124–125`); DataLoader shuffle True, `num_workers=0` (`:131`) | seed 0 | YES (determinism NOT enforced in Phase 1/H1-v2/H2 — later found nondeterministic; registry A.1) |
| Mixed precision | fp32 only (`exp_config` "precision": "fp32") | fp32 | YES |
| Gradient clipping | `clip_grad_norm_` with `max_norm=1e9` = effectively none (`:194–196`) | as implemented | YES |
| Checkpoint selection / early stopping | NONE — single final `torch.save` (`:247–252`); validation only logged at epochs %5 (`:213–217`) | n/a | n/a (best-epoch noted, not selected) |
| Evaluation frequency | validation every 5 epochs, 10 scenes (`validate(..., limit=10)`, `:98–112,213`) | as implemented | YES |

Known deviation (IMPLEMENTED and USED): **no batch-norm** — model reproduces the BN-folded export, not the training-time upstream model (`exp_train_convergence.py:15–21,158–162`). Effect on convergence: stated as expectation (slower), not measured against a BN variant here.

Phase 2 deltas (for identity clarity, not training claims): H1 changes only `cost_volume_shift`; H2 swaps in `StandardisedDisparityRegression`; E3b/E3c/O6 change block counts/dilations; all else imported from this recipe (registry).

---

## 8. EVALUATION PIPELINE

### 8.1 Protocols (code-defined, `metrics.py` + `kitti2015.py` + reproduction report)

- Resolution/crop: pad bottom/right to ≥368×1232, crop top-left 368×1232, no resize (`kitti2015.py:51–68`). For KITTI 375×1242: removes 7 bottom rows + 10 right cols (road/close-range pixels; `:57–61` comment).
- GT source: `disp_occ_0` (D1-all) default; `disp_noc_0` variant exists (`kitti2015.py:112–114`).
- GT scaling: TWO scales — official `1/256` vs Hailo `1/255` (1.0039× larger; `metrics.py:29–34`, `kitti2015.py:18–20`). Choice is per-variant.
- Valid mask: `gt > 0` everywhere (`metrics.py:58–66,103–110`). Occlusion handling: none beyond the `occ`/`noc` file choice (occluded pixels INCLUDED in `disp_occ_0`).
- Metric definitions: EPE = mean |pred−gt|; D1 = outlier fraction (err>3 AND err>5% of gt), in percent; bad1/2/3, RMSE likewise (`metrics.py:71–84`). Hailo variant uses strict `<` vs `<=` at boundary (lines 97–101; negligible, recorded).
- Averaging: per-image mean (Hailo) vs pooled over pixels (KITTI official) (`metrics.py:87–110` + demo lines 232–250).
- Output postprocessing: ReLU clamp in-graph (`stereonet.py:110–111`); NO depth conversion in the model or shipped app (app postprocess is an 8-bit cast per SR-007; depth via `StereoCalibration` guarded in viz only).

### 8.2 Result table (one row per recorded protocol; numbers quoted from records, not recomputed)

| Result | Dataset | Pixels | GT handling | Metric | Protocol | Source |
|---|---|---|---|---|---|---|
| Published figure | KITTI 2015 train slice (scenes 160–199, `_10`) | unstated by vendor | `disp_occ_0`, 1/255, `gt>0`, per-image avg, ×100 | D1 **mislabeled "EPE"** in config | `hailo_exact` (Hailo's evaluator reimplemented) | `stereonet.yaml:43–44` + `reproduction_report.md:22–46` |
| EXP-005 `hailo_exact` | same 40 scenes | 3,802,797 valid | `disp_occ_0`, 1/255, per-image | 8.2237% (gap +0.0007); pooled EPE 1.3423 on same preds | hailo_exact | `EXP-005/metrics.json:24–42` |
| EXP-005 `hailo_scale_kitti_gt` | same | 3,802,797 | `disp_occ_0`, 1/256, per-image | 8.1685%; EPE 1.3134 | Hailo-avg + official scale | same `:43–61` |
| EXP-005 `kitti_official_d1_all` (baseline) | same | 3,802,797 | `disp_occ_0`, 1/256, pooled | EPE **1.3134 px**, RMSE 2.5830, D1 **8.1544%**, bad1 38.90/bad2 16.08/bad3 8.64 | KITTI official D1-all | same `:62–80` |
| EXP-005 `kitti_official_d1_noc` | same | 3,721,216 | `disp_noc_0`, 1/256, pooled | EPE 1.2607, D1 7.5301% | KITTI official D1-noc | same `:81–99` |
| EXP-007 baseline | subset (recorded 1,837,304 valid px — NOT the full 3.8M) | 1,837,304 | per `EXP-007/metrics.json` baseline | EPE 1.4422, D1 9.3315% | ablation baseline (own pixel set) | `EXP-007/metrics.json:10–25` |
| EXP-007 ablations | same subset | 1,837,304 | right-image corruptions | D1 penalties +89.13…+89.96 pt; EPE up to 69.43 | same-subset ablation | same `:26–108` |
| EXP-016 validation | `hailo_val`, 10-scene `limit=10` pooled concat (`exp_train_convergence.py:98–112`) | ~5M train px/epoch; val pooled over 10 scenes | `gt>0` (+`max_disparity` in loss only) | val EPE 18.497→19.216; D1 88.66→91.12 | 10-scene train-time val | `EXP-016/metrics.json:9–18,151–159` + CORRECTION |
| Phase 2 late-window / 40-scene figures (H1v2, H2, E-series, factorial) | `hailo_val` 40 scenes or 10-scene late windows per record | 3,802,797 (40-scene) or 10-scene windows | per-record (mostly `gt>0`, official-ish pooling; exact per report) | see §9–10 | per-report (DO NOT mix with EXP-005) | `H1_V2_ANALYSIS.md`, `PHASE_2_REGISTRY.md`, `POST_CLOSURE_RESEARCH_AUDIT.md` |

Flags: (i) the vendor `eval_metric: EPE` label is wrong — the quantity is D1% (mislabeled-metric `CONTRADICTION - REQUIRES RESOLUTION` between `stereonet.yaml:43` and the evaluator source; resolved in favor of the evaluator per reproduction report §1); (ii) EXP-007's 9.3315% baseline and EXP-005's 8.154% are DIFFERENT pixel sets — never combine (post-closure audit Phase 5 §2 bookkeeping item); (iii) 1/255 vs 1/256 (+0.055 pt) and per-image vs pooled (+0.014 pt) are measured isolated effects (`reproduction_report.md:109–117`).

---

## 9. THE TRUE BASELINE

Strongest experimentally supported baseline for the model we HAVE (pretrained Hailo-exported weights, reference ONNX):

- **Baseline R (deployed reference, official protocol):** Protocol `kitti_official_d1_all` (40 `hailo_val` scenes, 368×1232 top-left crop, `disp_occ_0`, 1/256, `gt>0`, pooled) / Checkpoint: reference ONNX weights (SR-001 lineage) / EPE **1.3134 px** / RMSE 2.5830 / D1 **8.1544%** (D1-noc 7.5301%) / Valid pixels **3,802,797** / Dataset KITTI 2015 train slice. Source `EXP-005/metrics.json:62–80`.
- **Baseline R-Hailo (same model, vendor protocol):** Protocol `hailo_exact` (same scenes, 1/255, per-image avg) / same weights / headline **8.2237%** (reproduces published 8.223) / Valid pixels 3,802,797. Source same `:24–42`.
- **Baseline T (from-scratch training recipe, NOT competitive):** Protocol 10-scene train-time val / Checkpoint `convergence_run.pth` / EPE 19.216 px final (18.497 initial — did not improve) / D1 91.12% / KITTI `hailo_calib`-trained. Source EXP-016 + CORRECTION. This is a pipeline proof, not an accuracy baseline.
- **Phase 2 from-scratch references (same budget family, separate protocols — listed separately, not selected):** H1-v2 late-window EPE ~14.41/D1 ~80.8–81.4 (10-scene windows); H2 seed0 late-window EPE 4.199±0.090/D1 24.78±1.02; deterministic Stage B 6-block seed0 40-scene EPE 2.2955/D1 15.60 (`PHASE_2_REGISTRY.md:918–921`); shift factorial `none`+std 8.3717/66.60 vs `left`+std 2.2955/15.60 (one seed each; `REGISTRY A.3`); 3×3 block-count table (§10, Addendum B). None of these replaces Baseline R.

No "best-looking number" was selected: the reported baseline is the official-protocol figure with its exact protocol.

---

## 10. EXPERIMENT HISTORY (records read; nothing reproduced)

### Phase 1 (EXP-001…018; `phase_1_closure_audit.md:237–248`, individual `metrics.json`)

| Exp | What was run | Role | Status for future model work |
|---|---|---|---|
| EXP-001 (ONNX forensics/static analysis) | 168-node inventory, shapes, param/MAC accounting | control/measurement | TRUSTWORTHY as architecture/cost facts (56.04 GMAC, 423,586/623,138 params, stage split) |
| EXP-002→003 | calibration correction chain | corrected | use EXP-003 (fB etc.), not EXP-002 |
| EXP-004 | (geometry/error-propagation per charter sequence) | measurement | trustworthy within its record; not load-bearing here |
| EXP-005 | Hailo figure reproduction, 4 protocol variants | baseline anchor | TRUSTWORTHY — the frozen accuracy baseline (§9) |
| EXP-006/008 | stage-tap correlations, refinement-share (76.5%), anti-correlation r=−0.98 | diagnostic (identifiability-adjacent) | SUPERSEDED as mechanism evidence by EXP-010 (they described symptoms of the degeneracy); raw numbers remain but do not diagnose beyond EXP-010 |
| EXP-007 | right-image ablations (5 modes) | model-performance + correspondence-identification dual-use | TRUSTWORTHY for binocular dependence (+89–90 pt); mechanism reading bounded by Level D |
| EXP-009→011 | equivalence chain (independent implementation vs ONNX) | control | use EXP-011 (relative ~1e-7); EXP-009 superseded |
| EXP-010 | degenerate-shift proof (graph + 5-scene measurement) | MODEL-DEVELOPMENT + IDENTIFICATION anchor | TRUSTWORTHY, central — the cost-volume fact everything else builds on |
| EXP-012 | depth validation by range band | measurement | TRUSTWORTHY as disparity→depth error facts on this calibration |
| EXP-013/014 | GPU/CPU latency profiling (own stack) | measurement | TRUSTWORTHY as own-stack latency; NOT Hailo silicon |
| EXP-015 | quantization (fp16 neutral, int8 cost) | measurement | TRUSTWORTHY within own stack; int8 artifact is separate |
| EXP-016 (+CORRECTION) | 20-epoch convergence proof | control (pipeline) | TRUSTWORTHY as pipeline proof; conclusion corrected (val did NOT improve); NOT an accuracy baseline |
| EXP-017→018 | profiler decode + stage-mapping fix | corrected | use EXP-018; EXP-017's 95.18% WITHDRAWN (kept with CORRECTION) |

MODEL-DEVELOPMENT vs CORRESPONDENCE-IDENTIFICATION split: EXP-001/005/011/012/013/014/015/016/018 are development-usable measurements; EXP-006/007/008/010 carry identification load (010 is both; 007's dependence number is development-usable, its mechanism reading is capped at Level D).

### Phase 2 (registry + post-closure audit; `PHASE_2_REGISTRY.md`, `POST_CLOSURE_RESEARCH_AUDIT.md`)

- H1 v1 (20 ep): controlled A/B, INCONCLUSIVE (neither arm learned). Development value: none beyond motivating v2; identifiability value: none (voided by H2 mechanism finding).
- H1 v2 (200 ep): EPE Δ −0.008 (coin-flip), D1 Δ −0.64 pt (10/11). Development value: NARROWED — measured on an unlearnable pathway (saturated readout), so NOT a cost-volume worth measurement. Trustworthy as a record, void as an answer.
- H1 mechanism investigation (12 probes): saturated soft-argmin (entropy 0.0000), BASE monocular (bit-identical under right-image swaps, `disparity_initial`=11.0 const), gradient starvation (5/150 vs 0/150). Development value: HIGH — identified the readout as the binding defect. Trustworthy.
- H2 (standardised readout): entropy →1.67–1.88 nats, gradient 100%, r +0.98, right-dependence +75–84 pt, EPE 14.41→4.20 (seed 0). Replicated seeds 1–2 (EPE 3.745/4.333, dependence +74–84 pt). Development value: HIGH — first trainable matching pathway; the recipe change future work builds on. Invalid epoch-10 gate episode preserved as a methodological warning.
- H3 (10-ep gate, shift-off under H2): KILL (right-dependence −1.1…+1.9). Development value: screening only; superseded as evidence by the 200-ep deterministic shift factorial (A.3: 2.2955 vs 8.3717 EPE, 59× seed scale).
- Emergence (snapshots): onset seed1 (10,20], seed2 (20,50]; gradient≠function (100% throughout). Development value: HIGH as evaluation methodology (no stereo gate before epoch 20).
- E1 (3 oracles): NO CONSISTENT VERDICT; held-out fixed point ~⅓ upstream / ~⅔ downstream. Development value: error-budget accounting (capacity-dependent); the inverse-oracle run DIVERGED and is NEVER citable (false-confirmation warning).
- E3 (13 ablations ×3 seeds): pruning verdict RIGHT-SIZED (best saving 0.0%); block-redundancy is per-run; <5 blocks kills stereo. Development value: pruning facts + the retraining-vs-pruning distinction; NOT a capacity verdict (E3b onward answers that).
- E2 (9 temps ×3 seeds): READOUT-ROBUST (no T improves); T≠1 destroys stereo both directions. Development value: readout-optimum facts; inference-only limit stated.
- E3b/E3c/O6 (nondeterministic era): CAPACITY-REDUCIBLE readings recorded BUT resting on the invalidated single-sample band — see A.0 annotations: E3b "Four blocks match six" SUPERSEDED, E3c label CONTRADICTED by O6's +0.0603 remeasurement. Development value: HISTORICAL/INCONCLUSIVE only; do not carry forward as evidence.
- Deterministic campaign (A.1–A.8, B.1–B.2): Stage A (nondeterminism proven, `F.interpolate` backward isolated); Stage B (bit-identical weights SHA `3ad382c50d413ad2`; 40-scene EPE 2.2955182/D1 15.6045143); A.3 shift factorial (SHIFT-IMPORTANT, one seed/arm); A.4–A.8 3×3 block-count (5-block reproducible EPE penalty +0.0748 disjoint, 9/9; 4-block INCONCLUSIVE, straddles, 7/9; no causality; D1 exception 5b/seed1 beats 6b/seed1). Development value: the CURRENT trustworthy from-scratch evidence (with stated ceilings: 160 scenes, random init, one recipe/dataset).

---

## 11. CURRENT MODEL WEAKNESSES (no solutions; from code + already-recorded evidence only)

Scale note: "the model" below means the frozen deployed-configuration baseline (degenerate volume, plain readout) PLUS the H2-line findings where explicitly labeled — because future development starts from the H2 recipe, and the weaknesses that survive H2 are the ones that matter. Each item carries its classification.

1. Cost-volume candidate invariance (all 12 slices identical; no search) — CONFIRMED (EXP-010 identity + 0.0 measurement; code `reference_shift`). Survives as a property of the BASELINE; removed in H1/H2-line arms by `shift="left"`.
2. Candidate resolution / spacing (16 px steps; 24 m→infinity in one interval; far-range interpolation-only) — SUPPORTED (spacing arithmetic from verified D=12/stride 16 + calibration; `cost_volume_analysis.md:107–141` marks the error-concentration as HYPOTHESIS — hence SUPPORTED, not CONFIRMED).
3. Disparity search range ceiling 176 px (min range ~2.2 m) — CONFIRMED as geometry; as a binding constraint on `hailo_val`: NOT binding (0% of GT above ceiling, max 153.04 px; evidence-review §2.9 + `gt_range_ceiling.py`). Classify: CONFIRMED-non-binding on this slice; POSSIBLE elsewhere.
4. Feature resolution 1/16 (23×77) — CONFIRMED (shapes). Whether it limits accuracy: SUPPORTED (thin-structure/boundary risk + refinement-load evidence) but not isolated by any single-variable test → SUPPORTED, not CONFIRMED.
5. Linear downsampling stack (no activations; composes to one linear operator) — CONFIRMED (code + ONNX node walk). Accuracy cost: UNKNOWN (no with/without-activation comparison in records).
6. Feature quality (trained features only +0.414 hand-made-volume correlation vs +0.98 full path) — POSSIBLE (demoted from SUPPORTED on self-review 2026-09-14: single H2 mechanism record, single recipe — only one independent piece of evidence, so it does not meet the SUPPORTED bar).
7. Aggregation capacity (5 Conv3d, 4.2% MACs) — POSSIBLE as a limiter; E-series never isolated it (E4-cheap unrun). UNKNOWN-to-POSSIBLE → POSSIBLE.
8. Candidate discrimination / saturated readout (entropy 0.0000, zero derivative in H1 arms) — CONFIRMED for the plain readout; FIXED (mechanistically) by standardisation in H2 (entropy 1.67–1.88, gradient 100%). As a future risk (T≠1 re-collapses it): POSSIBLE (demoted from SUPPORTED on self-review 2026-09-14: E2 inference-only sweep alone — one independent source).
9. Softmax confidence cap under standardisation (~0.77 max weight; `disparity_initial` attainable 1.34–9.03 of 0–11) — CONFIRMED as numerics; accuracy cost: UNKNOWN (flagged in H2 report, untested).
10. Raw cost-scale inflation (std →~2e11 during training; made irrelevant, not prevented) — CONFIRMED as measurement; harmfulness: UNKNOWN.
11. Refinement dominance (90.6% MACs; downstream error share ~⅔ in E1 held-out) — CONFIRMED as cost/error-location; "oversized" verdict: CONTRADICTED-IN-PART (pruning says right-sized; retraining says 5-block penalty reproducible but small, 4-block inconclusive) → capacity risk SUPPORTED, oversize claim NOT supported.
12. Refinement as the stereo path (truncation <5 blocks → negative right-dependence) — SUPPORTED (E3, 3 seeds, inference-only; retrained small stacks stay stereo-functional per E3b/c + 3×3).
13. Edge/boundary handling, occlusions, thin structures, textureless regions — UNKNOWN (Middlebury scene-class analysis DEFERRED since Phase 1; no occlusion mask/repetition detector in any record).
14. Subpixel estimation between 16-px candidates — POSSIBLE concern (readout interpolates with no intermediate evidence; `regression.py:15–20`); measurement: UNKNOWN (E2 temperature sweep is the only probe and it is inference-only).
15. Training loss/dataset regime (masked smooth-L1, no BN, 160 scenes, random init; from-scratch gap vs reference is protocol-separated: deterministic 6-block seed0 40-scene official-ish EPE 2.2955 vs H2 late-window 10-scene EPE 3.7–4.3 vs reference-ONNX official-protocol EPE 1.3134) — CONFIRMED as regime description; how much of the gap is regime vs architecture: UNKNOWN (O6 Scene Flow pretraining blocked on download, ~22 h estimate).
16. Large-disparity / close-range handling — POSSIBLE (range floor 2.2 m confirmed geometrically; no close-range-stratified error in records) → POSSIBLE.
17. Export/deployment numerical effects (BN folding, int8 degradation EPE 1.31→1.66/D1 8.15→10.95 in own-stack OBS; HEF silicon EPE 10.3 SOURCE) — SUPPORTED for int8-on-own-stack; UNKNOWN for HEF silicon (no device).

---

## 12. FACT vs INFERENCE (mandatory table)

| Statement | Status | Evidence |
|---|---|---|
| Input is 2× `1×3×368×1232` NCHW, output `1×1×368×1232` disparity | CONFIRMED | `results/onnx_inspection/summary.json:11–45`; `stereonet.yaml:36–37` |
| Feature maps are `(B,32,23,77)` (1/16, 32 ch, Siamese-shared) | CONFIRMED | `feature_extractor.py:26–56`; ONNX branch weight-name match (`stereonet_architecture.md:66–70`); `onnx_weights.py:76–82` check |
| Downsampling stack has no activations (linear) | CONFIRMED | `feature_extractor.py:39–43`; architecture doc nodes 0–3 + SR-011 cross-read |
| Deployed cost volume has 12 candidates, subtraction, `shift="none"`, all slices identical | CONFIRMED | `cost_volume.py:44–55,84–97`; `EXP-010/metrics.json:275–408` (all 0.0); conclusion line 417 |
| Aggregation is 4×Conv3d(32→32,3³)+LeakyReLU(0.01) + Conv3d(32→1,3³), stride 1, no residuals | CONFIRMED | `aggregation.py:25–40`; `blocks.py:22–25` |
| Readout is upsample-cost-first bilinear (`align_corners=True`) soft-argmin, no temperature, candidate units | CONFIRMED | `regression.py:49–61` |
| Refinement is 6 dilated residual blocks (1,2,4,8,1,1), image-guided, residual+ReLU output | CONFIRMED | `refinement.py:39–56`; `stereonet.py:98–111` |
| Unique params 423,586; per-occurrence 623,138 (≈Hailo 623.1K); MACs 56.04 G (≈112.08 G ops vs 112.2G) | CONFIRMED | (a) read from pre-existing `summary.json:67–70` + onnx report largest-initializer list + architecture doc §8 + closure Finding 3; NOT measured here |
| Published 8.223 is D1-% (per-image, 1/255, occ-included), NOT EPE; true EPE 1.313 px | CONFIRMED | `EXP-005/metrics.json` 4 variants; evaluator-source reading in reproduction report §1 |
| Reference ONNX scores EPE 1.3134/D1 8.154% on the official 40-scene protocol | CONFIRMED | `EXP-005/metrics.json:62–80`; `variants.json:40–57` |
| The deployed model depends on the right image (+89–90 D1 pt) despite no search | CONFIRMED | `EXP-007/metrics.json:26–108` |
| EXP-016 validation did not improve (18.497→19.216) | CONFIRMED | `EXP-016/metrics.json:9–18,151–159`; `CORRECTION.md:6–18` |
| Plain-readout H1 arms are gradient-starved/monocular (entropy 0.0, const 11.0, 0–5/150) | SUPPORTED | H1 mechanism record via registry + evidence review §2.3 (3-record chain; single-recipe) |
| Standardised readout restores trainable binocular path (entropy ~1.7–1.9, 100% gradient, +75–84 pt dependence) | SUPPORTED | H2 + 2-seed replication (3 seeds, one recipe/budget) |
| Candidate-varying volume content is material given the fixed readout (6.08 px gap, ~59× seed scale) | POSSIBLE (demoted from SUPPORTED on self-review 2026-09-14: A.3 factorial is one seed per arm — a single independent run, replication pending; effect size noted but does not meet the multi-source SUPPORTED bar) | A.3 factorial (one seed/arm — effect size carries interest, replication pending) |
| 5 blocks carry a reproducible EPE penalty vs 6 (ranges disjoint +0.0748, 9/9) | SUPPORTED | 3×3 deterministic design (n=3/arm, descriptive, D1 exception recorded) |
| 4 blocks match/miss six; monotone capacity ordering; any causal (params/RF) reading | UNKNOWN | 4-block arm straddles (7/9); pre-registered INCONCLUSIVE; causality disclaimed in B.2 |
| HEF silicon accuracy/latency/FPS/power; int8-on-Hailo behavior | UNKNOWN | No device; SR-005/006 not decoded to measurements here; compiler FPS is an estimate |
| Any checkpoint in this session loads successfully / matches source arch | UNKNOWN | `NOT VERIFIED` — loading prohibited; stated for every `.pth`/`.onnx`/`.hef` in §6 |
| Scene-class failure modes (textureless/repetitive/reflective/thin/boundary/occlusion) | UNKNOWN | SR-022/023 never acquired; no mask/detector in any record |
| The 112.08 vs 112.2 G gap is elementwise-op accounting | POSSIBLE | Order-of-magnitude plausibility only (0.34 G elementwise pool); marked INFERENCE in architecture doc A6 |
| Upstream README "concat" describes the deployed model | POSSIBLE→REJECTED as deployment description | `PREVIOUS CLAIM NOT SUPPORTED BY CURRENT EVIDENCE` — ONNX uses subtraction (`summary.json` operator inventory: 12 Sub, first Conv3d 32-ch); README claim preserved only as a source note |

---

## 13. CONTRADICTIONS (actively hunted)

1. **Metric naming (EPE vs D1).** `stereonet.yaml:43–44` (`eval_metric: EPE`, `full_precision_result: 8.223`) vs the evaluator computing per-image D1% — `CONTRADICTION - REQUIRES RESOLUTION`, resolved in favor of the evaluator (EXP-005: 8.2237 vs true EPE 1.313). Do not quote 8.223 as pixels.
2. **Cost-volume mode (concat vs subtract).** Upstream README (concat) vs ONNX (12 Sub nodes, 32-ch Conv3d) — `CONTRADICTION - REQUIRES RESOLUTION`, resolved for the DEPLOYED model in favor of the ONNX (provenance (a): pre-existing `summary.json:47–63` operator inventory; `cost_volume_analysis.md:71–85`). Both values preserved; the README describes source options, not the artifact.
3. **Parameter/operation counts.** 623.1K vs 423,586 unique; 112.2G vs 56.04 GMAC — resolved as conventions (Siamese double-count; 2×MACs), residual −0.11% unattributed (`stereonet_architecture.md:228–257`). Not errors, but unstated conventions.
4. **GT scaling.** 1/255 (Hailo parser) vs 1/256 (KITTI) — confirmed deviation, +0.055 pt effect. Neither "wrong" once stated; mixing them is an error.
5. **Averaging.** Per-image mean vs pooled — +0.014 pt here; different quantities with the same name. Never interchange.
6. **Pixel-set mixing.** EXP-007 baseline (1,837,304 px, D1 9.3315) vs official protocol (3,802,797 px, D1 8.154) — different sets; the post-closure audit flags this as a table-construction requirement (Phase 5 §2).
7. **Checkpoint mismatch (EXP-016 conclusion).** `metrics.json:180` ("validation improves") vs its own series (18.497→19.216) — withdrawn by `CORRECTION.md`. `PREVIOUS CLAIM NOT SUPPORTED BY CURRENT EVIDENCE` (original sentence); current status: did-not-improve.
8. **PyTorch vs ONNX (activations/BN).** No conflict found: both lack downsampling activations and both lack runtime BN (folded) — agreement, not contradiction. Recorded because it is the expected mismatch site.
9. **ONNX vs HEF.** No verified relation: HEF behavior unmeasured; any ONNX→HEF equivalence claim is `NOT VERIFIED`.
10. **Stale results still on disk.** EXP-002 (superseded by 003), EXP-009 (by 011), EXP-017's 95.18% (WITHDRAWN, kept with CORRECTION), E3b "four match six" (SUPERSEDED per Addendum A.0), E3c "reducible-to-3" (CONTRADICTED by O6 remeasurement +0.0603 vs +2.0304), first-attempt records (aborted seeds, encoding-crash E3, reader-bug emergence, wrapper-defect Stage B) — all preserved with NOTEs. Rule: cite only the authoritative record named in each registry entry.
11. **Candidate range / disparity conventions.** 12 candidates × 16 px = 0–176 px ceiling vs GT max 153.04 px — no contradiction on `hailo_val` (0% exceedance); image-coordinate convention verified photometrically (left-referenced `x_right=x_left−d`) per evidence review §2.8.
12. **Documentation vs implementation drift risk.** `phase_1_closure_audit.md` asserts `cost_volume_shift` defaults to `"none"` with no script enabling `"left"` IN PHASE 1 — TRUE for Phase 1 (`stereonet.py:56`, `cost_volume.py:127`); Phase 2 scripts DO enable `"left"` (H1 onward). Reading the Phase 1 sentence as a whole-project claim would be false. Scope labels matter.
13. **Registry staleness (self-corrected).** Registry body entries for E3b/E3c/O6 predate Addenda A/B; the addenda explicitly supersede/retire them (A.0 table). Citing the body without the addendum is a known error mode.

---

## 14. FINAL ARCHITECTURE REPORT (verified values only; rest NOT VERIFIED)

| Item | Value |
|---|---|
| Input | 2× RGB rectified, `1×3×368×1232` NCHW each, float32; static batch 1 in graph; normalisation external (ImageNet mean `[123.675,116.28,103.53]`, std `[58.395,57.12,57.375]`) |
| Feature extractor | 4× Conv2d 5×5 s2 (3→32, 32→32 ×3), NO activations; 6× ResBlock(32, LeakyReLU 0.2); Conv2d 32→32 3×3; no BN; shared Siamese |
| Feature resolution | 23×77 (1/16), 32 ch per branch |
| Cost volume | 12 candidates (0–11 low-res ≡ 0–176 px full-res, 16-px steps); subtraction `left−right`; reference shift = no-op (all slices identical); shape `(B,32,12,23,77)`; 0 MACs |
| Aggregation | 4× Conv3d(32→32,3×3×3,s1,p1)+LeakyReLU(0.01); 1× Conv3d(32→1,3×3×3); out `(B,12,23,77)`; 111,585 params; 2.369 GMAC |
| Readout | bilinear cost upsample `(B,12,23,77)→(B,12,368,1232)` (`align_corners=True`); `softmax(−C)`; `Σk·p_k`; candidate units 0–11; no temperature |
| Upsampling | cost-tensor-first (not disparity-first); single `Resize` node |
| Refinement | concat(disparity, left-RGB)→Conv(4→32,3×3); 6× ResBlock dilations (1,2,4,8,1,1); Conv(32→1,3×3); residual add + ReLU; full-res throughout |
| Output | `(B,1,368,1232)` float32 disparity, ReLU-clamped |
| Parameter count | 423,586 unique learned; 623,138 per-occurrence (Hailo convention); file-weight bytes 1,694,344; index-grid constant 21,762,048 bytes (92% of ONNX) — provenance (a): read from pre-existing `results/onnx_inspection/summary.json:64–70` and `docs/stereonet_architecture.md:228–257`; NOT measured here |
| Compute | 56.04 GMAC/image (112.08 G ops @2×); stage split FE 5.1 / volume 0.0 / agg 4.2 / readout ~0.01 / refinement 90.6 % MAC; own-stack latency RTX4060 44.4 ms fp32 (NOT silicon); activation peak 55.34 MiB (refinement); traffic 1,899 MiB — provenance (a): read from pre-existing EXP-001/architecture-doc accounting (`docs/stereonet_architecture.md:228–257`); NOT measured here |
| Training-time architecture (BN, init, schedule details beyond §7) | `NOT VERIFIED` (BN-folded export is what we hold; upstream training-time graph not re-derived here) |
| HEF-internal architecture mapping | `NOT VERIFIED` (binary not decoded) |

---

## 15. FINAL STATUS (exactly these sections)

### WHAT WE ACTUALLY HAVE

- (A) An independent PyTorch StereoNet reproducing the deployed architecture including its two signature defects (linear downsampler; degenerate no-op shift; full-res soft-argmin in candidate units), with exact param-count agreement and ~1e-7 tapped-stage agreement vs the reference ONNX.
- (B) The reference ONNX (23.7 MB, 168 nodes) whose 40-scene behavior reproduces Hailo's published 8.223 as 8.2237 under the reimplemented Hailo-exact protocol.
- (C) The same ONNX as a weight source (positional mapping, Siamese sharing asserted in code).
- (D-as-bytes) The compiled HEF (24.1 MB) + compiler profiler HTML + vendor config/alls + upstream archives — HELD, not decoded, not run.
- A frozen 160-scene from-scratch training recipe (no BN, masked smooth-L1, 256×512 crops, Adam 1e-3 cosine) that is a pipeline proof (EXP-016), plus its Phase 2 descendants (H1/H2/E/O/factorial lines) with the standardised-readout variant as the current functional baseline.
- A complete, correction-preserving experiment record (18 Phase 1 + ~30 Phase 2 records, two addenda) with the deterministic 3×3 block-count design as the latest closed campaign.

### WHAT HAS BEEN VERIFIED

- Graph, shapes, counts, costs (§14); degeneracy identity + 0.0 measurement; 8.223 = D1% with true EPE 1.313 px; 40-scene official figures (EPE 1.3134 / D1 8.154%); right-image dependence (+89–90 pt); pipeline runs end-to-end with finite gradients; plain-readout unlearnability and its standardised fix (3 seeds); shift materiality given the fix (one-seed 6.08 px gap, ~59× seed scale); emergence timing (epochs 20–50); E1/E2/E3 verdicts with stated limits; 5-block EPE penalty (3 seeds, D1 exception); 4-block inconclusiveness; determinism controls (identical weight SHA).

### WHAT HAS NOT BEEN VERIFIED

- Every loadability/weight-correspondence question for every `.pth`/`.onnx`/`.hef` in THIS session (`NOT VERIFIED` — execution prohibited).
- All HEF silicon behavior (accuracy, latency, FPS, power, utilisation); compiler FPS is an estimate, not a measurement.
- Local `data/` completeness; Scene Flow/Middlebury (absent); training-time (pre-fold) architecture details; latency of any from-scratch model; scene-class failure modes; causal readings of block-count effects; any significance/p-value; transfer beyond (160 scenes, seed set, recipe, dataset, 12 candidates).

### CURRENT BASELINE

- Deployed reference: official-protocol EPE **1.3134 px** / D1 **8.154%** (3,802,797 px, 40 scenes, `disp_occ_0`, 1/256, pooled) + vendor-protocol headline **8.2237%** — both from the reference ONNX weights, NOT from any `.pth`.
- From-scratch references (separate protocols, not interchangeable): deterministic 6-block seed0 40-scene EPE 2.2955/D1 15.60; H2 seeds EPE 3.7–4.3/D1 24.8–29.9 (late-window); EXP-016 EPE 19.22/D1 91.1 (pipeline proof only).

### CURRENT MODEL BOTTLENECKS

- Numerics first: plain readout saturates (fixed by standardisation; cap ~0.77 and cost-inflation side-effects unverified for cost).
- Compute concentration: refinement = 90.6% MACs and ~⅔ of residual error (held-out E1); retrained 5-block penalty is real but small (+0.07–0.17 px EPE range separation); 4-block unresolved.
- Representation coarseness: 12 candidates × 16-px steps with 1/16 features and a linear downsampler; far-range error structure hypothesised, scene-class evidence absent.
- Regime confound: everything from-scratch rests on 160 scenes/random-init; the reference's 1.31 px remains far above every from-scratch number, so regime-vs-architecture is unpartitioned.

### DATA-EXPERIMENT GAPS

- No Scene Flow pretraining (blocked on download, ~22 h estimate); no Middlebury/scene-class set; no occlusion masks; no multi-view/multi-baseline data (the stated prerequisite for any correspondence attribution); no second seed on the factorial `none`+std arm; no latency measurement on from-scratch models; no E4-cheap run (minutes, unrun); no matched-budget H2 epoch-10 control (known missing).

### CONTRADICTIONS OR PREVIOUSLY MISSTATED CLAIMS

- "8.223 EPE" (vendor label) — actually D1%; "concat volume" (README) — actually subtraction in the artifact; "validation improves" (EXP-016 original) — withdrawn; "95.18%" (EXP-017) — withdrawn; "four blocks match six" (E3b-era) — superseded; "reducible-to-3" (E3c) — contradicted by O6; "H1 measured the cost volume's worth" — narrowed (unlearnable pathway); "~2-point materiality band" — retired (nondeterministic-harness heuristic); any HEF-silicon reading of compiler numbers — never valid.

### WHAT WE SHOULD NOT TOUCH YET

- The frozen Phase 1 tree (`src/`, `scripts/`, `experiments/EXP-*`, `reference/`) — read-only; Phase 2 variants stay additive (`phase2/models/`, `phase2/scripts/`).
- Stage E (cost-volume optimisation: candidate count, volume representation, hierarchical search) — formally CLOSED; only E4-cheap (minutes, on trained features) is a live rationale and it is unrun, not approved-by-default.
- Correspondence-identification variants (permutations, translated pairs, zone/field statistics, new paired nulls) — scientifically closed pending NEW DATA (post-closure Phase 10-D); reopening needs data where correspondence varies independently of content.
- Any claim requiring a p-value, a causal block-count story, a latency-from-MACs inference, or a transfer beyond the tested configuration.

### DECISION REQUIRED BEFORE MODEL DEVELOPMENT

No redesign is proposed here. These are the 3–5 highest-value questions to answer before changing the architecture:

1. **What is the verified evaluation contract for the next change?** Freeze (in writing) the single primary protocol (40-scene official vs late-window), the pixel set, GT scale, and the no-mixing rule — because every baseline number changes with these choices and past errors came from mixing them.
2. **What is the train-to-reference gap made of (regime vs architecture)?** Before editing the network, decide whether the next run is a data-regime run (Scene Flow/O6-scale or equivalent, with matched-budget controls) or an architecture run — the current 1.3-vs-2.3–4.3 px gap is unpartitioned and dictates different next steps.
3. **What are the acceptance gates for the next variant (C1–C10 equivalents)?** Pre-register entropy floor, right-image/matching-map dependence thresholds, gradient-health rules, determinism controls, and the seed/noise scale that makes a D1/EPE delta "material" — the campaign's two costliest errors were uncalibrated gates.
4. **Is the readout (cap, temperature, cost-scale control) or the refinement capacity the binding constraint?** E2/E5 (readout-as-training-variable) and a retrained-capacity follow-up compete for the same budget; the cheapest falsification order (E4-cheap minutes → E2b/E3b-stage-2 scale) must be chosen before any structural edit.
5. **What new data, if any, will be admitted?** State explicitly whether model development proceeds on the 160-scene slice alone (with its ceiling disclaimers) or waits on Scene Flow/Middlebury/multi-view acquisition — because data scope determines which questions are even answerable.

---

## Inspection method and limits

**Commands used (all read-only):** directory listings (`read` on directories), file reads (`read` on `src/models/stereonet/*.py`, `src/datasets/kitti2015.py`, `src/evaluation/metrics.py`, `src/losses/disparity.py`, `scripts/exp_train_convergence.py`, `phase2/models/scaled_regression.py`, `docs/*.md`, `phase2/docs/*.md`, `experiments/EXP-*/metrics.json` + `CORRECTION.md`, `results/reproduction/variants.json`, `results/training/history.json`, `results/onnx_inspection/summary.json` + `report.txt` head, `reference/manifest.json` + `MANIFEST.md` + `stereonet.yaml`), glob/file-size listings (`Get-ChildItem` for sizes, checkpoint/ONNX/HEF inventories), `git log --oneline -15` and `git status --short`.

**Questions that could not be answered without execution (all marked NOT VERIFIED above):** does any `.pth` load successfully; do any checkpoint weights match the current source architecture; does the ONNX load/infer in this environment; what the HEF contains or scores; local `data/` completeness; current test-suite pass state; any numeric re-verification of recorded metrics.

**Files unable to inspect:** `.audit_pytest/` (permission denied on listing); binary contents of `*.onnx`/`*.hef`/`*.pth`/`*.npz`/archives/`upstream/extracted/` (not opened per prohibitions); `data/` image/GT bytes beyond loader-code reading; `phase2/visualizations/` image bytes (listing only); any file requiring model-loading to interpret (tensors, weights). Prior conversation summaries and expected-textbook StereoNet descriptions were treated as claims only, never as evidence, per the brief.

---

## Self-review (adversarial pass, 2026-09-14)

Method: read-only re-read of the cited sources. No file other than this one was modified; no model/checkpoint/ONNX/HEF/weight file was loaded; nothing was executed beyond listing and reading files.

### 1. Citation integrity — sampled ~65, 1 wrong (fixed)

Sampled, at minimum as required: Section 3 feature-extractor (`feature_extractor.py:26–56` + `blocks.py:28–48`), aggregation (`aggregation.py:25–40` + `blocks.py:22–25` + docstring `8–14`), readout (`regression.py:33–61`), refinement (`refinement.py:39–56` + `stereonet.py:98–111`); every `metrics.json` range in §§8–10/12 (EXP-005 `:24–42`/`:43–61`/`:62–80`/`:81–99`; EXP-007 `:10–25`/`:26–108`; EXP-010 `:9–274`/`:276–408`/line 417; EXP-016 `:9–18`/`:151–159`/line 180 + `CORRECTION.md:6–18`); the `PHASE_2_REGISTRY.md` ranges (`:33–36` reuse statement, `:906–921` weight SHA, `:918–921` 40-scene result); plus `summary.json` (`:11–45` I/O, `:46–70` nodes/inventory, `:67–70` params, `:47–63` operators), `stereonet.yaml` (`:1–50`, `:36–37`, `:43–44`), `manifest.json` (`:22–27`, `:29–34`), `kitti2015.py`, `metrics.py`, `disparity.py`, `exp_train_convergence.py`, `onnx_weights.py`, `scaled_regression.py`, `docs/stereonet_architecture.md`, `docs/phase_1_closure_audit.md`, `docs/reproduction_report.md`, `docs/cost_volume_analysis.md`, `results/reproduction/variants.json`.
Result: 64 of ~65 supported their sentences exactly (several verified line-for-line, e.g. EXP-005 variant blocks, EXP-016 epoch 0/19, EXP-010 per-scene block `276–407` + aggregate line 408, registry SHA block). 1 was wrong:
- §2.2 cited `docs/phase_1_closure_audit.md:59–74` as evidence that the `scripts/exp_*` files each back a named experiment. Lines 59–74 are the "Fixes Performed" table, not experiment backing. FIXED to `docs/phase_1_closure_audit.md:237–248` (18 contiguous experiment records with configs) alongside `docs/stereonet_architecture.md:11–17` (reproduce commands). No other citation was wrong; ranges off by at most a blank/closing line (EXP-007 `:26–108` ends at line 107 + close; EXP-010 `:276–408` includes the aggregate flag line 408) were left as-is as immaterial.

### 2. Inference leaking into fact — 2 instances, both fixed

Searched for `intended`, `meant to`, `should have`, `clearly`, `obviously`, `in effect`: only 2 hits, both the known `shift="left"` case (§3.2 lines 97/100). The upstream-author intent IS stated in repo lines (`cost_volume.py:19–21` "The intended behaviour is available as `shift=\"left\"`", `:37–38` "what the construction is meant to do", `:72–75` "`\"left\"` performs the shift the construction was intended to perform"), so the fix is citation, not demotion: line 97 now cites `:19–21` + `:34–38` + `:72–75`; line 100 now explicitly says "the behavior the source files describe as intended" with the same citations. No `clearly`/`obviously`/`should have`/`in effect` instances exist. No other fact-worded inference was found.

### 3. Status-label discipline — 3 demotions

- §11 item 6 (feature quality): SUPPORTED → POSSIBLE. Reason: single H2 mechanism record, single recipe — one independent source, fails the multi-source SUPPORTED bar.
- §11 item 8 future-risk clause (temperature re-collapse): SUPPORTED → POSSIBLE. Reason: E2 inference-only sweep alone — one independent source. The CONFIRMED (plain-readout saturation) and FIXED (H2 standardisation) clauses are untouched; they rest on H1/H2 measurements.
- §12 A.3 row (candidate-varying materiality, 6.08 px gap): SUPPORTED → POSSIBLE. Reason: one seed per arm, replication pending — one independent run; effect size is noted but does not meet SUPPORTED.
Audited and KEPT: §11 items 1 (code + EXP-010 measurement), 3 (geometry + `gt_range_ceiling` measurement), 4/5 (code + ONNX node walk), 9/10 (recorded numerics/measurements with UNKNOWN cost/harm), 11 (EXP-001 cost + E1 error-location, two sources), 12 (E3 3-seed + E3b/c + 3×3, multiple), 15 (regime description from code/config), 17 (EXP-015 int8 measurement kept as SUPPORTED: `precision_comparison.json` + EXP-015 record are two artifacts of one run — borderline but a recorded measurement, not reasoning; flagged here rather than demoted); §12 CONFIRMED rows (all rest on code or recorded `metrics.json`/artifact reads, not reasoning), H1 plain-readout SUPPORTED (3-probe chain: entropy + const-11.0 + gradient counts — three independent probes within one recipe, kept with the single-recipe ceiling stated), H2 SUPPORTED (3 seeds), 5-block SUPPORTED (n=3/arm, 9/9 with D1 exception recorded).

### 4. Execution-dependent claims — provenance labelled

- Parameter count + per-occurrence count: (a) read from pre-existing `results/onnx_inspection/summary.json:67–70` + architecture-doc §8 + closure Finding 3 — now tagged `(a)` in §12 and §14. NOT measured here.
- Compute/GMAC/stage-split/latency/traffic figures: (a) read from pre-existing EXP-001/architecture-doc accounting — now tagged `(a)` in §14. NOT measured here.
- Tensor shapes at every §4 stage: (b) derived by hand from source (`stereonet.py:forward`, module definitions) — §4 now carries a provenance note stating this; labelled derived, not measured.
- ONNX operator inventory (168 nodes, 12 Sub, etc.): (a) read from pre-existing `summary.json:46–70` / `:47–63` — now tagged `(a)` in §13 item 2. NOT measured here.
- Checkpoints/HEF/ONNX loadability, weight↔source correspondence, HEF silicon behavior, `data/` completeness, test-suite pass state: (c) NOT VERIFIED — already so marked; unchanged.

### 5. Protocol hygiene — pass after 1 fix

Checked every occurrence of 8.223 / 8.2237 / 8.1544 / 9.3315 (incl. rounded 8.223 / 8.154 / 9.3315 / 1.3134 / 1.4422 forms): each is attached to its protocol everywhere (§§5, 8.2 table + flags, 9, 12, 13 items 1/6, 15 baselines). One violation found and fixed: §11 item 15 mixed "2.3–4.3 vs reference 1.31" across three protocols without labels. FIXED to name each: deterministic 6-block seed0 40-scene official-ish EPE 2.2955 vs H2 late-window 10-scene EPE 3.7–4.3 vs reference-ONNX official-protocol EPE 1.3134. No table row mixes protocols without labelling both.

### 6. Completeness against the brief (§§2–15) — no silent skips

All required sections 2–15 are present with their sub-questions answered: §2 inventory by class (source/generated/vendor/config/data/tests/records); §3 per-module trace incl. candidates, construction, shift/padding/slicing, slice-identity, before/after export, PyTorch behavior; §4 full forward pass with shapes; §5 A–E verdicts with letters on every "reproduced"; §6 per-checkpoint table + authoritative-checkpoint rule; §7 per-element IMPLEMENTED/CONFIGURED/ACTUALLY-USED + BN deviation + Phase 2 deltas; §8 protocols + one-row-per-protocol table + flags; §9 R / R-Hailo / T / Phase-2-separate baselines with no best-number selection; §10 Phase 1 table + Phase 2 verdicts with supersession notes; §11 17 weaknesses with labels; §12 fact/inference table; §13 13 contradictions; §14 architecture report; §15 the exact required subsections. Phase 2 per-arm epochs/seeds are given by pointer (registry + addenda A/B) rather than enumerated — that is the report's stated design (registry is authoritative), not a skip. Nothing else was skipped; no sub-question was left unanswered without a NOT VERIFIED + reason.

### 7. Correspondence over-claiming — no fault found

Searched all `correspond*` occurrences (ceiling line 5, weight-correspondence §§2.3/6/15, identification split §10, attribution prerequisite §15): none states or implies the deployed model lacks correspondence or does not use it. Degeneracy is worded only as candidate-invariance / "no disparity search" / "no candidate-varying information in the volume" / "single tensor replicated 12×", which the brief allows. EXP-010's "performs no disparity search" is a search claim, not a correspondence-absence claim. Level D ceiling is stated as binding in §1 and respected throughout (§§10, 15). No fix needed; none made.

### Could NOT be checked without violating the prohibitions

Whether any `.pth`/`.onnx`/`.hef` loads; whether checkpoint weights match source; whether recorded metrics reproduce; what the HEF contains or scores; local `data/` completeness; current test status; any tensor value or shape by execution. All are marked NOT VERIFIED with reasons in the report body (§§2.3, 2.5, 2.6, 5D, 6, 14, 15, method section) and were left that way.
