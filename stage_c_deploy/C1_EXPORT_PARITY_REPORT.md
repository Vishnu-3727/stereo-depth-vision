# C1 — ARM-P SEED 1 EXPORT + NUMERICAL PARITY REPORT

Device-independent half of C1 only. No Hailo work of any kind was performed.

Evidence discipline (same as the other `stage_c_deploy/` documents): every
factual line carries exactly one of **VERIFIED** (established by a file in
this repo, with the path cited), **UNKNOWN** (not established, with what
would establish it stated), or **ASSUMED** (a working assumption with basis
and break condition stated). No number below was invented: every number comes
from a cited repo file. This worker has no network access.

---

## 0. Tolerances (predeclared, inherited — stated BEFORE any result)

- Parity: `max_abs_diff_px < 1e-3` on 5 scenes. **[VERIFIED: inherited, not
  invented — `phase2/scripts/export_p2a.py:106-107` (`"tolerance_px": 1e-3`,
  `"pass": bool(max(diffs) < 1e-3)`).]**
- P2A precedent for the static derivation: 3.05e-5 px max (static PyTorch
  graph vs trained PyTorch graph, 5 scenes). **[VERIFIED:
  `phase2/deploy/P2A_STATIC_EXPORT_VALIDATION.json`,
  `parity_static_vs_trained_pytorch.max_abs_diff_px = 3.0517578125e-05`.]**
- P2A precedent for deployed-ONNX vs PyTorch EPE: 1.4149796 (torch) vs
  1.4150748 (ONNX), a delta of 9.52e-5 px. **[VERIFIED: both numbers checked
  before quoting — torch 1.4149796 from
  `phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md` Section 3 ("PyTorch seed 0"
  column); ONNX 1.4150748458291988 from
  `phase2/deploy/P2A_DEPLOYMENT_VALIDATION.json`,
  `deployed_accuracy_onnx.metrics.epe` (rounds to 1.4150748); the delta
  1.4150748458291988 − 1.4149796 = 9.5246e-05 is arithmetic on the quoted
  values.]**

---

## 1. Scope

- This document covers the device-independent half of C1 only: export of the
  frozen ARM-P seed-1 checkpoint to ONNX (original + static variants) and the
  host-side numerical parity test. **[VERIFIED: declared in this report; the
  artifacts are `stage_c_deploy/armp_stereonet.onnx` and
  `stage_c_deploy/armp_stereonet_static.onnx`.]**
- The target Hailo device remains UNKNOWN; no Hailo parsing, compilation,
  quantisation, DFC install, or DeGirum access was performed or attempted.
  **[VERIFIED: `stage_c_deploy/C0_DECISION_RECORD.md` Section 4 (target
  UNKNOWN) and Section 9 (device-dependent work BLOCKED; export/parity
  eligible to run).]**
- No training, no retraining, no weight modification, no architecture change,
  no preprocessing change, no change to the frozen KITTI evaluation contract.
  **[VERIFIED: declared in this report; the recipe proof
  (`stage_c_deploy/c1_recipe_diff.json`, verdict "NO DEFECT") shows the only
  script changes are path constants, label strings, the `parents[1]` depth
  fix, and the declared additive parity fields; `ARMP_STATIC_EXPORT_VALIDATION.json`
  records `params_unchanged: true` (397,954 params, 70 state-dict keys).]**

---

## 2. Provenance

- Checkpoint: `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth`.
  **[VERIFIED: `stage_c_deploy/c0_checkpoint_inventory.json`, seed-1 entry.]**
- Checkpoint sha256 re-verified at the start of this run before any export:
  `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` —
  matches the frozen value, so the export proceeded. **[VERIFIED: measured at
  run start via sha256 over the file; matches `C0_DECISION_RECORD.md`
  Section 3 and both export JSONs' `source_sha256`.]**
- Frozen best EPE of this checkpoint (PyTorch, frozen contract): 1.1912168 px
  (full 1.191216765057325). **[VERIFIED: quoted from
  `stage_b_armp/ARMP_CLOSURE_RECORD.md` Section 5, not re-measured here.]**
- Export timestamps (UTC): original `2026-09-19T15:51:20.348872+00:00`,
  static `2026-09-19T15:53:19.012472+00:00`. **[VERIFIED:
  `stage_c_deploy/ARMP_DEPLOYMENT_VALIDATION.json` (`utc`) and
  `stage_c_deploy/ARMP_STATIC_EXPORT_VALIDATION.json` (`utc`).]**
- Git HEAD at export time: `58e8a19908ddbd35451652c61aef478f56b51ebd` (both
  runs). **[VERIFIED: both JSONs' `git_head`; matches the C0 record.]**
- Host platform: Windows-11-10.0.26200-SP0. **[VERIFIED: both JSONs'
  `software.platform`.]**
- Software: python 3.12.9, torch 2.7.0+cu128, onnx 1.22.0, onnxruntime 1.27.0,
  numpy 2.5.1. **[VERIFIED: both JSONs' `software` blocks.]**
- Exact commands run, in order (stdout/stderr captured verbatim to log files
  held outside the repo): **[VERIFIED: executed in this run.]**
  - `python stage_c_deploy/export_armp.py`
  - `python stage_c_deploy/export_armp_static.py`
- Both scripts exited 0; neither raised. **[VERIFIED: exit codes observed in
  this run.]** No export shim was needed and none was written. **[VERIFIED:
  no shim file exists under `stage_c_deploy/`; directory listing at close.]**
- Recipe proof: `stage_c_deploy/c1_recipe_diff.json` — 5 hunks (original
  pair) + 8 hunks (static pair), every hunk classified as one of
  `path_constant`, `label_string`, `repo_depth`, or
  `declared_additive_parity_fields`, verdict NO DEFECT. **[VERIFIED: the JSON
  file; mirror shas `749ee30e…07e207c85d51198` (`export_armp.py`) and
  `d019f8a9…306cedc1038b` (`export_armp_static.py`).]**

---

## 3. Export record

Method for the graph facts below: read-only probe of each ONNX file
(`onnx.load` + `onnx.checker.check_model` + node/op inventory + I/O
inspection), recorded at report time. **[VERIFIED: measurement method stated;
  values cross-checked against the export JSONs where they overlap
  (sha256, size, opset, node count, op-type set all match).]**

### 3a. Original export — `stage_c_deploy/armp_stereonet.onnx`

- Path, sha256, size: `stage_c_deploy/armp_stereonet.onnx`,
  `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989`,
  1,701,550 bytes. **[VERIFIED: `ARMP_DEPLOYMENT_VALIDATION.json`
  (`export` block) and report-time file probe.]**
- Opset 13, IR version 7, producer `pytorch 2.7.0`, `onnx.checker` pass.
  **[VERIFIED: JSON `export` block + probe.]**
- Node count: 712. **[VERIFIED: JSON + probe.]**
- Full operator inventory with counts (27 types): Add 20, Cast 25, Concat 26,
  Constant 313, ConstantOfShape 23, Conv 51, Div 2, Gather 2, LeakyRelu 40,
  Mul 3, Neg 1, Pad 23, Range 1, ReduceMean 3, ReduceProd 1, ReduceSum 1,
  Relu 1, Reshape 47, Resize 1, Shape 3, Slice 47, Softmax 1, Sqrt 1,
  Squeeze 1, Sub 27, Transpose 24, Unsqueeze 24. **[VERIFIED: probe; the
  op-type set matches the JSON `export.op_types`.]**
- Inputs: `left` [1, 3, 368, 1232] FLOAT; `right` [1, 3, 368, 1232] FLOAT.
  **[VERIFIED: probe; matches JSON `export.input_shape`.]**
- Output: `disparity` [1, 1, 368, 1232] FLOAT. **[VERIFIED: probe.]**
- Static vs dynamic dimensions: all I/O dimensions are concrete values; no
  symbolic (`dim_param`) dimension anywhere on the I/O. **[VERIFIED: probe
  (`dynamic_dims: []` on all three I/O).]** Intermediate value shapes are not
  declared (`value_info` empty), as in the P2A export. **[ASSUMED to match
  the P2A precedent — rests on: identical exporter path and the matching
  node/op inventory; breaks if: a node-level dump shows otherwise. A
  node-level dump was not performed — UNKNOWN at node granularity.]**
- Every export warning, verbatim (5 warnings, both scripts emitted the same
  set): **[VERIFIED: captured stdout/stderr of this run.]**
  - `TracerWarning: Converting a tensor to a Python boolean might cause the
    trace to be incorrect. ...` at
    `src/models/stereonet/cost_volume.py:94` (`if left.shape != right.shape:`)
  - `TracerWarning: Converting a tensor to a Python boolean might cause the
    trace to be incorrect. ...` at
    `src/models/stereonet/stereonet.py:156` (`if not return_stages:`)
  - `UserWarning: Constant folding - Only steps=1 can be constant folded for
    opset >= 10 onnx::Slice op. Constant folding not applied.` at
    `torch/onnx/_internal/jit_utils.py:309`
  - the same Slice constant-folding `UserWarning` at
    `torch/onnx/utils.py:691`
  - the same Slice constant-folding `UserWarning` at
    `torch/onnx/utils.py:1161`
- The warnings arise from unchanged model/exporter code paths, not from the
  mirror edits (the mirror edits touch no model code and no exporter call).
  **[VERIFIED: `c1_recipe_diff.json` shows no model/exporter-call change.]**

### 3b. Static export — `stage_c_deploy/armp_stereonet_static.onnx`

- Path, sha256, size: `stage_c_deploy/armp_stereonet_static.onnx`,
  `e275e86e4cdb340b64b3ccd065ccd8286b82c3ef1dbbd046acaaa1f0c363a5af`,
  1,699,043 bytes. **[VERIFIED: `ARMP_STATIC_EXPORT_VALIDATION.json`
  (`export` block) and probe.]**
- Opset 13, IR version 7, producer `pytorch 2.7.0`, `onnx.checker` pass.
  **[VERIFIED: JSON + probe.]**
- Node count: 689 (delta −23 vs the original export). **[VERIFIED: JSON
  (`dynamic_op_removal.original_n_nodes` 712, `static_n_nodes` 689) + probe.]**
- Full operator inventory with counts (23 types): Add 20, Cast 23, Concat 25,
  Constant 306, ConstantOfShape 23, Conv 51, Div 1, LeakyRelu 40, Mul 3,
  Neg 1, Pad 23, ReduceMean 1, ReduceSum 2, Relu 1, Reshape 46, Resize 1,
  Slice 46, Softmax 1, Sqrt 1, Squeeze 1, Sub 25, Transpose 24,
  Unsqueeze 24. **[VERIFIED: probe.]**
- Dropped vs original: Gather, Range, ReduceProd, Shape. Added: none.
  `ConstantOfShape` (23) remains. **[VERIFIED: JSON `dynamic_op_removal`
  (`ops_dropped`, `ops_added: []`, `present_in_static: ["ConstantOfShape"]`)
  + probe.]**
- Inputs / output / dtypes / static dims: identical to Section 3a (`left`,
  `right` [1, 3, 368, 1232] FLOAT → `disparity` [1, 1, 368, 1232] FLOAT, no
  symbolic dims). **[VERIFIED: probe.]**
- Export warnings: the same 5 warnings as Section 3a, verbatim (same source
  lines). **[VERIFIED: captured stdout/stderr of the static run.]**
- Weight identity of the static derivation: 397,954 params,
  `params_unchanged: true`, 70 state-dict keys, asserted bit-identical inside
  the script before exporting. **[VERIFIED: `ARMP_STATIC_EXPORT_VALIDATION.json`
  (`params`, `params_unchanged`, `state_dict_keys`).]**

---

## 4. Preprocessing / postprocessing

- The ONNX receives ImageNet-normalised NCHW input; normalisation is
  host-side and is NOT in the graph. **[VERIFIED: `src/datasets/kitti2015.py:155-165`
  (`normalize`: subtracts ImageNet mean [123.675, 116.28, 103.53], divides by
  std [58.395, 57.12, 57.375], returns NCHW) whose docstring states the
  exported ONNX contains no normalisation (`kitti2015.py:156-164`); both
  mirror scripts apply `normalize()` before every `sess.run`
  (`export_armp.py` parity/deployed-accuracy blocks,
  `export_armp_static.py` same blocks, copied verbatim per the recipe
  proof).]**
- The ONNX emits disparity in full-resolution pixels at the input geometry
  ([1, 1, 368, 1232]). **[VERIFIED: output shape from the probe (Section 3a);
  the model `forward` returns the refined full-size disparity
  (`src/models/stereonet/stereonet.py:136-154`); semantics match the P2A
  precedent (`phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md` Section 1:
  "`disparity`, [1, 1, 368, 1232], full-resolution pixels").]**

---

## 5. Parity result — FAIL

- Compared tensors: PyTorch model output `[0, 0]` vs ONNX Runtime output
  `[0, 0]`, both 368×1232 float maps over the same 5 frozen-contract scenes
  (`hailo_val` scenes 160–164: `000160_10.png` … `000164_10.png`), same
  normalised inputs both sides. **[VERIFIED: `stage_c_deploy/export_armp.py`
  parity block + `Kitti2015Stereo` scene ordering (first five `hailo_val`
  names, confirmed at report time).]**
- Per-scene table (original ONNX vs trained PyTorch): **[VERIFIED:
  `ARMP_DEPLOYMENT_VALIDATION.json` (`parity_vs_pytorch`).]**

  | scene | max abs diff (px) | mean abs diff (px) | max rel diff (guard >1.0 px) |
  |---|---|---|---|
  | 000160_10 | 8.3447e-04 | 4.4459e-06 | 7.1735e-05 |
  | 000161_10 | 1.4668e-03 | 6.3361e-06 | 1.0545e-04 |
  | 000162_10 | 8.9645e-04 | 6.1929e-06 | 4.6611e-05 |
  | 000163_10 | 5.7602e-04 | 4.8265e-06 | 2.4060e-05 |
  | 000164_10 | 1.7090e-03 | 1.3906e-05 | 4.6084e-05 |

- Pooled: max abs diff **1.708984375e-03 px** against the predeclared
  tolerance 1e-3 px → **FAIL**. **[VERIFIED: JSON
  (`max_abs_diff_px = 0.001708984375`, `tolerance_px = 0.001`,
  `pass: false`).]** Mean abs diff pooled 7.14e-06 px; max relative diff
  (guard: PyTorch reference > 1.0 px, recorded as `rel_guard_px`) 1.0545e-04.
  **[VERIFIED: JSON. The divergence is a localised max-abs spike (worst on
  scene 000164_10), not a broad shift — stated as an observation on the
  recorded mean/rel figures, not as a pass criterion.]**
- Static ONNX vs trained PyTorch: max abs diff 1.7242431640625e-03 px →
  **FAIL** the same tolerance. **[VERIFIED:
  `ARMP_STATIC_EXPORT_VALIDATION.json`
  (`parity_static_onnx_vs_trained_pytorch`: `max_abs_diff_px =
  0.0017242431640625`, `pass: false`).]**
- Localisation evidence (recorded, not a substitute verdict): static PyTorch
  graph vs trained PyTorch graph PASSES at 5.340576171875e-05 px max, and
  static ONNX vs original ONNX PASSES at 5.340576171875e-05 px max.
  **[VERIFIED: `ARMP_STATIC_EXPORT_VALIDATION.json`
  (`parity_static_vs_trained_pytorch.pass: true`,
  `parity_static_onnx_vs_original_onnx.pass: true`).]** The static rewrite is
  therefore faithful; the ~1.7e-03 px spike enters at the torch→ONNX export
  step for these weights (the P2A weights passed the identical recipe at
  6.18e-04 px). **[VERIFIED: comparison of the three recorded parity blocks
  above against `P2A_DEPLOYMENT_VALIDATION.json` (`parity_vs_pytorch`,
  6.1798e-04 px). Why the exporter noise is larger on the ARM-P weights is
  UNKNOWN — establishing it would require exporter-level investigation, which
  is outside C1.]**

**FAIL, stated plainly: the frozen ARM-P seed-1 checkpoint is NOT represented
faithfully in ONNX at the inherited 1e-3 px parity tolerance. Per the C1
specification the document stops here: no downstream number below is
presented as meaningful. Sections 6–10 (deployed ONNX accuracy, original-vs-
static structural comparison, host runtime, establishment claims, next gate
detail) are WITHHELD for this reason.**

**Status (no downstream numbers; status only): the device-independent half of
C1 is complete as a measurement and its verdict is FAIL on parity. Hailo
parse/compile remains BLOCKED on the target-device decision.
[VERIFIED: `stage_c_deploy/C0_DECISION_RECORD.md` Sections 4–5.]**

---

## Provenance of this report

- Written 2026-09-19 (UTC) after both exports completed; the only file created is
  `stage_c_deploy/C1_EXPORT_PARITY_REPORT.md` (this file). **[VERIFIED:
  declared in this report.]**
- No existing file was modified for C1: `git status` at close shows only the
  modifications that pre-existed this run (`.gitignore`,
  `src/models/stereonet/{__init__,cost_volume,regression,stereonet}.py`,
  recorded identical before the run started) plus the eight new
  `stage_c_deploy/` files
  (`export_armp.py`, `export_armp_static.py`, `armp_stereonet.onnx`,
  `armp_stereonet_static.onnx`, `ARMP_DEPLOYMENT_VALIDATION.json`,
  `ARMP_STATIC_EXPORT_VALIDATION.json`, `c1_recipe_diff.json`, this report).
  **[VERIFIED: `git status --short` compared before/after the run.]**
