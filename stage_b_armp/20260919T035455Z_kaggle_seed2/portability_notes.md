# Portability notes — ARM-P Tier-2 seed-2 Kaggle bundle

Research source (`scripts/*.py`, `src/*`, `phase1/harness/*`) is byte-identical
to the seed-1 originals except the two declared `run_arm.py` seed-2 edits
(see `source_integrity.json`). Every portability accommodation lives in
`bundle/kaggle_bootstrap.py` (stdlib only) and is listed below with a reason.
No research file was edited to make Kaggle work.

## Full transitive import closure (AST-walked, not guessed)

From the four scripts (`finetune_pilot.py`, `run_pilot.py`, `run_arm.py`,
`eval_tier2.py`), following absolute `src.*` / `phase1.*` imports and the
relative imports inside the stereonet package:

- `phase1.harness.determinism` (seed_all, make_generator, worker_init_fn)
- `phase1.harness.frozen_eval` (pooled_metrics, refuse_unless_contract,
  sha256_file, strict_load_report, weight_sha)
- `src.datasets.kitti2015` (Kitti2015Stereo, normalize, pad_and_crop)
- `src.evaluation.metrics` (disparity_metrics)
- `src.losses.disparity` (masked_smooth_l1)
- `src.models.stereonet` + submodules: `__init__`, `aggregation`, `blocks`,
  `cost_volume`, `excitation`, `feature_extractor`, `refinement`,
  `regression`, `stereonet` (StereoNet, StereoNetConfig)
- package markers: `src/__init__.py`, `src/models/__init__.py`,
  `src/datasets/__init__.py`, `src/evaluation/__init__.py`,
  `src/losses/__init__.py`
  (`phase1/` has no `__init__.py` in the repo; it imports as a namespace
  package both locally and on Kaggle.)

Third-party packages imported by the closure: `torch`, `numpy`, `cv2`
(opencv), `PIL` (only inside the smoke test's audit, not the closure).
`torchvision` is NOT imported by any closure module; the smoke test reports
it as "if used".

## Deliberate exclusions (with reason)

- `src/models/stereonet/onnx_weights.py` — NOT in the import closure (nothing
  imports it); including it would drag in the `onnx` dependency on Kaggle
  for zero benefit.
- `src/datasets/pfm.py` — only imported by `src/datasets/driving.py`
  (SceneFlow/Driving loader), which is outside the closure. Excluded.
- `src/datasets/driving.py`, `src/common/*`, `src/geometry/*`,
  `src/profiling/*`, `src/deployment/*` — outside the closure. Excluded.
- `phase1/harness/{bottleneck_diag,crossproto_eval,data_inventory,
  repro_probe,rescore,timing_probe,zip_audit,zip_detail}.py` — outside the
  closure. Excluded.

## Accommodations (all in `kaggle_bootstrap.py`)

1. **Tree re-assembly (`assemble()`)**: `finetune_pilot.py` computes
   `REPO_ROOT = parents[3]` of `scripts/finetune_pilot.py` and reads
   `REPO_ROOT/data/kitti2015`. The bundle dataset mount is read-only and
   flat, so the wrapper copies `src/`, `phase1/`, `scripts/`,
   `checkpoints/`, `configs/` to `/kaggle/working/repo/` and runs the
   scripts with `cwd=<repo>`, restoring the expected layout. Reason:
   scripts must stay byte-identical, so the layout is recreated around
   them instead of editing their path logic.
2. **Data symlink (`data/kitti2015 -> <dataset-A mount>`)**: the KITTI
   subset lives in a separate dataset mount. The wrapper symlinks it into
   the assembled tree. No copy fallback (a ~363 MB copy would duplicate
   the single source of truth and risk divergence). Reason: read-only
   dataset mounts cannot be rearranged; symlink preserves the
   `REPO_ROOT/data/kitti2015` contract.
3. **Auto-discovery (`find_bundle_root`, `find_data_root`)**: Kaggle mount
   paths contain the dataset slugs, which are only known after upload.
   The wrapper locates mounts by marker (`BUNDLE_MARKER.json`) and by
   `training/image_2`, overridable via `KAGGLE_BUNDLE_ROOT` /
   `KAGGLE_DATA_ROOT`. Reason: avoids hardcoding slugs into code.
4. **Unpinned `requirements.txt`**: Kaggle GPU images ship their own
   torch/CUDA/cuDNN; pinning would break the kernel. Exact versions are
   recorded by the smoke test (P3) instead. Reason: portability over
   false reproducibility.
5. **`run_script()` fresh-process helper**: mirrors `run_arm.py`'s own
   fresh-subprocess discipline for any future Phase-8 use. Unused by the
   smoke test. Reason: documents the intended invocation without
   touching the orchestrator.
6. **`eval_tier2.py` checkpoint-label string**: it embeds the seed-1 pilot
   path as a *record label only* (never used to open a file). Left
   byte-identical per the integrity rule; the label staleness is cosmetic
   and declared here. Reason: only `run_arm.py` may differ.

7. **Flat-mount `training/` shim (`data_layout()`, `assemble()`,
   `find_data_root()`)**: dataset A
   (`vishnu3727/kitti2015-tier2-seed2-subset`) was auto-extracted by Kaggle
   WITHOUT the top-level `training/` dir — verified 2026-09-19: 600 files,
   prefixes exactly `{disp_occ_0: 200, image_2: 200, image_3: 200}`, no
   `training/` prefix (recorded in `upload_record.json`). `Kitti2015Stereo`
   was NOT edited (research source stays byte-identical). Instead, when the
   mount has the flat layout, the wrapper creates a real
   `data/kitti2015/` dir whose `training/` level symlinks `image_2`,
   `image_3`, `disp_occ_0` back into the read-only mount, so the mount stays
   the single source of truth and `Kitti2015Stereo` still sees
   `training/*`. `find_data_root()` (and the `KAGGLE_DATA_ROOT` override)
   accepts both layouts. Reason: re-uploading dataset A or editing the
   dataset class would violate the no-reupload / no-research-edit rules;
   the layout is reconstructed around the frozen code. (Verified locally
   with a stubbed `os.symlink` — Windows lacks symlink privilege; Kaggle
   Linux symlinks work.)

8. **Smoke bootstrap loader (`kaggle_smoke.py::main`)**: Kaggle script
   kernels execute ONLY `code_file` as `/kaggle/src/script.py` — sibling
   files are not present, so `import kaggle_bootstrap` failed with
   `ModuleNotFoundError` (kernel v1, status ERROR, evidence in
   `kaggle_output/`). The smoke test now loads the stdlib-only wrapper via
   `importlib` from the attached bundle dataset mount
   (`/kaggle/input/*/kaggle_bootstrap.py`, marker mount preferred),
   falling back to next-to-this-file for local runs. Reason: environment
   plumbing only; research code, preprocessing, architecture and the
   contract are untouched. Required a dataset-B v2 (same tree, fixed
   wrapper copies) and kernel v2.
9. **Recursive mount discovery (iteration 2, FINAL)**: kernel v2 still
   failed — `/kaggle/input/*/kaggle_bootstrap.py` matched nothing, so the
   mounts are either absent or nested deeper than one level. The smoke
   loader now searches `/kaggle/input` recursively (marker mount
   preferred) and records the input listing + candidates in the artefact;
   `find_bundle_root` / `find_data_root` search recursively the same way
   (shallowest/mount-root wins). Reason: same as 8 — environment plumbing
   only. Required dataset-B v4 and kernel v3. No further iterations.

11. **Missing architecture reference bundled (dataset B v6, FIX 1)**:
    `scripts/finetune_pilot.py:154` `integrity_guard()` loads
    `REPO_ROOT/phase1/runs/arm_v/arm_v_best.pth` as the architecture
    reference for its key/shape comparison. The seed-2 bundle (v1–v5)
    never contained it, so the real training entrypoint would have
    crashed with `FileNotFoundError` before the first optimizer step.
    The file is added to the bundle at exactly
    `phase1/runs/arm_v/arm_v_best.pth` (1,620,277 bytes, sha256
    f2dcf8a22e6f1711b67a70837066f84e68550d53bb4891d3b8d46bd50b452cc4,
    byte-identical to the repo copy). Adding a missing runtime data
    dependency is not a research-code change; no research file was
    touched. Required dataset-B v6 (`--dir-mode zip`, the layout that
    works); the sha256 is verified to survive the round trip.

12. **True nesting depth in `assemble()` (dataset B v6, FIX 2)**:
    `scripts/finetune_pilot.py:39` uses `parents[3]`, and `run_arm.py`
    uses `PILOT = parents[1]` / `REPO = PILOT.parents[1]`, so the
    scripts must live at `<root>/stage_b_armp/seed2/scripts/` for
    REPO_ROOT to resolve to `<root>`. The seed-2 wrapper (v1–v5)
    assembled a FLAT tree (`<root>/scripts/`, cwd=`<root>`), under
    which `parents[3]` resolves to `/kaggle` — wrong. `assemble()` now
    copies `src/`, `phase1/` (now including `runs/arm_v/`),
    `checkpoints/`, `configs/` directly under `<root>` and `scripts/`
    to `<root>/stage_b_armp/seed2/scripts/`, with an explicit
    fail-fast guard that `parents[3]` of the assembled
    `finetune_pilot.py` equals `<root>` (and the `run_arm.py`
    PILOT/REPO equivalent). `run_script()` now resolves the nested
    scripts dir (cwd stays `<root>`). Wrapper-side only —
    `finetune_pilot.py` was NOT edited to change the depth. The smoke
    test records the nesting contract (`result["nesting"]`) and its
    `source_hashes` keys moved to the nested script paths.

    Honest note: the earlier smoke test (v1–v4, PASS) never executed
    the `finetune_pilot.py` entrypoint — it audited imports, data,
    checkpoint and one manual batch, but never ran the script whose
    path logic and `integrity_guard` file these fixes concern. That is
    why both defects were missed, and why a real short entrypoint run
    (FIX 3, `kaggle_entrycheck.py`: `P2A_EPOCHS=2` through the actual
    script, NOT through `run_arm.py` which forces 200 epochs) now
    gates the training kernels.

10. **P6 single-sample batch axis (iteration 3, ONE-DEFECT FIX)**:
    kernel v3 failed only P6 with `ValueError: prediction and target must
    match: (1, 1, 368, 1232) vs (1, 368, 1232)`. Cause: the frozen recipe
    (`scripts/finetune_pilot.py:122`) returns `disp_c[None]` PER SAMPLE
    (channel axis) and the DataLoader supplies the batch axis, yielding
    `(B,1,H,W)`; the smoke test has no loader, so `s.disparity[None]`
    was exactly one axis short of the model output `(1,1,368,1232)`.
    Fix (wrapper only, `bundle/kaggle_smoke.py::phase6`): build the smoke
    target as `s.disparity[None, None]` so it matches what the DataLoader
    would produce. `left`/`right` verified already `(1,3,368,1232)` via
    `normalize(...)` (which returns `x.transpose(2,0,1)[None]`), so they
    needed no change — batch axis present exactly once. Reason: smoke
    harness correction to match the frozen recipe; recipe, loss, model,
    preprocessing and contract untouched. Required dataset-B v5 and
    kernel v4.

## Seed-2 edits (Kaggle copy of `run_arm.py` ONLY)

- Line 83: `env["P2A_SEED"] = "1"` -> `"2"` (real seed for this replication).
- Line 199: `"seed": 0` -> `"seed": 2` in the `record.json` payload
  (fixes the known stale literal documented in the seed-1 CORRECTIONS.json).
  Seed-0 and seed-1 copies untouched. Both blocks shown in
  `source_integrity.json`.
