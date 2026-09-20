# Seed-1 Kaggle Environment-ConFound Control — Report

**Run dir:** `stage_b_armp/20260919T130843Z_seed1_kaggle_envcontrol/`
**Binding document:** `stage_b_armp/ENV_CONTROL_PREREGISTRATION.md` (read first, followed exactly, NOT modified)
**Local eval completed (UTC):** 2026-09-19T13:09:49Z

---

## 1. Purpose and status

This is the ENVIRONMENT-CONFOUND CONTROL for seed 2. It is NOT a new seed, NOT seed 3 or 4, and it does NOT replace the original local seed-1 result. Both stand side by side in the record: the local seed-1 replication (`stage_b_armp/20260919T012646Z_tier2_seed1/`) remains the seed-1 replication, and this Kaggle seed-1 rerun stands alongside it purely as the environment-confound control.

Seed 2 was trained on Kaggle (Tesla T4, torch 2.10.0+cu128, Python 3.12.13) while seeds 0 and 1 were trained locally (torch 2.7.0+cu128). That between-seed environment difference is NOT the intended independent variable; the intended independent variable is initialization (ARM-P versus random initialization under the original P2A recipe). This control re-ran SEED 1, BOTH ARMS, in the seed-2 Kaggle environment to measure how much of the observed delta moves when the same seed is trained in the two different environments.

The seed-1 Kaggle kernels had COMPLETED before this task began (both arms: seed 1, 200 epochs, integrity_guard all_ok, 397954 params, control best_epoch 170, armp best_epoch 170). This task performed the LOCAL frozen evaluation of those completed checkpoints and wrote this report. No training was run in this task.

## 2. Integrity

- `ENV_CONTROL_PREREGISTRATION.md` was NOT modified (90 lines, as written; no result values added to it).
- No prior result or run dir was modified: `stage_b_armp/20260919T012646Z_tier2_seed1/`, `stage_b_armp/20260919T092454Z_tier2_seed2/`, `stage_b_armp/20260919T035455Z_kaggle_seed2/`, and `phase2/` are untouched. Artefacts were COPIED (not moved) into the new dir; copy-identity verified (sha256 of every copied `.pth` identical to its source).
- `stage_a_diagnostics/scripts/d3_matching.py` MD5: `ab29a921c121f5e404014989532172a6` — MATCHES the required value.
- `git status` at report time: the working tree contains PRE-EXISTING uncommitted modifications to tracked files `.gitignore` and `src/models/stereonet/__init__.py`, `cost_volume.py`, `regression.py`, `stereonet.py` (5 files, +113/−15). These predate this task and were not made by it; this task modified no tracked file. `stage_b_armp/` is untracked, as before. Eval git_head recorded in `tier2_eval.json`: `58e8a19908ddbd35451652c61aef478f56b51ebd`.
- No research source, eval logic, masks, bins, preprocessing, or disparity conventions were modified. No True GWC was run; cost volume, feature grouping, correlation, aggregation, and refinement were not touched.

## 3. Run completion

From the Kaggle provenance records (`record.json` / `p2a_record.json` in the new dir, copied verbatim):

| arm | seed | init | epochs_completed | train_monitor_best_epoch | wall (record.json `training_duration_s`) | wall (`p2a_record.json` `wall_clock_s`) | params | integrity_guard |
|---|---|---|---|---|---|---|---|---|
| control | 1 | random | 200 | 170 | 4532.153188705444 | 4522.121097326279 | 397954 | all_ok |
| armp | 1 | `/kaggle/working/repo/checkpoints/armp_stage1_best.pth` (sha256 `3ae6fb3b…f1a29be7`) | 200 | 170 | 4597.421860218048 | 4587.418308496475 | 397954 | all_ok |

Kaggle training software (both arms, from `p2a_record.json` `software`): Python 3.12.13, torch 2.10.0+cu128, numpy 2.0.2, opencv 4.13.0, Linux, CUDA 12.8, GPU Tesla T4.
Local evaluation software (from new `tier2_eval.json` `metadata`): Python 3.12.9, torch 2.7.0+cu128, numpy 2.5.1, Windows-11, device cuda (NVIDIA GeForce RTX 4060 Laptop GPU).

Both arms completed the full 200 epochs, so the control is not invalid on completeness grounds.

## 4. Configuration identity

Diff of `control/p2a_record.json` config versus `armp/p2a_record.json` config in the new dir: every non-outcome field is IDENTICAL (experiment `EXP-P2A-SCALE-COVERAGE-001`, intervention, dataset/split/resolution/crop, seed 1, 200 epochs, Adam/cosine schedule, loss, augmentation, device, `scale_draw_sample`, `software`, `integrity_guard`, `epochs_run` 200, `best_epoch` 170). The ONLY differing config keys are the initialization fields:

- `config.initialisation`: control `PyTorch defaults, random (P2A recipe)` vs armp `strict load of /kaggle/working/repo/checkpoints/armp_stage1_best.pth`
- `config.init_record.source`: `random` vs `/kaggle/working/repo/checkpoints/armp_stage1_best.pth`
- `config.init_record.note` (control only: `PyTorch defaults, random (P2A recipe)`) vs `config.init_record.strict/sha256/n_keys/param_count` (armp only: strict true, `3ae6fb3b…f1a29be7`, 70 keys, 397954 params)

Remaining diffs are outcome fields, not configuration: `arm` label, `best_sha256`, `final_sha256`, `best_val_epe_10scene` (10-scene in-training monitor, NOT the frozen contract), per-epoch `history`, and `wall_clock_s`.

## 5. Frozen-contract verification

Evaluator: `scripts/eval_tier2.py` in the new dir — byte-identical to the local seed-1 version except the line-134 provenance LABEL naming the new dir (proven by difflib; the single diff hunk is that one line, shown in Section 11). Same evaluator as seeds 0, 1, and 2: the measurement layer is constant.

Contract required for every checkpoint: scenes=40, valid_pixels=3802797, gt_scale=256.0, gt_source=disp_occ_0, contract_match=true.

| checkpoint | scenes | valid_pixels | gt_scale | gt_source | contract_match |
|---|---|---|---|---|---|
| control/best | 40 | 3802797 | 256.0 | disp_occ_0 | true |
| control/final | 40 | 3802797 | 256.0 | disp_occ_0 | true |
| armp/best | 40 | 3802797 | 256.0 | disp_occ_0 | true |
| armp/final | 40 | 3802797 | 256.0 | disp_occ_0 | true |

The frozen contract holds EXACTLY on all four checkpoints. Per preregistration Section 6, the control is therefore not invalid on contract grounds.

## 6. Results

Frozen local evaluation (`tier2_eval.json` in the new dir). sha256 values below were recomputed by this task and match both the `.sha256` sidecars and `record.json`. CONTROL is RANDOM INITIALIZATION under the original P2A recipe (never "P2A checkpoint" initialization).

| checkpoint | EPE (full precision) | D1 (%) | valid px | scenes | best epoch | sha256 |
|---|---|---|---|---|---|---|
| CONTROL best | 1.4178492233455906 | 8.533324287360067 | 3802797 | 40 | 170 | `e23ee0824462afe7640167a9741022b9872f2e823b284cbaed46b7ba9f81778a` |
| CONTROL final | 1.4283060535886267 | 8.417961831778031 | 3802797 | 40 | 170 | `9a28c9a2b0c8f177e91695bd2586e9aff3a82d7eb9964bcf2a8048965775fc81` |
| ARM-P best | 1.1872901156020892 | 6.255553478137276 | 3802797 | 40 | 170 | `40fa1296c1dcfc6fcc87a101390ea210e4409dc82ac4afb4645ea99fb10fc440` |
| ARM-P final | 1.2010106373630214 | 6.320374187736028 | 3802797 | 40 | 170 | `b0e7eb0124450c69921f4cdd4b6fb22d52a6c96e1544ddc340fd9b9bc5437342` |

## 7. Deltas and environment shifts (full precision)

Definitions per preregistration Section 3. `delta_local_best = -0.2528074`, `delta_local_final = -0.2499701` (fixed references).

| quantity | value (full precision) |
|---|---|
| delta_kaggle_best (ARMP_kaggle_best − CONTROL_kaggle_best) | -0.2305591077435014 |
| delta_kaggle_final (ARMP_kaggle_final − CONTROL_kaggle_final) | -0.22729541622560534 |
| environment_shift_best (delta_kaggle_best − (−0.2528074)) | 0.022248292256498603 |
| environment_shift_final (delta_kaggle_final − (−0.2499701)) | 0.02267468377439466 |

Per-arm absolute EPE changes between environments (reported SEPARATELY, per preregistration):

| arm / selection | kaggle − local (full precision) |
|---|---|
| CONTROL best (kaggle_best − 1.4440242) | -0.026174976654409488 |
| CONTROL final (kaggle_final − 1.4517077) | -0.023401646411373322 |
| ARM-P best (kaggle_best − 1.1912168) | -0.0039266843979108845 |
| ARM-P final (kaggle_final − 1.2017376) | -0.0007269626369785787 |

The 0.0444 px figure is quoted here ONLY as the contextual historical run-to-run spread it is (preregistration Section 4) — never as a threshold or test, and no pass/fail verdict is derived from it. For further context, the already-observed cross-seed spreads are: CONTROL best-checkpoint values spanning about 0.1147 px, ARM-P best values spanning about 0.0145 px — the control is the noisier arm. The measured shifts (best 0.022248292256498603, final 0.02267468377439466) sit below the 0.0444 px contextual spread, and the shift comes almost entirely from the CONTROL arm (−0.026/−0.023) while the ARM-P arm barely moved (−0.004/−0.001) — consistent with the CONTROL arm's already-observed noisiness, read that way and not as evidence about ARM-P specifically.

## 8. Per-bin results (both arms, best AND final)

Pixel counts: `<64` 3627325 (0.9538571214818987), `64–96` 141320 (0.037162120407689396), `96–128` 33072 (0.008696756624137445), `>=128` 1080 (0.00028400148627444484) — identical across all four checkpoints, as the mask is frozen. The `>=128` bin has held exactly 1080 px at every seed and is too thin to weigh; reported for completeness only. A bin improvement does NOT establish a mechanism.

| bin | selection | CONTROL EPE | ARM-P EPE | per-bin delta (ARMP − CONTROL) |
|---|---|---|---|---|
| <64 | best | 1.2590730040105913 | 1.0571074947156627 | -0.20196550929492862 |
| <64 | final | 1.2318783717918467 | 1.0609030104444936 | -0.17097536134735303 |
| 64–96 | best | 3.270490606945296 | 2.75395646459011 | -0.5165341423551859 |
| 64–96 | final | 3.6636716654710315 | 2.845849664886781 | -0.8178220005842505 |
| 96–128 | best | 10.14486649928026 | 8.036056267130935 | -2.1088102321493256 |
| 96–128 | final | 12.380652221378405 | 8.759072826343512 | -3.6215793950348925 |
| >=128 | best | 25.026880815294053 | 23.697468545701767 | -1.3294122695922859 |
| >=128 | final | 33.27013951760751 | 25.096393719425908 | -8.173745798181603 |

ARM-P is ahead in ALL FOUR bins under BOTH selections. The `>=128` final delta (−8.17) rests on 1080 px and carries no weight.

## 9. The four distinct runs (kept separate — NO pooling, NO averaging, NO combined statistic, NO significance, NO confidence intervals)

Prior values quoted exactly as fixed:

- **Seed 0, local:** CONTROL best 1.5587716, ARM-P best 1.2057590, delta −0.3530126; CONTROL final 1.4809434, ARM-P final 1.2087613, delta −0.2721821.
- **Seed 1, local:** CONTROL best 1.4440242, ARM-P best 1.1912168, delta −0.2528074; CONTROL final 1.4517077, ARM-P final 1.2017376, delta −0.2499701.
- **Seed 2, Kaggle:** CONTROL best 1.4932978, ARM-P best 1.1996447, delta −0.2936531; CONTROL final 1.4850973, ARM-P final 1.1963930, delta −0.2887042.
- **Seed 1, Kaggle rerun (THIS control):** CONTROL best 1.4178492233455906, ARM-P best 1.1872901156020892, delta −0.2305591077435014; CONTROL final 1.4283060535886267, ARM-P final 1.2010106373630214, delta −0.22729541622560534.

ARM-P is ahead of CONTROL in every one of the four runs under both selections, descriptively. Each run's delta stands on its own.

## 10. Which preregistered branch the measured shift selects, and why

The measured shifts are environment_shift_best = 0.022248292256498603 and environment_shift_final = 0.02267468377439466. Both are small relative to the context spreads quoted in preregistration Section 4 (the 0.0444 px historical spread, and the ~0.1147 px CONTROL cross-seed spread). The branch therefore selected is:

**Small shift relative to the quoted context spreads** — the environment-confound concern on seed 2 is substantially reduced, and the three seeds stand as they are. This is not proof that the environment has no effect — it is only the statement that this control did not reveal a material one.

This branch follows from the NUMBER, not from preference: had the measured shift been material, the "Material shift" branch would have applied instead (seed 2 carrying an explicitly attached environment caveat, never silently dropped); had the contract failed or either arm not completed 200 epochs, the "Invalid control" branch would have applied. No fourth branch was invented. Under this branch, no next experiment is launched automatically; any follow-up is a separate decision recorded separately.

## 11. Deviations

- None from the evaluation protocol: same frozen evaluator (difflib shows exactly one changed line — the line-134 provenance LABEL):
  - `- rec = {"checkpoint": "stage_b_armp/20260919T012646Z_tier2_seed1/%s/p2a_%s.pth" % (arm, tag),`
  - `+ rec = {"checkpoint": "stage_b_armp/20260919T130843Z_seed1_kaggle_envcontrol/%s/p2a_%s.pth" % (arm, tag),`
- Observation (not a deviation by this task): the local working tree contains pre-existing uncommitted modifications to `src/models/stereonet/*` (see Section 2). The evaluation ran against that working tree, as did the seed-2 local evaluation; the evaluator script itself is unchanged apart from the label above.
- No further experiment or seed was launched. Scratch scripts were kept in the system Temp dir.
