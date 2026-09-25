# Stage F follow-up — L1 continuation pilot launch note

- ID: L1-LAUNCH
- Date: 2026-09-25
- Status: pre-registration, written BEFORE any L1 training. No L1 training
  has been launched under the parent design or under this note.
- Parent (frozen, DO NOT EDIT): `docs/superpowers/specs/2026-09-24-stage-f-followup-design.md`
  (§8 L1 gate, §9 kill conditions). Amendments A1 (M1 lambda arm) and A2
  (twin preflight + fresh control) change nothing about §8.
- Approved by the user 2026-09-25 before launch.

Statement tags: [FACT] (recorded artefact), [MEASUREMENT] (number from a frozen
run), [INFERENCE] (conclusion drawn from measurements), [VERDICT] (frozen
decision).

## 1. Frozen design being executed (§8, verbatim)

[VERDICT] L1 pilot is eligible only if the §3 train-split EPE mass carried by
e < 1 px ≥ 0.1676 px (absolute mass floor = the gap; no ratio-only rule). If
eligible: continue from E3 seed0 final, 40 epochs, fresh Adam lr 1e-4
cosine→0, two arms (Smooth-L1 control vs masked L1), same seed/data/aug. Gate:
Δfinal ≤ −0.030 px.

## 2. Eligibility (already met, recorded)

[MEASUREMENT] `stage_f/followup/audit2/audit2.json`
`subjects.train_seed0_final.l1_eligibility.sub1_mass` = 0.31114592140183084 px
≥ 0.1676 px. [VERDICT] The L1 pilot is eligible; no new gate is introduced
by this note.

## 3. What is launched (tooling only so far; no training yet)

[FACT] Two arms, seed 0 only, 40 epochs each, on Kaggle T4:

- l1c: Smooth-L1 control (the E0 loss, unchanged).
- l1m: masked L1 (same valid mask / max-disparity as masked_smooth_l1,
  plain L1 instead of Smooth-L1; bundle-local definition inside the bundle's
  training script only — repo `src/` untouched).

[FACT] Common continuation recipe (both arms): init = E3 seed0 FINAL
`stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_final.pth`, sha256
`82e58bc441a4382ec449479fe26bf479f6b45e9ace4d79b530abeeb40ea79c6d`
(verified: local file hash matches; both bundles' guard gates carry this
sha); 40 epochs; FRESH Adam lr 1e-4 (not a resumed optimizer); cosine
annealing to 0 over 40 epochs (T_max=EPOCHS=40); same seed 0, data, aug,
batch 2, crop; params 397954 (same architecture).

[FACT] Bundles (built by `stage_f/followup/kaggle/build_bundle_l1.py`,
both integrity PASS, minimal diff):

- `bundle_l1c/` — marker `stage-f-l1c`; finetune sha256
  `ef23559028ab26f4596797a9bf8d45f7dafe9c9844ea077a0b0ecdf0b2deb31a`.
- `bundle_l1m/` — marker `stage-f-l1m`; finetune sha256
  `8e1c53f33b8bb3aaf0c2f747303b80d0ce555156c05ec675a22e568fdd0754ed`.
- Shared `run_arm.py` sha256
  `665f68cc5c54b14ddf34ffc83f9d0c380bc9605c4954a7123ea93a9a88731b79`
  (identical in both bundles).
- Per-bundle diff vs the frozen E0 bundle is exactly
  `BUNDLE_MARKER.json`, `checkpoints/e3_seed0_final.pth` (added),
  `scripts/finetune_pilot.py`, `scripts/run_arm.py` (+ stale
  `dataset-metadata.json` removal, regenerated at upload).
- Inter-arm diff (bundle_l1c vs bundle_l1m) is ONLY the loss knob in
  `scripts/finetune_pilot.py` (F import, bundle-local `masked_l1`, one call
  line, record string) + `BUNDLE_MARKER.json`.
- Local smoke (`smoke_l1.py`, both arms SMOKE PASS): strict E3-final load,
  params 397954, finite loss with nonzero grads, LR schedule starts at
  1e-4, guard all_ok.

## 4. Declared environment sameness

[FACT] Both arms run on `vishnuvardhanksks` (kernels
`vishnuvardhanksks/stage-f-l1c-seed0` and
`vishnuvardhanksks/stage-f-l1m-seed0`; bundles
`vishnuvardhanksks/stage-f-l1c-bundle` and
`vishnuvardhanksks/stage-f-l1m-bundle`; data
`vishnuvardhanksks/kitti2015-tier2-seed2-subset`, already uploaded and
verified). Same account, no cross-account difference. Same GPU type (T4),
same continuation recipe/seed/epochs, same init sha. Per-arm bundle slugs
(because the two bundle contents differ by the loss knob and immutable
datasets are per-content, same convention as f1m/f1m10).

## 5. Gate and kills (restated verbatim, no new gates)

[VERDICT] Gate: PASS iff contract Δfinal (l1m − l1c) ≤ −0.030 px, else FAIL.

[VERDICT] Kill conditions (per §9, pilot-applicable): NaN/Inf loss or
metric, zero valid pixels, strict-load failure → STOP, no verdict. Pilot
numeric gates as above, frozen before launch. If L1 FAILs (with capacity
WEAKENED and M1 FAIL) → no architecture proposal; reassess data / GT-noise
(brief §21).

## 6. Files (allowed outputs under this note, nothing else)

[FACT] New: this file; `stage_f/followup/kaggle/{build_bundle_l1.py,
bundle_l1c/, bundle_l1m/, source_integrity_l1c.json, source_integrity_l1m.json,
push_l1.py, kernel_l1c_seed0/, kernel_l1m_seed0/, smoke_l1.py, smoke_l1.json}`;
`stage_f/followup/{score_twin.py, twin_pilot/}` (twin scorer + selftest,
separate task). No existing file is edited. No Kaggle call is made by this
note (bundle/kernel/smoke tooling only; upload and push are explicit manager
commands listed in the worker's report).

(End of file)
