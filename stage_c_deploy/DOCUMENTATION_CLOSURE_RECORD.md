# DOCUMENTATION CLOSURE RECORD — MICROCHIP STEREONET

**Date:** 2026-09-20. **Type:** documentation / navigation closure pass. **Git HEAD:**
`58e8a19908ddbd35451652c61aef478f56b51ebd` (branch `master`, unchanged; no commit made).

---

## 1. Purpose

Repair the documentation, navigation and stale-record hazards identified by the two whole-system
audits (`FINAL_WHOLE_SYSTEM_AUDIT.md`, `FINAL_WHOLE_SYSTEM_AUDIT_SECOND_PASS.md`), so that a
future engineer, reviewer or test team cannot misread the current project state.

The objective was **not** to make the research better. It was to make the repository accurately
communicate what is established, what failed, what is unknown, and what is blocked.

## 2. Scope

Five known hazards (F1–F5) plus the historical INT8 reader hazard. Documentation only. No
research question was reopened, and no new claim, number, threshold or experiment was created.

## 3. Freeze statement

**Nothing technical was modified.** During this pass there was:

no training · no fine-tuning · no new experiment · no additional seed · no C1 variant re-run ·
no DR-1 Variant B · no quantization · no Hailo compilation · no HEF creation · no change to any
model, checkpoint, ONNX, inference algorithm, C2/C2.1/C2.1.1 implementation, training code, loss,
augmentation, optimizer/scheduler, dataset, experiment configuration, evaluation script,
evaluation tolerance, experimental result, hash, or historical evidence file · no change to the
C1, DR-0, DR-1, identifiability or deployment-target conclusions · no deletion of anything.

Operations used: read, search, `git status/diff/log/rev-parse`, SHA-256 verification, and writes
confined to documentation/navigation files.

**No technical defect was discovered during this pass, so none was deferred.** The pre-existing
disclosed defects (stale `EXPECTED_STATE_KEYS = 72` / `EXPECTED_PARAMS = 423586` in
`phase1/harness/frozen_eval.py:49-50`; stale seed-1 pre-launch text and `seed: 0` literals; the
transposed ONNX hash spelling `…deed1dce26…`) were **left exactly as they are** — they belong to
frozen evidence and are already self-recorded in the run records and audits.

## 4. Files inspected

`README.md`, `CLAUDE.md`, `AUDIT_REPORT.md`, `PHASE_1_FINAL_REPORT.md`, `PHASE_1_OVERVIEW.md`,
`PHASE_2_FINAL_REPORT.md`; `phase0/docs/*`; `phase1/docs/*`, `phase1/results/LEADERBOARD.md`,
`phase1/harness/frozen_eval.py`; `phase2/docs/*`, `phase2/paper/*`, `phase2/deploy/*`;
`stage_a_diagnostics/*`; `stage_b_armp/*` (all 8 run directories + the 5 stage-level records);
`stage_c_deploy/*` (C0, C1, C1 §10, DR-0, DR-1, C2, C2.1, C2.1.1, target/toolchain, blocker,
readiness, both whole-system audits, all validation JSONs); `results/quantization/*`;
`reference/`; `docs/`; plus the git working tree.

Every path referenced by the new index was existence-checked before publication (45 paths, zero
missing).

## 5. F1 resolution — stale Phase-2 deployment wording

**Action:** a supersession notice was **prepended** to `PHASE_2_FINAL_REPORT.md`. The historical
report below it is unchanged — not one word of its body, tables or numbers was edited.

The notice states: (1) the document is historical; (2) its deployment wording ("Model Deployed",
"Deployed model — P2A seed 0") reflects the historical Phase-2 state, in which "deployed" meant
selected-and-exported, and the same report already capped itself at "Furthest verified deployment
layer: ONNX"; (3) it is **not** the current deployment status; (4) the current candidate is
**ARM-P seed 1** selected at C0 (`stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth`,
sha256 `b2f6f5d5…fffeb7454`), with P2A seed 0 retained as a historical selection and valid
experimental baseline; (5) Hailo deployment remains **BLOCKED** — target device unspecified,
toolchain not executed, no ARM-P HEF.

**Status: RESOLVED (documentation-only, historical content preserved).**

## 6. F2 resolution — dynamic-range classification

**Action:** three labels in `stage_c_deploy/STAGE_C_DEPLOYMENT_READINESS_FINAL.md` were
re-classified, each with the superseded wording preserved in place and dated:

- §6 classification line → **DEPLOYMENT RISK — TARGET-NATIVE BEHAVIOUR UNKNOWN**, with an
  explicit MEASURED / ESTABLISHED / UNKNOWN / TARGET breakdown.
- §14 risk register row R3 → same label, with the prior "FAIL (property)" quoted inline.
- §19 final-state line → same label, cross-referenced to §6.

The corrected statement, in full:

- **MEASURED** — ARM-P carries extremely large fp32 intermediate values (`aggregated_cost`
  max abs ≈ 1.57e18–2.53e18) against a reference operating at ≈1e1–1e2. VERIFIED (DR-0).
- **ESTABLISHED** — the DR-1 Variant-A activation rescale **FAILED H1**; that mitigation is
  rejected and was not adopted.
- **UNKNOWN** — behaviour under target-native Hailo quantization/compilation. No quantization
  was attempted; `DYNAMIC_RANGE_AUDIT.md` §15 predicts no outcome.
- **TARGET** — still unspecified.

No threshold was invented; none exists. Dynamic range is **not** claimed safe and **not** claimed
to be a confirmed Hailo failure. The measured findings in `DYNAMIC_RANGE_AUDIT.md` were not
touched.

**Status: RESOLVED (classification corrected, measurements untouched, prior wording preserved).**

## 7. F3 resolution — global results index

**Action:** created `RESULTS_INDEX.md` at the repository root. It gives **STATUS / SOURCE /
SHORT PURPOSE** for every required item: the frozen contract; Phase 0; Phase 1 (incl. the
leaderboard, explicitly marked historical and non-current); Phase 2 re-scoped; the superseded
Phase-2 correspondence line; Stage A; ARM-P Stage 1; the three Tier-2 seeds; the environment
control; Stage B closure; the three identifiability records; Stage C0; C1 parity; the C1 §10
export investigation; DR-0; DR-1; C2; C2.1; C2.1.1; the target/toolchain resolution; the blocker
report; the Stage-C readiness final; the first-pass whole-system audit; the second-pass audit;
and this closure record. It also carries the current deployment statement, the F5 deliverable
statuses, a 13-item reader-hazard list, and a pointer table of frozen artifact hashes.

`phase1/results/LEADERBOARD.md` was **not rewritten** and remains the historical Phase-1 arm
table.

**Status: RESOLVED (new index; no historical index altered; no new claim created).**

## 8. F4 resolution — README navigation

**Action:** a current entry-point section was **prepended** to `README.md`; the original
Phase-1 README is preserved verbatim below a rule that labels it historical. The diff is
**57 insertions, 0 deletions**.

The new section distinguishes: **HISTORICAL** (Phase 0, Phase 1, both Phase-2 lines) ·
**CLOSED TECHNICAL INVESTIGATION** (Stage A, Stage B/ARM-P, Stage C device-independent work) ·
**CURRENT BLOCKER** (target device unspecified, toolchain not executed, ARM-P HEF absent, plus
the two frozen failures and the three passing gates) · **CURRENT STATUS** (technical
investigation closed; documentation closure complete). It points at `RESULTS_INDEX.md` as the
start-here index and warns explicitly against reading the Phase-1 leaderboard as current.

**Status: RESOLVED (navigational, concise, no research content added).**

## 9. F5 resolution — unfinished documentation deliverables

Both items received an explicit status, recorded in the file itself and in `RESULTS_INDEX.md` §9.

**A. `stage_b_armp/STAGE_B_SYNTHESIS_OUTLINE.md` — outline COMPLETE; downstream manuscript
NOT REQUIRED FOR CURRENT PROJECT.**
Basis: the document declares itself an outline and evidence map, and is complete as such; Stage B
is closed and `stage_b_armp/ARMP_CLOSURE_RECORD.md` is its authoritative record, already carrying
the results, limitations, confounds and identifiability verdicts in full. A manuscript would add
presentation, not evidence. A dated status notice was prepended; the body is unchanged. No
research is authorized or required to produce the manuscript.

**B. `phase2/paper/` (`main.tex`, `refs.bib`, `README.md`) — TODO — DOCUMENTATION ONLY;
retained; does not block project closure.**
Basis: it belongs to the superseded Phase-2 correspondence line, closed at claim ceiling **Level
D** (genuine geometric correspondence NOT-DEMONSTRATED) by
`phase2/docs/POST_CLOSURE_RESEARCH_AUDIT.md`, whose own verdict was "stop experimentation and
write the paper". The draft carries 6 unresolved TODOs (2 UNVERIFIED facts, 1 UNVERIFIED
checkpoint hash, 3 missing citations) and intentional placeholders F1–F7, and has never been
compiled (no local TeX toolchain). A status header was prepended to `phase2/paper/README.md`
stating that completion is a writing task only and that **no new research, experiment,
measurement or number may be produced to complete it**. `main.tex` and `refs.bib` were not
touched.

**Status: RESOLVED (both statuses explicit; neither expanded into research).**

## 10. Historical INT8 reader-hazard resolution

**Action:** created `results/quantization/README.md`. `report.txt`,
`precision_comparison.json` and `artifacts/stereonet_int8.onnx` were **not modified and not
deleted**.

The note records that this directory is **Phase-1 EXP-015**, a host-side precision study of the
**reference model** (fp32 1.313 / fp16 1.310 / **int8 1.655** px, onnxruntime CPU, 2026-09-05),
and that it is **NOT ARM-P**, **NOT Hailo quantization**, **NOT a Hailo deployment result**, and
**NOT evidence about the ARM-P activation range**. It states the current status plainly:
**ARM-P quantization is NOT TESTED and is BLOCKED** on the unspecified target and the un-executed
toolchain. The same hazard is item 11 of the `RESULTS_INDEX.md` reader-hazard list.

**Status: RESOLVED (artifact preserved, mislabelling risk removed).**

## 11. Current deployment statement (single, unambiguous)

> **ARM-P seed 1 is the frozen deployment candidate selected under C0, but deployment to Hailo
> has NOT occurred because the target Hailo device has not been specified and the corresponding
> target toolchain has therefore not been executed.**

This wording now appears in `README.md` (current status), `RESULTS_INDEX.md` §0, and the F1
supersession notice. No document in the repository states or implies that Hailo-8, Hailo-10H or
Hailo-15H is the target; all such references remain labelled HISTORICAL REFERENCE.

## 12. C1 preserved

**Unchanged and unsoftened.** ARM-P ONNX vs PyTorch **max absolute difference =
0.001708984375 px** against the frozen tolerance **1e-3 px** → **C1 = FAIL**. The later export
investigation found **no export-level mitigation** within the tested space (5 variants,
bit-identical outputs). Decomposition preserved: Δinitial 7.143e-4 px (would pass alone) /
Δresidual 1.6899e-3 px (carries the FAIL) / Δfinal 1.708984375e-3 px; the residual causal split
remains **NOT IDENTIFIED / NOT TESTED**. Tolerance untouched. Source records untouched
(`ARMP_DEPLOYMENT_VALIDATION.json` mtime 09-19 21:22, unchanged).

## 13. DR-1 preserved

**Unchanged.** Variant A activation rescale (`aggregated_cost / 1e17`): pooled max disparity
difference **1.4816284e-02 px** against the 1e-3 px equivalence threshold → **DR-1 H1 = FAIL**.
40-scene global EPE delta −4.793704046157643e-06, D1 delta 0.0. **Variant B NOT RUN.** Source
record untouched (`dr1_rescale_gate.json` mtime 09-20 07:47, unchanged).

**C1 and DR-1 are kept as two separate, scoped failures.** No document merges them into a generic
"deployment failed" statement.

## 14. C2 / C2.1 / C2.1.1 preserved

**Unchanged.** C2 metric depth **135 checks, 0 failures**; C2.1 spatial perception **251 checks,
0 failures**; C2.1.1 discontinuity **19 checks, 0 failures**. Implementations and validation
records untouched (record mtimes 09-20 11:36 / 11:39 / 11:40, unchanged; `discontinuity.py`
sha256 `2e81e9f2…1b5743` re-verified).

Every place these are quoted now states that they validate the **deterministic,
device-independent perception stack** — they add **no neural model**, change the neural model
in no way, and validate **nothing about Hailo hardware**.

## 15. Target / toolchain status

| | |
|---|---|
| Target device | **UNKNOWN / NOT SPECIFIED** — no authoritative company requirement exists in the repository |
| Hailo toolchain | **NOT EXECUTED** — `hailortcli`, `hailo`, `dfc`, `hailomz` absent; all Hailo Python modules unimportable |
| ARM-P HEF | **ABSENT** |
| Reference HEF (`reference/stereonet.hef`) | **HISTORICAL** |
| Therefore Hailo deployment | **BLOCKED** |

No target was inferred from the historical Hailo-8 / 10H / 15H references, and the documentation
now says so explicitly in three places (`README.md`, `RESULTS_INDEX.md` §10 item 4, the F1
notice).

## 16. Files modified (documentation only — 5 files)

| File | Change | Diff shape |
|---|---|---|
| `README.md` | current entry-point section prepended; original Phase-1 README preserved below a "historical" rule | **+57 / −0** (pure insertion) |
| `PHASE_2_FINAL_REPORT.md` | supersession notice prepended; body untouched | pure insertion (untracked file) |
| `stage_c_deploy/STAGE_C_DEPLOYMENT_READINESS_FINAL.md` | 3 dynamic-range classification labels corrected, each preserving the superseded wording inline | 3 localized edits, no number changed |
| `stage_b_armp/STAGE_B_SYNTHESIS_OUTLINE.md` | status notice prepended; body untouched | pure insertion |
| `phase2/paper/README.md` | status header prepended; body untouched | pure insertion |

## 16b. Files created (3)

- `RESULTS_INDEX.md` — the current global results index (F3).
- `results/quantization/README.md` — historical INT8 labelling note (§10).
- `stage_c_deploy/DOCUMENTATION_CLOSURE_RECORD.md` — this record.

## 17. Files explicitly NOT modified

- **All model artifacts:** every `.pth`, every `.onnx`, `reference/stereonet.hef`,
  `results/quantization/artifacts/stereonet_int8.onnx`.
- **All source code:** `src/**` (the five pre-existing tracked modifications predate this pass
  and were not touched), `scripts/**`, `phase1/harness/**` including `frozen_eval.py`,
  `phase2/scripts/**`, `stage_c_deploy/export_armp*.py`, `stage_c_deploy/metric_depth/**`,
  `stage_c_deploy/spatial_perception/**` (incl. `discontinuity.py`), `tests/**`.
- **All experimental results and records:** every `tier2_eval.json`, `p2a_record.json`,
  `c2*_validation.json`, `ARMP_DEPLOYMENT_VALIDATION.json`, `ARMP_STATIC_EXPORT_VALIDATION.json`,
  `c1_parity_diagnostic.json`, `c1_section10_export_investigation.json`, `dr0_audit/**`,
  `dr1_rescale/**`, `activation_range.json`, `results/quantization/report.txt` and
  `precision_comparison.json`, `stage_a_diagnostics/*.json`, all `phase1/runs/**`.
- **All closure and audit records:** `stage_b_armp/ARMP_CLOSURE_RECORD.md`,
  `CANDIDATE_C_SEPARABILITY.md`, `GWC_IDENTIFIABILITY_DEFINITION.md`,
  `POST_IDENTIFIABILITY_BOUNDARY.md`, every Tier-2 run report, `stage_a_diagnostics/*.md`,
  `stage_c_deploy/C0_DECISION_RECORD.md`, `STAGE_C0_DEPLOYMENT_CONTRACT.md`,
  `C1_EXPORT_PARITY_REPORT.md`, `diagnostics/*.md`, `DYNAMIC_RANGE_AUDIT.md`,
  `DR1_RESCALE_GATE.md`, `C2_*.md`, `C1_TARGET_TOOLCHAIN_RESOLUTION.md`,
  `STAGE_C_DEPLOYMENT_BLOCKER_REPORT.md`, `FINAL_WHOLE_SYSTEM_AUDIT.md` (+ its JSON),
  `FINAL_WHOLE_SYSTEM_AUDIT_SECOND_PASS.md`.
- **Historical indexes:** `phase1/results/LEADERBOARD.md`, `PHASE_1_FINAL_REPORT.md`,
  `PHASE_1_OVERVIEW.md`, `docs/**`, `phase2/docs/**`, `phase2/paper/main.tex`,
  `phase2/paper/refs.bib`.
- **Nothing was deleted, renamed, archived-by-removal, or committed.**

## 18. Hash verification (before → after this pass)

| Artifact | SHA-256 | Result |
|---|---|---|
| ARM-P checkpoint (seed 1) | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` | **UNCHANGED** |
| ARM-P ONNX | `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989` | **UNCHANGED** |
| ARM-P Stage-1 pretrain checkpoint | `3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7` | **UNCHANGED** |
| P2A checkpoint | `0868ffd137a9985306bf5563685fd2362799bdd630d313e1181c980a7fbb6033` | **UNCHANGED** |
| Reference ONNX | `b1a01d855bb22663f06dfd29eae11194e6fde952a319673e036f157c3e10095c` | **UNCHANGED** |
| `discontinuity.py` | `2e81e9f2e97dbde196ba2311c68171aa4ce1195430b5c5b260d23b495e1b5743` | **UNCHANGED** |
| Historical int8 ONNX | `32cf8d4efdcd998a13f85af962b2305073d429c082a04ad77aef4e4d2bb09349` | **UNCHANGED** (recorded here for the first time; artifact untouched) |

## 19. Git verification

- HEAD before == HEAD after == `58e8a19908ddbd35451652c61aef478f56b51ebd`. No commit, no tag,
  no branch change, no revert, no clean.
- Tracked modifications before this pass: `.gitignore`, `src/models/stereonet/{__init__,
  cost_volume, regression, stereonet}.py` — 5 files, **+113 / −15**.
- Tracked modifications after this pass: the same 5 files with the **same +113 / −15**, plus
  **`README.md` at +57 / −0**. The source diff is byte-for-byte the pre-existing one; this pass
  added exactly one tracked-file change, and it is documentation.
- Untracked additions by this pass: `RESULTS_INDEX.md`, `results/quantization/README.md`. Edits
  to already-untracked documentation (`PHASE_2_FINAL_REPORT.md`, the readiness report, the
  Stage-B outline, the paper README) do not appear in `git diff` because those trees are
  untracked — their pure-insertion shape is recorded in §16 instead.
- **No new experiment directory, run directory, checkpoint, ONNX, log or results file appeared.**
  Verified: no new `.pth`, `.onnx`, `.hef`, `*_eval.json` or `runs/` entry exists.
- **No source-code modification occurred.**

## 20. Final documentation status

| | |
|---|---|
| Technical investigation | **CLOSED** |
| Documentation | **CLEAN / NAVIGABLE** |
| Deployment | **BLOCKED BY UNSPECIFIED TARGET DEVICE** |
| C1 | **KNOWN FAIL** (0.001708984375 px vs 1e-3 px) |
| DR-1 | **KNOWN FAIL** (H1; 1.4816284e-02 px) |
| C2 | **PASS** (135/135) |
| C2.1 | **PASS** (251/251) |
| C2.1.1 | **PASS** (19/19) |
| GWC | **OPEN BUT UNSUPPORTED / NOT IDENTIFIABLE on the frozen platform / NOT REFUTED** |
| Coverage vs pretraining | **NOT IDENTIFIABLE** |
| New research | **NONE — CONFIRMED** |

All five known documentation hazards are addressed; historical material is labelled HISTORICAL or
SUPERSEDED rather than erased; current navigation is unambiguous (`README.md` →
`RESULTS_INDEX.md`); this closure record exists; frozen technical artifacts are unchanged; git
and hash verification pass.

**STOP CONDITION MET — final documentation closure pass complete.**

---

*Documentation only. No experiment, no model change, no evidence altered or deleted.*
