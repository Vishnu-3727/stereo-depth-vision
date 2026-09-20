# FINAL WHOLE-SYSTEM AUDIT — SECOND-PASS ADDENDUM

**Status:** addendum to `stage_c_deploy/FINAL_WHOLE_SYSTEM_AUDIT.md` (first pass, written
2026-09-20 12:22 local). **This document adds; it repairs nothing and supersedes nothing.**
The first-pass report and its JSON companion were left byte-untouched.

**Audit date (UTC):** 2026-09-20. **Type:** independent second-pass closure audit, READ-ONLY.
**Host:** Windows 11 10.0.26200, python 3.12.9, torch 2.7.0+cu128, onnx 1.22.0,
onnxruntime 1.27.0, numpy 2.5.1. **Git HEAD:** `58e8a19908ddbd35451652c61aef478f56b51ebd`
(branch `master`).

**Freeze compliance.** No training, fine-tuning, export, quantization, compilation, evaluation
re-run, architecture/loss/augmentation/dataset/checkpoint/weight/ONNX/exporter change, no Hailo
step, no tolerance or evaluator change, and no cleanup occurred during this pass. Execution was
limited to: SHA-256 hashing, document and JSON reads, a live binary/module availability probe,
`find` over model artifacts, and `git status/diff/log/tag/rev-parse`. The only file created is
this one.

---

## 0. Why a second pass exists (provenance)

This auditor began an independent whole-system audit under the same termination-gate brief.
At the point of writing the deliverable, `stage_c_deploy/FINAL_WHOLE_SYSTEM_AUDIT.md` and
`stage_c_deploy/final_whole_system_audit.json` were found already present on disk, created at
**12:22 local on 2026-09-20 — during this pass's audit window**, and were **not** written by
this auditor. They are a concurrent auditor's work.

Disposition, per the project's no-overwrite rule and §31 of the brief: the first-pass files were
**not overwritten, not edited, not renamed and not deleted**. This addendum is filed beside them.

**Concurrency is itself recorded as an integrity observation:** two audits of the same gate ran
in overlapping windows against the same tree. Both were read-only, and the artifact hashes
measured after the overlap (§2 below) are unchanged, so no interference occurred.

**Verdict concurrence:** both passes independently reach
**PROJECT CLOSED WITH DOCUMENTATION DISCREPANCIES** (category B). The material findings agree on
every point checked in common: C1 FAIL preserved, DR-1 H1 FAIL preserved, C2/C2.1/C2.1.1 PASS,
target UNKNOWN, toolchain absent, identifiability states not upgraded, no unsupported current
claim. This addendum therefore records only (a) verification this pass performed independently
and (b) five discrepancies the first pass does not carry.

---

## 1. Relationship to the first pass

| | First pass | Second pass (this document) |
|---|---|---|
| Artifacts hashed | 5 | **7** (adds the Stage-1 pretrain checkpoint and the C1 BASE re-export variant) |
| Toolchain evidence | quoted from `evidence_summary.json` | **live probe on this host** (binaries + Python modules + `.hef`/`.onnx` enumeration) |
| C2-family counts | quoted | **counted** from the records' `tests` / `failures` arrays |
| Documentation discrepancies | 9 classified | **+5 new**, plus one reader hazard (§4) |
| Verdict | category B | category B (concurring) |

Nothing in the first pass is contradicted by this pass.

---

## 2. Independent verification performed this pass

### 2a. Artifact hashes — all measured, all matching

| Artifact | Path | Measured SHA-256 | Verdict |
|---|---|---|---|
| ARM-P checkpoint | `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth` | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` | MATCH |
| ARM-P ONNX | `stage_c_deploy/armp_stereonet.onnx` | `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989` | MATCH |
| P2A checkpoint | `phase2/runs/p2a_scale_coverage/p2a_best.pth` | `0868ffd137a9985306bf5563685fd2362799bdd630d313e1181c980a7fbb6033` | MATCH |
| Reference ONNX | `reference/onnx/stereonet.onnx` | `b1a01d855bb22663f06dfd29eae11194e6fde952a319673e036f157c3e10095c` | MATCH |
| Discontinuity module | `stage_c_deploy/spatial_perception/discontinuity.py` | `2e81e9f2e97dbde196ba2311c68171aa4ce1195430b5c5b260d23b495e1b5743` | MATCH |
| **Stage-1 pretrain checkpoint** (ARM-P's initialization source) | `stage_b_armp/20260918T062146Z_stage1_pretrain/checkpoints/armp_stage1_best.pth` | `3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7` | **MATCH** (vs `ARMP_CLOSURE_RECORD.md` §2) |
| **C1 BASE re-export variant** | `…/diagnostics/c1_export_variants/c1_variant_BASE_opset13_foldTrue.onnx` | `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989` | **BIT-IDENTICAL to the frozen ONNX** |

The last row independently confirms the §10 claim that an independent re-export reproduces the
frozen ONNX bytes exactly. No artifact was unavailable; no substitution was made.

### 2b. Gate records — counted, not quoted

| Gate | Record (utc) | `tests` length | `failures` length | `overall` |
|---|---|---|---|---|
| C2 metric depth | `metric_depth/out/c2_validation.json` (2026-09-20T06:06:49Z, cuda) | **135** | **0** | PASS |
| C2.1 spatial perception | `spatial_perception/out/c2_1_validation.json` (06:09:50Z, cuda) | **251** | **0** | PASS |
| C2.1.1 discontinuity | `spatial_perception/out/c2_1_1_discontinuity_validation.json` (06:10:03Z) | **19** | **0** | PASS |

All three records carry `checkpoint_sha256_before == after` and `onnx_sha256_before == after`;
C2.1.1 additionally carries `discontinuity_sha256_before == after == 2e81e9f2…1b5743`.

### 2c. Frozen failures — re-read from the machine records

- C1: `ARMP_DEPLOYMENT_VALIDATION.json` → `max_abs_diff_px = 0.001708984375`, `pass: false`;
  per-scene `[8.3447e-4, 1.4668e-3, 8.9645e-4, 5.7602e-4, 1.7090e-3]`. **FAIL** against the
  frozen 1e-3 px criterion. Preserved.
- C1 decomposition (§10 investigation): Δinitial **7.143020629882812e-04** (would pass alone) /
  Δresidual **1.689910888671875e-03** (carries the FAIL) / Δfinal **1.708984375e-03**
  (instrumented == frozen). The "soft-argmin is the entire cause" reading is therefore not
  supportable as the final causal claim, and the record does not make it. The residual
  input-sensitivity vs own-kernel split remains **NOT IDENTIFIED / NOT TESTED** (requires a
  model/inference change; out of scope).
- DR-1: `dr1_rescale_gate.json` → pooled max **1.4816284e-02 px** (all five scenes above the
  1e-3 rescale-equivalence threshold) → **H1 FAIL**; `dr1_eval_40scene.json` → 40-scene EPE
  delta **−4.793704046157643e-06**, D1 delta **0.0**, `original_reproduces_frozen: true`.
  Variant B **NOT RUN**.

### 2d. Toolchain and HEF — live probe on this host

- PATH: `hailortcli` **absent**, `hailo` **absent**, `dfc` **absent**, `hailomz` **absent**.
- Python modules `hailo_sdk_client`, `hailo_platform`, `hailort`, `hailo`, `hailo_model_zoo`,
  `hailo_sdk_common`: **all unimportable**. Control imports (`onnx`, `onnxruntime`, `torch`)
  succeed, so the probe is sound.
- `.hef` files in the tree: exactly one — `reference/stereonet.hef` (HISTORICAL reference
  artifact). **No ARM-P HEF exists.**
- Model graphs in the tree: reference ONNX, P2A pair, ARM-P pair, five C1 export variants,
  three instrumented DEBUG copies, and the historical reference int8 ONNX (§4). Nothing
  downstream of disparity loads a model — the single-neural-stage deployment graph holds.

Classification unchanged: Hailo compilation **NOT PERFORMED**, HEF **NOT PRODUCED**,
quantization **NOT TESTED**, latency/power **UNKNOWN** — untested because target and toolchain
are unavailable, not failed.

### 2e. Working tree

HEAD `58e8a19908ddbd35451652c61aef478f56b51ebd`; tags `phase-1`, `phase-1-final`,
`phase-1-final-r2`, `phase-1-frozen`. Pre-existing tracked modifications (**not** made by either
audit): `.gitignore` + `src/models/stereonet/{__init__,cost_volume,regression,stereonet}.py`,
5 files, +113/−15, mtimes 2026-09-08 … 2026-09-17 — all **before** the seed-1 Tier-2 run
(2026-09-19), matching the set recorded in the seed-0 audit and the seed-1/seed-2/env-control
reports. Provenance of those edits remains **unverified**, as those records state. Untracked
evidence tree pre-existing. Created by this pass: this file only. No commit, no revert, no
delete. `.audit_pytest/` unreadable (permission denied) — same limitation Phase 0 recorded.

---

## 3. Five findings not carried by the first pass

All are **reported, not repaired**. None changes a frozen verdict.

### F1 — `PHASE_2_FINAL_REPORT.md` still names P2A as the deployed model — **STALE / SUPERSEDED**

Locations: document title ("Optimization Closed, **Model Deployed**"), §1 table
("**Deployed model** — P2A seed 0"), §10 closing state ("Deployed model  P2A seed 0").

The document is internally honest — the same §1 table records "Furthest verified deployment
layer: **ONNX**", and §7 marks Hailo parser, compiler, HEF, HailoRT and hardware all
NOT ATTEMPTED / NOT AVAILABLE. But it is now superseded on two axes:

- **Candidate:** `stage_c_deploy/C0_DECISION_RECORD.md` §1–§3 selects **ARM-P seed 1**
  (`…/20260919T012646Z_tier2_seed1/armp/p2a_best.pth`, sha `b2f6f5d5…`) under the predeclared
  argmin rule. P2A is no longer the deployment candidate.
- **State:** `STAGE_C_DEPLOYMENT_READINESS_FINAL.md` §19 records **NOT DEPLOYABLE on any Hailo
  target at this time**. Nothing is deployed.

A reader who stops at the Phase-2 report concludes that P2A is the deployed model of this
project. That is the single most consequential stale statement found in the tree.
**Classification: STALE / SUPERSEDED (not a fabricated claim).** Not repaired.

### F2 — "Dynamic range: **FAIL**" is a stronger label than the evidence carries — **DOCUMENTATION DISCREPANCY**

Locations: `STAGE_C_DEPLOYMENT_READINESS_FINAL.md` §6 ("Classification: FAIL as a deployment
property") and §19 ("Dynamic range: FAIL (property; ~1e18 activations)"); risk register R3
("FAIL (property)").

Against this:

- **No predeclared dynamic-range gate, threshold or criterion exists anywhere in the project.**
  Unlike C1 (1e-3 px) and DR-1 (1e-3 px), there is no number the activation range was measured
  against and failed.
- `DYNAMIC_RANGE_AUDIT.md` §15 records the decisive question — *whether the ~2.53e18 tensor
  survives per-tensor int8 quantisation* — as **UNKNOWN until quantisation is attempted with the
  toolchain**, and explicitly predicts no outcome. §13 lists only the measured range as VERIFIED.

What is established: an extreme measured activation range (ARM-P ~1e18 vs reference ~1e1–1e2),
VERIFIED, and an UNKNOWN target-native quantization outcome. The readiness report does qualify
the label ("INT8 quantization hostile **until proven otherwise**"), so this is a labelling
discrepancy rather than an invented result.

**Correct classification per the termination-gate brief: DEPLOYMENT RISK / UNKNOWN
TARGET-NATIVE BEHAVIOUR.** Recorded here; the readiness report was not edited.

### F3 — `phase1/results/LEADERBOARD.md` is stale and no global results index exists — **PROCESS DEVIATION**

The leaderboard's last entry is the ARM-V multi-seed decision (2026-09-15). It contains **zero**
occurrences of P2A or ARM-P. Absent from it: the three P2A scale-coverage seeds and the entire
Stage-B Tier-2 set (3 seeds × 2 arms + the environment control = 8 trained models, 16 scored
checkpoints).

`CLAUDE.md` states: "`phase1/results/LEADERBOARD.md` is updated after every completed
experiment." Taken literally, that rule has not been met since 2026-09-15.

Mitigation on record: no result is lost — P2A runs are indexed in
`phase2/docs/PHASE_2_REGISTRY.md` and `PHASE_2_FINAL_REPORT.md`, and Stage-B runs in
`stage_b_armp/ARMP_CLOSURE_RECORD.md` §§4–7 with per-run directories. The leaderboard is also
arguably Phase-1-scoped by name.

Residual issue: **no single index spans every trained model of the project.** A reader asking
"what was trained, and what did it score?" must know three separate documents exist.
**Classification: PROCESS DEVIATION / STALE INDEX.** Not repaired.

### F4 — `README.md` is Phase-1-only; the repository has no current entry point — **STALE (navigation gap)**

`README.md` is titled "Stereo Depth Vision — **Phase 1**" and its "Start here" table points only
at `docs/phase_1_project_report.md`, `PHASE_1_FINAL_REPORT.md`, `docs/phase_1_closure_audit.md`
and `docs/`. It does not mention Phase 0, the re-scoped Phase 2, Stage A, Stage B, ARM-P, Stage C,
the C1 FAIL, or the deployment blockers.

This bears directly on the closure test (brief §30). Every one of the eleven closure questions is
answerable from the tree — but the navigation to those answers is not. A cold-start engineer
entering at `README.md` lands on the state of the project as of Phase 1 and would have to
discover `stage_b_armp/ARMP_CLOSURE_RECORD.md` and
`stage_c_deploy/STAGE_C_DEPLOYMENT_READINESS_FINAL.md` unaided.

**Classification: STALE (navigation gap).** No content is false. Not repaired.

### F5 — Two declared deliverables were never finished — **OPEN**

Both are declared next actions of closed research lines, and neither appears as an open branch in
any closure record:

1. **Stage B synthesis/report — NOT WRITTEN.**
   `stage_b_armp/POST_IDENTIFIABILITY_BOUNDARY.md` §13 names its "EXACT NEXT ACTION" as writing
   the Stage B synthesis. What exists is `stage_b_armp/STAGE_B_SYNTHESIS_OUTLINE.md`, whose own
   header states: "**OUTLINE AND EVIDENCE MAP ONLY … does NOT draft the manuscript … Bullet-level
   content is correct; finished narrative is not.**" The synthesis itself does not exist.

2. **Superseded-Phase-2 campaign paper — DRAFT, INCOMPLETE, NEVER COMPILED.**
   `phase2/docs/POST_CLOSURE_RESEARCH_AUDIT.md` closes that campaign with
   "**Final verdict: 1 — STOP EXPERIMENTATION AND WRITE THE PAPER**". `phase2/paper/` holds
   `main.tex` + `refs.bib` + a README recording **6 outstanding TODOs** (UNVERIFIED aggregation
   rule; UNVERIFIED checkpoint hash; UNVERIFIED TR-001 t=96 detail; three missing citations —
   competitor survey, interpretability, accuracy anchor), **intentional placeholders F1–F7**, and
   "no local toolchain" (never compiled). Claim ceiling correctly carried: Level D, genuine
   geometric correspondence NOT-DEMONSTRATED.

**Classification: OPEN — deliverable incomplete** (both). Neither is a research question; neither
requires compute; neither blocks closure of the evidence. They are unfinished writing.

---

## 4. Reader hazard (not a discrepancy)

`results/quantization/artifacts/stereonet_int8.onnx` (6,105,895 B, 2026-09-05) and
`results/quantization/report.txt` exist and show an int8 row
(`int8_onnxruntime_cpu  EPE 1.655  D1 10.945`) beside fp32 1.313.

That is Phase-1 **EXP-015**, an **onnxruntime-CPU precision study of the reference model**
(fp32 EPE 1.313 identifies it), predating P2A and ARM-P entirely. It is **not** Hailo
quantization, **not** ARM-P, and **not** evidence that quantization of the deployment candidate
succeeds or is even possible. **Classification: HISTORICAL.** Quoted here only so the artifact's
existence cannot be mistaken for §18-claim-13 ("quantization succeeds"), which remains
**NOT TESTED / UNSAFE TO CLAIM**.

---

## 5. Claim-audit deltas

No claim status changes. Two wordings are tightened by this pass:

| Claim | First-pass status | Second-pass note |
|---|---|---|
| "Dynamic range is solved" | NOT SOLVED (risk open) | unchanged — but the *converse* label "Dynamic range: FAIL" is itself unsupported (F2); the safe statement is "extreme range VERIFIED; int8 survival UNKNOWN" |
| "Quantization succeeds" | NOT TESTED | unchanged — plus: a historical int8 artifact of the **reference** model exists in-tree and must not be cited for it (§4) |
| "ARM-P is deployed" / "deployment complete" | NO / BLOCKED | unchanged — but one in-tree document still reads "Model Deployed" of P2A (F1) |

Every other row of the first pass's §21 claim table is concurred with as written.

---

## 6. Open-branch deltas

Added to the first pass's §22 list, none of which may silently disappear:

| Branch | State |
|---|---|
| Stage B synthesis report | **OPEN — NOT WRITTEN** (outline only) |
| Superseded-Phase-2 campaign paper | **OPEN — draft with 6 TODOs + F1–F7 placeholders, never compiled** |
| Repository documentation refresh (F1, F3, F4) | **OPEN — deliberately not performed by either audit** |
| Application layer | **NOT STARTED** (no application code exists in the repo) |
| Post-Phase-1 evidence under version control | **OPEN** — `phase0/`, `phase2/`, `stage_a_diagnostics/`, `stage_b_armp/`, `stage_c_deploy/` are untracked; "nothing overwritten" rests on hashes, mtimes and run-record discipline, not on a clean tracked tree (PROVENANCE GAP, disclosed in Stage B open defect 4) |

---

## 7. Second-pass verification checklist (brief §34)

- [x] No experiment was run accidentally — hashing, reads, an import/PATH probe, `find`, git only.
- [x] No frozen artifact changed — seven hashes measured, all matching (§2a).
- [x] No historical result was overwritten — including the concurrent first-pass deliverables,
      which were found on disk and left untouched (§0).
- [x] No tolerance changed — C1 1e-3, DR-1 1e-3 quoted unchanged.
- [x] No target inferred — Hailo-8 / 8L / 10H / 15H remain HISTORICAL REFERENCE.
- [x] No unsupported claim promoted; no UNKNOWN became PASS.
- [x] No "NOT IDENTIFIABLE" became "FALSE"; no "OPEN BUT UNSUPPORTED" became "REFUTED".
- [x] C1 FAIL, DR-1 H1 FAIL, C2/C2.1/C2.1.1 PASS, Hailo BLOCKED — all correctly classified.
- [x] Git changes accounted for: 5 pre-existing modified tracked files (neither audit's),
      untracked evidence tree (pre-existing), this file (this pass's only creation).
- [x] All open branches explicitly classified (§6 + first pass §22).
- [x] The conclusion follows from evidence, not convenience.

---

## 8. Addendum verdict

**Concurring: PROJECT CLOSED WITH DOCUMENTATION DISCREPANCIES.**

Material integrity holds — seven frozen artifacts hash-verified, every gate record intact, both
known failures (C1 1.708984375e-3 px; DR-1 1.4816284e-2 px) preserved with exact numbers and
unrepaired, every identifiability boundary stated with the right word, and no unsupported current
claim anywhere in the tree. What stands between this project and an unqualified "no material
issue" is writing, not evidence: one headline report still calls P2A the deployed model (F1), one
classification is stronger than its own backing audit (F2), no index spans the trained models
(F3), the repository has no current entry point (F4), and two declared deliverables were never
finished (F5). None requires compute. None reopens a research question. All five should be closed
in writing alongside the project, or explicitly accepted as known limitations of the record.

### ARTIFACT INTEGRITY

No frozen artifact changed. Seven SHA-256 values measured this pass, all matching, including the
Stage-1 pretrain checkpoint and the C1 BASE re-export (bit-identical to the frozen ONNX). The
first-pass deliverables were not modified.

### EXPERIMENT INTEGRITY

No experiment, training, fine-tuning, export, quantization, compilation, evaluation re-run or
model/inference change occurred during this pass.

### PROJECT CLOSURE STATUS

**PROJECT CLOSED WITH DOCUMENTATION DISCREPANCIES**

---

*End of second-pass addendum. One file created; no existing file modified, renamed or deleted.*
