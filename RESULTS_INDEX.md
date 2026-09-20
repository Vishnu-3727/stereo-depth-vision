# RESULTS INDEX — MICROCHIP STEREONET (current)

Navigation index for the whole project, created 2026-09-20 by the documentation closure pass.
**Navigation only — this file creates no new claim, no new number and no new experiment.**
Every status below is quoted from the record cited beside it.

The historical Phase-1 leaderboard (`phase1/results/LEADERBOARD.md`) covers the Phase-1 arms
only, ends at ARM-V (2026-09-15), and remains **HISTORICAL** — it is not rewritten and is not
the current index. This file is.

---

## 0. One-line project state

| | |
|---|---|
| Technical investigation | **CLOSED** |
| Frozen deployment candidate | **ARM-P seed 1** — `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth`, sha256 `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454`, 397,954 params, frozen EPE **1.1912168 px** |
| Hailo deployment | **BLOCKED** — target device UNSPECIFIED, toolchain NOT EXECUTED, ARM-P HEF ABSENT |
| C1 export parity | **FAIL** (0.001708984375 px vs frozen 1e-3 px) |
| DR-1 activation rescale | **H1 FAIL** (pooled max 1.4816284e-02 px) |
| C2 / C2.1 / C2.1.1 | **PASS** (135/135, 251/251, 19/19) |
| Remaining work | documentation / navigation only |

**Current deployment statement (authoritative wording):**
*ARM-P seed 1 is the frozen deployment candidate selected under C0, but deployment to Hailo has
NOT occurred because the target Hailo device has not been specified and the corresponding target
toolchain has therefore not been executed.*

---

## 1. Contract and baselines

| Item | Status | Source | Purpose |
|---|---|---|---|
| Frozen evaluation contract | **CLOSED / AUTHORITATIVE** | `phase1/harness/frozen_eval.py`; `phase0/docs/BASELINE_CONTRACT.md` | 40 KITTI scenes (160–199), `disp_occ_0`, 368×1232, GT 1/256, 3,802,797 valid pixels. Never modified to make a number look better. |
| Phase 0 — baseline lock | **CLOSED** | `phase0/docs/PHASE_0_FINAL_REPORT.md` | Locks the contract; reference ONNX **1.3134471 px / D1 8.1543664%**; start PyTorch baseline 15.3958267 px; gap +14.0823796 px. |
| Reference baseline record | **CLOSED / HISTORICAL REFERENCE** | `phase0/docs/REFERENCE_BASELINE.md` | The project-controlled reference figure. **Not** Hailo's published 8.22/10.3 numbers — those are a different metric (D1 outlier %) and a different protocol. |

## 2. Phase 1 — reference forensics and arm campaign

| Item | Status | Source | Purpose |
|---|---|---|---|
| Phase 1 final report | **CLOSED / HISTORICAL** | `PHASE_1_FINAL_REPORT.md` | Charter deliverable; establishes that the published "EPE 8.223" is the KITTI D1 outlier percentage, reproduced at 8.2237. |
| Phase 1 closure audit | **CLOSED / HISTORICAL** | `docs/phase_1_closure_audit.md` | Release gate: what was verified, what stayed unknown. |
| Phase-1 leaderboard | **HISTORICAL (not current)** | `phase1/results/LEADERBOARD.md` | Phase-1 arms only, ends at ARM-V 1.7727436 px (3-seed mean). Contains no P2A and no ARM-P entry — by design; see this index instead. |
| Phase 1 bottleneck audit | **CLOSED** | `phase1/docs/FINAL_PHASE1_BOTTLENECK_AUDIT.md` | Decision not to run a fourth Phase-1 arm; localises the residual error to GT ≥ 64 px. |
| Knowledge base (12 docs) | **HISTORICAL** | `docs/` | Architecture, cost volume, failure analysis, compute/hardware profiles, source registry. |

## 3. Phase 2 (re-scoped) — P2A optimization and export

| Item | Status | Source | Purpose |
|---|---|---|---|
| Phase 2 final report | **HISTORICAL / SUPERSEDED deployment wording** | `PHASE_2_FINAL_REPORT.md` (carries a supersession notice) | Closes the re-scoped Phase 2. Its "Model Deployed / P2A seed 0" wording is historical; the current candidate is ARM-P seed 1 and nothing is deployed. |
| P2A experiment result | **CLOSED — verdict INCONCLUSIVE, mechanism NOT CONFIRMED (both permanent)** | `phase2/docs/PHASE2_P2A_SCALE_COVERAGE_RESULT.md`, `phase2/docs/PHASE2_HYPOTHESIS_01.md` | The single preregistered scale-coverage intervention; 3-seed method mean 1.4409826 px. |
| P2A deployment selection | **HISTORICAL** | `phase2/docs/PHASE2_DEPLOYMENT_SELECTION.md`, `PHASE2_DEPLOYMENT_VALIDATION.md` | The Phase-2 selection of P2A seed 0 and its ONNX export/parity (PASS at 6.1798e-4 px). Superseded as *the* candidate by C0. |
| Superseded Phase-2 line (H1/H2/E-series correspondence campaign) | **CLOSED at Level D — genuine geometric correspondence NOT-DEMONSTRATED** | `phase2/docs/POST_CLOSURE_RESEARCH_AUDIT.md`, `phase2/docs/PHASE_2_REGISTRY.md` | A different research line on the 423,586-param / 12-candidate reference graph. Do not confuse with the P2A/ARM-P line (397,954 params / 24 candidates). |

## 4. Stage A — gap diagnostics

| Item | Status | Source | Purpose |
|---|---|---|---|
| Stage A diagnostics D0–D7 | **CLOSED / COMPLETE** | `stage_a_diagnostics/DIAGNOSTIC_REPORT.md`, `stage_a_diagnostics/README.md` | Decomposes the 0.1275355 px P2A→reference gap (~31% below 64 px, ~66% in 64–128 px, ~3% ≥128 px; attribution sums exactly). |
| Mechanism verdict | **CLOSED — architecture gate NO ARCHITECTURE JUSTIFIED** | `stage_a_diagnostics/mechanism_verdict.json` | Per-mechanism labels: matching PLAUSIBLE; readout PLAUSIBLE + NOT IDENTIFIABLE limit; aggregation NOT IDENTIFIABLE; refinement REFUTED (scoped); supervision coverage SUPPORTED (DATA fix). |

## 5. Stage B — ARM-P initialization intervention

| Item | Status | Source | Purpose |
|---|---|---|---|
| Pre-flight / provenance audit | **CLOSED — upstream PROVENANCE NOT ESTABLISHED** | `stage_b_armp/20260918T053135Z/README.md` | Why pretraining used P2A's own architecture; FT3D corpus limits. |
| ARM-P Stage 1 (FT3D pretraining) | **CLOSED** | `stage_b_armp/20260918T062146Z_stage1_pretrain/README.md` | FT3D **A+C subset, 14,460 usable triplets** (B missing, Monkaa empty); source checkpoint sha256 `3ae6fb3b…1a29be7`. |
| Tier-2 seed 0 (pilot, local) | **CLOSED** | `stage_b_armp/20260918T142128Z_tier2_pilot/TIER2_FROZEN_AUDIT.md` | ARM-P best **1.2057590** vs CONTROL 1.5587716 (Δ −0.3530126). |
| Tier-2 seed 1 (local) | **CLOSED — selected candidate** | `stage_b_armp/20260919T012646Z_tier2_seed1/SEED1_REPLICATION_REPORT.md` | ARM-P best **1.1912168** vs CONTROL 1.4440242 (Δ −0.2528074). |
| Tier-2 seed 2 (Kaggle-trained, locally evaluated) | **CLOSED** | `stage_b_armp/20260919T092454Z_tier2_seed2/SEED2_REPLICATION_REPORT.md` | ARM-P best **1.1996447** vs CONTROL 1.4932978 (Δ −0.2936531). |
| Environment control (seed 1, Kaggle) | **CLOSED** | `stage_b_armp/20260919T130843Z_seed1_kaggle_envcontrol/ENV_CONTROL_REPORT.md`, `stage_b_armp/ENV_CONTROL_PREREGISTRATION.md` | Bounds the seed-2 environment confound: within-delta shift ≈ +0.022 px, driven mostly by the CONTROL arm. Not a new seed. |
| **Stage B closure record (authoritative)** | **CLOSED — ESTABLISHED BOUNDED FINDING** | `stage_b_armp/ARMP_CLOSURE_RECORD.md` | CONTROL = **random init**; ARM-P = **FT3D-pretrained**. ARM-P lower in 6/6 seed×selection cells. Explicitly NOT: universal superiority, significance, causal proof, generalization, deployment benefit. |
| Stage B synthesis outline | **COMPLETE as a planning artifact; the downstream manuscript is NOT REQUIRED FOR CURRENT PROJECT** | `stage_b_armp/STAGE_B_SYNTHESIS_OUTLINE.md` | Evidence map only. Stage B is closed by the closure record above; see §9 of this index. |

## 6. Identifiability boundary (Stage B)

| Item | Status | Source | Purpose |
|---|---|---|---|
| Coverage vs pretraining (Candidate C) | **NOT IDENTIFIABLE** | `stage_b_armp/CANDIDATE_C_SEPARABILITY.md` | Coverage cannot be varied independently of initialization/augmentation on the frozen platform. No experiment manufactured. |
| True GWC on frozen P2A | **NOT IDENTIFIABLE — branch remains OPEN** | `stage_b_armp/GWC_IDENTIFIABILITY_DEFINITION.md` | Specification only; no GWC code, no training. |
| GWC as a hypothesis | **OPEN BUT UNSUPPORTED — NOT REFUTED** | `stage_b_armp/STAGE_B_SYNTHESIS_OUTLINE.md` rule 2 | Distinct from the row above: "not identifiable on this platform" ≠ "does not work". |
| ARM-W (group-wise cost, Phase 1) | **REFUTED (≈1.776 px)** | `phase1/results/LEADERBOARD.md`, Stage A D3 | Mechanistically **non-equivalent** to True GWC; its refutation does not transfer. |
| Post-identifiability boundary | **OPEN BUT UNSUPPORTED — freeze decided** | `stage_b_armp/POST_IDENTIFIABILITY_BOUNDARY.md` | Records why no multi-component GWC program was launched. |

## 7. Stage C — deployment attempt (device-independent)

| Item | Status | Source | Purpose |
|---|---|---|---|
| Stage C0 contract | **CLOSED** | `stage_c_deploy/STAGE_C0_DEPLOYMENT_CONTRACT.md` | Deployment ground rules; forbids inferring a target. |
| C0 candidate decision | **CLOSED** | `stage_c_deploy/C0_DECISION_RECORD.md` | Predeclared argmin rule over three ARM-P seeds → **seed 1**; hash frozen. |
| C1 export parity | **FAIL (known, frozen, unrepaired)** | `stage_c_deploy/C1_EXPORT_PARITY_REPORT.md`, `stage_c_deploy/ARMP_DEPLOYMENT_VALIDATION.json` | ARM-P ONNX vs PyTorch **max abs diff 0.001708984375 px** against the frozen **1e-3 px** criterion → FAIL. (P2A control 6.1798e-4 px PASS.) |
| C1 parity diagnostic | **CLOSED** | `stage_c_deploy/diagnostics/C1_PARITY_DIAGNOSTIC_REPORT.md` | Localisation; refutes corruption, nondeterminism, ORT fusion, occlusion, high-texture and dynamic range as the explanation. |
| C1 §10 export investigation | **CLOSED — EXPORT-LEVEL MITIGATION NOT IDENTIFIED (tested space)** | `stage_c_deploy/diagnostics/C1_SECTION10_EXPORT_INVESTIGATION.md` | 5 export variants (opset 11/13/17 × folding) → bit-identical outputs, same 1.708984375e-3 px. Decomposition: Δinitial 7.143e-4 (would pass) / Δresidual 1.6899e-3 (carries the FAIL) / Δfinal 1.708984375e-3. Residual causal split **NOT IDENTIFIED / NOT TESTED**. |
| DR-0 dynamic-range audit | **CLOSED — measured; target-native behaviour UNKNOWN** | `stage_c_deploy/DYNAMIC_RANGE_AUDIT.md` | ARM-P `aggregated_cost` ~1.57e18–2.53e18 vs reference ~1e1–1e2. Dynamic range does **not** explain the C1 failure. int8 survival UNKNOWN. |
| DR-1 rescale gate | **H1 FAIL** | `stage_c_deploy/DR1_RESCALE_GATE.md`, `stage_c_deploy/dr1_rescale/` | Variant A (`aggregated_cost / 1e17`): pooled max **1.4816284e-02 px** vs the 1e-3 equivalence threshold → rejected, not adopted. 40-scene EPE delta −4.7937e-6, D1 delta 0.0. **Variant B NOT RUN.** |
| C2 metric depth | **PASS — 135/135** | `stage_c_deploy/C2_METRIC_DEPTH_REPORT.md`, `stage_c_deploy/metric_depth/out/c2_validation.json` | `Z = fB/d` + XYZ reprojection. Deterministic host code; **no neural model added, model unchanged**. |
| C2.1 spatial perception | **PASS — 251/251** | `stage_c_deploy/C2_1_SPATIAL_PERCEPTION_REPORT.md`, `…/out/c2_1_validation.json` | Point cloud, occupancy, nearest surface, spatial distance. Deterministic host code. Timings are development-machine only. |
| C2.1.1 discontinuity | **PASS — 19/19** | `stage_c_deploy/C2_1_1_DISCONTINUITY_VALIDATION.md`, `…/out/c2_1_1_discontinuity_validation.json` | Gradient-magnitude discontinuity gate; module sha256 `2e81e9f2…1b5743`. Evidence only — no ground-truth accuracy claim. |
| Target / toolchain resolution | **TARGET UNKNOWN / RESOLUTION BLOCKED** | `stage_c_deploy/C1_TARGET_TOOLCHAIN_RESOLUTION.md`, `stage_c_deploy/target_resolution/evidence_summary.json` | Exhaustive search: **zero** authoritative company device requirements in the repo. |
| Deployment blocker report | **BLOCKED** | `stage_c_deploy/STAGE_C_DEPLOYMENT_BLOCKER_REPORT.md` | What is blocked and why. |
| Pipeline demo | **RUNNABLE** (device-independent) | `stage_c_deploy/demo/pipeline_demo.py`, `stage_c_deploy/demo/README.md` | One command runs the frozen candidate end to end on a KITTI pair and draws disparity, metric depth, discontinuities and the point cloud; composes only validated modules, asserts the checkpoint hash, and prints the C1 FAIL and the Hailo blocker on every run. |
| Stage C readiness final | **NOT DEPLOYABLE on any Hailo target at this time** | `stage_c_deploy/STAGE_C_DEPLOYMENT_READINESS_FINAL.md` | The Stage-C deliverable; risk register R1–R10; the 14 company fields required to unblock. |

## 8. Audits

| Item | Status | Source | Purpose |
|---|---|---|---|
| Final whole-system audit (first pass) | **PROJECT CLOSED WITH DOCUMENTATION DISCREPANCIES** | `stage_c_deploy/FINAL_WHOLE_SYSTEM_AUDIT.md` (+ `final_whole_system_audit.json`) | Termination-gate audit: hashes, claim audit, open-branch audit, evidence ladder. |
| Final whole-system audit (second pass) | **CONCURRING — same verdict** | `stage_c_deploy/FINAL_WHOLE_SYSTEM_AUDIT_SECOND_PASS.md` | Independent second pass; 7 hashes measured; adds findings F1–F5 that this documentation closure pass resolves. |
| Documentation closure record | **CLOSED** | `stage_c_deploy/DOCUMENTATION_CLOSURE_RECORD.md` | What this documentation pass changed, and what it deliberately did not. |

## 9. Documentation deliverables — explicit status (F5)

| Item | Status | Basis |
|---|---|---|
| `stage_b_armp/STAGE_B_SYNTHESIS_OUTLINE.md` | **COMPLETE** as the planning artifact it declares itself to be. The downstream Stage-B synthesis manuscript is **NOT REQUIRED FOR CURRENT PROJECT.** | Stage B is closed by `ARMP_CLOSURE_RECORD.md`, which is the authoritative Stage-B record and contains the findings, limitations and confounds in full. The manuscript would add presentation, not evidence. No research is required or permitted to produce it. |
| `phase2/paper/main.tex` + `refs.bib` + `README.md` | **TODO — DOCUMENTATION ONLY** (retained; does not block project closure) | Belongs to the *superseded* Phase-2 correspondence line, whose own audit verdict was "stop experimentation and write the paper". The draft carries 6 unresolved TODOs (2 UNVERIFIED facts, 1 UNVERIFIED checkpoint hash, 3 missing citations) and intentional placeholders F1–F7, and has never been compiled (no local TeX). Completing it is a writing task at claim ceiling **Level D**; **no new research may be run to complete it.** |

## 10. Reader hazards — read this before quoting any number

1. **P2A seed 0 is NOT the currently deployed model.** It is a historical Phase-2 selection. Current candidate: ARM-P seed 1 (C0).
2. **Hailo deployment has NOT happened.** No compile, no HEF, no silicon, no emulator run.
3. **No ARM-P HEF exists.** The only `.hef` in the tree is `reference/stereonet.hef` — **HISTORICAL** reference artifact.
4. **The Hailo target device is UNKNOWN.** Model-Zoo `hailo8` (v2.19.0), `hailo15h/hailo10h` (master), legacy HAILO8 docs and DFC 3.34.0 are **HISTORICAL REFERENCE** and must never be promoted to "the target".
5. **Dynamic range is NOT proven to fail on Hailo** — and **NOT proven safe** either. Measured: ~1e18 fp32 intermediates. Established: DR-1 Variant A failed H1. Unknown: target-native behaviour. No threshold exists.
6. **C1 did NOT pass** (0.001708984375 px vs 1e-3 px). **DR-1 did NOT pass** (1.4816284e-02 px). These are two separate, scoped failures — never merge them into "deployment failed".
7. **C2/C2.1/C2.1.1 did NOT change the neural model.** They validate deterministic, device-independent host code downstream of disparity, and they validate **nothing about Hailo hardware**.
8. **Stage B is not an active campaign.** It is closed; U1–U5 follow-ups are listed but **NOT AUTHORIZED and NOT LAUNCHED**.
9. **GWC was neither proven nor disproven.** See §6.
10. **Coverage vs pretraining was NOT causally separated.** NOT IDENTIFIABLE.
11. **`results/quantization/artifacts/stereonet_int8.onnx` is NOT Hailo evidence and NOT ARM-P.** It is Phase-1 EXP-015: onnxruntime-CPU int8 of the **reference** model (int8 EPE 1.655 vs fp32 1.313). See `results/quantization/README.md`.
12. **Hailo's published 8.22 / 10.3 / 10.7 FPS / 11.6 FPS are a different metric and protocol** (D1 outlier %, Hailo-8 hardware), never the project-controlled 1.3134471 px reference.
13. **`phase2/` holds two different research lines** — the superseded correspondence campaign (423,586 params, 12 candidates) and the re-scoped P2A work (397,954 params, 24 candidates).

## 11. Frozen artifact hashes (documentation pointers)

| Artifact | Path | SHA-256 |
|---|---|---|
| ARM-P checkpoint (candidate) | `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth` | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` |
| ARM-P ONNX | `stage_c_deploy/armp_stereonet.onnx` | `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989` |
| ARM-P Stage-1 pretrain checkpoint | `stage_b_armp/20260918T062146Z_stage1_pretrain/checkpoints/armp_stage1_best.pth` | `3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7` |
| P2A checkpoint (historical candidate) | `phase2/runs/p2a_scale_coverage/p2a_best.pth` | `0868ffd137a9985306bf5563685fd2362799bdd630d313e1181c980a7fbb6033` |
| Reference ONNX | `reference/onnx/stereonet.onnx` | `b1a01d855bb22663f06dfd29eae11194e6fde952a319673e036f157c3e10095c` |
| Discontinuity module | `stage_c_deploy/spatial_perception/discontinuity.py` | `2e81e9f2e97dbde196ba2311c68171aa4ce1195430b5c5b260d23b495e1b5743` |

Known benign spelling issue: several records carry the transposed ARM-P ONNX spelling
`…deed1**dce**26…`; the on-disk value `…deed1**cde**26…` above is authoritative and is
self-annotated in those records. Not repaired (evidence is frozen).

---

*Index only. No technical artifact, result, tolerance or conclusion was created or changed to
produce this file.*
