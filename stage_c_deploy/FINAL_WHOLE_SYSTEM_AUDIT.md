# MICROCHIP STEREONET — FINAL WHOLE-SYSTEM INTEGRITY / EVIDENCE / CLOSURE AUDIT
**Project termination gate — READ-ONLY audit. No training, no fixes, no cleanup performed.**
**Audit date (UTC): 2026-09-20. Auditor role: final independent auditor.**

> Freeze compliance: during this audit no training, fine-tuning, architecture/loss/augmentation/
> dataset/checkpoint/weight/ONNX/exporter/quantization/Hailo/HEF/hardware/confidence/perception
> change was performed. Only reads + SHA-256 re-hashes + `git status/diff/log` + grep inventory
> were executed. The only files created by this audit are the two deliverables authorized by §32.

---

## 1. Audit scope

Complete project audited: `phase0/`, `phase1/`, `phase2/` (incl. re-scoped Stage A),
`stage_a_diagnostics/`, `stage_b_armp/`, `stage_c_deploy/` (C0/C1/DR-0/DR-1/C2/C2.1/C2.1.1/
target/toolchain), `diagnostics/` (under stage_c_deploy), `reference/`, root docs
(`PHASE_1_FINAL_REPORT.md`, `PHASE_1_OVERVIEW.md`, `PHASE_2_FINAL_REPORT.md`,
`AUDIT_REPORT.md`, `docs/*`), experiment records, audit records, checkpoint inventories,
deployment records, validation JSONs, manifests, source (`src/models/stereonet/*`).

Keyword sweep executed for: ARM-P, P2A, Hailo, C1, C2, C2.1, C2.1.1, DR-0, DR-1, GWC,
Candidate C, Stage A/B/C, parity, deployment, target, SceneFlow, FT3D, KITTI, EPE, D1,
checkpoint, ONNX, HEF, quantization, target device. ~90 device-name lines + 59
requirement-term matches reviewed; all are historical-reference or UNKNOWN-recording,
zero prescriptive (per `target_resolution/evidence_summary.json`, independently confirmed
by grep in this audit).

State hierarchy applied (§3): frozen contract > closure/audit record > immutable
experiment result > validated machine-readable artifact > earlier report > scratch/log >
historical reference > informal prose. Conflicts are classified, not reconciled.

## 2. Evidence inventory (abridged — full tree on disk)

- Phase 0: `phase0/docs/{BASELINE_CONTRACT,REFERENCE_BASELINE,PHASE_0_FINAL_REPORT}.md`
- Phase 1 (forensic): `PHASE_1_FINAL_REPORT.md`, `PHASE_1_OVERVIEW.md`,
  `docs/phase_1_closure_audit.md`, `docs/{hardware_analysis,reference_pipeline,source_registry}.md`,
  `reference/{MANIFEST.md,manifest.json,onnx/stereonet.onnx,hef}`, `experiments/EXP-005…018`
- Stage A: `PHASE_2_FINAL_REPORT.md`, `stage_a_diagnostics/*`, `phase2/runs/p2a_scale_coverage/*`
- Stage B: `stage_b_armp/*.md` (closure, prereg, Candidate C, GWC definition, post-identifiability
  boundary, synthesis outline, pilot README, ENV_CONTROL), tier2 dirs
  `20260918T142128Z_tier2_pilot/`, `20260919T012646Z_tier2_seed1/`,
  `20260919T092454Z_tier2_seed2/`, `20260919T130843Z_seed1_kaggle_envcontrol/`,
  each with `tier2_eval.json` + `armp|control/{p2a_best.pth,p2a_final.pth,record.json,…}`
- Stage C: `stage_c_deploy/{C0_DECISION_RECORD,STAGE_C0_DEPLOYMENT_CONTRACT,
  C1_EXPORT_PARITY_REPORT,C1_TARGET_TOOLCHAIN_RESOLUTION,C2_METRIC_DEPTH_REPORT,
  C2_1_SPATIAL_PERCEPTION_REPORT,C2_1_1_DISCONTINUITY_VALIDATION,DYNAMIC_RANGE_AUDIT,
  DR1_RESCALE_GATE,STAGE_C_DEPLOYMENT_BLOCKER_REPORT,STAGE_C_DEPLOYMENT_READINESS_FINAL}.md`,
  `ARMP_DEPLOYMENT_VALIDATION.json`, `ARMP_STATIC_EXPORT_VALIDATION.json`,
  `c0_checkpoint_inventory.json`, `stage_c_deployment_readiness_final.json`,
  `armp_stereonet{,_static}.onnx`, `diagnostics/{c1_parity_diagnostic.*,C1_PARITY_DIAGNOSTIC_REPORT.md,
  c1_section10_export_investigation.json,C1_SECTION10_EXPORT_INVESTIGATION.md,
  c1_export_variants/*}`, `dr0_audit/*`, `dr1_rescale/*`, `metric_depth/out/c2_validation.json`,
  `spatial_perception/out/{c2_1_validation.json,c2_1_1_discontinuity_validation.json}`,
  `target_resolution/evidence_summary.json`

## 3. Phase 0 status — CLOSED

Established: KITTI-2015 `_10` frozen contract (`hailo_val` scenes 160–199, 40 scenes,
368×1232 pad-bottom-right then top-left crop deleting 7 rows + 10 cols — explicitly
non-neutral, `disp_occ_0`, GT 1/256, `gt>0`, pooled 3,802,797 px, EPE + D1-all, no-mixing rule).
Reference ONNX verified live: EPE **1.3134470770188373**, D1 8.154366378221082.
Hailo-exact 8.223686464356268% is D1 (per-image 1/255), never EPE; vendor yaml mislabel
resolved in favor of evaluator. PyTorch baseline (EXP-016, 423,586 params, 72 tensors):
EPE 15.395826671257934, gap +14.0823796 px. Arch match YES (max diff 8.24e-4 px),
provenance DIFFERENT, mapping EXACT, regime share UNKNOWN.
No later document reopens Phase 0. **PHASE 0 = CLOSED, no discrepancy.**

## 4. Phase 1 status — CLOSED

Established and frozen (tag `phase-1-frozen`, 18 EXPs, 75 tests, 73 claim checks):
degenerate cost volume (12 slices bit-identical, EXP-010); "EPE 8.223" is D1-%;
refinement 90.6% MACs / 73.3% GPU time; independent impl 1e-7 (7.596e-7 worst);
fp16 free, int8 +2.79 D1 pt; no depth conversion in reference; `cost_volume_shift="none"`,
H1 never enabled. Project-controlled Hailo reference (1.3134471 EPE) kept distinct from
public benchmark numbers (8.22 float / 10.3 hw — D1-% mislabelled EPE on Hailo-8).
No stale merge found beyond the vendor yaml mislabel, which is explicitly resolved.
**PHASE 1 = CLOSED.**

## 5. Stage A status — CLOSED (no architecture justified)

P2A frozen: 397,954 params, 70 keys, 24 disparities, 3 levels, shift-right,
`regression_normalize` true, no BN. Baseline seed0 1.4149795736625577, mean 1.4409825930575104,
reference 1.3134471, gap 0.1275355 px. Conclusions preserved with exact wording:
gap splits 31% <64 / 66% 64–128 / 3% ≥128; range REFUTED twice; readout/aggregation
NOT IDENTIFIABLE (co-adaptation limit — not "refuted"); matching PLAUSIBLE; supervision
coverage SUPPORTED as data fix; D7 long-range context WEAK; ≥128 99.17% occluded;
Gate: NO ARCHITECTURE JUSTIFIED; U1–U5 open. EXP-P2A-SCALE-COVERAGE-001 INCONCLUSIVE.
Old `phase2/` H1/H2/E numbering is superseded by explicit note in `PHASE_2_FINAL_REPORT.md`
— scheme change, not contradiction. **STAGE A = CLOSED.**

## 6. Stage B status — CLOSED (with frozen limitations)

Frozen question preserved: does Stage-1 SceneFlow-pretrained init + exact P2A KITTI recipe
beat the same recipe from random init? Wording verified verbatim across closure,
pilot README, prereg Rule 6: **CONTROL = RANDOM-INIT P2A; ARM-P = FT3D-PRETRAINED P2A**
(strict `load_state_dict(strict=True)` of `["model"]` from stage-1 best
`3ae6fb3b…9be7`, no optimizer inheritance; 14,460-triplet / 7-of-20-epoch early-stop
limitation disclosed). No document calls the control a "frozen P2A checkpoint".
Environment control preregistered; shift small (best +0.0222 / final +0.0227 px),
concern reduced not disproven. **STAGE B = CLOSED.**

## 7. ARM-P status — VERIFIED (replicated directional finding, bounded claim)

| Seed | ARM-P best (px) | Control best (px) | Δ |
|---|---|---|---|
| 0 | **1.2057590024628537** | 1.5587715928467816 | −0.3530 |
| 1 | **1.191216765057325** | 1.4440242127250567 | −0.2528 |
| 2 | **1.1996447345404795** | 1.4932978077501271 | −0.2937 |

Mean ≈ 1.198874 px < controlled Hailo reference 1.3134471 px. Correct phrasing preserved:
"ARM-P had lower frozen-KITTI EPE than the historical controlled Hailo reference."
No deployed/hardware/universal/real-world/FPS/power superiority claimed. Both best and
final checkpoints favor ARM-P in 6/6 cells; seeds never pooled; 0.0444 px as context only
(pilot README's looser "SUPPORTS" wording superseded by closure/prereg — methodology
evolution, numbers unchanged). No significance / mechanism / generalization /
deployment-benefit claim. **ARM-P = VERIFIED within stated bounds.**

## 8. Identifiability status — PRESERVED

- Candidate C: **NOT IDENTIFIABLE** (6 routes fail, same-knob `disp*=sx`).
- GWC: **NOT IDENTIFIABLE** (op+width+sign/init bundle, non-equivalent ARM-W).
- True GWC: **OPEN BUT UNSUPPORTED — NOT REFUTED**.
- Post-identifiability boundary: **OPEN BUT UNSUPPORTED** (Decision A freeze+synthesize).
- Coverage vs pretraining: **NOT IDENTIFIABLE**.
No document upgrades these to FALSE/REFUTED/PROVEN. The four states (NOT IDENTIFIABLE /
NOT TESTED / OPEN BUT UNSUPPORTED / NOT REFUTED) remain distinct. **PASS.**

## 9. Stage C status overview

C0: seed 1 selected by predeclared lowest-EPE rule. Target UNKNOWN, toolchain absent
(see §§11/13/14). C1: FAIL (frozen). DR-0: risk, not explanation. DR-1: H1 FAIL,
Variant B not run. C2/C2.1/C2.1.1: all PASS as deterministic host post-processing.
Deployment: NOT DEPLOYABLE (target blocked). **Stage C substantive work = CLOSED;
deployment = BLOCKED (not failed).**

## 10. C0 — VERIFIED

Seed 1 `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth`,
SHA-256 `b2f6…eb7454` (re-measured this audit: MATCH), 1,621,341 B, 397,954 params /
70 tensors. Target NOT selected from historical artifacts. **PASS.**

## 11. C1 final status — FAIL (known, frozen)

- Frozen ONNX vs PyTorch pooled **max_abs_diff = 0.001708984375 px** > 0.001 criterion → **FAIL**.
  Re-measured values in `ARMP_DEPLOYMENT_VALIDATION.json` / parity / diagnostic reports agree exactly.
- §10 export-variant investigation (HISTORY, not repeated): BASE / V1-opset17 / V2-opset11 /
  V3-no-fold (976 nodes) / V4 — all pooled 0.001708984375, 194 discrepant px,
  cross-variant outputs bit-identical (max diff 0.0). Node counts ~690/712/976 yet identical.
  BASE re-export hash `4277090d…bcb6989` matches frozen ONNX (re-measured this audit: MATCH).
- Correct conclusion preserved: **C1 EXPORT-LEVEL MITIGATION NOT IDENTIFIED within the tested
  export space** — never "export can never fix it". **C1 = FAIL = VERIFIED.**

## 12. C1 decomposition — VERIFIED (new, consistent)

Δinitial **7.143e-4 px** (below threshold) / Δresidual **1.6899e-3 px** (carries failure) /
Δfinal **1.708984375e-3 px** (== frozen value exactly). Residual carries the failure;
near-tie soft-argmin behavior remains real but "soft-argmin is the entire cause" does NOT
appear as final claim. Remaining A (propagated initial sensitivity) vs B (independent
refinement-kernel accumulation divergence) = **NOT IDENTIFIED / NOT TESTED** (controlled
inference intervention outside frozen C1 scope). Consistent with earlier findings. **PASS.**

## 13. DR-0 / DR-1 status

- DR-0 (inference-only, no training/checkpoint change): ARM-P aggregated_cost ≈ 2.53e18
  (features 4.12e13, cost 7.08e13) vs P2A ≈ 6.16e9 vs reference 1e1–1e2. Growth progressive
  (cost→agg ARM-P ×3.58e4). Parity-magnitude correlations underpowered (ρ −0.169/+0.291,
  n=20). Dynamic range does NOT explain C1. "INT8 hostile" phrased as deployment
  risk / UNKNOWN target-native behavior, never as proof quantization will fail.
  **DR-0 = VERIFIED → DEPLOYMENT RISK.**
- DR-1 Variant A (`aggregated_cost/1e17`): five-scene max **1.4816284e-2 px** (all 5 > 1e-3,
  criterion A FIRED); 40-scene EPE delta **−4.7937e-6 px**, D1 unchanged → **H1 FAIL**.
  No model modification adopted; Variant B NOT RUN; no scaling search. **DR-1 = H1 FAIL
  (Variant A only) = VERIFIED.** No inference that all scalings fail.

## 14. C2 / C2.1 / C2.1.1 — all PASS (VERIFIED)

- C2: **135/135 PASS**, 40 scenes, Z=fB/d, two calibration groups, Q cross-check ≤4.6e-13 m,
  deterministic host post-processing, no new neural model. Record `c2_validation.json` on disk.
- C2.1: **251/251 PASS** (point cloud, nearest surface, spatial distance, occupancy, mask
  separation, determinism, depth unchanged). Dev-machine timing never presented as Hailo timing.
- C2.1.1: **19/19 PASS** (vertical/horizontal/constant/multi-region/poisoning/twin,
  hand-derived expectations, explicit √97 tolerance exception, KITTI sanity only, no GT-accuracy
  or algorithm-modification claim). `discontinuity.py` hash `2e81e9…1b5743` re-measured: MATCH.

## 15. Deployment-target status — UNKNOWN / BLOCKED (VERIFIED)

Historical only: v2.19.0 `supported_hw_arch=[hailo8]`; master `[hailo15h,hailo10h]`;
docs HAILO8; DFC 3.34.0 (Hailo-8 line) — all classified HISTORICAL REFERENCE, never promoted.
Final: **TARGET DEVICE = UNKNOWN; TARGET RESOLUTION = BLOCKED; no authoritative company
requirement exists.** No inference of Hailo-8/8L/10H/15H. **PASS.**

## 16. Toolchain status — NOT PERFORMED / UNKNOWN (VERIFIED)

This audit re-confirmed: `hailo/hailort/hailo_platform` unimportable, `hailortcli/hailo/dfc`
binaries absent, no `VEN_1E60` PCIe device (Windows 11 host; DFC requires Linux x86).
Therefore: Hailo compilation NOT PERFORMED, HEF NOT PRODUCED (for ARM-P), quantization
NOT TESTED, latency/power UNKNOWN. Correctly classified as untested-blocked, not failures.
**PASS.**

## 17. Deployment graph — VERIFIED

Stereo input → preprocessing → **ARM-P neural inference** (sole neural stage) → disparity →
metric depth → XYZ/point cloud → spatial perception → occupancy/nearest surface →
discontinuity → application layer. Everything after disparity is deterministic host-side
post-processing. No hidden neural stage. **PASS.**

## 18. Artifact hashes (re-measured this audit)

| Artifact | Expected SHA-256 | Measured | Verdict |
|---|---|---|---|
| ARM-P ckpt (seed1 `p2a_best.pth`) | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` | same | VERIFIED |
| ARM-P ONNX (`armp_stereonet.onnx`, 1,701,550 B, 712 nodes) | `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989` | same | VERIFIED |
| P2A ckpt (`phase2/runs/p2a_scale_coverage/p2a_best.pth`) | `0868ffd137a9985306bf5563685fd2362799bdd630d313e1181c980a7fbb6033` | same | VERIFIED |
| Reference ONNX (`reference/onnx/stereonet.onnx`, 23,690,586 B) | `b1a01d855bb22663f06dfd29eae11194e6fde952a319673e036f157c3e10095c` | same | VERIFIED |
| `discontinuity.py` | `2e81e9f2e97dbde196ba2311c68171aa4ce1195430b5c5b260d23b495e1b5743` | same | VERIFIED |

Unavailable: none among the five. P2A checkpoint hash matches readiness-record.
Known doc discrepancy (see §19.1): `evidence_summary.json:armp_onnx_sha256_spec` transposes
chars 13–15 (`dce` vs disk `cde`); disk authoritative — recorded, not repaired.

## 19. Git / working-tree status (pre-existing vs audit-created)

- `git log`: HEAD `58e8a19 docs: add the Phase 1 overview` (10 commits shown, all Phase-1 era).
- Pre-existing tracked modifications (NOT created by this audit, NOT touched by this audit):
  `M .gitignore`, `M src/models/stereonet/{__init__,cost_volume,regression,stereonet}.py`
  (113 insertions, 15 deletions — P2A architecture work predating audit).
- Pre-existing untracked (22 entries, incl. `phase0/`, `phase1/`, `phase2/`,
  `stage_a_diagnostics/`, `stage_b_armp/`, `stage_c_deploy/`, new tests/datasets) — project
  evidence accumulated outside git; reported, not cleaned/committed/deleted per §31.
- Created by this audit: ONLY `stage_c_deploy/FINAL_WHOLE_SYSTEM_AUDIT.md` (+ optional `.json`).
  No frozen source/model/evaluator changed by this audit; no historical record changed.
  **Experiment integrity: no training/fine-tuning/sweep/seed/tolerance/evaluator/benchmark
  change occurred during this audit (hash reads only).**

## 20. Documentation contradiction audit

Searched for: universal superiority, deployed-on-Hailo, Hailo-8/10H/15H-as-target,
C1-pass, DR-1-pass, range-solved, GWC/Candidate-C proven, SceneFlow-proven-causal,
deployment-complete, latency/power known, quantization-success, HEF-exists-for-ARM-P.

| # | Hit class | Disposition |
|---|---|---|
| 1 | ONNX sha transposition (`evidence_summary.json` spec `…dce…` vs disk `…cde…`, echoed in C2/C2.1/C2.1.1 §2 prompts + readiness §3) | DOCUMENTATION DISCREPANCY (disk authoritative; already self-recorded in evidence_summary) |
| 2 | Stale pilot `README.md:8-13` "NOT launched / control/armp/ do not exist" | STALE — documented in closure, preserved by rule |
| 3 | `run_arm.py:199` stale `seed:0` at seed1 / harness `EXPECTED_KEYS 72 / PARAMS 423586` flagging clean P2A (70/397954) | STALE/COSMETIC — documented (closure §10), operative guard passes |
| 4 | Resolution brief 375×1242 vs measured {(374,1238):10,(376,1241):30} | DOCUMENTATION DISCREPANCY — crop-only path still holds (C2 §6) |
| 5 | Rounding `1.709e-3` vs exact `0.001708984375` | COSMETIC |
| 6 | Instrumented-graph 1.92e-3 note vs frozen 1.708984375e-3 | EXPLAINED (ORT fusion perturbation; verdicts from unmodified graph; §10 shows equality with 5-output probe) |
| 7 | O11 INFERRED → VERIFIED progression; DeGirum "available" → ceased-2026-08-01/grace-2026-09-30 | SUPERSEDED — progression recorded, not contradiction |
| 8 | All Hailo-8/8L/10H/15H, FPS 10.7, HEF-exists, 8.22/10.3 mentions | HISTORICAL (reference-only) or CONDITIONAL (route descriptions) — zero prescriptive; never promoted to ARM-P target/claim |
| 9 | Any "ARM-P universally superior / deployed / C1 passes / DR-1 passes / GWC proven / SceneFlow causal / deployment complete / latency known" as current claim | NO UNSUPPORTED CURRENT CLAIM FOUND |

## 21. Claim audit (CLAIM | STATUS | EVIDENCE | SAFE TO SAY?)

| # | Claim | Status | Evidence | Safe? |
|---|---|---|---|---|
| 1 | ARM-P beats random-init P2A on frozen KITTI | VERIFIED (directional, 3 seeds, best+final) | tier2_eval.json ×3 + env control | YES (bounded) |
| 2 | Three-seed result reproducible | VERIFIED (replicated finding, not statistical proof) | same + Kaggle/local provenance | YES (bounded) |
| 3 | ARM-P beats historical controlled Hailo ref on frozen EPE | VERIFIED (1.198874 < 1.3134471) | §§7–8 | YES (EPE-only) |
| 4 | ARM-P better than Hailo hardware | UNKNOWN | no silicon run | NO |
| 5 | ARM-P deployed on Hailo | NOT PERFORMED | no HEF/compile | NO |
| 6–8 | Hailo-8 / 10H / 15H compatible | UNKNOWN/BLOCKED | target unknown, toolchain absent | NO |
| 9 | ONNX numerically identical to PyTorch | FAIL (measured 0.001708984375 > 0.001) | ARMP_DEPLOYMENT_VALIDATION.json | NO (state FAIL) |
| 10 | ONNX passes C1 | FAIL | same | NO |
| 11 | Export-only mitigation found | NOT IDENTIFIED (tested space) | §10 variants | NO |
| 12 | Dynamic range solved | NOT SOLVED (risk open) | DR-0/DR-1 | NO |
| 13 | Quantization succeeds | NOT TESTED | toolchain absent | NO |
| 14 | HEF exists (ARM-P) | NOT PRODUCED | same | NO |
| 15–16 | Hailo latency/power known | UNKNOWN | same | NO |
| 17 | Metric depth works | VERIFIED (135/135) | c2_validation.json | YES |
| 18 | Spatial perception works | VERIFIED (251/251) | c2_1_validation.json | YES |
| 19 | Discontinuity validation passes | VERIFIED (19/19) | c2_1_1…json + hash | YES |
| 20 | GWC demonstrated | NOT IDENTIFIABLE / OPEN BUT UNSUPPORTED | GWC definition | NO |
| 21 | SceneFlow isolated as unique causal mechanism | NOT IDENTIFIABLE | closure §§11–12 | NO |
| 22 | Deployment complete | BLOCKED (NOT DEPLOYABLE) | readiness-final | NO |

## 22. Open-branch audit (final states)

Architecture search CLOSED (none justified) · ARM-P CLOSED (bounded finding) ·
Coverage-vs-pretraining NOT IDENTIFIABLE · True GWC OPEN BUT UNSUPPORTED /
NOT IDENTIFIABLE · C1 export mitigation CLOSED — NOT IDENTIFIED ·
C1 refinement A-vs-B NOT IDENTIFIED / NOT TESTED · DR-1 Variant B NOT TESTED /
NOT AUTHORIZED · C2 CLOSED · C2.1 CLOSED · C2.1.1 CLOSED · Hailo target BLOCKED ·
Hailo compilation BLOCKED · Quantization BLOCKED · Hardware latency BLOCKED ·
Power BLOCKED · Confidence estimation NOT STARTED / NOT AUTHORIZED.
No branch silently disappears; no branch ambiguously open.

## 23. What we actually proved (evidence ladder)

- L1 DIRECTLY VERIFIED: 5 hashes (§18); P2A/reference contract numbers; C1 measured FAIL;
  C2 135/135, C2.1 251/251, C2.1.1 19/19 + determinism; target UNKNOWN; toolchain absent.
- L2 REPLICATED: ARM-P 3-seed directional benefit (best+final); env-control small shift.
- L3 STRONGLY SUPPORTED: supervision coverage as Stage-A candidate; fp32 accumulation-order
  amplification; refinement residual carrying C1 failure.
- L4 NOT IDENTIFIABLE / OPEN: coverage-vs-pretraining; true GWC; exact refinement A-vs-B.
- L5 UNKNOWN / BLOCKED: device, compiler compat, quantization, HEF, latency, power.

## 24. What we did NOT prove

Universal superiority · hardware superiority · Hailo deployment/compatibility ·
quantization/HEF/latency/power · SceneFlow as unique cause · coverage-vs-pretraining
separation · true GWC benefit · refinement causal source · universal real-world /
real-camera accuracy · production readiness. (All explicitly listed; none claimed.)

## 25. Remaining blockers

Company/owner decisions only: (1) target device part number; (2) resolution/accuracy/FPS/
power acceptance criteria; (3) hardware or farm access + schedule; (4) toolchain line
(DFC v3.x vs v5.x follows from (1)); (5) whether C1 FAIL is shippable or must gate;
(6) authorization for any of: Variant B, refinement-intervention, confidence estimation.

## 26. Project closure test

1. Model frozen? YES — seed-1 `p2a_best.pth` + hash. 2. Why selected? YES — predeclared
   lowest-EPE rule. 3. Achieved? YES — §§7,11–14. 4. NOT established? YES — §§24–25.
5–6. Rejected experiments + why? YES — §22 (Variant B / refinement intervention /
   confidence: not authorized / out of scope). 7–8. Deployment done vs impossible without
   input? YES — deterministic host stack done; silicon work blocked on target+toolchain.
9–10. Remaining failures + why unfixed? YES — C1 FAIL + DR-1 H1 FAIL frozen by audit-freeze
   rule, reported not repaired. 11. Reopen decision? YES — §25 + verdict below.
**CLOSURE TEST = YES → PROJECT CLOSURE READY.**

## 27. Second-pass verification (§34 checklist)

[x] No experiment run. [x] No frozen artifact changed (hashes match). [x] No historical
result overwritten. [x] No tolerance changed. [x] No target inferred. [x] No unsupported
claim promoted. [x] No UNKNOWN→PASS. [x] No NOT IDENTIFIABLE→FALSE. [x] No OPEN BUT
UNSUPPORTED→REFUTED. [x] C1/DR-1/C2/C2.1/C2.1.1 correctly classified. [x] Hailo deployment
correctly classified. [x] Hashes consistent. [x] Git changes accounted for (§19).
[x] All branches classified (§22). [x] Conclusion follows evidence.

## 28. Final verdict — category **B**

Material integrity holds (hashes, replication records, classification discipline);
remaining issues are documentation discrepancies (transposed sha spec, stale pilot text,
harness constants, resolution-brief vs measured, rounding, superseded notes) — all
self-recorded in-repo and none affecting the frozen verdicts. No evidence gap blocks
closure; no artifact-integrity failure exists.

---

## FINAL VERDICT

The project is internally coherent, frozen where required, correctly bounded in every
material claim, and blocked only where the evidence says blocked (Hailo target/toolchain/
silicon); known failures (C1 FAIL, DR-1 H1 FAIL) are preserved with exact numbers and the
remaining discrepancies are cosmetic and self-documented, so the project closes under
category B.

## WHAT IS CLOSED

- Phase 0, Phase 1, Stage A (no architecture justified), Stage B ARM-P (bounded finding)
- C0 selection, C1 (FAIL), DR-0 (risk), DR-1 Variant A (H1 FAIL), C2, C2.1, C2.1.1
- Architecture search, export-mitigation search (tested space), identifiability boundary

## WHAT IS VERIFIED

- 5 artifact hashes (ARM-P ckpt/ONNX, P2A ckpt, reference ONNX, discontinuity.py)
- ARM-P 3-seed directional benefit + env-control small shift (bounded, no significance claim)
- C1 FAIL at exactly 0.001708984375 px + residual-carries-failure decomposition
- C2 135/135, C2.1 251/251, C2.1.1 19/19 with determinism and hash stability
- Target UNKNOWN / toolchain absent as verified facts

## WHAT FAILED

- C1 export parity (0.001708984375 > 0.001; no mitigation in tested export space)
- DR-1 H1 (Variant A max 1.4816284e-2 px; 40-scene Δ −4.79e-6, D1 unchanged)

## WHAT IS UNKNOWN

- Target device, resolution decision, latency, FPS, power, quantization behavior,
  HEF outcome, real-camera/real-world accuracy, causal mechanisms (coverage-vs-pretraining,
  true GWC, refinement A-vs-B)

## WHAT IS BLOCKED

- Target resolution, Hailo compilation, HEF generation, quantization test, silicon
  latency/power measurement — all awaiting company input + toolchain/hardware

## WHAT WE ACTUALLY PROVED

- Bounded list in §23 (L1–L3); everything else explicitly not claimed

## WHAT WE DID NOT PROVE

- List in §24 (universal/hardware superiority, deployment, compatibility, quantization,
  HEF, latency/power, unique causality, GWC, real-world readiness)

## REMAINING DECISIONS

- Target part number; acceptance criteria; hardware/farm access + schedule;
  toolchain line; C1-FAIL shippability; authorization for Variant B /
  refinement intervention / confidence work — owner/company only

## ARTIFACT INTEGRITY

No frozen artifact changed: all five re-measured hashes match; C2-family records carry
matching before/after hashes; git shows only pre-existing modifications, nothing by this audit
except the two authorized deliverables.

## EXPERIMENT INTEGRITY

No experiment, training, fine-tuning, sweep, seed, tolerance, evaluator, or benchmark
change occurred during this audit (read + hash + status/diff/log + grep only).

## PROJECT CLOSURE STATUS

**PROJECT CLOSED WITH DOCUMENTATION DISCREPANCIES**
