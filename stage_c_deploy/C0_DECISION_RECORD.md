# STAGE C0 — DEPLOYMENT DECISION RECORD

This record freezes the Stage C deployment candidate and the state of every
Stage C entry condition. It is a documentation act only: it trains nothing,
evaluates nothing, exports nothing, compiles nothing, and modifies no model,
checkpoint, ONNX file or source file.

Evidence discipline (same as the contract): every factual line carries exactly
one of **VERIFIED** (established by a file in this repo or by a first-party
source already recorded in `docs/source_registry.md`, with the path cited),
**UNKNOWN** (not established, with what would establish it stated), or
**ASSUMED** (a working assumption with its basis and break condition stated).
Figures from outside this repo are additionally marked **EXTERNAL** with URL
and date. No number below was invented: every number comes from a cited repo
file. This worker has no network access.

Authoritative inputs, read before writing: the contract
(`stage_c_deploy/STAGE_C0_DEPLOYMENT_CONTRACT.md`), the inventory
(`stage_c_deploy/c0_checkpoint_inventory.json`), `ARMP_CLOSURE_RECORD.md`
Sections 4–7, `phase2/docs/PHASE2_DEPLOYMENT_SELECTION.md`,
`HAILO_TOOLCHAIN_REPORT.md` Sections 1, 2, 8,
`reference/hailo_model_zoo/stereonet.yaml` lines 40–50, and
`docs/research_questions.md` open questions O7–O17.

---

## 1. Candidate-selection rule (predeclared)

The rule, stated verbatim:

> *Select the ARM-P checkpoint with the lowest frozen-contract
> best-validation EPE among the three predeclared ARM-P seeds.*

- The rule was declared by the project owner BEFORE any ARM-P deployment
  outcome existed — no ARM-P export, parse, compile, or device run exists
  anywhere in the repo. **[VERIFIED:
  `stage_c_deploy/STAGE_C0_DEPLOYMENT_CONTRACT.md` Section 1 ("No ARM-P ONNX
  export exists yet"); `phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md`
  Section 0 (furthest verified point is ONNX, for P2A only).]**
- The rule mirrors the established P2A convention. **[VERIFIED:
  `phase2/docs/PHASE2_DEPLOYMENT_SELECTION.md` Section 4 ("Which checkpoint is
  deployed, and how it was chosen") — "Selection rule, stated before it was
  applied: the seed with the lowest frozen-contract EPE." P2A deployed seed 0
  (EPE 1.4149796) by that rule, with the single-seed reporting discipline
  (Section 4: the deployed figure "must be reported as the accuracy of *this
  artifact*, never as the method's expected accuracy").]**
- Three anti-gaming clauses apply to the rule's application, stated
  explicitly so the selection cannot be re-litigated on deployment grounds:
  1. the seed must not be switched because of deployment behaviour;
  2. the seed must not be selected because it compiles better;
  3. the seed must not be selected because it has better latency.
  **[VERIFIED: declared in this record by project-owner tasking, 2026-09-19;
  deployment behaviour of ARM-P is in any case UNKNOWN (no toolchain run
  exists), so none of the three clauses could yet have been violated.]**

---

## 2. Selected ARM-P seed

**Selected: seed 1.** The three predeclared seeds' frozen best EPEs side by
side, so the rule's application is auditable:

| Seed | Frozen best EPE (px) | Full precision (px) | Run |
|---|---|---|---|
| 0 | 1.2057590 | 1.2057590024628537 | `stage_b_armp/20260918T142128Z_tier2_pilot/` |
| **1** | **1.1912168** | **1.191216765057325** | `stage_b_armp/20260919T012646Z_tier2_seed1/` |
| 2 | 1.1996447 | 1.1996447345404795 | `stage_b_armp/20260919T092454Z_tier2_seed2/` |

- Seed EPEs. **[VERIFIED: quoted from
  `stage_b_armp/ARMP_CLOSURE_RECORD.md` Section 4 (seed 0), Section 5
  (seed 1), Section 6 (seed 2); NOT re-evaluated for this record.]**
- Seed 1 has the lowest of the three (1.1912168 < 1.1996447 < 1.2057590), so
  the Section 1 rule selects seed 1 deterministically. **[VERIFIED:
  arithmetic on the quoted values above.]**
- The selected figure 1.1912168 px is a single-seed figure and must be
  reported as the accuracy of *this artifact*, never as the method's expected
  accuracy — the same reporting discipline as the P2A precedent.
  **[VERIFIED (precedent):
  `phase2/docs/PHASE2_DEPLOYMENT_SELECTION.md` Section 4.]**

---

## 3. Exact frozen checkpoint

| Item | Value |
|---|---|
| Path | `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth` |
| SHA-256 | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` |
| Size | 1,621,341 bytes |
| Parameters / tensors | 397,954 / 70 |
| Strict load | clean (`strict_load_into_frozen_p2a_arch: true`) |

- Row contents (path, hash, size, param/tensor counts, strict-load outcome).
  **[VERIFIED: `stage_c_deploy/c0_checkpoint_inventory.json`, seed-1 ARM-P
  entry — read-only inventory (`torch.load(..., map_location="cpu")`, no
  re-save).]**
- This hash is now FROZEN for all of Stage C. Any Stage C artifact not
  traceable to `b2f6f5d5…fffeb7454` is out of contract. **[VERIFIED: declared
  in this record; the inventory hash is the traceability anchor, same rule as
  the P2A deployment selection (`PHASE2_DEPLOYMENT_SELECTION.md`
  Section 4).]**
- Disambiguation: the Kaggle seed-1 rerun
  (`stage_b_armp/20260919T130843Z_seed1_kaggle_envcontrol/`) is the
  ENVIRONMENT CONTROL and is NOT the candidate. **[VERIFIED:
  `stage_b_armp/ARMP_CLOSURE_RECORD.md` Section 7 — "This Kaggle seed-1 rerun
  is the ENVIRONMENT CONTROL. It is not a new seed, not seed 3 or 4, and
  never the seed-1 replication."; inventory note on the seed-1 entry records
  the same distinction.]**

---

## 4. Target Hailo device

**UNKNOWN.** No target device is chosen in this record, and none can be
inferred. The evidence sweep that produced this verdict:

- `reference/hailo_model_zoo/stereonet.yaml` lines 48–50 declare, verbatim:
  `supported_hw_arch:` / `- hailo15h` / `- hailo10h` — hailo8 is NOT listed
  (the file ends at line 50). **[VERIFIED:
  `reference/hailo_model_zoo/stereonet.yaml:48-50`.]**
- The published benchmark document is named
  `HAILO8_stereo_depth_estimation.rst` and reports Hailo-8 figures (float EPE
  8.22, hardware EPE 10.3, 10.7 FPS batch 1, 11.6 FPS batch 8).
  **[VERIFIED: `docs/source_registry.md:32` (SR-004).]**
- The compiled profiler report records `hw_arch: hailo8`. **[VERIFIED:
  `docs/hardware_analysis.md:209` (Section 4a model-level summary table).]**
- No file in this repository states a target device for ARM-P. A grep over
  the repo (excluding `reference/`) for device names (`hailo-8|hailo8|
  hailo-10|hailo10|hailo-15|hailo15` and variants) found device names only in
  reference-descriptive material, never as a project requirement:
  `docs/hardware_analysis.md:209` (the compiled REFERENCE report's `hw_arch`
  field), `:369` (the conflict recorded as UNKNOWN),
  `docs/reference_pipeline.md:217,248` (the conflict described),
  `docs/reproduction_report.md:186` (open defect D7, UNKNOWN),
  `docs/research_questions.md:63` (open question O11),
  `phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md:58-59,227-228`
  (descriptive, directing the user to confirm the target),
  `scripts/exp_profiler_report.py:80` and `scripts/hash_reference.py:24-25`
  (tooling/URLs for the reference artifacts),
  `results/profiler/` and `experiments/EXP-017|018/` (decoded reference
  profiler outputs), and the Stage C0 contract itself (descriptive). Nothing
  under `stage_b_armp/` names a target device. **[VERIFIED: grep performed
  for this record, 2026-09-19; per-file hits listed above.]**
- `docs/research_questions.md` records O11 ("Why does the config list
  hailo15h/hailo10h while the benchmark is Hailo-8?") as still open.
  **[VERIFIED: `docs/research_questions.md:63`.]**

Consequence: parser behaviour, quantisation, memory budget and every resource
number depend on the device, so C1 compilation cannot start until the target
is decided. **[UNKNOWN — establishing it requires the Section 5
decision.]**

---

## 5. Source of the target-device decision

**UNKNOWN — no source exists.** No company/project requirement, no project
documentation, no authoritative Hailo documentation held in this repo, and no
explicit project-owner decision currently supplies the target device for
ARM-P. **[VERIFIED: the Section 4 sweep found no nominating source; the
contract (`STAGE_C0_DEPLOYMENT_CONTRACT.md` Section 2) records the device as
UNKNOWN with the same basis.]**

The decision must come from one of: company/project requirements, project
documentation, authoritative Hailo documentation resolving the
yaml-vs-benchmark conflict, or an explicit project-owner decision — and none
of these currently supplies it. **[UNKNOWN.]** This is the blocking item for
all of device-dependent Stage C. **[VERIFIED: declared in this record.]**

---

## 6. Hailo SDK / DFC version for the target

**UNKNOWN**, because it is a function of the undecided device (Section 4).
What IS known:

- Nothing is installed on this host. **[VERIFIED:
  `phase2/deploy/validation/HAILO_COMPILATION_RECORD.json`,
  `toolchain_probe` — all 7 python modules not importable, all 6 binaries
  absent, no env vars, no install paths, `any_toolchain_found: false`; prose
  in `phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md` Sections 1–2.]**
- The reference artifacts came from Model Zoo v2.19.0 / DFC v3.x-era
  tooling. **[VERIFIED: `docs/source_registry.md:33-34` (SR-005, SR-006 rows,
  Model Zoo v2.19.0); `HAILO_TOOLCHAIN_REPORT.md:55` (v2.19.0 as the upstream
  measurement condition); DFC v3.x-era per the contract Section 3
  (EXTERNAL, dated 2026-09-19).]**
- Publicly referenced current versions: DFC v5.1.0, with v5.3.0 referenced
  for Hailo-10/15. **[EXTERNAL, dated 2026-09-19,
  https://hailo.ai/developer-zone/ — VERIFIED only as an external reference
  recorded in the contract (`STAGE_C0_DEPLOYMENT_CONTRACT.md` Section 3), not
  measured here.]**
- DFC is Linux-x86 only (Ubuntu 20.04/22.04/24.04); this host is Windows 11.
  **[VERIFIED (EXTERNAL, dated 2026-09-19) for the requirement —
  https://hailo.ai/developer-zone/; VERIFIED for this host:
  `HAILO_COMPILATION_RECORD.json`, `host.platform:
  Windows-11-10.0.26200-SP0`, and `HAILO_TOOLCHAIN_REPORT.md:45-47`.]**

The v3.x → v5.x gap is an unretired risk: whether the reference flow
reproduces under the current toolchain is itself open. **[UNKNOWN —
establishing it requires installing a current DFC and re-running the
reference parse, or finding first-party migration notes (contract
Section 3).]**

---

## 7. Latency / FPS requirement

**UNKNOWN — no requirement exists.** A sweep of `README.md`, `docs/` and
`phase2/docs/` for latency, FPS, throughput, and requirement language found
no latency, FPS or throughput requirement for ARM-P anywhere:

- `README.md`: the single latency/FPS hit (line 77) states the evidence gap —
  Hailo silicon behaviour UNKNOWN, the profiler report's model-level FPS and
  latency fields `N/A`. **[VERIFIED: `README.md:75-77`.]**
- `docs/`: hits are own-stack measurements (EXP-013/014 GPU/CPU timings),
  SOURCE reference figures (SR-004, SR-006), or UNKNOWN statements — never a
  requirement. **[VERIFIED: grep performed for this record, 2026-09-19.]**
- `phase2/docs/`: hits are "MACs are arithmetic, not latency" discipline
  statements, UNKNOWN-latency statements, and host/device separation rules —
  never a requirement. **[VERIFIED: grep performed for this record,
  2026-09-19.]**
- `stage_b_armp/`: no latency/FPS/throughput requirement; the only
  "requirement" hits are `requirements.txt` filenames and frozen-recipe
  wording (a training-recipe constraint, not a deployment target).
  **[VERIFIED: grep performed for this record, 2026-09-19.]**

Two reference figures exist and must NOT be mistaken for requirements:

- Hailo published 10.7 FPS batch 1 / 11.6 FPS batch 8 for the REFERENCE
  model on Hailo-8. **[VERIFIED (SOURCE, reference only):
  `docs/source_registry.md:32` (SR-004); `docs/hardware_analysis.md:28,359`.]**
- The post-placement profiler's modelled bottleneck of 43.03 FPS
  (`conv50`), which is a compiler estimate with model-level `fps` and
  `latency` both `N/A`. **[VERIFIED (SOURCE, reference only):
  `docs/hardware_analysis.md:194-197,209-218,256-293` (SR-006 via
  EXP-017/018); the ~4× gap to the 10.7 FPS published figure is a HYPOTHESIS
  (context-switch overhead), not a finding (`docs/hardware_analysis.md:
  287-293`).]**

Operative rule: **measurement ≠ requirement.** Stage C may measure latency
and FPS freely, but no measured figure may be called acceptable or
unacceptable until a real requirement exists. **[VERIFIED: declared in this
record, consistent with the contract Section 8 (no ARM-P latency target has
been set by anyone — UNKNOWN).]**

---

## 8. Power requirement

**UNKNOWN.** No power figure has ever been measured or required in this
project. **[VERIFIED: `docs/hardware_analysis.md:367` — "Power, FPS/W on any
device — UNKNOWN"; `docs/research_questions.md:62` (O10, open);
`docs/compute_profile.md:187` (no instrumentation available).]**

---

## 9. C1 entry conditions

Checklist as of this record:

| # | Condition | Status |
|---|---|---|
| 1 | Candidate seed selected and hash frozen | **MET** (Sections 1–3: seed 1, `b2f6f5d5…fffeb7454`) |
| 2 | Target device decided | **NOT MET** (Section 4 — UNKNOWN) |
| 3 | Toolchain version fixed for that device | **NOT MET** (Section 6 — UNKNOWN, function of 2) |
| 4 | A DFC installation exists | **NOT MET** (Section 6 — nothing installed) |
| 5 | Accuracy/latency requirements recorded (as values or as explicit UNKNOWN) | **MET, as explicit UNKNOWN** (Sections 7–8) |

The split, stated plainly:

- The **ARM-P ONNX export and its numerical parity test** are
  device-independent: they need only the frozen checkpoint (Section 3), the
  unchanged export scripts, and the host. They are **eligible to run** once
  the project owner authorises C1. **[VERIFIED: declared in this record; the
  export inputs are the frozen hash plus the P2A export path documented in
  the contract Section 4.]**
- **Hailo parsing, compilation, quantisation, resource analysis and any
  DeGirum silicon time** are device-dependent and are **BLOCKED** until
  Section 4 is resolved. **[VERIFIED: declared in this record; parser,
  quantiser, memory budget and performance targets all differ by device
  (contract Sections 2–3).]**

This record does NOT itself authorise the export; it states only that the
export is not blocked by the device UNKNOWN. **[VERIFIED: declared in this
record.]**

---

## 10. No-modification statement for C1

During C1 there is to be no retraining, no weight change, no architecture
change, no preprocessing change, and no change to the frozen KITTI evaluation
contract (40 scenes, 3,802,797 valid pixels, gt_scale 256.0, `disp_occ_0`).
**[VERIFIED: the contract terms are quoted from
`phase1/harness/frozen_eval.py` (CONTRACT) and
`stage_b_armp/ARMP_CLOSURE_RECORD.md` Section 8 (frozen contract held on all
scored checkpoints); the prohibition is declared in this record.]**

Any export shim, wrapper or compatibility patch must be recorded as a
separate artifact and must never be applied to the research source in place.
**[VERIFIED: declared in this record.]**

---

## Provenance

- Timestamp (UTC): 2026-09-19T15:40:30.3500260Z (this record's writing
  session). **[VERIFIED: measured at write time via system clock.]**
- Git HEAD: `58e8a19908ddbd35451652c61aef478f56b51ebd`. **[VERIFIED: quoted
  from `stage_c_deploy/c0_checkpoint_inventory.json` (`git_head`); matches
  `git rev-parse HEAD` observed in this session — NOT recomputed for this
  record.]**
- Host platform: Windows-11-10.0.26200-SP0. **[VERIFIED: quoted from
  `c0_checkpoint_inventory.json` (`host.platform`) — NOT re-probed.]**
- Software: python 3.12.9, torch 2.7.0+cu128, onnx 1.22.0, onnxruntime
  1.27.0. **[VERIFIED: quoted from `c0_checkpoint_inventory.json` (`host`
  block) — NOT re-measured; no code was run to obtain them.]**
- Frozen checkpoint hash: `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454`
  (seed 1, Section 3). **[VERIFIED:
  `stage_c_deploy/c0_checkpoint_inventory.json`.]**
- Statement: this record performed zero computation and modified no artifact.
  No training ran, no evaluation ran, no export ran, no ONNX file was touched,
  no compilation ran, no network access was used, and no existing file was
  modified — the only file created is `stage_c_deploy/C0_DECISION_RECORD.md`
  (this file). **[VERIFIED: git status check at close — see Report.]**

## Report

Files created (one, nothing else):

1. `stage_c_deploy/C0_DECISION_RECORD.md` (this file)

No existing file was modified.

---

## STOP

Stage C cannot proceed past the export/parity step until the project owner
supplies the target Hailo device (Sections 4–5), and optionally the
latency/FPS/power requirements (Sections 7–8) — the latter are not blocking,
because measurements can be collected without them, but no pass/fail
judgement can be made without them.
