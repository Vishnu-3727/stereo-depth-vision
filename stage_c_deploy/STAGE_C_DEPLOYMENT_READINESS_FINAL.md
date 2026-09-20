# STAGE C — DEPLOYMENT READINESS FINAL REPORT

Resumed audit; prior run died after C2 with no report written. This report is the deliverable.
Run date (UTC): 2026-09-20. Host: Windows 11, python 3.12.9, torch 2.7.0+cu128, numpy 2.5.1.
Frozen contract: `phase1/harness/frozen_eval.py` (authoritative, untouched).
Classifications used: CLOSED / VERIFIED / PASS / FAIL / UNKNOWN / BLOCKED / NOT IDENTIFIABLE / NOT TESTED / NOT APPLICABLE.

---

## 1. Executive status

Device-independent validation is complete and strong: C2 metric depth 135/135 PASS, C2.1
spatial perception 251/251 PASS, C2.1.1 discontinuity 19/19 PASS, all re-run this session
with exit 0 and frozen checkpoint/ONNX hashes unchanged before and after. The deployment
itself is BLOCKED on two frozen, unrepaired gates: C1 export parity FAIL (ARM-P ONNX vs
PyTorch max 1.7090e-3 px against the frozen 1e-3 criterion — a KNOWN FROZEN FAILURE, not a
regression) and DR-1 rescale H1 FAIL (1.4816e-2 px max per-pixel against the 1e-3 rescale
equivalence threshold). The company target device is UNKNOWN: no authoritative company
requirement for device, board, SDK/DFC version, resolution, FPS, power, or quantization
mode exists anywhere in the repo, so TARGET RESOLUTION = BLOCKED and every Hailo step
(toolchain, quantization, compilation, HEF, latency, power) is UNKNOWN or BLOCKED. No
training, no model/artifact modification, and no Hailo compilation occurred in this audit.

## 2. Frozen model identity

- Candidate: ARM-P seed 1, `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth`. VERIFIED
- Frozen best EPE (PyTorch, frozen contract, quoted not re-measured):
  1.1912168 px (full 1.191216765057325). Source: `stage_b_armp/ARMP_CLOSURE_RECORD.md`
  Section 5 via `stage_c_deploy/c0_checkpoint_inventory.json` seed-1 entry. VERIFIED
- 40-scene reproduction this audit's lineage: `stage_c_deploy/dr1_rescale/dr1_eval_40scene.json`
  original EPE 1.191216765057325 reproduces frozen 1.1912168 exactly
  (`original_reproduces_frozen: true`). VERIFIED

## 3. Artifact hashes

Measured this session with `Get-FileHash -Algorithm SHA256` (uppercase output; lowercase
spelling matches the frozen records):

| Artifact | Path | Size (bytes) | SHA256 | Match |
|---|---|---|---|---|
| ARM-P checkpoint | `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth` | 1621341 | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` | VERIFIED |
| ARM-P ONNX | `stage_c_deploy/armp_stereonet.onnx` | 1701550 | `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989` | VERIFIED |
| P2A checkpoint | `phase2/runs/p2a_scale_coverage/p2a_best.pth` | 1620893 | `0868ffd137a9985306bf5563685fd2362799bdd630d313e1181c980a7fbb6033` | VERIFIED |
| Reference ONNX | `reference/onnx/stereonet.onnx` | 23690586 | `b1a01d855bb22663f06dfd29eae11194e6fde952a319673e036f157c3e10095c` | VERIFIED |

No mismatch found; nothing repaired. The C2/C2.1/C2.1.1 validators also re-verified the
ARM-P checkpoint and ONNX hashes before AND after each run (all PASS). VERIFIED

DOCUMENTED DISCREPANCY (fixed nothing): `stage_c_deploy/target_resolution/evidence_summary.json`
field `armp_onnx_sha256_spec` and several task prompts carry the transposed spelling
`...deed1dce26...`; the on-disk `...deed1cde26...` is authoritative. VERIFIED

## 4. Model contract — COUNTED from the checkpoint, not quoted

Method: `torch.load` of the ARM-P checkpoint, `model = ck['model']` (OrderedDict),
`params = sum(v.numel())`, `tensors = len(model)`. Run this session:

- Parameter count: **397954**. VERIFIED (counted)
- State-dict tensors: **70**. VERIFIED (counted)
- Config from `stage_c_deploy/c0_checkpoint_inventory.json` architecture block:
  downsample_levels 3, num_disparities (candidates) 24, cost_volume_method subtract,
  cost_volume_shift **right**, regression_normalize **true**, aggregation_layers 4,
  residual_blocks 6, feature_channels 32. VERIFIED (inventory record; shift/normalize
  also corroborated by `c1_parity_diagnostic.json` provenance `config_both_models`:
  downsample_levels 3, num_disparities 24, cost_volume_shift right,
  regression_normalize true, and by the C2 record's `StereoNetConfig` string).

## 5. C1 export/parity status — FAIL (known frozen failure)

Source: existing artifacts `stage_c_deploy/C1_EXPORT_PARITY_REPORT.md`,
`stage_c_deploy/diagnostics/c1_parity_diagnostic.py` / `.json`. Re-execution of the
full export+parity was impractical in this audit window (requires GPU export runs);
existing artifacts are quoted and labelled REPRODUCTION UNAVAILABLE (reason: export
scripts not re-run this session; parity numbers below are quoted verbatim from the
frozen JSONs, never re-measured here). No number fabricated.

- Criterion (frozen, inherited from `phase2/scripts/export_p2a.py:106-107`):
  max abs diff < 1e-3 px on 5 scenes. Tolerance NEVER loosened. VERIFIED
- ARM-P ONNX vs PyTorch: pooled max **1.708984375e-03 px** (worst scene 000164_10 at
  1.7090e-3) → **FAIL**. Source: `ARMP_DEPLOYMENT_VALIDATION.json`
  (`max_abs_diff_px = 0.001708984375`, `pass: false`). KNOWN FROZEN FAILURE, not a
  regression. VERIFIED (quoted)
- P2A ONNX vs PyTorch: max 6.1798e-04 px → **PASS**. Source:
  `phase2/deploy/P2A_DEPLOYMENT_VALIDATION.json` via C1 report Section 5. VERIFIED (quoted)
- Localisation (recorded): static PyTorch graph vs trained PyTorch graph PASS at
  5.340576171875e-05 px; static ONNX vs original ONNX PASS at same value — the
  ~1.7e-3 spike enters at the torch→ONNX export step for the ARM-P weights. Why the
  exporter noise is larger on ARM-P weights is UNKNOWN. VERIFIED (quoted from
  `ARMP_STATIC_EXPORT_VALIDATION.json`)
- P2A precedent: static-vs-trained 3.05e-5 px; deployed-ONNX-vs-PyTorch EPE delta
  9.52e-5 px. VERIFIED (quoted, C1 report Section 0)

## 6. Dynamic-range status — extreme activation range CONFIRMED

Sources: `stage_c_deploy/diagnostics/activation_range.json` (utc 2026-09-19T16:35:32Z),
`stage_c_deploy/dr0_audit/dr0_activation_audit.json` (utc 2026-09-20T01:44:43Z). Quoted.

- ARM-P `aggregated_cost` max abs ≈ 2.13e18 (scene 000160_10; order ~10^18 across
  scenes, e.g. 1.57e18 scene 161). Rescaled variant (÷1e17) ≈ 15.7–21.3: this is the
  DR-1 probe, NOT a model change. VERIFIED (quoted from dr1_rescale_gate.json
  five_scene per-scene `agg_max_abs_original` / `agg_max_abs_rescaled`)
- Reference ONNX operates at ≈10^1–10^2 scale (per audit brief; reference instrumented
  debug copy `diagnostics/reference_instrumented_debug.onnx` sha
  `5846fc9c…0093441` per DR0 provenance). VERIFIED (quoted provenance)
- Classification (CORRECTED 2026-09-20 by the documentation closure pass):
  **DEPLOYMENT RISK — TARGET-NATIVE BEHAVIOUR UNKNOWN.** The model itself is frozen and
  unmodified; no activation rescaling was applied. Breakdown:
  **MEASURED** — ARM-P carries extremely large fp32 intermediate values (`aggregated_cost`
  max abs ~1.57e18–2.53e18) against a reference operating at ~1e1–1e2 (DR-0, VERIFIED).
  **ESTABLISHED** — the DR-1 Variant-A activation rescale FAILED H1 (§7), so that mitigation
  is rejected. **UNKNOWN** — behaviour under target-native Hailo quantization/compilation;
  no quantization was attempted and `DYNAMIC_RANGE_AUDIT.md` §15 predicts no outcome.
  **TARGET** — still unspecified (§12).
  *Superseded wording, preserved for the record:* this line previously read
  "Classification: FAIL as a deployment property (INT8 quantization hostile until proven
  otherwise)". No predeclared target-native dynamic-range gate or threshold was ever
  established, so "FAIL" overstated the evidence. The measured findings are unchanged;
  dynamic range is NOT claimed safe and NOT claimed to be a confirmed Hailo failure.
- C1 tolerance restated unchanged 0.001 in DR0 provenance; C1 verdict unchanged FAIL.

## 7. DR-1 status — H1 FAIL

Source: `stage_c_deploy/DR1_RESCALE_GATE.md` + `dr1_rescale/dr1_rescale_gate.json`
(five_scene) + `dr1_rescale/dr1_eval_40scene.json`. Quoted.

- Intervention: Variant A only — `aggregated_cost / s`, s = 1e17, same
  regression/refinement. Variant B NOT performed. VERIFIED
- Five-scene max |delta| per scene (px): 160: 0.0032158; 161: 0.0066586; 162:
  0.0055923; 163: 0.0037632; 164: 0.0148163. Pooled max **1.4816284e-02 px ≈ 15x**
  the 1e-3 DR-1 rescale equivalence threshold, exceeded on ALL scenes →
  rejection criterion A fired → **H1 FAIL**. VERIFIED (gate JSON + gate report §9)
- 40-scene EPE: original 1.191216765057325 vs rescaled 1.191211971353279; global delta
  **-4.793704046157643e-06** (D1 delta 0.0). The global EPE is insensitive; the
  per-pixel max rejects H1. VERIFIED (dr1_eval_40scene.json)
- C1 verdict unchanged FAIL; checkpoint byte-identical; rescale NOT adopted (probe
  only, no model edit). Whether the rescale affects INT8/Hailo behavior is UNKNOWN —
  no such test authorized or performed. VERIFIED

## 8. C2 metric-depth status — PASS (re-run this session)

Record: `stage_c_deploy/metric_depth/out/c2_validation.json` (utc 2026-09-20T06:06:49Z,
device cuda). Prior run's 11:36 local re-run quoted; this session verified the record
on disk (135 tests, FAILS []).

- **135/135 PASS, failures [], exit 0.** VERIFIED (record + test-name/FAIL enumeration)
- Covers: frozen checkpoint unchanged (before+after), params 397954, 40-scene
  hailo_val split, disparity units, resolution scaling, per-scene (160–199) accuracy
  preservation (new path vs `model()`), determinism (bitwise identical), calibration
  source, fx==fy, depth equation Z=fB/d, finiteness, XYZ reprojection, Q-matrix
  cross-checks, visualizations, invalid-disparity handling, ONNX hash (disk spelling,
  prompt typo noted). VERIFIED
- Hand pixel: scene 000162_10 (u,v)=(616,184), d=10.8078 px, Z=35.2354 m,
  X=0.7657 m, Y=0.1220 m. VERIFIED (record `hand_pixel`)

## 9. C2.1 spatial-perception status — PASS (re-run this session)

Command: `python stage_c_deploy/spatial_perception/validate_spatial.py`.
Record: `stage_c_deploy/spatial_perception/out/c2_1_validation.json`
(utc 2026-09-20T06:09:50Z, cuda).

- **251/251 PASS, overall PASS.** VERIFIED (validator stdout tail:
  `C2.1 validation: 251/251 passed; overall=PASS`)
- Frozen checkpoint sha `b2f6f5d5…eb7454` unchanged (after); ONNX disk spelling
  `4277090deed1cde26…` match (after, prompt typo noted). VERIFIED
- Timing/memory figures in the record are DEVELOPMENT-MACHINE MEASUREMENTS
  (infer ≈0.145 s/frame cuda; spatial ≈0.086 s/frame), never Hailo performance. NOT APPLICABLE to Hailo.

## 10. C2.1.1 discontinuity status — PASS (re-run this session)

Command: `python stage_c_deploy/spatial_perception/validate_discontinuity.py`.
Record: `stage_c_deploy/spatial_perception/out/c2_1_1_discontinuity_validation.json`
(utc 2026-09-20T06:10:03Z).

- **19/19 PASS, overall PASS.** VERIFIED (validator stdout tail:
  `C2.1.1 discontinuity validation: 19/19 passed; overall=PASS`)
- `discontinuity.py` sha256 before == after ==
  `2e81e9f2e97dbde196ba2311c68171aa4ce1195430b5c5b260d23b495e1b5743`. VERIFIED
- Real scene 000160_10: shape (368,1232), 451777 valid mag pixels, fraction with
  mag>1.0 m/px = 0.0979 (DESCRIPTIVE, not a threshold); C2 maxdiff 0.0. VERIFIED
- Discontinuity-only cost ≈27.8 ms/frame is a DEVELOPMENT-MACHINE MEASUREMENT, never
  Hailo performance. NOT APPLICABLE to Hailo.

## 11. Complete deployment graph

Input geometry everywhere: KITTI 368×1232 (`src/datasets/kitti2015.py` TARGET_H/W).
NEURAL stage = ARM-P only. Everything else is deterministic host code.

| # | Stage | Implementation path | In → Out (shape, dtype) | Frozen / variable | Neural / deterministic-host | Deterministic | HW dependency | Validation |
|---|---|---|---|---|---|---|---|---|
| 1 | Input load | `src/datasets/kitti2015.py` Kitti2015Stereo (hailo_val 40 scenes) | PNG stereo → uint8 HWC 368×1232×3 ×2 | Frozen | Host | Yes | None | C2 PASS (split/scenes) |
| 2 | Preprocessing / crop-pad + ImageNet normalize | `src/datasets/kitti2015.py:155-165` normalize; host-side, NOT in ONNX graph | uint8 HWC → float32 NCHW [1,3,368,1232] ×2 | Frozen | Host | Yes | None | C1 §4 VERIFIED |
| 3 | ARM-P inference (NEURAL — only neural stage) | ckpt `…/armp/p2a_best.pth` (sha b2f6f5d5…) → ONNX `armp_stereonet.onnx` (sha 4277090d…, opset 13, 712 nodes) | [1,3,368,1232]×2 float32 → disparity [1,1,368,1232] float32, full-res px | Frozen | NEURAL | Yes (bitwise-identical reruns, C2/C2.1) | Target UNKNOWN | C1 FAIL (parity); EPE frozen 1.1912 |
| 4 | Disparity | ONNX output tensor `disparity` | [1,1,368,1232] float32 px | Frozen | Host (passthrough) | Yes | None | C2 PASS |
| 5 | Metric depth | `stage_c_deploy/metric_depth/metric_depth.py`: Z=fB/d, d>1e-3 mask | disparity → depth (368,1232) float + invalid mask | Frozen | Host | Yes | None | C2 135/135 PASS |
| 6 | XYZ / point cloud | `spatial_perception/pointcloud.py`: X=(u-cx)Z/fx, Y=(v-cy)Z/fy; Q-matrix cross-checked | depth → points (~453k×3 float32, ~12.7 MB/scene) | Frozen | Host | Yes (bitwise-identical) | None | C2.1 PASS |
| 7 | Spatial / occupancy | `spatial_perception/spatial.py`, `occupancy.py`; 60 m physical cap is a separate mask only, never applied by default | points → spatial cells + occupancy | Frozen | Host | Yes | None | C2.1 251/251 PASS |
| 8 | Discontinuity | `spatial_perception/discontinuity.py` (sha 2e81e9f2…1b5743) gradient-magnitude + mask | depth → mag (368,1232) float32 + bool mask | Frozen | Host | Yes (equal_nan bitwise) | None | C2.1.1 19/19 PASS |
| 9 | Application | No application code exists in repo | N/A | N/A | N/A | N/A | Target UNKNOWN | NOT APPLICABLE |

## 12. Target-device evidence audit

Method: repo `git grep` for device/requirement terms + prior dedicated full-repo greps
recorded in `stage_c_deploy/C1_TARGET_TOOLCHAIN_RESOLUTION.md` (§A–B) and
`stage_c_deploy/target_resolution/evidence_summary.json` (`repo_search`). Every hit
classified 1–5. This session's `git grep -in target` confirms: hits are generic prose
("targets real time", "target silicon" as aspiration), profiler CSV column names
(`*_per_target` = compiler placement fields), `TARGET_H/W` (image geometry), loss-arg
names — zero prescriptive company requirements.

| Hit | Path + line | Class |
|---|---|---|
| v2.19.0 stereonet.yaml `supported_hw_arch: [hailo8]` | external, fetched live 2026-09-20 (evidence_summary O11) | 3 HISTORICAL REFERENCE — must NOT be promoted to target |
| master stereonet.yaml `supported_hw_arch: [hailo15h, hailo10h]` | external, fetched live 2026-09-20 | 3 HISTORICAL REFERENCE — must NOT be promoted |
| `stereonet.alls` directives | reference/ (master) | 3 HISTORICAL REFERENCE |
| `HAILO8_stereo_depth_estimation.rst` figures (EPE 8.22→10.3, 10.7/11.6 FPS) | reference docs (master legacy HAILO8/ subtree) | 3 HISTORICAL REFERENCE |
| `stereonet.hef` 24,057,165 B, header string `3.34.0` | reference Compiled/v2.19.0/hailo8/ | 3 HISTORICAL REFERENCE (DFC 3.34.0 corroborated as Hailo-8 line current by community staff post 2026-09-10) |
| Profiler HTML `hw_arch: hailo8`, 242 layers 8/8/8 | reference Compiled/v2.19.0/hailo8/ | 3 HISTORICAL REFERENCE |
| hailo-apps disparity-only 8-bit app | external reference | 3 HISTORICAL REFERENCE |
| Model Zoo master README branch pairing (v2.x+DFC3.x for 8/8L; master for 10/15; DFC v5.4.0/HailoRT v5.4.0) | external, fetched 2026-09-20 | 3 HISTORICAL REFERENCE |
| "Target accelerator hardware" (demo README future-work placeholder, no device named) | repo demo docs (per C1_TARGET_TOOLCHAIN_RESOLUTION §A) | 4 DESCRIPTIVE EXAMPLE |
| `TARGET_H, TARGET_W = 368, 1232` | `src/datasets/kitti2015.py:34` | 4 DESCRIPTIVE EXAMPLE (image geometry constant, not a device) |
| `hailo_val` split name (40 scenes 160–199) | `src/datasets/kitti2015.py`, frozen contract | 4 DESCRIPTIVE EXAMPLE (dataset split label, carries no device meaning) |
| Profiler `*_per_target` CSV columns | `results/profiler/model_summary.csv:1` | 4 DESCRIPTIVE EXAMPLE (compiler placement fields of the reference report) |
| "Measure on target silicon…" (aspirational prose) | `PHASE_1_FINAL_REPORT.md:294`, `docs/*.md` | 4 DESCRIPTIVE EXAMPLE |
| hailo_toolchain_available: False / NOT VERIFIED strings | `stage_c_deploy/export_armp.py:227-229` | 2 DEPLOYMENT IMPLEMENTATION EVIDENCE (absence recorded in code) |
| DeGirum farm Hailo-8/8L availability + 2026-09-30 grace end | external, accessed 2026-09-20 | 2 DEPLOYMENT IMPLEMENTATION EVIDENCE (route, not a requirement) |
| Any company device/board/SDK/DFC/resolution/FPS/power/quantization requirement | nowhere in repo | 1 AUTHORITATIVE TARGET REQUIREMENT: NONE FOUND |

Verdict: **TARGET DEVICE = UNKNOWN. TARGET RESOLUTION = BLOCKED.** This is a
SUCCESSFUL audit outcome. Selecting a target by inference (reference used hailo8),
convenience, or branch default is FORBIDDEN and was not done. The known non-authoritative
history above (v2.19.0→hailo8; master→hailo15h/hailo10h; docs→HAILO8; DFC 3.34.0) is
preserved as history, never promoted.

## 13. Toolchain evidence audit

- Binaries on PATH: hailortcli absent, hailo absent, dfc absent. Python modules
  hailo_sdk_client / hailo_platform / hailort / hailo / hailo_model_zoo /
  hailo_sdk_common not importable. Source: evidence_summary.json `toolchain_on_host`.
  → **No Hailo toolchain on this host. UNKNOWN.**
- Local Hailo hardware (PCIe): NONE. VERIFIED (prior scan; no change record).
- DeGirum Device Farm (Hailo-8/8L silicon): route AVAILABLE but DYING — ceased
  operations 2026-08-01, assets to LatticeWork; AI Hub free-Pro grace through
  2026-09-30, then publicly unavailable. No free-silicon route for 10/15. Quoted
  from evidence_summary.json (REPRODUCTION UNAVAILABLE — external URLs not re-fetched
  this session; one citation already dead as of 2026-09-20: community.degirum.com DNS failure).
- DFC version selection: FORBIDDEN without target. Known lines only: v3.x (v3.34.0
  current Hailo-8 line) vs v5.x (v5.4.0 master/10/15 line) — historical reference only.
- Calibration set for future quantize step: hailo_calib scenes 0–159 known, disjoint
  from eval 160–199 (contract §6); outcome UNKNOWN (run required, BLOCKED on target).

## 14. Deployment risk register

| ID | Risk | Status | Evidence | Impact | What closes it | Actionable now |
|---|---|---|---|---|---|---|
| R1 | Target device unknown | UNKNOWN | §12: zero authoritative requirements | All Hailo work blocked; wrong-guess HEF is scrap | Company names device/part (Q1) | No — external input needed |
| R2 | C1 export parity FAIL (1.709e-3 > 1e-3) | FAIL (frozen) | `ARMP_DEPLOYMENT_VALIDATION.json` pass:false; C1 report §5 | ONNX not a faithful copy of the frozen weights at tolerance; no downstream number meaningful per C1 spec | Exporter-level investigation + re-export passing 1e-3 (model untouched; tolerance frozen) | Yes — investigation authorized, no fix applied here |
| R3 | Extreme activation range (~1e18 vs ref 1e1–1e2) | **DEPLOYMENT RISK — TARGET-NATIVE BEHAVIOUR UNKNOWN** (corrected 2026-09-20; previously "FAIL (property)") | DR0/activation_range JSONs; §6 | INT8 quantization hostile until proven otherwise; Hailo parse/accuracy risk | Quantization experiment on target toolchain showing accuracy preserved, or target-native evidence | No — needs toolchain (R1) |
| R4 | DR-1 H1 FAIL (rescale not equivalent, 1.48e-2 max) | FAIL | DR1 gate JSON + report; §7 | The obvious mitigation (÷1e17) is rejected; range problem has no accepted host-side fix | A new, passing intervention proposal (Variant B or other) with predeclared criteria | Yes — can propose, not apply |
| R5 | Quantization behavior UNKNOWN | UNKNOWN | No quantize run exists for ARM-P | Unknown accuracy drop, unknown calibration outcome | hailomz optimize run on target DFC + accuracy eval | No — blocked on R1 |
| R6 | Compiler compatibility UNKNOWN | UNKNOWN | No parse/compile attempted (toolchain absent) | Unknown whether 712-node opset-13 graph parses for the target | hailomz parse on target DFC | No — blocked on R1 |
| R7 | HEF UNKNOWN | UNKNOWN | No .hef produced | Nothing deployable exists | Successful compile for the named target | No — blocked on R1 |
| R8 | Latency UNKNOWN | UNKNOWN | Dev-machine timings only (C2.1/C2.1.1 notes) | No FPS/latency claim possible; never present dev timing as Hailo | Silicon or authorized-emulator measurement | No — blocked on R1 |
| R9 | Power UNKNOWN | UNKNOWN | `docs/hardware_analysis.md:367`, O10 open | No power budget compliance claim | Target + board + measurement | No — blocked on R1 |
| R10 | Camera/stereo input spec UNKNOWN | UNKNOWN | No company camera interface on record | Geometry/resolution/FPS/baseline of the real rig unconfirmed (KITTI 368×1232 is eval, not product) | Company camera/stereo spec | No — external input needed |

No UNKNOWN was converted to PASS. Nothing above is inference.

## 15. Device-independent artifacts prepared

- ARM-P ONNX (`armp_stereonet.onnx`, 712 nodes, opset 13, checker pass) + static
  variant (`armp_stereonet_static.onnx`, 689 nodes). CLOSED (built, parity FAIL noted)
- C1 recipe proof (`c1_recipe_diff.json`, NO DEFECT). VERIFIED (quoted)
- C2 metric-depth module + 135/135 record + 2 visualizations. PASS
- C2.1 spatial module (pointcloud/spatial/occupancy) + 251/251 record + 2 PNGs. PASS
- C2.1.1 discontinuity module (frozen sha) + 19/19 record. PASS
- DR0 activation audit + C1 parity diagnostic + DR-1 gate records. CLOSED
- Target/toolchain resolution records (`C1_TARGET_TOOLCHAIN_RESOLUTION.md`,
  `target_resolution/evidence_summary.json`). CLOSED
- This report + JSON companion. CLOSED

## 16. Target-dependent work blocked

BLOCKED until the company answers (§17): Hailo toolchain install/DFC selection,
hailomz parse/optimize/compile, quantization mode + calibration run, HEF build,
runtime integration, latency/FPS measurement, power measurement, camera-interface
validation, any silicon or emulator claim. No emulator-only characterization was
performed or authorized.

## 17. Exact company information required

All fields currently `UNKNOWN / NOT SPECIFIED`:

1. Target device / part number (e.g. Hailo-8 / 8L / 10H / 15H — company's word, not ours)
2. Target board / module / SOM + carrier
3. Required SDK / HailoRT version
4. Required DFC / compiler version
5. Runtime environment (host SoC/CPU, OS, HailoRT language binding)
6. Input resolution (product geometry; KITTI 368×1232 is evaluation, not product)
7. FPS / latency requirement (per-stage or end-to-end, batch size)
8. Power limit (device and/or board level)
9. Memory limit
10. Physical hardware available to the team? (loaner / purchase / remote access)
11. Emulator-only characterization acceptable? (yes/no as interim evidence)
12. Quantization mode required (INT8 post-train / QAT / mixed / per-layer constraints)
13. HEF output + runtime expectations (single context? multi-context? batching?)
14. Camera / stereo input spec (sensor, baseline, sync, interface, ISP, rectification)

## 18. Final integrity audit

Each item ticked only with pasted evidence:

- [x] C2 re-run 135/135 exit 0 — `C2.1 validation` analog: c2 record has 135 tests,
  `FAILS: []` (enumerated this session); utc 2026-09-20T06:06:49Z; checkpoint/ONNX
  sha before==after inside the record.
- [x] C2.1 re-run 251/251 exit 0 — stdout: `C2.1 validation: 251/251 passed;
  overall=PASS`; record utc 2026-09-20T06:09:50Z.
- [x] C2.1.1 re-run 19/19 exit 0 — stdout: `C2.1.1 discontinuity validation: 19/19
  passed; overall=PASS`; discontinuity sha before==after
  `2e81e9f2e97dbde196ba2311c68171aa4ce1195430b5c5b260d23b495e1b5743`.
- [x] ARM-P checkpoint hash — measured `B2F6F5D558DBD2676D740CA55A0B60900C89FF2225F0445E98C59D6FFFEB7454`
  (1621341 B) == expected `b2f6f5d5…eb7454`.
- [x] ARM-P ONNX hash — measured `4277090DEED1CDE26DDB2A470D8DC097B26D5AD45A931E0AABF3635DBBCB6989`
  (1701550 B) == expected `4277090d…bcb6989`.
- [x] P2A checkpoint hash — measured `0868FFD137A9985306BF5563685FD2362799BDD630D313E1181C980A7FBB6033`
  (1620893 B) == expected `0868ffd1…bb6033`; path `phase2/runs/p2a_scale_coverage/p2a_best.pth`.
- [x] Reference ONNX hash — measured `B1A01D855BB22663F06DFD29EAE11194E6FDE952A319673E036F157C3E10095C`
  (23690586 B) == expected `b1a01d85…10095c`; path `reference/onnx/stereonet.onnx`.
- [x] Model contract counted — `n_tensors: 70`, `params: 397954` (torch.load,
  `ck['model']`, this session).
- [x] C1 ARM-P FAIL preserved — `max_abs_diff_px = 0.001708984375`, `pass: false`
  (quoted, REPRODUCTION UNAVAILABLE); tolerance 1e-3 untouched.
- [x] C1 P2A PASS preserved — 6.1798e-04 px (quoted, REPRODUCTION UNAVAILABLE).
- [x] DR-1 H1 FAIL preserved — pooled max 1.4816284e-02 px vs 1e-3 threshold;
  40-scene EPE delta -4.793704046157643e-06 (quoted JSONs).
- [x] Target UNKNOWN declared — §12 table; zero class-1 hits; forbidden inference
  not performed.
- [x] No banned language — report uses only the allowed classifications.
- [x] Scope respected — only new files under `stage_c_deploy/`; validators rewrote
  only their own `out/*.json` records (expected). No checkpoint/ONNX/src/validator/
  module/.gitignore/historical-record modification. No commits. (Pre-existing
  unrelated working-tree modifications noted in §19 were not made by this audit.)

## 19. Final state classification

- ARM-P checkpoint: VERIFIED (frozen, hash-stable across all three re-runs)
- ARM-P ONNX: VERIFIED as artifact / FAIL as faithful copy (C1 parity)
- C1 export/parity: FAIL (known frozen failure; tolerance intact)
- Dynamic range: **DEPLOYMENT RISK — TARGET-NATIVE BEHAVIOUR UNKNOWN** (measured ~1e18 activations; DR-1 Variant-A mitigation FAILED H1; Hailo-native behaviour untested). *Corrected 2026-09-20; previously read "FAIL (property; ~1e18 activations)" — see §6.*
- DR-1 rescale: FAIL (H1 rejected; probe not adopted)
- C2 metric depth: PASS (135/135, re-run)
- C2.1 spatial perception: PASS (251/251, re-run)
- C2.1.1 discontinuity: PASS (19/19, re-run)
- Target device: UNKNOWN → target resolution BLOCKED
- Toolchain / quantization / HEF / latency / power / camera: UNKNOWN or BLOCKED
- Hailo deployment: BLOCKED (on target decision + C1 FAIL + R3/R4)
- Overall: **NOT DEPLOYABLE on any Hailo target at this time. Device-independent
  validation is complete; every target-dependent step is blocked on company input.**

Note on working tree: `git status --short` shows pre-existing modifications
unrelated to this audit (`.gitignore`, `src/models/stereonet/*`, plus untracked
phase/stage directories and test files). This audit modified no tracked file and
created only the two files in scope.
