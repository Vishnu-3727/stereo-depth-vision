# STAGE C1 — TARGET DEVICE / HAILO TOOLCHAIN RESOLUTION (POST-DR1)

Contract-resolution task. No training, no model change, no compilation, no
benchmarking was performed. Every conclusion is labelled **VERIFIED**,
**INFERRED**, **UNKNOWN** or **NOT APPLICABLE**. Uncertainty is not hidden.

Date (UTC): 2026-09-20. Network access: **YES** (webfetch + websearch functional;
all external claims carry URL + access date). Git HEAD at time of writing:
`58e8a19` series (same head recorded in the blocker report; no files other than
the two named in Report were created or modified).

---

## 1. EXECUTIVE STATUS

**No authoritative deployment target was found.** VERIFIED by an independent
repository search (§2) that confirms the prior report: every Hailo/device
occurrence in this repository is descriptive of the historical reference or a
record of the UNKNOWN itself. Zero occurrences are prescriptive. Target device,
DFC/SDK version, latency/FPS/power requirements all remain **UNKNOWN**.

This report therefore pursues **SUCCESS condition B**: prove the absence
sufficiently and produce an explicit decision request (§13).

New work beyond the prior report (all VERIFIED, dated 2026-09-20):

1. **O11 resolved from INFERRED to VERIFIED.** The prior report left unchecked
   the v2.x-branch copy of `stereonet.yaml` for lack of authorized network
   access. Fetched live today: v2.19.0 lists `supported_hw_arch: [hailo8]`;
   master lists `[hailo15h, hailo10h]`. The yaml-vs-benchmark "conflict" is a
   branch-scoping artifact, not a contradiction (§3, §6).
2. **DFC 3.34.0 corroborated.** The HEF-header string `3.34.0` (previously
   INFERRED to be the DFC version) is now corroborated by a Hailo community
   staff post (2026-09-10, corrected from 2026-09-09 recorded during the run —
   see source 4) listing DFC v3.34.0 as the current Hailo-8/8L line.
3. **DeGirum status worsened.** DeGirum ceased operations 2026-08-01 (assets
   acquired by LatticeWork); the AI Hub free-Pro grace period ends 2026-09-30,
   after which it is unavailable publicly. Ten days remain. The farm covers
   Hailo-8/8L only and its Cloud Compiler cannot compile our ONNX in any case.
4. **Artifact integrity re-verified.** All three frozen hashes re-measured;
   the ONNX spec-transposition discrepancy is recorded explicitly (§14).
5. **Toolchain absence re-verified on this host** (6 modules, 3 binaries).

Prior-report claims confirmed vs not confirmed: §16 and `target_resolution/`.

---

## 2. AUTHORITATIVE TARGET EVIDENCE

**There is none. VERIFIED.**

Method: full-repo content search (dedicated grep tool, no path exclusions) for
(A) device names — `hailo-8|hailo8|Hailo-8|hailo15h|hailo10h` (~90 matching
lines) — and (B) requirement language — `target board|deployment board|latency
requirement|FPS requirement|power requirement|Dataflow Compiler|HailoRT|
hw_arch|supported_hw_arch` (59 matches). Every match was inspected.

Result: **zero prescriptive statements.** No file states which Hailo device
this project must deploy on; no file states a DFC/HailoRT version requirement;
no file states latency, FPS, throughput or power requirements for ARM-P.

- `README.md`, `CLAUDE.md`: zero device/requirement matches. VERIFIED.
- `requirements.txt`: torch/torchvision/numpy/opencv/onnx/onnxruntime/
  matplotlib/scipy/pandas/PyYAML/pytest/thop/tensorboard/tqdm/Pillow/
  scikit-image — **no Hailo package**. VERIFIED.
- Dockerfile, `.env`, `environment*.yml`, `setup.py`, `setup.cfg`,
  `pyproject.toml`: none exist in this repository. VERIFIED (prior report;
  directory listing confirms no such files at root).
- CI configuration / deployment scripts / model-conversion scripts / Docker
  files: none found. VERIFIED.
- `hw_arch` hits are decoded reference-profiler outputs (`results/profiler/`,
  `experiments/EXP-017|018/`). `supported_hw_arch` hits are Hailo's own
  `stereonet.yaml`. Latency/FPS hits are reference figures (SR-004/SR-006) or
  explicit UNKNOWN statements. All descriptive. VERIFIED.
- The single "target accelerator hardware" occurrence
  (`phase2/demo/README.md:208`) is listed as future work and **names no
  device**. VERIFIED (prior report, not contradicted).

Count note: absolute occurrence counts differ slightly from the prior report's
"101 of 101" because the search tools and exclusion sets differ (this search
excluded nothing; the prior excluded vendored trees, data, binaries). The
material conclusion is identical and independently confirmed: **101-of-101 vs
~149-of-~149 — every occurrence descriptive, zero prescriptive.** The count
difference is method, not evidence. VERIFIED.

---

## 3. HISTORICAL HAILO REFERENCE EVIDENCE

All VERIFIED. None of this is a project requirement (§5 explains why).

| # | Artifact | Device / version evidence |
|---|---|---|
| SR-002 | `reference/hailo_model_zoo/stereonet.yaml` (local master copy, lines 48–50) | `supported_hw_arch: hailo15h, hailo10h`; hailo8 NOT listed. Re-fetched from master live 2026-09-20: identical. |
| SR-002v2 | v2.19.0 `stereonet.yaml` (fetched live 2026-09-20 — **new**) | `supported_hw_arch: [hailo8]`. Same file, same fields, only the arch list differs by branch. |
| SR-003 | `stereonet.alls` (master) | Normalisation insertion + spatial defusion of conv42–conv52; device-agnostic directives. |
| SR-004 | `HAILO8_stereo_depth_estimation.rst` (master, legacy `HAILO8/` docs subtree) | Hailo-8 figures: float EPE 8.22, hardware EPE 10.3, 10.7 FPS batch 1 / 11.6 FPS batch 8. |
| SR-005 | `stereonet.hef` (24,057,165 bytes, from `.../Compiled/v2.19.0/hailo8/`) | v2.19.0 Hailo-8 build; header region contains version-like string `3.34.0` (prior read-only scan; not re-scanned, file untouched). |
| SR-006 | Profiler HTML (from `.../Compiled/v2.19.0/hailo8/`) | `hw_arch: hailo8`; 242 layers uniform 8/8/8; post-placement estimate, model-level fps/latency N/A. |
| SR-007 | hailo-apps stereo depth application | Disparity-only 8-bit cast; Hailo-8 era. |

**O11 verdict: RESOLVED (was INFERRED, now VERIFIED).** The master config lists
hailo15h/hailo10h because master targets Hailo-10/15; the benchmark page and
binaries are v2.x-era Hailo-8 artifacts; the v2.19.0 yaml confirms
`supported_hw_arch: hailo8`. Three scopes, one consistent story. This resolves
the documentation question. It does **not** nominate any device for ARM-P.

---

## 4. DEVICE CANDIDATES

Every device string ever appearing in repo or reference scope:

| Candidate | Where it appears | What it would entail |
|---|---|---|
| Hailo-8 | SR-004/005/006, profiler `hw_arch`, v2.19.0 yaml, DeGirum farm | Model Zoo v2.x + DFC v3.x (v3.34.0 current); discrete accelerator, 26 TOPS INT8, on-chip memory only |
| Hailo-8L | Model Zoo README compatibility notice; DeGirum farm | Same v2.x + DFC v3.x line; lower-cost vision part |
| Hailo-10H | Master yaml (`hailo10h`); Model Zoo master default `--hw-arch` target | Master branch + DFC v5.x (v5.4.0 current); 40 TOPS INT4 / 20 TOPS INT8; 4/8 GB LPDDR4/4X; PCIe modules |
| Hailo-15H | Master yaml (`hailo15h`) | Master branch + DFC v5.x; camera-SoC product class with Arm cores — different integration from discrete accelerators |
| Hailo-15L / Hailo-15M | Model Zoo master README model lists | Master branch + DFC v5.x; NOT APPLICABLE — never associated with StereoNet in any artifact |
| Hailo-8R | Appeared only in the prior report's search pattern | NOT APPLICABLE — zero occurrences in this repo or the reference scope |

---

## 5. DEVICE/TARGET CLASSIFICATION

Required classes: 1 AUTHORITATIVE REQUIREMENT / 2 HISTORICAL REFERENCE /
3 TOOLCHAIN ARTIFACT / 4 DESCRIPTIVE EXAMPLE / 5 UNKNOWN.

| Candidate / item | Class | Basis |
|---|---|---|
| Hailo-8 as *our* target | **5 UNKNOWN** (contender, not requirement) | No nominating source (§2). Q3 ≠ Q1 (§10). |
| Hailo-8L as *our* target | **5 UNKNOWN** (contender, not requirement) | Same; appears only in compatibility notices. |
| Hailo-10H as *our* target | **5 UNKNOWN** (contender, not requirement) | Master yaml lists it for Hailo's model, not ours. |
| Hailo-15H as *our* target | **5 UNKNOWN** (contender, not requirement) | Same. |
| Hailo-8 as the *reference* device | **2 HISTORICAL REFERENCE** | SR-004/005/006, v2.19.0 acquisition URLs. |
| `hailo15h/hailo10h` in master yaml | **2 HISTORICAL REFERENCE** | Hailo's branch scoping (O11 resolved). |
| `3.34.0` HEF-header string | **3 TOOLCHAIN ARTIFACT** | Property of the reference build binary. |
| DFC/HailoRT version numbers | **3 TOOLCHAIN ARTIFACT** | Properties of toolchains, not requirements on us. |
| "Target accelerator hardware" (demo README) | **4 DESCRIPTIVE EXAMPLE** | Future-work placeholder naming no device. |
| "26 TOPS / 40 TOPS" device ratings | **4 DESCRIPTIVE EXAMPLE** | Vendor ratings, not project requirements. |
| DeGirum Hailo-8/8L farm | **4 DESCRIPTIVE EXAMPLE** (of an access route) | Availability ≠ requirement (§10). |

**No item in this project is class 1. VERIFIED.** That absence is the finding.

---

## 6. DFC/SDK VERSION EVIDENCE

| Item | Value | Status |
|---|---|---|
| DFC version on this host | None installed | VERIFIED (re-probed 2026-09-20: 6 modules unimportable, 3 binaries absent; matches `HAILO_COMPILATION_RECORD.json`) |
| HailoRT version on this host | None installed | VERIFIED (same probe) |
| DFC behind reference HEF | `3.34.0` string in header | VERIFIED as a string in the binary (prior scan); interpretation as DFC version now **corroborated**: Hailo community staff post 2026-09-10 lists "Dataflow Compiler v3.34.0" as the current Hailo-8 line (with HailoRT v4.24.0, Model Zoo v2.19.0). Source: https://community.hailo.ai/t/download-problem/19749, accessed 2026-09-20. |
| DFC for Hailo-8/8L (current) | v3.34.0; Model Zoo v2.19.0; HailoRT v4.24.0 | VERIFIED (EXTERNAL, 2026-09-20, community staff post above) |
| DFC for Hailo-10/15 (current) | v5.4.0; HailoRT v5.4.0 | VERIFIED (EXTERNAL, 2026-09-20, Model Zoo master README fetched live) |
| DFC v3.33 / HailoRT v4.23 | Seen in Ultralytics Hailo docs as validated combo | VERIFIED (EXTERNAL, 2026-09-20, https://docs.ultralytics.com/integrations/hailo) — older point in the v3.x line |
| DFC host requirement | Linux x86_64; Ubuntu 20.04/22.04/24.04; this host is Windows 11 | VERIFIED (EXTERNAL docs + host platform record) |
| DFC for ARM-P | **UNKNOWN** — function of the undecided device | Follows §5: Hailo-8/8L ⇒ v3.x; Hailo-10/15 ⇒ v5.x. Cannot be fixed first. |
| ONNX opset accepted by DFC | Not established from a first-party DFC source in this task | UNKNOWN. ARM-P ONNX is opset 13 (re-verified today: 712 nodes). Whether DFC v3.x/v5.x accepts opset 13 for *this graph* is settled only by a parser run. |

---

## 7. DEVICE <-> TOOLCHAIN COMPATIBILITY

VERIFIED (EXTERNAL, 2026-09-20) from the Model Zoo master README (fetched
live), the v2.19.0 yaml (fetched live), the Hailo community staff post
(2026-09-10), and Ultralytics Hailo integration docs:

| Device | Model Zoo branch | DFC line | HailoRT line | HEF interchange |
|---|---|---|---|---|
| Hailo-8 | v2.x (v2.19.0 current) | v3.x (v3.34.0 current) | v4.x (v4.24.0 current) | Hailo-8/8L HEFs do NOT run on Hailo-10/15 and vice versa |
| Hailo-8L | v2.x | v3.x | v4.x | same line as Hailo-8 |
| Hailo-10H | master (default target) | v5.x (v5.4.0 current) | v5.x (v5.4.0 current) | separate generation |
| Hailo-15H/L | master | v5.x | v5.x (device-bundled; a DFC-5.3.0-HEF vs HailoRT-4.20.1 mismatch report exists — versions must be matched per the Software Suite) | separate generation |

**The newest toolchain is NOT the correct one for Hailo-8.** If the target is
Hailo-8/8L, the correct line is the older v2.x / DFC v3.x — consistent with the
reference HEF. INFERRED from the quoted pairing; load-bearing for §13.

---

## 8. AVAILABLE COMPILATION ROUTES

| Route | Status | Notes |
|---|---|---|
| DFC install via Hailo Developer Zone (free registration) | AVAILABLE in principle, NOT attempted | Requires Linux x86 (this host is Windows 11) ⇒ WSL2-Ubuntu or Docker container. PROPOSED isolated environment; the frozen research env was NOT modified and no install was attempted (per stop condition). |
| hailomz parse/optimize/compile flow for a custom ONNX | AVAILABLE once DFC installed | Standard flow: parse (start nodes `left`/`right`, end `disparity`, ImageNet normalize-in-net per yaml) → quantise (hailo_calib 0–159) → compile → HEF. Exact parser verdict for ARM-P UNKNOWN until run. |
| DeGirum Cloud Compiler | NOT APPLICABLE for ARM-P | Accepts only Ultralytics PyTorch checkpoints, not arbitrary ONNX. VERIFIED (EXTERNAL, docs.degirum.com). |
| Compiling against an assumed device | FORBIDDEN | Stop condition §18: no HEF against an assumed device. None compiled. |

---

## 9. AVAILABLE EXECUTION/HARDWARE ROUTES

| Route | Status | Notes |
|---|---|---|
| Local Hailo hardware (PCIe) | NONE | No device on this host (prior PCIe scan negative; nothing attached since). VERIFIED by absence of any change record. |
| Hailo Emulator (no silicon) | AVAILABLE once DFC installed | Functional (not timing) evaluation of the quantised model — route to the first deployed-accuracy number under the frozen contract. VERIFIED (EXTERNAL, Model Zoo README). |
| DeGirum Device Farm (Hailo-8/8L silicon) | AVAILABLE but DYING: grace ends **2026-09-30** | VERIFIED (EXTERNAL, 2026-09-20): DeGirum ceased operations 2026-08-01, assets to LatticeWork; AI Hub free-Pro grace through 2026-09-30, then unavailable publicly. Sources: https://degirum.com/ (accessed 2026-09-20) for cessation/acquisition/grace terms; https://community.degirum.com/t/an-important-announcement-on-degirums-products/433 — NO LONGER RESOLVING as of 2026-09-20 (host community.degirum.com, DNS failure), retained as dead citation; https://docs.degirum.com/ai-hub/hardware-explorer. Hailo-8/8L only; needs a valid HEF first (which needs the target decision). |
| Hailo-10/15 free silicon | NONE FOUND | No free hardware route identified. Emulator-only unless hardware purchased/supplied. UNKNOWN whether any exists. |
| Docker route on this host | UNAVAILABLE as configured | Prior probe: daemon unreachable, no Hailo images. Not re-probed (no change record; WSL2 route documented as alternative). |

---

## 10. PERFORMANCE REQUIREMENT STATUS

All UNKNOWN. VERIFIED by the §2 search — no latency, FPS, throughput or power
requirement for ARM-P exists in any project file.

| Figure in circulation | Fenced as | Must not be mistaken for |
|---|---|---|
| 10.7 FPS batch-1 / 11.6 batch-8 (SR-004) | Historical reference measurement, Hailo-8 | An ARM-P requirement or an ARM-P result |
| ~43 FPS modelled bottleneck conv50 (SR-006) | Compiler post-placement estimate, reference only, fps/latency N/A | A silicon measurement |
| EXP-013/014 GPU/CPU timings | Our-stack measurements of our models | Any Hailo-silicon proxy |

Operative rule (carried): measurement ≠ requirement. Q6 (what must be
satisfied) is unanswerable until the owner sets values. The six questions stay
distinct: Q1 required = UNKNOWN; Q2 accessible = emulator + dying DeGirum
(Hailo-8/8L only); Q3 historical = Hailo-8; Q4 compilable = UNKNOWN until
target fixed + parser run; Q5 executable = whichever device the HEF is built
for; Q6 requirement = UNKNOWN.

---

## 11. UNKNOWNs

1. Target Hailo device (Q1). 2. Target board/module. 3. Required DFC version
   (function of 1). 4. Required HailoRT version (function of 1). 5. DFC ONNX
   opset/operator acceptance for this graph (settled only by parser run).
   6. Quantisation path outcome for ARM-P (needs 1–3 + run). 7. Calibration
   beyond the known set (set known: hailo_calib 0–159; outcome UNKNOWN).
   8. HEF generability (needs 1–6). 9. Emulator availability in practice
   (ships with DFC; version pairing per §7). 10. Physical hardware for
   Hailo-10/15 (none found). 11. Latency/FPS/power requirements (Q6).
   12. Whether C1 parity (FAIL, frozen) matters for deployment (needs
   toolchain: emulator/quantised accuracy comparison). 13. Whether the DR-0
   dynamic-range observation blocks quantisation (needs toolchain; NOT the
   C1 mechanism — frozen diagnostic). 14. Post-2026-09-30 farm availability
   (announced unavailable publicly; whether LatticeWork continues anything:
   UNKNOWN).

---

## 12. BLOCKERS

| # | Blocker | Unblocked by |
|---|---|---|
| 1 | Target-device decision (project owner) | §13 decision request — the single root blocker |
| 2 | DFC/SDK version selection | 1 (line follows device, §7) |
| 3 | DFC installation (isolated Linux env) | 1 + 2 (which line to install) |
| 4 | Parser verdict on ARM-P ONNX (incl. Conv3D/rank-5 ruling, Pad-cluster survival) | 3 |
| 5 | Quantisation + calibration run | 4 |
| 6 | HEF compilation | 5 |
| 7 | Emulator accuracy under frozen contract | 6 |
| 8 | Any silicon measurement | 6 + (for 8/8L: action before 2026-09-30; for 10/15: hardware that does not exist here) |
| 9 | Dynamic-range risk resolution | 5 (quantiser verdict, not more rescaling sweeps — DR-1 closed) |
| 10 | C1-parity deployment relevance | 7 (compare quantised vs host accuracy; tolerance itself frozen) |
| 11 | Any latency/FPS/power pass-fail | Stated requirements (Q6), independent of 1–8 for measurement but required for judgement |

---

## 13. EXACT NEXT ACTION

**Decision request for the project owner.** Please supply exactly the
following; nothing else unblocks Stage C:

1. **Target device (mandatory):** Hailo-8, Hailo-8L, Hailo-10H, Hailo-15H, a
   target family, multi-target support — or an explicit statement that no
   target will be set (in which case Stage C closes as UNKNOWN-target with
   emulator-only characterisation if authorised).
2. **Board/module (if the device needs one):** e.g. M.2 module, AI HAT, camera
   SoC — or NOT APPLICABLE.
3. **Performance requirements (optional but needed for any pass/fail):**
   latency ms, FPS @ batch, powerW — or explicit "measure only, no gate".
4. **Hardware access (if any):** physical device available? purchase approved?
   DeGirum grace-period run authorised before 2026-09-30 (Hailo-8/8L only)?

Context the owner needs (so the answer is informed, not a guess):

- The choice selects the toolchain: Hailo-8/8L ⇒ v2.x + DFC v3.x (v3.34.0);
  Hailo-10/15 ⇒ master + DFC v5.x (v5.4.0). Different compilers, different
  operator support, non-interchangeable HEFs.
- Free silicon exists only for Hailo-8/8L (DeGirum) and only until
  2026-09-30, from a company that has ceased operations. Hailo-10/15 means
  emulator-only unless hardware is supplied.
- The accuracy reference (1.3134471 px) and 10.7 FPS figure are Hailo-8
  measurements of Hailo's model. A different target still yields a valid
  deployment result but weakens cross-device comparison.
- The model does not change with the answer. ARM-P seed 1 is frozen
  (`b2f6f5d5…fffeb7454`); the ONNX already exists; C1 FAIL, DR-0 risk and
  DR-1 H1-FAIL all stand regardless.

Until the answer arrives: no compilation, no quantisation, no DeGirum run,
no device chosen by us.

---

## 14. ARTIFACT INTEGRITY

Re-measured 2026-09-20 on disk (read-only; nothing modified, nothing
regenerated):

| Artifact | Measured SHA-256 | Expected | Verdict |
|---|---|---|---|
| ARM-P seed-1 checkpoint `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth` | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` | same | VERIFIED match |
| ARM-P ONNX `stage_c_deploy/armp_stereonet.onnx` (712 nodes, opset 13 — re-verified via onnx load) | `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989` | manager note value: identical | VERIFIED match against the ON-DISK authority |
| Reference ONNX `reference/onnx/stereonet.onnx` | `b1a01d855bb22663f06dfd29eae11194e6fde952a319673e036f157c3e10095c` | `b1a01d85…157c3e10095c` | VERIFIED match |

**Spec-transposition discrepancy (recorded, not repaired):** the owner
specification in this task brief lists the ARM-P ONNX hash with `...deed1dce...`
(chars 13–15 `dce`); the on-disk file — and the manager note — read
`...deed1cde...` (`cde`). The ON-DISK value is authoritative per the brief. The
file was NOT modified to match the spec. No working copies were made; no
deployment experiments required one.

---

## 15. SOURCES

Repo (all VERIFIED by direct read): `STAGE_C_DEPLOYMENT_BLOCKER_REPORT.md`
(§1–5, 8, 9), `C0_DECISION_RECORD.md` (§4–5, 7–8),
`STAGE_C0_DEPLOYMENT_CONTRACT.md` (§10, 10a), `docs/hardware_analysis.md`,
`docs/research_questions.md` (O11), `docs/source_registry.md`,
`reference/MANIFEST.md`, `reference/manifest.json`,
`reference/hailo_model_zoo/stereonet.yaml:48-50`, `DR1_RESCALE_GATE.md`,
`HAILO_TOOLCHAIN_REPORT.md` + `HAILO_COMPILATION_RECORD.json`
(`phase2/deploy/validation/`).

External (all accessed 2026-09-20; network access YES):

1. v2.19.0 stereonet.yaml — https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/v2.19.0/hailo_model_zoo/cfg/networks/stereonet.yaml (`supported_hw_arch: hailo8`)
2. master stereonet.yaml — https://raw.githubusercontent.com/hailo-ai/hailo_model_zoo/master/hailo_model_zoo/cfg/networks/stereonet.yaml (`hailo15h, hailo10h`)
3. Model Zoo master README (branch pairing; DFC v5.4.0 / HailoRT v5.4.0) — https://github.com/hailo-ai/hailo_model_zoo/blob/master/README.rst
4. Hailo community staff post 2026-09-10 (corrected from 2026-09-09 as recorded during the run; no separate post-date/edit-date explanation verified in this pass) (DFC v3.34.0 / HailoRT v4.24.0 / Model Zoo v2.19.0 as current Hailo-8 line) — https://community.hailo.ai/t/download-problem/19749
5. Ultralytics Hailo docs (DFC v3.x vs v5.x split; Linux x86_64; DFC 3.33 / HailoRT 4.23 validated point) — https://docs.ultralytics.com/integrations/hailo (published 2026-05-26)
6. DeGirum cessation + grace terms — https://degirum.com/ (accessed 2026-09-20) for the 2026-08-01 cessation, LatticeWork acquisition, and AI Hub grace access through 2026-09-30; the originally cited forum page https://community.degirum.com/t/an-important-announcement-on-degirums-products/433 (2026-07-31) is NO LONGER RESOLVING as of 2026-09-20 (host community.degirum.com, DNS failure) and is retained here as a dead citation. Whether that forum page was actually retrieved during the original run or the URL was constructed is UNKNOWN.
7. DeGirum hardware explorer (Hailo = HailoRT + Hailo-8/Hailo-8L) — https://docs.degirum.com/ai-hub/hardware-explorer

No random tutorials used. No source fabricated. Prior-report external claims
(Model Zoo README pairing, DeGirum sunset, Developer Zone routes) all
re-confirmed live except as newly refined above.

---

## 16. FINAL DEPLOYMENT CONTRACT STATUS

**Target remains UNKNOWN. No authoritative target exists (SUCCESS condition B
path).** The report proves the absence (§2) and the decision request (§13)
states exactly what the owner must supply. Nothing is deployment-ready; nothing
is declared deployment-impossible.

### Required final table

| Field | Status | Evidence | Confidence |
|---|---|---|---|
| Target device | UNKNOWN | §2 search: zero prescriptive occurrences; §5: no class-1 item | VERIFIED absence |
| Target board-module | UNKNOWN | Same search; no board/BOM anywhere | VERIFIED absence |
| DFC version | UNKNOWN (conditional: v3.34.0 if Hailo-8/8L; v5.4.0 if Hailo-10/15) | §6–7: lines established, selection blocked on target | VERIFIED conditional |
| HailoRT version | UNKNOWN (conditional: v4.24.0 if Hailo-8/8L; v5.4.0 if Hailo-10/15) | §6–7, same basis | VERIFIED conditional |
| ONNX support | UNKNOWN for this graph (artifact: opset 13, 712 nodes) | Parser run required; §6 | VERIFIED artifact / UNKNOWN acceptance |
| Quantization path | UNKNOWN (route known: hailomz optimize, uniform-int8 precedent 8/8/8 on reference) | SR-006 precedent is reference-only; no ARM-P run | UNKNOWN |
| Calibration path | UNKNOWN outcome (set known: hailo_calib scenes 0–159, disjoint from eval 160–199) | `src/datasets/kitti2015.py`, contract §6; run required | VERIFIED set / UNKNOWN outcome |
| HEF generation | UNKNOWN (no HEF compiled; forbidden until contract) | Stop condition honoured; §8 | UNKNOWN |
| Emulator | AVAILABLE in principle, NOT installed | Ships with DFC; §9 | INFERRED availability / UNKNOWN in practice |
| Physical hardware | NONE locally; DeGirum Hailo-8/8L until 2026-09-30; none for 10/15 | §9; DeGirum cessation notice | VERIFIED |
| Latency requirement | UNKNOWN | §2, §10: no requirement exists | VERIFIED absence |
| FPS requirement | UNKNOWN | Same | VERIFIED absence |
| Power requirement | UNKNOWN | Same; `hardware_analysis.md:367`, O10 open | VERIFIED absence |

### Prior-report verification ledger

Confirmed independently: zero-prescriptive-target finding; branch↔device↔
compiler pairing (now live re-verified); no-toolchain-on-host (re-probed);
DeGirum Hailo-8/8L-only + 2026-09-30 sunset (re-verified, with new cessation
detail); reference device/volume facts (SR-002–006 scope). Extended: O11
v2.x-yaml check (done — resolves O11); DFC 3.34.0 corroboration (done);
DeGirum operational status (done — worse than recorded). Could not do: nothing
in the brief was left undone — no file outside `stage_c_deploy/
C1_TARGET_TOOLCHAIN_RESOLUTION.md` + `stage_c_deploy/target_resolution/` was
created; no frozen artifact touched; no DR-1/C1 branch reopened.

### STOP

Target/toolchain resolution ends here. No training, fine-tuning, ARM-P/P2A
alteration, ONNX modification, tolerance change, DR-1 variant, scale sweep,
normalisation/architecture change, assumed-device compilation, or latency/FPS/
power claim follows without a resolved deployment contract.

---

## Report

Files created (two, nothing else):

1. `stage_c_deploy/C1_TARGET_TOOLCHAIN_RESOLUTION.md` (this file)
2. `stage_c_deploy/target_resolution/evidence_summary.json` (search/hash/fetch ledger)

No existing file was modified. No model, checkpoint, ONNX, script or
pre-existing document was created, modified or deleted. No training, evaluation,
export, compilation or environment change was performed.

Citation-correction pass (2026-09-20, text only): fixed (1) source 6 — the unreachable forum URL was replaced with https://degirum.com/ (accessed 2026-09-20) with the original URL retained as a dead citation (NO LONGER RESOLVING as of 2026-09-20, host community.degirum.com, DNS failure); and (2) the Hailo community staff-post date 2026-09-09 → 2026-09-10 everywhere it appears; the manager independently re-verified the v2.19.0/master supported_hw_arch claim and the DeGirum and Hailo-version facts; no search or measurement was re-run; no verdict, UNKNOWN, table finding, hash, or label was changed.
