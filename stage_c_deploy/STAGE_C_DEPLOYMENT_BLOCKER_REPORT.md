# STAGE C — DEPLOYMENT BLOCKER REPORT

What stands between the frozen ARM-P seed-1 model and an actual Hailo deployment
measurement, and what must be supplied to remove it.

Every statement is labelled **VERIFIED**, **INFERRED** or **UNKNOWN**. No device is
chosen here. No optimisation is proposed. No gate is reinterpreted.

**Authorship note:** this document was written by the manager (Claude Code) rather than
the `opencode` worker, at the project owner's explicit direction, after the worker
returned `Rate limit exceeded` from the free model on three consecutive attempts
(22:26, 22:29 and 22:38 local, the last after a 7-minute wait). Recorded because the
manager/worker separation is a standing project rule and this is a documented exception
to it, not a silent one. The search results the report rests on were produced and
verified before that decision.

---

## 1. TARGET DEVICE SEARCH SCOPE

**VERIFIED.** Ripgrep over the whole repository, 2026-09-19, excluding
`reference/public_repos/**` (vendored third-party stereo repos — AANet, PSMNet and
others; not project evidence), `data/**`, `**/__pycache__/**`, `**/*.onnx`, `**/*.hef`,
`**/*.html`.

The exclusion is recorded rather than hidden: a first attempt that did *not* exclude
`reference/public_repos/` exceeded a 120-second shell timeout, which is why the
exclusion exists. Nothing in those vendored trees is a statement about this project's
deployment target.

Two patterns were used:

| Pattern | Terms | Result |
|---|---|---|
| A | `Hailo-8`, `hailo8`, `Hailo-8L`, `Hailo-8R`, `Hailo-10H`, `Hailo-15H`, `hailo10`, `hailo15` | **76 occurrences across 26 files** |
| B | `target board`, `deployment board`, `hardware specification`, `BOM`, `SDK version`, `compiler version`, `FPGA`, `accelerator` | **25 occurrences across 12 files** |

Files inspected specifically, with their result:

| File / class | Result |
|---|---|
| `README.md` | **zero matches**, either pattern |
| `CLAUDE.md` | **zero matches**, either pattern |
| `requirements.txt` | exists; torch, torchvision, numpy, opencv-python, onnx, onnxruntime, matplotlib, scipy, pandas, PyYAML, pytest, thop, tensorboard, tqdm, Pillow, scikit-image — **no Hailo package** |
| Dockerfile, `.env`, `environment*.yml`, `setup.py`, `setup.cfg`, `pyproject.toml` | **none exist in this repository** |
| `docs/*.md`, `phase2/docs/*.md`, `stage_b_armp/*.md`, `stage_c_deploy/*.md` | matched; all descriptive (below) |
| `reference/hailo_model_zoo/*`, `reference/MANIFEST.md`, `reference/manifest.json`, `stereonet.alls` | matched; all describe the reference |
| `reference/stereonet.hef` metadata | scanned separately (§4) |

**Classification.** Every occurrence was classified as either **(a) descriptive** — it
describes the historical Hailo *reference* deployment, or records the UNKNOWN itself — or
**(b) prescriptive** — it states what *this* project must deploy on.

**Result: 101 of 101 occurrences are (a). Zero are (b).**

The three matches outside `docs/`, `phase2/docs/` and `stage_c_deploy/`, quoted in full so
the absence is auditable:

- `PHASE_2_FINAL_REPORT.md:235` — "and no PCIe accelerator. Every gap is environmental,
  not a property of the" — a statement about *this host lacking* hardware.
- `PHASE_1_OVERVIEW.md:34` — "publicly documented stereo model compiled for a real edge
  accelerator, with" — a description of the reference.
- `phase2/demo/README.md:208` — "- Deployment on the target accelerator hardware." —
  listed as future work, and **it names no device.**

---

## 2. VERIFIED TARGET EVIDENCE

**There is none.** No file in this repository states which Hailo device this project is
required to deploy on. **[VERIFIED: the §1 search; 101 occurrences, all descriptive.]**

The descriptive material falls into four groups, none of which is a requirement:

| Group | Example | Why it is not a target statement |
|---|---|---|
| Reference benchmark figures | Hailo-8: 10.7 FPS batch 1, hardware EPE 10.3 (SR-004) | Hailo's measurement of Hailo's model |
| Reference compiled artifacts | profiler `hw_arch: hailo8` (SR-006); HEF (SR-005) | properties of the reference build |
| Reference configuration | `supported_hw_arch: hailo15h, hailo10h` (SR-002) | what Hailo's master branch supports for that model |
| Our own records of the gap | `docs/research_questions.md` O11; `docs/hardware_analysis.md` §6; `C0_DECISION_RECORD.md` §4 | these record the UNKNOWN, they do not close it |

**[VERIFIED: `docs/source_registry.md` rows SR-002 to SR-006; `docs/research_questions.md:63`;
`stage_c_deploy/C0_DECISION_RECORD.md` §4.]**

---

## 3. TARGET DEVICE STATUS

**TARGET DEVICE = UNKNOWN.** **[VERIFIED by exhaustive search, §1–§2.]**

This is now a searched-and-confirmed absence rather than an assumption. No device is
selected in this report. Per the standing instruction, the reference's device is **not**
inherited as ours: that the reference was built for Hailo-8 is a fact about the reference,
not a requirement on us. **[VERIFIED: instruction of record; `C0_DECISION_RECORD.md` §4–5.]**

---

## 4. DFC / SDK STATUS

**Nothing is installed.** No Dataflow Compiler, no HailoRT, no Hailo Python module, no
Hailo binary on PATH, no Hailo Docker image, Docker daemon unreachable, and no Hailo PCIe
device present. **[VERIFIED: `phase2/deploy/validation/HAILO_COMPILATION_RECORD.json`,
`toolchain_probe` — 7 modules not importable, 6 binaries absent, `any_toolchain_found:
false`; PCIe vendor scan `VEN_1E60` negative.]**

### The version that built the reference HEF

A read-only scan of all 24,057,165 bytes of `reference/stereonet.hef` found exactly two
version-like strings — **`3.34.0`** (header region) and **`2.5.1`** — and **zero** matches
for any `hailo\d+` or `HAILO\d+` architecture string anywhere in the file.
**[VERIFIED: manager measurement, 2026-09-19, read-only.]**

- `3.34.0` is **INFERRED** to be the Dataflow Compiler version that produced this HEF. It
  is a version string in a DFC-produced binary, not a documented, parsed field, and the
  HEF format is not publicly specified. This refines the record's previous "DFC v3.x-era"
  to a specific candidate.
- What `2.5.1` denotes is **UNKNOWN**.
- The HEF names no device. **[VERIFIED.]**

### Branch ↔ device ↔ compiler pairing

From the Hailo Model Zoo README, verbatim **[VERIFIED (EXTERNAL, dated 2026-09-19) —
https://github.com/hailo-ai/hailo_model_zoo/blob/master/README.rst]**:

> "The Hailo-8 and Hailo-8L devices are supported on the **Hailo Model Zoo v2.x** branch,
> in combination with the **Hailo Dataflow Compiler v3.x** branch."

> "The master branch is intended for **Hailo-10** and **Hailo-15** devices only."

The same page records master upgraded to Dataflow Compiler v5.4.0 and HailoRT v5.4.0.

**Consequence: the correct DFC version is a function of the undecided device.** It cannot
be fixed until §3 is resolved, and **the newest toolchain is not assumed to be the correct
one** — if the target is Hailo-8 or Hailo-8L, the correct line is the *older* v2.x / DFC
v3.x branch, consistent with the `3.34.0` inferred above. **[INFERRED from the quoted
README pairing.]**

### What this says about open question O11

O11 asks why the configuration lists `hailo15h, hailo10h` while the benchmark is Hailo-8.
It has been carried as UNKNOWN in `docs/hardware_analysis.md`, `docs/reference_pipeline.md`,
`docs/reproduction_report.md` and `docs/research_questions.md` since Phase 1.

The acquisition URLs recorded in `reference/MANIFEST.md` and `reference/manifest.json`
settle where each artifact came from **[VERIFIED: manifest rows 14–16]**:

| Artifact | Acquisition URL |
|---|---|
| SR-002 `stereonet.yaml` | `…/hailo_model_zoo/**master**/hailo_model_zoo/cfg/networks/stereonet.yaml` |
| SR-003 `stereonet.alls` | `…/hailo_model_zoo/**master**/hailo_model_zoo/cfg/alls/generic/stereonet.alls` |
| SR-004 `HAILO8_stereo_depth_estimation.rst` | `…/hailo_model_zoo/**master**/docs/public_models/**HAILO8**/HAILO8_…rst` |
| SR-005 HEF, SR-006 profiler | Model Zoo **v2.19.0** (the v2.x branch) |

So SR-004 **does** sit in a legacy `HAILO8/` documentation subtree, on master.
**[VERIFIED: the manifest URL.]**

**The leading explanation, labelled INFERRED:** there is probably no contradiction in
Hailo's documentation. On master, `cfg/networks/stereonet.yaml` declares the devices
*master* targets (Hailo-10/15), while `docs/public_models/HAILO8/` preserves the older
Hailo-8 benchmark page, and our binaries (HEF, profiler) are v2.19.0 Hailo-8 artifacts.
Three different scopes, read side by side as though they described one build.

**O11 is NOT recorded as closed.** This is the leading explanation, not a confirmation.
What would confirm it: inspect the **v2.x-branch** copy of `stereonet.yaml` and read its
own `supported_hw_arch` list. That requires network access and is not authorised in this
task. **[UNKNOWN until that check is run.]**

---

## 5. AVAILABLE HAILO TOOLCHAIN ROUTE

Carried forward from `STAGE_C0_DEPLOYMENT_CONTRACT.md` §10/10a, updated with the branch
pairing. **The route itself differs by device**, which is part of why §3 blocks everything.

| Route | If target is Hailo-8 / 8L | If target is Hailo-10 / 15 |
|---|---|---|
| Model Zoo branch | v2.x | master |
| Dataflow Compiler | v3.x line (reference HEF: `3.34.0` inferred) | v5.x line (README cites v5.4.0) |
| Host requirement | Linux x86, Ubuntu 20.04/22.04/24.04, 16 GB+ RAM; this host is Windows 11, so WSL2 or Docker is required | same |
| Obtained via | Hailo Developer Zone, free registration | same |
| Accuracy without silicon | Hailo Emulator | Hailo Emulator |
| Free real silicon | **DeGirum Device Farm — Hailo-8/8L only, access ends 2026-09-30** | **none found** |

**[VERIFIED (EXTERNAL, dated 2026-09-19): Hailo Developer Zone and the WSL2 install
thread for the host requirement; the Model Zoo README for the branch pairing;
https://degirum.com/ai-hub for the farm and its 2026-09-30 sunset.]**

The DeGirum asymmetry is **information for the device decision, not a reason to choose
Hailo-8.** Its Cloud Compiler cannot compile our graph in any case — it accepts only
Ultralytics PyTorch checkpoints, not arbitrary ONNX. **[VERIFIED (EXTERNAL):
https://docs.degirum.com/ai-hub/workspaces/cloud-compiler.]**

---

## 6. DYNAMIC-RANGE RISK STATUS

**OBSERVED**, pooled `max |value|` over the 5 diagnostic scenes
**[VERIFIED: `stage_c_deploy/diagnostics/C1_PARITY_DIAGNOSTIC_REPORT.md` §6 and
`diagnostics/activation_range.json`; ARM-P and P2A via PyTorch, reference via
onnxruntime]**:

| stage | ARM-P seed 1 | P2A seed 0 | Hailo reference |
|---|---|---|---|
| `left_features` | 4.12e+13 | 1.58e+07 | 6.28e+01 |
| `cost_volume` | 7.08e+13 | 2.06e+07 | 8.82e+01 |
| `aggregated_cost` | **2.53e+18** | 6.16e+09 | 2.39e+01 |
| `disparity_final` | 1.26e+02 | 1.21e+02 | 1.36e+02 |

The reference column is the model Hailo **actually quantised to uniform int8** — 8/8/8
across all 242 compiled layers, no mixed precision, `total_4bit_macs_per_frame` zero.
**[VERIFIED: `docs/hardware_analysis.md` §4a, from SR-006.]**

Four qualifications, all load-bearing:

1. **This is not the C1 parity mechanism.** fp32 relative precision is scale-invariant,
   and upstream relative divergence is comparable between the two models (~3e-7 to 1e-6).
   The parity failure remains attributed to readout amplification at near-ties.
   **[VERIFIED: diagnostic report §6, §8.]**
2. **It is not an ARM-P-only phenomenon.** P2A already sits ~2.6e8× above the reference.
   **[OBSERVED, table above.]**
3. **The explanation is INFERRED, not proven.** `regression_normalize=True` standardises
   the cost tensor before the softmax, so the loss is invariant to that tensor's scale and
   nothing penalises upward drift. Plausible and testable; **not tested**.
4. **Measurement caveats:** different backends (PyTorch vs onnxruntime) and different
   reference geometry (×16 downsample, 12 candidates, degenerate no-op cost volume). These
   do not account for 17 orders of magnitude, but the comparison is of magnitudes, not of
   equivalent computation. **[VERIFIED: diagnostic report §6 caveat.]**

**Risk status: UNKNOWN and unretired.** Whether a per-tensor int8 scale can represent a
2.53e+18 tensor before `regression_normalize` rescues it cannot be established without the
toolchain. No outcome is predicted here, and **no fix is proposed** — Stage C forbids
optimisation until a concrete deployment failure names what must be fixed.

---

## 7. C1 PARITY STATUS

**C1 parity gate = FAIL.** max |Δ| = **1.709e-3 px** against the predeclared **1e-3 px**
tolerance. **[VERIFIED: `stage_c_deploy/ARMP_DEPLOYMENT_VALIDATION.json`,
`parity_vs_pytorch.pass: false`.]**

- The tolerance was inherited from `phase2/scripts/export_p2a.py:106-107`, not invented
  afterwards, and has not been changed. **[VERIFIED.]**
- The failing artifact is retained: `stage_c_deploy/armp_stereonet.onnx`, sha256
  `4277090d…`, 712 nodes, opset 13. **[VERIFIED.]**
- P2A control passes the same gate at 6.180e-4 px. **[VERIFIED.]**
- 194 pixels of 2,266,880 exceed the gate; exactly 1 reaches 1.7e-3. **[OBSERVED.]**
- Mechanism localised: fp32 accumulation-order divergence amplified at the soft-argmin
  readout, at ambiguous pixels (top1−top2 gap 0.161 vs 0.337 background; disparity
  gradient 3.86 vs 0.31). **[OBSERVED / INFERRED per diagnostic report §8.]**
- **No export correctness bug was found.** Every located stage matches its PyTorch
  counterpart to relative ≤1.5e-5; ORT graph fusions refuted as source (disabling them
  makes both models worse); both runtimes bit-identical to themselves.
  **[VERIFIED: diagnostic report §9.]**

No pass verdict is written anywhere, and this diagnostic did not and cannot change the
gate.

---

## 8. WHAT IS BLOCKED

| # | Blocked | Unblocked by |
|---|---|---|
| 1 | Hailo parser run on the ARM-P ONNX | the target-device decision (§3) |
| 2 | Choice of DFC/SDK version and Model Zoo branch | the target-device decision (§3) |
| 3 | Quantisation and calibration | 1 and 2 |
| 4 | Compilation to HEF | 1–3 |
| 5 | Emulator accuracy measurement under the frozen contract | 4 |
| 6 | Resource / mapping / bottleneck analysis (C2) | 4 |
| 7 | Any silicon measurement, incl. DeGirum | a valid HEF (4), and DeGirum additionally requires the target to be Hailo-8/8L and action before 2026-09-30 |
| 8 | Resolution of the dynamic-range risk (§6) | 3 |
| 9 | Any pass/fail judgement on latency, FPS or power | a project requirement, which does not exist (`C0_DECISION_RECORD.md` §7–8) |
| 10 | Any Level-C claim of deployed superiority | all of the above |

Not blocked, and already done: the ARM-P export and its parity measurement (C1,
device-independent half) — completed, verdict FAIL, recorded.

---

## 9. EXACT NEXT ACTION

**Obtain the target-device decision from the project/company side.** One question:

> **Which Hailo device is the intended deployment target for this project?**
> Hailo-8, Hailo-8L, Hailo-10H, Hailo-15H, or another part.

The minimum context that side needs in order to answer:

1. **The choice also selects the toolchain.** Hailo-8/8L ⇒ Model Zoo v2.x + Dataflow
   Compiler v3.x. Hailo-10/15 ⇒ master + DFC v5.x. These are different compilers with
   different operator support; the answer determines what we install and what the parser
   will accept.
2. **The choice determines whether free silicon exists.** DeGirum's cloud farm covers
   Hailo-8/8L only and shuts down **2026-09-30**. For Hailo-10/15 we found no free
   hardware route, so those targets mean emulator-only results unless hardware is
   purchased or supplied.
3. **The choice affects comparability.** Our accuracy reference — Hailo's published
   1.3134471 px equivalent, and the 10.7 FPS figure — comes from the **Hailo-8** build.
   Deploying ARM-P to a different device still yields a valid deployment result, but the
   latency and resource comparison against that reference becomes cross-device and
   correspondingly weaker.
4. **What is not at stake.** The model does not change with the answer. ARM-P seed 1 is
   frozen at `b2f6f5d5…fffeb7454`, and the exported ONNX already exists.

Until that answer arrives: **no compilation, no quantisation, no DeGirum, and no device
chosen by us.** **[Per the standing stop condition.]**

A separate, non-blocking decision also remains open, recorded here so it is not forgotten:
whether a bare max-abs criterion over 2.27 M pixels is the right *deployment*-parity gate.
Any change to it must be justified in writing **before** a new number is chosen, and must
leave the current FAIL in the record. **[Per `C1_EXPORT_PARITY_REPORT.md` and the
diagnostic report §12.]**

---

## PROVENANCE

- Written 2026-09-19 (local 22:4x IST) by the manager, per §0 authorship note.
- Git HEAD `58e8a19908ddbd35451652c61aef478f56b51ebd`.
- Host Windows-11-10.0.26200-SP0; python 3.12.9, torch 2.7.0+cu128, onnx 1.22.0,
  onnxruntime 1.27.0, numpy 2.5.1.
- Frozen candidate: ARM-P seed 1,
  `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth`, sha256
  `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454`.
- Exported artifact under discussion: `stage_c_deploy/armp_stereonet.onnx`, sha256
  `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989`.
- **No model, checkpoint, ONNX, script or pre-existing document was created, modified or
  deleted for this report.** The only file written is this one. No training was run, no
  evaluation was run, no Hailo tool was invoked.
