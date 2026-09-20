# STAGE C0 — ARM-P DEPLOYMENT CONTRACT

Stage C0 is a paper contract. It establishes what is known, what is not, and
what must happen before any compile is attempted. It trains nothing, exports
nothing, compiles nothing, and modifies no model.

Label rule, applying to every item below: each claim carries exactly one of
**VERIFIED** (established by a file in this repo or by a first-party source
already recorded in `docs/source_registry.md`, with the path cited),
**UNKNOWN** (not established, with what would establish it stated), or
**ASSUMED** (a working assumption we choose, with what it rests on and what
breaks if it is wrong stated). An item with no label is a defect. Figures from
outside this repo are additionally marked **EXTERNAL** with the URL and date;
EXTERNAL is provenance, not a substitute for the status label. No number below
was invented: every number comes from a cited repo file or from the External
facts block (gathered by the manager, dated 2026-09-19; this worker has no
network access).

Read first (authoritative inputs, in order): `phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md`,
`phase2/deploy/validation/HAILO_COMPILATION_RECORD.json`,
`phase2/deploy/P2A_STATIC_EXPORT_VALIDATION.json`,
`phase2/deploy/P2A_DEPLOYMENT_VALIDATION.json`, `docs/hardware_analysis.md`,
`docs/source_registry.md`, `docs/reference_pipeline.md`,
`phase2/docs/PHASE2_DEPLOYMENT_SELECTION.md`,
`phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md`,
`stage_b_armp/ARMP_CLOSURE_RECORD.md`, `phase1/harness/frozen_eval.py`,
`src/models/stereonet/` (`cost_volume.py`, `aggregation.py`, `regression.py`,
`refinement.py`, `stereonet.py`).

---

## 0. Scope and standing claims

The following three claims must never be merged into one another.

- A. MODEL ACCURACY — ARM-P records a 3-seed mean best-checkpoint EPE of
  approximately 1.198874 px against the historical Hailo reference figure of
  1.3134471 px on the frozen KITTI contract. **[VERIFIED:
  `stage_b_armp/ARMP_CLOSURE_RECORD.md` Sections 4–6 for the three seed EPEs
  (1.2057590, 1.1912168, 1.1996447, full precision in Sections 4–6);
  `phase1/harness/frozen_eval.py:37-41` for the 1.3134471 px reference figure;
  the mean (1.2057590 + 1.1912168 + 1.1996447) / 3 = 1.1988735 is arithmetic on
  quoted values.]** Supported by Stage B, which is closed (see below). This is
  a host-side PyTorch accuracy statement only.
- B. DEPLOYABILITY — NOT ESTABLISHED. **[VERIFIED:
  `phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md` Section 0 — the furthest
  verified point of the predecessor P2A validation is ONNX; parser, compiler,
  HEF, HailoRT and hardware are all NOT ATTEMPTED / NOT AVAILABLE. No ARM-P
  export, parse, compile, or device run exists anywhere in the repo.]** No
  statement in this contract upgrades this.
- C. DEPLOYED SYSTEM SUPERIORITY — NOT ESTABLISHED. **[VERIFIED: same basis as
  B — with no compiled ARM-P artifact and no on-device measurement, there is no
  deployed system to be superior. `stage_b_armp/ARMP_CLOSURE_RECORD.md`
  Section 12 explicitly records that Stage B "does not establish any Hailo
  deployment benefit".]**

- Stage B is closed. **[VERIFIED: `stage_b_armp/ARMP_CLOSURE_RECORD.md:1-4` —
  "Stage B / ARM-P is formally closed".]**
- C0 modifies no model. **[VERIFIED: this contract creates only the two files
  named in the task under `stage_c_deploy/`; the inventory in Section 12 was
  produced by read-only loads (`torch.load(..., map_location="cpu")`, no
  re-save). No training ran, no GPU job ran, no checkpoint/ONNX file was
  modified.]**

---

## 1. The deployment candidate

There are three ARM-P candidates, one per seed. The choice between them is
**UNKNOWN / to be decided by the user** (see the freeze consequence below).
Do not pick one in this contract.

| Seed | Checkpoint | SHA-256 | Size (bytes) | Params / tensors | Strict load | Frozen best EPE (px) |
|---|---|---|---|---|---|---|
| 0 | `stage_b_armp/20260918T142128Z_tier2_pilot/armp/p2a_best.pth` | `58968771d987acb0f6ad03743d8b8d952b28dc6b43cb72bc5dd723d6a3419818` | 1621341 | 397,954 / 70 | strict-ok | 1.2057590 (full 1.2057590024628537) |
| 1 | `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth` | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` | 1621341 | 397,954 / 70 | strict-ok | 1.1912168 (full 1.191216765057325) |
| 2 | `stage_b_armp/20260919T092454Z_tier2_seed2/armp/p2a_best.pth` | `47a2d3e288ad91a6ce448934b0aed08a415bd8ce11904df602e181f1348f8b6d` | 1621213 | 397,954 / 70 | strict-ok | 1.1996447 (full 1.1996447345404795) |

- Row contents (paths, sizes, hashes, param counts, tensor counts, strict-load
  outcome). **[VERIFIED: `stage_c_deploy/c0_checkpoint_inventory.json`, produced
  by read-only `torch.load(..., map_location="cpu")` against the frozen P2A
  architecture `StereoNetConfig(downsample_levels=3, num_disparities=24,
  cost_volume_shift="right", regression_normalize=True)`; seed-2 file found at
  the stated path so no bundle search was needed.]**
- Seed EPEs. **[VERIFIED: quoted from
  `stage_b_armp/ARMP_CLOSURE_RECORD.md` Sections 4 (seed 0), 5 (seed 1), 6 (seed
  2); NOT re-evaluated for this contract.]**
- Seed-2 file size differs slightly (1621213 vs 1621341 bytes) while parameter
  content is identical (70 tensors, 397,954 params, strict-ok on all).
  **[VERIFIED: inventory JSON.]** No deployment consequence is drawn from size
  alone. **[ASSUMED — rests on: strict-load success and parameter identity being
  the load-bearing properties for export; breaks if: a future loader validates
  file size, which no loader in this repo does.]**
- Seed-2 checkpoint config labels its init source with a Kaggle path string
  (`/kaggle/working/repo/checkpoints/armp_stage1_best.pth`) while recording the
  identical FT3D sha256 (`3ae6fb3b…`). **[VERIFIED: config dump of the seed-2
  checkpoint vs `stage_b_armp/ARMP_CLOSURE_RECORD.md` Section 2.]** Treated as a
  label difference only. **[ASSUMED — rests on: sha256 identity of the recorded
  source; breaks if: the bytes hashed at seed-2 training time differed from the
  local FT3D file, which the matching digest contradicts.]**
- Which seed becomes the Stage C candidate is UNKNOWN and must be decided by
  the user. **[UNKNOWN — establishing it requires a user decision, not a
  measurement; no selection rule has been declared for ARM-P.]** Consequence:
  whichever is chosen must then be frozen for all of Stage C — same rule as the
  P2A deployment selection. **[VERIFIED (precedent):
  `phase2/docs/PHASE2_DEPLOYMENT_SELECTION.md` Sections 4, 8 — P2A deployed
  seed 0 by a stated lowest-EPE rule with single-seed reporting discipline;
  no equivalent rule exists yet for ARM-P — that absence is UNKNOWN.]**

- No ARM-P ONNX export exists yet. **[VERIFIED: filesystem sweep at C0 time
  shows exactly four ONNX files — `phase2/deploy/p2a_stereonet.onnx`,
  `phase2/deploy/p2a_stereonet_static.onnx`,
  `reference/onnx/stereonet.onnx`, `results/quantization/artifacts/stereonet_int8.onnx`;
  no ONNX file exists under `stage_b_armp/`. Both `phase2/deploy/` artifacts
  record `source_checkpoint: phase2/runs/p2a_scale_coverage/p2a_best.pth`
  (P2A seed 0) with `source_sha256 0868ffd1…` in
  `phase2/deploy/P2A_STATIC_EXPORT_VALIDATION.json:2-4` and
  `phase2/deploy/P2A_DEPLOYMENT_VALIDATION.json:2-3`, matching the P2A entry in
  the C0 inventory — so they were exported from P2A seed 0, NOT from ARM-P.]**
- ARM-P and P2A share the identical architecture: 397,954 params, 70 tensors,
  24 disparity candidates, 3 downsample levels, shift right,
  `regression_normalize` true. **[VERIFIED:
  `stage_b_armp/ARMP_CLOSURE_RECORD.md` Section 8 (architecture held constant,
  strict load clean on every scored checkpoint);
  `phase2/deploy/P2A_DEPLOYMENT_VALIDATION.json:15-22` (params, config);
  `phase2/scripts/train_p2a_scale_coverage.py:62-64` (ARM_V_CONFIG, ARM_V_PARAMS).]**
  Therefore the P2A ONNX graph/operator inventory is a valid **structural**
  proxy for ARM-P, but not a substitute for an actual ARM-P export.
  **[ASSUMED (the proxy claim) — rests on: the VERIFIED architecture identity
  above plus the VERIFIED 70/70 bit-identical weight match of the P2A ONNX to
  its own checkpoint (`phase2/deploy/validation/HAILO_COMPILATION_RECORD.json`,
  `artifacts.*.weights`); breaks if: any export-time shape specialisation,
  constant folding, or exporter-version difference makes the ARM-P graph differ
  from the P2A graph despite identical architecture. An actual ARM-P export
  with parity checks is MANDATORY before C1 conclusions.]**

---

## 2. Target hardware contract

Central open item, stated first because everything else depends on it:

> The Hailo Model Zoo entry for the reference StereoNet declares
> `supported_hw_arch: hailo15h, hailo10h` while the published benchmark document
> is named `HAILO8_stereo_depth_estimation.rst` and reports Hailo-8 figures.
> Which device the ARM-P deployment actually targets is **UNKNOWN** and must be
> decided before C1. **[VERIFIED:
> `phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md:57-61` (yaml declares
> hailo15h/hailo10h, hailo8 not listed, doc named HAILO8);
> `docs/reference_pipeline.md:217-220` (same conflict, status UNKNOWN);
> `docs/hardware_analysis.md:369` (same conflict, status UNKNOWN).]**
> **[UNKNOWN — establishing it requires the user to nominate the target (or
> Hailo documentation resolving the yaml-vs-benchmark conflict); the parser,
> quantiser, memory budget, and performance targets all differ by device.]**

Remaining hardware items:

| Item | Statement | Status |
|---|---|---|
| Exact Hailo device | Undecided — Hailo-8, Hailo-8L, Hailo-10H, or Hailo-15 family | **UNKNOWN** — decided only by the user decision above; nothing in the repo nominates one for ARM-P |
| Hailo-8 class | 26 TOPS INT8; fully integrated on-chip memory; no on-module DRAM | **VERIFIED (EXTERNAL, dated 2026-09-19) — https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/** |
| Hailo-8L class | Lower-cost vision part (figures not established here) | **VERIFIED (EXTERNAL, dated 2026-09-19) — same Hailo product pages as above; no TOPS/memory figure for 8L is recorded in this contract because none was supplied** |
| Hailo-10H class | 40 TOPS INT4 / 20 TOPS INT8; 4 or 8 GB on-module LPDDR4/4X | **VERIFIED (EXTERNAL, dated 2026-09-19) — https://hailo.ai/products/ai-accelerators/hailo-10h-ai-accelerator/** |
| Hailo-15 family | Neural core integrated into a camera SoC with Arm cores — a different product class from the discrete accelerators | **VERIFIED (EXTERNAL, dated 2026-09-19) — Hailo product pages** |
| Accelerator memory (Hailo-8) | On-chip only; no on-module DRAM | **VERIFIED (EXTERNAL, dated 2026-09-19) — Hailo-8 product page; consistent with the reference compiler spatially defusing 55.34 MiB refinement activations [SOURCE: `docs/hardware_analysis.md` Sections 2, 6, via SR-003 + EXP-001]** |
| On-chip vs on-module (Hailo-10H) | On-module 4/8 GB LPDDR4/4X exists alongside on-chip memory | **VERIFIED (EXTERNAL, dated 2026-09-19) — Hailo-10H product page** |
| Which memory model ARM-P faces | Depends entirely on the undecided target device | **UNKNOWN — established by the target-device decision plus the DFC mapping report for the chosen device** |
| TOPS / precision available to ARM-P | 26 TOPS INT8 (Hailo-8) or 40 INT4 / 20 INT8 (Hailo-10H) at the device level; usable share is a compiler outcome | **VERIFIED (EXTERNAL) for the device ratings; UNKNOWN for ARM-P's realised utilisation — established only by compiling ARM-P for the chosen target** |
| DFC host platform | Linux x86 only (Ubuntu 20.04/22.04/24.04), 16+ GB RAM (32 GB recommended) | **VERIFIED (EXTERNAL, dated 2026-09-19) — https://hailo.ai/developer-zone/, https://community.hailo.ai/t/how-to-install-the-hailo-dataflow-compiler-dfc-on-wsl2/2890** |
| HailoRT host-OS support | Not established in this contract | **UNKNOWN — establishing it requires reading the HailoRT documentation (SR-030, recorded as not inspected in `docs/source_registry.md:86`); no HailoRT version is claimed anywhere below** |

---

## 3. Toolchain contract

| Item | Statement | Status |
|---|---|---|
| Dataflow Compiler version for ARM-P | No DFC exists on this machine, so no version is recorded for ARM-P | **VERIFIED: `phase2/deploy/validation/HAILO_COMPILATION_RECORD.json`, `toolchain_probe` — all 7 python modules not importable, all 6 binaries absent, no env vars, no install paths, `any_toolchain_found: false`; prose in `phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md` Sections 1–2** |
| HailoRT version | Nothing installed | **VERIFIED: same `toolchain_probe` record (no `hailo_platform`/`hailort` modules, no `hailortcli` binary)** |
| Model Zoo version behind the reference artifacts | v2.19.0 | **VERIFIED: `docs/source_registry.md:31,34` (SR-005, SR-006 rows, Model Zoo v2.19.0); `phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md:55` notes v2.19.0 as the upstream measurement condition** |
| DFC era behind the reference artifacts | v3.x-era tooling | **VERIFIED (EXTERNAL, dated 2026-09-19, per the manager-supplied brief)** — i.e. this row's provenance is the brief, not a repo file; the repo itself records only that the reference was built upstream |
| Latest public DFC versions | v5.1.0 referenced publicly, with v5.3.0 referenced for Hailo-10/15 | **VERIFIED (EXTERNAL, dated 2026-09-19) — Hailo Developer Zone / community install thread (URLs in Section 2)** |
| Version gap | Reference built on v2.19.0/v3.x-era stack vs current v5.x stack — whether the reference flow reproduces under the current toolchain is itself open | **UNKNOWN — establishing it requires installing a current DFC and re-running the reference parse, or finding first-party migration notes; until then no reference step may be assumed to transfer verbatim** |
| Host OS requirement | Linux x86 (Ubuntu 20.04/22.04/24.04); this host is Windows 11, so a Linux container or WSL2 distro must carry the DFC | **VERIFIED (EXTERNAL) for the requirement; VERIFIED for this host: `HAILO_COMPILATION_RECORD.json`, `host.platform: Windows-11-10.0.26200-SP0`, and `HAILO_TOOLCHAIN_REPORT.md:45-47`** |
| How the toolchain is obtained | Free registration at the Hailo Developer Zone; a Docker image and a WSL2 install route both exist | **VERIFIED (EXTERNAL, dated 2026-09-19) — https://hailo.ai/developer-zone/ and the WSL2 install thread** |
| What is installed here today | Nothing — no DFC, no HailoRT, no device, no Hailo image, Docker daemon unreachable, only a stopped `docker-desktop` WSL distro | **VERIFIED: `HAILO_TOOLCHAIN_REPORT.md` Sections 1, 8 and `HAILO_COMPILATION_RECORD.json` `toolchain_probe` (docker daemon_reachable false, hailo_images [])** |
| Hailo Emulator | The toolchain includes an emulator that can numerically evaluate a compiled model without physical hardware — the route to a quantised-accuracy number with no device | **VERIFIED (EXTERNAL, dated 2026-09-19) — https://github.com/hailo-ai/hailo_model_zoo** |
| PCIe device on this host | No Hailo accelerator present (vendor scan negative) | **VERIFIED: `HAILO_TOOLCHAIN_REPORT.md:43` (Get-PnpDevice, VEN_1E60, negative)** |

---

## 4. Model interface contract

All rows below describe the P2A exports (the only exports that exist). Each row
states whether it transfers to ARM-P.

| Item | P2A value (exported) | Status | ARM-P transfer |
|---|---|---|---|
| Input resolution | 368×1232, top-left crop, no resize | **VERIFIED: `P2A_DEPLOYMENT_VALIDATION.json:28-34,58-60`; `docs/reference_pipeline.md` Section 5** | **ASSUMED — rests on: identical architecture and identical frozen-contract scoring geometry; breaks if: the ARM-P export is traced at any other resolution, which would also break frozen-contract comparability** |
| Input tensor names | `left`, `right` | **VERIFIED: `HAILO_COMPILATION_RECORD.json`, `artifacts.original.inputs`** | **ASSUMED — rests on: reusing the P2A export scripts (`phase2/scripts/export_p2a.py`, `export_p2a_static.py`), which name them; breaks if: the ARM-P export script renames them (parser start-nodes must then be updated)** |
| Input shapes | `[1, 3, 368, 1232]` each | **VERIFIED: same record** | **ASSUMED, same basis and break condition as resolution** |
| Input dtype | FLOAT | **VERIFIED: same record (`dtype: FLOAT`)** | **ASSUMED, same basis** |
| Input layout | NCHW | **VERIFIED: `phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md` Section 1 ("ImageNet-normalised NCHW")** | **ASSUMED, same basis** |
| Normalisation | In-net vs host: the reference model inserts normalisation at parse time (`normalize_in_net: true`, ImageNet mean `[123.675, 116.28, 103.53]` / std `[58.395, 57.12, 57.375]`); the P2A ONNX itself receives pre-normalised input in host-side scoring | **VERIFIED: `HAILO_TOOLCHAIN_REPORT.md:119-123` (yaml values + "the ONNX itself must continue to receive pre-normalised input"); `docs/reference_pipeline.md:186-198`** | **UNKNOWN for ARM-P — no ARM-P export exists to inspect; establishing it requires performing the ARM-P export and recording whether normalisation is baked in or host-side. Working assumption (ASSUMED): same split as P2A — rests on reusing the P2A export path; breaks if the ARM-P export bakes normalisation in, which would double-normalise under P2A scoring code** |
| Output name / shape | `disparity`, `[1, 1, 368, 1232]` | **VERIFIED: `HAILO_COMPILATION_RECORD.json`, `artifacts.*.outputs`** | **ASSUMED, same export-reuse basis** |
| Output semantics | Disparity in full-resolution pixels | **VERIFIED: `phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md` Section 1; `docs/reference_pipeline.md:67-68,95-100`** | **ASSUMED, same basis; breaks only with an architecture change, which Stage C forbids** |
| Batch size | 1, fixed | **VERIFIED: `HAILO_COMPILATION_RECORD.json` (all I/O shapes batch 1); `docs/hardware_analysis.md:358`** | **ASSUMED, same basis** |
| Opset | 13 (both P2A artifacts; IR 7, producer pytorch 2.7.0) | **VERIFIED: `HAILO_COMPILATION_RECORD.json`, `artifacts.*.opset/ir_version/producer`** | **UNKNOWN for ARM-P until exported; ASSUMED 13 if the same exporter and script are reused — rests on exporter version being unchanged (torch 2.7.0); breaks if the export environment differs** |
| Static vs dynamic | Original export carries `Shape`/`Gather`/`Range`/`ReduceProd`; static export carries none (node delta −23, no op added) | **VERIFIED: `HAILO_COMPILATION_RECORD.json`, `cross_artifact` + `P2A_STATIC_EXPORT_VALIDATION.json:66-94`** | **UNKNOWN for ARM-P — both variants must be re-derived from the frozen ARM-P checkpoint; the static-export recipe (constant index buffer length 24, explicit `1/(D−1)` std identity with D = 24 literal, `scale_factor = 8.0`) is documented in `phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md` Section 5a for reuse** |

---

## 5. Operator and shape audit (paper audit only — no export, no compile)

Basis: P2A ONNX op counts in
`phase2/deploy/validation/HAILO_COMPILATION_RECORD.json`
(`artifacts.original.op_counts`, `artifacts.static.op_counts`) plus the model
source in `src/models/stereonet/`. State plainly: **none of this is a parser
result. It is a paper audit. The parser verdict is UNKNOWN** (establishing it
requires a DFC parser run against an ARM-P ONNX; see Sections 9 and 11).

### 5a. Full operator table (original / static export)

Counts VERIFIED per row against `HAILO_COMPILATION_RECORD.json`. Risk marks are
ASSUMED working judgements (basis and break conditions stated once here
rather than per row: they rest on the reference model compiling with the
overlapping op set for Hailo-8 [SOURCE: SR-005, `docs/hardware_analysis.md`
Section 3] and on standard NPU practice; they break if the chosen target
device, the v5.x compiler, or the parser's actual operator support differs —
all three are currently UNKNOWN, so every risk below is provisional).

| Op | Original | Static | Deployment risk (provisional) | Status |
|---|---|---|---|---|
| Conv | 51 | 51 | **Structural risk — see revised 5b, no longer framed as the largest** — the 5 aggregation `Conv3d` layers (`src/models/stereonet/aggregation.py:26-35`: four 32→32 plus one 32→1, 3×3×3) run on rank-5 tensors; 2D Convs are low risk. The reference StereoNet carried 5 Conv3D nodes on rank-5 internals with rank-4 model I/O and compiled to a Hailo-8 HEF, so the residual risk is the toolchain-gap / device / plumbing list in 5b, not blanket Conv3D-unsupported status | **VERIFIED (counts + source); risk ASSUMED per the paragraph above, revised by 5b** |
| Pad | 23 | 23 | **Load-bearing — see 5c. Any silent folding reverts the model to degenerate behaviour** | **VERIFIED (counts); load-bearing status VERIFIED via `phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md` Section 5 + `HAILO_TOOLCHAIN_REPORT.md:135-150`** |
| ConstantOfShape | 23 | 23 | Foldable in principle — each takes a literal `Constant` shape input — but compiler-dependent | **VERIFIED (counts + `phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md` Section 5a: "verified by tracing each node's producer … statically foldable by any parser"); fold outcome UNKNOWN until parsed** |
| Reshape | 47 | 46 | Layout-only; may be free or force copies | **VERIFIED (counts); cost UNKNOWN** |
| Slice | 47 | 46 | Low, but numerous | **VERIFIED (counts); cost UNKNOWN** |
| Transpose | 24 | 24 | Awkward if any is 5D (reference carried one 5D `[0,2,1,3,4]` transpose — that fact is about the REFERENCE, `docs/hardware_analysis.md:109`); per-Transpose roles in P2A not enumerated | **VERIFIED (counts); roles UNKNOWN — establishing them requires dumping the ONNX graph node list** |
| Concat | 26 | 25 | Low op risk; layout pressure | **VERIFIED (counts); cost UNKNOWN** |
| Unsqueeze | 24 | 24 | Layout-only | **VERIFIED (counts); cost UNKNOWN** |
| Squeeze | 1 | 1 | Layout-only (aggregation channel squeeze, `aggregation.py:40`) | **VERIFIED (counts + source)** |
| Add | 20 | 20 | Low | **VERIFIED (counts); cost UNKNOWN** |
| Sub | 27 | 25 | Low (includes the cost-volume subtractions, `cost_volume.py:110-111`) | **VERIFIED (counts + source)** |
| Mul | 3 | 3 | Low | **VERIFIED (counts); cost UNKNOWN** |
| Div | 2 | 1 | Low | **VERIFIED (counts); cost UNKNOWN** |
| Neg | 1 | 1 | Low (soft-argmin negation, `regression.py:49`) | **VERIFIED (counts + source)** |
| Sqrt | 1 | 1 | Low | **VERIFIED (counts); cost UNKNOWN** |
| ReduceMean | 3 | 1 | From the normalised readout (`regression.py:48`); low | **VERIFIED (counts + source)** |
| ReduceSum | 1 | 2 | Low (soft-argmin expectation, `regression.py:53`) | **VERIFIED (counts + source for the mechanism; count delta −/+ per record)** |
| LeakyRelu | 40 | 40 | Low; occasionally fused, occasionally not | **VERIFIED (counts); fusion UNKNOWN until compiled** |
| Relu | 1 | 1 | Low (final clamp, `stereonet.py:153-154`) | **VERIFIED (counts + source)** |
| Resize | 1 | 1 | **Moderate** — bilinear, `align_corners=True`, over the full 24-channel cost tensor to 368×1232 (`regression.py:66`) | **VERIFIED (counts + source); device cost UNKNOWN** |
| Softmax | 1 | 1 | **Moderate** — over the disparity axis (dim=1) at full resolution (`regression.py:49`) | **VERIFIED (counts + source); device cost UNKNOWN** |
| Cast | 25 | 23 | Low; mostly export plumbing | **VERIFIED (counts); per-node roles UNKNOWN** |
| Constant | 313 | 306 | Traced constants/initializers, not compute | **VERIFIED (counts + `HAILO_COMPILATION_RECORD.json` weights note)** |
| Shape | 3 | 0 | Dynamic-shape construction (regression `arange`/std/interpolate sizes, `phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md` Section 5) — absent in static | **VERIFIED (counts + doc attribution)** |
| Gather | 2 | 0 | Same as Shape | **VERIFIED (counts + doc attribution)** |
| Range | 1 | 0 | Same as Shape (`torch.arange(cost.shape[1])`, replaced by a constant index buffer of length 24 in static) | **VERIFIED (counts + Section 5a)** |
| ReduceProd | 1 | 0 | Same as Shape (unbiased-std denominator, replaced by explicit `1/(D−1)`, D = 24 literal) | **VERIFIED (counts + Section 5a)** |

### 5b. Conv3D / rank-5 tensors — precedent exists; residual risk is narrower

- The cost volume is `(1, 32, 24, 46, 154)` by construction: batch 1; 32
  feature channels; 24 disparity candidates; 46 = 368/8 rows and 154 = 1232/8
  columns at feature stride 8 (3 downsample levels). **[VERIFIED:
  `src/models/stereonet/cost_volume.py:118-119` (stack + permute to
  `(B, C, D, H, W)`); `src/models/stereonet/stereonet.py:86-88`
  (`feature_stride = 2**downsample_levels`); P2A config
  (`P2A_DEPLOYMENT_VALIDATION.json:17-22`: 3 levels, 24 disparities).]**
- The rank-5 family further includes the four `Conv3d(32→32)` filter outputs
  `(1, 32, 24, 46, 154)` and the `to_cost` output `(1, 24, 46, 154)` before the
  squeeze. **[VERIFIED: `aggregation.py:25-40`.]** Everything downstream of the
  squeeze is rank ≤ 4, ending at `disparity` `[1, 1, 368, 1232]`.
  **[VERIFIED: `HAILO_COMPILATION_RECORD.json` I/O shapes + source.]**
- Declared tensor ranks in the P2A ONNX read as `[4]` only because `value_info`
  is empty — the exporter did not emit intermediate shapes. This is NOT
  evidence that no 5-D tensor exists. **[VERIFIED:
  `HAILO_TOOLCHAIN_REPORT.md:184-193`.]**
- External constraint, quoted verbatim: *"Models that contain Conv3D layer must
  have rank-4 input and output (at most 4 dimensions), so the Conv3D layer must
  reside inside a '2D' model."* **[VERIFIED (EXTERNAL, dated 2026-09-19,
  quoting the DFC user guide) —
  https://community.hailo.ai/t/is-there-a-possibility-of-full-support-for-5d-input-and-conv3d-in-the-future/1621;
  hardware discussed: Hailo-8. A Hailo representative stated full rank-5 Conv3D
  support would require changes across the whole DFC and HailoRT and was not
  planned as of 2024.]** The constraint is about the **model's** input/output
  rank, not internal tensor rank.
- ARM-P's model I/O is rank-4 (`left`/`right` `[1,3,368,1232]` →
  `disparity` `[1,1,368,1232]`), so it satisfies the stated form of the
  constraint. **[VERIFIED: `phase2/deploy/validation/HAILO_COMPILATION_RECORD.json`,
  `artifacts.original.inputs` / `artifacts.original.outputs` (P2A export; ARM-P
  transfer ASSUMED per Section 4 — rests on reusing the P2A export path at the
  same resolution; breaks if the ARM-P export is traced at any other rank or
  shape).]**
- The reference StereoNet has the same shape — rank-4 model I/O with 5 Conv3D
  nodes on rank-5 internals — and that exact graph compiled to a Hailo-8 HEF.
  **[VERIFIED: worker measurement of `reference/onnx/stereonet.onnx` at
  correction time — inputs `input.1` `[1,3,368,1232]`, `input.83`
  `[1,3,368,1232]`, output `524` `[1,1,368,1232]`; Conv kernel ranks: 48 nodes
  with rank-2 kernels (2-D), 5 nodes with rank-3 kernels (3-D convolutions,
  all `[3,3,3]`, under `/cost_volume_filter/`); SR-005 `stereonet.hef`,
  24,057,165 bytes, `docs/source_registry.md:33`; the reference's 3-D
  aggregation also appears as compiled layers in Hailo's own profiler report —
  `docs/hardware_analysis.md:243` "Aggregation (3D) | 5 | 2,371,404,420 |
  4.2%" (SR-006 via EXP-018).]** Note that the reference `.alls` contains
  **nothing** about the cost volume or the 3-D aggregation — the reference
  parser handled both without intervention. **[VERIFIED:
  `phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md:124-129`.]**
- The residual risk is therefore narrower than "Conv3D may be unsupported",
  and honestly stated it is:
  (a) the toolchain version gap — the precedent is v2.19.0 / v3.x-era, the
  current stack is v5.x, and nothing guarantees the pass transfers
  **[UNKNOWN]**;
  (b) the target device is undecided, and the precedent is Hailo-8 only
  **[UNKNOWN]**;
  (c) P2A/ARM-P's rank-5 region is larger and differently plumbed than the
  reference's — 24 candidates vs 12, real shifts vs a no-op, 23 `Pad`
  clusters — so equal-shape does not mean equal-cost or equal-support
  **[UNKNOWN]**;
  (d) only a parser run settles it **[UNKNOWN]**.
- The "single largest C1 risk" wording is accordingly demoted. The leading C1
  risk is judged to be the **`Pad`-fold / disparity-shift survival** issue in
  Section 5c, because a silent fold produces a *plausible but wrong* model,
  whereas an unsupported operator produces a loud failure. **[ASSUMED — rests
  on: the VERIFIED load-bearing status of the 23 `Pad` clusters (Section 5c)
  plus the VERIFIED precedent that constant folding is normal compiler
  behaviour; breaks if: the v5.x parser rejects rank-5 internals outright, in
  which case Conv3D support retakes the lead.]**

### 5c. The 23 `Pad` clusters implementing the real disparity shift

- 23 clusters of (Pad + ConstantOfShape + Reshape + Cast + Constants), one per
  non-zero disparity candidate, implement the right-shift of the RIGHT features
  (`F.pad(x, (k, 0))` then slice to width, `cost_volume.py:45-56`), giving
  `cost_k(x) = left(x) − right(x−k)`. **[VERIFIED:
  `cost_volume.py:45-56,101-119`;
  `phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md` Section 5 ("437 of the 444
  extra nodes … 23 × (Pad + ConstantOfShape + Reshape + Cast + Constants)").]**
- These are load-bearing for ARM-P — unlike the reference model, whose shift
  is a proven no-op (EXP-010: all 12 slices bit-identical). **[VERIFIED:
  `HAILO_TOOLCHAIN_REPORT.md:137-150`; `src/models/stereonet/cost_volume.py:1-26`
  docstring; `docs/hardware_analysis.md:129`.]**
- Any compiler pass folding them away silently reverts the model to the
  degenerate reference behaviour. A post-compile numerical equivalence check
  against the host ONNX is therefore MANDATORY, not optional. **[ASSUMED
  (mandate) — rests on: the VERIFIED load-bearing status above plus the
  VERIFIED precedent that constant folding is a normal compiler behaviour;
  breaks (fails safe) if: the check is skipped and a fold happened, in which
  case deployed accuracy numbers would describe a degenerate model.]**

### 5d. Readout and reshape plumbing

- `Resize` (bilinear, `align_corners=True`) upsamples the whole cost tensor
  `(B, 24, 46, 154)` → `(B, 24, 368, 1232)` before the soft-argmin; `Softmax`
  runs over the disparity axis; expectation via the index grid; the static
  variant replaces the grid with a constant buffer of length 24.
  **[VERIFIED: `regression.py:33-67`;
  `phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md` Section 5a.]**
- `Transpose`/`Reshape`/`Slice` (24 / 47/46 / 47/46), `ReduceSum` (1/2),
  `LeakyRelu` (40) counts as tabulated. **[VERIFIED: compilation record.]**
  Per-node roles beyond the attributions in Sections 5a–5c are **UNKNOWN** —
  establishing them requires a node-name dump of the ONNX graph.
- The `Shape`/`Gather`/`Range`/`ReduceProd` dynamic-shape ops are present in
  the non-static export and absent in the static one (delta −23 nodes, no op
  added). **[VERIFIED: `HAILO_COMPILATION_RECORD.json`, `cross_artifact`.]**
  `ConstantOfShape` (23) remains in the static export and is statically
  foldable (literal-`Constant` shape inputs from the 23 `F.pad` calls, not the
  regression stage). **[VERIFIED: Section 5a doc.]**

---

## 6. Quantisation contract

| Item | Statement | Status |
|---|---|---|
| Supported precision path | Uniform int8: all 242 compiled reference layers report 8/8/8 (weights / input activations / output activations); no mixed precision, no 4-bit | **VERIFIED: `docs/hardware_analysis.md:295-305` (SR-006 via EXP-017)** |
| Calibration set | hailo_calib scenes 0–159 (160 scenes) — the same scenes P2A/ARM-P trained on; the split the frozen contract reserves for this purpose | **VERIFIED:   `src/datasets/kitti2015.py:35,125-128` (`CALIB_SPLIT_END = 160`); `HAILO_TOOLCHAIN_REPORT.md:130-133`; per-run `p2a_record.json` (`split_enforcement: train_base split='hailo_calib' only`)** |
| Calibration/eval overlap | The 160 calibration scenes (0–159) do NOT overlap the 40 evaluation scenes (160–199) | **VERIFIED: `src/datasets/kitti2015.py:125-128` (hailo_val takes `names[160:]`, hailo_calib takes `names[:160]`); every Tier-2 `p2a_record.json` records `overlap 0`; frozen contract scores exactly hailo_val 160–199 (`phase1/harness/frozen_eval.py:126-144`)** |
| Reference float-to-hardware degradation | Hailo publishes float 8.223 (config) / 8.22 (benchmark) and hardware 10.3 — about 2.08 points on Hailo's metric — and that cost is uniform int8 | **VERIFIED: `docs/hardware_analysis.md:170-172,301-305` (SOURCE SR-002, SR-004, SR-006)** |
| Our historical fp32→int8 result | EPE 1.313 → 1.655 px (+0.342 px), D1 8.154 → 10.945 % (+2.79 points), max per-pixel move 45.1 px, on onnxruntime CPU; int8 slower than fp32 there (677 vs 549 ms) as a CPU-QDQ artifact | **VERIFIED: `docs/hardware_analysis.md` Section 4 (MEASUREMENT EXP-015)** |
| What our int8 number predicts about Hailo int8 | Nothing. It is OUR measurement on OUR stack and is NOT a prediction of Hailo int8 — different quantiser, calibration, arithmetic, and metric aggregation | **VERIFIED (as a prohibition): `docs/hardware_analysis.md:168-181` ("the similarity is an observation, not evidence … UNKNOWN for any relationship")** |
| ARM-P quantisation outcome | No quantisation of ARM-P has been performed | **UNKNOWN — establishing it requires a DFC quantisation run with hailo_calib 0–159 followed by frozen-contract scoring of the quantised artifact (emulator suffices; no silicon needed)** |

---

## 7. Evaluation contract for deployed accuracy

- The frozen contract is unchanged: 40 scenes (hailo_val 160–199), 3,802,797
  valid pixels, `gt_scale` 256.0, source `disp_occ_0` (occluded included),
  metrics EPE and D1 (with RMSE/BAD1–3 recorded alongside), protocol-mixing
  guard refusing any non-matching comparison. **[VERIFIED:
  `phase1/harness/frozen_eval.py:28-47` (CONTRACT), `126-144`
  (`refuse_unless_contract`); `P2A_DEPLOYMENT_VALIDATION.json:91-105`
  (`contract_match true`, 3802797 px).]**
- How a device/emulator output would be scored: capture one 368×1232 float32
  disparity map in pixels per eval scene from the device (or emulator),
  pair each with its `disp_occ_0` ground truth scaled by 1/256, apply the
  valid mask (gt > 0) with the identical top-left crop, pool over all scenes,
  and run the unchanged `pooled_metrics` + `refuse_unless_contract` path.
  **[VERIFIED (procedure): `frozen_eval.py:121-144,147-182` — this is what the
  reference ONNX scorer does; only the prediction source changes.]**
- Wrapper code needed (named, NOT written): `stage_c_deploy/score_hef.py` —
  a thin harness that loads per-scene device/emulator disparity outputs,
  assembles them in eval-scene order, and calls the frozen-contract scoring
  path above with no augmentation, no multi-scale inference, and no threshold
  change. **[UNKNOWN — the wrapper does not exist; writing it is a C1 action.
  It must import, not duplicate, `phase1/harness/frozen_eval.py`, which is
  frozen.]**
- Host figures (P2A PyTorch CUDA EPE 1.4150748-class numbers, 50.6 ms CUDA,
  773 ms ORT CPU) and any future device figures must be kept strictly
  separate. **[VERIFIED (rule): `HAILO_TOOLCHAIN_REPORT.md:237-243`.]**

---

## 8. Performance / resource targets

Almost all of these are UNKNOWN — stated item by item, with the only two
reference figures that exist clearly fenced off as reference-only.

| Item | Statement | Status |
|---|---|---|
| ARM-P on-device latency | No number exists | **UNKNOWN — established only by running the compiled ARM-P on the chosen target (or emulator timing, suitably caveated); no proxy is accepted** |
| ARM-P FPS (batch 1 / batch 8) | No number exists | **UNKNOWN — same basis as latency** |
| ARM-P accelerator memory footprint | No number exists | **UNKNOWN — established only by the DFC mapping/placement report for ARM-P on the chosen target** |
| ARM-P power / FPS-per-watt | No number exists | **UNKNOWN — established only by on-device measurement; `docs/hardware_analysis.md:367` records power as UNKNOWN for the reference too** |
| ARM-P accelerator utilisation | No number exists | **UNKNOWN — established only by the compiled profiler report for ARM-P (cf. reference SR-006)** |
| CPU fallback budget (ops the compiler declines) | No number exists; which ops, if any, fall back to host is not known even for P2A | **UNKNOWN — `docs/hardware_analysis.md:360` records fallback mapping as UNKNOWN; establishing it requires the mapping report** |
| Reference published throughput | 10.7 FPS batch 1 / 11.6 FPS batch 8, Hailo-8, REFERENCE model | **VERIFIED (SOURCE, reference only): `docs/hardware_analysis.md:28,359`; SR-004. Must never be quoted as an ARM-P figure** |
| Reference post-placement profiler estimate | 43.03 FPS modelled bottleneck (`conv50`), 6 device contexts — REFERENCE model, compiler estimate, NOT a silicon measurement (`fps`/`latency` fields N/A) | **VERIFIED (SOURCE, reference only): `docs/hardware_analysis.md:256-293`. The ~4× gap to the 10.7 FPS published figure is a HYPOTHESIS (context-switch overhead), not a finding — `docs/hardware_analysis.md:287-293`** |
| ARM-P latency target | No ARM-P latency target has been set by anyone | **UNKNOWN — this is itself an open item the user must close: without a target, C1 measurements cannot pass or fail** |

---

## 9. Blockers

The four blockers from `HAILO_TOOLCHAIN_REPORT.md` Section 8, updated with the
External facts. Each carries the cheapest action that would close it.

1. **Hailo Dataflow Compiler is not installed** — blocks parser, optimizer,
   quantisation and compilation. **[VERIFIED:
   `HAILO_TOOLCHAIN_REPORT.md:101-103`.]** Update: the DFC is Linux-x86-only
   (Ubuntu 20.04/22.04/24.04, 16+ GB RAM) via free Developer Zone registration,
   with Docker-image and WSL2 install routes; current public versions are
   v5.1.0 / v5.3.0 (for Hailo-10/15) against the v2.19.0/v3.x-era reference
   stack — a version gap that is itself UNKNOWN (Section 3). **[VERIFIED
   (EXTERNAL, dated 2026-09-19) for the install facts; UNKNOWN for version-gap
   impact.]** Cheapest action: register at the Developer Zone and stand up the
   WSL2-Ubuntu route on this machine (no purchase, no new hardware).
2. **HailoRT is not installed** — blocks HEF loading and any runtime check.
   **[VERIFIED: `HAILO_TOOLCHAIN_REPORT.md:104`.]** Update: HailoRT ships with
   the toolchain flow, and the Emulator route (Section 3) yields a
   quantised-accuracy number with no device at all. **[VERIFIED (EXTERNAL) for
   the Emulator; UNKNOWN for HailoRT host requirements (SR-030 uninspected).]**
   Cheapest action: install HailoRT alongside the DFC in the same Linux
   environment; defer any device decision until emulator numbers exist.
3. **No Hailo accelerator is attached** — blocks all on-device measurement.
   **[VERIFIED: `HAILO_TOOLCHAIN_REPORT.md:105` + PCIe scan negative.]**
   Update: on-device measurement is NOT needed for the first quantised-accuracy
   number (Emulator), and real silicon is rentable per-run via a cloud Device
   Farm — but that route is time-limited (Section 10). **[VERIFIED (EXTERNAL)
   for both.]** Cheapest action: none yet — do not buy hardware before the
   emulator verdict; if silicon is needed, use the Device Farm before its
   2026-09-30 sunset.
4. **Docker daemon is not running and no Hailo image is present** — the
   containerised DFC route is unavailable as configured. **[VERIFIED:
   `HAILO_TOOLCHAIN_REPORT.md:106-107`; probe: daemon unreachable, zero Hailo
   images.]** Update: Docker is one of two DFC routes (the other is WSL2
   native); the daemon merely needs starting if that route is chosen.
   **[VERIFIED (EXTERNAL) for the route options.]** Cheapest action: start the
   Docker daemon and pull the Hailo DFC image — or skip Docker entirely for
   the WSL2 route, whichever is less work on this host.

None of these is a property of the model. **[VERIFIED:
`HAILO_TOOLCHAIN_REPORT.md:109`.]** Each is an environment gap.

---

## 10. Free / no-hardware routes to test this model

| Route | What it can establish | What it cannot establish | Status / notes |
|---|---|---|---|
| Linux/WSL2/Docker DFC install (Developer Zone, free registration) | Parser verdict on the ARM-P ONNX; quantisation; compilation to HEF; mapping + profiler reports | On-silicon latency/throughput/power; anything about the undecided target until one is chosen | **VERIFIED (EXTERNAL) that the route exists; UNKNOWN whether it succeeds until attempted (blockers 1, 4)** |
| Hailo Emulator (no silicon) | Numerical evaluation of the compiled (quantised) model — the first deployed-accuracy number under the frozen contract | Latency, FPS, power, utilisation, context-switch cost — it is a functional, not a timing, model | **VERIFIED (EXTERNAL, dated 2026-09-19) — https://github.com/hailo-ai/hailo_model_zoo** |
| DeGirum AI Hub Device Farm (HailoRT + Hailo-8/Hailo-8L silicon, no purchase) | Real-silicon runs: latency, FPS, on-device accuracy under the frozen contract | Anything after 2026-09-30 (access ends); compilation of our graph (see next column) | **VERIFIED (EXTERNAL, dated 2026-09-30 sunset) — https://degirum.com/ai-hub, https://docs.degirum.com/ai-hub/device-farm. Time-limited: access ends 2026-09-30 — eleven days after this contract's date. Any farm plan must be scheduled inside that window or not at all** |
| DeGirum Cloud Compiler | Nothing for ARM-P: it accepts only PyTorch checkpoints from the Ultralytics repository, not arbitrary ONNX — it cannot compile our ONNX. Its value, if any, is device access, not compilation | Compilation, quantisation, or mapping of our model | **VERIFIED (EXTERNAL, dated 2026-09-19) — https://docs.degirum.com/ai-hub/workspaces/cloud-compiler** |
| Pre-compiled reference HEF download (SR-005, already held) | Toolchain familiarity: what a HEF looks like byte-wise; nothing about ARM-P | No ARM-P accuracy, latency, or compatibility conclusion — it is a different model | **VERIFIED: `docs/source_registry.md:33` (SR-005: 24,057,165 bytes, acquired not decoded)** |

### 10a. Public stereo benchmarks (accuracy/generalisation routes — NOT deployment routes)

**None of these establishes anything about deployability.** They are
accuracy/generalisation routes, not deployment routes, and Stage C's open
question is deployment. Every route below scores a **different evaluation
contract** from our frozen one (40 KITTI-2015 training scenes, 3,802,797 valid
pixels, gt_scale 256.0, `disp_occ_0`) and therefore **cannot be compared
against 1.198874 px or 1.3134471 px**.

| Route | What it can establish | What it cannot establish | Status / notes |
|---|---|---|---|
| KITTI 2015 online test server (cvlibs.net) | Nothing for us in practice — see status | Any ARM-P number: submission is restricted and it scores the 200-image **test** split with withheld ground truth, a different contract from ours | **Effectively unavailable — VERIFIED (EXTERNAL, dated 2026-09-19) — https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php: only "submissions with significant novelty that are leading to a peer-reviewed paper in a conference or journal are allowed". This is a company engineering project, not a publication** |
| Middlebury 2014 stereo | Offline generalisation / failure-mode check only, on public training ground truth | Any comparison to frozen-contract EPE: higher resolution (1.5–5.9 MP), disparity ranges 256–800, imperfect rectification on most pairs — and a 24-candidate model cannot cover a 256–800 disparity range | **VERIFIED (EXTERNAL) for the dataset properties; VERIFIED (repo) for the status: already recorded as SR-023, not inspected, in `docs/source_registry.md:51`** |
| ETH3D stereo | Offline generalisation check only (13 training / 12 test scenes) | Any comparison to frozen-contract EPE — different contract, different scenes | **VERIFIED (EXTERNAL) for the benchmark size; VERIFIED (repo) for the status: already recorded as SR-033, not inspected, in `docs/source_registry.md:89`** |
| Our own frozen KITTI contract, run locally | The ONLY route that yields a number comparable to 1.198874 px and 1.3134471 px — same 40 scenes, same pixels, same guard | Anything about deployability on its own — it is the scoring procedure, not a device | **VERIFIED: `phase1/harness/frozen_eval.py`. Costs nothing and needs no network** |

---

## 11. Minimum next actions to close the UNKNOWNs

Ordered cheapest first. No training anywhere in this list.

1. User nominates the Stage C seed (Section 1) and the target device
   (Section 2); the chosen checkpoint is then frozen for all of Stage C.
   Closes: candidate UNKNOWN, target-device UNKNOWN, memory-model UNKNOWN.
2. User sets the ARM-P latency/FPS target (Section 8) — without it, C1 numbers
   cannot pass or fail.
3. Register at the Hailo Developer Zone (free) and stand up the DFC route on
   this host: WSL2-Ubuntu or Docker daemon + image (Sections 3, 9). Closes:
   blockers 1 and 4.
4. Install HailoRT in the same environment (Section 9, blocker 2). Record
   versions of everything (DFC, HailoRT, Model Zoo) verbatim.
5. Export the frozen ARM-P checkpoint to ONNX with the P2A scripts unchanged;
   record parity (host ONNX vs host PyTorch, ≤1e-3 px tolerance precedent from
   `P2A_DEPLOYMENT_VALIDATION.json:66-78`) and the static-variant derivation.
   Closes: all Section 4 transfer UNKNOWNs; produces the artifact Sections 5–7
   need.
6. Run the DFC parser on the ARM-P ONNX (start nodes `left`, `right`; end node
   `disparity`; ImageNet normalisation from the reference yaml); record the
   verdict verbatim, including any Conv3D/rank-5 ruling. Closes: parser
   UNKNOWN (Section 5). The run keeps this position because only it settles
   the residual (a)–(d) risks in 5b — not because Conv3D/rank-5 is the
   largest risk; the leading C1 risk is judged (ASSUMED) to be the Pad-fold
   / disparity-shift survival issue in 5c (a silent fold yields a plausible
   but wrong model, whereas an unsupported operator fails loudly).
7. Regenerate the model script for ARM-P layer names (do NOT reuse the
   reference `.alls`, whose every line targets the reference graph —
   `HAILO_TOOLCHAIN_REPORT.md:124-129`); quantise with hailo_calib 0–159.
8. Score the quantised artifact in the Hailo Emulator under the frozen
   contract via the `stage_c_deploy/score_hef.py` wrapper (Section 7, to be
   written in C1); run the MANDATORY Pad-survival equivalence check against
   the host ONNX (Section 5c) before trusting any number.
9. Only then, if silicon numbers are wanted: Device Farm run before the
   2026-09-30 sunset (Section 10), scoring device outputs under the unchanged
   frozen contract with host and device figures strictly separate.

---

## 12. Provenance

- Timestamp (UTC): 2026-09-19T15:16:51.128044+00:00 (inventory session; this
  contract written in the same session). **[VERIFIED: measured at inventory
  time via system clock.]**
- Correction-edit timestamp (UTC): 2026-09-19T15:25:28.317404+00:00 (Stage C0
  correction brief — Faults 1–3; targeted edits to Sections 5a, 5b, 10a, 11
  item 6, and this line only; original inventory timestamp above unchanged).
  **[VERIFIED: measured at correction time via system clock.]**
- Git HEAD: `58e8a19908ddbd35451652c61aef478f56b51ebd`. **[VERIFIED: `git
  rev-parse HEAD` at C0 time; matches the recorded head in
  `P2A_DEPLOYMENT_VALIDATION.json:6` and every Tier-2 `tier2_eval.json`.]**
- Host platform: Windows-11-10.0.26200-SP0. **[VERIFIED: measured at C0 time;
  matches `HAILO_COMPILATION_RECORD.json`, `host.platform`.]**
- Software: python 3.12.9, torch 2.7.0+cu128, onnx 1.22.0, onnxruntime 1.27.0,
  numpy 2.5.1. **[VERIFIED: measured at C0 time; matches the validation JSONs'
  `software` blocks.]**
- Checkpoint hashes (full values in `stage_c_deploy/c0_checkpoint_inventory.json`):
  seed 0 `58968771…3419818`; seed 1 `b2f6f5d5…fffeb7454`; seed 2
  `47a2d3e2…f1348f8b6d`; FT3D init `3ae6fb3b…f1a29be7` (matches
  `ARMP_CLOSURE_RECORD.md` Section 2); P2A seed 0 `0868ffd1…fbb6033`
  (matches `HAILO_COMPILATION_RECORD.json`, `source_checkpoint`).
  **[VERIFIED: read-only inventory, Section 1.]**
- Statement: no model artifact was modified. No file under `phase0/`,
  `phase1/`, `phase2/`, `stage_a_diagnostics/`, `stage_b_armp/`, `reference/`,
  `src/`, `scripts/`, `docs/`, or any existing `.md`/`.json` was written by
  this task; no checkpoint or ONNX file was re-saved, quantised, or
  re-exported; no training ran; no GPU job ran; no network access was used.
  The only files created are the two named below. **[VERIFIED: git status
  check at close (Section: Report).]**
- Seed EPEs are quotations, not measurements: 1.2057590 / 1.1912168 /
  1.1996447 (full precision in `ARMP_CLOSURE_RECORD.md` Sections 4–6). No
  evaluation ran for C0.

## Report

Files created (two, nothing else):

1. `stage_c_deploy/STAGE_C0_DEPLOYMENT_CONTRACT.md` (this file)
2. `stage_c_deploy/c0_checkpoint_inventory.json`

No existing file was modified.
