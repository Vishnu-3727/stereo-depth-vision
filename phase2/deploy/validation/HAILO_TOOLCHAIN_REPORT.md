# HAILO TOOLCHAIN REPORT — P2A deployment validation

Machine record: `phase2/deploy/validation/HAILO_COMPILATION_RECORD.json`.
Probe + integrity script: `phase2/deploy/validation/verify_onnx_preflight.py`.

**No ONNX artifact was modified, simplified, quantised or re-exported. No
training was run. No research artifact was touched.**

---

## 0. Validation ladder — explicit, not collapsed

| layer | status |
|---|---|
| **ONNX validated** | **PASS** — both artifacts, full integrity set below |
| **Hailo parser validated** | **NOT ATTEMPTED** — toolchain absent |
| **Hailo compiler validated** | **NOT ATTEMPTED** — toolchain absent |
| **HEF validated** | **NOT ATTEMPTED** — no HEF produced |
| **HailoRT runtime validated** | **NOT ATTEMPTED** — HailoRT absent |
| **Physical hardware validated** | **NOT AVAILABLE** — no Hailo device present |

The furthest verified point is **ONNX**. Everything beyond it is unverified,
and nothing beyond it is claimed.

---

## 1. Toolchain discovered

**None.** The probe was exhaustive, and every check came back negative.

| probe | result |
|---|---|
| pip packages matching `hailo` / `dataflow` | none |
| python imports — `hailo_sdk_client`, `hailo_sdk_common`, `hailo_platform`, `hailort`, `hailo_model_zoo`, `hailo_tools`, `hailo_dataflow_compiler` | all `ModuleNotFoundError` |
| binaries on PATH — `hailo`, `hailortcli`, `hailo_compiler`, `hailomz`, `hailo-dfc`, `dfc` | none found |
| environment variables containing `HAILO` | none |
| install directories — `C:\Program Files\Hailo`, `C:\Program Files (x86)\Hailo`, `C:\Hailo`, `%LOCALAPPDATA%\Hailo`, `~\.hailo`, `%APPDATA%\Hailo`, `/opt/hailo`, `/usr/local/hailo` | none exist |
| Windows installed-programs registry, `DisplayName` matching `hailo` | none |
| Docker | CLI present at `C:\Program Files\Docker\Docker\resources\bin\docker.exe`; daemon not reachable; **zero** images matching `hailo` |
| WSL distributions | only `docker-desktop` (Stopped) — no Hailo Linux distro |
| other virtualenvs on the machine (7 found) | none Hailo-related |
| filesystem sweep for `*hailo*` under `%USERPROFILE%` and `C:\Program Files` | only this project's own reference material (model-zoo YAML/ALLS/RST, upstream zips) — no compiler, no SDK |
| **PCIe device scan** (`Get-PnpDevice`, vendor `VEN_1E60`) | **no Hailo accelerator present** |

Host: Windows-11-10.0.26200, python 3.12.9, torch 2.7.0+cu128, onnx/onnxruntime
1.27.0. The Hailo Dataflow Compiler is distributed for Linux; this is a Windows
host with no Linux container or WSL distro carrying it.

## 2. Version(s)

| item | version |
|---|---|
| installed Hailo Dataflow Compiler | **none installed** |
| installed HailoRT | **none installed** |
| DFC version used for the *reference* model, per local documentation (`reference/hailo_model_zoo/HAILO8_stereo_depth_estimation.rst`) | **v2.19.0** — this is the upstream measurement condition, **not** something running here |

Also read from `reference/hailo_model_zoo/stereonet.yaml`, and worth carrying
forward: the reference entry declares `supported_hw_arch: hailo15h, hailo10h`
— **hailo8 is not listed**, despite the documentation file being named
`HAILO8_stereo_depth_estimation.rst`. Whoever runs the compile should confirm
the intended target before assuming Hailo-8.

## 3. Original ONNX parser result

**NOT ATTEMPTED.** `phase2/deploy/p2a_stereonet.onnx` was not submitted to a
parser because no parser exists on this machine. No acceptance, rejection,
unsupported-operator list, warning or error can be reported, and none is
invented.

## 4. Static ONNX parser result

**NOT ATTEMPTED**, same reason. `phase2/deploy/p2a_stereonet_static.onnx` was
not submitted to a parser.

## 5. Which artifact was selected

**None — no toolchain-driven selection was possible.** Decision cases A, B and
C in the brief all require a parser result. The situation is upstream of all
three: the toolchain itself is absent.

Both artifacts remain available and both passed every pre-compilation integrity
check. `p2a_stereonet.onnx` stays the primary compilation input;
`p2a_stereonet_static.onnx` is the fallback if the parser rejects dynamic-shape
construction.

## 6. HEF compilation result

**NO HEF PRODUCED.** No compilation was attempted.

## 7. Hardware / HailoRT result

**NOT AVAILABLE.** No Hailo device is present on this machine (PCIe scan
negative) and HailoRT is not installed. No device name, chip revision, latency,
throughput, memory figure or on-device accuracy number exists, and none is
fabricated.

---

## 8. Exact blockers

1. **Hailo Dataflow Compiler is not installed** — blocks parser, optimizer,
   quantisation and compilation. It is Linux-only and requires a Hailo
   developer-zone account to obtain.
2. **HailoRT is not installed** — blocks HEF loading and any runtime check.
3. **No Hailo accelerator is attached** — blocks all on-device measurement.
4. **Docker daemon is not running and no Hailo image is present** — the
   containerised route to the DFC is unavailable as configured.

None of these is a property of the model. Each is an environment gap.

### What a compile attempt will additionally need (from local documentation)

Recorded so the eventual attempt is not a guessing exercise, and flagged as
preparation rather than as validation:

- **Start/end node names.** The parser needs explicit graph boundaries. P2A's
  inputs are `left` and `right`, output `disparity` — named at export, unlike
  the reference's `input.1` / `input.83` / `524`.
- **Normalisation.** The reference is parsed with `normalize_in_net: true` and
  mean `[123.675, 116.28, 103.53]` / std `[58.395, 57.12, 57.375]`
  (`stereonet.yaml`). P2A uses the identical ImageNet statistics, so the same
  parser normalisation applies and the ONNX itself must continue to receive
  pre-normalised input in host-side scoring.
- **The existing `stereonet.alls` will not transfer as-is.** Every line in it is
  spatial defusion of `conv42`–`conv52` — the full-resolution refinement stage —
  keyed to the reference graph's layer names. P2A's graph has different node
  names and 4× more nodes, so the model script must be regenerated. Notably the
  `.alls` contains **nothing** about the cost volume or the 3-D aggregation:
  the reference parser handled those without intervention.
- **Calibration set.** The reference quantises with a KITTI stereo calibration
  tfrecord. The equivalent here is hailo_calib scenes 0–159 — the same 160
  scenes P2A trained on, which is the split the frozen contract reserves for
  exactly this purpose.

### Model-specific risk to carry into the compile attempt — inference, not measurement

The reference model's disparity shift is a **proven no-op** (EXP-010: all 12
slices bit-identical), so its cost volume is degenerate and may well have been
collapsed by constant folding during parsing. **P2A's shift is real**, and its
23 `Pad` clusters are load-bearing — they implement the disparity search that
produced the accuracy this deployment exists for.

**Do not let any toolchain "optimise away" the disparity-shift padding.** If a
compiler pass removes it, the model silently reverts to the degenerate
reference behaviour. A post-compile equivalence check against the host ONNX
output is therefore mandatory, not optional.

This risk is **inference** (grade C), not a measured parser result. Whether the
reference's fold actually happened, and whether P2A's cost volume compiles at
all, is exactly what the absent toolchain would tell us.

---

## 9. Pre-compilation integrity — verified, both artifacts

Every item the brief required, checked on both files:

| check | original | static |
|---|---|---|
| `onnx.checker` | **PASS** | **PASS** |
| opset | **13** ✔ | **13** ✔ |
| IR version / producer | 7 / pytorch 2.7.0 | 7 / pytorch 2.7.0 |
| input names | `left`, `right` ✔ | `left`, `right` ✔ |
| input shapes | `[1,3,368,1232]` ✔ | `[1,3,368,1232]` ✔ |
| output name / shape | `disparity` `[1,1,368,1232]` ✔ | `disparity` `[1,1,368,1232]` ✔ |
| fully static I/O shapes | ✔ | ✔ |
| BatchNorm / InstanceNorm ops | **none** ✔ | **none** ✔ |
| trained tensors found bit-identical in the graph | **70 / 70** | **70 / 70** |
| matched parameter elements | **397,954** ✔ | **397,954** ✔ |
| nodes | 712 | 689 |
| size | 1,701,550 B | 1,699,043 B |
| SHA-256 | `0995a6c5722b4625…` | `0665a9ae7d045596…` |
| dynamic-shape ops | `Shape`, `Gather`, `Range`, `ReduceProd` | **none** |
| overall | **ALL_OK** | **ALL_OK** |

Source checkpoint `phase2/runs/p2a_scale_coverage/p2a_best.pth`: 397,954
torch parameters, 70 state_dict tensors, no BatchNorm modules.

Cross-artifact: both match the trained weights bit-identically; the static
export differs only by −23 nodes and the removal of `Shape`, `Gather`, `Range`,
`ReduceProd`. **No operator was added.**

Two honest limits on the above:

- The ONNX initializer count exceeds 70 because traced constants are stored as
  initializers too. The check that matters, and the one performed, is that
  every trained tensor appears bit-identical in the graph and that they total
  exactly 397,954 elements.
- Declared tensor ranks read as `[4]` only because `value_info` is empty in
  both graphs — the exporter did not emit intermediate shapes. This is **not**
  evidence that no 5-D tensor exists; the cost volume is `(1,32,24,46,154)` by
  construction. Rank support is one of the things the absent parser would
  decide.

---

## 10. Files written

```
phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md      this report
phase2/deploy/validation/HAILO_COMPILATION_RECORD.json  probe + integrity, machine-readable
phase2/deploy/validation/verify_onnx_preflight.py       the script that produced both
```

One correction made to an earlier document: `PHASE2_DEPLOYMENT_VALIDATION.md`
§5 described the reference ONNX as "opset 9, 143 nodes". The measured values
are **opset 14, 168 nodes** (they were already correct in
`P2A_DEPLOYMENT_VALIDATION.json`; the markdown figure was mistyped). Corrected
in place.

## 11. Statement on research artifacts

No research or accuracy artifact was modified by this validation. Specifically
unchanged: `phase1/` in full, `phase1/results/LEADERBOARD.md`, `reference/`,
all ARM-V/W/X/Y/Z artifacts and verdicts, the P2A experiment verdict
(**INCONCLUSIVE**), the P2A mechanism verdict (**NOT CONFIRMED**), the formal
experimental incumbent (**ARM-V**), the P2A deployment-selection record, both
ONNX artifacts, and all trained checkpoints. No optimization arm was created.
No training was run. Phase 2 remains the final phase; no Phase 3 exists.

---

## 12. To resume when a toolchain exists

1. Install the Hailo Dataflow Compiler (Linux host, WSL2 Ubuntu, or the Hailo
   Docker image) and confirm the version.
2. Confirm the target architecture — the model-zoo entry lists `hailo15h` and
   `hailo10h`, not `hailo8`.
3. Parse `phase2/deploy/p2a_stereonet.onnx` with start nodes `left`, `right`
   and end node `disparity`, plus the ImageNet normalisation from
   `stereonet.yaml`. Record the parser's verdict verbatim.
4. If, and only if, it is rejected for dynamic-shape construction, parse
   `phase2/deploy/p2a_stereonet_static.onnx` instead and record that the
   toolchain selected the static artifact.
5. Generate a fresh model script for P2A's layer names; do not reuse the
   reference `.alls`.
6. Quantise with hailo_calib scenes 0–159. Compile. Record the HEF path.
7. **Verify the disparity-shift padding survived** by comparing HEF output
   against the host ONNX on the same inputs before trusting any number.
8. If hardware is attached, load the HEF with HailoRT, measure latency and
   throughput, and score the device output under the unchanged frozen contract
   — keeping host figures (EPE 1.4150748, 50.6 ms CUDA, 773 ms ORT CPU) and
   device figures strictly separate.
