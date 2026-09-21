# Stage D — baseline inventory

Recorded before any compilation attempt (spec section 6). Unknown values are
written **UNKNOWN**; nothing here is guessed.

## 1 Frozen reference — ARM-P seed 1

Immutable. Stage D never modifies, retrains, overwrites or re-hashes it.

| Item | Value |
|---|---|
| EPE (frozen 40-scene contract) | **1.1912168 px** |
| Checkpoint SHA-256 | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` |
| ONNX SHA-256 | `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989` |
| Parameters | 397,954 |
| Tensors / keys | 70 |
| Disparity candidates | 24 |
| Downsample levels | 3 |
| Shift convention | real disparity shifts, right-shift |
| `regression_normalize` | true |
| Contract | KITTI, 40 scenes, 3,802,797 valid pixels, `gt_scale` 256.0, `disp_occ_0`, existing frozen evaluator |

Historical results that Stage D preserves and must not rewrite:

| Record | Verdict | Value |
|---|---|---|
| C1 ONNX export parity | **FAIL** (historical) | 0.001708984375 px against a 1e-3 px criterion |
| DR-1 H1 rescale gate | **FAIL** (historical) | 1.4816284e-02 px against a 1e-3 px criterion |

Neither proves Hailo incompatibility, and neither proves compatibility. Stage D
exists to obtain target-native evidence.

## 2 ARM-P-H8-V0

**ARM-P-H8-V0 = EXACT FROZEN ARM-P.** No architecture, training, weight, loss,
augmentation, candidate-count or normalisation change.

Verified byte-identical to the frozen originals by `verify_v0.py`
(`hashes/v0_integrity.json`), verdict **PASS**:

| Copy | SHA-256 | Matches frozen |
|---|---|---|
| `v0/armp_h8_v0.pth` | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` | yes |
| `v0/armp_h8_v0.onnx` | `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989` | yes |
| `v0/armp_h8_v0_static.onnx` | `e275e86e4cdb340b64b3ccd065ccd8286b82c3ef1dbbd046acaaa1f0c363a5af` | yes |

The static export carries no spec-declared hash, so its value is **recorded**
here rather than asserted against the spec. It is byte-identical to
`../stage_c_deploy/armp_stereonet_static.onnx`.

## 3 Target inventory

| Item | Value |
|---|---|
| Target device | Hailo-8 — **ASSUMED**, see `TARGET.md` |
| Board | Raspberry Pi AI HAT+ 26 TOPS — **ASSUMED** |
| Host Raspberry Pi model | **UNKNOWN** (a Pi 5 exists on this network for an unrelated project; whether it is the Stage D host, and whether any AI HAT+ is attached to it, is **UNKNOWN**) |
| Device physically available | **UNKNOWN** |
| Device access from this machine | **NONE** |
| Hailo hardware serial / firmware | **UNKNOWN** |

## 4 Toolchain inventory

| Item | Value |
|---|---|
| Hailo Dataflow Compiler | **NOT INSTALLED** on this host |
| HailoRT | **NOT INSTALLED** on this host |
| `hailo_sdk_client` | **NOT INSTALLED** (import probe: absent) |
| `hailo_platform` | **NOT INSTALLED** (import probe: absent) |
| Hailo Model Zoo | **NOT INSTALLED** as a package (a source copy exists under `../reference/`, historical reference only) |
| `hailo` / `hailortcli` on PATH | **ABSENT** |
| Required DFC version | **UNKNOWN** — no requirement document exists |
| DFC version implied by historical artefacts | 3.34.0 (HISTORICAL REFERENCE, see `../stage_c_deploy/C1_TARGET_TOOLCHAIN_RESOLUTION.md`) |

## 5 Development host inventory

This is the machine the audit ran on. It is **not** a Hailo toolchain host.

| Item | Value |
|---|---|
| Operating system | Windows 11, 10.0.26200 |
| Architecture | AMD64 (x86-64) |
| Shell used | MSYS/MinGW bash and PowerShell |
| Python | 3.12.9 |
| PyTorch | 2.7.0+cu128 |
| ONNX | 1.22.0 |
| ONNX Runtime | 1.27.0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Compiler / toolchain paths for Hailo | **NONE** |

The Hailo Dataflow Compiler is distributed for x86-64 Linux (Ubuntu). This host
is x86-64 but runs Windows, so it cannot run the DFC directly. Whether a Linux
machine, WSL2 installation, or container host is available for Stage D is
**UNKNOWN**.

## 6 Reference (non-baseline) artefacts

Historical vendor material already in the repository, used in D1 for structural
comparison only. It is **not** the ARM-P baseline and never becomes one.

| Artefact | Note |
|---|---|
| `../reference/onnx/stereonet.onnx` | Vendor StereoNet ONNX |
| `../reference/stereonet.hef` | Vendor Hailo-8 HEF, Model Zoo v2.19.0 |
| `../reference/hailo_model_zoo/` | Vendor configs and docs |

Per spec section 17, the teammate's / vendor's Hailo stereo model is contextual
comparison only: different architecture, different accuracy protocol, different
deployment configuration.
