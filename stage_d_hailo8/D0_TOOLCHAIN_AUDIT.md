# D0 — target and toolchain audit

**VERDICT: BLOCKED. No Hailo toolchain and no confirmed Hailo device is
reachable from this project. D2 onwards cannot start.**

No model was changed by this stage, and none may be.

## 1 What D0 had to establish

Per the Stage D spec: actual Hailo-8 availability, and the compiler, HailoRT,
device support, supported ops, supported input shapes, supported layers,
quantization workflow and HEF generation workflow.

## 2 Measured on this host

| Probe | Result |
|---|---|
| `hailo`, `hailortcli` on PATH | **ABSENT** |
| `import hailo_sdk_client` | **ABSENT** |
| `import hailo_platform` | **ABSENT** |
| `import hailo_model_zoo` | **ABSENT** |
| Operating system | Windows 11 (10.0.26200), AMD64 |
| Attached Hailo device | **NONE** |

The Hailo Dataflow Compiler is distributed for x86-64 **Linux**. This host is
Windows, so the DFC cannot run on it as configured. No Linux, WSL2 or container
host has been identified for Stage D — that is **UNKNOWN**, not "unavailable".

## 3 Target availability

**UNKNOWN.** See `TARGET.md`. The device is ASSUMED from a retail product
listing supplied by the project owner. Nobody has verified that the board
exists physically, and no requirement document names it.

Consequence: even if a compiler host appeared tomorrow, D3 (target execution)
and D4 (performance) would still be blocked on hardware, and any compile would
have to declare its `--hw-arch` from an assumption rather than a requirement.

## 4 Supported ops / shapes / layers

**NOT ESTABLISHED FROM THE TOOLCHAIN.** Support lists are properties of a
specific DFC version, and no DFC is installed. Asserting a support list from
documentation alone would be exactly the kind of unverified claim this project
does not make.

What *is* established, from artefacts already in the repository, is a structural
comparison against a model the Hailo-8 toolchain demonstrably accepted. That is
D1, and it is genuine evidence about the model family — but it is **not** a
substitute for a compiler run.

## 5 Historical toolchain evidence (reference only)

Carried over from `../stage_c_deploy/C1_TARGET_TOOLCHAIN_RESOLUTION.md`. All of
it is **HISTORICAL REFERENCE**; none of it is a requirement and none of it is
"the target":

| Item | Value |
|---|---|
| Model Zoo v2.19.0 `stereonet.yaml` | `supported_hw_arch: [hailo8]` |
| Model Zoo master `stereonet.yaml` | `supported_hw_arch: hailo15h, hailo10h` (hailo8 not listed) |
| `../reference/stereonet.hef` | v2.19.0 Hailo-8 build; header carries version-like string `3.34.0` |
| Profiler HTML | `hw_arch: hailo8`, 242 layers, uniform 8/8/8 |
| DFC line for Hailo-8/8L | v3.x, v3.34.0 current at the time of that audit |
| Required versions for this project | **UNKNOWN** — no document states any |

## 6 Stop condition

The spec lists as absolute stop conditions: *target hardware cannot be
confirmed*, and *Hailo toolchain cannot be installed/accessed*. Both hold.

D0 therefore terminates here rather than improvising. What is **not** blocked is
static analysis of the artefacts already in hand, which is why D1 exists as a
partial record. D2, D3 and D4 are **NOT ATTEMPTED**, and no report files were
created for them.

## 7 What would unblock D0

| Need | Unblocks |
|---|---|
| A Linux x86-64 host (bare metal, WSL2 or container) with the Hailo AI Software Suite / DFC installed | D2 compile attempt |
| The Raspberry Pi AI HAT+ 26 TOPS board physically attached to a Pi 5, with HailoRT installed | D3 execution, D4 performance |
| A written statement of the deployment requirement — device, latency/FPS, power, accuracy | Turns `TARGET.md` from ASSUMED to CONFIRMED, and gives D4 a criterion to pass or fail against |
| The DFC version to target | Removes a guess from D2 |

Until the first of these exists, Stage D cannot produce target-native evidence,
and no amount of host-side work substitutes for it.
