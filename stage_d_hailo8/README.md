# Stage D — Hailo-8 deployment tuning

A **separate deployment-engineering branch**. It is not a reopening of the
closed MICROCHIP STEREONET research project, which remains frozen and
authoritative.

Stage D asks one question: **what is actually required to run ARM-P on the
assumed Hailo-8 target?** Not "prove ARM-P deploys".

## Current state

```
TARGET:              Hailo-8 / Raspberry Pi AI HAT+ 26 TOPS   [ASSUMED]
FROZEN ARM-P:        1.1912168 px, hash verified               PASS
ARM-P-H8-V0:         byte-identical copy of frozen ARM-P       PASS
EXACT ARM-P COMPILE: NOT ATTEMPTED  (no Dataflow Compiler host)
HEF:                 NO
TARGET EXECUTION:    NOT ATTEMPTED  (no confirmed hardware)
NUMERICAL PARITY:    NOT MEASURED
PERFORMANCE:         NOT MEASURED
MODIFIED MODEL:      NONE
CURRENT BLOCKER:     D0 - no Hailo toolchain on any reachable host,
                     and the target device is unconfirmed
```

## Stage progress

| Stage | Status | Record |
|---|---|---|
| D0 target / toolchain audit | **BLOCKED** | `D0_TOOLCHAIN_AUDIT.md` |
| D1 exact ARM-P export | **PARTIAL** — structure measured, compiler acceptance not established | `D1_EXPORT.md` |
| D2 quantization / compile | **NOT ATTEMPTED** | — |
| D3 target execution | **NOT ATTEMPTED** | — |
| D4 performance | **NOT ATTEMPTED** | — |

Reports exist only for stages actually executed. No file in this branch
describes a result that was not measured.

## The two findings so far

**1. ARM-P-H8-V0 is exactly the frozen ARM-P.** Verified byte-for-byte:
checkpoint `b2f6f5d5…eb7454` and ONNX `4277090d…dbcb6989` both match the
spec-declared hashes, and the copies match the originals
(`hashes/v0_integrity.json`, PASS).

**2. The 3D cost volume is not a Hailo-8 blocker for this model family.**
ARM-P carries 5 Conv3d nodes and rank-5 tensors, which is the first thing that
looks disqualifying on a 4D dataflow accelerator. The vendor StereoNet carries
**the same** 5 Conv3d, the same max rank 5, the same rank-5-producing op types
and the same I/O shapes — and Hailo published a compiled Hailo-8 HEF for it. So
the 3D structure must not be used to justify redesigning the model.

The real compile risk, if there is one, is ARM-P's shift construction: 689
nodes against the vendor's 168, and 12 op types the vendor never uses
(`Pad`, `Reshape`, `ConstantOfShape`, `Cast`, `Slice` machinery). That is a
hypothesis for D2, not a finding. See `D1_EXPORT.md`.

## Layout

```
stage_d_hailo8/
  README.md                  this file
  TARGET.md                  the device, and why it is ASSUMED not CONFIRMED
  BASELINE.md                frozen reference, toolchain and host inventory
  D0_TOOLCHAIN_AUDIT.md      what is installed and reachable: nothing Hailo
  D1_EXPORT.md               structural audit of the exact frozen graph
  verify_v0.py               artifact-integrity gate (spec section 7)
  graph_audit.py             structural ONNX audit vs the vendor model
  v0/                        ARM-P-H8-V0 = exact frozen ARM-P
  hashes/v0_integrity.json   integrity result
  reports/d1_graph_audit.json  measured graph facts
  experiments/ artifacts/ logs/   empty until D2 can run
```

## Rules this branch runs under

- **The frozen ARM-P is immutable.** Never modified, retrained, overwritten or
  re-hashed. Every candidate is compared against it, never against another
  candidate.
- **Stage C records are not edited.** Stage D results never appear
  retroactively inside the closed project. C1 stays **FAIL**
  (0.001708984375 px vs 1e-3 px); DR-1 H1 stays **FAIL** (1.4816284e-02 px).
- **No premature model tuning.** Large activations, a failed C1, a failed DR-1,
  a difficult-looking compile or a warning are not reasons to change the model.
  A modification requires a concrete deployment failure, a measurable
  criterion, a hypothesis, and a controlled baseline against ARM-P-H8-V0.
- **No research reopening.** A deployment issue that raises a research question
  is recorded as an OPEN DEPLOYMENT QUESTION and not investigated.
- **Accuracy is not the only objective.** A candidate is never selected on EPE
  alone, nor on "it compiles" alone.

## To reproduce what exists

```bash
python stage_d_hailo8/verify_v0.py      # integrity gate
python stage_d_hailo8/graph_audit.py    # structural audit
```

Both are read-only with respect to every frozen artefact.
