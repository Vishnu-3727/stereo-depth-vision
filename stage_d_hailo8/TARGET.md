# Stage D — target device

## Status

**TARGET ASSUMED FOR STAGE D — HAILO-8.**

This is an assumption of convenience for planning Stage D. It is **NOT** an
authoritative company requirement, and it must not be promoted to one anywhere
in this branch.

## What the assumption rests on

| Item | Value | Status |
|---|---|---|
| Accelerator | Hailo-8 | **ASSUMED** |
| Board | Raspberry Pi AI HAT+ 26 TOPS | **ASSUMED** |
| Host | Raspberry Pi 5 | **ASSUMED** |
| Evidence offered | A retail product listing (robu.in) for the Raspberry Pi AI HAT+ 26 TOPS, supplied by the project owner as the hardware *being considered* | Product exists; **not** a requirement document |
| Physical availability of the board | **UNKNOWN** — not confirmed, not seen, not tested | UNKNOWN |
| Company requirement naming this device | **NONE FOUND** | see below |

## Why this is only an assumption

Stage C ran an exhaustive search for authoritative device requirements and
found **zero** (`../stage_c_deploy/C1_TARGET_TOOLCHAIN_RESOLUTION.md`,
`../stage_c_deploy/target_resolution/evidence_summary.json`). No file in the
repository states which device the project must deploy on, and no file states a
required DFC or HailoRT version. Nothing found since changes that.

A product listing shows what hardware is *being considered for purchase*. It
does not state what the project is *required* to ship on, what latency, FPS,
power or accuracy the deployment must meet, or that the board is in hand.

## One consistency note (not a confirmation)

The "26 TOPS" in the product name is consistent with **Hailo-8**; the 13 TOPS
AI HAT+ variant carries the Hailo-8L. So *if* this board is the target, the
accelerator is Hailo-8 rather than Hailo-8L. This narrows the device class
under the assumption. It does not confirm the assumption.

## Devices that must NOT be silently substituted

Per the Stage D spec, and consistent with the Stage C reader hazards:

- Hailo-8L (13 TOPS AI HAT+ variant)
- Hailo-10H
- Hailo-15H
- any other Hailo device

The historical reference material in `../reference/` spans several of these
(Model Zoo v2.19.0 `hailo8`, master `hailo15h`/`hailo10h`, legacy HAILO8 docs,
DFC 3.34.0). All of it is **HISTORICAL REFERENCE** and none of it is "the
target".

## What would upgrade this to CONFIRMED

Any one of:

1. A company/competition document naming the device and its requirements.
2. Physical possession of the board, verified by `hailortcli fw-control identify`
   on the attached host.
3. A written instruction from the project owner that this device is the
   deployment target, recorded here with its date and author.

Until then this file reads ASSUMED, and every Stage D result must be read
against that.
