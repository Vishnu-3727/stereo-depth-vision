# ARM-P-H8-V0

**ARM-P-H8-V0 = EXACT FROZEN ARM-P.** No architecture, training, weight, loss,
augmentation, candidate-count or normalisation change. Its only purpose is to
answer: *can the existing ARM-P actually reach Hailo-8?*

## The binaries are not committed

`armp_h8_v0.pth`, `armp_h8_v0.onnx` and `armp_h8_v0_static.onnx` are
byte-identical duplicates of artefacts git already tracks, so they are left
untracked (the repository's `*.pth` / `*.onnx` ignore rules) rather than stored
twice.

Recreate and verify them in one step:

```bash
python stage_d_hailo8/verify_v0.py --create
```

## What is copied, and from where

| v0 file | Source (tracked, frozen) | SHA-256 |
|---|---|---|
| `armp_h8_v0.pth` | `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth` | `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` |
| `armp_h8_v0.onnx` | `stage_c_deploy/armp_stereonet.onnx` | `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989` |
| `armp_h8_v0_static.onnx` | `stage_c_deploy/armp_stereonet_static.onnx` | `e275e86e4cdb340b64b3ccd065ccd8286b82c3ef1dbbd046acaaa1f0c363a5af` |

The first two hashes are the spec-declared values and are **asserted**. The
static export carries no spec-declared hash, so its value is recorded and
checked against the tracked original only.

A mismatch is a **STOP** condition. The copy is never silently "fixed"; the
mismatch is reported and Stage D halts.

## Which file goes to the compiler

`armp_h8_v0_static.onnx`. The static export drops the dynamic-shape machinery
(712 → 689 nodes, 27 → 23 op types) while keeping identical I/O shapes. See
`../D1_EXPORT.md`.
