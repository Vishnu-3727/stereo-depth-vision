# D1 — exact ARM-P export (structural audit)

**VERDICT: PARTIAL. The graph is valid, static and structurally the same
family as a model the Hailo-8 toolchain demonstrably compiled. Compiler
acceptance is NOT ESTABLISHED and cannot be, on this host.**

No model was changed. ARM-P-H8-V0 is the exact frozen ARM-P, hash-verified
(`hashes/v0_integrity.json`, verdict PASS).

## 1 What could and could not be done

D1 asks for graph validity, I/O shapes, operator compatibility, static vs
dynamic dimensions, and preprocessing/postprocessing assumptions.

Graph validity, shapes and dimensions are properties of the ONNX file and were
**measured**. *Operator compatibility* is a property of a specific Dataflow
Compiler version, and no DFC is installed (D0) — so it is **NOT ESTABLISHED**.
What replaces it here is a structural comparison against the vendor StereoNet,
which has a published Hailo-8 HEF. That is evidence about the model family, not
a compile result.

Source of every number below: `reports/d1_graph_audit.json`, produced by
`graph_audit.py` (read-only).

## 2 Measured graph facts

| | ARM-P-H8-V0 | ARM-P-H8-V0 static | Vendor StereoNet |
|---|---|---|---|
| Nodes | 712 | 689 | 168 |
| Op types | 27 | 23 | 15 |
| Conv2d | 46 | 46 | 48 |
| **Conv3d** | **5** | **5** | **5** |
| Max tensor rank | 5 | 5 | 5 |
| Opset | ai.onnx 13 | ai.onnx 13 | ai.onnx 14 |
| IR version | 7 | 7 | 7 |
| Producer | pytorch 2.7.0 | pytorch 2.7.0 | pytorch 2.0.0 |
| Inputs | `left`, `right` — both `[1,3,368,1232]` | same | `input.1`, `input.83` — both `[1,3,368,1232]` |
| Output | `disparity` `[1,1,368,1232]` | same | `524` `[1,1,368,1232]` |

**All dimensions are static.** No dynamic axes, no symbolic dims, batch fixed at
1. Input and output **shapes are identical to the vendor model**; only the
tensor names differ, which no compiler cares about.

## 3 The 3D question, answered

The cost-volume aggregation uses 3D convolution, and the graph carries rank-5
tensors — 35 of them in ARM-P, produced by `Unsqueeze` (24), `Conv` (5),
`LeakyRelu` (4), `Concat` (1) and `Transpose` (1).

A Hailo accelerator is a 4D dataflow architecture, so this is the first thing
that looks like a blocker. **It is not, for this model family.**

The vendor StereoNet has **the same 5 Conv3d nodes, the same max rank of 5, and
the same set of rank-5-producing op types** — and Hailo published a compiled
Hailo-8 HEF for it (`../reference/stereonet.hef`, Model Zoo v2.19.0, profiler
`hw_arch: hailo8`, 242 layers). Whatever the toolchain does with a 3D cost
volume, it did it for this family on this target.

So Conv3d is **not** evidence that ARM-P cannot compile, and it must not be used
to justify redesigning the model. Per spec section 10, a modification needs a
concrete target-native failure, and no target-native evidence exists yet.

## 4 Where the real D2 risk is

ARM-P's graph is 689–712 nodes against the vendor's 168, and it uses **12 op
types the vendor does not**:

```
Cast, Constant, ConstantOfShape, Div, Gather, Pad, Range,
ReduceMean, ReduceProd, Reshape, Shape, Sqrt
```

The vendor uses **zero** op types ARM-P lacks — ARM-P is a strict superset.

Most of that surplus is the real-disparity-shift construction and the
normalisation: 306 `Constant`, 46 `Slice`, 46 `Reshape`, 23 `Pad`, 23
`ConstantOfShape`, 23 `Cast` in the static export. The vendor builds its cost
volume with 11 `Slice` and 12 `Sub` and no `Pad`/`Reshape`/`ConstantOfShape` at
all.

This is the honest statement of risk: **if** ARM-P fails to compile, the shift
construction and its `Pad`/`Reshape`/`ConstantOfShape`/`Cast` machinery is the
place to look first, not the 3D convolutions. That is a hypothesis for D2 to
test, not a finding.

`Shape`, `Gather`, `ReduceProd` and `Range` appear in the dynamic export and
mostly vanish in the static one (712 → 689 nodes, 27 → 23 op types), which is
what a static export is for. **The static export is the one D2 should feed to
the compiler.**

## 5 Preprocessing and postprocessing assumptions

Carried from the frozen Stage C contract, unchanged and unverified against any
Hailo pipeline:

- Input: RGB, 368x1232, normalised with the project's frozen mean/std, NCHW,
  float32. A Hailo pipeline is NHWC uint8 at the boundary with normalisation
  folded in or done on the host — **reconciling these is D2 work, NOT DONE**.
- Output: dense disparity in pixels, `[1,1,368,1232]` float. The vendor's Hailo
  application casts disparity to 8-bit
  (`../stage_c_deploy/C1_TARGET_TOOLCHAIN_RESOLUTION.md`, SR-007); what that
  would cost this model's accuracy is **NOT MEASURED**.
- Everything downstream of disparity — metric depth, point cloud, occupancy,
  discontinuity — is validated host code and target-independent (C2/C2.1/C2.1.1).

## 6 Historical parity status, preserved

C1 remains **FAIL**: max abs difference 0.001708984375 px against a 1e-3 px
criterion. Stage D does not rewrite it, and nothing in D1 revisits it. It is a
PyTorch-vs-ONNX-Runtime parity result on the development host and says nothing
about Hailo.

## 7 What D1 did NOT establish

- That ARM-P compiles for Hailo-8 — **NOT ESTABLISHED**. No compiler was run.
- That any op in the list above is supported or unsupported by a given DFC
  version — **NOT ESTABLISHED**.
- That the vendor's HEF implies anything about ARM-P's *accuracy* or *resource
  usage* after compilation — it does not; it implies only that the structure is
  compilable in principle for this target.
- Anything about quantization, which is D2.

## 8 Next step

D2 (quantization / compile) is **NOT ATTEMPTED** and is blocked on D0: no
Dataflow Compiler host exists. When one does, feed it
`v0/armp_h8_v0_static.onnx`, and record every failure verbatim rather than
patching around it.
