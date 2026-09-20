# PHASE-2 DEPLOYMENT VALIDATION — P2A seed 0

Selection decision: `phase2/docs/PHASE2_DEPLOYMENT_SELECTION.md`.
Experiment result (verdict unchanged, INCONCLUSIVE):
`phase2/docs/PHASE2_P2A_SCALE_COVERAGE_RESULT.md`.
Machine record: `phase2/deploy/P2A_DEPLOYMENT_VALIDATION.json`.
Script: `phase2/scripts/export_p2a.py`. Artifact:
`phase2/deploy/p2a_stereonet.onnx`.

No retraining. No change to the frozen evaluation contract. No evaluation-time
augmentation, no multi-scale inference.

---

## 1. Artifact

| | |
|---|---|
| source checkpoint | `phase2/runs/p2a_scale_coverage/p2a_best.pth` |
| checkpoint SHA-256 | `0868ffd137a9985306bf5563e0d0...` (full value in the JSON) |
| exported ONNX | `phase2/deploy/p2a_stereonet.onnx` |
| ONNX SHA-256 | `0995a6c5722b46250299653e9edaa69ece17c6b86303122345ebf958768bfca9` |
| opset | 13 |
| input | `left`, `right`, both `[1, 3, 368, 1232]`, ImageNet-normalised NCHW |
| output | `disparity`, `[1, 1, 368, 1232]`, full-resolution pixels |
| nodes | 712 |
| parameters | 397,954 (identical to ARM-V) |
| ONNX size | 1,701,550 B (1.62 MiB) |
| checkpoint size | 1,620,893 B (1.55 MiB) |
| `onnx.checker` | pass |
| git head | `58e8a19908dd` |
| software | torch 2.7.0+cu128, onnx / onnxruntime 1.27.0, numpy 2.5.1, python 3.12.9 |

---

## 2. Export parity — exported graph vs trained graph

Five frozen-contract scenes, exported ONNX (onnxruntime CPU) against the
trained PyTorch graph, same inputs:

| scene | max abs difference (px) |
|---|---|
| 000160_10 | 2.213e-04 |
| 000161_10 | 4.158e-04 |
| 000162_10 | 3.433e-04 |
| 000163_10 | 1.354e-04 |
| 000164_10 | 6.180e-04 |

Worst case **6.180e-04 px** against a 1e-3 px tolerance fixed before the run.
**PASS.** The difference is float32 accumulation-order noise, not a semantic
divergence.

---

## 3. Deployed accuracy — the exported artifact scored under the frozen contract

This is the ONNX itself scored end to end, not the PyTorch model. Contract
unchanged: KITTI 2015 `_10`, hailo_val 160–199, `disp_occ_0`, 368×1232
top-left crop, GT/256, valid = gt > 0, pooled. `contract_match true`,
**3,802,797** valid pixels.

| metric | deployed ONNX | PyTorch seed 0 | difference |
|---|---|---|---|
| EPE | **1.4150748** | 1.4149796 | +9.5e-05 |
| D1 | **8.4200 %** | 8.4195 % | +0.0005 pp |
| RMSE | 3.5348 | 3.5346 | +0.0002 |
| bad1 / bad2 / bad3 | 35.1937 / 14.9261 / 8.7528 % | 35.1865 / 14.9269 / 8.7526 % | — |

Strata and per-bin, deployed ONNX:

| GT bin | px | EPE | signed error |
|---|---|---|---|
| [0,16) | 527,871 | 1.340 | +0.44 |
| [16,32) | 1,166,214 | 1.238 | +0.11 |
| [32,48) | 1,118,114 | 1.180 | −0.24 |
| [48,64) | 815,126 | 1.259 | −0.39 |
| [64,80) | 112,726 | 3.060 | −1.59 |
| [80,96) | 28,594 | 4.861 | −3.32 |
| [96,112) | 21,249 | 9.597 | −8.72 |
| [112,128) | 11,823 | 13.537 | −13.29 |
| [128,144) | 1,071 | 33.787 | −33.79 |
| [144,160) | 9 | INSUFFICIENT (<1,000 px) | — |

GT < 64 EPE **1.2397** · GT ≥ 64 EPE **5.0407** (signed −3.72) ·
GT ≥ 96 EPE **11.7274** (signed −11.09) · slope_hi **+0.6226** ·
max prediction **130.57 px** over valid pixels, **157.70 px** over all pixels.

**Reporting rule, carried from the selection record:** 1.4150748 px is the
accuracy of *this artifact*, a single seed. The method's expected accuracy on a
retrain is the 3-seed mean, **1.4409826 px**. Both numbers are stated together
wherever this model's accuracy is quoted. Frozen reference: 1.3134471 px.
Frozen experimental incumbent ARM-V: 1.7727436 px (3-seed mean).

---

## 4. Runtime and resources

Measured on this machine (Windows 11, RTX 4060 Laptop GPU), batch 1 at
368×1232, after warm-up.

| configuration | median | mean | min | n |
|---|---|---|---|---|
| PyTorch CUDA fp32 | **50.62 ms** | 50.7 ms | 49.4 ms | 50 |
| PyTorch CPU fp32 | 767.5 ms | 771.0 ms | 748.9 ms | 20 |
| onnxruntime CPU fp32 | 773.4 ms | 777.7 ms | 756.0 ms | 20 |

Peak CUDA memory allocated during inference: **316.8 MiB**. Parameters 397,954;
ONNX 1.62 MiB on disk.

These are host-development figures. They are **not** target-device figures and
must not be quoted as such.

---

## 5. Deployment compatibility

Operator comparison against the reference deployment graph
(`reference/onnx/stereonet.onnx`, opset 14, 168 nodes):

| | |
|---|---|
| op types shared with the reference | Add, Concat, Conv, LeakyRelu, Mul, Neg, Relu, ReduceSum, Resize, Slice, Softmax, Squeeze, Sub, Transpose, Unsqueeze |
| op types **not** in the reference | Cast, Constant, ConstantOfShape, Div, Gather, Pad, Range, ReduceMean, ReduceProd, Reshape, Shape, Sqrt |
| op types only in the reference | (none that P2A lacks materially — full lists in the JSON) |
| new tensor ranks | none — the 5-D cost volume is the same rank the reference uses |

Every extra operator is attributable, and none of it is gratuitous:

- **437 of the 444 extra nodes come from the real disparity shift** in the cost
  volume — 23 × (Pad + ConstantOfShape + Reshape + Cast + Constants), one
  cluster per non-zero candidate. The reference graph has none of these because
  *its shift is a proven no-op* (EXP-010): it performs no disparity search at
  all. These ops exist precisely because this model does.
- **7 nodes come from the normalised readout** (`regression_normalize=True`):
  ReduceMean ×3, Sqrt, Div ×2, ReduceProd, plus Shape ×3 / Gather ×2 / Range
  from `arange` over the disparity axis inside the soft-argmin.

**Flagged risk — now mitigated, see §5a.** Those last few — `Shape`, `Gather`,
`Range`, `ReduceProd` (7 nodes, all in the regression stage) — are
dynamic-shape operators that a static-graph compiler may reject. They arise
from `torch.arange(cost.shape[1])` and from the standard-deviation reduction
being traced against a symbolic shape.

---

## 5a. Static-shape re-export

Requested and performed after the first export. Record:
`phase2/deploy/P2A_STATIC_EXPORT_VALIDATION.json`. Script:
`phase2/scripts/export_p2a_static.py`. Artifact:
`phase2/deploy/p2a_stereonet_static.onnx`.

**Export-only change. No weight is touched, no training, no change to the
evaluation contract.** The regression stage holds no parameters, so replacing
it cannot alter a weight; the script asserts the `state_dict` key set and every
tensor are bit-identical before exporting.

Three constants baked in, each replacing a shape-derived construction:

| was | now |
|---|---|
| `torch.arange(cost.shape[1])` → `Range` + `Shape` + `Gather` + `Cast` | a constant index buffer of length 24 |
| `cost.std(1, keepdim=True)`, whose unbiased denominator `N−1` comes from the shape → `ReduceProd` + `Shape` + `Gather` | the explicit identity `sqrt( Σ_d (x−mean)² · 1/(D−1) )` with `D = 24` literal |
| `F.interpolate(cost, size=(368,1232))`, whose sizes tensor is built from the input shape → `Shape` → `Slice` | `scale_factor = 8.0` constant |

The scale-factor substitution is exact rather than approximate: the deployment
input is fixed at 368×1232, the feature stride is 8, and 368 = 46·8 and
1232 = 154·8 exactly, so the output geometry is identical. `align_corners=True`
bilinear depends only on the input and output extents, which are unchanged.

### Result

| | original export | static export |
|---|---|---|
| nodes | 712 | **689** |
| size | 1,701,550 B | 1,699,043 B |
| opset | 13 | 13 |
| parameters | 397,954 | **397,954** (asserted identical) |
| `onnx.checker` | pass | **pass** |
| SHA-256 | `0995a6c5722b4625…` | `0665a9ae7d045596…` |

Dynamic operators dropped: **`Range`, `ReduceProd`, `Shape`, `Gather` — all
four gone.** No operator was added.

`ConstantOfShape` (23 nodes) remains, and it is **not** a dynamic-shape risk:
every one of the 23 takes a literal `Constant` as its shape input — verified by
tracing each node's producer — so all are statically foldable by any parser.
They come from the 23 `F.pad` calls of the real disparity shift in the cost
volume, not from the regression stage, and were not part of the flagged set.
**No genuinely dynamic-shape operator remains in the static export.**

### Parity and accuracy of the static export

| comparison | max abs difference (5 scenes) | tolerance | result |
|---|---|---|---|
| static PyTorch graph vs trained PyTorch graph | 3.052e-05 px | 1e-3 | **PASS** |
| static ONNX vs original ONNX | 3.052e-05 px | 1e-3 | **PASS** |
| static ONNX vs trained PyTorch graph | 6.218e-04 px | 1e-3 | **PASS** |

Frozen contract, static ONNX scored end to end, `contract_match true`,
3,802,797 valid pixels:

| metric | static ONNX | original ONNX | PyTorch seed 0 |
|---|---|---|---|
| EPE | **1.4150748** | 1.4150748 | 1.4149796 |
| D1 | **8.4200 %** | 8.4200 % | 8.4195 % |
| RMSE | 3.5348 | 3.5348 | 3.5346 |
| GT < 64 EPE | 1.2397 | 1.2397 | 1.2396 |
| GT ≥ 64 EPE | 5.0407 | 5.0407 | 5.0404 |
| GT ≥ 96 EPE | 11.7274 | 11.7274 | 11.7259 |
| slope_hi | +0.6226 | +0.6226 | +0.6227 |
| max prediction (valid px) | 130.57 | 130.57 | 130.60 |

The static export is accuracy-identical to the original at the reported
precision.

### Which artifact to compile

Both are kept, and the choice is deferred to the target toolchain rather than
guessed at here:

- `p2a_stereonet.onnx` — the direct export. Reference artifact; it is what §2–§4
  above were measured on.
- `p2a_stereonet_static.onnx` — use if the Hailo parser rejects dynamic-shape
  operators. Accuracy-identical, 23 fewer nodes, no dynamic-shape ops.

Neither has been through a compiler. That remains unverified.

### Hailo compilation: NOT VERIFIED

The Hailo SDK / dataflow compiler is **not installed** in this environment. No
`.hef` was produced and no on-device measurement was taken. The operator
comparison above is **structural evidence only** — it is not a compilation
result and must not be reported as one.

To complete this on hardware, the remaining steps are: run the Hailo parser
against `phase2/deploy/p2a_stereonet.onnx` using the existing
`reference/hailo_model_zoo/stereonet.yaml` / `stereonet.alls` as the starting
configuration (noting that the `.alls` was written for the 143-node reference
graph and will need its layer names updated), quantise with the hailo_calib
scenes 0–159, compile, then re-score on device under the same frozen contract
and record the on-device latency and resource figures.

---

## 6. Phase-2 end-condition status

| # | condition | status |
|---|---|---|
| 1 | final accuracy model frozen | **DONE** — P2A seed 0, SHA-256 recorded |
| 2 | evaluation reproducible | **DONE** — frozen contract, `contract_match true`, 3,802,797 px; scripts and JSON records committed |
| 3 | deployment/export validated | **PARTIAL** — both ONNX exports validated (checker pass, ≤6.2e-04 px parity, each scored end to end); static variant carries no dynamic-shape ops. Hailo compilation **NOT VERIFIED** (no SDK present) |
| 4 | deployed accuracy measured | **DONE** — 1.4150748 px EPE / 8.4200 % D1, measured through the exported ONNX |
| 5 | runtime/resource measurements recorded | **DONE on host** — 50.6 ms CUDA, 773 ms ORT CPU, 316.8 MiB peak. **On-device figures NOT MEASURED** |
| 6 | final artifacts complete | **DONE for this environment** — see §7 |

Conditions 3 and 5 cannot be completed without the Hailo toolchain and target
hardware. That is a tooling limit, not an open research question, and it does
not reopen any experiment.

---

## 7. Artifact index

```
phase2/deploy/p2a_stereonet.onnx                deployed graph (direct export)
phase2/deploy/p2a_stereonet_static.onnx         static-shape variant, accuracy-identical
phase2/deploy/P2A_DEPLOYMENT_VALIDATION.json    export, parity, accuracy, runtime, compatibility
phase2/deploy/P2A_STATIC_EXPORT_VALIDATION.json static re-export: op removal, parity, accuracy
phase2/runs/p2a_scale_coverage{,_s1,_s2}/       three trained seeds, records, logs, guards
phase2/runs/p2a_scale_coverage/P2A_EXPERIMENT_RECORD.json
phase2/runs/p2a_scale_coverage/P2A_DECISIONS.json
phase2/runs/p2a_scale_coverage/p2a_eval_best.json
phase2/scripts/train_p2a_scale_coverage.py      trainer (the one intervention)
phase2/scripts/eval_p2a.py                      frozen-contract scorer
phase2/scripts/p2a_decide.py                    preregistered gates
phase2/scripts/p2a_record.py                    record assembly
phase2/scripts/export_p2a.py                    export + validation
phase2/scripts/export_p2a_static.py             static-shape re-export + validation
phase2/scripts/run_p2a_remaining.ps1            detached execution launcher
phase2/diagnostics/arm_v_high_disparity/        the Phase-2 diagnostic that produced the hypothesis
phase2/docs/PHASE2_ARMV_HIGH_DISPARITY_DIAGNOSTIC.md
phase2/docs/PHASE2_HYPOTHESIS_01.md
phase2/docs/PHASE2_P2A_SCALE_COVERAGE_RESULT.md
phase2/docs/PHASE2_DEPLOYMENT_SELECTION.md
phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md     this file
```

Untouched throughout: `phase1/results/LEADERBOARD.md`, all ARM-V/W/X/Y/Z
artifacts and verdicts, all Phase-1 results, and `reference/`.

---

## 8. Closing state

- Deployed model: **P2A seed 0**, EPE **1.4150748** (3-seed mean 1.4409826).
- `EXP-P2A-SCALE-COVERAGE-001` verdict: **INCONCLUSIVE**, mechanism **NOT
  CONFIRMED** — unchanged, and not revised by the deployment decision.
- Formal experimental incumbent: **ARM-V**, unchanged.
- Phase 0 CLOSED · Phase 1 CLOSED · Phase 2 optimization CLOSED.
- No Phase 3.
