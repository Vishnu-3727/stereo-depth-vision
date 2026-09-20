# HISTORICAL — Phase-1 precision study (NOT Hailo, NOT ARM-P)

Added 2026-09-20 by the documentation closure pass. **Nothing in this directory was modified,
regenerated or deleted.** This file exists only to stop the artifacts here being mistaken for
deployment evidence.

## What this directory is

`results/quantization/` holds **Phase-1 EXP-015**, a host-side precision comparison of the
**Hailo reference model** run on 2026-09-05:

| configuration | EPE px | D1 % | latency ms |
|---|---|---|---|
| fp32 onnxruntime CPU | 1.313 | 8.154 | 549.1 |
| fp32 torch CUDA | 1.314 | 8.155 | 90.1 |
| fp16 torch CUDA | 1.310 | 8.141 | 64.8 |
| **int8 onnxruntime CPU** | **1.655** | 10.945 | 677.4 |

Source of truth: `report.txt`, `precision_comparison.json`,
`artifacts/stereonet_int8.onnx`.

## What it is NOT

- **NOT ARM-P.** The fp32 row (1.313 px) identifies the model as the reference ONNX
  (`reference/onnx/stereonet.onnx`, 423,586 params). ARM-P did not exist on 2026-09-05.
- **NOT Hailo quantization.** This is `onnxruntime` CPU int8, not the Hailo Dataflow Compiler,
  not `hailomz optimize`, not a Hailo calibration run.
- **NOT a Hailo deployment result.** No HEF was produced, no device was used, no Hailo
  toolchain has ever been executed in this project.
- **NOT evidence about the ARM-P activation range.** ARM-P's ~1e18 fp32 intermediates and their
  behaviour under target-native quantization remain **UNKNOWN** — see
  `stage_c_deploy/DYNAMIC_RANGE_AUDIT.md` and the corrected classification in
  `stage_c_deploy/STAGE_C_DEPLOYMENT_READINESS_FINAL.md` §6.

## Current status of quantization in this project

**ARM-P quantization: NOT TESTED.** It is BLOCKED on the unspecified target device and the
un-executed Hailo toolchain. Do not cite the int8 row above in its place.

See `RESULTS_INDEX.md` §10 (reader hazards) and
`stage_c_deploy/STAGE_C_DEPLOYMENT_READINESS_FINAL.md` §13.
