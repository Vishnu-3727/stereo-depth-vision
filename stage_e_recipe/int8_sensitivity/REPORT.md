# INT8 per-layer sensitivity � E3 seed-0 (measured)

Diagnostic only. No accept/reject. No training. NOT Hailo evidence; Stage D remains blocked.

- Branch: `contract40` (est 2.53 h vs 6 h budget).
- Single-score timing: 38.2 s; single-quantize: 49.4 s; configs: 104.
- Baseline fp32 EPE (measured): 1.1859853 (recorded ref 1.1859853).
- Baseline int8 EPE (measured): 5.2489414 (recorded ref 5.2489414).
- Gap G (measured): 4.0629561 px. Baseline gate: PASS: reproduction within 1e-6 px, guards true.
- Guards: fp32 contract_match=True, int8 contract_match=True.

## CORRECTION — what Arms A/B actually measured (2026-09-23, post-review)

The original spec §3 rationale ("Conv nodes are the quantized compute in
this graph") is MEASURED FALSE. Do not quote it. Read-only audit
(`stage_e_recipe/int8_qdq_coverage.py` → `qdq_coverage.json`):

- `baseline_int8.onnx`: 200 QuantizeLinear nodes (286 DequantizeLinear).
  DequantizeLinear outputs feed 16 op types (input-slot edges): Conv 153,
  Sub 51, Add 40, LeakyRelu 40, Concat 26, Pad 23, Slice 23, Mul 6,
  ReduceMean 3, Div 2, Shape 2, Softmax 1, ReduceSum 1, Resize 1, Squeeze 1,
  Transpose 1. (Mul is 6 measured edges across 3 distinct Mul nodes; the
  review brief quoted 5.)
- `only_00_*.onnx` (an Arm B config): still 191 QuantizeLinear nodes, with
  non-Conv activation QDQ fully retained (Add/LeakyRelu/Concat/Pad/Slice
  edges identical to baseline).

So Arm A measures "restore one Conv to fp32" (valid as such), but Arm B
does NOT isolate single-layer int8 — every Arm B row is "one Conv int8 on
top of full non-Conv int8". Valid reading of Arms A/B (inference): no
single Conv moves EPE by more than 0.06 px of the 4.06 px gap, so the damage
is not attributable to any single Conv; it must sit in non-Conv QDQ and/or
be distributed. Arm C below tests the non-Conv half directly.

## Arm A � leave-one-out (all int8 except L; sorted by EPE ascending)

| rank | layer | EPE | d_fp32 | d_int8 | R | D1 | nonfinite |
|---|---|---|---|---|---|---|---|
| 1 | `/refinement/blocks/blocks.3/conv1/Conv` | 5.2150 | +4.0290 | -0.0340 | 0.008 | 67.44 | 0 |
| 2 | `/refinement/output_conv/Conv` | 5.2280 | +4.0420 | -0.0209 | 0.005 | 67.58 | 0 |
| 3 | `/refinement/blocks/blocks.5/conv1/Conv` | 5.2295 | +4.0435 | -0.0195 | 0.005 | 67.60 | 0 |
| 4 | `/aggregation/filter/filter.0/Conv` | 5.2362 | +4.0503 | -0.0127 | 0.003 | 67.66 | 0 |
| 5 | `/feature_extractor/output_conv/Conv` | 5.2387 | +4.0527 | -0.0103 | 0.003 | 67.70 | 0 |
| 6 | `/refinement/blocks/blocks.5/conv2/Conv` | 5.2395 | +4.0535 | -0.0094 | 0.002 | 67.69 | 0 |
| 7 | `/feature_extractor/downsample/downsample.1/Conv` | 5.2424 | +4.0564 | -0.0065 | 0.002 | 67.73 | 0 |
| 8 | `/refinement/blocks/blocks.3/conv2/Conv` | 5.2434 | +4.0574 | -0.0056 | 0.001 | 67.73 | 0 |
| 9 | `/aggregation/filter/filter.2/Conv` | 5.2435 | +4.0575 | -0.0054 | 0.001 | 67.69 | 0 |
| 10 | `/aggregation/filter/filter.4/Conv` | 5.2438 | +4.0578 | -0.0052 | 0.001 | 67.78 | 0 |
| 11 | `/refinement/blocks/blocks.2/conv2/Conv` | 5.2442 | +4.0582 | -0.0047 | 0.001 | 67.76 | 0 |
| 12 | `/feature_extractor/downsample/downsample.0_1/Conv` | 5.2447 | +4.0588 | -0.0042 | 0.001 | 67.67 | 0 |
| 13 | `/feature_extractor/downsample/downsample.2/Conv` | 5.2464 | +4.0604 | -0.0025 | 0.001 | 67.78 | 0 |
| 14 | `/feature_extractor/downsample/downsample.2_1/Conv` | 5.2477 | +4.0617 | -0.0013 | 0.000 | 67.74 | 0 |
| 15 | `/refinement/blocks/blocks.2/conv1/Conv` | 5.2482 | +4.0622 | -0.0008 | 0.000 | 67.72 | 0 |
| 16 | `/feature_extractor/residual/residual.4/conv2/Conv` | 5.2482 | +4.0622 | -0.0007 | 0.000 | 67.79 | 0 |
| 17 | `/feature_extractor/residual/residual.1/conv2_1/Conv` | 5.2485 | +4.0625 | -0.0004 | 0.000 | 67.78 | 0 |
| 18 | `/feature_extractor/residual/residual.2/conv2_1/Conv` | 5.2485 | +4.0626 | -0.0004 | 0.000 | 67.79 | 0 |
| 19 | `/refinement/blocks/blocks.1/conv2/Conv` | 5.2486 | +4.0626 | -0.0003 | 0.000 | 67.76 | 0 |
| 20 | `/feature_extractor/residual/residual.5/conv2/Conv` | 5.2487 | +4.0627 | -0.0003 | 0.000 | 67.79 | 0 |
| 21 | `/feature_extractor/residual/residual.3/conv2_1/Conv` | 5.2487 | +4.0627 | -0.0002 | 0.000 | 67.79 | 0 |
| 22 | `/feature_extractor/residual/residual.0/conv1_1/Conv` | 5.2489 | +4.0629 | -0.0000 | 0.000 | 67.80 | 0 |
| 23 | `/feature_extractor/downsample/downsample.0/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 24 | `/feature_extractor/output_conv_1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 25 | `/feature_extractor/residual/residual.0/conv2/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 26 | `/feature_extractor/residual/residual.0/conv2_1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 27 | `/feature_extractor/residual/residual.1/conv1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 28 | `/feature_extractor/residual/residual.1/conv1_1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 29 | `/feature_extractor/residual/residual.1/conv2/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 30 | `/feature_extractor/residual/residual.2/conv1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 31 | `/feature_extractor/residual/residual.2/conv1_1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 32 | `/feature_extractor/residual/residual.2/conv2/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 33 | `/feature_extractor/residual/residual.3/conv1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 34 | `/feature_extractor/residual/residual.3/conv1_1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 35 | `/feature_extractor/residual/residual.3/conv2/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 36 | `/feature_extractor/residual/residual.4/conv1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 37 | `/feature_extractor/residual/residual.4/conv1_1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 38 | `/feature_extractor/residual/residual.4/conv2_1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 39 | `/feature_extractor/residual/residual.5/conv1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 40 | `/feature_extractor/residual/residual.5/conv1_1/Conv` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 |
| 41 | `/feature_extractor/residual/residual.0/conv1/Conv` | 5.2490 | +4.0630 | +0.0000 | -0.000 | 67.79 | 0 |
| 42 | `/feature_extractor/residual/residual.5/conv2_1/Conv` | 5.2491 | +4.0631 | +0.0002 | -0.000 | 67.79 | 0 |
| 43 | `/refinement/blocks/blocks.0/conv1/Conv` | 5.2499 | +4.0639 | +0.0010 | -0.000 | 67.82 | 0 |
| 44 | `/aggregation/filter/filter.6/Conv` | 5.2509 | +4.0649 | +0.0019 | -0.000 | 67.85 | 0 |
| 45 | `/refinement/blocks/blocks.0/conv2/Conv` | 5.2510 | +4.0650 | +0.0020 | -0.001 | 67.84 | 0 |
| 46 | `/refinement/input_conv/Conv` | 5.2513 | +4.0653 | +0.0024 | -0.001 | 67.82 | 0 |
| 47 | `/refinement/blocks/blocks.4/conv1/Conv` | 5.2525 | +4.0665 | +0.0035 | -0.001 | 67.82 | 0 |
| 48 | `/refinement/blocks/blocks.1/conv1/Conv` | 5.2536 | +4.0676 | +0.0047 | -0.001 | 67.78 | 0 |
| 49 | `/feature_extractor/downsample/downsample.1_1/Conv` | 5.2557 | +4.0697 | +0.0068 | -0.002 | 67.83 | 0 |
| 50 | `/aggregation/to_cost/Conv` | 5.2750 | +4.0890 | +0.0260 | -0.006 | 67.76 | 0 |
| 51 | `/refinement/blocks/blocks.4/conv2/Conv` | 5.2789 | +4.0929 | +0.0299 | -0.007 | 68.02 | 0 |

## Arm B � only-one (only L int8; sorted by EPE descending)

| rank | layer | EPE | d_fp32 | d_int8 | R | D1 | nonfinite |
|---|---|---|---|---|---|---|---|
| 1 | `/refinement/blocks/blocks.3/conv1/Conv` | 5.2742 | +4.0882 | +0.0253 | -0.006 | 67.60 | 0 |
| 2 | `/refinement/output_conv/Conv` | 5.2644 | +4.0784 | +0.0154 | -0.004 | 67.50 | 0 |
| 3 | `/refinement/blocks/blocks.5/conv1/Conv` | 5.2627 | +4.0767 | +0.0138 | -0.003 | 67.48 | 0 |
| 4 | `/refinement/blocks/blocks.5/conv2/Conv` | 5.2522 | +4.0662 | +0.0033 | -0.001 | 67.38 | 0 |
| 5 | `/refinement/blocks/blocks.3/conv2/Conv` | 5.2480 | +4.0621 | -0.0009 | 0.000 | 67.35 | 0 |
| 6 | `/refinement/blocks/blocks.2/conv2/Conv` | 5.2463 | +4.0603 | -0.0027 | 0.001 | 67.31 | 0 |
| 7 | `/aggregation/filter/filter.0/Conv` | 5.2443 | +4.0583 | -0.0046 | 0.001 | 67.32 | 0 |
| 8 | `/refinement/blocks/blocks.1/conv2/Conv` | 5.2438 | +4.0578 | -0.0052 | 0.001 | 67.32 | 0 |
| 9 | `/aggregation/filter/filter.2/Conv` | 5.2435 | +4.0575 | -0.0055 | 0.001 | 67.35 | 0 |
| 10 | `/refinement/blocks/blocks.0/conv1/Conv` | 5.2434 | +4.0574 | -0.0055 | 0.001 | 67.32 | 0 |
| 11 | `/refinement/blocks/blocks.2/conv1/Conv` | 5.2433 | +4.0573 | -0.0056 | 0.001 | 67.34 | 0 |
| 12 | `/refinement/input_conv/Conv` | 5.2432 | +4.0572 | -0.0057 | 0.001 | 67.28 | 0 |
| 13 | `/aggregation/filter/filter.4/Conv` | 5.2432 | +4.0572 | -0.0058 | 0.001 | 67.33 | 0 |
| 14 | `/refinement/blocks/blocks.0/conv2/Conv` | 5.2424 | +4.0564 | -0.0066 | 0.002 | 67.27 | 0 |
| 15 | `/refinement/blocks/blocks.4/conv1/Conv` | 5.2396 | +4.0536 | -0.0094 | 0.002 | 67.24 | 0 |
| 16 | `/aggregation/filter/filter.6/Conv` | 5.2388 | +4.0528 | -0.0102 | 0.003 | 67.26 | 0 |
| 17 | `/refinement/blocks/blocks.1/conv1/Conv` | 5.2371 | +4.0511 | -0.0118 | 0.003 | 67.29 | 0 |
| 18 | `/feature_extractor/residual/residual.3/conv2_1/Conv` | 5.2361 | +4.0501 | -0.0128 | 0.003 | 67.15 | 0 |
| 19 | `/feature_extractor/residual/residual.3/conv2/Conv` | 5.2360 | +4.0500 | -0.0129 | 0.003 | 67.15 | 0 |
| 20 | `/feature_extractor/output_conv_1/Conv` | 5.2355 | +4.0496 | -0.0134 | 0.003 | 67.21 | 0 |
| 21 | `/feature_extractor/output_conv/Conv` | 5.2352 | +4.0492 | -0.0137 | 0.003 | 67.13 | 0 |
| 22 | `/aggregation/to_cost/Conv` | 5.2339 | +4.0479 | -0.0150 | 0.004 | 67.29 | 0 |
| 23 | `/feature_extractor/residual/residual.2/conv1/Conv` | 5.2309 | +4.0449 | -0.0180 | 0.004 | 67.15 | 0 |
| 24 | `/feature_extractor/residual/residual.2/conv1_1/Conv` | 5.2309 | +4.0449 | -0.0180 | 0.004 | 67.15 | 0 |
| 25 | `/feature_extractor/downsample/downsample.0_1/Conv` | 5.2304 | +4.0444 | -0.0185 | 0.005 | 67.22 | 0 |
| 26 | `/feature_extractor/residual/residual.4/conv2_1/Conv` | 5.2293 | +4.0433 | -0.0197 | 0.005 | 67.21 | 0 |
| 27 | `/feature_extractor/residual/residual.4/conv2/Conv` | 5.2290 | +4.0430 | -0.0199 | 0.005 | 67.20 | 0 |
| 28 | `/feature_extractor/residual/residual.4/conv1/Conv` | 5.2279 | +4.0419 | -0.0210 | 0.005 | 67.23 | 0 |
| 29 | `/feature_extractor/residual/residual.4/conv1_1/Conv` | 5.2279 | +4.0419 | -0.0210 | 0.005 | 67.23 | 0 |
| 30 | `/feature_extractor/residual/residual.5/conv2/Conv` | 5.2271 | +4.0411 | -0.0218 | 0.005 | 67.13 | 0 |
| 31 | `/feature_extractor/residual/residual.5/conv2_1/Conv` | 5.2271 | +4.0411 | -0.0218 | 0.005 | 67.12 | 0 |
| 32 | `/feature_extractor/residual/residual.3/conv1/Conv` | 5.2255 | +4.0395 | -0.0235 | 0.006 | 67.13 | 0 |
| 33 | `/feature_extractor/residual/residual.3/conv1_1/Conv` | 5.2255 | +4.0395 | -0.0235 | 0.006 | 67.13 | 0 |
| 34 | `/feature_extractor/residual/residual.2/conv2_1/Conv` | 5.2244 | +4.0384 | -0.0246 | 0.006 | 67.19 | 0 |
| 35 | `/feature_extractor/residual/residual.2/conv2/Conv` | 5.2244 | +4.0384 | -0.0246 | 0.006 | 67.19 | 0 |
| 36 | `/feature_extractor/residual/residual.1/conv1/Conv` | 5.2231 | +4.0371 | -0.0259 | 0.006 | 67.16 | 0 |
| 37 | `/feature_extractor/residual/residual.1/conv1_1/Conv` | 5.2231 | +4.0371 | -0.0259 | 0.006 | 67.16 | 0 |
| 38 | `/feature_extractor/residual/residual.5/conv1/Conv` | 5.2209 | +4.0350 | -0.0280 | 0.007 | 67.08 | 0 |
| 39 | `/feature_extractor/residual/residual.5/conv1_1/Conv` | 5.2209 | +4.0350 | -0.0280 | 0.007 | 67.08 | 0 |
| 40 | `/feature_extractor/downsample/downsample.0/Conv` | 5.2205 | +4.0345 | -0.0285 | 0.007 | 67.11 | 0 |
| 41 | `/feature_extractor/residual/residual.1/conv2_1/Conv` | 5.2204 | +4.0344 | -0.0285 | 0.007 | 67.07 | 0 |
| 42 | `/feature_extractor/residual/residual.1/conv2/Conv` | 5.2201 | +4.0341 | -0.0289 | 0.007 | 67.07 | 0 |
| 43 | `/feature_extractor/residual/residual.0/conv2/Conv` | 5.2199 | +4.0339 | -0.0291 | 0.007 | 67.08 | 0 |
| 44 | `/feature_extractor/residual/residual.0/conv2_1/Conv` | 5.2199 | +4.0339 | -0.0291 | 0.007 | 67.08 | 0 |
| 45 | `/feature_extractor/downsample/downsample.2_1/Conv` | 5.2185 | +4.0325 | -0.0305 | 0.008 | 67.22 | 0 |
| 46 | `/feature_extractor/downsample/downsample.2/Conv` | 5.2175 | +4.0315 | -0.0314 | 0.008 | 67.24 | 0 |
| 47 | `/feature_extractor/residual/residual.0/conv1_1/Conv` | 5.2162 | +4.0302 | -0.0327 | 0.008 | 66.99 | 0 |
| 48 | `/feature_extractor/residual/residual.0/conv1/Conv` | 5.2161 | +4.0301 | -0.0328 | 0.008 | 66.99 | 0 |
| 49 | `/refinement/blocks/blocks.4/conv2/Conv` | 5.2123 | +4.0264 | -0.0366 | 0.009 | 67.06 | 0 |
| 50 | `/feature_extractor/downsample/downsample.1/Conv` | 5.2108 | +4.0248 | -0.0382 | 0.009 | 66.96 | 0 |
| 51 | `/feature_extractor/downsample/downsample.1_1/Conv` | 5.1970 | +4.0111 | -0.0519 | 0.013 | 67.03 | 0 |

## Pre-registered readout

Dominant layers (R >= 0.50): none measured.

## Arm C — op-type follow-up (pre-registered in spec §11, amendment e437f51)

- Script: `stage_e_recipe/int8_armc.py` → `armc.json` (+ per-config json).
  Same §2 quantizer / calibration, 40-scene contract, `contract_match`
  REQUIRED on every row (no monitor branch).
- Baselines re-scored: fp32 EPE 1.1859853, int8 EPE 5.2489414, gate PASS
  (within 1e-6 px, guards true). Gap G: 4.0629561 px.
- Budget: first config 85.7 s × 17 → est 0.40 h < 2 h → full run proceeded.
  Each config's `.onnx` deleted immediately after scoring (sha256 recorded).
- All 17 rows: contract_match=True, nonfinite=0.

| rank | config | EPE | d_fp32 | d_int8 | R | D1 | nonfinite | excluded |
|---|---|---|---|---|---|---|---|---|
| 1 | `C0_conv_only_int8` (only Conv int8) | 1.4963 | +0.3104 | -3.7526 | 0.924 | 7.66 | 0 | 661 |
| 2 | `C_Mul` (Mul fp32) | 2.9241 | +1.7381 | -2.3249 | 0.572 | 38.91 | 0 | 3 |
| 3 | `C_Concat` (Concat fp32) | 5.1944 | +4.0084 | -0.0545 | 0.013 | 67.52 | 0 | 26 |
| 4 | `C1_nonconv_only_int8` (all Conv fp32) | 5.2435 | +4.0575 | -0.0054 | 0.001 | 67.29 | 0 | 51 |
| 5 | `C_Div` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 | 2 |
| 6 | `C_LeakyRelu` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 | 40 |
| 7 | `C_Pad` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 | 23 |
| 8 | `C_ReduceMean` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 | 3 |
| 9 | `C_ReduceSum` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 | 1 |
| 10 | `C_Shape` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 | 3 |
| 11 | `C_Slice` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 | 47 |
| 12 | `C_Sub` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 | 27 |
| 13 | `C_Transpose` | 5.2489 | +4.0630 | +0.0000 | 0.000 | 67.79 | 0 | 24 |
| 14 | `C_Resize` | 5.2568 | +4.0708 | +0.0078 | -0.002 | 67.68 | 0 | 1 |
| 15 | `C_Squeeze` | 5.2568 | +4.0708 | +0.0078 | -0.002 | 67.68 | 0 | 1 |
| 16 | `C_Add` | 5.2932 | +4.1073 | +0.0443 | -0.011 | 68.59 | 0 | 20 |
| 17 | `C_Softmax` (Softmax fp32) | 6.6617 | +5.4757 | +1.4127 | -0.348 | 34.79 | 0 | 1 |

## Pre-registered readout — Arm C

Dominant configs (R >= 0.50): `C0_conv_only_int8` (R=0.924),
`C_Mul` (R=0.572).

- C1 (all Conv fp32, non-Conv int8) EPE 5.2435 ≈ int8 baseline 5.2489
  (R=0.001): non-Conv QDQ alone reproduces the full gap — CONFIRMED.
- C0 (only Conv int8, non-Conv fp32) EPE 1.4963, R=0.924: keeping all
  non-Conv in fp32 recovers 92% of the gap — CONFIRMED from the other side
  (0.31 px residual above fp32 remains; measured, mechanism unknown).
- C_Mul is the only dominant op type (R=0.572, 3 Mul nodes kept fp32).
- Measured anomaly (no interpretation claimed): keeping the single Softmax
  node in fp32 WORSENS EPE to 6.6617 (R=-0.348, D1 34.79 vs 67.79),
  contract_match=True, nonfinite=0. Mechanism unknown.
- No fix claim, no Hailo implication. Full rows: `armc.json`.

## Provenance

- UTC: 2026-09-23T04:09:28.386159+00:00; git HEAD: d2500322d364827bec67cd1bdd3d82a471312097.
- Env: {'python': '3.12.9', 'torch': '2.7.0+cu128', 'numpy': '2.5.1', 'onnxruntime': '1.27.0'}.
- Subject sha256 asserted: `7453b2be2be420d45e6a9d18104a23a8e25f02ec29312646886e380ff32b9645`; Conv groups: 51.
- Full rows: `int8_sensitivity.json` (+ per-layer json).
- Arm C provenance: UTC 2026-09-23T07:33:03Z; git HEAD e437f51a6aff79ed5ae5b1612d839deb6a6af6e8;
  env (measured, same machine): python 3.12.9, torch 2.7.0+cu128, numpy 2.5.1,
  onnxruntime 1.27.0. No `.onnx` remains in this directory (all Arm C
  artifacts deleted after scoring; Arms A/B artifacts deleted per Brief 5).
