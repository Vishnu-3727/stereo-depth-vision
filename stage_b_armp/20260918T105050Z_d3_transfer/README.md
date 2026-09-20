# ARM-P D3 transfer diagnostic (frozen inference ONLY)

DIAGNOSTIC ONLY. No training, no fine-tuning, no Stage B, P2A untouched.
`stage_a_diagnostics/scripts/d3_matching.py` was NOT modified (imported as a
module; only `d3.RUNS` monkey-patched in-memory). No existing repo file was
modified; everything new lives in this directory.

## Question

Does the ARM-P pretrained feature representation show improved correspondence
evidence on KITTI at GT >= 96 px, versus P2A?

## Method

- Checkpoint copy: `ckpt/p2a_best.pth` is a copy of
  `stage_b_armp/20260918T062146Z_stage1_pretrain/checkpoints/armp_stage1_best.pth`.
  sha256 of source AND copy = `3ae6fb3b...f1a29be7` (identical, matches expected).
- Wrapper `scripts/run_armp_d3.py` imports `d3_matching`, patches ONLY `d3.RUNS`
  to `{0: "../../stage_b_armp/20260918T105050Z_d3_transfer/ckpt"}` (verified to
  resolve to the copy), asserts config
  (downsample_levels=3, num_disparities=24, cost_volume_shift=right,
  regression_normalize=True, params=397954, strict load OK), then calls
  `d3.process_seed(0, device, None)` + `d3.summarize_seed(0, acc)` UNCHANGED.
- P2A numbers are READ from `stage_a_diagnostics/matching_diagnostic.json` and
  `stage_a_diagnostics/raw/d3_bins.csv` (3-seed method mean). P2A was NOT re-run.
- LIMITATION (real asymmetry): P2A = 3-seed method mean; ARM-P = SINGLE checkpoint.
- ARM-P checkpoint context: 7/20-epoch FT3D A+C subset pretrain, unannealed LR
  (7.27e-04). A negative result is NOT a falsification.
- Representation D was group-AVERAGED (G-invariant): known Stage-A limitation,
  PRESERVED, reported, not fixed.
- Runtime: 24.8 s wall-clock on CUDA (torch 2.7.0+cu128). GPU used.

## Global (ARM-P, seed-0 slot)

- Feature tensor shape: (1, 32, 46, 154) [arch check passed]; candidates: 24.
- Feature cells total: 283360; surviving GT cells sampled: 59431;
  dropped (left-edge GT-invalid): 1313; (cell,k) pairs excluded by left-edge rule: 79326.
  (Counts identical to P2A runs: same split, same geometry.)
- Pooled (n=58048 defined cells):

| rep | rank-1 | margin-frac-pos | P2A mm rank-1 | P2A mm mfrac | delta rank-1 | delta mfrac |
| --- | --- | --- | --- | --- | --- | --- |
| A_L1 | 0.4805 | 0.5666 | 0.6469 | 0.7320 | -0.1664 | -0.1654 |
| B_L2 | 0.4877 | 0.5761 | 0.6528 | 0.7386 | -0.1651 | -0.1625 |
| C_L1norm | 0.4805 | 0.5616 | 0.6286 | 0.7035 | -0.1481 | -0.1419 |
| D_G8 | 0.1841 | 0.2444 | 0.2501 | 0.3202 | -0.0660 | -0.0758 |

(P2A pooled method-mean values read from matching_diagnostic.json method_mean.)

## GT>=96 (headline)

| rep | n | rank-1 | mfrac-pos | mean margin | overlap | gap | P2A mm rank-1 | P2A mm mfrac | delta rank-1 | delta mfrac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_L1 | 393 | 0.2621 | 0.3537 | -4.97e10 | 0.2462 | 0.9335 | 0.3647 | 0.4436 | -0.1026 | -0.0899 |
| B_L2 | 393 | 0.2595 | 0.3537 | -5.64e10 | 0.2359 | 0.9569 | 0.3681 | 0.4580 | -0.1086 | -0.1043 |
| C_L1norm | 393 | 0.2621 | 0.3333 | -0.0042 | 0.1672 | 1.1503 | 0.3783 | 0.4427 | -0.1162 | -0.1094 |
| D_G8 | 393 | 0.1399 | 0.1908 | -2.27e23 | 0.2494 | 0.8803 | 0.1730 | 0.2188 | -0.0331 | -0.0280 |

P2A per-seed A_L1 GT>=96 rank-1: [0.3791, 0.3562, 0.3588]; mfrac-pos:
[0.4402, 0.4427, 0.4478]. ARM-P (0.2621 / 0.3537) sits below every P2A seed.
Scale note: ARM-P raw features are unnormalised with huge magnitudes
(A_L1 GT>=96 pos_mean 2.56e11, neg_mean 4.14e11 vs P2A ~7.7e5/1.3e7);
margin MEAN is therefore scale-dominated and not comparable across checkpoints;
rank and margin-fraction-positive are scale-free and are the comparable metrics.

## Per-bin A_L1 (headline representation)

| bin | n | rank-1 (P2A mm) | delta | mfrac (P2A mm) | delta | overlap (P2A mm) | low-n (<1000) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [64,80) | 1463 | 0.4614 (0.5616) | -0.1003 | 0.5229 (0.6343) | -0.1114 | 0.1246 (0.0627) | no |
| [80,96) | 377 | 0.3634 (0.4279) | -0.0645 | 0.4589 (0.5234) | -0.0645 | 0.1723 (0.0953) | YES |
| [96,112) | 286 | 0.2413 (0.3811) | -0.1399 | 0.3042 (0.4336) | -0.1294 | 0.2359 (0.1727) | YES |
| [112,128) | 107 | 0.3178 (0.3209) | -0.0031 | 0.4860 (0.4704) | +0.0156 | 0.1845 (0.0830) | YES |

Per-bin (P2A mm in brackets), all four reps:

- B_L2: [64,80) 0.4627 (0.5701) / 0.5270 (0.6432) / ov 0.1210 (0.0613);
  [80,96) 0.3740 (0.4279) / 0.4721 (0.5217) / ov 0.1690 (0.0916);
  [96,112) 0.2448 (0.3800) / 0.3217 (0.4417) / ov 0.2354 (0.1740);
  [112,128) 0.2991 (0.3364) / 0.4393 (0.5016) / ov 0.1751 (0.0854).
- C_L1norm: [64,80) 0.4580 (0.5443) / 0.5243 (0.6149) / ov 0.1244 (0.0486);
  [80,96) 0.3820 (0.4103) / 0.4615 (0.4987) / ov 0.0942 (0.0539);
  [96,112) 0.2343 (0.3741) / 0.2937 (0.4207) / ov 0.1927 (0.1130);
  [112,128) 0.3364 (0.3894) / 0.4393 (0.5016) / ov 0.0860 (0.0559).
- D_G8: [64,80) 0.1757 (0.2285) / 0.2249 (0.3049) / ov 0.2706 (0.2053);
  [80,96) 0.1459 (0.1945) / 0.1989 (0.2670) / ov 0.2114 (0.1552);
  [96,112) 0.1189 (0.1550) / 0.1503 (0.1946) / ov 0.3124 (0.2779);
  [112,128) 0.1963 (0.2212) / 0.2991 (0.2835) / ov 0.1237 (0.1534).

## Interpretation (kept separate)

- OBSERVED EFFECT: ARM-P shows numerically WORSE correspondence evidence than
  P2A at GT>=96 on every representation (A_L1 rank-1 -0.1026, mfrac -0.0899;
  same sign for B/C/D; per-bin deficits in [64,80)/[80,96)/[96,112); the
  [112,128) bin (n=107) is essentially tied). Overlap is uniformly higher
  (worse) for ARM-P. No representation shows an ARM-P advantage anywhere except
  noise-level [112,128) mfrac (+0.0156, low-n).
- MECHANISM EVIDENCE: under the D3 proxy, the ARM-P pretrained features do NOT
  display the hypothesised improved-correspondence signature at large disparity.
  This is evidence AGAINST the mechanism in its D3-observable form, but it is
  NOT a validation OR a falsification of ARM-P: D3 is a scalar diagnostic proxy,
  not the model's scoring function; the checkpoint is a partial (7/20-epoch,
  A+C-subset, unannealed-LR) pretrain; and nothing about post-fine-tuning EPE
  was tested here.
- NOT measurable with current artifacts: what a trained aggregation would do
  with these features; any claim about final EPE after fine-tuning.

## Files

- `armp_d3_matching.json` - full ARM-P D3 output (+ checkpoint/config/device metadata)
- `comparison_vs_p2a.json` - delta table vs P2A (cites source files)
- `raw/armp_d3_s0.npz` - per-cell arrays (same layout as d3_s{seed}.npz)
- `scripts/run_armp_d3.py` - wrapper (only new code that ran the diagnostic)
- `scripts/build_comparison.py`, `scripts/print_headlines.py`, `scripts/print_rep_bins.py`
- `ckpt/p2a_best.pth` - ARM-P checkpoint copy (sha256-verified)
