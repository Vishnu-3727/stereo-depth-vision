# RUN_B — one epoch, default flags, completely fresh process

Record: `run_B/record.json`. Identical invocation to RUN_A except the output
directory and the label:

```
python phase2/diagnostics/determinism/diag_determinism.py \
    --out phase2/diagnostics/determinism/20260909T022238Z/run_B \
    --label "RUN_B default flags, fresh process"
```

Machine-generated diff: `comparison_run_A_vs_run_B.json`.

## Environment — identical to RUN_A

`MEASURED` — HEAD `58e8a19`, python 3.12.9, torch 2.7.0+cu128, CUDA 12.8,
cuDNN 90701, RTX 4060 Laptop (cap 8.9), driver 616.64, and every flag equal.
No environment variable differs.

## Identical to RUN_A

| item | RUN_A | RUN_B | |
|---|---|---|---|
| initial weights sha | `c4d02385e3da28a5` | `c4d02385e3da28a5` | **SAME** |
| numpy / torch CPU / torch CUDA state after build | `129790f8…` / `96616bad…` / `374708ff…` | same | **SAME** |
| batch-0 sample indices | 111, 54 | 111, 54 | **SAME** |
| batch-0 scenes | `000111_10.png`, `000054_10.png` | same | **SAME** |
| batch-0 crops | (96,459), (8,11) | same | **SAME** |
| batch-0 jitter | −0.0132105 / +0.0640423, −0.0535669 / +0.0361595 | same | **SAME** |
| batch-0 left / right / disparity sha | `b87563da2698e20c` / `947f0d2c884b6275` / `c2cbb4218344e505` | same | **SAME** |
| **data order, all 160 items of the epoch** | — | — | **IDENTICAL, first difference: none** |
| batch-0 forward output sha | `9ac6c0574d69a414` | `9ac6c0574d69a414` | **SAME** |
| **batch-0 loss** | 17.472463607788086 | 17.472463607788086 | **SAME** |

`MEASURED` — the two runs see the same data, in the same order, with the same
crops and the same augmentation, from the same initial weights, and compute the
same batch-0 forward pass to the last bit.

One RNG stream does differ: python's stdlib `random`
(`96f75b923b1a9f58` vs a different digest at every checkpoint). `MEASURED` — it
is never seeded by the harness and never consumed by it; the identical data
order and identical batch-0 tensors prove it takes no part in the pipeline.

## Where RUN_B departs from RUN_A

| item | RUN_A | RUN_B | Δ |
|---|---:|---:|---:|
| batch-0 loss | 17.472463607788086 | 17.472463607788086 | **0** |
| **batch-0 grad norm** | 308.035553 | 308.035583 | **+3.05e-05 — the first difference** |
| batch-1 loss | 15.0798702240 | 15.0798749924 | +4.77e-06 |
| batch-2 loss | 19.7273731232 | 19.7267875671 | −5.86e-04 |
| batch-5 loss | 11.4435043335 | 11.4379863739 | −5.52e-03 |
| batch-20 loss | — | — | 2.97e-02 |
| batch-40 loss | — | — | 1.63e-01 |
| batch-60 loss | — | — | 7.94e-01 |
| batch-79 loss | 8.576350212097168 | 9.650322914123535 | **+1.074** |
| epoch mean loss | 10.765870976448060 | 10.792867290973664 | +2.70e-02 |
| median grad norm | 47.749754 | 46.424570 | −1.325 |
| weights after epoch, sha | `eda467853cd05550` | `6f437e1198893e43` | **differ** |
| wall clock | 36.1 s | 37.5 s | — |

## Validation after one epoch

| | EPE (px) | D1 (%) |
|---|---:|---:|
| RUN_A | 17.496794678353552 | 89.52611915841908 |
| RUN_B | 18.750414115637223 | 88.29045952570010 |
| **Δ** | **+1.253619** | **−1.235660** |

`MEASURED` — both runs' in-process validation repeat is bitwise identical, so
this 1.25 px / 1.24 point spread is produced during training, not measurement.

`DERIVED` — **after a single epoch**, two same-seed runs of the identical
configuration already differ by more than the E3b/E3c materiality band
(0.2138 px / 1.0000 point) on both metrics.

## In-process repeat probes

| probe | RUN_A | RUN_B |
|---|---|---|
| forward ×2, same weights, same batch | bitwise identical (0.000e+00) | bitwise identical (0.000e+00) |
| backward ×2, same weights, same batch | **differs**, max 8.297e-05, 309,529 elements | **differs**, max 5.150e-05, 307,640 elements |
