# RUN B — fresh-process reproducibility check, identical configuration

Record: `experiments/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNB/`,
`run_B_meta.json`, `eval40_run_B.json`. Machine-generated diff:
`comparison.json`.

```
python phase2/diagnostics/determinism/stageb_deterministic_baseline.py \
    train --run B --out <this directory>
```

Completely fresh process, same seed 0, same configuration, same deterministic
controls, 200 epochs. Launched only after Run A had finished, so the two never
shared the GPU and the wall-clock figures are comparable.

**This is not another architecture experiment.** It is the reproducibility check.

## Identical to Run A — every recorded quantity

`MEASURED`:

| quantity | Run A | Run B | |
|---|---|---|---|
| initial weight sha | `c4d02385e3da28a5` | `c4d02385e3da28a5` | **SAME** |
| epochs completed / aborted | 200 / `null` | 200 / `null` | **SAME** |
| **200-epoch train-loss series** | — | — | **IDENTICAL; first differing epoch: none** |
| **validation series (42 points)** | — | — | **IDENTICAL; first differing point: none** |
| late window EPE mean ± sd | 3.8670192 ± 0.1113151 | 3.8670192 ± 0.1113151 | **SAME** |
| late window D1 mean ± sd | 23.4954125 ± 0.9342410 | 23.4954125 ± 0.9342410 | **SAME** |
| all ten late-window points | see `RUN_A.md` | identical to 10 decimals | **SAME** |
| epoch 200 val EPE / D1 / loss | 3.8346742 / 23.0470937 / 3.3030083 | identical | **SAME** |
| epoch 0 val EPE / D1 | 31.9576479 / 95.1349394 | identical | **SAME** |
| gradient median / max | 21.9161959 / 308.0343933 | identical | **SAME** |
| NaN·Inf / batches > 1e4 | none / 0 | none / 0 | **SAME** |
| matching gradient (median, max, fraction) | 7.3321800 / 306.6581794 / 100 % | identical | **SAME** |
| epoch-10 gate | PASS, EPE 31.9576→9.0015 | identical | **SAME** |
| epoch-100 collapse check | PASS | identical | **SAME** |
| stereo verdict @200 (all five probes) | +80.8127994 / +66.1876924 / 1.7754723 / 18.0000028 / FUNCTIONAL | identical | **SAME** |
| **final weight sha** | **`3ad382c50d413ad2`** | **`3ad382c50d413ad2`** | **SAME** |
| **all 72 weight tensors** | — | — | **BITWISE EQUAL, max abs diff 0.0** |
| **all 6 intermediate snapshots (ep 10/20/50/100/150/200)** | — | — | **BITWISE EQUAL** |
| 40-scene pooled EPE / D1 | 2.2955182 / 15.6045143 | 2.2955182 / 15.6045143 | **SAME** |
| all 40 per-scene rows | — | — | **IDENTICAL** |

## The two quantities that are not identical, and why

`MEASURED` — **checkpoint *file* sha256 differs**:

| | |
|---|---|
| Run A | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` |
| Run B | `adcfb6d667d553679d68026310d7d63d6162775b34d8970d36c55fe6f56da1d0` |

`MEASURED` — the cause was measured, not assumed. Loading both files and
comparing their contents field by field: all **72** tensors are bitwise equal
(max abs diff **0.0**), and the *only* differing element of the saved payload is
the embedded config string

```
config.experiment: "…-ARM-A-RUNA"  vs  "…-ARM-A-RUNB"
```

i.e. the record's own name. `DERIVED` — the file hashes differ because each
checkpoint stores its own record id alongside the weights; the **weights are
identical**. The same applies to the six snapshot files.

`MEASURED` — **wall clock differs**: 4,948.997 s (A) vs 4,927.781 s (B),
−21.2 s (−0.43 %). `DERIVED` — wall clock is not a deterministic quantity and no
protocol claims it is.

## Run B figures

| | value |
|---|---|
| epochs completed | 200 / 200, `aborted: null` |
| wall clock | **4,927.781 s = 82.13 min** (24.639 s/epoch) |
| record status | `completed`, `error: null` |
| final weight sha | `3ad382c50d413ad2` |
| deterministic controls | `use_deterministic_algorithms=True`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| determinism errors raised or suppressed | none |
