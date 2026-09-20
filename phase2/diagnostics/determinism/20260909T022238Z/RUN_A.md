# RUN_A — one epoch, default flags, fresh process

Record: `run_A/record.json`. Command:

```
python phase2/diagnostics/determinism/diag_determinism.py \
    --out phase2/diagnostics/determinism/20260909T022238Z/run_A \
    --label "RUN_A default flags"
```

Flags are exactly those every Phase 2 experiment ran under: no
`use_deterministic_algorithms`, `cudnn.deterministic=False`,
`cudnn.benchmark=False`, `cudnn.allow_tf32=True`,
`cuda.matmul.allow_tf32=False`, `CUBLAS_WORKSPACE_CONFIG` unset.

## Provenance

`MEASURED` — HEAD `58e8a19`, `phase-1-frozen` → commit `b4207e5`,
`git diff phase-1-frozen -- src scripts` empty, working tree `M .gitignore` and
`?? phase2/`. Full detail in `ENVIRONMENT.txt`.

## RNG streams (sha256 of raw state, first 16 hex)

| point | python `random` | numpy global | torch CPU | torch CUDA |
|---|---|---|---|---|
| before `build_model` | `96f75b923b1a9f58` | `151bc7ceca853da9` | `a5bfc0da9e2c4494` | `89b56329ff74558c` |
| after `build_model` | `96f75b923b1a9f58` | `129790f8d58ee719` | `96616bad7a384d6c` | `374708fff7719dd5` |
| after DataLoader built | `96f75b923b1a9f58` | `129790f8d58ee719` | `96616bad7a384d6c` | `374708fff7719dd5` |
| at batch 0 | `96f75b923b1a9f58` | `129790f8d58ee719` | `d694240d043c4dc0` | `374708fff7719dd5` |

`MEASURED` — the pre-build states are whatever the process started with;
`build_model` calls `torch.manual_seed(0)` and `np.random.seed(0)`, and from that
point on the torch and numpy streams are pinned.

## Initialisation

`MEASURED` — 423,586 parameters, 6 refinement blocks, `cost_volume_shift=left`,
regression `StandardisedDisparityRegression`, concatenated-weight sha
**`c4d02385e3da28a5`**.

## Batch 0 — what the model actually saw

| | value |
|---|---|
| batch shape | `[2, 3, 256, 512]` |
| sample 0 | dataset index **111**, `000111_10.png`, crop y=**96** x=**459**, jitter L −0.013210486329130189 R +0.06404226504432821 |
| sample 1 | dataset index **54**, `000054_10.png`, crop y=**8** x=**11**, jitter L −0.0535669373161111 R +0.03615950549094848 |
| left tensor sha | **`b87563da2698e20c`** |
| right tensor sha | `947f0d2c884b6275` |
| disparity sha | `c2cbb4218344e505` |
| valid GT pixels | 68,599 |
| max GT disparity | 59.3046875 px |

## Batch 0 — computation

| | value |
|---|---|
| forward output sha | `9ac6c0574d69a414` |
| **batch-0 loss** | **17.472463607788086** |
| forward ×2 in one process | **bitwise identical**, max abs diff 0.000e+00 |
| backward ×2 in one process | **NOT identical** — max abs diff **8.297e-05**, **309,529** gradient elements differ |
| batch-0 grad norm | 308.035553 |

## Epoch

| | value |
|---|---|
| batches | 80 |
| first-batch loss | 17.472463607788086 |
| last-batch loss | 8.576350212097168 |
| mean loss | 10.765870976448060 |
| median grad norm | 47.749754 |
| max grad norm | 308.0355529785156 |
| NaN / Inf | none |
| weights after epoch, sha | `eda467853cd05550` |
| wall clock | 36.1 s |

## Validation after one epoch

Protocol: first 10 `hailo_val` scenes, full 368×1232, pooled over `gt > 0`
(the recorded protocol, `validate` imported unchanged).

| | EPE (px) | D1 (%) |
|---|---:|---:|
| run 1 | 17.496794678353552 | 89.52611915841908 |
| run 2, same weights, same process | 17.496794678353552 | 89.52611915841908 |

`MEASURED` — validation repeated in-process is **bitwise identical**
(ΔEPE 0.000e+00, ΔD1 0.000e+00). Evaluation is not a source of noise.
