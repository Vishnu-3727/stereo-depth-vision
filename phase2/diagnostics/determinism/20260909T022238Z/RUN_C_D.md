# RUN_C / RUN_D — the same epoch with determinism controls on

Records: `run_C_deterministic/record.json`, `run_D_deterministic/record.json`.
Machine-generated diff: `comparison_run_C_deterministic_vs_run_D_deterministic.json`.

```
python phase2/diagnostics/determinism/diag_determinism.py \
    --out .../run_C_deterministic --deterministic --label "..."
```

## Controls applied

| control | value | note |
|---|---|---|
| `torch.use_deterministic_algorithms(True)` | on | **accepted without error** |
| `torch.backends.cudnn.deterministic` | `True` | |
| `torch.backends.cudnn.benchmark` | `False` | already the default here |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | set before the CUDA context is created |

`MEASURED` — `determinism_setup_error: null` and `nondeterministic_op_error:
null` in both records. **No operation in this architecture refused to run under
deterministic algorithms.** The whole epoch and both validations completed.

## Result: bit-for-bit reproduction across fresh processes

| item | RUN_C | RUN_D | |
|---|---:|---:|---|
| batch-0 loss | 17.472482681274414 | 17.472482681274414 | SAME |
| batch-0 output sha | `4a11334df1550f42` | `4a11334df1550f42` | SAME |
| batch-0 grad norm | 308.034393 | 308.034393 | SAME |
| first batch whose loss differs | — | — | **none** |
| first batch whose grad norm differs | — | — | **none** |
| epoch mean loss | 10.532406491041183 | 10.532406491041183 | SAME |
| last-batch loss | 8.387198448181152 | 8.387198448181152 | SAME |
| median grad norm | 54.79760932922363 | 54.79760932922363 | SAME |
| **weights after epoch, sha** | **`356da4325d74ef4d`** | **`356da4325d74ef4d`** | **SAME** |
| validation EPE | 15.32539188316842 | 15.32539188316842 | **SAME** |
| validation D1 | 87.44677261821234 | 87.44677261821234 | **SAME** |
| forward ×2 in process | identical | identical | |
| **backward ×2 in process** | **identical, 0 elements differ** | **identical, 0 elements differ** | |

`MEASURED` — ΔEPE and ΔD1 between two fresh processes are **exactly 0.0**, and
the trained weight hash matches. Every quantity is reproduced.

## Cost

| | default flags | determinism on |
|---|---:|---:|
| one epoch, wall clock | 36.1 s / 37.5 s | 42.0 s / 41.2 s |

`DERIVED` — roughly **+13 %** wall clock. On the 200-epoch recipe that is
≈3,600 s → ≈4,100 s, about **8 minutes per run**.

## The flags change the answer, and that is expected

| | EPE | D1 | mean loss |
|---|---:|---:|---:|
| RUN_A (default) | 17.496795 | 89.526119 | 10.765871 |
| RUN_C (deterministic) | 15.325392 | 87.446773 | 10.532406 |

`MEASURED` — batch-0 loss also differs in its last bits
(17.472463607788086 vs 17.472482681274414). `DERIVED` — deterministic mode
selects different kernels, so it is a different point in the same noise
distribution, not a corrected version of RUN_A. `INFERRED`, stated as such — the
one-epoch gap of 2.17 px between A and C is **not** evidence that determinism
improves accuracy; one epoch is deep inside the chaotic early regime the
seed-replication report documented, and no comparison of that kind is claimed
here.
