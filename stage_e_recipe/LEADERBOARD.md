# Stage E leaderboard

One row per completed experiment. Three Kaggle T4 seeds each, Stage-1
pretrained ARM-P init, frozen architecture at 397,954 parameters, scored only
through `phase1/harness/frozen_eval.py` on the 40-scene contract.

| exp | intervention | seeds | mean best EPE | spread | mean final EPE | verdict |
|---|---|---|---|---|---|---|
| **E0** | none (control) | 0,1,2 | **1.2037841** | 0.0174488 | **1.2125409** | control — no verdict |
| E1 | EMA 0.999 | — | — | — | — | NOT RUN |
| E2 | native batch 8 | — | — | — | — | NOT RUN |
| E3 | 400 epochs | — | — | — | — | NOT RUN |
| E4 | broader pretraining | — | — | — | — | **BLOCKED** (FT3D absent) |

Acceptance thresholds instantiated from E0: `PREREGISTRATION.md`.

## Reference figures — not the acceptance baseline

| figure | value | what it is |
|---|---|---|
| ARM-P local 3-seed mean | 1.1988735 px | historical, different hardware and torch version |
| ARM-P seed 1 local | 1.1912168 px | a single seed, the best of three — never a baseline |
| E0 Kaggle 3-seed mean | **1.2037841 px** | **the Stage-E control** |

The Kaggle control sits 0.0049106 px above the historical local mean, inside
the ~0.022 px environment shift Stage B measured. Expected, pre-registered as
expected, and not a regression.

## Status

- E0 training: **COMPLETE**, 3/3 seeds, `contract_match` true on all six
  checkpoints.
- E0 INT8 control (`P_control`): **NOT MEASURED** — required before any
  candidate can be accepted.
- No candidate authorized.
