# Stage E leaderboard

One row per completed experiment. Three Kaggle T4 seeds each, Stage-1
pretrained ARM-P init, frozen architecture at 397,954 parameters, scored only
through `phase1/harness/frozen_eval.py` on the 40-scene contract.

| exp | intervention | seeds | mean best EPE | spread | mean final EPE | verdict |
|---|---|---|---|---|---|---|
| **E0** | none (control) | 0,1,2 | **1.2037841** | 0.0174488 | **1.2125409** | control — no verdict |
| E1 | EMA 0.999 | 0,1,2 | 1.2036174 | 0.0314905 | 1.2044355 | **INCONCLUSIVE** |
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
- E0 INT8 control: **MEASURED** — `P_control` = **5.5865268 px**, candidate
  gate `P_candidate <= 5.6365268`.
- **ARM-P does not survive INT8**: int8 D1 80–86 % against 6.4 % fp32, a
  penalty 16.3x the reference model's under the identical procedure. Not a
  Hailo statement; Stage D stays BLOCKED. See `INT8_CONTROL_REPORT.md`.
- **E1 complete — verdict INCONCLUSIVE** (`e1_verdict.json`, computed by
  `verdict.py`). `Δ_best` = +0.0001667: positive, but ~100x smaller than the
  `S0_best` bar of 0.0174488, so indistinguishable from the control's own seed
  noise. Per-seed best: 1.2164890 / 1.2093647 / 1.1849985.
- E1's INT8 limb passed (P 5.0562891 ≤ 5.6365268) and that decides nothing
  about survivability — E1 is destroyed by int8 just as the control is. This is
  the vacuity recorded in `INT8_CONTROL_REPORT.md` §4, now demonstrated.
- E1 seed spread is 0.0314905, **1.8x the control's 0.0174488**. EMA did not
  stabilise seed-to-seed variation here; it widened it. E1 produced both the
  best single run of the campaign so far (seed 2, 1.1849985) and the worst
  (seed 0, 1.2164890).
- No further candidate authorized.
