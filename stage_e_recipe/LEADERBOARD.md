# Stage E leaderboard

One row per completed experiment. Three Kaggle T4 seeds each, Stage-1
pretrained ARM-P init, frozen architecture at 397,954 parameters, scored only
through `phase1/harness/frozen_eval.py` on the 40-scene contract.

| exp | intervention | seeds | mean best EPE | spread | mean final EPE | verdict |
|---|---|---|---|---|---|---|
| **E0** | none (control) | 0,1,2 | **1.2037841** | 0.0174488 | **1.2125409** | control — no verdict |
| E1 | EMA 0.999 | 0,1,2 | 1.2036174 | 0.0314905 | 1.2044355 | **INCONCLUSIVE** |
| E2 | native batch 8 | 0,1,2 | 1.2390444 | 0.0037600 | 1.2480458 | **REJECT** |
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
- **E2 complete — verdict REJECT** (`e2_verdict.json`). `Δ_best` = −0.0352603,
  about **2x the control's entire seed spread in the wrong direction**. Every
  E2 seed is worse than every control seed and every E1 seed. This is not
  noise. Per-seed best: 1.2414324 / 1.2380283 / 1.2376725.
- E2's INT8 limb also passed (P 5.5653839 ≤ 5.6365268) while the model is
  rejected outright — a second demonstration that the gate certifies nothing.
- **E2's spread is 0.0037600: 4.6x tighter than the control and 8.4x tighter
  than E1.** Batch 8 made runs highly consistent with each other and
  consistently worse — the classic large-batch signature at fixed LR, with 20
  optimizer steps per epoch instead of 80. The pre-registration deliberately
  held LR at 1e-3 so batch size was the single lever; this is the measured cost
  of that lever as specified, not a defect. An LR-scaled variant would be a
  separate experiment with its own pre-registration.
- E2 was also the fastest (~70 min/seed against E0's ~78), the 18 % per-epoch
  gain the T4 rate probe predicted. Cheapest and worst.
- No further candidate authorized. E3 (400 epochs) is the last recipe lever.
