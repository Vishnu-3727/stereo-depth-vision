# F1 pilot report (Stage F M1, seed 0)

[FACT] Checkpoints scored (FINAL only, seed 0):
- control f1c: `f1c_seed0_final.pth`
- M1 lam=0.1 f1m: `f1m_seed0_final.pth`
- M1 lam=1.0 f1m10: `f1m10_seed0_final.pth`

[FACT] Declared environment difference (per task brief / amendment A1 section 3): f1c ran on Kaggle account vishnu3727; both M1 arms (f1m, f1m10) ran on account vishnuvardhanksece. Same GPU type (T4), same E0 recipe/seed/epochs.

## Sanity check (E3 seed0 FINAL, mandatory)

[MEASUREMENT] recomputed contract final EPE = 1.1628882758 (reference 1.1628882758, diff 0.000e+00, tol 0.0001)
[MEASUREMENT] recomputed train GT<64 EPE = 0.8620391636 (reference 0.8620391636, diff 0.000e+00, tol 0.0001)
[VERDICT] sanity status: PASS

## Numbers

| arm | contract final EPE | contract D1 | contract GT<64 EPE | train EPE | train GT<64 EPE | entropy mean | rank median | modes mean |
|---|---|---|---|---|---|---|---|---|
| f1c | 1.202031 | 6.2383 | 1.060605 | 0.975522 | 0.911515 | 1.8532 | 3.0 | 1.0138 |
| f1m | 1.216238 | 6.4398 | 1.070316 | 0.985657 | 0.919970 | 1.9142 | 1.0 | 1.0146 |
| f1m10 | 1.222542 | 6.4876 | 1.067016 | 0.988640 | 0.921859 | 1.9115 | 1.0 | 1.0132 |

## Gate outcomes (per arm vs shared control f1c)

[MEASUREMENT] f1m: dfinal=0.014207, dtrain=0.008456
[VERDICT] f1m: FAIL — dfinal=0.014207 (arm 1.216238 vs f1c 1.202031); dtrain=0.008456 (arm 0.919970 vs f1c 0.911515); not supported under this controlled pilot
[MEASUREMENT] f1m10: dfinal=0.020510, dtrain=0.010344
[VERDICT] f1m10: FAIL — dfinal=0.020510 (arm 1.222542 vs f1c 1.202031); dtrain=0.010344 (arm 0.921859 vs f1c 0.911515); not supported under this controlled pilot

[VERDICT] multiplicity (A1): No single-seed pass: nothing is admitted to 3-seed validation.

## Interpretation (separate from measured facts above)

[INFERENCE] A single-seed PASS/FLOOR-MOVE admits an arm to 3-seed validation only; it is not acceptance. See f1_verdict.json for the frozen gate record.
