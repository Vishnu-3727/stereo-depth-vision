# Twin pilot report — SELFTEST (f1c FINAL stands in for both arms)

[FACT] Checkpoints scored (FINAL only, seed 0):
- t48 (stand-in): `f1c_seed0_final.pth` (width 32)
- t48c (stand-in): `f1c_seed0_final.pth` (width 32)

[FACT] Selftest stands the f1c FINAL checkpoint in for BOTH arms (the twin checkpoints do not exist yet); d must be 0.0 exactly.

## Sanity check (E3 seed0 FINAL, mandatory)

[MEASUREMENT] recomputed contract final EPE = 1.1628882758 (reference 1.1628882758, diff 0.000e+00, tol 0.0001)
[MEASUREMENT] recomputed train GT<64 EPE = 0.8620391636 (reference 0.8620391636, diff 0.000e+00, tol 0.0001)
[VERDICT] sanity status: PASS

## Numbers

| arm | width | params | contract final EPE | contract D1 | contract GT<64 EPE | train EPE | train GT<64 EPE | s/epoch |
|---|---|---|---|---|---|---|---|---|
| t48 | 32 | 397954 | 1.202031 | 6.2383 | 1.060605 | 0.975522 | 0.911515 | 24.15121881365776 |
| t48c | 32 | 397954 | 1.202031 | 6.2383 | 1.060605 | 0.975522 | 0.911515 | 24.15121881365776 |

## Verdict (frozen §6)

[MEASUREMENT] d=0.000000, contract dfinal=0.000000 (information only)
[VERDICT] CAPACITY WEAKENED — d=train_GT<64(t48)-train_GT<64(t48c)=0.000000 (t48 0.911515 vs t48c 0.911515); contract dfinal for information: 0.000000 (t48 1.202031 vs t48c 1.202031); |d| < 0.020

## Interpretation (separate from measured facts above)

[INFERENCE] SUPPORTED/WEAKENED/INCONCLUSIVE is the frozen §6 reading of the train-floor move; neither is proof. See twin_verdict.json for the frozen gate record.
