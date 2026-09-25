# L1 pilot report (Stage F continuation, seed 0)

[FACT] Checkpoints scored (FINAL only, seed 0):
- control l1c: `l1c_seed0_final.pth` (Smooth-L1 control)
- masked L1 l1m: `l1m_seed0_final.pth`
- Both width 32, 397954 params (same architecture).

[FACT] Declared environment sameness (per L1 launch note section 4): l1c and l1m both run on Kaggle account vishnuvardhanksks — same account, no cross-account difference. Same GPU type (T4), same continuation recipe/seed/epochs (E3 seed0 FINAL init, 40 epochs, fresh Adam lr 1e-4, cosine to 0); the arms differ ONLY in the loss knob (Smooth-L1 vs masked L1).

## Sanity check (E3 seed0 FINAL, mandatory)

[MEASUREMENT] recomputed contract final EPE = 1.1628882758 (reference 1.1628882758, diff 0.000e+00, tol 0.0001)
[MEASUREMENT] recomputed train GT<64 EPE = 0.8620391636 (reference 0.8620391636, diff 0.000e+00, tol 0.0001)
[VERDICT] sanity status: PASS

## Numbers

| arm | contract final EPE | contract D1 | contract GT<64 EPE | train EPE | train GT<64 EPE |
|---|---|---|---|---|---|
| l1c | 1.166182 | 6.0025 | 1.023466 | 0.916667 | 0.859482 |
| l1m | 1.161015 | 5.9966 | 1.020096 | 0.914675 | 0.857414 |

[MEASUREMENT] per-arm sha256 / strict-load / params:
- l1c sha256=eee90ffcdd61d15418f6a6d53ebbb93cbf7fbab80609fa3bfb8121d02b9d6091 strict_load=True params=397954
- l1m sha256=a8119f0713879c452b7421907d84be54d53d7d4f44f9414c41054024533d4e77 strict_load=True params=397954

## Gate outcome (frozen section 8)

[MEASUREMENT] dfinal=-0.005167 (l1m 1.161015 vs l1c 1.166182)
[VERDICT] FAIL — dfinal=contract_final_EPE(l1m)-contract_final_EPE(l1c)=-0.005167 (l1m 1.161015 vs l1c 1.166182); dfinal > -0.030
[VERDICT] Single-seed FAIL: nothing is admitted to 3-seed validation.

## Information only (not part of the gate)

[MEASUREMENT] l1c vs E3 seed0 FINAL start: 0.003294 (l1c 1.166182 vs start 1.162888)
[MEASUREMENT] l1m vs E3 seed0 FINAL start: -0.001874 (l1m 1.161015 vs start 1.162888)

## Interpretation (separate from measured facts above)

[INFERENCE] A single-seed PASS admits to 3-seed validation only; it is not acceptance. A FAIL closes the L1 mechanism under this controlled pilot. See l1_verdict.json for the frozen gate record.
