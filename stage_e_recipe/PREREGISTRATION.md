# Stage E — pre-registration, instantiated

The acceptance **rule** was fixed in the spec before E0 ran. E0 supplies only
the **numbers**. Nothing below changes the rule, and no threshold here may be
revisited after a candidate result is seen.

Authority: `docs/superpowers/specs/2026-09-21-stage-e-recipe-design.md` §5, §6
(amendments A1.2, A2.x).

Instantiated 2026-09-21, after E0 completed and **before any candidate ran**.

## 1 E0 control — three fresh Kaggle T4 seeds

Stage-1 pretrained ARM-P init, frozen incumbent recipe, no intervention,
batch 2, 200 epochs, `T_max` 200. Scored through the frozen contract;
`contract_match` true on all six checkpoints.

| seed | best EPE | best D1 % | final EPE | final D1 % | train wall |
|---|---|---|---|---|---|
| 0 | 1.2137377 | 6.3613 | 1.2152821 | 6.3452 | 4,647.3 s |
| 1 | 1.1962889 | 6.3262 | 1.2066586 | 6.3687 | 4,556.6 s |
| 2 | 1.2013258 | 6.4304 | 1.2156821 | 6.4384 | 4,772.2 s |

## 2 The instantiated constants

```
M0_best  = 1.2037841 px      S0_best  = 0.0174488 px
M0_final = 1.2125409 px      S0_final = 0.0090235 px

min(control best)  = 1.1962889 px   (seed 1)
max(control best)  = 1.2137377 px   (seed 0)
```

## 3 What a candidate must beat

| requirement | threshold |
|---|---|
| ACCEPT (E1, E2) | `Mx_best <= 1.1863353` px, i.e. `Δ_best >= 0.0174488` |
| ACCEPT non-overlapping | additionally `max(candidate best seeds) < 1.1962889` |
| E3 also requires | `Mx_final <= 1.2035174` px, i.e. `Δ_final >= 0.0090235` |
| INT8 gate (all) | `P_candidate <= P_control + 0.05` px |
| INCONCLUSIVE band | `1.1863353 < Mx_best < 1.2037841` |
| REJECT | `Mx_best >= 1.2037841`, or INT8 fails, or fewer than 3 seeds |

Verdict order, seeds, tie-breaks and the E3 best+final rule are unchanged from
the spec.

## 4 Two things this control settles

**The environment shift is real and was correctly anticipated.** The Kaggle
control mean is 1.2037841 px against the historical local ARM-P mean of
1.1988735 px — **0.0049106 px higher**. That is well inside the ~0.022 px
local-versus-Kaggle shift Stage B's own environment control measured, and the
spec ruled in advance that such a difference is expected and is **not a
regression**. Had candidates been judged against the historical figure, every
one would have carried a ~0.005 px phantom handicap from the environment alone.
This is what E0 exists to remove.

**The noise floor moved, so the bar moved with it.** `S0_best` is 0.0174488 px,
against the historical local spread of 0.0145422 px — a slightly noisier
control, so a slightly harder bar. The bar is derived from *this* control, as
pre-registered, not carried over.

## 5 INT8 control — MEASURED

Procedure as pre-registered: `quantize_static`, QDQ, QInt8 activations and
weights, per-channel, calibration = first **32** scenes of `hailo_calib`,
`hailo_val` never used. Measured on CPU by `int8_control.py`.

```
P_control = 5.5865268 px    (per seed 5.7081378 / 5.8933010 / 5.1581416)
candidate gate: P_candidate <= 5.6365268 px
```

**ARM-P does not survive INT8.** The control's int8 D1 is 80–86 % against
6.3–6.4 % in fp32 — destroyed, not degraded. Full analysis, including why this
is neither an export problem nor a Hailo statement, in
`INT8_CONTROL_REPORT.md`.

The consequence for this gate is recorded and **not acted on**: at
`P_control = 5.5865268`, the threshold is cleared by any candidate that is
merely *equally* destroyed, so the gate can no longer certify survivability. It
still performs its literal function — forbidding a candidate that makes int8
worse than the control. The rule was fixed before the measurement and is not
being rewritten because the measurement was unwelcome.
