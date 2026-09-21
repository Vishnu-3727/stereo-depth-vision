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

## 5 INT8 control — NOT YET MEASURED

`P_control` does not exist yet. Until it is measured by the pre-registered
procedure (`quantize_static`, QDQ, QInt8 activations and weights, per-channel,
calibration = first **32** scenes of `hailo_calib`, never `hailo_val`), **no
candidate can be accepted on INT8 grounds**. A candidate that clears the EPE
bar while `P_control` is unmeasured is not an ACCEPT; it is an unfinished
comparison.
