# EXP-E2-READOUT-TEMPERATURE-001 — is H2's error sensitive to readout sharpness?

**Date:** 2026-09-08. **Compute:** 160 s, inference only, no training.
**Protocol frozen in advance:** `EXP_E2_READOUT_TEMPERATURE_001_PREREGISTRATION.md`.

**Claim tags:** `MEASURED`, `DERIVED`, `INFERRED`, `UNKNOWN`.

**Record:** `EXP-E2-READOUT-TEMPERATURE-001` — authoritative, completed, no rerun
needed. Raw data: `phase2/results/EXP-E2-READOUT-TEMPERATURE-001/temperature.json`.
Script: `phase2/scripts/exp_e2_readout_temperature.py`. Logic tests:
`phase2/tests/test_e2_readout_temperature.py` (6; full suite **169 passing**).

---

## 1. Verdict

    READOUT-ROBUST   (case B)

`MEASURED` — **no temperature in the frozen grid materially improves accuracy on
any seed.** Not one of the 24 non-control (seed, temperature) cells has a negative
ΔEPE or a negative ΔD1. The viable-temperature set is empty.

`DERIVED` — H2's readout sharpness is **not an exploitable remaining source of
error**. The question E2 was opened to answer is answered: no.

**Read the label precisely.** READOUT-ROBUST is defined in the pre-registration as
"no temperature materially improves accuracy", which is what was measured. It does
**not** mean the model is insensitive to temperature — the opposite is true, and
§4 states it. The sensitivity is entirely one-sided: every departure from T = 1.0,
in either direction, degrades accuracy sharply.

## 2. Hypothesis, intervention and frozen controls

**Hypothesis** (frozen): H2's readout has suboptimal confidence sharpness, such
that changing only the inference-time temperature materially improves accuracy
without destroying stereo functionality.

**Intervention** — exactly one, at exactly one place:

    p(d) = softmax(-C_standardised(d) / T)

implemented as `soft_argmin(scaled / T, dim=1)` in a subclass of
`StandardisedDisparityRegression` defined in the experiment script.
`standardise_across_disparity` and Phase 1's `soft_argmin` are **imported and
called unchanged**; `phase2/models/scaled_regression.py` was not modified.

**Held fixed:** weights, checkpoints, cost volume and `cost_volume_shift="left"`,
aggregation, refinement, the standardisation and its `eps = 1e-5`, the disparity
convention, the 40 evaluation scenes, the scoring code and the probe harnesses.
No training. No other parameter tuned.

**Checkpoint hashes** (identical before and after the run):

| model | SHA-256 |
|---|---|
| H2 (seed 0) | `e3d48021d7f6a3d4d16fe33031e8ba85fd24389c5ba892f52c7e873e48454927` |
| SEED1 | `849a371eb54847b642525c9ee8bc997897ee0a8e94db9894d45c6c582ba5c9a7` |
| SEED2 | `9528c5b68c89c929d71b7acbb9c4d3e1b72f39d9082f3b6c889a8b5c25f4b294` |

## 3. Gates — all passed

| gate | result |
|---|---|
| control matches the untempered H2 readout **in the same process** (≤ 1e-9) | **PASS** — exact |
| control matches `EXP-E1-ERROR-PARTITION-001`'s record (≤ 1e-6) | **PASS** — 2.4004243 / 16.7177738, 2.3193164 / 17.2505921, 2.5561498 / 19.7592193 |
| all outputs finite at every temperature | **PASS** |
| checkpoint SHA-256 unchanged | **PASS** |
| parameter count 423,586 unchanged | **PASS** |
| MACs @368×1232 unchanged (56.0339 G at T=1.0 and T=0.25) | **PASS** |
| `git diff phase-1-frozen -- src scripts` empty, before and after | **PASS** |

## 4. Accuracy and readout behaviour vs temperature

`MEASURED` — 40 `hailo_val` scenes, full 368×1232 frames, pooled over `gt > 0`
(3,802,797 pixels). Deltas are against the same seed's own T = 1.0.

| seed | T | EPE (px) | ΔEPE | D1 (%) | ΔD1 | band | mean entropy | mean peak p | right-dep | map-dep | C2/C3 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| H2 | 0.25 | 19.174 | +16.773 | 97.33 | +80.62 | DEGRADATION | 0.290 | 0.884 | +0.57 | −15.65 | FAIL |
| H2 | 0.5 | 11.563 | +9.163 | 90.16 | +73.44 | DEGRADATION | 0.757 | 0.748 | +5.74 | −8.89 | FAIL |
| H2 | 0.75 | 5.943 | +3.543 | 72.84 | +56.12 | DEGRADATION | 1.303 | 0.600 | +20.25 | +3.61 | FAIL |
| **H2** | **1.0** | **2.400** | — | **16.72** | — | control | 1.726 | 0.473 | **+83.95** | **+69.12** | **ok** |
| H2 | 1.25 | 4.608 | +2.208 | 54.07 | +37.35 | DEGRADATION | 1.992 | 0.380 | +27.44 | +23.04 | ok |
| H2 | 1.5 | 6.850 | +4.450 | 69.89 | +53.17 | DEGRADATION | 2.151 | 0.316 | +9.15 | +11.51 | FAIL |
| H2 | 2.0 | 9.700 | +7.300 | 80.11 | +63.39 | DEGRADATION | 2.309 | 0.240 | −2.55 | +4.51 | FAIL |
| H2 | 3.0 | 12.329 | +9.929 | 85.12 | +68.40 | DEGRADATION | 2.414 | 0.174 | −0.45 | +1.82 | FAIL |
| H2 | 4.0 | 13.501 | +11.101 | 86.73 | +70.01 | DEGRADATION | 2.447 | 0.146 | −1.69 | +0.59 | FAIL |
| SEED1 | 0.25 | 11.780 | +9.461 | 96.27 | +79.02 | DEGRADATION | 0.310 | 0.874 | −2.98 | −7.98 | FAIL |
| SEED1 | 0.5 | 10.965 | +8.646 | 96.35 | +79.10 | DEGRADATION | 0.713 | 0.746 | −2.50 | −9.24 | FAIL |
| SEED1 | 0.75 | 7.128 | +4.809 | 86.13 | +68.88 | DEGRADATION | 1.239 | 0.607 | −0.08 | −5.85 | FAIL |
| **SEED1** | **1.0** | **2.319** | — | **17.25** | — | control | 1.675 | 0.482 | **+76.78** | **+67.78** | **ok** |
| SEED1 | 1.25 | 6.320 | +4.001 | 78.36 | +61.11 | DEGRADATION | 1.957 | 0.388 | −2.56 | +1.85 | FAIL |
| SEED1 | 1.5 | 10.406 | +8.086 | 91.57 | +74.32 | DEGRADATION | 2.127 | 0.323 | −11.01 | −9.26 | FAIL |
| SEED1 | 2.0 | 15.870 | +13.550 | 95.81 | +78.56 | DEGRADATION | 2.297 | 0.245 | −13.04 | −8.23 | FAIL |
| SEED1 | 3.0 | 21.250 | +18.931 | 97.12 | +79.87 | DEGRADATION | 2.410 | 0.177 | −7.87 | −8.77 | FAIL |
| SEED1 | 4.0 | 23.774 | +21.455 | 97.42 | +80.17 | DEGRADATION | 2.446 | 0.149 | −5.15 | −7.34 | FAIL |
| SEED2 | 0.25 | 16.496 | +13.940 | 96.86 | +77.10 | DEGRADATION | 0.450 | 0.848 | −1.17 | −5.40 | FAIL |
| SEED2 | 0.5 | 11.979 | +9.423 | 92.23 | +72.47 | DEGRADATION | 0.934 | 0.702 | +1.49 | −4.76 | FAIL |
| SEED2 | 0.75 | 6.757 | +4.200 | 80.16 | +60.40 | DEGRADATION | 1.416 | 0.564 | +10.08 | −1.81 | FAIL |
| **SEED2** | **1.0** | **2.556** | — | **19.76** | — | control | 1.795 | 0.446 | **+74.88** | **+59.96** | **ok** |
| SEED2 | 1.25 | 5.262 | +2.706 | 65.42 | +45.66 | DEGRADATION | 2.036 | 0.359 | +16.51 | +16.59 | FAIL |
| SEED2 | 1.5 | 8.194 | +5.638 | 79.06 | +59.30 | DEGRADATION | 2.180 | 0.299 | −1.17 | +0.21 | FAIL |
| SEED2 | 2.0 | 11.995 | +9.439 | 84.98 | +65.22 | DEGRADATION | 2.324 | 0.228 | −5.30 | −1.40 | FAIL |
| SEED2 | 3.0 | 15.579 | +13.023 | 87.36 | +67.60 | DEGRADATION | 2.419 | 0.167 | −5.01 | +0.68 | FAIL |
| SEED2 | 4.0 | 17.201 | +14.645 | 87.95 | +68.19 | DEGRADATION | 2.450 | 0.142 | −2.08 | +0.59 | FAIL |

`MEASURED` — no cell has ΔEPE < 0 and no cell has ΔD1 < 0; every non-control cell
degrades on **both** metrics simultaneously, so no row is a mixed-direction case.

### Readout statistics at the control temperature

`MEASURED`, pooled over the same 40 scenes:

| seed | mean entropy | median entropy | mean peak p | **max peak p** | mean top-2 gap | `disparity_initial` mean / std (candidates) |
|---|---:|---:|---:|---:|---:|---:|
| H2 | 1.726 | 1.816 | 0.473 | **0.769** | 0.300 | 3.60 / 1.95 |
| SEED1 | 1.675 | 1.747 | 0.482 | **0.772** | 0.288 | 2.86 / 1.29 |
| SEED2 | 1.795 | 1.748 | 0.446 | **0.771** | 0.301 | 3.12 / 1.58 |

`MEASURED` — the maximum peak probability observed anywhere is **0.769–0.772**,
against the **≈0.77** ceiling the H2 report derived analytically for a lone winner
among 12 standardised candidates. `DERIVED` — that prediction is confirmed by
direct measurement over 3.8 M pixels per seed; the cap is real and is reached.

`MEASURED` — the control's entropy (1.675–1.795 nats of a possible 2.4849)
reproduces the H2 report's recorded 1.67–1.88, and the temperature knob moves
entropy monotonically across the whole available range: 0.290 nats at T = 0.25
(approaching H1's collapsed 0.0000) to 2.450 at T = 4.0 (approaching the uniform
2.4849).

## 5. Stereo functionality

`MEASURED` — the frozen constraints **C2** (right-image dependence ≥ +20 D1
points, worst case) and **C3** (matching-map dependence ≥ +20) are met at
**T = 1.0 on all three seeds and essentially nowhere else**. The single other
passing cell is H2 at T = 1.25 (+27.44 / +23.04) — and it is a MATERIAL
DEGRADATION of +2.208 px and +37.35 points, so it is not a candidate for
anything.

`MEASURED` — moving away from T = 1.0 in **either** direction drives stereo
dependence to zero or negative:

- sharpening (T ≤ 0.5): right-dependence +0.57 to −2.98, matching-map −15.65 to
  −7.98 — corrupting the right image or destroying the matching map makes the
  model *better*;
- softening (T ≥ 2.0): right-dependence −13.04 to −0.45, matching-map −8.77 to
  +4.51.

`DERIVED` — both extremes reproduce the **H1 monocular signature** behaviourally,
from a model whose weights, cost volume, aggregation and standardisation are
untouched. Sharpening recreates H1's saturation numerically as well: entropy
0.290 nats at T = 0.25 against H1's measured 0.0000.

`INFERRED` — H2's trained solution is tightly coupled to the readout sharpness it
was trained under, and its use of the second camera is a property of that
coupling, not of the cost volume alone. This is the same lesson E3 produced from
the other end of the network, where truncating refinement also destroyed stereo
dependence.

`DERIVED` — **no Case C failure occurred**: nothing improved accuracy at all, so
there was no apparent improvement to reject on stereo grounds. **No Case D
disagreement occurred**: all three seeds agree that T = 1.0 is best, and agree on
the direction and rough magnitude of every degradation.

## 6. Interpretation

1. `DERIVED` — **E2 answers its question cleanly: no.** H2's remaining error is
   not materially sensitive to readout sharpness in the improving direction. The
   `UNKNOWN` left open by the H2 report ("whether the confidence cap costs
   accuracy") is now closed for the inference-time case: it does not, at least
   not in a way a temperature can recover.
2. `MEASURED` — the sensitivity is **one-sided and steep**. T = 0.75 and T = 1.25
   — a ±25 % change in a single scalar with no parameters — cost +3.5 px / +56
   points and +2.2 px / +37 points respectively on seed 0. `INFERRED` — H2 sits
   at a sharp optimum of its own training, which is what one expects when the
   readout is the operator the network was optimised through.
3. `DERIVED` — the temperature knob is therefore **not a free lever**, and E2 is
   closed as a Stage E candidate. It is not evidence that temperature is
   uninteresting as a *training* variable; see §7.
4. `DERIVED` — this closes the last no-training candidate from the evidence
   review. Together with E1 (partition not identifiable; ≈⅓ upstream, ≈⅔
   downstream) and E3 (refinement right-sized; truncation destroys stereo), the
   cheap inference-time explanations for H2's remaining error are exhausted.

## 7. Limitations

- `UNKNOWN`, and the most important one — **what a model trained at a different
  temperature would do.** A network adapts to the readout it is optimised
  through, so a negative inference-time result is **weak** evidence about
  temperature as a design variable. This was stated before the run and is not a
  retrofit.
- `UNKNOWN` — a *learnable* or annealed temperature, which is a training change
  and is out of scope here.
- The band classifier's catch-all labels any non-negligible, non-improving cell
  as MATERIAL DEGRADATION, including hypothetical mixed-direction cells. No
  measured cell was mixed (§4), so every label in the table is accurate; the
  classifier's behaviour on cases that did not occur is noted rather than
  silently relied on.
- Three seeds, 40 scenes for accuracy, four focus scenes for the probes, one
  training budget. No significance test, none claimed.
- Inference-only: the matching-path gradient criterion was not remeasured, as
  pre-registered.
- The stereo probes use the worst case over corruptions and scenes; a per-scene
  or best-case reading would be more permissive and is not used.

## 8. Next step — recommendation only, nothing opened

Per the experiment's own instruction, this stops here for evidence review.

The cheap questions are now exhausted. Every remaining candidate requires
training:

1. **E3b — retrain refinement at four and five blocks** under the H2 recipe.
   E3 measured that pruning a trained six-block stack is never free and that
   which block is redundant differs by seed; whether a *trained* smaller stack
   matches H2 is the question E3 could not answer. ~1 GPU-hour per arm.
2. **E2b — train with a different (or learnable) readout temperature.** E2 shows
   the trained solution is tightly coupled to T = 1.0; whether a different T
   trains to a better optimum is untested and is the only honest follow-up to
   this result.
3. **O6 — Scene Flow pretraining.** Still the largest single confound behind
   every Phase 2 number: 160 training scenes from random initialisation.
   Estimated ~22 h at the measured 8.8 samples/s, blocked on dataset size.

`INFERRED` — of the three, (1) has the clearest hypothesis and the smallest cost,
and (3) has the largest expected effect on the absolute numbers. Nothing measured
so far implicates the cost volume, so **Stage E remains formally CLOSED**.
