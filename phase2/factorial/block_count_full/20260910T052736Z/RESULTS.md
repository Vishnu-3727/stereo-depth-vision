# EXP-BLOCKCOUNT-FULL-001 — BATCH 2 results

Record `phase2/factorial/block_count_full/20260910T052736Z/`.
Pre-registered in `PREREGISTRATION.md`, frozen before the run.

One run — 5 blocks, seed 2, 200 epochs. Completed 200/200, exit 0, no abort, no
NaN/Inf. Wall clock **4777.6 s (1.33 GPU-hours)**. Epoch-10 viability gate PASS,
epoch-100 collapse check PASS, epoch-200 verdict STEREO-FUNCTIONAL.

No other run was launched. See *HARD STOP*.

---

## QUESTION

> Does the 5-block arm remain separated from the 6-block seed distribution when
> a third seed is added?

---

## RESULTS

40-scene `hailo_val` evaluation of the final checkpoint at epoch 200, pooled
over `gt > 0` — the authoritative metric. Every row but the last is read from a
frozen record and was not retrained.

| Blocks | Seed | EPE | D1 | RMSE | source |
| -----: | ---: | -----: | ------: | -----: | ------ |
| 6 | 0 | 2.2955 | 15.6045 | 6.044 | Stage B |
| 6 | 1 | 2.3982 | 18.3357 | 5.677 | batch 1 |
| 6 | 2 | 2.3950 | 17.4699 | 5.935 | batch 1 |
| 5 | 0 | 2.6458 | 21.3712 | 6.053 | deterministic series |
| 5 | 1 | 2.4730 | 18.2062 | 6.215 | batch 1 |
| 5 | 2 | **2.6373** | **20.6594** | **6.444** | **this run** |
| 4 | 0 | 2.3387 | 16.9797 | 5.865 | deterministic series, not retrained |

The new run also recorded, for completeness: 10-scene monitor EPE 4.5207 / D1
30.64 at epoch 200; late window (150–199) EPE 4.559 ± 0.075, D1 30.58 ± 0.51;
gradient median 18.99, max 371.9; matching-path gradient present in **100.00 %**
of batches; per-scene 40-scene EPE mean 2.588, median 2.139, SD 1.879, min
1.221, max 11.403.

Initial weight SHA `540d81706f0d4850` — identical to the seed screen's 5b/seed2
initialisation, and the run reproduced that screen's epoch-10 and epoch-20
validation points exactly (15.698 / 86.58 and 12.336 / 78.76). Fifth run in a
row to reproduce its screen prefix; the deterministic protocol continues to hold.

---

## SEED RANGES

Individual seeds first, never hidden behind a mean.

**6 blocks** — EPE 2.2955 / 2.3982 / 2.3950 (seeds 0 / 1 / 2)

| | mean | SD | min | max | range |
| --- | ---: | ---: | ---: | ---: | ---: |
| EPE | 2.3629 | 0.0584 | 2.2955 | 2.3982 | **0.1027** |
| D1 | 17.137 | 1.396 | 15.605 | 18.336 | 2.731 |

**5 blocks** — EPE 2.6458 / 2.4730 / 2.6373 (seeds 0 / 1 / 2)

| | mean | SD | min | max | range |
| --- | ---: | ---: | ---: | ---: | ---: |
| EPE | 2.5853 | 0.0974 | 2.4730 | 2.6458 | **0.1728** |
| D1 | 20.079 | 1.660 | 18.206 | 21.371 | 3.165 |

**4 blocks** — one seed only: EPE 2.3387, D1 16.980. Not retrained here.

Difference of arm means: EPE **+0.2224 px** (5b worse), D1 **+2.94 pt** (5b
worse). The 5-block arm's own seed range is 1.68× the 6-block arm's on EPE and
1.16× on D1 — 5 blocks is the noisier arm as well as the worse one.

---

## OVERLAP

Computed in code (`comparison.json`), not by eye.

**EPE — the ranges do NOT overlap.**

```
min(5-block EPE) = 2.4730   >   max(6-block EPE) = 2.3982
margin = +0.0748 px
```

All **nine** 5b×6b seed pairings have the 5-block run worse on EPE. Every
5-block seed beats every 6-block seed in the wrong direction.

**D1 — the ranges DO overlap, by a hair.**

```
min(5-block D1) = 18.2062   <   max(6-block D1) = 18.3357
margin = −0.1295 pt
```

Eight of nine pairings have the 5-block run worse on D1. The single exception is
5b/seed1 (18.2062) versus 6b/seed1 (18.3357) — the 5-block run is better by
0.13 pt. That one inversion is what keeps the D1 ranges from separating.

This asymmetry matters and is not smoothed over: **the EPE claim is clean at
three seeds; the D1 claim is not.**

---

## PER-SCENE

Paired comparison — every model was evaluated on the same 40 scenes, so these
are per-scene differences, not independent samples.

| Pairing | scenes | 5b worse on EPE | mean ΔEPE | median ΔEPE | ΔEPE range | 5b worse on D1 | mean ΔD1 |
| ------- | -----: | --------------: | --------: | ----------: | ---------- | -------------: | -------: |
| 5b/s2 vs 6b/s2 | 40 | **32** | +0.233 | +0.132 | −0.327 … +1.136 | 35 | +3.05 |
| 5b/s2 vs 6b/s1 | 40 | **31** | +0.223 | +0.100 | −0.484 … +2.088 | 30 | +2.20 |
| 5b/s2 vs 6b/s0 | 40 | **37** | +0.329 | +0.204 | −0.937 … +1.739 | 38 | +4.83 |
| 5b/s1 vs 6b/s1 | 40 | 22 | +0.071 | +0.017 | −0.607 … +2.151 | 20 | **−0.07** |
| 5b/s0 vs 6b/s0 | 40 | **35** | +0.347 | +0.331 | −0.973 … +1.442 | 38 | +5.52 |

The new seed-2 run is consistently worse per scene (31–37 of 40 depending on the
6-block seed it is paired against), matching the seed-0 pattern (35–38 of 40).

**5b/seed1 is the outlier of the 5-block arm, not the rule.** Against 6b/seed1
it is worse in only 22 of 40 scenes on EPE, 20 of 40 on D1, and its mean D1
delta is *negative* — that pairing is close to a coin flip. Batch 1 rested its
"both 5-block draws above every 6-block draw" observation partly on this run;
with a third seed it is clear that 5b/seed1 was the best 5-block draw, not a
typical one.

Every pairing has scenes going the other way (ΔEPE min is negative in all five),
so the pooled result is a consistent tendency across scenes, not a handful of
outliers.

---

## STEREO

Established probes only, imported unchanged. `right` = minimum D1 penalty over
the three right-image corruptions; `map` = minimum over the two matching-map
corruptions.

| Epoch | right | map | entropy | disp. SD | init r(GT) |
| ----: | ----: | --: | ------: | -------: | ---------: |
| 10 | −2.651 | −4.088 | 1.973 | 8.78 | +0.518 |
| 20 | −0.026 | +3.568 | 1.865 | 10.91 | +0.709 |
| 50 | +42.048 | +32.965 | 1.793 | 16.95 | +0.806 |
| 100 | +61.287 | +52.094 | 1.820 | 17.00 | +0.882 |
| 150 | +74.089 | +60.943 | 1.837 | 17.79 | +0.886 |
| 200 | **+73.592** | **+60.096** | 1.844 | 18.05 | +0.884 |

STEREO-FUNCTIONAL at epoch 200, far above the frozen 20-point criterion, and
indistinguishable from the other three 200-epoch runs (right +73.6 to +76.4 across
6b/s1, 6b/s2, 5b/s1, 5b/s2). Binocular dependence emerges between epochs 20 and
50, as in every other run. Entropy 1.844 nats (floor 0.5), final disparity SD
18.05 px (floor 1.0) — no collapse.

**The 5-block accuracy penalty is not a stereo-functionality failure.** Both
architectures learn to use the right image to the same degree; they differ in
how accurately they regress disparity. The probes establish binocular
*dependence*, not correct disparity search (Phase 1 EXP-007 measured +89.7 D1 on
provably degenerate weights).

---

## INTERPRETATION

**MEASURED**

- 5b/seed2 at 200 epochs: 40-scene EPE 2.6373, D1 20.6594, RMSE 6.4439.
- 6-block EPE over seeds 0/1/2: 2.2955 / 2.3982 / 2.3950, range 0.1027 px.
- 5-block EPE over seeds 0/1/2: 2.6458 / 2.4730 / 2.6373, range 0.1728 px.
- `min(5b EPE) − max(6b EPE) = +0.0748 px`; `min(5b D1) − max(6b D1) = −0.1295 pt`.
- All 9 pairings 5b worse on EPE; 8 of 9 on D1.
- Per-scene: 5b/seed2 worse in 31–37 of 40 scenes against each 6-block seed.
- All four 200-epoch runs STEREO-FUNCTIONAL with right-image penalties +73.6…+76.4.
- Wall clock 4777.6 s; gates at 10 and 100 passed; matching gradient 100 %.

**DERIVED**

- Arm means differ by 0.2224 px EPE — 2.2× the 6-block seed range and 1.3× the
  5-block seed range.
- The 5-block arm's seed spread is 1.68× the 6-block arm's on EPE.
- 4b/seed0 (2.3387) lies **inside** the 6-block range; on the single seed
  available, 4 blocks does not show the 5-block penalty.

**INFERRED**

- The EPE separation is unlikely to be a seed artefact: the smallest 5-block
  draw exceeds the largest 6-block draw across three seeds each, and the
  per-scene direction is consistent. This is an inference from six runs on one
  dataset, not a test.
- 5b/seed1 appears to be a favourable draw for the 5-block arm rather than
  representative; batch 1's reading was based on it plus seed 0 and was
  correspondingly uncertain — which is why this run was commissioned.
- The mechanism is accuracy, not stereo capability: probe values are equal
  across arms while EPE and D1 are not.

**UNKNOWN**

- Whether the D1 ranges would separate with more seeds; at three each they do
  not, by 0.13 pt.
- Whether the ordering holds on any other dataset, recipe, schedule, training-set
  size, or at a different disparity-candidate count. Nothing here tests that.
- Whether 4 blocks behaves like 5 or like 6 — one seed exists, and it sits with
  the 6-block group, which the E3b/E3c "capacity-reducible" narrative did not
  predict. Not investigated here.
- Whether the penalty would survive pretraining or a larger training set. The
  known limitation of the frozen recipe (160 scenes from random init) still
  applies.

---

## VERDICT

```
5-BLOCK-PENALTY-AT-SEEDS-0-1-2
```

Chosen on the authoritative 40-scene EPE, where all three 5-block seeds are
worse than all three 6-block seeds, with a non-overlap margin of 0.0748 px and
9 of 9 pairings in the same direction, supported per-scene in 31–37 of 40 scenes.

**Stated with its exception:** the D1 ranges do *not* separate. 5b/seed1
(18.2062) is better than 6b/seed1 (18.3357) by 0.13 pt, so the D1 claim holds
for 8 of 9 pairings, not 9 of 9. Anyone quoting this verdict should quote that
alongside it.

This supports a capacity-related effect **for this tested setup**. It does not
identify a mechanism, and it does not make 6 blocks optimal — 4 blocks at its
single seed sits inside the 6-block range and was deliberately not retrained.

---

## CLAIM CEILING

> **This result concerns the tested StereoNet configuration, deterministic
> protocol, three seeds, and evaluated dataset. It does not establish a
> universal law that more refinement blocks always improve stereo depth.**

Also binding:

- Three seeds per arm is descriptive evidence, not a statistical test. No
  p-value is claimed and none is computable from six autocorrelation-free but
  tiny samples.
- The historical E3b/E3c materiality band stays **invalidated**. No band was
  reused and no new threshold was invented before or after seeing the result.
  Every number above is reported as measured.
- The 10-scene training monitor was recorded but is not the basis of any claim
  here — batch 1 measured that it overstates the 5-vs-6 gap by roughly 5×.
- The seed screen's 5b/seed2 run is a bit-identical prefix of this run, not an
  independent sample; the two must never be pooled.
- This result contradicts the E3b/E3c "capacity-reducible" reading, which held
  that refinement blocks could be removed for free. It does not retroactively
  invalidate those records — they measured a different thing at a different
  budget — but it does mean the reduction claim should not be carried forward
  for the 5-block case without this evidence attached.

---

## NEXT EXPERIMENT

Exactly one, and it is **not** executed here.

**Run 4 blocks at seeds 1 and 2, 200 epochs, deterministic protocol — two runs,
~2.6 GPU-hours.**

Why this one:

- It is the only remaining cell of the original 3×3 design, and it is now the
  *interesting* one. The 5-block penalty is established at three seeds; 4 blocks
  has one seed (2.3387) that sits **inside** the 6-block range. Either 4 blocks
  genuinely behaves unlike 5 blocks — which would make the effect non-monotonic
  in block count and rule out a simple capacity story — or 4b/seed0 was a
  favourable draw exactly as 5b/seed1 turned out to be.
- Batch 2 is precisely the cautionary case: a two-seed picture (5b seeds 0 and 1)
  looked like a 0.075 px effect and the third seed showed the arm's real centre
  was 0.22 px away. The 4-block arm currently rests on **one** seed.
- It completes the design at three seeds per arm for the same cost as batch 1,
  against a now-measured seed scale, and it is the last training this question
  needs.

The harness refuses these runs today. Authorising them means a new batch guard,
a new pre-registration and a new timestamped directory — a deliberate,
reviewable step, not a flag.

---

## HARD STOP

Stopping here, per §19 of the pre-registration. No 4-block run was launched, no
further 5-block or 6-block seed was added, no run was extended or rerun, Scene
Flow / dilation / candidate-count experiments were not started, and Stage E
stays CLOSED. The research lead decides what happens next.

---

## PROVENANCE

`HEAD` `58e8a19`; `phase-1-frozen^{commit}` `b4207e5`;
`git diff phase-1-frozen -- src scripts` **empty before and after**;
working tree `M .gitignore`, `?? phase2/` at both ends.
Preflight: **25/25 checks PASS** (`preflight.json`), including `seed_is_2`,
`block_count_is_5`, `dilations_are_1_2_4_8_1`, `batch_is_B2_only`,
`four_block_not_scheduled` and `six_block_not_scheduled`.

The batch-2 guard was tested against all three forbidden directions before
launch, not merely documented:

```
STOP: batch 2 authorises only arm B seed 2 (5 blocks, seed 2). Refusing arm C seed 1.
STOP: batch 2 authorises only arm B seed 2 (5 blocks, seed 2). Refusing arm A seed 2.
STOP: batch 2 authorises only arm B seed 2 (5 blocks, seed 2). Refusing arm B seed 0.
```

Phase 2 is not committed to git; no external immutability is claimed for any
Phase 2 record, including this one.

Stage A, Stage B, H2, E3b, E3c, O6, the deterministic block-count series, the
seed screen and batch 1 were read but never written. All output is confined to
this timestamped directory.
