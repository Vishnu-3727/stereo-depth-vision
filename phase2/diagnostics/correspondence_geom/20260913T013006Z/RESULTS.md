# RESULTS — EXP-CORRESPONDENCE-GEOM-002

## Pairing-Swap Interaction Diagnostic, on the corrected domain

```
STATUS  :  G-STOP AT STAGE 1
VERDICT :  STATISTIC-ARCHITECTURALLY-AVAILABLE
H*      :  UNDEFINED  (max(h_rand) == 6)

TRAINED AGGREGATION:  NEVER APPLIED.  Stage 2 was not run.
```

The preregistered random-aggregation gate fired its **terminal** outcome. A
randomly initialised aggregation, with **no training of any kind**, reproduces
the statistic **perfectly** — `h = 6`, every `tau` correct — in **31 of 576**
gate units, drawn from **9 of the 32 seeds**, on **all three** feature
extractors and **all six** scene pairs.

Under the frozen rule (`PREREGISTRATION.md` A.7), `max(h_rand) == 6` means the
statistic is reachable without training, `H*` is undefined, the trained
checkpoints are not loaded, and Stage 2 does not run.

**This result says nothing about correspondence in either direction.**

Record `phase2/diagnostics/correspondence_geom/20260913T013006Z/`.
Frozen and executed 2026-09-13 UTC.

Tags: **MEASURED** (this run) · **DERIVED** (algebra/source) · **INFERRED** ·
**UNKNOWN**.

---

## 1. FREEZE INTEGRITY

| artifact | sha256 |
|---|---|
| `PREREGISTRATION.md` | `5164c29689eef76905cea1b0e9d05d78d63ff4f4d6b1a3f2e2770ccbf1ed4835` |
| `preregistration.json` | `f6d28d42a5a3ce29a6747d1636ef1ff9c581d44a854e8c916684e594f79a6082` |
| `design.json` | `d28ad74a4f0155ee83d45157b5a80183af07a2ec12acf52194092f8697b21d26` |
| `run_geom2.py` | `fe8c69d7d4fd244e1e7a71c690619b8e044376e6bad463e9004cfc9c73ad7559` |
| `freeze_geom2.py` | `954421381f8f4c464c3978ef109672150d1e50a03b9913d80db5b04fcf44cf7d` |

All five verified over **file bytes** before the first model was instantiated,
and again at the end of Stage 1.

**63 historical records** verified before and after every stage — `0` mismatches
every time (**HS8 PASSED**). This includes the complete GEOM-001 record, whose
own two manifests were independently re-verified: `frozen.sha256` **5/5**,
`historical.sha256` **36/36**. **No historical record, and no Phase-1 file, was
modified.** `git diff phase-1-frozen` over `src/ data/ scripts/ tests/
reference/` is empty.

`gate.json`, `trained.json`, `results.json` and `RESULTS.md` were confirmed
absent before execution began.

---

## 2. THE GEOMETRY HELD — this is what GEOM-001 could not reach

GEOM-001 halted on the **first cost volume it built**. Every one of those checks
now passes, on the corrected domain, over the full run.

### 2.1 HS1-A — pixel construction (**MEASURED**, 216/216)

`right[:, :CROP_W-t] == left[:, t:]` asserted element-wise from the actual
arrays, at every `tau`, for both frames of every pair, for all three extractors.
**216 checks, all exact.**

### 2.2 HS1-B — the feature translation law (**MEASURED**, 216/216, tolerance 0)

```
Rf[w] == Lf[w + tau]     bit-exact,  w in [15, 55 - tau]
```

**216 checks · 4 406 400 feature values · every one bit-identical.** Tolerance
was exactly zero, not a threshold.

### 2.3 HS1-C — the matched anchor (**MEASURED**, 216/216, tolerance 0)

```
max |V[:, tau, :, w]| == 0.0     exactly,  w in [15, 44],  cells AA and BB
```

**216 checks · 3 525 120 cost-volume cells · maximum observed value `0.0`.**

This is the bound that fired in GEOM-001 at `7 233 118.0`. The difference is
entirely the left edge of the band: GEOM-001 started at `w = 0`, inside the
measured 15-cell extractor halo; this record starts at `w = 15`. The domain
audit's prediction is reproduced exactly.

### 2.4 HS1-D — the domain itself (**DERIVED**, checked before any model loaded)

```
extractor clean      w in [15, 55]      = [HALO, FW-1-HALO]
cost volume clean    w in [15, 44]      = [.., 55 - (D-1)]   tau-INDEPENDENT
fill-free bound      w <= 59            (not binding)
statistic supported  w in [20, 39]      = the 5-cell aggregation shrink
                     20 columns >= D = 12
pixel band           x in [325, 632]    = ceil(1135*20/70) .. floor(1135*39/70)
MASK                 y in [64,208) AND x in [325,633)  =  44 352 px
```

**HS9 PASSED** — every one of the 13 824 readouts was masked to exactly
**44 352** pixels.

### 2.5 HS4 — the 2x2 cancellation on real features (**MEASURED**, 108 checks)

```
max relative residual   1.8088509428138105e-07
median                  1.1943405141662960e-07
frozen tolerance        1e-05
```

The maximum agrees with the domain audit's independently recorded
`1.8089e-07` to five significant figures. Pure float32 round-off, ~55x inside
the bound. Asserted at the **raw cost-volume level only**, as A.13 requires; no
claim is made about the cancellation after the nonlinear z-score.

### 2.6 The remaining hard stops

| id | outcome |
|---|---|
| **HS2** | **PASSED** — 108 checks; each frame exactly once as a left and once as a right |
| **HS3** | **PASSED** — 108 checks; a single `tau` across all four cells, always |
| **HS5** | **PASSED** — three checkpoints, sha256 match; `num_disparities = 12`; `shift = left`; readout parameters `0`; aggregation parameters `111 585` (trained and random alike) |
| **HS6** | **NOT RUN** — the W4 restart test requires a completed Stage 2 reference, which does not exist. **Determinism is therefore not claimed for this run.** The frozen determinism block *was* active (§8) |
| **HS7** | **PASSED** — every trained aggregation was set to `None` immediately after its checkpoint load, before any forward pass. Stage 2 lives in a separate process that refuses to start without a `G-PASS`; the gate says `G-STOP`, so it cannot run |
| **HS8** | **PASSED** — 63 records byte-identical before and after |
| **HS9** | **PASSED** — 44 352 px, 13 824 times |
| **HS10** | **PASSED** — 3 456 finiteness checks, no NaN, no Inf |
| **HS11** | **PASSED** — no optimizer imported or constructed, no `.backward()`, no `.step()`, refinement never invoked |

`hard_stop.json`: **`any_fired: false`**. No hard stop fired in any stage.
**G-STOP is not a hard stop** — it is the preregistered terminal outcome of the
gate.

---

## 3. THE GATE RESULT

**MEASURED.** 3 feature extractors x 6 scene pairs x 32 seeds = **576 units**,
**13 824** readout passes, **241.4 s** wall clock. Aggregation weights randomised;
everything else — extractor, cost volume, candidate count, readout, geometry,
pairing, mask — frozen.

```
h_rand histogram        0 : 414
                        1 :  67
                        2 :  19
                        3 :  13
                        4 :  10
                        5 :  22
                        6 :  31        <-- G-STOP
max(h_rand)         =  6
min(h_rand)         =  0
mean(h_rand)        =  0.8333
per-tau hit rate    =  0.13889
chance rate (1/D)   =  0.08333
tie events          =  0
```

### 3.1 The `h = 6` units are not a corner case

| breakdown of the 31 units with `h = 6` | |
|---|---|
| seeds involved | **9 of 32** — 4, 11, 12, 15, 17, 20, 22, 27, 30 |
| per-seed counts | 27:**10**, 22:**6**, 17:**4**, 30:**4**, 15:**2**, 20:**2**, 4:1, 11:1, 12:1 |
| extractors | all three — seed0:6, seed1:15, seed2:10 |
| scene pairs | all six — 4, 7, 3, 7, 5, 5 |
| seeds reaching `h = 6` at least once | **9 of 32** |
| seeds reaching `h = 6` on all 18 cells | **0 of 32** |

No single random seed tracks every scene pair, but nine of them track *some*
pair perfectly, on every extractor, across every pair. This is not one unlucky
draw.

### 3.2 Why — the mechanism, and it is an exact identity

**MEASURED.** Classify each unit by whether `k_hat_tau - tau` is *constant*
across the six `tau`:

```
offset-constant units        153 of 576   (26.6%)
  offset  -1 :   7
  offset   0 :  31     <-- these are exactly the h = 6 units
  offset  +1 :  60
  offset  +2 :  33
  offset  +3 :  20
  offset  +4 :   2
non-constant units           423 of 576
  h histogram   0:292  1:67  2:19  3:13  4:10  5:22   (never 6)
```

**The identity is exact: `h = 6` if and only if the offset is constant AND zero.**
`31` on both sides, no exceptions. The constant-offset units carry `h = 0` when
the offset is non-zero and `h = 6` when it is zero — nothing in between.

**INFERRED (mechanism, one step beyond the measurement).** A randomly weighted
aggregation frequently produces a response whose `argmin` is an *affine* function
of the displacement, `k_hat = tau + c`. What varies between random draws is the
**offset `c`**, not whether tracking happens at all. The offsets cluster at
`+1` (60 units), with `0` the second largest bin. Tracking the displacement is
therefore **cheap** under this architecture and this statistic; landing the
offset on zero is what `h = 6` additionally requires, and 31 random draws did it.

### 3.3 The argmins are massively non-independent

**DERIVED.** If the six `argmin`s were independent and uniform over the 12
candidates, `P(h = 6) = (1/12)^6 = 3.35e-07`, giving an expected
**0.000193** units out of 576.

```
expected under independence :      0.0002
observed                    :     31
```

Five orders of magnitude. The per-`tau` hit rate (`0.1389`) is only modestly
above chance (`0.0833`), so the excess lives **entirely** in the *correlation
across `tau`*, exactly as §3.2 describes. A statistic built on `argmin`-tracking
cannot treat its six levels as six independent trials under this architecture.

---

## 4. WHAT THIS DOES AND DOES NOT ESTABLISH

### Establishes (**MEASURED**)

1. The corrected domain is correct. Every geometric premise GEOM-001 assumed and
   could not verify is now verified on real features, at tolerance exactly zero:
   the pixel construction, the translation law on `w in [15, 55-tau]`, and the
   anchor `V(tau) = 0` on `w in [15, 44]` over 3 525 120 cells.
2. The 2x2 cancellation holds on real features at `1.809e-07` relative —
   an independent reproduction of the domain audit's figure.
3. **Under this architecture, this mask and this statistic, a completely
   untrained aggregation attains the maximum score.** `31 / 576` units, `9 / 32`
   seeds, all three extractors, all six pairs.
4. The mechanism behind (3) is constant-offset tracking: `153 / 576` random units
   produce an `argmin` that is affine in `tau`, and `h = 6` is exactly the
   subset whose offset is zero.

### Does NOT establish

```
Level D -- geometric correspondence :  NOT-DEMONSTRATED    (UNCHANGED)

  evidence FOR correspondence      :  NONE produced
  evidence AGAINST correspondence  :  NONE produced
  trained aggregation measured     :  NEVER -- 0 trained readout passes
```

The claim ceiling in `PREREGISTRATION.md` A.16 is **not engaged**: the trained
arm was never run, so neither branch of it applies. Specifically **not** claimed:
stereo correspondence, disparity search, disparity on natural scenes, metric
depth, causal uniqueness, architectural necessity, generalisation — and **not**
the absence of any of them.

### What the G-STOP means, stated precisely

A `G-STOP` is **not** a finding that the model lacks correspondence. It is a
finding that **this statistic cannot distinguish** a trained aggregation from an
untrained one, because the untrained one already saturates it. Had Stage 2 run
and returned `h = 6`, that number would have been uninterpretable — which is the
entire reason the gate is placed before the trained arm and not after.

---

## 5. WHAT IT COST TO LEARN THIS

**MEASURED.** 13 824 readout passes, 241 s of GPU time, one process, no training,
no trained weight ever applied. The gate did its job at 1/3 of the planned
compute and stopped the experiment before any trained number existed to be
tempted by.

**INFERRED.** Had the gate been placed *after* the trained arm — or omitted, as
in two earlier records in this campaign — a trained `h = 6` would have been read
as a positive result. Thirty-one random draws produce exactly that number.

---

## 6. CONTROLS NOT RUN, AND WHAT THAT COSTS

`NEG_shift_none` — the search-free control, a `shift="none"` checkpoint whose
twelve disparity slices are provably identical (EXP-010) — was **excluded** by
the authorisation (A.8: *"load exactly these frozen H2 checkpoints"*).

Consequence: this record carries **one** null, the random-aggregation gate, and
**no architecture-level null**. That null was sufficient to terminate the
experiment, so the exclusion did not change this outcome — but it would have
mattered had the gate passed. Declared in `PREREGISTRATION.md` B.4 and in
`results.json` under `controls_not_run`.

The **vertical control** was also never reached: it lives in Stage 2, which did
not run. It remains what the design says it is — a contaminated null with no
halo-clean region, usable only as a downgrade trigger.

---

## 7. DETERMINISM

The frozen determinism block was set and active throughout
(`environment.json`): `use_deterministic_algorithms(True)`,
`cudnn.deterministic = True`, `cudnn.benchmark = False`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `no_grad`, fp32, CUDA, RTX 4060 Laptop,
torch 2.7.0+cu128, numpy 2.5.1, Python 3.12.9.

**W4 was NOT run**, because the process-restart comparison requires a completed
Stage 2 reference and none exists. **Determinism is therefore not claimed for
this run**, only that the controls were active.

---

## 8. IS A SUCCESSOR JUSTIFIED?

**Not decided here, and deliberately not designed here.** Designing one in this
record would be repairing the experiment and continuing, which A.17 forbids
("any such change requires a new experiment and new preregistration").

What this run changes, factually, for whoever writes that authorisation:

- The **geometry is settled.** The corrected domain is verified on real features
  at tolerance zero. A successor does not need to re-litigate the mask.
- The **statistic is not diagnostic.** `argmin`-at-`tau` over six levels is
  attained by untrained weights in 5.4% of draws, and the six levels are not
  independent — the failure is structural, not a matter of the threshold.
- The measured obstacle is **specific**: random aggregations produce
  constant-offset tracking, and the offset lands on zero often enough to saturate
  `h`. Any successor statistic must be insensitive to that, and must be
  calibrated against the same random-aggregation population **at the same
  budget** before any trained weight is loaded.
- No number from this record may be reused as a null for a different statistic.

---

## 9. FILES IN THIS RECORD

| file | role |
|---|---|
| `PREREGISTRATION.md` | the frozen protocol (Part A verbatim) + the declared resolutions (Part B) |
| `preregistration.json` | machine-readable protocol + resolutions |
| `design.json` | the domain derivation and frozen specification |
| `run_geom2.py` | the harness (frozen before execution) |
| `freeze_geom2.py` | the freeze tool |
| `frozen.sha256`, `historical.sha256` | integrity manifests (5 + 63) |
| `prereq.json` | Stage 0: prerequisites, read-only, no model instantiated |
| `environment.json` | execution environment and determinism block |
| `checkpoint_hashes.json` | the three checkpoints + the control not run |
| `gate.json`, `gate_raw.json` | Stage 1: all 576 units, all 3 456 candidate-response vectors |
| `results.json` | the machine-readable outcome |
| `metrics.csv` | the headline numbers |
| `per_scene.csv` | one row per unit |
| `per_tau.csv` | one row per unit per `tau`: `k_hat`, `I_min`, `I_second`, margin, tie |
| `hard_stop.json` | `any_fired: false` |
| `run_stage0.log`, `run_stage1.log`, `run_stagereport.log`, `stage1_console.log` | execution logs |
| `RELATED_RUNS.md` | provenance |

**No `trained.json`. No `w4_inline.json`. No `w4_restart.json`.**
Stage 2 was never entered and cannot be: the gate says `G-STOP`.
