# EXP-CORRESPONDENCE-GEOM-002 — RESULTS

**Synthetic fronto-parallel self-pair sweep.**

Record `phase2/diagnostics/correspondence_geom/20260911T115917Z/`.
Pre-registered in `PREREGISTRATION.md`, frozen before execution and **not
modified** (last write 12:02:50Z; execution began 12:07Z).

Inference only — **208 forward passes**, 13.7 s. No training, no fine-tuning, no
architecture change, no new checkpoint. Crops of a single source image; no fill,
no padding, no interpolation. Phase 1 untouched. GEOM-001 untouched. No
historical record modified.

```
VERDICT: SYNTHETIC-CORRESPONDENCE-NOT-DEMONSTRATED
         3 / 12 units pass
```

**Not rescued. Two protocol problems are reported in §6 and §7; neither is used
to alter the verdict.**

---

## 1. FREEZE VERIFICATION — 10/10 OK before the first model instantiation

`PREREGISTRATION.md` present (14 465 B) · `construction_spec.json`
`a9e8f250…` OK · `mask_spec.json` `b274ce03…` OK · `sha256sum -c frozen.sha256`
both **OK** · both digests present verbatim in the preregistration · no
`RESULTS.md` · `DESIGN_AUDIT.md` records the sign correction `a_R = a_L + t` ·
all four checkpoint sha256 verify · scenes `[1,2,3,4]` disjoint from the GEOM-001
focus set `{27,0,31,6}` · no protocol modification after preregistration.

Both digests were re-verified **over file bytes** inside the harness before any
model was built.

## 2. SANITY CHECKS

| check | result |
|---|---|
| all crops one shape | **`(272, 1136)`** only |
| volume shape | `(1, 32, 12, 17, 71)` in all 208 passes |
| no interpolation / no fill | integer slicing; every window asserted inside the source |
| left crop fixed per scene | 4 distinct hashes for 4 scenes — one per scene, invariant across all 13 conditions |
| mask | 145 152 px, identical every condition, axis, model, scene |
| `shift="none"` degeneracy | `max\|V_k − V_0\| = 0.0` — re-verified |
| weights unchanged | full parameter hash re-checked after every scene |
| training | **none** — no optimizer, no `.backward()`, no `.step()` |
| determinism | full block held |
| forward passes | **208 / 208** |

---

## 3. HORIZONTAL RESPONSE `m_h(t)` — candidate units

True disparity is `t/16` by construction.

| checkpoint | scene | t=0 | 16 | 32 | 48 | 64 | 80 | 96 | `Slope_h` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **TRUE** | | 0.000 | 1.000 | 2.000 | 3.000 | 4.000 | 5.000 | 6.000 | **1.000** |
| NEG | 000161_10 | **7.603** | 2.563 | 2.558 | 2.628 | 2.724 | 2.627 | 2.537 | −1.1522 |
| NEG | 000162_10 | **7.603** | 2.456 | 2.477 | 2.482 | 2.457 | 2.406 | 2.340 | −1.1969 |
| NEG | 000163_10 | **7.603** | 2.793 | 2.809 | 2.851 | 2.853 | 2.781 | 2.708 | −1.1114 |
| NEG | 000164_10 | **7.603** | 2.228 | 2.229 | 2.236 | 2.214 | 2.191 | 2.166 | −1.2468 |
| seed0 | 000161_10 | 1.703 | 2.325 | 4.047 | 5.808 | 7.225 | 8.572 | **7.693** | +1.2088 |
| seed0 | 000162_10 | 1.604 | 2.331 | 4.034 | 5.811 | 7.287 | 8.584 | **7.810** | +1.2427 |
| seed0 | 000163_10 | 1.598 | 2.323 | 4.039 | 5.786 | 7.256 | 8.576 | **7.937** | +1.2498 |
| seed0 | 000164_10 | 1.624 | 2.332 | 4.035 | 5.807 | 7.243 | 8.504 | **7.448** | +1.2076 |
| seed1 | 000161_10 | 1.619 | 1.972 | 3.081 | 4.292 | 5.416 | 6.078 | 6.699 | +0.8709 |
| seed1 | 000162_10 | 1.627 | 1.965 | 3.066 | 4.276 | 5.411 | 6.064 | 6.646 | +0.8638 |
| seed1 | 000163_10 | 1.632 | 1.966 | 3.056 | 4.278 | 5.392 | 6.076 | 6.686 | +0.8649 |
| seed1 | 000164_10 | 1.655 | 1.962 | 3.058 | 4.260 | 5.383 | 6.051 | 6.619 | +0.8528 |
| seed2 | 000161_10 | 1.460 | 2.042 | 3.451 | 4.832 | 6.208 | 6.778 | **6.691** | +1.0071 |
| seed2 | 000162_10 | 1.454 | 2.038 | 3.456 | 4.840 | 6.242 | 6.733 | **6.624** | +1.0035 |
| seed2 | 000163_10 | 1.416 | 2.041 | 3.454 | 4.836 | 6.226 | 6.928 | **6.838** | +1.0362 |
| seed2 | 000164_10 | 1.429 | 2.045 | 3.445 | 4.831 | 6.190 | 6.775 | **6.535** | +1.0028 |

Bold `t = 96` entries are where monotonicity breaks.

## 4. VERTICAL RESPONSE `m_v(t)` — no horizontal correspondence exists

| checkpoint | scene | t=0 | 16 | 32 | 48 | 64 | 80 | 96 | `Slope_v` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NEG | 000161_10 | 7.603 | 2.433 | 2.468 | 2.440 | 2.521 | 2.954 | 3.319 | −1.1012 |
| NEG | 000162_10 | 7.603 | 2.217 | 2.258 | 2.439 | 2.757 | 3.358 | 3.743 | −1.0477 |
| NEG | 000163_10 | 7.603 | 2.940 | 3.010 | 3.000 | 2.958 | 3.176 | 3.306 | −1.0347 |
| NEG | 000164_10 | 7.603 | 2.107 | 2.171 | 2.290 | 2.363 | 2.734 | 3.313 | −1.1357 |
| seed0 | 000161_10 | 1.703 | 7.201 | 6.544 | 6.941 | 7.551 | 7.164 | 7.559 | **+1.2827** |
| seed0 | 000162_10 | 1.604 | 7.261 | 6.978 | 7.669 | 6.867 | 7.629 | 7.667 | **+1.3425** |
| seed0 | 000163_10 | 1.598 | 7.720 | 7.497 | 7.231 | 7.418 | 7.746 | 7.838 | **+1.3877** |
| seed0 | 000164_10 | 1.624 | 7.612 | 6.766 | 7.604 | 7.644 | 7.659 | 7.866 | **+1.3836** |
| seed1 | 000161_10 | 1.619 | 5.546 | 5.005 | 4.886 | 5.422 | 5.506 | 5.456 | +0.8589 |
| seed1 | 000162_10 | 1.627 | 5.339 | 5.073 | 5.400 | 5.103 | 5.302 | 5.425 | +0.8461 |
| seed1 | 000163_10 | 1.632 | 5.716 | 5.656 | 4.568 | 4.985 | 5.445 | 5.914 | **+0.8694** |
| seed1 | 000164_10 | 1.655 | 5.330 | 4.911 | 4.991 | 5.367 | 4.914 | 5.323 | +0.8060 |
| seed2 | 000161_10 | 1.460 | 6.632 | 6.157 | 6.301 | 6.118 | 6.735 | 6.430 | **+1.1419** |
| seed2 | 000162_10 | 1.454 | 6.169 | 6.386 | 6.235 | 6.684 | 6.561 | 6.465 | **+1.1588** |
| seed2 | 000163_10 | 1.416 | 6.915 | 6.409 | 6.344 | 6.731 | 6.643 | 6.941 | **+1.2177** |
| seed2 | 000164_10 | 1.429 | 6.491 | 6.395 | 5.606 | 6.568 | 6.597 | 6.579 | **+1.1518** |

Bold `Slope_v` entries **exceed** the same unit's `Slope_h`.

---

## 5. PER-UNIT DECISION TABLE

| checkpoint | scene | `Slope_h` | `Slope_v` | `Slope_h`(NEG) | C1 | C2 | C3 | C4 | unit |
|---|---|---:|---:|---:|:--:|:--:|:--:|:--:|:--:|
| seed0 | 000161_10 | +1.2088 | +1.2827 | −1.1522 | Y | **n** | **n** | Y | **FAIL** |
| seed0 | 000162_10 | +1.2427 | +1.3425 | −1.1969 | Y | **n** | **n** | Y | **FAIL** |
| seed0 | 000163_10 | +1.2498 | +1.3877 | −1.1114 | Y | **n** | **n** | Y | **FAIL** |
| seed0 | 000164_10 | +1.2076 | +1.3836 | −1.2468 | Y | **n** | **n** | Y | **FAIL** |
| seed1 | 000161_10 | +0.8709 | +0.8589 | −1.1522 | Y | Y | Y | Y | **PASS** |
| seed1 | 000162_10 | +0.8638 | +0.8461 | −1.1969 | Y | Y | Y | Y | **PASS** |
| seed1 | 000163_10 | +0.8649 | +0.8694 | −1.1114 | Y | Y | **n** | Y | **FAIL** |
| seed1 | 000164_10 | +0.8528 | +0.8060 | −1.2468 | Y | Y | Y | Y | **PASS** |
| seed2 | 000161_10 | +1.0071 | +1.1419 | −1.1522 | Y | **n** | **n** | Y | **FAIL** |
| seed2 | 000162_10 | +1.0035 | +1.1588 | −1.1969 | Y | **n** | **n** | Y | **FAIL** |
| seed2 | 000163_10 | +1.0362 | +1.2177 | −1.1114 | Y | **n** | **n** | Y | **FAIL** |
| seed2 | 000164_10 | +1.0028 | +1.1518 | −1.2468 | Y | **n** | **n** | Y | **FAIL** |

```
C1 (sign)              12 / 12
C2 (6-level monotone)   4 / 12      breaks at t = 96 in all 8 failures
C3 (axis specificity)   3 / 12      <- the substantive failure
C4 (vs search-free)    12 / 12      but see §7
PASS                    3 / 12
```

**Three units pass. The global rule requires 12/12. The decision is FAIL.**
Per the standing instruction: 11/12 would also be FAIL, and 3/12 certainly is.

---

## 6. THE SUBSTANTIVE FINDING — the response is NOT axis-specific

**MEASURED.** `Slope_v ≥ Slope_h` at **9 of 12** units, by ratios of 1.005 to
1.175. The vertical arm — in which **no horizontal correspondence exists**, since
the cost volume compares within a row — produces a response as large as, and
usually larger than, the horizontal arm.

**But the two arms have different *shapes*, and the statistic cannot see it.**

- **Horizontal** rises progressively: e.g. seed1/000161 — 1.62, 1.97, 3.08, 4.29,
  5.42, 6.08, 6.70. A graded ramp.
- **Vertical** is a step: 1.62 → 5.55 at `t = 16`, then flat and non-monotone
  (5.01, 4.89, 5.42, 5.51, 5.46).

**DERIVED — a weakness in the frozen statistic, exposed by execution.**
`Slope_axis` regresses `m(t) − m(0)` on `t/16` through the origin. A **step** at
`t > 0` and a **ramp** in `t` produce similar slopes. `Slope` therefore conflates
"responds to any displacement" with "tracks the displacement", which is precisely
the distinction `C3` was meant to test.

**This is reported, not repaired.** `C3` failed as preregistered at 9/12 units.
The shape difference is **descriptive** and is **not** used to alter the verdict.
Promoting it would be exactly the post-hoc substitution this campaign has
repeatedly refused.

### Monotonicity (`C2`) — the turnover at `t = 96`

All 8 `C2` failures break at the **same place: `t = 96`**, i.e. candidate 6 — the
largest imposed disparity. Seeds 0 and 2 rise through `t = 80` then fall; seed 1
remains monotone throughout. **MEASURED**, not interpreted further.

---

## 7. PROTOCOL PROBLEMS — reported, not repaired

### 7.1 The `t = 0` wiring check FAILED, and the preregistration's derivation was wrong

`PREREGISTRATION.md` §5 predicted `NEG_shift_none` at `t = 0` would return
`disparity_initial = 5.5` **exactly**.

**MEASURED:** it returns **7.603198** — identical to the last digit on all four
scenes (spread `0.00e+00`), with range 1.888 … 9.184 across pixels.

**The identical value on four different images shows this is image-independent —
a weights-only artefact, not an implementation fault.** The derivation was wrong:
I argued that with `V = 0` the zero padding equals the signal, so no boundary
effect arises. **That holds only for the first convolution.** After layer 1 the
signal is a non-zero constant (the bias), so from layer 2 onward the zero padding
differs from the signal and the D-axis boundary effect reappears. `Agg(0)` is
therefore constant in the interior but not at the candidate boundaries, so
standardisation does not return 0 and the soft-argmin is not 5.5.

### 7.2 A declared HARD STOP was not enforced — a real deviation

`PREREGISTRATION.md` §11 lists the `t = 0` wiring check in the controls table
under "**HARD STOP** on any failure above". **The harness recorded the check but
did not halt on it.** The run should have stopped before the positive
checkpoints.

**Recorded as a protocol deviation.** The verdict does not depend on it — `C1`–`C4`
do not use the wiring check, and the outcome is FAIL either way — but the run is
**procedurally compromised** and must not be presented as a clean preregistered
execution. Root cause: the hard stop was written against the wrong trigger ("a
deviation means a bug"), when in fact the deviation meant the *prediction* was
wrong.

### 7.3 `C4` passed 12/12 but is uninformative

**DERIVED.** `Slope` subtracts `m(0)`. For `NEG_shift_none`, `t = 0` is the
**degenerate** case — left and right crops are byte-identical, so `V ≡ 0` exactly
— giving the anomalous 7.603, while every `t > 0` gives ≈2.2–2.9. Its
`Slope_h ≈ −1.1` is therefore **an artefact of that single anomalous baseline
point**, not a measure of its displacement response.

`C4` compared the positives against that artefact and passed trivially at 12/12.
**It carried no information.** Reported so the 12/12 is not mistaken for support.

---

## 8. FAILURE CLASSIFICATION — explicit, not collapsed

| # | class | count | detail |
|---|---|---:|---|
| 1 | correspondence-positive | **3** | seed1 on 000161, 000162, 000164 — all four conditions hold |
| 2 | flat response | 0 | `C1` passed 12/12; the model is not monocular-on-left |
| 3 | **non-specific response** | **9** (`C3` fails) | `Slope_v ≥ Slope_h`; the vertical arm responds as strongly — §6 |
| 4 | search-free-equivalent | 0 | `C4` passed 12/12, but uninformative — §7.3 |
| 5 | non-monotone / mixed | **8** (`C2` fails) | all break at `t = 96` |
| 6 | **protocol failure** | **yes** | §7.1 wrong derivation; §7.2 unenforced hard stop |

By first failure in rule order: non-monotone 8, non-specific 1, passing 3.

**This is not "stereo failed."** The response is present, correctly signed at
12/12, and graded in `t` on the horizontal arm — but the preregistered test could
not establish that it is specific to the epipolar axis, and the vertical control
responds at least as strongly.

---

## 9. CLAIM CEILING

```
SYNTHETIC-CORRESPONDENCE-NOT-DEMONSTRATED
```

**It is NOT concluded that correspondence is absent.** The input is
out-of-distribution (constant disparity is never seen in training), so a failure
is ambiguous between "no functioning correspondence machinery" and "the machinery
does not generalise to this input" — as declared in `PREREGISTRATION.md` §9
before execution.

Not claimed under any reading: real-scene correspondence, disparity correctness,
sub-candidate precision, genuine disparity search, learned-versus-architectural
origin, generalisation, or anything about `disparity_final`. No EPE, D1 or RMSE
was computed. Refinement never invoked.

---

## 10. PROVENANCE

Executed as frozen except for the unenforced hard stop of §7.2. No checkpoint,
scene, crop, mask, `t` level, construction, axis definition, statistic, control,
decision rule or claim ceiling was changed. `PREREGISTRATION.md` unmodified. No
training. No historical record modified; GEOM-001 untouched and its verdict
unchanged.

**Protocol deviations: one — §7.2.** Plus one erroneous derivation in the frozen
preregistration, §7.1.

`HEAD 58e8a19`; `phase-1-frozen b4207e5`; `phase2/` untracked, so no
cryptographic versioning is claimed.

## 11. FILES

`PREREGISTRATION.md`, `DESIGN_AUDIT.md`, `EXECUTION_PROMPT.md`, `RESULTS.md`
(this file), `results.json`, `run.log`, `ENVIRONMENT.txt`, `RELATED_RUNS.md`,
`construction_spec.json`, `mask_spec.json`, `frozen.sha256`, `freeze_geom2.py`,
`run_geom2.py`.
