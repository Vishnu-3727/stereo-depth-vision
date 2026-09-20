# EXP-CORRESPONDENCE-GEOM-A — RESULTS

**Signed odd-part translation response on held-out scenes, with a
matched-magnitude vertical comparison and the search-free `shift="none"`
model as artifact baseline.**

Record `phase2/diagnostics/correspondence_geom/20260914T072255Z/`.
Pre-registered in `PREREGISTRATION.md`, frozen before execution.
`PREREGISTRATION.md`, `translation_spec.json`, `mask_spec.json`,
`scene_selection.json` and `frozen.sha256` were **not modified** (frozen
digests re-verified over file bytes before execution and re-verified inside
the harness at HS-FREEZE before any model was instantiated; no `RESULTS.md`
existed before execution).

Inference only — **72 forward passes**, 6.69 s wall clock. No training, no
fine-tuning, no optimizer, no `.backward()`, no `.step()`, no weight change,
no architecture change, no new checkpoint. Crop-based translation: no fill,
no padding, no interpolation. Phase 1 untouched. No historical record
modified. No hard stop fired.

```
VERDICT: PASS
         6 / 6 positive units pass (C1∧C2∧C3∧C4 at every unit)
```

**This is a consistency-positive outcome under the preregistered decision
rule, and nothing more.** The claim ceiling of `PREREGISTRATION.md`
§0.1/§12 applies unchanged and is repeated verbatim in §7. No language
anywhere in this file exceeds it.

---

## 1. FREEZE VERIFICATION — 9/9 OK, before the first model instantiation

| # | check | result |
|---|---|---|
| 1 | `PREREGISTRATION.md` exists, unmodified | OK, frozen-not-executed record; digests recorded in it re-verified |
| 2 | `translation_spec.json` sha256 (file bytes) | `7d0f0e10910fcd72f76f130665a59bbe238d146d110c993d843ef3737fc4267c` OK |
| 3 | `mask_spec.json` sha256 (file bytes) | `61e81f72e6fe553f52abfaad97ef1cb56cf6a30f8194f235ce9d4dde1ac03a2e` OK |
| 4 | `scene_selection.json` sha256 (file bytes) | `edbe049119a1f0a9e0d6aeb08aee8785a711ffeb48a6b511560b6a2066980789` OK |
| 5–7 | `sha256sum -c frozen.sha256` equivalent | all three **OK** (certutil per-file hashes match) |
| 6 | no `RESULTS.md` / `results.json` before execution | OK — both absent; harness HS-FREEZE asserts the same |
| 7 | frozen checkpoint hashes (HS-FREEZE, at load) | all four OK (see §2.9) |
| 8 | 6-unit structure | 3 trained weight sets × 2 frozen scenes = 6 positive units, 2 control cells |
| 9 | no protocol modification after preregistration | OK — run completed with `protocol_deviations: none`; no hard stop fired |

All three spec digests were additionally re-verified **over file bytes**
inside the harness before any model was built, and each digest was asserted
present in `PREREGISTRATION.md`.

---

## 2. SANITY CHECKS — 12/12

| # | check | result |
|---|---|---|
| 1 | all crops identical | **one shape only: `(304, 1168)`** |
| 2 | no interpolation | integer slicing only |
| 3 | no fill / padding | every crop window asserted inside the image |
| 4 | frozen mask used | counts **41 068 / 24 692**, pooled 65 760 — exact match to `mask_spec.json` |
| 5 | same code path both axes | axis is a function argument to `right_origin`; no branch in the forward path |
| 6 | ± directions symmetric | `Δ` set verified symmetric and all multiples of 16; `max\|Δ\|` == crop margin 32 |
| 7 | no training | no optimizer imported or constructed; no `.backward()`, no `.step()` |
| 8 | no weight change | full parameter sha256 re-hashed after every scene, unchanged throughout |
| 9 | checkpoint hashes unchanged | verified at load, all four match (`581d62e6…`, `58117f26…`, `245a3f5b…`, `d40d805b…`) |
| 10 | deterministic execution | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `no_grad`, `eval`, fp32 |
| 11 | historical records untouched | no file written outside this record; predecessor `20260914T070551Z` unmodified |
| 12 | forward passes accounted | **72 / 72 expected** (4 models × 2 scenes × 9 conditions) |

Cost volume shape was `(1, 32, 12, 19, 73)` in every one of the 72 passes —
consistent with the 1168 × 304 crop at stride 16. HS-REPRESENTABILITY passed
from the specs in this record alone (the run reached completion, so no
representability halt fired).

---

## 3. PER-UNIT DECISION TABLE

`S` in candidates per image pixel. Anchor `−1/16 = −0.0625`, reported
**descriptively only** (never a pass criterion). `S_h(NEG)` is the
search-free baseline on the same scene.

| checkpoint | scene | S_h | S_v | S_h(NEG) | Odd_h(16) | Odd_h(32) | C1 | C2 | C3 | C4 | unit |
|---|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|:--:|:--:|
| seed0 | 000165_10 | −0.101703 | −0.004200 | −0.003503 | −1.8016 | −3.1673 | Y | Y | Y | Y | **PASS** |
| seed0 | 000169_10 | −0.102970 | −0.001797 | +0.002482 | −1.7862 | −3.2257 | Y | Y | Y | Y | **PASS** |
| seed1 | 000165_10 | −0.066013 | −0.002573 | −0.003503 | −1.2075 | −2.0368 | Y | Y | Y | Y | **PASS** |
| seed1 | 000169_10 | −0.068890 | −0.009030 | +0.002482 | −1.2289 | −2.1411 | Y | Y | Y | Y | **PASS** |
| seed2 | 000165_10 | −0.081630 | −0.004963 | −0.003503 | −1.5020 | −2.5142 | Y | Y | Y | Y | **PASS** |
| seed2 | 000169_10 | −0.085254 | −0.012312 | +0.002482 | −1.4318 | −2.6943 | Y | Y | Y | Y | **PASS** |

```
C1 (sign)                6 / 6
C2 (monotonicity)        6 / 6
C3 (axis specificity)    6 / 6
C4 (vs search-free)      6 / 6
PASS                     6 / 6
```

Descriptive anchor comparison only: `F_h = S_h / (−1/16)` is 1.63, 1.65
(seed0), 1.06, 1.10 (seed1), 1.31, 1.36 (seed2). `F_h` is **not** a pass
criterion and no evaluation of magnitude correctness is made; the anchor is
reported descriptively per the preregistration.

---

## 4. HORIZONTAL RESPONSE CURVES — raw `m_h(Δ)`, candidate units

| checkpoint | scene | −32 | −16 | 0 | +16 | +32 |
|---|---|---:|---:|---:|---:|---:|
| NEG | 000165_10 | 3.316 | 3.174 | 3.125 | 3.083 | 3.081 |
| NEG | 000169_10 | 2.932 | 2.994 | 2.952 | 2.887 | 3.183 |
| seed0 | 000165_10 | 8.658 | 6.800 | 4.937 | 3.197 | 2.324 |
| seed0 | 000169_10 | 7.980 | 6.312 | 4.521 | 2.740 | 1.528 |
| seed1 | 000165_10 | 6.098 | 4.887 | 3.666 | 2.472 | 2.024 |
| seed1 | 000169_10 | 5.695 | 4.728 | 3.503 | 2.270 | 1.413 |
| seed2 | 000165_10 | 7.205 | 5.930 | 4.272 | 2.926 | 2.177 |
| seed2 | 000169_10 | 6.791 | 5.245 | 3.777 | 2.382 | 1.402 |

Every positive unit is strictly monotone decreasing across the full
preregistered `Δ` range (C2 holds at 6/6). The search-free control shows no
comparable signed sweep (`S_h` −0.0035 / +0.0025 per scene).

## 5. VERTICAL RESPONSE CURVES — raw `m_v(Δ)`, CONTAMINATED comparison

| checkpoint | scene | −32 | −16 | 0 | +16 | +32 |
|---|---|---:|---:|---:|---:|---:|
| NEG | 000165_10 | 3.185 | 3.133 | 3.125 | 3.086 | 3.033 |
| NEG | 000169_10 | 3.032 | 2.934 | 2.952 | 2.812 | 2.812 |
| seed0 | 000165_10 | 7.656 | 7.208 | 4.937 | 7.328 | 7.260 |
| seed0 | 000169_10 | 7.845 | 8.088 | 4.521 | 7.758 | 7.867 |
| seed1 | 000165_10 | 5.445 | 4.258 | 3.666 | 4.732 | 5.002 |
| seed1 | 000169_10 | 5.654 | 5.983 | 3.503 | 4.834 | 5.506 |
| seed2 | 000165_10 | 7.236 | 6.698 | 4.272 | 7.171 | 6.603 |
| seed2 | 000169_10 | 7.495 | 7.586 | 3.777 | 7.365 | 7.621 |

The vertical arm shows excursion without directional structure and is used in
C3 exactly as preregistered — but per §0.1 (quoted in §7) it is a
CONTAMINATED comparison, never a valid null.

### Even parts (descriptive, per `PREREGISTRATION.md` §6)

| group | Δ | median Even_h | median Even_v |
|---|---:|---:|---:|
| POS | 16 | +4.121 | +7.102 |
| POS | 32 | +4.394 | +6.989 |
| NEG | 16 | +3.035 | +2.991 |
| NEG | 32 | +3.128 | +3.015 |

---

## 6. FAILURE CLASSIFICATION (explicit, not collapsed)

| # | class | count | detail |
|---|---|---:|---|
| 1 | **consistency-positive** | **6 / 6** | C1∧C2∧C3∧C4 at every positive unit; claim ceiling §7 |
| 2 | generic binocular dependence | 0 | no unit shows `S_h ≈ S_v` with C3 failing |
| 3 | even architectural artifact | 0 | C2 holds everywhere |
| 4 | search-free artifact with odd leakage | 0 | every positive `\|S_h\|` exceeds NEG on the same scene |
| 5 | mixed / ambiguous | — | not applicable; the outcome is uniform |
| 6 | protocol failure | none | no hard stop fired; `protocol_deviations: none` |

---

## 7. CLAIM CEILING — `PREREGISTRATION.md` §0.1, VERBATIM

The following is quoted verbatim from `PREREGISTRATION.md` section 0.1 and
governs this result. Its wording is not softened:

> - This design was assessed in `SUCCESSOR_OPTIONS.md` as "defensible" but
>   underpowered: weaker than the experiment it replaces, and was NOT the
>   recommended successor. (The recommended successor was Option B, the
>   synthetic fronto-parallel self-pair sweep.)
> - The image-translation slope class and the vertical-shift null were both
>   marked RETIRED / INVALID in
>   `direction_decision_20260913T052115Z/DECISION.md` section 1. Quoted verbatim:
>   - * **Image-translation slope test** — REJECTED; the degenerate mechanism imitates sign and monotonicity.
>   - * **Vertical shift as a null** — INVALID on two independent grounds (no axis asymmetry for `shift="none"`; and `FH = 17 < 30` cells makes a halo-free vertical band empty, unfixable at any crop size below 496 px).
>   The vertical arm is therefore reported as a CONTAMINATED comparison, never
>   as a valid null.
> - Only 2 signed offsets exist, so C2 is a single inequality: probability 1/2
>   under any null.
> - The 3 trained checkpoints are reused and irreplaceable; the designer has
>   seen the GEOM-001 response curve. Achievable status is at most CONSISTENCY
>   EVIDENCE, never independent confirmation.
> - This run is executed at the project owner's explicit direction, recorded as
>   such.
> - On PASS the claim ceiling is frozen verbatim and does not exceed:
>   "consistency evidence for a signed, axis-asymmetric horizontal translation
>   response on held-out scenes; it establishes nothing at Level D."
> - On FAIL the verdict string is GEOMETRIC-CORRESPONDENCE-NOT-DEMONSTRATED,
>   and it must not be read as evidence that correspondence is absent.

On this PASS, the maximum claim is therefore frozen verbatim:

> "consistency evidence for a signed, axis-asymmetric horizontal translation
> response on held-out scenes; it establishes nothing at Level D."

Status at most CONSISTENCY EVIDENCE, never independent confirmation. The
vertical arm remains a CONTAMINATED comparison and not a valid null. No
stronger verb — confirms, proves, demonstrates correspondence, validates —
appears in this file.

---

## 8. PROVENANCE

Executed exactly as frozen. `PREREGISTRATION.md`, `translation_spec.json`,
`mask_spec.json`, `scene_selection.json`, `frozen.sha256` unmodified; no
checkpoint, seed, scene, crop, mask, translation value, axis definition,
statistic, control, decision rule or claim ceiling was changed. No training.
No historical record modified. No previous experiment re-run. Execution ran
from the record directory with stdout+stderr tee'd to `run.log`. No hard stop
fired at any point (HS-FREEZE, HS-REPRESENTABILITY, HS-CROP, HS-NOTRAIN,
HS-DETERMINISM, HS-COUNT all passed by completion). `results.json` carries
per-cell `m`/`Odd`/`Even`/`S` and per-unit C1..C4; the harness-internal label
for this outcome is recorded there as
`CONSISTENCY-EVIDENCE-SIGNED-AXIS-ASYMMETRIC-RESPONSE`, which is the same
6/6 consistency-positive outcome reported here as PASS under the brief's
verdict-string rule. `protocol_deviations: none`.

## 9. FILES

`PREREGISTRATION.md`, `RESULTS.md` (this file), `results.json`, `run.log`,
`ENVIRONMENT.txt`, `translation_spec.json`, `mask_spec.json`,
`scene_selection.json`, `frozen.sha256`, `select_scenes.py`,
`freeze_geomA.py`, `run_geomA.py`, `RELATED_RUNS.md`.
