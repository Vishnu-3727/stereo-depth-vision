# EXP-CORRESPONDENCE-GEOM-001 — RESULTS

**Signed odd-part translation response, with a matched vertical control and the
search-free `shift="none"` model as artifact baseline.**

Record `phase2/diagnostics/correspondence_geom/20260911T111809Z/`.
Pre-registered in `PREREGISTRATION.md`, frozen before execution.
`PREREGISTRATION.md` was **not modified** (last write 11:23:32Z; execution began
11:26Z).

Inference only — **208 forward passes**, 15.9 s wall clock. No training, no
fine-tuning, no architecture change, no new checkpoint. Crop-based translation:
no fill, no padding, no interpolation. Phase 1 untouched. No historical record
modified.

```
VERDICT: GEOMETRIC-CORRESPONDENCE-NOT-DEMONSTRATED
         0 / 12 units pass
```

**A design defect in the frozen protocol was exposed by execution. It is
reported in §6, it was NOT repaired, and the verdict is NOT rescued.**

---

## 1. FREEZE VERIFICATION — 10/10 OK, before the first model instantiation

| # | check | result |
|---|---|---|
| 1 | `PREREGISTRATION.md` exists | OK, 22 080 bytes |
| 2 | `translation_spec.json` sha256 | `4f5047404b41ec906a98e62f9d31159f9ac8f11fc49455f0fcf64ef8e2397c97` OK |
| 3 | `mask_spec.json` sha256 | `5a2927a783b1569f67a7cca460ffb041bc281e26fc736d417a99f8dbcf417330` OK |
| 4–5 | `sha256sum -c frozen.sha256` | both **OK** |
| 6 | no `RESULTS.md` before execution | OK — absent |
| 7 | provenance anomaly documentation | OK — `PROVENANCE_ANOMALY.md`, 3 × `AUTHORSHIP-UNKNOWN` |
| 8 | frozen checkpoint hashes | all four OK |
| 9 | 12-unit structure | 3 weight sets × 4 scenes = 12 positive, 4 control |
| 10 | no protocol modification after preregistration | OK |

Both spec digests were additionally re-verified **over file bytes** inside the
harness before any model was built.

---

## 2. SANITY CHECKS — 12/12

| # | check | result |
|---|---|---|
| 1 | all crops identical | **one shape only: `(272, 1136)`** |
| 2 | no interpolation | integer slicing only |
| 3 | no fill / padding | every crop window asserted inside the image |
| 4 | frozen mask used | counts **34 245 / 46 199 / 33 250 / 50 747**, pooled 164 441 — exact match |
| 5 | same code path both axes | axis is a function argument to `right_origin`; no branch in the forward path |
| 6 | ± directions symmetric | `Δ` set verified symmetric and all multiples of 16 |
| 7 | no training | no optimizer imported or constructed; no `.backward()`, no `.step()` |
| 8 | no weight change | full parameter sha256 re-hashed after every scene, unchanged throughout |
| 9 | checkpoint hashes unchanged | verified at load, all four match |
| 10 | deterministic execution | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `no_grad`, `eval`, fp32 |
| 11 | historical records untouched | newest modification in every prior record predates execution |
| 12 | forward passes accounted | **208 / 208 expected** (4 models × 4 scenes × 13 conditions) |

Cost volume shape was `(1, 32, 12, 17, 71)` in every one of the 208 passes —
consistent with the 1136 × 272 crop at stride 16.

---

## 3. PER-UNIT DECISION TABLE

`S` in candidates per image pixel. Anchor `−1/16 = −0.0625`.

| checkpoint | scene | S_h | S_v | S_h(NEG) | Odd_h(16) | Odd_h(32) | Odd_h(48) | C1 | C2 | C3 | C4 | unit |
|---|---|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|:--:|:--:|
| seed0 | 000187_10 | −0.00071 | +0.00373 | −0.00028 | −1.2330 | +0.7450 | −0.1390 | Y | **n** | n | Y | **FAIL** |
| seed0 | 000160_10 | −0.00770 | −0.00401 | +0.00057 | −1.1838 | +0.1960 | −0.3113 | Y | **n** | Y | Y | **FAIL** |
| seed0 | 000191_10 | +0.00086 | +0.00068 | +0.00109 | −1.2820 | +0.7486 | −0.0078 | **n** | **n** | Y | Y | **FAIL** |
| seed0 | 000166_10 | −0.01589 | −0.00534 | −0.00213 | −1.4916 | −0.2183 | −0.5440 | Y | **n** | Y | Y | **FAIL** |
| seed1 | 000187_10 | −0.00264 | +0.00018 | −0.00028 | −0.8200 | +0.4392 | −0.2164 | Y | **n** | Y | Y | **FAIL** |
| seed1 | 000160_10 | +0.00251 | −0.00744 | +0.00057 | −0.7721 | +0.3249 | +0.2284 | **n** | **n** | n | n | **FAIL** |
| seed1 | 000191_10 | +0.00368 | −0.00397 | +0.00109 | −0.8283 | +0.4788 | +0.2319 | **n** | **n** | n | n | **FAIL** |
| seed1 | 000166_10 | −0.00787 | −0.00285 | −0.00213 | −1.0656 | −0.0713 | −0.1850 | Y | **n** | Y | Y | **FAIL** |
| seed2 | 000187_10 | +0.00477 | −0.00413 | −0.00028 | −0.9961 | +0.6703 | +0.2414 | **n** | **n** | Y | n | **FAIL** |
| seed2 | 000160_10 | −0.00050 | −0.00716 | +0.00057 | −0.9728 | +0.4893 | −0.0395 | Y | **n** | n | Y | **FAIL** |
| seed2 | 000191_10 | −0.00125 | −0.00595 | +0.00109 | −1.0157 | +0.3338 | +0.0226 | Y | **n** | n | Y | **FAIL** |
| seed2 | 000166_10 | −0.00337 | −0.00082 | −0.00213 | −1.2386 | +0.2533 | −0.0076 | Y | **n** | Y | Y | **FAIL** |

```
C1 (sign)                8 / 12
C2 (monotonicity)        0 / 12      <- binding failure
C3 (axis specificity)    7 / 12
C4 (vs search-free)      9 / 12
PASS                     0 / 12
```

---

## 4. HORIZONTAL RESPONSE CURVES — raw `m_h(Δ)`, candidate units

| checkpoint | scene | −48 | −32 | −16 | 0 | +16 | +32 | +48 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NEG | 000187_10 | 2.393 | 2.374 | 2.375 | 2.306 | 2.346 | 2.345 | 2.381 |
| NEG | 000160_10 | 3.103 | 3.050 | 3.006 | 2.977 | 2.947 | 3.054 | 3.205 |
| NEG | 000191_10 | 3.928 | 3.915 | 3.832 | 3.790 | 3.868 | 4.043 | 3.993 |
| NEG | 000166_10 | 2.900 | 2.845 | 2.799 | 2.736 | 2.732 | 2.770 | 2.655 |
| seed0 | 000187_10 | 7.929 | 5.961 | 4.168 | 2.319 | 1.702 | **7.452** | **7.651** |
| seed0 | 000160_10 | 8.103 | 6.229 | 4.350 | 2.521 | 1.982 | **6.621** | **7.481** |
| seed0 | 000191_10 | 7.902 | 5.924 | 4.126 | 2.382 | 1.562 | **7.422** | **7.887** |
| seed0 | 000166_10 | 8.542 | 6.892 | 4.948 | 3.186 | 1.965 | **6.455** | **7.454** |
| seed1 | 000187_10 | 5.642 | 4.451 | 3.189 | 1.966 | 1.549 | **5.330** | **5.210** |
| seed1 | 000160_10 | 5.789 | 4.535 | 3.287 | 2.080 | 1.743 | **5.185** | **6.246** |
| seed1 | 000191_10 | 5.603 | 4.354 | 3.138 | 2.025 | 1.481 | **5.312** | **6.067** |
| seed1 | 000166_10 | 5.922 | 4.985 | 3.756 | 2.518 | 1.625 | **4.843** | **5.552** |
| seed2 | 000187_10 | 6.504 | 4.995 | 3.478 | 2.051 | 1.486 | **6.335** | **6.987** |
| seed2 | 000160_10 | 6.698 | 5.200 | 3.650 | 2.173 | 1.705 | **6.178** | **6.619** |
| seed2 | 000191_10 | 6.488 | 4.948 | 3.462 | 2.110 | 1.430 | **5.616** | **6.533** |
| seed2 | 000166_10 | 6.880 | 5.611 | 4.173 | 2.702 | 1.696 | **6.117** | **6.865** |

The bolded `+32` and `+48` entries are discontinuous jumps of +3.1 to +5.9
candidates relative to `+16`. §6 identifies the cause.

## 5. VERTICAL RESPONSE CURVES — raw `m_v(Δ)`

| checkpoint | scene | −48 | −32 | −16 | 0 | +16 | +32 | +48 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NEG | 000187_10 | 3.250 | 2.784 | 2.429 | 2.306 | 2.338 | 2.296 | 2.234 |
| NEG | 000160_10 | 3.988 | 3.644 | 3.295 | 2.977 | 3.123 | 3.015 | 3.052 |
| NEG | 000191_10 | 4.891 | 4.569 | 4.215 | 3.790 | 3.916 | 3.918 | 3.843 |
| NEG | 000166_10 | 3.362 | 3.247 | 3.103 | 2.736 | 2.836 | 3.026 | 2.915 |
| seed0 | 000187_10 | 6.954 | 7.781 | 7.500 | 2.319 | 7.975 | 7.317 | 7.662 |
| seed0 | 000160_10 | 8.114 | 7.194 | 8.049 | 2.521 | 8.182 | 7.094 | 7.537 |
| seed0 | 000191_10 | 7.864 | 8.231 | 8.248 | 2.382 | 8.222 | 7.737 | 8.304 |
| seed0 | 000166_10 | 7.480 | 8.117 | 8.300 | 3.186 | 8.272 | 7.406 | 7.165 |
| seed1 | 000187_10 | 4.718 | 5.860 | 5.379 | 1.966 | 5.087 | 5.374 | 5.166 |
| seed1 | 000160_10 | 5.342 | 5.696 | 6.216 | 2.080 | 5.499 | 5.111 | 4.859 |
| seed1 | 000191_10 | 5.872 | 5.245 | 6.168 | 2.025 | 5.690 | 5.133 | 5.513 |
| seed1 | 000166_10 | 4.834 | 6.163 | 5.963 | 2.518 | 5.309 | 5.500 | 5.069 |
| seed2 | 000187_10 | 6.632 | 7.037 | 6.912 | 2.051 | 6.981 | 6.833 | 6.128 |
| seed2 | 000160_10 | 7.045 | 6.594 | 7.472 | 2.173 | 6.757 | 6.406 | 6.340 |
| seed2 | 000191_10 | 7.118 | 7.284 | 7.221 | 2.110 | 7.181 | 7.079 | 7.106 |
| seed2 | 000166_10 | 6.972 | 7.346 | 7.327 | 2.702 | 7.090 | 6.406 | 7.106 |

For the positive checkpoints, **any** non-zero vertical displacement — including
±16 px — moves the median from ≈2.0–3.2 to ≈5–8 candidates, with no directional
structure. The search-free model shows a far smaller vertical excursion.

### Even parts (descriptive, per `PREREGISTRATION.md` §5)

| group | Δ | median Even_h | median Even_v |
|---|---:|---:|---:|
| POS | 16 | +2.684 | **+7.158** |
| POS | 32 | +5.677 | +7.006 |
| POS | 48 | +6.702 | +6.756 |
| NEG | 16 | +2.871 | +3.089 |
| NEG | 32 | +2.930 | +3.233 |
| NEG | 48 | +2.966 | +3.329 |

---

## 6. THE DESIGN DEFECT THAT EXECUTION EXPOSED — reported, not repaired

**C2 fails at 12/12 because `Odd_h(32)` and `Odd_h(48)` do not continue the
`−Δ/16` trend. The cause is a range error in the frozen protocol, and it is mine.**

`PREREGISTRATION.md` §3.1 rejected `Δ = ±64` on rail risk using ARCH-001's
published identity medians of **3.84–5.83 candidates**. Those medians were
measured under the **`GT/16 ∈ [2,8]` band mask**. GEOM-001 deliberately uses a
**different, broader mask** (`GT > 0`), under which the identity medians are
**1.97–3.19 candidates** (the `Δ = 0` column of §4).

The correct rail check was available inside the frozen record all along —
`mask_spec.json` records this mask's own ground truth:

| scene | GT/16 median | Δ at which true disparity reaches 0 |
|---|---:|---:|
| 000187_10 | 1.045 | **+17 px** |
| 000160_10 | 1.165 | **+19 px** |
| 000191_10 | 1.105 | **+18 px** |
| 000166_10 | 1.552 | **+25 px** |

**Beyond roughly `Δ = +20 px` the true disparity of the median masked pixel is
negative, and negative disparity is not representable on a candidate axis that
spans `0 … 11`.** The `+32` and `+48` conditions therefore do not test the
geometric hypothesis at all — there is no correct answer inside the search range
— and the observed jump to ≈5–8 candidates is the readout's behaviour when no
candidate matches, not a geometric response.

The `−Δ` arm has no such problem: it *increases* disparity, and `2.2 + 3 = 5.2`
candidates is comfortably inside `0 … 11`.

**The Δ range was asymmetric in its validity and the preregistration treated it
as symmetric.** The frozen mask statistics needed to catch this were in
`mask_spec.json`; §3.1 used a figure from a different mask instead.

**This is recorded as a protocol design defect, not repaired.** The verdict
stands at 0/12. No condition was relaxed, no `Δ` was dropped, no sub-range was
substituted for the frozen statistic.

---

## 7. POST-HOC DESCRIPTIVE OBSERVATIONS — NOT EVIDENCE

**These were computed after seeing the results and are reported only because
suppressing them would be dishonest. They cannot be promoted to evidence, they
did not affect the verdict, and no claim rests on them.**

**7.1** Over `Δ ∈ {−48, −32, −16, 0, +16}` — the sub-range in which a correct
answer exists — `m_h(Δ)` is **strictly monotone decreasing at 12/12 positive
units**, with least-squares slopes of:

| checkpoint | slope (cand/px) | × anchor |
|---|---:|---:|
| seed0 (4 scenes) | −0.0997 … −0.1054 | 1.59 … 1.69 |
| seed1 (4 scenes) | −0.0659 … −0.0691 | 1.05 … 1.11 |
| seed2 (4 scenes) | −0.0810 … −0.0830 | 1.30 … 1.33 |
| **NEG (4 scenes)** | **−0.0010 … −0.0028** | **0.02 … 0.04**, monotone at only 2/4 |

**7.2** `Odd_h(16)` is completely separated between conditions: positive
checkpoints **−1.4916 … −0.7721** (anchor at `Δ=16` is `−1.0000`); search-free
control **−0.0334 … +0.0180**. No overlap.

**7.3** At `|Δ| = 16`, `Even_h` (+2.684) is far below `Even_v` (+7.158) for the
positive checkpoints, while the search-free control shows no such asymmetry
(+2.871 vs +3.089).

**Why none of this can be claimed.** The sub-range in 7.1 was selected *after*
observing where the curve breaks. `Odd_h(16)` alone was not the preregistered
statistic — `S_h` over all three offsets was. The even-part comparison in 7.3 was
declared descriptive in advance. Promoting any of them would be exactly the
post-hoc statistic selection that INDEX-001, INDEX-002 and ARCH-001 were each
faulted for. **A successor experiment must be preregistered afresh, and must
contend with the fact that these numbers are now public.**

---

## 8. FAILURE CLASSIFICATION (explicit, not collapsed)

| # | class | count | detail |
|---|---|---:|---|
| 1 | sign failure | 4 units | `S_h > 0` at seed0/000191, seed1/000160, seed1/000191, seed2/000187 — all with `\|S_h\| < 0.005`, i.e. a near-zero slope whose sign is arbitrary because the fit spans the invalid `+Δ` region |
| 2 | monotonicity failure | **12 units** | `C2` fails everywhere; this is the binding failure |
| 3 | axis-specificity failure | 5 units | `C3` fails at 5 of 12 |
| 4 | trained-vs-search-free failure | 3 units | `C4` fails at 3 of 12 |
| 5 | mixed / ambiguous | — | not applicable; the outcome is uniform |
| 6 | **protocol failure** | **yes — design range defect** | §6: the `+Δ` arm exceeds the representable disparity floor, so `Δ = +32, +48` do not test the hypothesis |

**This is not "stereo failed."** The preregistered statistic was evaluated over a
range in which one third of the conditions could not, even in principle, exhibit
the predicted behaviour.

---

## 9. CLAIM CEILING

```
GEOMETRIC-CORRESPONDENCE-NOT-DEMONSTRATED
```

**It is NOT concluded that correspondence is absent.** The experiment did not
establish a horizontal, sign-consistent odd response under the preregistered
statistic and decision rule, and that is the whole of the finding.

Not claimed: correct disparity magnitude, correct disparity, genuine full
disparity search, final stereo correctness, generalisation. No EPE, D1 or RMSE
was computed. Candidate indices were never converted to physical disparity in
the statistic.

---

## 10. PROVENANCE

Executed exactly as frozen. `PREREGISTRATION.md` unmodified (11:23:32Z, before
execution at 11:26Z). Protocol deviations: **none**. No checkpoint, seed, scene,
crop, mask, translation value, axis definition, statistic, control, decision rule
or claim ceiling was changed. No training. No historical record modified. No
previous experiment re-run.

`HEAD 58e8a19`; `phase-1-frozen b4207e5`; `git diff phase-1-frozen -- src scripts`
empty; `phase2/` untracked, so no cryptographic versioning is claimed for this
record.

## 11. FILES

`PREREGISTRATION.md`, `RESULTS.md` (this file), `results.json`, `run.log`,
`translation_spec.json`, `mask_spec.json`, `frozen.sha256`,
`PROVENANCE_ANOMALY.md`, `RELATED_RUNS.md`, `freeze_geom_specs.py`,
`run_geom.py`.
