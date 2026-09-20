# AUDIT — EXP-CORRESPONDENCE-TR-001

Execution audit. Provenance, hashes, deviations, and every hard-stop condition
checked against what actually happened.

---

## 1. PROVENANCE — exact hashes

### Freeze artifacts (verified over FILE BYTES before the first model load)

```
5e6a0bca39408018454fb69e3287f26f8524fc89475ee564a5da5c54a5fa0280  spec.json
8d58c412db5244dc16a8b59de7223ab4def8b70681837cb86928c4a5352e4722  random_reference.json
a9dd4bba226c6a4fd9e9767bab96a401f21c74890d21f181c922f3fe35be243b  PREREGISTRATION.md
21856ae1851761424fbfd716e036915668c5cf42cc91e3548700be7bf971b715  DESIGN_AUDIT.md
671e6546dc5243d371669971c05ba7aa7804337289d7bfd1c9367874aed1718b  EXECUTION_PROMPT.md
9d7d64704b7b7697bb647a198eaadd25b6dc7bce4e2acda937f4b1cbee376cce  PRE_EXECUTION_ADDENDUM.md
b0329a569bca6b4e9846f1d0dd5805e175c425e307eb9156926b40f7442e9f75  freeze_tr.py
```

`spec.json` and `random_reference.json` digests appear verbatim in
`PREREGISTRATION.md` — asserted at run time, not by hand.

### Random reference (READ ONLY)

```
source          phase2/diagnostics/correspondence_arch_rate/20260911T130507Z
results.json    1f0463ec08b950d3c1b69804407299f66b4b6cee5c04af51419a061dba7458f1   OK
  construction_spec.json    7a4e2341fc50e3695648cc286cdb4219390a0378a6669994ec0e4555b75cf9b0   OK
  randomisation_spec.json   e52b5b03c5d4406e5ed9212f6681ee53bb448927eebddd367a7698ce9ce203ad   OK
  classification_spec.json  273f7562b62dd4c29a2d617e3b2b4b24d98532fcc7d99531b82b55b377d0cd28   OK
384 units · scenes [5,7,8,9] · mask rule identical · t levels identical · T_FIT identical
frozen alpha_h value set == the source record's value set  (exact, sorted comparison)
```

**Not re-run. Not re-scored. Not regenerated. Seed 18 retained unchanged.**

### Checkpoints (verified at load, every load)

```
H2_seed0        581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a   OK
H2_seed1        58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf   OK
H2_seed2        245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69   OK
NEG_shift_none  d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1   OK
```

Trained aggregation parameter hashes, asserted **36 times** during execution
(once per cross unit, before every readout):

```
H2_seed0  d8d12446c584e7c0acdf26dbe9cb04169723adb7501ef4725ca301a45dd7696d
H2_seed1  75f3c7c64ddf481bf3fb037b25f81ba0e5df80397b209beb20d3a55a33c53d6e
H2_seed2  1e3bb32e902ea177ad432ba3c370957f35b9308eacaf24918aae4ff4cfa7881f
```

All three distinct — asserted, not assumed.

### Harness

```
a06911b6f07f1a618c948d19d7f5e632e1280de6abb748a781c948072771a8b6  run_tr.py
```

---

## 2. WHAT WAS EXECUTED

```
208 cost volumes        156 trained  (3 extractors x 4 scenes x 13 conditions)
                         52 search-free (1 checkpoint x 4 scenes x 13 conditions)
560 readout passes      504 trained     (36 cross units x 14)
                         56 search-free (4 scenes x 14)
 14 W4 repeat passes    fresh process, one complete matched unit
  0 training steps       no optimizer imported or constructed; no .backward(); no .step()
```

Device `cuda` (RTX 4060 Laptop), torch 2.7.0+cu128, numpy 2.5.1, fp32,
`no_grad`, `eval`, deterministic algorithms on, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Wall clock 23.68 s.

---

## 3. DECLARED DEVIATIONS

### 3.1 Cost-volume count: 208 executed vs 156 specified

**MEASURED.** `spec.json` `compute.cost_volumes = 156` counts only the trained
arm (`3 × 4 × 13`). The search-free control, which the same spec requires to be
run **fresh** on these scenes, needs its own 52 volumes (`1 × 4 × 13`).
`156 + 52 = 208`.

**The specified number was incomplete; the executed number is correct for the
specified work.** Readout passes matched exactly: `504 + 56 = 560` as specified.
No measurement is affected — a cost volume is an input to a readout, and every
readout the spec names was performed on the volume the spec names.

### 3.2 The `α_h` fit-window defect

Found and resolved **before execution**; see `PRE_EXECUTION_ADDENDUM.md`
(`9d7d6470…`) and `RESULTS.md` §8. Summary:

- Three frozen documents said "ALL SEVEN levels"; the hash-bound reference
  definition uses six (`T_FIT`, excluding `t = 0`).
- Settled empirically against the reference's own raw curves:
  six-level fit reproduces all 384 recorded `α_h` **exactly** (max error `0.0`);
  seven-level fit does not (max error `0.09443637132644654`).
- Primary statistic = six-level (reference-identical). The seven-level reading is
  computed and recorded as `α_h7`, compared to nothing.
- **No document was modified.** The defect stands in the record beside its
  correction.

**No other deviation.**

---

## 4. HARD STOPS — every one checked

| hard stop (`spec.json`) | status |
|---|---|
| trained weights modified | **did not occur** — inference only, no optimizer constructed |
| trained aggregation retrained | **did not occur** |
| random reference not hash-verifiable | **verified OK** at source and in the frozen copy |
| construction differs from spec | **does not** — t levels, crops, axes asserted against `spec.json` |
| translation levels differ | **do not** — asserted equal to the reference's |
| mask differs | **does not** — 145 152 px asserted; mask rule string compared to the reference's |
| preprocessing differs | **does not** — same `normalize`, same integer slicing, no fill/interpolation |
| deterministic repeat fails | **passed** — bit-identical after process restart |
| trained curves inspected before freeze | **did not occur** on these scenes (disclosed contamination: GEOM-002 curves on scenes `{1,2,3,4}` were read; see `DESIGN_AUDIT.md` §5) |
| post-hoc statistic introduced | **none** — `α_h` inherited from ARCH-RATE-001, reproduced bit-exactly |
| post-hoc threshold introduced | **none** — `1.0` / `10.0` / `1.0` inherited unchanged; no threshold anywhere derives from a trained observation |
| historical results overwritten | **none** — every file written is new; nothing outside this directory was touched |

**No hard stop fired.** Unlike GEOM-002, the harness raises `HardStop` and halts
on every one of these; none is recorded-but-ignored.

---

## 5. WIRING CHECKS AS EXECUTED

- **W1** — the left crop's SHA-256 was accumulated per scene across every
  condition and every model. `{000165: 1, 000167: 1, 000168: 1, 000169: 1}`
  distinct hashes. Byte-identical. **PASS.**
- **W2 (origin)** — 24 distinct `(axis, t, y0, x0)` tuples asserted against
  `(0,t)` horizontal / `(t,0)` vertical. **PASS.**
- **W2 (sign, verified from the arrays)** — 84 checks of
  `right_crop[:, :W-t] == left_crop[:, t:]`, exact array equality. This is the
  implementation-level statement that a feature at left column `x+t` sits at
  right column `x`, i.e. `d = a_L − a_R = +t` px and `k_true = t/16`.
  **The sign was not assumed. PASS.**
- **W3** — `k_true` recorded per unit. The model output was **not** required to
  equal it, and does not: intercepts run ≈ 1.4–1.8 candidates at `t = 0`.
- **W4** — `H2_seed0 / H2_seed0 / 000165_10.png` recomputed in a **separate
  Python process** via `subprocess`. All 14 values compared as IEEE-754 bytes.
  **Bit-identical. PASS.**
- **No numeric `t = 0` expectation existed.** The invalid `5.5` anchor from
  GEOM-002 was never reinstated. The measured trained `t = 0` values
  (1.418 – 1.823) would have fired it; there was nothing to fire.

---

## 6. STATISTICAL DISCIPLINE

- **No p-value is computed anywhere.** `results.json` carries
  `analysis.p_value = null` with the reason recorded.
- Units are **not** independent replicates: 12 matched units share 3 checkpoints
  and 4 scenes, and the checkpoints originate from related training conditions.
- The only inferential summary is **rank-biserial correlation from Mann-Whitney
  U, reported as an effect size only**.
- MATCHED and CROSS are **never pooled**. SEARCH-FREE is **never** mixed into the
  random-weight null.
- The invalid inference *"0/384 random qualified therefore any trained
  qualification proves learning"* is recorded as rejected in `results.json` and
  is used nowhere.

---

## 7. AN OBSERVATION, FLAGGED AND NOT ACTED ON

**MEASURED.** The search-free control's `t = 0` median is `7.603` to three
decimals, identical across all four scenes.

**DERIVED.** Under `shift="none"` with a self-pair at `t = 0`, both feature maps
are identical, so the cost volume is exactly zero and the readout returns a pure
architecture constant — scene-independent by construction.

**UNKNOWN / NOT ACTED ON.** GEOM-002 recorded a scene-independent `t = 0` value
of `7.603198`. Whether that is the same quantity is **not determined here**, and
this record makes **no** reinterpretation of GEOM-002, whose formal status
remains `PROCEDURALLY-COMPROMISED / NO-DETERMINATION`. Noted because a future
auditor should see the coincidence rather than rediscover it.

---

## 8. WHAT A SCEPTICAL READER SHOULD ATTACK

Stated deliberately, because the result is large and a large result deserves it:

1. **`α_h ≈ 1` is the architecturally predicted value.** The campaign's own
   algebra forces a unit ramp in the padding-free interior for *any* weights.
   The separation shows the trained aggregation **realises** it and 384 sampled
   random ones do not — not that it computes correspondence. `RESULTS.md` §7.
2. **The random population is 384 draws from one initialisation scheme.** It is
   not "random weights" as a space. Every claim is phrased against the *observed*
   population.
3. **The search-free control is doubly confounded** — different checkpoint *and*
   different construction. It cannot isolate the construction as the cause.
4. **`R_h` looks like a separator and is not.** The search-free control is 4/4
   outside the random `R_h` range while having no slope at all; its range is
   entirely the degenerate `t = 0 → 16` drop.
5. **Only 4 of 12 matched units are `QUALIFYING`, all from one checkpoint.** The
   separation quantity is uniform; the taxonomy is not.
6. **Contamination is real and disclosed** — GEOM-002's trained curves on other
   scenes were read before this freeze. This experiment is a preregistered
   descriptive comparison, not a strictly confirmatory test.

---

## 9. VERDICT

```
CLEAR-SEPARATION
```

Freeze intact, one defect found before execution and disclosed rather than
repaired away. No hard stop fired. No threshold, statistic, scene, seed,
translation or checkpoint was selected after any trained result was seen.

**Claim ceiling enforced:** the trained response is *distinguishable from the
observed random population*. It is **not** shown to be impossible under random
weights, and it does **not** prove learned correspondence. Level D remains
`NOT-DEMONSTRATED`.
