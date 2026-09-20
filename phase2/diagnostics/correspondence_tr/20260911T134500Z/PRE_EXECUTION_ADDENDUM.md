# PRE-EXECUTION ADDENDUM — EXP-CORRESPONDENCE-TR-001

**Written before any trained aggregation was executed.** No results file exists.
`spec.json`, `PREREGISTRATION.md`, `DESIGN_AUDIT.md` and `EXECUTION_PROMPT.md`
are **left unmodified**, defect included. This addendum is additive.

---

## 1. THE DEFECT

Three frozen documents describe the primary statistic's fit as spanning **all
seven** translation levels:

```
PREREGISTRATION.md:123   "OLS over ALL SEVEN levels, u = t/16"
EXECUTION_PROMPT.md:91   "OLS of m_h(t) on u = t/16 over ALL SEVEN levels"
spec.json:89             "... over ALL SEVEN levels t in {0,16,32,48,64,80,96}"
```

The same `spec.json` object simultaneously asserts:

```json
"identical_to_reference": true,
"reference_definition_sha256": "273f7562b62dd4c29a2d617e3b2b4b24d98532fcc7d99531b82b55b377d0cd28"
```

That digest is ARCH-RATE-001's `classification_spec.json`, which defines the
statistic as

> "free-intercept OLS slope of m_axis(t) on t/16 **over t in T_FIT**"

and `construction_spec.json` (`7a4e2341…`) defines

```json
"t_levels_px":     [0, 16, 32, 48, 64, 80, 96]
"t_fit_levels_px": [   16, 32, 48, 64, 80, 96]
```

**T_FIT is six levels. `t = 0` is excluded from the fit.** The prose clause and
the hash-bound clause contradict each other.

## 2. WHICH IS TRUE — settled empirically, not by preference

**MEASURED.** Both candidate fits were recomputed from the reference's own
published raw `m_h` curves and compared against its 384 recorded `α_h` values:

```
max | alpha_h_recorded  -  six-level fit  |  =  0.0            (all 384 units, exact)
max | alpha_h_recorded  -  seven-level fit |  =  0.09443637132644654
```

The frozen random population is a **six-level** fit, reproduced bit-exactly.
The "ALL SEVEN" phrasing is a **transcription defect I introduced when drafting
the TR-001 freeze**. It never described the reference.

## 3. RESOLUTION — by a criterion that predates the observation

TR-001 is, in its entirety, a comparison of trained `α_h` against **those 384
frozen values**. A statistic that is not identical to theirs makes the
comparison meaningless — this is the `spec.json` hard stop *"construction differs
from this spec"* / *"post-hoc statistic introduced"* in substance.

Therefore the controlling clause is `identical_to_reference: true`, hash-bound to
`273f7562…`. It is controlling because the experiment has no meaning without it,
**not** because of anything observed in a trained model. No trained curve has been
computed.

```
PRIMARY    alpha_h  :=  free-intercept OLS slope over T_FIT = {16,32,48,64,80,96}
                        IDENTICAL to the 384 frozen random values.
                        Everything in section 12 (comparison) and section 19
                        (decision) uses THIS and only this.

REPORTED   alpha_h7 :=  the same fit over all seven levels, the literal prose
                        reading. Recorded per unit in results.json for
                        completeness. NOT compared to the random population —
                        no comparable random values exist. NOT used in any
                        classification or decision.
```

**Both are computed for every unit before any of them is inspected.** Neither is
selected after the fact; the assignment above is fixed by this document.

## 4. UNAFFECTED BY THIS ADDENDUM

The seven-level span is correct everywhere else, and stays:

| quantity | levels | unchanged |
|---|---|---|
| `CLIPPED` (`min ≤ 1.0` or `max ≥ 10.0`) | all 7 | yes |
| `FLAT` (`range < 1.0`) | all 7 | yes |
| `R_h = max − min` | all 7 | yes |
| first differences (six of them) | all 7 | yes |
| `Q2` strict monotonicity | all 7 | yes |
| classification order and every threshold | — | yes |
| construction, crops, mask, scenes, checkpoints | — | yes |

Only the **slope fit window** is affected, and only to match the reference.

## 5. GOVERNANCE

- Nothing is modified. The defect stands in the record beside its correction.
- The random reference is **not** regenerated, re-scored or touched.
- `α_h7` is recorded so any reader can see exactly what the literal prose would
  have produced.
- If the two readings were to disagree about the verdict, that disagreement is
  **reported**, not resolved in favour of whichever is more interesting.

**Still true at the time of writing: no trained aggregation has been executed.**
