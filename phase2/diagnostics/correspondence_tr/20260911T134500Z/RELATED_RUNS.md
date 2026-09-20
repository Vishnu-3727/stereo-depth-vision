# RELATED RUNS — EXP-CORRESPONDENCE-TR-001

Every record below is preserved exactly as produced. **None was modified,
re-run, re-scored or deleted.** All prior verdicts stand unchanged.

First record under `phase2/diagnostics/correspondence_tr/`.

---

## Lineage

```
Phase 1   EXP-007  right-image ablation   ·   EXP-010  cost volume degenerate
        v
EXP-CORRESPONDENCE-SHIFT-001 (Stage A)    DIAGNOSTIC-INVALID-AS-PREREGISTERED
EXP-CORRESPONDENCE-INDEX-001 x2           both EXPLORATORY-ONLY
  + provenance audit                      NO-CONFIRMATORY-EVIDENCE
EXP-CORRESPONDENCE-INDEX-002              NOT-DEMONSTRATED
  + INDEX-003 design audit                DESIGN-INCONCLUSIVE
EXP-CORRESPONDENCE-ARCH-001               CASE-A-ARCHITECTURE-INDUCED
  + next-direction audit                  BYPASS-LEVEL-C
EXP-CORRESPONDENCE-GEOM-001               NOT-DEMONSTRATED (representability defect)
  + successor audit                       SUCCESSOR-DESIGN-JUSTIFIED
EXP-CORRESPONDENCE-GEOM-002               PROCEDURALLY-COMPROMISED / NO-DETERMINATION
  + postmortem                            C3 misspecified, C4 vacuous
  + pre-design algebra audit              DESIGN-PROBLEM-SOLVABLE
  + gate audit                            GATE-CAN-BE-MADE-RIGOROUS
EXP-CORRESPONDENCE-ARCH-RATE-001          QUALIFYING RATE = 0 / 384 (a measurement)
        v
EXP-CORRESPONDENCE-TR-001                                (this record)
  phase2/diagnostics/correspondence_tr/20260911T134500Z/
  VERDICT = CLEAR-SEPARATION   (12/12 matched units outside the observed
                                random population; claim ceiling enforced)
```

---

## 1. ARCH-RATE-001 — the reference this experiment consumes

**Record:** `correspondence_arch_rate/20260911T130507Z/`
**Result:** `QUALIFYING RATE = 0 / 384`, tally `FLAT 378 · NON_MONOTONE 6`.

TR-001 **reads it, never writes it.** Its `results.json`
(`1f0463ec08b9…`) and all three internal spec hashes were verified at execution.
The 384 `α_h` values are the entire comparison population.

| inherited unchanged | from |
|---|---|
| construction (crops, `t` levels, `a_R = a_L + t`, no fill) | `construction_spec.json` `7a4e2341…` |
| mask (`64 ≤ y < 208`, `64 ≤ x < 1072`, 145 152 px) | `classification_spec.json` `273f7562…` |
| statistic `α` (free-intercept OLS over `T_FIT`) | `classification_spec.json` `273f7562…` |
| classification order and all thresholds (`1.0` / `10.0` / `1.0`) | `classification_spec.json` `273f7562…` |
| scenes `[5,7,8,9]` | `classification_spec.json` `273f7562…` |
| 14-readout-per-unit measurement path (incl. the redundant `t = 0` vertical pass) | `run_arch_rate.py` |

**Verified, not assumed:** the six-level fit reproduces 32 of its recorded `α_h`
values **exactly** (max error `0.0`). See `PRE_EXECUTION_ADDENDUM.md`.

**Its preregistered prediction was falsified** (it predicted a *high* rate;
0/384 was measured). TR-001 rests on the measurement, not the prediction.

**Seed 18** — five of six non-flat random units — is retained in the population
exactly as measured. It was **not** excluded, downweighted, treated as an anomaly,
or used to construct a special null.

## 2. GEOM-002 — closed, and not reinterpreted

**Record:** `correspondence_geom/20260911T115917Z/`
**Formal status:** `PROCEDURALLY-COMPROMISED / NO-DETERMINATION`
(`correspondence_geom/postmortem_20260911/`). **Unchanged by this record.**

TR-001 inherits its **construction** (unchanged) and repairs the two defects the
postmortem identified:

| GEOM-002 defect | state here |
|---|---|
| through-origin slope maps a step of height `h` to `0.2308h` | **free-intercept** slope; every constant step maps to `α = 0` exactly |
| a declared hard stop fired and the harness did not halt | every hard stop raises and halts; **and no hard stop encodes a predicted numeric value** |

**Disclosed contamination:** GEOM-002 published trained `m_h(t)` curves on scenes
`{1,2,3,4}` under the same construction, and I have read them. TR-001 runs on
`{5,7,8,9}`, which no experiment had measured with a trained aggregation. Those
GEOM-002 values are used as no threshold, comparand or reference — but this
experiment is therefore a **preregistered descriptive comparison**, not a strictly
confirmatory test. See `DESIGN_AUDIT.md` §5.

`AUDIT.md` §7 records a numeric coincidence between GEOM-002's scene-independent
`t = 0` value and the search-free control's, **flagged as UNKNOWN and acted on in
no way.**

## 3. ARCH-001 — context only

**Record:** `correspondence_arch/20260911T025312Z/` — `CASE-A-ARCHITECTURE-INDUCED`.

Its descriptive contrast (random aggregation orbit ranges 0.09–1.42 vs trained
5.36–6.39) is qualitatively consistent with what TR-001 measures on a different
intervention (`R_h`: random median 0.354, trained matched median 5.42). **Cited
as context only** — its numbers enter no statistic, threshold or classification
here, and its verdict is unchanged.

## 4. The audit chain that made this design possible

| record | contribution used here |
|---|---|
| `correspondence_geom/postmortem_20260911/` | proved the through-origin slope misspecified and the `C4` baseline vacuous |
| `correspondence_geom/predesign_audit_20260911/` | derived `V(k,w) = F(k−τ, w+τ)` ⇒ a unit-slope ramp is **architecturally forced** in the padding-free interior — the reason `RESULTS.md` §7 refuses to read `α_h ≈ 1` as evidence of correspondence |
| `correspondence_geom/gate_audit_20260911/` | showed the two-stage gate was staged backwards; recommended the standalone random-only measurement that became ARCH-RATE-001 |

**All three are audits, not experiments. None is modified.**

## 5. Shared provenance

| key | sha256 | role here |
|---|---|---|
| `H2_seed0` / `POS_6b_seed0` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` | feature extractor **and** aggregation |
| `H2_seed1` / `POS_6b_seed1` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` | feature extractor **and** aggregation |
| `H2_seed2` / `POS_6b_seed2` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` | feature extractor **and** aggregation |
| `NEG_shift_none` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` | search-free control, `shift="none"`, reported separately |

The first three are the **same checkpoints** ARCH-RATE-001 used — there, as
feature extractors only, with their trained aggregations asserted **never
executed** 384 times. Here their aggregations are executed for the first time
under this construction. That asymmetry is the experiment.

Scenes `[5,7,8,9]` — identical to ARCH-RATE-001, and held out from both GEOM-001
`{27,0,31,6}` and GEOM-002 `{1,2,3,4}`. 40 scenes exist; 28 remain unused.

## 6. What this record adds

- **The trained arm of the campaign's first apples-to-apples comparison.** Same
  construction, same mask, same scenes, same statistic, same measurement path as
  the frozen 384-unit random population — the only change is which weights read
  the cost volume.
- **12/12 matched and 36/36 cross units outside the observed random population**,
  zero overlap, consistent across three checkpoints and four scenes.
- **A negative on the vertical control axis** (4/12 outside, overlap 0.667) and a
  **null on the search-free control** (0/4 outside): the separation is
  axis-specific and requires the candidate-dependent construction.
- **A warning about `R_h`:** it separates for the search-free control too, which
  has no slope at all. Only `α_h` distinguishes the behaviours.
- **A pre-execution freeze defect disclosed rather than repaired away** — the fit
  window — with the resolution bound to a criterion that predates any
  observation.

**Level D (geometric correspondence) remains `NOT-DEMONSTRATED`.** The claim
ceiling is enforced throughout: *outside the observed random population*, never
*impossible under random weights*, never *proves learned correspondence*.

**No follow-up experiment is designed or launched. The experiment owner audits
this result independently.**
