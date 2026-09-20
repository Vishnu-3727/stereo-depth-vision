# RELATED RUNS — EXP-CORRESPONDENCE-ARCH-RATE-001

Every record below is preserved exactly as produced. **None was modified,
re-run, re-scored or deleted.** GEOM-001 and GEOM-002 verdicts unchanged.

First record under `phase2/diagnostics/correspondence_arch_rate/`.

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
        v
EXP-CORRESPONDENCE-ARCH-RATE-001                         (this record)
  phase2/diagnostics/correspondence_arch_rate/20260911T130507Z/
  QUALIFYING RATE = 0 / 384 = 0.0000        (a measurement, not a verdict)
```

## 1. GEOM-002 — the immediate predecessor

**Record:** `correspondence_geom/20260911T115917Z/`
**Formal status:** `PROCEDURALLY-COMPROMISED / NO-DETERMINATION`
(`correspondence_geom/postmortem_20260911/`).

ARCH-RATE-001 **inherits its construction unchanged** — the same self-pair crops,
the same corrected sign `a_R = a_L + t`, the same crop geometry, the same `t`
levels. The construction was never the problem; the postmortem found the
statistic and the procedure were.

**Three defects it repairs:**

| GEOM-002 defect | repair here |
|---|---|
| `C3` through-origin slope: a step of height `h` maps to `0.2308h`, indistinguishable from a ramp | **free-intercept** slope: every step maps to `α = 0` exactly |
| `C4` vacuous — baseline drawn from the degenerate `t = 0` regime | no baseline subtraction; `t = 0` is a level, not a reference |
| a declared hard stop fired and the harness did not halt | every hard stop halts; **and no hard stop encodes a predicted numeric value** |

**Not modified, not re-run, not re-scored. Its verdict stands.**

## 2. ARCH-001 — the same question, a different intervention

**Record:** `correspondence_arch/20260911T025312Z/` — `CASE-A-ARCHITECTURE-INDUCED`.

ARCH-001 asked whether candidate-axis *orbit* behaviour is weight-dependent, and
its own `RESULTS.md` §6 reported that its statistic had almost no power. Its
**descriptive** measurement — random aggregation orbit ranges 0.09–1.42 against
trained 5.36–6.39 — is qualitatively consistent with what ARCH-RATE-001 measures
here on a different intervention (median random response range 0.354 candidates).

**Cited as context only.** ARCH-001's numbers are not used in any statistic,
threshold or classification here, and its verdict is unchanged.

## 3. The audit chain that produced this design

| record | contribution |
|---|---|
| `correspondence_geom/postmortem_20260911/` | proved `C3` misspecified (`Slope` is a linear functional, non-injective on shape) and `C4` vacuous |
| `correspondence_geom/predesign_audit_20260911/` | derived `V(k,w) = F(k−τ, w+τ)`; showed a unit-slope ramp is architecturally forced in the interior; established that no curve statistic alone separates learned from architectural |
| `correspondence_geom/gate_audit_20260911/` | showed the two-stage gate is vacuous for a class of statistics, redundant under a separation rule, and **staged backwards**; recommended a standalone random-only measurement reporting a **rate** |

**All three are audits, not experiments. None is modified.**

## 4. What this experiment adds

**The first result in this campaign that is a measurement rather than a verdict.**

- **Rate 0/384.** No untrained aggregation reproduced the signature.
- **378/384 FLAT** — median response range 0.354 candidates across a 6-candidate
  sweep; largest `|α_h|` anywhere is 0.244 against an anchor of 1.0.
- **CLIPPED = 0** — the zero rate is not a railing artefact. The frozen
  precedence, which evaluated `CLIPPED` before every ordinary failure class,
  removed the one identified non-termination risk.
- **The preregistered PREDICTION is falsified.** §2 predicted a *high* rate from
  approximate interior equivariance. The finite `D = 12` padding boundary
  dominates instead — resolving the `UNKNOWN` tier against the prediction.

**The three-tier DERIVED / PREDICTED / UNKNOWN discipline is what made that
falsification survivable.** Frozen as an exact expectation — as GEOM-002's
`t = 0 → 5.5` anchor was — it would have fired a hard stop against correct
behaviour.

## 5. Shared provenance

| key | sha256 | role here |
|---|---|---|
| `POS_6b_seed0` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` | **feature extractor only** |
| `POS_6b_seed1` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` | **feature extractor only** |
| `POS_6b_seed2` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` | **feature extractor only** |

All three verified at load. **No trained aggregation was executed** — asserted
384/384 times against the trained parameter hash. `NEG_shift_none` was not used;
this experiment needs no search-free arm.

Scenes `[5,7,8,9]` — held out from **both** GEOM-001 `{27,0,31,6}` and GEOM-002
`{1,2,3,4}`. **MEASURED:** 40 scenes exist; 28 remain unused.

## 6. Standing state of the claim

- **ESTABLISHED:** right-image dependence; binocular pathway engaged; final
  accuracy as a metric — all three for a model with provably zero disparity
  search (EXP-010).
- **INCONCLUSIVE:** candidate-coordinate sensitivity (Level C), bypassed as
  structurally the wrong axis.
- **NOT-DEMONSTRATED:** geometric correspondence (Level D). **Unchanged by this
  experiment — the trained arm was never measured.**
- **NEW, MEASURED:** the synthetic horizontal correspondence signature is **not**
  routinely produced by arbitrary aggregation weights (0/384 under
  framework-default initialisation, three feature extractors, four held-out
  scenes).

**Per the frozen interpretation, a trained-arm experiment now becomes
warrantable — under a separate preregistration. None is designed or launched.**
