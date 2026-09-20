# RELATED RUNS — EXP-CORRESPONDENCE-GEOM-002

Every record below is preserved exactly as produced. **None was modified,
re-run, re-scored or deleted.** GEOM-001's verdict is unchanged.

---

## Lineage

```
Phase 1   EXP-007  right-image ablation      EPE 1.44 -> 47.96 .. 69.43
          EXP-010  reference cost volume degenerate: all 12 slices identical
        |
        v
EXP-CORRESPONDENCE-SHIFT-001 (Stage A)   DIAGNOSTIC-INVALID-AS-PREREGISTERED
        |
        v
EXP-CORRESPONDENCE-INDEX-001 x2          both EXPLORATORY-ONLY
  + provenance audit                     NO-CONFIRMATORY-EVIDENCE
EXP-CORRESPONDENCE-INDEX-002             NOT-DEMONSTRATED (null 0/12, order 12/12)
  + INDEX-003 design audit               DESIGN-INCONCLUSIVE
EXP-CORRESPONDENCE-ARCH-001              CASE-A-ARCHITECTURE-INDUCED
  + next-direction audit                 BYPASS-LEVEL-C
        |
        v
EXP-CORRESPONDENCE-GEOM-001              GEOMETRIC-CORRESPONDENCE-NOT-DEMONSTRATED
  phase2/diagnostics/correspondence_geom/20260911T111809Z/      0/12
  + successor audit                      SUCCESSOR-DESIGN-JUSTIFIED
        |
        v
EXP-CORRESPONDENCE-GEOM-002                              (this record)
  phase2/diagnostics/correspondence_geom/20260911T115917Z/
  SYNTHETIC-CORRESPONDENCE-NOT-DEMONSTRATED               3/12
```

---

## 1. GEOM-001 — the direct predecessor

**Record:** `phase2/diagnostics/correspondence_geom/20260911T111809Z/`
**Verdict (unchanged):** `GEOMETRIC-CORRESPONDENCE-NOT-DEMONSTRATED`, 0/12.

GEOM-001 perturbed a real stereo pair and died on a **representability defect**:
at `Δ = +48` its own frozen mask left 99.8 % of pixels unable to exhibit the
hypothesis, because positive translation drives the true disparity below the
candidate floor.

**GEOM-002 was designed to dissolve exactly that defect** by *setting* the
disparity instead of perturbing it. **It succeeded on that axis**: every level
`t = 16…96` maps to candidates 1…6, all representable, and no clipping occurred.

**GEOM-002 then failed for a different reason** — the response is not
axis-specific (`C3` fails 9/12) and monotonicity breaks at `t = 96` (`C2` fails
8/12). These are not the GEOM-001 failure recurring; they are new information.

**GEOM-001 was not modified, re-run or re-scored. Its frozen specs still verify.**

## 2. Successor audit — the design source

**Record:** `phase2/diagnostics/correspondence_geom/successor_audit_20260911/`
(`AUDIT.md`, `SUCCESSOR_OPTIONS.md`, `RECOMMENDATION.md`).

It derived from ground truth alone that the symmetric-translation class is
exhausted on this dataset (±48 unreachable on **all 40** scenes) and recommended
Option B, the synthetic self-pair.

**One error in that record was found and corrected before execution.**
`RECOMMENDATION.md` specified the right crop at `cols [96−t, 1136−t)`, i.e.
`a_R = a_L − t`, imposing `d = −t` — negative and unrepresentable. Corrected to
`a_R = a_L + t` in `construction_spec.json` and `DESIGN_AUDIT.md` §2.
**The audit record itself was left unmodified.**

## 3. What GEOM-002 inherited unchanged

From INDEX-001/002, ARCH-001 and GEOM-001: the four frozen checkpoints; the
`np.median` aggregate; the `shift="none"` search-free artifact baseline; the
crop-based no-fill construction; unit-level all-or-nothing decisions with **no
partial counts**; no p-values; the "magnitude is never a pass criterion" rule;
and the freeze order *artefacts hashed → preregistration → execution*.

New in GEOM-002: **held-out scenes** (`hailo_val` indices 1–4; **MEASURED** — 40
scenes exist, 4 used by GEOM-001, these 4 never used in any correspondence
experiment), and a construction requiring **no ground truth at all**.

## 4. Shared provenance

| key | sha256 | used by |
|---|---|---|
| `NEG_shift_none` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` | Stage A, INDEX-001 ×2, INDEX-002, ARCH-001, GEOM-001, **GEOM-002** |
| `POS_6b_seed0` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` | INDEX-001 ×2, INDEX-002, ARCH-001, GEOM-001, **GEOM-002** |
| `POS_6b_seed1` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` | same |
| `POS_6b_seed2` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` | same |

All four verified at load. No checkpoint has ever been retrained, modified or
added; no extra trained seed has ever been introduced; GEOM-002 added no weights
of any kind.

## 5. Standing state of the claim after this record

- **ESTABLISHED:** right-image dependence (A); binocular pathway engaged (B);
  final accuracy as a metric (F) — **all three hold for a model with provably
  zero disparity search** (EXP-010).
- **INCONCLUSIVE:** candidate-coordinate sensitivity (C) — four executions, no
  admissible determination; bypassed as structurally the wrong axis.
- **NOT-DEMONSTRATED:** geometric correspondence (D). GEOM-001 failed on a design
  defect; GEOM-002 failed on axis-specificity and monotonicity. **In neither case
  may it be concluded that correspondence is absent.**
- **NOT-DEMONSTRATED:** disparity search (E); provably **absent** for the Phase-1
  reference model.

**New from GEOM-002, MEASURED:** on synthetic pairs the positive checkpoints
produce a graded horizontal response rising with imposed disparity, correctly
signed at 12/12 — but the vertical arm, where no horizontal correspondence
exists, responds as strongly or more strongly at 9/12 units. The frozen
`Slope` statistic cannot separate a ramp from a step (`RESULTS.md` §6).

**Two protocol problems are recorded in `RESULTS.md` §7** — an erroneous `t = 0`
derivation in the preregistration, and a declared hard stop the harness failed to
enforce. Neither altered the verdict; both are reported rather than repaired.

**No successor is designed or launched.**
