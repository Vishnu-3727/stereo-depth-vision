# RELATED RUNS — EXP-CORRESPONDENCE-GEOM-001

**Design artifacts only. The experiment has NOT been executed.** No `RESULTS.md`,
no `results.json`, no `ENVIRONMENT.txt`, no `run.log`. Every record below is
preserved exactly as produced; **none was modified, re-run, re-scored or
deleted.**

First record under `phase2/diagnostics/correspondence_geom/`.

---

## Lineage

```
Phase 1
  experiments/EXP-007   right-image ablation      EPE 1.4422 -> 47.96 .. 69.43, D1 9.33 -> 98-99 %
  experiments/EXP-010   reference cost volume degenerate: all 12 slices identical
        |
        |  A and F established for a model with provably zero disparity search.
        |  Accuracy is not evidence of correspondence.
        v
EXP-CORRESPONDENCE-SHIFT-001  (Stage A)
  phase2/diagnostics/correspondence/20260910T142622Z/
  DIAGNOSTIC-INVALID-AS-PREREGISTERED — halted at Stage A, positives never run
        |
        v
EXP-CORRESPONDENCE-INDEX-001
  .../correspondence_index/20260910T164346Z/  ESTABLISHED      -> EXPLORATORY-ONLY
  .../correspondence_index/20260910T164326Z/  NOT-DEMONSTRATED -> EXPLORATORY-ONLY
  .../correspondence_index/audit_20260911T011113Z/   NO-CONFIRMATORY-EVIDENCE
        |
        v
EXP-CORRESPONDENCE-INDEX-002
  .../correspondence_index/20260911T015124Z/  NOT-DEMONSTRATED (null 0/12, order 12/12)
  .../correspondence_index/design_audit_20260911/  INDEX-003: DESIGN-INCONCLUSIVE
        |
        v
EXP-CORRESPONDENCE-ARCH-001
  .../correspondence_arch/design_20260911/    CLEAN-DESIGN-AVAILABLE
  .../correspondence_arch/20260911T025312Z/   CASE-A-ARCHITECTURE-INDUCED
        |
        v
next-direction audit
  .../correspondence/next_direction_audit_20260911/   BYPASS-LEVEL-C
    AUDIT.md, design.json, RELATED_RUNS.md  — AUTHORSHIP-UNKNOWN (see PROVENANCE_ANOMALY.md)
    ADDENDUM.md                             — session-32 auditor
        |
        v
EXP-CORRESPONDENCE-GEOM-001                              (this record)
  phase2/diagnostics/correspondence_geom/20260911T111809Z/
  GEOMETRIC-DESIGN-READY   ** preregistered, NOT executed **
```

---

## 1. Phase 1 — the standing warning

| record | finding | role here |
|---|---|---|
| `experiments/EXP-007` | right-image ablations cost up to **+89.96 D1 points** (EPE 1.4422 → 69.43) | Level A is **ESTABLISHED** — and useless as correspondence evidence, because it holds for the model EXP-010 proves is search-free. This is why GEOM-001 tests a **signed geometric** response, not a dependence. |
| `experiments/EXP-010` | the reference `Slice` ops concat zero padding then slice `[0:77]`, returning the features unchanged; all 12 slices identical | the algebraic basis of the `shift="none"` artifact control (§6 of the preregistration) |

**Not modified.**

---

## 2. Stage A — `EXP-CORRESPONDENCE-SHIFT-001`

**Record:** `phase2/diagnostics/correspondence/20260910T142622Z/`
**Verdict (unchanged):** `DIAGNOSTIC-INVALID-AS-PREREGISTERED`

GEOM-001 is the closest relative of this record and must be read against it.

### What Stage A got right, and GEOM-001 keeps

- the intervention class: right-image translation with a **matched vertical**
  comparator on an identical code path;
- the geometry: `−1 candidate per 16 px` — independently re-derived in
  §1.5 of the preregistration, by two routes, from source;
- the discipline: it **refused to patch a live preregistration** when its anchor
  failed, and did not run the positives.

### What failed, precisely

Its analytic anchor reasoned cost *volume* → softmax, **skipping the aggregation
between them**. All 12 slices identical does not imply all 12 *aggregated* slices
identical, because the D-axis convolutions are zero-padded. Predicted 5.5;
measured 3.645–4.272; response up to 0.6975 candidates from a bit-exactly
constant volume.

**Its slope hard-stop (§14.1) never fired** — the measured horizontal OLS slope
was `−0.001713 cand/px`, 2.74 % of anchor, and non-monotone. What halted Stage A
was a falsified absolute-value expectation, not a failure of the slope statistic.

### What GEOM-001 changes, and why

| | Stage A | GEOM-001 | reason |
|---|---|---|---|
| statistic | plain OLS over 5 raw points | **odd-part slope `S_h`** | Stage A's own record: *"The near-zero OLS slope is an artefact of fitting a line through a V."* The artifact is **even**; geometry is **odd**. |
| translation | zero fill | **crop-based, no fill** | a fill discontinuity sits on the left edge for `+Δ` and the right edge for `−Δ`, so it can leak into the odd part |
| sweep | ±16, ±32 | **±16, ±32, ±48** | 3 signed pairs instead of 2; `±64` rejected because the soft-argmin would rail |
| mask | derived from each checkpoint's own Δ=0 baseline | **single fixed GT + geometric mask**, counts frozen | removes checkpoint dependence from the sample |
| `shift="none"` role | anchor to be *predicted* | baseline to be **measured** | the anchor is what failed; the measurement is what is needed |

**Stage A's record was not modified, re-run or re-scored. Its numbers are cited;
its verdict stands.**

---

## 3. The Level-C campaign — why GEOM-001 bypasses it

| record | verdict (unchanged) | what GEOM-001 takes from it |
|---|---|---|
| INDEX-001 `20260910T164346Z` / `20260910T164326Z` | ESTABLISHED / NOT-DEMONSTRATED, both `EXPLORATORY-ONLY` | the checkpoints, scenes, `np.median` aggregate, and the algebraic-gate discipline |
| provenance audit `audit_20260911T011113Z` | `NO-CONFIRMATORY-EVIDENCE` | that a control must be **calibrated**, not assumed; and the mask-occupancy finding that `GT/16 ∈ [2,8]` is effectively `[2,4]` — the reason that band is demoted to a secondary here |
| INDEX-002 `20260911T015124Z` | `NOT-DEMONSTRATED` | the all-units evaluation level (§7: 12/12, no partial counts) |
| INDEX-003 design audit `design_audit_20260911` | `DESIGN-INCONCLUSIVE` | the equivariance argument, and that ordered candidate response is architecturally expected for arbitrary weights |
| ARCH-001 design + execution | `CLEAN-DESIGN-AVAILABLE` / `CASE-A-ARCHITECTURE-INDUCED` | the identity medians (3.84–5.83 candidates) used in §3.1 to **reject `±64` on rail risk**; and the lesson that a scale-free statistic can be near-unfalsifiable |

**The structural reason for the bypass**, from `cost_volume.py`:
`V[:,c,k,y,u] = Lf[c,y,u+k] − Rf[c,y,u]` — the right image enters with **no `k`
dependence**, and `standardise_across_disparity` removes a `k`-constant term
exactly. Every Level-C experiment intervened on the one axis along which the
variable of interest is constant. GEOM-001 intervenes on the image's horizontal
axis instead.

**None of these records was modified, re-run or re-scored.**

---

## 4. Next-direction audit — and its unresolved authorship

**Record:** `phase2/diagnostics/correspondence/next_direction_audit_20260911/`

- `AUDIT.md`, `design.json`, `RELATED_RUNS.md` — **AUTHORSHIP-UNKNOWN**
- `ADDENDUM.md` — session-32 auditor

Full forensics in `PROVENANCE_ANOMALY.md` in this directory. Summary: no git
authorship (`phase2/` untracked), no generator script, no writer tag in any
alternate data stream, no temp artifacts, no unrecorded experiment between
ARCH-001 (03:03Z) and the files (03:16Z). The content is entirely derived from
the public record chain. Classified **unexplained concurrent-agent write, not a
provenance defect**.

**Disposition of `design.json`: SUPERSEDED, not adopted and not rejected.** It is
preserved unmodified. This preregistration departs from it on five material
points (statistic, sweep, fill mechanism, decision rule, claim ceiling), each
argued from source or geometry in the preregistration, and each recorded in
`PROVENANCE_ANOMALY.md` §5.

**Convergence:** `DESIGN-CONVERGENCE-OBSERVED — NOT CLAIMED`. The agreement
between the unknown author and the session-32 auditor on `BYPASS-LEVEL-C` and on
`EXP-CORRESPONDENCE-GEOM-001` is recorded as an observation only. Independence is
not demonstrated, so it is **not** treated as replication and **no weight is
placed on it** anywhere in the preregistration.

---

## 5. Shared provenance

| key | sha256 | used by |
|---|---|---|
| `NEG_shift_none` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` | Stage A, INDEX-001 ×2, INDEX-002, ARCH-001, **GEOM-001 (planned)** |
| `POS_6b_seed0` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` | INDEX-001 ×2, INDEX-002, ARCH-001, **GEOM-001 (planned)** |
| `POS_6b_seed1` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` | same |
| `POS_6b_seed2` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` | same |

No checkpoint loaded by this design task. None has ever been retrained, modified
or added in this campaign; no extra trained seed has ever been introduced.
GEOM-001 adds **no** new weights of any kind.

---

## 6. Source files read for this design (read-only)

| file | what it established |
|---|---|
| `src/models/stereonet/cost_volume.py` | `V = Lf[u+k] − Rf[u]`; `shift="none"` is a no-op |
| `src/models/stereonet/feature_extractor.py` | fully convolutional, total stride 16, no pooling/normalisation/global op |
| `src/models/stereonet/aggregation.py` | 5 × `Conv3d(3×3×3, padding=1)` on the D axis |
| `src/models/stereonet/regression.py` | `soft_argmin` over the index grid `0…11` ⇒ candidate units |
| `phase2/models/scaled_regression.py` | readout has **0 parameters** |
| `src/datasets/kitti2015.py` | GT scale 256.0 ⇒ full-resolution px |
| `phase2/viz/core.py` | `correspondence(x_L, d) = x_L − d`, photometrically confirmed |

`src/` is bit-identical to the frozen Phase-1 tag and was not touched.

---

## 7. Standing state of the claim

- **ESTABLISHED:** right-image dependence (A); binocular pathway engaged (B);
  final accuracy as a metric (F) — **all three hold for a model with provably
  zero disparity search.**
- **INCONCLUSIVE:** candidate-coordinate sensitivity (C) — four executions, no
  admissible determination either way, and now bypassed as structurally the wrong
  axis.
- **NOT-DEMONSTRATED:** geometric correspondence (D) — **never tested on
  non-degenerate weights**; Stage A halted before its positives. This is the only
  level whose gap is missing *data* rather than a missing *identifiable test*.
- **NOT-DEMONSTRATED:** disparity search (E), provably **absent** for the Phase-1
  reference model.

GEOM-001 addresses D and nothing else. It is preregistered and **not executed**.
