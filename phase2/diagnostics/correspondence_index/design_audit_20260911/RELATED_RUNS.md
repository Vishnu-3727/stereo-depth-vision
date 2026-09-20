# RELATED RUNS — DESIGN AUDIT FOR EXP-CORRESPONDENCE-INDEX-003

This is a **design-audit record**, not an experiment. No `RESULTS.md` exists
because nothing was executed. Every record listed below is preserved exactly as
produced; **none was modified, re-run, re-scored or reinterpreted by this
audit.**

---

## Lineage

```
EXP-CORRESPONDENCE-SHIFT-001            (Stage A — image-translation sweep)
  phase2/diagnostics/correspondence/20260910T142622Z/
  VERDICT: DIAGNOSTIC-INVALID-AS-PREREGISTERED — halted at Stage A
        |
        v
EXP-CORRESPONDENCE-INDEX-001            (single random permutation as control)
  .../correspondence_index/20260910T164346Z/   ESTABLISHED       -> EXPLORATORY-ONLY
  .../correspondence_index/20260910T164326Z/   NOT-DEMONSTRATED  -> EXPLORATORY-ONLY
        |
        v
  .../correspondence_index/audit_20260911T011113Z/
    Provenance audit. Overall: NO-CONFIRMATORY-EVIDENCE.
        |
        v
EXP-CORRESPONDENCE-INDEX-002            (empirical permutation null, M = 512)
  .../correspondence_index/20260911T015124Z/
  VERDICT: CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED
    null_pass 0/12, order_pass 12/12
        |
        v
DESIGN AUDIT — EXP-CORRESPONDENCE-INDEX-003            (this record)
  .../correspondence_index/design_audit_20260911/
  VERDICT: DESIGN-INCONCLUSIVE        ** nothing executed **
```

---

## 1. Stage A — `EXP-CORRESPONDENCE-SHIFT-001`

**Record:** `phase2/diagnostics/correspondence/20260910T142622Z/`
**Verdict:** `DIAGNOSTIC-INVALID-AS-PREREGISTERED`

Established that a provably degenerate cost volume still yields a structured,
input-dependent candidate response, because the aggregation's padded
disparity-axis convolutions create that structure. **This audit's decisive
finding (§2 of `DESIGN_AUDIT.md`) is the same mechanism carried one step
further**: those convolutions are not merely structure-generating, they are
*translation-equivariant in the interior*, which is why a cyclic shift of the
candidate axis produces an ordered response for arbitrary weights.

**Not modified.**

---

## 2. `EXP-CORRESPONDENCE-INDEX-001` — two executions

**Records:** `.../20260910T164346Z/` (ESTABLISHED) and `.../20260910T164326Z/`
(NOT-DEMONSTRATED). Both classified `EXPLORATORY-ONLY` by the provenance audit.

**What this audit takes from them.** The preregistration of `20260910T164326Z`
§8 already derived the ordered prediction *from the architecture* — five
disparity-axis 3-tap convolutions with zero padding, only `k ∈ {5,6}`
padding-free, "the stack is not cyclically equivariant and `Δd = m` is not
predicted". That reasoning is confirmed here from source and is the basis of the
`DESIGN-INCONCLUSIVE` verdict: the ordering was predicted from the architecture
before any candidate-axis data existed, which is precisely why observing it
cannot discriminate learned tracking.

**Neither record was modified, re-run or re-scored.**

---

## 3. Provenance audit — `audit_20260911T011113Z`

**Record:** `phase2/diagnostics/correspondence_index/audit_20260911T011113Z/`

Its first recommendation — replace the single-draw control with an empirical
permutation null — was executed as INDEX-002. Its finding that a control must be
*calibrated* rather than assumed is extended here to a further requirement: a
null must also be **structurally matched** to the family it is testing (§3, §5 of
`DESIGN_AUDIT.md`).

**Not modified.**

---

## 4. `EXP-CORRESPONDENCE-INDEX-002` — the direct predecessor

**Record:** `phase2/diagnostics/correspondence_index/20260911T015124Z/`
**Verdict:** `CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED`

### Values this audit read from it (already published; no new output generated)

| quantity | value |
|---|---|
| positive units | 12 = 3 checkpoints × 4 scenes |
| ordered arms, all 12 units | `d(−1) < d(0) < d(+1) < d(+2)` strictly |
| `S_order` range | +4.11 … +6.83 |
| null p99 range | +5.00 … +6.87 |
| `null_pass` / `order_pass` | 0/12 / 12/12 |
| `d(identity)` vs random `d(π)` | identity sits low (e.g. 5.6118 vs random median 6.7718) |
| null construction | 1 536 i.i.d. uniform `S_12` draws, 512 arbitrary disjoint triples |
| measured throughput | 24 640 arms in 264.1 s ⇒ ≈0.0107 s/arm |

### What this audit concludes about it

1. **Its null cannot be reused for an ordering test** (§3 of `DESIGN_AUDIT.md`).
   The ordered arms are powers of one generator; an INDEX-002 triple is three
   unrelated permutations; only `1/12` of uniform `S_12` draws are 12-cycles and
   their powers were never computed. This is a statement about *fitness for a
   different purpose*, **not** a criticism of INDEX-002, whose own magnitude
   statistic it served correctly and whose verdict stands unchanged.
2. **Its ordered-side values are now public**, which disqualifies any INDEX-003
   statistic confined to those four arms from being preregistered predictively
   (§1.2).
3. **Its α = 0.01 and its exact two-sided rank formula are carried forward
   unchanged**, together with its `d(x) = np.median(disparity_initial[mask])`
   aggregation, its mask, its checkpoints, its scenes and its algebraic
   `shift="none"` gate. Its `M = 64`-unreachable-threshold lesson is applied in
   §8 (`M ≥ 199` mandatory, `M = 1000` specified).

**Not modified, not re-run, not re-scored.** `null_pass 0/12` remains its result.

---

## 5. Shared provenance across the whole line

| key | sha256 | used by |
|---|---|---|
| `NEG_shift_none` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` | Stage A, INDEX-001 ×2, INDEX-002 |
| `POS_6b_seed0` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` | INDEX-001 ×2, INDEX-002 |
| `POS_6b_seed1` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` | INDEX-001 ×2, INDEX-002 |
| `POS_6b_seed2` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` | INDEX-001 ×2, INDEX-002 |

No checkpoint was loaded by this audit. No checkpoint was retrained, modified or
added anywhere in this line, and no extra seed was ever introduced.

The proposed INDEX-003 would reuse these same four checkpoints **plus** one
architecture-only arm with randomly initialised, untrained weights (§9.14 of
`DESIGN_AUDIT.md`) — the sole element requiring authorisation beyond the current
frozen protocol.

---

## 6. Source files read by this audit (read-only)

| file | what was verified |
|---|---|
| `src/models/stereonet/aggregation.py` | 4 × `Conv3d(3×3×3, padding=1)` + LeakyReLU, then `Conv3d(→1, 3×3×3, padding=1)`; 5 three-tap convolutions along the candidate axis |
| `src/models/stereonet/regression.py` | readout is `soft_argmin` over the candidate index |
| `src/models/stereonet/cost_volume.py` | `shift="left"` gives level `k` at column `u` = `left_feat[u+k] − right_feat[u]`; `shift="none"` is a no-op |
| `20260911T015124Z/results.json` | unit structure, ordered-arm values, null statistics, throughput |
| `20260911T015124Z/permutations.json` | INDEX-002 null construction |

`src/` is bit-identical to the frozen Phase-1 tag and was not touched.

---

## 7. Standing state of the claim after this audit

Unchanged by this audit, which produced no evidence:

- **Established:** right-image dependence; the algebraic negative-control gate
  (bit-exact 0.0 under every permutation tested); the ordered directional
  response, reproduced at 12/12 units in three independent executions.
- **Unresolved:** candidate-coordinate sensitivity (level C). INDEX-002's
  preregistered test did not pass and is not overturned here.
- **Not justified:** geometric correspondence (level D) and genuine disparity
  search (level E). Neither is designed, preregistered nor launched.

**New, and the reason for the `DESIGN-INCONCLUSIVE` verdict:** the ordered
response is an expected consequence of the aggregation's architecture and is
predicted for arbitrary weights, so an ordering-vs-permutation test cannot
discriminate learned tracking without an untrained-weights control. That
expectation is **INFERRED from source and explicitly untested** — no model was
instantiated by this audit.
