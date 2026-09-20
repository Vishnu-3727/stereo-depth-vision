# RELATED RUNS — EXP-CORRESPONDENCE-INDEX-002

This record is one of several executions in the Phase-2 candidate-axis /
correspondence line. **None of the records below was modified, re-run, re-scored
or superseded by this experiment.** They are preserved exactly as produced. This
file exists so that no reader of any single directory believes it is the only
execution — a provenance defect the INDEX-001 audit identified in both of that
experiment's runs.

---

## Lineage

```
EXP-CORRESPONDENCE-SHIFT-001            (Stage A — image translation sweep)
  phase2/diagnostics/correspondence/20260910T142622Z/
  VERDICT: DIAGNOSTIC-INVALID-AS-PREREGISTERED — halted at Stage A
        |
        |  Stage A showed a degenerate model still produces a non-uniform,
        |  input-dependent candidate curve, so response magnitude alone
        |  cannot evidence correspondence. Motivated intervening on the
        |  candidate axis directly rather than on the images.
        v
EXP-CORRESPONDENCE-INDEX-001            (single random permutation as control)
  phase2/diagnostics/correspondence_index/20260910T164346Z/   ("Run 2" by name)
    VERDICT: CANDIDATE-COORDINATE-SENSITIVITY-ESTABLISHED
  phase2/diagnostics/correspondence_index/20260910T164326Z/   ("Run 1" by name)
    VERDICT: CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED
        |
        v
  phase2/diagnostics/correspondence_index/audit_20260911T011113Z/
    Provenance audit of both runs.
    Both classified EXPLORATORY-ONLY. Overall: NO-CONFIRMATORY-EVIDENCE.
        |
        |  The audit's decisive finding: each run compared the ordered
        |  response to ONE uncalibrated random permutation with no null
        |  distribution, so criterion 6 had no demonstrated discriminative
        |  power. Its first recommendation was to replace the single draw
        |  with an empirical permutation null.
        v
EXP-CORRESPONDENCE-INDEX-002            (this record)
  phase2/diagnostics/correspondence_index/20260911T015124Z/
  VERDICT: CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED
```

---

## 1. Stage A — `EXP-CORRESPONDENCE-SHIFT-001`

**Record:** `phase2/diagnostics/correspondence/20260910T142622Z/`
**Verdict:** `DIAGNOSTIC-INVALID-AS-PREREGISTERED — STOP, NEW PREREGISTRATION REQUIRED`
**Status:** halted at Stage A; positive controls were deliberately never run.

Ran a horizontal + vertical image-translation sweep on the trained, converged
`shift="none" + standardised` negative control. The preregistration predicted a
uniform softmax and a constant soft-argmin of 5.5 for that degenerate model.
Observed instead: non-uniform, input-dependent candidate curves (soft-argmin
3.90 ± 1.96) responding to right-image translation by up to 0.70 candidates.

**What this experiment inherits from it.** Stage A established that a provably
degenerate cost volume still yields a structured, input-dependent candidate
response, because the aggregation stage's padded disparity-axis convolutions
create the structure. Therefore **response magnitude alone can never evidence
correspondence**, which is why INDEX-001 and INDEX-002 intervene directly on the
candidate axis and why both require an algebraic negative control rather than a
statistical one.

**Not modified by this experiment.**

---

## 2. `EXP-CORRESPONDENCE-INDEX-001` — two executions

Both used the identical model, checkpoints, scenes, mask, endpoint, ordered arms
and algebraic gate that this experiment uses. They differ from each other **only**
in the single random permutation each froze as its order-destruction control, and
they reached opposite verdicts on that basis.

### 2a. `20260910T164346Z`

**Verdict:** `CANDIDATE-COORDINATE-SENSITIVITY-ESTABLISHED`
**Audit status:** `EXPLORATORY-ONLY`
Random permutation `[7, 2, 10, 0, 5, 11, 1, 8, 4, 9, 6, 3]` — no generator or
seed recorded, not reproducible from the record. Its harness computes no success
criterion; the verdict is hand-written prose.

### 2b. `20260910T164326Z`

**Verdict:** `CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED`
**Audit status:** `EXPLORATORY-ONLY`
Random permutation `[1, 10, 9, 5, 3, 8, 11, 0, 6, 7, 2, 4]`, reproducible from
`numpy.random.default_rng(20260910).permutation(12)`. Its criterion-6 rule
(`|Δrand − Δ+1| < 0.5·|Δ+1|`) appears nowhere in its own preregistration and was
authored after the other run's verdict existed.

### What INDEX-002 carries over unchanged

- the four ordered arms and the `π_m(k) = (k − m) mod 12` convention;
- the four checkpoints (byte-identical sha256) and the four focus scenes;
- the mask `GT valid ∧ GT/16 ∈ [2,8]`, **deliberately not altered**, so this
  experiment remains comparable to INDEX-001;
- the per-arm aggregate `d(x) = np.median(disparity_initial[mask])`;
- the `index_select`-for-every-arm requirement, including identity;
- the algebraic negative-control gate and the level-C claim ceiling.

### What INDEX-002 changes, and why

| Change | Reason |
|---|---|
| 1 random permutation → **512 null triples** (1536 draws) | the audit's decisive finding: a single draw gives no null distribution and no discriminative power |
| criterion-6 prose → **frozen finite-sample rank test in code** | INDEX-001's operative rule existed only in a harness written after another run's verdict |
| magnitude-vs-one-draw comparison → **preregistered `S_order` with an exact two-sided p and a 99 % envelope** | no threshold may be invented after results |
| seed undocumented / partly documented → **seed 20260911 and a sha256 of the full permutation file, both recorded in the preregistration before execution** | INDEX-001's earlier run could not regenerate its own control |

### Cross-run numerical agreement

INDEX-001 and INDEX-002 were run from separate harnesses on the same checkpoints
and scenes. Their ordered arms agree: identity medians per checkpoint (pooled
INDEX-001 vs the four per-scene values here) and the ordered directional pattern
`d(+2) > d(+1) > d(identity) > d(−1)` reproduce at 12/12 units in all three
executions. Determinism holds across all of them.

**Neither INDEX-001 record was modified, re-run or re-scored by this experiment.**

---

## 3. Audit record — `audit_20260911T011113Z`

**Record:** `phase2/diagnostics/correspondence_index/audit_20260911T011113Z/`
Read-only provenance audit of both INDEX-001 executions.

Findings this experiment acts on:

1. Both runs `EXPLORATORY-ONLY`; overall `NO-CONFIRMATORY-EVIDENCE`.
2. The directory names of INDEX-001 invert the actual execution order.
3. Criterion 6 in both runs rested on one uncalibrated permutation with no null.
4. The retained band is nominally `[2,8]` but effectively `[2,4]` (≈99.4 % of
   retained pixels). **Reported again here as null-quality metadata; the mask was
   NOT changed**, because changing it would confound comparison with INDEX-001.
5. The right-referenced / left-referenced discrepancy remains open. It is
   documented explicitly in this preregistration (§4) and **not corrected**.

The audit recommended exactly the design executed here, and recommended this
`RELATED_RUNS.md`.

**The audit record was not modified.**

---

## 4. Checkpoint provenance shared across all records

| key | sha256 | used by |
|---|---|---|
| `NEG_shift_none` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` | Stage A, INDEX-001 ×2, INDEX-002 |
| `POS_6b_seed0` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` | INDEX-001 ×2, INDEX-002 |
| `POS_6b_seed1` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` | INDEX-001 ×2, INDEX-002 |
| `POS_6b_seed2` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` | INDEX-001 ×2, INDEX-002 |

No checkpoint was retrained, modified, or added. No extra seed was introduced at
any point in this line.

---

## 5. Standing state of the claim after this record

Level C — candidate-coordinate sensitivity — is **not** established. Two
INDEX-001 executions are exploratory and this preregistered test did not pass.

What is established across all three executions, on agreeing numbers and against
threshold-free preregistered criteria: the algebraic negative-control gate
(bit-exact 0.0 invariance of the degenerate model under every permutation
tested — 6 160 arms in this record alone) and the ordered directional response
(12/12 units, in every execution).

Levels D (geometric correspondence) and E (genuine disparity search) are neither
established nor claimed anywhere in this line.
