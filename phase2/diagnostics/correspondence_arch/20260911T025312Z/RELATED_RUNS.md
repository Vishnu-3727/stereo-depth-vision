# RELATED RUNS — EXP-CORRESPONDENCE-ARCH-001

Every record below is preserved exactly as produced. **None was modified,
re-run, re-scored or reinterpreted by this experiment.** The design in
`correspondence_arch/design_20260911/` was **not** rewritten after execution.

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
  .../correspondence_index/design_audit_20260911/
    INDEX-003 design audit. VERDICT: DESIGN-INCONCLUSIVE.
    Made an architecture-only control the mandatory precondition (its §9.14).
        |
        v
  .../correspondence_arch/design_20260911/
    ARCH-001 design + pre-flight. VERDICT: CLEAN-DESIGN-AVAILABLE.
        |
        v
EXP-CORRESPONDENCE-ARCH-001                             (this record)
  phase2/diagnostics/correspondence_arch/20260911T025312Z/
  VERDICT: CASE-A-ARCHITECTURE-INDUCED   (INSIDE 12/12)
  + a reported discrepancy: the frozen scale-free statistic had ~no power,
    and the amplitude dimension it discards separates completely.
```

---

## 1. Stage A — `EXP-CORRESPONDENCE-SHIFT-001`

**Record:** `phase2/diagnostics/correspondence/20260910T142622Z/`
**Verdict:** `DIAGNOSTIC-INVALID-AS-PREREGISTERED`

Its claim ceiling recorded, as **INFERRED**, that "the D-axis convolution
boundary effect is the cause — consistent with the architecture (five stacked
padded 3×3×3 convolutions over 12 candidates) … **but not isolated by an
ablation here**."

**ARCH-001 is that missing ablation.** It isolated the aggregation weights while
holding everything else bit-identical. The outcome is more nuanced than Stage A
anticipated: the conv stack *is* approximately equivariant for arbitrary weights
(§7 of `RESULTS.md`), yet random weights produce a readout response ~10× smaller
than trained (§6.2). Stage A's mechanism is confirmed at the aggregation output
and **not** confirmed at the readout amplitude.

**Not modified.**

---

## 2. `EXP-CORRESPONDENCE-INDEX-001` — two executions

**Records:** `.../20260910T164346Z/` (ESTABLISHED) and `.../20260910T164326Z/`
(NOT-DEMONSTRATED). Both `EXPLORATORY-ONLY` per the provenance audit.

ARCH-001 inherits from them unchanged: the four checkpoints, the four focus
scenes, the mask, the `π_m(k) = (k − m) mod 12` convention, the
`index_select`-for-every-arm requirement **including identity**, the
`d(x) = np.median(disparity_initial[mask])` aggregate, and the algebraic
negative-control gate.

The preregistration of `20260910T164326Z` §8 predicted the ordering *from the
architecture* and declined to predict a unit slope. ARCH-001's §7 confirms that
reasoning synthetically and extends it: the same equivariance holds for untrained
weights.

**Neither record was modified, re-run or re-scored.**

---

## 3. Provenance audit — `audit_20260911T011113Z`

**Record:** `phase2/diagnostics/correspondence_index/audit_20260911T011113Z/`

Established that a control must be calibrated, not assumed. ARCH-001 carries that
forward: 16 random weight-sets giving 120 internal baseline pairs per unit, all
hashes frozen before the preregistration, and a decision rule fixed before any
output. It also inherits the audit's discipline of reporting a design failure
rather than repairing it — which §6 of `RESULTS.md` does.

**Not modified.**

---

## 4. `EXP-CORRESPONDENCE-INDEX-002` — the numerical predecessor

**Record:** `phase2/diagnostics/correspondence_index/20260911T015124Z/`
**Verdict:** `CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED`

### Cross-run reproduction (determinism check)

ARCH-001's trained orbit positions `m = 11, 0, 1, 2` are exactly INDEX-002's four
arms `−1, 0, +1, +2`:

| checkpoint | scene | m=11 | m=0 | m=1 | m=2 |
|---|---|---:|---:|---:|---:|
| POS_6b_seed0 | 000187_10 | 3.8579 | 5.6118 | 7.5528 | 8.3431 |
| POS_6b_seed0 | 000160_10 | 3.9881 | 5.8301 | 7.7237 | 8.4018 |
| POS_6b_seed0 | 000191_10 | 3.8441 | 5.5845 | 7.5619 | 8.4345 |
| POS_6b_seed0 | 000166_10 | 3.3378 | 5.1169 | 7.0400 | 8.2469 |

Identical to INDEX-002's published values. Determinism held across separate
harnesses and separate experiments.

### What ARCH-001 adds

INDEX-002 measured only `m ∈ {11, 0, 1, 2}`. ARCH-001 measured all twelve and
found the orbit is **not monotone**: it peaks at `m = 2…7` and reaches its
**minimum at `m = 10` in 12/12 units**, then partially recovers at `m = 11`.
INDEX-002's window was a local rising segment of a wrapped, non-monotone orbit.

### What changed and what did not

| element | INDEX-002 | ARCH-001 |
|---|---|---|
| intervention | `π_m(k) = (k−m) mod 12` | **unchanged** |
| arms | `m ∈ {−1,0,+1,+2}` | **full orbit `m = 0…11`** |
| response `d(x)` | `np.median(disparity_initial[mask])` | **unchanged** |
| mask, scenes, checkpoints | frozen | **unchanged** |
| what varies | the permutation | **the aggregation weights** |
| null | 512 random-permutation triples | **none — paired trained-vs-random, no p-value** |
| verdict logic | `p ≤ 0.01` + 99 % envelope | **descriptive paired containment** |

**Not modified, not re-run, not re-scored.** Its `null_pass 0/12` remains its
result.

---

## 5. INDEX-003 design audit — `design_audit_20260911`

**Record:** `phase2/diagnostics/correspondence_index/design_audit_20260911/`
**Verdict:** `DESIGN-INCONCLUSIVE`

It blocked INDEX-003 because the ordered response is architecturally expected,
and made an architecture-only control the **mandatory precondition** (§9.14),
with this interpretation table fixed in advance:

| trained | untrained | conclusion |
|:---:|:---:|---|
| PASS | PASS | the ordering is architectural; it carries no information about learning |
| PASS | FAIL | the ordering is weight-dependent |
| FAIL | either | not distinguishable under this design |

**ARCH-001 is that precondition, executed.** Its `CASE A` corresponds to the
first row **as measured by the frozen statistic** — with the §6 caveat that the
statistic had almost no power, so this row should not be treated as settled.

### One prediction of that audit held, one did not

**Held.** The audit rejected circular rank TV for this comparison on floor-effect
grounds. Measured here: trained TV 22–32, random TV 22–62 (median 24) — complete
overlap, no separation. The rejection was correct.

**Did not hold.** The audit's §7 argued the trained and random orbits would both
be wrapped ramps differing only in phase and sharpness, and chose a scale-free
statistic on that basis. Measured here: random orbits are near-flat (range
0.09–1.42) while trained orbits swing 5.36–6.39. The scale-free statistic was
therefore the wrong instrument, and §6 of `RESULTS.md` reports that rather than
repairing it.

**The audit record was not modified.**

---

## 6. ARCH-001 design record — `correspondence_arch/design_20260911/`

**Verdict:** `CLEAN-DESIGN-AVAILABLE`, with two flagged changes (TV rejected as
primary; 16 control seeds instead of 3), both authorised before execution.

Executed exactly as frozen. `preflight_architecture_analysis.py` was copied here
unchanged and re-run; its output is preserved in
`preflight_architecture_analysis.log`.

**Not rewritten after execution**, as `PREREGISTRATION.md` §17 requires.

---

## 7. Shared provenance

| key | sha256 | used by |
|---|---|---|
| `NEG_shift_none` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` | Stage A, INDEX-001 ×2, INDEX-002, ARCH-001 |
| `POS_6b_seed0` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` | INDEX-001 ×2, INDEX-002, ARCH-001 |
| `POS_6b_seed1` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` | INDEX-001 ×2, INDEX-002, ARCH-001 |
| `POS_6b_seed2` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` | INDEX-001 ×2, INDEX-002, ARCH-001 |

All four verified at load, hard stop on mismatch. No checkpoint has ever been
retrained, modified or added in this line, and no extra trained seed has ever
been introduced.

ARCH-001 added **16 untrained random aggregations** (control seeds 0–15). They
are not checkpoints, were never saved as model state, received no training, and
exist only as a within-run control. Their parameter hashes are frozen in
`random_initialization_metadata.json` (sha256 `945e3283…`).

---

## 8. Standing state of the claim after this record

- **Established:** right-image dependence. The algebraic negative-control gate —
  now confirmed for random aggregation weights as well as trained, so the
  invariance is a property of the cost volume, not the weights. The ordered
  four-arm response, reproduced a fourth time at 12/12 trained units.
- **New, descriptive:** the trained 12-position orbit is non-monotone, minimum at
  `m = 10` in 12/12 units; INDEX-002's window was a local rising segment.
  Trained orbit amplitude exceeds every one of 192 random orbits.
- **Unresolved:** candidate-coordinate sensitivity (level C). INDEX-002's
  preregistered test did not pass; ARCH-001 returned `CASE A` under a statistic
  that §6 shows had almost no power. **Neither result settles it**, and `CASE A`
  must not be read as evidence against it.
- **Not justified:** geometric correspondence (level D) and genuine disparity
  search (level E). Neither is designed, preregistered nor launched.
- **Open design question for any successor:** an amplitude-sensitive statistic is
  the obvious next instrument, but ARCH-001's amplitude numbers are now public,
  so it cannot be preregistered predictively on this data. Any such experiment
  must confront that directly.
