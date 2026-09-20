# DESIGN AUDIT — EXP-CORRESPONDENCE-INDEX-003 (proposed)

**Ordering-sensitive test of candidate-axis structure against an empirical
permutation null.**

Audit date: 2026-09-11. Record
`phase2/diagnostics/correspondence_index/design_audit_20260911/`.

**This is a design audit. Nothing was executed.** No training, no inference, no
model was instantiated, no experimental output was generated or inspected. The
only computations performed are permutation algebra, counting arithmetic, and
reads of *already published* numbers from the INDEX-002 record. No historical
file was modified; INDEX-001, INDEX-002, Stage A and the provenance audit are
untouched.

Evidence tags: **MEASURED** (read from source or from an existing record),
**DERIVED** (arithmetic/algebra on measured values), **INFERRED**
(interpretation), **UNKNOWN** (not determinable without an experiment).

---

## FINAL DESIGN VERDICT

```
DESIGN-INCONCLUSIVE
```

An ordering statistic **can** be written down, and a structurally matched,
non-circular permutation null **does** exist (§4, §5, §9). But the resulting test
cannot be made discriminative for the question the investigation is actually
asking, because the ordered response is an **expected consequence of the
aggregation's architecture** — five zero-padded 3-tap convolutions along the
candidate axis are translation-equivariant in the interior — and would therefore
be produced by **arbitrary weights, trained or not**. Separating "the network
learned candidate-coordinate tracking" from "convolution is equivariant" requires
an architecture-only control that lies outside the currently frozen protocol.

The full conditional preregistration is given in §9 so that it is ready if that
control is authorised. **It must not be executed on the strength of this audit
alone.**

---

## 1. WHAT IS ACTUALLY BEING PROPOSED, AND WHAT IS ALREADY KNOWN

### 1.1 The unit structure — correcting the brief

**MEASURED**, from `20260911T015124Z/results.json`. The brief states "12 positive
units: 3 seeds × 4 checkpoints". The count is right, the factors are not:

```
distinct checkpoints : POS_6b_seed0, POS_6b_seed1, POS_6b_seed2   (3)
distinct seeds       : 0, 1, 2                                     (3)
distinct scenes      : 000160_10, 000166_10, 000187_10, 000191_10  (4)

=> 3 checkpoints (one per seed) x 4 scenes = 12 positive units
```

There are **three** checkpoints, not four; the seeds *index* the checkpoints
rather than multiplying them. This matters for §7 (independence), because the
dependence structure is "3 weight sets × 4 images", not "3 × 4 independent
draws".

### 1.2 The ordered-side answer is already published

**MEASURED.** All four ordered arms are recorded for all 12 units in
`20260911T015124Z/results.json`, and

```
d(-1) < d(0) < d(+1) < d(+2)   holds strictly at 12 / 12 units
```

Example rows (already in the record):

| checkpoint | scene | d(−1) | d(0) | d(+1) | d(+2) |
|---|---|---:|---:|---:|---:|
| POS_6b_seed0 | 000187_10 | 3.8579 | 5.6118 | 7.5528 | 8.3431 |
| POS_6b_seed0 | 000160_10 | 3.9881 | 5.8301 | 7.7237 | 8.4018 |
| POS_6b_seed0 | 000191_10 | 3.8441 | 5.5845 | 7.5619 | 8.4345 |

**Consequence for the design.** Any statistic computed **only from these four
arms** has a value that is *already determined by a published record*. Such a
statistic cannot be preregistered in the predictive sense; the most it can be is
a **null calibration of an already-measured quantity**. That is a legitimate
procedure, but its evidential status is strictly weaker than a prediction made
before measurement, and any preregistration must say so in those words rather
than presenting the result as confirmatory. This disqualifies statistics S1–S4
of §4 on their own and is the first reason to prefer a statistic that uses
offsets never yet computed (§4.5, §4.6).

### 1.3 What is genuinely unknown

**UNKNOWN.** `d(π_m)` for `m ∈ {3, …, 10}` has never been computed in any
record. The null side — the behaviour of *structurally matched* random families —
has never been computed. Those are the only quantities an INDEX-003 could
actually discover.

---

## 2. THE ARCHITECTURE — the decisive fact

**MEASURED**, from `src/models/stereonet/aggregation.py`:

```python
for _ in range(num_layers):                                  # num_layers = 4
    layers.append(nn.Conv3d(c_in, channels, 3, stride=1, padding=1))
    layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=False))
self.to_cost = nn.Conv3d(channels, 1, 3, stride=1, padding=1)
```

Five `Conv3d` layers, kernel `3×3×3`, `padding=1`, along `(D, H, W)` with
`D = 12`. Receptive field along the candidate axis: `1 + 5·2 = 11`, so output
candidate `k` depends on inputs `[k−5, k+5]` and only `k ∈ {5, 6}` are
padding-free — consistent with INDEX-001 §8.

**MEASURED**, from `src/models/stereonet/regression.py`: the readout is
`soft_argmin` over the candidate index, `Σ_d softmax(−cost)_d · d`.

**DERIVED — the equivariance argument.** A `Conv3d` with `padding=1` is
translation-equivariant along `D` **except within the padding-affected boundary
band**, and this is a property of the *operator*, not of its weights. For any
weight set `w` and any interior region,

```
Agg_w( shift_m(V) )  ≈  shift_m( Agg_w(V) )
soft_argmin( shift_m(c) )  =  soft_argmin(c) + m
=>  d(π_m)  ≈  d(π_0) + m
```

So a monotone increase of the soft-argmin with the cyclic offset `m` is what the
architecture *predicts for arbitrary weights*. INDEX-001's own preregistration
(§8) reached the same conclusion from the same structure and correctly declined
to predict a unit slope, because the zero padding breaks exact equivariance. The
measured deltas are indeed **not** `≈ m` — `d(+1) − d(0)` is +1.2 … +1.9 and
`d(+2) − d(0)` is +1.8 … +3.1 (**MEASURED**, INDEX-002 record) — but the *sign
and the ordering* follow from the operator, not from anything learned.

**INFERRED, and explicitly UNTESTED.** A randomly-initialised network of this
architecture, with `shift="left"` so the volume is non-degenerate, would be
expected to reproduce the same ordering. **No such model was instantiated and no
such measurement exists.** This is a prediction, offered as the design's decisive
open question — not as a result.

**Consequence.** An ordering-vs-permutation test asks whether the natural
candidate order behaves differently from a scrambled one. The aggregation is
*defined* as a local operator on the natural candidate order. The test therefore
confirms a near-theorem about the architecture. It would very likely pass, and
passing would carry almost no information about whether the trained weights
implement anything resembling candidate-coordinate tracking, let alone
correspondence.

---

## 3. IS THE INDEX-002 NULL REUSABLE FOR AN ORDERING TEST?

**No. Identified now, before any execution, as the brief requires.**

**MEASURED**, from `20260911T015124Z/permutations.json` and `results.json`: the
frozen null is 1 536 permutations drawn i.i.d. uniform from `S_12` via
`numpy.random.default_rng(20260911).permutation(12)`, partitioned into 512
**arbitrary disjoint triples**, each triple standing in for `(+2, +1, −1)` with a
shared `d(identity)`.

Three structural mismatches make it unusable for an ordering test (**DERIVED**):

1. **Wrong object.** The ordered arms are the **powers of a single generator**
   `c: k ↦ k−1` — a one-parameter cyclic family. An INDEX-002 triple is three
   *unrelated* permutations. Comparing a one-parameter family to unrelated
   triples confounds "the natural order matters" with "a one-parameter family is
   smoother than three unrelated draws", which is true by construction for any
   local operator.
2. **Wrong cycle type.** Only `11!/12! = 1/12` of uniform draws from `S_12` are
   single 12-cycles. The ordered arms have cycle types `[12]` (m = ±1), `[6,6]`
   (m = +2) and `[1¹²]` (identity) — **DERIVED**, computed in this audit. An
   INDEX-002 triple matches none of that.
3. **Powers were never computed.** Even for the ≈1/12 of draws that happen to be
   12-cycles, `γ²`, `γ³`, … were never evaluated, so no power-family statistic
   can be recovered from the record.

**Therefore INDEX-002's null cannot be reused, re-grouped or re-scored for an
ordering test.** Any INDEX-003 needs a new, purpose-built null. INDEX-002's
record stands exactly as produced; nothing in it is re-interpreted here.

One element **is** reusable and should be kept: INDEX-002's convention of a
**single shared `d(identity)`** across the ordered family and every null family,
since `identity` is a member of every power-family by definition.

---

## 4. CANDIDATE STATISTICS — audited against the ten required criteria

Notation: for a generator `g` (a 12-cycle), the family is `g^{-m}`, `m ∈ Z_12`;
`s_m = d(g^{-m})` is the masked-median response. The ordered family is `g = c`
with `c(k) = k − 1 mod 12`, giving exactly INDEX-001/002's `π_m(k) = (k−m) mod 12`.

### 4.1 S1 — strict-order indicator on four arms

`S1 = 1[ d(+2) > d(+1) > d(0) > d(−1) ]`

| # | criterion | assessment |
|---|---|---|
| 1 | definition | exact, boolean |
| 2 | information used | 4 values, ordinal only |
| 3 | permutation invariance | not invariant; depends on label assignment |
| 4 | null analogue | requires assigning labels −1,0,+1,+2 to a null family |
| 5 | exact null | yes, if the null family is label-matched |
| 6 | **leaks expected sign/order** | **yes — hard-codes the observed direction** |
| 7 | **restates the hypothesis** | **yes — this is literally the observed pattern** |
| 8 | directional / two-sided | one-sided by construction |
| 9 | ties | strict inequality; any tie ⇒ 0 |
| 10 | identical for ordered & null | yes |

**REJECT.** Criteria 6 and 7 fail outright. This is the circularity the brief
names. It is also the statistic whose value is already published (§1.2).

### 4.2 S2 — signed Spearman ρ between `m` and `s_m` over `m ∈ {−1,0,+1,+2}`

| # | criterion | assessment |
|---|---|---|
| 3 | invariance | not invariant; equivariant to relabelling of the four arms |
| 6 | leaks sign | **yes** — a one-sided "ρ > 0" test embeds the known direction |
| 7 | restates hypothesis | largely; on 4 points ρ = +1 ⟺ S1 = 1 |
| 8 | directional | signed; must be two-sided to avoid the leak |
| 9 | ties | midranks; generically absent for continuous medians |

**REJECT as primary** (signed form). Only 24 attainable orderings; on four points
`ρ = +1` is exactly the S1 event.

### 4.3 S3 — direction-free `|ρ|` on the same four arms

Fixes the sign leak (criterion 6) and is two-sided by construction. Still uses
only the four already-published arms (§1.2) and on 4 points remains close to a
restatement (criterion 7 partially fails). **REJECT as primary; acceptable as a
secondary descriptive.**

### 4.4 S4 — Kendall τ / inversion count on four arms

Equivalent information to S2/S3 on 4 points (τ and ρ induce the same ordering of
the 24 permutations up to ties). Same verdict. **REJECT as primary.**

### 4.5 S5 — circular rank total variation over **all twelve** offsets ★

```
s_m = d(g^{-m}),  m = 0 … 11
r   = midrank(s) within the family,  r_m ∈ {1 … 12}
TV(g) = Σ_{m=0}^{11} | r_{(m+1) mod 12} − r_m |          (circular, wraps 11 -> 0)
```

| # | criterion | assessment |
|---|---|---|
| 1 | definition | exact, closed form, integer-valued on ties-free data |
| 2 | information used | all 12 offsets, **ordinal only** — scale-free, no calibration |
| 3 | invariance | invariant to any monotone transform of `d`; equivariant to the choice of generator — which is exactly what is being tested |
| 4 | null analogue | identical formula with `g = γ`, a random 12-cycle |
| 5 | exact null | yes — enumerable by sampling generators (§5) |
| 6 | **sign/order leak** | **none.** TV is direction-free: reversing the family (`g → g^{-1}`) leaves TV unchanged |
| 7 | **restates hypothesis?** | **no.** TV measures *smoothness of the response along the natural cyclic order*, a strictly broader property than the observed 4-arm monotonicity. Monotone-over-a-window is neither necessary nor sufficient for low TV |
| 8 | directional / two-sided | naturally two-sided (low TV = structured either way); use a two-sided rank test |
| 9 | ties | midranks; declare in advance; exact ties among masked medians are possible in principle and must not change the formula |
| 10 | identical for ordered & null | yes — one function of one generator |

**Range (DERIVED).** On a 12-cycle, a rank sequence arranged monotonically around
the circle attains the minimum `TV = 22`; a maximally scrambled arrangement
approaches ≈72. Bounded, scale-free, distribution-free.

**Decisive advantage.** `s_m` for `m ∈ {3,…,10}` has **never been computed**
(§1.3). S5 therefore has genuine predictive content that S1–S4 entirely lack: two
thirds of its ordered-side input is unmeasured. It also removes the
window-selection problem, since no window is chosen.

**ACCEPT as the best available primary**, subject to §2.

### 4.6 S6 — first-harmonic energy fraction over all twelve offsets

```
ŝ_j = Σ_m s_m e^{-2πi jm/12};   S6(g) = |ŝ_1|² / Σ_{j=1}^{11} |ŝ_j|²
```

Same structural virtues as S5 (all 12 offsets, no window, direction-free —
`|ŝ_1|` is invariant to reversal). But it is **scale- and shape-sensitive rather
than purely ordinal**, so it is vulnerable to differences in the *spread* of `s`
between the natural and random families — a nuisance dimension that has nothing
to do with order. It would need a normalisation choice, and every such choice is
a free parameter that must be justified independently. **ACCEPT as a
preregistered secondary; REJECT as primary** in favour of the assumption-free
rank form.

### 4.7 S7 — isotonic-regression residual / magnitude-bearing trend statistic

Any statistic mixing magnitude with order re-imports the exact failure INDEX-002
already recorded: the ordered arms are **typical in magnitude** of arbitrary
re-indexing (**MEASURED**: `S_order` +4.11…+6.83 against null p99 +5.00…+6.87).
Mixing magnitude back in dilutes the ordinal signal with a dimension already
known to be non-discriminative. **REJECT.**

---

## 5. THE NULL — what an "ordered arm" and a "random arm" are

This section answers the brief's explicit questions.

**What constitutes an ordered arm.** A single element of the cyclic family
generated by the natural successor map `c: k ↦ k − 1 (mod 12)`. Arm `m` is
`c^m`, i.e. `π_m(k) = (k − m) mod 12`. `m = 0` is identity. The family is
**one generator plus an integer exponent** — that, not the list of four
permutations, is what makes it "ordered".

**What constitutes a random arm — the correct construction.** Draw a generator
`γ` uniformly from the **12-cycles** of `S_12` (there are `11! = 39 916 800` —
**DERIVED**), and take its powers `γ^m`. Then:

| arm | ordered family | null family | cycle type (**DERIVED**) |
|---|---|---|---|
| m = 0 | identity | identity | `[1¹²]` |
| m = +1 | `c^{+1}` | `γ^{+1}` | `[12]` |
| m = −1 | `c^{-1}` | `γ^{-1}` | `[12]` |
| m = +2 | `c^{+2}` | `γ^{+2}` | `[6,6]` |
| … | `c^m` | `γ^m` | matched at every `m` |

**Why this is the right null.** It is matched on *every* structural dimension —
group structure (powers of one generator), cycle type at every offset, inclusion
of identity, and the step relation (each arm is one further application of the
generator). The **only** thing that differs is *which* traversal of the candidate
axis the generator performs: the natural successor, or an arbitrary one. That is
precisely the hypothesis, isolated.

**Equivalence to conjugation (DERIVED).** Conjugating the whole ordered family by
a uniform `σ ∈ S_12` gives `σ π_m σ^{-1} = (σ c σ^{-1})^m = γ^m` with
`γ = σ c σ^{-1}`. Since the 12-cycles form a single conjugacy class and
conjugation acts transitively on it, `γ` is uniform over 12-cycles. So
"random-generator null" and "relabel the candidate axis" are the **same null**,
which is a reassuring internal consistency check and gives the null its
interpretation: *the candidate axis is relabelled, nothing else changes.*

**Can one random permutation generate the full statistic?** **Yes** — one
generator `γ` produces the entire family through its powers. This is the sharpest
practical contrast with INDEX-002, where three unrelated permutations had to be
*grouped* into an artificial triple (§3). No grouping is needed or permitted here.

**Do multiple random permutations create a valid empirical null?** Yes: `M`
i.i.d. uniform 12-cycle generators give `M` i.i.d. draws of the family statistic,
which is an exact finite-sample null for the exchangeability hypothesis *"the
natural generator is exchangeable with a uniformly random 12-cycle generator."*

**No screening.** Every drawn generator enters the null. No rejection for fixed
points (a 12-cycle has none by definition, so this criterion is automatically
satisfied and must not be used as a filter elsewhere), directional bias,
occupancy, displacement, or response magnitude.

### 5.1 A residual nuisance that must be declared, not repaired

**MEASURED** (INDEX-002 record): `d(identity)` sits **low** among random
permutations — e.g. seed0/000187: `d(id) = 5.6118` against a random `d(π)`
min/median/max of 2.1129 / 6.7718 / 8.9545. Identity occupies a fixed interior
position (`m = 0`) in both the ordered and the null family, so any tendency of
identity to be extreme relative to the other arms affects both families
identically, and therefore does **not** bias the comparison. It does, however,
*reduce* the null's chance of appearing structured, which makes the test more
permissive toward H1. This must be stated in the preregistration as a known
property of the design and must not be "corrected" afterwards.

---

## 6. CIRCULARITY ASSESSMENT — summary

| source of circularity | S1–S4 | S5 (proposed) |
|---|---|---|
| success defined as the observed 4-arm ordering | **yes** | no — smoothness over all 12 offsets |
| window `{−1,0,+1,+2}` chosen post-hoc | window is the whole statistic | no window is chosen |
| expected direction embedded | yes (S1, S2) | no — TV is reversal-invariant |
| ordered-side value already published | **fully** | only 4 of 12 inputs are known |
| null structurally matched to the ordered family | no, if INDEX-002's null is reused | yes (§5) |

S5 + the random-generator null is **not circular** in the sense the brief warns
about. The circularity objection is answerable. **It is §2, not circularity, that
blocks the design.**

---

## 7. UNIT OF ANALYSIS, INDEPENDENCE AND MULTIPLE TESTING

**MEASURED.** 12 units = 3 checkpoints × 4 scenes (§1.1). These are **not
independent**:

- units sharing a checkpoint share the entire weight set;
- units sharing a scene share the images, the features and the GT mask;
- within a unit, all arms share one cached cost volume.

**Therefore per-unit p-values must not be combined by any method that assumes
independence** (no Fisher, no Stouffer, no Bonferroni-as-if-independent, no
"k of 12 must pass"). Doing so would manufacture significance out of the shared
structure, and a "12/12 units" style rule would additionally be a threshold set
by the previously observed 12/12 — explicitly forbidden by the brief.

**Correct method: one global statistic with a shared-generator null.**

```
T(g) = Σ over the 12 units of TV_unit(g)        # or the mean; fixed in advance
```

Each null replicate uses **one generator applied across all twelve units**,
exactly as the ordered family is one generator applied across all twelve units.
The dependence among units is then reproduced identically in every null draw, so
it cancels: no independence assumption is made anywhere, and no correction is
needed. This is the standard exact permutation-test treatment of dependent units
and it is the only method here that is simultaneously valid and assumption-free.

Per-unit statistics are still **reported** (descriptively, with their own ranks)
but are **not** the primary test and cannot alter the verdict.

**Fixed in advance, not by outcome:** the aggregation is the **sum** of per-unit
`TV`, chosen because `TV` is already on a common bounded integer scale across
units (all units have exactly 12 offsets), so the sum needs no weighting and
introduces no free parameter. This choice is made here, in the audit, before any
null exists.

---

## 8. POWER AND RESOLUTION

**DERIVED.** Under the finite-sample rank procedure the smallest attainable
two-sided p is `2/(M+1)`:

| M generators | min two-sided p | reaches α = 0.01 |
|---:|---:|:---:|
| 64 | 0.030769 | **NO** |
| 128 | 0.015504 | **NO** |
| 199 | 0.010000 | yes (boundary) |
| 256 | 0.007782 | yes |
| 512 | 0.003899 | yes |
| **1000** | **0.001998** | **yes** |
| 2000 | 0.001000 | yes |

INDEX-002's near-miss (a nominal α = 0.01 that M = 64 could not reach) must not
recur: **M ≥ 199 is mandatory for α = 0.01, and M = 1000 is specified below** so
that the 1st and 99th percentiles of the null are estimated from ≈10 draws per
tail rather than from 2–5.

Generator supply is not a constraint: `11! = 39 916 800` distinct 12-cycles
(**DERIVED**), so `M = 1000` draws are effectively collision-free.

**Cost (DERIVED).** Using the measured INDEX-002 throughput of ≈0.0107 s per
aggregation/readout arm, and 11 new arms per generator per unit (`m = 0` is
identity, computed once):

| M | positive arms | negative-gate arms | total | estimated runtime |
|---:|---:|---:|---:|---:|
| 512 | 67 584 | 22 528 | 90 112 | ≈16 min |
| **1000** | **132 000** | **44 000** | **176 000** | **≈31 min** |
| 2000 | 264 000 | 88 000 | 352 000 | ≈63 min |

This is an order of magnitude beyond INDEX-002's 264 s and well beyond the
"order of seconds" budget those protocols assumed. It is not prohibitive, but it
must be **authorised in advance**, not discovered mid-run, or the §9.13 hard stop
will fire on the experiment's own budget rule.

---

## 9. CONDITIONAL PREREGISTRATION SPECIFICATION

**Status: NOT AUTHORISED FOR EXECUTION.** Recorded so the design is complete and
inspectable, and so that no part of it can be chosen later with knowledge of an
outcome. Its precondition is §9.14. **Executing it without that precondition
would produce a statistically valid but scientifically uninformative result.**

**9.1 Primary statistic.** For a 12-cycle generator `g`, and for each unit `u`:
`s^u_m = d_u(g^{-m})`, `m = 0…11`, where `d_u(x) = np.median(disparity_initial[mask_u])`
— **identical to INDEX-001 and INDEX-002**. Let `r^u` be the midrank vector of
`s^u` within the unit, and

```
TV_u(g) = Σ_{m=0}^{11} | r^u_{(m+1) mod 12} − r^u_m |
T(g)    = Σ_{u=1}^{12} TV_u(g)
```

Ordered value: `T(c)` with `c(k) = (k − 1) mod 12`.

**9.2 Null statistic.** `T(γ_i)` for `i = 1…M`, `γ_i` i.i.d. uniform over the
12-cycles of `S_12`. Identical formula, identical code path, identical mask,
identical `d_u`.

**9.3 Unit of analysis.** The 12 positive `(checkpoint, scene)` units. Primary
test on the **global** `T`; per-unit `TV_u` reported descriptively only (§7).

**9.4 Permutation generation.** Draw `σ_i` uniform from `S_12`; set
`γ_i = σ_i c σ_i^{-1}` (uniform over 12-cycles, §5). Reject **nothing**. Record
every generator, its inverse, its powers' cycle types, and its
retained-band displacement, as descriptive metadata that is **not** an input to
any decision.

**9.5 Number of permutations.** `M = 1000`. Fixed by §8's arithmetic
(`2/1001 = 0.001998 ≤ α`), never by any observed result.

**9.6 RNG seed policy.** A single seed, `numpy.random.default_rng(<YYYYMMDD of
the freeze date>)`, written into the preregistration **before** execution; the
full generator list serialised to `generators.json` and its sha256 recorded in
the preregistration before any arm runs — the INDEX-002 procedure, which
verified correctly.

**9.7 Tie handling.** Midranks (`scipy.stats.rankdata(..., method="average")` or
an explicit equivalent, named in the preregistration). Applied identically to
ordered and null families. Exact ties among masked medians are not expected but
are not excluded; the formula does not change if they occur.

**9.8 Empirical p.** Low `T` is the structured direction, but the test is
**two-sided**, computed exactly as in INDEX-002:

```
ge = #{ T(γ_i) >= T(c) } ;  le = #{ T(γ_i) <= T(c) }
p  = min(1, 2 · min( (1+ge)/(M+1), (1+le)/(M+1) ))
```

**9.9 Aggregation across units.** Already inside `T` (§7). No p-value combination
of any kind.

**9.10 α.** `α = 0.01`. Justified independently of every previous result: it is
the threshold INDEX-002 already froze, carried forward unchanged, and it is
reachable at `M = 1000` (§8). It is **not** derived from the observed 12/12.

**9.11 PASS / FAIL.** `PASS` iff the negative control passes (§9.12) **and**
`p ≤ 0.01` **and** `T(c) < ` the null's 1st percentile (the structured side). No
intermediate verdict. No effect-size threshold, no required number of per-unit
successes — both would be set by the prior 12/12 and are forbidden.

**9.12 Negative control.** Unchanged and algebraic: on the trained
`shift="none"` checkpoint, `V(k)` is constant in `k`, so for **every** arm of
**every** family, `max|V_perm − V| == 0` and `max|d_perm − d_id| == 0`. Any
non-zero ⇒ **HARD STOP**, no positive arms. Note (**DERIVED**) that `TV` is
*undefined* there — all 12 responses tie, midranks are all 6.5, `TV = 0` for
every generator and the null is degenerate. This is the correct behaviour for a
degenerate model and must be declared in advance so that it is not mistaken for a
structured result; the negative control's pass criterion is the algebraic
identity, **never** the ordering statistic.

**9.13 Hard stops.** INDEX-002's list, plus: any screening or rejection of a
generator; any generator drawn after an output is inspected; `M` changed after
execution begins; runtime exceeding the authorised budget of §8; the architecture
control of §9.14 absent or altered.

**9.14 Mandatory precondition — the architecture-only control.**
INDEX-003 may not be executed without an arm in which the **same architecture
with `shift="left"` and randomly initialised, untrained weights** is run through
the identical pipeline, under a seed frozen in the preregistration. Rationale in
§2: without it the test cannot separate learned candidate-coordinate tracking
from convolutional equivariance, and a `PASS` would be uninterpretable.

Interpretation, fixed in advance:

| trained | untrained | conclusion |
|:---:|:---:|---|
| PASS | PASS | the ordering is **architectural**; it carries no information about learning. Close the ordering line. |
| PASS | FAIL | the ordering is **weight-dependent**; candidate-coordinate ordering is distinguishable from the arbitrary-permutation null *and* is not merely structural. Strongest outcome available. |
| FAIL | either | candidate-coordinate ordering is not distinguishable under this design. |

**This control instantiates new weights.** It performs no training and modifies
no checkpoint, but it does step outside the current protocol's "do not change
weights/checkpoints" rule and therefore **requires explicit authorisation before
INDEX-003 is preregistered at all.**

**9.15 Claim ceiling.** On `PASS` **with** the untrained control failing, the
strongest permitted statement is exactly:

> "Candidate-coordinate ordering is statistically distinguishable from the
> preregistered arbitrary-permutation null under the specified candidate-axis
> intervention."

It does **not** establish geometric correspondence, correct matching, correct
disparity, genuine disparity search, physical translation correspondence, or
stereo correctness. Candidate indices are not converted to physical disparity
anywhere in the statistic (§10). On `PASS` **with** the untrained control also
passing, the only permitted statement is that the ordering is architectural.

---

## 10. REFERENCE-FRAME COMPLIANCE

The design converts nothing. `TV` is a function of **ranks of masked medians**
within a family — it is invariant to any monotone transform of `d`, so no unit
interpretation of `d` enters at all, and the labels `−1, 0, +1, +2, …, 11` are
used **only** as exponents of the frozen generator.

The distinction is maintained explicitly:

- **candidate-axis ordering** — a property of the map `m ↦ d(g^{-m})` on `Z_12`.
  This is what S5 tests.
- **physical disparity correspondence** — requires mapping candidate index `k` to
  a disparity of `16k` full-resolution pixels **and** resolving that the volume
  indexes that magnitude at the *right* image column while GT indexes the *left*
  (**MEASURED**, `cost_volume.py`: level `k` at column `u` is
  `left_feat[u+k] − right_feat[u]`). **Untested, and untouched by this design.**

No physical slope is claimed anywhere. The geometric experiment is not designed,
not preregistered and not launched here.

---

## 11. WHAT THIS AUDIT DOES *NOT* CONCLUDE

- It does **not** conclude that candidate-coordinate sensitivity is absent.
- It does **not** re-score, reinterpret or overturn INDEX-001 or INDEX-002. Both
  stand exactly as recorded; INDEX-002's `NOT-DEMONSTRATED` remains its verdict.
- It does **not** claim to have measured what an untrained network does. §2's
  expectation is **INFERRED and untested**, and it is the reason for §9.14 rather
  than a substitute for it.
- It does **not** recommend `ABANDON-ORDERING-HYPOTHESIS`. The ordering is real,
  reproduced three times, and the §9.14 design would **settle** whether it is
  architectural — which is worth knowing, and is the cheapest remaining question
  in this line.

---

## 12. FINAL DESIGN VERDICT

```
DESIGN-INCONCLUSIVE
```

An ordering statistic can be written down and given a clean, structurally
matched, non-circular, exactly-generable permutation null (S5 + random 12-cycle
generators, §4.5 + §5). The circularity objection in the brief is fully
answerable. But the test cannot be made **discriminative** without the
architecture-only control of §9.14, because the ordered response is an expected
consequence of the aggregation's five zero-padded 3-tap candidate-axis
convolutions and would be produced by arbitrary weights. Without that control the
design yields a near-guaranteed `PASS` that says nothing about the trained model
— statistically valid, scientifically vacuous. That is an assumption-dependent
design, which is precisely the `DESIGN-INCONCLUSIVE` criterion.

---

## CLOSING QUESTION

> **Is EXP-CORRESPONDENCE-INDEX-003 scientifically justified, and what exact
> preregistered statistic would it use?**

**Not as proposed.** As framed — ordered arms versus an arbitrary-permutation
null — it is not justified, for two independent reasons. First, `H0` as stated
("arbitrary candidate-axis permutations can produce equally ordered responses")
is refutable from the architecture alone: five zero-padded 3-tap convolutions
along the candidate axis are translation-equivariant in the interior, so
`d(π_m) ≈ d(π_0) + m` is expected for *any* weights, and a test that rejects that
`H0` confirms a near-theorem rather than a property of the trained model. Second,
every statistic confined to the four arms has a value that is **already published
in INDEX-002** — 12/12 strictly increasing — so it cannot be preregistered
predictively and would at best calibrate an already-measured quantity.

**It becomes justified under one modification**: add the architecture-only
control of §9.14 — the identical architecture with `shift="left"` and randomly
initialised, untrained weights — and reframe the question from *"is the ordering
significant?"* (answer effectively known) to *"is the ordering a property of the
trained weights or of the convolution?"* (genuinely unknown, and decisive for
whether this line continues). That version is cheap, and either outcome closes a
real question. It requires explicit authorisation because it instantiates new
weights, though it trains nothing.

**The exact statistic it would use** — fixed here, before any null exists:

> For a 12-cycle generator `g` and unit `u`, let `s^u_m = d_u(g^{-m})` for
> `m = 0…11`, where `d_u(x) = np.median(disparity_initial[mask_u])` exactly as in
> INDEX-001 and INDEX-002. Let `r^u` be the midrank vector of `s^u` within the
> unit. Define the **circular rank total variation**
> `TV_u(g) = Σ_{m=0}^{11} |r^u_{(m+1) mod 12} − r^u_m|`
> and the global statistic `T(g) = Σ_{u=1}^{12} TV_u(g)`.
> The observed value is `T(c)` for the natural successor generator
> `c(k) = (k−1) mod 12`. The null is `{T(γ_i)}` for `M = 1000` generators drawn
> i.i.d. uniform over the 12-cycles of `S_12` via `γ_i = σ_i c σ_i^{-1}`, each
> generator applied across **all twelve units** so that the units' shared
> checkpoints and scenes are reproduced identically in every null draw and no
> independence assumption is made. Two-sided exact p:
> `p = min(1, 2·min((1+#{T(γ)≥T(c)})/(M+1), (1+#{T(γ)≤T(c)})/(M+1)))`,
> `α = 0.01` (reachable: `2/1001 = 0.001998`), midrank tie handling, no screening
> of generators, algebraic `shift="none"` gate unchanged, and the same statistic
> computed identically for the trained and the untrained-weights arms.

Chosen because it is scale-free and ordinal (immune to the magnitude dimension
INDEX-002 already showed is non-discriminative), direction-free (no expected-sign
leak), window-free (no post-hoc offset selection), structurally matched to the
ordered family at every offset, and because two thirds of its ordered-side input
— the offsets `m = 3…10` — have never been computed and so retain real predictive
content.

**Nothing was executed. The geometric correspondence experiment remains not
designed, not preregistered and not launched.**
