# FINAL BLIND O1 IDENTIFIABILITY AUDIT

**Run id:** `o1_final_blind_audit_20260913T050232Z`
**Scope:** decide whether `O1` (candidate-axis permutation of the cost volume)
can identify genuine stereo correspondence / disparity search.
**Mode:** audit only. No checkpoint was opened, no weight was trained, no O1
execution was performed, no historical record was modified.

## 0. Blindness declaration — read this first, it is a limitation

The following were **not** opened and were not searched:
`o1_adversarial_audit_20260913T033554Z/`, `o1_audit2_20260913T041729Z/`,
`o1_audit3_nonblind_20260913T044259Z/`. Their existence is visible in the
directory listing; no content, verdict, counterexample or threshold from them
entered this audit.

Blindness is **not complete**, and the breach is disclosed rather than hidden.
Two project memory records were in context before the audit began:

1. `stereonet-diagnostic-confounds` states — as a prior conclusion, not as a
   measurement — that permuting the candidate axis is *"the one airtight
   construction"*. This is a **pro-O1** prior and is treated here as a
   hypothesis under test, not as authority. This audit contradicts it (§5, §26).
2. The GEOM-002 successor verdict states that a candidate-axis permutation
   control is *"justified, but only as a different experiment class"*. Also
   pro-O1, also treated as a hypothesis.

Because both leaked priors favour running O1, the leak cannot explain a
DO-NOT-RUN conclusion; it could only have biased this audit toward RUN.

One further prior is **experimental, not an audit**, and is used as an
authoritative fact: `EXP-CORRESPONDENCE-INDEX-001` already ran a candidate-axis
permutation control on the disparity output and returned **NOT-DEMONSTRATED**,
because the frozen random permutation control matched the ordered arms in
magnitude, with a post-hoc discovered directional bias in the control
(`mean pi^-1(d) = 7.14` against the band mean `5.0`). O1 is a re-specification of
that intervention class with a new statistic.

## 1. Sources used

* Frozen source: `src/models/stereonet/{cost_volume,aggregation,regression}.py`,
  `phase2/models/scaled_regression.py` (read; confirms the proposal's
  description of A and R exactly, including the z-score, which lives in
  `scaled_regression`, not in the frozen Phase-1 readout).
* GEOM-001: halted at HS1, no admissible result; cause was a false premise in a
  frozen domain bound; corrected domain `x in [325,633) and y in [64,208)`.
* GEOM-002: G-STOPped at Stage 1. Geometry held (`V[tau]` exactly `0.0` over
  3,525,120 cells). A randomly initialised aggregation scored a perfect `h = 6`
  in 31/576 gate units. Failure audit: with `Rf[w] = Lf[w+tau]`, substituting
  `u = w+tau`, `d = k-tau` makes every cell a function of `(u,d)` alone, so
  **tau is a coordinate change, not an experimental factor**; any statistic
  invariant under candidate-axis translation is dead on that intervention; a
  weight-randomised null fails for *flatness* (soft-argmin collapse to the
  window centre ~5.5), not for geometric blindness.
* New measurements made for this audit: `audit_numerics.py`,
  `measurements.json`. Random synthetic features, hand-set and randomly
  initialised weights inside the frozen `Aggregation`, frozen readout.

Notation: `A` = frozen aggregation, `R` = frozen readout (upsample -> z-score
over k -> softmax(-z) -> soft-argmin), `P_pi` = candidate-axis permutation
`(P_pi V)[k] = V[pi(k)]`, `m = R(A(V))`, `E_orig = |m - tau|`,
`S8 = |m_pi - tau| - |m - tau|`.

---

# TEST 1 — what O1 actually perturbs

| Mechanism | P1 `m_pi = R(A(V^pi))` | P2 `m_pi = R(A(V^pi) o pi^-1)` |
|---|---|---|
| A. candidate **content** (which 32-ch slice sits at each index) | changed | changed |
| B. candidate **coordinate** (which absolute `k` label carries it) | changed | restored |
| C. candidate **neighbourhood** (which slices are adjacent under Conv3D) | changed | changed |
| D. **stereo correspondence** (is a candidate the geometrically right alignment) | **not changed** | **not changed** |

D is untouched by either treatment, and this is structural, not incidental.
The left/right comparison is performed by `build_cost_volume`
(`levels.append(shifted - right)`) — **frozen, un-learned code upstream of the
module O1 tests**. Permuting the candidate axis of an already-built cost volume
relabels and reorders match evidence; it cannot remove, corrupt or displace the
correspondence operation, because that operation is not in `A`.

A, B and C are changed **simultaneously** by P1; P2 removes B and leaves A and C
entangled. **No mechanism is changed independently by either treatment.**

# TEST 2 — symmetry / equivariance, derived

Stage by stage, for an arbitrary permutation pi of the candidate axis:

* **Conv3D 3x3x3, padding 1.** `out[k] = sum_{delta in {-1,0,1}} W_delta * in[k+delta]`.
  Equivariant to **translations** of `k` (modulo padding), **not** to general
  permutations: `A(P_pi V)[k]` reads `V[pi(k-1)], V[pi(k)], V[pi(k+1)]` whereas
  `P_pi A(V)[k]` reads `V[pi(k)-1], V[pi(k)], V[pi(k)+1]`. Equal for all `V` iff
  pi is a translation **and** the boundary is unaffected.
* **LeakyReLU** — pointwise, commutes with `P_pi` exactly.
* **Bilinear upsample** — `F.interpolate` on `(B, D, h, w)` resamples `(h,w)`
  only and treats `D` as channels; commutes with `P_pi` exactly.
* **z-score over the candidate axis** — mean and std over `k` are
  permutation-**invariant**, so `z(P_pi C) = P_pi z(C)` exactly. Equivariant.
* **softmax(-z) over `k`** — equivariant.
* **soft-argmin `m = sum_k k p(k)`** — the **only** stage that consumes the
  absolute index label.
  `R(P_pi C) = sum_k k p_C(pi(k)) = sum_j pi^-1(j) p_C(j) = E_{p_C}[pi^-1]`.

Consequences:

* **`A(V^pi) != A(V) o pi` in general, and the entire discrepancy is produced by
  the candidate-axis 3-tap plus its zero padding** — i.e. by Conv3D adjacency,
  which is an architectural property present at random initialisation.
* **P2 isolates exactly that discrepancy and nothing else.** If `A` were
  candidate-pointwise, `A(V^pi) = P_pi A(V)` and composing with `pi^-1` returns
  `A(V)` bit-for-bit, so `S8 == 0`. **Measured: for the hand-built pointwise
  latch of TEST 5, `S8_P2 = 0.0` exactly in 48/48 (tau, pi) cells; for randomly
  initialised full-kernel aggregations, `S8_P2 != 0` in 48/48 cells, range
  -2.96 ... +4.76.** P2 therefore does not remove the confound — it *is* the
  confound, measured directly, and it is anti-correlated with the mechanism of
  interest: a perfect candidate-content detector scores exactly zero on P2 while
  untrained noise scores large.
* **P1 does not isolate anything.** It reports `E_p[pi^-1]`, which mixes the
  model's candidate distribution with the arithmetic of the chosen permutation.

Nothing in this derivation requires correspondence at any point. No quantity in
O1 is correspondence-specific.

# TEST 3 — synthetic cost-volume degeneracy

Verified from the frozen construction and re-measured here: with
`Rf[w] = Lf[w+tau]`, `V[k,w] = Lf[w+k] - Rf[w]`, so `V[tau,w] = 0` on the valid
domain. **Measured this audit: `max |V[:,:,tau,:,15:45]| = 0.0` for all
tau in {1..6}.** (GEOM-002 measured the same at scale: 3,525,120 cells, all `0.0`.)

The raw per-candidate L1 profile, measured (tau = 3):

```
k     :  0      1      2      3      4      5      6      7      8      9     10     11
L1(k) : 1.135  1.133  1.132  0.000  1.133  1.131  1.134  1.133  1.145  1.125  1.123  1.130
```

The matched candidate is **not** "the best of twelve". It is an **exact zero
against a flat non-zero background** — an infinite contrast ratio, a uniquely
detectable singularity. This answers the key question of §6 of the brief
directly: **yes** — a mechanism that never reasons about left/right can identify
tau, because tau is marked by a measure-zero anomaly that any magnitude
functional finds. And O1 does not distinguish such a mechanism (TEST 5).

# TEST 4 — observational equivalence

* **H_match** — `A` uses left/right correspondence to locate the correct
  candidate.
* **H_latch** — `A` detects the distinctive candidate-slice property created by
  the construction (norm of slice = 0) and lets the readout select it.

On an exact self-pair these two hypotheses **prescribe the same computation**.
"Minimise matching cost over candidates" and "find the zero-magnitude slice" are
the same function on this data; they differ only on inputs where no slice is
zero (real stereo, photometric difference, noise) — which this intervention
class never presents. `P(O1 | H_match) = P(O1 | H_latch)` is not merely
approximately true, it is exactly true for the pointwise latch on the matched
domain. Logical observational equivalence holds. **Correspondence is not
identifiable here.**

# TEST 5 — frozen-architecture counterexample (mandatory)

**Computation.** `c(k,y,x) = sum_{j<16} 0.99*|V[j,k,y,x]|`, built with hand-set
weights **inside the unmodified `Aggregation` class**:

* layer 1 (`Conv3d 32->32`): centre tap only; output channel `j` = `+V_j`,
  channel `16+j` = `-V_j` for `j < 16`;
* LeakyReLU(0.01) gives `LReLU(x) + LReLU(-x) = (1-0.01)|x|` — **measured exact
  to 4.44e-16**;
* layer 2: centre tap only, sums channel `j` and `16+j` into channel 0;
* layers 3, 4 and `to_cost`: centre-tap identity on channel 0.

**Why it is non-correspondence.** (i) It performs no left/right comparison — the
subtraction is upstream in frozen code, and `|L-R| = |R-L|`, so the operator is
blind to which image is which; (ii) every kernel tap outside the centre is zero,
so it uses **no candidate adjacency and no spatial context**; (iii) it never
reads the candidate index. It is a pointwise magnitude threshold. It contains no
search, no geometry and no notion of disparity.

**Why it is inside the frozen architecture.** It is an instance of
`Aggregation()` with `111,585` parameters in the frozen shape; only values are
set. The `|.|` construction is exact, not approximate.

**Does it produce the O1 signature? Measured, 6 tau x 8 random pi = 48 cells:**

| quantity | latch (hand-set, non-correspondence) | random init, 3 seeds |
|---|---|---|
| `E_orig` mean (min...max) | **0.647** (0.148 ... 1.346) | 2.385 (0.240 ... 6.224) |
| `S8` under **P1** mean (min...max) | **+2.373** (-0.708 ... +5.633) | +0.062 (-0.502 ... +0.867) |
| fraction `S8_P1 >= 1` | **70.8 %** | **0 %** |
| `S8` under **P2** | **exactly 0.0, 48/48** | != 0 in 48/48, -2.96 ... +4.76 |
| `m_0` mean | tracks tau | 5.089 (centre collapse) |

**This is not merely expressibility.** The latch beats the random-weight null on
the full conjunction (small `E_orig` **and** large positive `S8_P1`), which is
the decision rule O1 would apply. It therefore **defeats identifiability**: any
O1 P1 positive that a trained checkpoint could produce is reproduced by a
mechanism that provably contains no correspondence.

Two further measured facts from the same run:

* `E_orig` for the *perfect* detector is **tau-dependent**: 1.346 at tau=1
  falling to 0.148 at tau=6. The readout drags `m` toward the window centre, so
  "small `E_orig`" is partly a property of tau, not of the model.
* `m_pi` does **not** equal `pi^-1(tau)` exactly (max deviation 1.646) — the same
  centre pull — so `S8_P1` is a blurred version of the pure permutation
  arithmetic, not a clean read of it.

# TEST 6 — decoy / intervention adequacy

The distinguishing intervention would be a **decoy**: a candidate that carries
the detectable anomaly while *not* being the geometrically correct alignment.
H_latch follows the decoy; H_match does not.

**O1 contains no such intervention** — a permutation moves the anomaly and the
correct candidate *together*, because they are the same slice. Worse, the
construction **cannot host a decoy at all**, from the algebra
`dV[k,w] = dLf[w+k] - dRf[w]`:

* an **R-feature anomaly** at column `w` is **k-constant** — it perturbs every
  candidate identically and adds no candidate structure;
* an **L-feature anomaly** at column `u` appears at `(k, w = u-k)` for **every**
  `k` — one cell in every candidate, a diagonal, not a candidate-local mark;
* a **candidate-local perturbation** is not reachable by any change to `Lf` or
  `Rf`; it requires editing `V` directly, which is no longer a stereo pair;
* a **spatially localised** perturbation is smeared over `w` by the five 3-taps
  (radius 5 cells) and remains present in all candidates.

And a decoy that *does* become a second exact zero (a periodic `Lf`) is a
genuine matching ambiguity, which a real matcher must also be misled by.
**Conclusion: O1 does not contain the information needed to separate the
hypotheses, and the synthetic-self-pair class cannot be made to contain it
without ceasing to be a stereo pair.**

# TEST 7 — random-initialisation null

The **complete O1 signature** is the conjunction: (a) `E_orig` small, and
(b) `S8` positive and large, with (c) the treatment applied per the fixed pi.

Measured here (3 random seeds, full kernels, 48 cells each):

* **Weak signature** (output changes under permutation): present at random
  initialisation — `S8_P2 != 0` in 48/48 cells.
* **Intermediate signature** (`S8_P1 > 0`): weakly present, mean +0.062, max
  +0.867 — **never reaching 1**.
* **Full signature**: **absent**. `E_orig` mean 2.385 and `m_0` mean 5.089 —
  random weights collapse to the window centre, reproducing GEOM-002's measured
  flatness failure (`0/576` units within 1.0 candidate).

Stated honestly: **random initialisation does not produce the full O1
signature.** The claim "random initialisation already produces O1's result" is
**not** supported and is explicitly rejected below (§25).

But the correct question is whether the random null invalidates O1's
*identifying logic*. It does not — and it does not rescue it either. The null is
beaten by the hand-built latch (TEST 5), which contains no correspondence. The
random null therefore establishes only that training produces a non-trivial
effect; it does nothing to license a correspondence reading. GEOM-002 recorded
the same point from the other side: a weight-randomised null "is a floor, not a
control", failing for flatness, so "a merely sharper trained model would beat it
while inheriting tracking for free".

# TEST 8 — trained-vs-random distinction

If a trained checkpoint gave very small `E_orig` and large positive `S8`, while
random weights did not, the **logically justified** conclusions are:

* YES learned candidate-index structure (the trained weights produce a
  candidate-resolved, non-flat, input-dependent cost profile);
* YES learned candidate-ordering dependence (P1 sensitivity beyond the
  architectural P2 residual);
* NO geometric candidate-axis use — **not** justified: the latch has no geometry
  and produces the same signature;
* NO genuine correspondence — **not** justified, twice over: by TEST 4
  equivalence, and because the correspondence operation is not in the tested
  module (TEST 1).

The step "not random therefore correspondence" is invalid. The only thing the
contrast establishes is *sharpness plus content-localisation*, which is Level 3.

# TEST 9 — candidate-axis architectural null

Guaranteed by `3x3x3` Conv3D with `padding=1`, **before any training**:

| property | learned? | architectural? |
|---|---|---|
| candidate adjacency dependence | no | **yes** (3-tap in `k`) |
| non-pointwise candidate processing | no | **yes** |
| boundary / padding dependence | no | **yes** |
| local candidate smoothing | no | **yes** |
| candidate-order sensitivity | partly | **yes, in part** — measured `S8_P2 != 0` in 48/48 random cells |

All five are architectural. **Padding-free candidates after five padded 3-taps:
`k in {5, 6}` only** — recomputed and confirmed this audit, matching the
independently measured bit-identity of `k = 5` and `k = 6` recorded in the
confounds register. Every one of these properties is therefore unavailable as
evidence of learned stereo correspondence.

# TEST 10 — padding / boundary confound

Moving a slice changes its padding exposure: after five layers, candidates 0–4
and 7–11 are contaminated by the zero boundary and only 5–6 are clean. A
permutation systematically changes which content sits in the contaminated
positions, so permutation sensitivity follows from padding alone.

Adequacy of the confound as an explanation: **contributor, not sole cause, and
not separable.** Measured `S8_P2` under random weights spans -2.96 ... +4.76 with
100 % non-zero — far outside any plausible interior variation, so it is not a
nit. But it is not the whole of `S8_P1` either, since the pointwise latch (which
has *no* padding sensitivity — `S8_P2 = 0.0` exactly) still yields
`S8_P1 = +2.37` mean. The honest statement: **P2's signal is substantially
padding/adjacency, and P1's signal does not need padding at all.** Neither is
empirically distinguishable from the target mechanism within O1.

# TEST 11 — readout confound

`m = sum_k k p(k)` is **not** permutation-equivariant; it is the sole consumer of
the absolute index. Under P1 the index weighting becomes `E_p[pi^-1]`. P2
restores the physical coordinate system exactly, and in doing so drives `S8` to
identically zero for any candidate-pointwise mechanism (measured 48/48).

A degraded distribution moves `m` toward the centre: with `D = 12`, a uniform
`p` gives `m = 5.5`. This creates a **tau-dependent `S8` with no mechanism
content**: if permutation merely flattens the distribution,
`S8 = |5.5 - tau| - E_orig`, i.e.

```
tau    : 1     2     3     4     5     6
S8     : 4.5   3.5   2.5   1.5   0.5   0.5      (mean +2.167)
```

Measured corroboration: random weights give `m_0` mean 5.089, i.e. the collapse
is real in this exact pipeline, and the latch's own `E_orig` rises to 1.346 at
tau=1 purely from centre pull. **`S8` magnitude is controlled by pi, tau,
distribution peakedness and readout blur.** Correspondence appears nowhere in
that list.

# TEST 12 — sign of S8

`S8 > 0` has **no unique mechanistic interpretation**. Mechanisms that produce
it, all realizable here:

1. genuine matching (peak moves with the content);
2. **pointwise magnitude latch** — measured, mean +2.373, no adjacency, no
   geometry;
3. **sharpness loss / flattening** — `m_pi -> 5.5` gives `S8 = |5.5 - tau|`,
   mean +2.167 over the frozen tau set, with *zero* tracking of the permuted
   content;
4. adjacency / padding disruption — measured non-zero at random init;
5. readout distortion (centre pull), tau-dependent by construction;
6. an incorrect but sharp estimator whose error happens to grow under
   relabelling;
7. **pi arithmetic alone** — if pi is a cyclic shift by `s`, then by the GEOM-002
   reparametrisation (`tau` is a coordinate change) a translation-tracking model
   returns `m_pi ~ tau - s` and `S8 ~ s`, with no mechanism information at all.

Seven mechanisms, one sign. The sign is not correspondence-specific.

# TEST 13 — magnitude of S8

Rejected as a certificate. Measured and derived, magnitude is set by: how far pi
moves tau (`mean |pi^-1(tau) - tau| = 3.58` for the 8 random pi used here, versus
the +2.37 actually observed once blur is included); proximity of tau to the
window edge; collapse toward 5.5 (`mean +2.167`); sharpness of the original
prediction; destruction of candidate adjacency. The proposal controls **none** of
these. A larger `S8` is evidence of a larger permutation displacement, not of a
better matcher.

# TEST 14 — threshold

`S8 >= 1` has **no principled derivation** and is rejected.

* Not derived from a null: the measured random-weight `S8_P1` never reaches 1
  (max 0.867), so `1` is not a null quantile — it is adjacent to the null's
  range by coincidence.
* Not derived from readout resolution or candidate spacing: soft-argmin blur is
  measured at 0.15–1.35 candidates for a *perfect* detector, i.e. the threshold
  is the same size as the estimator's own bias.
* Not robust to tau: pure flattening clears it automatically for
  tau in {1,2,3,4} (`|5.5 - tau| >= 1.5`) and fails for tau in {5,6} (`0.5`).
  The threshold is a statement about tau.
* Not robust to pi: a cyclic pi with `s >= 1` clears it by arithmetic.

No replacement threshold is proposed. The defect is not the number.

# TEST 15 — null design

O1 has **no valid null**.

* Treatment vs null: the "null" (random weights) differs from the treatment in
  *sharpness*, which is the very quantity `S8` responds to. GEOM-002 measured
  this directly. A sharpness-matched null is what would be needed; random
  weights are not it, and the hand-set latch — which *is* sharpness-matched and
  correspondence-free — **passes**, which is the definition of an invalid null.
* Matched pairing: `S8` differences a permuted and unpermuted arm on the same
  volume, which correctly removes scene main effects and correctly removes
  nothing else.
* `pi` control: `EXP-CORRESPONDENCE-INDEX-001` already established, empirically,
  that a random pi is not a neutral control — its own `pi^-1` had a `+2.14`
  directional bias over the retained band.
* tau and scene dependence: `S8` is tau-dependent by construction (TEST 11).
* Checkpoint-level inference: three checkpoints, used for the eleventh time.

# TEST 16 — multiplicity

The proposal fixes **none** of the exploitable degrees of freedom: pi family,
fixed-vs-random pi, number of permutations, P1-vs-P2 as primary, tau set,
threshold, multiplicity correction. Every one of these is selectable after seeing
results, and `max_pi S8` is the same statistical error GEOM-002 already made with
`max(h_rand)` over `N = 576` units, where the measured pass probability was
`1.4e-14` — decided before it ran. A "best permutation" or "maximum `S8`" rule is
rejected in advance. Post-hoc selection of P1 over P2 is the single largest risk,
because the two treatments have *opposite* relationships to the confound (P2 is
zero for the clean mechanism and large for noise).

# TEST 17 — inference unit

* **Independent-ish:** training seed / checkpoint (3, and the same 3 as ten prior
  experiments — a fixed, non-random, exhausted sample); scene pair (6, from one
  split).
* **Not independent:** tau — by the GEOM-002 reparametrisation, tau is a
  **coordinate change**, so the six tau levels are six views of one measurement,
  not six samples. pi — a set of permutations applied to the same volume yields
  correlated measurements of the same `E_p[pi^-1]` functional. Pixels — the
  44,352-pixel mask is ~220 feature cells of real support, overstating
  independence ~16x.

Counting tau x pi x scene as `n` would inflate the sample by roughly two orders
of magnitude. The defensible unit is the **checkpoint**, giving `n = 3`.

# TEST 18 — claim ceiling (set before considering any trained result)

| Level | Additional assumption needed to reach it from the level below | reachable by O1? |
|---|---|---|
| 0 output changes under candidate permutation | none | **yes** — and already true at random init (`S8_P2 != 0`, 48/48) |
| 1 candidate ordering / coordinate sensitivity | that the change tracks the ordering, not just any perturbation | **yes** — but architectural in part (3-tap + padding) |
| 2 candidate neighbourhood / adjacency dependence | that the sensitivity survives coordinate restoration | **yes** — this is exactly what P2 measures, and it is architectural |
| 3 learned candidate-index structure | that the effect exceeds the untrained architectural baseline | **yes** — measured gap between latch/trained-style sharpness and random |
| 4 geometric candidate-axis use | that the index structure is *geometric*, i.e. no non-geometric mechanism produces it | **NO** — refuted by the TEST 5 latch |
| 5 genuine stereo correspondence / disparity search | that the tested module performs the left/right comparison, and that the anomaly and the match are separable | **NO** — the comparison is upstream and frozen (TEST 1); anomaly and match coincide (TEST 3, 4) |
| 6 correct disparity on natural stereo | generalisation off the synthetic self-pair class | **NO** — out of class |

**Ceiling: Level 3.** Level 4 requires an assumption this audit has disproved by
construction.

# TEST 19 — "what if O1 works perfectly?"

Grant `E_orig ~ 0`, very large positive `S8`, consistent across scenes, tau and
many permutations. Surviving non-correspondence explanations:

1. **The pointwise magnitude latch** — measured to produce exactly this
   signature in the frozen architecture with no correspondence, no adjacency and
   no geometry. Consistency across tau and pi is *predicted* by it, not evidence
   against it: `S8 ~ |pi^-1(tau) - tau|` is consistent by arithmetic.
2. **Sharpness plus collapse** — a sharp model whose sharpness is destroyed by
   any relabelling gives `S8 = |5.5 - tau|`, consistent across scenes and pi for
   the same reason.
3. **Adjacency/padding disruption** — untrained, measured, and consistent.

Consistency across many pi and tau is therefore *not* a discriminator; it is what
the confounds predict. **A non-correspondence explanation survives the perfect
result.** O1 remains non-identifying under its own best case.

# TEST 20 — "what if O1 fails?"

`S8 ~ 0` would **not** prove absence of correspondence. Benign causes: exact
permutation-equivariance of a pointwise mechanism under P2 (measured: `0.0`
for the latch — a *perfect* detector scores a perfect null on P2); flatness
collapse making both arms ~5.5 so the difference vanishes (measured: random
weights, `S8_P1` mean +0.06 despite `E_orig` 2.39); the wrong treatment chosen;
weak training; readout blur of the same size as the threshold. The asymmetry is
explicit: **a positive result does not identify correspondence, and a negative
result does not exclude it.** An experiment with neither implication cannot
change a belief.

# TEST 21 — value of information

What a run could resolve: whether trained aggregation weights produce a
candidate-resolved, ordering-dependent cost profile beyond the untrained
architectural baseline — **Level 3**. That is already established from the other
direction: `EXP-CORRESPONDENCE-TR-001` reached C' (the trained aggregation
realises a candidate-axis ramp the architecture makes available anyway, trained
slope 0.95–1.22 against the forced prediction), and GEOM-002 measured
`R^2 = 0.867` for `Phi(k-tau)` against `0.289` for `Psi(k)` under random weights
across 574/576 units.

What it cannot resolve: anything at Level 4 or above — which is the entire
scientific question.

**Verdict: DO NOT RUN.**

---

# 25. Self-correction — tempting claims, adjudicated

| Claim | Valid? |
|---|---|
| "Random initialisation already produces the full O1 signature" | **NO — rejected.** Measured: random `E_orig` mean 2.385, `S8_P1` max 0.867, 0 % >= 1. Random init produces only the *weak* signature (`S8_P2 != 0`, 48/48). Any prior reasoning that asserted the full conjunction at random init was wrong, and O1 is not refuted on that ground. |
| "Large `S8` proves correspondence" | **NO.** Seven mechanisms give `S8 > 0`; magnitude is set by pi, tau and blur (TESTs 12, 13). |
| "Candidate-axis dependence proves disparity search" | **NO.** Adjacency, smoothing, padding and order sensitivity are all architectural (TEST 9). |
| "Training above random proves geometric use" | **NO.** The latch is above random and has no geometry (TESTs 5, 8). |
| "Correct tau proves correspondence" | **NO.** tau is marked by an exact zero; any magnitude functional finds it (TEST 3). |
| "Permutation sensitivity proves stereo matching" | **NO.** P2's sensitivity is present untrained; P1's is present in a pointwise non-matcher. |
| "Permuting the candidate axis is the one airtight intervention" (leaked prior, `stereonet-diagnostic-confounds`) | **NO — this audit corrects it.** It is airtight only as a *null for the degenerate `shift="none"` volume*, where `V[:,:,pi] == V` by algebra. On a `shift="left"` self-pair, where O1 would actually run, it is not airtight: it is confounded by the latch mechanism, by centre collapse and by padding. The memory record should be amended. |
| "P2 removes the coordinate/readout confound" | **NO — it inverts it.** P2 is identically zero for a clean candidate-pointwise detector and large for untrained noise (measured 48/48 both ways). |
| "A negative O1 would close the correspondence question" | **NO** (TEST 20). |

---

# 26. FINAL DECISION

## A. Surviving evidence FOR O1

O1 genuinely identifies, and identifies cleanly:

* **Non-equivariance of the aggregation under candidate permutation** (P2), a
  real, measurable, exactly-defined quantity. It equals the candidate-adjacency
  plus padding reach of the Conv3D stack.
* **That the trained aggregation is not equivalent to random weights** on a
  candidate-resolved readout — i.e. **Level 3**, learned candidate-index
  structure. The random-weight floor is genuinely beaten (`E_orig` 2.39 and
  centre collapse at random init are a real failure to clear).
* Permutation is, correctly, **outside the candidate-axis translation group**
  that killed GEOM-002, so O1 is not an instance of the specific defect that
  stopped the previous experiment. That much of the successor reasoning is sound.

## B. Surviving evidence AGAINST O1

1. **The correspondence operation is not in the module under test.** `L - R` is
   computed by frozen, un-learned `build_cost_volume`. No permutation of its
   output can make the aggregation's behaviour evidence about correspondence.
   This alone caps O1 below Level 5 and is not fixable by any statistic.
2. **A correspondence-free mechanism realizable in the frozen architecture
   reproduces the full P1 signature.** Measured: the pointwise magnitude latch
   gives `E_orig` 0.15–1.35 and `S8_P1` mean **+2.37**, 70.8 % of cells >= 1,
   while using no adjacency, no index, no geometry and no left/right asymmetry.
3. **P2 is exactly the wrong statistic:** `0.0` in 48/48 cells for that clean
   mechanism, non-zero in 48/48 for untrained noise.
4. **The synthetic construction makes the correct candidate an exact zero**
   against a flat 1.13 background — a detectable singularity, not a matching
   optimum. H_match and H_latch are observationally identical on this data.
5. **`S8`'s sign and magnitude are governed by pi, tau and readout blur.** Pure
   flattening yields mean `S8 = +2.167` over the frozen tau set with no tracking;
   a cyclic pi yields `S8 ~ s` by arithmetic.
6. **No decoy is representable in the class:** R-anomalies are k-constant,
   L-anomalies are diagonal and touch every candidate, candidate-local edits are
   not stereo pairs. The one intervention that would separate the hypotheses
   cannot be built here.
7. **Design defects independent of the above:** no derived threshold, no valid
   null, unfixed pi family and count, unfixed primary treatment, tau and pi
   counted as independent when they are not, `n = 3` exhausted checkpoints.

## C. Strongest remaining uncertainty

Whether the trained aggregation is *in fact* a matcher. This audit does not and
cannot settle that: it never opened a checkpoint, and it is a property of the
scientific question, not of this report. Specifically unresolved without
changing the data/intervention class: whether the trained weights implement a
photometrically tolerant learned similarity (which would generalise off exact
self-pairs) or an exact-zero/magnitude latch (which would not). **Only an
intervention where the anomaly and the geometric match are separable can decide
it — and that requires leaving the synthetic global-shift self-pair class
entirely.** Also unresolved, and deliberately: the numerical size of a trained
checkpoint's `S8`, which this audit refused to measure.

## D. Claim ceiling

**Level 3 — learned candidate-index structure.** Level 4 (geometric
candidate-axis use) is refuted by a realizable counterexample; Levels 5 and 6
are structurally out of reach.

## E. Value-of-information

**DO NOT RUN.**

O1 cannot distinguish correspondence from a non-correspondence mechanism that is
realizable in the frozen architecture and that reproduces its complete signature.
Its only resolvable claim (Level 3) is already established by TR-001's C' and
GEOM-002's random-weight profile covariance. This is not `CLOSE BRANCH` in the
sense of condemning the statistic alone: the failure is in the **intervention
class**, and TEST 6 shows the self-pair construction cannot host the decoy that
would fix it. The synthetic global-shift class was already retired by the
GEOM-002 failure audit; permutation does not remove it from that class, it only
moves outside its translation subgroup. On the brief's four options, **DO NOT
RUN** is the decision for O1, and the branch-level statement of TEST 6 is the
reason.

## F. FINAL VERDICT

### **C — NOT IDENTIFIABLE**

---

# 27. Hard stop

All three conditions of §27 of the brief are established:

1. a non-correspondence mechanism **is** realizable in the frozen architecture —
   the hand-set pointwise magnitude latch, `|.|` exact to 4.44e-16, 111,585
   parameters, unmodified class;
2. it **does** reproduce the complete O1 signature — `E_orig` 0.15–1.35 with
   `S8_P1` mean +2.37 and 70.8 % of cells >= 1, beating the random-weight null
   which never reaches 1;
3. the synthetic construction provides **no** observation that distinguishes it
   from genuine correspondence — the two prescribe the same function where
   `V[tau] = 0` exactly, and no candidate-local decoy is representable.

> **O1 IS NOT AN IDENTIFICATION EXPERIMENT.**

**The O1 branch is closed.** No run, no further O1 audit, no additional auditor,
no threshold tuning, no increase in permutations, seeds, checkpoints or scenes,
no modification of `S8`, no search for a stronger effect.

Per §28, the next action is a research execution decision, not another audit.
This report proposes no successor experiment.

## Provenance

* Measured this audit: `measurements.json`, produced by `audit_numerics.py`
  (random synthetic features; hand-set and random weights in the frozen
  `Aggregation`; frozen `StandardisedDisparityRegression`; no checkpoint opened,
  no optimizer imported).
* Source read: `src/models/stereonet/{cost_volume,aggregation,regression}.py`,
  `phase2/models/scaled_regression.py`,
  `phase2/diagnostics/correspondence_geom/20260913T013006Z/run_geom2.py`.
* Inherited facts: GEOM-001 halt, GEOM-002 G-STOP and failure audit,
  `EXP-CORRESPONDENCE-TR-001` C', `EXP-CORRESPONDENCE-INDEX-001`
  NOT-DEMONSTRATED, the 15-cell halo and corrected mask.
* `phase2/` is untracked in git; this record's integrity rests on mtimes and
  the embedded provenance above, not on version control.
