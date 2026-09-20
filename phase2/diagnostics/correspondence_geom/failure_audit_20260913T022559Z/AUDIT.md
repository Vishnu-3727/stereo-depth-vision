# FAILURE-MECHANISM AND SUCCESSOR-FEASIBILITY AUDIT

Record `phase2/diagnostics/correspondence_geom/failure_audit_20260913T022559Z/`.
2026-09-13 UTC. Subject: `EXP-CORRESPONDENCE-GEOM-002` (`20260913T013006Z`),
`G-STOP`, verdict `STATISTIC-ARCHITECTURALLY-AVAILABLE`.

**Audit only. Designs nothing, preregisters nothing, launches nothing.**

## Standing constraints on this audit

1. **No trained aggregation is loaded anywhere in this record.** GEOM-002's
   `G-STOP` forbids it, and no measurement below touches a checkpoint. Every
   number comes from (a) the recorded random-weight gate, or (b) the frozen
   cost-volume operator applied to random tensors.
2. **GEOM-002's verdict is not revisited.** Its rules were frozen before
   execution and its outcome stands as recorded.
3. **Post-hoc knowledge is now contaminating.** The author of this audit has seen
   the null distribution and the failure mechanism. Anything proposed here is
   therefore suspect by construction, and §7 says so explicitly rather than
   pretending otherwise.

Tags: **MEASURED** (this record or GEOM-002) · **DERIVED** (algebra on frozen
source) · **INFERRED** · **UNKNOWN**.

Raw output: `measurements.log`. Scripts: `audit_m1.py`, `audit_m2.py`,
`audit_m5.py`, `audit_reparam.py`.

---
---

# 1. FAILURE-MECHANISM AUDIT

## 1.1 The real mechanism is not "constant-offset tracking". It is deeper.

Constant-offset tracking is a **symptom**. The cause is an exact algebraic
property of the intervention that was never written down.

**DERIVED.** With the synthetic construction `Rf[w] = Lf[w+tau]`, substitute
`u = w + tau` and `d = k - tau` into `V[k,w] = Lf[w+k] - Rf[w]`:

```
V_AA[k,w] = Af[w+k] - Af[w+tau] = Af[u+d] - Af[u]
V_BB[k,w] = Bf[w+k] - Bf[w+tau] = Bf[u+d] - Bf[u]
V_AB[k,w] = Af[w+k] - Bf[w+tau] = Af[u+d] - Bf[u]
V_BA[k,w] = Bf[w+k] - Af[w+tau] = Bf[u+d] - Af[u]
```

**All four cells are functions of `(u, d)` alone.** `tau` does not appear. It has
been absorbed entirely into a **joint translation of the candidate axis and the
spatial axis**.

**MEASURED (`audit_reparam.py`, frozen `build_cost_volume`, random tensors, no
checkpoint).** For every cell and every pair `tau1 < tau2`, with `s = tau2-tau1`:

```
max | V^{tau1}[k-s, w+s]  -  V^{tau2}[k, w] |  over the clean band  =  0.0
                                                                     EXACT
60 checks = 4 cells x 15 tau pairs
```

**`tau` is a coordinate change, not an experimental factor.** The aggregation is
convolutional in `(k, y, w)` and the mask was chosen to be free of boundary
effects, so the whole pipeline is equivariant to that coordinate change up to two
residual terms (§1.2). A response that "tracks `tau`" is therefore the *default*
behaviour of **any** operator whatsoever — trained, random, or arbitrary.

This is the same error class as GEOM-001's, one level further in: **the statistic
was verified against the hypothesis, and the group of transformations under which
the statistic is invariant was never computed.**

## 1.2 What the symmetry does *not* fully absorb

**DERIVED.** Exact shift-covariance is broken by exactly two terms:

| term | why it breaks covariance | size |
|---|---|---|
| **standardisation window** | `z` is standardised across `k in [0,12)`, a *fixed* window. In `d` coordinates that window is `[-tau, 12-tau)`, so its mean and s.d. move with `tau`. | moderate |
| **spatial translation of content** | `u = w + tau` shifts *which image content* the mask averages over. | moderate |

**MEASURED (`audit_m1.py`, 576 random units, per-unit RMS-normalised, fitted on
the 7 lags `d in [-1,5]` common to all six `tau`):**

```
model                                R^2 mean   median   p05     min
A  shift-covariant  Ibar_tau(k)=Phi(k-tau)   0.867   0.890   0.662   0.042
B  tau-constant     Ibar_tau(k)=Psi(k)       0.289   0.275     --    max 0.702

model A beats model B in 574 of 576 units
```

**INFERRED.** `Phi(k-tau)` explains ~87% of the profile variance and the residual
is the two terms above. Shift-covariance is the dominant structure, not a
tendency. And it is *strongest* where it hurts most: units scoring `h=6` have the
**highest** covariance (`R^2 = 0.934`) — the units that "passed" are the ones most
completely explained by the symmetry.

## 1.3 Is constant-offset tracking the only reason `h` fails?

**No — there are three distinct failure channels, and only the first is the one
that fired.**

**MEASURED (GEOM-002 gate, 576 units):**

```
offset-constant units        153 (26.6%)   k_hat = tau + c exactly, all six tau
   c = -1 :   7      c = 0 :  31      c = +1 : 60
   c = +2 :  33      c = +3 :  20     c = +4 :  2
   h histogram of these units:  0 -> 122,  6 -> 31,  nothing else
non-constant units           423 (73.4%)   h in 0..5, never 6
```

**The identity is exact: `h = 6` iff (offset constant AND `c = 0`).** 31 = 31, no
exceptions.

| # | channel | what it is | status |
|---|---|---|---|
| **F1** | **candidate-axis translation covariance** | the `(u,d)` symmetry. Makes *tracking* free; only the offset `c` varies. | **the channel that fired.** Kills every statistic invariant under `k -> k + const`. |
| **F2** | **`argmin` is a hard, scale-free operator** | `argmin` discards amplitude and curvature, so it reports the symmetry and nothing else. | amplifies F1; independently fixable |
| **F3** | **the gate rule is a maximum over a large null** | `max(h_rand)` over 576 draws saturates whenever `P(h=6) > ~1/576`. | **independent design defect**; see §5 |

F2 and F3 would have bitten even if F1 did not exist. Treating this as "a
constant-offset problem" would fix one third of it.

## 1.4 Is `c` a property of the weights, of the content, or both?

This matters: if `c` were weight-determined, "`c = 0`" would be one coin flip per
model. It is not.

**MEASURED (`audit_m2.py`):**

```
seeds with at least one offset-constant unit           27 of 32
  seeds whose c is identical across ALL their units      8 of 27
  seeds whose c varies across (extractor, scene pair)    19 of 27
```

**`c` is jointly determined by the aggregation weights and the image content.**
Consequently `c = 0` **simultaneously across 18 independent (extractor, pair)
contexts** is a far stronger property than a single hit — which is precisely what
§5 exploits and what the `max`-based gate threw away.

## 1.5 Where does a random aggregation actually put its minimum?

**MEASURED, mean aligned profile `Phi(d)`, per-unit RMS-normalised, 576 units:**

```
   d      -1       0      +1      +2      +3      +4      +5
 mean  -0.026  -0.066  -0.278  -0.090  -0.159  +0.054  +0.114
 sd     1.042   1.074   1.138   0.947   0.794   0.694   0.690

argmin of the mean aligned profile:  d = +1
per-unit argmin of the aligned profile:
   d=-1: 85   d=0: 79   d=+1: 149   d=+2: 81   d=+3: 73   d=+4: 43   d=+5: 66
```

**Random weights are biased *away* from `d = 0`, toward `d = +1`.** Zero lag is
not privileged for an untrained aggregation — `|V| = 0` there, and a random
linear-nonlinear map sends zero input to an arbitrary output, not a minimal one.
So `c = 0` is *not* architecturally free even though *tracking* is. That
asymmetry is the only exploitable gap the intervention leaves, and §3 and §5
build on it.

---

## 1.6 What must a replacement statistic satisfy to be immune to `k_hat = tau + c`?

**DERIVED.** Write `G` for the group generated by the coordinate change
`(k, w) -> (k - s, w + s)`, `s` integer. §1.1 shows the four-cell family is
*exactly* `G`-invariant on the clean band. Then:

> **Necessary condition.** A statistic `T` can carry correspondence information
> only if `T` is **not** invariant under the action of `G` on the candidate axis.
> Equivalently: `T` must change when the whole response profile is translated
> along `k`.

Four corollaries, each of which kills a family of statistics outright:

| requirement | what it rules out |
|---|---|
| **R-A. Must depend on the absolute candidate index, not only on profile shape.** | `argmin`-at-`tau`, argmax, curvature at the minimum, amplitude, peak width, entropy of the profile, any moment about the peak |
| **R-B. Must not be a function of `k_hat - tau` alone.** | slope of `k_hat` vs `tau` (forced to 1 for *any* operator — this is exactly the `C'` ceiling already recorded in `EXP-CORRESPONDENCE-TR-001`), rank correlation, monotonicity of `k_hat` |
| **R-C. Its null must be calibrated against a control that is *sharpness-matched*, not merely weight-randomised.** | the random-weight gate as the sole null (§5.3) |
| **R-D. It must be broken by a transformation outside `G`.** | any test whose control differs from the treatment only by a translation |

**R-D is the constructive one.** `G` is the group of *translations* of the
candidate axis. A control that **permutes** the candidate axis is outside `G`:
permutation preserves the multiset of slices, the marginal statistics and the
weights' sharpness, but destroys the candidate-axis *geometry*. This is the same
intervention already identified independently in
`[[stereonet-diagnostic-confounds]]` as "the only airtight intervention", and
§1.1 now supplies the reason why: **it is the smallest transformation that leaves
the symmetry group.**

## 1.7 Could another statistic accidentally encode the same symmetry?

**Yes, and three of the obvious candidates do.** Any statistic built from the
profile *shape* is `G`-invariant by construction, because translation does not
change shape. In particular:

- the interaction amplitude `max Ibar - min Ibar` — `G`-invariant, carries zero
  geometric information;
- the second-minimum margin `I_second - I_min` recorded in GEOM-002 — also
  `G`-invariant (this is why the preregistration was right to make it
  descriptive-only, though for a weaker stated reason);
- any "does the response have a single well-defined minimum" sharpness test —
  `G`-invariant.

**The test to apply to any proposed statistic, before anything else:** translate
the whole profile along `k` by a constant and recompute. If the value does not
change, the statistic is dead on arrival.

---
---

# 2. IDENTIFIABILITY AUDIT

## 2.1 The four hypotheses

| | hypothesis | observable signature on a **globally translated self-pair** |
|---|---|---|
| **H-a** | trained geometric correspondence | response peaks at the lag where left and right features agree, i.e. `d = 0` |
| **H-b** | generic right-image dependence | response depends on the right image at all |
| **H-c** | candidate-axis architectural effects | response is some fixed function of the lag `d`, peaking wherever the operator happens to peak |
| **H-d** | nonlinear aggregation artifacts | ditto, with the peak location set by the nonlinearity rather than by geometry |

## 2.2 The hard result

**DERIVED, and this is the central finding of the audit.**

On a **globally translated self-pair**, the entire observable family is a
function of `(u, d)` (§1.1). Under H-a the response peaks at `d = 0`. Under H-c
and H-d it peaks at `d = c` for some operator-determined `c`. **There is no
observable that separates "peaks at `d = 0` because it found the match" from
"peaks at `d = 0` because its fixed lag response happens to be centred".**

> **The distinction H-a vs (H-c, H-d) is NOT IDENTIFIABLE from a globally
> translated self-pair, by any statistic whatsoever.**
>
> This is a property of the **intervention**, not of the statistic, not of the
> mask, and not of the gate. No redesign of the statistic can recover it.

This retires an entire experiment class. It also explains, retrospectively, why
the three earlier geometric attempts and TR-001 all landed on ceilings rather
than determinations: they were all instances of the same non-identifiable class.

H-b is separable and was already established. H-c and H-d are not separable from
each other on this architecture either, and it is not clear the distinction is
meaningful — both are "the operator has a lag response that geometry did not
choose".

## 2.3 What observation *would* separate them

**DERIVED.** Something outside `G`. Three candidates, in increasing cost:

| # | observation | separates H-a from H-c/H-d because | cost |
|---|---|---|---|
| **O1** | **candidate-axis permutation control.** Apply a fixed permutation `pi` to the cost volume's candidate axis before aggregation, same weights, same input. | a correspondence mechanism follows the matched slice to index `pi(tau)`, so its *reported disparity* becomes wrong in a specific, predicted way. A lag-fixed operator is affected differently. `pi` is outside `G`. | one extra forward pass per unit |
| **O2** | **unmatchable regions.** Replace a band of the right crop with content having no counterpart in the left. Correspondence predicts a distinctive response where no match exists; a lag operator has no notion of "no match". | breaks `Rf[w] = Lf[w+tau]` locally, so the `(u,d)` identity fails in that band only | new construction, new domain derivation |
| **O3** | **non-global disparity field on real pairs with ground truth.** | the identity fails everywhere | large; re-enters the natural-scene regime |

**O1 is the only one that reuses the frozen geometry unchanged**, which is
decisive given that the geometry took two halted experiments to get right.

## 2.4 The residual ceiling even for O1

**INFERRED, stated before any successor is considered.** O1 establishes
*candidate-axis geometry is used*. It does **not** establish correspondence on
natural scenes, metric disparity, or generalisation. The strongest claim it could
support is roughly:

> the trained aggregation's reported disparity depends on the candidate axis'
> geometric ordering, in a way that a permutation-matched control with identical
> weights does not.

That is strictly stronger than TR-001's `C'` and strictly weaker than
correspondence. Anyone who wants "the model does stereo matching" will not get it
from this architecture on synthetic input.

---
---

# 3. STATISTIC AUDIT

Enumerated **before** choosing, and each screened against §1.6. The
`G`-invariance column is decisive: invariant means dead.

| # | statistic | `G`-invariant? | has a meaningful random-weight null? | verdict |
|---|---|---|---|---|
| **S1** | `h` = count of `argmin Ibar_tau == tau` (GEOM-002) | **YES** (shape only) | null saturates: `P(h=6) = 0.0538` **MEASURED** | **REJECT** — F1 |
| **S2** | slope of `k_hat` vs `tau` | **YES** | forced to 1.0 for any operator **DERIVED**; TR-001 measured 0.95-1.22 | **REJECT** — R-B, already retired |
| **S3** | `Ibar_tau(tau)` — the response *at* the true candidate | no | but it is `Phi(0)`, constant in `tau`: there is no `tau`-dependence left to test | **REJECT** — tests nothing |
| **S4** | interaction amplitude `max - min` | **YES** | calibratable but measures aggregation gain | **REJECT** — R-A |
| **S5** | margin `I_second - I_min` | **YES** | calibratable, measures sharpness | **REJECT** — R-A (GEOM-002 correctly kept it descriptive-only) |
| **S6** | **rate** of `c = 0` across contexts (not `max`) | **YES** (still `argmin`-based) | **yes and it is strong**: null `0.0538` per unit | **REJECT as primary** — same F1; see §3.2 |
| **S7** | **`\|m - tau\|`**, the soft-argmin **disparity output** against truth | **NO** — uses the absolute `k` index | random-weight null is **degenerate for the wrong reason** (§3.3) | **CANDIDATE, needs a matched null** |
| **S8** | **`\|m - tau\|` original vs candidate-axis-permuted, same weights** | **NO**, and the control is outside `G` | the control **is** the null, and it is sharpness-matched by construction | **RECOMMENDED** |
| **S9** | per-pixel `k_hat(x)` vs a varying field `tau(x)` | partly — still admits `k_hat = tau + c` | needs its own null | defer — requires O2/O3 |

## 3.1 Why `argmin` cannot be rescued by changing the threshold

**DERIVED.** `argmin` of a `G`-covariant profile is `G`-covariant. Any decision
rule applied to `argmin` — a tolerance, a rate, a max, a rank test — inherits the
covariance. The threshold was never the problem.

## 3.2 The one honest thing S6 can still do

**MEASURED.** If the `max`-gate is replaced by a **rate** comparison against the
same null, the statistic has real power:

```
null rate of c == 0 per unit              31/576 = 0.0538   95% CI [0.035, 0.072]
null rate of offset-constant per unit    153/576 = 0.2656
null rate of c == 0 GIVEN constant        31/153 = 0.2026

P(>=18 of 18 trained units at c==0 | null 0.0538) = 1.4e-23
P(>=15 of 18                        )             = 6.4e-17
P(>=12 of 18                        )             = 8.1e-12
```

**But this is exactly the trap the audit exists to name.** The mechanism is
`G`-covariance; `c = 0` is one bit sitting on top of it; a model that is merely
*centred* passes. S6 would produce a small p-value that means less than it looks
like. **It is admissible only as a secondary, explicitly-labelled descriptive
measure, never as the primary.** Recording it here, pre-commitment, is what stops
it from being discovered later and promoted.

## 3.3 Why S7's random-weight null is the wrong null

**MEASURED (`audit_m5.py`, 576 random units, soft-argmin output `m` in candidate
units, median over the mask):**

```
cell    slope of m vs tau              R^2 med   median |m - tau|
 AA     mean +0.008  med +0.003          0.685        2.144
 BB     mean +0.010  med +0.011          0.699        2.127
 AB     mean +0.000  med +0.001          0.502        2.093
 BA     mean +0.003  med +0.002          0.391        2.084

AA slope within +-0.5 of 1.0 :   0 / 576
AA median |m - tau| <= 1.0   :   0 / 576
```

**Not one random unit of 576 produces a disparity output that tracks `tau`.** So
`m` is *not* architecturally free, unlike `argmin`.

**INFERRED — and the reason matters more than the number.** `m` is a softmax
expectation over the candidate axis. For random weights the standardised profile
is flat, so the softmax is near-uniform and `m` collapses to the window centre
(~5.5) regardless of `tau`. The random null therefore fails for **flatness**, not
for **geometric blindness**.

Consequence: a trained model that is merely *sharper* than random would beat this
null while inheriting tracking for free from `G`. **A statistic whose null fails
for a reason unrelated to the hypothesis is not a test.** This is the
"scientifically interpretable minimum effect" problem in its exact form, and it
is why S7 needs S8's control.

---
---

# 4. PAIRING / INTERVENTION AUDIT

## 4.1 Does AA/BB vs AB/BA give the causal contrast we need?

**Partly — it gives a real contrast, but not the one the campaign needs.**

**What it does deliver (DERIVED + MEASURED).** The 2x2 removes every additive
per-image main effect exactly:

```
V_AA + V_BB - V_AB - V_BA
  = (Af[u+d]-Af[u]) + (Bf[u+d]-Bf[u]) - (Af[u+d]-Bf[u]) - (Bf[u+d]-Af[u]) = 0
```

pointwise, content-independent, valid inside the halo and inside the zero-fill.
**MEASURED on real features in GEOM-002:** max relative residual `1.809e-07`
over 108 checks, reproducing the domain audit's `1.8089e-07`. So the design does
isolate the *pairing* factor from image identity. That part works and is not in
question.

**What it does not deliver.** The contrast it isolates is
`matched-pair vs crossed-pair` **at a fixed lag coordinate `d`** — and §1.1 shows
`d` is all that survives. The intervention answers *"does pairing matter?"* (H-b,
already established) and cannot answer *"is the lag at which pairing matters the
geometrically correct one?"* — because every operator has *some* lag at which
pairing matters.

**The pairing is sound; the factor it crosses with is degenerate.**

## 4.2 What should cancel at each level — and what must NOT be claimed

| level | quantity | cancels? | status |
|---|---|---|---|
| **raw cost volume** | `V_AA + V_BB - V_AB - V_BA` | **exactly 0**, pointwise, content-independent | **DERIVED + MEASURED** `1.809e-07` (float32 round-off) |
| **aggregation, affine part** | the same combination after the conv stack | **exactly 0**, receptive field and padding included, because an affine map sends an identically-zero input to an identically-zero output | **DERIVED**; design-audit synthetic control (LeakyReLU slope 1.0) `<= 2.3e-06` relative |
| **aggregation, nonlinear** | ditto with the real LeakyReLU | **NO** | design-audit measured `0.71 - 0.86` relative — five orders of magnitude larger |
| **readout `z`** | after standardising across `k` | **NO — and no affine claim survives here** | the `z`-score is nonlinear (division by a `tau`- and cell-dependent s.d.) |
| **statistic `Ibar`** | median over the mask | **NO** | the median is not linear |

> **The warning stands, unchanged and explicit: there is NO affine-cancellation
> claim after the nonlinear standardisation.** The cancellation identity licenses
> exactly one thing — that per-image additive main effects are removed *at the
> raw cost volume*. It licenses nothing about the statistic's behaviour, which
> must be calibrated empirically. GEOM-001 died from conflating these two; this
> audit does not repeat it.

## 4.3 The one thing the intervention still has going for it

**MEASURED.** The geometry is now *settled*: pixel construction exact 216/216;
`Rf[w] == Lf[w+tau]` bit-exact on `w in [15, 55-tau]` over 4 406 400 values;
`|V[:, tau, :, 15:45]| == 0.0` exactly over 3 525 120 cells; mask `44 352` px
provably invariant to arbitrary halo content (24/24 bit-identical).

Any successor that keeps the same crops, the same `tau`, the same mask and the
same clean band **inherits all of that at zero cost**. That is the single
strongest argument for O1 over O2/O3 (§2.3).

---
---

# 5. POWER / CALIBRATION AUDIT

## 5.1 The gate rule was unsatisfiable, and this is provable without any data

**DERIVED — no measurement required.**

The frozen rule is: `H* = max(h_rand)` over `N` units; `G-STOP` if `H* = 6`; and
on `G-PASS`, require `min(h_trained) > H*`.

Let `p = P(h = 6 | random unit)`. Then:

```
P(gate passes) = (1 - p)^N
```

- With `N = 576`, the gate passes with probability >= 0.5 only if
  `p <= 1 - 0.5^(1/576) = 0.00120` — the null must produce a perfect unit **less
  than once in 831 draws**.
- Worse, the comparison rule: if `H* = 6` then `min(h_trained) > 6` is
  **impossible**, since `h <= 6`. **The design is unfalsifiable-in-the-positive
  whenever any single random unit scores 6.**
- And even on `G-PASS` with `H* = 5`, `min(h_trained) > 5` requires **all 18**
  trained units to score a perfect 6.

**The gate's strictness scales with the size of the null population, which is a
compute-budget choice, not a property of the model.** Running *more* random seeds
makes the experiment *harder to pass*. That is a diagnostic of a broken rule.

**MEASURED confirmation:** `p = 31/576 = 0.0538`, so
`P(gate passes) = (1 - 0.0538)^576 = 1.4e-14`. **GEOM-002 was decided before it
ran.** The 241 seconds of compute measured the null; they did not decide
anything.

## 5.2 The rule that should replace it

**DERIVED.** A null population is characterised by its **rate**, not its
**maximum**. The maximum of `N` draws is not a statistic of the null — it is a
statistic of `N`.

Requirements for any successor gate:

| # | requirement |
|---|---|
| **P-1** | The null population and its budget are fixed **before** the statistic's threshold is set, and the threshold is expressed as a **rate or a quantile**, never as a maximum. |
| **P-2** | The trained and null arms are measured at the **same budget per unit** and over the **same contexts** (same extractors, pairs, `tau`). |
| **P-3** | The null must be **matched on every nuisance property the statistic is sensitive to** — for `m`-based statistics that means *sharpness*, which weight-randomisation does not provide (§3.3). |
| **P-4** | The **minimum interpretable effect** is preregistered as a number, with its justification, before execution. |
| **P-5** | A `G-STOP` is declared when the null's **rate** exceeds a preregistered ceiling — not when its maximum touches the top of the scale. |

## 5.3 Can a finite random gate separate trained from random at all?

**It depends entirely on the statistic, and the answer differs by statistic.
MEASURED:**

| statistic | null behaviour | can a finite random gate separate? |
|---|---|---|
| `h` (argmin) | rate `0.0538`, max `6` | **as a max-gate: no, never.** As a rate-gate: yes with large power (`1.4e-23` at 18/18) — but it measures the wrong thing (§3.2) |
| `\|m - tau\|` | `0/576` units anywhere near correct | **yes numerically, but the null fails for flatness, not blindness** — so the separation is not interpretable (§3.3) |
| `\|m - tau\|` vs **permuted-axis control** | control shares the weights, hence the sharpness | **yes, and interpretably** — the control differs from the treatment in exactly one property |

**The lesson generalises past this campaign:** a weight-randomised null is a
floor, not a control. It answers "is the effect present at all", never "is the
effect due to the named mechanism". Every statistic needs *both*: the random
floor **and** a nuisance-matched control.

## 5.4 Minimum interpretable effect (to be preregistered, not chosen later)

**INFERRED, offered as the number a successor should have to commit to.** For the
S8 contrast, with 18 contexts (3 checkpoints x 6 pairs) x 6 `tau` = 108 paired
observations:

```
statistic     Delta = median |m - tau|_permuted  -  median |m - tau|_original
                      (candidate units, paired by (checkpoint, pair, tau))

minimum interpretable effect   Delta >= 1.0 candidate  (= 16 px disparity)
                               AND  median |m - tau|_original <= 1.0 candidate
```

Rationale for `1.0`: the candidate quantisation is 16 px; an effect smaller than
one candidate cannot be distinguished from readout smoothing. The second clause
is what stops "the control is worse" from being reported as "the model is right"
— a model can beat its control while still being wrong in absolute terms.

---
---

# 6. HARD-STOP AUDIT

GEOM-002's eleven hard stops all behaved correctly and none fired. The gap is not
in their enforcement, it is in **what they check**. Every existing stop verifies
**wiring**. None verifies that the **statistic is capable of answering the
question**.

## 6.1 The stop that was missing, and would have caught this

> **HS-SYM — symmetry stop.** Before any trained weight is loaded: compute the
> group of transformations under which the intervention leaves the observable
> family invariant, and verify that the primary statistic is **not** invariant
> under it. Assert numerically, on random tensors, with no model.

**This is cheap and it is decisive.** `audit_reparam.py` in this record runs in
under a second, uses no checkpoint, and returns `0.0` — it would have halted
GEOM-002 *before the preregistration was frozen*, let alone before execution.

## 6.2 The full set a successor should carry, in the order they must fire

| id | when | condition | why |
|---|---|---|---|
| **HS-SYM** | **pre-freeze** | the primary statistic is not invariant under the intervention's symmetry group; asserted numerically on random tensors | catches GEOM-002's failure at design time, at zero compute |
| **HS-NULL-RATE** | **pre-trained** | the null **rate** of the primary statistic is below a preregistered ceiling. Not its maximum. | catches F3; makes the gate independent of budget |
| **HS-NULL-MATCH** | **pre-trained** | a **nuisance-matched** control exists and has been measured, not only a weight-randomised floor | catches §3.3 — a null that fails for the wrong reason |
| **HS-DEGENERATE** | **pre-trained** | the null is not degenerate in the *other* direction either: if `0/N` null units produce any effect at all, verify that is not an artifact of an unrelated property | the `0/576` in §3.3 looks like a strong null and is not one |
| **HS-CEILING** | **pre-trained** | the strongest claim the design can support is written down **before** results exist, and is checked to be worth the compute | prevents discovering the ceiling after the fact |
| HS1..HS11 | as in GEOM-002 | geometry, pairing, cancellation, checkpoints, determinism, integrity, mask, finiteness, no-training | unchanged; all held |

## 6.3 Preventing the specific failure the brief named

The brief asked to prevent *"trained result looks impressive, then we discover
the statistic was architecturally available"*. Three mechanisms, in order of
strength:

1. **HS-SYM at design time.** The strongest, because it fires before the
   preregistration is frozen and costs nothing. Architectural availability is a
   *derivable* property; it should never be discovered empirically.
2. **The two-process HS7 split**, already in place and already proven: the trained
   arm lives in a separate process that cannot start without a passing gate.
   GEOM-002 confirms it works — the trained aggregation was destroyed at load and
   no trained number was ever produced.
3. **Preregistering the claim ceiling before execution.** GEOM-002 did this
   (A.16) and it is why this `G-STOP` produced no rescued narrative.

**What GEOM-002 got right and must be kept:** the gate ran *first*, in its own
process, and the result was terminal. The `31/576` was found *before* any trained
number existed to be tempted by. That ordering is the only reason this record can
state the failure cleanly instead of withdrawing a result.

---
---

# 7. SUCCESSOR FEASIBILITY

## 7.1 What has actually been bought

| asset | status |
|---|---|
| the corrected geometry and mask | **settled, verified on real features at tolerance zero.** Reusable at zero cost |
| the 2x2 cancellation on real features | **verified**, `1.809e-07` |
| the symmetry that kills `argmin` statistics | **now known and exactly measured** (`0.0`) |
| the random-weight null for `argmin` and for `m` | **measured**, 576 units |
| the harness, hard stops, two-process split | **working**, reusable |
| a determination at Level D | **none. Ten experiments, zero.** |

## 7.2 The decision

**A successor is justified — but NOT another instance of this experiment class.**

Continuing the pairing-swap probe with a repaired statistic would be a category
error: §2.2 shows the distinction is **not identifiable** from a globally
translated self-pair *by any statistic*. Repairing `h` would produce a fourth
geometric record with a fifth ceiling.

The successor that is justified is a **different experiment class**: the
**candidate-axis permutation control on the disparity output** (O1 + S8). It is
the only option that

- leaves the symmetry group `G` (permutation is not a translation) — **R-D**;
- reuses the settled geometry, mask and harness unchanged;
- has a **nuisance-matched** null (same weights, same sharpness, same marginals)
  — **P-3**;
- was independently identified as the airtight intervention before this audit
  existed (`[[stereonet-diagnostic-confounds]]`), which is weak but real evidence
  that it is not a post-hoc construction.

**Conditions, all of which must hold or the answer becomes "stop":**

| # | condition |
|---|---|
| **C-1** | The primary statistic passes **HS-SYM** at design time, asserted numerically before the preregistration is frozen. |
| **C-2** | The null is the permutation control at **equal budget**, plus the random-weight floor reported separately as a floor, never as the control. |
| **C-3** | The **minimum interpretable effect** (§5.4) is preregistered as a number, before execution. |
| **C-4** | The gate is a **rate/quantile** rule, never a maximum (**P-1**, **P-5**). |
| **C-5** | The claim ceiling (§2.4) is written into the preregistration and is checked to be worth the compute **before** running. |
| **C-6** | Every resolution of an under-specified point is declared in the preregistration, as GEOM-002 did in its Part B. That practice is the reason this failure is legible. |

## 7.3 What remains wrong no matter what

Stated now, so it cannot be presented as a discovery later:

1. **The same three checkpoints, for the eleventh time.** Three seeds of one
   recipe. Independence is gone and cannot be recovered without training new
   models, which the campaign has not authorised.
2. **The designer has read the null.** This audit's author has seen the failure
   mechanism and the offset distribution. S8 was selected with that knowledge.
   The mitigation — derive the requirement (§1.6) rather than shop the null — is
   real but partial.
3. **The ceiling is low** (§2.4). Even a clean S8 result does not say "the model
   does stereo correspondence".
4. **Base rate.** Ten experiments, ten ceilings or halts, each defect found only
   after freezing. The prior that an eleventh will find an eleventh defect is
   not small, and `HS-SYM` exists precisely because the last two defects were
   both derivable in advance and were not derived.

## 7.4 Recommended chain, if authorised

Not started here. Nothing below is designed, frozen or run in this record.

```
1. SYSTEM PROMPT        state the question, the permutation control, the
                        statistic, the ceiling; require HS-SYM and P-1..P-5
2. INDEPENDENT AUDIT    adversarial review: compute the symmetry group of the
                        NEW intervention and prove the new statistic is not
                        invariant under it.  This is the step GEOM-001 and
                        GEOM-002 both skipped, and it is the only step that has
                        ever caught a defect cheaply
3. PREREGISTRATION      Part A verbatim + Part B declared resolutions, as
                        GEOM-002 did; hard stops HS-SYM, HS-NULL-RATE,
                        HS-NULL-MATCH, HS-DEGENERATE, HS-CEILING + HS1..HS11
4. EXECUTION            permutation control first, trained arm in a separate
                        process behind the gate, exactly as now
```

**If C-1 through C-6 cannot all be met, the correct outcome is to stop at
`CORRESPONDENCE NOT DEMONSTRATED` and record the synthetic-translation class as
exhausted for the stated reason (§2.2) — which is itself a real result, and a
more useful one than an eleventh ceiling.**

---

# 8. WHAT THIS AUDIT ESTABLISHES

**MEASURED**

1. `tau` is an exact coordinate change on the four-cell cost-volume family:
   `max residual = 0.0` over 60 checks, frozen operator, random tensors.
2. The response profile is dominantly shift-covariant under random weights:
   `R^2 = 0.867` mean for `Phi(k-tau)` vs `0.289` for `Psi(k)`; 574/576 units.
3. `h = 6` **iff** the offset is constant and zero; `31 = 31`, exactly.
4. The offset `c` is jointly weight- and content-determined: it varies across
   `(extractor, pair)` for 19 of 27 seeds.
5. Random weights are biased **away** from zero lag, toward `d = +1`.
6. The soft-argmin disparity output does **not** track `tau` under random
   weights: slope `+0.008`, and `0 / 576` units within 1.0 candidate.
7. `P(GEOM-002 gate passes) = 1.4e-14`. It was decided before it ran.

**DERIVED**

8. Any statistic invariant under candidate-axis translation is dead on this
   intervention — including `argmin`, slope, amplitude, margin and curvature.
9. The `max`-based gate is unsatisfiable for any null with non-trivial mass at
   the ceiling, and gets stricter as the budget grows.
10. **H-a is not identifiable from a globally translated self-pair, by any
    statistic.** The fix must change the intervention, not the statistic.
11. Candidate-axis **permutation** is the minimal transformation outside the
    symmetry group, and is therefore the minimal valid control.

**NOT ESTABLISHED**

```
Level D -- geometric correspondence :  NOT-DEMONSTRATED   (UNCHANGED)
evidence FOR / AGAINST correspondence : NONE, either direction
trained aggregation measured          : NEVER, in GEOM-002 or in this audit
```

**UNKNOWN**

- whether the trained aggregation's disparity output tracks `tau` at all — not
  measured, and not measurable without loading trained weights, which this audit
  does not do;
- whether the permutation control behaves as §2.3 predicts — a prediction, not a
  result;
- whether any of this transfers to natural stereo pairs.
