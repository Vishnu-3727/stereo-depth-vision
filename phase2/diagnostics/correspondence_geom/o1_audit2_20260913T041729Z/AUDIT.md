# ADVERSARIAL AUDIT #2 — O1 CANDIDATE-AXIS PERMUTATION / S8 STATISTIC

Record `phase2/diagnostics/correspondence_geom/o1_audit2_20260913T041729Z/`.
2026-09-13 UTC. Second auditor seat, same brief.

**AUDIT ONLY.** No checkpoint opened. No trained aggregation loaded,
constructed from, or inferred about. No historical record modified. No
preregistration, threshold, permutation choice, seed list or run schedule
produced.

Evidence: `measurements.log`, from `a2_realizable.py` and
`a2_p2_and_boundary.py` in this record. Both use the **frozen** `Aggregation`
class, the **frozen** cost-volume operator and the **frozen** standardisation,
on random feature fields. Weights are either freshly randomised or **set by
hand**. No checkpoint file is read.

Tags: **MEASURED** · **DERIVED** · **INFERRED** · **UNKNOWN**.

---

## 0. DECLARED CONTAMINATION OF THIS SEAT

**This is not an independent seat in the strict sense, and saying otherwise
would be the first dishonesty in the chain.** The prior audit of this same
proposal (`o1_adversarial_audit_20260913T033554Z`) is in this auditor's working
context. It cannot be unseen.

Two consequences, both acted on:

1. This audit does **not** re-derive the prior audit's findings and present them
   as independent corroboration. Independent corroboration is not available from
   this seat and is not claimed.
2. This audit is therefore spent on what a contaminated second seat *can*
   legitimately do: **attack the first audit's load-bearing claims**, and measure
   the three things it asserted but did not test. Two of those tests **change**
   the first audit's conclusions — one strengthens it to a higher evidentiary
   standard, one **refutes its fallback claim**, and one finds it **overstated**.

A genuinely blind third seat is still owed, and §11 records that.

---

# 1. EXECUTIVE VERDICT

```
DECISION:  C — NOT IDENTIFIABLE
```

Same letter as the prior audit, **on partly different grounds, and stricter**:
the prior audit preserved a fallback claim at a lower level. **That fallback does
not survive measurement.** After this audit, **no claim level survives O1 at
all** — not correspondence, not geometric candidate-axis use, and not the
neighbourhood-dependence fallback.

Three findings, all **MEASURED**:

**F1 — the decisive counterexample is architecturally realizable, and provably
not a matcher.** The prior audit rejected O1 partly on an *abstract*
outlier-latching function. An abstract function is inadmissible unless the
architecture under test can express it. It can. A **hand-built** aggregation —
weights set by hand inside the frozen `Aggregation` class, no training, no
checkpoint — implementing *local contrast along the candidate axis* and nothing
else is proved non-corresponding by intervention, and still reproduces the
correspondence signature:

| operator | `E_orig` | Δ_P1 reverse | Δ_P1 rand A | Δ_P1 rand B |
|---|---|---|---|---|
| `HC_match` hand-built matcher | 0.783 | 2.768 | 1.854 | 2.628 |
| `HC_contrast` hand-built, **no matching** | 0.840 | 2.653 | 1.787 | 2.078 |

**Proof that they are different functions** (spike injection at a non-matched
candidate `k*`, 6/6 cases): the matcher **ignores** the decoy and keeps reporting
`τ`; the contrast latch **jumps to the decoy**.

```
spike k*= 8, tau=2 :  matcher argmin = 2 (correct)   contrast argmin =  8 (decoy)
spike k*= 9, tau=3 :  matcher argmin = 3 (correct)   contrast argmin =  9 (decoy)
spike k*=10, tau=2 :  matcher argmin = 2 (correct)   contrast argmin = 10 (decoy)
```

They agree **only** when the matched slice is the sole anomaly in the volume —
which is exactly and only the synthetic case O1 proposes to run.

**F2 — the fallback claim is already true at random initialisation.** The prior
audit's "highest defensible level" was the P2 reading: *the aggregation is not a
pointwise function of each candidate slice*. **MEASURED over 432 random-weight
units** (12 seeds × 6 permutations × 6 `τ`):

```
units with a non-zero cost deviation under P2 : 432 of 432   (100%)
relative deviation |A(V∘π)∘π⁻¹ − A(V)| / max|A(V)| :
        mean 0.874   median 0.786   min 0.531
```

A **randomly initialised** aggregation is non-pointwise in `k` in **every** unit,
at ~80% relative magnitude. The P2 claim is a property of the **architecture**,
not of training. It cannot separate a trained model from random weights, so it is
not a finding — it is a restatement of "the layers are `3×3×3` convolutions".
**The prior audit called this "close to vacuous by prior probability". It is
vacuous by measurement.**

**F3 — the prior audit overstated the padding objection.** It listed the `k`-axis
zero padding as a control mismatch. **MEASURED:** moving the matched slice to
each position in turn, the relative cost change at the padded ends (`0.599` at
`k=0`, `0.529` at `k=11`) sits **inside** the interior range (`0.502 … 0.702`).
The padding asymmetry is real but is **not** distinguishable from the generic
permutation effect at this measurement. **That objection should be withdrawn as a
separate defect.** It does not rescue O1 — the effect it was cited against is
large for every destination — but the audit chain should not carry an
overstatement forward.

---

# 2. FORMAL INTERVENTION DEFINITION

**DERIVED.** `V[k,w] = Lf[w+k] − Rf[w]`, `(1,32,D=12,17,71)`. Aggregation
`A = Conv3d(32→1,3³) ∘ (LeakyReLU(0.01) ∘ Conv3d(32→32,3³))⁴`, zero-padded on
`(k,y,w)`. Readout `R`: bilinear upsample on `(y,w)`, z-score across `k`,
`softmax(−z)`, soft-argmin with weights `k = 0…11`.

For a bijection `π` of `{0,…,11}`, `(V∘π)[k] := V[π(k)]`.

```
P1 :  m_π = R( A( V ∘ π ) )                 the proposal as written
P2 :  m_π = R( A( V ∘ π ) ∘ π⁻¹ )           the variant the proposal omits
S8 :  Δ = |m_π − τ| − |m − τ|
```

**MEASURED (new here):** the aggregation's receptive field along `k` is
`1 + 5×2 = 11` of `D = 12` candidates. **DERIVED consequence:** the architecture
can compute an approximately global across-candidate statistic. This is the
technical fact that makes F1's counterexample realizable and was not established
in the prior audit.

---

# 3. SYMMETRY-GROUP ANALYSIS

The prior audit's symmetry results are in this auditor's context and are **not**
re-derived here as corroboration (§0). This audit adds one item the prior one did
not state, and it is the one that matters for F1.

**DERIVED.** `A(V∘π) = A(V)∘π` holds exactly **iff `A` is pointwise along `k`**.
The prior audit used this to argue that no `π` separates two pointwise
mechanisms — correct, but it left a gap: *are the mechanisms of interest
pointwise?* They are not. The frozen aggregation has an 11-wide receptive field
along `k`. So the relevant invariance class is **not** "pointwise operators" but
the strictly larger class of **operators whose response to this particular volume
is dominated by the single degenerate slice**, pointwise or not.

`HC_contrast` is in that larger class and is **not** pointwise — it uses a 3-tap
kernel along `k`. It still reproduces the signature (F1). **The prior audit's
identifiability argument was therefore narrower than the failure it was
describing.** The failure is not about pointwise operators. It is about the
degeneracy.

**The invariance that actually defeats S8:**

> On a translated self-pair, `V[τ] ≡ 0` **exactly**. Every operator whose response
> is driven by *that slice being unlike the others* — by smallness, by local
> contrast, by outlier-ness, pointwise or convolutional — reports `τ`, and has
> its report moved by `π⁻¹` under permutation. S8 depends on the operator only
> through which slice it latches, and on this input every such operator latches
> the same one.

---

# 4. ALTERNATIVE-MECHANISM ANALYSIS

Signature = small `E_orig` **and** positive `Δ_P1`.

| id | mechanism | same signature? | basis |
|---|---|---|---|
| **A1** fixed-lag shortcut | **YES** | the cost volume already performs the match; "lag-0 preference" and "match readout" are the same function on this input |
| **A2** candidate-axis convolution artifact | **YES, and now realizably so** | `HC_contrast` is a 3-tap `k`-convolutional operator, hand-built in the frozen class, proven non-matching by spike injection, signature identical — **MEASURED** |
| **A3** absolute-index positional bias | **NO** | a content-blind operator gives `Δ_P1 = 0` (`π` permutes content, not positions) — **DERIVED**. But see §10: this exclusion is worthless, the alternative is already excluded by the model having non-trivial KITTI accuracy at all |
| **A4** candidate adjacency sensitivity | **YES** | `HC_contrast` *is* pure adjacency sensitivity with no matching — **MEASURED** |
| **A5** sharp but wrong estimator | **NO**, but excluded by the absolute-error clause, not by `Δ` | a confidently-wrong matcher has large `E_orig` while `Δ` stays positive; `Δ` alone admits it |
| **A6** nonlinear aggregation artifact | **YES** | `HC_contrast` uses the LeakyReLU stack to form `\|g\|`; that is the whole mechanism |
| **A7** readout interaction | **NO artifact of its own** | the z-score and softmax are exactly permutation-equivariant; the readout relabels the soft-argmin weights by `π⁻¹` and nothing more |
| **A8** anomaly latching | **YES — now demonstrated, not postulated** | F1 |

**§5 of the brief, answered directly.** A model with candidate-axis convolution,
candidate-index dependence, arbitrary nonlinearity and arbitrary learned
candidate filters, **and no correspondence mechanism at all** — can it produce a
systematic S8 improvement?

> **YES. MEASURED, with a constructed instance.** `HC_contrast` has all four
> properties, is provably not a matcher, and produces
> `Δ_P1 = +1.787 … +2.653` with `E_orig = 0.840`.

**What O1 establishes instead:** that the model's output moves when the candidate
content is re-indexed. Nothing more. §10.

---

# 5. PERMUTATION ANALYSIS

`π` is unspecified (fixed vs random, single vs ensemble, P1 vs P2) — four blanks,
and P1/P2 test different hypotheses. That alone would be verdict **D**; it is
subsumed by **C** because F1 holds for every way of filling them: `HC_match` and
`HC_contrast` track each other under every permutation tested.

**Labels vs positions — the distinction the brief demands.**

- **Labels** (reorder the soft-argmin weights only): the aggregation is untouched.
  Tests the readout, carries no information about the aggregation.
- **Positions** (P1): changes the `k`-convolution's neighbours **and** relabels
  the readout weights — two effects confounded.
- **Positions with `π⁻¹` restored (P2)**: isolates the neighbourhood effect — and
  by **F2** that effect is present at random initialisation in 432/432 units.

**What a genuine correspondence mechanism must do:** report `π⁻¹(τ)`, a point
prediction. **MEASURED here (new):** `HC_match`, an exact hand-built matcher,
does report it — its hard `argmin` is `[1,2,3,4,5,6]` on the clean volume. **But
so does `HC_contrast`** (`[0,2,3,4,5,6]`, differing only at `τ=1` where the
matched slice abuts the `k=0` padding). The point prediction does not separate
them either.

---

# 6. S8 STATISTIC ANALYSIS

**6.1 Symmetry.** S8 is not invariant under `G` — necessary, not sufficient, and
`HC_contrast` is the proof that it is not sufficient. The invariance that
actually defeats it is §3's degeneracy class.

**6.2 Absolute-index dependence.** S8 separates content-driven from index-driven
and nothing finer. That is one bit, and §10 shows it is a bit already known.

**6.3 Sharpness.** A sharper-but-wrong model gives positive `Δ`; only the
absolute-error clause rejects it. Any design reporting `Δ` without that clause is
broken. **Refinement (MEASURED here):** the z-score removes *global gain*
exactly, so the nuisance is profile **shape** peakedness, not scale — a point the
brief's framing of "sharpness" leaves ambiguous.

**6.4 Sign.** Not uniquely interpretable. The sign is set by whether `π⁻¹(τ)`
lands nearer or further than `m` did — a property of the arbitrary `π`.

**6.5 Magnitude — the 1.0-candidate threshold is REJECTED.** Three reasons, none
of which depends on GEOM-002's numbers:

1. **The scale is set by `π` and `τ`, not by the mechanism.** With `π` free, `Δ`
   for one fixed mechanism spans a range wider than the proposed threshold.
2. **`m_π = E_{j∼p}[π⁻¹(j)]` is pulled toward `5.5`**, so a fully disrupted model
   looks least disrupted at `τ ≈ 5–6`. The design's own effect is anti-correlated
   with `τ` for reasons unrelated to geometry.
3. **The threshold equals the method's own blur.** The soft readout averages
   `π⁻¹`, so even an exact matcher misses its own point prediction by ~1
   candidate. A threshold at 1.0 cannot resolve the effect it is thresholding.

**No threshold on `Δ` is derivable from the design as specified.**

---

# 7. SYNTHETIC COUNTEREXAMPLE ANALYSIS

The brief requires counterexamples for six mechanisms. The prior audit supplied
them as abstract functions. **This audit supplies the decisive pair as
weight-settings of the actual frozen module**, which is the standard the brief's
§5 ("do not accept 'the permutation destroys geometry' as sufficient") implies.

| mechanism | realized as | `E_orig` | Δ_P1 (rev / rA / rB) | distinct from the matcher? |
|---|---|---|---|---|
| genuine correspondence | `HC_match`, hand-set weights | 0.783 | 2.768 / 1.854 / 2.628 | — |
| **adjacency contrast, no matching** | `HC_contrast`, hand-set weights | 0.840 | 2.653 / 1.787 / 2.078 | **NO** |
| fixed-lag 0 | same function as the matcher | — | — | **NO — not a distinct mechanism** |
| constant-offset tracker | shifted matcher | large `E_orig` | positive | yes, via `E_orig` only |
| absolute-index bias | content-blind | — | **0.000** | **YES** |
| random Conv3D | frozen class, random weights | ~2.0 | ~0 | yes, but for flatness |
| sharp wrong-lag | shifted matcher | large `E_orig` | positive | yes, via `E_orig` only |

```
VERDICT ON §7:   NOT IDENTIFIABLE.
```

Two mechanisms that are **provably different functions** (spike test, 6/6)
produce materially the same observable prediction in every column.

**A closure result, new here (DERIVED).** The spike test *does* separate them —
so why not make it the experiment? Because the decoy cannot be built inside this
construction. `V[k,w] = Lf[w+k] − Rf[w]`. A spike in `Rf[w]` inflates `V` at
**all** `k` for that `w` — not candidate-selective. A spike in `Lf` at column `c`
inflates `V` at `k = c − w`, which **moves with `w`**; across the 20-column mask
the anomalous candidate takes 20 different values, so the spatial median removes
it. **The one intervention that separates the two mechanisms is not expressible
in the frozen synthetic pipeline with a spatially-reduced statistic.**

---

# 8. NULL / CONTROL ANALYSIS

| # | question | answer |
|---|---|---|
| 1 | correct null? | "`Δ = 0`: the output does not follow candidate content". The permutation arm is a **paired treatment**, not a null |
| 2 | random aggregation: floor or control? | **FLOOR**, and a degenerate one — it sits near zero because the standardised profile is weakly peaked and `m` is pinned near the window centre in **both** arms, not because of geometric blindness |
| 3 | is the permutation control the true null? | **No.** It is the treatment |
| 4 | paired on identical weights? | **Yes.** The one sound element of the proposal |
| 5 | does pairing remove nuisance variation? | **Partly.** Removes weights, scene, `τ`, mask, extractor. Does **not** remove profile peakedness or the `E[π⁻¹]` blur — both live inside the pair |
| 6 | unit of inference? | **UNSPECIFIED.** Must be at least `(checkpoint, pair, τ, π)` with `π` a **random effect** |
| 7 | rate/quantile statistic? | Not specifiable until 1–3 and the `π` ensemble are defined |
| 8 | can the null saturate? | Saturation is not the failure mode. The failure mode is that `Δ`'s scale is set by `π` |
| 9 | bigger budget perversely harmful? | **Only under a max rule**, which is forbidden: `P(max over N below threshold) = (1−p)^N → 0` for any null with mass `p > 0`. **DERIVED here, not inherited** |
| 10 | can the gate pass and fail? | **Numerically yes; interpretably no.** A pass is reproduced by `HC_contrast` |

## 9. CONTROL VALIDITY — "nuisance-matched"?

**Matched:** input, weights, cost volume, scene, `τ`, mask, feature extractor,
candidate count, the per-pixel multiset of input slices, absolute candidate
positions.

**Changed:** candidate-axis adjacency (for non-translation `π`); the Conv3D
receptive neighbourhood along `k`; which slice abuts the `k` zero padding; the
multiset of *aggregated* costs; the soft-argmin weight labelling.

**Not changed, contrary to intuition:** the z-score statistics for a pointwise
operator (exactly equivariant), and absolute candidate position.

**On the padding specifically — correcting the prior audit.** **MEASURED:**
relative cost change when the matched slice is moved to each position:
`k=0 → 0.599`, `k=11 → 0.529`, interior `0.502 … 0.702`. The padded ends are
**inside** the interior range. The padding is a real asymmetry but **not a
distinguishable nuisance at this measurement**, and should not be carried forward
as a separate defect.

**The control is not "perfect" — but the reason is not the padding.** It is that
peakedness and the readout blur are inside the pair, and that the two arms differ
in a way (`content re-indexed`) that every content-driven mechanism responds to.

---

# 10. CLAIM CEILING

| level | supported? |
|---|---|
| candidate-axis **ordering** dependence | **YES under P1** — and worthless: the only alternative it excludes is a model whose output ignores the cost volume entirely, which is already excluded by the model having non-trivial KITTI accuracy. **No experiment is needed to know this.** |
| candidate-axis **neighbourhood** dependence | **NO — refuted by F2.** True in 432/432 randomly initialised units at ~0.79 relative magnitude. A property of `3×3×3` convolutions, not of training |
| learned disparity-index structure | **NO** |
| geometric candidate-axis use | **NO** — F1 |
| stereo correspondence | **NO** |
| correct disparity search | **NO** |
| natural-scene correspondence | **NO** |
| metric depth | **NO** |

```
HIGHEST DEFENSIBLE LEVEL:  NONE THAT IS NOT ALREADY KNOWN.
```

**This is the substantive disagreement with the prior audit.** That audit set the
ceiling at the neighbourhood-dependence claim and called it "close to vacuous".
**F2 shows it is not a claim about the trained model at all.** O1 has no
surviving output that would not have been known without running it.

---

# 11. REMAINING METHODOLOGICAL DEFECTS

1. **Under-specification.** `π`, fixed/random, single/ensemble, P1/P2 — four
   blanks, and P1 vs P2 changes the hypothesis. Verdict **D** on its own;
   subsumed by **C**.
2. **Designer contamination (brief §11).** O1/S8 is **post-hoc**. Prior mention
   of candidate-axis permutation in an earlier diagnostic note is **not**
   independent preregistration. Mathematically justified ≠ preregistered before
   observing GEOM-002 — and this audit finds the mathematical justification
   **insufficient** regardless of provenance, so the contamination is not
   load-bearing for the verdict.
3. **Auditor contamination (this seat).** §0. The chain has two audits of this
   proposal written with the first in context. **A blind third seat is owed**, and
   should be given the proposal *without* either audit.
4. **Same three checkpoints (brief §12).** `n = 1` training recipe; three seeds
   bound seed-noise only. Agreement across them is **not** replication — they
   share data, schedule, architecture and initialisation family. They have been
   inspected repeatedly, so any threshold set on them is fitted to them. **What is
   lost precisely:** no inference about the training procedure is available; a
   surviving claim is a claim about **those three artifacts**, not about
   StereoNet-trained-this-way. They must never be described as independent
   replications.
5. **The synthetic degeneracy is load-bearing.** `V[τ] ≡ 0` exactly — a signal
   real stereo never presents. It is what makes matching and anomaly-latching
   the same observable (F1), and every design on this construction inherits it.
6. **The frozen readout blurs the only sharp prediction available**, by ~1
   candidate — the size of the proposed threshold.

---

# 12. MINIMUM HARD STOPS

Conditional, per the brief: these apply only if some *different* design is
pursued. They are **not** an endorsement of O1/S8.

| id | when | condition |
|---|---|---|
| **HS-SYM** | pre-freeze | derive the symmetry group; verify non-invariance on random tensors, no checkpoint. **Strengthened by F1:** non-invariance under `G` is necessary and *not* sufficient; the stop must additionally verify the statistic distinguishes operators **within the degeneracy class** — the class that actually defeated S8 |
| **HS-CEX** | pre-freeze | the counterexample suite must be **executed**, and each counterexample must be **realized as weights of the frozen module**, not written as an abstract function. Each must carry an **intervention proof** that it is not the mechanism under test — the spike test is the template. Halt if any reproduces the claimed signature |
| **HS-ARCH-NULL** | pre-freeze | **new, from F2.** Any claimed property must be shown **absent at random initialisation** before it may be attributed to training. A property present in 432/432 random units is an architecture fact |
| **HS-BLUR** | pre-trained | the design's irreducible blur measured and shown **smaller** than the preregistered threshold. A 1.0-candidate threshold fails this |
| **HS-NULL-MATCH** | pre-trained | a floor sitting at zero must be shown not to sit there for an unrelated reason |
| **HS-NULL-RATE** | pre-trained | rate/quantile only; **max-over-null forbidden**, by `(1−p)^N` |
| **HS-CEILING** | pre-freeze | the highest defensible claim written first and checked **against the executed counterexample suite**, not against intuition |
| HS1–HS11 | as in GEOM-002 | geometry, pairing, cancellation, checkpoint identity, determinism, integrity, mask, finiteness, no-training — all held; retain unchanged |

---

# 13. FINAL DECISION

```
C — NOT IDENTIFIABLE
```

A non-correspondence mechanism that is **realizable in the frozen architecture**
and **proved non-corresponding by intervention** reproduces the O1/S8 signature
under every permutation tested. The fallback claim the prior audit preserved is
**already true at random initialisation in 432/432 units** and is therefore not a
claim about training.

**The successor path as proposed is terminated, with no surviving lower claim.**

---

# 14. CONDITIONS FOR ANY FUTURE SUCCESSOR

Not a design. What any future proposal must clear before it is worth auditing:

1. **Break the degeneracy, or leave the construction.** `V[τ] ≡ 0` exactly is the
   root cause: it makes "find the match" and "find the odd slice" the same task.
   **DERIVED closure result (§7):** the decoy that separates them is *not
   constructible* in this pipeline under a spatially-reduced statistic, because a
   left-image anomaly at column `c` is candidate-selective only at `k = c − w`,
   which moves across the mask and is removed by the median.
2. **The one opening this audit found, stated as INFERRED and not designed:** a
   **per-pixel** statistic (no spatial reduction) would not suffer that
   cancellation, so a decoy-slice design might be expressible there. It would
   need its own domain derivation, its own null, and its own counterexample
   suite. This audit does not design it and does not endorse it.
3. **Every counterexample realized as weights, with an intervention proof.**
   HS-CEX. The abstract-function standard is too weak; the hand-built standard
   found a defect the abstract one only postulated.
4. **Every claimed property checked against random initialisation first.**
   HS-ARCH-NULL. This costs seconds and would have retired the P2 fallback before
   it was written down.
5. **Never treat the three checkpoints as replications.**
6. **If 1 cannot be met, stop.** Recording the synthetic translated self-pair
   class as exhausted — for the proven reason — is a real result, and more honest
   than a confident `Δ ≈ +2` that a hand-built non-matcher also produces.

---

# 15. WHAT THIS AUDIT ESTABLISHES

**MEASURED**

1. The aggregation's candidate-axis receptive field is 11 of 12 — the
   architecture can compute a near-global across-candidate statistic.
2. A non-matching operator (`HC_contrast`) is realizable as hand-set weights of
   the frozen module and reproduces the matcher's O1/S8 signature
   (`E_orig` 0.840 vs 0.783; `Δ_P1` +1.787…+2.653 vs +1.854…+2.768).
3. Spike injection separates them 6/6: the matcher ignores the decoy, the
   contrast latch jumps to it. They are different functions.
4. Under P2, **432 of 432** randomly initialised units are non-pointwise in `k`,
   relative deviation mean `0.874`, median `0.786`, min `0.531`.
5. The `k`-padding asymmetry is present but **not** distinguishable from the
   generic permutation effect (`0.599` / `0.529` at the ends vs `0.502…0.702`
   interior).

**DERIVED**

6. `A(V∘π) = A(V)∘π` iff `A` is pointwise along `k` — and the defeating
   invariance class is larger than "pointwise": it is every operator driven by the
   single degenerate slice.
7. The separating decoy is not constructible in this pipeline under a spatially
   reduced statistic.
8. A max-over-null rule is forbidden by `(1−p)^N`.

**CORRECTIONS TO THE PRIOR AUDIT**

9. Its fallback claim (neighbourhood dependence) is **refuted** — F2.
10. Its identifiability argument was **narrower than the failure**: the problem is
    the degeneracy, not pointwise-ness — F1/§3.
11. Its padding objection is **overstated** and should be withdrawn — F3.

**NOT ESTABLISHED**

```
Level D — geometric correspondence :  NOT-DEMONSTRATED   (UNCHANGED)
evidence FOR / AGAINST correspondence : NONE, either direction
trained aggregation measured          : NEVER
```

**UNKNOWN**

- how peaked the trained model's standardised profile is — not measurable without
  loading it, which this audit does not do;
- whether a per-pixel decoy design (§14.2) is viable — untested;
- what a blind third seat would conclude — this seat could not supply it.
