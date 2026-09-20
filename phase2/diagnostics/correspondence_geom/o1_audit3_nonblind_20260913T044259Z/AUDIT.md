# O1 AUDIT — THIRD PASS (**NOT BLIND**)

Record `phase2/diagnostics/correspondence_geom/o1_audit3_nonblind_20260913T044259Z/`.
2026-09-13 UTC.

---

## 0. THE BLINDNESS REQUIREMENT WAS NOT MET, AND CANNOT BE MET FROM THIS SEAT

The brief requires a **genuinely blind** third auditor and forbids reading,
inferring from, or relying on the two prior O1 audits.

**Both prior audits are in this auditor's working context.** They cannot be
unseen. Declaring blindness would fabricate precisely the independence the
three-seat process exists to buy, and would be a worse failure than any technical
error in the audit below.

**What this record is:** a third pass derived from the frozen source, the frozen
architecture and new measurements, deliberately *not* from the prior audits'
conclusions. Where it agrees with them, **that agreement is not corroboration** —
it is one contaminated opinion, not three.

**How to obtain the real blind seat:** issue this brief in a cold context (a
fresh session, or an isolated agent) carrying only the frozen project state, the
architecture, the GEOM-001/GEOM-002 facts and the O1 proposal — and none of the
audit records under `correspondence_geom/o1_*`.

**Two prior-pass claims are deliberately NOT carried forward here**; §7 and §5
re-derive their subject matter from measurement instead.

Tags: **MEASURED** · **DERIVED** · **INFERRED** · **UNKNOWN**.
Evidence: `measurements.log`, from `a3_joint_null.py` in this record. Frozen
`Aggregation` class, frozen cost-volume operator, frozen standardisation, random
feature fields, randomly initialised or hand-set weights. **No checkpoint was
opened.**

---

# 1. EXECUTIVE VERDICT

```
VERDICT:  C — NOT IDENTIFIABLE
DECISION: DO NOT RUN
```

**But the reason is not the one the brief's §8 anticipates, and this pass found
evidence that cuts the other way.**

**MEASURED, and it undercuts the strongest available attack on O1:** the O1
signature is a *conjunction* — small `E_orig` **and** positive `Δ`. Over 120
randomly initialised units (40 seeds × 3 scenes × 6 `τ`):

```
E_orig   : min 1.364   p05 1.442   median 2.155   max 3.646
Delta_P1 : min -0.233  median +0.011  p95 +0.167  max +0.288

joint (E_orig <= 1.5 AND Delta >= 0.5) :  0 of 120 units
joint (E_orig <= 1.0 AND Delta >= 1.0) :  0 of 120 units
```

**The joint signature is NOT available at random initialisation.** The brief's
§8 hard principle — *if the signature is obtainable from random aggregation it
cannot be evidence of learning* — **does not fire against O1.** An honest audit
has to say so.

**The verdict rests instead on §7, the degeneracy. MEASURED:**

```
tau=2 :  ||V[tau]|| over the band = 0.0  exactly
         ||V[k != tau]||  min 4.727   median 7.977   max 11.568
         gap ratio (min over k!=tau)/(median) = 0.592
```

The matched slice is an **isolated singularity**, not the smallest member of a
graded set. On this construction *"select the geometrically matching candidate"*
and *"detect the anomalous candidate"* are **the same observable**. And the
frozen architecture can express the second without the first (§6).

**The decisive structural point (DERIVED):** the one component of the O1
signature that is *not* architecturally free — small `E_orig` — **requires no
permutation to measure**. The permutation arm contributes only the components
that *are* free. O1's own intervention does no identifying work.

---

# 2. WHAT O1 ACTUALLY MEASURES

**DERIVED, from the frozen modules.**

`V[k,w] = Lf[w+k] − Rf[w]`, shape `(1,32,12,17,71)`.
`A` = 4×[`Conv3d(32→32,3³,p=1)` + `LeakyReLU(0.01)`] + `Conv3d(32→1,3³,p=1)`.
`R` = bilinear upsample on `(y,w)` → z-score across `k` → `softmax(−z)` →
soft-argmin over `k = 0…11`.

**MEASURED:** the aggregation's receptive field along the candidate axis is
`1 + 5×2 = 11` of `D = 12`. The stack can therefore compute an approximately
**global across-candidate statistic** — it is not restricted to local candidate
structure.

Separating the four sensitivities the brief demands be kept apart:

| | what it is | is O1 sensitive to it? |
|---|---|---|
| **A. candidate content** | the values in each slice | **YES** — this is most of `Δ_P1` |
| **B. candidate coordinate/order** | which absolute index a slice occupies | **YES** — the soft-argmin weights are pinned to absolute indices |
| **C. candidate neighbourhood** | which slices are adjacent under the 3-tap kernel | **YES under P2; confounded with B under P1** |
| **D. the stereo correspondence relationship** | that slice `τ` is the one where left and right agree | **NOT SEPARABLY** — §4 |

O1 measures **A + B** (P1) or **C** (P2). It does not isolate **D**, because on
this input **D** has no observable consequence distinct from **A**.

---

# 3. P1 / P2 SYMMETRY ANALYSIS

**DERIVED.**

- **Bilinear upsample** acts on `(y,w)` only → commutes exactly with any `π`.
- **z-score across `k`** is a symmetric function of the twelve values →
  `z(c∘π) = z(c)∘π` exactly.
- **softmax** is pointwise on `z` → equivariant.
- **soft-argmin** weights by absolute index → **not** equivariant. This is the
  only stage at which `π` becomes visible in the readout, and it gives

```
m(c ∘ π)  =  Σ_k k·p(π(k))  =  Σ_j π⁻¹(j)·p(j)  =  E_{j∼p}[π⁻¹(j)]
```

- **aggregation** is **not** equivariant in general; it *is* exactly equivariant
  for any operator that is pointwise along `k`.

**Do P1 and P2 answer the same question? No.**

| | what is applied | what it tests | class of transformation |
|---|---|---|---|
| **P1** | `π` on positions, read out in place | *does the output index follow the content?* | confounds a **physical** change (neighbourhoods) with a **coordinate relabelling** (readout indices) |
| **P2** | `π` on positions, `π⁻¹` on the aggregated cost | *is `A` non-pointwise along `k`?* | isolates the **physical** change |

**The proposal does not say which is primary.** They are different experiments.
Selecting between them after seeing which gives a stronger number would be
result-selection; the brief rightly forbids it and the proposal must not be run
until it is fixed in advance.

**Symmetries O1 preserves:** the per-pixel multiset of input slices; absolute
candidate positions; `D`; the z-score statistics for a pointwise `A`.
**Symmetries O1 breaks:** candidate adjacency (for non-translation `π`); the
association between a slice's content and its readout index.

---

# 4. SYNTHETIC DEGENERACY ANALYSIS

**MEASURED (`measurements.log`, M-B).** In the frozen statistic band, per pixel,
across the candidate axis:

| `τ` | `‖V[τ]‖` | `min‖V[k≠τ]‖` | `median‖V[k≠τ]‖` | ratio |
|---|---|---|---|---|
| 2 | **0.0 exactly** | 4.727 | 7.977 | 0.592 |
| 4 | **0.0 exactly** | 4.727 | 7.965 | 0.593 |

The matched slice is not "the smallest of a graded set". It is **exactly zero**
while the nearest competitor sits at ~59% of the typical slice norm. It is an
**isolated singularity in an otherwise unremarkable distribution**.

**DERIVED consequence, stated as the brief requires:**

> **"Select the geometrically matching candidate" and "detect the anomalous /
> zero candidate" are observationally equivalent on this construction. O1 cannot
> identify correspondence on this construction unless the intervention introduces
> information that distinguishes these mechanisms. It introduces none: a
> permutation relocates the singular slice, and both mechanisms follow it.**

**A known-correct candidate index is therefore not evidence of learned
correspondence here.** It is evidence that *something* latched the one slice that
is trivially distinguishable by any measure whatsoever.

**INFERRED:** this degeneracy is an artifact of translated *self*-pairs. Real
stereo never presents an exactly-zero matched slice — photometric noise,
sampling and lighting guarantee a small but non-zero residual, embedded in a
graded set. The property O1 would be probing does not exist in the deployment
regime.

---

# 5. ALTERNATIVE MECHANISMS

Four columns, as the brief requires. "Signature" = small `E_orig` **and**
positive `Δ_P1`.

| # | mechanism | produces the signature? | realizable in the frozen architecture? | present at random init? | would O1 distinguish it from correspondence? |
|---|---|---|---|---|---|
| 1 | **fixed-lag behaviour** | **YES** | yes | **NO** (M-A) | **NO** — and it is not even a distinct mechanism: the cost volume already computes the match residual, so "prefer lag 0" and "read the match" are the same function on this input (**DERIVED**) |
| 2 | **candidate-axis convolution effects** | **YES** | **yes — demonstrated, §6** | non-pointwise-ness yes; the signature **no** | **NO** |
| 3 | **adjacency sensitivity** | **YES** | **yes — demonstrated, §6** | **YES** for the property itself | **NO** |
| 4 | **absolute candidate-index effects** | **NO** (`Δ_P1 = 0`: `π` permutes content, not positions) | yes | yes | **YES** — the one mechanism O1 excludes |
| 5 | **nonlinear aggregation** | **YES** | yes (the `\|·\|` in §6 is built from LeakyReLU) | no | **NO** |
| 6 | **anomaly / outlier detection in candidate slices** | **YES** | **yes — demonstrated, §6** | **NO** (M-A) | **NO** — §4 |
| 7 | **sharp but incorrect estimator** | **NO** — fails the `E_orig` clause while still giving `Δ > 0` | yes | — | yes, **but via `E_orig`, not via `Δ`**. `Δ` alone admits it |
| 8 | **readout / soft-argmin interaction** | contributes, does not produce alone | n/a | n/a | it *manufactures* `Δ` for any content-following operator (§3) |
| 9 | **padding / boundary effects** | contributes | yes | yes | **MEASURED:** moving the matched slice to each position, the relative cost change at the padded ends (`0.599`, `0.529`) sits **inside** the interior range (`0.502–0.702`). Real, but **not a distinguishable nuisance** |
| 10 | **generic right-image dependence** | necessary, not sufficient | yes | yes | already established elsewhere in the project; O1 adds nothing |

**Mechanisms 1, 2, 3, 5 and 6 reproduce the signature and are not distinguished
from correspondence by O1.** Only mechanism 4 is excluded — and a model whose
output ignores the cost volume entirely is already excluded by the model having
non-trivial accuracy at all. **No experiment is needed to rule out mechanism 4.**

---

# 6. FROZEN-ARCHITECTURE REALIZABILITY

The brief's §9 sets the standard: an abstract "imagine an operator that detects
the zero slice" is **not** admissible. The mechanism must be shown to live inside
the frozen architecture.

**MEASURED — it does.** A non-correspondence operator was instantiated by
**setting weights by hand** inside the frozen `Aggregation` class (no training,
no optimizer, no checkpoint):

- **what it computes:** layer 1 splits each channel into `±`; the LeakyReLU pair
  reconstructs `(1−α)|V_c|`; layer 2 sums to a per-slice norm `n(k)`; layer 3
  applies a 3-tap kernel **along the candidate axis** giving
  `g(k) = n(k) − ½n(k−1) − ½n(k+1)`; layers 4–5 form `−|g|`. The readout's
  `argmin` therefore selects **the slice of greatest local contrast along the
  candidate axis**.
- **why it is non-correspondence:** it never compares left to right. It has no
  notion of "agreement". It responds to *any* slice that stands out from its
  neighbours, in either direction.
- **why it belongs to the frozen architecture:** it uses only the declared
  layers, kernel sizes, padding and `LeakyReLU(0.01)`; the 11-of-12 receptive
  field along `k` is what makes the contrast computation expressible.
- **why it reproduces the O1 signature:**

| operator | `E_orig` | `Δ_P1` reverse | rand A | rand B |
|---|---|---|---|---|
| hand-built **matcher** | 0.783 | 2.768 | 1.854 | 2.628 |
| hand-built **contrast latch** (no matching) | 0.840 | 2.653 | 1.787 | 2.078 |

- **can the distinction be observed without opening trained weights?**
  **Yes — but not by O1.** Injecting a decoy anomaly at a non-matched candidate
  separates them 6/6: the matcher ignores it and keeps reporting `τ`; the
  contrast latch jumps to the decoy. **O1 contains no such decoy**, and §11 shows
  why one cannot be added to this construction.

**HONEST LIMITATION, and it is the strongest objection to this audit's own
argument.** Hand-set weights prove the mechanism is **expressible**. They do
**not** prove gradient descent on KITTI would **find** it. Whether the trained
model is a matcher or a latcher is **UNKNOWN** and is not knowable without
opening a checkpoint, which this audit does not do.

**Why the verdict survives that limitation:** identifiability is not a question
of which mechanism is a priori more likely. It is a question of whether the
experiment can **tell them apart**. §4 shows it cannot — *even if the trained
model is a perfect matcher, O1 would produce a result indistinguishable from the
latcher's.* An experiment that returns the same number under both hypotheses does
not test between them.

---

# 7. RANDOM-INITIALIZATION ANALYSIS

**MEASURED (M-A), and reported against this audit's own thesis.** 40 random
seeds × 3 random scenes × 6 `τ`, one fixed `π`, 120 units:

```
E_orig   : min 1.364  p05 1.442  median 2.155  max 3.646
Delta_P1 : min -0.233 median +0.011 p95 +0.167 max +0.288
joint (E_orig <= 1.5 AND Delta >= 0.5) : 0 / 120
best unit by E_orig : seed 27, E_orig 1.364, Delta +0.191
```

**The O1 signature is NOT architecturally available at random initialisation.**
The brief's §8 principle does not apply to it. Any argument of the form *"random
weights already do this, therefore it proves nothing"* is **wrong for the joint
signature** and must not be used.

**What *is* available at random init, and therefore not creditable to training:**
non-pointwise-ness along `k` — `Δ_P2 ≠ 0` — which follows from the layers being
`3×3×3` convolutions.

**Interpretation.** `E_orig ≈ 1.4–3.6` for random weights means the untrained
readout is near-flat and `m` is pinned near the window centre. A trained model
with small `E_orig` would genuinely be doing *something* random weights do not.
**That something is "latching the singular slice" — and §4 is precisely the
finding that this does not identify correspondence.**

**LIMITATION, STATED:** this null uses **random feature fields**, because the
frozen feature extractor lives inside a checkpoint and is out of bounds for an
audit. Whether real features shift the null is **UNKNOWN** here.

---

# 8. S8 STATISTIC AUDIT

**8.1 Scale.** **Not determined by mechanism.** `m_π = E_{j∼p}[π⁻¹(j)]`, so the
magnitude of `Δ` is governed by where `π⁻¹(τ)` lands relative to `τ` and by how
peaked `p` is. `τ`, `π` and the soft-argmin geometry all set the scale. **Note
(DERIVED):** the z-score removes any *global* rescaling of the aggregated cost
exactly, so the residual nuisance is the *shape* peakedness of the standardised
profile, not its gain.

**8.2 Sign.** **No.** Positive `Δ` does not uniquely imply geometric
candidate-axis structure. It is produced by every content-following operator,
including all of mechanisms 1, 2, 3, 5, 6 — and the sign itself is set by whether
`π⁻¹(τ)` happens to land nearer or further than `m` did.

**8.3 Threshold.** **`S8 ≥ 1` is REJECTED, and no threshold is offered in its
place.** It cannot be derived from the design: (i) the achievable range of `Δ` is
fixed by the `π` and `τ` grids, not by the mechanism; (ii) `m_π` is pulled toward
the window centre `5.5`, so a fully disrupted model appears *least* disrupted at
`τ ≈ 5–6` — an artifact anti-correlated with `τ` for non-geometric reasons;
(iii) the soft readout's own averaging of `π⁻¹` blurs the prediction by roughly
one candidate, so a 1-candidate threshold cannot resolve the effect it thresholds.
**No threshold is invented here.**

**8.4 Null.** The permutation arm is a **paired treatment**, not a null. Pairing
removes weights, scene, `τ`, mask and extractor. It does **not** remove profile
peakedness or the readout blur — both live inside the pair. The only external
reference available is the random-init floor of §7, which does not cover the
joint signature.

**8.5 Inference unit.** **UNSPECIFIED in the proposal.** It must be at least
`(checkpoint, pair, τ, π)`. **`π` must be treated as a random effect**, not as a
fixed constant: permutations sharing a matched-slice destination are strongly
correlated, so a set of `π` is **not** a set of independent observations. `τ`
levels are also correlated through the shared underlying field.

**8.6 Multiplicity.** A max-over-null gate is inadmissible: for any null with
mass `p > 0` above threshold, `P(max over N below) = (1−p)^N → 0`, so the gate's
strictness is a function of the null budget rather than of the model — running
more null units makes the experiment harder to pass. **DERIVED; no replacement
design is offered, per the brief.**

---

# 9. NULL / INFERENCE / MULTIPLICITY AUDIT

Consolidated from §7 and §8.

- **Correct null:** "the output does not follow candidate content" — and it is
  **not** the random-weight arm, which fails the signature for flatness rather
  than for geometric blindness.
- **Matched?** Partially. Weights, input, scene, `τ`, mask and extractor are
  matched exactly. Adjacency, the readout index association, and which slice
  abuts the `k` zero padding all change. The padding change is real but
  **measured not to dominate** (§5, row 9).
- **Unit of inference:** `(checkpoint, pair, τ, π)`, `π` random, correlations
  across `τ` and across `π` explicit.
- **Multiplicity:** max-based gating forbidden; rate/quantile reasoning would be
  required for any future design, which this audit does not produce.

---

# 10. CLAIM CEILING

Set **before** any hypothetical result, per the brief.

| level | | identifiable? |
|---|---|---|
| **0** | candidate-axis output changes under permutation | **YES** — and trivially: true of any content-following operator, and of random weights under P2 |
| **1** | candidate-axis ordering / coordinate sensitivity | **YES** — trivially, same reason |
| **2** | candidate-axis neighbourhood / adjacency dependence | **YES but NOT CREDITABLE TO TRAINING** — present at random initialisation, a consequence of `3×3×3` kernels |
| **3** | learned candidate-index structure | **YES, and this is the ceiling.** The joint signature is *not* available at random init (§7), so a trained model showing it would be doing something untrained weights do not |
| **4** | geometric candidate-axis use | **NO** — §4: anomaly latching is not geometric and is not distinguished |
| **5** | genuine stereo correspondence / disparity search | **NO** |
| **6** | correct disparity on natural scenes | **NO** |

```
HIGHEST LOGICALLY IDENTIFIABLE LEVEL:  3
```

**And the finding that decides the value question (DERIVED):** Level 3 is
established by **`E_orig` alone** — "the trained model reports approximately the
correct candidate on a synthetic translated pair, which random weights do not".
**That measurement requires no permutation.** Everything the permutation arm
adds — Levels 0, 1 and 2 — is architecturally free.

> **O1's own intervention contributes nothing above what is already free. Its one
> non-trivial component is a measurement that does not need O1.**

---

# 11. VALUE-OF-INFORMATION DECISION

```
DECISION:  DO NOT RUN
```

Grounds, in order of weight:

1. **The degeneracy (§4).** `V[τ] ≡ 0` exactly makes matching and anomaly
   detection the same observable. A positive result does not discriminate.
2. **A realizable non-correspondence mechanism reproduces the signature (§6)**,
   built from the frozen layers, proved non-matching by decoy injection.
3. **The permutation does no identifying work (§10).** The one component above
   the random-init floor is measurable without it.
4. **The statistic has no derivable threshold, no interpretable sign and no
   mechanism-determined scale (§8).**

**On the decoy that would discriminate — why it cannot simply be added
(DERIVED):** `V[k,w] = Lf[w+k] − Rf[w]`. An anomaly in `Rf[w]` inflates `V` at
**all** `k` for that `w`, so it is not candidate-selective. An anomaly in `Lf` at
column `c` inflates `V` at `k = c − w`, which **moves with `w`**; across the
20-column statistic band the anomalous candidate takes 20 different values and
any spatially-reduced statistic averages it away. **The discriminating
observation is not expressible in this construction under a spatially reduced
statistic.**

**No successor experiment is recommended, and none is designed here.**

**INFERRED, offered as context rather than as the decision I was asked to make:**
because the degeneracy is a property of the *construction* rather than of O1, the
grounds above would also support the stronger `CLOSE BRANCH`. That is a larger
call than "is O1 worth the compute", and I do not make it here.

---

# 12. FINAL VERDICT

```
C — NOT IDENTIFIABLE
```

Realistic non-correspondence mechanisms — fixed-lag, candidate-axis convolution,
adjacency sensitivity, nonlinear aggregation, and anomaly detection — reproduce
the O1 signature. At least one is **realizable inside the frozen architecture**
and **proved non-matching by intervention**. The synthetic construction does not
distinguish it from correspondence.

**Not D.** The proposal is genuinely under-specified (permutation family, fixed
vs random, single vs ensemble, P1 vs P2, inference unit, null, threshold,
multiplicity — eight open items). But the determination does not depend on how
those blanks are filled: the degeneracy holds for every permutation family and
both readout treatments. **D would understate the finding.**

**Not B.** Level 3 is identifiable, but not *by O1* — it is identifiable by
`E_orig` alone. O1's intervention adds only Levels 0–2, which are free.

---

# 13. HARD STOP

The brief's §16 conditions are met:

1. a non-correspondence mechanism **is** realizable inside the frozen
   architecture — **MEASURED**, hand-set weights, §6;
2. it **does** reproduce the O1 signature — **MEASURED**, `E_orig` 0.840 vs
   0.783, `Δ_P1` +1.787…+2.653 vs +1.854…+2.768;
3. the synthetic construction does **not** distinguish it from correspondence —
   **MEASURED**, §4, and **DERIVED** in §11 that the distinguishing decoy is not
   expressible here.

```
HARD STOP.  O1 IS NOT AN IDENTIFICATION EXPERIMENT.
```

No rescue is to be attempted by new thresholds, larger effect sizes, more
permutations, more checkpoints, more seeds, more scenes, or post-hoc
modification of the statistic. This record proposes none of them.

**Standing state, unchanged by this audit:**

```
Level D — genuine geometric correspondence :  NOT-DEMONSTRATED
evidence FOR correspondence                :  NONE
evidence AGAINST correspondence            :  NONE
trained aggregation measured               :  NEVER
```

**UNKNOWN, and material:** whether the trained aggregation is a matcher or a
latcher — not knowable without opening a checkpoint, which this audit does not
do, and not decidable by O1 even if it were run.

**OUTSTANDING PROCEDURAL DEBT:** a genuinely blind third seat has still not been
provided (§0).
