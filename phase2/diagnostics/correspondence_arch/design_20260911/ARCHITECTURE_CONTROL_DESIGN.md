# DESIGN + PRE-FLIGHT — EXP-CORRESPONDENCE-ARCH-001

**Architecture-only control: is the candidate-axis ordering a property of the
trained aggregation weights, or of the Conv3D candidate-axis operator?**

Design date: 2026-09-11. Record
`phase2/diagnostics/correspondence_arch/design_20260911/`.

**This is a design + pre-flight document. The experiment was NOT executed.** No
checkpoint was loaded, no scene was loaded, no `StereoNet` or repository
`Aggregation` was instantiated, no `disparity_initial` was computed, no
`RESULTS.md` exists. Nothing in `phase2/` outside this directory was created or
modified; INDEX-001 (both executions), INDEX-002, the INDEX-003 design audit and
Stage A are untouched.

**Scope note on §O vs §H.** §O forbids instantiating models; §H requires the
architectural analysis to be completed *before* any real output exists and
explicitly permits "a tiny synthetic tensor test". These are reconciled as
follows: §5–§6 below instantiate **bare `torch.nn.Conv3d` primitives on synthetic
tensors** to establish a property of the *operator class*. No repository model
class, no checkpoint, no dataset and no experimental quantity is involved. The
artifact is `preflight_architecture_analysis.py` in this directory and is
re-runnable. If that reading of §O is wrong, the two analyses in §5–§6 should be
discarded and re-derived analytically; **the design verdict does not depend on
the synthetic runs**, which only confirm an analytic argument.

Evidence tags: **MEASURED** (read from source, or produced by the synthetic
pre-flight), **DERIVED** (algebra/arithmetic), **INFERRED** (interpretation),
**UNKNOWN** (requires the experiment).

---

## FINAL DESIGN VERDICT

```
CLEAN-DESIGN-AVAILABLE
```

with **one substitution and one parameter change**, both made here on structural
grounds before any output exists, and both flagged for authorisation:

1. The circular-rank-TV statistic proposed by the INDEX-003 design audit is
   **rejected** for this comparison — §6 shows it is floor-pinned for exactly the
   orbits this experiment will produce. A replacement primary statistic with
   demonstrated headroom is specified in §8.
2. **16 random initialisations** (seeds 0–15) rather than 3. §D's binding
   constraint is "at least THREE"; 3 yields only 3 random–random pairs, which
   cannot support the decision rule. Seeds 0, 1, 2 are a subset and are reported
   separately for §D compliance. Cost rises from ≈8 s to ≈35 s.

---

## 1. WHAT IS RANDOMISED — determined from source, not from expected results

**MEASURED**, parameter inventory of the pipeline:

| module | source | parameters | role in this experiment |
|---|---|---|---|
| `feature_extractor` | `src/models/stereonet/feature_extractor.py` | yes | **held TRAINED and identical** — it produces the cost volume |
| `cost_volume` | `src/models/stereonet/cost_volume.py` | **none** (pure function of its inputs) | held identical |
| `aggregation` | `src/models/stereonet/aggregation.py` | **yes — 111 585** (`4 × Conv3d(32→32,3³)` + `Conv3d(32→1,3³)`) | **THE RANDOMISED MODULE** |
| `regression` (`StandardisedDisparityRegression`) | `phase2/models/scaled_regression.py` | **ZERO** | held identical (nothing to randomise) |
| `refinement` | `src/models/stereonet/refinement.py` | yes | **never invoked** — endpoint is `disparity_initial` |

**The readout has no learned parameters.** Verified from source: the class holds
only `upsample_first` (bool) and `eps` (float `1e-5`); its docstring states "No
new parameters. No learnable temperature, no annealing, no clipping, no constant
divisor. The module holds no state and no weights, so the parameter count is
identical to the control's 423,586." Its `forward` is `interpolate →
standardise_across_disparity → soft_argmin`, all parameter-free. **§C's question
"if the readout contains learned parameters, must they remain frozen?" is
therefore vacuous: there are none.** This was determined by source inspection,
before any result.

**The feature extractor is NOT randomised.** Source inspection does not place the
ordering mechanism there: the candidate axis does not exist until
`build_cost_volume` stacks the 12 levels, and the hypothesised mechanism (§5) is
the disparity-axis convolution inside `aggregation`. Randomising the feature
extractor would additionally change the cost volume and destroy the isolation §B
requires.

**Primary control therefore = trained aggregation vs. the same aggregation
architecture with untrained random weights, on a bit-identical cost volume.**

---

## 2. WHAT IS HELD IDENTICAL

Everything except the 111 585 aggregation parameters:

- the trained feature extractor of the checkpoint under test;
- the cost volume `V(c, s)`, **built once per (checkpoint, scene) and reused as
  the same tensor object for every weight-set and every orbit position**;
- the candidate intervention `π_m(k) = (k − m) mod 12`, applied by
  `volume.index_select(2, idx)` — identity included, via the identical call;
- the parameter-free readout, `upsample_first` and `eps`;
- the scenes `FOCUS_SCENES = [27, 0, 31, 6]`, split `hailo_val`;
- the mask `GT valid ∧ GT/16 ∈ [2,8]`, ground-truth-derived, model-independent;
- the aggregate `d(x) = np.median(disparity_initial[mask])`, identical to
  INDEX-001/002;
- dtype fp32, `torch.no_grad()`, `eval()`, and the determinism controls.

**Bit-identity verification (§B, §K.1).** Because `V` is literally the same
tensor object across conditions, identity is guaranteed by construction; it is
nonetheless *verified* by recording `sha256` of `V.cpu().numpy().tobytes()` once
per unit and asserting every weight-set's run reports the same digest. Any
mismatch is a **HARD STOP**.

**No new extraction mechanism is required.** INDEX-002's harness already builds
`volume = model.cost_volume(lf, rf)` and reuses it across arms; ARCH-001 reuses
that pattern inside its own directory. **No historical code path is altered.**

---

## 3. UNIT OF ANALYSIS

**MEASURED** from `20260911T015124Z/results.json`: the frozen structure is
**3 checkpoints (seeds 0, 1, 2) × 4 scenes = 12 units** — not "3 seeds × 4
checkpoints"; there are three checkpoints, and the seeds index them.

A **unit** is one `(trained checkpoint c, scene s)` pair. Within a unit the cost
volume is fixed, so all weight-sets are compared **on identical input** — the
pairing §J requires. A trained checkpoint's feature extractor differs from
another's, so `V` differs *between* units; nothing is ever compared across units
at the arm level.

Within unit `u` the weight-sets are:

```
A_c        the trained aggregation of checkpoint c        (1)
R_0 … R_15 untrained random aggregations, seeds 0 … 15   (16)
```

---

## 4. THE RESPONSE AND THE ORBIT

For weight-set `w`, unit `u`, orbit position `m ∈ {0, …, 11}`:

```
V'      = V(c,s).index_select(2, tensor([(k − m) mod 12 for k in range(12)]))
cost    = Agg_w(V')
d_init  = StandardisedDisparityRegression(cost, size)
s^{u,w}_m = float(np.median(d_init[0,0].cpu().numpy().astype(np.float64)[mask_u]))
```

**The complete 12-position orbit is required — YES.** Three independent reasons:

1. The four-arm values `m ∈ {−1,0,+1,+2}` are **already published** for the
   trained condition (INDEX-002, 12/12 strictly increasing), so a four-arm
   statistic has no predictive content on the trained side.
2. `m ∈ {3,…,10}` has **never been computed for any condition** — it is the only
   part of the design with genuine predictive content on both sides.
3. §6 shows the orbit under equivariance is a **wrapped ramp**, whose
   distinguishing features (where it wraps, how sharply) lie outside the
   `{−1,0,+1,+2}` window entirely.

Note `m = −1 ≡ m = 11 (mod 12)`, so the four preregistered arms are the orbit
positions `{11, 0, 1, 2}` and are recovered exactly from the full orbit. No new
intervention is invented (§F).

**No physical-disparity conversion anywhere.** `m` is an exponent of the frozen
generator; `s_m` is in candidate units; the primary statistic (§8) is a
correlation of **z-scored** orbits and is invariant to any affine rescaling of
`s`, so no unit interpretation enters at all.

---

## 5. §H — DOES THE ARCHITECTURE PREDICT AN ORDERED RESPONSE FOR ARBITRARY WEIGHTS?

Completed **before** any real output exists, as §H requires.

### 5.1 Analytic argument

**MEASURED** (`aggregation.py`): `4 × [Conv3d(32→32, 3×3×3, padding=1) +
LeakyReLU(0.01)]` then `Conv3d(32→1, 3×3×3, padding=1)` — five 3-tap,
zero-padded convolutions along `D`, receptive field `1 + 5·2 = 11`.

**DERIVED.** A convolution with shared kernels is translation-equivariant; this
is a property of the **operator**, not of its weights. LeakyReLU is pointwise and
commutes with any index permutation exactly. Therefore, wherever the receptive
field of *both* frames lies strictly inside `[0, 11]`,

```
Agg_w( shift_m(V) )(k)  =  Agg_w(V)(k − m)      for ANY weight set w
```

Exactness requires `[k−5, k+5] ⊆ [0,11]` **and** `[k−m−5, k−m+5] ⊆ [0,11]`, i.e.
`k ∈ {5,6}` and `k−m ∈ {5,6}`.

**Standardisation commutes exactly** with a circular permutation of the candidate
axis: it uses the per-pixel mean and population sd across the 12 candidates, and
a permutation preserves that multiset, so `standardise(P c) = P standardise(c)`.

**The soft-argmin does not.** Its index grid `0…11` is linear, not circular, so
under a circular shift `s_m = Σ_e p(e)·((e+m) mod 12)` — a **wrapped ramp**, not
a monotone function of `m`.

### 5.2 Synthetic confirmation (`preflight_architecture_analysis.py`)

Bare `Conv3d` stack, **arbitrary untrained weights**, synthetic volume:

```
H-1  exact positions where Agg(shift_m V)(k) == Agg(V)(k-m)
   m=1 circular  -> [6]          m=1 zero-fill -> [5, 6]
   m=2 circular  -> none         m=2 zero-fill -> [6]
   m=3 circular  -> none         m=3 zero-fill -> none

H-2  per-position relative error of the circular-shift approximation
   m=1  k00:0.484 k01:0.237 k02:0.074 k03:0.059 k04:0.034 k05:0.018
        k06:0.000 k07:0.011 k08:0.027 k09:0.065 k10:0.070 k11:0.344
   m=2  k00:0.325 k01:0.412 k02:0.274 k03:0.054 k04:0.030 k05:0.045
        k06:0.018 k07:0.016 k08:0.032 k09:0.041 k10:0.062 k11:0.280
```

**MEASURED, synthetic.** Exact equivariance survives at 0–2 of 12 positions and
is gone by `m = 3`. But **approximate** equivariance is strong across the whole
interior — relative error 0.00–0.07 for `k = 2…10` at `m = 1` — and fails only at
the two padded edges (0.48 at `k = 0`, 0.34 at `k = 11`). **With untrained
weights.**

### 5.3 Answer to §H

**YES.** The architecture analytically predicts an approximately shifted — hence
locally ordered — candidate-axis response **for arbitrary weights**. The
observed trained deltas are consistent with, and larger than, a unit shift
(**MEASURED**, INDEX-002: `d(+1)−d(0)` +1.2…+1.9, `d(+2)−d(0)` +1.8…+3.1), which
is what the padded-boundary distortion of an approximately equivariant operator
would produce. No unit slope was ever predicted and none is claimed.

**Consequence:** the *presence* of the four-arm ordering is not identifying.
CASE A is the a-priori more likely outcome. That does not make the experiment
pointless — it makes it decisive, because CASE A would close the ordering line on
evidence rather than on this argument.

---

## 6. §G — IS CIRCULAR RANK TOTAL VARIATION APPROPRIATE HERE? **NO.**

§G requires this to be audited independently rather than inherited. It fails.

**Reason 1 — the statistic was built for a different contrast.** INDEX-003's TV
was designed for *natural generator vs. random generator*, where the null orbit
is scrambled and TV is large. ARCH-001 uses the **same natural generator** in
both conditions; both orbits are expected to be structured, so the statistic's
discriminative range is never exercised.

**Reason 2 — floor-pinning, demonstrated.** From
`preflight_architecture_analysis.py` (synthetic, ideal circular equivariance):

```
sharp peak @3   TV=22.0  range=5.79
broad peak @3   TV=22.0  range=4.72
sharp peak @8   TV=22.0  range=5.11
random-ish A    TV=24.0  range=4.22
random-ish B    TV=28.0  range=4.35
   TV floor for a monotone-around-the-circle rank sequence = 22
```

**MEASURED, synthetic.** Every *structured* profile — sharp or broad, peaked at
3 or at 8 — lands on the floor value 22, because a wrapped ramp is monotone
around the circle whatever its phase or sharpness. TV retains only coarse,
integer-valued sensitivity to fully unstructured orbits (24, 28). If the trained
and random orbits are both wrapped ramps, TV returns 22 for both and the paired
difference is identically zero with no variance.

**Reason 3 — it discards magnitude and shape**, which is where any
weight-dependence would live, and which §I's CASE A/B distinction is actually
about.

**Conclusion.** Per §G, this design **stops short of adopting TV** and states
why. TV is retained as a **reported descriptive only**, for continuity with the
INDEX-003 design audit, carrying the floor caveat. It may not enter any verdict.

---

## 7. WHY THE COMPARISON IS STILL IDENTIFIABLE

If both conditions are approximately equivariant, what can differ?

**DERIVED.** Under equivariance the orbit is `s_m ≈ (s_0 + m) mod 12` smoothed by
standardisation and softmax. Its *phase* is set by `s_0`, and its *amplitude and
sharpness* by how peaked the aggregated cost curve is — and both are
weight-dependent. The architecture fixes that the orbit is a wrapped ramp; it
does not fix which ramp.

So the identifiable contrast is not **"is the orbit ordered?"** (architecturally
forced, not identifying) but **"is the trained orbit distinguishable from one
more draw of the random family?"** — a paired question with an internal
reference. Answering it requires a statistic with headroom, and a baseline of
random-vs-random agreement.

**Headroom, verified synthetically** (same script):

```
G-2  orbit-shape agreement (Pearson r of z-scored 12-position orbits)
   peaked-peaked   n=  3 mean=+0.592 min=+0.276 max=+0.751
   random-random   n= 15 mean=-0.019 min=-0.682 max=+0.542
   peaked-random   n= 18 mean=+0.124 min=-0.552 max=+0.735
```

Continuous, spanning roughly `[−0.7, +0.8]`, not floor-pinned. **MEASURED,
synthetic.**

**Declared confound — phase sensitivity.** The same script:

```
G-3  peak2 vs peak3 : r=+0.751   peak2 vs peak4 : r=+0.276   peak3 vs peak4 : r=+0.751
```

Orbits produced by the *identical mechanism* but different phase can disagree
substantially. **This is exactly why the random–random baseline is mandatory and
is not optional:** random seeds also differ in phase from one another, so the
random–random spread absorbs phase dispersion, and the trained orbit is judged
against that spread rather than against an absolute threshold. `s_0` is reported
per (unit, weight-set) so the phase is visible. This confound is declared now and
must not be "corrected" later.

---

## 8. PRIMARY STATISTIC — frozen here, before any output

**8.1 Per-unit orbit.** `s^{u,w} ∈ R^12` as defined in §4.

**8.2 Within-unit z-score** (population sd across the 12 positions):

```
z^{u,w} = (s^{u,w} − mean(s^{u,w})) / sd(s^{u,w})
```

If `sd(s^{u,w}) == 0` the orbit is degenerate: set `z = 0` and mark the
(unit, weight-set) `degenerate`; agreement is **not** computed and the unit is
reported as such. (This is the expected state of the negative control, §10.)

**8.3 Orbit-shape agreement** between two weight-sets at the same unit — the same
volume, so this is strictly paired:

```
ρ^u(w1, w2) = (1/12) · Σ_m z^{u,w1}_m · z^{u,w2}_m          (Pearson r)
```

**8.4 The two families of pairs, per unit:**

```
TR_u = { ρ^u(A_c, R_j)      : j = 0 … 15 }        16 values   cross
RR_u = { ρ^u(R_j, R_k)      : j < k         }     120 values  internal baseline
```

There is exactly one trained weight-set per unit (the checkpoint defines the
volume), so no trained–trained pair exists within a unit and none is constructed
across units.

**8.5 Descriptive secondaries — reported, never decisive.** Orbit range
`max_m s_m − min_m s_m`; `s_0`; the four-arm ordering indicator
`s_11 < s_0 < s_1 < s_2` (known for trained, predictive for random); circular
rank TV with its §6 caveat; phase-aligned agreement `max_lag ρ` together with the
arg-max lag.

**8.6 No accuracy metric.** Per §E, no EPE, D1 or RMSE is computed at all.

---

## 9. DECISION RULE — frozen here, threshold-free, descriptive

**§J forbids an unjustified p-value, and there is no justified null here.**
Trained and random weight-sets are not exchangeable under any defensible
hypothesis — they differ by construction — so permuting their labels would test
nothing. The rule is therefore a **descriptive paired-separation criterion**,
using the random family's own observed spread as the reference. **No p-value, no
α, no effect-size threshold.**

Per unit `u`:

```
inside_u    = ( min(RR_u) <= ρ <= max(RR_u)  for every ρ in TR_u )
separated_u = ( max(TR_u) < min(RR_u) )
```

Global, over the 12 positive units:

```
CASE A  ARCHITECTURE-INDUCED     iff  inside_u    at all 12 units
CASE B  WEIGHT-DEPENDENT         iff  separated_u at all 12 units
CASE C  ARCHITECTURE-CONTROL-INCONCLUSIVE   otherwise
```

Rationale, fixed in advance: CASE A means the trained aggregation's orbit sits
within the range that random draws of the same architecture produce among
themselves — it is one more member of the architectural family. CASE B means the
trained orbit lies outside that range at every unit. Anything else is CASE C and
**is not forced into A or B** (§I).

Also reported, never decisive: per unit `Δ_u = mean(RR_u) − mean(TR_u)`, the full
distributions, and the §D-recommended `{0,1,2}` sub-analysis.

**`separated_u` is one-sided** (`max TR < min RR`) because the interpretable
weight-dependent outcome is the trained orbit agreeing with the random family
*less* than its members agree with each other. The reverse (`min TR > max RR`) is
recorded as an anomaly and routed to **CASE C**, never to CASE B.

---

## 10. NEGATIVE AND SANITY CONTROLS (§K) — all pre-declared

| # | check | criterion | on failure |
|---|---|---|---|
| K1 | cost-volume identity between conditions | `sha256(V)` identical across all weight-sets at each unit | HARD STOP |
| K2 | candidate permutation implementation | `idx_m == [(k−m) mod 12]`; `idx_0` is the identity list | HARD STOP |
| K3 | identity uses the same `index_select` path | `max_abs_diff(V.index_select(2, idx_0), V) == 0` | HARD STOP |
| K4 | deterministic execution | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `no_grad`, `eval`, fp32 | HARD STOP |
| K5 | random inits differ across seeds | pairwise `sha256` of every randomised tensor distinct across all 16 seeds | HARD STOP |
| K6 | trained checkpoint hashes match history | the four INDEX-001/002 sha256 values, verified at load | HARD STOP |
| K7 | no training occurs | harness imports no optimizer, constructs none, calls no `.backward()`/`.step()`; checkpoints opened read-only | HARD STOP |

**Degenerate `shift="none"` control.** On `NEG_shift_none`, `V(k)` is constant in
`k`, so for **every** weight-set (trained *and* random) and **every** orbit
position:

```
max_abs_diff(V_perm, V) == 0        and        max_abs_diff(d_perm, d_identity) == 0
```

Any non-zero ⇒ **HARD STOP**, positives not run. **Pre-declared consequence:**
the negative orbit is constant, so `sd = 0`, `z` is the zero vector, agreement is
undefined and circular rank TV is 0 for every weight-set. This is the correct
behaviour for a degenerate model and is **declared now so that a degenerate 0 is
never mistaken for a structured result.** The gate criterion is the algebraic
identity alone — never the orbit statistic.

Running the full orbit on the negative control with *random* aggregations is a
genuinely new check: it confirms the invariance is a property of the volume, not
of the trained weights.

---

## 11. RANDOM INITIALISATION (§D) — frozen here

**Procedure.** Exactly the framework default the architecture receives when
instantiated from scratch:

```python
torch.manual_seed(j)
agg = Aggregation(in_channels=32, channels=32, num_layers=4)   # repo class, identical forward code
```

No custom init, no scaling, no re-initialisation, no training.

**MEASURED**, default init of the constituent layer:

```
Conv3d(32,32,3,padding=1)
  weight (32,32,3,3,3), bias (32,)
  kaiming_uniform_(weight, a=sqrt(5)); bias ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in))
  source: torch.nn.modules.conv._ConvNd.reset_parameters
aggregation parameters: 4*(32*32*27+32) + (1*32*27+1) = 111,585
```

**Seed list, frozen: `0, 1, 2, 3, …, 15` (16 control seeds).** These are
**control** seeds, not training seeds. Chosen as the first 16 non-negative
integers — a non-adaptive rule containing §D's recommended `{0,1,2}` as a
subset. **No seed may be added, removed or substituted after any output is seen.**

**Deviation flagged.** §D recommends 3. Three seeds give `|RR_u| = 3`, so the
baseline interval `[min RR, max RR]` is the range of three numbers — far too
crude to support §9's containment rule, and §G-3 shows the random family's
internal spread is wide. 16 seeds give 120 baseline pairs at a cost of ≈35 s
total. §D's binding constraint is "at least THREE"; the recommendation is
superseded on stated structural grounds, **before** any output exists. The
`{0,1,2}` subset is reported separately so the recommended analysis is preserved.

**Recorded per seed:** PyTorch version, CUDA version, init method, seed, every
randomised parameter tensor's name / shape / dtype, and its `sha256`. Spot check
(**MEASURED**, bare `Conv3d`, first layer): seed 0 → `86578c2c4bf5b035…`,
seed 1 → `d86fa15ecd82cce4…`, seed 2 → `1af241b75fabce75…` — distinct, as K5
requires.

---

## 12. COMPUTE BUDGET (§L)

**DERIVED** from INDEX-002's measured throughput (24 640 arms in 264.1 s ⇒
0.0107 s/arm):

| | volume builds | arms | estimate |
|---|---:|---:|---:|
| positive: 12 units × 17 weight-sets × 12 positions | 12 | 2 448 | ≈26 s |
| negative: 4 scenes × 17 weight-sets × 12 positions | 4 | 816 | ≈9 s |
| **total** | **16** | **3 264** | **≈35 s** |

(With §D's 3 seeds: 768 arms, ≈8 s.) Volumes are cached per unit, so feature
extraction runs 16 times, exactly as in INDEX-002. **No training and no
expensive feature recomputation is required; if either becomes necessary,
HARD STOP.**

---

## 13. OUTPUTS TO BE CREATED AT EXECUTION (§M)

**Not created now.** At execution, a new directory
`phase2/diagnostics/correspondence_arch/<UTC_TIMESTAMP>/` containing
`PREREGISTRATION.md`, `RESULTS.md`, `results.json`, `ENVIRONMENT.txt`,
`RELATED_RUNS.md`, `run.log`, `random_init_metadata.json` (per-seed tensor shapes
and hashes), `generator.json` (the 12 orbit permutations, literal),
`statistic.md` (the frozen definition), and the harness with its sha256.
`RELATED_RUNS.md` must point to INDEX-001 both executions, INDEX-002, the
INDEX-003 design audit, and Stage A. The freeze order used by INDEX-002 —
artefacts hashed → preregistration written containing those hashes → execution —
is carried over unchanged.

---

## 14. CLAIM CEILING (§N)

On CASE B the maximum claim is exactly:

> "Candidate-axis response structure is weight-dependent under the tested
> aggregation architecture."

On CASE A:

> "Candidate-axis response structure is reproduced by untrained weights of the
> same architecture; the ordering carries no information about training."

Neither is equivalent to candidate-coordinate sensitivity proven, geometric
correspondence, correct matching, disparity search, correct disparity, or stereo
correctness. A later geometric experiment is required for any of those, and none
is designed, preregistered or launched here.

---

## 15. REQUIRED ANSWERS

**1. Exactly which weights are randomized?**
Only the **aggregation** module's 111 585 parameters — `4 × Conv3d(32→32, 3×3×3,
padding=1)` plus `Conv3d(32→1, 3×3×3, padding=1)`, weights and biases. The
feature extractor stays trained (it produces the volume that must be held
identical); the cost volume has no parameters; the readout has **zero**
parameters, verified from source; refinement is never invoked.

**2. Exactly which inputs are held identical?**
The cost volume `V(c,s)` — the *same tensor object*, `sha256`-verified — the
scenes, the GT mask, the permutation set, the identity `index_select` path, the
parameter-free readout with its `upsample_first` and `eps`, dtype, device and the
determinism controls. Only aggregation weights differ.

**3. Is the complete 12-position generator orbit required?**
**Yes.** The four-arm values are already published for the trained condition, so
only `m ∈ {3,…,10}` carries predictive content; and the orbit's distinguishing
features under equivariance — where it wraps, how sharply — lie outside the
four-arm window. `m = −1 ≡ 11`, so the four preregistered arms are recovered
exactly.

**4. What is the primary statistic?**
Orbit-shape agreement. `s^{u,w}_m = np.median(disparity_initial[mask_u])` at
orbit position `m`; `z^{u,w}` its within-unit z-score across the 12 positions;
`ρ^u(w1,w2) = (1/12) Σ_m z^{u,w1}_m z^{u,w2}_m`; partitioned per unit into
`TR_u` (trained vs each of 16 random, 16 values) and `RR_u` (random vs random,
120 values). **Circular rank TV is explicitly rejected as primary** (§6) and
retained only as a caveated descriptive.

**5. What is the exact decision rule?**
Per unit: `inside_u` iff every `ρ ∈ TR_u` lies in `[min RR_u, max RR_u]`;
`separated_u` iff `max TR_u < min RR_u`. Globally: CASE A iff `inside_u` at all
12 units; CASE B iff `separated_u` at all 12 units; CASE C otherwise. Descriptive
and threshold-free — **no p-value**, because trained and random weight-sets are
not exchangeable under any defensible null (§J). The reverse separation is CASE C,
never CASE B.

**6. Does the architecture analytically predict an ordered response even for
random weights?**
**Yes.** Convolutional translation-equivariance is a property of the operator,
not its weights; LeakyReLU commutes with any permutation; standardisation
commutes exactly with a circular candidate permutation. Synthetic confirmation
with untrained weights: relative equivariance error 0.00–0.07 across the interior
`k = 2…10` at `m = 1`, failing only at the two padded edges (0.48, 0.34). Exact
equivariance survives at just 0–2 of 12 positions, which is why the response is
an *approximate*, distorted shift — consistent with the measured trained deltas
being larger than `m`. No unit slope is predicted or claimed.

**7. If yes, does that make the trained-vs-random comparison still identifiable?**
**Yes — but only for the right contrast.** *Ordering presence* is not
identifying: it is architecturally forced, so both conditions are expected to show
it, and any test of it would be near-guaranteed to pass in both. *Orbit shape* is
identifying: the architecture fixes that the orbit is a wrapped ramp but not which
ramp — phase is set by `s_0`, amplitude and sharpness by the cost curve's
peakedness, and all are weight-dependent. The design is identifiable because the
random–random baseline supplies the reference dispersion (including phase
dispersion, §7) against which the trained orbit is judged, and because all three
outcomes are distinguishable and each is informative. **Honest expectation:
CASE A is the a-priori more likely outcome on §5's evidence.** That is a reason to
run it, not to skip it — CASE A would close the ordering line on measurement
rather than on an argument, which is precisely what INDEX-002 and the INDEX-003
audit left open.

**8. Is execution authorized?**
**No.** This task was design + pre-flight only, and nothing was executed. The
design is complete and ready, and awaits explicit authorisation — which should
cover the two flagged changes: the **rejection of circular rank TV** as primary
(§6) and the **16 control seeds** instead of 3 (§11). Without authorisation of
those two points the specification is not internally consistent and must not be
run.

---

**Nothing was executed. No model of the experiment was instantiated. No
`RESULTS.md` was created. The geometric correspondence experiment remains not
designed, not preregistered and not launched.**
