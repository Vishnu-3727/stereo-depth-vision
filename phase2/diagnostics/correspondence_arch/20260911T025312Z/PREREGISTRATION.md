# EXP-CORRESPONDENCE-ARCH-001 — PRE-REGISTRATION

**Architecture-only control: is the complete candidate-axis response structure
dependent on the TRAINED aggregation weights, or is it adequately explained by
the candidate-axis Conv3D architecture itself?**

Record: `phase2/diagnostics/correspondence_arch/20260911T025312Z/`.
Frozen 2026-09-11, **before any arm of this experiment was executed** and before
any response statistic of this experiment existed.

**This document preserves the design frozen in
`phase2/diagnostics/correspondence_arch/design_20260911/` and authorised without
further change.** Nothing below was written after execution. The authorised
deviations from the original brief are exactly the five listed in §0 and no
others.

Inference only. No training, no optimizer, no fine-tuning, no weight
optimisation, no architecture change, no padding change, no kernel-size change,
no aggregation-depth change, no readout change, no preprocessing change, no
cost-volume-construction change. Phase 1 untouched. INDEX-001, INDEX-002 and the
INDEX-003 design audit untouched and not re-run.

---

## 0. AUTHORISED CHANGES FROM THE ORIGINAL BRIEF

1. Circular rank total variation is **NOT** the primary statistic.
2. **16** random aggregation initialisations, seeds **0–15**.
3. The **full 12-position generator orbit** is used.
4. The preregistered **orbit-shape agreement** statistic is the primary.
5. The **random-vs-random baseline** is used exactly as specified.

No further protocol change is permitted. If execution reveals a discrepancy with
the design audit, execution **stops and reports** rather than adjusting the
protocol.

---

## 1. QUESTION

Is the repeatedly observed candidate-axis ordering behaviour a property of

- **A.** the trained aggregation/readout weights, or
- **B.** the architecture / Conv3D candidate-axis operator itself?

This experiment does **not** target geometric correspondence, correct matching,
disparity search, physical disparity correctness or stereo correctness.

---

## 2. FROZEN CONDITIONS — identical to INDEX-001 / INDEX-002

| item | value |
|---|---|
| configuration | frozen H2 6-block, `StereoNetConfig(cost_volume_shift=<per checkpoint>)` + `phase2.models.scaled_regression.apply_to` |
| feature extractor | the checkpoint's own trained weights, **not randomised** |
| cost-volume construction | `model.cost_volume(lf, rf)`, unchanged |
| cost-volume tensor | built **once per (checkpoint, scene)** and reused as the same object for every weight-set and every orbit position |
| scenes | `FOCUS_SCENES = [27, 0, 31, 6]`, split `hailo_val` |
| mask | `GT > 0 ∧ GT/16 ≥ 2.0 ∧ GT/16 ≤ 8.0`, ground-truth-derived, model-independent |
| dtype / device | fp32 / cuda |
| determinism | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.no_grad()`, `eval()` |
| standardisation | `standardise_across_disparity`, population sd, `eps = 1e-5`, unchanged |
| readout | `StandardisedDisparityRegression` — **parameter-free**, unchanged |
| candidate count | `N = 12` |
| generator convention | `V'(k) = V(π_m(k))`, `π_m(k) = (k − m) mod 12` |
| endpoint | `disparity_initial` only; refinement never invoked |

**Intervention point:** immediately after cost-volume construction and before
aggregation. **For every comparison the pre-aggregation cost volume must be
identical**, verified by `sha256` (§8, K1).

### Checkpoints (read-only, sha256 verified at load; HARD STOP on mismatch)

| key | shift | seed | sha256 |
|---|---|---|---|
| `NEG_shift_none` | `none` | — | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` |
| `POS_6b_seed0` | `left` | 0 | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` |
| `POS_6b_seed1` | `left` | 1 | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` |
| `POS_6b_seed2` | `left` | 2 | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` |

---

## 3. CONTROL CONDITIONS

**TRAINED.** The exact trained aggregation weights from the frozen checkpoint.

**RANDOM.** Only the aggregation module is instantiated afresh:

```
4 × Conv3d(32→32, kernel 3×3×3, padding=1) + LeakyReLU(0.01)
then Conv3d(32→1, kernel 3×3×3, padding=1)
Aggregation(in_channels=32, channels=32, num_layers=4)     # repository class, identical forward code
```

**Exactly 111 585 parameters are randomised.** Asserted at construction; any
other count is a HARD STOP.

**Not randomised:** feature extractor, cost-volume construction, readout,
standardisation, refinement, or anything outside `aggregation`. The readout has
**zero** parameters (verified from `phase2/models/scaled_regression.py`: it holds
only `upsample_first` and `eps`), so there is nothing in it to randomise or
freeze.

**Initialisation:** the framework's normal initialisation —
`torch.nn.modules.conv._ConvNd.reset_parameters`, i.e.
`kaiming_uniform_(weight, a=√5)` and `bias ~ U(−1/√fan_in, +1/√fan_in)` —
applied by `torch.manual_seed(seed)` immediately before construction. No custom
init, no scaling, no re-initialisation, **no training**.

**Seeds frozen before execution: `0, 1, 2, …, 15`.** Control seeds, not training
seeds. The 16 aggregations are constructed **once** and reused identically at
every unit.

**Frozen artefacts, written before this document and hashed into it:**

```
random_initialization_metadata.json  sha256 945e3283f1156eaf6c5a5dd608eff3d15ee9b56440155b98a8895db86ea860e1
generator.json                       sha256 fed055f06934f9626d542ca15e8d391687a2f7a7db46f9b00ef0619849e7bb7c
```

`random_initialization_metadata.json` records, per seed, every parameter
tensor's name, shape, dtype and `sha256`, plus a combined digest. Pairwise
tensor-hash collisions across seeds: **0** (K5 satisfied at freeze time and
re-verified at execution).

---

## 4. FULL GENERATOR ORBIT

One generator `c(k) = (k − 1) mod 12`, powers `c^m` for `m = 0 … 11`, i.e.
`π_m(k) = (k − m) mod 12`. For each unit and each weight-set:

```
s_m = median( disparity_initial[mask] )      for every m = 0 … 11
```

`m = 0` is identity. `m = 11` is the permutation INDEX-001/002 called `−1`.

**The previously observed four-arm result is NOT a new discovery and is not
treated as one.** It is reported descriptively only. The purpose of the full
orbit is to examine the previously unmeasured positions `m = 3 … 10`.

**No physical-disparity conversion.** `m` is an exponent of the frozen
generator; `s_m` is in candidate units; the primary statistic correlates
z-scored orbits and is invariant to any affine rescaling of `s`.

---

## 5. IDENTITY IMPLEMENTATION

Every arm, **including identity**, is built with

```python
volume.index_select(2, torch.tensor([(k - m) % 12 for k in range(12)]))
```

Identity is **not** special-cased. `max_abs_diff(identity_indexed_volume,
original_volume)` is recorded and required `== 0`.

---

## 6. PRIMARY STATISTIC — frozen, exactly as in `design.json`

For unit `u` and weight-set `w`, with `s^{u,w} ∈ R^12` from §4:

```
z^{u,w} = (s^{u,w} − mean(s^{u,w})) / std(s^{u,w})        # population std
```

Agreement between two weight-sets at the same unit (same cost volume — strictly
paired):

```
ρ^u(w1, w2) = (1/12) · Σ_{m=0}^{11} z^{u,w1}_m · z^{u,w2}_m
```

**Zero-variance rule, preregistered:** if `std(s^{u,w}) == 0` the orbit is
degenerate; set `z = 0`, mark the (unit, weight-set) `degenerate`, and **do not
compute agreement** for any pair involving it. This is the expected state of the
negative control (§9).

**Nothing may be altered:** normalisation, centering, denominator, mask, median,
orbit ordering, or the zero-variance rule.

### Pairing / counting convention

```
TR_u = { ρ^u(A_c, R_j)  : j = 0 … 15 }      ->  16 values
RR_u = { ρ^u(R_j, R_k)  : 0 ≤ j < k ≤ 15 }  -> 120 values
```

There is exactly one trained weight-set per unit (the checkpoint defines the cost
volume), so no trained–trained pair exists within a unit and none is constructed
across units.

**No p-value. No permutation of trained/random labels.** Trained and random
weight-sets are not exchangeable under any defensible null. The random-vs-random
distribution is the empirical reference for how much orbit-shape agreement can
arise from random aggregation weights under this architecture.

---

## 7. PRIMARY DECISION RULE — frozen

Per unit `u`:

```
INSIDE_u    :  min(RR_u) <= ρ <= max(RR_u)   for EVERY ρ in TR_u
SEPARATED_u :  max(TR_u) < min(RR_u)
CASE-C_u    :  neither
```

Reverse separation (`min(TR_u) > max(RR_u)`) does **NOT** count as SEPARATED and
is routed to CASE-C.

Global, over the 12 positive units:

```
CASE A : INSIDE    at 12/12 units
CASE B : SEPARATED at 12/12 units
CASE C : anything else
```

This rule is not changed after seeing results. No intermediate verdict exists.

---

## 8. UNIT OF ANALYSIS

A **unit** is one `(trained checkpoint c, scene s)` pair:
**3 checkpoints × 4 scenes = 12 positive units**, plus 4 negative-control units.
Within a unit the cost volume is fixed, so all 17 weight-sets are compared on
identical input.

---

## 9. NEGATIVE CONTROL — first and gating

Checkpoint `NEG_shift_none`. Its volume is degenerate (`reference_shift` is a
no-op, so `V(k) = left_feat − right_feat` for every `k`), hence `V[:,:,π] = V`
element-wise for **any** π. This is an algebraic identity.

Run **all 17 weight-sets × 12 orbit positions × 4 scenes**. Required for every
arm:

```
max_abs( V_perm − V_identity )                              == 0
max_abs( disparity_initial_perm − disparity_initial_identity ) == 0
```

**Any non-zero ⇒ HARD STOP.** Positives are not run and the failure is reported
as an implementation/determinism/layout failure, not interpreted scientifically.

**Pre-declared consequence:** the negative orbit is constant, so `std = 0`, the
orbit is `degenerate`, `z = 0` and agreement is not computed. This is correct
behaviour for a degenerate model and is declared **now** so that a degenerate 0
is never mistaken for a structured result. The gate criterion is the algebraic
identity alone.

---

## 10. SANITY CONTROLS — all pre-declared, all HARD STOP on failure

| # | check | criterion |
|---|---|---|
| K1 | cost-volume identity between conditions | `sha256(V)` identical across all 17 weight-sets at each unit |
| K2 | permutation implementation | `idx_m == [(k−m) mod 12]`; `idx_0` is the identity list; matches `generator.json` |
| K3 | identity via the same `index_select` path | `max_abs_diff(V.index_select(2, idx_0), V) == 0` |
| K4 | deterministic execution | the §2 determinism block |
| K5 | random inits differ across seeds | every parameter tensor's `sha256` distinct across all 16 seeds; and each seed's live tensors match `random_initialization_metadata.json` |
| K6 | trained checkpoint hashes | the four §2 values, verified at load |
| K7 | no training | harness imports no optimizer, constructs none, calls no `.backward()`/`.step()`; checkpoints opened read-only |
| K8 | aggregation architecture identical | parameter count `== 111 585` for every weight-set |

---

## 11. ARCHITECTURAL SANITY CHECK

The already-authorised synthetic architectural analysis
(`preflight_architecture_analysis.py`, carried over unchanged from
`design_20260911/`) is re-run and its output preserved. It confirms that the
architecture can generate approximate wrapped-ramp behaviour with random
weights.

**It is explanatory, not a post-hoc mechanism chosen from the real results.** It
was written and run before this experiment existed, it loads no checkpoint, no
scene and no repository model class, and **real experiment outputs may not be
used to alter it.**

---

## 12. SECONDARY ANALYSIS — descriptive only, never decisive

- the complete 12-position orbit for every trained unit;
- the complete 12-position orbit for every random seed;
- the trained-vs-random `ρ` values;
- the random-vs-random `ρ` values;
- min / max / median distributions;
- the four-arm ordering `s_11 < s_0 < s_1 < s_2`;
- whether `m = 3 … 10` add structure not visible in INDEX-002;
- circular rank TV, carried with its floor caveat from the design audit.

**The four-arm ordering is descriptive only and must not be reinterpreted as
statistical evidence.** No accuracy metric is computed: no EPE, no D1, no RMSE.

---

## 13. COMPUTE

Per unit: one feature extraction + one cost-volume build, cached; then
17 weight-sets × 12 orbit positions = 204 aggregation/readout arms from that same
cached tensor.

```
positive  12 units × 204 =  2 448 arms
negative   4 units × 204 =    816 arms
total     16 volume builds,  3 264 arms
```

Estimated ≈35 s at INDEX-002's measured 0.0107 s/arm. If the implementation
unexpectedly requires training or expensive feature recomputation: **HARD STOP**.

---

## 14. INTERPRETATION — fixed before execution

**CASE A.** Trained orbit shapes are inside the random-vs-random variability at
12/12 units ⇒ *the observed orbit shape is adequately explained by the tested
architecture / random-weight variability.* **Do NOT claim candidate-coordinate
sensitivity.**

**CASE B.** Trained orbit shape is separated from the random-vs-random baseline
at 12/12 units in the preregistered direction ⇒ *the orbit shape is
weight-dependent under the tested architecture.* This makes candidate-coordinate
sensitivity more plausible. It still does **not** establish geometric
correspondence.

**CASE C.** Anything else ⇒ `ARCHITECTURE-CONTROL-INCONCLUSIVE`. **No binary
conclusion is forced.**

---

## 15. CLAIM CEILING

Maximum claim in CASE B, verbatim:

> "Under the tested aggregation architecture, the full candidate-axis orbit
> structure differs from the random-initialized architecture baseline,
> indicating weight-dependent candidate-axis response."

Must **not** be claimed under any outcome: geometric correspondence, correct
matching, disparity search, physical disparity correctness, stereo correctness.

---

## 16. HARD STOPS

Execution stops immediately, preserving the partial record, if: training occurs;
a wrong checkpoint is loaded; the aggregation architecture differs; the feature
extractor is randomised; cost volumes differ between conditions; random seeds are
changed or removed; outputs are inspected and used to alter the protocol; the
negative algebraic control fails; deterministic controls fail; any historical
record would need modification; or a discrepancy is found between the design
audit and execution.

---

## 17. OUTPUTS

Confined to this directory: `PREREGISTRATION.md` (this file), `RESULTS.md`,
`results.json`, `ENVIRONMENT.txt`, `RELATED_RUNS.md`, `run.log`,
`random_initialization_metadata.json`, `generator.json`, `frozen.sha256`,
`orbit_statistics.json`, `freeze_random_init.py`, `run_arch_control.py`,
`preflight_architecture_analysis.py` and `preflight_architecture_analysis.log`.

No historical record is modified. The design in `design_20260911/` is **not**
rewritten after execution.
