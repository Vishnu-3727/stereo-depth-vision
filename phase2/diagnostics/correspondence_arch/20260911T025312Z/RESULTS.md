# EXP-CORRESPONDENCE-ARCH-001 — RESULTS

**Architecture-only control for candidate-axis response structure**

Record `phase2/diagnostics/correspondence_arch/20260911T025312Z/`.
Pre-registered in `PREREGISTRATION.md`, frozen before any arm ran. The 16 random
aggregation initialisations and the 12-position generator orbit were frozen in
`random_initialization_metadata.json` (sha256 `945e3283…`) and `generator.json`
(sha256 `fed055f0…`) **before** the preregistration was finalised, and their
hashes are recorded in it.

Inference only — 16 cost-volume builds, **3 264 aggregation/readout arms**,
40.9 s wall clock. No training, no optimizer, no fine-tuning, no architecture /
padding / kernel / depth / readout / preprocessing change. Only the aggregation
module's 111 585 parameters differ between conditions. Refinement never invoked.
Phase 1 untouched. INDEX-001, INDEX-002 and the INDEX-003 design audit untouched
and not re-run.

```
VERDICT: CASE-A-ARCHITECTURE-INDUCED
         INSIDE at 12/12 positive units
```

**A discrepancy between the design's expectation and the measurement was found.
It is reported in §6 and was NOT used to alter the protocol or the verdict.**

---

## 1. NEGATIVE ALGEBRAIC CONTROL — PASS

`NEG_shift_none` has `V(k) = left_feat − right_feat` for every `k`, so
`V[:,:,π] = V` element-wise for any π. Tested across **all 17 weight-sets**
(1 trained + 16 random) × 12 orbit positions × 4 scenes:

| quantity | value |
|---|---:|
| arms checked | **816** |
| `max_abs(V_perm − V_identity)` | **0.0** |
| `max_abs(disparity_initial_perm − disparity_initial_identity)` | **0.0** |
| `max_abs(identity_index_select − original volume)` | **0.0** |
| violations | **0** |

The degenerate model cannot respond, by algebra rather than by expectation —
**and this now holds for random aggregation weights too**, confirming the
invariance is a property of the cost volume, not of the trained weights. This is
a strictly stronger gate than any previous run: INDEX-001 tested 4 permutations
with one weight-set, INDEX-002 tested 1 540 with one weight-set, this tests 204
per scene across 17 weight-sets.

Every negative orbit is constant, so `std = 0`, `z = 0`, and agreement was not
computed — exactly as pre-declared in `PREREGISTRATION.md` §9. Classification
`DEGENERATE-NOT-CLASSIFIED`. **The gate criterion was the algebraic identity
alone.**

---

## 2. COST-VOLUME IDENTITY AND SANITY CONTROLS — ALL PASS

| # | check | result |
|---|---|---|
| K1 | cost volume identical across all 17 weight-sets at every unit | **PASS** — `sha256(V)` re-computed per weight-set, single distinct value at every one of the 16 units |
| K2 | permutation implementation `idx_m == [(k−m) mod 12]` | **PASS** — verified against `generator.json` for all 12 positions |
| K3 | identity via the same `index_select` path, not special-cased | **PASS** — `max_abs_diff(V.index_select(2, idx_0), V) = 0.0` at every unit and weight-set |
| K4 | deterministic execution | **PASS** — `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `no_grad`, `eval`, fp32 |
| K5 | random inits differ across seeds, and match the freeze | **PASS** — 0 tensor-hash collisions across all 16 seeds at freeze time; every live tensor re-verified against `random_initialization_metadata.json` at load |
| K6 | trained checkpoint hashes match INDEX-001/002 | **PASS** — all four sha256 verified at load |
| K7 | no training | **PASS** — no optimizer imported or constructed, no `.backward()`, no `.step()`, checkpoints opened read-only |
| K8 | aggregation architecture identical | **PASS** — parameter count `111 585` asserted for every one of the 17 weight-sets |

Cross-run reproduction: the trained orbit positions `m = 11, 0, 1, 2` reproduce
INDEX-002's four arms exactly (e.g. `POS_6b_seed0 / 000187_10`:
`3.8579 / 5.6118 / 7.5528 / 8.3431`). Determinism held across experiments.

---

## 3. TRAINED ORBIT SUMMARIES — the full 12 positions

`s_m = median(disparity_initial[mask])`, candidate units.

| checkpoint | scene | m=0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed0 | 000187_10 | 5.61 | 7.55 | 8.34 | 7.97 | 7.23 | 6.92 | 6.82 | 7.47 | 7.45 | 3.40 | 2.13 | 3.86 |
| seed0 | 000160_10 | 5.83 | 7.72 | 8.40 | 8.26 | 7.66 | 7.44 | 7.45 | 7.49 | 7.19 | 3.90 | 2.24 | 3.99 |
| seed0 | 000191_10 | 5.58 | 7.56 | 8.43 | 8.12 | 7.26 | 7.60 | 7.83 | 7.89 | 7.33 | 2.80 | 2.09 | 3.84 |
| seed0 | 000166_10 | 5.12 | 7.04 | 8.25 | 7.86 | 7.45 | 7.53 | 7.34 | 7.62 | 7.18 | 6.25 | 1.86 | 3.34 |
| seed1 | 000187_10 | 4.23 | 5.32 | 6.10 | 6.67 | 7.15 | 7.41 | 7.31 | 6.77 | 6.46 | 2.18 | 1.93 | 2.99 |
| seed1 | 000160_10 | 4.29 | 5.38 | 6.09 | 6.70 | 7.17 | 7.53 | 7.33 | 6.55 | 5.65 | 2.34 | 1.95 | 3.07 |
| seed1 | 000191_10 | 4.22 | 5.33 | 6.06 | 6.65 | 7.18 | 7.55 | 7.42 | 6.67 | 5.84 | 2.18 | 1.86 | 3.00 |
| seed1 | 000166_10 | 3.84 | 5.05 | 6.02 | 6.57 | 6.98 | 7.33 | 7.56 | 6.63 | 5.66 | 4.26 | 1.75 | 2.63 |
| seed2 | 000187_10 | 4.78 | 6.23 | 7.30 | 7.46 | 7.10 | 6.77 | 6.63 | 7.33 | 7.41 | 2.53 | 1.91 | 3.26 |
| seed2 | 000160_10 | 4.87 | 6.25 | 7.17 | 7.49 | 7.09 | 6.51 | 6.39 | 6.99 | 7.03 | 2.37 | 1.98 | 3.36 |
| seed2 | 000191_10 | 4.76 | 6.20 | 7.32 | 7.61 | 7.45 | 6.76 | 6.57 | 7.27 | 7.13 | 2.09 | 1.92 | 3.29 |
| seed2 | 000166_10 | 4.38 | 5.88 | 6.93 | 7.09 | 6.88 | 6.61 | 6.56 | 7.14 | 6.98 | 4.70 | 1.78 | 2.90 |

Orbit range 5.36 – 6.39 candidates (median 5.69). Circular rank TV 22 – 32.

### Do `m = 3 … 10` add structure not visible in INDEX-002? **Yes.**

**MEASURED.** The orbit is **not monotone** over `m = 0 … 11`. It peaks at
`m = 2` (seed0), `m = 5–6` (seed1) or `m = 3–7` (seed2), then **collapses to its
minimum at `m = 10` in 12/12 units** and partially recovers at `m = 11`.

INDEX-002's four arms are orbit positions `11, 0, 1, 2` — a **local rising
segment of a wrapped, non-monotone orbit**. The previously unmeasured positions
show the wrap: the response does not keep rising, it turns over and crashes.
This is new information and is reported descriptively; it is **not** reinterpreted
as evidence for or against any hypothesis.

---

## 4. RANDOM ORBIT SUMMARIES

Example, `POS_6b_seed0 / 000187_10`, first six of sixteen seeds:

| weight-set | m=0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | range | TV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **trained** | 5.61 | 7.55 | 8.34 | 7.97 | 7.23 | 6.92 | 6.82 | 7.47 | 7.45 | 3.40 | 2.13 | 3.86 | **6.21** | 30 |
| R0 | 5.75 | 5.69 | 5.66 | 5.63 | 5.53 | 5.56 | 5.74 | 5.79 | 5.89 | 5.69 | 5.64 | 5.71 | 0.36 | 34 |
| R1 | 5.44 | 5.53 | 5.59 | 5.72 | 5.67 | 5.63 | 5.55 | 5.55 | 5.70 | 5.67 | 5.50 | 5.43 | 0.29 | 34 |
| R2 | 5.08 | 5.05 | 5.02 | 5.04 | 5.13 | 5.11 | 4.89 | 4.79 | 4.80 | 4.90 | 5.01 | 5.00 | 0.34 | 30 |
| R3 | 4.14 | 4.08 | 4.11 | 4.18 | 4.20 | 4.13 | 4.35 | 4.52 | 4.38 | 4.19 | 4.11 | 4.11 | 0.44 | 38 |
| R4 | 3.00 | 3.04 | 2.97 | 2.99 | 3.02 | 2.98 | 3.05 | 3.00 | 2.99 | 2.99 | 2.96 | 3.01 | 0.09 | 62 |
| R5 | 6.27 | 6.18 | 5.98 | 6.29 | 6.57 | 6.67 | 6.87 | 6.92 | 6.69 | 6.65 | 6.57 | 6.41 | 0.94 | 22 |

Pooled over all 192 (unit, seed) cells: random orbit range **0.092 – 1.418**
(median 0.587). **0 of 192** were flagged degenerate (`std == 0`), so the
preregistered zero-variance rule never fired on the positives.

**Four-arm ordering `s_11 < s_0 < s_1 < s_2`** (descriptive only, per §12):

```
trained : 12 / 12  units
random  : 31 / 192 (unit, seed) cells = 16.1 %
random per-seed, out of 12 units: [0,4,1,1,3,0,0,3,5,0,0,2,6,0,0,6]
```

---

## 5. TR AND RR DISTRIBUTIONS, AND PER-UNIT CLASSIFICATION

`ρ = (1/12) Σ_m z_m^{w1} z_m^{w2}`; `TR_u` = 16 trained-vs-random,
`RR_u` = 120 random-vs-random.

| checkpoint | scene | TR min | TR med | TR mean | TR max | RR min | RR med | RR mean | RR max | class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| seed0 | 000187_10 | −0.7748 | −0.0782 | −0.1263 | +0.6562 | −0.9668 | −0.0763 | −0.0304 | +0.9613 | INSIDE |
| seed0 | 000160_10 | −0.7763 | −0.0888 | −0.0574 | +0.8255 | −0.9428 | −0.1521 | −0.0547 | +0.9911 | INSIDE |
| seed0 | 000191_10 | −0.6626 | −0.1778 | −0.0340 | +0.8069 | −0.9811 | −0.1467 | −0.0568 | +0.9910 | INSIDE |
| seed0 | 000166_10 | −0.6726 | −0.1809 | −0.0846 | +0.7645 | −0.9697 | −0.1530 | −0.0394 | +0.9945 | INSIDE |
| seed1 | 000187_10 | −0.8374 | +0.0497 | −0.0388 | +0.7648 | −0.9480 | −0.0279 | −0.0295 | +0.9711 | INSIDE |
| seed1 | 000160_10 | −0.9048 | +0.2110 | +0.0910 | +0.9130 | −0.9783 | −0.0252 | −0.0321 | +0.9674 | INSIDE |
| seed1 | 000191_10 | −0.9270 | +0.0955 | +0.0146 | +0.8396 | −0.9860 | +0.0123 | −0.0312 | +0.9804 | INSIDE |
| seed1 | 000166_10 | −0.8904 | +0.1597 | +0.1009 | +0.8560 | −0.9684 | −0.0838 | −0.0111 | +0.9543 | INSIDE |
| seed2 | 000187_10 | −0.8558 | −0.2942 | −0.1820 | +0.7168 | −0.9612 | +0.0542 | −0.0070 | +0.9739 | INSIDE |
| seed2 | 000160_10 | −0.8341 | −0.2246 | −0.0840 | +0.6175 | −0.9746 | −0.1617 | −0.0445 | +0.9851 | INSIDE |
| seed2 | 000191_10 | −0.8971 | −0.2663 | −0.0880 | +0.7694 | −0.9795 | −0.0862 | −0.0364 | +0.9820 | INSIDE |
| seed2 | 000166_10 | −0.9022 | +0.0965 | −0.0326 | +0.7330 | −0.9722 | −0.0462 | −0.0512 | +0.9840 | INSIDE |

```
POOLED TR  n=  192  min=-0.9270  median=-0.0647  mean=-0.0434  max=+0.9130  sd=0.5437
POOLED RR  n= 1440  min=-0.9860  median=-0.0591  mean=-0.0354  max=+0.9945  sd=0.6423

INSIDE     12 / 12        SEPARATED  0 / 12        CASE-C  0 / 12
reverse separation: 0 / 12
```

Per-unit `Δ_u = mean(RR_u) − mean(TR_u)` ranges −0.1232 … +0.1750, sign mixed
across units — no consistent direction. Descriptive only.

---

## 6. DISCREPANCY BETWEEN THE DESIGN'S EXPECTATION AND THE MEASUREMENT

Reported, per `PREREGISTRATION.md` §16, rather than acted upon. **Nothing below
changed the protocol, the statistic, or the verdict.**

### 6.1 The statistic behaved with almost no power on this data (MEASURED)

The random-vs-random containment interval turned out to be nearly the whole
attainable range of a correlation:

| unit | RR interval | width | % of full [−1,+1] |
|---|---|---:|---:|
| seed0 / 000187_10 | [−0.9668, +0.9613] | 1.9281 | 96.4 % |
| seed0 / 000191_10 | [−0.9811, +0.9910] | 1.9721 | 98.6 % |
| seed1 / 000191_10 | [−0.9860, +0.9804] | 1.9664 | 98.3 % |
| … all 12 units | | mean **1.9471** | **97.4 %** |

`INSIDE` requires the 16 TR values to fall inside an interval covering 97.4 % of
everything a correlation can be. **The INSIDE criterion was therefore close to
unfalsifiable at every unit**, and `CASE A` was close to a foregone conclusion
regardless of the data.

The cause is measurable: the random orbits are near-flat (§4), so their z-scored
shapes are essentially arbitrary unit vectors whose pairwise correlations fill
`[−1, +1]`. Z-scoring rescales a 0.1-candidate wobble to unit variance exactly as
it rescales a 6-candidate swing.

### 6.2 The dimension the statistic discards separates completely (MEASURED)

**Orbit amplitude**, `max_m s_m − min_m s_m`:

| condition | n | min | median | max |
|---|---:|---:|---:|---:|
| trained | 12 | **5.364** | 5.688 | 6.386 |
| random | 192 | 0.092 | 0.587 | **1.418** |

```
max(random) = 1.418  <  min(trained) = 5.364        COMPLETE SEPARATION
overlap: 0 / 192        median ratio: 9.7x
```

Random aggregations barely respond to candidate re-indexing at all; the trained
ones swing across half the candidate range.

### 6.3 What this means, stated carefully

**The design's §5/§7 expectation was that approximate convolutional equivariance
would make random weights produce a *comparable* wrapped ramp. That expectation
is not supported by the measurement.** The synthetic check (§7 below) confirms
the *aggregation output* is approximately equivariant for arbitrary weights — but
after standardisation and soft-argmin, the random condition's readout response
collapses to ≈0.5 candidates against the trained condition's ≈5.7.

**The preregistered primary statistic cannot see this**, because it is
scale-free by construction. That choice was made *before* any result, precisely
to avoid the magnitude dimension INDEX-002 had shown to be non-discriminative
for a *different* contrast (ordered vs permuted). It turns out to discard exactly
the dimension where the trained/random difference lives in *this* contrast.

**This is a design error found by execution, not a result.** The correct response
is to report it and stop, which is what is done here. Re-scoring ARCH-001 on
amplitude would be precisely the post-hoc statistic selection the whole protocol
line exists to prevent — the observed separation cannot be promoted to evidence,
because the statistic that reveals it was chosen after seeing it. Any
amplitude-based claim requires a **fresh preregistration** on data whose
amplitude has not been inspected, and that preregistration must contend with the
fact that ARCH-001's amplitude numbers are now public.

---

## 7. ARCHITECTURAL SANITY CHECK — explanatory, pre-authorised, unchanged

`preflight_architecture_analysis.py`, carried over unchanged from
`design_20260911/` and re-run here; output in
`preflight_architecture_analysis.log`. It loads no checkpoint, no scene and no
repository model class, and was written **before** this experiment existed.
**Real experiment outputs were not used to alter it.**

```
H-1  exact positions where Agg(shift_m V)(k) == Agg(V)(k-m)
   m=1 circular  -> [6]        m=1 zero-fill -> [5, 6]
   m=2 circular  -> none       m=2 zero-fill -> [6]
   m=3 circular  -> none       m=3 zero-fill -> none

H-2  per-position relative error, circular shift, ARBITRARY untrained weights
   m=1  k00:0.484 k01:0.237 k02:0.074 ... k06:0.000 ... k10:0.070 k11:0.344
```

**Confirmed:** the architecture *can* generate approximate wrapped-ramp behaviour
with random weights — the aggregation output is approximately shift-equivariant
in the interior (relative error 0.00–0.07 at `k = 2…10`) for arbitrary weights,
failing only at the two padded edges.

**Also confirmed, and now contrasted with the real measurement:** that property
holds at the *aggregation output*. It does not, on this data, survive to the
readout with comparable amplitude (§6.2). The synthetic analysis remains exactly
as authorised; the contrast is recorded, not resolved.

The design audit's prediction that circular rank TV would fail to separate also
held: trained TV 22–32, random TV 22–62 (median 24) — complete overlap.

---

## 8. VERDICT — the preregistered rule, applied literally

```
negative_gate_pass              = True   (816 arms, all diffs bit-exactly 0.0)
cost_volume_identity_all_units  = True
INSIDE units                    = 12 / 12
SEPARATED units                 =  0 / 12
CASE-C units                    =  0 / 12

CASE A iff INSIDE 12/12  ->  TRUE

-> CASE-A-ARCHITECTURE-INDUCED
```

No intermediate verdict was created. The rule was not changed after seeing
results. No p-value was used and trained/random labels were never permuted.

---

## 9. CLAIM CEILING

**MEASURED.** The degenerate `shift="none"` readout is bit-exactly invariant
(0.0 on both volume and `disparity_initial`) under all 12 orbit positions for all
17 weight-sets on all 4 scenes — 816 arms — including with random aggregation
weights. The trained 12-position orbit is non-monotone, peaking at `m = 2…7` and
reaching its minimum at `m = 10` in 12/12 units. Trained orbit amplitude is
5.36–6.39 candidates; random 0.09–1.42. The four-arm ordering holds at 12/12
trained units and 31/192 random cells. `TR` and `RR` overlap at every unit, with
the `RR` interval spanning 97.4 % of `[−1,+1]`.

**DERIVED.** `INSIDE` at 12/12 units; therefore the preregistered verdict is
`CASE-A-ARCHITECTURE-INDUCED`.

**INFERRED.** The preregistered scale-free statistic had almost no power on this
data, and the dimension it discards — amplitude — separates the two conditions
completely. The verdict reflects what the frozen statistic measures, not
necessarily what is true of the model.

**UNKNOWN.** Whether the candidate-axis response is weight-dependent. **This
experiment did not settle it**, and its `CASE A` must not be read as evidence
that it is not.

Per the frozen §14 interpretation, the only permitted conclusion is:

> **The observed orbit shape is adequately explained by the tested architecture /
> random-weight variability.**

and therefore, explicitly:

> **Candidate-coordinate sensitivity is NOT claimed.**

Not established and not claimed: geometric correspondence, correct matching,
disparity search, physical disparity correctness, stereo correctness. Candidate
indices were never converted to physical disparity: `m` is an exponent of the
frozen generator and the primary statistic is invariant to any affine rescaling
of `s`. No accuracy metric was computed — no EPE, no D1, no RMSE.

---

## 10. PROVENANCE

`HEAD` `58e8a19`; `phase-1-frozen^{commit}` `b4207e5`;
`git diff phase-1-frozen -- src scripts` **empty**; working tree `M .gitignore`,
`?? phase2/`. Phase 2 remains untracked — **no cryptographic versioning is
claimed for this record.** All output is confined to this directory. No
historical record was modified; INDEX-001, INDEX-002, the INDEX-003 design audit
and Stage A were not re-run and not re-scored; the design in `design_20260911/`
was **not** rewritten after execution. Checkpoints were opened read-only and
every sha256 was verified at load with a hard stop on mismatch.

No training occurred, no seed was added or removed, no threshold was tuned post
hoc, no statistic was changed after inspecting outputs, and the four-arm ordering
was treated as descriptive throughout.

See `RELATED_RUNS.md` for the full lineage.

---

## 11. FILES

`PREREGISTRATION.md`, `RESULTS.md` (this file), `results.json`,
`ENVIRONMENT.txt`, `RELATED_RUNS.md`, `run.log`,
`random_initialization_metadata.json`, `generator.json`, `frozen.sha256`,
`orbit_statistics.json`, `freeze_random_init.py`, `run_arch_control.py`,
`preflight_architecture_analysis.py`, `preflight_architecture_analysis.log`.
