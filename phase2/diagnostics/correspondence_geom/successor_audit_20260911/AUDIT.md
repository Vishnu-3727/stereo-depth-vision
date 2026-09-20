# POST-GEOM-001 SUCCESSOR DESIGN AUDIT

Record `phase2/diagnostics/correspondence_geom/successor_audit_20260911/`.
Date 2026-09-11.

**Audit only. Nothing executed.** No training, no inference, no model
instantiated, no new experimental output. No `RESULTS.md` created. Phase 1
untouched. **GEOM-001 not modified, not reinterpreted; its verdict stands at
`GEOMETRIC-CORRESPONDENCE-NOT-DEMONSTRATED`.**

All quantitative work below is **ground-truth-only**: KITTI GT disparity maps,
the frozen `mask_spec.json`, and integer geometry. **No GEOM-001 response value
was used in any derivation.**

Tags: **MEASURED** (from source, GT, or a frozen record), **DERIVED**
(arithmetic on measured values), **INFERRED** (interpretation), **UNKNOWN**.

---

## 0. DISCLOSURE — the auditor has seen the GEOM-001 response curve

I executed GEOM-001 and have read its response values. I cannot un-see them.

Every derivation in §1–§4 draws **only** on GT and on `mask_spec.json`, a frozen
pre-execution artifact. But no procedure can make me *provably* independent of
what I already know. This is recorded here because it bounds what any successor
I design can establish (§6, §7), and because concealing it would repeat the
failure mode this project has been correcting for since INDEX-001.

**In particular:** `Odd_h(16)`, the −48…+16 monotone stretch, and every slope
reported in GEOM-001 §7 are **excluded as evidence and as design inputs**
throughout this audit.

---

## 1. THE REPRESENTABILITY CONSTRAINT — derived from geometry alone

The geometric hypothesis is `d̂ = d − Δ/16` candidates. For a retained pixel to
be able to exhibit it, the shifted disparity must lie on the candidate axis:

```
0 ≤ d − Δ/16 ≤ 11
⇒ symmetric ±Δ requires    Δ/16 ≤ d ≤ 11 − Δ/16
```

**DERIVED.** Required GT band per offset:

| \|Δ\| px | candidates | required GT/16 band |
|---:|---:|---|
| 16 | 1 | `[1, 10]` |
| 32 | 2 | `[2, 9]` |
| 48 | 3 | `[3, 8]` |
| 64 | 4 | `[4, 7]` |

This is the check GEOM-001's §3.1 should have performed against its own frozen
mask and did not. **It is performed here from GT only, before any response is
consulted.**

---

## 2. WHAT THE FROZEN GEOM-001 MASK ACTUALLY CONTAINS

**MEASURED** — GT under the frozen mask (`GT > 0`, 112 px border), the 4 focus
scenes:

| scene | n | p01 | p25 | p50 | p75 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 000187_10 | 34 245 | 0.34 | 0.83 | 1.05 | 1.41 | 2.08 | 2.16 | 2.60 |
| 000160_10 | 46 199 | 0.35 | 0.82 | 1.16 | 1.60 | 1.92 | 1.98 | 3.17 |
| 000191_10 | 33 250 | 0.39 | 0.90 | 1.11 | 1.38 | 2.00 | 2.78 | 3.39 |
| 000166_10 | 50 747 | 0.37 | 1.10 | 1.55 | 2.00 | 2.31 | 2.38 | 2.84 |
| **pooled** | **164 441** | | | **1.232** | | | | **3.392** |

**DERIVED** — fraction of the frozen mask that remains representable:

```
|Δ| = 16 px  ->  66.8 %   (109 842 px)
|Δ| = 32 px  ->  10.1 %   ( 16 639 px)
|Δ| = 48 px  ->   0.2 %   (    271 px)
```

**This confirms the GEOM-001 defect quantitatively and independently of any
response value: at `Δ = ±48`, 99.8 % of the evaluation mask could not exhibit the
hypothesis.** The `+32` and `+48` conditions were not a test that failed; they
were not a test.

---

## 3. IS THE BORDER THE BINDING CONSTRAINT? NO — IT IS THE DATA

The 112 px border removes rows ≥ 256, and KITTI's large disparities live in the
bottom of the frame. **MEASURED**, focus scenes, border 0:

| rows | n | median GT/16 | p95 | max |
|---|---:|---:|---:|---:|
| 0–92 | 0 | — | — | — (sky, no GT) |
| 92–184 | 51 114 | 1.01 | 2.49 | 2.94 |
| 184–276 | 174 985 | 1.52 | 2.44 | 4.19 |
| 276–368 | 142 035 | **2.93** | 3.84 | 5.24 |

Relaxing the border recovers some of it — but not enough. **MEASURED**, pooled
over the focus scenes:

| border px | n | p50 | max | in `[1,10]` | in `[2,9]` | in `[3,8]` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 368 134 | 1.95 | 5.24 | 84.4 % | 47.9 % | 18.1 % |
| 16 | 342 049 | 1.88 | 5.24 | 83.3 % | 44.5 % | 13.1 % |
| 32 | 313 866 | 1.80 | 5.24 | 81.9 % | 40.1 % | 7.1 % |
| 64 | 254 244 | 1.57 | 4.28 | 78.0 % | 27.9 % | 1.4 % |
| 112 | 164 441 | 1.23 | 3.39 | 66.8 % | 10.1 % | 0.2 % |

**Even at zero border the maximum GT/16 is 5.24 candidates.** The model's
candidate axis spans 0…11; the *data* occupy roughly 0…5.

**Across all 40 `hailo_val` scenes** (border 32) — **MEASURED**:

```
[2,9] retention:  min 33.7 %   median 48.6 %   max 79.5 %     17 / 40 scenes ≥ 50 %
[3,8] retention:  best 38.8 %                                  0 / 40 scenes ≥ 50 %
max GT/16:        min 3.29     median 4.24     max 8.06
```

**DERIVED, decisive:** `|Δ| = 48` is infeasible on **every scene in the split**.
`|Δ| = 32` is feasible on 17 of 40 scenes. `|Δ| = 16` is feasible broadly.

---

## 4. MAXIMUM SYMMETRIC RANGE, AND WHETHER A SYMMETRIC TEST REMAINS USEFUL

**Objective 4 — DERIVED.** On the four frozen GEOM-001 scenes, the maximum
symmetric offset retaining ≥ 50 % of a reasonable mask is **±16 px (1 candidate)**;
±32 px is reachable only at 40–48 % retention and only with a `[2,9]` band mask.
On scene-selected subsets of `hailo_val`, ±32 px is reachable at ≥ 50 % on 17 of
40 scenes. **±48 px is unreachable anywhere.**

**Objective 5 — is a symmetric test still useful? Only marginally.**

GEOM-001's statistical structure needs signed offsets: `Odd(Δ)` requires ±Δ
pairs, the slope needs ≥ 2 points, and the monotonicity condition `C2` needs ≥ 3
to be a real constraint (probability of accidental ordering `1/3! = 1/6` at 3
points, `1/2` at 2 points).

The representable range supports **at most two** usable offsets (±16, ±32).
Therefore:

- `Odd(Δ)` has 2 points ⇒ the slope is a 2-point fit;
- `C2` monotonicity collapses to the single inequality `Odd(16) > Odd(32)`,
  which is a coin flip under any null;
- the design audit that produced GEOM-001 explicitly required ≥ 3 signed points
  and would have rejected a 2-point design.

**INFERRED:** the symmetric-translation statistic class, as preregistered in
GEOM-001, **cannot be rebuilt at adequate strength on this data**. A successor in
that class would be weaker than the experiment it replaces.

---

## 5. THE PROPOSED STATISTIC ONCE THE RANGE IS CONSTRAINED (objective 8)

| element | status under the constrained range |
|---|---|
| `Odd(Δ) = [m(+Δ) − m(−Δ)]/2` | still well defined, still the right separator of signed geometry from even magnitude artifact |
| `S_h = Σ Odd(Δ)Δ / ΣΔ²` | degenerates to a 2-point fit; the anchor `−1/16` remains exact |
| `C2` monotonicity | **loses its power** — one inequality, 1/2 under the null |
| `C3` axis specificity | unaffected and still valid |
| `C4` vs search-free baseline | unaffected and still valid |
| claim ceiling | unchanged |

**INFERRED:** the statistic is not *wrong* under the constraint; it is
*underpowered*. The failure is in the achievable number of levels, not the
estimator.

---

## 6. CAN THE SAME 12 MODEL × SCENE UNITS BE REUSED? (objective 6)

**Largely no, for the translation class.**

**MEASURED.** GEOM-001's `RESULTS.md` §4 publishes `m_h(Δ)` for
`Δ ∈ {−48,−32,−16,0,+16,+32,+48}` and §5 publishes `m_v(Δ)` for the same set, for
**all 16 cells** (12 positive + 4 control). A successor on the same checkpoints
and the same four scenes, using horizontal translation at `Δ ∈ {±16, ±32}`, would
recompute a statistic from **already-published inputs**. Changing the mask from
`GT > 0` to a band changes the numbers but not the qualitative answer, which is
visible in the published curve.

**This is the INDEX-C failure mode exactly:** a new statistic computed on data
whose response is public.

**What is reusable.**

| axis | reusable? | why |
|---|---|---|
| the 3 trained checkpoints | **unavoidably yes** | training is forbidden; these are the entire available population. Model-axis reuse is irreducible. |
| the 4 focus scenes, translation class | **no** | their translation response is public at every offset a successor could use |
| the 36 unused `hailo_val` scenes | **yes** | **MEASURED: 40 scenes exist; 4 have been used. 36 have never appeared in any correspondence experiment.** Fresh data on fixed weights. |
| a different intervention class | **yes** | never measured on these checkpoints |

**The single most valuable free lever available is the 36 held-out scenes.**

---

## 7. WHAT EVIDENTIAL STATUS COULD A SUCCESSOR HAVE? (objective 7)

| scenario | status |
|---|---|
| same scenes, same intervention, corrected Δ | **scientifically weak** — inputs already public |
| fresh scenes, same intervention | **replication / consistency evidence** — genuinely unmeasured data, but the mechanism's behaviour is known from the published curve, and the statistic is underpowered (§4) |
| fresh scenes, **different intervention never measured on these weights** | **strongest available: strong consistency evidence with real predictive content** — but still not fully confirmatory |
| any design whatever | **never fully confirmatory**, because the 3 checkpoints are reused and the designer has seen the prior response (§0) |

**INFERRED, and stated plainly: no experiment available with the current models
can be *confirmatory* in the strict sense this project has been using.** The
weights cannot be replaced without training. The honest ceiling for any successor
is *strong consistency evidence with predictive content on the data and
intervention axes*.

That is not nothing — it is exactly what is missing right now, since GEOM-001
produced **no** valid measurement of the geometric hypothesis — but it must be
labelled correctly in advance, not after the result is seen.

---

## 8. ADDITIONAL FAILURE MODES A SUCCESSOR MUST HANDLE (objective 9)

| # | failure mode | status | mitigation available |
|---|---|---|---|
| 1 | **candidate-axis clipping** | the defect that broke GEOM-001; **MEASURED** at 99.8 % of mask at Δ=48 | either a GT band derived in advance (§1), or an intervention that *sets* the disparity rather than perturbing it (§Options B) |
| 2 | **crop-induced changes** | crop size shrinks as max\|Δ\| grows; **DERIVED** crop is valid and stride-aligned for every candidate margin (1136×272 at M=48 … 1040×176 at M=96) | fix one crop size for every condition, as GEOM-001 did correctly |
| 3 | **translation direction / sign** | positive Δ reduces disparity and is the clipping-prone direction; the two directions are **not** equally valid | either band-mask both directions, or use a one-sided sweep and drop odd/even |
| 4 | **disparity quantisation** | candidate spacing is 16 px; only multiples of 16 give exact feature translation (**MEASURED**: fully convolutional, total stride 16) | keep every offset a multiple of 16 |
| 5 | **boundary effects** | feature receptive field ≈477 px exceeds any affordable border; crop edges always contaminate | not removable; **measure** it with the search-free control |
| 6 | **soft-argmin / readout artifacts** | a linear index grid saturates at 0 and 11; near the rails the response is compressive, not linear | keep the predicted response away from both rails by construction |
| 7 | **vertical-control contamination** | **MEASURED in GEOM-001 §5 that vertical displacement produces a large response**; a vertical arm is a *magnitude* control, never a second geometric test | keep it as a matched null only; never assign it a predicted slope |
| 8 | **scene / model dependence** | 12 units share 3 weight sets and 4 scenes; not independent replicates | one global statistic or unit-level all-or-nothing; never a partial count |
| 9 | **nonlinearity near the candidate boundaries** | closely related to 1 and 6; the response is not linear where the true answer is unrepresentable | design so the predicted value stays in roughly `[1, 7]` of the 0…11 axis |
| 10 | **OOD inputs** | a synthetic constant-disparity pair is not a natural stereo image | makes a *negative* result ambiguous; a positive result remains informative |
| 11 | **designer knowledge** | §0 | disclose; use point predictions derived from geometry; use fresh scenes |

---

## 9. SUMMARY OF FINDINGS

**MEASURED**
- Frozen GEOM-001 mask: pooled GT/16 median 1.232, max 3.392, 164 441 px.
- Retention under the representability band: 66.8 % at ±16, 10.1 % at ±32,
  **0.2 % at ±48**.
- Even at zero border, pooled GT/16 max is 5.24; the data occupy ~0…5 of a 0…11 axis.
- Across all 40 `hailo_val` scenes: `[3,8]` reaches ≥ 50 % on **0** scenes;
  `[2,9]` on 17.
- 40 scenes exist in the split; **4 used, 36 never used** in any correspondence
  experiment.
- GEOM-001 published `m_h(Δ)` and `m_v(Δ)` at all seven offsets for all 16 cells.

**DERIVED**
- Maximum symmetric offset with ≥ 50 % retention: **±16 px** on the frozen scenes;
  ±32 px only with a band mask and scene selection; ±48 px nowhere.
- A symmetric design therefore supports **at most 2 signed offsets**, so `C2`
  monotonicity degenerates to a coin flip and the GEOM-001 statistic class cannot
  be rebuilt at its intended strength.

**INFERRED**
- The symmetric-translation class is exhausted on this dataset.
- A successor must either constrain the mask by disparity in advance, or adopt an
  intervention in which the disparity is **set** rather than perturbed.
- No successor can be fully confirmatory; the achievable ceiling is strong
  consistency evidence with predictive content on the data and intervention axes.

**UNKNOWN**
- Whether these checkpoints exhibit geometric correspondence. **GEOM-001 did not
  measure it.** The question is open, not answered negatively.
- How the models behave on synthetic constant-disparity pairs — never measured on
  any checkpoint in this campaign.
- How they behave on the 36 held-out scenes.

See `SUCCESSOR_OPTIONS.md` for the candidate designs and `RECOMMENDATION.md` for
the decision.
