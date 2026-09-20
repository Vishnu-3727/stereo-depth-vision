# EXP-CORRESPONDENCE-INDEX-002 — PRE-REGISTRATION

**Empirical Permutation Null for Candidate-Axis Sensitivity**

Record: `phase2/diagnostics/correspondence_index/20260911T015124Z/`.
Frozen 2026-09-11, **before any arm of this experiment was executed** and before
any response statistic of this experiment existed.

Inference only. No training, no optimizer, no weight change, no architecture
change, no preprocessing change, no image change, no feature change. Phase 1 is
untouched. EXP-CORRESPONDENCE-INDEX-001 is untouched and is not re-run. No
historical record is modified.

---

## 1. QUESTION

Is the reproducible ordered candidate-axis response

```
d(−1) < d(identity) < d(+1) < d(+2)
```

**distinguishable from the response produced by arbitrary candidate-axis
permutations?**

This experiment targets **candidate-coordinate sensitivity (level C)** and
nothing above it. It is not intended to establish geometric correspondence,
correct disparity search, correct matching, final disparity correctness,
physical translation slope, or stereo depth correctness.

The maximum possible claim is stated in §14 and is bounded there.

---

## 2. WHY THIS EXPERIMENT EXISTS

EXP-CORRESPONDENCE-INDEX-001 produced two executions
(`20260910T164346Z`, `20260910T164326Z`) whose ordered arms agree to four
decimal places but whose random-permutation controls disagreed in sign. The
provenance audit (`audit_20260911T011113Z/`) found both **EXPLORATORY-ONLY** and
the overall state **NO-CONFIRMATORY-EVIDENCE**, with the decisive defect being
that each run compared the ordered response to a **single uncalibrated random
permutation with no null distribution**.

This experiment replaces that single draw with an empirical null of 512
independent draws and a decision rule frozen here, in full, before execution.

See `RELATED_RUNS.md`.

---

## 3. PRE-FLIGHT / PROVENANCE (§A) — recorded before execution

Verified by source inspection before this document was written. Any item
failing at execution time is a **HARD STOP**.

| # | Item | Value / requirement |
|---|---|---|
| A1 | Checkpoint hashes | recorded in §7; verified by sha256 at load, recorded in `results.json` |
| A2 | Model / configuration | `StereoNet(StereoNetConfig(cost_volume_shift=<per checkpoint>))` + `phase2.models.scaled_regression.apply_to`; asserted `type(model.regression).__name__ == "StandardisedDisparityRegression"` and `model.cost_volume.shift == expected` |
| A3 | Dataset / scenes | KITTI 2015, split `hailo_val`, `FOCUS_SCENES = [27, 0, 31, 6]` → `000187_10.png`, `000160_10.png`, `000191_10.png`, `000166_10.png` |
| A4 | GT mask | §5 below |
| A5 | Tensor shape | cost volume `(1, 32, 12, 23, 77)` = `(B, C, D, H, W)`, candidate axis `dim=2` |
| A6 | Candidate count | `N = 12` (`StereoNetConfig.num_disparities`) |
| A7 | Feature / cost-volume construction | §4 below |
| A8 | Intervention point | immediately after `model.cost_volume(lf, rf)` and before `model.aggregation` — asserted in code |
| A9 | Identity via the same op | every arm including identity is built with `volume.index_select(2, idx)`; `max_abs_diff(identity_indexed, original)` recorded and required `== 0` |
| A10 | Deterministic execution | `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.no_grad()`, `model.eval()`, fp32 |
| A11 | No training | no optimizer is constructed, no `.backward()`, no `.step()`, checkpoints opened read-only; asserted by the absence of any training import in the harness |

**A9 rationale (carried unchanged from INDEX-001 §4).** `build_cost_volume`
returns a permuted *view* with non-contiguous strides; `index_select`
materialises a contiguous tensor. Mixing the two across arms could select a
different cuDNN algorithm and change the last bits, corrupting the negative-
control gate for a purely numerical reason. Routing every arm through the
identical op holds layout constant.

---

## 4. REFERENCE FRAME — resolved from source, frozen (§B)

Resolved by reading `src/models/stereonet/cost_volume.py`,
`src/models/stereonet/stereonet.py`, `src/models/stereonet/regression.py`,
`phase2/models/scaled_regression.py` and `src/datasets/kitti2015.py` before this
document was written. Nothing below is a change; it is a statement of what the
frozen code already does.

**Candidate index `k`.** `k ∈ {0, …, 11}`, the `dim=2` axis of
`(B, C, D, H, W)`.

**Shift convention.** For the positive checkpoints (`shift="left"`):

```python
def shift_left(x, k):  return F.pad(x, (0, k))[..., k:]      # shifted[..., u] = x[..., u+k]
levels[k] = shift_left(left, k) - right                      # level k at column u
```

so **candidate slice `k` at output column `u` is `left_feat[u+k] − right_feat[u]`.**

For the negative checkpoint (`shift="none"`), `reference_shift` pads right and
slices `[0:W]`, which returns the input unchanged, so **every slice is
`left_feat[u] − right_feat[u]`** and the volume is degenerate by construction.

**Relationship between `k` and disparity units.** `feature_stride = 2**4 = 16`.
A left pixel at column `u+k` matching a right pixel at column `u` has disparity
`k` in feature-grid units, i.e. `16k` full-resolution pixels
(`max_disparity_px = (12−1)·16 = 176`).

**The referencing discrepancy — recorded, NOT corrected.** The magnitude of
candidate `k` is a disparity in feature-grid units, but the volume stores it at
the **right** image's column `u`, whereas GT and the mask are indexed by the
**left** image's column. This is the discrepancy recorded in INDEX-001 §12 and
in the audit. It is **preserved as found**.

**Why it does not enter the primary statistic.** Every arm in this experiment is
evaluated on the **identical pixel set**, and the primary statistic (§6) is
built **only from differences of `disparity_initial` between arms**. The
candidate index is never converted to a physical disparity anywhere in the
primary statistic. The referencing discrepancy therefore determines *which*
pixels are retained, identically for every arm, and cancels out of every
between-arm difference. Converting candidate indices to physical disparity
belongs to the next (geometric) experiment and is **not** done here.

**GT disparity convention.** `read_disparity_png(..., scale=256.0)`, the KITTI
convention — GT is in **full-resolution pixels**; `0` means no ground truth and
is never treated as a disparity of zero.

**`disparity_initial` units.** `soft_argmin` returns the expected value of the
candidate **index** under `softmax(−cost)`, i.e. `disparity_initial` is in
**units of disparity candidates** (0…11), not pixels. It is therefore directly
comparable in scale to `GT / 16`.

---

## 5. MASK — previous mask retained unchanged (§B)

The INDEX-001 mask is **retained exactly**, verified internally valid by the
above: `GT/16` puts ground truth into candidate units, the same units as
`disparity_initial`.

```
retain pixel  iff  GT > 0  and  GT/16.0 >= 2.0  and  GT/16.0 <= 8.0
```

equivalently `32 px ≤ GT ≤ 128 px`. Implemented as

```python
mask = (gt > 0) & (gt / 16.0 >= 2.0) & (gt / 16.0 <= 8.0)
```

with `gt = scene.gt_disparity.astype(np.float64)`.

Derived from **ground truth only**, so the identical pixel set applies to every
checkpoint and every arm. Not model-dependent. No border mask — no image is ever
translated. **The mask is not modified after results are seen.**

Expected retained counts (from GT alone, read during the audit, model-independent
and therefore not an experiment output): 33 026 / 42 676 / 37 892 / 62 908 for
`000187_10`, `000160_10`, `000191_10`, `000166_10`; pooled 176 502. These will be
re-recorded at execution and must match.

**Known property of this mask, recorded here, NOT acted upon.** The audit
measured the occupancy of the retained band and found ≈99.4 % of retained pixels
at `GT/16 ∈ [2,4]`, with bins 6–8 empty. The band is nominally `[2,8]` and
effectively `[2,4]`. The mask is **not** changed for this experiment: changing it
now would confound the comparison with INDEX-001, and the purpose here is to make
the reference frame explicit and frozen, not to optimise it. The occupancy is
reported in `RESULTS.md` as null-quality metadata (§K) and is **not** an input to
any decision rule.

---

## 6. PRIMARY STATISTIC — frozen before execution (§F)

### 6.1 The per-arm aggregate `d(x)`

**Exactly the aggregation used by INDEX-001**, reproduced from its harnesses:

```python
arr  = disparity_initial[0, 0].detach().float().cpu().numpy().astype(np.float64)
d_x  = float(np.median(arr[mask]))
```

i.e. `d(x)` is the **median of `disparity_initial` over the interior-masked
pixels**, in candidate units, for arm `x`. `np.median` (not `torch.median`): the
two differ on even-sized inputs and INDEX-001 used `np.median`.

### 6.2 The evaluation unit

One **(checkpoint, scene)** pair. There are 3 positive checkpoints × 4 scenes =
**12 positive units**, plus 4 negative-control units.

### 6.3 Primary (directional) statistic

```
S_order = [ d(m_plus_2) − d(identity) ] + [ d(m_plus_1) − d(m_minus_1) ]
```

### 6.4 Secondary (magnitude) statistic — descriptive only

```
S_abs = |d(m_plus_2) − d(identity)| + |d(m_plus_1) − d(identity)| + |d(m_minus_1) − d(identity)|
```

`S_abs` is reported for description. **The primary test is the directional
statistic `S_order`.** `S_abs` may not change any verdict.

### 6.5 The null analogue — same functional form

A single permutation cannot populate a four-arm contrast. The null therefore
substitutes a **random triple** for the ordered triple, preserving the functional
form exactly. For null draw `i` with frozen triple `(a, b, c)`:

```
S_random[i] = [ d(π_a) − d(identity) ] + [ d(π_b) − d(π_c) ]
S_abs_random[i] = |d(π_a) − d(identity)| + |d(π_b) − d(identity)| + |d(π_c) − d(identity)|
```

`π_a` stands in for `m_plus_2`, `π_b` for `m_plus_1`, `π_c` for `m_minus_1`.
`d(identity)` is the same value used by `S_order`. The triples are **disjoint**
(triple `i` uses permutations `3i, 3i+1, 3i+2`), so the 512 null values are
independent draws.

---

## 7. CHECKPOINTS — read-only, unchanged (§E)

| key | shift | seed | path | sha256 |
|---|---|---|---|---|
| `NEG_shift_none` | `none` | — | `phase2/factorial/shift_none_standardized/20260909T071500Z/checkpoints/EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001-ARM-A_checkpoint.pth` | `d40d805bbe4cca310683f8617c7518d660e30055c43431081bcbb6a7c388f9b1` |
| `POS_6b_seed0` | `left` | 0 | `phase2/diagnostics/determinism/20260909T041500Z_baseline/checkpoints/STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA_checkpoint.pth` | `581d62e699b159945580f9627b451d953ce07b5dc763d608a9f4a8b5ecba692a` |
| `POS_6b_seed1` | `left` | 1 | `phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED1_checkpoint.pth` | `58117f2613bba4b2c817328c7d6757f21b340ec5e64a828e633193ae091d4adf` |
| `POS_6b_seed2` | `left` | 2 | `phase2/factorial/block_count_full/20260910T005550Z/checkpoints/EXP-BLOCKCOUNT-FULL-001-ARM-A-SEED2_checkpoint.pth` | `245a3f5bcb801c7d9c3550b02cb8775cb86486beae606f04097900340754fa69` |

These are the 6-block H2 control at seeds 0, 1, 2 — the same frozen checkpoints
INDEX-001 used. **No extra seeds. No retraining. If a loaded hash differs from
the table above, HARD STOP.**

---

## 8. PERMUTATIONS — frozen before execution (§C, §D)

`N = 12`. Convention, identical to INDEX-001:

```
V'(k) = V(π(k)),   π_m(k) = (k − m) mod 12
```

so content originally at candidate `j` moves to `j + m mod 12`; **positive `m`
moves content toward larger candidate indices.**

### 8.1 Ordered arms

| arm | π as a literal 12-element index list |
| --- | --- |
| identity (m=0) | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` |
| m = +1 | `[11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` |
| m = +2 | `[10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` |
| m = −1 | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]` |

### 8.2 Random null — seed and size frozen here

```
FROZEN_SEED = 20260911
generator   = numpy.random.default_rng(20260911); rng.permutation(12) x 1536
numpy       = 2.5.1
N_RANDOM    = 1536 permutations
M           = 512 null triples (disjoint: triple i = permutations 3i, 3i+1, 3i+2)
duplicates  = 0
```

The seed `20260911` is today's date, the same convention INDEX-001 used
(`20260910`). It was chosen with **no reference** to any response, any previous
result, any occupancy statistic, any spatial statistic or any preliminary output.

**The full literal permutation list is frozen in `permutations.json` in this
directory, written before this preregistration was finalised and before any arm
ran.** Its content hash is recorded here so the set cannot be silently changed:

```
permutations.json  sha256 = f064d09f6b7d506caf5edc5644c2c104dc74d6fd00abe33f0d2eb9ffb6e243d7
```

Spot check — the first four and last four draws, verbatim:

```
[ 0] [3, 1, 11, 9, 6, 7, 8, 5, 10, 0, 4, 2]
[ 1] [3, 1, 7, 2, 6, 8, 11, 4, 9, 0, 5, 10]
[ 2] [0, 9, 11, 3, 1, 7, 10, 2, 4, 6, 8, 5]
[ 3] [2, 8, 6, 11, 10, 1, 9, 4, 3, 7, 0, 5]
...
[1532] [10, 2, 4, 8, 9, 7, 1, 3, 6, 5, 11, 0]
[1533] [0, 11, 2, 5, 7, 8, 10, 1, 6, 4, 9, 3]
[1534] [10, 7, 1, 5, 3, 4, 8, 9, 2, 6, 11, 0]
[1535] [6, 3, 10, 8, 2, 9, 7, 5, 0, 1, 11, 4]
```

`permutations.json` also stores, for every random permutation: the permutation,
its inverse, its fixed-point count and list, its mean displacement over all 12
candidates, its mean displacement over the retained band `d ∈ [2,8]`, and its
inverse restricted to that band. **These are descriptive metadata only.**

### 8.3 NO SCREENING — binding (§D)

**No permutation may be discarded, reweighted, reordered or replaced**, for any
reason, including appearing biased, directional, unusually strong, unusually
weak, spatially structured, or close to an ordered permutation. All 1536 draws
and all 512 triples belong to the null. If the null is heterogeneous, that is a
property of the null and is **reported**, not repaired.

`freeze_permutations.py` contains no rejection branch. `permutations.json`
records `"screening_applied": false, "rejection_applied": false`.

---

## 9. WHY M = 512 AND NOT 64 — recorded conflict, resolved before execution

The task specification asks for "at least 32" random permutations, "preferred 64
if runtime remains negligible" (§C), **and** a decision rule of "outside the
central 99 % empirical null … equivalently empirical two-sided p ≤ approximately
0.01" (§G).

**These two are not simultaneously satisfiable at M = 64.** Under the standard
finite-sample permutation-test rank (§10.1), the smallest attainable two-sided
p-value is `2/(M+1)`:

```
M =   64  ->  min two-sided p = 0.03077     > 0.01   NOT ATTAINABLE
M =  128  ->  min two-sided p = 0.01550     > 0.01   NOT ATTAINABLE
M =  256  ->  min two-sided p = 0.00778     <= 0.01  attainable
M =  512  ->  min two-sided p = 0.00390     <= 0.01  attainable, ~5 draws per 1% tail
```

The decision rule is the binding scientific criterion and §C's count is stated as
a preference conditioned on runtime. **M = 512 is therefore frozen**, which
satisfies §C's hard floor ("at least 32") and makes §G's frozen rule attainable
with a usable estimate of the 1st and 99th percentiles.

A pre-flight timing probe on the negative checkpoint measured **0.00698 s per
aggregation+readout arm** (no statistic computed, no positive checkpoint
touched). Budget: 16 units × (4 ordered + 1536 random) = **24 640 arms ≈ 172 s**
plus 16 cost-volume builds. This is chosen **before any response exists** and on
timing and rank arithmetic alone — never on any observed response.

---

## 10. DECISION RULE — frozen before execution (§G, §H, §I)

### 10.1 Empirical-null test (per unit), exact finite-sample rank

No parametric assumption. No fitted distribution. Computed in code exactly as:

```python
ge   = int(np.sum(S_null >= S_ord))
le   = int(np.sum(S_null <= S_ord))
p_ge = (1.0 + ge) / (M + 1.0)
p_le = (1.0 + le) / (M + 1.0)
p_two = min(1.0, 2.0 * min(p_ge, p_le))

p01, p99 = np.percentile(S_null, [1.0, 99.0], method="linear")
outside_central_99 = bool(S_ord < p01 or S_ord > p99)

null_pass = bool(p_two <= 0.01 and outside_central_99)
```

`M = 512`, so `min p_two = 0.003899`.

### 10.2 Ordering test (per unit)

```python
order_pass = bool(d_plus2 > d_plus1 > d_identity > d_minus1)
```

Reported as count / total / fraction. **The 12/12 ordering observed in INDEX-001
is a reproducibility observation, not proof by itself.**

### 10.3 Preregistered evaluation level

**All 12 positive (checkpoint, scene) units must pass.** Both tests are required
at every unit. No pooling, no averaging across units, no "majority of units", no
per-checkpoint aggregation. This is the strictest reading and is fixed here to
foreclose any post-hoc choice of aggregation level.

```python
global_null_pass  = all(u["null_pass"]  for u in positive_units)   # 12/12
global_order_pass = all(u["order_pass"] for u in positive_units)   # 12/12
```

### 10.4 Global verdict

```
CANDIDATE-COORDINATE-SENSITIVITY-ESTABLISHED
    iff  negative_gate_pass and global_null_pass and global_order_pass

otherwise
CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED
```

**No intermediate "almost passed" verdict exists and none may be created.**
There is exactly one additional terminal outcome, defined in §12
(`NULL-DESIGN-INCONCLUSIVE`), whose trigger is frozen there.

**No threshold, percentile, statistic, aggregation, evaluation level or
permutation may be changed after any result is seen.**

---

## 11. NEGATIVE CONTROL — first and gating (§J)

Checkpoint: the trained `shift="none" + standardised` model (§7, `NEG_shift_none`).

Its volume is degenerate — `V(k) = left_feat − right_feat` for every `k`, because
`reference_shift` is a no-op. Therefore for **any** π, `V[:, :, π] = V`
element-wise. This is an **algebraic identity, not a statistical expectation.**

Run **all 1540 arms** (4 ordered + 1536 random) on all 4 scenes. Required, for
every arm against identity:

```
max_abs_diff(volume_identity, volume_permuted)                    == 0
max_abs_diff(disparity_initial_identity, disparity_initial_perm)  == 0   (full frame)
```

and additionally `max_abs_diff(identity_indexed_volume, original_volume) == 0`.

**Any non-zero difference ⇒ HARD STOP IMMEDIATELY.** Do not run the positive
checkpoints. Report as an implementation / determinism / layout failure and **do
not interpret it scientifically.**

---

## 12. NULL-QUALITY REPORTING AND `NULL-DESIGN-INCONCLUSIVE` (§K)

`RESULTS.md` and `results.json` must report, without removing any draw:

- retained candidate occupancy of the mask (descriptive; from GT only);
- inverse-permutation mapping for every random permutation;
- displacement distribution across the null;
- the null response distribution per unit: **n, mean, median, SD, p01, p99**;
- the ordered response and its **empirical rank** within the null;
- the full per-unit table (§13).

**No draw may be removed for being inconvenient.** If the null is found
unsuitable for the intended test, it must **not** be repaired after execution.
The experiment is then classified

```
NULL-DESIGN-INCONCLUSIVE
```

and the result is preserved exactly as produced.

**Frozen trigger — the only condition that yields this classification:** the null
is unsuitable iff the null distribution of `S_order` is **degenerate**, defined
here as `std(S_null) == 0.0` for any positive unit, so that no rank is defined.
Any other property of the null — heterogeneity, skew, bias, multimodality, a
heavy tail, a mean far from zero — is a **property of the null that is reported,
not a trigger.** This trigger is fixed now precisely so that "the null looks
wrong" cannot become a post-hoc escape from an unwelcome verdict.

---

## 13. OUTPUTS (§N, §O)

All written to `phase2/diagnostics/correspondence_index/20260911T015124Z/` only:

`PREREGISTRATION.md` (this file), `RESULTS.md`, `results.json`,
`ENVIRONMENT.txt`, `RELATED_RUNS.md`, `run.log`, `permutations.json` +
`permutations.sha256` (the frozen permutation file), `null_summary.json`
(summary/statistics file), `freeze_permutations.py` and `run_index_null.py`
(execution harness), plus a harness snapshot hash in `ENVIRONMENT.txt`.

Per (checkpoint, scene) the results table must contain:

```
ordered:  d(-1)   d(identity)   d(+1)   d(+2)   S_order   S_abs
random :  n  mean  median  SD  p01  p99  rank(S_order)  p_two  null_pass  order_pass
```

plus an aggregate summary across all positive checkpoints.

---

## 14. CLAIM CEILING — binding (§P)

**If PASS**, the only allowed conclusion is:

> "Candidate-coordinate sensitivity is established under direct candidate-axis
> re-indexing because the preregistered ordered response is distinguishable from
> the empirical random-permutation null."

The following **must not** be said: genuine correspondence, correct search,
geometric matching, correct disparity, stereo correctness. The words
*correspondence*, *matching* and *disparity search* must not be used as synonyms
for a positive result.

**If FAIL**, the only allowed conclusion is:

> "Candidate-coordinate sensitivity was not demonstrated by this diagnostic."

It must **not** be concluded that candidate-coordinate sensitivity does not
exist.

Claim levels, unchanged from INDEX-001:

- A. right-image dependence — already established, known confounds.
- B. candidate dependence — already established; Stage A showed the padding
  artefact alone produces it.
- **C. candidate-coordinate sensitivity — what this experiment targets.**
- D. geometric correspondence — **not** established here.
- E. genuine disparity search — **not** established here.

---

## 15. NO SPATIAL-STRUCTURE STATISTIC (§L)

This experiment introduces **no** spatial-structure metric. INDEX-001 correctly
identified that choosing a spatial statistic after seeing the random result would
be post-hoc. Only the preregistered scalar response statistic of §6 is used. If
that scalar proves insufficient, a spatial-structure experiment requires a
**fresh preregistration**.

---

## 16. HARD STOP CONDITIONS (§Q)

Execution stops immediately, with the partial record preserved, if:

- training begins, or any optimizer / `.backward()` / `.step()` appears;
- any loaded checkpoint sha256 differs from §7;
- the deterministic protocol cannot be established;
- the negative control fails (§11);
- the random permutations were not frozen before execution (contradicted by
  `permutations.sha256` predating the run);
- this preregistration is modified after execution begins;
- the primary statistic cannot be computed exactly as frozen in §6;
- the mask or reference frame is ambiguous and cannot be resolved from source
  (resolved in §4–§5 before freezing);
- an unexpected code path changes the cost volume;
- any historical record would need modification;
- runtime becomes unexpectedly expensive — **stop rather than optimise the
  experiment after launch.**

---

## 17. COMPUTE (§M)

Per (checkpoint, scene): one feature extraction + one cost-volume build, cached;
then 1540 aggregation+readout arms from that same cached tensor. 4 checkpoints ×
4 scenes = 16 volume builds, 24 640 arms. Measured pre-flight cost 0.00698 s per
arm → ≈172 s expected. No refinement is ever invoked. `disparity_initial` is the
sole endpoint; `disparity_final` is not measured and is not an endpoint.

---

## 18. HARD STOP ON SCOPE

The experiment ends when the §10.4 verdict is determined. No geometric
experiment, no synthetic-patch work, no candidate-count change, no dilation,
architecture, block-count, Scene Flow, pretrained or O6 work follows. Any attempt
to establish geometric correspondence requires a **separate pre-registration**
and must not be launched automatically.
