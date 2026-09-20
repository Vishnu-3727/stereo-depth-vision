# EXP-CORRESPONDENCE-INDEX-002 — RESULTS

**Empirical Permutation Null for Candidate-Axis Sensitivity**

Record `phase2/diagnostics/correspondence_index/20260911T015124Z/`.
Pre-registered in `PREREGISTRATION.md`, frozen before any arm ran. Permutations
frozen in `permutations.json` (sha256 `f064d09f…`, recorded in the
preregistration) before the preregistration was finalised.

Inference only — 16 cost-volume builds, **24 640 aggregation/readout arms**,
264.1 s wall clock. No training, no optimizer, no weight or architecture change,
no preprocessing change, no image or feature modification, refinement never
invoked. Phase 1 untouched. No historical record modified.

```
VERDICT: CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED
```

Driven by the empirical-null test (§10.1 of the preregistration): the ordered
response did **not** fall outside the central 99 % of the empirical
random-permutation null at any of the 12 positive units. The ordering test passed
12/12.

---

## 1. NEGATIVE-CONTROL GATE — PASS

The `shift="none"` volume satisfies `V(k) = left_feat − right_feat` for every `k`
(`reference_shift` pads right and slices `[0:W]`, returning its input), so
`V[:,:,π] = V` element-wise for **any** π. This is an algebraic identity, not a
statistical expectation. The measurement confirms it exactly, across the full
frozen permutation set:

| quantity | value |
|---|---:|
| arms checked (4 scenes × [4 ordered + 1536 random]) | **6 160** |
| `max_abs_diff(volume_identity, volume_permuted)` | **0.0** |
| `max_abs_diff(disparity_initial_identity, disparity_initial_perm)` (full frame) | **0.0** |
| `max_abs_diff(identity_indexed_volume, original_volume)` | **0.0** |
| violations | **none** |

`S_order = 0.0` and the entire null is `0.0` at every negative unit — the
degenerate model cannot respond, by algebra rather than by expectation. The
mandatory same-`index_select`-for-every-arm requirement (§A9) held: no layout or
cuDNN-algorithm artefact appeared.

This is a stronger gate than either INDEX-001 execution ran: those tested 4
permutations per scene, this tested 1 540.

---

## 2. PRIMARY RESULT — per (checkpoint, scene) unit

`d(x) = np.median(disparity_initial[mask])` in candidate units;
`S_order = [d(+2) − d(identity)] + [d(+1) − d(−1)]`;
null `S_random[i] = [d(π_a) − d(identity)] + [d(π_b) − d(π_c)]` over 512 disjoint
frozen triples.

| checkpoint | scene | d(−1) | d(id) | d(+1) | d(+2) | **S_order** | null mean | null med | null SD | null p01 | null p99 | rank | **p (2-sided)** | null pass | order pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| NEG_shift_none | 000187_10 | 5.7945 | 5.7945 | 5.7945 | 5.7945 | **+0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1/513 | 1.000000 | — | — |
| NEG_shift_none | 000160_10 | 6.9897 | 6.9897 | 6.9897 | 6.9897 | **+0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1/513 | 1.000000 | — | — |
| NEG_shift_none | 000191_10 | 7.8486 | 7.8486 | 7.8486 | 7.8486 | **+0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1/513 | 1.000000 | — | — |
| NEG_shift_none | 000166_10 | 5.5131 | 5.5131 | 5.5131 | 5.5131 | **+0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 | 1/513 | 1.000000 | — | — |
| POS_6b_seed0 | 000187_10 | 3.8579 | 5.6118 | 7.5528 | 8.3431 | **+6.4262** | 0.6140 | 0.7364 | 2.7529 | −5.629 | +6.517 | 507/513 | 0.027290 | **False** | True |
| POS_6b_seed0 | 000160_10 | 3.9881 | 5.8301 | 7.7237 | 8.4018 | **+6.3072** | 0.5864 | 0.7178 | 2.6931 | −5.524 | +6.145 | 508/513 | 0.023392 | **False** | True |
| POS_6b_seed0 | 000191_10 | 3.8441 | 5.5845 | 7.5619 | 8.4345 | **+6.5678** | 0.7570 | 0.8879 | 2.8401 | −6.107 | +6.866 | 506/513 | 0.031189 | **False** | True |
| POS_6b_seed0 | 000166_10 | 3.3378 | 5.1169 | 7.0400 | 8.2469 | **+6.8322** | 1.0628 | 1.1204 | 2.7673 | −4.987 | +6.826 | 508/513 | 0.023392 | **False** | True |
| POS_6b_seed1 | 000187_10 | 2.9871 | 4.2267 | 5.3187 | 6.1026 | **+4.2074** | 0.9482 | 1.1461 | 2.2231 | −4.249 | +5.003 | 487/513 | 0.105263 | **False** | True |
| POS_6b_seed1 | 000160_10 | 3.0700 | 4.2854 | 5.3792 | 6.0900 | **+4.1137** | 0.8372 | 1.0186 | 2.1568 | −4.161 | +5.019 | 488/513 | 0.101365 | **False** | True |
| POS_6b_seed1 | 000191_10 | 3.0026 | 4.2231 | 5.3338 | 6.0577 | **+4.1659** | 0.8891 | 1.0376 | 2.2854 | −4.691 | +5.009 | 479/513 | 0.136452 | **False** | True |
| POS_6b_seed1 | 000166_10 | 2.6318 | 3.8439 | 5.0478 | 6.0156 | **+4.5877** | 1.2098 | 1.3587 | 2.2330 | −3.682 | +5.843 | 486/513 | 0.109162 | **False** | True |
| POS_6b_seed2 | 000187_10 | 3.2555 | 4.7753 | 6.2317 | 7.2953 | **+5.4962** | 1.2469 | 1.4669 | 2.2634 | −4.848 | +5.990 | 504/513 | 0.038986 | **False** | True |
| POS_6b_seed2 | 000160_10 | 3.3576 | 4.8671 | 6.2535 | 7.1748 | **+5.2037** | 0.9095 | 1.1523 | 2.2318 | −4.353 | +5.416 | 507/513 | 0.027290 | **False** | True |
| POS_6b_seed2 | 000191_10 | 3.2947 | 4.7635 | 6.1981 | 7.3183 | **+5.4582** | 1.2575 | 1.5711 | 2.3417 | −4.171 | +6.136 | 500/513 | 0.054581 | **False** | True |
| POS_6b_seed2 | 000166_10 | 2.9025 | 4.3755 | 5.8801 | 6.9263 | **+5.5284** | 1.2098 | 1.3425 | 2.3551 | −4.046 | +6.268 | 498/513 | 0.062378 | **False** | True |

Retained pixel counts are 33 026 / 42 676 / 37 892 / 62 908 for scenes
000187_10, 000160_10, 000191_10, 000166_10 — **identical for every checkpoint and
every one of the 1 540 arms**, because the mask comes from ground truth alone.
They match the counts recorded by both INDEX-001 executions.

### Aggregate across the 12 positive units

| quantity | min | median | max |
|---|---:|---:|---:|
| `S_order` | +4.1137 | +5.4772 | +6.8322 |
| `S_abs` | +4.1137 | +5.4772 | +6.8322 |
| null mean | +0.5864 | +0.9288 | +1.2575 |
| null SD | 2.1568 | 2.3135 | 2.8401 |
| null p99 | +5.0025 | +6.0629 | +6.8656 |
| two-sided p | **0.0234** | 0.0468 | 0.1365 |

```
null_pass   0 / 12      (required: 12 / 12)
order_pass 12 / 12      (required: 12 / 12)
alpha = 0.01            min attainable two-sided p at M=512 = 0.003899
```

`S_order == S_abs` at every positive unit. That is not a coincidence and not a
result: all three ordered deltas happened to carry the sign the absolute-value
form assumes, so the two expressions coincide. `S_abs` is secondary and
descriptive; it changed no verdict.

---

## 3. ORDERING TEST — 12/12 PASS

```
d(+2) > d(+1) > d(identity) > d(−1)
```

holds at **12 of 12** positive (checkpoint, scene) units — fraction **1.000**.
`+1` is positive at 12/12, `−1` negative at 12/12, `+2 > +1` at 12/12. The
response is large: 1.2–1.9 candidates at `m=+1` and 1.8–3.1 at `m=+2`, against a
negative control that is bit-exactly zero.

This reproduces INDEX-001's 12/12 for a third independent time. **Per §H of the
protocol it is a reproducibility observation, not proof by itself**, and it is
not sufficient for the verdict under §10.4.

---

## 4. EMPIRICAL-NULL TEST — 0/12 FAIL

The ordered statistic sits **high** in the null but **inside** the frozen 99 %
envelope at every unit.

- Between **5 and 34** of the 512 null draws equal or exceed the ordered
  statistic (ranks 479–508 of 513).
- Two-sided p ranges **0.0234 … 0.1365**; the frozen threshold is **0.01**.
- `outside_central_99` is **False** at all 12 units: `S_order` never exceeds the
  null's own p99 (e.g. seed0/000187: `S_order = +6.4262` vs `p99 = +6.5169`;
  seed1/000191: `+4.1659` vs `+5.0091`).

The failure is not marginal at seed 1 (p ≈ 0.10–0.14) and is closest at seed 0
(p ≈ 0.023), still more than double the frozen threshold. **No threshold,
percentile, statistic or evaluation level was changed after seeing this.**

---

## 5. WHY THE NULL IS WIDE — MEASURED, not a rescue

Reported because §K requires null-quality reporting, and flagged as what it is: a
property of the null, **not** grounds to alter, repair or re-score anything.

**MEASURED.** An arbitrary candidate permutation moves the readout a great deal.
Across the 1 536 frozen draws, `d(π)` spans roughly:

| checkpoint (scene 000187_10) | min `d(π)` | median `d(π)` | max `d(π)` | `d(identity)` | `d(+2)` |
|---|---:|---:|---:|---:|---:|
| POS_6b_seed0 | 2.1129 | 6.7718 | 8.9545 | 5.6118 | 8.3431 |
| POS_6b_seed1 | 1.5972 | 5.4502 | 7.4689 | 4.2267 | 6.1026 |
| POS_6b_seed2 | 1.8922 | 6.4421 | 7.9243 | 4.7753 | 7.2953 |

The median random permutation moves the masked median **above** identity, and the
extremes reach further from identity than `d(+2)` does. A triple drawn from that
spread routinely produces `S_random` of the same magnitude as `S_order`, which is
why the null's p99 (+5.0 … +6.9) brackets the ordered values (+4.1 … +6.8).

**MEASURED.** The null is not centred on zero: null means are **+0.59 … +1.26**
across units, consistent with the upward shift of the median random `d(π)`.

**DERIVED.** The ordered response is therefore *typical in magnitude* of what
arbitrary candidate-axis re-indexing produces in this model, while being
*atypical in ordering* — the ordering test passes 12/12 and the magnitude test
fails 12/12. The frozen primary statistic tests magnitude-and-direction, not
ordering-across-arms, so it does not credit the ordering pattern.

**UNKNOWN.** Whether a statistic sensitive to the ordering *relation* among the
four ordered arms — rather than to the signed size of their contrast — would
separate the ordered family from this null. **That statistic was not
preregistered here and is not computed here.** Constructing one now, after seeing
that the frozen statistic failed, would be exactly the post-hoc move the protocol
forbids (task rules 9 and 11, preregistration §15). It is recorded as a design
observation for a future, separately preregistered experiment, and it is **not**
used to modify this verdict.

---

## 6. NULL-QUALITY REPORT (§K) — nothing removed

`null_summary.json` holds the full detail. Summary:

**Frozen null.** Seed `20260911`, `numpy.random.default_rng`, numpy 2.5.1;
1 536 permutations; 512 disjoint triples; **0 duplicates**; **0 draws removed**;
**no screening, no rejection** (`permutations.json` records
`"screening_applied": false, "rejection_applied": false`).

**Fixed points across the null** (recorded, never used to filter):

```
0 fixed points: 557    1: 555    2: 304    3: 95    4: 17    5: 8
```

**Retained candidate occupancy** (from GT only, model-independent):

```
pooled retained pixels 176 502   mean 2.8357   median 2.7507  (candidate units)
occupancy d = 2..8:  0.3532  0.4878  0.1527  0.0064  0.0000  0.0000  0.0000
fraction in d in [2,4]: 0.9936
```

The band is nominally `[2,8]` and effectively `[2,4]`, as the INDEX-001 audit
found. **The mask was deliberately not changed** — altering it would confound
comparison with INDEX-001, and §5 of the preregistration fixed this in advance.

**Displacement distribution across the 1 536 draws** (`π⁻¹(d) − d`):

| statistic | over all 12 candidates | over band `[2,8]`, uniform | over band, GT-weighted |
|---|---:|---:|---:|
| mean | 0.0000 | +0.4983 | +2.6780 |
| median | 0.0000 | +0.4286 | +2.6529 |
| SD | 0.0000 | 0.8886 | 1.9287 |
| p01 | 0.0000 | −1.4286 | −1.2610 |
| p99 | 0.0000 | +2.4286 | +6.7501 |
| min / max | 0.0 / 0.0 | −2.000 / +2.857 | −1.9629 / +7.3501 |

Mean displacement over all 12 candidates is exactly 0 for every permutation, by
construction. Over the retained band it is not, and under GT weighting the null
is strongly biased positive (mean +2.68) — the same occupancy effect the audit
identified. This is **reported, not repaired**: no draw was removed for being
biased, directional, strong, weak or close to an ordered permutation (§D).

**Reference-frame caveat on the band figures.** Mapping `GT/16` onto candidate
index assumes candidate magnitude equals disparity in feature-grid units, while
the volume indexes that magnitude at the **right** image column and GT indexes the
**left**. Those displacement figures are therefore descriptive design metadata,
not results. **They never entered the primary statistic**, which takes only
between-arm differences on an identical pixel set (§4 of the preregistration).

**No association analysis** between permutation displacement and observed
response was performed. None was preregistered; inventing one after inspecting
results is forbidden (task rule 11).

**`NULL-DESIGN-INCONCLUSIVE` was not triggered.** Its frozen trigger is a
degenerate null (`std(S_null) == 0`) at any positive unit. Observed null SDs are
2.16–2.84 at all 12 units; `degenerate_null_units = 0`. The null being *wide* is
not the trigger, and the trigger was fixed in advance precisely so that "the null
looks wrong" could not become a post-hoc escape from an unwelcome verdict.

---

## 7. GLOBAL DECISION RULE — applied literally

```
negative_gate_pass  = True    (6 160 arms, all diffs bit-exactly 0.0)
global_null_pass    = False   (0 / 12 units; required 12 / 12)
global_order_pass   = True    (12 / 12 units)

ESTABLISHED  iff  gate and global_null_pass and global_order_pass
                  = True and False and True
                  = False

-> CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED
```

No intermediate verdict was introduced. The evaluation level (all 12 units must
pass both tests) was fixed in §10.3 before execution; the outcome does not depend
on it here, since **zero** units passed the null test.

---

## 8. PROTOCOL DEVIATIONS

**One, recorded before execution, not after.**

`M = 512` null triples were frozen rather than the "preferred 64" of task §C.
Reason, stated in full in §9 of the preregistration: the frozen decision rule
(two-sided p ≤ 0.01) is **unattainable** at M = 64, where the smallest possible
two-sided p under the finite-sample rank formula is `2/65 = 0.0308`. M = 512
satisfies §C's hard floor ("at least 32") and makes §G's rule attainable
(`min p = 0.0039`) with ~5 draws in each 1 % tail. The choice was made from a
pre-flight timing probe (0.00698 s/arm, negative checkpoint only, no statistic
computed) and rank arithmetic alone, **before any response existed**.

Note that this deviation made the test **easier to pass**, not harder: at M = 64
no unit could have passed regardless of the data. The verdict is negative anyway.

**No other deviation.** The statistic, the aggregation, the mask, the arms, the
seed, the percentile, the alpha, the evaluation level and the verdict logic are
exactly as frozen. No spatial-structure statistic was introduced (§L). Runtime
was 264.1 s against a 172 s estimate — larger than projected because the negative
gate additionally re-derives the volume difference per arm, but not
"unexpectedly expensive", so the §Q hard stop did not trigger.

---

## 9. CLAIM CEILING

**MEASURED.** The degenerate `shift="none"` readout is bit-exactly invariant
(0.0 on both volume and `disparity_initial`) under all 1 540 candidate
permutations tested, on all four scenes — 6 160 arms. All three 6-block
checkpoints show a large, consistently ordered
`d(+2) > d(+1) > d(0) > d(−1)` response at 12/12 units. The empirical null of 512
frozen random triples has mean +0.59…+1.26, SD 2.16…2.84 and p99 +5.00…+6.87; the
ordered statistic (+4.11…+6.83) ranks 479–508 of 513, two-sided p 0.023…0.137.

**DERIVED.** The ordering criterion passes 12/12; the empirical-null criterion
passes 0/12; therefore the preregistered verdict is NOT-DEMONSTRATED.

**INFERRED.** The ordered response is typical in magnitude of what arbitrary
candidate re-indexing produces in this model, while remaining atypical in its
ordering across arms. The frozen statistic measures the former.

**UNKNOWN.** Whether the aggregation/readout path is candidate-coordinate
sensitive. **This experiment did not settle it.**

Per §P, the only allowed conclusion is:

> **Candidate-coordinate sensitivity was not demonstrated by this diagnostic.**

It is **not** concluded that candidate-coordinate sensitivity does not exist.

Not established, and not claimed: **D geometric correspondence**, **E genuine
disparity search**, correct matching, final disparity correctness, physical
translation slope, stereo depth correctness. The words *correspondence*,
*matching* and *disparity search* are not used as synonyms for any result here.
Candidate indices are **not** reinterpreted as physical disparity anywhere in the
primary statistic. Nothing is claimed about `disparity_final`, the refinement
path, sub-candidate precision, occluded / textureless / repetitive regions, any
block count, parameter-count or receptive-field causality, latency, transfer, or
optimal architecture.

---

## 10. REFERENCE-FRAME NOTE — resolved from source, recorded, not corrected

Read from `src/models/stereonet/cost_volume.py` before freezing. For
`shift="left"`, `shift_left(x, k)[..., u] = x[..., u+k]`, so candidate slice `k`
at output column `u` is `left_feat[u+k] − right_feat[u]`. With
`feature_stride = 16`, candidate `k` is a disparity of `16k` full-resolution
pixels — but stored at the **right** image's column, while GT and the mask are
indexed by the **left** image's column. GT is loaded at KITTI scale 256.0 (full-
resolution pixels); `disparity_initial` is a soft-argmin over candidate indices,
so it is in **candidate units**, directly comparable to `GT/16`.

**Preserved as found.** It does not affect the primary statistic, which uses only
between-arm differences of `disparity_initial` on an identical pixel set, so the
reference frame cancels. Converting candidate indices to physical disparity
belongs to a separate geometric experiment and was not done here.

---

## 11. PROVENANCE

`HEAD` `58e8a19`; `phase-1-frozen^{commit}` `b4207e5`;
`git diff phase-1-frozen -- src scripts` **empty**; working tree `M .gitignore`,
`?? phase2/`. Phase 2 remains untracked in git — **no cryptographic versioning is
claimed for this record.** All output is confined to this directory. No
historical record was modified; INDEX-001 was not re-run and not re-scored;
Stage A was not touched; checkpoints were opened read-only and every sha256 was
verified against the preregistration at load, with a hard stop on mismatch.

The closed block-count campaign was not reopened, no extra seed was introduced,
no image-translation or vertical-shift arm was run, no fill values were
introduced, no mask was altered after results were seen, and no threshold was
tuned post hoc.

See `RELATED_RUNS.md` for the full lineage.

---

## 12. FILES

`PREREGISTRATION.md`, `RESULTS.md` (this file), `results.json`,
`ENVIRONMENT.txt`, `RELATED_RUNS.md`, `run.log`, `permutations.json`,
`permutations.sha256`, `null_summary.json`, `freeze_permutations.py`,
`run_index_null.py`, `summarize_null.py`.
