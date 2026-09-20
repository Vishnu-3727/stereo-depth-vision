# Phase 2 — H2 Evidence Review and Architecture Readiness Report

**Date:** 2026-09-08. **Scope:** everything Phase 2 has produced — H1 (v1, v2 and
its mechanism investigation), H2, H3, the H2 seed replication, the H2 emergence
measurement, the visualizer evidence, and the Phase 2 test and experiment
records. **Purpose:** decide whether Stage E (cost-volume optimization) can be
opened, and on what.

**This document measures nothing new except where §2.9 says so.** Every other
number is re-cited from a named record. Phase 1 is untouched (`phase-1-frozen`
= `b4207e5`; a test asserts the diff is empty).

## Claim tags

| Tag | Meaning |
|---|---|
| `MEASURED` | A number produced by a run in this project, with a named record and protocol. |
| `DERIVED` | Arithmetic or direct logical consequence of `MEASURED` numbers, adding no assumption. |
| `INFERRED` | A reading the evidence supports but does not force. Could be wrong without any measurement being wrong. |
| `UNKNOWN` | Not measured. Stated as a gap, never filled by argument. |

`OBSERVED VISUALLY` is used where the earlier reports used it: read off a named
image, quantitatively backed elsewhere.

---

## 1. Evidence inventory

| Record | What it is | Compute |
|---|---|---|
| `EXP-H1-BASE`, `EXP-H1-WORKING` | H1 v1, 20 epochs, seed 0 | ~0.2 h |
| `EXP-H1-BASE-v2`, `EXP-H1-WORKING-v2` | H1 v2, 200 epochs, seed 0 | ~2.0 h |
| `phase2/results/h1_mechanism/*` (12 probe jobs) | H1 mechanism investigation, inference/backward only | minutes |
| `EXP-H2-SOFTARGMIN-SCALE` | H2, 200 epochs, seed 0 | ~1.0 h |
| `phase2/results/h2_mechanism/*`, `h2_precheck/*` | H2 pre-check and mechanism probes | minutes |
| `EXP-H3-VIABILITY-001` | H3 cheap gate, 10 epochs | 215 s |
| `EXP-H2-SEED-REPLICATION-001-SEED{1,2}-RUN2` + `-SEED0-REFERENCE` + two aborted records | seed replication, 200 epochs ×2 | ~2.2 h |
| `EXP-H2-EMERGENCE-001-RUN2` (+ superseded `EXP-H2-EMERGENCE-001`) | emergence curve on saved snapshots | 42 s |
| `phase2/viz/`, `phase2/visualizations/`, `scene_ranking_hailo_val.json` | visualizer V1 (tooling, no record) | minutes |
| `phase2/tests/` | 78 Phase 2 tests | 45 s |

`MEASURED` — the full suite (Phase 1 + Phase 2) is **153 passing tests**,
re-run for this review (`python -m pytest tests phase2/tests`, 51.6 s, 4
warnings, 0 failures). Phase 2's own subset is 78.

`MEASURED` — total Phase 2 GPU cost to date is roughly **5.5 GPU-hours** on one
RTX 4060, dominated by five 200-epoch runs.

`UNKNOWN` — `PHASE_2_TASK.md` is cited by the registry and the H1 analysis (§24
stage order, §27 claim tags) but does not exist on disk or in git. The stage
ordering this report respects is therefore reconstructed from the registry and
the experiment reports, not from the original spec.

---

## 2. Claim ledger

### 2.1 H1 v1 — `EXP-H1-BASE` vs `EXP-H1-WORKING` (20 epochs)

| Claim | Tag |
|---|---|
| Under the 160-scene / 20-epoch / seed-0 protocol, final validation EPE and D1 are indistinguishable between `shift="none"` (19.191 px / 88.83 %) and `shift="left"` (19.212 px / 88.99 %). | `MEASURED` |
| **Neither arm's validation improved from initialisation at all.** | `MEASURED` |
| The working shift adds 0 parameters, 0 MACs, and no measurable latency (375 s vs 354 s wall clock, opposite sign to the hypothesis). | `DERIVED` |
| The mechanism was correct before training: `none` reproduces Phase 1's bit-identical 12 slices; `left` differs on all 11 non-zero slices and matches an independent reference to 1e-6. | `MEASURED` (4/4 tests) |
| The budget is too small to detect any effect — the null is confounded, not informative. | `INFERRED` |
| What a working cost volume is worth at a converged budget. | `UNKNOWN` at the time; answered in §2.2/§2.4 |

### 2.2 H1 v2 — 200 epochs

| Claim | Tag |
|---|---|
| The baseline now converges: train loss 11.49 → 4.98, val EPE 17.767 → 14.330, D1 88.74 → 81.19 %; early-window min EPE 17.767 vs late-window max 14.880, non-overlapping. | `MEASURED` |
| EPE difference between arms: −0.008 px late-window, direction inconsistent (5 of 11 checkpoints favour WORKING). | `MEASURED` |
| D1 difference: −0.64 pt late-window, 10 of 11 matched checkpoints favour WORKING. | `MEASURED` |
| Cost: identical 423,586 parameters, 0 MACs, 8 s in 3,630 (0.2 %). | `DERIVED` |
| EPE and D1 disagreeing is consistent — a working volume can move a thin band of near-threshold pixels without moving a mean dominated by large errors. | `INFERRED` |
| No p-value is claimable: one seed per arm, 11 autocorrelated checkpoints. | `DERIVED` |
| WORKING's max gradient norm 3.57e13 vs BASE's 1.36e4 (single batch; medians normal, loss stable). | `MEASURED` |
| Whether that spike is a numerical hazard or a curiosity. | `UNKNOWN` at the time; resolved in §2.3/§2.4 |

### 2.3 H1 mechanism investigation

This is the pivot of Phase 2 and its findings are load-bearing for everything after.

| Claim | Tag |
|---|---|
| The soft-argmin is saturated into a hard argmin in **both** H1 arms: aggregated cost 5e12 (BASE) / 6e19 (WORKING), best-vs-second-best gap ≥ 9.3e7, softmax entropy **0.0000** of a possible 2.485 nats. | `MEASURED` |
| The matching branch's derivative is therefore exactly zero. | `DERIVED` |
| BASE-v2's `disparity_initial` is the constant 11.0, and its output is **bit-identical** when the right image is replaced by black, noise, or another scene. | `MEASURED` |
| BASE-v2 is a monocular predictor. | `DERIVED` |
| WORKING's matching map is worth 0.30 pt of its 0.76 pt pooled D1 edge (78.69 / 77.93; flattened to a scalar 78.23; fed a constant 11 it *improves* to 77.19). | `MEASURED` |
| The D1 gap is a two-sided residue — 367,804 fixed vs 339,265 broken pixels of 3.8 M, mean \|BASE−WORKING\| 3.40 px everywhere, no concentration at discontinuities or by texture. | `MEASURED` |
| Gradient reaches the matching path on 5/150 crops (WORKING) and 0/150 (BASE); those 5 are exactly the >1e4 spikes; gradient enters the cost tensor at exactly 1 pixel of 131,072 where two candidates tie. | `MEASURED` |
| The spikes are tie-gated, not shift-specific (v1 had the larger max in the *other* arm), and show no border specificity. | `MEASURED` |
| Matching stages were trained and then saturated (drift from shared init: aggregation 1.92 / 2.50); a hand-made L1 cost volume on the trained features correlates with GT at only +0.165 / +0.215. | `MEASURED` |
| H1's 0.64-point answer was measured on a pathway that carried no correspondence information. | `DERIVED` |
| Causality of the tie → spike link. | `UNKNOWN` (no intervention was run) |
| Occlusion, repetitive texture and thin-structure behaviour. | `UNKNOWN` (no occlusion mask, no repetition detector) |

### 2.4 H2 — `EXP-H2-SOFTARGMIN-SCALE`

One changed variable: per-pixel standardisation of the aggregated cost across the
12 candidates immediately before the frozen soft-argmin. Control:
`EXP-H1-WORKING-v2`, not retrained.

| Claim | Tag |
|---|---|
| Zero new parameters — 423,586 in both arms, every weight tensor bit-identical after the swap. | `MEASURED` |
| Pre-check on the **control's own weights**: entropy 0.0000 → 2.0889 nats, max softmax weight 1.000000 → 0.253443, matching-path gradient 1/40 → 40/40 batches. | `MEASURED` |
| Training: matching-path gradient on **16,000 / 16,000 batches** (100.00 %), zero batches exactly zero, at every epoch from 0 to 199. Control: 5/150 crops at convergence. | `MEASURED` |
| Gradient became ordinary: total norm median 20.08, max **308.0**, zero batches above 1e4. Control max 3.567e13. | `MEASURED` |
| Tie-driven spikes absent: 0/150 crops vs the control's 5/150; ties still occur (14 in 800 batches) but no longer coincide with spikes. | `MEASURED` |
| That the standardisation *causes* the absence of spikes. | `INFERRED`, explicitly not established (association in one run per arm, no isolating intervention) |
| `disparity_initial` correlates with ground truth at **+0.978 … +0.988** (control −0.001 … +0.283). | `MEASURED` |
| The trained features themselves became more matchable: hand-made L1 volume argmin correlates +0.414 mean vs +0.215. | `MEASURED` |
| Destroying the matching map costs H2 **+71.5 D1 points**; the same intervention cost the control +0.30. | `DERIVED` |
| Corrupting the right image costs H2 **+81 to +84 D1 points** in every variant; the control moved by at most 0.8. | `MEASURED` |
| Switching the shift off at inference costs H2 +81.8 points; the control 0.75. | `MEASURED` |
| **H2 is the first configuration in this project whose output demonstrably depends on the second camera.** | `DERIVED` |
| Accuracy (secondary): late-window EPE 4.199 ± 0.090 vs 14.414 ± 0.216, D1 24.783 ± 1.024 vs 80.756 ± 0.379, same seed, budget, data and parameter count. | `MEASURED` |
| The improvement is global, not a residue: 2,416,765 control outliers become inliers (63.6 % of valid pixels), 88,814 break (2.3 %); every disparity, row, column and texture stratum improves by 41.8–78.1 points. | `MEASURED` |
| The raw aggregated cost still inflates during training (std 2.2e-3 → ~2e11); standardisation makes the inflation irrelevant to the softmax rather than preventing it. | `MEASURED` + `DERIVED` |
| The eps floor (1e-5) was never approached — smallest observed per-batch cost std 2.16e-3, 216× the epsilon. | `MEASURED` |
| Standardisation caps peak softmax confidence at ≈0.77 for a lone winner among 12; observed max weight 0.43–0.60. | `MEASURED` |
| Whether that cap costs accuracy. | `UNKNOWN` — "bounds attainable sub-pixel sharpness" is `INFERRED`, untested |
| Refinement still carries 0.867 of the final map's summed magnitude, but that share is unit-confounded (candidates vs pixels; one candidate is 16 px). | `MEASURED` + `DERIVED` |
| Refinement is no longer the only source of usable signal. | `INFERRED`, with the +71.5 pt intervention as the non-confounded evidence |
| Behaviour at full data scale. | `UNKNOWN` |
| What `shift="none"` does *with* a working soft-argmin. | `UNKNOWN` at the time; answered in §2.5 |

### 2.5 H3 — `EXP-H3-VIABILITY-001` (cheap gate, 10 epochs)

| Claim | Tag |
|---|---|
| H3 differs from H2 in exactly one field; identical initial weights from the same seed; cosine `T_max` kept at 200 so H3's ten epochs are the control's first ten. | `MEASURED` |
| Preflight: features bit-identical between arms, everything downstream differs by large margins (cost volume 3.7e9, aggregated cost 2.9e12, final disparity 69.7 max abs); no NaN/Inf; H2 hash unchanged. | `MEASURED` |
| The control is still stereo-functional at gate time: right-image corruption +86.5 … +90.9 D1 pt, matching-map destruction +75.4 pt. | `MEASURED` |
| H3 trains and learns: val EPE 31.99 → 15.4–16.5, train loss 10.37 → 8.11, matching gradient on 800/800 batches, no spikes, no non-finite values. | `MEASURED` |
| **H3 does not use the second camera**: corrupting the right image moves D1 by −1.1 to +1.9 pt; destroying its matching map makes D1 *better* by 0.07–1.14 pt (while EPE worsens). | `MEASURED` |
| Matched budget: H3 16.53 px / 88.71 % vs H2 10.54 px / 78.55 %; H3's init-vs-GT correlation is flat from epoch 1 (+0.585 → +0.616) while H2's climbs to +0.738. | `MEASURED` |
| **With the soft-argmin fixed, the disparity shift is what carries the stereo signal.** | `DERIVED` |
| With `shift="none"` there is no disparity search to learn, so the network reverts to predicting layout from the left image. | `INFERRED` |
| That H3 could never develop right-image dependence with far more epochs. | `UNKNOWN` — explicitly not established by a 10-epoch gate |
| H3's visual output is a monocular layout prior — smooth gradient, no vehicles, poles or façade steps, but not collapsed (span 6.64–50.07 px, per-scene std 8.7 px). | `OBSERVED VISUALLY` + `MEASURED` (non-collapse), with the budget confound stated |

### 2.6 H2 seed replication

| Claim | Tag |
|---|---|
| All three seeds are stereo-functional at 200 epochs against a threshold frozen before their results existed: right-image dependence +74.9 … +83.9 pt, matching-map +60.1 … +69.1 pt, matching gradient 100 % of all batches, entropy 1.73–1.88 nats, no NaN/Inf/spike. | `MEASURED` |
| Late-window D1 24.78 / 26.38 / 29.89 (span 5.11 pt, inside the frozen 10-pt band); late-window EPE 4.199 / 3.745 / 4.333 (span 0.588 px). | `MEASURED` |
| The H2 mechanism is not an artifact of seed 0. | `DERIVED` |
| Configuration identity verified field by field against the seed-0 record; 423,586 parameters and 16.1995 GMAC @256×512 in all arms; recipe imported, not restated. | `MEASURED` |
| **The first attempt's epoch-10 stereo gate was invalid** — the missing seed-0 control fails it too (−3.33 pt) despite reaching +83.9 by epoch 200. | `MEASURED` |
| Stereo dependence is a late-emerging property of this recipe. | `DERIVED` |
| **The harness is not bit-reproducible**: same seed, same code, epoch-0 loss 10.8798 vs 10.5762; four seed-0 runs span 10.576–10.880; divergence compounds to +2.44 train loss (≈33 %) by epoch 9. | `MEASURED` |
| No early-epoch difference between seeds may be attributed to the seed. | `DERIVED` |
| Whether the mechanism holds beyond three seeds, one dataset slice and one budget. | `UNKNOWN`; no significance claimed |
| The same-seed spread at epoch 200. | `UNKNOWN` — never measured, so the 10-point band cannot be claimed to separate seed effects from run-to-run noise |

### 2.7 H2 emergence

| Claim | Tag |
|---|---|
| Onset of stereo dependence (≥ +20 D1 pt, worst case): **seed 1 in (10, 20], seed 2 in (20, 50]**, holding at every later epoch. | `MEASURED` |
| Curve, seed 1: −1.26 → +32.35 → +58.68 → +73.43 → +78.08 → +76.78 pt; seed 2: −2.41 → −0.91 → +30.50 → +65.06 → +75.35 → +74.88 pt. | `MEASURED` |
| Re-measuring epochs 10/100/200 from the snapshots reproduces the training-time ablation **exactly** (0.00 pt at all six pairs). | `MEASURED` |
| **Matching-path gradient is not a proxy for stereo function**: 100 % of batches at every checkpoint, including seed 2 at epoch 20 where corrupting the right image *improves* D1 by 0.91 pt. | `MEASURED` → `DERIVED` |
| Softmax entropy falls from ≈2.0 to ≈1.67–1.79 nats as dependence appears, and stays there. | `MEASURED` |
| Dependence saturates after epoch 150 (both seeds dip slightly at 200). | `MEASURED` |
| Onset tracks achieved validation D1 more closely than epoch count (the two seeds align within ~1 grid step when matched on D1). | `INFERRED` — two seeds, six-point grid, and bounded by the non-reproducible harness |
| The onset epoch for seed 0. | `UNKNOWN` — its intermediate weights were never saved; only −3.33 pt at 10 and +83.95 pt at 200 exist |
| Anything finer than the six-point grid. | `UNKNOWN` |

### 2.8 Visualizer evidence

| Claim | Tag |
|---|---|
| The disparity convention is left-referenced `x_right = x_left − d`, verified photometrically on KITTI (7.6 vs 35.5 mean \|dI\| for the wrong sign). | `MEASURED` (test-pinned) |
| Phase 1's frozen `StereoCalibration.depth_from_disparity` turns `d = +inf` into a *valid* 0 m depth; guarded in `phase2/viz/core`, frozen code not edited. | `MEASURED` (test-pinned) |
| Scoring all 40 `hailo_val` scenes individually (a different protocol from the recorded 10-scene one), H1's WORKING-v2 wins D1 in 27/40, mean −0.77 pt, spread two-sided from −7.54 pt (scene 27) to +6.19 pt (scene 6). | `MEASURED` |
| All three H2 seeds produce structured maps — graded road, resolved car silhouettes, thin poles, façade steps; none is a smooth monocular gradient. H3's is. | `OBSERVED VISUALLY` |
| Right-image ablation figures agree with the numbers (seed 1 scene 27: 9.0 % → 97.9 %; seed 2 scene 6: 20.6 % → 97.8 %). | `OBSERVED VISUALLY` + `MEASURED` |
| The visual evidence never contradicts the quantitative evidence — no model looks plausible while failing the probes. | `INFERRED` |
| Failure-case categorisation (textureless / repetitive / reflective / thin structure / boundary). | `UNKNOWN` — no automatic categorisation, no object detector, chosen by eye only |
| The tool decided nothing; it is an observation instrument with a write-safety test. | `MEASURED` (mtime regression test) |

### 2.9 New measurement made for this review

`MEASURED` — `phase2/scripts/gt_range_ceiling.py`, output
`phase2/results/architecture_ceilings/gt_range_hailo_val.json`, all 40
`hailo_val` scenes, 3,802,797 valid pixels:

| quantity | value |
|---|---:|
| max ground-truth disparity | 153.04 px |
| median / p90 / p99 / p99.9 | 34.80 / 57.75 / 93.23 / 123.73 px |
| fraction above the architecture's 176 px ceiling | **0.0000** |
| fraction above 88 px | 0.0126 |
| EPE / D1 of an oracle that clips ground truth to 176 px | 0.000 / 0.000 |

`DERIVED` — **the 12-candidate × 16-px disparity range is not a binding
constraint on this dataset slice.** Any accuracy ceiling from the cost volume is
about candidate *spacing* and read-out, not *range*. This removes one candidate
architectural explanation before Stage E spends anything on it.

---

## 3. What H2 definitively establishes

Stated only at `MEASURED` / `DERIVED` strength, three seeds, 200 epochs.

1. **The failure that Phase 1 and H1 were both measuring was a read-out failure,
   not a correspondence-learning failure.** `MEASURED` — standardising the
   aggregated cost before the soft-argmin, changing nothing else and adding no
   parameter, moves softmax entropy from 0.0000 to 1.67–1.88 nats and
   matching-path gradient presence from ~3 % to 100 % of batches.
2. **This architecture *can* learn stereo correspondence.** `DERIVED` from three
   independent interventions, replicated across three seeds: destroying the
   matching map costs +60 to +71 D1 points, corrupting the right image costs +75
   to +84, switching off the shift costs +81.8. The matched control values are
   +0.30, ≤ +0.8 and +0.75.
3. **The disparity shift is the component that carries the stereo signal, once
   the read-out works.** `DERIVED` from H3: same architecture, same recipe, shift
   removed, right-image influence ≤ 1.9 D1 points and a matching map whose
   destruction *improves* D1.
4. **The fix is free.** `DERIVED` — 423,586 parameters and 16.1995 GMAC @256×512
   in every arm; the standardisation has no learnable state and the shift is pad
   + slice, 0 MACs.
5. **The mechanism reproduces.** `MEASURED` — 3/3 seeds stereo-functional by a
   factor of ~3–4 above a threshold frozen in advance, with identical gradient
   health and no H1 pathology in any run.
6. **Accuracy followed the mechanism, on the same budget.** `MEASURED` —
   late-window EPE 4.09 ± 0.31 px and D1 27.0 ± 2.6 % across seeds, against the
   H1 control's 14.41 px / 80.76 %. No significance is claimed and none is
   needed for a 56-point effect to be worth acting on.
7. **Stereo dependence is late-emerging, and its onset is now calibrated.**
   `MEASURED` — absent at epoch 10 in all three seeds, present by epoch 20
   (seed 1) or 50 (seed 2). Any future cheap gate has a curve to calibrate
   against instead of an assumption.
8. **Gradient presence is not evidence of stereo function.** `DERIVED` — 100 %
   of batches at every checkpoint of every seed, including checkpoints with
   negative right-image dependence. This retires a diagnostic the project used
   as a proxy twice.
9. **H1's answer is void as an answer about cost volumes.** `DERIVED` — its
   0.64-point D1 difference was measured between two models that both ignored
   the second camera. "What is a working cost volume worth?" has still never
   been measured at a converged budget with a working read-out.

---

## 4. What remains unknown

Ordered by how much each blocks architecture work.

1. **`UNKNOWN` — what the cost volume is worth on its own, at convergence.**
   H1 answered it on a broken pathway; H3 answered the shift question only at a
   10-epoch gate. There is no 200-epoch `shift="none"` + standardised read-out
   arm. The H3 kill decided *don't spend 200 epochs on it*, which is a compute
   decision, not a measurement.
2. **`UNKNOWN` — whether the confidence cap costs accuracy.** Standardisation
   caps a lone winner at ≈0.77 of the softmax mass; observed max weights are
   0.43–0.60 and `disparity_initial` occupies 1.34–9.03 of a possible 0–11
   candidate range. That the extremes are unreachable is `MEASURED`; that it
   costs EPE or D1 is untested.
3. **`UNKNOWN` — behaviour at real data scale.** Everything rests on 160
   training scenes from random initialisation. Scene Flow pretraining (O6) is
   deferred; the estimate is ~22 h at this GPU's measured 8.8 samples/s, blocked
   on download size rather than compute.
4. **`UNKNOWN` — the same-seed spread at 200 epochs.** Measured at 10 epochs
   (large, compounding to ≈33 % of train loss); never measured at 200. Until it
   is, the 5.11-point seed span cannot be separated from run-to-run noise.
5. **`UNKNOWN` — failure modes by scene class.** Textureless, repetitive,
   reflective, thin-structure and occlusion behaviour has never been measured in
   this project: no occlusion mask, no repetition detector, no Middlebury
   download (deferred since Phase 1, O4).
6. **`UNKNOWN` — whether the raw cost inflation (std → ~2e11) is harmful.**
   Standardisation makes it irrelevant to the softmax; nothing has tested whether
   preventing it helps optimisation.
7. **`UNKNOWN` — causality of the tie → gradient-spike link**, and whether the
   H1 spike regime could recur under a different read-out.
8. **`UNKNOWN` — physical device behaviour.** No Hailo silicon measurement exists
   anywhere in this project; the profiler report is a compiler estimate. Any
   latency claim about the target remains an estimate.
9. **`UNKNOWN` — where the remaining error lives.** H2 sits at 24.8–29.9 % D1
   and 3.6–4.3 px EPE against the pretrained reference's 8.15 % / 1.31 px, on a
   different training budget. Nothing has partitioned that gap into
   "architecture cannot represent it" versus "training has not reached it".

---

## 5. What H1 / H2 / H3 imply for architecture design

1. **Read-out numerics are a first-class architectural component, not a
   detail.** `DERIVED` — a zero-parameter change to the operator immediately
   before the soft-argmin moved D1 by 56 points and turned a monocular model
   into a stereo one. Nothing about the feature extractor, aggregation,
   refinement, candidate count or volume construction changed.
2. **Any component whose gradient can vanish by saturation must be
   instrumented, not assumed.** `DERIVED` — for the whole of Phase 1 and H1 this
   network trained to convergence with an exactly zero derivative through its
   matching branch, and every accuracy metric looked plausible throughout.
3. **A behaviour test is the only valid functional check.** `DERIVED` from §3.8:
   gradient presence, loss decrease and non-collapse were all satisfied by
   models with no stereo dependence. The right-image corruption and matching-map
   destruction probes are the instruments that separate them.
4. **The disparity shift earns its place; nothing else in the volume path has
   been shown to.** `DERIVED` from H3. Aggregation is 4.2 % of MACs and 2.8 % of
   GPU time (Phase 1, `MEASURED`), so the whole matching path is cheap — and it
   is now the part that works.
5. **Refinement is where the compute is and is no longer where the answer comes
   from.** `MEASURED` — refinement is 90.6 % of MACs and 73.3 % of GPU time;
   `DERIVED` — destroying the matching input now costs +71.5 D1 points, so the
   refinement network is consuming a signal it previously ignored. Its 90 %
   compute share has never been re-justified since the matching path started
   working.
6. **Efficiency work targeting aggregation is misdirected here.** `MEASURED`
   (Phase 1) — MACs do not predict latency on this stack (error 0.67× to 142×,
   ranking shifts 6× between GPU and CPU) and aggregation is 4.2 % of arithmetic.
   `DERIVED` — candidate-count reduction, the classic Stage E lever, is
   optimising a component that is 2.8 % of GPU time (6.4 % including volume
   construction) and 4.2 % of arithmetic.
7. **The disparity range is not the constraint.** `DERIVED` from §2.9 — 0 % of
   `hailo_val` ground truth exceeds the 176 px ceiling.
8. **Late emergence changes how any new architecture must be evaluated.**
   `DERIVED` — a 10-epoch screen cannot detect stereo function in this family;
   the earliest measured onset is epoch 20, and one seed needed 50.

---

## 6. Constraints any new architecture must satisfy

These are gates, not preferences. Each is justified by a `MEASURED` result above.

| # | Constraint | Why |
|---|---|---|
| C1 | The tensor consumed by the soft-argmin must have **softmax entropy > 0.5 nats** on validation frames, measured on the tensor the softmax actually consumes (not the raw cost). | H1's broken regime was 0.0000; the probe defect that measured the wrong tensor is recorded. |
| C2 | **Right-image dependence ≥ +20 D1 points**, worst case over the corruption variants, measured at ≥ 200 epochs or at a budget calibrated against a control. | The frozen criterion; H2 clears it by 3–4×, H1 and H3 fail it. |
| C3 | **Matching-map dependence ≥ +20 D1 points** (own-mean and shuffle substitutions). | Same criterion; separates "uses the volume" from "uses the left image". |
| C4 | Matching-path gradient on **≥ 95 % of batches**, and **no batch above 1e4** total norm. | Necessary but explicitly **not sufficient** — see C7. |
| C5 | No NaN/Inf anywhere; final disparity std > 1.0 px (no constant collapse). | H1's BASE arm emitted the constant 11.0. |
| C6 | Parameter and MAC cost stated against the 423,586 / 16.1995 GMAC @256×512 (56.04 GMAC @368×1232) reference, and latency measured, never inferred from MACs. | MACs mispredict latency by 0.67×–142× on this stack. |
| C7 | Any cheap gate must be **calibrated against a control measured at the same budget**, before it is used to kill anything. | Two healthy seeds were killed by an uncalibrated epoch-10 gate. |
| C8 | Any seed-to-seed claim must be bounded by same-seed run-to-run noise. | The harness is not bit-reproducible; divergence compounds to ≈33 % of train loss by epoch 9. |
| C9 | One changed variable per experiment, with the control's recipe **imported rather than restated**, and records never overwritten. | This is what made H2 and H3 interpretable at all. |
| C10 | Read-out sharpness must be reported: peak softmax weight and the attainable `disparity_initial` range. | H2's cap (≈0.77 max weight, 1.34–9.03 of 0–11 attainable) is an unmeasured accuracy risk. |

---

## 7. Candidate architectural hypotheses for Stage E

Five candidates, each stated as a falsifiable hypothesis with its evidence basis.
None is endorsed here; §8 gives the cheapest way to kill each and §9 picks the
first.

**E1 — Error partition: the remaining error is training budget, not
architecture.**
*Hypothesis:* an oracle that is allowed the best value the current cost-volume
read-out can express already reaches error far below H2's 3.6–4.3 px, so the
20-point gap to the reference is a training-data question (O6), not a Stage E
question. *Basis:* §2.9 shows range is not binding; §4.9 is unpartitioned.

**E2 — The confidence cap costs accuracy.**
*Hypothesis:* standardisation's ≈0.77 ceiling on peak softmax weight compresses
`disparity_initial` toward the candidate mean (measured attainable range
1.34–9.03 of 0–11) and costs sub-candidate precision; a tuned or learnable
temperature between "saturated" and "standardised" beats both. *Basis:* §2.4,
explicitly flagged `UNKNOWN` in the H2 report.

**E3 — Refinement is oversized now that matching works.**
*Hypothesis:* with the matching path supplying real structure, a substantial
fraction of the 6-block refinement stack can be removed for a small accuracy
cost — the largest available efficiency win, since refinement is 90.6 % of MACs
and 73.3 % of GPU time. *Basis:* §5.5.

**E4 — Cost-volume representation is suboptimal.**
*Hypothesis:* the subtraction volume is a weak matching representation; a
correlation or group-wise-correlation volume over the same features yields a
better-separated cost at equal or lower cost. *Basis:* H2's trained features
still only reach +0.414 hand-made-volume correlation with GT, against +0.98 for
the full aggregated path — so the raw representation is much weaker than the
learned aggregation that reads it.

**E5 — The cost-scale inflation is a real optimisation pathology.**
*Hypothesis:* the raw aggregated cost growing to std ~2e11 during training
reflects an ill-conditioned objective; constraining it at the source (bounded
cost, normalised features, cosine similarity) trains faster or further than
normalising after the fact. *Basis:* §2.4 — H2 made the inflation irrelevant to
the softmax but did not prevent it.

---

## 8. The cheapest falsification test for each candidate

Cost is stated in what it actually spends. "No training" means existing
checkpoints only.

| Candidate | Cheapest falsification | Cost | What kills it |
|---|---|---|---|
| **E1** | Oracle substitution, using the existing `pooled.json` intervention harness (it already replaces `disparity_initial` and re-runs refinement): replace `disparity_initial` with `a·gt + b`, where `a, b` are fitted once by least squares from the model's own realised `disparity_initial`-to-ground-truth relation (the model's candidate convention is learned, so an affine fit — not a hand-assumed ×16 — is what makes the substitution meaningful). Score EPE/D1 over the 40 scenes and compare with H2 as trained. | **minutes, no training** | If a *perfect* matching output leaves error close to H2's realised 3.6–4.3 px, the ceiling is downstream (refinement/read-out) and Stage E is an architecture problem. If it collapses the error, the matching stage is the limiter but is limited by training signal, not representation — Stage E stays closed pending O6. Report the fit's residual alongside, since a poor affine fit would itself be a finding. |
| **E2** | Inference-time temperature sweep on H2's trained weights: divide the standardised cost by τ over a range spanning saturated to smooth, and score EPE/D1 over 40 scenes. Uses the existing `cross_shift` / `pooled` protocol. | **minutes, no training** | If no τ beats τ = 1, the cap is not costing accuracy *for a model trained under it* — a weak negative. A positive result (any τ improves) is strong and promotes E2 to a retrain. |
| **E3** | Structured inference-time ablation of H2's refinement stack: drop residual blocks one at a time and in suffixes, score EPE/D1 and recount MACs. | **minutes, no training** | If dropping any block collapses accuracy, refinement is not oversized. If accuracy survives to 3–4 blocks, E3 is worth a retrain at reduced size. |
| **E4** | Build correlation and group-wise volumes **on H2's already-trained features**, and compare their argmin-vs-GT correlation with the +0.414 subtraction baseline, using the existing `feature_matching.json` protocol. | **minutes, no training** | If the alternatives do not beat +0.414 on the same features, the representation is not the limiter. A positive result needs a retrain to confirm, because features co-adapt to their volume. |
| **E5** | 10-epoch run with the constrained cost, gated against **H2's own first ten epochs** (already recorded) and judged on the emergence curve rather than on stereo dependence. | **~4 min training** (the H3 protocol, already scripted) | If it does not reach H2's epoch-10 EPE/loss band and its `disparity_initial`-vs-GT correlation does not track H2's, kill it. Note C7: the control at matched budget already exists, so this gate is calibrated. |

Two protocol notes carried from earlier failures: every "no training" test above
runs on checkpoints whose SHA-256 is recorded and must be re-asserted unchanged
afterwards (H3 did this), and no cheap gate may use stereo dependence before
epoch 20, per §2.7.

---

## 9. Which candidate should be tested first

**E1, the error partition.** `DERIVED` from §4.9 and §2.9.

It costs minutes and no training, and it is the only candidate whose result
changes what the *other four* are worth. Stage E is a cost-volume optimization
stage; optimising the cost volume is only rational if the cost volume's
attainable ceiling is actually near where the model sits. If the attainable-best
error is close to H2's realised 3.6–4.3 px, then the volume's geometry is the
binding constraint and E2/E4 become the substance of Stage E. If it is far
below — the outcome §2.9's range result makes plausible but does not establish —
then H2 is limited by training signal, Stage E stays closed, and the project's
next real move is O6 (Scene Flow pretraining), not architecture.

Second, conditional on E1 pointing at architecture: **E3**, because it is equally
cheap, targets 90.6 % of the compute rather than 4.2 %, and its answer is
directly a deployment number.

Third: **E2**, then **E4**, both of which need a retrain to be decisive and
should not be started until E1 has said whether read-out precision is on the
critical path.

**E5 last.** It is the only candidate that needs GPU time up front, and its
hypothesis is about optimisation quality — which E1 will have already
characterised more cheaply.

**Stage E stays closed until E1 has run.** That is one measurement, minutes of
compute, and it is the difference between optimising a component and optimising
noise — the exact failure this project has already made once, in H1.
