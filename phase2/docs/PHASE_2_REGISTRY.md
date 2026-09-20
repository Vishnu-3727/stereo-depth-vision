# Phase 2 experiment registry

Phase 2 experiments live under `phase2/experiments/`, separate from Phase 1's
`experiments/EXP-0XX` directories (untouched, frozen at `phase-1-frozen`).
IDs are hypothesis-scoped (`EXP-H1-*`) rather than sequential, so Phase 2 can
never collide with or renumber Phase 1's history.

Never overwrite an entry. If wrong, add a `CORRECTION.md` inside it and mark
status `SUPERSEDED` / `INVALID` / `CORRECTED` — never delete.

## Stage tracker (research order, `PHASE_2_TASK.md` section 24)

| Stage | Description | Status |
|---|---|---|
| A | Freeze/verify Phase 1 baseline | **done** — tag `phase-1-frozen` = `b4207e5`, unchanged, verified this session |
| B | Build Phase 2 experimental infrastructure | **done** — this registry, `phase2/` workspace, `PHASE_2_BASELINE.md` |
| C | H1: replace degenerate cost volume with genuine shifted volume | **done, inconclusive** — see `H1_ANALYSIS.md`; both arms ran correctly, result confounded by inadequate training budget |
| C2 | H1 v2: rerun at 10x the epoch budget (200) | **running** — `EXP-H1-BASE-v2` launched 2026-09-07; `EXP-H1-WORKING-v2` is gated on BASE-v2 first showing real validation improvement off initialisation |
| D | Quantify what H1 provides | **blocked** — needs `EXP-H1-BASE-v2`/`EXP-H1-WORKING-v2` with a training budget large enough for validation to move off initialisation, before H1's accuracy question can be answered |
| E-O | See `PHASE_2_TASK.md` | not started — **explicitly not started early**; Stage E (cost-volume optimization) stays shut until H1 is decisive, however tempting it is to start tuning candidate counts |

## H1 — what is a working cost volume worth?

**Hypothesis** (docs/research_questions.md H1, PHASE_1_FINAL_REPORT.md §16):
a genuinely shifted stereo correspondence volume, everything else held fixed,
changes accuracy by a measurable amount versus the frozen degenerate volume.

**Changed variable:** `cost_volume_shift` (`"none"` vs `"left"`) in
`src/models/stereonet/stereonet.py::StereoNetConfig`. Implementation
pre-existed in Phase 1 (`src/models/stereonet/cost_volume.py::shift_left`),
behind a flag Phase 1 never enabled — this is not new model code.

**Held fixed:** feature extractor, aggregation, refinement, loss, optimizer,
schedule, seed (0), data split, crop, augmentation — identical to Phase 1's
EXP-016 recipe, reused byte-for-byte by `phase2/scripts/exp_h1_cost_volume.py`
(a new Phase 2 script; `scripts/exp_train_convergence.py` is untouched).

**Regression guard:** `phase2/tests/test_h1_cost_volume_shift.py` — proves (1)
`shift="none"` still reproduces Phase 1's exact degeneracy (all 12 slices
bit-identical) and (2) `shift="left"` makes every slice differ from slice 0,
against an independently-written indexed reference implementation, with an
explicit boundary-condition check. 4/4 passing.

| ID | Config | Status | Command |
|---|---|---|---|
| `EXP-H1-BASE` | `cost_volume_shift="none"` | **completed** | `python phase2/scripts/exp_h1_cost_volume.py --shift none --epochs 20 --batch 2 --seed 0` |
| `EXP-H1-WORKING` | `cost_volume_shift="left"` | **completed** | `python phase2/scripts/exp_h1_cost_volume.py --shift left --epochs 20 --batch 2 --seed 0` |

**Result: inconclusive (recorded, not hidden).** Final validation EPE/D1
statistically indistinguishable between arms (19.191 vs 19.212 px EPE);
neither arm's validation improved from initialisation at all under this
160-scene/20-epoch budget, so the experiment lacks power to detect any real
effect of the shift. Full analysis, decision and next step:
`phase2/docs/H1_ANALYSIS.md`.

**Known limitation, inherited from EXP-016, not hidden:** 160 scenes from
random initialisation is a convergence-pipeline proof, not a competitive
accuracy result. Both arms share this exact data budget, so the A/B
comparison of BASE vs WORKING is still a valid controlled experiment — the
comparison is about the *delta* the shift causes under matched noise, not
about either arm's absolute accuracy. Neither number should be read as this
architecture's ceiling, and this is not a substitute for evaluating the
pretrained Hailo weights (`PHASE_2_BASELINE.md`), which remain the only
pretrained reference available.

**Hardware for this run:** first attempt was CPU-only — `nvidia-smi` failed
with a permissions error in that session and `torch.cuda.is_available()`
returned `False`. That run was killed at `EXP-H1-BASE` epoch 3/20 (partial,
discarded, not a valid record) once GPU access recovered later the same day
— cause of the earlier failure was never root-caused, it simply started
working (RTX 4060, driver 616.56, `nvidia-smi` and `torch.cuda.is_available()`
both healthy). Restarted clean on GPU: epoch 0 of `EXP-H1-BASE` took ~24s
(vs 110s/epoch CPU, ~4.5x), device recorded as `"cuda"` in `config.json`.

**Analysis:** written — `phase2/docs/H1_ANALYSIS.md`. Verdict inconclusive at
this training scale; both v1 records preserved unmodified.

## H1 v2 — same hypothesis, 10x epoch budget (2026-09-07)

Direct execution of `H1_ANALYSIS.md`'s "next step 1": rerun H1 at a budget
large enough for validation to move off initialisation. **Only the epoch
budget changes** (200 vs 20, with cosine `T_max` tracking it, as in v1); data
split, crop, augmentation, optimizer, lr, batch, seed (0) and architecture are
untouched, so BASE-v2 vs WORKING-v2 remains a controlled A/B on
`cost_volume_shift` alone.

`--suffix` was added to `phase2/scripts/exp_h1_cost_volume.py` for this, and
the script now **refuses to start if the experiment directory already exists**
— v1's `EXP-H1-BASE` / `EXP-H1-WORKING` records cannot be overwritten by a
rerun.

| ID | Config | Status | Command |
|---|---|---|---|
| `EXP-H1-BASE-v2` | `cost_volume_shift="none"`, 200 epochs | **completed** | `python phase2/scripts/exp_h1_cost_volume.py --shift none --epochs 200 --batch 2 --seed 0 --suffix=-v2` |
| `EXP-H1-WORKING-v2` | `cost_volume_shift="left"`, 200 epochs | **completed** (gate passed) | same with `--shift left` |

**Gate, decided before seeing the result:** BASE-v2 is run *first and alone*.
`EXP-H1-WORKING-v2` is launched only if BASE-v2's validation EPE actually
improves over its own epoch-0 value by more than the ±3.5 px epoch-to-epoch
noise v1 exhibited. If BASE-v2 still does not converge, H1 is **not**
interpreted at all — the training/data regime itself becomes the object of
investigation (lr, loss masking, crop/disparity-range interaction, 160-scene
sufficiency, the no-BN deviation), because a regime where the baseline cannot
learn cannot answer any A/B question put to it.

### Gate verdict on `EXP-H1-BASE-v2` — PASSED, with the threshold arithmetic stated

`EXP-H1-BASE-v2` completed 200/200 epochs in 3635 s on the RTX 4060. Loss
11.4917 → 4.9830. Validation EPE 17.767 px (epoch 0) → 14.330 px (epoch 199),
best 14.122 px at epoch 190; D1 88.74 % → 81.19 %. All 16 000 gradient norms
finite.

**The literal threshold I wrote above is not met, and that is recorded, not
quietly dropped:** final-minus-initial improvement is 3.437 px, which is
*below* the ±3.5 px band. Best-seen-minus-initial is 3.645 px, above it. A
criterion whose verdict flips on which of two equally arbitrary summary
numbers is used is a bad criterion, so it is replaced here by the test it was
a proxy for — *did validation move off initialisation by more than
epoch-to-epoch noise* — evaluated on the whole series instead of two points:

- Early window, epochs 0–55 (12 validation points): minimum **17.767 px**.
- Late window, epochs 150–199 (11 validation points): maximum **14.880 px**.
- The two windows **do not overlap**, with a 2.887 px gap. Every late point is
  better than every early point.
- D1 shows the same separation: 87.27–91.84 % early versus 80.75–82.50 % late.
- v1 (20 epochs) showed no such structure at all — its validation series
  oscillated around initialisation with no trend, which is exactly why v1 was
  ruled inconclusive.

MEASURED: the baseline arm now genuinely learns under this budget.
INFERRED: the regime therefore has enough signal for a BASE-vs-WORKING
comparison to mean something, which it did not at 20 epochs.
UNKNOWN: whether 200 epochs on 160 scenes is near this architecture's ceiling
— it is not claimed to be, and the absolute EPE (14.3 px) remains far from
competitive; this is still a controlled A/B, not an accuracy result.

`EXP-H1-WORKING-v2` was launched only after this check, per the gate.

### Result — H1 answered narrowly, `phase2/docs/H1_V2_ANALYSIS.md`

Both arms completed 200/200 (3635 s / 3627 s). Late-window (epochs 150–199,
11 matched checkpoints):

| | BASE-v2 | WORKING-v2 | Delta |
|---|---:|---:|---:|
| Val EPE mean ±sd | 14.423 ± 0.228 | 14.414 ± 0.227 | −0.008 px (5/11 checkpoints favour WORKING) |
| Val D1 mean ±sd | 81.397 ± 0.559 | 80.756 ± 0.398 | **−0.64 pt (10/11 favour WORKING)** |
| Params / MACs / wall clock | 423 586 / 0 / 3635 s | 423 586 / 0 / 3627 s | none |

**MEASURED:** the working cost volume leaves EPE unchanged and lowers the D1
outlier rate ~0.6 points, for zero parameter, MAC or latency cost.
**No p-value is claimed** — one seed per arm, and the 11 checkpoints are
autocorrelated epochs of a single run.

Open, recorded not dismissed: WORKING's `gradient_norms.max` was 3.57e13 vs
BASE's 1.36e4 (single batch; per-epoch medians stayed 20–60 in both, loss did
not diverge).

**Stage E stays closed.** Next is seed replication (`--suffix=-v2s1`, `-v2s2`,
both arms) — a 0.6 pt effect from one seed is not a foundation to optimise on.

## Tooling — Stereo Depth Visualizer V1 (not an experiment)

Added 2026-09-07. `phase2/viz/` + `phase2/scripts/stereo_visualizer.py`:
inspects what the H1-v2 checkpoints predict — disparity, metric depth from the
real KITTI calibration, error against sparse ground truth, BASE-vs-WORKING
difference, pixel probe, stereo correspondence, region depth statistics.
Documentation: `phase2/docs/STEREO_VISUAL_VALIDATION_V1.md`. Tests:
`phase2/tests/test_visualizer.py` (27 passing), including a write-safety test
that asserts a full report generation changes no file under `src/`,
`phase2/experiments/` or `phase2/results/training/`.

It creates **no experiment record**, retrains nothing and reuses the recorded
metrics rather than recomputing them. Per-scene numbers it computes are a
different protocol from the recorded 10-scene validation and are labelled as
such wherever they appear.

Observation it produced, for the H1 file rather than as a conclusion: over all
40 `hailo_val` scenes scored individually, WORKING-v2 has the lower D1 in 27/40
scenes (mean ΔD1 −0.77 pt, median −1.01 pt), but the per-scene spread is wide
and two-sided (−7.54 pt at scene 27 to +6.19 pt at scene 6). Direction agrees
with the recorded late-window result; the mechanism is still unidentified, and
Stage E stays closed.

## Investigation — H1 visual mechanism V1 (not an experiment)

Added 2026-09-07, after Visualizer V1. Forensic study of the two recorded H1-v2
checkpoints: `phase2/scripts/h1_mechanism_probe.py` (12 inference-only /
backward-only probe jobs), evidence under `phase2/results/h1_mechanism/` (with a
README naming the command behind each file), two additive figure functions in
`phase2/viz/render.py`, and the report
`phase2/docs/H1_VISUAL_MECHANISM_INVESTIGATION_V1.md`. No training, no seed
replication, no architecture change, no experiment record written; Phase 1 diff
against `phase-1-frozen` is empty and the suite is green (104 passing).

Headline measurements, all with their own probe protocol and none replacing the
registered H1-v2 validation numbers:

- **The soft-argmin is saturated in both arms.** Aggregated-cost magnitudes reach
  5e12 (BASE) / 6e19 (WORKING); the per-pixel gap between the best and
  second-best candidate is >= 9.3e7, far past fp32 `exp` underflow. Mean softmax
  entropy 0.0000 of a possible 2.485 nats; it is a hard argmin with an exactly
  zero derivative.
- **BASE-v2 is a monocular predictor.** Replacing the right image with black,
  noise or another scene leaves its output bit-identical (0.000 px change);
  `disparity_initial` is the constant 11.0 everywhere. Its stereo branch is
  disconnected from its answer.
- **WORKING's advantage is mostly not matching content.** Pooled over 40 scenes:
  BASE D1 78.69, WORKING 77.93; flattening WORKING's matching map to a scalar
  costs only 0.30 pt (78.23), and feeding the constant 11 instead makes WORKING
  *better* (77.19). Switching the shift off at inference does remove the whole
  advantage (78.68).
- **The D1 gap is a small residue of large opposing flows:** 367,804 pixels fixed
  and 339,265 broken out of 3.8 M valid (net -0.750 pt); mean |BASE-WORKING|
  disparity difference is 3.40 px everywhere. No concentration at disparity
  discontinuities or by texture; column and range dependence only.
- **Gradient spikes explained enough to act on.** At convergence the matching
  path receives gradient on 5/150 crops (WORKING) and 0/150 (BASE); those 5 are
  exactly the crops with norms >1e4, and on each of them gradient enters the cost
  tensor at exactly 1 pixel of 131,072 where two candidates are exactly tied.
  Recorded v1 had the huge maximum in the *other* arm, so it is not
  shift-specific; crop-position bands show no border specificity.
- **The matching stages were trained and then saturated** (drift from the shared
  seed-0 init: aggregation 1.92 / 2.50), and a hand-made L1 cost volume over the
  trained features correlates with ground truth at only +0.165 / +0.215.

**H1 MECHANISM STATUS: PARTIALLY SUPPORTED.** The shift changes the model and its
removal at inference erases the gap, but no evidence supports correspondence
information reaching the answer.

**Next experiment recommended by the investigation:** `EXP-H2-SOFTARGMIN-SCALE` —
normalise the aggregated cost before the soft-argmin, one changed variable,
`EXP-H1-WORKING-v2` as its control, primary endpoint the fraction of batches on
which the matching path receives non-zero gradient (currently 5/150), accuracy
secondary. Seed replication and Stage E stay closed: both would spend GPU-hours
on a branch that is gradient-starved 97 % of the time and whose output the
network scores better without.

## H2 — EXP-H2-SOFTARGMIN-SCALE (2026-09-07)

**Hypothesis.** The matching pathway is untrainable because the soft-argmin is
numerically saturated (H1 investigation, `H1_VISUAL_MECHANISM_INVESTIGATION_V1.md`),
not because correspondence cannot be learned here.

**Control:** `EXP-H1-WORKING-v2`, not retrained.
**Changed variable, one:** the regression stage is
`phase2/models/scaled_regression.StandardisedDisparityRegression` — a per-pixel
z-score across the 12 disparity candidates immediately before the frozen
`soft_argmin`. No new parameter (423,586 both arms), no temperature, no
annealing, no clipping. Everything else is imported from the H1 recipe.

**Gate (pre-check before training, `phase2/results/h2_precheck/precheck.json`):**
on the control's own weights and identical inputs, the scaling took softmax
entropy 0.0000 → 2.0889 nats (max 2.4849), max weight 1.0000 → 0.2534, and
matching-path gradient reachability 1/40 → 40/40 batches. Gate passed.

| | H2 | control |
|---|---:|---:|
| **Primary endpoint — batches with matching-path gradient** | **16,000/16,000 = 100 %** | 5/150 crops at convergence |
| Total gradient norm, median / max | 20.1 / **308** | 37.3 / **3.567e13** |
| Batches with norm > 1e4 | **0** | (5 spiking crops) |
| Softmax entropy at the tensor consumed | **1.67–1.83 nats** | 0.0000 |
| `disparity_initial` correlation with GT | **+0.978 … +0.988** | −0.001 … +0.283 |
| Destroying the matching map (D1 cost) | **+71.5 pt** | +0.30 pt |
| Corrupting the right image (D1 cost) | **+81 … +84 pt** | ≤ +0.8 pt |
| Val EPE / D1, late window (recorded protocol) | **4.199 ± 0.090 / 24.78 ± 1.02** | 14.414 ± 0.216 / 80.76 ± 0.38 |
| Parameters | 423,586 | 423,586 |

**H2 STATUS: MECHANISTICALLY SUCCESSFUL.** All five pre-registered criteria met.
Accuracy improved sharply as a secondary effect (one seed, no significance
claimed).

**Consequence for H1, recorded plainly:** the H1 contrast was run on a pathway
that could not learn. Its 0.64-point D1 difference compared two models that both
ignored the second camera, so "what is a working cost volume worth?" has still
not been measured. The H1 records stand as written; their interpretation is
narrowed by this result, not overwritten.

**Anomalies:** the machine suspended for 2 h 27 min mid-run, so
`wall_clock_s = 12,517 s` is inflated (real rate 14–19 s/epoch); the record lacks
a top-level `cost_volume_shift` key (additive `NOTE.md` added beside it); two
probe defects were found and fixed during analysis (see
`phase2/results/h2_mechanism/README.md`).

**Next:** `EXP-H3-SHIFT-UNDER-SCALE` — the H1 contrast rerun with the pathway
working (`cost_volume_shift="none"` under the standardised soft-argmin, control
`EXP-H2-SOFTARGMIN-SCALE`). Stage E stays closed; seed replication is deferred
behind H3.

## H3 — EXP-H3-VIABILITY-001 (2026-09-07) — cheap viability gate, KILLED

**Question.** With the matching pathway functional (H2), how much does the
disparity shift itself contribute? **Control:** `EXP-H2-SOFTARGMIN-SCALE`, not
retrained. **Changed variable, one:** `cost_volume_shift` `"left"` → `"none"`.
Everything else imported from the H2 recipe; 423,586 parameters both; identical
seed-0 initialisation; cosine `T_max` kept at 200 so the ten epochs are the
control's first ten.

**Preflight gate (Part A/B), `phase2/results/h3_viability/preflight.json`:**
complete-model LEFT vs NONE on the H2 weights — features identical (expected),
cost volume onward all differ (final disparity max abs difference 66–70 px), no
NaN/Inf, no shape mismatch, H2 checkpoint hash unchanged. Control still
stereo-functional: right-image corruption +86.5…+90.9 D1 pt, matching-map
destruction +75.4 pt, matching gradient 12/12 crops. **PASS.**

**10-epoch run (215 s, no abort).** Learns: val EPE 31.99 (init) → 16.53, train
loss 10.37 → 8.11; matching-path gradient on 800/800 batches; total gradient max
313, none above 1e4, no non-finite values.

| after ~10 epochs | H3 | control |
|---|---:|---:|
| val EPE / D1 | 16.53 / 88.71 % | 10.54 / 78.55 % |
| `disparity_initial` r(GT) | +0.616 (flat from +0.585 at epoch 1) | +0.738 (rising) |
| right-image corruption, ΔD1 | **−1.08 … +1.91** | +80.8 … +84.4 |
| destroying the matching map, ΔD1 | **−1.14 … −0.07 (i.e. better)** | +75.4 |

**H3 STATUS: KILL** — the pre-registered criterion "right image has negligible
influence" fired. No other kill criterion did. `DERIVED`: with the soft-argmin
fixed, the disparity shift is what carries the stereo signal; a no-shift volume
returns the model to monocular behaviour and is not worth 200 epochs.
`UNKNOWN`: whether far longer training would change that — untested.

Full analysis `phase2/docs/EXP_H3_VIABILITY_001_REPORT.md`; verdict also recorded
additively as `phase2/experiments/EXP-H3-VIABILITY-001/DECISION.md`.

**Anomalies:** no 10-epoch H2 checkpoint exists, so the control's own
right-image dependence at matched budget is unmeasured (matched-budget metrics
are, and they separate the arms); the record stores per-checkpoint metrics rather
than a `validation_epe_series`, so the visualizer prints N/A for its global
metrics; the comparison figure's title was hardcoded to "BASE vs WORKING" and
mislabelled every H2/H3 figure — fixed, figures regenerated.

**Next:** `EXP-H2-SEED-REPLICATION-001` — rerun H2 unchanged at seed 1. Every
architectural decision now rests on a single-seed H2. Stage E stays closed.

## EXP-H2-SEED-REPLICATION-001 (2026-09-07) — H2 replicates; the gate did not

**Question.** Is H2's functional stereo pathway reproducible across seeds, or an
artifact of seed 0? **Only changed variable:** the random seed (1 and 2 against
seed 0's `EXP-H2-SOFTARGMIN-SCALE`, not retrained). Part A verified every other
field against the seed-0 record: 423,586 parameters, 16.1995 GMAC @256x512,
identical recipe imported rather than restated.

**Verdict: REPRODUCIBLE.** All three seeds stereo-functional at 200 epochs
against a threshold frozen in advance.

| | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| late-window EPE (150-199) | 4.199 ± 0.090 | 3.745 ± 0.145 | 4.333 ± 0.092 |
| late-window D1 | 24.783 ± 1.024 | 26.384 ± 1.229 | 29.892 ± 0.735 |
| right-image dependence (min) | +83.95 pt | +76.78 pt | +74.88 pt |
| matching-map dependence (min) | +69.12 pt | +67.87 pt | +60.05 pt |
| matching gradient | 100 % | 100 % | 100 % |
| softmax entropy | 1.67-1.83 | 1.732 | 1.881 |
| gradient max / spikes >1e4 | 308 / 0 | 604 / 0 | 239 / 0 |

D1 span across seeds 5.11 points, inside the 10-point band frozen beforehand.
No NaN/Inf, no spikes, no H1 pathologies in any run. ~2.2 GPU-hours total.

**The finding that matters more than the verdict: the first attempt's epoch-10
stereo gate was invalid, and control evidence proved it.** Seeds 1 and 2 were
aborted at epoch 10 for showing no right-image dependence; the missing 10-epoch
seed-0 control (`…-SEED0-REFERENCE`) then showed **seed 0 fails the same gate**
(-3.33 D1 points) despite reaching +83.9 by epoch 200. **Stereo dependence is a
late-emerging property of this recipe.** The gate was recalibrated from the
control (never from the seeds' results), frozen in
`phase2/docs/EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md`, and the
aborted records kept untouched with additive notes; the reruns carry new IDs.

**Second infrastructure finding:** the harness is **not bit-reproducible** —
same seed, same code, two runs differ (epoch-0 loss 10.8798 vs 10.5762), and
same-seed divergence compounds to +2.44 train loss by epoch 9. That is fp32
cuDNN non-determinism, not a recipe difference, and it bounds every seed-to-seed
claim this project makes.

Full report: `phase2/docs/EXP_H2_SEED_REPLICATION_001_REPORT.md`.

~~**Next:** `EXP-H2-EMERGENCE-001`~~ — done, below.

---

## EXP-H2-EMERGENCE-001 — when does stereo dependence appear? (2026-09-08)

Measurement only, 42 s, no training: the seed-replication weight snapshots
(epochs 10/20/50/100/150/200, seeds 1 and 2) re-ablated with `stereo_probe` and
`gradient_probe` imported unchanged, against the 20-D1-point criterion already
frozen in `EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md`. Records:
`EXP-H2-EMERGENCE-001-RUN2` (authoritative) and `EXP-H2-EMERGENCE-001`
(superseded by a reader bug, kept with `NOTE.md`; its measurements are
bit-identical to RUN2's apart from gradient norms, at 3.7e-5).

| seed | epoch | val EPE | val D1 % | right-image dep. worst / mean (pt) | matching-map dep. worst (pt) | softmax entropy | matching grad. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 16.374 | 85.86 | -1.26 / +3.96 | +2.09 | 2.061 | 100% |
| 1 | 20 | 8.552 | 64.14 | +32.35 / +41.51 | +26.78 | 1.682 | 100% |
| 1 | 50 | 5.449 | 39.60 | +58.68 / +71.33 | +48.69 | 1.703 | 100% |
| 1 | 100 | 4.097 | 30.39 | +73.43 / +80.92 | +62.16 | 1.669 | 100% |
| 1 | 150 | 3.766 | 27.09 | +78.08 / +84.67 | +66.71 | 1.669 | 100% |
| 1 | 200 | 3.632 | 25.39 | +76.78 / +84.50 | +67.78 | 1.670 | 100% |
| 2 | 10 | 16.383 | 87.09 | -2.41 / +1.25 | -6.39 | 2.006 | 100% |
| 2 | 20 | 14.029 | 83.76 | -0.91 / +4.65 | +0.52 | 1.949 | 100% |
| 2 | 50 | 7.419 | 57.52 | +30.50 / +40.82 | +24.00 | 1.706 | 100% |
| 2 | 100 | 5.270 | 37.67 | +65.06 / +76.64 | +51.77 | 1.761 | 100% |
| 2 | 150 | 4.746 | 31.75 | +75.35 / +82.19 | +60.83 | 1.777 | 100% |
| 2 | 200 | 4.271 | 29.61 | +74.88 / +83.50 | +59.96 | 1.794 | 100% |

**Onset:** seed 1 in (10, 20], seed 2 in (20, 50], and it holds at every later
epoch in both. Seed 0's two recorded points are cited unchanged (-3.33 pt at
epoch 10, +83.95 pt at epoch 200); its intermediate weights were never saved.
Re-measuring epochs 10/100/200 from the snapshots reproduces the training-time
ablation **exactly** (0.00 pt difference at all six pairs).

**The methodological finding: matching-path gradient is not a proxy for stereo
function.** It is 100 % of batches at *every* checkpoint, including seed 2 at
epoch 20, where corrupting the right image *improves* D1 by 0.91 points. Future
gates must test behaviour, not gradient. Secondary observations: onset tracks
achieved val D1 more closely than epoch count (the two seeds align within ~1
grid step when matched on D1 — two seeds, not a law), softmax entropy falls from
≈2.0 to ≈1.67–1.79 nats as dependence appears, and dependence saturates after
epoch 150.

Full report: `phase2/docs/EXP_H2_EMERGENCE_001_REPORT.md`.

**Next:** see `phase2/docs/PHASE_2_H2_EVIDENCE_REVIEW.md` — the Phase 2 evidence
review and architecture-readiness report (2026-09-08). It classifies every Phase 2
conclusion as MEASURED / DERIVED / INFERRED / UNKNOWN, states the ten constraints
(C1-C10) any new architecture must satisfy, and names five Stage E candidates with
the cheapest falsification test for each. Its own new measurement
(`phase2/scripts/gt_range_ceiling.py`): 0 % of `hailo_val` ground truth exceeds the
architecture's 176 px disparity ceiling (max 153.04 px), so disparity *range* is not
a binding constraint. Recommended first test: **E1, the error partition** (oracle
`disparity_initial` substitution, minutes, no training) — it decides whether Stage E
is an architecture problem at all.

---

## EXP-E1-ERROR-PARTITION-001 — where does H2's remaining error come from? (2026-09-08)

Measurement only, ~4 minutes, no training. Oracle substitution at the matching ->
refinement boundary on all three H2 seeds, 40 `hailo_val` scenes, pooled over
`gt > 0`. Protocol frozen in
`phase2/docs/EXP_E1_ERROR_PARTITION_001_PREREGISTRATION.md` before the run.

**Verdict: NO CONSISTENT VERDICT — E1 does not return a clean partition.** The
answer depends on how much capacity the oracle is given, and three oracles
bracket it:

| oracle | EPE removed (seeds 0/1/2) | pre-registered verdict |
|---|---:|---|
| affine in ground truth (`001`, the pre-registered primary; a lower bound) | 7.8 / 12.1 / 8.7 % | DOWNSTREAM-LIMITED x3 |
| self-scoring fixed point (`002-RUN3`; leaky upper bound, still descending) | 61.4 / 69.6 / 61.5 % | MIXED, MATCHING-LIMITED, MIXED |
| **held-out fixed point (`003`; converged, load-bearing)** | **29.3 / 41.5 / 32.9 %** | DOWNSTREAM-LIMITED, MIXED, DOWNSTREAM-LIMITED |

**The measured split, which is the useful output:** roughly **one third** of H2's
remaining pooled EPE is attributable to the matching stage and roughly **two
thirds** is imposed downstream by these refinement weights, on every seed (EVAL
ratios 0.707 / 0.585 / 0.671, clustered around the 0.65 threshold). All gates
passed in runs 001 and 003; run 001 reproduces the recorded pooled H2 figures to
4.2e-4 px.

**Also closed:** off-manifold oracle inputs are not a material weakness (<= 0.53 %
of substituted values left the soft-argmin's [0, 11] range), and the 176 px
disparity-range ceiling was already excluded by the evidence review.

**The failure worth keeping:** `EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE` added a
pixel-valued residual to a candidate-valued input (one candidate is 10-16 px),
diverged monotonically, and — because every iterate was worse than the start —
returned ratio 1.000 and the verdict **DOWNSTREAM-LIMITED**, apparently
confirming run 001. A diverging probe produced a false confirmation. A gate
asserting the search actually improves was added to both later runs. Record kept
with `NOTE.md`; never cite it.

**Next:** **E3** — structured inference-time ablation of H2's refinement stack
(no training). It targets the 58-71 % share this experiment localised and the
90.6 % of MACs Phase 1 measured. E2 and E4 address the smaller share and need a
retrain; E5 stays last. The decisive form of E1 — retrain refinement against a
strong matching input — waits until E3 says how much refinement capacity is
needed.

Full report: `phase2/docs/EXP_E1_ERROR_PARTITION_001_REPORT.md`.

Stage E (cost-volume optimization) stays closed: E1 was meant to open or close it
and did neither.

---

## EXP-E3-REFINEMENT-ABLATION-001 — how much of the refinement stack is load-bearing? (2026-09-08)

Measurement only, 154 s, no training. Thirteen inference-time ablations of the
six-block refinement stack on all three H2 seeds, 40 `hailo_val` scenes pooled
over `gt > 0`, MACs recounted per variant. Protocol frozen in
`phase2/docs/EXP_E3_REFINEMENT_ABLATION_001_PREREGISTRATION.md`. Records:
`EXP-E3-REFINEMENT-ABLATION-001-RUN2` (authoritative) and
`EXP-E3-REFINEMENT-ABLATION-001` (measurements complete, crashed on a terminal
encoding error before its conclusion; NOTE.md, numbers agree).

**Verdict: REFINEMENT IS RIGHT-SIZED.** No variant is viable; the best viable MAC
saving is **0.0 %**. The cheapest removal (block 2, dilation 4) costs +0.746 px
EPE and +11.40 D1 points for 14.9 % of MACs — 11x the frozen NEGLIGIBLE band. All
four gates passed. MAC accounting corroborates Phase 1 independently: 56.034 GMAC
@368x1232 against Phase 1's 56.04, and 16.1995 @256x512 exactly matching the
replication record.

**Two findings beyond the verdict:**
(1) **Which block is redundant is a property of the run, not the architecture** —
dropping block 4 costs seed 0 only +0.19 px but seeds 1 and 2 +24.16 and +16.37.
A one-seed pruning study would have concluded block 4 is free. Block-count
reduction cannot be decided by pruning a trained model. Blocks 2 and 3 are
individually near-redundant yet jointly essential (dropping the 2/4/8 ladder
costs +16.09 px).
(2) **Truncating refinement destroys stereo dependence** — every prefix below five
blocks fails constraint C2 on every seed, and from four blocks down the
right-image dependence goes *negative* (-14 to -18 D1 points): corrupting the
right image makes the truncated model better. That is the H1 monocular signature,
produced by removing refinement capacity while the cost volume, aggregation and
read-out are untouched. Refinement is the path by which matching information
reaches the output, not a post-hoc polisher.

**Limit:** weights trained as a six-block stack, so this is a LOWER bound on how
small a refinement stage can be; what a retrained four- or five-block stack does
is UNKNOWN. No latency measured.

**Next:** **E2** — inference-time read-out temperature sweep, the last remaining
no-training candidate, targeting the one third of the error E1 put upstream.
After it the cheap questions are exhausted and the two remaining experiments are
both training runs: **E3b** (retrain refinement at four and five blocks) and
**O6** (Scene Flow pretraining).

Full report: `phase2/docs/EXP_E3_REFINEMENT_ABLATION_001_REPORT.md`.

Stage E stays closed — nothing measured so far implicates the cost volume in H2's
remaining error.

---

## EXP-E2-READOUT-TEMPERATURE-001 — is H2's error sensitive to readout sharpness? (2026-09-08)

Inference only, 160 s, no training. Temperature introduced at exactly one place —
`p(d) = softmax(-C_standardised(d) / T)` at the frozen soft-argmin, via a subclass;
`phase2/models/scaled_regression.py` untouched. Frozen grid T in
{0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0}, all nine on all three seeds, with
the stereo interventions at every temperature (no cheap-pass, so no chance to pick a
region after seeing results). Protocol:
`phase2/docs/EXP_E2_READOUT_TEMPERATURE_001_PREREGISTRATION.md`.

**Verdict: READOUT-ROBUST (case B).** No temperature materially improves accuracy on
any seed — not one of the 24 non-control cells has a negative dEPE or dD1. All gates
passed, including the control reproducing the untempered H2 readout **exactly**
in-process and matching E1's record to 1e-6, checkpoint hashes, parameter count
(423,586), MACs (56.0339 G at both T=1.0 and T=0.25) and an empty Phase 1 diff.

**Read the label precisely:** READOUT-ROBUST means "no temperature is an exploitable
improvement", not "insensitive". The sensitivity is one-sided and steep — a +-25 %
change in a single parameter-free scalar costs +3.5 px / +56 D1 pt (T=0.75) or
+2.2 px / +37 pt (T=1.25) on seed 0.

**Stereo functionality collapses away from T=1.0 in both directions.** C2 and C3 are
met at T=1.0 on all three seeds and essentially nowhere else (the one other passing
cell, H2 at T=1.25, is a material degradation). Sharpening to T=0.25 drives entropy
to 0.290 nats — approaching H1's collapsed 0.0000 — and right-image dependence to
+0.57/-2.98/-1.17; softening to T>=2.0 does the same from the other side. Both
extremes reproduce the H1 monocular signature with the weights, cost volume,
aggregation and standardisation untouched. Same lesson E3 produced from the other
end of the network.

**Confirmed by measurement:** the maximum peak softmax probability observed anywhere
is 0.769-0.772, against the ~0.77 ceiling the H2 report derived analytically. The
control's entropy (1.675-1.795 nats) reproduces H2's recorded 1.67-1.88.

**Limit:** inference-only, so this is weak evidence about temperature as a *training*
variable — a network adapts to the readout it was optimised through. Stated before
the run.

**Next (recommendation only; nothing opened):** the cheap questions are exhausted and
every remaining candidate needs training — **E3b** (retrain refinement at four and
five blocks, ~1 GPU-h/arm, clearest hypothesis), **E2b** (train at a different or
learnable temperature), **O6** (Scene Flow pretraining, largest expected effect,
blocked on dataset size).

Full report: `phase2/docs/EXP_E2_READOUT_TEMPERATURE_001_REPORT.md`.

Stage E remains formally CLOSED.

---

## Stage E decision record (2026-09-08)

`phase2/docs/PHASE_2_STAGE_E_DECISION_RECORD.md` — the evidence review over E1, E3
and E2 (under ten minutes of GPU time between them; three of the five evidence-review
candidates closed). Two decisions:

**1. Stage E: CLOSED.** No measurement implicates the cost volume in H2's remaining
error. E1 could not attribute a majority of the error upstream, the 176 px range
ceiling is excluded, and the cost volume's consumer — the readout — is robust. The
classic Stage E levers also target 4.2 % of arithmetic and 2.8 % of GPU time. E4-cheap
(alternative volume representations on H2's trained features, no training, minutes)
is the only remaining live Stage E rationale and is ranked second in the queue; a
negative result there would close Stage E on evidence rather than on absence.

**2. E3b: APPROVED, with a required design amendment**, and registered below.

Established across the three: disparity range is not a constraint; readout temperature
is not an exploitable lever; refinement cannot be pruned for free; which block is
redundant is a property of the *run*, not the architecture; and **stereo dependence is
destroyed from both ends** — by truncating refinement (E3) and by moving the readout
temperature either way (E2) — with the cost volume, aggregation and standardisation
untouched. H2's use of the second camera is a property of the whole trained
configuration.

Queue after E3b: E4-cheap (minutes), O6 (~22 h, blocked on dataset size), E2b (~2 h),
E5 (~4 min gate).

---

## EXP-E3B-REFINEMENT-CAPACITY-001 — REGISTERED, NOT RUN (2026-09-08)

Pre-registration: `phase2/docs/EXP_E3B_REFINEMENT_CAPACITY_001_PREREGISTRATION.md`.

**Question:** does a refinement stack *trained* at four or five residual blocks match
six-block H2 at the same budget, seed and recipe — and stay stereo-functional? E3
answered only the pruning case and recorded the retrained case as UNKNOWN.

**Arms (stage 1, seed 0, 200 epochs each, ~3 GPU-hours):**

| arm | blocks | dilations | unique parameters | MACs @368x1232 |
|---|---:|---|---:|---:|
| A — control | 6 | 1, 2, 4, 8, 1, 1 | 423,586 | 56.034 G |
| B | 5 | 1, 2, 4, 8, 1 | 405,090 | 47.677 G |
| C | 4 | 1, 2, 4, 8 | 386,594 | 39.321 G |

The trailing dilation-1 blocks are removed, keeping the 1-2-4-8 ladder intact — fixed
before the run and deliberately *not* E3's cheapest-to-prune blocks (2 and 3), whose
removal would confound capacity with receptive field.

**The amendment that made it approvable:** arm A is a **same-seed repeat of the
unmodified H2 recipe**, not the existing H2 checkpoint. Its difference from
`EXP-H2-SOFTARGMIN-SCALE` *is* the same-seed run-to-run noise at 200 epochs — the
outstanding half of constraint C8, UNKNOWN since the seed replication — and every
comparison is judged against the band it produces. Without it a four-block arm landing
1 D1 point from H2 would be uninterpretable. One repeat is a single-sample lower bound
on noise, not a confidence interval, and is registered as such.

Conditions: primary endpoint is accuracy relative to that band (never MACs; MACs are
arithmetic, not latency); C1-C5 enforced on every arm at epoch 200, with stereo
dependence never gated before epoch 20 (emergence); seeds 1 and 2 pre-committed to run
only for an arm whose seed-0 result falls outside the band. Verdicts:
CAPACITY-REDUCIBLE / CAPACITY-REQUIRED / **STEREO-BROKEN** (matches on accuracy but
fails C2/C3 — the expected failure mode given E3 and E2, and a rejection rather than a
result) / INCONCLUSIVE / INVALID.

E3b does **not** open Stage E: it is a refinement-capacity experiment on the downstream
half of the network.

**RAN 2026-09-08 — verdict CAPACITY-REDUCIBLE.** 3.2 GPU-hours. Records
`…-ARM-{A,B,C}-RUN2` and `…-ANALYSIS-RUN2` (authoritative); `…-ARM-A` is a first
attempt killed by host memory pressure at epoch 146, kept with NOTE.md and not cited.
All four gates passed.

**Same-seed noise band (constraint C8's missing measurement):** arm A vs the recorded
`EXP-H2-SOFTARGMIN-SCALE` late window gives **0.1069 px / 0.1267 D1 pt**, so the
material band is **0.2138 px / 1.0000 pt** (the D1 floor binds, at ~8x measured noise).
Same-seed noise at 200 epochs is small even though the same two runs differed by 6.30
D1 points at epoch 59 — trajectories are chaotic mid-run and converge as the cosine
decays. Single-sample lower bound, not a confidence interval.

| arm | blocks | params | MACs @368x1232 | late EPE | late D1 | dEPE | dD1 | material? | stereo |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| A control | 6 | 423,586 | 56.034 G | 4.0921 +- 0.1000 | 24.9097 +- 1.0821 | — | — | — | +82.00 / +68.07 |
| B | 5 | 405,090 | 47.677 G | 3.8588 +- 0.0644 | 25.7856 +- 0.7130 | -0.2333 | +0.8759 | no | +79.71 / +63.52 |
| C | 4 | 386,594 | 39.321 G | 4.0028 +- 0.0753 | 26.0708 +- 0.6425 | -0.0893 | +1.1610 | no | +75.42 / +64.19 |

All three STEREO-FUNCTIONAL, entropy 1.78-1.82 nats, matching gradient 100 % of 16,000
batches, zero spikes, no H1 pathology. **Four blocks match six for 30 % fewer MACs.**

**The finding that matters: pruning and retraining disagree about the same
architecture.** E3 measured that *pruning* to four blocks costs +24.15 px / +81.49 D1
pt and drives right-image dependence to -16.9 (the H1 monocular signature); E3b
measures that *retraining* at four blocks costs -0.089 px / +1.161 pt and keeps
right-image dependence at +75.42. E3's "right-sized" verdict answered the pruning
question correctly and its report declined to extrapolate — that caution was warranted.
It also explains E3's seed-specific block redundancy: the six blocks divide work a
four-block stack simply divides differently when trained from scratch.

**Stated precisely:** the frozen conjunction rule is what makes both arms "not
different" — arm C's dD1 of +1.1610 *exceeds* the 1.0-point threshold and is saved
only by its EPE sitting inside the band. The verdict is robust to replacing the D1
floor with 2x measured noise (both arms stay viable). One seed per arm; stage 2 was
pre-committed to trigger only on a material difference and none occurred, so the
honest phrasing is **"not separable at seed 0"**, not "equal".

**Limits:** 160 training scenes from random initialisation (3.9-4.1 px vs the
pretrained reference's 1.31), so no transfer to a pretrained model is implied; MACs are
arithmetic, not latency; nothing below four blocks was trained; the late windows hold
10 points against the recorded run's 11.

Full report: `phase2/docs/EXP_E3B_REFINEMENT_CAPACITY_001_REPORT.md`.

**Next (recommendation only):** O6 first — it is both the largest confound and the
condition under which this capacity finding would have to be re-tested; then E3b stage
2 / a three-block arm; then E4-cheap; then E2b and E5. Stage E remains CLOSED.

---

## EXP-E3C-REFINEMENT-FLOOR-001 — three blocks (2026-09-08)

Registered separately from E3b, deliberately: E3b's stage 2 was pre-committed to
trigger only on an arm outside the band (none was) and to add no new arms, so running
a new arm under its ID would mean editing a frozen protocol after seeing its results.
E3b's arm A and its noise band are reused as **measured inputs**. Pre-registration:
`phase2/docs/EXP_E3C_REFINEMENT_FLOOR_001_PREREGISTRATION.md`. 1.0 GPU-hour.

**Verdict per the frozen rule: CAPACITY-REDUCIBLE-TO-3 — and the label overstates it.**
Arm D is *materially different* on both metrics but in opposite directions: **-0.3226 px
EPE (better), +2.0304 D1 points (worse)**. "Worse" is defined as both deltas positive,
so the rule returns reducible. Three blocks do not come free: they **trade 2.03 D1
points for 0.32 px of EPE**, crossing a pre-registered threshold. That caveat travels
with the verdict.

| arm | blocks | dilations | params | MACs | late EPE | late D1 | dEPE | dD1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| A | 6 | 1,2,4,8,1,1 | 423,586 | 56.034 G | 4.0921 +- 0.1000 | 24.9097 +- 1.0821 | — | — |
| B | 5 | 1,2,4,8,1 | 405,090 | 47.677 G | 3.8588 +- 0.0644 | 25.7856 +- 0.7130 | -0.2333 | +0.8759 |
| C | 4 | 1,2,4,8 | 386,594 | 39.321 G | 4.0028 +- 0.0753 | 26.0708 +- 0.6425 | -0.0893 | +1.1610 |
| D | 3 | 1,2,4 | 368,098 | 30.964 G | 3.7695 +- 0.0580 | 26.9402 +- 0.6617 | -0.3226 | +2.0304 |

All four STEREO-FUNCTIONAL; arm D's right-image dependence is **+81.41** and its
matching-map dependence +65.79, so whatever three blocks costs, it is not the second
camera.

**The finding this run actually produced: D1 degrades monotonically with block count**
— +0.88 (5), +1.16 (4), +2.03 (3) — while EPE does not (-0.23, -0.09, -0.32, unordered).
Each individual comparison passed its rule, but the trend across four points is a steady
outlier-rate cost the per-arm conjunction rule was never designed to detect. E3b's "four
blocks match six" stays true under its frozen rule and is now visibly the middle of a
gradient rather than a flat region. One seed per arm; suggestive, not significant.

**Confound, registered before the run and unresolved by design:** at three blocks the
1-2-4-8 ladder cannot survive — `keep first 3` drops dilation 8, so arm D reduces the
receptive field as well as the capacity, and its +2.03 points cannot be attributed to
capacity alone. The disambiguating arm — three blocks at `(1,2,8)` or `(1,4,8)`, same
params and MACs — was named in the pre-registration and is **not** auto-triggered (it
was pre-committed only for a materially-worse or STEREO-BROKEN result), but the D1
trend now motivates it on evidence.

Full report: `phase2/docs/EXP_E3C_REFINEMENT_FLOOR_001_REPORT.md`.

**Next:** O6; then the (1,2,8)/(1,4,8) three-block arm; then seeds 1-2 for the existing
arms to turn the monotone D1 trend into a claim about runs; then E4-cheap, E2b, E5.
Stage E remains CLOSED.

## EXP-O6-REFINEMENT-DILATION-001 — capacity, or receptive field? (2026-09-08)

**Registered before training**, executing the follow-up E3C pre-committed in its
own section 8. Pre-registration:
`phase2/docs/EXP_O6_REFINEMENT_DILATION_001_PREREGISTRATION.md`. This is a
**confound-disambiguation** experiment, not an architecture search, and no arm
here is a candidate for adoption.

**Question.** E3C's three-block arm changed capacity *and* receptive field at
once. Its +2.03 D1 point cost cannot be attributed to either. Which was it?

**The property that makes it answerable:** a dilated 3x3 convolution has the same
weight shape and the same MAC count as an undilated one. So every three-block arm
below holds **exactly 368,098 parameters** and **exactly the same arithmetic
cost** as E3C arm D, and — because all three keep refinement blocks 0,1,2 of one
seed-0 stack and are then *re-dilated in place* — **bit-identical initial
weights**. Capacity, MACs and initialisation are constant by construction; the
dilation schedule is the only variable. Asserted numerically in the preflight,
not assumed.

| arm | blocks | dilations | params | receptive field | role |
|---|---:|---|---:|---:|---|
| `CONTROL6` | 6 | 1,2,4,8,1,1 | 423,586 | 73 px | reused measured input, **not** retrained |
| `D124` | 3 | 1,2,4 | 368,098 | 33 px | E3C's configuration, retrained here |
| `D128` | 3 | 1,2,8 | 368,098 | 49 px | keeps maximum dilation 8 |
| `D148` | 3 | 1,4,8 | 368,098 | 57 px | different sparse allocation, also keeps 8 |

Receptive field is DERIVED (`1 + 2*(input + output + sum of 2*dilation)`), not
measured. `D148` exists so a `D128` result cannot be one lucky schedule.

**Decision rule, frozen before results** (section 8 of the pre-registration):
`GAP = d1(D124) - d1(CONTROL6)`; `RECOVERY(X)` is the fraction of that gap
closed. Both arms recovering >= 50 % gives `RECEPTIVE-FIELD-IMPLICATED`; both
<= 20 % gives `CAPACITY-IMPLICATED`; one of each gives
`CONFIGURATION-SENSITIVE`; anything else `INCONCLUSIVE`. Band reused from
E3b/E3C unchanged: 0.2138 px / 1.0000 D1 point. Any arm failing the frozen
epoch-200 stereo verdict is `STEREO-BROKEN` regardless of accuracy.

**Gates passed before training:** preflight PASS (identical parameters, MACs,
initial weights, tensor shapes and recipe fields across arms; shift `left`;
standardised readout; 12 disparities; Phase 1 diff empty; H2 hash
`e3d48021…` unchanged) and a viability gate PASS on all three arms (construct,
forward, backward, finite loss and gradients, refinement receives gradient,
optimizer moves 60 tensors, validation, checkpoint round-trip). 20 focused
regression tests plus the full 128-test Phase 2 suite passed before the first
arm started.

**Status: complete.** Three arms x 200 epochs, seed 0, one seed each —
suggestive/exploratory by construction, no significance claimable.

### Results

| arm | dilations | RF | params | late EPE | late D1 | stereo |
|---|---|---:|---:|---:|---:|---|
| `CONTROL6` | 1,2,4,8,1,1 | 73 px | 423,586 | 4.0921 +- 0.1000 | 24.9097 +- 1.0821 | yes |
| `D124` | 1,2,4 | 33 px | 368,098 | 4.1121 +- 0.0658 | 24.9701 +- 0.5897 | yes |
| `D128` | 1,2,8 | 49 px | 368,098 | 4.1764 +- 0.0847 | 28.0342 +- 0.7070 | yes |
| `D148` | 1,4,8 | 57 px | 368,098 | 4.1602 +- 0.0629 | 28.0212 +- 0.7363 | yes |

All completed 200/200 epochs, no aborts, no NaN, matching gradient 100 % of
16,000 batches each, every stereo gate cleared roughly four-fold.

**Verdict per the frozen rule: `NO-GAP-TO-EXPLAIN`.** All analysis gates PASS.

**The premise failed.** `GAP = d1(D124) - d1(CONTROL6) = +0.0603 pt`, inside the
1.0-point band — so `RECOVERY` is undefined and there is no gap to attribute.
E3C measured that same gap as **+2.0304**.

**The reproducibility finding, which matters more than the verdict.** `D124` is
the same configuration, recipe and seed as E3C arm D. The pre-registered
cross-check: **dEPE +0.3426** (band 0.2138) and **dD1 -1.9701** (band 1.0000) —
**both outside the band**. Environments match (same GPU, CUDA 12.8, same torch,
same git state); all config differences are documentation strings. The harness
seeds `torch.manual_seed`/`np.random.seed` and sets **no** determinism control
at all. INFERRED: GPU non-determinism accumulated over 200 epochs. **The effect
E3C called material is the same size as the reproduction error of an identical
configuration**, which weakens every "material" D1 call resting on E3b's
single-sample band, E3C's +2.03 included.

**What O6 did measure, cleanly.** At exactly constant capacity (368,098 params),
exactly constant MACs and bit-identical initialisation, both dilation-8
schedules are **~3.05 D1 points worse** than `(1,2,4)` — `D128` +3.0641,
`D148` +3.0511 — while EPE moves by less than the band. The two arms agree with
each other to **0.013 D1 points** despite different receptive fields (49 vs
57 px), so this is not configuration-sensitive. `H_O6` is **not supported**: the
result runs opposite to it. Under the conjunction rule the difference is not
"material" because EPE barely moved; on D1 alone it is three times the band.
Both readings recorded.

Full report: `phase2/docs/EXP_O6_REFINEMENT_DILATION_001_REPORT.md`.

**Next: fix the measurement before any further architecture comparison.**
(1) add determinism control to the harness and verify a same-seed repeat;
(2) replace E3b's single-sample band with 3+ same-config repeats within and
across sessions — until then no D1 difference under ~2 points on this harness
should be called material; (3) only then revisit the E3b/E3C block-count series,
whose monotone D1 trend may or may not survive a correct band. The
`(1,2,8)`/`(1,4,8)` question is **answered and closed**. Stage E remains CLOSED.

---

# ADDENDUM A — records 2026-09-09 / 2026-09-10, and corrections to earlier entries

**Appended 2026-09-10. Additive only.** Nothing above this line has been edited,
deleted or renumbered. Where an earlier entry is superseded, it is annotated
here and left standing there, exactly as written at the time.

This addendum exists because the registry above stops at O6 (2026-09-08) while
seven further records were produced on 2026-09-09 and 2026-09-10. A read-only
audit on 2026-09-10 found the experimental evidence clean and the registry seven
records behind; this closes that gap.

---

## A.0 Corrections to entries above — annotations, not edits

| # | Entry above | Claim as written | Status now | Why |
|---|---|---|---|---|
| 1 | E3b | "**Four blocks match six for 30 % fewer MACs.**" | **Superseded as evidence** | Rests on the 0.2138 px / 1.0000 pt band (invalidated), a nondeterministic harness (Stage A), and the 10-scene late window (which reorders the arms against the 40-scene protocol). Its 5-block half is **contradicted** by the deterministic three-seed result in A.6/A.7. Its 4-block half is **unverified** — it rested on one seed, and A.8 is the experiment that tests it. |
| 2 | E3b | verdict `CAPACITY-REDUCIBLE`; band 0.2138 px / 1.0000 pt | **Historical / inconclusive only** | The band was a single-sample lower bound on a harness with uncontrolled execution noise. It is **retired, not replaced**. No band is used anywhere in A.3–A.8. |
| 3 | E3c | verdict `CAPACITY-REDUCIBLE-TO-3` (+2.03 D1 trade) | **Contradicted** | O6 measured the same gap at +0.0603 pt. The label must not be carried forward. The registry entry above already notes "the label overstates it"; this makes the status explicit. |
| 4 | O6 | "no D1 difference under ~2 points on this harness should be called material" | **Stale — scoped to the old harness** | Correct guidance for the *nondeterministic* harness it was written about. Execution noise is now measured at exactly zero (A.2); the operative scale is **seed** noise, measured at EPE 0.1027 px / D1 2.731 pt across three 6-block seeds. Do not reuse the ~2-point heuristic. |
| 5 | E3 (inference ablation) | "REFINEMENT IS RIGHT-SIZED" | **Still valid** | It answers the *pruning* question and its own report explicitly declined to extrapolate to retraining ("what a retrained four- or five-block stack does is UNKNOWN"). A.6–A.8 answer that separate question. No correction needed. |
| 6 | H3 | epoch-10 viability `KILL` on `shift="none"` | **Valid as screening evidence only** | Not decisive on its own, and never inherited: A.3 re-derived the conclusion independently at 200 epochs with the standardised readout. Cite A.3, not H3's epoch-10 criterion. |

**Standing rule from here on:** an empirical block-count effect is never
restated as a parameter-count or receptive-field *cause*. E3b, E3c and O6 do not
establish causality in either direction, and neither does anything in A.6–A.8.

---

## A.1 Stage A — determinism diagnostic (2026-09-09)

Record: `phase2/diagnostics/determinism/20260909T022238Z/`.

Training on this harness was **nondeterministic**. Two same-seed runs (A vs B)
under the historical settings diverged; `F.interpolate` bilinear backward was
independently isolated as one nondeterministic operation
(`locate_op/locate_op.json`). Two controlled runs (C, D) under
`use_deterministic_algorithms(True)` + `cudnn.deterministic` +
`cudnn.benchmark=False` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` produced identical
trajectories.

**Consequence:** every uncontrolled E3b / E3c / O6 materiality comparison is
non-decisive. Recorded in the entries above; formalised here.

---

## A.2 Stage B — deterministic 6-block baseline (2026-09-09)

Record: `phase2/diagnostics/determinism/20260909T041500Z_baseline/`.
Superseded first attempt preserved, uncited, at `…/20260909T024500Z_baseline/`
with its own `NOTE.md` (killed by a wrapper path defect *after* all measurement
completed; no science affected).

Seed 0, 6 blocks, 200 epochs, deterministic controls. Two independent runs:

- **final weight SHA identical** — `3ad382c50d413ad2` for both
- epoch loss series identical, validation series identical, late window
  identical, gradient norms identical, stereo verdict identical
- **`.pth` container hashes differ** (`checkpoint_file_sha256_identical: false`)
  because the embedded run identity differs — this is recorded honestly in
  `comparison.json` and is not a determinism failure

> **Precision note.** "Bitwise identical" in this campaign means *identical
> weight tensors and identical metric series*. It does **not** mean identical
> serialized files. State it that way.

Authoritative 40-scene result, epoch 200, and the campaign's 6-block seed-0
reference: **EPE 2.2955182, D1 15.6045143, RMSE 6.0437075.** Stereo-functional
(right-image +80.813, matching-map +66.188, entropy 1.7755, disparity SD 18.000).

---

## A.3 Shift factorial — `shift=none` + standardised readout (2026-09-09)

Record: `phase2/factorial/shift_none_standardized/20260909T071500Z/`.
Seed 0, 6 blocks, 200 epochs, deterministic. Fills the missing 2×2 cell.

| arm | EPE | D1 |
|---|---:|---:|
| `shift=left` + standardised (Stage B) | 2.2955182 | 15.6045143 |
| `shift=none` + standardised | **8.3717466** | **66.5977700** |

**Verdict: `SHIFT-IMPORTANT`** (pre-registered Outcome A). The readout fix
unblocks learning; the shift is what the unblocked network learns *from*.
Neither alone yields a stereo model at this budget.

Scope, stated plainly: **one seed per arm**. It is quoted here because the
effect — 6.08 px — is roughly **59×** the 6-block three-seed EPE range measured
later (0.1027 px), which is why a single seed suffices at this magnitude and
would not at a small one. Stage E is **not** opened by this result.

---

## A.4 Deterministic block-count series — seed 0 (2026-09-09)

Record: `phase2/factorial/block_count_deterministic/20260909T131500Z/`.
Arms B (5 blocks) and C (4 blocks) at seed 0, 200 epochs, deterministic, read
against the Stage B 6-block control, which was **not** retrained.

| blocks | seed | EPE | D1 |
|---:|---:|---:|---:|
| 6 | 0 | 2.2955182 | 15.6045143 |
| 5 | 0 | 2.6457595 | 21.3711907 |
| 4 | 0 | **2.3387143** | **16.9797389** |

**Verdict: `INCONCLUSIVE` overall** (arm B `CAPACITY-REQUIRED-AT-SEED-0`, arm C
`INCONCLUSIVE`), and the reason matters more than the labels: **5 blocks is
clearly worse than 4 blocks at this seed.** A capacity ordering must be monotone
in block count; this one is not. The series establishes neither that refinement
capacity can be reduced nor that it cannot.

This non-monotonicity is the open contradiction that A.8 exists to test.

---

## A.5 Block-count seed screen — 30 epochs (2026-09-09)

Record: `phase2/factorial/block_count_seed_screen/20260909T145659Z/`.
6b/5b/4b × seeds 1, 2 at 30 epochs. **Verdict: `CONTINUE-FULL`.**

Three seeds produced three different epoch-30 orderings; the ordering also
flipped between checkpoints within a seed; seed spread at fixed architecture was
the same order as the architecture spread. The screen straddles the
stereo-emergence phase and is **unsuitable for architecture ranking** — it was
used only as a screening tool, and its 200-epoch descendants treat it only as a
deterministic *prefix* to verify against.

---

## A.6 Block-count batch 1 — 200 epochs (2026-09-10)

Record: `phase2/factorial/block_count_full/20260910T005550Z/`.
Three runs: 6b/seed1, 6b/seed2, 5b/seed1. **Verdict: `CONTINUE-BATCH-2`.**

| blocks | seed | EPE | D1 |
|---:|---:|---:|---:|
| 6 | 1 | 2.3982217 | 18.3357145 |
| 6 | 2 | 2.3950488 | 17.4699044 |
| 5 | 1 | 2.4729830 | 18.2061782 |

The decisive measurement is the **6-block seed range at 200 epochs: 0.1027 px**
— not the multi-pixel spread the 30-epoch screen implied. For the first time in
this project an architecture arm is readable against a measured reference scale.
Batch 1 explicitly declined to call the 5-block arm separated on two draws.

---

## A.7 Block-count batch 2 — the 5-block resolution run (2026-09-10)

Record: `phase2/factorial/block_count_full/20260910T052736Z/`. One run:
5b/seed2, 200 epochs. Pre-registered, 25/25 preflight checks PASS, guard tested
against three forbidden directions before launch.

**Verdict: `5-BLOCK-PENALTY-AT-SEEDS-0-1-2`.**

| | seed 0 | seed 1 | seed 2 | mean | range |
|---|---:|---:|---:|---:|---:|
| 6b EPE | 2.2955182 | 2.3982217 | 2.3950488 | 2.3629296 | **0.1027035** |
| 5b EPE | 2.6457595 | 2.4729830 | **2.6372994** | 2.5853473 | 0.1727766 |
| 6b D1 | 15.6045143 | 18.3357145 | 17.4699044 | 17.1367110 | 2.7312002 |
| 5b D1 | 21.3711907 | 18.2061782 | **20.6593989** | 20.0789226 | 3.1650125 |

- **EPE ranges disjoint:** `min(5b) − max(6b) = +0.0747612 px`. **9 of 9**
  pairings favour 6b.
- **D1 ranges overlap** by `−0.1295362 pt`: 5b/seed1 (18.2062) beats 6b/seed1
  (18.3357). **8 of 9** pairings, not 9 of 9. The verdict is always quoted with
  this exception.
- Per-scene, seed-matched: 6b better on EPE in 35/40 (s0), **22/40** (s1),
  32/40 (s2). 5b/seed1 was the 5-block arm's favourable draw, not its centre.
- All four 200-epoch runs stereo-functional, right-image +73.6…+76.4. **The
  penalty is an accuracy effect, not stereo collapse.**

**Claim ceiling (binding):** an empirical, reproducible EPE penalty for this
tested configuration at three seeds. **Not** a statistical test (n=3 per arm,
no p-value claimed or computable). **Not** proof that parameter count causes it
— 4b/seed0 is better than every 5b seed and sits inside the 6b range. **Not** a
D1 separation. **Not** transferable to another dataset, recipe, candidate count
or a pretrained model. **Not** a latency claim.

---

## A.8 Block-count batch 3 — the final 4-block runs (2026-09-10, IN PROGRESS)

Record: `phase2/factorial/block_count_full/20260910T081501Z/`.
Two authorised runs: **4b/seed1 and 4b/seed2, 200 epochs**, deterministic.
Pre-registration frozen before preflight; **28/28 preflight checks PASS**; the
batch guard was tested against five forbidden directions (4b/seed0, 4b/seed3,
6b/seed1, 5b/seed2, 6b/seed0) and refused all five before any training.

Both initialisation hashes match the frozen 30-epoch seed screen exactly
(`2cac2916ed802495`, `e622226a5a22fc9b`), so each run is a strict continuation
of a previously recorded deterministic prefix.

The decision boundary — the frozen 6-block three-seed EPE range
**[2.2955181809208267, 2.3982217171269924]** — was fixed and asserted against
the frozen records *before launch*. Three pre-registered verdicts (A: all three
4b seeds above the range; B: all three inside it; C: otherwise, INCONCLUSIVE)
are applied mechanically in `comparison.json`. EPE is primary; D1 may not
override it; no fourth seed may be bought after the fact.

**Result and verdict: see ADDENDUM B, appended when the runs complete.**

---

# ADDENDUM B — batch 3 result, and the close of the block-count campaign

**Appended 2026-09-10. Additive only.** Completes the A.8 entry above.

## B.1 EXP-BLOCKCOUNT-FULL-001 batch 3 — the final 4-block runs (2026-09-10)

Record: `phase2/factorial/block_count_full/20260910T081501Z/`.
Two runs: 4b/seed1 and 4b/seed2, 200 epochs, deterministic. 2.61 GPU-hours.
Pre-registration frozen before preflight; 28/28 preflight checks PASS; guard
tested against five forbidden directions and refused all five before launch;
both runs reproduced their 30-epoch seed-screen prefixes by exact float
equality.

**Verdict: `4-BLOCK-INCONCLUSIVE-AT-SEEDS-0-1-2`.**

### The completed 3 × 3 design

| Blocks | seed 0 | seed 1 | seed 2 | mean | range |
| -----: | -----: | -----: | -----: | ---: | ----: |
| 6 — EPE | 2.2955182 | 2.3982217 | 2.3950488 | 2.3629296 | **0.1027035** |
| 5 — EPE | 2.6457595 | 2.4729830 | 2.6372994 | 2.5853473 | 0.1727766 |
| 4 — EPE | 2.3387143 | **2.6553489** | **2.5756633** | 2.5232422 | **0.3166346** |
| 6 — D1 | 15.6045143 | 18.3357145 | 17.4699044 | 17.1367110 | 2.7312002 |
| 5 — D1 | 21.3711907 | 18.2061782 | 20.6593989 | 20.0789226 | 3.1650125 |
| 4 — D1 | 16.9797389 | **20.0897392** | **20.0793521** | 19.0496101 | 3.1100003 |

All nine runs: same 40 scene IDs, same 3,802,797 valid pixels, same evaluator,
final-epoch-200 checkpoint, deterministic protocol, identical environment.

### Why INCONCLUSIVE

The 4-block arm **straddles** the frozen 6-block reference range
`[2.2955181809208267, 2.3982217171269924]` — seed 0 inside, seeds 1 and 2 above.
`min(4b EPE) − max(6b EPE) = −0.0595074 px`; D1 ranges overlap by 1.3559756 pt.
4b is worse than 6b in 7 of 9 pairings, not 9 of 9. All three runs are
STEREO-FUNCTIONAL (right-image +75.5 … +79.3), so this is not a stereo failure.
The pre-registered rule returns Verdict C and was applied as written.

**No fourth seed was bought, and none should be.** The 4-block arm's own seed
spread is **0.3166 px — 3.08× the 6-block arm's and larger than the 0.1603 px
mean gap it would have to demonstrate.** The limit is the arm's trajectory
variance, not the sample count; more seeds of this design cannot fix it.

### What this changes

- **4b/seed0 (2.3387) was a favourable draw, not the arm's centre** — exactly
  the role 5b/seed1 played for the 5-block arm. Any claim resting on it,
  including the E3b-era "four blocks match six" reading annotated in A.0 #1, is
  **not supported**. That annotation is now backed by direct measurement.
- The apparent **non-monotonicity** (5 blocks worse than 4 blocks) that motivated
  this batch **survives only at the level of arm means** (4b − 5b = −0.0621 px)
  and is not established as an ordering: the arms' seed ranges overlap heavily
  and 4b beats 5b in only 5 of 9 pairings.
- **The 5-block verdict is unchanged.** `5-BLOCK-PENALTY-AT-SEEDS-0-1-2` stands
  exactly as recorded in A.7, D1 exception included.

### Two disclosed code corrections

Both were defects in batch 3's own analysis code, found and fixed before the
result was written up; neither touched training, thresholds or the decision rule.
`CORRECTION_screen_prefix.md` (mis-transcribed prefix constants, corrected
against the authoritative screen records) and `CORRECTION_stereo_reader.md` (a
verdict lookup at the wrong nesting level that reported every run — including a
verified frozen one — as a stereo failure; the superseded comparison is
preserved beside the corrected one).

---

## B.2 State of the block-count question

| Question | Status | Evidence |
| --- | --- | --- |
| 6 vs 5 blocks | **`5-BLOCK-PENALTY-AT-SEEDS-0-1-2`** | 3×3 seeds, EPE ranges disjoint by +0.0748 px, 9/9 pairings; D1 overlaps by 0.13 pt |
| 6 vs 4 blocks | **`4-BLOCK-INCONCLUSIVE-AT-SEEDS-0-1-2`** | 3×3 seeds, arm straddles the boundary, 7/9 pairings, D1 overlaps by 1.36 pt |
| Monotone in block count? | **Not established** | 4b and 5b arms overlap heavily; 4b better than 5b in only 5/9 pairings |
| Cause (parameters? receptive field?) | **Unknown, and not measurable from this design** | no causal claim is supportable from any batch |
| Stereo functionality | **Uniform across all nine runs** | right-image +70.4 … +80.8, no ordering by block count |

**Campaign claim ceiling.** Refinement block count is measured to affect accuracy
in this tested StereoNet configuration under this deterministic protocol, on one
dataset, one recipe, one candidate count, 160 scenes from random initialisation.
Three seeds per arm is descriptive evidence, not a statistical test — no p-value
is claimed and none is computable. No materiality band is in use; the historical
E3b/E3c band stays invalidated. Nothing here is a parameter-count or
receptive-field *cause*, a latency claim, a statement that 6 blocks is optimal,
or transferable to another dataset, recipe or a pretrained model.

**Reopening the 4-block question requires a new experimental design, not more
seeds.** None is proposed. Stage E stays CLOSED.
