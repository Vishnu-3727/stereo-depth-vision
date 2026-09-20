# PHASE-2 HYPOTHESIS 01 — `EXP-P2A-SCALE-COVERAGE-001`

**SPECIFICATION ONLY. NO TRAINING HAS BEEN RUN. Execution requires explicit
approval.**

Control: frozen ARM-V (`phase1/runs/arm_v`, `arm_v_s1`, `arm_v_s2`), not
retrained. Diagnostic basis:
`phase2/docs/PHASE2_ARMV_HIGH_DISPARITY_DIAGNOSTIC.md` and
`phase2/diagnostics/arm_v_high_disparity/`.

---

## 1. Observed mechanism

ARM-V is an accurate, essentially unbiased disparity estimator over the
disparity band its training data covers, and loses dependence on the true
disparity outside it.

- GT < 64 px (95.386 % of contract pixels): pred = 0.99·GT + b, R² = 0.96,
  bias < 0.5 px, EPE 1.17–1.54 px — at parity with the frozen reference.
- GT ≥ 64 px (4.614 %): slope collapses to 0.14 / 0.53 / 0.12 with
  R² 0.046 / 0.480 / 0.033; from GT ≥ 80 px the signed error equals the EPE,
  i.e. essentially every pixel is under-predicted.
- Maximum prediction 91.5 / 114.0 / 101.7 px against GT max 153.0 px.

The band is set by training supervision, not by the graph:

- GT ≥ 96 px is **0.228 %** of training pixels against **0.898 %** of
  evaluation pixels (3.9× short); GT ≥ 112 px is 0.034 % against 0.339 %
  (10× short).
- The collapse is present **on the training split itself** (train
  GT [112,128) EPE 36.9 / 9.9 / 34.5 px).
- **16 of 16** frozen Phase-1 checkpoints collapse — ARM-U, ARM-V, ARM-W,
  ARM-X, ARM-Y, ARM-Z, three seeds each. High-stratum slope over all 16:
  mean +0.094, s.d. 0.217, range −0.425…+0.410. None track.
- The frozen reference ONNX — the **same graph**, with a proven no-op
  cost-volume shift, i.e. performing no disparity search at all — has slope
  **+0.727** and reaches 135.7 px. The graph is therefore not the cause.

---

## 2. Evidence

All grade **A** unless marked, all reproduced on 3/3 ARM-V seeds. Machine data:
`results.json`, `conditional.json`, `collapse_census.json`,
`scale_transfer.json`, `apparatus_and_reference.json`, `coverage_choice.json`.

**The decisive measurement — test-time scale transfer.** The same evaluation
pixels, the same frozen weights, the stereo pair downscaled by s before
inference and the prediction divided by s afterwards:

| seed | metric | s = 1.00 | s = 0.75 | s = 0.50 |
|---|---|---|---|---|
| 0 | EPE GT≥96 | 39.287 | 21.102 | **10.879** |
| 0 | signed GT≥96 | −39.29 | −20.73 | **−4.24** |
| 0 | slope_hi | +0.295 | +0.472 | **+1.010** |
| 0 | max prediction | 91.5 | 126.7 | **152.7** |
| 1 | EPE GT≥96 | 21.708 | 12.605 | **8.475** |
| 1 | slope_hi | +0.239 | +0.537 | **+1.253** |
| 2 | EPE GT≥96 | 38.935 | 20.680 | **10.851** |
| 2 | slope_hi | +0.075 | +0.219 | **+0.775** |

Downscaling by 2 halves the spatial resolution and coarsens the effective
candidate spacing from 8 px to 16 px of original image. It cannot improve
stereo matching. It only changes which disparity values are presented. It
nevertheless cuts the high-disparity error by 2.6–3.6× and restores slope ≈ 1
on every seed. **The failure is a function of the disparity value presented,
not of those pixels' content or matching difficulty.**

Supporting measurements:

- The raw cost volume still carries a genuine response at the true candidate in
  the high stratum: standardised cost −1.458 / −1.434 / −1.465, negative for
  93 % of high-disparity pixels (low stratum: −2.03, 98 %).
- The ceiling exists **before** refinement: an affine index→pixel map
  calibrated on GT < 64 px and extrapolated upward reproduces the ceiling
  (max 91.4 vs the model's 91.5) and is *worse* than the model
  (EPE 15.50 / 10.34 / 15.20 vs 13.74 / 8.08 / 13.44). Refinement mildly helps.
- Within GT [96,112), error is flat across raw-evidence quartiles on seeds 0
  and 2 (34.0 / 36.0 / 35.1 / 35.4 and 33.4 / 34.1 / 33.6 / 32.3) — decoupled
  from matching evidence.

---

## 3. Competing explanations ruled out

| class | ruled out by |
|---|---|
| readout calibration / scale compression | affine recalibration extrapolates *worse* than the model; within GT≥64 the readout index has R² 0.02–0.41 against GT, so there is no signal left to rescale; the readout is bit-identical at s = 0.5 where the stratum is fixed |
| cost-distribution saturation | identical distribution machinery at s = 0.5 yields slope +0.78…+1.25 on the same pixels; the reference uses the same soft-argmin and reaches +0.727 |
| high-disparity matching failure | halving the resolution — strictly less matching information — fixes the stratum; the raw volume retains a true-candidate response; error is evidence-independent at GT≥96 on 2 of 3 seeds |
| refinement suppression | the ceiling is already present pre-refinement; refinement improves the stratum on 3/3 seeds |
| model capacity | Phase-1 measurement: train-split EPE 1.03–1.14 px, below the 1.3134 px reference |
| architecture / cost-volume design | the reference shares the graph and a strictly weaker cost volume and does not collapse |

**Residual, stated not hidden:** at GT [64,80) training coverage already
*matches* evaluation coverage (2.957 % vs 2.964 %) and the model still fails
there (EPE 4.47–4.66), and that bin's error *is* evidence-dependent (EPE spans
1.93→8.67 across raw-evidence quartiles). So a second, smaller mechanism —
matching-evidence degradation — is active in [64,80) and this experiment does
not isolate it [Grade C]. The scale-transfer probe improves that bin too
(7.57 → 3.90 px at s = 0.5), so the disparity-value effect dominates there as
well, but the two are not separated.

---

## 4. Exact intervention — exactly one variable

**Random scale augmentation of the training crop.** Training-time only.

Current ARM-V sampling (`phase1/scripts/train_arm_v.py`,
`CroppedKitti.__getitem__`): uniform random 256×512 crop of the 375×1242 pair,
then ImageNet normalisation, then independent per-image gain jitter σ = 0.1.

Replacement, inserted **before** normalisation, with the normalisation and
jitter steps and their order untouched:

```
s  ~ logUniform(0.7, 1.7)                      # one draw per sample
w  = round(512 / s) ;  h = round(256 / s)      # source window, aspect preserved
(y, x) ~ uniform over valid top-left positions in the 375x1242 image
left, right  = bilinear resize of the (h, w) crop to (256, 512)
disparity    = NEAREST resize of the (h, w) disparity crop to (256, 512)
sx = 512 / w                                   # actual horizontal factor after rounding
disparity[disparity > 0] *= sx                 # disparity scales with the HORIZONTAL factor only
```

Frozen, unchanged from ARM-V: architecture
(`downsample_levels=3, num_disparities=24, cost_volume_shift="right",
regression_normalize=True`, 397,954 parameters, no BatchNorm); 200 epochs;
batch 2; Adam(0.9, 0.999) lr 1e-3; cosine annealing to 0; masked smooth-L1
β = 1.0 with valid = `gt > 0 and gt < 184`; gain jitter σ = 0.1; no horizontal
flip; ImageNet normalisation; fp32; split hailo_calib 0–159 train /
hailo_val 160–199 eval; best+final checkpoint policy using the train-time
10-scene monitor only, never the frozen score; seeds 0, 1, 2, all run to
completion before any number is read.

Implementation notes that must be honoured, not reinterpreted:

- **NEAREST** for disparity. KITTI GT is sparse (≈21 % dense) and 0 means "no
  ground truth"; bilinear would blend zeros into valid disparities.
- Disparity scales with **`sx`**, the realised horizontal factor after integer
  rounding, not with the drawn `s`. Rounding makes `sx` and `sy` differ by
  under 1 %; that anisotropy is accepted and recorded, not corrected.
- At s = 1.7 the source window is 151×301 and at s = 0.7 it is 366×731; both
  fit inside 375×1242, so no padding path is introduced.
- Scaled disparities ≥ 184 px fall outside the existing loss mask and are
  dropped by it. Measured share: 0.004 % of augmented training pixels. No new
  masking logic is added.
- **No scaling at evaluation.** The frozen contract is untouched.

### Why this range, and not a tuned one

The range was fixed by a measured, pre-registered rule
(`coverage_choice.py`, `coverage_choice.json`), not by trying values:

- **C1 (coverage):** the augmented training pixel fraction must be at least the
  evaluation pixel fraction in every *measured-failure* bin — [64,80), [80,96),
  [96,112), [112,128), [128,144).
- **C2 (minimum disturbance):** among the candidates passing C1, take the one
  with the smallest total-variation distance to the ARM-V control distribution,
  i.e. the narrowest range that fixes the measured shortfall.

| candidate | passes C1 | TV vs control | coverage ratio vs eval at [96,112) / [112,128) / [128,144) |
|---|---|---|---|
| s = 1.0 (control) | no | 0.000 | 0.35 / 0.10 / 0.07 |
| s ~ U(1.0, 2.0) | yes | 0.255 | 7.86 / 5.32 / 20.93 |
| **s ~ logU(0.7, 1.7)** | **yes** | **0.091** | **1.93 / 1.10 / 4.69** |
| s ~ logU(0.5, 2.0) | yes | 0.132 | 3.31 / 2.19 / 8.69 |

Two earlier versions of this rule were ill-posed and are kept in the script
beside their refutation rather than quietly replaced: (i) "augmented ≥ eval in
*every* bin" is unsatisfiable by construction, since both sides are probability
vectors over the same bins; (ii) defining the deficient set by histogram
shortfall pulls in [32,48) and [48,64), which are the model's *best* bins. The
deficient set must be defined by the measured failure.

`s` is **not** a tunable knob in this experiment. It is fixed at
logU(0.7, 1.7) before training and will not be changed, swept, or re-chosen
after seeing any result.

### Why it targets the measured failure

The measurement says the model is competent exactly over the disparity band its
supervision covers, and that presenting a high-disparity pixel at a lower
disparity fixes it. Scale augmentation is the minimal in-domain instrument that
moves the *supervision* band instead of the *input*: a 60 px training pixel
becomes a 102 px training pixel at s = 1.7, from the same sensor, the same
scenes, the same rectification, with no new data and no graph change.

### Disjointness from every closed arm

ARM-G was **photometric** augmentation (brightness/contrast/gamma), not
geometric — a different variable, and no photometric parameter is touched here.
ARM-D was **out-of-domain** synthetic pretraining (Driving, 4,400 pairs); this
introduces no new corpus and no pretraining stage. ARM-W (channel grouping),
ARM-X (deep supervision), ARM-Y (post-volume gate) and ARM-Z (feature L2
normalisation) are all graph or loss changes; the graph and the loss are
bit-identical to ARM-V here. Nothing is stacked, and no closed arm's verdict is
reopened, reinterpreted or reused as a component.

---

## 5. Parameter, compute and deployment impact

| | value |
|---|---|
| Trainable parameters | **397,954 — unchanged, delta 0** |
| Graph / operators / tensor shapes | **unchanged** |
| Inference compute | **unchanged** (augmentation is training-only) |
| ONNX export / Hailo compatibility | **unchanged** — no new operator, no new tensor rank |
| Training compute | one bilinear + one nearest resize per sample; the network still sees 256×512. Expected wall ≈ ARM-V's 3,660 s per seed, ~3 h for three seeds |
| Memory | unchanged |

A pre-training check will assert parameter count 397,954 and a bit-identical
`state_dict` key set before any optimizer step, as every previous arm did.

---

## 6. ARM-V control

The three frozen ARM-V checkpoints, **not retrained**:
1.8903392 / 1.5493088 / 1.8785827, mean 1.7727436, spread 0.3410304, and their
already-measured stratum metrics (GT≥96 EPE 39.287 / 21.708 / 38.935;
slope_hi +0.295 / +0.239 / +0.075; max prediction 91.5 / 114.0 / 101.7).

---

## 7. Evaluation protocol

The frozen global contract, unchanged and unmodified: KITTI 2015 `_10` frames,
hailo_val scenes 160–199, `disp_occ_0`, fixed 368×1232 top-left crop after
bottom/right padding, no resize, GT scale 1/256, valid = gt > 0, pooled-pixel
evaluation, exactly 3,802,797 valid pixels, `contract_match true`. No scaling is
applied at evaluation.

Reported for every seed, as the kickoff requires:

global EPE · global D1 · GT≥64 EPE · GT≥64 signed error · GT≥64 pixel fraction ·
per-bin EPE on the ten Phase-2 bins · maximum predicted disparity (both
valid-pixel and all-pixel) · training-split EPE · validation-split EPE ·
slope and R² of prediction on GT in the high stratum · parameter count ·
checkpoint SHA-256 · contract guard.

Produced by re-running `phase2/diagnostics/arm_v_high_disparity/` unchanged
against the new checkpoints.

---

## 8. Success / failure rule — pre-registered, fixed before execution

**(a) Mechanism check — did the diagnosed ceiling actually move?**

Instrument: `slope_hi`, the least-squares slope of prediction on GT over the
GT ≥ 96 px pixels. Justification for using it rather than stratum EPE: measured
3-seed relative spread of EPE(GT≥96) is *larger* than that of global EPE in 4
of 5 arms (ARM-V 0.528 vs 0.192), so stratum EPE is not a low-noise instrument;
whereas all 16 trained runs in this project lie in slope_hi ∈ [−0.425, +0.410]
with s.d. 0.217, and the reference sits at +0.727.

- **MECHANISM CONFIRMED** if 3-seed mean `slope_hi` ≥ **0.60** and every
  individual seed ≥ **0.45**.
- **MECHANISM NOT CONFIRMED** otherwise.

**(b) Accuracy rule — is the deliverable better?**

Inherited ARM-V-anchored gate, unchanged from the one already used for ARM-Z
and ARM-X, not invented for this experiment:

```
3-seed mean global EPE <  1.4317132              -> ACCEPTED   (= ARM-V mean - ARM-V spread)
1.4317132 <= mean EPE <  1.7727436               -> INCONCLUSIVE
mean EPE >= 1.7727436                            -> REFUTED
```

**(c) Reported cost, not a gate.** GT < 64 px 3-seed mean EPE against ARM-V's
1.2897 px. Any degradation is reported in full, including when (a) confirms and
(b) does not.

The four outcomes are reported as measured; (a) and (b) are independent and
neither is allowed to override the other. A CONFIRMED mechanism with an
INCONCLUSIVE accuracy result is a real and publishable outcome: it would mean
the diagnosis was right and the remedy insufficient.

---

## 9. Stop conditions

1. **Exactly one intervention.** The scale augmentation above and nothing else.
2. **Exactly three seeds: 0, 1, 2.** All run to completion before any result is
   read. No seed dropped, replaced, re-run or added afterwards.
3. **No tuning.** `s` is fixed at logU(0.7, 1.7). No sweep of the range, the
   distribution, the resize interpolation, the loss, the schedule or the epochs.
4. **No stacking.** No closed mechanism is combined with this one.
5. **No rescue experiment.** If the accuracy rule returns INCONCLUSIVE or
   REFUTED, ARM-V is frozen as the Phase-2 accuracy model and Phase 2 proceeds
   directly to export, deployment validation, deployed-accuracy measurement and
   final artifacts.
6. **Abort before training** if the pre-training check shows a parameter-count
   delta ≠ 0, a changed `state_dict` key set, any BatchNorm, or any change to
   the evaluation contract.
7. **Abort mid-run** if a seed fails to complete; the run is recorded as failed
   and is not silently replaced.
8. **Phase 2 is the last phase.** No Phase 3 is created under any outcome.

---

## 10. What this experiment does not claim

It does not claim that scale augmentation is the only remedy, that it will
reach the frozen reference (1.3134471 px is a reference, not a pass mark), or
that the GT [64,80) evidence-dependent component is addressed. It does not
claim the diagnostic counterfactual of 1.4644 px as an expected result — that
figure is arithmetic on ARM-V's own measured strata and is not a prediction.

Alternatives considered and set aside, with reasons:

- **Multi-scale inference** (run at s = 1.0 and s = 0.5, fuse). Already
  *measured* to work on frozen weights, which is why it is credible — but it
  doubles inference compute, needs a fusion rule that is a new free parameter,
  and changes the deployed graph. Recorded as a fallback if training fails.
- **Full SceneFlow pretraining.** Not on disk (`PRETRAIN_RESOURCE_AUDIT`: zero
  PFM files), and the only locally available proxy was ARM-D, refuted.
- **Disparity-weighted loss.** No measured misalignment between the loss and
  the metric, and it carries a free weighting parameter this budget forbids
  fitting.
