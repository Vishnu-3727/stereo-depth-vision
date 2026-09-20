# EXP-P2A-SCALE-COVERAGE-001 — RESULT

Preregistration: `phase2/docs/PHASE2_HYPOTHESIS_01.md` (frozen before any P2A
code existed). Diagnostic basis:
`phase2/docs/PHASE2_ARMV_HIGH_DISPARITY_DIAGNOSTIC.md`. Self-contained record:
`phase2/runs/p2a_scale_coverage/P2A_EXPERIMENT_RECORD.json`. Machine decisions:
`P2A_DECISIONS.json`. Scorer: `phase2/scripts/eval_p2a.py`. Trainer:
`phase2/scripts/train_p2a_scale_coverage.py`.

All three seeds ran to completion under the identical procedure before any
frozen number was read. No seed was dropped, replaced, re-run or selected on.
No threshold was changed. `phase1/results/LEADERBOARD.md` was not edited.

---

## 1. Executive Status

Both preregistered gates were missed, each by a small margin, and both verdicts
stand exactly as written.

| gate | threshold | measured | verdict |
|---|---|---|---|
| Accuracy | mean EPE < 1.4317132 | **1.4409826** | **INCONCLUSIVE** (short by 0.0092694 px) |
| Mechanism | mean slope_hi ≥ 0.60 **and** every seed ≥ 0.45 | mean **+0.5701**, min **+0.5374** | **MECHANISM NOT CONFIRMED** (mean short by 0.0299; the per-seed condition passed) |

What was nonetheless measured, under the unchanged frozen contract:

- Mean global EPE moved **1.7727436 → 1.4409826**, i.e. **−0.3317610 px**.
- Every P2A seed (1.4150 / 1.4486 / 1.4594) is better than every ARM-V seed
  (1.8903 / 1.5493 / 1.8786).
- Seed spread collapsed **0.3410304 → 0.0444151**, 7.7× tighter.
- The gap to the frozen reference narrowed **+0.4592965 → +0.1275355 px**.
- GT ≥ 96 px EPE fell from 33.310 (ARM-V mean) to **12.415**; signed bias from
  −33.3 to −11.7; maximum prediction rose from 91.5–113.9 px to
  **130.6–142.9 px** against a GT maximum of 153.0 px.
- The low-disparity stratum did **not** pay for it: GT < 64 px EPE
  **improved**, 1.2897 → 1.2617.

The intervention therefore produced a large, reproducible, one-directional
improvement while failing the two thresholds set in advance. Those are separate
facts and neither cancels the other.

---

## 2. Preregistered Hypothesis

ARM-V loses dependence on true disparity above ~64 px because the
high-disparity band is under-covered by training supervision (GT ≥ 96 px is
0.228 % of training pixels against 0.898 % of evaluation pixels). The decisive
frozen-weight evidence was the test-time scale transfer: presenting the same
evaluation pixels downscaled by 0.5 — halving their disparity at the cost of
half the spatial resolution — cut GT ≥ 96 px EPE by 2.6–3.6× and restored
slope ≈ 1 on all three seeds. The experiment tests whether widening the
*supervision* band by in-domain scale augmentation removes the learned ceiling.

Stated in the preregistration and repeated here: this was a hypothesis test,
not an expectation of success.

---

## 3. Exact Intervention

One variable. Training-time only. Zero parameter change, zero graph change.

```
s  ~ logUniform(0.7, 1.7)                 one draw per sample
w  = round(512 / s) ;  h = round(256 / s)
(y, x) uniform over valid top-left positions in the source frame
left/right : cv2.INTER_LINEAR  resize of the (h, w) crop -> 256x512
disparity  : cv2.INTER_NEAREST resize of the (h, w) crop -> 256x512
sx = 512 / w                              REALISED horizontal factor
disparity[disparity > 0] *= sx
```

Order of operations is ARM-V's: crop → resize → ImageNet normalise → gain
jitter. Everything else is ARM-V unchanged: `downsample_levels=3`,
`num_disparities=24`, `cost_volume_shift="right"`,
`regression_normalize=True`, 200 epochs, batch 2, Adam(0.9, 0.999) lr 1e-3,
cosine to 0, masked smooth-L1 β = 1.0 with valid = `gt > 0 and gt < 184`, gain
jitter σ = 0.1, no horizontal flip, fp32, hailo_calib 0–159 train /
hailo_val 160–199 eval, best+final checkpoints chosen by the train-time
10-scene monitor only.

Realised draws, sampled during training and recorded: s ∈ [0.7000, 0.6999…1.6999],
sx ∈ [0.7004, 1.7010], windows h ∈ [151, 366], w ∈ [301, 731].

**One deviation from the prompt's wording, recorded not hidden.** The source
frame is the dataset's `pad_and_crop` **368×1232** frame — the same source
ARM-V crops from — not the raw 375×1242. Both scale extremes still fit
(366×731 and 151×301), so the preregistered promise that no padding path is
introduced holds. Logged in every seed's
`p2a_record.json → config.intervention.source_frame`.

---

## 4. Integrity Checks

The preregistered pre-training guard ran before any optimizer step on every
seed and passed on every item (`integrity_guard.json` per run, embedded in the
record):

| check | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| parameter count == 397,954 | ✔ | ✔ | ✔ |
| state_dict key set identical to ARM-V (70 keys) | ✔ | ✔ | ✔ |
| state_dict tensor shapes identical to ARM-V | ✔ | ✔ | ✔ |
| no BatchNorm module | ✔ | ✔ | ✔ |
| config matches ARM-V, max_disparity 184, stride 8 | ✔ | ✔ | ✔ |
| evaluation contract unchanged | ✔ | ✔ | ✔ |

At evaluation: strict load `strict_ok true`, 70 keys matched, 397,954 params,
`contract_match true`, **3,802,797** valid pixels on all three seeds; training
split 14,216,313 pixels. Checkpoint SHA-256 (best):
`0868ffd137a9985306bf5563…` / `41b94aaaf14047c92a18817b…` /
`6ce4d9bd69efc0ba02a2d583…`. git head `58e8a19908dd` on all three.
Software: python 3.12.9, torch 2.7.0+cu128, numpy 2.5.1, opencv 5.0.0,
CUDA 12.8, RTX 4060 Laptop, Windows-11-10.0.26200.

**Execution history, recorded because it happened.** Two agent-harness
background shells were killed by the host for low system memory (15.1 GB total,
≈1.2 GB free, browsers dominant). Neither kill touched a training process — the
detached python child survived both times and seed 1 trained through
uninterrupted; this was verified by watching the log advance before anything was
done. Seed 2 and the evaluation were then run from a launcher detached from the
harness (`phase2/scripts/run_p2a_remaining.ps1`). **No seed was restarted,
dropped, replaced or re-run; no result was read before all three seeds
completed; nothing about the experiment changed.**

---

## 5. Training Results

| seed | wall clock | best epoch | best 10-scene monitor EPE |
|---|---|---|---|
| 0 | 3,829.4 s | 180 | 1.9771 |
| 1 | 3,865.2 s | 170 | 2.0248 |
| 2 | 3,708.5 s | 180 | 2.0330 |

Total ≈ 3.2 h. Comparable to ARM-V's 3,659 s per seed: the resize costs
essentially nothing.

The 10-scene monitor best is 1.977 / 2.025 / 2.033 against ARM-V's 3.236 /
2.272 / 3.234. It is quoted only for completeness: the monitor is a 10-scene
train-time curve on a different pixel population, it exists solely to select a
checkpoint, it is not comparable to the frozen contract, and no decision in
this report uses it.

Training-split EPE (diagnostic, **not** a frozen-contract score):

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| P2A train EPE | 1.1799 | 1.1328 | 1.1730 | 1.1619 |
| ARM-V train EPE | 1.1364 | 1.0350 | 1.1235 | 1.0983 |

P2A fits the training set slightly *less* well and generalises substantially
better — the generalisation gap narrows from 0.674 px (ARM-V) to 0.279 px.

---

## 6. Frozen Evaluation Results

Contract unchanged; no evaluation-time augmentation; no multi-scale inference.

| seed | EPE | D1 % | RMSE | bad1 % | bad2 % | bad3 % |
|---|---|---|---|---|---|---|
| 0 | 1.4149796 | 8.4195 | 3.5346 | 35.1865 | 14.9269 | 8.7526 |
| 1 | 1.4485736 | 8.5056 | 3.7749 | 34.9602 | 15.1205 | 8.8426 |
| 2 | 1.4593946 | 8.8306 | 3.6893 | 35.6651 | 15.5468 | 9.1855 |
| **mean** | **1.4409826** | **8.5852** | 3.6663 | 35.2706 | 15.1981 | 8.9269 |

Control (ARM-V, frozen, not retrained): 1.8903392 / 1.5493088 / 1.8785827,
mean 1.7727436, D1 mean 10.269 %. Frozen reference: 1.3134471, D1 8.1544 %.

- Δ vs ARM-V mean: **−0.3317610 px**, **−1.684 pp D1**.
- Gap to reference: **+0.1275355 px** (was +0.4592965).
- Spread: **0.0444151** (ARM-V 0.3410304).

Per-bin EPE and signed error:

| GT bin | px | P2A s0 / s1 / s2 | P2A signed s0 / s1 / s2 | ARM-V s0 / s1 / s2 |
|---|---|---|---|---|
| [0,16) | 527,871 | 1.340 / 1.298 / 1.327 | +0.44 / +0.47 / +0.47 | 1.257 / 1.216 / 1.298 |
| [16,32) | 1,166,214 | 1.238 / 1.253 / 1.264 | +0.11 / +0.17 / +0.08 | 1.185 / 1.168 / 1.214 |
| [32,48) | 1,118,114 | 1.180 / 1.230 / 1.256 | −0.24 / −0.07 / −0.21 | 1.325 / 1.236 / 1.314 |
| [48,64) | 815,126 | 1.259 / 1.287 / 1.328 | −0.39 / −0.18 / −0.30 | 1.535 / 1.334 / 1.490 |
| [64,80) | 112,726 | **3.060 / 2.886 / 2.960** | −1.59 / −1.12 / −1.38 | 4.664 / 3.637 / 4.469 |
| [80,96) | 28,594 | **4.861 / 5.283 / 5.033** | −3.32 / −3.84 / −3.42 | 19.009 / 9.338 / 18.377 |
| [96,112) | 21,249 | **9.596 / 10.121 / 9.404** | −8.72 / −9.47 / −8.28 | 35.120 / 16.658 / 33.361 |
| [112,128) | 11,823 | **13.535 / 17.970 / 14.598** | −13.29 / −17.43 / −14.16 | 45.483 / 28.064 / 47.196 |
| [128,144) | 1,071 | **33.785 / 31.054 / 35.282** | −33.78 / −31.05 / −35.28 | 53.353 / 51.424 / 58.088 |
| [144,160) | 9 | INSUFFICIENT (<1,000 px) | — | INSUFFICIENT |

Maximum predicted disparity:

| | s0 | s1 | s2 | ARM-V | reference | GT max |
|---|---|---|---|---|---|---|
| over valid GT pixels | 130.60 | 142.86 | 135.41 | 91.5 / 114.0 / 101.7 | 135.73 | 153.04 |
| over all pixels | 157.75 | 224.88 | 159.70 | 99.5 / 151.2 / 101.7 | 136.29 | — |

---

## 7. High-Disparity Mechanism Results

| metric | P2A s0 / s1 / s2 | P2A mean | ARM-V mean |
|---|---|---|---|
| EPE GT ≥ 64 | 5.040 / 5.342 / 5.061 | **5.148** | 11.755 |
| signed error GT ≥ 64 | −3.72 / −3.86 / −3.62 | **−3.73** | −10.99 |
| GT ≥ 64 pixel fraction | 0.04614 (all seeds) | 0.04614 | 0.04614 |
| slope GT ≥ 64 | 0.772 / 0.719 / 0.763 | 0.751 | — |
| R² GT ≥ 64 | 0.684 / 0.584 / 0.638 | 0.635 | 0.046 / 0.480 / 0.033 |
| EPE GT ≥ 96 | 11.726 / 13.500 / 12.019 | **12.415** | 33.310 |
| signed error GT ≥ 96 | −11.09 / −12.91 / −11.17 | −11.72 | −33.3 |
| **slope_hi (GT ≥ 96)** | **+0.6227 / +0.5374 / +0.5500** | **+0.5701** | +0.2029 |
| R²_hi | 0.1326 / 0.0696 / 0.0840 | 0.0954 | — |

Training-split high stratum (diagnostic): GT ≥ 96 EPE 4.509 / 4.607 / 6.521,
signed −1.56 / +1.29 / +0.75, slope 0.5716 / 0.6753 / 0.6076. On data the model
has seen, the high-disparity bias is now near zero — two of three seeds
*over*-predict slightly — where ARM-V still under-predicted by 23–37 px.

The mechanism moved substantially and in the predicted direction on every seed.
It did not move far enough to clear the preregistered mean gate.

---

## 8. Low-Disparity Cost

The protection metric. ARM-V reference 1.2897 px.

| | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| P2A GT < 64 EPE | 1.2396 | 1.2602 | 1.2852 | **1.2617** |

**There is no low-disparity cost: the stratum improved by 0.0280 px.** The
preregistration named this as the principal risk, and the risk did not
materialise. Within it, the two smallest bins degrade slightly
([0,16) +0.06, [16,32) +0.06) while [32,48) and [48,64) improve markedly
(−0.09 and −0.25), and the net is an improvement.

---

## 9. Mechanism Decision

### MECHANISM NOT CONFIRMED

Rule, applied exactly as written and with no reinterpretation:

```
MECHANISM CONFIRMED  iff  mean slope_hi >= 0.60  AND  every seed slope_hi >= 0.45
```

- mean slope_hi = (0.6227 + 0.5374 + 0.5500) / 3 = **0.5701** — below 0.60 by
  **0.0299**. The mean condition FAILS.
- min seed slope_hi = 0.5374 ≥ 0.45. The per-seed condition passes.

Both conditions are required, so the verdict is **MECHANISM NOT CONFIRMED**.

Recorded alongside, as measurement rather than as mitigation: slope_hi rose
from an ARM-V mean of +0.2029 to +0.5701, and all three P2A seeds now sit
above the entire population of the 16 previously trained checkpoints in this
project (which spanned −0.425…+0.410). The frozen reference is +0.727. The gate
was not met; the quantity it measures moved a long way.

---

## 10. Accuracy Decision

### INCONCLUSIVE

Inherited ARM-V-anchored rule, unchanged:

```
mean EPE <  1.4317132                 -> ACCEPTED
1.4317132 <= mean EPE < 1.7727436     -> INCONCLUSIVE
mean EPE >= 1.7727436                 -> REFUTED
```

Mean global EPE = (1.4149796 + 1.4485736 + 1.4593946) / 3 = **1.4409826**.
1.4317132 ≤ 1.4409826 < 1.7727436, therefore **INCONCLUSIVE**, short of
ACCEPTED by **0.0092694 px**.

No threshold was adjusted, no seed was excluded, and the best seed was not
compared on its own.

**Recorded caveat about how this gate was built, not a re-reading of it.** The
ACCEPTED threshold was defined as *ARM-V mean − ARM-V spread*. ARM-V's spread
(0.3410304) was itself inflated by the very high-disparity instability this
experiment set out to remove; P2A's own spread is 0.0444151, 7.7× smaller. A
gate calibrated on the control's run-to-run noise becomes harder to clear the
more the treatment reduces that noise. This is a property of the rule's
construction, stated for the record. **It does not change the verdict, which is
INCONCLUSIVE.**

---

## 11. Overall Experimental Interpretation

The preregistration enumerated five outcome combinations. This result sits
between listed cases 4 and 5 and is reported as measured rather than forced
into either:

- **Mechanism NOT CONFIRMED** — the gate was not met.
- **Accuracy INCONCLUSIVE** — the gate was not met, but the measured
  improvement is large, one-directional and reproduced on every seed.

What can be said without inventing causality:

1. The intervention produced a **−0.3318 px** mean improvement under the
   unchanged frozen contract, with **every** treatment seed beating **every**
   control seed and the seed spread shrinking 7.7×.
2. The improvement is **concentrated exactly where the diagnosis said the error
   was**: GT ≥ 96 px EPE fell 33.31 → 12.42 (−63 %), GT ≥ 64 px fell
   11.76 → 5.15 (−56 %), while GT < 64 px — 95.4 % of pixels — improved
   slightly. The remaining bins [128,144) are still poor (31–35 px).
3. The maximum expressible disparity rose from 91.5–113.9 px to 130.6–142.9 px,
   reaching the reference's 135.7 px, against a GT maximum of 153.0 px.
4. The diagnosed mechanism moved strongly in the predicted direction on every
   seed, but **the preregistered threshold for calling it confirmed was not
   reached**, so no causal claim is made here beyond the association between
   the single changed variable and the measured outcome.

What is **NOT** established: that scale coverage is the whole mechanism; that
the residual [128,144) failure has the same cause; that a wider or differently
shaped scale range would do better — untested, and out of scope under the no-
tuning rule.

---

## 12. Phase-2 Next State

Per the preregistered Phase-2 rule for an INCONCLUSIVE accuracy outcome:

- **No rescue experiment.** No tuning of the scale range or distribution, no
  added seeds, no changed interpolation, no loss/LR/epoch change, no stacking
  with ARM-W/X/Y/Z, no multi-scale inference, no SceneFlow, no new arm.
- **Proceed directly to export / deployment validation.**
- The rule as written directs that **ARM-V is frozen as the Phase-2 accuracy
  model**.

**One conflict is flagged rather than resolved unilaterally, because it is the
project owner's call, not the experiment's.** Following the rule literally ships
ARM-V (mean 1.7727 px) over P2A (mean 1.4410 px) although every P2A seed beats
every ARM-V seed, the spread is 7.7× tighter, D1 is 1.68 pp better, the
low-disparity stratum is better, and P2A is 0.128 px from the frozen reference
against ARM-V's 0.459 px. P2A has identical parameters (397,954), an identical
graph, identical operators and identical inference cost, so the deployment
consequences of the two are indistinguishable. The gate miss is 0.0093 px.

Two defensible readings exist and the choice belongs to the owner:

- **(a) Follow the rule as written.** Freeze ARM-V; record P2A as a measured
  but non-accepted result. Maximum preregistration discipline.
- **(b) Deploy P2A, on the record that its verdict is INCONCLUSIVE.** The
  deliverable is the better model on every measured axis; the verdict label is
  not revised, and the report continues to state INCONCLUSIVE.

No further accuracy experiment is proposed under either reading. Nothing has
been frozen or exported pending that decision.

---

## 13. Closure Status

- `EXP-P2A-SCALE-COVERAGE-001`: **COMPLETE**. Three seeds, one intervention, no
  tuning, no stacking, no rescue. Both gates applied exactly as preregistered.
- Accuracy: **INCONCLUSIVE**. Mechanism: **NOT CONFIRMED**.
- This was the final planned accuracy experiment. The optimization stage is
  closed.
- `phase1/results/LEADERBOARD.md` was not modified. ARM-V, ARM-W, ARM-X, ARM-Y
  and ARM-Z artifacts and verdicts are untouched. The frozen reference is
  untouched. Phase-1 results are untouched.
- Outstanding before Phase-2 can close: the §12 model choice, then export,
  deployment validation, deployed-accuracy measurement, runtime/resource
  measurement and final artifacts.
- Phase 2 remains the final phase. No Phase 3 was created.
