# EXP-O6-REFINEMENT-DILATION-001 — results

**Verdict: `NO-GAP-TO-EXPLAIN`** — and the reason is more interesting than the
verdict. The D1 degradation O6 was registered to attribute **did not reproduce**.

Pre-registration: `EXP_O6_REFINEMENT_DILATION_001_PREREGISTRATION.md`, frozen
before the first arm started. Records:
`EXP-O6-REFINEMENT-DILATION-001-ARM-{D124,D128,D148}` and `-ANALYSIS`.
Machine-generated metrics: `phase2/results/EXP-O6-REFINEMENT-DILATION-001/`.

One seed per arm. Everything here is **suggestive / exploratory**. No
significance is claimed, and none is claimable.

---

## 1. Objective

E3C's three-block refinement arm changed two things at once — it removed three
residual blocks *and* it dropped dilation 8. Its +2.03 D1 point cost could not
be attributed to either. O6 exists solely to separate them. It is not an
architecture search and recommends no architecture.

## 2. Hypothesis

`H_O6` — the D1 degradation of the three-block `(1, 2, 4)` stack is
substantially attributable to removing the dilation-8 receptive field rather
than to having fewer refinement blocks.

**`H_O6` is not supported.** See section 9.

## 3. Arms

| arm | dilations | blocks | params | MACs (256x512) | receptive field | trained here |
|---|---|---:|---:|---:|---:|---|
| `CONTROL6` | 1,2,4,8,1,1 | 6 | 423,586 | — | 73 px | no — reused measured input |
| `D124` | 1,2,4 | 3 | 368,098 | 8.952 G | 33 px | yes |
| `D128` | 1,2,8 | 3 | 368,098 | 8.952 G | 49 px | yes |
| `D148` | 1,4,8 | 3 | 368,098 | 8.952 G | 57 px | yes |

Receptive field is `DERIVED` (`1 + 2*(input + output + sum of 2*dilation)`), not
measured.

## 4. Controlled variables

A dilated 3x3 convolution has the same weight shape and the same MAC count as an
undilated one, so **capacity and arithmetic cost are held exactly constant, not
merely balanced**. All three arms keep refinement blocks 0,1,2 of one seed-0
six-block stack and are then re-dilated in place — `dilation` and `padding` only,
weight tensors untouched — so **initial weights are bit-identical across arms**.

Everything else is imported from the H2 / seed-replication recipe: dataset,
splits, crop 256x512, augmentation, 12 disparities, cost volume with
`shift="left"`, standardised soft-argmin, feature extractor, 3D aggregation,
masked smooth L1, Adam lr 1e-3, cosine `T_max=200`, batch 2, 200 epochs, fp32,
seed 0, validation protocol, training harness.

Asserted numerically, not assumed: preflight PASS on 15 checks (identical
parameters, MACs, initial weights, tensor shapes and recipe fields across arms).
20 focused regression tests and the full 128-test Phase 2 suite passed before
training and again after.

## 5. Protocol

Preflight -> viability gate (all three arms: construct, forward, backward,
finite loss and gradients, refinement receives gradient, optimizer moves 60
tensors, validation, checkpoint round-trip) -> three 200-epoch runs, sequential,
each from scratch, no arm initialised from another -> analysis under the frozen
rule. Gates at epoch 10 (viability), 100 (collapse) and 200 (stereo verdict).

## 6. Training results — `MEASURED`

Late window = mean +- sd of the ten validation points at epochs 150-199,
recorded protocol (first 10 scenes of `hailo_val`, full frames, pooled over
`gt > 0`).

| arm | RF | params | late EPE | late D1 | wall clock |
|---|---:|---:|---:|---:|---:|
| `CONTROL6` | 73 px | 423,586 | 4.0921 +- 0.1000 | 24.9097 +- 1.0821 | (earlier session) |
| `D124` | 33 px | 368,098 | **4.1121 +- 0.0658** | **24.9701 +- 0.5897** | 3,674 s |
| `D128` | 49 px | 368,098 | **4.1764 +- 0.0847** | **28.0342 +- 0.7070** | 3,606 s |
| `D148` | 57 px | 368,098 | **4.1602 +- 0.0629** | **28.0212 +- 0.7363** | 3,590 s |

All three completed 200/200 epochs, no aborts, no NaN or Inf, zero batches with
gradient norm above 1e4.

Deltas against `D124`, the within-session three-block baseline:

| comparison | dEPE | dD1 | material (band conjunction) |
|---|---:|---:|---|
| `D128` − `D124` | +0.0644 | **+3.0641** | no — EPE inside band |
| `D148` − `D124` | +0.0481 | **+3.0511** | no — EPE inside band |

Band: 0.2138 px / 1.0000 D1 point, reused from E3b/E3C unchanged. "Material"
requires **both** metrics to exceed it; EPE moved almost not at all, so the
conjunction returns false even though D1 moved by three times its band.

## 7. Stereo functionality — `MEASURED`

All three arms **STEREO-FUNCTIONAL** at epoch 200 under the frozen rule.

| arm | right-image min D1 penalty | matching-map min D1 penalty | softmax entropy | final disparity std | matching gradient |
|---|---:|---:|---:|---:|---:|
| `D124` | +79.56 | +64.79 | 1.868 | 18.041 | 100 % of 16,000 batches |
| `D128` | +76.87 | +64.03 | 1.813 | 18.307 | 100 % of 16,000 batches |
| `D148` | +80.36 | +65.20 | 1.814 | 18.128 | 100 % of 16,000 batches |

Threshold is +20.0 D1 points; every arm clears it roughly four-fold on
right-image dependence and three-fold on matching-map dependence. No arm became
monocular. Nothing here is a `FAIL` on stereo grounds.

## 8. The reproducibility finding — the most consequential result

`D124` is the **same configuration, same recipe, same seed 0** as E3C's arm D.
The pre-registered cross-check:

| | EPE | D1 |
|---|---:|---:|
| E3C arm D (earlier session) | 3.7695 | 26.9402 |
| O6 `D124` (this session) | 4.1121 | 24.9701 |
| **delta** | **+0.3426** (band 0.2138) | **−1.9701** (band 1.0000) |

**Both deltas exceed the band. Two runs that should be identical are not.**

`MEASURED`: the environments match — same RTX 4060, CUDA 12.8, same torch, same
git state. All config differences between the two records are documentation
strings, not recipe fields.

`MEASURED`: the harness seeds `torch.manual_seed` and `np.random.seed` and sets
**no** determinism control anywhere — no `torch.use_deterministic_algorithms`,
no `cudnn.deterministic`, no `cudnn.benchmark`. Same-seed GPU runs are free to
select different kernels.

`INFERRED`: the divergence is GPU non-determinism accumulated over 200 epochs,
not a recipe difference.

The consequence for the gap O6 was built to explain:

| | d1(3-block) − d1(control) |
|---|---:|
| as measured by E3C | **+2.0304** (called material) |
| as measured here | **+0.0603** (not material) |

The effect E3C called material is the same size as the reproduction error of an
identical configuration. Under the frozen rule the gap is not material, so
`RECOVERY` is undefined and the verdict is **`NO-GAP-TO-EXPLAIN`**.

## 9. The six questions

**Q1 — Does (1,2,8) recover the D1/EPE behavior lost by (1,2,4)?**
No, and the question is malformed on this session's data: `(1,2,4)` lost nothing
to recover (+0.0603 D1 vs control, inside the band). `(1,2,8)` is **3.06 D1
points worse** than `(1,2,4)`, the opposite direction.

**Q2 — Does retaining dilation=8 materially improve the 3-block architecture?**
No. Both dilation-8 arms are worse on D1 by ~3.05 points and marginally worse on
EPE. Under the conjunction rule the difference is not "material" because EPE
barely moved; on D1 alone it is three times the band. Reported both ways rather
than picking whichever reads better.

**Q3 — Is the result consistent across (1,2,8) and (1,4,8)?**
Yes, strikingly. The two arms land at D1 **28.0342** and **28.0212** — 0.013
points apart — and EPE 4.1764 vs 4.1602, despite different receptive fields
(49 vs 57 px) and different sparse allocations. This is not `CONFIGURATION-
SENSITIVE`; the two schedules agree.

**Q4 — Capacity, receptive field, both, or insufficient?**
`INSUFFICIENT EVIDENCE — but not neutrally so.` The premise failed: there is no
capacity-versus-receptive-field gap to attribute this session, because
three blocks matched six blocks (+0.0603 D1, inside the band). What O6 *did*
measure is the reverse of its hypothesis — at fixed capacity, fixed MACs and
fixed initialisation, **widening the dilation schedule cost ~3 D1 points**.
That does not support "receptive field was the main limitation"; if anything it
is evidence against wider dilation helping here.

**Q5 — Does the candidate remain genuinely stereo-functional?**
Yes. All three arms pass every stereo gate with large margins (section 7). No
arm is a stereo `FAIL`.

**Q6 — Accuracy/compute tradeoff versus the six-block control?**
`D124` at three blocks: 368,098 params vs 423,586 (−13.1 %), 30.964 G vs
56.034 G deploy MACs (−44.7 %), for +0.0200 px EPE and +0.0603 D1 — both inside
the band this session. That looks like a good trade, **but section 8 shows the
control comparison is cross-session and this harness's cross-session noise
exceeds the band on both metrics.** The trade is not established.

## 10. Confounds and limitations

- **One seed per arm.** Suggestive, not significant.
- **The band is not trustworthy.** It was a single-sample lower bound from E3b;
  this experiment produced a second same-config sample that exceeds it on both
  metrics. Every "material" call in E3b/E3C that rests on it is weakened,
  including E3C's own +2.03 point result.
- **No within-session same-config repeat exists.** The only same-config repeat
  is cross-session. `INFERRED`, not measured: within-session noise appears much
  smaller than cross-session, because `D128` and `D148` — different
  configurations — landed 0.013 D1 points apart, which would be a coincidence if
  run-to-run noise were ~2 points. This should be measured, not inferred.
- **`CONTROL6` was reused, not retrained** (disclosed in the pre-registration).
  The vs-control comparison carries cross-session drift; the
  three-block-vs-three-block comparisons do not, and those are the clean ones.
- **160 training scenes from random initialisation.** Nothing transfers to a
  pretrained or competitively trained stereo network.
- **MACs are arithmetic, not latency.** Phase 1 measured 0.67x–142x
  misprediction on this stack. No latency was measured; no latency harness
  exists.
- **Record completeness defect, disclosed:** the O6 `arm_config` omits four
  documentation-only fields E3b recorded (`checkpoint_epochs`, `gate_at_epoch`,
  `stereo_gate_epoch`, `validation_protocol`). Behaviour is unaffected — those
  come from module constants identical in both runs — but the O6 records are
  less self-describing. Not retro-fitted: records are immutable, and changing it
  for two arms only would break the single-variable condition across arms.

## 11. Verdict

**`NO-GAP-TO-EXPLAIN`** (frozen rule, all gates PASS).

Stated plainly: **O6 could not answer whether capacity or receptive field caused
E3C's three-block penalty, because that penalty did not reproduce.** The
experiment was correctly designed and correctly executed — capacity, MACs and
initialisation were held exactly constant, and the disambiguation would have
worked — but the effect it was pointed at was not stable enough to attribute.

Two findings survive and are worth more than the verdict:

1. `MEASURED` — same-seed, same-config, same-machine runs of this harness differ
   by 0.34 px EPE and 1.97 D1 points across sessions, exceeding the band used to
   call effects material throughout E3b/E3C.
2. `MEASURED` — at fixed capacity, fixed MACs and fixed initialisation, both
   dilation-8 three-block schedules are ~3.05 D1 points worse than `(1,2,4)`,
   and they agree with each other to 0.013 points. Wider dilation did not help
   here; on D1 it hurt, consistently.

## 12. Next action

**Recommended: fix the measurement before running any further architecture
comparison.** Concretely, and in this order:

1. **Add determinism control to the harness** (`torch.use_deterministic_algorithms`,
   `cudnn.deterministic=True`, `cudnn.benchmark=False`, seeded DataLoader
   workers) and verify a same-seed repeat reproduces bit-for-bit, or record why
   it cannot.
2. **Measure the noise band properly** — 3+ same-config repeats within one
   session and across sessions, replacing E3b's single-sample lower bound with
   a real estimate. Until that exists, no D1 difference under ~2 points on this
   harness should be called material.
3. **Only then** revisit the E3b/E3C block-count series, whose monotone D1 trend
   (+0.88 / +1.16 / +2.03 at 5 / 4 / 3 blocks) may or may not survive a correct
   band.

Stage E remains **CLOSED**; nothing here implicates the cost volume. The
`(1,2,8)` / `(1,4,8)` question is **answered and closed** — retaining dilation 8
at three blocks does not help on this data and costs D1 consistently.

**Not recommended:** adopting any three-block configuration on the strength of
this run, or re-running O6 hoping for a cleaner gap.
