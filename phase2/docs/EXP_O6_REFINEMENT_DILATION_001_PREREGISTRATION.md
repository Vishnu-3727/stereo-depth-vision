# EXP-O6-REFINEMENT-DILATION-001 — pre-registration

**Written before any O6 training run.** Everything below is frozen at the point
the first arm starts. Nothing in sections 1–9 may be edited after results are
seen; corrections go in a `CORRECTION.md` beside the record.

Status: registered 2026-09-08. Executes the follow-up pre-committed in
`EXP_E3C_REFINEMENT_FLOOR_001_PREREGISTRATION.md` section 8.

---

## 0. Why this experiment exists

E3C trained a three-block refinement stack, `keep first 3` = dilations
`(1, 2, 4)`, and classified it `CAPACITY-REDUCIBLE-TO-3`. That label is about a
conjunction rule, not about which architecture is better, and E3C stated its own
confound before it ran:

> `keep first 3` is dilations 1, 2, 4, so it drops dilation 8 and shrinks the
> receptive field as well as the capacity. A negative result here cannot
> separate the two.

The measured E3C outcome makes this ambiguity load-bearing rather than academic:
against the six-block control, three blocks moved **EPE down** (better) and
**D1 up** (worse) — a genuine trade, not a uniform loss. Two readings survive:

- fewer refinement parameters cost D1 (a **capacity** limit), or
- losing the dilation-8 receptive field cost D1 (a **receptive-field** limit).

O6 exists only to separate those two. It is **not** an architecture search, and
no arm here is a candidate for adoption on its own.

## 1. Hypothesis

`H_O6` — the D1 degradation observed in the three-block `(1, 2, 4)`
configuration is substantially attributable to removal of the dilation-8
receptive field rather than to having fewer refinement blocks.

The experiment is designed to be able to falsify this: if the dilation-8-
preserving arms sit with `(1, 2, 4)` rather than with the six-block control,
`H_O6` is not supported and capacity is implicated instead.

## 2. Why the arms are parameter- and MAC-matched

A dilated 3x3 convolution has the same weight shape and the same multiply-
accumulate count as an undilated one; only the sampling grid changes. So all
three-block arms below hold **exactly** 368,098 unique parameters and
**exactly** the same arithmetic cost as E3C arm D. Capacity is therefore held
constant *by construction*, not merely balanced, and the only variable across
the three three-block arms is the dilation schedule.

This is the property that makes the disambiguation possible, and the preflight
asserts it numerically rather than assuming it.

## 3. Arms

All arms are three residual refinement blocks unless stated. All are trained
from scratch; none is initialised from another arm's checkpoint.

| Arm | Dilations | Blocks | Role | Trained here |
|---|---|---|---|---|
| `CONTROL6` | `(1, 2, 4, 8, 1, 1)` | 6 | six-block control | **no** — reused as a measured input, see section 4 |
| `D124` | `(1, 2, 4)` | 3 | E3C's configuration, retrained under this session's harness | yes |
| `D128` | `(1, 2, 8)` | 3 | keeps maximum dilation 8 at three blocks | yes |
| `D148` | `(1, 4, 8)` | 3 | different sparse allocation, also keeping dilation 8 | yes |

`D148` exists so that a `D128` result cannot be a lucky single schedule. If the
two disagree, section 8 forbids generalising from either.

No further arms are added. Two blocks, wider channels, other dilation ladders
and any adoption decision are explicitly **out of scope**.

## 4. Construction, and the fairness condition

Every arm is built by seeding at `SEED = 0`, constructing the full six-block H2
model (`exp_h2_seed_replication.build_model`), keeping refinement blocks at
indices `(0, 1, 2)`, and then **re-dilating** those surviving blocks to the
arm's schedule by setting each convolution's `dilation` and `padding`. Weight
tensors are never touched.

Consequences, all asserted in the preflight:

- the three three-block arms hold **bit-identical initial weights**, everywhere,
  including `input_conv` and `output_conv`. They differ in `dilation`/`padding`
  and in nothing else. Initialisation is therefore not a confound.
- `D124` built this way is the identical construction to E3C arm D, so E3C's
  recorded arm D is a legitimate same-recipe cross-check on the retrained
  `D124`, and any gap between them is same-seed run-to-run noise.
- padding equals dilation in `ResBlock`, so spatial shape is preserved at every
  dilation and the residual add stays valid.

`CONTROL6` is **not retrained**. `EXP-E3B-REFINEMENT-CAPACITY-001-ARM-A-RUN2`
(6 blocks, seed 0, 200 epochs, same harness) is reused as a measured input, the
same way E3C reused it. Recorded as a limitation in section 9: the control was
trained in an earlier session, so a control-vs-three-block comparison carries
any harness drift between sessions, while the three-block-vs-three-block
comparisons — which are the ones `H_O6` turns on — do not.

## 5. Controlled variables

Imported unchanged from the H2 / seed-replication recipe via
`exp_h2_seed_replication.recipe_config`, not restated by hand:

dataset (KITTI 2015), train split `hailo_calib` scenes 0–159, validation split
`hailo_val` (first 10 scenes validated), crop 256x512, augmentation (random
crop, per-image gain jitter sigma 0.1, no flip), disparity range 12,
cost-volume method and `shift="left"`, standardised soft-argmin readout,
feature extractor, 3D aggregation, loss (masked smooth L1, beta 1.0, valid =
`gt > 0 and gt < max_disparity`), Adam betas (0.9, 0.999), lr 1e-3,
`CosineAnnealingLR T_max=200`, batch size 2, 200 epochs, fp32, seed 0,
validation protocol, checkpoint epochs, gate epochs, training harness
(`exp_e3b_refinement_capacity.run_training`).

**The only intended change is `refinement_dilations`.** The preflight asserts
that every other recorded recipe field is byte-equal across arms.

## 6. Metrics

**Primary** — validation EPE (px) and D1 (%), recorded protocol (first 10 scenes
of `hailo_val`, full 368x1232 frames, pooled over `gt > 0`), summarised as the
late-window mean and sd over epochs 150–199.

**Secondary** — right-image dependence (min D1 penalty over `right_black`,
`right_noise`, `right_equals_left`), matching-map dependence (min D1 penalty
over `initial_constant_mean`, `initial_shuffled`), matching-path gradient
fraction, softmax entropy, final disparity std, unique parameter count, MACs,
training loss. Latency is **not** measured: no calibrated latency harness exists
in Phase 2 and MACs are arithmetic, not time.

## 7. Comparison band — frozen now

Reused from E3b/E3C, **not** recomputed:

```
EPE band = 0.2138 px
D1  band = 1.0000 pt
```

A difference is called **material** only when `|dEPE| > 0.2138` **and**
`|dD1| > 1.0000`. This is a comparison band, not a significance test. One seed
per arm: every O6 result is **suggestive / exploratory**. The words "proved",
"significant" and "optimal" are not available to this experiment.

## 8. Decision rule — frozen before results

Let `d1(X)` be arm X's late-window D1 mean. Define the D1 gap E3C opened:

```
GAP = d1(D124) - d1(CONTROL6)
```

`RECOVERY(X) = (d1(D124) - d1(X)) / GAP`, the fraction of that gap closed by
arm X. Evaluated only if `GAP > D1 band`; otherwise there is no gap to recover
and the verdict is `NO-GAP-TO-EXPLAIN`.

An arm is **stereo-functional** iff the frozen `_stereo_verdict` rule at epoch
200 returns true. Any arm failing it is `FAIL` as a stereo architecture
regardless of accuracy, per the standing rule.

| Condition | Verdict |
|---|---|
| any arm aborted, or control not stereo-functional, or Phase 1 diff non-empty | `INVALID` |
| any O6 arm not stereo-functional at epoch 200 | `STEREO-BROKEN` |
| `GAP` not material | `NO-GAP-TO-EXPLAIN` |
| both `D128` and `D148` recover >= 50 % of `GAP` | `RECEPTIVE-FIELD-IMPLICATED` |
| both recover <= 20 % of `GAP` | `CAPACITY-IMPLICATED` |
| one recovers >= 50 % and the other <= 20 % | `CONFIGURATION-SENSITIVE` |
| anything else (including both landing between 20 % and 50 %) | `INCONCLUSIVE` |

The 50 % / 20 % thresholds are set now, without having seen any O6 number, and
may not be moved afterwards. `INCONCLUSIVE` is an acceptable outcome and will be
reported as such; no winner will be forced.

## 9. Known limitations, stated before the run

- **One seed per arm.** No significance is claimed or claimable. Results are
  suggestive.
- **160 training scenes from random initialisation.** Nothing here transfers to
  a pretrained model or to a competitively trained stereo network.
- **`CONTROL6` was trained in an earlier session** (section 4). The
  three-block-vs-three-block comparisons are within-session; the
  vs-control comparison is not.
- **MACs are arithmetic**, not latency. Phase 1 measured 0.67x–142x
  misprediction of latency from MACs on this stack.
- **The band is a single-sample lower bound** on same-seed noise, not a
  confidence interval.
- **Scope.** O6 answers *why the three-block result changed*. It does not
  recommend an architecture, and a favourable arm here is not an adoption
  decision.

## 10. What may not change

The arms, the construction, the controlled variables, the band, the decision
rule and the thresholds in sections 3–8. If any of them turns out to be wrong,
the run is recorded as-is and a new experiment ID is registered.
