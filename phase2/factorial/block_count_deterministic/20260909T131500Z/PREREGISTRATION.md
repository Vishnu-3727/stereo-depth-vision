# EXP-BLOCKCOUNT-DETERMINISTIC-001 — pre-registration

**Status: REGISTERED, NOT RUN.** Written and frozen before either treatment arm
started and before any deterministic 5-block or 4-block result existed.

---

## 1. Question

> How much of the six-block refinement capacity is actually necessary, now that
> the stereo mechanism is known to be working?

`HISTORICAL` — E3b asked this and returned `CAPACITY-REDUCIBLE`; E3c extended it
to three blocks; O6 then showed the harness's own same-config reproduction error
(0.3426 px / 1.9701 D1) equalled or exceeded every effect those experiments
measured, so their verdicts stand as **INDETERMINATE**. Stage A identified the
cause (nondeterministic backward pass) and Stage B removed it. This experiment
re-asks the question under the repaired protocol.

## 2. Control — frozen, not retrained

Stage B deterministic Run A, `STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA`
(`phase2/diagnostics/determinism/20260909T041500Z_baseline`).

`HISTORICAL` (`MEASURED` in Stage B):

```
40-scene pooled:   EPE 2.2955182 px   D1 15.6045143 %   RMSE 6.0437075
late window:       EPE 3.8670192 ± 0.1113151   D1 23.4954125 ± 0.9342410
stereo @200:       right +80.8127994, map +66.1876924 -- STEREO_FUNCTIONAL
final weight sha:  3ad382c50d413ad2
```

**It is not retrained.** Stage B established that a fresh-process rerun of this
exact configuration is bit-identical — same final weight sha, same loss series,
same validation series, same 40-scene evaluation. Retraining it could only
reproduce it, so doing so would add cost and no information. Its checkpoint and
records are read-only here.

## 3. Treatment arms — two, one changed variable

| arm | blocks | dilations | `HISTORICAL` E3b parameters |
|---|---:|---|---:|
| **B** | 5 | 1,2,4,8,1 | 405,090 |
| **C** | 4 | 1,2,4,8 | 386,594 |

```
refinement block count:  6  ->  5   (arm B)
refinement block count:  6  ->  4   (arm C)
```

The trailing dilation-1 blocks are removed and the 1-2-4-8 ladder is kept intact
— the choice `HISTORICAL` E3b froze in its own pre-registration, reused verbatim
so the arms remain comparable to the historical series.

Arms are built by `exp_e3b_refinement_capacity.build_arm`, **unmodified**: the
full six-block stack is constructed from seed 0 and then truncated, so every
surviving weight tensor is bit-identical to the control's. The preflight asserts
this.

Held fixed by importing the recipe rather than restating it — each arm is trained
by calling `exp_e3b_refinement_capacity.run_training(arm, ...)` itself: model,
feature extractor, cost volume with `shift="left"`, 12 candidates,
standardisation and its location, soft-argmin, loss, optimizer, learning-rate
schedule, seed 0, dataset, splits, crop policy, augmentation, batch size, 200
epochs, validation protocol, checkpoint and snapshot schedule.

**No tuning. One execution per arm. Arm C runs whether or not arm B succeeds.**

## 4. Protocol — mandatory

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Determinism errors are not suppressed. If an operation refuses deterministic
execution the run stops and the operation is reported.

## 5. Primary endpoint

`MEASURED` on **all 40 `hailo_val` scenes**, full 368×1232 frames, pooled over
`gt > 0` — the existing evaluator, unchanged. The historical 10-scene
training-time subset is reported for cross-reference only; Stage B measured it to
be 1.67× harder on EPE, so the two are not interchangeable.

Per-scene EPE, D1 and valid-pixel count are recorded for all 40 scenes and all
three configurations.

## 6. No materiality band

The historical E3b/E3c band (0.2138 px / 1.0000 D1 point) is **invalidated** and
is not reused. **No replacement band is invented, before or after the result.**
Effects are classified by:

1. the exact measured Δ against the frozen control on all 40 scenes;
2. per-scene consistency — how many of 40 scenes move in each direction;
3. stereo-functional evidence at epochs 10/50/100/150/200;
4. magnitude relative to measured reproducibility scales:
   controlled same-config variance **0** (Stage B); controlled-vs-uncontrolled
   protocol divergence 0.2250805 px / 1.4143253 pt (Stage B);
   `HISTORICAL` O6 uncontrolled same-config error 0.3426 px / 1.9701 pt;
5. the compute actually saved (parameters and MACs, measured not assumed);
6. mechanistic interpretation.

## 7. The claim ceiling — frozen before the result

**Bitwise reproducibility removes execution noise, not seed noise.** Stage B
proved that rerunning the *same* configuration returns bit-identical results. It
did **not** establish that a *different seed* would produce the same difference
between *different* architectures. Arms B and C differ from the control both by
architecture and by whatever trajectory seed 0 happens to induce for each.

Therefore the strongest statement this experiment can make is:

> at seed 0, under this deterministic protocol, the measured difference is
> exactly X — with no execution uncertainty and no generalisation to other seeds.

`UNKNOWN` — seed-to-seed variation of these differences. It is not measured here
and **no significance will be claimed**. Any statement of the form "N blocks
match six" must carry "at seed 0" or it is unsupported.

## 8. Verdict rule — frozen

Exactly one verdict per arm:

| verdict | condition |
|---|---|
| `CAPACITY-REDUCIBLE-AT-SEED-0` | the arm's 40-scene EPE and D1 are close to the control's relative to the reproducibility scales in §6.4, **and** it passes the frozen epoch-200 stereo criterion (right-image ≥ +20, matching-map ≥ +20, matching gradient ≥ 95 %, entropy > 0.5, disparity std > 1.0), **and** per-scene direction is not systematically adverse |
| `CAPACITY-REQUIRED-AT-SEED-0` | the arm degrades on both metrics by an amount large against those scales, with consistent per-scene direction |
| `STEREO-BROKEN` | accuracy is close but the frozen stereo criterion fails — a rejection, not a success |
| `INCONCLUSIVE` | the metrics disagree, the per-scene direction is mixed, or the magnitude sits inside the reproducibility scales without a consistent per-scene signal |

An arm judged `CAPACITY-REDUCIBLE-AT-SEED-0` is **not** thereby recommended for
adoption; §7 bounds what may be claimed.

## 9. Stop conditions

Stop and preserve the failure if: a surviving weight tensor differs from the
control's; the parameter count differs from the E3b registration; deterministic
execution fails; `git diff phase-1-frozen -- src scripts` is non-empty; the
dataset, crop policy, shift or standardisation differs; more than one scientific
variable changes; or training crashes. **Do not restart to obtain a successful
run.** Do not tune, rescue or improve a treatment.

## 10. Scope

Stage E stays **CLOSED**. Nothing in this experiment touches the cost volume's
representation, candidate count or aggregation. This is a refinement-capacity
question only, and its result may not be used to open cost-volume work.

## 11. Relationship to the historical series

`HISTORICAL` E3b, E3c and O6 records, metrics, verdicts, reports and
pre-registrations are **not modified**. This experiment is additive. Its numbers
are not a correction of theirs: the training protocol differs, so a
historical-vs-deterministic difference is a protocol characterisation, not an
architecture effect, exactly as Stage B established.
