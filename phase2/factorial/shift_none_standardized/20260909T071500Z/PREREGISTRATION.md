# EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001 — pre-registration

**Status: REGISTERED, NOT RUN.** Written and frozen before the treatment arm
started and before any 200-epoch result under this configuration existed. The
decision logic in §6 is the operator's, transcribed here verbatim in substance
before the data.

---

## 1. Question

> Does `cost_volume_shift` provide useful stereo value after the H2 standardised
> readout has removed the soft-argmin saturation failure?

This is the missing cell of the 2×2 factorial. Three cells are already filled at
200 epochs:

| | readout = raw | readout = standardised |
|---|---|---|
| **shift = none** | `HISTORICAL` `EXP-H1-BASE-v2` — 14.330 px / 81.19 % | **THIS EXPERIMENT** |
| **shift = left** | `HISTORICAL` `EXP-H1-WORKING-v2` — 14.340 px / 80.74 % | `HISTORICAL` H2 / Stage B control |

## 2. Control

Stage B deterministic Run A, `STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA`
(`phase2/diagnostics/determinism/20260909T041500Z_baseline`). Not retrained, not
modified.

`HISTORICAL` (Stage B, `MEASURED` there):

```
40-scene pooled:  EPE 2.2955182 px   D1 15.6045143 %   RMSE 6.0437075
late window:      EPE 3.8670192 ± 0.1113151   D1 23.4954125 ± 0.9342410
stereo @200:      right +80.8127994, map +66.1876924, entropy 1.7754723,
                  disparity std 18.0000028 -- STEREO_FUNCTIONAL
final weight sha: 3ad382c50d413ad2
```

## 3. Treatment — one changed variable

```
cost_volume_shift:  "left"  ->  "none"
```

Applied **in place on the built model**, after `build_model(seed=0)` has consumed
the RNG. `CostVolume` holds no parameters and reads `shift` at forward time, so
the initial weights must be bit-identical to the control's. The same in-place
pattern `exp_o6_refinement_dilation.py` uses for dilation.

Held fixed by importing the recipe rather than restating it — the arm is trained
by calling `exp_e3b_refinement_capacity.run_training("A", ...)` itself: model,
feature extractor, cost-volume construction, 12 candidates, standardisation and
its location, soft-argmin, refinement, six blocks (dilations 1,2,4,8,1,1), loss,
optimizer, learning-rate schedule, initialisation, seed 0, dataset, splits, crop
policy, augmentation, batch size, 200 epochs, validation protocol, checkpoint and
snapshot schedule.

**No tuning of any kind. One execution.**

## 4. Protocol — mandatory

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Stage B established that these give bit-identical fresh-process reproduction of
this configuration. Determinism errors are **not** suppressed: if an operation
refuses deterministic execution the run stops and the operation is reported.

Note, recorded because the instruction text and the working value differ: the
brief writes `CUBLAS_WORKSPACE_CONFIG=:4096:8:` with a trailing colon. cuBLAS
accepts `:4096:8` or `:16:8`; the trailing-colon form is not a valid setting.
Stage A and Stage B used `:4096:8` and it worked, so `:4096:8` is used here.

## 5. Stop conditions

Stop and preserve the failure if: the initial weight hash differs from
`c4d02385e3da28a5`; deterministic execution fails; `git diff phase-1-frozen --
src scripts` is non-empty; the dataset, crop policy or standardisation differs;
more than one scientific variable changes; training crashes; or a hidden protocol
difference appears. **Do not restart merely to obtain a successful run.**

## 6. Interpretation — frozen before the result

| outcome | condition | reading |
|---|---|---|
| **A — SHIFT-IMPORTANT** | large degradation in EPE/D1 **and** loss of meaningful stereo behaviour | the H2 readout stabilisation does not remove the importance of the disparity shift; the shift/search mechanism is materially useful |
| **B — SHIFT-NOT-NECESSARY** | similar accuracy **and** preserved meaningful stereo behaviour | the standardised readout may have made explicit shifted correspondence unnecessary for this task and configuration. **This does not license concluding that stereo matching is unnecessary**; it triggers deeper correspondence analysis first |
| **C — INCONCLUSIVE** | moderate change, or the accuracy and stereo evidence disagree | no binary conclusion is forced; the exact missing evidence is named |

## 7. No materiality band

The historical E3b/E3c band (0.2138 px / 1.0000 D1 point) is **invalidated** and
is not reused. **No replacement band is invented, before or after the result.**
The effect is classified by:

1. the exact measured Δ on all 40 scenes;
2. per-scene consistency across the 40 rows;
3. stereo-functional evidence across epochs 10/50/100/150/200;
4. magnitude relative to the previously measured protocol divergence
   (`HISTORICAL`: Stage B controlled-vs-uncontrolled 0.2250805 px / 1.4143253 pt;
   O6 uncontrolled same-config 0.3426 px / 1.9701 pt);
5. mechanistic interpretation.

## 8. Probe caveat, stated in advance

Right-image corruption and matching-map destruction establish **binocular
dependence**, not correct disparity search. `HISTORICAL` — Phase 1 EXP-007
measured **+89.7 D1 points** of right-image dependence on the *reference*
weights, whose shift is provably degenerate. These probes are therefore
supporting evidence only. `disparity_initial`'s correlation with ground truth and
the cost-volume degeneracy check carry the correspondence question.

## 9. What this experiment is not

Not an architecture search, not a capacity study, not a model-development
session. The treatment will not be improved, tuned or rescued, and an unfavourable
result is not grounds for a rerun.

## 10. Expectation explicitly not inherited

`HISTORICAL` — H3 measured `shift="none"` at **10 epochs** under the
nondeterministic protocol and returned KILL, partly on an epoch-10 stereo
criterion the project later invalidated (`EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION`).
**H3's verdict is not copied into this experiment.** This is the clean 200-epoch
factorial comparison under the repaired protocol, and its result stands on its
own measurements.
