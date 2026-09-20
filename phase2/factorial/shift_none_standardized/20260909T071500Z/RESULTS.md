# EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001 — RESULTS

Protocol frozen in advance: `PREREGISTRATION.md` (written before the run). One
treatment arm, one execution, no tuning, no rerun. Deterministic protocol from
Stage B. Nothing historical was modified.

Tags: `MEASURED` · `HISTORICAL` (previously recorded, cited unchanged) ·
`DERIVED` · `INFERRED` · `UNKNOWN`.

---

## QUESTION

Does `cost_volume_shift` provide useful value after standardized readout?

---

## CONTROL

`shift=left + standardized readout` — Stage B deterministic Run A,
`STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA`
(`phase2/diagnostics/determinism/20260909T041500Z_baseline`). Not retrained, not
modified; its checkpoint and records were read only.

## TREATMENT

`shift=none + standardized readout` —
`EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001`.

`MEASURED` (`results/preflight.json`) — exactly one scientific variable differs:

| | control | treatment |
|---|---|---|
| `cost_volume_shift` | `left` | **`none`** |
| **initial weight sha** | `c4d02385e3da28a5` | **`c4d02385e3da28a5`** — equal, and equal to the expected value |
| parameters / MACs @256×512 | 423,586 / 16.1995 G | 423,586 / 16.1995 G |
| refinement blocks / dilations | 6 / `[1,2,4,8,1,1]` | 6 / `[1,2,4,8,1,1]` |
| candidates / method | 12 / subtract | 12 / subtract |
| readout | `StandardisedDisparityRegression`, after the full-res upsample | same |
| seed / epochs / batch / crop | 0 / 200 / 2 / 256×512 | same |
| dataset / splits / augmentation | `hailo_calib` 0–159 → `hailo_val` 160–199 / gain jitter σ=0.1 | same |

`MEASURED` — the shift is flipped **in place** after `build_model(0)` has
consumed the RNG. `CostVolume` holds no parameters, so the initial weights are
bit-identical rather than merely similar — verified by the equal hashes above.

`MEASURED` — the search really is removed: the maximum absolute difference
between any of the 12 candidate slices and slice 0 is **exactly `0.0`**.

`MEASURED` — the readout really is the standardised one and is not saturated at
initialisation: standardised cost mean −3.21e-07, std 0.9949; softmax entropy
**1.5428 nats**; max softmax probability 0.7429; matching-path gradient present
(norm 327.9). `DERIVED` — this is genuinely the *standardised-readout* cell, not
a re-run of H1's broken regime.

---

## TRAINING

`MEASURED`:

| | control | treatment |
|---|---:|---:|
| epochs | 200 / 200 | **200 / 200** |
| aborted | `null` | **`null`** |
| seed | 0 | 0 |
| deterministic protocol | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | identical |
| determinism errors raised or suppressed | none | **none** |
| epoch-0 val EPE (untrained) | 31.9576479 | 31.9917053 |
| **final train loss** | **0.9037397** | **2.9703582** |
| gradient median / max | 21.9161959 / 308.0343933 | 27.5066547 / 313.0419617 |
| NaN·Inf / batches > 1e4 | none / 0 | none / 0 |
| matching-path gradient | 100 % of 16,000 batches | **100 % of 16,000 batches** |
| wall clock | 4,948.997 s = 82.48 min | **4,827.181 s = 80.45 min** |
| final weight sha | `3ad382c50d413ad2` | `287ad985be0cf448` |

`MEASURED` — the treatment trained cleanly. It is not a failed or unstable run:
no abort, no NaN, no gradient spike, matching-path gradient on every batch, and
its train loss fell from 10.53-scale to 2.9704. `DERIVED` — it **optimised**; it
simply optimised to a much worse solution.

### Late window (epochs 155–200, ten recorded points)

`MEASURED`:

| | control | treatment | Δ |
|---|---:|---:|---:|
| EPE mean ± sd | 3.8670192 ± 0.1113151 | **11.8376516 ± 0.5515163** | **+7.9706324** |
| D1 mean ± sd | 23.4954125 ± 0.9342410 | **71.1498619 ± 2.3017670** | **+47.6544494** |

---

## FULL-40-SCENE RESULTS

`MEASURED` — all 40 `hailo_val` scenes, full 368×1232 frames, pooled over
`gt > 0` (3,802,797 pixels), existing evaluator, unchanged. Raw:
`eval40.json` (treatment) and the control's `eval40_run_A.json`.

| Metric | Control | Treatment | Δ |
| ------ | ------: | --------: | -: |
| **EPE** | **2.2955182 px** | **8.3717466 px** | **+6.0762284** |
| **D1** | **15.6045143 %** | **66.5977700 %** | **+50.9932558** |
| RMSE | 6.0437075 | 13.0058681 | +6.9621606 |
| bad1 | 48.8446793 % | 88.0038824 % | +39.1592031 |
| bad2 | 25.1886966 % | 76.6490296 % | +51.4603330 |

`DERIVED` — the treatment is **3.65× worse on EPE** and **4.27× worse on D1**.

### Per-scene distribution

| statistic | control EPE | treatment EPE | control D1 | treatment D1 |
|---|---:|---:|---:|---:|
| mean | 2.2591602 | 8.2080144 | 15.4255496 | 65.9905559 |
| median | 1.6920491 | 6.9382080 | 13.9697973 | 65.4540434 |
| sd | 1.7051294 | 3.5214326 | 7.8169763 | 13.9756194 |
| min | 1.0931443 | 3.4190433 | 4.9776906 | 34.7521247 |
| max | 10.6710401 | 17.5237498 | 39.3893676 | 95.3668312 |

`MEASURED` — the treatment's **best** scene (3.4190 px, `000180_10.png`) is worse
than the control's **worst** scene on D1 (39.3894 %) → 34.7521 %, and worse than
30 of the control's 40 scenes on EPE.

---

## PER-SCENE RESULTS

`MEASURED` — **the treatment is worse on all 40 of 40 scenes on EPE and on all
40 of 40 on D1.** No scene favours removing the shift.

| scene | ctrl EPE | treat EPE | ΔEPE | ctrl D1 | treat D1 | ΔD1 | valid px |
|---|---:|---:|---:|---:|---:|---:|---:|
| 000160_10 | 1.1599 | 5.0530 | +3.8931 | 8.4623 | 57.9003 | +49.4380 | 100,174 |
| 000161_10 | 10.6710 | 17.5237 | +6.8527 | 39.3894 | 60.1580 | +20.7687 | 114,668 |
| 000162_10 | 2.1094 | 5.4236 | +3.3142 | 18.0050 | 51.7188 | +33.7138 | 58,878 |
| 000163_10 | 2.2240 | 14.1530 | +11.9290 | 21.3468 | 68.9981 | +47.6513 | 100,610 |
| 000164_10 | 5.0929 | 10.8358 | +5.7430 | 26.1691 | 57.9334 | +31.7644 | 104,123 |
| 000165_10 | 2.6617 | 14.6221 | +11.9604 | 32.3245 | 86.8413 | +54.5168 | 101,598 |
| 000166_10 | 1.6868 | 8.6232 | +6.9364 | 14.5787 | 79.8084 | +65.2297 | 109,818 |
| 000167_10 | 1.9050 | 7.0687 | +5.1637 | 16.5400 | 63.2833 | +46.7434 | 98,138 |
| 000168_10 | 5.2002 | 13.6745 | +8.4743 | 26.6739 | 84.0407 | +57.3669 | 88,776 |
| 000169_10 | 4.5741 | 17.1431 | +12.5690 | 23.6780 | **95.3668** | +71.6889 | 53,894 |
| 000170_10 | 1.6279 | 6.3428 | +4.7149 | 15.1587 | 63.7570 | +48.5983 | 88,820 |
| 000171_10 | 1.0931 | 6.5410 | +5.4479 | 5.6767 | 79.1286 | +73.4519 | 84,556 |
| 000172_10 | 1.5682 | 8.1830 | +6.6148 | 10.5529 | 80.4879 | +69.9350 | 86,649 |
| 000173_10 | 3.0891 | 10.5494 | +7.4602 | 21.2422 | 83.8688 | +62.6266 | 116,377 |
| 000174_10 | 2.1330 | 5.9844 | +3.8514 | 20.1199 | 52.3948 | +32.2748 | 98,882 |
| 000175_10 | 2.4455 | 7.9805 | +5.5350 | 20.1341 | 60.3808 | +40.2467 | 65,342 |
| 000176_10 | 2.5154 | 6.6544 | +4.1390 | 23.5217 | 52.6095 | +29.0879 | 93,637 |
| 000177_10 | 3.2421 | 6.7991 | +3.5570 | 14.4195 | 50.7840 | +36.3646 | 96,550 |
| 000178_10 | 2.1029 | 5.3252 | +3.2224 | 16.4147 | 43.6769 | +27.2622 | 97,571 |
| 000179_10 | 2.1505 | 5.2088 | +3.0583 | 24.2079 | 51.0614 | +26.8535 | 78,243 |
| 000180_10 | 1.1617 | **3.4190** | **+2.2574** | 6.7234 | **34.7521** | +28.0287 | 72,597 |
| 000181_10 | 1.4045 | 4.2983 | +2.8938 | 9.5326 | 39.0184 | +29.4857 | 73,327 |
| 000182_10 | 1.4197 | 6.0452 | +4.6254 | 10.9158 | 60.2307 | +49.3149 | 73,343 |
| 000183_10 | 1.1516 | 6.7973 | +5.6458 | 5.9315 | 59.1783 | +53.2469 | 97,632 |
| 000184_10 | 1.3324 | 5.1174 | +3.7850 | 8.1546 | 59.0205 | +50.8660 | 89,496 |
| 000185_10 | 1.1717 | 7.0331 | +5.8614 | 4.9777 | 85.9770 | **+80.9993** | 101,975 |
| 000186_10 | 1.6087 | 6.8433 | +5.2347 | 11.6164 | 72.1512 | +60.5348 | 79,095 |
| 000187_10 | 1.3206 | 7.5297 | +6.2091 | 9.6030 | 68.3688 | +58.7658 | 77,101 |
| 000188_10 | 1.2007 | 7.3365 | +6.1358 | 9.0410 | 74.5240 | +65.4830 | 113,240 |
| 000189_10 | 1.6973 | 9.7247 | +8.0274 | 16.9839 | 82.8858 | +65.9020 | 128,157 |
| 000190_10 | 1.2473 | 8.9970 | +7.7497 | 9.7701 | 76.0649 | +66.2948 | 118,771 |
| 000191_10 | 1.2915 | 4.1272 | +2.8356 | 8.6610 | 57.9324 | +49.2714 | 81,041 |
| 000192_10 | 1.3548 | 7.3176 | +5.9629 | 10.9703 | 66.1784 | +55.2081 | 112,239 |
| 000193_10 | 3.2353 | 9.1108 | +5.8756 | 24.2126 | 51.3666 | +27.1540 | 121,065 |
| 000194_10 | 1.8389 | 14.3798 | +12.5409 | 11.7274 | 76.8139 | +65.0865 | 104,209 |
| 000195_10 | 1.3927 | 6.7191 | +5.3264 | 8.5950 | 64.7297 | +56.1347 | 117,243 |
| 000196_10 | 1.3911 | 5.8865 | +4.4953 | 10.7030 | 66.7427 | +56.0397 | 122,704 |
| 000197_10 | 2.1596 | 11.6475 | +9.4879 | 13.5201 | 80.3008 | +66.7807 | 125,132 |
| 000198_10 | 2.2702 | 6.2716 | +4.0015 | 17.5977 | 70.0955 | +52.4978 | 108,997 |
| 000199_10 | 1.4633 | 6.0294 | +4.5661 | 9.1691 | 69.0914 | +59.9223 | 48,129 |

`MEASURED` — per-scene ΔEPE: mean +5.9489, median +5.4914, sd 2.6849, range
**+2.2574 … +12.5690** — every value positive. Per-scene ΔD1: mean +50.5650,
median +52.8723, sd 15.5966, range **+20.7687 … +80.9993** — every value
positive.

`DERIVED` — the **smallest** per-scene degradation anywhere in the set (+2.2574
px, +20.7687 pt) is an order of magnitude above every reproducibility scale this
project has measured: Stage B's controlled same-config variance is **0**, Stage
B's controlled-vs-uncontrolled protocol divergence was 0.2250805 px / 1.4143253
pt, and `HISTORICAL` O6's uncontrolled same-config error was 0.3426 px / 1.9701
pt. No materiality band is needed, and none was reused or invented.

---

## STEREO FUNCTIONALITY

`MEASURED` — probes at epochs 10/50/100/150/200, `stereo_probe` imported
unchanged, four focus scenes 27/0/31/6, run on weight snapshots both runs had
already saved (the `EXP-H2-EMERGENCE-001` method; the control was **not**
retrained). Right-image and matching-map figures are the **worst case** over
variants. Raw: `probes.json`.

| epoch | right-image ΔD1 |  | matching-map ΔD1 |  | entropy (nats) |  | `disparity_initial` r(GT) |  |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | **control** | **treatment** | **control** | **treatment** | **control** | **treatment** | **control** | **treatment** |
| 10 | +31.476 | **−4.747** | +33.439 | **−6.014** | 1.7350 | 2.0733 | +0.7548 | +0.5440 |
| 50 | +58.072 | **−6.947** | +47.845 | **−2.222** | 1.7151 | 2.1312 | +0.8596 | +0.5643 |
| 100 | +71.175 | **−9.441** | +57.584 | **−3.839** | 1.7561 | 2.0521 | +0.8865 | +0.5326 |
| 150 | +80.043 | **−5.398** | +65.027 | **+4.026** | 1.7591 | 2.0261 | +0.9158 | +0.5788 |
| 200 | +80.813 | **−7.083** | +66.188 | **+0.344** | 1.7755 | 2.0431 | +0.9168 | +0.5707 |

`MEASURED` — the frozen epoch-200 verdict:

| | control | treatment |
|---|---|---|
| right-image dependence (≥ +20) | +80.8127994 → **true** | **−7.0829172 → false** |
| matching-map dependence (≥ +20) | +66.1876924 → **true** | **+0.3442059 → false** |
| matching gradient (≥ 95 %) | 100 % → true | 100 % → true |
| entropy (> 0.5) | 1.7754723 → true | 2.0430784 → true |
| disparity std (> 1.0) | 18.0000028 → true | 12.3873029 → true |
| **STEREO_FUNCTIONAL** | **true** | **false** |

`MEASURED` — the treatment's right-image dependence is **negative at every one of
the five checkpoints**: corrupting the second camera makes it *better* by 4.7 to
9.4 D1 points. Destroying its matching map costs it essentially nothing
(−6.0 to +4.0 across the series, +0.34 at epoch 200).

`MEASURED` — its `disparity_initial` correlation with ground truth is flat across
training (+0.5440 → +0.5707, no trend) while the control's climbs +0.7548 →
+0.9168.

`MEASURED` — its softmax entropy sits at 2.03–2.13 nats of a possible 2.4849,
against the control's 1.72–1.78. `DERIVED` — the treatment's readout stays
closer to uniform: with all 12 candidate slices identical there is no
disparity-axis structure for it to sharpen on.

**Probe caveat, as pre-registered.** `HISTORICAL` — Phase 1 EXP-007 measured
**+89.7 D1 points** of right-image dependence on the *reference* weights despite a
provably degenerate shift, so a *passing* binocular probe would not prove
disparity search. Here the probes **fail**, which is the direction they can
support: a model that becomes better when the second camera is destroyed is not
using it at all. The accuracy result does not depend on the probes.

---

## INTERPRETATION

**`MEASURED`**

1. 40-scene pooled: EPE 2.2955182 → 8.3717466 (**+6.0762284**, ×3.65); D1
   15.6045143 → 66.5977700 (**+50.9932558**, ×4.27); RMSE 6.0437075 → 13.0058681.
2. Worse on **40/40 scenes on EPE and 40/40 on D1**; smallest degradation
   +2.2574 px / +20.7687 pt.
3. Late window: +7.9706324 px / +47.6544494 pt.
4. Right-image dependence negative at **all five** probe epochs (−4.7 to −9.4);
   matching-map dependence ≈0; `STEREO_FUNCTIONAL: false`.
5. `disparity_initial` r(GT) flat at ≈+0.55 across 200 epochs versus the
   control's rise to +0.9168.
6. The treatment trained cleanly: 200/200 epochs, no abort, no NaN/Inf, zero
   gradient spikes, matching-path gradient on 100 % of 16,000 batches, train loss
   10.53-scale → 2.9704.
7. Both arms started from bit-identical weights (`c4d02385e3da28a5`) and the
   cost-volume degeneracy was confirmed at `0.0` before training.

**`DERIVED`**

8. Removing the shift while holding the standardised readout costs about
   **6.1 px and 51 D1 points** on the authoritative 40-scene protocol. That is
   ~27× Stage B's measured protocol divergence on EPE and ~36× on D1, and
   infinitely many times the controlled same-config variance, which is zero.
   The effect is not a noise or protocol artefact.
9. This is **Outcome A** of the pre-registered logic: large accuracy degradation
   **and** loss of meaningful stereo behaviour.
10. Fixing the readout is **necessary but not sufficient**. The standardised
    readout is present, unsaturated and receiving gradient in the treatment, and
    the model still fails to use the second camera. Both the readout fix and the
    disparity shift are required; neither alone produces a stereo model.
11. The 2×2 is now complete at 200 epochs, and only the both-on cell works:

    | | readout raw | readout standardised |
    |---|---|---|
    | **shift none** | `HISTORICAL` 14.330 px / 81.19 % | `MEASURED` **8.372 px / 66.60 %** |
    | **shift left** | `HISTORICAL` 14.340 px / 80.74 % | `MEASURED` **2.296 px / 15.60 %** |

    (`HISTORICAL` H1-v2 figures are on the 10-scene protocol and are cited for
    direction only, not as a like-for-like comparison — see UNKNOWN 15.)

**`INFERRED`**

12. With every candidate slice identical, the aggregated cost carries no
    disparity-axis structure, so the soft-argmin's output can only encode the
    magnitude of the zero-shift photometric difference. `HISTORICAL` — Phase 1
    identified exactly this as the mechanism by which the deployed reference
    reaches 8.15 % D1 without search. The evidence does not force this reading.
13. The treatment's ≈+0.55 r(GT), flat and non-improving, is consistent with a
    monocular layout prior plus a weak photometric cue. Not established.

**`UNKNOWN`**

14. Whether `shift="none"` could develop right-image dependence at a much larger
    budget. 200 epochs on 160 scenes is what was tested.
15. A like-for-like comparison against the raw-readout cells. `HISTORICAL`
    H1-BASE-v2 and H1-WORKING-v2 were scored on the **10-scene** subset under the
    **uncontrolled** protocol; the treatment's own 10-scene figure is
    `MEASURED` 11.4451 px / 69.97 % at epoch 200. Those cells were not re-run
    here and no numerical delta against them is claimed.
16. One seed. No significance test, none claimed, none claimable.
17. Whether this transfers to a pretrained model. 160 training scenes from random
    initialisation; nothing here transfers.
18. Why the treatment's raw accuracy (8.37 px) still beats the raw-readout cells'
    (~14.3 px). Not measured under a matched protocol.

---

## VERDICT

## `SHIFT-IMPORTANT`

Pre-registered Outcome A: shift removal produced a large degradation in EPE/D1
**and** loss of meaningful stereo behaviour. The H2 readout stabilisation does
not eliminate the importance of the disparity shift mechanism; the shift/search
mechanism is materially useful.

---

## IMPACT ON PROJECT

**What this establishes**

- `DERIVED` — the missing 2×2 cell is filled, and the project's founding question
  — *what is a working cost volume worth?* — has a 200-epoch answer under a
  reproducible protocol: **≈6.1 px EPE and ≈51 D1 points**, on 40/40 scenes, for
  zero parameters and zero MACs (the shift is pad-and-slice).
- `DERIVED` — the H2 result is now correctly attributed. The readout fix
  *unblocked* learning; the shift is what the unblocked network learns *from*.
  Neither component alone yields a stereo model at this budget.
- `DERIVED` — this is the first architecture comparison in Phase 2 made against a
  control with **zero** measured same-config variance, so it needs no materiality
  band. It is also the first whose effect size makes the band question moot.
- `DERIVED` — H3's KILL decision is corroborated on accuracy grounds, by a
  200-epoch measurement, without relying on the epoch-10 stereo criterion the
  project later invalidated. `HISTORICAL` H3's verdict was not inherited; it was
  independently re-derived.

**What this does NOT establish**

- It says nothing about the *number* of candidates, the volume representation, or
  aggregation design. Only shift on/off was varied. **Stage E is not opened by
  this result.**
- It does not show that the current shift implementation is optimal, only that it
  is load-bearing.
- It does not revisit E3b/E3c/O6. Those materiality conclusions **remain
  INDETERMINATE** and were not touched.
- No latency claim. MACs are unchanged and identical between arms; `HISTORICAL`
  Phase 1 measured MAC-to-latency misprediction of 0.67×–142× on this stack.
- Nothing transfers to a pretrained model.

---

## NEXT EXPERIMENT

> **Re-run the E3b block-count series — 6, 5 and 4 refinement blocks — under the
> Stage B deterministic protocol, evaluated on all 40 scenes, using Stage B Run A
> as the 6-block control.** Two new arms (5 and 4 blocks); the 6-block control
> already exists. ≈2.7 GPU-hours.

Why this one, from the evidence: E3b and E3c are the project's only unresolved
verdicts, they are unresolved *specifically* because their materiality band was
unsound, and Stage B has now removed the cause — a controlled arm compared to a
controlled control needs no band at all. This factorial experiment demonstrates
the pattern working end to end on a large effect; the block-count question is the
outstanding small-effect case that the repaired protocol was built to settle.

Explicitly **not** recommended next: any cost-volume/Stage E work (nothing here
implicates the volume's representation), E2b, Scene Flow, or any new
architecture.

**Not executed in this task.**
