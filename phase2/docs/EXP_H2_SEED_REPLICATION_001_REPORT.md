# EXP-H2-SEED-REPLICATION-001 — is H2 a reproducible functional stereo baseline?

**Claim tags:** `MEASURED` (produced here, with its protocol), `DERIVED`
(arithmetic on measured numbers), `OBSERVED VISUALLY` (read off a named image),
`INFERRED` (a reading the evidence does not force), `UNKNOWN`.

Phase 1 untouched (`git diff phase-1-frozen -- src scripts` empty, asserted by a
test). Seed 0's checkpoint hash `e3d48021d7f6a3d4…` unchanged throughout, and
every replication record states the hash it was built against. Full suite: 149
tests pass.

The protocol correction this experiment had to make mid-flight is a separate,
frozen document: `EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md`. Read it
before the numbers below; it is the more important artefact.

---

## 1. Executive conclusion

    H2 IS A REPRODUCIBLE FUNCTIONAL STEREO BASELINE (verdict: REPRODUCIBLE)

`MEASURED` — all three seeds are stereo-functional at 200 epochs under the
criterion frozen before their results existed: destroying the right image costs
**+74.9 to +83.9 D1 points**, destroying the matching map costs **+60.1 to
+69.1**, the matching path receives gradient on **100 %** of all training
batches, softmax entropy stays at **1.73–1.88 nats** (H1's broken regime was
0.0000), and no run produced a NaN, an Inf or a gradient spike.

`MEASURED` — late-window (epochs 150–199) validation D1 across seeds: **24.78,
26.38, 29.89** — a span of **5.11 points**, inside the 10-point band frozen in
advance. Late-window EPE: 4.199, 3.745, 4.333 — span 0.588 px.

`DERIVED` — the H2 mechanism is not an artifact of seed 0. `INFERRED` — the
accuracy spread is real but modest relative to the effect H2 was established on
(56 D1 points against H1). `UNKNOWN` — whether it holds beyond three seeds, one
dataset slice and one budget; no significance is claimed.

**The experiment's most valuable output is not this verdict.** It is the
discovery, forced by control evidence, that **stereo dependence in this recipe is
a late-emerging property**: at epoch 10 *no* seed — including seed 0 — shows any.
A first attempt gated on it at epoch 10 and killed two healthy seeds. §3 records
that failure in full.

## 2. Configuration comparison

`MEASURED` (`config_check.json`, Part A; the check compares this replication's
configuration against the recorded seed-0 configuration field by field).

| Variable | H2 seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| architecture | StereoNet + standardised soft-argmin | identical | identical |
| `cost_volume_shift` | `left` | `left` (asserted at build time) | `left` |
| seed | 0 | **1** | **2** |
| parameters | 423,586 (623,138 per-occurrence) | 423,586 | 423,586 |
| MACs @ 1×3×256×512 | 16.1995 G | 16.1995 G | 16.1995 G |
| training recipe | Adam(0.9,0.999), lr 1e-3, cosine T_max=200, batch 2, 200 epochs, masked smooth-L1, random 256×512 crop + σ=0.1 gain jitter, fp32, `hailo_calib`→`hailo_val` | identical (imported, not restated) | identical |
| checkpoint sha256 | `e3d48021d7f6a3d4…` | `849a371eb54847b6…` | `9528c5b68c89c929…` |

`MEASURED` — the only field-level difference the checker found was the *wording*
of the augmentation description; the augmentation object itself is the same
imported `CroppedKitti` (jitter 0.1, no flip), which the checker now asserts
directly. Part A: **PASS**.

## 3. What went wrong first, and how it was corrected

`MEASURED` — the first attempt gated each seed at epoch 10 on right-image
dependence ≥ +5.0 D1 points. Seeds 1 and 2 passed learning and gradient checks
and failed that one; both aborted at epoch 10 as designed.

`MEASURED` — the criterion had never been calibrated: seed 0's dependence was
measured at 200 epochs. The missing control was then run
(`…-SEED0-REFERENCE`, 10 epochs, gate recorded but not enforced):

| at epoch 10 | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| val EPE, init → epoch 10 | 31.96 → 15.24 | 35.28 → 15.76 | 35.16 → 14.81 |
| matching-path gradient | 100 % | 100 % | 100 % |
| right-image ΔD1 | **−3.33** | **−1.45** | **−2.71** |
| would the epoch-10 gate pass? | **no** | no | no |

`DERIVED` — the gate was invalid: the reference itself fails it. `DERIVED` —
stereo dependence is a **late-emerging property** of this recipe. The correction
was calibrated on the control, never on the seeds' own numbers, frozen in a
pre-registration document, and the aborted records were kept untouched with an
additive `NOTE.md` each.

`MEASURED` — a second question had to be settled before trusting the reruns:
is the divergence between the seed-0 rerun and the original seed-0 run
non-determinism or a harness defect? The harness was run **twice against
itself**, same seed, same code: epoch-0 loss **10.8798 vs 10.5762**, epoch-1
**10.4597 vs 10.0278** — not bit-identical. `DERIVED` — fp32 cuDNN
non-determinism, not a recipe difference; the harness is faithful.

`MEASURED` — same-seed run-to-run divergence compounds: seed 0's original run vs
its rerun differ by +0.10 in train loss at epoch 0 and **+2.44 at epoch 9**
(≈33 % relative). Four independent seed-0 runs span 10.576–10.880 at epoch 0.
`DERIVED` — **no early-epoch difference between seeds can be attributed to the
seed**, and every seed-to-seed claim in this report is bounded by this.

## 4. Viability gate table

| Seed | Preflight | Gradient | 10-epoch learning | Stereo dependence @200 | Visual | Decision |
|---|---|---|---|---|---|---|
| 0 (reference, not retrained) | n/a | 100 % of 16,000 batches | n/a | **+83.9 pt** | structure present | CONTINUE (already complete) |
| 0 (10-epoch control) | PASS | 100 % of 12 crops | 31.96 → 15.24 px | −3.33 pt at epoch 10 (recorded) | n/a | control only |
| 1 | PASS | 100 % of 12 crops | 35.28 → 16.37 px | **+76.8 pt** | structure present | **CONTINUE → completed** |
| 2 | PASS | 100 % of 12 crops | 35.16 → 16.38 px | **+74.9 pt** | structure present | **CONTINUE → completed** |
| 1, 2 (first attempt) | PASS | PASS | PASS | gate invalid — see §3 | n/a | KILLED by a mis-calibrated gate; records kept |

Preflight detail (`preflight_seed{0,1,2}.json`): shift `left`, 423,586
parameters, no NaN/Inf at any stage, softmax entropy at initialisation
2.18–2.33 nats, matching gradient 12/12 crops, no spikes.

## 5. Accuracy table

`MEASURED`, recorded validation protocol (first 10 `hailo_val` scenes, full
368×1232 frames, pooled over `gt > 0`).

| Seed | EPE @200 | D1 @200 | Late-window EPE (150–199) | Late-window D1 | Best EPE (epoch) |
|---|---:|---:|---:|---:|---:|
| 0 | 4.141 | 24.01 % | 4.199 ± 0.090 | 24.783 ± 1.024 | 4.077 (170) |
| 1 | 3.632 | 25.39 % | 3.745 ± 0.145 | 26.384 ± 1.229 | 3.606 (179) |
| 2 | 4.271 | 29.61 % | 4.333 ± 0.092 | 29.892 ± 0.735 | 4.225 (179) |
| **across seeds** | | | **4.092 ± 0.308** (span 0.588) | **27.019 ± 2.613** (span **5.109**) | |

`DERIVED` — the span is inside the frozen 10-point band, so the accuracy arm of
the verdict passes. The ± values are seed-level standard deviations over three
runs; they are descriptive, not a significance test.

## 6. Mechanism table

`MEASURED`, identical probe for every seed (four focus scenes 27/0/31/6;
right-image and matching-map penalties are the **minimum** across variants, i.e.
the weakest case, not the best).

| Seed | Right-image effect | Matching-map effect | Matching gradient | Entropy | `disparity_initial` r(GT) | Final disparity std |
|---|---:|---:|---:|---:|---:|---:|
| 0 | **+83.95 pt** | **+69.12 pt** | 100 % (16,000 batches) | 1.67–1.83 | +0.978…+0.988 | — |
| 1 | **+76.78 pt** | **+67.87 pt** | 100 % (16,000 batches) | 1.732 | +0.930 | 18.73 px |
| 2 | **+74.88 pt** | **+60.05 pt** | 100 % (16,000 batches) | 1.881 | +0.894 | 17.08 px |
| frozen threshold | ≥ +20.0 | ≥ +20.0 | ≥ 95 % | > 0.5 | — | > 1.0 |

`MEASURED` — gradient health: median total norm 20.1 / 24.0 / 22.0, maxima 308 /
604 / 239, **zero** batches above 1e4 in any run, no non-finite values anywhere.
`DERIVED` — none of the H1 pathologies (saturation, tie-driven spikes, dead
matching path) reappeared in any seed.

## 7. Visual findings

`OBSERVED VISUALLY` (`phase2/visualizations/h2_vs_seed1/`,
`…/seed_replication/`, `…/seed_replication_stereo_ablation/`):

- All three seeds produce the same kind of map: graded road surface, resolved
  car silhouettes, vertical poles and sign posts as thin structures, façade
  steps at distinct levels. None is a smooth monocular gradient.
- Per-scene D1 on scene 27: 8.88 / 8.98 / 11.08 %; on scene 6: 11.60 / 12.99 /
  20.65 %. Seed 2 is consistently the weakest of the three, in the images as in
  the metrics.
- The right-image ablation figures are unambiguous: seed 1 on scene 27 goes from
  a coherent scene (D1 9.0 %) to inverted blobs (D1 97.9 %, **+88.9 points**);
  seed 2 on scene 6 goes 20.6 % → 97.8 % (**+77.1 points**). Error maps go from
  mostly dark to saturated.
- `INFERRED` — the visual evidence agrees with the quantitative evidence; no
  seed looks plausible while failing the probes, or vice versa.

## 8. Compute cost

`MEASURED` — seed 1: 3,727 s; seed 2: 3,727 s; seed-0 10-epoch control: ~215 s;
the two aborted 10-epoch runs: ~210 s each; harness self-check: ~90 s. Total
≈ 2.2 GPU-hours on an RTX 4060. Seed 0's original 3,627 s is comparable
(its own record's 12,517 s is inflated by a machine suspend, documented in the
H2 report).

## 9. Anomalies

1. **The invalidated epoch-10 gate** (§3) — the significant one. Two healthy
   seeds were killed by an uncalibrated criterion; the records are kept.
2. **The harness is not bit-reproducible** (§3). Consequence: same-seed variance
   is non-zero and compounds, so it bounds every seed-to-seed claim here.
3. **Part A's augmentation check** initially failed on wording rather than
   substance; the checker was changed to assert the shared `CroppedKitti` object
   instead of comparing prose.
4. **A test read an in-progress record**: `_replication_records()` picked up a
   record directory before its `metrics.json` was written. Fixed to require
   completion.
5. Seed 0's own late-window figures come from its original run, whose
   `wall_clock_s` is inflated by a machine suspend (recorded in the H2 report).

## 10. Limitations

- Three seeds, one run each. **No statistical significance is claimed** and none
  can be, particularly with same-seed variance unmeasured at 200 epochs.
- The 10-point reproducibility band and the +20-point stereo threshold are
  **stated conventions frozen in advance**, not tests derived from a noise model.
- 160 training scenes, 40 validation scenes, one crop size, one budget. Nothing
  here says how H2 behaves at full data scale.
- Ablation penalties are measured on four focus scenes chosen in earlier work
  for their H1 behaviour, not sampled at random.
- `UNKNOWN` — the epoch at which stereo dependence emerges. It is absent at 10
  and present at 200 in every seed; nothing in between was measured. §11 makes
  this the next experiment.

## 11. Final answer and next experiment

**Is H2 a reproducible functional stereo baseline?**

    REPRODUCIBLE

Mechanism reproduces without qualification: three of three seeds are
stereo-functional by a factor of ~3–4 above the frozen threshold, with identical
gradient health and no H1 pathologies. Accuracy reproduces within the band fixed
in advance (D1 span 5.11 of an allowed 10.0), with seed 2 the weakest and seed 1
the strongest.

    NEXT EXPERIMENT:
        EXP-H2-EMERGENCE-001 -- when does stereo dependence appear? Re-run the
        stereo ablation on the weight snapshots this experiment already saved
        (epochs 10, 20, 50, 100, 150, 200 for seeds 1 and 2), producing a
        dependence-vs-epoch curve. No training: the checkpoints exist.

Why this one: the experiment's own failure identified an unmeasured property
that every future gate depends on — this project cannot cheaply test a new
architecture for "stereo functionality" until it knows when that property is
supposed to appear. It costs minutes, not GPU-hours, and it converts the
accident that killed two seeds into a calibrated instrument.

Stage E remains **closed**.
