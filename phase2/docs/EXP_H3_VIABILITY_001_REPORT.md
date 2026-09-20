# EXP-H3-VIABILITY-001 — is `cost_volume_shift="none"` worth expensive training?

**Purpose.** A cheap gate, not an accuracy experiment. H2 established a
functioning stereo pathway and is the control. H3 removes the disparity shift
and nothing else. The question is only whether H3 deserves a 200-epoch run.

**Claim tags:** `MEASURED` (produced here, with its protocol), `DERIVED`
(arithmetic on measured numbers), `OBSERVED VISUALLY` (read off a named image),
`INFERRED` (a reading the evidence does not force), `UNKNOWN`.

Phase 1 is untouched (`git diff phase-1-frozen -- src scripts` is empty; a test
asserts it). The H1 and H2 records and checkpoints are untouched; the H2
checkpoint hash `e3d48021d7f6a3d4…` is identical before and after every stage
here.

Evidence: `phase2/results/h3_viability/{preflight,ablation,pooled,figures}.json`,
record `phase2/experiments/EXP-H3-VIABILITY-001/`, figures under
`phase2/visualizations/h2_vs_h3/`.

---

## 1. The one changed variable

| | H2 (control) | H3 |
|---|---|---|
| `cost_volume_shift` | `"left"` | **`"none"`** |
| regression stage | standardised soft-argmin | same |
| feature extractor / aggregation / refinement | same | same |
| loss / optimizer / lr / cosine schedule (`T_max=200`) | same | same |
| seed / dataset / split / crop / augmentation / batch | same | same |
| parameters | 423,586 | 423,586 |

`MEASURED` — built from the same seed, the two configurations produce
bit-identical initial weights and identical module types; only
`cost_volume.shift` differs (test
`test_h3_differs_from_h2_only_in_the_cost_volume_shift`). The recipe is imported
from the H2 and H1 scripts rather than restated.

`MEASURED` — the cosine schedule keeps `T_max = 200`, so H3's ten epochs are the
control's **first ten epochs**, not a compressed ten-epoch schedule.

---

## 2. Part A — does the complete LEFT vs NONE pipeline actually differ?

Protocol: the H2 checkpoint's weights loaded into two complete models differing
only in the shift; identical inputs; the full forward including regression and
refinement (not the old cross-shift probe, which rebuilt models without their
regression stage — that defect is recorded in
`phase2/results/h2_mechanism/README.md`).

`MEASURED`, all four focus scenes, max absolute difference between arms:

| stage | differs on every scene | scene 27 | scene 6 |
|---|---|---:|---:|
| left features | **no** (identical) | 0 | 0 |
| right features | **no** (identical) | 0 | 0 |
| cost volume | yes | 3.67e+09 | 3.64e+09 |
| aggregated cost | yes | 2.93e+12 | 2.99e+12 |
| softmax input (standardised) | yes | 4.85 | 4.23 |
| `disparity_initial` | yes | 7.55 | 7.62 |
| refinement residual | yes | 63.8 | 59.4 |
| final disparity | yes | 69.7 | 66.1 |

`DERIVED` — features are identical because the shift acts only on volume
construction; everything downstream differs by large margins. The pipeline
distinguishes the two settings.

`MEASURED` — no NaN, no Inf, no shape mismatch anywhere; the H2 checkpoint hash
is unchanged; both arms' weights equal the checkpoint's bit for bit.

---

## 3. Part B — is the control still stereo-functional?

`MEASURED` (H2 weights, scene 27; the other three scenes agree within a few
points):

| test | D1 | ΔD1 vs baseline | mean abs change |
|---|---:|---:|---:|
| baseline | 8.88 | — | — |
| right image = black | 99.34 | **+90.46** | 41.10 px |
| right image = noise | 99.81 | **+90.93** | 35.68 px |
| right image = the left image | 95.41 | **+86.53** | 15.79 px |
| `disparity_initial` := its own mean | 84.28 | **+75.41** | 13.17 px |
| `disparity_initial` := shuffled | 84.78 | **+75.91** | 13.40 px |

`MEASURED` — matching-path gradient present on **12/12** crops for both the
`left` and `none` arms; softmax entropy 1.71 nats for both; no non-finite loss.

`DERIVED` — the control is still the functional stereo model H2 reported, so the
comparison is functional-vs-H3 and not a return to the broken H1 situation.
**Gate: PASS.**

---

## 4. Parts C/D — the 10-epoch run, complete progression

`MEASURED` (`EXP-H3-VIABILITY-001`; 800 batches, 215 s wall clock, no abort).
This run validates *before* each listed epoch, so "epoch 0" is the untrained
initialisation.

| epoch | train loss | val loss | val EPE | val D1 | softmax entropy | init std | init r(GT) | final r(GT) | matching grad |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | — | 31.095 | 31.992 | 95.21 % | 1.541 | 0.272 | +0.173 | +0.158 | — |
| 1 | 10.373 | 16.035 | 16.919 | 87.97 % | 2.207 | 1.072 | +0.585 | +0.502 | 100 % |
| 2 | 10.122 | 14.637 | 15.428 | 87.77 % | 2.111 | 1.628 | +0.583 | +0.574 | 100 % |
| 5 | 8.533 | 20.087 | 20.878 | 91.67 % | 2.207 | 1.485 | +0.590 | +0.576 | 100 % |
| 10 | 8.113 | 15.779 | 16.532 | 88.71 % | 2.120 | 1.827 | +0.616 | +0.605 | 100 % |

`MEASURED` — gradients over all 800 batches: total median 45.5, max 313, zero
batches above 1e4, no non-finite values; matching-path gradient present on
**800/800** batches (median 18.6).

`DERIVED` — H3 trains, is numerically stable, and learns: validation EPE falls
from 31.99 (initialisation) to 15.4–16.5, train loss 10.37 → 8.11. The
validation curve is not monotone (15.4 at epoch 2, 20.9 at epoch 5, 16.5 at
epoch 10).

### Matched-budget comparison with the control

Both runs share seed, recipe and schedule, so their early epochs are directly
comparable. The control validated at the *end* of each epoch; this run validates
*before*, so the aligned pairs are stated explicitly.

| checkpoint | val EPE | val D1 | init std | init r(GT) |
|---|---:|---:|---:|---:|
| H2 after 1 epoch | 17.185 | 88.26 % | 1.483 | +0.239 |
| H3 after 1 epoch | 16.919 | 87.97 % | 1.072 | +0.585 |
| H2 after 6 epochs | 16.511 | 88.44 % | 1.780 | +0.591 |
| H3 after 5 epochs | 20.878 | 91.67 % | 1.485 | +0.590 |
| **H2 after 11 epochs** | **10.535** | **78.55 %** | 2.502 | **+0.738** |
| **H3 after 10 epochs** | **16.532** | **88.71 %** | 1.827 | **+0.616** |

`DERIVED` — the two are indistinguishable after one epoch. By ten/eleven epochs
the control has pulled away (−6.0 px EPE, −10.2 D1 points) and its matching
output keeps improving (+0.239 → +0.591 → +0.738) while H3's plateaus after the
first epoch (+0.585 → +0.590 → +0.616).

---

## 5. Does H3 use the second camera?

`MEASURED` (`phase2/results/h3_viability/ablation.json`, Phase 1's EXP-007
variant set, 10 scenes):

| right image replaced by | H2 ΔD1 | **H3 ΔD1** | H3 mean abs change |
|---|---:|---:|---:|
| the left image | +80.82 | **+1.91** | 3.14 px |
| another scene's right image | +82.51 | **+0.46** | 2.00 px |
| horizontally flipped | +82.23 | **−1.08** | 1.90 px |
| black | +84.36 | **+0.46** | 1.62 px |
| uniform noise | +84.16 | **+0.40** | 1.58 px |

`MEASURED` (`pooled.json`, 40 scenes, substituting only `disparity_initial`):

| variant | EPE | D1 |
|---|---:|---:|
| H3 as trained | 12.804 | 88.22 |
| H3, `disparity_initial` := its own mean | 16.780 | **87.75** |
| H3, `disparity_initial` := shuffled | 16.912 | **88.15** |
| H3, `disparity_initial` := constant 11 | 14.203 | **87.08** |
| H2 as trained | 2.400 | 16.72 |

`DERIVED` — corrupting the right image changes H3's D1 by at most 1.9 points and
its prediction by ~2–3 px on a ~25 px mean disparity. Destroying H3's matching
map does not hurt its D1 at all — every substitution scores *better* by
0.07–1.14 points (while its EPE worsens, so the map does carry something the
threshold metric does not reward). This is the H1 signature, measured on a model
whose gradient pathway is fully alive.

`INFERRED` — with `shift="none"` there is no disparity search to learn: the
volume is the zero-shift difference of the two feature maps, so the only stereo
cue available is photometric, and the network reverts to predicting layout from
the left image. **Not established:** that H3 could never develop right-image
dependence with far more epochs. This gate cannot answer that.

---

## 6. Part E — visual validation (existing visualizer only)

`OBSERVED VISUALLY` (`phase2/visualizations/h2_vs_h3/`, scenes 27, 0, 31, 6;
`*_stages.png` and each scene's `comparison.png`):

- H3's final disparity is a smooth top-to-bottom gradient — far above, near on
  the road — with faint intensity echoes of the left image. **No** vehicle
  silhouettes, **no** poles or sign posts, **no** façade steps.
- H3's `disparity_initial` is the same smooth layout at coarse resolution.
- H2's maps show the parked car, the lamp post and building façades at distinct
  levels.
- H3's error map is broadly above the 6 px display cap over most of the frame;
  H2's is mostly below it.
- **No constant-disparity collapse:** H3's map spans 6.64–50.07 px with
  per-scene standard deviation 8.7 px (test
  `test_h3_predictions_are_finite_and_not_a_constant`).
- Per-scene D1 at these settings: H2 7.7–11.6 %, H3 79.7–86.8 %.

`INFERRED` — H3's output looks like a monocular layout prior, not a stereo
reconstruction. **Caveat, stated rather than buried:** this compares H3 at 10
epochs with H2 at 200. The visual comparison is confounded by budget and is
*not* the basis of the verdict; the matched-budget numbers in §4 and the
stereo-dependence tests in §5 are.

---

## 7. Anomalies and limitations

1. **No 10-epoch H2 checkpoint exists**, so H2's own right-image dependence at
   matched budget could not be measured — only its validation metrics and stage
   statistics, which are recorded. Removing this caveat costs one 3.5-minute
   rerun of H2 at 10 epochs; it was not run here because the matched-budget
   metrics in §4 already separate the arms and compute discipline applies.
2. One seed, 10 epochs, 160 training scenes, 40 validation scenes. No
   significance test was run and none is claimed.
3. The record has no top-level `cost_volume_shift` key (it is in
   `changed_variable`); an additive `NOTE.md` sits beside it, and it also
   records the checkpoint-alignment difference described in §4.
4. `recorded_global_metrics` in the visualizer prints `N/A` for H3 because this
   record stores per-checkpoint metrics instead of a `validation_epe_series`.
   The tool reports it as unavailable rather than fabricating a number.
5. One tooling defect was found and fixed during this task: the visualizer's
   comparison figure had "BASE vs WORKING, H1-v2 checkpoints" hardcoded in its
   title, which mislabelled every H2/H3 figure. Fixed; figures regenerated.

---

## 8. Gate table

| Gate | Result | Evidence | Action |
|---|---|---|---|
| Preflight (complete LEFT vs NONE differs) | **PASS** | features identical; cost volume 3.7e9, aggregated cost 2.9e12, final disparity 69.7 max abs difference; no NaN/Inf; no shape mismatch; H2 hash unchanged | proceeded to Part B |
| Stereo dependence of the control | **PASS** | H2 right-image corruption +86.5…+90.9 D1 pt; matching-map destruction +75.4 pt | control confirmed functional |
| Matching gradient (both arms) | **PASS** | 12/12 preflight crops; 800/800 H3 training batches; entropy 1.7–2.2 nats; zero non-finite | proceeded to Part C |
| 10-epoch learning | **PASS** | val EPE 31.99 → 15.4–16.5; train loss 10.37 → 8.11; no abort; no gradient above 1e4 | progression recorded in full |
| Visual validation | **PASS (not collapsed) / FAIL (no stereo structure)** | span 6.64–50.07 px, std 8.7 px — not constant; but no vehicles, poles or façades in either `disparity_initial` or the final map | see verdict |
| **H3 viability** | **KILL** | right-image influence ≤ 1.9 D1 pt (control: +80.8…+84.4); destroying the matching map *improves* D1 by 0.07–1.14 pt; matched-budget D1 88.71 vs control 78.55 | do not spend 200 epochs on `shift="none"` |

**Kill criterion triggered, as pre-registered:** *"right image has negligible
influence."* No other kill criterion fired — there was no NaN/Inf, the matching
gradient was never zero, disparity did not collapse to a constant, training was
stable, and validation did improve.

---

## 9. Answers

1. **Is H3 actually functional?** `MEASURED` mechanically yes — it trains, is
   numerically stable, and its matching path receives gradient on every batch.
   As a *stereo* model, no: it does not depend on the second camera.
2. **Is it learning?** `MEASURED` yes: validation EPE 31.99 → 15.4–16.5 and
   train loss 10.37 → 8.11 in ten epochs, with a non-monotone validation curve.
3. **Does it use both cameras?** `MEASURED` no. Corrupting the right image moves
   D1 by −1.1 to +1.9 points; for the control the same tests move it by +80.8 to
   +84.4.
4. **Does the matching pathway learn?** `MEASURED` gradient reaches it on
   800/800 batches, and `disparity_initial` reaches r(GT) +0.616 — but that
   correlation is flat from epoch 1 (+0.585) while the control's climbs to
   +0.738 by epoch 11, and destroying the map costs H3 nothing in D1.
   `DERIVED` — the pathway is alive but is not learning correspondence.
5. **Does it look geometrically meaningful?** `OBSERVED VISUALLY` partially: a
   plausible smooth ground-plane gradient, no object structure, no thin
   structures, no disparity discontinuities. Not collapsed, not stereo.
6. **Better/worse/equivalent to H2 at this stage?** `MEASURED` worse at matched
   budget: after 10–11 epochs, EPE 16.53 vs 10.54 and D1 88.71 % vs 78.55 %.
   Equivalent after one epoch (16.92 vs 17.19).
7. **Worth serious compute?** `DERIVED` no. Its distinguishing feature — the
   removed shift — is exactly what makes it non-stereo, and it is behind the
   control on the metric while showing none of the control's mechanistic
   stereo signatures.
8. **Single next experiment?** See below.

---

## 10. Decision

    H3 STATUS: KILL

    Meaning: `cost_volume_shift="none"` does not warrant a 200-epoch run. It is
    not a failed experiment — it answered the question it was asked, cheaply, in
    215 seconds of training: with the soft-argmin fixed, the disparity shift is
    what carries the stereo signal, and removing it returns the model to
    monocular behaviour.

    NEXT EXPERIMENT:
        EXP-H2-SEED-REPLICATION-001 -- rerun EXP-H2-SOFTARGMIN-SCALE unchanged
        at seed 1 (one run, ~1 h, the recipe already scripted). Every
        architectural decision from here rests on H2, and H2 is currently a
        single-seed result; H1's history in this project is precisely the story
        of a conclusion drawn from an unreplicated, mechanistically
        misunderstood run.

Why this and not something more ambitious: H3 has now closed the one open
question about the *shift*. What remains unverified is the foundation itself.
Stage E stays **closed** until H2 replicates.
