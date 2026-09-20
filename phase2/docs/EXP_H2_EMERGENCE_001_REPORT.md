# EXP-H2-EMERGENCE-001 — when does stereo dependence appear during training?

**Status:** complete. **Date:** 2026-09-08. **Compute:** 42 s on the RTX 4060,
no training. **Records:** `phase2/experiments/EXP-H2-EMERGENCE-001-RUN2/`
(authoritative) and `phase2/experiments/EXP-H2-EMERGENCE-001/` (superseded, see
its `NOTE.md`). Raw data: `phase2/results/EXP-H2-EMERGENCE-001/emergence-run2.json`.
Script: `phase2/scripts/exp_h2_emergence.py`.

## Question

`EXP-H2-SEED-REPLICATION-001` established that seed 0 shows **no** right-image
dependence at epoch 10 (-3.33 D1 points) yet **+83.95 points** at epoch 200 —
which is why its original epoch-10 stereo gate was invalid. That left the shape
of the curve between those endpoints unmeasured. This experiment measures it.

## Method

Measurement only. No training, no architecture change, no new recipe. The
weight snapshots saved by the seed-replication runs (epochs 10/20/50/100/150/200,
seeds 1 and 2) are re-loaded and put through `stereo_probe` and `gradient_probe`
**imported unchanged** from `exp_h2_seed_replication.py`, on the same four
`FOCUS_SCENES` (27, 0, 31, 6).

The criterion was **not invented here**. It is `GATE_MIN_STEREO_D1_POINTS`,
imported from the seed-replication module and frozen in
`EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md`: an epoch counts as
stereo-dependent when the **worst case** over 3 right-image corruptions × 4
scenes costs **≥ 20 D1 points**. Because the snapshot grid is coarse, the answer
is reported as an interval between two saved epochs, never as a precise epoch.

Validation EPE/D1 are read from the source runs' recorded `metrics.json`, not
recomputed.

**Deviation, recorded not hidden:** during training the ablation drew from one
`np.random.default_rng(0)` advancing across epochs; here each checkpoint gets a
fresh `default_rng(0)`, so all twelve measurements are mutually comparable. The
cross-check below shows this changes nothing, because the worst case is always
set by a deterministic corruption (black or right-equals-left), never by the
random-noise one.

## Result

| seed | epoch | val EPE | val D1 % | right-image dep. worst / mean (pt) | matching-map dep. worst (pt) | softmax entropy | matching grad. | grad max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 16.374 | 85.86 | -1.26 / +3.96 | +2.09 | 2.061 | 100% | 95.03 |
| 1 | 20 | 8.552 | 64.14 | +32.35 / +41.51 | +26.78 | 1.682 | 100% | 73.94 |
| 1 | 50 | 5.449 | 39.60 | +58.68 / +71.33 | +48.69 | 1.703 | 100% | 52.84 |
| 1 | 100 | 4.097 | 30.39 | +73.43 / +80.92 | +62.16 | 1.669 | 100% | 36.66 |
| 1 | 150 | 3.766 | 27.09 | +78.08 / +84.67 | +66.71 | 1.669 | 100% | 32.86 |
| 1 | 200 | 3.632 | 25.39 | +76.78 / +84.50 | +67.78 | 1.670 | 100% | 29.95 |
| 2 | 10 | 16.383 | 87.09 | -2.41 / +1.25 | -6.39 | 2.006 | 100% | 110.4 |
| 2 | 20 | 14.029 | 83.76 | -0.91 / +4.65 | +0.52 | 1.949 | 100% | 80.82 |
| 2 | 50 | 7.419 | 57.52 | +30.50 / +40.82 | +24.00 | 1.706 | 100% | 76.31 |
| 2 | 100 | 5.270 | 37.67 | +65.06 / +76.64 | +51.77 | 1.761 | 100% | 43.79 |
| 2 | 150 | 4.746 | 31.75 | +75.35 / +82.19 | +60.83 | 1.777 | 100% | 35.2 |
| 2 | 200 | 4.271 | 29.61 | +74.88 / +83.50 | +59.96 | 1.794 | 100% | 45.21 |

**Emergence, against the frozen 20-point criterion:**

| seed | first snapshot meeting it | interval | holds at every later epoch |
|---|---:|---|---|
| 1 | epoch 20 | (10, 20] | yes |
| 2 | epoch 50 | (20, 50] | yes |

Seed 0 is not on this grid — only its epoch-10 and epoch-200 weights were kept —
so its two recorded points are cited, not re-measured: **-3.33 pt at epoch 10,
+83.95 pt at epoch 200**.

## Cross-check against the recorded training-time ablation

Epochs 10, 100 and 200 were also ablated during training. Re-measuring them from
the snapshots reproduces the recorded worst-case right-image penalty **exactly**
— difference 0.00 points at all six (seed, epoch) pairs. The snapshots are
faithful and the probe is deterministic where it matters.

## What this says

1. **Stereo dependence is genuinely late-emerging, and its onset varies by
   seed** — between epochs 10 and 20 for seed 1, between 20 and 50 for seed 2,
   on a 200-epoch schedule. The seed-replication protocol correction was
   right for the right reason.
2. **Matching-path gradient is not a proxy for stereo function.** It is at
   **100 % of batches at every single checkpoint**, including seed 2 at epoch 20,
   where corrupting the right image *improves* D1 by 0.91 points. A model can
   receive gradient through the matching path for tens of epochs while making
   no use of the second camera. Any future gate must test the behaviour, not
   the gradient.
3. **Onset tracks achieved accuracy more closely than epoch count.** Aligning
   the two seeds by val D1 rather than by epoch lines them up almost exactly:
   64.14 % → +32.35 pt (seed 1, epoch 20) against 57.52 % → +30.50 pt (seed 2,
   epoch 50); 30.39 % → +73.43 pt against 37.67 % → +65.06 pt; 27.09 % → +78.08 pt
   against 31.75 % → +75.35 pt. Seed 2 is simply about one grid step slower.
   **Observation on two seeds, not a law** — this is exactly the kind of claim
   the harness's fp32 non-determinism (documented in the seed-replication
   report) forbids stating more strongly.
4. **Softmax entropy falls as dependence rises** (≈2.0 nats before onset,
   ≈1.67–1.79 after) and then stays put. The regression stage sharpens as the
   matching signal starts being used, and it never approaches H1's collapsed
   0.0000 nats.
5. Dependence **saturates rather than keeps climbing**: both seeds are within
   ~1.5 points at epochs 150 and 200 (seed 1 dips 78.08 → 76.78, seed 2 dips
   75.35 → 74.88), so the last 50 epochs buy accuracy, not stereo reliance.

## Limits

Two seeds, four probe scenes, and a six-point epoch grid; the emergence point is
an interval between saved epochs, not a measured epoch. Seed 0 contributes only
its two previously recorded endpoints. The harness is not bit-reproducible
(fp32 cuDNN), which bounds any seed-to-seed comparison — including point 3.
No p-value is claimed anywhere.
