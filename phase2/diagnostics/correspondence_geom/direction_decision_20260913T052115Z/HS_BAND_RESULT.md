# HS-BAND result — `EXP-CORRESPONDENCE-CP-001` pre-freeze hard stop

**Run:** `hs_band.py` -> `hs_band.json`, this directory. Model-free: no checkpoint
opened, no training, no optimizer imported, no historical record written.
Random uniform content (conservative: real crops can only add accidental zeros),
3 randomly initialised `FeatureExtractor` seeds (11/12/13), one randomly
initialised `Aggregation` (21), frozen `build_cost_volume`, CPU, single thread,
`use_deterministic_algorithms(True)`. Recompute check: aggregated cost is
bit-reproducible (`max diff 0.000e+00`).

## Verdict

**PASS on the letter of the stated conditions. FAIL on their intent.
Recommendation: treat as FAIL and close the branch. CP-001 must NOT be frozen.**

## What passed

Measured receptive support, single-cell edit at cell 35, identical for all three
extractor seeds: affected feature columns **[21, 50]** — left reach **14** cells,
right reach **15** cells. This matches the independently measured 15/14-cell halo
already on record, and matches the analytic radius exactly: 13 stacked 3x3 convs
after the downsample stack (12 in the 6 ResBlocks + `output_conv`) give a 13-cell
radius at stride 16 = 208 px, plus a 30 px radius through the four 5x5/stride-2
convs = **238 px one-sided, 477 px total**. So the exact zero in C1 is a
**structural support property, not a float32 underflow**.

Admissible offsets `delta` (band 3 cells wide, tested column `w_test = 30`):

    right side  delta in {15, 16, 17, 18, 19}      (nearest band edge 15..19 cells away)
    left  side  delta in {-22, -21, -20, -19, -18} (nearest band edge 16..20 cells away)

10 offsets, 5 per side. `C1` (`max|V1[...,w_test] - V0[...,w_test]| == 0.0`
exactly) and `C2` (features differ somewhere within the 5-cell aggregation reach
**and** the aggregated cost at `w_test` differs) both hold, all three seeds. The
derived prediction in `DECISION.md` (`|delta| in [15,19]`) was correct on the
right side; the left side sits at 16..20 by nearest edge, from the measured 14/15
asymmetry.

## Why it fails in substance — the decisive number

Scale of the tested column's aggregated cost: `mean|c| = 1.31e-02`,
**`std across the 12 candidates = 3.09e-03`** — and that std is exactly what the
frozen `scaled_regression` z-score divides by, so it is the scale the soft-argmin
actually reads. One float32 ULP at this magnitude is `1.57e-09`.

| `delta` | `d_V_test` | `d_agg_test` | in ULPs | **as a fraction of `std_k`** |
|---|---|---|---|---|
| 15 | 0.0 | 1.68e-08 | 10.7 | **5.4e-06** |
| 16 | 0.0 | 1.30e-08 | 8.3 | **4.2e-06** |
| 19 | 0.0 | 9.31e-09 | 6.0 | **3.0e-06** |
| 5 | 5.96e-04 | 8.49e-05 | 54,201 | 2.8e-02 |
| 0 | 3.91e-02 | 5.81e-04 | 371,007 | 1.9e-01 |

**In the entire admissible band the context perturbs the tested pixel's cost
profile by ~5e-06 of the scale the readout reads — about six orders of magnitude
too small.** The implied movement of the predicted candidate is ~1e-05 candidates
against a measured soft-argmin blur of 0.15-1.35 candidates.

The cause is a structural mismatch, not bad luck: the extractor's support
boundary (15 cells) and the region where its influence is *numerically
meaningful* (about 5 cells, from the table above) are ~10 cells apart, while the
aggregation's spatial reach is only 5 cells. Any offset with a usable signal
(`|delta| <= ~5`, effect 1e-2..1e-1 of `std_k`) has a **contaminated** tested-pixel
slice, which destroys the analytic pointwise-latch null — the one property that
made CP-001 stronger than O1. The two requirements are satisfiable together only
at round-off magnitude.

This is not a power problem that more scenes, seeds or checkpoints can fix. It is
a magnitude problem below the float32 resolution of the readout itself.

## Two further defects found while auditing

1. **C1 as stated is insufficient.** The frozen readout upsamples the cost tensor
   bilinearly (`align_corners=True`, `70x/1135`), so every full-resolution pixel
   draws from **two adjacent feature columns**. The analytic null therefore
   requires `V` to be bit-identical at `w_test` **and at `w_test +/- 1`**, not at
   `w_test` alone. Correct condition `C1'` needs nearest-edge distance >= 16
   (right) / >= 17 (left), which narrows the band to 4 offsets per side and makes
   the magnitudes *smaller still* (8.3 -> 6.0 ULPs). HS-BAND as run did not test
   `C1'`; it does not matter, because the substantive failure is already decisive.
2. **Cosmetic bug in the record:** the `covers_w_test` flag in `hs_band.py`
   (`if 0 <= delta + BAND_CELLS - 1 and delta <= 0`) does not correctly detect
   band overlap with `w_test`. It is reported in `hs_band.json` and is **not**
   used in the PASS computation. Recorded, not silently fixed.

## What this does and does not establish

* It **does** establish that the CP-001 premise, as designed, is unusable: no
  offset simultaneously gives a bit-identical tested-pixel slice and a
  context effect above readout resolution.
* It **does not** say anything about whether the trained network performs
  correspondence. Non-identifiability and now non-executability are properties of
  the available interventions, not of the model.

**Claim ceiling unchanged: Level D — genuine geometric correspondence
NOT-DEMONSTRATED.** Not demonstrated is not absent.
