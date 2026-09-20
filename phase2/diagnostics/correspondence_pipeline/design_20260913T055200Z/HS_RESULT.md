# `EXP-CORRESPONDENCE-ZONE-001` — pre-freeze hard stops

**Script:** `hs_zone.py` (`--stage 1`, `--stage 2`) -> `hs_stage1.json`, `hs_stage2.json`.
**Model-free.** No checkpoint opened, no training, no optimizer, no historical
record written. Randomly initialised `FeatureExtractor` (seeds 11/12/13) used only
for receptive-field **support** questions, which are weight-independent.
CPU, single thread, `use_deterministic_algorithms(True)`.

## VERDICT: all four PASS

| Check | Result |
|---|---|
| `HS-SLACK` | **PASS.** Frames all `(368, 1232, 3)`; right-crop origins `[0,16,32,48,64,80,96]` for `Δ ∈ {0..6}` all in range; **zero all-zero rows and all-zero columns** in all 12 scenes, so no `pad_and_crop` fill enters any crop. |
| `HS-GT` | **PASS.** GT-valid pixels in every scene, both interiors. Worst case `2,222` (zone) / `7,982` (complement); typical 12k–22k / 8k–18k. |
| `HS-FRAME1` | **PASS, unanimous 12/12.** Convention is `right[y,x] ≈ left[y,x+d]`: MAE 7.9–21.1 for `x+d` versus 27.7–60.0 for `x−d` and 24.4–46.9 for `x` (no shift), ~1.1M GT pixels total. Sign pinned: `x_R = x_L + 16Δ` gives `d' = d + 16Δ`, i.e. candidate `k' = d/16 + Δ`. |
| `HS-FRAME2` | **PASS.** Synthetic feature-level anchor: `argmin_k = τ` for `τ ∈ {1,3,5}`, profile minimum **exactly 0.0** against a second-lowest of 1.118–1.121. |
| `HS-Z` | **PASS, and exactly.** Zone-interior features (cells `[0,16)`) are **bit-identical** to the fully-shifted image, and complement-interior features (cells `[56,71)`) **bit-identical** to the unshifted image — `0.0` on all three extractors — while the arms differ elsewhere by 0.28–0.38. Seam blindness is **measured, not derived**. |
| `HS-SYM` | **PASS.** `min_s max|V_zone[k−s] − V_none[k]| = 0.379` in the zone interior (no global candidate-axis translate reproduces it) and **exactly `0.0`** in the complement interior (the complement is untouched, `s = 0` reproduces it bit-for-bit). |

## Amendment forced by the measurements

**Offset direction reversed.** Run A used `x_L = 96`, `x_R = 96 − 16Δ`, i.e.
`d' = d − 16Δ`. `HS-GT` then measured the actual disparity distribution in these
crops: **p5–p95 ≈ 0.5–7.5 candidates**, mostly 0.5–4. Reducing disparity drives
almost every GT pixel below candidate 0 at `Δ = 1`, so the treatment would have
been pure saturation. Amended to `x_L = 0`, `x_R = +16Δ`, `d' = d + 16Δ`, which
sweeps predictions **up through** the candidate window. Recorded, not silently
changed; `DESIGN.md`'s `Δ ∈ [−6,+6]` is also corrected to one-sided `Δ ∈ {0..6}` —
the 96 px slack is 6 candidates **total**, so a two-sided set would have been
`±3`, and the original record was wrong on that point.

Consequence to carry into the preregistration: baseline (`Δ=0`) disparities sit low
in the window, near the padding-contaminated candidates (`k ∈ {5,6}` are the only
padding-free ones), and the Δ sweep moves them through the clean region. `S` is a
difference of slopes between two regions with similar `d` distributions, so
padding largely cancels, but it does not cancel exactly. Residual, declared.

## Limitation of `HS-SYM` as implemented

The scan covered candidate-axis shifts `s ∈ [−6,6]` **at fixed `w`**. It did not
scan the *combined* (candidate shift, spatial shift) group, and a **global** right
shift is exactly such a combination:
`V^Δ[k,w] = Lf[w+k] − Rf[w+Δ] = V[k−Δ, w+Δ]` — the GEOM-002 reparametrisation.
So the scan alone does not establish non-invariance.

The assertion holds for a stronger, measured reason: the zone intervention changes
the zone while leaving the complement **bit-identically unchanged**
(`HS-Z` and `HS-SYM` complement `= 0.0`), whereas any global reparametrisation acts
on both regions equally. And the primary statistic is a **difference between the
two regions**, so any global group element cancels in `S` identically. That is why
the global-Δ arm is only a positive control and never evidence.

## State

Hard stops complete and passing. Nothing has been executed on trained weights.
Next actions, in order: (1) write the preregistration from these measured values;
(2) Stage-1 **null arms only** — the `shift="none"` deployed checkpoint and the
swap control — and freeze the gate threshold from their measured spread;
(3) Stage-2 trained `shift="left"` arms. `HS-POSCTRL` (global Δ must give
`β ≈ −1`) runs in Stage 2 and gates interpretation.

Claim ceiling unchanged: **Level D — genuine geometric correspondence
NOT-DEMONSTRATED.**
