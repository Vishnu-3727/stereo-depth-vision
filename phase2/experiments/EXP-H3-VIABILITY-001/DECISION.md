# Decision (additive; the record itself is unmodified)

    H3 STATUS: KILL

Analysis and full evidence: `phase2/docs/EXP_H3_VIABILITY_001_REPORT.md`.

**Pre-registered kill criterion triggered:** *the right image has negligible
influence.* Corrupting it (black, noise, flipped, another scene, or the left
image itself) moves this model's D1 by −1.08 to +1.91 points and its prediction
by 1.6–3.1 px; the same tests move the control `EXP-H2-SOFTARGMIN-SCALE` by
+80.8 to +84.4 points. Destroying its matching map does not hurt its D1 either
(every substitution scores 0.07–1.14 points *better*).

**No other kill criterion fired.** There was no NaN or Inf, the matching-path
gradient was present on 800/800 batches, disparity did not collapse to a
constant (span 6.64–50.07 px, per-scene std 8.7 px), training was stable (max
gradient norm 313, none above 1e4), and validation did improve (EPE 31.99 →
15.4–16.5, train loss 10.37 → 8.11).

**Matched-budget comparison against the control** (same seed, same recipe, same
cosine schedule with `T_max=200`): after 10–11 epochs, H3 val EPE 16.53 / D1
88.71 % against the control's 10.54 / 78.55 %; H3's `disparity_initial`
correlation with ground truth is flat from epoch 1 (+0.585 → +0.616) while the
control's climbs (+0.239 → +0.591 → +0.738).

**What this does and does not establish.** It establishes that under the H2
architecture, with the soft-argmin no longer saturated, the disparity shift is
what carries the stereo signal, and that a no-shift volume is not worth a
200-epoch run. It does **not** establish that no amount of training would give
this configuration right-image dependence; that was not tested and is UNKNOWN.

**Next experiment:** `EXP-H2-SEED-REPLICATION-001` — rerun
`EXP-H2-SOFTARGMIN-SCALE` unchanged at seed 1. Stage E stays closed.
