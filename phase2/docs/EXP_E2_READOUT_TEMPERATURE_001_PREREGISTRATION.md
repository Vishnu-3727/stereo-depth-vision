# EXP-E2-READOUT-TEMPERATURE-001 — pre-registration, frozen before the run

**Status: pre-registration.** Written and frozen before the experiment was
executed and before any temperature result existed.

**Claim tags:** `MEASURED`, `DERIVED`, `INFERRED`, `UNKNOWN`.

---

## 1. Question

> Is H2's remaining error materially sensitive to the sharpness of its frozen
> softmax / soft-argmin readout?

Equivalently: does the H2 readout have suboptimal confidence sharpness, such that
changing **only** the inference-time temperature materially improves accuracy
**without destroying stereo functionality**?

`MEASURED` (H2 report §11.6) — H2's standardisation caps peak softmax confidence
at ≈0.77 for a lone winner among 12 candidates, and its observed max weight is
0.43–0.60. `MEASURED` (E1 §3) — `disparity_initial` occupies only 1.34–9.03 of
the available 0–11 candidate range. Whether that costs accuracy was recorded as
`UNKNOWN` in the H2 report and is what this run tests.

## 2. Exact intervention

`MEASURED` — H2's readout is
`phase2/models/scaled_regression.StandardisedDisparityRegression.forward`:
the cost is upsampled to full resolution, standardised per pixel across the 12
candidates, and handed to Phase 1's frozen `soft_argmin`, which computes
`softmax(-C, dim=1)` and takes the expectation over candidate indices.

Temperature is introduced **only** there:

    p(d) = softmax(-C_standardised(d) / T)

implemented as `soft_argmin(scaled / T, dim=1)` — the frozen `soft_argmin` and
the frozen `standardise_across_disparity` are both **imported and called
unchanged**. `T = 1.0` divides by exactly 1.0, which is exact in IEEE-754, so the
control must reproduce H2 **bit-for-bit**; that is a gate, not an expectation.

`phase2/models/scaled_regression.py` is **not modified**. The tempered readout is
a subclass defined in the experiment script, carrying the same `upsample_first`
and `eps` as the model it replaces.

## 3. Frozen controls — what may not change

Weights, checkpoints, cost volume, `cost_volume_shift="left"`, aggregation,
refinement, the standardisation operation and its `eps=1e-5`, the disparity
convention, the evaluation scenes, the scoring code, the probe harnesses. No
training. No other parameter is tuned. Records are never overwritten.

Reused unchanged: `h1_mechanism_probe._forward_parts` / `._finish` (scoring),
`exp_e1_error_partition.Pool` (pooling), `exp_h2_seed_replication.stereo_probe`
(both interventions), `phase2/viz/core` (scene loading, checkpoint hashes).
No second visualizer is created.

## 4. Temperature grid — frozen

    T ∈ {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0}

The prescribed grid, used unchanged. `T = 1.0` is the control.

**All nine temperatures are run on all three seeds, and the stereo interventions
are run at every temperature.** The cost is ~10 minutes, so no cheap-pass /
region-selection step is used — which removes any opportunity to choose a
temperature region after seeing results.

## 5. Checkpoints

`core.MODELS` keys `H2` (seed 0), `SEED1`, `SEED2`, unchanged, hashes recorded
before and after.

## 6. Measurements at every (seed, temperature)

**Accuracy** — pooled EPE and D1 over all 40 `hailo_val` scenes, full 368×1232
frames, `gt > 0`; the pooled protocol used by E1 and E3, whose T=1.0 values are
already recorded (H2 2.400 px / 16.72 %, SEED1 2.319 / 17.25, SEED2 2.556 /
19.76). The H2 training-time 10-scene validation protocol is a **different
population** and is not recomputed or mixed in.

**Readout behaviour** — computed on the exact tensor the softmax consumes,
pooled over the same 40 scenes: mean and median entropy (nats, max ln 12 =
2.4849), mean and maximum peak softmax probability, mean top-2 gap, and the mean
and standard deviation of `disparity_initial` in candidate units.

**Stereo functionality** — `stereo_probe` at every temperature: right-image
corruption (black / noise / right:=left) and matching-map destruction
(own-mean constant / spatial shuffle), reported as the **worst-case** D1 penalty
over four focus scenes, against the frozen constraints **C2** and **C3**:
≥ +20.0 D1 points each. Stereo functionality is never inferred from entropy or
EPE. The matching-gradient criterion is not remeasured: this is inference-only.

**Integrity** — checkpoint SHA-256 before and after; parameter count 423,586;
MACs at 368×1232 measured at a non-default temperature and required to equal the
T=1.0 value; `git diff phase-1-frozen -- src scripts` empty.

## 7. Bands — frozen

Relative to the same seed's own T=1.0 result. The bands are E3's, reused
verbatim so the two experiments speak the same language:

| band | condition |
|---|---|
| **NEGLIGIBLE** | \|ΔD1\| ≤ 1.0 point **and** \|ΔEPE\| ≤ 0.10 px |
| **MATERIAL IMPROVEMENT** | ΔD1 ≤ −1.0 point **and** ΔEPE ≤ −0.10 px |
| **MATERIAL DEGRADATION** | ΔD1 ≥ +1.0 point **or** ΔEPE ≥ +0.10 px |

A temperature is **VIABLE** only if it is a MATERIAL IMPROVEMENT on **all three
seeds** *and* meets C2 and C3 on all three.

## 8. Gates — must pass before any verdict

1. **Control fidelity, two tiers.**
   **(a) In-process** — at T=1.0 the tempered readout reproduces the *untempered*
   H2 readout, run in the same process on the same scenes, to within 1e-9 px EPE
   and 1e-9 D1 points. This is the real test of the intervention: division by
   exactly 1.0 is exact in IEEE-754, so anything else means the subclass changed
   the arithmetic.
   **(b) Against the record** — T=1.0 matches `EXP-E1-ERROR-PARTITION-001`'s
   recorded pooled figures to within 1e-6 px and 1e-6 points. The looser bound is
   deliberate and is fixed here, before the run: this harness's fp behaviour
   across separate processes is **measured** to be non-deterministic
   (`EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md` §3), so a cross-process
   comparison cannot honestly be gated at bit level. (a) is the gate that tests
   this experiment; (b) tests continuity with the record.
2. **Finiteness** — no NaN or Inf at any temperature.
3. **Integrity** — checkpoint hashes unchanged, parameter count unchanged, MACs
   unchanged, Phase 1 diff empty.

Any failure ⇒ verdict `INVALID`.

## 9. Verdict rule — frozen, mapped to the four cases

Exactly one primary verdict:

| verdict | condition |
|---|---|
| **READOUT-SENSITIVE** | Case A — some temperature is VIABLE (material improvement on all three seeds, stereo constraints met). |
| **READOUT-ROBUST** | Case B — no temperature materially improves accuracy on all three seeds. Whether some temperatures *degrade* accuracy is reported separately as one-sided sensitivity; degradation alone does not make the readout an exploitable error source, which is what the question asks. |
| **INCONCLUSIVE** | Case C — a temperature materially improves accuracy but fails C2 or C3 (reported prominently as a critical failure mode, and the improvement rejected); **or** Case D — seeds disagree on which temperatures materially improve. |
| **INVALID** | any gate in §8 fails. |

## 10. No post-hoc tuning

The grid is not expanded, trimmed or re-centred after seeing results. No
per-scene or per-seed temperature selection. No retraining. Any follow-up sweep
is a separately registered experiment.

## 11. Limitations, stated in advance

- `UNKNOWN` — what a model **trained** at a different temperature would do. A
  network adapts to the readout it was trained under, so a negative result here
  is **weak** evidence about temperature as a design variable and must be
  reported as such.
- A better EPE/D1 alone does not establish a better stereo solution; C2/C3 are
  reported for every temperature for exactly that reason.
- Three seeds, 40 scenes, one budget, four focus scenes for the probes. No
  significance test, none claimed.
- Inference-only: nothing here bounds training dynamics, and the matching-path
  gradient is not remeasured.
