# EXP-CORRESPONDENCE-SHIFT-001 — RESULTS

**Status: HALTED AT STAGE A. The diagnostic is invalid as pre-registered.
Positive controls were NOT run and NOT interpreted.**

Record `phase2/diagnostics/correspondence/20260910T142622Z/`.
Pre-registered in `PREREGISTRATION.md`, frozen before any sweep.
Inference only — 36 forward passes, 4.6 s wall clock, zero training.

---

## WHAT HAPPENED

Stage A ran the complete horizontal + vertical sweep on the trained, converged
`shift="none" + standardised` negative control, whose cost volume is provably
degenerate. The pre-registration (§5) and the task specification both predicted:

> all 12 slices identical → standardisation gives zero-valued candidate scores →
> softmax uniform → soft-argmin = (0+1+…+11)/12 = **5.5**, independent of input.

**That prediction is falsified.**

| quantity | predicted | measured |
| --- | ---: | ---: |
| `disparity_initial`, scene 000187_10 | 5.5 | **3.6452** |
| `disparity_initial`, scene 000160_10 | 5.5 | **3.9005** |
| `disparity_initial`, scene 000191_10 | 5.5 | **4.2717** |
| `disparity_initial`, scene 000166_10 | 5.5 | **4.0869** |
| max deviation from 5.5 | 0 | **1.8548 candidates** |
| max response to any translation | 0 | **0.6975 candidates** |

The degenerate model is **not** uniform, and it **is** input-dependent.

---

## WHY — mechanism, MEASURED

Traced and then measured directly (`stage_a_mechanism_diagnosis.log`,
`diagnose_negative_control.py`):

1. **The cost volume is degenerate, exactly as EXP-010 said.**
   `max |slice_k − slice_0|` over all 12 slices = **0.0**, re-confirmed on this
   trained checkpoint.

2. **The aggregated cost is nevertheless strongly non-constant across disparity.**
   Interior spread over d=1…10 = **9.487e12**; mean per-pixel standard deviation
   across d = **5.390e12**.

3. **So the softmax is not uniform.** Mean maximum softmax weight **0.3069**
   against a uniform value of 0.0833. Soft-argmin mean **3.9004**, standard
   deviation across pixels **1.9582**.

**Cause.** `Aggregation` is four `Conv3d(32→32, 3×3×3, padding=1)` + LeakyReLU
followed by `Conv3d(32→1, 3×3×3, padding=1)`. The 3×3×3 kernel spans the
**disparity axis** with zero padding, so identical input slices do **not** give
identical output slices. Five stacked convolutions propagate the D-axis boundary
effect five candidates inward from each end; with D=12 that reaches essentially
the entire volume. The resulting curve is shaped by convolution boundary effects
and by the (identical) slice content — which depends on the right image.

**The pre-registered anchor chain omitted the aggregation stage.** It reasoned
from the cost *volume* to the softmax, skipping the network that sits between
them. The error is in the premise, shared by the specification and this
pre-registration; the implementation tested it faithfully and it failed.

---

## THE CONSEQUENCE THAT MATTERS

> **A provably degenerate model — one that performs no disparity search
> whatsoever — still produces a structured, input-dependent candidate curve and
> responds to right-image translation by up to 0.70 candidates.**

Therefore **"the readout responds to right-image translation" is not evidence of
correspondence.** Any criterion of the form *response ≠ 0* is confounded by the
aggregation's D-axis boundary behaviour. This is the same class of error the
diagnostic was built to avoid, one stage further down the network.

---

## STAGE A SWEEP — the shape, for the record

Pooled over the four `FOCUS_SCENES`, mean response in candidates, 355,072
retained pixels per scene (identical count in every condition).

| Δ (px) | horizontal response | vertical response |
| ---: | ---: | ---: |
| −32 | +0.2645 | +0.2951 |
| −16 | +0.1876 | +0.2169 |
| 0 | 0 | 0 |
| +16 | +0.0017 | +0.4774 |
| +32 | +0.2203 | +0.5463 |

- Horizontal OLS slope **−0.001713 cand/px** = **2.74 %** of the −0.0625 anchor.
- Vertical OLS slope **+0.004769 cand/px** = **−7.63 %** of the anchor.
- Horizontal response is **not monotone** — it is V-shaped and roughly symmetric
  about Δ=0, i.e. a perturbation-**magnitude** effect, not a signed geometric
  one. The near-zero OLS slope is an artefact of fitting a line through a V.
- **The vertical response is larger than the horizontal response.** For this
  model the "null" channel carries the bigger effect.

---

## HARD-STOP RULE EVALUATION

| Rule | Triggered | Basis |
| --- | --- | --- |
| §14.1 systematic non-zero horizontal slope in the negative control | **No** | −0.00171 cand/px = 2.7 % of anchor, and non-monotone; not a systematic signed slope |
| §6 Stage A expected "`disparity_initial` approximately constant around 5.5" | **FAILED** | observed 3.645 … 4.272 |
| §6 Stage A expected "no systematic dependence on the translated right image" | **FAILED** | responses up to 0.6975 candidates |

Two of the three Stage A expectations failed. Per §14, the failure is reported;
the diagnostic is **not** patched and the positive controls are **not** run.

**Stage B and Stage C were not executed.** No positive-control number exists in
this record and none may be quoted.

---

## REFERENCE-FRAME CHECK (§9)

Traced, documented, **preserved — not corrected**.

- `build_cost_volume` with `shift="left"` gives slice *k* =
  `left_feat[u+k] − right_feat[u]`, stored at output column *u* — the **right**
  image's column.
- `viz/core.correspondence` states the left-referenced convention
  `x_right = x_left − d`, and GT plus the refinement guidance are left-referenced.

So the stored candidate carries a reference-frame offset equal to the disparity.
Deriving the perturbation response shows the matching slice obeys
`k = d − Δ/16` **regardless of which frame the value is stored in**, so this
offset affects the **intercept, not the slope**; the −1/16 prediction stands.

It cannot have biased the closed block-count campaign: all nine runs share this
construction identically.

---

## WHAT WOULD BE NEEDED — recommendation only, not executed

The diagnostic is repairable but **requires a new pre-registration**, because the
null must change and changing it after seeing Stage A would be post-hoc.

1. **Replace the analytic zero anchor with the measured degenerate-model
   response.** The empirical null is now known: V-shaped, non-monotone,
   horizontal slope ≈ −0.0017, vertical ≥ horizontal.
2. **Make the discriminating criterion the signed, monotone slope**, not response
   magnitude — the degenerate model produces magnitude but not sign-consistency.
3. **Keep the vertical channel**, but treat it as a measured comparator rather
   than an assumed zero.
4. Consider scoring the **aggregated cost curve's argmin** against GT/16 as a
   secondary read, now that the D-axis boundary contamination is quantified.

None of this is performed here.

---

## VERDICT

```
DIAGNOSTIC-INVALID-AS-PREREGISTERED — STOP, NEW PREREGISTRATION REQUIRED
```

No pattern (A/B/C) is assigned: the pattern logic requires valid positive-control
data, which was deliberately not collected.

---

## CLAIM CEILING

**MEASURED.** The trained `shift="none"` checkpoint has a bit-exactly degenerate
cost volume (max inter-slice difference 0.0) and nevertheless a non-uniform,
input-dependent candidate curve (mean max softmax weight 0.307, soft-argmin
3.90 ± 1.96), responding to right-image translation by up to 0.70 candidates.

**DERIVED.** The pre-registered uniform-softmax anchor is wrong because it omits
the aggregation stage; response magnitude alone cannot evidence correspondence.

**INFERRED.** The D-axis convolution boundary effect is the cause — consistent
with the architecture (five stacked padded 3×3×3 convolutions over 12
candidates) and with the measured curve, but not isolated by an ablation here.

**UNKNOWN.** Whether the 6-block models perform genuine disparity search. **This
diagnostic did not answer its question.** Nothing may be claimed about the
positive checkpoints from this record.

**Explicitly not claimed:** anything about the final disparity, the refinement
path, sub-candidate correspondence, occluded/textureless/repetitive regions,
global algorithm correctness, any block count, parameter-count or
receptive-field causality, latency, transfer, or optimal architecture. The
closed block-count campaign is untouched.

---

## PROVENANCE

`HEAD` `58e8a19`; `phase-1-frozen^{commit}` `b4207e5`;
`git diff phase-1-frozen -- src scripts` **empty**; working tree
`M .gitignore`, `?? phase2/`. No historical record was modified; all output is
confined to this directory. Phase 2 remains untracked in git — no cryptographic
versioning is claimed.
