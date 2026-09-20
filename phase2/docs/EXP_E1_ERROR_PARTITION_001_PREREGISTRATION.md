# EXP-E1-ERROR-PARTITION-001 — pre-registration, frozen before the run

**Status: pre-registration.** Written and frozen before the experiment was
executed and before any oracle number existed. It exists because this project has
already been burned once by a criterion shaped after the fact
(`EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md`), and because the verdict
here decides whether Stage E opens at all.

**Claim tags:** `MEASURED`, `DERIVED`, `INFERRED`, `UNKNOWN`.

---

## 1. Question

H2's remaining error has never been partitioned. Both readings are currently
live:

- the matching stage is as good as this architecture can make it and the
  residual error is imposed **downstream** (refinement, read-out precision), in
  which case Stage E is real architecture work; or
- the matching stage is simply under-trained and a better one would collapse the
  error, in which case Stage E is premature and the next move is O6 (Scene Flow
  pretraining), not architecture.

`MEASURED` (`phase2/results/architecture_ceilings/gt_range_hailo_val.json`) —
disparity *range* is already excluded as an explanation: 0 % of `hailo_val`
ground truth exceeds the architecture's 176 px ceiling.

## 2. Design

**No training.** Existing checkpoints only, inference and substitution.

The intervention point is the matching → refinement boundary, using
`h1_mechanism_probe._forward_parts` / `._finish` **imported unchanged** — the
same harness that produced `h1_mechanism/pooled.json` and
`h2_mechanism/pooled.json`.

**Models:** all three H2 seeds, `core.MODELS` keys `H2` (seed 0), `SEED1`,
`SEED2`. **Scenes:** all 40 `hailo_val` scenes, pooled over `gt > 0`, full
368×1232 frames — the recorded pooled protocol, not the 10-scene validation one.

### The oracle

`disparity_initial` lives in *candidate* units and the candidate→pixel scale is
learned (folded into refinement), so a hand-assumed ×16 would be an assumption,
not a measurement. Instead the model's own convention is recovered by least
squares:

    fit  initial ≈ a·gt + b   over every gt > 0 pixel of the 40 scenes, per model
    oracle_initial = a·gt + b

`a`, `b`, R² and residual standard deviation are recorded per model. A poor fit
is itself a finding and must be reported, not worked around.

### Variants scored

| # | Variant | Role |
|---|---|---|
| 1 | as trained | reference point; must reproduce the recorded pooled numbers |
| 2 | `initial :=` itself (identity substitution) | **fidelity gate** — proves the substitution path changes nothing on its own |
| 3 | `initial := a·gt + b` where `gt > 0`, model's own value elsewhere | **PRIMARY oracle** |
| 4 | `initial := a·gt + b` densified by nearest-valid fill | secondary oracle — checks the mixture artifact in (3) is not driving the result |
| 5 | variant 3 spatially shuffled | **negative control** — must be worse than as trained |

Variant 3 is primary because it fabricates nothing: every substituted pixel
carries real ground truth, and the evaluation population (`gt > 0`) is exactly
the substituted population. Variant 4 is the more realistic input for refinement
(a real matching stage outputs a dense map) but its fill is interpolation, which
is error the oracle did not have to earn.

## 3. Gates that must pass before any verdict is read

1. **Record fidelity** — variant 1 for `H2` reproduces the recorded
   `h2_mechanism/pooled.json` figures (EPE 2.400, D1 16.72) to within 0.01 px and
   0.05 points.
2. **Substitution fidelity** — variant 2 equals variant 1 to within 1e-9 px EPE
   for every model.
3. **Negative control** — variant 5's EPE is worse than variant 1's for every
   model.
4. **Checkpoint integrity** — every checkpoint's SHA-256 is recorded and
   re-asserted unchanged after the run.

If gate 1 or 2 fails the run is void and reports no verdict. If gate 3 fails, the
harness is not measuring what it claims and the run reports no verdict.

## 4. Verdict rule — frozen

Let `r = EPE(variant 3) / EPE(variant 1)`, per model.

| condition | verdict |
|---|---|
| `r ≤ 0.35` | **MATCHING-LIMITED** — a perfect matching stage removes ≥ 65 % of the error; the architecture is not the binding constraint. Stage E stays closed; next move is O6. |
| `r ≥ 0.65` | **DOWNSTREAM-LIMITED** — a perfect matching stage removes ≤ 35 % of the error; the ceiling is refinement/read-out. Stage E opens on E2/E3. |
| otherwise | **MIXED** — both contribute materially; report the split, open nothing on this evidence alone. |

The overall verdict is the one shared by all three seeds. If the seeds disagree,
the result is **NO CONSISTENT VERDICT** and is reported as such.

EPE is the deciding metric because the question is about how much error a perfect
matching stage removes; D1 is a threshold count and is reported alongside for
every variant but decides nothing.

**Where 0.35 and 0.65 come from, stated so they cannot be read as tuned:** they
are round thresholds either side of an even split, chosen to leave a deliberately
wide undecided band rather than to force a verdict. No oracle number existed when
they were written. For external context only, not as a threshold: Phase 1's
pretrained reference measures 1.313 px EPE, and H2 seed 0's recorded pooled
as-trained EPE is 2.400 px.

## 5. What may not change

No weight, checkpoint, recipe, protocol, scene set, or threshold in this document
may be altered once the run starts. The record is `EXP-E1-ERROR-PARTITION-001`
and is never overwritten; a rerun takes a new ID.

## 6. Known limitations, stated in advance

- `UNKNOWN` — this measures the ceiling of *these trained refinement weights*. A
  refinement stage retrained against a perfect matching input could behave
  differently; nothing here says what.
- The oracle is affine in ground truth, so it is perfect only up to the model's
  own linear convention. Any non-linearity in the true candidate→pixel mapping
  makes the oracle pessimistic, and the fit residual bounds how much.
- One dataset slice (40 `hailo_val` scenes), one budget, three seeds, no
  significance test and none claimed.
- `gt > 0` covers ~20 % of pixels (sparse LiDAR); variant 3 leaves the rest at
  the model's own values, so refinement sees a mixture. Variant 4 exists to bound
  that, and it introduces fill error in exchange.
