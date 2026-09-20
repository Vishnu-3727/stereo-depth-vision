# EXP-E3-REFINEMENT-ABLATION-001 — how much of H2's refinement stack is load-bearing?

**Date:** 2026-09-08. **Compute:** 154 s, no training. **Protocol frozen in
advance:** `EXP_E3_REFINEMENT_ABLATION_001_PREREGISTRATION.md`.

**Claim tags:** `MEASURED`, `DERIVED`, `INFERRED`, `UNKNOWN`.

**Records:** `EXP-E3-REFINEMENT-ABLATION-001-RUN2` (authoritative) and
`EXP-E3-REFINEMENT-ABLATION-001` (complete measurements, crashed on a terminal
encoding error before writing its conclusion — see its `NOTE.md`; its numbers
agree with RUN2's). Raw data:
`phase2/results/EXP-E3-REFINEMENT-ABLATION-001/ablation.json`. Script:
`phase2/scripts/exp_e3_refinement_ablation.py`. Logic tests:
`phase2/tests/test_e3_refinement_ablation.py` (5; suite now **163 passing**).

---

## 1. Verdict

    REFINEMENT IS RIGHT-SIZED.

`MEASURED` — **not one of the thirteen ablated variants is viable.** The best
viable MAC saving is **0.0 %**. Every removal, including the cheapest single
block, lands in the COSTLY band frozen before the run.

`DERIVED` — the hypothesis that motivated E3 ("with the matching path working, a
substantial part of the six-block stack can be removed for little accuracy cost")
is **false for these trained weights**. It is not thereby false for a retrained
smaller stack; see §5.

All four pre-registered gates passed: the full-stack variant reproduces
`EXP-E1-ERROR-PARTITION-001`'s as-trained pooled figures, "keep first 5" equals
"drop block 5" to 1e-9 px, every output finite, all three checkpoint SHA-256s
unchanged.

## 2. Results

`MEASURED` — 40 `hailo_val` scenes, full frames, pooled over `gt > 0`; deltas are
the **worst case over the three seeds**; MACs by `thop` on freshly built models.

| variant | blocks | MACs @368×1232 (G) | MAC saving | worst ΔEPE | worst ΔD1 | band |
|---|---:|---:|---:|---:|---:|---|
| full | 6 | 56.034 | 0.0 % | +0.000 | +0.00 | NEGLIGIBLE |
| drop block 0 (dilation 1) | 5 | 47.677 | 14.9 % | +2.782 | +54.05 | COSTLY |
| drop block 1 (dilation 2) | 5 | 47.677 | 14.9 % | +16.791 | +78.79 | COSTLY |
| drop block 2 (dilation 4) | 5 | 47.677 | 14.9 % | +0.746 | +11.40 | COSTLY |
| drop block 3 (dilation 8) | 5 | 47.677 | 14.9 % | +0.984 | +12.76 | COSTLY |
| drop block 4 (dilation 1) | 5 | 47.677 | 14.9 % | +24.158 | +81.00 | COSTLY |
| drop block 5 (dilation 1) | 5 | 47.677 | 14.9 % | +3.129 | +54.60 | COSTLY |
| keep first 0 | 0 | 5.894 | 89.5 % | +23.815 | +80.51 | COSTLY |
| keep first 1 | 1 | 14.251 | 74.6 % | +24.123 | +81.06 | COSTLY |
| keep first 2 | 2 | 22.607 | 59.7 % | +24.132 | +81.55 | COSTLY |
| keep first 3 | 3 | 30.964 | 44.7 % | +24.150 | +81.49 | COSTLY |
| keep first 4 | 4 | 39.321 | 29.8 % | +24.356 | +81.29 | COSTLY |
| keep first 5 | 5 | 47.677 | 14.9 % | +3.129 | +54.60 | COSTLY |
| drop ladder (blocks 1,2,3) | 3 | 30.964 | 44.7 % | +16.093 | +77.17 | COSTLY |

`MEASURED` — the cheapest removal available is block 2 (dilation 4): +0.746 px
EPE and +11.40 D1 points for 14.9 % of whole-model MACs. `DERIVED` — that is
**11× the +1.0-point NEGLIGIBLE band and ~4× the +3.0-point TOLERABLE band**, so
it is not a close call.

### MAC accounting corroborates Phase 1

`MEASURED` — the full model measures **56.034 GMAC** at 368×1232, against Phase 1's
independently measured **56.04 GMAC** (EXP-001/013), and **16.1995 GMAC** at
256×512, matching the seed-replication record exactly. `MEASURED` — one residual
block is 8.357 GMAC (14.9 % of the whole model); the six blocks together are
50.14 G, 89.5 %.

## 3. Which blocks are load-bearing is a property of the run, not the architecture

`MEASURED` — per-seed ΔEPE for the three cheapest single removals:

| removed block | H2 | SEED1 | SEED2 |
|---|---:|---:|---:|
| block 2 (dilation 4) | +0.668 | +0.737 | +0.746 |
| block 3 (dilation 8) | +0.416 | +0.422 | +0.984 |
| **block 4 (dilation 1)** | **+0.191** | **+24.158** | **+16.371** |

`DERIVED` — block 4 is nearly free in seed 0 (+0.19 px, +2.12 D1 points) and
catastrophic in the other two seeds. **There is no architecturally redundant
block; there is only per-run redundancy**, and it lands in different places in
different runs of the identical recipe.

`INFERRED` — this is what one would expect if the six blocks divide the work
between them differently on every run, which is consistent with the harness's
measured non-reproducibility. `DERIVED` — the practical consequence is direct:
**block-count reduction cannot be decided by pruning a trained model.** A
one-seed pruning study here would have concluded that block 4 is free.

`MEASURED` — the wide-dilation blocks (4 and 8) are consistently the cheapest to
remove individually, yet removing the whole 2/4/8 ladder costs +16.09 px and
+77.17 points. `DERIVED` — blocks 2 and 3 are individually near-redundant and
jointly essential.

## 4. Truncating refinement destroys stereo dependence

`MEASURED` — right-image dependence (worst case over three corruptions, four
focus scenes; constraint **C2** requires ≥ +20 D1 points):

| variant | H2 | SEED1 | SEED2 |
|---|---:|---:|---:|
| full | +83.95 | +76.78 | +74.88 |
| keep first 5 | +24.25 | +75.72 | +70.64 |
| keep first 4 | +18.76 | **−17.01** | **−16.74** |
| keep first 3 | +18.26 | −16.92 | −18.15 |
| keep first 2 | +18.34 | −16.84 | −18.17 |
| keep first 1 | −15.87 | −16.49 | −17.74 |
| keep first 0 | −15.43 | −16.14 | −14.30 |

`DERIVED` — **every truncation below five blocks fails C2 on every seed**, and
from four blocks down the dependence is *negative*: corrupting the right image
makes the truncated model better. That is the H1 monocular signature, produced
here by removing refinement capacity from a model whose matching path is intact.

`INFERRED` — the refinement stack is not a post-hoc polisher sitting downstream of
the answer; it is the path by which matching information reaches the output. Cut
it short and the second camera stops mattering, even though the cost volume, the
aggregation and the read-out are untouched.

`MEASURED` — this also sharpens the E1 result. E1 measured that 58–71 % of the
residual error is imposed downstream of the matching stage; E3 measures that the
same downstream stage is fully used, with no free capacity in it.

## 5. What this does not say

`UNKNOWN` — what a refinement stack **retrained** at four or five blocks would do.
Every number here comes from weights trained as a six-block stack, so this is a
**lower bound** on how small a refinement stage can be, exactly as the
pre-registration stated. The measured per-seed variability in §3 is itself
evidence that the six blocks are dividing work that a smaller stack might divide
differently.

`MEASURED` (Phase 1) — MACs mispredict latency on this stack by 0.67×–142×, and
this run measured **no latency**. The 14.9 %-per-block figure is arithmetic, not
a speedup.

Three seeds, 40 scenes, one budget, four focus scenes for the stereo probe. No
significance test, none claimed.

## 6. Consequences for Stage E

1. `DERIVED` — **E3 is closed as a free efficiency win.** Refinement cannot be
   shrunk by pruning: not one block, not the dilation ladder, not any prefix.
2. `DERIVED` — the efficiency question has moved from "which blocks are
   redundant" to "**how small a refinement stage can be trained**", which is a
   training experiment, not an inference probe.
3. `DERIVED` — combined with E1, the picture of H2 is now: two thirds of the
   residual error is downstream, and the downstream stage is running at full
   utilisation. Neither half of that is fixed by rearranging the cost volume.
4. `MEASURED` — a fourth candidate explanation is closed. Together with the
   disparity-range ceiling (evidence review §2.9) and off-manifold oracle inputs
   (E1 §3), the cheap architectural explanations are exhausted.

## 7. Next experiment

    NEXT: E2 -- inference-time read-out temperature sweep on the trained H2 seeds.

It is the last remaining no-training candidate, it costs minutes, and it targets
the one third of the error E1 attributed upstream: H2's standardisation caps peak
softmax confidence at ≈0.77 and its `disparity_initial` occupies only 1.34–9.03
of the available 0–11 candidate range. Sweep a fixed temperature on the
standardised cost and score the 40 scenes. A positive result promotes E2 to a
retrain; a negative one is weak evidence (a model adapts to the temperature it
was trained under) and should be reported as such.

After E2, the cheap questions are exhausted and the two real experiments left are
both training runs: **E3b** (retrain refinement at four and five blocks under the
H2 recipe, which is what E3 could not answer) and **O6** (Scene Flow
pretraining, which is what the whole 160-scene regime is standing in for).

**Stage E stays closed.** Nothing measured so far implicates the cost volume in
H2's remaining error.
