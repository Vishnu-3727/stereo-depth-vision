# H1 v2 analysis — what is a working cost volume worth, at a budget where the baseline actually learns?

**Supersedes nothing.** `H1_ANALYSIS.md` (v1, 20 epochs, inconclusive) stays as
written; this is the rerun it asked for, and the v1 records
`EXP-H1-BASE` / `EXP-H1-WORKING` are untouched.

**Experiments:** `EXP-H1-BASE-v2` (`cost_volume_shift="none"`) vs
`EXP-H1-WORKING-v2` (`cost_volume_shift="left"`). 200 epochs each — the only
change from v1, with cosine `T_max` tracking it. Everything else identical
between the two arms and to v1: KITTI 2015 `hailo_calib` (160 scenes) train /
`hailo_val` validate, random 256x512 crop, Adam lr 1e-3, batch 2, seed 0,
random init, fp32, RTX 4060. Validation every 5 epochs on 10 scenes.

Raw records: `phase2/experiments/EXP-H1-BASE-v2/metrics.json`,
`phase2/experiments/EXP-H1-WORKING-v2/metrics.json`.

## Precondition: did the baseline converge this time? — MEASURED, yes

v1's null result was uninterpretable because neither arm learned. That is
fixed:

| | v1 BASE (20 ep) | v2 BASE (200 ep) |
|---|---:|---:|
| Train loss, first → last | 11.49 → 8.48 | 11.49 → **4.98** |
| Val EPE, epoch 0 → final | 18.414 → 19.191 (worse) | 17.767 → **14.330** |
| Val D1, epoch 0 → final | 88.7 % → 88.83 % | 88.74 % → **81.19 %** |

Early window (epochs 0–55) min EPE **17.767** vs late window (epochs 150–199)
max EPE **14.880** — non-overlapping, 2.887 px gap, for both arms. The regime
now carries enough signal for an A/B to mean something. The gate that
authorised launching the WORKING arm, including the fact that the literal
±3.5 px threshold was missed by 0.063 px on the final-vs-initial reading and
why the window test replaced it, is recorded in `PHASE_2_REGISTRY.md`.

## Headline comparison — MEASURED

| | BASE-v2 (none) | WORKING-v2 (left) | Delta |
|---|---:|---:|---:|
| Final train loss (epoch 199) | 4.9830 | 4.9021 | −0.081 |
| Min train loss | 4.6238 | 4.5209 | −0.103 |
| Val EPE, final | 14.330 px | 14.340 px | **+0.010** |
| Val EPE, best | 14.122 (ep 190) | 14.139 (ep 170) | +0.017 |
| Val EPE, late-window mean ±sd | 14.423 ± 0.228 | 14.414 ± 0.227 | **−0.008** |
| Val D1, final | 81.19 % | 80.74 % | **−0.45 pt** |
| Val D1, late-window mean ±sd | 81.397 ± 0.559 | 80.756 ± 0.398 | **−0.64 pt** |
| Wall clock | 3635 s | 3627 s | −8 s |
| Parameters | 423 586 | 423 586 | 0 |

### Matched late-window checkpoints (epochs 150–199, 11 pairs)

| epoch | BASE EPE | WORK EPE | ΔEPE | BASE D1 | WORK D1 | ΔD1 |
|---:|---:|---:|---:|---:|---:|---:|
| 150 | 14.604 | 14.844 | +0.240 | 81.28 | 81.08 | −0.20 |
| 155 | 14.880 | 14.369 | −0.511 | 82.47 | 80.44 | −2.02 |
| 160 | 14.398 | 14.797 | +0.399 | 81.22 | 81.59 | +0.37 |
| 165 | 14.385 | 14.506 | +0.121 | 81.02 | 80.60 | −0.42 |
| 170 | 14.159 | 14.139 | −0.020 | 81.18 | 80.39 | −0.79 |
| 175 | 14.332 | 14.267 | −0.065 | 81.21 | 80.33 | −0.88 |
| 180 | 14.720 | 14.440 | −0.280 | 82.50 | 81.25 | −1.25 |
| 185 | 14.403 | 14.350 | −0.052 | 81.27 | 80.72 | −0.55 |
| 190 | 14.122 | 14.175 | +0.052 | 80.75 | 80.44 | −0.31 |
| 195 | 14.316 | 14.330 | +0.014 | 81.27 | 80.73 | −0.54 |
| 199 | 14.330 | 14.340 | +0.010 | 81.19 | 80.74 | −0.45 |

**EPE: 5 of 11 checkpoints favour WORKING** (mean −0.008 px, sd 0.241) — a
coin flip.
**D1: 10 of 11 checkpoints favour WORKING** (mean −0.64 pt, sd 0.616).

## Interpretation

**The two metrics disagree, and that disagreement is the result.**

- **MEASURED:** at this budget, enabling a genuine disparity shift leaves mean
  absolute disparity error unchanged (−0.008 px on a 14.4 px baseline,
  direction inconsistent across checkpoints) while lowering the D1 outlier
  rate by ~0.6 points, in the same direction at 10 of 11 matched checkpoints.
- **INFERRED:** these are consistent with each other. D1 counts pixels
  crossing a fixed 3 px / 5 % threshold; EPE is a mean dominated by the large
  errors that still dominate this under-trained model. A working
  correspondence volume can pull a thin band of near-threshold pixels onto
  the correct side without moving the mean. That is a small, real, and
  unglamorous effect.
- **DERIVED:** it costs nothing — same 423 586 parameters, same 0 MACs for
  volume construction (pad + slice, not a matmul), and wall clock differed by
  8 s in 3630 (0.2 %, below run-to-run noise). Unchanged from v1's finding.

**No p-value is claimed, deliberately.** There is one seed per arm, and the 11
late-window checkpoints are successive epochs of a single training run, so
they are autocorrelated rather than independent samples. A paired t-statistic
computed over them (D1: t = −3.45, df = 10) would look convincing and would
be invalid as an inference about what a rerun with a different seed would do.
The honest statement is the descriptive one: 10 of 11 matched checkpoints in
one seed. **Seed replication is the missing evidence.**

## Observation to follow up — gradient spike in the WORKING arm

`gradient_norms.max` was **3.57e13** for WORKING vs 1.36e4 for BASE — nine
orders of magnitude apart, on a single batch. All 16 000 norms were finite in
both arms, per-epoch *median* grad norms stayed in the 20–60 band for both
(WORKING's worst epoch median was 59.5), and loss did not diverge (last five
epochs 4.99, 4.59, 4.80, 4.79, 4.90), so this did not destabilise training.
It is recorded rather than dismissed: an isolated spike this large in the arm
that has the shift enabled, and not in the arm without it, is a plausible
signature of the volume's boundary handling (candidates that read past the
right image edge are zero-padded) meeting a near-zero-gradient region. It is
**UNKNOWN** whether it is a real numerical hazard or a one-batch curiosity —
`clip_grad_norm_` is currently called with `max_norm=1e9`, i.e. instrumented
but effectively not clipping.

## Verdict

**H1 answered, narrowly: a working cost volume is worth about 0.6 D1 points
and nothing in EPE at this training scale, for zero parameter, MAC, or
latency cost.**

This is a real answer, not v1's non-answer — the baseline demonstrably learns
now, so "no EPE difference" is a measurement rather than an absence of
measurement. But it is a *small* answer from *one seed* at a budget where
both arms sit at ~81 % D1, against the pretrained Hailo reference's 8.15 %.
Whether the effect grows, vanishes, or reverses under real pretraining is
**UNKNOWN** and is exactly the question deferred as O6.

## Next steps, in order

1. **Seed replication before anything is built on this.** Rerun both arms at
   seeds 1 and 2 (`--suffix=-v2s1`, `-v2s2`). Three seeds per arm turns
   "10 of 11 autocorrelated checkpoints" into a claim about runs. This is
   ~4 GPU-hours and is the cheapest evidence available.
2. Decide the gradient-spike question with a cheap instrumented rerun (log
   per-batch norms above a threshold, with the offending batch index), rather
   than reasoning about it further from one number.
3. Only then reconsider Stage E (cost-volume optimization: candidate count,
   representation, hierarchical search). **Stage E remains closed.** A 0.6 pt
   effect from one seed is not a foundation to optimise on top of; optimising
   a mechanism whose measured value is at the edge of this regime's noise is
   how a project ends up tuning noise.
4. The absolute-accuracy question (is ~81 % D1 closable to the reference's
   8.15 %?) is a training-budget question, not an architecture question, and
   belongs to O6 — estimated ~22 h of Scene Flow pretraining at this GPU's
   measured 8.8 samples/s, blocked mainly on dataset size, not compute.
