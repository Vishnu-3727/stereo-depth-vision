# EXP-BLOCKCOUNT-FULL-001 — BATCH 1 results

Record `phase2/factorial/block_count_full/20260910T005550Z/`.
Pre-registered in `PREREGISTRATION.md`, frozen before the first run.

Three 200-epoch runs, all completed 200/200 epochs, exit 0, no aborts, no
NaN/Inf. Batch training wall clock **14 865.5 s (4.13 GPU-hours)**.

Batch 2 (5b/seed2, 4b/seed1, 4b/seed2) was **not run** and is refused by the
harness. See *NEXT EXPERIMENT* and *HARD STOP*.

---

## QUESTION

> Does the 6-block seed range at 200 epochs stay as large as the 30-epoch screen
> implied, and does a 5-block arm separate from it once past the emergence
> region — enough to justify spending the remaining ~4 GPU-hours?

---

## MATRIX

Validation during training — the recorded protocol: first 10 scenes of
`hailo_val`, full 368×1232 frames, pooled over `gt > 0`.

| Blocks | Seed | EPE@10 | EPE@20 | EPE@50 | EPE@100 | EPE@150 | EPE@200 | D1@200 |
| ------ | ---: | -----: | -----: | -----: | ------: | ------: | ------: | -----: |
| 6 | 1 | 16.354 | 11.531 | 6.170 | 4.404 | 4.056 | **3.772** | 26.87 |
| 6 | 2 | 15.343 | 13.265 | 6.622 | 4.617 | 4.225 | **3.989** | 25.80 |
| 5 | 1 | 16.518 | 11.795 | 6.522 | 4.944 | 4.649 | **4.309** | 27.96 |

Late window (epochs 150–199, 10 validation points per run):

| Blocks | Seed | EPE mean ± SD | D1 mean ± SD |
| ------ | ---: | ------------: | -----------: |
| 6 | 1 | 3.855 ± 0.107 | 27.58 ± 1.01 |
| 6 | 2 | 3.965 ± 0.064 | 25.72 ± 0.47 |
| 5 | 1 | 4.367 ± 0.098 | 28.36 ± 0.55 |

### 40-scene evaluation at epoch 200 — the comparable numbers

All 40 `hailo_val` scenes, full frames, pooled over `gt > 0`, the existing
evaluator unchanged. Seed-0 rows are read out of the frozen records and were
**not** retrained.

| Blocks | Seed | EPE | D1 | RMSE | source |
| ------ | ---: | -----: | -----: | -----: | ------ |
| 6 | 1 | 2.3982 | 18.336 | 5.677 | this batch |
| 6 | 2 | 2.3950 | 17.470 | 5.935 | this batch |
| 5 | 1 | 2.4730 | 18.206 | 6.215 | this batch |
| 6 | 0 | 2.2955 | 15.605 | — | frozen (Stage B) |
| 5 | 0 | 2.6458 | 21.371 | — | frozen (deterministic series) |
| 4 | 0 | 2.3387 | 16.980 | — | frozen (deterministic series) |

Per-scene spread within each 40-scene evaluation (same 40 scenes for every
model, so comparisons are paired): EPE SD 1.49 / 1.64 / 1.83, median 1.94 /
1.93 / 1.92, max 9.31 / 10.27 / 11.47.

---

## DETERMINISM — AN UNPLANNED CHECK THAT PASSED

Every run reproduced the seed screen's 30-epoch prefix to the printed precision:

| Blocks | Seed | Epoch | this batch | seed screen |
| ------ | ---: | ----: | ---------: | ----------: |
| 6 | 1 | 10 | 16.354 / 86.713 | 16.3545 / 86.713 |
| 6 | 1 | 20 | 11.531 / 77.132 | 11.5311 / 77.132 |
| 6 | 2 | 10 | 15.343 / 86.421 | 15.3428 / 86.422 |
| 6 | 2 | 20 | 13.265 / 83.311 | 13.2647 / 83.311 |
| 5 | 1 | 10 | 16.518 / 87.150 | 16.5181 / 87.153 |
| 5 | 1 | 20 | 11.795 / 78.860 | 11.7946 / 78.857 |

Initial weight SHAs are identical to the screen's for the same seeds
(`c27a63f9fac30273`, `60551302b5834cfc`, `a469638e5ed79cce`).

Two consequences, both of which constrain the analysis:

1. The deterministic protocol holds across processes, machines-days and epoch
   budgets. The frozen-`T_max` design worked: a 30-epoch run really is a strict
   prefix of the 200-epoch run.
2. **The screen's 6b/seed1, 6b/seed2 and 5b/seed1 runs are not independent
   samples from these runs — they are the same trajectories.** Anything computed
   from both records double-counts. The screen's remaining three runs (5b/seed2,
   4b/seed1, 4b/seed2) are still independent of this batch.

This also means the O6 non-determinism finding does not apply to this harness as
currently configured: run-to-run execution noise here is zero, and the only
noise left is seed noise.

---

## SEED EFFECT AT 6 BLOCKS — the reference scale this batch existed to measure

Absolute seed-1 vs seed-2 difference on the 10-scene protocol:

| Epoch | 10 | 20 | 50 | 100 | 150 | 200 |
| ----- | -: | -: | -: | --: | --: | --: |
| ΔEPE | 1.012 | 1.734 | 0.453 | 0.213 | 0.169 | 0.218 |
| ΔD1  | 0.292 | 6.179 | 8.302 | 2.731 | 1.949 | 1.067 |

Late-window means: ΔEPE **0.110 px**, ΔD1 **1.86 pt**, against within-run late
SDs of 0.107 and 0.064 px.

**On the 40-scene protocol, across three seeds at 6 blocks:** EPE 2.2955 /
2.3982 / 2.3950 — a total spread of **0.103 px**. The two new seeds agree to
0.003 px; seed 0 sits 0.10 px below both. D1 spread 15.605–18.336 = 2.73 pt.

This is the batch's central measurement, and it **reverses the premise the
campaign was scoped against**. At 30 epochs the seed spread at fixed
architecture was 0.31–3.43 px and larger than the architecture spread; at 200
epochs it is 0.103 px. The screen's own caution — that a run can be worst at
epoch 30 and finish within 0.35 px at 200 — is what actually happened.

The seed effect is not zero. It is small, and at 200 epochs it is roughly the
size of the within-run epoch-to-epoch wobble.

---

## THE 5-BLOCK ARM AGAINST THAT RANGE

**On the 10-scene late window:** 5b/seed1 at 4.367 sits 0.402 px above the
nearer 6-block seed — about 3.7× the 6-block seed gap and 4× the within-run SD.
Read alone, this looks like a clear architecture separation.

**On the 40-scene evaluation it mostly disappears:** 5b/seed1 at 2.4730 is
0.075 px above the 6-block range max (2.3982) — a margin *smaller* than the
0.103 px three-seed spread within 6 blocks. On D1, 5b/seed1 (18.206) falls
**inside** the 6-block range [15.605, 18.336].

The 10-scene subset overstated the gap by roughly 5×. Any future comparison on
this project should be made on the 40-scene protocol; the 10-scene protocol is
a training-time monitor, not a measurement instrument.

Checkpoint-by-checkpoint, the 5-block arm's position relative to the two 6-block
seeds was also unstable: outside at epoch 10, inside at 20, **below both** at
40 / 60 / 80, **above both** from 100 onward. A single crossing pattern like
that is not a capacity signature.

What does survive both protocols: **both available 5-block draws sit above every
6-block draw on 40-scene EPE.** 5b = {2.4730 (seed 1), 2.6458 (seed 0)} versus
6b = {2.2955, 2.3950, 2.3982}. Two of two on the same side, with no overlap.
The 5-block pair also spreads wider (0.173 px) than the 6-block triple
(0.103 px), so the 5-block arm may simply be noisier — two draws cannot tell
those two explanations apart.

---

## TRAJECTORY

All three runs share one shape, and it is a recipe-level shape, not a seed or
architecture one:

- Steep descent to roughly epoch 50 (EPE 16 → 6.2–6.6).
- Slower descent to epoch 100.
- **A non-monotone interval at epochs 100 → 120 in all three runs**: EPE −0.06
  (6b/1), +0.32 (6b/2), +0.13 (5b/1) — validation degraded or stalled while
  training loss kept falling in every case. All three then recovered by epoch
  140. Three of three runs turning at the same window points at the schedule,
  not at chance.
- Annealing tail 140 → 200 delivering real improvement (EPE −0.29 to −0.78),
  ending at LR 0.
- Training loss is not monotone in any run (81–95 epoch-to-epoch increases out
  of 199), which is ordinary batch noise at batch size 2; the trend is downward
  in all three.

Gradients: median total 19.3–20.8, max 384.7–606.3, **no NaN or Inf anywhere**,
matching-path gradient present in **100.00 %** of batches in all three runs.
Both frozen in-run gates — epoch-10 viability and epoch-100 collapse — **passed
in all three runs**.

---

## STEREO FUNCTIONALITY

The established probes only, imported unchanged. `right` = minimum D1 penalty
over the three right-image corruptions; `map` = minimum over the two
matching-map corruptions.

| Blocks | Seed | Ep 10 | 20 | 50 | 100 | 150 | 200 |
| ------ | ---: | ----: | -: | -: | --: | --: | --: |
| 6 | 1 | −3.58 | +12.60 | +55.76 | +73.66 | +72.52 | **+76.38** |
| 6 | 2 | −3.82 | −4.03 | +41.16 | +71.57 | +75.85 | **+74.41** |
| 5 | 1 | −2.09 | +13.47 | +53.12 | +67.25 | +70.58 | **+75.18** |

All three are **STEREO-FUNCTIONAL** at epoch 200 by the frozen criterion, with
right-image penalties of +74 to +76 and matching-map penalties of +59 to +61 —
far above the 20-point threshold and tightly clustered across both block counts.

Supporting statistics at epoch 200: softmax entropy 1.762 / 1.819 / 1.812 nats
(floor 0.5), final disparity SD 18.57 / 17.97 / 17.62 px (floor 1.0),
initial-disparity correlation with GT +0.911 / +0.913 / +0.897. No collapse
anywhere.

Binocular dependence emerges between epochs 20 and 50 in all three runs — the
right-image probe is negative at epoch 10 in every run and above +41 by epoch 50
— confirming EXP-H2-EMERGENCE-001 on a third set of runs. By epoch 100 the
probes have essentially saturated and stop discriminating between the arms.

**Caution, restated:** these probes establish binocular *dependence*, not correct
disparity search. Phase 1 EXP-007 measured a +89.7 D1 right-image penalty on
reference weights that had a provably degenerate shift.

---

## VERDICT

```
CONTINUE-BATCH-2
```

Under the pre-registered rule (§16), CONTINUE-BATCH-2 applies when the 6-block
seed range is small enough that a single-seed architecture arm is readable
**and** the 5-block arm sits far enough outside it to be worth resolving.

The first condition is met decisively and was the open question: the 200-epoch
seed range at 6 blocks is **0.103 px across three seeds**, not the multi-pixel
spread the 30-epoch screen implied. Architecture arms at 200 epochs are readable
against a real, measured reference scale for the first time in this project.

The second condition is met **narrowly, and only on the balance of two
protocols**. The honest statement of it:

- 5b/seed1 is 0.075 px outside the 6-block 40-scene range — less than the range
  width itself, and inside the range on D1. On this evidence alone the arm is
  *not* separated.
- But both available 5-block draws (seeds 0 and 1) exceed every 6-block draw on
  40-scene EPE, with no overlap between the two sets.
- One more 5-block seed decides it: a third draw either keeps the two sets
  disjoint — which, with 3 vs 3 and a measured 0.103 px reference scale, would
  be the first architecture separation this project can support — or lands
  inside the 6-block range and settles the matter the other way.

This is a decision about which *single* run to buy next, not authorisation to
launch the remaining four.

What this verdict does **not** mean:

- It does not mean 5 blocks are worse. Two draws on one side is suggestive, not
  a result.
- It does not mean 6 blocks are superior, or that 4 blocks are anything — the
  4-block arm was not in this batch and its only 200-epoch record (seed 0,
  2.3387) sits *inside* the 6-block seed range.
- It does not license the naive 6-run campaign. The screen's warning about
  underpowering was based on a seed scale that has now been measured and found
  small; the campaign shape should be re-derived from 0.103 px, not from the
  30-epoch numbers.

---

## CLAIM CEILING

> **Batch 1 is two seeds at 6 blocks and one seed at 5 blocks. A single 5-block
> seed cannot separate an architecture effect from a seed effect.**

Nothing in this record supports "5 blocks are required", "5 blocks are
sufficient", "6 blocks are superior" or "4 blocks are equivalent". Every
ordering statement above is scoped to named seeds, a named protocol and a named
epoch.

Two further limits on the numbers as reported:

- The 10-scene and 40-scene protocols disagree about the size of the 5-vs-6
  gap by roughly 5×. Only the 40-scene figures are treated as measurements here.
- The seed screen's 6b/seed1, 6b/seed2 and 5b/seed1 runs are bit-identical
  prefixes of this batch's runs and must never be pooled with them as
  independent samples.

No materiality threshold was defined, before or after seeing the results. The
invalidated E3b/E3c band was neither reused nor replaced. The 0.103 px figure
is a measured spread over three seeds, reported as such — it is not a band, and
it is not a significance test.

---

## NEXT EXPERIMENT

Exactly one, and it is **not** executed here.

**Run 5 blocks, seed 2, 200 epochs, deterministic protocol. One run, ~1.35
GPU-hours.**

Why this one, and why only this one:

- It is the smallest run that resolves the only question batch 1 left open. It
  brings the 5-block arm to three seeds {0, 1, 2}, exactly matching the 6-block
  arm's three seeds, on the same protocol.
- It is decisive in both directions. If 5b/seed2 also exceeds 2.3982, the two
  arms are disjoint at 3-vs-3 against a 0.103 px reference scale — the first
  architecture claim this project could support. If it lands inside the 6-block
  range, the 5-block arm is not separated and the 5-vs-6 question closes.
- It costs one run instead of three. The 4-block arms answer a different
  question and should not be bought until the 5-vs-6 comparison is settled —
  particularly since 4 blocks at seed 0 (2.3387) already sits inside the
  6-block seed range, which is not what the E3b/E3c "capacity-reducible"
  narrative predicted.

The harness refuses this run today (`BATCH1` guard). Authorising it means
extending that tuple — a deliberate, reviewable edit, not a flag.

---

## HARD STOP

Stopping here, per §19 of the pre-registration. Batch 2 was not launched, no run
was continued or extended, no seed was added, no architecture was changed, Stage
E stays CLOSED, and H2 / E3b / E3c / O6 / dilation work was not touched. The
research lead decides whether the recommended run happens.

---

## PROVENANCE

`HEAD` `58e8a19`; `phase-1-frozen^{commit}` `b4207e5`;
`git diff phase-1-frozen -- src scripts` **empty before and after**;
working tree `M .gitignore`, `?? phase2/` at both ends.
Preflight: **20/20 checks PASS** (`results/preflight.json`), including
`batch_is_A1_A2_B1` and `batch_2_not_scheduled`.

The batch guard was tested before launch, not merely documented:

```
STOP: arm C seed 1 is NOT in the authorised batch [('A', 1), ('A', 2), ('B', 1)].
STOP: seed 0 is frozen and must not be retrained.
```

Phase 2 is not committed to git; no external immutability is claimed for any
Phase 2 record, including this one.

Historical E3b / E3c / O6 / H2 / Stage A / Stage B records, the deterministic
block-count series and the seed screen were read but never written. All output
is confined to this timestamped directory.
