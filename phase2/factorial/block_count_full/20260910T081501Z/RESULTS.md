# EXP-BLOCKCOUNT-FULL-001 — BATCH 3 (FINAL) results

Record `phase2/factorial/block_count_full/20260910T081501Z/`.
Pre-registered in `PREREGISTRATION.md`, frozen before preflight and before any
training.

Two runs — 4 blocks / seed 1 and 4 blocks / seed 2, 200 epochs each. Both
completed 200/200, exit 0, no abort, no NaN/Inf. Wall clock **4678.8 s + 4735.2 s
= 9414.0 s (2.61 GPU-hours)**. Epoch-10 viability gate PASS, epoch-100 collapse
check PASS, epoch-200 verdict STEREO-FUNCTIONAL — both runs.

No other run was launched. See *HARD STOP*.

Two defects in this batch's own analysis code were found and are disclosed in
`CORRECTION_screen_prefix.md` and `CORRECTION_stereo_reader.md`. Neither
affected training, thresholds or the decision rule. The superseded comparison is
preserved beside the corrected one.

---

## QUESTION

> Under the deterministic 200-epoch protocol, does reducing the refinement stack
> from 6 blocks to 4 blocks produce a reproducible EPE accuracy penalty?

---

## A. RUN TABLE

40-scene `hailo_val` evaluation of the final epoch-200 checkpoint, pooled over
`gt > 0` — the authoritative metric. Every row but the last two is read from a
frozen record and was not retrained.

| Blocks | Seed | EPE | D1 | RMSE | Stereo | source |
| -----: | ---: | ------: | ------: | -----: | ------ | ------ |
| 6 | 0 | 2.2955182 | 15.6045143 | 6.0437 | yes | Stage B |
| 6 | 1 | 2.3982217 | 18.3357145 | 5.6772 | yes | batch 1 |
| 6 | 2 | 2.3950488 | 17.4699044 | 5.9350 | yes | batch 1 |
| 5 | 0 | 2.6457595 | 21.3711907 | 6.0531 | yes | deterministic series |
| 5 | 1 | 2.4729830 | 18.2061782 | 6.2149 | yes | batch 1 |
| 5 | 2 | 2.6372994 | 20.6593989 | 6.4439 | yes | batch 2 |
| 4 | 0 | 2.3387143 | 16.9797389 | 5.8647 | yes | deterministic series, not retrained |
| 4 | 1 | **2.6553489** | **20.0897392** | **6.3510** | **yes** | **this batch** |
| 4 | 2 | **2.5756633** | **20.0793521** | **6.1759** | **yes** | **this batch** |

All nine evaluations cover the identical 40 scene IDs and identical
**3,802,797** valid pixels. Every pooled figure recomputes from its own
`per_scene` rows to a delta of exactly 0.

Run identity: 4b/seed1 init `2cac2916ed802495` → final `89caa83582ddbd39`;
4b/seed2 init `e622226a5a22fc9b` → final `fc7585ba79a4cf01`. Both 386,594
parameters, dilations 1-2-4-8.

**Prefix reproduction — both runs, exact.** Each run reproduced its 30-epoch
seed-screen prefix at epochs 1, 2, 10 and 20 by exact float equality, plus the
epoch-0 mean loss. Sixth and seventh consecutive runs to reproduce their screen
prefix; the deterministic protocol continues to hold. See
`prefix_verification.json` — and `CORRECTION_screen_prefix.md` for the
limitation that seed 1's check is confirmatory rather than blind.

---

## B. 6-BLOCK vs 4-BLOCK

Individual seeds first, never hidden behind a mean.

**6 blocks** — EPE 2.2955182 / 2.3982217 / 2.3950488 (seeds 0 / 1 / 2)
**4 blocks** — EPE 2.3387143 / 2.6553489 / 2.5756633 (seeds 0 / 1 / 2)

| arm | mean | SD | min | max | **range** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6b EPE | 2.3629296 | 0.0584015 | 2.2955182 | 2.3982217 | **0.1027035** |
| 5b EPE | 2.5853473 | 0.0974023 | 2.4729830 | 2.6457595 | 0.1727766 |
| 4b EPE | 2.5232422 | 0.1646977 | 2.3387143 | 2.6553489 | **0.3166346** |
| 6b D1 | 17.1367110 | 1.3957532 | 15.6045143 | 18.3357145 | 2.7312002 |
| 5b D1 | 20.0789226 | 1.6604338 | 18.2061782 | 21.3711907 | 3.1650125 |
| 4b D1 | 19.0496101 | 1.7925686 | 16.9797389 | 20.0897392 | 3.1100003 |

### Where each 4-block seed falls against the frozen boundary

Frozen 6-block range `[2.2955181809208267, 2.3982217171269924]`, fixed before
launch and asserted against the frozen records at preflight and again in
`compare`.

| 4b seed | EPE | position |
| ---: | ---: | --- |
| 0 | 2.3387143 | **INSIDE** the 6-block range |
| 1 | 2.6553489 | **ABOVE** the 6-block range |
| 2 | 2.5756633 | **ABOVE** the 6-block range |

**The arm straddles the boundary.** Ranges overlap:
`min(4b EPE) − max(6b EPE) = −0.0595074 px`.

### Mean differences

| comparison | ΔEPE | ΔD1 |
| --- | ---: | ---: |
| 4b − 6b | **+0.1603126** | **+1.9128990** |
| 4b − 5b | **−0.0621051** | **−1.0293125** |

### Seed pairings

- **4b vs 6b:** 4b worse on EPE in **7 of 9** pairings (not 9 of 9). The two
  exceptions are 4b/seed0 against 6b/seed1 and 6b/seed2.
- **4b vs 5b:** 4b better on EPE in **5 of 9** pairings — close to a coin flip.

### Per-scene, seed-matched (every model saw the same 40 scenes)

| seed | 4b worse than 6b | 4b worse than 5b |
| ---: | ---: | ---: |
| 0 | 22 / 40 | 3 / 40 |
| 1 | 32 / 40 | 28 / 40 |
| 2 | 34 / 40 | 17 / 40 |

Seed 0 is the outlier of the 4-block arm: near a coin flip against 6 blocks
(22/40) and decisively *better* than 5 blocks (worse in only 3/40). Seeds 1 and
2 behave nothing like it.

---

## C. D1 — SECONDARY, WITH EXPLICIT OVERLAP ARITHMETIC

All three 4-block values: **16.9797389 / 20.0897392 / 20.0793521**.
Mean 19.0496101, range 3.1100003.

```
min(4-block D1) = 16.9797389   <   max(6-block D1) = 18.3357145
margin = −1.3559756 pt
```

**The D1 ranges overlap, by 1.36 points.** They do not separate, and no D1
separation is claimed. 4b is worse on D1 in 7 of 9 pairings — the same 7 as on
EPE, so D1 adds no independent discrimination here.

D1 did not enter the verdict and could not have overridden it.

---

## D. STEREO FUNCTIONALITY

Established probes only, imported unchanged. `right` = minimum D1 penalty over
the three right-image corruptions; `map` = minimum over the two matching-map
corruptions. Epoch 200.

| run | right | map | entropy | disp. SD | r(GT) | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4b seed 0 (frozen) | +79.251 | +67.248 | 1.7812 | 18.305 | 0.9290 | STEREO-FUNCTIONAL |
| 4b seed 1 | +77.031 | +63.333 | 1.8412 | 17.910 | 0.9024 | STEREO-FUNCTIONAL |
| 4b seed 2 | +75.460 | +60.934 | 1.7749 | 17.977 | 0.9163 | STEREO-FUNCTIONAL |

Against the frozen 20-point criterion, entropy floor 0.5 and disparity-SD floor
1.0. Epoch-10 viability gate PASS and epoch-100 collapse check PASS on both new
runs; matching-path gradient present in **100.00 %** of batches on both; no
NaN/Inf; gradient medians 19.0 and 20.3.

Across all nine 200-epoch runs in this campaign the right-image penalty spans
+70.4 … +80.8 and **does not order by block count**: the best 4-block probe
(+79.3) exceeds every 6-block run but one, and the weakest is a 5-block run.

**The 4-block accuracy differences are not stereo collapse.** All three arms
learn to use the right image to the same degree; they differ in how accurately
they regress disparity.

*(Footnote: `probes.json` re-runs the corruption probes from a fresh RNG on the
epoch-200 weight snapshot, so its penalties differ from the training-time
verdict in `metrics.json` by ~0.2–1.9 pt — e.g. seed 1, +77.031 vs +76.838. The
same pattern appears in batches 1 and 2. Both are far above the criterion and
neither is used as a threshold.)*

---

## E. VERDICT

```
4-BLOCK-INCONCLUSIVE-AT-SEEDS-0-1-2
```

Applied mechanically in `comparison.json` from the pre-registered rule:

```
all three 4b EPE > 2.3982217171269924 ?   false   -> not Verdict A
all three 4b EPE inside the 6b range  ?   false   -> not Verdict B
all three 4b runs stereo-functional   ?   true    -> not a stereo failure
                                                   -> Verdict C
```

The three 4-block EPE values **straddle the boundary**: seed 0 inside, seeds 1
and 2 above. This is exactly the case §11 C was written for, and the rule was
applied as written.

**Per the pre-registration: no fourth seed is bought, no threshold is invented,
and the question is not reopened by more of the same design.**

---

## INTERPRETATION

**MEASURED**

- 4b/seed1 at 200 epochs: 40-scene EPE 2.6553489, D1 20.0897392, RMSE 6.3510172.
- 4b/seed2 at 200 epochs: 40-scene EPE 2.5756633, D1 20.0793521, RMSE 6.1758727.
- 4-block EPE over seeds 0/1/2: 2.3387143 / 2.6553489 / 2.5756633; mean
  2.5232422, range **0.3166346**.
- 4-block D1 over seeds 0/1/2: 16.9797389 / 20.0897392 / 20.0793521; mean
  19.0496101, range 3.1100003.
- `min(4b EPE) − max(6b EPE) = −0.0595074 px`; `min(4b D1) − max(6b D1) =
  −1.3559756 pt`. Neither metric separates.
- 4b worse than 6b in 7 of 9 EPE pairings and 7 of 9 D1 pairings.
- 4b better than 5b in 5 of 9 EPE pairings.
- Per-scene vs the same seed at 6 blocks: 22/40, 32/40, 34/40.
- All three 4-block runs STEREO-FUNCTIONAL; all gates passed; matching gradient
  100 %; no NaN/Inf.
- Both new runs reproduced their 30-epoch seed-screen prefixes exactly.
- Wall clock 9414.0 s total.

**DERIVED**

- The 4-block arm has the **widest seed spread of any arm measured**: 0.3166 px,
  **3.08×** the 6-block range and **1.83×** the 5-block range. Its own
  seed-to-seed variation is larger than the 4b-vs-6b mean difference (0.1603 px)
  it would have to establish.
- 4b/seed0 (2.3387) is now clearly the arm's **favourable draw**, not its
  centre — precisely the role 5b/seed1 turned out to play for the 5-block arm.
  The arm's centre is 0.18 px away from it.
- The 4-block mean (2.5232) sits **between** the 5-block mean (2.5853) and the
  6-block mean (2.3629), and much nearer the 5-block one.
- The non-monotonicity that motivated this batch — "5 blocks worse than 4
  blocks" — survives **at the level of arm means** (4b − 5b = −0.0621 px) but
  the two arms' seed ranges overlap heavily (4b 2.3387–2.6553, 5b
  2.4730–2.6458) and 4b beats 5b in only 5 of 9 pairings. It is **not**
  established as an ordering.

**INFERRED**

- The apparent 4-block non-monotonicity observed at seed 0 **does not survive
  additional seeds** as a clean effect. It was substantially a single-draw
  artefact. This is an inference from three runs per arm on one dataset, not a
  test.
- The 4-block arm is plausibly *somewhat* worse than 6 blocks — 7 of 9 pairings,
  a +0.16 px mean gap, and a consistent per-scene direction at seeds 1 and 2 —
  but the evidence cannot be sharpened by adding seeds when the arm's own spread
  is three times the reference arm's. The limitation is the arm's trajectory
  variance, not the sample count.
- The mechanism remains accuracy, not stereo capability: probe values are
  equivalent across all three arms while EPE and D1 are not.

**UNKNOWN**

- Whether 4 blocks carries a genuine penalty. Three seeds straddle the boundary
  and the pre-registered rule returns INCONCLUSIVE. **This is the honest state
  of the question.**
- Why the 4-block arm's seed spread is three times the 6-block arm's. Not
  investigated, and not investigable without a new design.
- Whether the effect is monotone in block count. 6b < 4b ≲ 5b on arm means is
  not an ordering the data support at these overlaps.
- Whether the ordering holds on any other dataset, recipe, schedule,
  training-set size, candidate count, or under pretraining. Nothing here tests
  that.
- Any causal attribution — parameter count, receptive field, or otherwise. Not
  measured, not claimed.

---

## F. CLAIM CEILING

> **The 4-block question is UNRESOLVED at three seeds under this protocol.**
> The measured 4-block EPE values straddle the frozen 6-block reference range,
> and the 4-block arm's own seed-to-seed spread (0.3166 px) exceeds the
> difference it would need to demonstrate (0.1603 px).

What this batch **does** establish:

- Three measured, reproducible, stereo-functional 4-block runs on the
  authoritative protocol, at the same three seeds as the other two arms.
- That the seed-0 4-block result was a favourable draw, so any claim resting on
  it — including the earlier "4 blocks matches 6" reading — is not supported.
- That the deterministic protocol held for two more runs (exact prefix
  reproduction, seventh and eighth consecutive).

What it **does not** establish, and what remains forbidden:

- "4 blocks carries a penalty" or "4 blocks matches 6" — neither. INCONCLUSIVE.
- "parameter count causes the penalty" / "receptive field causes the penalty".
- "6 blocks is optimal"; "more blocks always improve accuracy".
- "D1 proves the capacity penalty" — the D1 ranges overlap by 1.36 pt.
- "the block-count effect is non-monotone" — weakened, not established.
- "O6 proves receptive-field causality"; "E3b proved capacity reduction";
  "3 blocks is the floor".
- Transfer to another dataset, pretrained model, recipe or candidate count.
- Latency conclusions from MAC counts.
- Any ranking based on the 10-scene monitor.

**The 5-block verdict is unchanged by this batch.**
`5-BLOCK-PENALTY-AT-SEEDS-0-1-2` stands exactly as recorded in batch 2,
including its D1 exception. Nothing here touches it.

Three seeds per arm is descriptive evidence about this tested configuration. No
p-value is claimed and none is computable from three points per arm. No
materiality band was reused and none was invented.

---

## HARD STOP

Stopping here, per §17 of the pre-registration. No fourth 4-block seed was
launched, seed 0 was not rerun, no run was extended, no hyperparameter was
tuned, no architecture was altered, no dilation / candidate-count /
receptive-field / Scene Flow / pretrained experiment was started, O6 was not
reopened, no favourable checkpoint was searched for, EPE was not replaced by D1,
and no threshold was reinterpreted. Stage E stays CLOSED.

**A new experimental design — not more seeds of this one — would be required to
reopen the 4-block question.** No such design is proposed here. The research
lead decides what happens next.

---

## PROVENANCE

`HEAD` `58e8a19`; `phase-1-frozen^{commit}` `b4207e5`;
`git diff phase-1-frozen -- src scripts` **empty before and after**;
working tree `M .gitignore`, `?? phase2/` at both ends.
Preflight: **28/28 checks PASS** (`preflight.json`), including `seeds_are_1_and_2`,
`block_count_is_4`, `dilations_are_1_2_4_8`, `initialisation_matches_seed_screen`,
`decision_boundary_matches_frozen_records`, `seed_0_not_scheduled`,
`seed_3_not_scheduled`, `five_block_not_scheduled` and `six_block_not_scheduled`.

The batch-3 guard was tested against five forbidden directions before launch,
not merely documented (`guard_test.log`):

```
STOP: batch 3 authorises only arm C seeds 1 and 2 (4 blocks). Refusing arm C seed 0.
STOP: batch 3 authorises only arm C seeds 1 and 2 (4 blocks). Refusing arm C seed 3.
STOP: batch 3 authorises only arm C seeds 1 and 2 (4 blocks). Refusing arm A seed 1.
STOP: batch 3 authorises only arm C seeds 1 and 2 (4 blocks). Refusing arm B seed 2.
STOP: batch 3 authorises only arm C seeds 1 and 2 (4 blocks). Refusing arm A seed 0.
```

Phase 2 is not committed to git; no external immutability is claimed for any
Phase 2 record, including this one.

Stage A, Stage B, H2, H3, E1, E2, E3, E3b, E3c, O6, the shift factorial, the
deterministic block-count series, the seed screen, batch 1, batch 2 and the
existing 4b/seed0 record were read but never written. All output is confined to
this timestamped directory. The Phase 2 registry was updated **additively only**
(Addendum A and B appended; the pre-existing 848 lines verified byte-identical
after the append).
