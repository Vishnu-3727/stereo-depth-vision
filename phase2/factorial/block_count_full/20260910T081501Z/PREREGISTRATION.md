# EXP-BLOCKCOUNT-FULL-001 — BATCH 3 (FINAL) PRE-REGISTRATION

**Frozen 2026-09-10, before preflight and before any training.**
Record: `phase2/factorial/block_count_full/20260910T081501Z/`.

Nothing in this file may be changed after the first `train` command is issued.
If a rule here turns out to be inconvenient once results exist, the rule wins.

---

## 1. WHY THIS BATCH EXISTS

Batch 2 (`20260910T052736Z`) closed the 5-block question:
`5-BLOCK-PENALTY-AT-SEEDS-0-1-2`. On the authoritative 40-scene protocol every
5-block seed is worse on EPE than every 6-block seed, margin **+0.0748 px**
against a 6-block seed range of **0.1027 px**, 9 of 9 pairings.

The **4-block arm rests on one seed**: EPE 2.3387143, which lies *inside* the
6-block range. That is structurally the same position the 5-block arm occupied
after batch 1 — and batch 2 then showed that the two-point picture had misplaced
the 5-block arm's centre by 0.22 px, because 5b/seed1 had been the arm's
favourable draw rather than a typical one.

So the 4-block question is genuinely open, and it is the *interesting* one: if 4
blocks does not show the 5-block penalty, the effect is **non-monotone in block
count** and a simple capacity story is ruled out for this stack.

---

## 2. RESEARCH QUESTION

> Under the deterministic 200-epoch StereoNet protocol, does reducing the
> refinement stack from 6 blocks to 4 blocks produce a reproducible EPE accuracy
> penalty?

---

## 3. AUTHORISED RUNS — EXACTLY TWO

```
4 blocks, seed 1, 200 epochs      EXP-BLOCKCOUNT-FULL-001-ARM-C-SEED1
4 blocks, seed 2, 200 epochs      EXP-BLOCKCOUNT-FULL-001-ARM-C-SEED2
```

`BATCH3 = (("C", 1), ("C", 2))` in `exp_blockcount_batch3.py` is the guard.
`main()` refuses any other `--arm/--seed` before reaching the training code, and
`full.cmd_train` independently refuses seed 0 and anything outside `BATCH3`.

**Explicitly forbidden and unreachable through this harness:** a 4b/seed0 rerun,
4b/seed3, any additional 5-block or 6-block seed, 3-block training, dilation
variants, candidate-count variants, receptive-field experiments, Scene Flow,
pretrained initialisation, longer training, altered learning rate, altered
schedule, altered crop, altered augmentation, altered loss, altered inference
temperature, and any architecture change other than the registered block count.

The guard will be **tested against forbidden directions before launch**, not
merely documented.

---

## 4. FROZEN MODEL CONFIGURATION

The validated StereoNet/H2 configuration, verbatim. The **only** architectural
variable is the refinement block count.

| | |
| --- | --- |
| refinement blocks | **4** |
| dilations | **1, 2, 4, 8** |
| parameters | 386,594 (asserted against the registration) |
| cost-volume shift | `left` |
| readout | `StandardisedDisparityRegression` |
| disparity candidates | 12 |
| cost-volume method | `subtract` |
| train split | `hailo_calib` (scenes 0–159) |
| val split | `hailo_val` (scenes 160–199) |
| crop | 256 × 512, random |
| augmentation | random crop, per-image gain jitter σ=0.1, no flip |
| loss | masked smooth L1, β=1.0, valid = `gt > 0` and `gt < max_disparity` |
| optimizer | Adam, lr 1e-3, betas (0.9, 0.999) |
| scheduler | CosineAnnealingLR, T_max = 200 — native, not re-fitted |
| batch size | 2 |
| precision | fp32 |
| epochs | **200** |
| initialisation | fresh; **no checkpoint is loaded** |

Nothing is changed because of the 5-block or 6-block results.

---

## 5. DETERMINISTIC PROTOCOL

Exactly the Stage A / Stage B protocol, unchanged:

```
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark      = False
CUBLAS_WORKSPACE_CONFIG             = :4096:8
DataLoader num_workers              = 0
```

Same Python / PyTorch / CUDA / driver environment as the rest of the block-count
campaign (3.12.9 / 2.7.0+cu128 / 12.8 / 616.64). Seed semantics identical to the
existing 6b and 5b experiments: `e3b.SEED = <seed>`, seeding
`torch.manual_seed` and `np.random.seed` through the frozen recipe.

---

## 6. PRE-RUN GUARDS — BLOCKING

`preflight` writes `preflight.json` and returns non-zero on any failure. It
checks, for **both** runs:

`phase_1_diff_empty`, `deterministic_controls_enabled`, `seeds_are_1_and_2`,
`block_count_is_4`, `dilations_are_1_2_4_8`, `dilations_match_registration`,
`shift_is_left`, `readout_is_standardised`, `candidates_are_12`,
`parameter_count_matches_registration`, `epoch_budget_is_200`,
`schedule_T_max_is_200`, `schedule_not_refitted`,
`initialisation_is_fresh_not_seed0`, `initialisation_matches_seed_screen`,
`no_checkpoint_loaded`, `optimizer_state_fresh`,
`surviving_weights_bit_identical_within_seed`, `train_split_is_hailo_calib`,
`val_split_is_hailo_val`, `batch_size_is_2`, `run_count_is_2`,
`batch_is_C1_and_C2_only`, `seed_0_not_scheduled`, `seed_3_not_scheduled`,
`five_block_not_scheduled`, `six_block_not_scheduled`,
`decision_boundary_matches_frozen_records`.

**If any check fails: DO NOT TRAIN.**

---

## 7. PREFIX REPRODUCTION

The 30-epoch seed screen (`block_count_seed_screen/20260909T145659Z`) shortened
only the epoch *budget* while pinning T_max at 200, so its first 30 epochs are a
strict prefix of a 200-epoch run at the same seed. Its 4-block records are
frozen in `SCREEN_PREFIX` **in this batch's script before training**:

| seed | init weight SHA | epoch-0 mean loss | epoch-10 val EPE / D1 | epoch-20 val EPE / D1 |
| ---: | --- | ---: | --- | --- |
| 1 | `2cac2916ed802495` | 11.187826424837112 | 18.365300866 / 88.283045568 | 12.078822974 / 78.735587105 |
| 2 | `e622226a5a22fc9b` | 10.646129465103149 | 16.342920882 / 87.355548703 | 15.506311573 / 85.641635068 |

**Before training** — the initialisation hash is checked against the screen in
preflight (`initialisation_matches_seed_screen`). A mismatch is a blocking
preflight failure and **no run starts**.

**After training** — `verify_prefix` compares epoch-0 mean loss and the epoch
1 / 2 / 10 / 20 validation points by **exact float equality**, not tolerance.
The frozen recipe cannot be interrupted mid-run without modifying historical
code, which is forbidden; this is the same post-hoc verification batch 2 used,
and it is stated here as such rather than implied to be a live abort.

**If any prefix does not reproduce exactly: STOP.** Preserve both runs, do not
continue the analysis, and report the discrepancy as the finding.

---

## 8. VIABILITY GATES — EXISTING, UNCHANGED

- epoch-10 viability gate **PASS**
- epoch-100 collapse check **PASS**
- epoch-200 verdict **STEREO-FUNCTIONAL** (frozen 20-D1-point criterion)
- no NaN, no Inf
- matching-path gradient coverage remains valid
- deterministic execution intact

Gates are imported unchanged from the frozen recipe and are **not** altered.
If a gate fails, stop per the existing protocol.

**A stereo-functionality failure is NOT an accuracy penalty.** It is reported
separately as `4-BLOCK-STEREO-FUNCTIONALITY-FAILURE` and is never pooled into
the capacity comparison.

---

## 9. AUTHORITATIVE EVALUATION

Final **epoch-200** checkpoint (`*_checkpoint.pth`, the last-epoch weights — no
best-val selection), evaluated by the existing evaluator, unchanged:

- all **40** `hailo_val` scenes, full 368×1232 frames
- valid pixels `gt > 0` (KITTI convention)
- pooled over every valid pixel — **3,802,797** pixels, the same count as every
  other run in this campaign
- `src/evaluation/metrics.disparity_metrics`, unchanged

Recorded: pooled EPE, D1, RMSE, bad1, bad2; per-scene EPE, D1, RMSE, bad1, bad2,
valid-pixel count; the existing stereo probes at epochs 10/20/50/100/150/200.

**Primary metric: EPE. Secondary metric: D1.**
The 10-scene training monitor is recorded for continuity and **must not**
determine the verdict — batch 1 measured that it overstates the 5-vs-6 gap by
roughly 5×, and this audit measured that it reorders the arms outright.

**No new metric is introduced.**

---

## 10. DECISION BOUNDARY — FIXED NOW

The frozen 6-block three-seed EPE range, read from the frozen records:

```
SIX_BLOCK_EPE_MIN = 2.2955181809208267    (6b seed 0)
SIX_BLOCK_EPE_MAX = 2.3982217171269924    (6b seed 1)
```

`compare` re-reads the three 6-block `eval40` records and **asserts** these
values before using them. The boundary is not recomputed, widened or softened
after results exist.

The verdict is decided from **three** 4-block seeds — never from one new seed.
**D1 may not override EPE.**

---

## 11. PRE-REGISTERED VERDICTS — EXACTLY ONE

Applied mechanically in `cmd_compare`; no human judgement enters the choice.

### VERDICT A — `4-BLOCK-PENALTY-AT-SEEDS-0-1-2`
All three 4b EPE values **> 2.3982217171269924**, and all three runs
STEREO-FUNCTIONAL.
*Interpretation:* a reproducible empirical performance penalty at both the
5-block and 4-block reductions in this tested configuration. **This still does
not prove parameter count is causal.** STOP.

### VERDICT B — `4-BLOCK-NO-PENALTY-AT-SEEDS-0-1-2`
All three 4b EPE values within **[2.2955181809208267, 2.3982217171269924]**, and
all three runs STEREO-FUNCTIONAL.
*Interpretation:* no empirical EPE penalty at 4 blocks against the observed
6-block seed range. Combined with the established 5-block penalty this
establishes **non-monotonic** behaviour in this tested configuration and rules
out a simple "fewer blocks → worse accuracy" explanation. It does **not** prove
parameter count is irrelevant in general. STOP.

### VERDICT C — `4-BLOCK-INCONCLUSIVE-AT-SEEDS-0-1-2`
The three 4b EPE values straddle the boundary, or the data do not cleanly
satisfy A or B, or the evidence is internally inconsistent.
**Do not buy another seed. Do not run a 4th 4b seed. Do not invent a threshold
after the fact.** Report INCONCLUSIVE; a new experimental design would be
required to reopen the question. STOP.

*Note on the literal reading of B:* a 4-block EPE **below** 2.2955181809208267 —
better than the best 6-block draw — is outside the range and therefore yields C,
not B. That is the rule as written and it will be applied as written.

### Overriding case — `4-BLOCK-STEREO-FUNCTIONALITY-FAILURE`
Any 4-block run not STEREO-FUNCTIONAL at epoch 200. Reported separately, not
pooled into the capacity comparison. Checked first in `cmd_compare`.

---

## 12. D1 — SECONDARY, REPORTED IN FULL

Report all three 4b D1 values, the range, the mean, the nine pairwise
comparisons and the per-scene results, with **explicit overlap arithmetic**.

D1 does not enter the primary verdict. Batch 2 demonstrated that D1 can overlap
while EPE cleanly separates. **Never claim D1 separation unless the observed
ranges actually separate.**

---

## 13. STEREO FUNCTIONALITY

Existing probes only (`exp_h2_seed_replication.stereo_probe`,
`exp_h2_softargmin_scale.validation_stage_stats`), imported unchanged, at epochs
10/20/50/100/150/200: right-image dependence, matching-map perturbation,
matching-gradient coverage, softmax entropy, disparity standard deviation, GT
disparity correlation. No new probe.

They establish binocular **dependence**, not correct disparity search — Phase 1
EXP-007 measured +89.7 D1 on provably degenerate weights.

If 4b is stereo-functional but less accurate → an **accuracy** effect.
If 4b loses stereo functionality → `4-BLOCK-STEREO-FUNCTIONALITY-FAILURE`, stop.

---

## 14. CLAIM CEILING — BINDING

`RESULTS.md` separates **MEASURED / DERIVED / INFERRED / UNKNOWN**.

An empirical block-count effect is **never** restated as a causal claim.
The following remain forbidden unless independently demonstrated:

- "parameter count causes the penalty"
- "receptive field causes the penalty"
- "6 blocks is optimal"
- "more blocks always improve accuracy"
- "D1 proves the capacity penalty"
- "O6 proves receptive-field causality"
- "E3b proved capacity reduction"
- "3 blocks is the floor"
- transfer to another dataset, to pretrained models, to another recipe, or to
  another candidate count
- latency conclusions from MAC counts
- any ranking based on the 10-scene monitor

Three seeds per arm is descriptive evidence about this configuration. No
p-value is claimed and none is computable from three points per arm.

---

## 15. PROVENANCE

Recorded before and after: `git rev-parse HEAD`,
`git rev-parse phase-1-frozen^{commit}`, `git status --short`, and
`git diff phase-1-frozen -- src scripts`. **The Phase 1 diff must be empty at
both ends.**

Read, never written: Stage A, Stage B, H2, H3, E1, E2, E3, E3b, E3c, O6, the
shift factorial, the deterministic block-count series, the seed screen, batch 1,
batch 2, and the existing 4b/seed0 record. All output is confined to this
timestamped directory. No existing file is overwritten.

Any correction to the stale Phase 2 registry is **additive only** — an appended
addendum, never an edit to a historical entry.

Phase 2 is not committed to git; no external immutability is claimed for any
Phase 2 record, including this one.

---

## 16. OUTPUT

```
phase2/factorial/block_count_full/20260910T081501Z/
  PREREGISTRATION.md  README.md  RESULTS.md  ENVIRONMENT.txt  CONFIG.json
  preflight.json  guard_test.log  prefix_verification.json
  run_meta_*.json  train_C_seed{1,2}_stdout.log  eval40_*.json
  probes.json  comparison.json  checkpoints/  experiments/  results/
```

---

## 17. HARD STOP

After the two runs, the 40-scene evaluations, the prefix verification, the
probes, the comparison and `RESULTS.md`: **STOP.**

Do not train another seed, rerun seed 0, extend training, tune hyperparameters,
alter the architecture, start dilation or candidate-count experiments, reopen
O6, start Stage E, search for a favourable checkpoint, replace EPE with D1, or
reinterpret a threshold. Recommend nothing that is not already implied by the
verdict. The research lead decides what happens next.

---

## 18. SCIENTIFIC PRINCIPLE

The purpose is not to prove a preferred hypothesis. It is to find out whether
the apparent 4-block behaviour survives the same deterministic, three-seed,
40-scene protocol that resolved the 5-block question.

**A result that contradicts the capacity narrative is scientifically valuable
and will be preserved exactly as measured.** No post-hoc explanation is
permitted without a new pre-registration.
