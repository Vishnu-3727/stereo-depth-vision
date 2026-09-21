# Stage E — recipe optimization of ARM-P

Design specification. Written 2026-09-21. **No training is authorized by this
document.** Execution begins only after the acceptance mathematics in §5 are
reviewed and the first GPU run is explicitly authorized.

---

## 1 Objective

> **Improve the frozen ARM-P model's mean 3-seed KITTI EPE while preserving the
> approximately 400k-parameter footprint and passing the pre-registered INT8
> numerical-survivability gate.**

Current incumbent:

> **1.1988735 px mean across three seeds; 397,954 parameters.**

`1.1912168 px` (seed 1, best checkpoint) is a **historical single-seed
reference**, not the Stage E acceptance baseline. No Stage E result is ever
compared against it as if it were the expected ARM-P score.

### 1.1 Why the baseline was restated

ARM-P's three Stage-B best-checkpoint scores on the frozen contract:

| seed | environment | best-checkpoint EPE | final-checkpoint EPE |
|---|---|---|---|
| 0 | local | 1.2057590 | 1.2087613 |
| 1 | local | 1.1912168 | 1.2017376 |
| 2 | Kaggle-trained, locally evaluated | 1.1996447 | 1.1963930 |
| **mean** | | **1.1988735** | **1.2022973** |
| spread (max−min) | | **0.0145422** | **0.0123683** |

Judging a future 3-seed mean against 1.1912168 would compare a mean to a
maximum: a candidate identical to ARM-P in every respect would appear to
"regress" by ~0.0077 px on average. The mean is the only honest control.

The measured spread of ~0.0145 px is also the noise floor. It is an order of
magnitude tighter than the ARM V situation (0.3410304 px) that the Phase-1
leaderboard warns about, which is what makes this optimization problem
tractable: a ~0.03 px effect is detectable at 3 seeds; a ~0.005 px effect is
not, and will not be claimed.

## 2 Scope and separation

Stage E is a **new branch** under `stage_e_recipe/`. It is a deliberate,
user-authorized continuation of work on the model after the research project
closed. It does not edit the closed record.

Immutable, never modified by Stage E:

- the frozen ARM-P checkpoint, ONNX and all Stage A / B / C / D records;
- `phase1/harness/frozen_eval.py`, which remains the only scorer;
- every historical verdict: C1 **FAIL** (0.001708984375 px vs 1e-3 px), DR-1 H1
  **FAIL** (1.4816284e-02 px vs 1e-3 px), Stage D D0 **BLOCKED**.

Stage E results never appear retroactively inside those records.

## 3 Frozen variables (all Stage E experiments)

Only the optimization recipe changes, **one lever per experiment**. Everything
below is held fixed and asserted at run time:

| Variable | Value | Enforcement |
|---|---|---|
| Architecture | ARM-P, unmodified (`src/models/stereonet`) | param count asserted `== 397954` before training; abort otherwise |
| Initialization | frozen Stage-1 pretrain checkpoint `armp_stage1_best.pth` | SHA-256 asserted `== 3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7` |
| Fine-tune data | KITTI `hailo_calib`, 160 scenes | unchanged loader |
| Evaluation | `phase1/harness/frozen_eval.py`, 40 scenes, 3,802,797 valid px, `gt_scale` 256.0, `disp_occ_0`, 368x1232 top-left crop | contract match asserted in every eval |
| Precision | fp32 | — |
| Loss | masked smooth L1, beta 1.0, valid = `gt>0 AND gt<184` | — |
| Augmentation | scale-aug crop, logUniform(0.7,1.7), gain jitter sigma 0.1, no flip | — |
| Optimizer family | Adam, betas (0.9, 0.999) | LR/schedule/batch are the levers under test |

Because no architectural parameter moves, the ≤400k footprint constraint holds
**by construction** in every experiment, and is additionally asserted.

## 4 Experiments

| id | Intervention | Architecture | Params | Status |
|---|---|---|---|---|
| E0 | Control (exact ARM-P Stage-2 recipe), environment-matched | frozen ARM-P | 397,954 | first |
| E1 | Weight EMA during fine-tune | unchanged | 397,954 | after E0 |
| E2 | Gradient accumulation to effective batch 8 | unchanged | 397,954 | after E0 |
| E3 | Schedule 200 → 400 epochs | unchanged | 397,954 | after E0 |
| E4 | Larger / broader pretraining | unchanged | 397,954 | **BLOCKED** |

**E1, E2 and E3 are not combined.** If one is accepted it becomes the new
controlled baseline, and any combination is a separate, later experiment with
its own pre-registration.

### 4.1 E0 — environment-matched control

Seeds 0 and 1 were trained locally in Stage B; seed 2 was Kaggle-trained, and
the environment control measured a ~0.022 px shift — larger than the seed
spread itself. Mixing environments inside one control mean would put an
uncontrolled term in the baseline.

E0 therefore:

1. **Reproduces seed 1 locally** from the frozen Stage-1 init under the exact
   Stage-2 recipe. This is a **harness gate**: if the re-run does not reproduce
   the Stage-B seed-1 frozen-contract score, the Stage E harness differs from
   the Stage B harness and **Stage E stops** until the difference is explained.
   The reproduction tolerance is pre-registered in §5.1.
2. **Trains control seed 2 locally** (~1 h), replacing the Kaggle seed in the
   Stage E control mean.
3. Reuses the existing **local seed 0** Stage-B ARM-P run unmodified.

Step 3 is only legitimate because of step 1: reusing a Stage-B run inside a
Stage-E control mean assumes the two environments are equivalent, and the
seed-1 reproduction is exactly the measurement that tests that assumption. If
the harness gate fails, seed 0 may not be reused either, and E0 becomes three
fresh local runs (~3 h).

Output: a local 3-seed control distribution (mean, spread, per-seed best and
final), which instantiates the numbers in the §5 formula.

### 4.2 E1 — weight EMA

Exponential moving average of the weights maintained during fine-tuning; the
EMA weights are what the monitor selects on and what is scored. Decay is
pre-registered in §5.4 before the run. No parameter is added to the deployed
model: EMA is a training-time shadow copy, and the exported model has the same
397,954 parameters.

### 4.3 E2 — effective batch 8

Gradient accumulation of 4 steps at batch 2, giving effective batch 8, with a
pre-registered learning-rate rule (§5.4). The dataloader, crop, augmentation
and epoch definition are unchanged; only the optimizer step frequency changes.

### 4.4 E3 — 400 epochs

Epochs 200 → 400, with `CosineAnnealingLR T_max` following the epoch count as
in the frozen recipe. Roughly doubles run time to ~2 h/seed.

### 4.5 E4 — blocked

Larger or broader pretraining is **BLOCKED** and will not be attempted in this
phase. The extracted FlyingThings3D corpus (59 GB) was deleted from disk; only
`extract_ft3d.sh` and `extract.log` remain, and the archives they read from
(`D:\sceneflow_archives\flyingthings3d`) are on a drive that is not currently
mounted. Driving (24 GB extracted) and Monkaa (empty) do not reconstitute the
Stage-1 corpus. Recovering 59 GB is not justified while three untested recipe
levers are runnable today.

## 5 Pre-registration (safeguard 1)

**The acceptance formula is fixed by this document, before E0 runs.** E0
supplies the numbers; it does not supply the rule. No threshold, selection rule
or tie-break may be chosen, altered or reinterpreted after any candidate result
is visible.

### 5.1 Definitions

Let, computed over the three **local** control seeds from E0:

- `M0_best` = mean of the three control best-checkpoint frozen-contract EPEs
- `M0_final` = mean of the three control final-checkpoint EPEs
- `S0_best` = max − min of the three control best-checkpoint EPEs
- `S0_final` = max − min of the three control final-checkpoint EPEs

and for a candidate X over its three seeds: `Mx_best`, `Mx_final`, `Sx_best`.

Improvement is `Δ_best = M0_best − Mx_best` (positive means the candidate is
better).

**Harness gate (E0 step 1):** the local seed-1 reproduction must match the
Stage-B seed-1 best-checkpoint score of 1.191216765057325 px to within
**1e-6 px**. Outside that, Stage E stops and the divergence is investigated
before any candidate runs. A bit-exact match is expected but not required,
because CUDA kernel selection may differ across driver states.

### 5.2 Primary comparison

- **Primary statistic:** the **best-checkpoint** 3-seed mean, `Mx_best`,
  compared against `M0_best`. Best-checkpoint selection uses the frozen
  recipe's own 10-scene monitor, which is part of the recipe under test.
- **Secondary statistic, always reported:** the **final-checkpoint** 3-seed
  mean, `Mx_final`, against `M0_final`.
- Both are reported in every record. Reporting only the flattering selection is
  prohibited, carrying forward the Stage B rule.
- For E1, the monitor evaluates and selects on **the same weights that will be
  scored** (the EMA weights).

### 5.3 Verdicts

Evaluated in this order; the first matching verdict is the verdict:

| Condition | Verdict |
|---|---|
| Fewer than 3 candidate seeds completed | **VOID** — not a result; no verdict recorded |
| INT8 gate fails (§6) | **REJECT** |
| `Δ_best <= 0` | **REJECT** |
| `0 < Δ_best < S0_best` | **INCONCLUSIVE** — improvement inside the control's own noise; not an improvement |
| `Δ_best >= S0_best` **and** `max(candidate seeds) < min(control seeds)` | **ACCEPT (non-overlapping)** |
| `Δ_best >= S0_best` | **ACCEPT** |

The non-overlap tier mirrors the ARM V precedent, where the decisive evidence
was that the candidate's worst seed beat the incumbent's best seed.

- **All three candidate seeds must complete.** A candidate whose third seed
  fails is VOID and is re-run in full; it is never scored on two seeds.
- **Ties.** `Δ_best` exactly 0 at full float precision is REJECT (no
  improvement). Between two ACCEPTed candidates, the larger `Δ_best` wins; if
  those are equal to 7 decimal places, the earlier experiment id wins
  (E1 < E2 < E3). No subjective tie-break exists.
- **Seeds are 0, 1, 2** for every candidate, matching the control.

### 5.4 Hyperparameters fixed before their runs

Recorded here so they cannot be tuned against results:

- **E1 EMA decay = 0.999**, EMA updated every optimizer step, initialized from
  the Stage-1 weights. One value; no sweep. A sweep would be a separate
  experiment with its own pre-registration.
- **E2 learning rate = 1e-3, unchanged**, with effective batch 8. The frozen
  recipe's LR is kept rather than scaled, so that batch size is the single
  lever. (Linear or sqrt LR scaling would confound two changes; if E2 is
  INCONCLUSIVE, an LR-scaled variant is a separate later experiment.)
- **E3 epochs = 400**, `T_max = 400`. No other change.

### 5.5 What may not happen

- No threshold may be derived from, or adjusted after seeing, any candidate
  result.
- No candidate may be re-run with a different seed set to improve its mean.
- A run that is discarded for a technical failure is recorded as discarded,
  with the reason, before the replacement runs.
- A regression is recorded as a regression and never reinterpreted.

## 6 INT8 gate (safeguard 2)

**Purpose and limits.** This is an **INT8 numerical-survivability gate**, not
evidence of Hailo compatibility. The historical EXP-015 result was ONNX Runtime
**CPU** quantization of the **reference** model (int8 EPE 1.655 vs fp32 1.313);
it is not a Hailo measurement, not an ARM-P measurement, and Stage E does not
upgrade it into one. Nothing in this gate bears on Stage D, which remains
blocked.

**Procedure.** Identical and reproducible for the control and every candidate:

1. Export the scored checkpoint to ONNX by the existing frozen export path.
2. Quantize to INT8 with the same ONNX Runtime procedure, settings and
   calibration data for every model.
3. Score the INT8 model through `frozen_eval.py` on the same 40 scenes.
4. Record the penalty `P = EPE_int8 − EPE_fp32` for that model.

**Criterion.** A candidate passes if `P_candidate <= P_control + 0.05 px`,
where `P_control` is measured in E0 by this same procedure. The margin is fixed
here, before `P_control` is known.

If the INT8 procedure cannot be run reproducibly for any model, the gate is
recorded as **NOT MEASURED** for all of them and no candidate is accepted on
INT8 grounds — the gate is never silently skipped for one model only.

## 7 Records

```
stage_e_recipe/
  README.md                 objective, incumbent, current standing
  PREREGISTRATION.md        §5 and §6 instantiated with E0's numbers
  LEADERBOARD.md            one row per completed experiment
  e0_control/  e1_ema/  e2_batch8/  e3_epochs400/
      record.json           hypothesis, intervention, frozen vars, criteria
      seed{0,1,2}/          per-seed run dir, never overwritten
      results.json          per-seed and 3-seed statistics, verdict
```

Every experiment directory is written **before** the run with its hypothesis,
intervention, frozen variables, success criterion and rejection criterion. No
run directory, checkpoint or record is ever overwritten.

## 8 Cost

| Experiment | Per seed | 3 seeds |
|---|---|---|
| E0 (seed-1 reproduction + seed-2 local) | ~1 h | ~2 h |
| E1 | ~1 h | ~3 h |
| E2 | ~1 h | ~3 h |
| E3 | ~2 h | ~6 h |

Roughly 14 h of GPU time on the RTX 4060 Laptop GPU, all runnable without the
missing FlyingThings3D corpus.

## 9 Honest statement of what Stage E can and cannot deliver

E1, E2 and E3 are **optimization levers, not modelling insight**. The plausible
outcomes are:

- **One or more ACCEPT.** A genuinely better small model at the same footprint,
  and a new controlled baseline.
- **All INCONCLUSIVE or REJECT.** A real and publishable finding — *the ARM-P
  recipe was already near-optimal under the tested optimization levers* — but
  not a better model. The honest next move would then be the architecture (U2
  group-wise correlation) or supervision (U3/U5) branches, each of which
  requires its own design and pre-registration.

Stage E does not promise an improvement, and no result will be presented as one
unless it clears §5.3.

## 10 Execution

Per the repository's manager/worker rule, implementation is delegated to the
`opencode` worker with the manager reviewing artefacts. The user's standing
override — that the manager may execute directly when the worker is
unavailable — applies; the free worker model was rate-limited at the time of
writing.

**No GPU run starts until the §5 acceptance mathematics are reviewed and the
first run is explicitly authorized.**
