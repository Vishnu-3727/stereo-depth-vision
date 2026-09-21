# Stage E — implementation plan

**NO TRAINING IS AUTHORIZED BY THIS PLAN.** No GPU training run may start until
this plan is reviewed and the first run is explicitly authorized in a separate
instruction. Nothing in this document was executed; it is planning only.

Written 2026-09-21.

---

## 1 Objective

> Improve the frozen ARM-P model's mean 3-seed KITTI EPE while preserving the
> approximately 400k-parameter footprint and passing the pre-registered INT8
> numerical-survivability gate.

| | |
|---|---|
| Incumbent (Stage E baseline) | **1.1988735 px** mean best-checkpoint, 3 seeds; 397,954 params |
| Historical single-seed reference | 1.1912168 px (seed 1) — **never** the acceptance baseline |
| Question | Can one pre-registered recipe intervention produce a reproducible improvement over an environment-matched 3-seed control, at the same footprint, passing the INT8 gate? |

## 2 Authoritative specification

`docs/superpowers/specs/2026-09-21-stage-e-recipe-design.md`, committed as
`c87da9d`. Its acceptance mathematics are frozen and are **not** altered by this
plan. Where this plan discovers a problem with them, it is raised in §17 and
§21 as a concern for decision — never silently patched.

## 3 Repository boundaries

**Writable by Stage E:** `stage_e_recipe/**` only, plus this plan.

**Read-only, never modified:**

| Path | Why |
|---|---|
| `phase1/harness/frozen_eval.py` | the only scorer |
| `phase1/`, `phase2/`, `stage_a_diagnostics/`, `stage_b_armp/`, `stage_c_deploy/`, `stage_d_hailo8/` | closed records |
| `src/models/**` | architecture is frozen |
| `src/losses/disparity.py` | loss is frozen |
| the ARM-P checkpoint, ONNX, and Stage-1 pretrain checkpoint | frozen artefacts |

Stage E results live exclusively under `stage_e_recipe/`. No Stage A–D record
is edited to reflect a Stage E outcome. C1 stays **FAIL**; DR-1 H1 stays
**FAIL**; Stage D D0 stays **BLOCKED**.

## 4 E0 implementation

E0 produces an environment-matched 3-seed control on **this** machine.

Source of truth for the recipe: `stage_b_armp/20260919T012646Z_tier2_seed1/scripts/`
(`run_arm.py` → `finetune_pilot.py`), invoked **unchanged in behaviour**. Stage E
copies these scripts into `stage_e_recipe/harness/` rather than importing across
a closed record boundary, and asserts the copies are byte-identical to the
Stage-B originals (SHA-256 recorded in `stage_e_recipe/harness/HARNESS_HASHES.json`).
A behaviour change to a copied script is a Stage E intervention and must be
declared as one.

| seed | source | cost |
|---|---|---|
| 0 | reuse Stage-B local ARM-P run — **only if the §5 gate passes** | 0 h |
| 1 | fresh local reproduction (the gate itself) | ~1 h |
| 2 | fresh local training | ~1 h |

## 5 E0 step 1 — reproduction gate

**Purpose.** To test whether the Stage E local harness reproduces the Stage-B
reference run. It does **not** prove that two environments are mathematically
identical, and it is not used for any other claim.

**Procedure.** Train seed 1 from the frozen Stage-1 init under the exact Stage-2
recipe; score the best checkpoint through `frozen_eval.py`.

**Criterion.**

```
| reproduced_best_epe - 1.191216765057325 |  <=  1e-6   -> PASS
                                            >  1e-6   -> FAIL
```

**On PASS:** reuse of Stage-B local seed 0 is licensed; proceed to E0 step 2.

**On FAIL, all of the following hold — no exceptions:**

- **STOP.** Do not train E0 seed 2 under the assumption of equivalence.
- Do not reuse Stage-B seed 0.
- Do not modify the evaluator.
- Do not loosen the tolerance.
- Write `stage_e_recipe/e0_control/REPRODUCTION_DIVERGENCE.md`: the measured
  value, the delta, the environment diff (torch, driver, CUDA, GPU, OS, git
  HEAD), and the per-epoch monitor trace against the Stage-B trace to locate
  where the runs separate.
- The fallback path — already provided by the spec §4.1 — is that E0 becomes
  **three fresh local seeds** (seed 0 also trained locally, ~1 h more), and the
  control mean is built entirely from local runs. Taking that fallback requires
  authorization, because it is a GPU run this plan does not authorize.

**Honest expectation (see §21, concern A).** `finetune_pilot.py` calls
`seed_all()` but **never calls `enable_determinism()`**, so
`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic` and
`CUBLAS_WORKSPACE_CONFIG` are *not* in force during training. Seeds fix data
order, initialization and augmentation draws; they do not suppress
nondeterministic CUDA kernels in the backward pass. A 1e-6 px match after 200
epochs is therefore **not** guaranteed by construction, and a FAIL is a
plausible benign outcome rather than proof of a broken harness. The plan does
not loosen the criterion; it records this expectation in advance so that a FAIL
is interpreted correctly and the cheap fallback is taken without argument.

## 6 E0 step 2 — local seed 2

Train seed 2 locally under the exact frozen Stage-2 recipe, no changes.

Recorded in `stage_e_recipe/e0_control/seed2/record.json`:

training configuration · seed · checkpoint SHA-256 (best and final) · best epoch
· best EPE · final EPE · D1 (best and final) · wall-clock runtime · environment
(OS, GPU, driver, torch, python, git HEAD) · frozen-contract verification
(`contract_match`, valid-pixel count 3,802,797) · parameter count 397,954 ·
initialization checkpoint SHA-256.

## 7 E0 step 3 — control statistics

Over the three **local** control seeds:

```
M0_best  = mean(best EPE  over seeds 0,1,2)
M0_final = mean(final EPE over seeds 0,1,2)
S0_best  = max(best EPE)  - min(best EPE)
S0_final = max(final EPE) - min(final EPE)
```

Written to `stage_e_recipe/PREREGISTRATION.md` **before any candidate runs**.
These values instantiate the already-fixed rule; **the rule is not changed
because of them.**

## 8 INT8 control procedure

**What this is:** an INT8 numerical-survivability gate. **What it is not:**
Hailo validation, Hailo compatibility, HEF validation, or hardware validation.
The historical EXP-015 figure (int8 1.655 vs fp32 1.313) was ONNX Runtime CPU
quantization of the **reference** model, not ARM-P and not Hailo. No Stage E
text may describe this gate otherwise.

**Procedure**, identical for the control and every candidate:

1. Export the scored checkpoint to ONNX through the existing frozen export path
   (`stage_c_deploy/export_armp.py` semantics: static 1x3x368x1232 inputs,
   opset 13).
2. Quantize with the EXP-015 procedure (`scripts/exp_quantization.py`):
   `quantize_static`, `QuantFormat.QDQ`, `activation_type=QInt8`,
   `weight_type=QInt8`, `per_channel=True`.
3. **Calibration set — pre-registered here:** the first *N* scenes of
   `hailo_calib` (training split), with *N* fixed at the first run and reused
   identically for every model. The 40 `hailo_val` evaluation scenes are
   **never** used for calibration; doing so would leak the evaluation set into
   quantization.
4. Score the INT8 model through `frozen_eval.py` on the same 40 scenes.
5. Record `P = EPE_int8 - EPE_fp32` for that model.

**Gate:** `P_candidate <= P_control + 0.05 px`. The margin is fixed before
`P_control` is known.

**If the procedure cannot be run reproducibly:** mark the gate **NOT MEASURED**
for *all* models, accept no candidate on INT8 grounds, and never skip it for one
model only.

## 9 E2 gradient-accumulation correctness gate

E2 training is not authorized until accumulation is demonstrably correct.

**Definition under test:** physical batch 2, accumulation 4, effective batch 8,
one optimizer step per four minibatches, LR exactly 1e-3, no LR scaling, no
other recipe change.

**Test** (`stage_e_recipe/tests/test_accumulation.py`, CPU, tiny synthetic
tensors, no GPU, no training run) must establish:

| Property | Assertion |
|---|---|
| Optimizer-step count | exactly `ceil(batches_per_epoch / 4)` steps per epoch, **not** one per minibatch |
| Accumulation count | gradients accumulate across exactly 4 minibatches before `step()` |
| Gradient equivalence | accumulated gradient equals the gradient of a single true batch-8 forward, to float tolerance |
| Scheduler behaviour | `scheduler.step()` is called once per **epoch**, exactly as in the frozen recipe — not once per optimizer step |
| Epoch accounting | one epoch still traverses the same 160-scene dataset exactly once |
| Zeroing | `zero_grad` occurs once per optimizer step, not once per minibatch |

**Loss scaling — must be settled before the test is written (§21, concern B).**
`masked_smooth_l1` returns `F.smooth_l1_loss(pred[valid], target[valid])`, a
mean over the *valid pixels of that minibatch*, and the valid count varies per
minibatch. Two inequivalent accumulations exist:

- **pixel-weighted:** `L = Σ(nᵢ·Lᵢ) / Σnᵢ` — mathematically equal to a true
  batch-8 forward;
- **equal-minibatch:** `L = (1/4)·Σ Lᵢ` — simpler, but weights a
  100-valid-pixel minibatch like a 100,000-pixel one.

Only the pixel-weighted form actually implements "effective batch 8", so the
plan pre-registers **pixel-weighted**, and the gradient-equivalence assertion
above is what proves it. ARM-P contains **no BatchNorm**, so no batch-statistic
term breaks the equivalence.

**If the implementation is not demonstrably correct: STOP E2.** Do not train.

## 10 E1 implementation

E1 = weight EMA only.

- EMA decay **0.999**, fixed. No sweep, no alternate decay.
- EMA updated **every optimizer step**, initialized from the Stage-1 weights.
- The EMA is a training-time shadow copy; the exported and scored model has
  **397,954** parameters, asserted.
- The 10-scene monitor evaluates and selects on **the EMA weights** — the same
  weights that will be scored. Both `best` and `final` checkpoints store EMA
  weights.
- No other recipe change.

## 11 E2 implementation

E2 = effective batch 8 by accumulation, gated by §9.

Fixed: physical batch 2 · accumulation 4 · effective batch 8 · LR 1e-3.

Must not change: LR scaling, augmentation, epoch definition, optimizer,
architecture, initialization, loss. The sole intervention is optimizer-step
frequency.

## 12 E3 implementation

E3 = 400 epochs.

Fixed: `EPOCHS = 400`, `CosineAnnealingLR T_max = 400`. Everything else is the
frozen recipe. No LR change, no optimizer change, no augmentation change, no
batch change, no EMA, no initialization change. Expected runtime ~2x the
200-epoch run (~2 h/seed, ~6 h for three).

**Selection-count confound — see §21, concern C.** The frozen monitor runs at
`epoch % 5 == 0 or epoch == EPOCHS-1`, giving **41** selection opportunities at
200 epochs and **81** at 400. Best-of-81 on a noisy 10-scene monitor beats
best-of-41 even when training quality is identical, which biases E3's *primary*
statistic upward relative to the control. This plan does not alter the frozen
acceptance rule to compensate; it raises the issue for decision before E3 is
authorized.

## 13 Runtime assertions

Asserted before every training run; any failure **stops before training** and
writes `STOP.json` (the Stage-B `run_arm.py` mechanism, reused):

- parameter count `== 397954`
- architecture source unchanged (SHA-256 of `src/models/**` recorded)
- initialization checkpoint SHA-256 `== 3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7`
- dataset path resolves and scene count `== 160` (`hailo_calib`)
- optimizer family Adam, betas (0.9, 0.999)
- loss = masked smooth L1, beta 1.0, valid `gt>0 AND gt<184`
- augmentation = scale logUniform(0.7,1.7), gain jitter 0.1, **no flip**
- evaluator path = `phase1/harness/frozen_eval.py`, unmodified (SHA-256 recorded)
- evaluation contract: 40 scenes, 3,802,797 valid px, `gt_scale` 256.0,
  `disp_occ_0`, 368x1232 top-left crop
- seed is the intended one
- the experiment-specific intervention is enabled (EMA / accumulation / 400
  epochs) **and no other intervention is enabled**

Inherited live STOP conditions: NaN/Inf loss, zero valid pixels, strict-load
failure, wrong-checkpoint resume, wall-clock over timeout.

## 14 Artifact structure

Created **before** execution:

```
stage_e_recipe/
    README.md  PREREGISTRATION.md  LEADERBOARD.md
    harness/           copied Stage-B scripts + HARNESS_HASHES.json
    tests/             test_accumulation.py and other non-GPU gates
    e0_control/    record.json  seed0/ seed1/ seed2/  results.json
    e1_ema/        record.json  seed0/ seed1/ seed2/  results.json
    e2_batch8/     record.json  seed0/ seed1/ seed2/  results.json
    e3_epochs400/  record.json  seed0/ seed1/ seed2/  results.json
```

Each `record.json`, written before the run: hypothesis · intervention ·
frozen variables · experiment ID · seed set · success criterion · rejection
criterion · INT8 criterion · expected runtime · artifact naming · no-overwrite
rule.

**No run directory, checkpoint, ONNX, INT8 artefact or record is ever
overwritten.** A discarded run is recorded as discarded, with its reason,
before its replacement starts.

## 15 Hashing and provenance

| Artefact | Recorded |
|---|---|
| Checkpoint | SHA-256 · seed · experiment ID · epoch · best/final designation |
| ONNX | SHA-256 · parent checkpoint SHA-256 · export configuration · opset · export timestamp (UTC) |
| INT8 | SHA-256 · parent ONNX SHA-256 · quantization procedure · calibration source |

## 16 Evaluator enforcement

`phase1/harness/frozen_eval.py` is the only scorer. It is not edited, patched,
thresholded, substituted, nor is its dataset selection or masking altered. Its
SHA-256 is asserted before each scoring run. **If the evaluator cannot execute,
STOP and report** — no fallback scorer exists or may be written.

## 17 Verdict implementation

Implemented exactly as approved, evaluated in order, first match wins:

| # | Condition | Verdict |
|---|---|---|
| 1 | fewer than 3 candidate seeds completed | **VOID** |
| 2 | INT8 gate fails | **REJECT** |
| 3 | `Δ_best <= 0` | **REJECT** |
| 4 | `0 < Δ_best < S0_best` | **INCONCLUSIVE** |
| 5 | `Δ_best >= S0_best` **and** `max(candidate seeds) < min(control seeds)` | **ACCEPT (non-overlapping)** |
| 6 | `Δ_best >= S0_best` | **ACCEPT** |

where `Δ_best = M0_best - Mx_best`.

Primary statistic: **best checkpoint**. Secondary, always reported: **final
checkpoint**. Never report only the favourable one. Both checkpoints are
preserved for every run.

**Ties.** `Δ_best` exactly 0 at full precision → REJECT. Between ACCEPTed
candidates, larger `Δ_best` wins; if equal to 7 decimal places, earlier
experiment ID wins (E1 < E2 < E3).

No post-hoc threshold, no alternative interpretation, no subjective judgement,
no seed-set change, no cherry-picking. The verdict is computed by code
(`stage_e_recipe/verdict.py`) from `results.json`, not written by hand.

Acceptance does **not** automatically promote a candidate to the new baseline;
that requires a separate authorized decision, and E1/E2/E3 are never combined
in this campaign.

## 18 Failure handling and stop conditions

**STOP, report, do not improvise:**

- reproduction gate FAIL (§5) — do not proceed on the equivalence assumption
- accumulation correctness gate FAIL (§9) — do not train E2
- any runtime assertion FAIL (§13)
- evaluator cannot execute, or its hash changed (§16)
- frozen artefact hash mismatch (init checkpoint, architecture source)
- NaN/Inf loss, zero valid pixels, strict-load failure
- a run exceeding its wall-clock timeout
- fewer than 3 seeds completing for a candidate → VOID and full re-run

**Never:** loosen a tolerance, patch the evaluator, repair a checkpoint
silently, reinterpret a regression, or select a threshold after seeing results.

## 19 Estimated compute cost

| Item | Per seed | Total |
|---|---|---|
| E0 gate (seed 1) + seed 2 | ~1 h | ~2 h |
| E0 fallback if gate FAILs (seed 0 fresh) | ~1 h | +1 h |
| E1 | ~1 h | ~3 h |
| E2 | ~1 h | ~3 h |
| E3 | ~2 h | ~6 h |
| INT8 control + 3 candidates | — | ~1–2 h (CPU) |

Roughly **15–17 h** on the RTX 4060 Laptop GPU, plus CPU time for INT8. No
FlyingThings3D is required; E4 stays **BLOCKED** and no attempt is made to
recover the 59 GB corpus or mount `D:`.

## 20 Authorization status

**NO TRAINING IS AUTHORIZED BY THIS PLAN.**

Authorized by the instruction that produced this document: implementation
planning, harness construction, validation tests, dry-run checks, configuration
validation, and artifact/provenance preparation.

Not authorized: E0, E1, E2 or E3 training; any GPU training run; FT3D recovery;
architecture or supervision experiments. The first GPU run requires separate
explicit authorization after this plan is reviewed.

## 21 Implementation concerns discovered during planning

**Concern A — training determinism is not enabled, so the 1e-6 reproduction
gate may fail benignly.** `finetune_pilot.py:42` imports only `make_generator`,
`seed_all` and `worker_init_fn` from `phase1.harness.determinism`; it never
calls `enable_determinism()`. The deterministic controls that module provides
(`use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
`CUBLAS_WORKSPACE_CONFIG=":4096:8"`) were therefore **not** in force for the
Stage-B runs and would not be in force for a reproduction. Bit-exact
reproduction after 200 epochs is not guaranteed.

Note the trap: *enabling* determinism for the reproduction would change the
numerics relative to Stage B and could cause the very divergence the gate looks
for. The reproduction must therefore run exactly as Stage B ran — determinism
not enabled — and a FAIL must be read as "this harness does not reproduce that
run to 1e-6", not as "the harness is broken". The spec's fallback (three fresh
local seeds, +1 h) already covers this cleanly and costs little. **No rule
change is requested; this is an expectation-setting note.**

**Concern B — "effective batch 8" is ambiguous under a pixel-masked loss, and
the two readings are not equivalent.** Detailed in §9. The plan pre-registers
the pixel-weighted form because it is the one that actually equals a batch-8
forward. Flagging it because choosing the simpler equal-minibatch form *after*
seeing a disappointing E2 result would be a silent redefinition of the
intervention.

**Also worth a decision:** ARM-P has no BatchNorm, so a *native* batch of 8 is
mathematically identical to correct accumulation, and 8 crops of 256x512 on a
397,954-parameter network will very likely fit in 8 GB. If it fits, native
batch 8 removes the entire accumulation-correctness risk surface and the §9
gate becomes a VRAM check. The spec froze "physical batch 2, accumulation 4",
so this plan implements that as written — but the simpler equivalent is
available on request.

**Concern C — E3's primary statistic is structurally biased in its own
favour.** The frozen monitor fires at `epoch % 5 == 0 or epoch == EPOCHS-1`:
**41** selection opportunities at 200 epochs, **81** at 400. Best-of-81 on a
noisy 10-scene monitor is expected to score better than best-of-41 even if 400
epochs teaches the network nothing extra. Since the primary statistic is the
best checkpoint, part of any E3 improvement will be selection luck rather than
training.

Three mitigations exist, all requiring a decision because each touches frozen
material: (i) run E3's monitor every 10 epochs, holding selection count at 41 —
cleanest, but changes the recipe's monitor cadence; (ii) require E3 to clear the
bar on **both** best and final checkpoints; (iii) accept the bias and state it
explicitly in the E3 record. **Recommendation: (ii)** — it changes no recipe
parameter, costs nothing, and the final-checkpoint statistic is immune to
selection count. This plan implements none of them unilaterally.

**Concern D — the INT8 calibration set was previously unspecified.** EXP-015
calibrated on a scene count passed at the command line. §8 pins calibration to
the first *N* scenes of `hailo_calib`, fixed at the first run and reused
identically, and explicitly forbids calibrating on the 40 `hailo_val`
evaluation scenes. Without that, INT8 numbers across models would not be
comparable and could leak the evaluation set.
