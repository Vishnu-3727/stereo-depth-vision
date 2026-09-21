# Stage E — implementation plan

**NO TRAINING IS AUTHORIZED BY THIS PLAN.** No GPU training run may start until
this plan is reviewed and the first run is explicitly authorized in a separate
instruction. Nothing in this document was executed; it is planning only.

Written 2026-09-21. Amended 2026-09-21 by the pre-run correction pass
(A1.1–A1.3) and the execution-environment pass (A2.1–A2.4) recorded in the
spec's §0, both before any E0/E1/E2/E3 run existed.

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
| `phase1/harness/frozen_eval.py` | the only scorer (copied into the Kaggle bundle byte-identical, SHA-256 asserted) |
| `phase1/`, `phase2/`, `stage_a_diagnostics/`, `stage_b_armp/`, `stage_c_deploy/`, `stage_d_hailo8/` | closed records |
| `src/models/**` | architecture is frozen |
| `src/losses/disparity.py` | loss is frozen |
| the ARM-P checkpoint, ONNX, and Stage-1 pretrain checkpoint | frozen artefacts |

Stage E results live exclusively under `stage_e_recipe/`. No Stage A–D record
is edited to reflect a Stage E outcome. C1 stays **FAIL**; DR-1 H1 stays
**FAIL**; Stage D D0 stays **BLOCKED**.

## 4 E0 implementation — three fresh Kaggle runs

**Amended by A2.1/A2.2.** E0 is the same-environment ARM-P recipe control:
**three fresh Kaggle T4 runs, seeds 0, 1 and 2**, Stage-1 pretrained ARM-P
initialization, frozen incumbent recipe, no intervention.

| seed | source | environment |
|---|---|---|
| 0 | fresh run | Kaggle T4 |
| 1 | fresh run | Kaggle T4 |
| 2 | fresh run | Kaggle T4 |

No Stage-B run is reused. No local run enters the Stage-E primary comparison.
Stage B's random-init CONTROL arm is not E0; the initialization question is
closed and is not reopened.

Recipe source: `stage_b_armp/20260919T012646Z_tier2_seed1/scripts/`
(`run_arm.py` → `finetune_pilot.py`), behaviour unchanged, copied into the
Stage-E Kaggle bundle with SHA-256s recorded in `source_integrity.json`. A
behaviour change to a copied script is a Stage-E intervention and must be
declared as one.

Recorded per seed in `stage_e_recipe/e0_control/seed<N>/record.json`: training
configuration · seed · checkpoint SHA-256 (best and final) · best epoch · best
EPE · final EPE · D1 (best and final) · wall-clock runtime · Kaggle environment
(python, torch, CUDA, GPU model) · frozen-contract verification
(`contract_match`, 3,802,797 valid px) · parameter count 397,954 ·
initialization checkpoint SHA-256.

## 5 Reproduction gate — RETIRED

**Retired by A2.2.** The gate required reproducing local Stage-B seed 1 to
1e-6 px, and existed solely to license reusing local Stage-B seed 0 inside the
E0 mean. E0 now consists of three fresh Kaggle runs and reuses nothing, so
there is nothing to license and no Stage-E claim rests on reproducing a local
run.

This also retires the concern that made the gate likely to fail: Stage B never
enabled deterministic algorithms, so a 1e-6 match was never guaranteed. That
question no longer bears on Stage E.

What replaces it as a sanity anchor — **not a gate, and no verdict depends on
it** — is the historical ARM-P local mean of 1.1988735 px. A Kaggle E0 mean in
that neighbourhood is reassuring; a difference of a couple of hundredths of a
pixel is consistent with the ~0.022 px local-versus-Kaggle shift Stage B's own
environment control measured, and is **not** a regression.

## 6 Initialization integrity — EXECUTED, PASS

`stage_e_recipe/tests/init_integrity.py`. CPU only, no optimizer step, no
checkpoint written, no weight persisted. Result:
`stage_e_recipe/e0_control/init_integrity.json`, verdict **PASS** (14/14).

| Check | Result |
|---|---|
| checkpoint exists | PASS |
| SHA-256 == `3ae6fb3b…a29be7` | PASS |
| blob carries no optimizer/scheduler state | PASS — blob keys are exactly `['config', 'model']` |
| tensor/key contract | PASS — 70/70 |
| parameter count | PASS — 397,954 |
| strict load, missing keys | PASS — `[]` |
| strict load, unexpected keys | PASS — `[]` |
| 3 downsample levels · 24 disparities · shift `right` · `regression_normalize` true | PASS |
| optimizer state fresh | PASS — no accumulated moments |
| scheduler starts at epoch 0, LR 1e-3 | PASS |

**Only model weights are inherited from Stage 1.** This is true by construction
as well as by measurement: the checkpoint contains no optimizer or scheduler
state to inherit, and the frozen recipe constructs `Adam` and
`CosineAnnealingLR` *after* `apply_init()` runs
(`finetune_pilot.py`, init → optimizer → scheduler order).

This gate is re-run inside the Kaggle kernel before E0 training, so the
assertion holds in the environment that actually trains.

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
3. **Calibration set — fixed by amendment A1.3: the first `N = 32` scenes of
   `hailo_calib`** (training split). `N = 32` equals the historical EXP-015
   default (`--calib`, default 32, same split), so the procedure is inherited
   rather than invented, and it is fixed here before any candidate result
   exists. The identical set is used for the E0 control and for E1, E2 and E3.
   The 40 `hailo_val` evaluation scenes are **never** used for calibration;
   doing so would leak the evaluation set into quantization.
4. Score the INT8 model through `frozen_eval.py` on the same 40 scenes.
5. Record `P = EPE_int8 - EPE_fp32` for that model.

**Gate:** `P_candidate <= P_control + 0.05 px`. The margin is fixed before
`P_control` is known.

**If the procedure cannot be run reproducibly:** mark the gate **NOT MEASURED**
for *all* models, accept no candidate on INT8 grounds, and never skip it for one
model only.

## 9 E2 implementation choice — RESOLVED to native batch 8

**Amendment A1.1 (spec §0, §4.3).** The E2 intervention is `batch 2 -> batch 8`
with everything else frozen. Native batch 8 is preferred over gradient
accumulation because ARM-P has **no BatchNorm**, which makes the two
mathematically identical while native batch 8 avoids introducing accumulation
semantics as an extra implementation variable.

### 9.1 Memory probe — executed LOCALLY, PASS; T4 re-run required (A2.4)

`stage_e_recipe/tests/batch8_memory_probe.py`. **Not a training run:** no
optimizer is constructed, no optimizer step is taken, no scheduler exists, no
checkpoint is written, no weight is modified or persisted, and the dataset is
not read (a synthetic batch at the frozen 256x512 crop is used). One forward
and one backward per batch size, then teardown.

Pre-registered threshold, fixed before measuring: batch-8 peak reserve must
leave **>= 20%** of the card free.

Measured, RTX 4060 Laptop GPU, 8,188 MiB total
(`stage_e_recipe/e2_batch8/batch8_memory_probe.json`):

| batch | status | peak allocated | peak reserved |
|---|---|---|---|
| 2 | FIT | 1,112.1 MiB | 1,254.0 MiB |
| 8 | **FIT** | 4,295.8 MiB | 4,806.0 MiB |

Headroom **41.3%**; allocation scales 3.86x for a 4x batch, consistent with a
BatchNorm-free network.

**Local decision: `NATIVE_BATCH_8`.** Physical batch 8, one optimizer step per
batch, LR exactly 1e-3, no LR scaling, no other recipe change.

**Amended by A2.4:** this measurement is local (RTX 4060, 8,188 MiB) and Stage
E trains on Kaggle T4 (~15 GB). The same probe is re-run unchanged inside the
Kaggle kernel as a pre-E2 smoke test, and **the T4 result is what decides E2's
implementation**. Batch 8 is expected to fit with more headroom there; expected
is not measured. The decision is fixed by that probe *before* E2 trains and is
**not** revisited after seeing E2 results.

### 9.2 Accumulation fallback — not on the critical path

Retained in the spec only for the case the probe had failed. It is **not used**.
Had it been needed it would have required pixel-weighted aggregation
`Σ(nᵢ·Lᵢ)/Σnᵢ` rather than `(1/4)ΣLᵢ`, because `masked_smooth_l1` returns a mean
over each minibatch's valid pixels and those counts vary, plus verification of
optimizer-step count, accumulation count, loss scaling, scheduler-step count and
epoch accounting. Since E2 runs native batch 8, one optimizer step per batch,
none of that machinery exists to be got wrong.

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

E2 = **native physical batch 8** (resolved in §9).

Fixed: physical batch 8 · one optimizer step per batch · LR 1e-3 · no LR
scaling.

Must not change: augmentation, epoch definition (one pass over the 160 scenes),
optimizer family, architecture, initialization, loss, scheduler semantics
(`scheduler.step()` once per epoch, `T_max = 200`). The sole intervention is
the dataloader batch size.

Note on epoch accounting: at batch 8 an epoch contains 20 optimizer steps
instead of 80. That is the intended consequence of the intervention, not a
side change.

## 12 E3 implementation

E3 = 400 epochs.

Fixed: `EPOCHS = 400`, `CosineAnnealingLR T_max = 400`. Everything else is the
frozen recipe. No LR change, no optimizer change, no augmentation change, no
batch change, no EMA, no initialization change. Expected runtime ~2x the
200-epoch run (~2 h/seed, ~6 h for three).

**Selection-count confound — RESOLVED by amendment A1.2.** The frozen monitor
runs at `epoch % 5 == 0 or epoch == EPOCHS-1`, giving **41** selection
opportunities at 200 epochs and **81** at 400. Best-of-81 on a noisy 10-scene
monitor beats best-of-41 even when training quality is identical.

E3 therefore requires **both** statistics to clear the bar:

```
Δ_best >= S0_best   AND   Δ_final >= S0_final   AND   INT8 gate passes
```

`Δ_best >= S0_best` with `Δ_final < S0_final` is recorded as
**BEST PASS / FINAL FAIL** and is **NOT** an ACCEPT. The final checkpoint
carries no selection at all, so it is the honest tie-breaker. **Monitor cadence
is not changed**, and this requirement was fixed before any E3 run existed.

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

**E3 only (A1.2):** acceptance additionally requires `Δ_final >= S0_final`. If
`Δ_best` clears the bar and `Δ_final` does not, the verdict is **BEST PASS /
FINAL FAIL**, which is not an acceptance.

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
| E0 (3 fresh Kaggle seeds) | ~1.5–2.5 h | ~5–8 h |
| E1 | ~1.5–2.5 h | ~5–8 h |
| E2 | ~1.5–2.5 h | ~5–8 h |
| E3 (400 epochs) | ~3–5 h | ~9–15 h |
| INT8 control + 3 candidates | — | ~1–2 h (CPU) |

**Roughly 24–39 h of Kaggle T4 time**, against a weekly GPU quota of about 30 h
and a 12-hour cap per session. Per-seed estimates are scaled from the local
200-epoch run (62 min on an RTX 4060) and are **estimates, not measurements** —
the first E0 seed measures the real rate and the rest of the budget should be
re-derived from it.

Two consequences worth planning for rather than discovering: the campaign will
span **more than one quota week**, and each seed must fit inside a single
12-hour session (E3 at ~3–5 h/seed does, comfortably, but it is the one to
watch). No FlyingThings3D is required; E4 stays **BLOCKED** and no attempt is
made to recover the 59 GB corpus or mount `D:`.

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

**Concern B — RESOLVED by amendment A1.1.** The ambiguity was real: under a
pixel-masked loss whose valid count varies per minibatch, pixel-weighted
accumulation `Σ(nᵢ·Lᵢ)/Σnᵢ` and equal-minibatch `(1/4)ΣLᵢ` are not equivalent,
and only the former equals a batch-8 forward. It is now moot. The memory probe
(§9.1) measured batch 8 at 4,806 MiB reserved on an 8,188 MiB card — 41.3%
headroom against a 20% threshold fixed before the measurement — so E2 runs
**native batch 8**, one optimizer step per batch. ARM-P has no BatchNorm, so
this is mathematically identical to correct accumulation while removing the
accumulation machinery entirely. The ambiguity cannot be resolved in a
candidate's favour after the fact because there is no longer an aggregation
choice to make.

**Concern C — E3's primary statistic is structurally biased in its own
favour.** The frozen monitor fires at `epoch % 5 == 0 or epoch == EPOCHS-1`:
**41** selection opportunities at 200 epochs, **81** at 400. Best-of-81 on a
noisy 10-scene monitor is expected to score better than best-of-41 even if 400
epochs teaches the network nothing extra. Since the primary statistic is the
best checkpoint, part of any E3 improvement will be selection luck rather than
training.

**RESOLVED by amendment A1.2**, adopting mitigation (ii): E3 must clear the bar
on **both** best and final checkpoints, plus the INT8 gate. `Δ_best` passing
while `Δ_final` fails is recorded as **BEST PASS / FINAL FAIL** and is not an
acceptance. No recipe parameter changes and monitor cadence is untouched; the
final checkpoint carries no selection, so it is immune to the 41-vs-81 count.
Fixed before any E3 run exists.

**Concern D — RESOLVED by amendment A1.3.** EXP-015 took its calibration scene
count from the command line, leaving it unpinned. §8 now fixes **`N = 32`
scenes of `hailo_calib`**, which is EXP-015's own default, so the procedure is
inherited rather than invented; the same set is used for the control and all
three candidates, and calibrating on the 40 `hailo_val` evaluation scenes is
forbidden. Without this, INT8 numbers across models would not be comparable and
could leak the evaluation set into quantization.

**Concern A — DISSOLVED by A2.2**, not solved. The reproduction gate it warned
about no longer exists: E0 is three fresh Kaggle runs reusing nothing. The
underlying fact is unchanged and still worth knowing — Stage B never enabled
deterministic algorithms, so its runs are not bit-reproducible — but no Stage-E
decision now depends on reproducing one.

**Concern E (new, from A2.1) — the campaign is quota-bound, not compute-bound.**
At ~24–39 h of estimated T4 time against a ~30 h weekly allowance, Stage E
spans multiple quota weeks and the schedule, not the GPU, is the limiting
factor. The per-seed figures are scaled from local hardware and unverified on
T4; the first E0 seed should be used to re-derive the budget before committing
to E3's 9–15 h.
