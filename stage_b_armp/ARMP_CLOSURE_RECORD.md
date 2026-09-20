# ARM-P Closure Record — Stage B

Stage B / ARM-P is formally closed. Phase 0, Phase 1 and Phase 2 are CLOSED.
Stage A is COMPLETE. There is no Phase 3; none is created or proposed here.

This is a DOCUMENTATION record. No training was run for it, no evaluation was
run for it, and no new experiment was launched for it. Every number below is
quoted from an existing run record; no historical result is overwritten or
normalized. Seeds are reported separately; nothing is pooled or averaged
across seeds. No claim of statistical significance is made. No mechanism is
claimed from disparity bins.

## 1. Frozen experimental question

The frozen question for Stage B / ARM-P was: does initializing the P2A
architecture from a SceneFlow-pretrained checkpoint of itself, and then
running the exact P2A KITTI fine-tuning recipe, produce a lower frozen-contract
KITTI error than the same recipe from random initialization?

The question was fixed before the Tier-2 pilot ran
(`stage_b_armp/20260918T142128Z_tier2_pilot/README.md`, Design section) and
was not changed afterwards. The independent variable throughout is weight
initialization only. Everything else — architecture, loss, augmentation,
optimizer, scheduler, dataset, seed handling, device protocol, checkpoint
selection, and the frozen evaluation contract — is held identical between the
two arms within each run.

## 2. Exact intervention (ARM-P)

The ARM-P intervention is strict initialization from the Stage-1 pretrained
checkpoint:

- Source file:
  `stage_b_armp/20260918T062146Z_stage1_pretrain/checkpoints/armp_stage1_best.pth`
- sha256:
  `3ae6fb3be6b2bda9287b325ccbe6d1397744c8f20ef5e56c046176d9f1a29be7`
- Loading mode: `model.load_state_dict(sd, strict=True)` loading ONLY the
  `["model"]` weights, applied before optimizer/scheduler construction
  (verified in `stage_b_armp/20260919T012646Z_tier2_seed1/SEED1_REPLICATION_REPORT.md`
  Section 4, citing `scripts/finetune_pilot.py:185-201` and `:204-241`).
- No optimizer state is inherited: checkpoints store only
  `{"model", "config"}` (written at `finetune_pilot.py:316/329`), each arm
  runs in a fresh process with a freshly constructed Adam plus
  CosineAnnealingLR (seed-0 PIDs 26148 vs 22112 per
  `stage_b_armp/20260918T142128Z_tier2_pilot/TIER2_FROZEN_AUDIT.md` Section A7;
  seed-1 PIDs 9008/18280 and 23244/16544 per the seed-1 report Section 4;
  seed-2 PIDs 23/22 and 24/23 per
  `stage_b_armp/20260919T092454Z_tier2_seed2/SEED2_REPLICATION_REPORT.md`
  Section 4).
- No architecture change, no loss change, no augmentation change. The trainer
  is a line-for-line mirror of `phase2/scripts/train_p2a_scale_coverage.py`
  with ONLY `--init` / `--arm` added, proven by `recipe_diff.json`
  (6 non-equal blocks, all classified as the two flags or their direct
  consequences) as documented in the Tier-2 pilot README and re-verified for
  seed 1 (config diff shows ONLY `initialisation` / `init_record` differ) and
  seed 2 (same, plus a Kaggle path-string difference in the source label only;
  sha256 identical).

The Stage-1 checkpoint itself was produced by training the P2A architecture on
the FT3D A+C subset (14,460 usable triplets; see Section 13a), 20-epoch
schedule, P2A mirror with dataset and epoch count as the only recipe changes
(`stage_b_armp/20260918T062146Z_stage1_pretrain/README.md`;
 `stage_b_armp/20260918T062146Z_stage1_pretrain/pretrain_config.json`).

## 3. Exact CONTROL definition

The CONTROL is RANDOM INITIALIZATION under the original P2A recipe. The
control arm constructs the model fresh, runs the integrity guard on the
random-init model, applies a documented no-op for `random`, and only then
creates the optimizer and scheduler. It is never initialized from any
checkpoint, and it must never be described as P2A "checkpoint" initialization.

The distinction matters because a P2A-weights control would have given the
control arm 400 total KITTI epochs (200 from P2A plus 200 in the Tier-2 run)
against ARM-P's 200, so any comparison would have confounded "pretraining
helps" with "more KITTI optimization." Random initialization is P2A's own
starting point, so control-vs-ARM-P isolates exactly one variable: the weight
initialization (Stage-1 pretraining versus nothing). A P2A-initialized control
would have answered a different and uninteresting question — whether 400
epochs beat 200 — and was therefore rejected. This rationale is stated in
`stage_b_armp/20260918T142128Z_tier2_pilot/README.md` (Why the control is
RANDOM init) and restated without alteration in the seed-1 report (scope note),
the seed-2 report (scope paragraph), and
`stage_b_armp/ENV_CONTROL_PREREGISTRATION.md` Section 5 rule 6.

## 4. Seed 0 result (local)

Run directory: `stage_b_armp/20260918T142128Z_tier2_pilot/`. Training device:
local CUDA (NVIDIA GeForce RTX 4060 Laptop GPU, torch 2.7.0+cu128). Seed 0,
200 epochs per arm, checkpoint selection by minimum 10-scene in-training
validation EPE (control best epoch 150, ARM-P best epoch 185). Frozen scores
from `tier2_eval.json`, audited in `TIER2_FROZEN_AUDIT.md` Sections A2–A5.

Best checkpoints (monitor-selected):

- CONTROL best EPE 1.5587716, ARM-P best EPE 1.2057590, delta -0.3530126.
- Full precision: CONTROL 1.5587715928467816, ARM-P 1.2057590024628537,
  delta -0.35301259038392785.

Final checkpoints (epoch 199):

- CONTROL final EPE 1.4809434, ARM-P final EPE 1.2087613, delta -0.2721821.
- Full precision: CONTROL 1.4809434000217176, ARM-P 1.2087612591692831,
  delta -0.27218214085243453.

Delta is ARM-P minus CONTROL; negative means ARM-P lower. Both selections are
reported; only the flattering one is never reported alone. D1 figures
(CONTROL best 9.94207684501697, final 9.236333151624974; ARM-P best
6.337282794742922, final 6.349615822248729) are preserved in the audit table.
No significance is claimed; the 0.0444 px single-seed spread is context only.

## 5. Seed 1 result (local)

Run directory: `stage_b_armp/20260919T012646Z_tier2_seed1/`. Training device:
local CUDA (same stack as seed 0: RTX 4060 Laptop GPU, torch 2.7.0+cu128).
Seed 1, 200 epochs per arm, same argmin selection rule (both arms best epoch
185 on the 10-scene monitor). Frozen scores from that run's `tier2_eval.json`,
verified in `SEED1_REPLICATION_REPORT.md` Sections 8–10.

Best checkpoints:

- CONTROL best EPE 1.4440242, ARM-P best EPE 1.1912168, delta -0.2528074.
- Full precision: CONTROL 1.4440242127250567, ARM-P 1.191216765057325,
  delta -0.25280744766773156.

Final checkpoints:

- CONTROL final EPE 1.4517077, ARM-P final EPE 1.2017376, delta -0.2499701.
- Full precision: CONTROL 1.4517076971230227, ARM-P 1.2017375876950878,
  delta -0.24997010942793496.

This local seed-1 run IS the seed-1 replication. It stands permanently and is
never replaced by the Kaggle seed-1 rerun described in Section 7. The two are
distinct rows in every table, per `stage_b_armp/ENV_CONTROL_PREREGISTRATION.md`
Sections 1–2. Sign only: ARM-P lower on best and on final. No further
interpretation is attached to the seed-1 delta alone.

## 6. Seed 2 result (Kaggle)

Run directory: `stage_b_armp/20260919T092454Z_tier2_seed2/`. Training device:
Kaggle Tesla T4, torch 2.10.0+cu128, Python 3.12.13, Linux (from both arms'
`p2a_record.json > software`). Seed 2, 200 epochs per arm (control best epoch
180, ARM-P best epoch 145 on the 10-scene monitor). The frozen evaluation was
deliberately run LOCALLY with the same evaluator as seeds 0 and 1 (torch
2.7.0+cu128, device cuda, Windows) so the three seeds remain comparable under
one constant measurement layer
(`SEED2_REPLICATION_REPORT.md` Sections 5–7).

Best checkpoints:

- CONTROL best EPE 1.4932978, ARM-P best EPE 1.1996447, delta -0.2936531.
- Full precision: CONTROL 1.4932978077501271, ARM-P 1.1996447345404795,
  delta -0.2936530732096476.

Final checkpoints:

- CONTROL final EPE 1.4850973, ARM-P final EPE 1.1963930, delta -0.2887042.
- Full precision: CONTROL 1.4850972621507696, ARM-P 1.1963930184331233,
  delta -0.28870424371764636.

D1 deltas (same sign convention): best -3.0402096141340182, final
-3.0148335554067183. Sign only: ARM-P lower on best and on final. The
between-seed environment difference (Kaggle vs local) is NOT part of the
intended independent variable; it is bounded, not eliminated, by the Section 7
control.

## 7. Environment-control result (seed 1, Kaggle)

Run directory: `stage_b_armp/20260919T130843Z_seed1_kaggle_envcontrol/`.
Binding document: `stage_b_armp/ENV_CONTROL_PREREGISTRATION.md` (followed
exactly, not modified). This Kaggle seed-1 rerun is the ENVIRONMENT CONTROL.
It is not a new seed, not seed 3 or 4, and never the seed-1 replication. Both
stand side by side: the local seed-1 replication (Section 5) remains the
seed-1 replication, and this rerun stands alongside it purely as the
environment-confound control
(`ENV_CONTROL_REPORT.md` Sections 1–2; preregistration Sections 1–2).

Kaggle training (both arms seed 1, 200 epochs, best epoch 170 each, 397954
params, integrity_guard all_ok; software Python 3.12.13, torch 2.10.0+cu128,
Tesla T4) completed before the evaluation task began; the evaluation task
performed the LOCAL frozen evaluation (Python 3.12.9, torch 2.7.0+cu128,
Windows, device cuda) and wrote the report. No training ran in the evaluation
task.

Frozen local evaluation (full precision, from that run's `tier2_eval.json`):

- CONTROL best 1.4178492233455906, ARM-P best 1.1872901156020892,
  delta -0.2305591077435014.
- CONTROL final 1.4283060535886267, ARM-P final 1.2010106373630214,
  delta -0.2272954162256053.

Within-environment deltas and shifts (per preregistration Section 3, with
`delta_local_best = -0.2528074` and `delta_local_final = -0.2499701` as fixed
references):

- environment_shift_best +0.022248292256498603.
- environment_shift_final +0.02267468377439466.

Per-arm absolute EPE changes between environments (reported separately, not
folded into the delta): CONTROL best -0.026174976654409488, CONTROL final
-0.023401646411373322; ARM-P best -0.0039266843979108845, ARM-P final
-0.0007269626369785787. The shift comes almost entirely from the CONTROL arm
while the ARM-P arm barely moved — consistent with the CONTROL arm's
already-observed noisiness (CONTROL best spans about 0.1147 px across seeds
0/1/2 versus ARM-P best about 0.0145 px), read that way and not as evidence
about ARM-P specifically. The preregistered "small shift relative to the
quoted context spreads" branch therefore applied: the environment-confound
concern on seed 2 is substantially reduced, and the three seeds stand as they
are — which is not proof the environment has no effect, only that this control
did not reveal a material one (`ENV_CONTROL_REPORT.md` Sections 7, 10).

## 8. Frozen evaluation contract

The frozen evaluation contract held on every scored checkpoint across all
four runs (4 checkpoints per run, 16 scores total): 40 scenes, 3,802,797
valid pixels, gt_scale 256.0, gt_source disp_occ_0, contract_match true.
Verified per run in `TIER2_FROZEN_AUDIT.md` Section A4 (seed 0),
`SEED1_REPLICATION_REPORT.md` Section 7 (seed 1),
`SEED2_REPLICATION_REPORT.md` Section 7 (seed 2), and `ENV_CONTROL_REPORT.md`
Section 5 (environment control).

Architecture held constant at scoring: 397,954 params, 70 keys, 24 disparity
candidates, 3 downsample levels, shift right, regression_normalize true, no
BatchNorm. Strict load clean on every scored checkpoint (matched 70, missing
[], unexpected []). The operative eval-time guard
(`scripts/eval_tier2.py:38,130-132`, `EXPECTED_PARAMS = 397954` with abort on
mismatch) passed on all four checkpoints in every run; the `keys_ok /
params_ok false` flags in `tier2_eval.json` are informational comparisons
against stale harness constants (see Section 10) and do not reflect the
checkpoint bytes. The scoring core is a line-for-line mirror of
`phase2/scripts/eval_p2a.py` (same CFG, bins, dataset construction, valid
mask, pooled metrics, contract guard, bin/stratum math) with only the
documented differences: three extra strata, the four-target scoring loop with
its abort guard and strict-load recording, and the dropped per-image names
list (seed-0 audit Section A4). The `>=128` bin holds exactly 1,080 pixels
(0.028%) at every seed and in the environment control; see Sections 13e and
12 for its handling.

## 9. Integrity / provenance status

Byte-identity of the research code across runs: `scripts/finetune_pilot.py`,
`scripts/run_pilot.py` and `scripts/eval_tier2.py` copies are byte-identical
(ratio 1.0) between the seed-0 pilot and seed 1, with the seed-1 report noting
exactly ONE differing line in `eval_tier2.py` (line 134, the checkpoint
provenance label) and `run_arm.py` differing only at line 83 (`P2A_SEED`
0→1), consistent with that run's `harness_diff.json` verdict CLEAN
(`SEED1_REPLICATION_REPORT.md` Section 7). The Kaggle seed-2 bundle's
`source_integrity.json` records byte-identity (`identical: true`) for
`finetune_pilot.py`, `run_pilot.py`, `eval_tier2.py`, every `src/*` file in
the transitive closure, and `phase1/harness/determinism.py` plus
`phase1/harness/frozen_eval.py`, with verdict CLEAN and zero problems; the
environment-control evaluation used the same frozen evaluator with exactly
one changed line (the line-134 provenance label naming the new directory;
`ENV_CONTROL_REPORT.md` Sections 5, 11, 13; `SEED2_REPLICATION_REPORT.md`
Section 7). The `src` and `phase1/harness` files in the bundle are
byte-identical to the repo copies (per-file sha256 pairs in
`source_integrity.json`).

Declared `run_arm.py` edits per run: seed 1 changed ONLY line 83
(`P2A_SEED` 0→1); the stale `seed: 0` literal at line 199 was preserved as it
ran (see Section 10). The Kaggle copy for seed 2 made exactly two declared
edits — line 83 (`P2A_SEED` 1→2) and line 199 (record seed literal 0→2) — per
`source_integrity.json > run_arm_diff` (ratio 0.9908256880733946, two
replace blocks, declared edits listed) and
`stage_b_armp/20260919T035455Z_kaggle_seed2/portability_notes.md` (Seed-2
edits section). No research file was edited for portability; every Kaggle
accommodation lives in `bundle/kaggle_bootstrap.py` (stdlib only).

`stage_a_diagnostics/scripts/d3_matching.py` MD5 `ab29a921c121f5e404014989532172a6`
recomputed before and after: match at seed 0 (audit Section A7), seed 1
(Section 1), seed 2 (Section 1), and the environment control (Section 2).

The five pre-existing tracked working-tree modifications (`.gitignore` plus
`src/models/stereonet/__init__.py`, `cost_volume.py`, `regression.py`,
`stereonet.py`; `git diff --stat` 5 files, +113/-15) predate every Tier-2 run
(file mtimes 2026-09-08 through 2026-09-17, before the seed-1 run start) and
are outside the experiment scope; their provenance is unverified. They are
recorded identically in the seed-0 audit (Section A7), the seed-1 report
(Section 1), the seed-2 report (Section 1), and the environment-control
report (Section 2).

`phase1/`, `phase2/` and `stage_b_armp/` are UNTRACKED in this checkout, so
"eval script unmodified / no artefact overwritten" rests on absence of tracked
diffs (no `M phase2/...` line, empty `git diff -- phase2/scripts/eval_p2a.py`)
plus `git_head` agreement (`58e8a19908ddbd35451652c61aef478f56b51ebd` in
`tier2_eval.json > metadata`, both arms' `p2a_record.json`, and
`git rev-parse HEAD`) — not on a clean tracked tree. This limit is stated
explicitly in the seed-0 audit (Sections A4, A7, open defect 4).

## 10. Known deviations and their scientific relevance

- Seed 2 trained on Kaggle (Tesla T4, torch 2.10.0+cu128, Python 3.12.13,
  Linux) while seeds 0 and 1 trained locally (RTX 4060 Laptop GPU, torch
  2.7.0+cu128) — a between-seed environment difference that is NOT the
  intended independent variable. Its scientific relevance is bounded, not
  eliminated, by the environment control: the measured within-delta shifts
  are about +0.022 px on best and final (Section 7), sitting below the
  0.0444 px contextual spread and driven almost entirely by the CONTROL arm.
  The three seeds stand as they are, with the caveat attached rather than
  silently dropped.
- Evaluation was deliberately held local for all four runs (same frozen
  evaluator, torch 2.7.0+cu128), keeping the measurement layer constant so
  the runs remain comparable. This is a design choice, not a defect.
- `run_arm.py`'s stale `seed: 0` literal at seed 1 (line 199, writing
  `control/record.json:4` and `armp/record.json:4`) is cosmetic, symmetric
  across both arms, and never read by training, init, data-ordering, or
  evaluation paths — proved in `CORRECTIONS.json` (seed consumed exclusively
  via `P2A_SEED` at `finetune_pilot.py:59`; `record.json` written at
  `run_arm.py:210` after training; its only readers consume
  `train_monitor_best_epoch` and `training_duration_s`). The comparison is
  unbiased by it. The same root cause propagated stale seed-0 labels into
  seed-1 `pilot_results.json` (`experiment` / `seed` fields) and the
  `README.md:58` section header while all numeric content is seed-1 values
  (seed-1 report Section 13). Fixed at seed 2 (`record.json` reads 2 on both
  arms). `run_arm.py` was not edited after the fact, per instruction.
- `phase1/harness/frozen_eval.py` stale `EXPECTED_STATE_KEYS = 72` /
  `EXPECTED_PARAMS = 423586` (lines 49–50, for the old default
  architecture) produce informational `keys_ok / params_ok false` on clean
  70-key / 397954-param checkpoints in every run's `tier2_eval.json`; the
  operative guard in `eval_tier2.py:130-132` is unaffected and every
  evaluation completed. Any consumer reading the informational flags at face
  value would misread a clean checkpoint as failing (seed-0 audit Section
  A3, open defect 1).
- The first Kaggle portability gate PASSED without exercising the
  `finetune_pilot.py` entrypoint, so it missed the missing
  `phase1/runs/arm_v/arm_v_best.pth` runtime dependency (needed by
  `integrity_guard()` at `finetune_pilot.py:154`) and the `REPO_ROOT`
  nesting depth (`parents[3]` at `finetune_pilot.py:39`, requiring scripts
  at `<root>/stage_b_armp/seed2/scripts/`). Both were later fixed
  wrapper/bundle-side (dataset B v6: the reference file added at exactly
  `phase1/runs/arm_v/arm_v_best.pth`, sha256-verified; `assemble()` nesting
  fixed with a fail-fast guard) and re-gated with a 2-epoch entrypoint run
  (`kaggle_entrycheck.py`, `P2A_EPOCHS=2` through the actual script, not
  through `run_arm.py`), documented honestly in `portability_notes.md`
  Sections 11–12 including the note on the v1–v4 smoke test. No research
  file was edited for portability.
- CRLF line endings are present on bundled `run_arm.py` copies. This is a
  packaging property only; it affects no training, selection, or evaluation
  path and carries no scientific relevance beyond being recorded.
- Pre-existing artefact text carried forward: the seed-1 `README.md:8-13`
  pre-launch status text (`BUILD AND GATE ONLY ... NOT launched`,
  `control/ and armp/ do not exist`) is false after training completed and
  coexists with the correct seed-1 header; it was not edited
  (seed-1 report Section 13).

## 11. What the experiment establishes

Described in bounded, descriptive language only. Across three
same-evaluator seeds — seed 0 local, seed 1 local, seed 2 Kaggle-trained and
locally evaluated — the ARM-P arm recorded a lower frozen-contract EPE than
the random-initialization CONTROL arm on both the monitor-selected best
checkpoint and the final checkpoint, in every one of the six seed-by-selection
cells. The observed within-run deltas are: seed 0 best -0.3530126 / final
-0.2721821; seed 1 best -0.2528074 / final -0.2499701; seed 2 best -0.2936531 /
final -0.2887042. The environment control (seed 1, Kaggle-trained, locally
evaluated) recorded the same direction on both selections (best
-0.2305591077435014, final -0.2272954162256053), with a within-delta shift of
about +0.022 px against the local seed-1 reference. The frozen contract held
exactly on all 16 scored checkpoints. Configuration identity (all config
fields identical between arms except initialization), strict-load
initialization identity (sha256-verified Stage-1 source, fresh
optimizer/scheduler per arm), full 200-epoch completion on every arm, and
argmin checkpoint selection with the frozen score excluded were verified in
each run's record. That is the extent of what is established: a repeated
directional observation under the stated recipe and contract.

## 12. What it does NOT establish

It does not establish universal superiority: the observation covers three
seeds of one recipe on one dataset split, not all datasets or settings. It
does not establish statistical significance: no test was run, no confidence
interval was constructed, and the 0.0444 px figure is a previously observed
run-to-run spread quoted as context, never as a threshold or decision rule.
It does not establish causal proof that SceneFlow is the mechanism: the
intervention bundles SceneFlow pretraining with everything that comes with
it (weight scale, feature statistics, optimizer trajectory), and the
confounds in Section 13 are preserved. It does not establish causal proof
that pretraining alone explains the improvement, for the same reason. It does
not establish generalization to untested datasets: KITTI `hailo_val` is the
only scored target, and the ARM-P→KITTI D3 domain result was confounded (see
Section 13c). It does not establish any Hailo deployment benefit: the Hailo
toolchain is unverified and no compiler or on-device result is claimed. It
does not establish architectural superiority: the architecture never changed
(Stage-A gate: NO ARCHITECTURE JUSTIFIED), and both arms share it. It does
not infer a mechanism from disparity bins: per-bin tables (reported in full
in each run's record for `<64` / `64–96` / `96–128` / `>=128`) describe where
the EPE difference sits, not why it sits there, and the `>=128` bin's 1,080
pixels (0.028%) must not carry a substantive conclusion.

## 13. Remaining confounds

(a) Stage 1 used the available FT3D A+C subset — 14,460 usable triplets (A:
6,990; B: 0, wholly missing; C: 7,470; Monkaa empty) — not the full SceneFlow
corpus (`stage_b_armp/20260918T053135Z/README.md`; `training_distribution_audit.json`;
`stage_b_armp/20260918T062146Z_stage1_pretrain/README.md`, Naming integrity).
Nothing was copied from `D:\sceneflow_archives`. A later failure on this
subset would not fully discharge the original ARM-P; a pass was recorded as
stated.

(b) The ARM-P checkpoint was trained on the P2A architecture itself, not an
independently verified upstream checkpoint. Upstream provenance was STOPPED:
PROVENANCE NOT ESTABLISHED — no checkpoint file exists anywhere in
`reference/upstream/extracted/StereoNet-master/` (22 files, zero checkpoint
artifacts), and the architecture that would have produced it differs from P2A
everywhere load-bearing (`provenance_audit.json`;
`architecture_compatibility.json`; pre-flight README verdict 1). The viable
path taken (pretraining P2A's own architecture) satisfies strict loading by
construction but inherits this provenance limit.

(c) The ARM-P→KITTI D3 domain result was confounded by different training
histories and is not proof of mechanism. The D3 transfer diagnostic showed
ARM-P numerically worse than P2A on correspondence evidence at GT>=96
(`stage_b_armp/20260918T105050Z_d3_transfer/README.md`, e.g. A_L1 rank-1
-0.1026), but under the D3 proxy with a partial (7/20-epoch, A+C-subset,
unannealed-LR) checkpoint — evidence against the mechanism in its
D3-observable form, explicitly NOT a validation or falsification of ARM-P.

(d) The cross-domain symmetry control showed domain degradation is general
but did NOT show pretraining improves fine-tuning. P2A also degrades
substantially out of domain (KITTI 1.4150 → FT3D 9.0692, +7.6542, x6.409;
ARM-P zero-shot FT3D 3.0486 → KITTI 4.8172650, +1.7687, x1.580), so the D3
negative cannot discriminate between a weak frozen representation and pure
domain shift — classification DOMAIN CONFOUND PLAUSIBLE, with ARM-P neither
validated nor falsified and nothing claimed about EPE after fine-tuning
(`stage_b_armp/20260918T114100Z_domain_symmetry/README.md`).

(e) The `>=128` bin holds only 1,080 pixels (0.028%) — the same count at
every seed and in the environment control — and must not carry a substantive
conclusion. It is quoted in every run's per-bin table for completeness, not
weighted. The `144–160` 16-px bins (9 px) are INSUFFICIENT under the
`MIN_BIN = 1000` rule throughout.

## 14. Open mechanisms

Carried forward from `stage_a_diagnostics/mechanism_verdict.json` (detail and
citations in `stage_a_diagnostics/DIAGNOSTIC_REPORT.md` Sections 10–13),
unchanged by Stage B. The architecture gate remains NO ARCHITECTURE
JUSTIFIED: about 31% of the Stage-A gap sits below 64 px where every
diagnostic looks healthy; the largest single identified effect is a data fix;
both architecture-shaped candidates carry co-adaptation limits; D7 refutes
long-range context on the model's own numbers; the most extreme regime is
correspondence-invalid and worth about 3% of the gap.

Matching representation stays PLAUSIBLE: GT>=96 margin negative in the
majority in all four representations with rank-1 0.3647 (A_L1), persisting
occlusion-removed — against the counter-evidence that no representation
dominates pooled (B_L2 0.6528 > A_L1 0.6469 > C 0.6286 >> D 0.2501) and that
scalar reductions are diagnostic proxies, not the model scoring function.
Cost-volume ambiguity / readout stays PLAUSIBLE with an explicit NOT
IDENTIFIABLE limit from readout/residual co-adaptation (Protocol A holds the
residual fixed, so it cannot separate "readout is fine" from "readout and
refinement are co-adapted and would need joint retraining"). Aggregation
stays with its own NOT IDENTIFIABLE limit on the same grounds
(disparity_initial is co-adapted with the residual and is not a disparity
estimate; loss versus repurposing is not identifiable) and must not be
reported as the bottleneck. D7 refutes the long-range-context argument on the
model's own numbers (error autocorrelation length 32 px horizontal / 16 px
vertical against the 389 px end-to-end receptive field). TRUE GWC remains
UNTESTED: D3's group-averaged representation rescaled the full-channel dot by
1/G, giving identical ranks by construction (G4/G8/G16 rank-1 0.2501) — a
KNOWN GAP, not a refutation; ARM-W group-wise cost was separately REFUTED in
Phase 1 at about 1.776.

Also carried forward is the supervision-coverage finding: the largest single
identified effect is about 2x under-coverage at GT>=96 (effective training
mass 0.444% versus eval 0.898%), which is a DATA fix, not an architecture
change — and the staged FT3D [64,128) pixel count is about 1.23 billion
versus about 559,057 raw KITTI training pixels (about 2,200x), per the
pre-flight audit verdicts 4–5 and `training_distribution_audit.json`.

## 15. Recommendation for the next research branch

The mechanism gap is stated factually: Stage B established a repeated
directional error difference under one recipe without isolating why it
occurs, and Stage A closed with no architecture justified and three named
unknowns still open. The candidates on the table, without any winner picked
and without any subjective scoring between them, are: (U1) a controlled
retraining test of readout–refinement co-adaptation — P2A recipe, cost volume
frozen, refinement retrained from dense alternative inits (or full-model
retraining with an auxiliary GT-candidate loss), judged against the
1.4409826 method mean; (U2) a true group-wise correlation retraining test —
P2A variant replacing the 32-channel signed-difference volume with group-wise
correlation (G=8, G preserved into aggregation), identical recipe and seeds,
screened 40 epochs by 2 seeds against 1.4409826 with a 0.03 px threshold and
no GT<64 regression; (U3) fine-grained below-64 attribution from existing raw
dumps only (reference-vs-P2A signed error and slope per narrow bin, plus
texture and confidence deciles restricted to GT<64), escalating only if still
diffuse to a capacity-matched retrain or low-disparity loss-reweighting
experiment; (U4) a denser-GT or relaxed-threshold sensitivity diagnostic for
edge proximity, which is currently not measurable under its specified
definition; and (U5) a causal data test of the supervision hypothesis —
retraining with GT>=96 oversampled to eval-matched effective mass (about
0.9%) under the identical recipe, with success defined as GT>=96 EPE and
slope moving toward the reference (5.757 / 0.727) with no GT<64 regression.
No experiment is authorized or launched by this record. Any follow-up is a
separate decision, recorded separately, under the one-primary-intervention
rule with its own hypothesis, frozen variables, and success and rejection
criteria.
