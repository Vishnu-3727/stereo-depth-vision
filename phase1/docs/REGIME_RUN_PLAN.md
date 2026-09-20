# PHASE 1, STEP 0, TASK 3 — REGIME RUN PLAN (Priority-1 data-regime partition)

Status: PLAN ONLY. Nothing was trained, fine-tuned, or swept in this dispatch.
The only execution was the timing probe in §4 (N=5 steps, seconds-scale,
weights discarded). All evaluation will use `phase1/harness/frozen_eval.py`
under the Phase 0 frozen contract when real runs exist in a later dispatch.

Question this plan is built to answer: how much of the +14.0823796 px EPE gap
(PyTorch 15.395826671257934 − reference 1.3134470770188373, both VERIFIED LIVE
on 3,802,797 px per `phase1/docs/DETERMINISM_HARNESS.md` §4) is caused by the
training/data regime rather than the architecture? The architecture is already
verified equivalent to the reference ONNX, and the reference weights prove
that architecture can hold 1.3134471 px — so the first intervention improves
the training regime with the architecture frozen.

## 1. Arms — exactly which run, single variable differs

Architecture FROZEN and identical in both arms: feature extractor, cost volume
(shift=none, method=subtract), plain readout, 4 Conv3d aggregation blocks,
6 refinement blocks, dilations (1,2,4,8,1,1), candidate count and spacing —
no change permitted (per dispatch prohibitions; equivalence basis Phase 0 §6).

- ARM K (control): random-init + KITTI only. Random (PyTorch-default) init,
  KITTI `hailo_calib` 160 scenes, no pretraining, then frozen-contract scoring.
- ARM P (pretrain): SceneFlow-pretrained init + THE SAME KITTI phase.
  Upstream SceneFlow pretrain (default 20 epochs per
  `reference/upstream/extracted/StereoNet-master/pretrain-sceneflow/sceneflow-pretrain.py:24-25`)
  produces `checkpoint_sceneflow.tar`; the KITTI phase then runs BIT-IDENTICAL
  to ARM K (same scenes, budget, batch, crop, augmentation, optimizer, LR,
  scheduler, epochs, loss, norm, BN handling, seed handling, precision,
  checkpoint policy) starting from those weights instead of random init
  (mirroring `finetune-kitti15/finetune-kitti15.py:76-81` load-then-finetune).

Single variable that differs: INITIALIZATION (random vs SceneFlow-pretrained).
One variable only — dataset size, split, batch, crop, augmentation, optimizer,
LR, scheduler, epochs, loss, norm, BN handling, seed, precision, checkpointing
are all held equal between arms. Do not change several variables and then
attribute the result to pretraining.

Planned KITTI budget (both arms): 200 epochs. Rationale, pre-registered: 20
epochs reproduces the existing baseline (EXP-016 recipe) but is known too
small to separate "needs more KITTI" from "needs pretraining"; 200 epochs is
the smallest budget that (a) matches the deterministic-training scale already
exercised in-repo, (b) keeps wall-clock under an hour per arm on this machine
(§4), and (c) stays far below the upstream 2000-epoch default so a plateau at
200 epochs is informative, not a ceiling claim. The 20-epoch control
checkpoint (`convergence_run.pth`, EPE 15.3958267) is the reference point for
"did longer KITTI alone move at all", not an arm to re-run.

## 2. Full training-regime record for each arm (to be recorded verbatim per run)

Common (both arms) KITTI phase — values are the PLAN; the later dispatch
records the AS-RUN values field-for-field alongside the checkpoint:

- Initialization: ARM K = PyTorch-default random, seed 0; ARM P =
  `checkpoint_sceneflow.tar` from the upstream pretrain stage (seed 1 default
  per `sceneflow-pretrain.py:32-33` unless the run record states otherwise),
  optimizer state NOT carried (mirrors upstream `:78-81` which loads weights
  only into a fresh Adam); everything else identical.
- Dataset: KITTI 2015 training `_10`, `disp_occ_0`.
- Dataset size: 160 scenes train (`000000_10.png`…`000159_10.png`), 40 scenes
  eval (`000160_10.png`…`000199_10.png`) — disjoint VERIFIED (§6).
- Train/validation split: train `hailo_calib` (0–159); train-time val subset
  first-10 of `hailo_val` for curve monitoring (as in
  `exp_train_convergence.py:98-112,213-216`); FINAL scoring full 40-scene
  `hailo_val` via harness (no early stopping on the 40).
- Pretraining: ARM K = none; ARM P = SceneFlow per
  `pretrain-sceneflow/sceneflow-pretrain.py` (batch 16, RMSprop 1e-3 wd 1e-4,
  ExponentialLR γ=0.9, 20 epochs default, 256×512 random crops, ImageNet norm,
  smooth-L1 masked `disp < 160`, maxdisp 160) — full audit in
  `PRETRAIN_RESOURCE_AUDIT.md` §4.
- Fine-tuning (the common KITTI phase): batch 2; crop random 256×512
  (`CROP_H, CROP_W = 256, 512` per `exp_train_convergence.py:57`);
  augmentation random crop + independent per-image gain jitter σ=0.1, NO
  horizontal flip (inverts disparity — `exp_train_convergence.py:60-95`);
  optimizer Adam betas (0.9, 0.999); LR 1e-3; scheduler cosine annealing to 0
  over the 200 epochs (same family as the 20-epoch baseline
  `exp_train_convergence.py:137-139`, extended horizon pre-registered here);
  epochs 200; loss masked smooth-L1 beta=1.0, valid `gt > 0 and gt <
  max_disparity(=176)` (`src/losses/disparity.py:43-45`,
  `exp_train_convergence.py:154`); normalization ImageNet
  (`NORM_MEAN [123.675, 116.28, 103.53]`, `NORM_STD [58.395, 57.12, 57.375]`,
  `src/datasets/kitti2015.py:38-40,155-165`, NCHW); BN handling: none
  (frozen BN-folded artifact — KNOWN DEVIATION per
  `exp_train_convergence.py:15-21,158-162`, carried into both arms so it is
  NOT the varied variable); seed 0 (`torch.manual_seed` + `np.random.seed`,
  `DataLoader` shuffle True, `num_workers=0`, seeded crop RNG — harness
  `determinism.py` `seed_all`/`make_generator` at run time); precision fp32;
  checkpoint policy: save `{model, config}` every run end + best-train-time-val
  snapshot (config embedded as in `exp_train_convergence.py:247-252`), sha256 +
  weight-sha16 + `strict_load_report` (72 keys / 423,586 params expected per
  `frozen_eval.py:49-50`) recorded via harness.
- Anything the run changes from the above is a PROTOCOL DEVIATION and must be
  recorded as such; deviation voids the single-variable claim for that pair.

## 3. Seed policy

1 seed (seed 0) for exploration: both ARM K and ARM P run once at seed 0.
Multiple seeds ONLY to confirm a candidate that looks materially better:
if either arm clears the materiality bar (§5) AND the decision gate (§6)
points at CASE A or a B/C boundary dispute, re-run THAT arm at seeds 1, 2
(same recipe, new `seed_all` + data-order generator) and report mean ± sd
under the frozen contract. No p-values, no significance claimed beyond the
reported spread; a gate decision never rests on a within-variation difference.

## 4. Measured wall-clock per arm on this machine (timing probe, not a run)

Probe: `phase1/harness/timing_probe.py --steps 5 --batch 2` (VERIFIED LIVE
2026-09-14, CUDA, RTX 4060 Laptop GPU, torch 2.7.0+cu128). One warmup step
untimed, then 5 timed train steps (forward + `masked_smooth_l1` + backward +
Adam step, 256×512 crops, batch 2, `StereoNet(StereoNetConfig())`):

- Per-step times: [0.101, 0.100, 0.105, 0.101, 0.100] s.
- Mean per-step: 0.1016 s. Median: 0.1012 s.
- Steps/epoch @batch 2 over 160 scenes: 80.
- Extrapolated train-only: epoch ≈ 8.1 s; 20 epochs ≈ 162.6 s (0.05 h);
  200 epochs ≈ 1,626 s (0.45 h); 300 epochs ≈ 2,440 s (0.68 h).
- WEIGHTS DISCARDED: the probe never saved weights (by design — stated in
  probe output).
- Excludes: train-time validation passes, checkpoint writes, and the final
  40-scene `frozen_eval.py` scoring. Bound on the last item: the Phase 0
  full-contract PyTorch eval took 12.1 s CUDA (existing record,
  `PHASE_0_FINAL_REPORT.md` §4 — protocol overhead scale, not a new claim).
  Total per-arm wall-clock for the planned 200-epoch KITTI phase is therefore
  expected ≈ 0.5 h train + minutes of val/checkpoint/eval overhead — same for
  both arms (identical KITTI budget by construction).
- SceneFlow pretraining wall-clock: NOT VERIFIED — no SceneFlow pixels exist
  locally (audit §1), so no probe is possible and no extrapolation is given.
  Any pretrain-arm total = KITTI-phase time above PLUS an unmeasured pretrain
  cost (see §7 options). No invented estimate is provided.

If the probe cannot be run safely in a future session, that session writes
NOT VERIFIED and gives no estimate (this session the probe ran — values above).

## 5. What "materially better" means numerically, and where it comes from

Primary metric EPE (pixels, pooled, frozen contract). Materiality bar,
PRE-REGISTERED: an EPE difference of ≥ 1.0 px between two frozen-contract
scores decides "materially different"; anything smaller does not.

Basis (all VERIFIED LIVE, all frozen-contract):

- Identical-input repeat variation on this stack is 0.0: harness check 3
  (`repro_probe.py` → `repro_probe.json`): two identical seeded CPU forward
  passes, bitwise_identical true, max_abs_diff 0.000e+00; both binding
  endpoints re-scored EXACTLY through the harness (every digit —
  `DETERMINISM_HARNESS.md` §4). Eval noise is therefore ~0 px.
- Training carries one KNOWN non-zero nondeterminism source even with controls
  on: `F.interpolate(mode="bilinear")` CUDA backward atomicAdd scatter in
  `src/models/stereonet/regression.py:56-61` (per `DETERMINISM_HARNESS.md` §6;
  forward-only eval unaffected). Its isolated effect is sub-1e-3 px scale in
  the held diagnostics — three orders of magnitude below the 1.0 px bar.
- Cross-seed training variation for the FROZEN architecture is NOT VERIFIED:
  exactly one frozen-arch checkpoint exists (`convergence_run.pth`, seed 0);
  no multi-seed frozen run has been measured. The 1.0 px bar is therefore set
  deliberately conservative — ≈7% of the 14.08 px gap, infinitely above the
  measured 0.0 repeat noise — so a "material" call cannot be an artifact of
  unmeasured seed variance. If the confirmation seeds (§3) later show
  cross-seed sd approaching this scale, the bar must be re-registered upward,
  never argued downward post hoc.

A difference inside run-to-run variation does not decide the gate (§6) — if
ARM P beats ARM K by < 1.0 px EPE, that is NOT a pretraining effect regardless
of the sign.

## 6. Decision gate — PRE-REGISTERED NOW with numeric boundaries

Score every arm on the FULL 40-scene frozen contract (EPE primary). Let
E_P = ARM P (pretrain+KITTI) EPE, E_K = ARM K (random+KITTI) EPE, reference
R = 1.3134471, baseline B = 15.3958267, gap G = 14.0823796 (all px).

- CASE A — regime closes most of the gap → keep improving the regime
  (more data/schedule work, architecture stays frozen):
  E_P ≤ 4.0 px. (≈ R + 2.7 px; closes ≥ ~80% of G: B − 0.8·G ≈ 4.13 px;
  boundary rounded to 4.0 pre-registered.) Requires ALSO E_K − E_P ≥ 1.0 px
  (the pretrain effect itself is material per §5), else see tie rule below.
- CASE B — substantial improvement but plateaus above the reference →
  freeze the regime and move to architecture:
  4.0 px < E_P ≤ 12.5 px AND E_K − E_P ≥ 1.0 px.
  (12.5 px ≈ B − 0.2·G = 12.58 px rounded; i.e. closes ≥ ~20% of G but not
  most of it.)
- CASE C — little or no improvement → architecture becomes the primary target:
  E_P > 12.5 px, OR E_K − E_P < 1.0 px (no material pretrain effect at this
  budget, whatever the absolute level).

Tie/stall rules (pre-registered): (i) if |E_K − E_P| < 1.0 px, the gate reads
the ABSOLUTE level only for CASE A vs C placement and reports "no material
pretraining effect at 200-epoch KITTI budget" — it never upgrades B to A on a
sub-bar difference; (ii) if E_K itself ≤ 4.0 px (KITTI alone closes the gap),
the regime effect is CONFIRMED but the pretraining attribution is VOID —
report "budget, not pretraining" and re-plan (the single-variable question is
answered in favor of budget); (iii) no boundary is re-drawn after seeing the
numbers.

## 7. Evaluation

Every arm scored through `phase1/harness/frozen_eval.py`
(`score_checkpoint`) under the Phase 0 frozen contract: KITTI 2015 /
`hailo_val` 40 scenes / `disp_occ_0` / 1/256 / 368×1232 top-left crop /
`gt > 0` / pooled; guard `refuse_unless_contract` must pass (40 scenes,
3,802,797 px, scale 256.0, split `hailo_val`, gt `disp_occ_0`) or no comparison
is emitted. Primary metric EPE (px); secondary D1 / RMSE / BAD1 / BAD2 / BAD3
(all from `disparity_metrics`, `src/evaluation/metrics.py:52-84`). Reference
for orientation only: 1.3134471 EPE / 8.1543664% D1 (same contract, ONNX).
Each score records: checkpoint sha256 + weight-sha16 + `strict_load_report`
(72/72, 423,586 params) + config + provenance (git HEAD/subject/status, seed,
recipe, dataset/split, torch/numpy/cuda/cudnn, flags, env) per the harness.
No per-image/1/255/10-scene/late-window figure is ever placed beside a frozen
number (no-mixing rule, `BASELINE_CONTRACT.md` §12).

## 8. Data hygiene — how the plan guarantees clean separation

From the ACTUAL scene lists (VERIFIED LIVE, audit §2):

- Train pool: `hailo_calib` = `000000_10.png` … `000159_10.png` (160).
- Eval pool: `hailo_val` = `000160_10.png` … `000199_10.png` (40).
- Overlap: 0 scenes — disjoint VERIFIED. Both arms train ONLY on the 160;
  the 40 are never trained on in any way (no gradient step, no checkpoint
  selection beyond reporting, no crop mined from them, no test-time fitting).
  Checkpoint selection uses the train-time 10-scene curve + final-epoch
  weights, never the 40-scene score as a stopping rule.
- KITTI `testing/` (400+400 images, GT withheld) is used in NEITHER arm.
- Upstream split is the same boundary (`KITTIloader2015.py:26-27`
  `image[:160]`/`image[160:]`), so a future upstream-recipe replication
  cannot leak eval scenes by construction — the plan still requires the
  later dispatch to assert `overlap == 0` from its own loader lists before
  training.
- Pretraining data (SceneFlow, synthetic) is disjoint from KITTI by source;
  the later dispatch must assert the acquired archive's file list contains no
  `0001xx_10.png`-style KITTI names before use. Fine-tuning data (KITTI 160)
  and evaluation data (KITTI 40) are the disjoint sets above.

## 9. SceneFlow is NOT AVAILABLE locally — honest alternatives (decision required)

The plan above (ARM P) is NOT runnable today: §1 of the audit proves the
1.68 GB file is KITTI's own archive, and zero SceneFlow pixels exist on disk.
No pretraining arm runs until this is resolved. Two honest options — costs
stated, NEITHER chosen here; a user decision is REQUIRED:

OPTION A — Acquire SceneFlow, then run ARM K vs ARM P as specified.

- What: download the SceneFlow datasets named by the upstream recipe
  (FlyingThings3D + Monkaa + Driving) from the official source pointer held
  in-repo (`StereoNet-master/README.md:56`:
  Freiburg SceneFlow datasets page). Wire a SceneFlow loader (PFM GT,
  256×512 train crops — `SceneFlowLoader.py:30-79`) + the pretrain stage
  (`sceneflow-pretrain.py`) WITHOUT touching the frozen architecture, then
  execute §1–§8.
- Cost: download size NOT VERIFIED (no download performed; no estimate given
  — any prior size figure is explicitly untrusted); acquisition time
  NOT VERIFIED (bandwidth-dependent, unmeasured); pretrain compute
  NOT VERIFIED (dataset size unknown → steps/epoch unknown → the §4 probe
  cannot be extrapolated); storage outside `phase1/` required (data落地 —
  needs a user-approved location); licence/terms to be checked at acquisition.
- Establishes: the single-variable pretraining question directly (closes /
  plateaus / no-effect per §6).
- Risk: the upstream recipe's exact pretrain variant behind the Hailo export
  is NOT VERIFIED (audit §5), so even a faithful 20-epoch pretrain tests
  "a SceneFlow pretrain", not "THE pretrain".

OPTION B — Substitute a regime intervention runnable with what is on disk
(no SceneFlow).

- What, exactly: run ARM K (200-epoch KITTI, §2) against ARM K20 (the existing
  20-epoch baseline `convergence_run.pth`, no new run) PLUS an optional
  ARM K-LR variant that swaps ONLY the scheduler (e.g. upstream-style step
  LR 1e-3 → 1e-4 at epoch 200-equivalent point vs cosine — one variable,
  architecture frozen). All data from the local 160/40 split (§8).
- Cost: measured — ≈0.5 h per 200-epoch arm (§4) + minutes overhead; zero
  download; zero new data approval.
- CAN establish: whether KITTI budget/schedule alone moves the frozen EPE at
  all (CASE C vs B/C-boundary information); a clean extended-budget control
  for any future ARM P.
- CANNOT establish: the pretraining hypothesis. A stall here does NOT imply
  pretraining would fail, and an improvement here does NOT imply pretraining
  is unnecessary — it answers "budget", not "pretraining". Report must carry
  that ceiling in its conclusion.

DECISION REQUIRED (flagged, not made): the user chooses A or B (or sequences
B-then-A). This dispatch chooses neither, runs neither arm, and creates none
of the later-dispatch artefacts (leaderboard, training-regime results file,
model-development results file, Phase 1 final report).

## 10. NOT VERIFIED (this plan's honesty markers)

- Any SceneFlow EPE/D1 outcome, pretrain loss curve, or pretrain wall-clock:
  NOT VERIFIED (no data, no run).
- That 200 KITTI epochs suffices to reveal a pretraining effect, or that the
  cosine-vs-step scheduler choice is neutral: NOT VERIFIED (pre-registered
  budget/scheduler choice, §2 deviation rule applies).
- Cross-seed frozen-arch variation: NOT VERIFIED (single checkpoint exists;
  §5 bar set conservative for exactly this reason).
- Upstream exact pretrain/finetune variant behind the export: NOT VERIFIED
  (audit §5 — ARM P tests "a" pretrain, never "the" pretrain).
- Historical H1/H2/Phase-2 figures are intentionally ABSENT from this plan
  (different scenes/windows/architectures per the dispatch prohibition) and
  nothing here is calibrated against them.

## 11. Commands executed and outcomes (this dispatch, both tasks)

1. `Get-Location` — OK (repo root).
2. `python phase1\harness\zip_audit.py` — OK (after parents-index fix):
   1,681,488,619 B / 3,415 entries + 1,631,055 B / 1,208 entries (audit §1/§6).
3. `python phase1\harness\data_inventory.py` — OK: 200/400 image, 200/200
   disp, 160/40/200 splits, 0 overlap, full 40-scene list (audit §2).
4. `python phase1\harness\zip_detail.py` — OK: per-dir counts + all-zero
   SceneFlow markers (audit §1).
5. `python -c "import pathlib; ..."` — OK: `data/kitti2015/` = 2 zips +
   training/ + testing/ only.
6. Grep/glob probes — OK: upstream stage paths/lines (audit §3–§4), zero
   `*.pfm` on disk.
7. `python phase1\harness\timing_probe.py --steps 5 --batch 2` — OK:
   mean 0.1016 s/step, 8.1 s/epoch, 200 epochs ≈ 0.45 h train-only, weights
   discarded (§4 above).
8. Reads (no execution): `kitti2015.py`, `metrics.py`, `disparity.py`,
   `exp_train_convergence.py`, `exp_reproduce_hailo.py`, `EXP-016/config.json`,
   `StereoNet-master/{pretrain,finetune,README,dataloader/*}`,
   `DETERMINISM_HARNESS.md`, `BASELINE_CONTRACT.md`, `REFERENCE_BASELINE.md`,
   `PHASE_0_FINAL_REPORT.md`, `comparison.json`, `repro_probe.json`,
   `frozen_eval.py`.

Prohibitions honoured: no training/pretraining/fine-tuning/sweep; no
architecture change; no new loss/variant; no archive extracted; nothing
downloaded; no invented numbers/sizes/durations; no historical EPE placed
beside frozen-contract numbers; no user decision made (§9 flagged, not chosen).
Modified no existing file outside `phase1/` (only two new docs + three
new probe scripts under `phase1/harness/`).
