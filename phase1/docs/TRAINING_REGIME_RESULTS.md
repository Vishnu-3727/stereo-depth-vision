# PHASE 1 — TRAINING REGIME RESULTS (Option B dispatch, IN PROGRESS)

Status: BOTH ARMS SCORED 2026-09-14 through `phase1/harness/frozen_eval.py`
under the frozen contract. ARM K20 numbers are frozen harness re-scores
(DETERMINISM_HARNESS §4), NOT retrained. No number below is invented.

Binding numbers: Reference ONNX EPE 1.3134471 / D1 8.1543664%;
convergence_run.pth (= ARM K20) EPE 15.3958267 / D1 88.3922807%;
gap +14.0823796 px EPE. Frozen contract 3,802,797 px, 40 scenes.

Seed policy (plan §3): 1 seed (seed 0) exploration. Seed confirmation (seeds 1,
2) REQUIRED if an arm looks materially better (>= 1.0 px) — not run yet.

Gate (pre-registered, plan §6): CASE A E_P <= 4.0 px (needs also E_K − E_P >=
1.0 px); CASE B 4.0 < E_P <= 12.5 px with E_K − E_P >= 1.0 px; CASE C E_P >
12.5 px or arm difference < 1.0 px; tie rule (|E_K − E_P| < 1.0 px → no
material effect, never upgrades B to A); VOID rule (E_K <= 4.0 px → "budget,
not pretraining", re-plan). No boundary re-drawn after seeing numbers.

Option B ceiling (plan §7/§9): this answers budget, NOT pretraining; a stall
here does not imply pretraining would fail, and an improvement here does not
imply pretraining is unnecessary.

## ARM K20 (existing convergence_run.pth — NOT retrained)

Regime: PyTorch-default random init, seed 0; KITTI hailo_calib 160 scenes
train; no pretraining; batch 2; crop random 256x512; augmentation random crop
+ gain jitter σ=0.1, no h-flip; Adam betas (0.9, 0.999); LR 1e-3; cosine
annealing over 20 epochs; 20 epochs; masked smooth-L1 beta=1.0 valid gt>0 and
gt<176; ImageNet norm; no BN (BN-folded artifact); fp32.
Frozen scores (harness): EPE 15.3958267, D1 88.3922807%, RMSE 20.0022838,
BAD1 96.1074441%, BAD2 92.2553584%, BAD3 88.3922807%, 3,802,797 px.

## ARM K (200-epoch KITTI, cosine) — SCORED 2026-09-14

Full regime record (as-run): initialization PyTorch-default random, seed 0
(`seed_all(0)` superset: python+numpy+torch CPU/CUDA; deterministic algorithms
NOT enabled — baseline recipe did not use them and bilinear-backward has no
deterministic CUDA impl); dataset KITTI 2015 training `_10`, `disp_occ_0`;
dataset size 160 train / 40 eval; train/val split hailo_calib 000000–000159
train, hailo_val 000160–000199 eval, overlap 0 (asserted in code from loader
name lists before training; train_base split='hailo_calib' only, no hailo_val
sample enters the optimizer loop); pretraining none; fine-tuning n/a (KITTI
only); batch size 2; crop random 256x512; augmentation random crop +
independent per-image gain jitter σ=0.1, no h-flip; optimizer Adam betas
(0.9, 0.999); LR 1e-3; scheduler cosine annealing to 0 over 200 epochs;
epochs 200; loss masked smooth-L1 beta=1.0 valid gt>0 and gt<176; normalization
ImageNet; BN handling none (BN-folded artifact, same as baseline); seed 0;
precision fp32; DataLoader shuffle True, num_workers=0, generator
make_generator(0) + worker_init_fn; checkpoint policy final + best-10-scene-val
snapshots with config embedded.
Checkpoints: `phase1/runs/arm_k/arm_k_best.pth` (epoch 180, 10-scene val EPE
8.1202) sha256 bdcec7acd795e61d763e8fb5b924cefafc95f3412507a6030e8a32e0013d0caa;
`phase1/runs/arm_k/arm_k_final.pth` (epoch 199) sha256
04af527d506f9536652ca0c49511fe3ed7e6c185b552d03b3c4fb198d4fcc4cf.
Wall-clock 3564.2 s (~0.99 h). Script `phase1/scripts/train_arm_k.py`, log
`phase1/runs/arm_k/training_log.jsonl`, record `phase1/runs/arm_k/arm_k_record.json`.
Frozen-contract scores (frozen_eval.py, guard contract_match true, 3,802,797
px, strict-load 72/72, 423,586 params):
best snap (PRIMARY — selected by the train-time 10-scene curve per plan §8):
EPE 5.5271927, D1 45.6085613%, RMSE 10.7479083, BAD1 78.2206097%, BAD2
59.5722044%, BAD3 45.6218147%;
final snap: EPE 5.6693759, D1 45.8555111%, RMSE 11.0713642.
Full JSON: `phase1/runs/arm_k/frozen_eval_best.json`,
`phase1/runs/arm_k/frozen_eval_final.json`.
Deviations from plan: none (recipe bit-identical to plan §2 except
seed_all superset + explicit DataLoader generator, both determinism-only, no
training-dynamics effect intended).

## ARM K-LR (200-epoch KITTI, step LR) — SCORED 2026-09-14

Full regime record (as-run): identical to ARM K in every field EXCEPT
scheduler (one variable): step LR mirroring upstream
`finetune-kitti15.py adjust_learning_rate` (1e-3 epochs ≤200 else 1e-4 on a
2000-epoch horizon) scaled to the 200-epoch horizon at the 10% point: lr=1e-3
epochs 0–20, 1e-4 epochs 21–199 (`STEP_EPOCH = 20` in
`phase1/scripts/train_arm_k_lr.py`). Seed 0, batch 2, same split enforcement
(asserted 160/40, overlap 0), same loss/augmentation/norm/BN/precision.
Checkpoints: `phase1/runs/arm_k_lr/arm_k_lr_best.pth` (epoch 0, 10-scene val
EPE 14.7447 — the run NEVER beat its start; LR drop at epoch 21 froze learning,
loss flat ~7–9, val 16–19 throughout) sha256
dc23843296023e0eecc20e432927add623ca76bdd99a53351664370b2361b51a;
`phase1/runs/arm_k_lr/arm_k_lr_final.pth` (epoch 199) sha256
cf2b6c288ed03ca144575294a4bddec5f049a7a55e2e6abc7a9d818cf1250f7b.
Wall-clock 3552.3 s (~0.99 h). Script `phase1/scripts/train_arm_k_lr.py`, log
`phase1/runs/arm_k_lr/training_log.jsonl`, record
`phase1/runs/arm_k_lr/arm_k_lr_record.json`.
Frozen-contract scores (guard contract_match true, 3,802,797 px, strict-load
72/72, 423,586 params): best snap (PRIMARY, same selection rule):
EPE 12.7830 (12.7829883), D1 86.1811977%, RMSE 16.4897604, BAD1 95.2977243%,
BAD2 90.7393164%, BAD3 86.1842481%;
final snap: EPE 12.8253049, D1 83.8580%, RMSE 17.6281338.
Full JSON: `phase1/runs/arm_k_lr/frozen_eval_best.json`,
`phase1/runs/arm_k_lr/frozen_eval_final.json`.
Deviations from plan: the plan names the variant only as "e.g. upstream-style
step LR at the epoch-200-equivalent point" — the 10%-horizon mapping
(drop at epoch 20 of 200) is this dispatch's documented reading of that
"equivalent point"; recorded here, not hidden.

## Arm-vs-arm deltas (frozen EPE, primary snaps)

- Budget effect (K20 → K): 15.3958267 − 5.5271927 = 9.8686340 px, MATERIAL
  (≥ 1.0 px bar). Gap closed: 9.8686340 / 14.0823796 = 70.1%.
- Scheduler effect (K → K-LR): 12.7829883 − 5.5271927 = +7.2557956 px worse
  under step LR, MATERIAL. Cosine ≫ early-drop step at this budget.
- K-LR vs K20: 15.3958267 − 12.7829883 = 2.6128384 px improvement, material
  but small; K-LR absolute level stays near baseline.
- dEPE vs Ref (1.3134471): K20 +14.0823796; K +4.2137456; K-LR +11.4695412.
- Best-vs-final snap gaps (selection-rule robustness): K 0.1421832 px;
  K-LR 0.0423165 px — both far below the 1.0 px bar; snapshot choice does not
  move any verdict.

## Gate verdict (pre-registered boundaries, plan §6)

No ARM P exists in this dispatch (SceneFlow unavailable), so the full E_K −
E_P gate cannot be read off — the formal CASE A/B/C verdict on pretraining is
PENDING Option A, not decided here. What the boundaries say about the arms
that DO exist:
- ARM K EPE 5.5271927 lands in the 4.0 < E ≤ 12.5 absolute window (the CASE B
  band: "4.0 px < E_P ≤ 12.5 px"), quoting that boundary; it does NOT clear
  the CASE A boundary "E_P ≤ 4.0 px" (5.527 > 4.0 by 1.214 px, itself above
  the materiality bar, so this is not a boundary dispute).
- The VOID rule ("if E_K itself ≤ 4.0 px ... report budget, not pretraining")
  does NOT trigger (5.527 > 4.0).
- ARM K-LR EPE 12.7830 exceeds the 12.5 px boundary ("E_P > 12.5 px" is CASE C
  territory on the absolute axis) by 0.283 px — below the 1.0 px bar, so per
  the tie-rule spirit no firm C-vs-B-band claim is made for the scheduler arm;
  it is reported as "at the B/C absolute boundary, stalled".
- Tie rule applied where relevant: best-vs-final differences (< 1.0 px) decide
  nothing; snapshot choice never upgrades or downgrades a placement.

## Option B ceiling (verbatim in substance, per plan §7/§9)

This answers budget, NOT pretraining; a stall here does not imply pretraining
would fail, and an improvement here does not imply pretraining is unnecessary.
The 70% gap closure by budget alone (K20 → K) shows KITTI budget moves the
frozen EPE substantially without any pretraining — but whether SceneFlow
pretraining closes the remaining 4.21 px to the reference is NOT established
by this dispatch and requires Option A (ARM P vs ARM K).

## Seed policy actually applied

1-seed exploration (seed 0) per plan §3. Confirmation seeds 1–2 are REQUIRED
only if an arm clears the materiality bar AND the gate points at CASE A or a
B/C boundary dispute: neither condition holds (no CASE A, no ARM P to dispute
a boundary with), so confirmation was NOT run in this dispatch. No winner is
declared on one seed: cross-seed variation remains NOT VERIFIED, and the ARM K
budget effect, though large (9.87 px), rests on a single seed until a future
dispatch re-runs it at seeds 1–2 alongside ARM P.

## NOT VERIFIED

- ARM P (SceneFlow pretrain) anything: NOT IN THIS DISPATCH (Option B ceiling).
- Cross-seed variation (seeds 1–2): NOT VERIFIED (single seed per arm; see seed
  policy above).
- Whether 200 KITTI epochs / the cosine-vs-step choice generalizes beyond seed
  0: NOT VERIFIED.
- Anything about pretraining efficacy: NOT VERIFIED here by construction.
