# Stage F readiness report (F0) — sub-1.0 EPE final push

Committed BEFORE any audit code or measurement. Frozen: decision rules in §10.
Repo: `C:\Users\vishn\stereo_depth_vision`, branch `master`.
Authoritative plan: `C:\Users\vishn\.claude\plans\pasted-content-id-c08d-system-prompt-spicy-cloud.md`.
Research digest (manager-verified): `scratchpad/research_digest.md` (cited as Digest).

## 1. Incumbent: E3 (400-epoch recipe, ARM-P Stage-1 init, 397,954 params)

Source for EPE numbers: `stage_e_recipe/e3_verdict.json` (read directly).
Checkpoint hashes: computed by the worker with SHA-256 over the six `.pth` files
(paths per Digest §Project evidence).

| seed | best EPE | final EPE | best sha256 (first 16) | final sha256 (first 16) |
|---|---|---|---|---|
| 0 | 1.1860654 | 1.1628883 | `7d3c2109c11339b6` | `82e58bc441a4382e` |
| 1 | 1.1788394 | 1.1728822 | `133c19c41585445d` | `7d26844d511b646c` |
| 2 | 1.1713288 | 1.1669590 | `1ba8ca09bfabf2c6` | `228487aff2f80bd3` |

Full hashes:

- `stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_best.pth`: `7d3c2109c11339b6fb30cc33f5bd042667740f3bc7345768f4e4b04704836209`
- `stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_final.pth`: `82e58bc441a4382ec449479fe26bf479f6b45e9ace4d79b530abeeb40ea79c6d`
- `stage_e_recipe/kaggle/e3_output/seed1/e3_seed1_best.pth`: `133c19c41585445da9aabf18919e62f5805b5f8abcf80232808d26d33aa293b0`
- `stage_e_recipe/kaggle/e3_output/seed1/e3_seed1_final.pth`: `7d26844d511b646c96189e44e77f8da15ecaa0818cb1e1edd2d76420c017415e`
- `stage_e_recipe/kaggle/e3_output/seed2/e3_seed2_best.pth`: `1ba8ca09bfabf2c689b6ec7fbbbae1aa70367d33562b1aad06521ba71904cbcf`
- `stage_e_recipe/kaggle/e3_output/seed2/e3_seed2_final.pth`: `228487aff2f80bd3514f228f6bc095f18e97b845d7c4fd92da360fc43d229169`

Exact means: best 1.178744511774055, final **1.1675764836832425**.
Frozen contract: 40 scenes (`hailo_val` 160–199), valid_pixels 3802797,
contract_match true. Seed0 per-region detail (final):
`stage_e_recipe/kaggle/e3_output/seed0/e3_seed0.json` —
GT<64 EPE 1.0240654 (px 3627325), 64–96 EPE 2.7765638, 96–128 EPE 8.8771074,
≥128 EPE 20.0393092; bad1 28.2773969 %, D1 6.0150989 %.
`final` is the PRIMARY checkpoint: `best` is monitor-selected on hailo_val
scenes 160–169 (inside the contract) and every E3 seed scores better on final
than on best (Digest; `stage_e_recipe/LEADERBOARD.md`).

## 2. Exact target and gap arithmetic

- Target: 3-seed mean FINAL EPE **< 1.000 px** on the frozen 40-scene contract.
- Gap: 1.1675765 − 1.000 = **0.1675765 px** to eliminate.
- Error-mass table (E3 mean final, pooled contract; Digest):
  GT<64 EPE 1.028, contribution **0.981 of 1.1676**; 64–96 → 0.104;
  96–128 → 0.078; ≥128 → 0.006. bad1 = 28.5 %.
- Consequence: zeroing ALL GT≥64 error still leaves ~0.98. The target needs the
  GT<64 bulk to go ~1.03 → ≤~0.85–0.90, i.e. a sub-pixel precision problem.
- GT≥64 total contribution ≈ 0.187–0.188 px ≥ gap 0.1675765 only in the
  impossible limit of perfect high-disparity prediction; without also fixing the
  bulk it cannot reach the target → high-disparity coverage is REJECTED as the
  primary Stage-F candidate before training, by arithmetic.

## 3. Relevant historical findings (manager-verified digest, tagged FACT + path)

- FACT (`stage_e_recipe/e3_verdict.json`): E3 per-seed best 1.1860654 /
  1.1788394 / 1.1713288; final 1.1628883 / 1.1728822 / 1.1669590; mean best
  1.1787445, mean final 1.1675765; verdict ACCEPT (non-overlapping).
- FACT (`stage_e_recipe/LEADERBOARD.md`): E0 1.2037841/1.2125409; E1 EMA
  INCONCLUSIVE; E2 batch8 REJECT; E3 400ep ACCEPT; E4 broader FT3D pretrain
  REJECT (mean final 1.1949952). E3 seed spread 0.0147367 (S_E3 = 0.0147).
- FACT (`stage_e_recipe/kaggle/e3_output/seed{N}/e3_seed{N}.json` region
  blocks): region split above; E3's gain over E0 ~85 % from GT<64.
- FACT (`stage_e_recipe/kaggle/bundle_e3/scripts/finetune_pilot.py`): train
  hailo_calib 000000–000159, eval hailo_val 160–199; Stage-1 ARM-P init sha
  `3ae6fb3b…`; Adam lr 1e-3; cosine T_max=400; batch 2; crop 256x512; scale aug
  log-uniform [0.7,1.7]; gain jitter 0.1; no flip; loss masked smooth-L1 beta 1,
  valid 0<gt<184 (contract still scores >184). No distribution/deep supervision.
- FACT (same): monitor = first 10 hailo_val scenes (160–169, inside contract) →
  best checkpoints are selection-biased; FINAL is primary.
- FACT (`stage_e_recipe/kaggle/e3_output/seed{N}/…/training_log.jsonl` via
  Digest): train loss 300–349 → 350–399 falls 0.675→0.656 / 0.666→0.653 /
  0.672→0.643; LR annealed to ~0 (annealing convergence, not open headroom).
  T4 wall 9250/9163/9267 s per seed.
- FACT (`src/models/stereonet/`): 3× stride-2 downsample (1/8), 6 ResBlocks,
  difference cost volume shift "right", 24 candidates (max 184 px), 4 Conv3d
  aggregation, cost bilinearly upsampled to full res, standardised
  (unbiased std, eps 1e-6) soft-argmin in candidate units, refinement
  concat(init, RGB) → 6 dilated ResBlocks (1,2,4,8,1,1) → residual, final
  relu(init+res). 397,954 params. `return_stages=True` gives intermediates.
- FACT (`stage_a_diagnostics/`, on P2A 1.441, NOT E3): GT<64 ~1.20 of 1.44;
  refinement coarse 8.794 → 1.441; high-disparity supervision undercovered ~2×
  at GT≥96; verdict NO ARCHITECTURE JUSTIFIED, coverage SUPPORTED as data-side.
  Per-bin/refinement diagnostics were NEVER run on E3 — decomposition must be
  redone in F1.
- FACT (Stage B): ARM-P = P2A arch + scale-coverage recipe + Stage-1 FT3D
  (subsets A,C, 14,460 triplets) init. True GWC NOT IDENTIFIABLE on the old
  platform (changes op, aggregation width, sign at once) — open, not refuted
  (`GWC_IDENTIFIABILITY_DEFINITION.md`).
- FACT (Stage C): ONNX parity fail 1.709e-3 on ARM-P, attributed to near-tie
  readout pixels amplified via refinement residual (numerical, not FP32 claim).
- FACT (hazards): cuDNN non-deterministic (seed spread E3 0.0147); stale guards
  in `stage_e_recipe/kaggle/bundle_e3/scripts/run_arm.py`
  (:31/:121/:146/:170/:203/:205) — E3 ran on E0's bundle, recovered offline.
- FACT (device): E3 contract scoring ran locally (RTX 4060, torch 2.7.0+cu128,
  fp32); comparability calibrated in `stage_e_recipe/e0_device_recheck.json`,
  max |local−T4| = 0.000129 px vs S0_best 0.0174488.

## 4. External research (EXTERNAL — papers' own numbers, not evidence for E3)

- EXTERNAL (AcfNet, AAAI 2020, arxiv 2312.00343-class refs per Digest):
  unimodal distribution supervision + stereo focal loss on PSMNet SceneFlow
  1.101 → 0.920 (0 params); +confidence net 0.867.
- EXTERNAL (ADL, arXiv 2306.15612, 2023): PSMNet smooth-L1 0.97 → unimodal CE
  0.84 → multimodal CE + dominant-modal estimator 0.78.
- EXTERNAL (CDN, NeurIPS 2020): 1.09 → 0.98, boundary EPE 3.10 → 2.10.
- EXTERNAL (CoEx, IROS 2021): top-k soft-argmin 0.7426 → k=2 0.6854 (trained
  with top-k).
- EXTERNAL (StereoDRNet, CVPR 2019): refinement inputs + photometric warp
  error 1.03 → 0.95 (verify).
- EXTERNAL (GwcNet, CVPR 2019): KITTI15 val Gwc40 0.602 vs Cat64 0.615 (~2 %);
  OpenStereo SceneFlow difference 1.02 vs Gwc8 0.72 (backbones differ).
- EXTERNAL (BGNet, CVPR 2021): edge-aware cost upsampling 1.40 → 1.17.
- EXTERNAL (lightweight reference points): StereoNet ~360k KITTI15 test D1
  4.83 %; ADCPNet-M 0.43M D1 3.98 %; HITNet 0.45–0.66M D1 1.98 %; AnyNet 40k
  val 3px 6.2 %. INFERENCE: our D1 ~6 % suggests headroom is
  supervision/training, not footprint.
- Full source list: arxiv 2312.00343, 1903.04025, 2108.05773, 1909.03751,
  2306.15612, 2007.03085, 1807.08865, 1904.02251, 2101.01601, 2004.09548,
  2007.12140, 1810.11408, 2011.09023.

## 5. Audit plan (F1 — `stage_f/audit_e3.py` → `stage_f/audit/`)

Subjects: E3 FINAL seeds 0/1/2 (primary); BEST seeds 0/1/2 (secondary,
labelled monitor-selected-on-160..169). All six sha256-asserted against §1.
GATE first: each final's contract EPE equals `e3_verdict.json` per-seed final
within 1e-6, valid_pixels 3802797, contract_match true; else STOP (JSON status
STOP, exit 1). Reuse (do not reimplement): `StereoNet(StereoNetConfig(**CFG))`
with E3 CFG from `stage_e_recipe/kaggle/bundle_e3/scripts/eval_tier2.py`
(`ev.REPO` set to repo root as `stage_e_recipe/kaggle/e3_measure.py` does),
`model(l, r, return_stages=True)`; `eval_tier2.score` (contract score + bins);
`phase1/harness/frozen_eval.refuse_unless_contract`;
`phase1/harness/bottleneck_diag.py` (bins, occ/disc/texture strata,
init-vs-residual, GT-index oracle, entropy/modes);
`src/evaluation/metrics.disparity_metrics` per stratum;
`stage_a_diagnostics/scripts/d5_refinement.py` (`core_stats`, `oracle_alpha`,
`oracle_beta`); `d7_spatial.py` (`edge_distance_map`, `texture_magnitude`);
`Kitti2015Stereo(occluded=False)` for noc; obj_map via cv2 IMREAD_UNCHANGED +
the loader's `pad_and_crop`. Per checkpoint (pooled contract unless noted):
EPE, D1, RMSE, bad0.5/1/2/3; per-16px-bin EPE + contribution to pooled EPE
(contributions must sum to pooled EPE — assert); per-scene EPE/D1; noc vs occ;
fg/bg (obj_map); discontinuity/edge-distance strata; texture deciles; initial
(8·disparity_initial) vs final EPE per bin; refinement residual magnitude per
bin; corr(residual, initial error); fraction of pixels refinement
improves/worsens per bin; error tail (share of EPE from top 1/5/10 % pixels,
CDF quantiles); spatial clustering (error autocorrelation length, reuse d7);
valid-pixel counts per stratum; the four decision-rule quantities R-ORACLE,
R-READOUT (both readouts + bimodality ratio), R-WARP, R-GAP (train split
0–159) with per-seed booleans; mechanical selection per §10; mechanisms A–I
ranked by measured contribution. Memory: one scene at a time, float32, free
GPU tensors per scene; pixel-level arrays only as needed (reservoir-sample
≤2M pixels per checkpoint for Spearman/quantiles, sampling recorded).
CLI: `--limit N`, `--device cpu|cuda`, `--subjects final|all`.

## 6. Candidate mechanisms A–I (ranked by F1 measured contribution, px)

- A. Cost-volume distribution supervision (M1 family): unimodal Laplacian
  target over the 24 candidates, stereo-focal/CE added to smooth-L1, 0 params.
  Contribution basis: R-READOUT probe reductions + bimodality excess mass.
- B. Matching-quality ceiling (oracle-init family): contribution basis =
  R-ORACLE EPE reduction (GT-index oracle → own refinement).
- C. Warp-photometric error into refinement (M2 family, ~+300 params):
  contribution basis = Spearman² × pooled EPE (variance-explained proxy).
- D. High-disparity supervision coverage: contribution basis = GT≥64 EPE mass;
  REJECTED as primary by §2 arithmetic (recorded before training).
- E. Group-wise correlation cost volume: gated — considered ONLY if the audit
  shows matching (cost-volume winner-take-all accuracy) is the dominant error
  source; then pre-registered separately.
- F. Edge/discontinuity handling (edge-aware upsampling family): contribution
  basis = excess EPE mass within <2 px of GT discontinuities.
- G. Deep supervision of disparity_initial: contribution basis =
  residual sign-error mass (sign-only-scaled probe, d5 method).
- H. Data / regularisation (train–val gap family): contribution basis =
  max(0, contract EPE − train EPE); candidate ONLY if R-GAP and a
  leakage-free data source exists on disk.
- I. Occlusion handling: contribution basis = excess EPE mass on occluded
  pixels, (EPE_occ − EPE_noc) × frac_occ.

## 7. Experiment ordering F0–F8

- F0 (this report): readiness + frozen decision rules, committed before any
  measurement. F1: zero-training E3 audit (§5), local GPU, inference only.
- F2: select ONE mechanism by §10 rules; rejected mechanisms recorded with
  arithmetic/evidence. F3: pre-registration
  (`docs/superpowers/specs/2026-09-2x-stage-f-<mech>-design.md`, committed
  alone before any training): seeds 0/1/2; primary = 3-seed mean final EPE;
  SUCCESS < 1.000; ACCEPT-as-improvement if mean final ≤ E3 mean final − S_E3
  (0.0147) with non-overlapping ranges (Stage-E rule style); REJECT otherwise;
  params ≤ 397,954 + 1 %; INT8 reported descriptively (A16W8 path), not gating.
- F4: implementation behind an explicit flag in a `stage_f/bundle_f1/` copy
  (built by a new `build_bundle_f1.py` modelled on `build_bundle.py
  --experiment e3`, with an `e3`-style patch file); retargets every stale
  guard (epochs 400, init sha, seed/epochs record, projection denominator,
  integrity_guard config). Review: diff vs E3 bundle = only declared deltas;
  flag-off reproduces E3 loss/forward bit-for-bit on a fixed batch.
- F5: low-cost gate — (a) local 1-epoch smoke of main() end-to-end; (b)
  one-seed pilot, local RTX 4060, 200 epochs (~62 min) vs a same-environment
  control (E0/E3 recipe, 200 epochs, same ARM-P init sha `3ae6fb3b…`, same
  seed 0, same GPU + torch build, flag OFF). Margin: pilot final ≤ control
  final − 0.03 (≈2× E0 seed spread). Fail → mechanism REJECTED, no 3-seed
  spend. Pilot is NOT mixed into the 3-seed.
- F6: 3-seed campaign on Kaggle (3 × ~9.2k s T4 ≈ 7.7 GPU-h; 2 concurrent +
  autopush). Pull, score locally with the frozen contract (device gap ≤1.3e-4),
  `verdict.py`-style mechanical verdict. F7: next-or-stop per §9. F8: final
  report `stage_f/STAGE_F_FINAL_REPORT.md` (20 required sections,
  FACT/MEASUREMENT/EXTERNAL/INFERENCE/HYPOTHESIS/VERDICT tags), best model +
  sha256; LEADERBOARD + memory updated. Default runtime/demo untouched.

## 8. Compute estimates

- F1 audit (full): 6 subjects × 40 contract scenes × ~2 forward passes
  (contract score + instrumented loop) + 6 × 160 train-split scenes (light
  loop, finals only strictly required) ≈ ~1500 forwards at 368×1232 on RTX
  4060 ≈ ~10 min wall, peak RAM < 4 GB (one scene at a time, float32,
  ≤2M-pixel reservoirs ≈ tens of MB). No LLM worker resident during the run
  (RAM rule); ≥6 GB free at launch.
- F1 self-test (`--limit 1`, CPU): ~1–2 min, smoke status, no gate.
- F5 pilot: ~62 min local RTX 4060 (200 epochs) + same-cost control.
- F6: 3 × ~9.2k s T4 ≈ 7.7 GPU-h. Hard cap for the campaign: 2 three-seed
  campaigns ≈ 15 T4-h (Kaggle).

## 9. Stop criteria

- Brief §19 governs stopping; instantiated for Stage F as below.
- Hard cap: **2 three-seed campaigns (~15 T4-h)**. Stop on success,
  falsification, or cap; then freeze E3 (or the accepted F6 model).
- F7 rule: a second campaign is allowed ONLY if F6 ACCEPTs, <1.0 is not yet
  reached, AND the F1/F6 measurements predict a credible path to the
  remaining gap: the second mechanism's measured error mass (from the audit,
  re-measured on the F6 model) must be ≥ (F6 mean final − 1.000).
  "It improved, try another" is not a reason.
- Pre-training stop: if §10 selects STOP (SUB-1.0 NOT ACHIEVED; E3 frozen),
  the campaign ends before F4 — stopping without <1.0 is a valid outcome.

## 10. Decision rules (frozen before F1 runs)

  Decision" rules (frozen before F1 runs). Pooled contract EPE, E3 FINAL checkpoints, seeds 0/1/2.
 R-ORACLE: replacing disparity_initial with the GT-derived initial (the existing phase1/harness/bottleneck_diag.py GT-index oracle procedure, unchanged) and running the model's own refinement reduces pooled contract EPE by >= 0.168 px (i.e. the oracle ceiling crosses 1.0) on 3/3 seeds.
 R-READOUT: an inference-only dominant-mode readout (soft-argmin restricted to argmax +-1 candidate, renormalised) OR top-k (k=2,3) readout lowers pooled contract EPE on 3/3 seeds; OR the bimodality rate (second local softmax peak >= 0.5 x the first, >= 2 candidates apart) among the top-decile-error GT<64 pixels is >= 2x its rate among the lower-half-error GT<64 pixels on 3/3 seeds.
 R-WARP: Spearman rho(|I_L warp(I_R, d_final)| summed over RGB, |d_final GT|) >= 0.20 on GT<64 valid pixels on 3/3 seeds.
 R-GAP: train-split (hailo_calib 0-159, same metric code, NOT the contract) GT<64 EPE <= 0.70 x contract GT<64 EPE on 3/3 seeds.
 Selection: M1 (distribution supervision, 0 params) iff R-ORACLE and R-READOUT. M2 (warp-error channel into refinement, ~+300 params) iff NOT R-ORACLE and R-WARP. If neither: data/regularisation candidate only if R-GAP and a leakage-free data source exists on disk; otherwise STOP the campaign before training (SUB-1.0 NOT ACHIEVED; E3 frozen). GWC is considered only if the audit shows matching (cost-volume winner-take-all accuracy) is the dominant error source, and must then be pre-registered separately. High-disparity coverage is rejected by arithmetic (GT>=64 contributes ~0.187 px, target gap 0.168 cannot be met from it alone without also fixing the "bulk).

## Amendment A1 (2026-09-24, before the full F1 run; authorised by the user)

Disclosure: a 1-scene smoke run (scene 000160, three E3 final checkpoints, CPU) of the first audit script (e041f2b) was seen before this amendment. On that scene the GT-index oracle of R-ORACLE made pooled EPE WORSE than the unmodified model by ~4.2-4.4 px. KITTI GT is sparse, so substituting GT/8 only at valid pixels feeds the refinement an input far from its training distribution. No other smoke quantity is used to motivate this amendment.

A1.1 Probe validity gate (project lesson: a diverging probe must never produce a confirmation, EXP-E1 record). A ceiling probe counts only if it improves on the unmodified model. If the R-ORACLE oracle EPE is >= the model's pooled EPE on a seed, R-ORACLE on that seed is INCONCLUSIVE (off-manifold), not FAIL. R-ORACLE is INCONCLUSIVE overall if any seed is INCONCLUSIVE.

A1.2 R-READOUT correction (implementation fault, not a rule change): the dominant-mode and top-k readouts replace disparity_initial for the whole image and are passed through the model's own refinement (final_alt = relu(init_alt + refinement(init_alt, left))). Their pooled contract EPE is compared with the model's pooled contract EPE. Coarse pre-refinement readout EPEs are reported as secondary only. The bimodality limb is unchanged.

A1.3 Selection when R-ORACLE is INCONCLUSIVE: M1 iff an alternative readout (A1.2) lowers pooled contract EPE on 3/3 seeds (the bimodality limb alone is not sufficient in this branch); else M2 iff R-WARP; else the data/regularisation branch as frozen (R-GAP + leakage-free source on disk); else STOP. When R-ORACLE is PASS or FAIL, the frozen selection applies unchanged.

A1.4 Unchanged: all thresholds (0.168, 2.0, 0.20, 0.70), 3/3-seed requirements, subjects, contract gate, stop criteria, campaign cap.
