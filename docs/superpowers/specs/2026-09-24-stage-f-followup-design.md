# Stage F follow-up — zero-compute audit and training-diagnostic pre-registration

- Date: 2026-09-24
- Status: **FROZEN** before any audit run. This document is committed BEFORE the
  audit runs and its thresholds must not be edited after seeing audit output.
- Training is **NOT** authorized by this document. The audit (§2–§5) runs on
  local inference only; the pilots (§6–§8) are designs on paper. A separate
  authorization is required before any training launch.

Statement tags used below: [FACT] (recorded artefact), [MEASUREMENT] (number
from a frozen run), [INFERENCE] (conclusion drawn from measurements),
[HYPOTHESIS] (untested claim a design targets), [VERDICT] (frozen decision).

## Context

[FACT] E3 (397,954 params) is incumbent at 3-seed mean final EPE 1.1675765 on
the frozen 40-scene hailo_val contract; target < 1.0, gap 0.1675765 px.
[FACT] Stage F closed at F2 with zero training (`stage_f/STAGE_F_FINAL_REPORT.md`,
`stage_f/audit/audit_e3.json`). [INFERENCE] Two follow-up hypotheses stayed
open: (1) the ~0.862 train GT<64 floor is not decomposed (capacity /
optimisation / objective / data / GT noise); (2) training-time
probability-volume supervision (M1) is untested — the ~0.016 px bimodality
bound only constrains inference-time readout switching on the frozen model.

[VERDICT] This plan: run a zero-compute audit (local RTX 4060 inference only),
freeze a pre-registration for the training diagnostics, then STOP for
authorization. No training is launched by this plan. Historical records, frozen
contract, closed-stage conclusions untouched; new work lives only under
`stage_f/followup/` and `docs/superpowers/specs/`.

## 1. Files to inspect / reuse (worker reads, does not edit)

- `stage_f/audit_e3.py` — reuse by import: `load_model`, `ckpt_path`,
  `EXPECTED_SHA`, `Kitti2015Stereo` setup, fg/occ/disc masks, texture Sobel,
  `train_split_epe`, `Reservoir`, `d7.CorrAcc`, `atomic_write_json`,
  `git_head`, eval_tier2 gate.
- `stage_e_recipe/kaggle/bundle_e3/src/models/stereonet/{stereonet,regression,cost_volume,refinement}.py`
  — hook points: standardised cost into `DisparityRegression`,
  `disparity_initial`, refined output, 1/8 left/right features.
- `stage_e_recipe/kaggle/bundle_e3/scripts/{finetune_pilot,run_arm}.py`,
  `src/losses/disparity.py::masked_smooth_l1` — training harness (for the
  designs only).
- `stage_e_recipe/e3_verdict.json` — per-seed EPE reproduction gate.

## 2. E3 tensors required (per scene, 40 contract scenes, 3 FINAL ckpts; train split 160 scenes seed0 final)

[FACT] GT disparity (disp_occ_0, /256), noc mask, fg mask (obj_map), left image
(texture), standardised cost `C_std` [24,H,W] (full-res, as fed to softmax),
`p = softmax(-C_std)` if that is the sign the code uses (verify against
`soft_argmin` — sanity: recomputed soft-argmin == model's init, max-abs ≤1e-6),
`disparity_initial` (candidate units) and ×8 coarse px, refined final, 1/8 left
and right feature maps [32,H/8,W/8].

## 3. Error-magnitude + GT-bin audit (exact)

[MEASUREMENT protocol] e = |final − gt| on valid (0<gt<184 for the train-fit
numbers; contract mask for contract).

- Error-mass bins: [0,0.25),[0.25,0.5),[0.5,1),[1,2),[2,3),[3,∞): px frac,
  mean e, **EPE mass = Σe/N_valid** (masses sum to pooled EPE, tol 1e-9).
- GT bins: [0,4),[4,8),[8,16),[16,32),[32,64),[64,∞): px, EPE, mass, and the
  6×6 GT-bin × error-bin mass matrix.
- Top-error decile (per subject, threshold = pooled p90 of e): share of pixels
  and of mass by GT bin, fg/bg, occ/noc, scene, texture decile (existing Sobel
  deciles), disc/nondisc, image region (3×3 grid of the 368×1232 frame).
- Coarse vs final: EPE(coarse×8), EPE(final), Δ = final−coarse per GT bin and
  per error bin; frac improved / worsened / unchanged (|Δ|<1e-6); mass of
  worsened.
- Signed error: mean/median signed e vs integer GT (1-px bins 0..63) and vs
  fractional phases φ8 = (gt/8) mod 1 (10 bins) and φ1 = gt mod 1 (10 bins).
  [HYPOTHESIS] Discretisation signature = amplitude of mean signed error vs φ8.
- Per-scene: EPE, GT<64 EPE, sub-1px mass; paired seed-to-seed scene deltas.
- Same error-mass table on the train split (seed0 final, no aug) — this is the
  floor being decomposed.

## 4. Cost-volume diagnostics (exact)

On valid GT<64 pixels, k\* = round(gt/8) (target bin), p from §2: entropy H(p)
(nats, max ln24=3.178); rank of k\* under p (1 = top); margin = p_top1 − p_top2
and logit margin; mode count = local maxima of p with p ≥ 0.05;
|E[k] − argmax k| (candidate units); mass outside argmax±1; lower-bound mass
p[k=0]; peak-prob cap check (max p observed vs the analytic standardisation
cap for n=24, ≈0.854 with the unbiased torch std the model uses (0.866 with population std; the plan draft quoted 0.867) — decides whether an M1 target can be matched, §7). Each
reported overall, by error bin, and by top-decile vs rest.

Feature rank (seed0 final): SVD of 1/8 left features on ≤200k sampled
valid-location pixels per split → singular spectrum, effective rank (exp
entropy of σ²), rank for 90/95/99 % energy.

Frozen operator comparison (diagnostic for §14 of the brief, no training): on
the same features build 24-candidate volumes with difference-L1, dot product,
cosine correlation; report target-bin rank-1 rate and top1-top2 margin per
operator. [INFERENCE] Info only; no GWC claim.

## 5. Gates (audit only counts if all pass)

[VERDICT] G1 ckpt SHA256 == `EXPECTED_SHA`; G2 params 397954; G3 recomputed
final EPE per seed == e3_verdict.json within 1e-6 and valid px 3802797;
G4 recomputed soft-argmin == model init (1e-6); G5 error-bin and GT-bin masses
sum to pooled EPE (1e-9); G6 train-split seed0 EPE reproduces audit_e3.json
(0.9170082, 1e-6). Any failure → STOP.json, no report.

## 6. Capacity twin (designed, NOT run)

- [HYPOTHESIS] Train GT<64 floor ~0.862 is capacity-limited.
- Intervention: `feature_channels` 32 → 48 (single knob; feeds extractor,
  aggregation, refinement). Params ≈ 895k (exact count measured at preflight;
  must be in [800k, 1.0M] else amend before launch).
- Init — function-preserving embed of the ARM-P Stage-1 checkpoint: old
  weights in the [:32,:32] slice, new output channels default-init, all
  weights *from* new channels into old outputs = 0. Preflight gate: twin
  forward on 3 scenes equals ARM-P forward within 1e-5, and a 1-step gradient
  is nonzero on the new slices.
- Fixed: data, aug, loss, Adam 1e-3, batch 2, crop, seed 0, determinism,
  contract.
- Budget: 200 epochs, cosine T_max=200 (E0 recipe), T4, one seed. Shared
  concurrent control = unmodified width-32 E0-recipe seed 0 run in the same
  bundle and session type (standing rule: control never a historical record).
- Metrics: train GT<64 EPE and train global EPE (no-aug train split, via
  `train_split_epe`), contract final EPE, monitor EPE, D1, params, s/epoch.
- [VERDICT] Interpretation (frozen): Δ train GT<64 (twin − control) ≤ −0.050 →
  CAPACITY SUPPORTED; |Δ| < 0.020 → CAPACITY WEAKENED; else INCONCLUSIVE.
  Neither is proof.

## 7. M1 distribution-supervision pilot (designed, NOT run)

- L = masked Smooth-L1(final) + λ·CE(q, p), same valid mask, p = full-res
  softmax the model already computes (read via forward hook; no
  architecture/readout change).
- Target q_k ∝ exp(−|k − gt/8| / b), k=0..23, **b = 0.5 candidates (4 px),
  fixed**; peak 0.7616 (integer centre) and 0.4323 (half-integer) < 0.8536 cap (verified by §4 cap check before freezing).
- **λ = 0.1**, fixed, no sweep. No focal weighting, no confidence head.
- Budget/init/seed/control identical to §6 (shares the same concurrent control
  run).
- Metrics: train GT<64 + global EPE, monitor, contract final EPE, D1, and §4
  diagnostics (entropy, target rank, mode count) on the pilot vs control.
- [VERDICT] Gate (frozen): PASS if contract Δfinal ≤ −0.030 px; FLOOR-MOVE if
  train GT<64 Δ ≤ −0.050 with Δfinal ≤ 0; else FAIL → "not supported under this
  controlled pilot". Kill: NaN/Inf, zero valid px, strict-load fail, entropy
  collapse to <0.1 nats pooled.

## 8. L1 gate (exact)

[VERDICT] L1 pilot is eligible only if the §3 train-split EPE mass carried by
e < 1 px ≥ 0.1676 px (absolute mass floor = the gap; no ratio-only rule). If
eligible: continue from E3 seed0 final, 40 epochs, fresh Adam lr 1e-4
cosine→0, two arms (Smooth-L1 control vs masked L1), same seed/data/aug. Gate:
Δfinal ≤ −0.030 px.

## 9. Kill conditions (global)

[VERDICT] Audit gate failure → STOP. Pilot numeric gates as above, frozen
before launch. If capacity WEAKENED and M1 FAIL and L1 ineligible/FAIL → no
architecture proposal; reassess data / GT-noise (brief §21). Three-seed
validation (400 ep, seeds 0/1/2) only for one mechanism that passed; primary
stat 3-seed mean FINAL EPE. Fine-volume / correlation / HITNet stay closed
unless §4 shows recoverable signal.

## 10. Compute estimates

- Zero-compute audit: local RTX 4060 inference, 3 finals × 40 scenes + 160
  train scenes, ~15–25 min (F1 audit took ~10 min without volume/SVD).
- Training (after authorization): control 200 ep ≈1.3 T4-h (E0: 4557–4772 s);
  M1 ≈1.4 T4-h; twin ≈2.5–3 T4-h (measured in a 2-epoch timing smoke); L1 2×40
  ep ≈0.6 T4-h. Pilots total ≈6 T4-h. 3-seed 400-ep validation ≈8 T4-h.

## 11. Artifacts

- `docs/superpowers/specs/2026-09-24-stage-f-followup-design.md` — this
  design, frozen thresholds, committed BEFORE the audit runs.
- `stage_f/followup/audit2_e3.py` — audit script (imports from
  `stage_f/audit_e3.py`).
- `stage_f/followup/audit2/{audit2.json, per_scene.csv, run.log,
  AUDIT2_REPORT.md}` — every number tagged
  FACT/MEASUREMENT/INFERENCE/HYPOTHESIS/VERDICT and traceable to the JSON;
  provenance block (utc, git HEAD, ckpt SHAs, dataset, env, device).
- Pilot pre-registration (after audit):
  `docs/superpowers/specs/2026-09-2x-stage-f-pilots-prereg.md`.
