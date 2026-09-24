# Stage F final report — sub-1.0 EPE final push (closed at F2, no training)

Tags: **FACT** (repo/record state) / **MEASUREMENT** (audit-computed) /
**EXTERNAL** (papers' own numbers, not evidence for E3) / **INFERENCE**
(reasoning over measurements) / **HYPOTHESIS** (untested) / **VERDICT**
(mechanical outcome). Prose rounds; the audit report carries full precision.

## 1. Objective

**FACT**: beat EPE < 1.000 px as the 3-seed mean FINAL EPE on the frozen
40-scene contract (hailo_val scenes 160–199, valid_pixels 3802797,
contract_match true), starting from the E3 incumbent at mean final 1.1675765.
**FACT**: gap to eliminate 0.1675765 px. **FACT**: hard cap 2 three-seed
campaigns (~15 T4-h); stopping without <1.0 is a valid outcome.

## 2. Starting incumbent

**FACT**: E3 (400-epoch recipe, ARM-P Stage-1 init `3ae6fb3b…`, Adam lr 1e-3
cosine T_max=400, batch 2, crop 256×512, scale aug log-uniform [0.7,1.7], gain
jitter 0.1, masked smooth-L1 beta 1, valid 0<gt<184; monitor = first 10
hailo_val scenes 160–169 inside the contract, so FINAL is primary).
**MEASUREMENT**: per-seed best 1.1860654 / 1.1788394 / 1.1713288, final
1.1628883 / 1.1728822 / 1.1669590; mean best 1.178744511774055, mean final
**1.1675764836832425**; verdict ACCEPT (non-overlapping); spread 0.0147367.
**FACT**: architecture 3× stride-2 (1/8), 6 ResBlocks, difference cost volume
shift "right", 24 candidates (max 184 px), 4 Conv3d, bilinear upsample,
standardised soft-argmin, 6-block dilated refinement, relu(init+res).

## 3. Research findings

Project (**FACT**, readiness §1–3): GT<64 EPE ~1.03 contributes ~0.98 of 1.1676
— the target needs the GT<64 bulk at ~1.03 → ≤~0.85–0.90 (sub-pixel precision
problem); GT≥64 mass ~0.187 clears the gap only with perfect high-disparity
prediction, so coverage is rejected as primary by arithmetic; train loss
300–349 → 350–399 falls 0.675→0.656/0.666→0.653/0.672→0.643 with LR → ~0
(annealing convergence); E3 gain over E0 ~85 % from GT<64; E4 broader FT3D
pretrain REJECT (mean final 1.1949952); Stage A (on P2A, not E3) found
refinement coarse 8.794 → 1.441 with ~2× undercoverage at GT≥96 but
NO ARCHITECTURE JUSTIFIED; true GWC NOT IDENTIFIABLE (changes op, width, sign
at once) — open, not refuted; Stage C ONNX parity fail 1.709e-3 attributed to
near-tie readout pixels via refinement (numerical); device gap local-vs-T4
≤0.000129 px.
EXTERNAL (papers' numbers, not E3 evidence): AcfNet unimodal + focal
1.101 → 0.920 (+conf net 0.867); ADL smooth-L1 0.97 → unimodal CE 0.84 →
multimodal CE + dominant-modal 0.78; CDN 1.09 → 0.98, boundary 3.10 → 2.10;
CoEx top-k k=2 0.7426 → 0.6854 (trained with top-k); StereoDRNet warp-error
refinement 1.03 → 0.95; GwcNet KITTI15 Gwc40 0.602 vs Cat64 0.615 (~2 %),
OpenStereo SceneFlow difference 1.02 vs Gwc8 0.72 (backbones differ); BGNet
edge-aware upsampling 1.40 → 1.17; footprint references StereoNet ~360k D1
4.83 %, ADCPNet-M 0.43M 3.98 %, HITNet 0.45–0.66M 1.98 %, AnyNet 40k 3px 6.2 %
— our D1 ~6 % suggests headroom is supervision/training, not footprint
(**INFERENCE**).

## 4. E3 error audit (summary + link)

**FACT**: zero-training F1 audit, `stage_f/audit/audit_e3.py` → status OK,
authoritative `stage_f/audit/audit_e3.json` + `per_scene.csv` + `run.log`;
full numbers in `stage_f/audit/AUDIT_REPORT.md`. **MEASUREMENT** headline:
finals EPE ~1.163/1.173/1.167; GT<64 EPE ~1.024/1.031/1.031 carrying ~84 % of
mass; top decile ~52 % of EPE; fg ~1.77–1.83 vs bg ~1.03–1.04; occ ~4.1–4.3 on
2.1 %; disc ~3.46–3.60 on 0.33 %; lag-1 autocorr ~0.99 h / ~0.97 v; oracle
+3.5–3.7 worse; refined readouts ~17 px; bimodal rate ~2.6 % at ratio ~31–33x;
warp rho ~0.03; d5 affine gains ≤ ~0.015; train GT<64 ~0.86–0.87 (ratio ~0.84).
**VERDICT**: mechanical selection STOP (branch A1.3).

## 5. Hypotheses considered

- A (distribution supervision): basis R-READOUT reductions + bimodality mass.
  **MEASUREMENT**: no refined readout lowers EPE (all ~17.2–17.4); bimodal
  population bounds ~0.016 px. Status: bounded negligible as inference-only
  target; as training intervention UNTESTED (**HYPOTHESIS** open).
- B (matching-quality ceiling / oracle-init): basis R-ORACLE reduction.
  **MEASUREMENT**: oracle +3.5–3.7 worse on 3/3 → INCONCLUSIVE (off-manifold).
  Status: UNTESTED, not refuted.
- C (warp-photometric into refinement): basis Spearman²×EPE. **MEASUREMENT**:
  rho ~0.03 → mass ~0.0014 px. Status: REJECTED (no signal).
- D (high-disparity coverage): basis GT≥64 mass ~0.187. Status: REJECTED as
  primary by §1 arithmetic (**INFERENCE**).
- E (GWC cost volume): gated on WTA-matching dominance. **MEASUREMENT**: WTA
  hard-argmin EPE ~10.6–10.9 is off-manifold (pre-refinement, ×8 units), not a
  valid dominance showing; E entry 7.58 invalid (§4/audit §7a). Status: NOT
  TRIGGERED, not refuted; would need separate pre-registration.
- F (edge/discontinuity): basis <2 px excess mass = 0.0 from empty stratum.
  Status: UNMEASURED (no ≥3 px GT edges); disc stratum used instead.
- G (deep supervision of init): basis sign-only-scaled mass 13.7.
  Status: INVALID basis (exceeds total EPE); no measurement behind it.
- H (data/regularisation): basis max(0, contract−train) ~0.25.
  **MEASUREMENT**: R-GAP false (ratio ~0.84 > 0.70×). Status: REJECTED.
- I (occlusion): basis excess occ mass ~0.066. **MEASUREMENT**: occ EPE ~4.1–4.3
  but only 2.1 % of pixels. Status: real but far below gap; REJECTED as primary.
- M1 (training-time distribution loss): **HYPOTHESIS** open — frozen probes
  cannot test it (off-manifold, §4); obvious target population bounded ~0.016.
- M2 (warp channel, ~+300 params): REJECTED — no warp signal (C).
- DATA (leakage-free new data): REJECTED — R-GAP false and FT3D already
  rejected as E4; the "leakage-free" check is directory existence only.
- GWC as training candidate: NOT SUPPORTED as dominant, not cheaply
  identifiable — rejected for this campaign.
- Coverage (high-disparity supervision): REJECTED by arithmetic (D).
- Longer training: REJECTED — LR annealed to ~0 at 400 epochs with flattening
  train loss; no open headroom signal.

## 6. Experiments actually run

**FACT**: exactly one — the zero-training E3 audit (F1), local GPU inference
only, 6 subjects × 40 contract scenes + train-split loops, ~10 min wall.
**FACT**: A1 amendment history — committed (`cd2e2b3`) BEFORE the full run:
disclosed 1-scene smoke (scene 160, three finals, CPU) of script `e041f2b`
showed the GT-index oracle ~4.2–4.4 px worse (sparse-KITTI off-manifold feed);
A1.1 added the probe-validity gate (oracle ≥ model ⇒ INCONCLUSIVE); A1.2 fixed
readouts to pass through the model's own refinement (coarse EPEs secondary);
A1.3 redefined selection for the INCONCLUSIVE branch; thresholds/subjects/gates
unchanged. The scene-160 peek motivated only the validity gate; no quantity
from it enters any decision. No GPU training ran.

## 7. Experiments rejected before training and why

M1 rejected as a Stage-F campaign because its frozen probes are untestable
(off-manifold oracle +17 px readouts) and its measurable target bounds ~0.016
px — training it would be a blind spend against a 0.168 gap. M2 rejected: warp
rho ~0.03, mass ~0.0014. DATA rejected: R-GAP false (train GT<64 ~0.86, ratio
~0.84) and FT3D already rejected as E4. Coverage rejected by arithmetic
(~0.187 needs perfection plus bulk fix). GWC rejected: not shown dominant,
not identifiable cheaply. Longer schedule rejected: LR → ~0 at 400, E3
train/val trajectory flat in the readiness report. **VERDICT**: STOP before F4.

## 8. Per-seed results

**MEASUREMENT** (E3 finals/bests — no new model trained): seed0 final
1.1628882757801255 / best 1.1860654178803622; seed1 final 1.1728821951546784 /
best 1.1788393560886006; seed2 final 1.1669589801149234 / best
1.1713287613532017. Full strata/tail/readout/warp/d5/train tables in the audit
report. Every seed: final beats best; oracle +3.4–3.7 worse; refined readouts
~17.2–17.4; bimodal ratio ~30–33x at ~2.5–2.7 % absolute; warp rho ≤ 0.039;
d5 gains ≤ 0.0152.

## 9. Mean results

**MEASUREMENT**: campaign mean final 1.1675765 (exact 1.1675764836832425),
mean best 1.1787445 (exact 1.178744511774055). Gap to target 0.1675765 px.
Closest single checkpoint (seed0 final, 1.1629) still 0.1629 above <1.0.

## 10. Parameter counts

**FACT**: 397,954 parameters, unchanged — zero training, no architecture delta.

## 11. Runtime impact

**FACT**: none — no model change, default runtime/demo untouched.
**MEASUREMENT** (context, `stage_e_recipe/a16w8_latency/`): host ORT CPU EP,
E3 seed-0 fp32 median ~817.9 ms vs A16W8 whole-graph median ~1178.9 ms
(p90 ~1188.2 ms) — A16W8 is the SLOWEST config on that backend; descriptive
only, no claim transfers to Hailo.

## 12. INT8 implications

**FACT**: Stage F trains nothing, so INT8 status is inherited from Stage E.
**MEASUREMENT** (`stage_e_recipe/int8_a16w8/`, SUPPORTED): A16W8 whole-graph
meets R_A16 ≥ 0.5 and residual ≤ 0.5 px on 3/3 seeds (R 0.982/0.979/…,
resid +0.074/+0.089/… px; seed-0 e_A16 1.2603345881778643 vs fp32
1.18598534271453, d1 6.744377888170207). Full INT8 (XINT8) still destroys the
model (E3 int8 EPE ~5.2–5.9 vs ~1.17–1.19 fp32); the Stage-D blocker is
untouched.

## 13. Established

**VERDICT**: (i) SUB-1.0 NOT ACHIEVED under the tested evidence-driven search;
(ii) error is bulk GT<64 sub-pixel mass (~84 %), region-coherent, fg-heavy;
(iii) no readout/warp/calibration probe opens >0.02 px; (iv) train fit stalls
~0.86 on GT<64; (v) the mechanisms ranking G/E/F is invalid/unmeasured as
recorded; (vi) E3 remains the incumbent, frozen.

## 14. Unknown

**HYPOTHESIS** open: M1 as a genuine training-time distribution loss (frozen
probes off-manifold; population bound ~0.016 for the obvious target only);
whether a larger footprint fits train GT<64 below ~0.86; what the ~52 %
top-decile mass is composed of at region level; whether a true identifiable
GWC variant changes matching (unproven, needs its own pre-registration).

## 15. Best final model

**FACT**: E3 seed-0 final,
sha256 `82e58bc441a4382ec449479fe26bf479f6b45e9ace4d79b530abeeb40ea79c6d`
(`stage_e_recipe/kaggle/e3_output/seed0/e3_seed0_final.pth`), EPE
1.1628882757801255. Campaign mean 1.1675765.

## 16. <1.0 achieved

**VERDICT**: NO — SUB-1.0 NOT ACHIEVED UNDER THE TESTED EVIDENCE-DRIVEN SEARCH.

## 17. Why stopped

**FACT**: pre-committed rules returned STOP (R-ORACLE INCONCLUSIVE → A1.3
branch: no readout lowers EPE, R-WARP false, R-GAP false, no leakage-free
source) — F7 not reached. **FACT**: 0 of 2 campaign slots used, 0 GPU-h
training. Stopping without <1.0 is the valid pre-registered outcome.

## 18. Checkpoint SHA256 (all six)

**FACT**: seed0_final
`82e58bc441a4382ec449479fe26bf479f6b45e9ace4d79b530abeeb40ea79c6d`;
seed0_best
`7d3c2109c11339b6fb30cc33f5bd042667740f3bc7345768f4e4b04704836209`;
seed1_final
`7d26844d511b646c96189e44e77f8da15ecaa0818cb1e1edd2d76420c017415e`;
seed1_best
`133c19c41585445da9aabf18919e62f5805b5f8abcf80232808d26d33aa293b0`;
seed2_final
`228487aff2f80bd3514f228f6bc095f18e97b845d7c4fd92da360fc43d229169`;
seed2_best
`1ba8ca09bfabf2c689b6ec7fbbbae1aa70367d33562b1aad06521ba71904cbcf`.

## 19. Reproducibility

**FACT**: commits `476d2c7` (readiness + frozen rules), `cd2e2b3` (amendment
A1), `e041f2b` (audit script), `74a92c6` (A1 implementation — the run
commit). Command: `stage_f/audit_e3.py` with `--subjects all` on CUDA
(defaults: `--limit None`, device cuda-if-available). Env from provenance:
python 3.12.9, torch 2.7.0+cu128, numpy 2.5.1, device cuda. Outputs:
`stage_f/audit/audit_e3.json`, `per_scene.csv`, `run.log` (frozen).

## 20. Final recommendation

**VERDICT**: freeze E3; close the accuracy campaign. The only open lead worth
a future, separately authorised study is a training-time distribution loss
(M1) or a larger-footprint fit diagnostic (can train GT<64 go below ~0.86?) —
both labelled **HYPOTHESIS**, neither funded by this audit's measurements.
