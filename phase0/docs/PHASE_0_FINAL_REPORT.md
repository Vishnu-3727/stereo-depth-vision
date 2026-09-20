# PHASE 0 FINAL REPORT — baseline lock & reproduction gap

Phase 0 only. No architecture was changed, nothing was trained, no new
experiment was run, no existing file was modified. The three files
`phase0/docs/BASELINE_CONTRACT.md`, `phase0/docs/REFERENCE_BASELINE.md` and
this report are the only outputs. Numbers below are either VERIFIED LIVE
(measured by Phase 0 read-only execution, in-memory only, nothing written
outside `phase0/`) or quoted from the cited record with their full protocol
label. Unverifiable items are marked NOT VERIFIED, never filled in.

## 1. Executive Summary

- The primary evaluation contract is frozen (`BASELINE_CONTRACT.md`): KITTI
  2015, `hailo_val` scenes 160–199 (`_10`), `disp_occ_0`, 368x1232 top-left
  crop, GT 1/256, `gt > 0`, pooled pixels, EPE in px + D1-all.
- Reference baseline (ONNX) VERIFIED LIVE under that contract: EPE
  1.3134471 px, D1 8.1543664%, RMSE 2.5830, 3,802,797 px — exact match to the
  recorded EXP-005 figure, every digit. Hailo-exact 8.2236865% D1 also
  VERIFIED LIVE; it is a different protocol and is kept separate.
- PyTorch baseline (existing `results/training/convergence_run.pth`, EXP-016
  recipe, unmodified frozen architecture) evaluated live under the SAME
  contract for the first time: EPE 15.3958267 px, D1 88.3922807%, same
  3,802,797 px. Checkpoint loads strict-OK against current source (72/72 keys).
- Exact gap under the frozen contract: EPE gap +14.0823796 px, D1 gap
  +80.2379144 points (PyTorch minus reference).
- Architecture equivalence VERIFIED LIVE: ONNX weights load positionally into
  the PyTorch model (36 convolutions, 423,586 params, Siamese sharing
  asserted in code and confirmed); weight-loaded PyTorch vs ONNX final
  disparity on a real scene differs by max 8.24e-4 px — exact match to the
  recorded EXP-011 value. The gap is therefore NOT an implementation mismatch.
- Provenance is different (vendor SceneFlow->KITTI pretrained vs our random
  init, 160 scenes, 20 epochs, no BN, no SceneFlow) and its quantitative share
  of the gap is UNKNOWN: no matched experiment partitions regime vs
  architecture.
- Critical question answer: POSSIBLE (see §6–9). The frozen architecture is
  proven capable of 1.31 px (the reference weights live in it), but our recipe
  has never produced better than 15.40 px under the frozen contract, extended
  plain-readout training stalls (~14.4 px recorded), and the upstream data
  regime was never run here — plausibility is neither established nor excluded.

## 2. Frozen Evaluation Contract

Per `BASELINE_CONTRACT.md` (each item with source file+line evidence there):
KITTI 2015 `_10` / `hailo_val` 40 scenes (160–199) / 368x1232 pad-bottom-right
then top-left crop, no resize / `disp_occ_0` / GT 1/256 / valid `gt > 0` /
EPE = mean|pred-gt| in px / D1 = (err>3 AND err>5%gt) in % / pooled over all
valid pixels / output ReLU-clamped disparity, scored raw. Two verified
contract facts worth restating: the crop deletes 7 bottom rows + 10 right
columns per frame (`src/datasets/kitti2015.py:57-60` — not disparity-neutral);
training-only masks (`< max_disparity` in `src/losses/disparity.py:43-45`)
are NOT part of evaluation. No-mixing rule is binding (§12 of the contract).

## 3. Reference Baseline

Per `REFERENCE_BASELINE.md`. `reference/onnx/stereonet.onnx`, KITTI2015 /
hailo_val 40 scenes / 3,802,797 px / 1/256 / D1-all pooled:
EPE 1.3134470770188373, D1 8.154366378221082, RMSE 2.5829573145561677,
bad1 38.89637022433751, bad2 16.079375259841637, bad3 8.639640769675584 —
ALL VERIFIED LIVE (exact, every digit, vs `experiments/EXP-005/metrics.json:62-80`).
HAILO-EXACT RESULT: 8.2237% D1 (8.223686464356268, per-image, 1/255) —
VERIFIED LIVE vs `metrics.json:24-42`; never an EPE. Vendor config
`stereonet.yaml:43-44` still mislabels it `EPE`: documented discrepancy,
resolved in favor of the evaluator. HEF silicon behavior: NOT VERIFIED
(no device; unchanged from prior phases).

## 4. PyTorch Baseline

Strongest valid existing checkpoint under the frozen architecture and the
frozen contract: `results/training/convergence_run.pth` (1,723,643 bytes).
Identity: container `{'model': OrderedDict(72 tensors), 'config': EXP-016
recipe}` — verified live; `strict=True` load into current
`StereoNet(StereoNetConfig())` succeeds (72/72 keys, 423,586 params), so this
checkpoint IS the frozen architecture (shift="none", subtract, plain
readout, no BN). Recipe: seed 0, 20 epochs, Adam 1e-3 cosine, batch 2,
256x512 random crops + gain jitter, masked smooth-L1, KITTI `hailo_calib`
train (160 scenes), random init (`experiments/EXP-016/config.json`,
`scripts/exp_train_convergence.py:128-162`).
Live 40-scene evaluation under the frozen contract (CUDA, 12.1 s, read-only):
EPE 15.395826671257934, RMSE 20.002283814769715, D1 88.39228073441733,
bad1 96.10744407340177, bad2 92.25535835859763, bad3 88.39228073441733,
3,802,797 valid pixels. Full label: KITTI2015 / hailo_val 40 scenes /
3,802,797 px / 1/256 / D1-all pooled / EPE+D1 / convergence_run.pth.
Note: the prior EXP-016 record reported only 10-scene train-time val
(EPE 19.216, D1 91.12 — different protocol, never substituted); the
40-scene official-contract number above is new Phase 0 measurement of an
EXISTING checkpoint, not a new experiment. No other same-architecture
checkpoint exists under this contract: Phase 2 H1/H2/E arms all change the
readout and/or shift and/or block counts, so none qualifies as the frozen
PyTorch baseline (their numbers are cited in §10 with ceilings, never mixed).

## 5. Exact Gap

Under the frozen contract, same scenes/pixels/GT/scale/mask:
EPE gap = 15.395826671257934 − 1.3134470770188373 = +14.082379594239097 px.
D1 gap = 88.39228073441733 − 8.154366378221082 = +80.23791435619625 points.
Both VERIFIED LIVE (both endpoints measured in Phase 0). For orientation
only, with protocols labeled and NOT part of the gap: the best recorded
from-scratch number on 40 scenes (deterministic Stage B, 6-block seed 0,
shift=left + standardised readout — a MODIFIED architecture) is EPE
2.2955182 / D1 15.6045143 (`phase2/docs/PHASE_2_REGISTRY.md:919-921`),
i.e. 0.9821 px / 7.4501 pt above reference — informative about what a changed
architecture + 200 epochs can reach, not a measurement of the frozen baseline.

## 6. Architecture Equivalence

Q4 answer: YES — the PyTorch implementation matches the reference ONNX.
Verified live in Phase 0 (not quoted): `load_onnx_weights`
(`src/models/stereonet/onnx_weights.py:60-133`) loads 36 convolutions /
423,586 parameters with Siamese branch sharing confirmed by assertion
(`onnx_weights.py:76-82`); weight-loaded PyTorch vs ONNX final disparity on
hailo_val scene 0: max abs diff 0.000823974609375 px, mean abs 9.14e-6 px —
the max figure exactly reproduces recorded `results/equivalence/stage_differences.json:32-37`
(disparity_final). Topology/shape account per source: features (B,32,23,77)
(`feature_extractor.py:26-56`), 12-candidate subtract volume, no-op
reference shift (`cost_volume.py:45-55,123-138`), 4+1 Conv3d aggregation
(`aggregation.py:25-40`, slope 0.01 `blocks.py:22-25`), upsample-first
soft-argmin align_corners=True (`regression.py:56-61`), 6-block dilated
image-guided refinement + residual + ReLU (`refinement.py:39-56`,
`stereonet.py:98-111`), ResBlock slopes 0.2/0.01 (`blocks.py:22-48`), no BN
anywhere (folded at export, `blocks.py:8-15`). Degenerate shift is a property
of BOTH (all 12 slices identical — reference behavior reproduced, EXP-010
record stands, re-verified by construction, not re-measured here).
Consequence: the 14.08 px gap is not an implementation error.

## 7. Weight/Provenance Audit

Q5 answer: weights/provenance DIFFERENT, mapping EXACT.
Reference: vendor pretrained lineage (SceneFlow pretrain + KITTI fine-tune per
upstream `StereoNet-master` scripts held in `reference/upstream/extracted/`;
exact commit/seed/schedule for this artifact NOT VERIFIED — no training log
held). Ours: PyTorch-default random init, seed 0, 160 KITTI scenes, 20
epochs (config embedded in checkpoint, verified live — matches
`experiments/EXP-016/config.json` field for field). The positional
ONNX→PyTorch mapping exists (`onnx_weights.py`), is shape-asserted per
tensor, Siamese-sharing-checked, and verified live (§6) — so weight
EQUIVALENCE is achievable mechanically; weight IDENTITY is absent (ours never
saw the reference's data/schedule). Checkpoint audit: `convergence_run.pth`
= legitimate from-scratch baseline (selected by architecture match +
recipe record, not by prettiest number); every Phase 2 `.pth` verified
present by listing but disqualified as the frozen baseline by architecture
difference (readout/shift/blocks) — loadability of each NOT VERIFIED
(loading them was unnecessary; nothing hinges on it).

## 8. Preprocessing Audit

Q6 answer: evaluation pipelines EQUIVALENT by construction; training
preprocessing DIFFERENT by design; upstream training preprocessing NOT
VERIFIED in detail. Eval: both Phase 0 live runs (reference §3, PyTorch §4)
used the identical code path — same `Kitti2015Stereo(split="hailo_val",
disparity_scale=256.0)` + same `normalize()` (ImageNet stats
`kitti2015.py:38-40`, NCHW `kitti2015.py:155-165`; vendor stats match
`stereonet.yaml:21-30`, compiler-inserted norm `kitti2015.py:155-162` doc).
RGB order verified in code (`kitti2015.py:96`, BGR→RGB). Training differs:
256x512 random crops + per-image gain jitter, no h-flip
(`scripts/exp_train_convergence.py:56-95`) vs full-frame eval; upstream
KITTI loader/augmentation specifics beyond the held scripts were not
audited — recorded as NOT VERIFIED rather than assumed identical.

## 9. Training-Regime Audit

Reference regime (upstream, from held sources): SceneFlow pretraining (batch
16, Adam 1e-3) then KITTI fine-tune (batch 16, Adam 1e-3, exponential γ=0.9,
up to 2000 epochs, maxdisp 160, subtract volume) — from
`pretrain-sceneflow/sceneflow-pretrain.py:19-59` and
`finetune-kitti15/finetune-kitti15.py:23-79`; exact recipe behind THIS export
NOT VERIFIED. Our regime: batch 2, 20 epochs, cosine to 0, 160 scenes,
random init, no BN, fp32, no pretraining (`EXP-016/config.json`).
Differences (pretrained vs random; ~35k SceneFlow + full KITTI fine-tune vs
160 scenes/20 epochs; batch 16 vs 2; BN-training-time vs BN-folded-from-start;
exp-schedule vs cosine; full-frame vs 256x512 crops): CONFIRMED as regime
description from code+config. Q7 answer: training/data regime is the largest
IDENTIFIED contributor class, but its quantitative share is UNKNOWN — no
matched-budget experiment (e.g. our recipe + SceneFlow, or reference recipe
at our budget) exists, and Phase 0 runs none. Q8 answer: the unexplained
portion is likewise UNKNOWN as a number; candidates are listed in §11 with
ceilings, none quantified.

## 10. Existing Evidence Audit

Usefulness for Phase 0 only (no reruns; closed correspondence branch stays
closed; LEVEL D ceiling respected — nothing below is a correspondence claim):
- EXP-005 — USEFUL FOR BASELINE (the frozen reference record; re-verified
  live, §3). `experiments/EXP-005/metrics.json`, `scripts/exp_reproduce_hailo.py`.
- EXP-016 (+CORRECTION.md) — USEFUL FOR BASELINE (checkpoint identity/recipe)
  + USEFUL FOR TRAINING GAP (loss 11.49→7.61 with flat val proves pipeline
  runs without learning to generalise at this budget). Original "validation
  improves" sentence WITHDRAWN by its own CORRECTION (PREVIOUS CLAIM NOT
  SUPPORTED BY CURRENT EVIDENCE — honoured here).
- EXP-011 / `results/equivalence/stage_differences.json` — USEFUL FOR
  BASELINE (implementation↔ONNX equivalence; confirmed live, §6).
- EXP-001 / architecture doc / `results/onnx_inspection/summary.json` —
  USEFUL FOR BASELINE (params 423,586/623,138, 168 nodes, shapes).
- EXP-010 — IDENTIFICATION-ONLY + USEFUL FOR MODEL DEVELOPMENT (degenerate
  shift identity; the property Phase 1 must decide about, not a verdict).
- EXP-007 — USEFUL FOR BASELINE (right-image dependence +89–90 D1 pt on its
  OWN 1,837,304-px subset, `metrics.json:10-40`) + IDENTIFICATION-ONLY
  (mechanism reading capped at LEVEL D). Its baseline (EPE 1.4422/D1 9.3315%)
  is protocol-labeled and never mixed with §3.
- H1/H1-v2 — USEFUL FOR TRAINING GAP (plain-readout 200-epoch stall at EPE
  ~14.41, gradient starvation — recorded in registry; NOT re-verified here,
  cited with their ceilings: one recipe, pre-determinism harness for v1).
- H2 + seed replication — USEFUL FOR MODEL DEVELOPMENT (standardised readout
  restores trainability; 3 seeds) + USEFUL FOR TRAINING GAP (best changed-arch
  late-window EPE 3.7–4.3). Ceiling: 10-scene late windows, one recipe.
- Stage B deterministic 6-block + A.3 shift factorial — USEFUL FOR TRAINING
  GAP (40-scene EPE 2.2955 vs 8.3717, one seed/arm; effect 59x seed range) +
  USEFUL FOR MODEL DEVELOPMENT. Ceilings: single seed per factorial arm;
  standardised readout (not frozen arch).
- E1/E2/E3/E3b/E3c/O6, emergence, 3x3 block table — USEFUL FOR MODEL
  DEVELOPMENT with registry addenda ceilings (E3b "four match six"
  SUPERSEDED; E3c "reducible-to-3" CONTRADICTED by O6 remeasurement; 4-block
  INCONCLUSIVE; determinism controls required). No p-values anywhere; none
  invented here.
- EXP-002/009/017-original — SUPERSEDED/WITHDRAWN (kept with corrections;
  never cited as evidence). EXP-013/014/015 — NOT RELEVANT to the accuracy
  baseline (latency/quantization; deployment-relevant only).

## 11. Error Budget

Confidence in {CONFIRMED, SUPPORTED, POSSIBLE, UNKNOWN}. No causality claimed
beyond the evidence.

| Source | Status / Evidence | Confidence |
|---|---|---|
| DATA (160 scenes, no SceneFlow, random init, 20 epochs) | Regime gap vs upstream described from code+config (§9); contributes unquantified share | SUPPORTED (as contributor class); UNKNOWN (share) |
| TRAINING (no BN from start, batch 2, cosine, 256x512 crops) | Same regime description; BN deviation `exp_train_convergence.py:158-162`; H1-v2 stall recorded | SUPPORTED (class); UNKNOWN (share) |
| FEATURE EXTRACTION (linear downstack, 1/16, 32ch) | Code + ONNX walk; accuracy cost never isolated | POSSIBLE |
| COST VOLUME (degenerate no-op, 12×16px steps, 176px ceiling) | Identity CONFIRMED (EXP-010 + §6); ceiling non-binding on hailo_val (GT max 153.04 recorded); causal share UNKNOWN | CONFIRMED (property); UNKNOWN (share) |
| AGGREGATION (5 Conv3d, 4.2% MACs) | Never isolated (E4-cheap unrun) | POSSIBLE |
| READOUT (plain soft-argmin saturation, entropy 0.0000 recorded) | H1 mechanism record; fixed by standardisation in H2 (different arch) | SUPPORTED (plain-path unlearnability, single-recipe ceiling) |
| REFINEMENT (90.6% MACs, carries final scale per EXP-006/008 correlation records) | Cost share CONFIRMED; "oversized" verdict CONTRADICTED-IN-PART by pruning/retrain records | SUPPORTED (load); UNKNOWN (right-sizing) |
| POSTPROCESSING (ReLU clamp only) | `stereonet.py:110-111`; no evidence of error contribution | UNKNOWN |
| DEPLOYMENT (BN folding, int8, HEF) | int8 cost measured own-stack (EXP-015, cited not verified); HEF silicon unmeasured | POSSIBLE (int8); UNKNOWN (HEF) |
| UNKNOWN (scene-class modes, occlusion handling, subpixel-between-candidates) | No masks/detectors in any record; Middlebury absent | UNKNOWN |

## 12. Confirmed / Supported / Possible / Unknown

- CONFIRMED: frozen contract items (§2); reference numbers (§3, live);
  PyTorch numbers (§4, live); exact gap (§5, live); arch↔ONNX match (§6,
  live); weight-mapping exactness (§6–7, live); regime description (§9);
  degeneracy as a property; vendor metric mislabel; EXP-016 correction.
- SUPPORTED: data/training regime as the largest contributor CLASS (not
  share); plain-readout unlearnability at the tested recipe; refinement load
  concentration; int8-on-own-stack cost (cited).
- POSSIBLE: critical-question verdict (frozen arch plausibly trainable to
  reference — capability proven by reference weights, learnability not
  demonstrated, SceneFlow untested); feature/aggregation/subpixel limits;
  HEF/int8 deployment effects on the gap.
- UNKNOWN: quantitative regime-vs-architecture partition; unexplained gap
  share; 4-block verdict; causal block-count story; scene-class failure
  modes; HEF silicon numbers; per-checkpoint loadability beyond the two
  loaded here; upstream exact training recipe for this export.

## 13. Three Phase-1 Priorities

Ranked by information value per compute before any structural commitment.
Implemented none.

1. HIGHEST PRIORITY — Data-regime partitioning run under the FROZEN
   architecture: SceneFlow pretraining (already on disk:
   `data/kitti2015/data_scene_flow.zip`, 1.68 GB) + matched-budget controls,
   deterministic harness, frozen-contract scoring. Expected benefit:
   directly measures how much of the 14.08 px gap is regime — the only
   question that decides between data-first vs architecture-first Phase 1.
   Evidence: upstream recipe exists in-repo; reference weights prove the arch
   can hold 1.31 px. Risk: ~22 h-scale download/compute estimate recorded
   previously; single-recipe ceiling remains. Compute: HIGH (one-off, GPU).
   Ranked first because every architecture edit is uninterpretable until the
   gap is partitioned.
2. SECOND PRIORITY — Readout conditioning as a training variable
   (standardised readout), pre-registered with entropy/gradient gates and
   frozen-contract scoring alongside. Expected benefit: tests the SUPPORTED
   binding constraint (plain-readout starvation) with the H2 recipe as
   reference. Evidence: H1 mechanism + H2 3-seed replication. Risk: departs
   from the frozen baseline — requires explicit re-baselining and forbids
   comparison against §4 numbers. Compute: MODERATE (200-epoch scale, GPU).
   Ranked second because it is the cheapest architectural hypothesis with
   multi-seed support, but only meaningful after (1) frames it.
3. THIRD PRIORITY — Determinism + protocol-discipline harness (seed ranges,
   bit-identical weight checks, pre-registered materiality scale from the
   measured 6-block range, no-mixing CI guard). Expected benefit: de-risks
   (1) and (2); the campaign's costliest recorded errors were uncalibrated
   gates and mixed protocols. Evidence: Stage A/B determinism record.
   Risk: near zero (no science claims). Compute: NEGLIGIBLE. Ranked third
   because it is an enabler, not an answer — but it must precede (1).

## 14. Phase-1 Entry Decision

Q1 exact reference baseline: EPE 1.3134471 px / D1 8.1543664% (3,802,797 px,
40 scenes, 1/256, pooled, ONNX) — VERIFIED LIVE.
Q2 exact PyTorch baseline, same protocol: EPE 15.3958267 px / D1 88.3922807%
(same pixels/scenes/GT, convergence_run.pth) — VERIFIED LIVE.
Q3 exact numerical gap: +14.0823796 px EPE / +80.2379144 pt D1 — VERIFIED LIVE.
Q4 architecture match: YES — VERIFIED LIVE (§6).
Q5 weights/provenance: DIFFERENT provenance, EXACT mapping (§7).
Q6 preprocessing/evaluation: eval EQUIVALENT; training preprocessing
DIFFERENT; upstream training detail NOT VERIFIED (§8).
Q7 regime-explained share: largest identified class, share UNKNOWN (§9).
Q8 unexplained share: UNKNOWN as a number; candidates in §11.
Q9 highest-confidence bottlenecks: (i) data/training regime class,
(ii) plain-readout gradient starvation at tested recipe, (iii) degenerate
cost volume as CONFIRMED property with UNKNOWN causal share.
Q10 attack first: (1) data-regime partitioning, then (2) readout, on top of
(3) determinism/protocol harness — §13.

PHASE-1 ENTRY BLOCK:
REFERENCE BASELINE: EPE 1.3134471 px / D1 8.1543664% / 3,802,797 px /
  KITTI2015 hailo_val 40sc / 1/256 / pooled / reference/onnx/stereonet.onnx
  (VERIFIED LIVE)
PYTORCH BASELINE: EPE 15.3958267 px / D1 88.3922807% / same contract /
  results/training/convergence_run.pth (strict-load OK, VERIFIED LIVE)
GAP: +14.0823796 px EPE / +80.2379144 pt D1 (VERIFIED LIVE)
ARCHITECTURE MATCH: YES (36 convs, 423,586 params, Siamese-shared;
  final-output max diff 8.24e-4 px — VERIFIED LIVE)
WEIGHT MATCH: mapping EXACT, provenance DIFFERENT (vendor
  SceneFlow->KITTI vs random-init 160sc/20ep)
TRAINING-REGIME DIFFERENCE: CONFIRMED as description (pretrained/full-data/
  batch-16/BN-time vs random/160sc/batch-2/no-BN/cosine); share UNKNOWN
EVALUATION DIFFERENCE: NONE (identical code path both live runs)
KNOWN BOTTLENECKS: regime class (SUPPORTED) / plain-readout starvation
  (SUPPORTED, recipe ceiling) / degenerate volume property (CONFIRMED,
  causal share UNKNOWN)
UNKNOWN GAP: quantitative regime-vs-architecture partition; all §11 UNKNOWNs

## Execution log and limits

Every command executed in Phase 0 (all read-only; no installs, no config
changes, no training; every `python -c` ran fully in memory and wrote no
file — verified: `git status` shows only `M .gitignore` (pre-existing) and
untracked `phase2/` (pre-existing) plus the new `phase0/`):

1. `Get-ChildItem data / results/training / reference/onnx /
   phase2/results/training` — OK: data present (incl. 1.68 GB SceneFlow
   zip), convergence_run.pth present, ONNX present, Phase 2 arms present.
2. `git log --oneline -8; git status --short; scene counts;
   onnx summary keys` — OK: HEAD 58e8a19; 200 `_10` images, 200
   `disp_occ_0` files; summary keys listed (per-key shapes call returned
   None — NOT VERIFIED beyond keys; irrelevant, shapes verified from code).
3. `cat EXP-007/config.json; variants-vs-metrics equality;
   Test-Path phase0; phase2/docs listing` — OK: EXP-007 config 20 scenes
   (subset protocol confirmed); `results/reproduction/variants.json`
   content-identical to EXP-005 block (single reference record);
   `phase0/` absent before creation (created only `phase0/docs`).
4. `python` version/imports (`onnxruntime 1.27.0, torch 2.7.0+cu128, cv2`)
   — OK. `torch.cuda` — RTX 4060 Laptop, available.
5. ONNX 40-scene pooled repro, attempt 1 — FAILED on a throwaway-snippet
   masking bug (boolean-index size mismatch); inference itself ran (40
   scenes, 31.1 s). No file touched. Recorded honestly; snippet discarded.
6. Same, corrected masking — OK: EPE 1.3134470770188373, D1
   8.154366378221082, 3,802,797 px — exact match to EXP-005 official block.
7. ONNX per-image rates both scales — OK: 8.223686464356268 (hailo_exact)
   and 8.168465183586012 (kitti-scale) — exact matches to EXP-005.
8. `convergence_run.pth` inspection — OK: `{model: OrderedDict(72),
   config: EXP-016 recipe}`; config matches `EXP-016/config.json`.
9. PyTorch 40-scene frozen-contract eval (CUDA, 12.1 s) — OK:
   EPE 15.395826671257934, D1 88.39228073441733, 3,802,797 px.
10. `load_onnx_weights` live — OK: 36 convolutions, 423,586 params,
    Siamese-shared True.
11. Weight-loaded PyTorch vs ONNX, 1 scene — OK: max abs diff
    0.000823974609375 px (exactly the recorded EXP-011 figure).

Questions that could not be answered (with reason): HEF behavior/accuracy
(no device); upstream exact recipe for this export (no training log held);
quantitative regime-vs-architecture partition (requires new matched runs —
forbidden in Phase 0); per-checkpoint loadability beyond the two loaded
here (unnecessary — nothing hinges on them); `.audit_pytest/` listing
(permission denied); `data/` checksum completeness (functionally verified
instead: counts + full 40-scene evals ran clean); SceneFlow-zip contents
(unopened; Phase 1's problem).
Files that could not be inspected: binaries (`*.hef`, profiler HTML beyond
prior records); the bulk of `upstream/extracted/` beyond the cited
preprocessing/cost-volume/training files; image/GT bytes beyond loader-code
semantics (pixels were inferenced, never hand-inspected).
No scratch artefact was written under `phase0/` — only the three required files.
