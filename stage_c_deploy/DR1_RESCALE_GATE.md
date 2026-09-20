# STAGE C DR-1 — ACTIVATION RESCALE GATE (Variant A only)

Inference-only gate. No training, no fine-tuning, no optimizer, no backward
pass was performed. No model, architecture, loss, augmentation, candidate
count, normalization setting, checkpoint, tolerance, or device assumption was
changed. All compute is inference-only, `torch.no_grad()`.

Label convention: every substantive conclusion carries exactly one label —
**VERIFIED**, **INFERRED**, or **UNKNOWN**.

C1 is FAIL and remains FAIL. The 1e-3 number used in §9 is the **DR-1 RESCALE
EQUIVALENCE THRESHOLD**. It is NOT the C1 parity threshold, despite having the
same numerical value. Nothing in this experiment modifies, reinterprets,
re-runs, or repairs C1.

---

## 1 Status

**VERIFIED.** DR-1 executed as authorized: ONE experiment, Variant A
(activation rescale), inference-only, on the frozen ARM-P seed-1 checkpoint.
Verdict: **H1 FAIL** — rejection criterion A fired (five-scene pooled
max |delta disparity| = 1.4816284e-02 px > 1e-3 px). Rescale branch STOPPED.
Variant B was not performed. No training was performed. No Hailo work was
performed (target device UNKNOWN).

---

## 2 Objective

**VERIFIED.** Answer exactly one question: can the frozen ARM-P aggregation
output be rescaled downward by approximately 1e17 without materially changing
the stereo solution? Formally
`aggregated_cost_rescaled = aggregated_cost / s` with `s = 1e17`, applied only
in the inference computation between `aggregated_cost` and regression /
soft-argmin, with all downstream computation unchanged.

---

## 3 Frozen inputs

**VERIFIED.**

- ARM-P seed 1: `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth`,
  sha256 `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454`
  (asserted at start of BOTH scripts; re-verified after the five-scene run;
  mismatch would have stopped the run).
- Frozen configuration: params = 397,954 (asserted); tensors = 70 (asserted);
  disparity candidates = 24; downsample levels = 3; cost_volume_shift = right;
  regression_normalize = true (asserted). None changed.
- Frozen KITTI accuracy EPE = 1.1912168 px (reproduced exactly by the new
  evaluator before any rescaled measurement was trusted: 1.191216765057325).
- Historical Hailo reference EPE = 1.3134471 px (descriptive only; no
  deployment-superiority claim is made).
- C1 PyTorch->ONNX parity is FAIL (tolerance max |delta| < 1e-3 px) and
  remains FAIL regardless of DR-1 outcome. C1 was not touched.
- Target Hailo device UNKNOWN. No device assumed.

---

## 4 Exact intervention

**VERIFIED.** Variant A, inference-only, exact scalar `s = 1e17`
(`1e+17` in the JSON provenance):

- original path: `aggregated_cost -> regression -> refinement -> add -> ReLU`
- DR-1 path: `aggregated_cost / 1e17 -> regression -> refinement -> add -> ReLU`

Both paths run through the model's OWN `regression` and `refinement` modules
(`net.regression(agg, size)`, `net.refinement(disp_init, left)`, plus the
config `final_relu`), so downstream computation is identical by construction —
no replicated math. The rescale point is exactly one tensor:
`forward return_stages['aggregated_cost']`, shape `(1, 24, 46, 154)`, dtype
`torch.float32`. The aggregated cost is captured ONCE per scene; both paths
run from that same captured tensor (identical inputs by construction).
Self-check: module path A vs `forward()` disparity_final is 0.0 on all five
scenes (CPU gate) and 0.0 on the first five scenes (CUDA eval) — the
machinery adds no difference of its own.

---

## 5 Variant classification

**VERIFIED.** This is **Variant A (activation rescale)** only: the checkpoint
is byte-identical before and after (sha256 asserted equal); the in-memory
state dict is verified element-wise equal before and after; NO replacement
checkpoint was created. **Variant B (weight rescale)** was NOT performed —
no `to_cost` weight or bias was divided, and no new checkpoint artifact
exists. The mathematical equivalence of A and B at the aggregated-cost output
(linear `to_cost` Conv3d + channel squeeze only) is noted from DR-0 and was
not re-tested here.

---

## 6 Five-scene results

**VERIFIED.** Same five scenes as DR-0/C1, asserted exactly
(`000160_10.png … 000164_10.png`, KITTI2015 `hailo_val` indices 0–4), same
normalized inputs, CPU, `torch.no_grad()`. Per-scene
max |disparity_rescaled − disparity_original|:

| scene | max \|delta\| (px) | agg max orig | agg max rescaled | postnorm max orig → resc | disp_init max orig → resc | residual max orig → resc | final max orig → resc |
|---|---|---|---|---|---|---|---|
| 000160_10 | 3.2157898e-03 | 2.1330894e18 | 21.3309 | 4.5534 → 4.5534 | 14.9006 → 14.9005 | 70.6188 → 70.6189 | 79.9992 → 79.9992 |
| 000161_10 | 6.6585541e-03 | 1.5717139e18 | 15.7171 | 4.5940 → 4.5940 | 14.2959 → 14.2954 | 114.9820 → 114.9802 | 126.1495 → 126.1476 |
| 000162_10 | 5.5923462e-03 | 2.5326005e18 | 25.3260 | 4.6218 → 4.6218 | 13.1346 → 13.1345 | 97.5125 → 97.5125 | 105.2280 → 105.2280 |
| 000163_10 | 3.7631989e-03 | 2.1125215e18 | 21.1252 | 4.6062 → 4.6062 | 11.0097 → 11.0097 | 63.9815 → 63.9816 | 72.6361 → 72.6362 |
| 000164_10 | 1.4816284e-02 | 2.0187671e18 | 20.1877 | 4.6080 → 4.6080 | 15.9315 → 15.9313 | 106.4363 → 106.4357 | 116.3443 → 116.3436 |

Pooled max |delta disparity| = **1.4816284e-02 px** (scene 000164_10).
Non-finite counts: 0 in every tensor on every scene (delta, dA, dB).
Determinism: Path B re-run is bit-identical on all five scenes
(max rerun diff 0.0). Full per-scene record:
`stage_c_deploy/dr1_rescale/dr1_rescale_gate.json`.

Note the scale context **(VERIFIED)**: the rescale moves the pooled
aggregated-cost maximum from ~2.5e18 to ~25 — i.e. into the reference
deployment range (Hailo reference ~23.9 where the anchor exists) — while the
post-normalization maxima stay single-digit and identical to 4 decimals.

---

## 7 40-scene accuracy results

**VERIFIED.** Frozen evaluation contract, unmodified evaluator core
(`phase1.harness.frozen_eval.pooled_metrics` +
`refuse_unless_contract`): 40 `hailo_val` scenes; 3,802,797 valid pixels both
paths; gt_scale = 256.0; source = disp_occ_0; both contract guards pass
(`contract_match: true`). The new script
(`stage_c_deploy/dr1_rescale/dr1_eval_40scene.py`) FIRST demonstrated it
reproduces the frozen number on the unmodified model (original EPE
1.191216765057325 = frozen 1.1912168; `original_reproduces_frozen: true`) and
that its rescaled-path machinery reproduces `forward()` at s=1 bit-identically
(`s1_reproduces_forward: true`, maxdiff 0.0) — only then is the rescaled
measurement trusted. Device is CUDA, matching the frozen measurement
(`tier2_eval.json` metadata `device: cuda`).

| path | EPE (px) | D1 (%) | valid pixels |
|---|---|---|---|
| original ARM-P | 1.191216765057325 | 6.211033615520366 | 3,802,797 |
| rescaled ARM-P (÷1e17) | 1.191211971353279 | 6.211033615520366 | 3,802,797 |
| delta (rescaled − original) | −4.7937040e-06 | 0.0 | 0 |

EPE delta = **−4.79e-06 px** (rescaled trivially lower; far below any
regression). D1 is unchanged to all recorded digits. Full metric blocks
(EPE/RMSE/D1/BAD1-3) are in `stage_c_deploy/dr1_rescale/dr1_eval_40scene.json`.

---

## 8 Dynamic-range results

**VERIFIED** (from §6, CPU five-scene capture):

- Rescaled aggregated-cost maxima per scene: 15.72–25.33 (vs 1.57e18–2.53e18
  original) — single-to-double digits, the intended reference-order regime.
- Post-normalization maxima: unchanged to 4 decimals on every scene
  (4.5534/4.5940/4.6218/4.6062/4.6080), single-digit, matching the DR-0
  postnorm regime (DR-0 pooled max 4.62).
- disparity_initial maxima: changed only at the 4th significant digit
  (e.g. 14.9006 → 14.9005); refinement_residual and disparity_final maxima:
  unchanged to the recorded precision (finals 80.0/126.1/105.2/72.6/116.3).
- DR-0 probe extension at a = 1e-17 (equiv. s = 1e17): pooled
  max|disparity(a) − disparity(1)| = 1.4816284e-02 px — the same Variant-A
  comparison expressed in DR-0 probe units. DR-0's own measurements and files
  were not replaced or altered.

---

## 9 DR-1 equivalence test

**VERIFIED.** Against the predeclared **DR-1 RESCALE EQUIVALENCE THRESHOLD**
(1e-3 px — a DR-1 criterion, NOT the C1 gate): pooled five-scene
max |delta| = 1.4816284e-02 px is **~15× the threshold**, exceeded on ALL
FIVE scenes individually (smallest per-scene value 3.22e-03 px, scene
000160_10). This is ~240× larger than DR-0's worst-case invariance figure
(6.1e-05 px over a ∈ [1e-6, 1e6]): the invariance observed across 12 orders
of magnitude does NOT extend cleanly to a = 1e-17 in fp32 inference.

---

## 10 Rejection criteria

Predeclared DR-1 criteria (NOT the C1 gate; C1 unmodified, still FAIL):

- **A. Five-scene max |delta| > 1e-3 px → FIRED (VERIFIED).**
  Measured 1.4816284e-02 px pooled (per-scene 3.22e-03 … 1.48e-02 px).
- B. EPE regression ≥ 0.05 px → did NOT fire (VERIFIED).
  Measured rescaled − original = −4.7937040e-06 px.
- C. Postnorm/final range leaves the intended reference-order regime →
  did NOT fire (VERIFIED). Postnorm maxima identical to 4 decimals,
  single-digit (4.55–4.62); rescaled aggregated-cost maxima 15.7–25.3, i.e.
  AT the reference order (~23.9); disparity_final maxima unchanged
  (72.6–126.1 px both paths). Criterion C has no predeclared numeric
  deployment requirement; the comparison above is reported explicitly.

One fired criterion rejects H1. **Verdict: H1 FAIL.**

---

## 11 Integrity checks

Self-audit, each item verified and reported (no item failed):

- [x] frozen ARM-P checkpoint hash unchanged — VERIFIED (`b2f6f5d5…eb7454`
  before, after, and in both JSONs)
- [x] no model weights modified — VERIFIED (in-memory state dict
  element-wise equal before/after; no new checkpoint file)
- [x] no architecture modified — VERIFIED (no source file edited; only new
  files under `stage_c_deploy/dr1_rescale/`)
- [x] no loss modified — VERIFIED
- [x] no augmentation modified — VERIFIED
- [x] no candidate count modified — VERIFIED (24, asserted in config)
- [x] regression_normalize remains true — VERIFIED (asserted at load)
- [x] C1 tolerance unchanged — VERIFIED (never referenced as changeable)
- [x] C1 verdict remains FAIL — VERIFIED (stated; C1 not rerun/reinterpreted)
- [x] five-scene contract exact — VERIFIED (order assertion passed)
- [x] 40-scene contract exact — VERIFIED (both guards `contract_match: true`;
  40 / 3,802,797 / 256.0 / disp_occ_0)
- [x] no NaN/Inf — VERIFIED (all non-finite counts 0)
- [x] deterministic comparison — VERIFIED (bit-identical reruns, both scripts)
- [x] original vs rescaled comparison uses identical inputs — VERIFIED
  (single capture per scene feeds both paths)
- [x] DR-1 threshold explicitly named separately from C1 — VERIFIED (this
  report, §§9–10 and header)
- [x] no Hailo device assumed — VERIFIED (UNKNOWN throughout)
- [x] no training performed — VERIFIED (`torch.no_grad()` throughout; no
  optimizer/loss/backward in either script)
- [x] no silent patch — VERIFIED (new files only; DR-0/C1/evaluator/checkpoint
  untouched; pre-existing `git status` modifications predate this run per the
  C1 report and were not made here)
- [x] all deviations documented — VERIFIED (see §12 deviation note)

Since no integrity item failed, the FAIL verdict stands as measured.

---

## 12 Provenance

**VERIFIED.**

- New files only (nothing existing overwritten):
  `stage_c_deploy/dr1_rescale/dr1_rescale_gate.py`,
  `stage_c_deploy/dr1_rescale/dr1_rescale_gate.json`,
  `stage_c_deploy/dr1_rescale/dr1_eval_40scene.py`,
  `stage_c_deploy/dr1_rescale/dr1_eval_40scene.json`,
  `stage_c_deploy/DR1_RESCALE_GATE.md` (this file).
- UTC: five-scene gate 2026-09-20T02:16:27Z; 40-scene eval 2026-09-20T02:27:40Z
  (see JSONs). Git HEAD `58e8a19908ddbd35451652c61aef478f56b51ebd` (both runs).
- Exact commands: `python stage_c_deploy/dr1_rescale/dr1_rescale_gate.py`,
  then `python stage_c_deploy/dr1_rescale/dr1_eval_40scene.py`.
- Exact scalar s = 1e17; tensor
  `forward return_stages['aggregated_cost']`, shape `(1, 24, 46, 154)`,
  dtype `torch.float32`.
- Scene lists: five-scene `000160_10 … 000164_10`; 40-scene full `hailo_val`
  (contract guard passed). Preprocessing: `Kitti2015Stereo` +
  `normalize()` (ImageNet, NCHW), identical both paths.
- Versions: Python 3.12.9, PyTorch 2.7.0+cu128, NumPy 2.5.1 (both runs;
  recorded in JSONs). Host Windows-11-10.0.26200-SP0. Five-scene gate: CPU,
  1 thread (DR-0 conditions). 40-scene eval: CUDA (frozen-evaluator device).
- Deviation / error history (preserved, not deleted): the first script
  combined both gates but its 40-scene CPU loop exceeded the 600 s execution
  window after completing and writing the five-scene section; the 40-scene
  measurement was therefore re-run as a separate script on CUDA (the device
  the frozen EPE was measured on). Five-scene numbers come from the first
  run's written JSON and were not re-measured. No failed attempt was deleted;
  both scripts are shipped beside their outputs.

---

## 13 VERIFIED findings

1. Variant-A rescale (÷1e17) changes the frozen stereo solution by pooled
   max 1.48e-02 px over the five scenes — exceeding the 1e-3 px DR-1
   equivalence threshold on every scene. H1 is rejected by criterion A.
2. The same rescale leaves frozen 40-scene accuracy effectively unchanged:
   EPE delta −4.79e-06 px, D1 delta 0.0, valid pixels unchanged.
3. Post-normalization and final-output ranges remain in the intended
   reference-order regime (postnorm single-digit, identical to 4 decimals;
   rescaled aggregated-cost maxima 15.7–25.3).
4. The comparison itself is sound: bit-identical determinism, zero
   non-finite values, s=1 machinery reproduces `forward()` exactly, and the
   new evaluator reproduces the frozen EPE exactly before trusting the
   rescaled number.
5. C1 remains FAIL; the checkpoint is byte-identical; Variant B was not
   performed; no training occurred; no Hailo device was assumed.

---

## 14 INFERRED findings

1. The fp32 readout invariance measured by DR-0 over a ∈ [1e-6, 1e6]
   (≤6.1e-05 px) does not extend to a = 1e-17: the observed deltas are
   ~240× larger. The most plausible mechanism class is fp32 arithmetic
   noise amplified at the extreme rescale (the eps/a term in
   `z(a·c) = (c−mean)/(std+eps/a)` is still negligible at measured std, so
   the break is consistent with precision loss in the rescaled-cost
   arithmetic rather than with the eps-collapse mode) — but the mechanism
   was NOT isolated by this gate, so this remains INFERRED, not VERIFIED.
2. Per-pixel max deltas (~1e-2 px) and pooled accuracy (EPE delta ~5e-6 px)
   can disagree in direction and scale: worst-pixel equivalence and mean
   accuracy measure different things. No general claim beyond this
   measurement is made.

---

## 15 UNKNOWN findings

1. WHY the per-pixel deltas at a = 1e-17 exceed fp32-noise expectations —
   mechanism not isolated (candidates include rescaled-cost rounding and
   softmax-tail sensitivity, both untested).
2. Whether any intermediate scalar (between 1e6 and 1e17 in DR-0's probe
   units) would satisfy the equivalence threshold — untested; no other
   scalar may be tried under this authorization.
3. Whether Variant B would behave identically in fp32 bit-pattern (expected
   mathematically equivalent at the output, but NOT executed and therefore
   UNKNOWN as a measured fact).
4. Whether the rescale affects INT8 quantization, Hailo compilation, or any
   hardware behavior — UNKNOWN; no such test is authorized or performed.
5. Why training produced the huge activations — UNKNOWN; no training
   analysis authorized.

---

## 16 Verdict

**H1 FAIL (VERIFIED).** Rejection criterion A fired: five-scene
rescaled-vs-original max |delta disparity| = 1.4816284e-02 px, exceeding the
DR-1 RESCALE EQUIVALENCE THRESHOLD of 1e-3 px on all five scenes. Criteria B
and C did not fire (EPE delta −4.79e-06 px; ranges preserved). A DR-1 failure
means this audit did not establish that the activation scale is removable
under the tested intervention. No mechanism beyond that is inferred.

---

## 17 Stop/next-action condition

STOP this rescale branch. Do NOT train, fine-tune, change architecture, alter
normalization, alter loss, change augmentation, alter candidate count, modify
the C1 tolerance, try arbitrary scaling factors, or introduce new
architecture experiments. Do NOT create Variant B. Do NOT proceed to Hailo
compilation (target device still UNKNOWN). C1 remains FAIL. The frozen ARM-P
checkpoint `b2f6f5d5…eb7454` is preserved byte-identical with its DR-1 FAIL
record beside it (`stage_c_deploy/dr1_rescale/` + this report).
