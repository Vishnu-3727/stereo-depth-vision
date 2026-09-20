# C1 §10 — NARROWED EXPORT-VARIANT INVESTIGATION (ACTIONABILITY GATE)

Decisive question: **can an export-level change alone — no weight touched — bring ARM-P
ONNX-vs-PyTorch pooled max_abs_diff below the frozen 1e-3 px C1 criterion?**

Verdict: **C1 EXPORT-LEVEL MITIGATION NOT IDENTIFIED within the tested, evidence-motivated
export space.** All five export variants produce **bit-identical outputs** (cross-variant full-output
max abs diff = 0.0 on every scene) and the identical parity number 1.708984375e-3 px (FAIL).
Export representation therefore cannot move this number within the tested space.

Every substantive claim below is labelled
**VERIFIED** / **INFERRED** / **UNKNOWN** / **NOT APPLICABLE** / **NOT IDENTIFIABLE**.
Diagnosis only: no training, no weight/architecture/tolerance change was performed or proposed.

---

## 1. Scope

**VERIFIED.** This task tested exactly one thing first (Phase F): whether re-exporting the same
checkpoint with export-only changes (opset version, Resize lowering version, constant folding,
their combination) changes the frozen parity number. Secondary diagnostics (A–E) were run after
Phase F as far as technically possible, then stopped. Artefacts created, all new and only under
`stage_c_deploy/diagnostics/c1_export_variants/`:

- `run_phase_f.py`, `run_secondary.py` (drivers; export calls mirror `export_armp.py:76-78`)
- `c1_variant_{BASE,V1,V2,V3,V4}_*.onnx` (5 variant exports) + per-variant `*.json`
- `c1_phase_f_summary.json`, `c1_phase_f.log`
- `c1_instrumented_debug.onnx` (debug copy of the frozen graph + 5 explicit-shape probe
  outputs; never a deployment artifact), `c1_secondary.json`, `c1_secondary.log`
- this report + `c1_section10_export_investigation.json`

## 2. Previously settled findings explicitly NOT repeated

**VERIFIED** (cited, not re-proven; see `stage_c_deploy/diagnostics/C1_PARITY_DIAGNOSTIC_REPORT.md`):
intermediate-tensor localisation done; discrepancy deterministic (bit-identical self-runs);
~1e-7–1e-6 relative fp32 divergence from the first feature convs; amplification at
standardise→softmax→soft-argmin; 194 of 2,266,880 pixels responsible; near-tie gap 0.161 vs
0.337 background; flat-texture/high-gradient regions. Already REFUTED as the source:
ORT fusion, nondeterminism, occlusion, high-texture artifacts, export/weight-serialization
corruption, and dynamic range as the explanation (fp32 relative precision is scale-invariant).
That report states NO EXPORT BUG HAS BEEN ESTABLISHED. Nothing in §2 was re-run as a discovery;
numbers below that coincide with it are independent re-measurements, flagged as such.

## 3. Frozen contract

**VERIFIED.** Checkpoint `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth` sha256
`b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454` — re-verified before,
during (script start/end assertions), and after this task; unchanged. Frozen ONNX
`stage_c_deploy/armp_stereonet.onnx` sha256
`4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989` — re-verified identically;
never overwritten (no write to that path occurred; see §11). Contract: 397,954 params,
70 tensors, 24 candidates, 3 downsample levels, shift=right, regression_normalize=true.
Tolerance exactly **1e-3 px**, pooled max_abs_diff. Five scenes exactly:
000160_10, 000161_10, 000162_10, 000163_10, 000164_10 — asserted in-code (`scene order
changed -> STOP`); no substitutions. BASE export config mirrors `export_armp.py:76-78` exactly:
`torch.onnx.export(model, (dummy_l, dummy_r), path, input_names=[left,right],
output_names=[disparity], opset_version=13, do_constant_folding=True, dynamo=False)`.

## 4. Existing C1 baseline

**VERIFIED** (reproduced exactly here — BASE pooled max 0.001708984375 px, per-scene values
identical to §2 to the last digit; FAIL). Frozen baseline: pooled max_abs_diff 0.001708984375 px,
pass=false; per-scene max [8.3447e-04, 1.4668e-03, 8.9645e-04, 5.7602e-04, 1.7090e-03];
mean 7.141488885590075e-06; 194 discrepant pixels; worst scene 000164_10. P2A control ≈6.1798e-4
px PASS (cited from §2, not re-run).

## 5. Δinitial/Δresidual decomposition

**VERIFIED** (new measurement, `run_secondary.py` → `c1_secondary.json`). The known blocker
(Resize defeats `onnx.shape_inference`, so `/regression/ReduceSum_output_0` has uninferable shape)
was resolved with **explicit known shapes**: a debug copy of the frozen graph exposes
`/regression/Resize_output_0`, `/regression/Div_1_output_0`, `/regression/Softmax_output_0`,
`/regression/ReduceSum_output_0` (= disparity_initial), `/refinement/output_conv/Conv_output_0`
(= residual) with declared shapes; all shape cross-checks against PyTorch passed on all scenes.
Decomposition is therefore IDENTIFIABLE (the prior UNKNOWN is resolved for these probes):

| quantity | pooled max | per-scene max (000160…000164) |
|---|---|---|
| Δinitial (ReduceSum out) | 7.143020629882812e-04 px | 7.14e-04 / 4.95e-04 / 1.78e-04 / 2.57e-04 / 3.12e-04 |
| Δresidual (refine out_conv) | 1.689910888671875e-03 px | 7.88e-04 / 1.414e-03 / 8.62e-04 / 6.45e-04 / 1.690e-03 |
| Δfinal, instrumented graph | 1.708984375e-03 px | = frozen-graph values exactly |
| Δfinal, frozen graph | 1.708984375e-03 px | 8.3447e-04 / 1.4668e-03 / 8.9645e-04 / 5.7602e-04 / 1.7090e-03 |

**VERIFIED.** The 5-output instrumentation did **not** perturb the parity number (instrumented
Δfinal == frozen Δfinal exactly, pooled and — checked in JSON — consistent per scene).
**VERIFIED.** Δinitial alone (7.14e-4) would PASS; the FAIL is carried by the residual term
(Δresidual 1.690e-3 ≈ Δfinal 1.709e-3). **VERIFIED.** The worst readout-perturbation scene
(000160, Δinitial max) differs from the worst final-error scene (000164). Signed-error
correlation between init-err and resid-err over all pooled pixels is 0.015 (≈uncorrelated);
mean|Δinit| = 1.32e-6, mean|Δresid| = 6.76e-6. **INFERRED.** The residual error mixes (a) the
refinement net's response to a perturbed initial-disparity input with (b) its own kernel-level
divergence; the two are **NOT ISOLATED** — separating them needs a same-input intervention
(feed one backend's initial to both refinements), which is a model/inference change and therefore
OUT OF SCOPE — REQUIRES MODEL/INFERENCE CHANGE. The residual path is recorded here and stopped.

## 6. Operator attribution

**VERIFIED** probe diffs (pooled max abs / pooled relative = max|Δ|/max|value|), same 5 scenes,
PyTorch float32 reference reproducing `regression.py` exactly for the readout probes:

| operator | tensor | max \|Δ\| | mean \|Δ\| | relative | relation to final error | classification |
|---|---|---|---|---|---|---|
| Resize (bilinear, align_corners) | `/regression/Resize_output_0` | 7.42e12 (at value scale 2.52e18) | 2.71e10 | 2.95e-06 | same growth profile as upstream ~1e-6 readout-input divergence (§2); V2 rewired Resize-11↔13 with zero bit change (§10) | ASSOCIATED (carrier); NOT ISOLATED as cause; representation ruled out as lever (§10) |
| Standardise (Sub/Div chain) | `/regression/Div_1_output_0` | 3.374e-04 | 3.66e-07 | 7.30e-05 | measured input-side perturbation of the softmax | OBSERVED; ASSOCIATED, NOT ISOLATED (no intervention) |
| Softmax | `/regression/Softmax_output_0` | 4.286e-05 prob | 3.10e-08 | 5.09e-05 | measured prob perturbation feeding soft-argmin; near-tie gaps at failing pixels (§7) | OBSERVED; ASSOCIATED, NOT ISOLATED (no intervention) |
| ReduceSum (soft-argmin expectation) | `/regression/ReduceSum_output_0` | 7.143e-04 px | 1.32e-06 | 4.48e-05 | = Δinitial; alone below tolerance | OBSERVED; ASSOCIATED, NOT ISOLATED |
| Conv accumulation (feature→aggregation chain) | cited §2 §6 (not re-measured) | relative 3–5e-07 → ~1e-06 growth | — | ~1e-7–1e-6 | carrier of divergence from first feature convs; exonerated as bug by verified stage identities | ASSOCIATED (carrier); NOT A BUG (cited) |

No causality is claimed for any operator: no controlled per-operator intervention was performed
(any such intervention is a model/inference change — out of scope). Language is therefore
capped at OBSERVED / ASSOCIATED / NOT ISOLATED throughout.

## 7. Soft-argmin micro-analysis

**VERIFIED** at the actual failing pixels (authoritative frozen-graph final diff; 68 rows: all 28
discrepant pixels of 000161 + top-40 of 000164 by |Δ|; PT vs ONNX probabilities compared
pixel-wise): median top1−top2 gap 0.102 (PT) / 0.102 (ONNX), mean 0.136/0.136 — against the §2
background median 0.337. The near-tie mechanism is numerically visible at the failing pixels.
**VERIFIED.** Zero argmax flips in 68 rows (0.0): the winner index is stable; the error comes
from perturbed probability mass, not rank flips. **VERIFIED.** |Δinitial| at analysed rows:
median 3.17e-05, max 2.49e-04 — an order of magnitude below the final error at those pixels
(up to 1.7e-3), consistent with §5 (residual term dominates at failing pixels).
**INFERRED.** Readout converts a ~1e-4-scale standardised-cost perturbation into a ~1e-4-scale
initial-disparity perturbation at near-tie pixels (measured pair, §6); the step from there to the
1.7e-3 final error sits in the residual term whose input-sensitivity vs own-divergence split is
NOT ISOLATED (§5). The readout was not altered.

## 8. Border analysis

**VERIFIED** (all 194 discrepant pixels, per-edge distances; attribution only, no mask introduced
into evaluation): nearest-edge counts top 138 / right 56 / bottom 0 / left 0. Min-over-edges
distance: mean 23.63, median 28, max 90, p90 43 (reproduces the §2 reference values; the per-edge
split, previously UNKNOWN, is now identified: the mild concentration belongs to the top and
right borders). **INFERRED.** Mild top/right-border association; no mechanism asserted beyond
association (consistent with §2: flat-texture, high-gradient, non-occluded, high-disparity
regions — cited).

## 9. Version information

**VERIFIED** (recorded, `c1_secondary.json`): python 3.12.9, torch 2.7.0+cu128 (CPU path),
onnx 1.22.0, onnxruntime 1.27.0, numpy 2.5.1, Windows-11-10.0.26200-SP0, git HEAD
`58e8a19908ddbd35451652c61aef478f56b51ebd`. VERSION SENSITIVITY = NOT TESTED — no second
torch/onnx/onnxruntime build is installed and no environment change was performed per the brief;
that is the acceptable recorded answer.

## 10. Export variants tested

**VERIFIED.** Exactly the evidence-motivated matrix (4–6 allowed; 5 run; no brute force; no
further variant was evidence-motivated beyond V4):

- BASE: opset 13, folding on — reproduce frozen config (harness confirmation).
- V1: opset 17, folding on — alternative supported opset (exported cleanly, 712 nodes).
- V2: opset 11, folding on — Resize representation variant: opset 11 lowers `F.interpolate`
  to **Resize-11** (roi+scales as Constant inputs) instead of Resize-13 (sizes-form with empty
  roi/scales); 690 nodes. A genuine Resize wiring difference — **VERIFIED** in graph metadata.
- V3: opset 13, `do_constant_folding=False` — 976 nodes (366 Constant vs 313 folded).
- V4: opset 17 + folding off — most-different export-only combination, following V1–V3 evidence
  (none passed; stacking the two representation extremes was the remaining export-only stone).

Resize attributes identical in all variants (mode=linear, coordinate_transformation_mode=
align_corners, cubic_coeff_a=-0.75, nearest_mode=floor); only the input wiring (V2) and the
sizes-producer naming (unfolded Concat_1 in V3/V4) differ — **VERIFIED**.

| Variant | Opset | Resize | Fold | Nodes | Max Δ px | Mean Δ px | Failing pixels | PASS? |
|---|---|---|---|---|---|---|---|---|
| BASE | 13 | Resize-13, sizes-form | on | 712 | 1.708984375e-03 | 7.141488885590075e-06 | 194 | FAIL |
| V1 | 17 | Resize-13, sizes-form | on | 712 | 1.708984375e-03 | 7.141488885590075e-06 | 194 | FAIL |
| V2 | 11 | Resize-11, roi+scales Const | on | 690 | 1.708984375e-03 | 7.141488885590075e-06 | 194 | FAIL |
| V3 | 13 | Resize-13, sizes-form (unfolded) | off | 976 | 1.708984375e-03 | 7.141488885590075e-06 | 194 | FAIL |
| V4 | 17 | Resize-13, sizes-form (unfolded) | off | 976 | 1.708984375e-03 | 7.141488885590075e-06 | 194 | FAIL |

Per-scene max values are identical across all variants to the last printed digit
([8.3447e-04, 1.4668e-03, 8.9645e-04, 5.7602e-04, 1.7090e-03]); per-scene means likewise;
worst scene 000164_10 in all. **PASS is pooled max < 1e-3 px only** — no mean/EPE/D1/visual
substitution was used anywhere.

## 11. Variant integrity checks

**VERIFIED.** onnx.checker passes for all 5 variants (checked at export time; recorded per-variant).
Output contract identical for all: single output `disparity`, type tensor(float), shape
[1,1,368,1232] (via ORT session metadata). Exact export calls recorded per-variant JSON.
Variant file hashes (sha256):

- BASE `4277090deed1cde26ddb2a470d8dc097b26d5ad45a931e0aabf3635dbbcb6989` — **bit-identical
  to the frozen ONNX** (independent re-export reproduces the frozen bytes exactly; harness confirmed).
- V1 `2663af2425150d90894de270eb18a6ac6d0eb5e11b129622fcdd4572c3e9b829`
- V2 `ee1e49cb023e7f976d1bd6a5d9e0e0eeadb92f6a99dc351929662f4674b67abb`
- V3 `4c84587ec3d536e1bdbce1a56fe1412b3c80d5c6a9d1b0436720b3210a51f81b`
- V4 `496f5c261c8e01d17b0c8f570d40accb2179d8e3143a61f8baff8496a51c3aaa`

## 12. Per-variant parity results

**VERIFIED.** See §10 table (frozen parity evaluator: same 5 scenes, same `normalize()`
preprocessing, PyTorch CPU eval/no_grad vs ORT CPU, float64 diffs). Every variant: pooled
max_abs_diff = 0.001708984375 px, mean 7.141488885590075e-06, 194 discrepant pixels, worst
scene 000164_10, verdict FAIL. Beyond summary statistics: each variant's **full disparity
outputs are bit-identical to BASE on all 5 scenes** (cross-variant max abs diff = 0.0 —
`c1_secondary.json` §F). The parity number is therefore invariant to opset (11/13/17),
constant folding (on/off, 690/712/976 nodes), and Resize-11-vs-13 wiring within this space.

## 13. Independent reproduction of any passing variant

NOT APPLICABLE — no variant passed, so no reproduction was triggered and no candidate
replacement artifact exists. The authoritative ONNX was not replaced (it was never written).

## 14. Final actionability result

**C1 EXPORT-LEVEL MITIGATION NOT IDENTIFIED within the tested, evidence-motivated export space**
(opset {11, 13, 17} × folding {on, off} + Resize-11/13 representation change + most-different
combo; 5 variants; every failed variant preserved with hash). Connection to the established
mechanism (**INFERRED**, consistent with §2 + §§5–7): deterministic fp32 accumulation-order
divergence from the early convs (relative ~1e-7–1e-6, now traced through Resize at 2.95e-6
relative) → standardise→softmax→soft-argmin readout converts it to a ≤7.1e-4 px initial-disparity
perturbation at near-tie pixels (median gap 0.102, no rank flips) → the residual term carries it
to 1.709e-3 px final error. Export representation changes none of this: ORT executes bit-identical
numerics for all tested lowerings. Precisely what was tested: §10 matrix, nothing else.
**Not claimed**: that no possible export representation (different exporter, e.g. dynamo; different
ORT version; external converters) could ever pass — those are UNKNOWN and untested, and the
bit-identity result, while discouraging for that hope, does not disprove it.

## 15. What remains UNKNOWN

- The exact kernel/accumulation-order source of the fp32 divergence (unchanged from §2).
- Whether a same-input residual intervention would attribute the residual error to
  input-sensitivity or own-kernel divergence (NOT ISOLATED, §5).
- Whether untested export representations (dynamo exporter, other ORT/torch builds, external
  converters) change the number (UNKNOWN; §10 bit-identity covers only the tested space).
- Why ARM-P training produced ~1e13-scale feature activations (cited §2, outside scope).
- Whether the activation range survives int8 quantisation (cited §2 §10-quantisation-risk, outside scope).

## 16. What remains BLOCKED

- Residual input-sensitivity vs own-divergence split: BLOCKED — needs a same-input
  intervention = model/inference change (OUT OF SCOPE — REQUIRES MODEL/INFERENCE CHANGE, §5).
- Version sensitivity: BLOCKED-as-DEFINED — recorded NOT TESTED per brief (acceptable answer, §9).
- Any mitigation path touching weights, readout, candidates, units, tolerance, scenes,
  preprocessing, quantisation, or Hailo: forbidden here; each is OUT OF SCOPE.

## 17. Recommendation for C1 disposition

The FAIL cannot be cleared at the export level within the tested space — the number is
bit-invariant to export representation, so further export-only attempts (more opsets, more
folding flags) have no evidence behind them and are not recommended. The decision now belongs
to the owners of the three frozen quantities this task was forbidden to touch: (a) treat the
FAIL as blocking for deployment of these weights and keep downstream work gated behind a passing
re-measurement; (b) commission weight-side work (retraining/range control) as a new task with its
own hypothesis record; or (c) revisit, in writing and before choosing any new number, whether a
bare max-abs criterion is the right deployment-parity gate — leaving this FAIL in the record
either way. No recommendation is made between these; no threshold, weight, or source change is
proposed here.

---

## Diagnostic table

| Diagnostic | Result | Classification |
|---|---|---|
| Δinitial decomposition (ReduceSum probe) | max 7.143e-04 px; alone below tolerance | VERIFIED |
| Δresidual decomposition (refine out_conv probe) | max 1.690e-03 px; carries the FAIL | VERIFIED |
| Δfinal, instrumented vs frozen graph | identical (1.708984375e-03) | VERIFIED |
| Resize contribution | output relative divergence 2.95e-06; rewiring (V2) changes zero bits | OBSERVED / ASSOCIATED / NOT ISOLATED |
| Softmax contribution | prob perturbation ≤4.286e-05; near-tie gaps at failing pixels | OBSERVED / ASSOCIATED / NOT ISOLATED |
| ReduceSum contribution | = Δinitial 7.143e-04 px | OBSERVED / ASSOCIATED / NOT ISOLATED |
| Conv accumulation contribution | cited §2 profile (not re-measured) | ASSOCIATED; NOT A BUG (cited) |
| Near-tie perturbation visibility | median gap 0.102 at failing pixels; 0/68 rank flips | VERIFIED numbers; mechanism INFERRED |
| Border association | nearest edge top 138 / right 56 / bottom 0 / left 0 of 194 | VERIFIED (association only) |
| Version sensitivity | recorded §9; no second build tested | NOT TESTED (acceptable per brief) |
| Export-level mitigation | 5/5 variants FAIL with bit-identical outputs | VERIFIED → NOT IDENTIFIED in tested space |

## Integrity checklist

- [x] checkpoint sha unchanged (`b2f6f5d5…fffeb7454`, verified start/end/post-task) — VERIFIED
- [x] frozen ONNX sha unchanged (`4277090d…6989`, verified start/end/post-task; never written) — VERIFIED
- [x] model source unchanged (`src/**` untouched; `git status` shows only the pre-existing
  modification set recorded before the C1 run) — VERIFIED
- [x] preprocessing unchanged (same `normalize()`, same dataset class/indices) — VERIFIED
- [x] evaluator unchanged (`export_armp.py`, `c1_parity_diagnostic.py` untouched; parity block mirrored) — VERIFIED
- [x] tolerance still 1e-3 (asserted constant in both drivers; PASS computed as pooled max < 1e-3 only) — VERIFIED
- [x] five scenes unchanged (order asserted in-code in both drivers) — VERIFIED
- [x] no training (no optimizer, no backward, no weight write in either driver) — VERIFIED
- [x] no quantization (no quantize calls, no int8 artifacts) — VERIFIED
- [x] no Hailo work (no SDK, no HEF, no device measurement) — VERIFIED
- [x] all variant hashes recorded (§11) — VERIFIED
- [x] all failed variants preserved (5/5 `.onnx` + `.json` on disk; no export failed) — VERIFIED
- [x] passing-variant reproduction — NOT APPLICABLE (none passed)
- [x] no commits made — VERIFIED

No frozen artifact changed unexpectedly during this task; per the brief, nothing was repaired
because nothing was discrepant.
