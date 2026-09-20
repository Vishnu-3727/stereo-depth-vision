> ---
> ## ⚠ SUPERSESSION NOTICE — added 2026-09-20 by the documentation closure pass
>
> **THIS DOCUMENT IS HISTORICAL. IT IS NOT THE CURRENT DEPLOYMENT STATUS.**
>
> 1. **Historical.** This report closed the re-scoped Phase 2 on 2026-09-17. Its content below
>    is preserved unchanged as the historical record of that phase.
> 2. **Its deployment wording is historical.** The title phrase "Model Deployed", the §1 row
>    "Deployed model — P2A seed 0", and the §10 line "Deployed model   P2A seed 0" describe the
>    *Phase-2 deployment selection* as it stood on 2026-09-17. In this project "deployed" there
>    meant "selected and exported"; the same report already caps itself at
>    "Furthest verified deployment layer: **ONNX**".
> 3. **It is NOT the current deployment status.** Nothing has been deployed to any accelerator.
> 4. **Current deployment candidate: ARM-P seed 1**, selected under the predeclared C0 rule —
>    `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth`
>    (sha256 `b2f6f5d558dbd2676d740ca55a0b60900c89ff2225f0445e98c59d6fffeb7454`).
>    See `stage_c_deploy/C0_DECISION_RECORD.md`. P2A seed 0 remains a **historical**
>    deployment selection and a valid experimental baseline.
> 5. **Hailo deployment remains BLOCKED.** The target Hailo device is **UNSPECIFIED** (no
>    authoritative company requirement exists anywhere in the repository) and the target
>    toolchain has therefore **NOT been executed**. No ARM-P HEF exists. See
>    `stage_c_deploy/STAGE_C_DEPLOYMENT_READINESS_FINAL.md`.
>
> Current entry points: `README.md` → "Current project status", and `RESULTS_INDEX.md`.
> ---

# Phase 2 Final Report — Optimization Closed, Model Deployed

**Microchip Stereo Depth Vision.** Closeout for the final phase: the Phase-1
bottleneck audit, the Phase-2 diagnostic that found the cause, the one
preregistered optimization experiment, the deployment decision, the export, and
the Hailo validation attempt.

Written 2026-09-17. Everything below is traceable to a record under `phase1/`
or `phase2/`. Where something could not be established it says so.

> **A note on phase numbering, because the repository carries two schemes.**
> The original Phase 1 was the forensic reconstruction of the Hailo reference
> (`PHASE_1_FINAL_REPORT.md`, `README.md`, tag `phase-1-frozen`). The project
> was later re-scoped to exactly three phases — **0** baseline lock, **1** model
> development, **2** final optimization and deployment. This document closes
> Phase 2 under the *re-scoped* numbering. The older `phase2/` subtree
> (H1/H2/E-series correspondence campaign) belongs to a superseded Phase-2 line
> and was left untouched throughout. There is no Phase 3.

---

## 1. Where the project ended

| | |
|---|---|
| **Deployed model** | **P2A seed 0** — `phase2/runs/p2a_scale_coverage/p2a_best.pth` |
| Deployed accuracy (this artifact, PyTorch) | **1.4149796 px** EPE, 8.4195 % D1 |
| Deployed accuracy (this artifact, exported ONNX, scored end to end) | **1.4150748 px** EPE, 8.4200 % D1 |
| Method accuracy (3-seed mean — the honest expectation on a retrain) | **1.4409826 px** |
| **Formal experimental incumbent** | **ARM-V** — 1.7727436 px (3-seed mean) |
| **P2A experiment verdict** | **INCONCLUSIVE** |
| **P2A mechanism verdict** | **NOT CONFIRMED** |
| Frozen reference | 1.3134471 px |
| Original PyTorch baseline (start of Phase 1) | 15.3958267 px |
| Parameters | 397,954 |
| Furthest verified deployment layer | **ONNX** |

The single most important structural fact about this closeout: **the deployment
decision and the experimental verdict are recorded separately and neither was
bent to fit the other.** P2A is deployed; P2A is not marked confirmed.

---

## 2. Phase 1 closure — the bottleneck audit

`phase1/docs/FINAL_PHASE1_BOTTLENECK_AUDIT.md`.

Phase 1 ended with a decision **not** to run another experiment. The audit
measured where ARM-V's remaining error lived and found it was not distributed:

- **85.5 % of the 0.4593 px gap to the reference sits in the 4.61 % of pixels
  with GT ≥ 64 px** (per-seed 83.9 / 94.6 / 83.2 %). On the other 95.4 % of
  pixels ARM-V was already at parity with the reference.
- The symptom was an **output ceiling**: maximum prediction 91.5 / 114.0 /
  101.7 px against a 153.0 px GT maximum. Signed error within ±0.2 px up to
  64 px, then collapsing to −6.6 / −38.8 / −53.4 px.

Three plausible causes were **refuted by measurement**, not by argument:

| candidate | refuted by |
|---|---|
| disparity quantisation | EPE flat vs distance-to-candidate (1.93 → 1.85); fractional part uniform to ±7 % |
| disparity range | max GT 153 px < 184 px representable; zero out-of-range pixels |
| model capacity | train-split EPE 1.03–1.14 px — *below* the 1.3134 px reference |

The audit also established why no fourth Phase-1 arm was worth running:
ARM-V's 3-seed spread (0.3410 px) was 74 % of the entire remaining gap, and
ARM-X's real 0.1493 px improvement had already failed confirmation under that
same rule.

---

## 3. Phase 2 diagnostic — proving the cause without training

`phase2/docs/PHASE2_ARMV_HIGH_DISPARITY_DIAGNOSTIC.md`,
`phase2/diagnostics/arm_v_high_disparity/` (five probes, inference only).

**The decisive measurement was test-time scale transfer.** Same evaluation
pixels, same frozen weights, stereo pair downscaled by 0.5 — halving every
disparity at the cost of half the spatial resolution:

| seed | EPE GT≥96 | signed bias | slope(pred~GT) | max prediction |
|---|---|---|---|---|
| 0 | 39.29 → **10.88** | −39.29 → **−4.24** | +0.295 → **+1.010** | 91.5 → **152.7** |
| 1 | 21.71 → **8.48** | −21.70 → **−3.69** | +0.239 → **+1.253** | 113.9 → **144.3** |
| 2 | 38.94 → **10.85** | −38.94 → **−4.19** | +0.075 → **+0.775** | 101.7 → **149.0** |

Throwing away half the resolution cannot improve stereo matching. It can only
change which disparity values are presented. **The failure is a function of the
disparity value, not of those pixels.**

Four of the six candidate mechanism classes were eliminated by direct
measurement — refinement suppression, readout calibration, cost-distribution
saturation, and high-disparity matching failure. A census over **all 16 frozen
checkpoints** (ARM-U/V/W/X/Y/Z × 3 seeds) found the collapse in **16 of 16**,
while the frozen reference — *same graph, no-op disparity shift, no search at
all* — does not collapse. The graph was therefore not the cause. One class
survived: **the disparity band covered by training supervision** (GT ≥ 96 px is
0.228 % of training pixels against 0.898 % of evaluation pixels).

An honest negative result from the same work: EPE over the high stratum is
*relatively noisier* than global EPE in 4 of 5 arms. Stratum metrics were
necessary but were not a lower-noise instrument, and were not presented as one.

---

## 4. The one experiment — `EXP-P2A-SCALE-COVERAGE-001`

Preregistration `phase2/docs/PHASE2_HYPOTHESIS_01.md`, result
`phase2/docs/PHASE2_P2A_SCALE_COVERAGE_RESULT.md`, record
`phase2/runs/p2a_scale_coverage/P2A_EXPERIMENT_RECORD.json`.

**Intervention — one variable, training-time only, zero parameter change:**

```
s  ~ logUniform(0.7, 1.7)                 one draw per sample
w  = round(512 / s) ;  h = round(256 / s)
left/right : BILINEAR resize of the (h,w) crop -> 256x512
disparity  : NEAREST  resize of the (h,w) crop -> 256x512   (sparse GT)
sx = 512 / w                              REALISED horizontal factor
disparity[disparity > 0] *= sx
```

The scale range was fixed by a measured coverage rule, not tuned. Two earlier
versions of that rule were ill-posed and are preserved beside their refutation
in `coverage_choice.py` rather than quietly replaced.

### Both preregistered gates were missed, each by a hair

| gate | threshold | measured | verdict |
|---|---|---|---|
| Accuracy | mean EPE < 1.4317132 | **1.4409826** | **INCONCLUSIVE** (short by 0.0093 px) |
| Mechanism | mean slope_hi ≥ 0.60 **and** every seed ≥ 0.45 | mean **+0.5701**, min **+0.5374** | **NOT CONFIRMED** (mean short by 0.0299; per-seed condition passed) |

### What was nonetheless measured

| | ARM-V | P2A |
|---|---|---|
| global EPE (3-seed mean) | 1.7727436 | **1.4409826** (−0.3318) |
| seed spread | 0.3410304 | **0.0444151** (7.7× tighter) |
| D1 | 10.27 % | **8.59 %** |
| GT ≥ 64 EPE | 11.76 | **5.15** |
| GT ≥ 96 EPE | 33.31 | **12.42** |
| slope_hi | +0.203 | **+0.570** |
| max prediction | 91.5 / 114.0 / 101.7 | **130.6 / 142.9 / 135.4** |
| GT < 64 EPE (protection metric) | 1.2897 | **1.2617** — improved |
| gap to reference | +0.4593 | **+0.1275** |

Every P2A seed beat every ARM-V seed. The named risk — that widening the
supervision band would cost the 95 % of pixels already at parity — did not
materialise.

Integrity: 397,954 parameters, 70 state_dict keys identical to ARM-V, no
BatchNorm, `contract_match true`, 3,802,797 valid pixels on all three seeds.
No tuning, no stacking, no seed selection, no threshold touched.

**Execution note, recorded because it happened.** Two agent-harness background
shells were killed by the host for low system memory. Neither kill touched a
training process — the detached python children survived both times and seed 1
trained through uninterrupted, verified by watching the log advance before
anything was done. Seed 2 and the evaluation were then run from a launcher
detached from the harness. No seed was restarted, dropped, replaced or re-run;
no result was read before all three seeds completed.

---

## 5. The deployment decision

`phase2/docs/PHASE2_DEPLOYMENT_SELECTION.md`.

The preregistered gate is an *experimental acceptance criterion*. Using it as a
*deployment rule* would have shipped a model with 23 % higher mean error on the
strength of a 0.0093 px threshold miss. The gate was also defined as
*ARM-V mean − ARM-V spread*, and ARM-V's spread was itself inflated by the
instability this experiment removed — so reducing that instability made the
gate harder to clear. That is a property of the rule's construction; it is
recorded, and **it did not change the verdict.**

The project owner decided: **deploy P2A, keep INCONCLUSIVE on the record.**
Deployed checkpoint chosen by a stated rule — lowest frozen-contract EPE, which
is seed 0. Attached reporting rule: 1.4149796 px is this artifact's accuracy;
1.4409826 px is the method's. Both are quoted together everywhere.

Not claimed anywhere: that scale augmentation confirmed the mechanism; that
scale coverage is the whole story; that the residual [128,144) px failure
(31–35 px EPE) shares that cause.

---

## 6. Export and host validation

`phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md`,
`phase2/deploy/P2A_DEPLOYMENT_VALIDATION.json`,
`phase2/deploy/P2A_STATIC_EXPORT_VALIDATION.json`.

Two ONNX artifacts, same weights, both validated:

| | original | static |
|---|---|---|
| file | `p2a_stereonet.onnx` | `p2a_stereonet_static.onnx` |
| nodes | 712 | 689 |
| `onnx.checker` | PASS | PASS |
| frozen-contract EPE (scored end to end) | 1.4150748 | 1.4150748 |
| dynamic-shape ops | Shape, Gather, Range, ReduceProd | **none** |

Parity: trained PyTorch ↔ original ONNX **6.218e-04 px**; static ↔ original
**3.052e-05 px**; all within the 1e-3 px tolerance fixed in advance.

The static variant was produced on request as an export-only portability
artifact — the candidate index grid, the unbiased-variance denominator and the
Resize scale factor baked in as constants. **No weight touched**, asserted
bit-identical before export. It is not a new model.

Host runtime (explicitly **not** target-device figures): 50.6 ms PyTorch CUDA,
773 ms onnxruntime CPU, 316.8 MiB peak CUDA, 1.62 MiB ONNX.

---

## 7. Hailo validation — how far it actually got

`phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md`,
`HAILO_COMPILATION_RECORD.json`.

| layer | status |
|---|---|
| **ONNX validated** | **PASS** — both artifacts, full integrity set |
| Hailo parser validated | **NOT ATTEMPTED** — toolchain absent |
| Hailo compiler validated | **NOT ATTEMPTED** — toolchain absent |
| HEF validated | **NOT ATTEMPTED** — no HEF produced |
| HailoRT runtime validated | **NOT ATTEMPTED** — HailoRT absent |
| Physical hardware validated | **NOT AVAILABLE** — no device present |

An exhaustive probe found no Dataflow Compiler, no HailoRT, no binaries, no
environment variables, no install directories, no Docker image, no WSL distro
and no PCIe accelerator. Every gap is environmental, not a property of the
model. **No latency, throughput or on-device accuracy number was fabricated.**

Two things carried forward for whoever runs the compile:

- **The reference `.alls` will not transfer.** Every line is spatial defusion of
  `conv42`–`conv52`, keyed to the reference graph's layer names, and it says
  nothing about the cost volume. P2A's model script must be regenerated.
- **Do not let any pass optimise away the disparity-shift padding.** The
  reference's shift is a proven no-op; P2A's 23 `Pad` clusters are load-bearing.
  If a compiler removes them the model silently reverts to reference behaviour.
  A post-compile equivalence check against host ONNX output is mandatory. This
  is flagged as inference, not as a measured parser result.

---

## 8. What this project produced, end to end

1. **15.3958 → 1.4150 px** on the frozen contract, a 10.9× reduction, with the
   final model at 397,954 parameters — 25,632 *fewer* than the mid-project
   incumbent — on an unchanged graph, unchanged operators and unchanged
   inference cost.
2. A **frozen evaluation contract** honoured across roughly two dozen trained
   models without a single exception: 40 scenes, 3,802,797 valid pixels,
   `contract_match true`, every number scored through the same harness.
3. A **preregistration discipline** that produced four closed negative results
   (ARM-W, ARM-X, ARM-Y, ARM-Z) and one INCONCLUSIVE that was not upgraded when
   it would have been convenient.
4. A **measured diagnosis** of the model's dominant failure — high-disparity
   supervision coverage — proved on frozen weights by an intervention that
   could only hurt matching and nevertheless fixed the stratum.
5. A **deployment artifact** with validated export, end-to-end scored accuracy,
   and an explicit, un-collapsed statement of exactly how far verification got.

### What remains genuinely unknown

- Whether P2A compiles for Hailo at all — **NOT VERIFIED**, no toolchain.
- On-device latency, throughput, memory and accuracy — **NOT MEASURED**.
- Whether the residual [128,144) px failure (31–35 px EPE, 0.028 % of pixels)
  shares the diagnosed cause — **NOT TESTED**.
- Whether a different scale range would do better — **NOT TESTED**, and
  deliberately so under the no-tuning rule.

---

## 9. Document index

**Phase 1 closure**
- `phase1/docs/FINAL_PHASE1_BOTTLENECK_AUDIT.md` — the audit that closed Phase 1
- `phase1/results/LEADERBOARD.md` — unmodified throughout Phase 2
- `phase1/harness/bottleneck_diag.py`, `phase1/runs/arm_v_diag/`

**Phase 2 diagnostic**
- `phase2/docs/PHASE2_ARMV_HIGH_DISPARITY_DIAGNOSTIC.md`
- `phase2/diagnostics/arm_v_high_disparity/` — five probes, JSON, figure

**The experiment**
- `phase2/docs/PHASE2_HYPOTHESIS_01.md` — preregistration
- `phase2/docs/PHASE2_P2A_SCALE_COVERAGE_RESULT.md` — result, 13 sections
- `phase2/runs/p2a_scale_coverage{,_s1,_s2}/` — three seeds, guards, logs
- `phase2/scripts/train_p2a_scale_coverage.py`, `eval_p2a.py`, `p2a_decide.py`

**Deployment**
- `phase2/docs/PHASE2_DEPLOYMENT_SELECTION.md` — the decision, kept separate
- `phase2/docs/PHASE2_DEPLOYMENT_VALIDATION.md` — export, parity, accuracy, runtime
- `phase2/deploy/p2a_stereonet.onnx`, `p2a_stereonet_static.onnx`
- `phase2/deploy/validation/HAILO_TOOLCHAIN_REPORT.md`

**Earlier phases (superseded numbering, left untouched)**
- `PHASE_1_FINAL_REPORT.md`, `PHASE_1_OVERVIEW.md`, `docs/`
- `phase2/docs/PHASE_2_REGISTRY.md` and the H1/H2/E-series records

---

## 10. Closing state

```
Phase 0                CLOSED
Phase 1                CLOSED    incumbent ARM-V
Phase 2 optimization   CLOSED    no further accuracy experiments
Phase 3                does not exist
Deployed model         P2A seed 0
P2A verdict            INCONCLUSIVE        (permanent)
P2A mechanism          NOT CONFIRMED       (permanent)
Hailo compilation      NOT VERIFIED
Hardware validation    NOT AVAILABLE
```

Remaining work is compiler, HEF and hardware validation. It is not another
accuracy hunt.
