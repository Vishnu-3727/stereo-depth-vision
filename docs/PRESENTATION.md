# Presentation outline — Microchip StereoNet

**Audience:** Microchip / the sponsor. **Length:** ~20 minutes + questions (15 slides).
**Purpose:** report what was built and what it achieves, state plainly what failed and what is
unknown, and leave with **one decision**: name the target device.

Every number below is quoted from a record in this repository; the source is named on each
slide so any figure can be checked live. Nothing here may be softened into a claim the record
does not support — see *Guardrails* at the end.

---

## Slide 1 — Title

**Stereo Depth Vision for Edge Silicon**
Frozen model, validated perception stack, and one blocking decision.

Presenter, date, and: *"All figures traceable to the project record — github.com/…/stereo-depth-vision"*.

---

## Slide 2 — The one-minute version

| | |
|---|---|
| We built | a 397,954-parameter stereo-depth network + a deterministic perception stack |
| It scores | **1.1912168 px** EPE on a frozen 40-scene KITTI contract |
| The benchmark it was measured against | the Hailo Model Zoo StereoNet reference: **1.3134471 px** |
| Where we started | 15.3958267 px |
| Validated beyond the network | metric depth, point cloud, occupancy, object edges — **405 checks, 0 failures** |
| Two gates failed, and stayed failed | ONNX export parity; the activation-rescale mitigation |
| What we need from you | **the target Hailo device.** Everything hardware-side is blocked on it |

**Speaker note:** land the ask in the first minute, then earn it over the next eighteen.

---

## Slide 3 — Why this project existed

- Microchip wants stereo depth on edge silicon, competitive on the tradeoff that actually
  matters in deployment: accuracy × latency × memory × power.
- The practical benchmark was Hailo's deployed StereoNet. Before beating something, you have
  to know exactly what it does.
- So the project began as forensics on the reference, not as model training.

*Source: `PHASE_1_FINAL_REPORT.md`.*

---

## Slide 4 — What we found in the reference (credibility slide)

Two findings that changed how the benchmark should be read:

1. **The published "EPE 8.223" is not an end-point error.** It is the KITTI D1 outlier
   *percentage*. We recovered the protocol and reproduced it exactly: **8.2237 vs 8.223**.
   Anyone benchmarking against 8.223 as an EPE would misjudge the model by an order of magnitude.
2. **The deployed reference performs no disparity search.** All twelve cost-volume slices are
   bit-identical — the shift is an algebraic no-op, present in the upstream source, surviving
   export and compilation.

**Speaker note:** this is why the project's own numbers are trustworthy — we did not take the
vendor's framing on faith, and we published the correction.

*Source: `docs/reproduction_report.md`, `docs/failure_analysis.md`, EXP-005, EXP-010.*

---

## Slide 5 — Where we got to

| Model | Frozen-contract EPE |
|---|---|
| Starting PyTorch baseline | 15.3958267 px |
| Phase-1 incumbent (ARM-V) | 1.7727436 px |
| P2A | 1.4149796 px (3-seed mean 1.4409826) |
| **ARM-P seed 1 — the frozen candidate** | **1.1912168 px** |
| Hailo reference ONNX | 1.3134471 px |

**12.9× reduction from the starting baseline, at 397,954 parameters** — 25,632 *fewer* than the
mid-project incumbent, on an unchanged graph and unchanged inference cost.

*Source: `RESULTS_INDEX.md` §0, `stage_b_armp/ARMP_CLOSURE_RECORD.md` §§4–7.*

---

## Slide 6 — What the final step actually was

- One variable changed: **weight initialization**. ARM-P = pretrained on SceneFlow first, then
  the identical KITTI fine-tuning recipe. Control = random init, same recipe.
- Result: ARM-P lower in **all six** seed × checkpoint-selection cells, across three seeds,
  plus a separate environment control on different hardware.

**State the bounds on the same slide — do not put them in the appendix:**
this does **not** establish universal superiority, statistical significance, that SceneFlow is
the causal mechanism, generalization beyond this split, or any hardware benefit.

*Source: `stage_b_armp/ARMP_CLOSURE_RECORD.md` §§11–13.*

---

## Slide 7 — The product is more than the network

Figure: `docs/images/demo_city.png`

Stereo pair → disparity → **metric depth in metres** → point cloud → occupancy → object edges.
**Exactly one stage is neural**; everything after disparity is deterministic host code.

| Stage | Validation |
|---|---|
| Metric depth, `Z = fB/d` | **C2 — 135/135 PASS** |
| Point cloud, spatial cells, occupancy | **C2.1 — 251/251 PASS** |
| Depth discontinuities | **C2.1.1 — 19/19 PASS** |

**Speaker note:** these validate the geometry pipeline's correctness and determinism — they say
nothing about Hailo hardware, and nothing about ground-truth depth accuracy in the field.

---

## Slide 8 — Live demo (2 minutes)

Run `python stage_c_deploy/demo/pipeline_demo.py --cloud3d`, then:

1. Point at a car → **click** → *"that point is 12.3 m away"*.
2. Switch to the 3D cloud, rotate once: road surface, building facades, camera at origin.

Figures if the live demo cannot run: `docs/images/demo_city_cloud3d.png`,
`docs/images/demo_highway.png`.

**Speaker note:** the demo prints the failures and the blocker on every run. Do not hide that
panel — it is the point.

---

## Slide 9 — Why the numbers can be trusted

- **One frozen evaluation contract** for the whole project: 40 KITTI scenes, 3,802,797 valid
  pixels, `disp_occ_0`, GT 1/256. Held on all 16 scored Stage-B checkpoints. Never edited to
  improve a number.
- **Preregistration:** every experiment declared its hypothesis, frozen variables, and success
  *and rejection* criteria before the run. Four arms were closed as negative results; one was
  left INCONCLUSIVE rather than upgraded.
- **Two independent whole-system audits**, run separately, reaching the same verdict, with
  every artifact hash re-measured.

*Source: `phase1/harness/frozen_eval.py`, `stage_c_deploy/FINAL_WHOLE_SYSTEM_AUDIT{,_SECOND_PASS}.md`.*

---

## Slide 10 — What failed

| Gate | Criterion | Measured | Verdict |
|---|---|---|---|
| **C1** ONNX export parity | max abs diff < **1e-3 px** | **1.708984375e-3 px** | **FAIL** |
| **DR-1** activation rescale | equivalence within 1e-3 px | **1.4816284e-2 px** | **FAIL** |

- C1: five export variants (opset 11/13/17 × constant folding) produce **bit-identical**
  outputs — no export-level fix exists in the tested space. The error is carried by the
  refinement residual, not the readout.
- DR-1: the obvious mitigation (rescale the aggregated cost) is **rejected**, not adopted.

**Speaker note:** these are kept as failures on purpose. The decision on C1 belongs to you —
treat it as blocking for these weights, commission weight-side work, or re-specify the gate in
writing. We did not quietly move the threshold.

---

## Slide 11 — What is unknown (and stays unknown)

- ARM-P carries very large fp32 intermediates (~1e18, against the reference's ~1e1–1e2).
  Whether they survive target-native int8 quantization is **UNKNOWN** — no threshold exists, so
  we invented none, and we do **not** claim it will fail.
- Two research branches closed as **NOT IDENTIFIABLE** on this platform (coverage-vs-pretraining,
  true group-wise correlation). Group-wise correlation itself is **open but unsupported — not
  refuted**.

**Speaker note:** if anyone asks "so will it quantize?" — the honest answer is that nobody can
know until it is run on the real toolchain, and that needs slide 13.

---

## Slide 12 — What is complete, device-independently

- Frozen, hash-verified checkpoint and ONNX exports.
- The full perception stack, validated 405/405.
- Export parity, dynamic range, rescale, target and toolchain all audited and recorded.
- A runnable demo, a navigable repository, and two closure audits.

**Nothing further can be done here without slide 13.**

---

## Slide 13 — The blocker

> **The target Hailo device has never been specified.**

- No requirement for device, board, SDK/DFC version, resolution, FPS, power or quantization
  mode exists anywhere in the project record.
- Historical references to Hailo-8 / 10H / 15H come from the vendor's Model Zoo. We refused to
  promote any of them to "the target" — a guessed target produces a scrap HEF.
- Consequence: **no parse, no quantization, no compile, no HEF, no latency, no power figure.**

*Source: `stage_c_deploy/C1_TARGET_TOOLCHAIN_RESOLUTION.md`, `stage_c_deploy/STAGE_C_DEPLOYMENT_BLOCKER_REPORT.md`.*

---

## Slide 14 — The ask: 14 fields, four groups

| Group | What we need |
|---|---|
| **Target** | device / part number, board or module, physical hardware availability |
| **Toolchain** | required HailoRT version, required DFC / compiler version, runtime environment, whether emulator-only characterization is acceptable |
| **Product envelope** | input resolution, FPS / latency requirement, power limit, memory limit |
| **Integration** | quantization mode, HEF + runtime expectations, camera / stereo rig spec |

*Full list: `stage_c_deploy/STAGE_C_DEPLOYMENT_READINESS_FINAL.md` §17.*

**Speaker note:** ask for these in writing, and say which single one unblocks the most: the
device/part number.

---

## Slide 15 — What happens once we have them

1. Install the matching DFC / HailoRT → **parse** the 712-node opset-13 graph.
2. **Quantize** against the disjoint `hailo_calib` set (scenes 0–159) → measure the accuracy
   drop; this is where the dynamic-range question is finally answered.
3. **Compile** → HEF → on-device accuracy, latency, FPS and power, measured, never estimated.
4. Report against the same frozen contract, so the numbers stay comparable to everything above.

Parallel, and independent of the device: the C1 decision from slide 10.

**Closing line:** *"The model is frozen, the evidence is complete, and the failures are on the
record. One answer from you turns this from a validated prototype into a measured deployment."*

---

## Backup slides (only if asked)

- **B1 — Gap diagnosis:** where the remaining error lives (~31% below 64 px, ~66% in 64–128 px,
  ~3% above), and why the architecture gate returned NO ARCHITECTURE JUSTIFIED.
- **B2 — The superseded research line:** the correspondence campaign, closed at Level D —
  "genuine geometric correspondence not demonstrated" — and why that negative result was worth
  recording.
- **B3 — Risk register:** R1–R10 from the readiness report, with what closes each.
- **B4 — Repository tour:** README → `RESULTS_INDEX.md` → the closure records.
- **B5 — Cost of a wrong guess:** why a HEF compiled for the wrong part is scrap, not a draft.

---

## Guardrails — do not say these

The project's credibility rests on the same discipline in the room as in the record.

| Never say | Say instead |
|---|---|
| "We beat Hailo" | "Lower EPE than the reference model on our frozen contract" |
| "It's deployed" / "It runs on Hailo" | "Frozen deployment candidate; deployment is blocked on the target device" |
| "It's faster / lower power" | nothing — no latency or power figure exists |
| "Quantization will be fine" / "will fail" | "Unknown until run on the real toolchain" |
| "We proved pretraining causes the gain" | "A repeated directional result; the causal mechanism is not isolated" |
| "The export bug is minor" | "A frozen FAIL; the disposition decision is yours" |
| dev-machine timings as product performance | "Development-machine measurement, never Hailo" |

---

## Assets

| Asset | Path |
|---|---|
| Pipeline figure (city) | `docs/images/demo_city.png` |
| 3D point cloud | `docs/images/demo_city_cloud3d.png` |
| Pipeline figure (highway) | `docs/images/demo_highway.png` |
| Live demo | `python stage_c_deploy/demo/pipeline_demo.py --cloud3d` |
| Every number's source | `RESULTS_INDEX.md` |

Regenerate any figure with:
`python stage_c_deploy/demo/pipeline_demo.py --scene 12 --no-window --cloud3d --save docs/images/demo_city.png`
