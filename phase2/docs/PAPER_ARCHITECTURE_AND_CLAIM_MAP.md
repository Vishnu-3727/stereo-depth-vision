# Paper architecture and claim map

**Date:** 2026-09-13. **Type:** writing plan. **Nothing was executed** — no
experiment, no checkpoint, no training, no new measurement. Every number is
quoted from an existing record.

**Ceiling, unchanged and enforced throughout:**
**Level D — genuine geometric correspondence NOT-DEMONSTRATED.**

Predecessor: `phase2/docs/POST_CLOSURE_RESEARCH_AUDIT.md`.

---

## 1. Paper question

> **Can stereo accuracy and geometric correspondence be experimentally separated
> in a cost-volume stereo network, and what can controlled interventions
> establish about which of the two a given network has?**

It presupposes nothing: a negative answer ("no, they always co-occur, and
interventions attribute cleanly") is a coherent outcome the evidence could have
produced and did not.

---

## 2. Central thesis

> A deployed, production stereo network — Hailo's StereoNet — builds a cost
> volume that is **candidate-invariant by algebraic identity**: the disparity
> shift is implemented as a pad-right followed by a slice from zero, so all
> twelve candidate slices are the same tensor, `max_k |V_k − V_0| = 0.0` exactly.
> The construction is present in the upstream PyTorch source, survives export and
> compilation, and is re-confirmed bit-exactly on a trained checkpoint and across
> thousands of permutation arms. The same model nevertheless reaches **8.154 % D1
> under the official protocol** and is **strongly binocular under the EXP-007
> ablation protocol** (a different pixel set from the official protocol): five right-image
> ablations each cost roughly **+89 to +90 D1 points**. Useful stereo prediction
> is therefore achievable without any candidate-varying geometric information in
> the cost volume — while the network still consumes a single-alignment binocular
> difference, so this is a statement about the absence of a *search*, not about
> the absence of all geometric use. Against that empirical separation we place an
> **identification boundary**: across eight intervention families we find that
> each apparently geometric signature — translation response, candidate-axis
> ordering, matching-path gradient, softmax sharpness, accuracy — admits an
> explicit correspondence-free counterexample, and that two of the failures are
> structural rather than incidental, one architectural (a zone-local statistic is
> bit-identically the global one under a bounded receptive field) and one
> logical (for a fixed left image, no null can be simultaneously content-matched
> inside the tested window and geometrically non-corresponding). We therefore
> **do not claim that this network lacks correspondence**; we claim that stereo
> performance and *identifiable* geometric correspondence come apart, and we
> delimit what standard interventions can and cannot establish.

---

## 3. Contributions

### C1 — A candidate-invariant cost volume in a deployed model, with strong binocular dependence

| | |
|---|---|
| **Claim** | A shipped stereo network's cost volume contains no candidate-varying information, yet the network is strongly binocular and retains useful accuracy. |
| **Evidence** | EXP-010 (`max_k \|V_k − V_0\| = 0.0`, 5 scenes; algebraic identity in `reference_shift`; upstream source SR-011); re-confirmation at 0.0 across 6 160 arms (INDEX-002) and 816 arms over 17 weight-sets incl. 16 random (ARCH-001); `PHASE_2_BASELINE` D1 8.154 % official protocol; EXP-007 right-image ablations. |
| **Claim level** | **A** for the identity and the ablation magnitudes; **B** for "binocular dependence is causal" (direct intervention). |
| **Strongest surviving alternative** | The binocular dependence could be a non-geometric use of `Lf − Rf` — which is precisely what we do *not* adjudicate. |
| **Exact limitation** | One model, one dataset split, one candidate count. "Useful" is anchored to 8.154 % on `hailo_val` under the official protocol, not to a literature comparison. |

### C2 — A controlled 2×2 showing cost-volume candidate structure and readout conditioning interact

| | |
|---|---|
| **Claim** | Neither a genuinely shifted volume nor a conditioned readout alone produces a binocular trained model; only their combination does. |
| **Evidence** | `none+plain` 14.423 ± 0.228 EPE, `left+plain` 14.414 ± 0.227 (H1-v2 late-window means on the first 10 `hailo_val` scenes, pre-determinism harness), `none+std` 8.3717, `left+std` 2.2955 (deterministic 40-scene pooled, 3 802 797 px) — matched recipe, 423 586 parameters in every cell; corroborated directionally by H3 (right-image ΔD1 −1.08…+1.91 with the shift removed). |
| **Claim level** | **B** — single changed variable per cell, bit-reproducible harness. |
| **Strongest surviving alternative** | Materiality of candidate-varying content says nothing about *how* it is used; a candidate-pointwise mechanism reproduces the downstream signatures (O1 audit, GEOM-002 G-STOP). |
| **Exact limitation** | One seed per factorial cell — defensible only because the 6.08 px effect is ≈ 59× the separately measured 6-block three-seed EPE range of 0.1027 px. 160 scenes, random initialisation. Not a competitive-accuracy result. |

### C3 — An identification boundary for correspondence claims, with explicit counterexamples

| | |
|---|---|
| **Claim** | The standard signatures used to attribute correspondence in cost-volume networks each admit a correspondence-free counterexample, and several intervention families are structurally non-identifiable. |
| **Evidence** | SHIFT-001 (a provably search-free model responds to translation by up to 0.6975 candidates and is *not* uniform: 3.6452–4.2717 against a predicted 5.5); EMERGENCE-001 (matching-path gradient 100 % of batches at every checkpoint, including one where corrupting the right image *improves* D1 by 0.91 pt); E2 (the sharpest readout destroys stereo function); INDEX-001/002 (random permutation reproduces the ordered pattern); GEOM-002 (statistic architecturally available, random aggregation scores h = 6); O1 (candidate-pointwise latch); Zone-001 (`ZI-1`/`ZI-2` = 0.0 ⇒ the zone statistic *is* the global one); `S*` audit (A4s, A5r, null-impossibility); field gate (M1). |
| **Claim level** | **A** for each counterexample (each is a measurement or an identity); **C** for the generalisation beyond the enumerated families. |
| **Strongest surviving alternative** | That a cleverer intervention exists. We cannot exclude it; we can only show that every family we could construct from this data fails, and give the structural reason. |
| **Exact limitation** | The boundary is *proved* for the enumerated families and *argued* to generalise. The generalisation rests on two arguments, both stated explicitly and both attackable. |

### C4 — Methodological negatives that transfer

| | |
|---|---|
| **Claim** | Four commonly used proxies are invalid for mechanism attribution: gradient flow, softmax sharpness, inference-time pruning sensitivity, and oracle substitution. |
| **Evidence** | EMERGENCE-001 (gradient); E2 (sharpness: T = 0.25 gives entropy 0.290 nats and right-dependence +0.57 / −2.98 / −1.17); E3 (dropping block 4 costs seed 0 +0.19 px but seeds 1/2 +24.16 / +16.37 px — a one-seed pruning study would conclude block 4 is free); E1 (a diverging inverse-oracle returned ratio 1.000 and *falsely confirmed* the pre-registered verdict). |
| **Claim level** | **A**/**B** — each is a direct measurement in this system. |
| **Strongest surviving alternative** | These are demonstrations in one architecture; that they generalise is an inference. |
| **Exact limitation** | Single architecture, single recipe. Presented as cautions, not as laws. |

**Rejected as contributions** (implementation detail, not science): the
independent reimplementation and ONNX equivalence; the MAC accounting; the
visualizer; the determinism harness. All become *methods* text.

---

## 4. IEEE paper structure

The brief's skeleton is adopted with **one deliberate deviation**: a single
"V. Results" section would force two independent empirical pillars and a
methodological argument into one container. They are split into three sections,
which also lets the claim ceiling differ between them — sections V and VI carry
Level A/B claims, section VII carries Level C/D.

| § | Purpose | Key claims | Figures / tables | Experiments used | Explicitly excluded | Ceiling |
|---|---|---|---|---|---|---|
| **I. Introduction** | Pose the question; state the separation; state the boundary | C1, C3 in one sentence each | F1 | — | — | Mixed; must state Level D for correspondence in the intro itself |
| **II. Related Work** | Cost-volume stereo; mechanistic interpretability; identifiability in deep models | none | — | — | — | — |
| **III. The Deployed Model and Problem Formulation** | Define the architecture, the cost volume, `H_CORR` vs `H_ALT`, and what "identifiable" means here | Formal statement of the two hypotheses | F1 | Phase-1 reproduction (methods) | — | Definitional |
| **IV. Methodology and Protocol Discipline** | Determinism, pre-registration, the MEASURED/DERIVED/INFERRED/UNKNOWN convention, protocol/pixel-set labelling | Bit-reproducibility (identical weight SHA `3ad382c50d413ad2`); harness was *not* reproducible before the controls (epoch-0 loss 10.8798 vs 10.5762) | T1 | Determinism diagnostic; Stage B | — | A |
| **V. A Candidate-Invariant Cost Volume with Strong Binocular Dependence** | Pillar 1 | C1 | F2, F3, T2 | EXP-010, EXP-007, `PHASE_2_BASELINE`, INDEX-002/ARCH-001 negative controls | H1/H1-v2; all from-scratch accuracy | **A/B** |
| **VI. Shift × Readout: What Candidate-Varying Geometry Contributes** | Pillar 2 | C2 | F4, T3 | Shift factorial, H1-v2 cells, H2 + 3 seeds, H3 | E1 oracles | **B**, with an explicit "not correspondence" paragraph |
| **VII. Adversarial Identification Analysis** | Pillar 3 — the boundary | C3, C4 | F5, F6, F7, T4 | SHIFT-001, EMERGENCE-001, INDEX-001/002, ARCH-001/RATE, TR-001, GEOM-001/002, O1, CP-001, Zone-001, `S*`, field gate | the full chronology; every design-only record | **C/D** |
| **VIII. Architectural Sensitivity** | Pillar 4, secondary | Block depth; readout temperature; refinement truncation | T5 | Block-depth 3×3, E2, E3 | E3b/E3c/O6 legacy readings (invalidated) | **B** for the 5-vs-6 EPE penalty only; **D** for any 4-vs-6 comparison; empirical sensitivity only for E2/E3 |
| **IX. Discussion** | What the separation means for evaluating stereo networks | Performance ≠ identifiable correspondence | — | — | — | C |
| **X. Limitations** | 160 scenes; n = 3; one architecture; one dataset; no p-values; the generalisation step in C3 | — | — | — | — | — |
| **XI. Conclusion** | Restate C1–C3 at ceiling | — | — | — | — | — |

---

## 5. Title candidates, ranked

1. **"A Candidate-Invariant Cost Volume in a Deployed Stereo Network: Separating
   Stereo Accuracy from Identifiable Geometric Correspondence"** — most
   informative, entirely conservative, names both pillars. **Recommended.**
2. "Stereo Performance Without Candidate-Varying Cost-Volume Geometry: An
   Identifiability Analysis of a Deployed Stereo Network" — strong; slightly
   heavier.
3. "On the Identifiability of Geometric Correspondence in Cost-Volume Stereo
   Networks" — cleanest, but leads with the methodological pillar and buries the
   empirical anchor.
4. "What Controlled Interventions Can Establish About Cost-Volume Stereo
   Networks: An Adversarial Identification Study" — accurate, reads as a methods
   paper.
5. "Binocular Without Search: An Adversarial Analysis of Correspondence Claims in
   a Deployed Stereo Network" — most readable, ranked last because "without
   search" invites over-reading in citation.

None asserts that StereoNet lacks correspondence.

---

## 6. Abstract plan

| Slot | Content | Constraint |
|---|---|---|
| **Background / problem** | Cost-volume stereo networks are assumed to perform disparity search over candidate slices; that assumption is rarely tested on deployed artefacts. | No claim. |
| **Gap** | Signatures normally taken as evidence of correspondence have not been checked against models that provably cannot search. | No claim. |
| **Method** | Forensic reconstruction of a deployed model, a bit-reproducible deterministic training harness, pre-registered interventions, and an adversarial protocol in which each candidate signature must survive an explicit correspondence-free counterexample. | Factual. |
| **Main empirical result** | The deployed model's cost volume is candidate-invariant by algebraic identity (`max_k \|V_k − V_0\| = 0.0`); it reaches 8.154 % D1 under the official protocol and loses ≈ 89–90 D1 points under right-image ablation in the EXP-007 protocol (10 scenes, 1 837 304 px). A controlled 2×2 shows candidate-varying content and readout conditioning interact (2.2955 vs 8.3717 EPE, deterministic 40-scene pooled; plain cells are H1-v2 10-scene late-window means). | Numbers with protocols. |
| **Identification result** | Across eight intervention families each apparently geometric signature admits a correspondence-free counterexample; two failures are structural. Genuine local correspondence is therefore **not demonstrated** — which is not the same as absent. | Must contain the words "not demonstrated" and the disclaimer. |
| **Significance** | Stereo performance and *identifiable* geometric correspondence can come apart, and mechanism claims in cost-volume stereo need stronger evidence than the standard signatures. | No promotion. |
| **Limitation** | One architecture, one dataset, 160-scene from-scratch training, three seeds, no significance tests. | Explicit. |

Banned from the abstract: "prove", "demonstrate absence", "reveals that the
network does not", "surprisingly", "we are the first".

---

## 7. Claim–evidence matrix

Every sentence in the paper must trace to a row here.

| # | Claim | Evidence | Exact result | Type | Strongest alternative | Rules out | Does NOT rule out | Level |
|---|---|---|---|---|---|---|---|---|
| 1 | The deployed cost volume is candidate-invariant | EXP-010 + `reference_shift` | `max_k \|V_k − V_0\| = 0.0`, 5 scenes; identity: pad right, slice `[0:W]` | Identity + measurement | none | Any disparity search in that volume | Binocular use of `Lf − Rf` at one alignment | **A** |
| 2 | It is not an export artefact | Upstream PyTorch source [SR-011]; equivalence at 5 taps | agreement at all taps | Source + measurement | none | "our reimplementation is wrong" | — | **A** |
| 3 | Invariance is a property of the volume, not the weights | ARCH-001 | 816 arms, 17 weight-sets (1 trained + 16 random), all `0.0` | Measurement | none | weight-specific explanation | — | **A** |
| 4 | The deployed model is accurate | `PHASE_2_BASELINE` | **D1 8.154 %**, official protocol (EPE 1.3134 px on that protocol); fp32 CUDA 1.314 px / 8.155 % / 90.1 ms tabulated separately in `PHASE_2_BASELINE` | Measurement | none | — | Competitiveness vs modern networks | **A** |
| 5 | The deployed model is strongly binocular | EXP-007 | +89.678 / +89.957 / +89.128 / +89.649 D1 pt (right:=left, other scene, flip, black; noise +89.806 in the same record); baseline EPE 1.4422 / D1 9.3315 on 1 837 304 px (10-scene EXP-007 set); baseline correlation −0.310 / +0.00019 / +0.121 / +0.348 (noise −0.147) | Intervention | The dependence may be non-geometric | Pure monocular prediction | Geometric *use* of the right image | **A/B** |
| 6 | **Useful stereo prediction without candidate-varying correspondence information** | rows 1–5 | — | Composite | none | "search is necessary for useful stereo here" | "the network is correspondence-free" | **B** |
| 7 | A readout change alone makes a trained model binocular | H2 + 3 seeds | entropy 0.0000 → 1.67–1.83 nats; gradient 5/150 → 16 000/16 000; `r(GT)` −0.001…+0.283 → +0.978…+0.988; right-image ΔD1 ≤ +0.8 → +74.88…+83.95 pt; EPE 14.41 → 4.199/3.745/4.333 | Intervention, replicated | — | "the volume alone determines binocularity" | that H2's binocularity is correspondence | **B** |
| 8 | Candidate-varying content is material | Shift factorial | `left+std` 2.2955182 / D1 15.6045143 vs `none+std` 8.3717466 / D1 66.5977700 (both deterministic 40-scene pooled); gap 6.08 px ≈ 59× the 0.1027 px 6-block seed range | Intervention, deterministic | n = 1 seed/arm | "the shift is irrelevant" | that it is used as correspondence | **B** |
| 9 | The two factors **interact** | 2×2 | `none+plain` 14.423, `left+plain` 14.414 (H1-v2 10-scene late-window means, pre-determinism harness), `none+std` 8.3717, `left+std` 2.2955 (deterministic 40-scene pooled) | Factorial | — | single-factor explanations | mechanism | **B** |
| 10 | Binocular function is configuration-wide | E2 + E3 | T = 0.25 → right-dep +0.57 / −2.98 / −1.17; T = 0.75 costs +3.5 px / +56 pt; < 5 refinement blocks → right-dep −14…−18 pt | Intervention, 3 seeds | — | "binocularity lives in the volume" | mechanism | **B** |
| 11 | A search-free model still responds to translation | SHIFT-001 | max response **0.6975 candidates**; `disparity_initial` 3.6452–4.2717 vs predicted 5.5; max deviation 1.8548 | Measurement | none | translation response as correspondence evidence | — | **A** |
| 12 | Matching-path gradient is not a stereo proxy | EMERGENCE-001 | 100 % of batches at every checkpoint; at seed 2 / epoch 20 right-image corruption *improves* D1 by 0.91 pt; re-measurement reproduces to 0.00 pt | Measurement | none | gradient-based mechanism gates | — | **A** |
| 13 | Candidate-axis ordering is not discriminating | INDEX-001/002 | 24 640 arms; random permutation reproduced the ordered pattern; algebraic null 6 160 arms at `0.0` | Measurement | — | ordered-vs-random tests | — | **A** |
| 14 | Training installs a calibrated candidate-axis response absent from untrained weights | TR-001 vs ARCH-RATE | trained α_h median **+1.0014**, min +0.9534, 12/12 outside; random median +0.0032, max +0.2441; **0/384** untrained qualify, 378/384 flat | Measurement | A candidate-pointwise latch reproduces the signature | "the response is architectural alone" | that it is correspondence | **C** |
| 15 | The statistic is architecturally available | GEOM-002 | G-STOP, `STATISTIC-ARCHITECTURALLY-AVAILABLE`; random aggregation scores h = 6 | Measurement | — | that arm as evidence | — | **A** |
| 16 | A zone-local statistic is bit-identically the global one | Zone-001 | `ZI-1` = `ZI-2` = **0.0**, 3 seeds, Δ ∈ {1,3,6}; clean interiors `[0,224)` / `[912,1136)` | Measurement + RF argument | none | all zone-local designs | — | **A** |
| 17 | No paired null can be both content-matched and non-corresponding, for fixed `L` | `S*` audit | two-line argument; plus A4s, A5r | Logical | The argument's generalisation beyond fixed-`L` paired nulls | that family of designs | designs outside the family | **A** (within scope) / **C** (generalised) |
| 18 | A spatially varying field does not escape the boundary | Field gate | `D-y` feasible and artefact-free, kills A3/A5/A9/M1; A4s/A5r survive | Design analysis | — | that family | — | **C** |
| 19 | Five refinement blocks carry a reproducible penalty | Block-depth 3×3 | EPE ranges disjoint by **+0.0748 px**, 9/9 pairings; D1 ranges **overlap by 0.13 pt** | Measurement | — | "5 = 6 at these seeds" | causation, monotonicity, transfer | **B**, narrow |
| 20 | Four blocks are inconclusive | Block-depth 3×3 | arm straddles `[2.2955182, 2.3982217]`; 7/9 pairings; arm spread 0.3166 px = 3.08× reference | Measurement | — | "4 blocks match 6" | that 4 differs from 6 | **D** for that comparison |
| 21 | Which block is redundant is a property of the run | E3 | block 4 removal: seed 0 +0.19 px; seeds 1/2 +24.16 / +16.37 px | Measurement | — | one-seed pruning studies | — | **A** |
| 22 | The harness required determinism controls | Determinism diagnostic | same seed, epoch-0 loss 10.8798 vs 10.5762; `F.interpolate` bilinear backward isolated; controlled runs identical; Stage B weight SHA `3ad382c50d413ad2` twice | Measurement | — | uncontrolled comparisons | — | **A** |
| 23 | Genuine local correspondence is **not demonstrated** | rows 11–18 | — | Composite | — | nothing | **that it is absent** | **D** |

---

## 8. Figure plan (7 main figures)

| Fig | Shows | Must **not** be read as |
|---|---|---|
| **F1** — Pipeline and the shift construction | The deployed architecture, `V[k] = shift_k(Lf) − Rf`, and the pad-right/slice-from-zero identity that makes `shift_k` a no-op; states `H_CORR` vs `H_ALT` | A claim about what the aggregation does |
| **F2** — Candidate invariance | `max_k \|V_k − V_0\|` across scenes, checkpoints and weight-sets, all `0.0`, with the algebraic identity inset | "the network ignores the right image" — F3 immediately refutes that |
| **F3** — Binocular dependence | D1 penalty for each of five right-image ablations (+89.1…+90.0 pt) against the labelled baseline, with the pixel set and protocol named on the axis | Evidence of *geometric* use of the right image |
| **F4** — Shift × readout factorial | Four bars, EPE and D1, one changed variable per cell, with evaluation sets labelled (plain cells H1-v2 10-scene; std cells deterministic 40-scene) | Evidence that the shift is used as correspondence |
| **F5** — Emergence with the gradient proxy pinned | Right-image dependence vs epoch for seeds 1 and 2, with matching-path gradient at 100 % throughout, and the point where corrupting the right image *improves* D1 | A claim about when correspondence appears |
| **F6** — Destroyed from both ends | Right-image dependence vs readout temperature (E2) and vs refinement depth (E3) on a shared axis | A capacity or optimality claim |
| **F7** — Identification ladder | signature → counterexample → stronger intervention → new counterexample → boundary, as a single flow with the eight families labelled | A chronology, or a claim that no intervention could ever work |

Block depth is a **table**, not a figure — a three-level inconclusive result
should not be given a figure's rhetorical weight.

---

## 9. Table plan

**Main paper (5):**

| Tab | Content | Why it advances the argument |
|---|---|---|
| **T1** | Protocol table: for every number in the paper — dataset, split, scene count, valid-pixel set, checkpoint, seed, protocol, metric, pipeline stage | Makes §11's separation auditable by the reader; prevents the 8.154 / 9.3315 conflation |
| **T2** | Deployed model: accuracy under the official protocol, plus the five ablations under the EXP-007 protocol, **in two clearly separated blocks** | Pillar 1 |
| **T3** | The 2×2 factorial, EPE and D1, with parameters and MACs identical across cells and evaluation sets labelled per cell (plain cells: H1-v2 10-scene means; std cells: deterministic 40-scene pooled) | Pillar 2 |
| **T4** | Identification boundary: family → signature → explicit counterexample → verdict | Pillar 3, and the replacement for a chronology |
| **T5** | Block depth 3×3 (EPE and D1, means and ranges) with the pre-registered decision boundary | Pillar 4 |

**Supplementary:** H1/H1-v2 in full · H2 pre-check and per-seed tables ·
EMERGENCE-001's 12-row series · E1's three oracles · E2's 9 × 3 grid · E3's 13
ablations · TR-001's α_h per unit and raw response curves · ARCH-RATE's 384-unit
classification · the determinism diagnostic · every pre-registration hash · the
per-scene block-depth comparisons.

---

## 10. Correspondence-audit presentation

**Rule: the reader must reach the boundary in one section, not one chronology.**

**Main paper, §VII**, in this order, roughly one paragraph each:

1. The setup: what a correspondence signature is, and the standard of proof
   adopted (a signature counts only if no correspondence-free mechanism
   reproduces it).
2. **Three counterexamples, strongest first** — SHIFT-001's search-free model
   responding to translation (0.6975 candidates); EMERGENCE-001's gradient proxy
   (100 % everywhere, including where the right image *hurts*); the
   candidate-pointwise latch that reproduces candidate-axis ordering.
3. **The strongest attempted intervention and why it failed structurally** —
   zone-local translation, with `ZI-2 = 0.0` showing the statistic *is* the
   global one.
4. **The null-impossibility argument**, stated in two lines, with A4s and A5r as
   its concrete consequences.
5. **The boundary**, as T4 and F7: what the eight families share, and what would
   be needed to cross it (data in which correspondence varies independently of
   content).

**Supplement:** the audit trail — individual records, pre-registrations, hashes,
halted runs, superseded records, deterministic verification, extra seeds,
auxiliary statistics, and the design-only gates — covering INDEX-001/002/003,
ARCH-001/ARCH-RATE, TR-001, GEOM-001/002, O1, CP-001/HS-BAND, Zone-001
pre-freeze, the `S*` identification audit, and the spatial disparity-field
design gate. Each is *cited* from §VII, never narrated in it.

**Never write:** "we then tried…", "this also failed", or any sentence whose
subject is the project rather than the network.

---

## 11. Numerical protocol separation

**The two figures that must never share a table block:**

| | Official protocol | EXP-007 ablation protocol |
|---|---|---|
| D1 | **8.154 %** | **9.3315 %** |
| EPE | 1.3134 px (official; fp32 CUDA 1.314 px / 8.155 % tabulated separately) | 1.4422 px |
| Scenes | 40 (`hailo_val` 160–199) | 10-scene EXP-007 variant set |
| Valid pixels | 3 802 797 px on the full 40-scene eval | **1 837 304** |
| Checkpoint | pretrained Hailo-exported weights | same pretrained weights |
| Role | headline accuracy | ablation baseline |

Ablation *deltas* (+89.1…+90.0 pt) are computed **within** the EXP-007 protocol
and are valid there; they must never be subtracted from the 8.154 % figure.

**Audit fields required on every reported number**, to appear in T1: dataset ·
split · scene count · valid-pixel set and rule · checkpoint identity (and hash
where recorded) · seed · evaluation protocol · metric · pipeline stage
(`disparity_initial` vs final). Three stage-sensitive cases to watch: H2's
`r(GT)` is on `disparity_initial`; all EPE/D1 figures are on the final output;
the 40-scene deterministic figures (2.2955 etc.) are a different evaluation set
from both the 10-scene ablation set and the official protocol.

---

## 12. Claim language

### A. Safe sentences (usable verbatim)

* "The deployed model retains useful stereo prediction despite a
  candidate-invariant cost volume."
* "All twelve candidate slices are the same tensor; this is an algebraic
  identity of the construction, not an empirical observation."
* "Useful binocular stereo behaviour can exist without candidate-varying
  correspondence information in the cost volume."
* "The network still consumes a single-alignment binocular difference, so this is
  a statement about the absence of a disparity search, not about the absence of
  all geometric use."
* "Candidate-varying cost-volume geometry contributes materially to the trained
  solution under this recipe."
* "Binocular dependence is a property of the whole trained configuration: it is
  destroyed by truncating refinement below five blocks and by moving the readout
  temperature away from its trained value in either direction."
* "Genuine local geometric correspondence is **not demonstrated** by these
  experiments; this is not the same as showing it absent."
* "Each of these signatures admits an explicit correspondence-free
  counterexample."
* "Five refinement blocks carry a reproducible EPE penalty relative to six at
  three seeds; the D1 ranges overlap by 0.13 points."

### B. Dangerous sentences (look reasonable, exceed the evidence)

| Sentence | Why it fails | Safe rewrite |
|---|---|---|
| "The cost volume is useless." | It is load-bearing — it carries `Lf − Rf` | "The cost volume carries no candidate-varying information." |
| "The shift is what makes the model stereo." | Confuses materiality with mechanism | "Candidate-varying content is material to the trained solution." |
| "The model learns to match." | Level D | "The model becomes dependent on the second camera." |
| "Our interventions show the model relies on monocular cues." | Contradicted by row 5 | delete |
| "Six blocks are necessary." | No causal claim is supportable | "Six blocks are empirically useful in this configuration." |
| "The effect is significant." | No p-value is computable at n = 3 | "The arm ranges are disjoint by 0.0748 px, 9/9 pairings." |
| "We prove an identification boundary." | Proved for the enumerated families; argued beyond | "We establish an identification boundary for the families examined, and give the structural reason it is expected to extend." |
| "Trained aggregations perform matching, untrained ones do not." | Reads TR-001 as mechanism | "Training installs a calibrated candidate-axis response that untrained aggregations do not produce." |

### C. Prohibited sentences (must never appear)

* "StereoNet does not use correspondence." / "…does not perform matching."
* "StereoNet performs genuine geometric correspondence."
* "We prove the absence of disparity search in the trained models."
* "The network ignores the second camera."
* "Four blocks match six."
* Any p-value, "statistically significant", or confidence interval on an
  architecture comparison.
* Any latency, power or deployment claim for the from-scratch models.
* Any claim of transfer to another dataset, recipe or candidate count.
* Any sentence that reports a non-identifiability result as a negative finding
  about the network.

---

## 13. Reviewer attack test

### Reviewer 1 — "Your candidate-invariant volume only proves the *aggregation* cannot use candidate variation. The network still receives `L − R` information."

* **Strongest form:** the headline conflates "no search" with "no correspondence";
  a single-alignment binocular difference plus a deep aggregation could implement
  something correspondence-like without candidate slices.
* **Valid?** **Yes, entirely.** It is not a misreading; it is the correct reading.
* **Evidence available:** EXP-007 shows the binocular channel is load-bearing
  (+89–90 D1 pt), which *supports* the reviewer.
* **Response:** we concede and adopt it — the claim is stated as "no
  candidate-varying correspondence information in the cost volume", never as "no
  correspondence", in the title, abstract, every results paragraph and the
  conclusion. The reviewer's point is the reason the paper has §VII at all:
  having conceded that the residual binocular channel might be geometric, we ask
  whether interventions can decide, and find they cannot.
* **Remaining weakness:** a casual citer will still compress us to "StereoNet
  doesn't do stereo matching". Mitigation is editorial — the disclaimer appears
  in the abstract, not only in Limitations.

### Reviewer 2 — "8.15 % D1 may simply reflect monocular prediction."

* **Strongest form:** KITTI disparity is heavily predictable from a single image;
  a monocular model could reach single-digit D1, making the result unsurprising.
* **Valid?** **No, as stated** — and we have the direct measurement.
* **Evidence available:** EXP-007. Five right-image ablations cost
  +89.128 to +89.957 D1 points (noise +89.806 in the same record); correlation with the baseline prediction falls
  from 1.0 to −0.310 (right:=left), +0.00019 (other scene), +0.121 (flip),
  +0.348 (black), −0.147 (noise). A monocular predictor is by definition unaffected by any of
  these. Corroborated from the other direction by H3, where removing
  candidate-varying content *does* produce the monocular signature
  (ΔD1 −1.08…+1.91), showing the ablation is capable of detecting monocularity
  when it is present.
* **Response:** quote the ablation table, state the protocol and pixel set
  (1 837 304 px, EXP-007 baseline D1 9.3315 %), and note explicitly that this
  baseline is not the 8.154 % official-protocol figure.
* **Remaining weakness:** dependence on the right image is not *geometric* use of
  it. We say so, and it is the thesis, not a concession we are forced into.

### Reviewer 3 — "Your intervention failures merely show poor experimental design, not an identification boundary."

* **Strongest form:** a sequence of halted and inconclusive experiments is more
  parsimoniously explained by inexperience than by a boundary.
* **Valid?** **Partially, and the honest answer says so.** Several failures *were*
  design failures: GEOM-001's rerun halted at HS1 with no admissible result; the
  epoch-10 stereo gate was invalid and had to be recalibrated from a control; the
  inverse-oracle diverged and produced a *false confirmation*; Zone-001's `S` was
  adopted before its collapse was noticed. Those are errors, are recorded as
  errors, and must not be dressed as findings.
* **Evidence available that the boundary is not merely our incompetence:**
  1. **Counterexamples are constructions, not failures** — the candidate-pointwise
     latch, A4s, A5r and M1 are explicit mechanisms; a design error cannot
     manufacture them.
  2. **`ZI-2 = 0.0` is architectural** — bit-exact, three seeds, every Δ. It
     holds for *any* zone-local design under a bounded receptive field,
     regardless of who designs it.
  3. **The null-impossibility argument is two lines of logic** about functions of
     fixed arguments, and does not reference our interventions at all.
  4. **A live positive control exists** — SHIFT-001's search-free model produces
     the signature, so the tests were capable of firing; they did, on the wrong
     hypothesis.
* **Response:** separate the two categories explicitly in §VII — design errors go
  to the supplement labelled as errors; only the structural results and the
  counterexamples carry the boundary claim.
* **Remaining weakness:** the boundary is *proved* for the enumerated families
  and *argued* to extend. A reviewer may reject the extension, in which case the
  contribution narrows from "an identification boundary" to "an identification
  boundary for translation-based and field-based interventions on fixed-left-image
  pairs". We state that fallback in the paper ourselves rather than defend the
  stronger form.

---

## 14. Paper-level claim ceiling

| Level | Headline statements |
|---|---|
| **A — directly established** | `max_k \|V_k − V_0\| = 0.0` · D1 8.154 % (official) · EXP-007 ablations +89.128…+89.957 pt on 1 837 304 px (four fully-tabulated modes; noise +89.806 in the same record) · invariance across 17 weight-sets · SHIFT-001's 0.6975-candidate response · gradient at 100 % while the right image hurts · `ZI-1`/`ZI-2` = 0.0 · identical weight SHA `3ad382c50d413ad2` · the 3×3 block table · TR-001 and ARCH-RATE values |
| **B — causally supported** | Readout conditioning alone converts a monocular trained model to a binocular one · candidate-varying content is material · the two factors interact · binocular function is destroyed from both ends · five blocks carry a reproducible penalty · **useful stereo prediction without candidate-varying correspondence information** |
| **C — strong, alternatives survive** | Training installs a calibrated candidate-axis response · the ~1/3 – 2/3 error split · stereo dependence is late-emerging · the identification boundary generalises beyond the enumerated families |
| **D — not demonstrated** | **Genuine local geometric correspondence, in any model in this project** · whether H2's binocularity is search or a non-search binocular readout · whether four blocks differ from six |
| **E — requires new data** | Any correspondence attribution (needs correspondence varied independently of content) · competitive accuracy (needs full-scale training) · transfer of any result |
| **F — impossible to claim** | "Does not perform correspondence" · "performs correspondence" · any p-value on an architecture comparison · latency/power claims for the from-scratch models |

---

## 15. Minimum writing plan

**Can the paper be written now, with no further experiment? YES.**

Every Level A and Level B row in §7 is already on file. The three items that
remain are **not** experiments:

1. **Protocol labelling** (§11) — table construction.
2. **A literature anchor** for what 8.154 % D1 means — a citation task.
3. **Editorial enforcement** of the §12 language lists — writing discipline.

**No fatal writing gap exists.** Two *optional strengthening* runs are recorded
in the post-closure audit (E4-cheap; a second seed of the `none`+standardised
factorial arm). Neither gates a claim; neither should be started before a draft
exists, and if a reviewer asks, the second is ~1 GPU-hour under the deterministic
protocol.

> **STOP EXPERIMENTATION AND WRITE.**

---

## 16. Final blueprint

| Item | Decision |
|---|---|
| **Research question** | Can stereo accuracy and geometric correspondence be experimentally separated in a cost-volume stereo network, and what can controlled interventions establish about which of the two a given network has? |
| **Thesis** | §2, in full. |
| **Contributions** | C1 candidate-invariant volume with strong binocular dependence · C2 the shift × readout interaction · C3 the identification boundary with explicit counterexamples · C4 four transferable methodological negatives. |
| **Title** | *A Candidate-Invariant Cost Volume in a Deployed Stereo Network: Separating Stereo Accuracy from Identifiable Geometric Correspondence.* |
| **Section order** | I Introduction · II Related Work · III Deployed Model and Problem Formulation · IV Methodology and Protocol Discipline · V Candidate-Invariant Volume with Binocular Dependence · VI Shift × Readout · VII Adversarial Identification Analysis · VIII Architectural Sensitivity · IX Discussion · X Limitations · XI Conclusion. |
| **Figure order** | F1 pipeline and the shift identity · F2 candidate invariance · F3 binocular dependence · F4 factorial · F5 emergence with the gradient proxy pinned · F6 destroyed from both ends · F7 identification ladder. |
| **Table order** | T1 protocol · T2 deployed model (two protocol blocks) · T3 factorial · T4 identification boundary · T5 block depth. |
| **Main vs supplement** | Main: the four pillars and the boundary. Supplement: the full audit trail, pre-registrations and hashes, halted and superseded records, per-seed and per-scene detail, determinism verification, design-only gates. |
| **Claim ceiling** | Level D for correspondence, stated in the abstract, §VII, §X and the conclusion. |
| **Claims allowed** | §12A. |
| **Claims prohibited** | §12C. |
| **Limitations** | One architecture; one dataset; 160-scene from-scratch training; n = 3 seeds and no computable p-values; single-seed factorial cells; inference-only readout and pruning studies; the generalisation step in C3; no literature accuracy comparator. |
| **Readiness** | **Ready to write.** |

**Level D — genuine geometric correspondence NOT-DEMONSTRATED.** Unchanged by
this plan, as required.

---

## 17. Verdict reconciliation (2026-09-14)

Writing only. No experiment, script, training, evaluation, checkpoint open,
or new measurement was performed. Every number below is quoted from an
existing repo file; anything not found is listed as UNVERIFIED.

**Checked against the verdict:** (a) `max_k |V_k − V_0| = 0.0`, upstream
PyTorch presence, permutation re-confirmation, non-export-artefact status —
confirmed in EXP-010/audit/`reference_shift`/INDEX-002/ARCH-001 citations, no
overstatement found; (b) official D1 8.154 % — confirmed, but its EPE pairing
was fixed (see below); (c) EXP-007 deltas +89.13…+89.96 — confirmed in
`results/ablation/right_image_ablation.json` (89.678/89.957/89.128/89.649),
fifth noise mode +89.806 now named; (d) EXP-007 baseline 9.3315 % on
1 837 304 px kept protocol-separated — confirmed and strengthened with scene
counts; no table mixes 8.154 % and 9.3315 % without labels; (e) 2×2 EPE
14.423/14.414/8.3717/2.2955 with materiality-only ceiling — numbers confirmed,
evaluation-set labelling fixed; (f) block depth 6-useful/5-penalty/
4-inconclusive, no necessity claim — confirmed, section ceiling narrowed;
(g) E2/E3 empirical sensitivity only — confirmed; (h) SAFE vs PROHIBITED
claim distinction — confirmed present in §2 and §12A; (i) no "does not use
correspondence" claim — confirmed absent, disclaimer present; (j) audit as
one adversarial methodology with main/supplement hierarchy — supplement now
names every required ID explicitly; (k) §§1–16 present and per brief.

**Changed:** thesis ablation clause now names the EXP-007 protocol and its
separate pixel set; C2 evidence and rows 8–9/T3/F4 now label plain cells as
H1-v2 10-scene late-window means (pre-determinism harness) and std cells as
deterministic 40-scene pooled; row 4 and §11 EPE fixed to official 1.3134 px
with fp32 CUDA 1.314/8.155 % stated separately; row 5 and Reviewer 2 now
include the noise mode (+89.806, corr −0.147) and the 10-scene scope;
row 14 dropped the unverified "t = 96" raw-curve clause; §4/VIII ceiling is
now B for 5-vs-6 EPE only and D for 4-vs-6; §10 supplement explicitly lists
INDEX-001/002/003, ARCH-001/ARCH-RATE, TR-001, GEOM-001/002, O1,
CP-001/HS-BAND, Zone-001 pre-freeze, S*, and the field gate; §11 table adds
scenes (40 vs 10), pixel sets, checkpoint identity, and the corrected EPE;
abstract plan and Level A now carry the same protocol/pixel-set labels.

**UNVERIFIED (not filled in):** the removed "raw curves non-monotone at
t = 96 for seeds 0 and 2" detail — no supporting line found in the
TR-001 records searched; the metric-aggregation rule (per-image mean vs
pooled) behind EXP-007's 9.3315 % baseline — not stated in the files found;
the EXP-007 checkpoint hash — not quoted in the files found (T1 requires it
only "where recorded"). Level D ceiling unchanged.
