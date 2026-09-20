# Post-closure research audit

**Date:** 2026-09-13. **Type:** synthesis audit. **Nothing was executed.** No
checkpoint opened, no training, no inference, no new measurement, no historical
record modified, no new correspondence intervention proposed. Every number below
is quoted from a record already on file and is cited to it.

**Claim ceiling, unchanged throughout:**
**Level D — genuine geometric correspondence NOT-DEMONSTRATED.**

**Question:** what is the strongest technically defensible scientific
contribution extractable from the completed campaign?

**Answer, stated once up front:** the campaign's central result is **not** a
failure to prove correspondence. It is a **measured separation** — a deployed,
production stereo network whose cost volume provably contains *zero*
candidate-varying information nevertheless depends on the second camera to the
tune of **+89 to +90 D1 points** and achieves **8.154 % D1** on the official
protocol. Around that anchor sits an **identification boundary**: eight
intervention families, each closed with an explicit correspondence-free
counterexample or a structural impossibility result. Those two things are the
paper.

**Final verdict: 1 — STOP EXPERIMENTATION AND WRITE THE PAPER.** Reasoning in
Phase 10.

---

# PHASE 1 — Evidence matrix

Column 8 uses three grades: **P** = publishable evidence (load-bearing in a
paper), **D** = diagnostic evidence (supports internal decisions, belongs in
supplementary), **G** = design evidence only (a record of a decision, no
measurement that bears on a claim).

| # | Record | Question | Intervention | Control | Result | What it actually identifies | Strongest surviving alternative | Ceiling contributed | Grade |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Phase-1 reproduction** (`phase-1-frozen`, `results/equivalence`) | Can the deployed Hailo StereoNet be independently reimplemented from its exported graph? | Second implementation written from the spec, weights mapped by execution order | ONNX Runtime on the reference graph, tapped at 5 stages | Agreement at every tap; MAC count 56.034 GMAC @368×1232 vs Phase 1's 56.04 | That the object of study is correctly characterised | none material | Enables everything else | **P** (methods) |
| 2 | **EXP-010** | Does the deployed cost volume vary across disparity candidates? | none — read the graph, then measure | — | `max_k \|V_k − V_0\| = **0.0**`, exactly, all 12 slices, 5 scenes. The shift is `pad right, slice [0:W]`, an **algebraic identity**; present in the upstream PyTorch source [SR-011], so it survives export and compilation | **The deployed model performs no disparity search.** Algebraic, not statistical | none — it is an identity | Anchor of the whole campaign | **P — central** |
| 3 | **EXP-007** (`results/ablation/right_image_ablation.json`) | Does the deployed model use the right image? | 5 right-image ablations (right:=left, right from another scene, horizontal flip, black, noise) | unmodified pair, 1 837 304 valid px, EPE 1.4422 / D1 9.3315 | D1 penalty **+89.13 … +89.96 points** across the four fully-tabulated modes; correlation with baseline falls to −0.31 … +0.35 | **The search-free model is nevertheless strongly binocular** | Right-image dependence ≠ correspondence; it is dependence on `Lf − Rf` at a *single* alignment | Makes the separation sharp | **P — central** |
| 4 | **H1 / H1-v2** | What is a genuinely shifted cost volume worth? | `cost_volume_shift` none → left, 200 epochs, seed 0 | matched arm, identical recipe | EPE 14.423 ± 0.228 vs 14.414 ± 0.227 (Δ −0.008 px); D1 81.397 ± 0.559 vs 80.756 ± 0.398 (Δ −0.64 pt, 10/11 checkpoints) | **Nothing about correspondence** — H2 later showed both arms ignored the second camera | The contrast was run on a pathway that could not learn | none | **D** |
| 5 | **H2** (`EXP-H2-SOFTARGMIN-SCALE`) | Is the matching pathway untrainable because the soft-argmin is saturated? | Per-pixel z-score across the 12 candidates immediately before the frozen soft-argmin. No new parameters (423 586 both arms) | H1-WORKING-v2, not retrained | Softmax entropy 0.0000 → 1.67–1.83 nats; matching-path gradient 5/150 crops → 16 000/16 000 batches; `r(disparity_initial, GT)` −0.001…+0.283 → **+0.978…+0.988**; right-image corruption ΔD1 ≤ +0.8 → **+81…+84 pt**; EPE 14.41 → **4.199 ± 0.090** | **A readout change converts a monocular model into a binocular one** with the volume, aggregation and refinement untouched | "Binocular" is not "correspondence" — the ablations show dependence, not search | Level C for the readout claim | **P** |
| 6 | **H3** (`EXP-H3-VIABILITY-001`) | With the readout fixed, how much does the shift contribute? | `shift` left → none under the standardised readout, 10 epochs | H2, matched init and schedule | Right-image ΔD1 **−1.08 … +1.91**; matching-map destruction **−1.14 … −0.07** (i.e. *better*); EPE 16.53 vs 10.54 | **Without candidate-varying volume content the trained model reverts to monocular** — under this readout and budget | 10 epochs only; long-budget behaviour UNKNOWN | Level C, jointly with #12 | **P** (as a gate; #12 is the publishable form) |
| 7 | **Seed replication** (`EXP-H2-SEED-REPLICATION-001`) | Is H2 seed-specific? | seeds 1, 2 | seed 0, plus a purpose-built 10-epoch seed-0 reference | REPRODUCIBLE: EPE 4.199 / 3.745 / 4.333; right-dependence min +83.95 / +76.78 / +74.88 pt. **And**: the epoch-10 gate was invalid — seed 0 fails it too (−3.33 pt) yet reaches +83.95 by epoch 200 | **Stereo dependence is a late-emerging property of this recipe**; and the harness was **not bit-reproducible** (same seed, epoch-0 loss 10.8798 vs 10.5762) | — | Bounds every seed claim in the project | **P** (methods) |
| 8 | **EXP-H2-EMERGENCE-001** | When does stereo dependence appear? | none — re-ablate saved snapshots | frozen 20-point criterion | Onset seed 1 ∈ (10, 20], seed 2 ∈ (20, 50]; saturates after epoch 150. Re-measurement reproduces training-time ablation to **0.00 pt** at all six pairs | **Matching-path gradient is not a proxy for stereo function** — 100 % of batches at *every* checkpoint, including one where corrupting the right image *improves* D1 by 0.91 pt | — | Kills a whole class of proxy gates | **P** (methods) |
| 9 | **E1 error partition** | Where is H2's residual error? | Oracle `disparity_initial` substitution at the matching→refinement boundary | three oracles bracketing capacity | Held-out fixed point removes **29.3 / 41.5 / 32.9 %** of pooled EPE — about **one third upstream, two thirds downstream** | An error *budget*, not a mechanism | Oracle behaviour ≠ learned mechanism | none | **D** |
| 10 | **E2 readout temperature** | Is the error sensitive to readout sharpness? | `softmax(−C/T)`, 9 temperatures × 3 seeds, inference only | T = 1.0 reproducing H2 exactly (1e-6 vs E1) | READOUT-ROBUST: **not one** of 24 cells improves. But sensitivity is steep and one-sided: T = 0.75 costs +3.5 px / +56 pt. **Stereo function collapses at T ≠ 1 in both directions** (T = 0.25 → entropy 0.290 nats, right-dependence +0.57 / −2.98 / −1.17) | **Binocular function is a property of the whole trained configuration**, destroyed from the readout end | Inference-only; a network adapts to the readout it was trained through | Level C, jointly with #11 | **P** |
| 11 | **E3 refinement ablation** | How much of refinement is load-bearing? | 13 inference-time ablations × 3 seeds | full stack | Best viable MAC saving **0.0 %**; cheapest removal +0.746 px / +11.40 pt for 14.9 % of MACs. **Below five blocks right-image dependence goes negative (−14 to −18 pt)**. Dropping block 4 costs seed 0 +0.19 px but seeds 1/2 **+24.16 / +16.37** | Refinement is **the path by which matching information reaches the output**; and *which* block is redundant is a property of the run, not the architecture | Lower bound only — weights were trained as a six-block stack | Level C, jointly with #10 | **P** |
| 12 | **Deterministic replication + shift factorial** | Is the harness decisive, and does candidate-varying volume content matter? | `use_deterministic_algorithms` + cuDNN determinism + `CUBLAS_WORKSPACE_CONFIG`; then the missing 2×2 cell | Stage B 6-block seed-0 reference, two runs, **identical weight SHA `3ad382c50d413ad2`** | left+std **EPE 2.2955 / D1 15.60** vs none+std **EPE 8.3717 / D1 66.60**. The 6.08 px gap is ≈ **59×** the later-measured 6-block three-seed EPE range (0.1027 px) | **Candidate-varying cost-volume content is material to the trained solution** under this recipe | One seed per arm; and materiality ≠ correspondence (see Phase 6) | Level B for materiality | **P — central** |
| 13 | **Block-depth campaign** (3×3) | Can refinement depth be reduced? | 4 / 5 / 6 blocks × seeds 0/1/2, 200 epochs, deterministic, pre-registered boundary | frozen 6-block three-seed EPE range | **5 blocks: reproducible penalty** (EPE ranges disjoint by +0.0748 px, 9/9 pairings; D1 overlaps by 0.13 pt). **4 blocks: INCONCLUSIVE** (straddles; 7/9; its own seed spread 0.3166 px = 3.08× the reference arm's) | An empirical accuracy effect in this configuration | Not a cause, not monotone, not a p-value | Level A/B, narrow | **P** (secondary) |
| 14 | **SHIFT-001 Stage A** | Does the degenerate model respond to right-image translation? | horizontal + vertical sweep on `shift=none`+std | the preregistered analytic prediction | Prediction **falsified**: `disparity_initial` 3.6452–4.2717, not 5.5; max deviation 1.8548 candidates; **max translation response 0.6975 candidates** | **A provably search-free model still produces input-dependent, translation-responsive disparity** | — | Kills translation-response as evidence | **P — central methodological** |
| 15 | **INDEX-001 / INDEX-002** | Is the trained model sensitive to candidate-axis *coordinates*? | candidate-axis permutations; 24 640 arms in 002 | algebraic negative control (`shift=none`: 6 160 arms, all **0.0**) | `CANDIDATE-COORDINATE-SENSITIVITY-NOT-DEMONSTRATED` — the **random** permutation reproduced the ordered pattern | The ordered/random contrast does not separate | — | C, not B | **D** |
| 16 | **INDEX-003** | Is an ordering test constructible? | none | — | `DESIGN-INCONCLUSIVE`, nothing executed | — | — | — | **G** |
| 17 | **ARCH-001 / ARCH-RATE-001** | Is the candidate-axis signature architecture-induced? | 16 random aggregation initialisations; then 384 untrained units | trained aggregation; algebraic null (816 arms, 0.0) | ARCH-001 `CASE-A-ARCHITECTURE-INDUCED`, INSIDE at 12/12. ARCH-RATE: **0 / 384** untrained aggregations qualify; **378/384 flat** | The *signature* is architecturally available; the *calibrated response* is not produced by untrained weights | — | Supplies TR-001's null | **P** (as TR-001's null) |
| 18 | **TR-001** | Do trained and random aggregations separate on the synthetic horizontal response? | none new — scored against the frozen 384-unit random population | 384 untrained units | `CLEAR-SEPARATION`: trained α_h median **+1.0014** (min +0.9534) vs random median +0.0032 (max +0.2441) — **12/12 outside**, smallest trained value 3.9× the largest random | **Training installs a calibrated, unit-gain candidate-axis response that untrained weights do not have** | The statistic is the translated-self-pair slope, a class later closed; a candidate-pointwise latch reproduces it | **Not** correspondence | **P**, with the ceiling stated |
| 19 | **GEOM-001** | Does the response survive a real-pair pairing control? | pairing interaction | — | `GEOMETRIC-CORRESPONDENCE-NOT-DEMONSTRATED`; rerun **halted at HS1**, `NO-ADMISSIBLE-RESULT` | nothing | — | C | **G/D** |
| 20 | **GEOM-002** | Can random candidate-axis statistics identify correspondence? | — | — | `SYNTHETIC-CORRESPONDENCE-NOT-DEMONSTRATED`; rerun **G-STOP**, `STATISTIC-ARCHITECTURALLY-AVAILABLE` (random aggregation scores h = 6) | The statistic is available without correspondence | — | C | **D** |
| 21 | **O1** | Candidate-axis permutation of the trained volume | — | — | Final blind audit: **C — NOT IDENTIFIABLE**, *do not run* | A candidate-pointwise latch reproduces the signature in a provably degenerate model | — | C | **G** |
| 22 | **CP-001 / HS-BAND** | Context propagation at the RF boundary | — | — | Substantively **FAILED**: C1 and C2 co-satisfiable only at **3–5e-06** of readout scale = 6–11 float32 ULPs | nothing | — | C | **G** |
| 23 | **Zone-001 pre-freeze** | Zone-local translation, `S = β_in − β_out` | none executed | — | **HALTED**. `ZI-1` and `ZI-2` both **0.0 bit-exact**, 3 seeds ⇒ `β_out ≡ 0` and `S ≡ β_global` restricted to the zone | The proposed statistic *is* the rejected global statistic | — | C | **P** (methodological) |
| 24 | **S\* identification audit** | Swap-referenced statistic | none executed | — | **REJECT**; **A4s** (scene-identity conditioning) and **A5r** (readout pinning) survive; plus a null-impossibility theorem | Two explicit correspondence-free accounts of the whole contrast | — | C | **P** (methodological) |
| 25 | **Spatial disparity-field gate** | Spatially varying δ | none executed | — | **CLOSE**. `D-y` is feasible and artefact-free and kills A3/A5/A9/M1, but A4s/A5r survive because the null obstruction is intervention-agnostic | The boundary generalises | — | C | **P** (methodological) |

---

# PHASE 2 — Positive, negative, non-identifiable

### Bucket A — evidence FOR a specific mechanistic claim

| Claim | Evidence | Strength |
|---|---|---|
| The deployed model's cost volume contains **no candidate-varying information** | EXP-010 (`0.0` exactly); algebraic identity in `reference_shift`; re-confirmed on a trained checkpoint and across 6 160 (INDEX-002) + 816 (ARCH-001) permutation arms, all `0.0` | **Proof-level** (algebra + measurement) |
| The deployed model nevertheless **depends heavily on the right image** | EXP-007: +89.13 … +89.96 D1 points across four ablation modes | **Strong, measured** |
| **A readout change alone** converts a monocular trained model into a binocular one | H2, replicated at 3 seeds; entropy 0.0 → 1.67–1.83 nats; right-dependence ≤ +0.8 → +74.9…+84.0 pt | **Strong, replicated** |
| **Candidate-varying volume content is material** to the trained solution | Shift factorial 2.2955 vs 8.3717 EPE, deterministic, matched; effect ≈ 59× the measured seed scale; corroborated directionally by H3 | **Strong** (n = 1 seed/arm, effect size carries it) |
| Binocular function is **a property of the whole configuration**, destroyed from both ends | E2 (T ≠ 1 in either direction) and E3 (< 5 blocks ⇒ right-dependence −14…−18 pt) | **Strong, 3 seeds** |
| Training installs a **calibrated unit-gain candidate-axis response** absent from untrained weights | TR-001 (12/12 outside, min +0.9534) against ARCH-RATE's 0/384 | **Strong** — but see the ceiling in Phase 3 |
| Five refinement blocks carry a **reproducible accuracy penalty** vs six | 3×3 deterministic; EPE ranges disjoint by +0.0748 px; 9/9 pairings | **Moderate, descriptive** |

### Bucket B — evidence AGAINST a specific mechanistic claim

| Claim refuted | Evidence |
|---|---|
| "A degenerate cost volume forces a uniform soft-argmin (5.5)" | SHIFT-001 Stage A: measured 3.6452–4.2717; max deviation 1.8548 candidates. **Falsified** |
| "Translation response demonstrates disparity search" | The provably search-free model responds up to **0.6975 candidates** |
| "Matching-path gradient indicates stereo function" | EMERGENCE-001: 100 % of batches at every checkpoint, including one where corrupting the right image *improves* D1 |
| "Refinement is a post-hoc polisher" | E3: truncating it destroys stereo dependence entirely |
| "Four refinement blocks match six" (the E3b-era reading) | Batch 3: 4b/seed0 was a favourable draw; the arm straddles |
| "Pruning a trained model can decide block count" | E3: block 4 costs seed 0 +0.19 px, seeds 1/2 +24.16/+16.37 |

### Bucket C — evidence that a claim is NOT IDENTIFIABLE

O1 · GEOM-001 · GEOM-002 · CP-001/HS-BAND · INDEX-001/002 ordering ·
Zone-001 `S` · `S*` · the spatially varying field family.

**The distinction that must be preserved in every sentence of the paper:**

> The campaign supports *"the available interventions cannot establish whether
> this network performs local geometric correspondence"*.
> It does **not** support *"this network does not perform correspondence"*.

Bucket C is never to be reported as Bucket B. Where a negative-sounding phrase is
unavoidable, the sentence must name the intervention family it is scoped to.

---

# PHASE 3 — The strongest positive result that survives every audit

**1. Which is strongest.** The **EXP-010 + EXP-007 pair**, read together: a
deployed production network with an algebraically search-free cost volume that
loses ~90 D1 points when its right image is disturbed. It is the only result in
the campaign that is simultaneously (i) proof-level rather than statistical,
(ii) about the *real deployed artefact* rather than a 160-scene from-scratch
model, (iii) untouched by every adversary the campaign found, and (iv)
surprising.

**2. What it supports.** That **useful stereo prediction can be produced from a
binocular signal that contains no disparity-candidate structure** — here, a
single-alignment difference `Lf − Rf` replicated 12 times. The second camera is
load-bearing; the *search* is absent.

**3. What it does not support.** It says nothing about whether the *other*
models in this project (H2 and its seeds, which have a genuine `shift="left"`
volume) perform correspondence. It is a statement about one model class, and it
is an existence result, not a universal one.

**4. Adversaries actually eliminated.** For this claim specifically: none needed
eliminating, because there is no inference from a signature to a mechanism — the
degeneracy is an identity and the dependence is a direct ablation. That is
exactly why it survives while everything inferential did not.

**5. Still observational rather than mechanistic.**

* TR-001's separation — it is a *trained-vs-untrained* difference on a statistic
  whose mechanistic reading is closed. Report the separation; do not read it as
  matching.
* E1's one-third/two-thirds error split — an accounting, not a mechanism.
* The block-depth results — accuracy, not cause.
* H2's `r(disparity_initial, GT) = +0.978…+0.988` — a correlation.

**Explicitly not inflated, per the brief:** gradients are not correspondence
(EMERGENCE-001 proves the point empirically); softmax sharpness is not matching
(E2: the sharpest setting *destroys* stereo function); EPE/D1 are not mechanism;
global translation response is not geometric search (SHIFT-001: 0.6975 candidates
from a search-free model); pruning sensitivity is not theoretical necessity (E3);
oracle behaviour is not a learned mechanism (E1).

---

# PHASE 4 — Candidate narratives, scored

Scores are 1–5. "Sufficient?" asks whether the existing records support the
narrative with no further experiment.

| # | Narrative | Evidence | Novelty | Falsifiable | Reproducible | Adversarially robust | Claim:evidence | IEEE fit | Sufficient? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | StereoNet **performs** genuine correspondence | 1 | 3 | 4 | 3 | **1** | **1** | 2 | **No** |
| 2 | StereoNet **does not perform** correspondence | **1** | 3 | 4 | 3 | **1** | **1** | 2 | **No** — converts C into B; prohibited |
| 3 | Observed stereo behaviour **cannot be mechanistically attributed** to correspondence, because correspondence-free mechanisms are observationally equivalent | 5 | 4 | 4 | 5 | **5** | 5 | 4 | **Yes** |
| 4 | A **degenerate/search-free** cost volume retains useful stereo accuracy, separating performance from identifiable correspondence | **5** | **5** | 4 | **5** | **5** | 5 | **5** | **Yes** |
| 5 | An adversarial audit reveals an **identification boundary** for correspondence claims in cost-volume stereo | 5 | 4 | 3 | 5 | 5 | 4 | 4 | **Yes** |

**Chosen: 4 as the empirical thesis, with 3/5 as the methodological
contribution.** Not the most exciting — 1 would be — but the only pairing where
every claim is backed by an identity or a direct ablation, and where the
adversarial work becomes an asset rather than an apology. Narrative 4 alone is
one strong result; 3/5 alone is a negative methods paper; together they are a
complete argument: *here is a system where performance and identifiable
correspondence come apart, and here is why the standard tests cannot tell you
which one you have.*

---

# PHASE 5 — Is the degenerate checkpoint the central result?

**Yes.** Audited carefully, because the temptation is to overstate it.

### What is established

| Component | Status | Evidence |
|---|---|---|
| The volume is candidate-invariant | **Algebraic identity**, plus measurement | `reference_shift` pads right and slices `[0:W]`; `max_k \|V_k − V_0\| = 0.0` (EXP-010, 5 scenes); 6 160 arms at 0.0 (INDEX-002); 816 arms at 0.0 across 17 weight-sets incl. 16 random (ARCH-001) |
| It is not our bug | **Confirmed** | Present in the upstream PyTorch source [SR-011]; carried through export and compilation |
| Accuracy | **Measured** | **D1 8.154 %**, official protocol (`PHASE_2_BASELINE.md`); fp32 CUDA 1.314 px EPE / 8.155 % D1 / 90.1 ms |
| The right image is load-bearing | **Measured** | EXP-007: +89.13 … +89.96 D1 points over four ablation modes; baseline EPE 1.4422 / D1 9.3315 on 1 837 304 valid px |
| Independent reimplementation agrees | **Measured** | Phase-1 equivalence at five tapped stages |

### The two statements, kept apart

* ❌ *"The network does not use correspondence."* **Not supported, and not
  claimable.** The network consumes `Lf − Rf`, a binocular quantity, and depends
  on it heavily. What it cannot do is *search* — compare candidates. Absence of
  search is not absence of all geometric use.
* ✅ *"Useful stereo prediction can be achieved without candidate-varying
  correspondence information in the cost volume."* **Supported**, by an identity
  plus two direct measurements. This is the safer *and* the stronger claim,
  exactly as the brief anticipated.

### The exact comparison needed for rigour

1. **Degeneracy** — state it as algebra first, measurement second. Done.
2. **Accuracy on a named protocol** — 8.154 % D1, official protocol,
   `hailo_val`. Done. **Bookkeeping item:** EXP-007's baseline (9.3315 % D1,
   1 837 304 px) is a *different pixel set* from the official-protocol figure.
   The paper must quote one protocol per table and never mix them. This is an
   editing requirement, not an experiment.
3. **Binocular dependence** — five ablation modes. Done.
4. **A contrast showing the volume's candidate structure is not worthless** —
   otherwise a reader concludes candidate structure never matters. Supplied by
   the shift factorial (2.2955 vs 8.3717 EPE) and H3. Done.

Nothing in this list requires new work.

---

# PHASE 6 — What the shift factorial identifies

The 2×2 is complete, deterministic, seed-0, 200 epochs, 40-scene evaluation:

| | plain readout | standardised readout |
|---|---:|---:|
| `shift="none"` | EPE 14.423 (H1-BASE-v2) | EPE **8.3717** / D1 66.60 |
| `shift="left"` | EPE 14.414 (H1-WORKING-v2) | EPE **2.2955** / D1 15.60 |

**It identifies an interaction.** Neither factor alone produces a binocular
model: the shift without the readout fix is worth −0.008 px (H1-v2), and the
readout fix without the shift leaves EPE at 8.37. Only the combination reaches
2.30. Stated as a claim:

> ✅ **Candidate-varying cost-volume geometry contributes materially to the
> trained solution under this recipe — but only once the readout permits the
> matching pathway to learn.**

**It does not identify how that content is used.** The claim

> ❌ *"candidate-varying geometry is implemented as genuine correspondence"*

is **not** supported, and the campaign has explicit counterexamples: a
candidate-pointwise latch reproduces the ordered candidate-axis signature
(O1 audit), the statistic is architecturally available without correspondence
(GEOM-002 G-STOP), and a search-free model still responds to translation by
0.6975 candidates (SHIFT-001). Materiality is about *information content*;
correspondence is about *mechanism*. The factorial measures the first.

**Scope limits to carry with the number:** one seed per arm (defensible only
because 6.08 px is ≈ 59× the measured 6-block three-seed range of 0.1027 px);
160 scenes from random initialisation; one candidate count; one dataset; this
readout.

---

# PHASE 7 — What block depth supports

| Statement | Supported? | Basis |
|---|---|---|
| Six blocks are **empirically useful** in this configuration | **Yes** | 3×3 deterministic design; 6b arm has the tightest spread (0.1027 px) and the best mean |
| Five blocks carry a **reproducible penalty** | **Yes, with one stated exception** | EPE ranges disjoint by +0.0748 px, 9/9 pairings; **D1 ranges overlap by 0.13 pt** (5b/seed1 beats 6b/seed1) — always quote the exception |
| Four blocks are **inconclusive** | **Yes** | Arm straddles the frozen boundary; 7/9 pairings; its own seed spread 0.3166 px = 3.08× the reference arm's and larger than the 0.1603 px gap it would need to show |
| All six blocks are **theoretically necessary** | **NO** | No causal claim is supportable from this design. Not monotone (4b beats 5b in only 5/9). Not a parameter-count or receptive-field cause. Not a p-value (n = 3, none computable). Not a latency claim. Not transferable |

Supporting finding worth reporting alongside: **E3 showed that which block is
redundant is a property of the run, not the architecture** — the single most
useful methodological warning in this part of the campaign.

---

# PHASE 8 — Should the paper change its question?

**Yes — and the evidence, not convenience, forces it.**

The original question, *"Does StereoNet perform genuine geometric
correspondence?"*, is now known to be **not answerable with the available
interventions and data**. That is itself a result, but a paper cannot be
organised around a question it proves unanswerable without also offering what it
*can* answer.

**Revised question, adopted:**

> **Can stereo accuracy and identifiable geometric correspondence be separated in
> a cost-volume network — and what can controlled interventions actually
> establish about the source of the behaviour?**

The first clause is answered affirmatively and constructively by EXP-010 +
EXP-007. The second is answered by the identification boundary. The original
question survives as a *sub*-question whose negative answer is one of the
paper's results, stated at the correct ceiling.

---

# PHASE 9 — Claim ladder

### LEVEL A — directly reproduced / measured

* The deployed cost volume is candidate-invariant: `max_k |V_k − V_0| = 0.0`.
* Deployed reference accuracy: D1 8.154 % (official protocol); 1.314 px EPE /
  90.1 ms fp32 CUDA.
* Right-image ablation on the reference: +89.13 … +89.96 D1 points, five modes.
* Independent reimplementation agrees with the exported graph at five taps;
  56.034 GMAC @368×1232.
* H2 readout statistics: entropy 0.0000 → 1.67–1.83 nats; gradient 5/150 →
  16 000/16 000; `r(GT)` +0.978…+0.988.
* Stage B determinism: two runs, identical weight SHA `3ad382c50d413ad2`;
  reference EPE 2.2955182 / D1 15.6045143.
* The 3×3 block-count table, all nine runs, same 3 802 797 valid pixels.
* TR-001 α_h values; ARCH-RATE 0/384.
* `ZI-1` / `ZI-2` = 0.0; HS-BAND's 3–5e-06 co-satisfaction band.

### LEVEL B — causally supported

* A per-pixel z-score before the soft-argmin converts a monocular trained model
  into a binocular one, all else fixed (H2, 3 seeds).
* Candidate-varying volume content is material to the trained solution, given the
  fixed readout (shift factorial; H3).
* Truncating refinement below five blocks destroys right-image dependence (E3).
* Moving readout temperature off 1.0 in either direction destroys it (E2).
* Five refinement blocks carry a reproducible EPE penalty vs six (3×3).

### LEVEL C — strongly supported, alternatives survive

* Training installs a calibrated unit-gain candidate-axis response that untrained
  aggregations do not produce (TR-001 vs ARCH-RATE) — **alternative:** a
  candidate-pointwise latch reproduces the signature without correspondence.
* About one third of H2's residual error is attributable upstream, two thirds
  downstream (E1) — **alternative:** the split is oracle-capacity-dependent.
* Stereo dependence is a late-emerging property of the recipe (EMERGENCE-001,
  2 seeds + 2 seed-0 points).

### LEVEL D — not demonstrated

* **Genuine local geometric correspondence in any model in this project.**
  Ceiling unchanged. "Not demonstrated" ≠ "absent".
* Whether H2's binocular dependence is search or a non-search binocular readout.
* Whether the four-block arm differs from six.
* Any cause for the block-count effect.

### LEVEL E — would require new data

* Any correspondence attribution: needs data where correspondence varies
  **independently of content** — multi-view / multi-baseline capture, or a
  renderer with ground-truth control (and its own conceded ceiling).
* Competitive accuracy claims: needs a full-scale training set (the from-scratch
  models use 160 scenes).
* Transfer of any block-count or shift result to another dataset or recipe.

### LEVEL F — currently impossible to claim

* "The network does not perform correspondence."
* "The network performs correspondence."
* Any p-value on any architecture comparison in this project (n = 3, none
  computable).
* Any latency, power or deployment claim about the from-scratch models.

---

# PHASE 10 — Minimum remaining work

### A — REQUIRED before writing

**None.**

Every load-bearing claim in the chosen narrative is already on file:
degeneracy (identity + EXP-010), accuracy (8.154 %, official protocol),
binocular dependence (EXP-007, five modes), the materiality contrast (shift
factorial + H3), the configuration-wide fragility (E2 + E3), the replication
(3 seeds), the determinism (Stage B), and the identification boundary (eight
closed families with named counterexamples).

The one open item — reconciling EXP-007's 9.3315 % baseline with the official
protocol's 8.154 % — is a **table-construction requirement**, not an experiment:
quote one protocol per table and label the pixel set.

> **STOP EXPERIMENTATION AND WRITE THE PAPER.**

### B — HIGH-VALUE but optional

1. **E4-cheap** — alternative volume representations on H2's trained features,
   inference only, minutes. The only remaining live Stage E rationale; a negative
   result would close Stage E on evidence rather than on absence. Strengthens the
   discussion; no claim depends on it.
2. **A second seed of the `none`+standardised factorial arm.** The 6.08 px effect
   is ≈ 59× the seed scale, so n = 1 is defensible — but a reviewer will ask, and
   the run is ~1 GPU-hour under the deterministic protocol.

### C — interesting but unnecessary

E2b (trained/learnable temperature) · O6 (Scene Flow pretraining, ~22 h, blocked
on dataset size) · latency and quantisation profiling of the from-scratch models
· the decisive form of E1 (retrain refinement against a strong matching input).

### D — scientifically closed, do not pursue

Every correspondence-identification variant: O1 and candidate permutations ·
translated self-pairs · GEOM-001 / GEOM-002 · CP-001 and RF-boundary searches ·
zone translations and `S` · swap controls and `S*` · spatially varying disparity
fields · global translation as primary evidence · direct edits to `V` as pipeline
evidence · any new paired null over fixed-`L` interventions. Reopening requires
**new data**, not a new statistic.

---

# PHASE 11 — Paper-ready contribution

### 1. Thesis

> In a deployed cost-volume stereo network, **stereo performance and identifiable
> geometric correspondence come apart**. The shipped model's cost volume is
> provably free of candidate-varying information, yet it is strongly binocular
> and reaches 8.15 % D1; and across eight intervention families we find that the
> standard signatures used to attribute correspondence — translation response,
> candidate-axis structure, gradient flow, softmax sharpness, accuracy — each
> admit explicit correspondence-free counterexamples, several of them
> structurally non-identifiable.

### 2. Contributions

1. **A proof-level degeneracy in a shipped model**, traced from the exported
   graph to the upstream source, with an independent reimplementation
   confirming it — and the measurement that the same model is nevertheless
   ~90 D1 points dependent on its right image.
2. **A controlled 2×2 showing the interaction** between cost-volume candidate
   structure and readout conditioning: neither alone yields a binocular model;
   together they take EPE from 14.4 to 2.30, under a bit-reproducible
   deterministic protocol.
3. **An identification boundary with explicit counterexamples** — the
   candidate-pointwise latch, `A4s` (scene-identity conditioning), `A5r` (readout
   pinning), `M1` (intra-image deformation reading) — plus two structural results:
   the `ZI-2` collapse (a zone-local statistic *is* the global one) and the
   null-impossibility theorem (for fixed `L`, no null is both content-matched in
   the tested window and geometrically non-corresponding).
4. **Methodological negatives that generalise**: gradient flow is not a proxy for
   stereo function; pruning a trained model cannot decide capacity; a diverging
   probe can produce a false confirmation; and a project-wide MEASURED / DERIVED /
   INFERRED / UNKNOWN discipline with every failed run preserved.

### 3. Claims that can be made

* The deployed model's cost volume is candidate-invariant — an algebraic identity,
  confirmed bit-exactly.
* It achieves 8.154 % D1 (official protocol) and depends on the right image by
  +89 to +90 D1 points.
* **Useful stereo prediction is achievable without candidate-varying
  correspondence information in the cost volume.**
* A readout change alone converts a monocular trained model into a binocular one,
  replicated across three seeds.
* Candidate-varying volume content is material to the trained solution under this
  recipe, by a margin ≈ 59× the measured seed scale.
* Binocular function is a property of the whole trained configuration: it is
  destroyed by truncating refinement below five blocks and by moving readout
  temperature off 1.0 in either direction.
* Five refinement blocks carry a reproducible EPE penalty versus six, at three
  seeds, with the D1 exception stated.
* Several standard correspondence signatures have explicit correspondence-free
  counterexamples, and several intervention families are structurally
  non-identifiable.

### 4. Claims that must NOT be made

* "StereoNet does not perform correspondence" — Bucket C is not Bucket B.
* "StereoNet performs correspondence."
* "The model ignores the second camera" — flatly contradicted by EXP-007.
* Any p-value, significance test or "statistically significant" phrasing.
* "Four blocks match six", or any monotonicity or causal claim in block count.
* Competitive-accuracy or latency claims for the from-scratch models
  (160 scenes, random initialisation).
* Any transfer to another dataset, recipe or candidate count.
* Reading TR-001's separation as evidence of matching.
* Reading gradient flow, softmax entropy, oracle behaviour, pruning sensitivity
  or EPE/D1 as mechanism.

### 5. Strongest figures and tables

| # | Content | Why |
|---|---|---|
| F1 | The degenerate-shift construction (pad-right → slice `[0:W]`) beside `max_k \|V_k − V_0\| = 0.0` | The paper's anchor, in one picture |
| T1 | Deployed reference: accuracy (official protocol) and the five right-image ablations | The separation, in one table |
| T2 | The 2×2 shift × readout factorial, four cells, deterministic | The interaction |
| F2 | Stereo-dependence emergence vs epoch, seeds 1 and 2, with matching-path gradient pinned at 100 % throughout | Shows the proxy failing, visually |
| F3 | Stereo dependence vs readout temperature (E2) and vs refinement depth (E3), same axis | "Destroyed from both ends" |
| T3 | The identification-boundary table: family → statistic → counterexample → verdict | The methodological contribution |
| T4 | The 3×3 block-count design with the frozen decision boundary drawn on it | Honest inconclusiveness, pre-registered |

### 6. Supplementary material

H1 / H1-v2 in full · E1's three oracles and the inverse-oracle failure · E2's
full 9 × 3 grid · E3's 13 ablations · the seed screen and the invalid epoch-10
gate with its correction · the determinism diagnostic · INDEX-001/002,
ARCH-001, ARCH-RATE, TR-001 in full · the design gates for GEOM-001/002, O1,
CP-001, Zone-001, `S*`, the field gate.

### 7. Omit

The superseded and invalid records — the inverse-oracle run (never cite), the
aborted seed-1/2 epoch-10 runs, `EXP-H2-EMERGENCE-001` (reader bug), the
superseded Stage B attempt, the E3 run that crashed on encoding, INDEX-003 and
the other design-only records that produced no measurement. All stay on disk with
their `NOTE.md`s; none belongs in the paper, including supplementary.

### 8. Remaining evidence gaps

1. **Protocol reconciliation** between EXP-007's 9.3315 % baseline and the
   official protocol's 8.154 % — labelling, not measurement.
2. **n = 1 seed** on the factorial's `none`+standardised arm.
3. **No comparator** for what 8.15 % D1 means against contemporary stereo
   networks — a literature anchor, not an experiment.
4. **H2's mechanism is unattributed**, and by the campaign's own findings cannot
   be attributed with available data. This must be stated as a limitation, in
   those words.
5. **All from-scratch results are 160-scene**, random initialisation.

### 9. Readiness

**Ready.** The empirical anchor is proof-level, the supporting contrast is
deterministic and replicated, the negatives are pre-registered, and the
identification boundary is documented with explicit counterexamples rather than
asserted. The remaining gaps are labelling, citation and one optional seed —
none of them blocking.

---

# Final verdict

> ## 1 — STOP AND WRITE.

No further experiment is required to support the strongest defensible narrative.
The two optional runs in Phase 10-B may be done *during* writing if a reviewer
question makes them worth an hour; neither gates a claim. No correspondence
intervention is proposed, and none should be, until data arrives in which
correspondence can be varied independently of content.

**Level D — genuine geometric correspondence NOT-DEMONSTRATED.** Unchanged by
this audit, as required.
