# Post-Identifiability Research Boundary — Stage B ARM-P

Zero-compute research-boundary analysis document. No GPU was used. No training
was run, no evaluation was run, no GWC was implemented, no model was modified,
no new seed was created, and no experiment was prepared, authorized, or
launched by this document. No existing file was modified for this document.
Exactly one file was created: this one,
`stage_b_armp/POST_IDENTIFIABILITY_BOUNDARY.md`.
ARM-P is an INITIALIZATION INTERVENTION throughout. CONTROL means RANDOM
INITIALIZATION under the original P2A recipe — never "P2A checkpoint
initialization". No Phase 3 is invented or proposed. The purpose of this
document is to map what can legitimately be studied AFTER relaxing the frozen
P2A single-component constraint, and to state the scientific cost of that
relaxation. It does not choose a model, rank designs, or declare a winner.

---

## 1. CURRENT CLOSED STATE

Phases 0–2 are CLOSED. No Phase 3 exists; none is created here. Stage A is
COMPLETE with the architecture gate "NO ARCHITECTURE JUSTIFIED" (Fact I).

Stage B ARM-P replication is COMPLETE: three predeclared seeds (seed 0 local,
seed 1 local, seed 2 Kaggle-trained with local frozen evaluation) plus a
paired seed-1 Kaggle environment-confound control, all scored under the frozen
evaluation contract and recorded in `stage_b_armp/ARMP_CLOSURE_RECORD.md`
Sections 4–7 (Fact J). The finding is a repeated directional observation —
ARM-P lower than random-init CONTROL on best and final checkpoints in every
seed-by-selection cell — reported as an initialization intervention only, not
a mechanism proof (Fact J; Closure Record Sections 11–12).

Candidate C (supervision coverage vs pretraining as the explanation of the
ARM-P delta) is NOT IDENTIFIABLE on the frozen P2A platform, per
`stage_b_armp/CANDIDATE_C_SEPARABILITY.md` Sections 4 and 9 (Fact A).
Frozen-platform True GWC is NOT IDENTIFIABLE, per
`stage_b_armp/GWC_IDENTIFIABILITY_DEFINITION.md` Sections 4–6 and 11
(Fact B).

No experiment is authorized. There is no authorized training, no authorized
implementation, no authorized seed, and no authorized follow-up measurement
beyond the single zero-compute action in Section 13.

---

## 2. WHAT THE NEGATIVE IDENTIFIABILITY RESULTS MEAN

These negative results are findings about what CAN be measured — about the
structure of the platform — not failures to measure. Each one forecloses a
class of causal claim while leaving adjacent questions open.

**Candidate C NOT IDENTIFIABLE (Fact A).** What it forecloses: no
matched-coverage experiment on frozen P2A can attribute the ARM-P delta to
supervision coverage rather than to initialization, because high-disparity
supervision exposure is tied to the scale-sampling mechanism
(`finetune_pilot.py` `ScaledCroppedKitti`: `disp_c[disp_c > 0] *= sx`), and
FT3D pretraining packages coverage into initialization. Every additive route
moves a second variable — augmentation/scale sampling, dataset/domain and
curriculum, the loss function, or initialization itself — and the one route
that holds everything fixed (subtractive loss masking) moves coverage in the
wrong direction. What it leaves open: coverage remains a live DATA-side
hypothesis (Stage A recorded ~2x under-coverage at GT>=96 as a data fix, not
an architecture change); it is simply not separable from initialization on
this frozen platform. ARM-P stands as an initialization-bundle observation
whose internal decomposition (coverage content vs other pretraining content)
is unmeasured, not as a refuted observation.

**Frozen-platform True GWC NOT IDENTIFIABLE (Fact B).** What it forecloses:
no single-component True-GWC experiment exists on frozen P2A, because P2A's
native matching operation is a SIGNED FEATURE DIFFERENCE, not a correlation
(`src/models/stereonet/cost_volume.py:110-111`, `method "subtract"`, volume
`(B, 32, 24, H, W)`). Introducing True GWC changes the OPERATION TYPE
(difference → inner product, Fact C), and for every non-degenerate G it also
changes aggregation input width and parameter count (Fact D), with parameter
matching requiring an additional intervention (Fact E). The degenerate
G = C = 32 case preserves width but is an elementwise product with group
width 1 and no intra-group reduction, so it does not test the grouping
hypothesis (Fact F). What it leaves open: the grouping hypothesis itself is
UNTESTED, not refuted. ARM-W is mechanistically non-equivalent (Fact G) and
therefore silent on it. A multi-component GWC question can still be DEFINED —
it simply cannot be answered with single-component causal force on the frozen
platform.

**ARM-W outcome (Fact G).** `phase1/scripts/train_arm_w.py:1-14` and
`src/models/stereonet/cost_volume.py:160-171` show ARM-W applied
`nn.Conv2d(32, 8, 1, groups=8, bias=False)` to the 32-channel DIFFERENCE map:
a learned grouped 1x1 channel compression of a difference volume, with no
left/right feature product anywhere in it. ARM-W scored 1.7776 against
incumbent ARM-V 1.7727 (margin 0.0049 px), which sits INSIDE the 0.0444 px
historical run-to-run spread. That spread is CONTEXT ONLY and is never a
significance threshold in either direction. "Refuted" was the preregistered
decision-rule outcome for ARM-W's own compression proposition, not a
significance claim and not evidence about True GWC.

**Sign-convention consequence (Fact H).** A difference is a COST while a
correlation is a SIMILARITY, so any correlation-based design must handle the
readout's `softmax(-cost)` sign convention. Sign handling is a named
component of every such design, not a detail.

---

## 3. FROZEN-CONSTRAINT BOUNDARY

The constraints that currently bind, and which negative result each one
blocks:

1. **Architecture frozen** (P2A `StereoNet` graph and constants: 397,954
   params, 70 keys, 24 disparity candidates, 3 downsample levels, shift
   right, regression_normalize true — Fact K). Blocks any GWC construction,
   since every True-GWC construction edits the forward graph. Also blocks
   architecture sweeps generally (Stage A gate, Fact I).
2. **Matching operation fixed** (signed difference,
   `src/models/stereonet/cost_volume.py:110-111`). This is the primary
   blocker of frozen-platform True GWC (Facts B, C): the native axis is
   "difference → grouped inner product", an operation-type replacement, not
   a group-count turn.
3. **Aggregation input width fixed** (first `Conv3d` consumes 32 channels;
   `src/models/stereonet/aggregation.py:26-40`; wiring in
   `src/models/stereonet/stereonet.py:108-119`). Co-blocks True GWC for
   every G ≠ 32 (Fact D): width 32 → G moves parameters, FLOPs, memory,
   init shape, and optimization dynamics together with the operation swap.
4. **Parameter count fixed** (frozen 397,954 total; eval-time guard pattern
   per Closure Record Section 8). Co-blocks True GWC (Facts D, E):
   accepting the deficit leaves a capacity confound; removing it via an
   adapter adds a second intervention.
5. **Initialization shape fixed** (strict loading; ARM-P precedent rests on
   byte-identical shapes). Co-blocks True GWC for every G ≠ 32: the first
   `Conv3d` weight changes shape `[32, 32, 3, 3, 3]` → `[32, G, 3, 3, 3]`,
   so no identical-init control exists without init surgery that is itself
   an intervention.
6. **Readout sign convention fixed** (`softmax(-cost)`, i.e. an argmin over
   costs). Co-blocks every correlation-based design (Fact H): a similarity
   ported without negation inverts the matching evidence, and the negation
   or readout flip is a second graph edit.
7. **Recipe frozen** (dataset, augmentation/scale sampler, loss call,
   optimizer, scheduler, epochs, seeds). This is what blocks Candidate C
   (Fact A): on the KITTI path, coverage IS the scale-sampler output
   (`disp_c[disp_c > 0] *= sx`), so coverage cannot move while the recipe
   holds; and at pretraining time, coverage is absorbed into the weight
   point, so varying it moves initialization itself. The recipe does NOT
   block True GWC — the recipe can be held byte-identical under a GWC swap;
   the GWC failure lies in the graph change itself.
8. **Dataset frozen** (KITTI `hailo_calib` 160 train / `hailo_val` 40 eval;
   FT3D A+C-only subset with B missing per Closure Record Sections 13a–13b).
   Co-blocks Candidate C additive routes (joint/mixed training changes
   domain and curriculum) and limits any pretraining-corpus variant.
9. **Evaluation contract frozen** (40 scenes, 3,802,797 valid pixels,
   gt_scale 256.0, gt_source disp_occ_0 — Fact K). Blocks nothing
   scientifically on its own; it is the constant measurement layer that
   must stay fixed so any future numbers remain comparable. Any relaxation
   here destroys comparability with every closed result.

---

## 4. POSSIBLE RELAXATIONS

For each freezable constraint: whether relaxing it is NECESSARY, OPTIONAL, or
SCIENTIFICALLY DANGEROUS for a post-boundary GWC question. Relaxing one
constraint never preserves identifiability of the rest — each relaxation's
cost is stated explicitly.

- **Matching operation (difference → grouped inner product): NECESSARY.**
  Without this relaxation no GWC-family design exists at all. Cost: the
  operation swap bundles interaction change (subtractive vs multiplicative)
  with reduction change (per-channel identity vs pooling over C/G channels);
  it forces sign handling and changes what aggregation filters learn
  (similarity peaks vs cost minima). Nothing about the baseline's causal
  structure survives this edit.
- **Aggregation input width (32 → G): NECESSARY for every non-degenerate G.**
  A `(B, G, 24, H, W)` volume must enter aggregation for G to be "preserved
  into aggregation" (the load-bearing U2 property). Cost: capacity change
  (−20,736 first-layer weights at G=8), FLOP/memory change, init-shape
  change, and fan-in-dependent init-distribution and trajectory changes.
  There is no width relaxation without these travelling companions.
- **Parameter count (accept deficit or add adapter): NECESSARY to decide,
  either way confounded.** Accepting the deficit (OPTIONAL in the sense that
  it is the default) leaves the capacity confound uncontrolled. Adding a
  shape adapter or widening elsewhere to "match" parameters is
  SCIENTIFICALLY DANGEROUS if presented as a control: matching a ledger is
  a second architectural intervention, not a causal license. Either choice
  must be declared as a moved component.
- **Readout sign convention (negate similarity or flip readout):
  NECESSARY.** An un-negated similarity is incorrect, not a control. Cost:
  a second graph edit with its own justification burden; outcomes can never
  be attributed to grouping alone while this edit is present.
- **Initialization comparability: NECESSARY to abandon as a goal.** Same
  seed at different fan-in yields different effective init distributions;
  any shape conversion (truncate, average, tile, re-init) is researcher
  surgery. Cost: optimization-trajectory confounds become permanent
  residents of every relaxed design.
- **Normalization choice (unnormalized vs per-group L2 with eps 1e-05):
  NECESSARY to freeze by fiat.** Both public precedents exist; neither is
  derivable from the baseline. Cost: one unconstrained degree of freedom
  that must be preregistered, never swept, in any future design.
- **Recipe (data, loss, optimizer, schedule, seeds): SCIENTIFICALLY
  DANGEROUS to relax alongside the graph.** The recipe is the one thing
  that CAN be held fixed under a GWC swap; relaxing it (epochs, LR, loss
  reweighting, augmentation range) adds third and fourth moving components
  and destroys the only remaining anchor. Hold it.
- **Dataset/corpus (joint FT3D+KITTI, altered pretraining corpus):
  SCIENTIFICALLY DANGEROUS for a GWC question.** It re-opens the Candidate C
  confounds (domain, curriculum, coverage-into-init) on top of the GWC
  bundle. A GWC question must not also be a data question.
- **Evaluation contract: SCIENTIFICALLY DANGEROUS to relax.** Changing
  scenes, pixels, gt_scale, or gt_source breaks comparability with ARM-V,
  ARM-P, and every closed audit number. Hold it unconditionally.
- **Architecture gate ("NO ARCHITECTURE JUSTIFIED"): NECESSARY to reopen
  explicitly IF any relaxed work proceeds.** The gate is a Stage-A verdict
  on the frozen platform, not a law of nature. Reopening it is legitimate
  only as a declared new multi-component research question (Section 12,
  option B), never as a quiet continuation of the frozen program.

---

## 5. POSSIBLE TRUE-GWC DESIGNS

Baseline for every design (B0): P2A as frozen — `(B, 32, H, W)` siamese
features at stride 8; `build_cost_volume` with `method="subtract"`,
`shift="right"` (`src/models/stereonet/cost_volume.py:72-119`), volume
`(B, 32, 24, H, W)`; aggregation first `Conv3d(32→32, 3×3×3)` per
`src/models/stereonet/aggregation.py:26-40`; readout `softmax(-cost)`;
397,954 params; random initialization under the original P2A recipe as
CONTROL; frozen evaluation contract (Fact K). First-layer arithmetic
baseline: `32 × 32 × 27 = 27,648` weights (+32 bias = 27,680).

No ranking. No scoring. No winner. Neutral labels only.

**D1 — Unnormalized True GWC, G=8, G preserved into aggregation.**
Intervention: replace per-candidate `anchor − other` with per-group mean of
elementwise products over groups of width 4, same `shift_right` indexing.
Changed operation: signed difference → per-group inner product (mean over
C/G channels). Changed tensor shapes: volume `(B, 32, 24, H, W)` →
`(B, 8, 24, H, W)`. Changed parameter count: first-layer weights
`8 × 32 × 27 = 6,912` (+32 bias); delta −20,736 weights (~5.2% of 397,954);
unmatched total ≈ 377,218. Changed initialization: first-`Conv3d` weight
`[32, 8, 3, 3, 3]`, incomparable shape with baseline, fan-in 4× smaller so
same seed yields larger-magnitude init. Changed downstream aggregation:
first layer mixes 8 similarity channels instead of 32 cost channels; deeper
layers unchanged in shape but trained on similarity statistics. Changed
readout/sign: forced negation of similarities (or readout flip) before
`softmax(-cost)`. Changed compute: volume activations 4× smaller; first-layer
MACs ~4× fewer. Changed optimization behaviour: different gradient
magnitudes through products vs differences; different init scale and
trajectory; similarity-peak vs cost-minimum filter learning. Remaining
confounds: operation type + width/capacity + sign handling + init shape +
normalization-by-fiat (unnormalized chosen) + channel-receptive-field change.

**D2 — Per-group L2-normalized True GWC, G=8, G preserved.**
Baseline: B0. Intervention: as D1 plus per-group L2 normalization
(`fea/(norm(fea,2,dim)+1e-05)`) of each grouped vector before the product.
Changed operation: signed difference → normalized per-group inner product.
Changed tensor shapes: as D1 (`(B, 8, 24, H, W)`). Changed parameter count:
as D1 (−20,736 weights). Changed initialization: as D1. Changed downstream
aggregation: as D1, except input statistics are normalized-similarity
curves (bounded, gain-invariant within groups). Changed readout/sign: as D1
(negation still forced). Changed compute: as D1 plus per-group norm + eps
overhead per candidate. Changed optimization behaviour: as D1, plus
normalization reshapes gradient flow (scale-invariance within groups,
eps-floor effects on dead features). Remaining confounds: all of D1's, PLUS
the normalization intervention as a further separable component.

**D3 — Degenerate G=C=32 elementwise product (width-preserving).**
Baseline: B0. Intervention: per-channel product `L_c(x) × R_c(x−k)` with no
intra-group pooling (group width 1). Changed operation: signed difference →
per-channel Hadamard similarity. Changed tensor shapes: none
(`(B, 32, 24, H, W)` preserved). Changed parameter count: zero delta
(first layer stays 27,648 weights). Changed initialization: shape-identical
`[32, 32, 3, 3, 3]`, but identical bytes carry different meaning (trained
on similarities, not differences). Changed downstream aggregation: shape
unchanged; learned function operates on similarity curves. Changed
readout/sign: negation still forced (second component even here). Changed
compute: products replace subtractions; FLOPs/memory at first layer
unchanged. Changed optimization behaviour: product-gradient dynamics and
similarity-peak filter learning despite matched shapes. Remaining confounds:
operation type + sign handling + init-meaning change; AND hypothesis
vacuity — with no cross-channel pooling, nothing about GROUPING is
exercised (Fact F). Parameter matching is bought at the price of testing
nothing.

**D4 — G=8 True GWC with G→32 learned adapter (parameter-matched).**
Baseline: B0. Intervention: D1 volume `(B, 8, 24, H, W)` followed by a
learned shape adapter restoring 32 channels before aggregation (e.g. 1×1×1
`Conv3d(8→32)`: `8 × 32 = 256` weights + 32 bias = 288 params; net total
≈ 398,242, i.e. +288 vs baseline). Changed operation: signed difference →
grouped inner product PLUS learned adapter projection. Changed tensor
shapes: intermediate `(B, 8, 24, H, W)`, aggregation input restored to 32.
Changed parameter count: first-layer deficit erased (−20,736) at the cost
of +288 adapter params; capacity ledger balanced, architecture not held.
Changed initialization: adapter weights are new random parameters with
their own init distribution and trajectory. Changed downstream aggregation:
first layer shape-matched but fed adapter outputs (linear mixes of group
similarities), not native costs or native similarities. Changed
readout/sign: as D1. Changed compute: adapter MACs added; volume memory
still reduced upstream of the adapter. Changed optimization behaviour: as
D1 plus joint adapter/aggregation co-adaptation dynamics. Remaining
confounds: all of D1's operation/sign/init components PLUS the adapter as
an explicit second architectural intervention (same class of change as
ARM-W's projection). The ledger balances; the causal question does not
isolate.

**D5 — Full-channel single correlation, G=1 (single similarity channel).**
Baseline: B0. Intervention: mean (or sum) over ALL 32 channels of
`L(x) × R(x−k)` per candidate: volume `(B, 1, 24, H, W)`. Changed
operation: signed difference → full-channel inner product. Changed tensor
shapes: 32 → 1 channel. Changed parameter count: first-layer weights
`1 × 32 × 27 = 864` (+32 bias); delta −26,784 weights; unmatched total ≈
371,170. Changed initialization: `[32, 1, 3, 3, 3]`, maximally
shape-incomparable with baseline. Changed downstream aggregation: first
layer collapses from full-32 mixing to single-channel mixing. Changed
readout/sign: negation forced. Changed compute: smallest volume and
first-layer MACs of all designs. Changed optimization behaviour: as D1,
amplified (extreme fan-in change, single-channel bottleneck dynamics).
Remaining confounds: operation type + maximal capacity change + sign +
init + normalization-by-fiat; AND no group structure at all, so it tests
similarity-vs-difference under collapse, not grouping. Not a "G=1 control"
for D1 — it is a second new arm requiring the same sign surgery and an
even larger aggregation change.

(G=4/G=16 variants scale by the same formula — first-layer weights
`G × 864`, delta `(G − 32) × 864` — and inherit D1's confound structure
with magnitude varying by G. They are members of the D1 family, not
separate hypotheses, and no sweep over them is proposed or authorized.)

---

## 6. CAUSAL INTERPRETATION OF EACH

For every design, what a positive result (lower frozen-contract EPE than
CONTROL) and a negative result (EPE at or above CONTROL) would actually
license. Where several components move, only the broader, weaker claim is
supported. "True GWC improves EPE" must never become "grouping caused the
improvement" when several components moved.

- **D1 positive** licenses: "the bundle (grouped inner-product operation +
  32→8 width/capacity reduction + sign negation + new init shape/trajectory)
  improves EPE over random-init P2A under the frozen recipe." It does NOT
  license "grouping caused the improvement" — the capacity removal, sign
  edit, and init change are live alternative explanations. **D1 negative**
  licenses: "this bundle does not improve EPE." It does NOT license
  "grouping hurts" or "GWC refuted" — any bundle member (e.g. the −20,736
  capacity cut) could carry the negative.
- **D2 positive** licenses: "the D1 bundle PLUS per-group L2 normalization
  improves EPE." Weaker still: normalization and grouping are jointly
  credited, never separated. **D2 negative** licenses only: "this larger
  bundle does not improve EPE"; it says nothing about D1 (normalization
  could be the spoiler) and nothing about grouping alone.
- **D3 positive** licenses: "per-channel similarity with matched width and
  forced negation improves EPE over the difference baseline" — a
  similarity-vs-difference claim at full width, NOT a grouping claim, since
  no pooling occurs. **D3 negative** licenses: "elementwise similarity does
  not improve EPE at matched width." It says nothing about grouped pooling
  (D1/D2), which remains untested by D3 either way.
- **D4 positive** licenses: "the grouped-similarity-plus-adapter bundle
  improves EPE." The adapter is a credited co-cause by construction; no
  statement about grouping alone follows, and no statement about
  "parameter-matched GWC" as a single variable follows, because the match
  was purchased with a new module. **D4 negative** licenses: "this adapted
  bundle does not improve EPE"; silent on unadapted D1 and on grouping.
- **D5 positive** licenses: "full-channel collapsed similarity improves
  EPE" — a similarity-vs-difference-under-collapse claim, not a grouping
  claim (there are no groups). **D5 negative** licenses: "collapsed
  similarity does not improve EPE"; silent on grouped designs.

General rule: every non-degenerate relaxed design supports at most a
bundle-level claim of the form "this exact multi-component construction
moved EPE in this direction under this recipe." No relaxed design on record
supports a sentence with "grouping" as the grammatical subject of the
causal verb.

---

## 7. ARM-W RELATION

No relaxed design makes ARM-W a control for True GWC, because ARM-W never
computed a grouped feature correlation. From the binding records
(`phase1/scripts/train_arm_w.py:1-14`; `src/models/stereonet/cost_volume.py:160-171`),
ARM-W applied a learned grouped 1×1 projection to the difference map: its
mathematical object is a linear combination of subtractive residuals, while
every D1/D2/D4/D5 design computes products of raw left/right features. A
control must differ from its treatment in exactly the component under test;
ARM-W differs in the operation itself, so pairing any relaxed GWC arm
against ARM-W varies the operation, the reduction, the parameter path, and
the init — it supplies no single-component contrast.

The one informative contrast ARM-W could supply in a relaxed program is
explicitly bundle-vs-bundle, not causal: D4 (grouped similarity + adapter)
vs ARM-W (grouped projection of differences + the same-width aggregation)
would contrast "grouped linear combination of differences" against "grouped
products of features plus adapter" at matched aggregation width — i.e. two
named multi-component constructions differing in operation type AND adapter
content. That contrast is descriptive (which bundle scores lower), never
attributive (it cannot isolate grouping, products, or the adapter). This is
not resolved by assertion: it follows directly from the code-cited mechanism
records above.

Preservation: ARM-W is retained as mechanistically non-equivalent historical
evidence — it refutes only its own proposition (learned 8-group 1×1
compression of the 32-channel difference volume, 377,250 params, does not
beat the uncompressed difference baseline under the Phase-1 protocol, with
"refuted" as the preregistered decision-rule outcome at a 0.0049 px margin
inside context-only spread). It transfers zero evidential weight to any
D1–D5 design in either direction.

---

## 8. DEPLOYMENT IMPLICATIONS

Strictly separated from identifiability. Nothing here influences any
scientific judgement in Sections 2–7, and no deployability is claimed
without testing — there are no compiler results, no on-device numbers, and
no toolchain verification in this record.

- **D1/D2 (G=8 preserved):** parameter count falls by 20,736 first-layer
  weights; volume memory 4× smaller (`(B, 8, 24, H, W)` vs `(B, 32, 24, H,
  W)`); first-3D-layer MACs ~4× fewer; activation bandwidth reduced in the
  same ratio. Tensor layout into aggregation changes (8 channels), which
  any downstream compiler mapping must re-derive. Whether the Hailo
  toolchain maps products + per-group reductions + forced negation as
  efficiently as the subtraction it currently contains — including
  quantization behaviour of products vs differences — is UNKNOWN.
- **D3 (G=32):** no parameter, memory, compute, or layout change at the
  first layer; no deployment argument in either direction. Op mix changes
  (products for subtractions) with unknown quantization consequences.
- **D4 (adapter):** volume memory still reduced upstream, but the adapter
  adds parameters (+288 for a 1×1×1 `Conv3d(8→32)`), MACs, and a new module
  the compiler must map; net first-layer-plus-adapter arithmetic must be
  re-tallied per concrete adapter choice. Layout into aggregation is
  restored to 32 channels, which may simplify downstream mapping relative
  to D1 — a packaging observation, not a scientific point.
- **D5 (G=1):** largest reductions (volume 32× smaller in channels;
  −26,784 first-layer weights) and the most extreme layout change; same
  toolchain unknowns, amplified by the single-channel bottleneck.

Net: desirability (smaller/cheaper) must never be mistaken for evidence
(grouping works). An attractive-but-untested proposition stays untested
regardless of its memory/MAC direction.

---

## 9. LOW-COST VALIDATION REQUIREMENTS

For EVERY potentially valid future design (any of D1–D5, or any construction
not anticipated here), the following gates are mandatory preconditions —
each must pass before the next begins, and passing all of them authorizes
nothing beyond readiness. NO FULL TRAINING IS AUTHORIZED BY THIS DOCUMENT.

- **Gate 0 — Source audit.** Identify the proposed volume construction
  line-by-line against `build_cost_volume`
  (`src/models/stereonet/cost_volume.py:72-119`) and the `CostVolume.reduce`
  path (`:145-171`); state the operation (product vs difference), reduction
  (mean vs sum vs none), normalization (if any, with eps), grouping
  (contiguous partition, widths), and sign handling BEFORE any code exists.
- **Gate 1 — Mathematical tensor test.** Prove forward shapes on paper and
  in a shape-only check: treatment volume `(B, G, 24, H, W)` vs control
  `(B, 32, 24, H, W)`; aggregation `in_channels` wiring
  (`src/models/stereonet/stereonet.py:108-119`,
  `src/models/stereonet/aggregation.py:26-40`) shown held or changed, with
  every change counted as a moved component.
- **Gate 2 — Parameter/integrity test.** Reconcile exact first-layer
  arithmetic (`in_channels × 32 × 27`) against the 397,954 frozen total;
  count every adapter parameter as a second component, never netted away;
  prove strict-load consequences (which shapes strict-load, which require
  surgery documented as an intervention).
- **Gate 3 — Numerical operation test.** Prove the treatment is neither
  identical to the baseline (identical-zero at identical input must FAIL
  for similarity vs cost) nor identical to ARM-W's projection (product
  structure vs learned linear combination of differences).
- **Gate 4 — One-batch forward/backward.** Gradient flow to the feature
  extractor through the new op on CPU, one batch, no training; confirms the
  graph is differentiable end-to-end, nothing more.
- **Gate 5 — Checkpoint round-trip.** Strict-load / strict-shape audit
  proving init identity or documenting the init surgery as a second
  intervention with its sha and justification burden.
- **Gate 6 — Real-entrypoint smoke.** Lesson already learned and carried
  from the Closure Record: a smoke test must exercise the REAL entrypoint
  (the actual proposed training script, e.g. `P2A_EPOCHS=2` through it),
  not only imported modules — the earlier Kaggle gate passed without
  invoking `finetune_pilot.py` and missed the missing
  `phase1/runs/arm_v/arm_v_best.pth` dependency and the `REPO_ROOT`
  nesting depth. A gate exercising anything other than the real entrypoint
  is a non-gate and passes nothing.
- **ONLY THEN full training** — under a separate preregistration with its
  own hypothesis, frozen variables, and success/rejection criteria, recorded
  separately without overwriting any prior run directory, checkpoint, or
  record. That training is NOT authorized here; these gates VERIFY an
  intervention, they cannot CREATE identifiability.

---

## 10. CLOSED BRANCHES

Explicitly preserved as closed. Nothing in this document reopens them:

- Additional ARM-P seeds, ARM-P reruns, pooling of seeds, or any new
  ARM-P-vs-CONTROL comparison.
- The coverage experiment on frozen P2A (Candidate C NOT IDENTIFIABLE;
  Section 5 of the separability analysis is NOT APPLICABLE).
- The frozen-platform True GWC experiment (NOT IDENTIFIABLE; no matched
  CONTROL/TREATMENT pair exists).
- Architecture sweeps, recipe sweeps, and hyperparameter sweeps of any
  kind (no tuning, no new metric, no new seed).
- Arbitrary readout/aggregation changes outside a named design's forced
  components.
- Any experiment whose only justification is that it is cheap — cost is
  never a license, and the low-cost gates in Section 9 authorize no
  training by themselves.

---

## 11. REMAINING RESEARCH QUESTIONS

What is genuinely still unresolved, stated without ranking:

1. Does FT3D pretraining help KITTI fine-tuning because of high-disparity
   coverage content, or because of other pretraining content (weight scale,
   feature statistics, trajectory position)? Unseparable on frozen P2A
   (Candidate C); the ARM-P delta's internal decomposition is unmeasured.
2. Does grouped feature correlation with G preserved into aggregation help
   or hurt under any recipe? UNTESTED (D3's rescaling was a constructional
   gap; ARM-W was non-equivalent). Every constructible test is
   multi-component.
3. Is similarity-vs-difference at full channel width (D3) neutral, helpful,
   or harmful? This is the only width-preserving operation question, and it
   is not the grouping question.
4. How much of the Stage-A sub-64px gap (~31% where every diagnostic looks
   healthy) is attributable to which mechanism? Still diffuse; below-64
   attribution from existing dumps only was the U3 candidate, never run.
5. Are readout and refinement co-adapted such that neither can be judged
   without joint retraining (the U1 candidate)? Still NOT IDENTIFIABLE
   under frozen protocols.
6. Does any multi-component GWC bundle preserve accuracy while reducing
   memory/MACs enough to matter for the Hailo target — and does the Hailo
   toolchain actually map products, per-group reductions, and negation
   efficiently? Both halves unknown; the second half has zero measurements.

---

## 12. RESEARCH-BOUNDARY DECISION

**Decision: A — freeze current evidence and move toward paper/report
synthesis.**

Argued explicitly from each required criterion:

- **Unresolved scientific question.** Real unresolved questions exist
  (Section 11), but every GWC-shaped one is multi-component by construction
  (Sections 4–6): the only askable relaxed question is the weak
  bundle-level "does this exact construction move EPE", which cannot
  attribute anything to grouping. Synthesis preserves the strong negative
  findings (two NOT IDENTIFIABLE verdicts with independent blocking
  grounds); reopening trades them for weak bundle claims.
- **Identifiability.** Option B cannot supply it: D1/D2/D4/D5 move three or
  more components; D3 is identifiable but hypothesis-vacuous (Fact F). A
  program whose flagship experiment cannot isolate its named variable
  should not be opened. Option A keeps the identifiability discipline that
  produced the closed results intact.
- **Intervention definition.** The frozen program's interventions were
  single-edge (ARM-P: strict init load; CONTROL: random init). Every
  relaxed GWC intervention is a bundle whose membership (operation, width,
  sign, init, normalization, adapter) must be carried as a conjunction in
  every sentence. B would permanently weaken the meaning of "intervention"
  in this research line; A preserves it.
- **Compute.** B spends full 200-epoch multi-seed trainings (plus controls)
  on bundle-level claims of known-weak interpretability — the worst
  compute-per-bit-of-evidence ratio available. A spends zero compute and
  banks the closed evidence.
- **Deployment relevance.** The memory/MAC direction of D1/D2 is genuinely
  attractive for an edge target, but attractiveness is not evidence
  (Section 8), and the toolchain unknowns (quantization of products,
  per-group reductions, negation mapping) mean deployment relevance is
  speculative until measured on-device. Speculative relevance cannot carry
  the decision against five other criteria.
- **Provenance burden.** B inherits every open provenance limit (FT3D
  A+C-only subset with B missing; upstream provenance NOT ESTABLISHED per
  Closure Record Sections 13a–13b) and adds new ones (normalization fiat,
  adapter justification, init-surgery documentation). Each must be carried
  forever in every downstream claim. A caps the burden where it stands.
- **Interpretability of the resulting evidence.** Under B, both positive
  and negative outcomes license only bundle-level sentences (Section 6) —
  evidence that is easy to misread ("GWC works/doesn't work") and hard to
  build on. Under A, the evidence to be synthesized is crisp: a replicated
  initialization-bundle observation with bounded environment confound, plus
  two independently-grounded NOT IDENTIFIABLE boundary results. Crisp,
  limited evidence outranks abundant, uninterpretable evidence.

Option B is therefore declined not because GWC "seems unpromising" — no
promise assessment enters this decision — but because it fails the
criteria above: it buys weak, bundle-level, provenance-heavy evidence at
full compute cost while diluting the intervention standard. Freeze, write,
and let any future multi-component GWC program be proposed — if ever — as
a new research line with its own preregistration, not as a continuation.

---

## 13. EXACT NEXT ACTION

Write the Stage B synthesis/report outline as a zero-compute planning
document (headings, evidence inventory with file-path citations, and open
questions carried verbatim from Section 11) — no GPU, no code, no training,
no new measurement, and no modification of any existing file.

---

*End of boundary document. Zero GPU used. No file modified. No compute ran.*
