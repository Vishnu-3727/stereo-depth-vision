# GWC Identifiability Definition — Stage B (Specification Only)

Zero-GPU specification document. No training was run, no evaluation was run, no
architecture was implemented, no GWC code was written, no experiment was
prepared or launched, and no existing file was modified for this document. The
question answered here is ONLY whether a valid True-GWC experiment exists on
the frozen P2A platform — not whether GWC works. Phases 0–2 are CLOSED, Stage A
is COMPLETE (architecture gate: NO ARCHITECTURE JUSTIFIED), and no Phase 3 is
invented or proposed. ARM-P is an INITIALIZATION INTERVENTION throughout;
CONTROL means RANDOM INITIALIZATION under the original P2A recipe, per
`stage_b_armp/ARMP_CLOSURE_RECORD.md` Sections 2–3. This document is governed by
that Closure Record and by `stage_b_armp/CANDIDATE_C_SEPARABILITY.md` (verdict:
C NOT IDENTIFIABLE). A regression is recorded as a regression, never
reinterpreted.

---

## 1. GWC DEFINITION

True group-wise correlation (True GWC), as used in the public precedent shapes
cited in `phase1/docs/MATCHING_REPRESENTATION_AUDIT.md` Section 3, is the
following exact mathematics.

Inputs: left and right feature maps `L, R` of shape `(B, C, H, W)` with channel
dim `C = 32` (shared siamese extractor output at stride 8, e.g. `(B,32,32,64)`
for a 256x512 crop; see MATCHING_REPRESENTATION_AUDIT.md:52-57). Group count
`G` divides `C`; group width is `C/G` channels per group (e.g. `G=8` gives
width 4). Candidate dim `D = 24` disparities `k = 0..23`; spatial dims
`(H, W)` at feature resolution.

Exact operation per candidate `k`, per group `g`, per pixel `x` (unnormalized
form, Fast-ACVNet `models/submodule.py:104-110`, OpenStereo IGEV
`models/igev/submodule.py:158-177`):

- Partition channels into `G` contiguous groups.
- Elementwise multiply the grouped left vector by the grouped shifted-right
  vector: `L_g(x) * R_g(x-k)`.
- Reduce by MEAN over the `C/G` channels within the group (CoEx variant in
  OpenStereo `coex_cost_processor.py:10-35` reduces by SUM over
  channels-per-group; mean and sum differ by the constant factor `C/G`).

Output tensor shape: `(B, G, D, H, W)` — one scalar similarity per group per
candidate per pixel. The normalized variant (Fast-ACVNet
`models/submodule.py:124-132`, IGEV `:180-184`) first L2-normalizes each
per-group vector independently (`fea/(norm(fea,2,dim)+1e-05)`) and is a
different intervention from the unnormalized form; exactly one must be frozen
in any future preregistration, and neither is proposed here.

What True GWC is NOT (each is a distinct intervention, not a substitute):

- Full-channel correlation then `1/G` rescaling: computes the dot over all 32
  channels and divides by `G`. This is a post-hoc rescale of a different
  quantity, not a grouped reduction. Stage-A D3 did exactly this and produced
  identical G4/G8/G16 ranks BY CONSTRUCTION — a known gap, not a refutation
  (`stage_a_diagnostics/mechanism_verdict.json:20-27`; Closure Record
  Section 14).
- Averaging independently computed full-channel correlations: same objection;
  the grouping never constrains which channels interact.
- Post-hoc rescaling of any kind: changes magnitude without changing the
  channel-interaction structure.
- Channel shuffling or feature grouping WITHOUT grouped correlation (e.g. a
  grouped projection applied to an already-built difference volume): regroups
  channels but keeps the matching operation unchanged. This is what ARM-W did
  (see Section 8) and it is mechanistically non-equivalent to True GWC.
- Single-channel correlation (mean/sum over ALL `C`): collapses to `(B,D,H,W)`
  or `(B,1,D,H,W)`; it has no group structure at all.

The load-bearing property of True GWC, per `mechanism_verdict.json:144-147`
(U2), is that `G` is PRESERVED as channels into aggregation — the volume
entering the 3D aggregation has `G` channels, and the aggregation is trained on
that layout. Any construction that collapses groups before aggregation, or
averages them away in a diagnostic proxy, does not test the grouping
hypothesis.

## 2. BASELINE OPERATION

P2A actually computes a signed feature DIFFERENCE, not a correlation and not a
dot product. In `src/models/stereonet/cost_volume.py:101-119`
(`build_cost_volume`), with the P2A default `method="subtract"` and
`shift="right"` (ARM-V wiring per MATCHING_REPRESENTATION_AUDIT.md:48-50;
P2A constants 24 candidates, shift right, per Closure Record Section 8):

- Per candidate `k`: `anchor, other = left, shift_right(right, k)`
  (`cost_volume.py:108-109`), where `shift_right` pads `k` zeros on the left
  and slices `[..., :width]` (`cost_volume.py:45-56`), so candidate `k`
  computes `left(x) - right(x-k)` in LEFT-frame indexing, matching KITTI
  left-view ground truth.
- Per level: `levels.append(anchor - other)` (`cost_volume.py:110-111`), i.e.
  `cost_k(x) = left(x) - right(x-k)`, `k = 0..23`, a 32-dimensional signed
  vector per pixel per candidate. Exact match gives the zero vector; mismatch
  gives signed nonzero. It is a COST (low = good), not a similarity.
- Assembly: `torch.stack(levels, dim=1)` then `permute(0,2,1,3,4)`
  (`cost_volume.py:118-119`) yielding `(B, C, D, H, W) = (B, 32, 24, H, W)`
  with `C = 32` FULL CHANNELS preserved.

`method="concat"` (`cost_volume.py:112-113`) exists and doubles channels to
`(B, 64, D, H, W)`; it is not the default and is not the baseline. The
readout proves cost semantics: `DisparityRegression` computes `softmax(-cost)`
(MATCHING_REPRESENTATION_AUDIT.md:78-82, citing
`src/models/stereonet/regression.py:47-49`), i.e. an explicit argmin — low
cost becomes high weight. The frozen parameter count is 397,954 (Closure
Record Section 8; operative eval-time guard `scripts/eval_tier2.py:130-132`
with `EXPECTED_PARAMS = 397954`).

Consequence stated plainly: there is NO `G=1` correlation in the baseline, so
the framing "`G=1` -> `G>1`" is NOT the native axis on this platform. The
native axis would be "signed difference -> grouped inner product", which
changes the matching OPERATION TYPE, not a group count. This is addressed in
Sections 4–5: introducing GWC changes the operation type (signed difference ->
inner product) AND, for every non-degenerate `G`, the channel width entering
aggregation (32 -> `G`). That is TWO causal components, not one (details in
Sections 4–5; boundary case `G = C = 32` in Section 5).

## 3. TRUE GWC OPERATION

What would replace the Section-2 operation, exactly, under the U2 definition
(True GWC with `G` preserved into aggregation):

- Inputs unchanged: `(B, 32, H, W)` left/right feature maps at stride 8.
- Per candidate `k = 0..23`, with right features shifted right by `k`
  (same `shift_right` indexing as now, so the geometric correspondence is
  held fixed): partition the 32 channels into `G` groups of width `C/G`.
- Per group `g`: `s_{g,k}(x) = mean_{c in group g} (L_c(x) * R_c(x-k))`
  (unnormalized precedent; sum-variant differs by factor `C/G`; normalized
  variant adds per-group L2 normalization with eps `1e-05` before the
  product — a further sub-choice that must be frozen, not swept).
- Stack over candidates and permute to `(B, G, D, H, W)`.

What MUST also change for the network to remain correct (these are not
optional extras; they are forced by the operation swap):

- SIGN HANDLING. GWC outputs a SIMILARITY (high = good); the frozen readout
  computes `softmax(-cost)` (low = good). Porting a similarity into this
  graph without negation inverts the matching evidence: the true match would
  receive the LOWEST weight. The audit's cross-check on sign
  (MATCHING_REPRESENTATION_AUDIT.md:95-101, and AANet's explicit sign switch
  `nets/estimation.py:16-19` cited at :116-117) establishes that any
  similarity-valued proposal MUST be negated before the readout (or the
  readout sign flipped). That negation-or-flip is part of the port, and the
  audit's candidate matrix (Section 5, row B) judges it "arguably a second
  variable" — recorded here as a confound component, not absorbed silently.
- AGGREGATION INPUT WIDTH. The first `Conv3d` of
  `src/models/stereonet/aggregation.py:26-40` consumes the volume channel
  dim. A `(B, G, D, H, W)` volume with `G != 32` changes that layer's
  `in_channels` (wiring in `src/models/stereonet/stereonet.py:108-119`,
  `agg_in`, and `stereonet.py:125-127`). Arithmetic in Section 5.

## 4. IDENTIFIABILITY AUDIT

The question: can the native-axis intervention (signed difference -> grouped
inner product) be the ONLY intended change, with everything else controlled?
Audit of every quantity that changes automatically:

- Matching operation type: signed per-channel difference -> per-group inner
  product (mean/sum of products). CHANGES by definition. This is the intended
  intervention, but it bundles two sub-changes: the interaction (subtractive
  vs multiplicative) AND the reduction (identity per channel vs pooling over
  `C/G` channels). The reduction destroys per-channel identity within each
  group — an information-structure change inseparable from the operation swap.
- Correlation computation and representational connectivity: the aggregation's
  learned 3x3x3 filters would be trained on similarity curves (peaks at the
  match) instead of cost curves (minima at the match). The filter weights mean
  different things under the two layouts even at identical shapes. CHANGES
  (this is the treatment effect itself, not a separate confound — but it means
  "same weights, different op" is not a meaningful control; retraining is
  required, and retraining introduces the optimization confounds below).
- Sign/negation: forced as above (Section 3). CHANGES — a second causal
  component under the audit's own row-B judgement. It cannot be left unchanged
  (un-negated similarity is incorrect, not a control) and cannot be changed
  for free (it is an additional edit to the forward graph with its own
  justification burden).
- Normalization: the public GWC precedent exists in BOTH unnormalized
  (`submodule.py:104-110`) and L2-normalized (`submodule.py:124-132`) forms.
  The choice must be frozen; picking the normalized form adds a per-group
  normalization intervention on top of grouping. Even picking the
  unnormalized form is a choice against the cross-repo regularity (normalize
  before matching) documented in the audit. CHANGES-or-CHOICE: either way a
  degree of freedom that must be frozen by fiat, not derived from the
  baseline.
- Learnable parameter count: CHANGES for every `G != 32` (exact arithmetic in
  Section 5: first-layer weights `in_channels * 32 * 27`; `G=8` removes 20,736
  weights). Capacity, not just shape.
- FLOPs, activation size, memory: first-layer MACs scale linearly with
  `in_channels` (32 -> `G`); volume activations scale as `G/32`. These move
  WITH the channel-width change. Equal-FLOPs or equal-parameter padding (e.g.
  widening elsewhere to compensate) would require editing another component
  and is RECORDED AS A CONFOUND, not performed silently: there is no
  "compute-matched" version of this comparison that holds the architecture
  fixed, because matching compute IS changing the architecture. Equal
  parameter count or equal FLOPs does not make a comparison causal — the
  causal question is whether EXACTLY ONE graph edge changed, and padding
  another edge to balance a ledger changes a second edge. The ledger is an
  accounting identity, not a causal license.
- Initialization shape and random-seed behaviour: the first `Conv3d` weight
  tensor changes shape (`[32, in_channels, 3, 3, 3]`) for `G != 32`, so the
  same initialization bytes cannot be reused; default Kaiming-style init
  scales with fan-in (`in_channels * 27`), so even the same seed produces a
  different effective init distribution at different `G` (see Section 6).
  Optimization behaviour (gradient magnitudes, trajectory, convergence speed)
  then differs for reasons that are not the grouping hypothesis. CHANGES.
- Receptive field: the 3x3x3 kernels span the same (D,H,W) neighbourhood, so
  geometric receptive field is UNCHANGED; but the CHANNEL receptive field
  (which input channels each output sees) changes from full-32 mixing to
  G-mixing at the first layer. MIXED — geometry held, channel connectivity
  changed.
- Feature grouping: grouping the FEATURES (partitioning channels) with no
  other change is itself an intervention distinct from grouped CORRELATION
  (see Section 1 exclusions and Section 8 on ARM-W). True GWC bundles both.
  CHANGES as part of the bundle.

Summary: the native-axis swap moves the operation type, the sign convention,
the aggregation input width (hence parameters, FLOPs, memory, init shape, and
optimization dynamics), and the normalization choice — at minimum the
operation swap PLUS the width change PLUS the sign handling. That is three
components, not one. The `G=1 -> G>1` framing would additionally be
non-native (there is no `G=1` baseline on this platform) and is rejected as
the experiment axis: a "G=1 correlation" control would itself be a NEW
operation (full-channel correlation, single channel, requiring the same sign
surgery and a 32->1 aggregation change), i.e. a second new arm, not the
baseline.

## 5. PARAMETER/CAPACITY AUDIT

Yes — the implementation changes learnable parameter count for every True-GWC
`G` (every `G < C` with `G` preserved into aggregation, which is the U2
definition of True).

Exact arithmetic. `src/models/stereonet/aggregation.py:26-40`: four
`Conv3d(c_in -> 32, 3x3x3, padding 1)` + LeakyReLU, then
`to_cost = Conv3d(32 -> 1, 3x3x3, padding 1)`. First-layer weights =
`in_channels * 32 * 3 * 3 * 3 = in_channels * 32 * 27 = in_channels * 864`
(+ 32 bias terms):

- Baseline `in_channels = 32`: `32 * 32 * 27 = 27,648` weights (+32 bias =
  27,680).
- True GWC `G = 8`: `8 * 32 * 27 = 6,912` weights (+32 bias = 6,944).
- Delta at `G = 8`: `-20,736` weights (capacity REMOVED, ~5.2% of the frozen
  397,954 total). Cross-check: `phase1/scripts/train_arm_w.py:1-12` records
  ARM-W (`cost_volume_groups=8`) at 377,250 params versus ARM-V's 397,954 —
  a difference of 20,704 = 20,736 - 32, where the 32 is ARM-W's grouped
  `Conv2d(32, 8, 1, groups=8, bias=False)` projection
  (`src/models/stereonet/cost_volume.py:160-163`; params `(32/8)*8 = 32`).
  The arithmetic closes against an independent record.
- General `G`: first-layer weights `G * 864`; delta vs baseline
  `(G - 32) * 864` weights.

The confound is CAPACITY (fewer first-layer parameters at `G < 32`) fused
with REPRESENTATIONAL CONNECTIVITY (the remaining parameters mix grouped
similarities instead of full-channel costs). A worse (or better) number at
`G=8` cannot be credited to "grouping helps/hurts" rather than to "removing
~20.7k parameters and their optimizer trajectory helps/hurts".

Can an exactly matched control be built WITHOUT modifying more than one
causal component? No:

- Holding `in_channels = 32` while testing grouping requires `G = 32`
  (boundary case, assessed next) or a shape adapter (e.g. a learned
  projection `G -> 32`, or tiling/repeating groups). Any adapter ADDS learned
  parameters and a new module to the graph — a second architectural
  intervention by definition (same class of change as ARM-W's projection and
  ARM-Y's gate, both separately closed arms). Padding by widening another
  layer to equalize totals likewise edits a second component (Section 4:
  the ledger is not a license).
- Running WITHOUT matching (accept the 20,736-weight deficit) leaves the
  capacity confound uncontrolled: the comparison is
  "grouped-similarity-with-fewer-params vs full-channel-difference", i.e. two
  differences.

Boundary case `G = C = 32` (group width 1): preserves `in_channels = 32`
exactly — zero parameter delta, zero shape change, zero FLOP change at the
first layer. It is nevertheless NOT True GWC and does not test the grouping
hypothesis: with one channel per group, the "mean over channels-per-group"
reduces to the identity, so each output channel is the bare elementwise
product `L_c(x) * R_c(x-k)` — per-channel Hadamard similarity. No
cross-channel pooling occurs; nothing about GROUPING (the hypothesis that
pooling channels into group similarities helps) is exercised. What it tests
instead is similarity-vs-difference at full channel width (audit Section 5,
row B: "marginal", "resembles group-correlation with G=32; separable in
letter but not in spirit from ARM-W"), and it STILL requires the sign
negation (a second component per Section 4). It is therefore a degenerate
case: parameter-matched but hypothesis-vacuous AND still sign-confounded.
It cannot rescue identifiability.

## 6. INITIALIZATION AUDIT

No — `G=1`-style and `G>1`-style (more precisely, baseline-32ch and
True-GWC-Gch) initializations cannot be drawn from the same underlying
parameter representation without a second intervention.

- SHAPE MISMATCH. The first aggregation `Conv3d` weight is
  `[32, in_channels, 3, 3, 3]`: `[32, 32, 3, 3, 3]` at baseline vs
  `[32, G, 3, 3, 3]` at True GWC (`G != 32`). There is no identity mapping
  between these tensors. Any "conversion" (truncation, averaginginput-channel
  slices, tiling, re-initializing the surplus/deficit) is a researcher-chosen
  surgery on the initial weights — a second intervention with its own
  justification burden, invented merely to make the experiment identifiable.
  Per the brief, no such conversion is invented here; the limitation is
  stated instead.
- DISTRIBUTION MISMATCH even at matched shapes. Default conv initialization
  scales with fan-in. At `G=8` the fan-in is 4x smaller, so the same nominal
  seed yields larger-magnitude initial weights with different forward gain
  than at baseline — the two arms start the optimizer in different effective
  positions for reasons unrelated to grouping. At the degenerate `G=32` the
  shapes match, but the weights would be trained on similarity statistics
  rather than difference statistics, so identical init bytes do not mean
  identical init MEANING — noted, not counted twice; the sign confound
  (Section 4) already disqualifies that case.
- STRICT-LOAD PRECEDENT. The project's own initialization discipline
  (Closure Record Section 2: `model.load_state_dict(sd, strict=True)`) treats
  ANY shape change as a different initialization, not a convertible one.
  ARM-P's identifiability rested on byte-identical shapes with strict
  loading; True GWC breaks shapes by construction (for every non-degenerate
  `G`), so the ARM-P precedent cuts AGAINST convertibility here.

Limitation stated plainly: the two arms must either start from
incomparable-shaped random inits (confounding grouping with init
scale/trajectory) or undergo an init-surgery that is itself an uncontrolled
second intervention. Neither path isolates grouping.

## 7. RECIPE INVARIANTS

The invariants that WOULD have to hold for any single-component claim, stated
against the frozen P2A recipe (values from the audit record and the Closure
Record Sections 7–8):

dataset (KITTI `hailo_calib` 160 scenes train, `hailo_val` 40 scenes eval, no
overlap); split enforcement (no `hailo_val` sample in the optimizer loop);
augmentation (random 256x512 crop, independent per-image gain jitter sigma
0.1, normalize wrapper, no flip); loss (masked smooth-L1, beta 1.0, valid =
gt > 0 and gt < max_disparity); optimizer (Adam 0.9/0.999, lr 1e-3); LR
scheduler (cosine to 0); epochs (200); batch size (2); seed discipline
(0/1/2, all to completion before any verdict read); evaluation contract (40
scenes, 3,802,797 valid pixels, gt_scale 256.0, gt_source disp_occ_0,
contract_match true — Closure Record Section 8).

None of these must change for a GWC swap: the operation replacement is
forward-graph-local and the recipe (data, loss call with
`max_disparity=float(config.max_disparity_px)`, optimizer, schedule, seeds,
evaluator) can be held byte-identical. That is necessary but NOT sufficient
for identifiability — the recipe CAN be frozen, and the failure lies
elsewhere (Sections 4–6: the graph change itself bundles multiple causal
components even under a perfectly frozen recipe). If a future design
attempted parameter matching via an adapter or loss reweighting, THOSE would
be the recipe/graph changes that destroy single-component identifiability
(adapter = second architecture component; reweighting = loss change) — but no
such change is proposed here precisely because it would violate the
one-primary-intervention rule (Closure Record Section 15; CANDIDATE_C
Sections 4–5 precedent).

## 8. ARM-W RELATION

What ARM-W actually computed, from the documents (not asserted past them):

- `phase1/scripts/train_arm_w.py:1-12` (docstring, binding): "Identical to
  ARM V EXCEPT the cost volume: `cost_volume_groups=8`... Grouping passes the
  per-hypothesis 32-channel DIFFERENCE map through one shared learned grouped
  1x1 convolution `nn.Conv2d(32, 8, 1, groups=8, bias=False)`, reused for
  every hypothesis, reducing the cost volume from `(B,32,D,H,W)` to
  `(B,8,D,H,W)`." Config at `train_arm_w.py:134-136`. Mechanism at
  `src/models/stereonet/cost_volume.py:160-171` (`self.reduce =
  nn.Conv2d(channels, groups, 1, groups=groups, bias=False)` applied per
  disparity slice) with aggregation input rewired to `cost_volume_groups`
  channels (`src/models/stereonet/stereonet.py:108-119`).
- The audit's exclusion analysis (MATCHING_REPRESENTATION_AUDIT.md:139-157)
  classifies this, from the code, as "a channel COMPRESSION: 32 feature
  channels -> G correlation channels, discarding per-channel identity to save
  memory" — more precisely per the script, a learned grouped PROJECTION of
  the difference map, not a correlation of features. No elementwise feature
  product, no per-group mean/sum of `fea1 * fea2`, appears anywhere in the
  ARM-W path. The public GWC forms (Fast-ACVNet `:104-110`, IGEV `:158-177`,
  CoEx `:10-35`) compute products of FEATURES; ARM-W computes a learned
  linear combination of DIFFERENCES. These are different mathematical objects
  (multiplicative interaction of raw features vs linear projection of
  subtractive residuals), and only the former tests the grouping hypothesis.

The tension and its resolution:

- `FINAL_PHASE1_BOTTLENECK_AUDIT.md:426` lists "ARM-W | group-wise cost
  compression | refuted for accuracy, 1.7776" against the incumbent ARM-V
  1.7727 (`:425`); `:438` calls ARM-W "the channel-layout axis".
  MATCHING_REPRESENTATION_AUDIT.md:109 says "ARM-W tested the grouped form
  and refuted it", while `mechanism_verdict.json:20-27` records True GWC as
  UNTESTED (D3 group-averaged the full-channel dot by 1/G: identical ranks BY
  CONSTRUCTION — a known gap, not a refutation) with "ARM-W (group-wise
  cost) separately REFUTED in Phase 1 at ~1.776".
- Resolution: the `:109` sentence is loose language INSIDE the same document
  whose Section 4 (lines 139-157) gives the exact mechanism as compression,
  and the binding code record (`train_arm_w.py:1-12`) confirms projection of
  differences, not grouped correlation. The accurate reading, following the
  code over the sentence, is that ARM-W tested the CHANNEL-LAYOUT axis
  (32 -> 8 learned compression of the difference volume), which is why the
  bottleneck audit (`:438`) names it "the channel-layout axis" — a label
  consistent with compression and inconsistent with correlation. The
  "grouped form" phrase at `:109` refers at most to the family resemblance
  (grouped 1x1 convolution with `groups=8`), not to mechanistic identity.

Classification: MECHANISTICALLY NON-EQUIVALENT to True GWC. ARM-W provides
no evidence — not even partial evidence — on whether grouped FEATURE
CORRELATION with `G` preserved into aggregation (U2) helps or hurts, because
it never computed a grouped feature correlation. It refutes only its own
proposition: learned 8-group projection of the 32-channel difference volume
(377,250 params) does not beat the uncompressed difference baseline under
the Phase-1 protocol.

On "refuted" at a 0.0049 px margin: 1.7776 - 1.7727 = 0.0049 px sits INSIDE
the quoted 0.0444 px historical run-to-run spread, which is context ONLY and
is NOT a significance threshold in either direction (Closure Record Sections
4, 10–12; CANDIDATE_C Section 1). "Refuted" here can only mean the
preregistered decision-rule outcome (mean at or above the ARM-V-anchored bar
-> REFUTED and closed, per the audit's Section 9-style rule reproduced in
ARM_Y/Z records), NOT a statistically established inferiority and NOT proof
that no grouped construction could ever work. What it CANNOT mean is that
the 0.0049 px gap clears any noise bar — it does not, and no such bar
exists. The label closes ARM-W's own compression proposition under its rule;
it transfers zero evidential weight to True GWC, which remains UNTESTED per
the mechanism verdict. The available record IS sufficient to decide what
ARM-W computed (the script docstring + config + `CostVolume.reduce` code
agree), so no "cannot be determined" fallback is needed — but had the code
record been absent, this section would be required to say so rather than
guess.

## 9. MINIMUM MATCHED EXPERIMENT

NOT APPLICABLE - see Section 11.

(No CONTROL vs TREATMENT pair is specified because no single-component
intervention exists: every True-GWC construction bundles the operation-type
swap with the aggregation-width change and the sign handling per Sections
4–6. Proposing `G` values or a sweep would manufacture the appearance of a
viable experiment where none exists, and is prohibited for the same reason
CANDIDATE_C Section 5 was NOT APPLICABLE.)

## 10. LOW-COST GATES

NOT APPLICABLE as authorizing gates — see Section 11. They are recorded here
ONLY as the preconditions any future identifiable design (if one is ever
found) must pass, authorizing no training by themselves (same standing as
CANDIDATE_C Section 8):

- Gate 0 (source/code inspection): the proposed volume construction
  identified line-by-line against `build_cost_volume`
  (`src/models/stereonet/cost_volume.py:72-119`) and the `CostVolume.reduce`
  path (`:145-171`), with the operation (product vs difference), reduction
  (mean vs sum), normalization (if any, with eps), and sign handling stated
  before any code is written.
- Gate 1 (tensor-shape test): forward shape proof — treatment volume
  `(B,G,D,H,W)` vs control `(B,32,D,H,W)` — with the aggregation
  `in_channels` wiring (`src/models/stereonet/stereonet.py:108-119`,
  `src/models/stereonet/aggregation.py:26-40`) shown to be held or changed,
  and the change counted as a component.
- Gate 2 (numerical equivalence/non-equivalence test): proof the treatment
  is neither identical to the baseline (identical-zero at identical input
  must FAIL for a similarity vs a cost) nor identical to ARM-W's projection
  (product structure vs learned linear combination of differences).
- Gate 3 (parameter-count/integrity test): exact first-layer arithmetic
  (`in_channels * 32 * 27`) reconciled to the 397,954 frozen total, with any
  adapter parameters counted as a second component, not netted away.
- Gate 4 (one-batch forward/backward): gradient flow to the feature
  extractor through the new op on CPU, one batch, no training.
- Gate 5 (checkpoint round-trip): strict-load/strict-shape audit proving
  init identity or documenting the init surgery as a second intervention.

None of these gates was run (zero compute for this document), and none
rescues a design that fails Sections 4–6: gates VERIFY an identifiable
intervention; they cannot CREATE one.

## 11. STOP CONDITIONS

The stop condition is MET. True GWC cannot be isolated to a single causal
component on the frozen P2A platform, on three independent grounds (any one
alone is sufficient to stop):

1. OPERATION + WIDTH BUNDLE. Every True-GWC construction (G < C, G preserved
   into aggregation per U2) changes the matching operation type (signed
   difference -> grouped inner product) AND the aggregation input width
   (32 -> G, hence -20,736 first-layer weights at G=8 per Section 5). Two
   architectural components move together; the parameter-matched rescue
   (adapter/projection) is itself a second component, and the unmatched
   alternative carries a capacity confound. The degenerate rescue (G=C=32)
   preserves width but is hypothesis-vacuous (group width 1 = elementwise
   product, no channel pooling) and still sign-confounded.
2. SIGN/INITIALIZATION BUNDLE. The similarity-for-cost port forces sign
   handling (negation or readout flip — a second graph edit per the audit's
   own row-B judgement), and the shape change forces incomparable-shaped
   initializations with fan-in-dependent init distributions (Section 6). No
   conversion exists that is not itself an intervention.
3. VERIFIABILITY WITHOUT ISOLATION. Gates 0–5 (Section 10) could verify THAT
   a GWC tensor was built, but verification is not identifiability: a
   verified two-component change is still a two-component change, and under
   the one-primary-intervention rule (Closure Record Section 15) it cannot
   attribute any outcome to grouping.

The remaining stop clauses are checked and do not independently trigger but
are preserved for the record: the operation is NOT mathematically equivalent
to the baseline (so that clause is satisfied, not triggering); ARM-W
evidence does not make the comparison non-identifiable — it is simply
non-equivalent and therefore silent (Section 8) rather than blocking. The
blocking grounds are (1) and (2) above.

## 12. DEPLOYMENT RELEVANCE

Plausible relevance to the Hailo target, stated SEPARATELY and carrying zero
weight on the Sections 4–6 and 11 verdict (desirability does not influence
identifiability):

- A `(B,G,D,H,W)` volume with `G < 32` is strictly smaller than the baseline
  `(B,32,D,H,W)` (e.g. 4x smaller at G=8), reducing volume memory, first-3D-layer
  MACs (~4x at G=8), and activation bandwidth — all desirable directions for
  a memory- and MAC-constrained edge compiler target. The degenerate G=32
  case offers no such saving.
- Countervailing considerations (also plausible, also untested): the
  elementwise product + per-group reduction is a different op mix than the
  subtraction the current compiled artifact contains; whether the Hailo
  toolchain maps it as efficiently (quantization behaviour of products vs
  differences, per-group reductions, the forced sign negation) is UNKNOWN.
  No Hailo compatibility is claimed until actually tested — no compiler
  result, no on-device number, and no toolchain verification exists in this
  record (Closure Record Section 12 carries the same standing for ARM-P).
- Net: IF a future identifiable experiment ever showed a GWC variant to be
  accuracy-neutral-or-better under the frozen contract, the memory/MAC
  reduction would be a deployment argument for preferring it. That
  conditional does not move the present verdict one millimetre: an
  attractive-but-untestable proposition stays untested.

## 13. FINAL CLASSIFICATION

True group-wise correlation with `G` preserved into aggregation (U2) has no
single-component construction on the frozen P2A platform. The native axis is
operation-type replacement (signed 32-channel difference ->
group-pooled inner-product similarity), not `G=1 -> G>1`, and every
non-degenerate realization of it moves the operation type, the aggregation
input width (with -20,736 first-layer weights at G=8 against the frozen
397,954 total), and the sign convention together, with incomparable-shaped
initializations — while the only width-preserving realization (G=C=32,
group width 1) is a degenerate elementwise similarity that does not test
the grouping hypothesis and still needs sign surgery. ARM-W is
mechanistically non-equivalent (learned grouped projection of the difference
volume, `phase1/scripts/train_arm_w.py:1-12`, not grouped feature
correlation) and therefore silent on True GWC; D3's group-averaged
rescaling is a known constructional gap, not evidence. The recipe can be
frozen but that does not help: the graph change itself is multi-component.
No matched experiment is specified, no gates authorize training, and no
follow-up is launched by this document. Candidate C was already NOT
IDENTIFIABLE (CANDIDATE_C_SEPARABILITY.md Section 9); True GWC joins it as
not identifiable, with the branch remaining OPEN — an honest, valuable
result that forbids manufacturing a strained design to force a test.

GWC NOT IDENTIFIABLE — BRANCH REMAINS OPEN
