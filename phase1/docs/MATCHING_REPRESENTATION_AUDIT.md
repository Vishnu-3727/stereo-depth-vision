# MATCHING REPRESENTATION AUDIT

Read-only research. No training run, no file modified, no checkpoint created.
One new document only, as briefed.

## 1. Executive conclusion

**Yes — we are using an unnecessarily weak matching representation, and the
weakness is specific: our cost volume is a raw, unnormalized feature
difference, with no normalization anywhere in the matching path.**

- Current operation (source-verified): `cost_k(x) = left(x) - right(x-k)`,
  32 raw signed channels, `src/models/stereonet/cost_volume.py:108-111`
  (ARM-V uses `shift="right"`, `phase1/scripts/train_arm_v.py:131-132`).
  No L2 normalization, no BatchNorm anywhere
  (`phase1/scripts/train_arm_v.py:157` records the no-BN deviation;
  `src/models/stereonet/feature_extractor.py:37-46` shows norm-free layers).
- Consequence: a channel's contribution to the cost is proportional to its
  activation magnitude, not to its informativeness for correspondence. Nothing
  equalizes per-channel or per-pixel scale before the signed difference is
  taken, and the 3D aggregation applies the same 32 filter weights at every
  pixel (`src/models/stereonet/aggregation.py:31-35`), so it cannot undo a
  magnitude confound that varies per pixel.
- Every public matcher inspected normalizes the matching operation or the
  features feeding it: Fast-ACVNet `groupwise_correlation_norm` and
  `norm_correlation` (`reference/public_repos/Fast-ACVNet/models/submodule.py:124-150`),
  CoEx `AttentionCostVolume` L2 normalization
  (`reference/public_repos/OpenStereo/stereo/modeling/models/coex/coex_cost_processor.py:62-63`),
  IGEV `norm_correlation`
  (`reference/public_repos/OpenStereo/stereo/modeling/models/igev/submodule.py:180-184`).
  Ours is the only matcher in the comparison set with neither matching-time
  normalization nor normalization layers.
- The strongest isolated, untested, source-grounded fix is **L2-normalized
  feature difference**: unit-normalize each 32-dim feature vector per pixel
  (L and R independently, before shifting), then subtract exactly as now.
  Zero new parameters, identical tensor shapes, identical sign convention,
  no grouping (not ARM-W), disparity-dependent matching change (not ARM-Y).
  Specified in sections 6–7, gated in section 9, not implemented here.

What this audit does NOT claim: that normalized difference beats
correlation-style similarity in our regime. That is what the proposed ARM
would measure. The claim here is only that the unnormalized-difference
representation is evidentially the weakest in the comparison set and that a
clean one-variable fix exists.

## 2. Our current matching operation

ARM-V configuration: `StereoNetConfig(downsample_levels=3,
num_disparities=24, cost_volume_shift="right", regression_normalize=True)`
(`phase1/scripts/train_arm_v.py:131-132`).

- **L feature / R feature.** Shared siamese extractor, no norm layers, no
  activations in the 4→3-level downsample stack:
  `src/models/stereonet/feature_extractor.py:37-46`. ARM-V output:
  `(B, 32, H/8, W/8)` (stride 8; e.g. `(B,32,32,64)` for a 256×512 crop).
  Config defaults in `src/models/stereonet/stereonet.py:48-50`;
  forward wiring `src/models/stereonet/stereonet.py:126-127`.
- **Disparity shift.** `shift_right(right, k)`: pad k zeros on the left,
  slice `[..., :width]`
  (`src/models/stereonet/cost_volume.py:45-56`). Per-level pairing in
  `build_cost_volume` (`src/models/stereonet/cost_volume.py:101-119`):
  `anchor, other = left, shift_right(right, k)`.
- **Cost construction.** `method="subtract"` default
  (`src/models/stereonet/stereonet.py:53`):
  `levels.append(anchor - other)`
  (`src/models/stereonet/cost_volume.py:110-111`). So
  **`cost_k(x) = left(x) - right(x-k)`, k = 0..23**, left-frame indexing
  matching KITTI left-view ground truth. Stacked and permuted to
  **`(B, 32, 24, H, W)`** (`src/models/stereonet/cost_volume.py:118-119`).
- **Sign convention.** Signed difference; exact match gives 0, mismatch gives
  ±large. It is a **cost** (low = good), not a similarity. Channel semantics:
  32 raw signed per-channel differences, magnitudes unnormalized.
- **Aggregation.** 4× `Conv3d(32→32, 3×3×3)` + LeakyReLU(0.01), then
  `Conv3d(32→1)` and channel squeeze → `(B, 24, H, W)`
  (`src/models/stereonet/aggregation.py:26-40`). The first layer's input
  channel count (32) is wired from the cost-volume method in
  `src/models/stereonet/stereonet.py:100-104`.
- **Softmax / readout.** `soft_argmin` computes `softmax(-cost)` over dim=1
  (`src/models/stereonet/regression.py:49`), i.e. an explicit arg**min**:
  low cost becomes high weight. With ARM-V's `normalize=True`, cost is first
  standardized across the disparity axis
  (`src/models/stereonet/regression.py:47-48`). The negation in line 49 is
  what proves aggregation must output **cost** semantics. Full cost tensor is
  bilinearly upsampled to full resolution *before* argmin
  (`src/models/stereonet/regression.py:64-67`).
- **Refinement / loss.** Single full-resolution residual stage guided by the
  left RGB image (`src/models/stereonet/refinement.py:51-56`), added to the
  initial disparity with ReLU
  (`src/models/stereonet/stereonet.py:132-136`). Loss: masked smooth-L1,
  beta 1.0, valid = gt > 0 and gt < max_disparity
  (`phase1/scripts/train_arm_v.py:153,177`). Optimizer Adam lr 1e-3,
  cosine schedule, 200 epochs, batch 2
  (`phase1/scripts/train_arm_v.py:48-52,135-136`).

Cross-check on sign: the independent StereoNet reproduction reads out with
`softmax(+cost)` (`reference/public_repos/StereoNet/models/StereoNet.py:155-156`)
— its final Conv3d must therefore learn to emit *high* values at the true
match. Our `softmax(-cost)` is the opposite convention. Any similarity-valued
(similarity-high-at-match) proposal ported into our graph MUST be negated
(or the readout sign flipped) to remain correct; any difference-valued
proposal needs no sign change. This is load-bearing for section 8.

## 3. Public matching mechanisms

Source-verified facts only. README claims are excluded from this table.

| Repo | Mechanism | Exact operation | Shape | Normalization | Learned? | Our equivalent? |
|---|---|---|---|---|---|---|
| Fast-ACVNet `models/submodule.py:104-110` | unnormalized group-wise correlation | `cost = mean over channels-per-group of (fea1 * fea2)` | `(B, G, H, W)` per disparity; volume `(B, G, D, H, W)` (`:112-122`) | none (raw products) | No | No — ours is signed difference, not a product; ARM-W tested the grouped form and refuted it |
| Fast-ACVNet `models/submodule.py:124-132` | **L2-normalized group-wise correlation** | per-group vectors L2-normalized (`fea/(norm(fea,2,dim=2)+1e-05)`) for L and R independently, then elementwise product, mean over channels-per-group | same as above | **per-vector (per-group) L2, before matching, eps 1e-05** | No | No — closest public precedent for the section 6 proposal, but applied to products, not differences |
| Fast-ACVNet `models/submodule.py:148-150` | **full-vector normalized correlation (cosine)** | `mean over all C of (L/\|L\| * R/\|R\|)`, norm over dim=1, eps 1e-05 | `(B, 1, H, W)` per disparity; volume `(B, 1, D, H, W)` (`:152-161`) | **per-vector (full 32-ch) L2, before matching** | No | No — single-channel similarity; would change aggregation input 32→1 |
| Fast-ACVNet `models/submodule.py:90-101` | concatenation volume | `cat([L(x), R(x-k)], dim=1)` | `(B, 2C, D, H, W)` | none at matching time (features come from a BN network) | No (matching itself); learned 3D projection follows in `Fast_ACV.py:239,305` | Partly — our `method="concat"` builds the same layout (`cost_volume.py:112-113`) but is not the ARM-V default and doubles aggregation input channels |
| Fast-ACVNet `models/Fast_ACV.py:269-306` | attention concatenation volume (top-k) | normalized GWC volume → patch conv → `channelAtt` → hourglass → variance-weighted resampling → top-k=24 disparity samples → concat volume only at sampled disparities, weighted by attention | sparse `(B, 2C', k=24, H, W)`-equivalent | L2 inside `build_gwc_volume_norm`; attention weights are softmax probabilities | Yes (patch conv, hourglass, variance params `gamma`/`beta` `:203-204`) | No — two-stage schedule plus sparse candidates; not one intervention |
| AANet `nets/cost.py:22-29` | feature difference | `left(x)-right(x-k)` per channel | `(B, C, D, H, W)` | none at matching time (features from a BN/ResNet backbone, `nets/feature.py`, `nets/resnet.py`) | No | **Yes — same math as ours** (real search, unlike our frozen no-op baseline; ARM-V already uses the real `shift="right"` search) |
| AANet `nets/cost.py:31-38` | concatenation | `cat([L, R])` per disparity | `(B, 2C, D, H, W)` | as above | No | Same layout as our `method="concat"`; not our default |
| AANet `nets/cost.py:40-48` | correlation (mean product) | `mean over channels of (L*R)` | `(B, D, H, W)` (single channel) | none (raw products) | No | No — similarity semantics; AANet handles the sign explicitly (see below) |
| AANet `nets/estimation.py:16-19` + `nets/aanet.py:106` | sign convention switch | `cost if match_similarity else -cost`, then `softmax` | `(B, D, H, W)` | n/a | No | Convention evidence: difference/concat → negate before softmax; correlation → softmax directly. Proves sign handling is part of porting a similarity into a cost readout |
| MobileStereoNet `models/submodule.py:235-241` + `models/MSNet2D.py:141-157` | **learned similarity (interweave + 3D conv)** | interleave L/R channels pairwise (`L0,R0,L1,R1,…`), unsqueeze, `Conv3d(1→16→32→16)` with strides (8,4,2), `volume11` 1×1 conv → scalar per disparity | `(B, 1, D, H, W)` then squeezed | BatchNorm inside the learned matcher (`MSNet2D.py:73-84`) | **Yes** (~tens of k params in the 3D matcher + preconv `preconv11` `:65-71`) | No — rewrites cost construction and collapses 32 channels to 1; changes aggregation input shape |
| OpenStereo CoEx `coex_cost_processor.py:10-35` | group inner-product volume | `(x * unfolded_y).sum(2)` over channels-per-group, flipped | `(B, G, D, H, W)` | none in this block | No | Group-correlation family; ARM-W-adjacent when G < C |
| OpenStereo CoEx `coex_cost_processor.py:62-63` | **L2-normalized correlation inside AttentionCostVolume** | `costVolume(x_/‖x_‖₂, y_/‖y_‖₂)` after a learned `conv`+`desc` projection (`:50-56`); optional learned per-channel weights (`:46-48,58-61`) | `(B, head, D, H, W)` | **per-vector L2 over channel dim (dim=1)** | Yes (projection convs + optional weights) | No — normalized similarity, not a difference; supports the normalize-before-matching principle |
| OpenStereo `cost_volume/cost_volume.py:32-41` | correlation volume (mean product) | `mean over C of (L*R)` | `(B, D, H, W)` | none | No | No — single-channel similarity |
| OpenStereo `cost_volume/cost_volume.py:44-56` | subtraction volume (`compute_volume`) | `L - R` per channel with side-aware indexing | `(B, C, D, H, W)` | none | No | **Yes — same math as ours** |
| OpenStereo `cost_volume/cost_volume.py:108-117` | L1-norm difference volume (`build_sub_volume`) | `‖L(x)-R(x-k)‖₁` over channels | `(B, D, H, W)` single channel | none (norm itself is the reduction) | No | No — collapses channels via L1; changes aggregation input 32→1; unsigned (≥0) cost, sign needs no flip but channel identity is lost |
| OpenStereo `cost_volume/cost_volume.py:120-169` | learned interlaced volume | interleave L/R + `BasicConv3d` stack + 1×1 to `num_features` per disparity | `(B, F, D, H, W)` | BatchNorm3d/2d inside | **Yes** | No — same family as MobileStereoNet learned similarity |
| OpenStereo IGEV `models/igev/submodule.py:158-177` | unnormalized group-wise correlation | identical math to Fast-ACVNet `:104-110` | `(B, G, D, H, W)` | none | No | ARM-W family |
| OpenStereo IGEV `models/igev/submodule.py:180-196` | normalized correlation | same math as Fast-ACVNet `:148-161` (eps 1e-05) | `(B, 1, D, H, W)` | per-vector L2 | No | Single-channel similarity; aggregation-shape change |
| OpenStereo IGEV `models/igev/submodule.py:199-213` | sum-product correlation | `sum over C of (L*R)` (vs mean) | `(B, 1, D, H, W)` | none | No | No — unnormalized similarity, scale differs by factor C from the mean form |
| OpenStereo IGEV `models/igev/submodule.py:216-227` | concat volume | `cat([L(x), R(x-k)])` | `(B, 2C, D, H, W)` | none (norms live in backbone) | No | Same layout as our non-default concat |
| OpenStereo IGEV `models/igev/submodule.py:237-250` (`FeatureAtt`); also `models/igev/igev_stereo.py:122-123,156-160`; Selective-IGEV `core/igev_stereo.py:129-130,165-170` | left/feature-guided channel gate on the volume | `sigmoid(feat_att(feat)).unsqueeze(2) * cv` — character-identical to CoEx `channelAtt` | `(B, C, D, H, W)` preserved | BN inside `BasicConv` | Yes (~1k) | **Already tested — this is ARM-Y**, refuted (mean 3.4075053 vs ARM-V 1.7727436) |
| LightStereo `lightstereo.py:51` + `aggregation.py:39-57` | correlation volume + 2D aggregation with left attention | `correlation_volume` (mean product) then `AttentionModule` gates inside a 2D (D-as-channels) hourglass | 2D `(B, D-as-C, H, W)` | BN inside backbone and residual blocks | Yes | No — 2D aggregation is a different architecture, not an intervention |
| LightStereo `aggregation.py:104-134` | strip-convolution spatial attention gate | `attn = conv3(attn + strip-conv branches); return attn * cost` (**no sigmoid** — unbounded gate, unlike ARM-Y's (0,1)) | preserves gated tensor shape | BN in surrounding blocks | Yes | Not our equivalent (unbounded, multi-scale strip kernels); still a post-volume gate, so ARM-Y-adjacent — excluded by the same rule |
| Selective-IGEV `core/geometry.py:62-69` (+ OpenStereo IGEV/StereoBase copies) | all-pairs einsum correlation pyramid | `corr = einsum('aijk,aijh->ajkh')`, pooled into a pyramid, sampled around current disparity inside a GRU loop | `(B, H, W, levels×(2r+1)×(C+1))` lookup features | GroupNorm/BatchNorm in encoders (`core/extractor.py`) | Yes (motion encoder + GRUs) | No — inseparable from the iterative architecture |
| Selective-IGEV `core/update.py:16-45` | channel + spatial attention on GRU state | SE-style channel gate and 7×7 spatial gate applied to recurrent hidden state, not to a matching cost | n/a | none in gate | Yes | No — post-readout recurrent attention, not a matching representation |
| StereoNet neka-nat `models/StereoNet.py:143-147` | feature difference with real search | `ref[...,i:] - target[...,:-i]` | `(B, 32, D, H, W)` | **BatchNorm2d/3d everywhere** (`:10-24`, `:129-133`) | No | Same matching math as ours + BN; readout sign is `softmax(+cost)` (`:155-156`), opposite to ours |
| pareshvpk/hailostereovision | — | — | — | — | — | **NOT INSPECTED / UNVERIFIED** — not available locally; no claim made |

## 4. ARM-W / ARM-Y exclusion analysis (explicit, no hand-waving)

**ARM-W (group-wise cost, ~1.776 EPE, REFUTED) — what it was, exactly.**
`CostVolume` with `groups > 0` applies one grouped `Conv2d(channels→groups, 1)`
per disparity slice and stacks group channels
(`src/models/stereonet/cost_volume.py:160-171`); aggregation input becomes
`cost_volume_groups` channels (`src/models/stereonet/stereonet.py:100-104`).
It is a channel *compression*: 32 feature channels → G correlation channels,
discarding per-channel identity to save memory. Consequence for this audit:
**any mechanism whose definition requires grouping, channel compression, a
group projection, or a collapsed candidate layout OVERLAPS ARM-W and is
excluded from consideration**, no matter how interesting. That rules out:
unnormalized group-wise correlation (Fast-ACVNet `:104-110`, IGEV `:158-177`,
CoEx `:10-35` with G < C), the concat-then-project family where the
projection compresses channels, and every single-channel reduction (mean/sum
correlation, L1 difference, scalar learned similarity) *as a same-shape
replacement* — each changes the aggregation input from 32 channels to 1 (or
G), i.e. it re-runs the ARM-W compression axis with a different reduction
function. The section 6 choice (normalized difference) uses **no groups, no
projection, no reduction**: 32 channels in, 32 channels out. It is disjoint
from ARM-W by construction, not by assertion.

**ARM-Y (left-image-guided cost-volume excitation, mean 3.4075053, REFUTED) —
what it was, exactly.** Gate from left features alone through two 1×1 convs
+ sigmoid, broadcast over disparity, multiplied onto the already-built volume
(`src/models/stereonet/excitation.py:47-49`; result record
`phase1/docs/ARM_Y_RESULT.md:22-38`). It is channel-wise, pixel-wise,
**disparity-independent by construction**: it reweights *which channels* to
trust, never *which disparity* matches, and it operates **after** the
matching operation. Consequence: **any mechanism of the form
existing-volume → gate → aggregation OVERLAPS ARM-Y**, including every
`channelAtt`/`FeatureAtt` copy (Fast-ACVNet `Fast_ACV.py:88-102`, CoEx
`:68-81`, IGEV `submodule.py:237-250`), LightStereo's `AttentionModule`
(gate shape differs — unbounded strip-conv gate, no sigmoid — but position
and role are identical: post-volume, disparity-independent), and the GRU
channel/spatial attention (post-readout by an even larger margin).
The section 6 choice operates **before/within** the matching operation
itself (on the L/R features, per disparity shift) and its output **is**
disparity-dependent (each candidate's cost changes differently because the
shifted R operand differs). It adds no gate module, no learned parameters,
no left-features input. It is disjoint from ARM-Y by position (pre-volume vs
post-volume), by dependence (disparity-dependent vs disparity-independent),
and by parameterization (zero learned weights vs 1,056).

**The isolation test applied.** BASE = ARM-V. ONE VARIABLE = per-pixel
unit-normalization of L and R before the existing shift-and-subtract.
Everything else — 24 candidates, 8 px spacing, 184 px range, shift right,
32-channel `(B,32,24,H,W)` volume, existing 3D aggregation, existing
refinement, existing loss/optimizer/schedule/epochs/batch/crop/jitter/split,
frozen evaluator — IDENTICAL. Passes. Every similarity-valued alternative
fails or strains the test on sign (needs negation or a readout flip as a
second variable); every channel-collapsing alternative fails it on
aggregation input shape (first Conv3d weight shape changes); every gate
fails it on ARM-Y overlap. The choice below is the only candidate surveyed
that passes all three screens simultaneously.

## 5. Candidate mechanism matrix

| Mechanism | Source evidence | Changes information content? | Params | Compute | Isolatable? | ARM overlap | Research value |
|---|---|---|---|---|---|---|---|
| **A. L2-normalized feature difference (per-pixel unit-norm, then subtract)** | Principle source-verified: Fast-ACVNet `submodule.py:124-150` (normalize-before-matching, eps 1e-05); CoEx `coex_cost_processor.py:62-63` (normalize before correlation); IGEV `submodule.py:180-184`. Exact-to-difference composition is a transfer, flagged as such in section 10 | Yes — removes per-pixel activation-magnitude confound while preserving signed per-channel differences; vector direction kept, scale equalized | **0** | +1 norm pass over each feature map (B×32×H×W, 1/8 res) + broadcast divide; <1% of forward cost | **Yes** — shape, sign, aggregation, loss all unchanged | None (no groups; disparity-dependent; gateless) | **Highest.** Only candidate that (i) attacks the documented bottleneck (unnormalized magnitude), (ii) keeps cost semantics so no sign change is needed, (iii) costs nothing, (iv) is disjoint from both closed arms. Evidence is principle-level, not instance-level — stated honestly |
| B. Unnormalized per-channel product (Hadamard similarity, 32 ch preserved) | Shape-preserving variant of AANet `nets/cost.py:40-48` (mean product) and CoEx `:10-35` with group=C; sign handling per AANet `nets/estimation.py:16-19` | Yes — difference→similarity changes what low/high means | 0 | ~same as subtraction | **Marginal** — requires negating the volume (or flipping readout sign) as part of the intervention; that is arguably a second variable, and the sign choice needs its own justification | Resembles group-correlation with G=32; separable in letter but not in spirit from ARM-W | Medium. Real information change, but sign surgery + ARM-W resemblance make it a worse experiment than A |
| C. Full-vector cosine similarity (single channel) | Fast-ACVNet `submodule.py:148-161`; IGEV `submodule.py:180-196` | Yes — collapses 32 signed channels to 1 bounded similarity | 0 | small | **No** — aggregation input 32→1 changes first Conv3d weight shape; also needs sign negation | ARM-W axis (channel collapse) | Low as an isolated arm. Would need a shape adapter, which is a second change |
| D. Normalized group-wise correlation, G<32 | Fast-ACVNet `submodule.py:124-145`; IGEV `:158-177` | Yes, but coupled with compression | 0 | small | No — same two defects as C (shape + sign) | **Direct ARM-W overlap** (grouping + compression) | None. Closed |
| E. Concatenation + learned projection | Fast-ACVNet `submodule.py:90-101` + `Fast_ACV.py:239,305`; IGEV `submodule.py:216-227`; our own non-default layout `cost_volume.py:112-113` | Yes — defers similarity to learned 3D filters | +first-layer 3D filters (in 32→64: ~55k, +14%) | doubles volume memory `(B,64,24,H,W)` | No — aggregation input 32→64; memory ×2; not one variable | Touches the ARM-W channel-count axis | Low. Over budget, under-isolated |
| F. Learned interweave similarity (MobileStereoNet / InterlacedVolume) | `mobilestereonet/models/MSNet2D.py:141-157` + `submodule.py:235-241`; OpenStereo `cost_volume.py:120-169` | Yes — arbitrary learned per-disparity scalar | +~10-35k matcher params, plus preconv changes | per-disparity 3D conv loop, large | No — rewrites cost construction, collapses to 1 channel, needs different aggregation | Both axes (channels + architecture) | None as an ARM. Architecture rewrite |
| G. L1-norm difference (single channel) | OpenStereo `cost_volume.py:108-117` | Destroys information (channel identity + sign) | 0 | small | No — 32→1 shape change | ARM-W axis | None. Strictly less information than A with the same experiment cost |
| H. Any post-volume gate (channelAtt / FeatureAtt / AttentionModule / GRU attention) | CoEx `:68-81`; Fast-ACVNet `Fast_ACV.py:88-102`; IGEV `:237-250`; LightStereo `aggregation.py:104-134`; Selective `update.py:16-45` | Adds conditioning, not matching information | ~0.3-1% | small | Technically insertable, but — | **Direct ARM-Y overlap** | None. Closed by refutation |
| I. Top-k / attention-weighted sparse candidates | Fast-ACVNet `Fast_ACV.py:269-306` | Changes candidate layout itself | Large (hourglass + variance params) | Large | No — two-stage schedule, sparse layout, learned sampler | Neither arm, but not one variable | None as an ARM |

Research-value notes are evidence-based: only A is simultaneously (a)
motivated by a source-verified cross-repo regularity (normalize before
matching), (b) expressible as one variable in our graph, and (c) outside both
closed arms. B–I each fail at least one of those on source grounds stated in
the Source evidence column.

## 6. Best untested mechanism — choose EXACTLY ONE

**L2-normalized feature difference (per-pixel unit-norm of L and R
independently, before shifting, then the existing shift-right subtraction).**

- **Exact operation.** For each pixel x in each frame independently:
  `L̂(x) = L(x) / max(‖L(x)‖₂, ε)`, `R̂(x) = R(x) / max(‖R(x)‖₂, ε)`,
  norm over the 32 channel dims, `ε = 1e-05` (the Fast-ACVNet/IGEV value).
  Then per candidate k = 0..23:
  **`cost_k(x) = L̂(x) − R̂(x−k)`** (shift-right indexing, unchanged).
  At k = 0: `L̂ − R̂`. No other operator, no learned weight, no gate.
- **Tensor flow.** `(B,32,H,W)` L/R → per-pixel norms `(B,1,H,W)` →
  broadcast divide → same-shape `(B,32,H,W)` L̂/R̂ → existing
  `build_cost_volume(method="subtract", shift="right")` → `(B,32,24,H,W)`.
  Aggregation input shape bitwise identical to ARM-V.
- **Sign.** Unchanged: exact match → 0 vector; mismatch → signed nonzero.
  Low-cost-means-match preserved, so `softmax(-cost)` in
  `src/models/stereonet/regression.py:49` stays correct **with no negation
  and no readout change**. This is the decisive advantage over every
  similarity-valued candidate: there is no sign ambiguity to resolve by
  fiat (see section 8 for the one residual ordering ambiguity, which is
  explicitly listed, not hidden).
- **Normalization.** Per-vector (per-pixel, over channels), per-frame
  independent, applied before shifting. Global or per-channel normalizations
  are NOT proposed (they would either leak statistics across the batch or
  destroy relative channel scales the aggregation may use).
- **Why it should affect correspondence.** Two linked reasons, both grounded
  in our source, not in analogy: (1) with no BN anywhere, raw channel
  magnitudes are arbitrary scale artifacts of random init + unnormalized
  conv stacks; the difference volume lets the loudest channel decide the
  cost regardless of which channels actually discriminate the match —
  normalization removes exactly that confound while keeping every channel's
  sign and relative direction; (2) unlike ARM-Y's post-hoc reweighting,
  this changes the matching evidence *at the disparity level*: each
  candidate's 32-dim cost vector becomes a direction comparison rather than
  a magnitude contest, so the aggregation's learned 3×3×3 filters see cost
  curves whose minima reflect alignment, not amplitude. Expected effect
  direction: sharper, better-calibrated cost minima, most visible at
  texture edges and in previously magnitude-dominated regions; expected
  risk: in genuinely textureless areas unit-normalization amplifies noise
  direction — the experiment measures whether the trade helps on balance.
- **Why it is not ARM-W.** No groups, no group projection, no channel
  reduction, no candidate-layout change. Channel count stays 32 throughout;
  the `groups` code path (`cost_volume.py:160-171`) is untouched and
  disabled (`cost_volume_groups=0`).
- **Why it is not ARM-Y.** No gate module, no sigmoid, no left-features
  input, no learned parameters, no post-volume multiplication. The change is
  disparity-dependent (each k sees a differently shifted R̂) and lives
  inside the matching operation, before the volume exists — the complement
  of ARM-Y's disparity-independent post-volume gate
  (`excitation.py:47-49`).
- **Expected param delta.** Exactly **0** (+0.0% on ~398k). No new module.
- **Expected compute/memory delta.** One L2-norm + one broadcast divide per
  frame at 1/8 resolution: ~2·B·32·H·W multiply-adds plus a sqrt per pixel —
  under 1% of forward FLOPs (aggregation alone does 4 full 3D-conv passes
  over the 24-deep volume; refinement dominates MACs per the architecture
  notes). Memory: two transient `(B,1,H,W)` norm maps + normalized copies;
  no volume-size change. No custom CUDA, no candidate change, no
  aggregation-shape change.
- **Failure modes.** (1) Noise amplification on textureless pixels
  (unit-norm of near-zero vectors is direction noise; ε floors the blowup
  but does not remove it). (2) Interaction with `regression_normalize=True`
  (already standardizes across D — the two normalizations compose, and if
  both compress dynamic range the softmax could sharpen or flatten
  unexpectedly; this is held fixed, not tuned). (3) Gain-jitter interaction:
  per-image scalar jitter (`train_arm_v.py:74-76`) is absorbed by
  unit-normalization, slightly changing the augmentation's effective
  strength on the matching path but not on the refinement path — recorded,
  not adjusted. (4) Small-data regime (160 KITTI scenes): unlike ARM-Y
  there is no new capacity to overfit, so this risk is strictly smaller
  than the refuted arm's.

## 7. Exact ARM specification (SPEC ONLY — DO NOT IMPLEMENT)

```
BASE          = ARM-V (phase1/scripts/train_arm_v.py, seed discipline 0/1/2)
INTERVENTION  = per-pixel L2 unit-normalization of left and right feature
                maps independently (eps 1e-05), inserted inside the matching
                path before the existing shift-right subtraction; the
                subtraction, shift, and everything downstream are untouched.
```

Everything else fixed:

| Held fixed | Value | Source |
|---|---|---|
| Disparity candidates | 24 | `train_arm_v.py:131-132` |
| Spacing / stride | 8 px full-res (`downsample_levels=3`) | `train_arm_v.py:131-132` |
| Represented range | 184 px (`max_disparity_px`, 23×8) | `stereonet.py:76-78`, `train_arm_v.py:138-139` |
| Shift | `right`: `cost_k(x) = L̂(x) − R̂(x−k)` | `cost_volume.py:45-56,101-119` |
| Cost method / shape | `subtract`, `(B,32,24,H,W)` | `cost_volume.py:110-119` |
| Regression normalization | `True` (standardize across D, then `softmax(-cost)`) | `regression.py:47-49`, `train_arm_v.py:131-132` |
| Aggregation | 4× Conv3d(32→32) + Conv3d(32→1), unchanged | `aggregation.py:26-40` |
| Refinement | one stage, dilations (1,2,4,8,1,1), unchanged | `stereonet.py:58`, `refinement.py:47-56` |
| Loss | masked smooth L1, beta 1.0, valid = gt>0 and gt<max_disparity | `train_arm_v.py:153,177` |
| Optimizer / LR / schedule | Adam (0.9,0.999), 1e-3, cosine to 0 | `train_arm_v.py:135-136` |
| Epochs / batch | 200 / 2 | `train_arm_v.py:48-49` |
| Crop / augmentation | random 256×512; independent gain jitter σ=0.1; normalize wrapper; no flip | `train_arm_v.py:47,56-76,154` |
| Split | hailo_calib 0–159 train; hailo_val 160–199 eval; no overlap | `train_arm_v.py:118-123` |
| Seeds | 0, 1, 2, all to completion before any verdict read | preregistration pattern (`ARM_Y_PREREGISTRATION.md:236-243`) |
| Evaluation | frozen contract only: KITTI 2015 `_10`, disp_occ_0, top-left 368×1232 crop, GT/256, valid = gt>0, pooled EPE/D1, raw output, ReLU in graph, exactly 3,802,797 valid pixels | `phase1/harness/frozen_eval.py:28-47` |
| BatchNorm | none anywhere | `train_arm_v.py:157` |

Conceptual tensor flow (the only new lines are the two normalizations):

```
left_features   (B, 32, H, W)     # stereonet.py:126, unchanged
right_features  (B, 32, H, W)     # stereonet.py:127, unchanged
--- INTERVENTION (and nothing else) ---
Ln = L / max(||L||_2 over dim=1, 1e-05)     (B, 32, H, W)
Rn = R / max(||R||_2 over dim=1, 1e-05)     (B, 32, H, W)
volume = build_cost_volume(Ln, Rn, num_disparities=24,
                           method="subtract", shift="right")   # (B,32,24,H,W)
---
cost   = Aggregation(volume)                  # (B,24,H,W), unchanged
disp_0 = DisparityRegression(cost, size)       # unchanged
disp   = relu(disp_0 + Refinement(disp_0, left))  # unchanged
```

## 8. Ambiguities that must be resolved before training

Listed, not silently picked. The preregistration must freeze exactly one row.

1. **Normalize before vs after shifting.** (a) Before (recommended):
   normalize L and R in their own pixel coordinates, then shift R̂ —
   each norm reflects a real observed feature vector. (b) After: shift raw
   R, then normalize shifted vectors — zero-padded border columns produce
   zero/near-zero vectors whose normalization is ε-dominated noise, and the
   same R pixel gets different norms at different k. These are materially
   different at borders. Exactly one source-faithful formulation: (a),
   because every public normalization (`Fast_ACV.py:269`,
   `coex_cost_processor.py:62-63`, IGEV `:156-158`) normalizes features
   before volume construction, never shifted slices.
2. **Epsilon value.** Candidates: 1e-05 (Fast-ACVNet `:130`, IGEV `:182`),
   1e-06 (our `regression.py:48` uses 1e-6 for its own std), 1e-12.
   Source-faithful: **1e-05**. Must be frozen, not swept.
3. **Norm axis.** Candidates: per-pixel over channels (recommended —
   matches `torch.norm(fea,2,1)` in all three public sites); per-channel
   over space (destroys spatial contrast, NOT proposed); global (leaks
   batch/scene statistics, NOT proposed). Only the per-pixel form is
   source-faithful.
4. **Independent vs joint L/R normalization.** Candidates: independent
   (recommended — matches public code, which normalizes each input
   separately); joint scale from L applied to both (would preserve
   L-vs-R amplitude differences, but no public site does this and it
   reintroduces the magnitude confound across frames). Source-faithful:
   independent.
5. **Cosine similarity vs cosine distance vs normalized difference — do NOT
   conflate.** `cos(L,R)` (high-at-match, needs negation before our
   readout), `1 − cos(L,R)` (low-at-match, needs single-channel volume →
   shape change), and `L̂ − R̂` (low-at-match, shape preserved) are three
   different interventions with different signs and shapes. This audit
   proposes ONLY the third. The first two are listed here as considered
   and rejected (sign surgery + shape change, section 5 rows B–C).
6. **Whether to negate before softmax.** For the proposed formulation:
   NO negation — it is already cost-valued. Stated explicitly so no
   implementer "helpfully" adds one by analogy with AANet's similarity
   path (`nets/estimation.py:16-17`, which applies to similarities only).
7. **Interaction with `regression_normalize=True`.** Held ON (it is part of
   ARM-V). NOT ablated, NOT retuned — ablating it would be a second
   variable.
8. **Zero-vector handling.** Pixels whose raw feature norm is below ε keep
   a scaled raw vector (`v/ε`), not a unit vector; padded/shift borders
   are therefore large-magnitude, not NaN. Implementation must use
   `max(norm, ε)` division (as public code does), never raw `v/norm`.

## 9. Go / No-Go

**Go condition (all must hold; otherwise No-Go):**

1. The preregistration freezes exactly one formulation from section 8
   (recommended: before-shift, independent, per-pixel-over-channels,
   ε=1e-05, no negation, `regression_normalize` held on) with the section 7
   table verbatim, before any code is written.
2. A smoke test proves shape preservation (`(B,32,24,H,W)`), sign
   preservation (identical-zero at identical input), gradient flow to the
   feature extractor, zero new parameters (total stays 397,954), and no BN
   introduced.
3. Seeds 0, 1, 2 run to completion under the frozen contract
   (`frozen_eval.py:28-47`; 3,802,797 valid pixels); checkpoint selection
   uses the train-time curve only; no seed is dropped, rerun, or replaced.
4. Verdict uses the inherited ARM-V-anchored rule, fixed now and not
   fitted to future numbers: ARM-V mean 1.7727436, spread 0.3410304,
   CONFIRMED gate 1.4317132 (= mean − spread). Mean below 1.4317132 →
   CONFIRMED; between 1.4317132 and 1.7727436 → INCONCLUSIVE (does not
   replace ARM-V, no follow-up on this axis without tightening the
   baseline); at or above 1.7727436 → REFUTED and closed. The threshold is
   defensible because it reproduces the recorded ARM-X verdict
   (1.6234090 → not confirmed) without modification.

**No-Go conditions (any one blocks the arm):** implementation requires
changing the aggregation input channels, the readout sign, the candidate
layout, the loss/optimizer/schedule, or the normalization state of the
network; or the smoke test shows a parameter delta ≠ 0; or the brief's
hard prohibitions (training here, editing existing files, extra output
files) cannot be met. A regression is recorded as REFUTED, never
reinterpreted.

## 10. Sources

**Source-verified facts** (actually read at the stated lines):

- Ours: `src/models/stereonet/stereonet.py` (full, :1-161);
  `src/models/stereonet/feature_extractor.py` (full, :1-56);
  `src/models/stereonet/cost_volume.py` (full, :1-179);
  `src/models/stereonet/aggregation.py` (full, :1-40);
  `src/models/stereonet/regression.py` (full, :1-69);
  `src/models/stereonet/refinement.py` (full, :1-56);
  `src/models/stereonet/excitation.py` (full, :1-49);
  `src/models/stereonet/blocks.py` (full, :1-49);
  `phase1/scripts/train_arm_v.py` (full, :1-219);
  `phase1/harness/frozen_eval.py` (:1-60, contract table :28-47).
- Prior audits (read, not repeated): `phase1/docs/PUBLIC_STEREO_RESEARCH_AUDIT.md`
  (full); `phase1/docs/AUDIT_STAGE1_BASELINE_FACTS.md` (full);
  `phase1/docs/ARM_Y_RESULT.md` (full); `phase1/docs/ARM_Y_PREREGISTRATION.md` (full).
- Fast-ACVNet: `models/submodule.py` (full, incl. `:90-161`);
  `models/Fast_ACV.py` (full, incl. `channelAtt` `:88-102`, GWC-norm use
  `:269`, top-k `:292-306`).
- AANet: `nets/cost.py` (full, `:19-55`); `nets/estimation.py` (full,
  sign switch `:16-19`); `nets/aanet.py` (`:70-129`, similarity flag `:106`).
- MobileStereoNet: `models/submodule.py` (full, interleave `:235-241`,
  GWC `:244-263`); `models/MSNet2D.py` (full, learned matcher `:73-84`,
  `:141-157`).
- StereoNet (neka-nat): `models/StereoNet.py` (full; difference `:143-147`,
  `softmax(+cost)` `:155-156`, BN `:10-24`, cascade `:83-105`).
- OpenStereo: `stereo/modeling/models/coex/coex_cost_processor.py` (full;
  `CostVolume` `:10-35`, `AttentionCostVolume` norm `:62-63`, `channelAtt`
  `:68-81`); `.../coex/submodule.py` (full, `BasicConv` `:43-70`);
  `.../lightstereo/aggregation.py` (full; `AttentionModule` `:104-134`);
  `.../lightstereo/lightstereo.py` (full; correlation use `:51`);
  `stereo/modeling/cost_volume/cost_volume.py` (full; `correlation_volume`
  `:32-41`, `compute_volume` `:44-56`, `build_sub_volume` `:108-117`,
  `InterlacedVolume` `:120-169`); `.../models/igev/submodule.py`
  (`:140-265`: group/norm/sum correlation, concat, `FeatureAtt :237-250`);
  `.../models/igev/igev_stereo.py` (`:100-159`: GWC + FeatureAtt use);
  Selective-IGEV `core/update.py` (full, `:16-45` attention);
  Selective-IGEV `core/igev_stereo.py` (`:120-199`: GWC + FeatureAtt `:167-169`).

**Repository claims** (read but NOT relied on as evidence; all public
accuracy numbers are NON-COMPARABLE to our frozen protocol): AANet
`MODEL_ZOO.md` (StereoNet-AA SceneFlow claim); neka-nat `README.md`
(self-contradictory 1.32 attribution); OpenStereo model-zoo docs; any
KITTI/SceneFlow leaderboard figure in the public clones.

**Inference** (my synthesis, not source fact): that unnormalized magnitude
dominance is a binding constraint in our regime (supported by the
cross-repo normalization regularity + our no-BN state, but not directly
measured — the proposed ARM is the measurement); that normalized
difference preserves enough directional signal to beat raw difference
here; that textureless-noise amplification will not dominate on balance.
Section 6 marks these as expectations to be tested, not findings.

**NOT INSPECTED:** pareshvpk/hailostereovision (not available locally —
UNVERIFIED, no claim made); HITNet; StereoBase/IGEV++ internals beyond
grep-level confirmation of `FeatureAtt`/einsum-correlation presence
(`stereobase/gru_blocks.py`, `stereobase/hourglass.py` — grep hits only,
not line-verified reads); Selective-IGEV `core/geometry.py` and OpenStereo
`igev/geometry.py` (grep-level only for the einsum correlation).

---

NEXT MOVE:
Preregister ARM-Z as ARM-V plus per-pixel L2 unit-normalization of the left and right feature maps (eps 1e-05, before shifting) feeding the unchanged shift-right subtraction, with the section 9 thresholds frozen in writing before any code is written.
