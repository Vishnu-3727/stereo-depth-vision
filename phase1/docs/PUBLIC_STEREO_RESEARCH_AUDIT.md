# PUBLIC STEREO RESEARCH AUDIT

Research reconnaissance only. No training was run, no model file was modified, no
leaderboard entry was touched, no new phase was created. The purpose is to decide
whether our remaining error is a StereoNet architectural ceiling or an untested
mechanism, and to name exactly one candidate intervention.

Every claim about a public repository below is anchored to a file and line in a
shallow clone under `reference/public_repos/` (gitignored by the `reference/*`
rule). Claims that could not be verified from code are marked NOT INSPECTED or
UNVERIFIED CLAIM.

Baseline facts about our own model, with citations, are in
`phase1/docs/AUDIT_STAGE1_BASELINE_FACTS.md`.

---

## 1. Executive finding

- We are **not** at a StereoNet architectural ceiling. There is a specific,
  small, well-evidenced mechanism we have never tested.
- That mechanism is **left-image-guided cost-volume excitation**: gate the cost
  volume channel-wise, per pixel, with a function of the left image features,
  before aggregation.
- It appears in at least four independent public implementations under three
  different names, with near-identical code: CoEx `channelAtt`
  (`OpenStereo/stereo/modeling/models/coex/coex_cost_processor.py:68-81`),
  Fast-ACVNet `channelAtt` (`Fast-ACVNet/models/Fast_ACV.py:87-101`),
  LightStereo `AttentionModule` under the `LEFT_ATT` flag
  (`OpenStereo/stereo/modeling/models/lightstereo/aggregation.py:104-134`),
  and the IGEV/StereoBase/FoundationStereo family (17 model directories in
  OpenStereo reference `channelAtt`/`feature_att`).
- It is cheap enough for our regime: two 1x1 convolutions, about **1.1k
  parameters**, roughly **+0.3%** on our 397,954.
- It is **not** what ARM-W tested. ARM-W replaced the 32 feature channels with
  G group-correlation channels — a compression that *removes* information.
  Excitation keeps all 32 channels and *adds* conditioning information.
- It is **not** what ARM-X tested. ARM-X changed the loss only; the forward
  graph was untouched.
- The second-strongest untested finding is that every public matcher we
  inspected **normalizes** the matching operation (L2-normalized correlation, or
  BatchNorm on the features). Ours does neither: raw unnormalized feature
  subtraction, no BN anywhere (a deliberate, recorded deviation inherited from
  the BN-folded export).
- The independent StereoNet reproduction (neka-nat) differs from ours on six
  axes at once, and its headline number is **not comparable** to our protocol —
  different dataset, different crop anchor, and its own README contradicts
  itself about which model scored 1.32.
- Adaptive aggregation (AANet) and iterative GRU refinement (IGEV) are both
  well-evidenced but **not isolatable** for us: the first needs compiled
  deformable-convolution CUDA ops, the second is a different architecture, not
  an intervention.
- Recommendation: one ARM, one intervention, cost-volume excitation. Specified
  in section 8, gated in section 9, not implemented here.

---

## 2. Our architecture baseline

ARM-V configuration (`phase1/scripts/train_arm_v.py:131-132`):
`downsample_levels=3, num_disparities=24, cost_volume_shift="right",
regression_normalize=True`.

```
left RGB (B,3,256,512)   right RGB (B,3,256,512)
        |                        |
        +-- shared FeatureExtractor (no norm layers, no activations in the
        |   downsample stack) -- src/models/stereonet/feature_extractor.py:37-46
        v                        v
left_feat (B,32,32,64)   right_feat (B,32,32,64)          [stride 8]
        \                        /
         \   build_cost_volume(method="subtract", shift="right")
          \  cost_k(x) = left(x) - right(x-k), k = 0..23
           v  src/models/stereonet/cost_volume.py:96-120
        volume (B,32,24,32,64)          <-- 4D cost volume, 32 raw channels
             |
             |  Aggregation: 4x Conv3d(32->32,3x3x3)+LeakyReLU, then Conv3d(32->1)
             |  src/models/stereonet/aggregation.py:26-40
             v
        cost (B,24,32,64)
             |
             |  DisparityRegression: upsample the WHOLE cost tensor to full res,
             |  then normalize across D, then softmax(-cost), then index-weighted sum
             |  src/models/stereonet/regression.py:59-67
             v
        disparity_initial (B,1,256,512)   [units: candidates, spacing 8 px]
             |
             |  Refinement: concat with left RGB, Conv2d(4->32), 6 residual blocks
             |  dilations (1,2,4,8,1,1), Conv2d(32->1) residual
             |  src/models/stereonet/refinement.py:47-56
             v
        disparity = relu(initial + residual)   src/models/stereonet/stereonet.py:122-125
```

### Likely information bottleneck

The bottleneck is **not** the number of candidates (ARM-V already went to 24 at
8 px) and **not** the supervision signal (ARM-X showed that has gradient but
limited headroom). It is the **matching representation itself**, at the point
marked `volume (B,32,24,32,64)`:

1. The cost is a raw, unnormalized feature difference
   (`src/models/stereonet/cost_volume.py:110-111`). Its magnitude is dominated
   by whichever channels happen to have large activations, not by which channels
   are *informative* at that pixel. Every public matcher we inspected normalizes
   here; we do not, and we have no BatchNorm to compensate
   (`phase1/scripts/train_arm_v.py:157` records this as a known deviation).
2. Aggregation treats all 32 channels identically at every pixel
   (`src/models/stereonet/aggregation.py:31-35`). A textureless road pixel and a
   high-gradient pole edge get the same channel weighting. Nothing in the graph
   ever tells the cost volume what kind of surface it is looking at.
3. The left image enters the network exactly once as guidance — in refinement,
   *after* the disparity has already been read out
   (`src/models/stereonet/refinement.py:51-53`). By then the cost curve is
   collapsed to a scalar per pixel and the matching evidence is gone. Refinement
   can smooth an error; it cannot recover a cost curve that peaked on the wrong
   candidate.

That third point is the sharpest one. Every strong public model injects
left-image context into the cost volume *before* aggregation. We inject it only
after readout.

---

## 3. Repository-by-repository audit

| Repo/model | Matching | Aggregation | Refinement | Training/data | Key mechanism | Relevant to us? |
|---|---|---|---|---|---|---|
| **CoEx** (`OpenStereo/stereo/modeling/models/coex/`) | group-wise correlation | 3D hourglass, each stage gated by left features | spx context upsample | SceneFlow then KITTI | **Guided Cost volume Excitation**: `cv = sigmoid(Conv1x1(Conv1x1(left_feat))) * cv`, `coex_cost_processor.py:68-81` | **YES — highest.** ~1.1k params at our sizes, drops in unchanged |
| **Fast-ACVNet** (`Fast-ACVNet/`) | L2-normalized group-wise correlation, `models/submodule.py:124-145` | hourglass + `channelAtt` identical to CoEx, `models/Fast_ACV.py:87-101` | attention-guided top-k disparity resampling (k=24), `models/Fast_ACV.py:292-306` | SceneFlow then KITTI, 2-stage (attention weights first, then full net) | attention concatenation volume; **normalized** correlation; top-k sparse candidates | **Partly.** The `channelAtt` block is directly reusable. Top-k resampling and the 2-stage schedule are not one intervention |
| **LightStereo** (`OpenStereo/stereo/modeling/models/lightstereo/`) | correlation volume at 1/4, `lightstereo.py:51` | **2D** aggregation over D-as-channels, left-gated at 3 scales via `AttentionModule`, `aggregation.py:39-57` | context upsample with 9 learned weights, `disp_refinement.py:194-203` | SceneFlow then KITTI, 256x512 crops | cheap 2D aggregation plus left attention with strip convolutions (1x7, 7x1, 1x11, 11x1, 1x21, 21x1) | **Partly.** Left attention yes; replacing our 3D aggregation with 2D is a different architecture, not an intervention |
| **AANet** (`aanet/`) | correlation, difference or concat selectable, `nets/cost.py:19-55` | Adaptive Intra-Scale (deformable conv) plus Cross-Scale fusion, `nets/aggregation.py:313-405` | hierarchical, multi-scale | SceneFlow then KITTI | adaptive aggregation replacing 3D convs; MODEL_ZOO claims StereoNet-AA 1.08 SceneFlow EPE at 0.53M params, `MODEL_ZOO.md:15` | **No (blocked).** Requires compiled deformable-conv CUDA ops (`nets/deform_conv/`). Not installable under our no-new-dependency rule and not one isolated change |
| **Selective-IGEV** (`Selective-Stereo/Selective-IGEV/`) | geometry encoding volume plus correlation lookups | GRU-based recurrent updates, `core/update.py:54-59` | **iterative**: sequence of disparity updates, supervised with exponential decay `loss_gamma=0.9`, `train_stereo.py:37-59` | SceneFlow then KITTI, long schedules | sequence supervision over many refinement events; channel and spatial attention on the update, `core/update.py:16-45` | **No (too large).** The whole architecture is the mechanism. Confirms our reading that ARM-X only had *one* refinement event to supervise |
| **MobileStereoNet** (`mobilestereonet/`) | **interweave** left/right channels then a learned 1x1 conv collapses them to a scalar similarity, `models/submodule.py:235-241`, `models/MSNet2D.py:143-157` | 2D hourglass on D-as-channels | stacked hourglass outputs | SceneFlow then KITTI | learned matching function instead of a fixed one; deep supervision weights `[0.5,0.5,0.7,1.0]`, `models/submodule.py:281-286` | **Maybe, later.** A learned similarity is a genuine change to information content, but it rewrites the cost volume and interacts with our frozen 24-candidate layout |
| **neka-nat StereoNet** (`StereoNet/`) | difference with **real** disparity search, `models/StereoNet.py:148-152` | 4x Conv3d(32 to 32) plus Conv3d(32 to 1) — same as ours | **hierarchical**: r cascaded edge-aware refinements over an image pyramid, `models/StereoNet.py:83-105` | SceneFlow, RMSprop, 15 epochs | BatchNorm everywhere; cascaded multi-scale refinement; deep supervision over the whole pyramid | **Diagnostic value.** See section 5 |
| **HITNet** | — | — | — | — | — | **NOT INSPECTED.** Not cloned (it lives inside the multi-gigabyte `google-research` monorepo). No claim is made about it |

---

## 4. Teammate vs ours

Teammate repo `pareshvpk/hailostereovision`, previously audited. Their reported
EPE ~1.659 official / ~1.464 masked is **not pixel-comparable** to ours: their
crop anchor differs from our fixed 368x1232 top-left crop, so the two numbers
score different pixel sets.

| Difference | Already tested? | Result | Remaining value |
|---|---|---|---|
| 24 disparity hypotheses | Yes — ARM-V | Incumbent. 3-seed mean 1.7727436 | None. This is the base |
| Group-wise cost | Yes — ARM-W | ~1.776. REFUTED for accuracy | None. Do not re-open |
| Deep supervision | Yes — ARM-X | 3-seed mean 1.6234090. Refuted as a confirmed replacement, but real gradient signal | Low as tested. The public evidence (Selective-IGEV `train_stereo.py:37-59`) suggests the limit is that we have only **one** refinement event to supervise, not that the supervision is weak. Re-opening it would mean adding refinement stages — a different, larger intervention |
| BatchNorm | No | — | **Blocked, not untested.** Our no-BN state is a deliberate recorded deviation matching the BN-folded exported artifact (`phase1/scripts/train_arm_v.py:157`). Adding BN changes the deployment contract, not just the model |
| Far-disparity crop bias | No | — | Medium. But it changes data sampling, which interacts with the split and is harder to attribute cleanly than a graph change |
| SceneFlow then KITTI pretraining | No | — | **High but not architectural.** Universal in every public repo inspected. It is a *data* intervention requiring the SceneFlow corpus, not a one-line architectural arm. Tracked separately in `phase1/docs/SCENEFLOW_ACQUISITION.md` |
| Hourglass aggregation | No | — | Medium. Replaces our 4-layer flat 3D stack; a structural rewrite, not an isolated addition |
| Learned softmax temperature | Partly | ARM-T / `regression_normalize=True` already standardises the cost across the disparity axis before softmax (`src/models/stereonet/regression.py:42-48`), and ARM-V uses it | Low. Normalization already covers most of what a learned temperature buys |
| Channel packing | No | — | Low. An efficiency measure, not an accuracy mechanism |
| Extra refinement structure | No | — | Medium-high, but see the deep-supervision row: more refinement stages and sequence supervision are the same intervention seen from two sides, and together they are too large for one arm |

**Not in the teammate repo at all: left-image-guided cost-volume excitation.**
That is what makes it the best remaining candidate — it is orthogonal to
everything the teammate audit surfaced.

---

## 5. StereoNet reproduction audit

Compared against `reference/public_repos/StereoNet` (neka-nat).

### Architectural differences

| Axis | Ours | neka-nat | Evidence |
|---|---|---|---|
| Normalization | none anywhere | **BatchNorm2d after every conv, BatchNorm3d after every 3D conv** | `models/StereoNet.py:10-24` vs `src/models/stereonet/feature_extractor.py:37-46` |
| Residual block | no BN | `convbn` plus LeakyReLU(0.2) | `models/StereoNet.py:27-40` |
| Disparity search | `shift="right"` in ARM-V, no-op in the frozen baseline | real search, `cost[:,:,i,:,i:] = ref[...,i:] - target[...,:-i]` | `models/StereoNet.py:148-152` |
| Readout sign | `softmax(-cost)` — explicit arg**min** | `softmax(+cost)` — the final Conv3d must learn to output high values at the correct disparity | `models/StereoNet.py:159-161` vs `src/models/stereonet/regression.py:49` |
| Readout order | upsample the whole cost tensor to full res **first**, then soft-argmin | soft-argmin at low res, then upsample the scalar disparity | `src/models/stereonet/regression.py:64-67` vs `models/StereoNet.py:159-162` |
| Refinement | **one** stage at full resolution | **r cascaded** stages over an image pyramid, each doubling resolution and rescaling the disparity (`twice_disparity *= 2`) | `src/models/stereonet/stereonet.py:106-110` vs `models/StereoNet.py:83-105`, `models/StereoNet.py:163-172` |
| Output | single disparity map | list of predictions, one per pyramid stage | `models/StereoNet.py:174-184` |

### Training differences

| Axis | Ours | neka-nat | Evidence |
|---|---|---|---|
| Optimizer | Adam, betas (0.9, 0.999), no weight decay | **RMSprop** | `phase1/scripts/train_arm_v.py:135` vs `main8Xmulti.py:91` |
| LR schedule | cosine annealing to 0 over 200 epochs | StepLR | `phase1/scripts/train_arm_v.py:136` vs `main8Xmulti.py:92` |
| Epochs | 200 | 15 (over a far larger corpus) | `phase1/scripts/train_arm_v.py:48` vs `main8Xmulti.py:28` |
| Weight decay | none | 2e-4 declared | `main8Xmulti.py:41-42` |
| Loss | masked smooth L1, beta 1.0 | **GERF loss** — a Charbonnier-style robust loss, `sum(sqrt((gt-pred)^2 + 4)/2 - 1)/count`, mask `gt > 0` | `phase1/scripts/train_arm_v.py:153` vs `utils/utils.py:11-21` |
| Supervision | single output (ARM-V); two endpoints (ARM-X) | **every pyramid stage**, `loss_weights` all 1.0 | `main8Xmulti.py:152-158`, `main8Xmulti.py:26` |
| Init | PyTorch defaults | `normal_(0, 0.01)` | `main8Xmulti.py:262-276` |
| Training data | KITTI 2015 scenes 0-159 only | **SceneFlow** (tens of thousands of synthetic pairs) | `phase1/scripts/train_arm_v.py` split record vs `dataloader/SecenFlowLoader1.py` |

### Protocol differences

- **Dataset.** Their headline number is SceneFlow EPE. Ours is KITTI 2015
  pooled-pixel EPE over hailo_val scenes 160-199. Different images, different
  disparity statistics, different ground-truth density. Not comparable.
- **Crop anchor.** Their KITTI loader takes the **bottom-right** 368x1232 crop:
  `left_img.crop((w-1232, h-368, w, h))` (`dataloader/KITTILoader.py:67-71`).
  Ours is a fixed **top-left** 368x1232 crop after bottom/right padding. Even on
  identical images these score different pixel sets.
- **Valid mask.** Their GERF loss masks `gt > 0` (`utils/utils.py:13`); their
  eval masks `(disp_L < maxdisp) & (disp_L >= 0)` (`main8Xmulti.py:208`). Our
  contract is `valid = gt > 0` over exactly 3,802,797 pooled pixels. Not the
  same denominator.
- **Model scale.** Their `k` and `r` are 3 or 4, giving a 3- or 4-stage
  refinement cascade. Our single refinement stage is a different model.

### Unverified claims

- The README states "1.32 EPE_all with 8X single model" in prose
  (`README.md:7`) while the results table attributes 1.32 to "ours(16X multi)"
  (`README.md:34`). **The repository contradicts itself about which model
  produced 1.32.** No training log or checkpoint in the clone substantiates
  either attribution.
- The model files are decompiled bytecode (`uncompyle6` headers at the top of
  `models/StereoNet.py`), not original source. They are readable and internally
  consistent, but they are a reconstruction.
- **Conclusion for Question 4:** the 1.32 figure is not usable as evidence for
  anything about our model. It differs in dataset, crop anchor, mask, optimizer,
  loss, supervision scheme, normalization and refinement depth simultaneously,
  and its own README cannot say which architecture produced it. What the repo
  *does* establish, from code rather than claims, is that independent StereoNet
  implementations use BatchNorm, a real disparity search, a robust loss, a
  cascaded multi-scale refinement and synthetic pretraining — none of which is a
  single isolatable difference.

---

## 6. Mechanism matrix

Ranked by research value: evidence strength, isolatability, and fit to a 398k
parameter model.

| Mechanism | Public evidence (file:line) | Already tested | Expected relevance | Parameter cost | Isolatable? |
|---|---|---|---|---|---|
| **1. Left-guided cost-volume excitation** | CoEx `coex_cost_processor.py:68-81`; Fast-ACVNet `models/Fast_ACV.py:87-101`; LightStereo `aggregation.py:104-134` and `:39-57`; 17 OpenStereo model dirs reference it | **No** | **High.** Adds per-pixel left-image conditioning to the matching representation before aggregation — the exact gap identified in section 2 | ~1.1k (+0.3%) | **Yes.** One module between `cost_volume` and `aggregation` |
| 2. L2-normalized matching | Fast-ACVNet `models/submodule.py:124-145` (`groupwise_correlation_norm`), `:148-161` (`norm_correlation`) | No | Medium-high. Removes the activation-magnitude confound in our raw difference | **0** | Yes, but it changes the cost semantics from difference to correlation, which interacts with the frozen `shift="right"` indexing |
| 3. Learned similarity function | MobileStereoNet `models/submodule.py:235-241` and `models/MSNet2D.py:143-157` | No | Medium-high | ~35 (1x1 conv over 32 interwoven pairs) plus a channel-layout change | Partly. Rewrites the cost volume construction |
| 4. Sequence supervision over many refinement events | Selective-IGEV `train_stereo.py:37-59`; neka-nat `main8Xmulti.py:152-158` | **Partly** (ARM-X, two endpoints) | Medium. Public evidence says the win comes from *many* events, not from the supervision itself | Large — needs extra refinement stages | No. Two changes at once |
| 5. Hierarchical multi-scale refinement | neka-nat `models/StereoNet.py:83-105`, `:163-172` | No | Medium | ~100k+ per extra stage | No. Structural rewrite |
| 6. Context upsample (9 learned weights) | LightStereo `disp_refinement.py:194-203` | No | Medium | small | Partly — but our readout upsamples the *cost tensor*, not a low-res disparity, so there is nothing for it to replace without changing the frozen readout |
| 7. SceneFlow pretraining | Universal: every repo inspected | No | **High but not architectural.** Data intervention; corpus not yet acquired | 0 | Yes, but it is a separate research track |
| 8. Adaptive/deformable aggregation | AANet `nets/aggregation.py:313-405`; `MODEL_ZOO.md:15` | No | High in principle | Replaces the 3D stack | **No.** Needs compiled deformable-conv CUDA ops |
| 9. Iterative GRU refinement | Selective-IGEV `core/update.py:54-59` | No | High in principle | Millions | **No.** A different architecture |
| 10. Group-wise cost | Fast-ACVNet `models/submodule.py:104-122`; MobileStereoNet `models/submodule.py:244-251` | **Yes — ARM-W** | **Refuted** | — | Closed |

---

## 7. Strongest untested mechanism

**Left-image-guided cost-volume excitation** (CoEx calls it Guided Cost volume
Excitation; Fast-ACVNet and IGEV call the same block `channelAtt` /
`feature_att`; LightStereo exposes it as `LEFT_ATT`).

### What it does

Take the left feature map that already exists in our forward pass. Push it
through two 1x1 convolutions to produce one scalar per cost-volume channel per
pixel. Squash with a sigmoid. Multiply the cost volume by it, broadcasting over
the disparity axis. The reference implementation is five lines:

```python
# OpenStereo/stereo/modeling/models/coex/coex_cost_processor.py:72-81
self.im_att = nn.Sequential(
    BasicConv(im_chan, im_chan // 2, kernel_size=1, stride=1, padding=0),
    nn.Conv2d(im_chan // 2, cv_chan, 1))

def forward(self, cv, im):
    channel_att = self.im_att(im).unsqueeze(2)
    cv = torch.sigmoid(channel_att) * cv
    return cv
```

Fast-ACVNet's copy is character-for-character the same block
(`Fast-ACVNet/models/Fast_ACV.py:87-101`).

### Why it should help stereo correspondence

Our cost volume is `left(x) - right(x-k)` over 32 unnormalized feature channels
(`src/models/stereonet/cost_volume.py:110-111`). Two facts follow:

1. A channel's contribution to the cost is proportional to its activation
   magnitude, not to how informative it is for matching at that pixel. With no
   BatchNorm anywhere in our network, nothing equalizes this.
2. The 3D aggregation applies the same 32 filter weights at every pixel
   (`src/models/stereonet/aggregation.py:31-35`). It cannot weight channels
   differently for a textureless road surface and a high-gradient pole edge,
   because it has no per-pixel signal telling it which is which.

Excitation supplies exactly that signal. The left image already knows whether a
pixel sits on a texture edge, in a saturated region, or on uniform tarmac. The
gate lets that appearance information decide, per pixel, which channels' matching
evidence to trust *before* aggregation collapses the volume.

This attacks the error in a way our previous arms could not. In our current
graph, left-image context enters exactly once — as refinement guidance, after
readout (`src/models/stereonet/refinement.py:51-53`). At that point the cost
curve is already a scalar disparity. Refinement can smooth a wrong answer; it
cannot re-weight the evidence that produced it. Excitation is the only cheap way
to get image context in *before* the decision.

### Where it enters our graph

Exactly one insertion point, in `StereoNet.forward`
(`src/models/stereonet/stereonet.py:117-119`), between these two existing lines:

```
volume = self.cost_volume(left_features, right_features)   # (B,32,24,32,64)
                    <-- HERE
cost   = self.aggregation(volume)                          # (B,24,32,64)
```

`left_features` is already in scope at that point
(`src/models/stereonet/stereonet.py:115`). Nothing else moves.

### What information it adds

Per-pixel, per-channel reliability weighting of the matching evidence,
conditioned on left-image appearance. Formally the cost volume becomes
`g(left_feat)_c(x) * (left_c(x) - right_c(x-k))` where `g` is learned and
`g` is in (0,1). The gate is shared across disparity candidates by construction —
it modulates *which channels* are trusted, not *which disparities* are preferred.
That is deliberate: a disparity-dependent gate would let the module prejudge the
answer, which is what the aggregation network is for.

### Why ARM-W and ARM-X did not already test it

- **ARM-W** replaced the 32 feature channels with G group-correlation channels
  (`src/models/stereonet/cost_volume.py:160-171`). That is a *compression*: it
  discards channel identity to save memory, and it removed information. Result
  ~1.776, refuted. Excitation does the opposite — it keeps all 32 channels and
  adds a new conditioning input. A mechanism that adds information is not tested
  by an experiment that removed it, even though both act on the channel axis.
- **ARM-X** added an auxiliary loss term. The forward graph was byte-identical to
  ARM-V. No cost-volume mechanism was touched at all.

### Approximate parameter and compute cost

With `im_chan = cv_chan = 32`:

- `Conv2d(32, 16, 1)`: 512 weights (plus 16 bias, plus 32 BatchNorm affine if the
  reference `BasicConv` norm is kept)
- `Conv2d(16, 32, 1)`: 512 weights plus 32 bias
- **Total ~1.1k parameters, about +0.3% on 397,954.**

Compute: two 1x1 convolutions over a (B,32,32,64) map — negligible — plus one
broadcast multiply over the (B,32,24,32,64) volume, which is one elementwise pass
over a tensor the aggregation already reads four times.

### Risks and failure modes

- **Gate collapse.** The sigmoid can saturate to a near-constant, making the
  module a no-op with a scale factor. Harmless but wasted; detectable by
  inspecting the gate's per-channel variance.
- **No BatchNorm.** Every public implementation of this block sits in a network
  with BN. Ours has none. The gate output could be poorly scaled at
  initialization. Mitigation: keep the reference weight init
  (`normal_(0, sqrt(2/n))`, `coex_cost_processor.py:83-88`) rather than PyTorch
  defaults — but note this is itself a second change, so it must be decided
  before the run and recorded, not tuned afterwards.
- **Gradient attenuation.** A sigmoid gate multiplies gradients flowing back into
  the feature extractor by a factor below 1, which could slow feature learning
  over a fixed 200-epoch budget.
- **Small-data regime.** All public evidence comes from SceneFlow-pretrained
  models. We train on 160 KITTI scenes. An attention module has more capacity to
  overfit than to generalize here. This is the single largest reason the arm
  could fail, and it is not a reason to skip the experiment — it is what the
  experiment measures.

---

## 8. Proposed ARM-Y specification

**Not implemented. Specification only.**

```
BASE          = ARM-V, unchanged
INTERVENTION  = one CostVolumeExcitation module inserted between the cost volume
                and the aggregation
```

Everything else is **identical to ARM-V**, with no exceptions:

| Held fixed | Value | Source |
|---|---|---|
| Disparity candidates | 24 | `train_arm_v.py:131-132` |
| Spacing | 8 px full-resolution (`downsample_levels=3`) | `train_arm_v.py:131-132` |
| Shift | `"right"` | `train_arm_v.py:131-132` |
| Readout normalization | `regression_normalize=True` | `train_arm_v.py:131-132` |
| Aggregation | 4x Conv3d(32 to 32) plus Conv3d(32 to 1), unchanged | `aggregation.py:26-35` |
| Refinement | one stage, dilations (1,2,4,8,1,1), unchanged | `stereonet.py:58` |
| Loss | masked smooth L1, beta 1.0 | `train_arm_v.py:153` |
| Optimizer | Adam, lr 1e-3, betas (0.9, 0.999) | `train_arm_v.py:135` |
| Schedule | cosine annealing to 0 | `train_arm_v.py:136` |
| Epochs | 200 | `train_arm_v.py:48` |
| Batch | 2 | `train_arm_v.py:49` |
| Crop | random 256x512 | `train_arm_v.py:47` |
| Augmentation | gain jitter sigma 0.1, no flip | `train_arm_v.py:154` |
| Split | hailo_calib 0-159 train, hailo_val 160-199 eval | `train_arm_v.py` split record |
| Seeds | 0, 1, 2 — the same three as ARM-V | `train_arm_v_seed.py` |
| Evaluation | the frozen contract, unchanged, 3,802,797 pooled pixels | — |
| BatchNorm | still none, anywhere | `train_arm_v.py:157` |

### Conceptual tensor flow

```
left_features   (B, 32, 32, 64)        # already computed, stereonet.py:115
right_features  (B, 32, 32, 64)

volume = build_cost_volume(left_features, right_features,
                           num_disparities=24, method="subtract", shift="right")
                (B, 32, 24, 32, 64)     # (B, C, D, H, W)

--- ARM-Y intervention, and nothing else ---------------------------------
gate   = Conv2d(32, 16, kernel_size=1)(left_features)     (B, 16, 32, 64)
gate   = Conv2d(16, 32, kernel_size=1)(gate)              (B, 32, 32, 64)
gate   = sigmoid(gate).unsqueeze(2)                       (B, 32,  1, 32, 64)
volume = gate * volume                                    (B, 32, 24, 32, 64)
--------------------------------------------------------------------------

cost   = Aggregation(volume)            (B, 24, 32, 64)   # unchanged
disp_0 = DisparityRegression(cost, size=(256, 512))       # unchanged
disp   = relu(disp_0 + Refinement(disp_0, left))          # unchanged
```

The gate is broadcast over the disparity axis (`unsqueeze(2)`), so it is
explicitly **disparity-independent**: it reweights channels, never candidates.

Open decision to be recorded **before** the run, not after: whether the two
convolutions use the reference initialization (`normal_(0, sqrt(2/n))`,
`coex_cost_processor.py:83-88`) or PyTorch defaults. Pick one, write it into the
pre-registration, do not try both.

---

## 9. Go / No-Go criteria

Pre-registered now, before any ARM-Y code exists, using the existing discipline.

**Measured noise floor.** ARM-V, 3 seeds: 1.8903392, 1.5493088, 1.8785827.
Mean 1.7727436, spread (max minus min) 0.3410304 px. Any claimed improvement
smaller than that spread is indistinguishable from seed noise at n=3.

**Protocol.** ARM-Y runs seeds 0, 1, 2, the same three as ARM-V, under the frozen
evaluation contract. All three run to completion before any number is read. No
seed is dropped, re-run or replaced.

**Decision thresholds, fixed now:**

| ARM-Y 3-seed mean | Verdict |
|---|---|
| **below 1.4317132 px** (ARM-V mean minus measured spread) | **CONFIRMED.** The improvement exceeds run-to-run noise. ARM-Y becomes the incumbent |
| 1.4317132 to 1.7727436 px | **INCONCLUSIVE.** Directionally better but inside the noise floor. Does not replace ARM-V. Does not license a follow-up arm on the same mechanism without first tightening the ARM-V baseline with additional seeds |
| **above 1.7727436 px** | **REFUTED.** Mechanism does not transfer to our regime. Close it, as ARM-W was closed |

**Sanity check that this threshold is not invented to fit.** Applying it
retrospectively to ARM-X: mean 1.6234090 is above 1.4317132, therefore
INCONCLUSIVE at best, not CONFIRMED. That matches the verdict already recorded
for ARM-X. The rule reproduces an existing decision, so it is not tuned to
ARM-Y.

**Go condition for implementing ARM-Y at all:** it satisfies all eight
constraints from the research questions — untested (section 4, section 6 row 1),
evidenced in four independent public implementations, causally connected to the
bottleneck in section 2, a single intervention, no change to the evaluation
contract, not a copied architecture, about +0.3% parameters, and it attacks the
matching representation rather than the training recipe.

**No-Go condition:** if implementation cannot be done without also changing the
cost-volume semantics, the aggregation input channel count, or the normalization
state of the network, the arm is not isolated and must not run.

---

## 10. Sources

Inspected at source level from shallow clones under `reference/public_repos/`:

- **OpenStereo** — https://github.com/XiandaGuo/OpenStereo
  - `stereo/modeling/models/coex/coex_cost_processor.py:68-88` (channelAtt / GCE)
  - `stereo/modeling/models/lightstereo/lightstereo.py:12-89`
  - `stereo/modeling/models/lightstereo/aggregation.py:30-64`, `:104-134`
  - `stereo/modeling/disp_refinement/disp_refinement.py:194-203`
  - `docs/1.model_zoo.md:28-31`, `:48-51` (LightStereo SceneFlow EPE and KITTI
    D1 — README-level claims, not reproduced by us)
- **AANet** — https://github.com/haofeixu/aanet
  - `nets/cost.py:5-56`, `nets/aggregation.py:313-405`, `nets/deform_conv/`
  - `MODEL_ZOO.md:15` (StereoNet-AA 1.08 SceneFlow EPE, 0.53M params —
    repository claim)
- **Fast-ACVNet** — https://github.com/gangweiX/Fast-ACVNet
  - `models/Fast_ACV.py:87-101`, `:249-320`
  - `models/submodule.py:104-161`
- **Selective-Stereo** — https://github.com/Windsrain/Selective-Stereo
  - `Selective-IGEV/core/update.py:13-59`
  - `Selective-IGEV/train_stereo.py:37-62`
- **MobileStereoNet** — https://github.com/cogsys-tuebingen/mobilestereonet
  - `models/submodule.py:235-286`, `models/MSNet2D.py:48-162`
- **neka-nat StereoNet** — https://github.com/neka-nat/StereoNet
  - `models/StereoNet.py:10-184`, `models/StereoNet8Xmulti.py:80-184`
  - `main8Xmulti.py:25-42`, `:86-92`, `:152-158`, `:208`, `:262-276`
  - `utils/utils.py:11-28`, `dataloader/KITTILoader.py:50-71`
  - `README.md:7`, `:32-34` (self-contradictory 1.32 attribution)

Our own code, read-only:

- `src/models/stereonet/{stereonet,feature_extractor,cost_volume,aggregation,regression,refinement}.py`
- `phase1/scripts/train_arm_v.py`
- `phase1/docs/AUDIT_STAGE1_BASELINE_FACTS.md`

**NOT INSPECTED:**

- **HITNet** — https://github.com/google-research/google-research/tree/master/hitnet
  Not cloned; it lives inside the multi-gigabyte `google-research` monorepo and
  was excluded rather than guessed at. No claim in this document rests on it.
- **StereoBase, IGEV++, FoundationStereo, MonSter** — present in the OpenStereo
  tree and confirmed to reference `channelAtt`/`feature_att` by filename grep
  only. Their architectures were not read. They are counted as corroborating the
  prevalence of the excitation mechanism, nothing more.

---

NEXT MOVE:
Pre-register ARM-Y — ARM-V plus a single left-image-guided cost-volume excitation gate between `self.cost_volume` and `self.aggregation`, with the section 9 thresholds fixed in writing before any code is written.
