# Competitive landscape

The design space around StereoNet, so Phase 2 knows what has already been tried
and where the genuinely open ground is.

**Scope limit, stated up front.** Every entry rests on the paper's abstract and
stated claims, fetched from arXiv and read. **No competitor has been
implemented, run, or independently measured**, so every accuracy and speed figure
below is `SOURCE` — an author's claim about their own method, on their own
hardware, under their own protocol. Those numbers are not mutually comparable and
none is comparable with our measurements. Where an abstract does not state
something, the cell says UNKNOWN rather than a plausible guess.

Phase 1 does not implement competitors. The purpose here is to map the design
space.

---

## 1. Source registry additions

All inspected during this work. Two arXiv identifiers were wrong on first attempt
and the retrieved papers turned out to be unrelated (a wireless-networks paper
and a solar-physics paper); both were discarded rather than cited, and the
corrected identifiers are below.

| ID | Paper | arXiv | Inspected |
|---|---|---|---|
| SR-040 | HITNet: Hierarchical Iterative Tile Refinement Network for Real-time Stereo Matching | 2007.12140 | abstract |
| SR-041 | MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching | 2108.09770 | abstract |
| SR-042 | AANet: Adaptive Aggregation Network for Efficient Stereo Matching | 2004.09548 | abstract |
| SR-043 | Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet) | 2203.02146 | abstract |
| SR-044 | Accurate and Efficient Stereo Matching via Attention Concatenation Volume (Fast-ACVNet) | 2209.12699 | abstract |
| SR-045 | Correlate-and-Excite: Real-Time Stereo Matching via Guided Cost Volume Excitation (CoEx) | 2108.05773 | abstract |
| SR-046 | Group-wise Correlation Stereo Network (GwcNet) | 1903.04025 | abstract |
| SR-047 | Pyramid Stereo Matching Network (PSMNet) | 1803.08669 | abstract |
| SR-048 | RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching | 2109.07547 | abstract |
| SR-049 | Iterative Geometry Encoding Volume for Stereo Matching (IGEV-Stereo) | 2303.06615 | abstract |
| SR-050 | Hierarchical Neural Architecture Search for Deep Stereo Matching (LEAStereo) | 2010.13501 | abstract |

---

## 2. The design space

| Method | Matching representation | Aggregation | Refinement / disparity | Positioning |
|---|---|---|---|---|
| **StereoNet as deployed** [SR-001] | Subtraction volume, 32 ch, **1/16** res, 12 candidates — **and the shift is a no-op, so no search happens** (EXP-010) | 5 × Conv3d at 12×23×77 | Single full-resolution guided refinement, dilations 1/2/4/8/1/1; soft-argmin at full res | 423 K params, 56.0 GMAC; the refinement is 90.6 % of it |
| **StereoNet (paper)** [SR-010] | Low-resolution cost volume; "quantization-free" sub-pixel output | UNKNOWN from abstract | "Hierarchically re-introduces high-frequency details through a learned upsampling function… compact pixel-to-pixel refinement networks", colour-guided | Claims 60 fps on a Titan X. Layer detail UNKNOWN — full text not read |
| **PSMNet** [SR-047] | Cost volume built after spatial pyramid pooling | Stacked 3D hourglass with intermediate supervision | Regression from the regularised volume | The accuracy-first 3D-CNN baseline. Ranked 1st on KITTI 2012/2015 in early 2018 |
| **GwcNet** [SR-046] | **Group-wise correlation** — features split into channel groups, correlation per group | Improved 3D stacked hourglass | Regression | Middle ground between full concatenation and full correlation: "will not lose too much information like full correlation", better under parameter reduction |
| **AANet** [SR-042] | UNKNOWN from abstract | **Replaces 3D convolutions entirely**: sparse-points intra-scale aggregation plus a learned approximation of cross-scale aggregation | UNKNOWN from abstract | The clearest "kill the 3D convolutions" position. 62 ms; 4× faster than PSMNet, 41× than GC-Net, 38× than GA-Net. Explicitly reports improving StereoNet |
| **MobileStereoNet** [SR-041] | Two variants — a 2D model with a proposed new cost volume, and a 3D model | MobileNet blocks, extended to 3D | Encoder-decoder | The efficiency-engineering position. 2D: 27 % fewer parameters, 72 % fewer operations; 3D: 95 % fewer parameters, 38 % fewer operations |
| **ACVNet** [SR-043] | **Attention Concatenation Volume** — attention weights from correlation suppress redundancy in a concatenation volume; multi-level adaptive patch matching for textureless regions | Lightweight, because the volume is more informative | Regression | Makes the volume better so aggregation can be smaller: matches GwcNet accuracy with **1/25** of its aggregation parameters |
| **Fast-ACVNet** [SR-044] | The same ACV, tuned for speed | Lightweight | Regression | ACVNet ranks 2nd on KITTI 2015 and Scene Flow; Fast-ACVNet targets real time |
| **CoEx** [SR-045] | Standard volumetric | **Guided Cost volume Excitation** — simple image-guided channel excitation instead of expensive spatially varying operations | **top-k selection before soft-argmin** | Directly relevant to StereoNet's regression stage. Argues spatially varying aggregation is complex and memory-hungry, and channel excitation recovers most of the benefit |
| **HITNet** [SR-040] | **No explicit cost volume at all.** Fast multi-resolution initialisation, then differentiable 2D geometric propagation and warping | 2D operations, not 3D | Hierarchical iterative **tile** refinement, inferring **slanted plane hypotheses** | The strongest alternative to volumetric stereo. 1st on KITTI 2012/2015 among published methods under 100 ms; 1st–3rd on ETH3D |
| **RAFT-Stereo** [SR-048] | All-pairs correlation, RAFT-style | **Multi-level convolutional GRUs** | Iterative recurrent updates; a real-time variant exists | Accuracy reference. 1st on Middlebury, 29 % better than the next method on 1 px error |
| **IGEV-Stereo** [SR-049] | **Geometry Encoding Volume** — geometry, context and local matching detail combined; notes all-pairs correlation "lack[s] non-local geometry knowledge" | Iteratively indexed | ConvGRU updates, with GEV regression giving a good initialisation to speed convergence | Current accuracy reference. 1st on KITTI 2015 and 2012 (Reflective) among published, fastest of the top ten |
| **LEAStereo** [SR-050] | Searched | Searched | Searched | Hierarchical NAS over the whole feature-extraction → volume → matching pipeline. Claims top-1 on KITTI 2012/2015, Middlebury and Scene Flow with smaller size and faster inference |

---

## 3. What the field has converged on

Reading across the abstracts, four distinct answers to "3D convolutions are too
expensive" have emerged:

1. **Make the volume smaller.** StereoNet's answer — match at very low resolution
   and spend the savings on learned upsampling. Our profiling shows where that
   budget actually goes: 90.6 % of MACs into refinement [MEASUREMENT: EXP-001].
2. **Make the volume more informative, so aggregation can shrink.** GwcNet's
   group-wise correlation and ACVNet's attention volume. ACVNet's claim is the
   sharpest evidence for this direction: 1/25 of the aggregation parameters at
   higher accuracy [SOURCE: SR-043].
3. **Replace 3D aggregation with something cheaper.** AANet's sparse-point and
   cross-scale modules [SR-042]; CoEx's guided channel excitation [SR-045].
4. **Abandon the volume.** HITNet's tile hypotheses and geometric propagation
   [SR-040]; RAFT-Stereo and IGEV-Stereo's iterative recurrent refinement
   [SR-048, SR-049].

**Two observations that matter for Phase 2:**

The field's efficiency work overwhelmingly targets **cost-volume aggregation**,
because in PSMNet-class architectures that is where the cost is. In the deployed
StereoNet, aggregation is **4.2 % of MACs and 2.8 % of GPU time**
[MEASUREMENT: EXP-001, EXP-013]. **Most of the published efficiency literature
optimises a stage that is already nearly free in this architecture.** The
expensive stage here — full-resolution guided refinement — receives far less
attention.

And two of the ideas above address weaknesses we measured directly. CoEx's top-k
selection before soft-argmin [SR-045] speaks to a regression stage that in our
baseline is 142× over-represented in time relative to its MACs
[MEASUREMENT: EXP-013] and produces an anti-correlated estimate
[MEASUREMENT: EXP-008]. AANet reports improving StereoNet specifically
[SOURCE: SR-042] — though against which StereoNet, and whether that
implementation had a working cost volume, is **UNKNOWN**.

---

## 4. Edge suitability

Judged on architecture, not measurement. **All of this is INFERENCE** — no
competitor has been compiled or run on any accelerator here, and our own
profiling showed the stage ranking changing 6× between GPU and CPU
[MEASUREMENT: EXP-013, EXP-014], so architectural reasoning about edge silicon is
weak evidence.

| Method | Likely edge fit | Reasoning |
|---|---|---|
| StereoNet as deployed | **Demonstrated** | It is the only one here with a published, compiled Hailo-8 artifact [SR-005]. Static shapes, no dynamic control flow, only Conv/LeakyReLU/Add/Concat/Slice/Resize/Softmax/ReduceSum [VERIFIED: SR-001] |
| MobileStereoNet | Good | Designed for the constraint; the 2D variant avoids 3D convolutions |
| AANet, CoEx, Fast-ACVNet | Good | Explicitly real-time; AANet removes 3D convolutions, which are frequently unsupported or slow on NPUs |
| HITNet | Uncertain | Fast, but tile hypotheses and geometric warping need gather/scatter operations that fixed-function accelerators often handle poorly |
| GwcNet, ACVNet, PSMNet | Poor | 3D stacked hourglass aggregation; memory and 3D convolution support are both likely to bind |
| RAFT-Stereo, IGEV-Stereo | Poor | Iterative recurrent updates mean a data-dependent loop count, which most edge compilers cannot express |
| LEAStereo | UNKNOWN | The searched architecture's operator set is not stated in the abstract |

---

## 5. Explicit gaps

| Gap | Status |
|---|---|
| Full paper texts beyond abstracts | **Not read.** Every claim above is abstract-level |
| Any independent measurement of any competitor | **Not done.** Phase 1 does not implement competitors |
| A controlled comparison under one protocol | **Not done.** OpenStereo [SR-034] would be the vehicle; not inspected |
| Parameter counts and MACs on a common basis | **UNKNOWN.** Papers state these under differing conventions — and Hailo's own two conventions differ by 47 % (EXP-001), so cross-paper figures cannot be trusted without normalising |
| Whether any competitor's published StereoNet comparison used a working cost volume | **UNKNOWN**, and given EXP-010 this is worth checking before trusting any StereoNet baseline number in the literature |

---

*All competitor entries: SOURCE, from the arXiv abstracts listed in §1, each
fetched and read. Our own figures: [EXP-001], [EXP-013], [EXP-014] as marked.*
