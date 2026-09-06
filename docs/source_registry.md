# Source registry

Every external source this project relies on. A claim anywhere in `docs/` that
rests on outside information cites an `SR-xxx` id from this table.

**Rule: nothing is cited that was not opened.** The *Inspected* column is the
enforcement. A source marked *no* has been located but not read, and must not be
used as evidence for anything — it is a lead, not a citation.

Reliability grades:

- **primary-artifact** — the executable or machine-readable thing itself
  (an ONNX file, a compiler script, a config). Highest weight, because it is
  what actually runs.
- **primary-doc** — first-party documentation or a peer-reviewed paper.
- **secondary** — third-party description, community post, or an implementation
  by someone other than the original author.

When sources disagree, the conflict is preserved in both documents rather than
resolved by preference. The ordering used to decide which value describes the
*deployed baseline* is set out in `reference_pipeline.md`.

---

## Hailo artifacts

| ID | Title | Type | Author | Version / date | Provides | Reliability | Inspected |
|---|---|---|---|---|---|---|---|
| SR-001 | `stereonet.onnx` (inside `stereonet.zip`) | model artifact | Hailo | published 2023-05-31; sha256 `b1a01d855bb22663…` | The executable graph. 168 nodes, fully static shapes, complete weight set. Settles every architecture question the paper and the Python source leave open. | primary-artifact | **yes** |
| SR-002 | `hailo_model_zoo/cfg/networks/stereonet.yaml` | config | Hailo | master, sha256 `9655b95f3884d1c2…` | Input/output shapes, published parameter and operation counts, normalisation constants, calibration set, evaluation dataset, `full_precision_result: 8.223`, upstream source link, supported hardware. | primary-doc | **yes** |
| SR-003 | `hailo_model_zoo/cfg/alls/generic/stereonet.alls` | compiler script | Hailo | master, sha256 `9928fc4b01c07e60…` | The Dataflow Compiler directives actually applied: spatial defusion of `conv42`–`conv52` into up to 22 pieces, width splitting, re-concatenation trees, multi-way shortcut fanout, and the in-network normalisation layer. Direct evidence of hardware constraints. | primary-artifact | **yes** |
| SR-004 | `HAILO8_stereo_depth_estimation.rst` | benchmark doc | Hailo | master, sha256 `9490051ed8eb8535…` | Published Hailo-8 figures: float EPE 8.22, hardware EPE 10.3, 10.7 FPS at batch 1, 11.6 FPS at batch 8, on KITTI Stereo 2015. | primary-doc | **yes** |
| SR-005 | `stereonet.hef` (compiled Hailo-8 binary) | model artifact | Hailo | Model Zoo v2.19.0, 24,057,165 bytes, sha256 `541ee6cb841df318…` | The compiled binary. Held for completeness; its internals are not documented publicly and no Hailo device is available to run it. | primary-artifact | acquired, **not** decoded |
| SR-006 | `stereonet_profiler_results_compiled_runtime_data.html` | profiler report | Hailo | Model Zoo v2.19.0, 40,883,621 bytes, sha256 `cf69a03f5d359b37…` | Hailo's own compiled profile. Decoded (EXP-017): a 53-field model summary and a 242-row per-layer table giving MACs, modelled FPS, utilisation, memory cuts, quantisation bit-widths, context assignment and defusion groups. `profiling_mode: post_placement`, model-level fps and latency `N/A` -- a compiler estimate, not a measured run. | primary-artifact | **yes** |
| SR-007 | `hailo-apps` `hailo_apps/cpp/depth_estimation_stereo/` | application source | Hailo | main branch snapshot, from `hailo_apps_main.zip` sha256 `b89369fdfbaf073b…` | What the shipped application does around the model. Read: the entire postprocess is one line casting the output buffer to an 8-bit image. No calibration, no baseline, no disparity-to-depth conversion anywhere. | primary-artifact | **yes** |

## StereoNet: paper and implementations

| ID | Title | Type | Author | Version / date | Provides | Reliability | Inspected |
|---|---|---|---|---|---|---|---|
| SR-010 | *StereoNet: Guided Hierarchical Refinement for Real-Time Edge-Aware Depth Prediction* | paper | Khamis, Fanello, Rhemann, Kowdle, Valentin, Izadi (Google) | ECCV 2018; arXiv 1807.08865 | The original architecture and its claims: low-resolution cost volume, sub-pixel matching precision, learned edge-aware hierarchical upsampling, 60 fps on a Titan X. | primary-doc | partially — abstract and architecture description via publisher/arXiv listing; **full PDF not yet read** |
| SR-011 | `github.com/nivosco/StereoNet` | implementation | nivosco | master branch snapshot, sha256 `db1f11d80bc68db0…` | The implementation Hailo's config names as its source. `models/stereonet.py` and `utils/cost_volume.py` give the module structure, layer sizes, disparity range and training scripts. Its README states the cost volume uses concatenation rather than the paper's subtraction, and that it falls short of the paper's accuracy. | secondary (third-party reimplementation, but *the* upstream for this baseline) | **yes** |

## Datasets

| ID | Title | Type | Author | Version / date | Provides | Reliability | Inspected |
|---|---|---|---|---|---|---|---|
| SR-020 | KITTI 2015 stereo — `data_scene_flow_calib.zip` | dataset | Geiger et al., KIT / TTI-C | 1,631,055 bytes | Per-scene rectified calibration for all 200 training and 200 testing scenes. Source of the real `f`, `B` and `fB` used throughout the depth analysis. | primary-artifact | **yes** |
| SR-021 | KITTI 2015 stereo — `data_scene_flow.zip` | dataset | Geiger et al., KIT / TTI-C | — | Stereo image pairs and sparse LiDAR-derived disparity ground truth (`disp_occ_0`, `disp_noc_0`). The evaluation set Hailo's 8.223 figure is defined on. | primary-artifact | **yes** |
| SR-022 | Scene Flow / FlyingThings3D | dataset | Mayer et al., University of Freiburg | — | Large synthetic stereo data with dense ground truth; the pre-training set named by the upstream training scripts. | primary-doc | **no** — not yet downloaded or inspected |
| SR-023 | Middlebury 2014 stereo | dataset | Scharstein et al. | — | High-resolution indoor pairs with dense ground truth; intended for failure-mode analysis. | primary-doc | **no** — not yet downloaded or inspected |

## Competitor architectures

Inspected at abstract level from arXiv during the competitive landscape study.
Full texts have **not** been read, and no competitor has been implemented or
measured. Every figure attributed to these is the authors' claim about their own
method under their own protocol.

| ID | Title | arXiv | Inspected |
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

Two identifiers were wrong on first attempt and returned unrelated papers -- a
wireless-networks paper at 2004.02642 and a solar-physics paper at 1803.09046.
Both were discarded rather than cited. Recorded here because a citation that was
never verified is exactly the failure this registry exists to prevent.

## Leads not yet inspected

Recorded so they are not re-discovered, and so it stays visible that they have
**not** been used as evidence. None of these may be cited until inspected.

| ID | Title | Why it matters | Inspected |
|---|---|---|---|
| SR-030 | HailoRT documentation | Runtime behaviour, batching, host-device transfer costs. | no |
| SR-031 | Qualcomm AI Hub — StereoNet | An independent edge deployment of the same architecture; a second data point on export and quantisation. | no |
| SR-032 | *(superseded -- these are now SR-040 to SR-050, inspected at abstract level)* | The competitive landscape study. | superseded |
| SR-033 | ETH3D stereo benchmark | An additional real-world benchmark. | no |
| SR-034 | OpenStereo | A unified framework for controlled cross-architecture comparison. | no |

---

## Provenance

All acquired artifacts are hashed in `reference/MANIFEST.md` and
`reference/manifest.json`, regenerated by `python scripts/hash_reference.py`.
`reference/` is read-only after acquisition; nothing in it is edited.
