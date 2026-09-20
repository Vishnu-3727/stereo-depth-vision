# Stereo Depth Vision — Microchip StereoNet

**Current project entry point.** Updated 2026-09-20 by the documentation closure pass.
The original Phase-1 README is preserved below the rule, unchanged, as historical material.

## Current project status

| | |
|---|---|
| Technical investigation | **CLOSED** — no further experiments are authorized |
| Documentation | closure pass complete (this section, `RESULTS_INDEX.md`, `stage_c_deploy/DOCUMENTATION_CLOSURE_RECORD.md`) |
| Frozen deployment candidate | **ARM-P seed 1** — `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth` (sha256 `b2f6f5d5…fffeb7454`), 397,954 params, frozen-contract EPE **1.1912168 px** |
| Hailo deployment | **BLOCKED** |

> **ARM-P seed 1 is the frozen deployment candidate selected under C0, but deployment to Hailo
> has NOT occurred because the target Hailo device has not been specified and the corresponding
> target toolchain has therefore not been executed.**

### Current blocker

- **Hailo target device: UNSPECIFIED.** No authoritative company requirement for device, board,
  SDK/DFC version, resolution, FPS, power or quantization mode exists anywhere in this
  repository. Historical Hailo-8 / 10H / 15H references in `reference/` are **HISTORICAL** and
  must not be promoted to "the target".
- **Hailo toolchain: NOT EXECUTED.** No parse, no quantization, no compile.
- **ARM-P HEF: DOES NOT EXIST.** The only `.hef` present is the historical
  `reference/stereonet.hef`.
- Two known, frozen failures stand unrepaired: **C1 export parity FAIL**
  (0.001708984375 px vs the frozen 1e-3 px criterion) and **DR-1 H1 FAIL** (1.4816284e-02 px).
  The device-independent perception stack passes: **C2 135/135, C2.1 251/251, C2.1.1 19/19**.

What is needed to unblock: the 14 company-supplied fields listed in
`stage_c_deploy/STAGE_C_DEPLOYMENT_READINESS_FINAL.md` §17.

## Where to look

**See it run:** `python stage_c_deploy/demo/pipeline_demo.py --cloud3d` — one stereo
pair through the whole graph (disparity, metric depth, object edges, point cloud),
click any panel to measure a point in metres. See
[`stage_c_deploy/demo/README.md`](stage_c_deploy/demo/README.md).

**Start here:** [`RESULTS_INDEX.md`](RESULTS_INDEX.md) — the current index of every phase,
stage, run and audit, with status, source and purpose for each, plus a reader-hazard list.

| Material | State | Entry point |
|---|---|---|
| Phase 0 — baseline lock | **HISTORICAL / CLOSED** | `phase0/docs/PHASE_0_FINAL_REPORT.md` |
| Phase 1 — reference forensics, arm campaign | **HISTORICAL / CLOSED** | `PHASE_1_FINAL_REPORT.md`, `docs/`, and the original README below |
| Phase 2 (re-scoped) — P2A optimization + export | **HISTORICAL / CLOSED**; its "Model Deployed" wording is **SUPERSEDED** | `PHASE_2_FINAL_REPORT.md` (carries a supersession notice) |
| Phase 2 (superseded correspondence line) | **HISTORICAL / CLOSED at Level D** | `phase2/docs/POST_CLOSURE_RESEARCH_AUDIT.md` |
| Stage A — gap diagnostics | **CLOSED** (architecture gate: NO ARCHITECTURE JUSTIFIED) | `stage_a_diagnostics/DIAGNOSTIC_REPORT.md` |
| Stage B — ARM-P initialization intervention | **CLOSED** (bounded finding; identifiability limits recorded) | `stage_b_armp/ARMP_CLOSURE_RECORD.md` |
| Stage C — device-independent deployment work | **CLOSED where device-independent; BLOCKED where target-dependent** | `stage_c_deploy/STAGE_C_DEPLOYMENT_READINESS_FINAL.md` |
| Whole-system audits | **CLOSED WITH DOCUMENTATION DISCREPANCIES** | `stage_c_deploy/FINAL_WHOLE_SYSTEM_AUDIT.md`, `…_SECOND_PASS.md` |

**Do not read `phase1/results/LEADERBOARD.md` as the current results index** — it is the
historical Phase-1 arm table and stops at ARM-V. Use `RESULTS_INDEX.md`.

## Reproducing this work

**Environment:** Python 3.12, `pip install -r requirements.txt` (torch 2.7.0+cu128, onnx 1.22,
onnxruntime 1.27, numpy 2.5.1 were used for every frozen number).

**Datasets are not in this repository** (42 GB). Fetch them yourself:

- **KITTI 2015 stereo** (`data_scene_flow.zip` + `data_scene_flow_calib.zip`) from the KITTI
  benchmark site → extract to `data/kitti2015/`. This is all you need to re-score every frozen
  result: the contract evaluates scenes **160–199** of the *training* split.
- **SceneFlow** (FlyingThings3D + Driving) only if you want to re-run ARM-P Stage 1 pretraining.
  `data/sceneflow/*/fetch_*.sh` and `extract_ft3d.sh` are the exact scripts used; note Stage 1
  used the **A+C subset, 14,460 triplets** (subset B was missing, Monkaa empty) — see
  `stage_b_armp/20260918T062146Z_stage1_pretrain/README.md`.

**Re-score the frozen contract** (40 scenes, 3,802,797 valid pixels, `disp_occ_0`, GT 1/256):

```
python phase1/harness/frozen_eval.py      # contract definition — never modify it
python phase2/scripts/eval_p2a.py         # P2A / ARM-P checkpoints
python scripts/eval_tier2.py              # Stage-B four-checkpoint scoring
```

Expected: reference ONNX **1.3134471 px**, ARM-P seed 1 **1.1912168 px**, P2A seed 0
**1.4149796 px**. Any deviation means the contract or the environment differs — investigate
before trusting the number.

**Models shipped here** (verify the hash before use — `RESULTS_INDEX.md` §11):

| File | What |
|---|---|
| `stage_b_armp/20260919T012646Z_tier2_seed1/armp/p2a_best.pth` | **the deployment candidate**, ARM-P seed 1, 397,954 params |
| `…/tier2_seed1/control/p2a_best.pth` | its random-init control arm |
| `stage_b_armp/20260918T062146Z_stage1_pretrain/checkpoints/armp_stage1_best.pth` | the SceneFlow-pretrained initialization |
| `phase2/runs/p2a_scale_coverage/p2a_best.pth` | P2A seed 0 (historical candidate / baseline) |
| `stage_c_deploy/armp_stereonet.onnx`, `…_static.onnx` | ARM-P exports (**note: C1 parity FAIL**) |
| `phase2/deploy/p2a_stereonet.onnx` | P2A export (parity PASS) |
| `reference/onnx/stereonet.onnx` | the Hailo reference model the whole project is measured against |

**Not shipped, and why:** per-pixel raw dumps (`stage_a_diagnostics/raw/*.npz`, 774 MB), the
repacked KITTI upload bundle (348 MB), rendered PNGs, the vendor `upstream/`/`public_repos/`
clones and profiler HTML, the C1 export-variant and instrumented DEBUG graphs, and the
historical `stereonet.hef`. All are regenerable from the shipped code, and every number cited
from them is in the JSON/CSV records that *are* shipped. Vendor artifact hashes:
`reference/MANIFEST.md`.

**Rules that are not optional** (see `CLAUDE.md`): the frozen evaluation contract is
authoritative and is never edited to improve a number; no run directory, checkpoint or record is
ever overwritten; a regression is recorded as a regression.

---

*Everything below is the original Phase-1 README, preserved unchanged as historical material.*

# Stereo Depth Vision — Phase 1

Forensics, reproduction and characterisation of the Hailo StereoNet reference
deployment, for the Microchip stereo depth vision project.

**Start here:**

| Document | For |
|---|---|
| [`docs/phase_1_project_report.md`](docs/phase_1_project_report.md) | What was done, what was found, what it means, what is next — read this first |
| [`PHASE_1_FINAL_REPORT.md`](PHASE_1_FINAL_REPORT.md) | The charter deliverable: every Phase 1 question answered in order |
| [`docs/phase_1_closure_audit.md`](docs/phase_1_closure_audit.md) | The release gate: what was verified, and what remains unknown |
| [`docs/`](docs/) | Twelve knowledge-base documents with the technical detail |

---

## What this repository is

- A **faithful reconstruction** of a specific deployed model — Hailo's Model Zoo
  StereoNet — rebuilt from its public ONNX and verified against it to a relative
  1e-7 at every stage (EXP-011).
- An **evaluation framework** that reproduces Hailo's published accuracy figure
  exactly (8.2237 against 8.223, EXP-005) after recovering the protocol behind it.
- A **reverse-engineering record**: what the deployed system does, how it differs
  from the paper and from its own upstream implementation, and where the
  published numbers come from.
- A **profiling and analysis environment** for compute, memory, precision and
  failure behaviour.

## What this repository is **not**

**It is not a production stereo-depth system, and does not claim to be.** The
following are deliberately absent and were never in Phase 1's scope:

| Absent | Note |
|---|---|
| Camera synchronisation | The reference application assumes it; nothing here provides it |
| Rectification | Assumed by the architecture, never performed or checked |
| Calibration ingestion from a live rig | KITTI calibration is parsed for evaluation only |
| Confidence or validity prediction | The reference model has no such output |
| Production metric-depth inference | `src/geometry/stereo.py` exists to *evaluate* depth, not to serve it |
| Camera drivers or capture pipeline | Out of scope |

`src/geometry/stereo.py` implements `Z = fB/d` because the reference application
does not (see below), and depth accuracy could not otherwise be measured. It is
an evaluation component, not a deployable depth service.

**The Hailo application studied here outputs disparity, not metric depth.** Its
entire postprocess casts the network output to an 8-bit image; it contains no
calibration, baseline, focal length or depth conversion. See
[`docs/reference_pipeline.md`](docs/reference_pipeline.md).

## Evidence categories

Kept strictly apart throughout, and never merged:

| Tag | Meaning |
|---|---|
| `SOURCE` | Published by Hailo or another third party |
| `MEASUREMENT` | Measured by us, on our hardware, with an experiment ID |
| `INFERENCE` | Derived, with the derivation stated |
| `HYPOTHESIS` | Proposed, not established |
| `UNKNOWN` | Could not be established, and is not guessed |

Our RTX 4060 and CPU timings are **our measurements of our hardware**. They are
never presented as Hailo silicon performance, and Hailo's published figures are
never presented as ours.

## Known evidence gaps

Stated here so they are not mistaken for completed work:

- **Scene-class failure analysis: NOT COMPLETED.** Requires Middlebury 2014,
  which has not been downloaded. Remaining Phase 1 evidence gap.
- **Hailo silicon behaviour: UNKNOWN.** No physical device. Hailo's profiler
  report is a **post-placement compiler estimate**, not a measured run — its
  model-level FPS and latency fields are `N/A`.
- **Competitor measurement: NOT PERFORMED.** All twelve competitor entries are
  abstract-level source reading; none was implemented, run or benchmarked.
- **Full-scale training: NOT PERFORMED.** EXP-016 is a short convergence proof
  only, and its validation did **not** improve — see
  [`experiments/EXP-016/CORRECTION.md`](experiments/EXP-016/CORRECTION.md).

## Layout

```
PHASE_1_FINAL_REPORT.md   the report; read this first
docs/                     project report, closure audit, and 12 knowledge-base documents
src/                      independent implementation, geometry, datasets, metrics
scripts/                  one script per experiment, plus the verifiers
experiments/EXP-xxx/      one immutable record per run; none deleted
results/                  derived tables and analyses
reference/                vendor artifacts, read-only, hashed in MANIFEST.md
tests/                    unit and regression tests
```

## Restoring the reference artifacts

`reference/` holds vendor and upstream artifacts that are downloaded rather than
committed. What *is* committed is `reference/manifest.json`, recording every
artifact's URL, size and SHA-256. To restore them into a fresh clone:

```
python scripts/fetch_reference.py            # download and verify (~270 MB)
python scripts/fetch_reference.py --verify   # check what is present, no download
```

A mismatch means the upstream artifact changed. Investigate rather than
overwriting the recorded hash: every result was produced from the bytes the
manifest describes.

## Verifying

```
python -m compileall -q src scripts tests
python -m pytest tests/ -q        # 75 tests
python scripts/verify_claims.py   # 73 checks against the experiment records
```

`verify_claims.py` asserts that the figures quoted in the documents match the
corresponding `experiments/EXP-xxx/metrics.json`, so a transcription error or a
conclusion the data does not support cannot reach a release.

**Two counts are expected, and the difference is reported rather than hidden.**
A bare clone runs **65 of the 73** checks; the other 8 read the Hailo
application source and need `reference/` restored first. The script names every
skipped check and prints the full total, so a run performing fewer checks can
never be mistaken for a clean pass. Run `fetch_reference.py` for all 73.

The KITTI calibration test skips when the dataset is absent; that skip is
intentional and declared in the test, giving 74 passed + 1 skipped in a bare
clone against 75 passed with the dataset present.

## Tags

| Tag | Commit | Meaning |
|---|---|---|
| `phase-1` | `ba1cf84` | Original archival freeze |
| `phase-1-final` | `2278c10` | First corrective release: training-conclusion regression |
| `phase-1-final-r2` | `5dd30c7` | Second corrective release: profiler stage mapping |
| `phase-1-frozen` | see below | **Closure audit passed. The definitive Phase 1 freeze.** |

Earlier tags are never moved, so each release remains inspectable as it stood.

History is preserved and never rewritten: experiment records reference commit
hashes, so rewriting would break their traceability.

## Phase 2

Not started. The opening hypothesis is **H1 — what is a working cost volume
worth?**, since the deployed model performs no disparity search (EXP-010). The
corrected shift already exists behind a configuration flag that Phase 1 never
enables, and it must stay that way until Phase 2 begins. See
[`docs/research_questions.md`](docs/research_questions.md).
