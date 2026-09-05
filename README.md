# Stereo Depth Vision — Phase 1

Forensics, reproduction and characterisation of the Hailo StereoNet reference
deployment, for the Microchip stereo depth vision project.

**Start here:**

| Document | For |
|---|---|
| [`docs/phase_1_project_report.md`](docs/phase_1_project_report.md) | What was done, what was found, what it means, what is next — read this first |
| [`PHASE_1_FINAL_REPORT.md`](PHASE_1_FINAL_REPORT.md) | The charter deliverable: every Phase 1 question answered in order |
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
docs/                     the project report plus 12 knowledge-base documents
src/                      independent implementation, geometry, datasets, metrics
scripts/                  one script per experiment, plus the verifiers
experiments/EXP-xxx/      one immutable record per run; none deleted
results/                  derived tables and analyses
reference/                vendor artifacts, read-only, hashed in MANIFEST.md
tests/                    unit and regression tests
```

## Verifying

```
python -m pytest tests/ -q        # unit and regression tests
python scripts/verify_claims.py   # every headline number against its record
```

`verify_claims.py` asserts that the figures quoted in the documents match the
corresponding `experiments/EXP-xxx/metrics.json`, so a transcription error or a
conclusion the data does not support cannot reach a release.

The KITTI calibration test skips when the dataset is absent; that skip is
intentional and declared in the test.

## Tags

| Tag | Meaning |
|---|---|
| `phase-1` | Original archival freeze |
| `phase-1-final` | Final corrective archival release |

History is preserved and never rewritten: experiment records reference commit
hashes, so rewriting would break their traceability.

## Phase 2

Not started. The opening hypothesis is **H1 — what is a working cost volume
worth?**, since the deployed model performs no disparity search (EXP-010). The
corrected shift already exists behind a configuration flag that Phase 1 never
enables, and it must stay that way until Phase 2 begins. See
[`docs/research_questions.md`](docs/research_questions.md).
