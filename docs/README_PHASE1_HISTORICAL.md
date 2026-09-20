<!-- HISTORICAL. This is the Phase-1 README exactly as it stood at the `phase-1-frozen` tag,
     moved here unedited on 2026-09-20 when the root README was rewritten for the closed
     project. It describes the project as of 2026-09-05 and is NOT the current state.
     Current entry point: ../README.md   Current index: ../RESULTS_INDEX.md -->

# Stereo Depth Vision — Phase 1

Forensics, reproduction and characterisation of the Hailo StereoNet reference
deployment, for the Microchip stereo depth vision project.

**Start here:**

| Document | For |
|---|---|
| [`docs/phase_1_project_report.md`](phase_1_project_report.md) | What was done, what was found, what it means, what is next — read this first |
| [`PHASE_1_FINAL_REPORT.md`](../PHASE_1_FINAL_REPORT.md) | The charter deliverable: every Phase 1 question answered in order |
| [`docs/phase_1_closure_audit.md`](phase_1_closure_audit.md) | The release gate: what was verified, and what remains unknown |
| [`docs/`]() | Twelve knowledge-base documents with the technical detail |

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
[`docs/reference_pipeline.md`](reference_pipeline.md).

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
  [`experiments/EXP-016/CORRECTION.md`](../experiments/EXP-016/CORRECTION.md).

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
python -m pytest tests/ -q        # 75 tests at the Phase-1 freeze; 101 today
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
[`docs/research_questions.md`](research_questions.md).
