# Audit Report

> **Status: ACTIONED.** Both findings below were confirmed against the
> repository and fixed. The High finding (incorrect profiler stage mapping) was
> corrected in **EXP-018**; EXP-017 is preserved unaltered and carries
> `experiments/EXP-017/CORRECTION.md`. The Low finding (stale test count) was
> fixed. See the closure audit in `docs/phase_1_closure_audit.md` for the
> verification, and the integrity records in `PHASE_1_FINAL_REPORT.md` and
> `docs/phase_1_project_report.md`.
>
> This document is retained unedited below as the external record of what was
> found, and when.

Date: 2026-09-05

## Scope

This audit reviewed the repository's implementation, tests, experiment records,
claim-verification tooling, and the newly added Hailo profiler-report analysis.
It did not alter experiment records or vendor artifacts.

## Result

The Phase 1 inference reconstruction remains well supported: the recorded ONNX
equivalence run agrees at every tested stage to roughly `1e-7` relative error,
and the Hailo evaluation protocol is reproduced to `0.001` percentage points.

The project is not fully correct, however. The compiler-profiler stage mapping
misclassifies a substantial amount of work, invalidating its derived per-stage
MAC shares and every document claim based on the incorrect rollup.

## Findings

### High: compiler-profiler stages are mapped to the wrong architecture regions

`scripts/exp_profiler_report.py` maps compiled convolution names as follows:

- `conv1` through `conv17`: feature extraction
- `conv18` through `conv22`: 3D aggregation
- `conv23+`: refinement

That mapping is contradicted by the ONNX layer table and by the compiled layer
shapes. The correct mapping is:

- `conv1` through `conv17`: left feature-extractor branch
- `conv18` through `conv34`: right feature-extractor branch
- `conv35` through `conv39`: 3D aggregation
- `conv40+`: full-resolution refinement

For example, `conv18` has the same high-resolution first-convolution workload as
`conv1`, while `conv35` is the first 3D aggregation convolution. The current
mapping therefore classifies the right branch and the aggregation stage as
refinement.

Impact:

- EXP-017 reports refinement as `95.2%` of compiled MACs, feature extraction as
  `2.6%`, and aggregation as `2.2%`.
- Those figures are not valid per-stage measurements. The independent ONNX
  analysis instead places both feature branches together near `5.1%`, 3D
  aggregation near `4.2%`, and refinement near `90.6%` of MACs.
- The incorrect `95.2%` claim propagates to `PHASE_1_FINAL_REPORT.md`,
  `docs/hardware_analysis.md`, `docs/bottleneck_report.md`, and
  `docs/research_questions.md`.

Required remediation:

1. Correct `stage_of()` in `scripts/exp_profiler_report.py` to use the four
   ranges above.
2. Add unit tests covering `conv17`, `conv18`, `conv34`, `conv35`, `conv39`, and
   `conv40`, including defused-name variants.
3. Re-run the profiler extraction and EXP-017 analysis as a new immutable
   experiment; do not overwrite the existing record.
4. Update all documents that cite the incorrect rollup, and extend
   `scripts/verify_claims.py` to check the corrected stage totals.

### Low: stale test count in the reproduction instructions

`PHASE_1_FINAL_REPORT.md` correctly says elsewhere that 49 tests pass, but its
"Reproducing everything" command list still labels the test command as
"29 tests." The command is correct; its comment should say 49 tests.

## Verified Checks

- `pytest -q`: 49 passed. The only warnings were third-party `thop`
  deprecations and an environment-specific pytest-cache permission warning.
- `python -m compileall -q src scripts tests`: passed.
- `python scripts/verify_claims.py`: 65 checks, 0 failures.
- The training-conclusion regression remains fixed: conclusions are derived
  from recorded values in `src/common/conclusions.py`, with focused regression
  coverage in `tests/test_conclusions.py`.

## Scope Note

This remains a Phase 1 reconstruction and analysis repository, not a production
stereo-depth system. Its stated exclusions -- live calibration ingestion,
rectification, synchronisation, confidence output, and a production metric-depth
interface -- are accurately documented and are not audit defects within the
declared scope.
