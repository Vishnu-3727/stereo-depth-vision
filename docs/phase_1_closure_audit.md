# Phase 1 Closure Audit

Release gate before Phase 2. Every finding below was re-derived from the raw
artifacts and experiment records rather than read out of the documents.

---

## Final Decision

# **PHASE 1 FROZEN**

No HIGH or MEDIUM correctness defect remains. One MEDIUM defect was found during
this audit and fixed; one LOW documentation gap was found and fixed. All release
gates pass, including clean-clone reproducibility proven end to end.

---

## Critical Findings

**None outstanding.**

Two defects were found during this audit. Both were within closure scope and both
were fixed:

### MEDIUM — `verify_claims.py` silently lost 8 checks in a clean clone

`verify_claims.py` reported **73 checks, 0 failures** in the developer's working
tree but **65 checks, 0 failures** in a fresh clone. Eight checks — the ones that
read the Hailo application source from `reference/`, which is downloaded rather
than committed — were skipped, and the summary line gave no indication of it.

A reader in a fresh clone would have seen a clean pass over a smaller check set
and had no way to know. That is the same class of false assurance the script
exists to prevent.

**Fixed.** Skips are now counted, named individually, and reported against the
full total: `65 checks run, 0 failures, 8 skipped (73 defined)`, followed by the
list and the command that restores them.

### LOW — reference artifacts had no documented restore path

`reference/manifest.json` recorded every artifact's URL and SHA-256, so the
information existed, but nothing turned it into a reproducible action. A fresh
clone could not restore the eight-check set.

**Fixed.** `scripts/fetch_reference.py` replays the manifest and verifies every
hash. Proven end to end: a fresh clone with nothing in `reference/` was restored
and reached the full 73 checks (see Verification).

### Also fixed

`AUDIT_REPORT.md` — the external audit that found the profiler mapping defect —
was committed without any indication its findings had been actioned. It now
carries a status header pointing at the corrections, with the original text
retained unedited.

---

## Fixes Performed

Only changes actually made during this audit:

| File | Change |
|---|---|
| `scripts/verify_claims.py` | Skip accounting: skipped checks are counted, named and reported against the full total |
| `scripts/fetch_reference.py` | **New.** Restores and hash-verifies `reference/` from the committed manifest |
| `README.md` | Restore instructions; the 73-vs-65 and 75-vs-74 count distinction explained |
| `AUDIT_REPORT.md` | Status header marking the findings actioned; body unedited |
| `PHASE_1_FINAL_REPORT.md` | Verification section and reproduction commands updated |
| `docs/phase_1_project_report.md` | Verification table updated |
| `docs/phase_1_closure_audit.md` | **New.** This document |

No experiment was altered. No history was rewritten. No scientific conclusion was
changed.

---

## Verification

Actual results, run for this audit:

| Command | Working tree | Fresh clone |
|---|---|---|
| `python -m compileall -q src scripts tests` | **PASS** | **PASS** |
| `python -m pytest tests/ -q` | **75 passed** | **74 passed, 1 skipped** |
| `python scripts/verify_claims.py` | **73 run, 0 failures, 0 skipped** | **65 run, 0 failures, 8 skipped (73 defined)** |
| `python scripts/fetch_reference.py --verify` | **9 verified, 0 mismatched** | 9 missing before fetch |
| `python scripts/fetch_reference.py` | — | **9 downloaded, 9 verified, 0 mismatched** |
| `verify_claims.py` after restore | — | **73 checks, 0 failures** |

The single test skip is `tests/test_geometry.py::test_kitti_calibration_matches_published_values`,
which carries an explicit `pytest.skip` on a missing dataset. It is intentional
and declared.

**No untracked artifact is secretly required.** A bare clone compiles, runs 74 of
75 tests, and runs 65 of 73 checks, reporting exactly what it could not do.

**Git status:** working tree clean.

---

## Scientific Findings Verified

Each re-derived independently for this audit, from raw artifacts, without
importing the repository's own analysis modules where avoidable.

### Finding 1 — degenerate cost volume · **PASS**

From EXP-010's record: **max |slice_k − slice_0| = 0.0** across 60 measurements
(5 scenes × 12 slices). All 11 slice operations are `starts=[0] ends=[77]`, and
in every case the zero padding is concatenated *after* the features. The
mechanism and the consequence both hold.

### Finding 2 — the published 8.223 · **PASS**

`hailo_exact` reproduces **8.2237** against a published 8.223. The official
protocol gives **EPE 1.3134 px**. The Hailo variant is confirmed as per-image
averaging with ground truth scaled by 255 on `disp_occ_0`, and under the official
variant the headline figure is numerically identical to the D1 rate — which is
what establishes that 8.223 is an outlier percentage, not an end-point error.

### Finding 3 — independent reproduction · **PASS**

Worst relative mean absolute difference across **all six** tapped stages:
**7.596 × 10⁻⁷**. No tolerance was loosened. Parameter counts match exactly
(423,586 unique / 623,138 per occurrence).

### Finding 4 — compute split · **PASS**, and the two routes agree

Recomputed from the raw artifacts, with the profiler route implemented
independently of `src/common/profiler_stages.py`:

| Stage | ONNX route | Profiler route | Agreement |
|---|---:|---:|---:|
| Feature extraction | 5.13 % | 5.13 % | 0.000 points |
| Aggregation (3D) | 4.23 % | 4.22 % | 0.012 points |
| Refinement | 90.64 % | 90.62 % | 0.018 points |

EXP-018's recorded shares equal this independent recomputation to within 0.05
points. The agreement was not manufactured: the two routes start from different
artifacts (`results/architecture/layers.json` from the ONNX,
`results/profiler/layers.csv` from Hailo's compiler) and use separately written
classifiers.

### Profiler classifier · **PASS**

Boundary behaviour confirmed live at every case the brief names:

```
conv1  -> feature extraction (left)     conv35 -> aggregation (3D)
conv17 -> feature extraction (left)     conv39 -> aggregation (3D)
conv18 -> feature extraction (right)    conv40 -> refinement
conv34 -> feature extraction (right)    conv53 -> refinement
```

Defused variants resolve correctly (`conv47_sd12` → refinement,
`stereonet/conv34_sd0` → feature extraction (right), `conv48_ws` → refinement),
and routing layers are not mistaken for convolutions
(`ws_from_conv47_ws_to_conv47_sd0-1` → data movement).

**The percentages are not hard-coded.** The only percentage constants in the
analysis are the ONNX-derived values used as a divergence tripwire; the reported
figures are computed from the CSV, and were confirmed equal to an independent
recomputation. Shape-based validation exists in
`tests/test_profiler_stages.py`, classifying by `input_height`/`input_width`
without reference to the conv ranges at all.

### EXP-017 / EXP-018 provenance · **PASS**

- EXP-017 **still records the withdrawn 95.18 %**, unaltered. It was not rewritten.
- EXP-017 carries `CORRECTION.md`, which names EXP-018 as its successor and uses
  the word *withdrawn*.
- EXP-018 contains the corrected analysis and is reproducible.
- The raw profiler artifact is preserved and hash-verified.
- No experiment was deleted; ids are contiguous EXP-001…EXP-018.
- Model-level figures are **identical** in both records — `weights = 623,138`,
  `macs_per_image = 56,258,882,372`, `ops_per_image = 112,111,950,720`,
  ratio 1.9928 — confirming the mapping error touched only the per-stage rollup.

### Training conclusion (EXP-016) · **PASS**

Conclusions are derived in `src/common/conclusions.py` from measured values.
EXP-016's own numbers (18.497 → 19.216) classify as *did not improve*. The AST
guard against hard-coded outcome claims passes, and was previously confirmed to
fail against the pre-fix file recovered from tag `phase-1`.

### Hardware language · **PASS**

No document presents a compiler estimate as a device measurement. Every mention
of **43.03 FPS** — seven across five documents — is qualified as modelled,
estimated, post-placement or compiler-derived. Three regex hits for "measured
silicon" were inspected and are all negations: *"what it does **not** contain is
measured silicon behaviour"*, *"**Measured silicon behaviour is still
UNKNOWN**"*. Physical Hailo runtime remains explicitly UNKNOWN.

### Documentation consistency · **PASS**

Every occurrence of **95.2 %** is in a withdrawal, correction, historical or
known-bad-value context. It appears nowhere as a current valid result. The
current figures 90.6 / 4.2 / 5.1 are used consistently.

### Reference artifacts · **PASS**

All 10 manifest entries verified: **9 hash-verified downloads plus 1 correctly
marked derived**. 0 mismatches.

### Cost-volume boundary · **PASS**

`cost_volume_shift` defaults to `"none"` in both `StereoNetConfig` and
`build_cost_volume`. **No script anywhere enables `shift="left"`.** EXP-010
remains the reference finding. No Phase 2 experiment has entered Phase 1.

---

## Known Research Gaps

| Gap | Classification |
|---|---|
| Middlebury scene-class failure analysis | **DEFERRED / UNKNOWN.** No experiment exists. Documented as NOT COMPLETED in README, `failure_analysis.md`, `bottleneck_report.md` and the final report. Not fabricated, and not a defect |
| Physical Hailo latency, FPS, power, utilisation | **UNKNOWN.** No device was available. Not inferred from the compiler profiler anywhere |
| Competitor measurement | **DEFERRED.** Conceptual/abstract comparison complete for twelve architectures; measured comparison not performed and stated as such |
| Full-scale Scene Flow training | **DEFERRED.** EXP-016 is a convergence proof only; recipe recorded |
| Original StereoNet paper, full text | **DEFERRED.** Read at abstract level; the paper-vs-deployment comparison is marked partial |

None of these is a correctness defect. Each is correctly documented as incomplete.

## Out of Scope

Not Phase 1 defects, and verified as documented exclusions in `README.md`:
live calibration ingestion, rectification, frame synchronisation, confidence
output, production metric-depth interface, camera drivers.

---

## Repository Integrity

| | |
|---|---|
| Experiments | **18** (EXP-001…EXP-018), contiguous, all `completed`, all with a resolving git commit and full config |
| Tests | **75** (74 + 1 intentional dataset skip in a bare clone) |
| Claim checks | **73** (65 + 8 artifact-dependent, reported) |
| Corrections preserved | EXP-002→003, EXP-008 annotation, EXP-009→011, EXP-016 CORRECTION, EXP-017→018 CORRECTION |
| `phase-1` tag | `ba1cf84` — **unchanged** |
| History | 13 commits before this audit, linear, **no rewrite**. The one reflog entry matching `reset` is a no-op `reset: moving to HEAD` from a stash |
| Working tree | clean |
| Tracked binaries | none |

---

## Phase Boundary

**Phase 2 was NOT started during this audit.**

**The corrected disparity shift remains disabled** — `cost_volume_shift` defaults
to `"none"` and no script enables it. H1 was not run.

---

## Final Recommendation

**Phase 1 is frozen and ready for Phase 2.**

Every completed claim was re-derived and holds. Every known mistake is preserved
alongside its correction and is traceable. Every unknown is explicitly labelled.
The repository is reproducible from a clean clone, including the artifact restore
path, which was proven rather than asserted.

Phase 2 opens with **H1 — what is a working cost volume worth?**

---

*Audit performed against the repository at the closure commit. Verification
commands and their actual outputs are recorded above.*
