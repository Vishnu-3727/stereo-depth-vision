# Correction to the EXP-017 per-stage rollup

**The per-stage MAC breakdown recorded in this experiment is wrong and is
withdrawn.** It is superseded by **EXP-018**, which repeats the analysis with a
corrected stage mapping. This record is preserved unaltered.

Found by an external audit of the repository, not by us.

## What was wrong

`scripts/exp_profiler_report.py` mapped Hailo's compiled convolutions as:

```
conv1  .. conv17   feature extraction
conv18 .. conv22   aggregation (3D)
conv23 ..          refinement
```

The layer shapes in the profiler report contradict that. `conv18` is
368×1232, 3 → 32 channels, 5×5 — the same workload as `conv1`, because it is the
**right branch of the Siamese feature extractor**, not an aggregation layer. The
correct boundaries are:

```
conv1  .. conv17   feature extraction, left branch
conv18 .. conv34   feature extraction, right branch
conv35 .. conv39   3D aggregation   (25x79, 448 -> 384 channels; conv39 emits
                                     the 12 disparity costs)
conv40 .. conv53   full-resolution refinement  (conv40 takes 4 input channels:
                                     disparity concatenated with the left RGB)
```

The incorrect mapping folded the entire right feature-extractor branch **and**
the real 3D aggregation into refinement.

## What changes

| Stage | EXP-017 (withdrawn) | EXP-018 (corrected) | Our ONNX analysis (EXP-001) |
|---|---:|---:|---:|
| Refinement | 95.2 % | **90.6 %** | 90.6 % |
| Aggregation (3D) | 2.2 % | **4.2 %** | 4.2 % |
| Feature extraction (both branches) | 2.6 % | **5.1 %** | 5.1 % |

The corrected figures agree with our independent ONNX analysis to within 0.02
percentage points. The withdrawn figures disagreed with it by nearly five
points.

## What does not change

- **Refinement is still the dominant stage**, and still by a wide margin.
- **The eight slowest layers by modelled throughput are still all refinement
  convolutions**, with `conv50` setting the bottleneck at 43.03 FPS. Under the
  corrected mapping those layers are `conv40`+, which is refinement either way.
- Every model-level figure is unaffected: `weights = 623,138`,
  `macs_per_image = 56,258,882,372`, `ops_per_image = 112,111,950,720`,
  6 device contexts, uniform 8/8/8 quantisation, `profiling_mode: post_placement`
  with model-level FPS and latency `N/A`.
- The conclusion that this report is a **compiler estimate, not a measurement of
  silicon**, is unaffected.

So the direction of the finding stands; the specific per-stage split does not.

## Why it was not caught

The withdrawn rollup put refinement at 95.2 % while our own ONNX analysis,
recorded in EXP-001, put it at 90.6 %. **That five-point discrepancy between two
supposedly independent routes to the same quantity was visible at the time and
was not questioned.** Two measurements of the same thing disagreeing is exactly
the signal this project's evidence discipline exists to surface, and it was
treated as agreement because both supported the same qualitative conclusion.

## The fix

- `src/common/profiler_stages.py` holds the mapping, derived from and documented
  against the layer shapes, and is unit tested rather than living inline in a
  script.
- `scripts/exp_profiler_report.py` uses it, and now **fails loudly** if the
  compiled per-stage shares diverge from the ONNX analysis by more than one
  percentage point. The error that produced this correction would now raise
  rather than be recorded.
- `tests/test_profiler_stages.py` pins every range boundary, the
  defusion-suffix handling, a shape-based classification cross-check that does
  not use the ranges at all, and an assertion that the known-bad 95.2 % figure is
  unreachable.
- `scripts/verify_claims.py` checks the corrected shares against the records.

## Status

`metrics.json`, `env.json`, `config.json` and `log.txt` in this directory are
unaltered. Only this interpretation is corrected. Use **EXP-018** for any
per-stage figure.
