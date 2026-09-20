# CORRECTION — stereo-verdict reader in `cmd_compare`, 2026-09-10

**Status: disclosed, not hidden. The first `comparison.json` this batch produced
was WRONG and is preserved beside the corrected one. No training was affected,
no threshold changed, no gate altered.**

## What happened

The first run of `compare` printed:

```
stereo functional: {"0": false, "1": false, "2": false}
VERDICT: 4-BLOCK-STEREO-FUNCTIONALITY-FAILURE
```

That was a **defect in this batch's analysis code**, not a result.

`_stereo_from_metrics` read `stereo_verdict_epoch_200` from the top level of
`experiments/<run>/metrics.json`. The experiment recorder nests every metric
under a `metrics` object, so the top-level lookup returned `{}`, and
`{}.get("STEREO_FUNCTIONAL")` is `None` → `False`. Every run therefore reported
as a stereo failure — including **4b/seed0, a frozen record independently
confirmed as stereo-functional**, and the same lookup would have failed on
batch 2's runs too.

## How it was caught

The verdict claimed a stereo failure for a frozen record already known to be
stereo-functional (right-image penalty +79.251 at epoch 200). A verdict that
contradicts a verified frozen record is a code fault until proven otherwise, so
the record was inspected before anything was reported.

The training logs had recorded the truth all along:

```
EXP-BLOCKCOUNT-FULL-001-ARM-C-SEED1 [09:35:28] STEREO VERDICT at epoch 200: STEREO-FUNCTIONAL
```

## The fix

`_stereo_from_metrics` now reads `doc["metrics"]` (falling back to the document
itself) and **raises** if `stereo_verdict_epoch_200` is absent, rather than
silently treating a missing verdict as a failure. Failing loudly is the point:
an absent verdict is a bookkeeping fault, never evidence of stereo collapse.

The stereo criteria themselves — the frozen 20-D1-point threshold, the probes,
the entropy and disparity-SD floors — are **untouched**. Only the lookup path
changed.

## Effect on the verdict

| | first run (defective) | corrected |
| --- | --- | --- |
| stereo functional 0 / 1 / 2 | false / false / false | **true / true / true** |
| VERDICT | `4-BLOCK-STEREO-FUNCTIONALITY-FAILURE` | **`4-BLOCK-INCONCLUSIVE-AT-SEEDS-0-1-2`** |

Every EPE, D1, RMSE, range, pairing and per-scene figure was **identical in both
runs** — the defect touched only the stereo gate, which is checked first and
short-circuited the verdict.

The corrected verdict is the one the pre-registered rule yields from the
measured EPE values, which were never in question:
4b EPE = 2.3387143 / 2.6553489 / 2.5756633 straddles the frozen 6-block range
`[2.2955182, 2.3982217]` → Verdict C.

## Preserved

- `comparison_SUPERSEDED_stereo_reader_bug.json`
- `compare_stdout_SUPERSEDED_stereo_reader_bug.log`

Both are kept unmodified. Nothing was overwritten silently.

## Why this is not a post-hoc change to the decision rule

The three pre-registered verdicts and the frozen decision boundary are exactly
as registered. The corrected code makes the *stereo gate* report what the frozen
recipe actually measured. Had the fix gone the other way — had the runs truly
not been stereo-functional — the verdict would have stayed
`4-BLOCK-STEREO-FUNCTIONALITY-FAILURE`, which is the harsher outcome. The fix
was made to the reader, before the record was written up, and both versions are
retained so the change is auditable.
