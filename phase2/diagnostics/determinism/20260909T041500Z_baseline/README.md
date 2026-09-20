# Stage B — deterministic baseline re-establishment, run `20260909T041500Z_baseline`

**BASELINE/DIAGNOSTIC RECORD. Not an architecture experiment.** No architecture
was changed, no hyperparameter tuned, no hypothesis tested. Nothing outside this
directory was written.

**Purpose.** Establish a trustworthy contemporary H2 / E3b-arm-A baseline under
a reproducible training protocol, and test whether two fresh processes reproduce
each other bit for bit.

## Configuration

Identical to `EXP-E3B-REFINEMENT-CAPACITY-001-ARM-A`: `StereoNet`,
`cost_volume_shift="left"`, `StandardisedDisparityRegression`, six refinement
blocks (dilations 1,2,4,8,1,1), 423,586 parameters, 16.1995 GMAC @256×512,
`hailo_calib` (160 scenes) train, `hailo_val` validate, random 256×512 crops with
σ=0.1 gain jitter, masked smooth-L1, Adam lr 1e-3, cosine `T_max=200`, batch 2,
fp32, **seed 0**, **200 epochs**.

**The only protocol change from the historical run:**

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# CUBLAS_WORKSPACE_CONFIG=:4096:8, set before the CUDA context exists
```

The recipe is not restated anywhere: `stageb_deterministic_baseline.py` calls
`exp_e3b_refinement_capacity.run_training("A", ...)` itself and only repoints the
module's output paths and record identity — the same override pattern
`exp_e3c_refinement_floor.py` uses. Every scientific script is imported
read-only.

## Runs

| run | record id | role |
|---|---|---|
| `A` | `STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA` | the contemporary deterministic baseline |
| `B` | `STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNB` | fresh-process reproducibility check, identical configuration |

## Files

| file | contents |
|---|---|
| `README.md` | this file |
| `RESULTS.md` | **the answers, and the verdict** |
| `RUN_A.md`, `RUN_B.md` | the two runs in full |
| `ENVIRONMENT.txt` | git provenance, preflight and full environment for both runs |
| `results/preflight.json` | preflight, including the initial-weight-hash check |
| `experiments/*/{config,env,log,metrics}.json` | the harness's own records, written by the historical code path |
| `run_{A,B}_meta.json` | wall clock, control settings, weight and checkpoint hashes |
| `eval40_run_{A,B}.json` | **40-scene evaluation, including all 40 per-scene rows** |
| `comparison.json` | machine-generated A-vs-B, runtime and historical comparison |
| `checkpoints/` | final checkpoint + 6 weight snapshots per run |

Script (**diagnostic/baseline-only**, marked as such in its docstring) is one
level up: `phase2/diagnostics/determinism/stageb_deterministic_baseline.py`.

## Superseded first attempt

`phase2/diagnostics/determinism/20260909T024500Z_baseline/` is a **preserved,
non-authoritative** first attempt. Its 200 epochs completed, then it raised on
the final bookkeeping line because the wrapper passed a relative output path.
See its `NOTE.md`. It is never cited as the Stage B result; it is used only as an
additional reproduction sample, and `RESULTS.md` labels it as such.

## What was not touched

- `src/` and every Phase 1 path — `git diff phase-1-frozen -- src scripts` empty
  before and after.
- Every Phase 2 scientific script, preregistration, report and experiment record.
  **No E3b, E3c, O6 or H2 verdict, metric or text was altered.**
- The `phase-1-frozen` tag.

## Claim tags

`MEASURED` — produced by these runs. `HISTORICAL` — a previously recorded value,
cited unchanged. `DERIVED` — arithmetic on them. `INFERRED` — a reading the
evidence does not force. `UNKNOWN`.
