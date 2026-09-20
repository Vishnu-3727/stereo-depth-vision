# EXP-FACTORIAL-SHIFT-NONE-STANDARDISED-001 — `20260909T071500Z`

**Factorial cell. Not an architecture search, not a model-development session.**
One treatment arm, one execution, no tuning, no rescue.

**Question.** Does `cost_volume_shift` provide useful stereo value after the H2
standardised readout has removed the soft-argmin saturation failure?

**Answer.** `SHIFT-IMPORTANT` — see `RESULTS.md`.

## The 2×2, now complete at 200 epochs

| | readout = raw | readout = standardised |
|---|---|---|
| **shift = none** | `HISTORICAL` `EXP-H1-BASE-v2` | **this experiment** |
| **shift = left** | `HISTORICAL` `EXP-H1-WORKING-v2` | `HISTORICAL` H2 / Stage B control |

## Design

- **Control:** Stage B deterministic Run A, `STAGEB-DETERMINISTIC-BASELINE-ARM-A-RUNA`
  (`phase2/diagnostics/determinism/20260909T041500Z_baseline`). Read-only; not
  retrained, not modified.
- **Treatment:** identical in every respect except `cost_volume_shift: left → none`,
  flipped **in place** after `build_model(0)` consumed the RNG, so both arms start
  from the bit-identical initialisation `c4d02385e3da28a5`.
- **Protocol:** Stage B's mandatory deterministic controls
  (`use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). No error was
  suppressed; none was raised.
- **Recipe:** not restated anywhere — the arm is trained by calling
  `exp_e3b_refinement_capacity.run_training("A", ...)` itself, with only the
  module's output paths, record identity and model builder repointed.
- **No materiality band.** The historical E3b/E3c band is invalidated and was
  neither reused nor replaced. The effect is classified by measured Δ, per-scene
  consistency, stereo evidence, magnitude versus measured protocol divergence, and
  mechanism.

## Files

| file | contents |
|---|---|
| `PREREGISTRATION.md` | **frozen before the run** — question, control, treatment, protocol, stop conditions, outcomes A/B/C, no-band rule, probe caveat |
| `RESULTS.md` | **the result and the verdict** |
| `ENVIRONMENT.txt` | git provenance, preflight, full environment |
| `results/preflight.json` | initialisation hash, architecture, initial tensors, degeneracy check |
| `experiments/…/{config,env,log,metrics}.json` | the harness's own records, written by the historical code path |
| `run_meta.json` | wall clock, controls, weight and checkpoint hashes |
| `eval40.json` | 40-scene evaluation, all per-scene rows |
| `probes.json` | stereo probes at epochs 10/50/100/150/200 for **both** treatment and control |
| `comparison.json` | machine-generated treatment-vs-control, aggregate and per-scene |
| `checkpoints/` | final checkpoint + 6 weight snapshots |

Script: `phase2/factorial/shift_none_standardized/exp_factorial_shift.py`.

## Provenance

`HEAD` `58e8a19`; `phase-1-frozen^{commit}` `b4207e5`;
`git diff phase-1-frozen -- src scripts` **empty** at start and end;
working tree `M .gitignore`, `?? phase2/`.

**Phase 2 is not committed to git.** No claim of external immutability is made
for any Phase 2 record, including this one.

## What was not touched

H1, H2, H3, E1, E2, E3, E3b, E3c, O6, Stage A, Stage B, every historical report,
metric and preregistration, and all of Phase 1. This experiment is additive; no
historical conclusion was retroactively repaired.

## Claim tags

`MEASURED` · `HISTORICAL` · `DERIVED` · `INFERRED` · `UNKNOWN`.
