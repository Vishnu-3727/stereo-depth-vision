# EXP-BLOCKCOUNT-DETERMINISTIC-001 — `20260909T131500Z`

**Refinement-capacity re-measurement under the Stage B deterministic protocol.**
Not a cost-volume experiment; Stage E stays CLOSED.

**Question.** How much of the six-block refinement capacity is necessary, now
that the readout is known necessary for learning and the shift necessary for
useful correspondence?

**Answer.** Arm B (5 blocks) `CAPACITY-REQUIRED-AT-SEED-0`; arm C (4 blocks)
`INCONCLUSIVE`; **series `INCONCLUSIVE`** — the ordering is non-monotone in block
count, so the differences are not attributable to capacity at one seed. See
`RESULTS.md`.

## Design

- **Control:** Stage B deterministic Run A (6 blocks), **frozen, not retrained**.
  Stage B measured a fresh-process rerun of it to be bit-identical, so retraining
  could only reproduce it. Read-only here. This saved ~82 GPU-minutes and removed
  a variable.
- **Treatments:** arm B (5 blocks, 1-2-4-8-1) and arm C (4 blocks, 1-2-4-8),
  built by `exp_e3b_refinement_capacity.build_arm` unmodified, so every surviving
  weight tensor is bit-identical to the control's.
- **Protocol:** Stage B's mandatory deterministic controls. No error suppressed;
  none raised.
- **No materiality band.** The historical E3b/E3c band is invalidated and was
  neither reused nor replaced.
- **Claim ceiling (pre-registered §7):** bitwise reproducibility removes
  *execution* noise, not *seed* noise. Results are "at seed 0" only.

## Files

| file | contents |
|---|---|
| `PREREGISTRATION.md` | frozen before the run — question, control, arms, protocol, stop conditions, no-band rule, claim ceiling, verdict rule |
| `RESULTS.md` | the result and the verdicts |
| `ENVIRONMENT.txt` | git provenance, preflight, both arms' environments |
| `CONFIG_arm_B.json`, `CONFIG_arm_C.json` | the recorded per-arm configurations |
| `results/preflight.json` | weight identity, parameter counts, measured MACs |
| `experiments/…/{config,env,log,metrics}.json` | harness records from the historical code path |
| `run_meta_{B,C}.json` | wall clock, controls, weight and checkpoint hashes |
| `eval40_arm_{B,C}.json` | 40-scene evaluation, all per-scene rows |
| `probes.json` | stereo probes at epochs 10/50/100/150/200 for both arms **and** the frozen control |
| `comparison.json` | machine-generated three-way comparison, aggregate and per-scene |
| `checkpoints/` | final checkpoints + 6 snapshots per arm |

Script: `phase2/factorial/block_count_deterministic/exp_blockcount_deterministic.py`.

## Provenance

`HEAD` `58e8a19`; `phase-1-frozen^{commit}` `b4207e5`;
`git diff phase-1-frozen -- src scripts` **empty** at start and end;
working tree `M .gitignore`, `?? phase2/`.

**Phase 2 is not committed to git.** No external immutability is claimed for any
Phase 2 record, including this one.

## What was not touched

H1, H2, H3, E1, E2, E3, E3b, E3c, O6, Stage A, Stage B, the factorial shift cell,
every historical report/metric/preregistration, and all of Phase 1. Additive only.
