# NOTE — additive; this record is INCOMPLETE and is not evidence

Arm A was killed by the host at **epoch 146 of 200**, last log line 04:15:00 UTC
(`epoch 146  loss 1.1035  grad 17.580  match 4.976e+00 (100%)  lr 1.63e-04`).
The reason was **system memory pressure on the machine**, not anything about the
experiment: the background command running the arm was terminated externally.

Because the kill landed inside the `Experiment` context manager, the record was
never finalised — it holds `config.json`, `env.json` and `log.txt`, and **no
`metrics.json`**. Nothing here may be cited as a result.

## What the partial run did show, for the audit trail only

- The epoch-10 viability gate **PASSED** and the epoch-100 collapse check
  **PASSED**.
- Matching-path gradient was present on 100 % of batches at every logged epoch;
  no non-finite loss or gradient; no spike above 1e4.
- Validation tracked the recorded seed-0 H2 run closely and non-monotonically —
  epoch 19 7.970 px / 53.87 %, 39 6.493 / 42.72, 59 5.389 / 40.66, 79 4.842 /
  33.74, 99 5.018 / 31.34, 119 4.663 / 31.21, 139 4.189 / 25.51 — with the sign
  of the difference against the recorded run changing four times.

`INFERRED` — nothing above suggests the arm was failing; it was on a normal
trajectory and had cleared both enforced gates.

## Why this is not resumed

`MEASURED` — weight snapshots exist at epochs 10, 20, 50 and 100
(`phase2/results/training/EXP-E3B-REFINEMENT-CAPACITY-001-ARM-A_epoch*.pth`),
but the harness saves **only model weights** — not optimizer moments, not the
scheduler position, not the data-loader RNG state. Resuming from a snapshot
would therefore be a *different* run under a *different* effective recipe, which
the pre-registration forbids (§9, "no hyperparameter... or schedule may be
altered once the run starts"). The rerun restarts from epoch 0.

## What replaces it

`EXP-E3B-REFINEMENT-CAPACITY-001-ARM-A-RUN2`, and for consistency the other two
arms of the same attempt are `…-ARM-B-RUN2` and `…-ARM-C-RUN2`. The
pre-registration, the preflight (`phase2/results/EXP-E3B-REFINEMENT-CAPACITY-001/preflight.json`)
and every threshold are unchanged; only the record IDs differ.

`UNKNOWN` — whether the host had enough memory throughout the earlier part of
this run. Free memory measured **after** the kill was 5.8 GB of 15.1 GB with no
training process alive, so the pressure was transient and not attributable here.
The three prior 200-epoch runs of this same recipe (H2 seed 0, seeds 1 and 2)
completed on this machine without incident.
