# NOTE — additive, written after the run; nothing in this record was edited

This record is **superseded by `EXP-H2-EMERGENCE-001-RUN2`**, which re-ran the
identical measurement with one reader bug fixed. It is kept unchanged because
experiment records are never overwritten in this project.

## What is wrong with this record

`exp_h2_emergence.recorded_metrics()` read the source runs' `metrics.json` at
its **top level**, but `src.common.experiment.Experiment` nests everything it
recorded under a `"metrics"` key. `dict.get("checkpoints", {})` and
`dict.get("stereo_ablations", {})` therefore both returned `{}` silently, with
two consequences:

1. `validation_from_record` is empty, so the `val EPE` / `val D1 %` columns of
   `emergence_table.md` read `n/a` for every row. Those numbers exist and are
   correct in the source runs' records — they were simply not joined in.
2. `cross_check_vs_recorded_ablation` is an empty list, so the planned
   comparison of the re-measured epoch-10/100/200 ablation against the
   training-time ablation was not performed.

The fix is in `phase2/scripts/exp_h2_emergence.py`: read `blob["metrics"]`, and
assert that both keys are present so the same failure cannot be silent again.

## What is NOT wrong with this record

The ablation and gradient measurements themselves are unaffected — they are
computed from the weight snapshots by imported, unchanged probes and never
touch the source runs' `metrics.json`. Compared field by field against `-RUN2`
over all twelve snapshots (same SHA-256 in both runs), the right-image
dependence, matching-map dependence, softmax entropy, matching-gradient
fraction and probe-scene EPE/D1 are **bit-identical**; only the gradient norms
differ, by at most 3.7e-5 (fp32 non-determinism in the backward pass). The
emergence verdicts — seed 1 in (10, 20], seed 2 in (20, 50] — are the same in
both runs.

Read `-RUN2` for the complete table; read this record only as the audit trail.
