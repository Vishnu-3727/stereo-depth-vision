# Note (additive; this record is historical and unmodified)

This run was **aborted at epoch 10 by a gate that was later shown to be
invalid**. Nothing in `config.json`, `metrics.json` or `log.txt` has been
edited, and the run is kept as evidence of what happened.

The gate required a right-image D1 penalty of at least +5.0 points at epoch 10.
That threshold was never calibrated: seed 0's +80.8...+84.4 dependence was
measured at **200** epochs, and no 10-epoch measurement existed when this gate
was written.

The missing measurement was taken afterwards as
`EXP-H2-SEED-REPLICATION-001-SEED0-REFERENCE`: **seed 0 also fails this gate at
epoch 10** (right-image penalty -3.33 D1 points). Stereo dependence is a
late-emerging property of this recipe, so the abort says nothing about whether
this seed replicates H2.

Superseded by `EXP-H2-SEED-REPLICATION-001-SEED2-RUN2`, run under the corrected
protocol frozen in
`phase2/docs/EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md`.
