# Decision (additive; the record itself is unmodified)

    SEED 1: CONTINUE -> completed 200 epochs -> STEREO-FUNCTIONAL

Judged against the criterion frozen before any replication result existed
(`phase2/docs/EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md` section 4.2):
right-image dependence and matching-map dependence both >= +20.0 D1 points,
matching-path gradient on >= 95 % of batches, softmax entropy > 0.5 nats, no
NaN/Inf, final disparity standard deviation > 1.0 px.

Full analysis: `phase2/docs/EXP_H2_SEED_REPLICATION_001_REPORT.md`.

Supersedes `EXP-H2-SEED-REPLICATION-001-SEED1`, which was aborted at epoch 10
by a gate later shown to be invalid (seed 0 fails it too). That record is kept
untouched.
