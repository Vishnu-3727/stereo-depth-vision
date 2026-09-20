# Note (additive; this record is unmodified)

**This is an infrastructure/control measurement, not seed 0's result.**

It exists to answer one question: does the H2 recipe show right-image dependence
at epoch 10? It does not — this run reaches epoch 10 with a right-image D1
penalty of -3.33 points, which is why the epoch-10 stereo gate that aborted
`EXP-H2-SEED-REPLICATION-001-SEED1` and `-SEED2` is recorded as invalid.

It was stopped at 10 epochs by design (`--stop-after 10`); its epoch-10 gate
verdict is recorded but was not enforced. Its 200-epoch behaviour is UNKNOWN.

**Seed 0's result is `EXP-H2-SOFTARGMIN-SCALE`** (200 epochs), which remains the
primary reference and was not retrained.

This run also does not reproduce that reference bit-exactly (epoch-0 loss
10.7375 vs 10.6352). That is fp32 GPU non-determinism, not a harness difference:
the harness does not reproduce *itself* either (10.8798 vs 10.5762 on two
consecutive 2-epoch runs). See
`phase2/docs/EXP_H2_SEED_REPLICATION_001_PROTOCOL_CORRECTION.md` §3.
