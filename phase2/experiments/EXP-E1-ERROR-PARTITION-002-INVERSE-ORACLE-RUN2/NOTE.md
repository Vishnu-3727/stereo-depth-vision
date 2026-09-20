# NOTE — additive; this record holds no measurement

The run crashed with `NameError: name 'best_run' is not defined` immediately
after the experiment directory was created and before any model was scored. The
`metrics.json` here contains only the affine-oracle range diagnostic, which is
identical to the one already recorded in
`EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE`.

The cause was a botched in-place edit of `phase2/scripts/exp_e1_inverse_oracle.py`
(a helper the new code called was never inserted), not anything about the
experiment. The corrected run is `EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE-RUN3`.

This record is kept only because records are never deleted. It is not evidence
about anything.
