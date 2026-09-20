# Record-keeping note (additive; the record itself is unmodified)

Like `EXP-H2-SOFTARGMIN-SCALE`, this record's `config.json` has no top-level
`cost_volume_shift` key. The value is recorded in `changed_variable`:

    "cost_volume_shift: 'left' -> 'none'"

so this run used **`cost_volume_shift="none"`**, and its control
`EXP-H2-SOFTARGMIN-SCALE` used `"left"`.

Two further facts a reader needs before comparing numbers:

1. **This is a 10-epoch viability gate, not a converged run.** The cosine
   schedule kept `T_max = 200`, so these are the control's *first ten epochs*
   rather than a compressed 10-epoch schedule.
2. **Checkpoint alignment differs from the control.** This run validates
   *before* each listed epoch, so its "epoch 0" is the untrained
   initialisation. The control validated at the *end* of each epoch, so the
   control's "epoch 0" is after one epoch of training. The aligned pairs are
   H3 epoch 1 vs control epoch 0, and H3 epoch 10 vs control epoch 9 (which the
   control did not record; its epoch 10 is after eleven epochs).

This note is additive. `config.json`, `metrics.json` and `log.txt` are unchanged.
