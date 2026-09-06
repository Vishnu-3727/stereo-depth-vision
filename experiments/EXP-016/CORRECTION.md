# Correction to the EXP-016 conclusion

The conclusion recorded in `metrics.json` states that "validation improves". That
is not supported by the run's own numbers and is withdrawn.

Validation EPE across the run:

| Epoch | Validation EPE (px) | Validation D1 (%) |
|---:|---:|---:|
| 0 | 18.497 | 88.66 |
| 5 | 22.564 | 92.47 |
| 10 | 18.827 | 90.73 |
| 15 | 17.622 | 90.08 |
| 19 | 19.216 | 91.12 |

The series is noisy and ends slightly *worse* than epoch 15 and only marginally
better than epoch 0. **Validation did not improve meaningfully over this run.**

What the run does establish, and what the experiment was for:

- **Training loss decreased**, 11.494 to 7.614 over 20 epochs, a factor of 1.51,
  with the second half averaging lower than the first.
- **Gradients are finite and bounded** — median norm 38.4, maximum 814.2, no
  non-finite values at any step.
- The data pipeline, loss masking, optimiser, schedule and checkpointing all run
  end to end and produce a usable checkpoint.

That is a working training pipeline, which is what the Phase 1 charter asked for.
It is not evidence that the model learns to generalise, and 160 scenes from
random initialisation without batch normalisation would not be expected to show
that in 20 epochs.

The flat validation curve is itself worth recording as an observation for
Phase 2: with the reference weights the same model reaches EPE 1.313 px
(EXP-005), so the gap is about data budget and training schedule, not about the
implementation, which EXP-011 verified against the reference to a relative 1e-7.

The metrics in `metrics.json` are unaltered. Only the interpretation is corrected
here.

## The defect that produced it, and its fix

The withdrawn sentence was a **string literal** in
`scripts/exp_train_convergence.py`. It did not depend on the run at all, so
re-running the experiment would have recorded "validation improves" again
regardless of the numbers. Two neighbouring claims in the same block had the
same defect: "the loss decreases" and "no non-finite values" were also asserted
rather than measured.

The corrective audit fixed this without touching this experiment's record:

- `src/common/conclusions.py` derives every statement from the measured values.
  `classify_validation_change(initial, final)` returns `improved`,
  `did not improve` or `unchanged` by comparing the first and last recorded
  validation value — the same criterion this document applies above — and
  `describe_training_outcome(...)` assembles the conclusion from the loss
  series, the gradient norms and that classification.
- `scripts/exp_train_convergence.py` now calls those functions instead of
  writing a conclusion out by hand, and records
  `validation_change`, `validation_epe_initial`, `validation_epe_final`,
  `validation_epe_best` and `non_finite_gradient_count` as metrics so the
  verdict is auditable from the record alone.
- `tests/test_conclusions.py` pins the behaviour, including this experiment's
  exact numbers (18.497 → 19.216 must classify as `did not improve`) and an AST
  check that the script contains no hard-coded outcome claim in any string
  literal. That guard was verified against the pre-fix file recovered from tag
  `phase-1`: it flags all three literals there and passes on the fixed version.

So the trail is complete: the original record stands, this document withdraws
its interpretation, and the script can no longer reproduce the error.
