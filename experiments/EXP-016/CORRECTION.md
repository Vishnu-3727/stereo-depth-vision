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
