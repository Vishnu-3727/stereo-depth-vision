# NOTE — additive, written after the run; nothing in this record was edited

This record is **VOID as evidence about E1's verdict** and is superseded by
`EXP-E1-ERROR-PARTITION-002-INVERSE-ORACLE-RUN2`. It is kept unchanged because
records are never overwritten, and because the way it failed is worth keeping.

## The defect

The fixed-point update was written as

    initial <- clamp(initial + (gt - final), 0, 11)

which adds a **pixel-valued** residual to a **candidate-valued** input. Run 001
measured the model's own convention: the slope is `a` ≈ 0.063–0.095 candidates
per pixel, i.e. one candidate is roughly 10–16 px. The update therefore
overshoots by about an order of magnitude, and the recorded trace shows exactly
that — EPE rises monotonically from 2.400 to 22.482 (H2), 2.319 to 29.153
(SEED1), 2.556 to 23.156 (SEED2), with D1 climbing to 94–97 %. The iteration
diverges; it never searched anything.

## Why this record is dangerous, not merely useless

Because every iterate was worse than iteration 0, the best iterate *was*
iteration 0, the EPE ratio was exactly 1.000, and the frozen rule dutifully
returned **DOWNSTREAM-LIMITED** — the same verdict
`EXP-E1-ERROR-PARTITION-001` had reached on real evidence. A diverging probe
produced an apparent confirmation. **This record must never be cited as
corroborating E1.** The missing safeguard was a gate asserting the search
actually improves before its output is read as a verdict; RUN2 adds one.

## What in this record is still valid

The affine-oracle range diagnostic, which does not depend on the iteration:
run 001's substituted values fell outside the soft-argmin's own [0, 11] range
for **0.53 % (H2), 0.00 % (SEED1), 0.0002 % (SEED2)** of pixels, with maxima
15.58 / 10.79 / 12.78. That disposes of one of the two stated weaknesses of the
affine oracle: off-manifold inputs are negligible. The other weakness — the
`R^2` 0.86–0.89 fit residual — is what RUN2 exists to test.

`phase2/scripts/exp_e1_inverse_oracle.py` was corrected in place after this run.
This record's `config.method` states the rule that actually produced these
numbers, so the record remains self-describing.
