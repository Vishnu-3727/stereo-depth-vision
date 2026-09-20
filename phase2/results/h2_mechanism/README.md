# EXP-H2 mechanism comparison outputs

Evidence for `phase2/docs/EXP_H2_SOFTARGMIN_SCALE_REPORT.md`, comparing
`EXP-H1-WORKING-v2` (the control, listed as WORKING) against
`EXP-H2-SOFTARGMIN-SCALE` (listed as H2). Regenerable; not experiment records.
The H1 investigation's own evidence lives in `../h1_mechanism/` and is not
overwritten by anything here.

Every file is produced by the same probe as the H1 study, pointed at the other
pair of arms:

    python phase2/scripts/h1_mechanism_probe.py <job> \
        --models WORKING H2 --out-dir phase2/results/h2_mechanism

with `<job>` one of: `stages`, `spatial`, `pooled`, `saturation`, `tie_check`,
`feature_matching`, `weight_drift`, `figures`, `cross_shift`, and
`ablation --scenes 0 3 6 9 12 15 18 21 27 31`.

Figures go to `phase2/visualizations/working_vs_h2/` (git-ignored).

## Two probe defects found here and fixed

Both would have produced false statements about H2, so they are recorded rather
than quietly patched:

- `stage_stats` measured softmax saturation on the **raw** aggregated cost. For
  an arm that rescales the cost before the softmax that is the wrong tensor; it
  reported H2 as saturated (entropy 0.0000) when the tensor its softmax actually
  consumes has entropy ~1.7. Fixed by capturing the consumed tensor, and
  `stages.json` now carries `softmax_input_is_standardised` alongside the gap
  statistics.
- `cross_shift` rebuilt each arm's model **without** its regression stage, so the
  "H2 weights" rows were really H2 weights read out through the frozen
  saturating stage. Fixed; that configuration is now reported explicitly as
  `H2_weights_shift_left_no_standardisation`.

Both files were regenerated after the fixes.
