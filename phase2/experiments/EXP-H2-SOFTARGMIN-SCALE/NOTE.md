# Record-keeping note (additive; the record itself is unmodified)

`config.json` for this experiment does **not** carry a top-level
`cost_volume_shift` key, unlike the four H1 records. The value is recorded, but
inside two other fields:

- `changed_variable` — names the regression stage as the only change;
- `held_fixed` — "cost_volume_shift='left', feature extractor, cost volume,
  aggregation, refinement, loss, optimizer, schedule, seed, data split, crop,
  augmentation, epochs, batch size, precision".

So this run used **`cost_volume_shift="left"`**, identical to its control
`EXP-H1-WORKING-v2`.

Consequence, found when tooling tried to read the record: any loader that
requires a top-level `cost_volume_shift` will raise on this checkpoint.
`phase2/scripts/h1_mechanism_probe.py::load_model` and
`phase2/viz/core.ModelRunner` both treat an absent key as "not recorded" rather
than as a mismatch. A future H2-family script should write the key explicitly.

This note is additive. The recorded `config.json`, `metrics.json` and `log.txt`
are unchanged.
