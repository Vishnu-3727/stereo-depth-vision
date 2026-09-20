# H1 mechanism probe outputs

Evidence for `phase2/docs/H1_VISUAL_MECHANISM_INVESTIGATION_V1.md`. Every file
here is regenerable; none of them is an experiment record, and none replaces the
registered H1-v2 metrics under `phase2/experiments/`.

| file | regenerate with |
|---|---|
| `stages.json` | `python phase2/scripts/h1_mechanism_probe.py stages` |
| `spatial.json` | `... spatial` |
| `ablation.json` | `... ablation --scenes 0 3 6 9 12 15 18 21 27 31` |
| `intervene.json` | `... intervene` |
| `pooled.json` | `... pooled` |
| `cross_shift.json` | `... cross_shift` |
| `gradients.json` | `... gradients` |
| `saturation.json` | `... saturation` |
| `tie_check.json` | `... tie_check` |
| `weight_drift.json` | `... weight_drift` |
| `feature_matching.json` | `... feature_matching` |
| `figures.json` | `... figures` (writes PNGs under `phase2/visualizations/h1_mechanism/`) |

## Superseded scratch files, kept rather than deleted

These three were produced by throwaway snippets during the investigation, before
the corresponding probe jobs existed. They are **not reproducible from the
repository** and must not be cited; the replacements are equivalent and were
verified to give the same numbers.

| scratch file | superseded by | note |
|---|---|---|
| `pooled_intervention.json` | `pooled.json` | identical values, now reproducible |
| `gradient_saturation.json` | `saturation.json` | the scratch version measured softmax saturation on the 1/16-resolution aggregated cost; the soft-argmin actually consumes the **upsampled** cost, and only the upsampled measurement reveals the tied pixels. The scratch numbers are therefore the wrong tensor, and the corrected measurement is in `saturation.json` |
| `gradient_bands.json` | `saturation.json` (`column_bands`) | same crop-position test, folded into the probe |
