# NOT AUTHORITATIVE — first Stage B attempt, killed by a wrapper defect

**Status: preserved, never cited as the Stage B result.** Nothing in this
directory has been edited after the fact.

## What happened

`MEASURED` — the 200-epoch deterministic training run **completed**:
`epochs_completed: 200`, `aborted: null`, all nine checkpoints and all seven
weight snapshots written, a full 200-row history recorded, and the epoch-200
stereo verdict computed and `STEREO_FUNCTIONAL`.

It then raised on the **last bookkeeping line** of
`exp_e3b_refinement_capacity.run_training`:

```
File "phase2/scripts/exp_e3b_refinement_capacity.py", line 384, in run_training
    exp.metric("checkpoint", str(ckpt.relative_to(REPO_ROOT)))
ValueError: 'phase2\\diagnostics\\determinism\\20260909T024500Z_baseline\\checkpoints\\...pth'
           is not in the subpath of 'C:\\Users\\vishn\\stereo_depth_vision'
```

`DERIVED` — **the defect is in the Stage B wrapper, not in the science.**
`stageb_deterministic_baseline.py` repointed `e3b.OUT_DIR` to a *relative* path
(the `--out` argument was passed relative), and `run_training` records the
checkpoint as `ckpt.relative_to(REPO_ROOT)`, which requires an absolute path.
The exception fired **after** training, validation, the stereo probes and every
metric had been computed, so the record's `status` is `failed` and
`metrics.checkpoint` / `checkpoint_sha256` are absent, while the measurements
themselves are complete.

`MEASURED` — nothing scientific was affected: the model, seed, recipe,
determinism controls, dataset, crop policy, validation protocol and epoch budget
were the intended ones, and the epoch-0 result reproduced Stage A's controlled
runs exactly (mean loss `10.532406491041183`, val EPE `15.32539188316842`,
D1 `87.44677261821234` — identical to
`20260909T022238Z/run_C_deterministic` and `run_D_deterministic`, which were
produced by a different script).

## What was done about it

1. This directory is **preserved unmodified** and is **not** the Stage B result.
2. The wrapper was fixed — `Path(args.out).resolve()`, an output-path change
   only. **No model, seed, recipe, control, dataset or protocol change.**
3. Stage B was re-run from scratch under a **new** directory with new record
   IDs. Records are never overwritten.

## What this record is still good for

`DERIVED` — it is an additional same-config 200-epoch sample under the
deterministic protocol, produced by a slightly different wrapper build. Its
recorded history can be compared against the authoritative Stage B runs as a
third reproducibility datum. That comparison is reported in the authoritative
Stage B `RESULTS.md`; it is **not** an architecture comparison and no verdict
rests on this directory.

## Recorded figures (for cross-reference only, never as the Stage B result)

| quantity | value |
|---|---|
| epochs completed | 200 |
| aborted | `null` |
| wall clock | 4,945.07 s |
| late window (10 pts) | EPE 3.8670 ± 0.1113, D1 23.4954 ± 0.9342 |
| epoch 200 | EPE 3.834674 px, D1 23.047094 % |
| gradient norms | median 21.916, max 308.034, no NaN, 0 batches > 1e4 |
| matching gradient | 100 % of 16,000 batches |
| stereo verdict | STEREO_FUNCTIONAL — right-image +80.813, matching-map +66.188, entropy 1.7755, disparity std 18.000 |
