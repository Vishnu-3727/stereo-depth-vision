# EXP-BLOCKCOUNT-FULL-001 — batch 1

The 200-epoch block-count experiment, run **sequentially with a hard review
gate** after the seed screen returned `CONTINUE-FULL`.

| | |
| --- | --- |
| Date | 2026-09-10, 01:01–05:13 UTC |
| Runs | 3 of a possible 6 — batch 1 only |
| Cost | 14 865.5 s training = **4.13 GPU-hours** |
| Verdict | **CONTINUE-BATCH-2** (see `RESULTS.md`) |
| Next | 5 blocks / seed 2 / 200 epochs — **one run, not executed** |

## What was run

| Arm | Blocks | Seed | Epochs | Result |
| --- | -----: | ---: | -----: | ------ |
| A | 6 | 1 | 200 | 40-scene EPE 2.3982, D1 18.336 |
| A | 6 | 2 | 200 | 40-scene EPE 2.3950, D1 17.470 |
| B | 5 | 1 | 200 | 40-scene EPE 2.4730, D1 18.206 |

All three: 200/200 epochs, exit 0, no abort, no NaN/Inf, epoch-10 viability gate
PASS, epoch-100 collapse check PASS, epoch-200 verdict STEREO-FUNCTIONAL,
matching-path gradient in 100 % of batches.

**Not run, refused by the harness:** 5b/seed2, 4b/seed1, 4b/seed2 — batch 2
needs the research lead's review. Seed 0 is frozen and is never retrained.

## The headline measurement

The 200-epoch seed spread at fixed 6-block architecture, across seeds 0/1/2 on
the 40-scene protocol, is **0.103 px EPE**. The 30-epoch screen had implied a
multi-pixel seed effect; at 200 epochs it is small, and roughly the size of the
within-run epoch-to-epoch wobble.

The 5-block arm sits 0.075 px outside the 6-block range on EPE — *less* than the
range width — and inside it on D1. Both available 5-block draws (seeds 0 and 1)
nevertheless exceed every 6-block draw. One more 5-block seed decides it.

## Files

| File | What |
| --- | --- |
| `PREREGISTRATION.md` | frozen before the first run — scope, guards, decision rule, claim ceiling |
| `RESULTS.md` | the analysis and the verdict |
| `ENVIRONMENT.txt` | platform, versions, determinism controls, weight hashes, provenance |
| `CONFIG.json` | machine-readable configuration of all three runs |
| `results/preflight.json` | 20/20 blocking checks |
| `results/history_arm_*.json` | per-epoch history, checkpoints, stereo ablations, late window |
| `run_meta_*.json` | per-run identity, hashes, wall clock, provenance |
| `train_*_stdout.log` | full training logs |
| `eval40_*.json` | 40-scene evaluation at epoch 200, pooled and per-scene |
| `probes.json` | stereo probes at epochs 10/20/50/100/150/200 |
| `comparison.json` | seed spread, 5b-vs-6b position, interpretation limit |
| `checkpoints/` | final checkpoint + snapshots at 10/20/50/100/150/200 per run |

## Reproducing

```bash
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
S=phase2/factorial/block_count_full/exp_blockcount_full.py
O=phase2/factorial/block_count_full/20260910T005550Z
python $S preflight  --out $O
python $S train      --out $O --arm A --seed 1     # and A/2, B/1
python $S evaluate40 --out $O --arm A --seed 1
python $S probes     --out $O
python $S compare    --out $O
```

`run_batch1.sh` drives the whole batch in the pre-registered order.

Every run in this batch reproduced the seed screen's 30-epoch prefix exactly, so
the deterministic protocol holds across processes — and the screen's 6b/1, 6b/2
and 5b/1 runs are the *same* trajectories as these, not independent samples.
