# EXP-BLOCKCOUNT-FULL-001 — batch 2

One resolution run: **5 blocks / seed 2 / 200 epochs**, authorised after batch 1
returned `CONTINUE-BATCH-2`. It brings the 5-block arm to three seeds against the
6-block arm's three, on the authoritative 40-scene protocol.

| | |
| --- | --- |
| Date | 2026-09-10, 05:30–06:51 UTC |
| Runs | 1 — nothing else was authorised or reachable |
| Cost | 4777.6 s = **1.33 GPU-hours** |
| Result | 40-scene EPE **2.6373**, D1 **20.6594**, RMSE 6.4439 |
| Verdict | **`5-BLOCK-PENALTY-AT-SEEDS-0-1-2`** (see `RESULTS.md`) |
| Next | 4 blocks / seeds 1 and 2 / 200 epochs — two runs, **not executed** |

Run status: 200/200 epochs, exit 0, no abort, no NaN/Inf, epoch-10 viability gate
PASS, epoch-100 collapse check PASS, epoch-200 verdict STEREO-FUNCTIONAL
(right-image penalty +73.6), matching-path gradient in 100 % of batches.

## The three-seed picture

| Blocks | seed 0 | seed 1 | seed 2 | range |
| -----: | -----: | -----: | -----: | ----: |
| 6 — EPE | 2.2955 | 2.3982 | 2.3950 | 0.1027 |
| 5 — EPE | 2.6458 | 2.4730 | **2.6373** | 0.1728 |
| 6 — D1 | 15.605 | 18.336 | 17.470 | 2.731 |
| 5 — D1 | 21.371 | 18.206 | **20.659** | 3.165 |

**EPE ranges do not overlap** — `min(5b) = 2.4730 > max(6b) = 2.3982`, margin
0.0748 px, and all nine 5b×6b seed pairings run the same way.

**D1 ranges do overlap**, by 0.13 pt: 5b/seed1 (18.206) beats 6b/seed1 (18.336).
Eight of nine pairings, not nine. The verdict is quoted with that exception.

Per-scene, the new run is worse in 31–37 of 40 scenes against each 6-block seed.
5b/seed1 — worse in only 22 of 40 against 6b/seed1 — turns out to have been the
5-block arm's favourable draw, which is exactly the uncertainty this run was
commissioned to remove.

4 blocks still has **one** seed (2.3387), sitting *inside* the 6-block range. It
was deliberately not retrained here.

## Files

| File | What |
| --- | --- |
| `PREREGISTRATION.md` | frozen before the run — authorisation, guards, decision rule, claim ceiling |
| `RESULTS.md` | the analysis, MEASURED/DERIVED/INFERRED/UNKNOWN, verdict |
| `ENVIRONMENT.txt` | platform, versions, determinism controls, weight hashes, provenance |
| `CONFIG.json` | machine-readable configuration and result |
| `preflight.json` | 25/25 blocking checks (also in `results/`) |
| `results/history_arm_B-SEED2.json` | per-epoch history, checkpoints, ablations, late window |
| `run_meta_*.json` | identity, hashes, wall clock, provenance |
| `train_B_seed2_stdout.log` | full training log |
| `eval40_*.json` | 40-scene evaluation at epoch 200, pooled and per-scene |
| `probes.json` | stereo probes at epochs 10/20/50/100/150/200 |
| `comparison.json` | three-seed ranges, non-overlap tests, all nine pairings, paired per-scene |
| `checkpoints/` | final checkpoint + snapshots at 10/20/50/100/150/200 |

## Reproducing

```bash
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
S=phase2/factorial/block_count_full/exp_blockcount_batch2.py
O=phase2/factorial/block_count_full/20260910T052736Z
python $S preflight  --out $O
python $S train      --out $O
python $S evaluate40 --out $O
python $S probes     --out $O
python $S compare    --out $O
```

`run_batch2.sh` drives the whole batch. The harness authorises only arm B seed 2
and refuses every other configuration, including 4-block, 6-block and seed 0.

Batch 1's script and record directory (`20260910T005550Z/`) were read, never
written.
