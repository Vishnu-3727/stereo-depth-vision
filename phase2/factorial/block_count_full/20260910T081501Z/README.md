# EXP-BLOCKCOUNT-FULL-001 — batch 3 (final)

Two runs: **4 blocks / seed 1** and **4 blocks / seed 2**, 200 epochs each,
deterministic protocol. They bring the 4-block arm to three seeds against the
6-block and 5-block arms' three each, on the authoritative 40-scene protocol.

| | |
| --- | --- |
| Date | 2026-09-10, 08:17–10:56 UTC |
| Runs | 2 — nothing else was authorised or reachable |
| Cost | 4678.8 s + 4735.2 s = **2.61 GPU-hours** |
| Results | 4b/s1 EPE **2.6553489**, D1 20.0897392 · 4b/s2 EPE **2.5756633**, D1 20.0793521 |
| Verdict | **`4-BLOCK-INCONCLUSIVE-AT-SEEDS-0-1-2`** (see `RESULTS.md`) |
| Next | none proposed — a new *design*, not more seeds, would be needed |

Both runs: 200/200 epochs, exit 0, no abort, no NaN/Inf, epoch-10 viability gate
PASS, epoch-100 collapse check PASS, epoch-200 STEREO-FUNCTIONAL, matching-path
gradient in 100 % of batches, and an **exact** reproduction of their 30-epoch
seed-screen prefixes.

## The completed three-arm picture

| Blocks | seed 0 | seed 1 | seed 2 | mean | range |
| -----: | -----: | -----: | -----: | ---: | ----: |
| 6 — EPE | 2.2955182 | 2.3982217 | 2.3950488 | 2.3629296 | **0.1027** |
| 5 — EPE | 2.6457595 | 2.4729830 | 2.6372994 | 2.5853473 | 0.1728 |
| 4 — EPE | 2.3387143 | **2.6553489** | **2.5756633** | 2.5232422 | **0.3166** |
| 6 — D1 | 15.6045143 | 18.3357145 | 17.4699044 | 17.1367110 | 2.7312 |
| 5 — D1 | 21.3711907 | 18.2061782 | 20.6593989 | 20.0789226 | 3.1650 |
| 4 — D1 | 16.9797389 | **20.0897392** | **20.0793521** | 19.0496101 | 3.1100 |

**The 4-block arm straddles the frozen 6-block boundary**
`[2.2955182, 2.3982217]`: seed 0 inside, seeds 1 and 2 above. Under the
pre-registered rule that is Verdict C — INCONCLUSIVE. No fourth seed was bought
and no threshold was reinterpreted.

The finding that matters: **4b/seed0 (2.3387) was a favourable draw, not the
arm's centre** — exactly the role 5b/seed1 played for the 5-block arm. The
4-block arm's own seed spread (0.3166 px) is 3.1× the 6-block arm's and larger
than the 4b-vs-6b mean gap (0.1603 px) it would have to demonstrate. That is why
the question is not resolvable by adding seeds to this design.

The 5-block verdict is **unchanged**: `5-BLOCK-PENALTY-AT-SEEDS-0-1-2` stands as
recorded in batch 2, D1 exception included.

## Two disclosed corrections

Both were defects in *this batch's own analysis code*, found and fixed before the
result was written up. Neither touched training, thresholds or the decision rule.

| File | What |
| --- | --- |
| `CORRECTION_screen_prefix.md` | frozen prefix constants had mis-transcribed trailing digits; corrected against the authoritative screen records. Seed 1's prefix check is therefore confirmatory, seed 2's blind |
| `CORRECTION_stereo_reader.md` | the stereo-verdict lookup read the wrong nesting level and reported *every* run — including a verified frozen one — as a stereo failure. Fixed; superseded comparison preserved |

## Files

| File | What |
| --- | --- |
| `PREREGISTRATION.md` | frozen before preflight — authorisation, guards, decision boundary, verdicts, claim ceiling |
| `RESULTS.md` | the analysis, MEASURED/DERIVED/INFERRED/UNKNOWN, verdict |
| `ENVIRONMENT.txt` | platform, versions, determinism controls, weight hashes, provenance |
| `CONFIG.json` | machine-readable configuration and result |
| `preflight.json` | 28/28 blocking checks (also in `results/`) |
| `guard_test.log` | the guard refusing five forbidden runs, before launch |
| `prefix_verification.json` | exact-equality check against the 30-epoch seed screen |
| `results/history_arm_C-SEED{1,2}.json` | per-epoch history, checkpoints, ablations, late window |
| `run_meta_*.json` | identity, hashes, wall clock, provenance |
| `train_C_seed{1,2}_stdout.log` | full training logs |
| `eval40_*.json` | 40-scene evaluation at epoch 200, pooled and per-scene |
| `probes.json` | stereo probes at epochs 10/20/50/100/150/200 |
| `comparison.json` | three-arm ranges, boundary test, all pairings, paired per-scene, verdict |
| `comparison_SUPERSEDED_stereo_reader_bug.json` | the defective first comparison, preserved |
| `checkpoints/` | final checkpoints + snapshots at 10/20/50/100/150/200 |

## Reproducing

```bash
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
S=phase2/factorial/block_count_full/exp_blockcount_batch3.py
O=phase2/factorial/block_count_full/20260910T081501Z
python $S preflight     --out $O
python $S train         --out $O --arm C --seed 1
python $S evaluate40    --out $O --arm C --seed 1
python $S train         --out $O --arm C --seed 2
python $S evaluate40    --out $O --arm C --seed 2
python $S verify_prefix --out $O
python $S probes        --out $O
python $S compare       --out $O
```

`run_batch3.sh` drives the whole batch. The harness authorises only arm C seeds
1 and 2 and refuses every other configuration, including 4b/seed0, 4b/seed3, and
any 5-block or 6-block run.

Batch 1 (`20260910T005550Z/`), batch 2 (`20260910T052736Z/`), the deterministic
series, the seed screen and every historical record were read, never written.
