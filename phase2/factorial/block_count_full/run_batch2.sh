#!/usr/bin/env bash
# Driver for EXP-BLOCKCOUNT-FULL-001 BATCH 2 -- ONE run: 5 blocks, seed 2, 200 epochs.
# Then the 40-scene evaluation, the stereo probes and the three-seed comparison.
# Nothing else is authorised; the harness refuses every other configuration.
set -u
cd "$(dirname "$0")/../../.." || exit 1
OUT="phase2/factorial/block_count_full/20260910T052736Z"
SCRIPT="phase2/factorial/block_count_full/exp_blockcount_batch2.py"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

echo "=== TRAIN EXP-BLOCKCOUNT-FULL-001-ARM-B-SEED2 (5 blocks, seed 2) $(date -u +%FT%TZ) ==="
python "$SCRIPT" train --out "$OUT" > "${OUT}/train_B_seed2_stdout.log" 2>&1
echo "train exit=$?"
tail -4 "${OUT}/train_B_seed2_stdout.log"

echo "=== EVAL40 $(date -u +%FT%TZ) ==="
python "$SCRIPT" evaluate40 --out "$OUT" > "${OUT}/eval40_B_seed2_stdout.log" 2>&1
echo "eval40 exit=$?"
tail -2 "${OUT}/eval40_B_seed2_stdout.log"

echo "=== PROBES $(date -u +%FT%TZ) ==="
python "$SCRIPT" probes --out "$OUT" > "${OUT}/probes_stdout.log" 2>&1
echo "probes exit=$?"
tail -10 "${OUT}/probes_stdout.log"

echo "=== COMPARE $(date -u +%FT%TZ) ==="
python "$SCRIPT" compare --out "$OUT" > "${OUT}/compare_stdout.log" 2>&1
echo "compare exit=$?"
tail -60 "${OUT}/compare_stdout.log"

echo "=== BATCH 2 COMPLETE $(date -u +%FT%TZ) -- HARD STOP, no further runs ==="
