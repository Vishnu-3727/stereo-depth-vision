#!/usr/bin/env bash
# EXP-BLOCKCOUNT-FULL-001 -- BATCH 3 (FINAL) driver.
#
# Two authorised runs only: 4 blocks / seed 1 and 4 blocks / seed 2, 200 epochs
# each. The harness refuses everything else; this script cannot widen it.
#
#   bash phase2/factorial/block_count_full/run_batch3.sh <OUT_DIR>
#
# Writes only under OUT_DIR. Batch 1, batch 2, the deterministic series, the
# seed screen and every historical record are read-only.
set -eu

O="${1:?usage: run_batch3.sh <out dir>}"
S=phase2/factorial/block_count_full/exp_blockcount_batch3.py
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

echo "=== batch 3 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

for SEED in 1 2; do
  echo "--- train arm C seed ${SEED} ($(date -u +%H:%M:%SZ)) ---"
  python "$S" train --out "$O" --arm C --seed "$SEED" \
      2>&1 | tee "$O/train_C_seed${SEED}_stdout.log"
  echo "--- evaluate40 arm C seed ${SEED} ($(date -u +%H:%M:%SZ)) ---"
  python "$S" evaluate40 --out "$O" --arm C --seed "$SEED" \
      2>&1 | tee "$O/eval40_C_seed${SEED}_stdout.log"
done

echo "--- verify_prefix ($(date -u +%H:%M:%SZ)) ---"
python "$S" verify_prefix --out "$O" 2>&1 | tee "$O/prefix_stdout.log"

echo "--- probes ($(date -u +%H:%M:%SZ)) ---"
python "$S" probes --out "$O" 2>&1 | tee "$O/probes_stdout.log"

echo "--- compare ($(date -u +%H:%M:%SZ)) ---"
python "$S" compare --out "$O" 2>&1 | tee "$O/compare_stdout.log"

echo "=== batch 3 done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
