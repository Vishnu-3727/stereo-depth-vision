#!/usr/bin/env bash
# Sequential driver for EXP-BLOCKCOUNT-SEEDSCREEN-001.
# Six 30-epoch runs, one process each, then the 40-scene evaluation of each.
# Failures are preserved: a non-zero exit is recorded and the screen continues.
set -u
cd "$(dirname "$0")/../../.." || exit 1
OUT="phase2/factorial/block_count_seed_screen/20260909T145659Z"
SCRIPT="phase2/factorial/block_count_seed_screen/exp_blockcount_seed_screen.py"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

for SEED in 1 2; do
  for ARM in A B C; do
    ID="EXP-BLOCKCOUNT-SEEDSCREEN-001-ARM-${ARM}-SEED${SEED}"
    echo "=== TRAIN ${ID} $(date -u +%FT%TZ) ==="
    python "$SCRIPT" train --out "$OUT" --arm "$ARM" --seed "$SEED" \
      > "${OUT}/train_${ARM}_seed${SEED}_stdout.log" 2>&1
    echo "train ${ID} exit=$?"
    tail -3 "${OUT}/train_${ARM}_seed${SEED}_stdout.log"
  done
done

for SEED in 1 2; do
  for ARM in A B C; do
    echo "=== EVAL40 arm ${ARM} seed ${SEED} $(date -u +%FT%TZ) ==="
    python "$SCRIPT" evaluate40 --out "$OUT" --arm "$ARM" --seed "$SEED" \
      > "${OUT}/eval40_${ARM}_seed${SEED}_stdout.log" 2>&1
    echo "eval40 exit=$?"
    tail -2 "${OUT}/eval40_${ARM}_seed${SEED}_stdout.log"
  done
done

echo "=== PROBES $(date -u +%FT%TZ) ==="
python "$SCRIPT" probes --out "$OUT" > "${OUT}/probes_stdout.log" 2>&1
echo "probes exit=$?"
tail -25 "${OUT}/probes_stdout.log"
echo "=== SCREEN COMPLETE $(date -u +%FT%TZ) ==="
