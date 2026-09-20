#!/usr/bin/env bash
# Sequential driver for EXP-BLOCKCOUNT-FULL-001 BATCH 1.
# Three 200-epoch runs, one process each, in the lead's order:
#   6b/seed1, 6b/seed2, 5b/seed1
# then the 40-scene evaluation of each, the stereo probes, and the comparison.
# Failures are preserved: a non-zero exit is recorded and the batch continues.
# Batch 2 is NOT run here and is refused by the harness itself.
set -u
cd "$(dirname "$0")/../../.." || exit 1
OUT="phase2/factorial/block_count_full/20260910T005550Z"
SCRIPT="phase2/factorial/block_count_full/exp_blockcount_full.py"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

RUNS="A:1 A:2 B:1"

for PAIR in $RUNS; do
  ARM="${PAIR%%:*}"; SEED="${PAIR##*:}"
  ID="EXP-BLOCKCOUNT-FULL-001-ARM-${ARM}-SEED${SEED}"
  echo "=== TRAIN ${ID} $(date -u +%FT%TZ) ==="
  python "$SCRIPT" train --out "$OUT" --arm "$ARM" --seed "$SEED" \
    > "${OUT}/train_${ARM}_seed${SEED}_stdout.log" 2>&1
  echo "train ${ID} exit=$?"
  tail -4 "${OUT}/train_${ARM}_seed${SEED}_stdout.log"
done

for PAIR in $RUNS; do
  ARM="${PAIR%%:*}"; SEED="${PAIR##*:}"
  echo "=== EVAL40 arm ${ARM} seed ${SEED} $(date -u +%FT%TZ) ==="
  python "$SCRIPT" evaluate40 --out "$OUT" --arm "$ARM" --seed "$SEED" \
    > "${OUT}/eval40_${ARM}_seed${SEED}_stdout.log" 2>&1
  echo "eval40 exit=$?"
  tail -2 "${OUT}/eval40_${ARM}_seed${SEED}_stdout.log"
done

echo "=== PROBES $(date -u +%FT%TZ) ==="
python "$SCRIPT" probes --out "$OUT" > "${OUT}/probes_stdout.log" 2>&1
echo "probes exit=$?"
tail -25 "${OUT}/probes_stdout.log"

echo "=== COMPARE $(date -u +%FT%TZ) ==="
python "$SCRIPT" compare --out "$OUT" > "${OUT}/compare_stdout.log" 2>&1
echo "compare exit=$?"
tail -40 "${OUT}/compare_stdout.log"

echo "=== BATCH 1 COMPLETE $(date -u +%FT%TZ) -- HARD STOP, batch 2 not run ==="
