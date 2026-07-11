#!/bin/bash
# 并行 AlphaBeta 自博弈
# 用法: ./run/selfplay_ab.sh
# AB 纯 CPU 搜索，多进程并发榨干多核。
#
# 每个 worker 是独立的 selfplay 进程，各自做 AlphaBeta 搜索。
# 多进程并发 → CPU 利用率提升 → 相同时间产出更多对局数据。
set -e

cd "$(dirname "$0")/.."

cleanup() {
    echo ""
    echo "Caught signal — stopping all workers..."
    jobs -p | xargs -r kill -INT 2>/dev/null
    for _i in $(seq 1 5); do
        if ! jobs -p | grep -q . 2>/dev/null; then
            break
        fi
        sleep 1
    done
    jobs -p | xargs -r kill -9 2>/dev/null
    wait 2>/dev/null
    echo "All workers stopped."
}
trap cleanup SIGINT SIGTERM

ALPHABETA_DEPTH="${ALPHABETA_DEPTH:-6}"
OUTPUT_DIR="${OUTPUT_DIR:-data/selfplay_ab}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BASE_SEED="${BASE_SEED:-7021}"
GAMES="${GAMES:-500}"
OPENING_RANDOM_PLIES="${OPENING_RANDOM_PLIES:-8}"

mkdir -p "$OUTPUT_DIR"

echo "=== AB depth=$ALPHABETA_DEPTH: $NUM_WORKERS workers × $GAMES games each ==="
echo "    output:     $OUTPUT_DIR"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    OMP_NUM_THREADS=1 PYTHONPATH=interface:. \
    python3 scripts/selfplay.py \
        --black alphabeta --white alphabeta \
        --alphabeta-depth "$ALPHABETA_DEPTH" \
        --games "$GAMES" \
        --opening-random-plies "$OPENING_RANDOM_PLIES" \
        --output "$OUTPUT_DIR/selfplay_$(printf '%03d' "$i").jsonl" \
        --seed $((BASE_SEED + i)) &
done

echo "All $NUM_WORKERS workers launched."
wait
echo "AB selfplay done."
