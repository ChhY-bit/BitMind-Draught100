#!/bin/bash
# 并行 AlphaBeta 自博弈（双深度非对称）
# 用法: ./run/selfplay_ab.sh
# 黑白双方使用不同搜索深度，打破镜像僵局，减少超时平局。
#
# 前一半 worker：黑 depth=D, 白 depth=D-1（黑稍强）
# 后一半 worker：黑 depth=D-1, 白 depth=D（白稍强）
# 汇总后消除深度偏差，双方样本量相等。
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

DEEP=$ALPHABETA_DEPTH
SHALLOW=$((DEEP - 1))
HALF=$((NUM_WORKERS / 2))

mkdir -p "$OUTPUT_DIR"

echo "=== AB asymmetric: black=$DEEP/$SHALLOW white=$SHALLOW/$DEEP ==="
echo "    $NUM_WORKERS workers × $GAMES games each"
echo "    output: $OUTPUT_DIR"

# 前一半：黑强白弱
for i in $(seq 0 $((HALF - 1))); do
    OMP_NUM_THREADS=1 PYTHONPATH=interface:. \
    python3 scripts/selfplay.py \
        --black alphabeta --white alphabeta \
        --alphabeta-depth-black "$DEEP" \
        --alphabeta-depth-white "$SHALLOW" \
        --games "$GAMES" \
        --opening-random-plies "$OPENING_RANDOM_PLIES" \
        --output "$OUTPUT_DIR/selfplay_$(printf '%03d' "$i").jsonl" \
        --seed $((BASE_SEED + i)) &
done

# 后一半：白强黑弱
for i in $(seq "$HALF" $((NUM_WORKERS - 1))); do
    OMP_NUM_THREADS=1 PYTHONPATH=interface:. \
    python3 scripts/selfplay.py \
        --black alphabeta --white alphabeta \
        --alphabeta-depth-black "$SHALLOW" \
        --alphabeta-depth-white "$DEEP" \
        --games "$GAMES" \
        --opening-random-plies "$OPENING_RANDOM_PLIES" \
        --output "$OUTPUT_DIR/selfplay_$(printf '%03d' "$i").jsonl" \
        --seed $((BASE_SEED + i)) &
done

echo "All $NUM_WORKERS workers launched."
wait
echo "AB selfplay done."
