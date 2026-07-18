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

# 显式追踪所有 worker PID，Ctrl+C 时逐个 kill，不依赖 bash 作业表
PIDS=()

cleanup() {
    set +e
    echo ""
    echo "Caught signal — stopping all workers..."
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    # 等待所有子进程真正结束
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "All workers stopped."
    exit 1
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
    PIDS+=($!)
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
    PIDS+=($!)
done

echo "All $NUM_WORKERS workers launched (PIDs: ${PIDS[*]})."
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done
echo "AB selfplay done."
