#!/bin/bash
# 并行 MCTS 自博弈 — CPU（无 GPU 时使用）
# 用法: ./run/selfplay_cpu.sh
#
# 每个 worker 是独立的 selfplay 进程，各自做 MCTS 搜索。
# CPU 上网络推理较慢，建议 worker 数 ≤ 总核数/2，避免争抢。
# MCTS 搜索完全是顺序的，不依赖 GPU 批量推理。
set -e

cd "$(dirname "$0")/.."

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
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "All workers stopped."
    exit 1
}
trap cleanup SIGINT SIGTERM

CHECKPOINT="${CHECKPOINT:-trainer/checkpoints/round5/candidate_move_net_best.pt}"
MCTS_CONFIG="${MCTS_CONFIG:-draughts/config/mcts.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-data/selfplay_cpu}"
NUM_WORKERS="${NUM_WORKERS:-20}"
BASE_SEED="${BASE_SEED:-9999}"
GAMES="${GAMES:-50}"

mkdir -p "$OUTPUT_DIR"

echo "=== CPU MCTS: $NUM_WORKERS workers × $GAMES games each ==="
echo "    checkpoint: $CHECKPOINT"
echo "    mcts config: $MCTS_CONFIG"
echo "    output:      $OUTPUT_DIR"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    OMP_NUM_THREADS=2 PYTHONPATH=interface:. \
    python3 scripts/selfplay.py \
        --black mcts --white mcts \
        --black-checkpoint "$CHECKPOINT" \
        --white-checkpoint "$CHECKPOINT" \
        --games "$GAMES" \
        --opening-random-plies 8 \
        --mcts-config "$MCTS_CONFIG" \
        --device cpu \
        --output "$OUTPUT_DIR/selfplay_$(printf '%03d' "$i").jsonl" \
        --seed $((BASE_SEED + i)) &
    PIDS+=($!)
done

echo "All $NUM_WORKERS workers launched on CPU (PIDs: ${PIDS[*]})."
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done
echo "CPU done."
