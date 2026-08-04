#!/bin/bash
# 并行 MCTS 自博弈 — GPU 0
# 用法: ./run/selfplay_cuda0.sh
# 与 run/selfplay_cuda1.sh 配合，同时榨干两张 GPU。
#
# 每个 worker 是独立的 selfplay 进程，各自做 MCTS 搜索。
# 多进程并发 → CUDA 流交错执行 → GPU 利用率提升。
# 此方法不依赖 MCTS 内部批量推理，算法与 v1.0 顺序搜索等价。
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
MCTS_CONFIG="${MCTS_CONFIG:-draughts/config/mcts_selfplay.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-data/selfplay_cuda0}"
NUM_WORKERS="${NUM_WORKERS:-12}"
BASE_SEED="${BASE_SEED:-7021}"
GAMES="${GAMES:-500}"
FP16="${FP16:-1}"
COMPILE="${COMPILE:-1}"

mkdir -p "$OUTPUT_DIR"

# 构建推理加速参数
INFER_FLAGS=""
[ "$FP16" = "1" ] && INFER_FLAGS="$INFER_FLAGS --fp16"
[ "$COMPILE" = "1" ] && INFER_FLAGS="$INFER_FLAGS --compile"

echo "=== GPU 0: $NUM_WORKERS workers × $GAMES games each ==="
echo "    checkpoint: $CHECKPOINT"
echo "    mcts config: $MCTS_CONFIG"
echo "    output:      $OUTPUT_DIR"
echo "    fp16:        $([ "$FP16" = "1" ] && echo 'on' || echo 'off')"
echo "    compile:     $([ "$COMPILE" = "1" ] && echo 'on' || echo 'off')"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    OMP_NUM_THREADS=2 PYTHONPATH=interface:. \
    python3 scripts/selfplay.py \
        --black mcts --white mcts \
        --black-checkpoint "$CHECKPOINT" \
        --white-checkpoint "$CHECKPOINT" \
        --games "$GAMES" \
        --opening-random-plies 6 \
        --mcts-config "$MCTS_CONFIG" \
        --device cuda:0 \
        --output "$OUTPUT_DIR/selfplay_$(printf '%03d' "$i").jsonl" \
        --seed $((BASE_SEED + i)) \
        $INFER_FLAGS &
    PIDS+=($!)
done

echo "All $NUM_WORKERS workers launched on GPU 0 (PIDs: ${PIDS[*]})."
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done
echo "GPU 0 done."
