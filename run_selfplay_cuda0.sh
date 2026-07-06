#!/bin/bash
# 并行 MCTS 自博弈 — GPU 0
# 用法: ./run_selfplay_cuda0.sh
# 与 run_selfplay_cuda1.sh 配合，同时榨干两张 GPU。
#
# 每个 worker 是独立的 selfplay 进程，各自做 MCTS 搜索。
# 多进程并发 → CUDA 流交错执行 → GPU 利用率提升。
# 此方法不依赖 MCTS 内部批量推理，算法与 v1.0 顺序搜索等价。
set -e

cleanup() {
    echo ""
    echo "Caught signal — killing all workers..."
    jobs -p | xargs -r kill 2>/dev/null
    wait 2>/dev/null
    echo "All workers stopped."
}
trap cleanup SIGINT SIGTERM

CHECKPOINT="${CHECKPOINT:-trainer/checkpoints/round5/candidate_move_net_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-data/selfplay_cuda0}"
NUM_WORKERS="${NUM_WORKERS:-12}"
BASE_SEED="${BASE_SEED:-7021}"
GAMES="${GAMES:-500}"

mkdir -p "$OUTPUT_DIR"

echo "=== GPU 0: $NUM_WORKERS workers × $GAMES games each ==="
echo "    checkpoint: $CHECKPOINT"
echo "    output:     $OUTPUT_DIR"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    OMP_NUM_THREADS=2 PYTHONPATH=interface:. \
    python3 scripts/selfplay.py \
        --black mcts --white mcts \
        --black-checkpoint "$CHECKPOINT" \
        --white-checkpoint "$CHECKPOINT" \
        --games "$GAMES" \
        --opening-random-plies 6 \
        --device cuda:0 \
        --output "$OUTPUT_DIR/selfplay_$(printf '%03d' "$i").jsonl" \
        --seed $((BASE_SEED + i)) &
done

echo "All $NUM_WORKERS workers launched on GPU 0."
wait
echo "GPU 0 done."
