#!/bin/bash
# 并行 MCTS 自博弈 — GPU 1
# 用法: ./run_selfplay_cuda1.sh
# 与 run_selfplay_cuda0.sh 配合，同时榨干两张 GPU。
set -e

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

CHECKPOINT="${CHECKPOINT:-trainer/checkpoints/round5/candidate_move_net_best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-data/selfplay_cuda1}"
NUM_WORKERS="${NUM_WORKERS:-12}"
BASE_SEED="${BASE_SEED:-8021}"
GAMES="${GAMES:-500}"

mkdir -p "$OUTPUT_DIR"

echo "=== GPU 1: $NUM_WORKERS workers × $GAMES games each ==="
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
        --device cuda:1 \
        --output "$OUTPUT_DIR/selfplay_$(printf '%03d' "$i").jsonl" \
        --seed $((BASE_SEED + i)) &
done

echo "All $NUM_WORKERS workers launched on GPU 1."
wait
echo "GPU 1 done."
