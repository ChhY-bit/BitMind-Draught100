#!/bin/bash
# 并行 MCTS 自博弈 — 单卡或多卡
# 用法:
#   ./run/selfplay_cuda.sh            # 仅 GPU 0
#   ./run/selfplay_cuda.sh 0          # 仅 GPU 0
#   ./run/selfplay_cuda.sh 0 1        # GPU 0 + GPU 1
#   ./run/selfplay_cuda.sh 0 1 2      # GPU 0 + GPU 1 + GPU 2
#
# 作为前台脚本运行，Ctrl+C 会同时终止所有 GPU 上的 worker。
#
# 环境变量（一次传参，所有 GPU 共用）:
#   CHECKPOINT            网络权重路径
#   MCTS_CONFIG           MCTS 搜索配置
#   NUM_WORKERS           每卡并行进程数（默认 12）
#   GAMES                 每个 worker 生成局数（默认 500）
#   OUTPUT_DIR            输出根目录（每卡自动追加 _gpuN 后缀避免冲突）
#   BASE_SEED             起始种子（每卡自动递增 1000）
set -e

cd "$(dirname "$0")/.."

# ── GPU 列表 ──────────────────────────────────────────
GPUS=("$@")
if [ ${#GPUS[@]} -eq 0 ]; then
    GPUS=(0)
fi

# ── 验证 GPU ──────────────────────────────────────────
for gpu in "${GPUS[@]}"; do
    script="run/selfplay_cuda${gpu}.sh"
    if [ ! -f "$script" ]; then
        echo "Error: $script not found (only cuda0/cuda1 scripts exist)" >&2
        exit 1
    fi
done

# ── 共用参数 ──────────────────────────────────────────
export CHECKPOINT="${CHECKPOINT:-trainer/checkpoints/round5/candidate_move_net_best.pt}"
export MCTS_CONFIG="${MCTS_CONFIG:-draughts/config/mcts_selfplay.yaml}"
export NUM_WORKERS="${NUM_WORKERS:-12}"
export GAMES="${GAMES:-500}"

# ── 启动 — 每卡独立 OUTPUT_DIR 和 BASE_SEED ──────────
PIDS=()
echo "=== selfplay_cuda: launching ${#GPUS[@]} GPU(s) ==="
echo "    checkpoint:  ${CHECKPOINT}"
echo "    config:      ${MCTS_CONFIG}"
echo "    per GPU:     ${NUM_WORKERS} workers × ${GAMES} games"

for gpu in "${GPUS[@]}"; do
    # 每卡独立输出目录：用户若指定 OUTPUT_DIR 则自动加 _gpuN 后缀防冲突
    if [ -n "${OUTPUT_DIR+x}" ]; then
        export OUTPUT_DIR="${OUTPUT_DIR}_gpu${gpu}"
    fi
    # 每卡种子递增 1000 避免碰撞
    export BASE_SEED="$(( ${BASE_SEED:-7021} + gpu * 1000 ))"

    echo "    GPU ${gpu}: output=${OUTPUT_DIR:-"(default)"} base_seed=${BASE_SEED}"
    "./run/selfplay_cuda${gpu}.sh" &
    PIDS+=($!)
done

# ── Ctrl+C → 终止所有 GPU ─────────────────────────────
cleanup() {
    set +e
    echo ""
    echo "Caught signal — stopping all GPUs..."
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
    echo "All GPUs stopped."
    exit 1
}
trap cleanup SIGINT SIGTERM

# ── 等待全部完成 ──────────────────────────────────────
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done
echo ""
echo "=== selfplay_cuda: all GPUs done ==="
