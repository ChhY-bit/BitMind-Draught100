#!/bin/bash
# 多维度批量评测 — 并行跑 6 场对战（3 组 × 正反方向），输出含时间戳的汇总报告。
# 用法: ./run/benchmark.sh
#
# 三组评测：
#   1. AlphaBeta vs 新网络 MCTS
#   2. AlphaBeta vs 旧网络 MCTS
#   3. 旧网络 MCTS vs 新网络 MCTS
# 每组正反各跑一场（cuda0 正向, cuda1 反向），消除先后手偏差。
#
# 环境变量（均有默认值）:
#   NEW_CHECKPOINT       新网络权重路径
#   OLD_CHECKPOINT       旧网络权重路径
#   ALPHABETA_DEPTH      AlphaBeta 搜索深度（默认 3）
#   GAMES                每场对局数（默认 100）
#   MCTS_CONFIG          MCTS 搜索配置（默认 draughts/config/mcts.yaml）
#   OPENING_RANDOM_PLIES 开局随机半回合数（默认 6）
#   MAX_PLIES            最大半回合数（默认 300）
#   LOG_INTERVAL         日志输出间隔（默认 20）
#   SEED                 基准随机种子（默认 42）
#   REPORT_DIR           报告输出目录（默认 run/）
set -e

cd "$(dirname "$0")/.."

# ── 参数 ──────────────────────────────────────────────
NEW_CHECKPOINT="${NEW_CHECKPOINT:-trainer/checkpoints/round6/candidate_move_net_best.pt}"
OLD_CHECKPOINT="${OLD_CHECKPOINT:-trainer/checkpoints/round5/candidate_move_net_best.pt}"
ALPHABETA_DEPTH="${ALPHABETA_DEPTH:-3}"
GAMES="${GAMES:-100}"
MCTS_CONFIG="${MCTS_CONFIG:-draughts/config/mcts.yaml}"
OPENING_RANDOM_PLIES="${OPENING_RANDOM_PLIES:-8}"
MAX_PLIES="${MAX_PLIES:-300}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
SEED="${SEED:-1234}"
REPORT_DIR="${REPORT_DIR:-run}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="${REPORT_DIR}/benchmark_${TIMESTAMP}.txt"
TMPDIR="/tmp/benchmark_${TIMESTAMP}.d"
mkdir -p "$TMPDIR"

# ── PID 追踪 ──────────────────────────────────────────
PIDS=()

cleanup() {
    set +e
    echo ""
    echo "Caught signal — stopping all arena processes..."
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
    echo "All processes stopped."
    rm -rf "$TMPDIR"
    exit 1
}
trap cleanup SIGINT SIGTERM

echo "=== Benchmark $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "    new checkpoint: $NEW_CHECKPOINT"
echo "    old checkpoint: $OLD_CHECKPOINT"
echo "    AB depth: $ALPHABETA_DEPTH"
echo "    games per matchup: $GAMES"
echo "    mcts config: $MCTS_CONFIG"
echo "    report: $REPORT"
echo ""

# ── 启动 6 场 arena ────────────────────────────────────
# 约定：输出文件命名 = {组}_{黑}_{白}_{设备}.log
# 组名：ab_new / ab_old / old_new

run_arena() {
    local label="$1" device="$2" black="$3" white="$4"
    local black_ckpt="$5" white_ckpt="$6" seed="$7"
    local out="$TMPDIR/${label}.log"

    echo "  [$label] $black vs $white on $device (seed=$seed)  → $out"

    OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH=interface:. \
    python3 scripts/arena.py \
        --black "$black" --white "$white" \
        --black-checkpoint "$black_ckpt" \
        --white-checkpoint "$white_ckpt" \
        --alphabeta-depth "$ALPHABETA_DEPTH" \
        --mcts-config "$MCTS_CONFIG" \
        --games "$GAMES" \
        --max-plies "$MAX_PLIES" \
        --opening-random-plies "$OPENING_RANDOM_PLIES" \
        --log-interval "$LOG_INTERVAL" \
        --device "$device" \
        --seed "$seed" \
        > "$out" 2>&1 &
    PIDS+=($!)
}

# ── 组 1: AlphaBeta vs 新网络 MCTS ──────────────────────
run_arena "ab_new_cuda0" "cuda:0" "mcts" "alphabeta" \
    "$NEW_CHECKPOINT" "none" "$((SEED + 0))"
run_arena "ab_new_cuda1" "cuda:1" "alphabeta" "mcts" \
    "none" "$NEW_CHECKPOINT" "$((SEED + 1))"

# ── 组 2: AlphaBeta vs 旧网络 MCTS ──────────────────────
run_arena "ab_old_cuda0" "cuda:0" "mcts" "alphabeta" \
    "$OLD_CHECKPOINT" "none" "$((SEED + 2))"
run_arena "ab_old_cuda1" "cuda:1" "alphabeta" "mcts" \
    "none" "$OLD_CHECKPOINT" "$((SEED + 3))"

# ── 组 3: 旧网络 MCTS vs 新网络 MCTS ────────────────────
run_arena "old_new_cuda0" "cuda:0" "mcts" "mcts" \
    "$NEW_CHECKPOINT" "$OLD_CHECKPOINT" "$((SEED + 4))"
run_arena "old_new_cuda1" "cuda:1" "mcts" "mcts" \
    "$OLD_CHECKPOINT" "$NEW_CHECKPOINT" "$((SEED + 5))"

echo ""
echo "All 6 arenas launched (PIDs: ${PIDS[*]})."
echo ""
echo "Watch progress:"
echo "  tail -f $TMPDIR/*.log        # 实时滚动所有日志"
echo "  tail -f $TMPDIR/ab_new_cuda0.log  # 只看某一场"
echo "  grep arena: $TMPDIR/*.log    # 看各场当前进度"
echo "  ls -lh $TMPDIR/              # 看各场日志文件大小"
echo ""
echo "Waiting for completion..."
echo ""

# ── 等待全部完成 ──────────────────────────────────────
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done

# ── 生成报告 ──────────────────────────────────────────
{
    echo "========================================================================"
    echo "  BitMind-Draught100 Benchmark Report"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================================"
    echo ""
    echo "Parameters"
    echo "----------"
    echo "  new checkpoint: $NEW_CHECKPOINT"
    echo "  old checkpoint: $OLD_CHECKPOINT"
    echo "  AB depth:       $ALPHABETA_DEPTH"
    echo "  games/dir:      $GAMES"
    echo "  opening plies:  $OPENING_RANDOM_PLIES"
    echo "  max plies:      $MAX_PLIES"
    echo "  mcts config:    $MCTS_CONFIG"
    echo ""

    # 辅助函数：提取最后一行 arena 统计
    parse_stats() {
        local log="$1"
        if [ -f "$log" ] && [ -s "$log" ]; then
            tail -1 "$log"
        else
            echo "(no data)"
        fi
    }

    echo "Raw results (per direction)"
    echo "---------------------------"
    for label in ab_new_cuda0 ab_new_cuda1 ab_old_cuda0 ab_old_cuda1 old_new_cuda0 old_new_cuda1; do
        log="$TMPDIR/${label}.log"
        echo ""
        echo "  [$label]"
        if [ -f "$log" ] && [ -s "$log" ]; then
            # 打印 arena 配置头（第 1 行）
            head -1 "$log"
            # 打印最后一行统计
            echo "  $(tail -1 "$log")"
        else
            echo "  (missing or empty)"
        fi
    done

    echo ""
    echo "========================================================================="
    echo "  Summary"
    echo "========================================================================="
    echo ""
    printf "  %-30s %6s %6s %6s  %s\n" "Matchup" "NewWin" "OldWin" "Draws" "WinRate(New)"
    printf "  %-30s %6s %6s %6s  %s\n" "------------------------------" "------" "------" "------" "-----------"

    summarize() {
        local label="$1" new_side="$2" old_side="$3"
        local f0="$TMPDIR/${label}_cuda0.log"
        local f1="$TMPDIR/${label}_cuda1.log"

        # 从最后一行 arena 统计中提取数字
        # 格式: arena: N/N games, black_wins=X white_wins=Y draws=Z ...
        extract() {
            local f="$1" field="$2"
            if [ -f "$f" ] && [ -s "$f" ]; then
                tail -1 "$f" | grep -oP "${field}=\K[0-9]+" 2>/dev/null || echo "0"
            else
                echo "0"
            fi
        }

        local bw0=$(extract "$f0" "black_wins")  # cuda0 黑方胜
        local ww0=$(extract "$f0" "white_wins")
        local dw0=$(extract "$f0" "draws")
        local bw1=$(extract "$f1" "black_wins")  # cuda1 黑方胜
        local ww1=$(extract "$f1" "white_wins")
        local dw1=$(extract "$f1" "draws")

        # cuda0: new=黑 old=白, cuda1: old=黑 new=白
        local new_win=$((bw0 + ww1))
        local old_win=$((ww0 + bw1))
        local draws=$((dw0 + dw1))
        local total=$((new_win + old_win + draws))
        local rate
        if [ "$total" -gt 0 ]; then
            rate=$(awk "BEGIN {printf \"%.1f%%\", $new_win*100/$total}")
        else
            rate="N/A"
        fi

        printf "  %-30s %6d %6d %6d  %s\n" "$label" "$new_win" "$old_win" "$draws" "$rate"
    }

    summarize "AB vs NEW" "new" "ab"
    summarize "AB vs OLD" "old" "ab"
    summarize "NEW vs OLD" "new" "old"

    echo ""
    echo "Note: 'New' side aggregates both black and white games."
    echo "      Each row sums 2 × $GAMES = $((GAMES * 2)) games."

    echo ""
    echo "========================================================================="
    echo "  Full logs"
    echo "========================================================================="
    echo ""
    for label in ab_new_cuda0 ab_new_cuda1 ab_old_cuda0 ab_old_cuda1 old_new_cuda0 old_new_cuda1; do
        log="$TMPDIR/${label}.log"
        echo "─── $label ─────────────────────────────────────────────────────"
        if [ -f "$log" ]; then
            cat "$log"
        else
            echo "(missing)"
        fi
        echo ""
    done
} > "$REPORT"

# ── 清理临时文件 ──────────────────────────────────────
rm -rf "$TMPDIR"

echo ""
echo "Benchmark done."
echo "Report: $REPORT"
cat "$REPORT" | head -60  # 打印报告摘要到终端
