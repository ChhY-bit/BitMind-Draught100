#!/bin/bash
# 并行 search_improve — 用高等 AlphaBeta 给已有数据重新打软标签
# 用法: ./run/search_improve.sh path/to/data_dir/ path/to/data.jsonl ...
# 每个 worker 分到大致相等数量的 JSONL 文件，处理后合并输出。
#
# 环境变量覆盖（均有默认值）:
#   OUTPUT             输出文件路径
#   NUM_WORKERS        并行进程数
#   ALPHABETA_DEPTH    teacher 搜索深度
#   TEMPERATURE        softmax 温度
#   MIN_SCORE_MARGIN   最低分数边距过滤
#   BASE_SEED          起始随机种子
set -e

cd "$(dirname "$0")/.."

# ── 参数 ──────────────────────────────────────────────
INPUT="${INPUT:-data/alphabeta}"
OUTPUT="${OUTPUT:-data/alphabeta/improved_soft.jsonl}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ALPHABETA_DEPTH="${ALPHABETA_DEPTH:-7}"
TEMPERATURE="${TEMPERATURE:-100}"
MIN_SCORE_MARGIN="${MIN_SCORE_MARGIN:-0}"
BASE_SEED="${BASE_SEED:-2026}"

# ── 收集所有 JSONL 文件 ──────────────────────────────
echo "=== search_improve parallel ==="
echo "    input:       $INPUT"
echo "    output:      $OUTPUT"
echo "    workers:     $NUM_WORKERS"
echo "    ab depth:    $ALPHABETA_DEPTH"
echo "    temperature: $TEMPERATURE"
echo "    margin:      $MIN_SCORE_MARGIN"
echo "    seed:        $BASE_SEED"

TMPDIR="${OUTPUT}.tmp.d"
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

# 把输入路径展开为 JSONL 文件列表
ALL_FILES=()
for p in $INPUT; do
    if [ -d "$p" ]; then
        while IFS= read -r f; do
            ALL_FILES+=("$f")
        done < <(find "$p" -name '*.jsonl' | sort)
    else
        ALL_FILES+=("$p")
    fi
done

TOTAL_FILES=${#ALL_FILES[@]}
if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "ERROR: no .jsonl files found under: $INPUT"
    exit 1
fi

# 如果文件数少于 worker 数，只用文件数个 worker
if [ "$TOTAL_FILES" -lt "$NUM_WORKERS" ]; then
    NUM_WORKERS=$TOTAL_FILES
    echo "    (reduced workers to $NUM_WORKERS, fewer files than workers)"
fi

echo "    total .jsonl files: $TOTAL_FILES"

# ── 按文件数均分给各 worker ──────────────────────────
# 每个 worker 分到的文件数 = ceil(TOTAL_FILES / NUM_WORKERS)
FILES_PER_WORKER=$(( (TOTAL_FILES + NUM_WORKERS - 1) / NUM_WORKERS ))

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

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    START=$((i * FILES_PER_WORKER))
    END=$((START + FILES_PER_WORKER))
    if [ "$END" -gt "$TOTAL_FILES" ]; then
        END=$TOTAL_FILES
    fi

    # 收集该 worker 的文件列表
    WORKER_FILES=""
    for j in $(seq "$START" $((END - 1))); do
        WORKER_FILES="$WORKER_FILES ${ALL_FILES[$j]}"
    done

    WORKER_OUT="$TMPDIR/worker_$(printf '%03d' "$i").jsonl"

    echo "  worker $i: files [$START..$((END-1))] → $WORKER_OUT"

    PYTHONPATH=interface:. python3 scripts/search_improve.py \
        $WORKER_FILES \
        --output "$WORKER_OUT" \
        --alphabeta-depth "$ALPHABETA_DEPTH" \
        --temperature "$TEMPERATURE" \
        --min-score-margin "$MIN_SCORE_MARGIN" \
        --seed $((BASE_SEED + i)) \
        --log-interval 100 &
done

echo "All $NUM_WORKERS workers launched."
wait

# ── 合并 ──────────────────────────────────────────────
echo "Merging outputs..."
cat "$TMPDIR"/worker_*.jsonl > "$OUTPUT"
TOTAL_LINES=$(wc -l < "$OUTPUT")
echo "Merged → $OUTPUT ($TOTAL_LINES samples)"

# 清理临时目录
rm -rf "$TMPDIR"
echo "Done."
