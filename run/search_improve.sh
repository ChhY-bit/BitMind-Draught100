#!/bin/bash
# 并行 search_improve — 用高等 AlphaBeta 给已有数据重新打软标签
# 用法: ./run/search_improve.sh
#
# 先将所有输入 JSONL 拼接，再按行数均分给各 worker，保证负载均衡。
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
NUM_WORKERS="${NUM_WORKERS:-64}"
ALPHABETA_DEPTH="${ALPHABETA_DEPTH:-7}"
TEMPERATURE="${TEMPERATURE:-100}"
MIN_SCORE_MARGIN="${MIN_SCORE_MARGIN:-0}"
BASE_SEED="${BASE_SEED:-2026}"

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

# ── 收集所有 JSONL 文件 ──────────────────────────────
ALL_FILES=()
for p in $INPUT; do
    if [ -d "$p" ]; then
        while IFS= read -r f; do
            ALL_FILES+=("$f")
        done < <(find "$p" -name '*.jsonl' | sort)
    elif [ -f "$p" ]; then
        ALL_FILES+=("$p")
    fi
done

TOTAL_FILES=${#ALL_FILES[@]}
if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "ERROR: no .jsonl files found under: $INPUT"
    exit 1
fi

echo "    total .jsonl files: $TOTAL_FILES"

# ── 拼接所有输入，按行数均分 ──────────────────────────
MERGED_INPUT="$TMPDIR/all_input.jsonl"
> "$MERGED_INPUT"
for f in "${ALL_FILES[@]}"; do
    cat "$f" >> "$MERGED_INPUT"
done

TOTAL_LINES=$(wc -l < "$MERGED_INPUT")
echo "    total input lines: $TOTAL_LINES"

if [ "$TOTAL_LINES" -lt "$NUM_WORKERS" ]; then
    NUM_WORKERS=$TOTAL_LINES
    echo "    (reduced workers to $NUM_WORKERS, fewer lines than workers)"
fi

if [ "$TOTAL_LINES" -eq 0 ]; then
    echo "ERROR: all input files are empty"
    rm -rf "$TMPDIR"
    exit 1
fi

# split -n l/N: 按行数均分为 N 份
split -n "l/$NUM_WORKERS" -d --additional-suffix=.jsonl "$MERGED_INPUT" "$TMPDIR/chunk_"

# ── 各 worker 并行处理自己的 chunk ────────────────────
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

for chunk in "$TMPDIR"/chunk_*.jsonl; do
    i="${chunk##*chunk_}"
    i="${i%.jsonl}"
    WORKER_OUT="$TMPDIR/worker_${i}.jsonl"

    chunk_lines=$(wc -l < "$chunk")
    echo "  worker $i: $chunk_lines lines → $WORKER_OUT"

    PYTHONPATH=interface:. python3 scripts/search_improve.py \
        "$chunk" \
        --output "$WORKER_OUT" \
        --alphabeta-depth "$ALPHABETA_DEPTH" \
        --temperature "$TEMPERATURE" \
        --min-score-margin "$MIN_SCORE_MARGIN" \
        --seed $((BASE_SEED + 10#${i})) \
        --log-interval 100 &
done

echo "All $NUM_WORKERS workers launched."
wait

# ── 合并 ──────────────────────────────────────────────
echo "Merging outputs..."
cat "$TMPDIR"/worker_*.jsonl > "$OUTPUT"
FINAL_LINES=$(wc -l < "$OUTPUT")
echo "Merged → $OUTPUT ($FINAL_LINES samples)"

# 清理临时目录
rm -rf "$TMPDIR"
echo "Done."
