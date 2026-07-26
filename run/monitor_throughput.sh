#!/bin/bash
# ============================================================================
# monitor_throughput.sh — 实时监控自博弈数据产出速度
#
# 用法:
#   自动检测运行中的 selfplay 进程:
#     ./run/monitor_throughput.sh
#
#   指定监控目录:
#     ./run/monitor_throughput.sh -d data/selfplay_cpu -d data/selfplay_cuda0
#
#   调整采样间隔（秒）:
#     ./run/monitor_throughput.sh -i 15
#
#   单次采样（不循环，适合脚本调用）:
#     ./run/monitor_throughput.sh --once
#
# 输出:
#   实时终端仪表盘，按设备分组显示各数据源的样本数、产出速率、预估完成时间。
#   自动从运行中的 selfplay.py 进程解析 --output 和 --device 参数。
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

# ── 默认值 ──────────────────────────────────────────────
INTERVAL=10
ONCE=false
USER_DIRS=()
COLOR=${COLOR:-auto}   # auto | always | never

# ── 颜色定义 ────────────────────────────────────────────
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_DIM='\033[2m'
C_RED='\033[31m'
C_GREEN='\033[32m'
C_YELLOW='\033[33m'
C_CYAN='\033[36m'
C_WHITE='\033[37m'

# ── 参数解析 ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--interval) INTERVAL="$2"; shift 2 ;;
        -d|--dir)      USER_DIRS+=("$2"); shift 2 ;;
        --once)        ONCE=true; shift ;;
        --color)       COLOR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0" | grep -v '^#!/' | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── 终端检测 ────────────────────────────────────────────
if [[ "$COLOR" == "never" ]] || [[ ! -t 1 ]]; then
    C_RESET=''; C_BOLD=''; C_DIM=''; C_RED=''; C_GREEN=''
    C_YELLOW=''; C_CYAN=''; C_WHITE=''
fi

# ── 自动检测运行中的 selfplay 进程 ─────────────────────
auto_detect_sources() {
    local -A seen_dirs=()
    local sources=()

    while IFS= read -r cmdline; do
        # 提取 --output 路径
        local output_path=""
        if [[ "$cmdline" =~ --output[[:space:]]+([^[:space:]]+) ]]; then
            output_path="${BASH_REMATCH[1]}"
        fi
        [[ -z "$output_path" ]] && continue

        # 取目录（进程写的是具体文件如 selfplay_000.jsonl，取其 dirname）
        local dir_path
        dir_path=$(dirname "$output_path")
        [[ -d "$dir_path" ]] || continue

        # 设备类型
        local device="unknown"
        if [[ "$cmdline" =~ --device[[:space:]]+([^[:space:]]+) ]]; then
            device="${BASH_REMATCH[1]}"
        fi

        # Games 总数
        local games="?"
        if [[ "$cmdline" =~ --games[[:space:]]+([0-9]+) ]]; then
            games="${BASH_REMATCH[1]}"
        fi

        # Workers（同名目录下的进程数）
        local key="${device}:${dir_path}"
        if [[ -z "${seen_dirs[$key]:-}" ]]; then
            seen_dirs[$key]=1
            sources+=("${device}|${dir_path}|${games}")
        fi
    done < <(ps aux 2>/dev/null | grep '[s]elfplay\.py' || true)

    # 输出检测到的 source，一行一个：device|dir|games
    if [[ ${#sources[@]} -gt 0 ]]; then
        printf '%s\n' "${sources[@]}"
    fi
}

# ── 构建监控源列表 ──────────────────────────────────────
build_source_list() {
    if [[ ${#USER_DIRS[@]} -gt 0 ]]; then
        for d in "${USER_DIRS[@]}"; do
            local label
            label=$(basename "$d")
            echo "${label}|$(realpath "$d")|?"
        done
    else
        auto_detect_sources
    fi
}

# ── 高效行数统计（单目录） ──────────────────────────────
count_lines() {
    local dir="$1"
    local total=0
    local count=0
    shopt -s nullglob
    for f in "$dir"/*.jsonl; do
        [[ -f "$f" ]] || continue
        count=$((count + 1))
        # wc -l 返回 "行数 文件名"，取第一列
        local n
        n=$(wc -l < "$f" 2>/dev/null || echo 0)
        total=$((total + n))
    done
    shopt -u nullglob
    echo "${total}:${count}"
}

# ── 快照所有源 ──────────────────────────────────────────
take_snapshot() {
    local sources=("$@")
    for entry in "${sources[@]}"; do
        IFS='|' read -r device dir games <<< "$entry"
        local result
        result=$(count_lines "$dir")
        local samples="${result%%:*}"
        local files="${result##*:}"
        echo "${device}|${dir}|${games}|${samples}|${files}"
    done
}

# ── 安全整数比较（处理空值和浮点） ────────────────────
as_int() {
    local val="$1"
    # 去掉小数点及之后的部分，处理空值
    val="${val%%.*}"
    echo "${val:-0}"
}

# ── 格式化数字 ──────────────────────────────────────────
fmt_num() {
    local n
    n=$(as_int "$1")
    if [[ "$n" -ge 1000000 ]]; then
        printf "%.1fM" "$(echo "scale=1; $n/1000000" | bc 2>/dev/null || echo "$n")"
    elif [[ "$n" -ge 1000 ]]; then
        printf "%.1fK" "$(echo "scale=1; $n/1000" | bc 2>/dev/null || echo "$n")"
    else
        printf '%s' "$n"
    fi
}

fmt_rate() {
    local rate
    rate=$(as_int "$1")
    if [[ "$rate" -ge 1000 ]]; then
        printf "%.1fk/s" "$(echo "scale=1; $rate/1000" | bc 2>/dev/null || echo "$rate")"
    elif [[ "$rate" -ge 1 ]]; then
        printf "%d/s" "$rate"
    else
        printf "%.1f/s" "$(echo "scale=1; $rate" | bc 2>/dev/null || echo "$rate")"
    fi
}

fmt_duration() {
    local seconds="$1"
    if [[ "$seconds" -le 0 ]] 2>/dev/null; then
        echo "   --   "
        return
    fi
    if [[ "$seconds" -ge 86400 ]]; then
        local d=$((seconds / 86400))
        local h=$(((seconds % 86400) / 3600))
        printf "%dd%dh" "$d" "$h"
    elif [[ "$seconds" -ge 3600 ]]; then
        local h=$((seconds / 3600))
        local m=$(((seconds % 3600) / 60))
        printf "%dh%02dm" "$h" "$m"
    elif [[ "$seconds" -ge 60 ]]; then
        local m=$((seconds / 60))
        local s=$((seconds % 60))
        printf "%dm%02ds" "$m" "$s"
    else
        printf "%ds" "$seconds"
    fi
}

# ── 预定义重复字符串（避免 tr 对多字节字符的截断） ────
_HLINE="════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
_HSINGLE="────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────"
_BAR_FULL="██████████████████████████████████████████████████"
_BAR_WIDTH=20
_HLINE_LEN=200

_repeat() {
    # 取预定义字符串的前 $1 个字符
    local n="$1"
    local src="$2"
    [[ "$n" -gt "$_HLINE_LEN" ]] && n=$_HLINE_LEN
    [[ "$n" -le 0 ]] && n=0
    printf '%s' "${src:0:$n}"
}
draw_bar() {
    local fraction="$1"   # 0.0 ~ 1.0
    local filled
    filled=$(echo "scale=0; ($fraction * $_BAR_WIDTH + 0.5)/1" | bc 2>/dev/null || echo 0)
    [[ "$filled" -gt "$_BAR_WIDTH" ]] && filled=$_BAR_WIDTH
    [[ "$filled" -lt 0 ]] && filled=0
    printf '%s' "${_BAR_FULL:0:$filled}"
}

# ── 渲染仪表盘 ──────────────────────────────────────────
render_dashboard() {
    local -n _prev_map="$1"
    local -n _curr_map="$2"
    local -n _files_map="$3"
    local -n _games_map="$4"
    local elapsed_raw="$5"
    local sources=("${@:6}")

    # 用 bc 比较浮点：elapsed > 0 ?
    local elapsed_int
    elapsed_int=$(as_int "$elapsed_raw")
    local has_elapsed=false
    if [[ "$elapsed_int" -gt 0 ]]; then
        has_elapsed=true
    fi

    local total_prev=0 total_curr=0
    declare -A _dev_delta _dev_rate
    local max_rate=0

    for entry in "${sources[@]}"; do
        IFS='|' read -r device _dir _games <<< "$entry"
        local curr=0 prev=0
        curr="${_curr_map[$device]:-0}"
        prev="${_prev_map[$device]:-0}"
        curr=$(as_int "$curr")
        prev=$(as_int "$prev")

        local delta=$((curr - prev))
        [[ "$delta" -lt 0 ]] && delta=0

        local rate=0
        if $has_elapsed; then
            rate=$(echo "scale=0; $delta / $elapsed_raw" | bc 2>/dev/null || echo 0)
            rate=$(as_int "$rate")
        fi

        _dev_delta[$device]=$delta
        _dev_rate[$device]=$rate

        total_prev=$((total_prev + prev))
        total_curr=$((total_curr + curr))
        [[ "$rate" -gt "$max_rate" ]] && max_rate=$rate
    done

    local total_delta=$((total_curr - total_prev))
    [[ "$total_delta" -lt 0 ]] && total_delta=0
    local total_rate=0
    if $has_elapsed; then
        total_rate=$(echo "scale=0; $total_delta / $elapsed_raw" | bc 2>/dev/null || echo 0)
        total_rate=$(as_int "$total_rate")
    fi

    # ── 终端宽度 ─────────────────────────────────────────
    local tw=80
    if command -v tput &>/dev/null; then
        tw=$(tput cols 2>/dev/null || echo 80)
    fi
    [[ "$tw" -lt 60 ]] && tw=60

    # ── 渲染 ─────────────────────────────────────────────
    clear 2>/dev/null || printf '\n\033[2J\033[H'

    # 标题栏
    local now
    now=$(date '+%Y-%m-%d %H:%M:%S')
    printf "${C_BOLD}${C_CYAN}╔══ Selfplay Throughput Monitor ${C_RESET}"
    printf "${C_DIM}%s${C_RESET}  ${C_CYAN}refresh: %ds${C_RESET} " "$now" "$INTERVAL"
    _repeat $((tw - 55)) "$_HLINE" && echo
    printf "${C_CYAN}║${C_RESET}\n"

    # 表头
    printf "${C_CYAN}║${C_RESET} ${C_BOLD}%-16s %6s %10s %10s %10s %10s${C_RESET}\n" \
        "SOURCE" "FILES" "SAMPLES" "Δ" "RATE" "GAMES"
    printf "${C_CYAN}╠%s${C_RESET}\n" "$(_repeat $((tw-2)) "$_HSINGLE")"

    # 各行
    for entry in "${sources[@]}"; do
        IFS='|' read -r device _dir _games <<< "$entry"
        local curr="${_curr_map[$device]:-0}"
        local files="${_files_map[$device]:-0}"
        local games="${_games_map[$device]:-?}"
        local delta="${_dev_delta[$device]:-0}"
        local rate="${_dev_rate[$device]:-0}"

        # 速率颜色
        local rate_color="$C_GREEN"
        if [[ "$rate" -eq 0 ]]; then
            rate_color="$C_RED"
        elif [[ "$max_rate" -gt 0 ]]; then
            local pct=$((rate * 100 / max_rate))
            if [[ "$pct" -lt 30 ]]; then
                rate_color="$C_YELLOW"
            fi
        fi

        local label="${device}"
        [[ ${#label} -gt 16 ]] && label="${label:0:15}…"

        # 条形图（占总速率比）
        local bar_frac=0
        if [[ "$max_rate" -gt 0 ]]; then
            bar_frac=$(echo "scale=2; $rate / $max_rate" | bc 2>/dev/null || echo 0)
        fi

        printf "${C_CYAN}║${C_RESET} ${C_BOLD}%-16s${C_RESET} ${C_DIM}%6s${C_RESET} %10s ${C_GREEN}+%-9s${C_RESET} ${rate_color}%10s${C_RESET} %10s ${C_DIM}%s${C_RESET}\n" \
            "$label" \
            "$files" \
            "$(fmt_num "$curr")" \
            "$(fmt_num "$delta")" \
            "$(fmt_rate "$rate")" \
            "$games" \
            "$(draw_bar "$bar_frac")"
    done

    # 分隔线
    printf "${C_CYAN}╟%s${C_RESET}\n" "$(_repeat $((tw-2)) "$_HSINGLE")"

    # 合计行
    printf "${C_CYAN}║${C_RESET} ${C_BOLD}%-16s${C_RESET} %6s %10s ${C_GREEN}+%-9s${C_RESET} ${C_WHITE}%10s${C_RESET}\n" \
        "TOTAL" \
        "-" \
        "$(fmt_num "$total_curr")" \
        "$(fmt_num "$total_delta")" \
        "$(fmt_rate "$total_rate")"

    # 底部
    printf "${C_CYAN}╚%s${C_RESET}\n" "$(_repeat $((tw-2)) "$_HLINE")"

    # 速率分布图
    if [[ "$max_rate" -gt 0 ]]; then
        printf "\n${C_BOLD}Rate Distribution:${C_RESET}\n"
        for entry in "${sources[@]}"; do
            IFS='|' read -r device _dir _games <<< "$entry"
            local rate="${_dev_rate[$device]:-0}"
            local bar_frac=0
            if [[ "$max_rate" -gt 0 ]]; then
                bar_frac=$(echo "scale=2; $rate / $max_rate" | bc 2>/dev/null || echo 0)
            fi
            local pct
            pct=$(echo "scale=0; $bar_frac * 100" | bc 2>/dev/null || echo 0)
            pct=$(as_int "$pct")
            printf "  %-16s ${C_CYAN}%-20s${C_RESET} ${C_BOLD}%3d%%${C_RESET}  %s\n" \
                "$device" \
                "$(draw_bar "$bar_frac")" \
                "$pct" \
                "$(fmt_rate "$rate")"
        done
    fi

    # 提示
    if $has_elapsed && [[ "$total_rate" -gt 0 ]]; then
        printf "\n${C_DIM}Sampling every ${INTERVAL}s. Press Ctrl+C to exit.${C_RESET}\n"
    fi
}

# ── 主循环 ──────────────────────────────────────────────
main() {
    # 构建源列表
    local raw_sources
    raw_sources=$(build_source_list)
    if [[ -z "$raw_sources" ]]; then
        echo "No selfplay sources found." >&2
        echo "Specify directories with -d or start selfplay processes first." >&2
        exit 1
    fi

    mapfile -t SOURCES <<< "$raw_sources"

    # 关联数组：样本数、文件数、游戏数（按 device 索引）
    declare -A prev_samples curr_samples curr_files curr_games

    # 初始快照
    echo "Taking initial snapshot..." >&2
    local initial_data
    initial_data=$(take_snapshot "${SOURCES[@]}")

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        IFS='|' read -r device _dir _games samples files <<< "$line"
        prev_samples[$device]=$samples
        curr_samples[$device]=$samples
        curr_files[$device]=$files
        curr_games[$device]=$_games
    done <<< "$initial_data"

    local last_time
    last_time=$(date +%s.%N)

    # ── 单次模式 ─────────────────────────────────────────
    if $ONCE; then
        echo ""
        for entry in "${SOURCES[@]}"; do
            IFS='|' read -r device dir _games <<< "$entry"
            local result
            result=$(count_lines "$dir")
            local samples="${result%%:*}"
            local files="${result##*:}"
            printf "%-16s  files=%-3s  samples=%s\n" "$device" "$files" "$(fmt_num "$samples")"
        done
        exit 0
    fi

    # ── 持续监控 ─────────────────────────────────────────
    local tick=0
    while true; do
        sleep "$INTERVAL"
        tick=$((tick + 1))

        local now
        now=$(date +%s.%N)
        local elapsed
        elapsed=$(echo "scale=3; $now - $last_time" | bc 2>/dev/null || echo "$INTERVAL")
        last_time=$now

        # 刷新源列表（进程可能增减）
        if [[ $((tick % 6)) -eq 0 ]] && [[ ${#USER_DIRS[@]} -eq 0 ]]; then
            raw_sources=$(auto_detect_sources || true)
            if [[ -n "$raw_sources" ]]; then
                mapfile -t SOURCES <<< "$raw_sources"
            fi
        fi

        # 当前快照
        local snap_data
        snap_data=$(take_snapshot "${SOURCES[@]}")

        # 滚动：本轮 → 上轮，快照 → 本轮
        while IFS= read -r line; do
            [[ -z "$line" ]] && continue
            IFS='|' read -r device _dir _games samples files <<< "$line"
            prev_samples[$device]="${curr_samples[$device]:-0}"
            curr_samples[$device]=$samples
            curr_files[$device]=$files
            curr_games[$device]=$_games
        done <<< "$snap_data"

        # 渲染
        render_dashboard \
            prev_samples curr_samples curr_files curr_games \
            "$elapsed" \
            "${SOURCES[@]}"
    done
}

# ── 信号处理 ────────────────────────────────────────────
cleanup() {
    printf '\n\n%sMonitor stopped.%s\n' "$C_DIM" "$C_RESET"
    exit 0
}
trap cleanup SIGINT SIGTERM

main
