watch -n 5 '
  total=$(wc -l < data/alphabeta/improved_soft.jsonl.tmp.d/all_input.jsonl 2>/dev/null)
  done=$(cat data/alphabeta/improved_soft.jsonl.tmp.d/worker_*.jsonl 2>/dev/null | wc -l)
  if [ "$total" -gt 0 ] && [ "$done" -ge 0 ]; then
    pct=$(awk "BEGIN {printf \"%.1f\", $done*100/$total}")
    n=$(( done * 30 / total ))
    [ "$n" -gt 30 ] && n=30
    bar=$(printf "%${n}s" | sed "s/ /#/g")
    empty=$((30 - n))
    bar_empty=$(printf "%${empty}s" | sed "s/ /-/g")
    printf "+------------------------------------------+\n"
    printf "| done: %6d / %-6d  (%5.1f%%)              |\n" "$done" "$total" "$pct"
    printf "| [%s%s] |\n" "$bar" "$bar_empty"
    now=$(date +%s)
    if [ -f /tmp/_sp_ts ] && [ -f /tmp/_sp_cnt ]; then
      last_ts=$(cat /tmp/_sp_ts)
      last_done=$(cat /tmp/_sp_cnt)
      elapsed=$((now - last_ts))
      if [ "$elapsed" -gt 0 ]; then
        rate=$(awk "BEGIN {printf \"%.0f\", ($done - $last_done) * 60 / $elapsed}")
        remain=$((total - done))
        if [ "$rate" -gt 0 ]; then
          eta=$((remain * 60 / rate))
          eta_h=$((eta / 3600))
          eta_m=$(((eta % 3600) / 60))
          printf "| rate: %d/min  remain: %d  ETA: %dh%02dm     |\n" "$rate" "$remain" "$eta_h" "$eta_m"
        fi
      fi
    fi
    echo "$now" > /tmp/_sp_ts
    echo "$done" > /tmp/_sp_cnt
    printf "+------------------------------------------+\n"
  else
    echo "waiting for data..."
  fi'
