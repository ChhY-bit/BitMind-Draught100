# Run — 多进程并行脚本

`run/` 包含用于批量数据生成的多进程 shell 脚本。每个脚本将大批量对局拆分为多个独立进程并发执行，充分利用多核 CPU 或多 GPU。

所有脚本需在项目根目录执行，内部会自动 `cd` 到根目录。

## 脚本一览

| 脚本 | 用途 | 计算资源 |
|------|------|---------|
| `selfplay_ab.sh` | AlphaBeta 自博弈 | CPU 多核 |
| `selfplay_cuda0.sh` | MCTS 自博弈 — GPU 0 | GPU 0 + CPU |
| `selfplay_cuda1.sh` | MCTS 自博弈 — GPU 1 | GPU 1 + CPU |

## 用法

所有参数通过环境变量配置，均有默认值，按需覆盖：

```bash
# AlphaBeta 自博弈（CPU 多进程）
ALPHABETA_DEPTH=6 GAMES=2000 NUM_WORKERS=16 OPENING_RANDOM_PLIES=8 \
  ./run/selfplay_ab.sh

# MCTS 自博弈（GPU 多进程，双卡并行）
GAMES=2000 NUM_WORKERS=8 CHECKPOINT=trainer/checkpoints/round6/best.pt \
  ./run/selfplay_cuda0.sh &
./run/selfplay_cuda1.sh &
wait
```

## 参数

### selfplay_ab.sh

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `ALPHABETA_DEPTH` | 6 | AlphaBeta 搜索深度 |
| `OUTPUT_DIR` | `data/selfplay_ab` | JSONL 输出目录 |
| `NUM_WORKERS` | 8 | 并行进程数 |
| `BASE_SEED` | 7021 | 起始随机种子（各 worker 递增） |
| `GAMES` | 500 | 每个 worker 生成局数 |
| `OPENING_RANDOM_PLIES` | 8 | 开局随机半回合数 |

### selfplay_cuda0.sh / selfplay_cuda1.sh

| 环境变量 | cuda0 默认 | cuda1 默认 | 说明 |
|---------|-----------|-----------|------|
| `CHECKPOINT` | `trainer/checkpoints/round5/...` | 同 | 网络权重路径 |
| `OUTPUT_DIR` | `data/selfplay_cuda0` | `data/selfplay_cuda1` | JSONL 输出目录 |
| `NUM_WORKERS` | 12 | 12 | 并行进程数 |
| `BASE_SEED` | 7021 | 8021 | 起始随机种子 |
| `GAMES` | 500 | 500 | 每个 worker 生成局数 |

> **注意**：`selfplay_cuda0.sh` 和 `selfplay_cuda1.sh` 需配合使用，分别绑定 GPU 0 和 GPU 1。单卡环境只跑其中一个即可。
