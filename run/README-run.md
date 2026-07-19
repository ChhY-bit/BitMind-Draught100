# Run — 多进程并行脚本

`run/` 包含用于批量数据生成的多进程 shell 脚本。每个脚本将大批量对局拆分为多个独立进程并发执行，充分利用多核 CPU 或多 GPU。

所有脚本需在项目根目录执行，内部会自动 `cd` 到根目录。

## 脚本一览

| 脚本 | 用途 | 计算资源 |
|------|------|---------|
| `selfplay_ab.sh` | AlphaBeta 自博弈（双深度非对称） | CPU 多核 |
| `selfplay_cpu.sh` | MCTS 自博弈 — CPU | CPU 多核 |
| `selfplay_cuda0.sh` | MCTS 自博弈 — GPU 0 | GPU 0 + CPU |
| `selfplay_cuda1.sh` | MCTS 自博弈 — GPU 1 | GPU 1 + CPU |
| `search_improve.sh` | AlphaBeta 并行重新打分（打软标签） | CPU 多核 |

## 用法

所有参数通过环境变量配置，均有默认值，按需覆盖：

```bash
# AlphaBeta 自博弈（CPU 多进程）
ALPHABETA_DEPTH=6 GAMES=2000 NUM_WORKERS=16 OPENING_RANDOM_PLIES=8 \
  ./run/selfplay_ab.sh

# MCTS 自博弈（GPU 多进程，双卡并行）
GAMES=2000 NUM_WORKERS=8 CHECKPOINT=trainer/checkpoints/round6/best.pt \
  MCTS_CONFIG=draughts/config/mcts.yaml \
  ./run/selfplay_cuda0.sh &
./run/selfplay_cuda1.sh &
wait

# MCTS 自博弈（CPU，无 GPU 时使用）
CHECKPOINT=trainer/checkpoints/round5/best.pt \
  NUM_WORKERS=4 GAMES=200 \
  ./run/selfplay_cpu.sh
```

## 参数

### selfplay_ab.sh

采用**双深度非对称**策略：前一半 worker 黑强白弱（D vs D-1），后一半白强黑弱（D-1 vs D），打破镜像僵局以减少超时平局。

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `ALPHABETA_DEPTH` | 6 | 较强方的搜索深度（较弱方 = D-1） |
| `OUTPUT_DIR` | `data/selfplay_ab` | JSONL 输出目录 |
| `NUM_WORKERS` | 8 | 并行进程数（应为偶数，各半分配） |
| `BASE_SEED` | 7021 | 起始随机种子（各 worker 递增） |
| `GAMES` | 500 | 每个 worker 生成局数 |
| `OPENING_RANDOM_PLIES` | 8 | 开局随机半回合数 |

### selfplay_cuda0.sh / selfplay_cuda1.sh

| 环境变量 | cuda0 默认 | cuda1 默认 | 说明 |
|---------|-----------|-----------|------|
| `CHECKPOINT` | `trainer/checkpoints/round5/...` | 同 | 网络权重路径 |
| `MCTS_CONFIG` | `draughts/config/mcts.yaml` | 同 | MCTS 搜索配置文件 |
| `OUTPUT_DIR` | `data/selfplay_cuda0` | `data/selfplay_cuda1` | JSONL 输出目录 |
| `NUM_WORKERS` | 12 | 12 | 并行进程数 |
| `BASE_SEED` | 7021 | 8021 | 起始随机种子 |
| `GAMES` | 500 | 500 | 每个 worker 生成局数 |

> **注意**：`selfplay_cuda0.sh` 和 `selfplay_cuda1.sh` 需配合使用，分别绑定 GPU 0 和 GPU 1。单卡环境只跑其中一个即可。
>
> **关于 MCTS 配置**：自博弈时应使用开启了 `temperature` 和 `add_noise` 的配置以增加对局多样性；评测时则应关闭这两项以追求最强走法。详见 [`draughts/config/mcts.yaml`](../draughts/config/mcts.yaml) 中的注释。

### selfplay_cpu.sh

GPU 不可用时，MCTS 自博弈可退回到 CPU 运行。CPU 上网络推理较慢，建议 worker 数设为总核数的 1/4 ~ 1/2。

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `CHECKPOINT` | `trainer/checkpoints/round5/...` | 网络权重路径 |
| `MCTS_CONFIG` | `draughts/config/mcts.yaml` | MCTS 搜索配置文件 |
| `OUTPUT_DIR` | `data/selfplay_cpu` | JSONL 输出目录 |
| `NUM_WORKERS` | 4 | 并行进程数（建议 ≤ 核数/2） |
| `BASE_SEED` | 9021 | 起始随机种子（各 worker 递增） |
| `GAMES` | 500 | 每个 worker 生成局数 |

### search_improve.sh

用高等 AlphaBeta 对已有 JSONL 数据重新打分，生成软标签（policy target probabilities）。

采用**按文件数均分**策略：将输入目录/文件下的所有 JSONL 文件收集后，均分给各 worker。文件数少于 worker 数时自动缩减 worker 数。

```bash
# 基本用法（使用默认参数）
./run/search_improve.sh

# 自定义参数
INPUT="data/alphabeta/ab_base data/alphabeta/ab_plus" \
OUTPUT=data/alphabeta/improved_soft.jsonl \
NUM_WORKERS=8 \
ALPHABETA_DEPTH=7 \
TEMPERATURE=100 \
MIN_SCORE_MARGIN=0 \
./run/search_improve.sh
```

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `INPUT` | `data/alphabeta` | 输入 JSONL 文件或目录（空格分隔多个路径） |
| `OUTPUT` | `data/alphabeta/improved_soft.jsonl` | 合并后的输出 JSONL 路径 |
| `NUM_WORKERS` | 8 | 并行进程数 |
| `ALPHABETA_DEPTH` | 7 | Teacher AlphaBeta 搜索深度 |
| `TEMPERATURE` | 100 | Softmax 温度（越高分布越软） |
| `MIN_SCORE_MARGIN` | 0 | 最低分数边距（0 = 不过滤，设高可筛除模糊局面） |
| `BASE_SEED` | 2026 | 起始随机种子（各 worker 递增） |

> **注意**：如果输入只有少量大文件，各 worker 分到的行数可能不均。可先用 `split -n l/<N>` 将大文件按行数拆分为小文件后再跑。
