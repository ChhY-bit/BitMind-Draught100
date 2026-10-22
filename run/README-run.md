# 批量任务手册

`run/` 使用 shell 脚本并行启动数据生成和评测任务。它们供训练与研究使用，不是比赛界面的启动入口。

从项目根目录运行，先激活已安装依赖的环境并执行 `make interface`。模型与数据不随仓库提供；脚本中保留了训练期的默认路径，运行时应显式传入实际路径和合适的 worker 数。

## 脚本索引

| 脚本 | 用途 |
|---|---|
| `selfplay_ab.sh` | 多进程 AlphaBeta 数据生成，双方分别使用深度 D 和 D−1，再交换先后手配置 |
| `selfplay_cpu.sh` | CPU MCTS 自博弈 |
| `selfplay_cuda.sh` | 调用现有 GPU 0/1 子脚本 |
| `selfplay_cuda0.sh` / `selfplay_cuda1.sh` | 指定 GPU 的 MCTS 自博弈 |
| `search_improve.sh` | 合并并分片输入，启动多进程搜索重评分 |
| `benchmark.sh` | 新旧模型及 AlphaBeta 的双向批量对战 |
| `monitor_throughput.sh` | 自博弈产出速率监控 |
| `improve_preview.sh` | 固定默认路径的重评分进度预览 |

各变量的实际默认值以对应脚本为准。`GAMES` 在自博弈脚本中表示每个 worker 的局数，`NUM_WORKERS` 表示进程数；GPU 包装脚本中为每卡进程数。

## 自博弈

```bash
ALPHABETA_DEPTH=6 NUM_WORKERS=4 GAMES=50 \
  OUTPUT_DIR=data/selfplay_ab ./run/selfplay_ab.sh

CHECKPOINT=path/to/model.pt NUM_WORKERS=2 GAMES=20 \
  MCTS_CONFIG=draughts/config/mcts_selfplay.yaml \
  OUTPUT_DIR=data/selfplay_cpu ./run/selfplay_cpu.sh

CHECKPOINT=path/to/model.pt NUM_WORKERS=2 GAMES=50 \
  FP16=0 COMPILE=0 ./run/selfplay_cuda.sh 0 1
```

GPU 包装脚本当前只支持已有 `cuda0` 和 `cuda1` 子脚本，对应参数为 `0`、`1`；不支持直接传入任意 GPU 编号。设备编号按当前进程可见的 CUDA 列表解释。

常用变量包括 `CHECKPOINT`、`MCTS_CONFIG`、`OUTPUT_DIR`、`NUM_WORKERS`、`GAMES` 和 `BASE_SEED`。GPU 子脚本另有 `FP16` 和 `COMPILE`，以 `0` 关闭、`1` 开启。默认搜索配置为 `mcts_selfplay.yaml`，其中 `batch_size: 1` 表示未启用批量叶子推理。

worker 日志可能直接输出到终端或写入文件，数据位置以脚本输出为准。任务结束后检查 worker 日志与预期产出，不仅依据包装脚本的结束提示判断所有对局成功完成。Ctrl+C 用于停止前台任务及其 worker。

## 搜索重评分

```bash
INPUT=data/ab.jsonl OUTPUT=data/improved.jsonl \
  NUM_WORKERS=4 ALPHABETA_DEPTH=7 TEMPERATURE=100 MIN_SCORE_MARGIN=20 \
  ./run/search_improve.sh
```

`INPUT` 可以为文件、目录或空格分隔的多个路径；`OUTPUT` 指定合并结果。脚本将输入合并后按行分片，再分别运行 `scripts/search_improve.py`。分片和 worker 输出位于输出路径对应的临时目录，合并结束后清理；具体路径以运行输出为准。

`improve_preview.sh` 内置默认数据位置，仅适用于该布局。更换输出路径后需根据实际文件查看进度。

## 双向评测

```bash
NEW_CHECKPOINT=path/to/new_model.pt \
  OLD_CHECKPOINT=path/to/old_model.pt \
  ALPHABETA_DEPTH=6 GAMES=50 COMPILE=0 \
  ./run/benchmark.sh
```

脚本同时启动六个 arena 任务：新模型对 AlphaBeta、旧模型对 AlphaBeta、新模型对旧模型，各自交换先后手。固定使用两张可见 GPU，每卡三个任务。

常用变量为 `NEW_CHECKPOINT`、`OLD_CHECKPOINT`、`ALPHABETA_DEPTH`、`GAMES`、`MCTS_CONFIG`、`OPENING_RANDOM_PLIES`、`MAX_PLIES`、`LOG_INTERVAL`、`SEED`、`REPORT_DIR` 和 `COMPILE`。`GAMES` 是每个方向的局数，同一 `SEED` 传给各场任务。

报告默认写入 `run/benchmark_YYYYMMDD_HHMMSS.txt`。原始任务日志在脚本创建的临时目录中，报告列出其位置。检查任务是否完成及截断局数后，再解释胜负统计。

该脚本需要两张 GPU 并发运行。只有单卡、CPU 或显存不足时，按 [脚本手册](../scripts/README-scripts.md#批量对战) 逐项运行 `arena.py`。

## 产出监控

```bash
./run/monitor_throughput.sh
./run/monitor_throughput.sh -d data/selfplay_cpu -d data/selfplay_cuda0
./run/monitor_throughput.sh -i 15
./run/monitor_throughput.sh --once -d data/selfplay_cuda0
```

监控会尝试从运行中的 `selfplay.py` 进程发现输出位置，也可用重复的 `-d` 指定目录。显示文件数、样本数、采样增量及速率；`-i` 为采样间隔，`--once` 仅采样一次，`--color` 可设 `auto`、`always` 或 `never`。

监控以 JSONL 行数估计产出，不判断样本是否合法或模型是否有效。数据验证使用 `scripts/inspect_dataset.py`。

底层 CLI、参数和截断口径见 [脚本手册](../scripts/README-scripts.md)。历史训练结果见 [归档](../docs/archive/README.md)。
