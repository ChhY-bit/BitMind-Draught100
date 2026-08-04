# BitMind-Draught100

100 格国际跳棋 AI——使用神经网络引导 MCTS 搜索，构建超越传统 AlphaBeta 的博弈智能体。

## 软硬件要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| Python | 3.10+ | 3.10+ |
| PyTorch | ≥2.0 | ≥2.0 + CUDA |
| C 编译器 | gcc/clang（C11） | — |
| 操作系统 | Linux | Linux |
| GPU | 无（CPU 训练可运行） | NVIDIA GPU（CUDA） |
| 可选 | — | tkinter（GUI） |

唯一外部 Python 依赖是 PyTorch，其余全部使用标准库。

## 快速开始

以下从零开始，按顺序完成环境搭建、测试、训练、评测的完整流程。

### 1. 环境搭建

创建 conda 环境并安装 PyTorch（可参见 [环境配置指南](https://chhy-bit.github.io/docs/doc.html?id=doc001)）。

编译 C 引擎（生成 `build/libdraughts.so`）：

```bash
make interface
```

### 2. 测试

编译完后验证各层是否正常。

```bash
# C 接口（不依赖 torch，先确认底层没问题）
make interface-test
```
```bash
# Python 全线：draughts（核心库）→ trainer（训练器）→ scripts（脚本）
make all-python-test
```

> `make all-python-test` 等价于依次执行 `make draughts-test`、`make trainer-test`、`make scripts-test`，逐层排查问题时可以单独跑某一个。日常开发改完代码后，一行跑完：`make interface-test && make all-python-test`

### 3. 阶段一：AlphaBeta 蒸馏训练

先让 AlphaBeta 自己跟自己下棋产生局面，再用更强的 AlphaBeta 给每个局面打分，最后用这些"标准答案"训练神经网络。这一步产出一个具备基本棋感的基座网络。

**第一步**：AlphaBeta 自博弈，产生对局局面。

```bash
PYTHONPATH=interface:. python3 scripts/selfplay.py \
  --black alphabeta --white alphabeta \
  --games 600 --opening-random-plies 6 --seed 42 \
  --output data/selfplay.jsonl --fp16 --compile
```

> **参数说明**：`--black`/`--white` 指定双方 agent，`--alphabeta-depth` 覆盖搜索深度，`--opening-random-plies` 控制开局随机化。完整参数表见 [scripts/README-scripts.md](scripts/README-scripts.md)。
>
> 大规模生成数据时，可使用多进程脚本 `./run/selfplay_ab.sh` 并行加速，详见[多进程提升利用率](#多进程提升利用率)。

**第二步**：用更强的 AlphaBeta（depth=7）给每个局面重新评分，生成软标签。

```bash
PYTHONPATH=interface:. python3 scripts/search_improve.py data/ \
  --output data/improved_soft.jsonl --log-interval 100 \
  --alphabeta-depth 7 --temperature 100 --min-score-margin 20 --seed 42
```

> **参数说明**：`--alphabeta-depth` 设置 Teacher 搜索深度，`--min-score-margin` 过滤低置信度局面，`--temperature` 控制软标签平滑度。完整参数表见 [scripts/README-scripts.md](scripts/README-scripts.md)。
>
> 大规模数据重新打分可使用 `./run/search_improve.sh` 并行加速，详见[多进程提升利用率](#多进程提升利用率)。

**第三步**：训练神经网络。

```bash
PYTHONPATH=interface:. python3 trainer/train_supervised.py \
  data/improved_soft.jsonl --no-cache --device cuda \
  --checkpoint-dir trainer/checkpoints --seed 42
```

> **参数说明**：`--no-cache` 用于避免 OOM（详见下方注意事项），`--checkpoint-dir` 指定网络数据保存路径，`--seed` 保证训练可复现，`--epochs` 设置训练轮数，`--device` 选择计算设备。完整参数表见 [trainer/README-trainer.md](trainer/README-trainer.md)。

训练完成后，checkpoint 保存在 `trainer/checkpoints/candidate_move_net_best.pt`。

> **重要**：训练时务必加 `--no-cache`。默认情况下，训练器会将整个 JSONL 数据集全部加载到内存中以便加速。对于 MCTS 自博弈产生的大数据集（通常数 GB），不加此参数会迅速耗尽系统 RAM。加上 `--no-cache` 后，数据按需从磁盘逐行读取，内存占用几乎可忽略，训练速度损失极小。详见 [trainer/README-trainer.md](trainer/README-trainer.md)。

### 4. 阶段二：MCTS 自博弈训练

路线一的网络训练好后，就可以用它来做 MCTS 搜索。MCTS 自博弈时，搜索过程本身就是一次"深度思考"——产生的 visit count 分布天然是高质量的训练标签，不需要再让 AlphaBeta 来打分。

**第一步**：MCTS 自博弈，自动产出含软标签的训练数据。

```bash
PYTHONPATH=interface:. python3 scripts/selfplay.py \
  --black mcts --white mcts \
  --black-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --white-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 2000 --opening-random-plies 18 --device cuda \
  --output data/selfplay_mcts.jsonl --seed 42 --fp16 --compile
```

> **参数说明**：`--black-checkpoint`/`--white-checkpoint` 指定双方网络权重，`--seed` 保证对局可复现，`--mcts-config` 可传入自定义 MCTS 参数（sims、c_puct、温度等）。完整参数表见 [scripts/README-scripts.md](scripts/README-scripts.md)。
>
> 大规模生成数据时，有 GPU 用 `./run/selfplay_cuda0.sh`/`./run/selfplay_cuda1.sh`，无 GPU 用 `./run/selfplay_cpu.sh`，详见[多进程提升利用率](#多进程提升利用率)。

**第二步**：从头训练新一代网络。

```bash
PYTHONPATH=interface:. python3 trainer/train_supervised.py \
  data/selfplay_mcts.jsonl --no-cache --epochs 10 --device cuda \
  --checkpoint-dir trainer/checkpoints/round1 --seed 42
```

> **参数说明**：`--checkpoint-dir` 指定本轮 checkpoint 保存目录，`--seed` 保证训练可复现，`--epochs 10` 是 Phase 3 自博弈训练的推荐值。完整参数表见 [trainer/README-trainer.md](trainer/README-trainer.md)。

路线二可以反复执行——每轮用最新网络做 MCTS 自博弈，产出的数据训练下一代网络，形成自增强闭环。详见下方[多轮迭代](#多轮迭代)。

### 5. 评测

MCTS 对决 AlphaBeta，测试棋力：

```bash
PYTHONPATH=interface:. python3 scripts/arena.py \
  --black mcts --white alphabeta \
  --black-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --alphabeta-depth 3 --games 100 --device cuda --seed 42 --compile
```

> **参数说明**：`--black`/`--white` 指定对战双方（支持所有 agent 类型），`--alphabeta-depth` 设置 AlphaBeta 对手深度，`--seed` 保证结果可复现。完整参数表见 [scripts/README-scripts.md](scripts/README-scripts.md)。

两代网络对比（同样 MCTS 搜索，不同网络）：

```bash
PYTHONPATH=interface:. python3 scripts/arena.py \
  --black mcts --white mcts \
  --black-checkpoint new.pt --white-checkpoint old.pt \
  --games 100 --device cuda --seed 42
```

> **参数说明**：双方同为 `mcts` 但加载不同 checkpoint，可评测两代网络棋力差距，`--seed` 保证结果可复现。完整参数表见 [scripts/README-scripts.md](scripts/README-scripts.md)。

所有评测支持 `--opening-random-plies` 参数来随机化开局，避免固定开局的偏差。

---

## 项目架构

```
engine/          C 规则引擎（棋规唯一真源，不修改）
interface/       C↔Python ctypes 桥接（稳定层，不修改）
draughts/        核心 Python 库
  ├── env.py             训练环境（局面编码、走子、克隆）
  ├── model.py           神经网络（CandidateMoveNet，v2/v3 双模架构）
  ├── agents.py          智能体
  ├── mcts.py            MCTS 搜索（PUCT + 网络引导）
  ├── move_encoding.py   走法向量化
  ├── dataset.py         数据加载与批处理
  └── config/            YAML 配置文件
trainer/         训练入口
scripts/         独立应用脚本
gui/             Tkinter 棋盘界面
```

依赖方向（严格单向）：`scripts` / `trainer` → `draughts` → `interface` → `engine`

---

## 智能体

| 名称 | 类型 | 说明 |
|------|------|------|
| `random` | 随机 | 随机走子，基线 |
| `alphabeta` | AlphaBeta | Alpha-Beta 搜索，深度可通过 `--alphabeta-depth` 或 YAML 配置 |
| `supervised` | 网络 greedy | 加载网络权重，直接选最高分走法 |
| `mcts` | MCTS + 网络 | PUCT 搜索，网络提供先验和价值评估 |

所有智能体通过 `make_agent("名称")` 创建。

---

## 设备指定

所有脚本通过 `--device` 参数选择计算设备：

```bash
--device cpu       # 仅 CPU
--device cuda      # GPU 0（等价于 cuda:0）
--device cuda:0    # 指定 GPU 0
--device cuda:1    # 指定 GPU 1
```

### CUDA_VISIBLE_DEVICES vs --device

`CUDA_VISIBLE_DEVICES` 是环境变量，在进程启动前限制可见 GPU；`--device` 是代码内参数，在可见 GPU 中选择用哪张。两者结合使用时注意：`CUDA_VISIBLE_DEVICES` 会重映射 GPU 编号——物理卡 1 在进程内变成 `cuda:0`。

| | `CUDA_VISIBLE_DEVICES` | `--device cuda:N` |
|---|---|---|
| 层面 | 环境变量，进程级 | PyTorch API 参数 |
| 隔离性 | 物理隔离，进程看不到其他 GPU | 逻辑选择，仍能访问其他 GPU |
| 防泄漏 | 任何 CUDA 调用都无法"漏"到其他卡 | 仅约束显式传 device 的代码路径 |

多卡并行时推荐用 `CUDA_VISIBLE_DEVICES` 隔离进程：

```bash
# 终端 1：绑 GPU 0
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=interface:. python3 scripts/selfplay.py \
  --black mcts --white mcts --black-checkpoint ckpt.pt --white-checkpoint ckpt.pt \
  --games 2000 --opening-random-plies 6 --device cuda --output data_0.jsonl --seed 42 --fp16 --compile

# 终端 2：绑 GPU 1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=interface:. python3 scripts/selfplay.py \
  --black mcts --white mcts --black-checkpoint ckpt.pt --white-checkpoint ckpt.pt \
  --games 2000 --opening-random-plies 6 --device cuda --output data_1.jsonl --seed 123 --fp16 --compile
```

`CUDA_VISIBLE_DEVICES` 不会降低 GPU 利用率——它只是隐藏其他 GPU，被选中的 GPU 仍然完全可用。利用率取决于 workload（模型大小、进程数等）。真正影响利用率的是 GPU 上同时跑的任务数量。

### 多进程提升利用率

MCTS 搜索是顺序的（选子→推理→回传循环），单个进程无法让 GPU 持续饱和。项目 `run/` 目录提供了多进程并行脚本，将大批量对局拆分为多个独立进程并发执行：

```bash
# MCTS 自博弈（GPU 多进程）：每卡 8 进程 × 200 局
# 使用 selfplay_cuda.sh 统一管理，Ctrl+C 同时终止所有 GPU。
export GAMES=200 NUM_WORKERS=8 CHECKPOINT=trainer/checkpoints/round6/best.pt
./run/selfplay_cuda.sh 0 1

# MCTS 自博弈（CPU 无 GPU 时使用）：4 进程 × 200 局
CHECKPOINT=trainer/checkpoints/round5/best.pt \
  NUM_WORKERS=4 GAMES=200 \
  ./run/selfplay_cpu.sh

# AlphaBeta 自博弈（CPU 多进程）：16 进程 × 200 局
ALPHABETA_DEPTH=6 GAMES=200 NUM_WORKERS=16 OPENING_RANDOM_PLIES=8 \
  ./run/selfplay_ab.sh

# AlphaBeta 批量重新打分：深度 7 Teacher 给已有数据打软标签
INPUT="data/" OUTPUT=data/improved_soft.jsonl \
  NUM_WORKERS=8 ALPHABETA_DEPTH=7 TEMPERATURE=100 \
  ./run/search_improve.sh
```

所有参数均可通过环境变量覆盖，完整参数表见 [run/README-run.md](run/README-run.md)。每个进程独立搜索，多进程并发即可让 CPU/GPU 饱和。

## 推理加速

自博弈/评测时可叠加以下三项独立优化，综合可将 GPU 推理吞吐提升 3–5 倍：

| 优化手段 | 配置位置 | 加速比 | 说明 |
|------|------|------|------|
| `--fp16` | CLI flag（`selfplay.py`/`arena.py`） | ~1.5–2× | FP16 混合精度推理，无需修改 checkpoint，对搜索质量影响可忽略 |
| `--compile` | CLI flag（`selfplay.py`/`arena.py`） | ~20–40% | `torch.compile` 算子熔合，首次推理有 2–5 秒编译延迟，后续自动复用 |
| `batch_size` | `mcts_selfplay.yaml` 或 `mcts.yaml` | ~2–3× | MCTS 批量叶子推理，聚合多条模拟路径共享一次 GPU 推理，减少 kernel launch 开销 |

**推荐组合**：

| 场景 | 命令 |
|------|------|
| 自博弈（追求速度） | `--fp16 --compile` + `mcts_selfplay.yaml` 中 `batch_size: 4-8` |
| 评测（追求精度） | `--compile`（不推荐 `--fp16`）+ `mcts.yaml` 中 `batch_size: 1` |
| CPU 场景 | 关闭全部（`--no-fp16 --no-compile` + `batch_size: 1`），CPU 上这些优化收益有限甚至为负 |

> **注意**：`--fp16` 和 `--compile` 是运行时覆盖，不修改 checkpoint。同一份 `.pt` 文件可以在自博弈时开、评测时关，互不影响。
> 
> `batch_size` 内部使用虚拟损失（virtual loss）在批次内分散搜索路径，`batch_size=4` 对搜索质量几乎无影响，`batch_size=8` 适用于 sims≥800 的大规模搜索。详见 `mcts_selfplay.yaml` 中的注释。


---

## GUI

编译后启动棋盘界面：

```bash
make interface
```

```bash
PYTHONPATH=interface:. python3 gui/gui.py
```

支持人机对弈、AI 对弈、悔棋、视角旋转。

---

## 多轮迭代

参见上文[路线二：MCTS 自博弈训练](#4-路线二mcts-自博弈训练)。每轮产出的新网络作为下一轮 MCTS 的引擎，反复执行形成自博弈闭环：

```bash
ckpt="trainer/checkpoints/candidate_move_net_best.pt"
for round in 1 2 3; do
  PYTHONPATH=interface:. python3 scripts/selfplay.py \
    --black mcts --white mcts \
    --black-checkpoint "$ckpt" --white-checkpoint "$ckpt" \
    --games 2000 --opening-random-plies 6 --device cuda \
    --output "data/r${round}.jsonl" --seed "$((42 + round))" --fp16 --compile

  PYTHONPATH=interface:. python3 trainer/train_supervised.py \
    "data/r${round}.jsonl" --no-cache --epochs 10 --device cuda \
    --checkpoint-dir "trainer/checkpoints/round${round}" --seed "$((42 + round))"

  ckpt="trainer/checkpoints/round${round}/candidate_move_net_best.pt"
done
```

每轮从头训练，旧数据不混入，仅 checkpoint 传递到下一轮作为搜索引擎。

多卡环境下可将 selfplay 拆分到不同 GPU 并行加速（详见上文[设备指定](#设备指定)），最后合并 JSONL 训练即可。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [SPEC.md](SPEC.md) | 国际跳棋规则 |
| [engine/README-engine.md](engine/README-engine.md) | C 引擎参考 |
| [interface/README-interface.md](interface/README-interface.md) | C↔Python 接口参考 |
| [draughts/README-draughts.md](draughts/README-draughts.md) | 核心库参考 |
| [trainer/README-trainer.md](trainer/README-trainer.md) | 训练器参考 |
| [scripts/README-scripts.md](scripts/README-scripts.md) | 脚本参考 |
| [run/README-run.md](run/README-run.md) | 多进程并行脚本 |
| [gui/README-gui.md](gui/README-gui.md) | GUI 参考 |

---

## 已知限制

- 和棋规则未实现（三次重复、40 步规则），靠 `max_plies` 截断
- `env.clone()` 仍有优化空间（move/undo 替代方案待实现）
- MCTS 搜索可通过 `batch_size` 参数聚合批量叶子推理（`mcts_selfplay.yaml`），配合 `--fp16`/`--compile` CLI flag 和 `run/` 多进程并发，GPU 利用率已大幅提升
- 走法编码仍在 Python 层，C 层向量化可进一步提升速度
- 数据集默认全量载入内存，大数据集务必使用 `--no-cache` 避免 OOM
- 训练仅支持单 GPU；多卡自博弈需手动启动进程隔离（`CUDA_VISIBLE_DEVICES` + 多进程）
- v3 架构（~2.9M 参数）已就绪，通过 `model.yaml` 中 `use_global_pool: true` + `conv_layers: 5` 切换，推理更快、数据需求更低，待训练验证后可能成为新默认
