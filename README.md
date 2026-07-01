# BitMind-Draught100

100 格国际跳棋 AI——使用神经网络引导 MCTS 搜索，支持批量推理加速，构建超越传统 AlphaBeta 的博弈智能体。

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

---

## 快速开始

以下从零开始，按顺序完成环境搭建、测试、训练、评测的完整流程。

### 1. 环境搭建

创建 conda 环境并安装 PyTorch（可参见 [环境配置指南](https://chhy-bit.github.io/docs/doc.html?id=doc001)）。

编译 C 引擎（生成 `build/libdraughts.so`）：

```bash
make interface
```

### 2. 测试

编译完后立即验证各层是否正常。

先确认 C 接口可用（不依赖 torch）：

```bash
make interface-test
```

再验证核心库（需要 torch）：

```bash
make draughts-test
```

一次性跑完全部测试：

```bash
make all-python-test
```

### 3. 路线一：AlphaBeta 蒸馏训练

先让 AlphaBeta 自己跟自己下棋产生局面，再用更强的 AlphaBeta 给每个局面打分，最后用这些"标准答案"训练神经网络。这一步产出一个具备基本棋感的基座网络。

**第一步**：AlphaBeta 自博弈，产生对局局面。

```bash
PYTHONPATH=interface:. python3 scripts/selfplay.py \
  --black alphabeta --white alphabeta \
  --games 600 --opening-random-plies 6 --seed 42 \
  --output data/selfplay.jsonl
```

**第二步**：用更强的 AlphaBeta（depth=3）给每个局面重新评分，生成软标签。

```bash
PYTHONPATH=interface:. python3 scripts/search_improve.py data/ \
  --output data/improved_soft.jsonl \
  --alphabeta-depth 3 --temperature 100 --min-score-margin 20 --seed 42
```

**第三步**：训练神经网络。

```bash
PYTHONPATH=interface:. python3 trainer/train_supervised.py \
  data/improved_soft.jsonl --device cuda
```

训练完成后，checkpoint 保存在 `trainer/checkpoints/candidate_move_net_best.pt`。

### 4. 路线二：MCTS 自博弈训练

路线一的网络训练好后，就可以用它来做 MCTS 搜索。MCTS 自博弈时，搜索过程本身就是一次"深度思考"——产生的 visit count 分布天然是高质量的训练标签，不需要再让 AlphaBeta 来打分。

**第一步**：MCTS 自博弈，自动产出含软标签的训练数据。

```bash
PYTHONPATH=interface:. python3 scripts/selfplay.py \
  --black mcts --white mcts \
  --black-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --white-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 2000 --opening-random-plies 6 --device cuda \
  --output data/selfplay_mcts.jsonl
```

> **批量推理调参**：如需调整 batch_size，通过 `--mcts-config` 传入自定义 YAML：
> ```bash
> # CPU 模式示例（小 batch）
> PYTHONPATH=interface:. python3 scripts/selfplay.py \
>   --black mcts --white mcts \
>   --black-checkpoint ckpt.pt --white-checkpoint ckpt.pt \
>   --mcts-config draughts/config/mcts.yaml \
>   --games 100 --device cpu
> ```

**第二步**：从头训练新一代网络。

```bash
PYTHONPATH=interface:. python3 trainer/train_supervised.py \
  data/selfplay_mcts.jsonl --epochs 10 --device cuda \
  --checkpoint-dir trainer/checkpoints/round1
```

路线二可以反复执行——每轮用最新网络做 MCTS 自博弈，产出的数据训练下一代网络，形成自增强闭环。详见下方[多轮迭代](#多轮迭代)。

### 5. 评测

MCTS 对决 AlphaBeta，测试棋力：

```bash
PYTHONPATH=interface:. python3 scripts/arena.py \
  --black mcts --white alphabeta_plus \
  --black-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 100 --device cuda
```

两代网络对比（同样 MCTS 搜索，不同网络权重）：

```bash
PYTHONPATH=interface:. python3 scripts/arena.py \
  --black mcts --white mcts \
  --black-checkpoint new.pt --white-checkpoint old.pt \
  --games 100 --device cuda
```

自动晋升评测，判断新 checkpoint 是否值得替换基线：

```bash
PYTHONPATH=interface:. python3 scripts/compare_checkpoints.py \
  --candidate new.pt --baseline baseline.pt \
  --preset standard --device cuda
```

所有评测支持 `--opening-random-plies` 参数来随机化开局，避免固定开局的偏差。

---

## 项目架构

```
engine/          C 规则引擎（棋规唯一真源，不修改）
interface/       C↔Python ctypes 桥接（稳定层，不修改）
draughts/        核心 Python 库
  ├── env.py             训练环境（局面编码、走子、克隆）
  ├── model.py           神经网络（CandidateMoveNet）
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
| `material` | 启发式 | 材料+位置启发式，一步前瞻 |
| `alphabeta` | AlphaBeta | depth=2 搜索 |
| `alphabeta_plus` | AlphaBeta | depth=3 搜索，最强手工基线 |
| `supervised` | 网络 greedy | 加载网络权重，直接选最高分走法 |
| `mcts` | MCTS + 网络 | PUCT 搜索 + 批量推理，网络提供先验和价值评估 |

所有智能体通过 `make_agent("名称")` 创建。

---

## MCTS 批量推理

MCTS 搜索支持**批量推理**：将多个叶子节点的网络评估打包为一个 batch，一次 GPU 前向传播同时处理，大幅减少 CUDA kernel 启动开销。

### 原理

不再逐一走"选子→推理→回传"串行循环，而是：

1. **收集一批叶子**：用虚拟损失（virtual loss）把同一批次的多条路径推向不同分支
2. **一次批量推理**：将所有叶子局面打包为 `[B, 6, 10, 10]` 张量，一次 forward 输出 B 个 policy + value
3. **分别回传**：撤销虚拟损失，将真实值沿各自路径回传

### 配置

关键参数在 `draughts/config/mcts.yaml`：

```yaml
num_simulations: 800   # 总模拟次数（推理加速后可大幅提升）
batch_size: 64         # 每批收集的叶子数
virtual_loss_value: -1.0  # 虚拟损失值（-1.0 推开效果最强）
```

### 调参建议

| 硬件 | 推荐 batch_size | 推荐 num_simulations |
|------|----------------|---------------------|
| CPU 模式 | 1–8 | 80–200 |
| GPU (≥8 GB) | 64–256 | 400–800 |
| GPU (≥24 GB) | 128–512 | 800–1600 |
| GPU (96 GB) | 256–1024 | 800–3200 |

> **注意**：`CandidateMoveNet` 模型极小（3 层 Conv + 小 MLP），即使 batch_size=512 也远未触及显存瓶颈。CPU 模式下设 batch_size=1 即为传统串行模式。

### 速度预期

在 NVIDIA GPU 上，800 sims 批量推理（batch_size=64）相比逐次推理（batch_size=1）可获得约 **5–10 倍**加速，且模拟数可从 80 提升至 800，棋力随之增强。

---

## GUI

编译后启动棋盘界面：

```bash
make interface
```

```bash
PYTHONPATH=interface:. python3 gui/simple_gui.py
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
    --output "data/r${round}.jsonl"

  PYTHONPATH=interface:. python3 trainer/train_supervised.py \
    "data/r${round}.jsonl" --epochs 10 --device cuda \
    --checkpoint-dir "trainer/checkpoints/round${round}"

  ckpt="trainer/checkpoints/round${round}/candidate_move_net_best.pt"
done
```

每轮从头训练，旧数据不混入，仅 checkpoint 传递到下一轮作为搜索引擎。

> **批量推理**：MCTS 自博弈默认使用 `draughts/config/mcts.yaml` 中的 `batch_size`。
> 在 96 GB GPU 上可设 `batch_size: 256` 或更高以获得最大吞吐。CPU 模式建议
> `batch_size: 1`（退化为传统串行 MCTS）。详见[批量推理章节](#mcts-批量推理)。

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
| [gui/README-gui.md](gui/README-gui.md) | GUI 参考 |

---

## 已知限制

- 和棋规则未实现（三次重复、40 步规则），靠 `max_plies` 截断
- `env.clone()` 仍有优化空间（move/undo 替代方案待实现）
- 走法编码仍在 Python 层，C 层向量化可进一步提升速度
- reward 仅提供基础终局奖励（+1/-1/0）
- 数据集默认全量载入内存（35K 样本约 400 MB），超大数据集可通过 `cache_in_memory=False` 关闭
