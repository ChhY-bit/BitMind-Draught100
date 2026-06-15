# CLAUDE.md — AI Agent 工作指南

## 项目概述

**BitMind-Draught100**：100 格国际跳棋 AI。最终目标是构建一个网络引导 MCTS 搜索的智能体，棋力超越同等计算预算下的纯 AlphaBeta 搜索。

详见 [ROADMAP.md](ROADMAP.md)。

## 当前阶段：Phase 2 — 搜索增强（网络 + MCTS）

Phase 1 已完成。训练好的 `CandidateMoveNet` 基座网络位于 `trainer/checkpoints/candidate_move_net_best.pt`。

### Phase 1 成果速览

| 指标 | 值 |
|------|-----|
| 训练数据 | 600 局 selfplay → ~35,000 软标签样本（`data/improved_soft.jsonl`） |
| 模型架构 | `CandidateMoveNet`：CNN encoder + policy head + value head |
| Policy accuracy | 75%（best val）~ 86%（final train） |
| 棋力 | 70:30 胜 `material`；0:100 负 `alphabeta`（greedy 推理，缺搜索） |
| soft target | `search_improve.py` 输出 `policy_target_probs`，训练用软交叉熵 |
| 开局随机化 | `selfplay.py` 支持 `--opening-random-plies` |

### Phase 2 目标

实现 `draughts/mcts.py`，用网络引导 MCTS 搜索，使 MCTS+网络 战胜 AlphaBeta depth=3。

**技术要点**：
- 分支选择：PUCT 公式，prior 来自 `policy_logits`
- 叶子评估：`value_head` 输出局面价值
- 搜索返回：visit count 分布（新 policy target）+ 最终 value
- `env.clone()` 性能是 MCTS 瓶颈，需要关注

### 已有 checkpoint

| 文件 | 说明 |
|------|------|
| `trainer/checkpoints/candidate_move_net_best.pt` | best val loss (epoch 13) |
| `trainer/checkpoints/candidate_move_net_latest.pt` | 最终 epoch |

## 项目架构

```
engine/         C 规则引擎（只读，棋规的唯一真源）
interface/      C↔Python ctypes 桥接（只读）
draughts/       核心 Python 库（env, model, agents, dataset, move_encoding）
trainer/        训练入口（train_supervised.py）
scripts/        独立应用脚本（selfplay, search_improve, arena, compare, inspect）
gui/            Tkinter 棋盘界面
```

层间依赖：`scripts/trainer` → `draughts` → `interface` → `engine`

详见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 数据管线

```
selfplay (depth=2, opening_random_plies=6)
  → search_improve (depth=3, temperature=100, min_score_margin=20)
    → train_supervised (自动使用软标签)
```

## 常用命令

```bash
# 测试
make test && make interface-test && make all-python-test

# 数据生成
PYTHONPATH=interface:. python3 scripts/selfplay.py \
  --black alphabeta --white alphabeta \
  --games 600 --opening-random-plies 6 --seed 42 \
  --output data/selfplay_ab2.jsonl

PYTHONPATH=interface:. python3 scripts/search_improve.py data/ \
  --output data/improved_soft.jsonl \
  --alphabeta-depth 3 --temperature 100 --min-score-margin 20 --seed 42

# 训练
PYTHONPATH=interface:. python3 trainer/train_supervised.py data/improved_soft.jsonl \
  --device cuda --resume trainer/checkpoints/candidate_move_net_latest.pt

# 评测（arena 测绝对棋力，compare 测两代模型对比）
PYTHONPATH=interface:. python3 scripts/arena.py \
  --black supervised --white alphabeta \
  --black-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 100 --opening-random-plies 4 --device cuda
```

## 代码规范

### Python
- Python 3.10+
- 新模块优先放入 `draughts/`（库）或 `scripts/`（应用）
- `trainer/` 只放训练循环，不放数据生成或评测
- 所有 CLI 脚本用 argparse，配置用 YAML，CLI 参数覆盖 YAML
- `from __future__ import annotations` 在所有文件中
- 类型标注使用 `Optional`、`Sequence` 等 typing 泛型

### C
- C11，`-Wall -Wextra -Wpedantic -Werror`
- 规则逻辑只在 `engine/src/` 中
- 所有公开函数检查 NULL 参数，返回错误码
- 头文件在 `engine/include/`

### 测试
- 修改核心逻辑必须跑对应测试
- `make test` — C 引擎
- `make interface-test` — C↔Python 接口
- `make draughts-test` — 核心库
- `make trainer-test` — 训练器
- `make scripts-test` — 脚本
- `make all-python-test` — 全部 Python 测试

## 约束

- **不引入大型第三方库**。唯一外部依赖是 `torch`。
- **不修改 engine/ 和 interface/ 的公共 API**。它们是稳定的底层。
- **不删除已有功能的文件**，除非明确讨论过。
- **修改后必须更新相关 README** 和 `PROGRESS.md`。
- 生成新文件时说明理由和用途。
- 函数和接口保持兼容。

## 工作流程

1. 阅读 [ROADMAP.md](ROADMAP.md) 了解当前阶段和长期方向
2. 阅读 `PROGRESS.md` 了解最新状态
3. 修改代码
4. 运行 `make test && make interface-test && make all-python-test`
5. 更新 `PROGRESS.md`

## 无把握时

- 不确定需求时先追问，一次只问一个问题
- 给出可能的选项，而非开放式提问
- 超过 10 个疑问仍未澄清的，汇总列出，先做有把握的部分
