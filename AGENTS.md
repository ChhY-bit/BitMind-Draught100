# AGENTS.md — AI Agent 工作指南

## 项目概述

**BitMind-Draught100**：100 格国际跳棋 AI。最终目标是构建一个网络引导 MCTS 搜索的智能体，棋力超越同等计算预算下的纯 AlphaBeta 搜索。

详见 [ROADMAP.md](ROADMAP.md)。

## 当前阶段：Phase 1 — 蒸馏模仿

用 AlphaBeta (depth=3) 作为 Teacher，训练 `CandidateMoveNet` 逼近其棋力。

- 数据生成：`scripts/selfplay.py`（对局）+ `scripts/search_improve.py`（Teacher 重标注）
- 训练：`trainer/train_supervised.py`
- 评测：`scripts/compare_checkpoints.py`（quick → standard → full 晋级）

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
