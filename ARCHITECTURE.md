# 项目架构说明

本文档记录当前工程结构，便于快速恢复上下文。详细规则看 `SPEC.md`，阶段进度看 `PROGRESS.md`。

## 总体结构

```text
BitMind-Draught100
│
├── engine/                         C 底层规则引擎
│   ├── include/draughts.h            公共数据结构与函数声明
│   ├── src/board.c                   棋盘、局面、复制、终局
│   ├── src/move.c                    合法动作生成、严格校验、执行
│   ├── tests/                        C 单元测试与随机压力测试
│   └── tools/board_viewer.c          命令行调试工具
│
├── interface/                       C↔Python 接口层
│   ├── draughts_interface.c/.h       C 侧环境封装，导出共享库接口
│   ├── draughts_env.py              Python ctypes 包装
│   └── test_interface.py            Python 接口测试
│
├── draughts/                        核心 Python 库（被 import，不直接运行）
│   ├── env.py                        StandardDraughtsEnv 训练环境
│   ├── move_encoding.py              候选动作向量编码
│   ├── model.py                      CandidateMoveNet 网络定义
│   ├── agents.py                     所有 Agent 类与工厂函数
│   ├── dataset.py                    JSONL 数据读取与 batch 组装
│   ├── mcts.py                       Phase 2: MCTS 搜索（待实现）
│   ├── config/                       模型与 Agent 配置
│   └── tests/                        对应模块的单元测试
│
├── trainer/                         训练器（纯训练循环）
│   ├── train_supervised.py           监督/蒸馏学习训练入口
│   ├── config/                       训练参数配置
│   └── tests/
│
├── scripts/                         独立应用脚本
│   ├── selfplay.py                   Agent 对局数据生成
│   ├── search_improve.py            搜索改进数据生成
│   ├── arena.py                      批量 Agent 对战评测
│   ├── compare_checkpoints.py        自动 checkpoint 晋升评测
│   ├── inspect_dataset.py           JSONL 数据质量检查
│   ├── config/                       脚本参数配置
│   └── tests/
│
├── gui/                             Python GUI
│   └── simple_gui.py                 棋盘显示、人机对弈
│
├── Makefile                         编译、测试入口
├── SPEC.md                          国际跳棋规则规格
├── PROGRESS.md                      阶段进度摘要
└── ARCHITECTURE.md                  本文件
```

## 层间依赖

```text
scripts/                    ← 独立应用，调用 draughts 库 + interface
trainer/                    ← 训练循环，调用 draughts 库
gui/                        ← 用户界面，调用 draughts 库 + interface
draughts/                   ← 核心库，被上层 import
interface/                  ← C 共享库 Python 绑定
engine/                     ← C 规则引擎（编译进 libdraughts.so）
```

原则：上层只向下依赖，不允许跨层互相导入。棋规逻辑只在 `engine/` 内实现。Python 端调用 C 接口推进局面，不重新实现吃子、升王、最多吃子、终局等规则。

Phase 2 新增模块：`draughts/mcts.py`（MCTS 搜索），调用 `draughts/model.py`（网络推理）和 `draughts/env.py`（局面克隆）。

## 数据流（Phase 1）

```text
engine/ + interface/
        │
        v
draughts/env.py
StandardDraughtsEnv
        │
        ├──────────────────────────┐
        │                          │
        v                          v
draughts/agents.py           scripts/selfplay.py
agent 决策                  scripts/search_improve.py
        v                          │
scripts/arena.py                   v
scripts/compare_checkpoints.py  data/*.jsonl
批量评测                         │
                                 v
                          draughts/dataset.py
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    v                         v
          trainer/train_supervised.py
          监督/蒸馏训练
                    │
                    v
          checkpoints/*.pt
```

## 常用命令入口

```bash
make test              # C 引擎单元测试
make interface-test    # C↔Python 接口测试
make draughts-test     # 核心库测试
make trainer-test      # 训练器测试
make scripts-test      # 脚本测试
make all-python-test   # 全部 Python 测试
```

评测：

```bash
PYTHONPATH=interface:. python3 scripts/arena.py \
  --black supervised \
  --white alphabeta \
  --black-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 100 --device cpu
```

自动 checkpoint 对比：

```bash
PYTHONPATH=interface:. python3 scripts/compare_checkpoints.py \
  --candidate trainer/checkpoints/candidate_move_net_rl_best.pt \
  --baseline trainer/checkpoints/candidate_move_net_best.pt \
  --preset quick --report compare_result.json --device cpu
```

数据生成：

```bash
PYTHONPATH=interface:. python3 scripts/selfplay.py
PYTHONPATH=interface:. python3 scripts/search_improve.py data/input.jsonl \
  --output data/improved.jsonl --alphabeta-depth 3
```

训练：

```bash
PYTHONPATH=interface:. python3 trainer/train_supervised.py data \
  --resume trainer/checkpoints/candidate_move_net_latest.pt \
  --checkpoint-dir trainer/checkpoints --device cpu
```
