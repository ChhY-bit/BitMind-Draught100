# 项目架构说明

本文档用简化图线记录当前工程结构，便于快速恢复上下文。详细规则看 `SPEC.md`，阶段进度看 `PROGRESS.md`。

## 总体结构

```text
BitMind-Draught100
|
+-- engine/                         C 底层规则引擎
|   |
|   +-- include/draughts.h           公共数据结构与函数声明
|   +-- src/board.c                  棋盘、局面、复制、终局
|   +-- src/move.c                   合法动作生成、严格校验、执行
|   +-- tests/                       C 单元测试与随机压力测试
|   +-- tools/board_viewer.c         临时命令行调试工具
|
+-- interface/                       C-Python 接口层
|   |
|   +-- draughts_interface.c/.h       C 侧环境封装，导出共享库接口
|   +-- draughts_env.py              Python ctypes 包装
|   +-- README-interface.md           Python 调用说明
|
+-- gui/                             Python GUI
|   |
|   +-- simple_gui.py                 棋盘显示、点击走子、日志、棋谱、随机AI、监督模型AI
|   +-- README-gui.md                 GUI 使用说明
|
+-- trainer/                         训练、搜索、数据生成
|   |
|   +-- env.py                        标准化训练环境
|   +-- agents.py                     random/greedy/material/alphabeta/supervised/valuesearch
|   +-- arena.py                      批量 agent 对战评测
|   +-- compare_checkpoints.py        自动 checkpoint 晋升评测
|   +-- selfplay.py                   JSONL 对局数据生成
|   +-- rl_selfplay.py                训练专用 RL 探索自博弈数据生成
|   +-- inspect_dataset.py            JSONL 数据检查
|   +-- dataset.py                    JSONL 数据读取与 batch 组装
|   +-- move_encoding.py              候选动作向量编码
|   +-- model.py                      候选动作打分 policy/value 网络
|   +-- train_supervised.py           监督学习训练入口
|   +-- train_selfplay.py             RL self-play candidate 训练入口
|   +-- config/*.yaml                 arena、agent、模型与训练参数配置
|   +-- tests/test_*.py               trainer 测试
|
+-- data/                            本地生成的数据集，不进入版本控制
|
+-- Makefile                         编译、测试入口
+-- SPEC.md                          国际跳棋规则规格
+-- PROGRESS.md                      阶段进度摘要
+-- ARCHITECTURE.md                  本文件
```

## 规则与调用关系

```text
             +----------------------+
             |      SPEC.md         |
             |  国际跳棋规则来源    |
             +----------+-----------+
                        |
                        v
+------------------------------------------------+
|                    engine/                     |
| C 规则黑盒：初始化、合法动作、执行、终局、复制 |
+-----------------------+------------------------+
                        |
                        | make interface -> build/libdraughts.so
                        v
+------------------------------------------------+
|                  interface/                    |
| C 环境封装 + Python ctypes 包装 DraughtsEnv    |
+-----------+----------------------+-------------+
            |                      |
            v                      v
+----------------------+   +----------------------+
|        gui/          |   |       trainer/       |
| 用户交互、棋盘展示   |   | 环境、agent、数据管线 |
+----------------------+   +----------------------+
```

原则：棋规逻辑只在 `engine/` 内实现。Python 端调用 C 接口推进局面，不重新实现吃子、升王、最多吃子、终局等规则。

## Trainer 数据流

```text
engine/ + interface/
        |
        v
trainer/env.py
StandardDraughtsEnv
        |
        +-------------------------+
        |                         |
        v                         v
trainer/agents.py          trainer/arena.py
agent 决策                 批量评测胜率/耗时，可选随机 opening
        |
        v
trainer/compare_checkpoints.py
多 seed/opening 换边评测 candidate vs baseline，并可对外部基线做参考评测
        |
        v
trainer/selfplay.py
生成 JSONL 样本
        |
        v
data/*.jsonl
        |
        +-------------------------+-------------------------+
        |                         |                         |
        v                         v                         v
trainer/inspect_dataset.py trainer/dataset.py       trainer/move_encoding.py
检查数据质量               读取样本并组装 batch     编码候选动作
                                  |
                                  v
                         trainer/model.py
                         候选动作打分网络
                                  |
                                  v
                         trainer/train_supervised.py
                         监督学习训练与 checkpoint 保存
                                  |
                                  v
                         trainer/checkpoints/*.pt

trainer/checkpoints/*.pt
        |
        v
trainer/rl_selfplay.py
按 checkpoint policy temperature 采样
        |
        v
data/rl_*.jsonl
保存行为概率、value 预测和最终结果
        |
        v
trainer/train_selfplay.py
PPO 风格 clipped policy/value 更新
        |
        v
trainer/checkpoints/candidate_move_net_rl_*.pt
```

## 当前训练样本格式

`selfplay.py` 与 `rl_selfplay.py` 都生成 JSONL，一行表示“一手棋之前”的训练样本。

```text
sample
|
+-- raw_squares              长度 50 的原始棋盘
+-- observation              6 x 10 x 10 训练编码
+-- legal_action_ids         当前合法动作编号
+-- legal_moves              当前合法动作详情
+-- action_mask              长度 512 的合法动作掩码
+-- selected_action_id       agent 实际选择的动作
+-- selected_move            选择动作的 path/captures
+-- final_result             该局最终结果
+-- value_target             当前行棋方视角的胜负标签
```

`rl_selfplay.py` 在这些公共字段之外还记录 `policy_probs`、`selected_policy_prob`、`predicted_value` 和 `policy_checkpoint`，用于后续自博弈策略更新阶段辨认生成策略及其行为概率。

当前神经网络路线采用候选动作打分模型：`move_encoding.py` 将每个候选动作编码成固定长度向量，网络对当前局面的每个合法动作分别打分。训练侧局面和动作编码使用当前方视角归一化；白方行棋时按 `square_id -> 51 - square_id` 做 180 度旋转，使己方前进方向在输入中保持一致。

## 常用命令入口

```bash
make test
make interface-test
make trainer-test
make tools
```

多 opening arena 评测：

```bash
PYTHONPATH=interface:trainer python3 trainer/arena.py \
  --black supervised \
  --white supervised \
  --black-checkpoint trainer/checkpoints/candidate_move_net_rl_best.pt \
  --white-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 100 \
  --opening-random-plies 8 \
  --seed 2026 \
  --device cpu
```

自动 checkpoint 对比：

```bash
PYTHONPATH=interface:trainer python3 trainer/compare_checkpoints.py \
  --candidate trainer/checkpoints/candidate_move_net_rl_best.pt \
  --baseline trainer/checkpoints/candidate_move_net_rl_iter_003_best.pt \
  --preset quick \
  --report reports/compare_iter_004.json \
  --device cpu
```

`compare_checkpoints.py` 支持 `quick`、`standard`、`full`、`alphabeta` 评测预设。日常 PPO 迭代先用 `quick` 或 `standard`，只有候选模型接近晋级时再用包含 AlphaBeta 的慢速 `full` 评测。

生成数据：

```bash
PYTHONPATH=interface:trainer python3 trainer/selfplay.py
```

生成 RL 探索数据：

```bash
PYTHONPATH=interface:trainer python3 trainer/rl_selfplay.py \
  --checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 100 \
  --output data/rl_iter_001.jsonl \
  --device cpu
```

检查数据：

```bash
PYTHONPATH=interface:trainer python3 trainer/inspect_dataset.py data/
```

监督训练：

```bash
PYTHONPATH=interface:trainer python3 trainer/train_supervised.py data \
  --resume trainer/checkpoints/candidate_move_net_latest.pt \
  --checkpoint-dir trainer/checkpoints \
  --device cpu
```

RL candidate 训练：

```bash
PYTHONPATH=interface:trainer python3 trainer/train_selfplay.py data/rf_data \
  --resume trainer/checkpoints/candidate_move_net_best.pt \
  --checkpoint-dir trainer/checkpoints \
  --device cpu
```
