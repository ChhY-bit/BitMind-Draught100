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
|   +-- simple_gui.py                 棋盘显示、点击走子、日志、棋谱、随机AI
|   +-- README-gui.md                 GUI 使用说明
|
+-- trainer/                         训练、搜索、数据生成
|   |
|   +-- env.py                        标准化训练环境
|   +-- agents.py                     random/greedy/material/alphabeta
|   +-- arena.py                      批量 agent 对战评测
|   +-- selfplay.py                   JSONL 对局数据生成
|   +-- inspect_dataset.py            JSONL 数据检查
|   +-- dataset.py                    JSONL 数据读取与 batch 组装
|   +-- config/*.yaml                 agent 参数配置
|   +-- test_*.py                     trainer 测试
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
agent 决策                 批量评测胜率/耗时
        |
        v
trainer/selfplay.py
生成 JSONL 样本
        |
        v
data/*.jsonl
        |
        +-------------------------+
        |                         |
        v                         v
trainer/inspect_dataset.py trainer/dataset.py
检查数据质量               读取样本并组装 batch
                                  |
                                  v
                         后续 PyTorch 训练代码
```

## 当前训练样本格式

`selfplay.py` 生成 JSONL，一行表示“一手棋之前”的训练样本。

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

当前 `action_id` 是“当前局面合法动作列表编号”，不是全局固定动作空间。后续如果要训练更通用的策略网络，需要继续设计固定动作空间或动作候选打分模型。

## 常用命令入口

```bash
make test
make interface-test
make trainer-test
make tools
```

生成数据：

```bash
PYTHONPATH=interface:trainer python3 trainer/selfplay.py --black alphabeta --white alphabeta --games 100 --output data/selfplay.jsonl
```

检查数据：

```bash
PYTHONPATH=interface:trainer python3 trainer/inspect_dataset.py data/
```
