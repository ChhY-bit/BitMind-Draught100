# 项目进度

本文档用于新会话快速恢复上下文。详细棋规见 `SPEC.md`，架构见 `ARCHITECTURE.md`，路线见 `ROADMAP.md`，Agent 约束见 `AGENTS.md`。

## 当前阶段：Phase 1 蒸馏模仿

训练神经网络基座，使其棋力接近 AlphaBeta depth=3。当前项目已完成大规模重构清理，代码路径与 ROADMAP 对齐。

## 已完成

### 基础设施（稳定，不轻易改动）

| 层 | 内容 | 状态 |
|---|---|---|
| `engine/` | C 规则引擎，16 个公开 API，全覆盖单元测试 + 随机压力测试 | ✅ |
| `interface/` | C↔Python ctypes 桥接，`DraughtsEnv` | ✅ |
| `gui/` | Tkinter 棋盘，人/AI 对弈，悔棋，视角旋转 | ✅ |

### 核心库 `draughts/`

| 模块 | 内容 | 状态 |
|---|---|---|
| `env.py` | `StandardDraughtsEnv`，6×10×10 当前方视角编码 | ✅ |
| `move_encoding.py` | 候选动作 → 45 维向量 | ✅ |
| `model.py` | `CandidateMoveNet`：CNN encoder + move MLP + policy/value head | ✅ |
| `agents.py` | 4 个 Agent：`random`、`material`、`alphabeta`（深度通过 config 调）、`supervised` | ✅ |
| `dataset.py` | JSONL 读取、batch 组装、数据源权重 | ✅ |

### 训练 `trainer/`

| 模块 | 内容 | 状态 |
|---|---|---|
| `train_supervised.py` | 监督/蒸馏训练，checkpoint 管理，CSV 日志 | ✅（待加 KL/soft-label loss） |

### 应用脚本 `scripts/`

| 模块 | 内容 | 状态 |
|---|---|---|
| `selfplay.py` | Agent 对局 JSONL 数据生成 | ✅ |
| `search_improve.py` | AlphaBeta Teacher 重标注，`--min-score-margin` 过滤 | ✅（待升级 soft target） |
| `arena.py` | 批量对战评测，随机 opening | ✅ |
| `compare_checkpoints.py` | 多 seed/opening 换边评测，quick/standard/full 晋级判断 | ✅ |
| `inspect_dataset.py` | JSONL 数据质量检查 | ✅ |


## 关键实验结论（避免走弯路）

- **search_improve 高置信过滤有效**：depth=2 + margin≥50 的 5000 样本微调，quick 评测 decisive_wr=0.553。这是唯一有正面信号的路线。
- **无过滤 search_improve 无效**：直接全量标注会导致策略漂移，被 quick reject。
- **混合训练无效**：高置信 + 全量数据简单混合会拉坏策略（wr=0.417）。
- **纯 value_head 不可用**：不能单独指导落子，也不应用 value-only 微调（破坏共享 encoder）。
- **`AlphaBetaAgent` depth=3 明显强于 depth=2**（约 30:7 对局），但速度慢 5 倍以上——这正是蒸馏要解决的问题。

## Agent 阵容

| 名称 | 类 | 说明 |
|---|---|---|
| `random` | RandomAgent | 随机基线 |
| `material` | MaterialAgent | 材料评估基线 |
| `alphabeta` | AlphaBetaAgent | Alpha-Beta 搜索，深度通过 YAML config 调整（默认 depth=2，`alphabeta_plus.yaml` 为 depth=3） |
| `supervised` | SupervisedAgent | 加载 `CandidateMoveNet` checkpoint，greedy 推理 |

## 棋力层级

```
AlphaBeta (depth=3) > AlphaBeta (depth=2) > SupervisedAgent > MaterialAgent > RandomAgent

SupervisedAgent vs AlphaBeta(depth=2): 约 6% 胜率 — Phase 1 需大幅提升
```

## 测试

```bash
make test              # C 引擎
make interface-test    # C↔Python 接口
make draughts-test     # 核心库
make trainer-test      # 训练器
make scripts-test      # 脚本
make all-python-test   # 全部
```

## 当前待办

按照 ROADMAP Phase 1 路线：

1. **`search_improve.py` 升级为 soft target**：输出候选动作评分分布，而非单一 best move
2. **`train_supervised.py` 加入 KL loss**：支持 soft label 训练
3. **用 alphabeta depth=3 生成大规模训练数据**，替代 depth=2 教师
4. **训练 → compare_checkpoints 评测 → 晋级或回退** 闭环

## 未完成事项（非紧急）

- 和棋规则未实现（三次重复、40 步规则），当前靠 `max_plies` 截断
- streaming dataset（当前全量读入内存）
- MCTS 实现（Phase 2）
- 训练曲线可视化
