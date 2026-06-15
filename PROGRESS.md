# 项目进度

本文档用于新会话快速恢复上下文。详细棋规见 `SPEC.md`，架构见 `ARCHITECTURE.md`，路线见 `ROADMAP.md`，Agent 约束见 `CLAUDE.md`。

## 当前阶段：Phase 2 搜索增强

Phase 1 已完成。基座网络 `CandidateMoveNet` 已训练完毕，具备棋感但缺前瞻。当前任务：实现 MCTS，用网络引导搜索，超越 AlphaBeta。

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
| `agents.py` | 5 个 Agent：`random`、`material`、`alphabeta`、`alphabeta_plus`、`supervised` | ✅ |
| `dataset.py` | JSONL 读取、batch 组装、软标签支持、数据源权重 | ✅ |

### 训练 `trainer/`

| 模块 | 内容 | 状态 |
|---|---|---|
| `train_supervised.py` | 监督/蒸馏训练，软标签自动检测，checkpoint 管理，CSV 日志 | ✅ |

### 应用脚本 `scripts/`

| 模块 | 内容 | 状态 |
|---|---|---|
| `selfplay.py` | Agent 对局数据生成，`--opening-random-plies` 开局随机化 | ✅ |
| `search_improve.py` | AlphaBeta Teacher 重标注，`--temperature` 软标签输出，`--min-score-margin` 过滤 | ✅ |
| `arena.py` | 批量对战评测，随机 opening | ✅ |
| `compare_checkpoints.py` | 多 seed/opening 换边评测，quick/standard/full/alphabeta 晋级判断 | ✅ |
| `inspect_dataset.py` | JSONL 数据质量检查 | ✅ |

## Phase 1 训练结果（2026-06-15）

| 指标 | 值 |
|------|-----|
| 数据集 | 600 局 selfplay → search_improve(depth=3) → ~35,000 软标签样本 |
| 训练 epochs | 21（best val 在 epoch 13） |
| Train policy accuracy | 75.4%（best）→ 85.7%（final） |
| Train value MAE | 0.097（final） |
| Val policy accuracy | 59.1%（best） |
| Val value MAE | 0.232（best） |
| Checkpoint | `trainer/checkpoints/candidate_move_net_best.pt` |

### 棋力层级（实测）

```
AlphaBeta (depth=3) >> AlphaBeta (depth=2) >> SupervisedAgent > MaterialAgent >> RandomAgent

SupervisedAgent vs RandomAgent:    85% 胜率 (17:3)
SupervisedAgent vs MaterialAgent:  70% 胜率 (70:30)
SupervisedAgent vs AlphaBeta(d2):   0% 胜率 (0:100) ← greedy 推理无搜索，符合预期
```

## 关键实验结论（避免走弯路）

- **软标签 + 高置信过滤有效**：temperature=100 + min_score_margin=20，保留率 ~25%，训练收敛良好
- **Phase 1 网络已具备棋感**：会选吃子走法、value 判断方向正确，但 greedy 推理无法击败任何有搜索的对手
- **下一步是加搜索，不是加数据**：86% policy accuracy 意味着继续扩大 Phase 1 数据边际收益很低
- **纯 value_head 不可用**：不能单独指导落子，也不应用 value-only 微调（破坏共享 encoder）
- **AlphaBeta depth=3 明显强于 depth=2**：但速度慢 5 倍以上——这正是 MCTS 要解决的问题

## Agent 阵容

| 名称 | 类 | 说明 |
|---|---|---|
| `random` | RandomAgent | 随机基线，仅开局随机化使用 |
| `material` | MaterialAgent | 材料+位置启发式评估，一步前瞻 |
| `alphabeta` | AlphaBetaAgent | Alpha-Beta 搜索，depth 通过 config 调整（默认 depth=2） |
| `alphabeta_plus` | AlphaBetaAgent | 使用 `alphabeta_plus.yaml`（depth=3），当前最强 Teacher |
| `supervised` | SupervisedAgent | 加载 `CandidateMoveNet` checkpoint，greedy 推理 |

## 测试

```bash
make test              # C 引擎
make interface-test    # C↔Python 接口
make draughts-test     # 核心库（39 tests）
make trainer-test      # 训练器（13 tests）
make scripts-test      # 脚本（22 tests + 1 预存失败）
make all-python-test   # 全部
```

## Phase 2 待办

1. **实现 `draughts/mcts.py`**：MCTS + PUCT + 网络引导
2. **实现 `MCTSAgent`**：注册到 `agents.py` 和 `make_agent`
3. **评测 MCTS+网络 vs AlphaBeta depth=2**：首个小目标——站稳 50%+
4. **评测 MCTS+网络 vs AlphaBeta depth=3**：Phase 2 核心指标
5. **MCTS 性能优化**：`env.clone()` 瓶颈评估与优化

## 未完成事项（非紧急）

- 和棋规则未实现（三次重复、40 步规则），当前靠 `max_plies` 截断
- streaming dataset（当前全量读入内存）
- 训练曲线可视化
- GPU 驱动修复（RTX 5060 / CUDA 13.0，driver 已修复过一次，注意内核更新后可能复发）
