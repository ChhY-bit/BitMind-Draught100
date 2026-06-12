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
| `train_supervised.py` | 监督/蒸馏训练，checkpoint 管理，CSV 日志 | ✅ |

### 应用脚本 `scripts/`

| 模块 | 内容 | 状态 |
|---|---|---|
| `selfplay.py` | Agent 对局 JSONL 数据生成 | ✅ |
| `search_improve.py` | AlphaBeta Teacher 重标注，`--min-score-margin` 过滤，软标签输出 | ✅ |
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

1. ~~**`search_improve.py` 升级为 soft target**~~ ✅ 已完成（2026-06-12）
2. ~~**`train_supervised.py` 加入 KL loss**~~ ✅ 已完成（2026-06-12）
3. **用 alphabeta depth=3 生成大规模训练数据**，替代 depth=2 教师
4. **训练 → compare_checkpoints 评测 → 晋级或回退** 闭环

### Soft Target 升级详情（2026-06-12）

`search_improve.py` 现在输出 `policy_target_probs`（温度 softmax 概率分布），训练自动检测并使用软标签：

- **`search_improve.py`**：新增 `--temperature` 参数（默认 100），将 AlphaBeta 分数通过温度缩放 softmax 转换为概率分布，写入 `policy_target_probs` 字段（与 `legal_moves` 平行）。`teacher_temperature` 记录在样本中。
- **`dataset.py`**：`training_item_from_sample` 提取 `policy_target_probs`；`collate_batch` 和 `pad_probs` 处理 padding；`to_torch_batch` 转为张量。
- **`train_supervised.py`**：新增 `soft_cross_entropy()` 函数，当 batch 中存在 `policy_target_probs` 时自动使用软交叉熵 loss；无软标签时自动回退到硬标签 one-hot CE。完全向后兼容。
- **模型未修改**：`CandidateMoveNet` 无需改动，输出 `policy_logits` 天然支持软标签训练。

## 数据管线（新用户从此开始）

从零到训练模型的完整流程，详见 [README.md](README.md) 的「训练数据管线」章节。

**三步概述**：

```
selfplay (depth=2) → search_improve (depth=3, 软标签) → train_supervised
     出题                 标准答案                          学生学习
```

**为什么 selfplay 用 depth=2 而 improve 用 depth=3？**

| | selfplay | search_improve |
|---|---|---|
| **角色** | 生成多样化合理局面（"出题"） | 给每个局面打最高质量标签（"标准答案"） |
| **深度** | depth=2 | depth=3 |
| **速度** | 快（depth=3 的 5 倍+） | 慢，但只跑一次 |
| **要求** | 走法不能太离谱即可 | 越准确越好 |

depth=2 自博弈已经能产生人类可理解的对局进展；而 search_improve 用 depth=3 Teacher 搜索重新评估每个合法走法，通过温度 softmax 转为概率分布。这样在有限算力下最大化数据质量和数量。

## 未完成事项（非紧急）

- 和棋规则未实现（三次重复、40 步规则），当前靠 `max_plies` 截断
- streaming dataset（当前全量读入内存）
- MCTS 实现（Phase 2）
- 训练曲线可视化
