# 技术路线图

本文档定义 BitMind-Draught100 项目的核心技术路线。所有开发工作围绕此路线开展。

## 最终目标

构建一个**网络引导搜索的智能体**，棋力超越同等计算预算下的纯 AlphaBeta 搜索，接近甚至超过顶尖人类棋手水平。

## 三阶段路线

### Phase 1：蒸馏模仿（当前阶段）

**目标**：训练神经网络基座，使其棋力接近 AlphaBeta depth=3。

**方法**：

- 用 `alphabeta_plus`（depth=3 或更深）作为 Teacher，为大量局面生成训练标签
- 输出 **soft policy target**：不是单一 best move，而是所有候选动作的评分分布
- 输出 **value target**：搜索后从当前行棋方视角的局面价值
- 使用 `min_score_margin` 过滤低置信度标签
- 训练 `CandidateMoveNet`：KL 散度逼近 Teacher 的 policy 分布，MSE 逼近 value
- 用 `compare_checkpoints.py quick → standard → alphabeta` 评测，decisive_wr ≥ 0.53 晋级

**关键指标**：网络对 AlphaBeta depth=3 胜率 > 40%。

**技术依赖**：

| 模块 | 用途 |
|---|---|
| `draughts/agents.py` | AlphaBetaPlusAgent 作为 Teacher |
| `scripts/search_improve.py` | 搜索生成训练标签（需升级为 soft target） |
| `trainer/train_supervised.py` | 监督/蒸馏训练（需加 KL loss） |
| `scripts/compare_checkpoints.py` | 晋级评测 |

---

### Phase 2：搜索增强（网络 + MCTS）

**目标**：用网络引导 MCTS，推理棋力超越纯 AlphaBeta。

**方法**：

- 实现 MCTS，节点评估使用 Phase 1 训练好的网络：
  - 分支选择：PUCT 公式，prior 来自 `policy_logits`
  - 叶子节点评估：`value_head` 输出局面价值
  - 搜索结束时返回 visit count 分布（新的 policy target）和最终 value
- 与 AlphaBeta depth=3 对局评测
- 仍可用 AlphaBeta 辅助生成训练数据

**关键指标**：MCTS + 网络稳定战胜 AlphaBeta depth=3。

**技术依赖**：

| 模块 | 用途 |
|---|---|
| `draughts/model.py` | 网络需同时输出 policy_logits + value（已满足） |
| `draughts/env.py` | `clone()` 性能可能成为 MCTS 瓶颈 |
| 新模块 `draughts/mcts.py` | MCTS 搜索实现 |

---

### Phase 3：自我超越（自博弈强化学习）

**目标**：网络自博弈生成数据、自我训练、螺旋上升，棋力超越任何固定深度搜索。

**方法**：

- 网络 + MCTS 自博弈，生成高质量对局数据
- MCTS visit count 分布作为 policy target，对局胜负作为 value target
- 训练新网络 → 更强的 MCTS → 更强的自博弈数据 → 循环
- 不再依赖 AlphaBeta 作为教师

**关键指标**：每一轮迭代后，新一代网络对上一代胜率 > 55%。

**技术依赖**：Phase 1 + 2 的全部基础设施，加上多 GPU 训练和自动化迭代管线。

---

## 架构支撑

三阶段路线对代码架构的核心要求：

```
            Phase 1        Phase 2        Phase 3

网络        ✅ policy+value  ✅ 同上        ✅ 更深/更宽
搜索        ❌ 纯AlphaBeta   ✅ MCTS+网络    ✅ MCTS+更强网络
Teacher     AlphaBeta       AlphaBeta+MCTS  上一轮自己
数据来源    搜索蒸馏         搜索+自博弈     纯自博弈
评测        胜率对比         胜率对比        胜率对比+Elo
```

所有阶段共享同一网络架构（`CandidateMoveNet`），只需持续改进训练方法和数据来源。

## 不再采用的路线

以下路线已被实验证明低效或失败，**不再作为开发方向**：

| 路线 | 失败原因 |
|---|---|
| 简化 PPO（`rl_selfplay.py` + `train_selfplay.py`） | 多轮迭代卡在 51%-52% 胜率，无法稳定晋级 |
| 纯 value_head 搜索（`ValueSearchAgent`） | 对 supervised 约 32:166，当前 value 无法引导落子 |
| value-only 全网络微调 | 更新共享 encoder 后破坏 policy，wr=0.398 |
| 无过滤搜索标签直接训练 | 低置信度标签导致策略漂移，被 reject |
| 简单混合高置信+全量数据 | 拉坏策略，wr=0.417 |

## 里程碑

| 里程碑 | 阶段 | 判断标准 |
|---|---|---|
| M1 | Phase 1 | 网络对 AlphaBeta depth=3 胜率 > 30% |
| M2 | Phase 1 | 网络对 AlphaBeta depth=3 胜率 > 40%，晋级 Phase 2 |
| M3 | Phase 2 | MCTS+网络 对 AlphaBeta depth=3 胜率 > 50% |
| M4 | Phase 2 | MCTS+网络 对 AlphaBeta depth=3 胜率 > 70%，晋级 Phase 3 |
| M5 | Phase 3 | 自博弈迭代 1 轮晋升成功 |
| M6 | Phase 3 | 自博弈迭代 3 轮以上，棋力持续上升 |
