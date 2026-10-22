> 历史归档：2026-10-21 从原 `ROADMAP.md` 保存。早期三阶段研究路线，包含当时的目标、实验结论和待办。文中“当前”“待实现”等表述仅代表原记录语境，不作为现行功能或赛前任务依据。

# 技术路线图

本文档定义 BitMind-Draught100 项目的核心技术路线。所有开发工作围绕此路线开展。

## 最终目标

构建一个**网络引导搜索的智能体**，棋力超越同等计算预算下的纯 AlphaBeta 搜索，接近甚至超过顶尖人类棋手水平。

## 三阶段路线

### Phase 1：蒸馏模仿 ✅ 已完成（2026-06-15）

**目标**：训练神经网络基座，使其具备棋局"棋感"（policy + value）。

**完成情况**：

- 用 `alphabeta`（depth=3）作为 Teacher，为 600 局 selfplay 产生的 ~138,000 个局面生成训练标签
- 实现 **soft policy target**：温度 softmax 将 AlphaBeta 分数转为候选动作概率分布
- 训练 `CandidateMoveNet`：软交叉熵逼近 Teacher 的 policy 分布，MSE 逼近 value
- 使用 `min_score_margin=20` 过滤低置信度标签，保留率约 25%（~35,000 样本）
- 训练结果：policy accuracy 75%（val）~86%（train），value MAE 0.10

**棋力水平**：

| 对手 | 胜率 | 说明 |
|------|------|------|
| random | 85% (17:3) | 基础棋感已建立 |
| alphabeta (depth=2) | 0% (0:100) | greedy 推理无搜索，符合预期 |
| alphabeta (depth=3) | 未测 | 留待 Phase 2 |

**关键结论**：基座网络已具备棋感（会选吃子、value 能判断局面优劣），差的只是前瞻能力。这正是 Phase 2 要补齐的。

---

### Phase 2：搜索增强（网络 + MCTS）✅ 已完成

**目标**：用网络引导 MCTS，推理棋力超越纯 AlphaBeta。

**方法**：

- 实现 MCTS，节点评估使用 Phase 1 训练好的网络：
  - 分支选择：PUCT 公式，prior 来自 `policy_logits`
  - 叶子节点评估：`value_head` 输出局面价值
  - 搜索结束时返回 visit count 分布（新的 policy target）和最终 value
- 与 AlphaBeta depth=2 / depth=3 对局评测
- 仍可用 AlphaBeta 辅助生成训练数据

**关键指标**：MCTS + 网络稳定战胜 AlphaBeta depth=2 → depth=3。

**技术依赖**：

| 模块 | 用途 | 状态 |
|------|------|------|
| `draughts/model.py` | 网络需同时输出 policy_logits + value | ✅ 已满足 |
| `draughts/env.py` | `clone()` 性能可能成为 MCTS 瓶颈 | move/undo 方案待优化 |
| `trainer/checkpoints/candidate_move_net_best.pt` | Phase 1 训练的基座网络 | ✅ 可用 |
| `draughts/mcts.py` | MCTS 搜索 | ✅ 已实现 |
| MCTS 自博弈软标签 | `selfplay.py` 直接写入 `policy_target_probs` | ✅ 已实现 |

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
搜索        ❌ 纯AlphaBeta   ✅ MCTS+网络    ⏳ MCTS+更强网络（待规模化）
Teacher     AlphaBeta       AlphaBeta+MCTS  上一轮自己
数据来源    搜索蒸馏         搜索+自博弈     纯自博弈
评测        胜率对比         胜率对比        胜率对比+Elo
```

所有阶段共享同一网络架构（`CandidateMoveNet`），只需持续改进训练方法和数据来源。

## 不再采用的路线

以下路线已被实验证明低效或失败，相关代码已从仓库中清理：

| 路线 | 失败原因 |
|------|---------|
| 简化 PPO（`rl_selfplay.py` + `train_selfplay.py`） | 多轮迭代卡在 51%-52% 胜率，无法稳定晋级 |
| 纯 value_head 搜索（`ValueSearchAgent`） | 对 supervised 约 32:166，当前 value 无法引导落子 |
| value-only 全网络微调 | 更新共享 encoder 后破坏 policy，wr=0.398 |
| 无过滤搜索标签直接训练 | 低置信度标签导致策略漂移，被 reject |
| 简单混合高置信+全量数据 | 拉坏策略，wr=0.417 |

## 里程碑

| 里程碑 | 阶段 | 判断标准 | 状态 |
|--------|------|---------|------|
| M1 | Phase 1 | 网络对 AlphaBeta depth=3 胜率 > 30% | —（跳过，greedy 推理难达） |
| M2 | Phase 1 | 训练出可用基座网络，进入 Phase 2 | ✅ 已完成 |
| M3 | Phase 2 | MCTS+网络 对 AlphaBeta depth=2 胜率 > 50% | ✅ 已达成（高 sims ≈ 90% 胜率） |
| M4 | Phase 2 | MCTS+网络 对 AlphaBeta depth=3 胜率 > 50% | 待推进（高 sims 可提升棋力） |
| M5 | Phase 3 | 自博弈迭代 1 轮晋升成功 | 框架就绪 |
| M6 | Phase 3 | 自博弈迭代 3 轮以上，棋力持续上升 | — |
