# 项目现状

## 总体状态

三大阶段中，前两个已完成，第三个的框架已就绪、待规模化运行。

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 — 蒸馏模仿 | AlphaBeta 当老师，训练神经网络学会"棋感" | 完成 |
| Phase 2 — 搜索增强 | MCTS + 网络引导搜索，棋力超越纯 greedy | 完成 |
| Phase 3 — 自博弈闭环 | 网络自己产生数据、自己训练、螺旋上升 | 框架就绪，待跑 |

## 模块现状

### 底层（稳定，不修改）

| 模块 | 功能 | 测试 |
|------|------|------|
| `engine/` | C 规则引擎，16 个公开 API，棋规唯一真源 | 全覆盖 + 随机压力 |
| `interface/` | C↔Python ctypes 桥接，7 个导出函数 | 5 项 |

### 核心库 `draughts/`

| 模块 | 功能 | 测试 |
|------|------|------|
| `env.py` | 训练环境，6×10×10 当前方视角编码，局面克隆 | ✅ |
| `model.py` | CandidateMoveNet：CNN encoder + policy head + value head | ✅ |
| `agents.py` | 4 个智能体（见下方阵容表），统一工厂函数 | ✅ |
| `mcts.py` | MCTS + PUCT 搜索，visit count 输出可作训练软标签 | ✅ |
| `move_encoding.py` | 走法 → 45 维特征向量 | ✅ |
| `dataset.py` | JSONL 加载、内存缓存、worker 内 tensor 拼批、软标签支持 | ✅ |

### 上层应用

| 模块 | 功能 |
|------|------|
| `trainer/train_supervised.py` | 监督训练，硬标签/软标签自动兼容 |
| `scripts/selfplay.py` | 对局数据生成，MCTS agent 直接产出软标签 |
| `scripts/search_improve.py` | AlphaBeta 对已有数据重新打分 |
| `scripts/arena.py` | 批量对战评测 |
| `gui/simple_gui.py` | Tkinter 棋盘界面，人机/机机对弈 |

## 智能体阵容

| 名称 | 类型 | 搜索？ | 网络？ | 棋力定位 |
|------|------|--------|--------|---------|
| `random` | 随机 | — | — | 底层基线 |
| `alphabeta` | AlphaBeta | 可配置深度（`--alphabeta-depth`） | — | 手工基线 |
| `supervised` | 网络 greedy | — | CandidateMoveNet | 有棋感无前瞻 |
| `mcts` | MCTS + 网络 | 80-3200 sims | CandidateMoveNet | 当前最强 |

## 棋力层级

以下为无开局随机化条件下的实测结果：

```
alphabeta (depth=3+)        ← 手工基线（深度可配置）
    |
mcts (训练网络, 800sims)    ← 约 90% 胜 alphabeta (depth=2)
    |                         100% 胜 supervised (greedy)
    |
mcts (随机网络)             ← 约 17% 胜 alphabeta (depth=2)
    |
supervised (greedy)         ← 0% 胜 alphabeta (depth=2)
    |
random                     ← 基线
```

核心发现：同一网络，greedy 对 AB(d=2) 0% → MCTS(800sims) 对 AB(d=2) ≈90%。搜索是棋力质变的关键。

## 训练数据管线

```
Phase 1 路线:  selfplay(AB) → search_improve(AB d=3) → train    （两步）
Phase 3 路线:  selfplay(MCTS) → train                            （一步）
```

MCTS 自博弈时，搜索返回的 visit count 分布自动写入 JSONL 的 `policy_target_probs` 字段，训练器自动识别并切换为软交叉熵损失。

## 已训练网络

| 文件 | 来源 | policy accuracy | value MAE |
|------|------|-----------------|-----------|
| `trainer/checkpoints/candidate_move_net_best.pt` | Phase 1 蒸馏（~35,000 样本） | 75% (val) | 0.232 (val) |

## 已知局限

- 和棋规则（三次重复、40 步）未实现，靠 max_plies 截断
- 数据集默认全量载入内存，大数据集务必使用 `--no-cache` 避免 OOM
- `env.clone()` 仍有优化空间（move/undo 替代 clone 方案待实现）
- 走法编码仍在 Python 层，C 层向量化可进一步提升训练和推理速度
- 当前仅支持单 GPU 训练
