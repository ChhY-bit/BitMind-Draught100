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
| `mcts.py` | MCTS + PUCT 搜索，支持批量叶子推理（虚拟损失），visit count 输出可作训练软标签 | ✅ |
| `move_encoding.py` | 走法 → 45 维特征向量 | ✅ |
| `dataset.py` | JSONL 加载、内存缓存、worker 内 tensor 拼批、软标签支持 | ✅ |

### 上层应用

| 模块 | 功能 |
|------|------|
| `trainer/train_supervised.py` | 监督训练，硬标签/软标签自动兼容 |
| `scripts/selfplay.py` | 对局数据生成，MCTS agent 直接产出软标签 |
| `scripts/search_improve.py` | AlphaBeta 对已有数据重新打分 |
| `scripts/arena.py` | 批量对战评测 |
| `gui/gui.py` | Tkinter 棋盘界面，人机/机机对弈 |

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

## 网络版本

| 版本 | 参数 | 架构 | 状态 |
|------|------|------|------|
| v1 | ~1.8M（conv=64, board=256, hidden=128） | 3 层 conv + Flatten + 大 FC | 已退役（round1–7） |
| v2 | ~15M（conv=256, board=512, hidden=256） | 3 层 conv + Flatten + 大 FC | 当前使用 |
| v3 | ~2.9M（conv=256×5, board=512, BN, 全局池化） | 5 层 Conv→BN→ReLU + AdaptiveAvgPool2d + 小 FC | 架构就绪，待训练验证 |
| 配置文件 | `draughts/config/model.yaml` | — | — |

### v3 架构要点

v3 的核心改进是用**深卷积 + 全局平均池化**替代 v2 的 Flatten + 大 FC（13.1M → 131K 参数）：

- **5 层 3×3 卷积**：感受野 11×11，覆盖全部 10×10 棋盘（v2：3 层，感受野 7×7，仅覆盖 49%）
- **BatchNorm**：每层 Conv→BN→ReLU，保证 5 层深度下梯度畅通；推理时与 Conv 融合（零开销）
- **AdaptiveAvgPool2d**：汇总 100 个全局视角为固定向量，参数减少 80%，正则化更强
- **向后兼容**：`use_global_pool: false`（默认）走 v2 路径，旧 checkpoint 无需修改；设为 `true` 即切换 v3

v2 推荐训练数据量 **3M–5M 样本**，v3 推荐 **2M–4M 样本**（参数更少、BN 正则化 → 数据需求更低）。checkpoint 内保存 `model_config`，v1/v2/v3 可同台对战，旧 checkpoint 无需重新训练即可在 GUI/arena 中加载。

## 已训练网络

| 文件 | 来源 | policy accuracy | value MAE |
|------|------|-----------------|-----------|
| `trainer/checkpoints/candidate_move_net_best.pt` | Phase 1 蒸馏（~35,000 样本, v1） | 75% (val) | 0.232 (val) |

## 已知局限

- 和棋规则（三次重复、40 步）未实现，靠 max_plies 截断
- 数据集默认全量载入内存，大数据集务必使用 `--no-cache` 避免 OOM
- `env.clone()` 仍有优化空间（move/undo 替代 clone 方案待实现）
- 走法编码仍在 Python 层，C 层向量化可进一步提升训练和推理速度
- 当前仅支持单 GPU 训练
- v2 网络 15M 参数，单卡训练 GPU 显存需 ≥4 GB
