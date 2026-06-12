# BitMind-Draught100

100 格国际跳棋 AI 程序，目标是通过神经网络引导搜索，构建超越传统 AlphaBeta 搜索方法的博弈智能体。

## 文档导航

| 文档 | 内容 |
|---|---|
| [ROADMAP.md](ROADMAP.md) | **技术路线图**——三阶段路线，所有开发的指南 |
| [SETUP.md](SETUP.md) | 环境配置指南（conda/venv/pip） |
| [SPEC.md](SPEC.md) | 国际跳棋规则规格 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 项目工程架构 |
| [PROGRESS.md](PROGRESS.md) | 阶段进度与实验记录 |
| [engine/README-engine.md](engine/README-engine.md) | C 规则引擎参考 |
| [interface/README-interface.md](interface/README-interface.md) | C↔Python 接口参考 |
| [draughts/README.md](draughts/README.md) | 核心 Python 库参考 |
| [trainer/README.md](trainer/README.md) | 训练器参考 |
| [scripts/README.md](scripts/README.md) | 独立应用脚本参考 |
| [gui/README-gui.md](gui/README-gui.md) | GUI 使用说明 |

## 快速开始

```bash
# 安装依赖（只需 torch）
pip install -r requirements.txt

# 编译 C 引擎
make interface

# 运行 GUI（人类/AI 对弈）
PYTHONPATH=interface:. python3 gui/simple_gui.py

# 运行测试
make test              # C 引擎测试
make interface-test    # 接口测试
make draughts-test     # 核心库测试
make trainer-test      # 训练器测试
make scripts-test      # 脚本测试
```

## 训练数据管线

从零开始生成数据并训练模型的完整流程：

```bash
# Step 1: 生成自博弈对局数据
# AlphaBeta depth=2 自博弈 + 开局随机 6 步，快速产生大量多样化合理局面
PYTHONPATH=interface:. python3 scripts/selfplay.py \
  --black alphabeta --white alphabeta \
  --games 500 --opening-random-plies 6 \
  --output data/selfplay_ab2.jsonl

# Step 2: 用更强的 Teacher 重标注 + 生成软标签
# AlphaBeta depth=3 对每个局面搜索打分，输出 policy_target_probs
PYTHONPATH=interface:. python3 scripts/search_improve.py data/ \
  --output data/improved_soft.jsonl \
  --alphabeta-depth 3 --temperature 100 --min-score-margin 20

# Step 3: 监督训练
PYTHONPATH=interface:. python3 trainer/train_supervised.py data/improved_soft.jsonl \
  --device cpu

# Step 4: 评测新 checkpoint
PYTHONPATH=interface:. python3 scripts/compare_checkpoints.py \
  --candidate trainer/checkpoints/candidate_move_net_best.pt \
  --baseline trainer/checkpoints/candidate_move_net_best.pt \
  --preset quick --device cpu
```

> **数据管线原理**：selfplay 用较快的 depth=2 生成多样化合理局面（"出题"），search_improve 用更强的 depth=3 给每个局面打高质量标签（"标准答案"），训练时将 AlphaBeta 分数通过温度 softmax 转为概率分布（软标签），比单一走法标签包含更丰富的信息。详见 [PROGRESS.md](PROGRESS.md)。
