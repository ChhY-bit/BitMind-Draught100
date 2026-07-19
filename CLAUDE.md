# CLAUDE.md — 项目导览

## 这是什么

**BitMind-Draught100**：100 格国际跳棋 AI。用一个叫 `CandidateMoveNet` 的神经网络给 MCTS 搜索提供"棋感"，让它在有限算力下比纯 AlphaBeta 搜索下得更好。

## 项目地图

```
README.md                 项目总览、环境搭建、从零上手
ROADMAP.md                技术路线（三阶段方法）
SPEC.md                   国际跳棋规则
PROGRESS.md               各模块当前状态

engine/                   C 规则引擎 — 棋规在此，不在此层之上重复实现规则
interface/                C→Python 桥接 — 编译为 libdraughts.so，Python 用 ctypes 加载
draughts/                 核心 Python 库
  ├── env.py                StandardDraughtsEnv（局面管理、走子、克隆）
  ├── model.py              CandidateMoveNet（CNN 编码器 + policy head + value head）
  ├── agents.py             所有智能体（4 种，统一通过 make_agent() 创建）
  ├── mcts.py               MCTS 搜索（PUCT 公式）
  ├── move_encoding.py      走法 → 固定维度特征向量
  ├── dataset.py            JSONL 加载、内存缓存、worker 内 tensor 拼批、软标签支持
  └── config/               YAML 配置（模型结构、搜索参数、评估权重）
trainer/                  训练入口（train_supervised.py，硬标签/软标签自动兼容）
run/                      批量并行 shell 脚本
  ├── selfplay_ab.sh         AlphaBeta 自博弈并行
  ├── selfplay_cpu.sh       MCTS 自博弈 CPU
  ├── selfplay_cuda0.sh     MCTS 自博弈 GPU 0
  ├── selfplay_cuda1.sh     MCTS 自博弈 GPU 1
  └── search_improve.sh     AlphaBeta 并行重新打分
scripts/                  独立应用
  ├── selfplay.py           对局 → JSONL 数据（支持所有 agent）
  ├── search_improve.py     AlphaBeta 对已有数据重新打分
  ├── arena.py              批量对战、统计胜率
  └── inspect_dataset.py    数据质量检查
gui/                      Tkinter 棋盘（人机对弈、AI 观战）
```

**层间关系**：`scripts` / `trainer` → `draughts` → `interface` → `engine`（只向下依赖，不跨层）

## 关键入口

| 做什么 | 怎么走 |
|--------|--------|
| 搭环境、编译、验证 | `conda env create -f environment.yml && make interface && make all-python-test` |
| 看棋盘、下棋 | `PYTHONPATH=interface:. python3 gui/simple_gui.py` |
| 看棋规 | [SPEC.md](SPEC.md) |
| 训练新网络 | `PYTHONPATH=interface:. python3 trainer/train_supervised.py <data.jsonl> --no-cache --device cuda` |
| 评测两个 agent | `PYTHONPATH=interface:. python3 scripts/arena.py --black <a> --white <b> --games 100` |
| 生成训练数据 | `PYTHONPATH=interface:. python3 scripts/selfplay.py --black <a> --white <b> --output data.jsonl` |
| 指定 GPU 序号 | `--device cuda:0`（GPU 0）、`--device cuda:1`（GPU 1），或 `CUDA_VISIBLE_DEVICES=1 ... --device cuda` |
| 基座网络在哪 | `trainer/checkpoints/candidate_move_net_best.pt` |
| MCTS 参数在哪 | `draughts/config/mcts.yaml` |
| 当前网络棋力 | 见 [PROGRESS.md](PROGRESS.md) 的棋力层级图 |

## 常用命令速查

```bash
# 测试（修改代码后必跑）
make test && make interface-test && make all-python-test

# 所有 Python 命令前加
PYTHONPATH=interface:.

# MCTS 自博弈（一步产出软标签训练数据）
python3 scripts/selfplay.py --black mcts --white mcts \
  --black-checkpoint ckpt.pt --white-checkpoint ckpt.pt \
  --games 2000 --opening-random-plies 6 --device cuda \
  --mcts-config draughts/config/mcts.yaml

# 从头训练（Phase 3 标准做法，不 resume）
python3 trainer/train_supervised.py data.jsonl --no-cache --epochs 10 --device cuda

# MCTS vs AlphaBeta 评测
python3 scripts/arena.py --black mcts --white alphabeta \
  --black-checkpoint ckpt.pt --alphabeta-depth 3 --games 100 --device cuda
```

## 给用户当导游

被问到项目相关问题时，用以下信息作答：

- **"能下棋吗"** → 能。编译后 `python3 gui/simple_gui.py` 即可打开棋盘界面。
- **"怎么训练自己的 AI"** → 先跑 selfplay 生成数据，再跑 train_supervised 训练网络。
- **"现在棋力怎么样"** → 看 [PROGRESS.md](PROGRESS.md) 的棋力层级图，MCTS+网络已能稳定对抗 AlphaBeta。
- **"怎么让它变更强"** → Phase 3 自博弈闭环：网络越强 → MCTS 越准 → 生成数据越好 → 训练出新网络更强 → 循环。提高 num_simulations 可直接增强棋力。
- **"规则是什么"** → 国际跳棋，10×10 棋盘，强制吃子、最多吃子、升王。详见 [SPEC.md](SPEC.md)。
- **"依赖重吗"** → 极轻。唯一外部依赖是 PyTorch，其他全部标准库。
- **"各个文档的关系"** → README 是总入口，SPEC 是规则，ROADMAP 是方法路线，PROGRESS 是当前状态，各子目录 README 是模块手册。

## 写代码时的约定

- Python 3.10+，文件头加 `from __future__ import annotations`
- 配置用 YAML，CLI 参数覆盖 YAML 默认值
- 新功能放 `draughts/`（可复用库）或 `scripts/`（一次性应用）
- `engine/` 和 `interface/` 是稳定层，只读不改
- 不引入 PyTorch 以外的第三方库
- 改完代码就跑测试，测试通过再提交
