# 脚本手册

`scripts/` 提供数据生成、重评分、批量对战和数据检查工具。所有示例在项目根目录执行，数据与模型路径需替换为实际文件。

比赛界面使用见 [比赛操作手册](../docs/competition.md)；并行运行见 [批量任务手册](../run/README-run.md)。

## 对局数据生成

`selfplay.py` 使用两个智能体对局，将局面及训练标签写入 JSONL：

```bash
PYTHONPATH=interface:. python3 scripts/selfplay.py \
  --black alphabeta --white alphabeta --alphabeta-depth 3 \
  --games 50 --opening-random-plies 6 \
  --max-plies 300 --output data/ab.jsonl --seed 4040
```

MCTS 自博弈示例：

```bash
PYTHONPATH=interface:. python3 scripts/selfplay.py \
  --black mcts --white mcts \
  --black-checkpoint path/to/model.pt --white-checkpoint path/to/model.pt \
  --mcts-config draughts/config/mcts_selfplay.yaml \
  --device cpu --games 50 --output data/mcts.jsonl
```

MCTS 搜索访问次数分布自动写入 `policy_target_probs`，可直接用于软标签训练。`--append` 控制追加，`--auto-output-name` 控制目标已存在时自动生成新名称；对应的 `--no-append` 和 `--no-auto-output-name` 显式关闭功能。

默认应用配置见 [selfplay.yaml](config/selfplay.yaml)。JSONL 是训练数据格式，与 GUI 导出的比赛 TXT 棋谱用途不同。

## 搜索重评分

`search_improve.py` 用 AlphaBeta 对已有 JSONL 局面重新评分，生成策略软标签：

```bash
PYTHONPATH=interface:. python3 scripts/search_improve.py data/ab.jsonl \
  --output data/improved.jsonl --alphabeta-depth 3 \
  --temperature 100 --min-score-margin 50 --max-samples 5000 --seed 2026
```

`--temperature` 控制 AlphaBeta 分数转换成 softmax 标签时的温度。`--min-score-margin` 过滤最优与次优走法分差不足的局面，`--position-sample-rate` 控制输入采样比例。此温度与 MCTS 的访问次数采样温度不是同一处理步骤。

## 批量对战

`arena.py` 统计双方胜负和运行截断：

```bash
PYTHONPATH=interface:. python3 scripts/arena.py \
  --black mcts --white alphabeta \
  --black-checkpoint path/to/model.pt --alphabeta-depth 6 \
  --mcts-config draughts/config/mcts.yaml \
  --games 20 --opening-random-plies 8 --device cpu
```

比较模型时分别运行交换先后手的两组，并保留所用模型、配置和原始统计。默认应用配置见 [arena.yaml](config/arena.yaml)。历史胜率只能在相应搜索预算、开局和截断口径下解释。

## 常用对局参数

下表适用于 `selfplay.py` 与 `arena.py`；完整支持以各脚本 `--help` 为准。

| 参数 | 说明 |
|---|---|
| `--black` / `--white` | `random`、`alphabeta`、`supervised` 或 `mcts` |
| `--black-checkpoint` / `--white-checkpoint` | 对应监督模型或 MCTS 的 `.pt` 权重 |
| `--device` | `cpu`、`cuda` 或 `cuda:N`；不指定时自动选择 |
| `--config` | 应用默认参数配置，显式 CLI 参数覆盖对应值 |
| `--games` / `--max-plies` | 局数与单局最大半回合数 |
| `--opening-random-plies` | 使用随机合法走法的开局半回合数 |
| `--seed` / `--log-interval` | 随机种子与进度输出间隔 |
| `--alphabeta-config` / `--alphabeta-depth` | AlphaBeta 配置与深度覆盖 |
| `--mcts-config` | MCTS 搜索配置；自博弈默认探索配置，arena 默认评测配置 |
| `--fp16` / `--no-fp16` | 显式启用或关闭 FP16 推理 |
| `--compile` / `--no-compile` | 显式启用或关闭编译推理 |

`selfplay.py` 还支持 `--alphabeta-depth-black` 和 `--alphabeta-depth-white`，优先于统一的深度参数。

`max_plies` 按半回合计数，只用于限制批量任务运行长度。达到上限的对局在统计中单独记录，不能直接视为赛事和棋。

## MCTS 探索与动作选择

- 根节点噪声混合网络先验与 Dirichlet 分布，改变搜索探索的分支。
- 温度为 0 时选择访问次数最多的走法，并列时随机选择；温度大于 0 时对访问分布做温度缩放后采样，权重与 `visits ** (1 / temperature)` 成比例。
- `--opening-random-plies` 在正式使用智能体之前执行随机合法走法。

配置分别见 [mcts_selfplay.yaml](../draughts/config/mcts_selfplay.yaml) 与 [mcts.yaml](../draughts/config/mcts.yaml)。关闭根节点噪声不等于所有随机因素均被消除。

## 数据检查

`inspect_dataset.py` 检查 schema、字段及分布：

```bash
PYTHONPATH=interface:. python3 scripts/inspect_dataset.py data/train.jsonl
PYTHONPATH=interface:. python3 scripts/inspect_dataset.py --quick data/
PYTHONPATH=interface:. python3 scripts/inspect_dataset.py --max-samples 1000 data/
```

`--quick` 跳过较深的形状及一致性检查，不等同于完整验证。`--max-samples` 只检查限定数量，结论不代表整个数据集。

## 参数帮助与测试

```bash
PYTHONPATH=interface:. python3 scripts/selfplay.py --help
PYTHONPATH=interface:. python3 scripts/search_improve.py --help
PYTHONPATH=interface:. python3 scripts/arena.py --help
PYTHONPATH=interface:. python3 scripts/inspect_dataset.py --help
make scripts-test
```
