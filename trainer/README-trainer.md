# 训练环境说明

`trainer/` 目录当前包含标准化环境、基础 agent、数据生成/读取、候选动作神经网络、监督学习训练和第一版自博弈强化学习训练脚本。

## 当前文件

```text
trainer/
  env.py
  agents.py
  arena.py
  compare_checkpoints.py
  selfplay.py
  rl_selfplay.py
  inspect_dataset.py
  dataset.py
  move_encoding.py
  model.py
  train_supervised.py
  train_selfplay.py
  config/arena.yaml
  config/compare.yaml
  config/material.yaml
  config/alphabeta.yaml
  config/value_search.yaml
  config/selfplay.yaml
  config/rl_selfplay.yaml
  config/model.yaml
  config/supervised.yaml
  config/rl_train.yaml
  config/data_sources.yaml
  tests/
    test_env.py
    test_agents.py
    test_arena.py
    test_compare_checkpoints.py
    test_selfplay.py
    test_rl_selfplay.py
    test_inspect_dataset.py
    test_move_encoding.py
    test_dataset.py
    test_model.py
    test_train_supervised.py
    test_train_selfplay.py
  README-trainer.md
  README-network.md
```

- `env.py`：标准化训练环境 `StandardDraughtsEnv`。
- `agents.py`：基础 agent。
- `arena.py`：批量对战评测器。
- `compare_checkpoints.py`：自动 checkpoint 晋升评测器，支持多 seed、多 opening、换边和外部基线。
- `selfplay.py`：对局数据生成器，保存 JSONL 训练样本。
- `rl_selfplay.py`：训练专用 RL 自博弈生成器，按 temperature 从 checkpoint policy 中采样。
- `inspect_dataset.py`：JSONL 数据检查器。
- `dataset.py`：JSONL 数据读取器与 batch 组装工具。
- `move_encoding.py`：候选动作向量编码器。
- `model.py`：候选动作打分 policy/value 神经网络。
- `train_supervised.py`：监督学习训练脚本，读取 JSONL 数据训练 `CandidateMoveNet`。
- `train_selfplay.py`：第一版 Actor-Critic 风格 RL 训练脚本，读取 RL self-play JSONL 更新 candidate。
- `config/arena.yaml`：arena 批量评测与随机 opening 参数。
- `config/compare.yaml`：checkpoint 自动评测参数。
- `config/material.yaml`：`MaterialAgent` 的材料评估参数。
- `config/alphabeta.yaml`：`AlphaBetaAgent` 的搜索深度和增强评估参数。
- `config/value_search.yaml`：`ValueSearchAgent` 的浅层神经网络搜索参数。
- `config/selfplay.yaml`：selfplay 数据生成参数。
- `config/rl_selfplay.yaml`：RL 自博弈探索采样参数。
- `config/model.yaml`：`CandidateMoveNet` 的网络结构参数。
- `config/supervised.yaml`：监督学习训练参数。
- `config/rl_train.yaml`：RL self-play candidate 训练参数。
- `config/data_sources.yaml`：监督训练数据源权重配置示例。
- `tests/`：训练环境、agent、arena、自动 checkpoint 对比、selfplay、RL self-play、数据检查器、动作编码器、数据读取器、模型、监督训练和 RL 训练测试。
- `README-trainer.md`：本文档。
- `README-network.md`：神经网络组件与动作编码设计记录。

## 运行测试

在项目根目录执行：

```bash
make interface
PYTHONPATH=interface:trainer python3 -m unittest discover -s trainer/tests -p 'test_*.py'
```

如果 Makefile 已包含 `trainer-test` 目标，也可以执行：

```bash
make trainer-test
```

## 基本用法

```python
from env import StandardDraughtsEnv

env = StandardDraughtsEnv()

obs = env.reset()
mask = env.action_mask()
actions = env.legal_actions()

next_obs, reward, done, info = env.step(actions[0].action_id)
```

`step()` 返回四项：

```text
observation, reward, done, info
```

- `observation`：训练用局面编码。
- `reward`：刚刚行动一方视角的奖励，胜为 `1`，负为 `-1`，未结束或和棋为 `0`。
- `done`：是否终局。
- `info`：调试信息，包括终局状态、当前行棋方、合法动作数量等。

## Observation 编码

当前 observation 是纯 Python 嵌套列表，形状为：

```text
6 x 10 x 10
```

也就是：

```python
obs[channel][row][col]
```

通道含义：

| 通道 | 含义 |
| --- | --- |
| `0` | 当前行棋方普通子 |
| `1` | 当前行棋方王 |
| `2` | 对方普通子 |
| `3` | 对方王 |
| `4` | 合法棋位 |
| `5` | 保留平面，当前统一为 `0` |

说明：

- 当前使用“当前行棋方视角”做方向归一化。黑方行棋时保持原棋盘方向；白方行棋时按 180 度旋转，相当于把棋位 `square_id` 映射为 `51 - square_id`。
- 棋子通道使用“当前方/对方”视角，而不是固定黑白视角。
- 归一化后，当前方普通子在神经网络输入中总是朝同一个方向前进。
- 第 5 通道暂时不再提供黑/白身份信息，避免重新引入方向差异。

初始局面下，黑方先手：

```text
通道 0 有 20 个 1
通道 2 有 20 个 1
通道 4 有 50 个 1
通道 5 全为 0
```

黑方走一步后轮到白方：

```text
白方局面会被旋转到当前方视角
通道 5 仍全为 0
```

## Action 设计

当前 action 仍使用 C 接口给出的当前局面合法动作编号：

```text
action_id = 当前合法动作列表中的编号
```

获取合法动作：

```python
actions = env.legal_actions()
```

每个动作是 `Move`：

```python
move.action_id
move.path
move.captures
```

执行动作：

```python
obs, reward, done, info = env.step(move.action_id)
```

## Action Mask

调用：

```python
mask = env.action_mask()
```

返回长度为 `512` 的列表。

- `1` 表示该 action id 当前合法。
- `0` 表示该 action id 当前不可用。

当前 C 接口生成的动作编号是密集的，所以如果当前有 9 个合法动作：

```text
mask[0..8] = 1
mask[9..511] = 0
```

这个 action mask 适合后续模型或搜索代码过滤非法动作。

## 当前限制

- `dataset.py` 已提供可选 PyTorch batch 转换；当前项目仍不强制依赖 PyTorch。
- 当前 action id 是“当前局面合法动作列表编号”，不是全局固定动作空间。
- 还没有批量环境。
- 已有 JSONL 对局数据生成器、检查器、读取器和监督学习训练代码。
- reward 仍是最简单的终局奖励。

## 基础 Agent

当前已有六个 agent：

| 名称 | 命令行名称 | 行为 |
| --- | --- | --- |
| `RandomAgent` | `random` | 从当前合法动作中随机选择一个。 |
| `GreedyCaptureAgent` | `greedy` | 优先选择吃子数最多的动作；若并列，优先升王；仍并列则随机。 |
| `MaterialAgent` | `material` | 克隆环境并逐个试走候选动作，选择走后材料分最高的动作。 |
| `AlphaBetaAgent` | `alphabeta` | 使用 Alpha-Beta 搜索模拟未来若干步，叶子局面用可配置增强评估函数打分。 |
| `SupervisedAgent` | `supervised` | 加载监督学习 checkpoint，对当前所有合法候选动作打分并选择分数最高者。 |
| `ValueSearchAgent` | `valuesearch` | 加载 checkpoint，模拟候选动作后的局面，用 `value_head` 做浅层 minimax 估值。 |

示例：

```python
from agents import RandomAgent, GreedyCaptureAgent, MaterialAgent, AlphaBetaAgent
from env import StandardDraughtsEnv

env = StandardDraughtsEnv()
obs = env.reset()
raw = env.raw_observation()
actions = env.legal_actions()

agent = MaterialAgent(seed=1)
action_id = agent.select_action(raw, actions, env)
obs, reward, done, info = env.step(action_id)
```

### ValueSearchAgent

`ValueSearchAgent` 是当前“神经网络 + 搜索”的最小版本。它不改变 C 引擎，也不改变网络结构，只是在 Python 侧做浅层推演：

```text
枚举当前合法动作
-> clone 环境并执行候选动作
-> 若终局，直接按胜负给 -1/1
-> 否则用 checkpoint 的 value_head 评估走后局面
-> depth>1 时继续按当前方最大化、对手最小化做浅层 minimax
```

配置文件：

```text
trainer/config/value_search.yaml
```

关键参数：

| 参数 | 说明 |
| --- | --- |
| `search_depth` | 搜索半回合深度。默认 `1`，先只看当前走后局面。 |
| `top_k` | 每个节点只保留 policy score 最高的前 K 个动作；`0` 表示不裁剪。 |
| `terminal_value` | 搜索到胜负终局时使用的价值幅度，默认与 value head 的 `[-1,1]` 一致。 |

快速评测命令：

```bash
PYTHONPATH=interface:trainer python3 trainer/arena.py \
  --black valuesearch \
  --white supervised \
  --black-checkpoint trainer/checkpoints/candidate_move_net_rl_best.pt \
  --white-checkpoint trainer/checkpoints/candidate_move_net_rl_best.pt \
  --games 50 \
  --opening-random-plies 4 \
  --device cpu
```

这些 agent 的目的不是达到高水平，而是作为后续搜索、模型和强化学习的评测基线。

`MaterialAgent` 当前通过 `env.clone()` 复制局面，然后对每个合法动作试走一步。`clone()` 已使用 C 侧环境结构体直接复制，不再通过从开局重放历史动作实现，因此速度足以支持基础材料评测和后续浅层搜索。

`AlphaBetaAgent` 也只通过 `env.clone()` 和 `env.step(action_id)` 模拟未来局面，不在 Python 里重新实现任何规则。它可以看作“多步搜索 + 静态局面评估”：搜索到配置深度后，用增强评估函数给叶子局面打分；轮到自己时选择分数最高的分支，轮到对手时假设对手会选择让自己分数最低的分支。Alpha-Beta 剪枝不会改变同深度搜索的理论结果，只是跳过一部分已经不可能影响最终选择的分支。

### MaterialAgent 配置

`MaterialAgent` 的人为评估分数放在：

```text
trainer/config/material.yaml
```

默认内容：

```yaml
man_value: 100
king_value: 250
win_value: 1000000
draw_value: 0
```

含义：

| 参数 | 含义 |
| --- | --- |
| `man_value` | 普通子的材料价值。 |
| `king_value` | 王的材料价值。 |
| `win_value` | 走后直接获胜时的极大分。 |
| `draw_value` | 走后和棋时的分数。当前和棋规则暂未实际使用。 |

当前项目没有引入 PyYAML 依赖，因此这里支持的是一个很小的 YAML 子集：

```text
key: integer
```

可以写注释，例如：

```yaml
# 王比普通子更有价值
king_value: 300
```

如果你想实验不同参数，可以复制一份：

```bash
cp trainer/config/material.yaml /tmp/my-material.yaml
```

然后编辑 `/tmp/my-material.yaml`，再运行 arena：

```bash
PYTHONPATH=interface:trainer python3 trainer/arena.py --black material --white random --games 100 --material-config /tmp/my-material.yaml
```

### AlphaBetaAgent 配置

`AlphaBetaAgent` 的默认参数放在：

```text
trainer/config/alphabeta.yaml
```

默认内容：

```yaml
search_depth: 2
man_value: 100
king_value: 250
win_value: 1000000
draw_value: 0
advancement_value: 4
center_value: 3
edge_penalty: 2
back_row_guard_value: 5
```

含义：

| 参数 | 含义 |
| --- | --- |
| `search_depth` | 搜索深度，单位是半回合。`1` 时只看当前一步后的增强评估。 |
| `man_value` | 普通子的材料价值。 |
| `king_value` | 王的材料价值。 |
| `win_value` | 搜索到胜局时的极大分。 |
| `draw_value` | 搜索到和棋时的分数。当前和棋规则暂未实际使用。 |
| `advancement_value` | 普通子每向升王方向推进一行获得的奖励。 |
| `center_value` | 棋子位于中心区域时获得的奖励。 |
| `edge_penalty` | 棋子位于最左/最右边线时受到的惩罚。 |
| `back_row_guard_value` | 普通子停留在己方底线时获得的底线守卫奖励。 |

增强评估函数当前仍是静态评估，不生成额外合法动作。它包含：

- 材料：普通子与王的基础价值。
- 推进：普通子越接近升王行，得分越高。
- 中心：处于中心区域的棋子更灵活，得分更高。
- 边线：边线棋子活动空间较小，会扣分。
- 底线守卫：己方底线普通子保留少量守卫价值。

机动性、被威胁棋子、安全吃子等特征暂未加入，因为这些特征需要更频繁地为双方生成合法动作，当前 Python 搜索下会明显增加耗时。后续如果继续优化 AlphaBeta，可以优先考虑把这些计算做得更高效。

同样不依赖 PyYAML，只支持 `key: integer` 形式。可以复制后自行调整：

```bash
cp trainer/config/alphabeta.yaml /tmp/my-alphabeta.yaml
```

运行示例：

```bash
PYTHONPATH=interface:trainer python3 trainer/arena.py --black alphabeta --white random --games 20 --alphabeta-config /tmp/my-alphabeta.yaml
```

深度越大，单步决策会明显变慢。当前默认深度为 `2`，主要用于验证搜索框架和建立传统搜索基线；之后可以继续优化走法排序、缓存和搜索效率。

## 批量评测器 Arena

`arena.py` 可以让两个 agent 批量对战，并统计胜负、最大步数局数、平均步数和耗时。默认参数来自：

```text
trainer/config/arena.yaml
```

命令行显式参数会覆盖该文件。

示例：

```bash
make interface
PYTHONPATH=interface:trainer python3 trainer/arena.py --black random --white greedy --games 100 --max-plies 300 --log-interval 20 --seed 1
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--config` | `trainer/config/arena.yaml` | arena YAML 配置。 |
| `--black` | `random` | 黑方 agent，可选 `random`、`greedy`、`material`、`alphabeta`、`supervised`。 |
| `--white` | `greedy` | 白方 agent，可选 `random`、`greedy`、`material`、`alphabeta`、`supervised`。 |
| `--games` | `100` | 对战局数。 |
| `--max-plies` | `300` | 每局最大半回合数。超过后记为 `max_plies`。 |
| `--log-interval` | `20` | 每隔多少局打印一次进度。 |
| `--seed` | `1` | 随机种子。 |
| `--opening-random-plies` | `0` | 正式评测 agent 接管前，从标准初始局面随机走多少个合法半回合。 |
| `--material-config` | 默认配置 | 指定 `MaterialAgent` 的 YAML 参数文件。 |
| `--alphabeta-config` | 默认配置 | 指定 `AlphaBetaAgent` 的 YAML 参数文件。 |
| `--black-checkpoint` | 无 | 黑方 `SupervisedAgent` 的 checkpoint。 |
| `--white-checkpoint` | 无 | 白方 `SupervisedAgent` 的 checkpoint。 |
| `--device` | 自动 | `SupervisedAgent` 使用的设备，例如 `cpu`、`cuda`、`cuda:0`。 |

输出示例：

```text
arena: black=random white=greedy games=100 max_plies=300 seed=1
arena: 20/100 games, black_wins=8 white_wins=12 draws=0 max_plies=0 avg_plies=92.30 elapsed=0.100s
```

字段含义：

- `black_wins`：黑方胜局数。
- `white_wins`：白方胜局数。
- `draws`：和棋局数；当前和棋规则暂未实现，通常为 0。
- `max_plies`：达到最大半回合数仍未结束的局数。
- `avg_plies`：平均每局半回合数。
- `elapsed`：累计耗时。

### 随机 Opening

两个 greedy checkpoint 从标准初始局面开始时，重复跑很多局往往仍是同一条确定性主线。可用随机 opening 先打破开局重复：

```bash
PYTHONPATH=interface:trainer python3 trainer/arena.py \
  --black supervised \
  --white supervised \
  --black-checkpoint trainer/checkpoints/candidate_move_net_rl_best.pt \
  --white-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 100 \
  --opening-random-plies 8 \
  --seed 2026 \
  --device cpu
```

随机 opening 只在正式评测 agent 接管前执行：

- 每个 opening 半回合都由 C 引擎给出的合法动作随机产生。
- `--seed` 固定时 opening 序列可复现。
- opening 半回合计入该局 `--max-plies` 和 `avg_plies`。
- opening 阶段可能触发强制吃子或提前终局；仍按正常引擎规则处理。

`arena.py` 只负责单组对战；比较相近 candidate 与 base 时，建议优先使用下面的 `compare_checkpoints.py` 做自动换边汇总。

## 自动 Checkpoint 对比

`compare_checkpoints.py` 用于自动完成每轮 RL candidate 的晋升评测。它会复用 `arena.run_arena()`，对同一组 seed 和 opening depth 自动跑：

```text
candidate 黑 vs baseline 白
baseline 黑 vs candidate 白
```

并按 candidate / baseline 视角汇总胜负。

基本命令：

```bash
PYTHONPATH=interface:trainer python3 trainer/compare_checkpoints.py \
  --candidate trainer/checkpoints/candidate_move_net_rl_best.pt \
  --baseline trainer/checkpoints/candidate_move_net_rl_iter_003_best.pt \
  --preset quick \
  --report reports/compare_iter_004.json \
  --device cpu
```

评测预设：

| 预设 | 用途 | 特点 |
| --- | --- | --- |
| `quick` | 快速筛掉明显退步的候选模型 | 只跑 candidate vs baseline，不跑外部 agent，速度最快。 |
| `standard` | 常规晋级初筛 | 跑 candidate vs baseline，并用 `material` 做快速参考。 |
| `full` | 昂贵的最终确认 | 在 `standard` 基础上额外跑 `alphabeta`，耗时明显更长。 |
| `alphabeta` | 单独检查强基准退化 | 只少量跑 `alphabeta` 外部参考。 |

建议日常先用：

```bash
PYTHONPATH=interface:trainer python3 trainer/compare_checkpoints.py \
  --candidate trainer/checkpoints/rl_iter_next/candidate_move_net_rl_latest.pt \
  --baseline trainer/checkpoints/candidate_move_net_rl_best.pt \
  --preset quick \
  --device cpu
```

如果 `quick` 显示候选模型接近或超过晋级线，再跑：

```bash
PYTHONPATH=interface:trainer python3 trainer/compare_checkpoints.py \
  --candidate trainer/checkpoints/rl_iter_next/candidate_move_net_rl_latest.pt \
  --baseline trainer/checkpoints/candidate_move_net_rl_best.pt \
  --preset standard \
  --report reports/compare_rl_iter_next_standard.json \
  --device cpu
```

只有候选模型在 `standard` 中看起来值得晋级时，才建议用 `--preset full` 或显式 `--external-agents alphabeta` 做慢速确认。

默认配置文件：

```text
trainer/config/compare.yaml
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--candidate` | 必填 | 待评测的新 checkpoint。 |
| `--baseline` | 必填 | 当前基座 checkpoint。 |
| `--preset` | `custom` | 快速评测预设。可选 `quick`、`standard`、`full`、`alphabeta`。 |
| `--games` | `100` | 每个方向、每个 seed/opening 组合的局数。 |
| `--seeds` | 配置文件控制 | 多个 opening 随机种子。 |
| `--opening-random-plies` | 配置文件控制 | 多个随机 opening 半回合数。 |
| `--external-agents` | `material` | 外部参考 agent。可写 `random,greedy,material,alphabeta`；AlphaBeta 很慢，默认不跑。 |
| `--report` | `none` | JSON 报告输出路径。 |

输出分为两部分：

- **main**：candidate vs baseline，是晋升判断核心。
- **external**：candidate 和 baseline 分别对 random/greedy/material/alphabeta 等外部 agent 的参考结果，只作为风险提示。

第一版晋升建议规则：

```text
overall decisive win rate >= promote_threshold
并且没有任何 seed/opening bucket 低于 bucket_floor
=> promote

overall decisive win rate < reject_threshold
=> reject

其他情况
=> inconclusive
```

默认：

```text
promote_threshold = 0.53
reject_threshold  = 0.50
bucket_floor      = 0.45
```

注意：外部 agent 结果不会直接和 candidate/base 主胜率混合。它们用于发现 candidate 是否虽然打赢上一代，但对 AlphaBeta/material 等外部基线掉档。

可以用它做基线比较：

```bash
PYTHONPATH=interface:trainer python3 trainer/arena.py --black random --white random --games 1000
PYTHONPATH=interface:trainer python3 trainer/arena.py --black greedy --white random --games 1000
PYTHONPATH=interface:trainer python3 trainer/arena.py --black random --white greedy --games 1000
PYTHONPATH=interface:trainer python3 trainer/arena.py --black material --white random --games 100
PYTHONPATH=interface:trainer python3 trainer/arena.py --black alphabeta --white material --games 20
```

评测监督学习模型：

```bash
PYTHONPATH=interface:trainer python3 trainer/arena.py \
  --black supervised \
  --white alphabeta \
  --black-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 20 \
  --device cpu
```

比较两个不同 checkpoint：

```bash
PYTHONPATH=interface:trainer python3 trainer/arena.py \
  --black supervised \
  --white supervised \
  --black-checkpoint trainer/checkpoints/candidate_move_net_epoch_020.pt \
  --white-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 50 \
  --device cpu
```

## 对局数据生成 Selfplay

`selfplay.py` 用已有 agent 自动下棋，并把每一步保存成 JSONL 数据集。它的目标不是只保存人类可读棋谱，而是保存后续监督学习、价值学习和强化学习都能复用的结构化样本。

基本命令：

```bash
make interface
PYTHONPATH=interface:trainer python3 trainer/selfplay.py --config trainer/config/selfplay.yaml
```

`selfplay.py` 会先读取 `--config` 指定的 YAML 配置，再用命令行中显式给出的参数覆盖 YAML。默认配置文件是：

```text
trainer/config/selfplay.yaml
```

例如临时换 seed 和输出文件：

```bash
PYTHONPATH=interface:trainer python3 trainer/selfplay.py \
  --config trainer/config/selfplay.yaml \
  --seed 2026 \
  --output data/alphabeta_selfplay_seed_2026.jsonl
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--config` | `trainer/config/selfplay.yaml` | selfplay YAML 配置文件。 |
| `--black` | `alphabeta` | 黑方 agent，可选 `random`、`greedy`、`material`、`alphabeta`、`supervised`。 |
| `--white` | `alphabeta` | 白方 agent，可选 `random`、`greedy`、`material`、`alphabeta`、`supervised`。 |
| `--games` | `10` | 生成多少局。 |
| `--max-plies` | `400` | 每局最大半回合数。达到上限仍未结束时，该局样本的 `value_target` 记为 `0`。 |
| `--output` | `data/selfplay.jsonl` | 输出 JSONL 文件路径。父目录不存在时会自动创建。 |
| `--log-interval` | `1` | 每隔多少局打印一次进度。 |
| `--seed` | `1` | 随机种子。 |
| `--material-config` | 默认配置 | 指定 `MaterialAgent` 的 YAML 参数文件。 |
| `--alphabeta-config` | 默认配置 | 指定 `AlphaBetaAgent` 的 YAML 参数文件。 |
| `--black-checkpoint` | 无 | 黑方 `SupervisedAgent` 的 checkpoint。 |
| `--white-checkpoint` | 无 | 白方 `SupervisedAgent` 的 checkpoint。 |
| `--device` | 自动 | `SupervisedAgent` 使用的设备，例如 `cpu`、`cuda`、`cuda:0`。 |
| `--append` | 关闭 | 追加写入 `--output`，并从已有文件的最大 `game_id + 1` 继续编号。 |
| `--auto-output-name` | 开启 | 如果 `--output` 已存在，自动写入 `xxx_001.jsonl`、`xxx_002.jsonl` 等新文件，避免覆盖。 |
| `--no-auto-output-name` | 无 | 临时关闭 YAML 中的 `auto_output_name`。 |
| `--no-append` | 无 | 临时关闭 YAML 中的 `append`。 |

默认 YAML 开启 `auto_output_name`，因此目标文件已存在时不会覆盖旧数据。批量生成数据时也可以选择追加模式。

追加到同一个文件：

```bash
PYTHONPATH=interface:trainer python3 trainer/selfplay.py \
  --black alphabeta \
  --white alphabeta \
  --games 100 \
  --output data/alphabeta_selfplay.jsonl \
  --append
```

自动生成新文件，避免覆盖旧文件：

```bash
PYTHONPATH=interface:trainer python3 trainer/selfplay.py \
  --black alphabeta \
  --white alphabeta \
  --games 100 \
  --output data/alphabeta_selfplay.jsonl \
  --auto-output-name
```

如果 `data/alphabeta_selfplay.jsonl` 已存在，第二条命令会写入类似：

```text
data/alphabeta_selfplay_001.jsonl
```

## RL 自博弈数据生成

`rl_selfplay.py` 是训练专用数据生成器。它直接加载 `CandidateMoveNet` checkpoint，在每个局面先让网络给所有合法候选动作打分，再按：

```text
softmax(policy_logits / temperature)
```

采样动作。这个探索入口与普通 `SupervisedAgent` 分离：arena、GUI 和普通 supervised 对局仍保持 greedy，RL 生成器负责保存探索采样时的概率和值预测。

基本命令：

```bash
PYTHONPATH=interface:trainer python3 trainer/rl_selfplay.py \
  --checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 100 \
  --output data/rl_iter_001.jsonl \
  --device cpu
```

默认配置文件：

```text
trainer/config/rl_selfplay.yaml
```

配置文件给默认参数，命令行显式参数会覆盖 YAML。第一版可调参数包括：

| 参数 | 说明 |
| --- | --- |
| `--checkpoint` | 作为当前 policy/value 基座的 checkpoint。必须指定。 |
| `--games` | 生成局数。 |
| `--max-plies` | 每局最大半回合数。 |
| `--temperature` | temperature softmax 采样温度，必须大于 `0`。 |
| `--seed` | 探索采样随机种子。 |
| `--output` | 输出 JSONL 路径。 |
| `--log-interval` | 进度日志间隔。 |
| `--device` | 推理设备，例如 `cpu` 或 `cuda`。 |
| `--append` / `--auto-output-name` | 与普通 selfplay 一致的追加和防覆盖策略。 |

RL JSONL 保留普通训练样本需要的 `raw_squares`、`observation`、`legal_moves`、`selected_move`、`final_result`、`value_target` 等字段，同时额外保存：

| 字段 | 含义 |
| --- | --- |
| `sample_kind` | 当前为 `rl_selfplay`。 |
| `behavior_policy` | 当前为 `temperature_softmax`。 |
| `temperature` | 这一步采样使用的温度。 |
| `policy_checkpoint` | 生成该样本的 checkpoint 路径。 |
| `policy_probs` | 当前合法动作列表对应的采样概率。 |
| `selected_policy_prob` | 实际选中动作的采样概率。 |
| `predicted_value` | 采样时模型对当前局面的 value 预测。 |

这些字段供后续 self-play 训练和更正式的策略更新使用。当前步骤只生成 RL 数据，还没有自动训练 candidate checkpoint 或自动晋升模型。

JSONL 表示“一行一个 JSON 对象”。每一行是一手棋之前的训练样本，常用字段包括：

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 数据格式版本，当前为 `1`。 |
| `game_id` | 第几局，从 `0` 开始。 |
| `ply` | 当前局内第几个半回合，从 `0` 开始。 |
| `side_to_move` | 当前行棋方，黑方 `0`，白方 `1`。 |
| `agent` | 这一手由哪个 agent 选择。 |
| `raw_squares` | 长度 50 的原始棋盘数组，便于调试和复盘。 |
| `observation` | `6x10x10` 训练编码。 |
| `legal_action_ids` | 当前合法动作编号列表。 |
| `legal_moves` | 当前所有合法动作的路径、吃子列表和动作编号。 |
| `action_mask` | 长度 512 的合法动作掩码。 |
| `selected_action_id` | agent 实际选择的动作编号。 |
| `selected_move` | agent 实际选择动作的路径和吃子列表。 |
| `final_result` | 该局最终结果。黑胜 `1`，白胜 `2`，和棋 `3`，达到最大步数仍未结束通常为 `0`。 |
| `max_plies_reached` | 是否因达到最大半回合数而停止。 |
| `value_target` | 从该样本当前行棋方视角看最终结果：胜 `1`，负 `-1`，和棋或最大步数未结束 `0`。 |

查看前几条样本：

```bash
head -n 3 data/selfplay.jsonl
```

这种格式后续可以直接用于：

- 监督学习：用 `selected_action_id` 模仿 AlphaBeta 或其他老师。
- 价值学习：用 `value_target` 学习局面胜负倾向。
- 数据审计：用 `raw_squares`、`legal_moves` 和 `selected_move` 检查 agent 决策。

## 数据检查 Inspect Dataset

生成数据后，建议先用 `inspect_dataset.py` 检查数据是否干净、分布是否合理，再进入训练。

检查单个文件：

```bash
PYTHONPATH=interface:trainer python3 trainer/inspect_dataset.py data/selfplay.jsonl
```

检查多个文件：

```bash
PYTHONPATH=interface:trainer python3 trainer/inspect_dataset.py data/ab_selfplay_001.jsonl data/ab_vs_material_001.jsonl
```

检查整个目录，工具会递归读取其中所有 `*.jsonl` 文件：

```bash
PYTHONPATH=interface:trainer python3 trainer/inspect_dataset.py data/
```

输出字段示例：

```text
dataset inspection
files: 4
samples: 120000
valid_samples: 120000
invalid_json_lines: 0
invalid_samples: 0
games: 400
black_wins: 170
white_wins: 150
draws: 0
max_plies_reached: 80
avg_plies: 300.00
value_target: -1=42000 0=24000 1=54000
status: ok
```

重点看这些项目：

| 字段 | 含义 |
| --- | --- |
| `valid_samples` | 结构合法的样本数。 |
| `invalid_json_lines` | 无法解析为 JSON 的行数，正常应为 `0`。 |
| `invalid_samples` | 字段或结构不合法的样本数，正常应为 `0`。 |
| `games` | 数据集中局数；多个文件中相同 `game_id` 会按不同文件分别计数。 |
| `max_plies_reached` | 达到最大半回合数后截断的局数。比例太高时，`value_target=0` 会增多。 |
| `value_target` | 从当前行棋方视角的胜负标签分布。 |
| `validation_errors` | 具体结构错误统计；正常应为 `none`。 |
| `status` | `ok` 表示没有 JSON 或样本结构错误。 |

检查器当前会验证：

- 必需字段是否存在。
- `observation` 是否为 `6x10x10`。
- `raw_squares` 是否长度为 `50`。
- `action_mask` 是否长度为 `512`。
- `selected_action_id` 是否在 `legal_action_ids` 中。
- 合法动作编号在 `action_mask` 中是否为 `1`。
- `value_target` 是否只包含 `-1/0/1`。

## 数据读取 Dataset

`dataset.py` 负责读取 `selfplay.py` 生成的 JSONL，并转换成训练前更直接可用的样本结构。

基本用法：

```python
from dataset import DraughtsJsonlDataset, collate_batch, to_torch_batch

dataset = DraughtsJsonlDataset(["data/selfplay.jsonl"])
item = dataset[0]

print(item["policy_target"])
print(item["value_target"])

batch = collate_batch([dataset[0], dataset[1]])
```

`item` 的字段：

| 字段 | 含义 |
| --- | --- |
| `observation` | `6x10x10` 嵌套列表。 |
| `action_mask` | 长度 512 的合法动作掩码。 |
| `policy_target` | 要模仿的动作编号，即 `selected_action_id`。 |
| `move_features` | 当前所有合法动作的候选动作编码，形状为 `N x 45`。 |
| `policy_target_index` | 老师选择的动作在 `legal_moves` / `move_features` 中的位置。 |
| `value_target` | 当前行棋方视角的胜负标签。 |
| `metadata` | `game_id`、`ply`、`side_to_move`、`agent`、`final_result` 等调试信息。 |

读取整个目录：

```python
dataset = DraughtsJsonlDataset(["data/"])
```

流式读取大文件：

```python
from dataset import iter_jsonl_samples

for sample in iter_jsonl_samples(["data/"]):
    ...
```

如果环境安装了 PyTorch，可以把 batch 转成 tensor：

```python
torch_batch = to_torch_batch(batch)
```

当前 `DraughtsJsonlDataset` 是内存数据集，会把样本读入列表，适合当前几万到几十万样本的阶段。后续数据量变大后，可以继续实现真正的懒加载或分片读取。

动作编码和模型结构技术细节见：

```text
trainer/README-network.md
```

## 神经网络模型

`model.py` 当前实现了第一版 `CandidateMoveNet`，用于候选动作打分。

输入：

```text
observation:   [B, 6, 10, 10]
move_features: [B, M, 45]
move_mask:     [B, M]
```

输出：

```text
policy_logits: [B, M]
value:         [B]
```

其中 `M` 是当前 batch 中 padding 后的最大候选动作数。`move_mask=0` 的 padding 动作会被设置为极小 logit，不参与实际选择。

最小用法：

```python
from dataset import DraughtsJsonlDataset, collate_batch, to_torch_batch
from model import CandidateMoveNet

dataset = DraughtsJsonlDataset(["data/selfplay.jsonl"])
batch = collate_batch([dataset[0], dataset[1]])
torch_batch = to_torch_batch(batch)

model = CandidateMoveNet()
output = model(
    torch_batch["observation"],
    torch_batch["move_features"],
    torch_batch["move_mask"],
)
```

## 监督学习训练

`train_supervised.py` 用 JSONL 对局样本训练 `CandidateMoveNet`。训练目标有两个：

```text
policy loss: 模仿样本中的 selected_move
value loss:  预测 value_target
```

常用命令：

```bash
PYTHONPATH=interface:trainer python3 trainer/train_supervised.py data \
  --config trainer/config/supervised.yaml \
  --model-config trainer/config/model.yaml \
  --checkpoint-dir trainer/checkpoints \
  --device cpu
```

如果有 CUDA GPU，可以把 `--device cpu` 改成 `--device cuda`；如果不传 `--device`，脚本会自动优先使用 CUDA，否则使用 CPU。

旧命令仍然可用，等价于所有输入数据：

```text
policy_weight = 1.0
value_weight = 1.0
```

如果要让不同数据源承担不同作用，可以使用 `--source`：

```bash
PYTHONPATH=interface:trainer python3 trainer/train_supervised.py \
  --source data/alphabeta_selfplay.jsonl:1.0:1.0:alphabeta \
  --source data/value_selfplay.jsonl:0.0:1.0:value_selfplay \
  --resume trainer/checkpoints/candidate_move_net_latest.pt \
  --checkpoint-dir trainer/checkpoints \
  --device cpu
```

`--source` 格式为：

```text
PATH:POLICY_WEIGHT:VALUE_WEIGHT[:NAME]
```

| 字段 | 含义 |
| --- | --- |
| `PATH` | JSONL 文件或目录。目录会递归读取其中所有 `*.jsonl`。 |
| `POLICY_WEIGHT` | 该来源样本对 policy loss 的权重。设为 `0` 表示不模仿这些动作。 |
| `VALUE_WEIGHT` | 该来源样本对 value loss 的权重。 |
| `NAME` | 可选名称，只用于记录和辨认数据源。 |

也可以使用配置文件：

```text
trainer/config/data_sources.yaml
```

格式示例：

```yaml
source_1_path: data/alphabeta_selfplay.jsonl
source_1_policy_weight: 1.0
source_1_value_weight: 1.0
source_1_name: alphabeta

source_2_path: data/value_selfplay.jsonl
source_2_policy_weight: 0.0
source_2_value_weight: 1.0
source_2_name: value_selfplay
```

运行：

```bash
PYTHONPATH=interface:trainer python3 trainer/train_supervised.py \
  --source-config trainer/config/data_sources.yaml \
  --resume trainer/checkpoints/candidate_move_net_latest.pt \
  --checkpoint-dir trainer/checkpoints \
  --device cpu
```

当前建议：

- AlphaBeta 数据：`policy_weight=1.0`，`value_weight=1.0`。
- 自博弈或其他结果数据：如果动作质量没有保证，先设 `policy_weight=0.0`，`value_weight=1.0`，只用于 value 学习。

训练时会打印每个 epoch 的关键指标：

启动阶段也会打印配置读取、数据集加载、数据划分、DataLoader 构建、模型初始化和 checkpoint 加载进度，避免大数据集或 resume 时出现长时间空白。

```text
train_loss           训练集总损失
train_policy_acc     训练集动作模仿准确率
train_value_mae      训练集 value 平均绝对误差
val_loss             验证集总损失
val_policy_acc       验证集动作模仿准确率
val_value_mae        验证集 value 平均绝对误差
elapsed              当前这次命令已经训练的时间
cumulative_train_time 同一组参数和数据路径累计训练时间
```

默认保存：

```text
trainer/checkpoints/candidate_move_net_latest.pt
trainer/checkpoints/candidate_move_net_best.pt
trainer/checkpoints/candidate_move_net_epoch_<final_epoch>.pt
trainer/checkpoints/training_time.json
trainer/checkpoints/training_log_<experiment_key>.csv
```

其中：

- `candidate_move_net_latest.pt`：每个 epoch 后覆盖更新，便于继续训练。
- `candidate_move_net_best.pt`：验证集 loss 最好时覆盖更新。
- `candidate_move_net_epoch_<final_epoch>.pt`：本次训练命令结束时额外保存一份带 epoch 编号的 checkpoint，避免只有 latest 被下次训练覆盖。

默认不会保存所有中间 epoch，以免长时间训练产生大量 `.pt` 文件。确实需要保存每个 epoch 时，可以显式加入：

```bash
--save-every-epoch
```

checkpoint 中包含：

```text
model_state_dict          模型权重
optimizer_state_dict      优化器状态，便于继续训练
supervised_config         训练参数
model_config              模型结构参数
train_metrics             训练指标
val_metrics               验证指标
run_elapsed_seconds       当前这次训练命令耗时
cumulative_train_seconds  累计训练秒数
cumulative_train_time     累计训练时间，格式为 HH:MM:SS
```

继续训练：

```bash
PYTHONPATH=interface:trainer python3 trainer/train_supervised.py data \
  --resume trainer/checkpoints/candidate_move_net_latest.pt \
  --checkpoint-dir trainer/checkpoints
```

累计训练时间有两种接力方式：

- 使用 `--resume` 时，会从 checkpoint 中读取已有累计时间。
- 不使用 `--resume` 但配置、模型参数和数据路径相同时，会从 `training_time.json` 中读取同一实验的累计时间。

CSV 训练日志会按同一实验追加写入，不覆盖旧记录。默认每个 epoch 写一行，字段包括：

```text
epoch
train_loss / train_policy_loss / train_value_loss
train_policy_accuracy / train_value_mae
val_loss / val_policy_loss / val_value_loss
val_policy_accuracy / val_value_mae
best_val_loss
run_elapsed_seconds / run_elapsed_time
cumulative_train_seconds / cumulative_train_time
```

长时间训练时，可以在 `trainer/config/supervised.yaml` 中调大：

```yaml
csv_log_interval_epochs: 5
```

这表示每 5 个 epoch 写一行 CSV；最后一个 epoch 总会写入。CSV 使用逐行追加写入，不会把历史 CSV 全部读入内存，适合隔夜训练。

## RL 自博弈训练

`train_selfplay.py` 读取 `rl_selfplay.py` 生成的探索样本，从已有 checkpoint 训练一个新的 candidate。它与监督训练的区别是：RL 样本中被采样到的 `selected_move` 不是老师答案，不能直接按交叉熵模仿。

当前采用 PPO 风格 clipped policy objective。它仍是 Actor-Critic 路线，但会用生成样本时保存的行为概率限制单轮策略偏移：

```text
advantage   = value_target - 样本中的 predicted_value
old_prob    = 样本中的 selected_policy_prob
new_prob    = 当前模型在样本 behavior temperature 下的 selected_move 概率
ratio       = new_prob / old_prob

surrogate_1 = ratio * advantage
surrogate_2 = clip(ratio, 1-epsilon, 1+epsilon) * advantage
policy_loss = - min(surrogate_1, surrogate_2)
value_loss  = MSE(value, value_target)
entropy     = 当前合法动作策略分布的熵

total_loss =
    policy_loss_weight * policy_loss
  + value_loss_weight  * value_loss
  - entropy_weight     * entropy
```

其中 `score_head` 承担 Actor 的动作评分职责，`value_head` 承担 Critic 的局面价值估计职责。`predicted_value` 是 rollout 时旧策略写入样本的 baseline；当前只使用终局 `value_target` 作为 return；TD/GAE 和自动晋升暂未实现。

常用命令：

```bash
PYTHONPATH=interface:trainer python3 trainer/train_selfplay.py data/rf_data \
  --resume trainer/checkpoints/candidate_move_net_best.pt \
  --config trainer/config/rl_train.yaml \
  --checkpoint-dir trainer/checkpoints \
  --device cpu
```

`--resume` 必须指定：

- 从监督 checkpoint 起步时，只加载网络结构和权重，RL epoch 从 `1` 开始。
- 从 `candidate_move_net_rl_*.pt` 继续训练时，会恢复 RL optimizer、RL epoch 和累计训练时间。

RL 训练配置在：

```text
trainer/config/rl_train.yaml
```

关键参数：

| 参数 | 说明 |
| --- | --- |
| `learning_rate` | RL candidate 更新学习率，默认比监督训练保守。 |
| `policy_loss_weight` | Actor 策略损失权重。 |
| `value_loss_weight` | Critic 价值损失权重。 |
| `entropy_weight` | 熵奖励权重；总损失中以减号加入。 |
| `advantage_clip` | Advantage 绝对值裁剪；`0` 表示不裁剪。 |
| `ppo_clip_epsilon` | PPO ratio 裁剪宽度；当前稳定默认 `0.1` 对应主要限制在 `[0.9, 1.1]`。 |
| `validation_split` | RL 验证集比例。 |

当前 `trainer/config/rl_train.yaml` 采用偏稳定的默认参数：

```yaml
learning_rate: 0.00005
advantage_clip: 1.5
ppo_clip_epsilon: 0.1
```

这组参数会让单轮 PPO 更新更慢，但更不容易把已经通过评测的基座模型推坏。若连续多轮 candidate 都明显落后，可以继续降低 `learning_rate`；若多轮都只是小幅接近但无法超过基座，可以考虑在充分备份后略微提高 `epochs` 或恢复稍大的 `ppo_clip_epsilon`。

训练时会打印：

```text
loss / policy_loss / value_loss
entropy
advantage_abs
ratio / clip_fraction
value_mae
elapsed / cumulative_train_time
```

`policy_loss` 允许为负值，单独看它不能代表棋力提升。`ratio` 接近 `1` 表示 candidate 仍靠近生成数据时的行为策略，`clip_fraction` 过高表示大量样本已经撞到 PPO 限制。训练结束后优先用 `compare_checkpoints.py --preset quick` 快速筛选，再按需要升级到 `standard` 或 `full`。

默认保存：

```text
trainer/checkpoints/candidate_move_net_rl_latest.pt
trainer/checkpoints/candidate_move_net_rl_best.pt
trainer/checkpoints/candidate_move_net_rl_epoch_<final_epoch>.pt
trainer/checkpoints/rl_training_log_<experiment_key>.csv
trainer/checkpoints/training_time.json
```

默认只保存本次训练结束的带 epoch checkpoint；显式传 `--save-every-epoch` 才保存所有中间 epoch。

## 后续方向

建议后续依次实现：

1. 用 `arena.py` 评测 RL candidate 是否优于当前基座，再决定是否进入下一轮 self-play。
2. 继续把 AlphaBeta 数据作为稳定的 policy 监督来源。
3. 按 PPO candidate 实验结果再决定是否加入更强的 return/value 目标或自动晋升机制。
4. 数据规模继续增大后，实现 shard 采样或 streaming dataset。
5. 等后续方法更多时，再统一实现模型评测和晋升脚本。
