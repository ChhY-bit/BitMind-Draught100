# 训练环境说明

`trainer/` 目录当前只实现最基础的标准化环境，还没有开始神经网络训练。它的目标是把 `interface.DraughtsEnv` 包装成后续训练代码更容易使用的形式。

## 当前文件

```text
trainer/
  env.py
  agents.py
  arena.py
  selfplay.py
  inspect_dataset.py
  dataset.py
  config/material.yaml
  config/alphabeta.yaml
  test_env.py
  test_agents.py
  test_arena.py
  test_selfplay.py
  test_inspect_dataset.py
  test_dataset.py
  README-trainer.md
```

- `env.py`：标准化训练环境 `StandardDraughtsEnv`。
- `agents.py`：基础 agent。
- `arena.py`：批量对战评测器。
- `selfplay.py`：对局数据生成器，保存 JSONL 训练样本。
- `inspect_dataset.py`：JSONL 数据检查器。
- `dataset.py`：JSONL 数据读取器与 batch 组装工具。
- `config/material.yaml`：`MaterialAgent` 的材料评估参数。
- `config/alphabeta.yaml`：`AlphaBetaAgent` 的搜索深度和增强评估参数。
- `test_*.py`：训练环境、agent、arena、selfplay、数据检查器、数据读取器测试。
- `README-trainer.md`：本文档。

## 运行测试

在项目根目录执行：

```bash
make interface
PYTHONPATH=interface:trainer python3 -m unittest trainer/test_env.py trainer/test_agents.py trainer/test_arena.py trainer/test_selfplay.py trainer/test_inspect_dataset.py trainer/test_dataset.py
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
| `5` | 当前行棋方平面，黑方为 `0`，白方为 `1` |

说明：

- 当前没有对棋盘做几何翻转。
- 棋子通道使用“当前方/对方”视角，而不是固定黑白视角。
- 第 5 通道只在 50 个合法棋位上填当前行棋方编号，非合法棋位为 `0`。

初始局面下，黑方先手：

```text
通道 0 有 20 个 1
通道 2 有 20 个 1
通道 4 有 50 个 1
通道 5 全为 0
```

黑方走一步后轮到白方：

```text
通道 5 在合法棋位上的值为 1
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
- 已有 JSONL 对局数据生成器、检查器和读取器，但还没有模型训练代码。
- reward 仍是最简单的终局奖励。

## 基础 Agent

当前已有四个基础 agent：

| 名称 | 命令行名称 | 行为 |
| --- | --- | --- |
| `RandomAgent` | `random` | 从当前合法动作中随机选择一个。 |
| `GreedyCaptureAgent` | `greedy` | 优先选择吃子数最多的动作；若并列，优先升王；仍并列则随机。 |
| `MaterialAgent` | `material` | 克隆环境并逐个试走候选动作，选择走后材料分最高的动作。 |
| `AlphaBetaAgent` | `alphabeta` | 使用 Alpha-Beta 搜索模拟未来若干步，叶子局面用可配置增强评估函数打分。 |

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

`arena.py` 可以让两个 agent 批量对战，并统计胜负、最大步数局数、平均步数和耗时。

示例：

```bash
make interface
PYTHONPATH=interface:trainer python3 trainer/arena.py --black random --white greedy --games 100 --max-plies 300 --log-interval 20 --seed 1
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--black` | `random` | 黑方 agent，可选 `random`、`greedy`、`material`、`alphabeta`。 |
| `--white` | `greedy` | 白方 agent，可选 `random`、`greedy`、`material`、`alphabeta`。 |
| `--games` | `100` | 对战局数。 |
| `--max-plies` | `300` | 每局最大半回合数。超过后记为 `max_plies`。 |
| `--log-interval` | `20` | 每隔多少局打印一次进度。 |
| `--seed` | `1` | 随机种子。 |
| `--material-config` | 默认配置 | 指定 `MaterialAgent` 的 YAML 参数文件。 |
| `--alphabeta-config` | 默认配置 | 指定 `AlphaBetaAgent` 的 YAML 参数文件。 |

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

可以用它做基线比较：

```bash
PYTHONPATH=interface:trainer python3 trainer/arena.py --black random --white random --games 1000
PYTHONPATH=interface:trainer python3 trainer/arena.py --black greedy --white random --games 1000
PYTHONPATH=interface:trainer python3 trainer/arena.py --black random --white greedy --games 1000
PYTHONPATH=interface:trainer python3 trainer/arena.py --black material --white random --games 100
PYTHONPATH=interface:trainer python3 trainer/arena.py --black alphabeta --white material --games 20
```

## 对局数据生成 Selfplay

`selfplay.py` 用已有 agent 自动下棋，并把每一步保存成 JSONL 数据集。它的目标不是只保存人类可读棋谱，而是保存后续监督学习、价值学习和强化学习都能复用的结构化样本。

基本命令：

```bash
make interface
PYTHONPATH=interface:trainer python3 trainer/selfplay.py --black alphabeta --white alphabeta --games 10 --max-plies 300 --output data/selfplay.jsonl
```

参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--black` | `alphabeta` | 黑方 agent，可选 `random`、`greedy`、`material`、`alphabeta`。 |
| `--white` | `alphabeta` | 白方 agent，可选 `random`、`greedy`、`material`、`alphabeta`。 |
| `--games` | `10` | 生成多少局。 |
| `--max-plies` | `300` | 每局最大半回合数。达到上限仍未结束时，该局样本的 `value_target` 记为 `0`。 |
| `--output` | 必填 | 输出 JSONL 文件路径。父目录不存在时会自动创建。 |
| `--log-interval` | `1` | 每隔多少局打印一次进度。 |
| `--seed` | `1` | 随机种子。 |
| `--material-config` | 默认配置 | 指定 `MaterialAgent` 的 YAML 参数文件。 |
| `--alphabeta-config` | 默认配置 | 指定 `AlphaBetaAgent` 的 YAML 参数文件。 |

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

## 后续方向

建议后续依次实现：

1. 一个最小 policy/value 网络。
2. 监督学习模仿 AlphaBeta。
3. 训练日志、模型保存与评测流程。
4. 更合理的固定动作空间或候选动作打分模型。
5. 搜索与强化学习流程。
