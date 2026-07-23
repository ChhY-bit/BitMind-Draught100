# Scripts — 独立应用脚本参考

`scripts/` 包含可独立运行的应用程序。每个脚本有各自的 CLI，调用 `draughts/` 核心库完成特定任务。

所有脚本需在项目根目录执行：

```bash
PYTHONPATH=interface:. python3 scripts/<script>.py [args]
```

---

## `selfplay.py` — 对局数据生成

用两个 Agent 对局，生成 JSONL 训练数据。

```bash
PYTHONPATH=interface:. python3 scripts/selfplay.py
PYTHONPATH=interface:. python3 scripts/selfplay.py \
  --black alphabeta --white alphabeta \
  --alphabeta-depth 3 \
  --games 50 --max-plies 300 --output data/ab_plus.jsonl --seed 4040
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--config` | path | `scripts/config/selfplay.yaml` | 配置文件，CLI 参数覆盖文件中的值 |
| `--black` | str | `alphabeta` | 黑方 agent：`random`/`alphabeta`/`supervised`/`mcts` |
| `--white` | str | `alphabeta` | 白方 agent，同上 |
| `--games` | int | 100 | 生成局数 |
| `--max-plies` | int | 300 | 每局最大半回合数 |
| `--output` | path | `data/selfplay.jsonl` | 输出 JSONL 文件路径 |
| `--log-interval` | int | 1 | 每隔多少局打印一次进度 |
| `--seed` | int | 1 | 随机种子 |
| `--alphabeta-config` | path | none | AlphaBetaAgent 的 YAML 配置 |
| `--alphabeta-depth` | int | none | 覆盖 AlphaBeta 搜索深度（优先于 YAML 配置中的 `search_depth`） |
| `--black-checkpoint` | path | none | 黑方 supervised checkpoint |
| `--white-checkpoint` | path | none | 白方 supervised checkpoint |
| `--device` | str | auto | 推理设备：`cpu`、`cuda`（= `cuda:0`）、`cuda:1` |
| `--mcts-config` | path | `draughts/config/mcts_selfplay.yaml` | MCTSAgent 的 YAML 配置（自博弈默认使用探索配置） |
| `--append` | flag | off | 追加写入 output 文件 |
| `--auto-output-name` | flag | on | output 已存在时自动生成 `_001`/`_002` 新文件名 |

MCTS 自博弈会自动将 search visit count 分布作为 `policy_target_probs` 写入 JSONL，无需再跑 `search_improve`。

### 探索多样性：两种机制的分工

MCTS 自博弈时，为了让生成的对局数据足够多样化（而非每局走相同的棋），项目提供了两种不同层面的随机性机制，它们作用的位置截然不同：

| 机制 | 作用层面 | 配置参数 | 影响范围 |
|------|---------|---------|---------|
| Dirichlet 噪声 | **思考层**（MCTS 内部） | `dirichlet_alpha`、`dirichlet_epsilon` | 干预根节点 prior，影响"怎么搜" |
| temperature | **决策层**（MCTS 外部） | `temperature` | 干预最终走法选择，影响"怎么选" |

**Dirichlet 噪声**（思考层）：在 MCTS 搜索开始前，将根节点的网络先验概率与 Dirichlet 分布混合（`P_root = (1-ε)×P_net + ε×Dirichlet(α)`）。α<1 时噪声随机"偏爱"少数走法，模拟"棋手偶然的灵光一现"，迫使 MCTS 不要只盯着网络最推荐的那几步棋。由配置文件中的 `add_noise` 控制——自博弈用 `mcts_selfplay.yaml`（开启），评测用 `mcts.yaml`（关闭）。

**temperature**（决策层）：搜索完成后从 visit count 分布中选择最终走法时介入。temperature=0 直接选 visits 最多的走法（评测用 `mcts.yaml`）；temperature>0 时按 `softmax(visits/temperature)` 采样（自博弈用 `mcts_selfplay.yaml`）。

**两者的配合**：Dirichlet 让 MCTS "想得更广"（搜索阶段），temperature 让对局"走得更花"（决策阶段）。项目提供两套开箱即用的配置，自博弈脚本自动加载 `mcts_selfplay.yaml`，评测/GUI 自动加载 `mcts.yaml`。关闭 Dirichlet 噪声有两种等价方式：`add_noise = false` 或 `dirichlet_epsilon = 0.0`。当前项目的开局多样性主要靠 `--opening-random-plies`（前 N 步随机走），详见 [draughts/config/mcts_selfplay.yaml](../draughts/config/mcts_selfplay.yaml) 和 [draughts/config/mcts.yaml](../draughts/config/mcts.yaml)。

---

## `search_improve.py` — 搜索改进数据生成

用 AlphaBeta 对已有 JSONL 局面的合法动作重新打分，输出改进后的 policy target。

```bash
PYTHONPATH=interface:. python3 scripts/search_improve.py \
  data/alphabeta_selfplay.jsonl \
  --output data/search_improved.jsonl \
  --alphabeta-depth 3 --min-score-margin 50 \
  --max-samples 5000 --seed 2026
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `inputs` | path(s) | (必填) | 输入 JSONL 文件或目录，可多个 |
| `--output` | path | (必填) | 输出 JSONL 文件路径 |
| `--alphabeta-depth` | int | YAML 默认值 | Teacher AlphaBeta 搜索深度（覆盖 YAML 配置） |
| `--min-score-margin` | int | 0 | 只保留 best move 与次优 move 分差 ≥ 此值的局面 |
| `--max-samples` | int | 全部 | 最多输出样本数 |
| `--position-sample-rate` | float | 1.0 | 从输入中随机抽样的比例（0–1） |
| `--alphabeta-config` | path | none | 自定义 AlphaBeta 评估参数 YAML |
| `--log-interval` | int | 100 | 每隔多少样本打印一次进度 |
| `--seed` | int | 1 | 随机种子 |

### 并行化

单进程 `search_improve.py` 对大样本量会很慢，可用 `run/search_improve.sh` 多进程并行：

```bash
INPUT="data/alphabeta" OUTPUT=data/alphabeta/improved_soft.jsonl \
NUM_WORKERS=8 ALPHABETA_DEPTH=7 ./run/search_improve.sh
```

详见 [run/README-run.md](../run/README-run.md)。

---

## `arena.py` — 批量对战评测

批量 Agent 对战，统计胜率，支持随机开局。

```bash
PYTHONPATH=interface:. python3 scripts/arena.py \
  --black supervised --white alphabeta \
  --black-checkpoint path/to/checkpoint.pt \
  --games 20 --device cpu
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--config` | path | `scripts/config/arena.yaml` | 配置文件 |
| `--black` | str | `random` | 黑方 agent |
| `--white` | str | `alphabeta` | 白方 agent |
| `--games` | int | 100 | 评测局数 |
| `--max-plies` | int | 300 | 每局最大半回合数 |
| `--log-interval` | int | 20 | 每隔多少局打印一次进度 |
| `--seed` | int | 1 | 随机种子 |
| `--opening-random-plies` | int | 0 | 正式评测前随机走多少半回合（打破固定开局） |
| `--alphabeta-config` | path | none | AlphaBetaAgent YAML 配置 |
| `--alphabeta-depth` | int | none | 覆盖 AlphaBeta 搜索深度（优先于 YAML 配置中的 `search_depth`） |
| `--black-checkpoint` | path | none | 黑方 supervised checkpoint |
| `--white-checkpoint` | path | none | 白方 supervised checkpoint |
| `--device` | str | auto | 推理设备：`cpu`、`cuda`（= `cuda:0`）、`cuda:1` |
| `--mcts-config` | path | `draughts/config/mcts.yaml` | MCTSAgent YAML 配置（评测默认使用确定性配置） |

---

## `inspect_dataset.py` — 数据质量检查

检查 JSONL 数据的 schema 版本、字段完整性和分布统计。

```bash
PYTHONPATH=interface:. python3 scripts/inspect_dataset.py data/
PYTHONPATH=interface:. python3 scripts/inspect_dataset.py data/specific.jsonl
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `paths` | path(s) | (必填) | 一个或多个 JSONL 文件或目录 |
