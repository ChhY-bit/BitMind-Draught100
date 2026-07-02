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
  --alphabeta-config draughts/config/alphabeta_plus.yaml \
  --games 50 --max-plies 500 --output data/abplus.jsonl --seed 4040
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--config` | path | `scripts/config/selfplay.yaml` | 配置文件，CLI 参数覆盖文件中的值 |
| `--black` | str | `alphabeta` | 黑方 agent：`random`/`material`/`alphabeta`/`alphabeta_plus`/`supervised`/`mcts` |
| `--white` | str | `alphabeta` | 白方 agent，同上 |
| `--games` | int | 100 | 生成局数 |
| `--max-plies` | int | 500 | 每局最大半回合数 |
| `--output` | path | `data/selfplay.jsonl` | 输出 JSONL 文件路径 |
| `--log-interval` | int | 1 | 每隔多少局打印一次进度 |
| `--seed` | int | 1 | 随机种子 |
| `--material-config` | path | none | MaterialAgent 的 YAML 配置 |
| `--alphabeta-config` | path | none | AlphaBetaAgent 的 YAML 配置 |
| `--black-checkpoint` | path | none | 黑方 supervised checkpoint |
| `--white-checkpoint` | path | none | 白方 supervised checkpoint |
| `--device` | str | auto | 推理设备：`cpu`、`cuda`（= `cuda:0`）、`cuda:1` |
| `--mcts-config` | path | none | MCTSAgent 的 YAML 配置 |
| `--append` | flag | off | 追加写入 output 文件 |
| `--auto-output-name` | flag | on | output 已存在时自动生成 `_001`/`_002` 新文件名 |

MCTS 自博弈会自动将 search visit count 分布作为 `policy_target_probs` 写入 JSONL，无需再跑 `search_improve`。

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
| `--alphabeta-depth` | int | 2 | Teacher AlphaBeta 搜索深度 |
| `--min-score-margin` | int | 0 | 只保留 best move 与次优 move 分差 ≥ 此值的局面 |
| `--max-samples` | int | 全部 | 最多输出样本数 |
| `--position-sample-rate` | float | 1.0 | 从输入中随机抽样的比例（0–1） |
| `--alphabeta-config` | path | none | 自定义 AlphaBeta 评估参数 YAML |
| `--log-interval` | int | 100 | 每隔多少样本打印一次进度 |
| `--seed` | int | 1 | 随机种子 |

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
| `--white` | str | `material` | 白方 agent |
| `--games` | int | 100 | 评测局数 |
| `--max-plies` | int | 300 | 每局最大半回合数 |
| `--log-interval` | int | 20 | 每隔多少局打印一次进度 |
| `--seed` | int | 1 | 随机种子 |
| `--opening-random-plies` | int | 0 | 正式评测前随机走多少半回合（打破固定开局） |
| `--material-config` | path | none | MaterialAgent YAML 配置 |
| `--alphabeta-config` | path | none | AlphaBetaAgent YAML 配置 |
| `--black-checkpoint` | path | none | 黑方 supervised checkpoint |
| `--white-checkpoint` | path | none | 白方 supervised checkpoint |
| `--device` | str | auto | 推理设备：`cpu`、`cuda`（= `cuda:0`）、`cuda:1` |
| `--mcts-config` | path | none | MCTSAgent YAML 配置 |

---

## `compare_checkpoints.py` — 自动晋升评测

candidate vs baseline 多 seed/opening 换边评测，输出晋升/拒绝建议。

```bash
PYTHONPATH=interface:. python3 scripts/compare_checkpoints.py \
  --candidate trainer/checkpoints/new.pt \
  --baseline trainer/checkpoints/best.pt \
  --preset quick --report reports/compare.json --device cpu
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--candidate` | path | (必填) | 候选 checkpoint 路径 |
| `--baseline` | path | (必填) | 基线 checkpoint 路径 |
| `--preset` | str | `quick` | 评测预设：`quick`、`standard`、`full`、`alphabeta` |
| `--games` | int | 50/100 | 每 bucket 局数（由 preset 决定） |
| `--max-plies` | int | 300 | 每局最大半回合数 |
| `--seeds` | str | `2028` | 逗号分隔的种子列表 |
| `--opening-random-plies` | str | `4` | 逗号分隔的开局深度列表 |
| `--external-agents` | str | 无 | 逗号分隔的外部基线：`random`、`material`、`alphabeta` |
| `--promote-threshold` | float | 0.53 | 建议 promote 的最低 decisive 胜率 |
| `--reject-threshold` | float | 0.50 | 高于此值不 reject |
| `--bucket-floor` | float | 0.45 | 单个 bucket 胜率低于此值则 inconclusive |
| `--report` | path | none | 输出 JSON 报告路径 |
| `--material-config` | path | none | MaterialAgent YAML 配置 |
| `--alphabeta-config` | path | none | AlphaBetaAgent YAML 配置 |
| `--device` | str | auto | 推理设备：`cpu`、`cuda`（= `cuda:0`）、`cuda:1` |

### Preset 详情

| Preset | 局数 | 种子 | 开局 | 外部参考 |
|---|---|---|---|---|
| `quick` | 50 | 2028 | 4 | 无 |
| `standard` | 100 | 2028,2029 | 2,8 | material |
| `full` | 100 | 2028,2029 | 2,8 | material, alphabeta |
| `alphabeta` | 50 | 2028 | 8 | alphabeta |

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
