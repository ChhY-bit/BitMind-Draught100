# Draughts — 核心 Python 库参考手册

`draughts/` 是 100 格国际跳棋项目的核心 Python 库。它封装了 C 规则引擎，提供训练环境、神经网络模型、智能体和数据加载能力。上层 `trainer/`、`scripts/` 和 `gui/` 均依赖此库。

## 设计原则

- **纯库**：所有模块只被 `import`，不包含 `if __name__ == "__main__"` 入口。
- **依赖单向**：`draughts/` 只依赖下层 `interface/`，不依赖上层 `trainer/` 或 `scripts/`。
- **规则不上浮**：棋规逻辑封闭在 C 引擎中，Python 端不重实现。

## 模块一览

| 模块 | 职责 | 关键导出 |
|---|---|---|
| `env.py` | 封装 C 引擎，提供训练环境 | `StandardDraughtsEnv`, `encode_observation` |
| `move_encoding.py` | 候选动作 → 45 维特征向量 | `encode_legal_moves`, `MOVE_FEATURE_DIM` |
| `model.py` | 神经网络定义 | `CandidateMoveNet`, `ModelConfig`, `load_model_config` |
| `agents.py` | 所有智能体类 | 7 个 Agent 类, `make_agent`, `available_agents` |
| `dataset.py` | JSONL 数据加载与批处理 | `DraughtsJsonlDataset`, `collate_batch`, `to_torch_batch` |

## 依赖层次

```
move_encoding.py    ← 无内部依赖，纯工具模块
    │
model.py            ← 依赖 move_encoding
    │
agents.py           ← 依赖 model + move_encoding + interface
    │
env.py              ← 依赖 interface（无 draughts 内部依赖）
    │
dataset.py          ← 依赖 move_encoding + env
```

## API 索引

### `draughts.env`

```python
from draughts.env import StandardDraughtsEnv, encode_observation

env = StandardDraughtsEnv()
env.reset()
env.step(action_id)          # → (observation, info)
env.observation()            # → 6×10×10 张量格式
env.raw_observation()        # → dict（squares, side_to_move, ...）
env.legal_actions()          # → Sequence[Move]
env.action_mask()            # → List[int] 长度 512
env.clone()                  # → StandardDraughtsEnv
env.set_state(squares, side) # 从原始数据恢复局面
```

observation 为 6×10×10 的当前方视角编码（黑方行棋保持原方向，白方行棋自动旋转 180°）。

### `draughts.move_encoding`

```python
from draughts.move_encoding import encode_legal_moves, MOVE_FEATURE_DIM

# MOVE_FEATURE_DIM = 45
features = encode_legal_moves(move_dicts, side_to_move)
# → List[List[float]]  每个动作 45 维
```

### `draughts.model`

```python
from draughts.model import CandidateMoveNet, ModelConfig, load_model_config

config = load_model_config("draughts/config/model.yaml")
model = CandidateMoveNet(config)

# 前向传播
output = model(observation, move_features, move_mask)
# → {"policy_logits": Tensor[B, M], "value": Tensor[B]}
```

架构：CNN 板面编码器（3 层卷积 → 256 维） + MLP 动作编码器（45→128） + score_head（256+128→1） + value_head（256→1 + Tanh）。

### `draughts.agents`

```python
from draughts.agents import make_agent, available_agents

print(available_agents())
# ['random', 'material', 'alphabeta', 'supervised']

agent = make_agent("alphabeta")  # depth=2 (default)
agent = make_agent("alphabeta", alphabeta_config_path="draughts/config/alphabeta_plus.yaml")  # depth=3
agent = make_agent("supervised", supervised_checkpoint_path="checkpoint.pt", device="cpu")

action_id = agent.select_action(observation, legal_actions, env)
```

| Agent | 名称 | 说明 |
|---|---|---|
| `RandomAgent` | `random` | 随机选择合法动作 |
| `MaterialAgent` | `material` | 一步试走，材料评估 |
| `AlphaBetaAgent` | `alphabeta` | Alpha-Beta 搜索（深度通过 config 调整） |
| `SupervisedAgent` | `supervised` | 加载 `CandidateMoveNet` checkpoint，greedy 选择 |

### `draughts.dataset`

```python
from draughts.dataset import DraughtsJsonlDataset, collate_batch, to_torch_batch

dataset = DraughtsJsonlDataset(["data/file1.jsonl", "data/file2.jsonl"])
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_batch)
```

支持数据源权重（`--source` 或 `--source-config`），可按来源分别设置 policy/value loss 权重。

## 配置

配置 YAML 文件位于 `draughts/config/`：

| 文件 | 加载者 | 说明 |
|---|---|---|
| `model.yaml` | `model.py` | 网络结构参数 |
| `alphabeta.yaml` | `agents.py` | AlphaBeta 搜索深度与评估权重 |
| `alphabeta_plus.yaml` | `agents.py` | 更深搜索配置（depth=3） |
| `material.yaml` | `agents.py` | 材料评估参数 |

## 测试

```bash
make draughts-test
# 或
PYTHONPATH=interface:. python3 -m unittest discover -s draughts/tests -p 'test_*.py'
```
