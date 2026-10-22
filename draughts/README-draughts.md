# 核心 Python 库

`draughts/` 为界面、训练器和批量工具提供可复用能力。棋规由 C 引擎实现，经 `interface/` 调用；核心库不依赖上层应用。

使用入口见 [项目说明](../README.md)，维护和测试见 [维护与验证](../docs/development.md)。

## 模块

| 模块 | 职责 | 主要接口 |
|---|---|---|
| `env.py` | 训练与搜索环境、当前方视角编码 | `StandardDraughtsEnv`, `encode_observation` |
| `move_encoding.py` | 完整合法走法的 45 维编码 | `encode_legal_moves`, `move_to_dict`, `MOVE_FEATURE_DIM` |
| `model.py` | 棋盘编码、候选走法评分和局面估值 | `CandidateMoveNet`, `ModelConfig`, `load_model_config` |
| `agents.py` | 随机、AlphaBeta、监督模型和 MCTS 智能体 | `make_agent`, `available_agents` |
| `mcts.py` | PUCT 搜索、访问次数与搜索估值 | `mcts_search`, `MCTSConfig`, `MCTSResult` |
| `search_control.py` | 合作式搜索取消 | `SearchCancelled`, `check_cancelled` |
| `notation.py` | TXT 棋谱、完整走法记录与文件写入 | `GameRecord`, `format_move_notation`, `format_move_pairs`, `result_code` |
| `dataset.py` | JSONL 数据加载、批处理及软标签 | `DraughtsJsonlDataset`, `collate_batch`, `to_torch_batch` |

## 环境

从项目根目录、带有 `PYTHONPATH=interface:.` 的环境运行以下示例：

```python
from draughts.env import StandardDraughtsEnv

env = StandardDraughtsEnv()
observation = env.reset()
moves = env.legal_actions()
observation, reward, done, info = env.step(moves[0].action_id)
```

| 接口 | 返回或用途 |
|---|---|
| `reset()` | 标准初始局面的编码观察值 |
| `observation()` | 6×10×10 当前方视角编码，以嵌套列表表示 |
| `raw_observation()` | 原始局面字典，含棋子、行棋方、结果等字段 |
| `legal_actions()` | 当前局面的 `Move` 列表 |
| `action_mask()` | 合法动作掩码，默认长度 512，可由 `max_actions` 设置 |
| `step(action_id)` | `(observation, reward, done, info)` |
| `clone()` | 独立环境副本 |
| `set_state(squares, side)` | 从原始棋子数组与行棋方恢复局面 |

黑方编码保持标准方向，白方编码旋转 180°。这只用于模型输入，不改变原始 `Move.path`、棋盘或导出棋谱中的固定编号。

注意区分两层接口：`interface.DraughtsEnv.step()` 返回 `(observation, info)`，本层 `StandardDraughtsEnv.step()` 返回四元组。动作的 `action_id` 只在对应局面的合法走法集合中有效，不能当作固定棋谱编号。

## 模型与智能体

```python
from draughts.agents import make_agent

agent = make_agent('alphabeta', alphabeta_depth=3)
action_id = agent.select_action(
    env.raw_observation(), env.legal_actions(), env
)
```

| 名称 | 方式 | 是否需要模型 |
|---|---|---|
| `random` | 随机合法动作 | 否 |
| `alphabeta` | 手工估值与 AlphaBeta 搜索 | 否 |
| `supervised` | 网络直接选择评分最高的候选动作 | 是 |
| `mcts` | 网络提供先验和估值，MCTS 搜索后选择动作 | 是 |

`CandidateMoveNet` 接收棋盘、候选动作特征与动作掩码，输出 `policy_logits`（`B×M`）及 `value`（`B`）。默认结构配置见 [model.yaml](config/model.yaml)；加载已有模型时按 checkpoint 保存的 `model_config` 恢复结构，支持既有 v1/v2/v3 模型。

智能体的 `select_action()` 和 `mcts_search()` 接受可选 `cancel_event`。取消时抛出 `SearchCancelled`；检查发生于搜索节点或推理调用之间，不能强行打断单次网络推理。

## 搜索配置

| 配置 | 用途 |
|---|---|
| [alphabeta.yaml](config/alphabeta.yaml) | AlphaBeta 深度与评估权重 |
| [mcts.yaml](config/mcts.yaml) | GUI 与评测，关闭根节点探索噪声，温度为 0 |
| [mcts_selfplay.yaml](config/mcts_selfplay.yaml) | 自博弈，启用根节点噪声与温度采样 |

温度为 0 时优先选择访问次数最多的动作，访问次数相同的动作间仍可能随机选择，因此不能据此承诺整局完全确定。`batch_size: 1` 为逐次叶子推理，大于 1 启用批量叶子推理。

## 棋谱

```python
from draughts.notation import GameRecord

record = GameRecord(first_player='先手方', second_player='后手方')
# 实际走子后，将对应的完整 Move 对象追加到 record.moves。
record.write('game.txt')
```

`GameRecord` 保存实际走法、日期、双方名称及结果，支持独立快照。导出使用 UTF-8 BOM 与 CRLF 换行，通过同目录临时文件写入后替换目标。格式与结果约定见 [TXT 棋谱格式](../gui/README-gui.md#txt-棋谱格式)。格式支持不代表已经实现和棋裁定或棋谱导入。

## 数据

`DraughtsJsonlDataset` 支持硬标签和 `policy_target_probs` 软标签，`collate_batch` 在 DataLoader worker 中构造张量，`dataset.validate()` 可按需校验数据。

默认会将 JSONL 记录缓存到内存。大数据集可通过训练器 `--no-cache` 或 `cache_in_memory=False` 按需读取；仍需保存样本索引并消耗批次、预取和 worker 内存，实际开销随数据量与进程数变化。

训练方法见 [训练手册](../trainer/README-trainer.md)，数据生成与检查见 [脚本手册](../scripts/README-scripts.md)。

## 测试

```bash
make draughts-test
```
