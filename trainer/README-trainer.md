# 训练环境说明

`trainer/` 目录当前只实现最基础的标准化环境，还没有开始神经网络训练。它的目标是把 `interface.DraughtsEnv` 包装成后续训练代码更容易使用的形式。

## 当前文件

```text
trainer/
  env.py
  test_env.py
  README-trainer.md
```

- `env.py`：标准化训练环境 `StandardDraughtsEnv`。
- `test_env.py`：环境测试。
- `README-trainer.md`：本文档。

## 运行测试

在项目根目录执行：

```bash
make interface
PYTHONPATH=interface:trainer python3 -m unittest trainer/test_env.py
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

- 还没有 PyTorch 张量输出；后续可以把嵌套列表转换为 tensor。
- 当前 action id 是“当前局面合法动作列表编号”，不是全局固定动作空间。
- 还没有批量环境。
- 还没有自博弈采样器。
- reward 仍是最简单的终局奖励。

## 后续方向

建议后续依次实现：

1. `agents.py`：随机、贪心、材料评估等基础 agent。
2. `arena.py`：批量对战评测器。
3. observation 到 PyTorch tensor 的转换。
4. 候选动作打分模型或固定动作空间设计。
5. 搜索与强化学习流程。
