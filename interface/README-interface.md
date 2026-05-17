# C-Python 接口使用说明

本文档说明当前项目的 C-Python 接口如何使用。它面向刚开始接触 Python 调 C 的使用者，尽量按步骤说明。

## 1. 这个接口解决什么问题

底层规则引擎已经用 C 实现。Python 端后续要做 GUI、强化学习、自博弈和模型训练，但不应该在 Python 里重新实现复杂棋规。

所以当前接口采用下面的分工：

```text
C 语言：
  保存棋局状态
  生成合法动作
  校验动作
  执行动作
  判断终局

Python：
  发出 reset、step、legal_moves 这类宏观命令
  接收棋盘状态、合法动作列表、胜负结果
  后续负责 GUI、训练流程、模型推理
```

也就是说，Python 不判断棋规。Python 只告诉 C：“我选择第几个合法动作”，C 内部完成真正的规则处理。

## 2. 文件位置

接口相关文件都在 `interface/` 目录：

```text
interface/
  draughts_interface.h
  draughts_interface.c
  draughts_env.py
  test_interface.py
  README-interface.md
```

含义：

- `draughts_interface.c`：C 侧粗粒度环境接口。
- `draughts_interface.h`：C 侧接口声明。
- `draughts_env.py`：Python 侧包装，使用标准库 `ctypes` 调用 C 动态库。
- `test_interface.py`：Python 接口测试。
- `README-interface.md`：本文档。

## 3. 第一步：编译 C 动态库

Python 不能直接调用普通的 C 源文件。我们需要先把 C 引擎编译成 Linux 动态库。

在项目根目录执行：

```bash
make interface
```

成功后会生成：

```text
build/libdraughts.so
```

这里的 `.so` 是 Linux 下的动态库文件，大致类似 Windows 下的 `.dll`。Python 会加载这个文件，然后调用里面的 C 函数。

如果你想确认文件是否存在，可以执行：

```bash
ls build/libdraughts.so
```

## 4. 第二步：运行接口测试

推荐先运行：

```bash
make interface-test
```

它会：

1. 编译 `build/libdraughts.so`。
2. 运行 Python 测试 `interface/test_interface.py`。
3. 检查 Python 是否能成功调用 C 引擎。

如果看到类似：

```text
Ran 3 tests in 0.001s

OK
```

说明接口基本可用。

## 5. 第三步：在 Python 中使用

为了让 Python 能找到 `interface/` 目录中的模块，推荐在项目根目录这样运行 Python：

```bash
PYTHONPATH=interface python3
```

进入 Python 交互界面后，先导入环境类：

```python
from draughts_env import DraughtsEnv
```

创建环境：

```python
env = DraughtsEnv()
```

这一步会自动加载：

```text
build/libdraughts.so
```

如果这里报错说找不到动态库，请先退出 Python，然后执行：

```bash
make interface
```

## 6. reset：重置棋局

创建环境时会自动 reset。你也可以手动重置：

```python
obs = env.reset()
```

`obs` 是一个 Python 字典。你可以直接打印：

```python
print(obs)
```

其中常用字段有：

```python
obs["squares"]
obs["side_to_move"]
obs["side_to_move_name"]
obs["game_result"]
obs["game_result_name"]
obs["legal_move_count"]
```

含义：

- `squares`：长度为 50 的列表，表示 1 到 50 号棋位上的棋子。
- `side_to_move`：当前行棋方编号，`0` 是黑方，`1` 是白方。
- `side_to_move_name`：当前行棋方名称，`"black"` 或 `"white"`。
- `game_result`：当前胜负状态编号。
- `game_result_name`：当前胜负状态名称。
- `legal_move_count`：当前局面有多少个合法动作。

棋子编号含义：

```text
0: 空
1: 黑普通子
2: 黑王
3: 白普通子
4: 白王
```

初始局面下：

```python
obs = env.reset()
print(obs["side_to_move_name"])
print(obs["legal_move_count"])
```

应该看到黑方先手，并且初始合法动作数为 `9`。

## 7. 查看棋盘文本

你可以用：

```python
print(env.render_text())
```

输出一个简单文本棋盘。

符号含义：

```text
b: 黑普通子
B: 黑王
w: 白普通子
W: 白王
-: 空的合理棋位
.: 非合理棋位
```

这个功能只是方便人眼检查，不用于训练。

## 8. 查看合法动作

调用：

```python
moves = env.legal_moves()
```

`moves` 是一个列表。列表里的每个元素都是一个 `Move` 对象。

你可以这样查看：

```python
for move in moves:
    print(move)
```

初始局面可能输出类似：

```text
Move(action_id=0, path=(16, 21), captures=())
Move(action_id=1, path=(17, 21), captures=())
Move(action_id=2, path=(17, 22), captures=())
...
```

字段含义：

- `action_id`：动作编号，从 `0` 开始。
- `path`：棋子走过的完整路径。
- `captures`：被吃掉的棋子位置。

普通移动例子：

```text
path=(16, 21), captures=()
```

表示 16 号棋位的棋子走到 21 号棋位，没有吃子。

吃子或连跳例子可能长这样：

```text
path=(21, 32, 43), captures=(27, 38)
```

表示棋子从 21 跳到 32，再跳到 43，吃掉 27 和 38。

## 9. step：执行一个动作

Python 不需要自己构造复杂动作。它只需要告诉 C：“执行第几个合法动作”。

例如执行第 0 个合法动作：

```python
obs, info = env.step(0)
```

这里的 `0` 就是 `action_id`。

执行后：

```python
print(obs["side_to_move_name"])
print(obs["legal_move_count"])
print(info)
```

`obs` 是新局面的观察结果。

`info` 是这一步的结果信息，包含：

```python
info["reward"]
info["done"]
info["game_result"]
info["legal_move_count"]
```

含义：

- `reward`：从刚刚行动的一方视角看，赢棋为 `1`，输棋为 `-1`，未结束或和棋为 `0`。
- `done`：是否终局，`1` 表示已结束，`0` 表示未结束。
- `game_result`：当前胜负状态编号。
- `legal_move_count`：新局面下的合法动作数量。

注意：当前 reward 只是最基础的终局奖励。后续强化学习时，可以在 Python 环境上层继续设计更复杂的奖励。

## 10. 一个完整示例

在项目根目录执行：

```bash
make interface
PYTHONPATH=interface python3
```

进入 Python 后输入：

```python
from draughts_env import DraughtsEnv

env = DraughtsEnv()

print(env.render_text())

moves = env.legal_moves()
print("合法动作数量:", len(moves))
for move in moves:
    print(move)

obs, info = env.step(0)

print("执行第 0 个动作后:")
print(env.render_text())
print(obs)
print(info)
```

这段代码做了几件事：

1. 创建一个棋局。
2. 打印初始棋盘。
3. 打印所有合法动作。
4. 执行第 0 个合法动作。
5. 打印执行后的棋盘和结果。

## 11. 写成一个 Python 文件运行

你也可以创建自己的脚本，例如 `tmp_play.py`：

```python
from draughts_env import DraughtsEnv

env = DraughtsEnv()

for turn in range(10):
    print("turn:", turn)
    print(env.render_text())

    moves = env.legal_moves()
    if not moves:
        break

    print("choose:", moves[0])
    obs, info = env.step(moves[0].action_id)

    if info["done"]:
        print("game over:", obs["game_result_name"])
        break
```

运行：

```bash
PYTHONPATH=interface python3 tmp_play.py
```

这个脚本每回合都选择第一个合法动作，只用于验证接口，不代表智能下棋。

## 12. 为什么不用 Python 自己判断动作是否合法

这是当前接口设计的重点。

Python 端不要做这些事：

```text
判断普通子能不能走
判断王能不能吃
判断是否必须吃子
判断是否满足最多吃子
执行吃子移除
判断升王
```

这些规则全部由 C 引擎完成。

Python 端只做：

```text
拿到 observation
拿到 legal_moves
选择 action_id
调用 step(action_id)
读取结果
```

这样可以充分利用 C 语言执行规则逻辑的效率，也能避免 Python 和 C 两边规则不一致。

## 13. 常见问题

### 找不到 `libdraughts.so`

错误类似：

```text
Cannot find C shared library
```

解决：

```bash
make interface
```

并确认你是在项目根目录运行 Python。

### 找不到 `draughts_env`

错误类似：

```text
ModuleNotFoundError: No module named 'draughts_env'
```

解决：运行 Python 时加上 `PYTHONPATH=interface`：

```bash
PYTHONPATH=interface python3
```

或者：

```bash
PYTHONPATH=interface python3 your_script.py
```

### 执行非法 action_id

如果你执行：

```python
env.step(999)
```

而当前没有这么多合法动作，会抛出异常：

```text
RuntimeError
```

正确做法是先查看：

```python
moves = env.legal_moves()
```

然后选择其中某个 `move.action_id`。

## 14. 当前限制

- 目前是单局环境对象，不是批量并行环境。
- 目前 observation 是简单 Python 字典，还不是神经网络最方便使用的张量。
- reward 只提供基础终局奖励。
- 和棋规则仍未实现。

这些限制不影响当前阶段验证 C-Python 调用链。后续进入强化学习时，可以在这个接口之上继续封装更适合训练的环境。

## 15. Python API 速查表

### 主要类

| 名称 | 类型 | 用途 |
| --- | --- | --- |
| `DraughtsEnv` | 类 | Python 侧棋局环境。内部持有 C 环境对象，负责调用 C 引擎。 |
| `Move` | 数据类 | 表示一个当前局面的合法动作，包含动作编号、路径和吃子列表。 |

### 创建环境

| 写法 | 返回值 | 说明 |
| --- | --- | --- |
| `env = DraughtsEnv()` | `DraughtsEnv` 对象 | 创建一个棋局环境，并自动加载 `build/libdraughts.so`，同时初始化为标准开局。 |
| `env = DraughtsEnv("路径/libdraughts.so")` | `DraughtsEnv` 对象 | 手动指定 C 动态库路径。一般不需要这样写。 |

### `DraughtsEnv` 方法

| 方法 | 返回值 | 常用场景 | 说明 |
| --- | --- | --- | --- |
| `env.reset()` | `dict` | 开始新棋局 | 重置为标准初始局面，并返回 observation。 |
| `env.observation()` | `dict` | 查看当前局面 | 返回当前棋盘、行棋方、终局状态和合法动作数量。 |
| `env.legal_move_count()` | `int` | 快速查看动作数量 | 返回当前局面的合法动作数量。 |
| `env.legal_moves()` | `list[Move]` | 选择动作前查看可选动作 | 返回当前所有合法动作。 |
| `env.step(action_id)` | `(dict, dict)` | 执行动作 | 执行编号为 `action_id` 的合法动作，返回新 observation 和 info。 |
| `env.clone()` | `DraughtsEnv` | 复制环境 | 通过 C 侧结构体复制当前局面和合法动作缓存，常用于搜索或试走。 |
| `env.render_text()` | `str` | 人眼检查棋盘 | 返回一个简单文本棋盘，可用 `print(env.render_text())` 打印。 |

### `Move` 字段

| 字段 | 类型 | 示例 | 说明 |
| --- | --- | --- | --- |
| `move.action_id` | `int` | `0` | 当前局面下的动作编号。传给 `env.step(action_id)` 使用。 |
| `move.path` | `tuple[int, ...]` | `(21, 32, 43)` | 棋子经过的完整棋位路径。 |
| `move.captures` | `tuple[int, ...]` | `(27, 38)` | 被吃掉的棋子位置。普通移动时为空元组 `()`。 |
| `move.is_capture` | `bool` | `True` | 是否是吃子动作。 |

### Observation 字典字段

| 字段 | 类型 | 示例 | 说明 |
| --- | --- | --- | --- |
| `obs["squares"]` | `list[int]` | 长度 50 | 1 到 50 号棋位上的棋子。访问 1 号棋位用 `squares[0]`。 |
| `obs["side_to_move"]` | `int` | `0` | 当前行棋方编号，`0` 黑方，`1` 白方。 |
| `obs["side_to_move_name"]` | `str` | `"black"` | 当前行棋方名称。 |
| `obs["game_result"]` | `int` | `0` | 当前终局状态编号。 |
| `obs["game_result_name"]` | `str` | `"ongoing"` | 当前终局状态名称。 |
| `obs["legal_move_count"]` | `int` | `9` | 当前局面合法动作数量。 |

### Info 字典字段

`info` 来自：

```python
obs, info = env.step(action_id)
```

| 字段 | 类型 | 示例 | 说明 |
| --- | --- | --- | --- |
| `info["reward"]` | `int` | `0` | 从刚刚行动的一方视角看，赢为 `1`，输为 `-1`，未结束或和棋为 `0`。 |
| `info["done"]` | `int` | `0` | 是否终局，`1` 是，`0` 否。 |
| `info["game_result"]` | `int` | `0` | 执行动作后的终局状态编号。 |
| `info["legal_move_count"]` | `int` | `9` | 执行动作后的合法动作数量。 |

### 常量含义

| 数值 | 棋子含义 |
| --- | --- |
| `0` | 空棋位 |
| `1` | 黑普通子 |
| `2` | 黑王 |
| `3` | 白普通子 |
| `4` | 白王 |

| 数值 | 行棋方含义 |
| --- | --- |
| `0` | 黑方 |
| `1` | 白方 |

| 数值 | 终局状态 |
| --- | --- |
| `0` | 对局继续 |
| `1` | 黑方胜 |
| `2` | 白方胜 |
| `3` | 和棋，当前规则暂未实际使用 |

### 最常用模板

```python
from draughts_env import DraughtsEnv

env = DraughtsEnv()

obs = env.reset()
print(obs["side_to_move_name"])

moves = env.legal_moves()
for move in moves:
    print(move.action_id, move.path, move.captures)

chosen = moves[0]
obs, info = env.step(chosen.action_id)

print(obs["side_to_move_name"])
print(info["reward"], info["done"])
```
