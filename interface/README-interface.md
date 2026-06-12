# Interface Layer — C↔Python 绑定参考手册

`interface/` 是 C 引擎与 Python 上层之间的桥接层。它编译出 `libdraughts.so` 共享库，Python 通过 `ctypes` 加载并调用其中的 C 函数。上层训练器与 GUI 只与本层的 `DraughtsEnv` 交互，不直接接触 C 引擎 API。

## 架构

```
engine/                              interface/
  draughts.h ──┐                       draughts_interface.h    C 环境封装声明
  board.c      │                       draughts_interface.c    C 环境封装实现
  move.c       ├──► build/libdraughts.so ◄── draughts_env.py   Python ctypes 包装
                │                       test_interface.py      接口测试
                └────────────────────── README-interface.md    本文档
```

Python 侧的原则：**不自行判断棋规**。只通过 `action_id` 选择走法并传给 C 层执行，规则逻辑完全封闭在引擎中。

## 快速开始

```bash
# 编译共享库并运行接口测试
make interface-test
```

成功输出：

```
Ran 5 tests in ...s
OK
```

仅编译不测试：

```bash
make interface
# 生成 build/libdraughts.so
```

## C 侧概览

### 结构体设计

`DraughtsEnvHandle` 是 C 侧的核心运行时对象，缓存了底层状态和当前合法走法：

```c
typedef struct {
    DraughtsState state;                              // 引擎局面
    DraughtsMove legal_moves[512];                    // 缓存合法走法，避免重复生成
    int legal_move_count;
    int game_result;
} DraughtsEnvHandle;
```

合法走法在 `reset`、`step`、`set_state` 时通过内部 `refresh_env()` 刷新并缓存。后续查询（`legal_moves`、`legal_move_count`）和走子（`step`）均直接操作缓存，不重复调用 `draughts_generate_legal_moves`。

### 导出函数（7 个）

| 函数 | 说明 |
|---|---|
| `draughts_env_reset` | 初始局面 + 刷新缓存 |
| `draughts_env_copy` | 整块结构体拷贝（状态 + 走法缓存） |
| `draughts_env_set_state` | 加载任意局面（squares + side_to_move）并刷新 |
| `draughts_env_get_observation` | 快照：棋盘、行棋方、终局状态、合法走法数 |
| `draughts_env_get_legal_move_count` | 读缓存数 |
| `draughts_env_get_legal_move` | 按 action_id 读单个缓存走法 |
| `draughts_env_step` | 校验 → 执行走法 → 刷新 → 返回 reward/done |

`reward` 计算规则：从走子方视角，胜 `1`，负 `-1`，未结束或和棋 `0`。这是基础终局奖励，后续训练器可在上层设计更复杂的奖励。

## Python API 参考

### 类与类型

| 名称 | 类型 | 说明 |
|---|---|---|
| `DraughtsEnv` | 类 | 棋局环境，封装 C 句柄 |
| `Move` | frozen dataclass | 不可变走法对象 |

### `DraughtsEnv` 方法

| 方法 | 返回值 | 说明 |
|---|---|---|
| `__init__()` | — | 创建环境，自动加载 `build/libdraughts.so`，立即 `reset()` |
| `__init__(path)` | — | 手动指定 `.so` 路径 |
| `reset()` | `dict` | 重置为标准初始局面 |
| `set_state(squares, side)` | `dict` | 从 50 格棋子数组和行棋方加载任意局面 |
| `observation()` | `dict` | 当前局面快照（不走子） |
| `legal_move_count()` | `int` | 合法走法数量 |
| `legal_moves()` | `list[Move]` | 所有合法走法列表 |
| `step(action_id)` | `(dict, dict)` | 执行走法，返回 `(observation, info)` |
| `clone()` | `DraughtsEnv` | 深拷贝环境（用于搜索树的试走） |
| `render_text()` | `str` | ASCII 棋盘，调试用。训练器和 GUI 不使用此方法 |

### `Move` 字段

| 字段 | 类型 | 示例 | 说明 |
|---|---|---|---|
| `action_id` | `int` | `0` | 走法编号，传给 `step()` |
| `path` | `tuple[int, ...]` | `(21, 32, 43)` | 棋位路径 |
| `captures` | `tuple[int, ...]` | `(27, 38)` | 被吃棋子棋位；无吃子为空 |
| `is_capture` | `bool` | `True` | 是否为吃子走法 |

### Observation 字段

`reset()`、`set_state()`、`observation()` 和 `step()` 均返回此字典：

| 字段 | 类型 | 说明 |
|---|---|---|
| `squares` | `list[int]` (长度 50) | 1–50 号棋位上的棋子值 |
| `side_to_move` | `int` | 当前行棋方，`0` 黑 `1` 白 |
| `side_to_move_name` | `str` | `"black"` 或 `"white"` |
| `game_result` | `int` | 终局状态编号 |
| `game_result_name` | `str` | `"ongoing"` / `"black_win"` / `"white_win"` / `"draw"` |
| `legal_move_count` | `int` | 合法走法数量 |

### Info 字段

`step()` 的第二个返回值：

| 字段 | 类型 | 说明 |
|---|---|---|
| `reward` | `int` | 走子方视角终局奖励：`1` / `-1` / `0` |
| `done` | `int` | `1` 已终局，`0` 未结束 |
| `game_result` | `int` | 终局状态编号 |
| `legal_move_count` | `int` | 新局面合法走法数 |

### 常量

**棋子值**

| 值 | 含义 |
|---|---|
| `0` | 空 |
| `1` | 黑普通子 |
| `2` | 黑王 |
| `3` | 白普通子 |
| `4` | 白王 |

**行棋方**

| 值 | 含义 |
|---|---|
| `0` | 黑方 |
| `1` | 白方 |

**终局状态**

| 值 | 含义 |
|---|---|
| `0` | 进行中 |
| `1` | 黑胜 |
| `2` | 白胜 |
| `3` | 和棋（引擎暂未实现此规则） |

## 完整示例

```python
from draughts_env import DraughtsEnv

env = DraughtsEnv()

# 查看初始局面
obs = env.reset()
print(f"行棋方: {obs['side_to_move_name']}")
print(f"合法走法数: {obs['legal_move_count']}")

# 列出所有走法
moves = env.legal_moves()
for move in moves:
    print(f"  {move.action_id}: path={move.path} captures={move.captures}")

# 执行第一个走法
obs, info = env.step(moves[0].action_id)
print(f"新局面行棋方: {obs['side_to_move_name']}")
print(f"reward={info['reward']} done={info['done']}")

# 克隆环境，独立走子
clone = env.clone()
clone.step(clone.legal_moves()[0].action_id)
# env 不受 clone 影响
```

## 已知限制

- 单局环境，不支持批量并行
- observation 为 Python dict，训练器需通过 `StandardDraughtsEnv` 转为张量
- reward 仅提供基础终局奖励
- 和棋规则未实现（继承自引擎层）
