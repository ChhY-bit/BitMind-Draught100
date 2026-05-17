# 项目进度记录

本文档只保留阶段性摘要，详细规则以 `SPEC.md` 为准，工程约束以 `AGENTS.md` 为准。

## 当前阶段

C 底层引擎已形成可测试、可调试的第一版规则内核。实现集中在 `engine/`：

- `include/draughts.h`：公共接口与基础数据结构。
- `src/board.c`：棋盘编号、局面状态、状态复制、终局判断。
- `src/move.c`：合法动作生成、动作严格校验、动作执行。
- `tests/test_board.c`、`tests/test_moves.c`：单元测试。
- `tests/random_play.c`：随机对局压力测试。
- `tools/board_viewer.c`：临时文本棋盘、固定场景与简易命令行对局器。
- `interface/`：C-Python 接口层，包含 C 侧粗粒度环境封装和 Python `ctypes` 包装。
- `gui/simple_gui.py`：Python GUI v0，人-人模式，点击棋子与落点执行动作。
- `trainer/env.py`：训练用标准化环境，包装 C-Python 接口并输出固定 observation 与 action mask。

## 已实现规则

- 100 格国际跳棋棋盘，外部棋位编号为 `1-50`，内部数组为 `0-49`。
- 初始局面：黑方 `1-20`，白方 `31-50`，黑方先手。
- 普通子：非吃子只能向前，吃子可四向，支持连跳。
- 王：支持远程普通移动、远程吃子、多个落点与连跳。
- 吃子强制；所有普通子和王共同参与“最多吃子”规则；同为最多的路线全部保留。
- 被吃棋子在整着结束前仍作为屏障，且不能重复跳过。
- 普通子最终停在对方底线才升王；跳吃经过底线但离开时不升王。
- `draughts_apply_move` 已改为严格执行当前合法动作，不接受只满足结构但违反规则的动作。
- 终局判断已支持无子、当前方无合法动作；和棋枚举保留，但和棋规则暂不实现。

## 主要接口

公共接口见 `engine/include/draughts.h`。当前核心接口包括：

- `draughts_init`、`draughts_clear`、`draughts_copy`
- `draughts_get_piece`、`draughts_set_piece`
- `draughts_get_side_to_move`、`draughts_set_side_to_move`
- `draughts_generate_legal_moves`
- `draughts_is_legal_move`
- `draughts_apply_move`
- `draughts_get_game_result`

`DraughtsMove` 使用完整路径 `path[]` 和被吃棋子列表 `captures[]` 表示普通移动、吃子与连跳。

Python 侧优先使用 `interface.DraughtsEnv`，通过 `reset()`、`legal_moves()`、`step(action_id)` 和 `observation()` 调用 C 环境接口。规则逻辑仍封闭在 C 内部。

GUI 侧通过同一个 Python 接口推进棋局。当前已实现人-人、人-随机AI、随机AI-人、随机AI-随机AI模式；启动和新建对局时可选择双方决策源。人类方使用点击棋子、显示落点、点击落点或双击动作列表执行；随机AI方从当前合法动作中随机选取。左侧显示棋谱，右侧含中文状态、合法动作和彩色日志框，王棋统一显示为 `K`。GUI 已支持悔棋，方式是重置 C 环境并重放动作历史。

训练侧已新增 `StandardDraughtsEnv`：提供 `reset()`、`step(action_id)`、`observation()`、`legal_actions()`、`action_mask()`。当前 observation 为 `6x10x10` 嵌套列表，通道分别为当前方普通子、当前方王、对方普通子、对方王、合法棋位、当前行棋方平面；action mask 长度为 `512`。

## 测试状态

常用命令：

```bash
make test
make random-test
make interface-test
make trainer-test
build/random_play 10000 300 2000 1
```

测试覆盖棋盘编号、初始化、复制、终局、普通子移动/吃子/连跳、王移动/吃子/连跳、最多吃子、屏障、升王、严格动作校验、非法参数与随机对局。

随机测试曾发现并修复一个重要边界：王连跳可以合法回到出发格，执行器不能因起点等于终点而拒绝该动作。

## 临时调试工具

`board_viewer` 可查看编号、初始局面、固定测试场景、执行场景首个动作、批量场景，以及进入简易命令行对局器。它只是 C 端临时调试工具，不是正式 GUI。

常用命令：

```bash
make tools
build/board_viewer --ids
build/board_viewer --scenarios
build/board_viewer --scenario king-multi
build/board_viewer --play initial
```

Python GUI：

```bash
make interface
PYTHONPATH=interface python3 gui/simple_gui.py
```

## 未完成事项

- 基础 agent 与批量评测器。
- GUI 的模型AI接入。
- 更大规模的复杂残局回归测试。
- 和棋规则暂按当前讨论结果忽略。

## 建议下一步

可以继续实现 `trainer/agents.py` 与 `trainer/arena.py`，建立随机、贪心、材料评估等基础 agent 及批量对战评测能力。
