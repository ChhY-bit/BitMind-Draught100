# Draughts Engine — C 规则引擎参考手册

`engine/` 是 100 格国际跳棋的底层 C 规则引擎，也是整个项目中**所有棋规逻辑的唯一真源**。Python 训练器与 GUI 均通过 `interface/` 共享库调用此引擎，不在上层重复实现吃子、升王、强制吃子等规则。

## 架构原则

- **纯规则黑盒**：引擎只负责局面维护、合法动作生成/校验/执行、终局判断。不包含搜索、评估、哈希等 AI 逻辑。
- **严格校验**：所有公开函数对 `NULL` 指针、越界参数、非法枚举值均返回错误码；`draughts_apply_move` 拒绝任何不合法的动作，不会产生无效局面。
- **无全局状态**：所有状态保存在 `DraughtsState` 结构体中，调用者自行管理内存。引擎完全可重入。

## 编译说明

引擎本身不产出独立的可执行程序——它被编译为共享库 `build/libdraughts.so`，供 Python 上层通过 `interface/` 加载。大多数使用者只需在项目根目录执行：

```bash
make interface
```

这会自动编译引擎代码（含 `-fPIC`）并链接为共享库。**无需单独编译引擎测试或调试工具。**

以下 target 仅在开发/修改引擎代码时需要：

| 命令 | 产物 | 用途 |
|---|---|---|
| `make test` | `build/test_board`, `build/test_moves` | 运行单元测试 |
| `make random-test` | `build/random_play` | 随机对局压力测试 |
| `make tools` | `build/board_viewer` | 交互式调试工具 |

所有 `make` 命令需在**项目根目录**（`Makefile` 所在目录）下执行。

## 目录结构

```
engine/
  include/draughts.h       公共头文件（类型、常量、API 声明）
  src/board.c               棋盘：初始化、坐标映射、终局判断
  src/move.c                走法：生成、校验、执行、升变
  tests/test_board.c        棋盘单元测试
  tests/test_moves.c        走法单元测试（27 个测试函数）
  tests/random_play.c       随机对局压力测试
  tools/board_viewer.c      交互式调试工具（9 个预设场景）
```

## 数据类型

### 枚举

| 类型 | 值 | 说明 |
|---|---|---|
| `DraughtsResult` | `DRAUGHTS_OK` (0), `DRAUGHTS_ERR_NULL` (-1), `DRAUGHTS_ERR_INVALID_SQUARE` (-2), … `DRAUGHTS_ERR_INVALID_MOVE` (-8) | 函数返回码，0 表示成功 |
| `DraughtsPiece` | `DRAUGHTS_EMPTY` (0), `DRAUGHTS_BLACK_MAN` (1), `DRAUGHTS_BLACK_KING` (2), `DRAUGHTS_WHITE_MAN` (3), `DRAUGHTS_WHITE_KING` (4) | 棋子类型 |
| `DraughtsSide` | `DRAUGHTS_BLACK` (0), `DRAUGHTS_WHITE` (1) | 行棋方 |
| `DraughtsGameResult` | `DRAUGHTS_GAME_ONGOING` (0), `DRAUGHTS_GAME_BLACK_WIN` (1), `DRAUGHTS_GAME_WHITE_WIN` (2), `DRAUGHTS_GAME_DRAW` (3) | 终局状态 |

### 结构体

**`DraughtsState`** — 局面状态

| 字段 | 类型 | 说明 |
|---|---|---|
| `squares[50]` | `unsigned char` | 50 个可玩棋位（1-based 编号），值为 `DraughtsPiece` |
| `side_to_move` | `unsigned char` | 当前行棋方，值为 `DraughtsSide` |

**`DraughtsMove`** — 一个走法

| 字段 | 类型 | 说明 |
|---|---|---|
| `path[32]` | `unsigned char` | 依次经过的棋位编号 |
| `path_len` | `unsigned char` | 路径长度（安静走法为 2，吃子为 `capture_count + 1`） |
| `captures[20]` | `unsigned char` | 被吃棋子所在棋位 |
| `capture_count` | `unsigned char` | 吃子数量（安静走法为 0） |

### 常量

| 常量 | 值 | 说明 |
|---|---|---|
| `DRAUGHTS_BOARD_ROWS` | 10 | 棋盘行数 |
| `DRAUGHTS_BOARD_COLS` | 10 | 棋盘列数 |
| `DRAUGHTS_SQUARE_COUNT` | 50 | 可玩棋位数量（仅黑格） |
| `DRAUGHTS_MAX_MOVE_PATH` | 32 | 单着最大路径长度 |
| `DRAUGHTS_MAX_MOVE_CAPTURES` | 20 | 单着最大吃子数 |

## API 参考

### 坐标映射

棋位编号为 1–50，按行自顶向下、每行从左到右排列。坐标 `(row, col)` 为 0-based，可玩棋位满足 `(row + col) % 2 == 1`。

```c
int draughts_is_playable_coord(int row, int col);
int draughts_square_to_coord(int square_id, int *row, int *col);
int draughts_coord_to_square(int row, int col, int *square_id);
```

### 局面生命周期

```c
int draughts_init(DraughtsState *state);        // 标准初始局面，黑方先手
int draughts_clear(DraughtsState *state);       // 清空所有棋子，黑方先手
int draughts_copy(DraughtsState *dst, const DraughtsState *src);  // 深拷贝
```

`draughts_init` 将黑子置于棋位 1–20、白子置于 31–50，中间 10 个棋位为空。

### 棋子与行棋方

```c
int draughts_get_piece(const DraughtsState *state, int square_id, DraughtsPiece *piece);
int draughts_set_piece(DraughtsState *state, int square_id, DraughtsPiece piece);
int draughts_get_side_to_move(const DraughtsState *state, DraughtsSide *side);
int draughts_set_side_to_move(DraughtsState *state, DraughtsSide side);
char draughts_piece_to_char(DraughtsPiece piece);  // '.' 'b' 'B' 'w' 'W' '?'
```

### 走法生成与执行

```c
int draughts_generate_legal_moves(
    const DraughtsState *state,
    DraughtsMove *moves,
    int max_moves,
    int *move_count
);
```

生成当前行棋方的所有合法走法。**强制吃子优先**：若有任何吃子走法，安静走法（无吃子）不会被生成。所有吃子走法按最多吃子规则过滤。若 `max_moves` 不足容纳全部走法，返回 `DRAUGHTS_ERR_BUFFER_TOO_SMALL` 并在 `*move_count` 中记录总数。

```c
int draughts_is_legal_move(
    const DraughtsState *state,
    const DraughtsMove *move,
    int *is_legal
);
```

逐一比对候选走法与当前合法走法集合，精确匹配则 `*is_legal = 1`。本函数内部分配 512 个走法的缓冲区，调用者无需预生成。

```c
int draughts_apply_move(DraughtsState *state, const DraughtsMove *move);
```

执行走法：移除被吃棋子 → 清空起点 → 放置棋子到终点（兵到达底线时自动升变为王） → 切换行棋方。**严格校验**：对走法形状、行棋方、起点棋子、终点空格、被吃棋子逐一检查，任一不合法即返回错误且不修改局面。

### 终局判断

```c
int draughts_get_game_result(const DraughtsState *state, DraughtsGameResult *result);
```

判断逻辑：
1. 一方无子 → 对方胜
2. 当前行棋方无合法走法 → 对方胜
3. 否则 → 对局进行中

## 规则实现清单

| 规则 | 状态 |
|---|---|
| 普通子前进一步（安静走法） | ✅ |
| 王远程移动（四方向，遇子停止） | ✅ |
| 普通子吃子（四方向，单步） | ✅ |
| 普通子连跳（递归搜索所有路径） | ✅ |
| 王远程吃子（多落点，递归搜索） | ✅ |
| 王连跳 | ✅ |
| 强制吃子（有吃必吃） | ✅ |
| 最多吃子规则（所有路线按吃子数过滤） | ✅ |
| 被吃棋子作为屏障（整着结束前不可跨越） | ✅ |
| 升王（兵停在对方底线） | ✅ |
| 路过底线不升变（连跳中途经过底线不停留） | ✅ |
| 三次重复局面 | ❌ （暂不实现） |
| 40 步无吃子/无升变和棋 | ❌ （暂不实现） |

## 构建与测试

从项目根目录执行：

```bash
make test          # 编译并运行单元测试（test_board + test_moves）
make tools         # 构建调试工具 board_viewer
make random-test   # 运行随机对局压力测试（默认 1000 局）
make clean         # 清理构建产物
```

**随机压力测试**可自定义参数：

```bash
build/random_play [局数] [最大半回合数] [日志间隔] [随机种子]
# 示例：
build/random_play 10000 300 2000 20260517
```

输出示例：

```
random-play: 2000/10000 games ok, black=995 white=1005 draw=0 max-plies=0, avg-plies=92.39, elapsed=1.593s
```

随机种子固定即可复现。日常回归建议种子 `1`，扩大覆盖时换多个种子分别跑。

## 调试工具：board_viewer

```bash
make tools

# 查看棋位编号
build/board_viewer --ids

# 查看初始局面
build/board_viewer --initial

# 列出所有预设场景
build/board_viewer --scenarios

# 查看单个场景
build/board_viewer --scenario king-multi

# 显示场景并自动应用第一个合法走法
build/board_viewer --apply promotion

# 交互式对局器（输入走法编号执行，r 刷新，q 退出）
build/board_viewer --play initial

# 按脚本走子（逗号分隔的走法编号序列）
build/board_viewer --script initial 9,4

# 依次显示所有场景
build/board_viewer --all-scenarios
```

### 预设场景

| 场景名 | 说明 |
|---|---|
| `initial` | 标准初始局面 |
| `man-capture` | 强制吃子（有吃子时安静走法被抑制） |
| `man-multi` | 普通子连跳 |
| `king-quiet` | 王远程移动 |
| `king-capture` | 王吃子（多落点） |
| `king-multi` | 王连跳 |
| `max-capture` | 兵和王共享最多吃子规则 |
| `promotion` | 升变 |
| `terminal` | 有棋子但无合法走法（终局） |
