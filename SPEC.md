# 项目规格说明

## 1. 游戏规则
- 棋盘：国际跳棋的棋盘为一10×10的黑白相间的方格棋盘，实际上并不一定是白色和黑色的格子，只要它们能够对应白色和的黑色就可以。棋盘放在对弈双方的中间，每个玩家的右下角应该是白色格子
- 棋子：黑白双方各有20个扁圆柱形的棋子，通过掷硬币决定谁是黑方；
- 棋位：黑格为合理棋位，棋位统一编码如图2所示；
- 开局：开局时黑白双方的棋子各摆在棋盘靠近自己一方的4行黑格当中（见图3）。总是黑方先手，然后双方轮流走动自己一方的棋子；
- 目标：在整个对弈过程中，白格子是用不到的。棋子自始至终都是在黑格子中沿对角线方向移动和停止。对弈的目标是将对方所有的棋子吃掉或者形成一个局面逼使对方棋子不能移动；
- 跳吃：只要对角线方向邻近的黑格内有对方的棋子并且再过去的黑格是空位，就可以跳过对方的棋子并将对方棋子吃掉；
- 如果没有跳吃的着法，那就只能沿对角线方向前移一格；
- 加冕：任何一个棋子到达了对方底线便立刻加冕，从此以后便成为“王”。这时应在升王的棋子上面再放一个棋子，以便和普通棋子区别；
- 连跳：跳吃可以由多次跳吃组成，如果具备连续跳吃的条件，则必须连续跳吃。除非不再具备跳吃的条件，才可以结束跳吃；
- 未加冕的棋子只能向前移动，但是在跳吃或者连续跳吃的时候可以是向前、向后或者前后组合；
- 只有停止在对方底线上的棋子才能加冕。所以，如果一个棋子在跳吃过程中行进到底线又离开了底线，最后没有停止在底线上，则该棋子不能升王；
- 王可以在对角线方向上移动任意多个空格。同样在跳吃的时候，王可以跳过对方棋子前后任意数量的空格。因此王比一般棋子要强大和珍贵。不过一般棋子是可以吃掉王的。
- 当某一着法结束之后才将吃掉的棋子从棋盘上移出，任何被吃掉的棋子虽然还没有从棋盘上移出也不许再跳经该棋子。也就是说被吃掉的棋子形成了屏障；
- 跳吃的时候，在具有多种选择的情况下，必须选择吃子最多的着法。如果不止一个棋子或者不止一个路线可以跳吃对方同样最多的棋子，玩家可以自主选择哪个棋子或者向哪个方向行进。
- 对弈过程中经双方同意可以和棋。如果一方拒绝和棋，则该方需要在后续的40步内获胜，或者明确地显示出优势。对于国际跳棋和棋是经常的，特别是在高水平的对弈中。


## 2. 阶段性实现记录

本节用于记录当前已经确认并实现的工程约定。后续继续开发时，应优先保持这些约定兼容；如确需调整，应同步修改本文件并说明原因。

### 2.1 棋盘编号约定

- 棋盘为 `10×10`。
- 只使用 50 个合理棋位。
- 对外棋位编号为 `1-50`。
- 内部数组下标为 `0-49`，对应对外编号 `1-50`。
- 编号从棋盘最上方一行开始，逐行从左到右。
- 左上角格子不是合理棋位。
- 第一行合理棋位位于第 2、4、6、8、10 列，对应编号 `1-5`。
- 第二行合理棋位位于第 1、3、5、7、9 列，对应编号 `6-10`。
- 后续逐行交替。

编号视图如下，`.` 表示非合理棋位：

```text
  .  1  .  2  .  3  .  4  .  5
  6  .  7  .  8  .  9  . 10  .
  . 11  . 12  . 13  . 14  . 15
 16  . 17  . 18  . 19  . 20  .
  . 21  . 22  . 23  . 24  . 25
 26  . 27  . 28  . 29  . 30  .
  . 31  . 32  . 33  . 34  . 35
 36  . 37  . 38  . 39  . 40  .
  . 41  . 42  . 43  . 44  . 45
 46  . 47  . 48  . 49  . 50  .
```

### 2.2 双方方向与初始局面

- 黑方位于棋盘上方，初始占据 `1-20`。
- 白方位于棋盘下方，初始占据 `31-50`。
- `21-30` 初始为空。
- 黑方先手。
- 黑普通子非吃子移动方向为向下，即行号增大方向。
- 白普通子非吃子移动方向为向上，即行号减小方向。
- 普通子吃子时可向四个对角方向跳吃，不限前后。

### 2.3 当前 C 引擎文件结构

当前已经创建的 C 引擎相关文件：

```text
Makefile
engine/include/draughts.h
engine/src/board.c
engine/src/move.c
engine/tests/test_board.c
engine/tests/test_moves.c
engine/tools/board_viewer.c
```

各文件用途：

- `Makefile`：统一编译、测试、构建临时工具。
- `engine/include/draughts.h`：C 引擎当前公共接口。
- `engine/src/board.c`：棋盘编号、状态初始化、查询、设置、复制等基础逻辑。
- `engine/src/move.c`：合法动作生成逻辑。
- `engine/tests/test_board.c`：棋盘基础逻辑单元测试。
- `engine/tests/test_moves.c`：动作生成单元测试。
- `engine/tools/board_viewer.c`：临时文本棋盘查看器，不是正式 GUI。

### 2.4 当前公共数据结构

当前公共头文件为 `engine/include/draughts.h`。

核心常量：

```c
#define DRAUGHTS_BOARD_ROWS 10
#define DRAUGHTS_BOARD_COLS 10
#define DRAUGHTS_SQUARE_COUNT 50
#define DRAUGHTS_MAX_MOVE_PATH 32
#define DRAUGHTS_MAX_MOVE_CAPTURES 20
```

当前棋子类型：

```c
typedef enum {
    DRAUGHTS_EMPTY = 0,
    DRAUGHTS_BLACK_MAN = 1,
    DRAUGHTS_BLACK_KING = 2,
    DRAUGHTS_WHITE_MAN = 3,
    DRAUGHTS_WHITE_KING = 4
} DraughtsPiece;
```

当前行棋方：

```c
typedef enum {
    DRAUGHTS_BLACK = 0,
    DRAUGHTS_WHITE = 1
} DraughtsSide;
```

当前局面结构：

```c
typedef struct {
    unsigned char squares[DRAUGHTS_SQUARE_COUNT];
    unsigned char side_to_move;
} DraughtsState;
```

当前动作结构：

```c
typedef struct {
    unsigned char path[DRAUGHTS_MAX_MOVE_PATH];
    unsigned char path_len;
    unsigned char captures[DRAUGHTS_MAX_MOVE_CAPTURES];
    unsigned char capture_count;
} DraughtsMove;
```

动作结构约定：

- `path[]` 保存完整移动路径。
- 普通非吃子动作的 `path_len == 2`，`capture_count == 0`。
- 吃子或连跳动作的 `path[]` 保存每次落点序列。
- `captures[]` 保存被吃棋子的编号。
- `capture_count` 保存本动作吃子数量。

### 2.5 当前已实现的公共函数

棋盘与状态相关：

```c
int draughts_is_playable_coord(int row, int col);
int draughts_square_to_coord(int square_id, int *row, int *col);
int draughts_coord_to_square(int row, int col, int *square_id);
int draughts_init(DraughtsState *state);
int draughts_clear(DraughtsState *state);
int draughts_copy(DraughtsState *dst, const DraughtsState *src);
int draughts_get_piece(const DraughtsState *state, int square_id, DraughtsPiece *piece);
int draughts_set_piece(DraughtsState *state, int square_id, DraughtsPiece piece);
int draughts_get_side_to_move(const DraughtsState *state, DraughtsSide *side);
int draughts_set_side_to_move(DraughtsState *state, DraughtsSide side);
char draughts_piece_to_char(DraughtsPiece piece);
```

合法动作生成：

```c
int draughts_generate_legal_moves(
    const DraughtsState *state,
    DraughtsMove *moves,
    int max_moves,
    int *move_count
);
```

`draughts_generate_legal_moves` 当前行为：

- 生成当前行棋方的合法动作。
- 当前支持普通子的非吃子移动。
- 当前支持普通子的吃子和连跳。
- 如果存在吃子动作，只返回吃子动作。
- 如果存在多条吃子路线，只保留吃子数最多的路线。
- 如果多条路线同为最多吃子数，全部保留。
- `move_count` 返回实际合法动作数量。
- 如果 `max_moves` 不足，返回 `DRAUGHTS_ERR_BUFFER_TOO_SMALL`，并通过 `move_count` 返回实际需要的动作数量。
- 当 `max_moves == 0` 时，允许 `moves == NULL`，用于只查询动作数量。

当前错误码：

```c
typedef enum {
    DRAUGHTS_OK = 0,
    DRAUGHTS_ERR_NULL = -1,
    DRAUGHTS_ERR_INVALID_SQUARE = -2,
    DRAUGHTS_ERR_INVALID_COORD = -3,
    DRAUGHTS_ERR_INVALID_PIECE = -4,
    DRAUGHTS_ERR_INVALID_SIDE = -5,
    DRAUGHTS_ERR_BUFFER_TOO_SMALL = -6,
    DRAUGHTS_ERR_INVALID_ARGUMENT = -7
} DraughtsResult;
```

### 2.6 普通子吃子实现约定

当前已经实现普通子的跳吃规则：

- 普通子非吃子只能向前移动一格。
- 普通子吃子可以向前、向后、前后组合。
- 普通子跳吃目标必须满足：
  - 相邻对角合理棋位上是对方棋子。
  - 再后一格合理棋位为空。
- 连跳通过递归搜索完整路线。
- 搜索只返回已经无法继续跳吃的完整动作。
- 某一着法结束前，被吃棋子不会从虚拟棋盘移除。
- 已经被吃的棋子仍视为屏障。
- 同一路线中不允许再次跳过同一个已经被吃的棋子。
- 当前移动棋子的原起点在连跳搜索中视为空。
- 当前移动棋子的当前位置视为被己方棋子占据。
- 普通子可以吃王。

### 2.7 当前测试覆盖

当前可通过以下命令运行测试：

```bash
make test
```

当前测试目标：

- `test_board`：棋盘基础逻辑测试。
- `test_moves`：动作生成测试。

已覆盖内容包括：

- 棋位编号与坐标转换。
- 非合理棋位识别。
- 初始局面黑 20、白 20、空 10。
- 黑方先手。
- 查询、设置、清空、复制局面。
- 当前行棋方设置与查询。
- 初始局面黑方普通走法数量。
- 初始局面白方普通走法数量。
- 普通移动的边界处理。
- 当前行棋方过滤。
- 普通移动被己方或对方棋子占位阻挡。
- 缓冲区不足处理。
- 必吃规则。
- 黑普通子向后吃。
- 白普通子向后吃。
- 连跳路径与被吃棋子列表。
- 最多吃子规则。
- 多条同最大吃子路线全部保留。
- 己方棋子不能被跳吃。
- 落点非空不能吃。
- 被吃棋子在连跳中保留为屏障。

### 2.8 临时文本可视化工具

当前已有临时 C 文本查看器：

```bash
build/board_viewer --ids
build/board_viewer --initial
```

用途：

- `--ids`：输出棋位编号视图。
- `--initial`：输出初始棋盘棋子视图。

该工具只是开发调试辅助，不是正式 GUI。

### 2.9 当前未实现事项

以下内容尚未实现：

- 王的普通移动。
- 王的吃子。
- 动作执行。
- 执行动作后的吃子移除。
- 升王。
- 终局判断。
- 和棋规则。
- Python 绑定。
- 强化学习环境封装。
- 正式 Python GUI。
