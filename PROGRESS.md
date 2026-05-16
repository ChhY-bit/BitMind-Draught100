# 项目进度记录

本文档用于压缩记录当前实现状态。详细规则以 `SPEC.md` 为准，编码规范以 `AGENTS.md` 为准。

## 1. 当前结构

```text
Makefile
engine/include/draughts.h
engine/src/board.c
engine/src/move.c
engine/tests/test_board.c
engine/tests/test_moves.c
engine/tools/board_viewer.c
```

- `engine/include/draughts.h`：当前 C 引擎公共接口。
- `engine/src/board.c`：棋盘、局面状态、终局判断。
- `engine/src/move.c`：合法动作生成与动作执行。
- `engine/tests/`：C 单元测试。
- `engine/tools/board_viewer.c`：临时文本可视化工具。

## 2. 核心约定

- 棋盘为 `10x10`，只使用 50 个合理棋位。
- 对外棋位编号为 `1-50`，内部数组下标为 `0-49`。
- 编号从最上方一行开始，逐行从左到右；左上角不是合理棋位。
- 黑方初始在 `1-20`，白方初始在 `31-50`，`21-30` 为空。
- 黑方先手。黑普通子向下走，白普通子向上走。
- 普通子吃子可向四个对角方向。
- 吃子优先；多条吃子路线只保留吃子数最多者；同为最多则全部保留。
- 整着结束前，被吃棋子不从虚拟棋盘移除，仍作为屏障，且不能重复跳过。

## 3. 已实现能力

棋盘与状态：

- 坐标与棋位编号互转。
- 判断合理棋位。
- 初始化、清空、复制局面。
- 查询/设置棋子。
- 查询/设置当前行棋方。
- 非法参数、非法棋位、非法棋子、非法行棋方处理。

合法动作：

- 普通子非吃子移动。
- 普通子吃子与连跳。
- 王的远程非吃子移动。
- 王的远程吃子与连跳。
- 普通子和王的吃子统一参与最多吃子规则。
- 合法动作生成支持调用方缓冲区，缓冲区不足时返回实际动作数量。

动作执行：

- 执行普通移动、吃子、连跳。
- 移除被吃棋子。
- 执行后切换行棋方。
- 普通子最终停在对方底线时升王。
- 普通子跳吃经过底线但最终离开底线时不升王。

终局判断：

- 无黑子则白胜。
- 无白子则黑胜。
- 当前行棋方无合法动作则对方胜。
- 否则继续。
- 和棋结果枚举已预留，但和棋规则尚未实现。

## 4. 主要公共接口

当前公共接口集中在 `engine/include/draughts.h`：

```c
int draughts_init(DraughtsState *state);
int draughts_clear(DraughtsState *state);
int draughts_copy(DraughtsState *dst, const DraughtsState *src);
int draughts_get_piece(const DraughtsState *state, int square_id, DraughtsPiece *piece);
int draughts_set_piece(DraughtsState *state, int square_id, DraughtsPiece piece);
int draughts_generate_legal_moves(
    const DraughtsState *state,
    DraughtsMove *moves,
    int max_moves,
    int *move_count
);
int draughts_apply_move(DraughtsState *state, const DraughtsMove *move);
int draughts_get_game_result(const DraughtsState *state, DraughtsGameResult *result);
```

`DraughtsMove` 使用完整路径和被吃棋子列表表达普通移动、吃子和连跳。

## 5. 测试状态

测试命令：

```bash
make test
```

当前测试目标：

- `build/test_board`
- `build/test_moves`

覆盖重点：

- 棋盘编号、初始化、状态复制、查询/设置。
- 普通子移动、吃子、连跳、最多吃子。
- 王的远程移动、远程吃子、连跳、多个落点。
- 被吃棋子作为屏障。
- 动作执行、吃子移除、升王、切换行棋方。
- 终局判断。
- 缓冲区不足和非法参数。

## 6. 临时工具

```bash
build/board_viewer --ids
build/board_viewer --initial
```

该工具只用于文本预览棋位编号和初始棋盘，不是正式 GUI。

## 7. 未完成事项

- 和棋规则。
- 更严格的 `draughts_apply_move` 合法性校验，当前主要面向执行生成器产出的动作。
- Python 绑定。
- 强化学习环境封装。
- 正式 Python GUI。

## 8. 建议下一步

优先增强临时可视化/调试能力，让当前完整规则内核更容易人工验证：

- 给 `board_viewer` 增加打印当前局面的合法动作列表。
- 增加一个简单命令行演示模式，执行若干固定动作或测试局面。
- 后续再考虑 Python 绑定或正式 GUI。
