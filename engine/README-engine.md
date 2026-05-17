# C 引擎说明

`engine/` 目录实现 100 格国际跳棋的底层 C 规则引擎。它的职责是维护局面、生成合法动作、校验并执行动作、判断终局，并提供测试与临时调试工具。Python 训练器和后续 GUI 应尽量复用这里的规则内核，避免在 Python 端重复实现棋规。

## 目录结构

```text
engine/
  include/draughts.h
  src/board.c
  src/move.c
  tests/test_board.c
  tests/test_moves.c
  tests/random_play.c
  tools/board_viewer.c
```

- `include/draughts.h`：公共头文件，定义棋盘常量、棋子类型、局面结构、动作结构和接口函数。
- `src/board.c`：棋位编号、坐标转换、初始化、清空、复制、查询/设置棋子、当前行棋方、终局判断。
- `src/move.c`：合法动作生成、严格动作合法性判断、动作执行、升王和吃子移除。
- `tests/`：单元测试和随机对局测试。
- `tools/board_viewer.c`：临时文本调试工具，可查看棋盘、场景和进行简单命令行对局。

## 当前能力

引擎目前支持普通子移动、普通子吃子与连跳、王的远程移动、王的远程吃子与连跳、强制吃子、最多吃子、被吃棋子作为屏障、升王、严格动作校验和基础终局判断。

和棋规则暂未实现。当前项目约定中，和棋规则先忽略。

## 编译

在项目根目录执行：

```bash
make all
```

这会编译并运行单元测试，同时构建临时调试工具。

只编译/运行测试：

```bash
make test
```

只构建调试工具：

```bash
make tools
```

清理构建产物：

```bash
make clean
```

## 单元测试

运行：

```bash
make test
```

当前会执行：

- `build/test_board`
- `build/test_moves`

覆盖内容包括棋盘编号、初始局面、状态复制、终局判断、普通子与王的移动/吃子/连跳、最多吃子、屏障、升王、动作严格校验和非法参数。

## 随机对局测试

默认运行：

```bash
make random-test
```

等价于构建并运行：

```bash
build/random_play
```

默认参数：

```text
1000 局，每局最多 300 个半回合，每 200 局打印一次日志，随机种子为 1
```

也可以手动指定参数：

```bash
build/random_play [局数] [每局最大半回合数] [日志间隔] [随机种子]
```

常用示例：

```bash
build/random_play 10000 300 2000 1
build/random_play 100000 300 10000 1
build/random_play 10000 300 2000 20260517
```

日志示例：

```text
random-play: 2000/10000 games ok, black=995 white=1005 draw=0 max-plies=0, avg-plies=92.39, elapsed=1.593s
```

字段含义：

- `games ok`：已成功完成的对局数量。
- `black` / `white`：黑胜和白胜局数。
- `draw`：和棋局数；当前和棋规则未实现，通常为 0。
- `max-plies`：达到最大半回合数仍未结束的局数。
- `avg-plies`：平均每局半回合数。
- `elapsed`：累计耗时。

如果出现 `random-play error`，说明随机测试发现了生成、校验、执行或状态一致性问题，应保留完整命令和日志用于复现。固定随机种子的好处是可复现；日常回归建议使用种子 `1`，扩大覆盖时再换多个种子分别跑。

## 临时调试工具

构建：

```bash
make tools
```

查看棋位编号：

```bash
build/board_viewer --ids
```

查看初始局面：

```bash
build/board_viewer --initial
```

列出固定场景：

```bash
build/board_viewer --scenarios
```

查看某个场景：

```bash
build/board_viewer --scenario king-multi
```

进入简易命令行对局器：

```bash
build/board_viewer --play initial
```

在对局器中，根据打印出的动作编号输入数字即可执行对应动作。输入 `r` 可刷新，输入 `q` 可退出。

## 后续衔接

下一阶段进入 Python 端时，建议先围绕当前 C 接口设计最小绑定层，让训练器和 GUI 都通过同一套 C 引擎获取合法动作、执行动作和判断终局。
