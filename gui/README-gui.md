# GUI 参考手册

`gui/simple_gui.py` 是 100 格国际跳棋的 Tkinter 图形界面。它通过 `interface.DraughtsEnv` 调用 C 引擎，不在 GUI 中重实现棋规。

## 快速开始

在项目根目录执行：

```bash
make interface
PYTHONPATH=interface:. python3 gui/simple_gui.py
```

启动后弹出"新建对局"窗口，选择黑白双方决策源即可开始。

## 功能

- 三方对局：人类、随机AI、监督模型AI 任意组合
- 棋盘显示含棋位编号（支持 180° 旋转视角）
- 点击棋子 → 高亮合法落点 → 点击落点执行
- 多路线歧义时侧边栏显示候选动作，双击确认
- 左侧棋谱面板
- 侧边栏：行棋方、合法动作数、终局状态、最近动作
- 合法动作列表（表格视图）
- 彩色日志（信息/动作/警告/错误/终局）
- 按钮：新建对局、重开本局、悔棋、旋转棋盘
- AI 控制：暂停/继续自动走棋、手动触发单步
- 选择监督模型时弹出 `.pt/.pth` 文件选择框

## 操作

1. 启动 → 选择双方决策源
2. 人类方：点击棋子选中 → 点击高亮落点执行
3. 双击右侧动作列表可指定完整路线（王的多落点歧义）
4. 点击 `悔棋` 撤销上一步
5. 点击 `旋转棋盘` 切换 180° 视角
6. 点击 `暂停AI` / `继续AI` 控制 AI 自动走棋

## 架构

```
gui/simple_gui.py
    │
    ├── interface/DraughtsEnv  (C 引擎封装)
    ├── draughts.agents.SupervisedAgent
    ├── draughts.env (encode_observation, EncodedObservationAdapter)
    └── draughts.move_encoding (square_to_coord, coord_to_square)
```

GUI 不实现棋规——所有动作通过 `action_id → env.step(action_id)` 交给 C 引擎执行。悔棋使用动作历史重放实现：重置环境 → 重放除最后一步外的所有 `action_id`。

## 限制

- 监督模型仅用 policy logits 选最高分动作，不显示动作概率或 value
- 不支持棋局保存/读取
- 使用点击选择而非拖拽
