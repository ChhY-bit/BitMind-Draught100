# GUI 参考手册

`gui/simple_gui.py` 是 100 格国际跳棋的 Tkinter 图形界面。它通过 `interface.DraughtsEnv` 调用 C 引擎，不在 GUI 中重实现棋规。

## 快速开始

在项目根目录执行：

```bash
make interface
PYTHONPATH=interface:. python3 gui/simple_gui.py
```

启动后弹出"新建对局"窗口，选择黑白双方决策源即可开始。

## 决策源

| 选项 | 说明 | 需选文件 |
|------|------|---------|
| 人类 | 鼠标点击走子 | — |
| 随机AI | 随机选择合法走法 | — |
| AlphaBeta AI | Alpha-Beta depth=2 搜索 | — |
| 监督模型AI | 加载 `CandidateMoveNet`，greedy 推理 | `.pt` checkpoint |
| MCTS AI | MCTS + 网络引导搜索 | `.pt` checkpoint + 可选 `.yaml` 配置 |

MCTS AI 选择 checkpoint 后会弹出可选的 YAML 配置文件选择框（默认使用 `draughts/config/mcts.yaml`）。

## 功能

### 对局交互
- 七种决策源任意组合
- 棋盘显示含棋位编号，支持 180° 旋转视角
- 点击棋子 → 高亮合法落点 → 点击落点执行
- 多路线歧义时侧边栏显示候选动作，双击确认
- 左侧棋谱面板
- 侧边栏：行棋方、合法动作数、终局状态、最近动作
- 合法动作列表（表格视图）
- 彩色日志（信息/动作/警告/错误/终局）
- 按钮：新建对局、重开本局、悔棋、旋转棋盘
- AI 控制：暂停/继续自动走棋、手动触发单步

### 引擎分析可视化（v1.8 新增）

当对局中至少一方为 AI（监督模型或 MCTS）时，界面会实时展示神经网络的决策过程：

**评估条（Eval Bar）**
- 棋盘左侧的竖条，直观显示当前局面的胜率预估
- 上方深色区域 = 黑方胜率，下方浅色区域 = 白方胜率
- 金色分割线指示网络估值，中央虚线为均势参考线
- 百分比标签显示当前行棋方的预估胜率
- 数据来源：网络的 Value Head 输出（tanh ∈ [-1, 1]）

**走法概率列**
- 合法动作列表中新增"概率"列
- 显示每个走法被 AI 选中的概率（百分比）
- 最优走法以黄色高亮标记
- 候选走法（已选棋子的合法走法）以绿色高亮
- 数据来源：MCTS 的 visit count 分布（MCTS AI）或 softmax(policy_logits)（监督模型 AI）

**引擎分析面板**
- 位于侧边栏的"引擎分析"区域
- 估值（Value）：当前局面网络估值，含黑方胜率百分比
- 搜索量（Search）：MCTS 模拟次数和 c_puct 参数（或 AlphaBeta 搜索深度）
- 首选（Best Move）：AI 最倾向的走法及其概率
- 人类对局或无 AI 参与时显示 "—"

### 估值曲线（v1.10 新增）

棋谱面板下方的折线图，实时追踪对局估值变化。

- **数据来源**：对局中至少一方为 AI（监督模型或 MCTS）时，每步走完后自动推理当前局面的估值
- **混合采集**：AI 决策时免费记录估值；人类走棋后用 AI 的模型补推理一次，人机对弈也能获得完整曲线
- **Y 轴**：黑方胜率 0%–100%，50% 线为均势参考虚线
- **X 轴**：步数（ply），横向滚动条支持长对局
- **数据点颜色**：黑优段深色 → 均势中性灰 → 白优段浅色
- **当前步标记**：金色竖虚线指示最新估值位置

**交互**：
- 点击曲线上任意数据点 → 自动暂停对弈并回溯到对应步数
- 回溯后暂停按钮变红色，点击"继续AI"可从此位置继续
- 最大保留 300 个点（约 150 步对局），超出的旧数据自动丢弃
- 纯人机对弈（无 AI 参与）时曲线显示 "无 AI 参与"

### 引擎分析面板增强（v1.12）

引擎分析面板从 3 列扩展为 4 列，新增**搜索修正**列：显示网络原始估值 → MCTS 搜索后的修正估值，展示搜索是否"纠正"了网络的直觉。仅 MCTS AI 走棋时有效；其余情况显示 "—"。

### 系统监控（v1.12 新增）

棋盘下方的实时硬件资源监控面板：

- **左侧 CP​U 侧**：CPU 总利用率、系统内存、进程 CPU
- **右侧 GPU 侧**：GPU 利用率、显存占用、GPU 温度
- 每行：中文标签 + 水平条形图（绿→黄→红渐变）+ 实时数值
- 条形图带动态过渡动画（250ms ease-out）
- **无新依赖**：CPU/内存读 `/proc`，GPU 读 `nvidia-smi` + `torch.cuda`
- 每秒刷新，始终运行
- 无 GPU 时右侧显示 "无 GPU"

### 暂停按钮颜色提示（v1.10 新增）

- **绿色**：AI 正常运行中（可暂停）
- **红色**：AI 已暂停（可恢复）

### 数据来源说明

| AI 类型 | 估值来源 | 概率来源 |
|---------|---------|---------|
| MCTS AI | 网络根节点 Value Head 输出 | MCTS visit count 归一化分布 |
| 监督模型 AI | 网络 Value Head 输出 | softmax(policy_logits) |
| AlphaBeta AI | —（仅显示搜索深度） | — |
| 随机 AI / 人类 | — | — |

## 界面布局

```
┌────────────┬──────┬───────────┬─────────────────────┐
│  棋谱      │ 评   │           │  对局状态            │
│ (Listbox)  │ 估   │  棋盘     │  ─────────           │
│            │ 条   │ (Canvas)  │  引擎分析            │
│ ───────── │      ├───────────┤  ─────────           │
│  日志      │      │ 引擎分析  │  [按钮工具栏]         │
│ (Text)     │      │ (4 列)   │  [AI 控制]           │
│            │      ├───────────┤  ─────────           │
│            │      │ 系统监控  │  合法动作 (Tree)      │
│            │      │ (Canvas)  │  ─────────           │
│            │      │           │  估值曲线            │
└────────────┴──────┴───────────┴─────────────────────┘
```

## 操作

1. 启动 → 选择双方决策源（需 checkpoint 的 AI 会弹出文件选择框）
2. 人类方：点击棋子选中 → 点击高亮落点执行
3. 双击右侧动作列表可指定完整路线（王的多落点歧义）
4. 点击"悔棋"撤销上一步
5. 点击"旋转棋盘"切换 180° 视角
6. 点击"暂停AI"/"继续AI"控制 AI 自动走棋（绿色=运行中，红色=已暂停）
7. 点击工具栏右侧的语言按钮（"English"/"中文"）切换界面语言
8. 点击估值曲线上的数据点 → 自动暂停并回溯到对应步数

## 语言切换（v1.9 新增）

界面支持中文/英文即时切换，无需重启。点击工具栏右侧的语言切换按钮即可。

- 所有静态标签（窗口标题、按钮文字、表格列标题、状态面板等）立即更新
- 日志消息、状态变量、分析面板数据跟随当前语言显示
- 旧日志不回溯翻译，新日志使用当前语言
- 对话框（新建对局）每次打开时使用当前语言
- 翻译数据集中在 `_TRANSLATIONS` 字典中，新增字符串只需添加一条包含 `zh`/`en` 的条目

## 架构

```
gui/simple_gui.py
    │
    ├── interface/DraughtsEnv              (C 引擎封装)
    ├── draughts.env.StandardDraughtsEnv   (MCTS 搜索用)
    ├── draughts.agents.*                  (MCTSAgent, SupervisedAgent, AlphaBetaAgent)
    ├── draughts.env.encode_observation    (局面编码)
    ├── draughts.mcts.MCTSResult           (搜索分析数据)
    └── draughts.move_encoding             (坐标转换)
```

GUI 不实现棋规——所有动作通过 `action_id → env.step(action_id)` 交给 C 引擎执行。悔棋使用动作历史重放实现。

分析数据的传递路径：
```
Agent.select_action()
  → agent.last_value / agent.last_policy_logits / agent.last_result
  → Player.last_analysis (PlayerAnalysis dataclass)
  → GameController.last_player_analysis()
  → DraughtsApp.update_eval_bar() / update_move_list() / update_search_stats()
```

## 中文字体（Linux）

Linux 下 Conda 自带的 Tk 通常是 **noxft** 编译版本，无法使用系统安装的现代 Unicode 字体（如 Noto Sans CJK），只能看到少量 X11 核心位图字体。GUI 启动时会自动检测并适配：

1. 尝试 fontconfig 路径（Xft Tk）
2. 尝试 fontconfig 注册捆绑的 `DroidSansFallbackFull.ttf`
3. **（新增）** 尝试通过 X11 font path 加载系统黑体（文泉驿微米黑、Noto Sans CJK），这些无衬线字体比 X11 核心宋体圆滑得多
4. 尝试 X11 font path 注册捆绑的 `DroidSansFallbackFull.ttf`
5. 回退到 X11 核心宋体（`song ti`）

**推荐的根治方案**：将 Conda 的 Tk 替换为 Xft 版本，这样能直接使用系统所有 CJK 字体，渲染质量最佳：

```bash
conda install -c conda-forge "tk=8.6.13=xft_*"
```

可通过一行命令自查当前环境：

```bash
echo "会话:$XDG_SESSION_TYPE Tk字体:$(python3 -c 'import tkinter;r=tkinter.Tk();n=len(r.tk.call(\"font\",\"families\"));r.destroy();print(n)')个 Xft:$(ldd $(python3 -c 'import tkinter;print(tkinter._tkinter.__file__)') 2>/dev/null|grep -cq libXft&&echo 是||echo 否)"
```

- **Xft=是 + 字体数 >200**：现代字体路径正常，CJK 渲染完美
- **Xft=否 + 字体数 ~60**：noxft Tk，GUI 自动回退到 X11 位图字体

字体注册通过 `xset +fp` 实现（依赖 `xfonts-utils`），仅在当前 X 会话有效，注销后自动还原。

## 限制

- 仅支持单机对局，不支持网络对战
- 不支持棋局保存/读取
- 使用点击选择而非拖拽
- 和棋规则（三次重复、40 步）未实现，靠 max_plies 截断
