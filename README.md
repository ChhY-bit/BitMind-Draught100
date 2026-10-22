# BitMind-Draught100

100 格国际跳棋 AI 与本地对弈软件。提供棋盘界面、网络引导的 MCTS 搜索、对局控制与 TXT 棋谱导出，也保留训练和批量评测工具。

项目按[本赛事规程](http://computergames.caai.cn/jsgz08.html)使用黑方先手及固定 1–50 棋位编号，规则与程序支持范围见 [规则说明](SPEC.md)。当前维护重点为稳定运行、比赛操作和交付文档。

## 安装与启动

以下步骤在 Linux 下、项目根目录中执行。需要 C11 编译器、Make、Python 3.10 及以上版本；Python 外部依赖为 PyTorch。图形界面还需要 Tk 和可用的桌面显示环境。

```bash
conda env create -f environment.yml
conda activate draughts
make interface
PYTHONPATH=interface:. python3 gui/gui.py
```

已有 Python 环境时，可用 `python3 -m pip install -r requirements.txt` 安装 Python 依赖。`environment.yml` 指定 Python 3.10；Tk 是否可用需在所用环境中确认，可运行 `python3 -m tkinter` 检查。

`make interface` 生成 `build/libdraughts.so`。使用 CUDA 时，还需所安装的 PyTorch 与本机驱动相容。

## 开始对局

1. 在“新建对局”中选择双方类型：人类、随机 AI、AlphaBeta AI、监督模型 AI 或 MCTS AI。
2. 使用监督模型或 MCTS 时，选择 `.pt` 模型文件和运行设备；MCTS 可选择搜索配置，默认使用 `draughts/config/mcts.yaml`。
3. 填写棋谱中的双方名称，默认分别为“先手方”和“后手方”。黑方先手。
4. 人类方通过点击棋子和落点走棋；AI 可自动走棋，也可暂停后单步执行。
5. 对局结束后，在棋谱面板点击“导出棋谱”，保存为 TXT 文件。

模型权重与生成的训练数据不随 Git 仓库分发。使用模型 AI 前，需要另行准备已训练的模型文件；人类、随机 AI 和 AlphaBeta AI 无需模型文件。

比赛现场的设备确认、双方设置、异常判负及棋谱保存流程见 [比赛操作手册](docs/competition.md)。完整界面说明见 [GUI 使用手册](gui/README-gui.md)。

## 功能范围

| 功能 | 支持情况 |
|---|---|
| 棋规 | 强制吃子、最多吃子、完整连跳、王远程移动与跳吃、落点升王 |
| 对局 | 人机、人人、AI 对战；暂停、单步、悔棋、重开与旋转视角 |
| 设备 | 新建对局时选择 CPU 或可见 CUDA 设备，双方模型共用 |
| 异常处理 | 非法操作和 AI 异常弹出判负确认；取消后保留局面、重新落子 |
| 棋谱 | 固定编号、完整连跳路径、双方名称、实际结果、TXT 导出 |
| 分析展示 | 网络估值、走法分布、MCTS 搜索信息、估值曲线与系统监控 |

赛事中的和棋协商及拒和后的 40 步流程尚未实现，由现场按规程处理。当前不支持棋谱导入、崩溃恢复或网络对战。批量工具的最大步数是运行截断条件，不是赛事和棋规则。

## 验证

在已安装依赖的环境中执行：

```bash
make test
make interface-test
make all-python-test
```

这些命令覆盖 C 引擎、Python 接口、核心库、训练工具、批量脚本和 GUI 控制逻辑。GUI 控制测试无需显示器；比赛机器上的实际界面、设备及所用模型仍需完成现场试运行。验证范围见 [功能与验证状态](PROGRESS.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [比赛操作手册](docs/competition.md) | 赛前准备、现场操作、判负确认、棋谱保存 |
| [GUI 使用手册](gui/README-gui.md) | 界面功能、操作方式、棋谱格式与常见问题 |
| [规则说明](SPEC.md) | 赛事规则、固定棋位编号、实现边界 |
| [功能与验证状态](PROGRESS.md) | 当前支持范围及尚未完成的验证 |
| [维护范围](ROADMAP.md) | 当前维护方向和暂缓事项 |
| [技术文档索引](docs/README.md) | 模块手册、维护验证、训练与评测入口 |
| [历史资料](docs/archive/README.md) | 早期研究路线、训练代际与历史评测 |
