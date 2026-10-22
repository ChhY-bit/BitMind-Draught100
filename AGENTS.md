# AGENTS.md — 项目协作约定

## 项目定位

**BitMind-Draught100** 是 100 格国际跳棋 AI 与本地对弈软件，使用 `CandidateMoveNet` 引导 MCTS 搜索。当前重点为稳定运行、比赛使用和文档交付，暂不主动开展棋力提升或训练迭代。

面向使用者的入口为 [README.md](README.md) 和 [比赛操作手册](docs/competition.md)，详细目录职责与测试见 [维护与验证](docs/development.md)。

## 项目地图

| 目录或文件 | 职责 |
|---|---|
| `engine/` | C 规则引擎，棋规实现的唯一来源 |
| `interface/` | C–Python 桥接，生成 `build/libdraughts.so` |
| `draughts/` | 环境、模型、智能体、MCTS、数据、搜索取消及棋谱库 |
| `gui/` | Tkinter 界面、对局控制与 TXT 导出 |
| `scripts/` | 数据生成、重评分、检查及批量对战 |
| `trainer/` | 监督训练及配置 |
| `run/` | 并行任务与产出监控 |
| `SPEC.md` | 赛事规则、固定编号及实现边界 |
| `PROGRESS.md` | 当前功能与验证范围，链接历史评测 |
| `ROADMAP.md` | 当前维护方向与暂缓事项 |
| `docs/archive/` | 历史研究路线及代际评测 |

应用层向下依赖核心库、桥接层和引擎，不在上层重复实现棋规，核心库不反向依赖应用。

## 关键入口

| 用途 | 入口 |
|---|---|
| 安装 | `conda env create -f environment.yml`，然后 `conda activate draughts` |
| 编译 | `make interface` |
| 启动棋盘 | `PYTHONPATH=interface:. python3 gui/gui.py` |
| 完整规定测试 | `make test && make interface-test && make all-python-test` |
| 训练、数据与评测 | [训练手册](trainer/README-trainer.md)、[脚本手册](scripts/README-scripts.md) |
| MCTS 配置 | `draughts/config/mcts.yaml` 用于 GUI/评测；`mcts_selfplay.yaml` 用于自博弈 |
| 模型文件 | 约定路径 `trainer/checkpoints/round<N>/candidate_move_net_best.pt`；权重和生成数据不入 Git，通常仅在训练机上 |
| 当前网络棋力 | **须先按时效规则与用户核实。** 历史快照（2026-09-05）：r11（v3）对 r10 头对头 58%；AB depth=6 下 r10 为 87.7%、r11 为 84.1%。详见 [历史代际时间线](docs/archive/research-progress.md#自博弈代际时间线2026-09-实测)，不据此推断当前参赛模型 |

## 时效规则：棋力信息必须核实与回写

棋力数字、轮次、架构代际及锚点结论均为时效信息。

1. 涉及“当前多强／最新是哪一代／是否换代”时，先说明已有快照日期，主动向用户确认之后的新结果，不把历史数字当现状。
2. 对话出现新的 benchmark、新代晋升、架构换代、数据规模或锚点结论时，当场同步更新两处：`PROGRESS.md` 的历史评测入口所链接的代际时间线表，以及本文“当前网络棋力”快照，均注明日期。
3. 历史评测口径：头对头胜率约 55% 为晋升阈值；低于阈值视为平台期。研究恢复时再讨论表征改动，不自动重启训练计划。已有 AB 锚点为 depth=6，depth=3 已饱和；跨深度数字不可比。300 步截断局不计入决胜胜率，但需标注局数。
4. 不凭仓库中是否存在 `.pt` 或 JSONL 推断训练代际，以用户提供的信息为准；checkpoint 内的 `model_config` 用于恢复原架构。

## 使用与规则约定

- 以用户指定的赛事规程和 [SPEC.md](SPEC.md) 为准，黑方先手，棋谱使用固定 1–50 编号。
- 非法操作、不可走操作和 AI 异常保留判负确认：确定判负，取消保留局面、重新落子。
- 拒和后的 40 步流程暂缓实现，不改成无吃子自动和棋。
- TXT 导出已实现；导入、PDN 与自动恢复见维护范围，不表述为现有功能。
- 双方名称默认“先手方”“后手方”；先手胜 `1-0`，后手胜 `0-1`，按实际结果输出。
- 历史性能数字与实验待办不得作为当前能力或运行要求宣传。

## 代码约定

- Python 3.10 及以上，新文件加 `from __future__ import annotations`。
- 配置使用现有 YAML 文件与加载方式，CLI 参数覆盖默认值。
- 可复用功能放 `draughts/`，独立应用放 `scripts/`。
- `engine/` 和 `interface/` 是稳定层，只读不改。
- 不引入 PyTorch 以外的 Python 第三方库。
- 修改代码后必须运行 `make test && make interface-test && make all-python-test`，测试通过后再提交。
- 文档整理保持程序行为与默认参数不变，核对链接、命令及实际支持范围。
