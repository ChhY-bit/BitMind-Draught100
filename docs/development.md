# 维护与验证

日常使用从 [项目入口](../README.md) 开始。本文供维护者核对模块职责、配置和测试入口。

## 目录职责

| 目录 | 职责 |
|---|---|
| `engine/` | C 规则引擎，棋规实现的唯一来源 |
| `interface/` | C 环境封装及 Python ctypes 桥接 |
| `draughts/` | 环境编码、模型、智能体、搜索、数据和棋谱等可复用能力 |
| `gui/` | 本地棋盘、对局控制、分析展示和 TXT 导出入口 |
| `scripts/` | 数据生成、重评分、检查和批量对战应用 |
| `trainer/` | 监督训练及训练配置 |
| `run/` | 多进程任务包装与监控脚本 |
| `docs/` | 操作说明、技术文档索引和历史资料 |

上层应用依赖核心库及桥接层，桥接层调用 C 引擎。规则判断不得在 Python 层重复实现；核心库不反向依赖应用脚本。

## 构建与测试

从项目根目录，在安装了 PyTorch 的 Python 环境中执行：

```bash
make interface
make test
make interface-test
make all-python-test
```

| 命令 | 范围 |
|---|---|
| `make test` | C 棋盘与走法测试 |
| `make interface-test` | C–Python 桥接测试 |
| `make draughts-test` | 核心库测试 |
| `make trainer-test` | 训练器测试 |
| `make scripts-test` | 数据和评测应用测试 |
| `make gui-test` | 无显示器的 GUI 控制测试 |
| `make all-python-test` | 核心库、训练器、脚本和 GUI 控制测试 |
| `make random-test` | 可选的 C 随机对局压力测试 |

自动测试不能替代实际比赛机器上的模型加载、CUDA、桌面操作和外部打谱软件验证。验证边界见 [功能与验证状态](../PROGRESS.md)。

## 配置与模型

| 配置 | 用途 |
|---|---|
| [model.yaml](../draughts/config/model.yaml) | 新建网络的结构和推理选项 |
| [alphabeta.yaml](../draughts/config/alphabeta.yaml) | AlphaBeta 深度与评估参数 |
| [mcts.yaml](../draughts/config/mcts.yaml) | GUI 与批量评测搜索配置 |
| [mcts_selfplay.yaml](../draughts/config/mcts_selfplay.yaml) | 自博弈探索配置 |
| [supervised.yaml](../trainer/config/supervised.yaml) | 训练超参数 |
| [arena.yaml](../scripts/config/arena.yaml) / [selfplay.yaml](../scripts/config/selfplay.yaml) | 批量应用默认参数 |

CLI 中显式传入的参数覆盖相应默认值。模型加载按 checkpoint 中保存的 `model_config` 恢复结构；不要仅凭当前 YAML 判断旧模型的架构。比赛使用的模型和搜索配置应单独留存。

`.pt` 权重、生成数据和构建产物均不随 Git 仓库分发。仓库中没有某个模型文件，不能据此判断该代模型未训练。

## 修改约定

- Python 3.10 及以上，新文件使用 `from __future__ import annotations`。
- 配置沿用现有 YAML 文件及加载方式，不引入 PyTorch 以外的 Python 第三方库。
- 可复用功能放在 `draughts/`，独立应用放在 `scripts/`。
- `engine/` 与 `interface/` 为稳定层，保持其实现与接口不变。
- 程序修改后运行相应回归及完整规定检查；文档修改核对链接、命令和实际行为，不顺带改动默认参数。
- 面向使用者的文档描述实际能力；研究数字、训练轮次和实验待办保留在 [历史目录](archive/README.md)。

协作约定统一维护于 [AGENTS.md](../AGENTS.md)。
