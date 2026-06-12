# BitMind-Draught100

100 格国际跳棋 AI 程序，目标是通过神经网络引导搜索，构建超越传统 AlphaBeta 搜索方法的博弈智能体。

## 文档导航

| 文档 | 内容 |
|---|---|
| [ROADMAP.md](ROADMAP.md) | **技术路线图**——三阶段路线，所有开发的指南 |
| [SETUP.md](SETUP.md) | 环境配置指南（conda/venv/pip） |
| [SPEC.md](SPEC.md) | 国际跳棋规则规格 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 项目工程架构 |
| [PROGRESS.md](PROGRESS.md) | 阶段进度与实验记录 |
| [engine/README-engine.md](engine/README-engine.md) | C 规则引擎参考 |
| [interface/README-interface.md](interface/README-interface.md) | C↔Python 接口参考 |
| [draughts/README.md](draughts/README.md) | 核心 Python 库参考 |
| [trainer/README.md](trainer/README.md) | 训练器参考 |
| [scripts/README.md](scripts/README.md) | 独立应用脚本参考 |
| [gui/README-gui.md](gui/README-gui.md) | GUI 使用说明 |

## 快速开始

```bash
# 安装依赖（只需 torch）
pip install -r requirements.txt

# 编译 C 引擎
make interface

# 运行 GUI（人类/AI 对弈）
PYTHONPATH=interface:. python3 gui/simple_gui.py

# 运行测试
make test              # C 引擎测试
make interface-test    # 接口测试
make draughts-test     # 核心库测试
make trainer-test      # 训练器测试
make scripts-test      # 脚本测试
```
