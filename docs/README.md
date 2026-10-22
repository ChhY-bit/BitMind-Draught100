# 文档索引

## 使用与比赛

| 文档 | 用途 |
|---|---|
| [项目入口](../README.md) | 安装、启动及主要功能 |
| [比赛操作手册](competition.md) | 赛前准备、现场对局、异常处理与棋谱保存 |
| [GUI 使用手册](../gui/README-gui.md) | 详细界面操作与棋谱格式 |
| [规则说明](../SPEC.md) | 赛事规则、棋位编号及实现边界 |
| [功能与验证状态](../PROGRESS.md) | 当前支持范围与验证限制 |
| [维护范围](../ROADMAP.md) | 当前维护方向与暂缓事项 |

## 技术参考

| 文档 | 用途 |
|---|---|
| [维护与验证](development.md) | 目录职责、构建、测试与修改约定 |
| [核心 Python 库](../draughts/README-draughts.md) | 环境、模型、智能体、搜索、数据及棋谱 API |
| [C 规则引擎](../engine/README-engine.md) | 底层数据类型、规则 API、调试工具 |
| [C–Python 接口](../interface/README-interface.md) | 桥接接口、观察值和走法数据 |
| [训练手册](../trainer/README-trainer.md) | JSONL 训练、配置与模型文件 |
| [脚本手册](../scripts/README-scripts.md) | 数据生成、重评分、批量对战与检查 |
| [批量任务手册](../run/README-run.md) | 并行运行、参数覆盖与产出监控 |

训练和批量评测工具属于技术参考，不是启动比赛界面的前置步骤。

## 历史资料

[归档索引](archive/README.md)集中保留早期研究路线和历史评测。现行操作说明与历史记录冲突时，以使用手册、规则说明及实际程序为准。
