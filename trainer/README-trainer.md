# 训练手册

`trainer/train_supervised.py` 从 JSONL 训练 `CandidateMoveNet`，支持硬标签、软标签及按来源设置损失权重。训练工具保留供技术使用，不是启动比赛界面的必要步骤。

历史实验与代际数据见 [历史资料](../docs/archive/README.md)。本手册不指定“最新一代”模型。

## 基本使用

从项目根目录执行，将数据路径替换为实际文件或目录：

```bash
PYTHONPATH=interface:. python3 trainer/train_supervised.py data/train.jsonl \
  --no-cache --device cpu --checkpoint-dir trainer/checkpoints/my_model
```

使用 CUDA 时显式指定 `--device cuda:0` 或其他可见设备。继续已有训练可传入 `--resume path/to/checkpoint.pt`；新一轮独立训练不传 `--resume`。

## 常用参数

| 参数 | 说明 |
|---|---|
| `data` | 一个或多个 JSONL 文件或目录；使用数据源参数时可省略 |
| `--config` | 训练配置，默认 [supervised.yaml](config/supervised.yaml) |
| `--model-config` | 新建网络配置，默认 [model.yaml](../draughts/config/model.yaml) |
| `--checkpoint-dir` | 输出目录，默认 `trainer/checkpoints` |
| `--resume` | 从指定 checkpoint 继续训练 |
| `--device` | `cpu`、`cuda` 或 `cuda:N`；未指定时自动选择 |
| `--epochs` / `--batch-size` / `--lr` / `--seed` | 覆盖对应训练配置 |
| `--no-cache` | 关闭 JSONL 全量内存缓存，按需读取 |
| `--save-every-epoch` | 保存每个 epoch 的独立模型文件；默认仅保留最终 epoch 的独立文件 |
| `--source` | 重复指定带损失权重的数据源 |
| `--source-config` | 数据源配置文件 |

完整参数以 `PYTHONPATH=interface:. python3 trainer/train_supervised.py --help` 为准。训练轮数、batch、学习率、验证比例和 worker 数统一查阅配置文件，避免在文档中维护第二套默认值。

## 损失与数据源

有 `policy_target_probs` 时使用软交叉熵；否则使用 `policy_target_index` 的硬标签交叉熵。Value 使用均方误差，总损失结合 `value_loss_weight` 及各数据源权重。

```text
policy_loss = -sum(target_probs * log_softmax(policy_logits))
value_loss  = MSE(value, value_target)
```

| 数据来源 | 常见策略标签 |
|---|---|
| 普通 AlphaBeta 自博弈 | 实际走法的硬标签 |
| AlphaBeta 搜索重评分 | 分数转换后的软标签 |
| MCTS 自博弈 | 搜索访问次数分布软标签 |

数据源参数格式为 `PATH:POLICY_WEIGHT:VALUE_WEIGHT[:NAME]`：

```bash
PYTHONPATH=interface:. python3 trainer/train_supervised.py \
  --source data/ab.jsonl:1.0:1.0:alphabeta \
  --source data/improved.jsonl:1.0:0.5:improved \
  --no-cache --device cpu --checkpoint-dir trainer/checkpoints/my_model
```

策略权重为 0 时该来源只参与 value 损失，value 权重为 0 时只参与策略损失。配置示例见 [data_sources.yaml](config/data_sources.yaml)。

## 模型文件与日志

| 文件 | 内容 |
|---|---|
| `candidate_move_net_latest.pt` | 每个 epoch 更新，用于续训 |
| `candidate_move_net_best.pt` | 有验证集时保留验证损失最优模型；无验证集时随 epoch 更新 |
| `candidate_move_net_epoch_XXX.pt` | 最终 epoch 的独立文件；启用选项后每轮均保存 |
| `training_log_<key>.csv` | 按配置间隔记录训练与验证指标、时间等信息 |
| `training_time.json` | 训练时间记录 |

checkpoint 保存 `model_config`，加载已有 v1/v2/v3 模型时恢复其原始结构。当前 [model.yaml](../draughts/config/model.yaml) 为新建网络提供 v3 配置，不据此推断已有参赛模型或最新训练代际。

权重和训练数据被 Git 忽略，交付或迁移时需另行携带。训练损失最优与比赛效果是不同指标，应按实际用途另行评估。

## 内存与运行设置

默认会缓存全部 JSONL 记录；较大的数据集使用 `--no-cache`。关闭缓存后仍需样本索引、worker 和预取内存，不能视为零内存开销。按机器资源设置配置中的 `num_workers` 和 batch。

`model.yaml` 的 `use_fp16`、`use_compile` 供推理使用，不控制这里的训练流程。推理速度和编译等待时间随模型、设备及环境变化，比赛使用的选项应先在目标机器验证。

## 测试

```bash
make trainer-test
```
