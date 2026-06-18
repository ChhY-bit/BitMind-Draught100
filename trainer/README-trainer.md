# Trainer — 训练器参考

`trainer/` 包含神经网络训练脚本。

```bash
PYTHONPATH=interface:. python3 trainer/<script>.py [args]
```

---

## `train_supervised.py` — 监督/蒸馏训练

从 JSONL 数据训练 `CandidateMoveNet`。

```bash
PYTHONPATH=interface:. python3 trainer/train_supervised.py data \
  --resume trainer/checkpoints/candidate_move_net_latest.pt \
  --checkpoint-dir trainer/checkpoints --device cpu
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data` | path(s) | (必填) | 一个或多个 JSONL 文件或目录 |
| `--config` | path | `trainer/config/supervised.yaml` | 训练超参数配置 |
| `--model-config` | path | `draughts/config/model.yaml` | 网络结构配置 |
| `--checkpoint-dir` | path | `trainer/checkpoints` | checkpoint 保存目录 |
| `--resume` | path | none | 从此 checkpoint 继续训练 |
| `--device` | str | auto | 训练设备：`cpu`、`cuda` |
| `--epochs` | int | (YAML 中的值) | 覆盖配置文件中的训练轮数 |
| `--batch-size` | int | (YAML 中的值) | 覆盖配置文件中的批次大小 |
| `--lr` | float | (YAML 中的值) | 覆盖配置文件中的学习率 |
| `--save-every-epoch` | flag | off | 每个 epoch 都保存 checkpoint |
| `--source` | str | none | 数据源权重：`路径:policy_w:value_w:标签`（可重复） |
| `--source-config` | path | none | 数据源权重 YAML 配置文件 |

### 损失函数

训练器自动检测数据中的软标签（`policy_target_probs` 字段）。有软标签时使用软交叉熵（KL 散度），无软标签时回退为硬标签交叉熵，完全向后兼容。

```
# 软标签（MCTS 自博弈 / search_improve 产出）
policy_loss = -sum(policy_target_probs * log_softmax(policy_logits))

# 硬标签（普通 selfplay 产出）
policy_loss = CrossEntropy(policy_logits, policy_target_index)

value_loss  = MSE(value, value_target)
loss        = policy_loss + value_loss_weight * value_loss
```

### 训练数据来源

| 来源 | 产生方式 | 标签类型 |
|------|---------|---------|
| `selfplay.py` (AlphaBeta) | AB 对局，直接记录 | 硬标签 |
| `search_improve.py` | AB depth=3 重新评分 | 软标签 |
| `selfplay.py` (MCTS) | MCTS 自博弈，visit count 分布 | 软标签 |

### Checkpoint 输出

| 文件 | 说明 |
|---|---|
| `candidate_move_net_latest.pt` | 每个 epoch 覆盖，用于续训 |
| `candidate_move_net_best.pt` | 验证集 loss 最低时覆盖 |
| `candidate_move_net_epoch_<N>.pt` | 训练结束保存最终 epoch |
| `training_log_<key>.csv` | 逐 epoch 的 loss、lr、耗时 |
| `training_time.json` | 累计训练时间 |

### 数据源权重

`--source` 格式：`路径:policy_weight:value_weight:标签`

```bash
PYTHONPATH=interface:. python3 trainer/train_supervised.py \
  --source data/alphabeta_selfplay.jsonl:1.0:1.0:alphabeta \
  --source data/search_improved.jsonl:1.0:0.5:improved \
  --checkpoint-dir trainer/checkpoints --device cpu
```

`policy_weight=0` 时该来源只训 value，`value_weight=0` 时只训 policy。也可用 `--source-config` 指定 YAML 配置文件，参见 `trainer/config/data_sources.yaml`。

### 配置参数（`config/supervised.yaml`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `epochs` | 3 | 训练轮数 |
| `batch_size` | 64 | 批次大小 |
| `learning_rate` | 0.001 | 初始学习率 |
| `value_loss_weight` | 0.25 | value loss 权重 |
| `validation_split` | 0.1 | 验证集比例 |
| `seed` | 4 | 随机种子 |
| `shuffle` | 1 | 是否打乱训练样本（1=是, 0=否） |
| `log_interval` | 20 | 每隔多少 batch 打印日志 |
| `num_workers` | 0 | DataLoader 子进程数 |

以上参数均可被命令行 `--epochs`、`--batch-size`、`--lr` 覆盖。
