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
  --no-cache \
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
| `--device` | str | auto | 训练设备：`cpu`、`cuda`（= `cuda:0`）、`cuda:1` |
| `--epochs` | int | (YAML 中的值) | 覆盖配置文件中的训练轮数 |
| `--batch-size` | int | (YAML 中的值) | 覆盖配置文件中的批次大小 |
| `--lr` | float | (YAML 中的值) | 覆盖配置文件中的学习率 |
| `--no-cache` | flag | off | **推荐始终开启**。不将 JSONL 全部加载到内存，按需从磁盘逐行读取。MCTS 自博弈数据通常数 GB，不加此参数极易 OOM。 |
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
  --no-cache \
  --source data/alphabeta_selfplay.jsonl:1.0:1.0:alphabeta \
  --source data/search_improved.jsonl:1.0:0.5:improved \
  --checkpoint-dir trainer/checkpoints --device cpu
```

`policy_weight=0` 时该来源只训 value，`value_weight=0` 时只训 policy。也可用 `--source-config` 指定 YAML 配置文件，参见 [trainer/config/data_sources.yaml](config/data_sources.yaml)。

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
| `num_workers` | 4 | DataLoader 子进程数（GPU 推荐 4–8，CPU 设为 0） |

以上参数均可被命令行 `--epochs`、`--batch-size`、`--lr` 覆盖。

### 网络结构（`draughts/config/model.yaml`）

当前使用 **v2 结构**，参数量约 **15M**。各参数含义：

| 参数 | v1（旧） | v2（当前） | 说明 |
|---|---|---|---|
| `conv_channels` | 64 | 256 | 每层卷积通道数。越大 → 棋盘模式识别越精细 |
| `conv_layers` | 3 | 3 | 卷积层数。3 层 = 7×7 感受野，10×10 棋盘的合理上限 |
| `board_feature_dim` | 256 | 512 | 棋盘全局特征维度。参数大头（256×100×512 ≈ 13M），不宜超过 1024 |
| `move_embedding_dim` | 128 | 256 | 走法嵌入维度 |
| `hidden_dim` | 128 | 256 | Score/Value head 隐层维度 |

#### 数据量与模型规模匹配

| 参数量 | 推荐数据量 | 说明 |
|--------|-----------|------|
| ~2M（v1） | 0.5M–1M | 旧版默认，1.5M 以上边际收益递减 |
| ~15M（v2） | **3M–5M** | 当前推荐。1.5M 可跑但未饱和 |

> **现有数据（1.4M–1.5M）可直接用于 v2 训练**，先验证架构有效性（arena vs 旧版），再用 v2 网络做 MCTS 自博弈扩大数据量至 3M+。

#### 推理加速（`use_fp16` / `use_compile`）

`model.yaml` 提供两个推理加速开关，仅影响 MCTS 搜索和网络推理，**不影响训练**：

| 参数 | 默认 | 效果 | 首次延迟 |
|------|------|------|---------|
| `use_fp16` | `false` | FP16 混合精度，约 1.5-2× 加速 | 无 |
| `use_compile` | `false` | `torch.compile` 算子熔合，约 20-40% 加速 | 2-5 秒编译 |

**使用建议**：

| 场景 | `use_fp16` | `use_compile` | 说明 |
|------|-----------|--------------|------|
| 自博弈（大量产数据） | `true` | `true` | 最快速度，编译延迟可忽略 |
| 评测（arena） | `true` | `true` | 评测时间缩短 |
| **GUI 实战** | `true` | **`false`** | 避免第一步的编译延迟影响体验 |
| 纯 CPU | `false` | `false` | CPU 不支持 FP16 autocast |

> **关闭方法**：在 `draughts/config/model.yaml` 中将对应项设为 `false` 即可。两个开关互相独立，可单独开启。
>
> **叠加效果**：FP16 + compile + 400 sims ≈ v1 800 sims 的产出速度，同时保持 v2 的数据质量。

#### 跨版本兼容

checkpoint 文件内保存了 `model_config`，arena、GUI、selfplay 加载时自动按各自的结构建网，**不同结构的 checkpoint 可同台对战**。无需为旧 checkpoint 保留旧 YAML。

### 数据加载

训练管线做了以下优化以最大化 GPU 利用率：

| 优化项 | 说明 |
|--------|------|
| 内存缓存 | 小数据集（< 500 MB）可省略 `--no-cache`，全量预加载到内存避免磁盘 I/O。大数据集务必使用 `--no-cache`，按需从磁盘逐行读取，内存占用几乎可忽略 |
| Worker 内 tensor 化 | `collate_batch` 在 DataLoader worker 进程中直接构建 `torch.Tensor`，通过共享内存传 batch，消除 pickle 瓶颈 |
| `pin_memory=True` | 加速 CPU→GPU 数据传输 |
| `prefetch_factor=4` | 每个 worker 预取 4 个 batch，GPU 始终有数据待命 |

> **`--no-cache` 是大多数场景的正确选择。** 不开启内存缓存时，训练速度几乎不受影响（瓶颈在 GPU 前向/反向传播，而非磁盘读取）。仅当数据集极小（几百 MB）且反复训练时才建议省略此参数。

**调参建议**：

| 硬件 | `num_workers` | `batch_size` |
|------|--------------|-------------|
| CPU 训练 | 0 | 32–64 |
| GPU（单卡） | 4–8 | 64–256 |
| GPU（多卡/大显存） | 8–16 | 256–512 |

> 数据校验改为按需：`dataset.validate()` 可在拿到新数据时显式调用，日常训练默认跳过。
