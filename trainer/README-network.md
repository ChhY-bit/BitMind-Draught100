# 神经网络组件设计记录

本文档记录训练侧神经网络相关组件的设计。当前已经实现动作编码器与第一版候选动作打分网络；训练脚本后续继续补充。

结构示意图：

```text
trainer/assets/candidate_move_net.svg
```

## 总体思路

当前计划采用“局面 + 单个候选动作 -> 分数”的候选动作打分模型。

```text
observation
  |
  v
CNN board encoder
  |
  +-------------------------> value head -> value
  |
  v
board_feature
  |
  +------------------+
                     |
move feature vector  |
  |                  |
  v                  |
MLP move encoder     |
  |                  |
  v                  |
move_embedding ------+
  |
concat(board_feature, move_embedding)
  |
score head
  |
score for this candidate move
```

对一个局面有 `N` 个合法动作时，同一个网络对 `N` 个候选动作分别打分，得到 `N` 个 score。网络结构不随 `N` 改变；变化的是当前需要评分的候选动作数量。

## 棋盘输入

棋盘输入沿用 `trainer/env.py` 的 observation：

```text
6 x 10 x 10
```

通道含义：

| 通道 | 含义 |
| --- | --- |
| `0` | 当前行棋方普通子 |
| `1` | 当前行棋方王 |
| `2` | 对方普通子 |
| `3` | 对方王 |
| `4` | 合法棋位 |
| `5` | 当前行棋方平面 |

后续 CNN board encoder 会把它转换为 `board_feature`。这些特征通常不是人工指定的物理量，而是在训练过程中学到的抽象局面表示。

## 动作编码器 v1

实现文件：

```text
trainer/move_encoding.py
```

动作来自 selfplay JSONL 中的 `legal_moves` 或 `selected_move`：

```json
{
  "action_id": 3,
  "path": [32, 23, 14],
  "captures": [28, 19],
  "is_capture": true
}
```

### 坐标转换

棋位编号 `1..50` 先转换为 10x10 棋盘坐标：

```text
row = (square_id - 1) // 5
offset = (square_id - 1) % 5
col = offset * 2 + (1 if row % 2 == 0 else 0)
```

再归一化：

```text
row_norm = row / 9
col_norm = col / 9
```

示例：

```text
1  -> row=0 col=1
6  -> row=1 col=0
50 -> row=9 col=8
```

### 固定长度

当前常量：

```text
MAX_PATH_POINTS = 10
MAX_CAPTURES = 9
MOVE_FEATURE_DIM = 45
```

现有 `data/` 扫描结果：

```text
max path length = 6
max capture count = 5
```

因此 `10/9` 对当前数据有余量，同时比早先考虑的 `16/15` 更紧凑。若后续出现更长合法连跳，编码器会报错而不是静默截断，便于及时调整上限。

### 向量布局

每个动作编码为长度 `45` 的 float list：

```text
[
  start_row,
  start_col,
  end_row,
  end_col,

  path_length_norm,
  capture_count_norm,
  is_capture,

  path_0_row,
  path_0_col,
  ...
  path_9_row,
  path_9_col,

  capture_0_row,
  capture_0_col,
  ...
  capture_8_row,
  capture_8_col
]
```

维度计算：

```text
4                 start/end 坐标
3                 path_length、capture_count、is_capture
10 * 2 = 20       path 坐标
9 * 2 = 18        capture 坐标
合计 45
```

不足的 path/capture 槽位补 `0.0`。

### Dataset 输出

`trainer/dataset.py` 已接入动作编码器。每个 item 现在包含：

```text
observation
action_mask
policy_target
move_features
policy_target_index
value_target
metadata
```

其中：

- `move_features`：当前所有合法动作的编码，形状为 `N x 45`。
- `policy_target_index`：老师选择的 `selected_move` 在 `legal_moves` 中的位置。
- `policy_target`：旧的 `selected_action_id`，暂时保留用于兼容和调试。

`collate_batch()` 会按当前 batch 内最大候选动作数 padding：

```text
move_features: [B, M, 45]
move_mask:     [B, M]
```

其中：

- `B` 是 batch size。
- `M` 是当前 batch 内最大合法动作数。
- padding 动作的 `move_mask` 为 `0`。

## 候选动作打分网络 v1

实现文件：

```text
trainer/model.py
trainer/config/model.yaml
```

类名：

```text
CandidateMoveNet
```

输入：

```text
observation:   [B, 6, 10, 10]
move_features: [B, M, 45]
move_mask:     [B, M]
```

其中：

- `B` 是 batch size。
- `M` 是当前 batch 中 padding 后的最大候选动作数。
- `move_mask=1` 表示真实候选动作，`move_mask=0` 表示 padding 动作。

网络结构：

```text
board_encoder: CNN(6 x 10 x 10 -> board_feature)
move_encoder:  MLP(45 -> move_embedding)
score_head:    MLP(board_feature + move_embedding -> score)
value_head:    MLP(board_feature -> value)
```

结构参数从 `trainer/config/model.yaml` 读取。当前默认配置：

```yaml
board_channels: 6
board_size: 10
conv_channels: 64
conv_layers: 3
conv_kernel_size: 3
conv_padding: 1
board_feature_dim: 256
move_feature_dim: 45
move_embedding_dim: 128
hidden_dim: 128
```

这些参数的含义：

| 参数 | 含义 |
| --- | --- |
| `board_channels` | 棋盘输入通道数，当前 observation 是 6 通道。 |
| `board_size` | 棋盘宽高，当前为 10。 |
| `conv_channels` | 每个卷积层输出的特征通道数。 |
| `conv_layers` | board encoder 中连续卷积层数量。 |
| `conv_kernel_size` | 卷积核尺寸，`3` 表示观察 3x3 局部区域；当前要求为奇数。 |
| `conv_padding` | 边缘补零宽度，当前要求等于 `conv_kernel_size // 2`，以保持 10x10 空间大小。 |
| `board_feature_dim` | CNN 展平后压缩得到的局面特征维度。 |
| `move_feature_dim` | 单个候选动作输入特征维度，当前由动作编码器决定为 45。 |
| `move_embedding_dim` | move encoder 输出的动作特征维度。 |
| `hidden_dim` | score head 和 value head 的隐藏层维度。 |

具体参数：

```text
board_encoder:
  Conv2d(6, 64, kernel=3, padding=1)
  ReLU
  Conv2d(64, 64, kernel=3, padding=1)
  ReLU
  Conv2d(64, 64, kernel=3, padding=1)
  ReLU
  Flatten
  Linear(64*10*10, 256)
  ReLU

move_encoder:
  Linear(45, 128)
  ReLU
  Linear(128, 128)
  ReLU

score_head:
  Linear(256+128, 128)
  ReLU
  Linear(128, 1)

value_head:
  Linear(256, 128)
  ReLU
  Linear(128, 1)
  Tanh
```

输出：

```text
policy_logits: [B, M]
value:         [B]
```

padding 动作会被 mask 成极小 logit：

```text
MASKED_LOGIT = -1e9
```

这样训练时 softmax 几乎不会选择 padding 动作。

训练目标：

```text
policy loss: CrossEntropy(scores, policy_target_index)
value loss:  MSE(value, value_target)
```

当前模型只完成结构和前向计算，还没有训练脚本、模型保存或评测流程。
