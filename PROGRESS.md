# 项目进度摘要

本文档用于在上下文不足时快速恢复工程状态。详细棋规看 `SPEC.md`，协作约束看 `AGENTS.md`，整体结构看 `ARCHITECTURE.md`。

## 当前阶段

项目已经完成：

- C 规则引擎与随机压力测试。
- C-Python 接口与 Python 标准环境。
- Tkinter GUI，可人类/随机AI/监督模型AI对局。
- 传统基线 agent：random、greedy、material、alphabeta。
- 候选动作打分神经网络 `CandidateMoveNet`。
- selfplay JSONL 数据生成、检查、读取、训练 batch 组装。
- 训练专用 RL self-play JSONL 生成器，可从 `CandidateMoveNet` checkpoint 按 temperature softmax 探索采样并记录行为概率。
- RL candidate 训练器，读取 RL self-play JSONL 做 PPO 风格 clipped policy/value 更新。
- supervised 监督训练、checkpoint 保存、CSV 日志、累计训练时间。
- `SupervisedAgent`，可加载 `.pt/.pth` 模型并在 arena / selfplay / GUI 中决策。
- Arena 支持 YAML 默认参数和可复现随机 opening，可减少确定性 checkpoint 从固定初始局面重复同一主线。
- 监督训练数据源权重，支持按来源分别设置 policy/value loss 权重。
- 当前方视角归一化 observation 与动作编码：白方行棋时按 `square_id -> 51 - square_id` 做 180 度旋转，使己方前进方向一致。
- GUI 支持 180 度旋转棋盘显示，编号、棋子和点击位置一起旋转。

当前已经跳过“模型辅助 AlphaBeta”。曾实验过 `SupervisedAgent` 的 temperature/top-k/epsilon 探索自博弈，但未带来稳定棋力提升，相关接口已从主线清理。探索采样现放在独立的 `trainer/rl_selfplay.py`，策略更新放在 `trainer/train_selfplay.py`，普通 `SupervisedAgent` 仍保持 greedy。下一步应评测 RL candidate 是否值得晋升，再决定是否增强 RL 更新算法。

## 规则内核

C 引擎在 `engine/`，规则以 `SPEC.md` 为准。核心事实：

- 棋位编号 `1..50`，左上角非合法棋位，第一行合法位编号 `1..5`，第二行从 `6` 开始。
- 初始局面：黑方 `1..20`，白方 `31..50`，黑方先手。
- 普通子非吃子只能向前；普通子吃子可四向并可连跳。
- 王支持远程移动、远程吃子、多落点与连跳。
- 强制吃子，且所有路线按“最多吃子”过滤。
- 被吃棋子在整着结束前仍作为屏障，不能重复跳过。
- 普通子只有最终停在对方底线才升王。
- `draughts_apply_move` 严格执行当前合法动作，不接受结构合法但规则非法的动作。
- 终局：无子或当前方无合法动作。和棋枚举存在，但和棋规则按当前约定暂不实现。

核心 C API 位于 `engine/include/draughts.h`：

```text
draughts_init / draughts_clear / draughts_copy
draughts_get_piece / draughts_set_piece
draughts_get_side_to_move / draughts_set_side_to_move
draughts_generate_legal_moves
draughts_is_legal_move
draughts_apply_move
draughts_get_game_result
```

`DraughtsMove` 用 `path[]` 和 `captures[]` 表示完整路线。

## Python 接口与 GUI

Python 侧优先使用 `interface.DraughtsEnv`：

```text
reset()
observation()
legal_moves()
step(action_id)
clone()
```

规则逻辑仍封闭在 C 内部，Python 只按 `action_id` 推进局面。

GUI 在 `gui/simple_gui.py`：

- 运行：`PYTHONPATH=interface:trainer python3 gui/simple_gui.py`
- 支持人类、随机AI、监督模型AI任意组合。
- 选择监督模型AI时弹出 `.pt/.pth` 文件选择框。
- 支持中文界面、棋盘编号、棋谱、日志、合法动作列表、悔棋、暂停/继续AI、AI走一步。
- 悔棋通过重置 C 环境并重放 `action_id` 历史实现。

## Trainer 环境

`trainer/env.py` 的 `StandardDraughtsEnv` 包装 C-Python 接口：

```text
reset()
step(action_id)
observation()
raw_observation()
legal_actions()
action_mask()
clone()
```

训练 observation 为 `6x10x10`，使用当前方视角归一化：

```text
0 当前方普通子
1 当前方王
2 对方普通子
3 对方王
4 合法棋位
5 保留平面，当前统一为 0
```

黑方行棋时保持原棋盘方向；白方行棋时按 180 度旋转，即棋位 `square_id -> 51 - square_id`。棋子通道仍是“当前方/对方”，因此模型看到的己方普通子总是朝同一个方向前进。原始 `action_id` 仍是当前局面合法动作列表编号，`action_mask` 长度为 `512`，主要用于兼容和调试。

## Agent

`trainer/agents.py` 已有：

```text
RandomAgent       random
GreedyCaptureAgent greedy
MaterialAgent     material
AlphaBetaAgent    alphabeta
SupervisedAgent   supervised
```

- `MaterialAgent`：一步试走材料评估，配置在 `trainer/config/material.yaml`。
- `AlphaBetaAgent`：传统 Alpha-Beta 搜索，默认深度和增强评估在 `trainer/config/alphabeta.yaml`。叶子评估含材料、推进、中心、边线、底线守卫。
- `SupervisedAgent`：加载 `CandidateMoveNet` checkpoint，对所有合法动作编码并根据 policy logits 贪心选择最高分动作。同一 checkpoint 可执黑或执白。

`trainer/arena.py` 支持批量评测：

```bash
PYTHONPATH=interface:trainer python3 trainer/arena.py \
  --black supervised \
  --white alphabeta \
  --black-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 20 \
  --device cpu
```

参数默认值在 `trainer/config/arena.yaml`。比较相近 greedy checkpoint 时可加随机 opening：

```bash
PYTHONPATH=interface:trainer python3 trainer/arena.py \
  --black supervised \
  --white supervised \
  --black-checkpoint trainer/checkpoints/candidate_move_net_rl_best.pt \
  --white-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 100 \
  --opening-random-plies 8 \
  --seed 2026 \
  --device cpu
```

Opening 动作仍由引擎合法动作随机产生，计入该局 `max_plies` 与 `avg_plies`；同 seed 可复现。当前还没有自动做同一 opening 的换边 paired 晋升评测。

## 数据生成与数据集

`trainer/selfplay.py` 生成 JSONL 训练数据。默认读取：

```text
trainer/config/selfplay.yaml
```

命令行显式参数覆盖 YAML。常用命令：

```bash
PYTHONPATH=interface:trainer python3 trainer/selfplay.py
```

或：

```bash
PYTHONPATH=interface:trainer python3 trainer/selfplay.py \
  --seed 2026 \
  --output data/alphabeta_selfplay_seed_2026.jsonl
```

`selfplay.yaml` 当前默认 `alphabeta` vs `alphabeta`、`games=10`、`max_plies=400`、`auto_output_name=1`。支持：

```text
--append            追加写入并延续 game_id
--auto-output-name  文件已存在时自动生成 _001/_002
--no-auto-output-name
```

JSONL 每行是一手棋之前的样本，关键字段：

```text
raw_squares
observation
legal_action_ids
legal_moves
action_mask
selected_action_id
selected_move
final_result
max_plies_reached
value_target
```

`value_target` 为当前行棋方视角：胜 `1`，负 `-1`，和棋或达到最大半回合上限 `0`。

训练专用 RL 自博弈入口是 `trainer/rl_selfplay.py`：

```bash
PYTHONPATH=interface:trainer python3 trainer/rl_selfplay.py \
  --checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 100 \
  --output data/rl_iter_001.jsonl \
  --device cpu
```

默认参数在 `trainer/config/rl_selfplay.yaml`。它直接读取 `CandidateMoveNet` logits/value，按 `softmax(policy_logits / temperature)` 采样合法动作，并在 JSONL 中额外记录 `policy_probs`、`selected_policy_prob`、`predicted_value`、`policy_checkpoint` 等后续策略更新需要的信息。

数据工具：

```text
trainer/inspect_dataset.py  检查 JSONL 合法性和统计分布
trainer/dataset.py          读取 JSONL，构造训练 item / batch / torch tensor
```

注意：`DraughtsJsonlDataset` 当前会把样本读入内存。数据很大时应分 shard 训练，未来再实现 streaming dataset。

## 动作编码与网络

`trainer/move_encoding.py`：

- 棋位 `1..50` 按当前方视角处理后转 10x10 坐标并归一化；白方动作同样先做 `square_id -> 51 - square_id`。
- `MAX_PATH_POINTS=10`
- `MAX_CAPTURES=9`
- `MOVE_FEATURE_DIM=45`
- 编码含起点、终点、路径长度、吃子数、是否吃子、padding 后的 path/captures 坐标。

`trainer/model.py`：

- 模型：`CandidateMoveNet`
- 配置：`trainer/config/model.yaml`
- 输入：

```text
observation   [B, 6, 10, 10]
move_features [B, M, 45]
move_mask     [B, M]
```

- 输出：

```text
policy_logits [B, M]
value         [B]
```

结构：

```text
board_encoder: CNN -> 256 维局面特征
move_encoder:  MLP(45 -> 128)
score_head:    MLP(256+128 -> 1)
value_head:    MLP(256 -> 1) + Tanh
```

结构示意图：`trainer/assets/candidate_move_net.svg`。技术说明：`trainer/README-network.md`。

## 监督训练

`trainer/train_supervised.py` 用 selfplay JSONL 训练 `CandidateMoveNet`。

损失：

```text
policy_loss = CrossEntropy(policy_logits, policy_target_index)
value_loss  = MSE(value, value_target)
loss        = policy_loss + value_loss_weight * value_loss
```

配置：`trainer/config/supervised.yaml`。常用续训命令：

```bash
PYTHONPATH=interface:trainer python3 trainer/train_supervised.py data \
  --resume trainer/checkpoints/candidate_move_net_latest.pt \
  --checkpoint-dir trainer/checkpoints \
  --device cpu
```

启动阶段会立即打印配置读取、数据加载、划分、DataLoader、模型初始化和 checkpoint 加载进度。

Checkpoint 保存策略：

```text
candidate_move_net_latest.pt          每个 epoch 覆盖，便于续训
candidate_move_net_best.pt            验证集 loss 最佳时覆盖
candidate_move_net_epoch_<final>.pt   本次训练结束时保存最终 epoch
```

默认不保存所有中间 epoch；需要时加 `--save-every-epoch`。

训练还会记录：

```text
trainer/checkpoints/training_time.json
trainer/checkpoints/training_log_<experiment_key>.csv
```

CSV 是逐行追加，不读入历史 CSV；累计训练时间可跨同参数/同数据继续累加。

数据源权重：

```bash
PYTHONPATH=interface:trainer python3 trainer/train_supervised.py \
  --source data/alphabeta_selfplay.jsonl:1.0:1.0:alphabeta \
  --source data/value_selfplay.jsonl:0.0:1.0:value_selfplay \
  --resume trainer/checkpoints/candidate_move_net_latest.pt \
  --checkpoint-dir trainer/checkpoints \
  --device cpu
```

也可用 `--source-config trainer/config/data_sources.yaml`。推荐当前策略：AlphaBeta 数据 `policy=1,value=1`；动作质量没有保证的自博弈/结果数据先用 `policy=0,value=1`，只训练 value。

## RL 训练

`trainer/train_selfplay.py` 从已有 checkpoint 出发，用 `rl_selfplay.py` 的探索样本训练 RL candidate：

```bash
PYTHONPATH=interface:trainer python3 trainer/train_selfplay.py data/rf_data \
  --resume trainer/checkpoints/candidate_move_net_best.pt \
  --checkpoint-dir trainer/checkpoints \
  --device cpu
```

配置：`trainer/config/rl_train.yaml`。当前策略目标使用样本中的 `selected_policy_prob` 和 `temperature` 做 PPO ratio 限制：

```text
advantage   = value_target - rollout predicted_value
old_prob    = selected_policy_prob
new_prob    = 当前模型按行为 temperature 算出的 selected_move 概率
ratio       = new_prob / old_prob
policy_loss = - min(ratio * advantage, clip(ratio) * advantage)
value_loss  = MSE(value, value_target)
loss        = policy_weight * policy_loss
            + value_weight  * value_loss
            - entropy_weight * entropy
```

默认保存 `candidate_move_net_rl_latest.pt`、`candidate_move_net_rl_best.pt` 和本次结束的 `candidate_move_net_rl_epoch_<final>.pt`，并记录 `rl_training_log_<experiment_key>.csv` 与累计训练时间。RL 日志含 PPO `ratio` 和 `clip_fraction`。该脚本当前只产生 candidate，不自动评测或晋升。

## 当前模型表现

最新 `candidate_move_net_best.pt` 的阶段性 arena 结果：

```text
vs random:
  supervised 执黑 96/100
  supervised 执白 88/100

vs material:
  supervised 执黑 91/100
  supervised 执白 91/100

vs alphabeta:
  supervised 执黑 6/20，alphabeta 13/20，max_plies 1
  supervised 执白 2/20，alphabeta 18/20
```

当前棋力层级：

```text
alphabeta > supervised > material > random
```

监督学习已经跑通且有效。继续堆 epoch 容易过拟合，优先使用 `best.pt` 做评测。后续不做模型辅助 AlphaBeta，也不保留简单 supervised 探索自博弈作为主线，准备转向更正式的自博弈改进/强化学习路线。

## 测试状态

常用测试：

```bash
make test
make interface-test
make trainer-test
make random-test
```

当前覆盖：

- C 引擎：棋盘编号、初始化、复制、终局、普通子/王移动吃子连跳、最多吃子、屏障、升王、严格动作校验、随机对局。
- Python 接口：reset、observation、legal moves、step、clone 等。
- Trainer：环境、agent、arena YAML 与随机 opening、selfplay YAML/追加/自动命名、RL self-play temperature 采样与 JSONL 字段、PPO ratio 裁剪策略更新和 candidate checkpoint、数据检查、数据读取、当前方视角动作/局面编码、模型前向、监督训练 checkpoint/CSV/累计时间、数据源 policy/value 权重。

历史重要修复：王连跳可以合法回到出发格，执行器不能因起点等于终点而拒绝。

## 常用命令

```bash
make interface
PYTHONPATH=interface:trainer python3 gui/simple_gui.py

PYTHONPATH=interface:trainer python3 trainer/selfplay.py

PYTHONPATH=interface:trainer python3 trainer/rl_selfplay.py \
  --checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 100 \
  --output data/rl_iter_001.jsonl \
  --device cpu

PYTHONPATH=interface:trainer python3 trainer/inspect_dataset.py data

PYTHONPATH=interface:trainer python3 trainer/train_supervised.py data \
  --resume trainer/checkpoints/candidate_move_net_latest.pt \
  --checkpoint-dir trainer/checkpoints \
  --device cpu

PYTHONPATH=interface:trainer python3 trainer/train_selfplay.py data/rf_data \
  --resume trainer/checkpoints/candidate_move_net_best.pt \
  --checkpoint-dir trainer/checkpoints \
  --device cpu

PYTHONPATH=interface:trainer python3 trainer/arena.py \
  --black supervised \
  --white material \
  --black-checkpoint trainer/checkpoints/candidate_move_net_best.pt \
  --games 100 \
  --device cpu
```

## 未完成事项

- candidate 与基座模型的评测、接受或回退机制。
- 同一 opening 自动换边的 paired checkpoint 晋升评测。
- TD/GAE 或更强的 return/value 目标是否需要引入，待 PPO candidate 实验后判断。
- 更大数据时的 streaming dataset 或 shard 采样策略。
- 训练曲线可视化/导出。
- 更系统的多模型评测脚本，等后续方法更多时再统一设计。
- 更大规模复杂残局回归测试。
- 和棋规则暂按当前讨论结果忽略。

## 建议下一步

用 `arena.py --opening-random-plies` 黑白换边复测 PPO RL candidate 与基座 checkpoint、AlphaBeta、material。朴素 advantage policy gradient 已在首次 candidate 实验中明显退化，当前 PPO 目标用于限制 candidate 单轮偏离行为策略的幅度；随机 opening 用于减少固定开局 deterministic 主线对比较结果的支配。
