# 项目进度摘要

本文档用于在上下文不足时快速恢复工程状态。详细棋规看 `SPEC.md`，协作约束看 `AGENTS.md`，整体结构看 `ARCHITECTURE.md`。

## 当前阶段

项目已经完成：

- C 规则引擎与随机压力测试。
- C-Python 接口与 Python 标准环境。
- Tkinter GUI，可人类/随机AI/监督模型AI对局。
- 传统基线 agent：random、greedy、material、alphabeta。
- 强搜索 agent：`alphabeta_plus`，默认使用比 `alphabeta` 更深的搜索，用作强教师和高水平对局智能体入口。
- 候选动作打分神经网络 `CandidateMoveNet`。
- `ValueSearchAgent`：加载同一 checkpoint，用 value_head 对走后局面做浅层 minimax 搜索，作为“神经网络 + 搜索”的最小试验入口。
- selfplay JSONL 数据生成、检查、读取、训练 batch 组装。
- 训练专用 RL self-play JSONL 生成器，可从 `CandidateMoveNet` checkpoint 按 temperature softmax 探索采样并记录行为概率。
- RL candidate 训练器，读取 RL self-play JSONL 做 PPO 风格 clipped policy/value 更新。
- supervised 监督训练、checkpoint 保存、CSV 日志、累计训练时间。
- `SupervisedAgent`，可加载 `.pt/.pth` 模型并在 arena / selfplay / GUI 中决策。
- Arena 支持 YAML 默认参数和可复现随机 opening，可减少确定性 checkpoint 从固定初始局面重复同一主线。
- 自动 checkpoint 对比工具，支持 candidate vs baseline 多 seed、多 opening、换边汇总，并可对 material/alphabeta 等外部 agent 做参考评测。
- 第一版搜索改进数据生成器 `trainer/search_improve.py`：可从现有 JSONL 样本恢复 C 环境局面，用 AlphaBeta 为当前合法动作重新打分，并输出兼容现有监督训练器的改进 policy target。
- `trainer/search_improve.py` 支持 `--min-score-margin` 过滤低置信搜索标签，并支持 `--position-sample-rate` 从更宽输入局面中随机抽样。
- 监督训练数据源权重，支持按来源分别设置 policy/value loss 权重。
- 当前方视角归一化 observation 与动作编码：白方行棋时按 `square_id -> 51 - square_id` 做 180 度旋转，使己方前进方向一致。
- GUI 支持 180 度旋转棋盘显示，编号、棋子和点击位置一起旋转。

曾实验过 `SupervisedAgent` 的 temperature/top-k/epsilon 探索自博弈，但未带来稳定棋力提升，相关接口已从主线清理。探索采样现放在独立的 `trainer/rl_selfplay.py`，策略更新放在 `trainer/train_selfplay.py`，普通 `SupervisedAgent` 仍保持 greedy。当前已补入最小 `ValueSearchAgent`，但实验显示当前 `value_head` 尚不能单独指导落子。项目最终目标仍是“接近深层 AlphaBeta 棋力、但推理效率更高”的强化学习/神经网络智能体；AlphaBeta 应作为教师、评测基准和数据来源，而不是最终主线。

## 关键实验结论

后续新会话请优先读取本节，避免重复走弯路：

- 当前最强稳定传统 agent 仍是 `AlphaBetaAgent`；当前神经网络 agent 还不能稳定接近 AlphaBeta，更不能替代深层 AlphaBeta。
- 监督模型/当前 RL 模型能高胜率击败 random/material，但不是 100% 稳赢，也明显弱于 AlphaBeta。这是正常的近似模型局限，不代表引擎错误。
- 简化 PPO 路线已跑通闭环，但提升开始停滞：多轮 candidate 常在 candidate/base 对比中约 `51%~52%`，难稳定超过 `promote_threshold=0.53`。
- `compare_checkpoints.py` 的 `quick/standard/full` 分级评测可继续用于晋级判断；不要每次默认跑 AlphaBeta，太慢。
- `ValueSearchAgent` 纯 `value_head` depth=1 实验失败：同一 `candidate_move_net_rl_best.pt` 下，`valuesearch` 对 `supervised` 换边合计约 `32:166`，说明当前 `value_head` 不能直接做走后局面排序。
- 用 RL 数据做 `policy_weight=0,value_weight=1` 的 value-only 全网络微调失败：`value_tune/candidate_move_net_latest.pt` 对 `candidate_move_net_rl_best.pt` 的 quick 评测约 `37:56`，`decisive_wr=0.398`，因为 value loss 更新共享 `board_encoder` 后破坏了 policy。
- 不建议继续在“纯 ValueSearch / value-only 全网络微调 / 反复小批量简化 PPO”上消耗太多时间。
- 重要路线修正：强化学习仍是主线，但应从简化 PPO 小修小补转向更标准的“搜索改进策略目标 + 自博弈晋级”路线。更像 AlphaZero：用搜索生成更强 policy target / value target，再训练网络逼近搜索改进后的策略。

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
ValueSearchAgent  valuesearch
```

- `MaterialAgent`：一步试走材料评估，配置在 `trainer/config/material.yaml`。
- `AlphaBetaAgent`：传统 Alpha-Beta 搜索，默认深度和增强评估在 `trainer/config/alphabeta.yaml`。叶子评估含材料、推进、中心、边线、底线守卫。
- `SupervisedAgent`：加载 `CandidateMoveNet` checkpoint，对所有合法动作编码并根据 policy logits 贪心选择最高分动作。同一 checkpoint 可执黑或执白。
- `ValueSearchAgent`：加载 `CandidateMoveNet` checkpoint，用 `value_head` 对模拟后的局面做浅层 minimax，配置在 `trainer/config/value_search.yaml`。当前实验证明纯 value 引导效果很差，保留作研究入口，不应作为主力 agent。

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

Opening 动作仍由引擎合法动作随机产生，计入该局 `max_plies` 与 `avg_plies`；同 seed 可复现。相近 checkpoint 的晋升评测优先使用下面的自动换边工具。

自动晋升评测入口：

```bash
PYTHONPATH=interface:trainer python3 trainer/compare_checkpoints.py \
  --candidate trainer/checkpoints/candidate_move_net_rl_best.pt \
  --baseline trainer/checkpoints/candidate_move_net_rl_iter_003_best.pt \
  --preset quick \
  --report reports/compare_iter_004.json \
  --device cpu
```

默认配置：`trainer/config/compare.yaml`。主评测自动按 `seeds` 和 `opening_random_plies` 做 candidate/base 换边汇总；外部参考默认只跑 `material`，避免 AlphaBeta 评测过慢。`--preset quick` 只跑 candidate/base 快筛，`standard` 跑 material 参考，`full` 才跑 `material,alphabeta`，`alphabeta` 可单独做少量强基准检查。晋升建议使用 `promote_threshold=0.53`、`reject_threshold=0.50`、`bucket_floor=0.45`。

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

`trainer/config/rl_train.yaml` 当前采用稳定优先默认：`learning_rate=0.00005`、`advantage_clip=1.5`、`ppo_clip_epsilon=0.1`。这是在候选模型接近但未达晋级线、且额外训练出现 quick reject 后调整的，目的是减小单轮 PPO 对已通过基座的破坏风险。

## 当前模型表现

监督基座 `candidate_move_net_best.pt` 的阶段性 arena 结果：

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

监督学习已经跑通且有效。继续堆 epoch 容易过拟合，优先使用 `best.pt` 做评测。

RL/PPO 阶段重要结果：

```text
candidate_move_net_rl_best.pt 是当前较稳定 RL 基座。
PPO candidate 曾相对监督基座略有提升，但相对自身上一代经常只能到 51%-52%。
额外训练 candidate 有时 quick reject，有时 inconclusive；尚未形成稳定晋级趋势。
```

最近一次候选相关结论：

```text
rl_iter_extra/candidate_move_net_rl_latest.pt:
  quick 曾约 candidate=48 baseline=45，wr=0.516，inconclusive
  full 汇总 candidate=387 baseline=367，wr=0.513，仍 inconclusive
  对 alphabeta 外部参考略好于 baseline，但不够稳定，不应晋级

value_tune/candidate_move_net_latest.pt:
  quick 约 candidate=37 baseline=56，wr=0.398，reject
  不要使用该 checkpoint 继续训练或生成数据
```

当前认识：如果短期只追求最强对局，直接用 `AlphaBetaAgent`；但项目长期目标是神经网络/强化学习智能体达到接近深层 AlphaBeta 的水平且推理效率更高，因此强化学习仍应作为主线。下一阶段不要继续零散尝试，而应转向“搜索改进数据 / MCTS visit count policy target / 自动晋级”的更标准路线。

2026-05-24 自主实验补充：

```text
search_improved_ab_depth2_margin50_cons:
  数据：从 alphabeta_clean 中用 depth=2 AlphaBeta 重新标注，保留 search_score_margin >= 50 的 5000 样本。
  训练：从 candidate_move_net_rl_best.pt 低学习率 policy-only 微调 3 epoch。
  quick：candidate=52 baseline=42 maxed=6 decisive_wr=0.553，promote。
  standard：candidate=399 baseline=355 maxed=46 decisive_wr=0.529，inconclusive，略低于 0.53 晋级线。
  alphabeta 外部参考：candidate_wr=0.060，baseline_wr=0.050，仍远弱于 AlphaBeta。

search_improved_ab_depth2_margin50_tiny2:
  在同一 5000 样本上继续 2 epoch 极小学习率微调，quick 回落到 decisive_wr=0.500，不应使用。

search_improved_ab_margin50_broadmix:
  将高置信搜索样本与全量 alphabeta_clean 低权重混合训练，quick decisive_wr=0.417，reject，不应使用。

alphabeta_plus:
  新增强搜索 agent，默认 search_depth=3。
  对默认 alphabeta 换边实验：
    alphabeta_plus 黑 vs alphabeta 白：13:4，maxed=3。
    alphabeta 黑 vs alphabeta_plus 白：2:17，maxed=1。
  当前若只看对局性能，alphabeta_plus 已明显强于默认 alphabeta，但速度明显更慢。
```

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
- Trainer：环境、agent、arena YAML 与随机 opening、自动 checkpoint 对比报告、selfplay YAML/追加/自动命名、RL self-play temperature 采样与 JSONL 字段、PPO ratio 裁剪策略更新和 candidate checkpoint、数据检查、数据读取、当前方视角动作/局面编码、模型前向、监督训练 checkpoint/CSV/累计时间、数据源 policy/value 权重。

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

- 更成熟的自动接受/回退机制；当前 `compare_checkpoints.py` 只给建议，不移动 checkpoint。
- 搜索改进数据生成器目前先采用单一 best move 目标，尚未实现候选动作评分分布/visit-count 分布，也尚未接入自动训练与晋级闭环。初次未过滤 depth=2 search-improved 直接微调已被 quick reject，后续应优先使用 score margin 过滤、低学习率和混合训练防止策略漂移。
- 设计 AlphaZero 风格最小闭环：当前网络 -> 搜索自博弈/搜索改进数据 -> 训练 policy/value -> 自动评测 -> 晋级/回退。
- TD/GAE 或更强 return/value 目标尚未实现；当前简化 PPO 仅用终局胜负和 rollout predicted_value。
- `ValueSearchAgent` 目前纯 value 引导失败；若继续用，需要改成 policy/value 混合或等 value head 明显增强后再试。
- value-only 训练如果要继续，必须避免破坏 policy，例如冻结 `board_encoder/move_encoder/score_head` 只训 `value_head`；但这不是当前推荐主线。
- 更大数据时的 streaming dataset 或 shard 采样策略。
- 训练曲线可视化/导出。
- 更系统的多模型评测脚本，等后续方法更多时再统一设计。
- 更大规模复杂残局回归测试。
- 和棋规则暂按当前讨论结果忽略。

## 建议下一步

建议新会话不要继续把主要精力放在简化 PPO 小步微调上。下一步更值得实现：

```text
搜索改进数据生成器
  输入：已有 JSONL 局面或自博弈局面
  对每个局面：调用 C 引擎合法动作，使用 AlphaBeta/轻量搜索评估候选动作
  输出：更强 policy target（先可用单一 best move，后续扩展为分布/visit counts）和 value target
  训练：用 train_supervised.py 或新训练器学习搜索改进后的策略
  评测：compare_checkpoints.py quick -> standard -> full，只有 promote 才晋级
```

这条路线符合最终目标：AlphaBeta 作为教师/搜索改进器，神经网络学习逼近强搜索，从而获得接近深层搜索但更低推理成本的智能体。后续若该数据生成器有效，再演进到真正 MCTS 自博弈和 visit count target。
