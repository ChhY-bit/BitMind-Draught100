# 项目进度记录

本文档只保留阶段性摘要，详细规则以 `SPEC.md` 为准，工程约束以 `AGENTS.md` 为准。

## 当前阶段

C 底层引擎已形成可测试、可调试的第一版规则内核。实现集中在 `engine/`：

- `include/draughts.h`：公共接口与基础数据结构。
- `src/board.c`：棋盘编号、局面状态、状态复制、终局判断。
- `src/move.c`：合法动作生成、动作严格校验、动作执行。
- `tests/test_board.c`、`tests/test_moves.c`：单元测试。
- `tests/random_play.c`：随机对局压力测试。
- `tools/board_viewer.c`：临时文本棋盘、固定场景与简易命令行对局器。
- `interface/`：C-Python 接口层，包含 C 侧粗粒度环境封装和 Python `ctypes` 包装。
- `gui/simple_gui.py`：Python GUI v0，支持人类、随机AI、监督模型AI组合对局，点击棋子与落点执行动作。
- `trainer/env.py`：训练用标准化环境，包装 C-Python 接口并输出固定 observation 与 action mask。
- `trainer/agents.py`、`trainer/arena.py`：基础 agent、监督学习模型 agent 与批量对战评测器。

## 已实现规则

- 100 格国际跳棋棋盘，外部棋位编号为 `1-50`，内部数组为 `0-49`。
- 初始局面：黑方 `1-20`，白方 `31-50`，黑方先手。
- 普通子：非吃子只能向前，吃子可四向，支持连跳。
- 王：支持远程普通移动、远程吃子、多个落点与连跳。
- 吃子强制；所有普通子和王共同参与“最多吃子”规则；同为最多的路线全部保留。
- 被吃棋子在整着结束前仍作为屏障，且不能重复跳过。
- 普通子最终停在对方底线才升王；跳吃经过底线但离开时不升王。
- `draughts_apply_move` 已改为严格执行当前合法动作，不接受只满足结构但违反规则的动作。
- 终局判断已支持无子、当前方无合法动作；和棋枚举保留，但和棋规则暂不实现。

## 主要接口

公共接口见 `engine/include/draughts.h`。当前核心接口包括：

- `draughts_init`、`draughts_clear`、`draughts_copy`
- `draughts_get_piece`、`draughts_set_piece`
- `draughts_get_side_to_move`、`draughts_set_side_to_move`
- `draughts_generate_legal_moves`
- `draughts_is_legal_move`
- `draughts_apply_move`
- `draughts_get_game_result`

`DraughtsMove` 使用完整路径 `path[]` 和被吃棋子列表 `captures[]` 表示普通移动、吃子与连跳。

Python 侧优先使用 `interface.DraughtsEnv`，通过 `reset()`、`legal_moves()`、`step(action_id)` 和 `observation()` 调用 C 环境接口。规则逻辑仍封闭在 C 内部。

GUI 侧通过同一个 Python 接口推进棋局。当前已实现人类、随机AI、监督模型AI之间的组合对局；启动和新建对局时可选择双方决策源，若选择监督模型AI则弹出文件选择框为对应方加载 `.pt/.pth` checkpoint。人类方使用点击棋子、显示落点、点击落点或双击动作列表执行；随机AI方从当前合法动作中随机选取；监督模型AI方把当前局面编码为训练侧 observation 后调用 `SupervisedAgent` 选最高分动作。左侧显示棋谱，右侧含中文状态、合法动作和彩色日志框，王棋统一显示为 `K`。GUI 已支持悔棋，方式是重置 C 环境并重放动作历史。

训练侧当前处于“传统搜索基线 + 标准环境”阶段：

- `trainer/env.py`：`StandardDraughtsEnv` 包装 `interface.DraughtsEnv`，提供 `reset()`、`step(action_id)`、`observation()`、`raw_observation()`、`legal_actions()`、`action_mask()`、`clone()`。`clone()` 依赖 C 侧环境结构体直接复制，不再从开局重放历史。
- observation 为 `6x10x10` 嵌套列表：当前方普通子、当前方王、对方普通子、对方王、合法棋位、当前行棋方平面。当前不做棋盘翻转。
- action 原始字段仍保留“当前局面合法动作列表编号”和长度 `512` 的 `action_mask`，用于兼容与调试；神经网络路线改为候选动作打分，使用 `move_encoding.py` 将每个合法动作编码为固定长度向量。
- `trainer/agents.py`：已有 `RandomAgent`、`GreedyCaptureAgent`、`MaterialAgent`、`AlphaBetaAgent`、`SupervisedAgent`。所有 agent 都只通过环境接口读取局面和执行动作，规则逻辑仍在 C 内。
- `SupervisedAgent`：加载 `CandidateMoveNet` checkpoint，对当前所有合法动作编码后用 policy logits 选分数最高者。模型输入仍是“己方/对方”视角，所以同一个 checkpoint 可执黑或执白；`arena.py` 和 `selfplay.py` 支持 `--black-checkpoint`、`--white-checkpoint` 为双方分别加载不同模型。
- `MaterialAgent`：一步试走型材料评估，逐个 `clone()` 并 `step()` 当前合法动作，按 `trainer/config/material.yaml` 的普通子、王、胜局、和棋分值评分。
- `AlphaBetaAgent`：传统 Alpha-Beta 搜索基线，默认深度在 `trainer/config/alphabeta.yaml`。搜索只调用 `env.clone()` 与 `env.step()`；叶子局面用增强静态评估，特征包括材料、普通子推进、中心控制、边线惩罚、底线守卫。暂未加入机动性/威胁检测，因为这会频繁生成额外合法动作并拖慢 Python 搜索。
- `trainer/arena.py`：批量评测器，支持 `--black/--white random|greedy|material|alphabeta|supervised`、`--material-config`、`--alphabeta-config`、`--black-checkpoint`、`--white-checkpoint`，统计胜负、达到最大步数局数、平均半回合数和耗时。
- `trainer/selfplay.py`：对局数据生成器，支持用任意两个现有 agent 自动下棋并写出 JSONL。默认读取 `trainer/config/selfplay.yaml`，命令行显式参数可覆盖 YAML。每一行是一手棋之前的训练样本，包含 `raw_squares`、`observation`、`legal_action_ids`、`legal_moves`、`action_mask`、`selected_action_id`、`selected_move`、`final_result`、`max_plies_reached`、`value_target`。`value_target` 使用当前行棋方视角：胜 `1`，负 `-1`，和棋或达到最大步数未结束 `0`。支持 `--append` 追加写入并延续 `game_id`，也支持 `--auto-output-name` 在目标文件已存在时自动生成 `_001/_002` 新文件名以避免覆盖。
- `trainer/config/selfplay.yaml`：selfplay 数据生成参数，带中文注释。当前默认 `alphabeta` 对 `alphabeta`，`games=10`，`max_plies=400`，输出 `data/selfplay.jsonl`，并开启 `auto_output_name` 以避免误覆盖旧数据。
- `trainer/inspect_dataset.py`：JSONL 数据检查器，支持单文件、多文件或目录输入。统计文件数、样本数、局数、黑胜/白胜/和棋/截断局数、平均半回合数、`value_target` 分布、agent 分布，并验证必需字段、`6x10x10` observation、长度 50 的 `raw_squares`、长度 512 的 `action_mask`、动作合法性和值标签范围。
- `trainer/move_encoding.py`：候选动作编码器。棋位 `1..50` 转 10x10 坐标并归一化；当前上限 `MAX_PATH_POINTS=10`、`MAX_CAPTURES=9`，动作向量 `MOVE_FEATURE_DIM=45`，包含起终点、路径长度、吃子数、是否吃子、padding 后的 path/captures 坐标。现有 `data/` 扫描最大路径点数为 6、最大吃子数为 5。
- `trainer/dataset.py`：JSONL 数据读取器。`iter_jsonl_samples()` 支持流式读取文件或目录；`DraughtsJsonlDataset` 是内存数据集，返回 `observation`、`action_mask`、`policy_target`、`move_features`、`policy_target_index`、`value_target`、`metadata`；`collate_batch()` 会将不同候选动作数 padding 成 `[B, M, 45]` 并给出 `move_mask`；`to_torch_batch()` 在安装 PyTorch 时把 batch 转成 tensor，但当前不强制依赖 PyTorch。
- `trainer/model.py`：第一版候选动作打分网络 `CandidateMoveNet`，结构参数由 `trainer/config/model.yaml` 配置并做基础校验。输入 `observation [B,6,10,10]`、`move_features [B,M,45]`、`move_mask [B,M]`；CNN board encoder 默认输出 256 维局面特征，MLP move encoder 默认输出 128 维动作特征，score head 输出 `policy_logits [B,M]`，value head 输出 `value [B]`；padding 动作 logit 置为 `-1e9`。
- `trainer/train_supervised.py`：第一版监督学习训练脚本。读取 JSONL 数据训练 `CandidateMoveNet`，损失为 `CrossEntropy(policy_logits, policy_target_index) + value_loss_weight * MSE(value, value_target)`；支持 `--resume` 续训、CPU/CUDA 设备选择。启动阶段会立即打印配置读取、数据集加载、数据划分、DataLoader 构建、模型初始化和 checkpoint 加载进度，避免大数据集启动时长时间无输出。默认省空间保存策略为持续更新 `latest`、验证最佳时更新 `best`、本次训练结束时保存最终 `epoch_XXX`；只有显式传入 `--save-every-epoch` 才保存所有中间 epoch。checkpoint 与 `trainer/checkpoints/training_time.json` 记录当前运行耗时和同一组参数/数据的累计训练时间。训练指标还会追加写入 `training_log_<experiment_key>.csv`，使用逐行追加，不读入历史 CSV，适合长时间训练。
- `trainer/config/supervised.yaml`：监督学习训练参数，含 epoch、batch size、学习率、value loss 权重、验证集比例、随机种子、shuffle、终端日志间隔、CSV epoch 记录间隔和 DataLoader 子进程数，带中文注释。
- `trainer/README-network.md`：神经网络组件设计记录。当前记录候选动作打分模型思路、棋盘输入、动作编码器 v1、dataset 输出、`CandidateMoveNet` 结构；后续训练脚本继续写入该文档。
- `trainer/assets/candidate_move_net.svg`：当前 `CandidateMoveNet` 的结构示意图，展示棋盘分支、动作分支、score head、value head、mask 和 YAML 参数。
- `trainer/tests/`：trainer 单元测试目录，`make trainer-test` 使用 unittest discovery 自动运行 `test_*.py`，后续新增测试无需修改 Makefile。
- 经验现象：`AlphaBetaAgent` 默认深度 2 明显强于 `random/material`，对 `alphabeta` 自战则互有胜负；速度比随机慢得多但可作为早期强基线和后续监督学习老师。

## 测试状态

常用命令：

```bash
make test
make random-test
make interface-test
make trainer-test
build/random_play 10000 300 2000 1
```

测试覆盖棋盘编号、初始化、复制、终局、普通子移动/吃子/连跳、王移动/吃子/连跳、最多吃子、屏障、升王、严格动作校验、非法参数与随机对局。训练侧测试覆盖 observation/action mask、环境 clone、基础 agent、YAML 配置读取、Material 一步评估、AlphaBeta 深度搜索、增强评估的推进/中心/边线特征、SupervisedAgent checkpoint 加载与合法出招、arena 冒烟运行、supervised arena 冒烟运行、selfplay YAML 配置读取、命令行覆盖、JSONL 写出、追加写入、自动避让文件名和值目标计算、inspect_dataset 的合法/非法数据检查、move_encoding 的坐标/动作编码、dataset 的目录读取/样本转换/候选动作 padding/batch 组装，模型配置读取、配置校验、前向形状、mask 和 dataset batch 兼容性，以及监督训练的 checkpoint 保存策略、best/latest 文件、累计训练时间记录和 CSV 续写。

随机测试曾发现并修复一个重要边界：王连跳可以合法回到出发格，执行器不能因起点等于终点而拒绝该动作。

## 临时调试工具

`board_viewer` 可查看编号、初始局面、固定测试场景、执行场景首个动作、批量场景，以及进入简易命令行对局器。它只是 C 端临时调试工具，不是正式 GUI。

常用命令：

```bash
make tools
build/board_viewer --ids
build/board_viewer --scenarios
build/board_viewer --scenario king-multi
build/board_viewer --play initial
```

Python GUI：

```bash
make interface
PYTHONPATH=interface python3 gui/simple_gui.py
```

## 未完成事项

- 更大规模的复杂残局回归测试。
- AlphaBeta 的走法排序、缓存、限时搜索，以及机动性/威胁等更昂贵评估特征。
- 模型评测脚本、神经网络 agent、训练曲线导出。
- 和棋规则暂按当前讨论结果忽略。

## 建议下一步

可以实现更系统的模型评测脚本，批量加载多个 checkpoint，与 `random/material/alphabeta` 做固定种子对战并输出 CSV 报告。
