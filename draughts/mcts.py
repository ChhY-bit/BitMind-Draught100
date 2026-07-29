from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .move_encoding import (
    SQUARE_COUNT,
    encode_legal_moves,
    move_to_dict,
    perspective_square_id,
    square_to_coord,
)
from .model import CandidateMoveNet
from .env import EMPTY, piece_channel

DEFAULT_MCTS_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "mcts.yaml"

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class MCTSConfig:
    """Hyperparameters for MCTS search."""

    num_simulations: int = 800
    c_puct: float = 1.5
    temperature: float = 0.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    add_noise: bool = True
    batch_size: int = 1

    def validate(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.c_puct < 0:
            raise ValueError("c_puct must be non-negative")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")


# ---------------------------------------------------------------------------
# 树节点与搜索结果
# ---------------------------------------------------------------------------


@dataclass
class MCTSNode:
    """A node in the MCTS search tree.

    Each node represents a game state.  Its ``children`` dict maps
    ``action_id`` → ``MCTSNode``.  The statistics stored on a *child*
    node represent the edge (parent → child): ``visit_count`` and
    ``total_value`` track how often the parent chose this action and
    what cumulative value was back-propagated, **both from the
    parent's perspective**.
    """

    prior: float = 0.0
    visit_count: int = 0
    total_value: float = 0.0
    virtual_loss_count: int = 0
    children: dict[int, MCTSNode] = field(default_factory=dict)

    # Q(s,a)项的定义
    def q(self) -> float:
        """Average value of this edge (from the parent player's perspective)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass
class MCTSResult:
    """Result of an MCTS search at a position.

    Attributes:
        visit_counts: ``action_id`` → number of simulations that chose it.
        root_value: Average value from the root player's perspective.
        policy_probs: Visit-count distribution normalised to sum to 1,
                      aligned with ``action_ids``.
        action_ids: ``action_id`` corresponding to each entry in ``policy_probs``.
    """

    visit_counts: dict[int, int]
    root_value: float
    policy_probs: list[float]
    action_ids: list[int]
    network_value: float = 0.0
    q_values: dict[int, float] = field(default_factory=dict)

    def selected_action(self, temperature: float = 0.0) -> int:
        """Pick an action from the visit-count distribution.

        Args:
            temperature:
                0.0 → deterministic (most-visited action).
                > 0 → sample from ``softmax(log(visits) / temperature)``.
        """
        if not self.action_ids:
            raise ValueError("MCTSResult has no actions (terminal position?)")

        if temperature == 0.0:
            max_visits = max(self.visit_counts.values())
            candidates = [a for a, v in self.visit_counts.items() if v == max_visits]
            return random.choice(candidates)

        probs = torch.tensor(self.policy_probs, dtype=torch.float64)
        # Convert visits → probabilities via temperature scaling
        safe_probs = probs.clamp(min=1e-12)
        if temperature != 1.0:
            safe_probs = safe_probs ** (1.0 / temperature)
        safe_probs = safe_probs / safe_probs.sum()
        idx = int(torch.multinomial(safe_probs, 1).item())
        return self.action_ids[idx]


# ---------------------------------------------------------------------------
# 公开搜索接口
# ---------------------------------------------------------------------------


def mcts_search(
    env,
    model: CandidateMoveNet,
    config: MCTSConfig,
    device: torch.device,
    *,
    add_noise: bool = False,
) -> MCTSResult:
    """Run MCTS from the position represented by *env*.

    The supplied *env* is **not** modified; the search works on internal
    clones.

    Args:
        env: ``StandardDraughtsEnv`` at the root position.
        model: ``CandidateMoveNet`` for policy & value predictions.
        config: Search hyperparameters.
        device: ``torch.device`` for network inference.
        add_noise: If ``True``, add Dirichlet noise to the root prior
                   (enabled during self-play, disabled during match play).

    Returns:
        ``MCTSResult`` with visit-count distribution, root value, and
        normalised policy probabilities.
    """
    config.validate()

    legal_actions = env.legal_actions()
    if not legal_actions:
        raise ValueError("mcts_search called on a terminal position")

    # --- root network inference (for prior probabilities) ---
    policy_logits, _root_value = _infer_network(env, model, legal_actions, device)
    priors = _softmax(policy_logits)

    if add_noise:
        priors = _add_dirichlet_noise(priors, config)

    # --- build root with children ---
    root = MCTSNode()
    for i, move in enumerate(legal_actions):
        root.children[move.action_id] = MCTSNode(prior=float(priors[i]))

    # --- run simulations in batches ---
    batch_size = config.batch_size
    remaining = config.num_simulations

    while remaining > 0:
        n = min(batch_size, remaining)

        # Phase 1: collect n leaves (selection with virtual loss)
        leaf_data: list[tuple[MCTSNode, list[tuple[MCTSNode, int]], object, bool]] = []
        for _ in range(n):
            sim_env = env.clone_bare()
            leaf_node, path, is_terminal = _collect_leaf(root, sim_env, config.c_puct)
            leaf_data.append((leaf_node, path, sim_env, is_terminal))

        # Phase 2: categorize leaves — terminal vs. needing network inference
        leaf_values: list[float | None] = [None] * n
        batch_envs: list[object] = []
        batch_legal: list[list] = []
        batch_map: list[int] = []  # maps batch position → leaf_data index

        for i, (_leaf_node, _path, sim_env, is_terminal) in enumerate(leaf_data):
            if is_terminal:
                leaf_values[i] = _terminal_value_bare(sim_env)
            else:
                moves = sim_env.legal_actions()
                if not moves:
                    leaf_values[i] = -1.0
                else:
                    batch_envs.append(sim_env)
                    batch_legal.append(moves)
                    batch_map.append(i)

        # Phase 3: batch inference for all non-terminal leaves
        if batch_envs:
            output = _batch_infer(batch_envs, batch_legal, model, device)
            for j, i in enumerate(batch_map):
                leaf_node, _path, _sim_env, _is_terminal = leaf_data[i]
                leaf_moves = batch_legal[j]
                policy_logits = output["policy_logits"][j, : len(leaf_moves)]
                leaf_values[i] = float(output["value"][j].item())

                priors = _softmax(policy_logits)
                for k, move in enumerate(leaf_moves):
                    leaf_node.children[move.action_id] = MCTSNode(prior=float(priors[k]))

        # Phase 4: backprop all leaves (removes virtual loss, records real values)
        for i, (_leaf_node, path, _sim_env, _is_terminal) in enumerate(leaf_data):
            if leaf_values[i] is None:
                raise RuntimeError(f"leaf {i} has no value — this is a bug")
            _backprop_batch(path, leaf_values[i])

        remaining -= n

    # --- aggregate results ---
    visit_counts: dict[int, int] = {}
    q_values: dict[int, float] = {}
    total_visits = 0
    for action_id, child in root.children.items():
        visit_counts[action_id] = child.visit_count
        q_values[action_id] = child.q()
        total_visits += child.visit_count

    action_ids = [m.action_id for m in legal_actions]
    if total_visits > 0:
        policy_probs = [
            float(visit_counts.get(aid, 0)) / total_visits for aid in action_ids
        ]
    else:
        # Degenerate case — should not happen with num_simulations > 0
        policy_probs = [1.0 / len(action_ids)] * len(action_ids)

    # Compute root value as the visit-weighted average of children's Q,
    # since root.total_value is never set during backprop (only edges get
    # value updates; the root node itself is never a child of anyone).
    if total_visits > 0:
        root_value = sum(
            child.q() * child.visit_count for child in root.children.values()
        ) / total_visits
    else:
        root_value = float(_root_value)

    return MCTSResult(
        visit_counts=visit_counts,
        q_values=q_values,
        root_value=root_value,
        policy_probs=policy_probs,
        action_ids=action_ids,
        network_value=float(_root_value),
    )


def _maybe_autocast(model, device: torch.device):
    """Return ``torch.autocast`` if FP16 is enabled on a CUDA device, else a no-op context."""
    model_config = getattr(model, "config", None)
    use_fp16 = model_config is not None and getattr(model_config, "use_fp16", False)
    if use_fp16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return _null_context()


class _null_context:
    """A no-op context manager compatible with both ``with`` and ``async with``."""

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object) -> None:
        pass


# ---------------------------------------------------------------------------
# 单次模拟
# ---------------------------------------------------------------------------


def _collect_leaf(
    root: MCTSNode,
    env,
    c_puct: float,
) -> tuple[MCTSNode, list[tuple[MCTSNode, int]], bool]:
    """Walk from *root* down to a leaf, applying virtual loss at each edge.

    Returns:
        ``(leaf_node, path, is_terminal)`` where *path* is a list of
        ``(parent_node, action_id)`` pairs recording the walk and
        *is_terminal* is ``True`` when the game ended during the walk.
    """
    node = root
    path: list[tuple[MCTSNode, int]] = []
    done = False

    while node.children:
        action_id = _select_child(node, c_puct)
        child = node.children[action_id]
        child.virtual_loss_count += 1
        path.append((node, action_id))
        done = env.step_bare(action_id)
        node = child
        if done:
            break

    return node, path, done


def _backprop_batch(
    path: list[tuple[MCTSNode, int]],
    leaf_value: float,
) -> None:
    """Back-propagate *leaf_value* along *path*, removing virtual loss.

    Virtual loss counters that were incremented during leaf collection
    are decremented here, and real visit / value statistics are recorded
    on each traversed edge (from the parent's perspective).
    """
    v = leaf_value
    for parent, action_id in reversed(path):
        v = -v
        child = parent.children[action_id]
        child.virtual_loss_count -= 1
        child.visit_count += 1
        child.total_value += v


# ---------------------------------------------------------------------------
# PUCT 子节点选择
# ---------------------------------------------------------------------------


# PUCT 公式（含虚拟损失）
def _select_child(node: MCTSNode, c_puct: float) -> int:
    """Choose the child with the highest PUCT score.

    Accounts for ``virtual_loss_count`` so that parallel selection paths
    within the same batch are steered toward different branches.  When
    ``virtual_loss_count`` is 0 on every child this reduces to the
    classic PUCT formula.
    """
    total_visits = sum(c.visit_count + c.virtual_loss_count for c in node.children.values())
    sqrt_total = math.sqrt(max(total_visits, 1))

    best_action = -1
    best_score = -float("inf")
    for action_id, child in node.children.items():
        effective_n = child.visit_count + child.virtual_loss_count
        if effective_n > 0:
            # virtual loss = -1: each pending eval is assumed to be a loss
            effective_q = (child.total_value - child.virtual_loss_count) / effective_n
        else:
            effective_q = 0.0
        exploit = effective_q
        explore = c_puct * child.prior * sqrt_total / (1.0 + effective_n)
        score = exploit + explore
        if score > best_score:
            best_score = score
            best_action = action_id

    if best_action == -1:
        raise RuntimeError("select_child: node has no children")
    return best_action


# ---------------------------------------------------------------------------
# 网络推理辅助
# ---------------------------------------------------------------------------


def _raw_obs_to_tensor(raw_obs: dict) -> torch.Tensor:
    """Build a [6, 10, 10] board tensor directly from a raw observation dict.

    This avoids the intermediate nested-list allocation that
    ``StandardDraughtsEnv.observation()`` performs, and is designed to
    be called once per batch element inside ``_batch_infer``.
    """
    side_to_move = int(raw_obs["side_to_move"])
    squares = raw_obs["squares"]
    encoded = torch.zeros(6, 10, 10, dtype=torch.float32)

    for square_id in range(1, SQUARE_COUNT + 1):
        persp_sq = perspective_square_id(square_id, side_to_move)
        row, col = square_to_coord(persp_sq)
        piece = int(squares[square_id - 1])
        encoded[4, row, col] = 1.0  # channel 4: playable square
        if piece != EMPTY:
            channel = piece_channel(piece, side_to_move)
            encoded[channel, row, col] = 1.0

    return encoded


def _batch_infer(
    envs: list,
    legal_actions_list: list,
    model: CandidateMoveNet,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run the network on a batch of positions.

    Different positions may have different numbers of legal moves; the
    move-feature tensor is padded to ``max(M)`` within the batch and
    illegal slots are masked out.

    Args:
        envs: List of ``StandardDraughtsEnv`` at the positions to evaluate.
        legal_actions_list: ``legal_actions[i]`` is the move list for *envs[i]*.
        model: The network.
        device: ``torch.device`` for inference.

    Returns:
        ``{"policy_logits": [B, max_M], "value": [B]}``.
        Policy logits for padded (non-existent) slots are not
        meaningful; the caller should only use the first ``M_i``
        entries for each sample.
    """
    B = len(envs)
    max_M = max((len(la) for la in legal_actions_list), default=0)
    if max_M == 0:
        raise ValueError("_batch_infer called with positions that have no legal moves")

    # --- build tensors ---
    board_batch = torch.zeros(B, 6, 10, 10, dtype=torch.float32)
    moves_batch = torch.zeros(B, max_M, 45, dtype=torch.float32)
    mask_batch = torch.zeros(B, max_M, dtype=torch.float32)

    for i, env in enumerate(envs):
        raw_obs = env.raw_observation()
        board_batch[i] = _raw_obs_to_tensor(raw_obs)

        M = len(legal_actions_list[i])
        side = int(raw_obs["side_to_move"])
        move_dicts = [move_to_dict(m) for m in legal_actions_list[i]]
        features = encode_legal_moves(move_dicts, side_to_move=side)
        moves_batch[i, :M] = torch.tensor(features, dtype=torch.float32)
        mask_batch[i, :M] = 1.0

    board_batch = board_batch.to(device)
    moves_batch = moves_batch.to(device)
    mask_batch = mask_batch.to(device)

    with torch.no_grad(), _maybe_autocast(model, device):
        output = model(board_batch, moves_batch, mask_batch)

    return output  # {"policy_logits": [B, max_M], "value": [B]}


def _infer_network(
    env,
    model: CandidateMoveNet,
    legal_actions: list,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Run the network on a single position (convenience wrapper).

    Returns:
        policy_logits: 1-D tensor with raw logits for each legal action.
        value: scalar float in [-1, 1] (tanh output).
    """
    output = _batch_infer([env], [legal_actions], model, device)
    policy_logits = output["policy_logits"][0, : len(legal_actions)]
    value = float(output["value"][0].item())
    return policy_logits, value


# ---------------------------------------------------------------------------
# 终局评估
# ---------------------------------------------------------------------------


def _terminal_value(env) -> float:
    """Return the value of a terminal position from the current player's view."""
    return _terminal_value_bare(env)


def _terminal_value_bare(env) -> float:
    """Return terminal value using cached observation from ``step_bare()``.
    Avoids the C ``observation()`` call that ``raw_observation()`` would make."""
    raw = env._last_observation_dict
    if raw is None:
        raw = env.raw_observation()
    result = int(raw["game_result"])
    if result == 3:
        return 0.0
    return -1.0


# ---------------------------------------------------------------------------
# 数学辅助
# ---------------------------------------------------------------------------


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax along the last dimension."""
    logits = logits - logits.max()
    exp = torch.exp(logits)
    return exp / exp.sum()


def _add_dirichlet_noise(priors: torch.Tensor, config: MCTSConfig) -> torch.Tensor:
    """Mix Dirichlet noise into the root prior distribution.

    Returns a new tensor; does not modify *priors* in-place.
    """
    alpha = torch.full_like(priors, config.dirichlet_alpha)
    dirichlet = torch.distributions.Dirichlet(alpha)
    noise = dirichlet.sample()
    return (1.0 - config.dirichlet_epsilon) * priors + config.dirichlet_epsilon * noise


# ---------------------------------------------------------------------------
# 环境操作辅助
# ---------------------------------------------------------------------------


def _step_env(env, action_id: int) -> None:
    """Step the environment and ignore the returned observation."""
    env.step(action_id)


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------


def load_mcts_config(config_path: Optional[str] = None) -> MCTSConfig:
    """Load MCTSConfig from a YAML-like file, falling back to defaults."""
    path = Path(config_path) if config_path is not None else DEFAULT_MCTS_CONFIG_PATH
    default_config = MCTSConfig()
    defaults = {
        "num_simulations": default_config.num_simulations,
        "c_puct": default_config.c_puct,
        "temperature": default_config.temperature,
        "dirichlet_alpha": default_config.dirichlet_alpha,
        "dirichlet_epsilon": default_config.dirichlet_epsilon,
        "add_noise": default_config.add_noise,
        "batch_size": default_config.batch_size,
    }
    values = dict(defaults)
    values.update(_load_scalar_yaml(path, defaults, "mcts"))
    config = MCTSConfig(**values)
    config.validate()
    return config


def _load_scalar_yaml(
    path: Path, defaults: dict[str, object], config_name: str
) -> dict[str, object]:
    """Minimal YAML-like config parser (same convention as agents.py / model.py)."""
    if not path.exists():
        raise FileNotFoundError(f"{config_name} config not found: {path}")

    values: dict[str, object] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"invalid config line {line_number}: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key not in defaults:
            raise ValueError(
                f"unknown {config_name} config key on line {line_number}: {key}"
            )
        default_value = defaults[key]
        try:
            if isinstance(default_value, int) and not isinstance(default_value, bool):
                values[key] = int(value)
            elif isinstance(default_value, float):
                values[key] = float(value)
            else:
                values[key] = value
        except ValueError as exc:
            raise ValueError(
                f"{config_name} config value has wrong type on line {line_number}"
            ) from exc
    return values
