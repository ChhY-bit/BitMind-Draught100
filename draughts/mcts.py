from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .move_encoding import encode_legal_moves, move_to_dict
from .model import CandidateMoveNet

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

    def validate(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
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
    children: dict[int, MCTSNode] = field(default_factory=dict)

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

    # --- run simulations ---
    for _ in range(config.num_simulations):
        sim_env = env.clone()
        _simulate(root, sim_env, model, config, device)

    # --- aggregate results ---
    visit_counts: dict[int, int] = {}
    total_visits = 0
    for action_id, child in root.children.items():
        visit_counts[action_id] = child.visit_count
        total_visits += child.visit_count

    action_ids = [m.action_id for m in legal_actions]
    if total_visits > 0:
        policy_probs = [
            float(visit_counts.get(aid, 0)) / total_visits for aid in action_ids
        ]
    else:
        # Degenerate case — should not happen with num_simulations > 0
        policy_probs = [1.0 / len(action_ids)] * len(action_ids)

    return MCTSResult(
        visit_counts=visit_counts,
        root_value=root.q(),
        policy_probs=policy_probs,
        action_ids=action_ids,
    )


# ---------------------------------------------------------------------------
# 单次模拟
# ---------------------------------------------------------------------------


def _simulate(
    root: MCTSNode,
    env,
    model: CandidateMoveNet,
    config: MCTSConfig,
    device: torch.device,
) -> None:
    """Run one MCTS simulation: selection → expansion → evaluation → backprop.

    *env* is a clone that the simulation mutates (and discards afterwards).
    """
    node = root
    # path records (parent_node, action_id) so we can update edge
    # statistics during backprop.
    path: list[tuple[MCTSNode, int]] = []

    # ---- selection: walk down the existing tree ----
    while node.children and not env.is_terminal():
        action_id = _select_child(node, config.c_puct)
        path.append((node, action_id))
        _step_env(env, action_id)
        node = node.children[action_id]

    # ---- evaluation ----
    if env.is_terminal():
        leaf_value = _terminal_value(env)
    else:
        # expansion + network evaluation
        legal_actions = env.legal_actions()
        if legal_actions:
            policy_logits, value = _infer_network(env, model, legal_actions, device)
            priors = _softmax(policy_logits)
            for i, move in enumerate(legal_actions):
                node.children[move.action_id] = MCTSNode(prior=float(priors[i]))
            leaf_value = value
        else:
            # No legal moves but not terminal — treat as lost for current side
            leaf_value = -1.0

    # ---- backprop: walk up, flipping value at each level ----
    # leaf_value is from the leaf player's perspective.
    # Flip once to get the perspective of the player who made the
    # last action (the edge leading to the leaf), then continue.
    v = leaf_value
    for parent, action_id in reversed(path):
        v = -v  # switch to parent's perspective
        child = parent.children[action_id]
        child.visit_count += 1
        child.total_value += v
    root.visit_count += 1


# ---------------------------------------------------------------------------
# PUCT 子节点选择
# ---------------------------------------------------------------------------


def _select_child(node: MCTSNode, c_puct: float) -> int:
    """Choose the child with the highest PUCT score."""
    total_visits = sum(c.visit_count for c in node.children.values())
    # Use sqrt(total_visits) if there have been visits, else 0.
    sqrt_total = math.sqrt(max(total_visits, 1))

    best_action = -1
    best_score = -float("inf")
    for action_id, child in node.children.items():
        exploit = child.q()
        explore = c_puct * child.prior * sqrt_total / (1.0 + child.visit_count)
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


def _infer_network(
    env,
    model: CandidateMoveNet,
    legal_actions: list,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Run the network on a single position.

    Returns:
        policy_logits: 1-D tensor with raw logits for each legal action.
        value: scalar float in [-1, 1] (tanh output).
    """
    raw_obs = env.raw_observation()
    side_to_move = int(raw_obs["side_to_move"])
    move_dicts = [move_to_dict(m) for m in legal_actions]

    observation_tensor = torch.tensor(
        [env.observation()], dtype=torch.float32, device=device
    )
    move_features_tensor = torch.tensor(
        [encode_legal_moves(move_dicts, side_to_move=side_to_move)],
        dtype=torch.float32,
        device=device,
    )
    move_mask = torch.ones(
        (1, len(legal_actions)), dtype=torch.float32, device=device
    )

    with torch.no_grad():
        output = model(observation_tensor, move_features_tensor, move_mask)

    policy_logits = output["policy_logits"][0]  # [M]
    value = float(output["value"][0].item())
    return policy_logits, value


# ---------------------------------------------------------------------------
# 终局评估
# ---------------------------------------------------------------------------


def _terminal_value(env) -> float:
    """Return the value of a terminal position from the current player's view.

    In draughts the game ends when the side to move has no legal moves
    (they lose), or by draw (currently unimplemented in the engine).
    """
    raw = env.raw_observation()
    result = int(raw["game_result"])
    if result == 3:  # DRAW
        return 0.0
    # The side to move has no moves → they lost.
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
