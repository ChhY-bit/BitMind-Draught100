from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Dict, List, Optional, Sequence

import torch

from draughts.agents import AlphaBetaAgent, MCTSAgent, SupervisedAgent
from draughts.env import EncodedObservationAdapter, StandardDraughtsEnv
from draughts.mcts import _maybe_autocast
from draughts.move_encoding import encode_legal_moves, move_to_dict
from draughts.search_control import check_cancelled
from draughts_env import DraughtsEnv, Move

from .protocol import PlayerAnalysis
from .utils import _softmax_list, find_move_by_action_id


def available_devices() -> List[str]:
    """Explicit device identifiers available to this process."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    return devices


# ---------------------------------------------------------------------------
# Player adapters — uniform choose_move() contract for the GUI
# ---------------------------------------------------------------------------


@dataclass
class HumanPlayer:
    name: str
    kind: str = "human"

    def __post_init__(self) -> None:
        self.last_analysis: PlayerAnalysis = PlayerAnalysis()

    def is_human(self) -> bool:
        return True


@dataclass
class RandomPlayer:
    """GUI player adapter for random move selection.

    This is distinct from draughts.agents.RandomAgent — it uses the GUI's
    choose_move() contract (returns Move or None) rather than the standard
    Agent.select_action() interface (returns action_id).
    """

    name: str
    rng: random.Random
    kind: str = "random"

    def __post_init__(self) -> None:
        self.last_analysis: PlayerAnalysis = PlayerAnalysis()

    def is_human(self) -> bool:
        return False

    def choose_move(self, moves: Sequence[Move], env: Optional[DraughtsEnv] = None,
                    *, cancel_event: Optional[Event] = None) -> Optional[Move]:
        check_cancelled(cancel_event)
        if not moves:
            return None
        self.last_analysis = PlayerAnalysis()
        return self.rng.choice(list(moves))


@dataclass
class SupervisedPlayer:
    name: str
    checkpoint_path: str
    kind: str = "supervised"
    force_fp16: bool = False
    force_compile: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.agent = SupervisedAgent(
            self.checkpoint_path, device=self.device,
            force_fp16=self.force_fp16,
            force_compile=self.force_compile,
        )
        self.last_analysis: PlayerAnalysis = PlayerAnalysis()

    def is_human(self) -> bool:
        return False

    def choose_move(self, moves: Sequence[Move], env: Optional[DraughtsEnv] = None,
                    *, cancel_event: Optional[Event] = None) -> Optional[Move]:
        if not moves:
            return None
        if env is None:
            raise ValueError("监督模型AI需要当前棋局环境")
        action_id = self.agent.select_action(
            env.observation(),
            moves,
            EncodedObservationAdapter(env),
            cancel_event=cancel_event,
        )
        self._capture_analysis(moves)
        return find_move_by_action_id(moves, action_id)

    def _capture_analysis(self, moves: Sequence[Move]) -> None:
        logits = getattr(self.agent, "last_policy_logits", None)
        value = getattr(self.agent, "last_value", None)
        if logits is not None and value is not None and len(logits) == len(moves):
            probs = _softmax_list(logits)
            self.last_analysis = PlayerAnalysis(
                value=value,
                policy_probs={
                    moves[i].action_id: probs[i] for i in range(len(moves))
                },
                analysed_action_ids=set(m.action_id for m in moves),
            )
        else:
            self.last_analysis = PlayerAnalysis()

    def infer_value(self, gui_env: DraughtsEnv) -> Optional[float]:
        """Run a single forward pass on the current position.

        Used by the value curve to record a post-move evaluation even
        when it is a human's turn (mixed data-collection strategy).
        """
        obs = gui_env.observation()
        moves = gui_env.legal_moves()
        if not moves:
            return None
        adapter = EncodedObservationAdapter(gui_env)
        side = int(obs["side_to_move"])
        autocast_ctx = _maybe_autocast(self.agent.model, self.agent.device_obj)
        with torch.no_grad(), autocast_ctx:
            obs_t = torch.tensor(
                [adapter.observation()], dtype=torch.float32,
                device=self.agent.device_obj,
            )
            move_dicts = [move_to_dict(m) for m in moves]
            move_t = torch.tensor(
                [encode_legal_moves(move_dicts, side_to_move=side)],
                dtype=torch.float32, device=self.agent.device_obj,
            )
            mask_t = torch.ones(
                1, len(moves), dtype=torch.float32, device=self.agent.device_obj,
            )
            output = self.agent.model(obs_t, move_t, mask_t)
            return float(output["value"][0].item())


@dataclass
class AlphaBetaPlayer:
    name: str
    kind: str = "alphabeta"

    def __post_init__(self) -> None:
        self.agent = AlphaBetaAgent()
        self.last_analysis: PlayerAnalysis = PlayerAnalysis()

    def is_human(self) -> bool:
        return False

    def choose_move(self, moves: Sequence[Move], env: Optional[DraughtsEnv] = None,
                    *, cancel_event: Optional[Event] = None) -> Optional[Move]:
        if not moves:
            return None
        if env is None:
            raise ValueError("AlphaBeta AI需要当前棋局环境")
        wrapped = GuiEnvAdapter(env)
        obs = wrapped.raw_observation()
        action_id = self.agent.select_action(obs, moves, wrapped, cancel_event=cancel_event)
        self.last_analysis = PlayerAnalysis(
            search_info={"depth": self.agent.config.search_depth},
        )
        return find_move_by_action_id(moves, action_id)


@dataclass
class MCTSPlayer:
    name: str
    checkpoint_path: str
    config_path: Optional[str] = None
    kind: str = "mcts"
    force_fp16: bool = False
    force_compile: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.agent = MCTSAgent(
            self.checkpoint_path,
            config_path=self.config_path,
            device=self.device,
            force_fp16=self.force_fp16,
            force_compile=self.force_compile,
        )
        self.last_analysis: PlayerAnalysis = PlayerAnalysis()

    def is_human(self) -> bool:
        return False

    def choose_move(self, moves: Sequence[Move], env: Optional[DraughtsEnv] = None,
                    *, cancel_event: Optional[Event] = None) -> Optional[Move]:
        check_cancelled(cancel_event)
        if not moves:
            return None
        if env is None:
            raise ValueError("MCTS AI需要当前棋局环境")
        obs = env.observation()
        # Mirror the GUI env state into a StandardDraughtsEnv for MCTS search.
        search_env = StandardDraughtsEnv()
        search_env.set_state(
            squares=list(obs["squares"]),
            side_to_move=int(obs["side_to_move"]),
        )
        action_id = self.agent.select_action(
            search_env.raw_observation(),
            search_env.legal_actions(),
            search_env,
            cancel_event=cancel_event,
        )
        self._capture_analysis(moves)
        return find_move_by_action_id(moves, action_id)

    def _capture_analysis(self, moves: Sequence[Move]) -> None:
        result = self.agent.last_result
        if result is None:
            self.last_analysis = PlayerAnalysis()
            return

        search_lookup: dict[int, float] = dict(
            zip(result.action_ids, result.policy_probs)
        )
        probs_dict: dict[int, float] = {}
        visits_dict: dict[int, int] = {}
        qvals_dict: dict[int, float] = {}
        for move in moves:
            prob = search_lookup.get(move.action_id)
            if prob is not None:
                probs_dict[move.action_id] = prob
            if result.visit_counts:
                visits_dict[move.action_id] = result.visit_counts.get(move.action_id, 0)
            if result.q_values:
                qvals_dict[move.action_id] = result.q_values.get(move.action_id, 0.0)

        self.last_analysis = PlayerAnalysis(
            value=result.network_value,
            search_value=result.root_value,
            policy_probs=probs_dict,
            search_info={
                "simulations": self.agent.mcts_config.num_simulations,
                "c_puct": self.agent.mcts_config.c_puct,
            },
            analysed_action_ids=set(m.action_id for m in moves),
            visit_counts=visits_dict if visits_dict else None,
            q_values=qvals_dict if qvals_dict else None,
        )

    def infer_value(self, gui_env: DraughtsEnv) -> Optional[float]:
        """Run a single forward pass on the current position via MCTS net."""
        from draughts.mcts import _infer_network

        obs = gui_env.observation()
        search_env = StandardDraughtsEnv()
        search_env.set_state(
            squares=list(obs["squares"]),
            side_to_move=int(obs["side_to_move"]),
        )
        legal_actions = search_env.legal_actions()
        if not legal_actions:
            return None
        _, value = _infer_network(
            search_env, self.agent.model,
            legal_actions, self.agent.device_obj,
        )
        return value


@dataclass
class GuiEnvAdapter:
    """Wraps a C-level DraughtsEnv so it exposes the interface expected by
    alphabeta / MCTS agents (clone, step, raw_observation,
    legal_actions)."""

    _env: DraughtsEnv

    def clone(self) -> "GuiEnvAdapter":
        return GuiEnvAdapter(self._env.clone())

    def step(self, action_id: int):
        obs, info = self._env.step(action_id)
        return obs, info["reward"], info["done"], info

    def raw_observation(self) -> Dict[str, object]:
        return self._env.observation()

    def legal_actions(self) -> List[Move]:
        return self._env.legal_moves()
