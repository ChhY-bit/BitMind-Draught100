from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PlayerAnalysis:
    """Rich analysis data produced by an AI player during move selection.

    This is the data-transfer object shared by all agent types (local
    and remote).  Widgets consume this struct without knowing how it was
    produced.
    """

    value: Optional[float] = None
    search_value: Optional[float] = None  # MCTS root Q (may differ from network value)
    policy_probs: Optional[dict[int, float]] = None
    search_info: Optional[dict[str, object]] = None
    analysed_action_ids: Optional[set[int]] = None
    visit_counts: Optional[dict[int, int]] = None
    q_values: Optional[dict[int, float]] = None
