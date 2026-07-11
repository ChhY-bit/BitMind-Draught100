# Re-export core symbols from draughts/ for backward-compatible imports.
# New code should import from draughts/ directly.
from draughts.agents import (
    AlphaBetaAgent,
    RandomAgent,
)
from draughts.dataset import DraughtsJsonlDataset
from draughts.env import StandardDraughtsEnv

__all__ = [
    "AlphaBetaAgent",
    "DraughtsJsonlDataset",
    "RandomAgent",
    "StandardDraughtsEnv",
]
