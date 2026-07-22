from __future__ import annotations

import tkinter as tk
from typing import Optional

from ..utils import BLACK_SIDE, BOARD_PIXELS, CELL_SIZE


class EvalBar(tk.Canvas):
    """Vertical evaluation bar showing win probability from black's view.

    The top portion (dark fill) represents black's win chance; the bottom
    portion (light fill) represents white's.  A centred label shows the
    current player's win percentage.  Value changes are animated with a
    smooth ease-in-out transition (~350 ms).
    """

    BAR_WIDTH = 38
    BAR_HEIGHT = BOARD_PIXELS
    _ANIM_STEPS = 20
    _ANIM_INTERVAL = 17  # ms (~60 fps)

    def __init__(self, master: tk.Widget, **kwargs) -> None:
        super().__init__(
            master,
            width=self.BAR_WIDTH,
            height=self.BAR_HEIGHT,
            background="#f0f2f5",
            highlightthickness=1,
            highlightbackground="#b0b8c1",
            **kwargs,
        )
        self._value: float = 0.0
        self._side_to_move: int = BLACK_SIDE
        # Animation state
        self._anim_job: Optional[str] = None
        self._anim_target: float = 0.0
        self._anim_start: float = 0.0
        self._anim_step_i: int = 0
        self._display_value: float = 0.0

    def set_value(self, value: float, side_to_move: int) -> None:
        """Update the evaluation with smooth transition.

        Args:
            value: Network/MCTS evaluation **from the current player's
                   perspective**, in [-1, 1].  +1 = current player winning.
            side_to_move: ``BLACK_SIDE`` (0) or ``WHITE_SIDE`` (1).
        """
        target = max(-1.0, min(1.0, float(value)))
        if self._anim_job is not None:
            self.after_cancel(self._anim_job)
            self._anim_job = None
        # When the perspective flips (e.g. after a move), convert the
        # currently displayed value so the animation starts from the
        # same visual position instead of jumping.
        if self._side_to_move != side_to_move:
            self._anim_start = -self._display_value
            self._display_value = -self._display_value
        else:
            self._anim_start = self._display_value
        self._anim_target = target
        self._side_to_move = side_to_move  # target perspective
        self._anim_step_i = 0
        self._anim_tick()

    def clear(self) -> None:
        """Reset the bar to the neutral (even) position."""
        self.set_value(0.0, BLACK_SIDE)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _anim_tick(self) -> None:
        if self._anim_step_i >= self._ANIM_STEPS:
            self._display_value = self._anim_target
            self._value = self._display_value
            self._anim_job = None
            self._redraw()
            return

        # Smoothstep ease-in-out: 3t² - 2t³
        t = self._anim_step_i / self._ANIM_STEPS
        eased = 3.0 * t * t - 2.0 * t * t * t
        self._display_value = self._anim_start + (self._anim_target - self._anim_start) * eased
        self._value = self._display_value
        self._anim_step_i += 1
        self._redraw()
        self._anim_job = self.after(self._ANIM_INTERVAL, self._anim_tick)

    # ------------------------------------------------------------------
    # Internal drawing
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        self.delete("all")

        w = self.BAR_WIDTH
        actual_h = self.winfo_height()
        h = actual_h if actual_h > 1 else self.BAR_HEIGHT

        # --- value from black's perspective ---
        if self._side_to_move == BLACK_SIDE:
            value_black = self._value
        else:
            value_black = -self._value
        black_share = (value_black + 1.0) / 2.0  # [0, 1]

        # Clamp so the smaller slice never vanishes completely
        black_share = max(0.03, min(0.97, black_share))
        # Y coordinate of the boundary: black occupies the *top*
        # (0 → split_y), white the *bottom* (split_y → h).
        split_y = int(h * black_share)

        # Black portion (top)
        self.create_rectangle(0, 0, w, split_y, fill="#1a2332", outline="")
        # White portion (bottom)
        self.create_rectangle(0, split_y, w, h, fill="#f5f0e8", outline="")

        # Centre divider (neutral reference)
        self.create_line(0, h // 2, w, h // 2, fill="#8899aa", dash=(3, 4))

        # ── Confrontation marker at the split ──
        # Two opposing triangles (► ◄) pointing at each other across the
        # boundary, evoking a battle / stand-off.
        cx = w // 2
        tri_h = 7   # half-height of each triangle
        tri_w = 9   # half-width
        # Upper triangle ▼ (pointing down toward the split)
        self.create_polygon(
            cx - tri_w, split_y - tri_h,
            cx + tri_w, split_y - tri_h,
            cx, split_y,
            fill="#f2c94c", outline="#c99a2c",
        )
        # Lower triangle ▲ (pointing up toward the split)
        self.create_polygon(
            cx - tri_w, split_y + tri_h,
            cx + tri_w, split_y + tri_h,
            cx, split_y,
            fill="#f2c94c", outline="#c99a2c",
        )
