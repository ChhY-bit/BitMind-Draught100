from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from ..i18n import _tr_static
from ..utils import BLACK_SIDE


class ValueCurve(tk.Frame):
    """Scrollable line chart of position evaluation across plies.

    Plots black-win% (derived from the network value head) after each
    move, with a horizontal scrollbar for navigating long games.
    """

    POINT_SPACING = 30
    CURVE_HEIGHT = 160
    MARGIN_TOP = 12
    MARGIN_BOTTOM = 24
    MARGIN_LEFT = 42  # room for Y-axis % labels
    MARGIN_RIGHT = 24
    DOT_RADIUS = 3
    MAX_POINTS = 300

    # -- colour palette (light theme, matching the app) -------------------
    _BG = "#f5f6f8"
    _PLOT_BG = "#ffffff"
    _PLOT_BORDER = "#c8cdd5"
    _GRID_MINOR = "#e8ebf0"
    _GRID_MAJOR = "#c0c6d0"
    _AXIS_LABEL = "#889099"
    _LINE = "#3b6fb6"
    _LINE_FILL = "#dce8f5"
    _DOT_BORDER = "#ffffff"
    _DOT_FILL_HIGH = "#2c5282"
    _DOT_FILL_MID = "#7a8ba0"
    _DOT_FILL_LOW = "#c09070"
    _MARKER = "#e6b422"
    _MIN_PLOT_PX = 240  # minimum plot-area width so empty-state text fits

    def __init__(
        self, parent: tk.Widget, on_point_click=None, app=None,
    ) -> None:
        super().__init__(parent)
        self._on_point_click_cb = on_point_click
        self._app = app
        self._points: list[tuple[int, float, int, str]] = []  # (ply, value, side, notation)
        self._current_ply: int = -1
        self._dot_positions: list[tuple[int, float, float]] = []  # (ply, canvas_x, canvas_y)

        bar_frame = ttk.Frame(self)
        bar_frame.pack(side="bottom", fill="x")

        self.canvas = tk.Canvas(
            self, height=self._total_h,
            bg=self._BG, highlightthickness=0,
        )
        self.canvas.pack(side="top", fill="x", expand=False)

        self.scrollbar = ttk.Scrollbar(
            bar_frame, orient="horizontal", command=self.canvas.xview,
        )
        self.scrollbar.pack(side="left", fill="x", expand=True)
        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self._redraw()

    # -- layout helpers ---------------------------------------------------

    @property
    def _total_h(self) -> int:
        return self.CURVE_HEIGHT + self.MARGIN_TOP + self.MARGIN_BOTTOM

    @property
    def _plot_top(self) -> int:
        return self.MARGIN_TOP

    @property
    def _plot_bottom(self) -> int:
        return self.MARGIN_TOP + self.CURVE_HEIGHT

    @property
    def _plot_left(self) -> int:
        return self.MARGIN_LEFT

    def _plot_right(self) -> int:
        canvas_w = self.canvas.winfo_width()
        if canvas_w < 20:  # not yet drawn — fall back to hard-coded minimum
            canvas_w = self._MIN_PLOT_PX + self.MARGIN_LEFT + self.MARGIN_RIGHT
        available_w = max(canvas_w - self.MARGIN_LEFT - self.MARGIN_RIGHT, self._MIN_PLOT_PX)
        needed_w = len(self._points) * self.POINT_SPACING
        return self._plot_left + max(needed_w, available_w)

    # -- value → coordinate helpers ---------------------------------------

    @staticmethod
    def _value_to_black_pct(value: float, side: int) -> float:
        """Convert value (∈[-1,1], current-player perspective) to black win%."""
        if side == BLACK_SIDE:
            return (1.0 + value) / 2.0 * 100.0
        else:
            return (1.0 - value) / 2.0 * 100.0

    def _value_to_y(self, value: float, side: int) -> float:
        black_pct = self._value_to_black_pct(value, side)
        return self._plot_bottom - (black_pct / 100.0) * self.CURVE_HEIGHT

    def _point_x(self, idx: int) -> float:
        return self._plot_left + idx * self.POINT_SPACING + self.POINT_SPACING / 2

    # -- public API -------------------------------------------------------

    def add_point(self, ply: int, value: float, side: int, notation: str = "") -> None:
        self._points.append((ply, float(value), int(side), notation))
        if len(self._points) > self.MAX_POINTS:
            self._points = self._points[-self.MAX_POINTS:]
        self._redraw()

    def remove_last(self) -> None:
        if self._points:
            self._points.pop()
            self._redraw()

    def set_current_ply(self, ply: int) -> None:
        self._current_ply = ply
        self._redraw()

    def clear(self) -> None:
        self._points.clear()
        self._current_ply = -1
        self._dot_positions.clear()
        self._redraw()

    def update_all(self, history) -> None:
        """Bulk-replace all points (used after undo / rewind)."""
        self._points = list(history)
        if len(self._points) > self.MAX_POINTS:
            self._points = self._points[-self.MAX_POINTS:]
        self._redraw()

    # -- font helpers ---------------------------------------------------

    def _cjk_font(self, size: int = 7, bold: bool = False) -> tuple:
        """Build a font spec using the app's resolved CJK family."""
        family = getattr(self._app, "_cjk_family", None) if self._app else None
        if family is None:
            return ("TkDefaultFont", size, "bold") if bold else ("TkDefaultFont", size)
        return (family, size, "bold") if bold else (family, size)

    # -- drawing -----------------------------------------------------------

    def _redraw(self) -> None:
        self.canvas.delete("all")
        self._dot_positions.clear()

        total_w = self._plot_right() + self.MARGIN_RIGHT
        self.canvas.configure(scrollregion=(0, 0, total_w, self._total_h))

        plot_l = self._plot_left
        plot_t = self._plot_top
        plot_b = self._plot_bottom
        plot_r = max(self._plot_right(), plot_l + self._MIN_PLOT_PX)

        # Background
        self.canvas.create_rectangle(0, 0, total_w, self._total_h,
                                     fill=self._BG, outline="")
        # Plot area
        self.canvas.create_rectangle(plot_l, plot_t, plot_r, plot_b,
                                     fill=self._PLOT_BG, outline=self._PLOT_BORDER)

        # ── Y-axis: left-side axis line + tick marks + labels ──
        self.canvas.create_line(plot_l, plot_t, plot_l, plot_b,
                                fill=self._PLOT_BORDER, width=1)
        for pct in (0, 25, 50, 75, 100):
            y = plot_b - (pct / 100.0) * self.CURVE_HEIGHT
            is_major = (pct == 50)
            dash = (6, 4) if is_major else (2, 4)
            color = self._GRID_MAJOR if is_major else self._GRID_MINOR
            self.canvas.create_line(plot_l, y, plot_r, y, fill=color, dash=dash)
            self.canvas.create_line(plot_l - 4, y, plot_l, y, fill=self._PLOT_BORDER)
            self.canvas.create_text(plot_l - 8, y, text=f"{pct}%",
                                    anchor="e", font=self._cjk_font(7),
                                    fill=self._AXIS_LABEL)

        # ── X-axis: bottom line ──
        self.canvas.create_line(plot_l, plot_b, plot_r, plot_b,
                                fill=self._PLOT_BORDER, width=1)

        # ── Right-side labels ("Black Wins" / "White Wins") ──
        tr = self._app._tr if self._app else _tr_static
        self.canvas.create_text(
            plot_r + 6, plot_t + 3,
            text=tr("curve_black_wins"),
            anchor="nw", font=self._cjk_font(7, bold=True),
            fill=self._DOT_FILL_HIGH,
        )
        self.canvas.create_text(
            plot_r + 6, plot_b - 3,
            text=tr("curve_white_wins"),
            anchor="sw", font=self._cjk_font(7, bold=True),
            fill=self._DOT_FILL_LOW,
        )

        # ── Empty state ──
        if not self._points:
            tr = self._app._tr if self._app else _tr_static
            msg = tr("curve_empty") if self._app else "(No AI — curve unavailable)"
            self.canvas.create_text(
                (plot_l + plot_r) / 2, (plot_t + plot_b) / 2,
                text=msg, font=self._cjk_font(10), fill=self._AXIS_LABEL,
            )
            return

        # ── X-axis labels (every 5 plies) ──
        for i, (ply, _v, _s, _n) in enumerate(self._points):
            if ply % 5 == 0 or ply == 1:
                x = self._point_x(i)
                self.canvas.create_line(x, plot_b, x, plot_b + 4,
                                        fill=self._PLOT_BORDER)
                self.canvas.create_text(x, plot_b + 13, text=str(ply),
                                        font=self._cjk_font(7),
                                        fill=self._AXIS_LABEL, anchor="n")

        # ── Build data arrays ──
        coords: list[float] = []
        dot_data: list[tuple[float, float, float, int]] = []  # (x, y, black_pct, ply)
        for i, (_ply, value, side, _notation) in enumerate(self._points):
            x = self._point_x(i)
            y = self._value_to_y(value, side)
            coords.extend([x, y])
            black_pct = self._value_to_black_pct(value, side)
            dot_data.append((x, y, black_pct, self._points[i][0]))

        # ── Area fill under the curve (semi-transparent overlay) ──
        if len(coords) >= 4:
            mid_y = self._plot_bottom - self.CURVE_HEIGHT / 2.0
            poly_pts: list[float] = []
            for j in range(0, len(coords), 2):
                px, py = coords[j], coords[j + 1]
                poly_pts.extend([px, py])
            poly_pts.extend([coords[-2], mid_y, coords[0], mid_y])
            self.canvas.create_polygon(
                *poly_pts, fill=self._LINE_FILL, outline="",
            )

        # ── Connecting line ──
        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill=self._LINE, width=2.0,
                                    capstyle="round", joinstyle="round")

        # ── Data dots ──
        for x, y, black_pct, ply in dot_data:
            if black_pct >= 58:
                color = self._DOT_FILL_HIGH
            elif black_pct >= 42:
                color = self._DOT_FILL_MID
            else:
                color = self._DOT_FILL_LOW
            r = self.DOT_RADIUS
            # White border
            self.canvas.create_oval(
                x - r - 1, y - r - 1, x + r + 1, y + r + 1,
                fill=self._DOT_BORDER, outline=self._DOT_BORDER,
                tags=("dot",),
            )
            # Coloured centre
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color, outline=color, tags=("dot",),
            )
            self._dot_positions.append((ply, x, y))

        # ── Current-ply marker ──
        if 0 < self._current_ply <= len(self._points):
            idx = self._current_ply - 1
            if 0 <= idx < len(self._points):
                x = self._point_x(idx)
                self.canvas.create_line(x, plot_t, x, plot_b,
                                        fill=self._MARKER, width=2, dash=(4, 3))

        # Auto-scroll to latest
        self.canvas.xview_moveto(1.0)

    # -- click handling ---------------------------------------------------

    def _on_canvas_click(self, event: tk.Event) -> None:
        if not self._dot_positions or not self._on_point_click_cb:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        closest_ply = -1
        closest_dist = float("inf")
        for ply, dx, dy in self._dot_positions:
            dist = ((canvas_x - dx) ** 2 + (canvas_y - dy) ** 2) ** 0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_ply = ply

        if closest_ply >= 0 and closest_dist <= 10:
            self._on_point_click_cb(closest_ply)
