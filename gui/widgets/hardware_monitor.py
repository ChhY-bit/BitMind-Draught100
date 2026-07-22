from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from typing import Optional

from ..i18n import _tr_static


class HardwareMonitor(tk.Canvas):
    """Compact system monitor with animated bar charts.

    2 columns × 3 rows: CPU usage, sys RAM, proc CPU | GPU usage, VRAM, temp.
    Data sources: /proc/stat, /proc/meminfo, /proc/self/stat, nvidia-smi,
    torch.cuda — zero new dependencies.

    Data collection runs in a daemon thread so nvidia-smi never blocks the
    Tk event loop.  Canvas items are created once then updated incrementally.
    """

    _BAR_H = 12
    _ROW_H = 20
    _PAD_T = 6
    _PAD_B = 6
    _PAD_L = 4
    _PAD_R = 4
    _LABEL_W = 88
    _GAP_COL = 24

    def __init__(self, parent, draughts_app=None, height=120):
        super().__init__(parent, bg="#f5f5f5", highlightthickness=1,
                         highlightbackground="#c8cdd5", height=height)
        self._app = draughts_app
        self._metrics: list[dict] = [
            {"key": "cpu",  "current": 0.0, "target": 0.0, "fmt": "{:.0f}%",  "frac": 0.0,
             "bar_bg": None, "bar_fill": None, "text": None, "label": None},
            {"key": "ram",  "current": 0.0, "target": 0.0, "fmt": "{:.1f}G",  "frac": 0.0,
             "bar_bg": None, "bar_fill": None, "text": None, "label": None},
            {"key": "proc", "current": 0.0, "target": 0.0, "fmt": "{:.0f}%",  "frac": 0.0,
             "bar_bg": None, "bar_fill": None, "text": None, "label": None},
            {"key": "gpu",  "current": 0.0, "target": 0.0, "fmt": "{:.0f}%",  "frac": 0.0,
             "bar_bg": None, "bar_fill": None, "text": None, "label": None},
            {"key": "vram", "current": 0.0, "target": 0.0, "fmt": "{:.1f}G",  "frac": 0.0,
             "bar_bg": None, "bar_fill": None, "text": None, "label": None},
            {"key": "temp", "current": 0.0, "target": 0.0, "fmt": "{:.0f}°C", "frac": 0.0,
             "bar_bg": None, "bar_fill": None, "text": None, "label": None},
        ]
        self._gpu_available = False
        self._anim_id: Optional[str] = None
        self._anim_start = 0.0
        self._running = True
        self._items_created = False
        # Thread-safe: collector thread writes a fresh dict, main thread reads it.
        self._pending_data: Optional[dict] = None
        self._collect_lock = threading.Lock()

        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Destroy>", lambda _e: self._stop())

        # Start background collector + main-thread pick-up loop
        self._collector_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collector_thread.start()
        self._apply_loop()

    def _stop(self) -> None:
        self._running = False
        if self._anim_id:
            self.after_cancel(self._anim_id)
            self._anim_id = None

    # -- background data collection (never touches Tk) --------------------

    def _collect_loop(self) -> None:
        """Run in daemon thread: sample /proc + nvidia-smi, post results."""
        prev_cpu = None
        prev_proc_cpu = None
        while self._running:
            try:
                data = {}
                # CPU
                prev_cpu = self._sample_cpu(data, prev_cpu)
                # RAM
                self._sample_ram(data)
                # Proc CPU
                prev_proc_cpu = self._sample_proc_cpu(data, prev_proc_cpu)
                # GPU
                self._sample_gpu(data)
                # Publish
                self._pending_data = data
            except Exception:
                pass
            time.sleep(1.0)

    @staticmethod
    def _sample_cpu(data: dict, prev) -> tuple:
        try:
            with open("/proc/stat") as f:
                line = f.readline()
            parts = line.split()
            if parts[0] == "cpu":
                ticks = [int(x) for x in parts[1:]]
                idle, total = ticks[3], sum(ticks)
                if prev is not None:
                    p_idle, p_total = prev
                    d_idle = idle - p_idle
                    d_total = total - p_total
                    if d_total > 0:
                        data["cpu"] = (1.0 - d_idle / d_total) * 100.0
                return (idle, total)
        except Exception:
            pass
        return prev

    @staticmethod
    def _sample_ram(data: dict) -> None:
        try:
            total = available = 0
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        available = int(line.split()[1])
                    if total and available:
                        break
            if total > 0:
                used_gb = (total - available) / 1024.0 / 1024.0
                total_gb = total / 1024.0 / 1024.0
                data["ram"] = (used_gb, total_gb)
        except Exception:
            pass

    @staticmethod
    def _sample_proc_cpu(data: dict, prev) -> tuple:
        try:
            with open("/proc/self/stat") as f:
                stat = f.read().split()
            ticks = sum(int(stat[i]) for i in [13, 14, 15, 16])
            now = time.monotonic()
            if prev is not None:
                p_ticks, p_time = prev
                d_ticks = ticks - p_ticks
                d_time = now - p_time
                nprocs = os.cpu_count() or 1
                if d_time > 0:
                    pct = (d_ticks / os.sysconf("SC_CLK_TCK")) / d_time / nprocs * 100.0
                    data["proc"] = min(pct, 100.0 * nprocs)
            return (ticks, now)
        except Exception:
            pass
        return prev

    @staticmethod
    def _sample_gpu(data: dict) -> None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = [x.strip() for x in result.stdout.strip().split(",")]
                if len(parts) >= 4:
                    data["gpu"] = float(parts[0])
                    data["vram"] = (float(parts[1]) / 1024.0, float(parts[2]) / 1024.0)
                    data["temp"] = float(parts[3])
                    data["gpu_avail"] = True
                    return
        except Exception:
            pass
        # Fallback: torch.cuda
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(0) / 1024.0**3
                data["vram"] = (alloc, None)
                data["gpu"] = 0.0
                data["temp"] = 0.0
                data["gpu_avail"] = True
                return
        except Exception:
            pass
        data["gpu_avail"] = False

    # -- main-thread apply loop -------------------------------------------

    def _apply_loop(self) -> None:
        """Pick up data from the collector thread and update targets."""
        if not self._running:
            return
        data = self._pending_data
        if data is not None:
            self._pending_data = None
            # Apply CPU
            if "cpu" in data:
                self._set_target("cpu", data["cpu"])
            # Apply RAM
            if "ram" in data:
                used_gb, total_gb = data["ram"]
                self._set_target("ram", used_gb)
                m = next(x for x in self._metrics if x["key"] == "ram")
                m["fmt"] = f"{{:.1f}}/{total_gb:.1f}G"
                m["frac"] = used_gb / total_gb if total_gb > 0 else 0.0
            # Apply proc CPU
            if "proc" in data:
                self._set_target("proc", data["proc"])
            # Apply GPU
            gpu_avail = data.get("gpu_avail", False)
            self._gpu_available = gpu_avail
            if gpu_avail:
                if "gpu" in data:
                    self._set_target("gpu", data["gpu"])
                if "vram" in data:
                    vram_gb, vram_total = data["vram"]
                    self._set_target("vram", vram_gb)
                    m = next(x for x in self._metrics if x["key"] == "vram")
                    if vram_total is not None:
                        m["fmt"] = f"{{:.1f}}/{vram_total:.1f}G"
                        m["frac"] = vram_gb / vram_total if vram_total > 0 else 0.0
                    else:
                        m["fmt"] = "{:.1f}G"
                        m["frac"] = 0.0
                if "temp" in data:
                    self._set_target("temp", data["temp"])
                    m = next(x for x in self._metrics if x["key"] == "temp")
                    m["frac"] = data["temp"] / 100.0
        self.after(1000, self._apply_loop)

    def _set_target(self, key: str, value: float) -> None:
        for m in self._metrics:
            if m["key"] == key:
                m["target"] = value
                break
        self._start_animation()

    # -- animation (main thread) ------------------------------------------

    def _start_animation(self) -> None:
        if self._anim_id is not None:
            return
        self._anim_start = time.monotonic()
        self._animate()

    def _animate(self) -> None:
        self._anim_id = None
        now = time.monotonic()
        t = min((now - self._anim_start) / 0.25, 1.0)
        eased = 1.0 - (1.0 - t) * (1.0 - t)  # ease-out quad
        dirty = False
        for m in self._metrics:
            cur = m["current"]
            tgt = m["target"]
            if abs(cur - tgt) < 0.05:
                m["current"] = tgt
            else:
                m["current"] = cur + (tgt - cur) * eased
                dirty = True
        self._redraw()
        if dirty and t < 1.0:
            self._anim_id = self.after(50, self._animate)

    # -- drawing (incremental after first paint) --------------------------

    def _font(self, size=9, bold=False):
        family = getattr(self._app, "_cjk_family", None) if self._app else None
        if family is None:
            return ("TkDefaultFont", size, "bold") if bold else ("TkDefaultFont", size)
        return (family, size, "bold") if bold else (family, size)

    def _bar_widths(self, w: float, mid_x: float) -> tuple[float, float]:
        """Return (bar_max_w_left, bar_max_w_right) so neither side overflows.

        Each side reserves 55 px after the bar for the value label before
        hitting the divider (left) or the canvas edge (right).
        """
        divider_x = mid_x - self._GAP_COL / 2
        bar_x0_left = self._PAD_L + self._LABEL_W
        bar_x0_right = mid_x + self._GAP_COL / 2 + self._LABEL_W
        left = (divider_x - 55) - bar_x0_left
        right = (w - self._PAD_R - 55) - bar_x0_right
        return max(left, 20), max(right, 20)

    def _redraw(self) -> None:
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 20 or h < 20:
            return
        tr = self._app._tr if self._app else _tr_static

        mid_x = w / 2 + self._GAP_COL / 2
        bar_max_w_left, bar_max_w_right = self._bar_widths(w, mid_x)

        if not self._items_created:
            self._create_items(w, h, mid_x, bar_max_w_left, bar_max_w_right, tr)
            self._items_created = True
        else:
            # Update static geometry-dependent items
            self.coords(self._divider, mid_x - self._GAP_COL / 2, self._PAD_T,
                        mid_x - self._GAP_COL / 2, h - self._PAD_B)
            self.coords(self._bg_rect, 0, 0, w, h)

        # Update per-metric items
        for i, m in enumerate(self._metrics):
            col = 0 if i < 3 else 1
            row = i if i < 3 else i - 3
            y_mid = self._PAD_T + row * self._ROW_H + self._ROW_H / 2
            y_bar = y_mid - self._BAR_H / 2

            label_x = self._PAD_L if col == 0 else mid_x + self._GAP_COL / 2
            bar_x0 = label_x + self._LABEL_W
            bar_max_w = bar_max_w_left if col == 0 else bar_max_w_right
            bar_x1 = bar_x0 + bar_max_w

            # Update bar backgrounds and positions
            if m["bar_bg"] is not None:
                self.coords(m["bar_bg"], bar_x0, y_bar, bar_x1, y_bar + self._BAR_H)
                self.itemconfigure(m["bar_bg"], fill="#e8e8e8", outline="#ddd")

            # Bar fill — percentage-based metrics derive frac from current;
            # RAM/VRAM/temp have pre-computed frac from _apply_loop.
            if m["key"] in ("ram", "vram", "temp"):
                frac = m["frac"]
            else:
                frac = m["current"] / 100.0
            frac = max(0.0, min(1.0, frac))
            fill_w = frac * bar_max_w
            if m["bar_fill"] is not None:
                if fill_w > 0.5:
                    self.coords(m["bar_fill"], bar_x0, y_bar + 1,
                                bar_x0 + fill_w, y_bar + self._BAR_H - 1)
                    self.itemconfigure(m["bar_fill"], fill=self._bar_color(frac),
                                       outline="")
                else:
                    self.coords(m["bar_fill"], 0, 0, 0, 0)  # hide

            # Value text
            if m["key"] in ("gpu", "vram", "temp") and not self._gpu_available:
                val_str = tr("mon_no_gpu")
            else:
                val_str = m["fmt"].format(m["current"])
            if m["text"] is not None:
                self.coords(m["text"], bar_x1 + 8, y_mid)
                self.itemconfigure(m["text"], text=val_str)

    def _create_items(self, w: int, h: int, mid_x: float,
                      bar_max_w_left: float, bar_max_w_right: float, tr) -> None:
        """Create all canvas items once; _redraw() updates them in-place."""
        self._bg_rect = self.create_rectangle(0, 0, w, h, fill="#f5f5f5", outline="",
                                               tags=("bg",))
        self._divider = self.create_line(
            mid_x - self._GAP_COL / 2, self._PAD_T,
            mid_x - self._GAP_COL / 2, h - self._PAD_B,
            fill="#ddd", width=1, tags=("divider",),
        )

        for i, m in enumerate(self._metrics):
            col = 0 if i < 3 else 1
            row = i if i < 3 else i - 3
            y_mid = self._PAD_T + row * self._ROW_H + self._ROW_H / 2
            y_bar = y_mid - self._BAR_H / 2

            label_x = self._PAD_L if col == 0 else mid_x + self._GAP_COL / 2
            bar_x0 = label_x + self._LABEL_W
            bar_max_w = bar_max_w_left if col == 0 else bar_max_w_right
            bar_x1 = bar_x0 + bar_max_w

            tag = f"m{i}"
            # Label
            key_tr = f"mon_{m['key']}"
            m["label"] = self.create_text(label_x, y_mid, text=tr(key_tr),
                                          font=self._font(8), fill="#444",
                                          anchor="w", tags=(tag,))
            # Bar background
            m["bar_bg"] = self.create_rectangle(bar_x0, y_bar, bar_x1, y_bar + self._BAR_H,
                                                fill="#e8e8e8", outline="#ddd",
                                                tags=(tag,))
            # Bar fill (initially hidden)
            m["bar_fill"] = self.create_rectangle(0, 0, 0, 0,
                                                  fill="#27ae60", outline="",
                                                  tags=(tag,))
            # Value text
            m["text"] = self.create_text(bar_x1 + 8, y_mid, text="",
                                         font=self._font(9, bold=True),
                                         fill="#333", anchor="w", tags=(tag,))

    @staticmethod
    def _bar_color(frac: float) -> str:
        """Green → yellow → orange → red gradient."""
        frac = max(0.0, min(1.0, frac))
        stops = [(0.0, 0x27, 0xae, 0x60),
                 (0.5, 0xf0, 0xc0, 0x40),
                 (0.75, 0xe6, 0x7e, 0x22),
                 (1.0, 0xc0, 0x39, 0x2b)]
        for i in range(len(stops) - 1):
            t0, r0, g0, b0 = stops[i]
            t1, r1, g1, b1 = stops[i + 1]
            if t0 <= frac <= t1:
                progress = (frac - t0) / (t1 - t0) if t1 > t0 else 0.0
                r = int(r0 + (r1 - r0) * progress)
                g = int(g0 + (g1 - g0) * progress)
                b = int(b0 + (b1 - b0) * progress)
                return f"#{r:02x}{g:02x}{b:02x}"
        return "#c0392b"
