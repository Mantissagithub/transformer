from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── strict grayscale palette ──────────────────────────────────────────────────
_BG     = "#0e0e0e"
_BORDER = "#3c3c3c"
_DIM    = "#555555"
_MID    = "#888888"
_BRIGHT = "#c8c8c8"
_WHITE  = "#eeeeee"

# ── tunables ──────────────────────────────────────────────────────────────────
_RENDER_FPS  = 3
_MAX_HISTORY = 2000
_MAX_EVENTS  = 60
_LOG_EVERY   = 50

# ── braille dot bitmask layout (Unicode standard, dots 1-8) ──────────────────
#   Each braille char covers a 2 (cols) × 4 (rows) dot grid.
#   Unicode codepoint = 0x2800 | OR of set dot bits.
#
#   col→   0     1
#   row 0: 0x01  0x08
#   row 1: 0x02  0x10
#   row 2: 0x04  0x20
#   row 3: 0x40  0x80
_BRAILLE_DOTS = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]


# ── helpers ───────────────────────────────────────────────────────────────────

class _LogRedirect:
    """Proxy: sys.stdout writes go to log file; rich renders to original fd."""
    def __init__(self, f: Any) -> None:
        self._f = f
    def write(self, s: str) -> int:
        return self._f.write(s)
    def flush(self) -> None:
        self._f.flush()
    def isatty(self) -> bool:
        return False
    def __getattr__(self, name: str):
        return getattr(self._f, name)


def _fmt_dur(secs: float) -> str:
    s = int(secs)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _pbar(cur: int, tot: int, w: int = 18) -> str:
    if tot <= 0:
        return f"[{'?' * w}]"
    p = min(1.0, cur / tot)
    n = int(p * w)
    return f"[{'#' * n}{'-' * (w - n)}] {p * 100:5.1f}%"


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> Iterator[Tuple[int, int]]:
    """Yield integer (x, y) pixels along the line from (x0,y0) to (x1,y1)."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0  += sx
        if e2 < dx:
            err += dx
            y0  += sy


def _plot_braille(
    history: List[Tuple[int, float]],
    pw: int,    # braille character columns (plot area only, excl. y-axis labels)
    ph: int,    # braille character rows
) -> Text:
    """Render a line chart as braille Unicode text — always pixel-perfect sharp.

    Effective resolution: pw*2 horizontal × ph*4 vertical dots.
    Returns a rich Text block of exactly ph lines × pw chars each.
    """
    bw = pw * 2   # effective pixel columns
    bh = ph * 4   # effective pixel rows

    grid = [[0] * pw for _ in range(ph)]

    if len(history) >= 2:
        steps, losses = zip(*history)
        min_l, max_l = min(losses), max(losses)
        min_s, max_s = min(steps),  max(steps)
        if max_l <= min_l:
            max_l = min_l + 1.0
        if max_s <= min_s:
            max_s = min_s + 1

        # subsample to at most bw points so Bresenham stays O(pw) not O(history)
        pts = list(history)
        if len(pts) > bw:
            idx = np.round(np.linspace(0, len(pts) - 1, bw)).astype(int)
            pts = [pts[i] for i in idx]

        def to_dot(step: float, loss: float) -> Tuple[int, int]:
            x = int((step - min_s) / (max_s - min_s) * (bw - 1))
            y = int((max_l - loss) / (max_l - min_l) * (bh - 1))  # y-inverted
            return max(0, min(bw - 1, x)), max(0, min(bh - 1, y))

        for i in range(len(pts) - 1):
            x0, y0 = to_dot(*pts[i])
            x1, y1 = to_dot(*pts[i + 1])
            for px, py in _bresenham(x0, y0, x1, y1):
                cr, cc = py // 4, px // 2
                dr, dc = py % 4,  px % 2
                grid[cr][cc] |= _BRAILLE_DOTS[dr][dc]

    text = Text(no_wrap=True, overflow="crop")
    for r in range(ph):
        for c in range(pw):
            mask = grid[r][c]
            ch   = chr(0x2800 + mask)
            text.append(ch, style=_WHITE if mask else _BG)
        text.append("\n")
    return text


def _chart_with_axes(
    history: List[Tuple[int, float]],
    pw: int,   # total chars available (including y-axis label margin)
    ph: int,   # total char rows available (including x-axis label row)
) -> Text:
    """Wrap `_plot_braille` with y-axis tick labels and an x-axis step label row."""
    Y_LABEL_W = 7   # "  9.23 " — 7 chars for y-axis margin
    X_LABEL_H = 1   # one row for x-axis labels
    bw_chars   = max(4, pw - Y_LABEL_W)
    bh_chars   = max(4, ph - X_LABEL_H)

    braille = _plot_braille(history, bw_chars, bh_chars)

    # build y-axis tick values at 5 evenly-spaced rows
    if len(history) >= 2:
        losses   = [v for _, v in history]
        min_l, max_l = min(losses), max(losses)
        if max_l <= min_l:
            max_l = min_l + 1.0
    else:
        min_l, max_l = 0.0, 1.0

    tick_rows = set()
    n_ticks   = 5
    for i in range(n_ticks):
        row = round(i * (bh_chars - 1) / max(1, n_ticks - 1))
        tick_rows.add(row)

    # combine y-axis labels + braille lines
    braille_lines = braille.plain.split("\n")[:-1]  # strip trailing empty
    out = Text(no_wrap=True, overflow="crop")
    for row_idx, line in enumerate(braille_lines):
        if row_idx in tick_rows:
            frac  = 1.0 - row_idx / max(1, bh_chars - 1)
            label = f"{min_l + frac * (max_l - min_l):6.2f} "
        else:
            label = " " * Y_LABEL_W
        out.append(label, style=_DIM)
        # re-render this row's braille with correct styles
        for ch in line:
            mask = ord(ch) - 0x2800
            out.append(ch, style=_WHITE if mask else _BG)
        out.append("\n")

    # x-axis: step range labels
    if history:
        min_s = history[0][0]
        max_s = history[-1][0]
        mid_s = (min_s + max_s) // 2
        left  = str(min_s)
        mid   = str(mid_s)
        right = str(max_s)
        # place them at left edge, center, and right edge of the braille area
        axis_row = " " * Y_LABEL_W
        inner    = bw_chars
        mid_pos  = inner // 2 - len(mid) // 2
        axis_row += left
        pad_to_mid = mid_pos - len(left)
        axis_row += " " * max(1, pad_to_mid)
        axis_row += mid
        pad_to_right = inner - mid_pos - len(mid) - len(right)
        axis_row += " " * max(1, pad_to_right)
        axis_row += right
    else:
        axis_row = " " * pw

    out.append(axis_row[:pw], style=_DIM)
    return out


def _gpu_mem_str(device: torch.device) -> str:
    try:
        if device.type != "cuda":
            return ""
        idx  = device.index or 0
        used  = torch.cuda.memory_allocated(idx) / 1024 ** 3
        total = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
        return f"{used:.1f}/{total:.0f} GB"
    except Exception:
        return ""


def _ema(history: List[Tuple[int, float]], alpha: float = 0.97) -> float:
    if not history:
        return float("nan")
    val = history[0][1]
    for _, v in history:
        val = alpha * val + (1 - alpha) * v
    return val


# ── main class ────────────────────────────────────────────────────────────────

class TrainingTUI:
    def __init__(
        self,
        *,
        experiment_name: str,
        model_summary: Dict[str, Any],
        total_steps: int,
        steps_per_epoch: int,
        num_epochs: int,
        device: torch.device,
        log_file: str | Path = "training.log",
    ) -> None:
        self.experiment_name = experiment_name
        self.model_summary   = model_summary
        self.total_steps     = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.num_epochs      = num_epochs
        self.device          = device

        # mutable state — written by training thread, read by render thread
        self._lock            = threading.Lock()
        self._epoch: int      = 0
        self._step_in_epoch: int  = 0
        self._global_step: int    = 0
        self._loss: float         = float("nan")
        self._lr: float           = 0.0
        self._grad_norm: float    = 0.0
        self._toks_per_sec: float = 0.0
        self._history: Deque[Tuple[int, float]] = deque(maxlen=_MAX_HISTORY)
        self._events:  Deque[str] = deque(maxlen=_MAX_EVENTS)
        self._start: float    = time.monotonic()

        # rich — console writes to original stdout (preserved before redirect)
        self._orig_stdout = sys.stdout
        self._console = Console(
            file=self._orig_stdout,
            highlight=False,
            markup=False,
            color_system="truecolor",
        )
        self._live = Live(
            console=self._console,
            auto_refresh=False,
            screen=True,
        )
        self._layout = self._make_skeleton()

        # log file receives all print() output while TUI is active
        lp = Path(log_file)
        lp.parent.mkdir(parents=True, exist_ok=True)
        self._logf = open(lp, "w", buffering=1)

        # background render thread
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._render_loop, daemon=True)

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def __enter__(self) -> "TrainingTUI":
        self._live.__enter__()
        sys.stdout = _LogRedirect(self._logf)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        sys.stdout = self._orig_stdout
        self._live.__exit__(exc_type, exc, tb)
        self._logf.close()

    # ── training-thread API (non-blocking) ────────────────────────────────────

    def update_step(
        self,
        *,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        loss: float,
        lr: float,
        grad_norm: float = 0.0,
        tokens_per_sec: float = 0.0,
    ) -> None:
        with self._lock:
            self._epoch         = epoch
            self._step_in_epoch = step_in_epoch
            self._global_step   = global_step
            self._loss          = loss
            self._lr            = lr
            self._grad_norm     = grad_norm
            self._toks_per_sec  = tokens_per_sec
            self._history.append((global_step, loss))
        if global_step % _LOG_EVERY == 0:
            self.event(
                f"step {global_step:>6d}  loss={loss:.4f}  lr={lr:.2e}"
                f"  gnorm={grad_norm:.3f}  tok/s={tokens_per_sec:,.0f}"
            )

    def reset_epoch(self, epoch: int) -> None:
        with self._lock:
            self._epoch         = epoch
            self._step_in_epoch = 0

    def event(self, msg: str) -> None:
        ts   = time.strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        with self._lock:
            self._events.append(line)
        # logf's closed after __exit__ and the live display is gone, so just
        # dump late events (like the hf push result) to the real terminal
        # instead of blowing up on a closed file.
        if self._logf.closed:
            print(line, file=self._orig_stdout, flush=True)
            return
        self._logf.write(line + "\n")
        self._logf.flush()

    # ── render thread ──────────────────────────────────────────────────────────

    def _render_loop(self) -> None:
        interval = 1.0 / _RENDER_FPS
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._live.update(self._build_frame(), refresh=True)
            except Exception:
                pass
            dt = time.monotonic() - t0
            self._stop.wait(max(0.0, interval - dt))

    # ── layout skeleton ───────────────────────────────────────────────────────

    @staticmethod
    def _make_skeleton() -> Layout:
        root = Layout()
        root.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        root["body"].split_row(
            Layout(name="plot",    ratio=40),
            Layout(name="sidebar", ratio=22),
            Layout(name="log",     ratio=38),
        )
        return root

    # ── frame assembly ────────────────────────────────────────────────────────

    def _build_frame(self) -> Layout:
        with self._lock:
            epoch         = self._epoch
            step_in_epoch = self._step_in_epoch
            global_step   = self._global_step
            loss          = self._loss
            lr            = self._lr
            grad_norm     = self._grad_norm
            toks          = self._toks_per_sec
            history       = list(self._history)
            events        = list(self._events)
            start         = self._start

        elapsed = time.monotonic() - start
        eta: Optional[float] = None
        if global_step > 0:
            eta = elapsed / global_step * max(0, self.total_steps - global_step)

        self._layout["header" ].update(self._panel_header())
        self._layout["plot"   ].update(self._panel_plot(history))
        self._layout["sidebar"].update(self._panel_sidebar(
            epoch, step_in_epoch, global_step, loss, lr, grad_norm, toks, history,
        ))
        self._layout["log"    ].update(self._panel_log(events))
        self._layout["footer" ].update(self._panel_footer(elapsed, eta))
        return self._layout

    # ── panel builders ────────────────────────────────────────────────────────

    def _panel_header(self) -> Panel:
        m = self.model_summary
        t = Text(justify="center", no_wrap=True)
        t.append(self.experiment_name, style=f"bold {_WHITE}")
        t.append(
            f"   {m.get('n_params', 0) / 1e6:.2f}M params"
            f"  {m.get('n_layers', '?')}L"
            f"  d={m.get('d_model', '?')}"
            f"  attn={m.get('attention', '?')}"
            f"  ffn={m.get('feedforward', '?')}"
            f"  dev={self.device}",
            style=_MID,
        )
        return Panel(t, border_style=_BORDER, padding=(0, 1))

    def _panel_footer(self, elapsed: float, eta: Optional[float]) -> Panel:
        eta_s = _fmt_dur(eta) if eta is not None else "--:--:--"
        t = Text(justify="center", no_wrap=True)
        t.append(
            f"elapsed {_fmt_dur(elapsed)}   eta {eta_s}"
            f"   run: {self.experiment_name}   dev: {self.device}",
            style=_MID,
        )
        return Panel(t, border_style=_BORDER, padding=(0, 1))

    def _panel_plot(self, history: List[Tuple[int, float]]) -> Panel:
        cw = self._console.width
        ch = self._console.height
        # total chars available inside the panel (approx 40% minus borders)
        pw = max(16, int(cw * 0.40) - 4)
        ph = max(8,  ch - 12)
        chart = _chart_with_axes(history, pw, ph)
        return Panel(chart, title="loss", border_style=_BORDER, padding=(0, 1))

    def _panel_sidebar(
        self,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        toks: float,
        history: List[Tuple[int, float]],
    ) -> Panel:
        tbl = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        tbl.add_column("k", style=_MID,   no_wrap=True)
        tbl.add_column("v", style=_WHITE, no_wrap=True)

        def row(k: str, v: str) -> None:
            tbl.add_row(k, v)

        def sep() -> None:
            tbl.add_row("", "")

        attn   = self.model_summary.get("attention", "?")
        loss_s = f"{loss:.4f}" if not math.isnan(loss) else "---"
        ema_v  = _ema(history)
        ema_s  = f"{ema_v:.4f}" if not math.isnan(ema_v) else "---"
        gpu_s  = _gpu_mem_str(self.device)

        row("variant", attn)
        sep()
        row("step",    f"{global_step:,} / {self.total_steps:,}")
        row("epoch",   f"{epoch + 1} / {self.num_epochs}")
        sep()
        row("loss",    loss_s)
        row("ema",     ema_s)
        row("lr",      f"{lr:.2e}")
        row("grad",    f"{grad_norm:.4f}")
        row("tok/s",   f"{toks:,.0f}" if toks > 0 else "---")
        if gpu_s:
            sep()
            row("gpu mem", gpu_s)
        sep()
        row("ep",      _pbar(step_in_epoch, self.steps_per_epoch))
        row("total",   _pbar(global_step, self.total_steps))

        return Panel(tbl, title="metrics", border_style=_BORDER, padding=(1, 1))

    def _panel_log(self, events: List[str]) -> Panel:
        ch           = self._console.height
        visible_rows = max(1, ch - 10)
        recent       = events[-visible_rows:]

        t = Text(no_wrap=True, overflow="crop")
        for line in recent:
            if len(line) >= 8 and line[2] == ":" and line[5] == ":":
                t.append(line[:8], style=_DIM)
                t.append(line[8:] + "\n", style=_MID)
            else:
                t.append(line + "\n", style=_MID)

        return Panel(t, title="log", border_style=_BORDER, padding=(0, 1))
