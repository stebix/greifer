"""Real-time 6-DOF SpaceMouse visualization using pyqtgraph."""

import multiprocessing
import queue as queue_module
import signal

from collections import deque
from typing import NamedTuple

import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer


MAXLEN = 600  # ~10 s of visible history (producer decimates to ~VIS_HZ)
RENDER_HZ = 60
WINDOW_SECONDS = 10.0

type Sample = tuple[float, float, float, float, float, float, float]


class Channel(NamedTuple):
    name: str
    row: int
    col: int
    color: str


CHANNELS: tuple[Channel, ...] = (
    Channel("X", row=0, col=0, color="#1f77b4"),
    Channel("Y", row=1, col=0, color="#2ca02c"),
    Channel("Z", row=2, col=0, color="#17becf"),
    Channel("Roll", row=0, col=1, color="#d62728"),
    Channel("Pitch", row=1, col=1, color="#ff7f0e"),
    Channel("Yaw", row=2, col=1, color="#e377c2"),
)


class _Visualizer:
    """Encapsulates mutable plot state and the update loop."""

    def __init__(
        self,
        queue: multiprocessing.Queue,
        app: QApplication,
        win: pg.GraphicsLayoutWidget,
    ) -> None:
        self._queue = queue
        self._app = app

        self._t_offset: float | None = None
        self._bufs: list[deque[float]] = [deque(maxlen=MAXLEN) for _ in range(7)]
        self._curves: list[pg.PlotDataItem] = []

        first_plot: pg.PlotItem | None = None

        for ch in CHANNELS:
            p = win.addPlot(row=ch.row, col=ch.col, title=ch.name)
            p.setLabel("left", ch.name)
            p.setLabel("bottom", "Time", units="s")
            p.getAxis("bottom").enableAutoSIPrefix(False)
            p.setYRange(-1.0, 1.0)
            p.showGrid(x=True, y=True, alpha=0.3)

            if first_plot is None:
                first_plot = p
            else:
                p.setXLink(first_plot)

            self._curves.append(p.plot(pen=pg.mkPen(ch.color, width=2)))

        assert first_plot is not None
        self._first_plot: pg.PlotItem = first_plot

    def _drain_queue(self) -> bool:
        """Read all pending samples. Return False if shutdown sentinel received."""
        while True:
            try:
                sample: Sample | None = self._queue.get_nowait()
            except queue_module.Empty:
                return True
            if sample is None:
                return False
            if sample[0] < 0:
                continue  # skip uninitialized state (t defaults to -1.0)
            if self._t_offset is None:
                self._t_offset = sample[0]
            self._bufs[0].append(sample[0] - self._t_offset)
            for i in range(1, len(sample)):
                self._bufs[i].append(sample[i])
        return True

    def update(self) -> None:
        """Timer callback — drain queue then refresh curves."""
        if not self._drain_queue():
            self._app.quit()
            return

        if len(self._bufs[0]) == 0:
            return

        t_arr = np.array(self._bufs[0])

        for i in range(len(CHANNELS)):
            arr = np.array(self._bufs[i + 1])
            self._curves[i].setData(t_arr, arr)

        # Rolling x-axis window
        t_max = t_arr[-1]
        t_min = t_max - WINDOW_SECONDS
        self._first_plot.setXRange(t_min, t_max, padding=0)


def run_visualization(queue: multiprocessing.Queue) -> None:
    """Process entry point for the 6-DOF visualization window."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    app = QApplication([])

    win = pg.GraphicsLayoutWidget(title="SpaceMouse 6-DOF")
    win.resize(1200, 700)
    win.show()

    viz = _Visualizer(queue, app, win)

    timer = QTimer()
    timer.timeout.connect(viz.update)
    timer.start(1000 // RENDER_HZ)

    app.exec()
