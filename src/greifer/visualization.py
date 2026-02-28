"""Real-time 6-DOF SpaceMouse visualization using pyqtgraph."""

import multiprocessing
import queue as queue_module
import signal

from collections import deque
from typing import NamedTuple

import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QTimer

from greifer.transform import Axis


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


# Maps Channel names to Axis members for the lock panel buttons.
_CHANNEL_AXIS: tuple[tuple[str, str, Axis], ...] = (
    ("Translation", "X", Axis.X),
    ("Translation", "Y", Axis.Y),
    ("Translation", "Z", Axis.Z),
    ("Rotation", "Roll", Axis.ROLL),
    ("Rotation", "Pitch", Axis.PITCH),
    ("Rotation", "Yaw", Axis.YAW),
)


class DofLockPanel(QWidget):
    """Side panel with toggle buttons for locking individual DOF axes."""

    def __init__(
        self,
        cmd_queue: multiprocessing.Queue,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cmd_queue = cmd_queue
        self.setFixedWidth(130)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("DOF Locks")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        self.buttons: dict[Axis, QPushButton] = {}
        current_group: str | None = None

        for group, name, axis in _CHANNEL_AXIS:
            if group != current_group:
                if current_group is not None:
                    layout.addSpacing(12)
                group_label = QLabel(group)
                group_label.setStyleSheet("font-size: 11px; color: #aaa;")
                layout.addWidget(group_label)
                current_group = group

            color = next(ch.color for ch in CHANNELS if ch.name == name)
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet(self._button_style(color))
            btn.toggled.connect(lambda _checked, a=axis: self._on_toggle(a))
            layout.addWidget(btn)
            self.buttons[axis] = btn

        layout.addStretch()

    def _on_toggle(self, axis: Axis) -> None:
        try:
            self._cmd_queue.put_nowait(("toggle", axis))
        except (queue_module.Full, BrokenPipeError, OSError):
            pass

    @staticmethod
    def _button_style(color: str) -> str:
        return (
            f"QPushButton {{"
            f"  border: 2px solid {color};"
            f"  border-radius: 4px;"
            f"  padding: 4px;"
            f"  background: #2b2b2b;"
            f"  color: {color};"
            f"}}"
            f"QPushButton:checked {{"
            f"  background: {color};"
            f"  color: #1a1a1a;"
            f"}}"
        )


def run_visualization(
    data_queue: multiprocessing.Queue,
    cmd_queue: multiprocessing.Queue,
) -> None:
    """Process entry point for the 6-DOF visualization window."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    app = QApplication([])

    main_window = QWidget()
    main_window.setWindowTitle("SpaceMouse 6-DOF")
    main_window.resize(1350, 700)

    h_layout = QHBoxLayout(main_window)
    h_layout.setContentsMargins(0, 0, 0, 0)
    h_layout.setSpacing(0)

    graph_widget = pg.GraphicsLayoutWidget()
    h_layout.addWidget(graph_widget, stretch=1)

    panel = DofLockPanel(cmd_queue)
    h_layout.addWidget(panel, stretch=0)

    main_window.show()

    viz = _Visualizer(data_queue, app, graph_widget)

    timer = QTimer()
    timer.timeout.connect(viz.update)
    timer.start(1000 // RENDER_HZ)

    app.exec()
