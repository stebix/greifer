"""Real-time 6-DOF SpaceMouse visualization using pyqtgraph."""

from multiprocessing.queues import Queue
from queue import Empty, Full
import signal

from collections import deque
from typing import NamedTuple

import numpy as np
import pyqtgraph as pg

from collections.abc import Sequence

from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from greifer.transform import Axis, Sensitivity

type Command = tuple[str, Axis] | tuple[str, Sensitivity] | tuple[str, str]


MAXLEN = 600  # ~10 s of visible history (producer decimates to ~VIS_HZ)
RENDER_HZ = 60
WINDOW_SECONDS = 10.0


class Sample(NamedTuple):
    t: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


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


def _build_plot_grid(
    win: pg.GraphicsLayoutWidget,
) -> tuple[
    list[pg.PlotItem],
    list[pg.PlotDataItem],
    list[pg.TextItem],
    pg.PlotItem,
]:
    """Create the 6-panel plot grid.

    Returns (plots, curves, lock_labels, anchor_plot).
    """
    plots: list[pg.PlotItem] = []
    curves: list[pg.PlotDataItem] = []
    lock_labels: list[pg.TextItem] = []
    anchor_plot: pg.PlotItem | None = None

    bold_font = QFont()
    bold_font.setBold(True)

    for ch in CHANNELS:
        p = win.addPlot(row=ch.row, col=ch.col, title=ch.name)
        p.setLabel("left", ch.name)
        p.setLabel("bottom", "Time", units="s")
        p.getAxis("bottom").enableAutoSIPrefix(False)
        p.setYRange(-1.0, 1.0)
        p.showGrid(x=True, y=True, alpha=0.3)

        if anchor_plot is None:
            anchor_plot = p
        else:
            p.setXLink(anchor_plot)

        curves.append(p.plot(pen=pg.mkPen(ch.color, width=2)))
        plots.append(p)

        label = pg.TextItem("Locked", color="#ccc", anchor=(0, 0))
        label.setFont(bold_font)
        label.hide()
        p.addItem(label, ignoreBounds=True)
        lock_labels.append(label)

    if anchor_plot is None:
        raise ValueError("CHANNELS must not be empty")
    return plots, curves, lock_labels, anchor_plot


class _Visualizer:
    """Encapsulates mutable plot state and the update loop."""

    def __init__(
        self,
        queue: Queue[Sample | None],
        app: QApplication,
        *,
        plots: list[pg.PlotItem],
        curves: list[pg.PlotDataItem],
        lock_labels: list[pg.TextItem],
        anchor_plot: pg.PlotItem,
    ) -> None:
        self._queue = queue
        self._app = app
        self._plots = plots
        self._curves = curves
        self._lock_labels = lock_labels
        self._first_plot = anchor_plot

        self._t_offset: float | None = None
        self._bufs: list[deque[float]] = [
            deque(maxlen=MAXLEN) for _ in Sample._fields
        ]

    def _drain_queue(self) -> bool:
        """Read all pending samples. Return False if shutdown sentinel received."""
        while True:
            try:
                sample: Sample | None = self._queue.get_nowait()
            except Empty:
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

        for i, _ch in enumerate(CHANNELS):
            arr = np.array(self._bufs[i + 1])
            self._curves[i].setData(t_arr, arr)

        # Rolling x-axis window
        t_max = t_arr[-1]
        t_min = t_max - WINDOW_SECONDS
        self._first_plot.setXRange(t_min, t_max, padding=0)

        for label in self._lock_labels:
            if label.isVisible():
                label.setPos(t_min, 0.85)

    def set_axis_locked(self, index: int, locked: bool) -> None:
        """Toggle the visual lock indicator for *index*."""
        ch = CHANNELS[index]
        if locked:
            self._lock_labels[index].show()
            self._curves[index].setPen(
                pg.mkPen(ch.color, width=2, style=Qt.PenStyle.DotLine)
            )
        else:
            self._lock_labels[index].hide()
            self._curves[index].setPen(pg.mkPen(ch.color, width=2))


# Maps Channel names to Axis members for the lock panel buttons.
_CHANNEL_AXIS: tuple[tuple[str, str, Axis], ...] = (
    ("Translation", "X", Axis.X),
    ("Translation", "Y", Axis.Y),
    ("Translation", "Z", Axis.Z),
    ("Rotation", "Roll", Axis.ROLL),
    ("Rotation", "Pitch", Axis.PITCH),
    ("Rotation", "Yaw", Axis.YAW),
)


_TRANS_AXES = frozenset({Axis.X, Axis.Y, Axis.Z})


class TargetPanel(QWidget):
    """Side panel with a dropdown for selecting the active target structure."""

    def __init__(
        self,
        cmd_queue: Queue[Command],
        target_names: Sequence[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cmd_queue = cmd_queue

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Target")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        self._combo = QComboBox()
        self._combo.addItems(list(target_names))
        self._combo.currentTextChanged.connect(self._on_selection_changed)
        layout.addWidget(self._combo)

    def _on_selection_changed(self, name: str) -> None:
        try:
            self._cmd_queue.put_nowait(("switch_target", name))
        except (Full, BrokenPipeError, OSError):
            pass


class DofLockPanel(QWidget):
    """Side panel with toggle buttons for locking individual DOF axes."""

    def __init__(
        self,
        cmd_queue: Queue[Command],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cmd_queue = cmd_queue

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
        except (Full, BrokenPipeError, OSError):
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


class SensitivityPanel(QWidget):
    """Side panel with sliders for tuning per-axis sensitivity."""

    _SLIDER_STEPS = 1000
    _TRANS_MIN = 0.0
    _TRANS_MAX = 0.05
    _ROT_MIN = 0.0
    _ROT_MAX = 0.01

    def __init__(
        self,
        cmd_queue: Queue[Command],
        initial: Sensitivity,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cmd_queue = cmd_queue
        self._sensitivity = initial
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Sensitivity")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # ── Uniform tab ──────────────────────────────────────────
        uniform_widget = QWidget()
        uniform_layout = QGridLayout(uniform_widget)
        uniform_layout.setContentsMargins(4, 4, 4, 4)

        self._uniform_sliders: dict[str, QSlider] = {}
        self._uniform_spins: dict[str, QDoubleSpinBox] = {}

        s, sp = self._make_row(
            uniform_layout, 0, "Trans",
            self._TRANS_MIN, self._TRANS_MAX, initial.x,
        )
        self._uniform_sliders["trans"] = s
        self._uniform_spins["trans"] = sp

        s, sp = self._make_row(
            uniform_layout, 1, "Rot",
            self._ROT_MIN, self._ROT_MAX, initial.roll,
        )
        self._uniform_sliders["rot"] = s
        self._uniform_spins["rot"] = sp

        self._tabs.addTab(uniform_widget, "Uniform")

        # ── Per-Axis tab ─────────────────────────────────────────
        per_axis_widget = QWidget()
        per_axis_layout = QGridLayout(per_axis_widget)
        per_axis_layout.setContentsMargins(4, 4, 4, 4)

        self._axis_sliders: dict[Axis, QSlider] = {}
        self._axis_spins: dict[Axis, QDoubleSpinBox] = {}

        axis_info: tuple[tuple[Axis, str, str], ...] = (
            (Axis.X, "X", "#1f77b4"),
            (Axis.Y, "Y", "#2ca02c"),
            (Axis.Z, "Z", "#17becf"),
            (Axis.ROLL, "Roll", "#d62728"),
            (Axis.PITCH, "Pitch", "#ff7f0e"),
            (Axis.YAW, "Yaw", "#e377c2"),
        )

        for row, (axis, name, color) in enumerate(axis_info):
            vmin = self._TRANS_MIN if axis in _TRANS_AXES else self._ROT_MIN
            vmax = self._TRANS_MAX if axis in _TRANS_AXES else self._ROT_MAX
            s, sp = self._make_row(
                per_axis_layout, row, name,
                vmin, vmax, initial.for_axis(axis),
                color=color,
            )
            self._axis_sliders[axis] = s
            self._axis_spins[axis] = sp

        self._tabs.addTab(per_axis_widget, "Per-Axis")

        self._tabs.currentChanged.connect(self._on_tab_changed)

        layout.addStretch()

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _float_to_slider(value: float, vmin: float, vmax: float) -> int:
        if vmax <= vmin:
            return 0
        return round((value - vmin) / (vmax - vmin) * SensitivityPanel._SLIDER_STEPS)

    @staticmethod
    def _slider_to_float(pos: int, vmin: float, vmax: float) -> float:
        return vmin + pos / SensitivityPanel._SLIDER_STEPS * (vmax - vmin)

    def _make_row(
        self,
        layout: QGridLayout,
        row: int,
        label: str,
        vmin: float,
        vmax: float,
        initial: float,
        color: str | None = None,
    ) -> tuple[QSlider, QDoubleSpinBox]:
        lbl = QLabel(label)
        if color is not None:
            lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
        layout.addWidget(lbl, row, 0)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, self._SLIDER_STEPS)
        slider.setValue(self._float_to_slider(initial, vmin, vmax))
        layout.addWidget(slider, row, 1)

        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(vmin, vmax)
        spin.setSingleStep((vmax - vmin) / self._SLIDER_STEPS)
        spin.setValue(initial)
        layout.addWidget(spin, row, 2)

        slider.valueChanged.connect(
            lambda _val, s=slider, sp=spin, mn=vmin, mx=vmax: self._slider_moved(s, sp, mn, mx)
        )
        spin.valueChanged.connect(
            lambda _val, s=slider, sp=spin, mn=vmin, mx=vmax: self._spin_changed(s, sp, mn, mx)
        )

        return slider, spin

    def _slider_moved(
        self, slider: QSlider, spin: QDoubleSpinBox, vmin: float, vmax: float,
    ) -> None:
        if self._updating:
            return
        self._updating = True
        spin.setValue(self._slider_to_float(slider.value(), vmin, vmax))
        self._updating = False
        self._emit_sensitivity()

    def _spin_changed(
        self, slider: QSlider, spin: QDoubleSpinBox, vmin: float, vmax: float,
    ) -> None:
        if self._updating:
            return
        self._updating = True
        slider.setValue(self._float_to_slider(spin.value(), vmin, vmax))
        self._updating = False
        self._emit_sensitivity()

    def _emit_sensitivity(self) -> None:
        if self._tabs.currentIndex() == 0:
            # Uniform tab
            trans = self._uniform_spins["trans"].value()
            rot = self._uniform_spins["rot"].value()
            sens = Sensitivity.uniform(trans, rot)
        else:
            # Per-Axis tab
            sens = Sensitivity(
                x=self._axis_spins[Axis.X].value(),
                y=self._axis_spins[Axis.Y].value(),
                z=self._axis_spins[Axis.Z].value(),
                roll=self._axis_spins[Axis.ROLL].value(),
                pitch=self._axis_spins[Axis.PITCH].value(),
                yaw=self._axis_spins[Axis.YAW].value(),
            )
        self._sensitivity = sens
        try:
            self._cmd_queue.put_nowait(("sensitivity", sens))
        except (Full, BrokenPipeError, OSError):
            pass

    def _on_tab_changed(self, index: int) -> None:
        self._updating = True
        if index == 0:
            # Switching to Uniform — average translation and rotation axes
            trans_avg = (
                self._sensitivity.x
                + self._sensitivity.y
                + self._sensitivity.z
            ) / 3.0
            rot_avg = (
                self._sensitivity.roll
                + self._sensitivity.pitch
                + self._sensitivity.yaw
            ) / 3.0
            self._uniform_spins["trans"].setValue(trans_avg)
            self._uniform_sliders["trans"].setValue(
                self._float_to_slider(trans_avg, self._TRANS_MIN, self._TRANS_MAX)
            )
            self._uniform_spins["rot"].setValue(rot_avg)
            self._uniform_sliders["rot"].setValue(
                self._float_to_slider(rot_avg, self._ROT_MIN, self._ROT_MAX)
            )
        else:
            # Switching to Per-Axis — populate from current sensitivity
            for axis, spin in self._axis_spins.items():
                val = self._sensitivity.for_axis(axis)
                spin.setValue(val)
                vmin = self._TRANS_MIN if axis in _TRANS_AXES else self._ROT_MIN
                vmax = self._TRANS_MAX if axis in _TRANS_AXES else self._ROT_MAX
                self._axis_sliders[axis].setValue(
                    self._float_to_slider(val, vmin, vmax)
                )
        self._updating = False
        self._emit_sensitivity()


def _build_right_panel(
    cmd_queue: Queue[Command],
    sensitivity: Sensitivity,
    target_names: Sequence[str],
) -> tuple[QWidget, DofLockPanel]:
    """Assemble the target selector + lock + sensitivity sidebar."""
    panel = QWidget()
    panel.setFixedWidth(220)
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    layout.addWidget(TargetPanel(cmd_queue, target_names))
    lock_panel = DofLockPanel(cmd_queue)
    layout.addWidget(lock_panel)
    layout.addWidget(SensitivityPanel(cmd_queue, sensitivity))
    layout.addStretch()

    return panel, lock_panel


def run_visualization(
    data_queue: Queue[Sample | None],
    cmd_queue: Queue[Command],
    initial_sensitivity: Sensitivity | None = None,
    target_names: Sequence[str] = ("SpaceMouseTransform",),
) -> None:
    """Process entry point for the 6-DOF visualization window."""
    if initial_sensitivity is None:
        initial_sensitivity = Sensitivity.uniform(0.005, 0.001)

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

    right_panel, lock_panel = _build_right_panel(
        cmd_queue, initial_sensitivity, target_names,
    )
    h_layout.addWidget(right_panel, stretch=0)

    main_window.show()

    plots, curves, lock_labels, anchor = _build_plot_grid(graph_widget)
    viz = _Visualizer(
        data_queue, app,
        plots=plots,
        curves=curves,
        lock_labels=lock_labels,
        anchor_plot=anchor,
    )

    for axis, btn in lock_panel.buttons.items():
        idx = axis.index
        btn.toggled.connect(
            lambda checked, i=idx: viz.set_axis_locked(i, checked)
        )

    timer = QTimer()
    timer.timeout.connect(viz.update)
    timer.start(1000 // RENDER_HZ)

    app.exec()
