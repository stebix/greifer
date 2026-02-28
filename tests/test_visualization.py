"""Unit tests for the visualization queue drain logic and DOF lock panel.

The _Visualizer.__init__ requires a QApplication and pyqtgraph widgets,
but _drain_queue only touches _queue, _t_offset, and _bufs. We bypass
the Qt constructor via object.__new__ and set up just those fields.
"""

import queue as queue_module
from collections import deque

import pytest

from greifer.transform import Axis
from greifer.visualization import _Visualizer, DofLockPanel, MAXLEN


def _make_drain_target(q: queue_module.Queue) -> _Visualizer:
    """Create a _Visualizer with only the fields needed for _drain_queue."""
    viz = object.__new__(_Visualizer)
    viz._queue = q
    viz._t_offset = None
    viz._bufs = [deque(maxlen=MAXLEN) for _ in range(7)]
    return viz


def _put_sample(q, t, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
    """Helper to enqueue a 7-tuple sample."""
    q.put_nowait((t, x, y, z, roll, pitch, yaw))


# ── Empty queue ──────────────────────────────────────────────────────


class TestDrainEmptyQueue:

    def test_returns_true(self):
        q = queue_module.Queue()
        viz = _make_drain_target(q)
        assert viz._drain_queue() is True

    def test_buffers_remain_empty(self):
        q = queue_module.Queue()
        viz = _make_drain_target(q)
        viz._drain_queue()
        for buf in viz._bufs:
            assert len(buf) == 0


# ── Single sample ────────────────────────────────────────────────────


class TestSingleSample:

    def test_returns_true(self):
        q = queue_module.Queue()
        _put_sample(q, t=10.0, x=1.0, y=2.0, z=3.0)
        viz = _make_drain_target(q)
        assert viz._drain_queue() is True

    def test_populates_all_seven_buffers(self):
        q = queue_module.Queue()
        _put_sample(q, t=10.0, x=1.0, y=2.0, z=3.0, roll=4.0, pitch=5.0, yaw=6.0)
        viz = _make_drain_target(q)
        viz._drain_queue()
        for buf in viz._bufs:
            assert len(buf) == 1

    def test_time_offset_set_from_first_sample(self):
        q = queue_module.Queue()
        _put_sample(q, t=42.0)
        viz = _make_drain_target(q)
        viz._drain_queue()
        assert viz._t_offset == 42.0

    def test_time_stored_relative_to_offset(self):
        q = queue_module.Queue()
        _put_sample(q, t=42.0)
        viz = _make_drain_target(q)
        viz._drain_queue()
        # Relative time: 42.0 - 42.0 = 0.0
        assert viz._bufs[0][0] == pytest.approx(0.0)

    def test_channel_values_stored_as_is(self):
        q = queue_module.Queue()
        _put_sample(q, t=10.0, x=1.0, y=2.0, z=3.0, roll=4.0, pitch=5.0, yaw=6.0)
        viz = _make_drain_target(q)
        viz._drain_queue()
        assert viz._bufs[1][0] == pytest.approx(1.0)  # x
        assert viz._bufs[2][0] == pytest.approx(2.0)  # y
        assert viz._bufs[3][0] == pytest.approx(3.0)  # z
        assert viz._bufs[4][0] == pytest.approx(4.0)  # roll
        assert viz._bufs[5][0] == pytest.approx(5.0)  # pitch
        assert viz._bufs[6][0] == pytest.approx(6.0)  # yaw


# ── Shutdown sentinel ─────────────────────────────────────────────────


class TestShutdownSentinel:

    def test_none_returns_false(self):
        q = queue_module.Queue()
        q.put_nowait(None)
        viz = _make_drain_target(q)
        assert viz._drain_queue() is False

    def test_samples_before_sentinel_are_processed(self):
        q = queue_module.Queue()
        _put_sample(q, t=1.0, x=10.0)
        q.put_nowait(None)
        viz = _make_drain_target(q)
        result = viz._drain_queue()
        assert result is False
        # The sample before the sentinel should have been stored
        assert len(viz._bufs[0]) == 1
        assert viz._bufs[1][0] == pytest.approx(10.0)


# ── Negative timestamp filtering ─────────────────────────────────────


class TestNegativeTimestamp:

    def test_negative_t_skipped(self):
        q = queue_module.Queue()
        _put_sample(q, t=-1.0, x=99.0)
        viz = _make_drain_target(q)
        viz._drain_queue()
        # Sample should have been discarded
        for buf in viz._bufs:
            assert len(buf) == 0

    def test_negative_t_does_not_set_offset(self):
        q = queue_module.Queue()
        _put_sample(q, t=-1.0)
        viz = _make_drain_target(q)
        viz._drain_queue()
        assert viz._t_offset is None

    def test_negative_then_positive(self):
        q = queue_module.Queue()
        _put_sample(q, t=-1.0, x=1.0)   # skipped
        _put_sample(q, t=5.0, x=2.0)    # kept, sets offset
        viz = _make_drain_target(q)
        viz._drain_queue()
        assert len(viz._bufs[0]) == 1
        assert viz._t_offset == 5.0
        assert viz._bufs[0][0] == pytest.approx(0.0)
        assert viz._bufs[1][0] == pytest.approx(2.0)


# ── Time offset across multiple drains ────────────────────────────────


class TestTimeOffset:

    def test_offset_locked_to_first_valid_sample(self):
        q = queue_module.Queue()
        _put_sample(q, t=100.0)
        viz = _make_drain_target(q)
        viz._drain_queue()

        # Second drain with a later timestamp
        _put_sample(q, t=105.0)
        viz._drain_queue()

        # Offset should still be 100.0 (set from first sample)
        assert viz._t_offset == 100.0
        assert viz._bufs[0][0] == pytest.approx(0.0)    # 100 - 100
        assert viz._bufs[0][1] == pytest.approx(5.0)    # 105 - 100

    def test_multiple_samples_in_single_drain(self):
        q = queue_module.Queue()
        _put_sample(q, t=10.0, x=1.0)
        _put_sample(q, t=11.0, x=2.0)
        _put_sample(q, t=12.0, x=3.0)
        viz = _make_drain_target(q)
        viz._drain_queue()

        assert len(viz._bufs[0]) == 3
        assert list(viz._bufs[0]) == pytest.approx([0.0, 1.0, 2.0])
        assert list(viz._bufs[1]) == pytest.approx([1.0, 2.0, 3.0])


# ── Buffer capacity (MAXLEN) ─────────────────────────────────────────


class TestBufferCapacity:

    def test_buffers_bounded_by_maxlen(self):
        q = queue_module.Queue()
        for i in range(MAXLEN + 100):
            _put_sample(q, t=float(i), x=float(i))
        viz = _make_drain_target(q)
        viz._drain_queue()

        # Deques should have capped at MAXLEN
        assert len(viz._bufs[0]) == MAXLEN
        assert len(viz._bufs[1]) == MAXLEN
        # Oldest values should have been evicted
        assert viz._bufs[1][-1] == pytest.approx(float(MAXLEN + 99))


# ── DofLockPanel ─────────────────────────────────────────────────────


class TestDofLockPanel:

    def test_has_six_checkable_buttons(self, qapp):
        q = queue_module.Queue()
        panel = DofLockPanel(q)
        assert len(panel.buttons) == 6
        for btn in panel.buttons.values():
            assert btn.isCheckable()

    def test_clicking_button_enqueues_toggle_command(self, qapp):
        q = queue_module.Queue()
        panel = DofLockPanel(q)

        panel.buttons[Axis.X].toggle()

        cmd = q.get_nowait()
        assert cmd == ("toggle", Axis.X)

    def test_each_button_maps_to_correct_axis(self, qapp):
        q = queue_module.Queue()
        panel = DofLockPanel(q)

        for axis, btn in panel.buttons.items():
            btn.toggle()
            cmd = q.get_nowait()
            assert cmd == ("toggle", axis)
