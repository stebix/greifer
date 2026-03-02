"""Integration tests for the stream loop wiring.

These tests verify that _stream_loop correctly connects device reading,
transform computation, network sending, and visualization queuing — using
fakes at the hardware/network boundaries.
"""

import logging
import queue as queue_module

import numpy as np
from numpy.testing import assert_allclose

import pyigtl
import pytest

from greifer.app import _stream_loop
from greifer.target import TargetManager
from conftest import FakeDevice, FakeState, FakeClient, StopStreaming


# Use dt=0 in all tests to avoid real sleeps
DT = 0.0


def _run_loop(device, client, vis_queue=None):
    """Run _stream_loop until the FakeDevice is exhausted."""
    with pytest.raises(StopStreaming):
        _stream_loop(device, client, vis_queue, dt=DT)


# ── Transform sending ────────────────────────────────────────────────


class TestTransformSending:

    def test_nonzero_motion_sends_transform(self):
        """Device states with motion above threshold produce messages."""
        states = [FakeState(x=100.0, t=0.0)] * 5
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client)

        assert len(client.messages) == 5

    def test_zero_motion_sends_nothing(self):
        """All-zero device states produce no messages (below threshold)."""
        states = [FakeState()] * 10
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client)

        assert len(client.messages) == 0

    def test_mixed_motion_and_stillness(self):
        """Only states with sufficient motion produce messages."""
        states = [
            FakeState(x=100.0, t=0.0),   # motion → message
            FakeState(t=0.1),             # zero → no message
            FakeState(y=200.0, t=0.2),    # motion → message
            FakeState(t=0.3),             # zero → no message
        ]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client)

        assert len(client.messages) == 2

    def test_sent_matrix_is_4x4(self):
        """Each sent message wraps a 4x4 matrix."""
        device = FakeDevice([FakeState(x=100.0)])
        client = FakeClient()

        _run_loop(device, client)

        msg = client.messages[0]
        assert msg.matrix.shape == (4, 4)

    def test_single_x_translation_matrix(self):
        """A single x-axis device input produces the expected translation."""
        from greifer.app import SENSITIVITY
        device = FakeDevice([FakeState(x=100.0)])
        client = FakeClient()

        _run_loop(device, client)

        matrix = client.messages[0].matrix
        expected_dx = 100.0 * SENSITIVITY.x
        assert_allclose(matrix[0, 3], expected_dx, atol=1e-10)
        assert_allclose(matrix[1, 3], 0.0, atol=1e-10)
        assert_allclose(matrix[2, 3], 0.0, atol=1e-10)


# ── Transform accumulation across frames ──────────────────────────────


class TestAccumulationAcrossFrames:

    def test_translations_accumulate(self):
        """Two consecutive x-translations should sum."""
        from greifer.app import SENSITIVITY
        states = [
            FakeState(x=100.0, t=0.0),
            FakeState(x=100.0, t=0.1),
        ]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client)

        # Second message should have accumulated translation
        final_matrix = client.messages[-1].matrix
        expected_dx = 2 * 100.0 * SENSITIVITY.x
        assert_allclose(final_matrix[0, 3], expected_dx, atol=1e-10)

    def test_rotation_submatrix_stays_orthonormal(self):
        """After many frames, the rotation part should remain valid."""
        states = [
            FakeState(roll=50.0, pitch=30.0, yaw=20.0, t=i * 0.01)
            for i in range(200)
        ]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client)

        R = client.messages[-1].matrix[:3, :3]
        assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-8)


# ── Visualization queue ──────────────────────────────────────────────


class TestVisualizationQueue:

    def test_samples_enqueued_when_vis_queue_provided(self):
        """All device reads should produce visualization samples."""
        states = [
            FakeState(x=100.0, t=1.0),
            FakeState(y=200.0, t=2.0),
        ]
        device = FakeDevice(states)
        client = FakeClient()
        vis_queue = queue_module.Queue(maxsize=100)

        _run_loop(device, client, vis_queue=vis_queue)

        samples = []
        while not vis_queue.empty():
            samples.append(vis_queue.get_nowait())

        assert len(samples) == 2
        # First sample: t=1.0, x=100.0, rest zero
        assert samples[0] == (1.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_zero_motion_still_enqueues_visualization(self):
        """Visualization receives ALL samples, even those below motion threshold."""
        states = [FakeState(t=1.0), FakeState(t=2.0)]
        device = FakeDevice(states)
        client = FakeClient()
        vis_queue = queue_module.Queue(maxsize=100)

        _run_loop(device, client, vis_queue=vis_queue)

        samples = []
        while not vis_queue.empty():
            samples.append(vis_queue.get_nowait())

        # Both states enqueued for visualization even though no transform sent
        assert len(samples) == 2
        assert len(client.messages) == 0

    def test_full_queue_does_not_crash(self):
        """A full visualization queue should not block or crash the loop."""
        states = [FakeState(x=100.0, t=i * 0.01) for i in range(20)]
        device = FakeDevice(states)
        client = FakeClient()
        vis_queue = queue_module.Queue(maxsize=2)

        _run_loop(device, client, vis_queue=vis_queue)

        # Loop completed without hanging — the key assertion is that we get here
        assert len(client.messages) == 20

    def test_no_queue_works_fine(self):
        """Passing None for vis_queue should work without errors."""
        device = FakeDevice([FakeState(x=100.0)])
        client = FakeClient()

        _run_loop(device, client, vis_queue=None)

        assert len(client.messages) == 1


# ── Visualization decimation ─────────────────────────────────────────


class TestVisualizationDecimation:

    def test_stride_2_enqueues_half(self):
        """With stride=2, only every second sample is enqueued."""
        states = [FakeState(x=float(i), t=float(i)) for i in range(10)]
        device = FakeDevice(states)
        client = FakeClient()
        vis_queue = queue_module.Queue(maxsize=100)

        with pytest.raises(StopStreaming):
            _stream_loop(device, client, vis_queue, dt=DT, vis_stride=2)

        samples = []
        while not vis_queue.empty():
            samples.append(vis_queue.get_nowait())

        assert len(samples) == 5

    def test_stride_8_enqueues_one_eighth(self):
        """With stride=8, only every 8th sample is enqueued."""
        states = [FakeState(x=float(i), t=float(i)) for i in range(24)]
        device = FakeDevice(states)
        client = FakeClient()
        vis_queue = queue_module.Queue(maxsize=100)

        with pytest.raises(StopStreaming):
            _stream_loop(device, client, vis_queue, dt=DT, vis_stride=8)

        samples = []
        while not vis_queue.empty():
            samples.append(vis_queue.get_nowait())

        assert len(samples) == 3

    def test_stride_1_enqueues_all(self):
        """With stride=1 (default), all samples are enqueued."""
        states = [FakeState(x=float(i), t=float(i)) for i in range(7)]
        device = FakeDevice(states)
        client = FakeClient()
        vis_queue = queue_module.Queue(maxsize=100)

        with pytest.raises(StopStreaming):
            _stream_loop(device, client, vis_queue, dt=DT, vis_stride=1)

        samples = []
        while not vis_queue.empty():
            samples.append(vis_queue.get_nowait())

        assert len(samples) == 7

    def test_stride_does_not_affect_transform_sending(self):
        """Decimation only affects vis queue; all transforms are still sent."""
        states = [FakeState(x=100.0, t=float(i)) for i in range(16)]
        device = FakeDevice(states)
        client = FakeClient()
        vis_queue = queue_module.Queue(maxsize=100)

        with pytest.raises(StopStreaming):
            _stream_loop(device, client, vis_queue, dt=DT, vis_stride=8)

        # All 16 transforms sent
        assert len(client.messages) == 16

        # Only 2 vis samples (16 // 8)
        samples = []
        while not vis_queue.empty():
            samples.append(vis_queue.get_nowait())
        assert len(samples) == 2

    def test_no_queue_with_stride_does_nothing(self):
        """vis_queue=None with stride>1 works without error."""
        states = [FakeState(x=100.0, t=float(i)) for i in range(8)]
        device = FakeDevice(states)
        client = FakeClient()

        with pytest.raises(StopStreaming):
            _stream_loop(device, client, vis_queue=None, dt=DT, vis_stride=8)

        assert len(client.messages) == 8


# ── Round-trip identity ──────────────────────────────────────────────


class TestRoundTrip:

    def test_pure_translation_round_trip(self):
        """Pure translations forward then reverse cancel exactly."""
        forward = [
            FakeState(x=100.0, y=50.0, z=25.0, t=i * 0.01)
            for i in range(50)
        ]
        reverse = [
            FakeState(x=-100.0, y=-50.0, z=-25.0, t=(50 + i) * 0.01)
            for i in range(50)
        ]

        device = FakeDevice(forward + reverse)
        client = FakeClient()

        _run_loop(device, client)

        final = client.messages[-1].matrix
        assert_allclose(final, np.eye(4), atol=1e-6)

    def test_single_axis_rotation_round_trip(self):
        """Single-axis rotations forward then reverse cancel exactly.

        For a single Euler axis, R(-a) IS the exact inverse of R(a),
        so the round-trip produces identity. Multi-axis would leave a
        residual due to ZYX composition order (see test_math.py).
        """
        forward = [
            FakeState(yaw=10.0, t=i * 0.01)
            for i in range(50)
        ]
        reverse = [
            FakeState(yaw=-10.0, t=(50 + i) * 0.01)
            for i in range(50)
        ]

        device = FakeDevice(forward + reverse)
        client = FakeClient()

        _run_loop(device, client)

        final = client.messages[-1].matrix
        assert_allclose(final[:3, :3], np.eye(3), atol=1e-8)
        assert_allclose(final[:3, 3], [0, 0, 0], atol=1e-8)


# ── IGTL link health check ─────────────────────────────────────────


class FlippingClient(FakeClient):
    """FakeClient whose is_connected() flips to False after *flip_after* calls."""

    def __init__(self, flip_after: int = 3) -> None:
        super().__init__()
        self._flip_after = flip_after
        self._call_count = 0

    def is_connected(self) -> bool:
        self._call_count += 1
        return self._call_count <= self._flip_after


class TestLinkHealthCheck:

    def test_link_lost_warning(self, caplog):
        """A connected→disconnected transition emits a warning."""
        # With dt=0 the check interval is 500; use enough states to exceed it.
        states = [FakeState(x=100.0, t=float(i)) for i in range(600)]
        device = FakeDevice(states)
        client = FlippingClient(flip_after=1)

        with caplog.at_level(logging.WARNING, logger="greifer"):
            with pytest.raises(StopStreaming):
                _stream_loop(device, client, vis_queue=None, dt=DT)

        assert any("IGTL link lost" in r.message for r in caplog.records)


# ── Target switching ──────────────────────────────────────────────────


class TestTargetSwitching:

    def test_messages_use_active_target_device_name(self):
        """Sent messages carry the active target's name as device_name."""
        tm = TargetManager(["A", "B"])
        device = FakeDevice([FakeState(x=100.0)])
        client = FakeClient()

        with pytest.raises(StopStreaming):
            _stream_loop(device, client, vis_queue=None, dt=DT, target_manager=tm)

        assert client.messages[0].device_name == "A"

    def test_switch_target_command_changes_device_name(self):
        """After a switch_target command, subsequent messages use the new name."""
        tm = TargetManager(["A", "B"])
        states = [
            FakeState(x=100.0, t=0.0),  # sent as A
            FakeState(x=100.0, t=0.1),  # sent as B (after switch)
        ]
        device = FakeDevice(states)
        client = FakeClient()
        cmd_queue = queue_module.Queue()
        cmd_queue.put(("switch_target", "B"))

        with pytest.raises(StopStreaming):
            _stream_loop(
                device, client, vis_queue=None, dt=DT,
                cmd_queue=cmd_queue, target_manager=tm,
            )

        # First message: immediate send from switch_target (B's identity matrix)
        assert client.messages[0].device_name == "B"
        # Second message: first device read, still sent as A (cmd processed before read)
        # Wait — the command is processed before the first read, so:
        # cmd processed → switch to B → immediate send (B identity) → read state → send as B
        assert client.messages[1].device_name == "B"

    def test_switch_target_sends_immediate_matrix(self):
        """Switching target sends the new target's current matrix immediately."""
        tm = TargetManager(["A", "B"])
        # Pre-move B so its matrix is non-identity
        tm.switch_to("B")
        tm.update(5.0, 0, 0, 0, 0, 0)
        expected_matrix = tm.active_accumulator.matrix.copy()
        tm.switch_to("A")  # back to A as starting state

        device = FakeDevice([FakeState()])  # zero motion → no regular send
        client = FakeClient()
        cmd_queue = queue_module.Queue()
        cmd_queue.put(("switch_target", "B"))

        with pytest.raises(StopStreaming):
            _stream_loop(
                device, client, vis_queue=None, dt=DT,
                cmd_queue=cmd_queue, target_manager=tm,
            )

        # Only message is the immediate send from switch
        assert len(client.messages) == 1
        assert client.messages[0].device_name == "B"
        assert_allclose(client.messages[0].matrix, expected_matrix)

    def test_targets_accumulate_independently(self):
        """Moving target A does not affect target B's accumulated state."""
        tm = TargetManager(["A", "B"])
        states = [
            FakeState(x=100.0, t=0.0),  # moves A
            FakeState(x=100.0, t=0.1),  # moves A again
        ]
        device = FakeDevice(states)
        client = FakeClient()

        with pytest.raises(StopStreaming):
            _stream_loop(
                device, client, vis_queue=None, dt=DT, target_manager=tm,
            )

        # B should still be at identity
        assert_allclose(tm._targets["B"].matrix, np.eye(4))
        # A should have accumulated translation
        assert tm._targets["A"].matrix[0, 3] != 0.0


# ── Harden command ───────────────────────────────────────────────────


class TestHardenCommand:

    def test_harden_sends_string_then_identity(self):
        """Harden sends a StringMessage then an identity TransformMessage."""
        tm = TargetManager(["A"])
        tm.update(1.0, 0, 0, 0, 0, 0)  # accumulate something

        device = FakeDevice([FakeState()])  # zero motion → no regular send
        client = FakeClient()
        cmd_queue = queue_module.Queue()
        cmd_queue.put(("harden", None))

        with pytest.raises(StopStreaming):
            _stream_loop(
                device, client, vis_queue=None, dt=DT,
                cmd_queue=cmd_queue, target_manager=tm,
            )

        assert len(client.messages) == 2
        # First message: StringMessage with device_name "GreiferHarden"
        assert isinstance(client.messages[0], pyigtl.StringMessage)
        assert client.messages[0].device_name == "GreiferHarden"
        assert client.messages[0].string == "A"
        # Second message: identity TransformMessage with device_name "A"
        assert isinstance(client.messages[1], pyigtl.TransformMessage)
        assert client.messages[1].device_name == "A"
        assert_allclose(client.messages[1].matrix, np.eye(4))

    def test_harden_resets_accumulator(self):
        """After harden, the active accumulator is identity."""
        tm = TargetManager(["A"])
        tm.update(1.0, 0, 0, 0, 0, 0)

        device = FakeDevice([FakeState()])
        client = FakeClient()
        cmd_queue = queue_module.Queue()
        cmd_queue.put(("harden", None))

        with pytest.raises(StopStreaming):
            _stream_loop(
                device, client, vis_queue=None, dt=DT,
                cmd_queue=cmd_queue, target_manager=tm,
            )

        assert_allclose(tm.active_accumulator.matrix, np.eye(4))

    def test_harden_targets_active_only(self):
        """Hardening only resets the active target; inactive targets are unaffected."""
        tm = TargetManager(["A", "B"])
        tm.update(1.0, 0, 0, 0, 0, 0)  # move A
        tm.switch_to("B")
        tm.update(2.0, 0, 0, 0, 0, 0)  # move B
        matrix_a = tm._targets["A"].matrix.copy()
        tm.switch_to("A")  # switch back to A

        device = FakeDevice([FakeState()])
        client = FakeClient()
        cmd_queue = queue_module.Queue()
        cmd_queue.put(("harden", None))

        with pytest.raises(StopStreaming):
            _stream_loop(
                device, client, vis_queue=None, dt=DT,
                cmd_queue=cmd_queue, target_manager=tm,
            )

        # A was hardened → identity
        assert_allclose(tm._targets["A"].matrix, np.eye(4))
        # B is untouched
        assert tm._targets["B"].matrix[0, 3] != 0.0
