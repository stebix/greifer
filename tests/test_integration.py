"""Integration tests for the stream loop wiring.

These tests verify that _stream_loop correctly connects device reading,
transform computation, network sending, and visualization queuing — using
fakes at the hardware/network boundaries.
"""

import queue as queue_module

import numpy as np
from numpy.testing import assert_allclose

import pytest

from greifer.app import _stream_loop
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
        from greifer.app import TRANS_SCALE
        device = FakeDevice([FakeState(x=100.0)])
        client = FakeClient()

        _run_loop(device, client)

        matrix = client.messages[0].matrix
        expected_dx = 100.0 * TRANS_SCALE
        assert_allclose(matrix[0, 3], expected_dx, atol=1e-10)
        assert_allclose(matrix[1, 3], 0.0, atol=1e-10)
        assert_allclose(matrix[2, 3], 0.0, atol=1e-10)


# ── Transform accumulation across frames ──────────────────────────────


class TestAccumulationAcrossFrames:

    def test_translations_accumulate(self):
        """Two consecutive x-translations should sum."""
        from greifer.app import TRANS_SCALE
        states = [
            FakeState(x=100.0, t=0.0),
            FakeState(x=100.0, t=0.1),
        ]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client)

        # Second message should have accumulated translation
        final_matrix = client.messages[-1].matrix
        expected_dx = 2 * 100.0 * TRANS_SCALE
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
