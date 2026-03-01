"""Tests for per-axis DOF locking (Axis enum + DofLockFilter)."""

import queue as queue_module

import numpy as np
from numpy.testing import assert_allclose

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from greifer.transform import Axis, DofLockFilter, Sensitivity
from greifer.app import _stream_loop, SENSITIVITY
from conftest import FakeDevice, FakeState, FakeClient, StopStreaming


DT = 0.0

ALL_AXES = list(Axis)
TRANSLATION_AXES = [Axis.X, Axis.Y, Axis.Z]
ROTATION_AXES = [Axis.ROLL, Axis.PITCH, Axis.YAW]


def _run_loop(device, client, vis_queue=None, dof_filter=None, cmd_queue=None):
    """Run _stream_loop until the FakeDevice is exhausted."""
    with pytest.raises(StopStreaming):
        _stream_loop(
            device, client, vis_queue, dt=DT,
            dof_filter=dof_filter, cmd_queue=cmd_queue,
        )


# ── Axis enum ───────────────────────────────────────────────────────


class TestAxisEnum:

    def test_has_six_members(self):
        assert len(Axis) == 6

    def test_values_are_lowercase_strings(self):
        for member in Axis:
            assert member.value == member.value.lower()
            assert isinstance(member.value, str)

    def test_members_are_hashable(self):
        s = {Axis.X, Axis.Y, Axis.Z, Axis.ROLL, Axis.PITCH, Axis.YAW}
        assert len(s) == 6

    def test_expected_members_exist(self):
        expected = {"X", "Y", "Z", "ROLL", "PITCH", "YAW"}
        assert {m.name for m in Axis} == expected


# ── DofLockFilter unit tests ────────────────────────────────────────


class TestDofLockFilterInit:

    def test_default_init_has_no_locks(self):
        f = DofLockFilter()
        assert f.locked == frozenset()

    def test_init_with_locked_set(self):
        f = DofLockFilter(locked={Axis.X, Axis.YAW})
        assert f.locked == frozenset({Axis.X, Axis.YAW})

    def test_init_copies_input_set(self):
        s = {Axis.X}
        f = DofLockFilter(locked=s)
        s.add(Axis.Y)
        assert Axis.Y not in f.locked


class TestLockUnlockToggle:

    def test_lock_single_axis(self):
        f = DofLockFilter()
        f.lock(Axis.X)
        assert Axis.X in f.locked

    def test_lock_multiple_axes(self):
        f = DofLockFilter()
        f.lock(Axis.X, Axis.Y, Axis.Z)
        assert f.locked == frozenset({Axis.X, Axis.Y, Axis.Z})

    def test_lock_is_idempotent(self):
        f = DofLockFilter()
        f.lock(Axis.X)
        f.lock(Axis.X)
        assert f.locked == frozenset({Axis.X})

    def test_unlock_single_axis(self):
        f = DofLockFilter(locked={Axis.X, Axis.Y})
        f.unlock(Axis.X)
        assert f.locked == frozenset({Axis.Y})

    def test_unlock_is_idempotent(self):
        f = DofLockFilter()
        f.unlock(Axis.X)  # no-op, doesn't raise
        assert f.locked == frozenset()

    def test_toggle_locks_unlocked_axis(self):
        f = DofLockFilter()
        f.toggle(Axis.ROLL)
        assert Axis.ROLL in f.locked

    def test_toggle_unlocks_locked_axis(self):
        f = DofLockFilter(locked={Axis.ROLL})
        f.toggle(Axis.ROLL)
        assert Axis.ROLL not in f.locked

    def test_toggle_twice_restores_state(self):
        f = DofLockFilter()
        f.toggle(Axis.Z)
        f.toggle(Axis.Z)
        assert Axis.Z not in f.locked

    def test_locked_returns_frozenset(self):
        f = DofLockFilter(locked={Axis.X})
        assert isinstance(f.locked, frozenset)


class TestApply:

    SAMPLE = (1.0, 2.0, 3.0, 0.1, 0.2, 0.3)

    def test_no_locks_passes_through(self):
        f = DofLockFilter()
        assert f.apply(*self.SAMPLE) == self.SAMPLE

    @pytest.mark.parametrize("axis,index", [
        (Axis.X, 0),
        (Axis.Y, 1),
        (Axis.Z, 2),
        (Axis.ROLL, 3),
        (Axis.PITCH, 4),
        (Axis.YAW, 5),
    ])
    def test_single_axis_locked_zeros_only_that_axis(self, axis, index):
        f = DofLockFilter(locked={axis})
        result = f.apply(*self.SAMPLE)
        for i, val in enumerate(result):
            if i == index:
                assert val == 0.0
            else:
                assert val == self.SAMPLE[i]

    def test_all_translation_locked(self):
        f = DofLockFilter(locked=set(TRANSLATION_AXES))
        result = f.apply(*self.SAMPLE)
        assert result == (0.0, 0.0, 0.0, 0.1, 0.2, 0.3)

    def test_all_rotation_locked(self):
        f = DofLockFilter(locked=set(ROTATION_AXES))
        result = f.apply(*self.SAMPLE)
        assert result == (1.0, 2.0, 3.0, 0.0, 0.0, 0.0)

    def test_all_axes_locked(self):
        f = DofLockFilter(locked=set(ALL_AXES))
        result = f.apply(*self.SAMPLE)
        assert result == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_lock_then_unlock_restores_passthrough(self):
        f = DofLockFilter()
        f.lock(Axis.X, Axis.ROLL)
        f.unlock(Axis.X, Axis.ROLL)
        assert f.apply(*self.SAMPLE) == self.SAMPLE


# ── Hypothesis property-based tests ─────────────────────────────────


class TestApplyPropertyBased:

    @given(
        dx=st.floats(allow_nan=False, allow_infinity=False),
        dy=st.floats(allow_nan=False, allow_infinity=False),
        dz=st.floats(allow_nan=False, allow_infinity=False),
        rx=st.floats(allow_nan=False, allow_infinity=False),
        ry=st.floats(allow_nan=False, allow_infinity=False),
        rz=st.floats(allow_nan=False, allow_infinity=False),
        locked=st.frozensets(st.sampled_from(ALL_AXES)),
    )
    @settings(max_examples=200)
    def test_locked_always_zero_unlocked_unchanged(
        self, dx, dy, dz, rx, ry, rz, locked,
    ):
        f = DofLockFilter(locked=set(locked))
        inputs = (dx, dy, dz, rx, ry, rz)
        result = f.apply(*inputs)

        axis_index = {
            Axis.X: 0, Axis.Y: 1, Axis.Z: 2,
            Axis.ROLL: 3, Axis.PITCH: 4, Axis.YAW: 5,
        }
        for axis, idx in axis_index.items():
            if axis in locked:
                assert result[idx] == 0.0
            else:
                assert result[idx] == inputs[idx]


# ── Integration tests with _stream_loop ─────────────────────────────


class TestDofLockIntegration:

    def test_locking_x_prevents_x_translation(self):
        """Locking X should produce zero X-translation in the output."""
        f = DofLockFilter(locked={Axis.X})
        states = [FakeState(x=100.0, t=0.0)]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client, dof_filter=f)

        matrix = client.messages[0].matrix
        assert_allclose(matrix[0, 3], 0.0, atol=1e-10)

    def test_locking_yaw_prevents_z_rotation(self):
        """Locking YAW prevents rotation around Z in the output."""
        f = DofLockFilter(locked={Axis.YAW})
        states = [FakeState(yaw=100.0, t=0.0)]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client, dof_filter=f)

        matrix = client.messages[0].matrix
        # Rotation submatrix should be identity (no rotation applied)
        assert_allclose(matrix[:3, :3], np.eye(3), atol=1e-10)

    def test_dof_filter_none_is_backward_compatible(self):
        """dof_filter=None should behave exactly like the old code."""
        states = [FakeState(x=100.0, t=0.0)]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client, dof_filter=None)

        matrix = client.messages[0].matrix
        expected_dx = 100.0 * SENSITIVITY.x
        assert_allclose(matrix[0, 3], expected_dx, atol=1e-10)

    def test_all_axes_locked_produces_identity(self):
        """With all 6 DOF locked, the matrix stays at identity."""
        f = DofLockFilter(locked=set(ALL_AXES))
        states = [
            FakeState(x=100.0, y=50.0, roll=30.0, yaw=20.0, t=i * 0.01)
            for i in range(10)
        ]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client, dof_filter=f)

        # All messages should still be sent (above threshold before filter)
        assert len(client.messages) == 10
        # Final matrix is identity since all increments are zeroed
        assert_allclose(client.messages[-1].matrix, np.eye(4), atol=1e-10)

    def test_locked_axis_does_not_affect_unlocked(self):
        """Locking Y should not affect X translation."""
        f = DofLockFilter(locked={Axis.Y})
        states = [FakeState(x=100.0, y=100.0, t=0.0)]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client, dof_filter=f)

        matrix = client.messages[0].matrix
        expected_dx = 100.0 * SENSITIVITY.x
        assert_allclose(matrix[0, 3], expected_dx, atol=1e-10)
        assert_allclose(matrix[1, 3], 0.0, atol=1e-10)

    def test_vis_queue_receives_raw_unfiltered_data(self):
        """Visualization queue should receive raw device state, not filtered."""
        f = DofLockFilter(locked={Axis.X})
        states = [FakeState(x=100.0, t=1.0)]
        device = FakeDevice(states)
        client = FakeClient()
        vis_queue = queue_module.Queue(maxsize=100)

        _run_loop(device, client, vis_queue=vis_queue, dof_filter=f)

        sample = vis_queue.get_nowait()
        # Raw x=100.0 appears in vis queue even though X is locked
        assert sample == (1.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ── Command queue integration tests ─────────────────────────────


class TestCommandQueueIntegration:

    def test_toggle_command_locks_axis(self):
        """A pre-loaded toggle command should lock the axis in output."""
        f = DofLockFilter()
        cmd_queue = queue_module.Queue()
        cmd_queue.put_nowait(("toggle", Axis.X))

        states = [FakeState(x=100.0, t=0.0)]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client, dof_filter=f, cmd_queue=cmd_queue)

        matrix = client.messages[0].matrix
        assert_allclose(matrix[0, 3], 0.0, atol=1e-10)

    def test_cmd_queue_none_is_backward_compatible(self):
        """cmd_queue=None should not change behaviour."""
        states = [FakeState(x=100.0, t=0.0)]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client, dof_filter=None, cmd_queue=None)

        matrix = client.messages[0].matrix
        expected_dx = 100.0 * SENSITIVITY.x
        assert_allclose(matrix[0, 3], expected_dx, atol=1e-10)

    def test_empty_cmd_queue_no_effect(self):
        """An empty command queue should not affect output."""
        f = DofLockFilter()
        cmd_queue = queue_module.Queue()

        states = [FakeState(x=100.0, t=0.0)]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client, dof_filter=f, cmd_queue=cmd_queue)

        matrix = client.messages[0].matrix
        expected_dx = 100.0 * SENSITIVITY.x
        assert_allclose(matrix[0, 3], expected_dx, atol=1e-10)

    def test_toggle_without_dof_filter_does_not_crash(self):
        """Toggle command with dof_filter=None should be silently ignored."""
        cmd_queue = queue_module.Queue()
        cmd_queue.put_nowait(("toggle", Axis.X))

        states = [FakeState(x=100.0, t=0.0)]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client, dof_filter=None, cmd_queue=cmd_queue)

        # Should still produce a transform (no crash)
        assert len(client.messages) == 1


# ── Sensitivity command integration tests ─────────────────────


class TestSensitivityCommandIntegration:

    def test_sensitivity_command_changes_scaling(self):
        """A sensitivity command should change the scale factors used."""
        new_sens = Sensitivity.uniform(0.01, 0.001)
        cmd_queue = queue_module.Queue()
        cmd_queue.put_nowait(("sensitivity", new_sens))

        states = [FakeState(x=100.0, t=0.0)]
        device = FakeDevice(states)
        client = FakeClient()
        f = DofLockFilter()

        _run_loop(device, client, dof_filter=f, cmd_queue=cmd_queue)

        matrix = client.messages[0].matrix
        expected_dx = 100.0 * new_sens.x
        assert_allclose(matrix[0, 3], expected_dx, atol=1e-10)

    def test_sensitivity_command_without_dof_filter(self):
        """Sensitivity command with dof_filter=None should not crash."""
        new_sens = Sensitivity.uniform(0.01, 0.001)
        cmd_queue = queue_module.Queue()
        cmd_queue.put_nowait(("sensitivity", new_sens))

        states = [FakeState(x=100.0, t=0.0)]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client, dof_filter=None, cmd_queue=cmd_queue)

        matrix = client.messages[0].matrix
        expected_dx = 100.0 * new_sens.x
        assert_allclose(matrix[0, 3], expected_dx, atol=1e-10)

    def test_unknown_command_ignored(self):
        """An unknown command type should be silently ignored."""
        cmd_queue = queue_module.Queue()
        cmd_queue.put_nowait(("unknown", 42))

        states = [FakeState(x=100.0, t=0.0)]
        device = FakeDevice(states)
        client = FakeClient()

        _run_loop(device, client, dof_filter=None, cmd_queue=cmd_queue)

        assert len(client.messages) == 1
