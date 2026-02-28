"""Unit tests for greifer.transform.compute_increments."""

import pytest

from greifer.transform import (
    Axis,
    MOTION_THRESHOLD,
    Sensitivity,
    compute_increments,
)


class TestScaling:

    def test_translation_scaling(self):
        result = compute_increments(
            100.0, 200.0, 300.0, 0.0, 0.0, 0.0,
            sensitivity=Sensitivity.uniform(trans=0.005, rot=0.001),
        )
        assert result is not None
        dx, dy, dz, rx, ry, rz = result
        assert dx == pytest.approx(0.5)
        assert dy == pytest.approx(1.0)
        assert dz == pytest.approx(1.5)
        assert rx == 0.0
        assert ry == 0.0
        assert rz == 0.0

    def test_rotation_scaling(self):
        result = compute_increments(
            0.0, 0.0, 0.0, 100.0, 200.0, 300.0,
            sensitivity=Sensitivity.uniform(trans=0.005, rot=0.001),
        )
        assert result is not None
        dx, dy, dz, rx, ry, rz = result
        assert dx == 0.0
        assert dy == 0.0
        assert dz == 0.0
        assert rx == pytest.approx(0.1)
        assert ry == pytest.approx(0.2)
        assert rz == pytest.approx(0.3)

    def test_mixed_scaling(self):
        result = compute_increments(
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            sensitivity=Sensitivity.uniform(trans=0.01, rot=0.002),
        )
        assert result is not None
        dx, dy, dz, rx, ry, rz = result
        assert dx == pytest.approx(0.1)
        assert dy == pytest.approx(0.2)
        assert dz == pytest.approx(0.3)
        assert rx == pytest.approx(0.08)
        assert ry == pytest.approx(0.10)
        assert rz == pytest.approx(0.12)

    def test_negative_values(self):
        result = compute_increments(
            -100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            sensitivity=Sensitivity.uniform(trans=0.005, rot=0.001),
        )
        assert result is not None
        assert result[0] == pytest.approx(-0.5)

    def test_per_axis_scaling(self):
        """Each axis uses its own distinct scale factor."""
        sens = Sensitivity(x=0.1, y=0.2, z=0.3, roll=0.4, pitch=0.5, yaw=0.6)
        result = compute_increments(
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            sensitivity=sens,
        )
        assert result is not None
        dx, dy, dz, rx, ry, rz = result
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(2.0)
        assert dz == pytest.approx(3.0)
        assert rx == pytest.approx(4.0)
        assert ry == pytest.approx(5.0)
        assert rz == pytest.approx(6.0)


class TestThreshold:

    def test_zero_input_returns_none(self):
        result = compute_increments(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            sensitivity=Sensitivity.uniform(trans=0.005, rot=0.001),
        )
        assert result is None

    def test_below_threshold_returns_none(self):
        # Total motion after scaling: 1e-7 * 0.005 = 5e-10, well below 1e-6
        result = compute_increments(
            1e-7, 0.0, 0.0, 0.0, 0.0, 0.0,
            sensitivity=Sensitivity.uniform(trans=0.005, rot=0.001),
        )
        assert result is None

    def test_above_threshold_returns_tuple(self):
        result = compute_increments(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            sensitivity=Sensitivity.uniform(trans=0.005, rot=0.001),
        )
        assert result is not None
        assert len(result) == 6

    def test_exactly_at_threshold_passes_through(self):
        """Motion exactly equal to threshold passes (the check is strict <)."""
        result = compute_increments(
            MOTION_THRESHOLD, 0.0, 0.0, 0.0, 0.0, 0.0,
            sensitivity=Sensitivity.uniform(trans=1.0, rot=1.0),
        )
        assert result is not None

    def test_just_above_threshold_returns_value(self):
        result = compute_increments(
            MOTION_THRESHOLD * 1.1, 0.0, 0.0, 0.0, 0.0, 0.0,
            sensitivity=Sensitivity.uniform(trans=1.0, rot=1.0),
        )
        assert result is not None


class TestReturnType:

    def test_returns_six_tuple(self):
        result = compute_increments(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            sensitivity=Sensitivity.uniform(trans=0.1, rot=0.1),
        )
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 6

    def test_all_elements_are_float(self):
        result = compute_increments(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            sensitivity=Sensitivity.uniform(trans=0.1, rot=0.1),
        )
        assert result is not None
        for val in result:
            assert isinstance(val, float)


class TestSensitivity:

    def test_uniform_matches_manual(self):
        """uniform() produces the same result as specifying equal values."""
        uniform = Sensitivity.uniform(trans=0.005, rot=0.001)
        manual = Sensitivity(x=0.005, y=0.005, z=0.005,
                             roll=0.001, pitch=0.001, yaw=0.001)
        assert uniform == manual

    def test_for_axis_returns_correct_value(self):
        """for_axis() returns the correct scale for each Axis member."""
        sens = Sensitivity(x=0.1, y=0.2, z=0.3, roll=0.4, pitch=0.5, yaw=0.6)
        assert sens.for_axis(Axis.X) == 0.1
        assert sens.for_axis(Axis.Y) == 0.2
        assert sens.for_axis(Axis.Z) == 0.3
        assert sens.for_axis(Axis.ROLL) == 0.4
        assert sens.for_axis(Axis.PITCH) == 0.5
        assert sens.for_axis(Axis.YAW) == 0.6

    def test_frozen(self):
        """Sensitivity instances are immutable."""
        sens = Sensitivity.uniform(trans=0.005, rot=0.001)
        with pytest.raises(AttributeError):
            sens.x = 0.01  # type: ignore[misc]
