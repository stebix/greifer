"""Unit tests for greifer.transform.TransformAccumulator."""

import numpy as np
from numpy.testing import assert_allclose

from hypothesis import given, settings
from hypothesis import strategies as st

from greifer.transform import TransformAccumulator


ATOL = 1e-10

# Small angles typical of a real SpaceMouse frame (scaled by ROT_SCALE=0.001)
small_angles = st.floats(min_value=-0.01, max_value=0.01)
small_translations = st.floats(min_value=-0.05, max_value=0.05)


def _is_valid_rotation(R: np.ndarray, atol: float = 1e-6) -> bool:
    """Check orthonormality and det=+1."""
    return (
        np.allclose(R @ R.T, np.eye(3), atol=atol)
        and np.allclose(np.linalg.det(R), 1.0, atol=atol)
    )


# ── Initial state ────────────────────────────────────────────────────


class TestInitialState:

    def test_initial_matrix_is_identity(self):
        acc = TransformAccumulator()
        assert_allclose(acc.matrix, np.eye(4), atol=ATOL)

    def test_initial_rotation_is_identity(self):
        acc = TransformAccumulator()
        assert_allclose(acc.rotation, np.eye(3), atol=ATOL)

    def test_initial_translation_is_zero(self):
        acc = TransformAccumulator()
        assert_allclose(acc.translation, [0.0, 0.0, 0.0], atol=ATOL)

    def test_custom_reorthogonalize_interval(self):
        acc = TransformAccumulator(reorthogonalize_interval=50)
        assert acc._reorth_interval == 50


# ── Pure translation ─────────────────────────────────────────────────


class TestPureTranslation:

    def test_single_translation(self):
        acc = TransformAccumulator()
        acc.update(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
        assert_allclose(acc.translation, [1.0, 2.0, 3.0], atol=ATOL)

    def test_translations_accumulate(self):
        acc = TransformAccumulator()
        acc.update(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        acc.update(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert_allclose(acc.translation, [2.0, 0.0, 0.0], atol=ATOL)

    def test_translation_does_not_affect_rotation(self):
        acc = TransformAccumulator()
        acc.update(5.0, 10.0, 15.0, 0.0, 0.0, 0.0)
        assert_allclose(acc.rotation, np.eye(3), atol=ATOL)

    def test_three_axis_translation(self):
        acc = TransformAccumulator()
        acc.update(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        acc.update(0.0, 2.0, 0.0, 0.0, 0.0, 0.0)
        acc.update(0.0, 0.0, 3.0, 0.0, 0.0, 0.0)
        assert_allclose(acc.translation, [1.0, 2.0, 3.0], atol=ATOL)


# ── Pure rotation ────────────────────────────────────────────────────


class TestPureRotation:

    def test_single_rotation_preserves_orthonormality(self):
        acc = TransformAccumulator()
        acc.update(0.0, 0.0, 0.0, 0.1, 0.2, 0.3)
        assert _is_valid_rotation(acc.rotation)

    def test_rotation_does_not_affect_translation_when_starting_at_origin(self):
        acc = TransformAccumulator()
        acc.update(0.0, 0.0, 0.0, 0.5, 0.0, 0.0)
        assert_allclose(acc.translation, [0.0, 0.0, 0.0], atol=ATOL)

    def test_two_equal_yaw_rotations(self):
        """Two identical yaw increments should equal one rotation of double the angle."""
        from greifer.math import build_rotation_matrix

        angle = 0.1
        acc = TransformAccumulator()
        acc.update(0.0, 0.0, 0.0, 0.0, 0.0, angle)
        acc.update(0.0, 0.0, 0.0, 0.0, 0.0, angle)

        expected_R = build_rotation_matrix(0.0, 0.0, 2 * angle)
        assert_allclose(acc.rotation, expected_R, atol=ATOL)


# ── Zero motion ──────────────────────────────────────────────────────


class TestZeroMotion:

    def test_zero_update_preserves_identity(self):
        acc = TransformAccumulator()
        acc.update(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert_allclose(acc.matrix, np.eye(4), atol=ATOL)

    def test_many_zero_updates_preserve_identity(self):
        acc = TransformAccumulator()
        for _ in range(100):
            acc.update(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert_allclose(acc.matrix, np.eye(4), atol=ATOL)


# ── Accumulation order (left-multiply = world frame) ─────────────────


class TestAccumulationOrder:

    def test_translation_after_rotation_is_in_world_frame(self):
        """Left-multiply means increments are applied in world coordinates.

        Rotate 90° about Z, then translate along world-X.
        The translation should still go along world-X, not the rotated X.
        """
        acc = TransformAccumulator()
        acc.update(0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2)
        acc.update(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert_allclose(acc.translation[0], 1.0, atol=1e-8)

    def test_update_returns_current_matrix(self):
        acc = TransformAccumulator()
        result = acc.update(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
        assert_allclose(result, acc.matrix, atol=ATOL)


# ── Reorthogonalization ──────────────────────────────────────────────


class TestReorthogonalization:

    def test_triggers_at_interval(self):
        """After exactly N updates, the rotation submatrix should be orthonormal."""
        interval = 10
        acc = TransformAccumulator(reorthogonalize_interval=interval)
        for _ in range(interval):
            acc.update(0.0, 0.0, 0.0, 0.001, 0.002, 0.003)
        assert _is_valid_rotation(acc.rotation, atol=1e-12)

    def test_preserves_translation(self):
        """SVD correction must not corrupt the translation column."""
        interval = 5
        acc = TransformAccumulator(reorthogonalize_interval=interval)
        # Build up a known translation
        for _ in range(interval):
            acc.update(1.0, 0.0, 0.0, 0.001, 0.0, 0.0)
        # Force another reorthogonalization
        for _ in range(interval):
            acc.update(0.0, 0.0, 0.0, 0.001, 0.0, 0.0)
        # Translation should not have been zeroed out by the SVD step
        assert np.linalg.norm(acc.translation) > 0
        # The pure-rotation updates shouldn't change translation at origin,
        # but accumulated rotation means translation stays from before
        # Just verify it wasn't corrupted to NaN or zero
        assert np.all(np.isfinite(acc.translation))

    def test_explicit_reorthogonalize(self):
        """Calling reorthogonalize() directly should restore orthonormality."""
        acc = TransformAccumulator()
        # Manually corrupt the rotation submatrix slightly
        acc.matrix[:3, :3] *= 1.0001
        assert not _is_valid_rotation(acc.rotation, atol=1e-6)
        acc.reorthogonalize()
        assert _is_valid_rotation(acc.rotation, atol=1e-12)

    def test_reorthogonalize_corrects_artificial_corruption(self):
        """Simulate accumulated drift by corrupting the matrix, then verify
        that the periodic reorthogonalization during update() restores it.

        Note: analytic rotation matrices composed via numpy @ are stable
        to machine epsilon even over 500k iterations, so real drift is
        negligible. The reorthogonalization is a defensive safety net.
        """
        acc = TransformAccumulator(reorthogonalize_interval=5)
        # Manually corrupt the rotation submatrix to simulate drift
        acc.matrix[:3, :3] *= 1.001
        assert not _is_valid_rotation(acc.rotation, atol=1e-6)

        # Pump exactly enough updates to trigger reorthogonalization
        for _ in range(5):
            acc.update(0.0, 0.0, 0.0, 0.001, 0.0, 0.0)

        assert _is_valid_rotation(acc.rotation, atol=1e-12)

    def test_drift_corrected_with_reorthogonalization(self):
        """Same workload as above but with reorth enabled — det stays near 1.0."""
        acc = TransformAccumulator(reorthogonalize_interval=1000)
        rng = np.random.default_rng(42)
        for _ in range(50_000):
            rx, ry, rz = rng.normal(0, 0.001, size=3)
            acc.update(0.0, 0.0, 0.0, rx, ry, rz)

        assert _is_valid_rotation(acc.rotation, atol=1e-6)


# ── Property-based tests ─────────────────────────────────────────────


class TestAccumulatorProperties:

    @given(
        dx=small_translations, dy=small_translations, dz=small_translations,
        rx=small_angles, ry=small_angles, rz=small_angles,
    )
    def test_single_update_produces_valid_transform(
        self, dx, dy, dz, rx, ry, rz
    ):
        """Any single incremental update should produce a valid SE(3) matrix."""
        acc = TransformAccumulator()
        acc.update(dx, dy, dz, rx, ry, rz)
        assert _is_valid_rotation(acc.rotation, atol=1e-8)
        assert acc.matrix[3, 0] == 0.0
        assert acc.matrix[3, 1] == 0.0
        assert acc.matrix[3, 2] == 0.0
        assert acc.matrix[3, 3] == 1.0

    @given(
        dx=small_translations, dy=small_translations, dz=small_translations,
        rx=small_angles, ry=small_angles, rz=small_angles,
    )
    @settings(max_examples=30)
    def test_matrix_bottom_row_always_0001(
        self, dx, dy, dz, rx, ry, rz
    ):
        """The bottom row of the 4x4 matrix must always be [0, 0, 0, 1]."""
        acc = TransformAccumulator()
        for _ in range(10):
            acc.update(dx, dy, dz, rx, ry, rz)
        assert_allclose(acc.matrix[3, :], [0, 0, 0, 1], atol=ATOL)
