"""Unit tests for greifer.math — rotation matrix construction."""

import numpy as np
from numpy.testing import assert_allclose

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from greifer.math import build_rotation_matrix


ATOL = 1e-12

# Reusable hypothesis strategy: angles in [-2π, 2π]
angles = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi)


# ── Deterministic tests ──────────────────────────────────────────────


class TestKnownRotations:
    """Verify against analytically known rotation results."""

    def test_identity_at_zero_angles(self):
        R = build_rotation_matrix(0.0, 0.0, 0.0)
        assert_allclose(R, np.eye(3), atol=ATOL)

    def test_90deg_x_rotation(self):
        """Rx(π/2): z-axis maps to -y-axis."""
        R = build_rotation_matrix(np.pi / 2, 0.0, 0.0)
        result = R @ np.array([0.0, 0.0, 1.0])
        assert_allclose(result, [0.0, -1.0, 0.0], atol=ATOL)

    def test_90deg_y_rotation(self):
        """Ry(π/2): x-axis maps to z-axis (not -z, since Ry uses +sy in top-right)."""
        R = build_rotation_matrix(0.0, np.pi / 2, 0.0)
        result = R @ np.array([1.0, 0.0, 0.0])
        assert_allclose(result, [0.0, 0.0, -1.0], atol=ATOL)

    def test_90deg_z_rotation(self):
        """Rz(π/2): x-axis maps to y-axis."""
        R = build_rotation_matrix(0.0, 0.0, np.pi / 2)
        result = R @ np.array([1.0, 0.0, 0.0])
        assert_allclose(result, [0.0, 1.0, 0.0], atol=ATOL)

    def test_180deg_x_rotation(self):
        """Rx(π): y-axis maps to -y-axis, z-axis maps to -z-axis."""
        R = build_rotation_matrix(np.pi, 0.0, 0.0)
        assert_allclose(R @ [0, 1, 0], [0, -1, 0], atol=ATOL)
        assert_allclose(R @ [0, 0, 1], [0, 0, -1], atol=ATOL)
        assert_allclose(R @ [1, 0, 0], [1, 0, 0], atol=ATOL)

    def test_180deg_y_rotation(self):
        """Ry(π): x-axis maps to -x-axis, z-axis maps to -z-axis."""
        R = build_rotation_matrix(0.0, np.pi, 0.0)
        assert_allclose(R @ [1, 0, 0], [-1, 0, 0], atol=ATOL)
        assert_allclose(R @ [0, 0, 1], [0, 0, -1], atol=ATOL)
        assert_allclose(R @ [0, 1, 0], [0, 1, 0], atol=ATOL)

    def test_180deg_z_rotation(self):
        """Rz(π): x-axis maps to -x-axis, y-axis maps to -y-axis."""
        R = build_rotation_matrix(0.0, 0.0, np.pi)
        assert_allclose(R @ [1, 0, 0], [-1, 0, 0], atol=ATOL)
        assert_allclose(R @ [0, 1, 0], [0, -1, 0], atol=ATOL)
        assert_allclose(R @ [0, 0, 1], [0, 0, 1], atol=ATOL)


class TestCompositionOrder:
    """Verify the ZYX composition order of the rotation matrix."""

    def test_composition_order_is_zyx(self):
        """Rz @ Ry @ Rx: applying Rx first, then Ry, then Rz (intrinsic XYZ)."""
        rx, ry, rz = 0.3, 0.5, 0.7

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        expected = Rz @ Ry @ Rx
        actual = build_rotation_matrix(rx, ry, rz)
        assert_allclose(actual, expected, atol=ATOL)

    def test_single_axis_isolation(self):
        """Rotating about one axis should not affect the other components."""
        R = build_rotation_matrix(0.5, 0.0, 0.0)
        # x-axis should be unchanged by rotation about x
        assert_allclose(R @ [1, 0, 0], [1, 0, 0], atol=ATOL)


class TestSmallAngleApproximation:
    """For tiny angles, R ≈ I + skew(rx, ry, rz)."""

    @pytest.mark.parametrize("scale", [1e-6, 1e-8, 1e-10])
    def test_small_angle_matches_linearization(self, scale):
        rx, ry, rz = 1.0 * scale, 2.0 * scale, 3.0 * scale
        R = build_rotation_matrix(rx, ry, rz)

        # First-order approximation: R ≈ I + [[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]]
        R_approx = np.eye(3) + np.array([
            [0, -rz, ry],
            [rz, 0, -rx],
            [-ry, rx, 0],
        ])
        # Tolerance scales quadratically with the angle magnitude
        assert_allclose(R, R_approx, atol=scale**2 * 10)


class TestReturnShape:

    def test_output_shape_is_3x3(self):
        R = build_rotation_matrix(0.1, 0.2, 0.3)
        assert R.shape == (3, 3)

    def test_output_dtype_is_float(self):
        R = build_rotation_matrix(0.1, 0.2, 0.3)
        assert np.issubdtype(R.dtype, np.floating)


# ── Property-based (Hypothesis) tests ────────────────────────────────


class TestRotationMatrixProperties:
    """Algebraic invariants that must hold for all valid rotation matrices."""

    @given(rx=angles, ry=angles, rz=angles)
    def test_orthonormality(self, rx, ry, rz):
        """R @ R.T == I for any rotation matrix."""
        R = build_rotation_matrix(rx, ry, rz)
        assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    @given(rx=angles, ry=angles, rz=angles)
    def test_determinant_is_plus_one(self, rx, ry, rz):
        """det(R) == +1 (proper rotation, not reflection)."""
        R = build_rotation_matrix(rx, ry, rz)
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    @given(rx=angles, ry=angles, rz=angles)
    def test_inverse_equals_transpose(self, rx, ry, rz):
        """R^{-1} == R.T for orthonormal matrices."""
        R = build_rotation_matrix(rx, ry, rz)
        assert_allclose(np.linalg.inv(R), R.T, atol=1e-10)

    @given(rx=angles, ry=angles, rz=angles)
    def test_preserves_vector_norm(self, rx, ry, rz):
        """Rotation must not change the length of any vector."""
        R = build_rotation_matrix(rx, ry, rz)
        v = np.array([1.0, 2.0, 3.0])
        assert_allclose(np.linalg.norm(R @ v), np.linalg.norm(v), atol=1e-10)

    @given(rx=angles, ry=angles, rz=angles)
    def test_inverse_via_reversed_euler_order(self, rx, ry, rz):
        """Inverse of Rz·Ry·Rx is Rx(-rx)·Ry(-ry)·Rz(-rz) (reversed order).

        Since build_rotation_matrix composes as Rz @ Ry @ Rx, its inverse
        must apply the individual inverses in reversed order. We verify
        this equals R.T (the known orthogonal inverse).
        """
        R = build_rotation_matrix(rx, ry, rz)
        # Build inverse by reversing the composition order
        Rx_inv = build_rotation_matrix(-rx, 0.0, 0.0)
        Ry_inv = build_rotation_matrix(0.0, -ry, 0.0)
        Rz_inv = build_rotation_matrix(0.0, 0.0, -rz)
        R_inv = Rx_inv @ Ry_inv @ Rz_inv
        assert_allclose(R_inv @ R, np.eye(3), atol=1e-10)

    @given(rx=angles, ry=angles, rz=angles)
    @settings(max_examples=50)
    def test_columns_are_unit_vectors(self, rx, ry, rz):
        """Each column of R must have unit norm."""
        R = build_rotation_matrix(rx, ry, rz)
        for col in range(3):
            assert_allclose(np.linalg.norm(R[:, col]), 1.0, atol=1e-10)

    @given(rx=angles, ry=angles, rz=angles)
    @settings(max_examples=50)
    def test_columns_are_mutually_orthogonal(self, rx, ry, rz):
        """Columns of R must be pairwise orthogonal."""
        R = build_rotation_matrix(rx, ry, rz)
        assert_allclose(R[:, 0] @ R[:, 1], 0.0, atol=1e-10)
        assert_allclose(R[:, 0] @ R[:, 2], 0.0, atol=1e-10)
        assert_allclose(R[:, 1] @ R[:, 2], 0.0, atol=1e-10)
